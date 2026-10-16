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

"""Canonical protocols of the judged suites aa_lcr, beam_128k, healthbench and imo_answer_bench.

Pins the canonical protocols: verbatim upstream prompts (checked by sha256 against the pinned
upstream sources), the pinned datasets, the per-item judges, the official aggregation, the
official repeat counts, hard errors instead of fallbacks, and honest leaderboard flags.
"""

import asyncio
import csv
import hashlib
import io
import json
import os
import zipfile
from collections import Counter
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import aa_lcr, beam_128k, healthbench, imo_answer_bench

def _patch_detailed(module, judge):
    """Patch a suite's judge_generate_cascade_detailed with a (text, model) judge stub; the
    detailed call returns (text, model, info) with nothing cut off at the grader's cap."""
    async def detailed(prompt, config=None):
        text, model = await judge(prompt, config=config)
        return text, model, {"finish_reason": None, "max_tokens_without_text": 0}
    return mock.patch.object(module, "judge_generate_cascade_detailed", detailed)


ROOT = Path(__file__).resolve().parent.parent


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cascade(replies):
    """A judge_generate_cascade stand-in returning `replies` in order, recording each call."""
    calls = []
    it = iter(replies)

    async def _fake(contents, *, config=None):
        calls.append((contents, config))
        text = next(it)
        return (None, "JUDGE_OUTAGE") if text is None else (text, "gemini-test")
    return _fake, calls


def _system_instruction(config):
    return getattr(config, "system_instruction", None) if config is not None else None


# =============================================================================
# aa_lcr (#00, #01)
# =============================================================================
def test_aa_lcr_prompts_are_the_v1_1_card_verbatim():
    # sha256 of the dataset card text at revision 9a77ef5 (README.md, "Prompt Template" and
    # "Scoring Approach").
    assert _sha(aa_lcr.PROMPT_TEMPLATE) == "9a7153b9a88a057cca701001e88ee410a81453396a990a91e93a5368319fda98"
    assert _sha(aa_lcr.JUDGE_SYSTEM_PROMPT) == "e5ed5b4646e01151f43e9ea97ba8490208151d4317c594963719f37dcecf40ca"
    assert _sha(aa_lcr.JUDGE_USER_TEMPLATE) == "5b0774cae84cb77d86545705ec7efb2e671355d77a72dcd04f34b3ad8b2a2f3d"
    assert aa_lcr._REVISION == "9a77ef56b717057ade24ceab4d273712a0b4f19e"


def test_aa_lcr_prompt_numbers_documents_in_filename_order():
    prompt = aa_lcr.build_prompt(["first doc", "second doc"], "What?")
    assert prompt == ("BEGIN INPUT DOCUMENTS\n\nBEGIN DOCUMENT 1:\nfirst doc\nEND DOCUMENT 1\n\n"
                      "BEGIN DOCUMENT 2:\nsecond doc\nEND DOCUMENT 2\n\nEND INPUT DOCUMENTS\n\n"
                      "Answer the following question using the input documents provided above.\n\n"
                      "START QUESTION\n\nWhat?\n\nEND QUESTION\n")


def _aa_lcr_fixture(tmp, rows, entries):
    """CSV + zip laid out like the real archive. `entries` maps an ASCII placeholder path to
    (real name, text); names are patched into the bytes so they carry no UTF-8 flag (cp437)."""
    csv_path = os.path.join(tmp, "aa.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["document_category", "document_set_id", "question_id",
                                          "question", "answer", "data_source_filenames"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    zip_path = os.path.join(tmp, "aa.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for placeholder, (_real, text) in entries.items():
            zf.writestr(placeholder, text)
    data = Path(zip_path).read_bytes()
    for placeholder, (real, _text) in entries.items():
        raw = real.encode("utf-8")
        assert len(raw) == len(placeholder.encode("ascii"))
        data = data.replace(placeholder.encode("ascii"), raw)
    Path(zip_path).write_bytes(data)
    return csv_path, zip_path


def _load_with(csv_path, zip_path, **kw):
    revisions = []

    def _fake(repo_id, filename, **kwargs):
        revisions.append(kwargs.get("revision"))
        return zip_path if filename.endswith(".zip") else csv_path
    with mock.patch("huggingface_hub.hf_hub_download", side_effect=_fake):
        samples = aa_lcr._load_aa_lcr_samples(**kw)
    return samples, revisions


def test_aa_lcr_reads_cp437_flagged_names_and_nfd_names(tmp_path):
    # The real archive: 'you<em dash>and' stored as raw UTF-8 without the UTF-8 flag (zipfile
    # decodes it as cp437 mojibake), and a decomposed (NFD) 'Bas<cedilla>ev' the CSV lists composed.
    em = "you\u2014and.txt"                       # 3-byte em dash
    nfd = "Bas\u0327ev.txt"                       # s + combining cedilla (2 bytes)
    entries = {"lcr/Legal/s1/youXXXand.txt": (f"lcr/Legal/s1/{em}", "doc A"),
               "lcr/Legal/s1/BasXXev.txt": (f"lcr/Legal/s1/{nfd}", "doc B"),
               "lcr/Legal/s1/plain.txt": ("lcr/Legal/s1/plain.txt", "doc C")}
    rows = [{"document_category": "Legal", "document_set_id": "s1", "question_id": "98",
             "question": "Q? ", "answer": "A",
             "data_source_filenames": f"{em};Ba\u015fev.txt;plain.txt"}]
    csv_path, zip_path = _aa_lcr_fixture(str(tmp_path), rows, entries)
    with zipfile.ZipFile(zip_path) as zf:
        assert all(not i.flag_bits & 0x800 for i in zf.infolist() if "you" in i.filename)
    samples, revisions = _load_with(csv_path, zip_path)
    assert revisions == [aa_lcr._REVISION] * 2           # both files from the pinned revision
    prompt = samples[0][0][0]["content"]
    assert prompt.count("BEGIN DOCUMENT ") == 3            # nothing silently dropped
    assert prompt.index("doc A") < prompt.index("doc B") < prompt.index("doc C")
    assert samples[0][2]["question"] == "Q? "              # the raw question, unstripped


def test_aa_lcr_missing_listed_document_is_a_hard_error(tmp_path):
    entries = {"lcr/Legal/s1/plain.txt": ("lcr/Legal/s1/plain.txt", "doc C")}
    rows = [{"document_category": "Legal", "document_set_id": "s1", "question_id": "1",
             "question": "Q", "answer": "A", "data_source_filenames": "plain.txt;gone.txt"}]
    csv_path, zip_path = _aa_lcr_fixture(str(tmp_path), rows, entries)
    with pytest.raises(RuntimeError, match="gone.txt"):
        _load_with(csv_path, zip_path)


def test_aa_lcr_documents_are_keyed_by_set_not_basename(tmp_path):
    entries = {"lcr/Co/a/r.txt": ("lcr/Co/a/r.txt", "set a"),
               "lcr/Co/b/r.txt": ("lcr/Co/b/r.txt", "set b")}
    rows = [{"document_category": "Co", "document_set_id": "a", "question_id": "1",
             "question": "Q", "answer": "A", "data_source_filenames": "r.txt"}]
    csv_path, zip_path = _aa_lcr_fixture(str(tmp_path), rows, entries)
    samples, _ = _load_with(csv_path, zip_path)
    assert "set a" in samples[0][0][0]["content"] and "set b" not in samples[0][0][0]["content"]


@pytest.mark.parametrize("reply,verdict", [
    ('{"verdict": "CORRECT"}', "CORRECT"),
    ('```json\n{"verdict": "incorrect"}\n```', "INCORRECT"),
    ('{"result": "CORRECT"}', "CORRECT"),
    ("CORRECT", "CORRECT"),                  # a bare verdict: read by the relaxed parse
    ("CORRECT or INCORRECT", None),          # a format echo: asked again
    ('{"verdict": "maybe"}', None),
])
def test_aa_lcr_parse_verdict(reply, verdict):
    assert aa_lcr.parse_verdict(reply) == verdict


def _aa_trace(response="Equinix, $901 million"):
    return {"response_text": response, "gold_answer": "Equinix, $901 million",
            "messages": [{"role": "user", "content": "BEGIN INPUT DOCUMENTS ... huge"}],
            "extra_payload": {"question": "Which company?", "category": "Company_Documents"}}


def test_aa_lcr_judge_uses_card_prompts_and_raw_question(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    fake, calls = _cascade(['{"verdict": "CORRECT"}'])
    trace = _aa_trace()
    with mock.patch.object(aa_lcr, "judge_generate_cascade", fake):
        asyncio.run(aa_lcr._async_judge_aa_lcr([trace]))
    prompt, config = calls[0]
    assert prompt == aa_lcr.JUDGE_USER_TEMPLATE.format(
        question="Which company?", official_answer="Equinix, $901 million",
        candidate_answer="Equinix, $901 million")
    assert "BEGIN INPUT DOCUMENTS" not in prompt           # not the document prompt
    if config is not None:
        assert _system_instruction(config) == aa_lcr.JUDGE_SYSTEM_PROMPT
    assert trace["is_correct"] is True and trace["judge_grade"] == "CORRECT"
    assert trace["judge_model"] == "gemini-test"


def test_aa_lcr_unparsable_judge_is_judge_failed_and_outage_is_excluded(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    fake, calls = _cascade(["no json", "still none", "nope", None])
    bad, out = _aa_trace(), _aa_trace()
    with mock.patch.object(aa_lcr, "judge_generate_cascade", fake):
        asyncio.run(aa_lcr._async_judge_aa_lcr([bad]))
        asyncio.run(aa_lcr._async_judge_aa_lcr([out]))
    assert len(calls) == 4                                  # 3 tries, then 1 outage
    assert bad["judge_grade"] == "JUDGE_FAILED" and bad["is_correct"] is False
    assert out["judge_grade"] == "JUDGE_OUTAGE"


def test_aa_lcr_no_key_hard_errors_instead_of_substring_fallback(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    with pytest.raises(RuntimeError, match="INFRA REQUIRED"):
        asyncio.run(aa_lcr._async_judge_aa_lcr([_aa_trace()]))
    assert not hasattr(aa_lcr, "_eval_aa_lcr")


def test_aa_lcr_runs_three_repeats_and_is_flagged():
    seen = {}

    def _fake_run(**kw):
        seen.update(kw)
        return {"accuracy": 40.0, "sample_traces": [{"judge_model": "gemini-x"}], "temperature": 0.0}
    with mock.patch.object(aa_lcr, "gemini_required_skip", lambda *a: None), \
         mock.patch.object(aa_lcr, "_load_aa_lcr_samples", lambda **k: []), \
         mock.patch.object(aa_lcr, "run_eval_suite", _fake_run):
        r = aa_lcr.run_aa_lcr("m", "http://x/v1", attempt_count=1)
        r5 = aa_lcr.run_aa_lcr("m", "http://x/v1", attempt_count=5, limit=10)
    assert seen["attempt_count"] == 5 and seen["supports_attempts"] is True
    assert r["leaderboard_comparable"] is False
    assert "GPT-5.6 Luna" in r["leaderboard_comparable_reason"]
    assert "no judge call answered" in r["leaderboard_comparable_reason"]
    assert r["judge_models"] == {} and r["judge_cascade"] and r["judge"] == "gbench-gemini-cascade"
    assert "3 attempts" in r["metric"] and "--eval-limit 10" in r5["leaderboard_comparable_reason"]
    assert aa_lcr._REVISION in r["dataset"]


# =============================================================================
# beam_128k (#03, #34, #77)
# =============================================================================
def test_beam_prompts_are_upstream_verbatim():
    # sha256 of the strings in github.com/mohammadtavakoli78/BEAM @ b2da22e (src/prompts.py,
    # src/evaluation/compute_metrics.py, src/answer_probing_questions/long_term_memory_methods.py).
    assert _sha(beam_128k.JUDGE_PROMPT) == "593373c642a288a7b590577d8a8fc92c3f9a2b70e2f64ad6e59a040a6c56b7f5"
    assert _sha(beam_128k.EQUIV_SYSTEM) == "0863c78c1b27d36c97bb7bfd2d4d6b88ef69b093e92bb4719e665527722413e2"
    assert _sha(beam_128k.EQUIV_USER.format(first="<A>", second="<B>")) == \
        "e77cf22217032629e6fd094e2776d8ce37dcaa1b0b49c2b636cbcd5c59a9c81e"
    assert _sha(beam_128k.FINAL_TURN.format(query="<Q>")) == \
        "057808e3f0dd92e68dd01aaf90b3f0643779a439f9a2812bb128c06c0ee5a2e0"
    assert beam_128k._REVISION == "3205395e897e7318c7b094ef4e6047b9b82dbb03"


def test_beam_flattens_sessions_into_chat_turns():
    import numpy as np
    s1 = np.array([{"role": "user", "content": "a\nb", "id": 0, "time_anchor": "x"},
                   {"role": "assistant", "content": "c", "id": 1}], dtype=object)
    s2 = np.array([{"role": "user", "content": "d"}, {"role": "assistant", "content": "e"}],
                  dtype=object)
    msgs = beam_128k._flatten_chat(np.array([s1, s2], dtype=object))
    assert msgs == [{"role": "user", "content": "a\nb"}, {"role": "assistant", "content": "c"},
                    {"role": "user", "content": "d"}, {"role": "assistant", "content": "e"}]


def _beam_parquet(tmp, n_conv=2):
    import pandas as pd
    rows = []
    for c in range(n_conv):
        probing = {a: [{"question": f"{a} q{i} c{c}", "rubric": [f"{a} item"], "answer": "x"}
                       for i in range(2)] for a in beam_128k.ABILITIES}
        rows.append({"conversation_id": str(c),
                     "chat": [[{"role": "user", "content": f"u{c}-{k}"},
                               {"role": "assistant", "content": f"a{c}-{k}"}] for k in range(5)],
                     "probing_questions": repr(probing)})
    path = os.path.join(tmp, "beam.parquet")
    pd.DataFrame(rows).to_parquet(path)
    return path


def test_beam_limit_is_stratified_by_ability(tmp_path):
    path = _beam_parquet(str(tmp_path))
    with mock.patch("huggingface_hub.hf_hub_download", side_effect=lambda *a, **k: path):
        samples = beam_128k._load_beam_128k_samples(limit=10)
    assert Counter(s[2]["category"] for s in samples) == Counter({a: 1 for a in beam_128k.ABILITIES})


def test_beam_samples_are_chat_turns_plus_final_turn(tmp_path):
    path = _beam_parquet(str(tmp_path), n_conv=1)
    with mock.patch("huggingface_hub.hf_hub_download", side_effect=lambda *a, **k: path):
        samples = beam_128k._load_beam_128k_samples()
    messages, gold, meta = samples[0]
    assert len(messages) == 11 and messages[0] == {"role": "user", "content": "u0-0"}
    assert messages[-1] == {"role": "user", "content": beam_128k.FINAL_TURN.format(query=meta["question"])}
    assert json.loads(gold)["rubric"] == [f"{meta['category']} item"]
    assert "{'content'" not in json.dumps(messages)        # no dict reprs


def _char_bounds(messages, tools=None):
    n = sum(len(m["content"]) for m in messages) + 10 * len(messages)
    return n, n, "server"


def test_beam_prune_keeps_the_longest_even_tail_that_fits():
    history = []
    for k in range(10):
        history += [{"role": "user", "content": "u" * 100}, {"role": "assistant", "content": "a" * 100}]
    with mock.patch.object(beam_128k.base, "_prompt_token_bounds", _char_bounds):
        kept = beam_128k._prune_from_tail(history, "q" * 50, window=2000, reserve=500)
        full = beam_128k._prune_from_tail(history, "q" * 50, window=10_000, reserve=500)
        with pytest.raises(RuntimeError, match="INFRA REQUIRED"):
            beam_128k._prune_from_tail(history, "q" * 50, window=500, reserve=500)
    total = _char_bounds(kept + [{"role": "user", "content": "q" * 50}])[1]
    assert len(kept) % 2 == 0 and kept[0]["role"] == "user" and kept == history[-len(kept):]
    assert total + 16 + 500 <= 2000
    more = history[-(len(kept) + 2):] + [{"role": "user", "content": "q" * 50}]
    assert _char_bounds(more)[1] + 16 + 500 > 2000         # longest such tail
    assert full == history


@pytest.mark.parametrize("system,tau", [
    (["A", "B", "C"], 1.0),
    (["C", "B", "A"], 0.0),
])
def test_beam_event_ordering_tau_norm(system, tau):
    eo = beam_128k.event_ordering_score(["A", "B", "C"], system)
    assert eo["tau_norm"] == pytest.approx(tau) and eo["f1"] == 1.0


def test_beam_event_ordering_matches_upstream_on_a_partial_answer():
    # upstream: union [A,B,C,X], tie rank 5; ref ranks [1,2,3,5], sys ranks [2,5,1,5]... computed
    # with scipy kendalltau(variant="b") exactly as compute_metrics does
    from scipy.stats import kendalltau
    eo = beam_128k.event_ordering_score(["A", "B", "C"], ["C", "A", "X"])
    tau, _ = kendalltau([1, 2, 3, 5], [2, 5, 1, 3], variant="b")
    assert eo["tau_norm"] == pytest.approx((tau + 1) / 2)
    assert eo["precision"] == pytest.approx(2 / 3) and eo["recall"] == pytest.approx(2 / 3)


def _beam_trace(category, response, rubric):
    return {"category": category, "response_text": response,
            "gold_answer": json.dumps({"ideal": "", "rubric": rubric}),
            "extra_payload": {"question": "Q?", "category": category}}


def test_beam_judges_each_rubric_item_alone_and_averages(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    fake, calls = _cascade(['{"score": 1.0, "reason": "r"}', '```json\n{"score": 0.5}\n```',
                            '{"score": 0.0}'])
    usage = Counter()
    trace = _beam_trace("information_extraction", "The answer", ["i1", "i2", "i3"])
    with mock.patch.object(beam_128k, "judge_generate_cascade", fake):
        asyncio.run(beam_128k._make_scorer(usage, judge_concurrency=1)([trace]))
    assert [c[0] for c in calls] == [
        beam_128k.JUDGE_PROMPT.replace("<rubric_item>", i).replace("<llm_response>", "The answer")
        for i in ("i1", "i2", "i3")]
    assert all("Q?" not in c[0] for c in calls)             # the judge never sees the question
    assert trace["beam_score"] == pytest.approx(0.5) and trace["beam_rubric_scores"] == [1.0, 0.5, 0.0]
    assert usage == Counter({"gemini-test": 3})


def test_beam_event_ordering_uses_llm_alignment_and_tau(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    # 2 rubric-item scores, then equivalence answers: line "second" vs A (NO), vs B (YES);
    # line "first" vs A (YES).
    fake, calls = _cascade(['{"score": 1.0}', '{"score": 1.0}', "NO", "YES", "YES"])
    trace = _beam_trace("event_ordering", "second\nfirst", ["A", "B"])
    with mock.patch.object(beam_128k, "judge_generate_cascade", fake):
        asyncio.run(beam_128k._make_scorer(Counter(), judge_concurrency=1)([trace]))
    equiv = calls[2:]
    assert equiv[0][0] == beam_128k.EQUIV_USER.format(first="A", second="second")
    if equiv[0][1] is not None:
        assert _system_instruction(equiv[0][1]) == beam_128k.EQUIV_SYSTEM
    assert trace["beam_event_ordering"]["tau_norm"] == pytest.approx(0.0)   # reversed order
    assert trace["beam_score"] == pytest.approx(0.0)
    assert trace["beam_llm_judge_score"] == pytest.approx(1.0)


def test_beam_judge_failure_and_outage_are_not_scored(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    fake, _ = _cascade(["x", "y", "z", None])
    failed = _beam_trace("abstention", "r", ["i"])
    outage = _beam_trace("abstention", "r", ["i"])
    with mock.patch.object(beam_128k, "judge_generate_cascade", fake):
        asyncio.run(beam_128k._make_scorer(Counter(), judge_concurrency=1)([failed]))
        asyncio.run(beam_128k._make_scorer(Counter(), judge_concurrency=1)([outage]))
    assert failed["judge_grade"] == "JUDGE_FAILED" and "beam_score" not in failed
    assert outage["judge_grade"] == "JUDGE_OUTAGE" and "beam_score" not in outage


def test_beam_no_key_hard_errors(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    with pytest.raises(RuntimeError, match="INFRA REQUIRED"):
        asyncio.run(beam_128k._make_scorer(Counter())([_beam_trace("abstention", "r", ["i"])]))
    assert not hasattr(beam_128k, "_eval_beam_128k")


def test_beam_headline_is_the_mean_of_ability_means():
    traces = ([{"category": "abstention", "beam_score": 1.0, "judge_grade": "SCORE_1.00"}] * 3
              + [{"category": "event_ordering", "beam_score": 0.0, "judge_grade": "SCORE_0.00"}]
              + [{"category": "event_ordering", "judge_grade": "JUDGE_FAILED"}])
    r = beam_128k._finalize({"accuracy": 75.0, "sample_traces": traces,
                             "category_accuracy": {"abstention": {"accuracy": 100.0}}})
    assert r["accuracy"] == 50.0                            # macro, not 3/4 micro
    assert r["beam_ability_scores"] == {"abstention": 100.0, "event_ordering": 0.0}
    assert r["category_accuracy"]["abstention"]["pass_rate_at_0.5"] == 100.0


def test_beam_run_needs_a_window_and_reserves_the_output_budget(monkeypatch):
    seen = {}

    def _fake_run(**kw):
        seen.update(kw)
        return {"accuracy": 0.0, "sample_traces": [], "category_accuracy": {}}

    def _fake_load(**kw):
        seen["load"] = kw
        kw["stats"].update(conversations=20, pruned_conversations=17)
        return []
    patches = [mock.patch.object(beam_128k, "gemini_required_skip", lambda *a: None),
               mock.patch.object(beam_128k, "_load_beam_128k_samples", _fake_load),
               mock.patch.object(beam_128k, "run_eval_suite", _fake_run)]
    for p in patches:
        p.start()
    try:
        monkeypatch.delenv(beam_128k.WINDOW_ENV, raising=False)
        beam_128k.base._RUN_KNOBS.pop("max_model_len", None)
        beam_128k.base._RUN_KNOBS.pop("max_output_tokens", None)
        with pytest.raises(RuntimeError, match="INFRA REQUIRED"):
            beam_128k.run_beam_128k("m", "http://x/v1")
        monkeypatch.setenv(beam_128k.WINDOW_ENV, "131072")
        r = beam_128k.run_beam_128k("m", "http://x/v1", enable_thinking=True)
    finally:
        for p in patches:
            p.stop()
    assert seen["load"]["window"] == 131072
    # a --thinking run keeps its budget (the thinking floor here) as the answer room; no
    # max_tokens is sent (upstream's reader sends none)
    assert seen["load"]["output_reserve"] == beam_128k.base.THINKING_MIN_OUTPUT_TOKENS
    assert seen["official_sampling"].send_max_tokens is False
    assert seen["supports_attempts"] is True
    assert r["leaderboard_comparable"] is False
    assert "gpt-4.1-mini" in r["leaderboard_comparable_reason"]
    assert "17 of 20 conversations were tail-pruned" in r["leaderboard_comparable_reason"]


# =============================================================================
# healthbench (#11)
# =============================================================================
def test_healthbench_grader_template_is_simple_evals_verbatim():
    # sha256 of GRADER_TEMPLATE.strip() in openai/simple-evals @ 652c89d healthbench_eval.py
    assert _sha(healthbench.GRADER_TEMPLATE) == "2adffd51fd259554ebcd036ad1072d4aa2b7ce3aec2bbffe36271f911632ed3c"
    assert healthbench._REVISION == "40ee1968852fc57f625934251ac22be47077a8fb"


def _hb_trace(rubric, response="Go to the ER now."):
    return {"response_text": response, "gold_answer": json.dumps(rubric),
            "messages": [{"role": "user", "content": "chest pain?"}],
            "extra_payload": {"category": "theme:emergency_referrals",
                              "example_tags": ["theme:emergency_referrals"]}}


RUBRIC = [{"criterion": "Advises emergency care", "points": 10, "tags": ["axis:accuracy"]},
          {"criterion": "Mentions aspirin", "points": 5, "tags": ["axis:completeness"]},
          {"criterion": "Fails to refer", "points": -8, "tags": ["axis:accuracy"]}]


def test_healthbench_grades_each_item_with_points_and_the_conversation(monkeypatch):
    fake, calls = _cascade(['{"explanation": "e", "criteria_met": true}',
                            '```json\n{"explanation": "e", "criteria_met": false}\n```',
                            '{"explanation": "e", "criteria_met": true}'])
    trace = _hb_trace(RUBRIC)
    with _patch_detailed(healthbench, fake):
        asyncio.run(healthbench._make_scorer(Counter(), judge_concurrency=1)([trace]))
    convo = "user: chest pain?\n\nassistant: Go to the ER now."
    assert [c[0] for c in calls] == [
        healthbench.GRADER_TEMPLATE.replace("<<conversation>>", convo).replace(
            "<<rubric_item>>", item) for item in
        ("[10] Advises emergency care", "[5] Mentions aspirin", "[-8] Fails to refer")]
    if calls[0][1] is not None:
        assert _system_instruction(calls[0][1]) == healthbench.GRADER_SYSTEM_MESSAGE
    # (10 - 8) / 15: negative points count, and the example is NOT clipped
    assert trace["healthbench_score"] == pytest.approx(2 / 15, abs=1e-6)


def test_healthbench_per_example_score_is_not_clipped_but_the_mean_is():
    rubric = [{"criterion": "c", "points": 5, "tags": []}, {"criterion": "bad", "points": -10, "tags": []}]
    assert healthbench.calculate_score(rubric, [{"criteria_met": False}, {"criteria_met": True}]) == -2.0
    traces = [{"healthbench_score": -2.0, "judge_grade": "SCORE"}, {"healthbench_score": 1.0,
                                                                     "judge_grade": "SCORE"}]
    # upstream clip(mean) = 0.0; the old per-example clip would have given mean(0, 1) = 0.5
    assert healthbench.healthbench_summary(traces)["overall_score"] == 0.0


def test_healthbench_parse_failures_are_judge_failed_not_zero(monkeypatch):
    fake, calls = _cascade(["not json", '{"criteria_met": "yes"}', "{}"])
    trace = _hb_trace(RUBRIC[:1])
    with _patch_detailed(healthbench, fake):
        asyncio.run(healthbench._make_scorer(Counter(), judge_concurrency=1)([trace]))
    assert len(calls) == healthbench._GRADER_ATTEMPTS
    assert trace["judge_grade"] == "JUDGE_FAILED" and "healthbench_score" not in trace
    assert healthbench.healthbench_summary([trace]) == {}


def test_healthbench_tag_scores_follow_simple_evals():
    trace = {"healthbench_score": 2 / 15, "judge_grade": "SCORE",
             "extra_payload": {"example_tags": ["theme:emergency_referrals"]},
             "rubric_grades": [{**RUBRIC[0], "criteria_met": True}, {**RUBRIC[1], "criteria_met": False},
                               {**RUBRIC[2], "criteria_met": True}]}
    s = healthbench.healthbench_summary([trace])
    assert s["tag_scores"]["theme:emergency_referrals"] == pytest.approx(round(2 / 15, 4))
    assert s["tag_scores"]["axis:accuracy"] == pytest.approx(0.2)      # (10 - 8) / 10
    assert s["tag_scores"]["axis:completeness"] == 0.0


def test_healthbench_run_headline_and_flag():
    fake = {"accuracy": 0.0, "sample_traces": [
        {"healthbench_score": 1.4, "judge_grade": "SCORE", "extra_payload": {"example_tags": ["theme:t"]}},
        {"healthbench_score": 0.2, "judge_grade": "SCORE", "extra_payload": {"example_tags": ["theme:t"]}},
        {"judge_grade": "JUDGE_OUTAGE"}],
        "category_accuracy": {"theme:t": {"correct": 1, "total": 2, "accuracy": 50.0}}}
    with mock.patch.object(healthbench, "gemini_required_skip", lambda *a: None), \
         mock.patch.object(healthbench, "_load_healthbench_samples", lambda **k: []), \
         mock.patch.object(healthbench, "run_eval_suite", lambda **k: dict(fake)):
        r = healthbench.run_healthbench("m", "http://x/v1", 1)
    assert r["accuracy"] == 80.0                              # clip(mean(1.4, 0.2)) = 0.8
    assert r["category_accuracy"]["theme:t"]["accuracy"] == 80.0
    assert r["pass_rate_at_0.5"] == 0.0
    assert r["leaderboard_comparable"] is False and "gpt-4.1-2025-04-14" in r["leaderboard_comparable_reason"]


# =============================================================================
# imo_answer_bench (#13, #14, #26)
# =============================================================================
def _answerbench_text(broken=True):
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(["Problem ID", "Problem", "Short Answer", "Category", "Subcategory", "Source"])
    body = out.getvalue()
    lines = []
    for cat, slug in (("Algebra", "algebra"), ("Combinatorics", "combinatorics"),
                      ("Geometry", "geometry"), ("Number theory", "number_theory")):
        for i in range(1, 101):
            pid = f"imo-bench-{slug}-{i:03d}"
            if pid == "imo-bench-algebra-036":
                row = ('imo-bench-algebra-036,"Find all functions $Y$:\n\\[\nY(a)=b\n\\]\n'
                       + (',"' if broken else '","')
                       + '$Y(x)=A+\\frac{B}{x}-x$",Algebra,Functional Equation,Iran 2002\n')
                lines.append(row)
                continue
            s = io.StringIO()
            csv.writer(s, lineterminator="\n").writerow([pid, f"Problem {pid}", str(i), cat, "Sub", "Src"])
            lines.append(s.getvalue())
    return body + "".join(lines)


def test_imo_parse_repairs_the_one_malformed_v2_row():
    rows = imo_answer_bench.parse_answerbench_csv(_answerbench_text())
    assert len(rows) == 400
    r = [x for x in rows if x["Problem ID"] == "imo-bench-algebra-036"][0]
    assert r["Short Answer"] == "$Y(x)=A+\\frac{B}{x}-x$" and r["Category"] == "Algebra"
    assert r["Source"] == "Iran 2002" and r["Problem"].endswith("\\]\n")
    with pytest.raises(RuntimeError):                        # the pinned file changed
        imo_answer_bench.parse_answerbench_csv(_answerbench_text(broken=False))


def test_imo_fetch_is_pinned_and_sha256_verified(tmp_path, monkeypatch):
    assert imo_answer_bench.REVISION == "955558ad67ef24dccd27545c326988bdad1b4471"
    assert imo_answer_bench.REVISION in imo_answer_bench._CSV_URL
    assert imo_answer_bench._CSV_URL.endswith("imobench/answerbench_v2.csv")
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))
    path = imo_answer_bench._cache_path()
    path.parent.mkdir(parents=True)
    path.write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="sha256"):
        imo_answer_bench._fetch_csv()


def test_imo_grader_prompt_matches_the_paper_rendering():
    p = imo_answer_bench.GRADER_PROMPT
    assert p.startswith("# System Role: Deterministic Mathematical Autograder\n\n")
    assert p.endswith("Problem: {Problem_Statement}\nModel Solution: {Model_Solution}\n"
                      "Golden Answer: {Golden_Answer}")
    assert "Output exmaple:" in p and "# 3. Output Protocol" in p and "# 2." not in p
    filled = imo_answer_bench.grader_prompt("P {x}", "full solution ... \\boxed{3}", "3")
    assert "Model Solution: full solution ... \\boxed{3}" in filled and "Problem: P {x}" in filled


def _imo_trace(response):
    return {"response_text": response, "gold_answer": "3",
            "messages": [{"role": "user", "content": "Solve the following olympiad problem: ..."}],
            "extra_payload": {"problem": "Find n.", "category": "Algebra"}}


def test_imo_grader_sees_the_full_response_and_reads_the_boxed_verdict(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    response = "Long derivation.\nSo n = 3.\nFinal Answer: \\boxed{3}"
    fake, calls = _cascade(["<thinking>1. ... Correct</thinking>\n\\boxed{Correct}"])
    trace = _imo_trace(response)
    with mock.patch.object(imo_answer_bench, "judge_generate_cascade", fake):
        asyncio.run(imo_answer_bench._async_grade_imo([trace]))
    assert calls[0][0] == imo_answer_bench.grader_prompt("Find n.", response, "3")
    assert trace["is_correct"] is True and trace["judge_grade"] == "Correct"


def test_imo_every_response_goes_to_the_grader_and_failures_are_marked(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    # even a textual match is graded (no exact-match short-circuit), and a reply with no boxed
    # verdict is retried, then JUDGE_FAILED
    fake, calls = _cascade(["\\boxed{Incorrect}", "no verdict", "none", "nothing", None])
    exact, bad, out = (_imo_trace("\\boxed{3}"), _imo_trace("x"), _imo_trace("y"))
    with mock.patch.object(imo_answer_bench, "judge_generate_cascade", fake):
        for t in (exact, bad, out):
            asyncio.run(imo_answer_bench._async_grade_imo([t]))
    assert exact["is_correct"] is False and exact["judge_grade"] == "Incorrect"
    assert bad["judge_grade"] == "JUDGE_FAILED" and out["judge_grade"] == "JUDGE_OUTAGE"
    assert len(calls) == 5
    assert not hasattr(imo_answer_bench, "_eval_imo_answer_bench")


def test_imo_runs_eight_attempts_and_is_flagged():
    seen = {}

    def _fake_run(**kw):
        seen.update(kw)
        return {"accuracy": 10.0, "sample_traces": [{"judge_model": "gemini-x"}], "temperature": 0.0}
    with mock.patch.object(imo_answer_bench, "gemini_required_skip", lambda *a: None), \
         mock.patch.object(imo_answer_bench, "_load_imo_answer_bench_samples", lambda **k: []), \
         mock.patch.object(imo_answer_bench, "run_eval_suite", _fake_run):
        r = imo_answer_bench.run_imo_answer_bench("m", "http://x/v1", attempt_count=1)
    assert seen["attempt_count"] == 8 and seen["supports_attempts"] is True
    assert r["leaderboard_comparable"] is False
    reason = r["leaderboard_comparable_reason"]
    assert "Gemini 2.5 Pro" in reason and "no judge call answered" in reason and "temperature 0.0" in reason
    assert "answerbench_v2.csv" in r["dataset"] and "8 runs" in r["metric"]


def test_imo_docs_match_the_code():
    doc = (ROOT / "docs" / "evals" / "imo_answer_bench.md").read_text(encoding="utf-8")
    assert "lower bound" not in doc                         # #26: no degraded no-key mode
    assert "infra_required" in doc and "GEMINI_API_KEY" in doc
    assert imo_answer_bench.REVISION in doc and "answerbench_v2.csv" in doc
    assert "8" in doc and "Gemini 2.5 Pro" in doc


@pytest.mark.parametrize("suite,needles", [
    ("aa_lcr", [aa_lcr._REVISION, "GPT-5.6 Luna", "leaderboard_comparable", "3 repeats"]),
    ("beam_128k", [beam_128k._REVISION, beam_128k.UPSTREAM_COMMIT, beam_128k.WINDOW_ENV,
                   "tau_norm", "leaderboard_comparable"]),
    ("healthbench", [healthbench._REVISION, "gpt-4.1-2025-04-14", "JUDGE_FAILED",
                     "leaderboard_comparable"]),
])
def test_docs_name_the_pins_and_flags(suite, needles):
    doc = (ROOT / "docs" / "evals" / f"{suite}.md").read_text(encoding="utf-8")
    for n in needles:
        assert n in doc, f"docs/evals/{suite}.md does not mention {n!r}"


# =============================================================================
# end to end through base.run_eval_suite (fake server + fake judge)
# =============================================================================
def _fake_server(text):
    from gbench.runners.eval_suites import base

    async def _send(**kw):
        return base.Reply(text=text, tool_calls=None, finish_reason="stop")
    return mock.patch.object(base, "_send_single_request", side_effect=_send)


def _always(reply):
    async def _fake(contents, *, config=None):
        return reply, "gemini-e2e"
    return _fake


def test_e2e_aa_lcr_three_attempts(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    samples = [([{"role": "user", "content": "p1"}], "A", {"category": "c", "question": "q1"}),
               ([{"role": "user", "content": "p2"}], "B", {"category": "c", "question": "q2"})]
    with _fake_server("A"), \
         mock.patch.object(aa_lcr, "gemini_required_skip", lambda *a: None), \
         mock.patch.object(aa_lcr, "_load_aa_lcr_samples", lambda **k: samples), \
         mock.patch.object(aa_lcr, "judge_generate_cascade", _always('{"verdict": "CORRECT"}')):
        r = aa_lcr.run_aa_lcr("m", "http://x/v1")
    assert r["total_questions"] == 6 and r["accuracy"] == 100.0
    assert r["attempts"]["attempts_per_sample"] == 3 and r["judge_models"] == {"gemini-e2e": 6}


def test_e2e_imo_eight_attempts(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    samples = [([{"role": "user", "content": "Solve"}], "3", {"category": "Algebra", "problem": "P"})]
    with _fake_server("\\boxed{3}"), \
         mock.patch.object(imo_answer_bench, "gemini_required_skip", lambda *a: None), \
         mock.patch.object(imo_answer_bench, "_load_imo_answer_bench_samples", lambda **k: samples), \
         mock.patch.object(imo_answer_bench, "judge_generate_cascade", _always("\\boxed{Correct}")):
        r = imo_answer_bench.run_imo_answer_bench("m", "http://x/v1")
    assert r["total_questions"] == 8 and r["accuracy"] == 100.0 and r["status"] == "success"


def test_e2e_healthbench(monkeypatch):
    samples = [([{"role": "user", "content": "q"}], json.dumps(RUBRIC),
                {"category": "theme:t", "example_tags": ["theme:t"]})]
    with _fake_server("answer"), \
         mock.patch.object(healthbench, "gemini_required_skip", lambda *a: None), \
         mock.patch.object(healthbench, "_load_healthbench_samples", lambda **k: samples), \
         _patch_detailed(healthbench,
                           _always('{"explanation": "e", "criteria_met": true}')):
        r = healthbench.run_healthbench("m", "http://x/v1", 1)
    # all three met: (10 + 5 - 8) / 15
    assert r["accuracy"] == pytest.approx(round(7 / 15 * 100, 2))
    assert r["judge_models"] == {"gemini-e2e": 3} and r["status"] == "success"


def test_e2e_beam(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv(beam_128k.WINDOW_ENV, "131072")
    samples = [([{"role": "user", "content": "h"}, {"role": "user", "content": "q"}],
                json.dumps({"ideal": "", "rubric": ["i1", "i2"]}), {"category": "abstention"})]
    with _fake_server("answer"), \
         mock.patch.object(beam_128k, "gemini_required_skip", lambda *a: None), \
         mock.patch.object(beam_128k, "_load_beam_128k_samples", lambda **k: samples), \
         mock.patch.object(beam_128k, "judge_generate_cascade", _always('{"score": 0.5}')):
        r = beam_128k.run_beam_128k("m", "http://x/v1")
    assert r["accuracy"] == 50.0 and r["beam_ability_scores"] == {"abstention": 50.0}
    assert r["beam_window_tokens"] == 131072 and r["status"] == "success"


# =============================================================================
# Judged suites: a request that got no reply from the served endpoint is a harness error (left
# out of the means); key lists, judge record, pruning
# =============================================================================
def test_healthbench_failed_generation_is_a_harness_error_left_out_of_the_mean():
    # a request that got no reply from the served endpoint is a harness error: not graded, left
    # out of the mean, never scored as the model's outcome
    rubric = [{"criterion": "Refers to emergency care", "points": 10, "tags": ["axis:accuracy"]}]
    answered = _hb_trace(rubric, response="Call emergency services now.")
    failed = {**_hb_trace(rubric), "response_text": None, "status": "TIMEOUT",
              "harness_error": {"kind": "endpoint_timeout", "detail": "timed out"}}
    prompts = []

    async def _grader(contents, *, config=None):
        prompts.append(contents)
        met = "assistant: Call emergency services now." in contents
        return json.dumps({"explanation": "e", "criteria_met": met}), "gemini-test"

    with _patch_detailed(healthbench, _grader):
        asyncio.run(healthbench._make_scorer(Counter(), judge_concurrency=1)([answered, failed]))
    assert len(prompts) == 1 and not any("No response (bad request)." in p for p in prompts)
    assert "healthbench_score" not in failed and failed["scoring_excluded"] is True
    assert failed["judge_grade"] == "not_judged_failed_request"
    assert failed["status"] == "TIMEOUT"                   # base still counts the failure
    s = healthbench.healthbench_summary([answered, failed])
    assert s["n_samples"] == 1 and s["overall_score"] == 1.0


def test_e2e_healthbench_failed_request_is_left_out_of_the_headline():
    from gbench.runners.eval_suites import base
    samples = [([{"role": "user", "content": q}], json.dumps(RUBRIC[:1]),
                {"category": "theme:t", "example_tags": ["theme:t"]}) for q in ("q1", "q2")]

    async def _send(**kw):
        if kw["messages"][-1]["content"] == "q2":     # after simple-evals' system turn
            return base.Reply(text=None, tool_calls=None, error="server error")
        return base.Reply(text="Go to the ER.", tool_calls=None, finish_reason="stop")

    async def _grader(contents, *, config=None):
        met = "assistant: Go to the ER." in contents
        return json.dumps({"explanation": "e", "criteria_met": met}), "gemini-e2e"

    with mock.patch.object(base, "_send_single_request", side_effect=_send), \
         mock.patch.object(healthbench, "gemini_required_skip", lambda *a: None), \
         mock.patch.object(healthbench, "_load_healthbench_samples", lambda **k: samples), \
         _patch_detailed(healthbench, _grader):
        r = healthbench.run_healthbench("m", "http://x/v1", 1)
    assert r["accuracy"] == 100.0 and r["healthbench_scored_conversations"] == 1
    assert r["failed_requests"] == 1 and r["scoring_excluded"] == 1
    assert r["status"] == "completed_with_errors"
    assert r["judge"] == "gbench-gemini-cascade" and r["judge_models"] == {"gemini-e2e": 1}
    assert "1 request(s) got no reply from the served endpoint" in r["leaderboard_comparable_reason"]


def test_beam_failed_request_is_a_harness_error_without_a_judge_call(monkeypatch):
    # a request that got no reply from the served endpoint is a harness error: not graded, left
    # out of the ability means, never scored 0.0 as the model's answer
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    fake, calls = _cascade(['{"score": 1.0}'])
    answered = _beam_trace("abstention", "I do not know.", ["declines"])
    failed = {**_beam_trace("abstention", "", ["declines"]), "response_text": None,
              "status": "FAILED", "error": "HTTP 503"}
    failed_eo = {**_beam_trace("event_ordering", "", ["A", "B"]), "response_text": None,
                 "status": "TIMEOUT"}
    with mock.patch.object(beam_128k, "judge_generate_cascade", fake):
        asyncio.run(beam_128k._make_scorer(Counter(), judge_concurrency=1)(
            [answered, failed, failed_eo]))
    assert len(calls) == 1                                  # only the answered trace
    for t in (failed, failed_eo):
        assert "beam_score" not in t and t["scoring_excluded"] is True
        assert t["judge_grade"] == beam_128k.HARNESS_ERROR_GRADE
    assert failed["harness_error"]["kind"] == "endpoint_failure"
    assert failed_eo["harness_error"]["kind"] == "endpoint_timeout"
    r = beam_128k._finalize({"accuracy": 0.0, "sample_traces": [answered, failed, failed_eo],
                             "category_accuracy": {}})
    assert r["beam_ability_scores"] == {"abstention": 100.0}
    assert r["accuracy"] == 100.0
    assert r["harness_errors"] == {"endpoint_failure": 1, "endpoint_timeout": 1}


def _token_bounds(messages, tools=None):
    # "W" costs 10 tokens, any other character 1: the longest question in characters is not the
    # longest in tokens.
    n = sum(len(m["content"]) + 9 * m["content"].count("W") for m in messages) + 10 * len(messages)
    return n, n, "server"


def test_beam_prune_is_sized_for_the_question_with_the_most_tokens(tmp_path):
    import pandas as pd
    probing = {a: [{"question": f"{a[:4]} q{i}", "rubric": ["item"], "answer": "x"}
                   for i in range(2)] for a in beam_128k.ABILITIES}
    probing["abstention"] = [{"question": "W" * 8, "rubric": ["item"], "answer": "x"},
                             {"question": "x" * 40, "rubric": ["item"], "answer": "x"}]
    history_pairs = [[{"role": "user", "content": f"u0-{k}"},
                      {"role": "assistant", "content": f"a0-{k}"}] for k in range(5)]
    path = os.path.join(str(tmp_path), "beam.parquet")
    pd.DataFrame([{"conversation_id": "0", "chat": history_pairs,
                   "probing_questions": repr(probing)}]).to_parquet(path)
    history = [m for pair in history_pairs for m in pair]
    by_chars = beam_128k.FINAL_TURN.format(query="x" * 40)
    by_tokens = beam_128k.FINAL_TURN.format(query="W" * 8)
    reserve = 100
    # a window that holds three pairs with the char-longest question, fewer with the other one
    window = _token_bounds(history[-6:] + [{"role": "user", "content": by_chars}])[1] + 16 + reserve
    with mock.patch.object(beam_128k.base, "_prompt_token_bounds", _token_bounds):
        assert beam_128k._longest_final_turn([by_chars, by_tokens]) == by_tokens
        old = beam_128k._prune_from_tail(history, by_chars, window, reserve)
        assert _token_bounds(old + [{"role": "user", "content": by_tokens}])[1] + 16 + reserve > window
        # the same rule over whole tails (a question turn plus the earlier question turns
        # upstream's prompt carries): the tail with the most tokens wins, not the most characters
        tail_c, tail_t = ([{"role": "user", "content": by_chars}],
                          [{"role": "user", "content": by_tokens}])
        assert beam_128k._largest_tail([tail_c, tail_t]) == tail_t
        questions = [q["question"] for a in beam_128k.ABILITIES for q in probing[a]]
        largest = beam_128k._largest_tail([beam_128k._question_tail(questions, p)
                                           for p in range(len(questions))])
        window = _token_bounds(history[-6:] + largest)[1] + 16 + reserve
        with mock.patch("huggingface_hub.hf_hub_download", side_effect=lambda *a, **k: path):
            samples = beam_128k._load_beam_128k_samples(window=window, output_reserve=reserve)
    assert len(samples) == 20
    for messages, _gold, _meta in samples:                  # every question fits with the reserve
        assert _token_bounds(messages)[1] + 16 + reserve <= window
    assert {meta["history_messages"] for _m, _g, meta in samples} == {6}


def test_in_judge_key_checks_accept_a_list_set_only_as_gemini_api_keys(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEYS", "k1,k2")
    aa, _ = _cascade(['{"verdict": "CORRECT"}'])
    imo, _ = _cascade(["\\boxed{Correct}"])
    beam, _ = _cascade(['{"score": 1.0}'])
    a_trace, i_trace = _aa_trace(), _imo_trace("\\boxed{3}")
    b_trace = _beam_trace("abstention", "r", ["i"])
    with mock.patch.object(aa_lcr, "judge_generate_cascade", aa), \
         mock.patch.object(imo_answer_bench, "judge_generate_cascade", imo), \
         mock.patch.object(beam_128k, "judge_generate_cascade", beam):
        asyncio.run(aa_lcr._async_judge_aa_lcr([a_trace]))
        asyncio.run(imo_answer_bench._async_grade_imo([i_trace]))
        asyncio.run(beam_128k._make_scorer(Counter())([b_trace]))
    assert a_trace["judge_grade"] == "CORRECT" and i_trace["judge_grade"] == "Correct"
    assert b_trace["beam_score"] == 1.0
    for mod in (aa_lcr, beam_128k, imo_answer_bench):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert 'os.environ.get("GEMINI_API_KEY")' not in src and "gemini_api_keys()" in src


def test_judge_models_counts_answered_calls_including_retries(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    samples = [([{"role": "user", "content": "p1"}], "A", {"category": "c", "question": "q1"})]
    replies = iter(["no verdict", '{"verdict": "CORRECT"}'] * 3)

    async def _judge(contents, *, config=None):
        return next(replies), "gemini-e2e"
    with _fake_server("A"), \
         mock.patch.object(aa_lcr, "gemini_required_skip", lambda *a: None), \
         mock.patch.object(aa_lcr, "_load_aa_lcr_samples", lambda **k: samples), \
         mock.patch.object(aa_lcr, "judge_generate_cascade", _judge):
        r = aa_lcr.run_aa_lcr("m", "http://x/v1")
    # 3 attempts x (1 unparsable + 1 verdict) = 6 answered calls for 3 judged traces
    assert r["judge_models"] == {"gemini-e2e": 6} and r["accuracy"] == 100.0
    assert r["judge"] == "gbench-gemini-cascade" and "gemini-e2e" in r["leaderboard_comparable_reason"]


def test_e2e_imo_records_the_judge_schema(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    samples = [([{"role": "user", "content": "Solve"}], "3", {"category": "Algebra", "problem": "P"})]
    with _fake_server("\\boxed{3}"), \
         mock.patch.object(imo_answer_bench, "gemini_required_skip", lambda *a: None), \
         mock.patch.object(imo_answer_bench, "_load_imo_answer_bench_samples", lambda **k: samples), \
         mock.patch.object(imo_answer_bench, "judge_generate_cascade", _always("\\boxed{Correct}")):
        r = imo_answer_bench.run_imo_answer_bench("m", "http://x/v1")
    assert r["judge"] == "gbench-gemini-cascade" and r["judge_models"] == {"gemini-e2e": 8}
    assert "gemini-e2e" in r["leaderboard_comparable_reason"]


def test_e2e_beam_records_the_judge_schema(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv(beam_128k.WINDOW_ENV, "131072")
    samples = [([{"role": "user", "content": "q"}], json.dumps({"ideal": "", "rubric": ["i1", "i2"]}),
                {"category": "abstention"})]
    with _fake_server("answer"), \
         mock.patch.object(beam_128k, "gemini_required_skip", lambda *a: None), \
         mock.patch.object(beam_128k, "_load_beam_128k_samples", lambda **k: samples), \
         mock.patch.object(beam_128k, "judge_generate_cascade", _always('{"score": 1.0}')):
        r = beam_128k.run_beam_128k("m", "http://x/v1")
    assert r["judge"] == "gbench-gemini-cascade" and r["judge_models"] == {"gemini-e2e": 2}
    assert r["judge_cascade"] and r["harness_errors"] == {}


@pytest.mark.parametrize("suite,needles", [
    ("aa_lcr", ["it stays in the denominator", "`judge_no_verdict`"]),
    ("beam_128k", ["`harness_errors`", "most tokens"]),
    ("healthbench", ["No response (bad request).", "not_judged_failed_request"]),
    ("imo_answer_bench", ["stays in the\n  denominator"]),
])
def test_docs_match_the_key_gate_failures_judge_schema_and_cli(suite, needles):
    doc = (ROOT / "docs" / "evals" / f"{suite}.md").read_text(encoding="utf-8")
    run = doc.split("## Run", 1)[1].split("```")[1]
    assert "--max-output-tokens" in run                   # required by the CLI
    assert "`GEMINI_API_KEYS`" in doc and "`judge_cascade`" in doc and "`judge`" in doc
    assert "unless `--max-output-tokens` is set" not in doc
    for n in needles:
        assert n in doc, f"docs/evals/{suite}.md does not mention {n!r}"
    assert all(ord(ch) < 128 for ch in doc)


def test_e2e_beam_failed_request_is_left_out_of_the_headline(monkeypatch):
    from gbench.runners.eval_suites import base
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv(beam_128k.WINDOW_ENV, "131072")
    gold = json.dumps({"ideal": "", "rubric": ["i1"]})
    samples = [([{"role": "user", "content": q}], gold, {"category": "abstention"})
               for q in ("q1", "q2")]

    async def _send(**kw):
        if kw["messages"][0]["content"] == "q2":
            return base.Reply(text=None, tool_calls=None, error="timeout")
        return base.Reply(text="answer", tool_calls=None, finish_reason="stop")
    with mock.patch.object(base, "_send_single_request", side_effect=_send), \
         mock.patch.object(beam_128k, "gemini_required_skip", lambda *a: None), \
         mock.patch.object(beam_128k, "_load_beam_128k_samples", lambda **k: samples), \
         mock.patch.object(beam_128k, "judge_generate_cascade", _always('{"score": 1.0}')):
        r = beam_128k.run_beam_128k("m", "http://x/v1")
    # a harness error: out of the mean (never a 0.0), counted and named
    assert r["accuracy"] == 100.0 and r["beam_ability_scores"] == {"abstention": 100.0}
    assert sum(r["harness_errors"].values()) == 1 and r["failed_requests"] == 1
    assert r["status"] == "completed_with_errors"
    assert r["judge_models"] == {"gemini-e2e": 1}          # no judge call for the failure
    assert "harness errors, not the model's outcome" in r["leaderboard_comparable_reason"]
