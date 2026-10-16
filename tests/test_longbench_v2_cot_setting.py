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

"""longbench_v2_code: LongBench-v2's w/ CoT setting next to the non-CoT one.

THUDM/LongBench@2e00731f pred.py --cot: prompts/0shot_cot.txt verbatim, the CoT query at
max_new_tokens=1024, then prompts/0shot_cot_ans.txt with the stripped CoT at $COT$ as the answer
query (max_new_tokens=128), extract_answer on its reply; a CoT query that returns '' gets no
answer query. Both settings run by default and each is scored on its own; the token-limit policy
applies per query and mode.

No network and no model server: the dataset, the tokenizer and every request are faked in
process.
"""

import hashlib
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import base, longbench_v2_code as L

REPO = Path(__file__).resolve().parent.parent

#: sha256 of each prompt file at THUDM/LongBench@2e00731f8d0bff23dc4325161044d0ed8af94c1e.
_UPSTREAM_SHA256 = {
    "_TEMPLATE_0SHOT": "68a162252bc9ff71d5d7abca3d69bb31aac3c35f832d657a2866f2018b8a6950",
    "_TEMPLATE_0SHOT_COT": "6cc33d92fd8f611662b756c1f10f302c541ca005b58c46d5f3188ac3a939bfe8",
    "_TEMPLATE_0SHOT_COT_ANS": "85ec59df4ac0e27aabbc619a7ca5cf3ae1787dd5a76cf51a18944eb52c91955f",
}


class _CharTok:
    """One token per character, as the served tokenizer would be used by pred.py."""

    def encode(self, s):
        return [ord(c) for c in s]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


def _rows(n, context="class A:\n    pass"):
    return [{"_id": f"id{i}", "domain": "Code Repository Understanding", "context": context,
             "question": f" Which class is defined {i}? ", "choice_A": " A ", "choice_B": "B",
             "choice_C": "C", "choice_D": "D", "answer": "A", "difficulty": "easy",
             "length": "short"} for i in range(n)]


def _pred_py_answer_prompt(template, item, output):
    """pred.py --cot, verbatim: the answer prompt after the CoT query returned `output`."""
    response = output.strip()
    context = item["context"]
    return template.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip()).replace('$COT$', response)  # noqa: E501


def _query_kind(messages):
    text = messages[-1]["content"]
    if text.startswith(L._TEMPLATE_0SHOT_COT_ANS.split("$Q$")[0]):
        return "answer"
    if text.endswith("Let\u2019s think step by step:"):
        return "cot"
    return "non_cot"


def _run(monkeypatch, replies, *, window=131072, rows=None, thinking=False, **kwargs):
    """run_longbench_v2_code through base with the real loader on fake rows; `replies` maps a
    query kind (non_cot / cot / answer) to a Reply or a function of the messages. Returns
    (result, requests sent as (kind, kwargs))."""
    sent = []

    async def fake_send(**kw):
        kind = _query_kind(kw["messages"])
        sent.append((kind, kw))
        reply = replies[kind]
        return reply(kw["messages"]) if callable(reply) else reply

    monkeypatch.setattr(base, "_send_single_request", fake_send)
    monkeypatch.setenv("GBENCH_LONGBENCH_V2_CODE_ALL_DOMAINS", "1")
    monkeypatch.setenv("GBENCH_LONGBENCH_V2_CODE_WINDOW", str(window))
    if not thinking:
        # pred.py's 0.1, set explicitly (gbench's no-think 0.0 is not comparable), so these
        # tests see only the settings' own differences
        monkeypatch.setenv("GBENCH_LONGBENCH_V2_CODE_TEMPERATURE", "0.1")
    monkeypatch.setattr(L, "_load_tokenizer", lambda name: _CharTok())
    with mock.patch("datasets.load_dataset", return_value=rows if rows is not None else _rows(2)):
        res = L.run_longbench_v2_code("m", "http://x/v1", 1, enable_thinking=thinking, **kwargs)
    return res, sent


_RIGHT = base.Reply("The correct answer is (A)", None, "stop", completion_tokens=6)
_WRONG = base.Reply("The correct answer is (B)", None, "stop", completion_tokens=6)
_COT = base.Reply("  Class A is defined at the top.  ", None, "stop", completion_tokens=9)


# --------------------------------------------------------------------------- #
# the upstream files, verbatim
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,digest", sorted(_UPSTREAM_SHA256.items()))
def test_prompt_templates_are_the_upstream_files_verbatim(name, digest):
    assert hashlib.sha256(getattr(L, name).encode("utf-8")).hexdigest() == digest


def test_answer_prompt_is_pred_py_replace_chain_with_the_cot_last():
    # A question that itself contains placeholders shows the order: $COT$ is replaced last.
    item = {"context": " DOC $Q$ ", "question": " Q $COT$ $C_A$? ", "choice_A": " a ",
            "choice_B": "b", "choice_C": "c", "choice_D": "d"}
    output = "  step one $C_B$  "
    filled = L._fill_prompt(item, L._TEMPLATE_0SHOT_COT_ANS)
    assert "$COT$" in filled and "DOC" not in filled      # no $DOC$ in 0shot_cot_ans.txt
    assert L._cot_answer_prompt(filled, output.strip()) == _pred_py_answer_prompt(
        L._TEMPLATE_0SHOT_COT_ANS, item, output)


def test_cot_loader_uses_0shot_cot_and_carries_the_answer_fill(monkeypatch):
    monkeypatch.setenv("GBENCH_LONGBENCH_V2_CODE_ALL_DOMAINS", "1")
    with mock.patch("datasets.load_dataset", return_value=_rows(6)):
        plain = L._load_longbench_v2_code_samples(limit=3, tokenizer=_CharTok(),
                                                  max_prompt_tokens=120000)
        cot = L._load_longbench_v2_code_samples(limit=3, tokenizer=_CharTok(),
                                                max_prompt_tokens=120000, cot=True)
    # The same questions, in the same order, in both settings.
    assert [s[2]["category"] for s in plain] == [s[2]["category"] for s in cot]
    assert [s[1] for s in plain] == [s[1] for s in cot]
    for (pm, _, pmeta), (cm, _, cmeta) in zip(plain, cot):
        assert pm[0]["content"].endswith('"The correct answer is (insert answer here)".')
        assert cm[0]["content"].endswith("Let\u2019s think step by step:")
        assert "cot_answer_prompt" not in pmeta
        answer = cmeta["cot_answer_prompt"]
        assert answer.count("$COT$") == 1 and "class A" not in answer
        assert "max_tokens" not in cmeta              # the caps are OfficialSampling's
        assert pm[0]["content"].split("What is the correct")[1] == \
            cm[0]["content"].split("What is the correct")[1].replace(
                "Let\u2019s think step by step:",
                'Format your response as follows: "The correct answer is (insert answer here)".')


def test_official_caps_follow_the_token_policy_per_query():
    assert L.OFFICIAL_SAMPLING_COT.max_tokens == 1024
    assert L.OFFICIAL_SAMPLING_COT_ANSWER.max_tokens == 128
    for s in (L.OFFICIAL_SAMPLING, L.OFFICIAL_SAMPLING_COT, L.OFFICIAL_SAMPLING_COT_ANSWER):
        assert s.temperature == 0.1 and s.reasoning is None and "2e00731f" in s.reference
    assert L._output_budget(False, L.OFFICIAL_SAMPLING_COT) == 1024
    assert L._output_budget(False, L.OFFICIAL_SAMPLING_COT_ANSWER) == 128
    base.set_run_knobs(max_output_tokens=65536)
    assert L._output_budget(False, L.OFFICIAL_SAMPLING_COT) == 1024     # no-think: exact
    assert L._output_budget(True, L.OFFICIAL_SAMPLING_COT) == 65536     # never below the flag
    assert L._output_budget(True, L.OFFICIAL_SAMPLING_COT_ANSWER) == 65536


# --------------------------------------------------------------------------- #
# end to end through base
# --------------------------------------------------------------------------- #
def test_both_settings_run_and_are_scored_on_their_own(monkeypatch):
    base.set_run_knobs(max_output_tokens=65536)
    res, sent = _run(monkeypatch, {"non_cot": _WRONG, "cot": _COT, "answer": _RIGHT})
    kinds = [k for k, _ in sent]
    assert kinds == ["non_cot"] * 2 + ["cot"] * 2 + ["answer"] * 2
    caps = {k: {kw["max_output_tokens"] for kk, kw in sent if kk == k} for k in set(kinds)}
    assert caps == {"non_cot": {128}, "cot": {1024}, "answer": {128}}
    assert all(kw["temperature"] == 0.1 for _, kw in sent)
    # The answer query carries the stripped CoT at $COT$, and no document.
    answer = [kw["messages"] for k, kw in sent if k == "answer"][0][0]["content"]
    assert "Let\u2019s think step by step: Class A is defined at the top.\n\nBased on" in answer
    assert "class A:" not in answer
    # Each setting has its own accuracy; the headline is non-CoT.
    assert res["headline_setting"] == "non_cot"
    assert res["accuracy"] == 0.0 and res["settings"]["non_cot"]["accuracy"] == 0.0
    assert res["settings"]["cot"]["accuracy"] == 100.0
    assert res["headline_metrics"] == {"non_cot_accuracy": 0.0, "cot_accuracy": 100.0}
    assert res["settings"]["cot"]["total_questions"] == 2
    assert res["settings"]["cot"]["correct_answers"] == 2
    # The official caps were applied, and the flag recorded as not applied, per query.
    cot_os = res["settings"]["cot"]["official_sampling"]
    assert cot_os["cot_query"]["max_tokens"] == 1024 and cot_os["answer_query"]["max_tokens"] == 128
    assert cot_os["cot_query"]["max_tokens_applied"] is True
    assert cot_os["answer_query"]["max_tokens_applied"] is True
    assert res["official_sampling"]["max_tokens"] == 128
    assert "CoT query: --max-output-tokens 65536 not applied" in \
        res["settings"]["cot"]["max_output_tokens_note"]
    assert res["status"] == "success" and res["settings"]["cot"]["status"] == "success"
    assert res["leaderboard_comparable"] is True
    assert res["settings"]["cot"]["leaderboard_comparable"] is True
    report = res["settings"]["cot"]["longbench_v2_code_report"]
    assert report["output_budget_tokens"] == 1024 and report["answer_output_budget_tokens"] == 128
    assert report["max_prompt_tokens"] == 120000 and report["questions_without_cot"] == 0
    # Both settings' traces, marked; a w/ CoT trace is the answer query's.
    traces = res["sample_traces"]
    assert [t["setting"] for t in traces] == ["non_cot"] * 2 + ["cot"] * 2
    assert len({t["sample_idx"] for t in traces}) == 4
    cot_traces = [t for t in traces if t["setting"] == "cot"]
    assert all(t["cot_step"] == "answer" and t["is_correct"] for t in cot_traces)
    assert all(t["extra_payload"]["cot_query_max_tokens_sent"] == 1024 for t in cot_traces)
    assert all("cot_answer_prompt" not in t["extra_payload"] for t in cot_traces)


def test_thinking_sends_the_run_budget_to_every_query_and_the_content_as_cot(monkeypatch):
    base.set_run_knobs(max_output_tokens=65536)
    cot = base.Reply("visible chain", None, "stop", reasoning="hidden reasoning",
                     completion_tokens=50)
    res, sent = _run(monkeypatch, {"non_cot": _RIGHT, "cot": cot, "answer": _RIGHT},
                     thinking=True)
    assert {kw["max_output_tokens"] for _, kw in sent} == {65536}
    answer = [kw["messages"] for k, kw in sent if k == "answer"][0][0]["content"]
    assert "step by step: visible chain\n" in answer and "hidden reasoning" not in answer
    cot_os = res["settings"]["cot"]["official_sampling"]
    for block, cap in (("cot_query", 1024), ("answer_query", 128)):
        assert cot_os[block]["max_tokens"] == cap
        assert cot_os[block]["max_tokens_applied"] is False
        assert cot_os[block]["max_tokens_sent"] == 65536
    reason = res["leaderboard_comparable_reason"]
    assert "max_tokens=1024 (CoT query)" in reason and "(w/ CoT)" in reason
    assert "max_tokens=128 is not applied to reasoning (non-CoT)" in reason
    assert res["settings"]["cot"]["longbench_v2_code_report"]["max_prompt_tokens"] == \
        131072 - 65536 - 64


@pytest.mark.parametrize("cot_reply,accuracy,reason", [
    # pred.py: output == '' gets no prediction and a resumed run asks again; gbench's relay
    # re-asks it (NULL_REPLY_REASKS), and one still empty after them is scored wrong
    (base.Reply("", None, "stop"), 50.0,
     "1 question(s) (to the CoT query or the answer query) still came back with empty or null "
     "content when gbench stopped re-asking (after up to 5 re-asks)"),
    # 5 failed tries -> ''; no reply from the endpoint is a harness error, left out
    (base.Reply(None, None, None, error="HTTP 500"), 100.0,
     "1 question(s) (to the CoT query or the answer query) got no reply from the served "
     "endpoint"),
])
def test_an_empty_or_failed_cot_query_gets_no_answer_query(monkeypatch, cot_reply, accuracy,
                                                           reason):
    seen_q = []
    empties = []

    def cot(messages):
        seen_q.append(messages[0]["content"])
        if "defined 0?" in messages[0]["content"]:
            if cot_reply.text == "":
                empties.append(1)
            return cot_reply
        return _COT
    # base's requests are faked below the re-ask relay, so the relay sees none of them: an empty
    # reply that reached base is one the relay's re-asks did not get past (reasks_exhausted)
    real_summary = L.reask_summary

    def summary(relay, rule):
        out = real_summary(relay, rule)
        out["reasks_exhausted"] = len(empties)
        empties.clear()
        return out
    monkeypatch.setattr(L, "reask_summary", summary)
    res, sent = _run(monkeypatch, {"non_cot": _RIGHT, "cot": cot, "answer": _RIGHT})
    assert [k for k, _ in sent].count("answer") == 1
    entry = res["settings"]["cot"]
    assert entry["total_questions"] == 2 and entry["correct_answers"] == 1
    assert entry["accuracy"] == accuracy and entry["unanswered_questions"] == 1
    assert entry["longbench_v2_code_report"]["questions_without_cot"] == 1
    assert entry["status"] == "completed_with_errors" and res["status"] == "completed_with_errors"
    assert entry["status_reason"] and "w/ CoT: " in res["status_reason"]
    assert entry["leaderboard_comparable"] is False
    assert reason in entry["leaderboard_comparable_reason"]
    assert "(w/ CoT)" in res["leaderboard_comparable_reason"]
    assert res["settings"]["non_cot"]["leaderboard_comparable"] is True
    no_cot = [t for t in res["sample_traces"] if t.get("cot_step") == "cot"]
    assert len(no_cot) == 1 and no_cot[0]["is_correct"] is False and no_cot[0]["error"]
    assert "cot_answer_prompt" not in no_cot[0]["extra_payload"]
    # One trace per question in each setting: unique sample_idx, the question index shared.
    traces = res["sample_traces"]
    assert sorted(t["sample_idx"] for t in traces) == [0, 1, 2, 3]
    cot_traces = [t for t in traces if t["setting"] == "cot"]
    assert [(t["source_sample_idx"], t["cot_step"]) for t in cot_traces] == [(0, "cot"),
                                                                             (1, "answer")]
    assert [t["source_sample_idx"] for t in traces if t["setting"] == "non_cot"] == [0, 1]


def test_a_window_only_the_non_cot_query_fits_reports_cot_infra_required(monkeypatch):
    # 4,096 + 128 + 64 <= 5,000 < 4,096 + 1,024 + 64: room for the non-CoT query only.
    res, sent = _run(monkeypatch, {"non_cot": _RIGHT, "cot": _COT, "answer": _RIGHT},
                     window=5000)
    assert {k for k, _ in sent} == {"non_cot"}
    entry = res["settings"]["cot"]
    assert entry["status"] == "infra_required" and entry["accuracy"] is None
    assert "5,000-token window" in entry["infra_required"] and "[INFRA REQUIRED]" in entry["error"]
    assert res["settings_infra_required"] == ["cot"]
    assert res["status"] == "completed_with_errors"
    assert res["headline_metrics"] == {"non_cot_accuracy": 100.0, "cot_accuracy": None}
    assert res["accuracy"] == 100.0 and res["leaderboard_comparable"] is False
    assert "w/ CoT setting was not run" in res["leaderboard_comparable_reason"]


def test_a_window_no_query_fits_is_infra_required(monkeypatch):
    with pytest.raises(RuntimeError, match="INFRA REQUIRED"):
        _run(monkeypatch, {"non_cot": _RIGHT, "cot": _COT, "answer": _RIGHT}, window=4200)


def test_a_prompt_cut_shorter_only_for_the_cot_query_flags_only_that_setting(monkeypatch):
    # window 120,191: max_len 119,999 for the non-CoT query, 119,103 for the CoT query.
    rows = _rows(1, context="x" * 119500)
    res, _ = _run(monkeypatch, {"non_cot": _RIGHT, "cot": _COT, "answer": _RIGHT},
                  window=120191, rows=rows)
    assert res["settings"]["non_cot"]["longbench_v2_code_report"]["max_prompt_tokens"] == 119999
    assert res["settings"]["non_cot"]["leaderboard_comparable"] is True
    report = res["settings"]["cot"]["longbench_v2_code_report"]
    assert report["max_prompt_tokens"] == 119103
    assert report["prompts_cut_shorter_than_upstream"] == 1
    assert res["leaderboard_comparable"] is False
    assert "middle-truncated to 119,103 tokens" in res["leaderboard_comparable_reason"]
    assert "(w/ CoT)" in res["leaderboard_comparable_reason"]


def test_the_answer_prompt_is_middle_truncated_like_every_pred_py_query():
    meta = {"category": "c", "cot_answer_prompt": "HEAD $COT$ TAIL",
            "prompt_tokens_before_truncation": 50, "prompt_truncated": False}
    trace = {"extra_payload": meta, "response_text": " " + "y" * 100 + " ",
             "gold_answer": "A", "status": "OK", "source_sample_idx": 3}
    samples, no_cot = L._cot_answer_samples([trace], _CharTok(), 20)
    assert not no_cot and len(samples) == 1
    msgs, gold, out = samples[0]
    full = "HEAD " + "y" * 100 + " TAIL"
    # pred.py: input_ids[:max_len//2] + input_ids[-max_len//2:]
    assert msgs[0]["content"] == full[:10] + full[-10:] and gold == "A"
    assert out["prompt_truncated"] is True and out["prompt_tokens_before_truncation"] == len(full)
    assert out["long_prompt_tokens_before_truncation"] == 50
    assert out["cot_query_sample_idx"] == 3 and "cot_answer_prompt" not in out


def test_attempts_give_one_answer_query_per_cot(monkeypatch):
    base.set_run_knobs(attempt_count=2)
    calls = {"n": 0}

    def answer(messages):
        calls["n"] += 1
        return _RIGHT if calls["n"] % 2 else _WRONG
    res, sent = _run(monkeypatch, {"non_cot": _RIGHT, "cot": _COT, "answer": answer})
    kinds = [k for k, _ in sent]
    assert kinds.count("cot") == 4 and kinds.count("answer") == 4
    att = res["settings"]["cot"]["attempts"]
    assert att["attempts_per_sample"] == 2 and att["generations"] == 4
    assert att["avg_at_k"] == 50.0 and att["samples"] == 2
    assert res["settings"]["cot"]["total_questions"] == 4
    cot_traces = [t for t in res["sample_traces"] if t["setting"] == "cot"]
    assert [(t["source_sample_idx"], t["attempt_idx"]) for t in cot_traces] == [
        (0, 0), (0, 1), (1, 0), (1, 1)]


def test_both_settings_see_this_shards_questions(monkeypatch):
    monkeypatch.setenv("GBENCH_SHARD", "2/3")
    res, sent = _run(monkeypatch, {"non_cot": _RIGHT, "cot": _COT, "answer": _RIGHT},
                     rows=_rows(6))
    by_kind = {}
    for k, kw in sent:
        by_kind.setdefault(k, []).append(kw["messages"][0]["content"])
    # rows 1 and 4 (shard 2 of 3, round-robin), in every query
    for k in ("non_cot", "cot", "answer"):
        assert len(by_kind[k]) == 2, k
        assert any("defined 1?" in c for c in by_kind[k]) and any("defined 4?" in c
                                                                 for c in by_kind[k])
    assert res["shard"] == "2/3" and res["settings"]["cot"]["shard"] == "2/3"


# --------------------------------------------------------------------------- #
# docs match the code
# --------------------------------------------------------------------------- #
def test_docs_describe_the_cot_setting():
    doc = " ".join((REPO / "docs" / "evals" / "longbench_v2_code.md").read_text(
        encoding="utf-8").split())
    for needle in ("`prompts/0shot_cot.txt`", "`prompts/0shot_cot_ans.txt`",
                   "`max_tokens=1024`", "`python pred.py --cot`", "121,088", "`settings.cot`",
                   "`headline_metrics`", "`cot_accuracy`", "`settings_infra_required`",
                   "2e00731f8d0bff23dc4325161044d0ed8af94c1e", "`if output == '': continue`",
                   "`questions_without_cot`", "`cot_step: answer`"):
        assert needle in doc, needle
    module = " ".join((L.__doc__ or "").split())
    assert "`python pred.py --cot`" in module and "121,088" in module
    assert doc.isascii()
