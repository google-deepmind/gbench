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

"""Suites that compute their own headline, categories or @k from the traces: a request that got
no reply from the served endpoint is a harness error (base.run_eval_suite marks its trace
`harness_error` + `scoring_excluded`), and every such suite leaves it out of every figure it
computes, counts it and names it, so the headline equals the one the run would report without
that request. A run of nothing but harness errors withholds its score (null + reason). Suites
that send their own requests (MMLU family, WMDP, SciCode, OSWorld) follow the same rule. The
endpoint, Gemini, docker and the network are stubbed."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import mock

import pytest

from gbench.runners.eval_suites import (base, causalbench, chartqa, coco_caption, cruxeval,
                                        docvqa, gpqa, gsm8k, i18n_translate, ifeval,
                                        infographicvqa, lab_bench, lcb, loft_scifact,
                                        matharena_common, medxpertqa, mmlu_pro, mrcr,
                                        multilingual_mmlu, nexusraven_api_eval, scicode, spider2,
                                        t_eval, textvqa, ui_control_osworld, vqa_common)

REPO = Path(__file__).resolve().parent.parent
SUITES = REPO / "gbench" / "runners" / "eval_suites"


# --------------------------------------------------------------------------- #
# A fake endpoint
# --------------------------------------------------------------------------- #
def _text(content: Any) -> str:
    if isinstance(content, list):
        return " ".join(str(p.get("text") or "") for p in content if isinstance(p, dict))
    return str(content or "")


def _serve(reply_for):
    """A fake base._send_single_request: reply_for(conversation text, last user text) gives the
    reply text, or None for a request that gets no reply (HTTP 500 after the retries)."""
    async def _send(**kw):
        msgs = kw["messages"]
        convo = " ".join(_text(m.get("content")) for m in msgs)
        last = next((_text(m.get("content")) for m in reversed(msgs) if m.get("role") == "user"),
                    "")
        reply = reply_for(convo, last)
        if reply is None:
            return base.Reply(text=None, tool_calls=None, error="HTTP 500 server error")
        return base.Reply(text=reply, tool_calls=None, finish_reason="stop", completion_tokens=5)
    return _send


def _by_marker(replies: Dict[str, Optional[str]]):
    def reply_for(convo, _last):
        for marker, reply in replies.items():
            if marker in convo:
                return reply
        raise AssertionError(f"no marker in {convo[:80]!r}")
    return reply_for


def _patched(module, loader: str, samples, reply_for, extra=()):
    patches = [mock.patch.object(base, "_send_single_request", side_effect=_serve(reply_for)),
               mock.patch.object(module, loader, lambda *a, **k: samples)]
    patches += [mock.patch.object(*p) for p in extra]
    return patches


def _run(patches, fn):
    for p in patches:
        p.start()
    try:
        return fn()
    finally:
        for p in reversed(patches):
            p.stop()


# --------------------------------------------------------------------------- #
# One case per base-run suite that recomputes its headline: (run(replies), good, wrong, keys)
# --------------------------------------------------------------------------- #
def _gsm8k(replies):
    samples = [([{"role": "user", "content": f"Question: {m}\nAnswer:"}], "#### 4", {})
               for m in replies]
    return _run(_patched(gsm8k, "_load_gsm8k_samples", samples, _by_marker(replies)),
                lambda: gsm8k.run_gsm8k("m", "http://x/v1", 1, max_output_tokens=256))


def _medxpertqa(replies):
    samples = [([{"role": "system", "content": "sys"}, {"role": "user", "content": f"{m} q"}],
                {"label": "A", "n_options": 5},
                {"category": "Text/Reasoning", "medxpertqa_id": m, "subset": "Text"})
               for m in replies]

    def reply_for(convo, last):
        marker = next(k for k in replies if k in convo)
        if replies[marker] is None:
            return None                                        # the rationale request fails
        return replies[marker] if "Therefore, among" in last else "Some reasoning."
    return _run(_patched(medxpertqa, "_load_medxpertqa_samples", samples, reply_for),
                lambda: medxpertqa.run_medxpertqa("m", "http://x/v1", 1, max_output_tokens=2048))


def _i18n(replies):
    samples = [([{"role": "user", "content": f"Translate {m}"}], "Das ist ein gutes Haus.",
                {"category": "en-de", "segment": i}) for i, m in enumerate(replies)]
    return _run(_patched(i18n_translate, "_load_i18n_translate_samples", samples,
                         _by_marker(replies)),
                lambda: i18n_translate.run_i18n_translate("m", "http://x/v1", 1,
                                                          max_output_tokens=512))


def _vqa(module, loader, gold, run_name):
    def run(replies):
        samples = [([{"role": "user", "content": [{"type": "text", "text": f"{m} what?"}]}],
                    gold, {"category": "human"}) for m in replies]
        return _run(_patched(module, loader, samples, _by_marker(replies)),
                    lambda: getattr(module, run_name)("m", "http://x/v1", 1,
                                                      max_output_tokens=1024))
    return run


def _multilingual_mmlu(replies):
    samples = [([{"role": "user", "content": f"{m} pick"}], "A", {"category": "FR_FR"})
               for m in replies]
    return _run(_patched(multilingual_mmlu, "_load_multilingual_mmlu_samples", samples,
                         _by_marker(replies)),
                lambda: multilingual_mmlu.run_multilingual_mmlu("m", "http://x/v1", 1,
                                                                max_output_tokens=2048))


def _gpqa(replies):
    items = [{"question_idx": i, "repeat_idx": 0, "permutation": [0, 1, 2, 3],
              "sample": ([{"role": "user", "content": f"{m} pick"}], "A",
                         {"category": "Physics"})} for i, m in enumerate(replies)]
    return _run(_patched(gpqa, "_load_gpqa_items", items, _by_marker(replies)),
                lambda: gpqa.run_gpqa("m", "http://x/v1", 1, max_output_tokens=2048))


def _causalbench(replies):
    samples = [([{"role": "user", "content": f"{m} premise"}], gold,
                {"category": "3_vars", "num_variables": 3, "relation": "t"})
               for m, gold in zip(replies, ["entailment", "contradiction", "entailment"])]
    return _run(_patched(causalbench, "_load_causalbench_samples", samples, _by_marker(replies),
                         extra=[(causalbench, "_require_nltk", lambda: None)]),
                lambda: causalbench.run_causalbench("m", "http://x/v1", 1, max_output_tokens=10))


def _lab_bench(replies):
    gold = {"letter": "A", "letters": "ABCD", "refuse": "D", "category": "SeqQA",
            "subtask": "s"}
    samples = [([{"role": "user", "content": f"{m} question"}], gold, {"category": "SeqQA"})
               for m in replies]
    return _run(_patched(lab_bench, "_load_lab_bench_samples", samples, _by_marker(replies)),
                lambda: lab_bench.run_lab_bench("m", "http://x/v1", max_output_tokens=1024))


def _loft(replies):
    samples = [([{"role": "user", "content": f"{m} query"}],
                {"ids": ["3"], "pids": ["p3"], "num_docs": 5}, {"category": "scifact"})
               for m in replies]
    return _run(_patched(loft_scifact, "_load_loft_scifact_samples", samples,
                         _by_marker(replies)),
                lambda: loft_scifact.run_loft_scifact("m", "http://x/v1", 1))


def _ifeval(replies):
    samples = [([{"role": "user", "content": f"{m} write in lowercase"}],
                json.dumps({"key": i, "ids": ["change_case:english_lowercase"], "kwargs": [{}],
                            "prompt": f"{m} write in lowercase"}),
                {"category": "change_case"}) for i, m in enumerate(replies)]
    return _run(_patched(ifeval, "_load_ifeval_samples", samples, _by_marker(replies)),
                lambda: ifeval.run_ifeval("m", "http://x/v1", 1, max_output_tokens=1024))


CORPUS_SEEN: List[List[Any]] = []


def _fake_caption_metrics(gts, res):
    CORPUS_SEEN.append(sorted(gts))
    per_image = {iid: (1.0 if res[iid] == gts[iid][0] else 0.0) for iid in gts}
    cider = sum(per_image.values()) / len(per_image)
    return {name: cider for name in coco_caption.METRICS}, per_image


def _coco(replies):
    samples = [([{"role": "user", "content": [{"type": "text", "text": f"{m} caption"}]}],
                ["a cat on a mat"], {"image_id": i, "category": "coco"})
               for i, m in enumerate(replies)]
    return _run(_patched(coco_caption, "_load_coco_caption_samples", samples,
                         _by_marker(replies),
                         extra=[(coco_caption, "check_coco_caption_prerequisites",
                                 lambda: (True, "")),
                                (coco_caption, "compute_caption_metrics",
                                 _fake_caption_metrics)]),
                lambda: coco_caption.run_coco_caption("m", "http://x/v1", 1,
                                                      max_output_tokens=1024))


CASES = {
    "gsm8k": (_gsm8k, "so The answer is 4.", "The answer is 5.",
              ["accuracy", "flexible_extract_accuracy", "gsm8k_cot_report"]),
    "medxpertqa": (_medxpertqa, "A", "B",
                   ["accuracy", "category_accuracy", "medxpertqa_text_accuracy"]),
    "i18n_translate": (_i18n, "Das ist ein gutes Haus.", "Hallo Welt.",
                       ["accuracy", "i18n_bleu", "i18n_chrf", "mean_chrf",
                        "pass_rate_chrf40"]),
    "chartqa": (_vqa(chartqa, "_load_chartqa_samples", "4", "run_chartqa"), "4", "7",
                ["accuracy", "category_accuracy"]),
    "docvqa": (_vqa(docvqa, "_load_docvqa_samples", ["four"], "run_docvqa"), "four",
               "fxyz", ["accuracy", "pass_rate_at_0.5"]),
    "infographicvqa": (_vqa(infographicvqa, "_load_infographicvqa_samples", ["four"],
                            "run_infographicvqa"), "four", "fxyz",
                       ["accuracy", "pass_rate_at_0.5"]),
    "textvqa": (_vqa(textvqa, "_load_textvqa_samples", ["four"] * 10, "run_textvqa"), "four",
                "seven", ["accuracy", "binary_pass_rate"]),
    "multilingual_mmlu": (_multilingual_mmlu, "Answer: A", "Answer: B",
                          ["accuracy", "category_accuracy"]),
    "gpqa": (_gpqa, "Answer: A", "Answer: B", ["accuracy", "category_accuracy"]),
    "causalbench": (_causalbench, "entailment", "entailment",
                    ["accuracy", "raw_accuracy", "causalbench_report"]),
    "lab_bench": (_lab_bench, "[ANSWER]A[/ANSWER]", "[ANSWER]B[/ANSWER]",
                  ["accuracy", "lab_bench_report"]),
    "loft_scifact": (_loft, "Final Answer: ['3']", "Final Answer: ['1']", ["accuracy"]),
    "ifeval": (_ifeval, "all of this is lowercase.", "THIS IS NOT.",
               ["accuracy", "ifeval_report"]),
    "coco_caption": (_coco, "a cat on a mat", "a dog", ["accuracy", "per_image_pass_rate"]),
}


def _headline(res, keys):
    return {k: res.get(k) for k in keys}


@pytest.mark.parametrize("suite", sorted(CASES))
def test_an_endpoint_failure_leaves_every_figure_as_without_it(suite):
    run, good, wrong, keys = CASES[suite]
    clean = run({"MKa": good, "MKb": wrong})
    with_failure = run({"MKa": good, "MKb": wrong, "MKc": None})
    assert clean["status"] == "success", (suite, clean.get("status_reason"))
    assert _headline(with_failure, keys) == _headline(clean, keys), suite
    assert with_failure["status"] == "completed_with_errors", suite
    assert "harness errors left out of the score" in with_failure["status_reason"], suite
    failed = [t for t in with_failure["sample_traces"] if t.get("harness_error")]
    assert failed and all(t["scoring_excluded"] for t in failed), suite
    assert failed[0]["harness_error"]["kind"] == "endpoint_failure"
    reason = with_failure.get("leaderboard_comparable_reason") or ""
    assert "no reply" in reason and "harness error" in reason, (suite, reason)
    assert "scored as wrong" not in reason and "count as wrong" not in reason.split("empty")[0]


@pytest.mark.parametrize("suite", sorted(CASES))
def test_a_run_of_only_endpoint_failures_withholds_the_score(suite):
    run, _good, _wrong, _keys = CASES[suite]
    res = run({"MKa": None, "MKb": None})
    assert res["accuracy"] is None, suite
    assert res.get("accuracy_withheld_reason"), suite
    assert res["status"] == "failed", suite


def test_coco_caption_leaves_the_failed_image_out_of_the_corpus():
    CORPUS_SEEN.clear()
    _coco({"MKa": "a cat on a mat", "MKb": "a dog", "MKc": None})
    assert CORPUS_SEEN == [[0, 1]]                   # image 2 got no reply: not in the corpus


# --------------------------------------------------------------------------- #
# Every base-run suite that recomputes its headline filters the harness errors
# --------------------------------------------------------------------------- #
_EXCLUSION = re.compile(r"\b(is_scored|scoring_excluded|scorable_traces|finalize_mean_metric|"
                        r"tally_batch_scores|harness_error)\b")
#: Suites whose own headline is built only from base's already-filtered numbers.
_BUILT_FROM_BASE = {
    "bfcl_live.py": "Live Overall Acc from base's category_accuracy (tally_batch_scores leaves "
                    "the harness errors out of it)",
}
_NOT_SUITES = {"base.py", "metrics.py", "swe_bench_pro_agentic.py", "jetski_agent.py",
               "slashbin_agent.py"}


def test_every_base_run_suite_that_sets_its_headline_leaves_harness_errors_out():
    offenders = []
    for path in sorted(SUITES.glob("*.py")):
        if path.name in _NOT_SUITES or path.name in _BUILT_FROM_BASE:
            continue
        src = path.read_text(encoding="utf-8")
        if "run_eval_suite(" not in src:
            continue
        if not re.search(r'\["accuracy"\]\s*=|"accuracy":\s', src):
            continue
        if not _EXCLUSION.search(src):
            offenders.append(path.name)
    assert offenders == []
    for name in _BUILT_FROM_BASE:
        assert (SUITES / name).exists(), name


# --------------------------------------------------------------------------- #
# vqa_common: the no-reply marking and status for suites that send their own requests
# --------------------------------------------------------------------------- #
def test_mark_no_reply_marks_as_base_does():
    t = {"status": "OK", "is_correct": True}
    vqa_common.mark_no_reply(t, "TIMEOUT", "timeout after 30s")
    assert t["status"] == "TIMEOUT" and t["is_correct"] is False and t["scoring_excluded"]
    assert t["harness_error"] == base.endpoint_harness_error("TIMEOUT", "timeout after 30s")
    assert vqa_common.lost_requests([t]) == 1


def test_own_loop_status_follows_base_rule():
    ok = {"status": "OK"}
    lost = {"status": "FAILED"}
    vqa_common.mark_no_reply(lost, "FAILED", "HTTP 500")
    assert vqa_common.own_loop_status([]) == ("error", "no samples were run", {})
    assert vqa_common.own_loop_status([ok]) == ("success", None, {})
    status, reason, kinds = vqa_common.own_loop_status([ok, lost])
    assert status == "completed_with_errors" and kinds == {"endpoint_failure": 1}
    assert reason.startswith("harness errors left out of the score: 1 request(s)")
    status, reason, _ = vqa_common.own_loop_status([lost])
    assert status == "failed" and reason.startswith("nothing was scored: harness errors")
    status, reason, _ = vqa_common.own_loop_status([ok], ["empty_responses 1"])
    assert status == "completed_with_errors" and reason == "empty_responses 1"


def test_lost_requests_reason_says_harness_errors_left_out():
    t = {"status": "OK", "response_text": "x",
         "harness_error": {"kind": "endpoint_timeout", "detail": "t"}, "scoring_excluded": True}
    assert vqa_common.lost_requests_reason([t]) == (
        "1 request(s) got no reply (1 timed out) (harness errors, left out of the score)")


# --------------------------------------------------------------------------- #
# mrcr, ifeval: the breakdowns next to the headline
# --------------------------------------------------------------------------- #
def test_mrcr_bins_leave_out_the_requests_with_no_reply():
    traces = [{"category": "2/4K", "mrcr_grade": 0.5, "response_text": "x"},
              {"category": "2/4K", "response_text": None, "scoring_excluded": True,
               "harness_error": {"kind": "endpoint_failure", "detail": "HTTP 500"}},
              {"category": "2/4K", "mrcr_grade": 0.0, "response_text": ""}]
    out = mrcr._mrcr_breakdown(traces)
    assert out["mrcr_bin_counts"] == {"2": {"4K": 2}}
    assert out["mrcr_bins"] == {"2": {"4K": 25.0}}


def test_ifeval_report_is_over_the_scored_prompts():
    gold = json.dumps({"key": 1, "ids": ["change_case:english_lowercase"], "kwargs": [{}],
                       "prompt": "write in lowercase"})
    traces = [{"gold_answer": gold, "response_text": "all lowercase."},
              {"gold_answer": gold, "response_text": None, "scoring_excluded": True}]
    rep = ifeval._ifeval_report(traces)
    assert rep["prompts"] == 1 and rep["prompt_level_strict"] == 100.0
    assert ifeval._ifeval_report(traces[1:])["prompt_level_strict"] is None


# --------------------------------------------------------------------------- #
# t_eval: the evaluators never see a request with no reply
# --------------------------------------------------------------------------- #
def _t_eval_run(monkeypatch, failing_files):
    seen = []

    def fake_score(dim, traces, language="en"):
        seen.extend(traces)
        return {}

    def fake_run(**kw):
        (msgs, gold, meta), = kw["samples"]
        stem = msgs[0]["content"]
        failed = stem in failing_files
        trace = {"sample_idx": 0, "gold_answer": gold, "category": meta["category"],
                 "response_text": None if failed else "reply",
                 "status": "FAILED" if failed else "OK", "is_correct": False,
                 "finish_reason": None if failed else "stop", "max_tokens_sent": 512}
        if failed:
            trace.update(harness_error={"kind": "endpoint_failure", "detail": "HTTP 500"},
                         scoring_excluded=True)
        return {"sample_traces": [trace], "total_questions": 1,
                "failed_requests": int(failed), "scoring_excluded": int(failed),
                "request_harness_errors": {"endpoint_failure": 1} if failed else {},
                "status": "completed_with_errors" if failed else "success",
                "status_reason": ("harness errors left out of the score: 1 request(s)"
                                  if failed else None),
                "temperature": 0.0, "official_sampling": {}}
    monkeypatch.setattr(t_eval, "_load_file_samples",
                        lambda dim, stem, limit=None: [([{"role": "user", "content": stem}],
                                                        json.dumps({"dimension": dim}),
                                                        {"category": f"t_eval::{dim}"})])
    monkeypatch.setattr(t_eval, "run_eval_suite", fake_run)
    monkeypatch.setattr(t_eval, "_score_file", fake_score)
    return t_eval.run_t_eval("m", "http://x/v1", 1), seen


def test_t_eval_request_with_no_reply_is_not_scored_and_withholds_its_setting(monkeypatch):
    res, seen = _t_eval_run(monkeypatch, {"plan_json_v2"})
    assert seen and all(t["response_text"] is not None for t in seen)
    assert res["accuracy"] is None and "plan_json" in res["accuracy_withheld_reason"]
    assert res["overall_zh"] is not None                 # the Chinese files all scored
    assert res["t_eval_file_metrics"]["plan_json"] is None
    assert res["status"] == "completed_with_errors"
    assert res["status_reason"].startswith("plan_json: harness errors left out")
    assert res["scoring_excluded"] == 1
    assert res["request_harness_errors"] == {"endpoint_failure": 1}
    assert "1 request(s) got no reply from the served endpoint" in \
        res["leaderboard_comparable_reason"]


def test_t_eval_clean_run_scores_every_file(monkeypatch):
    res, seen = _t_eval_run(monkeypatch, set())
    assert len(seen) == 16 and res["accuracy"] is not None
    assert res["status"] == "success" and res["status_reason"] is None


# --------------------------------------------------------------------------- #
# nexusraven: a repair call with no reply is a harness error
# --------------------------------------------------------------------------- #
def test_nexusraven_repair_call_with_no_reply_is_left_out(monkeypatch):
    calls = []

    def fake_run(**kw):
        calls.append(kw)
        if len(calls) == 1:
            traces = [{"sample_idx": i, "gold_answer": gold, "response_text": text,
                       "is_correct": False, "status": "OK"}
                      for i, (gold, text) in enumerate([
                          (json.dumps({"task": "toolalpaca", "key": 0}), "good"),
                          (json.dumps({"task": "cve_cpe", "key": 1}), "bad")])]
            return {"sample_traces": traces, "status": "success", "status_reason": None,
                    "accuracy": 0.0, "official_sampling": {}, "scoring_excluded": 0}
        fixed = {"sample_idx": 0, "response_text": None, "is_correct": False,
                 "status": "FAILED", "scoring_excluded": True,
                 "harness_error": {"kind": "endpoint_failure", "detail": "HTTP 500"}}
        return {"sample_traces": [fixed], "failed_requests": 1}
    monkeypatch.setattr(nexusraven_api_eval, "_load_nexusraven_api_eval_samples",
                        lambda limit=None: [])
    monkeypatch.setattr(nexusraven_api_eval, "run_eval_suite", fake_run)
    monkeypatch.setattr(nexusraven_api_eval, "score",
                        lambda task, key, text: (text == "good", text == "bad"))
    res = nexusraven_api_eval.run_nexusraven_api_eval("m", "http://x/v1", 1)
    assert len(calls) == 2
    bad = res["sample_traces"][1]
    assert bad["scoring_excluded"] and bad["harness_error"]["detail"].startswith(
        "output-fixing call: HTTP 500")
    assert res["output_fixing"]["harness_errors"] == 1
    assert res["per_task"] == {"toolalpaca": {"n": 1, "accuracy": 100.0}}
    # The Cybersecurity domain ran but has no scored query: no mean of the two domains.
    assert res["accuracy"] is None and "Cybersecurity Domain" in res["accuracy_withheld_reason"]
    assert res["status"] == "completed_with_errors" and res["scoring_excluded"] == 1
    assert "output-fixing call(s) that got no reply" in res["status_reason"]
    assert res["effective_n"] == 1


# --------------------------------------------------------------------------- #
# spider2: a grader crash is a harness error
# --------------------------------------------------------------------------- #
def test_spider2_grader_crash_and_no_reply_are_left_out(monkeypatch, tmp_path):
    (tmp_path / "db1.sqlite").write_bytes(b"")
    monkeypatch.setattr(spider2, "_get_sqlite_result", lambda path, sql: object())

    def crash(*a, **k):
        raise OSError("gold CSV unreadable")
    monkeypatch.setattr(spider2, "_score_against_golds", crash)
    ok = {"response_text": "```sql\nSELECT 1\n```", "extra_payload": {"db": "db1"},
          "status": "PENDING_JUDGE"}
    lost = {"response_text": None, "extra_payload": {"db": "db1"}, "status": "FAILED",
            "scoring_excluded": True, "harness_error": {"kind": "endpoint_failure",
                                                         "detail": "HTTP 500"}}
    asyncio.run(spider2._make_scorer(str(tmp_path))([ok, lost]))
    assert ok["status"] == "SCORING_ERROR" and ok["scoring_excluded"] is True
    assert ok["harness_error"]["kind"] == "grader_error"
    assert lost["status"] == "FAILED"                   # not graded


# --------------------------------------------------------------------------- #
# cruxeval: a task with no scored generation withholds the mean
# --------------------------------------------------------------------------- #
def test_cruxeval_withholds_the_mean_when_a_task_has_no_scored_generation(monkeypatch):
    monkeypatch.setattr(cruxeval, "sandbox_skip_reason", lambda: None)
    monkeypatch.setattr(cruxeval, "_grader_self_check", lambda: None)
    monkeypatch.setattr(cruxeval, "_load_cruxeval_samples", lambda **k: [None, None])

    def fake_run(**kw):
        out = {"sample_idx": 0, "source_sample_idx": 0, "gold_answer": {"mode": "output"},
               "cruxeval_check": "passed", "is_correct": True, "status": "OK"}
        inp = {"sample_idx": 1, "source_sample_idx": 1, "gold_answer": {"mode": "input"},
               "response_text": None, "is_correct": False, "status": "FAILED",
               "scoring_excluded": True,
               "harness_error": {"kind": "endpoint_failure", "detail": "HTTP 500"}}
        return {"sample_traces": [out, inp], "status": "completed_with_errors",
                "status_reason": "harness errors left out of the score: 1 request(s)",
                "attempts": {"attempts_per_sample": 10}, "temperature": 0.2, "accuracy": 100.0}
    monkeypatch.setattr(cruxeval, "run_eval_suite", fake_run)
    res = cruxeval.run_cruxeval("m", "http://x/v1", 1)
    assert "CRUXEval-I" not in res["cruxeval_report"]
    assert res["accuracy"] is None and "CRUXEval-I" in res["accuracy_withheld_reason"]
    assert "pass@5 run: harness errors left out" in res["status_reason"]


# --------------------------------------------------------------------------- #
# matharena / lcb: a follow-up request that does not fit the window is an over_context error
# --------------------------------------------------------------------------- #
def test_matharena_last_chance_over_the_window_is_an_over_context_harness_error(monkeypatch):
    async def oversize(convo, budget, tools, ctk):
        return None, "prompt is ~9 tokens, which does not fit"
    monkeypatch.setattr(base, "clamp_to_context_async", oversize)
    drives = []

    def fake_driver(mode, items, config, **kw):
        drives.append(mode)
        return [{"valid": False}] * len(items) if mode == "check" else []
    monkeypatch.setattr(matharena_common, "run_driver", fake_driver)
    trace = {"messages": [{"role": "user", "content": "p"}], "response_text": "no box",
             "gold_answer": "5", "max_tokens_sent": 100}
    stats: Dict[str, Any] = {}
    scorer = matharena_common.make_scorer(
        None, {}, stats, eval_name="hmmt", model_name="m", base_url="http://x/v1",
        concurrency=1, thinking=False, temperature=1.0, extra_payload=None)
    asyncio.run(scorer([trace]))
    assert trace["harness_error"]["kind"] == "over_context" and trace["scoring_excluded"]
    assert stats["last_chance_failures"] == 1 and trace["status"] == "LAST_CHANCE_FAILED"


def test_lcb_self_repair_request_over_the_window_is_an_over_context_error(monkeypatch):
    async def oversize(messages, budget, tools, ctk):
        return None, "prompt is ~9 tokens, which does not fit"
    monkeypatch.setattr(base, "clamp_to_context_async", oversize)
    (reply, sent), = asyncio.run(lcb._send_self_repair_requests(
        [[{"role": "user", "content": "p"}]], model_name="m", base_url="http://x/v1",
        concurrency=1, thinking=False, temperature=0.2, payload={}, budget=100))
    assert reply.text is None and reply.error.startswith(base.OVER_CONTEXT_ERROR)
    assert sent is None


# --------------------------------------------------------------------------- #
# Suites that send their own requests
# --------------------------------------------------------------------------- #
def test_mmlu_pro_run_of_only_failed_questions_withholds_the_score():
    sample = ([{"role": "user", "content": "p"}], "A", {"category": "math", "row": 0})

    async def fake_harness(*args, **kw):
        return [{"category": "math", "is_correct": False, "predicted": None,
                 "response_text": None, "status": "FAILED", "error": "APIConnectionError"}]
    with mock.patch.object(mmlu_pro, "_load_mmlu_pro_samples", return_value=[sample]), \
            mock.patch.object(mmlu_pro, "_run_harness", side_effect=fake_harness):
        res = mmlu_pro.run_mmlu_pro("m", "http://x/v1", 1)
    assert res["accuracy"] is None and res["status"] == "failed"
    assert res["accuracy_withheld_reason"].startswith("nothing was scored: harness errors")
    assert res["scoring_excluded"] == 1 and res["effective_n"] == 0
    assert res["category_accuracy"] == {}
    assert res["sample_traces"][0]["harness_error"]["kind"] == "endpoint_failure"


def _sc_problem(pid, n):
    return {"problem_id": pid, "problem_name": f"p{pid}", "required_dependencies": "",
            "problem_description_main": "d", "problem_io": "",
            "sub_steps": [{"step_number": f"{pid}.{k + 1}", "step_description_prompt": "d",
                           "step_background": "", "function_header": f"def f{k}():",
                           "return_line": "    return 1", "test_cases": ["assert True"]}
                          for k in range(n)]}


def _sc_run(monkeypatch, problems, fail_steps):
    monkeypatch.setattr(scicode, "sandbox_skip_reason", lambda: None)
    monkeypatch.setattr(scicode, "resolve_test_data", lambda: ("/nonexistent.h5", "test"))
    monkeypatch.setattr(scicode, "_digest", lambda path: scicode._TEST_DATA_SHA256)
    monkeypatch.setattr(scicode, "_load_scicode_problems", lambda limit=None: problems)
    monkeypatch.setenv("GBENCH_SCICODE_TEMPERATURE", "0")
    monkeypatch.delenv("GBENCH_SCICODE_WITH_BACKGROUND", raising=False)
    asked = []

    def fake_complete(client, model, prompt, temperature, max_tokens, thinking, retries=5):
        asked.append(prompt)
        if any(k in prompt for k in fail_steps):
            return {"text": None, "error": "APIConnectionError: down", "timed_out": False}
        return {"text": "```python\ndef g():\n    return 1\n```", "reasoning": "",
                "finish_reason": "stop", "completion_tokens": 5}
    monkeypatch.setattr(scicode, "_complete", fake_complete)
    monkeypatch.setattr(scicode, "_score_step", lambda *a, **k: {
        "correct": True, "returncode": 0, "stdout_tail": "", "stderr_tail": ""})
    return scicode.run_scicode("m", "http://h/v1", concurrency=1), asked


def test_scicode_generation_with_no_reply_leaves_its_problem_out(monkeypatch):
    problems = [_sc_problem("1", 3), _sc_problem("2", 2)]
    marker = "FAILME"
    problems[0]["sub_steps"][1]["step_description_prompt"] = marker
    res, asked = _sc_run(monkeypatch, problems, [marker])
    assert len(asked) == 4                  # 1.1, 1.2 (no reply); 1.3 is not generated; 2.x
    assert res["total_problems"] == 1 and res["correct_problems"] == 1
    assert res["accuracy"] == 100.0 and res["effective_n"] == 1 and res["total_questions"] == 2
    assert res["total_steps"] == 3 and res["substep_accuracy"] == 100.0
    assert res["failed_generations"] == 1 and res["steps_not_generated"] == 1
    assert res["harness_error_problems"] == 1 and res["scoring_excluded"] == 2
    assert res["status"] == "completed_with_errors"
    assert "harness errors left out of the score: 1 problem(s)" in res["status_reason"]
    by_step = {t["step_number"]: t for t in res["sample_traces"]}
    assert by_step["1.2"]["harness_error"]["kind"] == "endpoint_failure"
    assert by_step["1.3"]["harness_error"]["kind"] == "not_generated"
    assert "harness errors" in res["leaderboard_comparable_reason"]


def test_scicode_run_of_only_generation_failures_withholds_the_score(monkeypatch):
    problems = [_sc_problem("1", 2)]
    problems[0]["sub_steps"][0]["step_description_prompt"] = "FAILME"
    res, _ = _sc_run(monkeypatch, problems, ["FAILME"])
    assert res["accuracy"] is None and res["status"] == "failed"
    assert res["accuracy_withheld_reason"].startswith("nothing was scored")


def test_osworld_headline_is_over_the_tasks_with_a_result():
    scored = ui_control_osworld.compute_ui_control_osworld_score(
        {"success_rate": 0.25, "n_selected": 4, "n_scored": 2,
         "per_domain": {"os": {"success_rate": 0.5, "n": 2, "n_scored": 1}}})
    assert scored["success_rate"] == pytest.approx(0.5)
    assert scored["success_rate_missing_as_zero"] == 0.25
    assert scored["per_domain"]["os"]["success_rate"] == pytest.approx(1.0)


def _osworld(monkeypatch, tmp_path, summary):
    monkeypatch.setenv("GBENCH_OSWORLD_VM_DIR", str(tmp_path))
    monkeypatch.setattr(ui_control_osworld, "check_ui_control_osworld_prerequisites",
                        lambda: (True, ""))

    def fake_run(cmd, *a, **k):
        if cmd[:2] == ["docker", "run"]:
            # the orchestrator writes its summary into this run's own directory
            run_dir = next(c.split("=", 1)[1] for c in cmd
                           if c.startswith("OSWORLD_RUN_DIR="))
            (Path(run_dir) / "ui_control_osworld_summary.json").write_text(
                json.dumps(summary), encoding="utf-8")
        return mock.MagicMock(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(ui_control_osworld.subprocess, "run", fake_run)
    return ui_control_osworld.run_ui_control_osworld(model_name="m",
                                                     base_url="http://127.0.0.1:1/v1")


def test_osworld_missing_tasks_are_harness_errors(monkeypatch, tmp_path):
    res = _osworld(monkeypatch, tmp_path, {"success_rate": 0.25, "per_domain": {},
                                           "n_selected": 4, "n_scored": 2, "n_missing": 2})
    assert res["accuracy"] == 50.0 and res["success_rate_missing_as_zero"] == 0.25
    assert res["status"] == "completed_with_errors" and res["scoring_excluded"] == 2
    assert res["effective_n"] == 2 and "2 of 4 task(s)" in res["status_reason"]
    assert "harness errors, left out of the score" in res["leaderboard_comparable_reason"]
    res = _osworld(monkeypatch, tmp_path, {"success_rate": 0.0, "per_domain": {},
                                           "n_selected": 3, "n_scored": 0, "n_missing": 3})
    assert res["accuracy"] is None and res["status"] == "failed"
    assert res["accuracy_withheld_reason"].startswith("nothing was scored")


# --------------------------------------------------------------------------- #
# Docs say what the code does
# --------------------------------------------------------------------------- #
def _doc(name):
    return " ".join((REPO / "docs" / "evals" / f"{name}.md").read_text(encoding="utf-8").split())


_DOCS = ["gsm8k", "medxpertqa", "i18n_translate", "chartqa", "docvqa", "infographicvqa",
         "textvqa", "coco_caption", "gpqa", "gpqa_diamond", "multilingual_mmlu", "aime", "hmmt",
         "arc_agi", "bundled_detection", "causalbench", "cruxeval", "lab_bench", "lcb",
         "longbench_v2_code", "loft_scifact", "mmmu_pro", "ruler", "screenspot",
         "screenspot_pointing", "ifeval", "mrcr", "t_eval", "nexusraven_api_eval", "spider2",
         "mmlu", "mmlu_pro", "mmlu_redux", "wmdp", "scicode", "ui_control_osworld"]


@pytest.mark.parametrize("name", _DOCS)
def test_docs_call_a_request_with_no_reply_a_harness_error(name):
    doc = _doc(name)
    assert "harness error" in doc, name
    for stale in ("scored as wrong", "a failed request is scored", "failed request is scored",
                  "counted as 0", "counts as 0", "scored as a miss",
                  "scored as images with no detection", "scored as empty translations",
                  "could not be scored counts as wrong", "no reply counts as wrong",
                  "stays in both denominators", "(scored 0, run non-comparable)"):
        assert stale not in doc, (name, stale)
