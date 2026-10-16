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

"""gdpval: attachments in, PAIRWISE WIN-RATE vs the expert reference out.

The canonical GDPval metric is a blind pairwise comparison of the model's deliverable against
the human-expert reference (`deliverable_files`), reported as a win-rate. This suite reproduces
that: the reference file is rendered to text, the model answer and reference are shown to the
judge as A/B, the comparison is position-swapped to cancel order bias, and the headline is the
win-rate. It is not OpenAI's number (a text endpoint cannot emit the file), and says so.

Nothing here touches the network or the judge API.
"""

import asyncio
import json
import os
from unittest import mock

import pytest

from gbench.runners.eval_suites import gdpval as G


# --------------------------------------------------------------------------- #
# attachments (input + reference rendering)
# --------------------------------------------------------------------------- #
def test_an_unrenderable_attachment_produces_a_note_never_silence():
    """5 of 261 inputs are .mp4/.step/.psd. A task whose input vanished without a word
    would be scored as though the model simply failed."""
    with mock.patch.object(G, "_fetch", return_value="/tmp/x.step"):
        text, note = G.render_attachment("reference_files/a/model.step")
    assert text is None
    assert note and "model.step" in note and "unavailable" in note


def test_a_failed_download_is_reported_not_swallowed():
    """A failed download is missing infra: a hard error naming the file, not a prompt note."""
    with mock.patch("huggingface_hub.hf_hub_download", side_effect=OSError("offline")):
        with pytest.raises(RuntimeError, match="INFRA REQUIRED.*Book.xlsx"):
            G.render_attachment("reference_files/a/Book.xlsx")


def test_a_corrupt_file_is_a_hard_error_not_a_silent_exclusion():
    with mock.patch.object(G, "_fetch", return_value="/dev/null"):
        with pytest.raises(RuntimeError, match="could not read 'Book.xlsx'"):
            G.render_attachment("reference_files/a/Book.xlsx")


def test_oversized_attachments_are_clipped():
    """A single spreadsheet can otherwise consume the whole context window."""
    big = "x" * (G._MAX_CHARS_PER_FILE + 5000)
    out = G._clip(big, "Huge.xlsx")
    assert len(out) < len(big) and "truncated" in out


def test_unavailable_attachments_are_counted_for_the_result():
    with mock.patch.object(G, "_fetch", return_value=None):
        parts, rendered, unavailable = G.attachment_parts(["reference_files/a/b.mp4"])
    assert rendered == 0 and unavailable == 1
    assert any("unavailable" in p.get("text", "") or "could not" in p.get("text", "")
               for p in parts)


def test_reference_render_of_a_text_file_returns_content():
    # The rendered reference carries no label that marks it as the expert's (blind judge).
    with mock.patch.object(G, "render_attachment", return_value=("A,B\n1,2\n", None)):
        text, note = G.render_reference(["deliverable_files/a/Sample.xlsx"])
    assert text == "A,B\n1,2\n" and "reference" not in text.lower()
    assert note is None


def test_reference_render_with_no_files_is_a_note_not_a_crash():
    text, note = G.render_reference([])
    assert text is None and "no reference deliverable" in note


def test_reference_render_of_an_image_only_reference_is_excluded():
    """An image reference cannot be compared to a text answer -> note, and (upstream) the
    task is excluded from the win-rate rather than scored as a loss."""
    text, note = G.render_reference(["deliverable_files/a/chart.png"])
    assert text is None and "image" in note.lower()


# --------------------------------------------------------------------------- #
# the loader
# --------------------------------------------------------------------------- #
_ROW = {
    "task_id": "t1", "prompt": "Do the thing.", "occupation": "Accountants",
    "sector": "Prof", "reference_files": ["reference_files/a/Book.xlsx"],
    "deliverable_files": [],
    "rubric_json": json.dumps([
        {"score": 2, "criterion": "The deliverable is an .xlsx workbook"},
        {"score": 3, "criterion": "The analysis identifies the largest variance"},
    ]),
}


def _load(rows, **kw):
    with mock.patch("datasets.load_dataset", return_value=rows), \
         mock.patch.object(G, "attachment_parts",
                           return_value=([{"type": "text", "text": "\n--- attachment ---\nA,B\n"}], 1, 0)):
        return G._load_gdpval_samples(**kw)


def test_loader_returns_samples_and_a_reference_map():
    samples, references = _load([_ROW])
    assert isinstance(samples, list) and isinstance(references, dict)
    assert "t1" in references


def test_loader_attaches_the_reference_files():
    samples, _ = _load([_ROW])
    content = samples[0][0][0]["content"]
    assert isinstance(content, list), "attachments must reach the model as content parts"
    assert "attachment" in content[1]["text"]
    assert samples[0][2]["attachments"] == 1
    assert samples[0][2]["attachments_rendered"] == 1


def test_reference_text_is_not_put_in_sample_metadata():
    """base.run_eval_suite merges sample metadata into the request payload sent to the model,
    so the (potentially 20 KB) reference deliverable must stay on the side channel, never in
    the per-sample metadata dict."""
    with mock.patch.object(G, "render_reference", return_value=("A,B\n1,2", None)):
        samples, references = _load([dict(_ROW, deliverable_files=["deliverable_files/a/Sample.xlsx"])])
    assert references["t1"]["text"] == "A,B\n1,2"
    meta = samples[0][2]
    assert "reference_text" not in meta and "reference" not in json.dumps(meta).lower()


def test_a_task_with_no_attachments_stays_plain_text():
    row = dict(_ROW, reference_files=[])
    samples, _ = _load([row])
    assert samples[0][0][0]["content"] == "Do the thing."


def test_gold_carries_the_structured_rubric_not_prose():
    """The pairwise judge grades against the rubric; a flattened string cannot guide it well."""
    samples, _ = _load([_ROW])
    crits = json.loads(samples[0][1])
    assert isinstance(crits, list) and crits[0]["score"] == 2


def test_loader_refuses_a_task_with_no_rubric():
    row = dict(_ROW, rubric_json="[]")
    with pytest.raises(RuntimeError, match="empty rubric"):
        _load([row])


def test_loader_refuses_a_bad_schema():
    with pytest.raises(RuntimeError, match="refusing to fabricate"):
        _load([{"prompt": "x"}])


# --------------------------------------------------------------------------- #
# the old rubric-fraction / keyword scorers are gone
# --------------------------------------------------------------------------- #
def test_the_keyword_overlap_scorer_is_deleted():
    """`_eval_gdpval` passed a criterion when 70% of its 4+ character words appeared
    anywhere in the response - it credited a model for restating the rubric."""
    assert not hasattr(G, "_eval_gdpval"), "the fake-score heuristic must not come back"


def test_the_rubric_fraction_scorer_is_replaced_by_pairwise():
    """The headline is now the canonical pairwise win-rate, not a rubric-point fraction."""
    assert not hasattr(G, "_JUDGE_PROMPT"), "the per-criterion rubric prompt is retired"
    assert hasattr(G, "_PAIRWISE_PROMPT")


# --------------------------------------------------------------------------- #
# the pairwise prompt + verdict scoring
# --------------------------------------------------------------------------- #
def test_the_pairwise_prompt_is_blind_and_warns_against_order_bias():
    p = G._PAIRWISE_PROMPT
    for tok in ("{prompt}", "{rubric}", "{a}", "{b}", "DELIVERABLE A", "DELIVERABLE B"):
        assert tok in p
    assert "order is arbitrary" in p.lower()
    # the prompt must not reveal which candidate is the model vs the expert reference
    lo = p.lower()
    for tell in ("the reference", "reference deliverable", "human expert", "the model", "candidate a is"):
        assert tell not in lo, f"pairwise prompt leaks candidate identity: {tell!r}"


def test_model_score_win_tie_loss_and_unparseable():
    # model in slot A
    assert G._model_score("a", "a") == 1.0
    assert G._model_score("b", "a") == 0.0
    assert G._model_score("tie", "a") == 0.5
    # model in slot B
    assert G._model_score("b", "b") == 1.0
    assert G._model_score("a", "b") == 0.0
    # an unparseable / unexpected winner never reaches scoring (it is a judge outage, not a tie)
    with pytest.raises(ValueError):
        G._model_score("", "a")
    with pytest.raises(ValueError):
        G._model_score("garbage", "b")


def test_task_prompt_extracts_text_from_string_or_multimodal():
    assert G._task_prompt({"messages": [{"role": "user", "content": "hello"}]}) == "hello"
    mm = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "the task"}, {"type": "image_url", "image_url": {"url": "x"}}]}]}
    assert G._task_prompt(mm) == "the task"
    assert G._task_prompt({}) == ""


# --------------------------------------------------------------------------- #
# the scorer end to end (fake judge, no network)
# --------------------------------------------------------------------------- #
_MARK = "__MODEL_DELIVERABLE__"


def _winner_by_marker(prompt: str) -> str:
    """Return the A/B slot whose deliverable contains the model marker (model 'wins')."""
    a = prompt.split("# DELIVERABLE A", 1)[1].split("# DELIVERABLE B", 1)[0]
    b = prompt.split("# DELIVERABLE B", 1)[1]
    if _MARK in a:
        return "A"
    if _MARK in b:
        return "B"
    return "tie"


def _trace(task_id="t1", response=None, rubric=None):
    return {
        "extra_payload": {"task_id": task_id},
        "gold_answer": json.dumps(rubric or [{"score": 5, "criterion": "identifies the variance"}]),
        "response_text": response if response is not None else f"My analysis {_MARK}.",
        "messages": [{"role": "user", "content": "Do the audit."}],
    }


def _score(traces, references, judge):
    """Run the async scorer with a fake judge. `judge(prompt)->winner|None`."""
    metrics = {}

    async def fake_cascade(prompt):
        w = judge(prompt)
        if w is None:
            return None, None
        return json.dumps({"winner": w, "reason": "test"}), "gemini-test"

    with mock.patch.object(G, "judge_generate_cascade", fake_cascade):
        asyncio.run(G._make_scorer("j", 4, metrics, references)(traces))
    return metrics


def test_model_wins_both_positions_gives_win_rate_one():
    traces = [_trace()]
    refs = {"t1": {"text": "The expert workbook content.", "note": None, "n_files": 1}}
    metrics = _score(traces, refs, _winner_by_marker)
    rep = metrics["gdpval_report"]
    assert traces[0]["pairwise_score"] == 1.0
    assert traces[0]["pairwise_detail"]["consistent"] is True
    assert rep["win_rate"] == 1.0 and rep["wins"] == 1 and rep["losses"] == 0
    assert rep["tasks_in_win_rate"] == 1


def test_model_loses_both_positions_gives_win_rate_zero():
    traces = [_trace()]
    refs = {"t1": {"text": "Expert content.", "note": None, "n_files": 1}}
    # judge always picks the slot WITHOUT the marker -> the reference wins
    def ref_wins(prompt):
        return "B" if _winner_by_marker(prompt) == "A" else "A"
    metrics = _score(traces, refs, ref_wins)
    assert traces[0]["pairwise_score"] == 0.0
    assert metrics["gdpval_report"]["win_rate"] == 0.0
    assert metrics["gdpval_report"]["losses"] == 1


def test_a_tie_scores_half():
    traces = [_trace()]
    refs = {"t1": {"text": "Expert content.", "note": None, "n_files": 1}}
    metrics = _score(traces, refs, lambda p: "tie")
    assert traces[0]["pairwise_score"] == 0.5
    # GDPval's win rate counts a tie ("as good as the expert"); the tie-0.5 mean is secondary.
    assert metrics["gdpval_report"]["win_rate"] == 1.0
    assert metrics["gdpval_report"]["mean_pairwise_score"] == 0.5
    assert metrics["gdpval_report"]["ties_or_split"] == 1


def test_a_task_with_no_reference_is_excluded_not_a_loss():
    traces = [_trace(task_id="t1"), _trace(task_id="t2")]
    refs = {"t1": {"text": "Expert content.", "note": None, "n_files": 1},
            "t2": {"text": None, "note": "[no reference deliverable shipped]", "n_files": 0}}
    metrics = _score(traces, refs, _winner_by_marker)
    assert traces[1]["status"] == "NO_REFERENCE" and traces[1].get("scoring_excluded") is True
    rep = metrics["gdpval_report"]
    assert rep["tasks_in_win_rate"] == 1 and rep["tasks_no_reference"] == 1
    assert rep["win_rate"] == 1.0     # the excluded task did not deflate the rate


def test_an_empty_model_deliverable_is_counted_as_a_loss():
    traces = [_trace(response="")]
    refs = {"t1": {"text": "Expert content.", "note": None, "n_files": 1}}
    metrics = _score(traces, refs, _winner_by_marker)
    assert traces[0]["pairwise_score"] == 0.0 and traces[0]["status"] == "OK"
    assert metrics["gdpval_report"]["win_rate"] == 0.0
    assert metrics["gdpval_report"]["losses"] == 1


def test_a_judge_outage_is_excluded_from_the_win_rate():
    traces = [_trace()]
    refs = {"t1": {"text": "Expert content.", "note": None, "n_files": 1}}
    metrics = _score(traces, refs, lambda p: None)   # cascade exhausted
    assert traces[0]["judge_grade"] == "JUDGE_OUTAGE"
    rep = metrics["gdpval_report"]
    assert rep["win_rate"] is None and rep["tasks_in_win_rate"] == 0
    assert rep["tasks_judge_outage"] == 1


def test_the_report_exposes_the_win_rate_breakdown():
    traces = [_trace()]
    refs = {"t1": {"text": "Expert content.", "note": None, "n_files": 1}}
    rep = _score(traces, refs, _winner_by_marker)["gdpval_report"]
    for key in ("win_rate", "tasks_in_win_rate", "wins", "ties_or_split", "losses",
                "consistent_pairs", "tasks_no_reference", "tasks_judge_outage"):
        assert key in rep, f"the report must record {key}"


# --------------------------------------------------------------------------- #
# honest reporting
# --------------------------------------------------------------------------- #
def test_the_suite_requires_a_judge_and_hard_errors_without_one():
    # The gate reads GEMINI_API_KEYS first, and `import gbench` copies GEMINI_API_KEY into it, so
    # both must be cleared for the no-key path.
    saved = {k: os.environ.pop(k, None) for k in ("GEMINI_API_KEY", "GEMINI_API_KEYS")}
    try:
        # No-skip policy: a missing judge key is missing infra -> hard-error, not a skip.
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            G.run_gdpval("m", "http://x", 1, limit=1)
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_results_are_marked_not_leaderboard_comparable():
    import inspect
    src = inspect.getsource(G.run_gdpval)
    assert 'result["leaderboard_comparable"] = False' in src


def test_the_comparability_reason_is_the_text_vs_file_limit_not_the_judge():
    """Per the gbench convention the Gemini judge is intentional, not a defect; the reason a
    gdpval run is not OpenAI's number is that a text endpoint cannot emit the file deliverable."""
    import inspect
    src = inspect.getsource(G.run_gdpval)
    assert "text endpoint" in src and ".xlsx" in src


def test_the_headline_names_how_many_tasks_the_win_rate_covers():
    import inspect
    src = inspect.getsource(G.run_gdpval)
    assert "tasks_scored" in src
    assert "tasks_no_reference" in src
    assert "not counted as losses" in src


def test_docx_fallback_recovers_files_python_docx_cannot_parse(tmp_path):
    """python-docx raises XMLSyntaxError on some real-but-nonstandard .docx (2 gdpval files,
    2026-08-20). The raw word/document.xml is still readable - the loader must fall back to it
    rather than dropping the attachment as '[could not be read]'."""
    import zipfile
    from gbench.runners.eval_suites.gdpval import render_local_file
    p = tmp_path / "d.docx"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml",
                   "<w:document><w:body><w:p><w:r><w:t>Hello Contract</w:t>"
                   "</w:r></w:p></w:body></w:document>")
    text, note = render_local_file(str(p), "d.docx")
    assert note is None, f"should render, got note={note!r}"
    assert "Hello Contract" in text


# --------------------------------------------------------------------------- #
# sampling: no temperature without thinking; 1.0 (base's rule) with it
# --------------------------------------------------------------------------- #
def test_a_thinking_run_samples_at_one_and_the_reason_says_who_set_it(monkeypatch):
    from gbench.runners.eval_suites import base
    monkeypatch.delenv("GBENCH_GDPVAL_TEMPERATURE", raising=False)
    monkeypatch.setattr(base, "_RUN_KNOBS", {})
    assert base.resolve_official_sampling("gdpval", G.GDPVAL_SAMPLING, thinking=False) == (
        0.0, "default")
    assert base.resolve_official_sampling("gdpval", G.GDPVAL_SAMPLING, thinking=True) == (
        1.0, "thinking")
    seen = {}

    def fake_suite(**kw):
        seen.update(kw)
        return {"total_questions": 0, "temperature": 1.0, "temperature_source": "thinking"}

    monkeypatch.setattr(G, "gemini_required_skip", lambda *a: None)
    monkeypatch.setattr(G, "_load_gdpval_samples", lambda limit=None: ([], {}))
    monkeypatch.setattr(G, "run_eval_suite", fake_suite)
    r = G.run_gdpval("m", "http://x", 1, enable_thinking=True)
    assert seen["thinking"] is True and seen["official_sampling"] is G.GDPVAL_SAMPLING
    assert ("temperature 1.0 set by gbench's rule (1.0 with --thinking, 0.0 without)"
            in r["leaderboard_comparable_reason"])
