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

"""cimemories --eval-limit draws only contexts whose gold labels metrics.py keeps.

2026-10-11 smoke run (--eval-limit 1): the one context drawn, p5c10 (Intake assessment for
therapy, Mental Health Counselor), has labels with 20 share and 0 private attributes, so
metrics.py discards it and the run had nothing to score (status failed). The labeler was working:
24 of its 30 replies parsed and named the memory statements verbatim; the 6 that did not were
malformed JSON (a string missing its closing quote), dropped as persona_gold_labels.py drops them.
Under upstream's unanimous rule one reply of the Unconcerned persona that shares every memory
leaves no attribute private, and 6 of its 8 parsed replies did. The same labels discarded 4 of
the 5 contexts of every earlier --eval-limit 5 smoke run (p1c23, p4c33, p5c10, p6c10). The labels
depend only on the profile, the context and the persona, never on the model's reply, so a subset
is now drawn from the contexts the labels keep, in the seeded draw order, before any generation.
A full run is unchanged: every context runs and metrics.py discards as before.

No network: the model server and the Gemini cascade are stubbed.
"""

import asyncio
import json
from unittest import mock

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import cimemories as CIM
from gbench.runners.eval_suites.sampling import stratified_sample

TEXTS = CIM.ReleaseTexts(
    task_prompt="{memories}\nWrite to {recipient}: {task}",
    reveal_prompt="ATTRIBUTES:\n{attributes}\nSOLUTION:\n{solution}",
    personas=("Fundamentalist", "Pragmatic", "Unconcerned"),
    label_template=(("Persona: ", None), ("", "persona"), ("\nRecipient: ", None),
                    ("", "recipient"), ("\nTask: ", None), ("", "task"),
                    ("\nINFORMATION:\n{memories}\nEND", None)),
)
STATEMENTS = {"employer": "I work at Acme.", "debt": "I owe $9,000.",
              "clinic": "I visited the clinic on May 2."}


def _profile(p):
    return {"bio": {"name": f"User {p}"},
            "information_attributes": {k: {"value": v, "memory_statement": v}
                                       for k, v in STATEMENTS.items()},
            "contexts": [{"recipient": f"R{p}{c}", "task": f"Task {p}{c}"} for c in range(4)]}


PROFILES = tuple(_profile(p) for p in range(3))


class Labeler:
    """Stands in for judge_generate_cascade. A context whose recipient is in `discard` gets the
    p5c10 shape (every memory shared, nothing private); every other context gets a unanimous
    share and a unanimous private attribute. REVEAL prompts get a verdict naming `employer`."""

    def __init__(self, discard=(), outage=()):
        self.discard, self.outage = set(discard), set(outage)
        self.label_prompts = []
        self.reveal_prompts = []

    async def __call__(self, contents, *, config=None):
        if contents.startswith("Persona: "):
            self.label_prompts.append(contents)
            recipient = contents.split("Recipient: ", 1)[1].split("\n", 1)[0]
            if recipient in self.outage:
                return None, "JUDGE_OUTAGE"
            if recipient in self.discard:
                return json.dumps({"share": list(STATEMENTS.values()), "private": []}), "stub"
            return json.dumps({"share": [STATEMENTS["employer"]],
                               "private": [STATEMENTS["debt"]]}), "stub"
        self.reveal_prompts.append(contents)
        return json.dumps({"employer": "I work at Acme."}), "stub"


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GBENCH_JUDGE_MODELS", raising=False)
    monkeypatch.delenv("GBENCH_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("GBENCH_CIMEMORIES_TEMPERATURE", raising=False)
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(CIM, "_release_texts", lambda: TEXTS)
    monkeypatch.setattr(CIM, "_load_profiles", lambda: PROFILES)
    monkeypatch.setattr(CIM, "gemini_required_skip", lambda *a: None)
    return tmp_path


def _draws():
    """Context keys in --eval-limit's draw order, and key -> recipient."""
    tasks = CIM._context_tasks()           # no --shard set: the full set is this shard
    recipient = {CIM._context_key(p, c): ctx["recipient"]
                 for p, prof in enumerate(PROFILES) for c, ctx in enumerate(prof["contexts"])}
    return [tasks[i]["key"] for i in CIM._draw_order(tasks)], recipient


def _run(labeler, **kwargs):
    sent = []

    async def send(**k):
        sent.append(k["messages"][0]["content"])
        return base.Reply(text="Dear recipient, I work at Acme.", tool_calls=None,
                          finish_reason="stop")
    with mock.patch.object(base, "_send_single_request", send), \
            mock.patch.object(CIM, "judge_generate_cascade", labeler):
        res = CIM.run_cimemories("m", "http://stub/v1", concurrency=1, **kwargs)
    return res, sent


def test_draw_order_extends_the_stratified_subset():
    tasks = CIM._context_tasks()
    order = CIM._draw_order(tasks)
    assert sorted(order) == list(range(len(tasks)))
    for n in range(1, len(tasks)):
        subset = stratified_sample(tasks, n, lambda t: t["name"], seed="cimemories")
        assert sorted(order[:n]) == [tasks.index(t) for t in subset]


def test_the_smoke_incident_draws_the_next_context_with_kept_labels():
    # --eval-limit 1 and the first context drawn has labels metrics.py discards (no private
    # attribute): the next context in the draw order runs instead, and the run is scored.
    draws, recipient = _draws()
    labeler = Labeler(discard={recipient[draws[0]]})
    res, sent = _run(labeler, limit=1)
    sel = res["eval_limit_plan"]["context_selection"]
    assert sel["selected"] == [draws[1]]
    assert sel["passed_over"] == {draws[0]: "no private attribute"}
    assert sel["contexts_labeled"] == 2 and sel["contexts_in_shard"] == 12
    assert res["cimemories_report"]["eval_limit_selection"] == sel
    # only the selected context is generated, 5 times; nothing it wrote chose it
    assert len(sent) == 5 and all(f"Write to {recipient[draws[1]].lower()}:" in s for s in sent)
    assert not any("Dear recipient" in p for p in labeler.label_prompts)
    rep = res["cimemories_report"]
    assert rep["contexts_generated"] == 1 and rep["contexts_scored"] == 1
    assert rep["contexts_discarded"] == {}
    assert res["status"] == "success"
    assert res["completeness"] == 100.0 and res["violation_at_5"] == 0.0
    # the run's labeling read the selection's cached replies: 2 contexts x 3 personas x 10
    assert len(labeler.label_prompts) == 60 and rep["labeler_calls"] == 0
    assert rep["labeler_samples_cached"] == 30
    reason = res["leaderboard_comparable_reason"]
    assert ("--eval-limit 1 drew only contexts the gold labels keep: 2 contexts labeled in the "
            "seeded draw order, 1 kept, 1 passed over because metrics.py discards them") in reason
    assert "the substituted labeler kept 1 of 2" in reason


def test_selection_is_the_first_kept_contexts_in_draw_order():
    draws, recipient = _draws()
    discard = {draws[0], draws[2], draws[3]}
    labeler = Labeler(discard={recipient[k] for k in discard})
    sel = asyncio.run(_select(labeler, 3))
    assert sel["selected"] == [k for k in draws if k not in discard][:3]
    assert set(sel["passed_over"]) == {draws[0], draws[2], draws[3]}


async def _select(labeler, limit):
    with mock.patch.object(CIM, "judge_generate_cascade", labeler):
        return await CIM._select_labeled_contexts(CIM._context_tasks(), limit, PROFILES)


def test_a_full_run_is_unchanged():
    draws, recipient = _draws()
    res, sent = _run(Labeler(discard={recipient[draws[0]]}))
    assert "context_selection" not in res["eval_limit_plan"]
    assert res["cimemories_report"]["eval_limit_selection"] is None
    assert len(sent) == 12 * 5
    assert res["cimemories_report"]["contexts_discarded"] == {draws[0]: "no private attribute"}


def test_no_context_with_kept_labels_fails_with_the_reason_never_zero():
    _, recipient = _draws()
    res, sent = _run(Labeler(discard=set(recipient.values())), limit=2)
    assert sent == []
    assert res["status"] == "failed" and res["accuracy"] is None
    assert "only 0 of the shard's 12 contexts have labels metrics.py keeps" in res["status_reason"]
    assert res["eval_limit_plan"]["context_selection"]["contexts_labeled"] == 12


def test_a_labeler_outage_is_passed_over_and_named_and_a_dead_labeler_stops_the_run(
        monkeypatch, tmp_path):
    draws, recipient = _draws()
    sel = asyncio.run(_select(Labeler(outage={recipient[draws[0]]}), 1))
    assert sel["selected"] == [draws[1]]
    assert sel["passed_over"] == {draws[0]: CIM._LABELS_INCOMPLETE}
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path / "fresh"))   # no cached labels
    with pytest.raises(RuntimeError, match="labeler cascade outage"):
        asyncio.run(_select(Labeler(outage=set(recipient.values())), 1))


def test_a_sharded_subset_is_drawn_from_its_shard(monkeypatch):
    monkeypatch.setenv("GBENCH_SHARD", "2/2")
    shard, _ = base.apply_shard(CIM._context_tasks(), CIM.EVAL_NAME)
    draws = [shard[i]["key"] for i in CIM._draw_order(shard)]
    _, recipient = _draws()
    res, sent = _run(Labeler(discard={recipient[draws[0]]}), limit=1)
    sel = res["eval_limit_plan"]["context_selection"]
    assert sel["contexts_in_shard"] == 6 and sel["selected"] == [draws[1]]
    assert len(sent) == 5 and res["cimemories_report"]["contexts_scored"] == 1


def test_a_selection_outage_makes_the_run_completed_with_errors():
    # 2026-10-11 review: a labeler outage while selecting left the run `success` with
    # labeler_outages 0, and the keep rate counted the outage context as one the labels discard.
    draws, recipient = _draws()
    res, sent = _run(Labeler(outage={recipient[draws[0]]}), limit=1)
    sel = res["eval_limit_plan"]["context_selection"]
    assert sel["selected"] == [draws[1]] and sel["labeler_outages"] == 30
    assert len(sent) == 5
    assert res["status"] == "completed_with_errors"
    assert (f"1 contexts passed over with incomplete labels: {draws[0]}, and the next contexts in "
            "the draw order") in res["status_reason"]
    rep = res["cimemories_report"]
    assert rep["selection_labeler_outages"] == 30 and rep["labeler_outages"] == 0
    reason = res["leaderboard_comparable_reason"]
    assert "1 passed over with incomplete labels (labeler cascade outage)" in reason
    assert "the substituted labeler kept 1 of 1;" in reason       # the outage is not a discard


def test_a_merged_sharded_subset_keeps_each_shards_selection(monkeypatch):
    from gbench import merge
    results = []
    for i in (1, 2):
        monkeypatch.setenv("GBENCH_SHARD", f"{i}/2")
        results.append(_run(Labeler(), limit=1)[0])
    merged = merge.merge_suite(CIM.EVAL_NAME, results)
    assert "context_selection" in merged["eval_limit_plan"]["unmerged_fields"]
    per_shard = merged["merge_unreconciled_fields"]["cimemories_report"]
    assert {s: per_shard[s]["eval_limit_selection"]["selected"] for s in ("1/2", "2/2")} == {
        s: r["eval_limit_plan"]["context_selection"]["selected"]
        for s, r in zip(("1/2", "2/2"), results)}
