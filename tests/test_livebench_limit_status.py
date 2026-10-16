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

"""livebench's status counts real failures only, not categories the run asked no question of.

2026-10-11 validation run (results/validation/pool-10-11-132525/livebench): with --eval-limit 5
the stratified draw gave instruction_following 0 questions (eval_limit_plan per_setting
instruction_following 0). The status gate counted every category without a score as missing, so
the complete run (5 of 5 questions judged, accuracy 80.0, no failed request, no server abort)
was marked completed_with_errors with no status_reason.

The gate now counts categories the run asked for with no score (categories_missing: answered but
not graded, or selected but not answered; in a full run, which lists no questions, any category
without a score as before), failed requests and server aborts, and says which in status_reason.
Categories gbench's question selection gave no question are recorded apart:
categories_not_run_by_limit (active at the release, none selected) and
categories_inactive_at_release (LiveBench's loader lists none at the release).

No network, no docker: the question list and the LiveBench container are stubbed.
"""
import json
import re
import types

import gbench.runners.eval_suites.livebench as LB
from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import tau_common as TC_GUARD

OK_ROW = {"question_id": "q", "choices": [{"index": 0, "turns": ["ok"]}],
          "total_output_tokens": 10}


def _listed(categories, per_category=2):
    """A release's question list: `per_category` questions in each of `categories`."""
    return [(c, f"{c}_task", f"{c}-{i}") for c in categories for i in range(per_category)]


def _write_answers(tmp_path, category):
    d = tmp_path / LB._ANSWER_DIR / "data" / "live_bench" / category / "t" / "model_answer"
    d.mkdir(parents=True, exist_ok=True)
    (d / "m.jsonl").write_text(json.dumps(OK_ROW) + "\n", encoding="utf-8")


def _run(monkeypatch, tmp_path, listed, limit=None, drop_score=(), drop_answers=()):
    """run_livebench with docker stubbed. With `limit`, gbench selects the questions from
    `listed`; the fake container answers and scores every category with a selected question
    (every category of _CATEGORIES in a full run), except that `drop_score` categories get no
    all_groups.csv column and `drop_answers` categories no answer rows."""
    base._RUN_KNOBS.clear()
    for name in ("GBENCH_LIVEBENCH_RELEASE", "GBENCH_LIVEBENCH_IMAGE",
                 "GBENCH_LIVEBENCH_CATEGORIES", "GBENCH_LIVEBENCH_TEMPERATURE",
                 "GBENCH_LIVEBENCH_MODEL", "GBENCH_LIVEBENCH_EXTRA_ARGS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(LB, "_check_prereqs", lambda image: None)
    monkeypatch.setattr(TC_GUARD, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC_GUARD.container_url(url))
    monkeypatch.setattr(LB, "upstream_model_config",
                        lambda image, names: {"name": None, "config": None,
                                              "commit": LB.LIVEBENCH_COMMIT})
    monkeypatch.setattr(LB, "list_questions", lambda image, release: sorted(listed))
    monkeypatch.setattr(LB.tempfile, "mkdtemp", lambda prefix="": str(tmp_path))
    category_of = {q: c for c, _t, q in listed}

    def fake_run(cmd, **_kw):
        m = re.search(r"--question-id ((?:\S+ )+)", cmd[-1])
        run = (sorted({category_of[q] for q in m.group(1).split()}) if m
               else list(LB._CATEGORIES))
        scored = [c for c in run if c not in drop_score]
        header = "model," + ("average," if len(scored) > 1 else "") + ",".join(scored)
        values = "m," + ("50.0," if len(scored) > 1 else "") + ",".join("50.0" for _ in scored)
        (tmp_path / "all_groups.csv").write_text(header + "\n" + values + "\n", encoding="utf-8")
        for c in run:
            if c not in drop_answers:
                _write_answers(tmp_path, c)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(LB.subprocess, "run", fake_run)
    return LB.run_livebench("m", "http://h/v1", 4, max_output_tokens=4096, limit=limit)


def _zero_categories(r):
    return sorted(c for c, n in r["eval_limit_plan"]["per_setting"].items() if not n)


def test_a_category_the_limit_gave_no_question_is_not_an_error(monkeypatch, tmp_path):
    # the validation run's shape: --eval-limit 5 over the 6 core categories
    r = _run(monkeypatch, tmp_path, _listed(LB._CATEGORIES), limit=5)
    zero = _zero_categories(r)
    assert len(zero) == 1
    assert r["status"] == "success" and "status_reason" not in r
    assert r["categories_missing"] == [] and r["categories_unscored"] == []
    assert r["categories_not_run_by_limit"] == zero
    assert r["categories_inactive_at_release"] == []
    assert r["accuracy"] == 50.0
    # still a subset, so still not comparable, and the category is named there
    assert r["leaderboard_comparable"] is False
    assert (f"categories with no question run (none among the selected questions, or none "
            f"active at this release): {zero}") in r["leaderboard_comparable_reason"]


def test_a_category_with_no_question_at_the_release_is_not_an_error(monkeypatch, tmp_path):
    # LiveBench's loader lists no coding question after 2024-11-25 (removal date 2025-04-02)
    listed = _listed([c for c in LB._CATEGORIES if c != "coding"])
    r = _run(monkeypatch, tmp_path, listed, limit=100)
    assert r["status"] == "success" and "status_reason" not in r
    assert r["categories_inactive_at_release"] == ["coding"]
    assert r["categories_not_run_by_limit"] == [] and r["categories_missing"] == []


def test_selected_questions_left_unanswered_are_an_error_with_a_reason(monkeypatch, tmp_path):
    r = _run(monkeypatch, tmp_path, _listed(LB._CATEGORIES), limit=100,
             drop_score=("math",), drop_answers=("math",))
    assert r["status"] == "completed_with_errors"
    assert r["categories_missing"] == ["math"] and r["categories_unscored"] == []
    assert "did not answer the selected questions of ['math']" in r["status_reason"]
    assert "categories whose selected questions LiveBench did not answer" in \
        r["leaderboard_comparable_reason"]
    assert r["categories_not_run_by_limit"] == []


def test_answered_but_ungraded_questions_are_an_error_with_a_reason(monkeypatch, tmp_path):
    r = _run(monkeypatch, tmp_path, _listed(LB._CATEGORIES), limit=100,
             drop_score=("instruction_following",))
    assert r["status"] == "completed_with_errors"
    assert r["categories_missing"] == r["categories_unscored"] == ["instruction_following"]
    assert "grading wrote no score" in r["status_reason"]


def test_a_full_run_still_counts_a_missing_category_and_now_says_why(monkeypatch, tmp_path):
    r = _run(monkeypatch, tmp_path, [], drop_score=("coding",), drop_answers=("coding",))
    assert r["status"] == "completed_with_errors"
    assert r["categories_missing"] == ["coding"]
    assert r["categories_not_run_by_limit"] == r["categories_inactive_at_release"] == []
    assert "no score for ['coding'] and no answer rows" in r["status_reason"]


def test_a_complete_full_run_is_a_success(monkeypatch, tmp_path):
    r = _run(monkeypatch, tmp_path, [])
    assert r["status"] == "success" and "status_reason" not in r
    assert r["categories_missing"] == [] and r["categories_not_run_by_limit"] == []


def _shard(label, category, not_run=(), status="success", reason=None):
    r = {"eval_name": "livebench", "shard": label, "status": status, "accuracy": 50.0,
         "categories_missing": [], "categories_unscored": [],
         "categories_inactive_at_release": [], "categories_not_run_by_limit": list(not_run),
         "sample_traces": [{"sample_key": f"question:{category}/{label}", "question_id": label,
                            "category": category, "task": f"{category}_task", "score": 0.5}]}
    if reason:
        r["status_reason"] = reason
    return r


def test_the_merge_keeps_only_categories_no_shard_ran():
    merged = LB.merge_shards([_shard("1/2", "math", not_run=["coding", "language"]),
                              _shard("2/2", "coding", not_run=["math", "language"])])
    assert merged["categories_not_run_by_limit"] == ["language"]
    assert merged["status"] == "success" and "status_reason" not in merged
    merged = LB.merge_shards([_shard("1/2", "math"),
                              _shard("2/2", "coding", status="completed_with_errors",
                                     reason="1 request(s) failed at the served endpoint")])
    assert merged["status"] == "completed_with_errors"
    assert merged["status_reason"] == "shard 2/2: 1 request(s) failed at the served endpoint"


def test_a_merge_of_only_empty_shards_keeps_their_status_reason():
    """2026-10-11 review of the livebench-gate fix: merge_shards dropped status_reason whenever the
    merged status was success, so merging shards that were all empty (more shards than
    questions) lost the 'empty shard I/N' reason that a single empty shard's result carries."""
    def empty(label):
        return {"eval_name": "livebench", "shard": label, "status": "success",
                "status_reason": f"empty shard {label}: no items on this node (1 in the full set)",
                "empty_shard": True, "accuracy": None, "sample_traces": [],
                "accuracy_withheld_reason": f"empty shard {label}"}
    merged = LB.merge_shards([empty("1/2"), empty("2/2")])
    assert merged["status"] == "success" and merged["accuracy"] is None
    assert merged["status_reason"] == (
        "empty shard 1/2: no items on this node (1 in the full set); "
        "empty shard 2/2: no items on this node (1 in the full set)")
    # An empty shard next to a scored one adds no reason to a successful merge.
    merged = LB.merge_shards([empty("1/2"), _shard("2/2", "math")])
    assert merged["status"] == "success" and "status_reason" not in merged
