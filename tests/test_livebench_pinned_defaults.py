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

"""livebench runs a commit-tagged local image and a pinned LiveBench release by default.

The image is docker/livebench.Dockerfile built with a LiveBench checkout at LIVEBENCH_COMMIT as
the context and tagged gbench-livebench:<LIVEBENCH_COMMIT[:8]> (an untagged gbench-livebench is
not used). The release is LIVEBENCH_RELEASE, 2024-11-25: LiveBench's README tells users to pass
it to evaluate all categories on the public questions; GBENCH_LIVEBENCH_RELEASE overrides it and
`latest` leaves the release to LiveBench (unpinned). A category LiveBench answered but did not
score is reported as unscored, not as a category with no active question.
"""
import json
import os
import re
import types

import pytest

import gbench.runners.eval_suites.livebench as LB
from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import tau_common as TC_GUARD

ROOT = os.path.join(os.path.dirname(__file__), "..")
DOC = os.path.join(ROOT, "docs", "evals", "livebench.md")
DOCKERFILE = os.path.join(ROOT, "docker", "livebench.Dockerfile")

OK_ROW = {"question_id": "q", "choices": [{"index": 0, "turns": ["ok"]}],
          "total_output_tokens": 10}


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _flat(text):
    """Shell text with line continuations and comment markers folded into single spaces."""
    text = re.sub(r"^#\s?", "", text, flags=re.M)
    return re.sub(r"\s+", " ", text.replace("\\\n", " "))


# --- image ----------------------------------------------------------------------------------

def test_default_image_is_tagged_with_the_pinned_commit():
    assert LB.LIVEBENCH_COMMIT == "0baa4aa7a78dc0276c0ccf28779dba1113580e18"
    assert LB.LIVEBENCH_IMAGE_DEFAULT == "gbench-livebench:" + LB.LIVEBENCH_COMMIT[:8]
    assert LB.LIVEBENCH_IMAGE_DEFAULT == "gbench-livebench:0baa4aa7"


def test_image_knob_overrides_and_empty_means_default(monkeypatch):
    monkeypatch.delenv("GBENCH_LIVEBENCH_IMAGE", raising=False)
    assert LB._image() == LB.LIVEBENCH_IMAGE_DEFAULT
    monkeypatch.setenv("GBENCH_LIVEBENCH_IMAGE", "  ")
    assert LB._image() == LB.LIVEBENCH_IMAGE_DEFAULT
    monkeypatch.setenv("GBENCH_LIVEBENCH_IMAGE", "my-livebench:abc")
    assert LB._image() == "my-livebench:abc"


def test_build_command_tags_the_commit_and_uses_the_livebench_checkout_as_context():
    cmd = LB.BUILD_COMMAND.split()
    assert cmd[:4] == ["docker", "build", "-t", LB.LIVEBENCH_IMAGE_DEFAULT]
    assert cmd[4:6] == ["-f", "docker/livebench.Dockerfile"]
    assert cmd[-1] == "/path/to/LiveBench"          # the checkout, not gbench's docker/ dir
    assert len(cmd) == 7


def test_doc_and_dockerfile_carry_the_build_command_and_the_pinned_checkout():
    for text in (_flat(_read(DOC)), _flat(_read(DOCKERFILE).split("FROM ")[0])):
        assert LB.BUILD_COMMAND in text
        assert f"git -C /path/to/LiveBench checkout {LB.LIVEBENCH_COMMIT}" in text
        assert "git -C /path/to/LiveBench lfs pull" in text
        assert "docker build -t " + "gbench-livebench -f" not in text     # no untagged build


def test_dockerfile_edit_touched_only_comments():
    lines = [ln for ln in _read(DOCKERFILE).splitlines() if ln.strip()
             and not ln.lstrip().startswith("#")]
    # the base as it was, now pinned by the digest the built image already sits on
    assert lines[0] == ("FROM python:3.11-slim@sha256:"
                        "9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534")
    assert lines[-1] == "WORKDIR /opt/LiveBench/livebench"
    assert "COPY . /opt/LiveBench" in lines


def test_missing_image_names_the_tagged_build_from_the_pinned_checkout(monkeypatch):
    monkeypatch.setattr(LB.shutil, "which", lambda _x: "/usr/bin/docker")

    def fake_run(cmd, **_kw):
        return types.SimpleNamespace(returncode=0 if cmd[:2] == ["docker", "info"] else 1,
                                     stdout="", stderr="")

    monkeypatch.setattr(LB.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError) as err:
        LB._check_prereqs(LB.LIVEBENCH_IMAGE_DEFAULT)
    msg = str(err.value)
    assert "gbench-livebench:0baa4aa7' not found" in msg
    assert LB.BUILD_COMMAND in msg
    assert f"git -C /path/to/LiveBench checkout {LB.LIVEBENCH_COMMIT}" in msg
    assert "docs/evals/livebench.md" in msg


def test_doc_configuration_table_states_both_defaults():
    doc = _read(DOC)
    assert "| `GBENCH_LIVEBENCH_IMAGE` | image tag (default `gbench-livebench:0baa4aa7`" in doc
    assert "| `GBENCH_LIVEBENCH_RELEASE` | the LiveBench release date (default `2024-11-25`" in doc
    assert "(default = latest)" not in doc
    assert "GBENCH_LIVEBENCH_RELEASE=2025-04-02" not in doc


# --- release --------------------------------------------------------------------------------

def test_default_release_is_the_newest_fully_public_one():
    assert LB.LIVEBENCH_RELEASE == "2024-11-25"
    assert "LIVEBENCH_RELEASE (2024-11-25)" in LB.__doc__
    # The pin cites LiveBench's own instruction for its public questions.
    doc = _read(DOC)
    assert "`--livebench-release-option 2024-11-25` to all scripts to use the most recent public" \
        in _flat(doc)
    assert "| 2024-11-25 | 128 | 150 | 200 | 140 | 232 | 150 | 1000 |" in doc


@pytest.mark.parametrize("env, expected", [
    (None, ("2024-11-25", "default")),
    ("", ("2024-11-25", "default")),
    ("  ", ("2024-11-25", "default")),
    ("2025-04-25", ("2025-04-25", "env:GBENCH_LIVEBENCH_RELEASE")),
    (" 2024-08-31 ", ("2024-08-31", "env:GBENCH_LIVEBENCH_RELEASE")),
    ("latest", ("", "env:GBENCH_LIVEBENCH_RELEASE")),
    ("LATEST", ("", "env:GBENCH_LIVEBENCH_RELEASE")),
])
def test_release_knob(monkeypatch, env, expected):
    if env is None:
        monkeypatch.delenv("GBENCH_LIVEBENCH_RELEASE", raising=False)
    else:
        monkeypatch.setenv("GBENCH_LIVEBENCH_RELEASE", env)
    assert LB._release() == expected


def _write_answers(tmp_path, category, rows=(OK_ROW,)):
    d = (tmp_path / LB._ANSWER_DIR / "data" / "live_bench" / category / "t" / "model_answer")
    d.mkdir(parents=True, exist_ok=True)
    (d / "m.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _run(monkeypatch, tmp_path, release=None, scored=None, answered=None, output="",
         image=None):
    """run_livebench with docker stubbed: all_groups.csv holds `scored` (default: all 6) and
    LiveBench's answer rows exist for `answered` (default: the scored categories)."""
    base._RUN_KNOBS.clear()
    if release is None:
        monkeypatch.delenv("GBENCH_LIVEBENCH_RELEASE", raising=False)
    else:
        monkeypatch.setenv("GBENCH_LIVEBENCH_RELEASE", release)
    if image is None:
        monkeypatch.delenv("GBENCH_LIVEBENCH_IMAGE", raising=False)
    else:
        monkeypatch.setenv("GBENCH_LIVEBENCH_IMAGE", image)
    for name in ("GBENCH_LIVEBENCH_CATEGORIES", "GBENCH_LIVEBENCH_TEMPERATURE",
                 "GBENCH_LIVEBENCH_MODEL", "GBENCH_LIVEBENCH_EXTRA_ARGS"):
        monkeypatch.delenv(name, raising=False)
    scored = list(LB._CATEGORIES) if scored is None else scored
    answered = scored if answered is None else answered
    seen = {}
    monkeypatch.setattr(LB, "_check_prereqs", lambda image: seen.setdefault("checked", image))
    # the network checks before the run container are docker calls of their own
    # (tests/test_guard_code_images.py)
    monkeypatch.setattr(TC_GUARD, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC_GUARD.container_url(url))

    def fake_lookup(image, names):
        seen["lookup_image"] = image
        return {"name": None, "config": None, "commit": LB.LIVEBENCH_COMMIT}

    monkeypatch.setattr(LB, "upstream_model_config", fake_lookup)
    monkeypatch.setattr(LB.tempfile, "mkdtemp", lambda prefix="": str(tmp_path))

    def fake_run(cmd, **_kw):
        seen["cmd"] = cmd
        header = "model," + ("average," if len(scored) > 1 else "") + ",".join(scored)
        values = "m," + ("50.0," if len(scored) > 1 else "") + ",".join("50.0" for _ in scored)
        (tmp_path / "all_groups.csv").write_text(header + "\n" + values + "\n", encoding="utf-8")
        for category in answered:
            _write_answers(tmp_path, category)
        return types.SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(LB.subprocess, "run", fake_run)
    result = LB.run_livebench("m", "http://h/v1", 4, max_output_tokens=4096)
    return result, seen


def test_a_default_run_uses_the_tagged_image_and_the_pinned_release(monkeypatch, tmp_path):
    r, seen = _run(monkeypatch, tmp_path)
    assert seen["checked"] == seen["lookup_image"] == "gbench-livebench:0baa4aa7"
    cmd = seen["cmd"]
    assert cmd[cmd.index("-v") + 2] == "gbench-livebench:0baa4aa7"
    inner = cmd[-1]
    # both LiveBench scripts get the release
    assert inner.count("--livebench-release-option 2024-11-25") == 2
    assert r["livebench_release"] == "2024-11-25"
    assert r["livebench_release_source"] == "default"
    assert r["categories_missing"] == [] and r["categories_unscored"] == []
    assert r["status"] == "success"
    assert r["leaderboard_comparable"] is True, r.get("leaderboard_comparable_reason")


def test_the_release_knob_still_overrides_the_pin(monkeypatch, tmp_path):
    r, seen = _run(monkeypatch, tmp_path, release="2024-08-31")
    assert seen["cmd"][-1].count("--livebench-release-option 2024-08-31") == 2
    assert "2024-11-25" not in seen["cmd"][-1]
    assert r["livebench_release"] == "2024-08-31"
    assert r["livebench_release_source"] == "env:GBENCH_LIVEBENCH_RELEASE"


def test_latest_leaves_the_release_to_livebench_and_is_not_comparable(monkeypatch, tmp_path):
    r, seen = _run(monkeypatch, tmp_path, release="latest")
    assert "--livebench-release-option" not in seen["cmd"][-1]
    assert r["livebench_release"] == "latest"
    assert r["leaderboard_comparable"] is False
    assert "unpinned release" in r["leaderboard_comparable_reason"]
    assert "the pinned release is 2024-11-25" in r["leaderboard_comparable_reason"]


def test_the_image_knob_still_overrides_the_tag(monkeypatch, tmp_path):
    _, seen = _run(monkeypatch, tmp_path, image="my-livebench:abc")
    assert seen["checked"] == seen["lookup_image"] == "my-livebench:abc"
    assert "my-livebench:abc" in seen["cmd"]


# --- categories with no score ---------------------------------------------------------------

def test_an_answered_category_without_a_score_is_unscored_not_inactive(monkeypatch, tmp_path):
    # LiveBench@0baa4aa7 answers instruction_following but its batch grading raises on it.
    scored = [c for c in LB._CATEGORIES if c != "instruction_following"]
    r, _ = _run(monkeypatch, tmp_path, scored=scored, answered=list(LB._CATEGORIES),
                output="TypeError: unhashable type: 'dict'")
    assert r["categories_missing"] == ["instruction_following"]
    assert r["categories_unscored"] == ["instruction_following"]
    reason = r["leaderboard_comparable_reason"]
    assert "answered but not scored by LiveBench's grading" in reason
    assert "no active questions" not in reason
    assert r["status"] == "completed_with_errors"
    assert r["leaderboard_comparable"] is False
    assert r["dimension_scores"] == {c: 50.0 for c in scored}


def test_a_category_without_answers_had_no_active_question(monkeypatch, tmp_path):
    scored = [c for c in LB._CATEGORIES if c != "coding"]
    r, _ = _run(monkeypatch, tmp_path, release="2025-04-02", scored=scored)
    assert r["categories_missing"] == ["coding"] and r["categories_unscored"] == []
    reason = r["leaderboard_comparable_reason"]
    assert "categories with no active questions at this release: ['coding']" in reason
    assert "answered but not scored" not in reason
    assert r["status"] == "completed_with_errors"


def test_nothing_scored_but_answers_present_is_a_harness_failure(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="grading scored none of them - a harness failure"):
        _run(monkeypatch, tmp_path, scored=["agentic_coding"],
             answered=["instruction_following"])


def test_nothing_scored_and_nothing_answered_points_at_the_pinned_release(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError) as err:
        _run(monkeypatch, tmp_path, release="2026-06-25", scored=["agentic_coding"], answered=[])
    msg = str(err.value)
    assert "no categories were scored for release '2026-06-25'" in msg
    assert "2024-11-25, the default" in msg
    assert "2025-04-02 for all 6" not in msg


def test_answered_categories_reads_only_non_empty_live_bench_answer_files(tmp_path):
    _write_answers(tmp_path, "math")
    _write_answers(tmp_path, "coding", rows=())                      # empty file
    other = tmp_path / LB._ANSWER_DIR / "data" / "elsewhere" / "language" / "t" / "model_answer"
    other.mkdir(parents=True)
    (other / "m.jsonl").write_text(json.dumps(OK_ROW) + "\n", encoding="utf-8")
    assert LB.answered_categories(str(tmp_path)) == ["math"]
    assert LB.answered_categories(str(tmp_path / "missing")) == []


def test_an_image_of_another_commit_names_the_tagged_build(monkeypatch, tmp_path):
    base._RUN_KNOBS.clear()
    monkeypatch.delenv("GBENCH_LIVEBENCH_RELEASE", raising=False)
    monkeypatch.delenv("GBENCH_LIVEBENCH_IMAGE", raising=False)
    for name in ("GBENCH_LIVEBENCH_CATEGORIES", "GBENCH_LIVEBENCH_TEMPERATURE",
                 "GBENCH_LIVEBENCH_MODEL", "GBENCH_LIVEBENCH_EXTRA_ARGS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(LB, "_check_prereqs", lambda image: None)
    # the network checks before the run container are docker calls of their own
    # (tests/test_guard_code_images.py)
    monkeypatch.setattr(TC_GUARD, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC_GUARD.container_url(url))
    monkeypatch.setattr(LB, "upstream_model_config", lambda image, names: {
        "name": None, "config": None, "commit": "bc5164c1841e3cf099b4fb48c59b49ee0a734bef"})
    monkeypatch.setattr(LB.tempfile, "mkdtemp", lambda prefix="": str(tmp_path))

    def fake_run(cmd, **_kw):
        cols = LB._CATEGORIES
        (tmp_path / "all_groups.csv").write_text(
            "model,average," + ",".join(cols) + "\nm,50.0," + ",".join("50.0" for _ in cols)
            + "\n", encoding="utf-8")
        for c in cols:
            _write_answers(tmp_path, c)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(LB.subprocess, "run", fake_run)
    r = LB.run_livebench("m", "http://h/v1", 4, max_output_tokens=4096)
    reason = r["leaderboard_comparable_reason"]
    assert "holds LiveBench bc5164c1841e, not the pinned 0baa4aa7a78d" in reason
    assert LB.BUILD_COMMAND in reason
