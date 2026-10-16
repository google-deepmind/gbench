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

"""Run directories, shard merges and the test-file license headers, checked by behaviour:

* a run directory is one run's alone: an existing run directory given as --results-dir is
  continued in place only while no other run is writing there; a run that finds it in use (swarm
  nodes pointed at one shared directory) writes its own <timestamp> directory inside it, so no
  node overwrites another's metadata.json, summary or results; a dry run creates nothing;
* `gbench --merge`: every shard's traces are kept (rows with identical content stay apart, an
  item in two shards is refused), no field of the merged result is one shard's value standing
  for the union (counts summed, the Gemini models of every shard combined, disagreeing fields
  set aside per shard), a refusal or a dry run writes nothing, and two merges never share a
  directory;
* mmlu and mmlu_redux: the merge of a run's shards equals the unsharded run (headline, counts,
  subject mean, reasons, status), thresholds judged on the union;
* every test file carries the Apache license header (the maintainer's plugin tests are not
  checked).

No test contacts a network, a model server, Gemini or Docker; everything is written under tmp.
"""

import json
import os
import re
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# run directories
# --------------------------------------------------------------------------- #
_HOLDER = textwrap.dedent("""
    import sys
    sys.path.insert(0, sys.argv[2])
    from gbench.utils.log_manager import LogManager
    lm = LogManager(sys.argv[1])
    lm.save_metadata({"node": 1})
    lm.save_summary({"node": 1})
    print(lm.results_dir, flush=True)
    sys.stdin.read()          # holds the run directory until the test closes stdin
""")


class _OtherRun:
    """A second process that runs LogManager on a directory and keeps running (its lock held)
    until stopped."""

    def __init__(self, run_dir: Path):
        env = {k: v for k, v in os.environ.items() if k != "GBENCH_RESULTS_DIR"}
        self.proc = subprocess.Popen([sys.executable, "-c", _HOLDER, str(run_dir), str(REPO)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, env=env,
                                     cwd=str(run_dir.parent))
        self.results_dir = Path(self.proc.stdout.readline().strip())

    def stop(self):
        try:
            self.proc.communicate(input="", timeout=30)
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait()


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """An existing run directory (it holds metadata.json), as an interrupted run leaves it."""
    monkeypatch.setenv("GBENCH_RESULTS_DIR", "unchanged-by-the-test")
    run = tmp_path / "shared"
    run.mkdir()
    (run / "metadata.json").write_text(json.dumps({"node": 0}))
    return run


def test_an_existing_run_directory_in_use_by_another_run_is_not_shared(run_dir):
    from gbench.utils.log_manager import LogManager, RUN_LOCK_NAME
    other = _OtherRun(run_dir)
    try:
        assert other.results_dir == run_dir                      # free: continued in place
        holder = (run_dir / RUN_LOCK_NAME).read_text()
        assert holder.startswith("pid %d on " % other.proc.pid)
        assert "." not in holder.split(" on ", 1)[1].split(",", 1)[0]     # the short host name
        mine = LogManager(run_dir)
        # in use: this run writes its own directory inside it and says why
        assert mine.results_dir.parent == run_dir and mine.results_dir != run_dir
        assert mine.shared_dir_note.startswith("in use by pid %d on " % other.proc.pid)
        mine.save_metadata({"node": 2})
        mine.save_summary({"node": 2})
        # neither run overwrote the other's files
        assert json.loads((run_dir / "metadata.json").read_text()) == {"node": 1}
        assert json.loads((run_dir / "summary.json").read_text()) == {"node": 1}
        assert json.loads((mine.results_dir / "metadata.json").read_text()) == {"node": 2}
        # a third run finds both in use and claims yet another directory
        third = LogManager(run_dir)
        assert third.results_dir not in (run_dir, mine.results_dir)
        assert third.results_dir.parent == run_dir
    finally:
        other.stop()
    # the other run has ended: the directory is continued in place again (sequential reuse)
    mine.release_run_lock()
    third.release_run_lock()
    again = LogManager(run_dir)
    assert again.results_dir == run_dir and again.shared_dir_note is None


def test_a_dry_run_names_the_directory_it_would_use_and_creates_nothing(run_dir):
    from gbench.utils.log_manager import LogManager
    other = _OtherRun(run_dir)
    try:
        before = sorted(p.name for p in run_dir.iterdir())
        dry = LogManager(run_dir, create_dirs=False)
        assert dry.results_dir.parent == run_dir and dry.results_dir != run_dir
        assert dry.shared_dir_note.startswith("in use by pid")
        assert sorted(p.name for p in run_dir.iterdir()) == before
        assert os.environ["GBENCH_RESULTS_DIR"] == "unchanged-by-the-test"
    finally:
        other.stop()
    free = LogManager(run_dir, create_dirs=False)
    assert free.results_dir == run_dir and free.shared_dir_note is None


def test_a_second_log_manager_of_one_process_is_a_second_run(run_dir):
    from gbench.utils.log_manager import LogManager
    first = LogManager(run_dir)
    second = LogManager(run_dir)
    assert first.results_dir == run_dir
    assert second.results_dir.parent == run_dir and second.results_dir != run_dir
    assert second.shared_dir_note.startswith("in use by pid %d on " % os.getpid())
    first.release_run_lock()                     # the first run is over: continued in place
    assert LogManager(run_dir).results_dir == run_dir


def test_without_file_locks_no_run_directory_is_shared(run_dir, monkeypatch):
    from gbench.utils import log_manager
    monkeypatch.setattr(log_manager, "fcntl", None)
    lm = log_manager.LogManager(run_dir)
    assert lm.results_dir.parent == run_dir and lm.results_dir != run_dir
    assert lm.shared_dir_note.startswith("cannot lock")
    dry = log_manager.LogManager(run_dir, create_dirs=False)      # the dry run says the same
    assert dry.results_dir != run_dir and dry.shared_dir_note.startswith("cannot lock")


def test_a_fresh_run_directory_is_locked_for_a_later_run(tmp_path, monkeypatch):
    from gbench.utils.log_manager import LogManager, RUN_LOCK_NAME, run_dir_in_use
    monkeypatch.setenv("GBENCH_RESULTS_DIR", "unchanged-by-the-test")
    lm = LogManager(tmp_path / "results")
    assert (lm.results_dir / RUN_LOCK_NAME).is_file()
    assert run_dir_in_use(lm.results_dir).startswith("in use by pid %d" % os.getpid())
    lm.release_run_lock()
    assert run_dir_in_use(lm.results_dir) == ""


def _cli_main(monkeypatch, argv):
    from gbench import cli
    from gbench.core import models
    monkeypatch.setenv("GBENCH_SHARD", "")
    monkeypatch.delenv("GBENCH_SHARD")
    # the model's config.json is not fetched (no network)
    monkeypatch.setattr(models, "_enrich_from_config", lambda model: None)
    try:
        return cli.main(argv)
    except SystemExit as e:
        return e.code


def test_a_cli_dry_run_on_a_directory_in_use_creates_nothing(run_dir, monkeypatch, caplog):
    other = _OtherRun(run_dir)
    try:
        before = sorted(p.name for p in run_dir.iterdir())
        caplog.set_level("INFO")
        code = _cli_main(monkeypatch, [
            "--models", "org/m", "--evals", "gsm8k", "--remote-endpoint",
            "http://127.0.0.1:9/v1", "--tokenizer", "org/m", "--max-output-tokens", "1024",
            "--dry-run", "--results-dir", str(run_dir)])
        assert code == 0
        assert sorted(p.name for p in run_dir.iterdir()) == before
        assert "(dry run: not created)" in caplog.text
        assert "this run writes its own directory" in caplog.text
    finally:
        other.stop()


def test_a_run_that_finds_the_directory_in_use_records_why_in_its_metadata(run_dir,
                                                                           monkeypatch):
    from gbench import utils
    from gbench.cli import EXIT_HARNESS_ERROR
    # the endpoint check fails without contacting anything; metadata.json is written before it
    monkeypatch.setattr(utils, "verify_endpoint_functional",
                        lambda *a, **k: (False, "stubbed: not contacted", None))
    other = _OtherRun(run_dir)
    try:
        code = _cli_main(monkeypatch, [
            "--models", "org/m", "--evals", "gsm8k", "--remote-endpoint",
            "http://127.0.0.1:9/v1", "--tokenizer", "org/m", "--max-output-tokens", "1024",
            "--results-dir", str(run_dir)])
        assert code == EXIT_HARNESS_ERROR
        assert json.loads((run_dir / "metadata.json").read_text()) == {"node": 1}
        mine = [p for p in run_dir.iterdir() if (p / "metadata.json").is_file()]
        assert len(mine) == 1
        meta = json.loads((mine[0] / "metadata.json").read_text())
        assert meta["results_dir_not_shared"].startswith("in use by pid %d" % other.proc.pid)
    finally:
        other.stop()


# --------------------------------------------------------------------------- #
# the generic merge
# --------------------------------------------------------------------------- #
def _shard_reason(i, n):
    from gbench.runners.eval_suites.base import SHARD_REASON
    return SHARD_REASON.format(label=f"{i}/{n}", index=i, count=n)


def _shard(i, n, traces, **extra):
    correct = sum(1 for t in traces if t.get("is_correct"))
    res = {"benchmark_type": "eval", "eval_name": "gsm8k", "model_short": "m", "format": "hf",
           "shard": f"{i}/{n}", "status": "success", "sample_traces": traces,
           "total_questions": len(traces), "correct_answers": correct,
           "accuracy": round(correct / len(traces) * 100, 2) if traces else None,
           "leaderboard_comparable": False,
           "leaderboard_comparable_reason": _shard_reason(i, n)}
    res.update(extra)
    return res


def _t(key, gi, ok, **extra):
    return {"sample_key": key, "global_idx": gi, "attempt_idx": 0, "is_correct": ok,
            "category": "c", **extra}


def test_rows_with_identical_content_are_all_kept():
    from gbench.merge import generic_merge
    # the same question at positions 0, 1 and 3 (a dataset with duplicate rows)
    s1 = _shard(1, 2, [_t("sha256:dup", 0, True), _t("sha256:x", 2, False)])
    s2 = _shard(2, 2, [_t("sha256:dup", 1, False), _t("sha256:dup", 3, True)])
    merged = generic_merge("gsm8k", [s1, s2])
    assert merged["total_questions"] == 4 and merged["correct_answers"] == 2
    assert merged["accuracy"] == 50.0
    assert [t["global_idx"] for t in merged["sample_traces"]] == [0, 1, 2, 3]


def test_an_item_in_two_shards_is_refused():
    from gbench.merge import MergeError, generic_merge
    s1 = _shard(1, 2, [_t("id:q1", 0, True)])
    s2 = _shard(2, 2, [_t("id:q1", 0, True), _t("id:q2", 1, False)])
    with pytest.raises(MergeError, match=r"'id:q1', global_idx 0 .* in shard 1/2 and in shard "
                                         r"2/2: the shards overlap"):
        generic_merge("gsm8k", [s1, s2])
    # without positions, the same sample_key in two shards is the overlap
    s1 = _shard(1, 2, [{"sample_key": "id:q1", "is_correct": True}])
    s2 = _shard(2, 2, [{"sample_key": "id:q1", "is_correct": True}])
    with pytest.raises(MergeError, match="the shards overlap"):
        generic_merge("gsm8k", [s1, s2])


def test_no_field_of_the_merge_is_one_shards_value():
    from gbench.merge import generic_merge
    s1 = _shard(1, 2, [_t("id:a", 0, True), _t("id:b", 2, False)],
                failed_requests=1, response_health={"ok": 2, "truncated": 1},
                gemini_models_answered={"gemini-3.8-flash": 2},
                judge_models_used={"gemini-3.8-flash": 2}, self_judge=False,
                self_judge_models=[], decode_tok_s={"observed_median": 41.0},
                tool_rounds={"samples_using_tools": 2, "total_rounds": 5,
                             "per_turn_max_tokens": 8192, "loop_ends": {"answered": 2}},
                duration_s=10.0, temperature=0.0, max_tokens_sent=[1024],
                finished_at="2026-09-28T10:00:00+00:00")
    s2 = _shard(2, 2, [_t("id:c", 1, True)],
                failed_requests=0, response_health={"ok": 1},
                gemini_models_answered={"gemini-3.7-flash": 1},
                judge_models_used={"gemini-3.7-flash": 1}, self_judge=False,
                self_judge_models=[], decode_tok_s={"observed_median": 39.5},
                tool_rounds={"samples_using_tools": 1, "total_rounds": 1,
                             "per_turn_max_tokens": 8192, "loop_ends": {"answered": 1}},
                duration_s=12.5, temperature=0.0, max_tokens_sent=[512, 1024],
                finished_at="2026-09-28T11:00:00+00:00")
    m = generic_merge("gsm8k", [s1, s2])
    assert m["gemini_models_answered"] == {"gemini-3.8-flash": 2, "gemini-3.7-flash": 1}
    assert m["judge_models_used"] == {"gemini-3.8-flash": 2, "gemini-3.7-flash": 1}
    assert m["failed_requests"] == 1 and m["response_health"] == {"ok": 3, "truncated": 1}
    assert m["tool_rounds"] == {"loop_ends": {"answered": 3}, "per_turn_max_tokens": 8192,
                                "samples_using_tools": 3, "total_rounds": 6}
    assert m["duration_s"] == 22.5 and m["duration_s_by_shard"] == {"1/2": 10.0, "2/2": 12.5}
    assert m["max_tokens_sent"] == [512, 1024] and m["temperature"] == 0.0
    assert m["finished_at"] == "2026-09-28T11:00:00+00:00"
    # a per-shard measurement no rule combines keeps no shard's value, each shard's is recorded
    assert "decode_tok_s" not in m
    assert m["merge_unreconciled_fields"]["decode_tok_s"] == {
        "1/2": {"observed_median": 41.0}, "2/2": {"observed_median": 39.5}}
    assert m["accuracy"] == round(2 / 3 * 100, 2) and m["leaderboard_comparable"] is True


def test_an_empty_shard_sets_no_field_of_the_merge():
    from gbench.merge import generic_merge
    empty = {"eval_name": "gsm8k", "shard": "1/2", "status": "success", "accuracy": None,
             "empty_shard": True, "sample_traces": [],
             "leaderboard_comparable_reason": "empty shard 1/2: no items on this node"}
    full = _shard(2, 2, [_t("id:a", 1, True)], temperature=0.0, scoring_mode="deterministic")
    m = generic_merge("gsm8k", [empty, full])
    assert m["temperature"] == 0.0 and m["scoring_mode"] == "deterministic"
    assert "merge_unreconciled_fields" not in m and m["accuracy"] == 100.0
    assert m["leaderboard_comparable"] is True


def test_a_suites_own_merge_records_every_shards_gemini_models(monkeypatch):
    from gbench.merge import merge_suite
    mod = types.ModuleType("gbench.runners.eval_suites.w4_fake_suite")
    mod.merge_shards = lambda results: dict(results[0], accuracy=75.0)   # copies shard 1
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    a = {"shard": "1/2", "gemini_models_answered": {"gemini-3.8-flash": 3},
         "judge_models_used_by_group": {"G1": {"gemini-3.8-flash": 3}}, "self_judge": False}
    b = {"shard": "2/2", "gemini_models_answered": {"gemini-3.7-flash": 2},
         "judge_models_used_by_group": {"G1": {"gemini-3.7-flash": 1},
                                        "G2": {"gemini-3.7-flash": 1}}, "self_judge": True}
    m = merge_suite("w4_fake_suite", [a, b])
    assert m["accuracy"] == 75.0
    assert m["gemini_models_answered"] == {"gemini-3.8-flash": 3, "gemini-3.7-flash": 2}
    assert m["judge_models_used_by_group"] == {
        "G1": {"gemini-3.8-flash": 3, "gemini-3.7-flash": 1}, "G2": {"gemini-3.7-flash": 1}}
    assert m["self_judge"] is True


# --------------------------------------------------------------------------- #
# merge_runs and `gbench --merge`
# --------------------------------------------------------------------------- #
def _write(root, i, n, traces, commit="abc"):
    d = Path(root) / f"node-{i}" / "09-28-000000"
    d.mkdir(parents=True)
    (d / "eval_gsm8k_m_hf.json").write_text(json.dumps(_shard(i, n, traces)))
    meta = {"gbench": {"commit": commit}, "eval_limit": None,
            "conditions": [{"pillar": "evals", "model": "m", "format": "hf",
                            "scaffold": {"pinned": {"x": 1, "shard": f"{i}/{n}"}},
                            "subject": {"subject_id": "sb_1"}}]}
    (d / "metadata.json").write_text(json.dumps(meta))
    return d.parent


def test_a_refused_or_dry_merge_writes_nothing(tmp_path):
    from gbench.merge import MergeError, merge_runs
    d1 = _write(tmp_path / "a", 1, 2, [_t("id:a", 0, True)])
    with pytest.raises(MergeError, match="missing shard"):
        merge_runs([d1], tmp_path / "out")
    assert not (tmp_path / "out").exists()
    d2 = _write(tmp_path / "a", 2, 2, [_t("id:b", 1, False)])
    planned = merge_runs([d1, d2], tmp_path / "out", dry_run=True)
    assert [p.name for p in planned] == ["eval_gsm8k_m_hf.json"]
    assert not (tmp_path / "out").exists()
    written = merge_runs([d1, d2], tmp_path / "out")
    merged = json.loads(written[0].read_text())
    assert merged["total_questions"] == 2 and merged["merged_shards"] == 2
    meta = json.loads((tmp_path / "out" / "metadata.json").read_text())
    assert meta["gbench"] == {"commit": "abc"} and meta["suites"]["gsm8k"]["shards"] == 2


def test_the_cli_merge_dry_run_creates_nothing(tmp_path, monkeypatch, capsys):
    d1 = _write(tmp_path / "a", 1, 2, [_t("id:a", 0, True)])
    d2 = _write(tmp_path / "a", 2, 2, [_t("id:b", 1, False)])
    out = tmp_path / "r"
    assert _cli_main(monkeypatch, ["--merge", str(d1), str(d2), "--results-dir", str(out),
                                   "--dry-run"]) == 0
    assert not out.exists()
    assert "nothing was written" in capsys.readouterr().out


def test_merges_started_in_the_same_second_get_their_own_directory(tmp_path, monkeypatch):
    from gbench import cli
    d1 = _write(tmp_path / "a", 1, 2, [_t("id:a", 0, True)])
    d2 = _write(tmp_path / "a", 2, 2, [_t("id:b", 1, False)])
    monkeypatch.setattr(cli.time, "strftime", lambda fmt, *a: "09-28-120000")
    out = tmp_path / "r"
    for _ in range(2):
        assert _cli_main(monkeypatch, ["--merge", str(d1), str(d2), "--results-dir",
                                       str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["merged-09-28-120000",
                                                     "merged-09-28-120000-2"]


# --------------------------------------------------------------------------- #
# mmlu and mmlu_redux: the merge of a run's shards equals the unsharded run
# --------------------------------------------------------------------------- #
def _fake_datasets(monkeypatch, rows_by_config):
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda name, config=None, split=None, revision=None: [
        dict(r) for r in rows_by_config(name, config, split)]
    monkeypatch.setitem(sys.modules, "datasets", fake)


def _run(fn, shard, monkeypatch, **kw):
    if shard:
        monkeypatch.setenv("GBENCH_SHARD", shard)
    else:
        monkeypatch.delenv("GBENCH_SHARD", raising=False)
    res = fn("m", "http://127.0.0.1:9/v1", 1, **kw)
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    res["eval_limit"] = kw.get("limit")               # stamped by the eval runner
    return json.loads(json.dumps(res, default=str))


_BOOKKEEPING = {"duration_s", "duration_s_by_shard", "shard", "shard_index", "shard_count",
                "shard_applied"}


def _differences(full, merged):
    def plain(res):
        res = {k: v for k, v in res.items() if k not in _BOOKKEEPING}
        res["sample_traces"] = [{k: v for k, v in t.items() if k != "sample_idx"}
                                for t in res.get("sample_traces") or []]
        return res
    a, b = plain(full), plain(merged)
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def _redux_setup(monkeypatch, non_letter_questions):
    from gbench.runners.eval_suites import mmlu_redux as R
    from gbench.runners.eval_suites.base import Reply
    rows = [{"question": f"Q{i}", "choices": ["a", "b", "c", "d"], "answer": 0,
             "error_type": "ok"} for i in range(30)]
    subject = R.MMLU_REDUX_SUBJECTS[0]
    _fake_datasets(monkeypatch, lambda name, config, split:
                   rows if (name == R.DATASET and config == subject) else [])
    monkeypatch.setattr(R, "_dev_blocks", lambda: {subject: []})
    monkeypatch.setattr(R, "helm_prompt", lambda subj, blocks, q, choices: f"{subj}:{q}")

    async def _gen(base_url, model_name, prompts, concurrency, temperature, **k):
        out = []
        for p in prompts:
            q = p.split(":", 1)[1]
            if q == "Q5":
                out.append(Reply(None, None, None, None, "HTTP 500: boom"))
            elif q in non_letter_questions:
                out.append(Reply("The answer is", None, "stop", None, None))
            else:
                out.append(Reply("A" if int(q[1:]) % 3 else "B", None, "stop", None, None))
        return out
    monkeypatch.setattr(R, "_generate_all", _gen)
    return R


def test_mmlu_redux_shards_merge_to_the_unsharded_run(monkeypatch):
    from gbench.merge import merge_suite
    # Q0 and Q2 are not a bare letter: 2 of shard 1/2's 15 rows (over the 10% threshold), none of
    # shard 2/2's, 2 of the 29 rows read in the whole run (Q5 got no reply; not over it)
    R = _redux_setup(monkeypatch, {"Q0", "Q2"})
    full = _run(R.run_mmlu_redux, None, monkeypatch)
    shards = [_run(R.run_mmlu_redux, f"{i}/2", monkeypatch) for i in (1, 2)]
    assert "not a bare option letter" in shards[0]["leaderboard_comparable_reason"]
    merged = merge_suite("mmlu_redux", shards)
    assert _differences(full, merged) == []
    assert "not a bare option letter" not in merged["leaderboard_comparable_reason"]
    assert merged["subject_mean_accuracy"] == full["subject_mean_accuracy"]
    assert merged["status"] == full["status"] == "completed_with_errors"
    assert [t["global_idx"] for t in merged["sample_traces"]] == list(range(30))


def test_mmlu_redux_traces_carry_their_position_in_the_scored_rows(monkeypatch):
    R = _redux_setup(monkeypatch, set())
    monkeypatch.setenv("GBENCH_SHARD", "2/3")
    items = R._load_mmlu_redux_items(limit=None)
    assert [it["global_idx"] for it in items] == list(range(1, 30, 3))


def _mmlu_setup(monkeypatch):
    from gbench.runners.eval_suites import mmlu
    rows = [{"subject": "anatomy" if i % 2 else "virology", "question": f"Q{i}",
             "choices": ["a", "b", "c", "d"], "answer": i % 4} for i in range(12)]
    _fake_datasets(monkeypatch, lambda name, config, split: rows)

    class _Tok:
        bos_token_id = None

        def encode(self, text):
            return [0] * (len(text) // 4)
    monkeypatch.setattr(mmlu, "_load_tokenizer", lambda name: _Tok())

    async def _score_all(base_url, model_name, enc, samples, concurrency,
                         temperature=0.0):
        out = []
        for s in samples:
            q = int(re.search(r"Q(\d+)", s["context"]).group(1))
            if q == 7:
                out.append({"loglikelihoods": None, "letters_missing": None,
                            "top_logprobs_returned": None, "pred": None, "is_correct": False,
                            "status": "FAILED", "error": "HTTP 500"})
                continue
            pred = 0 if q % 3 else 1
            out.append({"loglikelihoods": [0.0] * 4, "letters_missing": ["D"] if q == 4 else [],
                        "top_logprobs_returned": 20 if q == 4 else 100, "pred": pred,
                        "is_correct": pred == s["gold"], "status": "OK", "error": None})
        return out
    monkeypatch.setattr(mmlu, "_score_all", _score_all)
    return mmlu


def test_mmlu_shards_merge_to_the_unsharded_run(monkeypatch):
    from gbench.merge import merge_suite
    mmlu = _mmlu_setup(monkeypatch)
    full = _run(mmlu.run_mmlu, None, monkeypatch, eval_n_shot=0)
    for n in (2, 3):
        shards = [_run(mmlu.run_mmlu, f"{i}/{n}", monkeypatch, eval_n_shot=0)
                  for i in range(1, n + 1)]
        merged = merge_suite("mmlu", shards)
        assert _differences(full, merged) == [], n
    reason = merged["leaderboard_comparable_reason"]
    assert "1 question(s) got no usable reply" in reason and "0-shot" in reason
    assert [t["global_idx"] for t in merged["sample_traces"]] == list(range(12))


def test_mmlu_merge_keeps_a_reason_the_eval_runner_stamped(monkeypatch):
    from gbench.merge import merge_suite
    mmlu = _mmlu_setup(monkeypatch)
    shards = [_run(mmlu.run_mmlu, f"{i}/2", monkeypatch) for i in (1, 2)]
    stamped = "a reason the eval runner added; with a semicolon inside"
    shards[1]["leaderboard_comparable_reason"] += "; " + stamped
    merged = merge_suite("mmlu", shards)
    assert merged["leaderboard_comparable_reason"].endswith(stamped)
    assert merged["leaderboard_comparable"] is False


# --------------------------------------------------------------------------- #
# license headers of the test files
# --------------------------------------------------------------------------- #
#: The maintainer's plugin tests, not checked here.
_PLUGIN_TESTS = {"test_swe_bench_pro_agentic.py", "test_jetski_agent.py",
                 "test_slashbin_agent.py"}

#: Test files exempt from the header check. Every test file now carries the header; no file may
#: be added to this set.
_WITHOUT_HEADER_BEFORE_THE_CHECK: frozenset = frozenset()

_HEADER = re.compile(r"\A(?:#![^\n]*\n)?(?:# -\*- coding: utf-8 -\*-\n)?"
                     r"# Copyright \d{4} Google LLC\n#\n"
                     r"# Licensed under the Apache License, Version 2\.0 \(the \"License\"\);\n")


def _test_files():
    return sorted(p for p in (REPO / "tests").rglob("*.py")
                  if ".bak" not in p.name and "__pycache__" not in p.parts
                  and p.name not in _PLUGIN_TESTS)


def test_every_test_file_carries_the_license_header():
    missing = [str(p.relative_to(REPO)) for p in _test_files()
               if p.name not in _WITHOUT_HEADER_BEFORE_THE_CHECK
               and not _HEADER.match(p.read_text(encoding="utf-8", errors="replace"))]
    assert missing == [], ("add the Apache license header (as in this file) to: "
                           + ", ".join(missing))
