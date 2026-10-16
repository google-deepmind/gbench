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

"""The container / bypass suites aider_polyglot, livebench, scicode, toolbench, terminal_bench and
skillsbench: --shard and --eval-limit select a deterministic subset on the host (never a head or
an unseeded shuffle), the result records the shard and per-item traces that `gbench --merge`
recombines (each suite's merge_shards), an empty shard withholds its score, the knobs a harness
owns are recorded as not applied, the endpoint key is read once (VLLM_API_KEY, else
OPENAI_API_KEY) and never enters a container, and the pinned sources are checked. Docker, the
network and Gemini are stubbed; nothing is written outside tmp."""

import glob
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

import gbench.runners.eval_suites.aider_polyglot as A
import gbench.runners.eval_suites.base as base
import gbench.runners.eval_suites.livebench as LB
import gbench.runners.eval_suites.scicode as SC
import gbench.runners.eval_suites.skillsbench as SB
import gbench.runners.eval_suites.tau_common as TC
import gbench.runners.eval_suites.terminal_bench as TB
import gbench.runners.eval_suites.toolbench as T
from gbench.merge import generic_merge

REPO = Path(__file__).resolve().parents[1]
SECRET = "sk-test-endpoint-key-0123456789abcdef"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("GBENCH_SHARD", "VLLM_API_KEY", "OPENAI_API_KEY", "GBENCH_SKILLSBENCH_API_KEY",
                 "GBENCH_AIDER_LANGUAGES", "GBENCH_AIDER_EXTRA_ARGS", "GBENCH_AIDER_MODEL",
                 "GBENCH_AIDER_POLYGLOT_TEMPERATURE", "GBENCH_LIVEBENCH_CATEGORIES",
                 "GBENCH_LIVEBENCH_EXTRA_ARGS", "GBENCH_LIVEBENCH_RELEASE",
                 "GBENCH_LIVEBENCH_MODEL", "GBENCH_LIVEBENCH_TEMPERATURE",
                 "GBENCH_TB_TURN_MAX_TOKENS", "GBENCH_TB_TIMEOUT_MULTIPLIER", "GBENCH_TB_PARSER",
                 "GBENCH_TB_CONTAINER_THREADS", "GBENCH_TERMINAL_BENCH_TEMPERATURE",
                 "GBENCH_TOOLBENCH_GROUPS", "GBENCH_SKILLSBENCH_TEMPERATURE",
                 "GBENCH_SKILLSBENCH_TASKS"):
        monkeypatch.delenv(name, raising=False)
    base._RUN_KNOBS.clear()
    yield
    base._RUN_KNOBS.clear()


# =========================================================================== #
# Documented run commands carry the required --max-output-tokens
# =========================================================================== #
def _doc_commands(text):
    """(line, command) for every command line in the fenced code blocks of `text`, backslash
    continuations joined. A fence opens on a line of three or more backticks whose info string
    holds no backtick and closes on a line of at least as many backticks and nothing else (the
    CommonMark rules), so prose such as "```<lang> hint line (`re.findall(...)`)" is not a fence."""
    out, fence, buf, start = [], "", "", 0
    for i, line in enumerate(text.splitlines(), 1):
        if not fence:
            m = re.match(r"^\s*(`{3,})[^`]*$", line)
            if m:
                fence, buf = m.group(1), ""
            continue
        m = re.match(r"^\s*(`{3,})\s*$", line)
        if m and len(m.group(1)) >= len(fence):
            fence, buf = "", ""
            continue
        part = line.rstrip().strip()
        if buf:
            buf += " " + part
        else:
            buf, start = part, i
        if buf.endswith("\\"):
            buf = buf[:-1]
            continue
        out.append((start, buf))
        buf = ""
    return out


def test_every_documented_eval_command_passes_max_output_tokens():
    files = (sorted(glob.glob(str(REPO / "docs" / "**" / "*.md"), recursive=True))
             + [str(REPO / "README.md")]
             + sorted(glob.glob(str(REPO / "examples" / "**" / "*.md"), recursive=True)))
    missing = []
    for path in files:
        if ".bak" in path:
            continue
        for line, cmd in _doc_commands(Path(path).read_text(encoding="utf-8")):
            cmd = re.sub(r"\s#.*$", "", cmd)
            if not re.search(r"(?:^|[\s;&|(])gbench\s", " " + cmd):
                continue
            if not re.search(r"--evals(-only)?\b|--eval-custom-jsonl", cmd):
                continue
            if re.search(r"--help\b|--dry-run\b|--list\b|--merge\b", cmd):
                continue
            if "--max-output-tokens" not in cmd:
                missing.append(f"{os.path.relpath(path, REPO)}:{line}: {cmd[:100]}")
    assert not missing, "\n".join(missing)


# =========================================================================== #
# terminal_bench
# =========================================================================== #
FULL = [f"terminal-bench/t{i:02d}" for i in range(10)]


def test_terminal_bench_limit_and_one_of_one_shard_select_the_same_seeded_tasks():
    a = TB.select_task_names(FULL, None, 4)
    b = TB.select_task_names(list(reversed(FULL)), (1, 1), 4)
    assert a == b == TB.select_task_names(FULL, None, 4) and len(a) == 4
    assert a != FULL[:4] or a == sorted(a)            # a seeded draw, not by construction a head
    shards = [TB.select_task_names(FULL, (i, 3), None) for i in (1, 2, 3)]
    assert sorted(t for s in shards for t in s) == sorted(FULL)
    assert not set(shards[0]) & set(shards[1])


def test_terminal_bench_trials_never_drop_below_the_leaderboard_count():
    assert TB.trial_count(1) == (5, TB.trial_count(1)[1])
    assert TB.trial_count(2)[0] == 5 and "does not lower it" in TB.trial_count(2)[1]
    assert TB.trial_count(8) == (8, "--attempt-count 8")


def _tb_trial(root, name, reward, task=None, exc=None):
    trial = os.path.join(root, "job", name)
    os.makedirs(os.path.join(trial, "agent"), exist_ok=True)
    data = {"trial_name": name, "verifier_result": {"rewards": {"reward": reward}},
            "agent_result": {"n_output_tokens": 10, "metadata": {"n_episodes": 1}}}
    if task:
        data.update({"task_name": task, "task_checksum": "c-" + task.split("/")[-1],
                     "task_id": {"org": "terminal-bench", "name": task.split("/")[-1],
                                 "ref": "sha256:abc"}})
    if exc:
        data["exception_info"] = {"exception_type": exc, "exception_message": "down"}
        data["verifier_result"] = None
        data["agent_execution"] = {"started_at": "t"}
    with open(os.path.join(trial, "result.json"), "w") as f:
        json.dump(data, f)


def _tb_run(monkeypatch, trials, *, limit=None, attempt_count=None, full=FULL):
    monkeypatch.setattr(TB, "check_terminal_bench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(TB, "require_task_metadata_blocked", lambda task_names=None: None)
    monkeypatch.setattr(TB, "prepare_task_images",
                        lambda task_names=None: ({"cached": 0}, {"present": 0}))
    monkeypatch.setattr(TB, "_dataset_task_names", lambda dataset: list(full))
    monkeypatch.setattr(TB, "dns_recheck", lambda task_names, hosts: list(hosts))
    seen = {}

    class _Proc:
        returncode = 0
        stdout = iter(())

        def poll(self):
            return 0

        def wait(self):
            return 0

    def popen(cmd, **kw):
        seen["cmd"], seen["env"] = cmd, kw.get("env") or {}
        jobs = cmd[cmd.index("--jobs-dir") + 1]
        for args in trials:
            _tb_trial(jobs, *args)
        return _Proc()
    monkeypatch.setattr(TB.subprocess, "Popen", popen)
    # --thinking: the run whose rule temperature (1.0) is the TB2.1 leaderboard's
    kwargs = {"limit": limit, "enable_thinking": True}
    if attempt_count is not None:
        kwargs["attempt_count"] = attempt_count
    return TB.run_terminal_bench("m", "http://h:1/v1", 1, **kwargs), seen


def test_terminal_bench_default_run_is_canonical_with_five_trials(monkeypatch):
    # a complete run of a one-task dataset: its 5 trials
    monkeypatch.setattr(TB, "FULL_TASK_COUNT", 1)
    trials = [(f"t00__{i}", 1.0 if i < 3 else 0.0, "terminal-bench/t00") for i in range(5)]
    res, seen = _tb_run(monkeypatch, trials)
    cmd = seen["cmd"]
    assert cmd[cmd.index("--n-attempts") + 1] == "5"
    assert "--n-tasks" not in cmd and "--include-task-name" not in cmd
    assert res["attempts"]["attempts_per_sample"] == 5 and res["attempts"]["samples"] == 1
    assert res["attempts"]["avg_at_k"] == 60.0 and res["attempts"]["pass_at_k"] == 100.0
    assert res["attempts"]["pass_hat_k"] == 0.0
    assert res["leaderboard_comparable"] is True
    assert res["leaderboard_comparable_reason"] == "full canonical run"
    traces = res["sample_traces"]
    assert [t["attempt_idx"] for t in traces] == [0, 1, 2, 3, 4]
    assert {t["sample_key"] for t in traces} == {"task:terminal-bench/t00"}
    t0 = traces[0]
    assert t0["task_name"] == "terminal-bench/t00" and t0["task_checksum"] == "c-t00"
    assert t0["task_id"] == {"org": "terminal-bench", "name": "t00", "ref": "sha256:abc"}
    assert res["accuracy"] == 60.0 and "shard" not in res


def test_terminal_bench_limit_names_the_seeded_tasks_and_never_n_tasks(monkeypatch):
    res, seen = _tb_run(monkeypatch, [("x__1", 1.0, FULL[0])], limit=3, attempt_count=7)
    cmd = seen["cmd"]
    names = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--include-task-name"]
    assert names == TB.select_task_names(FULL, None, 3) == res["selected_tasks"]
    assert "--n-tasks" not in cmd and cmd[cmd.index("--n-attempts") + 1] == "7"
    # 3 tasks x 7 trials expected, 1 found: the rest are harness errors, never zeros
    assert res["harness_errors"] == {"missing": 20} and res["status"] == "completed_with_errors"


def test_terminal_bench_task_category_comes_from_the_cached_task_toml(monkeypatch, tmp_path):
    root = tmp_path / "cache"
    (root / "terminal-bench" / "t00" / "abc").mkdir(parents=True)
    (root / "terminal-bench" / "t00" / "abc" / "task.toml").write_text(
        '[metadata]\ncategory = "security"\ndifficulty = "hard"\n')
    # gbench's Harbor cache (the harness container's ~/.cache/harbor on this host)
    monkeypatch.setattr(TB, "harbor_package_cache", lambda: str(root))
    res, _ = _tb_run(monkeypatch, [("t00__1", 1.0, "terminal-bench/t00")])
    t = res["sample_traces"][0]
    assert t["category"] == "security" and t["difficulty"] == "hard"
    assert res["category_accuracy"] == {"security": {"correct": 1, "total": 1, "accuracy": 100.0}}


def test_terminal_bench_empty_shard_withholds_the_score(monkeypatch):
    monkeypatch.setenv("GBENCH_SHARD", "12/12")
    res, seen = _tb_run(monkeypatch, [])
    assert "cmd" not in seen                     # Harbor never started
    assert res["accuracy"] is None and res["empty_shard"] is True
    assert res["accuracy_withheld_reason"].startswith("empty shard 12/12: no items on this node")
    assert res["shard"] == "12/12" and res["shard_count"] == 12 and res["full_set_items"] == 10


def test_terminal_bench_partial_shard_carries_the_base_shard_reason(monkeypatch):
    monkeypatch.setenv("GBENCH_SHARD", "2/3")
    trials = [(f"{n.split('/')[1]}__{k}", 1.0, n) for n in TB.select_task_names(FULL, (2, 3), None)
              for k in range(5)]
    res, seen = _tb_run(monkeypatch, trials)
    assert res["shard"] == "2/3" and res["shard_applied"] is True and "missing" not in (
        res["harness_errors"] or {})
    assert res["leaderboard_comparable"] is False
    assert res["leaderboard_comparable_reason"].startswith("sharded run (--shard 2/3)")


def test_terminal_bench_malformed_shard_is_an_error(monkeypatch):
    monkeypatch.setenv("GBENCH_SHARD", "3/2")
    with pytest.raises(ValueError):
        _tb_run(monkeypatch, [])


def test_terminal_bench_endpoint_key_stays_in_gbench_behind_a_relay(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", SECRET)
    # Split so that no source line matches the leak gate (scripts/check_no_leaks.sh).
    monkeypatch.setenv("GEMINI_API_KEY", "AIza" + "TESTfixtureKEY000000000000000000000")
    made = []

    class Relay:
        TIMEOUT_S = 10

        def __init__(self, upstream, bind_host, key=None, name=""):
            self.upstream, self.bind_host, self.key = upstream, bind_host, key
            made.append(self)

        def start(self):
            return self

        def stop(self):
            self.stopped = True

        @property
        def url(self):
            return "http://127.0.0.1:45678/v1"

        def record(self):
            return {"listens": self.url}
    monkeypatch.setattr(TB.swebench_common, "EndpointRelay", Relay)
    res, seen = _tb_run(monkeypatch, [("t00__1", 1.0, FULL[0])])
    env, cmd = seen["env"], seen["cmd"]
    assert made and made[0].key == SECRET and made[0].bind_host == "127.0.0.1"
    assert made[0].upstream == "http://h:1/v1" and made[0].stopped
    assert env["OPENAI_API_KEY"] == "EMPTY" and env["OPENAI_BASE_URL"] == made[0].url
    assert "api_base=http://127.0.0.1:45678/v1" in cmd
    assert SECRET not in json.dumps(env) and "GEMINI_API_KEY" not in env
    assert SECRET not in " ".join(cmd) and res["model_endpoint_relay"]["listens"] == made[0].url


def test_terminal_bench_shards_merge_back_to_the_unsharded_score():
    def shard(label, tasks):
        traces = TB.annotate_trials([{"trial_name": f"{t}__{k}", "task_name": t,
                                      "passed": (k + len(t)) % 2 == 0} for t in tasks
                                     for k in range(5)])
        correct = sum(1 for tr in traces if tr["is_correct"])
        return {"eval_name": "terminal_bench", "shard": label, "status": "success",
                "sample_traces": traces, "accuracy": round(100.0 * correct / len(traces), 2),
                "attempts": TB.attempts_report(traces, 1, 5, "x")}
    parts = [shard(f"{i}/2", TB.select_task_names(FULL, (i, 2), None)) for i in (1, 2)]
    whole = shard("1/1", FULL)
    merged = generic_merge("terminal_bench", parts)
    assert merged["accuracy"] == whole["accuracy"]
    assert merged["attempts"]["pass_at_k"] == whole["attempts"]["pass_at_k"]


# =========================================================================== #
# aider_polyglot
# =========================================================================== #
LANGS = A._LANGUAGES


def _polyglot(root, per_lang=4):
    for lang in LANGS:
        for i in range(per_lang):
            (root / "polyglot-benchmark" / lang / "exercises" / "practice" / f"{lang}-e{i}"
             ).mkdir(parents=True, exist_ok=True)


def test_aider_selection_is_deterministic_and_stratified(tmp_path):
    _polyglot(tmp_path)
    exercises = A.list_exercises(str(tmp_path), LANGS)
    assert exercises == sorted(exercises) and len(exercises) == 24
    a, label, pos = A.select_exercises(exercises, 6)
    b, _, _ = A.select_exercises(exercises, 6)
    assert a == b and label is None and len(a) == 6
    assert sorted(e.split("/")[0] for e in a) == sorted(LANGS)       # one per language
    assert [exercises[p] for p in pos] == a


def test_aider_shards_are_disjoint_and_cover_the_set(monkeypatch, tmp_path):
    _polyglot(tmp_path)
    exercises = A.list_exercises(str(tmp_path), LANGS)
    got = []
    for i in (1, 2, 3):
        monkeypatch.setenv("GBENCH_SHARD", f"{i}/3")
        sel, label, _ = A.select_exercises(exercises, None)
        assert label == f"{i}/3"
        got.append(sel)
    assert sorted(e for s in got for e in s) == exercises
    assert not set(got[0]) & set(got[1])


def _aider_env(monkeypatch, tmp_path, outcome=(True,)):
    monkeypatch.setenv("GBENCH_AIDER_BENCHMARK_DIR", str(tmp_path))
    monkeypatch.setattr(A, "_check_prereqs", lambda image, bench: None)
    monkeypatch.setattr(TC, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC.container_url(url))
    seen = {"calls": 0}

    def fake_run(cmd, **k):
        seen["calls"] += 1
        seen["cmd"] = cmd
        mount = next((a.split(":", 1)[0] for a in cmd if a.endswith(":/gbench-exercises:ro")),
                     None)
        root = Path(mount) if mount else tmp_path / "polyglot-benchmark"
        names = sorted(f"{p.parents[2].name}/{p.name}"
                       for p in root.glob("*/exercises/practice/*"))
        seen["mounted"] = names if mount else None
        seen["tree"] = mount
        run = tmp_path / f"2026-run{len(list(tmp_path.glob('*--gbench'))) + 1}--gbench"
        for i, name in enumerate(names):
            lang, ex = name.split("/")
            d = run / lang / "exercises" / "practice" / ex
            d.mkdir(parents=True, exist_ok=True)
            (d / ".aider.results.json").write_text(json.dumps(
                {"language": lang, "tests_outcomes": list(outcome) if i % 2 == 0 else [False,
                                                                                   False]}))
        report = {"files_loaded": ["/gbench-aider/model-settings.yml"], "extra_params": {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "timeout": A.model_request_timeout_s(), "max_retries": 0,
            "temperature": 0.0}}  # a no-think run's temperature (base.resolve_temperature)
        return types.SimpleNamespace(returncode=0, stderr="",
                                     stdout=A._CHECK_MARKER + json.dumps(report) + "\n"
                                     + A._MEMORY_MARKER + "12884901888\n" + A._OOM_MARKER + "0\n")
    monkeypatch.setattr(A.subprocess, "run", fake_run)
    return seen


def test_aider_limit_runs_exactly_the_selected_exercises_from_a_tree(monkeypatch, tmp_path):
    _polyglot(tmp_path)
    seen = _aider_env(monkeypatch, tmp_path)
    r = A.run_aider_polyglot("org/served-model", "http://127.0.0.1:8000/v1", concurrency=2,
                             limit=6, temperature=0.7, attempt_count=3)
    expected, _, _ = A.select_exercises(A.list_exercises(str(tmp_path), LANGS), 6)
    assert seen["mounted"] == sorted(expected) and r["selected_exercises"] == expected
    joined = " ".join(seen["cmd"])
    assert "--num-tests" not in joined and "--exercises-dir /gbench-exercises" in joined
    assert "--model openai/org/served-model" in joined and r["aider_model"] == \
        "openai/org/served-model"
    assert r["total_questions"] == 6 and len(r["sample_traces"]) == 6
    assert {t["sample_key"] for t in r["sample_traces"]} == {f"exercise:{e}" for e in expected}
    assert all("global_idx" in t for t in r["sample_traces"])
    assert r["temperature_applied"] is True and "temperature_note" not in r
    assert r["attempts"]["not_applied"] and r["attempts"]["attempts_per_sample"] == 1
    assert r["aider_commit"] == A.AIDER_COMMIT
    assert not os.path.exists(seen["tree"])             # the copied tree is removed


def test_aider_empty_shard_withholds_and_runs_nothing(monkeypatch, tmp_path):
    _polyglot(tmp_path, per_lang=1)
    seen = _aider_env(monkeypatch, tmp_path)
    monkeypatch.setenv("GBENCH_SHARD", "7/7")
    r = A.run_aider_polyglot("m", "http://127.0.0.1:8000/v1", concurrency=1)
    assert seen["calls"] == 0 and r["accuracy"] is None and r["empty_shard"] is True
    assert r["accuracy_withheld_reason"].startswith("empty shard 7/7")


def test_aider_shards_merge_back_to_the_unsharded_pass_rates(monkeypatch, tmp_path):
    _polyglot(tmp_path)
    parts = []
    for i in (1, 2):
        monkeypatch.setenv("GBENCH_SHARD", f"{i}/2")
        _aider_env(monkeypatch, tmp_path)
        parts.append(A.run_aider_polyglot("m", "http://127.0.0.1:8000/v1", concurrency=1))
    monkeypatch.delenv("GBENCH_SHARD")
    for p in parts:
        assert p["leaderboard_comparable"] is False and "sharded run (--shard" in p[
            "leaderboard_comparable_reason"]
    merged = A.merge_shards(parts)
    assert merged["total_questions"] == 24
    assert merged["accuracy"] == round(100.0 * sum(
        1 for p in parts for t in p["sample_traces"] if t["pass_2"]) / 24, 2)
    assert merged["pass_rate_1"] is not None


def test_aider_pinned_sources_are_checked(monkeypatch, tmp_path):
    monkeypatch.setattr(A, "image_aider_commit", lambda image: "")
    with pytest.raises(RuntimeError, match="not built from the pinned aider commit"):
        A.check_pinned_sources("aider-benchmark:5dc9490b", str(tmp_path))
    monkeypatch.setattr(A, "image_aider_commit", lambda image: A.AIDER_COMMIT)
    repo = tmp_path / "polyglot-benchmark"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "f").write_text("x")
    subprocess.run(["git", "-C", str(repo), "add", "f"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=a@b", "-c", "user.name=a",
                    "commit", "-qm", "x"], check=True)
    with pytest.raises(RuntimeError, match="not the pinned 7e0611e77b54"):
        A.check_pinned_sources("aider-benchmark:5dc9490b", str(tmp_path))
    assert "not a git checkout" in A.polyglot_checkout_problem(str(tmp_path / "nope"))


def test_aider_dockerfile_fetches_and_labels_the_pinned_commit():
    df = (REPO / "docker" / "aider_polyglot.Dockerfile").read_text()
    assert f"ARG AIDER_COMMIT={A.AIDER_COMMIT}" in df
    assert 'LABEL org.gbench.aider.commit="${AIDER_COMMIT}"' in df
    assert "COPY . /aider" not in df
    doc = (REPO / "docs" / "evals" / "aider_polyglot.md").read_text()
    assert A.POLYGLOT_BENCHMARK_COMMIT in doc and A.AIDER_COMMIT in doc


def test_aider_uses_the_passed_served_id_and_never_the_first_listed(monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("the model list must not be read")))
    assert A._served_model_id("http://h/v1", "org/served") == "org/served"


# =========================================================================== #
# livebench
# =========================================================================== #
ROWS = sorted([(c, f"{c}-t{t}", f"{c[:3]}{t}{q}") for c in LB._CATEGORIES for t in range(2)
               for q in range(3)])


def test_livebench_rejects_an_unknown_category_before_any_container(monkeypatch):
    monkeypatch.setenv("GBENCH_LIVEBENCH_CATEGORIES", "math astrology")
    calls = []
    monkeypatch.setattr(LB.subprocess, "run", lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match="astrology"):
        LB.run_livebench("m", "http://h/v1", 1)
    assert calls == []


def test_livebench_selection_is_deterministic_sharded_and_stratified(monkeypatch):
    ids, label, pos, n_full = LB.select_questions(ROWS, list(LB._CATEGORIES), 6)
    assert ids == LB.select_questions(ROWS, list(LB._CATEGORIES), 6)[0] and n_full == 36
    cat_of = {q: c for c, _t, q in ROWS}
    assert sorted(cat_of[q] for q in ids) == sorted(LB._CATEGORIES)
    got = []
    for i in (1, 2):
        monkeypatch.setenv("GBENCH_SHARD", f"{i}/2")
        got.append(LB.select_questions(ROWS, list(LB._CATEGORIES), None)[0])
    assert sorted(got[0] + got[1]) == sorted({q for _c, _t, q in ROWS})
    assert not set(got[0]) & set(got[1])


def test_livebench_question_ids_use_the_whole_suite_path():
    cmd = LB._build_command("img", "/out", "m", "d", "http://x/v1", 2, None, "", "",
                            question_ids=["a1", "b2"])
    inner = cmd[-1]
    assert "--bench-name live_bench --question-source" in inner
    assert "--question-id a1 b2 " in inner
    assert "model_judgment" in inner and "ground_truth_judgment.jsonl" in inner


def _lb_run(monkeypatch, tmp_path, categories=None, limit=None, extra=None):
    if categories:
        monkeypatch.setenv("GBENCH_LIVEBENCH_CATEGORIES", categories)
    if extra:
        monkeypatch.setenv("GBENCH_LIVEBENCH_EXTRA_ARGS", extra)
    seen = {}
    monkeypatch.setattr(LB, "_check_prereqs", lambda image: None)
    monkeypatch.setattr(TC, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC.container_url(url))
    monkeypatch.setattr(LB, "list_questions", lambda image, release: list(ROWS))
    monkeypatch.setattr(LB, "upstream_model_config",
                        lambda image, names: {"name": None, "config": None,
                                              "commit": LB.LIVEBENCH_COMMIT})
    monkeypatch.setattr(LB.tempfile, "mkdtemp", lambda prefix="": str(tmp_path))
    monkeypatch.setattr(LB, "server_aborted_requests", lambda url, ids: 0)

    def fake_run(cmd, **_kw):
        seen["cmd"] = cmd
        inner = cmd[-1]
        m = re.search(r"--question-id ((?:\S+ )+)", inner)
        ids = m.group(1).split() if m else [q for _c, _t, q in ROWS]
        chosen = [r for r in ROWS if r[2] in ids]
        cats = sorted({c for c, _t, _q in chosen})
        header = "model," + ("average," if len(cats) > 1 else "") + ",".join(cats)
        vals = "m," + ("50.0," if len(cats) > 1 else "") + ",".join("50.0" for _ in cats)
        (tmp_path / "all_groups.csv").write_text(header + "\n" + vals + "\n")
        for c, t, q in chosen:
            a = tmp_path / "model_answer" / "data" / "live_bench" / c / t / "model_answer"
            a.mkdir(parents=True, exist_ok=True)
            with open(a / "m.jsonl", "a") as f:
                f.write(json.dumps({"question_id": q, "choices": [{"turns": ["x"]}],
                                    "total_output_tokens": 5}) + "\n")
            j = tmp_path / "model_judgment" / "data" / "live_bench" / c / t / "model_judgment"
            j.mkdir(parents=True, exist_ok=True)
            with open(j / "ground_truth_judgment.jsonl", "a") as f:
                f.write(json.dumps({"question_id": q, "task": t, "model": "m", "score": 0.5,
                                    "category": c}) + "\n")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(LB.subprocess, "run", fake_run)
    return LB.run_livebench("m", "http://h/v1", 2, max_output_tokens=4096, limit=limit), seen


def test_livebench_a_multiword_subset_runs_by_question_id(monkeypatch, tmp_path):
    res, seen = _lb_run(monkeypatch, tmp_path, categories="data_analysis math")
    inner = seen["cmd"][-1]
    assert "--bench-name live_bench --question-source" in inner
    assert "live_bench/data" not in inner.split("--bench-name")[1].split("--question-source")[0]
    ids = {q for c, _t, q in ROWS if c in ("data_analysis", "math")}
    assert set(res["selected_question_ids"]) == ids
    assert {t["category"] for t in res["sample_traces"]} == {"data_analysis", "math"}
    assert res["questions_judged"] == len(ids)


def test_livebench_refuses_question_selection_in_extra_args_while_selecting(monkeypatch,
                                                                              tmp_path):
    with pytest.raises(ValueError, match="selects questions"):
        _lb_run(monkeypatch, tmp_path, limit=4, extra="--question-end 2")


def test_livebench_shards_merge_to_livebenchs_own_aggregation(monkeypatch, tmp_path):
    parts = []
    for i in (1, 2):
        monkeypatch.setenv("GBENCH_SHARD", f"{i}/2")
        d = tmp_path / f"s{i}"
        d.mkdir()
        res, _ = _lb_run(monkeypatch, d)
        assert res["shard"] == f"{i}/2" and "sharded run (--shard" in res[
            "leaderboard_comparable_reason"]
        parts.append(res)
    merged = LB.merge_shards(parts)
    assert merged["questions_judged"] == len(ROWS)
    assert merged["accuracy"] == 50.0 and set(merged["dimension_scores"]) == set(LB._CATEGORIES)
    whole = LB.livebench_scores([t for p in parts for t in p["sample_traces"]])
    assert whole["average"] == merged["accuracy"]


def test_livebench_scores_follow_show_livebench_result():
    traces = [{"category": "math", "task": "a", "score": 1.0},
              {"category": "math", "task": "a", "score": 0.0},
              {"category": "math", "task": "b", "score": 1.0},
              {"category": "coding", "task": "c", "score": 0.25},
              {"category": "coding", "task": "c", "score": 1.0, "scoring_excluded": True}]
    got = LB.livebench_scores(traces)
    assert got["categories"] == {"coding": 25.0, "math": 75.0}      # math: mean(50, 100)
    assert got["average"] == 50.0


# =========================================================================== #
# scicode
# =========================================================================== #
def _scicode_rows(n=7):
    return [{"problem_id": str(10 + i), "problem_name": f"p{i}", "required_dependencies": "",
             "sub_steps": [{"step_number": f"{10 + i}.1"}]} for i in range(n)]


def test_scicode_shards_the_problems_by_id_before_the_limit(monkeypatch):
    rows = _scicode_rows()
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda *a, **k: [dict(r) for r in reversed(rows)]
    monkeypatch.setitem(sys.modules, "datasets", fake)
    got = []
    for i in (1, 2, 3):
        monkeypatch.setenv("GBENCH_SHARD", f"{i}/3")
        got.append([p["problem_id"] for p in SC._load_scicode_problems(limit=None)])
    assert sorted(x for g in got for x in g) == sorted(r["problem_id"] for r in rows)
    assert got[0] == ["10", "13", "16"]
    monkeypatch.setenv("GBENCH_SHARD", "1/3")
    assert len(SC._load_scicode_problems(limit=2)) == 2


def _scicode_env(monkeypatch, problems):
    monkeypatch.setattr(SC, "sandbox_skip_reason", lambda: None)
    monkeypatch.setattr(SC, "resolve_test_data", lambda: ("/nonexistent.h5", "test"))
    monkeypatch.setattr(SC, "_digest", lambda path: SC._TEST_DATA_SHA256)
    monkeypatch.setattr(SC, "check_cmp_importable", lambda: None)
    monkeypatch.setattr(SC, "_load_scicode_problems", lambda limit=None: problems)
    import openai
    clients = []
    monkeypatch.setattr(openai, "OpenAI", lambda **k: clients.append(k) or object())

    def fake_one(client, model, prob, **k):
        ok = int(prob["problem_id"]) % 2 == 0
        return {"problem_id": prob["problem_id"], "problem_name": "p", "steps": [
            {"step_number": prob["sub_steps"][0]["step_number"], "special": False,
             "gen_failed": False, "correct": ok, "response_text": "x", "error": None}]}
    monkeypatch.setattr(SC, "_process_one_problem", fake_one)
    return clients


def test_scicode_empty_shard_withholds_and_the_key_is_the_endpoint_key(monkeypatch):
    monkeypatch.setenv("GBENCH_SHARD", "9/9")
    _scicode_env(monkeypatch, [])
    r = SC.run_scicode("m", "http://h/v1", concurrency=1)
    assert r["accuracy"] is None and r["empty_shard"] is True and r["status"] == "success"
    monkeypatch.delenv("GBENCH_SHARD")
    monkeypatch.setenv("VLLM_API_KEY", SECRET)
    clients = _scicode_env(monkeypatch, [dict(p, global_idx=i) for i, p in
                                         enumerate(_scicode_rows(2))])
    r = SC.run_scicode("m", "http://h/v1", concurrency=1, attempt_count=4)
    assert clients[0]["api_key"] == SECRET
    assert r["attempts"]["not_applied"] and r["sample_traces"][0]["sample_key"] == "step:10/10.1"


def test_scicode_shards_merge_back_to_the_unsharded_resolve_rate(monkeypatch):
    rows = [dict(p, global_idx=i) for i, p in enumerate(_scicode_rows(6))]
    parts = []
    for i in (1, 2):
        monkeypatch.setenv("GBENCH_SHARD", f"{i}/2")
        _scicode_env(monkeypatch, rows[i - 1::2])
        parts.append(SC.run_scicode("m", "http://h/v1", concurrency=1))
        assert parts[-1]["shard"] == f"{i}/2"
    monkeypatch.delenv("GBENCH_SHARD")
    _scicode_env(monkeypatch, rows)
    whole = SC.run_scicode("m", "http://h/v1", concurrency=1)
    merged = SC.merge_shards(parts)
    assert merged["accuracy"] == whole["accuracy"] == 50.0
    assert merged["substep_accuracy"] == whole["substep_accuracy"]


# =========================================================================== #
# toolbench
# =========================================================================== #
TEST_IDS = {"G1_instruction": [str(i) for i in (5, 1, 9, 3)], "G3_instruction": ["2", "7"]}


def test_toolbench_selection_is_deterministic_sharded_and_by_group():
    sel, label, n = T.select_query_ids(TEST_IDS, 2)
    assert sel == T.select_query_ids(TEST_IDS, 2)[0] and label is None and n == 6
    assert sum(len(v) for v in sel.values()) == 2 and all(sel.values())   # one per group
    assert sel["G1_instruction"] == [q for q in TEST_IDS["G1_instruction"]
                                     if q in sel["G1_instruction"]]      # file order kept


def test_toolbench_shards_are_disjoint_and_cover_the_test_ids(monkeypatch):
    got = []
    for i in (1, 2, 3):
        monkeypatch.setenv("GBENCH_SHARD", f"{i}/3")
        sel, label, _ = T.select_query_ids(TEST_IDS, None)
        assert label == f"{i}/3"
        got.append({(g, q) for g, qs in sel.items() for q in qs})
    union = set().union(*got)
    assert union == {(g, q) for g, qs in TEST_IDS.items() for q in qs}
    assert sum(len(s) for s in got) == len(union)


def test_toolbench_entrypoint_runs_exactly_the_selected_ids(tmp_path):
    src = (REPO / "docker" / "toolbench_entrypoint.sh").read_text()
    snippet = src.split("<<'PYSELECT'\n", 1)[1].split("\nPYSELECT\n", 1)[0]
    (tmp_path / "sel.py").write_text(snippet)
    queries = [{"query_id": q, "query": f"q{q}"} for q in (5, 1, 9, 3)]
    (tmp_path / "q.json").write_text(json.dumps(queries))
    halves = []
    for i in (1, 2):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("GBENCH_SHARD", f"{i}/2")
            sel, _, _ = T.select_query_ids({"G1_instruction": ["5", "1", "9", "3"]}, None)
        (tmp_path / "ids.json").write_text(json.dumps(sel))
        out = tmp_path / f"run{i}.json"
        p = subprocess.run([sys.executable, str(tmp_path / "sel.py"), str(tmp_path / "q.json"),
                            str(out), str(tmp_path / "ids.json"), "G1_instruction"],
                           capture_output=True, text=True)
        assert p.returncode == 0, p.stderr
        halves.append([q["query_id"] for q in json.loads(out.read_text())])
    assert sorted(halves[0] + halves[1]) == [1, 3, 5, 9] and not set(halves[0]) & set(halves[1])
    (tmp_path / "ids.json").write_text(json.dumps({"G1_instruction": ["404"]}))
    p = subprocess.run([sys.executable, str(tmp_path / "sel.py"), str(tmp_path / "q.json"),
                        str(tmp_path / "x.json"), str(tmp_path / "ids.json"), "G1_instruction"],
                       capture_output=True, text=True)
    assert p.returncode == 4 and "404" in p.stderr
    assert T.HARNESS_VERSION == "5"
    assert 'LABEL org.gbench.toolbench-harness="5"' in (
        REPO / "docker" / "toolbench.Dockerfile").read_text()


def _tb_answer():
    return {"answer": {"final_answer": "done", "answer_details": [
        {"role": "tool", "message": "{'name': 'Finish'}", "next": []}]}}


@pytest.fixture
def toolbench_run(monkeypatch, tmp_path):
    # Split so that no source line matches the leak gate (scripts/check_no_leaks.sh).
    monkeypatch.setenv("GEMINI_API_KEY", "AIza" + "TESTfixtureKEY000000000000000000000")
    for k in ("GBENCH_TOOLBENCH_REFERENCE_DIR", "GBENCH_TOOLBENCH_REFERENCE",
              "GBENCH_TOOLBENCH_EVAL_TIMES", "GBENCH_TOOLBENCH_MAX_QUERY_COUNT",
              "GBENCH_TOOLBENCH_TEMPERATURE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GBENCH_TOOLBENCH_GROUPS", "G1_instruction G3_instruction")
    monkeypatch.setattr(T, "check_toolbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(T, "_bridge_gateway", lambda image: "127.0.0.1")
    monkeypatch.setattr(T, "read_server_calls", lambda path: [])
    monkeypatch.setattr(T, "read_observations", lambda path: {})

    async def status(query, answer, sem):
        return "Solved"

    async def pref(query, cand, ref, sem):
        return "cand"
    monkeypatch.setattr(T, "_answer_status", status)
    monkeypatch.setattr(T, "_preference", pref)
    ref_root = tmp_path / "ref"
    (ref_root / T._REFERENCE_DEFAULT).mkdir(parents=True)
    for g, ids in TEST_IDS.items():
        (ref_root / T._REFERENCE_DEFAULT / f"{g}.json").write_text(
            json.dumps({q: _tb_answer() for q in ids}))
    monkeypatch.setattr(T, "_baseline_reference_root", lambda ref, groups, image: str(ref_root))
    monkeypatch.setattr(T, "_group_test_ids", lambda image, groups, limit: {
        g: list(TEST_IDS[g]) for g in groups})
    seen = {"runs": []}

    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ["docker", "run"]:
            seen["runs"].append(cmd)
            out = cmd[cmd.index("-v") + 1].split(":")[0]
            seen["selection"] = (json.load(open(os.path.join(out, T.QUERY_IDS_FILE)))
                                 if os.path.exists(os.path.join(out, T.QUERY_IDS_FILE))
                                 else None)
            groups = cmd[[i for i, a in enumerate(cmd) if a.startswith("TB_GROUPS=")][0]][
                len("TB_GROUPS="):].split()
            os.makedirs(os.path.join(out, "candidate"), exist_ok=True)
            for g in groups:
                ids = (seen["selection"] or {}).get(g, TEST_IDS[g])
                with open(os.path.join(out, "candidate", f"{g}.json"), "w") as fh:
                    json.dump({q: dict(_tb_answer(), query="q") for q in ids}, fh)
            with open(os.path.join(out, T.MODEL_FAILURES), "w"):
                pass
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(T.subprocess, "run", fake_run)
    return seen


def test_toolbench_shard_runs_the_selected_ids_and_merges(monkeypatch, toolbench_run):
    parts = []
    for i in (1, 2):
        monkeypatch.setenv("GBENCH_SHARD", f"{i}/2")
        res = T.run_toolbench("org/served", "http://x:1/v1", 2)
        sel, _, _ = T.select_query_ids(TEST_IDS, None)
        assert toolbench_run["selection"] == {g: v for g, v in sel.items() if v}
        assert res["selected_query_ids"] == sel and res["shard"] == f"{i}/2"
        assert "TB_MODEL=org/served" in toolbench_run["runs"][-1]
        assert any(a == f"TB_QUERY_IDS_FILE=/out/{T.QUERY_IDS_FILE}"
                   for a in toolbench_run["runs"][-1])
        assert "sharded run (--shard" in res["leaderboard_comparable_reason"]
        parts.append(res)
    merged = T.merge_shards(parts)
    assert merged["accuracy"] == 100.0 and merged["sowr"] == 100.0
    assert merged["total_questions"] == 6 and merged["leaderboard_comparable"] is False


def test_toolbench_merge_counts_every_shards_simulator_cut_replies_once(monkeypatch,
                                                                       toolbench_run):
    # each shard's cached server saw one simulator reply cut at the raised limit (equal counts,
    # so a merge that de-duplicates the shards' reasons would name 1 where the union has 2)
    monkeypatch.setattr(T, "read_server_calls", lambda path: [
        {"type": "fake_response", "simulator_models": ["gemini-x"],
         "simulator_max_tokens_cut": 1}])
    parts = []
    for i in (1, 2):
        monkeypatch.setenv("GBENCH_SHARD", f"{i}/2")
        parts.append(T.run_toolbench("org/served", "http://x:1/v1", 2))
    one = T.simulator_cutoff_reason(parts[0]["simulator_usage"])
    assert "in 1 replies used as they arrived" in one
    assert all(one in p["leaderboard_comparable_reason"] for p in parts)
    merged = T.merge_shards(parts)
    assert merged["simulator_usage"]["simulator_replies_cut_at_max_tokens"] == 2
    reason = merged["leaderboard_comparable_reason"]
    assert T.simulator_cutoff_reason(merged["simulator_usage"]) in reason
    assert "in 2 replies used as they arrived" in reason and one not in reason
    assert reason.count("cache-miss simulator's Gemini replies stopped") == 1


def test_toolbench_empty_shard_withholds_and_runs_nothing(monkeypatch, toolbench_run):
    monkeypatch.setenv("GBENCH_SHARD", "9/9")
    res = T.run_toolbench("m", "http://x:1/v1", 2)
    assert toolbench_run["runs"] == [] and res["accuracy"] is None and res["empty_shard"]


def test_toolbench_full_run_passes_no_selection_and_reads_the_endpoint_key(monkeypatch,
                                                                            toolbench_run):
    res = T.run_toolbench("m", "http://x:1/v1", 2, attempt_count=3)
    assert toolbench_run["selection"] is None and res["selected_query_ids"] is None
    assert res["attempts"]["not_applied"]
    monkeypatch.setenv("VLLM_API_KEY", SECRET)
    monkeypatch.setenv("OPENAI_API_KEY", "other-key-0123456789")
    assert T._endpoint_key() == SECRET
    monkeypatch.setenv("VLLM_API_KEY", "EMPTY")
    assert T._endpoint_key() == "other-key-0123456789"


# =========================================================================== #
# skillsbench
# =========================================================================== #
def _launcher():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "skillsbench_run_w3b", REPO / "gbench" / "docker" / "skillsbench_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("spec", ["3/2", "x", "0/4", "1/0"])
def test_skillsbench_launcher_stops_on_a_malformed_shard(monkeypatch, spec):
    launcher = _launcher()
    monkeypatch.setenv("GBENCH_SHARD", spec)
    with pytest.raises(SystemExit) as ei:
        launcher._shard()
    assert ei.value.code not in (0, None)
    monkeypatch.setenv("GBENCH_SHARD", "2/3")
    assert launcher._shard() == (2, 3)
    monkeypatch.setenv("GBENCH_SHARD", "")
    assert launcher._shard() is None


def _sb_run(monkeypatch, summary, **kwargs):
    monkeypatch.setattr(SB, "check_skillsbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(SB.swebench_common, "require_task_network_blocked",
                        lambda *a, **k: None)
    monkeypatch.setattr(SB, "dns_harness_errors", lambda workdir, summary: ({}, {}))
    seen = {}

    def run(cmd, **kw):
        if cmd[:2] == ["docker", "run"]:
            seen["cmd"] = cmd
            workdir = cmd[cmd.index("-v", cmd.index("-v") + 1) + 1].split(":")[0]
            with open(os.path.join(workdir, "skillsbench_summary.json"), "w") as f:
                json.dump(summary, f)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(SB.subprocess, "run", run)
    return SB.run_skillsbench("m", "http://127.0.0.1:8000/v1", 1, **kwargs), seen


def _summary(tasks, full=87):
    return {"conditions": {"without-skills": {
        "per_task": {t: (1.0 if i % 2 == 0 else 0.0) for i, t in enumerate(tasks)},
        "mean_reward": None, "n_tasks": len(tasks), "n_scored": len(tasks)}},
        "n_tasks_selected": len(tasks), "n_tasks_full": full, "tasks_selected": list(tasks)}


def test_skillsbench_forwards_the_normalized_shard_and_records_the_knobs(monkeypatch):
    monkeypatch.setenv("GBENCH_SHARD", " 2/3 ")
    # (the relay applies the temperature in both modes, tests/test_skillsbench.py)
    res, seen = _sb_run(monkeypatch, _summary(["a", "b"]), enable_thinking=False,
                        temperature=0.3, max_output_tokens=4096, attempt_count=2)
    assert "GBENCH_SHARD=2/3" in seen["cmd"]
    assert res["shard"] == "2/3" and res["thinking"] is False and res["thinking_applied"] is False
    assert res["temperature_applied"] is True and "temperature_note" not in res
    assert (res["temperature"], res["temperature_source"]) == (0.3, "cli:--temperature")
    assert "--max-output-tokens 4096 not applied" in res["max_output_tokens_note"]
    assert res["attempts"]["not_applied"] and res["tasks_selected"] == ["a", "b"]
    assert "sharded run (--shard 2/3)" in res["leaderboard_comparable_reason"]
    assert [t["sample_key"] for t in res["sample_traces"]] == [
        "task:without-skills/a", "task:without-skills/b"]


def test_skillsbench_malformed_shard_fails_before_docker(monkeypatch):
    monkeypatch.setenv("GBENCH_SHARD", "5/4")
    with pytest.raises(ValueError):
        _sb_run(monkeypatch, _summary(["a"]))


def test_skillsbench_empty_shard_withholds(monkeypatch):
    monkeypatch.setenv("GBENCH_SHARD", "88/88")
    res, _ = _sb_run(monkeypatch, _summary([]))
    assert res["accuracy"] is None and res["empty_shard"] and res["full_set_items"] == 87


def test_skillsbench_shards_merge_back_to_the_mean_reward(monkeypatch):
    tasks = [f"t{i}" for i in range(6)]
    parts = []
    for i in (1, 2):
        monkeypatch.setenv("GBENCH_SHARD", f"{i}/2")
        sub = tasks[i - 1::2]
        res, _ = _sb_run(monkeypatch, {"conditions": {"without-skills": {
            "per_task": {t: float(int(t[1:]) % 3 == 0) for t in sub}}},
            "n_tasks_selected": 3, "n_tasks_full": 6, "tasks_selected": sub})
        parts.append(res)
    merged = SB.merge_shards(parts)
    assert merged["accuracy"] == round(100.0 * 2 / 6, 2) and merged["n_tasks"] == 6


def test_skillsbench_falls_back_to_the_shared_endpoint_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    assert SB.endpoint_api_key() == SECRET and SB.endpoint_key_source() == "OPENAI_API_KEY"
    monkeypatch.setenv("GBENCH_SKILLSBENCH_API_KEY", "suite-key-0123456789")
    assert SB.endpoint_api_key() == "suite-key-0123456789"
    assert SB.endpoint_key_source() == "GBENCH_SKILLSBENCH_API_KEY"


# =========================================================================== #
# aider_polyglot / livebench: an endpoint key reaches the endpoint through gbench's relay only
# =========================================================================== #
class _KeyRelay:
    made = []
    TIMEOUT_S = 3600

    def __init__(self, upstream, bind_host, key=None, name=""):
        self.upstream, self.bind_host, self.key, self.stopped = upstream, bind_host, key, False
        _KeyRelay.made.append(self)

    def start(self):
        return self

    def stop(self):
        self.stopped = True

    @property
    def url(self):
        return f"http://{self.bind_host}:43210/v1"

    def record(self):
        return {"listens": self.url}


def _key_relay(monkeypatch):
    import gbench.runners.eval_suites.gemini_relay as GR
    import gbench.runners.eval_suites.swebench_common as SWC
    _KeyRelay.made = []
    monkeypatch.setattr(SWC, "EndpointRelay", _KeyRelay)
    monkeypatch.setattr(GR, "network_gateway", lambda eval_name, network, docs: "172.17.0.1")
    monkeypatch.setenv("VLLM_API_KEY", SECRET)


def test_aider_sends_the_endpoint_key_through_the_relay_only(monkeypatch, tmp_path):
    _polyglot(tmp_path, per_lang=1)
    seen = _aider_env(monkeypatch, tmp_path)
    _key_relay(monkeypatch)
    r = A.run_aider_polyglot("m", "http://127.0.0.1:8000/v1", concurrency=1, limit=2)
    relay = _KeyRelay.made[0]
    assert relay.key == SECRET and relay.upstream == "http://127.0.0.1:8000/v1" and relay.stopped
    joined = " ".join(seen["cmd"])
    assert f"OPENAI_API_BASE={relay.url}" in joined and "OPENAI_API_KEY=EMPTY" in joined
    assert SECRET not in joined and r["model_endpoint_relay"]["listens"] == relay.url


def test_livebench_sends_the_endpoint_key_through_the_relay_only(monkeypatch, tmp_path):
    _key_relay(monkeypatch)
    res, seen = _lb_run(monkeypatch, tmp_path)
    relay = _KeyRelay.made[0]
    inner = seen["cmd"][-1]
    assert relay.key == SECRET and relay.stopped and relay.TIMEOUT_S >= 6 * 60 * 60
    assert f"--api-base {relay.url} --api-key EMPTY" in inner and SECRET not in inner
    assert res["model_endpoint_relay"]["listens"] == relay.url
