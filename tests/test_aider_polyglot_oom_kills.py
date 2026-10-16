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

"""aider_polyglot: OOM kills in the shared 12g benchmark container are counted and named.

The container has upstream's memory limit (benchmark/docker.sh --memory=12g --memory-swap=12g),
shared by the exercises benchmark.py runs at once (--threads). After benchmark.py exits the
container prints the OOM-kill count of its cgroup as its last line; the result records it
(container_oom_kills), and any kill, or a count that could not be read, makes the run not
comparable, since a kill there cannot be attributed to the exercise whose code used the memory.

Docker and the endpoint are stubbed; nothing is contacted. One test runs the report's shell
snippet with the host's bash (it only reads /proc and /sys/fs/cgroup).
"""

import json
import subprocess
import types
from pathlib import Path

import pytest

from gbench.runners.eval_suites import aider_polyglot as A
from gbench.runners.eval_suites import tau_common as TC

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs" / "evals"


def _flat(text):
    return " ".join(text.split())


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in ("GBENCH_AIDER_LANGUAGES", "GBENCH_AIDER_MODEL", "GBENCH_AIDER_EDIT_FORMAT",
                 "GBENCH_AIDER_EXTRA_ARGS"):
        monkeypatch.delenv(name, raising=False)


def _cmd(threads=4, extra=""):
    return A._build_command("aider-benchmark:5dc9490b", "/b", "gbench", "openai/m",
                            "http://host.docker.internal:8000/v1", threads, A._LANGUAGES, "",
                            None, extra, settings_dir="/s")


# --------------------------------------------------------------------------- #
# the container prints the count last and keeps benchmark.py's exit status     #
# --------------------------------------------------------------------------- #
def test_the_report_runs_after_benchmark_py_and_keeps_its_exit_status():
    script = _cmd(extra="--sleep 1")[-1]
    assert script.endswith(A._OOM_REPORT)
    assert script.index("./benchmark/benchmark.py") < script.index("--sleep 1") \
        < script.index(A._OOM_MARKER)
    # the chain before it ends with benchmark.py, so `rc=$?` is its status; the shell exits with it
    assert A._OOM_REPORT.startswith("; rc=$?; echo;") and A._OOM_REPORT.endswith("exit $rc")
    # cgroup v2 (the container's own cgroup, private or host cgroup namespace), then v1
    assert "/proc/self/cgroup" in A._OOM_REPORT
    assert "memory.events" in A._OOM_REPORT and "memory/memory.oom_control" in A._OOM_REPORT
    assert "/^oom_kill /" in A._OOM_REPORT          # not oom_group_kill


@pytest.mark.parametrize("status", [0, 3])
def test_the_report_snippet_runs_in_bash(status):
    # the same snippet the container runs, after a command with a known status
    out = subprocess.run(["bash", "-c", f"(exit {status})" + A._OOM_REPORT],
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == status
    last = [ln for ln in out.stdout.splitlines() if ln.strip()][-1]
    assert last.startswith(A._OOM_MARKER)
    value = last[len(A._OOM_MARKER):].strip()
    assert value == "" or value.isdigit()
    assert A.container_oom_kills(out.stdout) == (int(value) if value else None)


def test_the_report_is_its_own_line_after_output_without_a_newline():
    # benchmark.py's last output may not end with a newline; the report still stands alone
    out = subprocess.run(["bash", "-c", "printf 'last output'; (exit 1)" + A._OOM_REPORT],
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == 1
    assert "last output\n" in out.stdout
    last = [ln for ln in out.stdout.splitlines() if ln.strip()][-1]
    assert last.startswith(A._OOM_MARKER)
    value = last[len(A._OOM_MARKER):].strip()
    assert A.container_oom_kills(out.stdout) == (int(value) if value else None)


@pytest.mark.parametrize("stdout, expected", [
    (A._OOM_MARKER + "0\n", 0),
    ("benchmark output\n" + A._OOM_MARKER + "3\n\n", 3),
    (A._OOM_MARKER + "\n", None),                         # neither cgroup file was readable
    ("", None),
    ("x\n" + A._OOM_MARKER + "0\nmore output\n", None),   # not the last line: not the report
    (A._OOM_MARKER + "two\n", None),
])
def test_only_the_last_line_is_the_report(stdout, expected):
    assert A.container_oom_kills(stdout) == expected


# --------------------------------------------------------------------------- #
# the result                                                                   #
# --------------------------------------------------------------------------- #
def _aider_run(monkeypatch, tmp_path, oom_line, returncode=0, exercises=None):
    langs = A._LANGUAGES
    n = A._EXPECTED_FULL if exercises is None else exercises

    def fake_run(cmd, **_kw):
        fake_run.cmd = cmd
        root = tmp_path / "2026-run1--gbench"
        for i in range(n):
            d = root / langs[i % len(langs)] / "exercises" / "practice" / f"x{i}"
            d.mkdir(parents=True, exist_ok=True)
            (d / ".aider.results.json").write_text(json.dumps(
                {"language": langs[i % len(langs)], "testcase": f"x{i}",
                 "tests_outcomes": [True] if i % 2 else [False, True]}))
            (d / ".aider.chat.history.md").write_text("")
        report = {"files_loaded": ["/gbench-aider/model-settings.yml"], "extra_params": {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "timeout": A.model_request_timeout_s(), "max_retries": 0,
            "temperature": 0.0}}  # a no-think run's temperature (base.resolve_temperature)
        return types.SimpleNamespace(
            returncode=returncode, stderr="",
            stdout=A._CHECK_MARKER + json.dumps(report) + "\n" + A._MEMORY_MARKER
            + "12884901888\n" + oom_line)
    monkeypatch.setenv("GBENCH_AIDER_BENCHMARK_DIR", str(tmp_path))
    monkeypatch.setenv("GBENCH_AIDER_MODEL", "openai/m")
    for i in range(n):      # the exercises checkout a limited run selects from
        (tmp_path / "polyglot-benchmark" / langs[i % len(langs)] / "exercises" / "practice"
         / f"x{i}").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(A, "_check_prereqs", lambda image, bench: None)
    monkeypatch.setattr(TC, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC.container_url(url))
    monkeypatch.setattr(A.subprocess, "run", fake_run)
    return fake_run


def test_no_oom_kill_keeps_the_run_comparable(monkeypatch, tmp_path):
    _aider_run(monkeypatch, tmp_path, A._OOM_MARKER + "0\n")
    r = A.run_aider_polyglot("m", "http://127.0.0.1:8000/v1", concurrency=10)
    assert r["container_oom_kills"] == 0
    assert r["leaderboard_comparable"] is True, r.get("leaderboard_comparable_reason")
    assert "leaderboard_comparable_reason" not in r


@pytest.mark.parametrize("threads, beyond", [(64, True), (10, False), (4, False)])
def test_an_oom_kill_is_recorded_and_makes_the_run_not_comparable(monkeypatch, tmp_path,
                                                                  threads, beyond):
    _aider_run(monkeypatch, tmp_path, A._OOM_MARKER + "3\n")
    r = A.run_aider_polyglot("m", "http://127.0.0.1:8000/v1", concurrency=threads)
    assert r["container_oom_kills"] == 3
    assert r["container_memory_limit"] == {"memory": "12g", "memory_swap": "12g",
                                           "threads": threads, "enforced_bytes": 12884901888}
    assert r["leaderboard_comparable"] is False
    reason = r["leaderboard_comparable_reason"]
    assert ("the kernel's OOM killer killed 3 process(es) in the benchmark container "
            "(container_oom_kills, its cgroup's memory.events oom_kill) at the 12g memory limit "
            f"shared by the {threads} exercise(s) benchmark.py runs at once (--threads)") in reason
    assert "cannot be attributed to the exercise whose code used the memory" in reason
    assert ("more than the 10 of upstream's documented run (benchmark/README.md --threads 10)"
            in reason) is beyond
    # the score is kept as upstream computes it
    assert r["status"] == "success" and r["accuracy"] == 100.0
    assert r["total_questions"] == A._EXPECTED_FULL
    assert r["pass_rate_1"] == round(100.0 * (A._EXPECTED_FULL // 2) / A._EXPECTED_FULL, 2)


@pytest.mark.parametrize("oom_line", ["", A._OOM_MARKER + "\n"])
def test_an_unread_count_makes_the_run_not_comparable(monkeypatch, tmp_path, oom_line):
    _aider_run(monkeypatch, tmp_path, oom_line)
    r = A.run_aider_polyglot("m", "http://127.0.0.1:8000/v1", concurrency=1)
    assert r["container_oom_kills"] is None
    assert r["leaderboard_comparable"] is False
    assert ("the benchmark container's OOM-kill count could not be read (no GBENCH_OOM_KILLS "
            "line with a number at the end of its output") in r["leaderboard_comparable_reason"]
    assert r["status"] == "success"


def test_a_full_run_killed_at_the_end_is_not_comparable(monkeypatch, tmp_path):
    # benchmark.py killed after every exercise finished: exit 137, the report still prints
    _aider_run(monkeypatch, tmp_path, A._OOM_MARKER + "1\n", returncode=137)
    r = A.run_aider_polyglot("m", "http://127.0.0.1:8000/v1", concurrency=12)
    assert r["container_oom_kills"] == 1 and r["leaderboard_comparable"] is False
    assert "more than the 10" in r["leaderboard_comparable_reason"]


@pytest.mark.parametrize("limit", [None, 5])
def test_the_hard_errors_name_the_oom_kills(monkeypatch, tmp_path, limit):
    _aider_run(monkeypatch, tmp_path, A._OOM_MARKER + "2\n", returncode=137, exercises=10)
    with pytest.raises(RuntimeError) as ei:
        A.run_aider_polyglot("m", "http://127.0.0.1:8000/v1", concurrency=16, limit=limit)
    msg = str(ei.value)
    assert "137: its main process was killed with SIGKILL" in msg
    assert ("the kernel's OOM killer killed 2 process(es) in the container at its 12g limit, "
            "shared by 16 thread(s) (container_oom_kills)") in msg


def test_a_hard_error_without_oom_kills_says_nothing_about_them(monkeypatch, tmp_path):
    _aider_run(monkeypatch, tmp_path, A._OOM_MARKER + "0\n", returncode=1, exercises=10)
    with pytest.raises(RuntimeError) as ei:
        A.run_aider_polyglot("m", "http://127.0.0.1:8000/v1", concurrency=2, limit=5)
    assert "OOM killer" not in str(ei.value)
    assert A._oom_text(None, 2) == "" and A._oom_text(0, 2) == ""


# --------------------------------------------------------------------------- #
# docs match the code                                                          #
# --------------------------------------------------------------------------- #
def test_the_doc_describes_the_oom_count():
    doc = _flat((DOCS / "aider_polyglot.md").read_text(encoding="utf-8"))
    assert "`container_oom_kills`" in doc and "`GBENCH_OOM_KILLS <n>`" in doc
    assert "`--threads 10`" in doc and "`memory.oom.group`" in doc
    assert "`container_oom_kills: null`" in doc
    assert ("no harness error and a container that reported 0 OOM kills "
            "(`container_oom_kills`)") in doc
    assert A.UPSTREAM_THREADS == 10
    assert "container_oom_kills" in A.__doc__
