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

"""Canonical Aider Polyglot: containerized delegation to aider's own benchmark.py (pass@2)."""
import json
import types
import warnings

warnings.filterwarnings("ignore")

import pytest
import gbench.runners.eval_suites.aider_polyglot as A
import gbench.runners.eval_suites.tau_common as TC


def test_aider_hard_errors_without_docker(monkeypatch):
    monkeypatch.setattr(A.shutil, "which", lambda _x: None)
    with pytest.raises(RuntimeError, match="docs/evals/aider_polyglot.md"):
        A._check_prereqs("aider-benchmark:5dc9490b", "/tmp/bench")


def test_aider_hard_errors_without_image(monkeypatch):
    monkeypatch.setattr(A.shutil, "which", lambda _x: "/usr/bin/docker")

    def fake_run(cmd, **_kw):
        rc = 0 if cmd[:2] == ["docker", "info"] else 1   # daemon ok, image inspect fails
        return types.SimpleNamespace(returncode=rc, stdout=b"", stderr=b"")

    monkeypatch.setattr(A.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="not found"):
        A._check_prereqs("aider-benchmark:5dc9490b", "/tmp/bench")


def test_aider_hard_errors_without_exercises(monkeypatch, tmp_path):
    monkeypatch.setattr(A.shutil, "which", lambda _x: "/usr/bin/docker")
    monkeypatch.setattr(A.subprocess, "run",
                        lambda cmd, **k: types.SimpleNamespace(returncode=0, stdout=b"", stderr=b""))
    # docker+image OK, but the exercises dir is absent -> hard-error (never a silent skip)
    with pytest.raises(RuntimeError, match="polyglot"):
        A._check_prereqs("aider-benchmark:5dc9490b", str(tmp_path))


def test_build_command_drives_aider_benchmark_natively():
    cmd = A._build_command("aider-benchmark:5dc9490b", "/bench", "gbench", "openai/m",
                           "http://127.0.0.1:8000/v1", 8, A._LANGUAGES, "", None, "")
    joined = " ".join(cmd)
    # docker's bridge network with the host at host.docker.internal, as upstream's
    # benchmark/docker.sh runs the image; never the host network (see test_guard_code_images.py)
    assert cmd[0] == "docker" and cmd[3:7] == ["--network", "bridge", "--add-host",
                                               "host.docker.internal:host-gateway"]
    assert "host" not in cmd
    assert "-v" in cmd and "/bench:/benchmarks" in cmd
    assert "AIDER_DOCKER=1" in joined and "AIDER_BENCHMARK_DIR=/benchmarks" in joined
    assert "OPENAI_API_BASE=http://127.0.0.1:8000/v1" in joined
    assert "benchmark.py gbench" in joined and "--model openai/m" in joined
    assert "--tries 2" in joined                      # canonical pass@2
    assert "--exercises-dir polyglot-benchmark" in joined
    assert "--edit-format" not in joined              # native default when unset
    for lang in A._LANGUAGES:
        assert lang in joined
    # a selected subset (--eval-limit / --shard) runs from its own exercises tree, mounted
    # read-only; benchmark.py's --num-tests (an unseeded shuffle) is never passed
    cmd2 = A._build_command("i", "/b", "gbench", "openai/m", "http://h/v1", 1,
                            A._LANGUAGES, "diff", "/tmp/tree", "")
    j2 = " ".join(cmd2)
    assert "--num-tests" not in j2 and "--num-tests" not in joined
    assert "/tmp/tree:/gbench-exercises:ro" in cmd2
    assert "--exercises-dir /gbench-exercises" in j2 and "--edit-format diff" in j2


def test_parse_results_computes_pass_at_1_and_2(tmp_path):
    """pass@1 = passed on try 1; pass@2 = passed within two tries (aider stops early on pass)."""
    def _write(lang, name, outcomes):
        d = tmp_path / lang / "exercises" / "practice" / name
        d.mkdir(parents=True)
        (d / ".aider.results.json").write_text(json.dumps(
            {"language": lang, "testcase": name, "tests_outcomes": outcomes}))

    _write("python", "p1", [True])            # pass@1 and pass@2
    _write("python", "p2", [False, True])     # pass@2 only
    _write("rust", "r1", [False, False])      # neither
    _write("go", "g1", [])                    # no outcomes -> counted, not passed
    agg = A._parse_results(str(tmp_path))
    assert agg["n"] == 4
    assert agg["pass1"] == 1 and agg["pass2"] == 2
    assert agg["per_lang"]["python"] == {"total": 2, "pass1": 1, "pass2": 2}
    assert agg["per_lang"]["rust"]["pass2"] == 0


def test_aider_leaderboard_comparable_only_for_full_native_run(monkeypatch, tmp_path):
    """A full 6-language native-edit-format run (no --eval-limit) is comparable; a subset is not.
    The mocked `docker run` simulates aider's `--new` by creating a fresh timestamped results dir
    each call (so run_aider_polyglot's before/after snapshot discovers it)."""
    import os
    monkeypatch.setenv("GBENCH_AIDER_BENCHMARK_DIR", str(tmp_path))
    monkeypatch.setenv("GBENCH_AIDER_MODEL", "openai/m")   # skip the served-id network lookup
    monkeypatch.setattr(A, "_check_prereqs", lambda image, bench: None)
    # the network checks before the container are docker calls of their own
    # (tests/test_guard_code_images.py)
    monkeypatch.setattr(TC, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC.container_url(url))
    calls = {"n": 0}

    import re
    langs = ["cpp", "go", "java", "javascript", "python", "rust"]

    # the exercises checkout a limited run selects from (a full run needs all 225 results)
    for i in range(12):
        (tmp_path / "polyglot-benchmark" / langs[i % len(langs)] / "exercises" / "practice"
         / f"x{i}").mkdir(parents=True, exist_ok=True)

    def fake_run(cmd, **k):
        calls["n"] += 1
        # A full run scores the whole 225-exercise set; a subset run scores the exercises of the
        # tree mounted at /gbench-exercises. The no-partial gate requires every one finished.
        mount = next((a.split(":", 1)[0] for a in cmd if a.endswith(":/gbench-exercises:ro")),
                     None)
        names = (sorted(str(p.relative_to(mount)).split("/")[0] + "/" + p.name
                        for p in __import__("pathlib").Path(mount).glob("*/exercises/practice/*"))
                 if mount else [f"{langs[i % len(langs)]}/x{i}" for i in range(A._EXPECTED_FULL)])
        base = tmp_path / f"2026-run{calls['n']}--gbench"
        for name in names:
            lang, ex = name.split("/")
            d = base / lang / "exercises" / "practice" / ex
            d.mkdir(parents=True, exist_ok=True)
            (d / ".aider.results.json").write_text(
                json.dumps({"language": lang, "tests_outcomes": [True]}))
        # The in-container pre-flight's report that aider loaded gbench's model settings.
        report = {"files_loaded": ["/gbench-aider/model-settings.yml"], "extra_params": {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "timeout": A.model_request_timeout_s(), "max_retries": 0,
            "temperature": 0.0}}  # a no-think run's temperature (base.resolve_temperature)
        # ... and the OOM-kill count the container prints last (none here)
        return types.SimpleNamespace(returncode=0, stderr="",
                                     stdout=A._CHECK_MARKER + json.dumps(report) + "\n"
                                     + A._MEMORY_MARKER + "12884901888\n" + A._OOM_MARKER + "0\n")

    monkeypatch.setattr(A.subprocess, "run", fake_run)

    r = A.run_aider_polyglot("m", "http://127.0.0.1:8000/v1", concurrency=1)
    assert r["eval_name"] == "aider_polyglot" and r["accuracy"] == 100.0
    assert r["pass_rate_2"] == 100.0 and r["tries"] == 2
    assert r["leaderboard_comparable"] is True           # full, native, no limit

    r2 = A.run_aider_polyglot("m", "http://127.0.0.1:8000/v1", concurrency=1, limit=5)
    assert r2["leaderboard_comparable"] is False          # subset
    assert len(r2["selected_exercises"]) == 5 and r2["total_questions"] == 5
