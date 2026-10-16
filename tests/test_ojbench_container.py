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

"""ojbench container-judging regression tests.

The judge moved off the host (DMOJ's cptbox won't build on the 3.12 serving env) into the
gbench-ojbench:5e94480b image; these assert the image resolver, the empty-records short-circuit, and the
docker command (SYS_PTRACE + unconfined seccomp for the inner sandbox, read-only testdata mount)."""

import json
import os

import gbench.runners.eval_suites.ojbench as OJ


class _P:
    stdout = ""
    stderr = ""


def test_image_default_and_override(monkeypatch):
    monkeypatch.delenv("GBENCH_OJBENCH_IMAGE", raising=False)
    assert OJ._image() == "gbench-ojbench:5e94480b"
    monkeypatch.setenv("GBENCH_OJBENCH_IMAGE", "myimg:1")
    assert OJ._image() == "myimg:1"


def test_empty_records_never_launches_docker(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(OJ.subprocess, "run",
                        lambda *a, **k: calls.__setitem__("n", calls["n"] + 1) or _P())
    assert OJ._judge_in_container([], 4) == []
    assert calls["n"] == 0


def test_judge_command_and_result_parsing(monkeypatch, tmp_path):
    td = tmp_path / "testdata"
    (td / "NOI").mkdir(parents=True)
    (td / "ICPC").mkdir()
    monkeypatch.setenv("GBENCH_OJBENCH_TESTDATA", str(td))
    captured = {}

    def fake_run(cmd, capture_output=False, text=False, timeout=None):
        if "run" in cmd:                                   # the judge run (not the reap rm -f)
            captured["cmd"] = cmd
            workdir = next(a.split(":", 1)[0] for a in cmd if a.endswith(":/work"))
            with open(os.path.join(workdir, "results.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps({"id": 1, "language": "cpp", "is_passed": True}) + "\n")
        return _P()

    monkeypatch.setattr(OJ.subprocess, "run", fake_run)
    out = OJ._judge_in_container(
        [{"id": 1, "dataset": "NOI", "language": "cpp", "difficulty": "easy", "content": "int main(){}"}], 4)
    assert out == [{"id": 1, "language": "cpp", "is_passed": True}]

    cmd = captured["cmd"]
    assert cmd[:2] == ["docker", "run"] and OJ._image() in cmd
    assert "SYS_PTRACE" in cmd and "seccomp=unconfined" in cmd          # inner cptbox sandbox
    assert any(a.endswith("/testdata:ro") for a in cmd)                 # testdata read-only
    assert "OJBENCH_TESTDATA=/testdata" in cmd


def test_missing_results_is_infra_failure_not_zero(monkeypatch, tmp_path):
    td = tmp_path / "testdata"
    (td / "NOI").mkdir(parents=True)
    (td / "ICPC").mkdir()
    monkeypatch.setenv("GBENCH_OJBENCH_TESTDATA", str(td))
    # container "runs" but writes no results.jsonl -> must raise, never silently score 0
    monkeypatch.setattr(OJ.subprocess, "run", lambda *a, **k: _P())
    try:
        OJ._judge_in_container([{"id": 1, "language": "cpp", "content": "x"}], 4)
        assert False, "expected RuntimeError on missing results"
    except RuntimeError as e:
        assert "no results.jsonl" in str(e)
