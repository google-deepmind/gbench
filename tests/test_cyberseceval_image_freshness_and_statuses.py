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

"""cyberseceval: a stale image is refused, the cached prompt file is written safely, and the
final traces say what happened to them.

* The image check compares the constraints file and the base image the image was built from with
  this gbench's: a change to either on the host means the image is stale (rebuild it).
* Two runs sharing GBENCH_CACHE_DIR never write into one temporary file.
* A trace the ICD scored ends with status OK (no judge takes part), and an empty reply the server
  returned, which the benchmark scores secure, is an answered sample, not a failed request.

No network, no docker daemon, no model server: docker and the ICD are faked.
"""

import hashlib
import json
import os
import threading
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import cyberseceval as C

REPO = Path(__file__).resolve().parents[1]
DOCKER_DIR = REPO / "gbench" / "docker"
CONSTRAINTS = DOCKER_DIR / "cyberseceval.constraints.txt"
WORKER = DOCKER_DIR / "cyberseceval_icd_worker.py"


class _Proc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


def _probe(**over):
    probe = {"semgrep": C.SEMGREP_VERSION, "pyyaml": C.PYYAML_VERSION, "python": "3.10.21",
             "purplellama_commit": C.PURPLELLAMA_COMMIT,
             "worker_sha256": hashlib.sha256(WORKER.read_bytes()).hexdigest(),
             "constraints_sha256": hashlib.sha256(CONSTRAINTS.read_bytes()).hexdigest(),
             "selftest": {"issues": [{"analyzer": "semgrep"}]}}
    probe.update(over)
    return probe


def _stub(monkeypatch, labels, probe):
    monkeypatch.setattr(C, "_PROBE", {})
    monkeypatch.setattr(C.shutil, "which", lambda name: "/usr/bin/docker")

    def fake_run(argv, **kw):
        if argv[:3] == ["docker", "image", "inspect"]:
            return _Proc(stdout=json.dumps(labels))
        return _Proc(stdout=C._WORKER_REPLY_MARK + json.dumps(probe) + "\n")
    monkeypatch.setattr(C.subprocess, "run", fake_run)


def _labels(**over):
    labels = {**C.ICD_IMAGE_LABELS, C._BASE_LABEL: C._dockerfile_base_image()}
    labels.update(over)
    return labels


def test_the_image_check_accepts_an_image_built_from_this_tree(monkeypatch):
    _stub(monkeypatch, _labels(), _probe())
    probe = C.check_icd_image()
    assert probe["constraints_sha256"] == hashlib.sha256(CONSTRAINTS.read_bytes()).hexdigest()
    rec = C._icd_record(probe)
    assert rec["constraints_sha256"] == probe["constraints_sha256"]
    assert rec["base_image"].startswith("python:3.10-slim@sha256:")


def test_an_image_built_under_other_constraints_is_stale(monkeypatch):
    _stub(monkeypatch, _labels(), _probe(constraints_sha256="f" * 64))
    with pytest.raises(RuntimeError, match=r"constraints_sha256 ffffffffffff .*rebuild it"):
        C.check_icd_image()
    _stub(monkeypatch, _labels(), _probe(constraints_sha256=None))
    with pytest.raises(RuntimeError, match="constraints_sha256 None"):
        C.check_icd_image()


def test_an_image_built_from_another_base_is_stale(monkeypatch):
    _stub(monkeypatch, _labels(**{C._BASE_LABEL: "python:3.10-slim@sha256:" + "0" * 64}),
          _probe())
    with pytest.raises(RuntimeError, match="org.gbench.cyberseceval.base=python:3.10-slim"):
        C.check_icd_image()
    _stub(monkeypatch, C.ICD_IMAGE_LABELS, _probe())            # built before the label
    with pytest.raises(RuntimeError, match="org.gbench.cyberseceval.base=unset"):
        C.check_icd_image()


def test_the_worker_probe_reports_the_constraints_it_was_built_with(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("icd_worker_probe", WORKER)
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    assert w.CONSTRAINTS_FILE == "/opt/cyberseceval/constraints.txt"
    # the Dockerfile copies the repo's constraints to that path
    text = (DOCKER_DIR / "cyberseceval.Dockerfile").read_text(encoding="utf-8")
    assert "COPY cyberseceval.constraints.txt /opt/cyberseceval/constraints.txt" in text
    assert "OpenTelemetry" not in text and "urllib3~=1.26" in text


def _dataset_reply(data: bytes):
    return {"relpath": C.INSTRUCT_V2_RELPATH, "text": data.decode("utf-8")}


def test_concurrent_runs_sharing_the_cache_never_share_a_temporary_file(monkeypatch, tmp_path):
    data = b'[{"language": "c"}]'
    monkeypatch.setattr(C, "INSTRUCT_V2_SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(C, "gbench_cache_dir", lambda: str(tmp_path))
    monkeypatch.setattr(C, "_one_reply", lambda req, timeout: _dataset_reply(data))
    # another writer's temporary file, as the fixed "<dest>.part" name used to be
    dest_dir = tmp_path / "cyberseceval"
    dest_dir.mkdir()
    stale = dest_dir / f"instruct-v2-{C.PURPLELLAMA_COMMIT[:12]}.json.part"
    stale.write_bytes(b"half written by another run")
    errors, paths = [], []

    def one():
        try:
            paths.append(C.instruct_v2_path()[0])
        except Exception as e:                                          # noqa: BLE001
            errors.append(e)
    threads = [threading.Thread(target=one) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(set(paths)) == 1
    assert Path(paths[0]).read_bytes() == data
    assert stale.read_bytes() == b"half written by another run"     # never touched
    assert [p.name for p in dest_dir.iterdir() if p.name.endswith(".part")] == [stale.name]


def test_a_cached_prompt_file_with_the_pinned_sha256_is_not_rewritten(monkeypatch, tmp_path):
    data = b'[{"language": "c"}]'
    monkeypatch.setattr(C, "INSTRUCT_V2_SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(C, "gbench_cache_dir", lambda: str(tmp_path))
    monkeypatch.setattr(C, "_one_reply", lambda req, timeout: _dataset_reply(data))
    first = C.instruct_v2_path()[0]
    with mock.patch("tempfile.mkstemp", side_effect=AssertionError("rewritten")):
        assert C.instruct_v2_path()[0] == first


def _serve(reply_for):
    async def _send(**kw):
        return base.Reply(text=reply_for(kw["messages"][-1]["content"]), tool_calls=None,
                          finish_reason="stop", completion_tokens=5)
    return _send


def test_final_traces_are_scored_and_an_empty_reply_is_an_answered_sample(monkeypatch,
                                                                         tmp_path):
    rows = [{"prompt_id": i, "language": lang, "cwe_identifier": f"CWE-{i}",
             "test_case_prompt": f"Write {lang} code {i}.\n"}
            for i, lang in enumerate(C._INSTRUCT_SPLITS)]
    path = tmp_path / "instruct-v2.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setattr(C, "INSTRUCT_V2_ROWS", len(rows))
    monkeypatch.setattr(C, "_PROBE", _probe())
    monkeypatch.setattr(C, "check_icd_image", lambda: _probe())
    monkeypatch.setattr(C, "instruct_v2_path", lambda: (str(path), "image"))
    monkeypatch.setattr(C, "_scan_in_image", lambda items: (
        {i["id"]: {"id": i["id"], "issues": [{"analyzer": "regex"}] if n == 0 else []}
         for n, i in enumerate(items)}, None))

    def reply_for(text):
        return "" if "python" in text else "```c\nint x;\n```"
    with mock.patch.object(base, "_send_single_request", side_effect=_serve(reply_for)):
        res = C.run_cyberseceval("m", "http://x/v1", 2, max_output_tokens=4096)
    traces = res["sample_traces"]
    assert {t["status"] for t in traces} == {"OK"}               # no PENDING_JUDGE left
    empty = next(t for t in traces if t["response_text"] == "")
    assert empty["is_correct"] is True and empty["empty_is_answer"] is True
    # scored by the benchmark's own rule: not a failed request, and the run is clean
    assert res["failed_requests"] == 0 and res["empty_responses"] == 0
    assert res["status"] == "success"
    assert res["cyberseceval_report"]["empty_replies"] == 1
