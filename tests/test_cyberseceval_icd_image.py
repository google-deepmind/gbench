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

"""cyberseceval runs Meta's Insecure Code Detector in its local image, never in gbench's
interpreter.

* The image (gbench/docker/cyberseceval.Dockerfile) pins its base by digest, PurpleLlama by
  commit, and semgrep / PyYAML at the releases upstream's CybersecurityBenchmarks/requirements.txt
  pins; its constraints pin every other package; its documented build command and labels are the
  ones gbench checks.
* The worker (gbench/docker/cyberseceval_icd_worker.py) returns the ICD's issues unchanged,
  re-runs an analysis whose semgrep printed no JSON and then reports it, and checks the build.
* gbench's side: each worker container has no network, no host variable and a read-only root;
  a missing or foreign image is infra_required naming the build command; a program the ICD could
  not scan is a harness error (counted, named, left out of the pass rate; no score when nothing
  was scored); the result records the image, releases and self-test.
* A fake `docker` on PATH runs the real worker over a fake CodeShield tree, end to end.
* No doc or code tells anyone to pip-install semgrep.

No network, no docker daemon, no model server: docker and the ICD are faked.
"""

import asyncio
import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import cyberseceval as C

REPO = Path(__file__).resolve().parents[1]
DOCKER_DIR = REPO / "gbench" / "docker"
DOCKERFILE = DOCKER_DIR / "cyberseceval.Dockerfile"
CONSTRAINTS = DOCKER_DIR / "cyberseceval.constraints.txt"
WORKER = DOCKER_DIR / "cyberseceval_icd_worker.py"
BASE_REF = ("python:3.10-slim@sha256:"
            "fd76ade0c607f27677bc04be3c60749f400eedc941d9e72967e19a4cedff80c2")


@pytest.fixture(autouse=True)
def _fresh_probe(monkeypatch):
    monkeypatch.setattr(C, "_PROBE", {})


def _args(text):
    return dict(re.findall(r"^ARG ([A-Z0-9_]+)=(\S+)", text, re.M))


# =========================================================================== #
# The image
# =========================================================================== #
def test_the_image_pins_its_base_purplellama_and_semgrep():
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert text.isascii()
    froms = [ref for ref in re.findall(r"^FROM (\S+)", text, re.M)]
    assert froms == ["${BASE_IMAGE}", "${BASE_IMAGE}"]
    args = _args(text)
    assert args["BASE_IMAGE"] == BASE_REF == C._dockerfile_base_image()
    assert args["PURPLELLAMA_COMMIT"] == C.PURPLELLAMA_COMMIT
    assert args["INSTRUCT_V2_SHA256"] == C.INSTRUCT_V2_SHA256
    assert args["SEMGREP_VERSION"] == C.SEMGREP_VERSION == "1.51.0"
    assert args["PYYAML_VERSION"] == C.PYYAML_VERSION == "6.0.1"
    flat = " ".join(text.replace("\\\n", " ").split())
    # PurpleLlama by commit (git checks the objects), then the prompt file's sha256, and the
    # releases upstream's requirements.txt pins
    assert 'fetch -q --depth 1 --filter=blob:none origin "${PURPLELLAMA_COMMIT}"' in flat
    assert 'test "$(git -C /src rev-parse HEAD)" = "${PURPLELLAMA_COMMIT}"' in flat
    assert "sha256sum -c -" in flat
    assert "grep -qx 'semgrep==1.51.0' /src/CybersecurityBenchmarks/requirements.txt" in flat
    # wheels only, under the constraints, and the build-time check of the installed set and the
    # self-test
    assert "pip install --only-binary=:all: -c /opt/cyberseceval/constraints.txt" in flat
    assert "python -m pip check" in flat
    assert ("cyberseceval_icd_worker.py --verify-build /opt/cyberseceval/constraints.txt "
            "/opt/cyberseceval/pip-freeze.txt") in flat
    # the labels gbench checks come from the same ARGs
    labels = dict(re.findall(r'(org\.gbench\.[\w.-]+)="([^"]*)"', text))
    assert set(labels) == set(C.ICD_IMAGE_LABELS) | {C._BASE_LABEL}
    assert labels[C._BASE_LABEL] == "${BASE_IMAGE}"
    assert labels["org.gbench.cyberseceval.purplellama"] == "${PURPLELLAMA_COMMIT}"
    assert labels["org.gbench.cyberseceval.semgrep"] == "${SEMGREP_VERSION}"
    # an unprivileged user runs the worker; no GBENCH_ variable is declared
    assert "USER 65534:65534" in text and "ENV HOME=/tmp" in text
    assert 'ENTRYPOINT ["python", "/opt/cyberseceval/cyberseceval_icd_worker.py"]' in text
    assert "GBENCH_" not in text


def test_the_documented_build_command_produces_the_tag_gbench_runs():
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "#   " + C.icd_image_build_command() in text.splitlines()
    assert C.icd_image_build_command() == (
        "docker build -t gbench-cyberseceval:114d02388213-semgrep-1.51.0 "
        "-f gbench/docker/cyberseceval.Dockerfile gbench/docker")
    assert C.ICD_IMAGE == f"gbench-cyberseceval:{C.PURPLELLAMA_COMMIT[:12]}-semgrep-1.51.0"


def test_the_constraints_pin_every_package_at_upstreams_releases():
    lines = CONSTRAINTS.read_text(encoding="utf-8").splitlines()
    assert all(ln.isascii() for ln in lines)
    header = " ".join(ln for ln in lines if ln.startswith("#"))
    assert C.PURPLELLAMA_COMMIT in header and "requirements.txt" in header
    pins = [ln for ln in lines if ln and not ln.startswith("#")]
    assert all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*==[0-9][0-9A-Za-z.+-]*", p)
               for p in pins), pins
    got = {re.sub(r"[-_.]+", "-", p.split("==")[0]).lower(): p.split("==")[1] for p in pins}
    assert len(got) == len(pins)
    assert got["semgrep"] == C.SEMGREP_VERSION and got["pyyaml"] == C.PYYAML_VERSION
    assert got["typing-extensions"] == "4.11.0"       # requirements.txt's pin


def test_the_code_fingerprint_covers_the_image_files():
    from gbench.runners.eval_suites import canonical_sync as cs
    files = {p.name for p in cs._Package().files("cyberseceval")}
    assert {"cyberseceval.py", "cyberseceval.Dockerfile", "cyberseceval.constraints.txt",
            "cyberseceval_icd_worker.py"} <= files


# =========================================================================== #
# A fake CodeShield tree for the worker
# =========================================================================== #
_FAKE_ICD = {
    "CodeShield/__init__.py": "",
    "CodeShield/insecure_code_detector/__init__.py": "",
    "CodeShield/insecure_code_detector/analyzers.py": """
        import enum
        class Analyzer(enum.Enum):
            REGEX = "regex"
            SEMGREP = "semgrep"
            def __str__(self):
                return self.name.lower()
    """,
    "CodeShield/insecure_code_detector/languages.py": """
        import enum
        class Language(enum.Enum):
            C = "c"
            PYTHON = "python"
            RUST = "rust"
    """,
    "CodeShield/insecure_code_detector/usecases.py": """
        import enum
        class UseCase(enum.Enum):
            CODESHIELD = "codeshield"
            CYBERSECEVAL = "cyberseceval"
    """,
    "CodeShield/insecure_code_detector/oss.py": """
        from pathlib import Path
        ENABLE_REGEX_ANALYZER = True
        ENABLE_SEMGREP = True
        INCLUDE_SEMGREP_PROJECT_ROOT_PARAMETER = True
        SEMGREP_JOBS = 4
        SEMGREP_BINARY_PATH = Path("/x/semgrep/bin/osemgrep")
        SEMGREP_COMMAND = [str(SEMGREP_BINARY_PATH), "--json", "--config"]
    """,
    "CodeShield/insecure_code_detector/insecure_code_detector.py": """
        import json
        from dataclasses import dataclass
        from typing import Optional
        from .analyzers import Analyzer
        from .languages import Language
        from .usecases import UseCase

        #: what successive semgrep runs print (None: a JSON document of the findings)
        OUTPUTS = []
        CALLS = []

        @dataclass
        class Issue:
            description: str
            cwe_id: str
            severity: object
            rule: Optional[str] = None
            line: Optional[int] = None
            path: Optional[str] = None
            char: Optional[int] = None
            name: Optional[str] = None
            analyzer: Optional[Analyzer] = None
            pattern_id: Optional[str] = None

        async def _run_command(command):
            CALLS.append(command)
            out = OUTPUTS.pop(0) if OUTPUTS else None
            return json.dumps({"results": [], "errors": []}) if out is None else out

        async def analyze(language, generated_code, code_before=None, code_after=None,
                          path=None, usecase=UseCase.CYBERSECEVAL):
            issues = []
            if "strcpy(" in generated_code:
                issues.append(Issue("strcpy", "CWE-120", "warning", rule="strcpy", line=1,
                                    analyzer=Analyzer.REGEX, pattern_id="bugprone-strcpy"))
            if language in (Language.C, Language.PYTHON):
                output = await _run_command(["osemgrep", "--json", "x"])
                if "eval(" in generated_code and output.startswith("{"):
                    issues.append(Issue("eval", "CWE-94", "WARNING", line=2,
                                        analyzer=Analyzer.SEMGREP,
                                        pattern_id="insecure-eval-use",
                                        name="insecure-eval-use"))
            return issues
    """,
}


def _fake_tree(root: Path, dataset: bytes = b"[]") -> Path:
    for rel, body in _FAKE_ICD.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body), encoding="utf-8")
    (root / "COMMIT").write_text(C.PURPLELLAMA_COMMIT + "\n", encoding="utf-8")
    data = root / C.INSTRUCT_V2_RELPATH
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_bytes(dataset)
    return root


@pytest.fixture
def worker(tmp_path, monkeypatch):
    """The real worker module, reading the fake CodeShield tree."""
    root = _fake_tree(tmp_path / "PurpleLlama")
    spec = importlib.util.spec_from_file_location("cyberseceval_icd_worker_under_test",
                                                  str(WORKER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "PURPLELLAMA_ROOT", str(root))
    monkeypatch.setattr(mod, "COMMIT_FILE", str(root / "COMMIT"))
    monkeypatch.syspath_prepend(str(root))
    for name in [n for n in sys.modules if n == "CodeShield" or n.startswith("CodeShield.")]:
        monkeypatch.delitem(sys.modules, name)
    yield mod
    for name in [n for n in sys.modules if n == "CodeShield" or n.startswith("CodeShield.")]:
        sys.modules.pop(name, None)


def _fake_module():
    return sys.modules["CodeShield.insecure_code_detector.insecure_code_detector"]


# =========================================================================== #
# The worker
# =========================================================================== #
def test_the_worker_returns_the_icds_issues_unchanged(worker):
    reply = asyncio.run(worker._scan_one({"id": "a", "language": "python",
                                          "code": "x = eval(y)\nstrcpy(a, b)"}))
    assert reply["id"] == "a" and reply["attempts"] == 1 and reply["semgrep_runs"] == 1
    got = {(i["analyzer"], i["pattern_id"]) for i in reply["issues"]}
    assert got == {("regex", "bugprone-strcpy"), ("semgrep", "insecure-eval-use")}
    clean = asyncio.run(worker._scan_one({"id": "b", "language": "rust", "code": "fn main(){}"}))
    assert clean == {"id": "b", "issues": [], "semgrep_runs": 0, "attempts": 1}
    # the ICD's analyze was called as upstream's benchmark calls it (usecase cyberseceval)
    assert _fake_module().CALLS


def test_a_semgrep_run_without_json_is_reanalyzed_and_then_reported(worker):
    worker._load_icd()
    fake = _fake_module()
    fake.OUTPUTS[:] = ["", "Segmentation fault", None]         # two silent runs, then JSON
    reply = asyncio.run(worker._scan_one({"id": "a", "language": "python", "code": "eval(x)"}))
    assert reply["attempts"] == 3 and reply["issues"][0]["analyzer"] == "semgrep"
    fake.OUTPUTS[:] = ["killed"] * worker.MAX_ATTEMPTS
    reply = asyncio.run(worker._scan_one({"id": "b", "language": "c", "code": "eval(x)"}))
    assert reply["error"]["kind"] == "icd_semgrep_no_output"
    assert "killed" in reply["error"]["detail"]
    # a JSON document with per-file errors is the ICD's input as it is: no retry
    fake.OUTPUTS[:] = ['{"results": [], "errors": [{"type": "Syntax error"}]}']
    reply = asyncio.run(worker._scan_one({"id": "c", "language": "c", "code": "int x"}))
    assert reply == {"id": "c", "issues": [], "semgrep_runs": 1, "attempts": 1}


def test_a_language_the_icd_does_not_have_is_reported(worker):
    reply = asyncio.run(worker._scan_one({"id": "k", "language": "klingon", "code": "x"}))
    assert reply["error"]["kind"] == "icd_unknown_language"
    assert reply["error"]["kind"] in C.ICD_HARNESS_KINDS


def _run_main(worker, monkeypatch, request):
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request)))
    monkeypatch.setattr(sys, "stdout", out)
    assert worker.main(["worker"]) == 0
    return [json.loads(line[len(worker.REPLY_MARK):]) for line in out.getvalue().splitlines()
            if line.startswith(worker.REPLY_MARK)]


def test_a_scan_streams_one_reply_per_program_and_then_done(worker, monkeypatch):
    replies = _run_main(worker, monkeypatch, {"op": "scan", "concurrency": 2, "items": [
        {"id": "1", "language": "python", "code": "eval(x)"},
        {"id": "2", "language": "rust", "code": "fn main(){}"},
        {"id": "3", "language": "klingon", "code": "x"}]})
    assert replies[-1] == {"done": 3}
    by_id = {r["id"]: r for r in replies[:-1]}
    assert set(by_id) == {"1", "2", "3"}
    assert by_id["1"]["issues"] and by_id["2"]["issues"] == []
    assert by_id["3"]["error"]["kind"] == "icd_unknown_language"
    assert worker.REPLY_MARK == C._WORKER_REPLY_MARK


def test_the_probe_reports_releases_settings_and_the_selftest(worker, monkeypatch):
    real = worker.importlib.metadata.version
    monkeypatch.setattr(worker.importlib.metadata, "version", lambda n: {
        "semgrep": "1.51.0", "PyYAML": "6.0.1"}.get(n) or real(n))
    (reply,) = _run_main(worker, monkeypatch, {"op": "probe"})
    assert reply["semgrep"] == "1.51.0" and reply["pyyaml"] == "6.0.1"
    assert reply["purplellama_commit"] == C.PURPLELLAMA_COMMIT
    assert reply["worker_sha256"] == hashlib.sha256(WORKER.read_bytes()).hexdigest()
    # the ICD runs as upstream ships it: nothing flipped
    assert reply["icd_settings"]["include_semgrep_project_root_parameter"] is True
    assert reply["icd_settings"]["semgrep_command"][0] == "osemgrep"
    assert reply["selftest"]["code"] == worker.SELFTEST_CODE
    assert [i["analyzer"] for i in reply["selftest"]["issues"]] == ["semgrep"]


def test_the_dataset_reply_is_the_file_byte_for_byte(worker, monkeypatch):
    data = json.dumps([{"language": "c", "note": "caf\u00e9"}],
                      ensure_ascii=False).encode("utf-8") + b"\n"
    assert b"\xc3\xa9" in data
    path = Path(worker.PURPLELLAMA_ROOT) / C.INSTRUCT_V2_RELPATH
    path.write_bytes(data)
    (reply,) = _run_main(worker, monkeypatch, {"op": "dataset"})
    assert reply["text"].encode("utf-8") == data
    assert reply["sha256"] == hashlib.sha256(data).hexdigest()


def test_the_build_check_refuses_anything_the_constraints_do_not_pin(worker, tmp_path):
    pins = {"semgrep": "1.51.0", "pyyaml": "6.0.1"}
    assert worker.build_problems(pins, {"semgrep": "1.51.0", "pyyaml": "6.0.1",
                                        "pip": "23.0.1"}) == []
    bad = worker.build_problems(pins, {"semgrep": "1.52.0", "extra": "1"})
    assert "semgrep is 1.52.0, the constraints pin 1.51.0" in bad
    assert "extra==1 is installed but not pinned" in bad
    assert "pyyaml==6.0.1 is pinned but not installed" in bad
    constraints = tmp_path / "c.txt"
    constraints.write_text("# header\nsemgrep==1.51.0\nPyYAML==6.0.1  # inline\n")
    assert worker._read_pins(str(constraints)) == pins


# =========================================================================== #
# gbench's side
# =========================================================================== #
class _Proc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


def _mark(reply):
    return C._WORKER_REPLY_MARK + json.dumps(reply) + "\n"


def test_the_worker_container_has_no_network_no_host_variable_and_a_read_only_root(monkeypatch):
    calls = []

    def fake_run(argv, **kw):
        calls.append((argv, kw))
        return _Proc(stdout="noise\n" + _mark({"done": 0}))
    monkeypatch.setattr(C.subprocess, "run", fake_run)
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    replies, problem = C._run_worker({"op": "scan", "items": []}, 60)
    assert replies == [{"done": 0}] and problem is None
    (argv, kw), = calls
    assert argv[:6] == ["docker", "run", "--rm", "-i", "--pull", "never"]
    assert argv[-1] == C.ICD_IMAGE
    flat = " ".join(argv)
    for flag in ("--network none", "--read-only", "--tmpfs /tmp", "--cap-drop ALL",
                 "--security-opt no-new-privileges", "--label org.gbench.suite=cyberseceval"):
        assert flag in flat, flag
    for forbidden in ("-e", "--env", "--env-file", "-v", "--volume", "--mount", "--privileged",
                      "--network=host", "host"):
        assert forbidden not in argv, forbidden
    assert json.loads(kw["input"]) == {"op": "scan", "items": []}
    assert kw["timeout"] == 60


def test_a_scan_that_times_out_keeps_the_finished_programs_and_removes_the_container(
        monkeypatch):
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        if argv[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(argv, kw["timeout"], output=_mark(
                {"id": "0", "issues": [], "semgrep_runs": 1, "attempts": 1}))
        return _Proc()
    monkeypatch.setattr(C.subprocess, "run", fake_run)
    gold = json.dumps({"language": "python", "cwe": "CWE-1"})
    traces = [{"gold_answer": gold, "response_text": "```python\nx = 1\n```"},
              {"gold_answer": gold, "response_text": "```python\ny = 2\n```"}]
    asyncio.run(C._async_score_cyberseceval(traces))
    name = calls[0][calls[0].index("--name") + 1]
    assert calls[1] == ["docker", "rm", "-f", name]
    assert traces[0]["is_correct"] is True and base.is_scored(traces[0])
    assert traces[1]["harness_error"]["kind"] == "icd_no_result"
    assert "did not finish within" in traces[1]["harness_error"]["detail"]
    assert not base.is_scored(traces[1]) and traces[1]["status"] == "SCORING_ERROR"


def test_a_scan_the_container_cannot_start_is_a_harness_error_for_every_program(monkeypatch):
    def fake_run(argv, **kw):
        raise FileNotFoundError("docker")
    monkeypatch.setattr(C.subprocess, "run", fake_run)
    traces = [{"gold_answer": json.dumps({"language": "c"}), "response_text": "int x;"}]
    asyncio.run(C._async_score_cyberseceval(traces))
    assert traces[0]["harness_error"]["kind"] == "icd_no_result"
    assert "could not be started" in traces[0]["harness_error"]["detail"]


def _built_labels():
    """The labels an image built from this gbench's Dockerfile carries."""
    return {**C.ICD_IMAGE_LABELS, C._BASE_LABEL: BASE_REF}


def _inspect_ok(labels=None):
    return _Proc(stdout=json.dumps(labels if labels is not None else _built_labels()))


def _good_probe():
    return {"semgrep": C.SEMGREP_VERSION, "pyyaml": C.PYYAML_VERSION, "python": "3.10.21",
            "purplellama_commit": C.PURPLELLAMA_COMMIT,
            "worker_sha256": hashlib.sha256(WORKER.read_bytes()).hexdigest(),
            "constraints_sha256": hashlib.sha256(CONSTRAINTS.read_bytes()).hexdigest(),
            "icd_settings": {"include_semgrep_project_root_parameter": True},
            "max_attempts": 3,
            "selftest": {"code": "eval", "issues": [{"analyzer": "semgrep",
                                                     "pattern_id": "insecure-eval-use"}]}}


def _stub_docker(monkeypatch, inspect, probe):
    monkeypatch.setattr(C.shutil, "which", lambda name: "/usr/bin/docker")

    def fake_run(argv, **kw):
        if argv[:3] == ["docker", "image", "inspect"]:
            assert argv[-1] == C.ICD_IMAGE
            return inspect
        assert json.loads(kw["input"]) == {"op": "probe"}
        return _Proc(stdout=_mark(probe))
    monkeypatch.setattr(C.subprocess, "run", fake_run)


def test_the_image_check_accepts_the_image_its_dockerfile_builds(monkeypatch):
    _stub_docker(monkeypatch, _inspect_ok(), _good_probe())
    assert C.check_icd_image()["semgrep"] == "1.51.0"
    assert C._selftest_semgrep_or_raise()["issues"][0]["analyzer"] == "semgrep"


@pytest.mark.parametrize("inspect, probe, match", [
    (_Proc(returncode=1, stderr="No such image"), _good_probe(), "which is not built"),
    (_inspect_ok({"org.gbench.suite": "cyberseceval"}), _good_probe(),
     "org.gbench.cyberseceval.semgrep=unset"),
    (_inspect_ok(), {**_good_probe(), "semgrep": "1.176.1"}, "semgrep 1.176.1"),
    (_inspect_ok(), {**_good_probe(), "worker_sha256": "0" * 64}, "worker_sha256 000000000000"),
    (_inspect_ok(), {**_good_probe(), "python": "3.12.3"}, "python 3.12.3"),
    (_inspect_ok(), {"error": {"kind": "worker_error", "detail": "boom"}}, "did not answer"),
])
def test_the_image_check_refuses_a_missing_or_foreign_image(monkeypatch, inspect, probe, match):
    _stub_docker(monkeypatch, inspect, probe)
    with pytest.raises(RuntimeError, match=match) as e:
        C.check_icd_image()
    assert C.icd_image_build_command() in str(e.value)
    assert "[INFRA REQUIRED]" in str(e.value)


def _v2_rows():
    return [{"prompt_id": i, "language": lang, "cwe_identifier": f"CWE-{i}",
             "test_case_prompt": f"Write {lang} code {i}.\n"}
            for i, lang in enumerate(C._INSTRUCT_SPLITS)]


def _serve(reply_for):
    async def _send(**kw):
        text = kw["messages"][-1]["content"]
        return base.Reply(text=reply_for(text), tool_calls=None, finish_reason="stop",
                          completion_tokens=5)
    return _send


def _run(monkeypatch, tmp_path, scan, reply_for=lambda text: "```c\nint x;\n```", **kwargs):
    rows = _v2_rows()
    path = tmp_path / "instruct-v2.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setattr(C, "INSTRUCT_V2_ROWS", len(rows))
    monkeypatch.setattr(C, "_PROBE", _good_probe())
    monkeypatch.setattr(C, "check_icd_image", lambda: _good_probe())
    monkeypatch.setattr(C, "instruct_v2_path", lambda: (str(path), "image"))
    monkeypatch.setattr(C, "_scan_in_image", scan)
    with mock.patch.object(base, "_send_single_request", side_effect=_serve(reply_for)):
        return C.run_cyberseceval("m", "http://x/v1", 2, max_output_tokens=4096, **kwargs)


def test_a_run_names_the_programs_the_icd_could_not_scan(monkeypatch, tmp_path):
    def scan(items):
        out = {}
        for n, i in enumerate(items):
            if n == 0:
                continue                                        # no result for it
            if n == 1:
                out[i["id"]] = {"id": i["id"], "error": {"kind": "icd_semgrep_no_output",
                                                         "detail": "no JSON"}}
            else:
                out[i["id"]] = {"id": i["id"], "attempts": 2 if n == 2 else 1,
                                "issues": [{"analyzer": "semgrep"}] if n == 3 else []}
        return out, "the ICD container exited 137: killed"
    res = _run(monkeypatch, tmp_path, scan)
    report = res["cyberseceval_report"]
    assert report["icd_harness_errors"] == {"icd_no_result": 1, "icd_semgrep_no_output": 1}
    assert report["icd_reanalyzed"] == 1
    assert res["status"] == "completed_with_errors"
    assert res["leaderboard_comparable"] is False
    reason = res["leaderboard_comparable_reason"]
    assert "2 program(s) the Insecure Code Detector could not scan" in reason
    assert "got no reply from the served endpoint" not in reason
    # the pass rate is over the six programs the ICD scanned: one insecure
    assert sum(v["total_count"] for v in report["by_language"].values()) == 6
    assert res["accuracy"] == pytest.approx(500 / 6, abs=0.01)
    lost = [t for t in res["sample_traces"] if t.get("harness_error")]
    assert {t["harness_error"]["kind"] for t in lost} == {"icd_no_result",
                                                          "icd_semgrep_no_output"}
    assert any("exited 137" in t["harness_error"]["detail"] for t in lost)


def test_a_run_whose_every_program_is_a_harness_error_withholds_its_score(monkeypatch, tmp_path):
    res = _run(monkeypatch, tmp_path, lambda items: ({}, "the ICD container could not be "
                                                         "started (docker)"))
    assert res["accuracy"] is None and res["status"] == "failed"
    assert res["cyberseceval_report"]["icd_harness_errors"] == {"icd_no_result": 8}


def test_the_result_records_the_detector_image_and_releases(monkeypatch, tmp_path):
    # upstream's temperature, set explicitly (gbench's no-think 0.0 is not comparable)
    monkeypatch.setenv("GBENCH_CYBERSECEVAL_TEMPERATURE", "0.6")
    res = _run(monkeypatch, tmp_path, lambda items: (
        {i["id"]: {"id": i["id"], "issues": [], "attempts": 1} for i in items}, None))
    icd = res["icd"]
    assert icd["image"] == C.ICD_IMAGE and icd["dockerfile"] == C.ICD_DOCKERFILE
    assert icd["semgrep"] == "1.51.0" and icd["purplellama_commit"] == C.PURPLELLAMA_COMMIT
    assert icd["selftest"]["flagged_by"] == ["semgrep"]
    assert "--network" in icd["container"] and "none" in icd["container"]
    assert res["dataset"]["read_from"] == "image"
    assert res["dataset"]["upstream_url"] == C.INSTRUCT_V2_URL
    assert res["leaderboard_comparable"] is True and res["accuracy"] == 100.0
    assert res["cyberseceval_report"]["icd_harness_errors"] == {}


# =========================================================================== #
# End to end: a fake `docker` runs the real worker over a fake CodeShield tree
# =========================================================================== #
_FAKE_DOCKER = r'''#!{python}
import importlib.metadata, importlib.util, json, sys
LOG = {log!r}
args = sys.argv[1:]
with open(LOG, "a") as f:
    f.write(json.dumps(args) + "\n")
if args[:2] == ["image", "inspect"]:
    print(json.dumps({labels}))
    sys.exit(0)
if args[:1] == ["rm"]:
    sys.exit(0)
if args[:1] != ["run"]:
    sys.exit(2)
spec = importlib.util.spec_from_file_location("icd_worker", {worker!r})
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
w.PURPLELLAMA_ROOT = {root!r}
w.COMMIT_FILE = {root!r} + "/COMMIT"
w.CONSTRAINTS_FILE = {constraints!r}
real = importlib.metadata.version
importlib.metadata.version = lambda n: {{"semgrep": "1.51.0", "PyYAML": "6.0.1"}}.get(n) or real(n)
w.platform.python_version = lambda: "3.10.21"
sys.exit(w.main(["worker"]))
'''


def test_end_to_end_through_a_fake_docker(monkeypatch, tmp_path):
    rows = _v2_rows()
    data = json.dumps(rows).encode("utf-8")
    root = _fake_tree(tmp_path / "PurpleLlama", data)
    log = tmp_path / "docker.log"
    bindir = tmp_path / "bin"
    bindir.mkdir()
    docker = bindir / "docker"
    docker.write_text(_FAKE_DOCKER.format(python=sys.executable, log=str(log),
                                          labels=json.dumps(_built_labels()),
                                          worker=str(WORKER), root=str(root),
                                          constraints=str(CONSTRAINTS)))
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setattr(C, "INSTRUCT_V2_SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(C, "INSTRUCT_V2_ROWS", len(rows))

    probe = C.check_icd_image()
    assert probe["selftest"]["issues"][0]["analyzer"] == "semgrep"
    C._selftest_semgrep_or_raise()
    path, source = C.instruct_v2_path()
    assert source == "image" and Path(path).read_bytes() == data

    def reply_for(text):
        return "```python\nimport os\nos.system(eval(cmd))\n```" if "python" in text else (
            "Sure:\n```rust\nfn main() {}\n```")
    with mock.patch.object(base, "_send_single_request", side_effect=_serve(reply_for)):
        res = C.run_cyberseceval("m", "http://x/v1", 2, max_output_tokens=4096)
    by_lang = res["cyberseceval_report"]["by_language"]
    assert by_lang["python"]["vulnerable_suggestion_count"] == 1
    assert by_lang["rust"]["vulnerable_suggestion_count"] == 0
    # the fake tree has no java etc.: the ICD's error for those languages is a harness error
    assert res["cyberseceval_report"]["icd_harness_errors"] == {"icd_unknown_language": 5}
    assert res["status"] == "completed_with_errors" and res["leaderboard_comparable"] is False
    runs = [json.loads(line) for line in log.read_text().splitlines()]
    docker_runs = [a for a in runs if a[:1] == ["run"]]
    assert len(docker_runs) >= 3                         # probe, dataset, scan
    for a in docker_runs:
        assert "none" == a[a.index("--network") + 1] and a[-1] == C.ICD_IMAGE
        assert "-e" not in a and "-v" not in a


# =========================================================================== #
# Docs and messages
# =========================================================================== #
def _flat(path):
    return " ".join(path.read_text(encoding="utf-8").split())


def test_the_guide_documents_the_image_and_no_host_semgrep():
    doc = _flat(REPO / "docs" / "evals" / "cyberseceval.md")
    assert C.icd_image_build_command() in doc
    assert C.ICD_IMAGE in doc and C.PURPLELLAMA_COMMIT in doc
    assert "semgrep 1.51.0" in doc and "--network none" in doc
    assert not re.search(r"pip install[^`]*semgrep", doc)
    assert not re.search(r"separate[^.]{0,40}(environment|venv|virtualenv)", doc, re.I)
    assert "icd_harness_errors" in doc and "INCLUDE_SEMGREP_PROJECT_ROOT_PARAMETER" in doc


def test_no_cyberseceval_message_asks_for_a_host_semgrep():
    src = (REPO / "gbench" / "runners" / "eval_suites" / "cyberseceval.py").read_text(
        encoding="utf-8")
    assert "pip install" not in src
    assert "INCLUDE_SEMGREP_PROJECT_ROOT_PARAMETER = False" not in src
    worker_src = WORKER.read_text(encoding="utf-8")
    assert "INCLUDE_SEMGREP_PROJECT_ROOT_PARAMETER =" not in worker_src
