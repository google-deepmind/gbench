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

"""The Insecure Code Detector worker of gbench's cyberseceval suite (the image built from
gbench/docker/cyberseceval.Dockerfile).

The image holds Meta's PurpleLlama at the commit gbench pins (CodeShield/, the ICD, and the
instruct-v2.json prompt file) and semgrep at the release upstream's
CybersecurityBenchmarks/requirements.txt pins, on the Python version its README names. semgrep
1.51.0 cannot be installed next to gbench (it pins urllib3~=1.26 and ruamel.yaml<0.18, which would
downgrade the serving environment's urllib3 2.x and ruamel.yaml, and upstream requires Python
3.10), so the ICD runs here: gbench starts one container per request with no network, no credential, a
read-only file system and an unprivileged user, and reads the replies from stdout.

One JSON request on stdin; each reply is one line on stdout that starts with REPLY_MARK:

* {"op": "probe"}: the semgrep, PyYAML and Python releases, the PurpleLlama commit, this file's
  sha256, the sha256 of the constraints file the image was built with (CONSTRAINTS_FILE), the
  ICD settings this worker runs with, and the self-test: the ICD's issues for
  SELFTEST_CODE (Python `eval(user_input)`, which only the semgrep analyzer flags).
* {"op": "dataset"}: {"relpath", "sha256", "text"} of the pinned instruct-v2.json.
* {"op": "scan", "items": [{"id", "language", "code"}, ...], "concurrency": N}: for each item,
  as it finishes, {"id", "issues": [...], "semgrep_runs", "attempts"} or
  {"id", "error": {"kind", "detail"}}; then {"done": <number of items>}.

A scan calls `insecure_code_detector.analyze(Language(language), code,
usecase=UseCase.CYBERSECEVAL)`, as upstream's instruct_or_autocomplete_benchmark.py does, and
returns its issues unchanged. The ICD runs semgrep as a subprocess and reads its JSON output;
when a semgrep run prints no JSON at all (the process crashed or was killed), the ICD finds no
semgrep issue and would call the program secure. This worker notices that (it wraps the ICD's
_run_command and only looks at the output) and runs the analysis again, up to MAX_ATTEMPTS times
(upstream's own ICD tests retry a suspected transient osemgrep failure three times); an item
that still has a semgrep run without output is reported as the error `icd_semgrep_no_output`,
which gbench counts as a harness error. A semgrep run that prints JSON (with or without
per-file errors) is used as the ICD parses it.

Build-time check: `python cyberseceval_icd_worker.py --verify-build <constraints> <freeze out>`
fails (exit 1, naming each problem) unless the installed distributions are exactly the pinned
set, the ICD imports (which links osemgrep to semgrep-core in the semgrep package, as upstream's
oss.py does on import, while the build still runs as root) and the self-test is flagged by the
semgrep analyzer. It writes the installed set, one `name==version` per line, to <freeze out>.
"""

import asyncio
import contextvars
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
from typing import Any, Dict, List, Optional

REPLY_MARK = "ICD_WORKER_REPLY="
#: PurpleLlama's checkout in the image (the repository root: upstream imports
#: CodeShield.insecure_code_detector from it) and the file that records its commit.
PURPLELLAMA_ROOT = "/opt/PurpleLlama"
COMMIT_FILE = os.path.join(PURPLELLAMA_ROOT, "COMMIT")
#: The constraints file the image's packages were installed under (the Dockerfile copies it).
CONSTRAINTS_FILE = "/opt/cyberseceval/constraints.txt"
INSTRUCT_V2_RELPATH = "CybersecurityBenchmarks/datasets/instruct/instruct-v2.json"
#: The self-test program and its language: only the semgrep analyzer flags it.
SELFTEST_LANGUAGE = "python"
SELFTEST_CODE = "def handler(user_input):\n    return eval(user_input)\n"
#: Analyses of one item before a semgrep run without output is reported as an error.
MAX_ATTEMPTS = 3
#: Installer tooling of the base image, not part of the pinned set.
TOOLING = frozenset({"pip", "setuptools", "wheel"})

#: The semgrep runs of the analysis in progress: [{"json": bool, "tail": str}, ...].
_RUNS: contextvars.ContextVar = contextvars.ContextVar("icd_semgrep_runs")
_ICD: Dict[str, Any] = {}


def _has_json(output: str) -> bool:
    """Whether a semgrep run's output holds the JSON document the ICD parses
    (insecure_code_detector._parse_semgrep_issues reads from the first '{')."""
    start = output.find("{")
    if start < 0:
        return False
    try:
        json.loads(output[start:])
    except ValueError:
        return False
    return True


def _load_icd() -> Dict[str, Any]:
    """Import the ICD from PURPLELLAMA_ROOT as upstream's benchmark does, and observe (never
    change) the output of each semgrep run it starts."""
    if _ICD:
        return _ICD
    if PURPLELLAMA_ROOT not in sys.path:
        sys.path.insert(0, PURPLELLAMA_ROOT)
    from CodeShield.insecure_code_detector import insecure_code_detector as icd
    from CodeShield.insecure_code_detector import oss
    from CodeShield.insecure_code_detector.languages import Language
    from CodeShield.insecure_code_detector.usecases import UseCase

    original = icd._run_command

    async def _observed_run_command(command: List[str]) -> str:
        output = await original(command)
        runs = _RUNS.get(None)
        if runs is not None:
            ok = _has_json(output)
            runs.append({"json": ok, "tail": "" if ok else " ".join(output.split())[-300:]})
        return output

    icd._run_command = _observed_run_command
    _ICD.update(icd=icd, oss=oss, Language=Language, UseCase=UseCase)
    return _ICD


def _issue(issue: Any) -> Dict[str, Any]:
    """An ICD Issue as JSON (enums as their str(), as upstream prints them)."""
    def _s(value: Any) -> Optional[str]:
        return None if value is None else str(value)
    return {"description": issue.description, "cwe_id": issue.cwe_id,
            "severity": _s(issue.severity), "rule": issue.rule, "line": issue.line,
            "char": issue.char, "name": issue.name, "pattern_id": issue.pattern_id,
            "analyzer": _s(issue.analyzer), "path": issue.path}


async def _scan_one(item: Dict[str, Any]) -> Dict[str, Any]:
    """The ICD's verdict on one program: {"id", "issues", "semgrep_runs", "attempts"} or
    {"id", "error"}."""
    key = item.get("id")
    lib = _load_icd()
    try:
        language = lib["Language"](str(item.get("language") or ""))
    except ValueError:
        return {"id": key, "error": {"kind": "icd_unknown_language",
                                     "detail": f"the ICD has no language {item.get('language')!r}"}}
    code = str(item.get("code") or "")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        runs: List[Dict[str, Any]] = []
        token = _RUNS.set(runs)
        try:
            issues = await lib["icd"].analyze(language, code,
                                              usecase=lib["UseCase"].CYBERSECEVAL)
        except Exception as e:                                          # noqa: BLE001
            return {"id": key, "error": {"kind": "icd_error",
                                         "detail": f"{type(e).__name__}: {e}"[:300]}}
        finally:
            _RUNS.reset(token)
        silent = [r for r in runs if not r["json"]]
        if not silent:
            return {"id": key, "issues": [_issue(i) for i in issues],
                    "semgrep_runs": len(runs), "attempts": attempt}
    return {"id": key, "error": {
        "kind": "icd_semgrep_no_output",
        "detail": (f"semgrep printed no JSON in {MAX_ATTEMPTS} attempts; last output: "
                   f"{silent[-1]['tail'] or 'none'}")}}


def _emit(reply: Dict[str, Any]) -> None:
    sys.stdout.write(REPLY_MARK + json.dumps(reply, ensure_ascii=True, default=str) + "\n")
    sys.stdout.flush()


async def _scan(items: List[Dict[str, Any]], concurrency: int) -> int:
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(item: Dict[str, Any]) -> None:
        async with sem:
            reply = await _scan_one(item)
        _emit(reply)

    await asyncio.gather(*[_one(item) for item in items])
    return len(items)


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _commit() -> Optional[str]:
    try:
        with open(COMMIT_FILE, encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


def probe() -> Dict[str, Any]:
    """Releases, settings and the self-test (see the module docstring)."""
    lib = _load_icd()
    selftest = asyncio.run(_scan_one({"id": "selftest", "language": SELFTEST_LANGUAGE,
                                      "code": SELFTEST_CODE}))
    oss = lib["oss"]
    return {
        "semgrep": importlib.metadata.version("semgrep"),
        "pyyaml": importlib.metadata.version("PyYAML"),
        "python": platform.python_version(),
        "purplellama_commit": _commit(),
        "worker_sha256": _sha256(os.path.abspath(__file__)),
        "constraints_sha256": (_sha256(CONSTRAINTS_FILE) if os.path.isfile(CONSTRAINTS_FILE)
                               else None),
        "icd_settings": {
            "enable_regex_analyzer": bool(oss.ENABLE_REGEX_ANALYZER),
            "enable_semgrep": bool(oss.ENABLE_SEMGREP),
            "include_semgrep_project_root_parameter":
                bool(oss.INCLUDE_SEMGREP_PROJECT_ROOT_PARAMETER),
            "semgrep_jobs": int(oss.SEMGREP_JOBS),
            "semgrep_command": [os.path.basename(str(oss.SEMGREP_BINARY_PATH))]
                               + [str(a) for a in oss.SEMGREP_COMMAND[1:]],
        },
        "max_attempts": MAX_ATTEMPTS,
        "selftest": {"language": SELFTEST_LANGUAGE, "code": SELFTEST_CODE, **selftest},
    }


def dataset() -> Dict[str, Any]:
    path = os.path.join(PURPLELLAMA_ROOT, INSTRUCT_V2_RELPATH)
    with open(path, "rb") as f:
        data = f.read()
    return {"relpath": INSTRUCT_V2_RELPATH, "sha256": hashlib.sha256(data).hexdigest(),
            "text": data.decode("utf-8")}


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _read_pins(path: str) -> Dict[str, str]:
    pins: Dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            name, sep, version = line.partition("==")
            if not sep or not name.strip() or not version.strip():
                raise SystemExit(f"cyberseceval_icd_worker: not a name==version pin: {raw!r}")
            pins[_canonical(name.strip())] = version.strip()
    return pins


def build_problems(pins: Dict[str, str], found: Dict[str, str]) -> List[str]:
    """What keeps the installed set `found` from being exactly the pinned set `pins`."""
    out = []
    for name, version in sorted(found.items()):
        if name in TOOLING:
            continue
        if name not in pins:
            out.append(f"{name}=={version} is installed but not pinned")
        elif pins[name] != version:
            out.append(f"{name} is {version}, the constraints pin {pins[name]}")
    out += [f"{name}=={pins[name]} is pinned but not installed"
            for name in sorted(set(pins) - set(found))]
    return out


def verify_build(constraints: str, freeze_out: str) -> int:
    found = {_canonical(d.metadata["Name"]): d.version
             for d in importlib.metadata.distributions()}
    problems = build_problems(_read_pins(constraints), found)
    report: Dict[str, Any] = {}
    if not problems:
        report = probe()
        issues = report["selftest"].get("issues")
        if not any((i or {}).get("analyzer") == "semgrep" for i in issues or []):
            problems.append("the self-test found no semgrep issue for Python eval(user_input): "
                            f"{json.dumps(report['selftest'])[:400]}")
        if not report.get("purplellama_commit"):
            problems.append(f"{COMMIT_FILE} is missing")
    if problems:
        for p in problems:
            print("cyberseceval_icd_worker --verify-build: " + p, file=sys.stderr)
        return 1
    with open(freeze_out, "w", encoding="utf-8") as f:
        f.write("".join(f"{n}=={v}\n" for n, v in sorted(found.items()) if n not in TOOLING))
    print("cyberseceval_icd_worker --verify-build: OK " + json.dumps(
        {k: report[k] for k in ("semgrep", "pyyaml", "python", "purplellama_commit")}))
    return 0


def main(argv: List[str]) -> int:
    if len(argv) > 1 and argv[1] == "--verify-build":
        return verify_build(argv[2], argv[3])
    try:
        req = json.loads(sys.stdin.read() or "{}")
        op = req.get("op")
        if op == "probe":
            _emit(probe())
        elif op == "dataset":
            _emit(dataset())
        elif op == "scan":
            items = [i for i in (req.get("items") or []) if isinstance(i, dict)]
            done = asyncio.run(_scan(items, int(req.get("concurrency") or 1)))
            _emit({"done": done})
        else:
            _emit({"error": {"kind": "bad_request", "detail": f"unknown op {op!r}"}})
    except Exception as e:  # noqa: BLE001 - the host names every failure
        _emit({"error": {"kind": "worker_error", "detail": f"{type(e).__name__}: {e}"[:300]}})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
