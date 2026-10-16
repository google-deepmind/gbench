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

"""pytest plugin for CI: a test must never skip because a DECLARED dependency is missing.

Load it with `PYTHONPATH=scripts pytest -p ci_skip_gate` (the Unit Tests workflow does). It
records every module a `pytest.importorskip` call could not import (at collection or in a test)
and every skip whose reason says "No module named 'x'" or "'x' is not installed", and fails the
session (exit status 1)
when that module belongs to a distribution pyproject.toml declares in [project.dependencies] or
in the extras named by --skip-gate-extras (default: dev,evals). Such a skip means the CI install
is incomplete and the test silently stopped protecting anything. Modules nothing declares (an
optional serving stack, a harness that runs in its own image) are reported, not failed. A
requirement whose environment marker is false on the running interpreter (for example a
requirement marked `python_full_version >= '3.12'` on Python 3.11) is not installed there, so it
counts as undeclared on that interpreter.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set

import pytest
from packaging.markers import InvalidMarker, Marker

if sys.version_info >= (3, 11):
    import tomllib
else:                                                                     # pragma: no cover
    tomllib = None

#: Distributions whose import name is not their normalized project name.
IMPORT_NAMES: Dict[str, List[str]] = {
    "bfcl-eval": ["bfcl_eval"],
    "google-genai": ["google.genai"],
    "mini-swe-agent": ["minisweagent"],
    "python-docx": ["docx"],
    "python-pptx": ["pptx"],
    "pymupdf": ["fitz", "pymupdf"],
    "scikit-learn": ["sklearn"],
    "pyyaml": ["yaml"],
    "pillow": ["PIL"],
    "multi-swe-bench": ["multi_swe_bench"],
    "pytest-asyncio": ["pytest_asyncio"],
}

_NO_MODULE = re.compile(r"No module named '([\w.]+)'")
#: A skip reason that names a missing package in words: "'x' is not installed", "package x is not
#: installed". The name counts only when pyproject.toml declares it (owning_distribution).
_NOT_INSTALLED = re.compile(r"(?:package |module )?'?([A-Za-z_][\w.]*)'? is not installed")
_REPO = Path(__file__).resolve().parent.parent


def _dist_name(requirement: str) -> str:
    name = requirement.split(";")[0].split("[")[0]
    for sep in ("<", ">", "=", "!", "~", " "):
        name = name.split(sep)[0]
    return name.strip().lower().replace("_", "-")


def applies_here(requirement: str, extra: str = "") -> bool:
    """Whether `requirement`'s environment marker (the part after ';') holds on the running
    interpreter; True when it has none. An unparsable marker counts as applying, so a typo can
    never exempt a dependency from the gate."""
    _, sep, marker = requirement.partition(";")
    if not sep or not marker.strip():
        return True
    try:
        return Marker(marker.strip()).evaluate({"extra": extra})
    except InvalidMarker:
        return True


def declared_modules(pyproject: Path, extras: Iterable[str]) -> Dict[str, str]:
    """{import name: distribution} for [project.dependencies] and the given extras, leaving out
    a requirement whose environment marker is false on the running interpreter (pip installs
    nothing for it there)."""
    if tomllib is None:                                                   # pragma: no cover
        raise RuntimeError("ci_skip_gate needs Python 3.11 or later (tomllib)")
    with open(pyproject, "rb") as f:
        project = tomllib.load(f)["project"]
    reqs = [(r, "") for r in project.get("dependencies") or []]
    for extra in extras:
        reqs += [(r, extra) for r in
                 (project.get("optional-dependencies") or {}).get(extra) or []]
    out: Dict[str, str] = {}
    for req, extra in reqs:
        if not applies_here(req, extra):
            continue
        dist = _dist_name(req)
        for module in IMPORT_NAMES.get(dist, [dist.replace("-", "_")]):
            out[module] = dist
    return out


def owning_distribution(module: str, declared: Dict[str, str]) -> str:
    """The declared distribution `module` (or a parent package of it) belongs to, else ''."""
    parts = module.split(".")
    for i in range(len(parts), 0, -1):
        dist = declared.get(".".join(parts[:i]))
        if dist:
            return dist
    return ""


_MISSING: Set[str] = set()
_REAL_IMPORTORSKIP = pytest.importorskip


def _recording_importorskip(modname, *args, **kwargs):
    __tracebackhide__ = True        # the skip is reported at the test's own line
    try:
        return _REAL_IMPORTORSKIP(modname, *args, **kwargs)
    except pytest.skip.Exception:
        _MISSING.add(str(modname))
        raise


def pytest_addoption(parser):
    parser.addoption("--skip-gate-pyproject", default=str(_REPO / "pyproject.toml"),
                     help="pyproject.toml whose declared dependencies may not cause a skip")
    parser.addoption("--skip-gate-extras", default="dev,evals",
                     help="comma-separated extras counted as declared (default: dev,evals)")


def pytest_configure(config):
    pytest.importorskip = _recording_importorskip


def pytest_unconfigure(config):
    pytest.importorskip = _REAL_IMPORTORSKIP


def _record_skip(report) -> None:
    if report.skipped and isinstance(report.longrepr, tuple) and len(report.longrepr) == 3:
        reason = str(report.longrepr[2])
        for pattern in (_NO_MODULE, _NOT_INSTALLED):
            m = pattern.search(reason)
            if m:
                _MISSING.add(m.group(1))


def pytest_runtest_logreport(report):
    _record_skip(report)


def pytest_collectreport(report):
    _record_skip(report)


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    config = session.config
    extras = [e.strip() for e in config.getoption("--skip-gate-extras").split(",") if e.strip()]
    declared = declared_modules(Path(config.getoption("--skip-gate-pyproject")), extras)
    bad = sorted((m, owning_distribution(m, declared)) for m in _MISSING
                 if owning_distribution(m, declared))
    session.config._ci_skip_gate = {"missing": sorted(_MISSING), "declared_missing": bad,
                                    "extras": extras}
    if bad and session.exitstatus in (0, 5):     # passed, or every test skipped
        session.exitstatus = 1


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    info = getattr(config, "_ci_skip_gate", None)
    if not info or not info["missing"]:
        return
    terminalreporter.section("ci_skip_gate")
    for module, dist in info["declared_missing"]:
        terminalreporter.write_line(
            f"FAIL: tests skipped because '{module}' is missing, but pyproject.toml declares "
            f"{dist}: install it in CI (pip install -e \".[{','.join(info['extras'])}]\")")
    others = [m for m in info["missing"] if m not in {b[0] for b in info["declared_missing"]}]
    if others:
        terminalreporter.write_line("note: skips from modules nothing declares for this "
                                    f"interpreter: {', '.join(others)}")
