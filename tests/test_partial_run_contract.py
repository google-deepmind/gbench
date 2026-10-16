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

"""A degraded run must SAY SO in the result, not only in a log line.

`skipped_result` already covers "the prerequisite is missing, so nothing ran". The gap was
the middle case: a prerequisite missing that removes PART of a benchmark. `aider_polyglot`
dropped every Java exercise for want of `gradle`, logged one warning, and published a
5-language score whose JSON was indistinguishable from a complete 6-language run.

Two invariants:
  1. a warning that announces degradation must cite the suite's docs page, so the reader
     knows where the prerequisites are written down;
  2. the docs page must actually say how to install them.
"""

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SUITES = os.path.join(os.path.dirname(__file__), "..", "gbench", "runners", "eval_suites")
DOCS = os.path.join(os.path.dirname(__file__), "..", "docs", "evals")

#: Warning text that announces a run measured less than the benchmark defines.
_DEGRADED = re.compile(r"skip|missing|not installed|unavailable|no .*toolchain", re.I)

#: Suites whose degradation warnings are not about a missing PREREQUISITE - they report
#: data/agent conditions the operator cannot install their way out of. Listed explicitly so
#: the exemption is a decision rather than a gap.
_NOT_PREREQ = {
    "base.py",            # --attempt-count unsupported by a batch-scored suite
    "custom_jsonl.py",    # a malformed row in the operator's own file
    "swe_bench_pro.py",   # upstream table missing harness columns
    "coco_caption.py",    # METEOR needs a JRE; reported per-metric, not per-run
    "sandbox.py",         # process-wide, has no single suite or docs page
}


def _suite_files():
    for f in sorted(os.listdir(SUITES)):
        if f.endswith(".py") and not f.startswith("_"):
            yield f


def test_degradation_warnings_cite_their_docs_page():
    """The exact defect the user hit: `aider_polyglot: skipping languages with missing
    toolchains: ['java (gradle not installed...)']` told you what broke but not where the
    fix is written down."""
    offenders = []
    for f in _suite_files():
        if f in _NOT_PREREQ:
            continue
        src = open(os.path.join(SUITES, f), encoding="utf-8").read()
        for m in re.finditer(r'logger\.warning\(\s*(f?["\'].{0,200})', src, re.S):
            if not _DEGRADED.search(m.group(1)):
                continue
            window = src[m.start():m.start() + 500]
            if "DOCS_URL" not in window and "partial_run(" not in window:
                line = src[:m.start()].count("\n") + 1
                offenders.append(f"{f}:{line}  {m.group(1)[:70].strip()}")
    assert not offenders, (
        "these announce a degraded run without pointing at the docs:\n  "
        + "\n  ".join(offenders))


def test_multipl_e_delegates_to_the_official_container():
    """multipl_e now runs inside the official MultiPL-E evaluator image (all 24 language
    toolchains), like aider_polyglot - so there is no host-toolchain SUBSET / partial_run anymore.
    It HARD-ERRORS (infra_required) if Docker or the locally-built image is missing, and records
    the languages it ran + honest leaderboard_comparable."""
    src = open(os.path.join(SUITES, "multipl_e.py"), encoding="utf-8").read()
    assert "partial_run(" not in src, "multipl_e no longer subsets by host toolchain"
    assert "infra_required(" in src
    assert "languages_evaluated" in src
    assert "leaderboard_comparable" in src


def test_partial_run_marks_the_result_not_comparable():
    from gbench.runners.eval_suites.swebench_common import partial_run
    out = partial_run("x", "java excluded", "docs/evals/x.md", languages_skipped=["java"])
    assert out["partial_run"] is True
    assert out["leaderboard_comparable"] is False
    assert "docs/evals/x.md" in out["partial_reason"]
    assert out["languages_skipped"] == ["java"]


def test_aider_polyglot_doc_says_how_to_build_the_container():
    """aider_polyglot now delegates to aider's own benchmark container (which bundles the 6
    toolchains), so the doc must teach the container build + exercises, not host apt installs."""
    doc = open(os.path.join(DOCS, "aider_polyglot.md"), encoding="utf-8").read()
    assert "docker build" in doc and "benchmark/Dockerfile" in doc
    assert "polyglot-benchmark" in doc and "GBENCH_AIDER_BENCHMARK_DIR" in doc


def test_every_suite_with_a_docs_url_has_the_doc():
    missing = []
    for f in _suite_files():
        src = open(os.path.join(SUITES, f), encoding="utf-8").read()
        m = re.search(r'DOCS_URL\s*=\s*["\']docs/evals/([\w.]+\.md)["\']', src)
        if m and not os.path.isfile(os.path.join(DOCS, m.group(1))):
            missing.append(f"{f} -> docs/evals/{m.group(1)}")
    assert not missing, "DOCS_URL points at a file that does not exist:\n  " + "\n  ".join(missing)


def test_no_undefined_names_anywhere_in_gbench():
    """Twice in two days a name was used and never bound: `repetition_run` in
    terminal_bench (every trial parse raised NameError, silently swallowed at debug level)
    and `DOCS_URL` in an f-string in another suite. Both are invisible until the line executes.
    A hand-rolled AST walk got this wrong (it flagged comprehension variables), so defer to
    ruff's F821, which does real scope analysis.

    This also caught two pre-existing ones: a branch calling a
    helper that has never existed, and a missing `import sys` in scicode's sandbox path.
    """
    import subprocess
    pytest.importorskip("ruff")         # declared (dev extra): CI's skip gate fails a skip here
    root = os.path.join(os.path.dirname(__file__), "..")
    r = subprocess.run([sys.executable, "-m", "ruff", "check", "--select", "F821",
                        "--output-format", "concise", "gbench/"],
                       cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, f"undefined names:\n{r.stdout}{r.stderr}"


# --- offline-scorable capture of the emitted call ---------------------------------------
def test_emitted_function_calls_are_diffable():
    """Emitted calls are normalized so they can be scored offline by diffing against
    references, without a re-run."""
    from gbench.runners.eval_suites.base import normalize_function_calls as N
    got = N([{"id": "x", "type": "function",
              "function": {"name": "default_api:retrieve",
                           "arguments": '{"query": "acme", "limit": 5}'}}])
    assert got == [{"name": "default_api:retrieve",
                    "arguments": {"query": "acme", "limit": 5}}]


def test_malformed_arguments_are_kept_not_dropped():
    """A malformed argument list is itself a result and must survive to be scored."""
    from gbench.runners.eval_suites.base import normalize_function_calls as N
    got = N([{"function": {"name": "f", "arguments": "{not json"}}])
    assert got[0]["arguments"] is None and got[0]["arguments_raw"] == "{not json"


def test_zero_argument_calls_are_preserved():
    """Many device-action declarations are genuine no-argument calls."""
    from gbench.runners.eval_suites.base import normalize_function_calls as N
    assert N([{"function": {"name": "take_screenshot", "arguments": "{}"}}]) == [
        {"name": "take_screenshot", "arguments": {}}]


def test_normalizer_tolerates_junk():
    from gbench.runners.eval_suites.base import normalize_function_calls as N
    assert N(None) == [] and N([]) == [] and N(["x", {"no_name": 1}]) == []


def test_traces_carry_the_emitted_calls():
    import inspect
    from gbench.runners.eval_suites import base
    assert '"emitted_function_calls": normalize_function_calls(tool_calls)' in \
        inspect.getsource(base._run_suite_async)


# --- reference-free spec validation --------------------------------------------------
def test_spec_validator_catches_every_error_class():
    """Each class of spec violation is reported under its own error code."""
    from gbench.runners.eval_suites.fc_spec_validator import validate_emitted
    T = [{"function": {"name": "t", "parameters": {
        "type": "object",
        "properties": {"a": {"type": "string"}, "n": {"type": "integer"}},
        "required": ["a"]}}}]
    def codes(calls, tools=T):
        return sorted({e["code"] for e in validate_emitted(calls, tools)[1]})
    assert codes([{"name": "t", "arguments": {"a": "x"}}]) == []
    assert codes([{"name": "nope", "arguments": {}}]) == ["error_invalid_tool_name"]
    assert codes([{"name": "t", "arguments": {}}]) == ["error_missing_required_property"]
    assert codes([{"name": "t", "arguments": {"a": "x", "z": 1}}]) == ["error_unknown_argument_name"]
    assert codes([{"name": "t", "arguments": {"a": "x", "n": "five"}}]) == ["error_unknown_argument_type"]
    assert codes([{"name": "t", "arguments": None, "arguments_raw": "{oops"}]) == ["error_unparseable_fc"]


def test_a_bool_is_not_an_integer():
    """`isinstance(True, int)` is True in Python; an integer argument must reject it."""
    from gbench.runners.eval_suites.fc_spec_validator import validate_emitted
    T = [{"function": {"name": "t", "parameters": {
        "type": "object", "properties": {"n": {"type": "integer"}}}}}]
    assert not validate_emitted([{"name": "t", "arguments": {"n": True}}], T)[0]


def test_no_call_is_spec_valid():
    """Whether a call was WARRANTED is for whatever scores the turn, not for the validator."""
    from gbench.runners.eval_suites.fc_spec_validator import validate_emitted
    assert validate_emitted([], [{"function": {"name": "t", "parameters": {}}}])[0]


def test_unknown_declarations_do_not_manufacture_a_violation():
    """With no declarations we cannot tell a hallucinated tool from an undeclared real one."""
    from gbench.runners.eval_suites.fc_spec_validator import validate_emitted
    assert validate_emitted([{"name": "anything", "arguments": {}}], None)[0]


def test_validation_is_per_row_not_per_tool_name():
    """The same tool name can be declared with different parameters on different rows.
    Caching a schema by tool name across rows would produce phantom violations."""
    from gbench.runners.eval_suites.fc_spec_validator import validate_emitted
    A = [{"function": {"name": "g", "parameters": {
        "type": "object", "properties": {"params_json": {"type": "string"}}}}}]
    B = [{"function": {"name": "g", "parameters": {
        "type": "object", "properties": {"start": {"type": "string"}},
        "required": ["start"]}}}]
    call = [{"name": "g", "arguments": {"params_json": "{}"}}]
    assert validate_emitted(call, A)[0]        # valid against row A's schema
    assert not validate_emitted(call, B)[0]    # invalid against row B's


# (The host java-toolchain probes were removed 2026-09-08: aider_polyglot now runs inside aider's
# own benchmark container, which bundles openjdk-21 + the other 5 toolchains, so gbench no longer
# probes host toolchains. See test_aider_polyglot_canonical.py.)


def test_aider_doc_covers_the_containerized_setup():
    """The JDK/toolchain is now baked into aider's benchmark image (openjdk-21), so the doc
    teaches the container build + endpoint wiring rather than a host JDK fix."""
    import os
    doc = open(os.path.join(DOCS, "aider_polyglot.md"), encoding="utf-8").read()
    assert "docker build" in doc and "aider-benchmark:5dc9490b" in doc
    # the container is on docker's bridge network, as upstream's benchmark/docker.sh runs it
    assert "openjdk-21" in doc and "--add-host host.docker.internal:host-gateway" in doc
    assert "--network host" not in doc
