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
"""Regression: code-exec suites must feed a large program/harness on STDIN (`python -`), not as a
`-c <script>` argument. LCB functional inputs (>100KB, embedded via {inp!r}/{args!r})
and the codeforces checker driver (embeds the input twice) exceeded Linux MAX_ARG_STRLEN (128KB),
raising OSError('Argument list too long') that `except Exception: return False` silently scored as a
wrong answer -> correct solutions marked failed (lcb 33%->67%, codeforces 67%->100%)."""

import inspect
import sys

import pytest

from gbench.runners.eval_suites import codeforces, lcb
from gbench.runners.eval_suites.sandbox import run_sandboxed, sandbox_available


def test_large_program_runs_via_stdin():
    if not sandbox_available():
        pytest.skip("bubblewrap unavailable")
    big = "print('OK')\n# padding: " + ("A" * 200000)   # ~200KB, well over MAX_ARG_STRLEN
    r = run_sandboxed([sys.executable, "-"], input=big, capture_output=True, text=True, timeout=15)
    assert r.returncode == 0 and (r.stdout or "").strip() == "OK"


def test_large_program_via_dash_c_would_have_failed():
    # Document the failure mode the fix avoids: the same script as a `-c` argument is rejected.
    if not sandbox_available():
        pytest.skip("bubblewrap unavailable")
    big = "print('OK')\n# padding: " + ("A" * 200000)
    with pytest.raises(OSError):
        run_sandboxed([sys.executable, "-c", big], capture_output=True, text=True, timeout=15)


def test_lcb_does_not_pass_test_script_via_dash_c():
    src = inspect.getsource(lcb)
    assert '"-c", test_script' not in src, "lcb must feed test_script on stdin, not -c (E2BIG)"


def test_codeforces_checker_does_not_pass_script_via_dash_c():
    src = inspect.getsource(codeforces)
    assert '"-c", script' not in src, "codeforces checker must feed the driver on stdin, not -c (E2BIG)"
