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

"""A pool child imports the installed gbench whatever directory the pool was started from.

2026-10-12: the release check was started from /projects/gemma4-vllm, whose gbench/ directory
(the repository root, no __init__.py) shadowed the package under `python -m gbench`: all 84 suites
of the smoke and the validation run failed in seconds with "cannot import name '__version__' from
'gbench' (unknown location)". The pool now runs its children with python -P.
"""

import subprocess

from gbench import pool


def test_children_run_without_the_working_directory_on_sys_path():
    assert pool.GBENCH_CMD[1] == "-P"


def test_a_shadowing_directory_in_the_cwd_does_not_break_the_child(tmp_path):
    (tmp_path / "gbench").mkdir()                       # like the repository root: no __init__.py
    (tmp_path / "inspect.py").write_text("raise SystemExit('shadowed stdlib')\n")
    proc = subprocess.run([*pool.GBENCH_CMD, "--help"], cwd=tmp_path, capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-500:]
