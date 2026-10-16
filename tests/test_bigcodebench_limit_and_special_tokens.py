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

"""bigcodebench draws its --eval-limit tasks once, and says it decodes as the server does.

* The loader draws N tasks (both splits of each); base is not asked to draw again, so Instruct
  and Complete are scored on the same tasks.
* Upstream's skip_special_tokens=False is not sent; the reference and the result say so.

No network, no docker.
"""

import ast
from pathlib import Path

from gbench.runners.eval_suites import bigcodebench as BCB

SRC = Path(BCB.__file__).read_text(encoding="utf-8")


def test_base_is_not_asked_to_draw_the_limit_again():
    calls = [n for n in ast.walk(ast.parse(SRC)) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "run_eval_suite"]
    assert calls
    for call in calls:
        limit = next(kw.value for kw in call.keywords if kw.arg == "limit")
        assert isinstance(limit, ast.Constant) and limit.value is None


def test_the_special_tokens_setting_is_recorded():
    assert "skip_special_tokens=False is not sent" in BCB.SAMPLING[True].reference
    assert "skip_special_tokens=False" in BCB.SPECIAL_TOKENS_NOTE
    assert 'result["special_tokens_note"] = SPECIAL_TOKENS_NOTE' in SRC
