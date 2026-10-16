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

"""codeforces grades the LAST fenced code block (canonical open-r1), and the judge model resolves
loudly, records its source and flags a model judging itself.

The imports are plain on purpose: a module that fails to import must fail these tests, never
turn them into skips."""

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites.codeforces import _eval_codeforces


def test_codeforces_grades_last_code_block_not_first():
    """Models draft then emit a corrected final block; canonical extraction takes the LAST."""
    gold = {"official_tests": [{"input": "", "output": "RIGHT"}],
            "generated_tests": [], "checker": "", "time_limit": 5}

    # first block WRONG, last block RIGHT -> must pass (grades the last)
    two_blocks = ("Attempt 1:\n```python\nprint('WRONG')\n```\n"
                  "Corrected:\n```python\nprint('RIGHT')\n```")
    assert _eval_codeforces(two_blocks, gold) is True

    # only the wrong block -> must fail (sanity: not silently passing)
    assert _eval_codeforces("```python\nprint('WRONG')\n```", gold) is False

    # single correct block still works
    assert _eval_codeforces("```python\nprint('RIGHT')\n```", gold) is True


def test_resolve_judge_model_default_env_and_self_judge(monkeypatch):
    """Judge model must resolve loudly, track its source, and flag the model judging itself."""
    UT = "google/gemma-4-26B-A4B-it"  # the model under test (served locally)

    # default when unset: the head of the one Gemini cascade (base.GEMINI_CASCADE)
    monkeypatch.delenv("GBENCH_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("GBENCH_JUDGE_MODELS", raising=False)
    base._JUDGE_RESOLVE_LOGGED.clear()
    m, src, selfj = base.resolve_judge_model(UT)
    assert (m, src, selfj) == (base.GEMINI_CASCADE[0], "default", False)
    assert m == "gemini-3.8-flash"

    # env override is honored AND its source is recorded (not silently swallowed)
    monkeypatch.setenv("GBENCH_JUDGE_MODEL", "gemini-3.5-flash")
    base._JUDGE_RESOLVE_LOGGED.clear()
    m, src, selfj = base.resolve_judge_model(UT)
    assert (m, src, selfj) == ("gemini-3.5-flash", "env:GBENCH_JUDGE_MODEL", False)

    # judge == model under test -> flagged as self-judge
    monkeypatch.setenv("GBENCH_JUDGE_MODEL", "gemma-4-26b")
    base._JUDGE_RESOLVE_LOGGED.clear()
    m, src, selfj = base.resolve_judge_model(UT)
    assert selfj is True

    # family matcher: catches the gemini-hosted gemma vs the local hf id, and provider prefixes
    assert base._same_model_family("gemma-4-26b", UT) is True
    assert base._same_model_family("gemini/gemma-4-26b", UT) is True
    assert base._same_model_family("gemini-3.6-flash", UT) is False
    assert base._same_model_family("", UT) is False
