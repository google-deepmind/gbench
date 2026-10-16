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

"""Every description of `--eval-limit` states the rule: N samples per suite in total, spread over
a suite's settings, attempts per sample unchanged, the plan logged at start and recorded as
eval_limit_plan."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _flat(text):
    return re.sub(r"\s+", " ", text)


def _cli_help():
    from gbench import cli
    return next(action.help for action in cli.create_parser()._actions
                if "--eval-limit" in action.option_strings)


def _assert_rule(text):
    text = _flat(text)
    assert "samples per suite in total" in text
    assert "never N per tab, split, setting, language, domain, scenario, pass or mode" in text
    assert "spreads the N over them" in text
    assert "are unchanged and are not extra samples" in text
    assert "eval_limit_plan" in text
    assert "per benchmark suite" not in text


def test_cli_help_states_the_rule():
    _assert_rule(_cli_help())


def test_readme_flag_table_states_the_rule():
    # The README CLI argument tables moved to docs/cli-reference.md.
    row = next(line for line in (ROOT / "docs/cli-reference.md").read_text().splitlines()
               if line.startswith("| `--eval-limit`"))
    _assert_rule(row)
    assert "Cap the number of evaluation questions evaluated per suite" not in row


def test_env_knobs_states_the_rule_and_no_per_group_limit():
    text = (ROOT / "docs/evals/env-knobs.md").read_text()
    _assert_rule(text)
    assert "TASKS per group" not in text
