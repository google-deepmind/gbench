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

"""`bfcl evaluate` summarizes one request latency instead of failing the suite.

Upstream's evaluate takes statistics.stdev of the model's latencies for its leaderboard CSV
(eval_runner_helper.get_cost_latency_info), which raises StatisticsError with one latency. The
2026-10-10 smoke test (--eval-limit 5: one agentic entry) failed the whole bfcl suite there. The
harness child now makes that standard deviation NaN; gbench reads no latency column.

No docker: upstream's module is faked. The same guard was run against the pinned bfcl_eval in
gbench-bfcl:2026.3.23 (one latency -> (0.0, 4.2, nan, 4.2); two -> stdev 1.41).
"""

import math
import statistics
import sys
import types

import pytest

from gbench.runners.eval_suites import bfcl_v4_agentic as AG


def test_one_latency_has_an_undefined_standard_deviation():
    stats = AG._OnePointStatistics()
    assert math.isnan(stats.stdev([4.2]))
    assert math.isnan(stats.stdev(iter([])))
    assert stats.stdev([1.0, 2.0, 3.0]) == statistics.stdev([1.0, 2.0, 3.0])
    assert stats.mean([1.0, 3.0]) == 2.0                  # every other function is the module's
    with pytest.raises(statistics.StatisticsError):
        stats.variance([1.0])


def test_the_guard_replaces_only_the_helpers_reference(monkeypatch):
    helper = types.ModuleType("bfcl_eval.eval_checker.eval_runner_helper")
    helper.statistics = statistics
    for name in ("bfcl_eval", "bfcl_eval.eval_checker"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    monkeypatch.setattr(sys.modules["bfcl_eval.eval_checker"], "eval_runner_helper", helper,
                        raising=False)
    AG._allow_one_latency()
    assert math.isnan(helper.statistics.stdev([4.2]))
    with pytest.raises(statistics.StatisticsError):
        statistics.stdev([4.2])                           # the real module is unchanged


def test_the_child_installs_the_guard_before_the_cli_dispatches():
    code = AG._CHILD_BOOTSTRAP.format(model="m", args=["evaluate"])
    assert code.index("from bfcl_eval.__main__ import cli") < code.index("_allow_one_latency()") \
        < code.index("\ncli()")
