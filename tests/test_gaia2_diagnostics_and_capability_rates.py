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

"""gaia2 says what to fix, reports no 0% for an unscored capability, and checks its limit.

* A run with no validated score names what to check per harness-error kind, the capability
  configs that raised and where the artifacts were kept.
* A capability without a validated run has no success rate (never ARE's 0.0).
* GBENCH_GAIA2_LIMIT must be a positive integer, checked before any container starts.
* The merge keeps each shard's scenario draw apart.

No docker, no model server.
"""

import pytest

from gbench.runners.eval_suites import gaia2 as G


def test_the_remediation_follows_the_harness_error_kinds():
    text = G.harness_remediation({"judge": 2, "server_parser_text_lost": 1})
    assert "GEMINI_API_KEY" in text and "/detokenize" in text
    assert "model endpoint stays up" not in text
    assert "model endpoint" in G.harness_remediation({"model_endpoint": 3})
    assert "GEMINI_API_KEY" in G.harness_remediation({})


def test_the_diagnostics_name_the_failed_configs_and_the_artifacts():
    tail = G.run_diagnostics({"failed_configs": [{"capability": "time", "error": "boom"}]},
                             "/r/gaia2_artifacts")
    assert "failed configs: time: boom" in tail and "artifacts kept in /r/gaia2_artifacts" in tail
    assert G.run_diagnostics({}, None) == ""


def test_an_unscored_capability_has_no_success_rate():
    per = {"search": {"validated_runs": 3, "success_runs": 1, "success_rate": 33.3},
           "time": {"validated_runs": 0, "success_runs": 0, "success_rate": 0.0}}
    out = G.unscored_capabilities_as_none(per)
    assert out["search"]["success_rate"] == 33.3
    assert out["time"]["success_rate"] is None
    assert out["time"]["success_rate_are_formula"] == 0.0
    assert per["time"]["success_rate"] == 0.0                   # the input is not changed
    assert G.unscored_capabilities_as_none(None) is None


@pytest.mark.parametrize("value", ["0", "-1", "abc", "1.5"])
def test_a_limit_that_is_not_a_positive_integer_is_refused(monkeypatch, value):
    monkeypatch.setenv("GBENCH_GAIA2_LIMIT", value)
    with pytest.raises(ValueError, match="GBENCH_GAIA2_LIMIT"):
        G.explicit_limit_env()


def test_a_positive_limit_or_none_passes(monkeypatch):
    monkeypatch.setenv("GBENCH_GAIA2_LIMIT", "4")
    assert G.explicit_limit_env() == "4"
    monkeypatch.delenv("GBENCH_GAIA2_LIMIT")
    assert G.explicit_limit_env() == ""


def test_the_merge_keeps_each_shards_draw():
    def shard(label, sid, status, limit):
        return {"shard": label, "status": "success", "selection": {"limit": limit},
                "sample_traces": [{"config": "search", "phase_name": "standard",
                                   "scenario_id": sid, "run_number": 0, "status": status,
                                   "capability": "search"}]}
    merged = G.merge_shards([shard("1/2", "s1", "success", 1), shard("2/2", "s2", "failed", 2)])
    assert "selection" not in merged
    assert merged["selection_by_shard"] == {"1/2": {"limit": 1}, "2/2": {"limit": 2}}
