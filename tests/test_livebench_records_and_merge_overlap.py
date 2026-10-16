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

"""livebench records what it ran and refuses overlapping shards.

* The merge refuses a question found in two shards, as the generic merge does.
* The run's output directory is kept under GBENCH_CACHE_DIR and handed to the host user.
* A model-id override is checked against the endpoint's model list.

No network, no docker.
"""

import os

import pytest

from gbench import merge
from gbench.runners.eval_suites import livebench as LB


def _shard(label, keys):
    return {"eval_name": "livebench", "shard": label, "status": "success", "accuracy": 50.0,
            "sample_traces": [{"sample_key": f"question:math/{k}", "question_id": k,
                               "category": "math", "task": "math_comp", "score": 0.5}
                              for k in keys]}


def test_the_merge_refuses_a_question_in_two_shards():
    with pytest.raises(merge.MergeError, match="the shards overlap"):
        LB.merge_shards([_shard("1/2", ["a", "b"]), _shard("2/2", ["b", "c"])])
    merged = LB.merge_shards([_shard("1/2", ["a"]), _shard("2/2", ["b", "c"])])
    assert merged["questions_judged"] == 3 and merged["total_questions"] == 3
    assert merged["correct_answers"] is None and "not applicable" in \
        merged["correct_answers_note"]


def test_the_output_directory_is_kept_under_the_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))
    out = LB._output_dir()
    assert os.path.isdir(out)
    assert os.path.dirname(out) == str(tmp_path / "livebench" / "runs")


def test_the_container_hands_its_output_to_the_host_user():
    cmd = LB._build_command("img", "/o", "m", "m", "http://x/v1", 2, None, "", "",
                            owner="1000:1000")
    assert cmd[-1].endswith("; chown -R 1000:1000 /out 2>/dev/null || true")
    assert "chown" not in LB._build_command("img", "/o", "m", "m", "http://x/v1", 2, None, "",
                                            "")[-1]


def test_a_model_id_override_is_checked(monkeypatch):
    from gbench.utils import endpoint as EP
    monkeypatch.delenv("GBENCH_LIVEBENCH_MODEL", raising=False)
    assert LB._request_model_id("http://h/v1", "m") == ("m", "the served model id gbench passed")
    monkeypatch.setenv("GBENCH_LIVEBENCH_MODEL", "other")

    def refuse(url, candidates, override=None, **kw):
        raise EP.EndpointError("serves only m")
    monkeypatch.setattr(EP, "resolve_served_model", refuse)
    with pytest.raises(RuntimeError, match="GBENCH_LIVEBENCH_MODEL='other' is not a model"):
        LB._request_model_id("http://h/v1", "m")
    monkeypatch.setattr(EP, "resolve_served_model",
                        lambda url, c, o=None, **kw: EP.ServedModel("other", "listed", None))
    assert LB._request_model_id("http://h/v1", "m") == ("other",
                                                        "GBENCH_LIVEBENCH_MODEL (listed)")
