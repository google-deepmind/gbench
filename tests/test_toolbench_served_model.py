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
"""Regression: toolbench must send the endpoint's ACTUAL served model id to the container.

StableToolBench's OpenAI client sends the `model` field verbatim to vLLM. The id gbench's eval
runner passes is the served id it resolved against --models; `_served_model_id` returns it
unchanged and never adopts the first entry of the endpoint's /v1/models (on an endpoint serving
several models that entry can be another model)."""

import io
import json
from unittest import mock

from gbench.runners.eval_suites import toolbench


def _fake_models_response(model_id):
    body = json.dumps({"data": [{"id": model_id}]}).encode("utf-8")
    cm = mock.MagicMock()
    cm.__enter__.return_value = io.BytesIO(body)
    cm.__exit__.return_value = False
    return cm


def test_served_model_id_is_the_id_gbench_passes_not_the_first_listed():
    with mock.patch("urllib.request.urlopen",
                    return_value=_fake_models_response("other-org/another-model")) as op:
        got = toolbench._served_model_id("http://127.0.0.1:8000/v1", "org/served-model")
    assert got == "org/served-model"
    op.assert_not_called()


def test_served_model_id_falls_back_when_endpoint_unreadable():
    with mock.patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
        got = toolbench._served_model_id("http://127.0.0.1:8000/v1", "fallback-model")
    assert got == "fallback-model"
