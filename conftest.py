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

"""Repository-level pytest setup.

* vllm is stubbed when it is not installed, so modules that import it at the top level can be
  collected on a machine without the serving stack.
* The model registry never reaches the HuggingFace Hub in a unit test: gbench.core.models'
  own hf_hub_download and model_info are replaced for the whole session by a stub that hands
  back a small dense text config (and no safetensors metadata, so the parameter estimate is
  exercised). The stub is scoped to the registry module ON PURPOSE: suites that fetch their
  datasets with huggingface_hub keep the real functions, so a test that exercises such a loader
  must provide its own fake instead of silently receiving this config file.
"""

import json
import os
import sys
import tempfile
from unittest import mock

import pytest

# Mock vllm and its submodules before anything else imports them
try:
    import vllm  # noqa: F401
except ImportError:
    vllm_mock = mock.MagicMock()

    class MockTaskType:
        GENERATION = "GENERATION"
    vllm_mock.benchmarks = mock.MagicMock()
    vllm_mock.benchmarks.serve = mock.MagicMock()
    vllm_mock.benchmarks.serve.TaskType = MockTaskType
    vllm_mock.benchmarks.datasets = mock.MagicMock()
    vllm_mock.tokenizers = mock.MagicMock()

    sys.modules["vllm"] = vllm_mock
    sys.modules["vllm.benchmarks"] = vllm_mock.benchmarks
    sys.modules["vllm.benchmarks.serve"] = vllm_mock.benchmarks.serve
    sys.modules["vllm.benchmarks.datasets"] = vllm_mock.benchmarks.datasets
    sys.modules["vllm.tokenizers"] = vllm_mock.tokenizers


# The config the registry stub returns for every model: a dense text model.
_temp_dir = tempfile.TemporaryDirectory()
_dummy_config_path = os.path.join(_temp_dir.name, "config.json")
_dummy_config = {
    "hidden_size": 2048,
    "num_hidden_layers": 10,
    "num_attention_heads": 8,
    "intermediate_size": 8192,
    "vocab_size": 256000,
    "max_position_embeddings": 131072,
    "architectures": ["DenseForCausalLM"],
}
with open(_dummy_config_path, "w") as f:
    json.dump(_dummy_config, f)

import gbench.core.models as _registry_module  # noqa: E402

_mock_download = mock.patch.object(_registry_module, "hf_hub_download",
                                   return_value=_dummy_config_path)
_mock_download.start()
_mock_info = mock.patch.object(_registry_module, "model_info")
_mock_info_func = _mock_info.start()
_mock_info_func.return_value = mock.MagicMock(safetensors=None, sha=None)


@pytest.fixture(scope="session", autouse=True)
def cleanup_temp_dir():
    yield
    _temp_dir.cleanup()
