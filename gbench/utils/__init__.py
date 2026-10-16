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

"""Utility subpackage for gbench."""

from gbench.utils.dependency_checks import (
    check_vllm_available,
    require_vllm_engine,
    safe_get_tokenizer,
    vllm_cli,
    vllm_engine_record,
)
from gbench.utils.endpoint import (
    EndpointError,
    probe_multimodal,
    probe_multimodal_support,
    resolve_served_model,
    verify_endpoint_functional,
)
from gbench.utils.log_manager import LogManager

__all__ = [
    "EndpointError",
    "check_vllm_available",
    "require_vllm_engine",
    "safe_get_tokenizer",
    "vllm_cli",
    "vllm_engine_record",
    "verify_endpoint_functional",
    "probe_multimodal",
    "probe_multimodal_support",
    "resolve_served_model",
    "LogManager",
]
