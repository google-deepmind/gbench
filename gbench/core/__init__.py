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

"""Core package for domain models and configuration."""

from .config import (
    BenchmarkConfig,
    DEFAULT_CONFIG,
    DEFAULT_GEMMACLAW_COMMIT,
    PerformanceTargets,
    QUICK_CONFIG,
    check_gpu_ready,
    get_available_gpus,
    get_batch_sizes,
    get_gpu_memory_utilization,
    get_max_model_len,
    get_num_gpus,
    get_server_timeout,
    get_tensor_parallel,
    validate_gpu_config,
    weights_fit_advisory,
    resolve_campaign_slo,
    campaign_ctx,
)
from .models import (
    ModelCategory,
    ModelConfig,
    ModelFormat,
    ModelRegistry,
    enrich_metadata_from,
    registry,
)

__all__ = [
    "BenchmarkConfig",
    "DEFAULT_CONFIG",
    "DEFAULT_GEMMACLAW_COMMIT",
    "PerformanceTargets",
    "QUICK_CONFIG",
    "check_gpu_ready",
    "get_available_gpus",
    "get_batch_sizes",
    "get_gpu_memory_utilization",
    "get_max_model_len",
    "get_num_gpus",
    "get_server_timeout",
    "get_tensor_parallel",
    "validate_gpu_config",
    "weights_fit_advisory",
    "resolve_campaign_slo",
    "campaign_ctx",
    "ModelCategory",
    "ModelConfig",
    "ModelFormat",
    "ModelRegistry",
    "enrich_metadata_from",
    "registry",
]
