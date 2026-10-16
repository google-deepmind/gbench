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

"""Unit tests for the parameter-derived sizing helpers of gbench/core/config.py (GPU count,
tensor parallel size, default concurrency, server start-up timeout, served context length and
GPU memory utilization)."""

from gbench.core.config import (
    get_batch_sizes,
    get_gpu_memory_utilization,
    get_max_model_len,
    get_num_gpus,
    get_server_timeout,
    get_tensor_parallel,
)


def test_gpu_allocation_fairness():
    """Models of similar size get the same GPU count."""
    # All <=20B -> 1 GPU
    assert get_num_gpus(1.0) == 1
    assert get_num_gpus(14.0) == 1
    assert get_num_gpus(20.0) == 1

    # All 20-80B -> 2 GPUs
    assert get_num_gpus(25.0) == 2
    assert get_num_gpus(36.0) == 2
    assert get_num_gpus(80.0) == 2

    # All >80B -> 8 GPUs
    assert get_num_gpus(100.0) == 8
    assert get_num_gpus(235.0) == 8


def test_tensor_parallel_matches_gpus():
    """TP size always equals GPU count (get_tensor_parallel is public API: the shipped
    examples/usage_examples.py uses it)."""
    for params in [1.0, 14.0, 30.0, 120.0, 235.0]:
        assert get_tensor_parallel(params) == get_num_gpus(params)


def test_batch_sizes_default_single_stream():
    """Default serving is single-stream ([1]) for every model size and preset.

    Concurrency sweeping is opt-in via --batch-sizes; sustained load is measured by
    the stress pillar (open-loop QPS), so the default serving pillar isolates
    single-stream latency. See BenchmarkConfig.get_batch_sizes."""
    assert get_batch_sizes(4.0, "quick") == [1]
    assert get_batch_sizes(4.0, "default") == [1]
    assert get_batch_sizes(30.0, "default") == [1]
    assert get_batch_sizes(120.0, "default") == [1]


def test_timeout_increases_with_params():
    """Larger models get longer timeouts."""
    t_small = get_server_timeout(4.0, False)
    t_large = get_server_timeout(30.0, False)
    t_xlarge = get_server_timeout(120.0, False)

    assert t_small < t_large < t_xlarge


def test_timeout_moe_multiplier():
    """MoE models get longer timeouts than dense at same size."""
    t_dense = get_server_timeout(30.0, False)
    t_moe = get_server_timeout(30.0, True)

    assert t_moe > t_dense


def test_timeout_bounds():
    """Timeouts are within floor/cap bounds."""
    assert get_server_timeout(1.0, False) >= 600
    assert get_server_timeout(500.0, True) <= 3600


def test_max_model_len_uniform():
    """max_model_len is always 4096."""
    assert get_max_model_len() == 4096


def test_gpu_memory_utilization():
    """GPU memory utilization is uniform 0.90."""
    assert get_gpu_memory_utilization(4.0) == 0.90
    assert get_gpu_memory_utilization(120.0) == 0.90
