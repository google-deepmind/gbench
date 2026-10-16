#!/usr/bin/env python3
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

"""Example usage of the gbench package.

Demonstrates common programmatic usage patterns for the benchmark suite.
All examples use dry_run=True - set to False to run actual benchmarks.

Models are resolved on demand from their full HuggingFace id (the registry has
no built-in table). Always pass an 'org/model' id that contains '/' - e.g.
'google/gemma-4-26B-A4B-it' - so registry.get() auto-registers it and derives
metadata from the model's config.json. Bare short-names ('gemma-4-E4B-it') no
longer resolve and return None.

To actually serve a model, either point at a vLLM OpenAI-compatible endpoint
(the runnable path today), e.g.
    python -m gbench --models google/gemma-4-26B-A4B-it \\
        --remote-endpoint http://127.0.0.1:8000/v1 \\
        --tokenizer google/gemma-4-26B-A4B-it
(served model id: google/gemma-4-26B-A4B-it), or let gbench launch a local vLLM
server (a run without --remote-endpoint; pass --num-gpus and the model's
--tool-call-parser / --reasoning-parser for tool-calling or --thinking evals).
"""

from gbench.runners import (
    ServingBenchmarkRunner,
    ThroughputBenchmarkRunner,
    StressTestRunner,
)
from gbench.core import (
    BenchmarkConfig,
    QUICK_CONFIG,
    DEFAULT_CONFIG,
    ModelCategory,
    ModelFormat,
    registry,
    get_batch_sizes,
    get_num_gpus,
    get_server_timeout,
    get_tensor_parallel,
)

def main():
    #############################################
    # Example 1: Quick smoke test on single model
    #############################################

    print("=" * 60)
    print("Example 1: Quick Smoke Test")
    print("=" * 60)

    model = registry.get("google/gemma-4-E4B-it")
    if model:
        config = BenchmarkConfig(
            batch_sizes=[1, 50],
            input_lengths=[128],
            output_lengths=[512],
            num_prompts=200,
            results_dir="./example_results",
            dry_run=True,
        )
        config.initialize()

        runner = ServingBenchmarkRunner(config)
        result = runner.run(
            model=model,
            format=ModelFormat.HF,
            batch_size=1,
        )
        print(f"Result: {result}")

    #############################################
    # Example 2: Inspect auto-derived model info
    #############################################

    print("\n" + "=" * 60)
    print("Example 2: Auto-Derived Model Info")
    print("=" * 60)

    for hf_id in ["google/gemma-4-E4B-it", "google/gemma-4-E2B-it",
                  "google/gemma-4-26B-A4B-it"]:
        m = registry.get(hf_id)
        if m:
            moe_str = f"MoE {m.num_experts}x{m.num_active_experts}" if m.is_moe else "dense"
            print(f"  {m.short_name:<25} {m.total_params_b:>6.1f}B  {moe_str:<12} "
                  f"mm={m.supports_multimodal}  gpus={get_num_gpus(m.total_params_b)}  "
                  f"timeout={get_server_timeout(m.total_params_b, m.is_moe)}s")

    #############################################
    # Example 3: Serving + Throughput for a model
    #############################################

    print("\n" + "=" * 60)
    print("Example 3: Full Benchmark (Serving + Throughput)")
    print("=" * 60)

    model = registry.get("google/gemma-4-31B-it")
    if model:
        tp = get_tensor_parallel(model.total_params_b)
        config = BenchmarkConfig(
            batch_sizes=[1, 4, 8, 16],
            input_lengths=[128, 512],
            output_lengths=[512, 1024],
            num_prompts=500,
            num_gpus=tp,
            tensor_parallel_size=tp,
            results_dir="./gemma4_31b_benchmarks",
            dry_run=True,
        )
        config.initialize()

        serving_runner = ServingBenchmarkRunner(config)
        print("\nServing benchmarks (text):")
        serving_runner.run_all(model, ModelFormat.HF)

        print("\nServing benchmarks (multimodal):")
        serving_runner.run_all(model, ModelFormat.HF, multimodal=True)

        throughput_runner = ThroughputBenchmarkRunner(config)
        print("\nThroughput benchmarks:")
        throughput_runner.run_all(model, ModelFormat.HF)

    #############################################
    # Example 4: Stress test with custom threshold
    #############################################

    print("\n" + "=" * 60)
    print("Example 4: Stress Test")
    print("=" * 60)

    model = registry.get("google/gemma-4-E4B-it")
    if model:
        config = BenchmarkConfig(
            num_prompts=500,
            results_dir="./stress_results",
            dry_run=True,
        )
        config.initialize()

        stress_runner = StressTestRunner(config, ttft_threshold_ms=500)
        print(f"\nStress test for {model.short_name} (threshold: 500ms P99 TTFT):")
        stress_runner.run_all(model, ModelFormat.HF)

    #############################################
    # Example 5: Filter and explore the registry
    #############################################

    print("\n" + "=" * 60)
    print("Example 5: Registry Exploration")
    print("=" * 60)

    # The registry has no built-in table; list_all() returns only the models
    # registered on demand so far - i.e. the gemma-4 ids resolved in Examples
    # 1-4 above. Register more via registry.get("org/model") first.
    all_models = registry.list_all()
    print(f"\nTotal models registered so far: {len(all_models)}")

    text_models = registry.get_by_category(ModelCategory.TEXT)
    mm_models = registry.get_by_category(ModelCategory.MULTIMODAL)
    print(f"  Text models: {len(text_models)}")
    print(f"  Multimodal models: {len(mm_models)}")

    print("\nMoE models:")
    for m in all_models:
        if m.is_moe:
            print(f"  - {m.short_name}: {m.total_params_b:.0f}B total, "
                  f"{m.num_experts}x{m.num_active_experts}")

    #############################################
    # Example 6: Default concurrency + GPU sizing
    #############################################

    print("\n" + "=" * 60)
    print("Example 6: Default Concurrency + GPU Sizing")
    print("=" * 60)

    # Default concurrency is a single stream (batch size 1) for every model,
    # regardless of parameter count or preset - get_batch_sizes() always
    # returns [1]. Sweeping concurrency is opt-in via --batch-sizes; the load
    # dimension is otherwise covered by the stress test's arrival-rate (QPS)
    # sweep. GPU count, however, IS param-derived (get_num_gpus tiers).
    for m in sorted(all_models, key=lambda x: x.total_params_b):
        gpus = get_num_gpus(m.total_params_b)
        default_concurrency = get_batch_sizes(m.total_params_b)
        print(f"  {m.short_name:<25} {m.total_params_b:>6.1f}B  "
              f"gpus={gpus}  default_concurrency={default_concurrency}")

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nTo run actual benchmarks:")
    print("  1. Set dry_run=False in config")
    print("  2. Ensure vLLM is installed and GPU available")
    print("  3. Call config.initialize() after setting all fields")
    print("\nOr use the CLI. The default run does a serving sweep AND (by")
    print("default) an arrival-rate stress QPS ramp - it is not serving-only:")
    print("  # Serve with vLLM, then point gbench at the endpoint:")
    print("  python -m gbench --models google/gemma-4-26B-A4B-it \\")
    print("      --remote-endpoint http://127.0.0.1:8000/v1 \\")
    print("      --tokenizer google/gemma-4-26B-A4B-it")
    print("  # Or let gbench launch a local vLLM server (no --remote-endpoint):")
    print("  python -m gbench --models google/gemma-4-26B-A4B-it --num-gpus 1")
    print("  gbench --help")


if __name__ == "__main__":
    main()
