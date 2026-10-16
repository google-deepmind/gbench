# Performance benchmarks: models, GPUs, workloads and statistics

How gBench reads a model, sizes and selects GPUs, which workload campaigns and datasets the performance pillars run, and how repeatability is checked.

[Back to the README](../README.md)

## Model discovery and GPU allocation

gBench operates without static model lookup tables. Passing any valid HuggingFace model identifier via `--models` triggers automatic enrichment from `config.json`:

### Model properties read from config.json
- Total parameter count (`total_params_b`)
- Mixture of Experts (MoE) topology (`num_local_experts`, `num_experts_per_tok`)
- Multimodal support (`vision_config` or multimodal architecture strings)
- Maximum context window length (`max_position_embeddings`)

### Recommended GPU tiers

gBench recommends a GPU count per model size, so that models of the same size are compared on the same resources. The recommendation is not applied for you: a local run requires `--num-gpus`, and `validate_gpu_config` checks that value against the recommendation: a warning when it is below the tier and the model still fits (or more than twice the tier), a refusal when the model will not fit in the requested GPUs, when the count is not 1, 2, 4, 8 or 16, or when fewer GPUs are available.

#### VRAM estimate

Required VRAM (GB) = (Total Parameters (B) * Bytes per Parameter) + KV Cache Space + CUDA Overhead

| Precision Format | Bytes / Parameter | 4B Model Weights | 12B Model Weights | 31B Model Weights | 80B Model Weights |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BF16 / FP16** | 2.0 B/param | 8.0 GB | 24.0 GB | 62.0 GB | 160.0 GB |
| **FP8 / INT8** | 1.0 B/param | 4.0 GB | 12.0 GB | 31.0 GB | 80.0 GB |
| **INT4** | 0.5 B/param | 2.0 GB | 6.0 GB | 15.5 GB | 40.0 GB |

#### Recommended datacenter tier (`get_num_gpus()`)
The recommended tiers assume 80GB datacenter GPUs (e.g. NVIDIA A100 80GB, H100 80GB) running unquantized BF16/FP16 models:

| Total model parameters | Recommended GPUs | Tensor parallelism (TP) | Min Target VRAM | Default Batch Size Sweep (`default` / `quick`) |
| :--- | :--- | :--- | :--- | :--- |
| `<= 15B` | 1 GPU | TP=1 | 24 GB - 48 GB | `[1]` (opt-in sweep: `--batch-sizes`) |
| `15.1B - 20B` | 1 GPU | TP=1 | 48 GB - 80 GB | `[1]` (opt-in sweep: `--batch-sizes`) |
| `20.1B - 80B` | 2 GPUs | TP=2 | 160 GB (2x 80GB) | `[1]` (opt-in sweep: `--batch-sizes`) |
| `> 80B` | 8 GPUs | TP=8 | 640 GB (8x 80GB) | `[1]` (opt-in sweep: `--batch-sizes`) |

#### Workstations and smaller GPUs (8GB to 24GB)
When executing on workstations with smaller VRAM GPUs (e.g., NVIDIA L4 24GB, RTX 4090 24GB, RTX 3090 24GB, RTX 4070 8GB):
- Unquantized BF16 models of 12B or more need several GPUs. Pass `--num-gpus <N>` explicitly (e.g., `--num-gpus 2` for a 12B model on 24GB GPUs); `--tensor-parallel` defaults to it.
- Model-specific serve settings: gBench names no model. A model whose serving recipe needs a setting passes it: `--tool-call-parser` and `--reasoning-parser` for the eval, quality and Golden Set server, `--vllm-serve-args '...'` for anything else (every server gBench launches).

### GPU selection

A local run launches its vLLM server on `--num-gpus` GPUs (`--tensor-parallel` of them per
replica, `--num-gpus / --tensor-parallel` data-parallel replicas). `--gpus 0,3` (indices or UUIDs)
names them; each must run no other process. Without it, gBench picks the first idle GPUs (no
compute process and at least 80% of memory free) among those `CUDA_VISIBLE_DEVICES` exposes, and
too few idle GPUs is an error that names the busy ones. The server runs with
`CUDA_VISIBLE_DEVICES` set to the chosen UUIDs (and `CUDA_DEVICE_ORDER=PCI_BUS_ID`), the
pre-flight process check looks only at them, and metadata.json records them (`selected_gpus`,
`parallel`). `--dry-run` shows the selection; a busy GPU is advisory there.

### Server startup timeout
Server startup timeouts scale dynamically with model size and MoE complexity to account for parallel weight loading and CUDA graph compilation:

TimeoutSeconds = max(600, min(3600, (300 + Parameters (B) * 10) * 3 * 1.3 (if MoE)))

### Context length
- Performance benchmarks: a uniform `max_model_len=4096`, or with `--campaign` the window sized to that campaign's prompt and output lengths; `--max-model-len` overrides both.
- Performance request content: every serving and stress request is new to the server's caches (a nonce from a fresh per-run salt leads each prompt, and every image carries a per-request code), and each result records the cache counters it saw; prefix caching is also off on the servers gBench launches for performance, as defence in depth. `GBENCH_PERF_SEED` reproduces lengths and the schedule; the salt is recorded but never reused.
- Evals and quality: the server gBench launches serves the model's native context window (for Gemma 4, 128K on E2B/E4B and 256K on the larger models), so long-context suites and large `--max-output-tokens` budgets fit. Pass `--max-model-len <tokens>` to cap it (it must exceed `--max-output-tokens`), or `-1` to let vLLM fit the window to GPU memory. A `--campaign` sizes only the performance server. Prefix caching stays on for these servers; it is off only on the servers gBench launches for performance runs, as defence in depth (performance requests are already unique). They are also started with `--max-logprobs 100` (vLLM's default is 20), because `mmlu` sends hendrycks/test evaluate.py's request with `logprobs=100`; the cap only validates requests and changes no suite's request or score. A server you start yourself for `mmlu` (`--remote-endpoint`) needs the same flag.

## Performance campaigns and workload shapes

gBench includes 6 predefined performance campaigns (`--campaign`) that configure dataset backends, input token lengths, output token lengths, and prompt geometry to simulate production workloads:

| Campaign ID | Dataset | Input length (tokens) | Output length (tokens) | Workload it models |
| :--- | :--- | :--- | :--- | :--- |
| `chat-like` | `sharegpt` | ShareGPT V3 prompts | ShareGPT V3 replies | Real chat prompts: the first turn of each ShareGPT conversation (single-turn requests; the second turn's length is the expected output) |
| `agentic` | `random` | 8000 | 400 | Long prompt, short reply (the token shape of an agent step; no tools are sent) |
| `prefill-heavy` | `random` | 8192 | 128 | Document summarization or RAG ingestion |
| `decode-heavy` | `random` | 128 | 2048 | Code generation or long prose |
| `mixed` | `random` | 4096 | 1024 | Chat with analytical reasoning |
| `long-decode` | `random` | 8192 | 8192 | Long prompt and long generation |

A campaign sets no prompt count. The serving pillar sizes its sample count per configuration
(30 to 500 requests, aiming at about 5 minutes per configuration: more with a larger batch size,
fewer with longer outputs), and the throughput pillar uses the preset's fixed count (1000 with
`--preset default`, 10 with `--preset quick`).

```bash
# Run agentic campaign with custom single batch size override
gbench --models google/gemma-4-12B-it --num-gpus 1 --campaign agentic --batch-sizes 1 --num-iterations 1 --warmup-iterations 0
```

## Dataset backends and custom prompts

gBench supports 4 dataset backends (`--dataset`):

| Dataset ID | Flag Syntax | Description |
| :--- | :--- | :--- |
| `random` | `--dataset random` | Default. Generates synthetic token prompts matching exact `--input-lengths` and `--output-lengths`. |
| `sharegpt` | `--campaign chat-like` | Loads the ShareGPT V3 unfiltered chat dataset (`ShareGPT_V3_unfiltered_cleaned_split.json`; each request is a conversation's first turn, the second turn's length the expected output) with chat-like SLOs/context, through vLLM's dataset loader (vllm must be installed, also against `--remote-endpoint`). Not selectable via `--dataset`. |
| `custom` | `--dataset custom --dataset-path <path>` | Loads local custom JSON prompt dataset via `vllm.benchmarks.datasets.CustomDataset`. |
| `hf` | `--dataset hf --dataset-path <name>` | Loads public dataset from HuggingFace Hub via `vllm.benchmarks.datasets.get_samples`. |

```bash
# Run serving performance test using custom prompt dataset
gbench --models google/gemma-4-12B-it --num-gpus 1 --dataset custom --dataset-path ./data/my_prompts.json --input-lengths 512 --output-lengths 512
```

## Statistical methodology and repeatability

gBench checks that performance numbers repeat across iterations:

1. Iterations: `--num-iterations` measured runs per configuration, after `--warmup-iterations` discarded warmup runs.
2. Summary: mean, median, standard deviation, P50, P95 and P99 of each metric.
3. Repeatability: the coefficient of variation (CV) of the iterations,

CV (%) = (Standard Deviation / Mean) * 100

- CV at or below `--max-cv-percent` (default 5.0): valid.
- CV above it and up to twice it: valid, reported as acceptable but high.
- CV above twice `--max-cv-percent`: invalid (poor repeatability).

Serving judges the CV of requests/s. With a real dataset (`--campaign chat-like`, `--dataset custom` or `hf`) each iteration replays different prompts, so requests/s moves with their reply lengths; output tokens/s is judged instead (`repeatability_metric` in the result), and the summary's Input column names the dataset.
