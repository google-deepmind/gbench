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

# Evaluation Suite: hmmt
# Description: HMMT (Harvard-MIT Mathematics Tournament Competition Math Benchmark)

"""gbench native built-in runner for hmmt (STEM & Scientific Reasoning).

HMMT February 2026 as MathArena runs it: `MathArena/hmmt_feb_2026` (33 problems, pinned below),
the competition that eth-sri/matharena lists among its current final-answer competitions, with
MathArena's whole protocol (see matharena_common):

* the instruction of configs/competitions/hmmt/hmmt_feb_2026.yaml, a blank line, then the
  problem, as one user message;
* 4 runs per problem (MathArena's `--n` default); the headline `accuracy` is the mean over all
  runs, which is MathArena's score;
* MathArena's last-chance reprompt for every reply whose final message has no parseable
  `\\boxed{}` answer;
* grading by MathArena's own parser.py / grader.py (vendored verbatim), run in the local
  `gbench-matharena:b89f2f0a` image. The suite hard-errors (infra_required) if Docker or the image is
  missing, or if the image fails the grader self-check.

`leaderboard_comparable` is True only with 4 or more runs per problem, MathArena's sampling
(below), the full problem set and every response graded; the reason names anything else.

Sampling: MathArena's rule, with and without --thinking: each model runs with its own
configs/models/<provider>/<model>.yaml, which carries the provider's recommended sampling and the
model's max_tokens, on the first request and on the last-chance request alike
(hmmt.served_model_sampling, shared by hmmt and aime):

* MathArena ships a config for the served model (hmmt.MATHARENA_MODEL_CONFIGS, the 59 open-weight
  configs at the pinned commit, matched by huggingface_id against the served model name and the
  run's --tokenizer; for example Qwen/Qwen3.5-27B): its sampling fields (temperature, top_p,
  presence_penalty, top_k, min_p, repetition_penalty, ...), its reasoning_effort and its
  max_tokens are sent. max_tokens is that model's own setting, for reasoning too, through base's
  token-limit policy: exactly without --thinking, never below --max-output-tokens with it; a
  config without one sends none. Its reasoning switch must match --thinking, and a field with no
  local counterpart makes the run non-comparable; the result records the config under
  `matharena_model_config`.
* MathArena ships none (every Gemma model): the temperature, top_p and top_k the served model's
  own generation_config.json sets (read as ojbench reads it: the run's --tokenizer, else the
  served model name); a field the file does not set is not sent. For Gemma 4
  (matharena_common.SAMPLING) that is temperature 1.0, top_p 0.95, top_k 64. max_tokens: none
  declared (MathArena defines no max_tokens for a model it never ran), so gbench sends the run's
  --max-output-tokens (base's token-limit policy, house budget: gbench's choice, not upstream's).
  The suite hard-errors (infra_required) when the file cannot be read, rather than send another
  model's settings.

Temperature: gbench's rule for every suite, whatever the config or generation_config.json says
(base.resolve_temperature): 1.0 with --thinking (for Gemma 4 its recommended value), 0.0 without;
a run whose temperature is not the config's or the file's is not comparable.
The other sampling fields stay the config's or the file's.
`--temperature`, or `GBENCH_HMMT_TEMPERATURE` for this suite alone (it takes precedence),
overrides the temperature in either mode; the result records it under `official_sampling` (and
the fields sent under `recommended_sampling`), and any other value, any other sampling field, a
config's max_tokens not sent in full (the served window too small), or a reply (first or last-chance) cut off by a
limit other than the config's max_tokens (`replies_cut_off_by_budget`; for Gemma, any reply that
reached --max-output-tokens) is not leaderboard-comparable.
No LLM judge is involved (MathArena's parser grades), so gbench's judge temperature (pinned at
0.0) plays no part.
"""

import json
import logging
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
from . import matharena_common as MA
from . import ojbench as _oj
from .base import (OFFICIAL_SAMPLING_PARAM_KEYS, OfficialSampling, apply_shard,
                   finalize_shard_fields, get_run_knob, resolve_official_sampling,
                   resolve_token_budget, run_eval_suite)
from .metrics import budget_cutoffs
from .limit_plan import eval_limit_plan
from .sampling import stratified_sample, presharded_for_base
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

PILLAR = "STEM & Scientific Reasoning"
DOCS_URL = "docs/evals/hmmt.md"

_COMP = MA.COMPETITIONS["hmmt/hmmt_feb_2026"]
_DATASET = _COMP.dataset
_REVISION = _COMP.revision
_COMPETITION = _COMP.name

#: The competition's instruction as the model sees it: configs/competitions/hmmt/
#: hmmt_feb_2026.yaml has `instruction: "Put your final answer within \\boxed{{}}."`, and the
#: `{{}}` is a str.format escape that renders as `{}` (tests hold the two together).
_ANSWER_INSTRUCTION = "Put your final answer within \\boxed{}."


def _config() -> Dict[str, Any]:
    """The vendored MathArena competition config (read on use, blob-checked)."""
    return MA.competition_config(_COMP)

#: MathArena's `--n` default: runs per problem, averaged into the reported accuracy.
_MATHARENA_RUNS = MA.OFFICIAL_RUNS


def _hmmt_prompt(problem: str) -> str:
    """MathArena's prompt for this competition (instruction, blank line, problem)."""
    return MA.build_prompt(_config(), problem)


def _problem_type(row: Any) -> str:
    """`problem_type` labels joined, e.g. "Algebra/Number Theory"."""
    types = (row or {}).get("problem_type") or []
    if isinstance(types, str):
        types = [types]
    return "/".join(str(t) for t in types) or "unlabelled"


def _load_hmmt_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load the pinned MathArena HMMT February 2026 set from the HF Hub."""
    try:
        from datasets import load_dataset
        rows = list(load_dataset(_DATASET, split="train", revision=_REVISION))
    except Exception as e:
        logger.error(f"Failed to load dataset for hmmt: {e}")
        raise RuntimeError(f"Could not load dataset for hmmt ({_DATASET}@{_REVISION[:8]}): {e}") from e

    if not rows:
        raise RuntimeError("Dataset for hmmt returned empty rows")
    # MathArena sorts by problem_idx; a subset is stratified by problem type, not a head.
    rows = sorted(rows, key=lambda r: int(r.get("problem_idx") or 0))
    # --shard first, over the full ordered set, then the --eval-limit draw within the shard.
    rows, _ = apply_shard(rows, "hmmt")
    rows = stratified_sample(rows, limit, _problem_type, seed="hmmt")

    samples = []
    for item in rows:
        prompt = item.get("problem")
        gold = item.get("answer")
        if not prompt or gold is None:
            raise RuntimeError(
                "hmmt: unexpected dataset schema (missing 'problem'/'answer'); "
                "refusing to fabricate sample data"
            )
        messages = [{"role": "user", "content": _hmmt_prompt(str(prompt))}]
        # MathArena's gold is str(problem["answer"]) (runs.py).
        samples.append((messages, str(gold),
                        {"category": _problem_type(item),
                         "problem_idx": item.get("problem_idx")}))

    logger.info(f"Loaded {len(samples)} hmmt samples.")
    return samples


def _eval_hmmt(response_text: str, gold_target: str) -> bool:
    """MathArena's grader on one final response, run in THIS interpreter.

    Not used by run_hmmt, which grades whole conversations (after the last-chance reprompt) in
    the gbench-matharena:b89f2f0a image. Kept for unit tests and diagnostics; answers that need sympy's
    LaTeX parser can grade differently here, because this interpreter's antlr runtime is not the
    4.11 MathArena pins.
    """
    if not response_text or not str(gold_target).strip():
        return False
    return MA.grade_final_response(response_text, str(gold_target), _COMP.key)


#: MathArena's own model configs (configs/models/<provider>/<model>.yaml at MA.MATHARENA_COMMIT)
#: for the models whose weights are public (other_params.huggingface_id set: 59 of the 170), the
#: ones a local server can serve. Key: the path under configs/models without .yaml; value: (the
#: file's git blob sha at that commit, its huggingface_id, its fields). The fields are the file's
#: minus the ones MathArena uses only for its website, costs and client setup (human_readable_id,
#: date, other_params, read_cost, write_cost, cache_read_cost, cache_write_cost,
#: concurrent_requests, api_key_env, base_url). Transcribed from the pinned commit; each blob sha
#: was checked against the tree when this table was written. MathArena is MIT-licensed, Copyright
#: (c) 2024 SRI Lab, ETH Zurich (full text in matharena_src/LICENSE).
MATHARENA_MODEL_CONFIGS: Dict[str, Tuple[str, str, Dict[str, Any]]] = {
    "deepseek/deepseek_distill_1.5b": (
        "f3435d7cfc4a9ad97d9629715ca253ddc39c8de2", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        {"model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "api": "together", "max_tokens":
         32000, "temperature": 0.6, "top_p": 0.95}),
    "deepseek/deepseek_distill_14b": (
        "e40cc438162b175bc56109138df11e7b73003b09", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        {"model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "api": "vllm", "max_tokens": 32000,
         "temperature": 0.6, "top_p": 0.95}),
    "deepseek/deepseek_distill_32b": (
        "107496330d873a55663d733f80ab2957b7dc8932", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        {"model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "api": "vllm", "max_tokens": 32000,
         "temperature": 0.6, "top_p": 0.95}),
    "deepseek/deepseek_distill_70b": (
        "99428eb0f0cc6bb4bd1876792e483ccdfaaee60f", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        {"model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "api": "together", "max_tokens":
         32000, "temperature": 0.6, "top_p": 0.95}),
    "deepseek/deepseek_r1": (
        "4ccd1d87ad09f325dce52a0a373265bbfd19e587", "deepseek-ai/DeepSeek-R1",
        {"model": "deepseek-ai/DeepSeek-R1", "api": "together", "max_tokens": 32000, "temperature":
         0.6, "top_p": 0.95}),
    "deepseek/deepseek_r1_0528": (
        "98a5e67af6509f4746c36d3665e34238789b59fd", "deepseek-ai/DeepSeek-R1-0528",
        {"model": "deepseek-reasoner", "api": "deepseek", "max_tokens": 64000, "temperature": 0.6,
         "top_p": 0.95}),
    "deepseek/deepseek_v3": (
        "24eea722ab6f6b0f6c2aab839cfc9295b781408b", "deepseek-ai/DeepSeek-V3",
        {"model": "deepseek-ai/DeepSeek-V3", "api": "together", "max_tokens": 32000, "temperature":
         0.6, "top_p": 0.95}),
    "deepseek/deepseek_v31": (
        "33d5ea7b8464d80a6dfa6db858db441adff6d618", "deepseek-ai/DeepSeek-V3.1",
        {"model": "deepseek-reasoner", "api": "deepseek", "max_tokens": 64000, "temperature": 0.6,
         "top_p": 0.95}),
    "deepseek/deepseek_v31_terminus": (
        "17d92b2228b3b4d1fdc7d6a120265cc866c629df", "deepseek-ai/DeepSeek-V3.1-Terminus",
        {"model": "deepseek-reasoner", "api": "deepseek", "max_tokens": 64000, "temperature": 0.6,
         "top_p": 0.95}),
    "deepseek/deepseek_v32": (
        "ef898e17773a33207a852e009cca43fe4e7e5148", "deepseek-ai/DeepSeek-V3.2-Exp",
        {"model": "deepseek-reasoner", "api": "deepseek", "max_tokens": 64000, "temperature": 0.6,
         "top_p": 0.95}),
    "deepseek/deepseek_v32_dec": (
        "e3d570bfd97e195bc56673cefe5f92d7ae5e50bb", "deepseek-ai/DeepSeek-V3.2",
        {"model": "deepseek-reasoner", "api": "deepseek", "max_tokens": 64000, "temperature": 1,
         "top_p": 0.95}),
    "deepseek/deepseek_v32_special": (
        "3c485b2058800b1b82ce3b4670a59c74b8ee8626", "deepseek-ai/DeepSeek-V3.2-Speciale",
        {"model": "deepseek-reasoner", "api": "deepseek_special", "max_tokens": None,
         "temperature": 1, "top_p": 0.95}),
    "deepseek/deepseek_v3_03_24": (
        "8f1e4e3f5f4f75e17cddef3f99619451b3de13bd", "deepseek-ai/DeepSeek-V3-0324",
        {"model": "deepseek-chat", "api": "deepseek", "max_tokens": 64000, "temperature": 0.6,
         "top_p": 0.95}),
    "deepseek/deepseek_v4_flash": (
        "fce00b107e3f93de220f69cf22ca9874befc7e95", "deepseek-ai/DeepSeek-V4-Flash",
        {"model": "deepseek-v4-flash", "api": "deepseek", "harness": "deepcode", "harness_version":
         "0.3.1", "harness_config": {"auth": "api", "container_executable": "deepcode",
         "cpu_limit": 4, "memory_gb": 12, "pids_limit": 768, "tmpfs_size": "1g"}, "max_tokens":
         384000, "temperature": 1, "top_p": 1, "reasoning_effort": "max"}),
    "deepseek/deepseek_v4_pro": (
        "e219a8e27b07aa56c0c041fc8ad406ebcad1e2f7", "deepseek-ai/DeepSeek-V4-Pro",
        {"model": "deepseek-v4-pro", "api": "deepseek", "max_tokens": 384000, "temperature": 1,
         "top_p": 1, "reasoning_effort": "max"}),
    "glm/glm-45-air": (
        "a61a1a8bb26c8f96258fd5d5a3a47ff92a1327a3", "zai-org/GLM-4.5-Air",
        {"model": "glm-4.5-air", "api": "glm", "max_tokens": 81920, "temperature": 0.6, "top_p":
         0.95}),
    "glm/glm-45": (
        "2fb21b7547ea2cc0e94a265c454df29a23919d34", "zai-org/GLM-4.5",
        {"model": "glm-4.5", "api": "glm", "max_tokens": 81920, "temperature": 0.6, "top_p":
         0.95}),
    "glm/glm-45v": (
        "afb6a5a191449bbb7db173fed8c156777f9dbd93", "zai-org/GLM-4.5V",
        {"model": "glm-4.5v", "api": "glm", "max_tokens": 16000, "temperature": 0.6, "top_p":
         0.95}),
    "glm/glm-46": (
        "b77aa2e49c57d9f727ab62f8e41edf5adb90899b", "zai-org/GLM-4.6",
        {"model": "glm-4.6", "api": "glm", "max_tokens": 122880, "temperature": 1, "top_p": None}),
    "glm/glm-47": (
        "b863ca0e0bf663dd9e3937b4e4b18694b074bcaf", "zai-org/GLM-4.7",
        {"model": "glm-4.7", "api": "glm", "max_tokens": 122880, "temperature": 1, "top_p": None}),
    "glm/glm-5": (
        "0cfb45c6adb23bb4a84a943498ece05796049f77", "zai-org/GLM-5",
        {"model": "glm-5", "api": "glm", "max_tokens": 131072, "temperature": 1, "top_p": 0.95,
         "stream_openai_chat_completions": True}),
    "glm/glm-51": (
        "37e1067793fd1c39c98865cf382df331575c4e81", "zai-org/GLM-5.1",
        {"model": "glm-5.1", "api": "glm", "max_tokens": 131072, "temperature": 1, "top_p": 0.95,
         "stream_openai_chat_completions": True}),
    "glm/glm-52": (
        "0405749fc32c3be5987fd7c98e7352bf4408aeb9", "zai-org/GLM-5.2",
        {"model": "glm-5.2", "api": "bigmodel", "max_tokens": 131072, "temperature": 1.0, "top_p":
         0.95, "reasoning_effort": "max", "extra_body": {"thinking": {"type": "enabled"}},
         "stream_openai_chat_completions": True}),
    "glm/glm-53-flash": (
        "d86e11d0a19417c5f00d49ff1d4e0ed6a7e07bdf", "zai-org/GLM-5.3-Flash",
        {"harness": "kimi", "harness_version": "0.36.0", "model": "glm-5.3-flash", "api": "glm",
         "max_tokens": 131072, "temperature": 1.0, "top_p": 0.95, "reasoning_effort": "max",
         "extra_body": {"thinking": {"type": "enabled", "clear_thinking": False}},
         "harness_config": {"auth": "api", "container_executable": "kimi", "environment":
         {"KIMI_MODEL_MAX_CONTEXT_SIZE": "1048576"}, "cpu_limit": 4, "memory_gb": 12, "pids_limit":
         768, "tmpfs_size": "1g"}}),
    "glm/glm-53": (
        "a1fa7a5e03d32e09aca4ca24a6a04a39313b2b05", "zai-org/GLM-5.3",
        {"model": "glm-5.3", "api": "glm", "harness": "opencode", "harness_version": "1.18.27",
         "max_tokens": 131072, "temperature": 1.0, "top_p": 0.95, "reasoning_effort": "max",
         "extra_body": {"thinking": {"type": "enabled"}}, "stream_openai_chat_completions": True,
         "harness_config": {"auth": "api", "container_executable": "opencode", "cpu_limit": 4,
         "memory_gb": 12, "pids_limit": 768, "tmpfs_size": "1g"}}),
    "llama4/maverick": (
        "af9874080eb0816d35747b749a3c924fd655d818",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        {"model": "meta-llama/llama-4-maverick", "api": "openrouter", "max_tokens": 32000,
         "temperature": 0.6, "top_p": 0.9, "provider": {"allow_fallbacks": False, "order":
         ["Fireworks"]}}),
    "mistral/leanstral_1_5": (
        "1000f48a76e88d25c872a6b84dcab5a151206f5f", "mistralai/Leanstral-1.5-119B-A6B",
        {"model": "leanstral-1.5", "api": "custom", "max_tokens": None, "temperature": None,
         "top_p": None}),
    "mistral/leanstral_1_5_kimi_k3": (
        "29021a057ab84dc9f971d708c7a49b5cf08d0f2d", "mistralai/Leanstral-1.5-119B-A6B",
        {"model": "leanstral-1.5-kimi-k3", "api": "custom", "max_tokens": None, "temperature":
         None, "top_p": None, "lean_environment_override": "lean-4.30.0"}),
    "moonshot/k2-thinking": (
        "788968d63502a88271fd650c9b7beb6e3ed1e77b", "moonshotai/Kimi-K2-Thinking",
        {"model": "moonshotai/kimi-k2-thinking", "api": "openrouter", "max_tokens": 256000,
         "temperature": 1.0, "context_limit": 256000, "extra_body": {"provider":
         {"allow_fallbacks": False, "order": ["moonshotai"]}}}),
    "moonshot/k25": (
        "596ecb461fb6388d61eac4bcd66c3de5762865da", "moonshotai/Kimi-K2.5",
        {"model": "moonshotai/kimi-k2.5", "api": "openrouter", "max_tokens": 256000, "temperature":
         1.0, "context_limit": 256000, "reasoning_effort": "high", "extra_body": {"provider":
         {"allow_fallbacks": False, "order": ["moonshotai"]}}}),
    "moonshot/k26": (
        "5b96bc3e77140d847fcd2bae7eb8b253ba1d0e75", "moonshotai/Kimi-K2.5",
        {"model": "moonshotai/kimi-k2.6", "api": "openrouter", "max_tokens": 256000, "temperature":
         1.0, "top_p": 0.95, "context_limit": 256000, "reasoning_effort": "high", "extra_body":
         {"provider": {"allow_fallbacks": False, "order": ["moonshotai"]}}}),
    "moonshot/k3": (
        "4eb89b49015f71898957e056ac6c9f5aa8a71546", "moonshotai/Kimi-K3",
        {"model": "moonshotai/kimi-k3", "api": "openrouter", "harness": "kimi", "harness_config":
         {"auth": "api", "container_executable": "kimi"}, "max_tokens": 1000000, "temperature":
         1.0, "top_p": 0.95, "context_limit": 1000000, "reasoning_effort": "max", "extra_body":
         {"provider": {"allow_fallbacks": False, "order": ["moonshotai"]}}}),
    "nvidia/nemotron-3-super": (
        "a8fbbdc3c21f5d444f63aa22b6d35bec35ffc7a4",
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
        {"model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", "api": "vllm", "max_tokens":
         192000, "temperature": 1.0, "top_p": 0.95, "extra_body": {"chat_template_kwargs":
         {"enable_thinking": True}}}),
    "openai/oss-120b": (
        "8d734c11fe9a4b52389597ff6abb9e091b05723e", "openai/gpt-oss-120b",
        {"model": "openai/gpt-oss-120b", "api": "together", "max_tokens": 128000,
         "reasoning_effort": "high"}),
    "openai/oss-20b": (
        "f41da76553d9aa8d2bec034b3ea557f1b5ff9995", "openai/gpt-oss-20b",
        {"model": "openai/gpt-oss-20b", "api": "together", "max_tokens": 128000,
         "reasoning_effort": "high"}),
    "other/k2-think": (
        "52943665146db1a032ba2d368c2d28293b421775", "LLM360/K2-Think",
        {"model": "LLM360/K2-Think", "api": "vllm", "max_tokens": 64000, "temperature": 1.0,
         "top_p": 0.95}),
    "other/qed-nano": (
        "8816cec072d13994f10037ca60d22180a43a5ecb", "lm-provers/QED-Nano",
        {"model": "lm-provers/QED-Nano", "api": "custom", "max_tokens": 120000, "temperature": 0.6,
         "top_p": 0.95}),
    "phi4/14b-reasoning-plus": (
        "843cf64f9cb66e3fa4ded3c1a0277150e6f9b028", "microsoft/Phi-4-reasoning",
        {"model": "microsoft/phi-4-reasoning-plus:free", "api": "openrouter", "max_tokens": 31500,
         "temperature": 0.8, "top_p": 0.95}),
    "qwen/qwen3.5_0.8b": (
        "155d4ae3aad0e71cd5016a8de51cfa962256a722", "Qwen/Qwen3.5-0.8B",
        {"model": "qwen/qwen3.5-0.8b", "api": "custom", "max_tokens": 65500, "temperature": 1.0,
         "top_p": 0.95, "presence_penalty": 1.5, "extra_body": {"top_k": 20, "min_p": 0.0,
         "repetition_penalty": 1.0}}),
    "qwen/qwen3.5_27b": (
        "7f373a25ea8bc060f03177591337df2c32984b4c", "Qwen/Qwen3.5-27B",
        {"model": "qwen/qwen3.5-27b", "api": "custom", "max_tokens": 192000, "temperature": 1.0,
         "top_p": 0.95, "presence_penalty": 1.5, "extra_body": {"top_k": 20, "min_p": 0.0,
         "repetition_penalty": 1.0}}),
    "qwen/qwen3.5_2b": (
        "8f721779e3cd7cbe2eef3642e5e078d7042416c6", "Qwen/Qwen3.5-2B",
        {"model": "qwen/qwen3.5-2b", "api": "custom", "max_tokens": 81920, "temperature": 1.0,
         "top_p": 0.95, "presence_penalty": 1.5, "extra_body": {"top_k": 20, "min_p": 0.0,
         "repetition_penalty": 1.0, "chat_template_kwargs": {"enable_thinking": True}}}),
    "qwen/qwen3.5_2b_arxivmath_brokenarxiv_sft": (
        "fc14e75824ed4d1393217ade2d13aa9b8520017c",
        "MathArena/qwen3.5-2b-arxivmath-brokenarxiv-sft",
        {"model": "MathArena/qwen3.5-2b-arxivmath-brokenarxiv-sft", "api": "custom", "max_tokens":
         81920, "temperature": 1.0, "top_p": 0.95, "presence_penalty": 1.5, "extra_body": {"top_k":
         20, "min_p": 0.0, "repetition_penalty": 1.0, "chat_template_kwargs": {"enable_thinking":
         True}}}),
    "qwen/qwen3.5_2b_arxivmath_grpo": (
        "279fe9c6a795f4242b4352d32850f8fbebfbcf94", "MathArena/qwen3.5-2b-arxivmath-grpo",
        {"model": "matharena/qwen3.5-2b-arxivmath-grpo", "api": "custom", "max_tokens": 32768,
         "temperature": 1.0, "top_p": 0.95, "presence_penalty": 1.5, "extra_body": {"top_k": 20,
         "min_p": 0.0, "repetition_penalty": 1.0, "chat_template_kwargs": {"enable_thinking":
         True}}}),
    "qwen/qwen3.5_2b_brokenarxiv_sft": (
        "03004db2ebfbe742e4455d0bbfcde78a02f4e8e2", "MathArena/qwen3.5-2b-brokenarxiv-sft",
        {"model": "MathArena/qwen3.5-2b-brokenarxiv-sft", "api": "custom", "max_tokens": 81920,
         "temperature": 1.0, "top_p": 0.95, "presence_penalty": 1.5, "extra_body": {"top_k": 20,
         "min_p": 0.0, "repetition_penalty": 1.0, "chat_template_kwargs": {"enable_thinking":
         True}}}),
    "qwen/qwen3.5_2b_sft": (
        "1fdf3ab2289f27f7bf470d0fed9fc9e85be7a8a3", "MathArena/qwen3.5-2b-arxivmath-sft",
        {"model": "MathArena/qwen3.5-2b-arxivmath-sft", "api": "custom", "max_tokens": 81920,
         "temperature": 1.0, "top_p": 0.95, "presence_penalty": 1.5, "extra_body": {"top_k": 20,
         "min_p": 0.0, "repetition_penalty": 1.0, "chat_template_kwargs": {"enable_thinking":
         True}}}),
    "qwen/qwen3.5_35b_a3b": (
        "9ab3f04dccf2ed0fabccb91e537dcadfc55c3af8", "Qwen/Qwen3.5-35B-A3B",
        {"model": "Qwen/Qwen3.5-35B-A3B", "api": "vllm", "max_tokens": 192000, "temperature": 1.0,
         "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "repetition_penalty": 1.0}),
    "qwen/qwen3.5_397b": (
        "8076643294a409f051bbe920924c72bbcb9c6ec1", "Qwen/Qwen3.5-397B-A17B",
        {"model": "qwen/qwen3.5-397b-a17b", "api": "openrouter", "max_tokens": 65536,
         "temperature": 0.6, "top_p": 0.95, "extra_body": {"provider": {"allow_fallbacks": False,
         "order": ["alibaba"]}, "top_k": 20}}),
    "qwen/qwen3.5_4b": (
        "60f9f8665e5afc1765b742c485541eb194478390", "Qwen/Qwen3.5-4B",
        {"model": "qwen/qwen3.5-4b", "api": "custom", "max_tokens": 65500, "temperature": 1.0,
         "top_p": 0.95, "presence_penalty": 1.5, "extra_body": {"top_k": 20, "min_p": 0.0,
         "repetition_penalty": 1.0}}),
    "qwen/qwen3.5_9b": (
        "262ad4491917914f09c5f590805fe4c2c373c362", "Qwen/Qwen3.5-9B",
        {"model": "Qwen/Qwen3.5-9B", "api": "together", "max_tokens": 192000, "temperature": 1.0,
         "top_p": 0.95, "presence_penalty": 1.5, "extra_body": {"top_k": 20, "min_p": 0.0,
         "repetition_penalty": 1.0}}),
    "qwen/qwen3.6_35b": (
        "6857cc7ec86dca2b22e300fea9b22562150e61cb", "Qwen/Qwen3.6-35B-A3B",
        {"model": "Qwen/Qwen3.6-35B-A3B", "api": "custom", "max_tokens": 81920, "temperature": 1.0,
         "top_p": 0.95, "extra_body": {"top_k": 20, "chat_template_kwargs": {"enable_thinking":
         True}}}),
    "qwen/qwen3.8_max": (
        "4c4655d420f3d021a565a6b2a021ff9d90850730", "Qwen/Qwen3.8-2.4T-A95B",
        {"model": "qwen/qwen3.8-max", "api": "openrouter", "harness": "qwen", "harness_config":
         {"auth": "api", "container_executable": "qwen", "environment":
         {"QWEN_STREAM_MAX_LIFETIME_MS": "0"}}, "max_tokens": 131072, "temperature": 1.0, "top_p":
         0.95, "reasoning_effort": "max", "extra_body": {"provider": {"allow_fallbacks": False,
         "order": ["alibaba"]}, "top_k": 20}}),
    "qwen/qwen3_235b_a22b_2507": (
        "1b1e02efab7ae7a5b29aa5f4145c01b9b9f5e9f3", "Qwen/Qwen3-235B-A22B-Thinking-2507",
        {"model": "qwen/qwen3-235b-a22b-thinking-2507", "api": "openrouter", "max_tokens": 81920,
         "temperature": 0.6, "top_p": 0.95, "top_k": 20, "extra_body": {"provider":
         {"allow_fallbacks": False, "order": ["deepinfra"]}}}),
    "qwen/qwen3_30b_a3b_2507": (
        "1e43e21053dbb9c635b6d2afea46d3b278216237", "Qwen/Qwen3-30B-A3B-Thinking-2507",
        {"model": "Qwen/Qwen3-30B-A3B-Thinking-2507", "api": "vllm", "max_tokens": 81920,
         "temperature": 0.6, "top_p": 0.95}),
    "qwen/qwen3_4b_2507": (
        "c1fcda94cdef6feeb2056a05743d02e6f24c102c", "Qwen/Qwen3-4B-Thinking-2507",
        {"model": "Qwen/Qwen3-4B-Thinking-2507", "api": "vllm", "max_tokens": 81920, "temperature":
         0.6, "top_p": 0.95}),
    "qwen/qwen3_vl_235b_a22b_instruct": (
        "21c80e9bdc5e491343e9e84784932c3e7417b851", "Qwen/Qwen3-VL-235B-A22B-Instruct",
        {"model": "qwen/qwen3-vl-235b-a22b-instruct", "api": "openrouter", "max_tokens": 128000,
         "temperature": 0.6, "top_p": 0.95, "extra_body": {"provider": {"allow_fallbacks": False,
         "order": ["parasail"]}}}),
    "qwen/qwq-preview": (
        "5f9cc95517da4f9ab1cf7c330bdcf51b215fba27", "Qwen/QwQ-32B-Preview",
        {"model": "Qwen/QwQ-32B-Preview", "api": "together", "max_tokens": 24000, "temperature":
         0.6, "top_p": 0.95}),
    "stepfun/3.5-flash": (
        "6c753a6a8c227dcc37dc49b7a07aa29e236d2920", "stepfun-ai/Step-3.5-Flash",
        {"model": "step-3.5-flash", "api": "stepfun", "max_tokens": 250000, "batch_processing":
         False, "use_openai_responses_api": False, "temperature": 1, "top_p": 1,
         "stream_openai_chat_completions": True}),
    "stepfun/3.7-flash": (
        "d4e91b33460bba05f96a69e216ea466e661f79ca", "stepfun-ai/Step-3.7-Flash",
        {"model": "step-3.7-flash", "api": "stepfun", "max_tokens": 250000, "batch_processing":
         False, "use_openai_responses_api": False, "reasoning_effort": "high",
         "stream_openai_chat_completions": True}),
    "tiiuae/7b-reasoning": (
        "33dc18bace852a565d4103409776e7e0cf140840", "tiiuae/Falcon-H1R-7B",
        {"model": "tiiuae/Falcon-H1R-7B", "api": "tiiuae", "max_tokens": 65536, "batch_processing":
         False, "use_openai_responses_api": False, "temperature": 0.6, "top_p": 0.95, "extra_body":
         {"repetition_penalty": 1.1}}),
}

#: Fields MathArena's runner and APIClient consume themselves (client setup, streaming, retries,
#: tool environments, batch submission): none of them reaches the chat request. batch_processing
#: is an APIClient constructor argument that picks the provider's batch API, and APIClient turns
#: it off for every API but anthropic and openai (api_client.py APIClient.__init__).
_MA_CLIENT_FIELDS = frozenset({
    "model", "api", "timeout", "context_limit", "background", "stream_openai_chat_completions",
    "stream_openai_responses", "sleep_after_request", "max_retries", "max_retries_inner",
    "no_system_messages", "max_tokens_param", "lean_environment_override", "harness_version",
    "harness_config", "batch_processing"})
#: Sampling fields an OpenAI-compatible request carries at the top level of the JSON body
#: (MathArena passes them as create() kwargs, or inside extra_body, which the client merges into
#: the body).
_MA_SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty",
                       "frequency_penalty", "repetition_penalty", "seed", "stop")
#: MathArena's offline vLLM path (api: vllm, APIClient._initialize_vllm) builds SamplingParams from
#: these fields only.
_MA_VLLM_FIELDS = ("temperature", "top_p", "max_tokens", "top_k", "repetition_penalty",
                   "presence_penalty")


def _request_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    """A config's fields that shape the request (for comparing two configs)."""
    return {k: v for k, v in fields.items() if k not in _MA_CLIENT_FIELDS}


def matharena_model_config(model_name: Optional[str]
                           ) -> Optional[Tuple[str, str, Dict[str, Any], str]]:
    """MathArena's config for the served model: (its path under configs/models, blob sha, fields,
    the name that matched), or None when MathArena ships none (Gemma, for one).

    The served model name and the run's --tokenizer are matched against each config's
    huggingface_id, case-insensitively: the full id first, then the last path component (a local
    directory named after the model). Two configs with one huggingface_id (MathArena's k25 and
    k26 both name moonshotai/Kimi-K2.5) are told apart by their `model` field; configs that still
    tie must send the same request, or this raises (a table error, caught by the tests)."""
    names: List[str] = []
    for ident in (model_name, get_run_knob("tokenizer")):
        ident = str(ident or "").strip().rstrip("/")
        if ident and ident not in names:
            names.append(ident)
    for whole in (True, False):
        for name in names:
            want = (name if whole else name.rsplit("/", 1)[-1]).lower()
            hits = [path for path, (_, hf, _) in sorted(MATHARENA_MODEL_CONFIGS.items())
                    if (hf if whole else hf.rsplit("/", 1)[-1]).lower() == want]
            if not hits:
                continue
            if len(hits) > 1:
                last = name.rsplit("/", 1)[-1].lower()
                named = [p for p in hits if str(MATHARENA_MODEL_CONFIGS[p][2].get("model") or "")
                         .rsplit("/", 1)[-1].lower() == last]
                hits = named or hits
            requests = {json.dumps(_request_fields(MATHARENA_MODEL_CONFIGS[p][2]), sort_keys=True,
                                   default=str) for p in hits}
            if len(requests) > 1:
                raise RuntimeError(f"MathArena ships several configs for {name!r} that send "
                                   f"different requests: {hits}")
            blob, _, fields = MATHARENA_MODEL_CONFIGS[hits[0]]
            return hits[0], blob, fields, name
    return None


def translate_matharena_config(path: str, blob: str, fields: Dict[str, Any], thinking: bool
                               ) -> Tuple[OfficialSampling, Dict[str, Any], List[str], List[str]]:
    """What MathArena's config `path` sends, for the local endpoint: (the OfficialSampling it
    defines, other request fields to send as they are, fields that do not apply to a local
    endpoint, fields the run cannot reproduce).

    As APIClient does (api_client.py `_kwarg_remover`): a temperature is dropped for a model name
    containing o1/o3/o4, and a top_p, top_k or temperature of None is not sent. The api: vllm
    configs run MathArena's offline vLLM path, which passes only _MA_VLLM_FIELDS to
    SamplingParams. Elsewhere every field is a request field: the sampling fields (top level or in
    extra_body) and max_tokens are sent, reasoning_effort too; the reasoning switch
    (extra_body.chat_template_kwargs.enable_thinking, or GLM's extra_body.thinking type enabled)
    must match --thinking; OpenRouter's provider routing does not apply to a local server; any
    other field is not reproduced. max_tokens is the model's own setting, for reasoning too: sent
    exactly without --thinking, never below --max-output-tokens with it (base's token-limit
    policy); a config without one (or with null) sends none."""
    cfg = dict(fields)
    model = str(cfg.get("model") or "")
    if any(kw in model for kw in ("o1", "o3", "o4")):
        cfg.pop("temperature", None)
    for key in ("top_p", "top_k", "temperature"):
        if key in cfg and cfg[key] is None:
            del cfg[key]
    sent: Dict[str, Any] = {}
    extra: Dict[str, Any] = {}
    not_applicable: List[str] = []
    gaps: List[str] = []
    mode = "--thinking" if thinking else "no-think"
    if cfg.get("api") == "vllm":
        for key, value in cfg.items():
            if key in _MA_VLLM_FIELDS:
                sent[key] = value
            elif key not in _MA_CLIENT_FIELDS:
                not_applicable.append(f"{key}={value!r} (MathArena's offline vLLM path passes only "
                                      f"{', '.join(_MA_VLLM_FIELDS)} to SamplingParams)")
    else:
        body = dict(cfg)
        extra_body = body.pop("extra_body", None) or {}
        items = list(body.items()) + [(f"extra_body.{k}", v) for k, v in extra_body.items()]
        for key, value in items:
            name = key.split(".", 1)[1] if key.startswith("extra_body.") else key
            if key in _MA_CLIENT_FIELDS:
                continue
            if name in _MA_SAMPLING_FIELDS or name == "max_tokens":
                sent[name] = value
            elif name == "reasoning_effort":
                extra["reasoning_effort"] = value
            elif name == "provider":
                not_applicable.append(f"{key}={value!r} (OpenRouter provider routing; the local "
                                      "server is the provider)")
            elif key == "harness":
                not_applicable.append(f"harness={value!r} (the competition sets no allow_harness, "
                                      "so MathArena runs the plain model API)")
            elif key == "use_openai_responses_api" and not value:
                continue
            elif (key == "extra_body.chat_template_kwargs" and isinstance(value, dict)
                  and set(value) <= {"enable_thinking"}):
                wanted = value.get("enable_thinking")
                if wanted is not None and bool(wanted) != bool(thinking):
                    gaps.append(f"MathArena's config {path} sets enable_thinking={wanted}; this "
                                f"run is {mode}")
            elif key == "extra_body.thinking" and value == {"type": "enabled"}:
                if not thinking:
                    gaps.append(f"MathArena's config {path} runs with thinking enabled "
                                f"(extra_body.thinking); this run is {mode}")
            else:
                gaps.append(f"{key}={value!r} of MathArena's config {path} has no counterpart on "
                            "a local endpoint and was not sent")
    cap = sent.pop("max_tokens", None)
    send_temperature = "temperature" in sent
    common = dict(temperature=sent.pop("temperature", None), send_temperature=send_temperature,
                  params=dict(sent))
    ref = (f"{MA.MATHARENA_REPO}@{MA.MATHARENA_COMMIT[:8]} configs/models/{path}.yaml (blob "
           f"{blob[:12]}): {json.dumps(_request_fields(fields), sort_keys=True, default=str)}")
    if cap is None:
        sampling = OfficialSampling(reference=ref + "; no max_tokens is sent",
                                    send_max_tokens=False, **common)
    else:
        sampling = OfficialSampling(reference=ref, max_tokens=int(cap), **common,
                                    reasoning=OfficialSampling(reference=ref, max_tokens=int(cap),
                                                               **common))
    return sampling, extra, not_applicable, gaps


class ServedModelSampling(NamedTuple):
    """How MathArena samples the served model (served_model_sampling)."""
    sampling: OfficialSampling
    #: Request fields other than sampling that MathArena's config sends (reasoning_effort).
    extra_fields: Dict[str, Any]
    #: The result's `recommended_sampling`.
    record: Dict[str, Any]
    #: The result's `matharena_model_config` (None: MathArena ships no config for the model).
    model_config: Optional[Dict[str, Any]]
    #: Fields of MathArena's config the run cannot reproduce.
    gaps: List[str]


def served_model_sampling(eval_name: str, model_name: str, docs_url: str,
                          thinking: bool) -> ServedModelSampling:
    """MathArena's sampling for the served model, shared by hmmt and aime.

    MathArena runs each model with its configs/models yaml, whose fields carry the provider's
    recommended sampling and the model's max_tokens. When MathArena ships a config for the served
    model (matharena_model_config), its fields are reproduced (translate_matharena_config). When
    it ships none (every Gemma model), the provider's recommended temperature, top_p and top_k are
    read from the served model's own generation_config.json (the run's --tokenizer, else the
    served model name; see ojbench), a field the file does not set is not sent, and max_tokens is
    the run's (no official cap). An unreadable generation_config.json is infra_required, never
    another model's settings."""
    found = matharena_model_config(model_name)
    if found:
        path, blob, fields, matched = found
        sampling, extra, not_applicable, gaps = translate_matharena_config(path, blob, fields,
                                                                           thinking)
        source = f"{MA.MATHARENA_REPO}@{MA.MATHARENA_COMMIT[:8]} configs/models/{path}.yaml"
        record = {"source": source,
                  **({"temperature": sampling.temperature} if sampling.send_temperature else {}),
                  **dict(sampling.params or {}),
                  "max_tokens": sampling.max_tokens if sampling.send_max_tokens else "not sent",
                  **extra}
        config = {"path": f"configs/models/{path}.yaml", "blob": blob,
                  "huggingface_id": MATHARENA_MODEL_CONFIGS[path][1], "matched_name": matched,
                  "fields": dict(fields), "not_applicable": not_applicable,
                  "not_reproduced": gaps}
        if fields.get("api") == "vllm":
            # MathArena's offline vLLM path renders the model's default chat template
            # (apply_chat_template with no chat_template_kwargs); gbench's requests carry the run's
            # thinking switch. A template that does not read enable_thinking renders the same.
            config["chat_template"] = (
                "MathArena renders the model's default chat template (offline vLLM, no "
                f"chat_template_kwargs); this run sends enable_thinking={bool(thinking)}, which "
                "renders differently only for a template that reads the switch and defaults "
                "otherwise")
        return ServedModelSampling(sampling, extra, record, config, gaps)
    try:
        recommended, source = _oj.served_model_recommended_sampling(model_name)
    except RuntimeError as e:
        raise infra_required(
            eval_name, "MathArena samples each model at its provider's recommended temperature, "
            f"top_p and top_k, from its configs/models yaml, which MathArena ships none of for "
            f"this model, else read from its generation_config.json, and {e}", docs_url) from e
    shown = ", ".join(f"{k} {recommended[k]}" for k in _oj.RECOMMENDED_FIELDS
                      if k in recommended)
    reference = (f"{MA.MATHARENA_REPO}@{MA.MATHARENA_COMMIT[:8]} configs/models/<provider>/"
                 "<model>.yaml (each model at its provider's recommended sampling; none ships for "
                 "this model): the served model's generation_config.json, read from "
                 f"{source}: {shown or 'none set'}")
    return ServedModelSampling(_oj.recommended_official_sampling(recommended, reference), {},
                               {"source": source, **recommended}, None, [])


def served_model_reasons(eval_name: str, result: Dict[str, Any], served: ServedModelSampling,
                         user_payload: Dict[str, Any], thinking: bool) -> List[str]:
    """How the run departs from MathArena's settings for the served model, beyond what
    matharena_common.comparability_reasons checks (temperature, the recommended fields, decoding
    penalties): fields of its config not reproduced, a request field it does not send or sends
    otherwise, its max_tokens not sent in full, and replies cut off by a limit other than its
    max_tokens (recorded as `replies_cut_off_by_budget` / `replies_cut_off_in_reasoning`)."""
    reasons = list(served.gaps)
    official = dict(served.sampling.params or {})
    added = sorted(k for k in user_payload
                   if k in OFFICIAL_SAMPLING_PARAM_KEYS and k not in official)
    if added:
        reasons.append("sampling " + ", ".join(f"{k}={user_payload[k]!r}" for k in added)
                       + "; MathArena sends no such field for this model")
    for key, value in served.extra_fields.items():
        if user_payload.get(key, value) != value:
            reasons.append(f"{key}={user_payload[key]!r}; MathArena's config sends {value!r}")
    plan = resolve_token_budget(eval_name, served.sampling, thinking=thinking)
    path = (served.model_config or {}).get("path")
    if plan.policy in ("official", "official_reasoning"):
        sent = result.get("max_tokens_sent") or []
        if not sent or min(sent) < plan.official_cap:
            reasons.append(f"max_tokens={min(sent) if sent else 'not sent'}; MathArena's config "
                           f"{path} sends {plan.official_cap} (a request whose prompt left less "
                           "room in the served context window was sent less; serve a larger "
                           "--max-model-len)")
    # A reply (first or last-chance) stopped at a limit other than the config's max_tokens
    # measures that limit, not the model: --max-output-tokens for a model MathArena ships no
    # config for, a max_tokens the window lowered, or the window's end where none is sent.
    cut, in_reasoning = budget_cutoffs(result)
    official = set(result.get("official_max_tokens_sent") or [])
    # The last-chance request's own max_tokens (its prompt holds the whole first reply, so the
    # window often lowers it below the first request's); the first request's value only for a
    # record that does not carry one.
    last = [t["matharena_last_chance"] for t in result.get("sample_traces") or []
            if isinstance(t.get("matharena_last_chance"), dict)
            and t["matharena_last_chance"].get("finish_reason") == "length"
            and t["matharena_last_chance"].get("max_tokens_sent", t.get("max_tokens_sent"))
            not in official]
    last_empty = sum(1 for r in last if not str(r.get("response_text") or "").strip())
    result["replies_cut_off_by_budget"] = cut + len(last)
    result["replies_cut_off_in_reasoning"] = in_reasoning + last_empty
    if cut or last:
        limit = (f"a limit other than the max_tokens of MathArena's config {path}" if path
                 else "--max-output-tokens (MathArena ships no config for this model, so it "
                      "defines no max_tokens for it; gbench sends --max-output-tokens, its house "
                      "budget: raise the flag)")
        reasons.append(f"{cut + len(last)} repl(ies) ({cut} first, {len(last)} last-chance) cut "
                       f"off by {limit}, {in_reasoning + last_empty} of them inside the reasoning "
                       "channel with no answer")
    return reasons


def run_hmmt(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute hmmt native built-in evaluation suite."""
    ok, why = MA.check_prerequisites()
    if not ok:
        raise infra_required("hmmt", why, DOCS_URL)
    config = _config()
    broken = MA.self_check(config)
    if broken:
        raise infra_required("hmmt", broken, DOCS_URL)
    limit = kwargs.get("limit")
    samples = _load_hmmt_samples(limit=limit)
    attempts = max(_MATHARENA_RUNS, int(kwargs.get("attempt_count") or 1))
    # --eval-limit N is N problems for the whole suite; MathArena's runs per problem are attempts.
    limit_plan = eval_limit_plan("hmmt", limit, len(samples), attempts=attempts)
    # MathArena samples each model with its own config's settings (or, when it ships none, the
    # served model's own generation_config.json values), never another model's.
    served = served_model_sampling("hmmt", model_name, DOCS_URL, enable_thinking)
    sampling = served.sampling
    temperature, _ = resolve_official_sampling("hmmt", sampling, thinking=enable_thinking)
    user_payload = dict(kwargs.get("extra_payload") or {})
    extra_payload = {**served.extra_fields, **user_payload} or None
    payload = MA.request_payload(extra_payload, sampling)
    stats: Dict[str, Any] = {}
    result = run_eval_suite(
        eval_name="hmmt",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=presharded_for_base(samples),  # this shard's problems, kept as is
        async_eval_fn=MA.make_scorer(
            _COMP, config, stats, eval_name="hmmt", model_name=model_name,
            base_url=base_url, concurrency=concurrency, thinking=enable_thinking,
            temperature=temperature, extra_payload=payload),
        declared_scoring_mode="deterministic",   # MathArena's parser, not an LLM judge
        thinking=enable_thinking,
        extra_payload=extra_payload,
        limit=limit,
        max_output_tokens=kwargs.get("max_output_tokens", 8192),
        attempt_count=attempts,
        supports_attempts=True,    # every run is graded on its own
        official_sampling=sampling,
    )
    result["eval_limit_plan"] = limit_plan
    result["recommended_sampling"] = served.record
    result["matharena_model_config"] = served.model_config
    result["dataset"] = f"{_DATASET}@{_REVISION}"
    result["competition"] = _COMPETITION
    result["matharena"] = MA.matharena_record(_COMP, config, sampling)
    result["last_chance"] = {"reprompted": stats.get("last_chance_reprompts", 0),
                             "request_failures": stats.get("last_chance_failures", 0)}
    result["grading"] = {"errors": stats.get("grading_errors", 0) + stats.get("check_errors", 0),
                         "warnings": stats.get("warnings", {})}
    result["metric"] = (f"mean accuracy over {MA.attempts_run(result)} run(s) per problem "
                        f"(MathArena reports the mean over {_MATHARENA_RUNS}), graded by "
                        "MathArena's grader after its last-chance reprompt")
    MA.apply_grader_failure(result, stats)
    reasons = MA.comparability_reasons(result, stats, limit, payload=payload,
                                       recommended=sampling)
    reasons += served_model_reasons("hmmt", result, served, user_payload, enable_thinking)
    result["leaderboard_comparable"] = not reasons
    if reasons:
        result["leaderboard_comparable_reason"] = "; ".join(reasons)
    # One shard of a --shard I/N run is a partial set: the shard reason (finalize_shard_fields).
    return finalize_shard_fields(result)
