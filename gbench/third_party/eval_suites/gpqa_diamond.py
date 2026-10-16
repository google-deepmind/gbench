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

"""Native GPQA Diamond evaluation suite, simple-evals protocol.

The 198-question expert-curated subset, run exactly like `gpqa` (simple-evals `GPQAEval`: 10
repeats with fresh option permutations, the zero-shot CoT template and the `Answer: X`
pattern, see gpqa.py). Diamond is simple-evals' default GPQA variant, and the pinned Hugging
Face rows are identical, row for row, to the `gpqa_diamond.csv` simple-evals downloads, so
every copy gets the same permutation as in a simple-evals run.

Sampling: simple-evals' chat sampler, which also shapes the request, as in gpqa.py
(gpqa.SIMPLE_EVALS_SAMPLING, applied per mode by base's token-limit policy). Every request, with
and without `--thinking`, is shaped as simple_evals.py's `gpt-4o` entry sends it
(`ChatCompletionSampler`): the system message "You are a helpful assistant." before the user
message. The temperature follows gbench's rule (base.resolve_temperature), not the sampler's
default 0.5: 1.0 with `--thinking`, 0.0 without; the official 0.5 is recorded. A no-think request is sent its `max_tokens=2048` exactly; a reply
cut off at 2048 tokens is scored on what arrived, as the sampler returns it, and
`--max-output-tokens` is not applied. A `--thinking` request is sent the run's budget
(`--max-output-tokens`) instead, the 2048 being written for non-reasoning generation (recorded
as not applied to reasoning). simple-evals' o-series samplers (ResponsesSampler with
`reasoning_model=True`, OChatCompletionSampler) are keyed on OpenAI model names and are not
reasoning settings for a served model (see gpqa.py). See base.OfficialSampling; the result
records `official_sampling`. Override the temperature for a whole run with `--temperature`, or
for this suite alone with `GBENCH_GPQA_DIAMOND_TEMPERATURE`, which takes precedence over both
(and over the rule's 1.0 / 0.0); a value other than 0.5 makes the run non-comparable. LLM-judge grading
is pinned at 0.0 and is not affected by
either.

Sharding: `--shard I/N` selects shard I of the questions before the `--eval-limit` draw, so
every copy of a question runs in the same shard (gpqa._gpqa_items).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from .gpqa import (  # noqa: F401  (re-exported: tests and callers import them from here)
    ANSWER_PATTERN_MULTICHOICE, DATASET, DATASET_REVISION, N_REPEATS, QUERY_TEMPLATE_MULTICHOICE,
    SIMPLE_EVALS_SYSTEM_MESSAGE, _eval_gpqa, _flag_comparability, _gpqa_sample, _load_gpqa_items,
    _permutations, _run_simple_evals_gpqa, merge_shards)

logger = logging.getLogger(__name__)


def _load_gpqa_samples(enable_thinking: bool, limit: Optional[int] = None,
                       n_repeats: int = N_REPEATS
                       ) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """GPQA Diamond ('Idavidrein/gpqa', 'gpqa_diamond', pinned revision) as simple-evals
    requests, one per (question, repeat) copy, each carrying the chat sampler's system message
    (the same in both modes)."""
    return [it["sample"] for it in _load_gpqa_items("gpqa_diamond", limit, n_repeats,
                                                     "gpqa_diamond", enable_thinking)]


def run_gpqa_diamond(model_name: str, base_url: str, concurrency: int, enable_thinking: bool = False, **kwargs) -> Dict[str, Any]:
    """Run native GPQA Diamond evaluation suite (simple-evals GPQAEval, 10 repeats)."""
    return _run_simple_evals_gpqa("gpqa_diamond", "gpqa_diamond", 198, model_name, base_url,
                                  concurrency, enable_thinking, **kwargs)
