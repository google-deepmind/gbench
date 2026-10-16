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

"""Example Custom Evaluation Plugin for gbench.

This file demonstrates how any developer can build a custom, private domain evaluation
suite that runs inside gbench.

The plugin contract is small: expose a `run_<name>(model_name, base_url, concurrency,
enable_thinking=False, **kwargs)` function; gbench's loader discovers it from any file passed via
`--eval-plugins-dir` and registers `<name>` as a first-class `--evals` target. Build a list of
`(messages, gold, extra)` samples and hand them to the shared `run_eval_suite` helper with a scoring
function; it returns the standard gbench result dict.

Run it:
    gbench --evals-only --eval-plugins-dir examples/custom_evals --evals custom_qa \\
           --remote-endpoint http://127.0.0.1:8000/v1 --tokenizer <model> \\
           --max-output-tokens 16384

If you only need to score a JSONL file of prompts with simple string/number checks (no custom loading
or judge), you do NOT need a plugin: use the built-in `custom_jsonl` runner instead
(`--eval-custom-jsonl your.jsonl`; see sample_benchmark.jsonl here and docs/evals/plugins.md). This
plugin is the pattern to copy when you outgrow that (custom sample sources, an LLM judge, or a metric
the built-in scorer does not cover).
"""

from typing import Any, Dict, List, Optional, Tuple
from gbench.runners.eval_suites.base import run_eval_suite

# Optional: Set a custom capability pillar title in CLI results table
PILLAR = "7. CUSTOM DOMAIN KNOWLEDGE & SAFETY"


def _load_custom_qa_samples(
    enable_thinking: bool = False,
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load private domain QA samples (e.g. from local file, database, or API)."""
    raw_data = [
        {
            "question": "What is the primary law of thermodynamics concerning conservation of energy?",
            "answer": "First Law of Thermodynamics",
            "category": "Physics"
        },
        {
            "question": "In Python, which built-in function returns the unique identity integer of an object?",
            "answer": "id",
            "category": "Computer Science"
        },
        {
            "question": "What is the capital of Australia?",
            "answer": "Canberra",
            "category": "Geography"
        }
    ]

    samples = []
    for item in raw_data:
        q = item["question"]
        gold = item["answer"]
        cat = item["category"]

        if enable_thinking:
            prompt = f"Question: {q}\n\nThink step by step and provide your final concise answer."
        else:
            prompt = f"Question: {q}\n\nAnswer concisely in one short phrase."

        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, gold, {"category": cat}))

    if limit is not None:
        samples = samples[:limit]

    return samples


def _eval_custom_qa(response_text: str, gold_answer: str) -> bool:
    """Evaluation matching function."""
    if not response_text or not gold_answer:
        return False
    return gold_answer.strip().lower() in response_text.strip().lower()


def run_custom_qa(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Runner entrypoint called by gbench.
    
    Function name must start with 'run_<eval_name>'.
    """
    samples = _load_custom_qa_samples(enable_thinking=enable_thinking, limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="custom_qa",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_custom_qa,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
