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

"""Native AIME Olympiad Competition Math evaluation suite.

AIME 2026 as MathArena runs it: `MathArena/aime_2026` (AIME I and II, 30 problems, pinned
below), the AIME competition that eth-sri/matharena lists among its current final-answer
competitions, with MathArena's whole protocol (see matharena_common):

* the instruction of configs/competitions/aime/aime_2026.yaml ("Put your final answer within
  \\boxed{}." and "The answer is an integer between 0 and 999 inclusive."), a blank line, then
  the problem, as one user message, identical with and without --thinking;
* 4 runs per problem (MathArena's `--n` default); the headline `accuracy` is the mean over all
  runs, which is MathArena's score;
* MathArena's last-chance reprompt for every reply whose final message has no parseable
  `\\boxed{}` answer;
* grading by MathArena's own parser.py / grader.py (vendored verbatim), run in the local
  `gbench-matharena:b89f2f0a` image. The suite hard-errors (infra_required) if Docker or the image is
  missing, or if the image fails the grader self-check.

`leaderboard_comparable` is True only with 4 or more runs per problem, MathArena's sampling
(below), the full problem set and every response graded; the reason names anything else.
Older, pre-cutoff AIME years are in the `amc_aime` suite (a contaminated reference).

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
`--temperature`, or `GBENCH_AIME_TEMPERATURE` for this suite alone (it takes precedence),
overrides the temperature in either mode; the result records it under `official_sampling` (and
the fields sent under `recommended_sampling`), and any other value, any other sampling field, a
config's max_tokens not sent in full (the served window too small), or a reply (first or last-chance) cut off by a
limit other than the config's max_tokens (`replies_cut_off_by_budget`; for Gemma, any reply that
reached --max-output-tokens) is not leaderboard-comparable.
No LLM judge is involved (MathArena's parser grades), so gbench's judge temperature (pinned at
0.0) plays no part.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from . import hmmt as _hmmt
from . import matharena_common as MA
from .base import apply_shard, finalize_shard_fields, resolve_official_sampling, run_eval_suite
from .limit_plan import eval_limit_plan
from .sampling import stratified_sample, presharded_for_base
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

DOCS_URL = "docs/evals/aime.md"

_COMP = MA.COMPETITIONS["aime/aime_2026"]
_DATASET = _COMP.dataset
_REVISION = _COMP.revision
_COMPETITION = _COMP.name
_CATEGORY = "aime_2026"

#: MathArena's `--n` default: runs per problem, averaged into the reported accuracy.
_MATHARENA_RUNS = MA.OFFICIAL_RUNS


def _config() -> Dict[str, Any]:
    """The vendored MathArena competition config (read on use, blob-checked)."""
    return MA.competition_config(_COMP)


def _load_aime_samples(enable_thinking: bool = False,
                       limit: Optional[int] = None) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load the pinned MathArena AIME 2026 set. The prompt does not depend on
    `enable_thinking`: MathArena sends the same message either way."""
    try:
        from datasets import load_dataset
        rows = list(load_dataset(_DATASET, split="train", revision=_REVISION))
    except Exception as e:
        raise RuntimeError(f"aime: could not load {_DATASET}@{_REVISION[:8]}: {e}. "
                           f"See {DOCS_URL}") from e
    if not rows:
        raise RuntimeError(f"aime: {_DATASET}@{_REVISION[:8]} returned no rows")
    rows = sorted(rows, key=lambda r: int(r.get("problem_idx") or 0))
    # --shard first, over the full ordered set, then the --eval-limit draw within the shard.
    rows, _ = apply_shard(rows, "aime")
    rows = stratified_sample(rows, limit, None, seed="aime")

    config = _config()
    samples = []
    for item in rows:
        problem, answer = item.get("problem"), item.get("answer")
        if not problem or answer is None:
            raise RuntimeError("aime: unexpected dataset schema (missing 'problem'/'answer'); "
                               "refusing to fabricate sample data")
        messages = [{"role": "user", "content": MA.build_prompt(config, str(problem))}]
        # MathArena's gold is str(problem["answer"]) (runs.py); the column is an int.
        samples.append((messages, str(answer),
                        {"category": _CATEGORY, "problem_idx": item.get("problem_idx")}))
    logger.info(f"Loaded {len(samples)} AIME 2026 samples.")
    return samples


def _eval_aime(response_text: str, gold_answer: str) -> bool:
    """MathArena's grader on one final response, run in THIS interpreter.

    Not used by run_aime, which grades whole conversations (after the last-chance reprompt) in
    the gbench-matharena:b89f2f0a image. Kept for unit tests and diagnostics.
    """
    if not response_text:
        return False
    return MA.grade_final_response(response_text, str(gold_answer), _COMP.key)


def run_aime(model_name: str, base_url: str, concurrency: int, enable_thinking: bool = False, **kwargs) -> Dict[str, Any]:
    """Run AIME 2026 under MathArena's protocol (4 runs, last chance, MathArena's grader)."""
    ok, why = MA.check_prerequisites()
    if not ok:
        raise infra_required("aime", why, DOCS_URL)
    config = _config()
    broken = MA.self_check(config)
    if broken:
        raise infra_required("aime", broken, DOCS_URL)
    limit = kwargs.get("limit")
    samples = _load_aime_samples(enable_thinking, limit=limit)
    attempts = max(_MATHARENA_RUNS, int(kwargs.get("attempt_count") or 1))
    # --eval-limit N is N problems for the whole suite; MathArena's runs per problem are attempts.
    limit_plan = eval_limit_plan("aime", limit, len(samples), attempts=attempts)
    # MathArena samples each model with its own config's settings (or, when it ships none, the
    # served model's own generation_config.json values), never another model's.
    served = _hmmt.served_model_sampling("aime", model_name, DOCS_URL, enable_thinking)
    sampling = served.sampling
    temperature, _ = resolve_official_sampling("aime", sampling, thinking=enable_thinking)
    user_payload = dict(kwargs.get("extra_payload") or {})
    extra_payload = {**served.extra_fields, **user_payload} or None
    payload = MA.request_payload(extra_payload, sampling)
    stats: Dict[str, Any] = {}
    result = run_eval_suite(
        eval_name="aime",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=presharded_for_base(samples),  # this shard's problems, kept as is
        async_eval_fn=MA.make_scorer(
            _COMP, config, stats, eval_name="aime", model_name=model_name,
            base_url=base_url, concurrency=concurrency, thinking=enable_thinking,
            temperature=temperature, extra_payload=payload),
        declared_scoring_mode="deterministic",   # MathArena's parser, not an LLM judge
        thinking=enable_thinking,
        extra_payload=extra_payload,
        limit=limit,
        max_output_tokens=kwargs.get("max_output_tokens"),
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
    reasons += _hmmt.served_model_reasons("aime", result, served, user_payload, enable_thinking)
    result["leaderboard_comparable"] = not reasons
    if reasons:
        result["leaderboard_comparable_reason"] = "; ".join(reasons)
    # One shard of a --shard I/N run is a partial set: the shard reason (finalize_shard_fields).
    return finalize_shard_fields(result)
