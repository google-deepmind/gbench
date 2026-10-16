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

# Evaluation Suite: ifeval
# Description: Google IFEval - verifiable instruction-following constraints (strict prompt-level)

"""gbench native built-in runner for ifeval (Instruction Following).

IFEval (Zhou et al., 2023): 541 prompts, each carrying an `instruction_id_list` plus
per-instruction `kwargs`. Scoring is google-research `instruction_following_eval` itself,
vendored in `_ifeval_vendor` (commit e6890f85, checkers unchanged): every instruction is
built from its kwargs and checked with the upstream `check_following`, through upstream's
`test_instruction_following_strict` / `_loose`. The headline is prompt-level STRICT accuracy (a
prompt counts when the response follows EVERY instruction); instruction-level and loose
accuracies are reported alongside, as upstream's `print_report` does.

The checkers need `langdetect` (response language, English capital/lowercase) and nltk with
its punkt data (sentence counting, word tokenizing). If either is missing the suite stops
with an error; there is no heuristic substitute.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_IFEVAL_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.

Sharding: `--shard I/N` selects shard I of the prompts (round-robin in dataset order) before
the `--eval-limit` draw.
"""

import functools
import json
import logging
import random
import re
from typing import Any, Dict, List, Optional, Tuple
from .arc_agi import _for_base   # lays out an already-sharded sample list for run_eval_suite
from .base import apply_shard, run_eval_suite, strip_thinking_tags
from .sampling import stratified_sample
from .fc_common import base_suite_plan, run_attempts
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

PILLAR = "Instruction Following"
DOCS_URL = "docs/evals/ifeval.md"

_DATASET_ID = "google/IFEval"
#: HF dataset revision the loader reads (541 prompts).
_DATASET_REVISION = "966cd89545d6b6acfd7638bc708b98261ca58e84"
#: google-research commit 26d8ccd ("Fix an eval prompt", 2024-06-11) corrected key 2785 in
#: data/input_data.jsonl so the prompt asks for the 3 placeholders its kwargs check; the HF copy
#: still has the old text. With this applied the prompts equal upstream input_data.jsonl.
_UPSTREAM_PROMPT_FIXES = {
    2785: ("Your answer must also contain at least one placeholder (an example of a "
           "placeholder is [address]).",
           "Your answer must also contain at least 3 placeholders (an example of a "
           "placeholder is [address])."),
}


@functools.lru_cache(maxsize=1)
def _scorer() -> Any:
    """The vendored upstream evaluation library, with its dependencies checked.

    langdetect is seeded (DetectorFactory.seed = 0) so rescoring the same responses gives the
    same verdicts; upstream leaves it unseeded, which only matters for text whose language is
    ambiguous."""
    try:
        from langdetect import DetectorFactory
    except ImportError:
        raise infra_required(
            "ifeval", "IFEval's language checkers (response_language, english_capital, "
            "english_lowercase) need the `langdetect` package, which is not installed. "
            "Install it (`pip install langdetect`).", DOCS_URL)
    DetectorFactory.seed = 0
    try:
        from ._ifeval_vendor import evaluation_lib, instructions_util
        instructions_util.count_sentences("One. Two.")
        instructions_util.nltk.word_tokenize("One two.")
    except ImportError as e:
        raise infra_required("ifeval", f"IFEval's checkers need `nltk` ({e}). Install it "
                             "(`pip install nltk`).", DOCS_URL)
    except LookupError as e:
        raise infra_required(
            "ifeval", "IFEval's sentence and word counting needs nltk's punkt tokenizer data, "
            f"which is not downloaded ({str(e).strip().splitlines()[0]}). Run "
            "`python -m nltk.downloader punkt punkt_tab`.", DOCS_URL)
    return evaluation_lib


def _load_ifeval_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load IFEval prompts + instruction constraints from HF Hub; raises on load/schema failure."""
    try:
        from datasets import load_dataset
        ds = load_dataset(_DATASET_ID, split="train", revision=_DATASET_REVISION)
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for ifeval: {e}")
        raise RuntimeError(f"Could not load dataset for ifeval: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for ifeval returned empty rows")
    # --shard first, over the full ordered prompt list, then the --eval-limit draw.
    rows, _ = apply_shard(rows, "ifeval")
    # Stratified, not a contiguous head.
    rows = stratified_sample(rows, limit, None, seed="ifeval")

    samples = []
    for item in rows:
        prompt = item.get("prompt")
        rule_ids = item.get("instruction_id_list") or []
        kwargs_list = item.get("kwargs") or []
        if not prompt or not rule_ids:
            raise RuntimeError(
                "ifeval: unexpected schema (missing 'prompt'/'instruction_id_list'); "
                "refusing to fabricate sample data"
            )
        fix = _UPSTREAM_PROMPT_FIXES.get(item.get("key"))
        if fix is not None:
            if prompt.count(fix[0]) != 1:
                raise RuntimeError(f"ifeval: key {item.get('key')} no longer has the text the "
                                   "upstream correction applies to; re-check the dataset pin")
            prompt = prompt.replace(fix[0], fix[1])
        # The HF rows fill every kwarg name with null (one Arrow schema for all instructions);
        # upstream's jsonl lists only the kwargs an instruction takes.
        kwargs_clean = [{k: v for k, v in (kw or {}).items() if v is not None}
                        for kw in kwargs_list]
        gold = json.dumps({"key": item.get("key"), "ids": list(rule_ids),
                           "kwargs": kwargs_clean, "prompt": prompt})
        cat = rule_ids[0].split(":")[0] if rule_ids else "general"
        messages = [{"role": "user", "content": str(prompt)}]
        samples.append((messages, gold, {"category": cat}))

    logger.info(f"Loaded {len(samples)} ifeval samples.")
    return samples


def _follow_lists(gold_target: str, response: str) -> Tuple[List[bool], List[bool], List[str]]:
    """(strict per-instruction verdicts, loose verdicts, instruction ids) via upstream code.

    Where a kwarg is invalid, upstream's build_description draws a random value instead (keys
    1122 and 1129 ask for the frequency of the "letters" '#' and '!', so a random letter is
    counted). That draw is seeded per prompt here, restoring the global RNG afterwards, so the
    per-sample verdict and the report agree and a rescore is reproducible."""
    lib = _scorer()
    data = json.loads(gold_target)
    inp = lib.InputExample(key=data.get("key") or 0, instruction_id_list=list(data["ids"]),
                           prompt=data["prompt"], kwargs=list(data["kwargs"]))
    prompt_to_response = {inp.prompt: response}
    state = random.getstate()
    random.seed(f"ifeval:{inp.key}")
    try:
        strict = lib.test_instruction_following_strict(inp, prompt_to_response)
        random.seed(f"ifeval:{inp.key}")
        loose = lib.test_instruction_following_loose(inp, prompt_to_response)
    finally:
        random.setstate(state)
    return strict.follow_instruction_list, loose.follow_instruction_list, list(data["ids"])


def _eval_ifeval(response_text: str, gold_target: str) -> bool:
    """Prompt-level STRICT accuracy: the response must satisfy EVERY instruction."""
    if not response_text:
        return False
    strict, _, _ = _follow_lists(gold_target, response_text)
    return all(strict)


def _raw_content(text: Optional[str]) -> str:
    """The reply with thinking tags removed but NOT whitespace-trimmed (what upstream checks);
    base.strip_thinking_tags, which feeds the per-sample scorer, also trims."""
    if not text:
        return ""
    cleaned = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<reasoning>.*?</reasoning>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned if cleaned.strip() else text


def _ifeval_report(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The four upstream accuracies, per-instruction strict accuracy, and how many prompts
    would get a different strict verdict on the untrimmed reply.

    Over the scored prompts, as base's accuracy (the headline): a request that got no reply from
    the served endpoint is a harness error that base marks scoring_excluded, left out of every
    number here (None when no prompt is scored). An empty reply is the model's and follows no
    instruction."""
    from .base import is_scored
    p_strict = p_loose = i_total = i_strict = i_loose = n = trim_diffs = 0
    by_id: Dict[str, List[int]] = {}
    for t in traces:
        gold = t.get("gold_answer")
        if not gold or not is_scored(t):
            continue
        n += 1
        text = strip_thinking_tags(t.get("response_text"))
        if text:
            strict, loose, ids = _follow_lists(gold, text)
        else:
            ids = json.loads(gold)["ids"]
            strict = loose = [False] * len(ids)
        raw = _raw_content(t.get("response_text"))
        if raw != text and raw.strip():
            raw_strict, _, _ = _follow_lists(gold, raw)
            trim_diffs += int(all(raw_strict) != all(strict))
        p_strict += int(all(strict))
        p_loose += int(all(loose))
        i_total += len(ids)
        i_strict += sum(strict)
        i_loose += sum(loose)
        for iid, ok in zip(ids, strict):
            by_id.setdefault(iid, [0, 0])
            by_id[iid][0] += int(ok)
            by_id[iid][1] += 1

    def pct(a: int, b: int) -> Optional[float]:
        return round(a / b * 100.0, 2) if b else None

    return {
        "prompt_level_strict": pct(p_strict, n),
        "instruction_level_strict": pct(i_strict, i_total),
        "prompt_level_loose": pct(p_loose, n),
        "instruction_level_loose": pct(i_loose, i_total),
        "prompts": n,
        "instructions": i_total,
        "by_instruction_strict": {k: {"followed": v[0], "total": v[1], "accuracy": pct(*v)}
                                  for k, v in sorted(by_id.items())},
        "prompts_changed_by_whitespace_trim": trim_diffs,
    }


def run_ifeval(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute ifeval native built-in evaluation suite (prompt-level strict)."""
    _scorer()   # fail before any generation when a checker dependency is missing
    limit = kwargs.get("limit")
    samples = _load_ifeval_samples(limit=limit)
    limit_plan = base_suite_plan("ifeval", _for_base(samples), limit, run_attempts(kwargs))
    result = run_eval_suite(
        eval_name="ifeval",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=_for_base(samples),  # already this shard's prompts; base keeps exactly them
        eval_fn=_eval_ifeval,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=None,                  # already drawn by the loader, after the shard
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
    result["eval_limit_plan"] = limit_plan
    report = _ifeval_report(result.get("sample_traces") or [])
    result["ifeval_report"] = report
    result["metric"] = ("prompt-level strict accuracy (google-research instruction_following_eval "
                        "checkers); instruction-level and loose accuracies in ifeval_report")
    noncanon = []
    if limit:
        noncanon.append(f"--eval-limit subset ({len(samples)} of 541 prompts)")
    temp = result.get("temperature")
    if temp:
        noncanon.append(f"sampled at temperature {temp} (IFEval results are normally greedy)")
    # base's failed_requests counts the requests with no reply and the empty replies.
    empty = int(result.get("empty_responses") or 0)
    no_reply = max(sum(1 for t in result.get("sample_traces") or [] if t.get("harness_error")),
                   int(result.get("failed_requests") or 0) - empty)
    if no_reply:
        noncanon.append(f"{no_reply} request(s) got no reply from the served endpoint (failed, "
                        "timed out or over the context window: harness errors, left out of the "
                        "score)")
    if empty:
        noncanon.append(f"{empty} request(s) came back empty and count as not following their "
                        "instructions")
    cut_off = int(result.get("truncated_responses") or 0)
    if cut_off:
        noncanon.append(f"{cut_off} repl(ies) were cut off at the max_tokens budget "
                        f"(max_tokens_sent {result.get('max_tokens_sent')}) and were checked on "
                        "what arrived")
    if report["prompts_changed_by_whitespace_trim"]:
        noncanon.append(
            f"{report['prompts_changed_by_whitespace_trim']} prompt(s) score differently on the "
            "untrimmed reply (gbench trims leading/trailing whitespace before checking; upstream "
            "checks the raw reply)")
    result["leaderboard_comparable"] = not noncanon
    if noncanon:
        result["leaderboard_comparable_reason"] = "; ".join(noncanon)
    return result
