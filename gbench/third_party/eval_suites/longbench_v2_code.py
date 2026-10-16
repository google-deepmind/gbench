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

"""Native LongBench-v2 code suite: the "Code Repository Understanding" domain of LongBench-v2.

`longbench_v2_code` runs the code domain of LongBench-v2 (Bai et al., 2024): 50 four-choice
questions over multi-file repositories whose context runs from ~100K characters to 16M. It is a
LongBench-v2 domain score, not a separate upstream benchmark.
`GBENCH_LONGBENCH_V2_CODE_ALL_DOMAINS=1` scores all 503 LongBench-v2 questions instead.
Renamed from `culer` (gbench's earlier name for the same domain); the old name still runs this
suite, with a deprecation warning, and the `GBENCH_CULER_*` variables are still read as
deprecated aliases.

The protocol is THUDM/LongBench `pred.py` (commit 2e00731f), in both of its settings, which
LongBench-v2 reports side by side (w/o CoT and w/ CoT). Both run by default, on the same
questions, and each is scored and reported on its own (`settings.non_cot`, `settings.cot`,
`headline_metrics`); the non-CoT setting is the headline (`accuracy`):
  * non-CoT (`python pred.py`): `prompts/0shot.txt` verbatim, filled with pred.py's `.replace`
    chain, one query with `max_new_tokens=128`, whose reply is scored;
  * w/ CoT (`python pred.py --cot`): `prompts/0shot_cot.txt` verbatim (it ends "Let's think
    step by step:"), one query with `max_new_tokens=1024`. Its reply, stripped, is the chain of
    thought: `prompts/0shot_cot_ans.txt` is filled with the same `.replace` chain and then
    `.replace('$COT$', response)`, and that answer query (`max_new_tokens=128`) is the reply
    scored. A CoT query that returns '' gets no answer query (pred.py's `if output == '':
    continue`).
Common to every query:
  * truncation: the whole prompt is tokenized with the served model's tokenizer and, when it is
    longer than `max_len`, the MIDDLE is dropped (`input_ids[:max_len//2] +
    input_ids[-max_len//2:]`) and the rest decoded back to text (pred.py's query_llm does this
    for every query, the CoT answer query included). Upstream sets `max_len` per model to the
    model's context window (README: "add your model path and its context window length to
    config/"; config/model2maxlen.json: 120,000 for every 128K-context model, 200,000 for the
    200K-context claude-3.5-sonnet-20241022). gbench takes it from the served window: 120,000
    for a window up to 131,072 tokens (upstream's value for 128K-context models); for a larger
    window, the window minus the largest official output cap of the two settings (1,024, the CoT
    query's) minus 64 tokens of chat template (the largest prompt the window holds), one max_len
    for every query of both settings, as pred.py uses one per model. The max_len sent is that value, or less when
    the window cannot hold it plus the query's actual output budget (a --thinking budget, or a
    window under 120,000 + budget + 64): the window minus the budget minus 64;
  * output budget: each query's `query_llm(..., max_new_tokens=...)` (128 non-CoT; 1024 for the
    CoT query and 128 for the CoT answer query) is declared as that query's official cap
    (base.OfficialSampling.max_tokens) and applied by base's token-limit policy: a no-think
    request is sent the cap exactly (`--max-output-tokens` is not applied, see
    `max_output_tokens_note`). A `--thinking` request is sent the run's budget instead
    (`--max-output-tokens`, else the thinking floor), since the caps were written for
    non-reasoning replies and the reasoning shares the budget; the result records that the
    official cap was not applied to reasoning (`official_sampling.max_tokens_applied: false`);
  * extraction: `extract_answer` verbatim ("The correct answer is (X)" or "... is X", else no
    answer, scored wrong), on the scored reply;
  * sampling: gbench's rule (base.resolve_temperature): every query at temperature 1.0 with
    thinking, 0.0 without, not upstream's 0.1 (`query_llm(..., temperature=0.1, ...)` for every
    query), which is recorded.

`leaderboard_comparable` is computed per setting from what the run actually did, and
`leaderboard_comparable_reason` names every difference from upstream: the code-domain slice, a
thinking run (upstream has no native-reasoning setting), prompts that were cut shorter than
upstream would cut them (a window too small for upstream's max_len plus the query's budget, or
a served window smaller than the model's own context window, read from its config's
max_position_embeddings when the config is in the local Hugging Face cache), a no-think
request sent a max_tokens other than its official cap, a temperature other than 0.1, questions
whose request got no reply from the served endpoint (failed, timed out or over the context
window: harness errors, left out of the score, counted and named in `status_reason`) or still
came back empty or null after NULL_REPLY_REASKS re-asks (scored wrong: pred.py writes no
prediction for a question whose reply is empty or null, and a resumed run asks it again, which
gbench reproduces with upstream_reask_relay), answer queries asked again with the same CoT, and
an --eval-limit subset. The
top-level flag is true only when both settings are comparable; its reason lists each setting's
differences, naming the setting when only one has it. A full, all-domain, no-think run at
temperature 0.1, served at the model's context window, with every request answered, is
comparable in both settings when that window is at least 121,088 tokens (120,000 + 1,024 + 64);
from 120,192 tokens the non-CoT setting is.

Sampling: gbench's rule, as for every suite (base.resolve_temperature): 1.0 for `--thinking`
runs, 0.0 otherwise; LongBench-v2's 0.1 is recorded (base.OfficialSampling; the result records
`official_sampling`). Override it for a whole run
with `--temperature`, or for this suite alone with `GBENCH_LONGBENCH_V2_CODE_TEMPERATURE` (the
pre-rename `GBENCH_CULER_TEMPERATURE` is still read as a deprecated alias), which takes
precedence over both. LLM-judge grading is pinned at 0.0 and is not affected by either (this
suite has no judge).

Sharding: `--shard I/N` selects shard I of the questions (round-robin in dataset order, after
the domain filter) before the `--eval-limit` draw.

`--eval-limit N` is N samples for the whole suite, a (setting, question) pair being one sample:
the N are dealt over the two settings (sampling.allocate_limit), each setting drawing its share
stratified by domain with the same seed, so the questions of the smaller share are among the
larger's; a setting dealt 0 is not run. Without a limit both settings run every question.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from .arc_agi import _for_base   # lays out an already-sharded sample list for run_eval_suite
from .base import (OfficialSampling, answer_in_thinking_fields, apply_shard,
                   finalize_shard_fields, get_run_knob, is_scored, resolve_token_budget,
                   run_eval_suite, strip_thinking_tags, suite_env)
from .limit_plan import base_attempts, eval_limit_plan
from .sampling import allocate_limit, limit_dataset
from .swebench_common import (infra_required, reask_divergence_reasons, reask_summary,
                              reply_content_is_null_or_empty, upstream_reask_relay)

logger = logging.getLogger(__name__)

#: The registry name (renamed from `culer`; the stratified-sample seed keeps the old string).
EVAL_NAME = "longbench_v2_code"

#: pred.py writes no prediction for a question whose reply content is '' (`if output == '':
#: continue`) or None (`output.strip()` raises and ends the worker), and a resumed pred.py run
#: (it skips the questions already in its output file) asks it again. gbench re-asks such a
#: reply, the same request, up to this many times (swebench_common.upstream_reask_relay), so a
#: run ends; a reply still null or empty after them is scored wrong and makes the setting
#: non-comparable. (query_llm's own `tries < 5` loop covers request exceptions, which base's
#: transport retries and harness-error accounting cover.)
NULL_REPLY_REASKS = 5
#: pred.py's rule, as the comparability reasons name it (reask_divergence_reasons)
_PRED_PY_UPSTREAM = ("a resumed pred.py run asks a question with an empty or null output again "
                     "without limit")
PRED_PY_REASK_RULE = ("LongBench pred.py: a question whose reply content is '' or None gets no "
                      "prediction and a resumed run asks it again (the same request)")
DOCS_URL = f"docs/evals/{EVAL_NAME}.md"

_DATASET_ID = "zai-org/LongBench-v2"
#: HF dataset revision the loader reads (503 questions; 50 in Code Repository Understanding).
_DATASET_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"

# THUDM/LongBench prompts/0shot.txt at commit 2e00731f, verbatim (the file has no trailing newline).
_TEMPLATE_0SHOT = (
    "Please read the following text and answer the question below.\n"
    "\n"
    "<text>\n"
    "$DOC$\n"
    "</text>\n"
    "\n"
    "What is the correct answer to this question: $Q$\n"
    "Choices:\n"
    "(A) $C_A$\n"
    "(B) $C_B$\n"
    "(C) $C_C$\n"
    "(D) $C_D$\n"
    "\n"
    'Format your response as follows: "The correct answer is (insert answer here)".'
)

# THUDM/LongBench prompts/0shot_cot.txt at commit 2e00731f, verbatim (no trailing newline; the
# apostrophe of "Let's" is U+2019 in the file).
_TEMPLATE_0SHOT_COT = (
    "Please read the following text and answer the questions below.\n"
    "\n"
    "<text>\n"
    "$DOC$\n"
    "</text>\n"
    "\n"
    "What is the correct answer to this question: $Q$\n"
    "Choices:\n"
    "(A) $C_A$\n"
    "(B) $C_B$\n"
    "(C) $C_C$\n"
    "(D) $C_D$\n"
    "\n"
    "Let\u2019s think step by step:"
)

# THUDM/LongBench prompts/0shot_cot_ans.txt at commit 2e00731f, verbatim (no trailing newline;
# U+2019 as above). It has no $DOC$: the answer query does not repeat the document.
_TEMPLATE_0SHOT_COT_ANS = (
    "Please read the following text and answer the questions below.\n"
    "\n"
    "The text is too long and omitted here.\n"
    "\n"
    "What is the correct answer to this question: $Q$\n"
    "Choices:\n"
    "(A) $C_A$\n"
    "(B) $C_B$\n"
    "(C) $C_C$\n"
    "(D) $C_D$\n"
    "\n"
    "Let\u2019s think step by step: $COT$\n"
    "\n"
    "Based on the above, what is the single, most likely answer choice? Format your response as "
    'follows: "The correct answer is (insert answer here)".'
)

#: pred.py's `temperature=0.1` (query_llm, for every query).
_UPSTREAM_TEMPERATURE = 0.1
#: pred.py non-CoT `query_llm(..., max_new_tokens=128)`: the answer's max_tokens.
_UPSTREAM_MAX_NEW_TOKENS = 128
#: pred.py --cot: the CoT query's `max_new_tokens=1024`; its answer query sends 128.
_UPSTREAM_COT_MAX_NEW_TOKENS = 1024
#: The official sampling, cap included; base's token-limit policy sends the 128 exactly on a
#: no-think request and not on a --thinking one (it was written for a non-reasoning answer).
OFFICIAL_SAMPLING = OfficialSampling(
    reference=("THUDM/LongBench@2e00731f pred.py query_llm(..., temperature=0.1, "
               "max_new_tokens=128) (non-CoT)"),
    temperature=_UPSTREAM_TEMPERATURE, max_tokens=_UPSTREAM_MAX_NEW_TOKENS)
#: The w/ CoT setting's first query (0shot_cot.txt), under the same policy.
OFFICIAL_SAMPLING_COT = OfficialSampling(
    reference=("THUDM/LongBench@2e00731f pred.py --cot: query_llm(0shot_cot.txt prompt, ..., "
               "temperature=0.1, max_new_tokens=1024) (w/ CoT, CoT query)"),
    temperature=_UPSTREAM_TEMPERATURE, max_tokens=_UPSTREAM_COT_MAX_NEW_TOKENS)
#: The w/ CoT setting's answer query (0shot_cot_ans.txt with the CoT), under the same policy.
OFFICIAL_SAMPLING_COT_ANSWER = OfficialSampling(
    reference=("THUDM/LongBench@2e00731f pred.py --cot: query_llm(0shot_cot_ans.txt prompt, "
               "..., temperature=0.1, max_new_tokens=128) (w/ CoT, answer query)"),
    temperature=_UPSTREAM_TEMPERATURE, max_tokens=_UPSTREAM_MAX_NEW_TOKENS)
#: config/model2maxlen.json: the prompt budget upstream gives its 128K-context models.
_UPSTREAM_MAX_LEN_128K = 120000
#: The largest window upstream's 120,000 entry stands for (its 128K-context models: 131,072 or
#: 128,000 tokens). A larger window gets max_len from the window itself (_upstream_max_len).
_UPSTREAM_128K_WINDOW = 131072
#: How upstream_max_prompt_tokens is derived, recorded in each setting's report.
UPSTREAM_MAX_LEN_RULE = (
    "pred.py max_len = the model's context window (README; config/model2maxlen.json: 120,000 "
    "for 128K-context models, 200,000 for the 200K claude-3.5-sonnet; one value per model for "
    "every query): 120,000 for a served window up to 131,072 tokens, else the served window "
    "minus the largest official output cap of the two settings (1,024, the CoT query's) minus 64 "
    "tokens of chat template, the same for the non-CoT and the w/ CoT setting")
#: Room for the chat template around the user message and for re-tokenization drift after the
#: truncated ids are decoded back to text.
_TEMPLATE_SLACK_TOKENS = 64
_MIN_PROMPT_TOKENS = 4096

#: LongBench-v2's settings (pred.py without and with --cot), in the order they run. The first is
#: the headline.
SETTINGS = ("non_cot", "cot")
HEADLINE_SETTING = "non_cot"
_SETTING_LABELS = {"non_cot": "non-CoT", "cot": "w/ CoT"}
#: The query of each setting that carries the long document: its budget sets the prompt's max_len.
_LONG_QUERY_SAMPLING = {"non_cot": OFFICIAL_SAMPLING, "cot": OFFICIAL_SAMPLING_COT}
#: Sample-meta key of the CoT answer prompt (0shot_cot_ans.txt filled up to $COT$). Not a request
#: field, so base never sends it.
_COT_ANSWER_PROMPT_KEY = "cot_answer_prompt"


def _fill_prompt(item: Dict[str, Any], template: str = _TEMPLATE_0SHOT) -> str:
    """pred.py's template fill, in its order (so a `$Q$` inside the document is replaced too)."""
    return (template
            .replace("$DOC$", str(item.get("context") or "").strip())
            .replace("$Q$", str(item.get("question") or "").strip())
            .replace("$C_A$", str(item.get("choice_A") or "").strip())
            .replace("$C_B$", str(item.get("choice_B") or "").strip())
            .replace("$C_C$", str(item.get("choice_C") or "").strip())
            .replace("$C_D$", str(item.get("choice_D") or "").strip()))


def _cot_answer_prompt(template_filled: str, cot: str) -> str:
    """pred.py --cot: the answer prompt is the 0shot_cot_ans.txt fill followed by
    `.replace('$COT$', response)`, response being the CoT query's reply, stripped."""
    return template_filled.replace("$COT$", cot)


def _middle_truncate(prompt: str, tokenizer: Any, max_len: int) -> Tuple[str, int, bool]:
    """pred.py's `query_llm` truncation. Returns (prompt, token count before, truncated?)."""
    input_ids = tokenizer.encode(prompt)
    n = len(input_ids)
    if n > max_len:
        input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
        return prompt, n, True
    return prompt, n, False


def _all_domains() -> bool:
    """`GBENCH_LONGBENCH_V2_CODE_ALL_DOMAINS=1` (legacy aliases `GBENCH_CULER_ALL_DOMAINS`,
    `CULER_ALL_DOMAINS`) scores every domain, i.e. a full LongBench-v2 run."""
    return bool(suite_env("GBENCH_LONGBENCH_V2_CODE_ALL_DOMAINS", "GBENCH_CULER_ALL_DOMAINS",
                          "CULER_ALL_DOMAINS"))


def _load_longbench_v2_code_samples(
    limit: Optional[int] = None,
    tokenizer: Any = None,
    max_prompt_tokens: Optional[int] = None,
    cot: bool = False,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """LongBench-v2 code-domain questions as 0shot.txt prompts (0shot_cot.txt with `cot`),
    middle-truncated to `max_prompt_tokens` under `tokenizer` (the runner always passes both).
    With `cot` each sample's meta also carries its 0shot_cot_ans.txt fill, up to `$COT$`."""
    from datasets import load_dataset

    ds = load_dataset(_DATASET_ID, split="train", revision=_DATASET_REVISION)

    # The CODE slice of LongBench-v2, unless every domain was asked for.
    if not _all_domains():
        def _is_code(row):
            return "code" in str(row.get("domain") or "").lower()
        before = len(ds)
        ds = ds.filter(_is_code) if hasattr(ds, "filter") else [r for r in ds if _is_code(r)]
        logger.info("[%s] filtered LongBench-v2 to the code domain: %d/%d rows (set "
                    "GBENCH_LONGBENCH_V2_CODE_ALL_DOMAINS=1 to score every domain)", EVAL_NAME,
                    len(ds), before)

    # --shard first (by row index, so the long contexts are not read here), then the
    # --eval-limit draw.
    keep, shard_label = apply_shard(list(range(len(ds))), EVAL_NAME)
    if shard_label:
        ds = ds.select(keep) if hasattr(ds, "select") else [ds[i] for i in keep]
    # Stratified, not a contiguous head.
    ds = limit_dataset(ds, limit, "domain", seed="culer")         # pre-rename seed
    raw_samples = list(ds)
    logger.info(f"Loaded {len(raw_samples)} LongBench-v2 samples from HF Hub ('{_DATASET_ID}')"
                f" for the {_SETTING_LABELS['cot' if cot else 'non_cot']} setting.")

    samples = []
    for item in raw_samples:
        prompt = _fill_prompt(item, _TEMPLATE_0SHOT_COT if cot else _TEMPLATE_0SHOT)
        n_tokens, truncated = None, False
        if tokenizer is not None and max_prompt_tokens:
            prompt, n_tokens, truncated = _middle_truncate(prompt, tokenizer, max_prompt_tokens)
        messages = [{"role": "user", "content": prompt}]
        # No per-sample max_tokens: upstream's caps are the settings' OfficialSampling.max_tokens,
        # which base's token-limit policy applies per mode.
        meta = {
            "category": item.get("domain", "code"),
            "difficulty": item.get("difficulty"),
            "length": item.get("length"),
            "prompt_tokens_before_truncation": n_tokens,
            "prompt_truncated": truncated,
        }
        if cot:
            meta[_COT_ANSWER_PROMPT_KEY] = _fill_prompt(item, _TEMPLATE_0SHOT_COT_ANS)
        samples.append((messages, str(item.get("answer", "")).strip().upper(), meta))
    return samples


def _extract_answer(response: str) -> Optional[str]:
    """LongBench-v2 pred.py `extract_answer`, verbatim."""
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None


def _eval_longbench_v2_code(response_text: str, gold_answer: str) -> bool:
    """pred.py: `pred = extract_answer(response.strip())`, `judge = pred == answer`."""
    if not response_text or not gold_answer:
        return False
    return _extract_answer(response_text.strip()) == gold_answer.strip().upper()


def _cot_query_is_not_scored(response_text: str, gold_answer: str) -> bool:
    """The CoT query's reply is the chain of thought for the answer query; pred.py scores only
    the answer query's reply."""
    return False


def _output_budget(thinking: bool, official: OfficialSampling = OFFICIAL_SAMPLING) -> int:
    """The max_tokens a request under `official` is sent, as base.run_eval_suite resolves it under
    the token-limit policy (resolve_token_budget): upstream's cap (128; 1024 for the CoT query)
    without thinking; with it, the run's --max-output-tokens, else the thinking floor."""
    return int(resolve_token_budget(EVAL_NAME, official, thinking=thinking).max_tokens)


def _upstream_max_len(window: int) -> int:
    """pred.py's `max_len` for a model with a `window`-token context: upstream sets it to the
    model's context window (README), 120,000 for every 128K-context model and 200,000 for the
    200K claude-3.5-sonnet (config/model2maxlen.json), one value per model that every query of
    both settings is truncated to. So 120,000 up to a 131,072-token window; above it, the largest
    prompt the window holds with the largest official output cap of the two settings (the CoT
    query's 1,024) and the template slack, for both settings alike."""
    if int(window) <= _UPSTREAM_128K_WINDOW:
        return _UPSTREAM_MAX_LEN_128K
    cap = max(int(s.max_tokens) for s in _LONG_QUERY_SAMPLING.values())
    return int(window) - cap - _TEMPLATE_SLACK_TOKENS


def _prompt_budget(window: int, budget: int, upstream: Optional[int] = None) -> int:
    """The `max_len` this run truncates to: `upstream` (default _upstream_max_len(window)) when the window holds it plus the query's output budget and the template slack,
    else what the window leaves (window - budget - 64)."""
    if upstream is None:
        upstream = _upstream_max_len(window)
    return min(int(window) - budget - _TEMPLATE_SLACK_TOKENS, int(upstream))


def _model_context_window(name: str) -> Tuple[Optional[int], str]:
    """The model's own context window (its config's max_position_embeddings, or its
    text_config's), read from the local Hugging Face cache only, and where it came from; None
    when no local config gives one (the served window then stands for it)."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(name, local_files_only=True)
    except Exception as e:                                              # noqa: BLE001
        return None, f"unknown (no local config for {name!r}: {type(e).__name__})"
    for part in (cfg, getattr(cfg, "text_config", None)):
        n = getattr(part, "max_position_embeddings", None) if part is not None else None
        if isinstance(n, int) and n > 0:
            return n, f"{name} config max_position_embeddings"
    return None, f"unknown ({name} config has no max_position_embeddings)"


def _load_tokenizer(name: str) -> Any:
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(name)
    except Exception as e:
        raise infra_required(
            EVAL_NAME,
            f"LongBench-v2 truncates each prompt by tokens with the served model's tokenizer, "
            f"and tokenizer {name!r} could not be loaded ({e}). Pass --tokenizer <HF id of the "
            f"served model> (and make sure `transformers` is installed).", DOCS_URL)


def _breakdown(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Accuracy by difficulty and length, the columns LongBench-v2's result.py reports, over the
    scored questions (a request with no reply from the endpoint is a harness error, left out)."""
    out: Dict[str, Any] = {}
    for key, values in (("difficulty", ("easy", "hard")), ("length", ("short", "medium", "long"))):
        for v in values:
            rows = [t for t in traces
                    if (t.get("extra_payload") or {}).get(key) == v and is_scored(t)]
            if rows:
                ok = sum(1 for t in rows if t.get("is_correct"))
                out[v] = {"correct": ok, "total": len(rows),
                          "accuracy": round(ok / len(rows) * 100.0, 2)}
    return out


def _window_override() -> Optional[int]:
    """`GBENCH_LONGBENCH_V2_CODE_WINDOW` (legacy alias `GBENCH_CULER_WINDOW`), in tokens."""
    raw = (suite_env("GBENCH_LONGBENCH_V2_CODE_WINDOW", "GBENCH_CULER_WINDOW") or "").strip()
    try:
        return int(raw) if raw else None
    except ValueError:
        raise infra_required(EVAL_NAME, f"GBENCH_LONGBENCH_V2_CODE_WINDOW={raw!r} is not a "
                             "number of tokens.", DOCS_URL)


def _shorter_than_upstream(metas: List[Dict[str, Any]], max_len: int,
                           upstream: int = _UPSTREAM_MAX_LEN_128K) -> int:
    """Prompts this run cut shorter than upstream's truncation to `upstream` tokens would have
    left them: only possible when max_len is below `upstream`, for a prompt longer than
    max_len."""
    if max_len >= upstream:
        return 0
    return sum(1 for m in metas
               if m.get("prompt_truncated")
               and int(m.get("prompt_tokens_before_truncation") or 0) > max_len)


def _truncation_reasons(metas: List[Dict[str, Any]], max_len: int, upstream: int, window: int,
                        budget: int, model_window: Optional[int], what: str = "prompt(s)"
                        ) -> List[str]:
    """The comparability reasons of one query's truncation: prompts cut shorter than upstream's
    max_len for the served window (the window cannot hold it plus the output budget), and
    prompts cut although the model's own context window is larger than the served one."""
    out = []
    shorter = _shorter_than_upstream(metas, max_len, upstream)
    if shorter:
        out.append(f"{shorter} {what} middle-truncated to {max_len:,} tokens, shorter than "
                   f"upstream's max_len {upstream:,} for a {window:,}-token window (the window "
                   f"minus the {budget:,}-token output budget cannot hold {upstream:,})")
    if model_window and model_window > window:
        cut = sum(1 for m in metas if m.get("prompt_truncated"))
        if cut:
            out.append(f"{cut} {what} middle-truncated to {max_len:,} tokens on a served "
                       f"{window:,}-token window, smaller than the model's {model_window:,}-token "
                       "context window (LongBench-v2 serves a model at its context window and "
                       "truncates at it)")
    return out


def _no_extractable_answer(traces: List[Dict[str, Any]]) -> int:
    """Scored replies with no "The correct answer is" statement (empty ones included)."""
    return sum(1 for t in traces if is_scored(t)
               and _extract_answer(strip_thinking_tags(t.get("response_text"))) is None)


def _category_accuracy(traces: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_cat: Dict[str, List[bool]] = {}
    for t in traces:
        if is_scored(t):
            by_cat.setdefault(t.get("category") or "unknown", []).append(bool(t.get("is_correct")))
    return {cat: {"correct": sum(v), "total": len(v),
                  "accuracy": round(100.0 * sum(v) / len(v), 2)}
            for cat, v in sorted(by_cat.items())}


def _other_caps(traces: List[Dict[str, Any]], cap: int) -> List[int]:
    """max_tokens values no-think requests were sent other than the official cap (the context
    clamp lowers the cap when the prompt leaves less room in the window)."""
    return sorted({t.get("max_tokens_sent") for t in traces
                   if t.get("max_tokens_sent") not in (None, cap)})


def _reasons(all_domains: bool, own: List[str], temperature: Any, unanswered: Optional[str],
             limit: Optional[int], n_questions: int) -> List[str]:
    """A setting's differences from upstream, in a fixed order: the domain slice, the setting's
    own (thinking, caps, truncation), the temperature, unanswered questions, --eval-limit."""
    reasons = []
    if not all_domains:
        reasons.append("Code Repository Understanding domain only (50 of 503 LongBench-v2 "
                       "questions), not the overall LongBench-v2 score")
    reasons += own
    if temperature is None or abs(float(temperature) - _UPSTREAM_TEMPERATURE) > 1e-9:
        reasons.append(f"temperature {temperature} (LongBench-v2 uses {_UPSTREAM_TEMPERATURE})")
    if unanswered:
        reasons.append(unanswered)
    if limit:
        reasons.append(f"--eval-limit subset ({n_questions} question(s))")
    return reasons


def _set_comparable(result: Dict[str, Any], reasons: List[str]) -> None:
    result["leaderboard_comparable"] = not reasons
    if reasons:
        result["leaderboard_comparable_reason"] = "; ".join(reasons)
    else:
        result.pop("leaderboard_comparable_reason", None)
    # One shard of a --shard I/N run (`shard`, recorded by base) is a partial set: the shard
    # reason is added after the rewrite above (finalize_shard_fields).
    finalize_shard_fields(result)


def _run_non_cot(run_kwargs: Dict[str, Any], samples: List[Any], *, window: int, budget: int,
                 max_len: int, tok_name: str, thinking: bool, all_domains: bool,
                 limit: Optional[int], upstream: int = _UPSTREAM_MAX_LEN_128K,
                 model_window: Optional[int] = None,
                 model_window_source: Optional[str] = None) -> Dict[str, Any]:
    """pred.py (no --cot): one 0shot.txt query per question, its reply scored."""
    with upstream_reask_relay(EVAL_NAME, run_kwargs["base_url"], reply_content_is_null_or_empty,
                              NULL_REPLY_REASKS) as relay:
        result = run_eval_suite(
            samples=_for_base(samples),  # already this shard's questions; base keeps them all
            eval_fn=_eval_longbench_v2_code,
            official_sampling=OFFICIAL_SAMPLING,   # its max_tokens=128: base's token-limit policy
            **{**run_kwargs, "base_url": relay.url})
    result["upstream_reask"] = reask_summary(relay, PRED_PY_REASK_RULE)
    traces = result.get("sample_traces") or []
    metas = [m for _, _, m in samples]
    shorter = _shorter_than_upstream(metas, max_len, upstream)
    result["longbench_v2_code_report"] = {
        "dataset": f"{_DATASET_ID}@{_DATASET_REVISION[:8]}",
        "scope": "all LongBench-v2 domains" if all_domains else "Code Repository Understanding",
        "setting": "non-CoT (pred.py, prompts/0shot.txt)",
        "window_tokens": window,
        "model_context_window": model_window,
        "model_context_window_source": model_window_source,
        "output_budget_tokens": budget,
        "max_prompt_tokens": max_len,
        "upstream_max_prompt_tokens": upstream,
        "upstream_max_len_rule": UPSTREAM_MAX_LEN_RULE,
        "tokenizer": tok_name,
        "prompts_truncated": sum(1 for m in metas if m.get("prompt_truncated")),
        "prompts_cut_shorter_than_upstream": shorter,
        "no_extractable_answer": _no_extractable_answer(traces),
        "by_difficulty_and_length": _breakdown(traces),
    }

    own = []
    if thinking:
        own.append(f"--thinking run: LongBench-v2 has no native-reasoning setting, and each "
                   f"request was sent max_tokens={budget} for the reasoning and the answer, as "
                   f"the official max_tokens={_UPSTREAM_MAX_NEW_TOKENS} is not applied to "
                   "reasoning")
    else:
        other = _other_caps(traces, _UPSTREAM_MAX_NEW_TOKENS)
        if other:
            own.append(f"some requests were sent max_tokens {other} (LongBench-v2 sends "
                       f"{_UPSTREAM_MAX_NEW_TOKENS})")
    own += _truncation_reasons(metas, max_len, upstream, window, budget, model_window)
    own += reask_divergence_reasons(result["upstream_reask"], "empty or null content",
                                    "wrong", _PRED_PY_UPSTREAM, exhausted_reason=False)
    # A request with no reply from the endpoint is a harness error, left out of the score by
    # base; an empty reply is the model's and is scored wrong. base's failed_requests counts
    # both (this suite has no judge).
    empty = int(result.get("empty_responses") or 0)
    no_reply = max(sum(1 for t in traces if t.get("harness_error")),
                   int(result.get("failed_requests") or 0) - empty)
    _set_comparable(result, _reasons(
        all_domains, own, result.get("temperature"),
        _unanswered_reason(no_reply, empty, "request(s)",
                           in_thinking=int(result.get("answer_in_thinking") or 0),
                           still_empty=int(result["upstream_reask"].get("reasks_exhausted")
                                           or 0),
                           cut_short=int(result["upstream_reask"].get("reasks_cut_short") or 0)),
        limit, len(samples)))
    return result


def _unanswered_reason(no_reply: int, empty: int, what: str, in_thinking: int = 0,
                       still_empty: Optional[int] = None, cut_short: int = 0) -> Optional[str]:
    """The comparability reason for questions without an answer, or None. pred.py writes no
    prediction for a question whose request fails five times (query_llm returns '') or whose
    reply is empty or null (a reply with its answer inside the thinking channel has null
    content), and a resumed run asks it again. gbench leaves out the ones with no reply (harness
    errors). With the re-ask relay (`still_empty` given: upstream_reask.reasks_exhausted, of
    which `cut_short` ended before the limit, reasks_cut_short) an empty or null reply the model
    ended itself was asked again up to NULL_REPLY_REASKS times, so only the ones still empty
    after them differ from upstream (scored wrong; a reply cut off at the budget is never
    re-asked, named apart by reask_divergence_reasons); without it (`still_empty` None) every
    empty reply and answer-in-thinking reply was scored wrong."""
    parts = []
    if no_reply:
        parts.append(f"{no_reply} {what} got no reply from the served endpoint (failed, timed out "
                     "or did not fit the window: harness errors, left out of the score)")
    if still_empty is not None:
        if still_empty:
            how = f"after up to {NULL_REPLY_REASKS} re-asks"
            if cut_short:
                how += (f"; {cut_short} of them cut short before the limit, at the client's "
                        "request deadline or on a re-ask the endpoint did not answer")
            parts.append(f"{still_empty} {what} still came back with empty or null content "
                         f"when gbench stopped re-asking ({how}) and were scored wrong")
    else:
        if empty:
            parts.append(f"{empty} {what} came back empty and were scored wrong")
        if in_thinking:
            parts.append(f"{in_thinking} {what} stopped with the answer inside the thinking "
                         "channel (no content) and were scored wrong")
    if not parts:
        return None
    return "; ".join(parts) + (" (pred.py writes no prediction for such a question: query_llm "
                               "returns '' after 5 failed tries, and get_pred skips an empty "
                               "output; a resumed run asks it again)")


def _cot_answer_samples(
    turn1_traces: List[Dict[str, Any]], tokenizer: Any, max_len: int,
) -> Tuple[List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]], List[Dict[str, Any]]]:
    """The answer query for every CoT reply, plus the CoT-query traces that get none.

    pred.py --cot: `if output == '': continue` (a failed or empty CoT query leaves the question
    out); otherwise `response = output.strip()` goes into 0shot_cot_ans.txt at `$COT$`, and
    query_llm truncates that prompt like any other. The CoT is the reply's content, without a
    reasoning channel (the chat completion's `message.content`, as pred.py reads it).
    """
    samples, no_cot = [], []
    for tr in turn1_traces:
        meta = dict(tr.get("extra_payload") or {})
        template = meta.pop(_COT_ANSWER_PROMPT_KEY, None)
        if template is None:
            raise RuntimeError(f"[{EVAL_NAME}] a CoT query trace carries no answer prompt "
                               f"({_COT_ANSWER_PROMPT_KEY} in its sample meta)")
        text = tr.get("response_text")
        if text is None or text == "":
            no_cot.append(tr)
            continue
        cot = strip_thinking_tags(text)
        prompt, n_tokens, truncated = _middle_truncate(_cot_answer_prompt(template, cot),
                                                       tokenizer, max_len)
        meta.update({
            "cot_query_sample_idx": tr.get("source_sample_idx"),
            "cot_query_attempt_idx": tr.get("attempt_idx"),
            "cot_query_status": tr.get("status"),
            "cot_query_finish_reason": tr.get("finish_reason"),
            "cot_query_completion_tokens": tr.get("completion_tokens"),
            "cot_query_max_tokens_sent": tr.get("max_tokens_sent"),
            "long_prompt_tokens_before_truncation": meta.get("prompt_tokens_before_truncation"),
            "long_prompt_truncated": meta.get("prompt_truncated"),
            "prompt_tokens_before_truncation": n_tokens,
            "prompt_truncated": truncated,
        })
        samples.append(([{"role": "user", "content": prompt}], tr.get("gold_answer"), meta))
    return samples, no_cot


#: Per-query fields kept in the w/ CoT entry, for its CoT query and its answer query.
_QUERY_FIELDS = ("status", "failed_requests", "empty_responses", "truncated_responses",
                 "over_context_prompts", "timed_out_requests", "looping_responses",
                 "max_tokens_sent", "official_max_tokens_sent", "replies_at_official_max_tokens",
                 "response_health", "max_output_tokens_note", "duration_s")
#: Counters that add up over the two queries.
_SUMMED_COUNTERS = ("failed_requests", "empty_responses", "truncated_responses",
                    "over_context_prompts", "timed_out_requests", "looping_responses",
                    "replies_at_official_max_tokens", "answer_in_thinking")


def _query_summary(turn: Dict[str, Any]) -> Dict[str, Any]:
    """One query's request counts (base's result for it, without its traces)."""
    return {"requests": turn.get("total_questions"), **{k: turn.get(k) for k in _QUERY_FIELDS}}


def _run_cot(run_kwargs: Dict[str, Any], samples: List[Any], *, tokenizer: Any, window: int,
             budget: int, max_len: int, tok_name: str, thinking: bool, all_domains: bool,
             limit: Optional[int], upstream: int = _UPSTREAM_MAX_LEN_128K,
             model_window: Optional[int] = None,
             model_window_source: Optional[str] = None) -> Dict[str, Any]:
    """pred.py --cot: the 0shot_cot.txt query, then the 0shot_cot_ans.txt query with its reply."""
    with upstream_reask_relay(EVAL_NAME, run_kwargs["base_url"], reply_content_is_null_or_empty,
                              NULL_REPLY_REASKS) as relay1:
        turn1 = run_eval_suite(samples=_for_base(samples), eval_fn=_cot_query_is_not_scored,
                               official_sampling=OFFICIAL_SAMPLING_COT,
                               **{**run_kwargs, "base_url": relay1.url})
    reask = {"cot_query": reask_summary(relay1, PRED_PY_REASK_RULE)}
    answer_budget = _output_budget(thinking, OFFICIAL_SAMPLING_COT_ANSWER)
    # pred.py truncates both queries of the CoT setting to the model's one max_len.
    answer_max_len = _prompt_budget(window, answer_budget, upstream)
    answer_samples, no_cot = _cot_answer_samples(turn1.get("sample_traces") or [], tokenizer,
                                                 answer_max_len)
    turn2: Optional[Dict[str, Any]] = None
    if answer_samples:
        # One answer query per CoT reply (each --attempt-count attempt already has its own CoT).
        with upstream_reask_relay(EVAL_NAME, run_kwargs["base_url"],
                                  reply_content_is_null_or_empty, NULL_REPLY_REASKS) as relay2:
            turn2 = run_eval_suite(samples=_for_base(answer_samples),
                                   eval_fn=_eval_longbench_v2_code,
                                   official_sampling=OFFICIAL_SAMPLING_COT_ANSWER,
                                   **{**run_kwargs, "attempt_count": 1,
                                      "base_url": relay2.url})
        reask["answer_query"] = reask_summary(relay2, PRED_PY_REASK_RULE)
    turns = [turn1] + ([turn2] if turn2 is not None else [])

    for tr in no_cot:
        tr["is_correct"] = False
        tr["error"] = tr.get("error") or ("the CoT query returned an empty reply, so no answer "
                                          "query was sent")
        tr["cot_step"] = "cot"
        tr["extra_payload"] = {k: v for k, v in (tr.get("extra_payload") or {}).items()
                               if k != _COT_ANSWER_PROMPT_KEY}
    answer_traces = list((turn2 or {}).get("sample_traces") or [])
    for tr in answer_traces:
        # One trace per question and attempt, as base numbers them: the answer query's trace
        # takes its CoT query's question index and attempt number.
        meta = tr.get("extra_payload") or {}
        tr["cot_step"] = "answer"
        tr["source_sample_idx"] = meta.get("cot_query_sample_idx")
        tr["attempt_idx"] = meta.get("cot_query_attempt_idx") or 0
    traces = sorted(answer_traces + no_cot, key=lambda t: (t.get("source_sample_idx") or 0,
                                                           t.get("attempt_idx") or 0))
    for i, tr in enumerate(traces):
        tr["sample_idx"] = i
    total = len(traces)
    # A question whose CoT query or answer query got no reply from the endpoint is a harness
    # error (base marks the trace in that query): left out of every number, counted and named.
    # An empty reply to either query is the model's and is scored wrong.
    scored = [t for t in traces if is_scored(t)]
    correct = sum(1 for t in scored if t.get("is_correct"))
    no_reply = sum(1 for t in traces if t.get("harness_error"))
    # an empty reply (a failed request, a status cause) apart from a reply with its answer inside
    # the thinking channel (the model's outcome: no status cause; base's ANSWER_IN_THINKING_POLICY)
    empty = (sum(1 for t in no_cot if not t.get("harness_error")
                 and t.get("health") != "empty_reasoned")
             + int((turn2 or {}).get("empty_responses") or 0))
    in_thinking_q = (sum(1 for t in no_cot if not t.get("harness_error")
                         and t.get("health") == "empty_reasoned")
                     + int((turn2 or {}).get("answer_in_thinking") or 0))
    unanswered = no_reply + empty + in_thinking_q
    kinds: Dict[str, int] = {}
    for t in traces:
        kind = (t.get("harness_error") or {}).get("kind")
        if kind:
            kinds[kind] = kinds.get(kind, 0) + 1
    answer_metas = [m for _, _, m in answer_samples]
    metas = [m for _, _, m in samples]
    shorter = _shorter_than_upstream(metas, max_len, upstream)
    answer_shorter = _shorter_than_upstream(answer_metas, answer_max_len, upstream)

    result: Dict[str, Any] = {
        "total_questions": total,
        "correct_answers": correct,
        "accuracy": round(100.0 * correct / len(scored), 2) if scored else None,
        "effective_n": len(scored),
        "scoring_excluded": total - len(scored),
        **({"request_harness_errors": dict(sorted(kinds.items()))} if kinds else {}),
        "category_accuracy": _category_accuracy(traces),
        "temperature": turn1.get("temperature"),
        "temperature_source": turn1.get("temperature_source"),
        "official_sampling": {"cot_query": turn1.get("official_sampling"),
                              "answer_query": (turn2 or {}).get("official_sampling")},
        "max_output_tokens_note": "; ".join(
            f"{name}: {t.get('max_output_tokens_note')}"
            for name, t in (("CoT query", turn1), ("answer query", turn2 or {}))
            if t.get("max_output_tokens_note")) or None,
        **{key: sum(int(t.get(key) or 0) for t in turns) for key in _SUMMED_COUNTERS},
        "unanswered_questions": unanswered,
        "upstream_reask": reask,
        "cot_query": _query_summary(turn1),
        "answer_query": _query_summary(turn2) if turn2 is not None else None,
        "duration_s": round(sum(float(t.get("duration_s") or 0) for t in turns), 2),
        "sample_traces": traces,
    }
    # over the requests of both queries (base's ANSWER_IN_THINKING_POLICY)
    result.update(answer_in_thinking_fields(
        result["answer_in_thinking"],
        sum(len(t.get("sample_traces") or []) for t in turns)))
    if turn1.get("shard"):
        result["shard"] = turn1["shard"]
    attempts = turn1.get("attempts") or {}
    per_sample = int(attempts.get("attempts_per_sample") or 1)
    if per_sample > 1:
        # Over the scored attempts only; a question with none scored is not counted.
        by_q: Dict[Any, List[bool]] = {}
        for t in scored:
            by_q.setdefault(t.get("source_sample_idx"), []).append(bool(t.get("is_correct")))
        n_q = len(by_q)
        # None, never 0, when no question has a scored attempt (as base's @k block)
        result["attempts"] = {
            **attempts, "generations": total, "avg_at_k": result["accuracy"], "samples": n_q,
            "pass_at_k": (round(100.0 * sum(1 for v in by_q.values() if any(v)) / n_q, 2)
                          if n_q else None),
            "pass_hat_k": (round(100.0 * sum(1 for v in by_q.values() if v and all(v)) / n_q, 2)
                           if n_q else None),
            "unstable_samples": sum(1 for v in by_q.values() if any(v) and not all(v))}
    else:
        result["attempts"] = attempts or None
    # A query that stopped at its official cap is the protocol's outcome, not a truncation.
    cut = sum(int(t.get("truncated_responses") or 0) - int(t.get("replies_at_official_max_tokens")
                                                            or 0) for t in turns)
    causes = []
    if no_reply:
        causes.append(f"harness errors left out of the score: {no_reply} question(s) whose CoT "
                      "query or answer query got no reply from the served endpoint ("
                      + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())) + ")")
    if empty:
        causes.append(f"empty_responses {empty} (to the CoT query or the answer query)")
    if cut * 10 > total:
        causes.append(f"replies_cut_off_by_non_official_budget {cut} of {total} (over 10%)")
    result["status_reason"] = None
    if total == 0:
        result["status"] = "error"
        result["status_reason"] = "no samples were run"
    elif not scored:
        result["status"] = "failed"
        result["status_reason"] = "nothing was scored: " + ("; ".join(causes)
                                                            or "every question left out")
        result["accuracy_withheld_reason"] = result["status_reason"]
    elif not causes:
        result["status"] = "success"
    else:
        result["status"] = "completed_with_errors"
        result["status_reason"] = "; ".join(causes)
    result["longbench_v2_code_report"] = {
        "dataset": f"{_DATASET_ID}@{_DATASET_REVISION[:8]}",
        "scope": "all LongBench-v2 domains" if all_domains else "Code Repository Understanding",
        "setting": ("w/ CoT (pred.py --cot, prompts/0shot_cot.txt then "
                    "prompts/0shot_cot_ans.txt)"),
        "window_tokens": window,
        "model_context_window": model_window,
        "model_context_window_source": model_window_source,
        "output_budget_tokens": budget,
        "answer_output_budget_tokens": answer_budget,
        "max_prompt_tokens": max_len,
        "answer_max_prompt_tokens": answer_max_len,
        "upstream_max_prompt_tokens": upstream,
        "upstream_max_len_rule": UPSTREAM_MAX_LEN_RULE,
        "tokenizer": tok_name,
        "prompts_truncated": sum(1 for m in metas if m.get("prompt_truncated")),
        "prompts_cut_shorter_than_upstream": shorter,
        "answer_prompts_truncated": sum(1 for m in answer_metas if m.get("prompt_truncated")),
        "answer_prompts_cut_shorter_than_upstream": answer_shorter,
        "questions_without_cot": len(no_cot),
        "no_extractable_answer": _no_extractable_answer(traces),
        "by_difficulty_and_length": _breakdown(traces),
    }

    own = []
    if thinking:
        own.append(f"--thinking run: LongBench-v2 has no native-reasoning setting, and each query "
                   f"was sent max_tokens={budget} for the reasoning and the reply, as the "
                   f"official max_tokens={_UPSTREAM_COT_MAX_NEW_TOKENS} (CoT query) and "
                   f"max_tokens={_UPSTREAM_MAX_NEW_TOKENS} (answer query) are not applied to "
                   "reasoning")
    else:
        other = _other_caps(turn1.get("sample_traces") or [], _UPSTREAM_COT_MAX_NEW_TOKENS)
        other_answer = _other_caps(answer_traces, _UPSTREAM_MAX_NEW_TOKENS)
        if other:
            own.append(f"some CoT queries were sent max_tokens {other} (LongBench-v2 sends "
                       f"{_UPSTREAM_COT_MAX_NEW_TOKENS})")
        if other_answer:
            own.append(f"some answer queries were sent max_tokens {other_answer} (LongBench-v2 "
                       f"sends {_UPSTREAM_MAX_NEW_TOKENS})")
    own += _truncation_reasons(metas, max_len, upstream, window, budget, model_window)
    own += _truncation_reasons(answer_metas, answer_max_len, upstream, window, answer_budget,
                               model_window, what="answer prompt(s)")
    for query, rec in reask.items():
        own += [f"{query}: {r}" for r in reask_divergence_reasons(
            rec, "empty or null content", "wrong", _PRED_PY_UPSTREAM, exhausted_reason=False)]
    answer_reasked = int((reask.get("answer_query") or {}).get("requests_reasked") or 0)
    if answer_reasked:
        # pred.py --cot skips a question whose answer query came back empty, and a resumed run
        # asks its CoT query again before a new answer query; gbench asks the answer query again
        # with the CoT it already has.
        own.append(f"{answer_reasked} answer query(ies) came back with empty or null content and "
                   "were asked again with the same CoT (a resumed pred.py --cot run asks the CoT "
                   "query again too)")
    _set_comparable(result, _reasons(
        all_domains, own, result["temperature"],
        _unanswered_reason(no_reply, empty,
                           "question(s) (to the CoT query or the answer query)",
                           in_thinking=in_thinking_q,
                           still_empty=sum(int(r.get("reasks_exhausted") or 0)
                                           for r in reask.values()),
                           cut_short=sum(int(r.get("reasks_cut_short") or 0)
                                         for r in reask.values())),
        limit, len(samples)))
    return result


def _infra_required_setting(setting: str, reason: str) -> Dict[str, Any]:
    """The entry of a setting whose prerequisite is missing while the other runs: not run, not
    scored, the missing prerequisite named."""
    label = _SETTING_LABELS[setting]
    msg = (f"[INFRA REQUIRED] {EVAL_NAME}: LongBench-v2's {label} setting was not run and not "
           f"scored: {reason} See '{DOCS_URL}'.")
    logger.error(msg)
    return {"status": "infra_required", "infra_required": reason,
            "error": msg, "leaderboard_comparable": False,
            "leaderboard_comparable_reason": (f"the {label} setting was not run (missing "
                                              "prerequisite, infra_required)"),
            "total_questions": 0, "correct_answers": 0, "accuracy": None}


def _grouped_reasons(results: Dict[str, Dict[str, Any]]) -> str:
    """Each setting's non-comparability reasons as one string: a reason every setting shares is
    written once, one that only some have names them."""
    order: List[str] = []
    where: Dict[str, List[str]] = {}
    for setting, res in results.items():
        reason = res.get("leaderboard_comparable_reason") or (
            "" if res.get("leaderboard_comparable") else "not comparable (no reason recorded)")
        for part in (p.strip() for p in reason.split("; ")):
            if part:
                if part not in where:
                    order.append(part)
                    where[part] = []
                where[part].append(_SETTING_LABELS[setting])
    return "; ".join(part if len(where[part]) == len(results)
                     else f"{part} ({', '.join(where[part])})" for part in order)


def run_longbench_v2_code(
    model_name: str,
    base_url: str,
    concurrency: int = 1,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run the LongBench-v2 code-domain MCQA suite in both settings (non-CoT headline, w/ CoT)."""
    start = time.time()
    window = _window_override() or get_run_knob("max_model_len")
    if not window:
        raise infra_required(
            EVAL_NAME, "the server does not report its context window (GET /v1/models), and "
            "LongBench-v2 prompts (up to ~5M tokens) must be truncated to it. Set "
            "GBENCH_LONGBENCH_V2_CODE_WINDOW to the served window in tokens (e.g. 131072).",
            DOCS_URL)
    window = int(window)
    budgets = {s: _output_budget(enable_thinking, _LONG_QUERY_SAMPLING[s]) for s in SETTINGS}
    # pred.py's max_len for the served window, from each setting's official cap (the policy's
    # no-think budget), and the max_len this run can send with its own budget.
    upstream = {s: _upstream_max_len(window) for s in SETTINGS}
    max_lens = {s: _prompt_budget(window, budgets[s], upstream[s]) for s in SETTINGS}
    too_small = {s: (f"the {window:,}-token window leaves only {max_lens[s]:,} tokens for the "
                     f"document after the {budgets[s]:,}-token output budget. Serve a larger "
                     "window" + (" or lower --max-output-tokens." if enable_thinking else "."))
                 for s in SETTINGS if max_lens[s] < _MIN_PROMPT_TOKENS}
    if len(too_small) == len(SETTINGS):
        raise infra_required(EVAL_NAME, too_small[SETTINGS[0]], DOCS_URL)
    tok_name = kwargs.get("tokenizer") or get_run_knob("tokenizer") or model_name
    tokenizer = _load_tokenizer(tok_name)
    model_window, model_window_source = _model_context_window(tok_name)
    if model_window and model_window > window:
        logger.warning("[%s] the served %s-token window is smaller than the model's %s-token "
                       "context window (%s): LongBench-v2 truncates at the model's context "
                       "window, so prompts cut here are not comparable", EVAL_NAME,
                       f"{window:,}", f"{model_window:,}", model_window_source)
    for s in SETTINGS:
        if s in too_small:
            continue
        if max_lens[s] == upstream[s]:
            logger.info("[%s] %s: middle-truncating prompts to %d tokens (pred.py's max_len for "
                        "a %s-token window) with tokenizer %r", EVAL_NAME, _SETTING_LABELS[s],
                        max_lens[s], f"{window:,}", tok_name)
        else:
            logger.warning("[%s] %s: middle-truncating prompts to %d tokens (%s window - %d "
                           "output budget - %d template slack), below pred.py's %d: prompts "
                           "longer than that are cut shorter than upstream cuts them", EVAL_NAME,
                           _SETTING_LABELS[s], max_lens[s], f"{window:,}", budgets[s],
                           _TEMPLATE_SLACK_TOKENS, upstream[s])

    limit = kwargs.get("limit")
    all_domains = _all_domains()
    run_kwargs = dict(
        eval_name=EVAL_NAME,
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=None,                  # already drawn by the loader, after the shard
        max_output_tokens=kwargs.get("max_output_tokens"),
        limit_seed="culer",          # pre-rename seed, as the loader's --eval-limit draw uses
    )
    common = dict(window=window, tok_name=tok_name, thinking=enable_thinking,
                  all_domains=all_domains, limit=limit, model_window=model_window,
                  model_window_source=model_window_source)
    # --eval-limit N is N samples for the suite: a (setting, question) pair is one, so the N are
    # dealt over the settings that can run (a setting dealt 0 is not run). Each setting draws its
    # share with the same seeded stratified draw, so the smaller share's questions are among the
    # larger's. No limit: every question in both settings.
    runnable = [s for s in SETTINGS if s not in too_small]
    # the headline setting (non-CoT, the result's accuracy) gets the first sample
    shares = allocate_limit(limit, runnable, seed=EVAL_NAME, first=(HEADLINE_SETTING,))
    results: Dict[str, Dict[str, Any]] = {}
    loaded: Dict[str, List[Any]] = {}
    for s in SETTINGS:
        if s in too_small:
            results[s] = _infra_required_setting(s, too_small[s])
            continue
        if shares[s] == 0:
            continue
        loaded[s] = _load_longbench_v2_code_samples(limit=shares[s], tokenizer=tokenizer,
                                                    max_prompt_tokens=max_lens[s],
                                                    cot=(s == "cot"))
    per_setting = {s: (len(loaded[s]) if s in loaded else 0) for s in runnable}
    # One CoT rollout is two queries (the CoT, then the answer with it).
    plan = eval_limit_plan(EVAL_NAME, limit, sum(per_setting.values()), per_setting,
                           attempts=base_attempts(), unit="rollouts")
    for s in SETTINGS:
        if s not in loaded:
            continue
        samples = loaded[s]
        if s == "cot":
            results[s] = _run_cot(run_kwargs, samples, tokenizer=tokenizer, budget=budgets[s],
                                  max_len=max_lens[s], upstream=upstream[s], **common)
        else:
            results[s] = _run_non_cot(run_kwargs, samples, budget=budgets[s],
                                      max_len=max_lens[s], upstream=upstream[s], **common)

    # The headline setting's result, with every setting beside it. `sample_idx` stays unique over
    # both settings' traces; `source_sample_idx` is the question's index, the same in both.
    # The headline setting, unless an --eval-limit too small for both settings left it out.
    headline = HEADLINE_SETTING if HEADLINE_SETTING in results else next(iter(results))
    result = dict(results[headline])
    traces: List[Dict[str, Any]] = []
    for s, res in results.items():
        offset = len(traces)
        for tr in res.get("sample_traces") or []:
            tr["setting"] = s
            if isinstance(tr.get("sample_idx"), int):
                tr["sample_idx"] += offset
            traces.append(tr)
    result.update({
        "headline_setting": headline,
        "eval_limit_plan": plan,
        "metric": ("accuracy (%) of LongBench-v2's non-CoT setting (pred.py, extract_answer "
                   "== gold); the w/ CoT setting (pred.py --cot) is scored on its own in "
                   "settings.cot and headline_metrics.cot_accuracy"),
        "headline_metrics": {f"{s}_accuracy": (None if res.get("status") in ("error",
                                                                             "infra_required")
                                               else res.get("accuracy"))
                             for s, res in results.items()},
        "settings": {s: {k: v for k, v in res.items() if k != "sample_traces"}
                     for s, res in results.items()},
        "sample_traces": traces,
        "duration_s": round(time.time() - start, 2),
    })
    if headline != HEADLINE_SETTING:
        result["metric"] = ("accuracy (%) of LongBench-v2's w/ CoT setting (pred.py --cot, "
                            "extract_answer == gold): this --eval-limit run dealt no sample to the "
                            "non-CoT setting")
    statuses = [res.get("status") for res in results.values()]
    missing = [s for s, res in results.items() if res.get("status") == "infra_required"]
    errored = [s for s, res in results.items() if res.get("status") in ("error", "failed")]
    if all(st == "success" for st in statuses):
        result["status"] = "success"
    elif len(missing) + len(errored) == len(results):
        result["status"] = "failed" if "failed" in statuses else "error"
        result["error"] = "; ".join(f"{_SETTING_LABELS[s]}: {res.get('error') or res.get('status')}"
                                    for s, res in results.items())
    else:
        result["status"] = "completed_with_errors"
    # What made each setting other than success, labelled by setting (None on success).
    result["status_reason"] = "; ".join(
        f"{_SETTING_LABELS[s]}: {res.get('status_reason') or res.get('error') or res.get('status')}"
        for s, res in results.items() if res.get("status") != "success") or None
    if errored:
        result["settings_with_errors"] = errored
    if missing:
        # Each such setting's entry names its missing prerequisite (_infra_required_setting).
        result["settings_infra_required"] = missing
    comparable = all(res.get("leaderboard_comparable") for res in results.values())
    result["leaderboard_comparable"] = comparable
    if comparable:
        result.pop("leaderboard_comparable_reason", None)
    else:
        result["leaderboard_comparable_reason"] = _grouped_reasons(results)
    return finalize_shard_fields(result)
