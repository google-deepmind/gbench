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

"""Native Multilingual MMLU (openai/MMMLU, 14 languages) evaluation suite, simple-evals protocol.

MMMLU is OpenAI's human-translated MMLU test set, released with openai/simple-evals as the
reference harness. Prompt, extraction and scoring are ported verbatim from simple-evals
(commit 652c89d0): the English zero-shot CoT `QUERY_TEMPLATE_MULTICHOICE`, `normalize_response`,
the `MULTILINGUAL_ANSWER_REGEXES` cascade (the first anchor word that matches wins, and within
it the first occurrence), `normalize_extracted_answer`, and an exact compare with the gold
letter (`mmlu_eval.py`). The user message is the same with and without `--thinking`.

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
for this suite alone with `GBENCH_MULTILINGUAL_MMLU_TEMPERATURE`, which takes precedence over
both (and over the rule's 1.0 / 0.0); a value other than 0.5 makes the run non-comparable. LLM-judge
grading is pinned at 0.0 and is not
affected by either.

Sharding: `--shard I/N` selects shard I of the interleaved rows before the `--eval-limit` draw.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from .arc_agi import _for_base   # lays out an already-sharded sample list for run_eval_suite
from .limit_plan import base_attempts, eval_limit_plan, per_category
from .base import apply_shard, run_eval_suite
from .gpqa import QUERY_TEMPLATE_MULTICHOICE  # simple-evals' shared multiple-choice template
from .gpqa import SIMPLE_EVALS_SAMPLING, _flag_comparability, simple_evals_messages
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

DATASET = "openai/MMMLU"
DATASET_REVISION = "325a01dc3e173cac1578df94120499aaca2e2504"

OPTION_LETTERS = ["A", "B", "C", "D"]

# The 14 language configs of openai/MMMLU (LANG_REGION codes); each is the full 14,042-row
# MMLU test set, human-translated. The 'default' config is excluded (it is not a language
# split). Columns: 'Question', 'A'..'D', 'Answer' (letter A-D), 'Subject'. Rows are
# interleaved across languages so an --eval-limit subset still spans them.
MMMLU_CONFIGS = [
    "AR_XY", "BN_BD", "DE_DE", "ES_LA", "FR_FR", "HI_IN", "ID_ID",
    "IT_IT", "JA_JP", "KO_KR", "PT_BR", "SW_KE", "YO_NG", "ZH_CN",
]


def _load_multilingual_mmlu_samples(
    enable_thinking: bool = False,
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load Multilingual MMLU from HF Hub (openai/MMMLU, per-language configs, pinned revision).

    Raises on load/schema failure (no fabricated fallback, no silent fallback to
    English cais/mmlu). Every request carries the chat sampler's system message
    (gpqa.simple_evals_messages); `enable_thinking` does not change the messages.
    """
    from datasets import load_dataset

    per_lang: List[Tuple[str, List[Dict[str, Any]]]] = []
    for cfg in MMMLU_CONFIGS:
        lang = cfg.split("_")[0].lower()          # AR_XY -> ar, PT_BR -> pt
        try:
            ds = load_dataset(DATASET, cfg, split="test", revision=DATASET_REVISION)
        except Exception as e:
            logger.error(f"Failed to load MMMLU language '{cfg}': {e}")
            raise RuntimeError(
                f"Could not load multilingual_mmlu language '{cfg}': {e}"
            ) from e
        per_lang.append((lang, list(ds)))

    # Round-robin interleave so a --eval-limit still spans multiple languages.
    lang_rows: List[Tuple[str, Dict[str, Any]]] = []
    max_len = max((len(rows) for _, rows in per_lang), default=0)
    for i in range(max_len):
        for lang, rows in per_lang:
            if i < len(rows):
                lang_rows.append((lang, rows[i]))

    if not lang_rows:
        raise RuntimeError("Dataset for multilingual_mmlu returned empty rows")

    # --shard first (round-robin over the interleaved rows), then the --eval-limit draw.
    lang_rows, _ = apply_shard(lang_rows, "multilingual_mmlu")
    # Stratified, not a contiguous head.
    lang_rows = stratified_sample(lang_rows, limit, lambda r: r[0], seed="multilingual_mmlu")
    logger.info(f"Loaded {len(lang_rows)} Multilingual MMLU samples from HF Hub.")

    samples = []
    for lang, item in lang_rows:
        q_text = item.get("Question")
        opts = [item.get(c) for c in ("A", "B", "C", "D")]
        ans = item.get("Answer")
        if (not q_text or any(o is None for o in opts)
                or not (isinstance(ans, str) and ans.strip().upper() in OPTION_LETTERS)):
            raise RuntimeError(
                "multilingual_mmlu: unexpected dataset schema "
                "(need 'Question', 'A'..'D', 'Answer' letter); "
                "refusing to fabricate sample data"
            )
        prompt = QUERY_TEMPLATE_MULTICHOICE.format(
            Question=q_text, A=opts[0], B=opts[1], C=opts[2], D=opts[3])
        messages = simple_evals_messages([{"role": "user", "content": prompt}], enable_thinking)
        samples.append((messages, ans.strip().upper(), {"category": lang}))

    return samples


# openai/simple-evals common.py, verbatim (escapes spell out the invisible and combining
# characters). The letter class also admits the Arabic, Bengali and full-width forms of A-D.
MULTILINGUAL_ANSWER_PATTERN_TEMPLATE = (
    "(?i){}[ \t]*([A-D]|[\u0623-\u062f]|[\u0985]|[\u09ac]|[\u09a1]|[\u09a2]|[\uff21]|[\uff22]|[\uff23]|[\uff24])"
)
# All the different ways "Answer" is written in different languages
MULTILINGUAL_ANSWER_REGEXES = [
    "Answer\\s*:",
    "Answer\\s*:\u200b\u200b\u200b\u200b\u200b\u200b",
    "উত\u09cdতর\\s*:",
    "उत\u094dतर\\s*:",
    "উত\u09cdতরঃ",
    "উত\u09cdতর\\s*:",
    "Antwort\\s*:",
    "답변\\s*:",
    "정답\\s*:",
    "답\\s*:",
    "答案\\s*：",
    "答案\\s*:",
    "答\\s*：",
    "答\\s*:",
    "答复\\s*：",
    "答曰\\s*：",
    "الإجابة:",
    "الجواب:",
    "إجابة:",
    "الإجابة النهائية:",
    "الإجابة الصحيحة:",
    "الإجابة الصحيحة هي:",
    "الإجابة هي:",
    "الجواب النهائي:",
    "Respuesta\\s*:",
    "Risposta\\s*:",
    "答え\\s*:",
    "答え\\s*：",
    "回答\\s*:",
    "回答\\s*：",
    "解答\\s*:",
    "Jawaban\\s*:",
    "Réponse\\s*:",
    "Resposta\\s*:",
    "Jibu\\s*:",
    "Idahun\\s*:",
    "Ìdáhùn\\s*:",
    "Idáhùn\\s*:",
    "A\u0300mọ\u0300na\u0300\\s*:",
    "Àdáhùn\\s*:",
    "A\u0300nu\u0301go\u0323\\s*:",
    "Àṣàyàn\\s*:",
]


def normalize_response(response: str) -> str:
    """
    Normalize the response by removing markdown and LaTeX formatting that may prevent a match.
    """

    return (
        response.replace("**", "")
        .replace("$\\boxed{", "")
        .replace("}$", "")
        .replace("\\$", "")
        .replace("$\\text{", "")
        .replace("$", "")
        .replace("\\mathrm{", "")
        .replace("\\{", "")
        .replace("\\text", "")
        .replace("\\(", "")
        .replace("\\mathbf{", "")
        .replace("{", "")
        .replace("\\boxed", "")
    )


def normalize_extracted_answer(extracted_answer: str) -> str:
    return (
        # In arabic these are the letters used for A-D in multiple choice questions
        extracted_answer.replace("\u0623", " A")
        .replace("\u0628", " B")
        .replace("\u062c", " C")
        .replace("\u062f", " D")
        # In Bengali these are the letters used for A-D in multiple choice questions
        .replace("\u0985", " A")
        .replace("\u09ac", " B")
        .replace("\u09a1", " C")
        .replace("\u09a2", " D")
        # In Japanese these are the letters sometimes used for A-D in multiple choice questions
        .replace("\uff21", " A")
        .replace("\uff22", " B")
        .replace("\uff23", " C")
        .replace("\uff24", " D")
        .strip()
    )


def _extract_mc_letter(resp: str) -> Optional[str]:
    """simple-evals `MMLUEval` extraction: the answer letter, or None when no anchor matches.

    The anchors are tried in list order and the first one that matches decides, so a reply
    must state its answer as an anchor word, a colon and the letter (`Answer: B`, `答案：B`).
    A letter that starts a word ("La respuesta correcta es B") is not an answer statement.
    """
    response_text = normalize_response(resp)
    for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
        regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
        match = re.search(regex, response_text)
        if match:
            return normalize_extracted_answer(match.group(1))
    return None


def _eval_multilingual_mmlu(response_text: str, gold_letter: str) -> bool:
    """Extract the answer letter and compare it exactly (case-sensitive) with the gold."""
    if not response_text or not gold_letter:
        return False
    return _extract_mc_letter(response_text) == gold_letter


def run_multilingual_mmlu(
    model_name: str,
    base_url: str,
    concurrency: int = 1,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native Multilingual MMLU evaluation suite."""
    samples = _load_multilingual_mmlu_samples(
        enable_thinking=enable_thinking,
        limit=kwargs.get("limit"),
    )
    eval_plan = eval_limit_plan(
        "multilingual_mmlu", kwargs.get("limit"), len(samples), per_category(samples),
        setting_noun="language", attempts=base_attempts())
    result = run_eval_suite(
        eval_name="multilingual_mmlu",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=_for_base(samples),   # already this shard's rows; base keeps exactly them
        eval_fn=_eval_multilingual_mmlu,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=None,                   # already drawn by the loader, after the shard
        max_output_tokens=kwargs.get("max_output_tokens"),
        official_sampling=SIMPLE_EVALS_SAMPLING,   # per mode, cap included (token-limit policy)
    )
    result["eval_limit_plan"] = eval_plan
    # 14 languages x the 14,042-question MMLU test set.
    return _flag_comparability(result, kwargs.get("limit"), len(samples),
                               len(MMMLU_CONFIGS) * 14042, thinking=enable_thinking)
