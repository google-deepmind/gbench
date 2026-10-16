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

"""Native CausalBench suite: Corr2Cause (Jin et al., ICLR 2024), the release protocol.

The protocol is causalNLP/corr2cause `code/run_model.py` at commit 21d80350, the path its paper
uses for chat models (`GPT.query_by_prem_and_hypo` without CoT, `GPT.query_unseen_text`,
`Model.evaluate`), with the zhijing-jin/efficiency `Chatbot` it calls (commit 69d191bc):
  * data: `causal-nlp/corr2cause`, split `test` (1,162 items), pinned. Each row's `input` is
    `Premise: <premise>\\nHypothesis: <hypothesis>` (run_model.py's `prompt_tmpl_generic`), which
    is split back into the two parts;
  * prompt: `prompt_tmpl_direct` (the last assignment in `Constants`),
    `Question: <premise>\\nCan we deduct the following: <hypothesis>? Just answer "Yes" or
    "No."\\n\\nAnswer:` with the hypothesis stripped of its final period, sent after the system
    prompt `GPT.__init__` gives the Chatbot ("You are a highly intelligent question-answering
    bot with profound knowledge of causal inference.");
  * generation: `Chatbot.raw_query`'s `temperature=0.` and `GPT.query_unseen_text`'s
    `max_tokens=10` (a prompt ending in "Answer:");
  * reading the reply: `GPT.clean_pred_func` (lower-case, strip spaces, quotes, periods and
    colons, drop a leading "that there is ") and `Constants.rel_normalize` (the
    `pred_rel_lookup` table, then its first two nltk word tokens, then its first one; anything
    else is "contradiction"), ported verbatim. Label 1 is "entailment" (valid), label 0
    "contradiction";
  * metric: `perf_df2report_dict`: F1, precision and recall of the minority label (the valid
    class, "entailment", on the test set), and accuracy.
The headline is that F1, the paper's main number.

A reply whose `message.content` is None is asked again, the same request, as the Chatbot's
`ask` retries it (swebench_common.upstream_reask_relay, up to NULL_REPLY_REASKS; the result
records `upstream_reask`); an empty string is read as "contradiction", as upstream reads it.

The `max_tokens=10` is the suite's official cap (base.OfficialSampling.max_tokens), applied by
base's token-limit policy: a no-think request is sent `max_tokens=10` exactly, and
`--max-output-tokens` is not applied to it (`max_output_tokens_note`). A `--thinking` request is
sent the run's budget instead (`--max-output-tokens`, else the thinking floor), since the 10 was
written for a direct answer and the reasoning shares the budget; the result records that the
official cap was not applied to reasoning. A thinking run is not the published protocol and is
flagged non-comparable; its replies longer than 10 tokens are counted in
`causalbench_report.replies_over_upstream_cap`.

Sampling: gbench's rule (base.resolve_temperature): temperature 1.0 with `--thinking`, 0.0
without; Corr2Cause's temperature 0 is recorded (base.OfficialSampling; the result records
`official_sampling`). Override it
for a whole run with `--temperature`, or for this suite alone with
`GBENCH_CAUSALBENCH_TEMPERATURE`, which takes precedence over both (and over the rule's 1.0 / 0.0);
a value other than 0 makes the run non-comparable. LLM-judge grading is pinned at 0.0 and is not
affected by either (this suite has
no judge).
"""

import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
from .base import OfficialSampling, finalize_shard_fields, run_eval_suite, strip_thinking_tags
from .sampling import limit_dataset
from .swebench_common import (infra_required, reask_divergence_reasons, reask_summary,
                              reply_content_is_null, upstream_reask_relay)

logger = logging.getLogger(__name__)

DATASET = "causal-nlp/corr2cause"
#: causal-nlp/corr2cause revision the loader reads (test split, 1,162 rows).
_DATASET_REVISION = "42ba12c769e11ff6427c9f52d7db58e3f9bf3e53"
DOCS_URL = "docs/evals/causalbench.md"
TEST_SIZE = 1162
CORR2CAUSE_COMMIT = "21d80350e27a6206f536cf90a09e95d9e856c761"
#: zhijing-jin/efficiency `Chatbot` (commit 69d191bc) re-asks a reply whose `message.content` is
#: None: raw_query's `response_text.strip()` raises on None, and `ask` catches it ("Unknown
#: exception when calling openai") and asks again after a doubling wait, with no limit. An empty
#: string is a reply (read as "contradiction"). gbench re-asks a null reply, the same request, up
#: to this many times (swebench_common.upstream_reask_relay; the wait changes nothing sampled
#: and is not reproduced); a reply still null after them is scored as base scores it and makes
#: the run non-comparable.
NULL_REPLY_REASKS = 5
CHATBOT_REASK_RULE = ("efficiency Chatbot.ask: a reply whose message.content is None is asked "
                      "again (the same request)")

# --- causalNLP/corr2cause code/run_model.py (commit 21d80350), verbatim --------------------------
PROMPT_TMPL_DIRECT = ('Question: {premise}\nCan we deduct the following: {hypothesis}? Just answer '
                      '"Yes" or "No."\n\nAnswer:')
SYSTEM_PROMPT = ("You are a highly intelligent question-answering bot with profound knowledge of "
                 "causal inference.")
PRED_REL_LOOKUP = {
    'necessarily true': 'entailment',
    'necessarily false': 'contradiction',
    'neither to say': 'neutral',
    'yes': 'entailment',
    'no': 'contradiction',
    'true': 'entailment',
    'false': 'contradiction',
    'neither': 'neutral',

    'must be true': 'entailment',
    'must be false': 'contradiction',
    'not enough information': 'neutral',

    'neutral': 'contradiction',  # TODO: comment this out if you are doing 3-way NLI.
    'not necessarily true': 'contradiction',
}
CLASSES = ['entailment', 'contradiction', 'neutral']
#: GPT.query_unseen_text: a prompt ending in 'Answer:' gets max_tokens=10.
UPSTREAM_MAX_TOKENS = 10


def lstrip_word(word, pref):
    """zhijing-jin/efficiency function.lstrip_word (commit 69d191bc)."""
    if word.startswith(pref):
        return word[len(pref):]
    return word


def clean_pred(i):
    """GPT.clean_pred_func."""
    return lstrip_word(i.lower().strip().strip(' ".:').strip(), 'that there is ')


def rel_normalize(surface_form):
    """Constants.rel_normalize."""
    from nltk import word_tokenize
    rel_normalize = lambda i: PRED_REL_LOOKUP.get(i, i)  # noqa: E731  (verbatim)

    normalized = rel_normalize(rel_normalize(surface_form))
    if normalized not in CLASSES:
        surface_form = ' '.join(word_tokenize(surface_form)[:2])
        normalized = rel_normalize(rel_normalize(surface_form))
        if normalized not in CLASSES:
            surface_form = ' '.join(word_tokenize(surface_form)[:1])
            normalized = rel_normalize(rel_normalize(surface_form))
            if normalized not in CLASSES:
                return 'contradiction'
    return normalized
# --- end verbatim ------------------------------------------------------------------------------

OFFICIAL_SAMPLING = OfficialSampling(
    reference=("causalNLP/corr2cause@21d80350 code/run_model.py GPT.query_unseen_text "
               "(max_tokens=10 for a prompt ending in 'Answer:') -> "
               "zhijing-jin/efficiency@69d191bc nlp.Chatbot.raw_query(temperature=0.)"),
    temperature=0.0, max_tokens=UPSTREAM_MAX_TOKENS)


def _require_nltk() -> None:
    """rel_normalize tokenizes with nltk (punkt data); without it the suite cannot score."""
    try:
        from nltk import word_tokenize
        word_tokenize("yes, it can")
    except ImportError as e:
        raise infra_required("causalbench", f"Corr2Cause reads replies with nltk's word_tokenize "
                             f"({e}). Install it (`pip install nltk`).", DOCS_URL) from e
    except LookupError as e:
        raise infra_required(
            "causalbench", "Corr2Cause reads replies with nltk's word_tokenize, which needs the "
            "punkt tokenizer data: `python -m nltk.downloader punkt punkt_tab` (set NLTK_DATA if "
            f"it lives elsewhere). {str(e).strip().splitlines()[0] if str(e).strip() else ''}",
            DOCS_URL) from e


def _split_input(text: str) -> Tuple[str, str]:
    """(premise, hypothesis) from `Premise: <premise>\\nHypothesis: <hypothesis>`."""
    head, sep, hypothesis = text.partition("\nHypothesis: ")
    if not sep or not head.startswith("Premise: "):
        raise RuntimeError("causalbench: unexpected 'input' format (need 'Premise: ...\\n"
                           "Hypothesis: ...'); refusing to fabricate sample data")
    return head[len("Premise: "):], hypothesis


def build_prompt(premise: str, hypothesis: str) -> str:
    """GPT.query_by_prem_and_hypo (no CoT) + Model.query_text's prompt.strip()."""
    return PROMPT_TMPL_DIRECT.format(premise=premise,
                                     hypothesis=hypothesis.strip().rstrip('.')).strip()


def _load_causalbench_samples(
    limit: Optional[int] = None, enable_thinking: bool = False,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Corr2Cause test items as (messages, gold relation, meta). Upstream's `max_tokens=10` is
    OFFICIAL_SAMPLING.max_tokens, applied per mode by base's token-limit policy;
    `enable_thinking` does not change the samples."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, split="test", revision=_DATASET_REVISION)
    # Stratified, not a contiguous head.
    ds = limit_dataset(ds, limit, "num_variables", seed="causalbench")
    raw_samples = list(ds)
    logger.info(f"Loaded {len(raw_samples)} Corr2Cause samples from HF Hub ('{DATASET}').")

    samples = []
    for item in raw_samples:
        premise, hypothesis = _split_input(str(item.get("input") or ""))
        label = item.get("label")
        if label not in (0, 1):
            raise RuntimeError(f"causalbench: unexpected label {label!r} (need 0 or 1)")
        gold = "entailment" if label == 1 else "contradiction"
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(premise, hypothesis)}]
        meta = {"category": f"{item.get('num_variables')}_vars",
                "num_variables": item.get("num_variables"), "relation": item.get("template")}
        samples.append((messages, gold, meta))
    return samples


def predicted_relation(response_text: Optional[str]) -> str:
    """The relation upstream reads from a reply (Chatbot strips it; then clean + normalize)."""
    return rel_normalize(clean_pred((response_text or "").strip()))


def _eval_causalbench(response_text: str, gold_relation: str) -> bool:
    """Per-sample accuracy: the normalized reply equals the gold relation."""
    return predicted_relation(response_text) == str(gold_relation)


def perf_report(golds: List[str], preds: List[str]) -> Dict[str, Any]:
    """Model.perf_df2report_dict: F1, P and R of the minority label (in %), and accuracy.

    classification_report's labels are those in the gold or the predictions; the minority is
    the one with the least gold support (ties by name), as `min((support, label))` picks it; a
    label with no predictions (or no support) has precision (or recall) 0."""
    support = Counter(golds)
    labels = sorted(set(golds) | set(preds))
    if not labels:
        # nothing scored: no figure (None), never 0
        return {"F1": None, "P": None, "R": None, "Acc": None, "minority_label": None, "tp": 0,
                "fp": 0, "fn": 0, "total": 0}
    minority = min((support.get(label, 0), label) for label in labels)[-1]
    tp = sum(1 for g, p in zip(golds, preds) if g == minority and p == minority)
    fp = sum(1 for g, p in zip(golds, preds) if g != minority and p == minority)
    fn = sum(1 for g, p in zip(golds, preds) if g == minority and p != minority)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    acc = sum(1 for g, p in zip(golds, preds) if g == p) / len(golds) if golds else 0.0
    return {"F1": round(f1 * 100, 2), "P": round(precision * 100, 2),
            "R": round(recall * 100, 2), "Acc": round(acc * 100, 2), "minority_label": minority,
            "tp": tp, "fp": fp, "fn": fn, "total": len(golds)}


def run_causalbench(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run Corr2Cause with the release protocol; headline = F1 of the minority (valid) class."""
    _require_nltk()
    limit = kwargs.get("limit")
    # base applies --shard and then --eval-limit (stratified over the number of variables).
    samples = _load_causalbench_samples(enable_thinking=enable_thinking)
    # The Chatbot asks a null-content reply again: base's requests go through a relay that does
    # the same (NULL_REPLY_REASKS), so base reads the reply the Chatbot would return.
    with upstream_reask_relay("causalbench", base_url, reply_content_is_null,
                              NULL_REPLY_REASKS) as relay:
        result = run_eval_suite(
            eval_name="causalbench",
            model_name=model_name,
            base_url=relay.url,
            concurrency=concurrency,
            samples=samples,
            eval_fn=_eval_causalbench,
            thinking=enable_thinking,
            extra_payload=kwargs.get("extra_payload"),
            limit=limit,
            max_output_tokens=kwargs.get("max_output_tokens"),
            official_sampling=OFFICIAL_SAMPLING,   # its max_tokens=10: base's token-limit policy
        )
    result["upstream_reask"] = reask_summary(relay, CHATBOT_REASK_RULE)
    traces = result.get("sample_traces") or []
    # Over the scored items, as base's accuracy: a request that got no reply from the endpoint is
    # a harness error (scoring_excluded), left out of every figure, never read as a prediction.
    from .base import is_scored
    scored = [t for t in traces if is_scored(t)]
    golds, preds, groups = [], [], {"by_num_variables": {}, "by_relation": {}}
    for t in scored:
        gold = str(t.get("gold_answer") or "")
        # The text the per-sample scorer saw (inline thinking removed).
        pred = predicted_relation(strip_thinking_tags(t.get("response_text")))
        t["predicted_relation"] = pred
        golds.append(gold)
        preds.append(pred)
        meta = t.get("extra_payload") or {}
        for key, field in (("by_num_variables", "num_variables"), ("by_relation", "relation")):
            bucket = groups[key].setdefault(str(meta.get(field)), ([], []))
            bucket[0].append(gold)
            bucket[1].append(pred)
    overall = perf_report(golds, preds)
    over_cap = sum(1 for t in traces if int(t.get("completion_tokens") or 0) > UPSTREAM_MAX_TOKENS)
    result["raw_accuracy"] = overall["Acc"] if scored else None
    result["accuracy"] = overall["F1"] if scored else None
    if not scored and traces:
        result["accuracy_withheld_reason"] = result.get("accuracy_withheld_reason") or (
            "no item was scored: " + str(result.get("status_reason")
                                         or "every request left out of the score"))
    result["causalbench_report"] = {
        "f1": overall["F1"], "precision": overall["P"], "recall": overall["R"],
        "accuracy": overall["Acc"], "minority_label": overall["minority_label"],
        "tp": overall["tp"], "fp": overall["fp"], "fn": overall["fn"],
        "by_num_variables": {k: perf_report(*v)
                             for k, v in sorted(groups["by_num_variables"].items())},
        "by_relation": {k: perf_report(*v) for k, v in sorted(groups["by_relation"].items())},
        "upstream_max_tokens": UPSTREAM_MAX_TOKENS,
        # the per-request max_tokens base sent (10 without thinking; the run's budget with it)
        "max_tokens_sent": result.get("max_tokens_sent"),
        "replies_over_upstream_cap": over_cap,
    }
    result["metric"] = ("F1 of the minority label (the valid class) in percent, Corr2Cause "
                        "run_model.py perf_df2report_dict; precision, recall and accuracy in "
                        "causalbench_report")
    result["headline_metrics"] = {"F1": overall["F1"], "P": overall["P"], "R": overall["R"],
                                  "Acc": overall["Acc"]}
    # The rows next to the headline carry the headline metric: per number of variables, the
    # minority-label F1 (causalbench_report.by_num_variables); base's per-sample exact-label
    # tally moves to category_label_accuracy, and correct_answers (exact labels) is not the
    # F1's numerator.
    by_vars = result["causalbench_report"]["by_num_variables"]
    result["category_label_accuracy"] = result.get("category_accuracy") or {}
    result["category_accuracy"] = {
        f"{k}_vars": {"correct": None, "total": v["total"], "accuracy": v["F1"]}
        for k, v in by_vars.items() if v.get("total")}
    result["exact_label_matches"] = result.get("correct_answers")
    result["correct_answers"] = None
    result["correct_answers_note"] = ("not applicable: the headline is the minority-label F1; "
                                      "exact_label_matches counts replies whose relation equals "
                                      "the gold (raw_accuracy)")

    reasons = []
    k = int((result.get("attempts") or {}).get("attempts_per_sample") or 1)
    if k > 1:
        reasons.append(f"{k} samples per item, pooled into the metric (Corr2Cause scores one "
                       "generation)")
    if enable_thinking:
        reasons.append("--thinking run: the model reasons before answering; the published "
                       "protocol reads a direct answer of at most 10 tokens (the official "
                       "max_tokens=10 is not applied to reasoning)")
    if (result.get("official_sampling") or {}).get("temperature_matches_official") is False:
        reasons.append(f"temperature {result.get('temperature')} "
                       f"({result.get('temperature_source')}; Corr2Cause decodes at 0)")
    other = sorted({t.get("max_tokens_sent") for t in traces
                    if t.get("max_tokens_sent") not in (None, UPSTREAM_MAX_TOKENS)})
    if other and not enable_thinking:
        reasons.append(f"some requests were sent max_tokens {other} (the prompt plus upstream's "
                       f"{UPSTREAM_MAX_TOKENS} did not fit the context window)")
    no_reply = sum(1 for t in traces if t.get("harness_error"))
    if no_reply:
        reasons.append(f"{no_reply} request(s) got no reply from the served endpoint (harness "
                       "errors, left out of the score); upstream's Chatbot retries them")
    # A null-content reply (an answer inside the thinking channel among them) was asked again,
    # as the Chatbot asks it (upstream_reask); an empty-string reply is read as 'contradiction'
    # by upstream too. Only a reply still null after gbench's NULL_REPLY_REASKS differs.
    reasons.extend(reask_divergence_reasons(
        result.get("upstream_reask"), "null content", "as base reads them ('contradiction')",
        "upstream's Chatbot asks a null reply again without limit"))
    if limit:
        reasons.append(f"--eval-limit subset ({len(traces)} of {TEST_SIZE:,} items)")
    result["leaderboard_comparable"] = not reasons
    if reasons:
        result["leaderboard_comparable_reason"] = "; ".join(reasons)
    else:
        result.pop("leaderboard_comparable_reason", None)
    # One shard of a --shard I/N run is a partial set: the shard reason (finalize_shard_fields).
    return finalize_shard_fields(result)
