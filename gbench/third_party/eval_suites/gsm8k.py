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

"""Native GSM8K evaluation suite: lm-evaluation-harness `gsm8k_cot`.

The protocol is EleutherAI lm-evaluation-harness at commit d6de8164, task `gsm8k_cot`
(lm_eval/tasks/gsm8k/gsm8k-cot.yaml, version 3.0), run the way lm-eval runs it against a chat
endpoint (`local-chat-completions` with `--apply_chat_template`, which turns
`fewshot_as_multiturn` on by default):
  * data: `openai/gsm8k`, config `main`, split `test` (the yaml's dataset_path), pinned;
  * few-shot: the yaml's 8 fixed exemplars (Wei et al. 2022; `fewshot_config.samples`, sampler
    `first_n`) as alternating turns: user "Q: <question>\\nA:" and assistant "<target>", each
    target ending "The answer is N."; then the test question as user "Q: <question>\\nA:"
    (doc_to_text). No system message (the task has no description);
  * generation: greedy (`do_sample: false`; local-chat-completions then sends temperature 0 and
    seed 1234) with the stop sequences "Q:", "</s>" and "<|im_end|>" (`until`), which
    local-chat-completions sends to the server as the request's `stop` (its parse_generations
    applies no stop strings itself). A `--thinking` request carries no stop strings, since a
    stop string also matches inside the reasoning channel; its reply is cut at them after the
    reasoning is removed, the way lm-eval's HF backend cuts a generation (models/huggingface.py:
    postprocess_generated_text with `until` and think_end_token);
  * filters: `strict-match` (regex `The answer is (\\-?[0-9\\.\\,]+).`, first match) and
    `flexible-extract` (regex `(-?[$0-9.,]{2,})|(-?[0-9]+)`, the last match, its first
    non-empty group), each followed by take_first; no match gives "[invalid]";
  * metric: `exact_match` against the text after "####", with ignore_case, no punctuation
    stripping, and regexes_to_ignore `,`, `\\$`, `(?s).*#### ` and `\\.$` applied to both sides
    (lm_eval/api/metrics.py exact_match_hf_evaluate).
The headline is exact_match under strict-match; flexible-extract is reported alongside, as
lm-eval's results table lists both filters.

lm-eval's chat backend caps a reply at `max_gen_toks` (256 by default in
lm_eval/models/api_models.py; the task sets none), which local-chat-completions sends as the
request's `max_tokens`. That is the suite's official cap (base.OfficialSampling.max_tokens),
applied by base's token-limit policy: a no-think request is sent `max_tokens=256` exactly, and
`--max-output-tokens` is not applied to it (`max_output_tokens_note`); a reply cut off at 256
tokens is scored on what arrived, as lm-eval scores it. A `--thinking` request is sent the run's
budget instead (`--max-output-tokens`, else the thinking floor), since the 256 was written for
non-reasoning generation and the reasoning shares the budget; the result records that the
official cap was not applied to reasoning. Its replies longer than 256 tokens, reasoning
included, are counted (`replies_over_max_gen_toks`) and make the run non-comparable, since
lm-eval would have cut them off.

Sharding (`--shard I/N`) and `--eval-limit` are applied by base.run_eval_suite (shard first).

Sampling: lm-eval's is temperature 0 and seed 1234 (base.OfficialSampling; the result records
`official_sampling`). lm-eval's `openai-chat-completions` model (the class for OpenAI's hosted
API) sends temperature 1 and no stop for model names containing o1, o3, o4 or 5
(OpenAIChatCompletion._create_payload), which adapts the request to what OpenAI's hosted
reasoning models accept; `local-chat-completions`, the backend for a served model, has no such
branch and sends the same request whether the model reasons or not. gbench's temperature is its
rule for every suite (base.resolve_temperature): 0.0 without --thinking (lm-eval's 0), 1.0 with
it, seed 1234 in both; so a thinking run is not comparable. Apart from the thinking switch
itself, that temperature and the token budget (the token-limit policy above), a gbench thinking
request differs from that request in one more way:
the stop sequences are applied after the reasoning instead of sent (see above), since a stop
string would also match inside the reasoning channel. Override the temperature for a whole run
with `--temperature`, or for this suite alone with `GBENCH_GSM8K_TEMPERATURE`, which takes
precedence over both, in either mode; any temperature other than 0 makes the run
non-comparable. LLM-judge grading is pinned at 0.0 and is not affected by either
(this suite has no judge).
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import (OfficialSampling, finalize_shard_fields, is_scored, run_eval_suite,
                   strip_thinking_tags)
from .sampling import limit_dataset

logger = logging.getLogger(__name__)

DATASET = "openai/gsm8k"
#: openai/gsm8k revision the loader reads (main/test, 1,319 problems; data unchanged since the
#: 2024-01 Parquet conversion).
_DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
TEST_SIZE = 1319
LM_EVAL_COMMIT = "d6de81643928d653435c431bae19945d41d32520"

# lm_eval/tasks/gsm8k/gsm8k-cot.yaml at LM_EVAL_COMMIT, verbatim: fewshot_config.samples as
# (question, target).
GSM8K_COT_SAMPLES: List[Tuple[str, str]] = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After "
     "they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were planted. So "
     "there must have been 21 - 15 = 6. The answer is 6."),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the "
     "parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have "
     "left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. "
     "After eating 35, they had 74 - 35 = 39. The answer is 39."),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many "
     "lollipops did Jason give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave "
     "Denny 20 - 12 = 8. The answer is 8."),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many "
     "toys does he have now?",
     "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more "
     "toys. 5 + 4 = 9. The answer is 9."),
    ("There were nine computers in the server room. Five more computers were installed each day, "
     "from monday to thursday. How many computers are now in the server room?",
     "There were originally 9 computers. For each of 4 days, 5 more computers were added. So "
     "5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29."),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 "
     "more. How many golf balls did he have at the end of wednesday?",
     "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After "
     "losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has "
     "23 - 15 dollars left. 23 - 15 is 8. The answer is 8."),
]
#: num_fewshot.
N_SHOT = 8
#: generation_kwargs.until.
UNTIL = ["Q:", "</s>", "<|im_end|>"]
#: filter_list.
STRICT_PATTERN = r"The answer is (\-?[0-9\.\,]+)."
FLEXIBLE_PATTERN = r"(-?[$0-9.,]{2,})|(-?[0-9]+)"
FALLBACK = "[invalid]"
#: metric_list[0].regexes_to_ignore (ignore_case: true, ignore_punctuation: false).
REGEXES_TO_IGNORE = [",", "\\$", "(?s).*#### ", "\\.$"]
#: lm_eval/models/api_models.py: the default max_gen_toks of the chat backend.
LM_EVAL_MAX_GEN_TOKS = 256

OFFICIAL_SAMPLING = OfficialSampling(
    reference=(f"EleutherAI/lm-evaluation-harness@{LM_EVAL_COMMIT[:8]} gsm8k-cot.yaml "
               "(do_sample: false) via models/openai_completions.py LocalChatCompletion "
               "(temperature 0, seed 1234, max_tokens=max_gen_toks 256)"),
    temperature=0.0, params={"seed": 1234}, max_tokens=LM_EVAL_MAX_GEN_TOKS)


def doc_to_text(question: str) -> str:
    """gsm8k-cot.yaml doc_to_text: 'Q: {{question}}\\nA:'."""
    return f"Q: {question}\nA:"


def doc_to_target(answer: str) -> str:
    """gsm8k-cot.yaml doc_to_target for a dataset row: the text after '####', stripped."""
    return answer.split("####")[-1].strip()


def fewshot_messages(question: str, n_shot: int = N_SHOT) -> List[Dict[str, str]]:
    """Task.fewshot_context with a chat template and fewshot_as_multiturn: the first `n_shot`
    exemplars as user/assistant turns, then the question as the last user turn."""
    if n_shot < 0 or n_shot > len(GSM8K_COT_SAMPLES):
        raise ValueError(f"gsm8k: --eval-n-shot {n_shot} exceeds the {len(GSM8K_COT_SAMPLES)} "
                         "gsm8k_cot exemplars")
    messages: List[Dict[str, str]] = []
    for q, target in GSM8K_COT_SAMPLES[:n_shot]:
        messages.append({"role": "user", "content": doc_to_text(q)})
        messages.append({"role": "assistant", "content": target})
    messages.append({"role": "user", "content": doc_to_text(question)})
    return messages


def postprocess(generation: str) -> str:
    """lm_eval/models/utils.py postprocess_generated_text: cut at every stop sequence."""
    for term in UNTIL:
        generation = generation.split(term)[0]
    return generation


def regex_filter(resp: str, pattern: str, group_select: int = 0) -> str:
    """lm_eval/filters/extraction.py RegexFilter for one response (fallback "[invalid]")."""
    match = re.compile(pattern).findall(resp if isinstance(resp, str) else "")
    if match:
        match = match[group_select]
        if isinstance(match, tuple):
            match = [m for m in match if m]
            match = match[0] if match else FALLBACK
        return match.strip()
    return FALLBACK


def exact_match(prediction: str, reference: str) -> bool:
    """lm_eval/api/metrics.py exact_match_hf_evaluate with the task's metric settings."""
    for s in REGEXES_TO_IGNORE:
        prediction = re.sub(s, "", prediction)
        reference = re.sub(s, "", reference)
    return prediction.lower() == reference.lower()


def _reply_text(response_text: Optional[str]) -> str:
    """The reply the filters read: the message content (a vLLM server with a reasoning parser
    returns the reasoning separately), without inline thought blocks, cut at the stop sequences.

    lm-eval's chat backend cuts nothing client-side (parse_generations passes stop=None): the
    server stops a no-think reply at the `stop` strings it was sent, so the cut changes nothing
    there. A thinking request carries no stop strings, and its reply is cut here after the
    reasoning is removed, as lm-eval's HF backend applies `until` (postprocess_generated_text
    with think_end_token)."""
    text = response_text or ""
    if re.search(r"<(thought|reasoning)>", text, re.IGNORECASE):
        text = strip_thinking_tags(text)
    return postprocess(text)


def score_reply(response_text: Optional[str], reference: str) -> Dict[str, Any]:
    """Both filters and their exact_match verdicts for one reply."""
    reply = _reply_text(response_text)
    strict = regex_filter(reply, STRICT_PATTERN)
    flexible = regex_filter(reply, FLEXIBLE_PATTERN, group_select=-1)
    return {"strict_match": strict, "strict_match_correct": exact_match(strict, reference),
            "flexible_extract": flexible,
            "flexible_extract_correct": exact_match(flexible, reference)}


def _load_gsm8k_samples(
    limit: Optional[int] = None, n_shot: int = N_SHOT, enable_thinking: bool = False,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """GSM8K test problems as gsm8k_cot chat requests: (messages, reference, meta).

    A no-think request carries the stop sequences as `stop`, as lm-eval's chat backend sends
    them (its `max_gen_toks` is OFFICIAL_SAMPLING.max_tokens, applied by base). A thinking
    request carries no stop strings (a stop string also matches inside the reasoning channel);
    its reply is cut at the stop sequences after the reasoning is removed (_reply_text), the way
    lm-eval's HF backend applies `until`."""
    from datasets import load_dataset
    ds = load_dataset(DATASET, "main", split="test", revision=_DATASET_REVISION)
    # Stratified, not a contiguous head.
    ds = limit_dataset(ds, limit, None, seed="gsm8k")
    raw_samples = list(ds)
    logger.info(f"Loaded {len(raw_samples)} GSM8K samples from HF Hub ('{DATASET}'); "
                f"gsm8k_cot {n_shot}-shot.")
    meta: Dict[str, Any] = {} if enable_thinking else {"stop": list(UNTIL)}
    return [(fewshot_messages(str(item["question"]), n_shot), doc_to_target(str(item["answer"])),
             dict(meta)) for item in raw_samples]


async def _async_score_gsm8k(sample_traces: List[Dict[str, Any]]) -> None:
    """Apply both filters to every reply; the strict-match verdict is the sample's."""
    for t in sample_traces:
        if t.get("status") != "PENDING_JUDGE":
            continue
        scores = score_reply(t.get("response_text"), str(t.get("gold_answer") or ""))
        t.update(scores)
        t["is_correct"] = scores["strict_match_correct"]
        t["status"] = "TRUNCATED" if t.get("health") in ("truncated", "looping") else "OK"


def _eval_gsm8k(response_text: str, gold: str) -> bool:
    """strict-match exact_match for one reply (the headline filter)."""
    return score_reply(response_text, gold)["strict_match_correct"]


def run_gsm8k(model_name: str, base_url: str, concurrency: int, enable_thinking: bool = False,
              **kwargs) -> Dict[str, Any]:
    """Run GSM8K with lm-evaluation-harness gsm8k_cot (8-shot; `--eval-n-shot N` uses the first N
    exemplars and makes the run non-comparable)."""
    _ns = kwargs.get("eval_n_shot")
    if _ns is None:
        _ns = kwargs.get("n_shot")
    n_shot = N_SHOT if _ns is None else int(_ns)
    if n_shot > len(GSM8K_COT_SAMPLES):
        raise ValueError(f"gsm8k: --eval-n-shot {n_shot} exceeds the {len(GSM8K_COT_SAMPLES)} "
                         "gsm8k_cot exemplars")
    n_shot = max(0, n_shot)
    limit = kwargs.get("limit")
    # base applies --shard and then --eval-limit to the full set.
    samples = _load_gsm8k_samples(n_shot=n_shot, enable_thinking=enable_thinking)
    result = run_eval_suite(
        eval_name="gsm8k",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        async_eval_fn=_async_score_gsm8k,
        declared_scoring_mode="deterministic",
        supports_attempts=True,      # each trace is scored on its own, never keyed by id
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=limit,
        max_output_tokens=kwargs.get("max_output_tokens"),
        official_sampling=OFFICIAL_SAMPLING,   # its max_tokens=256: base's token-limit policy
    )
    traces = result.get("sample_traces") or []
    n = len(traces)
    # Both filters over the scored replies, as base's accuracy: a request that got no reply from
    # the endpoint is a harness error (scoring_excluded), left out; with none scored the score is
    # withheld (None, base's accuracy_withheld_reason), never 0.
    scored = [t for t in traces if is_scored(t)]

    def pct(key: str) -> Optional[float]:
        if not scored:
            return None
        return round(sum(1 for t in scored if t.get(key)) / len(scored) * 100.0, 2)

    over_cap = sum(1 for t in traces
                   if int(t.get("completion_tokens") or 0) > LM_EVAL_MAX_GEN_TOKS)
    result["gsm8k_cot_report"] = {
        "strict-match": {"exact_match": pct("strict_match_correct")},
        "flexible-extract": {"exact_match": pct("flexible_extract_correct")},
        "max_gen_toks_sent": None if enable_thinking else LM_EVAL_MAX_GEN_TOKS,
        "replies_over_max_gen_toks": over_cap,
    }
    result["accuracy"] = pct("strict_match_correct")
    result["flexible_extract_accuracy"] = pct("flexible_extract_correct")
    result["metric"] = ("exact_match, strict-match filter (lm-evaluation-harness gsm8k_cot, "
                        f"{n_shot}-shot); flexible-extract in gsm8k_cot_report")
    result["n_shot"] = n_shot

    reasons = []
    if n_shot != N_SHOT:
        reasons.append(f"{n_shot}-shot (gsm8k_cot is 8-shot)")
    k = int((result.get("attempts") or {}).get("attempts_per_sample") or 1)
    if k > 1:
        reasons.append(f"{k} samples per problem, accuracy averaged over them (gsm8k_cot scores "
                       "one greedy generation)")
    if (result.get("official_sampling") or {}).get("temperature_matches_official") is False:
        reasons.append(f"sampled at temperature {result.get('temperature')} "
                       f"({result.get('temperature_source')}; gsm8k_cot is greedy)")
    if over_cap:
        reasons.append(f"{over_cap} repl(ies) ran past lm-eval's {LM_EVAL_MAX_GEN_TOKS}-token "
                       "max_gen_toks, which would have cut them off (a thinking request is sent "
                       "the run's budget: the official cap is not applied to reasoning)")
    other = sorted({t.get("max_tokens_sent") for t in traces
                    if t.get("max_tokens_sent") not in (None, LM_EVAL_MAX_GEN_TOKS)})
    if other and not enable_thinking:
        reasons.append(f"some requests were sent max_tokens {other} (the prompt plus lm-eval's "
                       f"{LM_EVAL_MAX_GEN_TOKS} did not fit the context window)")
    no_reply = sum(1 for t in traces if t.get("harness_error"))
    if no_reply:
        reasons.append(f"{no_reply} request(s) got no reply from the served endpoint (harness "
                       "errors, left out of the score)")
    empty = int(result.get("empty_responses") or 0)
    if empty:
        reasons.append(f"{empty} request(s) came back empty and count as wrong")
    if limit:
        reasons.append(f"--eval-limit subset ({n} of {TEST_SIZE:,} problems)")
    result["leaderboard_comparable"] = not reasons
    if reasons:
        result["leaderboard_comparable_reason"] = "; ".join(reasons)
    else:
        result.pop("leaderboard_comparable_reason", None)
    # One shard of a --shard I/N run is a partial set: the shard reason (finalize_shard_fields).
    return finalize_shard_fields(result)
