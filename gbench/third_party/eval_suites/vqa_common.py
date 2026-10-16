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

"""Canonical scoring metrics for the visual-question-answering suites.

These suites were all scored with the same bidirectional substring test
(``gold in pred or pred in gold``), which is far more lenient than any of their real
metrics: a verbose answer that merely mentions the gold passes, and a one-character
prediction contained in the gold passes too. This module implements what each benchmark
actually uses:

* **ANLS** (DocVQA, InfographicVQA) - average normalized Levenshtein similarity, correct
  when >= 0.5, which tolerates OCR-level typos but not different answers.
* **VQA accuracy** (TextVQA) - ``min(#matching annotators / 3, 1)`` over the 10 human
  answers, leave-one-out averaged, after the EvalAI answer processor that the TextVQA
  evaluation server uses (vendored verbatim below from facebookresearch/mmf
  ``mmf/utils/m4c_evaluators.py`` @ b0ffc7fba39f210f833c127362146aec01a2f259; lmms-eval's
  ``_task_utils/vqa_eval_metric.py`` carries the same class).
* **Relaxed accuracy** (ChartQA) - ``relaxed_correctness`` from google-research/pix2struct
  ``pix2struct/metrics.py`` @ 6fe25c1dc8151823ee3b479519d8d5948812fee4: the WHOLE answer is
  parsed as a float (a trailing ``%`` divides by 100) and must be within 5% of a non-zero
  gold, otherwise the strings must match case-insensitively.

chartqa, textvqa, docvqa and infographicvqa score the reply as lmms-eval does, with the
``lmms_*`` functions below (EvolvingLMMs-Lab/lmms-eval @ 42296f8fd2f0929d88a6ac4f7a1f2d784537f8d8):
the task scorer receives ``results[0]``, the reply text as the chat endpoint returned it with only
its surrounding whitespace stripped (api/task.py process_results), and no answer extraction.
``extract_short_answer`` is kept for callers that still need it.
"""

from __future__ import annotations

import re
import statistics
from typing import Any, Dict, List, Optional, Tuple

_WS = re.compile(r"\s+")


# Verbatim copy (whitespace included) of the upstream class. Its quirks are part of the
# metric and are kept on purpose: PERIOD_STRIP's lookbehind is written `(?!<=\d)`, and
# `re.UNICODE` lands in `sub`'s count argument, exactly as on the evaluation server.
class EvalAIAnswerProcessor:
    """
    Processes an answer similar to Eval AI
        copied from
        https://github.com/facebookresearch/mmf/blob/c46b3b3391275b4181567db80943473a89ab98ab/pythia/tasks/processors.py#L897
    """

    CONTRACTIONS = {
        "aint": "ain't",
        "arent": "aren't",
        "cant": "can't",
        "couldve": "could've",
        "couldnt": "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        "didnt": "didn't",
        "doesnt": "doesn't",
        "dont": "don't",
        "hadnt": "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        "hasnt": "hasn't",
        "havent": "haven't",
        "hed": "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        "hes": "he's",
        "howd": "how'd",
        "howll": "how'll",
        "hows": "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        "Im": "I'm",
        "Ive": "I've",
        "isnt": "isn't",
        "itd": "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        "itll": "it'll",
        "let's": "let's",
        "maam": "ma'am",
        "mightnt": "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        "mightve": "might've",
        "mustnt": "mustn't",
        "mustve": "must've",
        "neednt": "needn't",
        "notve": "not've",
        "oclock": "o'clock",
        "oughtnt": "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        "shant": "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        "shouldve": "should've",
        "shouldnt": "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": "somebodyd",
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        "somebodyll": "somebody'll",
        "somebodys": "somebody's",
        "someoned": "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        "someonell": "someone'll",
        "someones": "someone's",
        "somethingd": "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        "somethingll": "something'll",
        "thats": "that's",
        "thered": "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        "therere": "there're",
        "theres": "there's",
        "theyd": "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        "theyll": "they'll",
        "theyre": "they're",
        "theyve": "they've",
        "twas": "'twas",
        "wasnt": "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        "weve": "we've",
        "werent": "weren't",
        "whatll": "what'll",
        "whatre": "what're",
        "whats": "what's",
        "whatve": "what've",
        "whens": "when's",
        "whered": "where'd",
        "wheres": "where's",
        "whereve": "where've",
        "whod": "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        "wholl": "who'll",
        "whos": "who's",
        "whove": "who've",
        "whyll": "why'll",
        "whyre": "why're",
        "whys": "why's",
        "wont": "won't",
        "wouldve": "would've",
        "wouldnt": "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        "yall": "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        "youd": "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        "youll": "you'll",
        "youre": "you're",
        "youve": "you've",
    }

    NUMBER_MAP = {
        "none": "0",
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    ARTICLES = ["a", "an", "the"]
    PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
    COMMA_STRIP = re.compile(r"(?<=\d)(\,)+(?=\d)")
    PUNCTUATIONS = [
        ";",
        r"/",
        "[",
        "]",
        '"',
        "{",
        "}",
        "(",
        ")",
        "=",
        "+",
        "\\",
        "_",
        "-",
        ">",
        "<",
        "@",
        "`",
        ",",
        "?",
        "!",
    ]

    def __init__(self, *args, **kwargs):
        pass

    def word_tokenize(self, word):
        word = word.lower()
        word = word.replace(",", "").replace("?", "").replace("'s", " 's")
        return word.strip()

    def process_punctuation(self, in_text):
        out_text = in_text
        for p in self.PUNCTUATIONS:
            if (p + " " in in_text or " " + p in in_text) or (
                re.search(self.COMMA_STRIP, in_text) is not None
            ):
                out_text = out_text.replace(p, "")
            else:
                out_text = out_text.replace(p, " ")
        out_text = self.PERIOD_STRIP.sub("", out_text, re.UNICODE)
        return out_text

    def process_digit_article(self, in_text):
        out_text = []
        temp_text = in_text.lower().split()
        for word in temp_text:
            word = self.NUMBER_MAP.setdefault(word, word)
            if word not in self.ARTICLES:
                out_text.append(word)
            else:
                pass
        for word_id, word in enumerate(out_text):
            if word in self.CONTRACTIONS:
                out_text[word_id] = self.CONTRACTIONS[word]
        out_text = " ".join(out_text)
        return out_text

    def __call__(self, item):
        item = self.word_tokenize(item)
        item = item.replace("\n", " ").replace("\t", " ").strip()
        item = self.process_punctuation(item)
        item = self.process_digit_article(item)
        return item


_EVALAI_PROCESSOR = EvalAIAnswerProcessor()


def normalize_answer(text: Any) -> str:
    """Standard VQA normalization: the EvalAI answer processor (TextVQA / VQAv2)."""
    return _EVALAI_PROCESSOR(str(text or ""))


def as_gold_list(gold: Any) -> List[str]:
    if gold is None:
        return []
    if isinstance(gold, str):
        return [gold] if gold.strip() else []
    if isinstance(gold, (list, tuple, set)):
        return [str(g) for g in gold if str(g).strip()]
    return [str(gold)]


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _anls_normalize(text: Any) -> str:
    """Canonical ANLS normalization (DocVQA / ST-VQA): lowercase + whitespace-collapse ONLY.

    The full VQA ``normalize_answer`` strips articles, punctuation and maps number words,
    which is the wrong (over-aggressive) preprocessing for the edit-distance-based ANLS and
    made it diverge from published DocVQA/InfographicVQA numbers.
    """
    return _WS.sub(" ", str(text or "").replace("\n", " ").strip().lower()).strip()


def anls_score(pred: Any, golds: Any, threshold: float = 0.5) -> float:
    """ANLS over all reference answers (DocVQA / InfographicVQA)."""
    p = _anls_normalize(pred)
    best = 0.0
    for g in as_gold_list(golds):
        gn = _anls_normalize(g)
        if not gn and not p:
            return 1.0
        denom = max(len(p), len(gn))
        if denom == 0:
            continue
        # Canonical DocVQA/ST-VQA ANLS = 1 - normalized Levenshtein, thresholded at
        # 0.5, with NO numeric special case (kept faithful to the published metric).
        sim = 1.0 - (_levenshtein(p, gn) / denom)
        best = max(best, sim)
    return best if best >= threshold else 0.0


def eval_anls(pred: Any, golds: Any, threshold: float = 0.5) -> bool:
    """Binary form of the ANLS metric."""
    return anls_score(pred, golds, threshold) > 0.0


def vqa_accuracy(pred: Any, golds: Any) -> float:
    """Canonical VQA accuracy (Antol et al.; TextVQA/VQAv2 as in mmf / lmms-eval):
    the 10-way leave-one-out average of min(#matches-among-the-other-9 / 3, 1). With a
    full 10-annotator ensemble this is the published metric; with fewer annotations it
    reduces to the simple min(#matches / 3, 1) soft-accuracy form."""
    p = normalize_answer(pred)
    if not p:
        return 0.0
    gl = [normalize_answer(g) for g in as_gold_list(golds)]
    n = len(gl)
    if n < 10:
        matches = sum(1 for g in gl if g == p)
        return min(matches / 3.0, 1.0)
    accs = []
    for i in range(n):
        others = gl[:i] + gl[i + 1:]
        m = sum(1 for g in others if g == p)
        accs.append(min(m / 3.0, 1.0))
    return sum(accs) / n


def eval_vqa(pred: Any, golds: Any, threshold: float = 0.5) -> bool:
    """Binary form of VQA accuracy (>=2 of the annotators agree by default)."""
    golds_l = as_gold_list(golds)
    if len(golds_l) <= 2:
        # not an annotator ensemble -> exact match on the normalized answer
        p = normalize_answer(pred)
        return bool(p) and any(normalize_answer(g) == p for g in golds_l)
    return vqa_accuracy(pred, golds) >= threshold


def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """ChartQA relaxed correctness (pix2struct ``metrics.relaxed_correctness``, same logic).

    A numeric answer is correct within 5% of the gold (Methani et al. 2020, as adopted by
    ChartQA, arXiv:2203.10244 section 5.1); a non-numeric one needs an exact match. The
    whole string is parsed, so "Q3 2019" is not the number 3 and "Between 20 and 30" is not 20.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith("%"):
                # Convert percentages to floats.
                return float(text.rstrip("%")) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


def eval_relaxed(pred: Any, golds: Any, tolerance: float = 0.05) -> bool:
    """ChartQA relaxed accuracy: best ``relaxed_correctness`` over the gold answers.

    pix2struct's ``aggregate_metrics`` takes the max over targets and applies no
    normalization; only surrounding whitespace is stripped here. charxiv also imports this,
    but only on its no-judge path, which never runs (it hard-requires the judge).
    """
    p = str(pred or "").strip()
    if not p:
        return False
    return any(relaxed_correctness(str(g).strip(), p, tolerance) for g in as_gold_list(golds))


# --------------------------------------------------------------------------- #
# lmms-eval scorers of the raw reply (@ 42296f8f)
# --------------------------------------------------------------------------- #
def _reply_text(pred: Any) -> str:
    """The reply as lmms-eval's task layer hands it to the scorer: api/task.py process_results
    strips generate_until results (`res.strip() if res is not None else ""`)."""
    return "" if pred is None else str(pred).strip()


def _target_list(gold: Any) -> List[str]:
    """doc_to_target as a list, entries kept as they are (lmms-eval does not filter them)."""
    if gold is None:
        return []
    if isinstance(gold, (list, tuple)):
        return [str(g) for g in gold]
    return [str(gold)]


def lmms_chartqa_relaxed(pred: Any, gold: Any) -> bool:
    """tasks/chartqa/utils.py chartqa_process_results: relaxed_correctness(results[0], answer).

    The reply is only stripped of surrounding whitespace (by the task layer); markdown, labels
    and a trailing period stay. With more than one gold string the best one counts.
    """
    reply = _reply_text(pred)
    return any(relaxed_correctness(target, reply) for target in _target_list(gold))


def lmms_textvqa_accuracy(pred: Any, answers: Any) -> float:
    """tasks/textvqa/utils.py textvqa_process_results: EvalAI processor on the reply and on each
    of the answers, then the leave-one-out mean of min(1, #matching other answers / 3)."""
    res_ans = _EVALAI_PROCESSOR(_reply_text(pred))
    gts = [_EVALAI_PROCESSOR(a) for a in _target_list(answers)]
    if not gts:
        return 0.0
    gt_acc = []
    for i in range(len(gts)):
        other_gt_ans = [gts[j] for j in range(len(gts)) if i != j]
        matching_ans = [item for item in other_gt_ans if item == res_ans]
        gt_acc.append(min(1, float(len(matching_ans)) / 3))
    return statistics.mean(gt_acc)


def lmms_anls(pred: Any, references: Any, thresh_hold: float = 0.5) -> float:
    """lmms_eval/api/metrics.py anls, the docvqa_val / infovqa_val metric.

    Both strings are lower-cased and whitespace-collapsed for the edit distance, while the
    length that divides it is taken from the gold answer as stored and the (stripped) reply
    (``len(answer.upper())``, ``len(pred.upper())``), as upstream.
    """
    reply = _reply_text(pred)
    values = []
    for answer in _target_list(references):
        gt_answer = " ".join(answer.strip().lower().split())
        det_answer = " ".join(reply.strip().lower().split())
        dist = _levenshtein(gt_answer, det_answer)
        length = max(len(answer.upper()), len(reply.upper()))
        values.append(0.0 if length == 0 else float(dist) / float(length))
    if not values:
        return 0.0
    question_result = 1 - min(values)
    if question_result < thresh_hold:
        question_result = 0
    return question_result


def lmms_temperature_reason(result: dict) -> Optional[str]:
    """The leaderboard_comparable reason for a temperature other than the one lmms-eval's chat
    client sends (the task's, 0 unless the task sets another; gbench samples at 1.0 with
    --thinking and 0.0 without, base.resolve_temperature), or None."""
    official = result.get("official_sampling") or {}
    if official.get("temperature_matches_official") is not False:
        return None
    return f"temperature {result.get('temperature')} (lmms-eval decodes greedily)"


#: lmms-eval's chat-endpoint client, which sends every lmms-eval task's request.
LMMS_EVAL_CLIENT = "lmms-eval models/chat/openai.py @42296f8f"


def lmms_eval_sampling(task_reference: str, max_new_tokens: int, temperature: float = 0.0,
                       params: Optional[dict] = None) -> Any:
    """The base.OfficialSampling of an lmms-eval task sent through the chat-endpoint client.

    The client (build_payload_for_index in lmms_eval/models/chat/openai.py @42296f8f, lines
    190-223) sends the task's generation_kwargs: max_new_tokens as max_tokens, the temperature
    (default 0) and top_p / seed / penalties / n when the task sets them. A thinking model served
    by vLLM goes through the same payload with the client's enable_thinking_kwarg, which only adds
    chat_template_kwargs.enable_thinking: the temperature and the other generation_kwargs stay as
    they are. gbench sends the other generation_kwargs in both modes, but the temperature
    follows gbench's rule (base.resolve_temperature): 1.0 with --thinking, 0.0 without; the
    task's is recorded. The client's other
    branch (model names containing o1, o3, o4 or gpt-5: temperature dropped, max_new_tokens sent
    as max_completion_tokens) matches only those OpenAI model names and is not a thinking setting
    for any other model, so it is not used. lmms-eval declares no reasoning-model cap: the task's
    max_new_tokens was written for non-reasoning generation, so base.resolve_token_budget sends
    it exactly without --thinking and does not apply it with --thinking (the run's
    --max-output-tokens, else the thinking floor; the result records the cap as not applied to
    reasoning)."""
    from .base import OfficialSampling
    return OfficialSampling(
        reference=(f"{task_reference} (temperature {temperature:g}, max_new_tokens "
                   f"{max_new_tokens}, sent as max_tokens by {LMMS_EVAL_CLIENT}; the same "
                   "sampling for a thinking model, which the client runs with "
                   "enable_thinking_kwarg)"),
        temperature=temperature, params=params, max_tokens=max_new_tokens)


#: Trace statuses base.run_eval_suite gives a request that produced no reply at all.
LOST_REQUEST_STATUSES = ("FAILED", "TIMEOUT", "OVER_CONTEXT")
#: base's harness-error kind of a request with no reply -> the trace status it stands for.
_LOST_KIND_STATUS = {"endpoint_failure": "FAILED", "endpoint_timeout": "TIMEOUT",
                     "over_context": "OVER_CONTEXT", "request_failed_mid_loop": "FAILED"}


def _lost_status(trace: dict) -> Optional[str]:
    """The no-reply status of a trace (FAILED / TIMEOUT / OVER_CONTEXT), or None.

    base marks such a request with `harness_error`, which stays whatever a batch scorer later
    writes in `status` (some scorers set FAILED for a wrong answer); a trace without that marker
    counts when its status is a no-reply status and it carries no reply text."""
    err = trace.get("harness_error")
    if isinstance(err, dict) and err.get("kind") in _LOST_KIND_STATUS:
        return _LOST_KIND_STATUS[err["kind"]]
    if trace.get("status") in LOST_REQUEST_STATUSES and trace.get("response_text") is None:
        return trace["status"]
    return None


def lost_requests(traces: List[dict]) -> int:
    """How many requests failed, timed out or did not fit the context window (no reply)."""
    return sum(1 for t in traces if _lost_status(t))


def lost_requests_reason(traces: List[dict]) -> Optional[str]:
    """The leaderboard_comparable reason for requests that got no reply, or None.

    base leaves such a request out of the score as a harness error (never the model's outcome)
    and names it in status_reason; the reference harness has a reply for every question, so a
    run that lost any covers fewer questions and is not comparable.
    """
    lost = [_lost_status(t) for t in traces]
    counts = {s: lost.count(s) for s in LOST_REQUEST_STATUSES}
    total = sum(counts.values())
    if not total:
        return None
    parts = ", ".join(f"{n} {label}" for n, label in (
        (counts["FAILED"], "failed"), (counts["TIMEOUT"], "timed out"),
        (counts["OVER_CONTEXT"], "over the context window")) if n)
    return (f"{total} request(s) got no reply ({parts}) (harness errors, left out of the "
            "score)")


def mark_no_reply(trace: dict, status: str, error: Any) -> None:
    """Mark a trace whose request got no reply from the served endpoint (`status` FAILED,
    TIMEOUT or OVER_CONTEXT) the way base.run_eval_suite marks one, for a suite that sends its
    own requests: a harness error (`harness_error`), left out of the score (`scoring_excluded`),
    never the model's outcome."""
    from .base import endpoint_harness_error
    trace["status"] = status
    trace["is_correct"] = False
    trace["harness_error"] = endpoint_harness_error(status, error)
    trace["scoring_excluded"] = True


def own_loop_status(traces: List[dict], other_causes: Optional[List[str]] = None
                    ) -> Tuple[str, Optional[str], Dict[str, int]]:
    """(status, status_reason, request_harness_errors) of a suite that sends its own requests,
    by base.run_eval_suite's rule: `error` with no trace, `failed` when no trace is scored (the
    score is then withheld), `completed_with_errors` when a request got no reply from the
    endpoint (named as base names it) or a suite-specific cause (`other_causes`) applies,
    `success` otherwise."""
    kinds: Dict[str, int] = {}
    for t in traces:
        kind = (t.get("harness_error") or {}).get("kind")
        if kind:
            kinds[kind] = kinds.get(kind, 0) + 1
    causes = list(other_causes or [])
    if kinds:
        causes.insert(0, f"harness errors left out of the score: {sum(kinds.values())} "
                         "request(s) that got no reply from the served endpoint ("
                         + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())) + ")")
    if not traces:
        return "error", "no samples were run", kinds
    if all(t.get("scoring_excluded") for t in traces):
        return ("failed", "nothing was scored: " + ("; ".join(causes) or "every sample left out"),
                kinds)
    if causes:
        return "completed_with_errors", "; ".join(causes), kinds
    return "success", None, kinds


#: An `Answer:` / `Final Answer:` label. Group 1 is an emphasis run opening BEFORE the label
#: ('**Answer: 14**'), group 2 emphasis between the label and its colon ('**Answer**: 14'),
#: group 3 the span; emphasis after the colon ('**Answer:** 14') is removed from the span.
_ANSWER_LABEL = re.compile(r"(?i)([*_]*)\s*(?:final\s+answer|answer)\s*([*_]+\s*)?[:=]\s*(.+)")
#: A short label prefix on the last line ("Final value: 42.5%", "**Total:** 7"), with the same
#: three groups.
_SHORT_LABEL = re.compile(r"^([*_]*)[A-Za-z][A-Za-z ]{0,24}([*_]*)[:=]\s*(\S.*)$")
_EMPHASIS = "*_`"


def _strip_markdown(span: str) -> str:
    """The answer span without the markdown a chat model puts around it.

    Removes the label's closing emphasis ('**Answer:** 14' leaves '** 14'), a dangling
    emphasis run at the end, and emphasis or code markers wrapping the whole span
    ('**14**', '`STOP`'), one matching pair at a time; a wrapped span may be followed by one
    sentence-final '.' or ',' ('**14**.' -> '14'). Characters that belong to the answer stay:
    'LAAP*' (a ChartQA gold with a footnote asterisk) and '**LAAP***' -> 'LAAP*'. A trailing
    period on a bare answer is kept: golds such as 'U.S.' (ChartQA) and 'No.' (DocVQA) end with
    one, and the canonical metrics compare the answer as given.
    """
    s = span.strip()
    s = re.sub(r"^[*_]+\s+(?=\S)", "", s)
    s = re.sub(r"(?<=\S)\s+[*_]+$", "", s)
    wrapped = re.fullmatch(r"([*_`]+)(.+?)\1[.,]", s)
    if wrapped:
        s = wrapped.group(2).strip()
    while len(s) >= 2 and s[0] in _EMPHASIS and s[0] == s[-1]:
        s = s[1:-1].strip()
    return s


def _close_label_emphasis(opening: str, closed_before_colon: Optional[str], span: str) -> str:
    """Drop the closing half of an emphasis run that opened before the label.

    '**Answer: 14**' puts the opening '**' before the label, so the span is '14**'. The run is
    removed only when it opened before the label, was not closed at the label ('**Answer**:')
    or right after the colon ('**Answer:** 14'), and the span ends with that same run, so a bare
    gold such as 'Rabbit**' or 'LAAP*' after a plain label keeps its asterisks.
    """
    span = span.strip()
    if (not opening or closed_before_colon or span[:1] in ("*", "_")
            or not span.endswith(opening) or len(span) <= len(opening)):
        return span
    return span[:-len(opening)].rstrip()


def extract_short_answer(text: Any) -> str:
    """Pull the stated answer out of a verbose response before metric comparison.

    Models often answer "The total is 42." - the metrics expect "42". Prefer an explicit
    Answer:/Final Answer: span (markdown around the label, the answer or the whole label line
    is accepted: '**Answer:** 14', 'Answer: **14**', '**Answer: 14**'), else the last non-empty
    line; markdown emphasis and backticks around the span are removed (see _strip_markdown and
    _close_label_emphasis).
    """
    s = str(text or "").strip()
    if not s:
        return ""
    m = _ANSWER_LABEL.findall(s)
    if m:
        opening, closed, span = m[-1]
        span = span.strip().split("\n")[0]
        return _strip_markdown(_close_label_emphasis(opening, closed, span))
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    last = lines[-1] if lines else s
    # Strip a short label prefix ("Final value: 42.5%", "Total: 7"). This is extraction,
    # not leniency: the value still has to match, a wrong value still fails.
    label = _SHORT_LABEL.match(last)
    if label:
        return _strip_markdown(_close_label_emphasis(label.group(1), label.group(2),
                                                     label.group(3)))
    return _strip_markdown(last)
