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

# Evaluation Suite: seal_tools
# Description: Seal-Tools (NLPCC Multi-Tool Single-Turn and Nested Function Calling Benchmark)

"""gbench native built-in runner for seal_tools (Tool Use & Function Calling).

Runs Seal-Tools as fairyshine/Seal-Tools `LLM_eval.ToolLearning_eval` does: each test row's own
instruction (the first conversation turn) is sent as a single user message, the reply goes through
`transform_output_format("ToolLearning", ...)`, and `calculate_score_ToolLearning` pools the
Format ACC, Tool P/R/F1 and Parameter P/R/F1 counts over the whole test set (in-domain +
out-domain, merged as the repo's src/eval.py does).

Sampling: as the repo's chat-model wrapper (LLM_Evaluation/src/llm_tools/llms/ChatGPT.py sends
only `model` and `messages`), no sampling field other than the temperature is sent. The
temperature is gbench's rule for every suite (base.resolve_temperature): 1.0 with `--thinking`, 0.0
without (the wrapper sends none). `--temperature`, or `GBENCH_SEAL_TOOLS_TEMPERATURE` for this
suite alone (it takes precedence), sets another; the result records upstream's setting under
`official_sampling`. No max_tokens is sent
either, in either mode (the server grants the room left in the context window; --max-output-tokens
is not applied and `max_output_tokens_note` says so). No LLM judge.
"""

import ast
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import OfficialSampling, run_eval_suite, strip_thinking_tags
from .sampling import parse_shard, stratified_sample
from .fc_common import base_suite_plan, run_attempts
from .metrics import scorable_traces
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Function Calling"

_REPO = "casey-martin/Seal-Tools"
#: HF revision of the test split; its 1354 rows equal fairyshine/Seal-Tools
#: Seal-Tools_Dataset/dataset_for_finetune/test_in_domain.json + test_out_domain.json at
#: ce753ecd60ed08dd376984035de531ab8421f1c6 (checked row by row on 2026-09-24).
_REVISION = "d0fe2245740d01a22b8fdd22ec1f49e48fcb1fbf"
_TEST_ROWS = 1354
#: fairyshine/Seal-Tools revision the protocol is ported from.
SEAL_TOOLS_COMMIT = "ce753ecd60ed08dd376984035de531ab8421f1c6"
#: LLM_Evaluation/src/llm_tools/llms/ChatGPT.py answer(): ChatCompletion.create(model, messages):
#: no temperature and no max_tokens.
SEAL_TOOLS_SAMPLING = OfficialSampling(
    reference=(f"fairyshine/Seal-Tools@{SEAL_TOOLS_COMMIT[:12]} LLM_Evaluation/src/llm_tools/llms/"
               "ChatGPT.py (model and messages only)"),
    send_temperature=False,
    send_max_tokens=False,
)


def _load_seal_tools_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load the Seal-Tools test split (700 in-domain + 654 out-domain rows) from HF Hub."""
    rows = []
    try:
        from datasets import load_dataset
        # The TEST split is the evaluation set. The train split's `domain` column is the literal
        # string "train" for every row; on test it is in-domain / out-domain.
        ds = load_dataset(_REPO, split="test", revision=_REVISION)
        rows = list(ds)
    except Exception as e:
        raise infra_required("seal_tools", f"{_REPO}@{_REVISION} (test split) could not be "
                             f"loaded ({e}).", "docs/evals/seal_tools.md") from e

    if not rows:
        raise RuntimeError("Dataset for seal_tools returned empty rows")

    # Stratified, not a contiguous head.
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("domain"), seed="seal_tools")

    samples = []
    for item in rows:
        convs = item.get("conversations") or []
        if len(convs) < 2 or not convs[0].get("value") or not convs[1].get("value"):
            raise RuntimeError(
                f"seal_tools: unexpected row schema (id={item.get('id')!r}); "
                "refusing to fabricate sample data")
        # ToolLearning_eval sends conversations[0]["value"] as is and scores against
        # conversations[1]["value"].
        messages = [{"role": "user", "content": convs[0]["value"]}]
        samples.append((messages, convs[1]["value"], {"category": str(item.get("domain"))}))

    logger.info(f"Loaded {len(samples)} seal_tools samples.")
    return samples


# --------------------------------------------------------------------------- #
# Port of LLM_Evaluation/src/llm_tools/evaluation/transform_output.py ("ToolLearning" branch)
# and calculate.py::calculate_score_ToolLearning.
# --------------------------------------------------------------------------- #
_API_LIST_START = re.compile(r'\[\s*\{\s*"api"', re.DOTALL)


def _match_square_bracket(text: str, pos_s: int) -> int:
    counter = -1
    for i in range(pos_s + 1, len(text)):
        if text[i] == "[":
            counter -= 1
        elif text[i] == "]":
            counter += 1
        if counter == 0:
            return i
    return -1


def _transform_output(output_text: str) -> Any:
    """transform_output_format("ToolLearning", ...): the first `[{"api"...]` list in the reply
    after every ' becomes " and newlines are removed, or -1 if there is none or it does not
    parse. Calls written any other way (prose, name/arguments objects) are not read."""
    text = re.sub("'", '"', output_text)
    text = re.sub("\n", "", text)
    search_result = re.search(_API_LIST_START, text)
    if search_result is None:
        return -1
    pos_s = search_result.span()[0]
    pos_e = _match_square_bracket(text, pos_s)
    text = text[pos_s:pos_e + 1]
    if "api" in text and "parameters" in text and "responses" in text:
        try:
            return json.loads(text)
        except Exception:
            return -1
    return -1


def _gold_calls(gold_text: str) -> List[Dict[str, Any]]:
    """calculate.py: json.loads(json.dumps(eval(gold))). literal_eval gives the same value for
    these Python-literal golds without executing code."""
    return json.loads(json.dumps(ast.literal_eval(gold_text)))


def _sample_counts(gold_answer: List[Dict[str, Any]], predict: Any) -> Dict[str, int]:
    """One row of calculate_score_ToolLearning's counting loop."""
    c = {"format": 0, "correct_api": 0, "predict_api": 0, "gold_api": len(gold_answer),
         "correct_param": 0, "predict_param": 0,
         "gold_param": sum(len(g["parameters"]) for g in gold_answer)}
    if predict == -1:
        return c
    c["format"] = 1
    for predict_api in predict:
        # Upstream indexes non-dict items and would raise; such an item cannot match anything.
        if not isinstance(predict_api, dict):
            continue
        if "api" in predict_api:
            c["predict_api"] += 1
            if "parameters" in predict_api and type(predict_api["parameters"]) == dict:
                c["predict_param"] += len(predict_api["parameters"])
            gold_idx = -1
            for idx in range(len(gold_answer)):
                if gold_answer[idx]["api"] == predict_api["api"]:
                    gold_idx = idx
                    break
            if gold_idx != -1:
                c["correct_api"] += 1
                if "parameters" in predict_api and type(predict_api["parameters"]) == dict:
                    for parameter_name in predict_api["parameters"]:
                        if parameter_name in gold_answer[gold_idx]["parameters"] and \
                                str(predict_api["parameters"][parameter_name]) == \
                                str(gold_answer[gold_idx]["parameters"][parameter_name]):
                            c["correct_param"] += 1
    return c


def _scores(counts: List[Dict[str, int]]) -> Dict[str, float]:
    """calculate_score_ToolLearning's result dict (a metric whose counts multiply to 0 is left
    out upstream; it is reported as 0.0 here)."""
    tot = {k: sum(c[k] for c in counts) for k in
           ("format", "correct_api", "predict_api", "gold_api",
            "correct_param", "predict_param", "gold_param")}
    out = {"format_accuracy": tot["format"] / len(counts) if counts else 0.0}
    for name, correct, pred, gold in (("tool", "correct_api", "predict_api", "gold_api"),
                                      ("parameter", "correct_param", "predict_param", "gold_param")):
        p = r = f1 = 0.0
        if tot[correct] * tot[pred] * tot[gold] > 0:
            p = tot[correct] / tot[pred]
            r = tot[correct] / tot[gold]
            f1 = 2 * p * r / (p + r)
        out[f"{name}_precision"], out[f"{name}_recall"], out[f"{name}_f1"] = p, r, f1
    return out


def _eval_seal_tools(response_text: str, gold_target: str) -> bool:
    """Per-sample verdict: every gold call and parameter reproduced (by calculate.py's rules)
    with nothing extra. Secondary; the headline is the pooled Parameter F1."""
    if not response_text or not str(gold_target).strip():
        return False
    try:
        gold = _gold_calls(str(gold_target))
    except Exception:
        return False
    c = _sample_counts(gold, _transform_output(response_text))
    return bool(c["format"]) and c["correct_api"] == c["predict_api"] == c["gold_api"] \
        and c["correct_param"] == c["predict_param"] == c["gold_param"]


def run_seal_tools(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute seal_tools; headline is the pooled Parameter F1 (calculate_score_ToolLearning)."""
    samples = _load_seal_tools_samples(limit=kwargs.get("limit"))
    limit_plan = base_suite_plan("seal_tools", samples, kwargs.get("limit"), run_attempts(kwargs))
    result = run_eval_suite(
        eval_name="seal_tools",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_seal_tools,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
        official_sampling=SEAL_TOOLS_SAMPLING,
    )
    result["eval_limit_plan"] = limit_plan
    counts: List[Dict[str, int]] = []
    by_domain: Dict[str, List[Dict[str, int]]] = {}
    for t in scorable_traces(result):
        # The same reasoning-stripped text the per-sample verdict saw.
        c = _sample_counts(_gold_calls(t.get("gold_answer")),
                           _transform_output(strip_thinking_tags(t.get("response_text") or "")))
        t["seal_counts"] = c
        counts.append(c)
        by_domain.setdefault(str(t.get("category")), []).append(c)
    scores = _scores(counts)
    result["exact_match_rate"] = result.get("accuracy")
    for key, val in scores.items():
        result[key] = round(val * 100.0, 2)
    result["per_domain"] = {d: {k: round(v * 100.0, 2) for k, v in _scores(cs).items()}
                            for d, cs in sorted(by_domain.items())}
    result["metric"] = ("Parameter F1 pooled over the test set (Seal-Tools "
                        "calculate_score_ToolLearning F1_param); format_accuracy (AMOUNT), "
                        "tool P/R/F1, parameter P/R and per_domain also reported")
    result["dataset_revision"] = f"{_REPO}@{_REVISION}"
    if counts:
        result["accuracy"] = result["parameter_f1"]
    reasons = []
    if kwargs.get("limit"):
        reasons.append(f"--eval-limit {kwargs.get('limit')} (Seal-Tools scores all {_TEST_ROWS} test rows)")
    if parse_shard(os.environ.get("GBENCH_SHARD")):
        reasons.append("sharded run (a subset of the test rows)")
    if (result.get("official_sampling") or {}).get("temperature_matches_official") is False:
        reasons.append(f"temperature {result.get('temperature')} set by "
                       f"{result.get('temperature_source')} (upstream sends none)")
    if result.get("decoding_penalties"):
        reasons.append("decoding penalties set")
    if ((result.get("attempts") or {}).get("attempts_per_sample") or 1) > 1:
        reasons.append("--attempt-count > 1 (counts pool every attempt)")
    result["leaderboard_comparable"] = not reasons
    result["leaderboard_comparable_reason"] = "; ".join(reasons) if reasons else None
    return result
