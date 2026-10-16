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

# Evaluation Suite: t_eval
# Description: T-Eval (Shanghai AI Lab 6 Separable Sub-Skills for Tool-Augmented LLMs)

"""gbench native built-in runner for t_eval (Tool Use & Function Calling).

Canonical T-Eval (Chen et al., ACL 2024; arXiv:2312.14033; github.com/open-compass/T-Eval;
HF `lovesnowbest/T-Eval`) measures SIX separable sub-skills over 8 data files, in the two
settings it publishes: English (the headline) and Chinese (the `_zh` files, `overall_zh`). Each
sample is a pre-baked single-turn context; the model emits ONE completion scored against
ground_truth by the authors' own deterministic evaluators (NO LLM judge). The upstream
evaluators are vendored verbatim in `_teval_vendor/` and driven per sample here; the six
dimensions are combined exactly as upstream `convert_results.py`:

  Instruct  = mean([(json_format+json_args_em)/2, (string_format+string_args_em)/2])
  Plan      = mean([plan_str.f1, plan_json.f1])                (bertscore graph match + LIS)
  Reason    = mean([reason_str.thought, rru_json.thought])     (sentence-embedding cosine)
  Retrieve  = mean([retrieve_str.name, rru_json.name])
  Understand= mean([understand_str.args, rru_json.args])
  Review    = review_str.review_quality
  Overall   = mean of the six  (headline `accuracy`)

Plan and Reason score with a sentence-transformers embedder chosen per setting as test.py does
(`if "_zh" in args.dataset_path: bert_score_model = "thenlper/gte-large-zh"`, else
`all-mpnet-base-v2`, passed to PlanningEvaluator / ReasonRetrieveUnderstandEvaluator): the
English files use `sentence-transformers/all-mpnet-base-v2` (~440 MB) and the Chinese files
`thenlper/gte-large-zh` (~650 MB), each pinned to a Hub revision, downloaded once and shared across
that setting's evaluators (GPU if it has room, else CPU). No API key, no judge, no server.
The vendored evaluators need sentence-transformers and networkx; they are imported when the suite
runs (checked before any request, infra_required naming a missing one), never when this module is
imported, so listing or running other suites does not need them.

Every reply goes through test.py's `split_special_tokens` (cut at a leaked chat-template marker,
strip a leading ```json fence and surrounding backticks) before it is scored, as upstream does
with every prediction.

Sampling: temperature 0.0 without `--thinking`, 1.0 with it (gbench's rule for every suite,
base.resolve_temperature; test.py generates greedily, `llm.chat(..., do_sample=False)`), and
max_tokens=512 on every request of a no-think run: test.py builds a
Hugging Face model, as an open-weight model runs there (`--model_type hf`), as
`HFTransformerCasualLM(..., max_new_tokens=512)`. A `--thinking` run does not apply that cap (it
was written for non-reasoning generation): each request gets the run's --max-output-tokens, else
the thinking floor, and the result records the cap as not applied (base.resolve_token_budget). A
reply cut off by any other budget (an empty reply after reasoning included) is counted and makes
the run not leaderboard-comparable. Override the temperature for a whole run with
`--temperature`, or for this suite with `GBENCH_T_EVAL_TEMPERATURE`, which takes precedence; the
result records upstream's setting under `official_sampling`.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import (OfficialSampling, answer_in_thinking_fields, is_scored, run_eval_suite,
                   strip_thinking_tags)
from .fc_common import eval_limit_plan, run_attempts
from .sampling import allocate_limit, parse_shard, shard_select, stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Function Calling"
DOCS_URL = "docs/evals/t_eval.md"

_HF_REPO = "lovesnowbest/T-Eval"
#: HF revision of the data files (2024-02-18, the v2 data the T-Eval repo evaluates).
_HF_REVISION = "af355ab2b62cdbae4262ac41c7529ffeae395012"
#: open-compass/T-Eval commit the `_teval_vendor/` evaluators match (HEAD, 2024-04-03; they differ
#: only in import shims).
TEVAL_COMMIT = "58f22406404d7e2a4f36856a19c7f4dc28a0a5f0"
#: test.py's bert_score_model per setting, as (Hub repo, revision). test.py names the English one
#: "all-mpnet-base-v2", which sentence-transformers resolves to the repo below; the revisions are
#: the repos' HEAD commits, pinned.
BERT_MODELS: Dict[str, Tuple[str, str]] = {
    "en": ("sentence-transformers/all-mpnet-base-v2", "e8c3b32edf5434bc2275fc9bab85f82640a19130"),
    "zh": ("thenlper/gte-large-zh", "64c364e579de308104a9b2c170ca009502f4f545"),
}

#: test.py infer(): `llm.chat(batch_infer_list, do_sample=False)`, greedy decoding, by the model
#: test.py builds for `--model_type hf` (line 115: HFTransformerCasualLM(path, meta_template,
#: max_new_tokens=512)). The token-limit policy (base.resolve_token_budget) sends the 512 exactly
#: on a no-think run and does not apply it to a --thinking run; T-Eval defines no reasoning
#: settings.
UPSTREAM_MAX_TOKENS = 512
T_EVAL_SAMPLING = OfficialSampling(
    reference=(f"open-compass/T-Eval@{TEVAL_COMMIT[:12]} test.py infer (llm.chat(..., "
               "do_sample=False); HFTransformerCasualLM max_new_tokens=512)"),
    temperature=0.0,
    max_tokens=UPSTREAM_MAX_TOKENS,
)

# (dimension key, data file stem). All 8 English files count toward the headline; the Chinese
# setting runs the same 8 with the LANGUAGES suffix.
_FILES: List[Tuple[str, str]] = [
    ("instruct", "instruct_v2"),
    ("plan_str", "plan_str_v2"),
    ("plan_json", "plan_json_v2"),
    ("reason_str", "reason_str_v2"),
    ("retrieve_str", "retrieve_str_v2"),
    ("understand_str", "understand_str_v2"),
    ("rru_json", "reason_retrieve_understand_json_v2"),
    ("review_str", "review_str_v2"),
]
#: The two settings T-Eval publishes (its EN and ZH leaderboards): the same 8 files, the Chinese
#: ones with a `_zh` suffix (test_all_en.sh / test_all_zh.sh).
LANGUAGES = {"en": "", "zh": "_zh"}

#: The Python packages the vendored evaluators need that gbench's base install does not carry:
#: sentence-transformers (the Plan/Reason embedder and its cosine util) and networkx (imported at
#: module load by the planning and reason/retrieve/understand evaluators). run_t_eval checks them
#: before any request and hard-errors (infra_required) naming the missing ones.
REQUIRED_PACKAGES: Tuple[Tuple[str, str], ...] = (
    ("sentence_transformers", "sentence-transformers"),
    ("networkx", "networkx"),
)

#: The vendored evaluator package (`_teval_vendor`), imported on first use by _vendor(), never at
#: module load: `import gbench.runners.eval_suites` (every `gbench --list evals` and every --evals
#: run, whichever suites it selects) must work without the packages above.
_VENDOR: Optional[Any] = None


def _importable(module: str) -> bool:
    """True when `module` can be imported (find_spec finds it). A finder that raises, or a
    module blocked with a None entry in sys.modules, counts as missing."""
    import importlib.util
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def missing_packages() -> List[str]:
    """The pip names of REQUIRED_PACKAGES that cannot be imported, in order."""
    return [pip for module, pip in REQUIRED_PACKAGES if not _importable(module)]


def _unscored_text(setting: str, unscored: List[str], empty: List[str]) -> str:
    """Why a setting's files have no metrics: files with no sample in this shard / limit, and
    files whose every request got no reply from the served endpoint (harness errors)."""
    parts = []
    shard_empty = [d for d in unscored if d in empty]
    failed = [d for d in unscored if d not in empty]
    if shard_empty:
        parts.append(f"no sample of the {setting} file(s) {', '.join(shard_empty)} in this shard "
                     "or limit")
    if failed:
        parts.append(f"no scored reply in the {setting} file(s) {', '.join(failed)} (every "
                     "request got no reply from the served endpoint: harness errors)")
    return "; ".join(parts) or f"no scored reply in the {setting} files"


def _vendor():
    """The vendored T-Eval evaluator package, imported once. The evaluators load a
    SentenceTransformer in __init__; that is neutered so construction is cheap, and _score_file
    injects the setting's shared model into the evaluators that need it. A module the vendored
    files import that is not installed raises infra_required naming it."""
    global _VENDOR
    if _VENDOR is None:
        try:
            from gbench.third_party import _teval_vendor as vendor
            from gbench.third_party._teval_vendor import planning_evaluator as pe_mod
            from gbench.third_party._teval_vendor import (
                reason_retrieve_understand_evaluator as rru_mod,
            )
        except ModuleNotFoundError as e:
            from .swebench_common import infra_required
            raise infra_required(
                "t_eval",
                f"the vendored T-Eval evaluators import the Python package "
                f"'{e.name or e}', which is not installed (pip install sentence-transformers "
                "networkx).",
                DOCS_URL,
            ) from e
        except (ImportError, OSError, RuntimeError) as e:
            # installed but broken: a torch or sentence-transformers build whose import fails
            # (a missing shared library, an ABI mismatch)
            from .swebench_common import infra_required
            raise infra_required(
                "t_eval",
                f"the vendored T-Eval evaluators could not be imported: "
                f"{getattr(e, 'name', None) or 'an import'} failed ({type(e).__name__}: "
                f"{str(e)[:200]}); reinstall torch and sentence-transformers.",
                DOCS_URL,
            ) from e
        pe_mod.SentenceTransformer = lambda *a, **k: None
        rru_mod.SentenceTransformer = lambda *a, **k: None
        _VENDOR = vendor
    return _VENDOR


#: language -> the loaded embedder (one per setting, shared by its Plan + Reason scorers).
_SHARED_ST: Dict[str, Any] = {}
#: language -> {model, revision, device, fallback}: the device each setting's embedder ran on
#: (the result's `t_eval_embedder`).
_ST_INFO: Dict[str, Dict[str, Any]] = {}


def _shared_st(language: str = "en"):
    """Lazily load the setting's embedder (BERT_MODELS[language]) shared by Plan + Reason scorers.

    Uses the GPU when torch sees one (test.py's default device is cuda), else the CPU; when the
    GPU load fails (a full vLLM server owns its VRAM, gpu-memory-utilization 0.95, so
    `torch.cuda.is_available()` is True yet the load OOMs) it falls back to the CPU. The device
    can change a score: the plan graph matching (networkx max_weight_matching over BERTScore
    similarities) breaks ties on float noise, which differs between CPU and GPU kernels (on the
    pinned data one English Plan reply scores 0.4444 on GPU and 0.6667 on CPU). The device and
    whether it was a fallback are recorded (_ST_INFO, the result's `t_eval_embedder`), and a run
    whose embedder did not run on the GPU names it in leaderboard_comparable_reason. Force a
    device with GBENCH_T_EVAL_DEVICE=cpu|cuda."""
    if language not in _SHARED_ST:
        from sentence_transformers import SentenceTransformer
        name, revision = BERT_MODELS[language]
        forced = os.environ.get("GBENCH_T_EVAL_DEVICE", "").strip().lower()
        if forced in ("cpu", "cuda"):
            device = forced
        else:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        from .swebench_common import infra_required

        def _unavailable(e: Exception) -> RuntimeError:
            return infra_required(
                "t_eval", f"the {language} setting's embedder {name}@{revision} (test.py's "
                f"bert_score_model) could not be loaded ({type(e).__name__}: {str(e)[:200]})",
                DOCS_URL)

        logger.info("t_eval: loading %s@%s on %s (one-time)", name, revision[:12], device)
        fallback = False
        try:
            _SHARED_ST[language] = SentenceTransformer(name, device=device, revision=revision)
        except Exception as e:  # CUDA OOM etc.: the served model owns the VRAM -> use CPU
            if device == "cuda" and forced != "cuda":
                logger.warning("t_eval: GPU load of %s failed (%s: %s) - the served model likely "
                               "owns all VRAM; falling back to CPU (slower, and plan-matching "
                               "ties can resolve differently than on the GPU).",
                               name, type(e).__name__, str(e)[:120])
                try:
                    _SHARED_ST[language] = SentenceTransformer(name, device="cpu",
                                                               revision=revision)
                except Exception as e2:
                    raise _unavailable(e2) from e2
                device, fallback = "cpu", True
            else:
                raise _unavailable(e) from e
        _ST_INFO[language] = {"model": name, "revision": revision, "device": device,
                              "fallback": fallback,
                              "forced": forced if forced in ("cpu", "cuda") else None}
    return _SHARED_ST[language]


def embedders_in_use() -> Dict[str, Dict[str, Any]]:
    """The record (_ST_INFO) of each setting's embedder that is loaded (in _SHARED_ST), the one
    this process scores with, by language."""
    return {lang: dict(rec) for lang, rec in sorted(_ST_INFO.items()) if lang in _SHARED_ST}


def embedder_device_reasons(info: Dict[str, Dict[str, Any]]) -> List[str]:
    """The leaderboard_comparable_reason entries for the settings whose embedder did not run on
    the GPU (test.py's default device): plan-matching ties can resolve differently there."""
    out = []
    for language, rec in sorted((info or {}).items()):
        if rec.get("device") == "cuda":
            continue
        why = ("the GPU load failed, fallback" if rec.get("fallback") else
               "GBENCH_T_EVAL_DEVICE=cpu" if rec.get("forced") == "cpu" else "no GPU visible")
        out.append(f"the {language} setting's Plan/Reason embedder ran on {rec.get('device')} "
                   f"({why}): scores can differ from a GPU run, as test.py runs it, where "
                   "plan-matching ties resolve differently")
    return out


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
_ROLE_MAP = {"system": "system", "user": "user", "assistant": "assistant", "function": "user"}


def _to_chat(origin_prompt: Any) -> List[Dict[str, str]]:
    """Map a T-Eval origin_prompt to a chat message list, one message per turn as upstream's meta
    templates render them: `function` turns go out as `user` turns (the `api_role` the templates
    give them); consecutive turns of one role stay separate messages."""
    if not isinstance(origin_prompt, list):
        return [{"role": "user", "content": str(origin_prompt)}]
    msgs = [{"role": _ROLE_MAP.get(turn.get("role", "user"), "user"),
             "content": turn.get("content", "") or ""}
            for turn in origin_prompt if isinstance(turn, dict)]
    return msgs or [{"role": "user", "content": ""}]


def file_limits(limit: Optional[int]) -> Dict[str, Optional[int]]:
    """`--eval-limit N` over the 16 files (key: the dimension, `<dim>_zh` for Chinese), N in
    total (sampling.allocate_limit, seeded on the suite). No limit: every file is None (all of
    its rows)."""
    keys = [dim if lang == "en" else f"{dim}_{lang}" for lang in LANGUAGES for dim, _ in _FILES]
    return allocate_limit(limit, keys, seed="t_eval")


def _load_file_samples(
    dim: str, stem: str, limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load one T-Eval file; carry the WHOLE sample (minus origin_prompt) in the gold JSON so
    the vendored evaluator gets the exact keys it needs (template / meta / meta_data / gt)."""
    from huggingface_hub import hf_hub_download
    from .swebench_common import infra_required
    try:
        path = hf_hub_download(_HF_REPO, filename=f"data/{stem}.json", repo_type="dataset",
                               revision=_HF_REVISION)
    except Exception as e:
        raise infra_required("t_eval", f"{_HF_REPO}@{_HF_REVISION} data/{stem}.json could not be "
                             f"downloaded ({e}).", "docs/evals/t_eval.md") from e
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = list(data.values()) if isinstance(data, dict) else list(data)
    if not rows:
        raise RuntimeError(f"t_eval: {stem}.json returned empty rows")
    rows = stratified_sample(rows, limit, None, seed=f"t_eval:{dim}")

    samples = []
    for row in rows:
        msgs = _to_chat(row.get("origin_prompt"))
        sample = {k: v for k, v in row.items() if k != "origin_prompt"}
        gold = json.dumps({"dimension": dim, "sample": sample})
        samples.append((msgs, gold, {"category": f"t_eval::{dim}"}))
    return samples


# --------------------------------------------------------------------------- #
# Scoring - drive each vendored evaluator per sample, return the file's aggregate.
# --------------------------------------------------------------------------- #
def split_special_tokens(text: str) -> str:
    """test.py split_special_tokens, verbatim: applied to every prediction before scoring."""
    text = text.split('<eoa>')[0]
    text = text.split('<TOKENS_UNUSED_1>')[0]
    text = text.split('<|im_end|>')[0]
    text = text.split('\nuser')[0]
    text = text.split('\nassistant')[0]
    text = text.split('\nUSER')[0]
    text = text.split('[INST]')[0]
    text = text.split('<|user|>')[0]
    text = text.strip()
    if text.startswith('```json'):
        text = text[len('```json'):]
    text = text.strip('`').strip()
    return text


def _datum(trace: Dict[str, Any]) -> Dict[str, Any]:
    """The evaluator's input for one trace: the dataset sample plus the prediction, with any
    inline reasoning (`<thought>...</thought>` etc.) removed as every other suite scores it, then
    post-processed by test.py's split_special_tokens as upstream does."""
    try:
        blob = json.loads(trace.get("gold_answer") or "{}")
    except Exception:
        blob = {}
    d = dict(blob.get("sample") or {})
    d["prediction"] = split_special_tokens(strip_thinking_tags(trace.get("response_text") or ""))
    return d


def _score_file(dim: str, traces: List[Dict[str, Any]], language: str = "en") -> Dict[str, float]:
    """Aggregate metrics for one file, storing each per-sample metric back on its trace. The
    setting's embedder (BERT_MODELS[language]) scores Plan and Reason, as test.py configures it."""
    vendor = _vendor()
    if dim == "instruct":
        ev = vendor.InstructEvaluator(dataset_path=None)
        results = []
        for t in traces:
            m = ev._evaluate(ev._process_response(_datum(t)))
            t["t_eval_metric"] = m
            results.append(m)
        agg = ev._post_process(results)

    elif dim in ("plan_str", "plan_json"):
        ev = vendor.PlanningEvaluator(dataset_path=None)
        ev.sentence_model = _shared_st(language)
        results = []
        for t in traces:
            ds, _err = ev._process_response(_datum(t))
            m = ev._evaluate(ds)
            t["t_eval_metric"] = m
            results.append(m)
        agg = ev._post_process(results)

    elif dim in ("reason_str", "retrieve_str", "understand_str", "rru_json"):
        eval_type = {"reason_str": "reason", "retrieve_str": "retrieve",
                     "understand_str": "understand", "rru_json": "reason"}[dim]
        prompt_type = "json" if dim == "rru_json" else "str"
        ev = vendor.ReasonRetrieveUnderstandEvaluator(
            dataset_path=None, default_prompt_type=prompt_type, eval_type=eval_type)
        if dim in ("reason_str", "rru_json"):  # thought cosine needs the embedder
            ev.sentence_model = _shared_st(language)
        data_samples = []
        for t in traces:
            ds, _err = ev._process_response(_datum(t))
            data_samples.append(ev._evaluate(ds))  # main class scores in _post_process
        agg = ev._post_process(data_samples)

    elif dim == "review_str":
        ev = vendor.ReviewEvaluator(dataset_path=None)
        results = []
        for t in traces:
            m = ev._evaluate(ev._process_response(_datum(t)))
            t["t_eval_metric"] = m
            results.append(m)
        agg = ev._post_process(results)
    else:
        raise ValueError(f"t_eval: unknown dimension {dim}")

    return {k: float(v) for k, v in dict(agg).items()}


def _f(d: Dict[str, float], k: str) -> float:
    try:
        return float(d.get(k, 0.0))
    except Exception:
        return 0.0


def _safe_mean(vals: List[float]) -> float:
    vs = [v for v in vals if v == v]  # drop NaN (empty response_format group at tiny --eval-limit)
    return float(np.mean(vs)) if vs else 0.0


#: Per-file run_eval_suite counters that the suite result sums over the 16 files.
_SUMMED_COUNTERS = ("failed_requests", "empty_responses", "truncated_responses",
                    "non_convergent_responses", "genuinely_truncated", "over_context_prompts",
                    "timed_out_requests", "looping_responses", "discarded_outputs",
                    "judge_failures", "judge_outages", "scoring_excluded",
                    "replies_at_official_max_tokens", "answer_in_thinking")


def _merge_file_results(results: List[Dict[str, Any]],
                        labels: Optional[List[str]] = None) -> Dict[str, Any]:
    """One suite-level result from the 16 per-file run_eval_suite results: counters and response
    health summed, token budgets unioned, durations added, and the status recomputed with
    base's rule over the summed counts (replies that stopped at the official 512 are the
    protocol's outcome, not truncations)."""
    merged = dict(results[0]) if results else {}
    for key in _SUMMED_COUNTERS:
        merged[key] = sum(int(r.get(key) or 0) for r in results)
    health: Dict[str, int] = {}
    for r in results:
        for k, v in (r.get("response_health") or {}).items():
            health[k] = health.get(k, 0) + int(v or 0)
    merged["response_health"] = health
    budgets = sorted({b for r in results for b in (r.get("max_tokens_sent") or [])})
    merged["max_tokens_sent"] = budgets or None
    merged["duration_s"] = round(sum(float(r.get("duration_s") or 0.0) for r in results), 2)
    total = sum(int(r.get("total_questions") or 0) for r in results)
    merged["total_questions"] = total
    merged.update(answer_in_thinking_fields(merged["answer_in_thinking"], total))
    kinds: Dict[str, int] = {}
    for r in results:
        for k, v in (r.get("request_harness_errors") or {}).items():
            kinds[k] = kinds.get(k, 0) + int(v or 0)
    merged.pop("request_harness_errors", None)
    if kinds:
        merged["request_harness_errors"] = dict(sorted(kinds.items()))
    failed = merged["failed_requests"]
    # base's scored_count, summed: every sample less the ones left out of the score (requests
    # that got no reply from the endpoint among them) and the judge outages.
    scored = total - merged["scoring_excluded"] - merged["judge_outages"]
    if total == 0:
        status = "error"
    elif failed == total or merged["judge_outages"] >= total or scored <= 0:
        status = "failed"
    elif (failed == 0 and merged["judge_outages"] == 0 and not merged["scoring_excluded"]
          and max(0, merged["truncated_responses"]
                  - merged["replies_at_official_max_tokens"]) * 10 <= total):
        status = "success"
    else:
        status = "completed_with_errors"
    merged["status"] = status
    # What made the run other than success: each such file's own status_reason (base's), by
    # file; None on success.
    merged["status_reason"] = None if status == "success" else "; ".join(
        f"{label}: {r.get('status_reason') or r.get('status')}"
        for label, r in zip(labels or [str(i) for i in range(len(results))], results)
        if r.get("status") != "success") or status
    merged.pop("accuracy_withheld_reason", None)
    return merged


def replies_cut_off(traces: List[Dict[str, Any]], thinking: bool) -> int:
    """Replies stopped by a budget before they finished (finish reason length, an empty reply
    after reasoning included) other than at test.py's 512: with --thinking every such stop (the
    run's budget), without it a stop below the 512 (a served context window that left less)."""
    return sum(1 for t in traces if t.get("finish_reason") == "length"
               and (thinking or t.get("max_tokens_sent") != UPSTREAM_MAX_TOKENS))


def _overall(fr: Dict[str, Optional[Dict[str, float]]]
             ) -> Tuple[Optional[float], Dict[str, float]]:
    """convert_results.py for one language: (Overall, the six dimensions), as fractions.

    (None, {}) when a file has no scored reply (every request of it got no reply from the served
    endpoint: harness errors): its dimensions, and so the Overall, have no value."""
    if any(fr.get(dim) is None for dim, _ in _FILES):
        return None, {}
    instruct = _safe_mean([
        (_f(fr["instruct"], "json_format_metric") + _f(fr["instruct"], "json_args_em_metric")) / 2,
        (_f(fr["instruct"], "string_format_metric") + _f(fr["instruct"], "string_args_em_metric")) / 2,
    ])
    plan = _safe_mean([_f(fr["plan_str"], "f1_score"), _f(fr["plan_json"], "f1_score")])
    reason = _safe_mean([_f(fr["reason_str"], "thought"), _f(fr["rru_json"], "thought")])
    retrieve = _safe_mean([_f(fr["retrieve_str"], "name"), _f(fr["rru_json"], "name")])
    understand = _safe_mean([_f(fr["understand_str"], "args"), _f(fr["rru_json"], "args")])
    review = _f(fr["review_str"], "review_quality")
    dims = {"instruct": instruct, "plan": plan, "reason": reason, "retrieve": retrieve,
            "understand": understand, "review": review}
    return _safe_mean(list(dims.values())), dims


def run_t_eval(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run all 6 T-Eval dimensions in both published settings (the 8 English files and the 8
    Chinese ones); the English Overall is the headline `accuracy`, the Chinese one `overall_zh`."""
    from .swebench_common import infra_required
    missing = missing_packages()
    if missing:
        raise infra_required(
            "t_eval",
            f"requires {' and '.join(missing)} (not installed): the vendored T-Eval evaluators "
            "need sentence-transformers (+torch) for the canonical BERTScore semantic matching "
            "used by the Plan/Reason/Retrieve/Understand scorers and networkx for the plan graph "
            "matching (pip install sentence-transformers networkx).",
            DOCS_URL,
        )
    _vendor()   # import the evaluators now, so an import failure stops the run before any request

    limit = kwargs.get("limit")

    file_metrics: Dict[str, Dict[str, Optional[Dict[str, float]]]] = {
        lang: {} for lang in LANGUAGES}
    # files with no sample in this shard / limit (versus files whose every request failed)
    file_empty: Dict[str, List[str]] = {lang: [] for lang in LANGUAGES}
    all_traces: List[Dict[str, Any]] = []
    file_results: List[Dict[str, Any]] = []
    file_keys: List[str] = []
    file_status: Dict[str, str] = {}

    # --eval-limit N is N samples for the suite, never N (or one) per file: the N are dealt over
    # the 16 files (8 per setting) round-robin in a seeded order, one per file before any gets
    # two. A file that gets 0 is not run, and a setting missing a dimension has no Overall.
    per_file = file_limits(limit)
    attempts = run_attempts(kwargs)
    limited = bool(limit and int(limit) > 0)
    # The plan counts the rows base runs on this node: each file's drawn rows, then
    # GBENCH_SHARD's shard of them (base shards each file's list), so the shards' plans add up.
    shard = parse_shard(os.environ.get("GBENCH_SHARD"))
    on_shard = {k: (None if n is None else len(shard_select(range(n), shard)))
                for k, n in per_file.items()}
    limit_plan = eval_limit_plan(
        "t_eval", limit, sum(on_shard.values()) if limited else None,
        per_setting=on_shard, attempts=attempts, setting_word="file")

    for lang in LANGUAGES:
        for dim, stem in _FILES:
            stem = stem + LANGUAGES[lang]
            key = dim if lang == "en" else f"{dim}_{lang}"
            if per_file[key] == 0:
                # not drawn in this limited run: not run, and named as such
                file_empty[lang].append(dim)
                file_metrics[lang][dim] = None
                file_status[key] = "not_drawn"
                logger.info("t_eval: [%s] not drawn by --eval-limit %s, not run", key, limit)
                continue
            samples = _load_file_samples(dim, stem, per_file[key])
            logger.info("t_eval: [%s] %d samples", key, len(samples))
            res = run_eval_suite(
                eval_name="t_eval",
                model_name=model_name,
                base_url=base_url,
                concurrency=concurrency,
                samples=samples,
                eval_fn=lambda pred, gold: False,  # no per-sample verdict; the metric is below
                thinking=enable_thinking,
                extra_payload=kwargs.get("extra_payload"),
                limit=None,  # already limited per file in the loader
                max_output_tokens=kwargs.get("max_output_tokens"),
                official_sampling=T_EVAL_SAMPLING,
            )
            file_results.append(res)
            file_keys.append(key)
            file_status[key] = res.get("status", "success")
            traces = res.get("sample_traces", []) or []
            for t in traces:
                t["t_eval_dimension"] = dim
                t["t_eval_language"] = lang
            all_traces.extend(traces)
            # Over the scored replies, as base's accuracy: a request that got no reply from the
            # served endpoint is a harness error (scoring_excluded), never scored as an empty
            # prediction. A file with none scored has no metrics (None).
            scored = [t for t in traces if is_scored(t)]
            if not traces:
                file_empty[lang].append(dim)
            file_metrics[lang][dim] = (_score_file(dim, scored, language=lang) if scored
                                       else None)
            logger.info("t_eval: [%s] metrics=%s", key, file_metrics[lang][dim])

    overall, dims = _overall(file_metrics["en"])
    overall_zh, dims_zh = _overall(file_metrics["zh"])
    # The scored replies of each setting (the requests with no reply are left out).
    n_en = sum(1 for t in all_traces if t.get("t_eval_language") == "en" and is_scored(t))
    n_zh = sum(1 for t in all_traces if t.get("t_eval_language") == "zh" and is_scored(t))

    result = _merge_file_results(file_results, file_keys)
    result["benchmark_type"] = "eval"
    result["eval_name"] = "t_eval"
    result["model_name"] = model_name
    result["file_status"] = file_status
    result["accuracy"] = round(overall * 100, 2) if overall is not None else None
    result["overall_zh"] = round(overall_zh * 100, 2) if overall_zh is not None else None
    unscored = {lang: [dim for dim, _ in _FILES if file_metrics[lang].get(dim) is None]
                for lang in LANGUAGES}
    if overall is None:
        result["accuracy_withheld_reason"] = (
            _unscored_text("English", unscored["en"], file_empty["en"])
            + ", so the Overall has no value"
            + (": " + str(result.get("status_reason"))
               if result.get("status_reason") and set(unscored["en"]) - set(file_empty["en"])
               else ""))
    if overall_zh is None:
        result["overall_zh_withheld_reason"] = (
            _unscored_text("Chinese", unscored["zh"], file_empty["zh"])
            + ", so overall_zh has no value")
    # T-Eval scores are continuous (no per-sample right/wrong), so there is no per-category
    # correct/total to report: the six dimensions are in dimension_scores(_zh). correct_answers
    # weighs each setting's Overall by its sample count, so pillar subtotals weigh the suite by
    # its size.
    result["category_accuracy"] = {}
    result["correct_answers"] = round((overall or 0.0) * n_en + (overall_zh or 0.0) * n_zh)
    result["dimension_scores"] = {k: round(v * 100, 2) for k, v in dims.items()}
    result["dimension_scores_zh"] = {k: round(v * 100, 2) for k, v in dims_zh.items()}
    result["t_eval_file_metrics"] = file_metrics["en"]
    result["t_eval_file_metrics_zh"] = file_metrics["zh"]
    result["metric"] = ("T-Eval Overall (English) = mean of 6 dimensions (Instruct/Plan/Reason/"
                        "Retrieve/Understand/Review), str & json weighted equally; canonical "
                        "convert_results. The Chinese setting's Overall is overall_zh")
    result["dataset_revision"] = (f"{_HF_REPO}@{_HF_REVISION} (data/*_v2.json and "
                                  f"data/*_v2_zh.json); evaluators open-compass/T-Eval@{TEVAL_COMMIT}")
    result["embedding_models"] = {lang: f"{name}@{rev}" for lang, (name, rev) in BERT_MODELS.items()}
    # the device each setting's embedder ran on (a CPU run can score plan ties differently)
    result["t_eval_embedder"] = embedders_in_use()
    result["sample_traces"] = all_traces
    result["eval_limit_plan"] = limit_plan
    result["replies_cut_off"] = replies_cut_off(all_traces, bool(enable_thinking))
    reasons = []
    if limit:
        reasons.append(f"--eval-limit {limit} ({sum(per_file.values())} samples over the 16 "
                       f"files in total; files not drawn: "
                       f"{', '.join(k for k, n in per_file.items() if not n) or 'none'})")
    if parse_shard(os.environ.get("GBENCH_SHARD")):
        reasons.append("sharded run")
    if (result.get("official_sampling") or {}).get("temperature_matches_official") is False:
        reasons.append(f"temperature {result.get('temperature')} (test.py decodes greedily)")
    if result.get("decoding_penalties"):
        reasons.append("decoding penalties set")
    if ((result.get("attempts") or {}).get("attempts_per_sample") or 1) > 1:
        reasons.append("--attempt-count > 1 (metrics pool every attempt)")
    no_reply = sum(1 for t in all_traces if t.get("harness_error"))
    if no_reply:
        reasons.append(f"{no_reply} request(s) got no reply from the served endpoint (failed, "
                       "timed out or over the context window: harness errors, left out of the "
                       "score)")
    if result["replies_cut_off"]:
        reasons.append(f"{result['replies_cut_off']} replies cut off by the token budget before "
                       f"they finished (not at test.py's max_new_tokens {UPSTREAM_MAX_TOKENS})")
    reasons.extend(embedder_device_reasons(result["t_eval_embedder"]))
    result["leaderboard_comparable"] = not reasons
    result["leaderboard_comparable_reason"] = "; ".join(reasons) or None
    return result
