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

"""Native RULER long-context evaluation suite (Hsieh et al., NVIDIA, COLM 2024).

Canonical RULER is a tokenizer-defined DATA-GENERATION benchmark: 13 tasks x 6 context
bands (4k/8k/16k/32k/64k/128k), 500 samples each, with haystacks padded to exact token
counts under the TARGET model's own tokenizer. This runner drives NVIDIA/RULER's own
`prepare.py` (from a checkout at GBENCH_RULER_DIR, pinned to _RULER_REVISION and verified at
run time) to generate the per-tokenizer data once (cached to disk, reused on later runs),
sends each prompt single-shot, and scores with the canonical metrics: string_match_all
(niah/vt/cwe/fwe, fraction of references present) and string_match_part (qa, any reference
present), after RULER's postprocess_pred. The headline `accuracy` is the canonical Avg = mean
over bands of (mean of the 13 task scores); per-band, per-task, per-cell scores +
effective_length + wAvg are also reported.

Protocol details that follow upstream exactly:
  * The band is RULER's max_seq_length: chat template + prompt + answer prefix + the task's
    generation cap all fit inside it. The server applies the model's chat template, so the data
    is generated with RULER's `base` template and the chat template's token count is subtracted
    from the band (prepare.py's `--prepare_for_ns` path, fed the served model's template), so
    the 128k data fits a 131072-token window.
  * Each prompt is `input + answer_prefix` in one user turn, as RULER's chat-API clients send
    it (scripts/pred/call_api.py + client_wrappers.OpenAIClient, `base` template). RULER's
    open-model path (a chat template in scripts/data/template.py) instead puts the answer
    prefix after the assistant header, as a prefill. gbench's chat requests end with the user
    turn and a generation prompt (base sends no continue_final_message, and its /tokenize count
    assumes add_generation_prompt), so it follows the chat-API path.

The suite hard-errors (does not silently skip) if the server reports a max_model_len below the
largest band, and whenever a prompt would not be sent: RULER data is built to fit, so a prompt
the harness's pre-send check refuses (OVER_CONTEXT) is a serving or counting mismatch, never a
model answer, and is never scored as 0. The check needs min(256, cap) tokens of answer room and
counts each prompt exactly with the server's /tokenize; the longest prompt (by RULER's token
count) of every task@band cell is checked before anything is sent, and any OVER_CONTEXT trace
after the run also raises.
Generation is greedy with RULER's per-task token caps (each sample's own max_tokens), sent as
declared without `--thinking`. RULER defines no reasoning-model settings, so with `--thinking`
those caps, written for non-reasoning generation, are not applied: base's token-limit policy
sends each request the run's `--max-output-tokens` (else the thinking floor) and records the
lifted caps in `official_cap_not_applied_to_reasoning` and `max_output_tokens_note`.

Sampling: RULER's settings (scripts/config_models.sh TEMPERATURE="0.0", TOP_P="1.0"; the
chat-API client in scripts/pred/client_wrappers.py sends temperature, top_p and seed, the
`--random_seed` default 0 of call_api.py) are recorded (base.OfficialSampling); top_p and seed
are sent in both modes, and the temperature follows gbench's rule (base.resolve_temperature):
1.0 with `--thinking`, 0.0 without. Override the
temperature for a whole run with `--temperature`, or for
this suite alone with `GBENCH_RULER_TEMPERATURE`, which takes precedence over both; the result
records `official_sampling`. Only a 0.0 run is leaderboard-comparable.
"""

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import (DEFAULT_MIN_OUTPUT_TOKENS, SUITE_MIN_OUTPUT_TOKENS, clamp_to_context,
                   effective_chat_template_kwargs, gbench_cache_dir, get_run_knob, run_eval_suite,
                   thinking_run_budget, OfficialSampling)
from .metrics import record_budget_cutoffs
from .limit_plan import base_attempts, eval_limit_plan
from .sampling import allocate_limit, parse_shard, shard_select, stratified_sample
from .swebench_common import infra_required, prereqs_path
from .mrcr import _get_server_max_model_len

logger = logging.getLogger(__name__)

#: RULER @ c3f5e3b4: scripts/config_models.sh TEMPERATURE="0.0", TOP_P="1.0" (TOP_K="32" is not
#: sent by the chat-API client), and call_api.py `--random_seed` default 0, which that client
#: sends as `seed`.
OFFICIAL_SAMPLING = OfficialSampling(
    reference="NVIDIA/RULER scripts/config_models.sh + pred/call_api.py @c3f5e3b4 "
              "(temperature 0.0, top_p 1.0, seed 0)",
    temperature=0.0, params={"top_p": 1.0, "seed": 0})

PILLAR = "Long Context"
DOCS_URL = "docs/evals/ruler.md"

_ENV = "GBENCH_RULER_DIR"
#: NVIDIA/RULER main as of 2026-09-24. The data generators live in the checkout, so a different
#: commit can generate different haystacks; the checkout is verified against this at run time.
_RULER_REVISION = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
#: Name under which the served model's chat template is registered as a RULER model template.
_TEMPLATE_NAME = "gbench-served-chat"

# The 13 headline tasks (RULER scripts/synthetic.yaml).
_TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe", "qa_1", "qa_2",
]
_DEFAULT_LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072]
# Canonical per-base-type generation caps (RULER scripts/data/synthetic/constants.py).
_GEN_CAP = {"niah": 128, "variable_tracking": 30,
            "common_words_extraction": 120, "freq_words_extraction": 50, "qa": 32}
_EFF_THRESHOLD = 85.6  # RULER "effective length": largest band scoring >= 85.6 (0.856)


def _base_type(task: str) -> str:
    if task.startswith("niah"):
        return "niah"
    return {"vt": "variable_tracking", "cwe": "common_words_extraction",
            "fwe": "freq_words_extraction", "qa_1": "qa", "qa_2": "qa"}[task]


def _band_label(n: int) -> str:
    return f"{n // 1024}k"


# --- canonical scoring (ported from RULER scripts/eval/synthetic/constants.py + evaluate.py) ---
def _postprocess_pred(text: str) -> str:
    text = (text or "").strip()
    return re.sub(r"[\x00-\x1f]", "\n", text).strip()


def _string_match_all(pred: str, refs: List[str]) -> float:
    """Fraction of reference strings (case-insensitive) present in the prediction."""
    if not refs:
        return 0.0
    p = pred.lower()
    return sum(1.0 if str(r).lower() in p else 0.0 for r in refs) / len(refs)


def _string_match_part(pred: str, refs: List[str]) -> float:
    """1.0 if ANY reference string is present, else 0.0."""
    if not refs:
        return 0.0
    p = pred.lower()
    return max(1.0 if str(r).lower() in p else 0.0 for r in refs)


def _item_score(task: str, pred: str, refs: List[str]) -> float:
    fn = _string_match_part if _base_type(task) == "qa" else _string_match_all
    return fn(_postprocess_pred(pred), refs)


def _eval_ruler(pred: str, gold: str) -> bool:
    """Phase-1 pass/fail (all references matched); the reported metric is the mean score."""
    try:
        blob = json.loads(gold)
    except Exception:
        return False
    return _item_score(blob.get("task", ""), pred or "", blob.get("outputs") or []) >= 0.999


# --- config / prerequisites ------------------------------------------------- #
def _ruler_dir() -> str:
    d = prereqs_path("RULER", (os.environ.get(_ENV) or "").strip()) or ""
    hint = (f"Set {_ENV} to an NVIDIA/RULER checkout at the pinned commit:\n"
            "  git clone https://github.com/NVIDIA/RULER   (needs git-lfs for its *.json data)\n"
            f"  git -C RULER checkout {_RULER_REVISION}\n"
            "  run its scripts/data/synthetic/json/{download_paulgraham_essay.py,download_qa_dataset.sh}\n"
            f"  export {_ENV}=<path>/RULER")
    if not d or not os.path.isfile(os.path.join(d, "scripts", "data", "prepare.py")):
        raise infra_required("ruler", f"RULER checkout not found ({_ENV}={d!r}). {hint}", DOCS_URL)
    return d


def _verify_revision(ruler_dir: str) -> str:
    """The checkout's HEAD, which must be _RULER_REVISION with no edits to tracked files.

    The data generators run from the checkout, so its revision decides what is measured; an
    unverifiable or different checkout is a provisioning error, not something to run anyway.
    """
    fix = f"git -C {ruler_dir} fetch origin && git -C {ruler_dir} checkout {_RULER_REVISION}"
    if not shutil.which("git"):
        raise infra_required("ruler", "git not found; it is needed to verify the RULER checkout "
                             f"is at the pinned commit {_RULER_REVISION}.", DOCS_URL)
    head = subprocess.run(["git", "-C", ruler_dir, "rev-parse", "HEAD"],
                          capture_output=True, text=True)
    rev = (head.stdout or "").strip()
    if head.returncode != 0 or not rev:
        raise infra_required(
            "ruler", f"cannot read the revision of {ruler_dir} (is it a git clone of NVIDIA/RULER?): "
            f"{(head.stderr or '').strip()[-300:]}", DOCS_URL)
    if rev != _RULER_REVISION:
        raise infra_required(
            "ruler", f"RULER checkout is at {rev}, not the pinned {_RULER_REVISION}. Run: {fix}",
            DOCS_URL)
    dirty = subprocess.run(["git", "-C", ruler_dir, "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True)
    if dirty.returncode != 0 or (dirty.stdout or "").strip():
        raise infra_required(
            "ruler", f"RULER checkout {ruler_dir} has local changes to tracked files "
            f"({(dirty.stdout or dirty.stderr or '').strip()[:300]}); restore them "
            f"(git -C {ruler_dir} checkout -- .) so the pinned generators run unmodified.",
            DOCS_URL)
    return rev


def _served_chat_template(tokenizer: str, enable_thinking: bool) -> Tuple[str, int]:
    """The served model's chat template around RULER's `{task_template}` slot, and its token count.

    Registered as a RULER model template so prepare.py subtracts its tokens from the band: the
    server wraps every prompt in this template, and RULER's max_seq_length includes it (as the
    chat templates in RULER's template.py are included). The count is prepare.py's own
    arithmetic (HF `tokenize` of the template text), recorded on the result.
    """
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer)
        text = tok.apply_chat_template(
            [{"role": "user", "content": "{task_template}"}], tokenize=False,
            add_generation_prompt=True, enable_thinking=enable_thinking)
        n_tokens = len(tok.tokenize(text))
    except Exception as e:
        raise infra_required(
            "ruler", f"could not render the chat template of tokenizer {tokenizer!r} ({e}). RULER "
            "budgets the band with the served model's chat template; pass --tokenizer (or "
            "GBENCH_RULER_TOKENIZER) naming an HF tokenizer that carries it.", DOCS_URL)
    if not isinstance(text, str) or text.count("{task_template}") != 1:
        raise infra_required(
            "ruler", f"the chat template of {tokenizer!r} did not render the user turn exactly once.",
            DOCS_URL)
    return text, n_tokens


def _lengths() -> List[int]:
    raw = (os.environ.get("GBENCH_RULER_LENGTHS") or "").strip()
    if raw:
        return [int(x) for x in raw.replace(",", " ").split()]
    return list(_DEFAULT_LENGTHS)


def _require_context(base_url: str, max_band: int) -> None:
    # RULER's band already includes the chat template and the generation cap (max_seq_length is
    # "max sequence length including all input tokens and generated tokens"), so a window equal
    # to the band is enough: 131072 runs the 128k band.
    need = max_band
    max_len = _get_server_max_model_len(base_url)
    if max_len is None:
        logger.warning("ruler: could not read the server's max_model_len; ensure it covers >= %d "
                       "tokens for the %s band.", need, _band_label(max_band))
        return
    logger.info("ruler: server max_model_len=%d (need >= %d for %s band)", max_len, need, _band_label(max_band))
    if max_len < need:
        raise infra_required(
            "ruler",
            f"served max_model_len ({max_len}) is below the {_band_label(max_band)} band "
            f"(needs >= {need}). Serve the model with a larger context (e.g. vLLM "
            f"--max-model-len {need}) or restrict bands via GBENCH_RULER_LENGTHS.",
            DOCS_URL)


_FIT_HINT = ("RULER builds every band so the chat template, prompt, answer prefix and generation "
             "cap fit it, which leaves about the cap as answer room, so a prompt that is not sent "
             "is a serving or counting mismatch, not a model answer. The pre-send check counts a "
             "prompt exactly only with the server's /tokenize (vLLM), and with --thinking it needs "
             "256 tokens of room for the reasoning. Serve the model with vLLM, serve a window above "
             "the band where the model supports it, or drop the band via GBENCH_RULER_LENGTHS.")


#: The suite-level budget run_ruler passes to base when the caller gives none: RULER's largest
#: per-task cap. Every sample declares its own cap, so this only sets base's house budget.
_DEFAULT_SUITE_BUDGET = max(_GEN_CAP.values())


def _run_budget(enable_thinking: bool, max_output_tokens: Optional[int]) -> Optional[int]:
    """The per-request budget base.run_eval_suite resolves for this suite (its house budget:
    OFFICIAL_SAMPLING declares no cap), which replaces every declared per-task cap on a
    --thinking run: the run's --max-output-tokens; else the suite-level `max_output_tokens`
    raised to the floor of the mode (the thinking floor with --thinking)."""
    knob = get_run_knob("max_output_tokens")
    if knob:
        return int(knob)
    floor = (thinking_run_budget("ruler") if enable_thinking
             else max(SUITE_MIN_OUTPUT_TOKENS.get("ruler", 0), DEFAULT_MIN_OUTPUT_TOKENS))
    return max(int(max_output_tokens or 0), floor)


def _require_prompts_fit(samples: List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]],
                         enable_thinking: bool, extra_payload: Optional[Dict[str, Any]],
                         max_output_tokens: Optional[int] = None) -> None:
    """Refuse up front when base's pre-send check would not send a prompt.

    The longest prompt of every task@band cell goes through base.clamp_to_context with the
    budget, tools and chat_template_kwargs base will use for it, so a refusal here is exactly
    the OVER_CONTEXT the run would record; it is reported before any generation instead.
    `max_output_tokens` is the suite-level budget run_ruler passes to base (see _run_budget).
    "Longest" is RULER's own token count of the row (`length`, carried as `ruler_length`),
    which differs from the served count by the same template offset for every row, so the
    token-longest prompt is the one checked; character length is only the fallback for a row
    without it.
    """
    longest: Dict[str, Tuple[Tuple[int, int], List[Dict[str, Any]], Dict[str, Any]]] = {}
    for messages, _gold, meta in samples:
        cell = (meta or {}).get("category") or ""
        chars = len(str(messages[-1].get("content") or ""))
        size = (int((meta or {}).get("ruler_length") or 0), chars)
        if cell not in longest or size > longest[cell][0]:
            longest[cell] = (size, messages, meta or {})
    for cell, (_size, messages, meta) in sorted(longest.items()):
        # The request payload as base's _fetch_sample merges it (suite payload, then sample meta),
        # and the budget as base's token-limit policy resolves it: the declared cap without
        # --thinking; with it, the run's budget, larger or smaller (a cap written for
        # non-reasoning generation is not applied to reasoning).
        payload = {**(extra_payload or {}), **{k: v for k, v in meta.items() if k != "category"}}
        declared = payload.get("max_tokens")
        if enable_thinking or not (isinstance(declared, int) and declared > 0):
            budget = _run_budget(enable_thinking, max_output_tokens)
        else:
            budget = declared
        ctk = effective_chat_template_kwargs(payload.get("chat_template_kwargs"), enable_thinking)
        _max_tokens, refused = clamp_to_context(messages, budget, None, payload.get("tools"), ctk)
        if refused:
            raise infra_required(
                "ruler", f"the longest {cell} prompt would not be sent ({refused}). {_FIT_HINT}",
                DOCS_URL)


def _raise_on_unsent(traces: List[Dict[str, Any]]) -> None:
    """Any OVER_CONTEXT trace is an infrastructure error: it must not enter the scores as 0."""
    unsent = [t for t in traces if t.get("status") == "OVER_CONTEXT"]
    if not unsent:
        return
    cells: Dict[str, int] = {}
    for t in unsent:
        cells[t.get("category") or "?"] = cells.get(t.get("category") or "?", 0) + 1
    listed = ", ".join(f"{c} x{n}" for c, n in sorted(cells.items()))
    raise infra_required(
        "ruler", f"{len(unsent)} prompt(s) did not fit the served context window and were not "
        f"answered ({listed}; first: {unsent[0].get('error')}). The run is not scored. {_FIT_HINT}",
        DOCS_URL)


# --- per-tokenizer data generation (RULER prepare.py; cached + reused) ------- #
#: Runs prepare.py unmodified after registering the served chat template in RULER's template
#: table. With --prepare_for_ns prepare.py renders the data with the `base` template and passes
#: the chat template's token count to the generators as --model_template_token.
_PREPARE_BOOTSTRAP = (
    "import runpy, sys\n"
    "import template\n"
    "template.Templates[sys.argv[1]] = sys.argv[2]\n"
    "sys.argv = ['prepare.py'] + sys.argv[3:]\n"
    "runpy.run_path('prepare.py', run_name='__main__')\n"
)


def _prepare_task(ruler_dir: str, tokenizer: str, task: str, band: int,
                  num_samples: int, chat_template: Optional[str] = None,
                  revision: Optional[str] = None) -> List[Dict[str, Any]]:
    if chat_template is None:
        chat_template = _served_chat_template(tokenizer, enable_thinking=False)[0]
    if revision is None:
        revision = _verify_revision(ruler_dir)
    tok_slug = re.sub(r"[^A-Za-z0-9._-]", "_", tokenizer)
    # Keyed by RULER revision and chat template: either one changes the generated haystacks.
    variant = f"{revision[:12]}-{hashlib.sha256(chat_template.encode('utf-8')).hexdigest()[:12]}"
    cache = Path(gbench_cache_dir()) / "ruler" / tok_slug / variant / str(band)
    out = cache / task / "validation.jsonl"

    if out.is_file():  # reuse: RULER writes exactly num_samples lines when complete
        with open(out, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if len(rows) == num_samples:
            return rows

    cache.mkdir(parents=True, exist_ok=True)
    data_dir = os.path.join(ruler_dir, "scripts", "data")
    # RULER's prepare.py hardcodes a bare `python` for its nested generator subprocess, so put
    # THIS interpreter's dir first on PATH (it has wonderwords/nltk/etc.).
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    cmd = [sys.executable, "-c", _PREPARE_BOOTSTRAP, _TEMPLATE_NAME, chat_template,
           "--save_dir", str(cache), "--benchmark", "synthetic", "--task", task,
           "--tokenizer_path", tokenizer, "--tokenizer_type", "hf",
           "--max_seq_length", str(band), "--num_samples", str(num_samples),
           "--model_template_type", _TEMPLATE_NAME, "--prepare_for_ns"]
    logger.info("ruler: generating %s @ %s (%d samples) under %s", task, _band_label(band), num_samples, tokenizer)
    proc = subprocess.run(cmd, cwd=data_dir, env=env, capture_output=True, text=True)
    rows: List[Dict[str, Any]] = []
    if out.is_file():
        with open(out, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    # prepare.py swallows a failing generator and exits 0, so the row count is the real check.
    if proc.returncode != 0 or len(rows) != num_samples:
        raise RuntimeError(
            f"ruler: prepare.py failed for {task}@{_band_label(band)} "
            f"(rc={proc.returncode}, {len(rows)}/{num_samples} rows): "
            f"{(proc.stderr or proc.stdout)[-600:]}")
    return rows


def run_ruler(model_name: str, base_url: str, concurrency: int,
              enable_thinking: bool = False, **kwargs) -> Dict[str, Any]:
    """Run the full RULER matrix (per-tokenizer generated), canonical string-match scoring."""
    ruler_dir = _ruler_dir()
    revision = _verify_revision(ruler_dir)
    tokenizer = kwargs.get("tokenizer") or os.environ.get("GBENCH_RULER_TOKENIZER") or model_name
    lengths = _lengths()
    num_samples = int(os.environ.get("GBENCH_RULER_NUM_SAMPLES", "500"))
    _require_context(base_url, max(lengths))
    # Rendered with the run's thinking flag, as the server will render it for every request.
    chat_template, template_tokens = _served_chat_template(tokenizer, enable_thinking)

    # --eval-limit N is N samples for the suite, dealt over the task@band cells
    # (sampling.allocate_limit); a cell dealt 0 is neither generated nor run, and a cell's share
    # is a seeded random draw of its rows. No limit: every row of every cell.
    limit = kwargs.get("limit")
    cells = [f"{task}@{_band_label(band)}" for band in lengths for task in _TASKS]
    shares = allocate_limit(limit, cells, available={c: num_samples for c in cells}, seed="ruler")
    samples: List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]] = []
    for band in lengths:
        for task in _TASKS:
            share = shares[f"{task}@{_band_label(band)}"]
            if share == 0:
                continue
            rows = _prepare_task(ruler_dir, tokenizer, task, band, num_samples,
                                 chat_template=chat_template, revision=revision)
            if share is not None:
                rows = stratified_sample(rows, share, None,
                                         seed=f"ruler/{task}@{_band_label(band)}")
            cap = _GEN_CAP[_base_type(task)]
            for r in rows:
                gold = json.dumps({"outputs": r.get("outputs") or [], "task": task})
                # RULER's per-task cap is the sample's own max_tokens. base's token-limit policy
                # sends it as declared without --thinking and, with it, replaces it with the
                # run's budget (a cap written for non-reasoning generation would cut the
                # reasoning before any answer), recording the lifted caps.
                # `ruler_length` (RULER's token count of the row) only selects the pre-flight's
                # longest prompt; base sends nothing but its allowed API keys.
                meta = {"category": f"{task}@{_band_label(band)}",
                        "ruler_length": int(r.get("length") or 0), "max_tokens": cap}
                # RULER generates from input + answer_prefix (scripts/pred/call_api.py), sent as
                # one user turn by its chat-API clients.
                prompt = r["input"] + r.get("answer_prefix", "")
                samples.append(([{"role": "user", "content": prompt}], gold, meta))

    # The plan counts the rows base runs on this node (GBENCH_SHARD's shard of the generated
    # list), per cell (a cell dealt 0 stays 0), so the shards' plans add up to the rows drawn.
    on_shard = shard_select(samples, parse_shard(os.environ.get("GBENCH_SHARD")))
    plan = eval_limit_plan("ruler", limit, len(on_shard),
                           {c: sum(1 for _m, _g, meta in on_shard if meta["category"] == c)
                            for c in cells} if limit else None,
                           setting_noun="cell", attempts=base_attempts())
    # The caller's budget, as every suite passes it to base; RULER's largest cap otherwise.
    suite_budget = kwargs.get("max_output_tokens") or _DEFAULT_SUITE_BUDGET
    _require_prompts_fit(samples, enable_thinking, kwargs.get("extra_payload"), suite_budget)

    res = run_eval_suite(
        eval_name="ruler",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_ruler,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=None,  # already fully materialised (and stratified) above
        max_output_tokens=suite_budget,  # the per-task caps ride in each sample's meta
        official_sampling=OFFICIAL_SAMPLING,
    )

    res["eval_limit_plan"] = plan
    _raise_on_unsent(res.get("sample_traces") or [])

    # --- canonical aggregation (nested mean), via trace-walk ---
    # Over the scored traces, as base's accuracy: a request that got no reply from the endpoint
    # (failed or timed out) is a harness error (scoring_excluded), left out of its cell, never a
    # 0. With nothing scored the score is withheld (None).
    from .base import is_scored
    cells: Dict[Tuple[str, str], List[float]] = {}
    for t in res.get("sample_traces", []) or []:
        if not is_scored(t):
            continue
        try:
            blob = json.loads(t.get("gold_answer") or "{}")
        except Exception:
            continue
        task = blob.get("task", "")
        s = _item_score(task, t.get("response_text") or "", blob.get("outputs") or [])
        t["ruler_score"] = round(s, 4)
        band = (t.get("category") or "@").split("@")[-1]
        cells.setdefault((task, band), []).append(s)

    cell_mean = {k: sum(v) / len(v) for k, v in cells.items() if v}
    by_band: Dict[str, List[float]] = {}
    by_task: Dict[str, List[float]] = {}
    for (task, band), m in cell_mean.items():
        by_band.setdefault(band, []).append(m)
        by_task.setdefault(task, []).append(m)
    length_score = {b: round(100 * sum(v) / len(v), 2) for b, v in by_band.items()}
    task_score = {t: round(100 * sum(v) / len(v), 2) for t, v in by_task.items()}

    present = [_band_label(b) for b in lengths if _band_label(b) in length_score]
    sl = [length_score[b] for b in present]
    headline = round(sum(sl) / len(sl), 2) if sl else None

    eff = None
    for b in lengths:
        if length_score.get(_band_label(b), -1) >= _EFF_THRESHOLD:
            eff = b
    n = len(sl)
    wavg_inc = round(sum(s * (i + 1) for i, s in enumerate(sl)) / (n * (n + 1) / 2), 2) if n else 0.0
    wavg_dec = round(sum(s * (n - i) for i, s in enumerate(sl)) / (n * (n + 1) / 2), 2) if n else 0.0

    res["accuracy"] = headline
    if headline is None:
        res["accuracy_withheld_reason"] = res.get("accuracy_withheld_reason") or (
            "no RULER cell has a scored sample: " + str(res.get("status_reason")
                                                        or "every request left out of the score"))
    res["metric"] = ("RULER Avg = mean over length bands of (mean of the 13 task scores); "
                     "string_match_all (niah/vt/cwe/fwe) + string_match_part (qa)")
    res["ruler_length_scores"] = length_score
    res["ruler_task_scores"] = task_score
    res["ruler_cell_scores"] = {f"{t}@{b}": round(100 * m, 2) for (t, b), m in cell_mean.items()}
    res["length_bands"] = present
    res["effective_length"] = eff
    res["wavg_inc"] = wavg_inc
    res["wavg_dec"] = wavg_dec
    res["ruler_revision"] = revision
    res["ruler_template_tokens"] = template_tokens
    # Fully canonical only for the complete greedy matrix: all 13 tasks x the EXACT 6 canonical
    # bands (4k..128k) at 500 samples each, temperature 0.0, no --thinking/--eval-limit. (Every
    # prompt was sent: _raise_on_unsent refused the run otherwise.) Anything else is a valid run
    # but not comparable to the published RULER table, and the reason says which part differs.
    _canonical_bands = {_band_label(b) for b in _DEFAULT_LENGTHS}
    temperature = res.get("temperature")
    noncanon: List[str] = []
    if kwargs.get("limit"):
        noncanon.append(f"--eval-limit subset ({len(samples)} samples over the task@band "
                        "cells), not the full matrix")
    if temperature is None or float(temperature) != 0.0:
        noncanon.append(f"temperature {temperature} (RULER decodes greedily at 0.0)")
    if set(present) != _canonical_bands:
        noncanon.append(f"bands {present} are not the canonical 4k-128k set (GBENCH_RULER_LENGTHS)")
    if len(by_task) != len(_TASKS):
        noncanon.append(f"{len(by_task)} of {len(_TASKS)} tasks scored")
    if num_samples != 500:
        noncanon.append(f"GBENCH_RULER_NUM_SAMPLES={num_samples}, not 500")
    from .vqa_common import lost_requests_reason
    lost = lost_requests_reason(res.get("sample_traces") or [])
    if lost:
        noncanon.append(lost)
    cut = record_budget_cutoffs(res)
    if cut:
        noncanon.append(cut)
    res["leaderboard_comparable"] = not noncanon
    if noncanon:
        res["leaderboard_comparable_reason"] = "; ".join(noncanon)
    return res
