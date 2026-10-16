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

# Evaluation Suite: loft_scifact
# Description: LOFT long-context retrieval (SciFact scientific-paper passages)

"""gbench native built-in runner for loft_scifact (Long Context & Retrieval).

Renamed from `loft_x_arxiv`: LOFT has no arXiv task, and the suite runs LOFT's SciFact retrieval
task; the old name still runs this suite, with a deprecation warning.

Canonical LOFT (Long Context Frontiers, arXiv:2406.13121) in-context retrieval, task
`retrieval_scifact_128k_test:few_shot_with_cot`: LOFT's SciFact task (scientific-paper
title+abstract passages, claim -> verifying passage). The whole 128k corpus is placed in the
prompt, built exactly as LOFT's prompt registry builds it (google-deepmind/loft @ 219f68e:
prompts/prompts_retrieval.py, prompts/constants/{common,retrieval}.py, prompts/utils.py):
corpus instruction, formatting instruction, every passage as
`ID: <n> | TITLE: ... | CONTENT: ... | END ID: <n>` with passages numbered by corpus
position, the few-shot examples with separators and reverse-CoT
`TITLE: ... | ID: <n>` lines, then the test claim. Predictions are extracted with LOFT's
`utils.extract_prediction`, mapped back to passage ids, and scored with LOFT's primary metric,
recall@1. The data is LOFT's public GCS bundle, pinned by sha256.

Sampling: LOFT's inference settings (inference/models.py @ 219f68e: TEMPERATURE = 0.0, top_p
1.0) are recorded (base.OfficialSampling); top_p 1.0 is sent in both modes, and the temperature
follows gbench's rule (base.resolve_temperature): 1.0 with `--thinking`, 0.0 without. Override
the temperature for a whole run with `--temperature`, or for this suite alone with
`GBENCH_LOFT_SCIFACT_TEMPERATURE` (the pre-rename `GBENCH_LOFT_X_ARXIV_TEMPERATURE` is still read
as a deprecated alias), which takes precedence over both; the result records
`official_sampling`. Only a 0.0 run is leaderboard-comparable. LOFT's GenerationConfig sets no
max_output_tokens, so no request carries a max_tokens (the server's default applies: the rest of
the context window) and `--max-output-tokens` is not applied in either mode
(`max_output_tokens_note`).
"""

import ast
import hashlib
import json
import logging
import os
import re
import tempfile
import urllib.request
import zipfile
from typing import Any, Dict, List, Optional, Tuple
from .limit_plan import base_attempts, eval_limit_plan
from .base import (OfficialSampling, apply_shard, finalize_shard_fields, gbench_cache_dir,
                   run_eval_suite)
from .metrics import record_budget_cutoffs
from .sampling import stratified_sample, presharded_for_base
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

PILLAR = "Long Context & Retrieval"
#: The registry name (renamed from `loft_x_arxiv`; the stratified-sample seed keeps the old
#: string).
EVAL_NAME = "loft_scifact"
DOCS_URL = f"docs/evals/{EVAL_NAME}.md"

_SCIFACT_URL = "https://storage.googleapis.com/loft-bench/retrieval/scifact.zip"
#: sha256 of the bundle LOFT's download.sh fetches (GCS object last modified 2024-08-28).
_SCIFACT_SHA256 = "f7ef0719c1edb7adc343f3b5411c17868602421db1683fddb5e85aa4dda84a91"
#: inference/models.py @ 219f68e: GenerationConfig(temperature=TEMPERATURE (0.0), top_p=1.0),
#: with no max_output_tokens.
OFFICIAL_SAMPLING = OfficialSampling(
    reference=("google-deepmind/loft inference/models.py @219f68e (temperature 0.0, top_p 1.0; "
               "no max_output_tokens)"),
    temperature=0.0, params={"top_p": 1.0}, send_max_tokens=False)
#: Cache directory override; None = $GBENCH_CACHE_DIR/loft (default ~/.cache/gbench/loft).
_CACHE_DIR: Optional[str] = None


def _cache_dir() -> str:
    return _CACHE_DIR or os.path.join(gbench_cache_dir(), "loft")
_CONTEXT_LENGTH = "128k"
_SPLIT = "test"

# --- LOFT prompt constants, verbatim (google-deepmind/loft @ 219f68e) ----------------------- #
# prompts/constants/retrieval.py CORPUS_INSTRUCTION["scifact"] (.strip()ed there).
_CORPUS_INSTRUCTION = """
You will be given a list of passages. You need to read carefully and understand all of them. Then you will be given a claim, and your goal is to find all passages from the list that can help verify the claim as true of false. Print out the ID and TITLE of each passage.
""".strip()
# prompts/constants/retrieval.py FORMATTING_INSTRUCTION (not stripped upstream).
_FORMATTING_INSTRUCTION = """
Your final answer should be a list of IDs, in the following format:
Final Answer: [id1, id2, ...]
If there is only one ID, it should be in the format:
Final Answer: [id1]

If there is no perfect answer output the closest one. Do not give an empty final answer.
"""
# prompts/constants/common.py CORPUS_FORMAT_ECHO (CORPUS_FORMAT_ECHO["scifact"]).
_CORPUS_FORMAT = "ID: {pid} | TITLE: {title} | CONTENT: {passage} | END ID: {pid}"
# prompts/constants/retrieval.py QUERY_FORMAT_SIMPLE["scifact"]; few_shot_with_cot uses the
# QUERY_FORMAT_SIMPLE_REVERSE variant ("ID and TITLE" -> "TITLE and ID").
_QUERY_FORMAT_SIMPLE = """
Which passage is most relevant to the claim, and can help verify the claim as true or false? Print out the ID and TITLE of the document. Then format the IDs into a list.
If there is no perfect answer output the closest one. Do not give an empty final answer.
claim: {query}
The following passages can help verify this sentence:
""".strip()
_QUERY_FORMAT = _QUERY_FORMAT_SIMPLE.replace("ID and TITLE", "TITLE and ID").replace(
    "ID and CONTENT", "CONTENT and ID")
# prompts/constants/common.py FEW_SHOT_EXAMPLE_ANSWER_FORMAT_SIMPLE_REVERSE, FINAL_ANSWER_FORMAT,
# FEW_SHOT_SEPARATOR, TEST_QUERY_SEPARATOR.
_FEW_SHOT_COT_FORMAT = "TITLE: {title} | ID: {pid}"
_FINAL_ANSWER_FORMAT = "Final Answer: {final_answer}"
_FEW_SHOT_SEPARATOR = "====== Example {example_id} ======\n"
_TEST_QUERY_SEPARATOR = "====== Now let's start! ======\n"
# prompts/utils.py DEFAULT_TITLE and utils.load_data_from_file's blacklist_words default.
_DEFAULT_TITLE = "None"
_BLACKLIST_WORDS = ("childhood abuse",)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _ensure_scifact(context_length: str) -> str:
    """Download (once), verify and unzip the pinned LOFT SciFact bundle; return scifact/<len>/."""
    cache_dir = _cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    zip_path = os.path.join(cache_dir, "scifact.zip")
    target_dir = os.path.join(cache_dir, "scifact", context_length)
    if not os.path.exists(zip_path) or _sha256(zip_path) != _SCIFACT_SHA256:
        logger.info(f"Downloading LOFT SciFact from {_SCIFACT_URL}")
        fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".zip.part")
        os.close(fd)
        try:
            urllib.request.urlretrieve(_SCIFACT_URL, tmp)
        except Exception as e:
            raise infra_required(EVAL_NAME, f"could not download {_SCIFACT_URL} ({e}).",
                                 DOCS_URL) from e
        got = _sha256(tmp)
        if got != _SCIFACT_SHA256:
            raise infra_required(
                EVAL_NAME, f"{_SCIFACT_URL} has sha256 {got}, not the pinned "
                f"{_SCIFACT_SHA256}: the upstream bundle changed and the pin must be re-reconciled.",
                DOCS_URL)
        os.replace(tmp, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(cache_dir)
    elif not os.path.isdir(target_dir):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(cache_dir)
    if not os.path.isdir(target_dir):
        raise RuntimeError(f"{EVAL_NAME}: expected LOFT dir missing: {target_dir}")
    return target_dir


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _clean(text: Any) -> Any:
    """utils.load_data_from_file: strip, then drop blacklisted words (titles and passages)."""
    if isinstance(text, str):
        text = text.strip()
        for word in _BLACKLIST_WORDS:
            if word in text:
                text = text.replace(word, "")
    return text


def _gold_pids(row: Dict[str, Any]) -> List[str]:
    """Gold passage ids: metadata.qrels when present, else answers (prompt_registry.get_examples)."""
    pairs = (row.get("metadata") or {}).get("qrels") or row.get("answers") or []
    return [str(pair[0]) for pair in pairs]


def _chunk(text: str) -> str:
    """prompts/utils.get_text_chunk: every chunk is its text plus a newline."""
    return f"{text}\n"


def build_loft_prompts(corpus: List[Dict[str, Any]], few_shot: List[Dict[str, Any]],
                       test_queries: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, str]]:
    """One prompt per test query, as LOFT's registry renders `retrieval_scifact_*:few_shot_with_cot`.

    Returns the prompts and the pid -> ID map. The prompt is the concatenation of LOFT's content
    chunks, stripped (dump_prompts.py's text rendering via utils.concatenate_chunks).
    """
    passages: Dict[str, Dict[str, str]] = {}
    for doc in corpus:
        pid, title, passage = doc.get("pid"), doc.get("title_text"), doc.get("passage_text")
        if pid is None or title is None or passage is None:
            raise RuntimeError(f"{EVAL_NAME}: unexpected corpus schema (pid/title_text/passage_text)")
        passages[str(pid)] = {"title": _clean(title), "passage": _clean(passage)}
    # prompt_registry.py: passages are numbered by their position in the corpus.
    pid_mapper = {pid: str(idx) for idx, pid in enumerate(passages)}

    context = [_chunk(_CORPUS_INSTRUCTION), _chunk(_FORMATTING_INSTRUCTION)]
    for pid, p in passages.items():
        context.append(_chunk(_CORPUS_FORMAT.format(
            pid=pid_mapper[pid], title=p["title"] or _DEFAULT_TITLE, passage=p["passage"])))
    for n, fs in enumerate(few_shot, start=1):
        gold = _gold_pids(fs)
        context.append(_chunk((_FEW_SHOT_SEPARATOR + _QUERY_FORMAT).format(
            query=(fs.get("query_text") or "").strip(), example_id=str(n))))
        for pid in gold:
            context.append(_chunk(_FEW_SHOT_COT_FORMAT.format(
                pid=pid_mapper[pid], title=passages[pid]["title"] or _DEFAULT_TITLE)))
        answers = [pid_mapper[str(pair[0])] for pair in fs.get("answers") or []]
        context.append(_chunk(_FINAL_ANSWER_FORMAT.format(final_answer=answers)))
    shared = "".join(context)

    prompts = []
    for q in test_queries:
        query = (q.get("query_text") or "").strip()     # queries are stripped, not cleaned
        turn = _chunk((_TEST_QUERY_SEPARATOR + _QUERY_FORMAT).format(query=query, example_id=None))
        prompts.append((shared + turn).strip())
    return prompts, pid_mapper


def _load_loft_scifact_samples(
    limit: Optional[int] = None,
    context_length: str = _CONTEXT_LENGTH,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load LOFT SciFact (corpus-in-prompt) samples; raises on load/schema failure."""
    base = _ensure_scifact(context_length)
    try:
        corpus = _read_jsonl(os.path.join(base, "corpus.jsonl"))
        test_queries = _read_jsonl(os.path.join(base, f"{_SPLIT}_queries.jsonl"))
        few_shot = _read_jsonl(os.path.join(base, "few_shot_queries.jsonl"))
    except Exception as e:
        raise RuntimeError(f"Could not load dataset for {EVAL_NAME}: {e}") from e
    if not corpus or not test_queries or not few_shot:
        raise RuntimeError(f"{EVAL_NAME} returned empty corpus/queries/few-shots")

    prompts, pid_mapper = build_loft_prompts(corpus, few_shot, test_queries)
    samples = []
    for q, prompt in zip(test_queries, prompts):
        answers = q.get("answers")
        if not q.get("query_text") or not answers:
            raise RuntimeError(f"{EVAL_NAME}: unexpected query schema (query_text/answers)")
        gold_pids = [str(pair[0]) for pair in answers]
        # recall@1 = |{first prediction} & gold| / |gold|; with one gold passage per query (true of
        # every pinned SciFact test query) that is exactly the pass flag the harness counts.
        if len(gold_pids) != 1:
            raise RuntimeError(f"{EVAL_NAME}: query {q.get('qid')} has {len(gold_pids)} gold "
                               "passages; the pinned bundle has exactly one per query")
        # IDs are corpus positions, so the valid IDs are exactly "0".."num_docs-1".
        gold = {"ids": [pid_mapper[p] for p in gold_pids], "pids": gold_pids,
                "num_docs": len(pid_mapper)}
        samples.append((
            [{"role": "user", "content": prompt}], gold,
            {"category": "loft_retrieval_scifact", "qid": q.get("qid"),
             "context_length": context_length},
        ))

    # --shard first, over the full ordered set, then the --eval-limit draw within the shard.
    samples, _ = apply_shard(samples, EVAL_NAME)
    # Stratified, not a contiguous head.
    samples = stratified_sample(samples, limit, None, seed="loft_x_arxiv")
    logger.info(f"Loaded {len(samples)} {EVAL_NAME} (SciFact {context_length}) samples.")
    return samples


def _escape_single_quotes(s: str) -> str:
    """utils.extract_prediction helper: "['the devil's sleep']" -> escaped inner quote."""
    return re.sub(r"([a-zA-Z0-9])'([a-zA-Z0-9])", r"\1\'\2", s)


def _extract_prediction(text: str) -> List[Any]:
    """LOFT utils.extract_prediction: the FIRST line holding [...] is literal_eval'd; nothing else.

    A line that does not parse yields no prediction (upstream breaks out of the loop either way).
    """
    if not text:
        return []
    lines = text.replace("*", "").replace("`", "").strip().split("\n")
    preds: Any = []
    for line in lines:
        if "[" in line and "]" in line:
            frag = line[line.find("["): line.rfind("]") + 1].strip()
            try:
                preds = ast.literal_eval(_escape_single_quotes(frag))
            except Exception:
                pass
            break
    return preds if isinstance(preds, list) else []


def _eval_loft_scifact(response_text: str, gold: Any) -> bool:
    """LOFT recall@1 for a single-gold query: the first predicted ID is the gold passage's ID.

    `gold` is {"ids": [...], "pids": [...], "num_docs": N}. LOFT maps every predicted ID back to
    a passage (inference/models.py); an ID outside the corpus raises there, the query is left
    unanswered and run_evaluation.py scores it 0, so one unknown ID anywhere in the list zeroes
    the query. A bare list of gold IDs is also accepted and compared without that check.
    """
    preds = [str(x) for x in _extract_prediction(response_text)]
    if isinstance(gold, dict):
        gold_ids = {str(g) for g in gold.get("ids") or []}
        valid = {str(i) for i in range(int(gold.get("num_docs") or 0))}
        if any(p not in valid for p in preds):
            return False
    else:
        gold_ids = {str(g).strip() for g in gold or []}
        preds = [p.strip() for p in preds]
    return bool(preds) and preds[0] in gold_ids


def run_loft_scifact(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run LOFT retrieval (SciFact, 128k, test split) with LOFT's own prompt and recall@1."""
    samples = _load_loft_scifact_samples(limit=kwargs.get("limit"))
    eval_plan = eval_limit_plan(
        EVAL_NAME, kwargs.get("limit"), len(samples), attempts=base_attempts())
    result = run_eval_suite(
        eval_name=EVAL_NAME,
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=presharded_for_base(samples),  # this shard's queries, kept as is
        eval_fn=_eval_loft_scifact,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
        official_sampling=OFFICIAL_SAMPLING,   # no max_tokens, as LOFT sends none
    )
    result["eval_limit_plan"] = eval_plan
    result["benchmark"] = f"LOFT retrieval_scifact_{_CONTEXT_LENGTH}_{_SPLIT}:few_shot_with_cot"
    result["metric"] = "LOFT recall@1 (SciFact 128k, test split)"
    temperature = result.get("temperature")
    noncanon = []
    if kwargs.get("limit"):
        noncanon.append(f"--eval-limit subset ({len(samples)} of 100 test queries)")
    if enable_thinking:
        noncanon.append("--thinking run; LOFT decodes single-pass without a reasoning channel")
    if temperature is None or float(temperature) != 0.0:
        noncanon.append(f"temperature {temperature} (LOFT decodes at 0.0)")
    from .vqa_common import lost_requests_reason
    lost = lost_requests_reason(result.get("sample_traces") or [])
    if lost:
        noncanon.append(lost)
    cut = record_budget_cutoffs(result)
    if cut:
        noncanon.append(cut)
    result["leaderboard_comparable"] = not noncanon
    if noncanon:
        result["leaderboard_comparable_reason"] = "; ".join(noncanon)
    # One shard of a --shard I/N run is a partial set: the shard reason (finalize_shard_fields).
    return finalize_shard_fields(result)
