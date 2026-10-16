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

"""Multilingual translation on the WMT19 news translation test sets (newstest2019).

Data: the official WMT19 news test sets, all 19 directions (cs-de, de-cs, de-en, de-fr, en-cs,
en-de, en-fi, en-gu, en-kk, en-lt, en-ru, en-zh, fi-en, fr-de, gu-en, kk-en, lt-en, ru-en, zh-en;
31,387 segments, one reference each), as sacrebleu distributes them (test set "wmt19":
data.statmt.org/wmt19/translation-task/test.tgz). The suite downloads that tarball itself (with a
timeout, into a `.part` file that is renamed only after its sha256 matches; a cached tarball
with another sha256 is fetched again), extracts it where sacrebleu reads it and lets
sacrebleu turn the SGML into text. A download that fails is `infra_required` for this suite only
and a checksum mismatch raises; neither ends the gbench run (sacrebleu's own downloader calls
sys.exit on a TLS error or an md5 mismatch, so it is never asked to download). The download, the
extraction and the text preparation run under a file lock in the cache directory, so processes
that share the cache (GBENCH_SHARD shards on a first run) never race on the same files. An
extraction is marked complete only after it finishes, so a tarball that was never extracted, or
whose extraction was cut short, is extracted again; a text file cut short is rebuilt from the
SGML, and when that does not restore the expected segment count the tarball is extracted again
and the text rebuilt once more.

Scoring: sacreBLEU, the scorer WMT reports automatic scores with: per direction, corpus BLEU
(sacrebleu 2.6.0 BLEU with the target language's default tokenizer: `zh` for Chinese, `13a`
otherwise; mixed case, exponential smoothing) and corpus chrF (sacrebleu CHRF defaults, chrF2). The
headline is the mean of the 19 directions' BLEU; the mean chrF and every direction's BLEU and chrF,
with their sacrebleu signatures, are reported too. An empty generation stays in its direction's
corpus as an empty hypothesis; a request that got no reply from the endpoint is a harness error,
left out of the corpus. The headline and the mean chrF average the exact corpus scores and are rounded once.

The prompt is gbench's (WMT defines none), and WMT19 reports each direction on its own (the
headline average is gbench's), so the result is `leaderboard_comparable: false`; each direction's
BLEU is the number comparable with WMT19 systems.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_I18N_TRANSLATE_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import contextlib
import copy
import hashlib
import logging
import os
import tarfile
import urllib.request
from collections import Counter
from typing import Any, Dict, Iterator, List, Optional, Tuple
from .base import apply_shard, gbench_cache_dir, is_scored, run_eval_suite
from .metrics import record_budget_cutoffs
from .sampling import stratified_sample, presharded_for_base
from .fc_common import base_suite_plan, run_attempts
from .swebench_common import infra_required

_DOCS_URL = "docs/evals/i18n_translate.md"
EVAL_NAME = "i18n_translate"

logger = logging.getLogger(__name__)

#: sacrebleu's WMT19 test set and the release that scores it.
TEST_SET = "wmt19"
SACREBLEU_RELEASE = "2.6.0"
#: sha256 of the test.tgz sacrebleu lists for "wmt19" (sacrebleu's own record is its md5,
#: 84de7162d158e28403103b01aeefc39a); the suite downloads and checks it itself.
TEST_SET_SHA256 = "5ef66d76e183ed40450086ad6fa7beccda1fbb290c5556b93273d515fd6683fb"
#: Direction -> segments in newstest2019.
DIRECTIONS = {
    "cs-de": 1997, "de-cs": 1997, "de-en": 2000, "de-fr": 1701, "en-cs": 1997, "en-de": 1997,
    "en-fi": 1997, "en-gu": 998, "en-kk": 998, "en-lt": 998, "en-ru": 1997, "en-zh": 1997,
    "fi-en": 1996, "fr-de": 1701, "gu-en": 1016, "kk-en": 1000, "lt-en": 1000, "ru-en": 2000,
    "zh-en": 2000,
}
N_SEGMENTS = 31387
LANGUAGE_NAMES = {"cs": "Czech", "de": "German", "en": "English", "fi": "Finnish",
                  "fr": "French", "gu": "Gujarati", "kk": "Kazakh", "lt": "Lithuanian",
                  "ru": "Russian", "zh": "Chinese"}
PROMPT = "Translate the following text from {src} into {tgt}. Output only the translation:\n\n{text}"


def _require_sacrebleu() -> Any:
    try:
        import sacrebleu
    except ImportError:
        raise infra_required(EVAL_NAME, "the WMT19 test sets and BLEU / chrF come from sacrebleu; "
                             "install it (pip install -e '.[evals]')", _DOCS_URL)
    return sacrebleu


#: Seconds without data before the test-set download gives up.
DOWNLOAD_TIMEOUT_S = 120


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_test_tarball(url: str, tarball: str) -> None:
    """Fetch `url` to `tarball`: into `<tarball>.part`, sha256-checked, then renamed. The caller
    holds _cache_lock, so no other process writes the same `.part` file.

    A network, TLS or HTTP failure is infra_required (this suite only); a file with another
    sha256 raises RuntimeError and is left as the `.part` file, so the cache never holds it."""
    part = tarball + ".part"
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as resp, \
                open(part, "wb") as out:
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                out.write(chunk)
    except Exception as e:                                              # noqa: BLE001
        raise infra_required(EVAL_NAME, f"the WMT19 test sets ({url}) could not be downloaded "
                             f"({type(e).__name__}: {e})", _DOCS_URL)
    digest = _sha256(part)
    if digest != TEST_SET_SHA256:
        raise RuntimeError(f"{EVAL_NAME}: {url} downloaded with sha256 {digest}, expected "
                           f"{TEST_SET_SHA256} (kept as {part})")
    os.replace(part, tarball)


def _extracted_marker(rawdir: str) -> str:
    """The file written after a complete extraction of the release into `rawdir`."""
    return os.path.join(rawdir, f".extracted-{TEST_SET_SHA256[:16]}")


def _extract(tarball: str, rawdir: str) -> None:
    """Unpack the test tarball into sacrebleu's raw directory (sacrebleu.utils.extract_tarball,
    with tarfile's "data" filter so no member can land outside it), then mark it complete: a
    run stopped mid-extraction leaves no marker, so the next run extracts again."""
    marker = _extracted_marker(rawdir)
    if os.path.exists(marker):
        os.remove(marker)
    with tarfile.open(tarball) as tar:
        tar.extractall(path=rawdir, filter="data")
    with open(marker, "w", encoding="utf-8") as fh:
        fh.write(TEST_SET_SHA256 + "\n")


#: Seconds a process waits for another that holds the cache lock (downloading, extracting or
#: preparing the text) before it gives up.
LOCK_TIMEOUT_S = 1800


@contextlib.contextmanager
def _cache_lock(outdir: str) -> Iterator[None]:
    """An inter-process lock on `<outdir>/.gbench.lock` (portalocker, a sacrebleu dependency,
    the library sacrebleu's own downloader locks with). A different file from the
    `<tarball>.lock` sacrebleu takes inside process_to_text, so holding it never blocks that."""
    import portalocker
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, ".gbench.lock")
    try:
        lock = portalocker.Lock(path, mode="a", timeout=LOCK_TIMEOUT_S)
        lock.acquire()
    except portalocker.exceptions.LockException as e:
        raise RuntimeError(f"{EVAL_NAME}: another process has held {path} for more than "
                           f"{LOCK_TIMEOUT_S} s ({type(e).__name__})") from None
    try:
        yield
    finally:
        lock.release()


def _raw_files(dataset: Any) -> List[str]:
    """The raw SGML files sacrebleu reads for the 19 directions."""
    return sorted({os.path.join(dataset._rawdir, path)
                   for direction in DIRECTIONS for path in dataset.langpairs[direction]})


def _wmt19_dataset() -> Any:
    """sacrebleu's "wmt19" test set, stored under $GBENCH_CACHE_DIR/sacrebleu/wmt19.

    The tarball is fetched and checked here (_download_test_tarball) and extracted unless a
    complete extraction of it is marked (or a raw file is missing), so sacrebleu only converts
    files that are already on disk. The caller holds _cache_lock."""
    sacrebleu = _require_sacrebleu()
    from sacrebleu.dataset import DATASETS
    dataset = copy.copy(DATASETS[TEST_SET])
    dataset._outdir = os.path.join(gbench_cache_dir(), "sacrebleu", TEST_SET)
    dataset._rawdir = os.path.join(dataset._outdir, "raw")
    if sorted(dataset.langpairs) != sorted(DIRECTIONS):
        raise RuntimeError(f"{EVAL_NAME}: sacrebleu {sacrebleu.__version__} lists WMT19 directions "
                           f"{sorted(dataset.langpairs)}, expected {sorted(DIRECTIONS)}")
    url = dataset.data[0]
    os.makedirs(dataset._rawdir, exist_ok=True)
    tarball = os.path.join(dataset._rawdir, dataset._get_tarball_filename(url))
    fresh = False
    if not os.path.isfile(tarball) or _sha256(tarball) != TEST_SET_SHA256:
        # Absent, or not the release (e.g. cut short by an interrupted download): fetch it again.
        if os.path.isfile(tarball):
            logger.warning("%s: %s is not the WMT19 release (sha256 mismatch); downloading it "
                           "again.", EVAL_NAME, tarball)
        _download_test_tarball(url, tarball)
        fresh = True
    if (fresh or not os.path.exists(_extracted_marker(dataset._rawdir))
            or any(not os.path.isfile(f) for f in _raw_files(dataset))):
        _extract(tarball, dataset._rawdir)
    missing = [f for f in _raw_files(dataset) if not os.path.isfile(f)]
    if missing:
        raise RuntimeError(f"{EVAL_NAME}: {tarball} does not contain {missing[:3]} "
                           f"({len(missing)} missing)")
    return dataset


def _direction_text(dataset: Any, direction: str, expected: int) -> Tuple[List[str], List[str]]:
    """(sources, first references) of one direction, from sacrebleu's text files.

    A text file cut short (a run stopped while sacrebleu wrote it) is rebuilt from the raw SGML;
    when the SGML itself is short (an extraction cut short before the completion marker existed,
    or a file changed since), the checked tarball is extracted again and the text rebuilt once
    more. sacrebleu exits the process on some errors; that is turned into an exception here."""
    try:
        for attempt in (0, 1, 2):
            if attempt == 2:
                tarball = os.path.join(dataset._rawdir,
                                       dataset._get_tarball_filename(dataset.data[0]))
                _extract(tarball, dataset._rawdir)
            if attempt:
                dataset.process_to_text(direction)
            sources = list(dataset.source(direction))
            references = [refs[0] for refs in dataset.references(direction)]
            if len(sources) == expected and len(references) == expected:
                return sources, references
    except SystemExit as e:
        raise RuntimeError(f"{EVAL_NAME}: sacrebleu stopped while preparing WMT19 {direction} "
                           f"(exit code {e.code})") from None
    raise RuntimeError(f"{EVAL_NAME}: WMT19 {direction} has {len(sources)} sources / "
                       f"{len(references)} references after extracting the release again, "
                       f"expected {expected}")


def _load_i18n_translate_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Every newstest2019 segment of the 19 WMT19 directions (limit: seeded, spread evenly)."""
    _require_sacrebleu()
    outdir = os.path.join(gbench_cache_dir(), "sacrebleu", TEST_SET)
    with _cache_lock(outdir):
        dataset = _wmt19_dataset()
        texts = {direction: _direction_text(dataset, direction, expected)
                 for direction, expected in DIRECTIONS.items()}
    samples = []
    for direction in DIRECTIONS:
        src_code, tgt_code = direction.split("-")
        sources, references = texts[direction]
        for i, (src_text, ref) in enumerate(zip(sources, references)):
            prompt = PROMPT.format(src=LANGUAGE_NAMES[src_code], tgt=LANGUAGE_NAMES[tgt_code],
                                   text=src_text)
            samples.append(([{"role": "user", "content": prompt}], ref,
                            {"category": direction, "segment": i}))
    # --shard first, over the full ordered set, then the --eval-limit draw within the shard.
    samples, _ = apply_shard(samples, "i18n_translate")
    # Stratified across the directions, not a contiguous head.
    samples = stratified_sample(samples, limit, key_fn=lambda s: s[2]["category"],
                                seed="i18n_translate")
    logger.info(f"Loaded {len(samples)} WMT19 translation segments.")
    return samples


def _eval_i18n_translate(response_text: str, gold_text: str) -> bool:
    """Secondary pass flag: sentence chrF >= 40 against the gold reference.

    chrF is case-sensitive by construction (it credits partial word/morphology matches), so
    neither side is lowercased - lowercasing would inflate the score and diverge from the
    canonical metric. The reply is scored as returned (surrounding whitespace removed).
    """
    if not response_text or not gold_text:
        return False

    pred = response_text.strip()
    gold = gold_text.strip()

    return chrf_score(pred, gold) >= 40.0


# --- chrF, ported from sacrebleu 2.6.0 sacrebleu/metrics/chrf.py (CHRF defaults) ------------- #
# char_order=6, word_order=0, beta=2, lowercase=False, whitespace=False, eps_smoothing=False.
# Ported rather than imported so the metric is one code path with no optional dependency.
_CHRF_CHAR_ORDER = 6
_CHRF_BETA = 2


def _char_ngrams(line: str) -> List[Counter]:
    """helpers.extract_all_char_ngrams with include_whitespace=False."""
    line = "".join(line.split())
    return [Counter([line[i:i + n] for i in range(len(line) - n + 1)])
            for n in range(1, _CHRF_CHAR_ORDER + 1)]


def _chrf_segment_stats(hyp: str, ref: str) -> List[int]:
    """[hyp, ref, match] counts per order (CHRF._get_match_statistics, single reference)."""
    stats: List[int] = []
    for hyp_ngrams, ref_ngrams in zip(_char_ngrams(hyp), _char_ngrams(ref)):
        match_count, hyp_count = 0, 0
        for ng, count in hyp_ngrams.items():
            hyp_count += count
            if ng in ref_ngrams:
                match_count += min(count, ref_ngrams[ng])
        # sacrebleu does not count hypothesis n-grams of an order the reference lacks
        stats.extend([hyp_count if ref_ngrams else 0, sum(ref_ngrams.values()), match_count])
    return stats


def _chrf_from_stats(stats: List[int]) -> float:
    """CHRF._compute_f_score without eps smoothing (effective-order averaging)."""
    factor = _CHRF_BETA ** 2
    avg_prec, avg_rec, effective_order = 0.0, 0.0, 0
    for i in range(_CHRF_CHAR_ORDER):
        n_hyp, n_ref, n_match = stats[3 * i: 3 * i + 3]
        if n_hyp > 0 and n_ref > 0:
            avg_prec += n_match / n_hyp
            avg_rec += n_match / n_ref
            effective_order += 1
    if effective_order == 0:
        return 0.0
    avg_prec /= effective_order
    avg_rec /= effective_order
    if avg_prec + avg_rec:
        return 100 * (1 + factor) * avg_prec * avg_rec / ((factor * avg_prec) + avg_rec)
    return 0.0


def chrf_score(pred: str, gold: str) -> float:
    """Sentence-level chrF (0-100), as sacrebleu.sentence_chrf(pred, [gold]).

    Character-level, so it credits partial word matches and morphology; order-aware through the
    n-grams (a bag of the right words in any order does not score like a translation).
    """
    return _chrf_from_stats(_chrf_segment_stats(str(pred or ""), str(gold or "")))


def chrf_corpus(hyps: List[str], refs: List[str]) -> float:
    """CORPUS-level chrF (0-100) - the metric WMT actually reports; sacrebleu.corpus_chrf.

    chrF is a corpus statistic: character n-gram match/total counts are POOLED across every
    sentence before precision/recall/F-beta is taken - not averaged per sentence. A macro
    (mean-of-sentence) chrF over-weights short sentences and is not comparable to the WMT
    number. An empty hypothesis is kept in the pool (it contributes ref n-grams with no
    matches, penalising recall) rather than dropped, so a model that fails on hard sentences
    cannot inflate the score. (A request that got no reply from the endpoint never reaches the
    pool: run_i18n_translate leaves it out as a harness error.)
    """
    total = [0] * (3 * _CHRF_CHAR_ORDER)
    pairs = 0
    for hyp, ref in zip(hyps, refs):
        for i, v in enumerate(_chrf_segment_stats(str(hyp or ""), str(ref or ""))):
            total[i] += v
        pairs += 1
    return _chrf_from_stats(total) if pairs else 0.0


def _hypothesis(trace: Dict[str, Any]) -> str:
    """The reply as scored: the system output as returned, surrounding whitespace removed (quotes
    are kept: many references are quoted sentences, so removing a pair would change a correct
    translation)."""
    return (trace.get("response_text") or "").strip()


def wmt_scores(traces: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per direction: sacreBLEU corpus BLEU and chrF with their signatures, in segment order."""
    _require_sacrebleu()
    from sacrebleu.metrics import BLEU, CHRF
    by_dir: Dict[str, List[Tuple[int, str, str]]] = {}
    for t in traces:
        payload = t.get("extra_payload") or {}
        direction = t.get("category") or payload.get("category")
        by_dir.setdefault(direction, []).append(
            (int(payload.get("segment", len(by_dir.get(direction, [])))), _hypothesis(t),
             str(t.get("gold_answer") or "")))
    out: Dict[str, Dict[str, Any]] = {}
    for direction in sorted(by_dir):
        rows = sorted(by_dir[direction], key=lambda r: r[0])
        hyps, refs = [r[1] for r in rows], [r[2] for r in rows]
        bleu = BLEU(trg_lang=direction.split("-")[1])
        chrf = CHRF()
        bleu_score = bleu.corpus_score(hyps, [refs])
        chrf_score_ = chrf.corpus_score(hyps, [refs])
        out[direction] = {"bleu": round(bleu_score.score, 2), "chrf": round(chrf_score_.score, 2),
                          # the unrounded corpus scores the means are taken over
                          "bleu_exact": float(bleu_score.score),
                          "chrf_exact": float(chrf_score_.score),
                          "segments": len(rows), "bleu_signature": str(bleu.get_signature()),
                          "chrf_signature": str(chrf.get_signature())}
    return out


def run_i18n_translate(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run WMT19 newstest2019 translation (19 directions); headline = mean sacreBLEU BLEU."""
    limit = kwargs.get("limit")
    sacrebleu = _require_sacrebleu()
    samples = _load_i18n_translate_samples(limit=limit)
    limit_plan = base_suite_plan("i18n_translate", presharded_for_base(samples), limit,
                                 run_attempts(kwargs))
    result = run_eval_suite(
        eval_name=EVAL_NAME,
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=presharded_for_base(samples),  # this shard's segments, kept as is
        eval_fn=_eval_i18n_translate,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=None,        # the loader already drew the --eval-limit subset
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
    result["eval_limit_plan"] = limit_plan
    traces = result.get("sample_traces", []) or []
    # The corpus scores are over the scored segments, as base's accuracy is: a request that got
    # no reply from the endpoint is a harness error (scoring_excluded), left out of the corpus,
    # never scored as an empty translation. An empty reply is the model's and stays in.
    scored = [t for t in traces if is_scored(t)]
    for trace in scored:
        # sentence-level, for inspection only
        trace["chrf"] = round(chrf_score(_hypothesis(trace), str(trace.get("gold_answer") or "")), 2)
    per_dir = wmt_scores(scored) if scored else {}
    result["pass_rate_chrf40"] = result.get("accuracy")
    result["i18n_directions"] = per_dir
    result["i18n_bleu"] = {d: v["bleu"] for d, v in per_dir.items()}
    result["i18n_chrf"] = {d: v["chrf"] for d, v in per_dir.items()}
    result["sacrebleu_version"] = sacrebleu.__version__
    result["metric"] = ("mean over the WMT19 directions of sacreBLEU corpus BLEU (newstest2019; "
                        "per direction in i18n_bleu, chrF in i18n_chrf)")
    if per_dir:
        # means of the exact per-direction scores, rounded once
        result["accuracy"] = round(sum(v["bleu_exact"] for v in per_dir.values()) / len(per_dir),
                                   2)
        result["mean_chrf"] = round(sum(v["chrf_exact"] for v in per_dir.values()) / len(per_dir),
                                    2)
    result["headline_metrics"] = {"bleu_mean": result.get("accuracy") if per_dir else None,
                                  "chrf_mean": result.get("mean_chrf")}
    # The rows next to the headline carry the headline metric: per direction, its corpus BLEU
    # (total = its segments); base's per-segment chrF >= 40 tally moves to
    # category_pass_rate_chrf40, and correct_answers (segments at chrF >= 40) is not the BLEU's.
    result["category_pass_rate_chrf40"] = result.get("category_accuracy") or {}
    result["category_accuracy"] = {d: {"correct": None, "total": v["segments"],
                                       "accuracy": v["bleu"]} for d, v in per_dir.items()}
    result["segments_chrf40"] = result.get("correct_answers")
    result["correct_answers"] = None
    result["correct_answers_note"] = ("not applicable: the headline is the mean corpus BLEU; "
                                      "segments_chrf40 counts the segments with sentence chrF of "
                                      "at least 40 (pass_rate_chrf40)")
    result["dataset"] = (f"WMT19 news test sets (newstest2019), sacrebleu test set {TEST_SET!r}, "
                         f"{len(DIRECTIONS)} directions")
    reasons = ["the headline averages the directions' BLEU (WMT19 reports each direction on its "
               "own; i18n_bleu holds the comparable values) and the prompt is gbench's (WMT "
               "defines none)"]
    if limit:
        reasons.append(f"--eval-limit {int(limit)} subset of the {N_SEGMENTS:,} segments")
    if os.environ.get("GBENCH_SHARD"):
        reasons.append(f"GBENCH_SHARD {os.environ['GBENCH_SHARD']} shard")
    if set(per_dir) != set(DIRECTIONS):
        reasons.append(f"{len(per_dir)} of the {len(DIRECTIONS)} directions scored")
    if sacrebleu.__version__ != SACREBLEU_RELEASE:
        reasons.append(f"scored with sacrebleu {sacrebleu.__version__} (this suite is checked "
                       f"with {SACREBLEU_RELEASE})")
    from .vqa_common import lost_requests_reason
    lost = lost_requests_reason(traces)
    if lost:
        reasons.append(lost)
    cut = record_budget_cutoffs(result)
    if cut:
        reasons.append(cut)
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = "; ".join(reasons)
    return result
