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

# Evaluation Suite: ojbench
# Description: OJBench (NOI/ICPC) - online-judge Pass@1 via the official ojbench + DMOJ sandbox

"""gbench native built-in runner for ojbench (Code & Competitive Programming).

Canonical OJBench (He-Ren/OJBench_testdata): 232 NOI/ICPC problems, each posed once in Python and
once in C++. The model emits a full stdin/stdout solution; a solution is Accepted iff ALL
testcases pass within the per-problem time/memory limits, judged by the official `ojbench` library
over the DMOJ sandbox. SANDBOX_EVAL. Hard-errors (infra_required) unless Docker, the local
gbench-ojbench:5e94480b judge image and the testdata (GBENCH_OJBENCH_TESTDATA; the bare OJBENCH_TESTDATA
still works as a deprecated alias) are all present.

Reporting follows the paper (arXiv 2506.16395, Table 2) and the official leaderboard: eight
candidate solutions per problem, Pass@1 (mean over the candidates) and Pass@8 reported SEPARATELY
for Python and C++, plus per-difficulty pass rates. The leaderboard ranks by C++ Pass@1, so that
is the headline `accuracy`; `per_language` carries every column. A generation whose request got
no reply from the served endpoint (base's harness error) or that the judge returned no verdict for
is left out of every figure and named apart (`request_harness_errors`, `judge_no_verdict`); when
C++ generations were run and none was scored, the headline is withheld (accuracy None with
`accuracy_withheld_reason`), never replaced by the Python or pooled figure.

Sampling: OJBench samples its eight candidates with "the hyper parameters officially recommended
by each model, including temperature, top_k, and top_p" (paper Section 3.1). The suite reads them
from the served model's own generation_config.json (the run's --tokenizer, else the served model
name: a local directory or a Hugging Face model id, read through the HF cache) and sends them with
and without --thinking; for Gemma 4 they are temperature 1.0, top_p 0.95 and top_k 64
(GEMMA4_RECOMMENDED). A field the file does not set is not sent, except the temperature, which is
gbench's rule for every suite whatever the file says (base.resolve_temperature): 1.0 with
--thinking (for Gemma 4 the recommended value), 0.0 without. The suite hard-errors
(infra_required) when the file cannot be read, rather than send another model's settings.
`--temperature`, or `GBENCH_OJBENCH_TEMPERATURE` for this suite alone (it takes precedence),
overrides the temperature in either mode; the result records the recommended value under
`official_sampling`, and any other temperature, or any other sampling field (decoding penalties
included), is not leaderboard-comparable.

max_tokens is the paper's, per mode (Section 3.1), through base's token-limit policy: "For
reasoning-oriented models ... we set the maximum number of tokens (max_tokens) to 64k" is the
explicit reasoning-model setting, so a --thinking request carries 65536 (or the run's
--max-output-tokens when that is larger; a thinking run is never capped below it); "For
non-reasoning-oriented models, we set max_tokens to the default context length of the model", so a
no-think request carries no max_tokens and the server grants the rest of the context window
(--max-output-tokens is not applied). `max_output_tokens_note` states the mode's choice. A reply
that stopped at another limit (without --thinking, the end of the served context window; with it,
a --max-output-tokens above 64k or a max_tokens the window lowered) is counted in
`replies_cut_off_by_budget` (those with no answer text also in `replies_cut_off_in_reasoning`) and
makes the run not leaderboard-comparable.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from math import comb
from typing import Any, Dict, List, Optional, Tuple
from .base import (OFFICIAL_SAMPLING_PARAM_KEYS, OfficialSampling, apply_shard,
                   finalize_shard_fields, get_run_knob, resolve_token_budget, run_eval_suite,
                   strip_thinking_tags, suite_env)
from .metrics import budget_cutoffs
from .limit_plan import eval_limit_plan
from .sampling import stratified_sample, presharded_for_base
from .swebench_common import infra_required, prereqs_path

logger = logging.getLogger(__name__)

PILLAR = "Coding & Software Engineering"
DOCS_URL = "docs/evals/ojbench.md"
_IMAGE_DEFAULT = "gbench-ojbench:5e94480b"
#: HF commit of He-Ren/OJBench_testdata the prompts are read from (the testdata snapshot the judge
#: mounts should be the same revision; see the docs).
_DATASET_REVISION = "61cf9986f22c25d08e1657b03742124099c74353"
#: Candidates per problem in the paper and on the leaderboard (Pass@1 and Pass@8 columns).
_OFFICIAL_SAMPLES = 8
_LANGUAGES = ("cpp", "python")

#: The paper's sampling (arXiv 2506.16395 Section 3.1): each model's officially recommended
#: temperature, top_k and top_p, read from the served model's generation_config.json.
RECOMMENDED_FIELDS = ("temperature", "top_p", "top_k")
#: What Gemma 4's generation_config.json recommends (every Gemma 4 size ships the same values).
GEMMA4_RECOMMENDED = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
_PAPER_REFERENCE = ("OJBench paper (arXiv 2506.16395) Section 3.1: the hyper parameters officially "
                    "recommended by each model (temperature, top_k, top_p)")
#: Paper Section 3.1: "For reasoning-oriented models ... we set the maximum number of tokens
#: (max_tokens) to 64k" (taken as 65536).
REASONING_MAX_TOKENS = 65536


def served_model_recommended_sampling(model_name: Optional[str] = None
                                      ) -> Tuple[Dict[str, Any], str]:
    """(the RECOMMENDED_FIELDS the served model's generation_config.json sets, where it was read).

    The model is the run's tokenizer (--tokenizer, else the served model name) or `model_name`:
    a local directory holding generation_config.json, or a Hugging Face model id read through the
    HF cache (downloaded when not cached; gated models need the usual HF token). Raises
    RuntimeError naming what failed."""
    ident = str(get_run_knob("tokenizer") or model_name or "").strip()
    if not ident:
        raise RuntimeError("no model id to read generation_config.json from (pass --tokenizer)")
    try:
        if os.path.isdir(ident):
            path = os.path.join(ident, "generation_config.json")
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=ident, filename="generation_config.json")
        with open(path, encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:                                              # noqa: BLE001
        raise RuntimeError(f"could not read generation_config.json of {ident!r} "
                           f"({type(e).__name__}: {e}); pass --tokenizer with the served model's "
                           "Hugging Face id or a local directory that holds the file") from e
    if not isinstance(config, dict):
        raise RuntimeError(f"generation_config.json of {ident!r} is not a JSON object")
    return ({k: config[k] for k in RECOMMENDED_FIELDS if config.get(k) is not None},
            f"{ident}/generation_config.json")


def recommended_official_sampling(recommended: Dict[str, Any], reference: str) -> OfficialSampling:
    """The OfficialSampling that sends a model's recommended fields (the RECOMMENDED_FIELDS its
    generation_config.json sets, served_model_recommended_sampling): the temperature when set
    (none sent otherwise), top_p and top_k when set. Shared with the MathArena suites, whose rule
    is the same (each model at its provider's recommended settings)."""
    return OfficialSampling(
        reference=reference, temperature=recommended.get("temperature"),
        send_temperature="temperature" in recommended,
        params={k: recommended[k] for k in ("top_p", "top_k") if k in recommended})


def official_sampling(recommended: Dict[str, Any], source: str) -> OfficialSampling:
    """The OfficialSampling for a model's recommended fields (served_model_recommended_sampling),
    with the paper's max_tokens per mode: none sent without --thinking (the model's default
    context length), 64k with it (the reasoning-model setting, in `reasoning`)."""
    shown = ", ".join(f"{k} {recommended[k]}" for k in RECOMMENDED_FIELDS if k in recommended)
    sampling = dict(temperature=recommended.get("temperature"),
                    send_temperature="temperature" in recommended,
                    params={k: recommended[k] for k in ("top_p", "top_k") if k in recommended})
    ref = f"{_PAPER_REFERENCE}; read from {source}: {shown or 'none set'}"
    return OfficialSampling(
        reference=(f"{ref}; non-reasoning-oriented models: max_tokens = the model's default "
                   "context length (none sent)"),
        send_max_tokens=False, **sampling,
        reasoning=OfficialSampling(
            reference=f"{ref}; reasoning-oriented models: max_tokens 64k (65536)",
            max_tokens=REASONING_MAX_TOKENS, **sampling))


def _image() -> str:
    return os.environ.get("GBENCH_OJBENCH_IMAGE", _IMAGE_DEFAULT)


def _testdata_dir() -> Optional[str]:
    return prereqs_path("OJBench_testdata", suite_env("GBENCH_OJBENCH_TESTDATA", "OJBENCH_TESTDATA"))


def _samples_per_problem(attempt_count: Optional[int] = None) -> int:
    """GBENCH_OJBENCH_SAMPLES if set; else the official 8 (or more, if --attempt-count asks)."""
    raw = (suite_env("GBENCH_OJBENCH_SAMPLES", default="") or "").strip()
    if not raw:
        return max(_OFFICIAL_SAMPLES, int(attempt_count or 1))
    try:
        n = int(raw)
    except ValueError:
        n = 0
    if n < 1:
        raise ValueError(f"GBENCH_OJBENCH_SAMPLES must be a positive integer, got {raw!r}")
    return n


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021) for n samples with c correct."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def check_ojbench_prerequisites() -> Tuple[bool, str]:
    """Docker + the locally-built gbench-ojbench:5e94480b image (which carries ojbench + DMOJ judge-server +
    PyPy3 + g++ on Python 3.11) + the NOI/ICPC testdata (bind-mounted read-only at run time).

    The judge is containerized because DMOJ's cptbox does not build on the 3.12 serving env
    (PyLongObject.ob_digit was removed in 3.12); on the image's Python 3.11 it compiles cleanly.
    """
    build = (f"Build it LOCALLY (gbench never pulls):\n"
             f"  docker build -t {_image()} -f docker/ojbench.Dockerfile docker\nSee {DOCS_URL}.")
    if not shutil.which("docker"):
        return False, "docker CLI not found. " + build
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        return False, "docker daemon not reachable. " + build
    if subprocess.run(["docker", "image", "inspect", _image()], capture_output=True).returncode != 0:
        return False, f"image {_image()!r} not found. " + build
    td = _testdata_dir()
    if not td or not os.path.isdir(os.path.join(td, "NOI")) or not os.path.isdir(os.path.join(td, "ICPC")):
        return False, ("OJBench testdata not found: set GBENCH_OJBENCH_TESTDATA (legacy alias "
                       "OJBENCH_TESTDATA still works) to a snapshot of "
                       "He-Ren/OJBench_testdata containing NOI/ and ICPC/ (7.85 GB, "
                       "`hf download He-Ren/OJBench_testdata --repo-type dataset`).")
    return True, ""


def _load_ojbench_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load OJBench prompts (full.jsonl); prompt sent verbatim; raises on load/schema failure."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id="He-Ren/OJBench_testdata",
                               filename="prompts/full.jsonl", repo_type="dataset",
                               revision=_DATASET_REVISION)
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    except Exception as e:
        logger.error(f"Failed to load dataset for ojbench: {e}")
        raise RuntimeError(f"Could not load dataset for ojbench: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for ojbench returned empty rows")
    # --shard first, over the full ordered set, then the --eval-limit draw within the shard.
    rows, _ = apply_shard(rows, "ojbench")
    # Stratified by language, not a contiguous head: both columns stay measured.
    rows = stratified_sample(rows, limit, lambda r: r.get("language"), seed="ojbench")

    samples = []
    for item in rows:
        pid = item.get("id")
        prompt = item.get("prompt")
        lang = item.get("language")
        dataset = item.get("dataset")
        difficulty = item.get("difficulty")
        if pid is None or not prompt or not lang:
            raise RuntimeError(
                "ojbench: unexpected schema (id/prompt/language); refusing to fabricate sample data")
        # Canonical: the prompt already embeds the response-format constraint; send verbatim.
        messages = [{"role": "user", "content": str(prompt)}]
        samples.append((messages, pid, {
            "category": lang,
            "row": {"id": pid, "dataset": dataset, "language": lang, "difficulty": difficulty},
        }))

    logger.info(f"Loaded {len(samples)} ojbench samples.")
    return samples


def _get_timeout_env() -> List[str]:
    """`-e OJBENCH_GET_TIMEOUT_S=<s>` for the judge container when the host sets it (the poll
    interval of the patched judger's result queue, docker/ojbench_patch_judger.py; the image
    defaults to 120 s), else nothing. A value that is not a positive integer is an error here,
    before any container runs (the patched judger would crash on it)."""
    raw = (os.environ.get("OJBENCH_GET_TIMEOUT_S") or "").strip()
    if not raw:
        return []
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError(f"OJBENCH_GET_TIMEOUT_S must be a positive integer (seconds), got {raw!r}")
    return ["-e", f"OJBENCH_GET_TIMEOUT_S={value}"]


def _judge_in_container(records: List[Dict[str, Any]], num_workers: int) -> List[Dict[str, Any]]:
    """Judge the generated submissions inside gbench-ojbench:5e94480b. The DMOJ sandbox (cptbox) uses
    ptrace + seccomp, so the run needs SYS_PTRACE and an unconfined seccomp profile (Docker's
    default profile blocks installing the inner seccomp filter). Records in / results out via a
    bind-mounted work dir; the 7.85 GB testdata is bind-mounted read-only. The judge needs
    nothing from the network (the testdata is mounted, the judge and compilers are in the image),
    so the container that runs the model's programs has none (`--network none`)."""
    if not records:
        return []
    td = os.path.abspath(_testdata_dir())
    workdir = tempfile.mkdtemp(prefix="gbench_ojbench_")
    os.chmod(workdir, 0o777)                        # the container writes results as its own uid
    name = "gbench_ojbench_" + os.path.basename(workdir)

    def _reap():
        # docker --rm only fires on a clean EXIT; reap the named container on timeout too.
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)

    with open(os.path.join(workdir, "records.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    cmd = ["docker", "run", "--rm", "--name", name, "--network", "none",
           "--security-opt", "seccomp=unconfined", "--cap-add", "SYS_PTRACE",
           "-v", f"{workdir}:/work",
           "-v", f"{td}:/testdata:ro",
           "-e", "OJBENCH_TESTDATA=/testdata",
           "-e", f"OJBENCH_WORKERS={max(1, num_workers)}",
           *_get_timeout_env(),
           _image()]
    # Bounded, configurable timeout so a stuck submission can NEVER hang the whole run (a compiled
    # solution's DMOJ sandbox execution can deadlock under Docker). Scales with the batch size;
    # GBENCH_OJBENCH_JUDGE_TIMEOUT_S overrides for large runs.
    _env_to = suite_env("GBENCH_OJBENCH_JUDGE_TIMEOUT_S", default="")
    timeout_s = int(_env_to) if _env_to else max(1200, 180 * len(records))
    logger.info("ojbench: judging %d submissions in %s (timeout %ds)", len(records), _image(), timeout_s)
    proc = None
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"ojbench: judge container exceeded {timeout_s}s and was killed - a compiled submission's "
            f"DMOJ cptbox sandbox can deadlock under Docker. Raise GBENCH_OJBENCH_JUDGE_TIMEOUT_S if "
            f"this was a large but healthy run; see {DOCS_URL}.") from e
    finally:
        _reap()
    results_path = os.path.join(workdir, "results.jsonl")
    if not os.path.isfile(results_path):
        raise RuntimeError(
            "ojbench: the judge container produced no results.jsonl (judging failed) - this is a "
            "harness/infra failure, not a 0%. Last output:\n"
            + ((proc.stderr or proc.stdout or "") if proc else "")[-1200:])
    with open(results_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _make_scorer(num_workers: int):
    async def _score(sample_traces: List[Dict[str, Any]]) -> None:
        import asyncio

        records = []
        for i, tr in enumerate(sample_traces):
            resp = tr.get("response_text")
            row = (tr.get("extra_payload") or {}).get("row") or {}
            if resp and row.get("id") is not None:
                # One record per GENERATION: with eight candidates per problem, (id, language) is
                # no longer unique. The judge echoes every input field back, so the uid maps each
                # verdict to its own trace.
                records.append({**row, "gbench_uid": i, "content": strip_thinking_tags(resp)})

        results = await asyncio.to_thread(_judge_in_container, records, num_workers)

        # A result counts ONLY if the judge actually returned a verdict (`is_passed`). A submitted
        # program that comes back without one was NOT graded - e.g. the DMOJ worker crashed on that
        # problem (a wrong judge-server build with no Problem.cases() returns records unchanged) -
        # which is a harness failure, NOT a model 0. Scoring it False silently turns an ungraded
        # run into a fake 0% (measured 2026-09-11).
        graded = [r for r in (results or []) if isinstance(r, dict) and "is_passed" in r]
        by_uid = {r["gbench_uid"]: bool(r["is_passed"]) for r in graded if "gbench_uid" in r}
        # the whole graded record per generation, for the trace's execution detail
        rec_by_uid = {r["gbench_uid"]: r for r in graded if "gbench_uid" in r}
        # A verdict without the uid still identifies its trace when (id, language) was submitted
        # exactly once; an ambiguous one is left ungraded rather than copied onto k attempts.
        pair_count: Dict[Tuple[Any, Any], int] = {}
        for rec in records:
            pair = (rec.get("id"), rec.get("language"))
            pair_count[pair] = pair_count.get(pair, 0) + 1
        by_pair = {(r.get("id"), r.get("language")): bool(r["is_passed"]) for r in graded
                   if "gbench_uid" not in r and pair_count.get((r.get("id"), r.get("language"))) == 1}
        rec_by_pair = {(r.get("id"), r.get("language")): r for r in graded
                       if "gbench_uid" not in r
                       and pair_count.get((r.get("id"), r.get("language"))) == 1}
        if records and not graded:
            # Nothing was graded at all: never report this as a clean 0%.
            raise infra_required(
                "ojbench",
                "the judge container returned NO graded verdicts (`is_passed` missing on every "
                "submission) - the DMOJ judge could not score any program (commonly a wrong "
                "judge-server build: OJBench needs the commit its README pins, whose Problem.cases() "
                "the PyPI dmoj lacks). This is a judge/infra failure, not a 0%% resolved rate.",
                DOCS_URL)
        for i, tr in enumerate(sample_traces):
            row = (tr.get("extra_payload") or {}).get("row") or {}
            key = (row.get("id"), row.get("language"))
            verdict = by_uid.get(i, by_pair.get(key))
            if verdict is not None:
                tr["is_correct"] = verdict               # genuinely judged
                tr["status"] = "OK"
                graded_rec = rec_by_uid.get(i, rec_by_pair.get(key))
                if graded_rec is not None:
                    tr["execution"] = judge_execution(graded_rec)
            elif tr.get("response_text") and row.get("id") is not None:
                # submitted but the judge returned no verdict for it -> harness failure: exclude
                # from the numerator/denominator (like a judge outage) rather than score it wrong.
                tr["is_correct"] = False
                tr["status"] = "OK"
                tr["scoring_excluded"] = True
                tr["scoring_note"] = "judge returned no verdict for this submission"
            else:
                tr["is_correct"] = False              # empty/no model answer: a genuine non-answer
                tr["status"] = "OK"
    return _score


#: The most per-test entries of the judge's detailed_results a trace keeps.
_DETAIL_ENTRIES = 20


def judge_execution(record: Dict[str, Any]) -> Dict[str, Any]:
    """The judge's verdict detail of one graded generation, for its trace: the overall `verdict`
    (AC, WA, TLE, RTE, ...), the per-scale verdicts (the record's `<scale>verdict` fields) and a
    capped summary of `detailed_results`."""
    scales = {k: v for k, v in record.items()
              if isinstance(k, str) and k.endswith("verdict") and k != "verdict"}
    out: Dict[str, Any] = {"verdict": record.get("verdict"),
                           "scale_verdicts": dict(sorted(scales.items()))}
    detail = record.get("detailed_results")
    if isinstance(detail, list):
        out["detailed_results"] = [
            {k: (v if not isinstance(v, str) else v[:200]) for k, v in d.items()}
            if isinstance(d, dict) else str(d)[:200] for d in detail[:_DETAIL_ENTRIES]]
        if len(detail) > _DETAIL_ENTRIES:
            out["detailed_results_left_out"] = len(detail) - _DETAIL_ENTRIES
    elif detail is not None:
        out["detailed_results"] = str(detail)[:2000]
    return out


def cpp_attempts(traces: List[Dict[str, Any]], k: int) -> Optional[Dict[str, Any]]:
    """The @k block over the C++ generations only (the headline's language), as base computes
    it over every generation: avg@k over the scored generations, pass@k over the problems with a
    scored generation, pass^k over the problems with k scored generations. None without one."""
    by_problem: Dict[Any, List[bool]] = {}
    for t in traces:
        row = (t.get("extra_payload") or {}).get("row") or {}
        if row.get("language") != "cpp" or t.get("scoring_excluded"):
            continue
        by_problem.setdefault(row.get("id"), []).append(bool(t.get("is_correct")))
    gens = [ok for v in by_problem.values() for ok in v]
    if not gens:
        return None
    complete = [v for v in by_problem.values() if len(v) >= k]
    return {"attempts_per_sample": k, "samples": len(by_problem), "generations": len(gens),
            "avg_at_k": round(100.0 * sum(gens) / len(gens), 2),
            "pass_at_k": round(100.0 * sum(1 for v in by_problem.values() if any(v))
                               / len(by_problem), 2),
            "pass_hat_k": (round(100.0 * sum(1 for v in complete if all(v)) / len(complete), 2)
                           if complete else None),
            "pass_hat_k_samples": len(complete),
            "unstable_samples": sum(1 for v in by_problem.values() if any(v) and not all(v))}


def _per_language_report(traces: List[Dict[str, Any]], n: int) -> Dict[str, Any]:
    """Paper/leaderboard columns per language: Pass@1 (mean over candidates), Pass@k with the
    unbiased estimator (k = min(8, n)), and Pass@1 by difficulty. Judge-excluded generations
    are left out, as in the headline."""
    k = min(_OFFICIAL_SAMPLES, n)
    report: Dict[str, Any] = {}
    for lang in _LANGUAGES:
        per_problem: Dict[Any, List[bool]] = {}
        by_diff: Dict[str, List[int]] = {}
        for t in traces:
            row = (t.get("extra_payload") or {}).get("row") or {}
            if row.get("language") != lang or t.get("scoring_excluded"):
                continue
            ok = bool(t.get("is_correct"))
            per_problem.setdefault(row.get("id"), []).append(ok)
            d = by_diff.setdefault(str(row.get("difficulty")), [0, 0])
            d[0] += int(ok)
            d[1] += 1
        gens = sum(len(v) for v in per_problem.values())
        if not gens:
            continue
        correct = sum(sum(v) for v in per_problem.values())
        entry = {
            "problems": len(per_problem),
            "generations": gens,
            "correct": correct,
            "pass_at_1": round(100.0 * correct / gens, 2),
            "pass_rate_by_difficulty": {
                diff: round(100.0 * c / tot, 2) for diff, (c, tot) in sorted(by_diff.items())},
        }
        if k > 1 and all(len(v) >= k for v in per_problem.values()):
            entry[f"pass_at_{k}"] = round(100.0 * sum(
                _pass_at_k(len(v), sum(v), k) for v in per_problem.values()) / len(per_problem), 2)
        report[lang] = entry
    return report


def run_ojbench(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run OJBench online-judge Pass@1/Pass@8 per language (hard-errors if the judge/testdata are
    unavailable)."""
    ok, reason = check_ojbench_prerequisites()
    if not ok:
        raise infra_required("ojbench", reason, DOCS_URL)
    _get_timeout_env()   # a malformed OJBENCH_GET_TIMEOUT_S fails here, before any generation
    try:
        recommended, source = served_model_recommended_sampling(model_name)
    except RuntimeError as e:
        raise infra_required(
            "ojbench", f"OJBench samples with the model's own recommended temperature, top_p and "
            f"top_k, read from its generation_config.json, and {e}", DOCS_URL) from e
    sampling = official_sampling(recommended, source)
    limit = kwargs.get("limit")
    n = _samples_per_problem(kwargs.get("attempt_count"))
    samples = _load_ojbench_samples(limit=limit)
    # --eval-limit N is N problem rows for the whole suite, spread over the two languages (the
    # loader's stratified draw), never N per language.
    by_language: Dict[str, int] = {}
    for _m, _g, meta in samples:
        by_language[str(meta.get("category"))] = by_language.get(str(meta.get("category")), 0) + 1
    limit_plan = eval_limit_plan("ojbench", limit, len(samples), by_language,
                                 setting_noun="language", attempts=n)
    result = run_eval_suite(
        eval_name="ojbench",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=presharded_for_base(samples),  # this shard's problems, kept as is
        async_eval_fn=_make_scorer(concurrency),
        declared_scoring_mode="execution",  # sandboxed judge (test execution), not an LLM judge
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=limit,
        max_output_tokens=kwargs.get("max_output_tokens", 8192),
        attempt_count=n,          # candidates per problem; each is judged on its own
        supports_attempts=True,   # the scorer keys verdicts per generation (gbench_uid)
        official_sampling=sampling,
    )
    result["recommended_sampling"] = {"source": source, **recommended}
    result["eval_limit_plan"] = limit_plan
    per_lang = _per_language_report(result.get("sample_traces") or [], n)
    result["per_language"] = per_lang
    result["samples_per_problem"] = n
    result["dataset_revision"] = f"He-Ren/OJBench_testdata@{_DATASET_REVISION}"
    result["pooled_pass_at_1"] = result.get("accuracy")
    noncanon: List[str] = []
    traces = result.get("sample_traces") or []
    cpp = per_lang.get("cpp")
    cpp_run = any(((t.get("extra_payload") or {}).get("row") or {}).get("language") == "cpp"
                  for t in traces)
    if cpp:
        # Headline = the leaderboard's ranking column; counts follow it so accuracy == correct/total.
        result["accuracy"] = cpp["pass_at_1"]
        result["correct_answers"] = cpp["correct"]
        result["total_questions"] = cpp["generations"]
        # base's @k block pools both languages; the headline is C++'s, so the block follows it
        # and the pooled one is kept apart
        if isinstance(result.get("attempts"), dict):
            pooled = dict(result["attempts"])
            result["attempts_pooled_both_languages"] = pooled
            cpp_block = cpp_attempts(traces, int(pooled.get("attempts_per_sample") or n))
            if cpp_block:
                result["attempts"] = {**{k: pooled[k] for k in ("attempts_requested",)
                                         if k in pooled},
                                      **cpp_block, "language": "cpp"}
        result["metric"] = (f"C++ Pass@1 (headline, mean over {n} candidate(s)/problem; the "
                            "leaderboard's ranking column); Python Pass@1, Pass@8 and difficulty "
                            "pass rates in per_language")
    elif cpp_run:
        # C++ generations were run and every one was left out of the score (no reply from the
        # served endpoint, or no judge verdict): the headline is withheld, never replaced by the
        # other language's figure.
        result["accuracy"] = None
        result["correct_answers"] = 0
        result["total_questions"] = 0
        why = ("no C++ generation was scored: every one got no reply from the served endpoint or "
               "no judge verdict (harness errors), so the C++ Pass@1 headline is withheld")
        result["accuracy_withheld_reason"] = (
            f"{result['accuracy_withheld_reason']}; {why}"
            if result.get("accuracy_withheld_reason") else why)
        result["metric"] = ("C++ Pass@1 (headline, withheld: no C++ generation was scored); "
                            "Python figures in per_language")
        noncanon.append(why)
    else:
        result["metric"] = ("pooled Pass@1 (no C++ problem was scored in this subset, so the C++ "
                            "headline could not be computed); see per_language")
        noncanon.append("no C++ problems scored (headline is the pooled Pass@1)")
    if limit:
        noncanon.append(f"subset run (--eval-limit {limit})")
    if n < _OFFICIAL_SAMPLES:
        noncanon.append(f"{n} candidate(s)/problem (OJBench uses {_OFFICIAL_SAMPLES})")
    temp = result.get("temperature")
    want = recommended.get("temperature")
    if (temp is None) != (want is None) or (
            temp is not None and abs(float(temp) - float(want)) > 1e-9):
        shown = want if want is not None else "not set"
        noncanon.append(f"temperature={temp} ({result.get('temperature_source')}; OJBench samples "
                        "its candidates with the model's recommended parameters: temperature "
                        f"{shown} in {source})")
    # The request's sampling fields (decoding penalties are merged in last, as base sends them)
    # against the recommended ones: those, and nothing else.
    official = dict(sampling.params or {})
    sent_params = {**official, **(kwargs.get("extra_payload") or {}),
                   **(result.get("decoding_penalties") or {})}
    changed = sorted(k for k in set(sent_params) | set(official)
                     if k in OFFICIAL_SAMPLING_PARAM_KEYS and sent_params.get(k) != official.get(k))
    if changed:
        noncanon.append("sampling " + ", ".join(f"{k}={sent_params.get(k, 'not sent')}"
                                                for k in changed)
                        + " (the model's recommended "
                        + ", ".join(f"{k}={official.get(k, 'not set')}" for k in changed) + ")")
    sent = result.get("max_tokens_sent") or []
    plan = resolve_token_budget("ojbench", sampling, thinking=enable_thinking)
    if plan.policy == "official_reasoning":
        # 64k, or a larger --max-output-tokens; a prompt too long for that in the served window
        # was sent less (base clamps each request to what fits).
        if not sent or min(sent) < REASONING_MAX_TOKENS:
            noncanon.append(f"max_tokens={min(sent) if sent else 'not sent'} (OJBench gives "
                            f"reasoning models 64k, {REASONING_MAX_TOKENS}; a request whose "
                            "prompt left less room in the served context window was sent less)")
    elif sent:
        # The paper gives non-reasoning models their default context length: no max_tokens.
        noncanon.append(f"max_tokens={sent} (OJBench gives non-reasoning models their default "
                        "context length; none is sent)")
    # A reply stopped at a limit other than the paper's (the 64k with --thinking; without it the
    # model's context length, which is only the served window's end) measures that limit, not
    # the model.
    cut, in_reasoning = budget_cutoffs(result)
    result["replies_cut_off_by_budget"] = cut
    result["replies_cut_off_in_reasoning"] = in_reasoning
    if cut:
        window = get_run_knob("max_model_len")
        where = (f"the end of the served {window or 'unknown'}-token context window (no "
                 "max_tokens is sent)" if plan.policy == "not_sent"
                 else f"this run's max_tokens {plan.max_tokens}, or a max_tokens the context "
                      "window lowered, instead of the paper's 65536")
        noncanon.append(f"{cut} repl(ies) stopped at {where} before finishing ({in_reasoning} of "
                        "them inside the reasoning channel with no answer)")
    # The result's scoring_excluded counts every generation left out of the score: those the
    # judge returned no verdict for (the scorer's marker) and those whose request got no reply
    # from the served endpoint (base's harness errors, named in status_reason and counted in
    # request_harness_errors). Each is named apart.
    no_verdict = sum(1 for t in traces if t.get("scoring_excluded") and not t.get("harness_error"))
    no_reply = sum(1 for t in traces if t.get("harness_error"))
    result["judge_no_verdict"] = no_verdict
    if no_verdict:
        noncanon.append(f"{no_verdict} generation(s) got no judge verdict and were excluded")
    if no_reply:
        noncanon.append(f"{no_reply} generation request(s) got no reply from the served endpoint "
                        "(harness errors, left out of the score; request_harness_errors)")
    result["leaderboard_comparable"] = not noncanon
    if noncanon:
        result["leaderboard_comparable_reason"] = "; ".join(noncanon)
    # One shard of a --shard I/N run is a partial set: the shard reason (finalize_shard_fields).
    return finalize_shard_fields(result)
