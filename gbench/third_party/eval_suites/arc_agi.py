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

# Evaluation Suite: arc_agi
# Description: ARC-AGI-1 (Abstraction & Reasoning Corpus - grid-transformation puzzles)

"""gbench native built-in runner for arc_agi (Abstract Reasoning).

Canonical ARC-AGI-1 (Francois Chollet). Loads the public 400-task evaluation set from the
official repository, fchollet/ARC-AGI at a pinned commit (`data/evaluation/*.json`, the data
arc-agi-benchmarking's README clones into `data/arc-agi`), serialises each task's train
demonstrations + test input with the official ARC-Prize prompt, and scores by EXACT grid match
under the canonical **pass@2** rule (a test pair is solved if either of two attempts matches
the gold grid exactly). This is NOT allenai/ai2_arc (a different grade-school-science MCQ set).

The data comes from GitHub's archive of that commit, is checked against the commit's git tree
id for `data/evaluation` (so every file is byte-identical to upstream's) and is cached under
`$GBENCH_CACHE_DIR/arc_agi/<commit>/evaluation`. A failed download or a mismatch stops the suite
(infra_required); there is no other source.

Aggregation follows arcprize/arc-agi-benchmarking `scoring.py` (commit 28e67d54): a task scores
the fraction of its test pairs solved (`score_task`: task_score / num_pairs) and the headline is
the mean over tasks (`score_submission`: total_score / total_tasks). 19 of the 400 tasks have
two or three test pairs; averaging over the 419 pairs would weight those tasks double.

Each attempt is read the way upstream builds its `Attempt` (schemas.py): the reply goes through
upstream's own parser (`utils/parsing.py` at the same commit, ported verbatim below; the first
`\\boxed{...}` holding a JSON list of lists, else the last bracketed JSON list of lists), then
through pydantic's `Union[str, List[List[int]]]` validation, which turns integer-valued floats,
bools and digit strings into ints. When that raises (no reply, nothing parseable, cells pydantic
rejects), ARCTester tries the attempt again, up to `--retry_attempts 2` tries in all; gbench
does the same. An attempt still without an answer (nothing parseable, an empty reply, or cells
pydantic rejects) is a null attempt, scored wrong as upstream scores it; an attempt whose last
try got no reply from the served endpoint is a harness error, left out of the score (never a
null attempt). (main.py's fallback to the provider's JSON extractor only runs when the
answer is still a string, which the Attempt validator never leaves, so it is not ported.)

Sharding (`--shard I/N`) splits the set by whole tasks, before the `--eval-limit` draw, so every
test pair of a task is scored in the same shard and the per-task score is exact in each shard.

Sampling: arc-agi-benchmarking sends a model only the fields its entry in models.yml sets
(schemas.ModelConfig collects them into the API kwargs), and most entries set no temperature, so
upstream's request carries none (base.OfficialSampling with `send_temperature=False`; the result
records `official_sampling`). gbench always sends one, by its rule for every suite
(base.resolve_temperature): 1.0 with --thinking, 0.0 without; so its requests carry a field
arc-agi-benchmarking does not send and the run is not leaderboard-comparable. `--temperature`, or
for this suite alone `GBENCH_ARC_AGI_TEMPERATURE` (which takes precedence), overrides it in either
mode.
LLM-judge grading is pinned at 0.0 and is not affected by either (this suite has no judge).
"""

import hashlib
import io
import json
import logging
import os
import re
import tarfile
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import TypeAdapter, ValidationError
from .base import (OfficialSampling, answer_in_thinking_fields, apply_shard, gbench_cache_dir,
                   is_scored, run_eval_suite, strip_thinking_tags)
from .limit_plan import eval_limit_plan
from .sampling import parse_shard, stratified_sample
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

PILLAR = "Language, Knowledge & Reasoning"

DOCS_URL = "docs/evals/arc_agi.md"

#: The official ARC-AGI-1 data: github.com/fchollet/ARC-AGI at this commit (master, 2025-04-04).
ARC_REPO = "fchollet/ARC-AGI"
ARC_COMMIT = "399030444e0ab0cc8b4e199870fb20b863846f34"
#: git tree id of `data/evaluation` at ARC_COMMIT (400 task files, 419 test pairs). The loader
#: recomputes it over the files it reads, downloaded or cached.
ARC_EVALUATION_TREE = "4a19f8327d45dda5e1809ee47be08bf1484b493b"
#: The revision the loader reads (the commit).
_DATASET_REVISION = ARC_COMMIT
_ARC_TARBALL_URL = f"https://codeload.github.com/{ARC_REPO}/tar.gz/{ARC_COMMIT}"
_DOWNLOAD_TIMEOUT_S = 120
#: ARC-AGI's rule: two attempts per test pair.
_CANONICAL_ATTEMPTS = 2
#: main.py / cli/run_all.py `--retry_attempts` default: tries per attempt.
_RETRY_ATTEMPTS = 2
#: The provider adapters send only the fields a model's models.yml entry sets; without a
#: temperature there, none is sent.
OFFICIAL_SAMPLING = OfficialSampling(
    reference=("arcprize/arc-agi-benchmarking@28e67d54 adapters/openai_base.py + "
               "schemas.ModelConfig: the API kwargs are the model config's own fields, and a "
               "config without a temperature sends none"),
    send_temperature=False)

# Verbatim arcprize/arc-agi-benchmarking prompts/system_prompt.txt (commit 28e67d54), filled the
# way prompt_manager.convert_task_pairs_to_prompt fills it.
_PROMPT_TEMPLATE = (
    "You are participating in a puzzle solving competition. You are an expert at "
    "solving puzzles.\n\n"
    "Below is a list of input and output pairs with a pattern. Your goal is to "
    "identify the pattern or transformation in the training examples that maps the "
    "input to the output, then apply that pattern to the test input to give a final "
    "output.\n\n"
    "Respond in the format of the training output examples\n\n"
    "--Training Examples--\n{training_examples}\n--End of Training Examples--\n\n"
    "--Test Input--\n{test_input}\n--End of Test Input--\n\n"
    "Your response:"
)


def _grid_ok(g: Any) -> bool:
    return (
        isinstance(g, list) and len(g) > 0
        and all(isinstance(r, list) and len(r) > 0 for r in g)
        and all(isinstance(c, int) for r in g for c in r)
    )


def _git_tree_id(files: Dict[str, bytes]) -> str:
    """The git tree id of a flat directory holding `files` (name -> bytes) as regular files."""
    body = b"".join(
        b"100644 " + name.encode("utf-8") + b"\0"
        + hashlib.sha1(b"blob %d\0" % len(data) + data).digest()
        for name, data in sorted(files.items()))
    return hashlib.sha1(b"tree %d\0" % len(body) + body).hexdigest()


def _cache_dir() -> str:
    return os.path.join(gbench_cache_dir(), "arc_agi", ARC_COMMIT, "evaluation")


def _download_tarball() -> bytes:
    import urllib.request
    with urllib.request.urlopen(_ARC_TARBALL_URL, timeout=_DOWNLOAD_TIMEOUT_S) as resp:
        return resp.read()


def _evaluation_files(tarball: bytes) -> Dict[str, bytes]:
    """`data/evaluation/*.json` of the commit archive (top-level folder ARC-AGI-<commit>)."""
    files: Dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:gz") as tf:
        for member in tf.getmembers():
            parts = member.name.split("/")
            if (member.isfile() and len(parts) == 4 and parts[1:3] == ["data", "evaluation"]
                    and parts[3].endswith(".json")):
                handle = tf.extractfile(member)
                if handle is not None:
                    files[parts[3]] = handle.read()
    return files


def _read_cache(path: str) -> Dict[str, bytes]:
    files: Dict[str, bytes] = {}
    for name in os.listdir(path):
        if name.endswith(".json"):
            with open(os.path.join(path, name), "rb") as fh:
                files[name] = fh.read()
    return files


def _write_cache(path: str, files: Dict[str, bytes]) -> None:
    """Write the verified files next to `path`, then move them into place in one rename (an
    existing copy that failed verification is moved aside, not reused)."""
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=".evaluation-", dir=parent)
    for name, data in files.items():
        with open(os.path.join(staging, name), "wb") as fh:
            fh.write(data)
    if os.path.exists(path):
        os.replace(path, f"{path}.stale-{int(time.time())}")
    os.replace(staging, path)


def _official_evaluation_files() -> Dict[str, bytes]:
    """The 400 ARC-AGI-1 evaluation task files at ARC_COMMIT, verified against its git tree.

    Read from the cache when the cached copy verifies, else downloaded from GitHub's archive
    of the commit, verified and cached. Any failure is infra_required."""
    cache = _cache_dir()
    if os.path.isdir(cache):
        cached = _read_cache(cache)
        if _git_tree_id(cached) == ARC_EVALUATION_TREE:
            return cached
        logger.warning("[arc_agi] the cached data in %s does not match %s@%s; downloading it "
                       "again", cache, ARC_REPO, ARC_COMMIT[:8])
    try:
        tarball = _download_tarball()
        files = _evaluation_files(tarball)
    except Exception as e:
        raise infra_required(
            "arc_agi", f"the ARC-AGI-1 evaluation data could not be downloaded from "
            f"{_ARC_TARBALL_URL} ({type(e).__name__}: {e}). Allow network access to "
            f"codeload.github.com, or put the 400 files of data/evaluation at {ARC_REPO}@"
            f"{ARC_COMMIT[:8]} in {cache}.", DOCS_URL) from e
    got = _git_tree_id(files)
    if got != ARC_EVALUATION_TREE:
        raise infra_required(
            "arc_agi", f"the ARC-AGI-1 data downloaded from {_ARC_TARBALL_URL} does not match the "
            f"pinned commit: data/evaluation has {len(files)} file(s) with git tree {got}, "
            f"expected {ARC_EVALUATION_TREE}.", DOCS_URL)
    try:
        _write_cache(cache, files)
    except OSError as e:
        logger.warning("[arc_agi] could not cache the ARC-AGI data in %s (%s); it will be "
                       "downloaded again next run", cache, e)
    return files


def _load_arc_agi_tasks() -> List[Dict[str, Any]]:
    """The 400 evaluation tasks as {"id", "train", "test"}, in task-id order."""
    files = _official_evaluation_files()
    tasks = []
    for name in sorted(files):
        try:
            task = json.loads(files[name].decode("utf-8"))
        except ValueError as e:
            raise infra_required("arc_agi", f"{name} is not valid JSON ({e}).", DOCS_URL) from e
        tasks.append({"id": name[:-len(".json")], "train": task.get("train"),
                      "test": task.get("test")})
    return tasks


def _load_arc_agi_samples(
    enable_thinking: bool = False,
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """ARC-AGI-1 evaluation tasks as one sample per test pair; raises on load/schema failure.

    `--shard` (GBENCH_SHARD) and then `limit` draw whole TASKS (every test pair of a drawn task
    is kept), since the score is per task."""
    rows = _load_arc_agi_tasks()
    if not rows:
        raise RuntimeError("arc_agi: the evaluation data holds no tasks")
    rows, _ = apply_shard(rows, "arc_agi")

    # Stratified, not a contiguous head.
    rows = stratified_sample(rows, limit, None, seed="arc_agi")

    samples: List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]] = []
    for idx, item in enumerate(rows):
        train_pairs = item.get("train")
        test_pairs = item.get("test")
        task_id = item.get("id") or f"task_{idx}"
        if (not isinstance(train_pairs, list) or not train_pairs
                or not isinstance(test_pairs, list) or not test_pairs):
            raise RuntimeError(
                "arc_agi: unexpected dataset schema (missing 'train'/'test'); "
                "refusing to fabricate sample data"
            )

        demos = ""
        for i, pair in enumerate(train_pairs):
            inp, out = pair.get("input"), pair.get("output")
            if not _grid_ok(inp) or not _grid_ok(out):
                raise RuntimeError(f"arc_agi: bad train grid in task {task_id}")
            demos += (
                f"--Example {i}-- \n\n INPUT: \n\n{json.dumps(inp)}"
                f"\n\nOUTPUT: \n\n{json.dumps(out)}\n\n"
            )

        for k, tpair in enumerate(test_pairs):
            t_in, t_out = tpair.get("input"), tpair.get("output")
            if not _grid_ok(t_in) or not _grid_ok(t_out):
                raise RuntimeError(f"arc_agi: bad/missing test grid in task {task_id}")
            prompt = _PROMPT_TEMPLATE.format(
                training_examples=demos, test_input=json.dumps(t_in)
            )
            messages = [{"role": "user", "content": prompt}]
            samples.append((
                messages, t_out,
                {"category": "grid_transformation", "task_id": task_id, "pair_index": k},
            ))

    logger.info(f"Loaded {len(samples)} arc_agi test pairs from {len(rows)} tasks "
                f"({ARC_REPO}@{ARC_COMMIT[:8]}).")
    return samples


# --- BEGIN verbatim port: arcprize/arc-agi-benchmarking src/arc_agi_benchmarking/utils/parsing.py
# (commit 28e67d54b05df5be10281892243c509a42a874f1, Apache-2.0). Upstream's Attempt model
# (schemas.py) runs parse_and_validate_json on every raw answer: the first \boxed{...} that
# holds a JSON list of lists wins, else the last bracketed JSON list of lists (backscan).
# Unchanged except that ProviderJsonExtractor (unused here) is not ported. ---

def backscan_json_parser(log_str: str) -> Optional[List[List[int]]]:
    """
    Extract the last valid JSON substring that matches the List[List] structure
    from the given log string by scanning backwards from the end.

    Parameters:
        log_str (str): The full log output text.

    Returns:
        The parsed List[List] object if found and valid, otherwise None.
    """
    last_bracket_idx = -1
    closing_bracket = None
    for i in range(len(log_str) - 1, -1, -1):
        char = log_str[i]
        if char in (']', '}'):
            last_bracket_idx = i
            closing_bracket = char
            break

    if last_bracket_idx == -1:
        return None

    opening_bracket = '[' if closing_bracket == ']' else '{'

    bracket_counter = 1 # Start at 1 to account for the found closing bracket
    start_idx = -1

    for i in range(last_bracket_idx - 1, -1, -1):
        char = log_str[i]
        if char == closing_bracket:
            bracket_counter += 1
        elif char == opening_bracket:
            bracket_counter -= 1
            if bracket_counter == 0:
                start_idx = i
                break

    if start_idx == -1:
        return None

    json_candidate = log_str[start_idx:last_bracket_idx+1]

    try:
        parsed_json = json.loads(json_candidate)

        # Validate the structure: must be a non-empty list of lists.
        if isinstance(parsed_json, list) and parsed_json and all(isinstance(row, list) for row in parsed_json):
            return parsed_json
        else:
            return None

    except json.JSONDecodeError:
        return None

def extract_from_boxed(log_str: str) -> Optional[List[List[int]]]:
    """
    Extracts JSON from a LaTeX-style \\boxed{} command in a string.
    """
    match = re.search(r"\\boxed\{(.*?)\}", log_str, re.DOTALL)
    if match:
        content = match.group(1).strip()
        try:
            # The content inside boxed is often a list of lists
            parsed_json = json.loads(content)
            if isinstance(parsed_json, list) and all(isinstance(i, list) for i in parsed_json):
                return parsed_json
        except json.JSONDecodeError:
            # If json.loads fails, it's not the JSON we're looking for.
            pass
    return None


# --- Main Parsing Orchestrator ---

def parse_and_validate_json(response: str) -> Optional[List[List[int]]]:
    """
    Orchestrates parsing by trying the provider extractor.
    Returns the parsed List[List[int]] or raises ValueError if validation fails.
    """
    parsing_attempts = [
        extract_from_boxed,
        backscan_json_parser
    ]

    for parser in parsing_attempts:
        result = parser(response)
        if result is not None:
            # Validate the structure: must be list of lists
            if isinstance(result, list) and all(isinstance(row, list) for row in result):
                return result # Return immediately on first success and validation
            else:
                # Raise error if structure is invalid, triggering retry in main loop
                raise ValueError(f"Parser {parser.__name__} produced invalid structure: {result!r}")

    # Raise an error here if no method works, triggering retry in main loop
    raise ValueError(f"Failed to parse response after all attempts: {response!r}")
# --- END verbatim port ---


def _upstream_parse(text: Optional[str]) -> Optional[List[Any]]:
    """Upstream's answer parse, or None where upstream raises (and so retries the attempt)."""
    if not text:
        return None
    try:
        return parse_and_validate_json(text)
    except ValueError:
        return None


#: schemas.py `Attempt.answer: Union[str, List[List[int]]]`, validated by pydantic after the
#: `parse_answer` before-validator has run parse_and_validate_json on the reply text.
_ATTEMPT_ANSWER = TypeAdapter(Union[str, List[List[int]]])


def _parse_grid(text: Optional[str]) -> Optional[List[List[int]]]:
    """The answer upstream's `Attempt(answer=<reply>)` holds, or None where building it raises.

    The reply is parsed by parse_and_validate_json, then validated as List[List[int]] with
    pydantic's default (lax) mode, which turns integer-valued floats, bools and digit strings
    into ints. An unparseable reply, a request without a reply, or a grid pydantic rejects
    (`1.5`, `"a"`, `None` cells) makes upstream's attempt raise, so ARCTester retries it."""
    parsed = _upstream_parse(text)
    if parsed is None:
        return None
    try:
        answer = _ATTEMPT_ANSWER.validate_python(parsed)
    except ValidationError:
        return None
    return answer if isinstance(answer, list) else None


def _eval_arc_agi(response_text: str, gold: Any) -> bool:
    """main.py: `attempt_obj.answer == test_pairs[pair_index].output` (exact grid equality)."""
    pred = _parse_grid(response_text)
    if pred is None:
        return False
    try:
        g = [[int(c) for c in row] for row in gold]
    except Exception:
        return False
    return pred == g


def _needs_retry(trace: Dict[str, Any]) -> bool:
    """Would ARCTester's try have raised (no reply, or no valid Attempt from it)?"""
    return _parse_grid(strip_thinking_tags(trace.get("response_text"))) is None


def _score_per_task(sample_traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Upstream aggregation: pass@k per test pair, pair fraction per task, mean over tasks.

    A pair is solved when ANY of its attempts is correct (score_task's `any_attempt_correct`);
    a task scores solved_pairs / num_pairs; the headline averages task scores. An attempt left
    without a valid answer after its tries is a null attempt, scored wrong as upstream scores
    it, and counted in `unparsed_attempts`. An attempt whose last try got no reply from the
    endpoint is a harness error (base marks it scoring_excluded) and is left out: a pair is
    scored on its other attempts, and a pair (task) with none scored is not counted."""
    pairs: Dict[int, Dict[str, Any]] = {}
    unparsed = 0
    for t in sample_traces:
        if not is_scored(t):
            continue
        meta = t.get("extra_payload") or {}
        src = t.get("source_sample_idx", t.get("sample_idx"))
        entry = pairs.setdefault(src, {"task_id": meta.get("task_id"), "solved": False})
        entry["solved"] = entry["solved"] or bool(t.get("is_correct"))
        if _needs_retry(t):
            unparsed += 1
    tasks: Dict[Any, List[bool]] = {}
    for src, entry in pairs.items():
        tasks.setdefault(entry["task_id"] or f"pair_{src}", []).append(entry["solved"])
    task_scores = {tid: sum(v) / len(v) for tid, v in tasks.items()}
    total = sum(task_scores.values())
    return {
        "tasks": len(task_scores),
        "test_pairs": len(pairs),
        "pairs_solved": sum(1 for e in pairs.values() if e["solved"]),
        "task_score_sum": round(total, 4),
        "score": round(total / len(task_scores) * 100.0, 2) if task_scores else None,
        "unparsed_attempts": unparsed,
    }


#: base.classify_reply healths of a reply that stopped at the max_tokens budget, including the
#: relabel base.reclassify_non_convergent applies once the run is over.
_CUT_OFF_HEALTHS = ("truncated", "looping", "non_convergent")


def _cut_off(trace: Dict[str, Any]) -> bool:
    """Did the server stop this try at the max_tokens budget?

    Read from what the server did (finish_reason "length", or base's TRUNCATED status), not only
    from the health label: base.run_eval_suite relabels a budget hit `non_convergent` after
    scoring when it ran GBENCH_NON_CONVERGENT_RATIO (4 by default) times the run's healthy
    median length or more, and keeps its status TRUNCATED.
    A budget hit with no answer text after the reasoning is one of these too (base gives it
    health `truncated` or `looping` and status TRUNCATED)."""
    return (trace.get("status") == "TRUNCATED" or trace.get("finish_reason") == "length"
            or trace.get("health") in _CUT_OFF_HEALTHS)


def _for_base(samples: List[Any]) -> List[Any]:
    """`samples` (this shard's already), laid out so base.run_eval_suite keeps exactly them.

    base applies GBENCH_SHARD "I/N" to the list it is given (sampling.shard_select:
    items[I-1::N]). The suite draws its shard over whole tasks itself, so each sample is put at
    a position base selects, in order, and the positions in between stay empty. Without a shard
    the list is returned as is."""
    shard = parse_shard(os.environ.get("GBENCH_SHARD"))
    if not shard:
        return list(samples)
    index, count = shard
    laid_out: List[Any] = [None] * (len(samples) * count)
    laid_out[index - 1::count] = samples
    return laid_out


def _retry_failed_tries(traces: List[Dict[str, Any]], samples: List[Any],
                        run_kwargs: Dict[str, Any]) -> Dict[str, int]:
    """ARCTester's inner loop: each attempt gets up to `--retry_attempts` (2) tries; a try whose
    Attempt cannot be built (see _parse_grid) is asked again. The retried traces replace the
    failed ones in place, keeping their pair and attempt numbers."""
    pending = [i for i, t in enumerate(traces) if _needs_retry(t)]
    retried = recovered = 0
    for _ in range(_RETRY_ATTEMPTS - 1):
        if not pending:
            break
        # The retry batch is already this shard's own attempts; _for_base lays it out so that
        # base's GBENCH_SHARD selection keeps all of it, in order.
        again = run_eval_suite(
            samples=_for_base([samples[traces[i]["source_sample_idx"]] for i in pending]),
            attempt_count=1, limit=None, **run_kwargs)
        by_idx = {t.get("sample_idx"): t for t in again.get("sample_traces") or []}
        still = []
        for j, i in enumerate(pending):
            new = by_idx.get(j)
            if new is None:
                still.append(i)
                continue
            old = traces[i]
            retried += 1
            traces[i] = {**new, "sample_idx": old.get("sample_idx"),
                         "source_sample_idx": old.get("source_sample_idx"),
                         "attempt_idx": old.get("attempt_idx"),
                         "tries": int(old.get("tries") or 1) + 1,
                         "earlier_try": {"status": old.get("status"), "error": old.get("error"),
                                         "response_text": old.get("response_text")}}
            if _needs_retry(traces[i]):
                still.append(i)
            else:
                recovered += 1
        pending = still
    return {"retried_attempts": retried, "recovered_by_retry": recovered}


def run_arc_agi(
    model_name: str,
    base_url: str,
    concurrency: int = 1,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native ARC-AGI-1 grid-transformation evaluation suite.

    Canonical metric is **pass@2**, averaged per task: a test pair is solved if EITHER of two
    attempts reproduces the gold grid exactly, a task scores the fraction of its test pairs
    solved, and the headline is the mean over tasks (arc-agi-benchmarking score_task /
    score_submission). Each attempt gets up to two tries, as ARCTester's --retry_attempts 2.
    base.py samples the k attempts (accuracy = avg@k over pairs, attempts = pass@k over pairs);
    the per-task pass@2 replaces the headline here."""
    limit = kwargs.get("limit")
    samples = _load_arc_agi_samples(enable_thinking=enable_thinking, limit=limit)
    # ARC-AGI canonical rule = 2 attempts. Force a MINIMUM of 2: the CLI passes attempt_count=1 by
    # default, and `1 or 2` == 1 would silently run pass@1 (understating vs the canonical pass@2 and
    # contradicting the canonical_sync claim). Allow a higher explicit override.
    attempts = max(_CANONICAL_ATTEMPTS, int(kwargs.get("attempt_count") or _CANONICAL_ATTEMPTS))
    # --eval-limit N is N tasks for the whole suite; a task's test pairs are scored together
    # (they are parts of one task, not more tasks) and its 2 attempts are attempts.
    n_tasks = len({meta.get("task_id") for _m, _g, meta in samples})
    limit_plan = eval_limit_plan("arc_agi", limit, n_tasks, attempts=attempts,
                                 unit="task attempts")
    limit_plan["test_pairs"] = len(samples)
    limit_plan["generations"] = len(samples) * attempts
    # ARCTester's retries: an attempt with no valid answer is asked again, up to
    # _RETRY_ATTEMPTS tries in all (the retries that ran are recorded as `retry_generations`).
    limit_plan["retry_generations_at_most"] = len(samples) * attempts * (_RETRY_ATTEMPTS - 1)
    run_kwargs = dict(
        eval_name="arc_agi",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        eval_fn=_eval_arc_agi,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        max_output_tokens=kwargs.get("max_output_tokens"),   # sovereign; base.py defaults when unset
        official_sampling=OFFICIAL_SAMPLING,
    )
    result = run_eval_suite(
        samples=_for_base(samples),
        # The loader already drew whole tasks; limiting again here would cut test pairs.
        limit=None,
        attempt_count=attempts,
        **run_kwargs,
    )
    rep = result.get("attempts") or {}
    k = rep.get("attempts_per_sample", attempts)
    traces = result.get("sample_traces") or []
    # Traces number `samples` in order (the loader drew this shard's tasks).
    shown = samples
    retry_report = _retry_failed_tries(traces, shown, run_kwargs)
    limit_plan["retry_generations"] = retry_report["retried_attempts"]
    result["sample_traces"] = traces
    result["eval_limit_plan"] = limit_plan
    per_task = _score_per_task(traces)
    # After the retries: an attempt whose last try got no reply from the endpoint is a harness
    # error (left out of the score, counted, named); an empty last reply is the model's.
    harness = [t for t in traces if not is_scored(t)]
    # An empty last reply that stopped with the answer inside the thinking channel is the model's
    # outcome too, but not an empty_response: it is counted in answer_in_thinking (base's
    # ANSWER_IN_THINKING_POLICY), over the attempts' last tries.
    empty = sum(1 for t in traces if is_scored(t) and t.get("status") == "EMPTY_RESPONSE"
                and t.get("health") != "empty_reasoned")
    result.update(answer_in_thinking_fields(
        sum(1 for t in traces if t.get("health") == "empty_reasoned"), len(traces)))
    no_reply = len(harness)
    cut_off = [t for t in traces if _cut_off(t)]
    result["failed_requests"] = no_reply + empty
    result["empty_responses"] = empty
    result["scoring_excluded"] = no_reply
    kinds: Dict[str, int] = {}
    for t in harness:
        kind = (t.get("harness_error") or {}).get("kind") or "left_out_of_the_score"
        kinds[kind] = kinds.get(kind, 0) + 1
    result.pop("request_harness_errors", None)
    if kinds:
        result["request_harness_errors"] = dict(sorted(kinds.items()))
    result["attempts_cut_off"] = len(cut_off)
    if traces:
        causes = []
        if no_reply:
            causes.append(f"harness errors left out of the score: {no_reply} attempt(s) whose "
                          f"last try got no reply from the served endpoint ("
                          + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())) + ")")
        if empty:
            causes.append(f"empty_responses {empty}")
        if len(cut_off) * 10 > len(traces):
            causes.append(f"attempts_cut_off {len(cut_off)} of {len(traces)} (over 10%)")
        result.pop("accuracy_withheld_reason", None)
        if not per_task["tasks"]:
            result["status"] = "failed"
            result["status_reason"] = "nothing was scored: " + ("; ".join(causes) or
                                                                 "every attempt left out")
        else:
            result["status"] = "completed_with_errors" if causes else "success"
            result["status_reason"] = "; ".join(causes) or None
    result["avg_at_k_accuracy"] = rep.get("avg_at_k")         # per pair, per attempt (first tries)
    result["pass_at_k_per_pair"] = rep.get("pass_at_k")       # per pair, unweighted by task
    result["arc_agi_report"] = {**per_task, **retry_report, "tries_per_attempt": _RETRY_ATTEMPTS}
    result["accuracy"] = per_task["score"]
    if traces and not per_task["tasks"]:
        result["accuracy_withheld_reason"] = result["status_reason"]
    # every task drawn; the headline's denominator is the scored ones
    result["total_questions"] = len({(t.get("extra_payload") or {}).get("task_id")
                                     or f"pair_{t.get('source_sample_idx', t.get('sample_idx'))}"
                                     for t in traces}) or per_task["tasks"]
    result["effective_n"] = per_task["tasks"]
    result["correct_answers"] = round(per_task["task_score_sum"], 2)
    result["category_accuracy"] = {"grid_transformation": {
        "correct": round(per_task["task_score_sum"], 2), "total": per_task["tasks"],
        "accuracy": per_task["score"]}}
    result["metric"] = (
        f"pass@{k} averaged per task (canonical ARC-AGI: a test pair is solved if either "
        f"attempt reproduces the gold grid exactly; a task scores the fraction of its pairs "
        f"solved; the headline is the mean over tasks; each attempt gets up to "
        f"{_RETRY_ATTEMPTS} tries)")
    noncanon = []
    if limit:
        noncanon.append(f"--eval-limit subset ({per_task['tasks']} of 400 tasks)")
    if k != _CANONICAL_ATTEMPTS:
        noncanon.append(f"{k} attempts per test pair (ARC-AGI allows 2)")
    if (result.get("official_sampling") or {}).get("temperature_matches_official") is False:
        noncanon.append(f"temperature {result.get('temperature')} "
                        f"({result.get('temperature_source')}; arc-agi-benchmarking sends none, "
                        "so the server default applies)")
    if no_reply:
        noncanon.append(
            f"{no_reply} attempt(s) got no reply from the served endpoint on their last try "
            "(request failed, timed out or did not fit the context window; harness errors, left "
            "out of the score)")
    if empty:
        noncanon.append(f"{empty} attempt(s) came back empty on their last try and count as "
                        "null attempts (wrong)")
    if cut_off:
        no_text = sum(1 for t in cut_off
                      if not (strip_thinking_tags(t.get("response_text")) or "").strip())
        loops = sum(1 for t in cut_off if t.get("health") in ("looping", "non_convergent"))
        noncanon.append(
            f"{len(cut_off)} attempt(s) were cut off at the max_tokens budget (max_tokens_sent "
            f"{result.get('max_tokens_sent')}) on their last try and scored on what arrived"
            + (f"; {no_text} of them with no answer text" if no_text else "")
            + (f"; {loops} of them looping or non_convergent, which a larger budget may not "
               "have let finish" if loops else ""))
    result["leaderboard_comparable"] = not noncanon
    if noncanon:
        result["leaderboard_comparable_reason"] = "; ".join(noncanon)
    else:
        result.pop("leaderboard_comparable_reason", None)
    return result
