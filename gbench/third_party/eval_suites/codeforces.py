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

"""Native Codeforces competitive programming evaluation suite.

Follows the protocol the `open-r1/codeforces` dataset card prescribes, which is the only
reproducible open Codeforces protocol there is. (Frontier labs report a Codeforces **Elo
percentile** obtained by simulating real contests; that needs the live platform and is not
reproducible here, so this is a pass-rate over a held-out split, not an Elo.)

    dataset `open-r1/codeforces` pinned at `_REVISION` (problems, prompts and the mirror)
    config  `verifiable`        - `executable` AND (`official_tests_complete` OR has
                                  `generated_tests`); 8,338 train / 422 test
    split   `test`              - "problems from late 2024 and early 2025"
    prompts `verifiable-prompts`- the dataset's own prompt, Python variant
    tests   `official_tests` + the separately-downloaded `generated_tests/` shards
    checker `generated_checker` - ~30% of problems accept MULTIPLE valid outputs
    limits  `time_limit` (s) / `memory_limit` (MB), per problem

What this replaced, on 2026-08-20, and why each mattered:

* `open-r1/codeforces-cots` **train** split. That is a chain-of-thought *distillation* set
  carrying five generations per problem, so the previous "500 samples" were **100 distinct
  problems counted five times each**. It also starts at 2010, well inside any current
  model's training window; the `test` split here is late-2024/2025 and the card says
  "Please avoid training on these."
* A hardcoded **40/30/30 CF/ICPC/IOI quota**, invented here and described as "balanced".
  IOI is 85 of 10,024 problems (0.85%), so the quota over-represented it ~35x, and 46% of
  IOI problems ship no `input_format` at all because they are grader-based - unrunnable as
  stdin/stdout, which is what produced IOI 17.33% against CF 68% / ICPC 76%. The natural
  distribution is used instead; no suite should invent a category mix.
* Grading against `examples`, the 1-2 sample cases printed in the problem statement
  (mean 1.95; 205 of 500 rows had exactly one). The card is explicit that platforms
  truncate visible tests to ~400 chars, so those cases "can be solved with an easy brute
  force solution".
* **Exact stdout match on every problem.** ~30% of Codeforces problems accept multiple
  valid answers and need a checker program; exact match marks correct solutions wrong.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_CODEFORCES_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import hashlib
import logging
import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union
from .base import apply_shard, finalize_shard_fields, is_scored, run_eval_suite, strip_thinking_tags
from .limit_plan import base_attempts, eval_limit_plan
from .sampling import stratified_sample, presharded_for_base
from .sandbox import run_sandboxed
from .swebench_common import infra_required, prereqs_path

logger = logging.getLogger(__name__)

DOCS_URL = "docs/evals/codeforces.md"

#: The dataset, and the subset/split its card prescribes for evaluation. Pinned to the
#: snapshot reconciled on 2026-09-24; the generated_tests mirror must come from the same one.
_REPO = "open-r1/codeforces"
_REVISION = "fbe3f6e903ee854eec2e69e9d96d0306cde59baf"
_CONFIG = os.environ.get("GBENCH_CODEFORCES_CONFIG", "verifiable")
_SPLIT = os.environ.get("GBENCH_CODEFORCES_SPLIT", "test")

#: `generated_tests` live in the SAME repo under `generated_tests/test_cases_<contest>.parquet`
#: but are NOT part of the dataset load - the full set is ~110GB. Point this at a directory
#: populated with:
#:     hf download open-r1/codeforces --repo-type dataset --revision <_REVISION> \
#:         --include 'generated_tests/*.parquet' --max-workers 8 --local-dir <DIR>
#: Only the shards for contests in the split are read, so a partial mirror is fine.
_GENERATED_TESTS_DIR = prereqs_path("codeforces", os.environ.get("GBENCH_CODEFORCES_GENERATED_TESTS")) or ""

#: Where the warning tells the operator to put the mirror. Sibling tools already live under
#: gbench-prereqs/ (aider, OJBench, SWE-bench-Live, ...), so the shards go beside them.
_SUGGESTED_MIRROR = os.environ.get(
    "GBENCH_PREREQS_DIR", os.path.expanduser("~/.cache/gbench/gbench-prereqs")) + "/codeforces"

#: Cap on generated test cases executed per problem. Canonical open-r1 verification runs ALL
#: generated tests (a solution that fails only a capped-out case must be marked wrong), so the
#: default is 0 = NO CAP. The shards carry a plain `test_i` index (NOT a hardness ordering), so a
#: cap would drop arbitrary cases, not "keep the hardest". Set the env var to bound wall-clock on
#: very large shards if you must, but that makes the run non-canonical (a subset of the tests).
_MAX_GENERATED_TESTS = int(os.environ.get("GBENCH_CODEFORCES_MAX_GENERATED_TESTS", "0"))

#: Below this many executed test cases a problem's verdict is called out as weak. Measured
#: 2026-08-20 on the full mirror: 32 of 422 problems, against a suite median of 29.
_THIN_TEST_THRESHOLD = 5

#: Fallback per-case wall-clock when a problem declares no `time_limit`, and the multiple
#: applied to the declared limit (CPython is far slower than the C++ the limits assume).
_DEFAULT_TIME_LIMIT_S = 10.0
_TIME_LIMIT_MULTIPLIER = float(os.environ.get("GBENCH_CODEFORCES_TIME_MULTIPLIER", "3.0"))

#: Where the extraction and the judging come from. open-r1's `extract_code`
#: (src/open_r1/rewards.py) and `score_submission` (src/open_r1/utils/competitive_programming/
#: cf_scoring.py) at this commit; the judge is the piston `codeforces` package's `run` script
#: (packages/codeforces/1.0.0/run) the open-r1/codeforces card points to, at this commit.
OPENR1_REF = "github.com/huggingface/open-r1@1416fa0cf21595d2083b399a2a0bbddd7f6e9563"
PISTON_REF = ("github.com/guipenedo/piston@7a38da202481bb7a85647a2c6cdc00990ad051bc "
              "packages/codeforces/1.0.0/run")

#: The run script's checker timeout (`run_with_timeout 10 python3 checker.py ...`), in seconds.
_CHECKER_TIMEOUT_S = 10.0


def _as_int(v: Any) -> int:
    """Ordering key, tolerant of the column arriving as str or int."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _load_generated_tests(contest_id: str, problem_id: str) -> List[Dict[str, str]]:
    """Generated cases for one problem, from a local `generated_tests/` mirror.

    Returns [] when no mirror is configured or the shard is absent or unreadable. The loader
    compares the count with the row's declared `generated_tests` and hard-errors on a
    shortfall, so a gap in the mirror can never quietly grade a problem on its official
    tests alone.
    """
    if not _GENERATED_TESTS_DIR:
        return []
    try:
        cid = int(str(contest_id))
    except (TypeError, ValueError):
        return []
    path = os.path.join(_GENERATED_TESTS_DIR, "generated_tests",
                        f"test_cases_{cid:04d}.parquet")
    if not os.path.exists(path):
        path = os.path.join(_GENERATED_TESTS_DIR, f"test_cases_{cid:04d}.parquet")
        if not os.path.exists(path):
            return []
    try:
        import pyarrow.parquet as pq
        tbl = pq.read_table(path).to_pydict()
    except Exception as e:                                              # noqa: BLE001
        logger.warning("codeforces: could not read %s: %s", path, e)
        return []
    # The card documents the ordering column as `test_case_i`; the shipped shards actually
    # use `test_i`. Accept either, and treat its absence as "unordered" rather than
    # indexing a one-element default, which raised IndexError on the second row.
    ids = tbl.get("problem_id") or []
    inputs = tbl.get("input") or []
    outputs = tbl.get("output") or []
    order = tbl.get("test_i") or tbl.get("test_case_i") or []
    if not (len(ids) == len(inputs) == len(outputs)):
        logger.warning("codeforces: %s has ragged columns (%d/%d/%d); skipping it. See %s",
                       os.path.basename(path), len(ids), len(inputs), len(outputs), DOCS_URL)
        return []
    rows = []
    pid = str(problem_id)
    for i, pid_i in enumerate(ids):
        if str(pid_i) != pid:
            continue
        rows.append({"input": str(inputs[i]), "output": str(outputs[i]),
                     "i": _as_int(order[i]) if i < len(order) else i})
    rows.sort(key=lambda r: r.get("i") or 0)
    if _MAX_GENERATED_TESTS > 0:
        rows = rows[:_MAX_GENERATED_TESTS]
    return [{"input": r["input"], "output": r["output"]} for r in rows]


def _tests_sha256(tests: List[Dict[str, Any]]) -> str:
    """sha256 over the cases in order: each case's input, NUL, output, NUL (UTF-8), hashed case
    by case so a problem with a gigabyte of generated tests is never copied into one string."""
    h = hashlib.sha256()
    for t in tests:
        h.update(str(t.get("input", "")).encode("utf-8"))
        h.update(b"\0")
        h.update(str(t.get("output", "")).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _resolve_gold(gold: Any, tests_store: Optional[Dict[str, Dict[str, Any]]]) -> Any:
    """A gold with its tests and checker, for the judge. A gold the loader built with a
    `tests_store` names them (`tests_ref`) instead of holding them; they are looked up in that
    run's store. Any other gold comes back as it is."""
    if not (isinstance(gold, dict) and gold.get("tests_ref") is not None
            and "official_tests" not in gold):
        return gold
    ref = gold["tests_ref"]
    if tests_store is None or ref not in tests_store:
        raise RuntimeError(f"codeforces: the tests of problem {ref!r} are not in this run's "
                           "test store, so the program cannot be judged")
    return {**gold, **tests_store[ref]}


def _load_codeforces_samples(enable_thinking: bool = False,
                             limit: Optional[int] = None,
                             tests_store: Optional[Dict[str, Dict[str, Any]]] = None
                             ) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load the `verifiable` / `test` split with its official prompts and full test sets.

    Every gold records its test counts and their sha256 (_tests_sha256) and the checker's
    sha256. With `tests_store` (a dict the run owns), the official and generated tests and the
    checker go into the store under the gold's `tests_ref` and the gold holds only those counts
    and digests: the gold is stored in every trace, and the generated tests of the full split
    are about 20 GB. Without a store the gold holds them too."""
    from datasets import load_dataset
    ds = list(load_dataset(_REPO, _CONFIG, split=_SPLIT, revision=_REVISION))

    # The dataset ships its own prompt (Python and C++ variants). Use it rather than a bespoke
    # one so the number is about the model, not about our phrasing. It is REQUIRED: silently
    # swapping in a built-in prompt would change the measured quantity, so a load failure
    # hard-errors (never a silent degradation).
    prompts: Dict[str, str] = {}
    try:
        for r in load_dataset(_REPO, "verifiable-prompts", split=_SPLIT, revision=_REVISION):
            if str(r.get("language", "")).lower().startswith("py"):
                prompts[str(r.get("id"))] = str(r.get("prompt") or "")
    except Exception as e:                                              # noqa: BLE001
        raise infra_required(
            "codeforces",
            f"the canonical prompts ('{_REPO}' config 'verifiable-prompts', split '{_SPLIT}') "
            f"could not be loaded ({e}). They ship in the same dataset repo; ensure it is "
            "downloaded/cached (needs network on first load). Refusing to substitute a "
            "non-canonical built-in prompt.",
            DOCS_URL) from e
    if not prompts:
        raise infra_required(
            "codeforces",
            f"'{_REPO}' 'verifiable-prompts'/{_SPLIT} loaded no Python prompts; cannot run "
            "canonically without the dataset's own prompts.",
            DOCS_URL)

    # Seeded random subset, NOT stratified by contest_type. Stratifying would balance the
    # categories, which is a milder version of the 40/30/30 quota this replaced: IOI is
    # 0.85% of the corpus and 1 of the 422 test problems, so any key that spreads across
    # contest types over-represents it again. The dataset's own distribution IS the
    # measurement. `stratified_sample` with no key degrades to a seeded subset, which still
    # beats a contiguous head (dataset order correlates with contest date and difficulty).
    # --shard first, over the full ordered set, then this draw within the shard.
    ds, _ = apply_shard(ds, "codeforces")
    ds = stratified_sample(ds, limit, None, seed="codeforces")

    samples = []
    incomplete: List[str] = []
    n_gen = n_checker = 0
    for item in ds:
        pid = str(item.get("id"))
        official = [t for t in (item.get("official_tests") or []) if isinstance(t, dict)]
        generated = _load_generated_tests(item.get("contest_id"), pid)
        # The row declares how many generated cases it has (all 422 test problems matched the
        # mirror exactly on 2026-09-24); fewer means a missing or partial shard.
        declared = _as_int(item.get("generated_tests"))
        expected = min(declared, _MAX_GENERATED_TESTS) if _MAX_GENERATED_TESTS > 0 else declared
        if len(generated) < expected:
            incomplete.append(f"{pid} (contest {item.get('contest_id')}: "
                              f"{len(generated)}/{expected})")
        if generated:
            n_gen += 1
        checker = item.get("generated_checker") or ""
        if checker:
            n_checker += 1

        prompt = prompts.get(pid)
        if not prompt:
            # Every problem must carry the dataset's own canonical prompt; a bespoke substitute
            # would change the measured quantity. A missing one is a data gap, not something to
            # paper over.
            raise RuntimeError(
                f"codeforces: no canonical Python prompt for problem {pid!r} in "
                f"'{_REPO}' verifiable-prompts/{_SPLIT}; refusing to substitute a built-in "
                f"prompt. See {DOCS_URL}")

        gold = {
            "problem_id": pid,
            "official_tests_count": len(official),
            "generated_tests_count": len(generated),
            "tests_sha256": {"official_tests": _tests_sha256(official),
                             "generated_tests": _tests_sha256(generated)},
            "has_checker": bool(checker),
            "checker_sha256": (hashlib.sha256(str(checker).encode("utf-8")).hexdigest()
                               if checker else None),
            "time_limit": item.get("time_limit"),
            "memory_limit": item.get("memory_limit"),
            "testset_size": item.get("testset_size"),
            "official_tests_complete": bool(item.get("official_tests_complete")),
            # "stdio", or "file" for older problems that read input.txt and write output.txt
            # (the run script's INPUT_MODE); every verifiable/test row is stdio.
            "input_mode": str(item.get("input_mode") or "stdio"),
        }
        bulky = {"official_tests": official, "generated_tests": generated, "checker": checker}
        if tests_store is None:
            gold.update(bulky)
        else:
            ref = pid if pid not in tests_store else f"{pid}#{len(tests_store)}"
            tests_store[ref] = bulky
            gold["tests_ref"] = ref
        meta = {"category": str(item.get("contest_type") or "CF"),
                "rating": item.get("rating"),
                "year": item.get("contest_start_year")}
        samples.append(([{"role": "user", "content": prompt}], gold, meta))

    if incomplete:
        raise infra_required(
            "codeforces",
            f"the generated_tests mirror at {_GENERATED_TESTS_DIR} is missing cases for "
            f"{len(incomplete)} problem(s) in the split, e.g. {', '.join(incomplete[:5])}. "
            f"Grading those on official_tests alone would be a partial run. Re-download the "
            f"shards (--revision {_REVISION}) for the contests listed.",
            DOCS_URL)
    logger.info("Loaded %d codeforces problems (%s/%s); %d with generated tests, "
                "%d needing a checker.", len(samples), _CONFIG, _SPLIT, n_gen, n_checker)
    return samples


def _require_generated_tests_mirror() -> None:
    """The generated_tests mirror is REQUIRED. `official_tests` alone is a median 9.7% of each
    problem's real suite (the card: the visible cases 'solved with an easy brute force
    solution'), so grading on them is a near-meaningless partial - hard-error instead."""
    if _GENERATED_TESTS_DIR and os.path.isdir(_GENERATED_TESTS_DIR):
        return
    raise infra_required(
        "codeforces",
        "the generated_tests mirror is required (official_tests alone is ~9.7% coverage). "
        "Download the shards and point GBENCH_CODEFORCES_GENERATED_TESTS at the directory:\n"
        f"  hf download open-r1/codeforces --repo-type dataset --revision {_REVISION} "
        f"--include 'generated_tests/*.parquet' --max-workers 8 --local-dir {_SUGGESTED_MIRROR}\n"
        f"  export GBENCH_CODEFORCES_GENERATED_TESTS={_SUGGESTED_MIRROR}",
        DOCS_URL)


def extract_code(completion: str, language: Optional[str] = "python") -> str:
    """open-r1's `extract_code`, verbatim (OPENR1_REF src/open_r1/rewards.py): the LAST
    ```<language> fenced block, or '' when there is none.

    The closing fence is the next ``` wherever it stands, indented or not, so a reply whose
    drafts hold indented fences still yields its final program. open-r1 scores '' as no
    submission (cf_scoring.score_submission's `no_submission_reward`), so a reply with no
    ```python block fails; the reply is never run as code."""
    if language is None:
        return ""
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


# The judge: the piston `codeforces` package's `run` script (PISTON_REF) for a python3
# submission, step by step. What it does to the files before comparing them:
#   * the test input loses a carriage return at the end of each line (`sed -i 's/\r$//'`), before
#     the program reads it;
#   * the program's output loses its empty lines (`sed -i '/^$/d'`);
#   * the expected output is read with `$(cat ...)` (trailing newlines dropped) and written back
#     with `echo` (one newline added), then loses carriage returns at line ends and empty lines.
# Then, with no checker, both outputs go through `normalize_output` (whole-word yes/no folded to
# YES/NO case-insensitively, every run of whitespace one space, trimmed) and must be equal
# (`diff -bq`). With a checker, `python3 checker.py <input> <expected> <output>` runs for at most
# 10 s and the submission passes iff it exits 0 and its stdout, all whitespace removed, is "1" or
# "100" (open-r1 then requires the run's exit code 0 and stdout "1"). The script's
# `ulimit -s unlimited` applies to the program and the checker. The script works on bytes in the
# C locale, and so does this port: `\b` and case folding are ASCII, whitespace is [ \t\n\v\f\r].

#: Whitespace as the C locale's `[:space:]` class has it.
_C_SPACE = rb"[ \t\n\v\f\r]"

#: A bash prefix that applies the run script's `ulimit -s unlimited` (the hard limit when that is
#: lower), enters the working directory and executes the command.
_LAUNCH = ('ulimit -s unlimited 2>/dev/null || ulimit -s hard 2>/dev/null; '
           'cd "$1" || exit 125; shift; exec "$@"')


def _as_bytes(data: Union[str, bytes, None]) -> bytes:
    if isinstance(data, bytes):
        return data
    return str(data or "").encode("utf-8")


def _strip_cr(data: bytes) -> bytes:
    """`sed 's/\\r$//'`: one carriage return removed from the end of every line."""
    return b"\n".join(line[:-1] if line.endswith(b"\r") else line
                      for line in data.split(b"\n"))


def _drop_empty_lines(data: bytes) -> bytes:
    """`sed '/^$/d'`: lines with no character at all removed (a line holding only spaces or a
    carriage return stays)."""
    if not data:
        return data
    lines = data.split(b"\n")
    terminated = lines[-1] == b""
    if terminated:
        lines.pop()
    kept = [line for line in lines if line]
    return b"\n".join(kept) + (b"\n" if kept and terminated else b"")


def _expected_as_judged(expected: Union[str, bytes, None]) -> bytes:
    """The expected output as the run script leaves correct_output.txt: `$(cat ...)` drops the
    trailing newlines, `echo` adds one, then carriage returns at line ends and empty lines go."""
    return _drop_empty_lines(_strip_cr(_as_bytes(expected).rstrip(b"\n") + b"\n"))


def _normalize_output(data: Union[str, bytes, None]) -> bytes:
    """The run script's `normalize_output`: `sed -E 's/\\b(yes)\\b/YES/ig; s/\\b(no)\\b/NO/ig'`,
    newlines to spaces, whitespace runs squeezed to one space, leading/trailing space trimmed."""
    folded = re.sub(rb"\b(yes)\b", b"YES", _as_bytes(data), flags=re.IGNORECASE)
    folded = re.sub(rb"\b(no)\b", b"NO", folded, flags=re.IGNORECASE)
    return b" ".join(tok for tok in re.split(_C_SPACE + b"+", folded) if tok)


def _outputs_match(actual: Union[str, bytes, None], expected: Union[str, bytes, None]) -> bool:
    """The run script's diff path: `diff -bq` of the two normalized outputs, i.e. the same
    tokens after whole-word yes/no case folding. The dataset carries the platform's `\\r\\n`
    line endings verbatim, which this ignores, as the script does."""
    return _normalize_output(actual) == _normalize_output(expected)


def _launch(workdir: str, argv: List[str]) -> List[str]:
    return ["bash", "-c", _LAUNCH, "run", workdir, *argv]


def _write(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)


def _checker_accepts(checker: str, test_input: bytes, expected: bytes, output: bytes) -> bool:
    """The run script's checker path: `python3 checker.py <input> <expected> <output>` (argv:
    input, jury answer, submission) for at most _CHECKER_TIMEOUT_S; accepted iff it exits 0 and
    prints "1" or "100" (all whitespace removed). The checker runs in the sandbox and reads the
    submission's output as a file."""
    with tempfile.TemporaryDirectory(prefix="gbench_cf_checker_") as work:
        paths = {name: os.path.join(work, name) for name in
                 ("checker.py", "input.txt", "correct_output.txt", "output.txt")}
        _write(paths["checker.py"], _as_bytes(checker))
        _write(paths["input.txt"], test_input)
        _write(paths["correct_output.txt"], expected)
        _write(paths["output.txt"], output)
        try:
            proc = run_sandboxed(
                _launch(work, [sys.executable, paths["checker.py"], paths["input.txt"],
                               paths["correct_output.txt"], paths["output.txt"]]),
                writable=[work], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, timeout=_CHECKER_TIMEOUT_S)
        except (subprocess.TimeoutExpired, Exception):                  # noqa: BLE001
            return False
    verdict = re.sub(_C_SPACE, b"", proc.stdout or b"")
    return proc.returncode == 0 and verdict in (b"1", b"100")


def _run_case(code: str, test: Dict[str, str], timeout: float,
              checker: str, input_mode: str = "stdio") -> bool:
    """One test case, judged as the run script judges it (see above). True iff accepted.

    The program is main.py in a fresh working directory, run with `python main.py` as the script
    runs `python3 $1`. input_mode "file" (older problems) writes the input to input.txt and reads
    the answer from output.txt, the program's stdout discarded, as the script does."""
    test_input = _strip_cr(_as_bytes(test.get("input")))
    expected = _expected_as_judged(test.get("output"))
    file_mode = str(input_mode or "stdio") == "file"
    with tempfile.TemporaryDirectory(prefix="gbench_cf_") as work:
        _write(os.path.join(work, "main.py"), _as_bytes(code))
        if file_mode:
            _write(os.path.join(work, "input.txt"), test_input)
        try:
            res = run_sandboxed(
                _launch(work, [sys.executable, "main.py"]), writable=[work],
                input=b"" if file_mode else test_input,
                stdout=subprocess.DEVNULL if file_mode else subprocess.PIPE,
                stderr=subprocess.DEVNULL, timeout=timeout)
        except (subprocess.TimeoutExpired, Exception):                  # noqa: BLE001
            return False
        if res.returncode != 0:
            return False
        if file_mode:
            try:
                with open(os.path.join(work, "output.txt"), "rb") as f:
                    output = f.read()
            except OSError:
                return False
        else:
            output = res.stdout or b""
    output = _drop_empty_lines(output)
    if not checker:
        return _outputs_match(output, expected)
    return _checker_accepts(checker, test_input, expected, output)


def _eval_codeforces(response_text: str, gold: Any) -> bool:
    """Accepted iff EVERY official and generated test case passes.

    The program is open-r1's extract_code of the reply (the last ```python block); a reply with
    none is no submission and fails. A problem with no usable tests is reported incorrect,
    never auto-passed: an untested submission measured nothing, and crediting it inflates the
    suite with rows that carry no signal.
    """
    if not response_text:
        return False
    if isinstance(gold, list):          # legacy shape: a bare list of {input, output}
        gold = {"official_tests": gold, "generated_tests": [], "checker": ""}
    if not isinstance(gold, dict):
        return False

    code = extract_code(response_text, "python")
    if not code:
        return False

    tests = [t for t in (gold.get("official_tests") or []) if isinstance(t, dict)]
    tests += [t for t in (gold.get("generated_tests") or []) if isinstance(t, dict)]
    if not tests:
        return False

    tl = gold.get("time_limit")
    try:
        timeout = float(tl) * _TIME_LIMIT_MULTIPLIER if tl else _DEFAULT_TIME_LIMIT_S
    except (TypeError, ValueError):
        timeout = _DEFAULT_TIME_LIMIT_S
    timeout = max(1.0, min(timeout, 60.0))

    checker = str(gold.get("checker") or "")
    input_mode = str(gold.get("input_mode") or "stdio")
    return all(_run_case(code, t, timeout, checker, input_mode) for t in tests)


def _test_counts(gold: Dict[str, Any]) -> Tuple[int, int]:
    """(official, generated) test counts of a gold, stored (counts) or holding its tests."""
    def _n(field: str) -> int:
        if isinstance(gold.get(f"{field}_count"), int):
            return gold[f"{field}_count"]
        return len(gold.get(field) or [])
    return _n("official_tests"), _n("generated_tests")


#: The result fields _record_coverage computes from the traces.
_COVERAGE_FIELDS = ("test_coverage_pct", "tests_per_problem_median", "tests_per_problem_min",
                    "problems_under_5_tests", "problems_with_generated_tests",
                    "problems_with_checker", "replies_without_python_block")


def _record_coverage(result: Dict[str, Any]) -> None:
    """How much of each problem's real test suite was executed, from the traces' golds, and the
    scored replies with no ```python block. Without it a thin-coverage run is indistinguishable
    from a rigorous one."""
    for field in _COVERAGE_FIELDS:
        result.pop(field, None)
    covs, per_problem, with_gen, with_chk = [], [], 0, 0
    for tr in result.get("sample_traces", []) or []:
        g = tr.get("gold_answer")
        if not isinstance(g, dict):
            continue
        n_official, n_generated = _test_counts(g)
        n = n_official + n_generated
        per_problem.append(n)
        ts = g.get("testset_size") or 0
        if ts:
            covs.append(min(1.0, n / ts))
        if n_generated:
            with_gen += 1
        if g.get("has_checker", bool(g.get("checker"))):
            with_chk += 1
    if covs:
        result["test_coverage_pct"] = round(sum(covs) / len(covs) * 100.0, 1)
    # Absolute counts alongside the ratio. `test_coverage_pct` divides by `testset_size`,
    # the real hidden suite, which ranges from a handful to 198 - so a problem run on 29
    # adversarial tests against a 71-test official suite reads as "41% covered" while being
    # thoroughly tested. open-r1 ranked candidate tests by hardness (using wrong solutions
    # that passed the public tests) and kept the hardest, so these are not random samples.
    # The number that says whether a verdict is trustworthy is how many tests actually ran.
    if per_problem:
        per_problem.sort()
        mid = per_problem[len(per_problem) // 2]
        result["tests_per_problem_median"] = mid
        result["tests_per_problem_min"] = per_problem[0]
        thin = sum(1 for n in per_problem if n < _THIN_TEST_THRESHOLD)
        result["problems_under_5_tests"] = thin
        if thin:
            logger.warning(
                "[codeforces] %d/%d problems were graded on fewer than %d test cases "
                "(median across the suite is %d). Those verdicts are weak in both "
                "directions - open-r1 discards generated tests when correct solutions "
                "disagree, so these are the problems where generation mostly failed. "
                "See %s", thin, len(per_problem), _THIN_TEST_THRESHOLD, mid, DOCS_URL)
    result["problems_with_generated_tests"] = with_gen
    result["problems_with_checker"] = with_chk
    # A scored reply with no ```python block is no submission (open-r1's extract_code returns
    # ''), which fails; counted so a missing block reads apart from a wrong program.
    result["replies_without_python_block"] = sum(
        1 for tr in result.get("sample_traces", []) or []
        if is_scored(tr) and not extract_code(strip_thinking_tags(tr.get("response_text")), "python"))


def merge_shards(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The merged result of a sharded Codeforces run (`gbench --merge`): the generic merge (the
    union of the shards' traces re-counted; the accuracy is correct over scored, as each
    shard's is) with the test-coverage fields recomputed over that union from the stored golds
    (their test counts: the tests themselves are not in the traces)."""
    from ...merge import generic_merge
    merged = generic_merge("codeforces", results)
    _record_coverage(merged)
    unreconciled = merged.get("merge_unreconciled_fields")
    if isinstance(unreconciled, dict):
        for field in _COVERAGE_FIELDS:
            unreconciled.pop(field, None)
        if not unreconciled:
            merged.pop("merge_unreconciled_fields")
    return merged


def run_codeforces(model_name: str, base_url: str, concurrency: int,
                   enable_thinking: bool = False, **kwargs) -> Dict[str, Any]:
    """Run native Codeforces competitive programming evaluation suite."""
    _require_generated_tests_mirror()   # fail fast: full test coverage is required, not optional
    # Fail fast on a broken sandbox. _run_case() catches every exception and returns False, so a
    # missing/blocked bubblewrap would otherwise be swallowed sample-by-sample and reported as a
    # fake 0% (every problem "incorrect") instead of the infra error it is. GBENCH_SANDBOX=none is
    # the explicit unsandboxed opt-out and is honoured; any other mode requires working isolation.
    from .sandbox import sandbox_mode, sandbox_available, sandbox_skip_reason
    if sandbox_mode() != "none" and not sandbox_available():
        raise infra_required(
            "codeforces",
            "executes model-written code and requires bubblewrap process isolation, which is "
            "unavailable or blocked here: "
            + (sandbox_skip_reason() or "the bubblewrap probe failed")
            + " (set GBENCH_SANDBOX=none to run UNSANDBOXED on purpose)",
            DOCS_URL)
    # The tests and checkers stay in this run's store, never in a trace (_load_codeforces_samples);
    # the store is emptied when the run ends.
    tests_store: Dict[str, Dict[str, Any]] = {}
    try:
        samples = _load_codeforces_samples(enable_thinking=enable_thinking,
                                           limit=kwargs.get("limit"), tests_store=tests_store)
        limit_plan = eval_limit_plan("codeforces", kwargs.get("limit"), len(samples),
                                     attempts=base_attempts())

        def _judge(response_text: str, gold: Any) -> bool:
            return _eval_codeforces(response_text, _resolve_gold(gold, tests_store))

        result = run_eval_suite(
            eval_name="codeforces",
            model_name=model_name,
            base_url=base_url,
            concurrency=concurrency,
            samples=presharded_for_base(samples),  # this shard's problems, kept as is
            eval_fn=_judge,
            thinking=enable_thinking,
            extra_payload=kwargs.get("extra_payload"),
            limit=kwargs.get("limit"),
            max_output_tokens=kwargs.get("max_output_tokens"),
        )
    finally:
        tests_store.clear()
    _record_coverage(result)
    result["eval_limit_plan"] = limit_plan
    result["dataset"] = f"{_REPO}@{_REVISION}:{_CONFIG}/{_SPLIT}"
    result["generated_tests_mirrored"] = bool(_GENERATED_TESTS_DIR)
    # 0 = no cap (every generated case runs, the canonical default).
    result["max_generated_tests"] = _MAX_GENERATED_TESTS
    result["tests_in_traces"] = (
        "not stored: each trace's gold_answer names its problem (problem_id) in `dataset` with "
        "its official and generated test counts, their sha256 and the checker's sha256")
    result["extraction"] = (f"{OPENR1_REF} src/open_r1/rewards.py extract_code: the last "
                            "```python block; none is no submission (fails)")
    result["judge"] = (f"{PISTON_REF} (the run script the open-r1/codeforces card points to), "
                       "ported for python3: carriage returns at line ends stripped from the input, "
                       "empty output lines "
                       "dropped, yes/no case folded and whitespace normalized for the diff; a "
                       "checker passes on exit 0 with stdout 1 or 100; stack unlimited")
    reasons = [
        "pass rate over the open-r1/codeforces held-out split, not the Codeforces Elo/percentile "
        "labs publish (that needs live contest simulation); CPython time limits scaled by "
        f"{_TIME_LIMIT_MULTIPLIER}x and memory limits not enforced"]
    if _MAX_GENERATED_TESTS > 0:
        reasons.append(f"generated tests capped at {_MAX_GENERATED_TESTS} per problem "
                       "(GBENCH_CODEFORCES_MAX_GENERATED_TESTS), so each problem ran an arbitrary "
                       "subset of its tests")
    if (_CONFIG, _SPLIT) != ("verifiable", "test"):
        reasons.append(f"config/split override {_CONFIG}/{_SPLIT} (GBENCH_CODEFORCES_CONFIG/"
                       "_SPLIT); the card's held-out set is verifiable/test")
    if kwargs.get("limit"):
        reasons.append(f"subset run (--eval-limit {kwargs.get('limit')})")
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = "; ".join(reasons)
    result["metric"] = ("all official+generated tests pass (checker-aware); NOT the "
                        "Codeforces Elo percentile frontier labs report")
    # One shard of a --shard I/N run is a partial set: the shard reason (finalize_shard_fields).
    return finalize_shard_fields(result)
