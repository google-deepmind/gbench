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

# Evaluation Suite: multipl_e
# Description: MultiPL-E - execution-based pass@1 across languages (HumanEval family)

"""gbench native built-in runner for multipl_e (Polyglot Coding & Software).

Canonical MultiPL-E (nuprl/MultiPL-E): HumanEval translated into 23 target languages, each program
run as `prompt + completion + "\\n" + tests`; canonical metric = unbiased pass@1 as pass_k.py
computes it (per problem c/n over its completions, then the mean over problems; see
per_problem_pass_at_1). A completion the evaluator returned no result for is a harness error, left
out of the score.

Generation follows the pinned revision's own CHAT protocol (MultiPL-E 3025a531,
dataset_builder/chat_completions.py, which the upstream README names as the way to evaluate chat
models): a DSPy ChainOfThought over its `Completion` signature, formatted by DSPy 3.0.0b1's
ChatAdapter (a system message describing the fields and a user message carrying `lang` and the
program `prefix`). gbench sends those two messages verbatim, parses the `[[ ## completion ## ]]`
field exactly like ChatAdapter.parse, and post-processes it with the script's own
`postprocess_completion`. The evaluation file then carries `prompt: ""` and the completion as the
whole program, as upstream writes it.

A reply ChatAdapter.parse rejects is scored as upstream scores it, with completion "". The script
calls `completion_cot.aforward`, and in DSPy 3.0.0b1 that async path (Predict.aforward ->
Adapter.acall) has no JSONAdapter fallback: ChatAdapter overrides only the synchronous `__call__`
with one. The async fallback (ChatAdapter.acall) came in stanfordnlp/dspy commit 1df67689
(2025-06-21), after the 3.0.0b1 tag (c3e79ce8, 2025-06-11). So the AdapterParseError reaches the
script's `do_task`, which records `completion: ""`, and such a run stays comparable. An EMPTY reply
(message content "" or null) is different: DSPy's Adapter._call_postprocess tests `if text:`,
parses nothing and returns every field as None, and the script's postprocess_completion then fails
on None outside do_task's error handling, which stops upstream's run. gbench scores an empty reply
as an empty program and reports the run as not comparable. The test is on the raw content, as
DSPy's: a reply of whitespace only is not empty there, so it is a parse failure (completion "").

chat_completions.py is licensed BSD-3-Clause with a Machine Learning Restriction, so gbench does
not ship it: the suite downloads it from the pinned commit at run time, checks its sha256, caches it
under $GBENCH_CACHE_DIR/multipl_e/ and reads `stop_at_stop_token`, `postprocess_completion` and the
`Completion` signature (docstring and field descriptions) out of it. The suite hard-errors
(infra_required) if the file cannot be fetched or does not match.

Execution is delegated to the OFFICIAL MultiPL-E evaluator, built LOCALLY (never pulled) into the
image `gbench-multipl-e:3025a531` with MultiPL-E's own evaluation/Dockerfile and evaluation/ directory at the
pinned commit, downloaded and sha256-checked at build time (docker/multipl_e.Dockerfile, which also
moves the recipe's javatuples jar download to Maven Central, sha256-checked, because the original
host now serves an HTML page); it ships the language toolchains and the canonical per-language
status rule
(OK/SyntaxError/Exception/Timeout). See docs/evals/multipl_e.md for the one-time build. The suite
HARD-ERRORS (infra_required, never skips, never a fabricated 0%) if Docker or the image is
unavailable.

Sampling: the upstream chat recipe (OFFICIAL_SAMPLING): no top_p (the script sends top_p only
when --top-p is given). The temperature is gbench's rule for every suite (base.resolve_temperature),
not the recipe's 0.2: 1.0 with --thinking, 0.0 without. `--temperature`, or
`GBENCH_MULTIPL_E_TEMPERATURE` for this suite alone (it takes precedence), overrides it in either
mode; the result records the recipe's 0.2 under `official_sampling`. Completions per problem: the
recipe's 20 (a larger `--attempt-count` is honoured); `GBENCH_MULTIPL_E_SAMPLES` sets another count
explicitly.

max_tokens follows base's token-limit policy (OFFICIAL_SAMPLING.max_tokens). The script has one
mode: dspy.LM(max_tokens=...) from its --max-tokens, default 2048, which the recipe does not
change, and no setting for reasoning models (DSPy 3.0.0b1's dspy.LM only refuses OpenAI o-series
names unless temperature=1.0 and max_tokens >= 20000, an OpenAI API constraint). So a no-think
request carries 2048 (less only when its prompt leaves less room in the served context window,
which the result names) and the run's --max-output-tokens is not applied; with --thinking the
2048 cap, written for non-reasoning generation, is not applied and each request is sent the run's
budget (--max-output-tokens, else base's thinking floor). `max_output_tokens_note` states the
mode's choice.
A reply that stopped at a limit other than the script's 2048 (with --thinking, the run's budget;
without it, a max_tokens the context window lowered) is counted in `replies_cut_off_by_budget`, and
those with no answer text, cut off inside the reasoning channel, in `replies_cut_off_in_reasoning`.
Fewer than 20 completions, another temperature, any sampling field the script does not send
(decoding penalties included), or such a cut-off reply is a valid pass@1 but is reported as
leaderboard_comparable=False.
"""

import ast
import glob
import gzip
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from .metrics import budget_cutoffs
from .limit_plan import eval_limit_plan
from .sampling import stratified_sample, presharded_for_base
from .base import (OFFICIAL_SAMPLING_PARAM_KEYS, OfficialSampling, apply_shard,
                   finalize_shard_fields, gbench_cache_dir, is_scored, resolve_token_budget,
                   run_eval_suite, strip_thinking_tags)
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

PILLAR = "Coding & Software Engineering"
DOCS_URL = "docs/evals/multipl_e.md"

#: The canonical HumanEval MultiPL-E language configs (nuprl/MultiPL-E `humaneval-<lang>`), all
#: provided by the official evaluator image (23 of the dataset's 24; Ada is excluded below). MultiPL-E TRANSLATES Python's HumanEval into other
#: languages, so there is no `humaneval-py` config (requesting it raises BuilderConfig-not-found);
#: Python is intentionally absent. Ada (`adb`) ships in a SEPARATE upstream dockerfile and is
#: excluded from the default image; add it via GBENCH_MULTIPL_E_LANGS if the image includes it.
_CANONICAL_LANGS = [
    "cpp", "cs", "d", "go", "java", "js", "jl", "lua", "php", "pl", "r", "rb",
    "rkt", "rs", "scala", "sh", "swift", "ts", "clj", "dart", "elixir", "hs", "ml",
]

_IMAGE_DEFAULT = "gbench-multipl-e:3025a531"
#: HF commit of nuprl/MultiPL-E the prompts and tests are read from.
_DATASET_REVISION = "28441b6024e71d4a1c1c0f6bf171c935cd5a43f2"

#: MultiPL-E's chat-model script at the pinned commit, fetched at run time (see the docstring).
UPSTREAM_COMMIT = "3025a531af7450e7df8b96fe0440e9804480bbad"
CHAT_SCRIPT = "dataset_builder/chat_completions.py"
CHAT_SCRIPT_URL = (f"https://raw.githubusercontent.com/nuprl/MultiPL-E/{UPSTREAM_COMMIT}/"
                   f"{CHAT_SCRIPT}")
#: sha256 of that file (git blob 72b8aeb9576b96de80f81de7d5d530b856f0378e, 13282 bytes).
CHAT_SCRIPT_SHA256 = "604b7afc237ae416e02c97166759b3a56667ad76bd214f6799fcd1d3cfbef29f"

#: max_tokens the script sends: dspy.LM(max_tokens=...) from --max-tokens, default 2048.
UPSTREAM_MAX_TOKENS = 2048

#: The recipe in chat_completions.py's docstring (`--max-completions 20 ... --temperature 0.2`);
#: `--top-p` defaults to None, and the script then sends no top_p; --max-tokens defaults to 2048.
OFFICIAL_SAMPLING = OfficialSampling(
    reference=(f"github.com/nuprl/MultiPL-E@{UPSTREAM_COMMIT[:8]} {CHAT_SCRIPT}: "
               "--temperature 0.2 --max-completions 20; top_p not sent (--top-p default None); "
               "max_tokens 2048 (--max-tokens default)"),
    temperature=0.2, max_tokens=UPSTREAM_MAX_TOKENS)

# DSPy's ChatAdapter text below (the frame of _chat_system and _CHAT_OUTPUT_REQUIREMENTS) and
# `parse_chat_fields`, a port of ChatAdapter.parse, come from DSPy 3.0.0b1. DSPy's license
# (LICENSE at tag 3.0.0b1):
#
#     MIT License
#
#     Copyright (c) 2023 Stanford Future Data Systems
#
#     Permission is hereby granted, free of charge, to any person obtaining a copy
#     of this software and associated documentation files (the "Software"), to deal
#     in the Software without restriction, including without limitation the rights
#     to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#     copies of the Software, and to permit persons to whom the Software is
#     furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included in all
#     copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#     SOFTWARE.

_CHAT_OUTPUT_REQUIREMENTS = (
    "Respond with the corresponding output fields, starting with the field "
    "`[[ ## reasoning ## ]]`, then `[[ ## completion ## ]]`, and then ending with the marker for "
    "`[[ ## completed ## ]]`.")
_OUTPUT_FIELDS = ("reasoning", "completion")
_FIELD_HEADER = re.compile(r"\[\[ ## (\w+) ## \]\]")


# --- the upstream script, fetched at run time ------------------------------------------------

class _TqdmStub:
    """postprocess_completion calls `tqdm.write` for a log line only; this keeps it a log line."""

    @staticmethod
    def write(msg: Any, *args: Any, **kwargs: Any) -> None:
        logger.debug("multipl_e: %s", msg)


class _Upstream:
    """What gbench uses from chat_completions.py, read out of the fetched file."""

    def __init__(self, text: str):
        tree = ast.parse(text)
        funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
        missing = [n for n in ("stop_at_stop_token", "postprocess_completion") if n not in funcs]
        if missing or "Completion" not in classes:
            raise RuntimeError(f"multipl_e: {CHAT_SCRIPT} has no {missing or ['Completion']}")
        ns: Dict[str, Any] = {"Path": Path, "List": List, "Optional": Optional,
                              "tqdm": _TqdmStub}
        for name in ("stop_at_stop_token", "postprocess_completion"):
            exec(compile(ast.get_source_segment(text, funcs[name]),
                         f"multipl_e_{CHAT_SCRIPT}", "exec"), ns)
        self.stop_at_stop_token = ns["stop_at_stop_token"]
        self.postprocess_completion = ns["postprocess_completion"]
        cls = classes["Completion"]
        #: The signature's instructions, as DSPy reads a class docstring (cleaned).
        self.instructions = ast.get_docstring(cls, clean=True) or ""
        self.descriptions: Dict[str, str] = {}
        for node in cls.body:
            if (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                    and isinstance(node.value, ast.Call)):
                for kw in node.value.keywords:
                    if kw.arg in ("desc", "description") and isinstance(kw.value, ast.Constant):
                        self.descriptions[node.target.id] = str(kw.value.value)
        if set(self.descriptions) != {"lang", "prefix", "completion"} or not self.instructions:
            raise RuntimeError(f"multipl_e: {CHAT_SCRIPT}'s Completion signature does not have "
                               f"the lang/prefix/completion fields ({sorted(self.descriptions)})")


_UPSTREAM: Optional[_Upstream] = None
_UPSTREAM_LOCK = threading.Lock()


def chat_script_cache_path() -> Path:
    return Path(gbench_cache_dir()) / "multipl_e" / f"chat_completions-{UPSTREAM_COMMIT[:12]}.py"


def fetch_chat_script() -> str:
    """chat_completions.py at UPSTREAM_COMMIT: from the cache if its sha256 matches, else
    downloaded, checked and cached. Raises RuntimeError naming what failed."""
    path = chat_script_cache_path()
    if path.exists():
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() == CHAT_SCRIPT_SHA256:
            return data.decode("utf-8")
        logger.warning("multipl_e: cached %s does not match the pinned sha256; fetching it again",
                       path)
    try:
        with urllib.request.urlopen(CHAT_SCRIPT_URL, timeout=60) as resp:
            data = resp.read()
    except Exception as e:                                              # noqa: BLE001
        raise RuntimeError(f"multipl_e: could not download {CHAT_SCRIPT_URL} ({e}). The suite "
                           f"needs network access to github.com for this one file, or a copy "
                           f"with sha256 {CHAT_SCRIPT_SHA256} at {path}") from e
    digest = hashlib.sha256(data).hexdigest()
    if digest != CHAT_SCRIPT_SHA256:
        raise RuntimeError(f"multipl_e: {CHAT_SCRIPT_URL} has sha256 {digest}, expected "
                           f"{CHAT_SCRIPT_SHA256}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".part")
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)
    return data.decode("utf-8")


def upstream() -> _Upstream:
    """The parsed upstream script (fetched once per process)."""
    global _UPSTREAM
    with _UPSTREAM_LOCK:
        if _UPSTREAM is None:
            _UPSTREAM = _Upstream(fetch_chat_script())
        return _UPSTREAM


def _chat_system() -> str:
    """The ChatAdapter system message for `ChainOfThought(Completion)`: DSPy 3.0.0b1's
    format_field_description + format_field_structure + format_task_description, with the
    signature's descriptions and instructions from the fetched script."""
    up = upstream()
    d = up.descriptions
    objective = ("\n" + " " * 8).join([""] + up.instructions.splitlines())
    return ("Your input fields are:\n"
            f"1. `lang` (str): {d['lang']}\n"
            f"2. `prefix` (str): {d['prefix']}\n"
            "Your output fields are:\n"
            "1. `reasoning` (str): \n"
            f"2. `completion` (str): {d['completion']}\n"
            "All interactions will be structured in the following way, with the appropriate "
            "values filled in.\n"
            "\n"
            "[[ ## lang ## ]]\n"
            "{lang}\n"
            "\n"
            "[[ ## prefix ## ]]\n"
            "{prefix}\n"
            "\n"
            "[[ ## reasoning ## ]]\n"
            "{reasoning}\n"
            "\n"
            "[[ ## completion ## ]]\n"
            "{completion}\n"
            "\n"
            "[[ ## completed ## ]]\n"
            f"In adhering to this structure, your objective is: {objective}")


def _image() -> str:
    return os.environ.get("GBENCH_MULTIPL_E_IMAGE", _IMAGE_DEFAULT)


def _langs() -> List[str]:
    raw = (os.environ.get("GBENCH_MULTIPL_E_LANGS") or "").replace(",", " ").split()
    return raw or list(_CANONICAL_LANGS)


#: Completions per problem in the upstream chat recipe.
_OFFICIAL_SAMPLES = 20


def _samples_per_problem(attempt_count: Optional[int] = None) -> int:
    """GBENCH_MULTIPL_E_SAMPLES if set; else the recipe's 20 (or more, if --attempt-count asks).
    A malformed value is an error, not 1."""
    raw = (os.environ.get("GBENCH_MULTIPL_E_SAMPLES") or "").strip()
    if not raw:
        return max(_OFFICIAL_SAMPLES, int(attempt_count or 1))
    try:
        n = int(raw)
    except ValueError:
        n = 0
    if n < 1:
        raise ValueError(f"GBENCH_MULTIPL_E_SAMPLES must be a positive integer, got {raw!r}")
    return n


def check_multipl_e_prerequisites() -> Tuple[bool, str]:
    """`datasets` + Docker daemon + the locally-built MultiPL-E evaluator image + the upstream
    chat script at the pinned commit."""
    try:
        import datasets  # noqa: F401
    except ImportError:
        return False, ("Python package 'datasets' is not installed (pip install -e \".[evals]\" "
                       "from the gbench checkout).")
    image = _image()
    build = (f"Build the evaluator LOCALLY (never pulled) with MultiPL-E's own "
             f"evaluation/Dockerfile at the pinned commit, downloaded and sha256-checked by "
             f"docker/multipl_e.Dockerfile:\n"
             f"  docker build -t gbench-multipl-e-src:3025a531 -f docker/multipl_e.Dockerfile docker\n"
             f"  docker run --rm gbench-multipl-e-src:3025a531 | docker build -t {image} -\n"
             f"See " + DOCS_URL)
    if not shutil.which("docker"):
        return False, "docker CLI not found. " + build
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        return False, "docker daemon not reachable. " + build
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode != 0:
        return False, f"image {image!r} not found. " + build
    try:
        upstream()
    except Exception as e:                                              # noqa: BLE001
        return False, str(e)
    return True, ""


def chat_messages(lang: str, prefix: str) -> List[Dict[str, str]]:
    """The ChatAdapter messages for one problem (ChatAdapter.format_user_message_content)."""
    user = (f"[[ ## lang ## ]]\n{lang}\n\n[[ ## prefix ## ]]\n{prefix}\n\n"
            f"{_CHAT_OUTPUT_REQUIREMENTS}").strip()
    return [{"role": "system", "content": _chat_system()}, {"role": "user", "content": user}]


def parse_chat_fields(completion: str) -> Optional[Dict[str, str]]:
    """Port of dspy 3.0.0b1 ChatAdapter.parse for this signature (both fields are `str`).

    Returns None when either output field is missing, where DSPy raises AdapterParseError. On the
    script's async path nothing retries it (see the module docstring), so upstream records the
    completion as ""."""
    sections: List[Tuple[Optional[str], List[str]]] = [(None, [])]
    for line in completion.splitlines():
        match = _FIELD_HEADER.match(line.strip())
        if match:
            header = match.group(1)
            remaining_content = line[match.end():].strip()
            sections.append((header, [remaining_content] if remaining_content else []))
        else:
            sections[-1][1].append(line)
    fields: Dict[str, str] = {}
    for k, v in ((k, "\n".join(v).strip()) for k, v in sections):
        if k not in fields and k in _OUTPUT_FIELDS:
            fields[k] = v
    return fields if set(fields) == set(_OUTPUT_FIELDS) else None


def stop_at_stop_token(decoded_string: str, stop_tokens: List[str]) -> str:
    """chat_completions.py's stop_at_stop_token (the fetched upstream function)."""
    return upstream().stop_at_stop_token(decoded_string, stop_tokens)


def postprocess_completion(stop_tokens: List[str], prompt: str, completion: str,
                           lang: str = "", name: str = "") -> str:
    """chat_completions.py's postprocess_completion (the fetched upstream function): drop a
    surrounding Markdown fence, keep the prompt-length head, and cut the rest at the first stop
    token. `lang` and `name` only feed its log line."""
    return upstream().postprocess_completion(lang, Path(f"{name or 'problem'}.json.gz"),
                                             stop_tokens, prompt, completion)


def _load_multipl_e_samples(
    langs: Optional[List[str]] = None,
    benchmark: str = "humaneval",
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load MultiPL-E (HumanEval family) for ALL requested languages; raises on failure."""
    langs = langs or _langs()
    from datasets import load_dataset
    samples = []
    for lang in langs:
        cfg = f"{benchmark}-{lang}"
        try:
            ds = load_dataset("nuprl/MultiPL-E", cfg, split="test", revision=_DATASET_REVISION)
        except Exception as e:
            raise RuntimeError(f"Could not load nuprl/MultiPL-E '{cfg}': {e}") from e
        for item in ds:
            name = item.get("name")
            prompt = item.get("prompt")
            tests = item.get("tests")
            if not name or not prompt or not tests:
                raise RuntimeError(
                    f"multipl_e: unexpected schema for {cfg} (name/prompt/tests); "
                    "refusing to fabricate sample data")
            gold = json.dumps({
                "name": name, "language": lang, "prompt": prompt,
                "tests": tests, "stop_tokens": item.get("stop_tokens") or [],
            })
            # max_tokens: OFFICIAL_SAMPLING's, resolved per mode by base's token-limit policy.
            # `problem_id` (language/name) is the sample's id across shards (base.sample_key):
            # a HumanEval problem name repeats in every language.
            samples.append((chat_messages(lang, prompt), gold,
                            {"category": lang, "name": name, "problem_id": f"{lang}/{name}"}))

    # --shard first, over the full ordered set (every language), then the --eval-limit draw
    # within the shard.
    samples, _ = apply_shard(samples, "multipl_e")
    if limit is not None and limit > 0:
        # Stratify by language: the loop concatenates one language at a time, so a head is ONE
        # language; `--eval-limit 20` must sample ACROSS languages, not measure just the first.
        samples = stratified_sample(
            samples, limit,
            lambda x: (x[2] or {}).get("category") if len(x) > 2 and isinstance(x[2], dict) else None,
            seed="multipl_e")
    drawn = languages_drawn(x[2] if len(x) > 2 else None for x in samples)
    logger.info("Loaded %d multipl_e samples from %d language(s): %s (requested: %d).",
                len(samples), len(drawn), drawn, len(langs))
    return samples


def languages_drawn(metas: Any) -> List[str]:
    """The sorted languages (`category`) present in samples' or traces' metadata dicts: what a
    shard or an --eval-limit subset actually covers, which can be fewer than the requested
    list."""
    return sorted({m.get("category") for m in metas
                   if isinstance(m, dict) and m.get("category")})


def _make_scorer(image: str, workers: int, metrics: Dict[str, Any]):
    """Delegate execution to the official MultiPL-E evaluator container (one input file per trace,
    1:1 result mapping so no completion-ordering assumptions), then set is_correct per trace."""
    async def _score(sample_traces: List[Dict[str, Any]]) -> None:
        import asyncio
        workdir = tempfile.mkdtemp(prefix="gbench_mpe_")
        os.chmod(workdir, 0o777)   # the evaluator may run as a non-root uid in the image
        by_uid: Dict[str, Dict[str, Any]] = {}
        parse_failures = 0
        empty_responses = 0
        no_response = 0
        try:
            for i, tr in enumerate(sample_traces):
                try:
                    g = json.loads(tr.get("gold_answer") or "{}")
                except Exception:
                    g = {}
                raw = tr.get("response_text")
                text = strip_thinking_tags(raw or "")
                fields = parse_chat_fields(text) if text else None
                if fields is None:
                    # Every case scores as an empty program, which still runs and fails its tests.
                    if raw is None:
                        # The request got no response from the served endpoint: base marks the
                        # trace a harness error and leaves it out of the score whatever this
                        # scorer makes of it (request_harness_errors).
                        no_response += 1
                        tr["scoring_note"] = "no response to the request"
                    elif not raw:
                        # DSPy tests the raw content (`if text:`): for an empty one it returns no
                        # fields, and upstream's script then stops (postprocess_completion(None));
                        # counted, and the run is not comparable. Whitespace alone is not empty
                        # there: ChatAdapter.parse rejects it (the branch below).
                        empty_responses += 1
                        tr["scoring_note"] = "empty response"
                    else:
                        # AdapterParseError, which upstream's do_task records as completion "".
                        parse_failures += 1
                        tr["scoring_note"] = ("no [[ ## completion ## ]] field in the response "
                                              "(completion \"\" as upstream records it)")
                    completion = ""
                else:
                    completion = fields["completion"]
                code = postprocess_completion(g.get("stop_tokens") or [], g.get("prompt", ""),
                                              completion, lang=g.get("language", ""),
                                              name=g.get("name", ""))
                tr["assembled_program"] = code
                uid = f"p{i:06d}"
                by_uid[uid] = tr
                # prompt="" as upstream writes it: the completion is the whole program (the model
                # was asked to repeat the prefix), so the container runs `code + "\n" + tests`.
                payload = {"name": uid, "language": g.get("language", ""), "prompt": "",
                           "completions": [code], "tests": g.get("tests", "")}
                with open(os.path.join(workdir, f"{uid}.json"), "w", encoding="utf-8") as f:
                    json.dump(payload, f)
                os.chmod(os.path.join(workdir, f"{uid}.json"), 0o644)

            cmd = ["docker", "run", "--rm", "--network", "none", "-v", f"{workdir}:/out:rw",
                   image, "--dir", "/out", "--output-dir", "/out", "--recursive",
                   "--max-workers", str(max(1, workers))]
            proc = await asyncio.to_thread(
                lambda: subprocess.run(cmd, capture_output=True, text=True))

            parsed = 0
            # The completions the container returned a result for. base marks every trace
            # is_correct=False / PENDING_JUDGE before this scorer runs, so "no result" is read
            # from this set, never from the trace's fields.
            with_result: set = set()
            # The evaluator writes `<name>.results.json` (uncompressed) or `.results.json.gz`
            # depending on version - read both (verified against the built image: it emits plain
            # `.results.json`).
            result_files = (glob.glob(os.path.join(workdir, "*.results.json.gz"))
                            + glob.glob(os.path.join(workdir, "*.results.json")))
            for p in result_files:
                try:
                    opener = gzip.open if p.endswith(".gz") else open
                    with opener(p, "rt", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue
                tr = by_uid.get(data.get("name"))
                if tr is None or data.get("name") in with_result:
                    continue
                with_result.add(data.get("name"))
                res = data.get("results") or []
                tr["is_correct"] = bool(res) and res[0].get("status") == "OK" \
                    and res[0].get("exit_code") == 0
                tr["status"] = "OK"
                # Trajectory completeness: record what the sandbox actually did so a failure can be
                # read back (SyntaxError vs assertion vs timeout) instead of just "is_correct=False".
                if res:
                    r0 = res[0]
                    tr["execution"] = {
                        "status": r0.get("status"),
                        "exit_code": r0.get("exit_code"),
                        "stdout_tail": (r0.get("stdout") or "")[-1200:],
                        "stderr_tail": (r0.get("stderr") or "")[-1200:],
                    }
                parsed += 1

            if parsed == 0:
                # The container ran but scored nothing - a harness failure, not "0% solved".
                logger.error("multipl_e: container produced no parsed results. stderr tail: %s",
                             (proc.stderr or "")[-800:])
                metrics["multipl_e_report"] = {
                    "error": "the MultiPL-E evaluator container produced no parsed results",
                    "stderr_tail": (proc.stderr or "")[-400:]}
            # A completion the container returned no result for (while it scored others) is a
            # harness error, never the model's outcome: left out of the score (base:
            # completed_with_errors), counted and named. One whose request got no reply is base's
            # harness error already and keeps that kind.
            unscored = 0
            if parsed:
                for uid, tr in by_uid.items():
                    if uid not in with_result and not tr.get("harness_error"):
                        tr["is_correct"] = False
                        tr["status"] = "HARNESS_ERROR"
                        tr["scoring_excluded"] = True
                        tr["harness_error"] = {"kind": "no_harness_result",
                                               "detail": "the MultiPL-E evaluator returned no "
                                                         f"result for {uid}"}
                        tr["scoring_note"] = "no container result"
                        unscored += 1
            else:
                # The container scored nothing: every completion is a harness error, and the run
                # withholds its score (base: accuracy null) rather than reporting 0.
                for uid, tr in by_uid.items():
                    tr["is_correct"] = False
                    tr["status"] = "HARNESS_ERROR"
                    tr["scoring_excluded"] = True
                    if not tr.get("harness_error"):
                        tr["harness_error"] = {"kind": "no_results",
                                               "detail": "the MultiPL-E evaluator container "
                                                         "produced no parsed results"}
                    tr["scoring_note"] = "no container result"
            for tr in by_uid.values():
                # a no-reply trace the container did not score (base keeps it a harness error)
                tr.setdefault("is_correct", False)
            if unscored:
                metrics["multipl_e_unscored"] = unscored
            metrics["chat_adapter_parse_failures"] = parse_failures
            metrics["chat_adapter_empty_responses"] = empty_responses
            metrics["multipl_e_no_response"] = no_response
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
    return _score


def per_problem_pass_at_1(traces: List[Dict[str, Any]]
                          ) -> Tuple[Optional[float], Dict[str, float]]:
    """pass@1 as upstream's pass_k.py computes it: for each problem (one results file) the
    unbiased estimator with k=1, c/n over its completions, then the mean over problems; overall
    and per language (percent). Over the scored completions only (a harness error is left out),
    so a problem with no scored completion is left out too. (None, {}) when nothing is scored."""
    by_problem: Dict[Any, List[bool]] = {}
    lang_of: Dict[Any, Any] = {}
    for t in traces:
        if not is_scored(t):
            continue
        # The problem's id (base's sample_key, the same for its completions and on every shard);
        # a trace without one is keyed by its source sample's index in this run.
        key = t.get("sample_key") or t.get("source_sample_idx", t.get("sample_idx"))
        by_problem.setdefault(key, []).append(bool(t.get("is_correct")))
        lang_of[key] = t.get("category")
    if not by_problem:
        return None, {}
    shares = {k: sum(v) / len(v) for k, v in by_problem.items()}
    per_lang: Dict[Any, List[float]] = {}
    for k, share in shares.items():
        per_lang.setdefault(lang_of[k], []).append(share)
    return (100.0 * sum(shares.values()) / len(shares),
            {lang: 100.0 * sum(v) / len(v) for lang, v in per_lang.items()})


def apply_per_problem_pass_at_1(result: Dict[str, Any]) -> None:
    """Set `accuracy`, each language's `category_accuracy` accuracy and the attempts' avg@k to
    upstream's per-problem pass@1 (per_problem_pass_at_1). base's figures pool every completion
    (correct / total), which equals it only when every problem has the same number of scored
    completions; a harness error makes them differ. The counts stay as base reports them."""
    overall, per_lang = per_problem_pass_at_1(result.get("sample_traces") or [])
    if overall is None:
        return
    result["accuracy"] = round(overall, 2)
    for lang, acc in per_lang.items():
        entry = (result.get("category_accuracy") or {}).get(lang)
        if isinstance(entry, dict):
            entry["accuracy"] = round(acc, 2)
    if isinstance(result.get("attempts"), dict) and "avg_at_k" in result["attempts"]:
        result["attempts"]["avg_at_k"] = round(overall, 2)


def merge_shards(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The merged result of a sharded run (`gbench --merge`): the union of the shards'
    completions (the generic merge: traces de-duplicated by sample_key, counts re-counted), with
    the headline recomputed as upstream's per-problem pass@1 over that union
    (apply_per_problem_pass_at_1), which a pooled correct / scored count is not once a harness
    error leaves a problem with fewer completions."""
    from ...merge import generic_merge
    merged = generic_merge("multipl_e", [dict(r, accuracy=None) for r in results])
    for key in ("multipl_e_unscored", "chat_adapter_parse_failures",
                "chat_adapter_empty_responses", "multipl_e_no_response"):
        if any(isinstance(r.get(key), int) for r in results):
            merged[key] = sum(int(r.get(key) or 0) for r in results)
    # Every shard's languages, not shard 1's alone.
    merged["languages_evaluated"] = sorted(
        {lang for r in results for lang in (r.get("languages_evaluated") or [])}
        | set(languages_drawn(merged.get("sample_traces") or [])))
    apply_per_problem_pass_at_1(merged)
    return merged


def run_multipl_e(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run canonical MultiPL-E execution pass@1 via the local evaluator container."""
    ok, reason = check_multipl_e_prerequisites()
    if not ok:
        raise infra_required("multipl_e", reason, DOCS_URL)
    langs = _langs()
    limit = kwargs.get("limit")
    n = _samples_per_problem(kwargs.get("attempt_count"))
    samples = _load_multipl_e_samples(langs=langs, limit=limit)
    # --eval-limit N is N problems for the whole suite, spread over the languages (the loader's
    # stratified draw), never N per language; a language that gets 0 does not run.
    per_lang = {lang: 0 for lang in langs}
    for x in samples:
        per_lang[(x[2] or {}).get("category")] = per_lang.get((x[2] or {}).get("category"), 0) + 1
    limit_plan = eval_limit_plan("multipl_e", limit, len(samples), per_lang,
                                 setting_noun="language", attempts=n)
    # max_tokens: 2048 exactly without --thinking; the run's budget with it (base resolves the
    # same plan for every request).
    plan = resolve_token_budget("multipl_e", OFFICIAL_SAMPLING, thinking=enable_thinking)
    metrics: Dict[str, Any] = {}
    result = run_eval_suite(
        eval_name="multipl_e",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=presharded_for_base(samples),  # this shard's problems, kept as is
        async_eval_fn=_make_scorer(_image(), concurrency, metrics),
        declared_scoring_mode="execution",  # sandboxed test execution, not an LLM judge
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=limit,
        max_output_tokens=kwargs.get("max_output_tokens", 2048),
        attempt_count=n,          # n completions/problem (pass@1: apply_per_problem_pass_at_1)
        supports_attempts=True,   # scoring is per-completion, so @k is real
        official_sampling=OFFICIAL_SAMPLING,
    )
    result.update(metrics)
    result["eval_limit_plan"] = limit_plan
    if isinstance(metrics.get("multipl_e_report"), dict) and metrics["multipl_e_report"].get("error"):
        result["status"] = "error"
        result["error"] = metrics["multipl_e_report"]["error"]
    apply_per_problem_pass_at_1(result)
    # max_output_tokens_note: base's, which states the mode's max_tokens choice.
    result["languages_requested"] = list(langs)
    # The languages the run covered (a shard or an --eval-limit subset can miss some).
    result["languages_evaluated"] = languages_drawn(result.get("sample_traces") or [])
    result["samples_per_problem"] = n
    result["dataset_revision"] = f"nuprl/MultiPL-E@{_DATASET_REVISION}"
    result["chat_script"] = {"url": CHAT_SCRIPT_URL, "sha256": CHAT_SCRIPT_SHA256,
                             "fetched_at_run_time": True}
    result["metric"] = ("pass@1 (per problem c/n over its scored completions, then the mean over "
                        "problems, as pass_k.py) with MultiPL-E's chat protocol "
                        "(chat_completions.py @3025a531: DSPy ChainOfThought + ChatAdapter), "
                        "executed by the official evaluator image")
    # The upstream chat recipe is 20 completions/problem at temperature 0.2 and max_tokens 2048 over
    # the FULL language set, with no other sampling field. Anything short (a --eval-limit subset, a
    # language subset, fewer completions, other sampling) is a valid pass@1 but not that number.
    temp = result.get("temperature")
    noncanon: List[str] = []
    if result.get("status") == "error":
        noncanon.append(f"harness error: {result.get('error') or 'unknown'}")
    if metrics.get("multipl_e_unscored"):
        noncanon.append(f"{metrics['multipl_e_unscored']} completion(s) got no evaluator result "
                        "(harness errors, left out of the score; multipl_e_unscored)")
    if limit:
        noncanon.append(f"subset run (--eval-limit {limit})")
    if set(langs) != set(_CANONICAL_LANGS):
        noncanon.append("language subset")
    if n < _OFFICIAL_SAMPLES:
        noncanon.append(f"{n} completion(s)/problem (the upstream chat recipe uses "
                        f"{_OFFICIAL_SAMPLES})")
    if temp is None or abs(float(temp) - float(OFFICIAL_SAMPLING.temperature)) > 1e-9:
        noncanon.append(f"temperature={temp} ({result.get('temperature_source')}; the upstream "
                        f"chat recipe uses {OFFICIAL_SAMPLING.temperature})")
    # dspy.LM sends temperature and max_tokens only: any other sampling field is not upstream's.
    sent = {**(kwargs.get("extra_payload") or {}), **(result.get("decoding_penalties") or {})}
    extra = sorted(k for k in sent if k in OFFICIAL_SAMPLING_PARAM_KEYS)
    if extra:
        noncanon.append("sampling " + ", ".join(f"{k}={sent[k]!r}" for k in extra)
                        + " (the upstream chat script sends no such field)")
    budgets = result.get("max_tokens_sent") or []
    if plan.policy == "official" and list(budgets) != [UPSTREAM_MAX_TOKENS]:
        noncanon.append(f"max_tokens={budgets or 'not sent'} (the upstream chat script sends "
                        f"{UPSTREAM_MAX_TOKENS}; a request whose prompt left less room in the "
                        "context window was sent less)")
    # A reply stopped at a limit other than the chat script's 2048 (with --thinking, this run's
    # budget: the 2048 cap is not applied to reasoning; without it, a max_tokens the context
    # window lowered) measures that limit, not the model.
    thinking_run = plan.policy == "reasoning_run_budget"
    all_cut, empty_cut = budget_cutoffs(result)
    cut = empty_cut if thinking_run else 0
    result["replies_cut_off_by_budget"] = all_cut
    result["replies_cut_off_in_reasoning"] = cut
    if cut:
        noncanon.append(f"{cut} repl(ies) reached this run's max_tokens {plan.max_tokens} "
                        "inside the reasoning channel with no answer (the chat script's "
                        f"{UPSTREAM_MAX_TOKENS} was defined for non-reasoning generation and "
                        "is not applied with --thinking; raise --max-output-tokens)")
    if all_cut - cut:
        noncanon.append(f"{all_cut - cut} repl(ies) stopped at a max_tokens other than the chat "
                        f"script's {UPSTREAM_MAX_TOKENS} before finishing "
                        + ("(this run's budget; raise --max-output-tokens)" if thinking_run
                           else "(the context window left less room)"))
    if metrics.get("chat_adapter_empty_responses"):
        noncanon.append(
            f"{metrics['chat_adapter_empty_responses']} empty response(s) scored as empty "
            "programs; upstream's script stops on an empty reply (DSPy returns no fields and "
            "postprocess_completion fails on None), so it has no score for such a run")
    if metrics.get("multipl_e_no_response"):
        noncanon.append(f"{metrics['multipl_e_no_response']} request(s) got no response from the "
                        "served endpoint (harness errors, left out of the score; "
                        "request_harness_errors)")
    result["leaderboard_comparable"] = not noncanon
    if noncanon:
        result["leaderboard_comparable_reason"] = "; ".join(noncanon)
    # One shard of a --shard I/N run is a partial set: the shard reason (finalize_shard_fields).
    return finalize_shard_fields(result)
