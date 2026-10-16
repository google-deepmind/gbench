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

# Evaluation Suite: livebench
# Description: LiveBench (Abacus/NYU monthly-refreshed, contamination-limited; 6 core categories)

"""gbench native built-in runner for livebench (General Knowledge & Reasoning).

Canonical LiveBench (White et al.; livebench.ai; github.com/LiveBench/LiveBench) scores 6
core categories - coding, data_analysis, instruction_following, language, math, reasoning -
with its OWN ground-truth judges (no LLM judge), including real code EXECUTION for the coding
category. Its package + coding-execution deps (tensorflow/numba/opencv/...) would churn a
torch/vLLM serving venv, so gbench runs LiveBench's pipeline inside an isolated Docker image
(LIVEBENCH_IMAGE_DEFAULT, `gbench-livebench:<first 8 characters of LIVEBENCH_COMMIT>`, built
locally from docker/livebench.Dockerfile with a LiveBench checkout at LIVEBENCH_COMMIT as the
build context; BUILD_COMMAND): it invokes LiveBench's `run_livebench.py` (inference via
--api-base against the gbench-served model, then grading) and reads the resulting
`all_groups.csv` (per-category + `average`).

The `agentic_coding` 7th category is excluded (LiveBench's default benchmark set already
excludes it; it needs a separate ~150GB Multi-SWE-Bench container harness).

The release is pinned: LIVEBENCH_RELEASE (2024-11-25), passed to both LiveBench scripts as
`--livebench-release-option`; GBENCH_LIVEBENCH_RELEASE overrides it (`latest` passes none, so
LiveBench takes its own default, its newest release, and the run is not comparable). LiveBench's
README (at LIVEBENCH_COMMIT) says not all questions of its current release are public on Hugging
Face and tells users to pass 2024-11-25, "the most recent public questions", to evaluate all
categories; it is the newest release at which all 6 core categories have active public
questions (1000, the release size LiveBench's changelog keeps fixed).

This suite hard-errors (never silently skips) if Docker or the image is absent. Because it
shells out to LiveBench's own scorers, it does NOT use run_eval_suite.

Question selection (`--shard I/N`, `--eval-limit`, and a GBENCH_LIVEBENCH_CATEGORIES subset that
names a multi-word category): gbench lists the release's questions in the image with LiveBench's
own loader (livebench.common get_categories_tasks + load_questions, the release filter
gen_api_answer.py applies; list_questions, on the run's bridge network since it reads Hugging
Face), takes the categories asked for, orders their question ids, takes the shard first
(base.apply_shard_indexed, round-robin over the sorted ids), then draws `--eval-limit` ids from it
stratified by category with the seed "livebench" (sampling.stratified_sample), and passes the ids
to both LiveBench scripts with run_livebench.py's `--question-id` on the whole-suite bench name
(`live_bench`), which also avoids LiveBench's per-category loader that truncates multi-word
category names (data_analysis -> livebench/data). The result records `selected_question_ids`, the
shard identity (base.shard_identity) and a partial shard's comparability reason; an empty shard
reports `accuracy` null with `accuracy_withheld_reason`. GBENCH_LIVEBENCH_CATEGORIES accepts only
the 6 core category names (anything else is a ValueError before any container runs); a subset of
single-word categories with no shard or limit keeps LiveBench's per-category bench names. Question
selection in GBENCH_LIVEBENCH_EXTRA_ARGS (`--question-id`, `--question-begin`, `--question-end`) is
refused while gbench selects the questions, and otherwise makes the run not comparable.

Per-question results: the container also copies LiveBench's judgment rows
(data/live_bench/<category>/<task>/model_judgment/ground_truth_judgment.jsonl); the result carries
one trace per judged question (`sample_traces`: `question_id`, `category`, `task`, `score` in
[0, 1], `sample_key`) and `questions_judged`. merge_shards (gbench --merge) recomputes LiveBench's
aggregation over the union of the shards' rows as show_livebench_result.py does (per task the mean
score x 100, per category the mean of its tasks, the average the mean of the categories). LiveBench
answers each question once: `--attempt-count` above 1 is recorded as not applied
(`attempts.not_applied`).

Networks: the image checks (the patch check and the model-config lookup) run with
`--network none`. The run itself (inference, then grading, which executes the model's code for
the coding category) needs the endpoint and Hugging Face, so it runs on docker's bridge network,
never the host network: the host's DOCKER-USER rule keeps the cloud metadata server out of reach
of a bridge network only. Before it starts, tau_common.require_endpoint_network checks that the
metadata server is blocked there (container_guard), that the probe container resolves
huggingface.co (CONTAINER_HOSTS) and the endpoint's host when it is a name, and that an endpoint
on this host answers at host.docker.internal, the `--api-base` LiveBench then gets; any of them
failing is a hard error (infra_required). A run that scores nothing and shows a failed host-name
lookup in its output, or in which every request failed at the endpoint, reports no score
(infra_required). The model's code in grading keeps the container's general internet access, as
upstream runs it; only the metadata server is blocked.

The image carries one local change to the pinned checkout (LIVEBENCH_PATCH), applied by
docker/livebench.Dockerfile at build time and recorded on every result (`livebench_patch`): at
LIVEBENCH_COMMIT, livebench/gen_ground_truth_judgment.py line 480 builds the old-format
instruction_following question list with list(set(...)) over question dicts, which raises
TypeError: unhashable type: 'dict', so LiveBench never grades that category; the patch
de-duplicates the questions by question_id instead. Before each run _check_prereqs checks the
patched file's sha256 in the image (infra_required when it differs). The same grading call labels
every instruction_following judgment row with one task name, so the category score LiveBench
writes is the pooled mean of those rows; it equals the mean of the task means when the tasks
contribute the same number of rows (50 each at LIVEBENCH_RELEASE), and a run where they do not,
or where instruction_following was scored but none of its answer rows were found to count, is
reported as not comparable.

Generation settings reach LiveBench the way its own model configs carry them. LiveBench keeps a
config per model (livebench/model/model_configs/*.yml; run_livebench.py --model looks it up by
display name or alias with get_model_config). Before the run, gbench asks the image's LiveBench
for the served model's config (the served model id, then the run's --tokenizer, each as given and
as its last path component) and writes a one-off config (display name `gbench-served-model`,
provider `local`, API name = the served model id) that the container copies into LiveBench's
`model_configs/` directory:

* served model with a LiveBench config (for example `gemma-4-31b-it`): the request kwargs that
  config sends on its leaderboard provider (api_kwargs `default`, updated with the entry of its
  `default_provider`, as gen_api_answer.py setup_model merges them): temperature, top_p and the
  other sampling fields, the token cap, and its `prompt_prefix` / `preserve_reasoning`. Its
  reasoning toggle for that provider (`extra_body.reasoning.enabled`) becomes the local
  endpoint's `chat_template_kwargs.enable_thinking`, which must match --thinking. A field with no
  local counterpart is not applied and makes the run non-comparable.
* served model without one: LiveBench's defaults, `--max-tokens` (the temperature is gbench's,
  below).

The reasoning toggle is `chat_template_kwargs.enable_thinking` in the one-off config's
`api_kwargs`. max_tokens follows base's token-limit policy (resolve_token_budget):

* no-think: the official cap exactly, i.e. the model config's cap, else gen_api_answer.py's
  `--max-tokens` default 4096; the run's --max-output-tokens is not applied;
* --thinking: a model config's cap is that model's own setting and is sent (or the run's
  --max-output-tokens when larger; a thinking run is never capped below it); the 4096 default was
  written for non-reasoning generation and is not applied, so each request gets the run's budget
  (--max-output-tokens, else base's thinking floor).

LiveBench sends that one max_tokens with every request, and vLLM rejects a request whose prompt +
max_tokens exceeds the served context window. So, whatever its source, a max_tokens that would
leave the longest prompt no room (_PROMPT_RESERVE_TOKENS) is lowered to fit the window, as base's
request path lowers each request of the other suites; the result says so
(`max_output_tokens_source` ends in `fit-to-window`, `max_output_tokens_note`,
`official_sampling.max_tokens_sent`), and an official cap lowered this way (a model config's
65536 on a 32768-token window, say) makes the run not comparable, naming the window it needs.

A reply cut off by a max_tokens other than the official cap (the run's budget, one lowered to fit
the window, or GBENCH_LIVEBENCH_EXTRA_ARGS's) measures that budget, not the model: gbench reads
LiveBench's answer rows (copied out of the container) and counts them (read_answer_rows:
`replies_cut_off_by_budget`, `replies_cut_off_in_reasoning`), with and without --thinking, both
the replies cut off inside the reasoning channel (LiveBench's `token_exhaustion`) and those cut
off after the answer began. LiveBench's total_output_tokens adds the server's reasoning_tokens to
its completion_tokens, which in vLLM already include the reasoning, so a total above max_tokens
means the reply's completion tokens are (total + the answer's tokens) / 2, the answer counted
with the run's tokenizer, --tokenizer or else the served model's id (`replies_cut_off_note`). The
rule takes the server's reasoning_tokens from its usage, as vLLM reports them whenever it runs a
reasoning parser; for a server that streams reasoning without them, litellm counts the reasoning
with its own tokenizer. Any cut-off makes the run not comparable, as do answer rows it cannot
find and rows it cannot check (`replies_cut_off_uncounted`: no output token count, several turns
at or above max_tokens, or one turn above it when that tokenizer cannot be loaded).
LiveBench scores every `api_error` row 0, and gbench splits them by cause. A reply the model
finished with no answer text (after its reasoning, or none at all) comes back with a
finish_reason other than length, and LiveBench's litellm path raises "No message returned from
litellm (finish_reason=...)" for it: that is the model's outcome, counted in `empty_replies` and
scored 0 as LiveBench scores it, with no effect on the status or on comparability. Every other
`api_error` row is a request that failed at the endpoint (connection, HTTP error, timeout), a
harness failure: counted in `failed_requests` (per category in `failed_requests_by_category`),
the status is `completed_with_errors` and the run is not comparable. LiveBench scored those rows
0 inside its averages, so the headline is withheld (`accuracy` None, `accuracy_withheld_reason`)
and so is the score of every category with a failed request (`dimension_scores_withheld`);
`dimension_scores` keeps the categories whose every request was answered. gbench does not re-ask
them: LiveBench's own `--retry-failures` re-asks every `$ERROR$` answer, the model's empty replies
included. Requests the served vLLM server aborted or failed during the run
(`server_aborted_requests`) reach LiveBench as empty or cut-short replies in unknown categories,
so they withhold the headline and every category score the same way.

When max_tokens was lowered to fit the window or replaced by GBENCH_LIVEBENCH_EXTRA_ARGS,
`max_output_tokens_note` says what the token-limit policy gives and then the max_tokens each
request was sent; it never names a value as sent that was not.

The budget, the model config and the temperature are recorded on the result
(`max_output_tokens`, `max_output_tokens_note`, `livebench_model_config`, `official_sampling`).

Sampling: LiveBench's own (OFFICIAL_SAMPLING), or the served model's config, except the
temperature, which follows gbench's rule for every suite (base.resolve_temperature): 1.0 with
--thinking, 0.0 without; `--temperature`, or `GBENCH_LIVEBENCH_TEMPERATURE` for this suite alone
(it takes precedence), overrides it in either mode. get_answer uses temperature 0, or a question's
`required_temperature`, unless `--force-temperature` is given, and a model config's temperature
replaces it on every request (chat_completion_* apply the config's api_kwargs after get_answer's
temperature). So gbench always writes the run's temperature as the one-off config's temperature,
in both modes (never `--force-temperature`, on which LiveBench stops at any question that carries
a required temperature). The result records LiveBench's own value under `official_sampling`. A
temperature other than get_answer's 0 (no model config; the LiveBench question sets gbench has
read carry no `required_temperature`, which the config's temperature would also replace), or
other than the model config's, makes the run not leaderboard-comparable, and the reason names it.
LiveBench uses no LLM judge, so gbench's judge temperature (pinned at 0.0) plays no part. A
gbench-livebench image whose LiveBench checkout is not the pinned commit (LIVEBENCH_COMMIT), or
whose commit cannot be read (no .git in the build context), is not comparable.
"""

import csv
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from .base import (OFFICIAL_SAMPLING_PARAM_KEYS, RAISED_REASONING_CAP_NOTE, REASONING_CAP_NOTE,
                   SHARD_REASON, OfficialSampling, add_comparability_reason, apply_shard_indexed,
                   current_shard, gbench_cache_dir, get_run_knob, resolve_official_sampling,
                   resolve_token_budget, shard_identity)
from .limit_plan import eval_limit_plan
from .sampling import stratified_sample
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

PILLAR = "General Knowledge & Reasoning"
DOCS_URL = "docs/evals/livebench.md"
#: The host names the run container looks up itself besides the model endpoint's: the Hugging
#: Face Hub, where `--question-source huggingface` loads the questions (both gen_api_answer.py and
#: gen_ground_truth_judgment.py load them). The grading data LiveBench would otherwise download
#: (NLTK, spaCy) is baked into the image. tau_common.require_endpoint_network has the probe
#: container resolve them before the run starts.
CONTAINER_HOSTS = ("huggingface.co",)

# The 6 core categories (LiveBench run_livebench.py DEFAULT_BENCHMARKS; agentic_coding excluded).
_CATEGORIES = ["coding", "data_analysis", "instruction_following", "language", "math", "reasoning"]
#: Printed by the question-list probe below; the JSON after it is [[category, task, id], ...].
_QUESTIONS_MARKER = "GBENCH_LIVEBENCH_QUESTIONS "
#: Run in the image: the release's questions, with LiveBench's own loader and the release filter
#: gen_api_answer.py applies (the release set is every release up to the option; no option is
#: LiveBench's default, max(LIVE_BENCH_RELEASES)).
_QUESTIONS_SRC = """import json, sys
from livebench.common import LIVE_BENCH_RELEASES, get_categories_tasks, load_questions
release = (sys.argv[1] if len(sys.argv) > 1 else "") or max(LIVE_BENCH_RELEASES)
release_set = set(r for r in LIVE_BENCH_RELEASES if r <= release)
categories, tasks = get_categories_tasks("live_bench")
out = []
for cat, task_names in tasks.items():
    for task in task_names:
        for q in load_questions(categories[cat], release_set, release, task, None):
            out.append([cat, task, str(q["question_id"])])
print("GBENCH_LIVEBENCH_QUESTIONS " + json.dumps(out), flush=True)
"""
#: Where the container copies LiveBench's judgment rows, under the run's output directory.
_JUDGMENT_DIR = "model_judgment"

#: The LiveBench commit the suite and docker/livebench.Dockerfile are reconciled against.
LIVEBENCH_COMMIT = "0baa4aa7a78dc0276c0ccf28779dba1113580e18"

#: The local image gbench runs by default: docker/livebench.Dockerfile built with a LiveBench
#: checkout at LIVEBENCH_COMMIT as the context, tagged with that commit (it is never pulled; an
#: untagged `gbench-livebench`, i.e. :latest, is not used). GBENCH_LIVEBENCH_IMAGE overrides it.
LIVEBENCH_IMAGE_DEFAULT = f"gbench-livebench:{LIVEBENCH_COMMIT[:8]}"
#: How to build it, run from the gbench repository root (the context is the LiveBench checkout at
#: LIVEBENCH_COMMIT with its LFS *.json pulled and its .git kept, not gbench's docker/ directory).
BUILD_COMMAND = (f"docker build -t {LIVEBENCH_IMAGE_DEFAULT} -f docker/livebench.Dockerfile "
                 "/path/to/LiveBench")

#: gbench's one local change to the pinned LiveBench checkout. docker/livebench.Dockerfile checks
#: the file's sha256, replaces the line and checks the patched sha256 (the build fails otherwise);
#: _check_prereqs checks the patched sha256 in the image before each run; every result records it.
#: Verified 2026-09-26 in gbench-livebench:0baa4aa7 built from a GitHub clone at LIVEBENCH_COMMIT,
#: with a model-free grading run of stub answers to all 200 instruction_following questions of
#: release 2024-11-25: gen_ground_truth_judgment.py exits 0 and writes 400 judgments (the unpatched
#: image raises the TypeError), and all_groups.csv's instruction_following equals the mean of the
#: four per-task scores LiveBench's own instruction_following scorer gives task by task.
LIVEBENCH_PATCH = {
    "file": "livebench/gen_ground_truth_judgment.py",
    "line": 480,
    "upstream": ("if_questions = list(set([m.question for m in "
                 "old_instruction_following_matches]))"),
    "patched": ("if_questions = list({m.question['question_id']: m.question for m in "
                "old_instruction_following_matches}.values())"),
    "reason": ("upstream builds the old-format instruction_following question list with "
               "list(set(...)) over question dicts, which raises TypeError: unhashable type: "
               "'dict' (introduced by LiveBench commit 366e385, 2025-12-12), so that category "
               "is never graded; the patch de-duplicates the questions by question_id"),
    "sha256_upstream": "dd6bdce67dafd270030980fc864a37940f09a8aa83ce6f94d8657052afbdbd37",
    "sha256_patched": "19ed4fef82fd839acc0f558380adb4631fdbc5694948575e0c0d319f58ec3a85",
    "applied_by": "docker/livebench.Dockerfile, RUN step after COPY (build time)",
    "side_effect": ("the same pooled grading call labels every instruction_following judgment "
                    "row with one task name (task_name = the first question's task), so "
                    "all_tasks.csv shows one instruction_following column and the category "
                    "score is the pooled mean of the rows, equal to the mean of the task means "
                    "when the tasks contribute the same number of rows"),
}
#: Printed by the in-image patch check; the sha256 after it is the image's file.
_PATCH_MARKER = "GBENCH_LIVEBENCH_PATCH_SHA256 "
_PATCH_CHECK_SRC = ("import hashlib, os, livebench; "
                    "p = os.path.join(os.path.dirname(os.path.dirname(livebench.__file__)), "
                    f"{LIVEBENCH_PATCH['file']!r}); "
                    f"print({_PATCH_MARKER!r} + hashlib.sha256(open(p, 'rb').read()).hexdigest())")

#: The LiveBench release run by default (`--livebench-release-option`). At LIVEBENCH_COMMIT,
#: livebench/common.py LIVE_BENCH_RELEASES ends at 2026-06-25, but README.md says: "The current
#: LiveBench release is 2025-04-25; however, not all questions for this release are public on
#: Huggingface. In order to evaluate all categories, you will need to pass
#: `--livebench-release-option 2024-11-25` to all scripts to use the most recent public
#: questions." The public datasets (huggingface.co/datasets/livebench/<category>, last modified
#: 2025-04-07) hold questions released up to 2024-11-25; with load_questions' filter (release
#: date <= the release, removal date empty or after it) 2024-11-25 has coding 128,
#: data_analysis 150, instruction_following 200, language 140, math 232 and reasoning 150 active
#: questions (1000), while every later release has no active coding question (all 128 carry
#: removal date 2025-04-02) and 682 in all. GBENCH_LIVEBENCH_RELEASE overrides it.
LIVEBENCH_RELEASE = "2024-11-25"

#: gen_api_answer.py's `--max-tokens` default (argparse default=4096): the cap of every request of a
#: model whose LiveBench config sets none.
LIVEBENCH_MAX_TOKENS = 4096

#: LiveBench's sampling: livebench/gen_api_answer.py get_answer sets the temperature itself (0, or
#: the question's `required_temperature`) unless `--force-temperature` is passed; max_tokens is
#: `--max-tokens` (default 4096).
OFFICIAL_SAMPLING = OfficialSampling(
    reference=(f"github.com/LiveBench/LiveBench@{LIVEBENCH_COMMIT} "
               "livebench/gen_api_answer.py get_answer: temperature 0, or the question's "
               "required_temperature; no --force-temperature; --max-tokens default 4096"),
    send_temperature=False, max_tokens=LIVEBENCH_MAX_TOKENS)

#: The api_kwargs that carry a token cap (LiveBench's litellm path checks all three; any one of
#: them replaces `--max-tokens`).
_CAP_KEYS = ("max_tokens", "max_completion_tokens", "max_output_tokens")
#: api_kwargs a local OpenAI-compatible endpoint takes as they are, besides the sampling fields and
#: the token cap: LiveBench's transport settings and vLLM's reasoning_effort.
_PORTABLE_KEYS = ("stream", "timeout", "reasoning_effort")
#: Top-level config fields that shape the prompt or the conversation, kept in the one-off config.
_PROMPT_FIELDS = ("prompt_prefix", "preserve_reasoning")

#: Printed by the in-image lookup below; the JSON after it is the lookup's result.
_CONFIG_MARKER = "GBENCH_LIVEBENCH_MODEL_CONFIG "
#: Run in the image with LiveBench's own lookup (get_model_config), for each name in argv until
#: one is a config LiveBench ships; also reports the checkout's commit when git can read it.
_CONFIG_LOOKUP_SRC = """import dataclasses, json, os, subprocess, sys
import livebench
from livebench.model.api_model_config import get_model_config, load_all_configs
root = os.path.dirname(os.path.dirname(os.path.abspath(livebench.__file__)))
out = {"commit": None, "name": None, "config": None, "error": None}
try:
    out["commit"] = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True,
                                   text=True).stdout.strip() or None
except Exception:
    pass
configs = load_all_configs()
for name in sys.argv[1:]:
    try:
        cfg = get_model_config(name)
    except ValueError as e:
        out.update(name=name, error=str(e))
        break
    if configs.get(cfg.display_name) is cfg:
        out.update(name=name, config=dataclasses.asdict(cfg))
        break
print("GBENCH_LIVEBENCH_MODEL_CONFIG " + json.dumps(out, default=str), flush=True)
"""

#: The one-off LiveBench model config gbench writes per run (display name chosen so it cannot
#: collide with a config LiveBench ships; a duplicate name makes get_model_config raise).
_MODEL_CONFIG_NAME = "gbench-served-model"
_MODEL_CONFIG_FILE = "gbench_model_config.yml"

#: Room left for the prompt when max_tokens is fitted to the served window (fit_to_window). The
#: longest core-category prompt is ~8.1k characters (measured on the HF test splits, 2026-09-24),
#: well under 4k tokens even at 2 characters per token; the rest covers the chat template.
_PROMPT_RESERVE_TOKENS = 8192


def _image() -> str:
    return (os.environ.get("GBENCH_LIVEBENCH_IMAGE") or "").strip() or LIVEBENCH_IMAGE_DEFAULT


def _release() -> Tuple[str, str]:
    """(the release passed to LiveBench, where it came from).

    LIVEBENCH_RELEASE unless GBENCH_LIVEBENCH_RELEASE is set; `latest` there passes no
    `--livebench-release-option`, so LiveBench takes its own default (max(LIVE_BENCH_RELEASES)),
    and returns "" (an unpinned run)."""
    raw = (os.environ.get("GBENCH_LIVEBENCH_RELEASE") or "").strip()
    if not raw:
        return LIVEBENCH_RELEASE, "default"
    return ("" if raw.lower() == "latest" else raw), "env:GBENCH_LIVEBENCH_RELEASE"


def _categories() -> List[str]:
    """The 6 core categories, or a subset via GBENCH_LIVEBENCH_CATEGORIES (space/comma list,
    without repeats). A name that is not one of the 6 core categories is a ValueError."""
    raw = (os.environ.get("GBENCH_LIVEBENCH_CATEGORIES") or "").replace(",", " ").split()
    raw = list(dict.fromkeys(c.strip().lower() for c in raw if c.strip()))
    unknown = [c for c in raw if c not in _CATEGORIES]
    if unknown:
        raise ValueError(f"livebench: GBENCH_LIVEBENCH_CATEGORIES names {unknown}, which are not "
                         f"LiveBench core categories (choose from {_CATEGORIES}); see {DOCS_URL}")
    return raw or list(_CATEGORIES)


def list_questions(image: str, release: str) -> List[Tuple[str, str, str]]:
    """[(category, task, question_id)] of the release's questions, sorted, read in the image with
    LiveBench's own loader (_QUESTIONS_SRC) on the run's bridge network (it loads the questions
    from Hugging Face). No answer from the image is infra_required."""
    from .tau_common import ENDPOINT_NETWORK_ARGS
    cmd = ["docker", "run", "--rm", *ENDPOINT_NETWORK_ARGS, image, "python", "-c",
           _QUESTIONS_SRC, release or ""]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except (OSError, subprocess.SubprocessError) as e:
        raise infra_required("livebench", f"could not list the release's questions in image "
                             f"{image!r} ({type(e).__name__}: {e})", DOCS_URL) from e
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith(_QUESTIONS_MARKER):
            try:
                rows = json.loads(line[len(_QUESTIONS_MARKER):])
            except ValueError:
                break
            return sorted((str(c), str(t), str(q)) for c, t, q in rows)
    _raise_if_unresolved(proc)
    raise infra_required(
        "livebench", f"LiveBench's question list for release '{release or 'latest'}' could not be "
        f"read in image {image!r} (exit {proc.returncode}); it loads the questions from Hugging "
        "Face. Last output: " + (proc.stderr or proc.stdout or "")[-600:], DOCS_URL)


def select_questions(rows: List[Tuple[str, str, str]], categories: List[str],
                     limit: Optional[int]) -> Tuple[List[str], Optional[str], List[int], int]:
    """(the question ids to run, the shard label or None, their positions in the ordered full
    list, the size of that list): the ids of `categories` in `rows`, each once, sorted; the shard
    first (base.apply_shard_indexed), then `limit` ids drawn stratified by category with the seed
    "livebench" (sampling.stratified_sample). An id LiveBench uses in two tasks is one unit."""
    category_of: Dict[str, str] = {}
    for cat, _task, qid in rows:
        if cat in categories:
            category_of.setdefault(qid, cat)
    ids = sorted(category_of)
    selected, label, positions = apply_shard_indexed(ids, "livebench")
    if limit and int(limit) > 0 and len(selected) > int(limit):
        tagged = stratified_sample(list(zip(selected, positions)), int(limit),
                                   key_fn=lambda t: category_of[t[0]], seed="livebench")
        selected = [t[0] for t in tagged]
        positions = [t[1] for t in tagged]
    return selected, label, positions, len(ids)


#: The result's correct_answers_note.
_CORRECT_ANSWERS_NOTE = ("not applicable: LiveBench scores each question in [0, 1] and reports "
                         "per-task and per-category means; total_questions counts the questions "
                         "judged with a score (sample_traces holds each)")


def _output_dir() -> str:
    """A new directory for one run's LiveBench output, under GBENCH_CACHE_DIR/livebench/runs."""
    import time
    root = os.path.join(gbench_cache_dir(), "livebench", "runs")
    os.makedirs(root, exist_ok=True)
    # an absolute prefix places mkdtemp's directory under `root`
    return tempfile.mkdtemp(prefix=os.path.join(root, time.strftime("%Y%m%d-%H%M%S-")))


def _request_model_id(base_url: str, model_name: str) -> Tuple[str, str]:
    """(the model id LiveBench's requests carry, where it came from): `model_name`, the served id
    gbench passes (checked against --models by the eval runner), or GBENCH_LIVEBENCH_MODEL,
    checked against the endpoint's /v1/models first (an id it does not serve is
    infra_required, never sent)."""
    override = (os.environ.get("GBENCH_LIVEBENCH_MODEL") or "").strip()
    if not override or override == model_name:
        return model_name, "the served model id gbench passed"
    from ...utils.endpoint import EndpointError, resolve_served_model
    try:
        served = resolve_served_model(base_url, [override], override)
    except EndpointError as e:
        raise infra_required("livebench", f"GBENCH_LIVEBENCH_MODEL={override!r} is not a model "
                             f"the endpoint serves ({e}); unset it or name a served id",
                             DOCS_URL) from e
    return served.id, f"GBENCH_LIVEBENCH_MODEL ({served.source})"


def read_judgment_rows(outdir: str, display: str) -> List[Dict[str, Any]]:
    """LiveBench's judgment rows for the run's model in `outdir` (the container copies
    data/live_bench/<category>/<task>/model_judgment/ground_truth_judgment.jsonl there), as
    traces: question_id, category, task, score (0..1; -1 rows, which LiveBench leaves out, are
    kept with `scoring_excluded`), eval_status, sample_key; sorted by task and id."""
    import glob
    out: List[Dict[str, Any]] = []
    root = os.path.join(outdir, _JUDGMENT_DIR)
    rows: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(root, "**", "ground_truth_judgment.jsonl"),
                                 recursive=True)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    models = {str(r.get("model") or "").lower() for r in rows}
    # the run's container is fresh, so its judgment files hold one model: take it when LiveBench
    # wrote the display name in another form
    wanted = display.lower() if display.lower() in models or len(models) != 1 else models.pop()
    for row in rows:
        if str(row.get("model") or "").lower() != wanted:
            continue
        try:
            score = float(row.get("score"))
        except (TypeError, ValueError):
            score = None
        trace = {"question_id": str(row.get("question_id")),
                 "category": row.get("category"), "task": row.get("task"),
                 "score": score, "eval_status": row.get("eval_status") or None,
                 "sample_key": f"question:{row.get('task')}/{row.get('question_id')}"}
        if score is None or score == -1:
            trace["scoring_excluded"] = True
        out.append(trace)
    return sorted(out, key=lambda t: (str(t["task"]), t["question_id"]))


def livebench_scores(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """LiveBench's aggregation over judgment traces, as show_livebench_result.py computes it: per
    task the mean score x 100, per category the mean of its tasks' means, and the average of the
    category means (when more than one category), rounded to one decimal."""
    by_task: Dict[Tuple[str, str], List[float]] = {}
    for t in traces:
        if t.get("scoring_excluded") or t.get("score") is None:
            continue
        by_task.setdefault((str(t.get("category")), str(t.get("task"))), []).append(
            float(t["score"]) * 100.0)
    by_cat: Dict[str, List[float]] = {}
    for (cat, _task), vals in by_task.items():
        by_cat.setdefault(cat, []).append(sum(vals) / len(vals))
    cats = {c: sum(v) / len(v) for c, v in sorted(by_cat.items())}
    average = sum(cats.values()) / len(cats) if len(cats) > 1 else (
        next(iter(cats.values())) if cats else None)
    return {"average": round(average, 1) if average is not None else None,
            "categories": {c: round(v, 1) for c, v in cats.items()},
            "tasks": {f"{c}/{t}": round(sum(v) / len(v), 3) for (c, t), v in
                      sorted(by_task.items())}}


def _check_prereqs(image: str) -> None:
    build = (f"Build it from the gbench repository root, with a LiveBench checkout at the pinned "
             f"commit {LIVEBENCH_COMMIT[:8]} as the context (its LFS *.json pulled, its .git "
             "kept):\n"
             f"  git -C /path/to/LiveBench checkout {LIVEBENCH_COMMIT}\n"
             "  git -C /path/to/LiveBench lfs pull\n"
             f"  docker build -t {image} -f docker/livebench.Dockerfile /path/to/LiveBench")
    if not shutil.which("docker"):
        raise infra_required("livebench", "docker CLI not found. " + build, DOCS_URL)
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        raise infra_required("livebench", "docker daemon not reachable. " + build, DOCS_URL)
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode != 0:
        raise infra_required("livebench", f"image {image!r} not found. " + build, DOCS_URL)
    _check_patch(image, build)


def _check_patch(image: str, build: str) -> None:
    """infra_required unless the image's gen_ground_truth_judgment.py is the pinned file with
    LIVEBENCH_PATCH applied (its sha256, read in the image with no network)."""
    cmd = ["docker", "run", "--rm", "--network", "none", image, "python", "-c", _PATCH_CHECK_SRC]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as e:
        raise infra_required("livebench", f"could not check gbench's instruction_following "
                             f"grading patch in image {image!r} ({type(e).__name__}: {e}). "
                             + build, DOCS_URL) from e
    found = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith(_PATCH_MARKER):
            found = line[len(_PATCH_MARKER):].strip()
    if found == LIVEBENCH_PATCH["sha256_patched"]:
        return
    file = LIVEBENCH_PATCH["file"]
    if found == LIVEBENCH_PATCH["sha256_upstream"]:
        why = (f"image {image!r} holds LiveBench's unpatched {file}, whose instruction_following "
               "grading raises TypeError: unhashable type: 'dict'; rebuild it with the current "
               "docker/livebench.Dockerfile, which applies gbench's recorded patch "
               "(LIVEBENCH_PATCH). ")
    elif found:
        why = (f"image {image!r} holds a {file} with sha256 {found}, not the pinned file with "
               f"gbench's recorded patch applied ({LIVEBENCH_PATCH['sha256_patched']}); rebuild "
               "it from the pinned checkout with docker/livebench.Dockerfile. ")
    else:
        why = (f"gbench's instruction_following grading patch could not be checked in image "
               f"{image!r} (exit {proc.returncode}): "
               + (proc.stderr or proc.stdout or "")[-300:].strip() + ". ")
    raise infra_required("livebench", why + build, DOCS_URL)


def _config_candidates(model_id: str) -> List[str]:
    """Names to look the served model up by in LiveBench's configs: the served model id, then the
    run's --tokenizer, each as given and as its last path component (a Hugging Face id such as
    google/gemma-4-31B-it is gemma-4-31b-it there; LiveBench matches case-insensitively)."""
    names: List[str] = []
    for ident in (model_id, get_run_knob("tokenizer")):
        ident = str(ident or "").strip().rstrip("/")
        for name in (ident, ident.rsplit("/", 1)[-1]):
            if name and name not in names:
                names.append(name)
    return names


def upstream_model_config(image: str, names: List[str]) -> Dict[str, Any]:
    """LiveBench's own config for the served model, looked up in the image by LiveBench's
    get_model_config (run_livebench.py --model's lookup) under each of `names` in turn.

    Returns {"name": the name that matched or None, "config": the config as a dict or None,
    "commit": the image's LiveBench commit or None}. The image not answering is infra_required;
    a name LiveBench finds in several configs (its get_model_config raises) is an error."""
    cmd = ["docker", "run", "--rm", "--network", "none", image, "python", "-c",
           _CONFIG_LOOKUP_SRC, *names]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as e:
        raise infra_required("livebench", f"could not run LiveBench's model-config lookup in "
                             f"image {image!r} ({type(e).__name__}: {e})", DOCS_URL) from e
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith(_CONFIG_MARKER):
            data = json.loads(line[len(_CONFIG_MARKER):])
            break
    else:
        raise infra_required(
            "livebench", f"LiveBench's model-config lookup in image {image!r} printed no result "
            f"(exit {proc.returncode}); the image must hold a LiveBench checkout with "
            "livebench.model.api_model_config. Last output: "
            + (proc.stderr or proc.stdout or "")[-600:], DOCS_URL)
    if data.get("error"):
        raise RuntimeError(f"livebench: LiveBench's get_model_config({data.get('name')!r}) "
                           f"failed: {data['error']}")
    return {"name": data.get("name"), "config": data.get("config"), "commit": data.get("commit")}


def _leaderboard_provider(config: Dict[str, Any]) -> Optional[str]:
    """The provider LiveBench runs a config on (gen_api_answer.py setup_model): its
    default_provider, else the first api_name key."""
    return config.get("default_provider") or next(iter(config.get("api_name") or {}), None)


def translate_model_config(config: Dict[str, Any], enable_thinking: bool
                           ) -> Tuple[OfficialSampling, Dict[str, Any], Optional[str], List[str]]:
    """What LiveBench's config of the served model sends, for the local endpoint.

    The kwargs are the ones setup_model gives the config on its leaderboard provider (api_kwargs
    `default` updated with that provider's entry). Returns (the OfficialSampling they define, the
    kwargs gbench sends as they are, the key that carries the token cap or None, why the run
    cannot reproduce the config: a field with no local counterpart, or a reasoning toggle other
    than --thinking)."""
    name = config.get("display_name")
    api_kwargs = config.get("api_kwargs") or {}
    provider = _leaderboard_provider(config)
    sent = dict(api_kwargs.get("default") or {})
    if provider and isinstance(api_kwargs.get(provider), dict):
        sent.update(api_kwargs[provider])
    gaps: List[str] = []
    kept: Dict[str, Any] = {}
    cap_key = next((k for k in _CAP_KEYS if k in sent), None)
    cap = sent.get(cap_key) if cap_key else None
    if cap_key and (isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0):
        gaps.append(f"{cap_key}={cap!r} is not a token count gbench can send")
        cap_key, cap = None, None
    for key, value in sent.items():
        if key == "temperature" or key in OFFICIAL_SAMPLING_PARAM_KEYS or key in _PORTABLE_KEYS:
            kept[key] = value
        elif key in _CAP_KEYS:
            continue
        elif key == "extra_body" and isinstance(value, dict):
            for field, inner in value.items():
                if field == "reasoning" and isinstance(inner, dict) and set(inner) == {"enabled"}:
                    # The provider's reasoning switch: --thinking on the local endpoint.
                    if bool(inner["enabled"]) != bool(enable_thinking):
                        gaps.append(f"LiveBench's config for {name} runs with reasoning "
                                    f"{'enabled' if inner['enabled'] else 'disabled'} "
                                    f"({provider} extra_body.reasoning.enabled); this run is "
                                    f"{'--thinking' if enable_thinking else 'no-think'}")
                elif field == "chat_template_kwargs" and isinstance(inner, dict):
                    wanted = inner.get("enable_thinking")
                    if wanted is not None and bool(wanted) != bool(enable_thinking):
                        gaps.append(f"LiveBench's config for {name} sets enable_thinking="
                                    f"{wanted}; this run is "
                                    f"{'--thinking' if enable_thinking else 'no-think'}")
                    others = {k: v for k, v in inner.items() if k != "enable_thinking"}
                    if others:
                        kept.setdefault("extra_body", {})["chat_template_kwargs"] = others
                else:
                    gaps.append(f"extra_body.{field}={inner!r} of LiveBench's config for {name} "
                                f"has no counterpart on a local endpoint and was not sent")
        else:
            gaps.append(f"{key}={value!r} of LiveBench's config for {name} has no counterpart on "
                        "a local endpoint and was not sent")
    common = dict(temperature=sent.get("temperature"), send_temperature="temperature" in sent,
                  params={k: v for k, v in sent.items() if k in OFFICIAL_SAMPLING_PARAM_KEYS})
    ref = (f"github.com/LiveBench/LiveBench@{LIVEBENCH_COMMIT[:8]} livebench/model/model_configs: "
           f"{name} on {provider} ({json.dumps(sent, sort_keys=True, default=str)})")
    if cap is None:
        sampling = OfficialSampling(
            reference=ref + "; no token cap, so gen_api_answer.py's --max-tokens default 4096",
            max_tokens=LIVEBENCH_MAX_TOKENS, **common)
    else:
        # The model's own cap: sent without --thinking, and it is the model's setting for
        # reasoning too (never below the run's --max-output-tokens).
        sampling = OfficialSampling(reference=ref, max_tokens=int(cap), **common,
                                    reasoning=OfficialSampling(reference=ref,
                                                               max_tokens=int(cap), **common))
    return sampling, kept, cap_key, gaps


def _budget(sampling: OfficialSampling, enable_thinking: bool,
            explicit: Optional[int]) -> Tuple[int, str, Any]:
    """The max_tokens LiveBench sends, where it came from, and the token-limit plan.

    base.resolve_token_budget decides it. The thinking floor (no --max-output-tokens) is fitted
    to the served window here (source "default:fit-to-window"); run_livebench fits every other
    source the same way (fit_to_window), since LiveBench sends one fixed max_tokens and vLLM
    rejects every request whose prompt + max_tokens exceeds the window.
    """
    plan = resolve_token_budget("livebench", sampling, thinking=enable_thinking,
                                max_output_tokens=explicit)
    if plan.policy == "official":
        return int(plan.max_tokens), "official", plan
    if plan.policy == "official_reasoning":
        return (int(plan.max_tokens),
                "official" if plan.max_tokens == plan.official_cap else "cli:--max-output-tokens",
                plan)
    if explicit:
        return int(plan.max_tokens), "cli:--max-output-tokens", plan
    budget = int(plan.max_tokens)
    fitted, window = fit_to_window(budget)
    if window:
        return fitted, "default:fit-to-window", plan
    return budget, "default", plan


def fit_to_window(max_tokens: int) -> Tuple[int, Optional[int]]:
    """(the max_tokens LiveBench can send in the served context window, the window when that
    lowered it, else None).

    base clamps each request of the other suites to the room its prompt leaves in the window;
    LiveBench sends one fixed max_tokens with every request, and vLLM rejects a request whose
    prompt + max_tokens exceeds the window, so the same clamp is taken once, for the longest
    prompt: _PROMPT_RESERVE_TOKENS stay free for it (half the window when that is smaller).
    Unknown window: unchanged."""
    window = get_run_knob("max_model_len")
    if not window or int(window) <= 0:
        return int(max_tokens), None
    room = max(int(window) // 2, int(window) - _PROMPT_RESERVE_TOKENS)
    if int(max_tokens) > room:
        return room, int(window)
    return int(max_tokens), None


def _write_model_config(outdir: str, model_id: str, enable_thinking: bool,
                        api_kwargs: Optional[Dict[str, Any]] = None,
                        fields: Optional[Dict[str, Any]] = None) -> str:
    """LiveBench model config carrying the reasoning toggle; returns its path under /out.

    `api_kwargs` (the served model's LiveBench config kwargs gbench sends, and its token cap) and
    `fields` (its prompt_prefix / preserve_reasoning) come from translate_model_config; without
    them the config carries only the toggle. `api_kwargs.default` is merged into every request
    LiveBench sends, and LiteLLM forwards `extra_body` to the OpenAI-compatible endpoint verbatim,
    which is how vLLM receives `chat_template_kwargs`. JSON is valid YAML, so no YAML writer is
    needed.
    """
    default = json.loads(json.dumps(api_kwargs or {}))
    body = default.setdefault("extra_body", {})
    ctk = body.setdefault("chat_template_kwargs", {})
    ctk["enable_thinking"] = bool(enable_thinking)
    cfg = {"display_name": _MODEL_CONFIG_NAME,
           "api_name": {"local": model_id},
           "aliases": [],
           "api_kwargs": {"default": default},
           **{k: v for k, v in (fields or {}).items() if v is not None}}
    with open(os.path.join(outdir, _MODEL_CONFIG_FILE), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return f"/out/{_MODEL_CONFIG_FILE}"


def _build_command(image: str, outdir: str, model_id: str, display: str, base_url: str,
                   concurrency: int, temperature: Optional[float], release: str, extra: str,
                   max_tokens: Optional[int] = None,
                   model_config: Optional[str] = None,
                   question_ids: Optional[List[str]] = None,
                   owner: Optional[str] = None) -> List[str]:
    """The `docker run` that runs LiveBench's own inference+grading+report inside the image.

    `question_ids` (select_questions) runs exactly those questions: the whole-suite bench name
    with run_livebench.py's `--question-id`, which it passes to both LiveBench scripts.

    `model_config` (a path under /out) is copied into the installed package's model_configs/
    directory first and selected with `--model`, so its api_kwargs apply to every request.
    `temperature` is forwarded as `--force-temperature`; run_livebench passes None (the run's
    temperature goes in the model config instead), which leaves the flag out.

    The container runs on docker's bridge network (tau_common.ENDPOINT_NETWORK_ARGS), never the
    host network: its grading executes the model's code (the coding category), inference needs
    the endpoint and both load the questions from Hugging Face, and the host's DOCKER-USER rule
    keeps the cloud metadata server out of reach of a bridge network only. `base_url` is the
    endpoint as that container reaches it (run_livebench passes require_endpoint_network's URL).

    `owner` ("uid:gid", the host user running gbench) gets everything the container (root)
    wrote under /out once it is done, so the run's output directory stays readable and removable
    by that user.
    """
    # LiveBench's per-category bench-name path truncates a category at the first
    # underscore (common.py: split_bench_name[1].split('_')[0]), so "live_bench/data_analysis"
    # resolves to the non-existent HF dataset "livebench/data" and the whole run crashes
    # ("No objects to concatenate"). The whole-suite name "live_bench" uses the correct
    # code path (it iterates LIVE_BENCH_CATEGORIES by full name), so use it whenever the
    # full default set is requested. A subset that includes a multi-word category can't be
    # isolated by bench-name upstream; warn rather than silently emit a broken name.
    cats = _categories()
    qids = ""
    if question_ids is not None:
        # The whole-suite path, filtered by id in both scripts (load_questions' question_ids).
        benches = "live_bench"
        qids = "--question-id " + " ".join(shlex.quote(q) for q in question_ids) + " "
    elif set(cats) == set(_CATEGORIES):
        benches = "live_bench"
    else:
        multiword = [c for c in cats if "_" in c]
        if multiword:
            # run_livebench selects such a subset by question id (select_questions); a
            # per-category bench name would be truncated at the underscore upstream.
            raise ValueError(f"livebench: categories {multiword} need the question-id selection "
                             "(LiveBench's per-category loader truncates them at the underscore)")
        benches = " ".join(f"live_bench/{c}" for c in cats)
    rel = f"--livebench-release-option {release} " if release else ""
    # Before any user EXTRA_ARGS, so an explicit --max-tokens there wins (argparse keeps the last).
    budget = f"--max-tokens {max_tokens} " if max_tokens else ""
    force = f"--force-temperature {temperature} " if temperature is not None else ""
    setup, model = "", model_id
    if model_config:
        # Resolve the directory from the installed package so any LiveBench image layout works.
        setup = ("cp -f " + shlex.quote(model_config) + " \"$(python -c 'from "
                 "livebench.model.api_model_config import MODEL_CONFIGS_DIR; "
                 "print(MODEL_CONFIGS_DIR)')/zz_" + _MODEL_CONFIG_FILE + "\" && ")
        model = _MODEL_CONFIG_NAME
    # --mode single runs inference+grading in one process (no tmux, unlike sequential/parallel).
    # --ignore-missing-judgments lets show_livebench_result score whatever categories completed
    # instead of dropping the model for lacking every task (a full run has them all anyway).
    inner = (
        f"{setup}python run_livebench.py "
        f"--model {model} --model-display-name {display} "
        f"--api-base {base_url} --api-key EMPTY --use-litellm "
        f"--bench-name {benches} --question-source huggingface --mode single "
        f"--parallel-requests {concurrency} --parallel-grading {concurrency} "
        f"{force}{budget}{rel}{qids}{extra} "
        f"&& python show_livebench_result.py --model-list {display} --bench-name live_bench "
        f"--ignore-missing-judgments {rel}"
        "&& cp -f all_groups.csv all_tasks.csv /out/ 2>/dev/null || true; "
        # LiveBench's answer rows (data/<task>/model_answer/<display>.jsonl), read for replies cut
        # off by the token budget (answer_cutoffs).
        f"mkdir -p /out/{_ANSWER_DIR} && "
        f"find data -path '*/model_answer/*' -name {shlex.quote(display + '.jsonl')} "
        f"-exec cp --parents {{}} /out/{_ANSWER_DIR}/ \\; 2>/dev/null || true; "
        # LiveBench's judgment rows (data/<task>/model_judgment/ground_truth_judgment.jsonl), the
        # per-question scores (read_judgment_rows).
        f"mkdir -p /out/{_JUDGMENT_DIR} && "
        f"find data -path '*/model_judgment/*' -name ground_truth_judgment.jsonl "
        f"-exec cp --parents {{}} /out/{_JUDGMENT_DIR}/ \\; 2>/dev/null || true"
        + (f"; chown -R {shlex.quote(owner)} /out 2>/dev/null || true" if owner else "")
    )
    from .tau_common import ENDPOINT_NETWORK_ARGS
    return ["docker", "run", "--rm", *ENDPOINT_NETWORK_ARGS,
            "-v", f"{outdir}:/out", image, "bash", "-lc", inner]


#: Where the container copies LiveBench's answer rows, under the run's output directory.
_ANSWER_DIR = "model_answer"


#: How far below max_tokens the completion tokens estimated from a LiveBench answer row
#: (read_answer_rows) may fall and still count as a cut-off. The estimate
#: (total_output_tokens + answer tokens) / 2 misses, by half their number, the generated tokens
#: that are neither reasoning nor answer (the reasoning delimiters) and any difference between
#: the answer's tokens as generated and as retokenized.
_CUTOFF_TOLERANCE_TOKENS = 16

#: The error of an `api_error` row whose reply the model finished with no content. LiveBench's
#: litellm path (completions.py chat_completion_litellm) raises it when the reply's content is
#: empty and its finish_reason is not "length" (empty at "length" is its token_exhaustion
#: instead); retry_fail records the exception's class and message as `error` / `error_msg`.
_EMPTY_REPLY_ERROR = "Exception"
_EMPTY_REPLY_MSG = re.compile(r"No message returned from litellm \(finish_reason=([^)]*)\)$")
#: The result's empty_replies_note.
_EMPTY_REPLIES_NOTE = (
    "replies the model finished with no answer text (after its reasoning, or none at all): "
    "LiveBench's litellm path raises 'No message returned from litellm (finish_reason=...)' for "
    "them and records an api_error row with the answer $ERROR$, which LiveBench scores 0; they "
    "are the model's outcome, scored as LiveBench scores them. litellm also reports a stream "
    "with no finish_reason, or vLLM's abort, as stop, so such a row is counted here too; "
    "server_aborted_requests counts the requests a vLLM server aborted or failed during the run")

#: vLLM's Prometheus counter of finished requests by reason; "abort" and "error" are requests the
#: server ended itself (for example while draining at shutdown), not replies the model finished.
_VLLM_FINISHED = re.compile(
    r'^vllm:request_success_total\{(?P<labels>[^}]*)\}\s+(?P<value>[0-9.eE+-]+)\s*$', re.M)


def server_aborted_requests(base_url: str, model_ids: List[str]) -> Optional[int]:
    """Requests the served vLLM server has aborted or failed so far (its request_success_total
    with finished_reason abort or error), summed over engines for the served model (every model
    when no line names it); None when the endpoint exposes no vLLM /metrics. LiveBench records
    such a request as an empty reply, so the count taken before and after a run is how many of
    the run's empty replies the server, not the model, produced."""
    import urllib.request
    url = re.sub(r"/v1/?$", "", base_url.rstrip("/")) + "/metrics"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            text = resp.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - not a vLLM endpoint, or unreachable
        return None
    lines = [(m.group("labels"), float(m.group("value"))) for m in _VLLM_FINISHED.finditer(text)]
    if not lines:
        return None
    ended = [(labels, v) for labels, v in lines
             if re.search(r'finished_reason="(abort|error)"', labels)]
    named = [(labels, v) for labels, v in ended
             if any(f'model_name="{mid}"' in labels for mid in model_ids if mid)]
    return int(sum(v for _, v in (named or ended)))


def _token_count(text: str) -> Optional[int]:
    """Tokens of `text` under the run's tokenizer (base's loader: --tokenizer, else the served
    model's id), or None when it cannot be loaded."""
    from .base import _count_tokens
    return _count_tokens(text)


def read_answer_rows(outdir: str, max_tokens: int) -> Dict[str, int]:
    """What LiveBench's answer rows in `outdir` show about replies stopped at `max_tokens`.

    Returns counts: `rows` read; `cut`, the replies that stopped at max_tokens; `in_reasoning`,
    how many of those stopped inside the reasoning channel; `uncounted`, rows that may have
    stopped there but cannot be told apart. LiveBench scores every `api_error` row 0; they are
    split by cause: `empty`, replies the model finished with no content (error `Exception`,
    error_msg "No message returned from litellm (finish_reason=X)" with X other than length),
    the model's outcome; `failed`, the others, requests that failed at the endpoint after the
    retries of LiveBench's litellm path (_failed_request; failed_requests_by_category counts them
    per category). litellm (1.102.1 in the image) reports a stream that carried no
    finish_reason, or one it does not map (vLLM's "abort", sent when the server aborts in-flight
    requests, as at shutdown), as "stop" too, so such an empty reply is counted in `empty`; a
    server that stops mid-run also fails the requests after it, counted in `failed`.

    A row carries the reply's answer text (choices[0].turns) and `total_output_tokens`, which
    LiveBench's litellm path (completions.py chat_completion_litellm) builds as the server's
    completion_tokens C plus its completion_tokens_details.reasoning_tokens R when the server
    reports them (checked in an image built from LIVEBENCH_COMMIT against a stub streaming
    endpoint: completion_tokens 50 with reasoning_tokens 30 is recorded as 80). vLLM's
    completion_tokens already includes the reasoning, vLLM reports reasoning_tokens whenever it
    runs a reasoning parser, and C never exceeds max_tokens, so with K the answer's tokens:

    * `eval_status: token_exhaustion` (finish_reason length, no content): cut off inside the
      reasoning channel;
    * total below max_tokens: C is below it too, so the reply was not cut off;
    * total equal to max_tokens: cut off (C = max_tokens with no reasoning tokens reported;
      a reply that finished with C + R landing exactly on max_tokens is not told apart);
    * total above max_tokens: the server reported R, so C = (total + K) / 2; cut off when that
      reaches max_tokens less _CUTOFF_TOLERANCE_TOKENS. K is counted with the run's
      tokenizer (--tokenizer, else the served model's id); when it cannot be loaded the row is
      uncounted. (A server that streams reasoning but reports no reasoning_tokens is recorded
      with litellm's own count of the reasoning, which this rule does not assume.)

    A row without an output token count (LiveBench records -1 when the server returned no
    usage), or with several turns at or above max_tokens, is uncounted. Every core question is
    one turn."""
    import glob
    out = {"rows": 0, "cut": 0, "in_reasoning": 0, "uncounted": 0, "empty": 0, "failed": 0}
    pattern = os.path.join(outdir, _ANSWER_DIR, "**", "*.jsonl")
    for path in glob.glob(pattern, recursive=True):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(row, dict):
                    continue
                out["rows"] += 1
                status = row.get("eval_status")
                if status == "api_error":
                    out["failed" if _failed_request(row) else "empty"] += 1
                    continue
                if status == "token_exhaustion":
                    out["cut"] += 1
                    out["in_reasoning"] += 1
                    continue
                try:
                    total = int(row.get("total_output_tokens"))
                except (TypeError, ValueError):
                    total = -1
                if total < 0:
                    out["uncounted"] += 1
                    continue
                if total < int(max_tokens):
                    continue
                turns = ((row.get("choices") or [{}])[0] or {}).get("turns") or []
                if len(turns) != 1:
                    out["uncounted"] += 1
                    continue
                if total == int(max_tokens):
                    out["cut"] += 1
                    continue
                answer_tokens = _token_count(str(turns[0] or ""))
                if answer_tokens is None:
                    out["uncounted"] += 1
                elif (total + answer_tokens) / 2 >= int(max_tokens) - _CUTOFF_TOLERANCE_TOKENS:
                    out["cut"] += 1
    return out


def _failed_request(row: Dict[str, Any]) -> bool:
    """Whether an `api_error` answer row is a request that failed at the endpoint: every one but a
    reply the model finished with no content (error `Exception`, error_msg "No message returned
    from litellm (finish_reason=X)" with X other than length)."""
    empty = _EMPTY_REPLY_MSG.match(str(row.get("error_msg") or ""))
    return not (row.get("error") == _EMPTY_REPLY_ERROR and empty and empty.group(1) != "length")


def failed_requests_by_category(outdir: str) -> Dict[str, int]:
    """The answer rows in `outdir` that are requests which failed at the endpoint (`api_error`
    rows, _failed_request), per category: the <category> of the answer file's
    data/live_bench/<category>/<task>/model_answer/ path, `unknown` for any other path."""
    import glob
    root = os.path.join(outdir, _ANSWER_DIR)
    counts: Dict[str, int] = {}
    for path in glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True):
        parts = os.path.relpath(path, root).split(os.sep)
        category = (parts[2] if len(parts) >= 3 and parts[:2] == ["data", "live_bench"]
                    else "unknown")
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if (isinstance(row, dict) and row.get("eval_status") == "api_error"
                        and _failed_request(row)):
                    counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


def answered_categories(outdir: str) -> List[str]:
    """The categories with a non-empty LiveBench answer file in `outdir` (the container copies
    data/live_bench/<category>/<task>/model_answer/<display>.jsonl there). A category missing
    from all_groups.csv that has answers was answered but not scored by LiveBench's grading; one
    without any had no active question at the release."""
    import glob
    root = os.path.join(outdir, _ANSWER_DIR)
    found = set()
    for path in glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True):
        parts = os.path.relpath(path, root).split(os.sep)
        if (len(parts) >= 3 and parts[:2] == ["data", "live_bench"]
                and os.path.getsize(path) > 0):
            found.add(parts[2])
    return sorted(found)


def answer_cutoffs(outdir: str, max_tokens: int, thinking: bool) -> Tuple[int, int, int]:
    """(replies cut off at `max_tokens`, how many of them in the reasoning channel, answer rows
    read), by read_answer_rows. The rule is the same with and without --thinking (`thinking` is
    accepted for callers): a no-think row reports no reasoning tokens."""
    rows = read_answer_rows(outdir, max_tokens)
    return rows["cut"], rows["in_reasoning"], rows["rows"]


def instruction_following_rows(outdir: str) -> Dict[str, int]:
    """Answer rows per instruction_following task in `outdir` (the container copies
    data/live_bench/instruction_following/<task>/model_answer/<display>.jsonl there)."""
    import glob
    root = os.path.join(outdir, _ANSWER_DIR)
    counts: Dict[str, int] = {}
    for path in glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True):
        parts = os.path.relpath(path, root).split(os.sep)
        if len(parts) >= 4 and parts[:3] == ["data", "live_bench", "instruction_following"]:
            with open(path, encoding="utf-8") as f:
                n = sum(1 for line in f if line.strip())
            if n:
                counts[parts[3]] = counts.get(parts[3], 0) + n
    return dict(sorted(counts.items()))


def _parse_all_groups(path: str, display: str) -> Dict[str, float]:
    """Read LiveBench's all_groups.csv (index=model, columns=categories + 'average')."""
    scores: Dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        cols = header[1:]  # first col is the (unnamed) model index
        for row in reader:
            if not row:
                continue
            # LiveBench lowercases the model slug in its outputs -> match case-insensitively.
            row_slug = re.sub(r"[^A-Za-z0-9._-]", "-", row[0]).lower()
            if row[0].strip().lower() == display.lower() or row_slug == display.lower():
                for name, val in zip(cols, row[1:]):
                    try:
                        scores[name] = float(val)
                    except (TypeError, ValueError):
                        pass
                break
    return scores


def _token_note(plan: Any, whose: str, knob: Optional[int], sent: int,
                lowered_for: Optional[int], extra_tokens: Optional[int]) -> str:
    """The result's max_output_tokens_note.

    The token-limit plan's own note when LiveBench was given the plan's max_tokens; otherwise
    (lowered to fit the served window, or replaced by GBENCH_LIVEBENCH_EXTRA_ARGS) a note built
    from what the policy gives and what each request was sent, so it never says that a value
    was sent when it was not. `whose` names the official cap (the model config's, or
    gen_api_answer.py's default)."""
    if not lowered_for and extra_tokens is None:
        return plan.note or ""
    cap = plan.official_cap
    not_applied = (f" (--max-output-tokens {knob} not applied)"
                   if knob and int(knob) != cap else "")
    if plan.policy == "official":
        parts = [f"{whose}, the cap a no-think run gets{not_applied}"]
    elif plan.policy == "official_reasoning" and int(plan.max_tokens) > int(cap):
        parts = [f"{whose}; --max-output-tokens {knob} is larger and is the budget a thinking run "
                 "gets (a thinking run is never capped below --max-output-tokens)"]
    elif plan.policy == "official_reasoning":
        parts = [f"{whose}, the model's own setting, which a thinking run gets{not_applied}"]
    else:
        source = (f"--max-output-tokens {knob}" if knob
                  else f"the thinking floor {plan.max_tokens}")
        parts = [f"{REASONING_CAP_NOTE} ({whose}); this run's budget is {source}"]
    if extra_tokens is not None:
        parts.append(f"GBENCH_LIVEBENCH_EXTRA_ARGS asked for --max-tokens {extra_tokens} instead")
    if lowered_for:
        parts.append(f"lowered to {sent} to fit the served context window ({lowered_for} tokens, "
                     f"{lowered_for - sent} left for the prompt)")
    return "; ".join(parts) + f"; each request was sent max_tokens={sent}"


def _raise_if_unresolved(proc: Any) -> None:
    """infra_required when a run that produced no scores shows a failed host-name lookup in its
    output (for example Hugging Face unreachable for the question download): a DNS failure on the
    bridge network, never the model's outcome."""
    from .tau_common import network_failures
    failed = network_failures(f"{proc.stdout or ''}\n{proc.stderr or ''}", dns_only=True)
    if failed:
        from .container_guard import FIREWALL_COMMAND
        raise infra_required(
            "livebench", "LiveBench's run container could not resolve a host name, so it scored "
            f"nothing ({failed[0]}). It loads the questions from Hugging Face on the docker "
            "bridge network, which needs working DNS: on a cloud VM, block the metadata server "
            f"with `{FIREWALL_COMMAND}` (a plain DROP rule for 169.254.169.254 also drops DNS, "
            "since the VPC DNS resolver is that address), and check the docker daemon's DNS "
            "settings.", DOCS_URL)


def run_livebench(model_name: str, base_url: str, concurrency: int,
                  enable_thinking: bool = False, **kwargs) -> Dict[str, Any]:
    """Run LiveBench's 6 core categories inside the gbench-livebench image; report the average."""
    image = _image()
    _categories()                 # a category that is not a core one is an error before docker
    _check_prereqs(image)
    relay = _endpoint_relay(base_url)
    try:
        return _run_livebench(model_name, base_url, concurrency, enable_thinking, image, relay,
                              kwargs)
    finally:
        if relay is not None:
            relay.stop()


def _endpoint_relay(base_url: str):
    """An EndpointRelay on the docker bridge gateway that adds the endpoint's key
    (base.endpoint_api_key) to LiveBench's requests, started; None when the endpoint has no key
    (LiveBench then sends its placeholder `EMPTY` straight to it). The key never enters the
    container, where grading runs the model's code."""
    from .base import endpoint_api_key
    key = endpoint_api_key()
    if not key:
        return None
    from . import gemini_relay
    from .swebench_common import EndpointRelay
    gateway = gemini_relay.network_gateway("livebench", "bridge", DOCS_URL)
    try:
        relay = EndpointRelay(base_url, gateway, key=key, name="livebench-model-relay")
        # a --thinking reply may run for a long time; the relay must not cut it short
        relay.TIMEOUT_S = max(EndpointRelay.TIMEOUT_S, 6 * 60 * 60)
    except ValueError as e:
        raise infra_required("livebench", f"the model endpoint is not an http(s) URL with a host "
                             f"({e}).", DOCS_URL) from None
    try:
        return relay.start()
    except OSError as e:
        raise infra_required("livebench", f"gbench's relay to the model endpoint could not listen "
                             f"on {gateway} ({type(e).__name__}: {e}).", DOCS_URL) from None


def _run_livebench(model_name: str, base_url: str, concurrency: int, enable_thinking: bool,
                   image: str, relay: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """run_livebench after the prerequisite checks (with the key relay, when there is one)."""
    # The run container executes the model's code on the bridge network: the metadata server must
    # be blocked there, and a loopback endpoint is reached at host.docker.internal.
    from .tau_common import require_endpoint_network
    container_base_url = require_endpoint_network("livebench", image,
                                                  relay.url if relay else base_url, DOCS_URL)

    model_id, model_id_source = _request_model_id(base_url, model_name)
    display = re.sub(r"[^A-Za-z0-9._-]", "-", model_id).lower()  # LiveBench lowercases the slug
    release, release_src = _release()
    extra = os.environ.get("GBENCH_LIVEBENCH_EXTRA_ARGS", "").strip()
    limit = kwargs.get("limit")
    limit = int(limit) if limit and int(limit) > 0 else None
    cats = _categories()
    shard = current_shard()
    # gbench selects the questions by id (select_questions) for a shard, a limit, or a category
    # subset LiveBench's per-category loader cannot isolate (a multi-word name).
    subset_cats = set(cats) != set(_CATEGORIES)
    select_ids = bool(limit or (shard and shard.partial)
                      or (subset_cats and any("_" in c for c in cats)))
    extra_selects = re.search(r"--question-(id|begin|end)\b", extra)
    if select_ids and extra_selects:
        raise ValueError(f"livebench: GBENCH_LIVEBENCH_EXTRA_ARGS selects questions "
                         f"({extra_selects.group(0)}) while gbench selects them for --shard, "
                         f"--eval-limit or the category subset; drop it. See {DOCS_URL}")
    question_ids: Optional[List[str]] = None
    shard_label: Optional[str] = None
    n_full = None
    limit_plan: Dict[str, Any]
    # When gbench selects the questions it has the release's question list, so it knows which
    # categories have no active question at the release (LiveBench's loader lists none) and which
    # have some but none among the selected questions (--eval-limit's stratified draw or the
    # shard gave them 0). Neither is a failure of the run: their missing score is expected and is
    # recorded apart from categories_missing. A full run passes no ids and lists nothing, so both
    # stay empty there.
    inactive_at_release: List[str] = []
    not_selected: List[str] = []
    if select_ids:
        listed = list_questions(image, release)
        selected, shard_label, _positions, n_full = select_questions(listed, cats, limit)
        # --eval-limit N is N questions for the whole suite, spread over the categories (the
        # stratified draw in select_questions), never N per category.
        category_of: Dict[str, str] = {}
        for cat, _task, qid in listed:
            if cat in cats:
                category_of.setdefault(qid, cat)
        per_category = {cat: 0 for cat in cats}
        for qid in selected:
            per_category[category_of[qid]] += 1
        limit_plan = eval_limit_plan("livebench", limit, len(selected), per_category,
                                     setting_noun="category", attempts=1)
        listed_cats = set(category_of.values())
        inactive_at_release = [c for c in cats if c not in listed_cats]
        not_selected = [c for c in cats if c in listed_cats and not per_category[c]]
        if shard and shard.partial and not selected:
            reason = (f"empty shard {shard_label}: no items on this node ({n_full} in the full "
                      "set)")
            logger.info("livebench: %s", reason)
            return {
                "benchmark_type": "eval", "eval_name": "livebench", "model_name": model_name,
                "thinking": bool(enable_thinking), "status": "success", "status_reason": reason,
                "total_questions": 0, "correct_answers": 0, "accuracy": None,
                "accuracy_withheld_reason": reason, "empty_shard": True, "full_set_items": n_full,
                "dimension_scores": {}, "sample_traces": [], "selected_question_ids": [],
                "livebench_release": release or "latest", **shard_identity(shard),
                "eval_limit_plan": limit_plan,
                "leaderboard_comparable": False, "leaderboard_comparable_reason": reason,
            }
        if not selected:
            raise RuntimeError(f"livebench: no question of {cats} is active at release "
                               f"'{release or 'latest'}', so nothing can be selected (not a 0%)")
        question_ids = selected
    else:
        # A full run passes no question ids: LiveBench runs every active question of the
        # categories at the release (not listed here, which would need the image's network).
        limit_plan = {"limit": None, "samples": None, "per_setting": None,
                      "attempts_per_sample": 1, "generations": None,
                      "note": f"every active question of {len(cats)} category(ies) at release "
                              f"{release or 'latest'}"}
        logger.info("[livebench] full run (no --eval-limit): every active question of %d "
                    "category(ies) (%s) at release %s x 1 attempt", len(cats), ", ".join(cats),
                    release or "latest")
    # LiveBench's own config for the served model, if it ships one, decides the request kwargs.
    lookup = upstream_model_config(image, _config_candidates(model_id))
    upstream = lookup.get("config")
    config_gaps: List[str] = []
    config_kwargs: Dict[str, Any] = {}
    cap_key: Optional[str] = None
    if upstream:
        sampling, config_kwargs, cap_key, config_gaps = translate_model_config(upstream,
                                                                               enable_thinking)
    else:
        sampling = OFFICIAL_SAMPLING
    temperature, temp_src = resolve_official_sampling("livebench", sampling,
                                                      thinking=enable_thinking)
    # A config's temperature replaces get_answer's (and --force-temperature) on every request, so
    # the run's temperature (both modes) goes into the config itself: --force-temperature would
    # stop LiveBench on a question with a required temperature.
    config_kwargs["temperature"] = temperature
    force_temperature = None
    explicit = kwargs.get("max_output_tokens") or get_run_knob("max_output_tokens")
    max_tokens, tokens_src, plan = _budget(sampling, enable_thinking, explicit)
    m = re.search(r"--max-tokens[= ](\d+)", extra)
    if m:                       # the operator's own LiveBench flag wins; record what it says
        max_tokens, tokens_src = int(m.group(1)), "env:GBENCH_LIVEBENCH_EXTRA_ARGS"
    # Whatever its source, the one max_tokens LiveBench sends must leave the prompt room in the
    # served window, or vLLM rejects every request; it is lowered as base's request path lowers a
    # request's, and the result records it (a lowered official cap is not comparable).
    requested = max_tokens
    if tokens_src == "default:fit-to-window":       # _budget fitted the thinking floor already
        requested, lowered_for = int(plan.max_tokens), int(get_run_knob("max_model_len"))
    else:
        max_tokens, lowered_for = fit_to_window(max_tokens)
        if lowered_for:
            tokens_src += ":fit-to-window"
            logger.warning("livebench: max_tokens %d does not leave %d tokens of the %d-token "
                           "context window for the prompt; sending %d.", requested,
                           _PROMPT_RESERVE_TOKENS, lowered_for, max_tokens)
            if m:
                extra = extra[:m.start()] + f"--max-tokens {max_tokens}" + extra[m.end():]
    if cap_key:
        # A config's cap replaces --max-tokens, so the value to send goes into the config.
        config_kwargs[cap_key] = max_tokens
        forward = None
    else:
        forward = None if m else max_tokens

    # The run's output directory (LiveBench's answer and judgment rows, all_groups.csv, the model
    # config gbench wrote), kept and recorded as livebench_output_dir.
    outdir = _output_dir()
    model_config = _write_model_config(
        outdir, model_id, enable_thinking, config_kwargs,
        {k: upstream.get(k) for k in _PROMPT_FIELDS} if upstream else None)
    cmd = _build_command(image, outdir, model_id, display, container_base_url, concurrency,
                         force_temperature, release, extra, max_tokens=forward,
                         model_config=model_config, question_ids=question_ids,
                         owner=f"{os.getuid()}:{os.getgid()}")
    logger.info("livebench: %s", " ".join(cmd))
    aborts_before = server_aborted_requests(base_url, [model_id, model_name])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    aborts_after = server_aborted_requests(base_url, [model_id, model_name])
    server_aborted = (max(0, aborts_after - aborts_before)
                      if aborts_before is not None and aborts_after is not None else None)

    groups = os.path.join(outdir, "all_groups.csv")
    if not os.path.isfile(groups):
        _raise_if_unresolved(proc)
        raise RuntimeError(
            "livebench: LiveBench produced no all_groups.csv (inference/grading failed) - this "
            "is a harness/infra failure, not a 0%. Last output:\n"
            + (proc.stderr or proc.stdout or "")[-1200:])

    scores = _parse_all_groups(groups, display)
    dims = {c: scores[c] for c in cats if c in scores}
    answered = answered_categories(outdir)
    if not dims:
        if any(c in answered for c in cats):
            raise RuntimeError(
                f"livebench: LiveBench answered {[c for c in cats if c in answered]} at release "
                f"'{release or 'latest'}' but its grading scored none of them - a harness "
                "failure, not a 0%. Last output:\n" + (proc.stderr or proc.stdout or "")[-1200:])
        raise RuntimeError(
            "livebench: no categories were scored for release "
            f"'{release or 'latest'}' - nothing to report (not a 0%). LiveBench's public "
            "questions (Hugging Face livebench/<category>) reach release "
            f"{LIVEBENCH_RELEASE}, the default; at every later release `coding` has no active "
            "question (all carry removal date 2025-04-02) and other tasks are withheld. Unset "
            "GBENCH_LIVEBENCH_RELEASE (or pick a release where your categories are active), or "
            "narrow GBENCH_LIVEBENCH_CATEGORIES.")

    overall = scores.get("average")
    if overall is None:  # single active category: no 'average' column -> use it directly
        overall = round(sum(dims.values()) / len(dims), 2)

    # Categories with no score that the run did not ask LiveBench to answer (gbench's question
    # selection: none active at the release, or none selected) are expected to be missing; they
    # are recorded as categories_inactive_at_release / categories_not_run_by_limit and do not
    # make the status completed_with_errors. Every other category without a score is in
    # categories_missing and does.
    expected_missing = set(inactive_at_release) | set(not_selected)
    missing = [c for c in cats if c not in dims and c not in expected_missing]
    no_question_run = [c for c in cats if c not in dims and c in expected_missing]
    # A missing category LiveBench answered was not scored by its grading (a harness failure).
    # One it never answered: in a full run it had no active question at the release (coding
    # after 2024-11-25; a full run does not list the questions, so this is not checked); with
    # gbench's selection it had selected questions LiveBench did not answer (a harness failure).
    unscored = [c for c in missing if c in answered]
    inactive = [c for c in missing if c not in answered]
    unanswered = inactive if question_ids is not None else []
    if unscored or unanswered:
        logger.warning("livebench: categories without a score: answered but not graded %s, "
                       "selected but not answered %s. Last output:\n%s", unscored, unanswered,
                       (proc.stderr or proc.stdout or "")[-1200:])
    default_cats = set(cats) == set(_CATEGORIES)
    # Comparable to LiveBench's leaderboard when it's the FULL question set for a PINNED,
    # reproducible release with all default categories active. NOT gated on thinking: the
    # leaderboard includes reasoning models, and the budget and reasoning mode are forwarded
    # (and recorded below), so a --thinking run is a faithful measurement of that mode.
    reasons = []
    if not release:
        reasons.append("unpinned release (GBENCH_LIVEBENCH_RELEASE=latest: LiveBench's own "
                       f"default, its newest release; the pinned release is {LIVEBENCH_RELEASE})")
    if not default_cats:
        reasons.append(f"category subset {sorted(cats)}")
    if no_question_run:
        reasons.append(f"categories with no question run (none among the selected questions, "
                       f"or none active at this release): {no_question_run}")
    if unanswered:
        reasons.append(f"categories whose selected questions LiveBench did not answer (no "
                       f"answer rows; its output is in the log): {unanswered}")
    elif inactive:
        reasons.append(f"categories with no active questions at this release: {inactive}")
    if unscored:
        reasons.append(f"categories answered but not scored by LiveBench's grading (it wrote no "
                       f"judgment for them; its output is in the log): {unscored}")
    if limit:
        reasons.append(f"subset run (--eval-limit {limit}: {len(question_ids or [])} question(s) "
                       "drawn stratified by category with a fixed seed)")
    if extra_selects and not select_ids:
        reasons.append(f"GBENCH_LIVEBENCH_EXTRA_ARGS selects questions ({extra_selects.group(0)}), "
                       "a subset of the release")
    if not sampling.send_temperature and abs(float(temperature)) > 1e-9:
        reasons.append(f"temperature={temperature} ({temp_src}; sent as the config's temperature) "
                       "(LiveBench samples at temperature 0, or the question's required "
                       "temperature)")
    if (sampling.send_temperature
            and not (sampling.temperature is not None
                     and abs(float(temperature) - float(sampling.temperature)) < 1e-9)):
        reasons.append(f"temperature={temperature} ({temp_src}; LiveBench's config for "
                       f"{upstream.get('display_name')} samples at {sampling.temperature})")
    if m and requested != plan.max_tokens:
        reasons.append(f"max_tokens={requested} from GBENCH_LIVEBENCH_EXTRA_ARGS (the token-limit "
                       f"policy gives {plan.max_tokens}: {plan.note})")
    official_cap = plan.official_cap if plan.policy in ("official", "official_reasoning") else None
    if lowered_for and official_cap and max_tokens < official_cap:
        whose = (f"LiveBench's config for {upstream.get('display_name')}" if cap_key
                 else f"LiveBench's config for {upstream.get('display_name')} sets no cap, so "
                 "gen_api_answer.py's --max-tokens default" if upstream
                 else "gen_api_answer.py's --max-tokens default")
        reasons.append(f"max_tokens={max_tokens}: {whose} sends {official_cap}, which does not fit "
                       f"the served {lowered_for}-token context window with room for the prompt "
                       f"(serve with --max-model-len >= {official_cap + _PROMPT_RESERVE_TOKENS})")
    # A reply cut off by a max_tokens other than the official cap (the run's budget, one lowered to
    # fit the window, or GBENCH_LIVEBENCH_EXTRA_ARGS's) measures that budget, not the model.
    rows = read_answer_rows(outdir, max_tokens)
    # LiveBench's judgment of each question the run covered (the per-question traces)
    judged = read_judgment_rows(outdir, display)
    n_scored = sum(1 for t in judged if not t.get("scoring_excluded") and t.get("score") is not None)
    if rows["rows"] and rows["failed"] == rows["rows"]:
        # Every question's request failed at the endpoint: LiveBench scored them all 0, which
        # measures the harness, not the model, so no score is reported.
        raise infra_required(
            "livebench", f"every one of the {rows['rows']} requests LiveBench sent failed at the "
            "served endpoint (api_error rows: a connection error, an HTTP error, a timeout), so "
            "no score is reported. Check that the endpoint is up and reachable from the docker "
            "bridge network (host.docker.internal for an endpoint on this host, the name "
            "resolving for a remote one).", DOCS_URL)
    cut = in_reasoning = uncounted = 0
    cutoff_note = None
    if not (official_cap and max_tokens == official_cap):
        cut, in_reasoning, uncounted = rows["cut"], rows["in_reasoning"], rows["uncounted"]
        if not rows["rows"]:
            reasons.append("LiveBench's answer rows were not found in the run's output, so replies "
                           f"cut off by this run's max_tokens {max_tokens} could not be counted")
        if cut:
            reasons.append(f"{cut} repl(ies) cut off by this run's max_tokens {max_tokens}, not by "
                           f"an official cap ({in_reasoning} of them inside the reasoning channel "
                           "with no answer, LiveBench's token_exhaustion)")
        if uncounted:
            reasons.append(
                f"{uncounted} answer row(s) could not be checked for a cut-off at this run's "
                f"max_tokens {max_tokens}: no output token count, several turns at or above it, or "
                "more output tokens than max_tokens with no tokenizer to count the answer's "
                "tokens (the run's --tokenizer, else the served model's id, could not be loaded; "
                "pass --tokenizer with the served model's tokenizer)")
        cutoff_note = ("replies cut off after the answer began are counted from LiveBench's "
                       "total_output_tokens, which adds the server's reasoning_tokens to its "
                       "completion_tokens (vLLM's already include the reasoning): a total at "
                       "max_tokens, or (total + the answer's tokens under --tokenizer) / 2 within "
                       f"{_CUTOFF_TOLERANCE_TOKENS} tokens of it, is a cut-off (the run's "
                       "tokenizer: --tokenizer, else the served model's id; the server's usage "
                       "must carry reasoning_tokens, as vLLM's does whenever it runs a reasoning "
                       "parser); replies cut off inside the reasoning channel carry LiveBench's "
                       "token_exhaustion")
    if rows["failed"]:
        reasons.append(f"{rows['failed']} request(s) failed at the served endpoint (LiveBench's "
                       "api_error rows other than empty replies, which LiveBench scores 0): a "
                       "harness failure, not the model's answer")
    if server_aborted:
        reasons.append(f"{server_aborted} request(s) aborted or failed inside the served vLLM "
                       "server during the run (its request_success_total, finished_reason abort "
                       "or error): LiveBench records them as empty or cut-short replies, which "
                       "are harness failures, not the model's answers")
    # LiveBench scored every failed request 0 inside its category and overall averages. Those
    # figures measure the harness, so they are withheld: the headline, and each category with a
    # failed request (every category when the server aborted requests, which cannot be placed).
    failed_by_cat = failed_requests_by_category(outdir) if rows["failed"] else {}
    withheld: Dict[str, str] = {}
    for cat in dims:
        why = []
        if failed_by_cat.get(cat):
            why.append(f"{failed_by_cat[cat]} request(s) failed at the served endpoint "
                       "(api_error rows LiveBench scored 0)")
        if server_aborted:
            why.append(f"{server_aborted} request(s) aborted or failed inside the served vLLM "
                       "server during the run, category unknown")
        if why:
            withheld[cat] = "; ".join(why)
    headline_withheld = bool(server_aborted or any(
        cat in dims or cat not in _CATEGORIES for cat in failed_by_cat))
    withheld_reason = None
    if headline_withheld:
        parts = []
        if rows["failed"]:
            parts.append(f"{rows['failed']} request(s) failed at the served endpoint "
                         f"({', '.join(f'{c} {n}' for c, n in sorted(failed_by_cat.items()))})")
        if server_aborted:
            parts.append(f"{server_aborted} request(s) were aborted or failed inside the served "
                         "vLLM server during the run")
        withheld_reason = (
            "; ".join(parts) + ". LiveBench scored them 0 inside its averages, which would count "
            "harness failures as the model's answers, so the average is withheld"
            + (f" and so are the scores of {sorted(withheld)} (dimension_scores_withheld)"
               if withheld else "")
            + (f"; dimension_scores keeps {sorted(set(dims) - set(withheld))}, where every "
               "request was answered" if set(dims) - set(withheld) else "")
            + ". Check that the endpoint stays up and reachable from the docker bridge network "
            "for the whole run, then run the suite again.")
    scored_dims = {c: v for c, v in dims.items() if c not in withheld}
    if_rows = instruction_following_rows(outdir)
    if "instruction_following" in dims and not if_rows:
        reasons.append("instruction_following answer rows were not found in the run's output, so "
                       "the equal task counts that make LiveBench's pooled old-format "
                       "instruction_following score equal the mean of the task means (which the "
                       "recorded patch relies on) could not be verified")
    elif "instruction_following" in dims and len(set(if_rows.values())) > 1:
        reasons.append(_if_unequal_reason(if_rows))
    reasons.extend(config_gaps)
    commit = lookup.get("commit")
    if not commit:
        reasons.append(f"the {image} image's LiveBench commit could not be read (git rev-parse "
                       "HEAD in its checkout gave nothing), so it is not verified as the pinned "
                       f"{LIVEBENCH_COMMIT[:12]} (build the image from a checkout of that commit "
                       f"with its .git directory in the build context: {BUILD_COMMAND}; see "
                       f"{DOCS_URL})")
    elif commit != LIVEBENCH_COMMIT:
        reasons.append(f"the {image} image holds LiveBench {commit[:12]}, not the pinned "
                       f"{LIVEBENCH_COMMIT[:12]} (rebuild it from a checkout of that commit: "
                       f"{BUILD_COMMAND}; see {DOCS_URL})")
    mode_sampling = sampling.for_mode(enable_thinking)
    # What was actually sent: the plan's value, unless fitted to the window or set in EXTRA_ARGS.
    token_record = plan.record()
    if "max_tokens_sent" in token_record or max_tokens != plan.max_tokens:
        token_record["max_tokens_sent"] = max_tokens
    if official_cap and max_tokens != official_cap:
        token_record["max_tokens_applied"] = False
    # The official cap's source: the model config's own cap, or gen_api_answer.py's default
    # (a config that sets no cap runs with that default too).
    whose = (f"LiveBench's config for {upstream.get('display_name')} sets max_tokens="
             f"{plan.official_cap}" if cap_key
             else f"LiveBench's config for {upstream.get('display_name')} sets no cap, so "
             f"gen_api_answer.py's --max-tokens default is {plan.official_cap}" if upstream
             else f"gen_api_answer.py's --max-tokens default is {plan.official_cap}")
    note = _token_note(plan, whose, explicit, max_tokens, lowered_for, requested if m else None)
    if (max_tokens != plan.max_tokens
            and token_record.get("max_tokens_note") == RAISED_REASONING_CAP_NOTE):
        token_record["max_tokens_note"] = (
            f"--max-output-tokens is larger than the official reasoning-model cap; max_tokens "
            f"{max_tokens} was sent (" + ("fitted to the served context window" if lowered_for
                                         else "GBENCH_LIVEBENCH_EXTRA_ARGS") + ")")
    # The status counts real failures only: categories without a score the run asked for
    # (categories_missing), requests that failed at the endpoint, and server aborts. 2026-10-11
    # validation run (pool-10-11-132525, --eval-limit 5): the stratified draw gave
    # instruction_following 0 questions, and the old gate, which counted it as missing, marked a
    # complete run completed_with_errors with no status_reason.
    status_causes = []
    if unscored:
        status_causes.append(f"LiveBench answered {unscored} but its grading wrote no score for "
                             "them (categories_unscored)")
    if unanswered:
        status_causes.append(f"LiveBench did not answer the selected questions of {unanswered} "
                             "(no answer rows)")
    if inactive and not unanswered:
        status_causes.append(
            f"no score for {inactive} and no answer rows: in a full run this is a category with "
            "no active question at the release (coding after 2024-11-25), not checked against "
            "the release's question list")
    if rows["failed"]:
        status_causes.append(f"{rows['failed']} request(s) failed at the served endpoint "
                             "(failed_requests)")
    if server_aborted:
        status_causes.append(f"{server_aborted} request(s) aborted or failed inside the served "
                             "vLLM server (server_aborted_requests)")
    result = {
        "benchmark_type": "eval",
        "eval_name": "livebench",
        "model_name": model_name,
        "status": "completed_with_errors" if status_causes else "success",
        **({"status_reason": "; ".join(status_causes)} if status_causes else {}),
        # None when a failed or aborted request sits inside LiveBench's average
        # (accuracy_withheld_reason)
        "accuracy": None if headline_withheld else round(float(overall), 2),
        **({"accuracy_withheld_reason": withheld_reason} if headline_withheld else {}),
        # the questions LiveBench judged with a score (per question in sample_traces); LiveBench
        # scores each in [0, 1] and averages per task and category, so there is no count of
        # correct answers
        "total_questions": n_scored,
        "effective_n": n_scored,
        "correct_answers": None,
        "correct_answers_note": _CORRECT_ANSWERS_NOTE,
        # the model id LiveBench's requests carry, and where it came from
        "request_model_id": model_id,
        "request_model_id_source": model_id_source,
        # the run's output directory: LiveBench's model answers (model_answer/) and judgments
        # (model_judgment/), all_groups.csv / all_tasks.csv and gbench's model config
        "livebench_output_dir": outdir,
        "dimension_scores": scored_dims,
        # categories whose LiveBench score counts a failed or aborted request as 0: withheld
        "dimension_scores_withheld": withheld,
        "livebench_release": release or "latest",
        "eval_limit_plan": limit_plan,
        "livebench_release_source": release_src,   # default (LIVEBENCH_RELEASE) or the env knob
        # categories the run asked for with no score (unscored, unanswered, or in a full run
        # inactive at the release): each makes the status completed_with_errors
        "categories_missing": missing,
        "categories_unscored": unscored,   # of them, answered but not scored by LiveBench
        # categories gbench's question selection gave no question (expected to have no score;
        # not errors): none active at the release (LiveBench's loader lists none), and active but
        # none among the selected questions (--eval-limit's stratified draw, or the shard)
        "categories_inactive_at_release": inactive_at_release,
        "categories_not_run_by_limit": not_selected,
        "metric": ("LiveBench average over the core categories it scored for the release (equal "
                   "weight; categories_missing, categories_inactive_at_release and "
                   "categories_not_run_by_limit list any it did not), each scored by LiveBench's "
                   "own ground-truth judges (no LLM judge); agentic_coding excluded"),
        "thinking": bool(enable_thinking),
        "max_output_tokens": max_tokens,
        "max_output_tokens_source": tokens_src,
        "max_output_tokens_note": note,
        "temperature": temperature,
        "temperature_source": temp_src,
        "official_sampling": {**mode_sampling.record(temperature, temp_src), **token_record,
                              **({"reasoning_settings": True}
                                 if mode_sampling is not sampling else {})},
        # LiveBench's own config for the served model (None: it ships none, so its defaults apply)
        "livebench_model_config": ({
            "display_name": upstream.get("display_name"), "matched_name": lookup.get("name"),
            "provider": _leaderboard_provider(upstream), "sent": config_kwargs,
            "not_reproduced": config_gaps} if upstream else None),
        "livebench_image_commit": commit,
        # gbench's one local change to that checkout (checked in the image before the run)
        "livebench_patch": dict(LIVEBENCH_PATCH),
        # replies cut off by a max_tokens other than the official cap (read_answer_rows)
        "replies_cut_off_by_budget": cut,
        "replies_cut_off_in_reasoning": in_reasoning,
        "replies_cut_off_uncounted": uncounted,
        **({"replies_cut_off_note": cutoff_note} if cutoff_note else {}),
        # LiveBench's api_error rows, by cause (read_answer_rows): replies the model finished
        # with no content (its outcome, scored 0 as LiveBench scores it) and requests that
        # failed at the served endpoint (harness failures)
        "empty_replies": rows["empty"],
        **({"empty_replies_note": _EMPTY_REPLIES_NOTE} if rows["empty"] else {}),
        "failed_requests": rows["failed"],
        "failed_requests_by_category": failed_by_cat,
        # requests the served vLLM server aborted or failed during the run (None: the endpoint
        # exposes no vLLM /metrics, so aborts cannot be told apart from empty replies)
        "server_aborted_requests": server_aborted,
        "leaderboard_comparable": not reasons,
        # the question ids gbench selected (select_questions; None: LiveBench ran the categories'
        # whole release), and LiveBench's judgment of each question run
        "selected_question_ids": question_ids,
        "questions_judged": len(judged),
        "sample_traces": judged,
        **shard_identity(shard),
        **_attempts_record(kwargs),
    }
    if relay is not None:
        result["model_endpoint_relay"] = relay.record()
    if reasons:
        result["leaderboard_comparable_reason"] = "; ".join(reasons)
    if shard is not None and shard.partial:
        add_comparability_reason(result, SHARD_REASON.format(
            label=shard.label, index=shard.index, count=shard.count))
    return result


def _attempts_record(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """{"attempts": {...not_applied}} when --attempt-count asked for more than 1: LiveBench answers
    each question once."""
    k = int(kwargs.get("attempt_count") or get_run_knob("attempt_count") or 1)
    if k <= 1:
        return {}
    return {"attempts": {"attempts_requested": k, "attempts_per_sample": 1,
                         "not_applied": ("livebench: LiveBench answers and grades each question "
                                         f"once; --attempt-count {k} is not applied")}}


#: The comparability reasons a shard can give only because it holds part of the questions:
#: merge_shards recomputes them over the union instead of keeping a shard's copy.
_SHARD_ONLY_REASONS = ("instruction_following tasks answered in unequal numbers ",
                       "categories with no question run ")


def _if_unequal_reason(counts: Dict[str, int]) -> str:
    """The comparability reason for instruction_following tasks answered in unequal numbers."""
    return (f"instruction_following tasks answered in unequal numbers {counts}: "
            "LiveBench's pooled old-format instruction_following grading labels every "
            "row with one task, so its category score is the pooled mean of the rows, "
            "not the mean of the task means")


def _listed_categories(reason: str) -> List[str]:
    """The categories a "categories with no question run ...: [...]" reason lists."""
    import ast
    try:
        value = ast.literal_eval(reason.rsplit(": ", 1)[1])
    except (IndexError, ValueError, SyntaxError):
        return []
    return [str(c) for c in value] if isinstance(value, (list, tuple)) else []


def merge_shards(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The merged result of a sharded run's results (gbench --merge): LiveBench's aggregation
    (livebench_scores, as show_livebench_result.py computes it) over the union of the shards'
    judgment traces, de-duplicated by sample_key. A headline a scored shard withheld (a failed or
    aborted request inside LiveBench's averages) stays withheld, with the shards' reasons; an
    empty shard (more shards than questions) withholds nothing. The comparability reasons a
    shard gives only because it holds part of the questions (_SHARD_ONLY_REASONS: unequal
    instruction_following task counts, categories no question of the shard fell in) are
    recomputed over the union, and so are the category lists (categories_not_run_by_limit keeps
    only categories no shard ran a question of); the status is completed_with_errors when a
    shard's is, with the shards' status_reason. When every shard is empty, the merged result is
    success with accuracy null and keeps the shards' 'empty shard ...' status_reason, as a single
    empty shard's result does."""
    from ...merge import MergeError
    union: Dict[str, Dict[str, Any]] = {}
    first: Dict[str, int] = {}
    for n, r in enumerate(results):
        traces = r.get("sample_traces")
        if not isinstance(traces, list):
            raise MergeError("livebench: a shard carries no judgment traces (sample_traces); "
                             "re-run it with the current gbench")
        for t in traces:
            key = str(t.get("sample_key"))
            if key in first and first[key] != n:
                raise MergeError(
                    f"livebench: question {key!r} is in shard {results[first[key]].get('shard')} "
                    f"and in shard {r.get('shard')}: the shards overlap, so they are not the "
                    "partition of one ordered set (the nodes ran different data or code); re-run "
                    "the shards from one gbench version and LiveBench release")
            first.setdefault(key, n)
            union.setdefault(key, t)
    traces = sorted(union.values(), key=lambda t: (str(t.get("task")), str(t.get("question_id"))))
    agg = livebench_scores(traces)
    scored = [r for r in results if not r.get("empty_shard")]
    merged = dict(scored[0] if scored else results[0])
    merged["sample_traces"] = traces
    merged["questions_judged"] = len(traces)
    merged["selected_question_ids"] = sorted(
        {q for r in results for q in (r.get("selected_question_ids") or [])})
    withheld = [r.get("accuracy_withheld_reason") for r in scored if r.get("accuracy") is None]
    merged["dimension_scores_withheld"] = {k: v for r in results
                                           for k, v in (r.get("dimension_scores_withheld")
                                                        or {}).items()}
    merged["dimension_scores"] = {c: v for c, v in agg["categories"].items()
                                  if c not in merged["dimension_scores_withheld"]}
    n_scored = sum(1 for t in traces
                   if not t.get("scoring_excluded") and t.get("score") is not None)
    merged["total_questions"] = n_scored
    merged["effective_n"] = n_scored
    merged["correct_answers"] = None
    merged["correct_answers_note"] = _CORRECT_ANSWERS_NOTE
    merged["livebench_output_dirs"] = [r.get("livebench_output_dir") for r in results
                                       if r.get("livebench_output_dir")]
    merged.pop("livebench_output_dir", None)
    merged["task_scores"] = agg["tasks"]
    if not scored:
        merged["accuracy"] = None
        merged["accuracy_withheld_reason"] = "every shard is empty: no question was run"
    elif withheld:
        merged["accuracy"] = None
        merged["accuracy_withheld_reason"] = "; ".join(str(w) for w in withheld if w)
    else:
        merged["accuracy"] = agg["average"]
        merged.pop("accuracy_withheld_reason", None)
    for field in ("failed_requests", "empty_replies", "replies_cut_off_by_budget",
                  "replies_cut_off_in_reasoning", "replies_cut_off_uncounted"):
        merged[field] = sum(int(r.get(field) or 0) for r in results)
    aborted = [r.get("server_aborted_requests") for r in scored]
    merged["server_aborted_requests"] = (None if any(a is None for a in aborted)
                                         else sum(aborted))
    statuses = {r.get("status") for r in results}
    merged["status"] = "completed_with_errors" if "completed_with_errors" in statuses else "success"
    shard_reasons = [f"shard {r.get('shard')}: {r['status_reason']}" for r in results
                     if r.get("status") == "completed_with_errors" and r.get("status_reason")]
    if merged["status"] == "completed_with_errors" and shard_reasons:
        merged["status_reason"] = "; ".join(shard_reasons)
    elif not scored:
        # Every shard is empty: keep the shards' 'empty shard ...' reasons, as a single empty
        # shard's result does, so the success with no question run is explained.
        merged["status_reason"] = "; ".join(
            str(r["status_reason"]) for r in results if r.get("status_reason")) or (
            "every shard is empty: no question was run")
    else:
        merged.pop("status_reason", None)
    # The category lists over the union: a category a shard ran no question of is not a gap of
    # the merged run when another shard ran one; a category missing in any shard stays missing.
    ran = {t.get("category") for t in traces}
    for field in ("categories_missing", "categories_unscored"):
        merged[field] = sorted({c for r in results for c in (r.get(field) or [])})
    merged["categories_inactive_at_release"] = sorted(
        {c for r in results for c in (r.get("categories_inactive_at_release") or [])} - ran)
    merged["categories_not_run_by_limit"] = sorted(
        {c for r in results for c in (r.get("categories_not_run_by_limit") or [])} - ran
        - set(merged["categories_missing"]) - set(merged["categories_inactive_at_release"]))
    shard_prefix = SHARD_REASON.split("{", 1)[0]
    kept: List[str] = []
    no_question_categories: List[str] = []
    for r in results:
        for reason in str(r.get("leaderboard_comparable_reason") or "").split("; "):
            if not reason or reason.startswith(shard_prefix) or reason.startswith("empty shard "):
                continue
            if reason.startswith(_SHARD_ONLY_REASONS[0]):
                continue
            if reason.startswith(_SHARD_ONLY_REASONS[1]):
                no_question_categories += _listed_categories(reason)
                continue
            if reason not in kept:
                kept.append(reason)
    # Recomputed over the union: an instruction_following task count, and a category, that only
    # a shard lacked is not a gap of the merged run.
    union_categories = {t.get("category") for t in traces}
    still_missing = sorted({c for c in no_question_categories if c not in union_categories})
    if still_missing:
        kept.append("categories with no question run (none among the selected questions, "
                    f"or none active at this release): {still_missing}")
    if_counts: Dict[str, int] = {}
    for t in traces:
        if t.get("category") == "instruction_following":
            task = str(t.get("task"))
            if_counts[task] = if_counts.get(task, 0) + 1
    if len(set(if_counts.values())) > 1:
        kept.append(_if_unequal_reason(dict(sorted(if_counts.items()))))
    merged["leaderboard_comparable"] = not kept
    merged["leaderboard_comparable_reason"] = "; ".join(kept) or None
    return merged
