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

# Evaluation Suite: bigcodebench
# Description: BigCodeBench - calibrated pass@1 via the official BCB Docker harness

"""gbench native built-in runner for bigcodebench (Coding & Software Engineering).

Canonical BigCodeBench (bigcode/bigcodebench, v0.1.4) scored by calibrated pass@1, for both
columns the leaderboard publishes: **Instruct** (the task as a natural-language instruction) and
**Complete** (the function signature and docstring to complete). Every task is generated once
per split. BigCodeBench-Hard's 148 tasks are identical rows of the full set, so a full run also
reports both Hard figures; `GBENCH_BIGCODEBENCH_SUBSET=hard` runs only those 148 tasks.

Prompts follow upstream's generation code (bigcodebench@09dd993f `generate.py` +
`provider/utility.py::make_raw_chat_prompt`): the default `instruction_prefix`, then the task
prompt (fenced for Complete). How the reply starts depends on the run, as it does upstream:

* no-think runs follow upstream's default `--backend vllm` path (`prefill=not skip_prefill`),
  which every Gemma row on the leaderboard used (`prefill: true`): the assistant turn is
  prefilled with the default `response_prefix` and an open python fence, sent as the final
  assistant message with `continue_final_message` so the model continues it;
* `--thinking` runs follow upstream's `--backend openai` path, its path for a model behind an
  OpenAI-compatible endpoint (`--base_url`): one user turn, no prefill, because a prefilled
  assistant turn cannot think first.

On a no-think run the server's reasoning parser can move part of a reply into its reasoning field;
base's server-parser policy recovers that text in generation order (from the reply's token ids,
else reasoning then content) and it is what is sanitized and scored, as upstream sanitizes the
whole generation of its in-process vLLM. Such replies are named (`replies_split_by_reasoning_parser`);
one rebuilt as reasoning then content lacks the special tokens upstream's text keeps and makes the
run not comparable. A task the harness returns no result for while it scores others is a harness
error, left out of the score.

The result records `prefill` so it is compared with leaderboard rows carrying the same flag. Each
path samples as upstream's backend does (see Sampling). The model's output is then sanitized as
generate.py does (`sanitize(completion, entry_point)`, i.e. `bigcodebench.sanitize` without
`--calibrate`) and executed against each task's unittest TestCase by the official harness, whose
`evaluate` calibrates each solution (its default `calibrated=True`). SANDBOX_EVAL.

The harness runs in a LOCAL image built from docker/bigcodebench.Dockerfile (upstream's
Evaluate.Dockerfile pinned to bigcode-project/bigcodebench@09dd993f); gbench never pulls a
registry image. The suite hard-errors (infra_required) if Docker or that image is missing. The
evaluate container sees at most MAX_HARNESS_CPUS CPUs (GBENCH_BIGCODEBENCH_CPUS), because the
harness's TensorFlow tasks hang on hosts where it sees many more, and runs no more test workers
than it has CPUs.

Network: the sanitize and evaluate containers run on docker's default bridge network
(HARNESS_NETWORK), the network upstream's documented `docker run` of its evaluate image gives it,
and both load the dataset from Hugging Face at run time. The evaluate container executes the
model's code, so
before generation check_bigcodebench_prerequisites runs container_guard.require_metadata_blocked
on that network: a cloud metadata server reachable there, or huggingface.co not resolving there
(HARNESS_HOSTS), is a hard error (infra_required) naming the host's DOCKER-USER rule. A split whose
harness still produces nothing is a harness error: its samples are not scored and the status is
completed_with_errors (error when no split is scored); when that split is BigCodeBench-Instruct,
the headline, `accuracy` is None (withheld), never 0. The error names what the harness output
shows (_harness_network_failure): a failed host-name lookup (kind `dns`), or the Hugging Face Hub
not reached (kind `hub_unreachable`: datasets' "Couldn't reach '<repo>' on the Hub (...)", which
is all the image prints when the lookup fails, since datasets raises it `from None`, or
huggingface_hub's LocalEntryNotFoundError); otherwise kind `no_results`. The model's code keeps
the container's general internet access, as upstream runs it; only the metadata server is
blocked.

Sampling: per path (SAMPLING; max_tokens resolved by base's token-limit policy). The temperature
is gbench's rule for every suite (base.resolve_temperature): 1.0 with --thinking, 0.0 without
(run_codegen's own is 0.0 on both paths).

* no-think (vllm path): upstream's defaults, run_codegen `temperature=0.0` and `n_samples=1`
  (greedy, the 0.0 gbench sends), with the backend's own request fields: `top_p=1.0` (greedy,
  `do_sample=False`) and `stop=EOS` (provider/utility.py); tabs in the output are replaced with four spaces
  (provider/vllm.py). Every request carries run_codegen's `max_new_tokens=1280`
  (`SamplingParams(max_tokens=...)`); the run's --max-output-tokens is not applied. Only a prompt
  that leaves less room in the served context window gets less, which the result names.
* --thinking (openai path): gen/util/openai_request.py make_request as it runs for a served
  model: run_codegen's `temperature=0.0` (greedy, `n_samples=1`), `top_p=0.95` and
  `max_completion_tokens=1280`, except the temperature: a --thinking run samples at 1.0
  (base.resolve_temperature); top_p 0.95 stays. make_request's reasoning-model branch (a model
  name starting or ending with "o1-", "o3-", "reasoner" or "grok-3-mini-beta": top_p,
  max_completion_tokens and temperature popped, `reasoning_effort` added) keys on hosted vendor model names, so it is not a
  reasoning-model setting for a served model and is not used; BigCodeBench declares no
  model-agnostic one. So base's token-limit policy applies: the 1280 cap was written for
  non-reasoning generation and is not applied; each request is sent the run's budget
  (--max-output-tokens, else base's thinking floor; less only where the prompt leaves less room
  in the served window), and the result records the cap as not applied to reasoning. A reply that
  stopped at that budget (finish_reason "length") is counted in `replies_cut_off_by_budget` (those
  with no answer text, cut off inside the reasoning channel, also in
  `replies_cut_off_in_reasoning`) and makes the run not leaderboard-comparable.

`--temperature`, or `GBENCH_BIGCODEBENCH_TEMPERATURE` for this suite alone (it takes precedence),
overrides the temperature in either mode; the result records it under `official_sampling`, and
`max_output_tokens_note` states the mode's max_tokens choice. Both paths are greedy upstream
(temperature 0.0), so a run is comparable only at temperature 0.0, with the path's request fields
and no other sampling field (decoding penalties included).
"""

import glob
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from .base import (NO_REPLY_HARNESS_KINDS, OFFICIAL_SAMPLING_PARAM_KEYS, SERVER_PARSER_MOVED_TEXT,
                   OfficialSampling, apply_shard, finalize_shard_fields, get_run_knob,
                   run_eval_suite)
from .metrics import budget_cutoffs
from .limit_plan import eval_limit_plan
from .sampling import allocate_limit, stratified_indices, presharded_for_base
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

PILLAR = "Coding & Software Engineering"
DOCS_URL = "docs/evals/bigcodebench.md"
_IMAGE_DEFAULT = "gbench-bigcodebench:09dd993f"
_DATASETS = {"full": "bigcode/bigcodebench", "hard": "bigcode/bigcodebench-hard"}
#: HF dataset commits the prompts are read from. The v0.1.4 split is the one the pinned harness
#: evaluates against (bigcodebench/data/bigcodebench.py BIGCODEBENCH_VERSION).
_REVISIONS = {"full": "b74c0d0bf70d2c0bc459be537895cca163007f1a",
              "hard": "298d2cc7b96612e15e47313c3603ee124cee0c1f"}
_SPLIT = "v0.1.4"
#: The leaderboard's two columns, i.e. bigcodebench's `--split` values.
SPLITS = ("instruct", "complete")
#: Verbatim defaults of `run_codegen(instruction_prefix=..., response_prefix=...)` in
#: bigcodebench/generate.py.
INSTRUCTION_PREFIX = ("Please provide a self-contained Python script that solves the following "
                      "problem in a markdown code block:")
RESPONSE_PREFIX = ("Below is a Python script with a self-contained function that solves the "
                   "problem and passes corresponding tests:")

#: Request fields that make an OpenAI-compatible (vLLM) chat endpoint continue the final
#: assistant message instead of opening a new turn (base forwards both, and counts the prompt
#: with them; see base.prefill_fields).
_PREFILL_REQUEST = {"continue_final_message": True, "add_generation_prompt": False}

#: bigcodebench/provider/utility.py EOS: the stop strings of the vllm backend's SamplingParams.
EOS = ["<|endoftext|>", "<|endofmask|>", "</s>", "\nif __name__", "\ndef main(", "\nprint("]
#: OpenAI-compatible servers (vLLM's included) accept at most 4 stop strings, while the backend
#: runs vLLM in-process with all 6. The first 4 are sent, and apply_eos cuts the reply at the
#: earliest of all 6, which gives the text SamplingParams(stop=EOS) returns (a stop string ends
#: the output and is not part of it); the server only generates a little past a stop it was not
#: sent.
SERVER_STOP = EOS[:4]


def apply_eos(text: str) -> str:
    """`text` cut before the earliest EOS stop string, as vLLM's stop handling returns it."""
    cuts = [i for i in (text.find(s) for s in EOS) if i >= 0]
    return text[:min(cuts)] if cuts else text
_REF = "github.com/bigcode-project/bigcodebench@09dd993f46c3fbf3a799465bb96d524edcb0b199"

#: run_codegen's `max_new_tokens` default: the vllm backend's SamplingParams(max_tokens=...) and
#: the openai backend's max_completion_tokens (gen/util/openai_request.py make_request).
UPSTREAM_MAX_NEW_TOKENS = 1280

#: The model names make_request (and OpenAIChatDecoder.codegen) give their reasoning-model branch:
#: a model name starting or ending with one of these. They name hosted vendor models, so that
#: branch is not a reasoning-model setting for a served model.
VENDOR_REASONING_NAMES = ("o1-", "o3-", "reasoner", "grok-3-mini-beta")

#: The sampling of each generation path: run_codegen's `temperature=0.0` (greedy with
#: n_samples=1), the backend's own request fields and max_new_tokens 1280. Neither declares a
#: reasoning-model setting that applies to a served model, so a --thinking run (openai path)
#: gets base's token-limit policy: the cap is not applied to reasoning.
SAMPLING = {
    True: OfficialSampling(
        reference=(f"{_REF} generate.py run_codegen temperature=0.0 (greedy), "
                   "max_new_tokens=1280; provider/vllm.py SamplingParams(max_tokens=1280, "
                   "top_p=1.0 when not sampling, stop=EOS; the first 4 stop strings are sent, "
                   "the server limit, and all 6 are applied to the reply; its "
                   "skip_special_tokens=False is not sent: the reply is the server's default "
                   "decode, special tokens stripped, see SPECIAL_TOKENS_NOTE)"),
        temperature=0.0, params={"top_p": 1.0, "stop": list(SERVER_STOP)},
        max_tokens=UPSTREAM_MAX_NEW_TOKENS),
    False: OfficialSampling(
        reference=(f"{_REF} generate.py run_codegen temperature=0.0 (greedy), "
                   "max_new_tokens=1280; gen/util/openai_request.py make_request for a served "
                   "model: top_p=0.95, max_completion_tokens=1280 (its branch for model names "
                   "matching 'o1-', 'o3-', 'reasoner' or 'grok-3-mini-beta' keys on hosted "
                   "vendor models and is not used)"),
        temperature=0.0, params={"top_p": 0.95}, max_tokens=UPSTREAM_MAX_NEW_TOKENS),
}

#: The one decoding setting of upstream's in-process vLLM (provider/vllm.py SamplingParams) that is
#: not sent: recorded on every prefill run's result.
SPECIAL_TOKENS_NOTE = (
    "upstream's provider/vllm.py decodes with skip_special_tokens=False; gbench does not send "
    "that field (the served endpoint's reasoning and tool-call parsers read the decoded text), so "
    "a reply is the server's default decode with special tokens stripped: a reply whose text "
    "held a special token differs from upstream's there (a reply the server's reasoning parser "
    "split is rebuilt from its token ids with special tokens kept, see server_parser_events)")

#: Added to a --thinking run's max_output_tokens_note.
_NO_REASONING_SETTING_NOTE = (
    "bigcodebench declares no reasoning-model setting for a served model: make_request's "
    "reasoning-model branch, which sends no max_completion_tokens, is for model names starting or "
    "ending with "
    + ", ".join(repr(n) for n in VENDOR_REASONING_NAMES[:-1])
    + f" or {VENDOR_REASONING_NAMES[-1]!r} (hosted vendor models)")

GENERATION_PATHS = {
    True: ("upstream default `bigcodebench.generate --backend vllm` path (prefill on): the chat "
           "template with the assistant turn prefilled by response_prefix and an open python "
           "fence, which the model continues (continue_final_message)"),
    False: ("upstream `bigcodebench.generate --backend openai` path: one user turn, no response "
            "prefill (a thinking run cannot be prefilled)"),
}
#: The one byte the chat API cannot reproduce on the prefill path, stated with the result.
_PREFILL_NOTE = (
    "upstream renders the prefilled prompt ending in an open ```python fence plus a newline; a "
    "chat template trims that trailing newline from an unfinished assistant turn, so the model "
    "generates the newline itself as its first token (prefill_first_char_newline counts the "
    "completions where it did, i.e. where the token sequence then matches upstream's); for the "
    "same reason a stop string that starts with a newline (\\nprint( ...) could also end a "
    "completion whose very first line matches it, which upstream's cannot")


#: Most CPUs the evaluate container may see (docker --cpuset-cpus). The pinned harness caps the
#: address space of every test process (reliability_guard: RLIMIT_AS 30 GB), and TensorFlow sizes
#: its thread pools by the CPUs it can see: when the container sees more than about 128 CPUs, the
#: TensorFlow tasks run out of address space and hang until the 240 s task limit. Measured with
#: BigCodeBench/418's reference solution: it passes in about 15 s with 104 visible CPUs and times
#: out with 160 or 208.
MAX_HARNESS_CPUS = 64


def _image() -> str:
    return os.environ.get("GBENCH_BIGCODEBENCH_IMAGE", _IMAGE_DEFAULT)


def harness_cpus() -> List[int]:
    """The CPU ids the evaluate container runs on: the first GBENCH_BIGCODEBENCH_CPUS (default
    MAX_HARNESS_CPUS) of the CPUs this process may use."""
    try:
        available = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = list(range(os.cpu_count() or 1))
    raw = (os.environ.get("GBENCH_BIGCODEBENCH_CPUS") or "").strip()
    want = MAX_HARNESS_CPUS
    if raw:
        try:
            want = int(raw)
        except ValueError:
            want = 0
        if want < 1:
            raise ValueError(f"GBENCH_BIGCODEBENCH_CPUS must be a positive integer, got {raw!r}")
    return available[:want]


def _subset(kwargs: Dict[str, Any]) -> str:
    subset = str(kwargs.get("subset") or os.environ.get("GBENCH_BIGCODEBENCH_SUBSET")
                 or "full").strip().lower()
    if subset not in _DATASETS:
        raise ValueError(f"bigcodebench: unknown subset {subset!r} (GBENCH_BIGCODEBENCH_SUBSET "
                         f"must be one of {sorted(_DATASETS)})")
    return subset


def make_raw_chat_prompt(task_prompt: str, split: str = "instruct") -> str:
    """Port of bigcodebench.provider.utility.make_raw_chat_prompt: the user turn for `split`
    (the chat-template step is the server's)."""
    if split == "complete":
        return f"""\
{INSTRUCTION_PREFIX}
```
{task_prompt.strip()}
```
"""
    return f"""\
{INSTRUCTION_PREFIX}
{task_prompt.strip()}
"""


def prefill_message() -> Dict[str, str]:
    """The prefilled assistant turn: make_raw_chat_prompt's `response` up to the splitter."""
    return {"role": "assistant", "content": f"{RESPONSE_PREFIX}\n```python\n"}


#: The docker network of the sanitize and evaluate containers: docker's default bridge, the
#: network upstream's documented `docker run` of its evaluate image gives it (both containers load
#: the dataset from Hugging Face at run time; the evaluate container runs the tasks' tests over the
#: model's code).
HARNESS_NETWORK = "bridge"
#: The host names those containers look up themselves: the Hugging Face Hub, where both load the
#: v0.1.4 split (bigcodebench/data/bigcodebench.py load_dataset). The metadata guard's probe
#: container resolves them before generation, so a network without DNS is a hard error then, not
#: after every reply was generated.
HARNESS_HOSTS = ("huggingface.co",)
#: How the harness reports that it could not reach the Hugging Face Hub, with no resolver text:
#: datasets (2.17.0 in the image) load.py dataset_module_factory raises "Couldn't reach
#: '<repo>' on the Hub (<ConnectionError | ConnectTimeout | OfflineModeIsEnabled>)" `from None`
#: when its dataset_info request cannot connect (a failed lookup included), and datasets' file
#: download raises "Couldn't reach <url> (<error>)"; huggingface_hub raises
#: LocalEntryNotFoundError when a file can be neither downloaded nor found in its cache.
HUB_UNREACHABLE = re.compile(
    r"Couldn't reach \S+(?: on the Hub)? \(|LocalEntryNotFoundError|"
    r"cannot find the requested files in the local cache")


def _harness_network_failure(output: str) -> Optional[Dict[str, str]]:
    """{kind, detail} for a harness pass whose output shows it could not reach its dataset: a
    failed host-name lookup (`dns`, tau_common.network_failures), else the Hugging Face Hub not
    reached (`hub_unreachable`, HUB_UNREACHABLE: the last such line, the exception message, not
    the source line a traceback quotes above it); None otherwise."""
    from .tau_common import network_failures
    dns = network_failures(output, dns_only=True)
    if dns:
        return {"kind": "dns", "detail": dns[0]}
    for line in reversed((output or "").splitlines()):
        if HUB_UNREACHABLE.search(line):
            return {"kind": "hub_unreachable", "detail": " ".join(line.split())[:300]}
    return None


def check_bigcodebench_prerequisites() -> Tuple[bool, str]:
    """datasets + reachable Docker + the locally built BCB evaluate image, and the cloud metadata
    server blocked on HARNESS_NETWORK, where the evaluate container executes the model's code,
    with HARNESS_HOSTS resolving there (container_guard.require_metadata_blocked raises
    infra_required naming the host's DOCKER-USER rule otherwise)."""
    try:
        import datasets  # noqa: F401
    except ImportError:
        return False, "Python package 'datasets' is not installed."
    image = _image()
    build = (f"Build the harness image LOCALLY (gbench never pulls it):\n"
             f"  docker build -t {image} -f docker/bigcodebench.Dockerfile docker\n"
             f"See {DOCS_URL}.")
    if not shutil.which("docker"):
        return False, "Docker CLI is not found on PATH. " + build
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=20)
        if r.returncode != 0:
            return False, "Docker daemon is not reachable. " + build
    except Exception as e:
        return False, f"Cannot connect to Docker daemon: {e}. " + build
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode != 0:
        return False, f"image {image!r} not found. " + build
    from .container_guard import require_metadata_blocked
    require_metadata_blocked("bigcodebench", image, HARNESS_NETWORK, DOCS_URL,
                             resolve=HARNESS_HOSTS)
    return True, ""


def _prefill_supported() -> bool:
    """Whether gbench's request builder forwards the prefill fields to the server.

    Probes base._send_single_request with a stand-in session, so a request builder that drops
    them (and would silently open a new assistant turn after the prefill) is caught before the
    run.
    """
    sent = _probe_request(dict(_PREFILL_REQUEST),
                          [{"role": "user", "content": "x"}, prefill_message()])
    return all(sent.get(k) == v for k, v in _PREFILL_REQUEST.items())


def _probe_request(extra_payload: Dict[str, Any],
                   messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The request body base._send_single_request builds for `extra_payload` and `messages`,
    captured by a stand-in session (nothing is sent anywhere); {} when the builder raises."""
    import asyncio
    from . import base
    sent: Dict[str, Any] = {}

    class _Response:
        status = 200

        async def json(self):
            return {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
                    "usage": {}}

        async def text(self):
            return ""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _Session:
        def post(self, url, json=None, **kw):
            sent.update(json or {})
            return _Response()

    async def _probe():
        await base._send_single_request(
            session=_Session(), api_url="http://probe/chat/completions", model_name="probe",
            messages=messages, extra_payload=dict(extra_payload),
            semaphore=asyncio.Semaphore(1), pbar=None, max_output_tokens=1)
    try:
        asyncio.run(_probe())
    except Exception:                                                   # noqa: BLE001
        return {}
    return sent


def _load_bigcodebench_samples(
    limit: Optional[int] = None,
    subset: str = "full",
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load BigCodeBench (both splits per task); raises on load/schema failure."""
    dataset_id = _DATASETS[subset]
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_id, split=_SPLIT, revision=_REVISIONS[subset])
        rows = list(ds)
        if subset == "full":
            # Hard's rows are identical rows of the full set; read its ids to report both.
            hard_ids = {r["task_id"] for r in load_dataset(
                _DATASETS["hard"], split=_SPLIT, revision=_REVISIONS["hard"])}
        else:
            hard_ids = {r.get("task_id") for r in rows}
    except Exception as e:
        logger.error(f"Failed to load dataset for bigcodebench: {e}")
        raise RuntimeError(f"Could not load dataset for bigcodebench: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for bigcodebench returned empty rows")
    # --shard first, over the full ordered set, then the --eval-limit draw within the shard:
    # N samples for the whole suite, spread over the two splits (sampling.allocate_limit), never
    # N tasks x both splits. Each split draws its tasks from the same seeded order (a seeded
    # subset, not a contiguous head), so the split with fewer tasks runs a subset of the other's.
    rows, _ = apply_shard(rows, "bigcodebench")
    keep: Dict[str, Optional[set]] = {split: None for split in SPLITS}
    if limit is not None and limit > 0:
        # Instruct is the headline split: it gets the first sample of a limited run
        counts = allocate_limit(limit, SPLITS, available={sp: len(rows) for sp in SPLITS},
                                seed="bigcodebench", first=("instruct",))
        keep = {split: (set(stratified_indices([None] * len(rows), counts[split],
                                               "bigcodebench")) if counts[split] else set())
                for split in SPLITS}
        rows_kept = sorted(set().union(*keep.values()))
        position = {i: j for j, i in enumerate(rows_kept)}
        keep = {split: {position[i] for i in chosen} for split, chosen in keep.items()}
        rows = [rows[i] for i in rows_kept]

    samples = []
    for index, item in enumerate(rows):
        task_id = item.get("task_id")
        entry_point = item.get("entry_point")
        prompts = {"instruct": item.get("instruct_prompt"),
                   "complete": item.get("complete_prompt")}
        if not task_id or not entry_point or not all(prompts.values()):
            raise RuntimeError(
                "bigcodebench: unexpected dataset schema "
                "(task_id/instruct_prompt/complete_prompt/entry_point); refusing to fabricate "
                "sample data"
            )
        for split in SPLITS:
            if keep[split] is not None and index not in keep[split]:
                continue
            messages = [{"role": "user",
                         "content": make_raw_chat_prompt(str(prompts[split]), split)}]
            # max_tokens: SAMPLING's, resolved per mode by base's token-limit policy.
            samples.append((messages, task_id,
                            {"category": split, "task_id": task_id, "split": split,
                             "hard": task_id in hard_ids}))

    logger.info(f"Loaded {len(samples)} bigcodebench samples ('{dataset_id}' {_SPLIT}, "
                f"{len(rows)} tasks, {len(SPLITS)} splits).")
    return samples


def _with_prefill(samples: List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]):
    """Add the prefilled assistant turn and the fields that make the server continue it."""
    return [(list(messages) + [prefill_message()], gold, {**meta, **_PREFILL_REQUEST})
            for messages, gold, meta in samples]


def _parse_bcb_results(workdir: str) -> Tuple[Dict[str, bool], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Parse the harness eval_results.json (per-task pass + per-test detail) + pass_at_k.json."""
    results: Dict[str, bool] = {}
    details: Dict[str, Dict[str, Any]] = {}
    metrics: Dict[str, Any] = {}
    for p in glob.glob(os.path.join(workdir, "**", "*_eval_results.json"), recursive=True):
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        for tid, entries in (data.get("eval") or {}).items():
            if isinstance(entries, list) and entries:
                e0 = entries[0]
                results[tid] = e0.get("status") == "pass"
                # Trajectory completeness: keep the harness's own status + per-test detail so a
                # failure is readable (which assertion / traceback) instead of a bare boolean.
                det = e0.get("details")
                details[tid] = {
                    "status": e0.get("status"),
                    "details": det if isinstance(det, (dict, str)) else None,
                }
    for p in glob.glob(os.path.join(workdir, "**", "*_pass_at_k.json"), recursive=True):
        try:
            with open(p, encoding="utf-8") as f:
                pk = json.load(f)
            if "pass@1" in pk:
                metrics["bigcodebench_pass_at_1"] = pk["pass@1"]
        except Exception:
            pass
    return results, details, metrics


def _trace_split(trace: Dict[str, Any]) -> str:
    return (trace.get("extra_payload") or {}).get("split") or "instruct"


def _solution(trace: Dict[str, Any]) -> str:
    """The raw solution as upstream's backend returns it: the vllm (prefill) path replaces every
    tab with four spaces (provider/vllm.py codegen), the openai path keeps the message content.
    A prefilled sample carries `continue_final_message` in its meta."""
    text = trace.get("response_text") or ""
    prefilled = (trace.get("extra_payload") or {}).get("continue_final_message") is True
    return apply_eos(text).replace("\t", "    ") if prefilled else text


def _make_scorer(subset: str, parallel: int, metrics: Dict[str, Any], image: str):
    async def _score(sample_traces: List[Dict[str, Any]]) -> None:
        # One harness pass per split: `bigcodebench.evaluate --split` names the column the
        # samples belong to.
        by_split: Dict[str, List[Dict[str, Any]]] = {}
        for tr in sample_traces:
            by_split.setdefault(_trace_split(tr), []).append(tr)
        for split in [s for s in SPLITS if s in by_split] + sorted(set(by_split) - set(SPLITS)):
            await _score_split(split, by_split[split])

    async def _score_split(split: str, sample_traces: List[Dict[str, Any]]) -> None:
        workdir = tempfile.mkdtemp(prefix="gbench_bcb_")
        try:
            await _score_split_in(workdir, split, sample_traces)
        finally:
            # The harness writes only files here (its HF cache stays in the container, see
            # workdir_env), so the directory can be removed even where uid 1000 wrote them.
            shutil.rmtree(workdir, ignore_errors=True)
            if os.path.exists(workdir):
                logger.warning("bigcodebench: could not remove the work directory %s", workdir)

    async def _score_split_in(workdir: str, split: str,
                              sample_traces: List[Dict[str, Any]]) -> None:
        import asyncio
        # `mkdtemp` creates the directory 0700 owned by the HOST uid, but the image runs as
        # `bigcodebenchuser` (uid 1000). Bind-mounted at /app that uid cannot traverse it, so
        # every relative path resolution from the container's CWD raises PermissionError -
        # including the `Path("bigcode/bigcodebench/state.json").exists()` probe that
        # `datasets.load_dataset` does before it falls back to the Hub. That surfaced as
        # "bigcodebench: no eval results parsed" with the whole suite scoring 0.
        # Reproduced directly: a 0700 mount raises, the same mount at 0777 resolves and is
        # writable. The harness must also write its results back here, hence 0777 not 0755.
        os.chmod(workdir, 0o777)
        samples_path = os.path.join(workdir, "samples.jsonl")
        with open(samples_path, "w", encoding="utf-8") as f:
            for tr in sample_traces:
                tid = (tr.get("extra_payload") or {}).get("task_id")
                if tid:
                    # The harness asserts each row carries "completion" or "solution"
                    # (bigcodebench/data/utils.py::load_solutions). Writing "raw_solution"
                    # got past the sanitizer's file read and then died with
                    # "AssertionError: No completion or solution found in sample!", which
                    # surfaced as "harness produced no parsed results" and a 0/N score.
                    # On the prefill path the response is the continuation after the open
                    # fence, which is what upstream's vllm backend sanitizes too.
                    f.write(json.dumps({"task_id": tid,
                                        "solution": _solution(tr)}) + "\n")
        os.chmod(samples_path, 0o644)   # a 077 umask would leave this unreadable to uid 1000

        mount = f"{workdir}:/app:rw"
        # The container's default CWD is not writable/traversable, and the harness resolves
        # its dataset through a RELATIVE path first
        # (`Path('bigcode/bigcodebench/state.json').exists()` in
        # bigcodebench/data/bigcodebench.py:29). That stat raised
        # `PermissionError: [Errno 13] ... 'bigcode/bigcodebench/state.json'` before any
        # evaluation ran, so the whole suite reported "harness produced no parsed results".
        # Run in the writable mount. The HF caches go to the container's own /tmp: in the mount
        # they became directories owned by uid 1000 that the host user cannot remove.
        workdir_env = ["-w", "/app",
                       "-e", "HF_HOME=/tmp/.hf",
                       "-e", "HF_DATASETS_CACHE=/tmp/.hf/datasets"]
        # generate.py sanitizes each completion with sanitize(completion, entry_point); the
        # script without --calibrate does exactly that (its --calibrate rewrite is not part of
        # generate.py). evaluate then calibrates (calibrated=True by default).
        sanitize = ["docker", "run", "--rm", "--network", HARNESS_NETWORK, *workdir_env,
                    "--entrypoint", "bigcodebench.sanitize",
                    "-v", mount, image, "--samples", "/app/samples.jsonl"]
        # The evaluator asserts the samples file covers the WHOLE split:
        #   "Missing problems in samples. Expected 1140 problems, got 20"
        # so any --eval-limit run failed outright and reported 0/N. `--selective_evaluate`
        # takes a comma-separated task-id list and restricts the run to it, which is what a
        # subset run needs.
        task_ids = [(tr.get("extra_payload") or {}).get("task_id") for tr in sample_traces]
        selective = ",".join(t for t in task_ids if t)
        # At most MAX_HARNESS_CPUS visible CPUs (see there), and no more test workers than
        # CPUs (upstream's own default is half the CPUs).
        cpus = harness_cpus()
        workers = min(max(1, parallel), len(cpus))
        metrics["harness_cpus"] = len(cpus)
        metrics["harness_parallel"] = workers
        evaluate = ["docker", "run", "--rm", "--network", HARNESS_NETWORK, "-m", "16g",
                    "--cpuset-cpus", ",".join(str(c) for c in cpus),
                    *workdir_env, "-v", mount, image,
                    "--execution", "local", "--subset", subset, "--split", split,
                    "--samples", "/app/samples-sanitized.jsonl",
                    # Fire parses `--pass-k 1` as an int, and evaluate() iterates it:
                    #   TypeError: 'int' object is not iterable
                    # A trailing comma makes it the 1-tuple the signature expects.
                    "--no-gt", "--pass-k", "1,", "--parallel", str(workers)]
        if selective:
            evaluate += ["--selective_evaluate", selective]

        def _run():
            s = subprocess.run(sanitize, capture_output=True, text=True)
            if s.returncode != 0:
                return s
            return subprocess.run(evaluate, capture_output=True, text=True)
        proc = await asyncio.to_thread(_run)

        results, details, m = _parse_bcb_results(workdir)
        # Tasks that are harness errors, never the model's outcome: a request base marked
        # `harness_error` (no reply from the served endpoint; its solution is empty), and a task
        # the harness returned no result for while it scored others (`missed`, marked below).
        no_reply = sum(1 for tr in sample_traces if tr.get("harness_error"))
        missed = ([tr for tr in sample_traces if not tr.get("harness_error")
                   and (tr.get("extra_payload") or {}).get("task_id") not in results]
                  if results else [])
        # The harness's own pass@1 is over the tasks in its results: it counts a no-reply task
        # (empty solution) as failed and leaves out a task it returned nothing for, so it is not
        # over the tasks the headline and `splits` score. That figure is withheld for such a split.
        if "bigcodebench_pass_at_1" in m:
            own = None if (no_reply or missed) else m["bigcodebench_pass_at_1"]
            metrics.setdefault("bigcodebench_pass_at_1_by_split", {})[split] = own
            if split == "instruct":
                metrics["bigcodebench_pass_at_1"] = own
            why = []
            if no_reply:
                why.append(f"it counts the {no_reply} task(s) whose request got no reply from "
                           "the served endpoint as failed")
            if missed:
                why.append(f"it returned no result for {len(missed)} task(s)")
            if why:
                metrics.setdefault("bigcodebench_pass_at_1_withheld", {})[split] = (
                    "the harness's own pass@1 is not over the scored tasks: " + " and ".join(why)
                    + "; accuracy and splits leave those tasks out")
        if not results:
            # The harness ran but produced nothing. Reporting every sample as simply
            # "incorrect" would surface as a clean success + 0%, which is indistinguishable
            # from a model that genuinely solved nothing. The split's samples are harness errors:
            # none is scored, and a failed host-name lookup (the dataset download) is named.
            logger.error("bigcodebench: no eval results parsed for the %s split. stderr tail: %s",
                         split, (proc.stderr or "")[-800:])
            net = _harness_network_failure(f"{proc.stdout or ''}\n{proc.stderr or ''}")
            error = (f"harness produced no parsed results ({split} split)"
                     if split != "instruct" else "harness produced no parsed results")
            if net and net["kind"] == "dns":
                error += (f": its containers could not resolve a host name ({net['detail']}); "
                          "they load the dataset from Hugging Face and need working DNS on the "
                          "docker bridge network")
            elif net:
                error += (f": its containers could not reach the Hugging Face Hub "
                          f"({net['detail']}); they load the dataset from Hugging Face and need "
                          "working DNS and internet access on the docker bridge network")
            report = metrics.get("bigcodebench_report")
            metrics["bigcodebench_report"] = {
                "error": error if not report else f"{report['error']}; {error}",
                "stderr_tail": (proc.stderr or "")[-400:],
            }
            metrics.setdefault("bigcodebench_harness_errors", {})[split] = {
                "kind": net["kind"] if net else "no_results", "samples": len(sample_traces),
                **({"detail": net["detail"]} if net else {})}
            for tr in sample_traces:
                tr["is_correct"] = False
                tr["status"] = "HARNESS_ERROR"
                tr["scoring_excluded"] = True
                tr["scoring_note"] = error
            return
        for tr in sample_traces:
            tid = (tr.get("extra_payload") or {}).get("task_id")
            tr["is_correct"] = bool(results.get(tid, False))
            tr["status"] = "OK"
            if tid in details:
                tr["execution"] = details[tid]
        for tr in missed:
            # The harness scored other tasks but not this one: a harness error, not the model's
            # outcome. Left out of the score (base: completed_with_errors), counted and named.
            tid = (tr.get("extra_payload") or {}).get("task_id")
            tr["is_correct"] = False
            tr["status"] = "HARNESS_ERROR"
            tr["scoring_excluded"] = True
            tr["harness_error"] = {"kind": "no_harness_result",
                                   "detail": f"the {split} harness pass returned no result for "
                                             f"{tid}"}
            tr["scoring_note"] = "no harness result for this task"
        if missed:
            metrics["bigcodebench_unscored"] = metrics.get("bigcodebench_unscored", 0) + len(missed)
    return _score


def _split_scores(traces: List[Dict[str, Any]], subset: str) -> Dict[str, Dict[str, Any]]:
    """Calibrated pass@1 per split, over the run's tasks and over BigCodeBench-Hard's."""
    out: Dict[str, Dict[str, Any]] = {}
    for split in SPLITS:
        # a split whose harness produced nothing is not scored (its samples are harness errors)
        trs = [t for t in traces if _trace_split(t) == split and not t.get("scoring_excluded")]
        if not trs:
            continue
        hard = [t for t in trs if (t.get("extra_payload") or {}).get("hard")]
        entry: Dict[str, Any] = {}
        for name, group in ((subset, trs), ("hard", hard)):
            if group and name not in entry:
                ok = sum(1 for t in group if t.get("is_correct"))
                entry[name] = {"correct": ok, "total": len(group),
                               "pass_at_1": round(100.0 * ok / len(group), 2)}
        out[split] = entry
    return out


def run_bigcodebench(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run BigCodeBench calibrated pass@1, Instruct and Complete (hard-errors if Docker or the
    local image is missing)."""
    subset = _subset(kwargs)
    ok, reason = check_bigcodebench_prerequisites()
    if not ok:
        raise infra_required("bigcodebench", reason, DOCS_URL)
    harness_cpus()      # a malformed GBENCH_BIGCODEBENCH_CPUS stops the run before generation
    prefill = not enable_thinking
    if prefill and not _prefill_supported():
        raise infra_required(
            "bigcodebench",
            "gbench's request builder does not forward `continue_final_message` / "
            "`add_generation_prompt` to the server, so the prefilled assistant turn of upstream's "
            "default path cannot be continued (base._send_single_request must pass both fields "
            "through)", DOCS_URL)
    limit = kwargs.get("limit")
    samples = _load_bigcodebench_samples(limit=limit, subset=subset)
    per_split = {split: sum(1 for _m, _g, meta in samples if meta.get("split") == split)
                 for split in SPLITS}
    limit_plan = eval_limit_plan("bigcodebench", limit, len(samples), per_split,
                                 setting_noun="split", attempts=1)
    if prefill:
        samples = _with_prefill(samples)
    # The settings of this run's path. Neither declares a reasoning-model setting for a served
    # model, so a --thinking run sends the path's sampling with base's token-limit policy.
    mode = SAMPLING[prefill]
    user_payload = dict(kwargs.get("extra_payload") or {})
    extra_payload = kwargs.get("extra_payload")
    metrics: Dict[str, Any] = {}
    result = run_eval_suite(
        eval_name="bigcodebench",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=presharded_for_base(samples),  # this shard's tasks, kept as is
        async_eval_fn=_make_scorer(subset, concurrency, metrics, _image()),
        declared_scoring_mode="execution",  # sandboxed test execution, not an LLM judge
        thinking=enable_thinking,
        extra_payload=extra_payload,
        # already drawn by the loader (N samples over the two splits): base must not draw again
        limit=None,
        max_output_tokens=kwargs.get("max_output_tokens", 4096),
        official_sampling=SAMPLING[prefill],
    )
    result.update(metrics)
    result["eval_limit_plan"] = limit_plan
    traces = result.get("sample_traces", []) or []
    splits = _split_scores(traces, subset)
    result["splits"] = splits
    if isinstance(metrics.get("bigcodebench_report"), dict) and metrics["bigcodebench_report"].get("error"):
        # A split whose harness produced nothing is a harness error: with another split scored
        # the run completed with errors, with none it is an error and reports no score.
        result["status"] = "completed_with_errors" if splits else "error"
        result["error"] = metrics["bigcodebench_report"]["error"]
    if "instruct" in splits and subset in splits["instruct"]:
        # The headline stays BigCodeBench-Instruct; Complete (and Hard) are reported beside it.
        head = splits["instruct"][subset]
        result["blended_splits_accuracy"] = result.get("accuracy")
        result["accuracy"] = head["pass_at_1"]
        result["correct_answers"] = head["correct"]
        result["total_questions"] = head["total"]
    elif metrics.get("bigcodebench_harness_errors", {}).get("instruct"):
        # The headline split's harness produced nothing: no headline (not a 0%).
        result["blended_splits_accuracy"] = result.get("accuracy")
        result["accuracy"] = None
        result["correct_answers"] = 0
        result["total_questions"] = 0
        result["accuracy_withheld_reason"] = (
            "BigCodeBench-Instruct was not scored: " + metrics["bigcodebench_report"]["error"])
    elif any(_trace_split(t) == "instruct" for t in traces):
        # Every Instruct task was left out of the score (harness errors: its request got no reply
        # from the served endpoint, or the harness returned no result for it): no headline,
        # never the Complete figure and never a 0%.
        kinds: Dict[str, int] = {}
        for t in traces:
            if _trace_split(t) == "instruct":
                kind = (t.get("harness_error") or {}).get("kind") or "not scored"
                kinds[kind] = kinds.get(kind, 0) + 1
        if set(kinds) <= NO_REPLY_HARNESS_KINDS:
            why = ("BigCodeBench-Instruct was not scored: every Instruct request got no reply from "
                   "the served endpoint (harness errors)")
        else:
            why = ("BigCodeBench-Instruct was not scored: every Instruct task is a harness error ("
                   + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())) + ")")
        result["blended_splits_accuracy"] = result.get("accuracy")
        result["accuracy"] = None
        result["correct_answers"] = 0
        result["total_questions"] = 0
        result["accuracy_withheld_reason"] = (
            f"{result['accuracy_withheld_reason']}; {why}"
            if result.get("accuracy_withheld_reason") else why)
    # max_output_tokens_note: base's, which states the mode's max_tokens choice; a --thinking run
    # also says why no reasoning-model cap applies.
    if enable_thinking:
        result["max_output_tokens_note"] = "; ".join(filter(None, (
            result.get("max_output_tokens_note"), _NO_REASONING_SETTING_NOTE)))
    result["subset"] = subset
    result["dataset_revision"] = f"{_DATASETS[subset]}@{_REVISIONS[subset]} ({_SPLIT})"
    result["metric"] = ("calibrated pass@1, BigCodeBench-Instruct"
                        + (" (Hard)" if subset == "hard" else "")
                        + " headline; Complete and, for a full run, both Hard figures in "
                        "`splits`: upstream chat prompt, generate.py's sanitize, calibrated "
                        "unittest execution in the pinned harness image")
    result["prefill"] = prefill
    result["generation_path"] = GENERATION_PATHS[prefill]
    if prefill:
        result["special_tokens_note"] = SPECIAL_TOKENS_NOTE
    result["sanitize"] = ("bigcodebench.sanitize (generate.py's sanitize(completion, "
                          "entry_point)), then evaluate --calibrated (default)")
    if prefill:
        answered = [t for t in traces if t.get("response_text")]
        result["prefill_first_char_newline"] = {
            "yes": sum(1 for t in answered if str(t["response_text"]).startswith("\n")),
            "of": len(answered)}
        result["prefill_note"] = _PREFILL_NOTE
    # The leaderboard is greedy pass@1 over the whole split.
    temp = result.get("temperature")
    noncanon: List[str] = []
    if result.get("status") == "error" or metrics.get("bigcodebench_harness_errors"):
        noncanon.append(f"harness error: {result.get('error') or 'unknown'}")
    if metrics.get("bigcodebench_unscored"):
        noncanon.append(f"{metrics['bigcodebench_unscored']} task(s) got no harness result "
                        "(harness errors, left out of the score; bigcodebench_unscored)")
    no_reply = sum(1 for t in traces
                   if (t.get("harness_error") or {}).get("kind") in NO_REPLY_HARNESS_KINDS)
    if no_reply:
        noncanon.append(f"{no_reply} request(s) got no reply from the served endpoint (harness "
                        "errors, left out of the score; request_harness_errors)")
    other = sum(1 for t in traces if t.get("harness_error")
                and t["harness_error"].get("kind") not in NO_REPLY_HARNESS_KINDS
                and t["harness_error"].get("kind") != "no_harness_result")
    if other:
        noncanon.append(f"{other} repl(ies) are other harness errors (left out of the score; "
                        "see each trace's harness_error and status_reason)")
    # Text the server's reasoning parser moved out of a no-think reply into its reasoning field
    # is the model's output: base's server-parser policy recovered it in generation order and it
    # is what was sanitized and scored, as upstream sanitizes the whole generation of its
    # in-process vLLM (provider/vllm.py, skip_special_tokens=False). Rebuilt from the reply's
    # token ids it is that text; rebuilt as reasoning then content (the server returned no token
    # ids) it lacks the special tokens between the two, so it is not upstream's text.
    moved = [t for t in traces
             if (t.get("server_parser") or {}).get("event") == SERVER_PARSER_MOVED_TEXT]
    if moved:
        sources: Dict[str, int] = {}
        for t in moved:
            src = t["server_parser"].get("recovered_from") or "unknown"
            sources[src] = sources.get(src, 0) + 1
        result["replies_split_by_reasoning_parser"] = {"count": len(moved),
                                                       "recovered_from": sources}
        partial = len(moved) - sources.get("token_ids", 0)
        if partial:
            noncanon.append(f"{partial} no-think repl(ies) the server's reasoning parser split "
                            "were rebuilt as reasoning then content, without the special tokens "
                            "between them that upstream's generation text keeps (the server "
                            "returned no token ids)")
    if limit:
        noncanon.append(f"subset run (--eval-limit {limit})")
    if temp is None or float(temp) != 0.0:
        noncanon.append(f"temperature={temp} ({result.get('temperature_source')}; the "
                        "leaderboard is greedy, temperature 0.0)")
    # The request's sampling fields (decoding penalties are merged in last, as base sends them)
    # against the mode's: every field it sends, and nothing else.
    official = dict(mode.params or {})
    sent = {**official, **user_payload, **(result.get("decoding_penalties") or {})}
    changed = sorted(k for k in set(sent) | set(official)
                     if k in OFFICIAL_SAMPLING_PARAM_KEYS and sent.get(k) != official.get(k))
    if changed:
        noncanon.append("request fields " + ", ".join(f"{k}={sent.get(k, 'not sent')!r}"
                                                      for k in changed)
                        + " differ from upstream's backend defaults ("
                        + ", ".join(f"{k}={official.get(k, 'not sent')!r}" for k in changed)
                        + ")")
    if "reasoning_effort" in user_payload:
        noncanon.append(f"reasoning_effort={user_payload['reasoning_effort']!r} (make_request sends "
                        "none for a served model; only its reasoning-model branch for vendor "
                        "model names does)")
    budgets = result.get("max_tokens_sent") or []
    if not enable_thinking and list(budgets) != [UPSTREAM_MAX_NEW_TOKENS]:
        noncanon.append(f"max_tokens={budgets or 'not sent'} (run_codegen's max_new_tokens is "
                        f"{UPSTREAM_MAX_NEW_TOKENS}; a request whose prompt left less room in the "
                        "context window was sent less)")
    # A reply that stopped at a limit other than the official cap (run_codegen's 1280) measures
    # that limit, not the model: with --thinking, the run's budget (the 1280 is not applied to
    # reasoning), or less where the prompt left less room in the window; without it, a request
    # whose prompt left less room in the window.
    cut, in_reasoning = budget_cutoffs(result)
    result["replies_cut_off_by_budget"] = cut
    result["replies_cut_off_in_reasoning"] = in_reasoning
    if cut and enable_thinking:
        noncanon.append(f"{cut} repl(ies) stopped at this run's max_tokens ("
                        + ", ".join(str(b) for b in budgets) + f"), {in_reasoning} of them inside "
                        "the reasoning channel with no answer (run_codegen's "
                        f"{UPSTREAM_MAX_NEW_TOKENS} is not applied to reasoning; raise "
                        "--max-output-tokens, or serve a larger --max-model-len where the context "
                        "window left less room)")
    elif cut:
        window = get_run_knob("max_model_len")
        served = f"the served {window}-token context window" if window else \
            "the served context window"
        noncanon.append(f"{cut} repl(ies) stopped at a max_tokens other than run_codegen's "
                        f"{UPSTREAM_MAX_NEW_TOKENS} ({in_reasoning} of them with no answer): "
                        f"{served} left less room")
    if prefill:
        nl = result["prefill_first_char_newline"]
        if nl["yes"] != nl["of"]:
            noncanon.append(f"{nl['of'] - nl['yes']} prefilled completion(s) did not start with "
                            "the newline upstream's prompt ends with, so their prompt differs")
    result["leaderboard_comparable"] = not noncanon
    if noncanon:
        result["leaderboard_comparable_reason"] = "; ".join(noncanon)
    else:
        result["leaderboard_comparable_note"] = (
            f"compare with leaderboard rows whose prefill flag is {str(prefill).lower()} "
            f"(bigcode/bigcodebench-results `prefill`; every Gemma row there is true)")
    # One shard of a --shard I/N run is a partial set: the shard reason (finalize_shard_fields).
    return finalize_shard_fields(result)
