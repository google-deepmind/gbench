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

"""MathArena's final-answer protocol, shared by the `hmmt` and `aime` suites.

MathArena (github.com/eth-sri/matharena, pinned below) runs a final-answer competition like this,
and so does gbench:

* Prompt: the competition config's `instruction`, then a blank line, then the problem, as one
  user message (runner.py `prepare_run` + solvers/pure_model_solver.py `build_query`).
* Runs: `--n` runs per problem, default 4; the reported score is the mean accuracy over all runs.
* Last chance: when the final assistant message has no parseable `\\boxed{}` answer
  (`extract_answer(content, True)` returns None), the conversation is sent back once with
  MathArena's last-chance prompt, and the reply to it is what gets graded
  (runner.py `process_solver_responses` + `PureModelSolver.last_chance`).
* Grading: `grader.extract_and_grade` on the final assistant message, with the competition
  config (both competitions set `strict_parsing: false`).
* Sampling: each model runs with the settings of its configs/models/<provider>/<model>.yaml,
  which carry the provider's recommended sampling (for example qwen/qwen3.5_27b.yaml:
  temperature 1.0, top_p 0.95, top_k 20). MathArena ships no Gemma config; Gemma 4's recommended
  sampling is its generation_config.json (temperature 1.0, top_p 0.95, top_k 64), so that is
  SAMPLING below, sent with the first request and with the last-chance request alike. The
  temperature is gbench's rule for every suite whatever the config says
  (base.resolve_temperature): 1.0 with --thinking, 0.0 without; the other fields stay the
  config's.

The grader is MathArena's own code, vendored byte-for-byte in `matharena_src/` (parser.py,
parse_manual.py, grader.py, utils.py and the two tools modules grader.py imports; git blob
shas below, checked before every use). It runs in the local `gbench-matharena:b89f2f0a` image
(docker/matharena.Dockerfile) because MathArena pins `antlr4-python3-runtime==4.11`: sympy's
LaTeX parser, which the parser falls back to for answers sympify cannot read, refuses any other
antlr runtime, and the gbench environment carries 4.13 for math_verify. Without it every
LaTeX answer that needs the fallback (for example `(3+\\sqrt{6})^{-1/3}`, an HMMT February
2026 gold answer) would parse as None and grade as wrong.
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
from collections import UserDict
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from .base import OfficialSampling

logger = logging.getLogger(__name__)

MATHARENA_REPO = "github.com/eth-sri/matharena"
#: Commit every vendored file and competition config below comes from (main HEAD on
#: 2026-09-25).
MATHARENA_COMMIT = "b89f2f0ad64ced464d2944f08c3c0aaeaa0df64b"

SRC_DIR = Path(__file__).resolve().parents[2] / "third_party" / "matharena_src"

#: Vendored path -> git blob sha at MATHARENA_COMMIT (the path under src/ or configs/ upstream).
VENDORED_BLOBS: Dict[str, str] = {
    "LICENSE": "0ea1b32de980a7c8b77e0a37e2ea72ff7e4fc863",
    "matharena/__init__.py": "afc1f8e166a2680fbe4e102f22306895ee4e683d",
    "matharena/grader.py": "5da905aadc2d8e61596070eca78008857a84ec7e",
    "matharena/parse_manual.py": "a6fec4a06f6c448b63258296ad3c90c9fb249d05",
    "matharena/parser.py": "fc7284aa6aa621dc6e6d2b4aaac5ed3e00a062b0",
    "matharena/utils.py": "d3d0d771d0991377688f04362d859f7824f1102d",
    "matharena/tools/lean_execution.py": "5681a99179df1fa3640d6410a7480045e8661e5f",
    "matharena/tools/submit_answer.py": "147cd580181f56b8d3c11281a57897a2e4002a6a",
    "configs/competitions/aime/aime_2026.yaml": "477c34f71173e0df66da83283e3bbe2035936c66",
    "configs/competitions/hmmt/hmmt_feb_2026.yaml": "b6fb8081c862c5d1eff7753d61c7fb11da451033",
}

#: runner.py / scripts/run.py `--n` default: runs per problem.
OFFICIAL_RUNS = 4
#: The suites that run MathArena's protocol, and the runs per problem each always makes
#: (max(OFFICIAL_RUNS, --attempt-count)): the evals scaffold and each result's run_shape record
#: the attempts these suites actually run.
SUITE_MIN_ATTEMPTS: Dict[str, int] = {"aime": OFFICIAL_RUNS, "hmmt": OFFICIAL_RUNS}

#: The sampling MathArena gives a model: its provider's recommended settings, as every
#: configs/models/*.yaml at MATHARENA_COMMIT carries them. For Gemma 4 those are the values of
#: its generation_config.json.
SAMPLING = OfficialSampling(
    reference=(f"{MATHARENA_REPO}@{MATHARENA_COMMIT[:8]} configs/models/<provider>/<model>.yaml "
               "(the provider's recommended sampling for each model; e.g. qwen/qwen3.5_27b.yaml "
               "sets temperature 1.0, top_p 0.95, top_k 20); for Gemma 4, its "
               "generation_config.json: temperature 1.0, top_p 0.95, top_k 64"),
    temperature=1.0,
    params={"top_p": 0.95, "top_k": 64})


def request_payload(extra_payload: Optional[Dict[str, Any]],
                    recommended: Optional[OfficialSampling] = None) -> Dict[str, Any]:
    """The request fields of every MathArena request: the recommended sampling's params
    (`recommended`, the served model's own settings; default SAMPLING, Gemma 4's), then the run's
    own extra_payload (which wins, and makes the run non-comparable if it changes one of them)."""
    rec = SAMPLING if recommended is None else recommended
    return {**dict(rec.params or {}), **dict(extra_payload or {})}

IMAGE_DEFAULT = "gbench-matharena:b89f2f0a"
DOCKERFILE = "docker/matharena.Dockerfile"

#: Per-item grading limit inside the image. MathArena sets none; a pathological answer that
#: keeps sympy busy longer than this is recorded as a grading error (a harness error, left out
#: of the score, never a wrong answer) and makes the run non-comparable.
ITEM_TIMEOUT_S = 300

_RESULT_MARKER = "GBENCH_MATHARENA_RESULT "


class Competition(NamedTuple):
    """One MathArena final-answer competition, pinned."""
    key: str          # MathArena config path, e.g. "hmmt/hmmt_feb_2026"
    name: str         # website name, e.g. "HMMT February 2026"
    dataset: str      # HF dataset id (the config's dataset_path)
    revision: str     # HF dataset commit


COMPETITIONS: Dict[str, Competition] = {
    "hmmt/hmmt_feb_2026": Competition(
        "hmmt/hmmt_feb_2026", "HMMT February 2026", "MathArena/hmmt_feb_2026",
        "02fba4f74d8e68e73e66a02d540fd979c05c274c"),
    "aime/aime_2026": Competition(
        "aime/aime_2026", "AIME 2026", "MathArena/aime_2026",
        "d2de22f3c656b4f56cf8981212186377d1e23bc3"),
}


# --- vendored sources ---------------------------------------------------------------------

def _git_blob_sha(data: bytes) -> str:
    import hashlib
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def vendored_files() -> Dict[str, str]:
    """The vendored MathArena files, each checked against its upstream git blob sha."""
    out: Dict[str, str] = {}
    for rel, blob in VENDORED_BLOBS.items():
        path = SRC_DIR / rel
        try:
            data = path.read_bytes()
        except OSError as e:
            raise RuntimeError(f"matharena: vendored file {rel} is missing ({e}); reinstall "
                               f"gbench") from e
        got = _git_blob_sha(data)
        if got != blob:
            raise RuntimeError(f"matharena: vendored file {rel} is not MathArena's (git blob "
                               f"{got}, expected {blob} at {MATHARENA_COMMIT[:8]})")
        out[rel] = data.decode("utf-8")
    return out


def competition_config(comp: Competition) -> Dict[str, Any]:
    """The competition's MathArena config (configs/competitions/<key>.yaml), parsed as the
    runner parses it (yaml.safe_load)."""
    import yaml
    rel = f"configs/competitions/{comp.key}.yaml"
    text = vendored_files()[rel]
    config = yaml.safe_load(text)
    if config.get("dataset_path") != comp.dataset:
        raise RuntimeError(f"matharena: {rel} names {config.get('dataset_path')!r}, not "
                           f"{comp.dataset!r}")
    return config


# --- prompts ------------------------------------------------------------------------------

class _PromptFields(UserDict):
    """solvers/pure_model_solver.py `_PromptFields`, verbatim."""
    def __missing__(self, key):
        return "{" + key + "}"


def prompt_template(config: Dict[str, Any]) -> str:
    """runner.py `prepare_run`: the instruction, with "\\n\\n{problem}" appended when it has no
    {problem} slot."""
    default_prompt_template = f"{config['instruction']}"
    if "{problem}" not in default_prompt_template:
        default_prompt_template += "\n\n{problem}"
    return default_prompt_template


def build_prompt(config: Dict[str, Any], problem: str) -> str:
    """PureModelSolver.build_query for a text problem: the template through format_map, so the
    config's `{{}}` renders as `{}` and braces inside the problem stay literal."""
    return prompt_template(config).format_map(_PromptFields({"problem": problem}))


#: runner.py `Runner._build_last_chance_prompt`, verbatim apart from being dedented out of the
#: class. Executed below (with options=None, as for both competitions) to produce the exact
#: prompt text.
_LAST_CHANCE_SOURCE = '''
def _build_last_chance_prompt(self, options):
    prompt = """
        Your last message does not provide a final answer in a way that follows the formatting instructions.
        Please based on the conversation history, report the final answer again within \\\\boxed{}.
        If you did not find the answer, please use \\\\boxed{None}.
        Do not reason about the problem again or use tools, simply try to extract the final answer from the previous reasoning.
        Boxed answers in thinking/reasoning stages will be ignored; only the final response message is considered.
        """
    if options is not None:
        option_str = ", ".join([opt for opt in options])
        prompt += f"""
            Recall that this is a multiple choice problem, and the only valid answers to put within \\\\boxed{{}} are: {option_str}."""
    # Remove spaces at the start of each line
    prompt = "\\n".join([line.strip() for line in prompt.split("\\n")])
    return prompt
'''


def _last_chance_prompt(options: Any = None) -> str:
    ns: Dict[str, Any] = {}
    exec(compile(_LAST_CHANCE_SOURCE, "matharena_runner_last_chance", "exec"), ns)
    return ns["_build_last_chance_prompt"](None, options)


LAST_CHANCE_PROMPT = _last_chance_prompt(None)


# --- the grading driver -------------------------------------------------------------------
#
# Runs in the gbench-matharena:b89f2f0a image (or, for unit tests only, in this interpreter): writes the
# vendored files to a temporary directory, imports MathArena's modules from it, and for every
# item either reports whether the final message carries a valid answer ("check", the runner's
# last-chance test) or grades it ("grade", grader.extract_and_grade). The payload arrives on
# stdin; the result is one marked JSON line on stdout.

_DRIVER = r'''
import json, os, shutil, signal, sys, tempfile


class _ItemTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _ItemTimeout()


def _main():
    payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    root = tempfile.mkdtemp(prefix="matharena_")
    try:
        _grade(payload, root)
    finally:
        # The image's /tmp is a tmpfs, but a host run (unit tests) would leave one directory
        # per call in the host's /tmp.
        shutil.rmtree(root, ignore_errors=True)


def _grade(payload, root):
    for rel, text in payload["files"].items():
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    sys.path.insert(0, root)
    if payload.get("stub_axle"):
        # Unit-test runs in the gbench interpreter only: grader.py imports
        # matharena.tools.lean_execution, which imports axle at module level. The final-answer
        # path never calls it; the image installs the real package.
        try:
            import axle  # noqa: F401
        except ImportError:
            import types
            stub = types.ModuleType("axle")
            stub.AxleClient = stub.CheckResponse = None
            sys.modules["axle"] = stub
    from matharena.grader import extract_and_grade
    from matharena.parser import extract_answer
    from matharena.utils import convert_answer_to_string, normalize_conversation

    config = payload["competition_config"]
    options = config.get("options", None)
    mode = payload["mode"]
    limit = int(payload["item_timeout_s"])
    signal.signal(signal.SIGALRM, _on_alarm)
    out = []
    for item in payload["items"]:
        rec = {"id": item["id"]}
        signal.alarm(limit)
        try:
            clean = normalize_conversation(item["conversation"])
            if mode == "check":
                # runner.py process_solver_responses
                last_block = clean[-1]
                last_role, last_content = (last_block.get("role", ""),
                                           last_block.get("content", ""))
                valid_answer_found = True
                if last_role != "assistant":
                    valid_answer_found = False
                else:
                    answer = extract_answer(last_content, True)[0]
                    if answer is None or (options is not None and str(answer) not in options):
                        valid_answer_found = False
                rec["valid"] = bool(valid_answer_found)
            else:
                grading_config = config.copy()
                answer, is_correct, warning = extract_and_grade(
                    clean, int(item.get("output_tokens") or 0), item["gold"], grading_config)
                rec["answer"] = None if answer is None else convert_answer_to_string(answer)
                rec["is_correct"] = is_correct is True
                rec["warning"] = warning
        except _ItemTimeout:
            rec["error"] = "grading took longer than %d s" % limit
        except BaseException as e:
            rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
        finally:
            signal.alarm(0)
        out.append(rec)
    sys.stdout.write("GBENCH_MATHARENA_RESULT " + json.dumps(out) + "\n")
    sys.stdout.flush()


_main()
'''


def image() -> str:
    return os.environ.get("GBENCH_MATHARENA_IMAGE", IMAGE_DEFAULT)


def _build_hint() -> str:
    return (f"Build the grader image LOCALLY (gbench never pulls it):\n"
            f"  docker build -t {image()} -f {DOCKERFILE} docker\n")


def check_prerequisites() -> Tuple[bool, str]:
    """Docker reachable and the local gbench-matharena:b89f2f0a image present."""
    try:
        import yaml  # noqa: F401
    except ImportError:
        return False, "Python package 'pyyaml' is not installed."
    if not shutil.which("docker"):
        return False, "Docker CLI is not found on PATH. " + _build_hint()
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=30)
        if r.returncode != 0:
            return False, "Docker daemon is not reachable. " + _build_hint()
    except Exception as e:                                              # noqa: BLE001
        return False, f"Cannot connect to the Docker daemon: {e}. " + _build_hint()
    if subprocess.run(["docker", "image", "inspect", image()],
                      capture_output=True).returncode != 0:
        return False, f"image {image()!r} not found. " + _build_hint()
    return True, ""


def _docker_command() -> List[str]:
    return ["docker", "run", "--rm", "-i", "--network", "none", "--read-only",
            "--tmpfs", "/tmp:rw,size=256m", "--memory", "4g", "--pids-limit", "256",
            "--user", "65534:65534", image(), "python", "-I", "-c", _DRIVER]


def _host_command() -> List[str]:
    return [sys.executable, "-I", "-c", _DRIVER]


def run_driver(mode: str, items: List[Dict[str, Any]], config: Dict[str, Any],
               *, where: str = "image", timeout: Optional[float] = None) -> List[Dict[str, Any]]:
    """Run MathArena's check or grade over `items`; one record per item, in order.

    `where="image"` is how the suites grade. `where="host"` runs the same driver in this
    interpreter and exists for unit tests and diagnostics only: gemma4-env's antlr runtime is
    not the 4.11 MathArena pins, so answers that need sympy's LaTeX parser grade differently.
    """
    if mode not in ("check", "grade"):
        raise ValueError(mode)
    if not items:
        return []
    payload = json.dumps({"files": vendored_files(), "mode": mode,
                          "competition_config": config, "item_timeout_s": ITEM_TIMEOUT_S,
                          "items": items, "stub_axle": where != "image"})
    cmd = _docker_command() if where == "image" else _host_command()
    if timeout is None:
        timeout = 600 + (ITEM_TIMEOUT_S + 5) * len(items)
    proc = subprocess.run(cmd, input=payload.encode("utf-8"), capture_output=True,
                          timeout=timeout)
    stdout = proc.stdout.decode("utf-8", "replace")
    for line in reversed(stdout.splitlines()):
        if line.startswith(_RESULT_MARKER):
            records = json.loads(line[len(_RESULT_MARKER):])
            if [r.get("id") for r in records] != [i["id"] for i in items]:
                raise RuntimeError("matharena: the grader returned records out of order")
            return records
    tail = proc.stderr.decode("utf-8", "replace")[-800:]
    raise RuntimeError(f"matharena: the {mode} pass produced no result (exit "
                       f"{proc.returncode}); stderr tail: {tail}")


#: Known answers the image must grade correctly before a run starts. The second needs sympy's
#: LaTeX parser, i.e. the antlr 4.11 runtime the image exists to provide.
_CANARIES = [
    (r"So the answer is $\boxed{\frac{1}{2}}$.", "0.5", True),
    (r"Therefore $\boxed{(3+\sqrt{6})^{-1/3}}$.", r"(3+\sqrt{6})^{-1/3}", True),
    (r"The answer is $\boxed{7}$.", "8", False),
]


def self_check(config: Dict[str, Any]) -> Optional[str]:
    """Why the MathArena grader cannot run here, or None (grades the known canaries)."""
    items = [{"id": i, "gold": gold, "output_tokens": 0,
              "conversation": [{"role": "user", "content": "canary"},
                               {"role": "assistant", "content": text}]}
             for i, (text, gold, _) in enumerate(_CANARIES)]
    try:
        records = run_driver("grade", items, config, timeout=600)
    except Exception as e:                                              # noqa: BLE001
        return f"the MathArena grader did not run in {image()!r}: {e}. " + _build_hint()
    for rec, (text, gold, want) in zip(records, _CANARIES):
        if rec.get("error") or rec.get("is_correct") is not want:
            return (f"the MathArena grader in {image()!r} graded a known answer wrongly "
                    f"({text!r} vs gold {gold!r}: {rec}); rebuild it from {DOCKERFILE} "
                    f"(it needs antlr4-python3-runtime 4.11)")
    return None


def grade_final_response(text: str, gold: str, comp_key: str) -> bool:
    """MathArena's extract_and_grade on one final response, run in THIS interpreter.

    For unit tests and diagnostics only; run_hmmt / run_aime grade in the image, which also
    provides the antlr 4.11 runtime the LaTeX fallback needs.
    """
    config = competition_config(COMPETITIONS[comp_key])
    item = {"id": 0, "gold": str(gold), "output_tokens": 0,
            "conversation": [{"role": "user", "content": "problem"},
                             {"role": "assistant", "content": text or ""}]}
    rec = run_driver("grade", [item], config, where="host", timeout=300)[0]
    if rec.get("error"):
        raise RuntimeError(f"matharena: grading failed: {rec['error']}")
    return rec.get("is_correct") is True


# --- conversations ------------------------------------------------------------------------

def _assistant_turn(content: Optional[str], reasoning: Optional[str]) -> List[Dict[str, Any]]:
    """An assistant reply the way MathArena's OpenAI chat client records it: the reasoning as a
    separate `cot` message, then the message itself (content "" when the server sent none)."""
    turn: List[Dict[str, Any]] = []
    if reasoning:
        turn.append({"role": "assistant", "type": "cot", "content": reasoning})
    turn.append({"role": "assistant", "content": content or ""})
    return turn


def _prompt_messages(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"role": m.get("role"), "content": m.get("content")}
            for m in (trace.get("messages") or [])]


def first_conversation(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    return _prompt_messages(trace) + _assistant_turn(trace.get("response_text"),
                                                     trace.get("reasoning"))


def last_chance_request(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The messages PureModelSolver.last_chance sends: the conversation with the reasoning
    dropped (APIClient._drop_cot), then the last-chance prompt."""
    return (_prompt_messages(trace)
            + [{"role": "assistant", "content": trace.get("response_text") or ""},
               {"role": "user", "content": LAST_CHANCE_PROMPT}])


# --- the scorer ---------------------------------------------------------------------------

def _new_stats() -> Dict[str, Any]:
    return {"last_chance_reprompts": 0, "last_chance_failures": 0, "grading_errors": 0,
            "check_errors": 0, "ungraded_failed_requests": 0, "warnings": {},
            "grader_failure": None}


def make_scorer(comp: Competition, config: Dict[str, Any], stats: Dict[str, Any], *,
                eval_name: str, model_name: str, base_url: str, concurrency: int,
                thinking: bool, temperature: float, extra_payload: Optional[Dict[str, Any]]):
    """The async_eval_fn for run_eval_suite: last chance where needed, then grading.

    Harness errors are left out of the score (`scoring_excluded`), counted and named, never
    scored as the model's answer: a first request with no reply (base marks it), a last-chance
    request with no reply (marked here with its `harness_error`), and a response the grader
    could not grade (`judge_grade` JUDGE_FAILED, counted by base under judge failures)."""
    stats.update(_new_stats())

    async def _score(traces: List[Dict[str, Any]]) -> None:
        from . import base

        live = [t for t in traces if t.get("response_text") is not None]
        for t in traces:
            if t.get("response_text") is None:
                # The request itself got no reply (failed, timed out, did not fit the window):
                # nothing to grade. MathArena retries until it has a response; base marks it a
                # harness error (left out of the score), and the result is not comparable.
                t["is_correct"] = False
                stats["ungraded_failed_requests"] += 1

        # 1. The runner's last-chance test on every first reply.
        try:
            checks = await asyncio.to_thread(
                run_driver, "check",
                [{"id": i, "conversation": first_conversation(t)} for i, t in enumerate(live)],
                config)
        except Exception as e:                                          # noqa: BLE001
            # The grader itself did not run (Docker died mid-run, say): keep every trace,
            # mark them ungraded, and let the result say so rather than lose the run.
            _grader_failed(live, stats, f"the last-chance check did not run: {e}")
            return
        needs = []
        for t, rec in zip(live, checks):
            if rec.get("error"):
                t["matharena_check_error"] = rec["error"]
                stats["check_errors"] += 1
            elif not rec.get("valid"):
                needs.append(t)

        # 2. One last-chance turn for each reply without a valid boxed answer, sent with the
        #    run's own settings (MathArena reuses the solver's client arguments).
        if needs:
            payload = dict(extra_payload or {})
            payload.update(base.decoding_penalties())
            ctk = base.effective_chat_template_kwargs(payload.get("chat_template_kwargs"),
                                                      thinking)
            budgets = [t.get("max_tokens_sent") for t in needs if t.get("max_tokens_sent")]
            timeout_s = base.request_timeout_s(max(budgets) if budgets else None)
            semaphore = asyncio.Semaphore(max(1, concurrency))
            import aiohttp
            connector = aiohttp.TCPConnector(limit=max(1, concurrency) + 20)
            api_url = f"{base_url.rstrip('/')}/chat/completions"

            async def _one(session, t):
                convo = last_chance_request(t)
                budget = t.get("max_tokens_sent")
                sent_max, oversize = await base.clamp_to_context_async(convo, budget, None, ctk)
                if oversize:
                    return t, base.Reply(None, None, None, None,
                                         f"{base.OVER_CONTEXT_ERROR}: {oversize}"), None
                reply = await base._send_single_request(
                    session=session, api_url=api_url, model_name=model_name, messages=convo,
                    extra_payload=payload, semaphore=semaphore, pbar=None,
                    max_output_tokens=sent_max, temperature=temperature, thinking=thinking)
                return t, reply, sent_max

            async with aiohttp.ClientSession(
                    connector=connector,
                    timeout=aiohttp.ClientTimeout(total=timeout_s)) as session:
                done = await asyncio.gather(*[_one(session, t) for t in needs])
            for t, reply, sent_max in done:
                stats["last_chance_reprompts"] += 1
                t["matharena_last_chance"] = {
                    # the max_tokens this request carried (the window often lowers it below the
                    # first request's, since its prompt holds the whole first reply); a lower
                    # value a context-length retry sent, when the reply says so
                    "max_tokens_sent": (getattr(reply, "max_tokens_sent", None) or sent_max),
                    "response_text": reply.text,
                    "reasoning": reply.reasoning,
                    "reasoning_chars": len(reply.reasoning or ""),
                    "finish_reason": reply.finish_reason,
                    "completion_tokens": reply.completion_tokens,
                    "error": reply.error,
                }
                if reply.text is None:
                    stats["last_chance_failures"] += 1
                    # No reply from the served endpoint to the last-chance request: a harness
                    # error, left out of the score (never the model's answer).
                    health = base.classify_reply(reply)
                    status = ("OVER_CONTEXT"
                              if str(reply.error or "").startswith(base.OVER_CONTEXT_ERROR)
                              else "TIMEOUT" if health == "request_timeout" else "FAILED")
                    t["harness_error"] = base.endpoint_harness_error(
                        status, f"last-chance request: {reply.error or 'no reply'}")
                    t["scoring_excluded"] = True

        # 3. Grade the final conversation of every live trace.
        to_grade, items = [], []
        for t in live:
            if t.get("matharena_check_error"):
                continue
            lc = t.get("matharena_last_chance")
            if lc is not None and lc.get("response_text") is None:
                continue                      # the last-chance request itself failed
            convo = first_conversation(t)
            tokens = int(t.get("completion_tokens") or 0)
            if lc is not None:
                convo = (convo + [{"role": "user", "content": LAST_CHANCE_PROMPT}]
                         + _assistant_turn(lc.get("response_text"), lc.get("reasoning")))
                tokens += int(lc.get("completion_tokens") or 0)
            items.append({"id": len(items), "conversation": convo, "gold": str(t["gold_answer"]),
                          "output_tokens": tokens})
            to_grade.append(t)
        try:
            grades = await asyncio.to_thread(run_driver, "grade", items, config)
        except Exception as e:                                          # noqa: BLE001
            _grader_failed(to_grade, stats, f"grading did not run: {e}")
            grades = []

        for t in live:
            if t.get("matharena_check_error"):
                t["is_correct"] = False
                t["judge_grade"] = "JUDGE_FAILED"
                t["status"] = "GRADING_ERROR"
                t["scoring_excluded"] = True
            elif (t.get("matharena_last_chance") or {}).get("response_text", "") is None:
                t["is_correct"] = False
                t["status"] = "LAST_CHANCE_FAILED"
        for t, rec in zip(to_grade, grades):
            if rec.get("error"):
                stats["grading_errors"] += 1
                t["is_correct"] = False
                t["judge_grade"] = "JUDGE_FAILED"
                t["status"] = "GRADING_ERROR"
                t["scoring_excluded"] = True
                t["matharena_grading_error"] = rec["error"]
                continue
            t["is_correct"] = rec.get("is_correct") is True
            t["status"] = "OK" if t["is_correct"] else "FAILED"
            t["matharena_answer"] = rec.get("answer")
            t["matharena_warning"] = rec.get("warning")
            level = str(rec.get("warning"))
            stats["warnings"][level] = stats["warnings"].get(level, 0) + 1

    return _score


def _grader_failed(traces: List[Dict[str, Any]], stats: Dict[str, Any], why: str) -> None:
    logger.error("matharena: %s", why)
    stats["grader_failure"] = why[:500]
    stats["grading_errors"] += len(traces)
    for t in traces:
        t["is_correct"] = False
        t["judge_grade"] = "JUDGE_FAILED"
        t["status"] = "GRADING_ERROR"
        t["scoring_excluded"] = True
        t["matharena_grading_error"] = why[:300]


def apply_grader_failure(result: Dict[str, Any], stats: Dict[str, Any]) -> None:
    """A grader that did not run makes the whole result an error, never a low score."""
    if stats.get("grader_failure"):
        result["status"] = "error"
        result["error"] = stats["grader_failure"]


def attempts_run(result: Dict[str, Any]) -> int:
    attempts = result.get("attempts")
    return int((attempts or {}).get("attempts_per_sample", 1) or 1) if isinstance(attempts, dict) else 1


def comparability_reasons(result: Dict[str, Any], stats: Dict[str, Any],
                          limit: Optional[int],
                          recommended_temperature: Optional[float] = None,
                          payload: Optional[Dict[str, Any]] = None,
                          recommended: Optional[OfficialSampling] = None) -> List[str]:
    """Every way this run differs from MathArena's protocol (empty = comparable).
    `recommended` is the served model's recommended sampling (default SAMPLING, Gemma 4's);
    `recommended_temperature` defaults to its temperature; `payload` is the request_payload the
    run sent (default: its params)."""
    rec = SAMPLING if recommended is None else recommended
    whose = "for Gemma 4" if recommended is None else "for the served model"
    reasons: List[str] = []
    runs = attempts_run(result)
    if runs < OFFICIAL_RUNS:
        reasons.append(f"{runs} run(s) per problem; MathArena averages {OFFICIAL_RUNS}")
    temp = result.get("temperature")
    want = rec.temperature if recommended_temperature is None else recommended_temperature
    if want is None:
        if temp is not None:
            reasons.append(f"temperature={temp} ({result.get('temperature_source')}); MathArena "
                           "samples each model at its provider-recommended settings, which set "
                           f"no temperature {whose}")
    elif temp is None or abs(float(temp) - float(want)) > 1e-9:
        reasons.append(f"temperature={temp} ({result.get('temperature_source')}); MathArena "
                       f"samples each model at its provider-recommended temperature ({want} "
                       f"{whose})")
    sent = request_payload(None, rec) if payload is None else payload
    changed = sorted(k for k, v in (rec.params or {}).items() if sent.get(k) != v)
    if changed:
        reasons.append("sampling " + ", ".join(f"{k}={sent.get(k)}" for k in changed)
                       + "; MathArena uses the provider's recommended "
                       + ", ".join(f"{k}={rec.params[k]}" for k in changed))
    if result.get("decoding_penalties"):
        reasons.append(f"decoding penalties {result['decoding_penalties']}; the recommended "
                       "sampling has none")
    if limit:
        reasons.append(f"subset run (--eval-limit {limit})")
    if stats.get("ungraded_failed_requests"):
        reasons.append(f"{stats['ungraded_failed_requests']} generation request(s) got no reply "
                       "from the served endpoint (harness errors, left out of the score); "
                       "MathArena retries until it has one")
    if stats.get("last_chance_failures"):
        reasons.append(f"{stats['last_chance_failures']} last-chance request(s) got no reply "
                       "from the served endpoint (harness errors, left out of the score)")
    errors = int(stats.get("grading_errors") or 0) + int(stats.get("check_errors") or 0)
    if errors:
        reasons.append(f"{errors} response(s) could not be graded by MathArena's grader "
                       "(harness errors, left out of the score)")
    return reasons


def matharena_record(comp: Competition, config: Dict[str, Any],
                     recommended: Optional[OfficialSampling] = None) -> Dict[str, Any]:
    """The result's `matharena` block; `recommended` is the served model's recommended sampling
    (default SAMPLING, Gemma 4's)."""
    rec = SAMPLING if recommended is None else recommended
    return {
        "repo": MATHARENA_REPO,
        "commit": MATHARENA_COMMIT,
        "competition_config": f"configs/competitions/{comp.key}.yaml",
        "instruction": config.get("instruction"),
        "strict_parsing": config.get("strict_parsing", False),
        "grader": "matharena grader.extract_and_grade (vendored verbatim)",
        "grader_image": image(),
        "last_chance_prompt": LAST_CHANCE_PROMPT,
        "sampling": rec.reference,
    }
