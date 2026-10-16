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

# Evaluation Suite: gaia
# Description: GAIA (General AI Assistants) - canonical question_scorer exact-match

"""gbench native built-in runner for gaia (Agentic & Assistant Tasks).

Canonical GAIA (gaia-benchmark/GAIA 2023, pinned revision) with the official system prompt and
the official `question_scorer` (number/list/string normalized exact match on the FINAL ANSWER).
The paper's protocol is "zero-shot prompt an AI assistant with the questions and attached
evidence if there are some", so a task that ships a file gets it in the same user message:
images, audio and video as media parts, documents rendered to text (see `attachment_parts`).
The gated dataset requires HF auth and a search backend (GEMINI_API_KEYS or GEMINI_API_KEY); the
suite HARD-ERRORS (infra_required, never skips) if either is absent, and before the download it
checks the key live (base.gemini_key_live_valid): a key the Gemini API rejects, or a model of the
Gemini cascade the API does not serve to it, is infra_required too (an inconclusive check passes).

Two settings, both published by the GAIA leaderboard (gaia-benchmark/results_public 2023 has a
validation and a test table):

* `validation` (default): the 165 public tasks, scored locally with `question_scorer`.
* `test` (`GBENCH_GAIA_SPLIT=test`): the 301 tasks the leaderboard ranks on, whose answers are
  hidden. gbench runs them with the same protocol and writes the leaderboard submission file
  (JSON lines of `task_id`, `model_answer`, `reasoning_trace`, the format the leaderboard
  space's submission form reads); the score comes only from uploading that file, so the run
  ends with infra_required naming the file and the upload.

gbench runs GAIA with its `web_search` tool in a tool loop, not the full browsing agent (page
fetch, code execution) leaderboard entries use, so `leaderboard_comparable` is always False. The
tool's results are written by Gemini: each call is one Gemini generateContent request grounded in
Google Search with the model's query as the prompt, and it returns the cited sources' titles and
URLs with segments of that Gemini answer, never page text. The tool description the model sees
(GROUNDED_SEARCH_DESCRIPTION) says so, as do the result (`search_backend_kind`) and
`leaderboard_comparable_reason`. Every trace keeps its web_search calls (`tool_transcript`: the
query, the Gemini backend that answered, and the reply the model read).

Harness errors are never the model's outcome: a sample whose model request produced no reply
(an endpoint error, a timeout, a prompt longer than the served window), or that made a web_search
call the Gemini search backend could not answer, is left out of the accuracy (numerator and
denominator) and the per-level and per-attachment tables, counted in `harness_errors` and named in
`leaderboard_comparable_reason`; the status is then completed_with_errors, and a run with no
sample left withholds its score (accuracy null, status failed).

Sampling: GAIA prescribes none, and the open reference agent the loop settings come from sends no
temperature (run_gaia.py model_params); gbench always sends one, its rule for every suite
(base.resolve_temperature): 1.0 with `--thinking`, 0.0 without. `--temperature`, or
`GBENCH_GAIA_TEMPERATURE` for this suite alone (it takes precedence), sets another; the result
records the reference agent's setting under `official_sampling`. max_tokens follows the same reference agent through
the token-limit policy (base.resolve_token_budget). Its model_params give model_id "o1"
reasoning_effort "high" and max_completion_tokens 8192 and every other model max_tokens 4096. The
o1 branch keys on OpenAI's model name, so it is not a reasoning-model setting for a served model,
which takes the max_tokens 4096 path (the script's commented-out open-model line also gives
Qwen/Qwen3-32B max_tokens=4096). Without `--thinking` every turn sends max_tokens 4096; with
`--thinking` the 4096 is not applied, because it was written for non-reasoning generation, and
every turn is sent the run's `--max-output-tokens`, else the thinking floor (recorded as not
applied to reasoning). No reasoning_effort is sent in either mode. Final replies that stopped at
max_tokens on a request that did not carry the 4096 itself (with `--thinking` every such reply;
without it, those of a sample whose first request the context window lowered below 4096) are
counted in `replies_cut_off_by_budget` (`replies_cut_off_in_reasoning` for those with no answer
text) and named in `leaderboard_comparable_reason`.
"""

import base64
import contextvars
import json
import logging
import os
import re
import shutil
import string
import subprocess
import tempfile
import time
import zipfile
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree

from .search_tool import (WEB_SEARCH_TOOL, backend_tally, execute_tool, reset_backend_tally,
                          search_available, search_backend_name)
from .base import redact_text
from .swebench_common import infra_required
from .base import (OfficialSampling, ToolLoopProtocol, apply_shard, run_eval_suite,
                   strip_thinking_tags)
from .base import MODELS_NOT_SERVED_PREFIX, gemini_api_keys, gemini_key_live_valid
from .metrics import record_budget_cutoffs
from .sampling import presharded_for_base, stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "Agentic & Assistant Tasks"
DOCS_URL = "docs/evals/gaia.md"

DATASET_REPO = "gaia-benchmark/GAIA"
#: Pinned dataset commit (2023/validation: metadata.parquet + 38 attachments; 2023/test:
#: metadata.parquet + 71 attachments, answers hidden).
DATASET_REVISION = "682dd723ee1e1697e00360edccf2366dc8418dd9"

#: The GAIA 2023 splits the leaderboard publishes. `validation` is scored locally; `test` is
#: scored only by the leaderboard from an uploaded submission file.
SPLITS = ("validation", "test")
#: gaia-benchmark/leaderboard app.py at LEADERBOARD_REVISION: `ref_level_len`, the tasks per level
#: a submission must contain (it rejects any other count).
LEADERBOARD_LEVEL_COUNTS = {"validation": {1: 53, 2: 86, 3: 26}, "test": {1: 93, 2: 159, 3: 49}}
LEADERBOARD_URL = "https://huggingface.co/spaces/gaia-benchmark/leaderboard"
LEADERBOARD_REVISION = "9f133d71362e77b3539f1514f31b9c101a545fec"

#: GAIA prescribes no agent loop. The round cap is the open GAIA reference agent's: its
#: web-search ToolCallingAgent runs max_steps=20 and, at the limit, is made to produce a final
#: answer, which is what gbench's forced final does. Each turn gets the reference agent's
#: max_tokens (GAIA_SAMPLING, through the token-limit policy), never the house turn ceiling.
GAIA_TOOL_LOOP = ToolLoopProtocol(
    reference=("GAIA prescribes no loop; round cap from huggingface/smolagents "
               "examples/open_deep_research/run_gaia.py@227ef5e49ddd (web-search "
               "ToolCallingAgent max_steps=20, final answer forced at the step limit)"),
    max_rounds=20,
    max_tool_calls=None,
    house_turn_ceiling=False,
    forced_final=True,
    rendered_calls_in_history=False,
)

#: GAIA prescribes no sampling; the reference agent (run_gaia.py answer_single_question) builds
#: LiteLLMModel(model_id, custom_role_conversions, max_tokens=4096) with no temperature, so the
#: model's defaults apply. Only model_id "o1" gets reasoning_effort="high" and
#: max_completion_tokens=8192 instead: that branch keys on OpenAI's model name, so it is not a
#: reasoning-model setting for a served model, and none is declared here. A --thinking run sends
#: the same request with the run's budget in place of the 4096 (the token-limit policy).
REFERENCE_MAX_TOKENS = 4096
_RUN_GAIA = "huggingface/smolagents examples/open_deep_research/run_gaia.py@227ef5e49ddd"
GAIA_SAMPLING = OfficialSampling(
    reference=(f"GAIA prescribes no sampling; {_RUN_GAIA} answer_single_question model_params "
               "for every model_id but o1 (the path a served model takes in both modes): no "
               "temperature, max_tokens=4096"),
    send_temperature=False,
    max_tokens=REFERENCE_MAX_TOKENS,
)

#: Characters of rendered text kept per attachment: the same limit as the reference agent's
#: TextInspectorTool(model, 100000). Only the 2.9 MB .pdb file in the validation split is
#: longer; its first ATOM records sit past the cut, as they do for the reference agent.
MAX_ATTACHMENT_CHARS = 100_000

# ---- The web_search tool of the web-research suites (gaia, deepsearch_qa) -----------------
#: What the tool returns, said to the model. search_tool answers a query with one Gemini
#: generateContent call grounded in Google Search: each result's title and URL are a source the
#: grounding cited, and its snippet is the part of Gemini's own answer that cites that source (or
#: the start of the answer), not text taken from the page. Neither GAIA nor DeepSearchQA defines a
#: tool or its description (GAIA prescribes no agent; DeepSearchQA's reference gives the model its
#: own Google Search), so the description is gbench's and says what the tool is.
GROUNDED_SEARCH_DESCRIPTION = (
    "Search the web through a Gemini model grounded in Google Search. Each result gives the title "
    "and URL of a source the Gemini model cited, with the part of that model's written answer "
    "that cites the source, or the start of its answer when it cited none (text written by the "
    "Gemini model, not quoted from the page). Use it whenever the answer depends on information "
    "you do not already know, and call it repeatedly to refine or cross-check. Prefer specific "
    "queries.")
#: search_tool's web_search tool with GROUNDED_SEARCH_DESCRIPTION.
GROUNDED_SEARCH_TOOL: Dict[str, Any] = {
    **WEB_SEARCH_TOOL,
    "function": {**WEB_SEARCH_TOOL["function"], "description": GROUNDED_SEARCH_DESCRIPTION}}
#: How the result and the docs name the tool (search_backend_kind).
SEARCH_BACKEND_KIND = ("Gemini Google-Search-grounded answers: each web_search call is one Gemini "
                       "generateContent request with the google_search tool, the model's query "
                       "as the prompt; the results carry the cited sources' titles and URLs and "
                       "segments of the Gemini model's written answer, no page text")
#: search_with_fallback's backend names for a lookup no grounding model answered.
FAILED_SEARCH_BACKENDS = ("all-grounding-models-failed", "deadline-passed", "none")

#: The web_search calls of the running suite, per sample: {sample_idx: [call, ...]}. Filled by
#: record_web_search (the suite's tool executor) and put on each trace as `tool_transcript`.
_TOOL_TRANSCRIPTS: Dict[int, List[Dict[str, Any]]] = {}
#: The running run's store (reset_tool_transcripts): a context variable, so two suites running in
#: one process (another thread, or back to back) never read each other's calls; the tool loop's
#: tasks inherit it from the run that set it.
_RUN_TRANSCRIPTS: "contextvars.ContextVar[Optional[Dict[int, List[Dict[str, Any]]]]]" = \
    contextvars.ContextVar("gbench_web_search_transcripts", default=None)


def _transcripts() -> Dict[int, List[Dict[str, Any]]]:
    store = _RUN_TRANSCRIPTS.get()
    return store if store is not None else _TOOL_TRANSCRIPTS


def reset_tool_transcripts() -> Dict[int, List[Dict[str, Any]]]:
    """A new store for this run's web_search calls (at the start of a run, as
    reset_backend_tally); returns it."""
    global _TOOL_TRANSCRIPTS
    _TOOL_TRANSCRIPTS = {}
    _RUN_TRANSCRIPTS.set(_TOOL_TRANSCRIPTS)
    return _TOOL_TRANSCRIPTS


def transcript_entry(name: str, args: Any, out: str) -> Dict[str, Any]:
    """One tool call as a transcript entry: the tool, its arguments, the Gemini backend that
    answered (search_tool's `backend`, "gemini:<model>"), whether the search backend failed (no
    grounding model answered, or it returned an error result) and the reply the model read."""
    backend, failed = None, False
    try:
        obj = json.loads(out)
    except (TypeError, ValueError):
        obj = None
    if isinstance(obj, dict):
        backend = obj.get("backend")
        results = obj.get("results") or []
        first = results[0] if results and isinstance(results[0], dict) else {}
        failed = backend in FAILED_SEARCH_BACKENDS or (
            backend is not None and first.get("title") == "error")
    return {"tool": name, "arguments": args, "backend": backend, "search_failed": failed,
            "reply": out}


async def record_web_search(name: str, args: Dict[str, Any], sample_idx: int) -> str:
    """The suites' tool executor: search_tool.execute_tool, with the call recorded against its
    sample (base hands a three-parameter executor the sample index). A call whose tool raised is
    recorded too, as a failed search (backend `tool-raised`, the exception's type), and the
    exception goes on to base, which answers the model with an error."""
    try:
        out = await execute_tool(name, args)
    except Exception as e:
        entry = transcript_entry(name, args, json.dumps(
            {"error": f"{type(e).__name__}: {redact_text(str(e))[:200]}"}))
        entry.update(backend="tool-raised", search_failed=True, exception=type(e).__name__)
        _transcripts().setdefault(sample_idx, []).append(entry)
        raise
    _transcripts().setdefault(sample_idx, []).append(transcript_entry(name, args, out))
    return out


def attach_tool_transcripts(traces: List[Dict[str, Any]]) -> None:
    """Put each sample's web_search calls on its trace (`tool_transcript`, [] for none) and count
    them (`web_search_calls`)."""
    store = _transcripts()
    for t in traces:
        calls = list(store.get(t.get("sample_idx"), []))
        if "tool_transcript" not in t:
            t["tool_transcript"] = calls
        t["web_search_calls"] = len(t["tool_transcript"])


#: base's statuses of a request that produced no reply, and the harness error each is.
_NO_REPLY_KINDS = {"FAILED": "endpoint_failure", "TIMEOUT": "endpoint_timeout",
                   "OVER_CONTEXT": "over_context"}


def web_research_harness_error(trace: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """The harness error a web-research sample ended in, or None when its outcome is the model's:

    * `endpoint_failure` / `endpoint_timeout` / `over_context`: a model request produced no reply
      (an endpoint error, a timeout, a prompt longer than the served context window);
    * `request_failed_mid_loop`: a model request of the tool loop failed after earlier turns, so
      the reply base kept is not the model's answer;
    * `search_backend_failure`: a web_search call the Gemini search backend could not answer (every
      grounding model failed), so the model worked without a lookup it asked for.
    """
    status = str(trace.get("status") or "")
    if status in _NO_REPLY_KINDS:
        return {"kind": _NO_REPLY_KINDS[status],
                "detail": redact_text(str(trace.get("error") or "no reply"))[:300]}
    if trace.get("tool_loop_end") == "request_failed":
        return {"kind": "request_failed_mid_loop",
                "detail": "a model request of the tool loop failed after earlier turns"}
    failed = [c for c in trace.get("tool_transcript") or [] if c.get("search_failed")]
    if failed:
        return {"kind": "search_backend_failure",
                "detail": (f"{len(failed)} of {len(trace.get('tool_transcript') or [])} "
                           f"web_search call(s) got no answer from the Gemini search backend "
                           f"({failed[0].get('backend')})")}
    return None


def harness_error_counts(traces: List[Dict[str, Any]]) -> Dict[str, int]:
    """{kind: samples} over the traces' `harness_error`."""
    out: Dict[str, int] = {}
    for t in traces:
        err = t.get("harness_error")
        if err:
            out[err["kind"]] = out.get(err["kind"], 0) + 1
    return dict(sorted(out.items()))


def harness_error_reason(counts: Dict[str, int]) -> str:
    """The leaderboard_comparable_reason / status_reason text for harness-error counts."""
    labels = {"endpoint_failure": "a model request failed",
              "endpoint_timeout": "a model request timed out",
              "over_context": "the prompt did not fit the served context window",
              "request_failed_mid_loop": "a model request failed mid tool loop",
              "search_backend_failure": "a web_search call got no answer from the Gemini search "
                                        "backend"}
    return ("harness errors, not the model's outcome and left out of every metric: "
            + "; ".join(f"{n} sample(s) where {labels.get(k, k)} ({k})"
                        for k, n in counts.items()))


# Verbatim canonical GAIA system prompt (paper / leaderboard).
_GAIA_SYSTEM = (
    "You are a general AI assistant. I will ask you a question. Report your thoughts, "
    "and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. "
    "YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated "
    "list of numbers and/or strings. If you are asked for a number, don't use comma to write "
    "your number neither use units such as $ or percent sign unless specified otherwise. If "
    "you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), "
    "and write the digits in plain text unless specified otherwise. If you are asked for a "
    "comma separated list, apply the above rules depending of whether the element to be put in "
    "the list is a number or a string."
)


# ---- Official GAIA question_scorer (huggingface.co/spaces/gaia-benchmark/leaderboard) ----
def _is_float(x: Any) -> bool:
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


def _normalize_number_str(s: str) -> float:
    for ch in ["$", "%", ","]:
        s = s.replace(ch, "")
    try:
        return float(s)
    except ValueError:
        return float("inf")


def _split_string(s: str, char_list: Optional[List[str]] = None) -> List[str]:
    char_list = char_list or [",", ";"]
    return re.split("[" + "".join(char_list) + "]", s)


def _normalize_str(s: str, remove_punct: bool = True) -> str:
    no_spaces = re.sub(r"\s", "", s)
    if remove_punct:
        return no_spaces.lower().translate(str.maketrans("", "", string.punctuation))
    return no_spaces.lower()


def _question_scorer(model_answer: str, ground_truth: str) -> bool:
    gt, ma = str(ground_truth), str(model_answer)
    if _is_float(gt):
        return _normalize_number_str(ma) == float(gt)
    if any(c in gt for c in [",", ";"]):
        gt_elems, ma_elems = _split_string(gt), _split_string(ma)
        if len(gt_elems) != len(ma_elems):
            return False
        for g, m in zip(gt_elems, ma_elems):
            if _is_float(g):
                if _normalize_number_str(m) != float(g):
                    return False
            elif _normalize_str(m, remove_punct=False) != _normalize_str(g, remove_punct=False):
                return False
        return True
    return _normalize_str(ma) == _normalize_str(gt)


_FINAL_ANSWER_RE = re.compile(r"FINAL\s+ANSWER\s*:?\s*(.*)", re.IGNORECASE)


def _extract_final_answer(text: str) -> str:
    """Text after the last `FINAL ANSWER` marker.

    The old version sliced a fixed `len("FINAL ANSWER: ")` == 14 characters past the
    marker, so any spacing other than exactly `FINAL ANSWER: ` mis-cut the answer:
    `FINAL ANSWER:42` lost its first two characters and `FINAL ANSWER:  42` kept a
    stray space. It also took the FIRST marker, not the model's last word on it.
    """
    matches = _FINAL_ANSWER_RE.findall(text or "")
    if not matches:
        return ""
    return matches[-1].strip().splitlines()[0].strip() if matches[-1].strip() else ""


def _eval_gaia(response_text: str, gold_target: Any) -> bool:
    ans = _extract_final_answer(response_text or "")
    if not ans:
        return False
    return _question_scorer(ans, gold_target)


def _hf_token() -> Optional[str]:
    tok = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        try:
            from huggingface_hub import get_token
            tok = get_token()
        except Exception:
            pass
    return tok


def check_gaia_prerequisites() -> Tuple[bool, str]:
    """huggingface_hub/pandas importable + HF token granting access to the gated repo."""
    try:
        import huggingface_hub  # noqa: F401
        import pandas  # noqa: F401
        import pyarrow  # noqa: F401  (pandas' parquet engine for metadata.parquet)
    except ImportError:
        return False, "Python packages 'huggingface_hub'/'pandas'/'pyarrow' are not installed."
    tok = _hf_token()
    gate_msg = ("gaia-benchmark/GAIA is gated: set HF_TOKEN and accept the license at "
                "https://huggingface.co/datasets/gaia-benchmark/GAIA")
    if not tok:
        return False, gate_msg
    try:
        from huggingface_hub import HfApi
        HfApi().dataset_info("gaia-benchmark/GAIA", token=tok)
    except Exception as e:
        return False, f"{gate_msg} ({type(e).__name__})"
    return True, ""


# ---- Attachments ---------------------------------------------------------------------------
_IMAGE_MIME = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
               "gif": "image/gif", "webp": "image/webp"}
_AUDIO_FORMATS = {"mp3": "mp3", "wav": "wav"}
#: Audio containers an OpenAI-style `input_audio` part does not take (it names mp3 and wav):
#: decoded to wav with ffmpeg first (the test split's one .m4a file).
_TRANSCODED_AUDIO = {"m4a"}
#: Video files, sent whole as a `video_url` data URI (the test split's one .MOV file).
_VIDEO_MIME = {"mov": "video/quicktime", "mp4": "video/mp4"}
#: Plain-text formats, passed as their own text.
_TEXT_EXTS = {"txt", "csv", "tsv", "py", "json", "jsonld", "xml", "md", "html", "pdb"}

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


class AttachmentDependencyError(RuntimeError):
    """A package needed to render an attachment is missing (missing infra, not a task fault)."""


def _audio_to_wav(data: bytes, name: str) -> bytes:
    """The audio file decoded to wav by ffmpeg (channels and sample rate kept)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise AttachmentDependencyError(f"ffmpeg (on PATH, to decode the .{name.rsplit('.', 1)[-1]} "
                                        "audio attachment)")
    with tempfile.TemporaryDirectory(prefix="gbench_gaia_audio_") as tmp:
        src = os.path.join(tmp, "in." + name.rsplit(".", 1)[-1].lower())
        dst = os.path.join(tmp, "out.wav")
        with open(src, "wb") as fh:
            fh.write(data)
        proc = subprocess.run([ffmpeg, "-nostdin", "-loglevel", "error", "-i", src, "-vn",
                               "-f", "wav", dst], capture_output=True, text=True)
        if proc.returncode != 0 or not os.path.isfile(dst):
            raise RuntimeError(f"gaia: ffmpeg could not decode the attachment '{name}': "
                               f"{(proc.stderr or '').strip()[-300:]}")
        with open(dst, "rb") as fh:
            return fh.read()


def _clip(text: str, name: str) -> str:
    text = (text or "").strip()
    if len(text) > MAX_ATTACHMENT_CHARS:
        text = (text[:MAX_ATTACHMENT_CHARS]
                + f"\n[... '{name}' truncated at {MAX_ATTACHMENT_CHARS} characters]")
    return text


def _cell_color(fill: Any) -> Optional[str]:
    """A solid cell fill as a color string, or None for no fill."""
    if fill is None or getattr(fill, "fill_type", None) != "solid":
        return None
    col = fill.fgColor
    if getattr(col, "type", None) == "rgb" and isinstance(col.rgb, str):
        return None if col.rgb in ("00000000",) else col.rgb
    if getattr(col, "type", None) == "theme":
        return f"theme{col.theme}" + (f"(tint {col.tint:+.2f})" if col.tint else "")
    return None


def _render_xlsx(path: str) -> str:
    """Every sheet as rows of values, plus the cells' fill colors.

    Several GAIA spreadsheets encode the task in cell colors ("Green cells are plots owned
    by ..."), so the colors are part of the content, not decoration.
    """
    try:
        import openpyxl
    except ImportError as e:
        raise AttachmentDependencyError("openpyxl (for .xlsx attachments)") from e
    from openpyxl.utils import get_column_letter
    wb = openpyxl.load_workbook(path, data_only=True)
    out = []
    for ws in wb.worksheets:
        out.append(f"Sheet: {ws.title} ({ws.max_row} rows x {ws.max_column} columns)")
        fills: Dict[str, List[str]] = {}
        for r, row in enumerate(ws.iter_rows(), start=1):
            vals = ["" if c.value is None else str(c.value) for c in row]
            out.append(",".join(vals))
            for ci, c in enumerate(row, start=1):
                color = _cell_color(c.fill)
                if color:
                    fills.setdefault(color, []).append(f"{get_column_letter(ci)}{r}")
        if fills:
            out.append("Cell fill colors:")
            out.extend(f"  {color}: {', '.join(cells)}" for color, cells in fills.items())
    return "\n".join(out)


def _render_xls(path: str) -> str:
    try:
        import xlrd  # noqa: F401  (pandas' engine for legacy .xls)
        import pandas as pd
    except ImportError as e:
        raise AttachmentDependencyError("xlrd (for legacy .xls attachments)") from e
    sheets = pd.read_excel(path, sheet_name=None, engine="xlrd")
    return "\n".join(f"Sheet: {name} ({df.shape[0]} rows x {df.shape[1]} columns)\n"
                     + df.to_csv(index=False) for name, df in sheets.items())


def _render_pdf(path: str) -> str:
    try:
        import pymupdf
    except ImportError as e:
        raise AttachmentDependencyError("pymupdf (for .pdf attachments)") from e
    doc = pymupdf.open(path)
    return "\n\n".join(f"[page {i + 1}]\n{doc[i].get_text()}" for i in range(doc.page_count))


def _render_docx(path: str) -> str:
    """Paragraphs and tables, in document order, from word/document.xml (stdlib only)."""
    body = ElementTree.fromstring(zipfile.ZipFile(path).read("word/document.xml")).find(f"{_W}body")
    lines: List[str] = []
    for el in list(body) if body is not None else []:
        if el.tag == f"{_W}p":
            lines.append("".join(t.text or "" for t in el.iter(f"{_W}t")))
        elif el.tag == f"{_W}tbl":
            for tr in el.iter(f"{_W}tr"):
                lines.append(" | ".join("".join(t.text or "" for t in tc.iter(f"{_W}t"))
                                        for tc in tr.iter(f"{_W}tc")))
    return "\n".join(lines)


def _render_pptx(path: str) -> str:
    """Slide text in slide order, from ppt/slides/slideN.xml (stdlib only)."""
    zf = zipfile.ZipFile(path)
    slides = sorted((n for n in zf.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                    key=lambda n: int(re.findall(r"\d+", n)[-1]))
    out = []
    for i, name in enumerate(slides, start=1):
        root = ElementTree.fromstring(zf.read(name))
        paras = ["".join(t.text or "" for t in p.iter(f"{_A}t")) for p in root.iter(f"{_A}p")]
        out.append(f"Slide {i}:\n" + "\n".join(p for p in paras if p.strip()))
    return "\n\n".join(out)


_RENDERERS = {"xlsx": _render_xlsx, "xlsm": _render_xlsx, "xls": _render_xls,
              "pdf": _render_pdf, "docx": _render_docx, "pptx": _render_pptx}


def _file_parts(name: str, data: bytes) -> List[Dict[str, Any]]:
    """Content parts for one attached file (a zip expands to its members)."""
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    header = {"type": "text", "text": f"\n\nAttached file: {name}"}
    if ext in _IMAGE_MIME:
        b64 = base64.b64encode(data).decode("ascii")
        return [header, {"type": "image_url",
                         "image_url": {"url": f"data:{_IMAGE_MIME[ext]};base64,{b64}"}}]
    if ext in _AUDIO_FORMATS or ext in _TRANSCODED_AUDIO:
        fmt = _AUDIO_FORMATS.get(ext)
        if fmt is None:
            data, fmt = _audio_to_wav(data, name), "wav"
        return [header, {"type": "input_audio",
                         "input_audio": {"data": base64.b64encode(data).decode("ascii"),
                                         "format": fmt}}]
    if ext in _VIDEO_MIME:
        b64 = base64.b64encode(data).decode("ascii")
        return [header, {"type": "video_url",
                         "video_url": {"url": f"data:{_VIDEO_MIME[ext]};base64,{b64}"}}]
    if ext == "zip":
        import io
        zf = zipfile.ZipFile(io.BytesIO(data))
        members = [i for i in zf.infolist() if not i.is_dir()]
        parts: List[Dict[str, Any]] = [{"type": "text", "text": (
            f"\n\nAttached file: {name} (zip archive with {len(members)} file(s), "
            f"each attached below)")}]
        for m in members:
            parts.extend(_file_parts(f"{name}/{m.filename}", zf.read(m)))
        return parts
    if ext in _TEXT_EXTS:
        text = data.decode("utf-8", errors="replace")
    elif ext in _RENDERERS:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix="." + ext) as tmp:
            tmp.write(data)
            tmp.flush()
            text = _RENDERERS[ext](tmp.name)
    else:
        raise RuntimeError(f"gaia: no renderer for attachment '{name}' (.{ext}); refusing to "
                           "send the task without its file")
    return [{"type": "text", "text": f"\n\nAttached file: {name}\n{_clip(text, name)}"}]


def attachment_parts(path: str, name: str) -> List[Dict[str, Any]]:
    """The attachment at `path`, as chat content parts to append after the question."""
    with open(path, "rb") as fh:
        return _file_parts(name, fh.read())


def gaia_split() -> str:
    """The split to run: `GBENCH_GAIA_SPLIT` (validation or test), default validation."""
    raw = (os.environ.get("GBENCH_GAIA_SPLIT") or "validation").strip().lower()
    if raw not in SPLITS:
        raise ValueError(f"GBENCH_GAIA_SPLIT={os.environ.get('GBENCH_GAIA_SPLIT')!r} is not a GAIA "
                         f"split; use one of {', '.join(SPLITS)}")
    return raw


def _load_gaia_samples(
    limit: Optional[int] = None,
    split: str = "validation",
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """GAIA 2023 `split` at the pinned revision, attachments included; raises on failure.

    The test split's `Final answer` is "?" on every row (the answers are hidden)."""
    if split not in SPLITS:
        raise ValueError(f"gaia: unknown split {split!r}")
    tok = _hf_token()
    try:
        import pandas as pd
        from huggingface_hub import snapshot_download
        data_dir = snapshot_download(repo_id=DATASET_REPO, repo_type="dataset",
                                     revision=DATASET_REVISION, token=tok,
                                     allow_patterns=[f"2023/{split}/*"])
        rows = pd.read_parquet(os.path.join(data_dir, "2023", split,
                                            "metadata.parquet")).to_dict("records")
    except Exception as e:
        logger.error(f"Failed to load dataset for gaia: {e}")
        raise RuntimeError(f"Could not load dataset for gaia: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for gaia returned empty rows")
    # --shard first, over the full ordered set (base.apply_shard), then a subset stratified by
    # Level within the shard, not a contiguous head.
    rows, _shard_label = apply_shard(rows, "gaia")
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("Level"), seed="gaia")

    samples = []
    for item in rows:
        question = item.get("Question")
        gold = item.get("Final answer")  # exact key: capital F, space
        if not question or gold is None:
            raise RuntimeError(
                "gaia: unexpected dataset schema (missing 'Question'/'Final answer'); "
                "refusing to fabricate sample data"
            )
        file_name = str(item.get("file_name") or "")
        content: Any = str(question)
        if file_name:
            local = os.path.join(data_dir, str(item.get("file_path") or ""))
            if not item.get("file_path") or not os.path.isfile(local):
                raise RuntimeError(f"gaia: attachment {file_name} is missing from the "
                                   f"snapshot at {data_dir}; re-download the dataset")
            try:
                parts = attachment_parts(local, file_name)
            except AttachmentDependencyError as e:
                raise infra_required(
                    "gaia", f"rendering the task attachments needs the Python package {e} "
                    "in the gbench environment", DOCS_URL) from e
            content = [{"type": "text", "text": str(question)}] + parts
        messages = [
            {"role": "system", "content": _GAIA_SYSTEM},
            {"role": "user", "content": content},
        ]
        samples.append((messages, str(gold), {
            "category": f"Level_{item.get('Level', '1')}",
            "task_id": str(item.get("task_id") or ""),
            "has_file": bool(file_name),
            "attachment_type": (file_name.rsplit(".", 1)[-1].lower()
                                if "." in file_name else None),
        }))

    n_file = sum(1 for _, _, m in samples if m.get("has_file"))
    logger.info(
        f"Loaded {len(samples)} gaia {split} samples ({DATASET_REPO}@{DATASET_REVISION[:12]}); "
        f"{n_file} carry their attached file in the prompt.")
    return samples


def run_gaia(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run GAIA with the canonical question_scorer (hard-errors without HF gated access / search).

    `GBENCH_GAIA_SPLIT=test` runs the leaderboard's test split instead: it writes the submission
    file and ends with infra_required, since only the leaderboard can score it."""
    split = gaia_split()
    # No-skip: missing HF gated dataset access is missing infra -> hard-error, never a skip row.
    ok, reason = check_gaia_prerequisites()
    if not ok:
        raise infra_required("gaia", reason, DOCS_URL)
    # gaia is a web-research benchmark: without a lookup tool the model answers "I do not have
    # access to the internet" and scores a structural 0 (measured 0/20 on 2026-08-15), so the
    # search backend (GEMINI_API_KEY) is a REQUIRED prerequisite. Check it BEFORE loading the
    # gated dataset so a keyless run hard-errors immediately instead of downloading first.
    if not search_available():
        raise infra_required(
            "gaia",
            "requires a search backend (GEMINI_API_KEYS or GEMINI_API_KEY) - GAIA is a "
            "web-research benchmark and without a lookup tool every answer scores a structural 0",
            DOCS_URL)
    # Live check so a present-but-rejected key, or a cascade model the API does not serve, fails
    # here, not after a full generation run in which every search failed (each such sample is a
    # harness error, so the run would end with no score). An inconclusive probe passes.
    _ok, _why = gemini_key_live_valid(",".join(gemini_api_keys()))
    if not _ok:
        if str(_why).startswith(MODELS_NOT_SERVED_PREFIX):
            raise infra_required(
                "gaia", f"{_why}; every model of the Gemini cascade must be one the API serves "
                "for the web_search tool's Google-Search grounding.", DOCS_URL)
        raise infra_required(
            "gaia",
            f"the Gemini key (GEMINI_API_KEYS / GEMINI_API_KEY) was rejected by the search "
            f"endpoint ({_why}); a valid key is required for the web_search tool's Google-Search "
            "grounding.", DOCS_URL)
    samples = (_load_gaia_samples(limit=kwargs.get("limit")) if split == "validation"
               else _load_gaia_samples(limit=kwargs.get("limit"), split=split))
    reset_backend_tally()   # per-run, not per-process (a sweep runs many search suites)
    reset_tool_transcripts()
    extra_payload = dict(kwargs.get("extra_payload") or {})
    extra_payload.setdefault("tools", [GROUNDED_SEARCH_TOOL])
    if split == "test":
        return _run_test_split(model_name, base_url, concurrency, enable_thinking, samples,
                               extra_payload, **kwargs)

    result = run_eval_suite(
        eval_name="gaia",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=presharded_for_base(samples),  # this shard's rows, kept as they are
        eval_fn=_eval_gaia,
        thinking=enable_thinking,
        extra_payload=extra_payload,
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
        tool_executor=record_web_search,
        tool_loop=GAIA_TOOL_LOOP,
        official_sampling=GAIA_SAMPLING,
    )
    result["dataset"] = f"{DATASET_REPO}@{DATASET_REVISION} (2023_all validation)"
    result["split"] = "validation"
    result["search_backend"] = search_backend_name()
    result["search_backend_kind"] = SEARCH_BACKEND_KIND
    result["search_backend_calls"] = backend_tally()
    result["search_tool_description"] = GROUNDED_SEARCH_DESCRIPTION
    traces = result.get("sample_traces", []) or []
    harness = _exclude_harness_errors(result, traces)
    cut_reason = record_budget_cutoffs(result)
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = (
        "validation split (public answers) scored locally with the official question_scorer; the "
        "leaderboard ranks the hidden-answer test split (GBENCH_GAIA_SPLIT=test writes its "
        "submission file); the model's only web access is gbench's web_search tool, whose "
        "results are Gemini-written: each call is a Gemini model grounded in Google Search "
        "answering the model's query, and it returns the cited sources' titles and URLs with "
        "segments of that Gemini answer, no page text, so the score partly reflects the Gemini "
        "model's research; not a browsing agent with page fetch and code execution; "
        "attachments are inlined (text renders, images, audio)")
    official = result.get("official_sampling") or {}
    if official.get("temperature_matches_official") is False:
        result["leaderboard_comparable_reason"] += (
            f"; temperature {result.get('temperature')} set by "
            f"{str(result.get('temperature_source') or '').split(':', 1)[-1]} (the reference "
            "agent sends none)")
    if cut_reason:
        result["leaderboard_comparable_reason"] += f"; {cut_reason}"
    if harness:
        result["leaderboard_comparable_reason"] += "; " + harness_error_reason(harness)
    # Accuracy with and without an attached file, and per attachment type: the file tasks
    # also test document/vision/audio reading, which a text-only endpoint cannot do. Samples
    # that ended in a harness error are left out, as from the headline.
    scored = [t for t in traces if not t.get("scoring_excluded")]
    free = [t for t in scored if not (t.get("extra_payload") or {}).get("has_file")]
    with_file = [t for t in scored if (t.get("extra_payload") or {}).get("has_file")]
    for key, group in (("file_free", free), ("with_file", with_file)):
        if group:
            result[f"accuracy_{key}"] = round(
                100.0 * sum(1 for t in group if t.get("is_correct")) / len(group), 2)
    result["file_free_n"] = len(free)
    result["file_dependent_n"] = len(with_file)
    by_type: Dict[str, List[bool]] = {}
    for t in with_file:
        by_type.setdefault((t.get("extra_payload") or {}).get("attachment_type") or "?",
                           []).append(bool(t.get("is_correct")))
    result["accuracy_by_attachment_type"] = {
        k: {"n": len(v), "accuracy": round(100.0 * sum(v) / len(v), 2)}
        for k, v in sorted(by_type.items())}
    return result


def _exclude_harness_errors(result: Dict[str, Any], traces: List[Dict[str, Any]]
                            ) -> Dict[str, int]:
    """Attach each sample's web_search transcript, mark the samples that ended in a harness error
    (web_research_harness_error) and take them out of the headline, its numerator and denominator
    and the per-level table. Records `harness_errors` ({kind: samples}), `scoring_excluded` and
    `effective_n`; sets the status to completed_with_errors when any sample is out, and withholds
    the score (accuracy null, status failed) when none is left. Returns the counts."""
    attach_tool_transcripts(traces)
    for t in traces:
        err = web_research_harness_error(t)
        if err:
            t["harness_error"] = err
            t["scoring_excluded"] = True
            t["is_correct"] = False
    counts = harness_error_counts(traces)
    result["harness_errors"] = counts
    if not traces:
        return counts
    scored = [t for t in traces if not t.get("scoring_excluded")]
    result["scoring_excluded"] = len(traces) - len(scored)
    result["effective_n"] = len(scored)
    if not counts:
        return counts
    correct = sum(1 for t in scored if t.get("is_correct"))
    result["correct_answers"] = correct
    cats: Dict[str, Dict[str, int]] = {}
    for t in scored:
        if t.get("category"):
            st = cats.setdefault(t["category"], {"correct": 0, "total": 0})
            st["total"] += 1
            st["correct"] += int(bool(t.get("is_correct")))
    result["category_accuracy"] = {c: {**st, "accuracy": round(100.0 * st["correct"]
                                                               / st["total"], 2)}
                                   for c, st in sorted(cats.items())}
    why = harness_error_reason(counts)
    if scored:
        result["accuracy"] = round(100.0 * correct / len(scored), 2)
        if result.get("status") in (None, "success"):
            result["status"] = "completed_with_errors"
        result["status_reason"] = "; ".join(x for x in (result.get("status_reason"), why) if x)
    else:
        result["accuracy"] = None
        result["status"] = "failed"
        result["accuracy_withheld_reason"] = f"every sample ended in a harness error ({why})"
    return counts


# ---- The test split: the leaderboard submission file ---------------------------------------
def _submission_dir() -> str:
    """`<run results dir>/gaia_submission` (the GBENCH_RESULTS_DIR a gbench run exports), else a
    new directory in the system temp dir."""
    run_dir = (os.environ.get("GBENCH_RESULTS_DIR") or "").strip()
    if not run_dir:
        return tempfile.mkdtemp(prefix="gbench_gaia_submission_")
    path = os.path.join(run_dir, "gaia_submission")
    os.makedirs(path, exist_ok=True)
    return path


def reasoning_trace(trace: Dict[str, Any], steps: List[Tuple[str, Any, str]]) -> str:
    """The steps by which the model reached its answer: the tool calls it made with their results,
    then its reasoning (when the server returned it) and its final response."""
    parts: List[str] = []
    for i, (name, args, out) in enumerate(steps, start=1):
        parts.append(f"Step {i}: {name}({json.dumps(args, ensure_ascii=False)})\n"
                     f"Result:\n{out}")
    if (trace.get("reasoning") or "").strip():
        parts.append("Reasoning:\n" + str(trace["reasoning"]).strip())
    parts.append("Response:\n" + str(trace.get("response_text") or "").strip())
    return "\n\n".join(parts)


def submission_lines(traces: List[Dict[str, Any]],
                     steps: Dict[int, List[Tuple[str, Any, str]]]) -> List[Dict[str, str]]:
    """One leaderboard line per task, in dataset order: `task_id`, `model_answer` (the text after
    the last FINAL ANSWER marker, the string the leaderboard's question_scorer grades; empty when
    the model gave none), `reasoning_trace`."""
    lines = []
    for t in sorted(traces, key=lambda x: x["sample_idx"]):
        task_id = str((t.get("extra_payload") or {}).get("task_id") or "")
        if not task_id:
            raise RuntimeError("gaia: a test-split trace carries no task_id; refusing to write a "
                               "submission the leaderboard would reject")
        answer = _extract_final_answer(strip_thinking_tags(t.get("response_text") or ""))
        lines.append({"task_id": task_id, "model_answer": answer,
                      "reasoning_trace": reasoning_trace(t, steps.get(t["sample_idx"], []))})
    return lines


def _run_test_split(model_name: str, base_url: str, concurrency: int, enable_thinking: bool,
                    samples: List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]],
                    extra_payload: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Run the test split with the validation protocol, write the submission, then raise."""
    requested = kwargs.get("attempt_count")
    if requested not in (None, 1):
        logger.warning("[gaia] --attempt-count=%s is not applied to the test split: a leaderboard "
                       "submission holds one answer per task.", requested)
    result = run_eval_suite(
        eval_name="gaia",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=presharded_for_base(samples),  # this shard's rows, kept as they are
        # The answers are hidden ("?"), so nothing is scored here.
        eval_fn=lambda response, gold: False,
        thinking=enable_thinking,
        extra_payload=extra_payload,
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
        tool_executor=record_web_search,
        tool_loop=GAIA_TOOL_LOOP,
        attempt_count=1,
        official_sampling=GAIA_SAMPLING,
    )
    cut_reason = record_budget_cutoffs(result)
    traces = result.get("sample_traces", [])
    attach_tool_transcripts(traces)
    steps = {t["sample_idx"]: [(c["tool"], c["arguments"], c["reply"])
                               for c in t.get("tool_transcript") or []]
             for t in traces if "sample_idx" in t}
    lines = submission_lines(traces, steps)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    tag = "_".join((re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("._-") or "model",
                    "think" if enable_thinking else "nothink", stamp))
    out_dir = _submission_dir()
    sub_path = os.path.join(out_dir, f"gaia_2023_test_{tag}.jsonl")
    with open(sub_path, "w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    levels: Dict[int, int] = {}
    for t in traces:
        cat = str(t.get("category") or "")
        if cat.startswith("Level_") and cat[6:].isdigit():
            levels[int(cat[6:])] = levels.get(int(cat[6:]), 0) + 1
    expected = LEADERBOARD_LEVEL_COUNTS["test"]
    # A prompt that does not fit the served context window is refused before it is sent: a re-run
    # does not help, a larger --max-model-len on the server does. Other failed requests can be
    # re-run.
    over_context = [t for t in traces if t.get("response_text") is None
                    and t.get("status") == "OVER_CONTEXT"]
    failed = [t for t in traces if t.get("response_text") is None
              and t.get("status") != "OVER_CONTEXT"]
    no_answer = sum(1 for line in lines if not line["model_answer"])
    # a task whose lookup the Gemini search backend could not answer was answered without it
    search_failed = sorted(str((t.get("extra_payload") or {}).get("task_id") or "")
                           for t in traces if any(c.get("search_failed")
                                                  for c in t.get("tool_transcript") or []))
    for key in ("accuracy", "correct_answers", "category_accuracy"):
        result.pop(key, None)
    result.update({
        "dataset": f"{DATASET_REPO}@{DATASET_REVISION} (2023_all test)",
        "split": "test",
        "search_backend": search_backend_name(),
        "search_backend_kind": SEARCH_BACKEND_KIND,
        "search_backend_calls": backend_tally(),
        "search_tool_description": GROUNDED_SEARCH_DESCRIPTION,
        "leaderboard_submission": {"path": sub_path, "tasks": len(lines),
                                   "tasks_per_level": dict(sorted(levels.items())),
                                   "tasks_without_final_answer": no_answer,
                                   "failed_requests": len(failed),
                                   "replies_cut_off_by_budget":
                                       result["replies_cut_off_by_budget"],
                                   "search_backend_failure_tasks": search_failed,
                                   "over_context_tasks": sorted(
                                       str((t.get("extra_payload") or {}).get("task_id") or "")
                                       for t in over_context),
                                   "upload": LEADERBOARD_URL},
        "leaderboard_comparable": False,
    })
    traces_path = os.path.join(out_dir, f"gaia_2023_test_{tag}_result.json")
    with open(traces_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1, default=str)
    notes = []
    if levels != expected:
        notes.append(f"this file holds {len(lines)} task(s) {dict(sorted(levels.items()))} per "
                     f"level; the leaderboard accepts only the full split "
                     f"{dict(expected)}, so it will reject this subset")
    if over_context:
        notes.append(f"{len(over_context)} task(s) do not fit the served model's context "
                     f"window and have an empty model_answer (over_context_tasks); serve the "
                     f"model with a larger --max-model-len and re-run before uploading")
    if failed:
        notes.append(f"{len(failed)} request(s) failed and have an empty model_answer; re-run "
                     "before uploading")
    if search_failed:
        notes.append(f"{len(search_failed)} task(s) had a web_search call the Gemini search "
                     "backend could not answer (search_backend_failure_tasks), so their answers "
                     "were given without that lookup; re-run them before uploading")
    if cut_reason:
        notes.append(cut_reason)
    raise infra_required(
        "gaia",
        f"the GAIA test split's answers are hidden, so gbench cannot score it. It wrote the "
        f"leaderboard submission {sub_path} ({len(lines)} line(s) of task_id, model_answer, "
        f"reasoning_trace) and the run's traces {traces_path}. The score comes from uploading the "
        f"submission at {LEADERBOARD_URL}, which grades model_answer with the same "
        f"question_scorer" + ("; " + "; ".join(notes) if notes else "") + ".",
        DOCS_URL)
