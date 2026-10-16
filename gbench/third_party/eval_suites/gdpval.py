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

# Evaluation Suite: gdpval
# Description: GDPval (OpenAI Real Economic Knowledge-Work Deliverables Benchmark)

"""gbench native built-in runner for gdpval (Economic & Knowledge Work).

GDPval: 220 economically-valuable knowledge-work tasks (the public "gold" subset). 125 of
them (57%) take an attached spreadsheet/PDF/document as input; every task has a long grading
rubric, and 185 ship the EXPERT REFERENCE DELIVERABLE (`deliverable_files`; the other 35 ship
none, so they cannot enter the pairwise rate).

**Metric: GDPval's win rate = the wins-or-ties rate.** GDPval is scored by comparing a model's
deliverable against the human-expert reference in a blind pairwise judgement (graders output
{0, 0.5, 1}: loss / tie / win), and its headline is the share of tasks where the model's
deliverable was rated "better than (wins) or as good as (ties)" the expert's (arXiv:2510.04374,
Section 3.1 and Figure 5), with wins-only reported alongside. gbench reproduces that: each
generation's answer and the expert reference are shown to the judge as A/B (order arbitrary),
judged against the rubric, position-swapped to cancel order bias, and the two verdicts are
averaged. A comparison is a win when that average is above 0.5, a tie at exactly 0.5, a loss
below. The headline is (wins + ties) / scored comparisons, pooled over every generation of every
task; the wins-only rate and the tie-0.5 mean are secondary metrics. The judge is
gbench's standard Gemini cascade (a gbench convention); OpenAI's automated grader is a trained
model served at evals.openai.com whose prompt is not public, so the pairwise prompt is gbench's.

**This is not OpenAI's GDPval number and says so on every result** (`leaderboard_comparable`
False): a text `/v1` endpoint cannot emit an `.xlsx`/`.docx`/`.pdf`, so the expert reference
FILE is rendered to text and the model competes on rendered content only - it can never match
the file's live formulas/formatting. A task whose reference cannot be rendered to text, or a
comparison without a usable judge verdict (no answer, or a reply without a valid winner), is
excluded from the win-rate (not counted as a loss, never as a tie). A generation whose request
failed (no reply at all: a server error, a request timeout or a prompt over the context window,
base status FAILED / TIMEOUT / OVER_CONTEXT) is a harness error, not the model's deliverable: it
is excluded too, counted in `comparisons_generation_failed` by status and named in
`leaderboard_comparable_reason`. An empty deliverable the model did return IS counted, as a loss.
When no comparison can be scored, the run fails with its score withheld (accuracy null and the
reason) instead of reporting 0. A full run keeps every task; an --eval-limit subset is drawn only
from tasks whose reference renders to text (known before generation; _load_gdpval_samples), and
the draw is recorded in `eval_limit_draw`. Every comparison records both judge verdicts with the
Gemini model that gave each and its reply (`pairwise_detail`). Like the paper's headline plots
(footnote 2), every task is sampled 3 times and the win rate pools all samples' comparisons; the
paper then has 3 human experts grade each sample, where gbench's judge grades each sample once.

Attachments are fetched with `hf_hub_download` and converted per type; `load_dataset` returns
only the file PATHS. Images go to the model as image parts and audio files as input_audio
parts (as gaia sends them); a .zip is opened and its members rendered (text and code as text,
documents through the same converters, images and audio as parts of an input). A missing
converter package or a failed download is a hard error (infra_required); only formats with no
text, image or audio rendering (video, CAD, .psd) reach the model as a note. See
docs/evals/gdpval.md.

The judge is kept as blind as a text rendering allows: the rendered reference carries no label
naming it the expert's (render_reference), but rendering artifacts (file, sheet and slide
headers, truncation markers) can still set it apart from a prose answer, and the result says so.
The judge sees the task's text with the text rendering of its input files (the model's user
message without its image and audio parts; comparisons whose task had such parts are counted),
the rubric and the two deliverables. Clipping, which the paper does not do, is recorded: an input file rendered past GBENCH_GDPVAL_MAX_FILE_CHARS (20,000) or a PDF
past GBENCH_GDPVAL_MAX_PDF_PAGES (10) is listed in the trace's `inputs_clipped`; a rubric past
GBENCH_GDPVAL_MAX_RUBRIC_CHARS (8,000, cut at a criterion boundary with a visible marker) and a
reference, model deliverable or task prompt past GBENCH_GDPVAL_MAX_JUDGE_CHARS (24,000) in
`judge_input_clipped`; gdpval_report totals them and leaderboard_comparable_reason names them.

Sampling: the paper names no sampling parameters (models were sampled 3 times per task through
their own products and APIs). gbench always sends the temperature of its rule
(base.resolve_temperature): 1.0 with `--thinking` (for gemma-4 the same value as its default,
sent explicitly), 0.0 without, where the 3 samples are greedy and differ only through server
nondeterminism; it is recorded under `official_sampling`. `--temperature`, or
`GBENCH_GDPVAL_TEMPERATURE` for
this suite alone (it takes precedence), sends another value instead and the result records it under
`official_sampling`. LLM-judge grading is pinned at 0.0 and is not affected by either. No
max_tokens is sent either, as the paper sets none (the token-limit policy's "not sent" case): each
generation may use the room the context window leaves, in both modes, `--max-output-tokens` is
recorded as not applied in `max_output_tokens_note`, and a generation cut off there is counted in
`truncated_responses`.
"""

import base64
import contextlib
import contextvars
import importlib
import json
import logging
import os
import re
import struct
import tempfile
import zipfile
import zlib
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import (OfficialSampling, apply_shard, finalize_shard_fields, gemini_required_skip,
                   judge_cascade, judge_generate_cascade, run_eval_suite, strip_thinking_tags)
from .sampling import stratified_sample, presharded_for_base
from .fc_common import base_suite_plan
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

PILLAR = "Economic & Knowledge Work"
DOCS_URL = "docs/evals/gdpval.md"
_REPO = "openai/gdpval"
#: Pinned dataset commit (HF `openai/gdpval` main as of 2026-09-24), used for the task table and
#: every attachment / deliverable download so a run is reproducible.
_REVISION = "11e7900cdcac61bc4daf59e65feb238acda98fbf"

#: Attachment budget per file. GDPval inputs run to hundreds of KB of spreadsheet; without a
#: cap a single task can blow the context window on one CSV dump.
_MAX_CHARS_PER_FILE = int(os.getenv("GBENCH_GDPVAL_MAX_FILE_CHARS", "20000"))
_MAX_PDF_PAGES = int(os.getenv("GBENCH_GDPVAL_MAX_PDF_PAGES", "10"))

#: Per-deliverable clip handed to the pairwise judge (model answer and rendered reference each),
#: and the rubric clip. Keeps a single pairwise prompt inside a sane context budget.
_MAX_JUDGE_CHARS = int(os.getenv("GBENCH_GDPVAL_MAX_JUDGE_CHARS", "24000"))
_MAX_RUBRIC_CHARS = int(os.getenv("GBENCH_GDPVAL_MAX_RUBRIC_CHARS", "8000"))

#: The clip events of the rendering in progress (collect_clips), or None when nobody collects.
_CLIPS: "contextvars.ContextVar[Optional[List[Dict[str, Any]]]]" = contextvars.ContextVar(
    "gdpval_clips", default=None)


@contextlib.contextmanager
def collect_clips():
    """Collect every clip the renderers make inside the block: a list of {"file", "cut"
    ("chars" or "pdf_pages"), "full", "kept"}, one per clipped file (archive members included)."""
    log: List[Dict[str, Any]] = []
    token = _CLIPS.set(log)
    try:
        yield log
    finally:
        _CLIPS.reset(token)


def _note_clip(name: str, cut: str, full: int, kept: int) -> None:
    log = _CLIPS.get()
    if log is not None:
        log.append({"file": name, "cut": cut, "full": int(full), "kept": int(kept)})


def clip_limits() -> Dict[str, int]:
    """The caps this run clips at (each has its GBENCH_GDPVAL_* knob)."""
    return {"max_file_chars": _MAX_CHARS_PER_FILE, "max_pdf_pages": _MAX_PDF_PAGES,
            "max_judge_chars": _MAX_JUDGE_CHARS, "max_rubric_chars": _MAX_RUBRIC_CHARS}


# --------------------------------------------------------------------------- #
# attachments
# --------------------------------------------------------------------------- #
_IMAGE_EXTS = ("png", "jpg", "jpeg", "webp", "gif", "bmp")
_AUDIO_EXTS = ("wav", "mp3", "m4a", "flac")
#: Plain-text formats, passed as their own text (the .py/.yaml/.overpassql/.ipynb deliverables
#: are code and query files an expert handed in).
_TEXT_EXTS = ("txt", "csv", "tsv", "md", "json", "jsonl", "py", "yaml", "yml", "xml", "html",
              "overpassql")
#: The Python package each converter needs: (import name, pip name). A missing one is missing
#: infrastructure (infra_required), never a silently unreadable file.
_CONVERTER_PACKAGES = {
    "xlsx": ("openpyxl", "openpyxl"), "xlsm": ("openpyxl", "openpyxl"),
    "pdf": ("pymupdf", "pymupdf"),
    "docx": ("docx", "python-docx"),
    "pptx": ("pptx", "python-pptx"),
    **{ext: ("soundfile", "soundfile") for ext in _AUDIO_EXTS},
}
_RENDERABLE_EXTS = (set(_CONVERTER_PACKAGES) | set(_TEXT_EXTS) | set(_IMAGE_EXTS)
                    | {"ipynb", "zip"})
#: Resolution for a PDF page with no text layer (a scan, or text drawn as vector paths), which
#: is sent to the model as page images instead of text.
_PDF_PAGE_DPI = int(os.getenv("GBENCH_GDPVAL_PDF_PAGE_DPI", "150"))

#: Archive members rendered as text, by extension (or by the whole lower-cased file name for
#: extensionless files such as a Dockerfile). The gdpval archives are source-code projects
#: (TypeScript, Solidity, Terraform, ...), a set of photographs, audio stems and CAD files.
_ARCHIVE_TEXT_EXTS = frozenset(_TEXT_EXTS) | {
    "ts", "tsx", "js", "jsx", "mjs", "cjs", "css", "scss", "less", "sol", "tf", "tfvars", "hcl",
    "sh", "bash", "zsh", "ps1", "bat", "sql", "java", "kt", "scala", "c", "h", "cc", "cpp", "hpp",
    "cs", "go", "rs", "rb", "php", "swift", "r", "m", "lua", "pl", "toml", "ini", "cfg", "conf",
    "env", "properties", "gradle", "vue", "svelte", "graphql", "gql", "proto", "rst", "tex", "svg",
    "gitignore", "dockerignore", "editorconfig", "npmrc", "dockerfile", "makefile", "license",
    "readme", "procfile"}
#: Archive members that are OS metadata, version control, installed dependencies, build caches or
#: generated lockfiles rather than authored content: counted in the archive summary, not rendered.
_ARCHIVE_SKIP_DIRS = frozenset({"__MACOSX", ".git", "node_modules", "__pycache__", ".next",
                                ".venv", "venv"})
_ARCHIVE_SKIP_FILES = frozenset({".ds_store", "thumbs.db", "package-lock.json", "yarn.lock",
                                 "pnpm-lock.yaml"})
#: Archive members converted with the same converters as a top-level file.
_ARCHIVE_DOC_EXTS = frozenset({"xlsx", "xlsm", "pdf", "docx", "pptx", "ipynb"})


def _require_converter(ext: str) -> None:
    """Hard-error (infra_required) when the package that renders `.ext` files is missing."""
    mod_pip = _CONVERTER_PACKAGES.get(ext)
    if not mod_pip:
        return
    try:
        importlib.import_module(mod_pip[0])
    except ImportError as e:
        raise infra_required(
            "gdpval", f"rendering the .{ext} files of the selected tasks needs the Python "
            f"package {mod_pip[1]} (`pip install {mod_pip[1]}`), which is not importable "
            f"({type(e).__name__}: {e})", DOCS_URL) from e


def _unrenderable_note(name: str, ext: str) -> str:
    # mp4 video, .step CAD, .psd and anything else with no text or image projection.
    return (f"[attachment '{name}' is a .{ext} file, which this suite cannot render; "
            f"its content is unavailable]")


def _fetch(rel_path: str) -> str:
    """Resolve one dataset-relative attachment path to a local file (cached).

    A failed download is missing infrastructure: the task would otherwise be sent without its
    input, or scored without its reference, so it hard-errors naming the file.
    """
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=_REPO, filename=rel_path, repo_type="dataset",
                               revision=_REVISION)
    except Exception as e:                                          # noqa: BLE001
        raise infra_required(
            "gdpval", f"could not download {_REPO}@{_REVISION[:12]} file '{rel_path}' "
            f"({type(e).__name__}: {e}); check HF Hub access or pre-fetch the files",
            DOCS_URL) from e


def render_attachment(rel_path: str) -> Tuple[Optional[str], Optional[str]]:
    """Convert one attachment to text. Returns (text, note).

    Exactly one of the two is set. `note` carries the reason a file has no text rendering
    (a format this suite cannot render, audio, a PDF with no text layer), which is attached to
    the prompt verbatim - a task whose input silently went missing would look like a model
    failure, which is the thing this suite exists to avoid. A missing converter package or a
    failed download raises infra_required; a file of a supported format that fails to parse
    raises too. Images are handled separately by :func:`attachment_parts`.
    """
    name = os.path.basename(rel_path)
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in _RENDERABLE_EXTS:
        return None, _unrenderable_note(name, ext)      # nothing to download for it
    _require_converter(ext)
    return render_local_file(_fetch(rel_path), name)


def _render_ipynb(local: str) -> str:
    """Cell sources and text outputs of a notebook, in order (image outputs are counted)."""
    with open(local, encoding="utf-8") as fh:
        nb = json.load(fh)
    out = []
    for i, cell in enumerate(nb.get("cells") or [], 1):
        src = cell.get("source") or ""
        out.append(f"### cell {i} ({cell.get('cell_type', '?')})\n"
                   + ("".join(src) if isinstance(src, list) else str(src)))
        for o in cell.get("outputs") or []:
            data = o.get("data") or {}
            text = o.get("text") or data.get("text/plain")
            if text:
                out.append("[output]\n" + ("".join(text) if isinstance(text, list) else str(text)))
            images = [k for k in data if k.startswith("image/")]
            if images:
                out.append(f"[output: {len(images)} image(s) not rendered]")
    return "\n".join(out)


def _member_kind(member: str) -> str:
    """How an archive member is handled: 'skip' (metadata/dependency/lockfile), 'text',
    'document' (xlsx/pdf/docx/pptx/ipynb), 'image', 'audio' or 'other' (video, CAD, ...)."""
    parts = [p for p in member.split("/") if p]
    base = parts[-1] if parts else member
    if (any(p in _ARCHIVE_SKIP_DIRS for p in parts[:-1]) or base.startswith("._")
            or base.lower() in _ARCHIVE_SKIP_FILES):
        return "skip"
    ext = base.rsplit(".", 1)[-1].lower() if "." in base else base.lower()
    if ext in _ARCHIVE_TEXT_EXTS:
        return "text"
    if ext in _ARCHIVE_DOC_EXTS:
        return "document"
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _AUDIO_EXTS:
        return "audio"
    return "other"


def _walk_local_headers(data: bytes, want: Callable[[str], bool]
                        ) -> Tuple[List[Tuple[str, Optional[bytes]]], bool]:
    """Read a zip archive entry by entry from its local file headers.

    For an archive without a readable central directory (the upstream file was cut off: one
    gdpval deliverable is exactly 32 MiB with no end record), `zipfile` cannot open it, but
    every member before the break is intact. Returns ([(name, content or None)],
    stopped_early); content is read only when `want(name)`, and `stopped_early` is True when
    the walk stopped at a damaged or truncated entry.
    """
    out: List[Tuple[str, Optional[bytes]]] = []
    mv = memoryview(data)
    n, pos = len(data), 0
    while pos + 30 <= n and data[pos:pos + 4] == b"PK\x03\x04":
        (_sig, _ver, flag, method, _tm, _dt, _crc, csize, _usize, fnl,
         exl) = struct.unpack_from("<IHHHHHIIIHH", data, pos)
        name = bytes(mv[pos + 30:pos + 30 + fnl]).decode(
            "utf-8" if flag & 0x800 else "cp437", "replace")
        start = pos + 30 + fnl + exl
        if method not in (0, 8):
            return out, True
        if flag & 0x08:                     # sizes live in a data descriptor after the data
            if method == 8:
                dec, chunks, cur = zlib.decompressobj(-15), [], start
                while not dec.eof and cur < n:
                    piece = mv[cur:cur + 65536]
                    chunks.append(dec.decompress(piece))
                    cur += len(piece)
                if not dec.eof:
                    return out, True
                end = cur - len(dec.unused_data)
                content = b"".join(chunks)
            else:
                end = data.find(b"PK\x07\x08", start)
                if end < 0:
                    return out, True
                content = bytes(mv[start:end])
            end += 16 if data[end:end + 4] == b"PK\x07\x08" else 12
            if end > n:
                return out, True
        else:
            end = start + csize
            if end > n:
                return out, True
            content = None
            if want(name):
                raw = bytes(mv[start:end])
                content = zlib.decompress(raw, -15) if method == 8 else raw
        out.append((name, content if want(name) else None))
        pos = end
    return out, pos < n and data[pos:pos + 4] not in (b"PK\x01\x02", b"PK\x05\x06")


def _archive_entries(local: str, want: Callable[[str], bool]
                     ) -> Tuple[List[Tuple[str, Optional[bytes]]], Optional[str]]:
    """File members of a zip (directories dropped) with the content of the wanted ones, and a
    note when the archive had to be read without its central directory."""
    try:
        with zipfile.ZipFile(local) as zf:
            return [(i.filename, zf.read(i) if want(i.filename) else None)
                    for i in zf.infolist() if not i.is_dir()], None
    except zipfile.BadZipFile:
        with open(local, "rb") as fh:
            entries, stopped_early = _walk_local_headers(fh.read(), want)
        note = ("the archive has no readable central directory (damaged or truncated), so its "
                "members were read from their local headers"
                + (", up to the first damaged one" if stopped_early else ""))
        return [(m, c) for m, c in entries if not m.endswith("/")], note


def _render_archive(local: str, name: str, collect_media: bool = False
                    ) -> Tuple[Optional[str], str, List[Tuple[str, bytes]]]:
    """Render a zip's authored content: (text of its members or None, summary, media).

    Text and code members are decoded as UTF-8, and documents go through the same converters
    as a top-level file (a missing converter is infra_required). With `collect_media` (an
    input the model receives) image and audio members are returned for the caller to attach;
    otherwise (a reference for the text judge) they are counted. OS metadata, version control,
    installed dependencies, build caches and lockfiles are counted but not rendered, and
    formats with no rendering (video, CAD, binaries) are counted by extension. Members appear
    in archive order; the caller clips the result like any other file.
    """
    def want(member: str) -> bool:
        kind = _member_kind(member)
        return kind in ("text", "document") or (collect_media and kind in ("image", "audio"))

    try:
        entries, damaged = _archive_entries(local, want)
    except Exception as e:                                          # noqa: BLE001
        raise RuntimeError(
            f"gdpval: could not read '{name}' ({type(e).__name__}: {e}); refusing to send the "
            f"task without it or to score it without its reference (see {DOCS_URL})") from e
    for member, _ in entries:
        if _member_kind(member) == "document":
            _require_converter(member.rsplit(".", 1)[-1].lower())
    chunks: List[str] = []
    media: List[Tuple[str, bytes]] = []
    skipped = 0
    others: Dict[str, int] = {}
    for member, content in entries:
        kind = _member_kind(member)
        if kind == "skip":
            skipped += 1
        elif kind == "text" and content is not None:
            chunks.append(f"### {member}\n" + content.decode("utf-8", "replace"))
        elif kind == "document" and content is not None:
            ext = member.rsplit(".", 1)[-1].lower()
            with tempfile.TemporaryDirectory(prefix="gbench_gdpval_") as tmp:
                path = os.path.join(tmp, f"member.{ext}")
                with open(path, "wb") as fh:
                    fh.write(content)
                text, note = render_local_file(path, os.path.basename(member))
            chunks.append(f"### {member}\n" + (text or note or ""))
        elif kind in ("image", "audio") and content is not None:
            media.append((member, content))
        elif kind == "image":
            others["image"] = others.get("image", 0) + 1
        else:
            ext = member.rsplit(".", 1)[-1].lower() if "." in member.rsplit("/", 1)[-1] else "(none)"
            others[ext] = others.get(ext, 0) + 1
    rendered = len(chunks) + len(media)
    summary = [f"archive '{name}': {len(entries)} file(s), {rendered} rendered"]
    if skipped:
        summary.append(f"{skipped} OS/VCS/dependency/lockfile file(s) not rendered")
    if others:
        labels = {"image": "image(s)", "(none)": "file(s) without an extension"}
        summary.append("not renderable: " + ", ".join(
            f"{count} {labels.get(ext, '.' + ext)}" for ext, count in sorted(others.items())))
    if damaged:
        summary.append(damaged)
    return ("\n\n".join(chunks) if chunks else None), "; ".join(summary), media


def _media_part(member: str, content: bytes) -> Dict[str, Any]:
    """An image_url part (with the file's own mime type) or an input_audio part."""
    ext = member.rsplit(".", 1)[-1].lower() if "." in member else ""
    b64 = base64.b64encode(content).decode("ascii")
    if ext in _AUDIO_EXTS:
        return {"type": "input_audio", "input_audio": {"data": b64, "format": ext}}
    from .dataset_utils import _guess_mime
    return {"type": "image_url", "image_url": {"url": f"data:{_guess_mime(member)};base64,{b64}"}}


def render_local_file(local: str, name: str) -> Tuple[Optional[str], Optional[str]]:
    """Convert an already-materialised file to text. Returns (text, note).

    Split out of `render_attachment` so callers that already hold the BYTES can reuse the
    same converters. A text-gradeable plugin variant needs this: its attachments are embedded in the
    export rather than fetched by path, and until 2026-08-19 they were dropped entirely -
    the model was asked to work from .xlsx/.docx/.pdf documents it never received, and
    unsurprisingly replied that it could not produce the deliverable.
    """
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in _RENDERABLE_EXTS or ext in _IMAGE_EXTS:
        return None, _unrenderable_note(name, ext)
    _require_converter(ext)
    if ext == "zip":
        body, summary, _images = _render_archive(local, name)
        if not body:
            return None, f"[attachment '{name}' has no text member to render ({summary})]"
        return _clip(f"[{summary}]\n\n{body}", name), None
    try:
        if ext in ("xlsx", "xlsm"):
            import pandas as pd
            sheets = pd.read_excel(local, sheet_name=None, engine="openpyxl")
            out = []
            for sname, df in sheets.items():
                out.append(f"### sheet: {sname}  ({df.shape[0]} rows x {df.shape[1]} cols)\n"
                           + df.to_csv(index=False))
            text = "\n".join(out)
        elif ext == "pdf":
            import pymupdf
            doc = pymupdf.open(local)
            pages = [doc[i].get_text() for i in range(min(doc.page_count, _MAX_PDF_PAGES))]
            if not "".join(pages).strip():
                return None, (f"[attachment '{name}' is a PDF with no text layer "
                              f"({doc.page_count} page(s) of scanned or drawn content); "
                              f"not rendered to text]")
            extra = ("" if doc.page_count <= _MAX_PDF_PAGES
                     else f"\n[... {doc.page_count - _MAX_PDF_PAGES} further pages omitted]")
            if doc.page_count > _MAX_PDF_PAGES:
                _note_clip(name, "pdf_pages", doc.page_count, _MAX_PDF_PAGES)
            text = "\n\n".join(pages) + extra
        elif ext == "docx":
            try:
                import docx
                d = docx.Document(local)
                parts = [p.text for p in d.paragraphs if p.text.strip()]
                for t in d.tables:
                    parts.append("\n".join(" | ".join(c.text for c in row.cells)
                                           for row in t.rows))
                text = "\n".join(parts)
            except Exception:
                # python-docx raises XMLSyntaxError on some real-but-nonstandard docx
                # (measured 2026-08-20: 2 gdpval files). The raw word/document.xml is valid
                # enough to read - unzip it and strip tags so the model still gets the
                # content instead of "[could not be read]".
                import zipfile
                xml = zipfile.ZipFile(local).read("word/document.xml").decode(
                    "utf-8", "replace")
                text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", xml)).strip()
                if not text:
                    raise
        elif ext == "pptx":
            from pptx import Presentation
            pr = Presentation(local)
            slides = []
            for i, s in enumerate(pr.slides, 1):
                txt = "\n".join(sh.text for sh in s.shapes if hasattr(sh, "text") and sh.text)
                slides.append(f"### slide {i}\n{txt}")
            text = "\n".join(slides)
        elif ext == "ipynb":
            text = _render_ipynb(local)
        elif ext in _TEXT_EXTS:
            with open(local, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        else:                                               # audio
            import soundfile as sf
            i = sf.info(local)
            return None, (f"[attachment '{name}' is {i.duration:.0f}s of audio; this suite "
                          f"does not transcribe audio, so its content is unavailable]")
    except Exception as e:                                          # noqa: BLE001
        raise RuntimeError(
            f"gdpval: could not read '{name}' ({type(e).__name__}: {e}); refusing to send the "
            f"task without it or to score it without its reference (see {DOCS_URL})") from e
    text = _clip(text, name)
    if not text:
        return None, f"[attachment '{name}' contains no text]"
    return text, None


def _clip(text: str, name: str) -> str:
    text = (text or "").strip()
    if len(text) > _MAX_CHARS_PER_FILE:
        _note_clip(name, "chars", len(text), _MAX_CHARS_PER_FILE)
        text = text[:_MAX_CHARS_PER_FILE] + f"\n[... '{name}' truncated at {_MAX_CHARS_PER_FILE} chars]"
    return text


def _pdf_page_images(local: str, name: Optional[str] = None) -> List[str]:
    """PNG data URIs of the first _MAX_PDF_PAGES pages (for a PDF with no text layer)."""
    import base64
    import pymupdf
    doc = pymupdf.open(local)
    if doc.page_count > _MAX_PDF_PAGES:
        _note_clip(name or os.path.basename(local), "pdf_pages", doc.page_count,
                   _MAX_PDF_PAGES)
    return ["data:image/png;base64," + base64.b64encode(
                doc[i].get_pixmap(dpi=_PDF_PAGE_DPI).tobytes("png")).decode("ascii")
            for i in range(min(doc.page_count, _MAX_PDF_PAGES))]


def attachment_parts(rel_paths: List[str]) -> Tuple[List[Dict[str, Any]], int, int]:
    """Build the multimodal content parts for a task's attachments.

    Returns (parts, rendered, unavailable). Images, the pages of a PDF with no text layer and
    the images inside a .zip go in as image parts, audio files (also inside a .zip) as
    input_audio parts, as gaia sends them; everything else is rendered to text. `unavailable`
    counts the files the model gets only a note for (formats with no rendering, an archive with
    nothing renderable).
    """
    from .dataset_utils import to_data_uri
    parts: List[Dict[str, Any]] = []
    rendered = unavailable = 0
    for rel in rel_paths or []:
        name = os.path.basename(rel)
        ext = rel.rsplit(".", 1)[-1].lower() if "." in rel else ""
        if ext in _IMAGE_EXTS:
            uri = to_data_uri(_fetch(rel))
            if not uri:
                raise RuntimeError(f"gdpval: could not read image attachment '{name}'; refusing "
                                   f"to send the task without it (see {DOCS_URL})")
            parts.append({"type": "text", "text": f"\n--- attachment: {name} ---"})
            parts.append({"type": "image_url", "image_url": {"url": uri}})
            rendered += 1
            continue
        if ext in _AUDIO_EXTS:
            with open(_fetch(rel), "rb") as fh:
                data = fh.read()
            parts.append({"type": "text", "text": f"\n--- attachment: {name} ---"})
            parts.append(_media_part(name, data))
            rendered += 1
            continue
        if ext == "zip":
            body, summary, media = _render_archive(_fetch(rel), name, collect_media=True)
            if not body and not media:
                parts.append({"type": "text", "text": (
                    f"\n[attachment '{name}' has nothing this suite can render ({summary})]")})
                unavailable += 1
                continue
            parts.append({"type": "text", "text": f"\n--- attachment: {name} ({summary}) ---"
                          + (f"\n{_clip(body, name)}" if body else "")})
            for member, content in media:
                parts.append({"type": "text", "text": f"\n--- {name}: {member} ---"})
                parts.append(_media_part(member, content))
            rendered += 1
            continue
        if ext == "pdf":
            _require_converter(ext)
            local = _fetch(rel)
            text, note = render_local_file(local, name)
            if not text:
                with collect_clips() as page_clips:
                    pages = _pdf_page_images(local, name)
                for clip in page_clips:          # passed on to an outer collector
                    _note_clip(clip["file"], clip["cut"], clip["full"], clip["kept"])
                shown = (f"the first {len(pages)} of {page_clips[0]['full']} pages as images"
                         if page_clips else f"{len(pages)} page image(s)")
                parts.append({"type": "text", "text": (
                    f"\n--- attachment: {name} (no text layer; {shown}) ---")})
                parts.extend({"type": "image_url", "image_url": {"url": u}} for u in pages)
                rendered += 1
                continue
        else:
            text, note = render_attachment(rel)
        if text:
            parts.append({"type": "text", "text": f"\n--- attachment: {name} ---\n{text}"})
            rendered += 1
        else:
            parts.append({"type": "text", "text": "\n" + (note or f"[{name} unavailable]")})
            unavailable += 1
    return parts, rendered, unavailable


def render_reference(rel_paths: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Render the expert reference deliverable(s) to text for the pairwise comparison.

    Returns (text, note). The reference is a produced FILE (.xlsx/.docx/.pdf/...); it is
    rendered to text with the same converters the inputs use, so the model's prose answer is
    compared against the reference's rendered CONTENT. A reference with nothing the text judge
    can read (no deliverable shipped, images, a PDF with no text layer, video) yields
    (None, note) and the task is excluded from the win-rate rather than scored as a loss. A
    missing converter package or a failed download raises (infra_required) instead.

    The text carries no label naming it the expert's (the paper's comparisons are blind and its
    deliverables unlabeled). One rendered file is its text alone; several are separated by a
    `--- <file name> ---` line. The rendering can still set it apart from a prose answer (those
    headers, sheet and slide headers, truncation markers), which the result names.
    """
    if not rel_paths:
        return None, "[no reference deliverable shipped for this task]"
    chunks: List[Tuple[str, str]] = []
    notes: List[str] = []
    for rel in rel_paths:
        name = os.path.basename(rel)
        ext = rel.rsplit(".", 1)[-1].lower() if "." in rel else ""
        if ext in _IMAGE_EXTS:
            notes.append(f"[reference '{name}' is an image; not rendered to text]")
            continue
        text, note = render_attachment(rel)
        if text:
            chunks.append((name, text))
        else:
            notes.append(note or f"[reference '{name}' unavailable]")
    if not chunks:
        return None, (" ".join(notes) if notes
                      else "[reference deliverable could not be rendered to text]")
    if len(chunks) == 1:
        return chunks[0][1], (" ".join(notes) if notes else None)
    return ("\n\n".join(f"--- {name} ---\n{text}" for name, text in chunks),
            (" ".join(notes) if notes else None))


def reference_format_reason(rel_paths: List[str]) -> Optional[str]:
    """Why a task's reference can never be rendered to text, read from the file NAMES alone
    (no download), or None when one of its files has a format render_reference renders to text.

    "no_deliverable_shipped": the task ships no deliverable. "no_text_rendering": every file is
    an image, audio, or a format with no converter (video, CAD); render_reference returns no text
    for those whatever the bytes. None does not promise text: a PDF without a text layer or a
    .zip without a text member is known only once rendered. The two reasons are the scorer's
    `no_reference_reason` values."""
    if not rel_paths:
        return "no_deliverable_shipped"
    for rel in rel_paths:
        name = os.path.basename(rel)
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if ext in _RENDERABLE_EXTS and ext not in _IMAGE_EXTS and ext not in _AUDIO_EXTS:
            return None
    return "no_text_rendering"


#: The --eval-limit draw of the last _load_gdpval_samples in this context (None for a full run),
#: read by run_gdpval for the result. A side channel so the loader keeps its (samples,
#: references) return.
_DRAW: "contextvars.ContextVar[Optional[Dict[str, Any]]]" = contextvars.ContextVar(
    "gdpval_draw", default=None)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _load_gdpval_samples(
    limit: Optional[int] = None,
) -> Tuple[List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    """Load GDPval tasks with inputs rendered and expert references resolved.

    Returns (samples, references) where references maps task_id -> {"text", "note", "n_files"}.
    The reference text is carried on a SIDE CHANNEL, not in the sample metadata, because
    base.run_eval_suite merges sample metadata into the request payload sent to the model - a
    20 KB reference deliverable does not belong in every generation request.

    A full run (no --eval-limit, or one at least the size of the FULL set) keeps every task, the ones
    whose reference cannot be rendered to text included (the scorer excludes them as
    NO_REFERENCE, as before). An --eval-limit draw is taken only from tasks whose reference can
    be scored, known before any generation: the tasks reference_format_reason clears, stratified
    by occupation, and a drawn task whose reference then renders no text (a PDF without a text
    layer, a .zip of audio stems) is dropped from the pool and the draw retaken. 2026-10-11
    smoke run: `--eval-limit 1` drew task e222075d, whose only deliverable is an .mp4; all 3
    generations finished (finish_reason stop, ~2.6k tokens) and none could be scored, so the run
    failed with nothing measured. The draw is recorded (_DRAW -> result `eval_limit_draw` and
    `eval_limit_plan`). Selection never looks at a model output.

    Which of the two applies is decided from the limit against the full set, not against this
    shard: shards differ in size (220 tasks over 8 shards are 28,28,28,28,27,27,27,27), and a
    per-shard decision sent `--eval-limit 27` through the scorable draw on the 28-task shards and
    through the full-shard path, unscorable tasks included, on the 27-task ones (2026-10-11
    review), so the merged eval_limit_plan named a draw rule half the shards had not followed.
    With a shard, an N at or above the shard's scorable tasks runs them all (fewer than N
    samples); the shortfall is recorded as `scorable_short_of_limit`.
    """
    _DRAW.set(None)
    try:
        from datasets import load_dataset
        rows = list(load_dataset(_REPO, split="train", revision=_REVISION))
    except Exception as e:
        logger.error(f"Failed to load dataset for gdpval: {e}")
        raise RuntimeError(f"Could not load dataset for gdpval: {e}") from e
    if not rows:
        raise RuntimeError("Dataset for gdpval returned empty rows")

    # --shard first, over the full ordered set, then the --eval-limit draw within the shard.
    # Whether the draw applies is decided against the full set (see the docstring).
    limited = limit is not None and 0 < int(limit) < len(rows)
    rows, shard_label = apply_shard(rows, "gdpval")
    rendered_refs: Dict[int, Dict[str, Any]] = {}       # id(row) -> its rendered reference

    def _reference(item: Dict[str, Any]) -> Dict[str, Any]:
        if id(item) not in rendered_refs:
            deliverables = list(item.get("deliverable_files") or [])
            with collect_clips() as ref_clips:
                ref_text, ref_note = render_reference(deliverables)
            rendered_refs[id(item)] = {"text": ref_text, "note": ref_note,
                                       "n_files": len(deliverables), "clipped": ref_clips}
        return rendered_refs[id(item)]

    def _occupation(r: Any) -> Any:
        return (r or {}).get("occupation")

    if limited:
        # The --eval-limit draw, stratified by occupation (not a contiguous head), from the tasks
        # whose reference can be scored (see the docstring).
        by_format: Dict[str, int] = {}
        pool = []
        for r in rows:
            reason = reference_format_reason(list(r.get("deliverable_files") or []))
            if reason:
                by_format[reason] = by_format.get(reason, 0) + 1
            else:
                pool.append(r)
        n_format_ok = len(pool)
        no_text: List[Dict[str, Any]] = []
        while True:
            chosen = stratified_sample(pool, int(limit), _occupation, seed="gdpval")
            dropped = [r for r in chosen if not _reference(r)["text"]]
            if not dropped:
                break
            no_text += [{"task_id": str(r.get("task_id") or ""),
                         "note": _reference(r)["note"]} for r in dropped]
            gone = {id(r) for r in dropped}
            pool = [r for r in pool if id(r) not in gone]
        _DRAW.set({
            "limit": int(limit),
            "tasks_in_pool": len(rows),
            "tasks_without_reference_by_format": dict(sorted(by_format.items())),
            "tasks_with_renderable_reference_format": n_format_ok,
            "drawn_without_text_rendering": no_text,
            "tasks_drawn": len(chosen),
            "scorable_short_of_limit": max(0, int(limit) - len(chosen)),
            "rule": ("drawn only from tasks whose expert reference renders to text (known "
                     "before generation); a full run keeps every task"),
        })
        if no_text:
            logger.info("[gdpval] --eval-limit draw: %d drawn task(s) whose reference rendered "
                        "no text were replaced: %s", len(no_text),
                        ", ".join(d["task_id"] for d in no_text))
        logger.info("[gdpval] --eval-limit %d drawn from the %d of %d task(s) whose reference has "
                    "a text-renderable format (%s), %d dropped on rendering.", int(limit),
                    n_format_ok, len(rows),
                    ", ".join(f"{k} {v}" for k, v in sorted(by_format.items())) or "none excluded",
                    len(no_text))
        if len(chosen) < int(limit):
            logger.warning("[gdpval] --eval-limit %d exceeds the %d task(s) whose reference "
                           "renders to text%s: ran all %d (the unscorable tasks are not drawn).",
                           int(limit), len(chosen), " in this shard" if shard_label else "",
                           len(chosen))
        rows = chosen
    else:
        # Full run: every task of the shard (no limit, or one covering the full set).
        rows = list(rows)

    samples: List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]] = []
    references: Dict[str, Dict[str, Any]] = {}
    with_files = missing_any = with_reference = 0
    for item in rows:
        task_id = str(item.get("task_id") or "")
        prompt = str(item.get("prompt") or "").strip()
        if not task_id or not prompt:
            raise RuntimeError("gdpval: unexpected schema (task_id/prompt); "
                               "refusing to fabricate sample data")
        try:
            criteria = json.loads(item.get("rubric_json") or "[]")
        except Exception:
            criteria = []
        if not criteria:
            raise RuntimeError(f"gdpval: task {task_id} has no parseable rubric_json; "
                               "refusing to score against an empty rubric")

        refs = list(item.get("reference_files") or [])
        parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        rendered = unavailable = 0
        input_clips: List[Dict[str, Any]] = []
        if refs:
            with_files += 1
            with collect_clips() as input_clips:
                extra, rendered, unavailable = attachment_parts(refs)
            parts += extra
            if unavailable:
                missing_any += 1
        content: Any = parts if len(parts) > 1 else prompt

        # Expert reference deliverable(s) for the pairwise comparison (side channel).
        references[task_id] = _reference(item)    # rendered once (the draw may have done it)
        if references[task_id]["text"]:
            with_reference += 1

        samples.append((
            [{"role": "user", "content": content}],
            json.dumps(criteria),
            {"category": str(item.get("occupation") or item.get("sector") or "knowledge_work"),
             "task_id": task_id,
             "attachments": len(refs),
             "attachments_rendered": rendered,
             "attachments_unavailable": unavailable,
             # input files the model saw only part of (not a request field; base sends none)
             "inputs_clipped": input_clips},
        ))

    logger.info("Loaded %d gdpval samples (%d with input attachments; %d had an unrenderable "
                "input; %d have a text-renderable expert reference for the pairwise score).",
                len(samples), with_files, missing_any, with_reference)
    return samples, references


# --------------------------------------------------------------------------- #
# scoring: pairwise wins-or-ties rate vs the expert reference (GDPval's headline)
# --------------------------------------------------------------------------- #
_PAIRWISE_PROMPT = """You are an expert grader for economically-valuable knowledge work.

You are given a task, its grading rubric, and TWO candidate deliverables, A and B. Their
order is arbitrary and carries no meaning - do not prefer a deliverable for being first.
Decide which deliverable better completes the task, judged against the SUBSTANCE of the
rubric: correctness, completeness, sound reasoning, and coverage of what the task asks for.
Both deliverables are shown as text (a file deliverable has been rendered to text), so judge
the CONTENT, not the file format, byte layout, or presentation.

# TASK
{prompt}

# GRADING RUBRIC
{rubric}

# DELIVERABLE A
{a}

# DELIVERABLE B
{b}

Reply with ONLY a JSON object:
{{"winner": "A", "reason": "<one short sentence>"}}
where "winner" is "A", "B", or "tie" (use "tie" only when the two are genuinely of equal quality)."""


def _task_prompt(trace: Dict[str, Any]) -> str:
    """The task as the judge sees it: every text part of the trace's first user message, in
    order (the task text, then each input file's text rendering), so the judge can check the
    rubric's references to the inputs."""
    msgs = trace.get("messages") or []
    if not msgs:
        return ""
    content = msgs[0].get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text") or "") for part in content
                       if isinstance(part, dict) and part.get("type") == "text")
    return ""


def _media_inputs(trace: Dict[str, Any]) -> int:
    """The image and audio parts of the trace's first user message (the judge sees none)."""
    msgs = trace.get("messages") or []
    content = msgs[0].get("content") if msgs else None
    if not isinstance(content, list):
        return 0
    return sum(1 for part in content if isinstance(part, dict)
               and part.get("type") in ("image_url", "input_audio", "image", "audio"))


def judge_view(text: str, cap: int = 0) -> Tuple[str, Optional[Dict[str, int]]]:
    """`text` as the pairwise judge sees it, and the clip made ({"full", "kept"} characters) or
    None: the whole text, or its first `cap` characters (default _MAX_JUDGE_CHARS) and a neutral
    marker that reads the same for either candidate."""
    cap = int(cap or _MAX_JUDGE_CHARS)
    text = text or ""
    if len(text) <= cap:
        return text, None
    return (text[:cap] + f"\n[... truncated for this comparison: the first {cap:,} of "
            f"{len(text):,} characters are shown]", {"full": len(text), "kept": cap})


def rubric_for_judge(criteria: List[Dict[str, Any]]) -> Tuple[str, Optional[Dict[str, int]]]:
    """The rubric as the pairwise judge sees it ("[<score> pts] <criterion>" per line), and the
    clip made or None. Over _MAX_RUBRIC_CHARS the rubric keeps the whole criteria that fit
    (never one cut mid-way, unless the first alone is longer than the cap) and ends with a
    visible "[rubric truncated: ...]" line."""
    lines = [f'[{c.get("score")} pts] {c.get("criterion")}' for c in criteria]
    full = "\n".join(lines) or "(no rubric)"
    if len(full) <= _MAX_RUBRIC_CHARS:
        return full, None
    kept: List[str] = []
    size = 0
    for line in lines:
        add = len(line) + (1 if kept else 0)
        if size + add > _MAX_RUBRIC_CHARS:
            break
        kept.append(line)
        size += add
    text = "\n".join(kept) if kept else full[:_MAX_RUBRIC_CHARS]
    shown = (f"{len(kept)} of {len(lines)} criteria" if kept
             else f"part of the first of {len(lines)} criteria")
    info = {"full_chars": len(full), "kept_chars": len(text), "criteria": len(lines),
            "kept_criteria": len(kept)}
    return (text + f"\n[rubric truncated: {shown} shown ({len(text):,} of {len(full):,} "
            "characters)]"), info


#: The only verdicts a pairwise judge reply may carry (compared lower-cased).
_VALID_WINNERS = ("a", "b", "tie")


def _parse_winner(text: Optional[str]) -> Optional[str]:
    """The judge's verdict, 'a' / 'b' / 'tie', or None for a reply without a valid one.

    A reply with no JSON object, unparseable JSON, a missing or null "winner", or any other
    value ("Deliverable B", "reference", ...) is not a verdict. It goes to the judge-outage
    exclusion like an unanswered call, never to a tie: ties count toward the wins-or-ties
    headline, so reading a malformed reply as one would score it as a success.
    """
    if not text:
        return None
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:                                               # noqa: BLE001
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("winner"), str):
        return None
    w = obj["winner"].strip().lower()
    return w if w in _VALID_WINNERS else None


def _model_score(winner: str, model_position: str) -> float:
    """Score a single pairwise verdict from the MODEL's perspective.

    winner is the judge's "a"/"b"/"tie" (as returned by `_parse_winner`); model_position is
    which slot the model occupied in that call ("a" or "b"). Win = 1.0, tie = 0.5, loss = 0.0.
    Anything else raises: a reply without a valid verdict never reaches scoring.
    """
    w = (winner or "").strip().lower()
    if w == "tie":
        return 0.5
    if w in ("a", "b"):
        return 1.0 if w == model_position else 0.0
    raise ValueError(f"not a pairwise verdict: {winner!r}")


def _outcome(pairwise: float) -> str:
    """Map the position-swapped mean onto GDPval's win / tie / loss.

    Above 0.5 the judge preferred the model overall (for example a win in one position and a
    tie in the other), exactly 0.5 is a tie (two ties, or a win and a loss that cancel), and
    below 0.5 the expert reference was preferred.
    """
    if pairwise > 0.5:
        return "win"
    if pairwise == 0.5:
        return "tie"
    return "loss"


#: Base statuses of a generation whose request failed (no reply at all): an endpoint error, a
#: request timeout, or a prompt over the context window. A harness error, not a deliverable.
GENERATION_FAILED_STATUSES = ("FAILED", "TIMEOUT", "OVER_CONTEXT")
#: How much of each judge reply a trace keeps (the replies are one short JSON object).
_MAX_STORED_REPLY_CHARS = 4000


def _make_scorer(judge_model: Optional[str], judge_concurrency: int, metrics: Dict[str, Any],
                 references: Dict[str, Dict[str, Any]]):
    """Grade each task by a position-swapped pairwise comparison vs the expert reference.

    Every pairwise call goes through the whole Gemini cascade (judge_cascade()); `judge_model`
    is not used to route them and is kept for callers."""
    async def _score(sample_traces: List[Dict[str, Any]]) -> None:
        import asyncio
        from tqdm import tqdm

        sem = asyncio.Semaphore(max(1, judge_concurrency))
        judges_used: Dict[str, int] = {}
        invalid_replies = [0]

        async def _compare(prompt_text: str, rubric_text: str, a_text: str, b_text: str
                           ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
            """One pairwise judge call -> (winner 'a' | 'b' | 'tie' or None, the Gemini model
            that answered or None, its reply clipped for the trace or None). The winner is None
            for no answer or a reply without a valid winner. The texts come already clipped
            (judge_view, rubric_for_judge), with their markers."""
            prompt = _PAIRWISE_PROMPT.format(prompt=prompt_text, rubric=rubric_text, a=a_text,
                                             b=b_text)
            async with sem:
                text, judge_used = await judge_generate_cascade(prompt)
            if text is None:
                return None, None, None
            judges_used[judge_used] = judges_used.get(judge_used, 0) + 1
            winner = _parse_winner(text)
            if winner is None:
                invalid_replies[0] += 1
            return winner, judge_used, text.strip()[:_MAX_STORED_REPLY_CHARS]

        async def _one(tr: Dict[str, Any], pbar) -> None:
            ep = tr.get("extra_payload") or {}
            task_id = str(ep.get("task_id") or "")
            ref = references.get(task_id) or {}
            ref_text = ref.get("text")
            try:
                criteria = json.loads(tr.get("gold_answer") or "[]")
            except Exception:
                criteria = []
            rubric_text, rubric_clip = rubric_for_judge(criteria)
            # Input files the model saw only part of (recorded at load; empty when none).
            tr["inputs_clipped"] = list(ep.get("inputs_clipped") or [])
            raw_reply = tr.get("response_text")
            model_text = strip_thinking_tags(raw_reply or "").strip()
            tr["reference_excerpt"] = (ref_text or "")[:2000]
            tr["reference_note"] = ref.get("note")
            if tr.get("status") in (None, "PENDING_JUDGE"):
                tr["status"] = "OK"

            if not ref_text:
                # No text-renderable expert reference -> the pairwise metric is undefined for
                # this task. Exclude it (not a loss), so it cannot deflate the win-rate.
                tr["is_correct"], tr["status"] = False, "NO_REFERENCE"
                tr["scoring_excluded"] = True
                tr["no_reference_reason"] = ("no_deliverable_shipped" if not ref.get("n_files")
                                             else "no_text_rendering")
                # The protocol's own rule (the pairwise needs a reference), not a harness error:
                # it does not by itself make the run completed_with_errors.
                tr["excluded_by_protocol"] = tr["no_reference_reason"]
                pbar.update(1)
                return
            if raw_reply is None and tr.get("status") in GENERATION_FAILED_STATUSES:
                # The request failed (endpoint error, timeout, over-context prompt): a harness
                # error, not the model's deliverable. Excluded from the win-rate, counted by
                # status and named; its base status is kept.
                tr["is_correct"] = False
                tr["scoring_excluded"] = True
                tr["generation_failed"] = tr.get("status")
                pbar.update(1)
                return
            if not model_text:
                # An empty deliverable the model returned loses the pairwise (it produced
                # nothing), rather than being dropped - a model cannot inflate its win-rate by
                # answering nothing.
                tr["pairwise_score"] = 0.0
                tr["pairwise_detail"] = {"empty_response": True}
                tr["pairwise_outcome"] = "loss"
                tr["is_correct"] = False
                pbar.update(1)
                return

            prompt_text, prompt_clip = judge_view(_task_prompt(tr))
            model_view, model_clip = judge_view(model_text)
            ref_view, ref_clip = judge_view(ref_text)
            # What the judge saw only part of: counted in gdpval_report and named in the reason.
            clipped: Dict[str, Any] = {}
            if rubric_clip:
                clipped["rubric"] = rubric_clip
            if ref.get("clipped") or ref_clip:
                clipped["reference"] = {"files": list(ref.get("clipped") or []),
                                        "judge": ref_clip}
            if model_clip:
                clipped["deliverable"] = model_clip
            if prompt_clip:
                clipped["task_prompt"] = prompt_clip
            if clipped:
                tr["judge_input_clipped"] = clipped
            media = _media_inputs(tr)
            if media:
                # input images / audio the model got and the text judge does not see
                tr["judge_without_media_inputs"] = media
            # Position-swapped: model as A (ref as B), then model as B (ref as A). Averaging the
            # two cancels the judge's order bias.
            w1, m1, r1 = await _compare(prompt_text, rubric_text, model_view, ref_view)  # model = A
            w2, m2, r2 = await _compare(prompt_text, rubric_text, ref_view, model_view)  # model = B
            # Both calls, auditable: which cascade model answered each and what it replied.
            detail = {"call1_winner": w1, "call2_winner": w2,
                      "call1_judge_model": m1, "call2_judge_model": m2,
                      "call1_judge_reply": r1, "call2_judge_reply": r2}
            if w1 is None or w2 is None:
                # Judge cascade exhausted, or a reply without a valid winner: excluded from the
                # win-rate and from base.run_eval_suite's pass/fail accuracy, NOT scored as a
                # loss and NOT as a tie.
                tr["judge_grade"] = "JUDGE_OUTAGE"
                tr["pairwise_detail"] = detail
                pbar.update(1)
                return
            s1 = _model_score(w1, "a")
            s2 = _model_score(w2, "b")
            pairwise = (s1 + s2) / 2.0
            tr["pairwise_score"] = pairwise
            tr["pairwise_detail"] = {**detail, "consistent": s1 == s2}
            tr["pairwise_outcome"] = _outcome(pairwise)
            tr["is_correct"] = pairwise >= 0.5   # "better than or as good as" == not a loss
            pbar.update(1)

        with tqdm(total=len(sample_traces), desc="Judging [GDPVAL pairwise]") as pbar:
            await asyncio.gather(*[_one(t, pbar) for t in sample_traces])

        def _task(t: Dict[str, Any]) -> str:
            return str((t.get("extra_payload") or {}).get("task_id") or "")

        def _tasks(pred) -> int:
            return len({_task(t) for t in sample_traces if pred(t)})

        # One comparison per generated sample (the paper pools every sample's comparisons, 9
        # per task); the task counts below are distinct tasks.
        eligible = [t for t in sample_traces
                    if t.get("pairwise_score") is not None and not t.get("scoring_excluded")
                    and t.get("judge_grade") != "JUDGE_OUTAGE"]
        scores = [t["pairwise_score"] for t in eligible]
        n = len(scores)
        outcomes = [_outcome(s) for s in scores]
        wins, ties, losses = (outcomes.count(o) for o in ("win", "tie", "loss"))
        metrics["gdpval_report"] = {
            # GDPval's headline: rated better than or as good as the expert.
            "win_rate": round((wins + ties) / n, 4) if n else None,
            "wins_or_ties_rate": round((wins + ties) / n, 4) if n else None,
            "wins_only_rate": round(wins / n, 4) if n else None,
            # Secondary: mean of the position-swapped scores with a tie counted as 0.5.
            "mean_pairwise_score": round(sum(scores) / n, 4) if n else None,
            "comparisons_in_win_rate": n,
            "tasks_in_win_rate": len({_task(t) for t in eligible}),
            "wins": wins,
            "ties": ties,
            "losses": losses,
            # Tasks that were not a consistent win or loss in both positions (a tie, or the two
            # position-swapped verdicts disagreed); a judge-consistency diagnostic.
            "ties_or_split": sum(1 for s in scores if s not in (0.0, 1.0)),
            "consistent_pairs": sum(
                1 for t in eligible if (t.get("pairwise_detail") or {}).get("consistent")),
            "tasks_no_reference": _tasks(lambda t: t.get("status") == "NO_REFERENCE"),
            # Why: the public gold subset ships no deliverable for some tasks, and some
            # deliverables have nothing a text judge can read (images, video, scanned PDFs,
            # an archive of audio stems).
            "tasks_no_reference_by_reason": {
                reason: _tasks(lambda t, r=reason: t.get("no_reference_reason") == r)
                for reason in ("no_deliverable_shipped", "no_text_rendering")},
            # Comparisons excluded because the judge gave no usable verdict: no answer from the
            # cascade, or a reply without a valid winner (`judge_invalid_replies` of the calls).
            "tasks_judge_outage": _tasks(lambda t: t.get("judge_grade") == "JUDGE_OUTAGE"),
            "comparisons_judge_outage": sum(
                1 for t in sample_traces if t.get("judge_grade") == "JUDGE_OUTAGE"),
            "comparisons_no_reference": sum(
                1 for t in sample_traces if t.get("status") == "NO_REFERENCE"),
            # Generations whose request failed (a harness error), by base status: excluded.
            "comparisons_generation_failed": {
                st: sum(1 for t in sample_traces if t.get("generation_failed") == st)
                for st in GENERATION_FAILED_STATUSES
                if any(t.get("generation_failed") == st for t in sample_traces)},
            "judge_invalid_replies": invalid_replies[0],
            "tasks_with_unavailable_attachments": _tasks(
                lambda t: (t.get("extra_payload") or {}).get("attachments_unavailable")),
            # Clipped inputs and judge inputs (the caps are in clip_limits): input files the
            # model saw only the start of, per task; and the comparisons whose judge saw a
            # clipped rubric, expert reference, model deliverable or task prompt.
            "tasks_with_clipped_inputs": _tasks(
                lambda t: (t.get("extra_payload") or {}).get("inputs_clipped")),
            # distinct files (a PDF cut by pages and by characters is one file)
            "input_files_clipped": sum(
                len({str(c.get("file")) for c in v}) for v in {
                    _task(t): (t.get("extra_payload") or {}).get("inputs_clipped") or []
                    for t in sample_traces}.values()),
            # comparisons whose task had image or audio inputs the text judge did not see
            "comparisons_judge_without_media_inputs": sum(
                1 for t in sample_traces if t.get("judge_without_media_inputs")),
            **{f"comparisons_{key}_clipped": sum(
                1 for t in sample_traces if key in (t.get("judge_input_clipped") or {}))
               for key in ("rubric", "reference", "deliverable", "task_prompt")},
            "clip_limits": clip_limits(),
            # Which cascade models actually answered the pairwise calls.
            "judge_models_used": dict(judges_used),
        }
    return _score


def clip_reasons(rep: Dict[str, Any]) -> List[str]:
    """The comparability reasons for clipped inputs and judge inputs (arXiv:2510.04374 describes
    no truncation: graders get the request and the full reference files)."""
    lim = rep.get("clip_limits") or clip_limits()
    out = []
    if rep.get("tasks_with_clipped_inputs"):
        out.append(f"{rep['tasks_with_clipped_inputs']} task(s) sent with clipped input files "
                   f"({rep.get('input_files_clipped', 0)} file(s) over "
                   f"{lim['max_file_chars']:,} characters or {lim['max_pdf_pages']} PDF pages: "
                   "the model saw only their start; inputs_clipped on each trace)")
    seen = [(rep.get(f"comparisons_{key}_clipped") or 0, label) for key, label in (
        ("rubric", f"a rubric clipped to its whole criteria within "
                   f"{lim['max_rubric_chars']:,} characters"),
        ("reference", f"an expert reference clipped (files over {lim['max_file_chars']:,} "
                      f"characters or {lim['max_pdf_pages']} PDF pages, or the rendering over "
                      f"{lim['max_judge_chars']:,} characters)"),
        ("deliverable", f"a model deliverable clipped to {lim['max_judge_chars']:,} characters"),
        ("task_prompt", f"a task prompt clipped to {lim['max_judge_chars']:,} characters"))]
    seen = [f"{label} in {n} comparison(s)" for n, label in seen if n]
    if seen:
        out.append("the judge saw " + "; ".join(seen) + " (judge_input_clipped on each trace)")
    return out


#: arXiv:2510.04374 footnote 2: "For plots shown, we sampled each model 3 times for each prompt,
#: and then had 3 different human graders grade each sample (yielding 9 comparisons per prompt,
#: per model, across 220 tasks)."
PAPER_SAMPLES_PER_TASK = 3
PAPER_GRADERS_PER_SAMPLE = 3

#: The paper sets no sampling parameters for the models it evaluates (recorded; gbench sends the
#: temperature of its rule, base.resolve_temperature); nor an output budget, so no max_tokens
#: is sent.
GDPVAL_SAMPLING = OfficialSampling(
    reference=("arXiv:2510.04374 (GDPval): models sampled 3 times per task through their "
               "products and APIs; no sampling parameters or output budget are given"),
    send_temperature=False,
    send_max_tokens=False,
)


def run_gdpval(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute gdpval: attachments in, wins-or-ties rate vs the expert reference out.

    Each task is generated 3 times (the paper's samples per task, footnote 2; a larger
    --attempt-count is honoured) and every sample is judged against the human-expert reference
    by one position-swapped pairwise comparison. The headline `accuracy` is GDPval's win rate:
    the share of those comparisons where the model's deliverable was judged better than or as
    good as the reference (wins + ties). Wins-only and the tie-0.5 mean are reported alongside.
    It is NOT OpenAI's GDPval number (`leaderboard_comparable` False): a text endpoint cannot
    emit the .xlsx/.docx/.pdf deliverable, so the reference file is rendered to text and the
    model competes on rendered content only.
    """
    skip = gemini_required_skip("gdpval", model_name)
    if skip:
        return skip
    samples_per_task = max(PAPER_SAMPLES_PER_TASK, int(kwargs.get("attempt_count") or 1))
    metrics: Dict[str, Any] = {}
    _DRAW.set(None)
    samples, references = _load_gdpval_samples(limit=kwargs.get("limit"))
    draw = _DRAW.get()
    limit_plan = base_suite_plan("gdpval", presharded_for_base(samples), kwargs.get("limit"),
                                 samples_per_task)
    if draw:
        # The pool the --eval-limit draw was taken from (counts sum over shards).
        limit_plan["tasks_in_pool"] = draw["tasks_in_pool"]
        limit_plan["tasks_with_renderable_reference_format"] = (
            draw["tasks_with_renderable_reference_format"])
        limit_plan["drawn_without_text_rendering"] = len(draw["drawn_without_text_rendering"])
        limit_plan["scorable_short_of_limit"] = draw["scorable_short_of_limit"]
        limit_plan["draw_rule"] = draw["rule"]
    result = run_eval_suite(
        eval_name="gdpval",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=presharded_for_base(samples),  # this shard's tasks, kept as is
        async_eval_fn=_make_scorer(kwargs.get("judge_model"), concurrency, metrics, references),
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
        attempt_count=samples_per_task,
        supports_attempts=True,   # every sample is judged on its own against the reference
        official_sampling=GDPVAL_SAMPLING,
    )
    result["eval_limit_plan"] = limit_plan
    if draw:
        result["eval_limit_draw"] = draw
    result.update(metrics)
    result["samples_per_task"] = samples_per_task
    # A text endpoint cannot produce the FILE deliverable the reference is, so the model
    # competes on rendered content only - this is not OpenAI's GDPval number. The grader is
    # also not OpenAI's (their trained autograder is not public), so the reason names both.
    reasons = [
        "text endpoint vs an expert FILE deliverable rendered to text (the model cannot emit "
        ".xlsx/.docx/.pdf and competes on rendered content only)",
        "graded by gbench's standard Gemini cascade with a gbench pairwise prompt (OpenAI's "
        "grader is a trained model whose prompt is not public)",
        f"{samples_per_task} generations per task, each judged once by the cascade (one "
        f"position-swapped pair), where the paper has {PAPER_GRADERS_PER_SAMPLE} human experts "
        f"grade each of its {PAPER_SAMPLES_PER_TASK} samples (9 comparisons per task, "
        "arXiv:2510.04374 footnote 2)",
        "tasks without a text-renderable reference are excluded",
        "the model is offered no tools (the paper ran the OpenAI models with the web search and "
        "code interpreter tools and Claude through its UI with file creation, arXiv:2510.04374 "
        "footnote 2)",
        "the rendered expert reference can be told apart from a prose answer (file, sheet and "
        "slide headers, truncation markers), so the comparison is not fully blind",
    ]
    if result.get("temperature") == 0.0:
        reasons.append(f"the {samples_per_task} generations are greedy (temperature 0.0), so "
                       "they differ only through server nondeterminism")
    elif result.get("temperature") is not None:
        source = str(result.get('temperature_source') or '')
        setter = ("gbench's rule (1.0 with --thinking, 0.0 without)"
                  if source in ("thinking", "default") else source.split(':', 1)[-1])
        reasons.append(f"temperature {result.get('temperature')} set by {setter} (the "
                       "paper sets none; the served model's defaults apply otherwise)")
    if draw:
        # No per-shard counts in the text, so every shard states the same reason and the merge
        # keeps it once; the counts are in eval_limit_plan (summed over shards) and eval_limit_draw.
        reasons.append(
            f"--eval-limit {draw['limit']} runs a subset of the 220 tasks, drawn only from tasks "
            "whose expert reference renders to text (counts in eval_limit_plan: "
            "tasks_with_renderable_reference_format, drawn_without_text_rendering, "
            "scorable_short_of_limit; a full run keeps every task)")
    elif kwargs.get("limit"):
        reasons.append(f"--eval-limit {kwargs.get('limit')} runs a subset of the 220 tasks")
    rep = metrics.get("gdpval_report") or {}
    by_reason = rep.get("tasks_no_reference_by_reason") or {}
    if any(by_reason.values()):
        reasons[3] = ("tasks without a text-renderable reference are excluded ("
                      + ", ".join(f"{k} {v}" for k, v in sorted(by_reason.items()) if v) + ")")
    if rep.get("comparisons_judge_without_media_inputs"):
        reasons.append(f"the judge sees the task's text and the text rendering of its input "
                       f"files, not its image or audio inputs "
                       f"({rep['comparisons_judge_without_media_inputs']} comparison(s))")
    gen_failed = rep.get("comparisons_generation_failed") or {}
    if gen_failed:
        reasons.append(f"{sum(gen_failed.values())} generation(s) whose request failed ("
                       + ", ".join(f"{st} {n}" for st, n in sorted(gen_failed.items()))
                       + ") excluded from the win rate as harness errors, not scored as losses")
    if rep.get("comparisons_judge_outage"):
        reasons.append(f"{rep['comparisons_judge_outage']} comparison(s) without a usable judge "
                       f"verdict excluded ({rep.get('judge_invalid_replies', 0)} judge repl(ies) "
                       "without a valid winner)")
    reasons += clip_reasons(rep)
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = (
        "; ".join(reasons) + "; a gbench-internal wins-or-ties rate, not OpenAI's GDPval number")
    result["judge"] = "gbench-gemini-cascade"
    # Answered pairwise calls per Gemini model; each comparison names its two in pairwise_detail.
    result["judge_models"] = dict(rep.get("judge_models_used") or {})
    result["judge_cascade"] = judge_cascade()
    # The distinct prompts (tasks) the headline scored.
    result["effective_n"] = int(rep.get("tasks_in_win_rate") or 0)

    win_rate = rep.get("win_rate")
    if win_rate is not None:
        result["accuracy"] = round(win_rate * 100.0, 2)   # headline = wins-or-ties rate (%)
        result["win_rate_pct"] = result["accuracy"]
        if rep.get("wins_only_rate") is not None:
            result["wins_only_pct"] = round(rep["wins_only_rate"] * 100.0, 2)
        if rep.get("mean_pairwise_score") is not None:
            result["mean_pairwise_score_pct"] = round(rep["mean_pairwise_score"] * 100.0, 2)
        scored = rep.get("tasks_in_win_rate") or 0
        comparisons = rep.get("comparisons_in_win_rate", scored) or 0
        # `total_questions` counts GENERATIONS (tasks x samples); the win-rate is over the
        # comparisons that could be SCORED (a text-renderable reference + a usable verdict).
        # Name the gap rather than hiding it in a denominator: an excluded comparison was
        # unmeasurable, not asked-and-lost.
        result["tasks_scored"] = scored
        result["comparisons_scored"] = comparisons
        no_ref = rep.get("tasks_no_reference") or 0
        outage = rep.get("tasks_judge_outage") or 0
        result["tasks_no_reference"] = no_ref
        result["tasks_no_reference_by_reason"] = rep.get("tasks_no_reference_by_reason")
        result["tasks_judge_outage"] = outage
        result["comparisons_judge_outage"] = rep.get("comparisons_judge_outage", outage) or 0
        total_q = result.get("total_questions") or 0
        result["comparisons_generation_failed"] = sum(gen_failed.values())
        result["comparisons_unaccounted"] = max(
            0, total_q - comparisons - (rep.get("comparisons_no_reference", no_ref) or 0)
            - result["comparisons_judge_outage"] - result["comparisons_generation_failed"])
        result["scoring_note"] = (
            f"accuracy is GDPval's WIN RATE, the share of comparisons where the model's "
            f"deliverable was judged better than or as good as the expert reference (wins + "
            f"ties), over {comparisons} comparison(s) ({samples_per_task} samples per task, "
            f"{scored} task(s)) with a text-renderable reference and a usable judge verdict, "
            f"position-swapped to cancel order bias (wins_only_pct and mean_pairwise_score_pct "
            f"are reported alongside); NOT a pass rate over the {total_q} generation(s). Tasks "
            f"without a renderable reference ({no_ref}), comparisons without a usable "
            f"verdict ({result['comparisons_judge_outage']}) and generations whose request "
            f"failed ({result['comparisons_generation_failed']}) are excluded from the "
            f"win-rate, not counted as losses; an empty deliverable the model returned IS "
            f"counted, as a loss.")
        if no_ref or outage or gen_failed:
            logger.warning(
                "[gdpval] win-rate covers %d comparison(s) over %d task(s): %d task(s) had no "
                "text-renderable expert reference, %d comparison(s) had no usable judge "
                "verdict and %d generation(s) failed at the endpoint (all excluded, not scored "
                "as losses).",
                comparisons, scored, no_ref, result["comparisons_judge_outage"],
                result["comparisons_generation_failed"])
    elif result.get("status") not in ("error", "failed"):
        # No comparison could be scored (every generation had no text-renderable reference, no
        # usable judge verdict or a failed request): nothing was measured, so the score is
        # withheld instead of reporting base's 0.
        why = (f"no comparison could be scored: {rep.get('comparisons_no_reference', 0)} without "
               f"a text-renderable reference, {rep.get('comparisons_judge_outage', 0)} without a "
               f"usable judge verdict, {sum(gen_failed.values())} failed generation(s), of "
               f"{result.get('total_questions') or 0} generation(s)")
        result["accuracy"] = None
        result["status"] = "failed"
        result["status_reason"] = why
        result["accuracy_withheld_reason"] = why
    else:
        result["accuracy"] = None
        result.setdefault("accuracy_withheld_reason", result.get("status_reason")
                          or "no comparison could be scored")
    logger.warning(
        "[gdpval] wins-or-ties rate vs an expert FILE deliverable rendered to text. A "
        "text endpoint cannot emit .xlsx/.docx/.pdf, so this is not comparable with OpenAI's "
        "published GDPval win-rate.")
    # One shard of a --shard I/N run is a partial set: the shard reason (finalize_shard_fields).
    return finalize_shard_fields(result)
