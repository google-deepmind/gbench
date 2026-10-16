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

# Evaluation Suite: omnidocbench
# Description: OmniDocBench v1.6 (Multimodal Document Parsing, Layout & Formula Recognition)

"""gbench native built-in runner for omnidocbench (Multimodal & Vision).

The model transcribes each page image with OmniDocBench's own general-VLM instruction and
decoding (tools/model_infer/Qwen2VL_img2md.py: one user turn, image then prompt, greedy,
max_new_tokens=32000); OmniDocBench's evaluator, run from a locally built image pinned to
_EVALUATOR_REVISION, scores the transcriptions. The dataset is pinned to _DATASET_REVISION.

Sampling: the reference script decodes greedily (Qwen2VL_img2md.py: do_sample=False), i.e.
temperature 0.0 (base.OfficialSampling, recorded); the request follows gbench's rule
(base.resolve_temperature): 1.0 with `--thinking`, 0.0 without.
Without `--thinking` every request carries the script's max_new_tokens=32000 as its max_tokens,
so `--max-output-tokens` is not applied (`replies_at_official_max_tokens` counts pages that stop
there). The script defines no reasoning-model settings, so with `--thinking` that cap, written for
non-reasoning generation, is not applied: each request is sent the run's `--max-output-tokens`
(the token-limit policy, base.resolve_token_budget; the result records it). Override the
temperature for a whole run with `--temperature`, or for this suite alone with
`GBENCH_OMNIDOCBENCH_TEMPERATURE`, which takes precedence over both; the result records
`official_sampling`.

Network: the evaluator container compiles the model's LaTeX (the CDM metric runs pdflatex on each
predicted formula) and reads only its mounted files, so it runs with `--network none`
(_build_command), as does the image's revision check: no container of this suite can reach the
cloud metadata server or anything else. The page images and annotations are downloaded on the host
before any container starts.
"""

import glob
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from huggingface_hub import hf_hub_download
from .limit_plan import base_attempts, eval_limit_plan
from .base import OfficialSampling, apply_shard, finalize_shard_fields, run_eval_suite
from .sampling import stratified_indices, presharded_for_base
from .metrics import record_budget_cutoffs
from .charxiv import image_mime   # the MIME type the bytes carry (PNG, GIF, WebP, else JPEG)
from .dataset_utils import extract_lossless_image_b64
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

PILLAR = "Multimodal & Vision"
DOCS_URL = "docs/evals/omnidocbench.md"

#: The evaluator image (Python 3.10 + TeX Live/CJK + ImageMagick 7 + Ghostscript for the CDM
#: metric). Built LOCALLY from docker/omnidocbench.Dockerfile (gbench never pulls registry
#: images). Scored in a container because OmniDocBench pins Python <3.12 and numpy==1.24.4, which
#: would break the torch/vLLM serving env.
_SCORER_IMAGE_DEFAULT = "gbench-omnidocbench:193627ae"

#: opendatalab/OmniDocBench on the HF Hub (v1.6 data, 1651 annotated pages), 2026-06-26. The
#: OmniDocBench.json annotations are byte-identical to the v1.6 commit d386947f.
_DATASET_REVISION = "aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec"

#: opendatalab/OmniDocBench commit the evaluator image must be built from: the newest commit that
#: changes anything besides the README leaderboard tables as of 2026-09-24 (upstream HEAD f133a71e
#: differs from it only in README.md / README_zh-CN.md). Verified inside the image at run time.
_EVALUATOR_REVISION = "193627ae9e97d89188468ed1ee3b7a856ff76044"

#: OmniDocBench's general-VLM instruction, verbatim (whitespace included) from
#: tools/model_infer/Qwen2VL_img2md.py @ 193627ae; the gemini25/gemini3/internvl2 scripts use the
#: same text. The evaluator's TEDS/CDM matching relies on its HTML-table and LaTeX conventions.
_PROMPT = r'''You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

        1. Text Processing:
        - Accurately recognize all text content in the PDF image without guessing or inferring.
        - Convert the recognized text into Markdown format.
        - Maintain the original document structure, including headings, paragraphs, lists, etc.

        2. Mathematical Formula Processing:
        - Convert all mathematical formulas to LaTeX format.
        - Enclose inline formulas with \( \). For example: This is an inline formula \( E = mc^2 \)
        - Enclose block formulas with \\[ \\]. For example: \[ \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \]

        3. Table Processing:
        - Convert tables to HTML format.
        - Wrap the entire table with <table> and </table>.

        4. Figure Handling:
        - Ignore figures content in the PDF image. Do not attempt to describe or convert images.

        5. Output Format:
        - Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
        - For complex layouts, try to maintain the original document's structure and format as closely as possible.

        Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.
        '''

#: Qwen2VL_img2md.py generates with max_new_tokens=32000 (greedy, do_sample=False).
_MAX_NEW_TOKENS = 32000
#: The same script's decoding: greedy (do_sample=False), i.e. temperature 0.
OFFICIAL_SAMPLING = OfficialSampling(
    reference=(f"opendatalab/OmniDocBench tools/model_infer/Qwen2VL_img2md.py "
               f"@{_EVALUATOR_REVISION[:8]} (greedy, do_sample=False, max_new_tokens=32000)"),
    temperature=0.0, max_tokens=_MAX_NEW_TOKENS)

#: The end2end config the evaluator runs (canonical per-modality composite): text Edit_dist,
#: display-formula Edit_dist + CDM, table TEDS + Edit_dist, reading-order Edit_dist. Written INSIDE
#: the container; paths are relative to /workspace (the image's WORKDIR).
_END2END_CONFIG = """end2end_eval:
  metrics:
    text_block:
      metric: [Edit_dist]
    display_formula:
      metric: [Edit_dist, CDM]
      cdm_workers: {workers}
    table:
      metric: [TEDS, Edit_dist]
      teds_workers: {workers}
    reading_order:
      metric: [Edit_dist]
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: ./gt/gt.json
    prediction:
      data_path: ./data_md/predictions
    match_method: quick_match
    match_workers: {workers}
    quick_match_truncated_timeout_sec: 300
    match_timeout_sec: 420
    timeout_fallback_max_chunk_span: 10
    timeout_fallback_order_penalty: 0.10
"""

#: OmniDocBench excludes these from the end2end text metric (page furniture, not content).
_IGNORED_CATS = {"header", "footer", "page_number", "page_footnote", "abandon"}


def _page_attribute(anno: Dict[str, Any], key: str) -> str:
    attr = (anno.get("page_info") or {}).get("page_attribute") or {}
    return str(attr.get(key) or "unknown") if isinstance(attr, dict) else "unknown"


def page_source(anno: Dict[str, Any]) -> str:
    """A page's data source (page_info.page_attribute.data_source: book, PPT2PDF,
    academic_literature, ...)."""
    return _page_attribute(anno, "data_source")


def page_subset(anno: Dict[str, Any]) -> str:
    """A page's subset (page_info.page_attribute.subset: v1.5, equation_hard, layout_hard,
    table_hard)."""
    return _page_attribute(anno, "subset")


#: How an --eval-limit subset is drawn (recorded as `subset_selection`).
SUBSET_SELECTION = ("stratified by the page's subset (page_info.page_attribute.subset: v1.5, "
                    "equation_hard, layout_hard, table_hard), then within each subset by its data "
                    "source (page_attribute.data_source), seeded 'omnidocbench'")


def select_pages(annos: List[Dict[str, Any]], limit: Optional[int]) -> List[int]:
    """Indices (ascending) of the pages a run scores: every page, or with --eval-limit a subset
    stratified by the page's subset and, within each subset, by its data source (SUBSET_SELECTION;
    not a contiguous head).

    The composite needs pages with tables (TEDS) and with display formulas (CDM). The split runs
    one data source at a time (its first 30 pairable pages are PPT2PDF slides without a display
    formula, the first 13 without a table), so a head subset of 30 pages or fewer never had the
    composite; stratifying by subset puts a table_hard page (all hold a table) and an
    equation_hard page (91 of 100 hold a display formula) into any subset of 4 pages or more."""
    subsets = [page_subset(a) for a in annos]
    if limit is None or limit <= 0 or limit >= len(annos):
        return list(range(len(annos)))
    per_subset: Dict[str, int] = {}
    for k in stratified_indices(subsets, limit, seed="omnidocbench"):
        per_subset[subsets[k]] = per_subset.get(subsets[k], 0) + 1
    chosen: List[int] = []
    for subset, count in per_subset.items():
        members = [k for k, s in enumerate(subsets) if s == subset]
        inner = stratified_indices([page_source(annos[k]) for k in members], count,
                                   seed=f"omnidocbench/{subset}")
        chosen.extend(members[j] for j in inner)
    return sorted(chosen)


def _load_omnidocbench_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load OmniDocBench (opendatalab/OmniDocBench @ _DATASET_REVISION) from the HF Hub."""
    try:
        from datasets import load_dataset, Image
        ds = load_dataset("opendatalab/OmniDocBench", split="train", revision=_DATASET_REVISION)
        # The split has EXACTLY ONE column, `image` - no filename anywhere - so pairing by
        # `item["image_path"]`/`file_name`/`page_name` always missed and every page fell
        # back to positional pairing against a different page's ground truth. Measured
        # 2026-08-17: 20/20 unpaired, hence a guaranteed 0/20 even once the requests
        # started going through. Decoding drops the source path; the UNDECODED feature
        # keeps it, and its basename ("PPT_1001115_eng_page_003.png") is exactly the key
        # OmniDocBench.json uses in `page_info.image_path`. Without the names no page can be
        # paired, so a failure here is a load failure, not something to work around.
        undecoded = ds.cast_column("image", Image(decode=False))
        page_names: List[str] = [os.path.basename((undecoded[i]["image"] or {}).get("path") or "")
                                 for i in range(len(undecoded))]
        json_file = hf_hub_download(repo_id="opendatalab/OmniDocBench", filename="OmniDocBench.json",
                                    repo_type="dataset", revision=_DATASET_REVISION)
        with open(json_file) as f:
            annotations = json.load(f)
    except Exception as e:
        raise infra_required(
            "omnidocbench", f"could not load opendatalab/OmniDocBench@{_DATASET_REVISION[:12]} "
            f"from the HF Hub ({e}).", DOCS_URL) from e

    if not ds:
        raise RuntimeError("Dataset for omnidocbench returned empty rows")

    # Pair image <-> annotation by page filename. Indexing both by position assumes the
    # HF split and OmniDocBench.json enumerate the pages in the same order; when they do
    # not, every page is scored against a different page's ground truth.
    anno_by_name = {}
    for anno in annotations:
        name = str((anno.get("page_info") or {}).get("image_path") or "").strip()
        if name:
            anno_by_name[os.path.basename(name)] = anno

    # Pair EVERY page by name first (no image is decoded for this), then pick the pages: all
    # of them, or with --eval-limit a stratified subset (select_pages; not a contiguous
    # head, which could never produce the composite). A page with no annotation (the split
    # carries a few dataset artwork images that are not benchmark pages) is dropped: positional
    # fallback would grade it against a DIFFERENT page's ground truth, a guaranteed-wrong score
    # that looks like a model failure.
    pairable: List[Tuple[int, str, Dict[str, Any]]] = []
    unpaired = 0
    for i in range(len(ds)):
        page_name = page_names[i] if i < len(page_names) else ""
        if not page_name:
            row = undecoded[i]
            page_name = os.path.basename(str(row.get("image_path") or row.get("file_name")
                                             or row.get("page_name") or "").strip())
        anno = anno_by_name.get(page_name)
        if anno is None:
            unpaired += 1
            continue
        pairable.append((i, page_name, anno))
    # --shard first, over the full ordered set of pairable pages, then the --eval-limit draw
    # (select_pages) within the shard.
    pairable, _ = apply_shard(pairable, "omnidocbench")
    chosen = select_pages([anno for _, _, anno in pairable], limit)

    samples = []
    for i, page_name, anno in (pairable[k] for k in chosen):
        item = ds[i]
        image_val = item.get("image")

        layout_dets = anno.get("layout_dets", [])
        page_info = anno.get("page_info", {})
        page_attr = page_info.get("page_attribute", {})
        if isinstance(page_attr, dict):
            doc_type = str(page_attr.get("subset") or page_attr.get("data_source") or "unknown")
        else:
            doc_type = str(page_attr or "unknown")

        # A page's ground truth is not text alone: tables are annotated as `html` and
        # formulas as `latex`. Keeping only `text` asked the model to transcribe the whole
        # page and then graded it against a reference with the tables and equations
        # removed, so a correct table cost the page its score.
        # Order blocks by the annotation's READING ORDER (not JSON order), and exclude
        # ignored blocks + page furniture (header/footer/page number/abandon) which
        # OmniDocBench does not score in the end2end text metric.
        ordered = sorted(layout_dets, key=lambda d: (d.get("order") if isinstance(d.get("order"), int) else 10**9))
        text_parts: List[str] = []
        table_htmls: List[str] = []
        formula_latex: List[str] = []
        all_parts: List[str] = []
        for det in ordered:
            if det.get("ignore"):
                continue
            ct = str(det.get("category_type") or "").lower()
            if ct in _IGNORED_CATS:
                continue
            if ct == "table":
                v = det.get("html")
                if v:
                    table_htmls.append(str(v))
                    all_parts.append(str(v))
            elif "equation" in ct or "formula" in ct:
                v = det.get("latex") or det.get("text")
                if v:
                    formula_latex.append(str(v))
                    all_parts.append(str(v))
            else:
                v = det.get("text") or det.get("html") or det.get("latex")
                if v:
                    text_parts.append(str(v))
                    all_parts.append(str(v))
        # gold carries the per-modality streams (for the composite) plus the concatenated
        # `all` (for the overall edit distance + strict pass).
        gt_text = json.dumps({
            "all": "\n".join(all_parts),
            "text": "\n".join(text_parts),
            "tables": table_htmls,
            "formulas": formula_latex,
            "page_name": page_name,          # image basename; names the .md the evaluator matches
        })

        # One user turn, image first then the instruction, as Qwen2VL_img2md.py builds it.
        b64_str = extract_lossless_image_b64(image_val)
        if not b64_str:
            raise RuntimeError(f"omnidocbench: page {page_name!r} has no decodable image")
        content_payload: List[Dict[str, Any]] = [
            {"type": "image_url",
             "image_url": {"url": f"data:{image_mime(b64_str)};base64,{b64_str}"}},
            {"type": "text", "text": _PROMPT},
        ]

        messages = [{"role": "user", "content": content_payload}]
        samples.append((messages, gt_text, {"category": doc_type}))

    if unpaired:
        logger.warning("[omnidocbench] %d page(s) had no annotation and were EXCLUDED "
                       "(scoring them against another page's ground truth would be a "
                       "guaranteed-wrong result attributed to the model). %d scored.",
                       unpaired, len(samples))
    logger.info(f"Loaded {len(samples)} omnidocbench samples with lossless direct images.")
    return samples


def _normalize_doc_text(text: str) -> str:
    """OmniDocBench text normalization before edit distance.

    Whitespace, markdown emphasis and heading markers are transcription style, not
    transcription accuracy, so they are removed on both sides.
    """
    t = str(text or "")
    # Keep a link's DISPLAY TEXT, not delete the whole link. Deleting `[text](url)` entirely was
    # ASYMMETRIC: a model that renders a bare URL as a markdown link (`[http://x](http://x)`) lost the
    # URL from the prediction while the gold's bare `http://x` was kept, inflating the edit distance
    # and flipping a faithful transcription to FAIL. Reducing to the display text normalizes both
    # sides consistently (`[http://x](http://x)` -> `http://x`; `[Grammarly](url)` -> `Grammarly`).
    t = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", t)       # images/links -> their display text
    t = re.sub(r"[*_`#>|]+", " ", t)                       # markdown emphasis/heading/table pipes
    t = re.sub(r"\s+", " ", t)
    return t.strip().lower()


def normalized_edit_distance(pred: str, gold: str) -> float:
    """Character-level normalized Levenshtein distance in [0, 1]; 0 is a perfect transcription.

    Feeds only the interim per-page pass flag; the reported scores come from OmniDocBench's
    evaluator. `rapidfuzz` (part of gbench[evals]) is required and checked before generation.
    """
    from rapidfuzz.distance import Levenshtein
    p, g = _normalize_doc_text(pred), _normalize_doc_text(gold)
    if not g:
        return 0.0 if not p else 1.0
    return float(Levenshtein.normalized_distance(p, g))


def _omnidoc_gold(gold_target: Any) -> Dict[str, Any]:
    """The per-modality gold payload {all, text, tables[], formulas[]} from the JSON gold."""
    try:
        g = json.loads(gold_target)
        if isinstance(g, dict):
            return g
    except Exception:
        pass
    s = str(gold_target or "")
    return {"all": s, "text": s, "tables": [], "formulas": []}


def _eval_omnidocbench(response_text: str, gold_target: str) -> bool:
    """Strict per-page pass: a transcription within 10% edit distance of the annotation.

    The headline metric of the suite is the *continuous* mean edit distance (see
    `run_omnidocbench`); this boolean only feeds the harness' pass count. The previous
    scorer passed a page when 40% of the ground truth's 3+ character word types appeared
    anywhere in the response, in any order - a page of the right document with more than
    half its content missing scored as a correct transcription.
    """
    gold_all = _omnidoc_gold(gold_target).get("all", "")
    if not response_text or not str(gold_all).strip():
        return False
    return normalized_edit_distance(response_text, str(gold_all)) <= 0.10


def _scorer_image() -> str:
    return os.environ.get("GBENCH_OMNIDOCBENCH_IMAGE", _SCORER_IMAGE_DEFAULT)


def _check_prereqs(image: str) -> None:
    """rapidfuzz, Docker and the evaluator image built from _EVALUATOR_REVISION (never a skip)."""
    build = (f"OmniDocBench's evaluator (with the CDM TeX Live / ImageMagick 7 / Ghostscript "
             f"runtime) runs in a locally-built image. Build it from an OmniDocBench git checkout "
             f"at {_EVALUATOR_REVISION}:\n"
             f"  git -C $GBENCH_PREREQS_DIR/OmniDocBench checkout {_EVALUATOR_REVISION}\n"
             f"  docker build -t {image} -f docker/omnidocbench.Dockerfile "
             "$GBENCH_PREREQS_DIR/OmniDocBench\n"
             "(Override the tag with GBENCH_OMNIDOCBENCH_IMAGE.)")
    if importlib.util.find_spec("rapidfuzz") is None:
        raise infra_required("omnidocbench", "rapidfuzz is not installed (pip install -e "
                             "\".[evals]\" from the gbench checkout, or pip install rapidfuzz).",
                             DOCS_URL)
    if not shutil.which("docker"):
        raise infra_required("omnidocbench", "docker CLI not found. " + build, DOCS_URL)
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        raise infra_required("omnidocbench", "docker daemon not reachable. " + build, DOCS_URL)
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode != 0:
        raise infra_required("omnidocbench", f"image {image!r} not found. " + build, DOCS_URL)
    _verify_evaluator_revision(image, build)


def _verify_evaluator_revision(image: str, build: str) -> None:
    """The image's /workspace checkout must be _EVALUATOR_REVISION with no edits to tracked files.

    The Dockerfile copies the whole checkout (including .git) into /workspace, so the evaluator
    that will score is identified by its own git metadata.
    """
    probe = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", "--entrypoint", "bash", image, "-c",
         "git -C /workspace rev-parse HEAD && "
         "git -C /workspace status --porcelain --untracked-files=no"],
        capture_output=True, text=True)
    lines = (probe.stdout or "").strip().splitlines()
    if probe.returncode != 0 or not lines:
        raise infra_required(
            "omnidocbench", f"cannot read the OmniDocBench revision inside {image!r} (it must be "
            f"built from a git checkout): {(probe.stderr or '').strip()[-300:]} " + build, DOCS_URL)
    if lines[0].strip() != _EVALUATOR_REVISION:
        raise infra_required(
            "omnidocbench", f"image {image!r} holds OmniDocBench {lines[0].strip()}, not the pinned "
            f"{_EVALUATOR_REVISION}. Rebuild it. " + build, DOCS_URL)
    if len(lines) > 1:
        raise infra_required(
            "omnidocbench", f"image {image!r} was built from an OmniDocBench checkout with local "
            f"changes ({'; '.join(l.strip() for l in lines[1:6])}). Rebuild it from a clean "
            "checkout. " + build, DOCS_URL)


def _build_command(image: str, gt_json: str, pred_dir: str, out_dir: str, workers: int) -> List[str]:
    """The `docker run` that scores predictions with OmniDocBench's own pipeline in its image.

    Mounts the GT json, the predictions dir and an output dir; writes the end2end config inside
    the container (paths relative to the image's /workspace WORKDIR) and runs pdf_validation.py.
    The container runs with `--network none`: the evaluator compiles the model's LaTeX (CDM's
    pdflatex) and needs nothing beyond the mounted files and the image (the end2end config's
    metrics, Edit_dist, CDM and TEDS, load nothing at run time; BLEU and METEOR, which would load
    through `evaluate.load`, are not in it).
    """
    inner = ("set -e; mkdir -p configs\n"
             "cat > configs/gbench_end2end.yaml <<'GBENCH_EOF'\n"
             + _END2END_CONFIG.format(workers=workers) +
             "GBENCH_EOF\n"
             "python pdf_validation.py --config configs/gbench_end2end.yaml")
    return [
        "docker", "run", "--rm", "--network", "none", "--entrypoint", "bash",
        "-v", f"{gt_json}:/workspace/gt/gt.json:ro",
        "-v", f"{pred_dir}:/workspace/data_md/predictions:ro",
        "-v", f"{out_dir}:/workspace/result",
        image, "-lc", inner,
    ]


#: The result's `metric`.
_METRIC = ("OmniDocBench end2end composite = ((1-text_EditDist)*100 + table_TEDS*100 + "
           "formula_CDM*100)/3, computed by the official OmniDocBench evaluator image (CDM "
           "formula + TEDS table + edit-distance text + reading order); per-modality values also "
           "reported.")


def _nb_value(metrics: Dict[str, Any], key: str) -> Optional[float]:
    return (metrics.get(key) or {}).get("notebook_value")


def _raw_value(metrics: Dict[str, Any], key: str) -> Optional[float]:
    return (metrics.get(key) or {}).get("raw")


def _parse_summary(out_dir: str) -> Optional[Dict[str, Any]]:
    """The evaluator writes result/<pred_basename>_quick_match_run_summary.json (save_name =
    basename(pred_dir) + '_quick_match' = 'predictions_quick_match')."""
    path = os.path.join(out_dir, "predictions_quick_match_run_summary.json")
    if not os.path.isfile(path):
        cand = glob.glob(os.path.join(out_dir, "*_run_summary.json"))
        if not cand:
            return None
        path = cand[0]
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("notebook_metric_summary")


def _annotations_by_name() -> Dict[str, Dict[str, Any]]:
    """OmniDocBench's annotations (OmniDocBench.json at the pinned revision) by image basename."""
    gt_file = hf_hub_download(repo_id="opendatalab/OmniDocBench", filename="OmniDocBench.json",
                             repo_type="dataset", revision=_DATASET_REVISION)
    with open(gt_file, encoding="utf-8") as f:
        annotations = json.load(f)
    anno_by_name = {}
    for anno in annotations:
        nm = os.path.basename(str((anno.get("page_info") or {}).get("image_path") or "").strip())
        if nm:
            anno_by_name[nm] = anno
    return anno_by_name


def run_omnidocbench(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 4,
    enable_thinking: bool = False,
    results_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run canonical OmniDocBench end2end.

    gbench GENERATES the per-page markdown transcriptions against the served model (multimodal
    prompts, in-process), then delegates SCORING to OmniDocBench's own evaluator inside its
    official image - the canonical per-modality composite:
    ``((1 - text_EditDist)*100 + table_TEDS*100 + formula_CDM*100) / 3`` (CDM formula metric, TEDS
    tables, edit-distance text + reading order). Hard-errors (never skips) if Docker/the image is
    absent or the image was not built from _EVALUATOR_REVISION. Not scored in-process
    (OmniDocBench pins Python <3.12 + numpy 1.24.4)."""
    image = _scorer_image()
    _check_prereqs(image)

    # Ground truth: OmniDocBench's annotations, matched to predictions by image basename.
    anno_by_name = _annotations_by_name()

    samples = _load_omnidocbench_samples(limit=limit)
    eval_plan = eval_limit_plan("omnidocbench", limit, len(samples), attempts=base_attempts())
    result = run_eval_suite(
        eval_name="omnidocbench",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=presharded_for_base(samples),  # this shard's pages, kept as is
        eval_fn=_eval_omnidocbench,     # interim per-page pass count; real score comes from the evaluator
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
        # The reference script's max_new_tokens, applied by the token-limit policy.
        official_sampling=OFFICIAL_SAMPLING,
    )
    result["eval_limit_plan"] = eval_plan
    if result.get("empty_shard"):
        # No page on this node (--shard I/N with more shards than pages): base's honest empty row,
        # nothing to hand to the evaluator.
        result["metric"] = _METRIC
        result["omnidocbench_overall"] = None
        return finalize_shard_fields(result)
    _score_pages(result, image=image, anno_by_name=anno_by_name, limit=limit,
                 enable_thinking=enable_thinking)
    # One shard of a --shard I/N run is a partial set: the shard reason (finalize_shard_fields).
    return finalize_shard_fields(result)


def _score_pages(result: Dict[str, Any], *, image: str, anno_by_name: Dict[str, Dict[str, Any]],
                 limit: Optional[int], enable_thinking: bool) -> Dict[str, Any]:
    """Score the pages of `result["sample_traces"]` with OmniDocBench's evaluator (one run over
    all of them) and set the composite, the per-modality values and the comparability on
    `result`. Used by the run and by merge_shards (the union of a sharded run's pages)."""
    # Write each response as <image-stem>.md, and a GT json holding exactly the generated pages
    # (so a --eval-limit subset scores only what it generated). A page whose request got no reply
    # from the served endpoint (base marks the trace `harness_error`) is a harness error, never
    # the model's outcome: it is left out of the predictions and the GT, so the evaluator does not
    # score it as an empty transcription, and it is counted in `pages_left_out`.
    workdir = tempfile.mkdtemp(prefix="gbench_omnidoc_")
    try:
        return _score_pages_in(result, workdir, image=image, anno_by_name=anno_by_name,
                               limit=limit, enable_thinking=enable_thinking)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _score_pages_in(result: Dict[str, Any], workdir: str, *, image: str,
                    anno_by_name: Dict[str, Dict[str, Any]], limit: Optional[int],
                    enable_thinking: bool) -> Dict[str, Any]:
    pred_dir = os.path.join(workdir, "predictions")
    out_dir = os.path.join(workdir, "result")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    kept: List[Dict[str, Any]] = []
    left_out: List[str] = []
    answered = 0
    for trace in result.get("sample_traces", []):
        gold = _omnidoc_gold(trace.get("gold_answer"))
        page_name = str(gold.get("page_name") or "").strip()
        if not page_name:
            continue
        if trace.get("harness_error"):
            left_out.append(page_name)
            continue
        answered += 1
        stem = os.path.splitext(os.path.basename(page_name))[0]
        with open(os.path.join(pred_dir, stem + ".md"), "w", encoding="utf-8") as f:
            f.write(trace.get("response_text") or "")
        if page_name in anno_by_name:
            kept.append(anno_by_name[page_name])
    result["pages_left_out"] = len(left_out)
    result["omnidocbench_evaluator_revision"] = _EVALUATOR_REVISION
    if not kept and left_out and not answered:
        # Every page's request got no reply: nothing to score. No composite (not a 0); base
        # already made the status `failed` with the accuracy withheld.
        why = (f"no page was scored: all {len(left_out)} page request(s) got no reply from the "
               "served endpoint (harness errors)")
        result["accuracy"] = None
        result["correct_answers"] = 0
        result["accuracy_withheld_reason"] = (
            result["accuracy_withheld_reason"] if result.get("accuracy_withheld_reason") else why)
        result["omnidocbench_overall"] = None
        result["pages_scored"] = 0
        result["metric"] = _METRIC
        result["leaderboard_comparable"] = False
        result["leaderboard_comparable_reason"] = why
        return result
    if not kept:
        raise RuntimeError("omnidocbench: no predictions could be paired to ground-truth pages")
    gt_json = os.path.join(workdir, "gt.json")
    with open(gt_json, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False)

    workers = max(1, min(16, (os.cpu_count() or 8) // 3))
    cmd = _build_command(image, gt_json, pred_dir, out_dir, workers)
    logger.info("omnidocbench: scoring %d pages via %s", len(kept), image)
    proc = subprocess.run(cmd, capture_output=True, text=True)

    summary = _parse_summary(out_dir)
    if not summary:
        raise RuntimeError(
            "omnidocbench: the evaluator produced no run summary (scoring failed) - a "
            "harness/infra failure, not a 0. Last output:\n"
            + (proc.stderr or proc.stdout or "")[-1500:])

    metrics = summary.get("metrics", {})
    overall = summary.get("overall_notebook")
    result["omnidocbench_overall"] = round(overall, 2) if overall is not None else None
    result["text_edit_distance"] = _raw_value(metrics, "text_block_Edit_dist")
    result["formula_cdm"] = _nb_value(metrics, "display_formula_CDM")             # 0-100, higher better
    result["table_teds"] = _nb_value(metrics, "table_TEDS")                       # 0-100, higher better
    result["reading_order_edit_distance"] = _raw_value(metrics, "reading_order_Edit_dist")
    result["pages_scored"] = len(kept)
    # base's category_accuracy tallies the interim per-page pass flag (normalized edit distance
    # <= 0.10), not the composite: kept under its own name, and category_accuracy left empty so
    # no per-subset "accuracy" row reads as the headline's
    result["category_edit_distance_pass_rate"] = result.get("category_accuracy") or {}
    result["category_edit_distance_pass_rate_note"] = (
        "interim per-page pass flag (normalized edit distance <= 0.10) per subset; not the "
        "OmniDocBench composite")
    result["category_accuracy"] = {}
    if overall is not None:
        result["accuracy"] = round(overall, 2)          # canonical headline = the composite
        result["correct_answers"] = 0
        result.pop("accuracy_withheld_reason", None)
    else:
        # The composite needs all three of text/table/formula; a small --eval-limit subset may
        # contain no tables or formulas (TEDS/CDM then have no data). Don't pass off the interim
        # per-page pass rate as the score - mark it undefined, with the reason where the
        # summaries read it (not a harness error: the status stays).
        result["accuracy"] = None
        result["correct_answers"] = None
        result["effective_n"] = len(kept)
        result["omnidocbench_note"] = (
            "composite undefined: the scored pages have no tables and/or display formulas, so "
            "TEDS/CDM have no data. Run the full set (no --eval-limit) for the OmniDocBench "
            "composite; the per-modality values above are over whatever modalities were present.")
        result["accuracy_withheld_reason"] = result["omnidocbench_note"]
        result["status_reason"] = ("accuracy withheld: " + result["omnidocbench_note"])
    result["metric"] = _METRIC
    # Comparable only for the full set, the official prompt (always used) and greedy decoding: the
    # resolved temperature is checked, since --temperature / GBENCH_OMNIDOCBENCH_TEMPERATURE can
    # make even a no-think run sample.
    temperature = result.get("temperature")
    noncanon = []
    if limit:
        noncanon.append(f"--eval-limit subset ({int(limit)} pages), not the full set")
        result["subset_selection"] = SUBSET_SELECTION
    if enable_thinking:
        noncanon.append("--thinking run, not the single-pass greedy protocol of OmniDocBench's "
                        "reference VLM script")
    if temperature is None or float(temperature) != 0.0:
        noncanon.append(f"temperature {temperature} (the reference VLM script decodes greedily)")
    if left_out:
        noncanon.append(f"{len(left_out)} page request(s) got no reply from the served endpoint "
                        "(harness errors, left out of the score; request_harness_errors)")
    cut = record_budget_cutoffs(result)
    if cut:
        noncanon.append(cut)
    result["leaderboard_comparable"] = not noncanon
    if noncanon:
        result["leaderboard_comparable_reason"] = "; ".join(noncanon)
    else:
        result.pop("leaderboard_comparable_reason", None)
    return result


def merge_shards(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The merged result of a sharded run (`gbench --merge`): the composite is a corpus metric
    (the evaluator matches and averages over every page at once), so the shards' composites are
    never averaged. The union of the shards' pages (traces de-duplicated by sample_key, counts
    and statuses re-counted as the generic merge does) is scored again by the same evaluator
    image, which the merge therefore needs (a MergeError names what is missing)."""
    from ...merge import MergeError, generic_merge
    # The generic merge re-counts the union; the shards' composites are set aside (not a
    # correct-over-scored fraction) and the union is scored below.
    merged = generic_merge("omnidocbench", [dict(r, accuracy=None) for r in results])
    live = [r for r in results if not r.get("empty_shard")]
    if not any(t for r in live for t in (r.get("sample_traces") or [])):
        raise MergeError("omnidocbench: no shard carries a generated page to score")
    image = _scorer_image()
    try:
        _check_prereqs(image)
        anno_by_name = _annotations_by_name()
    except RuntimeError as e:
        raise MergeError(f"omnidocbench: merging re-scores the union of the shards' pages with "
                         f"OmniDocBench's evaluator, which is not available here: {e}") from e
    except (OSError, ValueError) as e:
        # huggingface_hub's download errors (LocalEntryNotFoundError, HfHubHTTPError) are
        # OSErrors; an unreadable annotation file is a ValueError (JSONDecodeError).
        raise MergeError(f"omnidocbench: merging re-scores the union of the shards' pages "
                         f"against OmniDocBench's annotation file OmniDocBench.json "
                         f"(opendatalab/OmniDocBench@{_DATASET_REVISION}), which could not be "
                         f"loaded ({type(e).__name__}: {e})") from e
    limit = next((r.get("eval_limit") for r in results if r.get("eval_limit")), None)
    if limit is None and any(r.get("subset_selection") for r in results):
        limit = len(merged.get("sample_traces") or [])
    thinking = any(r.get("thinking") for r in live)
    for key in ("omnidocbench_note", "subset_selection"):
        merged.pop(key, None)
    return _score_pages(merged, image=image, anno_by_name=anno_by_name, limit=limit,
                        enable_thinking=thinking)
