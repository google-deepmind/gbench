# omnidocbench

Canonical OmniDocBench (`opendatalab/OmniDocBench`, CVPR 2025): multimodal document parsing.
The model reads a PDF page **image** and converts it to **Markdown**, with tables as HTML
(`<table>...</table>`) and formulas as LaTeX (inline `\( \)`, block `\[ \]`), as OmniDocBench's
own VLM instruction asks. Scoring is OmniDocBench's **end2end per-modality composite**,
computed by OmniDocBench's own evaluator:

**Overall = ((1 - Text Edit Distance) * 100 + Table TEDS + Formula CDM) / 3**

(TEDS and CDM on a 0-100 scale.)

- **Text** (text_block/title/references/captions) -> normalized edit distance
- **Table** -> TEDS (tree-edit similarity)
- **Display formula** -> **CDM** (Character Detection Matching, renders LaTeX and matches)
- **Reading order** -> edit distance

> **On figures:** OmniDocBench does **not** score figure *image content*. The `figure` category
> is in no scored group (only its text caption/footnote is scored), and the instruction tells the
> model to ignore figures. It is a text/table/formula parsing benchmark, not figure reproduction.

## Pinned sources

| What | Pin |
| --- | --- |
| Dataset: `opendatalab/OmniDocBench` (HF Hub) | revision `aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec` (v1.6 data: 1651 annotated pages; `OmniDocBench.json` unchanged since the v1.6 commit `d386947f`) |
| Evaluator: `github.com/opendatalab/OmniDocBench` | commit `193627ae9e97d89188468ed1ee3b7a856ff76044` (upstream HEAD `f133a71e` on 2026-09-24 differs from it only in `README.md`/`README_zh-CN.md`) |

Both are pinned in code. The dataset is downloaded at that revision. The evaluator runs from a
locally built image (below); before generating anything the suite runs `git rev-parse HEAD` and
`git status` on the checkout inside the image and **hard-errors** unless it is exactly the pinned
commit with no changes to tracked files. The verified commit is recorded on the result as
`omnidocbench_evaluator_revision`.

## Protocol

- **Prompt:** OmniDocBench's general-VLM instruction **verbatim** ("You are an AI assistant
  specialized in converting PDF images to Markdown format...", from
  `tools/model_infer/Qwen2VL_img2md.py`; the gemini25/gemini3/internvl2 scripts use the same
  text). The evaluator's TEDS/CDM matching relies on its HTML-table and LaTeX conventions.
- **Request layout:** one user turn holding the page image followed by the instruction, as the
  Qwen2VL reference script builds it.
- **Decoding:** the reference script decodes greedily (`do_sample=False`); gbench sends the
  temperature of its rule, **1.0 with `--thinking`, 0.0 without**, and the result records `official_sampling` (the
  reference, the temperature used, its source and `temperature_matches_official`). Without
  `--thinking` every request carries the reference script's `max_new_tokens`, **32000**, as its
  `max_tokens` (shrunk only when the page prompt leaves less of the context window), so
  `--max-output-tokens` is not applied (`max_output_tokens_note` says so) and a page that stops
  there is scored as it arrived (`replies_at_official_max_tokens` counts them). The script
  defines no reasoning-model settings, so with `--thinking` that cap, written for non-reasoning
  generation, is not applied: each request is sent the run's `--max-output-tokens` (the
  token-limit policy; `official_sampling` records `max_tokens_applied: false` and the budget
  sent).
- **Pages:** every annotated page of the split (1651). The HF split also carries a few dataset
  artwork images with no annotation (`data_diversity.*`, `show_pdf_types_*`); they are not
  benchmark pages and are excluded, and the count is logged.

## Why it's containerized

OmniDocBench pins **Python <3.12** and **numpy==1.24.4**, which would break the torch/vLLM
serving env, and its CDM metric needs a TeX Live / ImageMagick 7 / Ghostscript runtime. So gbench
does **not** install it into the gbench serving environment. Instead it runs OmniDocBench's
`pdf_validation.py` **inside a locally built image**: gbench generates the markdown
transcriptions in-process (multimodal prompts against your endpoint), writes them out, and scores
them in the image with OmniDocBench's end2end config (`quick_match`, the metric set of its
`configs/end2end.yaml`). gbench hard-errors (never silently skips) if Docker or the image is
missing or the image holds another revision.

## Container and network

The evaluator container compiles the model's LaTeX: the CDM formula metric renders every
predicted display formula with `pdflatex` (OmniDocBench's `latex_render_percentage.py`, no
`-shell-escape`). It reads only its mounted files (the ground-truth JSON, the predictions and the
result directory) and the image; the end2end metrics (`Edit_dist`, `CDM`, `TEDS`) load nothing at
run time (`BLEU` and `METEOR`, which would load through `evaluate.load`, are not in the config),
and the NLTK data is baked into the image. So gbench runs it with `--network none`, as it does the
image's revision check: no container of this suite reaches the network, so the cloud metadata
server needs no host rule for it and there is no host name to resolve. (Checked 2026-09-27: the
pinned image scores two pages with display formulas and a table, run as gbench runs it with
`--network none`, and returns the same run summary as on docker's bridge network.) The page
images and annotations are downloaded on the host before any container starts.

## Setup

gbench builds the evaluator image **locally** (it never pulls registry images). The gbench
Dockerfile mirrors OmniDocBench's own repro runtime (Python 3.10 + Ghostscript + TeX Live with CJK
+ ImageMagick 7, PDF-enabled) and installs the checkout's pinned deps. Build it from a **git
checkout** at the pinned commit (the build context must include `.git`, which the run-time
revision check reads):

```bash
git clone https://github.com/opendatalab/OmniDocBench $GBENCH_PREREQS_DIR/OmniDocBench
git -C $GBENCH_PREREQS_DIR/OmniDocBench checkout 193627ae9e97d89188468ed1ee3b7a856ff76044

# run from the gbench repo root; context = the OmniDocBench checkout
docker build -t gbench-omnidocbench:193627ae \
    -f docker/omnidocbench.Dockerfile \
    $GBENCH_PREREQS_DIR/OmniDocBench
```
ImageMagick is the 7.1.2-31 clang AppImage, checked against its sha256
(`87790f774f349089f83a138721ec3fffa0602b174a3a6c064d119c870f687c87`). OmniDocBench pins its direct
dependencies; theirs install against the constraints written inline in the Dockerfile
(`PIP_CONSTRAINT`: the installed set of the image built 2026-09-08), and the build fails if the
installed set differs. The NLTK `punkt` and `punkt_tab` data come from the nltk_data commit
`4f15a3d89eefe9748ec1c05be495d91289197155`, each checked against its sha256. The image records its
Python and Debian packages in `/opt/gbench`.
This is a large image (TeX Live); the first build is slow. Rebuild it whenever the pinned commit
changes. Override the tag with `GBENCH_OMNIDOCBENCH_IMAGE`. The dataset (page images +
`OmniDocBench.json` annotations) is pulled from the HF Hub at the pinned revision at run time.

Also required on the host:
- Docker CLI + a running daemon.
- `rapidfuzz` (in gbench's `evals` extra), used for the interim per-page pass flag;
  checked before generation.
- A **multimodal** OpenAI-compatible endpoint (image input).

## Run

```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-26B-A4B-it --evals omnidocbench \
       --max-output-tokens 32000 \
       --suite-timeout 36000            # CDM rendering is slow; raise the timeout
```
`--max-output-tokens` is required by the CLI; it is the budget only with `--thinking` (without
it, every request carries the reference script's 32000).

What happens: gbench generates one markdown transcription per page against your endpoint, writes
each as `<image-stem>.md` plus a ground-truth json of exactly the pages it generated, then
`docker run`s the evaluator image (mounting the predictions, GT, and an output dir) to compute the
composite. It reads back `result/predictions_quick_match_run_summary.json` and reports
**`accuracy` = the composite**, with `text_edit_distance`, `table_teds`, `formula_cdm`,
`reading_order_edit_distance`, `pages_scored` and `omnidocbench_evaluator_revision` alongside.
A page whose request got no reply from the served endpoint (it failed, timed out or did not fit
the served context window after gbench's retries) is a harness error, never the model's outcome:
it is left out of the predictions and the ground truth the evaluator scores, counted in
`pages_left_out` (and in `request_harness_errors` by kind), named in `status_reason`, and the
status is `completed_with_errors`. When every page is one, the evaluator does not run: the status
is `failed` and `accuracy` is `null` with `accuracy_withheld_reason`, never 0.

## Reading the result

- **`leaderboard_comparable`** is `true` for a full run (no `--eval-limit`) decoded greedily:
  the resolved `temperature` on the result must be 0.0, `--thinking` off, and every page must
  have got a reply. A `--temperature` or `GBENCH_OMNIDOCBENCH_TEMPERATURE` override, a
  `--thinking` run, a limited run, requests that got no reply (failed, timed out or over the
  context window; left out of the score) or replies cut off by the run's token budget rather than at
  an official cap (`replies_cut_off_by_budget`; `replies_cut_off_in_reasoning` counts those that
  ran out inside the reasoning channel with no answer) sets it to `false`, and
  `leaderboard_comparable_reason` names each cause.
- `category_accuracy` is empty: base's per-subset tally of the interim per-page pass flag
  (normalized edit distance <= 0.10) is kept as `category_edit_distance_pass_rate` (with a note),
  so no per-subset "Accuracy" row reads as the composite.
- `accuracy` is `null` (with `omnidocbench_note`, repeated as `accuracy_withheld_reason` and in
  `status_reason`; `correct_answers` null and `effective_n` the pages scored; the status is not
  changed, this is no harness error) when the scored pages contain no tables or no
  display formulas, because the composite then has no TEDS/CDM term; this only happens on small
  `--eval-limit` subsets. A subset needs pages with tables and with display formulas for the
  composite to exist, so `--eval-limit N` does not take the split's first N pages (the split runs
  one data source at a time, and its first 30 pairable pages are all PPT2PDF slides without a
  display formula): it pairs every page with its annotation first, then picks N of the 1651
  pairable pages stratified by their subset (`page_info.page_attribute.subset`: `v1.5` 1355,
  `equation_hard` 100, `layout_hard` 99, `table_hard` 97 pages) and, within each subset, by their
  data source (`page_attribute.data_source`: book, PPT2PDF, academic_literature, exam_paper, ...),
  seeded (`omnidocbench`), and records how as `subset_selection`. With N of at least 4 every
  subset is represented, so the pages include a `table_hard` page (all 97 hold a table) and an
  `equation_hard` page (91 of 100 hold a display formula). The strata are sampled evenly, so a
  small subset over-represents the three hard subsets relative to the full set; it is a smoke
  run, never comparable.
- The harness's per-page pass flag (normalized edit distance <= 0.10 on each trace) is interim
  only and `correct_answers` is reported as 0: the headline is the evaluator's composite.

## Configuration

| Env var | Purpose |
| --- | --- |
| `GBENCH_OMNIDOCBENCH_IMAGE` | evaluator image tag (default `gbench-omnidocbench:193627ae`, the image built LOCALLY from `docker/omnidocbench.Dockerfile` above; gbench never pulls from a registry) |
| `GBENCH_OMNIDOCBENCH_TEMPERATURE` | per-suite sampling temperature override (a non-zero value makes the run non-comparable) |

## Notes / caveats

- **Runtime:** CDM renders every predicted formula with pdflatex + ImageMagick. The full run is
  slow. Use `--eval-limit N` for a smoke test (N pages).
  The run logs its planned workload at its start and records it as `eval_limit_plan`.
- **No LLM judge:** scoring is OmniDocBench's static per-modality evaluator (CDM/TEDS/edit
  distance), not a model.

## Sharding

`--shard I/N` (or `GBENCH_SHARD=I/N`) runs shard I of N of the suite's full, ordered annotated
pages (round-robin: positions I, I+N, I+2N, ...), selected before the `--eval-limit` draw, which
is then taken within the shard (N shards run up to N x `--eval-limit` annotated pages). The
composite is a corpus metric, so the shards' composites are never averaged. The result records
`shard`, `shard_index` and `shard_count`, and a shard of more than one is not
leaderboard-comparable (the shard reason is named in `leaderboard_comparable_reason`); `gbench
--merge` scores the union of the shards' pages again with the same evaluator image, which the
merge therefore needs.
