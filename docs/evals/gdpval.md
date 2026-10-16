# gdpval: wins-or-ties rate vs the expert reference

Canonical GDPval (`openai/gdpval`, the public 220-task "gold" subset, arXiv:2510.04374): 220
economically-valuable knowledge-work tasks whose answer is a **produced file deliverable**
(`.pdf`, `.xlsx`, `.docx`, `.pptx`). Every task has a long grading rubric; 185 of them ship the
**expert reference deliverable** (`deliverable_files`). The other 35 ship no deliverable in the
public subset, so they can never enter the pairwise rate (see *Edge cases*).

**Dataset revision:** gbench pins `openai/gdpval` at commit
`11e7900cdcac61bc4daf59e65feb238acda98fbf` (the task table and every attachment/deliverable
download use it; see `_REVISION` in `gdpval.py`).

## Metric: GDPval's win rate (wins + ties)

GDPval grades a model's deliverable against the human-expert reference in a blind pairwise
comparison; graders output {0, 0.5, 1} (loss / tie / win). Its headline "win rate" is the share of
deliverables graded **better than (wins) or as good as (ties)** the expert's (paper Section 3.1
and Figure 5, e.g. Claude Opus 4.1 at 47.6%), with wins-only shown alongside. `run_gdpval`
reproduces this:

1. The model answers each task **3 times** (with input attachments; see below). The paper's
   headline plots sample every model 3 times per task (Section 3.1, footnote 2), so gbench runs
   3 generations per task by default; a larger `--attempt-count` is honoured.
2. The expert reference (`deliverable_files`) is rendered to text with the same converters. The
   rendering carries **no label that marks it as the expert's** (the paper's comparisons are blind
   and its deliverables unlabeled): one rendered file is its text alone, and several are separated by
   a `--- <file name> ---` line. Rendering artifacts (those file headers, sheet and slide headers,
   truncation markers) can still set it apart from a prose answer, so the comparison is not fully
   blind; `leaderboard_comparable_reason` says so.
3. For every generation, a judge is shown the task (its text with the text rendering of its input
   files: every text part of the model's user message, so the rubric's references to the inputs
   can be checked; the input images and audio are not shown to the text judge, and the comparisons
   whose task had any are counted in `gdpval_report.comparisons_judge_without_media_inputs` and
   named), the rubric, and the model answer and the reference as **A / B** (order arbitrary),
   judged against the rubric's content. The comparison is **position-swapped** (model
   as A, then as B) and the two verdicts averaged to cancel order bias. A judge reply counts only
   if its JSON `winner` is exactly `A`, `B` or `tie`; anything else (no JSON, a missing or null
   `winner`, "Deliverable B", ...) is treated like an unanswered call (see *Edge cases*), never
   as a tie.
4. The averaged score maps onto GDPval's outcome: above 0.5 is a **win** (for example a win in
   one position and a tie in the other), exactly 0.5 a **tie** (two ties, or a win and a loss
   that cancel), below 0.5 a **loss**.
5. Headline `accuracy` = **(wins + ties) / scored comparisons** x 100, pooled over all
   generations of all tasks as the paper pools its comparisons; reported as `win_rate_pct` too.
   Secondary: `wins_only_pct` (wins / scored comparisons) and `mean_pairwise_score_pct` (the mean
   of the position-swapped scores with a tie counted as 0.5). A generation's `is_correct` is true
   exactly when it is a win or a tie.

The judge is gbench's standard Gemini cascade (a gbench convention; `judge:
"gbench-gemini-cascade"` on the result; `judge_models`, like `gdpval_report.judge_models_used`,
counts which cascade models answered, and `judge_cascade` lists the cascade). Every comparison
keeps both verdicts auditable in its `pairwise_detail`: `call1_winner` / `call2_winner`, the
Gemini model that gave each (`call1_judge_model` / `call2_judge_model`) and its reply
(`call1_judge_reply` / `call2_judge_reply`, up to 4,000 characters). OpenAI's automated grader is a trained model served at evals.openai.com whose
prompt is not public, so the pairwise prompt (`_PAIRWISE_PROMPT` in `gdpval.py`) is gbench's own.

**`leaderboard_comparable` is always False.** `leaderboard_comparable_reason` names every
protocol difference:
- a plain-text `/v1` endpoint cannot emit an `.xlsx`/`.docx`/`.pdf`, so the expert reference FILE
  is rendered to text and the model competes on rendered **content** only (it can never match live
  formulas/formatting);
- the grader is the Gemini cascade with a gbench prompt rather than OpenAI's graders;
- gbench samples each task **3 times**, like the paper's headline plots (Section 3.1, footnote
  2), but judges each sample once (one position-swapped pair by the cascade), where the paper has
  **3 different human experts** grade each sample (9 comparisons per task);
- tasks without a text-renderable reference are excluded (see *Edge cases*), with their counts by
  cause (`no_deliverable_shipped`, `no_text_rendering`);
- the model is offered no tools, where the paper ran the OpenAI models with the web search and
  code interpreter tools and Claude through its UI with file creation (arXiv:2510.04374,
  footnote 2);
- the rendered reference can be told apart from a prose answer (see step 2);
- the judge does not see the task's image or audio inputs, named with the number of comparisons
  when any task had them;
- generations whose request failed and comparisons without a usable verdict are excluded and
  counted (see *Edge cases*);
- the temperature sent (gbench's 1.0 with `--thinking` or 0.0 without, or a value set with
  `--temperature` or `GBENCH_GDPVAL_TEMPERATURE`; the paper sets none;
  at 0.0 the 3 generations differ only through server nondeterminism), and an `--eval-limit`
  subset, are named when used;
- clipped inputs and clipped judge inputs (see *Clipping*), named with their counts when any
  occurred.
It is a gbench-internal wins-or-ties rate, not OpenAI's GDPval number.

### Edge cases
- A task whose reference has nothing a text judge can read is **excluded** from the rate (not
  scored as a loss) and marked `NO_REFERENCE`. At the pinned revision that is 39 of the 220 tasks:
  35 ship no deliverable (`no_deliverable_shipped`), and 4 have only deliverables with no text
  rendering (`no_text_rendering`: two `.mp4` videos, one PDF drawn without a text layer, and one
  `.zip` of `.wav` audio stems). The other 181 enter the rate. `tasks_no_reference_by_reason`
  reports the split for the run. A full run generates these tasks too; an `--eval-limit` subset
  never draws them (see *Run*). This is the protocol's own rule (the pairwise needs a
  reference), not a harness error: such a trace carries `excluded_by_protocol` (its reason), and
  it does not by itself make the run `completed_with_errors`.
- A **`.zip` deliverable** is opened and its members rendered: text and code files as text,
  documents with the converters below, in archive order and clipped like any other file. OS
  metadata (`__MACOSX`, `.DS_Store`), `.git`, installed dependencies (`node_modules`, virtualenvs),
  build caches (`.next`, `__pycache__`) and lockfiles are counted but not rendered, and formats
  with no text rendering (audio, CAD `.step`, video) are counted by extension. Three tasks have a
  source-code zip as their only deliverable (a React component, a Terraform backend, and a
  Solidity/Next.js project whose upstream archive is cut off at 32 MiB with no central directory,
  so the members before the break are read from their local headers). An archive with no text
  member leaves its task without a reference (the audio stems above); the CAD zip of task
  5e2b6aab contributes nothing, and that task is scored on its three PDF drawings.
- A comparison without a usable verdict, because the cascade gave no answer or its reply had no
  valid `winner`, is excluded too (`comparisons_judge_outage`; `tasks_judge_outage` counts the
  tasks with at least one, and `judge_invalid_replies` the replies without a valid winner).
- A generation whose request failed (no reply at all: a server error, a request timeout or a
  prompt over the context window; base status `FAILED`, `TIMEOUT` or `OVER_CONTEXT`, kept on the
  trace) is a harness error, not the model's deliverable: it is excluded from the rate (never a
  loss), marked `generation_failed`, counted by status in
  `gdpval_report.comparisons_generation_failed` (their total in `comparisons_generation_failed`),
  named in `leaderboard_comparable_reason`, and the run is `completed_with_errors`.
- An empty deliverable the model did return (its reply had no text) **is** counted, as a loss.
- When no comparison can be scored (every sampled task without a text-renderable reference, or
  every comparison without a usable verdict or with a failed request), the run is `failed` and
  its score withheld (`accuracy` null, `accuracy_withheld_reason`), never reported as 0.
- A missing converter package or a failed download is **not** an exclusion: it is a hard error
  (`infra_required`) naming the package or the file. A file of a supported format that fails to
  parse is a hard error as well.
### Clipping
The paper describes no truncation (graders get the request and the full reference files), but
gbench clips five things to keep the model's prompt and each judge prompt inside a context budget.
Each cap has a knob, and every clip is recorded, counted and named:

| what is clipped | cap (knob, default) | recorded |
| --- | --- | --- |
| an input or reference file rendered to text | `GBENCH_GDPVAL_MAX_FILE_CHARS`, 20,000 characters (then `[... '<file>' truncated at N chars]`) | inputs: `inputs_clipped` on the trace; reference: `judge_input_clipped.reference.files` |
| the pages read from a PDF (text, or page images for a PDF with no text layer) | `GBENCH_GDPVAL_MAX_PDF_PAGES`, 10 pages (then `[... N further pages omitted]`, or "the first 10 of N pages as images") | as above |
| the model deliverable, the rendered reference and the task prompt in a judge prompt | `GBENCH_GDPVAL_MAX_JUDGE_CHARS`, 24,000 characters each (then a neutral `[... truncated for this comparison: the first N of M characters are shown]`, the same for either candidate) | `judge_input_clipped.deliverable`, `.reference.judge`, `.task_prompt` |
| the rubric in a judge prompt | `GBENCH_GDPVAL_MAX_RUBRIC_CHARS`, 8,000 characters: the whole criteria that fit, never one cut mid-way (unless the first alone is longer), then `[rubric truncated: K of N criteria shown (X of Y characters)]` | `judge_input_clipped.rubric` |

Each clip entry gives the full and kept size (`{"file", "cut": "chars" or "pdf_pages", "full",
"kept"}` for a file; a PDF cut by pages and then by characters has two entries).
`gdpval_report` totals them: `tasks_with_clipped_inputs`, `input_files_clipped` (distinct files,
per task), `comparisons_rubric_clipped`, `comparisons_reference_clipped`,
`comparisons_deliverable_clipped`, `comparisons_task_prompt_clipped`, and `clip_limits` (the caps
of the run). Non-zero counts are named in `leaderboard_comparable_reason`. Measured offline at the
pinned revision with gbench's converters (two input files, the `reel footage.zip` and a `.pptx`,
were not measured): 13 input files in 12 tasks render to more than 20,000 characters, 4 input PDFs in 3 tasks have more than 10 pages, 32 of 220 rubrics exceed 8,000
characters (the longest 15,978), 17 deliverable files exceed 20,000 characters, 7 deliverable
PDFs exceed 10 pages, and 4 tasks render a reference over 24,000 characters.

`comparisons_scored`, `tasks_scored`, `tasks_no_reference`, `comparisons_judge_outage`,
`comparisons_generation_failed` and `comparisons_unaccounted` are reported so the denominator is
legible. `effective_n` is the number of distinct tasks (prompts) the win rate scored; the
summary CSV's `Effective N` column shows it.

## Requirements

- **`GEMINI_API_KEY`**: the judge. `run_gdpval` hard-errors (`infra_required`) when it is unset
  or rejected; it never returns a fabricated number.
- **HF Hub access** to `openai/gdpval` (the loader downloads the task table and the files at the
  pinned revision; pre-fetching is recommended, see *Fetching the attachments*).
- **Converter packages** (see *Converter dependencies*): `pandas` + `openpyxl` (xlsx), `PyMuPDF`
  (pdf, imported as `pymupdf`), `python-docx` (docx), `python-pptx` (pptx), `soundfile` (the
  duration note for an audio file on the reference side; no reference deliverable at the pinned
  revision is a bare audio file). When a selected task has a file whose converter is not
  importable, the suite hard-errors (`infra_required`) naming the package; it never drops the
  file or the task.
  Images, text/code files, `.ipynb` and `.zip` need only the standard library (a document inside
  a `.zip` needs its own converter).

## Run

```bash
export GEMINI_API_KEY="..."
pip install openpyxl PyMuPDF python-docx python-pptx soundfile
hf download openai/gdpval --repo-type dataset \
    --revision 11e7900cdcac61bc4daf59e65feb238acda98fbf \
    --include "reference_files/*" --include "deliverable_files/*"
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals gdpval --max-output-tokens 65536
```

`--max-output-tokens` is required for every eval run and is recorded as not applied (no
`max_tokens` is sent, see [Sampling](#sampling)). The inputs include images (also the
photographs inside two input `.zip` files), page images for the 5 input PDFs with no text layer,
and 13 audio files (10 `.wav`/`.mp3` inputs across 4 tasks, plus the 3 audio clips inside the
`reel footage.zip` of one of them), sent as `input_audio` parts the way gaia sends its mp3
files. The served model must accept `image_url` and `input_audio` parts; otherwise those
requests fail and are counted as failed requests (gemma-4 26B-A4B and 31B have no audio encoder,
so a server for them rejects the 4 tasks with audio inputs). Prompt sizing counts each audio
part as `GBENCH_AUDIO_SOFT_TOKENS` tokens, as for gaia.

The full 220 tasks run by default, each generated 3 times (660 generations, 1,320 judge calls);
`--eval-limit N` is an opt-in stratified (by occupation) subset.

### The `--eval-limit` subset is drawn from scorable tasks only
A full run keeps every task, the 39 without a text-renderable reference included (they are
generated and then excluded as `NO_REFERENCE`, see *Edge cases*). An `--eval-limit N` subset is
drawn only from tasks whose reference can be scored, which is known before any generation:

1. the file names rule out a task that ships no deliverable (`no_deliverable_shipped`) or whose
   every deliverable is an image, audio or a format with no converter such as `.mp4`
   (`no_text_rendering`): 37 of the 220 at the pinned revision, leaving 183;
2. the N tasks are drawn from those 183, stratified by occupation, and their references are
   rendered; a drawn task whose reference renders no text (the PDF without a text layer, the
   `.zip` of `.wav` stems) is dropped from the pool and the draw retaken.

Selection never looks at a model output. The result records the draw in `eval_limit_draw`
(`tasks_in_pool`, `tasks_without_reference_by_format`, `tasks_with_renderable_reference_format`,
`drawn_without_text_rendering` with each replaced task's id and note, `tasks_drawn`,
`scorable_short_of_limit`; per shard, so `gbench --merge` keeps the shards' differing values under
`merge_unreconciled_fields` while their counts are summed in `eval_limit_plan`), adds
`tasks_in_pool`, `tasks_with_renderable_reference_format`, `drawn_without_text_rendering` (a
count), `scorable_short_of_limit` and `draw_rule` to `eval_limit_plan`, and names the rule in
`leaderboard_comparable_reason` (the counts are in the plan, not in the reason text, so every
shard states the same reason). An N above the scorable tasks (181 unsharded, or a shard's
scorable tasks) runs them all, not the unscorable ones: fewer than N samples, the shortfall
recorded as `scorable_short_of_limit` and logged as a warning. Unsharded, any N from 182 to 219
therefore runs 181 tasks, while N of 220 or more is the full run of all 220.

Whether the draw applies is decided from N against the full 220 tasks, also under `--shard`: any
N below 220 draws from the scorable tasks on every shard, and an N of 220 or more is a full run
on every shard. (2026-10-11 review: the decision was first taken against each shard's size;
220 tasks over 8 shards are 28,28,28,28,27,27,27,27, so `--eval-limit 27` drew from the scorable
pool on the 28-task shards and ran the 27-task shards whole, unscorable tasks included.) Why the
draw exists: the 2026-10-11 smoke run (`--eval-limit 1`) drew task e222075d, whose only deliverable
is an `.mp4`; its 3 generations finished normally and none could be scored, so the run failed
with nothing measured.

The run logs its planned workload in one line at its start (for example `[gdpval] --eval-limit 5:
5 sample(s) x 3 attempts = 15 generations`, preceded by the draw's line `[gdpval] --eval-limit 5
drawn from the 183 of 220 task(s) whose reference has a text-renderable format
(no_deliverable_shipped 35, no_text_rendering 2), 0 dropped on rendering.`) and records it in the
result as `eval_limit_plan` (`limit`, `samples`, `per_setting`, `attempts_per_sample`,
`generations`, plus the draw's fields above).

### Sampling
The paper gives no sampling parameters for the models it evaluates (they were sampled 3 times
per task through their own products and APIs). gbench sends no sampling field other than the
temperature of its rule, which it always sends: **1.0 with `--thinking`, 0.0 without** (for gemma-4, 1.0 is
its default, sent explicitly; at 0.0 the 3 samples are greedy and differ only through server
nondeterminism). `--temperature` or
`GBENCH_GDPVAL_TEMPERATURE` (which takes precedence) sends another value instead; the result
records it under `official_sampling` and names it in `leaderboard_comparable_reason`. The judge
stays at 0.0.

The paper gives no output budget either, so no `max_tokens` is sent, in either mode (the
token-limit policy's "not sent" case, `official_sampling.send_max_tokens: false`): each generation
may use the room left in the context window, the server's default. `--max-output-tokens`, which
the CLI requires, is recorded as not applied in `max_output_tokens_note`, and a generation cut off
at the end of the window is counted in `truncated_responses`.

## What the benchmark actually demands (measured 2026-08-17)

| | |
| --- | ---: |
| tasks | 220 |
| tasks whose input is one or more **attached files** | **125 (57%)** |
| rubric criteria | 10,453 (**47.5 per task**) |
| criteria that depend on file format/properties | **1,474 (14.1%)**, **15.1%** of all rubric points |
| **criteria that are content-only** | **8,979 (85.9%)**, 84.9% of points |

Input file types (all 261): `xlsx` 86, `pdf` 74, `docx` 67, `png` 9, `wav` 8, `zip` 3,
`jpg` 3, `mp3` 2, `webp` 2, `mp4` 2, `step` 2, `txt` 1, `psd` 1, `pptx` 1.
Expected deliverable types: `pdf` 85, `xlsx` 65, `docx` 64, `pptx` 17, and a small tail.

File-property criteria are a bigger share of points (15.1%) than of criteria (14.1%)
because they are weighted higher: 1.84 points each on average versus 1.69 for content
criteria. GDPval scores "is this even the right kind of artifact" above any single
content point.

So the "it's all file properties" framing this doc used to carry was wrong. ~85% of the
points are reachable by a text endpoint. The larger obstacle is the **input** side, and
that one is fixable.

## The files are downloadable, but NOT by `load_dataset`

This is the detail that keeps catching people. The dataset is two-stage:

```python
# stage 1 - metadata only. Pulls ~2 MB: README + the parquet holding prompts,
# rubrics, and the file PATHS as plain strings. NO binaries.
ds = load_dataset("openai/gdpval", split="train")
ds[0]["reference_files"]      # ['reference_files/cc781e.../Population v2.xlsx']  <- a path

# stage 2 - the actual bytes. One call per file, cached afterwards.
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="openai/gdpval",
                       filename=ds[0]["reference_files"][0],
                       repo_type="dataset")
```

Verified on a clean cache: after `load_dataset` alone, **1 of 261** reference files and
**0 of 248** deliverable files are present locally. The suite's loader only ever did stage
one, which is why the attachments never reached the model.

`deliverable_files` are the **expert reference deliverables** and are **REQUIRED**: they are
the reference side of the pairwise comparison. A file that cannot be fetched is a hard error
(`infra_required`), so fetch them before a graded run (see below).

## Fetching the attachments (do this once)

The `hf` CLI pulls everything in one call. `--dry-run` first if you want the size.

```bash
# 1,668 MB, 302 objects - the INPUTS, without which 125 tasks are unanswerable
hf download openai/gdpval --repo-type dataset \
    --revision 11e7900cdcac61bc4daf59e65feb238acda98fbf --include "reference_files/*"

# 623 MB, 249 objects - the EXPERT REFERENCE deliverables (REQUIRED: the pairwise reference)
hf download openai/gdpval --repo-type dataset \
    --revision 11e7900cdcac61bc4daf59e65feb238acda98fbf --include "deliverable_files/*"
```

Both land in the same HF cache `hf_hub_download` reads, so the loader picks them up with
no further network access. The globs fetch slightly more than the tasks reference (302 vs
261 objects, 249 vs 248) because the repo holds a few unreferenced files, harmless.

> **If the download stalls, disable Xet.** `hf_xet` fetches deduplicated chunks and then
> reassembles them (the `Reconstructing ...` progress line). Observed 2026-08-17 on this box:
> wedged at 1.27 GB / 1.31 GB with the process alive but zero throughput, leaving 16
> `.incomplete` blobs. `--force-download` does not help - it re-runs the same backend.
> ```bash
> export HF_HUB_DISABLE_XET=1      # plain HTTPS per file; completed in ~22 s
> ```
> Further knobs if needed: `--max-workers 2`, and `HF_HUB_DOWNLOAD_TIMEOUT` /
> `HF_HUB_ETAG_TIMEOUT` (both default to a rather tight **10 s**).

Validated after download: 261/261 referenced reference files and 248/248 deliverables
present, all with correct magic bytes, 0 missing, 0 zero-byte; **all 220 tasks have their
inputs**. Clean up leftovers with
`find ~/.cache/huggingface/hub/datasets--openai--gdpval -name '*.incomplete' -delete`.

## Converter dependencies

Verified against the real attachments on 2026-08-18: **256 of 261 input files (98%)
read successfully.** Re-checked on 2026-09-24 at the pinned revision with the current converters:
every xlsx, docx and text file and every image renders; 5 input PDFs and 3 deliverable PDFs have
no text layer (scans or vector drawings). The input ones are sent as page images (up to
`GBENCH_GDPVAL_MAX_PDF_PAGES` pages at `GBENCH_GDPVAL_PDF_PAGE_DPI`, default 150 dpi); a
deliverable one contributes a note, and its task is excluded only when no other deliverable of
the task has text.

| type | input files | package | verified read |
| --- | ---: | --- | --- |
| `xlsx` | 86 | `openpyxl` 3.1.5 (via pandas) | 1 sheet, (1516, 8) |
| `pdf` | 74 | `PyMuPDF` (`pymupdf`) | 8 pages, text extracts |
| `docx` | 67 | `python-docx` 1.2.0 | 13 paragraphs, 123 chars |
| `png`/`jpg`/`webp` | 14 | stdlib (sent as a data URI with the file's own mime type) | dimensions + mode |
| `wav`/`mp3` | 10 | stdlib (sent as an `input_audio` part with its own format) | whole file |
| `zip` | 3 | stdlib `zipfile` | 2 of photographs (sent as image parts), 1 of video clips (a count) + 3 audio clips (`input_audio` parts) |
| `pptx` | 1 | `python-pptx` 1.0.2 | 6 slides |
| `txt` | 1 | stdlib | 12,423 chars |
| **`mp4`, `step`, `psd`** | **5** | - | **no converter; see below** |

```bash
pip install openpyxl PyMuPDF python-docx python-pptx soundfile
```

The 5 unreadable input files are 2 `.mp4`, 2 `.step` (CAD) and 1 `.psd`. None has a sensible
text or image projection, so the loader attaches an explicit "this suite cannot render" note for
those tasks rather than sending the prompt as if nothing were missing, and does not download
them. Audio inputs (`wav`/`mp3`) are sent whole as `input_audio` parts. The `reel footage.zip`
input (333 MB) is downloaded and opened: its 3 audio clips go in as `input_audio` parts, and its
13 `.mp4` clips are counted in the archive summary the model sees.
The deliverables add code and query files (`.py`, `.ipynb`, `.yaml`, `.overpassql`), which are
rendered as text (a notebook as its cell sources and text outputs).

> **Trap: `pip install docx` is the WRONG package.** `docx` 0.2.4 is an abandoned 2014
> Python-2 project that also imports as `docx`, so it shadows the real one and fails at
> import with `ModuleNotFoundError: No module named 'exceptions'` (a Py2 builtin removed in
> Python 3). The maintained package is **`python-docx`**. If `docx` got installed by
> mistake, `pip uninstall -y docx` first; installing `python-docx` alongside it leaves the
> broken module winning the import.

## What is implemented

1. **Input ingestion** (the 125 file-input tasks). The loader resolves each `reference_files`
   path and converts it: pdf -> text (PyMuPDF), or page images when it has no text layer;
   xlsx -> CSV (pandas + openpyxl); docx -> text (python-docx); pptx -> text (python-pptx);
   png/jpg/webp -> an image part; wav/mp3 -> an `input_audio` part; zip -> its text members as
   text, its images as image parts and its audio as `input_audio` parts. An unrenderable input
   attaches an explicit note so the task does not read as a model failure.
2. **Reference ingestion + pairwise judge.** The expert `deliverable_files` are rendered to
   text and each of the 3 model answers is compared against them by the Gemini cascade,
   position-swapped, for the wins-or-ties rate (see *Metric* above).

The only path to a number directly comparable with OpenAI's published GDPval win-rate is a
**file-producing agent harness** (tool use / code execution that writes real `.xlsx`/`.docx`/
`.pdf` deliverables), so the judge compares like with like. That is a separate project; until
then this is a gbench-internal wins-or-ties rate and `leaderboard_comparable` stays False.

## Reporting

* `accuracy` / `win_rate_pct`: the wins-or-ties rate (%); `wins_only_pct` and
  `mean_pairwise_score_pct` alongside. `gdpval_report` carries the same as fractions
  (`win_rate` = `wins_or_ties_rate`, `wins_only_rate`, `mean_pairwise_score`).
* `leaderboard_comparable: false`; a 60% rate here is not OpenAI's 60% (text vs a file
  deliverable rendered to text, and a different grader).
* `samples_per_task` (3), `comparisons_scored` and `tasks_scored` name the denominator, with
  `tasks_no_reference` (`tasks_no_reference_by_reason` splits `no_deliverable_shipped` from
  `no_text_rendering`), `comparisons_judge_outage` / `tasks_judge_outage`,
  `comparisons_generation_failed` and `comparisons_unaccounted`; excluded comparisons are not
  counted as losses, an empty answer the model returned is. `total_questions` counts generations
  (tasks x 3); `effective_n` the distinct tasks scored.
* `judge_models` / `judge_cascade`, and per comparison `pairwise_detail` with both verdicts,
  the Gemini model of each and its reply. A run with nothing scored is `failed` with its score
  withheld (`accuracy_withheld_reason`).
* `wins` / `ties` / `losses` are the per-comparison outcomes defined above. `ties_or_split`
  (comparisons that were not a consistent win or loss in both positions), `consistent_pairs` (how
  often the two position-swapped verdicts agreed) and `judge_invalid_replies` expose the
  judgement quality. `attempts` (from the shared runner) adds pass@3 / pass^3 over the tasks.

## Sharding

`--shard I/N` (or `GBENCH_SHARD=I/N`) runs shard I of N of the suite's full, ordered tasks
(round-robin: positions I, I+N, I+2N, ...), selected before the `--eval-limit` draw, which is then
taken within the shard (N shards run up to N x `--eval-limit` tasks). The result records `shard`,
`shard_index` and `shard_count`, and a shard of more than one is not leaderboard-comparable (the
shard reason is named in `leaderboard_comparable_reason`); `gbench --merge` combines the shards'
results as docs/distributed-gbench.md describes: it re-counts the union of their traces, and
refuses a suite whose headline is not a correct / scored count over them unless the suite defines
its own merge.
