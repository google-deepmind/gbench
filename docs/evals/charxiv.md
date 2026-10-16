# charxiv

Canonical CharXiv (Wang et al., NeurIPS 2024 D&B; arXiv:2406.18521;
[princeton-nlp/CharXiv](https://github.com/princeton-nlp/CharXiv);
HF `princeton-nlp/CharXiv`, pinned revision `f441eb632fc62f6f777830a0f47619e6e86459b0`):
realistic academic-chart understanding over the 1000-chart `validation` split, in **two
tracks**:

- **Descriptive**: 4 questions per chart (one of 19 element-reading templates each), 4000 total.
- **Reasoning**: 1 open-ended synthesis question per chart, 1000 total.

Each chart is fanned out into its 5 canonical questions and asked as single-shot multimodal
VQA: the exact upstream prompt text, then the chart image, in the order CharXiv's chat-model
client `src/generate_lib/gpt.py` sends them. The image goes as a base64 data URL whose MIME type
is the one of the image bytes (`image/jpeg` for CharXiv's charts, which `gpt.py` also sends). A
chart whose image cannot be found stops the run; its questions are never sent text-only.

**The chart bytes are the official harness's files.** CharXiv's README has the charts
downloaded as the dataset's `images.zip` and unzipped into `images/`, and `generate.py` sends
`images/{figure_id}.jpg` byte for byte (`gpt.py` `encode_image`). gbench reads the same member
(`<figure_id>.jpg`, the basename of the row's `figure_path`) from `images.zip` at the pinned
revision (141,454,217 bytes, sha256
`c4b151cef2aba35523764dd1091d719d64acf81a16904d72f31da2723efbd855`, checked on every run). The
`image` column of the HF `val.parquet` is not used: its JPEGs are re-encoded (same dimensions,
but none of the 1,000 validation charts is byte- or pixel-identical to the zip's file). The
result records the file under `chart_images`.

## Scoring

**LLM-judge against the exact upstream rubrics**, not string matching. The descriptive and
reasoning grading prompts (per-template / per-inst_category) are vendored verbatim from the
CharXiv repo in `gbench/third_party/eval_suites/charxiv_constants.py`, byte-identical to the values
upstream's `src/constants.py` (commit `7ebe88f`) holds at run time, including the two control
characters (BEL and backspace) that upstream's non-raw `enum` example string produces from
`\alpha` and `\beta`; the judge returns a JSON
`score` of 0 or 1 per answer.

The grading calls are the ones upstream's `src/evaluate.py` makes:

- **Descriptive** answers are grouped by template id (dataset order across charts) and graded
  **up to 5 per prompt** (`build_descriptive_grading_queries`, `nq_per_query=5`), with
  `<|NUM_TRIPLETS|>` set to the batch size, the batch's JSON keys (`extract_answer_T1`,
  `score_T1`, ..., `build_json_keys`) and one `T{i}` triplet per answer. The prompts are
  byte-identical to what upstream's builder produces for the same answers. With `--eval-limit`
  or `GBENCH_SHARD` the batches are formed within the subset, so they are not the full set's
  batches (named in `leaderboard_comparable_reason`). A descriptive reply the server returned
  empty takes its place in its batch and is graded, as upstream batches every response; a request
  that got no reply from the endpoint (a harness error) is not graded and not in any batch.
- **Reasoning** answers are graded one per prompt (an empty reasoning answer is not sent to the
  judge and scores 0). The reasoning question and its answer are used as the dataset holds them
  (upstream's `build_reasoning_queries` and grading queries do not strip them).
- A reply is accepted as upstream accepts it: `json.loads` of the whole reply and, for
  descriptive prompts, `verify_grading_output` (every key present, every score in `[0, 1]`);
  reasoning needs the `extracted_answer` and `score` keys. A reply that fails is asked again,
  up to upstream's **10 attempts** per prompt (`max_retries=10`). After that the prompt's
  answers are `JUDGE_FAILED`. The grader here is gbench's Gemini substitute, so its failure is a
  judge failure, never the model's outcome: those answers are left out of their track's
  denominator (`scoring_excluded`), counted in `judge_failures` and each track's `judge_failed`,
  named in `leaderboard_comparable_reason`, and the run reports `completed_with_errors`.
  Upstream gives them the dummy score -1, which `get_stats.py` counts as invalid and as 0; that
  figure is kept beside each track's accuracy as `accuracy_upstream_invalid_as_zero`. A reasoning
  score outside `[0, 1]` is `JUDGE_FAILED` without a retry, as upstream accepts that reply and
  `get_stats.py` scores it 0.
- A judge-cascade outage (every Gemini model failed every round) is `JUDGE_OUTAGE`: a harness
  error, excluded from the denominator and counted in `judge_outages`.
- Every graded trace records `judge_model` (the Gemini model that answered), `judge_raw` (the
  reply, so it can be parsed again), `judge_score`, `extracted_answer` and `judge_batch`
  (template id, position, batch size, calls). The result records `judge`, `judge_cascade` and
  `judge_models` (answered judge calls per model).

Two numbers are reported **separately** (the paper reports no combined figure; its
leaderboard is sorted by Reasoning, so that is the headline `accuracy`):

- `charxiv_reasoning_accuracy`, the headline.
- `charxiv_descriptive_accuracy`.

`total_questions`, `correct_answers` and `effective_n` (the answers graded) count the headline
track, so the summary row's Questions / Correct / Accuracy agree; the pooled counts over both
tracks are `questions_all_tracks` / `correct_all_tracks`, and `charxiv_tracks` holds each track's
questions, graded answers, correct answers, outages and judge failures. When neither track has a
graded answer the accuracy is withheld (`null` with `accuracy_withheld_reason`).

**Judge backend.** Canonical CharXiv pins `gpt-4o-2024-05-13`; gbench grades with its standard
Gemini judge cascade - the reference grader it uses across all judged suites, by convention (a
grader choice, not a fidelity defect). The prompts (batched as upstream batches them), rubrics and JSON contract are identical. The
numbers are CharXiv-protocol-faithful, graded by gbench's cascade rather than the paper's
GPT-4o, so they are a gbench-internal number rather than a like-for-like leaderboard entry
(`leaderboard_comparable=false`); label accordingly against the paper's GPT-4o leaderboard.
`leaderboard_comparable_reason` states this, names the judge `max_tokens` that is not sent (see
Requirements), judge failures and outages, an `--eval-limit` subset, a `GBENCH_SHARD`
shard, a temperature other than the mode's (gbench's 1.0 of a thinking run, or an operator-set value) and
replies cut off by the run's token budget rather
than at an official cap (`replies_cut_off_by_budget`; `replies_cut_off_in_reasoning` counts
those that ran out inside the reasoning channel with no answer); `dataset` names the pinned
revision.

## Requirements

- A **vision-language** (multimodal) model endpoint (observations are chart images).
- **`GEMINI_API_KEY`** or **`GEMINI_API_KEYS`** (one key, or a comma/semicolon list;
  `GEMINI_API_KEYS` wins when both are set): required (the judge grades every question; without
  it the suite **hard-errors** via `infra_required`, never skipping and never scoring on
  fabricated data). No other API keys.
- `datasets` (a core gbench dependency) to fetch the pinned `princeton-nlp/CharXiv`
  `validation` split from the HF Hub (about 150 MB with the test split; needs network on first
  run, cached thereafter), and `huggingface_hub` for the pinned `images.zip` (about 141 MB,
  downloaded once into the HF cache; a failed download is a hard error, `infra_required`). To
  pre-fetch it: `hf download princeton-nlp/CharXiv images.zip --repo-type dataset --revision
  f441eb632fc62f6f777830a0f47619e6e86459b0`.
- No new pip packages: the grading rubrics are vendored in-tree.
- **Sampling**, per mode, as CharXiv's API clients send it (`src/generate_lib/` @ `7ebe88f`).
  The temperature follows gbench's rule in both modes, **1.0 with `--thinking`, 0.0 without**
  (`gpt.py`'s 0.0 is recorded). Without `--thinking`, the rest of the chat-model client `gpt.py`:
  top_p 1.0, seed 42 and **`max_tokens` 1,000**, so `--max-output-tokens` is not applied (`max_output_tokens_note`) and a
  reply that stops at 1,000 tokens is graded as it arrived (`replies_at_official_max_tokens`
  counts them). With `--thinking` the same `gpt.py` settings apply except the 1,000-token cap,
  written for non-reasoning generation, which is replaced by `--max-output-tokens` under the
  token-limit policy (recorded in `official_sampling`). CharXiv routes OpenAI's o1 models to its
  `o1.py` client by model name; a setting keyed on a hosted model's name is not a reasoning-model
  setting for a served model, so `o1.py` is not used. `GBENCH_CHARXIV_TEMPERATURE` (or
  `--temperature`) overrides the temperature in both modes; the result records `official_sampling`,
  and a
  temperature other than the official one is named in `leaderboard_comparable_reason`.
- **Judge call:** CharXiv's `get_descriptive_result_gpt` / `get_reasoning_result_gpt`
  (`descriptive_utils.py`, `reasoning_utils.py`): JSON mode, temperature 0, top_p 1 and seed 42,
  sent to the Gemini cascade, with upstream's batching (5 descriptive answers per prompt) and
  retries (10 attempts); `judge_sampling` records them. Their `max_tokens` (256, doubled up
  to 1,024 when the JSON is cut off) is not sent: it was written for gpt-4o, which does not
  think, and the Gemini judge counts its thinking against the same limit. This deviation is
  named in `leaderboard_comparable_reason`.

## Run
```bash
export GEMINI_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals charxiv \
       --max-output-tokens 65536 --eval-limit 20
```
`--eval-limit N` runs **N questions** in total, dealt over the two tracks (descriptive and
reasoning, one before either gets two); each track draws its share of questions stratified by
chart subject with a fixed seed. `--eval-limit 5` sends 5 questions (e.g. 3 descriptive and 2
reasoning), not 5 charts x 5 questions. The headline track (Reasoning) always gets a question: `--eval-limit 1`
runs one reasoning question. The run logs the plan at its start and records it as
`eval_limit_plan`.

## Sharding

`--shard I/N` (or `GBENCH_SHARD=I/N`) runs shard I of N of the suite's full, ordered charts (the
validation split's order) (round-robin: positions I, I+N, I+2N, ...), selected before the
`--eval-limit` draw, which is then taken within the shard (N shards run up to N x `--eval-limit`
questions). A chart's five questions stay in one shard. The result records `shard`, `shard_index` and
`shard_count`, and a shard of more than one is not leaderboard-comparable (the shard reason is
named in `leaderboard_comparable_reason`); `gbench --merge` combines the shards' results as
docs/distributed-gbench.md describes: it re-counts the union of their traces, and refuses a suite
whose headline is not a correct / scored count over them unless the suite defines its own merge.
