# coco_caption setup

MS-COCO image captioning on the **Karpathy test split** (5,000 images), the split published
COCO captioning results (papers, PaliGemma/Gemma reports) use, with lmms-eval's
`coco_karpathy_test` protocol and the COCO caption metrics from `pycocoevalcap`.

## Data
- Split and references: `yerevann/coco-karpathy` `test`, pinned revision `448fdb1b` (the
  dataset lmms-eval's `coco_karpathy_test` loads): 5,000 COCO-2014 val images, each with its
  Karpathy reference sentences (5 per image; 6 for ten images).
- Images: lmms-eval downloads each row's `url` (`http://images.cocodataset.org/val2014/...`).
  gbench takes the same files from `lmms-lab/COCO-Caption` `val`, pinned revision `c3213b54`,
  matched by file name; those bytes are the val2014 JPEGs served at the COCO URLs (checked byte
  for byte on a random sample of 30 test images), and they are sent unchanged. All 5,000 test
  file names are present in that split; a missing or undecodable image stops the run.

## Protocol
- As lmms-eval `lmms_eval/tasks/coco_cap/coco_karpathy_test.yaml` and `utils.py`
  (@ `42296f8f`): one request per image, the image followed by the prompt
  `Describe the image briefly.`
- Sampling (the token-limit policy): lmms-eval generates greedily (temperature 0, top_p 1.0)
  with at most 64 new tokens, which its chat-endpoint client (`lmms_eval/models/chat/openai.py`)
  sends as `max_tokens` (`num_beams` and `do_sample` are not sent by that client). Without
  `--thinking` every request carries temperature 0, top_p 1.0 and **`max_tokens` 64**, so
  `--max-output-tokens` is not applied (`max_output_tokens_note` says so), and a caption that
  stops at 64 tokens is scored as it arrived (`replies_at_official_max_tokens` counts them).
  With `--thinking` lmms-eval's sampling is the same, temperature 0 and top_p 1.0, but gbench
  samples every thinking request at **temperature 1.0** (top_p stays 1.0), by its rule
  (1.0 with `--thinking`, 0.0 without)
  (`official_sampling` records `temperature_matches_official: false`). lmms-eval's client
  runs a thinking model with its `enable_thinking_kwarg`, which only adds
  `chat_template_kwargs.enable_thinking` and keeps the task's `generation_kwargs`
  (`build_payload_for_index`, lines 190-223). The client's other branch, for model names
  containing o1, o3, o4 or gpt-5 (temperature dropped, `max_completion_tokens`), matches only
  those OpenAI models and is not used. lmms-eval declares no reasoning-model cap, so the 64,
  written for non-reasoning generation, is not applied to a `--thinking` run: each request is
  sent the run's `--max-output-tokens`, else the 32,768 thinking floor, and the result records
  the cap as not applied to reasoning (`official_sampling.max_tokens_applied: false`,
  `max_output_tokens_note`). `--temperature` or `GBENCH_COCO_CAPTION_TEMPERATURE` override the
  temperature in both modes. Any thinking text is stripped before scoring. The result records
  `official_sampling` (the settings, the temperature used, its source,
  `temperature_matches_official` and `max_tokens_applied`, false with `--thinking`).
- Scoring as `coco_aggregation_result`: captions and references are tokenized with the PTB
  tokenizer and scored at corpus level with BLEU-1..4, METEOR, ROUGE-L and CIDEr
  (pycocoevalcap's CIDEr-D implementation), the seven metrics of the task's `metric_list`
  (SPICE is commented out there, and it is not computed). CIDEr is the headline. An empty reply
  is the model's caption and is scored as an empty one (lmms-eval: `result[0] or ""`). A request
  that got no reply from the served endpoint (failed, timed out, or over the context window) is
  a harness error, never scored as an empty caption: its image is left out of the corpus (and of
  `per_image_pass_rate`), counted and named in `status_reason`; with no image left, the corpus
  metrics are not computed and `accuracy` is null with `accuracy_withheld_reason`.
- Before generating anything the suite runs every scorer on a two-image fixture; if Java, the
  tokenizer jar or the METEOR jar fails, it hard-errors (`infra_required`). A scorer failure
  after generation raises; no metric is ever dropped from the result.

## Reading the result
- `accuracy` is corpus CIDEr x100 (a score, not a percentage); `cider`, `bleu1`..`bleu4`,
  `meteor` and `rouge_l` are the raw corpus metrics (lmms-eval's `coco_CIDEr`, `coco_Bleu_1`..,
  `coco_METEOR`, `coco_ROUGE_L`).
- `per_image_pass_rate` is the share of images with per-image CIDEr >= 0.5, a convenience
  figure only (`per_image_passes` counts them, `category_cider_pass_rate` tallies them per
  category); each trace records its per-image `cider`. The rows next to the headline carry the
  headline: `category_accuracy` holds the corpus CIDEr x100 for the one category (`captioning`),
  and `correct_answers` is `null` (`correct_answers_note`), so the summary's Correct column and
  category rows never show the per-image flag next to the corpus score.
- `leaderboard_comparable` is `true` for a full, unsharded run at the official temperature (0;
  a `--thinking` run samples at 1.0, so it is not comparable) in which every request got a reply and
  no reply was cut off by the
  run's token budget. Otherwise `leaderboard_comparable_reason` names each difference: the
  `--eval-limit` subset (CIDEr's IDF weights then come from the scored images' references, so a
  subset's CIDEr is not the full-split CIDEr), the shard of a `--shard I/N` run (a shard's IDF
  weights come from its own images), a temperature
  other than 0 (set with `--temperature` or the variable above, or the 1.0 of a thinking run), the
  requests that got no reply (failed, timed out or over
  the context window; harness errors, left out of the score), and the replies cut off by the run's
  token budget (`replies_cut_off_by_budget`; `replies_cut_off_in_reasoning` counts those that ran
  out inside the reasoning channel with no answer).
- `dataset` names both pinned revisions.
- `--eval-limit N` draws a seeded random subset of N images.
  The run logs its planned workload at its start and records it as `eval_limit_plan`.

## Requirements
- The `evals` extra (`pip install -e ".[evals]"` in the gbench checkout; installs `pycocoevalcap`,
  which pulls in `pycocotools`).
- A **Java runtime** on `PATH` (the PTB tokenizer and METEOR run bundled Java jars), for
  example `apt install default-jre-headless`. Checked with OpenJDK 21.
- A **vision-capable** endpoint (the image is sent as a base64 data URL to
  `/v1/chat/completions`).
- `datasets`, which downloads both pinned revisions from the Hugging Face Hub on first use:
  the Karpathy split (about 25 MB) and `lmms-lab/COCO-Caption` (its `val` and `test` parquet
  files, about 6.2 GB each; the 5,000 Karpathy test images are taken from `val`), plus the
  prepared Arrow cache `datasets` writes next to them. Plan for about 26 GB of free disk.
- No API keys.

If `pycocoevalcap`, Java or a scorer is missing the suite hard-errors (`infra_required`)
before generating anything; it never skips or emits a fabricated number.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals coco_caption \
       --max-output-tokens 65536 --eval-limit 500
```
Drop `--eval-limit` for the full 5,000-image run.

## Sharding

`--shard I/N` (or `GBENCH_SHARD=I/N`) runs shard I of N of the suite's full, ordered images (the
Karpathy test split's order) (round-robin: positions I, I+N, I+2N, ...), selected before the
`--eval-limit` draw, which is then taken within the shard (N shards run up to N x `--eval-limit`
images). The result records `shard`, `shard_index` and `shard_count`, and a shard of more than one
is not leaderboard-comparable (the shard reason is named in `leaderboard_comparable_reason`).
The corpus metrics are never averaged over shards: `gbench --merge` computes them again over the
union of the shards' captions (so the merge needs pycocoevalcap and Java, as a run does) and
re-counts the per-image pass rate from the union's per-image CIDEr.
