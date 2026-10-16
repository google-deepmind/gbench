# bundled_detection

Zero-shot object detection by a chat VLM on the MS-COCO 2017 validation set (all 5,000 val2017
images, 80 categories), scored by the COCO detection challenge's own evaluator (pycocotools
`COCOeval`). The prompt is gbench's, because COCO defines none for a chat model, so the result
always carries `leaderboard_comparable: false`.

## Dataset (pinned, official COCO release)
- Images: `http://images.cocodataset.org/zips/val2017.zip` (5,000 JPEGs, about 780 MB), sha256
  `4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05`.
- Ground truth: `annotations/instances_val2017.json` (sha256 `e8c7f790...`) from
  `http://images.cocodataset.org/annotations/annotations_trainval2017.zip` (about 250 MB, sha256
  `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268`): 36,781 boxes of the 80
  categories, 446 of them crowd regions; 48 images have no box and are requested too.
- Both zips are downloaded once to `$GBENCH_CACHE_DIR/coco/` (default `~/.cache/gbench/coco/`)
  and checked against their sha256 on every run; the ground truth is extracted there once. A
  download failure is a hard error (`infra_required`), a checksum mismatch raises, and so does a
  ground truth that is not 5,000 images, 36,781 boxes and the 80 categories with their COCO ids.

## Protocol
- One request per image: the image (its JPEG bytes from val2017.zip), then the fixed COCO-80
  category list and the instruction to output ONLY a JSON array of
  `{"label", "box_2d", "confidence"}`, with `box_2d` as `[ymin, xmin, ymax, xmax]` normalized to
  [0, 1000] and `confidence` in [0, 1]. The prompt does not reveal which categories are present.
  This prompt is gbench's own.
- Parsing: the first JSON array of objects in the reply; a box given in [0, 1] is scaled to
  [0, 1000]; a missing confidence counts as 1.0.
- Each detection becomes a COCO result: `category_id` from its label (a label outside the 80
  categories cannot be a COCO result and is counted in `coco_unknown_labels`), the box converted
  to `[x, y, width, height]` in pixels of that image (a box without positive width and height is
  counted in `coco_invalid_boxes`), and the confidence as the `score`.
- **Scoring: pycocotools `COCOeval` with `iouType="bbox"`**, exactly as the COCO challenge scores
  detections: per category, detections ranked by score and matched at IoU .50:.05:.95, crowd
  regions ignored as COCOeval ignores them, the area ranges by segmentation area, at most 100
  detections per image, 101-point interpolated precision. Every image the model answered
  counts: an image whose reply has no usable box has no detections, and its ground truth still
  counts against recall. An image whose request got no reply from the served endpoint (failed,
  timed out, or over the context window) is a harness error, never the model's outcome: it is
  left out of the evaluation (its ground truth too), counted and named in `status_reason`; with
  no image scored, `accuracy` is null with `accuracy_withheld_reason`. Checked: on the official
  ground truth file, the
  suite's numbers equal a direct pycocotools evaluation bit for bit (subsets of 200 and 500
  images and the full 5,000).
- Sampling: gbench's temperature rule (0.0 greedy without `--eval-thinking`, 1.0 with it); COCO
  defines none. `--temperature` or `GBENCH_BUNDLED_DETECTION_TEMPERATURE` override it.

## Reading the result
- `accuracy` = `map_50_95` = COCO **AP** (IoU .50:.95, area all, 100 detections) x100, the
  challenge's primary metric.
- `coco_stats`: the twelve `COCOeval.summarize()` numbers x100: `AP`, `AP50`, `AP75`,
  `AP_small`, `AP_medium`, `AP_large`, `AR1`, `AR10`, `AR100`, `AR_small`, `AR_medium`,
  `AR_large` (null where the scored images have no box of that size). `map_50` = `AP50`.
- `coco_detections`, `coco_unknown_labels`, `coco_invalid_boxes`: detections scored and
  dropped.
- `hit_rate_any_box`: share of images with at least one correctly labelled box at IoU >= 0.5
  against a non-crowd box, a coarse secondary figure (`images_with_any_box` counts them,
  `category_hit_rate` tallies them by each image's most frequent ground-truth category). COCOeval
  AP is a corpus metric, so `category_accuracy` is empty and `correct_answers` is `null`
  (`correct_answers_note`): the summary shows no per-image flag next to the AP headline.
- `leaderboard_comparable` is always `false`: the prompt is gbench's and the detections are
  ranked by the model's self-reported confidences. `leaderboard_comparable_reason` says so,
  gives the number of images scored (naming an `--eval-limit` subset or a `GBENCH_SHARD` shard),
  and counts requests that got no reply (harness errors, left out of the score) and replies cut
  off
  by the run's token budget rather than at an official cap (`replies_cut_off_by_budget`;
  `replies_cut_off_in_reasoning` counts those that ran out inside the reasoning channel with no
  answer).
- `--eval-limit N` draws N images stratified over each image's most frequent category ("none"
  for the 48 images without a box); COCOeval then scores exactly those images.
  The run logs its planned workload at its start and records it as `eval_limit_plan`.

## Requirements
- A **vision-language model** endpoint (images are sent as data URIs).
- `pycocotools` (in the `evals` extra: `pip install -e '.[evals]'`); without it the suite
  hard-errors (`infra_required`) before generating.
- Network access to `images.cocodataset.org` on the first run (about 1 GB), then the cache.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals bundled_detection \
       --max-output-tokens 65536 --eval-limit 20
```
Drop `--eval-limit` for the full 5,000-image run.
