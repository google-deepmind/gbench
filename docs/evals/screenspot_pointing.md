# screenspot_pointing

**ScreenSpot-v2 with a text pointing prompt**, renamed from `semantic_keypoint` (there is no
upstream "semantic keypoint" benchmark). The old name still runs this suite and logs a one-line
deprecation warning; it is not listed in `--help`, `--list evals` or `--evals all`. The suite
runs every target of the official ScreenSpot-v2 release and asks for the target's centre point
as text. The result always carries `leaderboard_comparable: false` (see Prompt).

## Dataset (pinned)
- `OS-Copilot/ScreenSpot-v2` on the Hugging Face Hub (the release of the ScreenSpot-v2 authors,
  OS-Atlas, ICLR 2025; Apache-2.0), revision `5efbb1f1b5463a575f2eb7bc30fe29e49c15f93c`:
  `screenspot_mobile_v2.json` (501 targets), `screenspot_desktop_v2.json` (334) and
  `screenspot_web_v2.json` (437), **1,272** in all (text 718, icon 554), and the screenshots in
  `screenspotv2_image.zip` (730 images, about 1.3 GB). Each file must match its sha256
  (`fdd1595b...`, `96f494e0...`, `3e89e245...`, `750555b8...`); a download failure is a hard
  error (`infra_required`) and a checksum or size mismatch raises. A target whose screenshot is
  missing from the zip, or that has no box, is a hard error, never dropped.
- The target box is converted as ScreenSpot's reference script does
  (`njucckevin/SeeClick` `pretrain/screenspot_test.py` @ `0ef37ac4`): `[x, y, w, h]` in pixels
  becomes `[x0, y0, x1, y1]` divided by the image size.
- `HongxinLi/ScreenSpot_v2`, the copy this suite read before, is not identical: 3 of its 334
  desktop targets differ from the release (1,272 targets checked one by one).

## Protocol
- One request per target: the screenshot (the bytes of the release, as a data URL whose MIME
  type is read from the bytes: 109 of the 730 screenshots are JPEG, 78 of them in files named
  `.png`, and go out as `image/jpeg`, the rest as `image/png`), then
  `Target Element: <instruction>` and a gbench instruction to output the element's centre as
  `Point: (x, y)` with x and y in [0, 1000]. With `--eval-thinking` the instruction also asks the
  model to analyse the element step by step first.
- **Prompt**: gbench's own. ScreenSpot defines no model-agnostic prompt: its reference script asks
  with SeeClick's training prompt and reads SeeClick's own output format (a [0, 1] point, or a
  Qwen-VL `<box>`), and each published model is scored with its own grounding prompt. That is why
  the result is never leaderboard-comparable. The `screenspot` suite asks with a `mouse_click`
  tool instead.
- The last `(x, y)` / `x, y` pair in the reply is the prediction (a JSON `{"x": .., "y": ..}` also
  works); a prediction in [0, 1] is scaled to [0, 1000].
- **Scoring: ScreenSpot's click accuracy**, as the reference script scores it: correct when the
  point falls inside the target box, edges included.
- **Sampling**: the reference script decodes with the model's own generation config
  (`GenerationConfig.from_pretrained`, no temperature or token limit of its own), so no
  `max_tokens` is sent; `--max-output-tokens` is not applied in either mode
  (`max_output_tokens_note`, `official_sampling.send_max_tokens: false`). gbench always sends the
  temperature of its rule: **1.0 with `--eval-thinking`, 0.0 without**. `--temperature` or `GBENCH_SCREENSPOT_POINTING_TEMPERATURE` override it in both
  modes (the pre-rename
  `GBENCH_SEMANTIC_KEYPOINT_TEMPERATURE` is still read as a deprecated alias);
  `official_sampling` records the setting, the temperature sent and
  `temperature_matches_official`.

## Reading the result
- `accuracy`: the share of all targets whose predicted point lands inside the box, x100: the "Avg"
  column of the ScreenSpot-v2 table in the OS-Atlas paper (a micro average over the 1,272
  targets).
- `screenspot_v2_cells`: the six columns of that table, `mobile/text`, `mobile/icon`,
  `desktop/text`, `desktop/icon`, `web/text`, `web/icon` (accuracy x100); `category_accuracy` is
  keyed the same way.
- `dataset` names the pinned release.
- `leaderboard_comparable` is always `false`; `leaderboard_comparable_reason` says why (the
  gbench prompt) and also names an `--eval-limit` subset, a `GBENCH_SHARD` shard, the temperature
  sent (gbench's 1.0 with thinking or 0.0 without, or a value set with `--temperature` or the
  variable above), requests that got
  no reply (failed, timed out
  or over the context window; harness errors, left out of the score and of
  `screenspot_v2_cells`), and replies cut off by the run's token budget
  rather than at an official cap (`replies_cut_off_by_budget`; `replies_cut_off_in_reasoning`
  counts those that ran out inside the reasoning channel with no answer).
- `--eval-limit N` draws N targets stratified over the six cells.
  The run logs its planned workload at its start and records it as `eval_limit_plan`.

## Requirements
- A **vision-language model** endpoint (images are sent as data URIs).
- `huggingface_hub` and Pillow (base install); Hugging Face Hub access to download the pinned
  release (public, about 1.3 GB, cached after the first run); `HF_TOKEN` optional.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals screenspot_pointing \
       --max-output-tokens 65536 --eval-limit 20
```
`--max-output-tokens` is required by the CLI but not applied (see Sampling). Drop `--eval-limit`
for the full run. With a server that gbench launches,
`--eval-max-soft-tokens` (`70`/`140`/`280`/`560`/`1120`) is accepted for this suite as well (it
cannot be combined with `--remote-endpoint`).
