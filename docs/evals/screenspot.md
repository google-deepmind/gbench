# screenspot

**ScreenSpot-v2 GUI grounding with a mouse-click tool.** The suite runs every target of the
official ScreenSpot-v2 release (web, mobile and desktop screenshots) and asks the model to click
the described element with a `mouse_click` tool. The result always carries
`leaderboard_comparable: false` (see Prompt).

## Dataset (pinned)
- `OS-Copilot/ScreenSpot-v2` on the Hugging Face Hub (the release of the ScreenSpot-v2 authors,
  OS-Atlas, ICLR 2025; Apache-2.0), revision `5efbb1f1b5463a575f2eb7bc30fe29e49c15f93c`, read
  exactly as `screenspot_pointing` reads it: `screenspot_mobile_v2.json` (501 targets),
  `screenspot_desktop_v2.json` (334) and `screenspot_web_v2.json` (437), **1,272** in all, and the
  screenshots in `screenspotv2_image.zip` (730 images, about 1.3 GB). Each file must match its
  sha256; a download failure is a hard error (`infra_required`) and a checksum or size mismatch
  raises. A target whose screenshot is missing from the zip, or that has no box, is a hard error,
  never dropped.
- The target box is converted as ScreenSpot's reference script does
  (`njucckevin/SeeClick` `pretrain/screenspot_test.py` @ `0ef37ac4`): `[x, y, w, h]` in pixels
  becomes `[x0, y0, x1, y1]` divided by the image size, then scaled to [0, 1000] for the tool's
  coordinates.
- This suite read `HongxinLi/ScreenSpot_v2` before; that copy is not identical to the release (3
  of its 334 desktop targets differ), so numbers from before the switch are not comparable.
- **ScreenSpot-Pro (optional)**: when `GBENCH_SCREENSPOT_DATA_DIR` (legacy alias
  `SCREENSPOT_PRO_DATA_DIR`) is set, the local ScreenSpot-Pro data dir it names is scored instead:
  `annotations/*.json` (entries with `instruction`, `img_filename` and a pixel `bbox`
  `[x0, y0, x1, y1]`) and `images/`. A named dir without both subdirectories, an entry without
  those fields, or an entry whose image is missing is a hard error; there is no silent switch
  back to ScreenSpot-v2. Unset the variable to score ScreenSpot-v2.

## Protocol
- One request per target: `Click on the UI element: "<instruction>".` and `Use the computer tool
  at your disposal.`, then the screenshot (the bytes of the release, as a data URL whose MIME type
  is read from the bytes), in one user turn, with the `mouse_click` tool (`button`, `repeats`,
  and `x` / `y` in [0, 1000]) offered with `tool_choice: auto`.
- **Prompt**: gbench's own. ScreenSpot defines no model-agnostic prompt: its reference script asks
  with SeeClick's training prompt and reads SeeClick's own output format, and each published model
  is scored with its own grounding prompt. That is why the result is never
  leaderboard-comparable. `screenspot_pointing` asks for a `Point: (x, y)` in text instead.
- The click is read from the tool call (or a `mouse_click(x=.., y=..)` / JSON `"x"`/`"y"` form in
  the reply text).
- **Scoring: ScreenSpot's click accuracy**, as the reference script scores it: correct when the
  point falls inside the target box, edges included.
- **Sampling**: the reference script decodes with the model's own generation config (no
  temperature or token limit of its own), so no `max_tokens` is sent; `--max-output-tokens` is not
  applied in either mode (`max_output_tokens_note`). gbench always sends the temperature of its
  rule: **1.0 with `--eval-thinking`, 0.0 without**.
  `--temperature` or `GBENCH_SCREENSPOT_TEMPERATURE` override the temperature in both modes;
  `official_sampling` records the
  setting, the temperature sent and `temperature_matches_official`.

## Reading the result
- `accuracy`: the share of all targets whose click lands inside the box, x100: the "Avg" column
  of the ScreenSpot-v2 table in the OS-Atlas paper (a micro average over the 1,272 targets).
- `screenspot_v2_cells`: the six columns of that table, `mobile/text`, `mobile/icon`,
  `desktop/text`, `desktop/icon`, `web/text`, `web/icon` (accuracy x100); `category_accuracy` is
  keyed the same way. A ScreenSpot-Pro run has no `screenspot_v2_cells` and keys
  `category_accuracy` by the entries' `ui_type`.
- `dataset` names the pinned release (or the ScreenSpot-Pro dir).
- `leaderboard_comparable` is always `false`; `leaderboard_comparable_reason` says why (the gbench
  prompt) and also names a ScreenSpot-Pro run, an `--eval-limit` subset, the shard of a `--shard
  I/N` run, the temperature sent (gbench's 1.0 with thinking or 0.0
  without, or a value set with `--temperature` or the variable above), requests that got no
  reply (failed, timed out or over the context window; harness errors, left out of the score and
  of `screenspot_v2_cells`), and replies cut off by the run's token budget rather than at an
  official cap (`replies_cut_off_by_budget`; `replies_cut_off_in_reasoning` counts those that ran
  out inside the reasoning channel with no answer).
- `--eval-limit N` draws N targets stratified over the six cells.
  The run logs its planned workload at its start and records it as `eval_limit_plan`.

## Requirements
- A **vision-language model** endpoint that accepts tools (images are sent as data URIs).
- `huggingface_hub` and Pillow (base install); Hugging Face Hub access to download the pinned
  release (public, about 1.3 GB, cached after the first run and shared with
  `screenspot_pointing`); `HF_TOKEN` optional.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals screenspot \
       --max-output-tokens 65536 --eval-limit 20
```
`--max-output-tokens` is required by the CLI but not applied (see Sampling). Drop `--eval-limit`
for the full run. With a server that gbench launches, `--eval-max-soft-tokens`
(`70`/`140`/`280`/`560`/`1120`) sets the image soft-token budget; grounding benefits from a
higher budget (it cannot be combined with `--remote-endpoint`).

## Sharding

`--shard I/N` (or `GBENCH_SHARD=I/N`) runs shard I of N of the suite's full, ordered targets (the
release's mobile, desktop and web order; a local ScreenSpot-Pro dir's annotation order)
(round-robin: positions I, I+N, I+2N, ...), selected before the `--eval-limit` draw, which is then
taken within the shard (N shards run up to N x `--eval-limit` targets). The result records
`shard`, `shard_index` and `shard_count`, and a shard of more than one is not
leaderboard-comparable (the shard reason is named in `leaderboard_comparable_reason`); `gbench
--merge` combines the shards' results as docs/distributed-gbench.md describes: it re-counts the
union of their traces, and refuses a suite whose headline is not a correct / scored count over
them unless the suite defines its own merge.
