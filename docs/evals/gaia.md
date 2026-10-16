# gaia

GAIA (General AI Assistants, [arXiv:2311.12983](https://arxiv.org/abs/2311.12983)): questions
that a person answers in a few minutes with a browser, files and a calculator, each with one
short, verifiable answer. It measures web research, document and media reading, and multi-step
reasoning. gbench gives the model a web search tool and the task's attached file, and scores the
final answer with the official scorer.

## Settings
The [GAIA leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard) publishes two
tables (`gaia-benchmark/results_public`, config 2023: `validation` and `test`). gbench runs both:

| Setting | `GBENCH_GAIA_SPLIT` | Tasks | Scored by |
|---|---|---|---|
| validation (default) | `validation` or unset | 165 (53 / 86 / 26 per level) | gbench, locally, with the official `question_scorer` |
| test | `test` | 301 (93 / 159 / 49 per level) | the leaderboard only: gbench writes the submission file |

The leaderboard ranks the test split; its answers are hidden, and the leaderboard space no
longer accepts validation submissions ("we closed the validation leaderboard, as it was no
longer informative"). The validation split's answers are public, so gbench scores it locally.

## Dataset
- `gaia-benchmark/GAIA` (gated), config `2023_all`, pinned to revision
  `682dd723ee1e1697e00360edccf2366dc8418dd9`. gbench downloads `2023/<split>/*` (the metadata
  and the attached files) with `snapshot_download`: about 10 MB for validation and 100 MB for
  test.
- Validation: 38 of the 165 tasks ship a file: 13 xlsx, 10 images (png/jpg), 3 mp3, 3 pdf, 2
  zip (holding an xls, an xml, a pdf and an xlsx), and one each of txt, csv, py, pdb, jsonld,
  docx, pptx.
- Test: 71 of the 301 tasks ship a file: 16 xlsx, 12 txt, 12 pdf, 15 images (png/jpg), 5 csv,
  4 mp3, 2 xml, and one each of json, docx, pptx, m4a and mov. Every `Final answer` is `?`.
- `--eval-limit N` draws a subset of N tasks stratified by `Level`.

## Protocol
- **Prompt.** The official GAIA system prompt (verbatim from the paper and leaderboard), then
  the question as the user message. The paper's protocol is to "zero-shot prompt an AI
  assistant with the questions and attached evidence if there are some", so a task with a file
  carries it in the same user message, after the question:
  - images: the file's bytes as an `image_url` data URI (`image/png`, `image/jpeg`);
  - audio: an `input_audio` part (`format: mp3`); the test split's one `.m4a` file is decoded
    to wav with `ffmpeg` first (an `input_audio` part takes mp3 or wav) and sent with
    `format: wav`;
  - video: the test split's one `.MOV` file (30.7 MB), whole, as a `video_url` data URI
    (`video/quicktime`); the server samples its frames. Prompt sizing counts a video part as
    `GBENCH_VIDEO_SOFT_TOKENS` tokens (default 2496, gemma-4's maximum per video in vLLM), not by
    the length of its base64, so a known context window does not refuse the task because of the
    payload;
  - txt, csv, py, json, jsonld, xml, pdb: the file's own text;
  - xlsx: every sheet as comma-separated rows plus a `Cell fill colors:` list (several tasks
    encode their data in cell colors); xls: every sheet via pandas/xlrd;
  - pdf: the text of every page (pymupdf); docx: paragraphs and tables in document order;
    pptx: the text of each slide in slide order (both read from the file's XML);
  - zip: one entry per member file, each rendered by the rules above.
  Each file is introduced by a text part `Attached file: <file name>`. Rendered text is capped
  at 100,000 characters per file, the limit the open GAIA reference agent uses for its text
  inspector; only the 2.9 MB pdb file is longer, and it is cut with a
  `[... '<name>' truncated at 100000 characters]` marker. A file type with no renderer, a file
  missing from the snapshot, or a missing renderer package is a hard error, never a task sent
  without its file.
- **Tools.** The request declares one tool, `web_search(query)`, backed by gbench's Gemini
  Google-Search grounding cascade. There is no browser, file download or code execution tool.
  **The tool's results are written by Gemini, not taken from web pages.** Each call is one
  Gemini `generateContent` request with the `google_search` tool and the model's query as the
  prompt (`search_tool.gemini_search`). Each result carries the title and URL of a source the
  grounding cited and, as its snippet, the part of the Gemini model's own answer that cites that
  source (`groundingSupports[].segment.text`; the start of the answer when no segment cites it);
  when the grounding cites no source the tool returns the start of the answer as one result. No
  page text is fetched. A model can therefore hand the whole task to the tool and read back a
  Gemini model's grounded answer, so a GAIA number here partly measures the Gemini model's
  research. GAIA defines no tool, so the description the model sees is gbench's, and it says
  this (`GROUNDED_SEARCH_DESCRIPTION`, recorded as `search_tool_description`): "Search the web
  through a Gemini model grounded in Google Search. Each result gives the title and URL of a
  source the Gemini model cited, with the part of that model's written answer that cites the
  source, or the start of its answer when it cited none (text written by the Gemini model, not
  quoted from the page). ..." The result names the backend in `search_backend_kind`.
- **Tool transcripts.** Every trace keeps its `web_search` calls in `tool_transcript` (the
  arguments, the Gemini backend that answered, `gemini:<model>`, whether the search backend
  failed, and the reply the model read) and their number in `web_search_calls`, in both
  settings, so an answer can be traced to the tool reply it came from. A call whose search tool
  raised is kept too, as a failed search (backend `tool-raised`, with the exception's type), so
  its sample is a `search_backend_failure` harness error. Each run keeps its calls in a store of
  its own, so two suites in one process (gaia and deepsearch_qa share the executor) never read
  each other's calls.
- **Tool loop.** GAIA prescribes no agent loop. gbench takes its limits from the open GAIA
  reference agent (huggingface/smolagents `examples/open_deep_research/run_gaia.py` at
  `227ef5e`, whose web-search agent runs `max_steps=20` and is made to give a final answer at
  the limit):
  - up to **20 tool-enabled model turns** (the tool calls of each are executed), then one
    final turn with the tools withdrawn;
  - every turn sends the reference agent's `max_tokens` 4096 without `--thinking`, and with it
    the run's `--max-output-tokens` (see [Sampling](#sampling); clamped only to fit the context
    window); the global tool-loop turn ceiling (`GBENCH_TOOL_LOOP_MAX_OUTPUT_TOKENS`) does not
    apply to this suite;
  - the assistant turns sent back to the model carry the model's own content and tool calls.
  gbench adds the following user turns, verbatim, only when needed:
  - after the 20th tool-enabled turn: "No further tool calls are possible. Answer now, using only
    what you already have, and state your final answer explicitly." (tools withdrawn);
  - when a turn ends inside the reasoning channel with no content and no tool call: "You ran
    out of room to think. Do not reason further. State your best answer now from what you
    already have, in the requested format." (tools withdrawn, thinking off for that turn);
  - after either of those, if the model answered with a tool call that the server discarded:
    "That response could not be read because it was a tool call. Tools are disabled and no
    tool call can succeed. Reply with plain text only - no function call, no tool syntax -
    stating your final answer." (once; set `GBENCH_RETRY_DISCARDED_FINAL=0` to disable).
  The result records these settings in `tool_loop_protocol` and how each sample's loop ended
  (`tool_rounds.loop_ends`, per sample `tool_loop_end`).
- **Scoring.** The text after the last `FINAL ANSWER:` marker is scored with the official
  leaderboard `question_scorer` (number, comma/semicolon list, or normalized string exact
  match). No judge.
- **Attempts.** One answer per task, as a leaderboard submission holds; the test setting
  ignores `--attempt-count`.

## The test setting: the leaderboard submission
With `GBENCH_GAIA_SPLIT=test` gbench runs the 301 test tasks with exactly the protocol above
and writes, under `<results dir>/gaia_submission/` (a new temporary directory when the run has
no results directory):
- `gaia_2023_test_<model>_<think|nothink>_<UTC time>.jsonl`: one JSON line per task, in dataset
  order, with the fields the leaderboard's submission form reads: `task_id`, `model_answer` (the
  text after the last `FINAL ANSWER:` marker, which the leaderboard grades with the same
  `question_scorer`; empty when the model gave no marker) and `reasoning_trace` (each
  `web_search` call with its result, the model's reasoning when the server returns it, and its
  final response);
- `..._result.json`: the run's result with every sample trace (each with its `tool_transcript`).

The run then ends with `infra_required`: the answers are hidden, so only the leaderboard can
score the file. Upload it at <https://huggingface.co/spaces/gaia-benchmark/leaderboard>
(space revision `9f133d71362e77b3539f1514f31b9c101a545fec`). The leaderboard accepts only a
complete file (93 / 159 / 49 tasks per level, no duplicates); the message says so when the run
was an `--eval-limit` subset, names failed requests (their `model_answer` is empty) so they
can be re-run first, and lists the tasks whose prompt does not fit the served context window
(`leaderboard_submission.over_context_tasks`; they need a server with a larger `--max-model-len`,
not just a re-run), and the tasks that made a `web_search` call the Gemini search backend could
not answer (`leaderboard_submission.search_backend_failure_tasks`; re-run them). The leaderboard
reports the average score and the three level scores.

## Headline metric and how to read it
For the validation setting, `accuracy` is the share of tasks whose final answer the official
scorer accepts (the leaderboard's average score) and `category_accuracy` holds the three level
scores. Alongside it:
`accuracy_file_free` / `file_free_n`, `accuracy_with_file` / `file_dependent_n`,
`accuracy_by_attachment_type`, the per-`Level` `category_accuracy`, `search_backend`,
`search_backend_kind`, `search_tool_description` and `search_backend_calls` (which grounding model
served each lookup; a run that made no real searches shows up here).

`leaderboard_comparable` is always **false**. `leaderboard_comparable_reason`: the validation
split's answers are public and it is scored locally, while the leaderboard ranks the test split;
the model's only web access is the Gemini-written `web_search` above (the cited sources' titles
and URLs with segments of a Gemini model's grounded answer, no page text), so the score partly
reflects the Gemini model's research, rather than a browsing agent with page fetch and code
execution; attachments are inlined. It also names the temperature gbench sent (the reference
agent sends none), the final
replies cut off by the run's token budget (see [Sampling](#sampling)) and any harness errors.

### Harness errors
A sample that ended in a harness error is never scored as the model's outcome: it is left out of
`accuracy` (numerator and denominator), `category_accuracy` and the per-attachment figures,
marked on its trace (`harness_error` with its kind and detail, `scoring_excluded`), counted in
`harness_errors` (`{kind: samples}`) and named in `leaderboard_comparable_reason`; the run's
`status` is then `completed_with_errors`, `effective_n` the samples scored and `scoring_excluded`
the samples left out. When no sample is left the score is withheld (`accuracy: null`,
`status: failed`, `accuracy_withheld_reason`). The kinds:
- `endpoint_failure`, `endpoint_timeout`, `over_context`: a model request produced no reply (an
  endpoint error, a timeout, or a prompt that does not fit the served context window);
- `search_backend_failure`: a `web_search` call no Gemini grounding model answered
  (`all-grounding-models-failed`, `deadline-passed`, or no key), so the model worked without a
  lookup it asked for.

The file tasks also test reading images, audio, video and documents. An endpoint that cannot
take image, audio or video input rejects those requests; they are harness errors
(`endpoint_failure`: the endpoint lacks a prerequisite of the task), so such a run's accuracy is
over the tasks the endpoint could take, and the reason names how many it could not.

### Audio attachments
- The 3 mp3 tasks (22.5 s, 25.7 s and 35.3 s of audio) send the whole file as one
  `input_audio` part. Prompt sizing counts each audio part as `GBENCH_AUDIO_SOFT_TOKENS` tokens
  (default 750), not by the length of its base64, so a known context window never refuses or
  clamps these tasks because of the audio payload. Traces keep only the first 48 characters of
  the audio base64.
- Remaining limitations, both on the serving side: gemma-4's audio processor caps a clip at 750
  soft tokens (30 s at 40 ms per token), so the 35.3 s clip exceeds what the model represents;
  and gemma-4 26B-A4B and 31B have no audio encoder (no `audio_config`), so a server for them
  cannot take the 3 audio tasks, and those requests count as failed requests.
- The budget is a fixed per-clip value; set `GBENCH_AUDIO_SOFT_TOKENS` for a model whose audio
  encoder emits more tokens per clip.

## Sharding

`--shard I/N` (or `GBENCH_SHARD=I/N`) runs shard I of N of the split's full, ordered rows (the
pinned metadata.parquet's order) (round-robin: positions I, I+N, I+2N, ...), selected before the
`--eval-limit` draw, which is then taken within the shard, stratified by Level (N shards run up to
N x `--eval-limit` tasks). The result records `shard`, `shard_index` and `shard_count`, and a
shard of more than one is not leaderboard-comparable (the shard reason is named in
`leaderboard_comparable_reason`); `gbench --merge` combines the validation shards' results as
docs/distributed-gbench.md describes, re-counting the union of their traces.

## Requirements
- A running OpenAI-compatible `/v1` chat endpoint that accepts tool calls (vLLM:
  `--enable-auto-tool-choice --tool-call-parser <parser>`). To answer the image and audio
  tasks the served model must accept `image_url` and `input_audio` parts.
- **Gated dataset:** accept the terms at <https://huggingface.co/datasets/gaia-benchmark/GAIA>
  and export `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`). Without access the suite hard-errors
  (`infra_required`).
- **`GEMINI_API_KEY`** (or the key list `GEMINI_API_KEYS`, which wins when both are set): powers
  the `web_search` tool. Required; checked before the download, and the suite hard-errors
  (`infra_required`) without it. The key is then checked live (`base.gemini_key_live_valid`, as
  `deepsearch_qa` does): a key the Gemini API rejects, or a model of the Gemini cascade the API
  does not serve to it (the error names the model and the variable it came from), hard-errors
  before the download instead of after a run in which every search failed; an inconclusive check
  (network error, 429, 5xx) lets the run start.
- Python packages: `huggingface_hub`, `pandas`/`pyarrow` (base install), plus the attachment
  renderers `openpyxl` (xlsx), `pymupdf` (pdf) and `xlrd` (the legacy `.xls` inside one zip
  attachment): `pip install openpyxl pymupdf xlrd`. A missing renderer package hard-errors
  (`infra_required`) when a selected task needs it; docx and pptx are read with the standard
  library.
- Test setting only: `ffmpeg` on `PATH` (for the `.m4a` attachment; `apt-get install ffmpeg`),
  and a served model that accepts `video_url` parts for the `.MOV` task. A missing `ffmpeg`
  hard-errors (`infra_required`) when that task is selected.
- Optional: `GBENCH_GAIA_SPLIT` (`validation`, the default, or `test`; any other value is an
  error); `GBENCH_GAIA_TEMPERATURE` sets the temperature for this suite (default 1.0 with
  `--thinking`, 0.0 without); the web-search cascade knobs (`GBENCH_SEARCH_MODELS`, `GBENCH_SEARCH_MAX_RPS`, ...)
  are in [env-knobs](env-knobs.md).

## Sampling
GAIA prescribes no sampling, and the open reference agent the loop settings come from
(`run_gaia.py` at `227ef5e`, `answer_single_question`) builds its model with no temperature.
gbench sends no other sampling field, but always sends a temperature, its rule for every suite:
1.0 with `--thinking`, 0.0 without. `--temperature` or `GBENCH_GAIA_TEMPERATURE` (which takes
precedence) sets another. The result records it under `official_sampling`, and a validation run
names it in `leaderboard_comparable_reason`.

`max_tokens` follows the same reference agent through the token-limit policy
(`resolve_token_budget` in `gbench/runners/eval_suites/base.py`). Its `model_params` give
`model_id == "o1"` `reasoning_effort="high"` and `max_completion_tokens=8192`, and every other
model `max_tokens=4096`. The `o1` branch keys on OpenAI's model name, so it is not a
reasoning-model setting for a served model: a served model takes the `max_tokens=4096` path in
both modes (the script's commented-out open-model line also gives `Qwen/Qwen3-32B`
`max_tokens=4096`), and no `reasoning_effort` is sent.
- without `--thinking` every turn sends `max_tokens` 4096 exactly (`--max-output-tokens` is
  recorded as not applied in `max_output_tokens_note`); a reply cut there is the reference
  agent's outcome, counted in `replies_at_official_max_tokens`;
- with `--thinking` the 4096 is not applied, because it was written for non-reasoning
  generation: every turn is sent the run's `--max-output-tokens` (a thinking run is never capped
  below it), and the result records the cap as not applied to reasoning
  (`official_sampling.max_tokens_applied: false`, `max_tokens_sent`).

A final reply that stopped at `max_tokens` on a request that did not carry the 4096 itself (with
`--thinking` every such reply; without it, those of a sample whose first request the context
window lowered below 4096) is counted in `replies_cut_off_by_budget`
(`replies_cut_off_in_reasoning` for those with no answer text) and named in
`leaderboard_comparable_reason`; a test-setting run records the count in
`leaderboard_submission.replies_cut_off_by_budget` and names it in the `infra_required`
message. The count reads each sample's final reply, so a turn that ran out inside the reasoning
channel and was followed by the "You ran out of room to think" turn (see
[Protocol](#protocol)) is counted only when that turn is cut off too.

## Run
```bash
export HF_TOKEN=hf_xxx
export GEMINI_API_KEY=...
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals gaia --max-output-tokens 16384
```
`--max-output-tokens` is required for every eval run; it is applied only to a `--thinking` run
(see [Sampling](#sampling)). Add `--eval-limit 20` for a quick check. The test setting (writes
the submission file, then stops with `infra_required` naming it):
```bash
GBENCH_GAIA_SPLIT=test gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals gaia --max-output-tokens 16384
```
