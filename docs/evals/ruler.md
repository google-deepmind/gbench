# ruler

Canonical RULER (Hsieh et al., NVIDIA, COLM 2024; arXiv:2404.06654;
[NVIDIA/RULER](https://github.com/NVIDIA/RULER), Apache-2.0). A tokenizer-defined
long-context benchmark: **13 tasks x 6 context bands** (4k/8k/16k/32k/64k/128k), 500
samples each, with haystacks padded to exact token counts under the **target model's own
tokenizer**. It measures how well a model retrieves, tracks and aggregates information as
the context grows (needle-in-a-haystack variants, variable tracking, common/frequent word
extraction, and QA over padded documents).

## Pinned source

| What | Pin |
| --- | --- |
| NVIDIA/RULER checkout (data generators + task configs) | commit `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a` (main, 2026-07-22) |

RULER is a data-*generation* benchmark, not a fixed dataset, so the checkout's revision
decides what is measured. The suite runs `git rev-parse HEAD` and `git status` in the
checkout before generating anything and **hard-errors** unless it is exactly that commit
with no changes to tracked files. The revision is recorded on the result as
`ruler_revision`.

## How it runs (per-tokenizer data generation, cached)

This suite drives NVIDIA's own `scripts/data/prepare.py` from your checkout to generate the
per-tokenizer data on the first run, **caches it to
`$GBENCH_CACHE_DIR/ruler/<tokenizer>/<revision[:12]>-<template-hash>/<band>/`** (default
`~/.cache/gbench/ruler/...` when `GBENCH_CACHE_DIR` is unset), and reuses it on every later run. Pay the generation cost once (the 64k/128k bands are slow); it is reused
until you change the tokenizer, the chat template, the bands, the sample count or the RULER
revision. (Data cached by earlier gbench versions under
`$GBENCH_CACHE_DIR/ruler/<tokenizer>/<band>/` was generated without the chat-template budget
below and is not reused.)

- Tokenizer = your `--tokenizer` value, else `GBENCH_RULER_TOKENIZER`, else the served model
  name. It must be an HF tokenizer that carries the model's chat template.
- Bands via `GBENCH_RULER_LENGTHS` (default `4096 8192 16384 32768 65536 131072`);
  samples/cell via `GBENCH_RULER_NUM_SAMPLES` (default 500).

### Band budget (what "128k" means)

RULER defines the band as `max_seq_length`: "max sequence length including all input tokens
and generated tokens". The chat template, the prompt, the answer prefix and the task's
generation cap all fit inside it. The server applies the model's chat template to every
request, so gbench renders that template around RULER's `{task_template}` slot (with the
run's `--thinking` setting), registers it as a RULER model template and runs `prepare.py`
with `--prepare_for_ns`. In that mode `prepare.py` renders the data with RULER's `base`
template and passes the chat template's token count to the generators as
`--model_template_token`, which they subtract from the band. The count is recorded on the
result as `ruler_template_tokens`. The generated data therefore fits a server whose
`max_model_len` equals the band (131072 for 128k, as on Gemma 4 E2B/E4B), and the suite only
requires `max_model_len >= band`.

Pre-send context check (harness, not RULER): gbench's shared check (`base.clamp_to_context`)
requires `min(256, cap)` tokens of answer room, where `cap` is the task's generation cap. On a
vLLM endpoint it counts each prompt with the server's `/tokenize`, forwarding the same tools and
`chat_template_kwargs` (for example `enable_thinking`) as the chat request, so a text-only
prompt is counted exactly and gets no slack. On a window exactly equal to the band, the tightest
128k task (`vt`) leaves only about 20 tokens of room beyond its cap (measured with the Gemma 4
E4B tokenizer; the other 128k tasks leave more), so those prompts fit only with this exact,
slack-free count, and they are sent.

Because RULER data is built to fit, a prompt this check would refuse (`OVER_CONTEXT`) is a
serving or counting mismatch, never a model answer, so the suite **hard-errors**
(`infra_required`) instead of scoring it 0:

- before anything is sent, the longest prompt of every `task@band` cell (by RULER's own token
  count of the row, which differs from the served count by the same template offset for every
  row) goes through the same check with the budget and `chat_template_kwargs` its request will
  carry;
- after the run, any `OVER_CONTEXT` trace (including a prompt the server itself rejected as too
  long) raises, and the run is not scored.

With a window exactly equal to the largest band, two setups hit this error: an endpoint without
`/tokenize` (the local count adds template allowances and a 64-token slack, which the tightest
128k cells, such as `vt`, cannot absorb), and `--thinking` (the check needs 256 tokens of room for the reasoning, and RULER
leaves only the cap). Run the 128k band against vLLM, serve a window above the band where the
model supports it (for example the 26B/31B models at 262144), or drop the band via
`GBENCH_RULER_LENGTHS`.

### Prompt

Each request is a single user turn holding `input + answer_prefix`, which is what RULER
generates from (`scripts/pred/call_api.py`) and how its chat-API clients send it
(`client_wrappers.OpenAIClient` with the `base` template). The answer prefix is the task's
lead-in, for example `The special magic number for <key> mentioned in the provided text is`
(niah) or `Answer: The top 10 words that appear most often in the list are:` (cwe).

RULER has a second path for open models served without a chat API: `prepare.py` wraps the task
in a chat template from `scripts/data/template.py` (for example `meta-llama3` or `Phi3`) and the
answer prefix follows the assistant header, so the model continues it as a prefill. gbench talks
to an OpenAI-compatible chat endpoint and its requests end with the user turn and a generation
prompt (no `continue_final_message` assistant prefill), so it follows RULER's chat-API path (the
prefix at the end of the user turn) for every model, Gemma included.

## Scoring

Canonical string match (from RULER `scripts/eval/synthetic/constants.py`), after
`postprocess_pred`: **string_match_all** (niah / vt / cwe / fwe, fraction of references
present) and **string_match_part** (qa_1 / qa_2, any reference present). Greedy, with RULER's
per-task generation caps (niah 128 / vt 30 / cwe 120 / fwe 50 / qa 32), each request's own
`max_tokens`. RULER defines no reasoning-model settings, so with `--thinking` the token-limit
policy does not apply those caps, which were written for non-reasoning generation (they would cut
the reasoning before any answer): each request is sent the run's `--max-output-tokens` (else the
thinking floor), and the result records the lifted caps in
`official_cap_not_applied_to_reasoning` and `max_output_tokens_note`.

Headline `accuracy` = **RULER Avg** = mean over bands of (mean of the 13 task scores).
Also reported: `ruler_length_scores`, `ruler_task_scores`, `ruler_cell_scores`,
`effective_length` (largest band scoring >= 85.6), `wavg_inc`/`wavg_dec`, `length_bands`,
`ruler_revision`, `ruler_template_tokens`.

### Reading the result

`leaderboard_comparable` is `true` only for the complete canonical matrix: all 13 tasks x the
exact 6 bands (4k-128k) at 500 samples each, temperature 0.0 (the resolved value on the result's
`temperature`, which a `--temperature` or `GBENCH_RULER_TEMPERATURE` override changes), no
`--eval-limit`, and every request got a reply (a request that failed or timed out is a harness
error: left out of its cell and of every score, counted and named in `status_reason`; with no
cell scored, `accuracy` is null with `accuracy_withheld_reason`), with no reply cut off by the
run's token budget rather than at an official cap
(`replies_cut_off_by_budget`; `replies_cut_off_in_reasoning` counts those that ran out inside
the reasoning channel with no answer). Otherwise it is `false` and
`leaderboard_comparable_reason` lists each difference, for example `temperature 1.0 (RULER
decodes greedily at 0.0)`. A prompt that could not be sent never reaches the result: the suite
hard-errors instead (see the pre-send context check above).

## Requirements

- **A serving endpoint whose context window covers the largest band** (`max_model_len >=
  131072` for the 128k band). The suite reads the server's `max_model_len` (`GET /v1/models`)
  and **hard-errors** (does not silently skip) if it is smaller: serve with e.g. vLLM
  `--max-model-len 131072`, or restrict `GBENCH_RULER_LENGTHS`. With a window exactly equal to
  the band, the endpoint must expose vLLM's `/tokenize` (see the pre-send context check).
- **The NVIDIA/RULER checkout at the pinned commit**, at `GBENCH_RULER_DIR` (or
  `$GBENCH_PREREQS_DIR/RULER`). Because RULER LFS-tracks its `*.json` data, cloning it needs
  **git-lfs**, and its corpora are fetched by its own scripts:
  ```bash
  # git-lfs - apt (system) OR pip:
  sudo apt-get install -y git-lfs        # Debian/Ubuntu
  #   or:  pip install git-lfs
  git lfs install

  git clone https://github.com/NVIDIA/RULER
  cd RULER
  git checkout c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a
  git lfs pull                           # fetch english_words.json (and other *.json blobs)
  # corpora for niah + qa tasks (written as untracked files, which the revision check allows):
  python scripts/data/synthetic/json/download_paulgraham_essay.py
  bash   scripts/data/synthetic/json/download_qa_dataset.sh
  export GBENCH_RULER_DIR=$PWD
  ```
- `git` on `PATH` (the revision check runs it).
- pip: `wonderwords`, `nltk`, `html2text`, `tenacity`. These are **RULER's own data-generation
  deps** (from the RULER checkout's `requirements.txt`), NOT part of gbench's `evals` extra; install them
  into the same environment gbench runs in:
  ```bash
  pip install wonderwords nltk html2text tenacity
  ```
  plus `transformers` (already present in a gbench serving environment) to load the tokenizer
  and render its chat template.
- No LLM judge, no API key.
- **Sampling:** RULER's settings are temperature 0.0 and top_p 1.0 (`scripts/config_models.sh`),
  and seed 0 (the `--random_seed` default of `scripts/pred/call_api.py`), the fields its
  chat-API client sends. gbench sends top_p and seed in both modes and the temperature of its
  rule: **1.0 with `--thinking`, 0.0 without**.
  `GBENCH_RULER_TEMPERATURE` overrides the temperature for this suite, in both modes; else
  `--temperature`. The result records
  `official_sampling` (the reference, the temperature used, its source and
  `temperature_matches_official`).

## Run
```bash
export GBENCH_RULER_DIR=/path/to/RULER
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals ruler \
       --max-output-tokens 65536
```
Without `--thinking` each request carries its task's RULER cap, so `--max-output-tokens`
(required by the CLI) only sets the budget of a `--thinking` run (the token-limit policy; the caps
are recorded as not applied to reasoning).

`--eval-limit N` runs N samples in total, dealt over the `(task@band)` cells (one per cell
before any cell gets two, the cells visited in a seeded order); a cell's share is a seeded
random draw of its generated rows. Only the cells dealt a sample are generated (each at the
full `GBENCH_RULER_NUM_SAMPLES`, so the cache is the full run's), so `--eval-limit 5` generates
and runs 5 cells, not all of them. The run logs the plan at its start, records it as
`eval_limit_plan`, and is `leaderboard_comparable=false`.
