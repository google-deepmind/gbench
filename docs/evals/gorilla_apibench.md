# gorilla_apibench

Gorilla APIBench (Patil et al., 2023; [ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla)
`eval/`): can the model write the one-line API call that performs a task, choosing from real
machine-learning APIs? gbench runs Gorilla's own evaluation on all three API hubs (TorchHub,
HuggingFace, TensorFlow Hub) in the four settings the repo ships and the paper reports: 0-shot (the
headline) and the BM25, GPT-Index and Oracle retriever settings.

## Dataset (pinned)
- Questions: `eval/eval-data/questions/<hub>/questions_<hub>_<setting>.jsonl` for the settings
  `0_shot`, `bm25`, `gpt_index` and `oracle`, from `ShishirPatil/gorilla` at commit
  `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` (186 TorchHub, 904 HuggingFace, 685 TensorFlow Hub
  questions per setting, the same question ids in every setting: 1775 per setting, 7100 in total;
  12 files, about 8 MB). The retriever files are the 0-shot questions with the document Gorilla's
  BM25 or GPT-Index retriever returned, or the reference document (oracle), already appended to
  the question text ("Use this API documentation for reference: ..."). gbench runs these shipped,
  pre-retrieved files through `get_llm_responses.py`'s `encode_question`: they are the fixed
  inputs behind the repo's own responses for the three settings
  (`eval/eval-data/responses/<hub>/response_<hub>_Gorilla_{FT,RT}_{bm25,gpt_index,oracle}.jsonl`),
  so every run sees the same documents. The eval README's "Getting Responses with Retrievers"
  section documents another path, which gbench does not run: `get_llm_responses_retriever.py`
  retrieves on the fly from the 0-shot questions (a BM25 index over `data/api/<hub>_api.jsonl`, or
  a GPT index built with OpenAI embeddings) and appends the result under another suffix
  ("Here are some reference docs:" then "API 0: ..."), so its documents and prompt differ from the
  shipped settings. Downloaded once to `~/.cache/gbench/gorilla_apibench/<commit>/questions/`
  (`GBENCH_CACHE_DIR` overrides the root) and checked against each file's git blob id at that
  commit (`HUBS` for 0-shot, `RETRIEVER_QUESTION_BLOBS` for the others; a cached file that does not
  match is downloaded again).
- Reference answers and API databases: `gorilla-llm/APIBench` at revision
  `ac21e1892e634dfa25f8ad75f16cbdbfb0a5736d` (`<hub>_eval.json` and `<hub>_api.jsonl`,
  byte-identical to the repo's `data/apibench/` and `data/api/`).

## Protocol
- **Prompt**: `get_llm_responses.encode_question`, verbatim, in every setting: a system message
  ("You are a helpful API writer who can write APIs based on requirements.") and a user message with
  the question text of the setting's file (for the retriever settings, with its API document), the
  requested `<<<domain>>> ... <<<api_call>>> ... <<<api_provider>>> ...` format and the hub's full
  list of domains. Only the oracle setting's document is the reference API entry, as upstream.
- **Scoring**: the hub's `eval-scripts/ast_eval_{th,hf,tf}.py`, ported line for line:
  1. cut the text between `api_call` and `api_provider` out of the reply (TorchHub scores a reply
     with no `api_call` field as neither correct nor hallucinated; the other hubs parse the whole
     reply);
  2. parse it with tree-sitter (Python grammar) and walk its sub-trees;
  3. find the first entry of the hub's API database whose callee appears in the reply and whose
     arguments all occur in that sub-tree (the three scripts differ in which arguments count:
     HuggingFace all of them, TensorFlow Hub all with `model=` values unwrapped, TorchHub only
     `repo_or_dir` / `model` values);
  4. **correct** iff that entry's domain equals the question's domain; **hallucination** if no
     entry matches; otherwise an **error** (a real API of the wrong domain).

  Inline reasoning tags are removed from the reply first; the per-sample verdict and the
  per-hub tallies score the same text. The scorer is the same in every setting.
- The port was checked against the upstream scripts (run with Gorilla's own compiled grammar) on
  1775 x 6 synthetic replies per hub: identical accuracy and hallucination rates, both with that
  grammar and with the `tree-sitter-python` 0.21.0 wheel gbench uses.

## Headline and how to read the result
- `per_setting.<setting>.per_hub.<hub>`: `accuracy` (Gorilla's "Final Functionality accuracy"),
  `hallucination`, `error` and `n` for each of `0_shot`, `bm25`, `gpt_index` and `oracle`, plus
  the setting's `mean_accuracy` and `mean_hallucination` over the hubs. These are the numbers
  Gorilla publishes (per hub and setting). `per_hub` repeats the 0-shot ones.
- `accuracy` (headline): the 0-shot setting's unweighted mean of the three hub accuracies;
  `hallucination` is the same mean of the 0-shot hub hallucination rates. `headline_n` is the
  number of 0-shot questions it rests on. When a small `--eval-limit` draw has 0-shot questions
  from fewer than the three hubs, the `metric` names the hubs used with their n (for example
  "mean over 1 of 3 hubs (torchhub, n=1)") and `leaderboard_comparable_reason` says so; when it has
  no 0-shot question at all (`--eval-limit 1` or `2`), `accuracy` is null with
  `accuracy_withheld_reason`, never the retriever settings' pooled number under the 0-shot name.
- `category_accuracy` is keyed `<hub>/<setting>` (for example `torchhub/0_shot`).
- `leaderboard_comparable` is always `false`: APIBench publishes each hub separately, so the
  cross-hub mean is a gbench aggregate. Compare `per_setting` / `per_hub` with published numbers.
  The reason also
  lists `--eval-limit`, a shard, a temperature other than 0 (upstream samples at 0), decoding
  penalties or `--attempt-count > 1` when present.
- `official_sampling`: Gorilla's settings (`get_llm_responses.py` requests `n=1, temperature=0`),
  the temperature this run used, where it came from (`temperature_source`) and whether it matched
  (`temperature_matches_official`).
- Failed requests are reported in `failed_requests` and left out of the denominators.

## Requirements
- A running OpenAI-compatible `/v1` endpoint.
- Network on the first run (HF Hub for APIBench, raw.githubusercontent.com for the question files).
  A file that cannot be downloaded, or a question file that does not match its pinned blob id,
  stops the suite with an `[INFRA REQUIRED]` error.
- Python packages for the AST scorer: `tree-sitter==0.21.3` and `tree-sitter-python==0.21.0`.
  Both are pinned in gbench's `evals` extra (`pip install -e '.[evals]'`); to add them to an
  existing environment:
  ```bash
  pip install 'tree-sitter==0.21.3' 'tree-sitter-python==0.21.0'
  ```
  The suite hard-errors (`infra_required`) before generating anything if either is missing, if
  `tree-sitter` is not a 0.21.x release (0.22 changed the `Language` / `Parser` API the scorer
  uses) or if `tree-sitter-python` is not 0.21.0 (the grammar decides the parse trees).
- No API key and no LLM judge.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals gorilla_apibench \
       --max-output-tokens 4096
```
Every run covers all four settings (7100 requests). Add `--eval-limit N` for a quick check (a total
budget, stratified across the 12 hub/setting pairs; flagged in the result).

The run logs its planned workload in one line at its start (for example `[gorilla_apibench]
--eval-limit 5: 5 sample(s) x 1 attempt = 5 generations`) and records it in the result as
`eval_limit_plan` (`limit`, `samples`, `per_setting`, `attempts_per_sample`, `generations`).

Sampling: temperature 0.0 without `--thinking`, 1.0 with it (gbench's rule for every suite);
Gorilla's `get_llm_responses.py` sends 0 (`gorilla/eval/get_llm_responses.py` at the pinned
commit), recorded under `official_sampling`. `--temperature`, or
`GBENCH_GORILLA_APIBENCH_TEMPERATURE` for this suite alone (it takes precedence), overrides it and
is recorded under `official_sampling`. No `max_tokens` is sent, with or without `--thinking`, as
`get_llm_responses.py` sends none: the server grants the room left in the context window,
`--max-output-tokens` is not applied (`max_output_tokens_note` says so), and a reply cut off at
the end of the context window is counted in `truncated_responses`, as it would be cut off
upstream.
