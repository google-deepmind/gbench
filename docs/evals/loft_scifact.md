# loft_scifact

LOFT (Long Context Frontiers, arXiv:2406.13121,
[google-deepmind/loft](https://github.com/google-deepmind/loft)) in-context retrieval:
a whole document corpus is placed in the prompt and the model must name the passage that
answers a query. It measures whether a long-context model can replace a retrieval system.

**Which LOFT task:** LOFT's **SciFact** retrieval task (scientific-paper title + abstract
passages; given a scientific claim, find the passage that can verify it) at the **128k** context
length on the **test** split (100 queries), i.e. LOFT's
`retrieval_scifact_128k_test:few_shot_with_cot`. The result carries
`benchmark: "LOFT retrieval_scifact_128k_test:few_shot_with_cot"`.

> **Renamed.** This suite was previously registered as `loft_x_arxiv`. LOFT ships no "arXiv"
> task, so the old name did not say what runs. `loft_x_arxiv` still works as a deprecated alias
> (one warning per run), and `GBENCH_LOFT_X_ARXIV_TEMPERATURE` is still read as a deprecated
> alias of `GBENCH_LOFT_SCIFACT_TEMPERATURE`. `--eval-limit` draws the same queries as before the
> rename.

## Pinned sources

| What | Pin |
| --- | --- |
| Data: LOFT's public bundle `https://storage.googleapis.com/loft-bench/retrieval/scifact.zip` (the file LOFT's `download.sh` fetches) | sha256 `f7ef0719c1edb7adc343f3b5411c17868602421db1683fddb5e85aa4dda84a91` (GCS object last modified 2024-08-28) |
| Prompt, extraction and metric ported from | `google-deepmind/loft` commit `219f68e470260c22fc53cc8cc2416b78313db774` |

The bundle is downloaded once to `$GBENCH_CACHE_DIR/loft/` (default `~/.cache/gbench/loft/` when
`GBENCH_CACHE_DIR` is unset) and its sha256 is checked on every run.
A download failure or a different hash is a hard error (`infra_required`), never a silent
substitute. The 128k split holds 357 passages, 5 few-shot queries and 100 test queries, each
test query with exactly one relevant passage (checked at load time).

## Protocol

The prompt is built exactly as LOFT's prompt registry builds
`retrieval_scifact_128k_test:few_shot_with_cot` (`prompts/prompts_retrieval.py`,
`prompts/constants/{common,retrieval}.py`, `prompts/utils.py`), rendered as one string the way
LOFT's `dump_prompts.py` does (the content chunks concatenated, then stripped) and sent as one
user turn:

1. SciFact corpus instruction ("You will be given a list of passages. ...") and LOFT's
   formatting instruction (`Final Answer: [id1, id2, ...]`).
2. Every passage as `ID: <n> | TITLE: <title> | CONTENT: <passage> | END ID: <n>`, where `<n>`
   is the passage's position in the corpus (LOFT's `pid_mapper`), not its raw SciFact id.
   Titles and passages are stripped and cleaned as `utils.load_data_from_file` does.
3. The 5 few-shot examples, each opened by `====== Example <k> ======`, with LOFT's
   reverse-CoT query format ("Which passage is most relevant to the claim, ... Print out the
   TITLE and ID of the document. ..."), one `TITLE: <title> | ID: <n>` line per gold passage,
   and `Final Answer: ['<n>']`.
4. `====== Now let's start! ======` and the test claim in the same query format.

gbench checks this against the upstream code: for all 100 test queries the prompt is
byte-identical to `concatenate_chunks(example.all_chunks).strip()` from LOFT's own registry at
the pinned commit.

**Extraction and scoring.** LOFT's `utils.extract_prediction`: the first response line that
contains `[...]` is parsed with `ast.literal_eval` (nothing else is tried). Each predicted ID is
mapped back to a passage; an ID outside the corpus makes LOFT drop the whole response
(`inference/models.py` raises and `run_evaluation.py` scores the unanswered query 0), and gbench
does the same. The metric is LOFT's primary retrieval metric, **recall@1**
(`evaluation/retrieval.py`): `|{first prediction} & gold| / |gold|`. Every test query has one
gold passage, so recall@1 is 0 or 1 per query and the headline `accuracy` is the percentage of
queries whose first predicted passage is the gold one. A request that got no reply from the
served endpoint (failed, timed out, or over the context window) is a harness error, never the
model's answer: it is left out of `accuracy`, counted and named in `status_reason`; when every
request got none, `accuracy` is null with `accuracy_withheld_reason`.

**Decoding:** LOFT's inference settings (`inference/models.py` @ `219f68e`:
`TEMPERATURE = 0.0`, top_p 1.0): top_p 1.0 is sent in both modes, and the temperature follows
gbench's rule: **1.0 with `--thinking`, 0.0 without**. The result
records `official_sampling` (the reference, the temperature used, its source,
`temperature_matches_official` and `send_max_tokens: false`). LOFT's `GenerationConfig` sets no
`max_output_tokens`, so no request carries a `max_tokens`: the server grants each reply the rest
of the context window, and `--max-output-tokens` is not applied in either mode
(`max_output_tokens_note` says so).

## Reading the result

`leaderboard_comparable` is `true` for the full 100-query test split decoded at temperature 0.0
without `--thinking`, and the number is then comparable with LOFT's published SciFact 128k
recall@1 (it is not a LOFT "arXiv" number; there is none). A `--eval-limit` subset, a `--thinking`
run, a non-zero resolved `temperature` (gbench's 1.0 of a `--thinking` run, or a value from
`--temperature` or `GBENCH_LOFT_SCIFACT_TEMPERATURE`) or a request that got no reply
(failed, timed out or over the
context window; a harness error, left out of the score) or a reply cut off by the run's token
budget rather than at an official cap (`replies_cut_off_by_budget`; `replies_cut_off_in_reasoning`
counts those that ran out inside the reasoning channel with no answer) sets it to `false`, and
`leaderboard_comparable_reason` names each cause.

## Requirements

- A serving endpoint with a **context window of at least ~128k tokens**: the prompt is the
  whole 128k corpus plus the few-shot examples (115,068-115,112 tokens per query with the Gemma 4
  E4B chat template), and the answer needs room after it. A prompt that does not fit is recorded
  as `OVER_CONTEXT` and scored 0.
- Network access to `storage.googleapis.com` on the first run (then cached).
- No API key, no judge, no extra packages.

License: LOFT's README licenses its software under Apache-2.0 and its other materials under
CC-BY 4.0, and notes that individual tasks may carry their original owners' terms (SciFact comes
from the BEIR collection); check them before publishing results.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals loft_scifact \
       --max-output-tokens 65536
```
`--max-output-tokens` is required by the CLI but not applied (see Decoding). Add
`--eval-limit N` for a quick subset of N queries (stratified, and not leaderboard-comparable).
The run logs its planned workload at its start and records it as `eval_limit_plan`.

## Sharding

`--shard I/N` (or `GBENCH_SHARD=I/N`) runs shard I of N of the suite's full, ordered queries
(round-robin: positions I, I+N, I+2N, ...), selected before the `--eval-limit` draw, which is then
taken within the shard (N shards run up to N x `--eval-limit` queries). The result records
`shard`, `shard_index` and `shard_count`, and a shard of more than one is not
leaderboard-comparable (the shard reason is named in `leaderboard_comparable_reason`); `gbench
--merge` combines the shards' results as docs/distributed-gbench.md describes: it re-counts the
union of their traces, and refuses a suite whose headline is not a correct / scored count over
them unless the suite defines its own merge.
