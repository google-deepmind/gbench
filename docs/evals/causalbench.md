# causalbench

CausalBench here is **Corr2Cause** (`causal-nlp/corr2cause`, Jin et al., ICLR 2024, "Can Large
Language Models Infer Causation from Correlation?"): pure causal inference from correlations.
Each item states the full set of correlations and independencies among 2 to 6 variables (the
premise) and a causal hypothesis ("A directly causes B", "there is a collider of A and B",
...); the model must say whether the hypothesis follows. **Scoring: the release protocol of
causalNLP/corr2cause; the direct Yes/No prompt at temperature 0, the reply read by upstream's
own normalizer, and the headline is the F1 of the valid class.**

## Dataset
- `causal-nlp/corr2cause`, split `test`, pinned to revision
  `42ba12c769e11ff6427c9f52d7db58e3f9bf3e53`: 1,162 items, 180 valid (label 1) and 982 invalid
  (label 0). Each row's `input` is `Premise: <premise>\nHypothesis: <hypothesis>`, which is
  upstream's `prompt_tmpl_generic`; gbench splits it back into premise and hypothesis.
- `--shard I/N` and `--eval-limit N` are applied by gbench's common runner, shard first; the
  subset is stratified over the number of variables with a fixed seed; a shard of more than one is
  not leaderboard-comparable (the shard reason is named in `leaderboard_comparable_reason`).

## Protocol
causalNLP/corr2cause `code/run_model.py` at commit `21d80350e27a6206f536cf90a09e95d9e856c761`,
the path the paper uses for chat models (`GPT.query_by_prem_and_hypo` without CoT,
`GPT.query_unseen_text`, `Model.evaluate`), with the `Chatbot` it calls from
zhijing-jin/efficiency (`efficiency/nlp.py` at commit
`69d191bc1009be5d4f1320388367ff6a0733433b`):
- **Messages**: the system prompt `GPT.__init__` gives the Chatbot, `You are a highly
  intelligent question-answering bot with profound knowledge of causal inference.`, then the
  user message `prompt_tmpl_direct` (the last of its assignments in `Constants`):
  `Question: <premise>\nCan we deduct the following: <hypothesis>? Just answer "Yes" or
  "No."\n\nAnswer:`, with the hypothesis stripped and its final period removed.
- **Generation**: temperature 0 (`Chatbot.raw_query`'s default) and `max_tokens=10`
  (`GPT.query_unseen_text` for a prompt ending in `Answer:`). The 10 is the suite's official cap
  (`OfficialSampling.max_tokens`), applied by gbench's token-limit policy: a no-think request
  carries `max_tokens=10` exactly, whatever `--max-output-tokens` says (`max_output_tokens_note`
  records a flag that was not applied; `causalbench_report.max_tokens_sent` is `[10]`), and a
  reply that stops at 10 tokens is the protocol's outcome (`replies_at_official_max_tokens`),
  not a truncation problem. A `--eval-thinking` request carries the run's budget instead
  (`--max-output-tokens`, else the thinking floor), because the 10 was written for a direct
  answer and the reasoning shares the budget; `official_sampling` records `max_tokens: 10` and
  `max_tokens_applied: false`. A thinking run is not the published protocol and is flagged; its
  replies longer than 10 tokens are counted in `causalbench_report.replies_over_upstream_cap`.
- **Reading the reply**, ported verbatim: `GPT.clean_pred_func` (lower-case; strip spaces,
  quotes, periods and colons; drop a leading `that there is `), then `Constants.rel_normalize`:
  the whole reply looked up in `pred_rel_lookup` (`yes`, `true`, `necessarily true`,
  `must be true` mean "entailment"; `no`, `false`, `necessarily false`, `neutral`,
  `not necessarily true` and the like mean "contradiction"), else its first two nltk word
  tokens, else its first one; anything else is "contradiction". So `Yes`, `"Yes".` and
  `Yes, because ...` read as valid; `No.`, an empty reply, `**Yes**` and `I think yes` read as
  invalid, as upstream reads them. Label 1 is "entailment", label 0 "contradiction".
- **Metric** (`Model.perf_df2report_dict`): F1, precision and recall of the minority label (the
  label with the least gold support among those in the gold or the predictions; on the test set
  the valid class, "entailment"), and accuracy.

## Headline metric
`accuracy` is the **F1 of the minority (valid) class**, in percent, Corr2Cause's main number:
about 85% of the test set is invalid, so plain accuracy rewards always answering "No".

Also reported:
- `raw_accuracy`: plain accuracy, in percent.
- `causalbench_report`: `f1`, `precision`, `recall` and `accuracy` (in percent), the
  `minority_label`, the `tp`, `fp` and `fn` counts, the same metrics `by_num_variables` and
  `by_relation` (the dataset's `template`, upstream's `causal_relation`),
  `upstream_max_tokens` (10), `max_tokens_sent` (the per-request max_tokens values sent: `[10]`
  without thinking, the run's budget with it) and `replies_over_upstream_cap`.
- `headline_metrics`: `F1`, `P`, `R` and `Acc` (so the summary's CSV and console show them).
- `category_accuracy`: the headline metric per number of variables (`2_vars` ... `6_vars`: the
  minority-label F1, as `causalbench_report.by_num_variables`); the plain per-sample accuracy
  per number of variables is `category_label_accuracy`. `correct_answers` is `null`
  (`correct_answers_note`): the exact-label matches are `exact_label_matches`, the numerator of
  `raw_accuracy`, not of the F1. With nothing scored every figure is `null`, never 0.
- Each scored trace records its `predicted_relation`.

Every figure above is over the scored items, as base's `accuracy`: a request that got no reply
from the served endpoint (failed, timed out, or over the context window) is a harness error,
never read as a prediction. It is left out, counted and named in `status_reason` (status
`completed_with_errors`); when every request got none, `accuracy` and `raw_accuracy` are null
and `accuracy_withheld_reason` says why.

`leaderboard_comparable` is true for a full run at temperature 0 without thinking in which every
request was answered. Otherwise `leaderboard_comparable_reason` names each cause: more than one
sample per item (`--attempt-count`, pooled into the metric), a `--eval-thinking` run (the
published protocol reads a direct answer), a temperature other than 0 (gbench's 1.0 of a
thinking run, or an operator-set value), a no-think request sent
a `max_tokens` other than 10 (only when the prompt plus 10 does not fit the context window),
requests that got no reply (harness errors, left out of the score; upstream's Chatbot retries
them), replies still null after gbench's re-asks (below), or an `--eval-limit` subset.

**Null replies are asked again, as the Chatbot does.** zhijing-jin/efficiency's `Chatbot.raw_query`
(commit 69d191bc) calls `response_text.strip()` on the reply's `message.content`; on None that
raises, and `Chatbot.ask` catches it ("Unknown exception when calling openai") and asks again
after a doubling wait, with no limit. An empty-string reply is not asked again: it is read as
"contradiction". gbench's requests go through a loopback relay that applies the same rule
(`swebench_common.upstream_reask_relay`, `reply_content_is_null`), up to `NULL_REPLY_REASKS` = 5
re-asks of the same request (the waits are not reproduced; they change nothing that is
sampled). A reply with its answer inside the thinking channel has null content, so it is asked
again rather than read as "contradiction". Only a reply the model ended itself (`finish_reason`
"stop") is asked again: a reply cut off at the token budget (`finish_reason` "length", e.g. a
thinking reply that ran out of budget inside its reasoning) is the model's own outcome, never
re-asked, scored as base scores a truncated reply and named (`upstream_reask.cut_off_not_reasked`).
The re-asks stay inside base's per-request timeout (`EndpointRelay.reask_deadline`); a request whose
re-asks run out, reach that deadline or get no reply from the endpoint is scored on the last reply
it got (`reasks_exhausted`, of them `reasks_cut_short`), never a harness error. A reply still null
after the 5 re-asks is read as base
reads it ("contradiction"; an answer-in-thinking one is counted in `answer_in_thinking`) and
named in `leaderboard_comparable_reason`. The result's `upstream_reask` records the rule,
`reask_limit`, `requests_reasked`, `reasks_sent`, `reasks_exhausted`, `reasks_cut_short`,
`cut_off_not_reasked`, the replies the relay saw and their empty-reply classes
(`answer_in_thinking_seen`, `answer_in_thinking_rate_seen`, ...); see
[answer_in_thinking.md](answer_in_thinking.md).

## Sampling
gbench's rule: temperature 1.0 with `--eval-thinking`, 0.0 without (Corr2Cause's 0). The result
records Corr2Cause's setting under `official_sampling`
(`temperature_matches_official: false` for a thinking run). `--temperature`, or for this suite
alone `GBENCH_CAUSALBENCH_TEMPERATURE` (which takes precedence), overrides it in both modes; a
value other than 0 makes the run non-comparable.

## Requirements
- A running OpenAI-compatible `/v1` chat endpoint. No judge, no sandbox, no API key.
- `nltk` (in gbench's `evals` extra, `pip install -e ".[evals]"`, or `pip install nltk`) with its
  punkt tokenizer data: `python -m nltk.downloader punkt punkt_tab` (set `NLTK_DATA` if the data
  lives elsewhere). Without them the suite stops with an INFRA REQUIRED error before generating.
- The `datasets` package (base install) and network access to the Hugging Face Hub on the
  first run.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals causalbench --max-output-tokens 65536
```
`--max-output-tokens` applies to a `--eval-thinking` run only. Add `--eval-limit 20` for a quick
check.
