# medxpertqa

MedXpertQA (`TsinghuaC3I/MedXpertQA`, pinned revision `7e7c465a`): expert-level medical exam
questions over 17 specialties, as multiple choice. Both official subsets run:

| Subset | Questions | Options | Images |
|---|---|---|---|
| Text | 2,450 | A-J | none |
| MM | 2,000 | A-E | 1-6 per question, from the dataset's `images.zip` |

Each subset splits into `Reasoning` and `Understanding` questions.

## Protocol
The official zero-shot chain-of-thought protocol of TsinghuaC3I/MedXpertQA
(`eval/main.py` zero_shot_cot, `eval/config/prompt_templates.py`, `eval/utils.py`
answer_cleansing, @ `5f1db162`), a two-turn conversation per question:

1. System `You are a helpful medical assistant.` (in both modes); user `Q: {question}\nA:
   Let's think step by step.` followed by the question's images (MM only) as JPEG data URIs.
   The question text already ends with its `Answer Choices: (A) ... (E) ...` list, so the
   options appear once.
2. The model's reply goes back as the assistant turn, then the user sends
   `Therefore, among A through E, the answer is` (`A through J` for Text).

The reply to turn 2 is read as upstream does: the text after that trigger, or after
` answer is `, and then the FIRST standalone option letter in it. It is correct when that
letter is the gold label.

- Token limit (the token-limit policy): upstream's `APIAgent` (`eval/model/api_agent.py`)
  sends **`max_tokens` 2048** on every call for a served model's name (see Sampling below).
  Without `--eval-thinking` both turns carry 2048, so `--max-output-tokens` (required by the
  CLI) is not applied (`max_output_tokens_note` says so) and a turn that stops there goes on as
  it arrived (`replies_at_official_max_tokens` counts them, for turn 1 under
  `medxpertqa_rationale_turn`). With `--eval-thinking` the 2048 is not applied, because it was
  written for non-reasoning generation: both turns are sent the run's `--max-output-tokens`,
  and the result records the cap as not applied to reasoning
  (`official_sampling.max_tokens_applied: false`, `max_tokens_sent`).
- An empty turn-1 reply (for example reasoning that used the whole budget) still gets the
  answer trigger, after an empty assistant turn, as upstream does (it appends whatever turn 1
  returned). A request of either turn that gets no reply from the served endpoint (an error, a
  timeout, or a prompt over the context window; a failed turn-1 request leaves no turn to
  continue from) is a harness error, never the model's answer: the question is left out of
  every accuracy (`scoring_excluded`, `request_harness_errors` by kind) and named in
  `status_reason`, which lists each turn's causes. Turn-1 health is reported under
  `medxpertqa_rationale_turn`; the top-level failure counters add both turns, so an empty
  turn-1 reply is counted there even when its turn 2 is answered. A reply of either turn that
  stopped with its answer inside the thinking channel is the model's outcome, not a failed
  request: counted in `answer_in_thinking` (both turns added; `answer_in_thinking_rate` over the
  requests of both turns; see [answer_in_thinking.md](answer_in_thinking.md)).
  `medxpertqa_unanswered_questions` counts the questions with no reply in either turn. The status
  is `failed` when no question is scored (`accuracy` null with `accuracy_withheld_reason`),
  `success` when neither turn has a failure (and at most 10% of the replies are truncated), and
  `completed_with_errors` otherwise.
- Under `--shard I/N` (`GBENCH_SHARD`), turn 1 runs this node's shard and turn 2 answers every
  rationale from it; the process environment is not changed.
- A missing `images.zip`, or an image missing from it, is a hard error (`infra_required`).
  The suite never falls back to another dataset or to text-only MM questions.
- `--eval-limit N` draws N questions stratified over subset x question type.
  The run logs its planned workload at its start and records it as `eval_limit_plan`.

## Reading the result
- `accuracy` is accuracy over all questions of both subsets (4,450 in a full run), as in
  upstream `eval.ipynb`, leaving out the questions that got no reply (harness errors, above);
  the per-subset and per-category numbers and the @k block leave them out too.
- `medxpertqa_text_accuracy` and `medxpertqa_mm_accuracy` are the per-subset numbers that
  the MedXpertQA leaderboard lists. `category_accuracy` splits each subset by question type
  (`Text/Reasoning`, `Text/Understanding`, `MM/Reasoning`, `MM/Understanding`).
- `leaderboard_comparable` is true only for a full, unsharded run at the official temperature
  (0; a `--eval-thinking` run samples at 1.0, so it is not comparable) in which every request of
  both turns got a reply and no
  reply was cut off by the run's token budget. Otherwise `leaderboard_comparable_reason` names
  what differs: an `--eval-limit` subset, a `GBENCH_SHARD` shard, a temperature other than 0
  (gbench's 1.0 of a thinking run, or a value set with `--temperature` or
  `GBENCH_MEDXPERTQA_TEMPERATURE`),
  the requests that got no reply (failed,
  timed out or over the context window; harness errors, left out of the score), or the replies
  cut off by the run's token budget.
- Sampling, as `APIAgent` sends both turns for a served model, in both modes. `APIAgent` keys
  its settings on the model name: `get_response` sends names containing `o1` or `o3`, and
  `deepseek-reasoner`, down a branch that drops the system message and sends no temperature and
  `max_completion_tokens`, and every other name down the chat branch; `__init__` gives names
  containing `o1`, `o3`, `qwq`, `qvq` or `reasoner` 8192 tokens and every other name 2048
  (`config/prompt_templates.py` also changes the prompts for `deepseek-reasoner` and `qvq` by
  name). Those are model names of hosted APIs (OpenAI's o-series, DeepSeek's
  `deepseek-reasoner`) and of two Qwen checkpoints, not settings upstream declares for reasoning
  models in general, so they are not reasoning-model settings for a served model: a served
  model's name (for example `google/gemma-4-31b-it`) takes the chat branch and the 2048, as would
  the other open models listed in upstream's `config/model_info.json` (`qwen2.5-72b-instruct`,
  `qwen2.5-vl-72b-instruct`, the Llama 3 models) once added to a client list, as the README says
  to do for more models. The chat branch keeps the system message and sends temperature 0
  (`eval/scripts/run.sh` `temperature=${7:-0}`, `main.py` `--temperature` default 0) and seed 0;
  with `--eval-thinking` the request is the same except for the token limit above and the
  temperature: gbench samples every request by its rule, **1.0 with `--eval-thinking` and 0.0
  without**, whatever the upstream sends. The chat
  branch's `logprobs=True` only returns token log-probabilities for upstream's confidence field;
  it does not change the reply or the score and is not requested. The result records
  `official_sampling` (the settings, the temperature used, its source and
  `temperature_matches_official`).
- A reply that stops at the official cap of its turn (2048, without `--eval-thinking`) is the
  protocol's outcome and is counted in `replies_at_official_max_tokens`. A reply cut off by
  another budget (with `--eval-thinking` the run's `--max-output-tokens`, or a `max_tokens` the
  context window lowered) is counted in `replies_cut_off_by_budget` over both turns
  (`replies_cut_off_in_reasoning` for those that ran out inside the reasoning channel with no
  answer) and makes the run not comparable.

## Requirements
- A **vision-language model** endpoint (OpenAI-compatible `/v1/chat/completions` that accepts
  `image_url` data URIs and multi-turn conversations with a system message).
- `datasets` and `huggingface_hub`: the suite downloads both dataset configs and `images.zip`
  (the pinned revision) from the Hugging Face Hub on first use and extracts the zip next to
  the cached file (`images.zip_extracted`).
- `GBENCH_MEDXPERTQA_TEMPERATURE` (or `--temperature`) overrides the temperature for this suite,
  in both modes; otherwise gbench's rule applies: 1.0 with `--eval-thinking`, 0.0 without.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals medxpertqa \
       --max-output-tokens 65536 --eval-limit 20
```
Drop `--eval-limit` for the full 4,450-question run; add `--eval-thinking` for reasoning mode.
