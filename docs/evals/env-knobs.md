# gbench environment knob reference

Every gbench evaluation knob is an environment variable named `GBENCH_<SUITE>_<KNOB>` (for a per
suite knob) or `GBENCH_<AREA>_<KNOB>` (for a cross cutting one). This page is the consolidated list.
Per suite provisioning details also live in each `docs/evals/<suite>.md`.

Command line flags (`--temperature`, `--max-output-tokens`, `--eval-limit`, `--sandboxes`,
`--thinking`, `--eval-n-shot`, ...) are documented by `gbench --help`; this page covers the
environment knobs, which tune behavior that has no dedicated flag.

`--eval-limit N` is N samples per suite in total (one sample is one distinct dataset item:
question, task, instance, problem, conversation), never N per tab, split, setting, language,
domain, scenario, pass or mode; a suite with several settings spreads the N over them, and a
setting that gets 0 is not run. The attempts a protocol makes per sample (lcb's n=10, arc_agi's 2
attempts, avg@k / pass@k repeats, `--attempt-count`, and the trial counts and per-task budgets
below) are unchanged and are not extra samples. Each suite logs its planned workload in one line
at its start and records it in its result as `eval_limit_plan`.

## Convention and precedence

- Canonical form is `GBENCH_<SUITE>_<KNOB>`, upper snake case, where `<SUITE>` is the registered
  suite name. Example: `GBENCH_SCICODE_EVAL_TIMEOUT_S`.
- Some suites shipped older bare prefixed names (for example `TAU2_NUM_TRIALS`,
  `SCICODE_EVAL_TIMEOUT_S`, `SPIDER2_REPO_DIR`, `PUTNAM_LEAN_IMAGE`). Those still work as
  deprecated aliases: the canonical `GBENCH_` name takes precedence, and using a bare name logs a one
  time warning naming its replacement. New scripts should use the canonical names.
- Temperature (the maintainer's rule, the same for every suite): a `--thinking` run samples the
  model under test at temperature 1.0 (`temperature_source` "thinking") and a run without
  `--thinking` at 0.0 (`temperature_source` "default"). gbench always sends that temperature,
  through the suite's own request or, for a suite whose harness, agent or container sets
  sampling, through that harness's own mechanism. A benchmark's official or upstream temperature
  is never applied, and a suite whose official harness sends no temperature gets one too. The
  operator overrides the rule, in both modes, with (highest first) `GBENCH_<SUITE>_TEMPERATURE`,
  then `--temperature`. Other official sampling fields (top_p, top_k, penalties, seed, stop
  sequences, system message) are not part of this rule: a suite keeps them as listed below. A
  suite that declares its official sampling (`OfficialSampling`) still records the official
  values on the result as `official_sampling`, with `temperature_source` and
  `temperature_matches_official`. See `resolve_temperature` and `resolve_official_sampling` in
  `gbench/runners/eval_suites/base.py`.
- Suites that declare official sampling, with the fields other than the temperature they send
  (in both modes unless noted):
  - gsm8k: seed 1234. wildbench, loft_scifact and coco_caption: top_p 1.0. ruler: top_p 1.0, seed
    0. acebench: top_p 0.9. complexfuncbench: max_tokens 2048 without `--thinking`
    (ComplexFuncBench models/qwen.py), the run's budget with it. bigcodebench: without
    `--thinking` top_p 1.0 and upstream's stop sequences on the prefilled path; with `--thinking`
    make_request's served-model request (top_p 0.95, the run's token budget). The lmms-eval suites
    (chartqa, textvqa, docvqa, infographicvqa, coco_caption) keep the task's other
    generation_kwargs with `--thinking` (lmms-eval's chat client runs a thinking model with
    enable_thinking_kwarg, which keeps them).
  - charxiv: top_p 1.0 and seed 42 (gpt.py; o1.py is keyed on OpenAI o1 model names, not a
    reasoning-model setting for a served model).
  - medxpertqa: seed 0 and the system turn (api_agent.py's chat branch, the one a served model
    takes; its reasoning branch, which drops the system turn, is keyed on o1/o3 and
    deepseek-reasoner model names).
  - mmlu_pro: top_p 0.95 (TIGER-Lab evaluate_from_apiX.py; its max_tokens=32768 is sent exactly
    without `--thinking` and is also its reasoning-model setting, so a `--thinking` run sends
    32768, or `--max-output-tokens` when that is larger).
  - humanitys_last_exam: no max_tokens of its own in either mode.
  - gpqa, gpqa_diamond, multilingual_mmlu, simpleqa, healthbench and browsecomp: simple-evals' chat
    system message "You are a helpful assistant." in both modes (simple-evals' chat sampler, the
    path a served model takes in both modes); max_tokens 2048 without `--thinking`, the run's
    budget with it (the 2048 is not applied to reasoning).
  - gaia2 (the ARE paper's agent protocol, arXiv:2509.17158 section 4.1): the agent's stop
    sequences `<end_action>` and `Observation:` (the paper's appendix B.5, which adds them for
    models that continue past a single tool call, naming Claude, Kimi and Qwen, and ran GPT-5
    without them; gbench sends them for every served model) sent as `stop` without `--thinking`
    (with it none is sent: the served stack matches stop strings inside the reasoning too, so the
    agent request is streamed and closed once the answer holds a stop sequence; ARE's client-side
    split of the answer applies on both runs); max_tokens 16000 per agent turn without
    `--thinking`, the run's budget with it; 200 steps.
  - lcb: top_p 0.95, frequency and presence penalties 0. hmmt and aime: the sampling fields,
    reasoning_effort and max_tokens of MathArena's own configs/models yaml for a served model it
    ships one for (hmmt.MATHARENA_MODEL_CONFIGS, 59 open-weight configs at b89f2f0a matched by
    huggingface_id, e.g. Qwen/Qwen3.5-27B top_p 0.95, presence_penalty 1.5, top_k 20, min_p 0.0,
    repetition_penalty 1.0, max_tokens 192000), else the top_p and top_k of the served model's
    generation_config.json (0.95 and 64 for Gemma 4, which MathArena ships no config for; an
    unreadable file is infra_required); ojbench the same fields of the same file (an unreadable
    file is infra_required).
  - multi_swe_bench: top_p 0.95 (MSWE-agent's run.py / multirun.py get_args defaults); the
    temperature is passed as MSWE-agent's own --temperature.
  - terminal_bench: the TB2.1 leaderboard's Terminus-2 settings (configs/leaderboard.yaml):
    Terminus-2's json parser, agent-timeout multiplier 1.0, no thread-cap env. Terminus-2 itself
    sends no max_tokens; gbench's relay sends the run's `--max-output-tokens` on every turn that
    carries none (`GBENCH_AGENT_MAX_TOKENS_FROM_RUN`, on by default; `0` sends none, as upstream).
  - cyberseceval (PurpleLlama CybersecurityBenchmarks llms/llm_base.py DEFAULT_TOP_P, sent by
    llms/openai.py OPENAI.query): top_p 0.9 and no max_tokens.
  - toolbench: its StableToolBench requests also carry frequency_penalty 0 and presence_penalty 0,
    recorded as its official params. livebench: a served model with a LiveBench config sends that
    config's top_p (gemma-4-31b-it: 0.95).
  - No max_tokens sent (the server's default applies: the room left in the context window;
    --max-output-tokens is recorded as not applied in `max_output_tokens_note`): mcp_atlas,
    cyberseceval (PurpleLlama's OpenAI adapter sends max_completion_tokens only for its
    proprietary reasoning-model list), mrcr (the openai/mrcr card reference loop),
    cimemories, deepsearch_qa (the starter notebook's model call sets none), gdpval (the paper sets
    none), scicode (SciCode sends none for litellm/* models), humanitys_last_exam, bfcl, bfcl_live,
    seal_tools and api_bank (their chat wrappers send only the model and messages),
    nexusraven_api_eval (LangChain 0.0.294 OpenAIChat), gorilla_apibench (get_llm_responses.py),
    lab_bench (LAB-Bench's openai_utils sends only model and messages), loft_scifact, beam_128k,
    screenspot, screenspot_pointing, and ojbench without `--thinking` (the paper gives
    non-reasoning models their default context length); with `--thinking` also bigcodebench and
    aa_lcr.
  - Upstream sends no max_tokens, gbench sends the run's `--max-output-tokens` on every agent
    request that carries none (`GBENCH_AGENT_MAX_TOKENS_FROM_RUN`, on by default; `0` sends none,
    as upstream): terminal_bench (Terminus-2 and LiteLLM send none), aider_polyglot (aider sends
    none), skillsbench, swe_lancer, tau2, tau3 and bfcl_v4_agentic (gbench's relay fills it into
    requests without max_tokens), agent_dojo (AgentDojo's OpenAILLM sends none; gbench's client
    sets it), swe_bench_live (every tab), swe_bench_pro and multi_swe_bench (MSWE-agent) (a relay
    body field), copilot_bench_swe and swe_bench_multilingual (mini-swe-agent's `model_kwargs`).
- Token-limit policy (`resolve_token_budget` in `gbench/runners/eval_suites/base.py`;
  `OfficialSampling.max_tokens`, `send_max_tokens` and `reasoning`): official max-token caps come
  from harnesses written for non-reasoning generation. Without `--thinking` a suite sends its
  official cap exactly, with no `--max-output-tokens`, floor or tool-loop turn ceiling, or no
  max_tokens at all when the official sampler sends none. With `--thinking` an official cap is not
  applied: the request is sent the run's budget (`--max-output-tokens`, else the 32,768 thinking
  floor) and the result records it as "official cap not applied to reasoning"
  (`official_sampling.max_tokens_applied: false`, or `official_cap_not_applied_to_reasoning` for a
  per-request max_tokens a sample declares, which is replaced by the run's budget whether it is
  larger or smaller). A suite whose official sampler sends no max_tokens cannot declare one per
  request (the run stops with an error). When the upstream defines reasoning-model settings
  (`OfficialSampling.reasoning`), a `--thinking` run uses them instead, never capped below
  `--max-output-tokens` (a larger `--max-output-tokens` is sent instead of the reasoning cap and
  recorded as `official_sampling.max_tokens_applied: false` with `max_tokens_sent`). A reply that
  stops at an official cap, still writing or repeating itself, is counted in
  `replies_at_official_max_tokens`: it is the protocol's outcome, scored on what arrived, and
  neither counts in `genuinely_truncated` nor toward the share of truncated replies that makes a
  run `completed_with_errors`. A suite that declares no official cap keeps `--max-output-tokens`
  and the house floors. Official caps sent without `--thinking`, recorded in
  `max_output_tokens_note`: lcb 2000 (self-repair included), multipl_e 2048, bigcodebench 1280,
  livebench 4096 (gen_api_answer.py --max-tokens default; a served model's LiveBench config cap
  instead, e.g. gemma-4-31b-it 65536; LiveBench sends one fixed max_tokens, so any budget that
  would leave the longest prompt under 8192 tokens of the served window (or half of it) is lowered
  to fit, recorded as max_output_tokens_source ending in fit-to-window, and a lowered official cap
  makes the run not comparable), mmlu_pro 32768 (also its reasoning-model setting: a `--thinking`
  run sends 32768, or `--max-output-tokens` when larger), gpqa, gpqa_diamond, multilingual_mmlu,
  simpleqa, healthbench and browsecomp 2048, gsm8k 256, causalbench 10, cruxeval 100 (direct
  prompts; the CoT
  prompts' 1000 is recorded as not applied to reasoning), longbench_v2_code 128 (its w/ CoT
  setting: 1024 for the CoT query and 128 for the answer query), mmlu_redux 1
  (HELM's one-token answer; a `--thinking` run thinks with the run's budget), acebench 1024 (within
  LLMInfer's 8192-token window as ACEBench's vLLM 0.6.1.post1 applies it: a request whose prompt
  leaves less room gets that room, and a prompt longer than the window is not sent: its reply is the
  empty text that vLLM returns for it), nestful 1000,
  t_eval 512, ruler its per-task caps, chartqa 16, docvqa 32, infographicvqa 32, textvqa 1024 and
  coco_caption 64 (lmms-eval declares no reasoning-model cap, so with `--thinking` these are not
  applied), charxiv 1000, mmmu_pro 4096, wildbench 4096, omnidocbench 32000, medxpertqa 2048
  (MedXpertQA api_agent.py, the chat branch a served model takes; with `--thinking` the 2048 is
  not applied and `--max-output-tokens` is sent: the 8192 is keyed on o1/o3/qwq/qvq/reasoner model
  names), gaia 4096 (run_gaia.py model_params), gaia2 16000 per agent turn (the ARE paper, applied
  in its image's hooked ARE engine), toolbench 1024 on every DFSDT turn (chatgpt_function_model.py),
  mcp_bench its per-call caps (planning rounds and the final answer 12000, tool-result summaries
  10000, format fixes 8000, a compression call its target length; config/benchmark_config.yaml),
  and aa_lcr 16384. Reasoning-model settings used with `--thinking`, never below
  `--max-output-tokens`: ojbench 65536 (the paper's 64k), mmlu_pro 32768, hmmt
  and aime a MathArena model config's max_tokens (e.g. 192000 for Qwen3.5-27B) and livebench a
  LiveBench model config's cap; aa_lcr sends none, and bigcodebench sends the run's budget (its
  1280 cap is not applied to reasoning). mmlu_redux (HELM) and gsm8k (lm-eval) keep their other no-think sampling with `--thinking`
  (gsm8k's seed 1234), and gpqa, gpqa_diamond, multilingual_mmlu, simpleqa, healthbench and
  browsecomp keep simple-evals' chat sampler's system message (its 2048 is not applied to
  reasoning). The path a served model takes (HELM's
  VLLMChatClient, an OpenAIClient; lm-eval's local-chat-completions) sends the same request whether
  the model reasons or not. The o-series branches of HELM's OpenAIClient (model names ^o<digits> or
  ^gpt-5) and of lm-eval's openai-chat-completions (o1, o3, o4, 5) key on OpenAI model names to fit
  OpenAI's hosted API, so they are not reasoning-model settings for a served model. Neither are
  the o-series entries of simple-evals' simple_evals.py (lines 145-212 at 652c89d0:
  ResponsesSampler with reasoning_model=True for o3, o4-mini and o1-pro, OChatCompletionSampler
  for o1, o1-preview, o1-mini and o3-mini; no system message, temperature or max_tokens), which
  are keyed on OpenAI model names the same way. lcb (self-repair included), multipl_e,
  bigcodebench, ojbench, scicode, hmmt,
  aime and livebench count every reply that stopped at a limit other than the official cap (the
  run's budget, a max_tokens the context window lowered, or the end of the window where no
  max_tokens is sent) in `replies_cut_off_by_budget`, and those with no answer text in
  `replies_cut_off_in_reasoning`, and report such a run as not comparable (livebench reads
  LiveBench's answer rows, with and without `--thinking`: its token_exhaustion rows, and replies cut
  off after the answer began, from total_output_tokens and the answer's tokens under the run's
  `--tokenizer`; rows it cannot check are counted in `replies_cut_off_uncounted` and also make the
  run not comparable, and its api_error rows, requests that failed at the served endpoint, are
  counted in `failed_requests` with status `completed_with_errors`; LiveBench scores them 0 inside
  its averages, so the headline (`accuracy` null with `accuracy_withheld_reason`) and the score of
  each category with one (`dimension_scores_withheld`, `failed_requests_by_category`) are withheld,
  and requests the served vLLM server aborted during the run withhold every score).
  aider_polyglot counts aider's
  own num_exhausted_context_windows in `exhausted_context_windows` and
  `replies_cut_off_by_budget`: replies cut at the run's `--max-output-tokens`, which gbench's relay
  sends (or at the end of the served window with `GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0`, where aider
  sends no max_tokens), and requests the window refused. Such a run is reported as not comparable. With
  `--thinking`, mmlu_redux counts replies cut off at the run's budget inside the reasoning channel
  (`replies_cut_off_in_reasoning`) and reports such a run as not comparable; it also counts
  thinking replies that stopped with the answer inside the thinking channel (`answer_in_thinking`,
  the model's outcome, scored wrong, see answer_in_thinking.md) and replies with nothing at all
  (`empty_responses`, counted in `failed_requests`), and reports `completed_with_errors` when more than 10% of its replies are cut
  off or any request failed, as base.run_eval_suite does. gpqa, gpqa_diamond, multilingual_mmlu,
  ifeval and arc_agi report a run as not comparable when a reply was cut off before it finished at
  anything but an official cap (for the simple-evals suites with `--thinking`, a reply at the
  run's budget or at a max_tokens the context window lowered). arc_agi counts every attempt the
  server stopped at the budget (finish_reason length), whatever health base gives it afterwards,
  non_convergent included, for both the flag and the >10% status rule. mcp_bench and toolbench
  apply the policy inside their images (gbench/docker/mcp_bench_request_policy.py and
  docker/toolbench_request_policy.py, identical copies; an image without the label
  org.gbench.model-request-policy=2 is refused). Each
  model-under-test request carries chat_template_kwargs.enable_thinking for the run's mode. With
  `--thinking` the request is sent the run's budget, cut to the room the context window leaves
  after the prompt, with a request timeout sized to it (`GBENCH_REQUEST_TIMEOUT_S` + budget /
  `GBENCH_MIN_DECODE_TOK_S` seconds, recorded in `model_request_timeout_s`; without `--thinking`
  the harness client's own timeout applies). The replies are counted in `model_replies`, and those
  cut at max_tokens in a `--thinking` run are named in `leaderboard_comparable_reason`. A request
  that ends in an error of the served endpoint or of the connection is a harness error
  (`model_replies.failed_requests`, `timed_out`, `model_failed_by_error`, `harness_errors`), is
  named in the reason, and makes the status `completed_with_errors`. gaia names final replies cut by
  a budget other than its 4096 (with `--thinking` the run's budget: run_gaia.py's reasoning_effort
  "high" and 8192 are keyed on its o1 model_id, so none is sent). In either mode, chartqa,
  docvqa, infographicvqa, textvqa, coco_caption, charxiv, mmmu_pro, medxpertqa (both turns),
  wildbench, omnidocbench, simpleqa, browsecomp, healthbench, humanitys_last_exam, frames,
  imo_answer_bench, aa_lcr, beam_128k, loft_scifact,
  screenspot, screenspot_pointing, ruler, i18n_translate, bundled_detection and putnam_formal count
  every reply cut off by the run's token budget rather than at an official cap
  (`replies_cut_off_by_budget`; `replies_cut_off_in_reasoning` for those with no answer after
  reasoning) and report such a run as not comparable. copilot_bench_swe, swe_bench_multilingual,
  swe_bench_live and multi_swe_bench (agentic) send the run's `--max-output-tokens` as max_tokens
  with every agent request (none with `GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0`, as upstream), count
  model turns cut off at that limit (`truncated_turns`) and report such a run as not comparable; turns
  that stopped with reasoning but no content and no tool call or action are classified by token
  ids: the answer inside the thinking channel (`answer_in_thinking`) is the model's outcome, counted
  and not a comparability reason; a native tool call the server's parser dropped
  (`tool_call_dropped_turns`) and a turn with no token ids to tell (`unclassified_empty_turns`) are
  counted and make the run not comparable (answer_in_thinking.md);
  swe_bench_live and multi_swe_bench count them in gbench's relay (swebench_common.EndpointRelay),
  per tab or run.
- Harness errors and withheld scores in `base.run_eval_suite` (every suite that runs through it;
  a suite may add its own on top): a request that got no reply from the served endpoint after
  base's retries (status FAILED: an endpoint or connection error, recorded as `HTTP <code>: ...`
  or the exception; TIMEOUT; OVER_CONTEXT: a prompt longer than the served context window), and
  a tool-loop request that failed after earlier turns in a loop that keeps no forced answer, is
  a harness error, never the model's outcome: its trace carries `harness_error` (`kind`
  endpoint_failure, endpoint_timeout, over_context or request_failed_mid_loop, and the error) and
  `scoring_excluded`, whatever a batch scorer made of it; it is left out of the accuracy's
  numerator and denominator and of the per-category and @k figures, still counted in
  `failed_requests` (with `timed_out_requests` / `over_context_prompts`), counted per kind in
  `request_harness_errors` and named in `status_reason`, and the status is `completed_with_errors`.
  A suite that computes its own headline, categories or @k from the traces leaves such a trace out
  the same way (`base.is_scored`, `metrics.scorable_traces`). Suites that send their own requests
  (mmlu, mmlu_pro, mmlu_redux, wmdp, scicode) mark a request with no reply the same way
  (`vqa_common.mark_no_reply`, `vqa_common.own_loop_status`): out of the score, counted in
  `scoring_excluded` and `request_harness_errors`, named in `status_reason`, withheld when nothing
  is scored. A spider2 grader crash (SCORING_ERROR) and a ui_control_osworld task without a
  result.txt are harness errors too.
  HTTP 400/422 and context-window rejections follow one rule (`base.attribute_endpoint_rejection`,
  listed per suite in `base.ENDPOINT_REJECTION_ATTRIBUTION`): where the benchmark's own harness
  defines the outcome of such a rejection, its semantics win (for example an agent loop whose
  history the model grew past the window fails the task, as upstream's loop does) and it is the
  model's outcome; otherwise a 400/422 on a request the harness built is a harness error, never the
  model's outcome. Per suite:
  - base (`_send_single_request` statuses: every suite that runs through `run_eval_suite`, including
    a tool-loop request that fails mid-loop), bfcl, bfcl_live and bfcl_v4_agentic
    (`ENDPOINT_FAILURE_KINDS`), copilot_bench_swe, swe_bench_multilingual and multi_swe_bench
    (`swebench_common.ENDPOINT_EXCEPTIONS` lists BadRequestError and UnprocessableEntityError) and
    swe_bench_pro (the endpoint failures gbench's relay counts): harness error; a context-window
    rejection is `over_context` in base and bfcl;
  - swe_bench_live and terminal_bench: a 400/422 is a harness error
    (`swebench_common.ENDPOINT_EXCEPTIONS`); a history the model grew past the window follows the
    agent's own handling (SWE-agent's exit_context, Terminus-2's ContextLengthExceededError), the
    model's outcome;
  - mcp_bench counts every endpoint or connection error of a model request as a harness error
    (`model_replies.failed_requests`); an over-context request is left to MCP-Bench's own handling
    (`rejected_over_context`);
  - gaia2 (ARE's agent loop takes the rejection of a request carrying the model's own conversation
    into the agent's history and goes on), toolbench (StableToolBench's chat_completion_request
    turns it into an error reply the DFSDT loop goes on with; `model_content_rejections`) and the
    quality runner (`runners/quality.py` `_MODEL_CONTENT_REJECTION_STATUSES`): the upstream loop's
    semantics, the model's outcome; a gaia2 run in which no agent request got a reply is
    infra_required;
  - scicode: a context-window rejection is not retried; a sub-step prompt that carries the
    model's own earlier code fails that step and the steps after it, the problem staying in the
    score (upstream's gencode.py has no handling and test_generated_code.py scores against fixed
    denominators), the model's outcome (`over_context_steps`); a prompt built from the dataset
    alone that the window cannot hold is a harness error (`over_context`); any other
    400/422 is a harness error.
- The server-parser policy (`base.recover_server_parser_text`; every request base sends, and the
  reference for suites with their own model client). A served model runs behind the server's
  reasoning parser and tool-call parser (vLLM `--reasoning-parser` / `--tool-call-parser`, as the
  model's serving recipe sets them, e.g. `--reasoning-parser gemma4 --enable-auto-tool-choice
  --tool-call-parser gemma4`). On a request sent with thinking off, text the reasoning parser put
  in `reasoning` (gemma-4: a thought channel the model opened on its own, or output the parser
  reads as one) is the model's output: it is recovered in generation order and scored, from the
  reply's generated token ids (every request asks for them with `return_token_ids`; decoded with
  special tokens kept by the server's `/detokenize`, else the run's tokenizer, the stop token left
  out and a matched stop string cut as the server cuts it), else as the reasoning field followed
  by the content (`server_parser_moved_text`). A reply the tool-call parser deleted (no content,
  no reasoning and no tool call although the server generated tokens: a native tool call on a
  request that declared no tools) is recovered from its token ids and scored
  (`server_parser_dropped_text`); when the server returned no token ids or they cannot be decoded
  it is a harness error (`server_parser_text_lost`), left out of the score, counted in
  `request_harness_errors` and named in `status_reason`. Every event is counted per run
  (`server_parser_events`, `server_parser_samples`, `server_parser_note`) and each trace records its
  own (`server_parser`: event and `recovered_from`, token_ids or reasoning_then_content; the
  trace's `reasoning` keeps the server's field). A reply the server bound tool calls for, and a
  --thinking reply with reasoning (the reasoning channel is the model's reasoning there), are left
  as they are; a forced final answer whose tool call was deleted is asked again as before. A
  tool-call parser can also delete a native call from a reply that keeps content or reasoning (a
  reply sent with thinking off whose content is the text before the call, or a --thinking reply
  whose reasoning holds the thought and whose answer was the call): for every reply to a request
  that declared no tools, the returned reasoning, then the returned content, are located verbatim
  in the generation rebuilt from the token ids, and when the text after the last located segment
  holds a letter or digit outside tag markup (`<...>`), the reply is `server_parser_dropped_text`
  (`base.server_parser_residue`): the model's output is the generation after the reasoning on a
  --thinking reply, else the whole generation (without token ids nothing can be located, and the
  reply is left as it is). Server
  requirement: the served endpoint returns `token_ids` for `return_token_ids` (vLLM does; other
  servers ignore the field and the recovery falls back as above) and serves `/detokenize` next to
  `/tokenize`, or `--tokenizer` names the served model's tokenizer. An endpoint that refuses the
  return_token_ids field (HTTP 400/422 naming it) gets the request again without it, and the rest
  of the run omits it; the result records `server_parser_token_ids` {available: false}.
  A reply that stopped with its answer inside the thinking channel (reasoning, no content, no tool
  call: health `empty_reasoned`) is the model's outcome, scored as an empty answer, not a failed
  request, and counted in `answer_in_thinking` / `answer_in_thinking_rate` (answer_in_thinking.md).
  An empty reply the server returned (EMPTY_RESPONSE) is the model's outcome, scored wrong unless
  the benchmark's own scorer scores it (bfcl irrelevance; cyberseceval, where the ICD finds nothing
  in it: secure), in which case it is an answered sample (`empty_is_answer`), not a failed request;
  a sample the protocol's wall clock stopped stays the model's outcome, scored wrong. A sample a suite leaves out of the score
  (`scoring_excluded`: a grader failure, a sample it cannot measure) also makes the status
  `completed_with_errors` (`scoring_excluded N` in the reason; a judge failure, judge_grade
  judge_error, unparsed or JUDGE_FAILED, is named once, under `judge failures N, of which M left out
  of the score`), unless the suite marks it `excluded_by_protocol` (an exclusion the benchmark's own
  metric makes). When nothing is scored (every sample a harness error, a judge outage or left out of
  the score, in any mix) the status is `failed`, `accuracy` is null and `accuracy_withheld_reason`
  names the causes, never 0. pass@k counts only scored attempts; pass^k counts only the samples
  with every attempt scored (`pass_hat_k_samples`), and the samples with fewer are counted in
  `samples_with_fewer_scored_attempts` (null when none). A suite's own follow-up request that got no
  reply (a MathArena last-chance reprompt) is a request harness error named in the status, not a
  suite exclusion. A reply that ran to the cap made almost entirely of one repeated character (a
  run of spaces) is `looping`, not `truncated`. The run
  summary's Effective N is the suite's own `effective_n` (its headline's denominator) when it
  records one, else, for a run with one trace per question, the number of scored samples, else the
  question count. A judge call every Gemini model refused (a block of the prompt, reported in the
  cascade's `refused` and `block_reasons`) is not retried. A tool-loop suite's traces keep the
  loop's turns after the prompt (`tool_loop_transcript`: the tool calls with their arguments and
  the tool replies).
- Renamed suites still read their old temperature variable as a deprecated alias (the new name wins
  when both are set): `GBENCH_SEMANTIC_KEYPOINT_TEMPERATURE` (screenspot_pointing),
  `GBENCH_LMSYS_NONCODING_HARD_TEMPERATURE` (wildbench), `GBENCH_BFCL_V3_LIVE_TEMPERATURE`
  (bfcl_live), `GBENCH_LOFT_X_ARXIV_TEMPERATURE` (loft_scifact), `GBENCH_CULER_TEMPERATURE`
  (longbench_v2_code) and `GBENCH_NEXUS_FUNCTION_CALLING_TEMPERATURE` (nexusraven_api_eval).
- Standard third party variables are read directly and are not gbench knobs: `GEMINI_API_KEY`
  (and `GEMINI_API_KEYS`, a comma/semicolon key list that wins over `GEMINI_API_KEY` when both are
  set; `GEMINI_OPENAI_BASE_URL`, the Gemini API's OpenAI-compatible endpoint, default
  `https://generativelanguage.googleapis.com/v1beta/openai/`: the key's live check pings it and the
  judge proxies of some suites call it. Its only accepted values are the Gemini API over https (host
  `generativelanguage.googleapis.com`) and a loopback address (`localhost`, `127.0.0.0/8` or `::1`,
  over http or https: a local stub or relay, for tests); any other value, plain http to the Gemini
  API or a URL with a user name or password included, is refused and never contacted: the live check
  rejects the key naming the variable, and a suite that hands the variable to a container or proxy
  hard-errors (infra_required) whatever a skip-key-validation knob says (gaia2, wildclawbench,
  tau2 and tau3). The host side reads it through `base.gemini_openai_base_url()`
  (`base.require_gemini_openai_base_url()` raises the hard error); tau2/tau3's image never gets it:
  gbench's Gemini relay on the host reads it through the same helper.
  gbench's own judge client (google-genai) always calls the Gemini API: google-genai's
  `GOOGLE_GEMINI_BASE_URL` and `GOOGLE_GENAI_USE_VERTEXAI` do not move it), `OPENAI_API_KEY`,
  `HF_TOKEN` /
  `HUGGING_FACE_HUB_TOKEN`, `XDG_CACHE_HOME`,
  and the MCP-Atlas server credentials, passed only into mcp_atlas's credentialed-servers
  container, never into the container the model's shell and code tools run in (`AIRTABLE_API_KEY`,
  `ALCHEMY_API_KEY`, `E2B_API_KEY`, `GITHUB_TOKEN`, `GOOGLE_MAPS_API_KEY`,
  `EXA_API_KEY`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REFRESH_TOKEN`,
  `LARA_ACCESS_KEY_ID`, `LARA_ACCESS_KEY_SECRET`, `MONGODB_CONNECTION_STRING`, `NPS_API_KEY`,
  `NOTION_TOKEN`, `OXYLABS_USERNAME`, `OXYLABS_PASSWORD`, `SLACK_MCP_XOXC_TOKEN`,
  `SLACK_MCP_XOXD_TOKEN`, `TWELVE_DATA_API_KEY`, `WEATHER_API_KEY`;
  mcp_atlas reads no `BRAVE_API_KEY` (brave-search does not run): the search tools of its four
  search-engine servers are served by Gemini Google Search grounding, and their page tools run as
  upstream on the release's servers, exa's and oxylabs' with `EXA_API_KEY` and
  `OXYLABS_USERNAME` / `OXYLABS_PASSWORD`). No suite reads a SerpAPI, Brave or OpenAI
  embedding key: every model role a harness has goes through the Gemini API (see the Gemini scope
  below). No knob points a judge,
  search or relay of gbench elsewhere: `GBENCH_WILDCLAWBENCH_JUDGE_ENDPOINT`,
  `GBENCH_GAIA2_JUDGE_ENDPOINT` and `GBENCH_GAIA2_JUDGE_MODEL` are removed (setting one logs a
  warning and changes nothing).
- Gemini scope: model roles only. The Gemini API (the one cascade, `GBENCH_JUDGE_MODELS`) takes the
  place of the model roles a harness has: judges, graders and raters, labelers, user and API
  simulators, web search / grounding, and embeddings (tau3's retrieval, `gemini-embedding-001`).
  A benchmark's non-model tool backends run as upstream runs them, with upstream's own keys and
  services; a missing key or an unreachable backend is `infra_required` for the suite (or for the
  setting that needs it), never a Gemini substitution. The suites with such backends, and what
  each needs (each suite's doc has the details and the checks):
  - `toolbench`: StableToolBench's real-API proxy (`toolbench_url`
    `http://8.130.32.149:8080/rapidapi` in the server config, with the README's empty toolbench
    key), reachable from the container (checked before the run); calls it does not answer go to
    the API simulator, a model role on the cascade.
  - `api_bank`: the Translate API through `googletrans`: upstream's pinned `googletrans==3.1.0a0`
    runs in the local image `gbench-api-bank:googletrans-3.1.0a0` (Docker; network access from
    docker's bridge network to `translate.googleapis.com`; `HTTP_PROXY` / `HTTPS_PROXY` /
    `ALL_PROXY` / `NO_PROXY` are honored, and a proxy URL with a password is refused); no key. A
    Google reply other than HTTP 200 is an exception, and a Translate step whose gold call fails the
    same way is a `backend_unavailable` harness error (from a Google Cloud VM, Google answered HTTP
    429 on 2026-09-27).
  - `complexfuncbench`: upstream's response_based call-match tier queries the live Booking.com API on
    RapidAPI (`booking-com15`) with `RAPID_API_KEY` (the key of an account subscribed to
    `booking-com15`). If it is unset or empty, or a live check before the run refuses it, the
    suite is `infra_required`. A pair the API could not answer that the judge then finds not
    equivalent is a `rapidapi_unavailable` harness error.
  - `mcp_bench`: the servers' own APIs: `GOOGLE_MAPS_API_KEY`, `NPS_API_KEY`, `NASA_API_KEY`,
    `NCI_API_KEY` and `HF_TOKEN` for the five credentialed servers (run in a container of their
    own), and the keyless public APIs the other servers call; each of the five unset or empty is
    `infra_required` before any container starts; each set key is also sent once to its service
    before any container starts (Google Maps once per Maps Platform API its server calls), and a
    refused key, or one with no answer after 3 attempts, is `infra_required`
    ([mcp_bench.md](mcp_bench.md#checking-the-keys)).
  - `mcp_atlas`: the release environment's servers with the release's `env.template` credentials
    (`AIRTABLE_API_KEY`, `ALCHEMY_API_KEY`, `E2B_API_KEY`, `EXA_API_KEY`, `GITHUB_TOKEN`,
    `GOOGLE_MAPS_API_KEY`, `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` / `GOOGLE_REFRESH_TOKEN`,
    `LARA_ACCESS_KEY_ID` / `LARA_ACCESS_KEY_SECRET`, `MONGODB_CONNECTION_STRING` (optional),
    `NPS_API_KEY`, `NOTION_TOKEN`, `OXYLABS_USERNAME` / `OXYLABS_PASSWORD`,
    `SLACK_MCP_XOXC_TOKEN` / `SLACK_MCP_XOXD_TOKEN`, `TWELVE_DATA_API_KEY`, `WEATHER_API_KEY`),
    passed only to its credentialed-servers container; the search tools of its four search-engine
    servers are served through the Gemini API (search is a model role), and their page tools run
    as upstream with those keys (see [mcp_atlas.md](mcp_atlas.md)); a page-tool reply saying Exa
    or Oxylabs could not be reached, or refused the run's account (Exa HTTP 401/402/429 or its
    stdio rate-limit text; Oxylabs' reply with no JSON body, which is how its API refuses
    credentials and how a provider-side error page reads, its error reply with no job record
    (`Error: 'job'`), or HTTP 401/403/407/429), is a harness
    error (`harness_errors.provider_account_error_calls`), never the model's outcome; a
    provider-side HTTP 5xx of those page tools (Exa's `web_fetch_exa error (503):
    SERVICE_OVERLOADED` and 500/502/504, an Oxylabs `HTTP error during POST request: 5xx - ...`)
    is one too (`harness_errors.provider_unavailable_calls`); every
    variable except the
    optional `MONGODB_CONNECTION_STRING` is required: an unset one is `infra_required` before any
    container starts, naming it per server.
- Credential values never reach what gbench writes: the value of every credential variable
  (`GEMINI_API_KEY`, `GEMINI_API_KEYS`, `OPENAI_API_KEY`, `HF_TOKEN`,
  `HUGGING_FACE_HUB_TOKEN`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `VLLM_API_KEY`,
  `GBENCH_MODEL_API_KEY`, `GBENCH_TAU_AGENT_KEY`, `GBENCH_TOOLBENCH_SIMULATOR_KEY`,
  `TB_SIMULATOR_KEY`, `MONGODB_CONNECTION_STRING`, and any variable whose name ends in
  `_API_KEY(S)`, `_TOKEN(S)`, `_SECRET(S)`, `_SECRET_KEY`, `_ACCESS_KEY`, `_PASSWORD`, `_PASSWD` or
  `_CREDENTIAL(S)`, and each member of such a variable's comma, semicolon or space separated list),
  the password of a `scheme://user:password@host` URL in any variable (a user name without a
  password is left alone), the token in huggingface_hub's token file (`HF_TOKEN_PATH`, else
  `$HF_HOME/token`, else `$XDG_CACHE_HOME/huggingface/token`, else `~/.cache/huggingface/token`) and
  the GCP identity tokens gbench fetches at run time (`GBENCH_GCP_AUTH`) is replaced by
  `[REDACTED]`, as written, JSON-escaped or URL-encoded. Placeholders (`EMPTY`, `dummy`,
  `placeholder`, `not-needed`, `sk-no-key-required` and the others in `PLACEHOLDER_SECRET_VALUES`),
  values under 8 characters and all-digit values are not treated as credentials. It covers results,
  traces, summaries, metadata, CSV files, log lines and every other file under the results
  directory: gbench's writers redact the final text they write, after serialization
  (`write_redacted_json` / `write_redacted_csv` / `write_redacted_text` in
  `gbench/utils/log_manager.py`), every file a suite, a harness or a subprocess wrote there itself
  is rewritten after each suite and at the end of the run (also when the run stops on an error or
  Ctrl-C; `redact_tree`: symlinks are not followed, a file that cannot be rewritten is logged by
  name, a file is rewritten in place and keeps what another process appends meanwhile, then is
  checked again so a key appended during the rewrite is replaced too, and a file modified in the
  last minute is not shortened: the rewritten text is padded with newlines to its old length, so a
  process still writing it continues after the text), and every log record is redacted as it is
  created (`install_log_redaction`; the redaction code is in `gbench/runners/eval_suites/base.py`).

## Cross cutting knobs

### Serving and run infrastructure
- `GBENCH_RELAY_IDLE_TIMEOUT_S` (default 300): an EndpointRelay request that receives nothing for
  this long while the server's engines stood still (token counters and KV-cache usage) fails as
  `stalled_no_tokens` ([relay.md](../relay.md)). `GBENCH_RELAY_EXPECTED_SHAPE` (JSON): the request
  shape every relay checks and records (`shape_violations`); `gbench --smoke` sets it.
- `GBENCH_JUDGE_REQUEST_TIMEOUT_S` (default 600): the per-request timeout of gbench's Gemini
  judge client (the same model is retried once before the cascade falls back).
  `GBENCH_MODEL_REQUEST_TIMEOUT_S` is not an operator setting: gbench sets it inside the
  skillsbench orchestrator, for the deepagents agent's model client, to `base.request_timeout_s`
  of the served context window (a host value is ignored). The agent suites' model-client and relay
  timeouts are not knobs either: `swe_agentic.agent_request_timeouts` sizes them with
  `base.request_timeout_s` (see `GBENCH_REQUEST_TIMEOUT_S`).
- `GBENCH_SKIP_CALIBRATION_GUARD=1`: bypasses the pre-run guard that refuses a `--sandboxes`
  value that would over-subscribe the machine's CPU/RAM, or the free Docker networks when
  terminal_bench, swe_bench_pro or swe_lancer is selected ([docker-sandboxes.md](docker-sandboxes.md)).
- `gbench --preflight` / `--smoke` / `--parallel-suites` ([smoke.md](../smoke.md)):
  `GBENCH_PREFLIGHT_MIN_DISK_GB` (100), `GBENCH_PREFLIGHT_TIMEOUT_S` (300),
  `GBENCH_POOL_MAX_INFLIGHT` (48; 0 turns the admission gate off), `GBENCH_POOL_STALL_S` (600; 0
  turns the stall alarm off), `GBENCH_POOL_DURATIONS_FILE` (default
  `<gbench cache dir>/pool_durations.json`),
  `GBENCH_SMOKE_HANG_S` (300), `GBENCH_SMOKE_DEADLINE_S` (3900). Set by gbench for each suite
  process, never by you: `GBENCH_SMOKE`, `GBENCH_SMOKE_RELAY_REPLIES`, `GBENCH_POOL_RECORD`,
  `GBENCH_POOL_ACTIVITY_FILE`, `GBENCH_REUSE_ROOT`.
- `GBENCH_SERVER_PORT`: the port of the vLLM server gbench launches (`config.resolve_server_port`).
  Set: that port, and a port another process listens on is an error naming it (never another
  port); unset: the first free port from 8000 (a port counts as free when nothing on 127.0.0.1
  accepts a connection and binding all interfaces succeeds). Once the server answers /health,
  gbench checks the listener is its own child and that /v1/models lists the model it launched. A
  local run launches one server per model for the quality, Golden Set and eval pillars together
  (`config.local_endpoint`, the base URL each of them targets), after the performance servers.
  An automatically chosen port that another process takes before the server binds it is retried
  on the next free port (up to 3 attempts, `serving.PORT_RACE_ATTEMPTS`); an explicit
  `GBENCH_SERVER_PORT` never is.
- `GBENCH_SERVER_HOST`: the bind address (`--host`) of every server gbench launches. Unset:
  `127.0.0.1` for the performance (serving, stress) servers and for the quality / Golden Set
  server of a run that selects no eval suite (quality and the Golden Set reach it from gbench's
  own process); all interfaces for the server of a run that selects eval suites, where the suites
  whose containers run on docker's default bridge reach it at the bridge gateway (`172.17.0.1`).
  It must be a loopback address (127.0.0.0/8, ::1, localhost) or all interfaces (0.0.0.0, ::); any
  other value is refused before a server starts (gbench reaches its server at 127.0.0.1). The
  address and the reason are recorded in `result['server']['host']` and
  `result['server']['host_source']` of performance, eval, quality and Golden Set results, and
  metadata.json records each native server (`native_servers`).
- `GBENCH_VLLM_BIN`: replaces the default vLLM command `<python> -m vllm.entrypoints.cli.main`
  (gbench's own interpreter) of every server gbench launches (shell-split). Results record
  `result['server']['vllm']` and `run_fingerprint['vllm']` (`cli`, `cli_source`,
  `package_version`, `package_path`). A `GBENCH_VLLM_BIN` server may run outside gbench's process
  tree, so it is checked by its model listing only (no process-tree ownership check).
- Performance results record `server` (endpoint, port, port_source, host, host_source,
  endpoint_key_enforced (never the key), gpus, tensor / data parallel size, serve_settings, vllm) and `served_model_id` / `served_model_id_source`. MM results
  record `per_image_tokens` / `per_image_tokens_source` (measured by the preflight on the server;
  `processor` = the model's HF processor for offline throughput; `config` / `processor config` =
  declared; `unknown`).
- Offline throughput applies only the engine arguments of `--vllm-serve-args` (vLLM's
  AsyncEngineArgs parser) and records `vllm_serve_args_applied` / `vllm_serve_args_not_applied`;
  a data-parallel run measures one tensor-parallel replica (`run_fingerprint`
  `data_parallel_size` 1, `run_data_parallel_size` N), because vLLM's offline benchmark supports
  data parallelism only under an external launcher.
- `--eval-max-soft-tokens N` is passed as `--mm-processor-kwargs {"max_soft_tokens": N}`, on the
  eval server only.
- Model-specific serve settings (gbench names no model): `--tool-call-parser NAME` (the eval,
  quality and Golden Set server gets `--enable-auto-tool-choice --tool-call-parser NAME`; required
  for a local run of a suite in `eval_suites.TOOL_CALLING_SUITES`, of the quality pillar or of the
  Golden Set's tool cases), `--reasoning-parser NAME` (the same server gets `--reasoning-parser
  NAME`), and `--vllm-serve-args '...'` or `GBENCH_VLLM_SERVE_ARGS` (the flag wins): extra `vllm
  serve` arguments appended last to every server gbench launches (`config.serve_args_for`), with
  the arguments gbench owns refused (`config.GBENCH_OWNED_SERVE_ARGS`: port, host, context,
  parallelism, memory, logprobs, the parser flags, the scheduler and prefix-caching flags, the
  served model name, credentials, a config file, and abbreviations of them). All three are
  recorded in the scaffold and metadata.json (`serve_settings`), and refused with
  `--remote-endpoint`.
- GPUs of a local server: `--gpus` (indices or UUIDs), else the first idle GPUs (no compute process,
  at least 80% of memory free; `config.select_gpus`) among those `CUDA_VISIBLE_DEVICES` exposes.
  The server gets `CUDA_VISIBLE_DEVICES=<uuids>` and `CUDA_DEVICE_ORDER=PCI_BUS_ID`
  (`config.cuda_visible_devices_env`); metadata.json records `selected_gpus`. `--num-gpus N
  --tensor-parallel M` serves N / M data-parallel replicas (`config.parallel_layout`).
- Endpoint key: `VLLM_API_KEY`, else `OPENAI_API_KEY` (`base.endpoint_api_key`; placeholder values
  such as `EMPTY` are ignored), sent as `Authorization: Bearer` by gbench's own process only: every
  request of a `run_eval_suite` suite, the `/tokenize` and `/detokenize` calls, the Golden Set and
  quality pillars, gbench's endpoint relays, which add it host-side for container suites, the
  serving and stress pillars' requests (remote or to the server gbench launched; the local text
  benchmark through vLLM's `extra_headers`, since vLLM's benchmark client otherwise sends only
  `OPENAI_API_KEY`), the multimodal preflight, the endpoint preflight (a 401/403 on `/v1/models`
  is a hard failure, never passed by `/health`), the served-model listing and the multimodal
  probe. The server gbench launches gets `VLLM_API_KEY` stripped of surrounding whitespace, and a
  placeholder `VLLM_API_KEY` is removed from its environment, so it then enforces no key
  (`server.endpoint_key_enforced` records which).
- `GBENCH_SHARD` (`I/N`): the run's shard when `--shard` is not given; a malformed value is an
  error. `base.current_shard` reads it; see docs/distributed-gbench.md for the shard identity
  fields and `--merge`.
- The eval and quality servers gbench starts are given
  `--max-logprobs 100` (vLLM's default is 20), because mmlu sends hendrycks/test evaluate.py's
  request with logprobs=100; the cap only validates requests. A server passed with
  `--remote-endpoint` needs it for mmlu too.
- `GBENCH_SANDBOX` (`required` default, `bwrap`, `none`): process isolation for the suites that
  execute model-written code (codeforces, cruxeval, lcb, scicode): a bubblewrap jail with a
  read-only root, private `/tmp` and `/run` (no Docker socket), no network, new PID, IPC and UTS
  namespaces with a `/proc` of its own, no capabilities, and credential files hidden: the home
  directory (`$HOME`, and the account's home when it differs) is replaced by an empty one into which
  only the interpreter's directories and site-packages (when they live there), the Hugging Face hub
  cache and the file `GBENCH_SCICODE_TEST_DATA` names are bound back read-only (for scicode also the
  vendored `scicode` package its test cases import, `gbench/third_party/_scicode_vendor`) (a
  path reached through a symlink inside the home is bound at both its target and where the path
  lands in the
  emptied home, also when the home itself is reached through a symlink), and the credential files
  the jail would otherwise show are hidden. A home that holds the interpreter (it is
  one of its prefixes or lies inside one) cannot be emptied; it stays, with a warning, and only its
  credential files are hidden. The hidden credential files are huggingface_hub's token files,
  `$GOOGLE_APPLICATION_CREDENTIALS`, `$CLOUDSDK_CONFIG`, `$DOCKER_CONFIG`,
  `$AWS_SHARED_CREDENTIALS_FILE`, `$AWS_CONFIG_FILE`, `$KUBECONFIG`, `$NETRC`, `$BOTO_CONFIG`,
  `$AZURE_CONFIG_DIR`, `$GH_CONFIG_DIR`, and `gcloud` and `gh` under `$XDG_CONFIG_HOME` (a symlink
  is hidden by its target, and a credential under a directory bound at a second path is hidden at
  both). In every mode, `none` included, the
  program is given an environment without the credential variables listed above and without any
  variable whose value carries a credential; with `none` nothing else is isolated. acebench's
  evaluator and nestful's Win Rate worker run in containers of their own whatever this says
  (nestful's: no network, only PATH and HOME=/tmp in its environment, only the interpreter and
  the reference functions, read-only; without a working bubblewrap the Win Rate alone is
  infra_required; a Win Rate call that does not win after a failed use of the network is a harness
  error, left out of the Win Rate (win_rate_network_blocked_calls), completed_with_errors, not
  comparable).
- Cloud metadata server guard (no knob turns it off): `require_metadata_blocked(eval_name, image,
  network, docs_url=None, *, resolve=())` (`gbench/runners/eval_suites/container_guard.py`) is the
  check a suite runs before it starts a container that runs model-controlled code with network
  access (a docker bridge, the host network, a user-defined network or `container:<name>`). It
  starts one short-lived container of the suite's own local image on that network (`--pull
  never`, read-only, no capabilities) that only opens and closes a TCP connection to
  `169.254.169.254:80`, and to `[fd20:ce::254]:80` (no IPv6 route there is "unreachable", not an
  error): no request is sent, so no token is read. The probe program is the image's `python3`,
  else its `bash` (`/dev/tcp`) under coreutils `timeout`. When a connection opens, the suite
  hard-errors (infra_required) naming the host rule
  `sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP` (for IPv6,
  `sudo ip6tables -I DOCKER-USER -d fd20:ce::254/128 -p tcp ! --dport 53 -j DROP`) and how to make
  it persistent (for example
  `sudo apt-get install -y iptables-persistent && sudo netfilter-persistent save`). The
  rule spares port 53 because on GCE the VPC DNS resolver is the same address (a plain `-j DROP`
  cuts DNS in every container). With `resolve` (the host names the suite's containers need), the
  same container also resolves each name, and one that does not resolve is a hard error saying
  that containers on that network cannot resolve host names and naming the rule above; a name the
  resolver answers does not exist (getaddrinfo's EAI_NONAME or EAI_NODATA, nslookup's NXDOMAIN)
  is a hard error asking to check the URL or host name instead, since DNS itself answered. Docker's
  DOCKER-USER chain does not see the host network, so on a cloud VM a container on the host
  network hard-errors until it runs on a bridge or user-defined network instead (dropping the
  address for the whole VM would also cut the VM's own DNS, guest agent and credentials). A probe
  that gives no clear verdict (neither `python3` nor `bash` with `timeout` in the image, no
  resolver program for a name it has to check, a docker error, a docker timeout) is a hard error
  too. `--network none` needs no probe. Each network's verdict, and each name that resolved there,
  is cached for the rest of the run. gBench never runs sudo or changes the host's firewall itself.
  The suites that call it: `mcp_atlas`, `mcp_bench` and `wildclawbench`; `bfcl` (multi-turn
  part), `bfcl_v4_agentic` (`bfcl generate`), `tau2`, `tau3`, `livebench`, `aider_polyglot` and
  `bigcodebench`, on docker's bridge network; `quality` (`--network none`, no probe); and the SWE
  family (`copilot_bench_swe`, `swe_bench_multilingual`, `swe_bench_live`, `multi_swe_bench`,
  `swe_bench_pro`), `terminal_bench`, `skillsbench`, `swe_lancer` and `ui_control_osworld`, through
  `swebench_common.require_task_network_blocked` before their task containers start. That one
  probes docker's default bridge and a user-defined bridge network gbench creates for the probe
  (`gbench-metadata-probe-<id>`, removed right after), because their harnesses also create their
  task networks at run time. The probe never pulls an image: without one of the suite's images on
  the host the gate hard-errors naming the `docker pull` or build to run. `terminal_bench` and
  `skillsbench` pass `resolve` with their verifiers' package hosts
  (`swebench_common.TASK_INSTALL_HOSTS`), `swe_bench_live`'s agentic Linux tabs with SWE-ReX's
  build hosts (`SWEREX_BUILD_HOSTS`), `swe_bench_pro` with the hosts its sandboxes install the
  agent from (`swe_bench_pro.SANDBOX_INSTALL_HOSTS`), and `multi_swe_bench` with its builds'
  package hosts for the run's languages (`multi_swe_bench.runtime_hosts`). `swe_bench_live`'s
  Windows tab runs only after its own PowerShell probe on the Windows-containers engine
  (`swebench_common.check_windows_task_network`, in one of the tab's instance images already on
  the host, to `169.254.169.254:80` and `[fd20:ce::254]:80`) found the metadata server
  unreachable, and is reported `infra_required` otherwise. `gaia2` and `toolbench` run their
  containers (which run the model's tool calls) on docker's bridge network after the guard (neither
  resolves a name: gaia2 reads its datasets offline from the image), with gbench's endpoint and
  Gemini relays (and gaia2's token-id decoder, DetokenizeRelay) on the gateway and only the relay's
  placeholder inside; `ojbench`, `multipl_e`, `omnidocbench` (its evaluator compiles the model's
  LaTeX), `putnam_formal` and `bfcl_live` run the model's
  code with `--network none` (see
  [toolchains.md](toolchains.md#cloud-vms-the-metadata-server)).
- `GBENCH_RESULTS_DIR`: base directory for result artifacts (set by the run manager).
- quality: needs `GEMINI_API_KEYS` or `GEMINI_API_KEY`, git and Docker.
  `GBENCH_QUALITY_GIT_CONFIG_GLOBAL` (default `/dev/null`) is the git global config of its host-side
  git commands (clone, fetch, rev-parse, archive). The GemmaClaw QA suite's alternate model
  `google/gemini-3.1-pro-preview`, image-generation model `google/gemini-3.1-flash-image-preview`
  and memory embeddings `google/gemini-embedding-001` run on the Gemini API. The suite runs in
  gbench's own Docker container (built by gbench from `docker/quality.Dockerfile`: no network, a
  read-only checkout, a user without capabilities, no host path writable) and gets no key: its
  environment is a fixed set of variables with nothing of gbench's, its `VLLM_API_KEY` is the
  placeholder `gbench-relay`, and a forwarder in the container hands its calls to a relay in gbench
  over a unix socket; the relay alone holds the Gemini keys and `VLLM_API_KEY` (a remote endpoint's
  key) and forwards only the Gemini API calls the suite makes. `GBENCH_GEMINI_RELAY_BASE_URL` is
  set by gbench for the suite (not a knob), and `GBENCH_SANDBOX` does not apply (the suite always
  runs in the container). See docs/quality.md.
- `GBENCH_EVAL_PLUGINS_PATH`: path searched for eval plugins (same as `--eval-plugins-dir`).
- `GBENCH_SWE_OMP_THREADS`: OpenMP/BLAS thread cap applied inside the SWE-family agent containers and
  harness containers, including the task containers the harness images create (a mounted shim) and
  SWE-bench Pro V2's agent and verifier environments.
- `GBENCH_CACHE_DIR` (default `~/.cache/gbench`): root of every gbench cache. bfcl and bfcl_live
  keep the pinned BFCL data under `bfcl_data/<commit>/`, and the BFCL harness project root
  defaults to `bfcl/` under it; gorilla_apibench (all four question settings), acebench
  (`acebench/<commit>/data_en/` and `data_zh/`) and imo_answer_bench cache their pinned files under
  it; toolbench caches the published SoWR reference answers it converted under
  `toolbench_baselines/<revision>/converted/<reference>/<group>.json`; putnam and putnam_formal cache the pinned PutnamBench release under
  `putnambench/<commit>/` and putnam_formal its compiled references under `putnam_formal_ref/`;
  loft_scifact keeps the LOFT bundle under `loft/`, ruler its generated data under `ruler/`, and
  multi_swe_bench its repository clones under `multi_swe_bench/repos/` (unless
  `GBENCH_MULTI_SWE_REPO_DIR` is set). cimemories caches its raw labeler replies under
  `cimemories_labels/<labeler key>/<prompt sha256>.json` (up to ~120 MB), keyed by the judge
  cascade and temperature, and reads its release prompt and persona files from
  `cimemories_release/<commit>/<path>` (sha256-checked; about 12 KB, downloaded on first use).
  arc_agi caches the ARC-AGI-1 evaluation files it verified under `arc_agi/<commit>/evaluation/`.
  The SWE-bench-schema suites (copilot_bench_swe,
  swe_bench_multilingual, swe_bench_live) write their pinned dataset snapshot to
  `swebench_snapshots/<dataset>@<revision>/<split>.jsonl` under it (`/` in the dataset id becomes
  `__`); the same file feeds the rollout and the harness. The quality pillar's GemmaClaw checkout
  (`gemmaclaw/`) and the default ShareGPT file also live under it.
- `GBENCH_PREREQS_DIR` (default `~/.cache/gbench/gbench-prereqs`): shared root for suites that need a
  local checkout / data dir. When a suite's own dir var is unset, it falls back to
  `$GBENCH_PREREQS_DIR/<subdir>` **if that path exists** (else it hard-errors as before - the fallback
  never masks a missing prereq). Provision the tree once and set this, instead of exporting a per-suite
  var each time. Subdir per suite: `codeforces` (GBENCH_CODEFORCES_GENERATED_TESTS),
  `NESTFUL/data_v2/executable_functions` and
  `NESTFUL/data_v2/nestful_data.jsonl` (nestful),
  `OJBench_testdata` (ojbench), `RULER` (ruler), `Spider2/spider2-lite[/evaluation_suite/gold |
  /resource/databases/spider2-localdb]` (spider2), `SWE-bench_Pro-os` (swe_bench_pro,
  maintained-engine route only),
  `WildClawBench` (wildclawbench), `aider-bench` (aider_polyglot),
  `apibank/api-bank` (api_bank). An explicit per-suite var always overrides the fallback.
- `SHAREGPT_PATH`: path to the ShareGPT dataset for the `--campaign chat-like` serving workload; if
  unset and the default copy (`$GBENCH_CACHE_DIR/ShareGPT_V3_unfiltered_cleaned_split.json`) is
  missing, it is downloaded just-in-time.
- `GBENCH_GCP_AUTH` (default unset; `1`, `true` or `yes` enables it): opt in to automatic GCP
  identity-token auth when no `VLLM_API_KEY` / `OPENAI_API_KEY` is set. The Golden Set fetches a
  token with `gcloud auth print-identity-token` for any remote endpoint (never for `localhost`,
  `127.0.0.1` or `0.0.0.0`); the quality pillar does it for a `*.run.app` / `*.googleapis.com`
  endpoint (metadata server, then the gcloud fallback). Unset, gbench makes no metadata-server or
  `gcloud` call: pass `VLLM_API_KEY` instead. Only the golden and quality pillars send the token.
  The CLI's endpoint check at the start of every remote run sends none, so an endpoint that
  refuses unauthenticated calls (an IAM-protected Cloud Run service) stops the run there.

Stress pillar knobs (all of them resolve **flag > `GBENCH_STRESS_*` env > default**):
- `GBENCH_STRESS_CLIENT_PROCS` (flag `--stress-client-procs`; default `min(32, cpu_count-1)`): OS
  processes for the multi-process load generator, which removes the single-asyncio-client CPU
  confound at the knee. **Auto-capped to `cpu_count-1`**, so a large value is safe on small hosts
  (it won't spawn more procs than cores). Raise it alongside `GBENCH_STRESS_MAX_QPS` so the
  client can actually offer the higher load; if it can't, those points are flagged `client_bound`
  and excluded rather than reported as a server capacity result.
- `GBENCH_STRESS_REPS` (flag `--stress-reps`; default 3): sweeps per stress run. Each gives a
  bracket [pass, fail]; the median repeat's limit is certified (`GBENCH_STRESS_CERTIFY`).
- `GBENCH_STRESS_MAX_QPS` (flag `--stress-max-qps`; default 8192): safety cap on the open-loop
  arrival-rate sweep. A server that meets the SLO at the cap has its knee censored there rather
  than measured; raise it (with `--stress-client-procs`) for such a server. The `client_bound`
  guard and the `--stress-client-procs` auto-cap protect weak clients and small hosts.
- `GBENCH_STRESS_MAX_PROMPTS` (flag `--stress-max-prompts`; default: no cap): optional, recorded
  cap on requests per rate point. Without it a point sends max(40, rate x 60 s, rate x
  (3 x latency + 10 s)) requests, never fewer.
- `GBENCH_STRESS_MIN_SAMPLES` (flag `--stress-min-samples`; default 15): minimum requests in a
  point's cohort (those planned after its ramp-up) to judge it. Lowering it only makes the
  bracket sweep cheaper; when certification runs, the reported limit is certified with
  percentile-sized probes either way.
- `GBENCH_STRESS_CERTIFY` (flag `--stress-certify`; default `auto`): certify the reported stress
  limit with percentile-sized probes (a cohort of >= 1099 requests for the P99 SLO, about
  1400 sent) inside the sweep's bracket,
  to +-9% when reached (else a certified lower bound with its resolution, a lower bound censored
  at the rate cap, or no limit when certification refutes the sweep); `always` ignores the budget,
  `off` reports the bracket labelled UNCERTIFIED.
- `GBENCH_STRESS_CERTIFY_BUDGET_MIN` (flag `--stress-certify-budget`; default 120): minutes the
  certification may take, per stress pass (text and multimodal each); 0 = do not certify (in `auto` mode; `always` ignores the
  budget). The
  budget is checked before each probe: if it runs out after a probe passed, that rate is
  reported as a certified lower bound; if none passed, the bracket is reported UNCERTIFIED with
  the time certifying would need.
- `GBENCH_PERF_SEED` (default 83): seed of the perf request content (lengths, base tokens, base
  images, the stress schedule). A fresh salt per run still makes every request new to the
  server's caches; the result records both.
- `GBENCH_PERF_LOCK_WAIT_S` (default 0): how long a performance pillar waits for the endpoint lock
  (other gbench runs on the same server) before failing with the holders named.
- `GBENCH_LOCK_DIR` (default `<GBENCH_CACHE_DIR>/endpoint-locks`): where the host-wide endpoint
  locks live. `GBENCH_ENDPOINT_LOCK_KEY` is set by gbench for its children; do not set it. Use one
  URL per server: host aliases, round-robin pools and container-IP vs published-port are not unified.

### Decoding and scoring (apply to every native suite, defined in base.py)
These change headline numbers and their comparability, so record them when you set them.
- `GBENCH_AGENT_MAX_TOKENS_FROM_RUN` (default on; `0` turns it off): in the agent suites
  (`swe_agentic.AGENT_MAX_TOKENS_SUITES`: the SWE suites, terminal_bench, skillsbench,
  wildclawbench, gaia2, tau2, tau3, toolbench, agent_dojo, swe_lancer, ui_control_osworld,
  bfcl_v4_agentic, aider_polyglot), every agent request that carries no `max_tokens` /
  `max_completion_tokens` is sent the run's `--max-output-tokens`, never a fixed number (the
  user's decision of 2026-10-11: one runaway turn otherwise runs to the served window). gbench's
  relay fills it in. agent_dojo sets it in its client, copilot_bench_swe / swe_bench_multilingual
  in mini-swe-agent's `model_kwargs`, and swe_bench_live / swe_bench_pro / multi_swe_bench as a
  relay body field. gaia2, toolbench, ui_control_osworld and wildclawbench (OpenClaw, 8192) send
  their own `max_tokens`, so nothing is filled there. A run whose relay filled requests records
  `agent_max_tokens_from_run` and a `leaderboard_comparable` reason; tau2, tau3 and
  bfcl_v4_agentic also write `official_sampling.agent_max_tokens_from_run`. The body-field and
  model_kwargs suites (swe_bench_live, swe_bench_pro, multi_swe_bench, copilot_bench_swe,
  swe_bench_multilingual) record `official_sampling.agent_max_tokens_from_run` and the reason
  from the run's value whenever it is sent, whether or not the relay filled anything. Their
  relay's `requests_filled` is usually 0 (the value goes out as a relay body field, or in the
  agent's own request); for copilot_bench_swe and swe_bench_multilingual it can be above 0,
  because the mini-swe-agent 2.x driver resends a turn the server refused for its cap with no
  `max_tokens`, and the relay then fills the run's value.
- `GBENCH_RELAY_STREAM_UPSTREAM` (default off): EndpointRelay requests the endpoint with
  `stream=true` and rebuilds the non-streamed reply; replies can differ in rare fields (see
  [relay.md](../relay.md)).
- `GBENCH_REPETITION_PENALTY`, `GBENCH_FREQUENCY_PENALTY`, `GBENCH_PRESENCE_PENALTY`: decoding
  penalties added to every request. Unset by default (no penalty). `repetition_penalty` and
  `min_p` are allowed request fields from this release on, so `GBENCH_REPETITION_PENALTY` now
  reaches the server; earlier results that recorded it under `decoding_penalties` did not apply it.
- `GBENCH_IMAGE_SOFT_TOKENS` (default 1120): per image soft token budget used to estimate multimodal
  prompt length for context clamping. See also the roadmap note on per model derivation. A result of
  `run_eval_suite` whose samples carry media records the budgets used (`media_soft_token_budget`).
- `GBENCH_AUDIO_SOFT_TOKENS` (default 750): per audio part (`input_audio` / `audio_url`) soft token
  budget used the same way, instead of counting the audio's base64 as text.
- `GBENCH_VIDEO_SOFT_TOKENS` (default 2496, gemma-4's maximum per video in vLLM: 32 frames x (70
  + 2 + 6)): per video part (`video_url`) soft token budget used the same way; the video is not
  sent to /tokenize.
- `GBENCH_REQUEST_TIMEOUT_S` (default 1200): the floor and fixed part of every request timeout
  gbench sizes: `base.request_timeout_s(budget)` = max(floor, floor + budget /
  `GBENCH_MIN_DECODE_TOK_S`) seconds. The eval client passes the request's token budget (8192
  when none is given). The agent relays and agent model clients
  (`swe_agentic.agent_request_timeouts`) and complexfuncbench pass the max_tokens sent, else the
  served context window, else 262,144 (`base.UNCAPPED_TIMEOUT_TOKENS`); skillsbench passes the
  served context window, else 262,144.
- `GBENCH_MIN_DECODE_TOK_S` (default 8): the per-sequence decode rate that timeout assumes. The
  mcp_bench and toolbench images use the same formula for a `--thinking` budget. A suite whose
  median measured decode rate falls below it logs a warning. It is not a stall detector.
- `GBENCH_NON_CONVERGENT_RATIO` (default 4.0; `0` disables): threshold for flagging a non
  convergent generation.
- `GBENCH_RETRY_DISCARDED_FINAL` (default 1; `0` disables): whether to retry a discarded final
  turn.
- `GBENCH_TOOL_LOOP_MAX_OUTPUT_TOKENS` (default 8192; `0` disables): per turn output ceiling
  inside the house tool loop (plugins
  and suites without their own protocol or an official cap); a result it cut records
  `tool_loop_turn_ceiling` ({applied, requested}) and is not leaderboard-comparable; gaia follows
  its reference agent and sends max_tokens 4096 on every turn (with --thinking 8192, or
  --max-output-tokens when larger);
  deepsearch_qa and mcp_atlas send no max_tokens (the starter notebook and the harness send none),
  so neither this ceiling nor --max-output-tokens applies to them.
- `GBENCH_<SUITE>_TEMPERATURE`: per suite temperature override for any suite (highest precedence;
  with `--temperature`, the only way to sample a suite at anything but the rule's 1.0 with
  `--thinking` or 0.0 without it).
- Run-shaping knobs and the scaffold: every env variable that changes what a run measures is
  declared in `base.RUN_SHAPING_GLOBAL_ENV` (the penalties, the media soft-token budgets, the
  tool-loop ceiling, the discarded-final retry, the Gemini cascade knobs, the web_search result
  count) or `base.RUN_SHAPING_SUITE_ENV` (per suite: protocol modes, splits, versions, languages,
  categories, task and query selections, trials, runs, samples, seeds, windows, sampling and
  budget overrides, agent and parser choices, simulator and judge models, judge-input limits and
  scoring timeouts), plus `GBENCH_<SUITE>_TEMPERATURE` for every suite.
  `base.run_shaping_env(suites)` snapshots the ones that are set, by class (dataset, prompt,
  decode, scoring), for the evals scaffold and the result's `run_shape`, so two runs that differ
  in one get different scaffold ids and `--skip-existing` does not reuse across them; a
  deprecated alias (`base.RUN_SHAPING_ALIASES`, e.g. `TAU2_NUM_TRIALS`) is recorded under its
  canonical name, and `TAU2_TEMPERATURE` (a deprecated alias of `GBENCH_TAU2_TEMPERATURE` /
  `GBENCH_TAU3_TEMPERATURE`, `base.RUN_SHAPING_TEMPERATURE_ALIASES`) moves the run-shaping
  snapshot like its canonical name. Every other `GBENCH_*` variable a suite reads is listed in
  `base.NON_RUN_SHAPING_ENV` with the reason it changes no measurement (infra, credential,
  diagnostic, internal, removed); a test fails on a suite variable in neither, so a new knob
  gets a decision when it is added.

### Judge cascade (every Gemini text role: judges, graders, labelers, user simulators, search)
- The default cascade is `base.GEMINI_CASCADE`: `gemini-3.8-flash`, `gemini-3.7-flash`,
  `gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-3.1-pro-preview`, tried in order. It is the one
  list for every Gemini text role: LLM judges, graders and raters, labelers (cimemories), user
  simulators (tau2 / tau3 user simulator and NL judge, acebench's agent_multi_turn user simulator,
  toolbench's API simulator), web search / grounding (unless `GBENCH_SEARCH_MODELS` is set), and
  the judges inside the gaia2, wildclawbench, mcp_bench and toolbench containers. Code that cannot
  import gbench gets the run's list at run time as `GBENCH_JUDGE_MODELS` and
  `GBENCH_SEARCH_MODELS` (`base.gemini_cascade_env()`), so it calls exactly the models gbench
  records. Not used by the non-text roles: tau3's retrieval embeddings (`gemini-embedding-001`)
  and the quality pillar's pinned helper models.
- `GBENCH_JUDGE_MODELS`: comma separated cascade that replaces the default (tried in order).
- `GBENCH_JUDGE_MODEL`: a one-model cascade, read only when `GBENCH_JUDGE_MODELS` is unset or
  empty (`GBENCH_JUDGE_MODELS` wins when both are set; gbench logs which one is active).
- Model check: the Gemini key gate of every judged suite (`base.gemini_key_live_valid`, which
  `base.gemini_required_skip` calls) asks the API, with the first key it accepts, for each model of
  the judge and search cascades (`GET <GEMINI_OPENAI_BASE_URL>/models/<name>`, once per model per
  process). A model it answers 404 for (or 403: the key may not use it) is infra_required naming
  the model and the variable to fix; an inconclusive answer (no network, 429, 5xx) never blocks and
  is asked again at the next gate. During a run, a model that answers 404 is not asked again in
  that call; any other failure moves to the next model.
- Results record which Gemini model answered: `gemini_models_answered` ({model: replies}) and, on
  every result of `base.run_eval_suite`, the configured cascade (`gemini_cascade`,
  `gemini_cascade_source`), and per verdict where the suite records it. A suite whose Gemini gate
  ran (`gemini_required_skip`) or whose result names an answering model also records
  `self_judge` (a cascade model is the model under test: `base.judge_provenance`, a
  model-agnostic name comparison); a self-judged result is graded and flagged, not refused:
  `leaderboard_comparable: false` with the reason "judge cascade contains the model under
  test" (`base.finalize_central_stamps` re-applies it after the suite returns).
- Gate timing: the key gate lists the models (no generation quota), so an invalid or expired key
  and an unserved model fail fast, while an exhausted quota surfaces at run time as
  `JUDGE_OUTAGE` samples (harness errors; a run whose every sample is one is failed with its
  score withheld).
- `GBENCH_JUDGE_CASCADE_ROUNDS` (default 3, at least 1): retry rounds across the cascade before a
  judge outage.
- `GBENCH_JUDGE_BACKOFF` (default 1.0 s): backoff seconds between judge retries.
- `GBENCH_JUDGE_REQUEST_TIMEOUT_S` (default `600`): how long one request of gbench's own judge
  client (`base.judge_generate_cascade`, google-genai) may take. Until 2026-10-11 the client had no
  timeout, so a request the API never answered held its sample, and the cascade, indefinitely. A
  request past the timeout is asked once more of the same model, then the cascade moves to its next
  model. Each fallback is counted with its reason (`base.judge_fallback_counts()`, `{"<model>:
  <reason>": n}`: the error class, with the HTTP code when there is one, and `same-model retry after
  timeout` for each timeout asked again); complexfuncbench records its run's counts as
  `judge_fallbacks` and the client settings as `judge_client`. The timeout bounds the judge only,
  never the model under test (`base.judge_client_record()` describes it). It is set on the shared
  judge clients (`base._judge_clients`), so the acebench user simulator and the toolbench ToolEval
  judge, which call those clients directly in their own cascade loops, get it too; their loops do
  not yet ask a timed-out model again or count the fallback (a timeout moves straight to their
  next cascade model, as any other error does).
- `GBENCH_JUDGE_MODELS` / `GBENCH_JUDGE_MODEL` also select the cimemories labeler and key its label
  cache, so changing them rebuilds the labels (14,700 calls).
- The judge runs at temperature 0 (`judge_config` in base.py) unless a suite follows its
  upstream grader's own settings: simpleqa, healthbench and browsecomp grade as simple-evals'
  grading_sampler (temperature 0.5, max_output_tokens 2048; recorded in `judge_sampling`),
  wildbench's judge sends src/eval.py's temperature 0 and max_tokens 1024 as max_output_tokens,
  toolbench at ToolEval's 0.2, acebench's user simulator at APIUSER's 0.001,
  mcp_bench's in-image judge, humanitys_last_exam's judge (run_judge_results.py) and aa_lcr's
  equality checker send no temperature, and deepsearch_qa's rater gets no generation config.
- toolbench's ToolEval calls run at ToolEval's temperature 0.2 and max_tokens 1000 (sent as
  max_output_tokens; replies cut there without the call are counted in
  judge_replies_cut_at_max_tokens_by_group) with its config.yaml function (name, description,
  parameters) forced through Gemini function calling; toolbench's cache-miss API simulator also
  runs on the cascade (models, rounds, backoff, key rotation) in place of gpt-4-turbo; mcp_bench's in-image judge sends MCP-Bench's request (model, messages, max_tokens; no
  temperature) through gbench's Gemini relay, which holds the key and rotates GEMINI_API_KEYS (the
  container gets only the relay's placeholder); deepsearch_qa's rater is called with no generation config, as the
  Kaggle notebook's rater is.
- acebench's agent_multi_turn user simulator runs at APIModel_user.APIUSER's own sampling:
  temperature 0.001, top_p 1 and max_tokens 1000 (sent as max_output_tokens); only the model
  (gpt-4o) is replaced. Replies that stop at that limit are counted
  (`user_simulator.replies_cut_at_max_tokens` with text, `empty_replies_at_max_tokens` without, when
  the cascade asks its next model) and make the run not leaderboard-comparable.

### Web search cascade (gaia, deepsearch_qa, bfcl web_search, mcp_atlas and wildclawbench search, ...)
- `GBENCH_SEARCH_MODELS` / `GBENCH_SEARCH_MODEL`: grounding model cascade / single model, for search
  only. Default: the judge cascade above (`GBENCH_JUDGE_MODELS`, else `GBENCH_JUDGE_MODEL`, else
  `base.GEMINI_CASCADE`). A model the API answers 404 for is not retried in place and is not asked
  again in that lookup; a 429 moves to the next model; other errors are retried in place
  (`GBENCH_SEARCH_ATTEMPTS`).
- `GBENCH_SEARCH_MAX_RESULTS` (default 8): results returned per query (not for BFCL's web_search,
  whose own `max_results` argument cuts the results, as upstream; mcp_atlas: its search tools take
  the call's own count (its page tools run on the release's servers); wildclawbench's web_search
  takes the Brave request's `count`).
- `GBENCH_SEARCH_ATTEMPTS` (default 3), `GBENCH_SEARCH_BACKOFF` (default 1.0 s),
  `GBENCH_SEARCH_CASCADE_ROUNDS` (default 3, at least 1): retry shape.
- `GBENCH_SEARCH_HEDGE_DELAY_S` (default 12): for a lookup with a deadline (wildclawbench's agent
  web_search, mcp_atlas' search tools), the seconds a grounding request may run unanswered before
  the next cascade model is asked in parallel (a hedged request; the first successful answer is
  the lookup's, every hedge takes a rate-limiter token). 0 or less disables only the timed hedge:
  a further model is then asked only after a failure, but the first `GBENCH_SEARCH_INITIAL_WIDTH`
  models (default 2) are still asked at once at the start, so the models are asked strictly one
  after another only with `GBENCH_SEARCH_INITIAL_WIDTH=1` as well. Lookups without a deadline
  (gaia, deepsearch_qa, bfcl_v4_agentic) never hedge.
- A lookup with a deadline is a **search-engine lookup** (since 2026-10-12): the request asks
  `Search the web and list the most relevant sources for: <query>`, only an answer citing at least
  one grounding source counts (one citing none moves to the next model, and a later cascade
  round does not ask that model again, so a query no model cites a source for costs at most one
  request per cascade model, not one per model per round), and each result is the
  source's title, its real URL and an EMPTY snippet: no text a Gemini model wrote (no
  groundingSupports segment, no answer text, no grounded-answer fallback) reaches the agent.
  Measured 2026-10-12 on 5 recorded wildclawbench agent queries x 2 models: 2/10 answers within
  27 s with the bare query, 5/10 with the sources prompt, 4/5 queries with the sources prompt sent
  to both models at once; `thinkingLevel` low had no effect and is not set; `site:` and quoted
  queries missed 27 s in every variant. Lookups without a deadline are unchanged. The deadline is
  the caller's: wildclawbench's agent web_search gets 57 s since 2026-10-13 (it was 27 s; OpenClaw's
  web_search timeout there is raised to 60 s), mcp_atlas keeps its own.
- `GBENCH_SEARCH_INITIAL_WIDTH` (default 2, at least 1): for a lookup with a deadline, how many
  cascade models are asked at once when it starts (each request takes a rate-limiter token and
  grounding quota, also when it loses the race: a lookup costs at least this many requests; the
  hedge delay then applies to the rest). 1 asks one model first, as before 2026-10-12.
- `GBENCH_SEARCH_REDIRECT_TIMEOUT_S` (default 3): for a lookup with a deadline, the longest one
  grounding redirect link (`vertexaisearch.cloud.google.com/grounding-api-redirect/...`) may take
  to resolve to the source URL (one HEAD request, the `Location` header read, the source page not
  contacted; measured 0.08-0.13 s on 2026-10-12); the time left before the deadline lowers it. A
  link that does not resolve is returned as the redirect link and counted (`links_unresolved`),
  as is one whose `Location` is longer than 2048 characters. 0 or less resolves none. A value of
  this knob, `GBENCH_SEARCH_INITIAL_WIDTH` or `GBENCH_SEARCH_HEDGE_DELAY_S` that is not a number
  fails the lookup before any request is sent (an error naming the variable).
- `GBENCH_GEMINI_RELAY_URL`: not an operator setting. It is one of the variables
  `gemini_relay.GeminiRelay.client_env()` gives a container whose Gemini calls go through gbench's
  host-side key relay (`gbench/runners/eval_suites/gemini_relay.py`: the container holds a per-run
  placeholder as its Gemini key, and the relay adds the real one). With it set, the grounding
  search sends its requests to that relay, and only while its key is such a placeholder: with a
  real key it sends nothing and the lookup fails, naming the variable.
- `GBENCH_SEARCH_MAX_RPS` (default 8 per process), `GBENCH_SEARCH_MAX_BURST` (default int(rps), at
  least 1): client side rate limiting. Under `--parallel-suites` the pool runs at most k search
  suites at once (browsecomp, deepsearch_qa, gaia, bfcl, bfcl_v4_agentic, mcp_atlas,
  wildclawbench) and gives each `GBENCH_SEARCH_MAX_RPS`/k, where k = min(number of search suites
  selected, `--parallel-suites`, 4), fixed when the plan is built (not how many happen to run at
  a given moment); this is recorded as `search_max_rps` in the suite's pool record.

## Per suite knobs

Each suite reads only the knobs listed under it. Names are canonical; where a bare legacy alias
exists it is noted. Full provisioning (checkouts, images, keys) is in each `docs/evals/<suite>.md`.

### Pillar 1: general knowledge and reasoning
- livebench: `GBENCH_LIVEBENCH_IMAGE` (default `gbench-livebench:0baa4aa7`, the local image tagged
  with the first 8 characters of LIVEBENCH_COMMIT; never pulled), `GBENCH_LIVEBENCH_MODEL`,
  `GBENCH_LIVEBENCH_RELEASE` (default `2024-11-25`, LIVEBENCH_RELEASE; `latest` leaves the release
  to LiveBench, unpinned and not comparable), `GBENCH_LIVEBENCH_CATEGORIES`,
  `GBENCH_LIVEBENCH_EXTRA_ARGS` (a `--max-tokens` there overrides
  gbench's budget; it is written into the one-off model config when the served model's LiveBench
  config carries the cap, and a value other than the token-limit policy's is not comparable).
  Before the run the suite looks the served model up in the image's LiveBench model configs
  (docker run --network none); `GBENCH_LIVEBENCH_MODEL` is the first name tried, then
  `--tokenizer`, each also as its last path component. A `GBENCH_LIVEBENCH_MODEL` other than the
  served id gbench passed is checked against the endpoint's /v1/models listing (infra_required
  when it is not served) and is the id LiveBench's requests carry, recorded as request_model_id /
  request_model_id_source.
- cyberseceval: no path knob; Meta's Insecure Code Detector and the prompts come from the local
  image `gbench-cyberseceval:114d02388213-semgrep-1.51.0` (see docs/evals/cyberseceval.md).
  `GBENCH_CYBERSECEVAL_ICD` and `CYBERSECEVAL_ICD_RULES` (a host CodeShield checkout) are removed;
  a run that sets one logs a warning to unset it.
### Pillar 2: mathematics and proofs
- aime / hmmt: `GBENCH_MATHARENA_IMAGE` (default `gbench-matharena:b89f2f0a`; the local MathArena grader
  image, see docs/evals/hmmt.md).
- putnam_formal: `GBENCH_PUTNAM_FORMAL_LEAN_IMAGE`, `GBENCH_PUTNAM_FORMAL_COMPILE_TIMEOUT_S` (legacy
  aliases `PUTNAM_LEAN_IMAGE`, `PUTNAM_COMPILE_TIMEOUT_S`; the Lean/Mathlib toolchain image and the
  per proof compile timeout (default 300 s), both of which affect the verified score),
  `GBENCH_PUTNAM_FORMAL_CHECK_TIMEOUT_S` (default 900, caps the trusted checker and, separately,
  the answer printer that prints a no-answer reply for the judge review; an answer the printer
  cannot print within it or its 8 GB is not reviewed and not scored) and
  `GBENCH_PUTNAM_FORMAL_LEAN_PROJECT` (default /opt/putnamenv). The default image is the local
  `gbench-putnam-formal:97b5e945` built from docker/putnam_formal.Dockerfile (never pulled); the
  startup check requires its toolchain, Mathlib rev and PutnamBench commit to match the pins.
  Its Lean no answer answer review uses the gbench Gemini judge cascade (GEMINI_API_KEY /
  GEMINI_API_KEYS; the judge cascade knobs above apply), and the suite hard-errors without a key.

### Pillar 3: coding and software engineering
- scicode: `GBENCH_SCICODE_TEST_DATA`, `GBENCH_SCICODE_TEST_DATA_REPO`, `GBENCH_SCICODE_EVAL_TIMEOUT_S`
  (default 1800), `GBENCH_SCICODE_HTTP_TIMEOUT_S`, `GBENCH_SCICODE_THREADS` (default 1),
  `GBENCH_SCICODE_WITH_BACKGROUND` (the
  first four have legacy bare `SCICODE_` aliases). GBENCH_SCICODE_HTTP_TIMEOUT_S defaults to
  base's request timeout for a reply the length of the served context window (the suite sends no
  max_tokens). The default mirror is read at a pinned commit
  and its sha256 is checked (a mismatch hard-errors); a GBENCH_SCICODE_TEST_DATA file or another
  GBENCH_SCICODE_TEST_DATA_REPO whose digest differs makes the run not leaderboard-comparable.
- lcb: `GBENCH_LCB_VERSION` (default `release_v6`), `GBENCH_LCB_MIN_DATE`, `GBENCH_LCB_INCLUDE_TEST_GEN` (release tag,
  contamination cutoff, and `GBENCH_LCB_INCLUDE_TEST_GEN=0` to leave test output prediction out (it
  runs by default); legacy alias `LCB_INCLUDE_TEST_GEN`), `GBENCH_LCB_INCLUDE_SELF_REPAIR` (`0`
  leaves the self-repair scenario out; it runs by default); leaving either scenario out makes the
  run not leaderboard-comparable.
- bigcodebench: `GBENCH_BIGCODEBENCH_SUBSET` (`full` default: also reports the Hard figures; `hard`:
  the 148 Hard tasks only), `GBENCH_BIGCODEBENCH_IMAGE` (default `gbench-bigcodebench:09dd993f`),
  `GBENCH_BIGCODEBENCH_CPUS` (CPUs the evaluate container may use, default 64: the harness
  TensorFlow tasks hang when it sees more than about 128; a value that is not a positive integer
  raises; the harness --parallel is capped at this count).
- ojbench: `GBENCH_OJBENCH_TESTDATA` (legacy alias `OJBENCH_TESTDATA`; path to the online judge test
  data, bind-mounted read-only into the judge container), `GBENCH_OJBENCH_SAMPLES` (candidates per
  problem, default 8 = the official protocol; `--attempt-count` above 8 raises it), `GBENCH_OJBENCH_IMAGE` (default
  `gbench-ojbench:5e94480b`; the locally-built judge image carrying ojbench + dmoj + PyPy3 + g++ on Py3.11),
  `GBENCH_OJBENCH_JUDGE_TIMEOUT_S` (host-side cap on the whole judge container; default
  `max(1200, 180*N)`), and `OJBENCH_GET_TIMEOUT_S` (set on the host and forwarded into the judge
  container; a positive integer, default 120 s; the inter-result poll of the dead-worker break that
  stops a crashed worker from hanging the judge loop; any other value stops the suite before
  generation).
- codeforces: `GBENCH_CODEFORCES_SPLIT` (default `test`), `GBENCH_CODEFORCES_CONFIG` (default
  `verifiable`),
  `GBENCH_CODEFORCES_GENERATED_TESTS`, `GBENCH_CODEFORCES_MAX_GENERATED_TESTS` (0 = no cap; a cap
  is recorded as max_generated_tests and named in leaderboard_comparable_reason),
  `GBENCH_CODEFORCES_TIME_MULTIPLIER` (default 3.0).
- aider_polyglot: `GBENCH_AIDER_IMAGE` (default `aider-benchmark:5dc9490b`, built locally, never
  pulled), `GBENCH_AIDER_MODEL`, `GBENCH_AIDER_EDIT_FORMAT`,
  `GBENCH_AIDER_LANGUAGES` (comma or space list, any case), `GBENCH_AIDER_BENCHMARK_DIR`,
  `GBENCH_AIDER_EXTRA_ARGS` (must not contain
  `--read-model-settings`: gbench owns it to carry `--thinking`). `GBENCH_AIDER_MODEL` must be
  `openai/<id>`; an id other than the served id gbench passed is checked against the endpoint's
  /v1/models listing (infra_required when it is not served), and the result records
  request_model_id / request_model_id_source.
- multipl_e: `GBENCH_MULTIPL_E_IMAGE` (default `gbench-multipl-e:3025a531`, built locally, never
  pulled), `GBENCH_MULTIPL_E_LANGS`, `GBENCH_MULTIPL_E_SAMPLES`
  (completions per problem; default 20, the upstream chat recipe, or more if --attempt-count asks;
  a value that is not a positive integer raises), `GBENCH_MULTIPL_E_TEMPERATURE`; the pinned
  dataset_builder/chat_completions.py is fetched at run time (sha256-checked) and cached as
  `$GBENCH_CACHE_DIR/multipl_e/chat_completions-3025a531af74.py`.
- SWE family protocol: `GBENCH_COPILOT_BENCH_SWE_MODE`, `GBENCH_SWE_BENCH_MULTILINGUAL_MODE`,
  `GBENCH_SWE_BENCH_LIVE_MODE`, `GBENCH_MULTI_SWE_BENCH_MODE`, `GBENCH_SWE_BENCH_PRO_MODE` (`agentic`
  default; `oneshot` opt-in, never leaderboard-comparable; an invalid value is an error).
- SWE family trajectories: `GBENCH_COPILOT_BENCH_SWE_TRAJ_DIR`,
  `GBENCH_SWE_BENCH_MULTILINGUAL_TRAJ_DIR`, `GBENCH_SWE_BENCH_LIVE_TRAJ_DIR`,
  `GBENCH_MULTI_SWE_BENCH_TRAJ_DIR`, `GBENCH_SWE_BENCH_PRO_TRAJ_DIR` (parent directory for the
  agentic run's agent files: mini-swe-agent 2.x trajectories for copilot_bench_swe /
  swe_bench_multilingual, SWE-agent-for-eval's run-batch output and Win-Agent's run for
  swe_bench_live, MSWE-agent's per-language work directories for multi_swe_bench; for swe_bench_pro,
  the parent of the Harbor work directory (jobs/<job>/, harbor_<job>.log); default `<run results
  dir>/agent_trajectories/`, the system temp dir outside a gbench run).
- swe_bench_live Windows tab: `GBENCH_SWE_BENCH_LIVE_WIN_AGENT_DIR` (checkout of
  njukenanli/Win-Agent at 47d85cff) and `GBENCH_SWE_BENCH_LIVE_HARNESS_DIR` (checkout of
  microsoft/SWE-bench-Live at 62d02936 with its launch submodule), read only on a Windows Docker
  host; on any other host a Windows tab selected with other tabs is reported infra_required in
  settings.windows while they run.
- SWE family images: `GBENCH_MINI_SWE_AGENT_V2_IMAGE` (default `gbench-mini-swe-agent-v2:2.4.2-2.4.6`;
  copilot_bench_swe / swe_bench_multilingual rollouts), `GBENCH_SWEBENCH_V5_IMAGE` (default
  `gbench-swebench-v5:5.0.2`; their scoring), `GBENCH_SWE_BENCH_PRO_IMAGE` (default
  `gbench-swe-bench-pro-v2:harbor-0.22.0-66f92766-sidecar-rt`), `GBENCH_SWE_BENCH_LIVE_IMAGE` (default
  `gbench-swe-bench-live:a145aa87`; swe_bench_live's harness and scoring image, built locally from
  gbench/docker, never pulled; see docs/evals/swe_bench_live.md), `GBENCH_SWE_BENCH_LIVE_AGENT_IMAGE` (default
  `gbench-sweagent-for-eval:82422422-rex3`; swe_bench_live's SWE-agent-for-eval rollouts),
  `GBENCH_MULTI_SWE_BENCH_AGENT_IMAGE` (default `gbench-mswe-agent:88217624`; multi_swe_bench's MSWE-agent
  rollouts).
- swe_bench_pro: `GBENCH_SWE_BENCH_PRO_TASK_HOST` (the address gbench's relay listens on, default
  `172.17.0.1`: the V2 task sandboxes reach the endpoint only through it, it adds the endpoint
  key (VLLM_API_KEY, else OPENAI_API_KEY) when one is set, and the endpoint failures it counts
  are harness errors),
  `GBENCH_SWE_BENCH_PRO_TASK_ENDPOINT` (not used by the V2 locked protocol or the one-shot mode:
  the relay forwards to the endpoint gbench was given),
  `GBENCH_SWE_BENCH_PRO_MODE`; for the separately maintained engine only
  (`GBENCH_SWEBENCH_PRO_AGENTIC=1`, v1 harness): `GBENCH_SWE_BENCH_PRO_HARNESS_DIR`,
  `GBENCH_SWE_BENCH_PRO_RAW_SAMPLE` (a `.csv` or `.jsonl` table of the pinned v1 set, tag v1.0 /
  307bd006; checked against it, a mismatch is a prerequisite failure) (legacy bare
  `SWE_BENCH_PRO_` aliases); `GBENCH_SWEBENCH_PRO_AGENTIC=1` together with
  `GBENCH_SWE_BENCH_PRO_MODE=oneshot` is an error.
- multi_swe_bench: `GBENCH_MULTI_SWE_REPO_DIR` (repository clones; default
  `$GBENCH_CACHE_DIR/multi_swe_bench/repos`, else `$XDG_CACHE_HOME/gbench/multi_swe_bench/repos`).
- terminal_bench (opt-in deviations from the TB2.1 leaderboard's settings, none set by default,
  each named in leaderboard_comparable_reason): `GBENCH_TB_TIMEOUT_MULTIPLIER`,
  `GBENCH_TB_TURN_MAX_TOKENS`,
  `GBENCH_TB_CONTAINER_THREADS`, `GBENCH_TB_PARSER`, `GBENCH_TB_KEEP_JOBS_DIR`,
  `GBENCH_TB_SALVAGE_PATCH`. Trials: 5 per task (Harbor `-k 5`, the TB2.1 leaderboard minimum);
  an `--attempt-count` above 5 raises it, below keeps 5. Harbor runs in the local image
  `gbench-terminal-bench:harbor-0.20.0`, with gbench's Harbor task cache at
  `$GBENCH_CACHE_DIR/harbor`; of this host's environment the harness container gets only
  OPENAI_API_KEY=EMPTY, OPENAI_BASE_URL and, when set, the `HARBOR_*` variables (other than
  `HARBOR_TELEMETRY`: gbench always sets `HARBOR_TELEMETRY=0`, Harbor's telemetry off) and the
  proxy variables.
- `--attempt-count` above 1 is recorded as not applied (`attempts.not_applied`) by aider_polyglot,
  livebench, scicode, toolbench and skillsbench, whose harnesses run a fixed protocol.
- spider2: `GBENCH_SPIDER2_REPO_DIR`, `GBENCH_SPIDER2_GOLD_DIR`, `GBENCH_SPIDER2_LOCALDB_DIR` (legacy
  bare `SPIDER2_` aliases).
- swe_lancer (not in `--evals all`; runs when named): `GBENCH_SWE_LANCER_IMAGE` (the locally built
  harness image, default `gbench-swe-lancer-harness:51052ced`), `GBENCH_SWE_LANCER_TRAJ_DIR` (parent of its
  run directory, default `<run results dir>/agent_trajectories/`), `GBENCH_SWE_LANCER_TEMPERATURE`
  (overrides the rule's temperature, which the relay sets on every request).

### Pillar 4: long context and retrieval
- ruler: `GBENCH_RULER_DIR`, `GBENCH_RULER_TOKENIZER`, `GBENCH_RULER_LENGTHS`,
  `GBENCH_RULER_NUM_SAMPLES` (the NVIDIA RULER checkout, tokenizer, context bands, and per band count;
  default 500).
- mrcr: `GBENCH_MRCR_WINDOW` (tokens; overrides the served context window used to skip rows that do
  not fit; required when the server does not report max_model_len via GET /v1/models).
- beam_128k: `GBENCH_BEAM_128K_WINDOW` (tokens; overrides the served context window used to tail-prune
  BEAM conversations; required when the server does not report max_model_len via GET /v1/models;
  mirrors `GBENCH_MRCR_WINDOW`).
- longbench_v2_code: `GBENCH_LONGBENCH_V2_CODE_ALL_DOMAINS` (legacy aliases
  `GBENCH_CULER_ALL_DOMAINS`, `CULER_ALL_DOMAINS`; score all domains vs the default code subset),
  `GBENCH_LONGBENCH_V2_CODE_WINDOW` (legacy alias `GBENCH_CULER_WINDOW`; served context window in
  tokens; overrides the window read from GET /v1/models and is required when the server does not
  report one; LongBench-v2 prompts are middle-truncated to pred.py's max_len for the served
  window (120,000 up to a 131,072-token window, else the window - the setting's official cap - 64),
  or to window - the query's output budget - 64 when that is smaller).

### Pillar 5: tool use and agentic workflows
- acebench: `GBENCH_ACEBENCH_TEMPERATURE` (default 0.0 without `--thinking`, 1.0 with it;
  ACEBench's top_p 0.9 in both modes);
  agent_multi_turn's user simulator is the gbench Gemini judge cascade (`GEMINI_API_KEY` /
  `GEMINI_API_KEYS`, checked live before generating; the judge cascade knobs apply); without a
  usable key (unset or rejected) agent_multi_turn alone is reported infra_required
  (agent_multi_turn_category) and the other categories run (a hard error only when agent_multi_turn
  is all the run selected). BinOp arguments are evaluated with Python eval in a bubblewrap
  container of its own (`GBENCH_SANDBOX` does not apply, not even `none`; the suite hard-errors when
  bubblewrap is unavailable or the container is not isolated). Without --thinking every request of
  the model under test carries ACEBench's max_tokens=1024 (--max-output-tokens is not applied), or
  the room LLMInfer's 8192-token window (model_infer.py max_model_len) leaves after the prompt when
  that is less (8193 minus the prompt, as vLLM 0.6.1.post1 stops a sequence); a prompt longer than
  the window is not sent and its reply is empty, as that vLLM answers it; with --thinking the run's
  budget and no window.
- api_bank: `GBENCH_APIBANK_DIR`.
- bfcl: `GBENCH_BFCL_PROJECT_ROOT` (bfcl-eval project root for the agentic harness, mounted into
  the harness image at the same path, default `$GBENCH_CACHE_DIR/bfcl`, shared with
  bfcl_v4_agentic; upstream's own `BFCL_PROJECT_ROOT` is read as a legacy alias when it is unset;
  it must hold no `.env`, which upstream's `bfcl generate` / `bfcl evaluate` load into the process
  that runs the model's calls: one there is infra_required before anything is generated),
  `GBENCH_BFCL_IMAGE` (the harness image
  tag, default `gbench-bfcl:2026.3.23`; its labels must match the release),
  `GBENCH_BFCL_TIMEOUT_S` (legacy `BFCL_TIMEOUT_S`; unset by default, meaning no wall clock, as
  upstream; when set, seconds for each harness part on its own: the single-turn checker
  (bfcl_live's too), the multi-turn part, `bfcl generate` and `bfcl evaluate`; a part it stops
  is recorded under `harness_errors` and the rest of the run is kept),
  `GEMINI_API_KEY` / `GEMINI_API_KEYS` (the Web Search setting, web_search_base and
  web_search_no_snippet, served by Gemini Google Search grounding; without a key, with one the
  Gemini API rejects, or with a model of gbench's Gemini cascade the API does not serve to it, that
  setting is reported infra_required in `web_search_setting` and the other categories run); a real
  `OPENAI_API_KEY` reaches the multi-turn part and `bfcl generate` only
  through gbench's endpoint relay (bfcl.md, Containers and network); the agentic part reads the
  bfcl_v4_agentic knobs except its temperature: it generates at bfcl's temperature (0.0 without
  `--thinking`, 1.0 with it, or `GBENCH_BFCL_TEMPERATURE` or `--temperature`), and a
  differing
  `GBENCH_BFCL_V4_AGENTIC_TEMPERATURE` is ignored with a warning. `GBENCH_BFCL_ENABLE_THINKING`
  (bfcl and bfcl_v4_agentic) is set by gbench from `--thinking` for the harness container. Do not
  set it by hand: a host value of `1` reaches the container and turns thinking on in a run without
  `--thinking`.
- bfcl_v4_agentic: `GBENCH_BFCL_PROJECT_ROOT`, `GBENCH_BFCL_IMAGE`, `GBENCH_BFCL_TIMEOUT_S` (unset
  by default: no wall clock; when set, for `bfcl generate` and `bfcl evaluate` each),
  `GBENCH_BFCL_REUSE_GENERATIONS` (legacy bare `BFCL_` aliases; default project root
  `$GBENCH_CACHE_DIR/bfcl`; the bfcl_v4_agentic temperature defaults to 0.0 without
  `--thinking` and 1.0 with it).
  `GBENCH_BFCL_SEARCH_LOG` is set by gbench for the `bfcl generate` container that runs a
  web_search category (the web search log `<project root>/web_search_requests.jsonl`), the only
  job that also uses gbench's Gemini relay (it receives the relay's per-run placeholder as
  `GEMINI_API_KEYS` / `GEMINI_API_KEY` and its address as `GBENCH_GEMINI_RELAY_URL`, never the key)
  and the grounding-cascade knobs `GBENCH_SEARCH_MODELS`, `GBENCH_SEARCH_MODEL`, `GBENCH_SEARCH_MAX_RPS`,
  `GBENCH_SEARCH_MAX_BURST`, `GBENCH_SEARCH_CASCADE_ROUNDS`, `GBENCH_SEARCH_ATTEMPTS` and
  `GBENCH_SEARCH_BACKOFF` (BFCL's web_search tool is served by Gemini Google Search grounding;
  gbench itself sets `GBENCH_SEARCH_MODELS` and `GBENCH_JUDGE_MODELS` in that container to the
  resolved cascades (base.gemini_cascade_env; default base.GEMINI_CASCADE), so its searches ask
  exactly the models the Gemini relay forwards, and the result records the models that answered
  (`gemini_models_answered`)); do not set `GBENCH_BFCL_SEARCH_LOG` by hand.
  `GBENCH_BFCL_REQUEST_TIMEOUT_S` is set by gbench for the harness container (each request's
  timeout: base.request_timeout_s of the served context window, or of 262,144 tokens when the server
  reports none); do not set it by hand. `GBENCH_BFCL_GUARD_LOG` is set by gbench for every harness
  container of a `bfcl` command (the execution-guard log
  `<project root>/execution_guard_refusals.jsonl`); do not set it by hand. A real
  `OPENAI_API_KEY` never enters a BFCL harness container: gbench's endpoint relay on the docker
  bridge gateway adds it and the containers hold `EMPTY` (a placeholder key is passed as it is);
  `OPENAI_DEFAULT_HEADERS` is not passed. The project root must hold no `.env` (bfcl_v4_agentic.md,
  Credentials and the execution guard).
- mcp_atlas: `GBENCH_MCP_ATLAS_IMAGE` (agent-environment image tag, default `gbench-mcp-atlas:f24ba3fb`),
  `GBENCH_MCP_ATLAS_SANDBOX_URL` (use an agent environment that is already running in place of
  the environment container: it must be configured with no server that needs a credential and hold
  no credential variable, which gbench checks through its own code tool (the tool's environment
  and every process environment it can read, `/proc/1/environ` included; names only), gbench
  establishes its cloud metadata server verdict through its own code tool before the model starts
  and hard-errors without either verdict, and the credentialed servers and the MongoDB data still
  run in gbench's own container, so docker and the image are needed too),
  `GBENCH_MCP_ATLAS_MONGO_IMAGE` (the MongoDB sample-data image, default
  `gbench-mcp-atlas-mongo:f24ba3fb`, used when `MONGODB_CONNECTION_STRING` is unset).
- gaia: `GBENCH_GAIA_SPLIT` (`validation`, the default, scored locally; `test` writes the
  leaderboard submission JSONL under `<results dir>/gaia_submission/` and ends `infra_required`).
- nestful: `GBENCH_NESTFUL_FUNC_DIR` (data_v2/executable_functions of an IBM/NESTFUL checkout at
  fc2c4123e73500a56185a5fb354f05d1c8b4890c; the dataset is read from the same checkout's
  data_v2/nestful_data.jsonl, the only data source: without it the whole suite is infra_required,
  without the functions, or with functions that differ from the pinned revision, only the Win Rate
  is (win_rate_metric) and the other metrics run; a data file that differs from the pinned
  revision is a hard error (infra_required) either way; the Win Rate runs the model's calls in a
  bubblewrap sandbox of its own whatever GBENCH_SANDBOX says (no network: a call that does not win
  after a failed use of the network is a harness error, marked nestful_win_blocked_by_sandbox and
  left out of the Win Rate), and without a working bubblewrap only the Win Rate is
  infra_required).
- mcp_bench: `GBENCH_MCP_BENCH_IMAGE` (default `gbench-mcp-bench:7a8eaeae`, built locally, never
  pulled), `GBENCH_MCP_BENCH_TIMEOUT_S` (default 86400), `GBENCH_MCP_BENCH_DISTRACTION`,
  `GBENCH_MCP_BENCH_DISABLE_STABILITY` (a distraction count other than upstream's 10, or stability
  off, is named in `protocol_overrides` and the reason; a pre-run gap always ends the suite
  infra_required), `GBENCH_MCP_BENCH_ASSUME_NETWORK` (skips the launcher's network check, the name lookups of the
  servers' API hosts and a connection, in the pre-run check and in the run; MCP-Bench's connection
  check and gbench's own name check before the container starts still run),
  `GBENCH_MCP_BENCH_NETWORK` (the container's docker network, default `bridge`; the cloud metadata
  guard checks it before the pre-run check, and a model endpoint on this host, the judge's
  Gemini relay and the relay to the credentialed servers are reached at its gateway, 172.17.0.1
  on the default bridge). The five server keys (`GOOGLE_MAPS_API_KEY`, `NCI_API_KEY`, `HF_TOKEN`,
  `NPS_API_KEY`, `NASA_API_KEY`) go only to gbench's credentialed-servers container (a second
  container of the image, on a docker network of its own), never to the model container, whose
  entrypoint refuses one; the in-container judge gets the run's cascade as `GBENCH_JUDGE_MODELS`;
  a model endpoint URL with a user name or password is refused (infra_required); a
  credentialed-server session that fails outside the model (not opened, its server exiting
  before it answered, or lost while in use) makes the task running then a harness error
  (`credentialed_relay`), and a relay failure during the run or a credentialed-servers container
  not running at its end is a harness error of the run (status completed_with_errors); the
  image must carry the label `org.gbench.mcp-bench-harness=3`.
- toolbench: `GBENCH_TOOLBENCH_IMAGE` (default `gbench-toolbench:aa4ed9f4-v5`, built locally, never
  pulled), `GBENCH_TOOLBENCH_METHOD` (default `DFS_woFilter_w2`), `GBENCH_TOOLBENCH_GROUPS`,
  `GBENCH_TOOLBENCH_REFERENCE` (SoWR reference, default `chatgpt_cot`, a model of the published
  stabletoolbench/baselines answers), `GBENCH_TOOLBENCH_REFERENCE_DIR` (converted reference answers
  of your own, `<dir>/<reference>/<group>.json`, instead of the published baselines, which must
  cover every test id the run covers, else infra_required), `GBENCH_TOOLBENCH_EVAL_TIMES` (default
  3, StableToolBench's evaluation count), `GBENCH_TOOLBENCH_MAX_QUERY_COUNT` (per-task DFS search
  budget, canonical 200; distinct from `--eval-limit`, which caps the number of TASKS in total, spread over the groups).
  The ToolEval grader and the cache-miss API simulator run on the Gemini judge cascade only (the
  simulator on the Gemini API's OpenAI-compatible endpoint through gbench's Gemini relay; the
  container, which runs the model's tool calls on docker's bridge network, holds only the relay's
  placeholder, and gbench's endpoint relay adds the model endpoint's `OPENAI_API_KEY`), with
  `GEMINI_API_KEYS`, else `GEMINI_API_KEY`; no toolbench variable points either at another
  provider, base URL, key or model.
- complexfuncbench: `GBENCH_COMPLEXFUNCBENCH_MAX_ROUNDS` (default 40),
  `GBENCH_COMPLEXFUNCBENCH_TIMEOUT_S` (the timeout of each model request; default
  `base.request_timeout_s` of the turn's max_tokens (2048 without `--thinking`, the run's budget
  with it), or of the served context window when none is sent; a fixed 600 s before 2026-10-11;
  upstream's 5 attempts stay), `GBENCH_COMPLEXFUNCBENCH_EMBED_MODEL` (default
  `BAAI/bge-large-en-v1.5`), `GBENCH_COMPLEXFUNCBENCH_EMBED_DEVICE` (default `cpu`),
  `GBENCH_COMPLEXFUNCBENCH_RAPIDAPI_TIMEOUT_S` (the timeout
  of one RapidAPI request of the response-based tier, 60 s by default; upstream sets none).
- skillsbench: `GBENCH_SKILLSBENCH_IMAGE` (default `gbench-skillsbench:9a1f4dd5-h7`, built locally,
  never pulled), `GBENCH_SKILLSBENCH_AGENT` (default `deepagents`), `GBENCH_SKILLSBENCH_TASKS`,
  `GBENCH_SKILLSBENCH_CONDITIONS` (default `without-skills`), `GBENCH_SKILLSBENCH_TIMEOUT_S`
  (default 86400), `GBENCH_SKILLSBENCH_API_KEY`,
  `GBENCH_SKILLSBENCH_TASK_HOST`, `GBENCH_SKILLSBENCH_TASK_ENDPOINT`. `GBENCH_SKILLSBENCH_API_KEY`
  (the served endpoint's key; unset, `dummy` or `EMPTY` means none) never enters a container or a
  command line. The task containers reach the model through gbench's relay at
  `GBENCH_SKILLSBENCH_TASK_HOST` (default `172.17.0.1`), which adds the key, and
  `GBENCH_SKILLSBENCH_TASK_ENDPOINT` is then not used. The orchestrator and the agent get the
  placeholder `dummy` as `GBENCH_MODEL_API_KEY`, passed to docker by name.
- agent_dojo: `GBENCH_AGENTDOJO_ATTACK`, `GBENCH_AGENTDOJO_VERSION`, `GBENCH_AGENTDOJO_MODEL`,
  `GBENCH_AGENTDOJO_MODEL_LABEL` (a GBENCH_AGENTDOJO_VERSION, GBENCH_AGENTDOJO_ATTACK or
  GBENCH_AGENTDOJO_MODEL_LABEL other than the pinned v1.2.2 / important_instructions /
  vllm_parsed makes the run non-comparable; the label is recorded as attack_model_label).
  `GBENCH_AGENTDOJO_MODEL` defaults to the served id gbench passes; another value is checked
  against the endpoint's /v1/models listing (infra_required when it is not served). The id sent is
  recorded as request_model_id / request_model_id_source.
- t_eval: `GBENCH_T_EVAL_DEVICE` (`cpu` or `cuda`; the device for the Plan/Reason embedders, default
  GPU when it has room, else CPU; scores are the same).
- tau2 and tau3 (shared, canonical prefix `GBENCH_TAU2_`, legacy bare `TAU2_`):
  `GBENCH_TAU2_NUM_TRIALS`, `GBENCH_TAU2_MAX_STEPS`, `GBENCH_TAU2_MAX_ERRORS`, `GBENCH_TAU2_SEED`,
  `GBENCH_TAU2_TOP_P`, `GBENCH_TAU2_TOP_K`, `GBENCH_TAU2_USER_TEMPERATURE`, `GBENCH_TAU2_USER_LLM`,
  `GBENCH_TAU2_EVAL_LLM`, `GBENCH_TAU2_IMAGE`, `GBENCH_TAU2_SAVE_TRACES`,
  `GBENCH_TAU2_PROGRESS_SECS`, `GBENCH_TAU2_VERBOSE`, `GBENCH_TAU2_LLM_BACKOFF`,
  `GBENCH_TAU_SANDBOX_MAX_AGE_H`. Defaults: `GBENCH_TAU2_MAX_STEPS` 200, `GBENCH_TAU2_MAX_ERRORS` 10,
  `GBENCH_TAU2_SEED` 300, `GBENCH_TAU2_USER_TEMPERATURE` 0.0, `GBENCH_TAU2_PROGRESS_SECS` 30,
  `GBENCH_TAU_SANDBOX_MAX_AGE_H` 2 (0 disables). The
  assistant temperature follows the standard precedence via `GBENCH_TAU2_TEMPERATURE` /
  `GBENCH_TAU3_TEMPERATURE`. `GBENCH_TAU2_TOP_P` / `GBENCH_TAU2_TOP_K` are opt-in (unset by default:
  not sent, as upstream only sends temperature). The agent temperature defaults to 0.0 without
  `--thinking` and to 1.0 with it. `GBENCH_TAU2_NUM_TRIALS` defaults to 4, the leaderboard's trial
  count; when it is unset, an `--attempt-count` above 4 raises the trials to that value (recorded in
  `attempts`). `--shard` shards each domain's tasks before `--eval-limit`, whose N tasks are the suite's total, spread over the domains (the trials per task are unchanged). `GBENCH_TAU2_SAVE_TRACES` is the parent of the per-run directory
  `<eval>_<model>_<think|nothink>_<config hash>_<time>-<id>/` (empty = a temporary directory
  removed at the end). Both suites run in the local gbench-tau2 image (`GBENCH_TAU2_IMAGE`, default
  `gbench-tau2:b7ea9074`, docker/tau2.Dockerfile), working in `<gbench cache>/tau2`. Every
  `GBENCH_TAU2_*` / `GBENCH_TAU3_*` / `TAU2_*` knob (except credential-named ones, the removed
  knobs below, `GBENCH_TAU2_IMAGE` and `GBENCH_TAU2_SAVE_TRACES`), `GBENCH_TAU_SANDBOX_MAX_AGE_H`,
  `LITELLM_LOG` and the proxy variables (`HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY` and their
  lowercase forms) are passed into it by name, and the launcher adds the run's Gemini cascade as
  `GBENCH_JUDGE_MODELS` (`base.gemini_cascade_env()`): the user simulator and the NL judge try its
  models in order unless `GBENCH_TAU2_USER_LLM` / `GBENCH_TAU2_EVAL_LLM` name one `gemini/` model;
  `GBENCH_SEARCH_*` does not apply.
  The container runs on docker's bridge network, never the host network: a proxy variable on a
  loopback address is passed as `host.docker.internal`, `host.docker.internal` is added to
  `NO_PROXY` / `no_proxy` when a proxy is set, and an endpoint on this host is reached at
  `host.docker.internal` (see tau2.md, Container and network).
  No key goes in: every external model a tau run calls (the user simulator, the judge and tau3's
  retrieval embeddings) goes through the Gemini API by way of gbench's Gemini relay
  (`gemini_relay.GeminiRelay` on the docker bridge gateway), which adds `GEMINI_API_KEYS` /
  `GEMINI_API_KEY` on the host; the container gets only its per-run placeholder as
  `GEMINI_API_KEY` and its address as `GBENCH_GEMINI_RELAY_URL`. The key and every cascade model
  are checked on the host before the container starts. `GEMINI_OPENAI_BASE_URL` (accepted by
  `base.check_gemini_openai_base_url()`; any other value hard-errors before the container starts)
  is the relay's upstream and never goes in. A user simulator or judge that is not a `gemini/` model
  (an OpenAI,
  `anthropic/`, `openrouter/`, `vertex_ai/`, `bedrock/` or `azure/` model, or a bare `gemini-*`
  name, which LiteLLM routes to Vertex AI) hard-errors (infra_required) naming it before the
  container starts; inside the container a name tau2's LiteLLM routes elsewhere or cannot route
  hard-errors too. tau3 always runs upstream's alltools retrieval with its dense search on
  gemini-embedding-001; a Gemini embedding call that fails after its retries is a harness error
  (counted per domain in `harness_errors`, status completed_with_errors, named in
  leaderboard_comparable_reason), and so is a simulation that ends in an infrastructure error
  from a harness failure (a Gemini user simulator or judge outage, an endpoint outage, ...: a
  `harness_errors` entry per domain with its sources and error classes, left out of pass^k and
  the mean reward); one raised on the agent's own turn (a reply tau2 rejects, a request over the
  endpoint's context window) is left out the same way and also makes the status
  completed_with_errors, but is listed in `agent_turn_errors`, not as a harness failure. The
  host's `OPENAI_API_KEY` is the model endpoint's key: a real one stays with gbench's endpoint
  relay on the bridge gateway (`swebench_common.EndpointRelay`), which adds it to the agent's
  requests; a placeholder (EMPTY, dummy, ...) reaches the image only as the agent's key.
  `GBENCH_TAU2_SAVE_TRACES` is resolved on the host and
  mounted. Removed (setting one logs a warning; none is passed into the image):
  `GBENCH_TAU2_RETRIEVAL_CONFIG` / `TAU2_RETRIEVAL_CONFIG`, `GBENCH_TAU2_OPENAI_API_KEY`,
  `GBENCH_TAU2_OPENAI_BASE_URL`, `GBENCH_TAU2_EMBED_MODEL`, `GBENCH_TAU2_EMBED_BASE_URL`,
  `GBENCH_TAU2_EMBED_API_KEY`, `GBENCH_TAU2_EMBED_BATCH` (and the `TAU2_EMBED_*` aliases).
  `GBENCH_TAU2_BENCH_SRC`, `GBENCH_TAU2_ENV_RUN` and `TAU2_DATA_DIR` are no longer used.
- wildclawbench: `GBENCH_WILDCLAWBENCH_IMAGE` (default `gbench-wildclawbench:316334cc-v9`, built
  locally, never pulled), `GBENCH_WILDCLAWBENCH_HOST_DIR`,
  `GBENCH_WILDCLAWBENCH_TASK_IMAGE` (default `wildclawbench-ubuntu:v1.3`, built locally, never
  pulled), `GBENCH_WILDCLAWBENCH_CATEGORIES`, `GBENCH_WILDCLAWBENCH_TASK`,
  `GBENCH_WILDCLAWBENCH_THINKING` (default `high`, passed only with `--thinking`; OpenClaw's
  thinkingDefault, no effect on the wire, where the relay sets enable_thinking),
  `GBENCH_WILDCLAWBENCH_PARALLEL` (default the suite's concurrency),
  `GBENCH_WILDCLAWBENCH_TASK_NETWORK`, `GBENCH_WILDCLAWBENCH_TASK_HOST`,
  `GBENCH_WILDCLAWBENCH_TASK_ENDPOINT`, `GBENCH_WILDCLAWBENCH_JUDGE_PORT`,
  `GBENCH_WILDCLAWBENCH_JUDGE_REQUEST_TIMEOUT_S` (legacy
  alias `WILDCLAW_JUDGE_REQUEST_TIMEOUT_S`; unset: the in-container default 40 s),
  `GBENCH_WILDCLAWBENCH_MODEL_API_KEY`,
  `GBENCH_WILDCLAWBENCH_TIMEOUT_S` (default 86400), `GBENCH_WILDCLAWBENCH_SKIP_KEY_VALIDATION`. The Gemini keys
  (`GEMINI_API_KEYS` / `GEMINI_API_KEY`) and `GBENCH_WILDCLAWBENCH_MODEL_API_KEY` are passed into
  the orchestrator by name, never on its command line, and only its entrypoint and proxy hold them
  (the agent's
  model calls go through the proxy's model relay, which adds the model key; the task containers get
  the placeholder `gbench-model-relay`); the judge-cascade knobs (`GBENCH_JUDGE_MODELS`,
  `GBENCH_JUDGE_MODEL`, `GBENCH_JUDGE_CASCADE_ROUNDS`, `GBENCH_JUDGE_BACKOFF`) and the
  grounding-search knobs (`GBENCH_SEARCH_MODELS`,
  `GBENCH_SEARCH_MODEL`, `GBENCH_SEARCH_MAX_RPS`, `GBENCH_SEARCH_MAX_BURST`,
  `GBENCH_SEARCH_CASCADE_ROUNDS`, `GBENCH_SEARCH_ATTEMPTS`, `GBENCH_SEARCH_BACKOFF`,
  `GBENCH_SEARCH_HEDGE_DELAY_S`, `GBENCH_SEARCH_INITIAL_WIDTH`, `GBENCH_SEARCH_REDIRECT_TIMEOUT_S`)
  are forwarded to its Gemini proxy (the OpenClaw web_search is served by Gemini Google Search grounding;
  wildclawbench reads no Brave key). `GEMINI_OPENAI_BASE_URL` is passed by name, only as
  `base.require_gemini_openai_base_url()` accepts it. The task image's DuckDuckGo
  (`local-web-search`, `web-search`)
  and Google Custom Search (`google-search`) skills are deleted from every task container, so
  web_search is the agent's only search tool. The task containers run on
  `GBENCH_WILDCLAWBENCH_TASK_NETWORK`, default `bridge` (the upstream default; the cloud metadata
  guard checks it before the run, and resolves the host names the tasks need there; `host` is not
  covered by the DOCKER-USER rule). During the run the orchestrator's network monitor checks DNS and
  web egress on it, and before a task's grading starts the orchestrator stops every process the
  agent left in the task container. The proxy
  listens on the docker bridge gateway `172.17.0.1`, or on `GBENCH_WILDCLAWBENCH_TASK_HOST`, which
  must be a gateway of the task network (`docker network inspect`), or on `127.0.0.1` with the
  host network, on port 18795 by default; with the host network `GBENCH_WILDCLAWBENCH_JUDGE_PORT`
  may not be 18789-18793 or 18800-18899. A task with a
  harness error (a provisioned input missing, web-search setup failed, the task container or agent
  backend failed before the agent ran, never run, grading that could not run (including every 120 s
  grade() cap hit, and a cap hit or a Gemini service failure of a judge call after a stop of the
  agent's leftover processes that failed for a reason the agent cannot cause) or returned no
  numeric score, a judge fallback, a grader judge call not served during its grade() exec, the
  task network losing DNS or web egress (the orchestrator's network monitor; a failed lookup in
  the task's logs counts only with a monitor round where at least half of the names did not
  resolve), an agent helper-model call that
  failed on the service side, an agent web search not served, or a model-endpoint failure no later
  relay call of the task got past; attributed by time window and the requesting container) is left
  out of the headline, counted per kind and makes the status completed_with_errors, and so does a
  task graded while the agent's leftover processes could not all be stopped; the judge requests
  keep each task's own temperature (none when it sends none), a judge call that turns thinking
  off (06_Safety_Alignment tasks 2, 3 and 6) is sent at Gemini thinking level low, the lowest
  level every model of the default cascade accepts (`judge_thinking_off`), and gbench passes the
  run's Gemini lists into the orchestrator (`GBENCH_JUDGE_MODELS`, `GBENCH_SEARCH_MODELS`); each
  chat request's max_tokens is sent plus the Gemini thinking allowance (8192) and replies still
  cut there are counted (`gemini_output_limit`); the orchestrator image must carry the label
  `org.gbench.wildclawbench-harness-errors=9`. The agent's web_search timeout is 60 s, not
  OpenClaw's 30 s default (the launcher sets `tools.web.search.timeoutSeconds`; the proxy's
  search deadline is 57 s; a fixed constant, `AGENT_WEB_SEARCH_TIMEOUT_S`, not a knob; recorded
  on the result; 2026-10-13 A/B: 94.8% of the agent's searches answered in time at width 4 with
  a 57 s deadline, 65.2% at width 2 with 27 s; width 2 with 57 s, the shipped setting: 94.3%
  (Wilson 95% 90.3-96.7%) answered in time, no generated text (2026-10-13 A/B; the 2026-10-14
  confirmation run on the v9 image answered 21 of 21 agent searches); see
  docs/evals/wildclawbench.md).
  `GBENCH_WILDCLAWBENCH_MODEL_API_KEY` falls back to `VLLM_API_KEY`, else `OPENAI_API_KEY`, else
  the placeholder `dummy`. `GBENCH_WILDCLAWBENCH_TASK_NETWORK=host` requires the OpenClaw ports
  18789-18793 and 18800-18899 to be free on 127.0.0.1 (checked before the run; infra_required
  otherwise). Removed: `GBENCH_WILDCLAWBENCH_JUDGE_ENDPOINT` (setting it logs a warning).
- ui_control_osworld (not in `--evals all`; runs when named): `GBENCH_UI_CONTROL_OSWORLD_IMAGE`
  (default `gbench-ui-control-osworld:fc31a904`, built locally, never pulled),
  `GBENCH_UI_CONTROL_OSWORLD_TASK_IMAGE` (must be unset or `happysixd/osworld-docker`, the name
  OSWorld's docker provider starts, which must be the pinned
  happysixd/osworld-docker@sha256:0e6497a9295647cf05bf2b2af522fdd79bdeba2737595259cab310a3bcf6baa9;
  another value is infra_required; see ui_control_osworld.md step 4),
  `GBENCH_UI_CONTROL_OSWORLD_TIMEOUT_S` (default 172800), `GBENCH_OSWORLD_VM_DIR`, `GBENCH_OSWORLD_DOMAIN`,
  `GBENCH_OSWORLD_LIMIT` (a positive integer; anything else is infra_required before any container
  starts), `GBENCH_OSWORLD_MAX_STEPS` (default 15), `GBENCH_OSWORLD_OBS_TYPE` (default
  `screenshot`), `GBENCH_OSWORLD_TEST_META` (default `test_all.json`),
  `GBENCH_OSWORLD_MODEL_ENDPOINT`, `GBENCH_OSWORLD_MODEL_API_KEY` (the endpoint's key). The
  orchestrator, which holds the docker socket, always reaches the endpoint through gbench's relay
  on 127.0.0.1, which adds that key when there is one; the orchestrator holds only the placeholder
  `dummy`.
- mobilegym and mobilegym_pp: `GBENCH_MOBILEGYM_IMAGE` (default `gbench-mobilegym:5fbd3f34`, built
  locally, never pulled; shared by both `mobilegym` and `mobilegym_pp`), `GBENCH_MOBILEGYM_WORKERS`
  (parallel Playwright Chromium workers inside the container; default `min(sandboxes, 32)` when
  `--sandboxes` is set, else `min(concurrency, 16)`),
  `GBENCH_MOBILEGYM_TEMPERATURE` / `GBENCH_MOBILEGYM_PP_TEMPERATURE` (sampling temperature override;
  defaults to 0.0 without `--thinking` and 1.0 with `--thinking`), `GBENCH_MOBILEGYM_TIMEOUT`
  (container wall-clock timeout in seconds, default 86400). Both suites run on docker's `bridge`
  network with cloud metadata server guard (`container_guard.require_metadata_blocked`),
  `swebench_common.EndpointRelay` for model calls (adding the endpoint's key and `--max-output-tokens`
  when `GBENCH_AGENT_MAX_TOKENS_FROM_RUN` is on), and `DetokenizeRelay` for server-parser recovery;
  no credential enters the container.

### Pillar 6: multimodal vision and grounding
- mmmu_pro: `GBENCH_MMMU_PRO_PROMPT` (`both` default, `cot`, `direct`): prompt modes to run. MMMU-Pro
  reports the higher of CoT and Direct per setting, so a single mode is flagged not
  leaderboard-comparable.
- screenspot: `GBENCH_SCREENSPOT_DATA_DIR` (legacy alias `SCREENSPOT_PRO_DATA_DIR`): a local
  ScreenSpot-Pro data dir to score instead of the official ScreenSpot-v2 release; a named dir
  without `annotations/` and `images/` is a hard error (no silent switch back). The result records
  which dataset was scored.
- omnidocbench: `GBENCH_OMNIDOCBENCH_IMAGE` (default `gbench-omnidocbench:193627ae`, built locally,
  never pulled).

### Cross pillar
- gaia2: `GBENCH_GAIA2_IMAGE` (default `gbench-gaia2:are-1.2.0`, built locally, never pulled),
  `GBENCH_GAIA2_MODE` (default `run`), `GBENCH_GAIA2_CONFIG`, `GBENCH_GAIA2_LIMIT`
  (a positive integer; anything else is refused before any container starts),
  `GBENCH_GAIA2_NUM_RUNS`, `GBENCH_GAIA2_MAX_CONCURRENT`, `GBENCH_GAIA2_SCENARIO_TIMEOUT`,
  `GBENCH_GAIA2_TIMEOUT_S` (default 86400), `GBENCH_GAIA2_JUDGE_PORT` (default 18790), `GBENCH_GAIA2_MODEL_ENDPOINT`,
  `GBENCH_GAIA2_MODEL_API_KEY`, `GBENCH_GAIA2_SKIP_KEY_VALIDATION`, `GBENCH_GAIA2_KEEP_WORKDIR`.
  Removed (setting one logs a warning): `GBENCH_GAIA2_JUDGE_ENDPOINT`, `GBENCH_GAIA2_JUDGE_MODEL`:
  ARE's judge is always the gbench Gemini cascade behind the in-container proxy on 127.0.0.1.
  No key or token enters the container (it runs the agent's tool calls): gbench's endpoint relay
  adds `GBENCH_GAIA2_MODEL_API_KEY`, gbench's Gemini relay adds the Gemini key (the container gets
  the relay's placeholder by name), and no HF token goes in; `GEMINI_OPENAI_BASE_URL`, where the
  relay sends the key, only as `base.require_gemini_openai_base_url()` accepts it.
  `GBENCH_GAIA2_TEMPERATURE` (default 0.0 without `--thinking`; 1.0 with it).
  `GBENCH_GAIA2_NUM_RUNS` defaults to max(3, `--attempt-count`); `GBENCH_GAIA2_MAX_CONCURRENT`
  defaults to the suite's concurrency (`--eval-concurrency` / `--batch-sizes`), not ARE's CPU
  count; `GBENCH_GAIA2_MODEL_API_KEY` falls back to `VLLM_API_KEY`, else `OPENAI_API_KEY`.
- gdpval: `GBENCH_GDPVAL_MAX_FILE_CHARS` (default 20000), `GBENCH_GDPVAL_MAX_PDF_PAGES` (default 10),
  `GBENCH_GDPVAL_MAX_JUDGE_CHARS` (default 24000), `GBENCH_GDPVAL_MAX_RUBRIC_CHARS` (default 8000),
  `GBENCH_GDPVAL_PDF_PAGE_DPI` (default 150: resolution of the page images sent for an input PDF
  with no text layer).

Suites not listed here read no suite specific environment knob beyond the cross cutting ones (they are
configured entirely by the command line flags and their pinned dataset).
