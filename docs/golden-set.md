# Golden Set capability smoke test (`--golden-only`)

Tier 2 of the capability evaluation: the 16 deterministic capability checks.

[Back to the README](../README.md)

## Tier 2: Golden Set capability smoke test (`--golden-only`)

A deterministic capability smoke-test suite of 16 cases, meant as a deployment check for the model under test: a healthy model passes every case that applies to it (a case for a modality the model does not have, such as audio, is not applicable and leaves the denominator). A change to the cases changes the Golden Set's `scaffold_id` (see [docs/scaffold-versioning.md](scaffold-versioning.md)). The Golden Set measures a model; it is not a gate for gBench's own pull requests, whose gate is the unit-test suite.

- Two-sided presence matching: case matchers (`contains_all`, `refusal`) require both the expected tokens (`expected_outputs`) and the absence of forbidden ones (`expected_not_contains`), so a reply cannot pass by accident; the four `chat_template` cases also forbid the served model's own chat-template tokens, read from its tokenizer at run time.
- Python execution matcher (`python_exec`): for code generation (`code_canonical`), gBench extracts the generated Python code block and runs unit-test assertions (`assert is_palindrome('racecar') == True`) in an isolated namespace.
- Media: local images are sent as Base64 data URIs (`data:image/...;base64,...`) in `image_url` parts, and audio as Base64 in `input_audio` parts.
- Chat template sensitivity pairs: four `chat_template` cases put an assistant `tool_calls` block, a `tool` turn or a multi-turn history on the wire, each paired with a twin asking for the same answer through one plain user turn, isolating a broken chat template from a weak model. See [Chat Template Sensitivity Pairs](#chat-template-sensitivity-pairs).
- Endpoint auth: the Golden Set sends `VLLM_API_KEY` (or `OPENAI_API_KEY`); for a remote endpoint with `GBENCH_GCP_AUTH=1` and no key it fetches a gcloud identity token (never for `localhost`, `127.0.0.1` or `0.0.0.0`).

| Case ID | Category | What it checks | Match type |
| :--- | :--- | :--- | :--- |
| `audio_transcription_smoketest` | `multimodal_audio` | Audio input processing and transcription | `contains_all` |
| `code_canonical` | `code_generation` | Emits code passing unit test assertions | `python_exec` |
| `function_call_single` | `tool_use` | Fills tool call argument from prompt | `tool_call` |
| `identify_person` | `entity_identification` | Factual entity naming without hallucination | `contains_all` |
| `location_identification` | `location_identification` | Grounding factuality | `contains_all` |
| `math_canonical` | `math` | Arithmetic reasoning final answer | `answer_exact` |
| `multimodal_vision_landmark` | `multimodal_vision` | Image landmark recognition | `contains_all` |
| `safety_boundary` | `safety` | Refuses malware request without code | `refusal` |
| `structured_json` | `structured_output` | Valid JSON emission without Markdown | `json_exact` |
| `template_multi_turn` | `chat_template` | Five turns arrive in order, so "second warehouse" resolves | `contains_all` |
| `template_multi_turn_twin` | `chat_template` | Control: the same content in one plain user turn | `contains_all` |
| `template_tool_response` | `chat_template` | A replayed tool result is serialized and readable | `contains_all` |
| `template_tool_response_twin` | `chat_template` | Control: the same record in one plain user turn | `contains_all` |
| `tool_call_minimal` | `tool_use` | Well-formed tool call without arguments | `tool_call` |
| `translation_romance` | `translation` | Romance language translation | `contains_all` |
| `two_rivers_zurich` | `exact_knowledge` | Recalls both halves of a two-part fact | `contains_all` |

```bash
# Run specific Golden Set tasks against a multi-model endpoint
gbench --golden-only --remote-endpoint http://127.0.0.1:8000/v1 --tokenizer google/gemma-4-12B-it --golden-tasks code_canonical function_call_single --served-model-id google/gemma-4-12B-it
```

### Chat template sensitivity pairs

A chat template is the Jinja that flattens structured messages into the single string the model was trained on. Get it wrong and nothing errors. The model receives a prompt in a shape it never saw in training and quietly gets worse, which reads as a weak checkpoint rather than as a packaging bug.

The four `chat_template` cases separate those two. They are ordinary Golden Set cases sent to the same served endpoint over the same `/chat/completions` call, because the template that matters is the one the server applies at request time and not whichever file happens to be on disk. What makes them a subset is the label: they are the only cases that put an assistant turn, a tool turn or a multi-turn history on the wire, so they are the only ones whose result depends on how those get serialized.

| Sensitive Case | Twin | Structure Verified |
| :--- | :--- | :--- |
| `template_tool_response` | `template_tool_response_twin` | An assistant `tool_calls` block and the `tool` turn answering it, tied together by `tool_call_id` |
| `template_multi_turn` | `template_multi_turn_twin` | Five alternating turns that have to arrive in order |

Read a pair together the way you read the two `tool_use` cases:

| Sensitive Case | Twin | Diagnosis |
| :--- | :--- | :--- |
| pass | pass | Serialization healthy |
| fail | pass | The model can read the value, so the structure is what broke. Look at the chat template |
| fail | fail | The model cannot answer the question at all. The template is not implicated |

That table covers the failure where the value never arrives. Both halves of both pairs also catch the opposite failure, where the template emits its scaffolding instead of consuming it. They forbid in the visible answer the served model's own chat-template tokens (`forbid_chat_template_tokens`), which gBench reads at run time from that model's tokenizer: every special token plus every added token that occurs in its chat template. For a local server started with `--vllm-serve-args='--chat-template t.jinja'`, the template is that file; for a remote endpoint it is the tokenizer's own. With such a `--chat-template` file, its delimiter-shaped literals (`<|...|>`, and `<...>` names with an underscore such as `<start_of_turn>`) that the tokenizer has no token for are forbidden too (`template_literal_delimiters`). Whitespace-only tokens, single characters and tokens made only of letters, digits, underscores and spaces are left out, because correct answers contain them. The list is derived rather than written into the cases, so a leak is caught in whatever delimiters the model under test uses, Gemma or any other family. The tool pair also forbids two plain strings, `tool_call_id` and `<tool_response>`. The tokenizer is `--tokenizer` when given (on a local run, else the `--tokenizer` in `--vllm-serve-args`), else the model's id, else the id the endpoint serves it under. It is loaded with `transformers`, which gBench does not declare: a machine that serves the model has it with vLLM; with `--remote-endpoint`, install it (`pip install transformers`) on the machine running gbench. It is read from the Hugging Face cache or the Hub, or from a local directory, with remote code trusted only with `--tokenizer-trust-remote-code` or when `--vllm-serve-args` carries `--trust-remote-code` (any spelling vLLM accepts, e.g. `--trust_remote_code`). The Golden Set summary records which tokenizer was read and the exact tokens checked (`chat_template_leak_check`). If the tokenizer cannot be loaded, or yields no token, those four cases are harness errors and the run reports no score; they are never passed unchecked.

Results break down by category, so the subset reads as one signal before you read the individual cases. A broken model leaves it green while other categories go red:

```text
gemma-4-E2B-it       google/gemma-4-E2B-it    13/16      FAIL
  - math                                      0/1        FAIL
  - safety                                    0/1        FAIL
  - tool_use                                  1/2        FAIL
  - chat_template                             4/4        PASS
```

A broken template inverts that. Here the tool turn never reached the model, and the twin still passes because it was handed the same record as plain text:

```text
gemma-4-E2B-it       google/gemma-4-E2B-it    15/16      FAIL
  - chat_template                             3/4        FAIL
  - code_generation                           1/1        PASS
  ...

  FAIL   template_tool_response: Missing expected substrings: ['QX7R2M', 'delayed']
```

Two limits are worth knowing before trusting the signal.

**The isolation is a structural argument, not a measured one.** A twin is claimed to be template-insensitive because a single plain user turn is the smallest surface a template has, and a template that breaks that fails every case in the set, at which point the category row is not what you are reading anyway. Nobody has run these against a deliberately corrupted template on real weights. The argument holds only while the twins stay trivial, so `test_template_subset_is_paired` in [`tests/test_golden.py`](../tests/test_golden.py) asserts each twin is still one user turn with no tools and no media, and that both halves of a pair still assert the same thing.

**The subset says a template is broken, not which line of it.** For a Gemma 4 template, [`tools/template_checks/test_chat_template_gemma4.py`](../tools/template_checks/test_chat_template_gemma4.py) renders a `chat_template.jinja` locally and diffs it against `google-deepmind/dialog`, the training-time serialization. It checks that model artifact, not gBench, so it lives outside `tests/` and never runs in gBench's unit-test gate. It needs no GPU, no weights and no endpoint, so it can gate a change to the template itself before any model exists to serve it. It skips unless pointed at a file:

```bash
curl -L -o /tmp/gemma4.jinja \
  https://huggingface.co/google/gemma-4-31B-it/resolve/main/chat_template.jinja
GEMMA4_TEMPLATE_PATH=/tmp/gemma4.jinja pytest tools/template_checks
```

29 checks assert behaviour the shipped template already has. 19 assert dialog-correct behaviour it does not, and are `xfail(strict=True)`, so a template that closes one of those gaps reports an XPASS and the stale marker gets removed rather than quietly lying. Full detail in [`tools/template_checks/README.md`](../tools/template_checks/README.md).
