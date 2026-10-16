# api_bank

Canonical API-Bank (Li et al., EMNLP 2023; arXiv:2304.08244; code
[AlibabaResearch/DAMO-ConvAI/api-bank](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank);
dataset `liminghao1630/API-Bank`). A **runnable** tool-use benchmark: a simulated pool of
~73 Python APIs backed by a local fake DB. Three progressively harder ability levels, two
tracks each:

- **Level-1 Call**: all API descriptions given, emit the right call.
- **Level-2 Retrieve+Call**: first `ToolSearcher`-retrieve the API, then call it.
- **Level-3 Plan+Retrieve+Call**: batch-inference steps over unknown tools.

## Scoring (execution-based, no LLM judge)

- **Correctness (Accuracy)**: the predicted call is **executed against the simulated
  backend** and compared to the gold result via each API's `check_api_call_correctness`
  (name + executed `output`/`exception`), exactly as upstream `evaluator.py`; parameter
  values that execute to the same result are correct, not lenient string matching. L1/L2
  reuse the upstream `Evaluator.evaluate` verbatim (fresh `ToolManager` per call); L3 uses
  `lv3_apis` + `level-3.json` gold and a shared `ToolManager`.
- **Response quality**: ROUGE-L F over the AI-turn generations.
- **Level-3 sample success rate**: `lv3_evaluator.py`'s `(50 - #dialogues-with-any-errored-step)/50`.
  That formula assumes every one of the 245 Level-3 call steps was predicted, so gbench computes
  it only on a run that scored all of them (`dimension_scores.L3_sample_success` is `null` on an
  `--eval-limit` subset, and `L3_sample_success_note` says why). A dialogue whose only failed steps
  are harness errors (below) has no known outcome and leaves the denominator, which the note
  records.

## Prompts

The API-call and response prompts are `evaluator.py`'s `api_call_prompt` and `response_prompt`
byte for byte, including the trailing spaces upstream's lines carry (`...actual parameters. `,
`...in your output. `, `Input: `). `PROMPT_SHA256` holds each string's sha256 at the pinned commit
and a test checks them.

## Harness errors

A failure of the harness, the endpoint or an API backend outside gbench's process is never the
model's outcome. Such a step is not scored: it is left out of its level's mean (numerator and
denominator), marked `status: HARNESS_ERROR` with `harness_error` on its trace, counted in
`harness_errors` / `n_harness_errors`, and named in `leaderboard_comparable_reason`; the run's
`status` is then `completed_with_errors`. When no call-track step is scored at all, the score is
withheld (`accuracy: null`, `status: failed`, `accuracy_withheld_reason`).

- **No reply** (`endpoint_failure`, `endpoint_timeout`, `over_context`): the model request failed,
  timed out, or its prompt did not fit the served context window.
- **An API backend that could not answer** (`backend_unavailable`, counted per API). Three
  simulated APIs reach outside the process (`backend_apis` in the result):

  | API | backend |
  | --- | --- |
  | `Translate` | `apis/translate.py`: googletrans, the Google Translate web service (`translate.googleapis.com`); upstream's `requirements.txt` pin `googletrans==3.1.0a0` runs in the local image `gbench-api-bank:googletrans-3.1.0a0` (see "The Translate API" below) |
  | `Dictionary` | `apis/dictionary.py`: `requests.get` to `https://api.dictionaryapi.dev` |
  | `ToolSearcher` | `apis/tool_search.py` and `lv3_apis/tool_search.py`: `SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')`, loaded on every call |

  A failed call step is the backend's when the model called the gold API and the execution came
  back with an exception that is plainly the harness's or the backend's, whatever the gold call
  would give now: gbench's googletrans worker container failed, timed out or could not start
  (`translate_worker`), an httpx transport error (the request never got an answer: a connection
  error or a timeout, `translate_transport`), or Google answering HTTP 429, 5xx, 403 or 407
  (`translate_backend_status`); the kind is named in the harness error's `detail`. For any other
  exception it is the backend's when gbench's re-run of the step's **gold** call on the same
  backend fails too: then no call could have passed the step (a Hugging Face or network outage,
  the Translate service answering other than HTTP 200). A call to
  another API, a failure
  without an exception (a different output), or a failure the gold call does not share stays the
  model's error. Each failed call step also keeps the evaluator's reason (or the executed result)
  in `api_bank_detail`.

Upstream has none of this: `evaluator.py` counts every failure as a wrong call and
`lv3_evaluator.py` re-raises an unexpected exception, ending the run.

## The Translate API

`apis/translate.py` calls `googletrans.Translator().translate(text, dest=..., src=...).text`, the
synchronous API of the release upstream's `requirements.txt` pins, `googletrans==3.1.0a0`, which
pins `httpx==0.13.3`. That release cannot be installed next to gbench (it would downgrade `httpx`
and its dependencies; the current 4.x releases are asynchronous, so upstream's code cannot use
them), so it runs in a local image of its own, and upstream's `translate.py` runs unchanged:

- **The image.** `gbench-api-bank:googletrans-3.1.0a0`, built from
  `gbench/docker/api_bank.Dockerfile` (`python:3.10-slim`, `googletrans==3.1.0a0` under
  `gbench/docker/api_bank.constraints.txt`, which pins its resolved dependency set, and the worker
  `gbench/docker/api_bank_translate_worker.py`). gbench never pulls it; build it once from the
  gbench repository root:
  ```bash
  docker build -t gbench-api-bank:googletrans-3.1.0a0 -f gbench/docker/api_bank.Dockerfile gbench/docker
  ```
- **The call.** For the checkout's modules, `googletrans` is a stand-in whose `Translator` hands
  each `translate` call to googletrans 3.1.0a0 in a fresh container of that image (docker's bridge
  network, never the host network; read-only, unprivileged, no key or token in its environment)
  and whose `LANGUAGES` are 3.1.0a0's. The reply's `text` is what `translate.py` returns; an
  exception in the container becomes an exception with the same message, which `translate.py`
  records as the call's `exception`, as it would record googletrans' own.
- **Outages are exceptions.** The container builds `Translator(raise_exception=True)`. With
  googletrans' default (`raise_exception=False`), a Google reply other than HTTP 200 (a 429 rate
  limit, a 403) comes back as the untranslated source text in place of the translation, with no
  exception, so upstream would score an outage as the model's wrong call. With it, such a reply
  raises `Unexpected status code "<status>" from ['translate.googleapis.com']`, and a step whose
  gold call fails the same way is a `backend_unavailable` harness error (above).
- **Proxies.** As in upstream, where httpx 0.13.3 reads the proxy variables when `Translator()` is
  built: `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY` and `NO_PROXY` (upper or lower case) set when a
  Translate call builds its `Translator` are passed to the container, a proxy on a loopback address
  as `host.docker.internal` (the host's address on docker's bridge network, so such a proxy must
  accept connections from docker's bridge network). A proxy whose URL holds a password is refused
  (`infra_required` before the run): no credential enters a container that runs the model's calls.
- **Checks before the run** (`infra_required` when one fails): the docker CLI and daemon, the image
  and its labels, the image's googletrans 3.1.0a0 and httpx 0.13.3 and a worker identical to this
  checkout's `gbench/docker/api_bank_translate_worker.py` (a probe container), and the cloud
  metadata server blocked on docker's bridge network (the host rule
  `sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP`), with
  `translate.googleapis.com` (or the proxy's host) resolving there (the host looked up is the
  one the worker's httpx would use: `HTTPS_PROXY` / `HTTP_PROXY`, else `ALL_PROXY`, else the
  service itself).
- **The result** records `translate_backend`: the googletrans and httpx releases, the image, the
  host, `raise_exception: true`, the proxy variables passed (names only), the run's `calls`, the
  calls that raised by exception class (`calls_raised`) and the worker failures (`worker_failed`).

Google Translate may refuse `translate.googleapis.com` from some networks: from a Google Cloud VM
it answered HTTP 429 ("Sorry...", its unusual-traffic page) to every request on 2026-09-27, so
there every Translate step whose gold call fails the same way is a `backend_unavailable` harness
error, named in the result; a proxy (above) or another network avoids it.

The Dictionary API calls `api.dictionaryapi.dev` from gbench's process. On 2026-09-28 that host
did not answer from a Google Cloud VM: its Cloudflare edge accepted the connection and returned
HTTP 522 (the origin server timed out) after about 20 seconds, twice, while `dictionaryapi.dev`
itself answered 200. That is an outage of the service, which a proxy or another network does not
avoid. While it lasts, the 5 Level-1 Dictionary call steps are `backend_unavailable` harness
errors: left out of the level means and named in the result, with the status
`completed_with_errors`, and the run is not leaderboard-comparable. Check before a run that
`curl -s -o /dev/null -w '%{http_code}\n' https://api.dictionaryapi.dev/api/v2/entries/en/hello`
prints 200.

## Pinned sources

- L1/L2 conversations and the execution backend: an `api-bank/` checkout of
  `AlibabaResearch/DAMO-ConvAI` at commit `f30ccf22b4e2617fab32958d4c03f5c1f2e7dfcf` (the last
  commit touching `api-bank/`; the directory is unchanged at HEAD). Before a run gbench hashes
  `evaluator.py`, `tool_manager.py`, `api_call_extraction.py` and everything under `apis/`,
  `lv3_apis/` and `lv1-lv2-samples/` (git blob ids, bytecode caches ignored) and hard-errors if
  the result differs from that commit's. `init_database/` is not checked because the simulated
  APIs write to it.
- L3: `liminghao1630/API-Bank` at revision `12e8158b7628c168f07e8f31fbbe3445e99f44cf`
  (`test-data/level-3-batch-inf.json`, `level-3-batch-inf-response.json`, `level-3.json`).

## Reading the result

Headline `accuracy` = mean execution Correctness over L1/L2/L3 (call track);
`dimension_scores` carries all six numbers (`L{1,2,3}_call`, `L{1,2,3}_response`) plus
`L3_sample_success`; `category_accuracy` holds the call-track `correct` / `total` / `accuracy`
per level (the `L{1,2,3}_call` numbers); `dataset_revision` records the two pins above.
`leaderboard_comparable` is always `false`, and `leaderboard_comparable_reason` says why: API-Bank
reports each level separately, so the headline is gbench's unweighted mean of the three
call-correctness levels (compare `dimension_scores` with published per-level numbers). The reason
also lists `--eval-limit`, the temperature gbench sent (evaluator.py sends none) and any harness
errors.
`translate_backend` records where the Translate API's googletrans ran and its calls (see "The
Translate API").
`official_sampling` records evaluator.py's settings (no sampling field sent), the temperature this
run sent and whether it matched. `total_questions` counts the call-track steps (the
headline's), `effective_n` those scored (harness errors left out) and `correct_answers` the correct
ones; `response_track_samples` counts the ROUGE-L steps, which are not in the headline. A step whose
request produced no reply is a harness error (above), not a 0. Inline
reasoning (`<thought>...</thought>` and the like) is removed from each reply before the API call
is extracted and scored, as every gbench suite scores replies; otherwise a call the model only
considered inside its reasoning would be taken as its answer. No request carries `max_tokens`,
with or without `--thinking`, as `ChatGPTWrapper` (and `GPT4Wrapper`, which `evaluator.py`
builds) sends none: the server grants the room left in the context window, `--max-output-tokens`
is not applied (`max_output_tokens_note` says so), and a reply cut off at the end of the context
window is counted in `truncated_responses`, as it would be cut off upstream.

## Messages

L1/L2 requests are built as `evaluator.py` builds them, message for message: the API-call or
response prompt (with the API descriptions) as a `system` message, then the conversation so far
with `User` turns as `user`, `AI` turns as `assistant` and each earlier API result as a `system`
message `[ApiName(key='value', ...)] Response: <output>`. Nothing is merged or re-roled, so the
served chat template must accept a `system` message after the first turn and two turns of one
role in a row (gemma-4's does); a server whose template rejects them fails those requests, which
count as failed requests. L3 steps are sent as one `user` message, the row's `instruction` and
`input` joined by a newline.

## Requirements

- An OpenAI-compatible `/v1` endpoint (single-turn per step; **no** GEMINI/LLM judge).
- **The API-Bank source checkout**, pointed at by **`GBENCH_APIBANK_DIR`** (execution
  backend + L1/L2 raw conversations). The suite **hard-errors** with the clone command if it
  is absent, and if its scoring files differ from the pinned tree:
  ```bash
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/AlibabaResearch/DAMO-ConvAI /path/apibank
  (cd /path/apibank && git sparse-checkout set api-bank \
    && git fetch --depth 1 origin f30ccf22b4e2617fab32958d4c03f5c1f2e7dfcf \
    && git checkout FETCH_HEAD)
  export GBENCH_APIBANK_DIR=/path/apibank/api-bank
  ```
  The fetch pins the checkout to that commit, so it keeps matching if `api-bank/` ever changes
  upstream (it has not changed since). With `GBENCH_PREREQS_DIR` set,
  `$GBENCH_PREREQS_DIR/apibank/api-bank` is used when the variable is unset.
- pip: the **checkout's own runtime deps** (NOT part of gbench's `evals` extra; install into the
  same environment gbench runs in): `rouge` (ROUGE-L scoring in the checkout's `evaluator.py`),
  `sentence-transformers` (ToolSearcher loads `paraphrase-MiniLM-L3-v2`, ~60MB, downloaded
  once) and `nltk` (the checkout's `apis/search_engine.py`):
  ```bash
  pip install rouge sentence-transformers nltk
  ```
  Use the environment's current `sentence-transformers`. Do **not** downpin to the checkout's
  `requirements.txt==2.2.2`; gbench's ToolSearcher path works with the modern release.
- **Docker and the googletrans image** for the Translate API (its gold steps are the 4 Level-1
  steps of `Translate-level-1-1.jsonl` to `Translate-level-1-4.jsonl`, one in each; Level 2 and
  Level 3 have none): `gbench-api-bank:googletrans-3.1.0a0`, built locally as shown in "The
  Translate API" above, and the host's metadata-server rule. `googletrans` itself is not
  installed in gbench's environment (a copy there is not used).
- The ToolSearcher model must load: when it checks the checkout, before sending any request,
  gbench loads `paraphrase-MiniLM-L3-v2` once and hard-errors (`infra_required`) when it cannot
  (no `sentence-transformers`, or the model neither cached nor downloadable). ToolSearcher is the
  gold first step of every Level-2 dialogue and of many Level-3 steps.
- ToolSearcher's encoder runs on the CPU (`TOOL_SEARCHER_DEVICE`), so no GPU memory is needed:
  upstream's `SentenceTransformer(...)` names no device and would take `cuda:0`, where the model
  server under test usually holds all the memory (the suite then failed with CUDA out of memory
  before its first request). gbench binds the class to a CPU default while it runs the checkout;
  upstream's files are unchanged.
- `ALL_PROXY`: `apis/translate.py` saves `ALL_PROXY`, sets it to `''` and assigns the saved value
  back, which raises `TypeError` when the variable is unset. gbench sets an unset `ALL_PROXY` to
  `''` (no proxy) for the scoring phase and unsets it afterwards; a set value is restored. The
  Translate call's own proxy is the one set when its `Translator` is built (see "The Translate
  API").
- L3 data auto-downloads from HF (`liminghao1630/API-Bank` at the pinned revision:
  `level-3-batch-inf*.json`, `level-3.json`). Needs network + `huggingface_hub` on first run; a
  file that cannot be downloaded stops the suite with an `[INFRA REQUIRED]` error.
- Live-network simulated APIs: the Translate (4 Level-1 call steps, through the googletrans image)
  and Dictionary (5 Level-1 call steps) gold steps call web services; when those cannot answer,
  the steps are harness errors (above).
- **Gold steps the pinned scorer cannot pass.** A self-check that feeds every Level-1/Level-2
  gold call (rendered from its `param_dict`) through upstream's `Evaluator.evaluate` on a checkout
  matching the pinned tree passed 338 of the 399 Level-1 call steps and 94 of the 135 Level-2 call
  steps. The failures come from upstream itself, so no model can pass those steps and the reachable
  Level-1 / Level-2 call accuracy is about 85% / 70%:
  - API drift: `apis/timed_switch.py` takes no `device_id`, which the gold calls of
    `TimedSwitch` / `CancelTimedSwitch` pass; `QueryScene` returns `light` where the gold has
    `Light`; `ToolSearcher`'s best match differs from the recorded one for 31 of the Level-2
    failures;
  - `api_call_extraction.py` `parse_api_call` turns every list argument into a string, so
    `RecordHealthData` ("The health data has invalid format."), `AddMeeting`, `QueryMeeting` and
    `ModifyMeeting` fail under any rendering;
  - `evaluate()` builds a fresh `ToolManager` for every call, so a step that depends on an
    earlier one cannot pass (`ForgotPassword`, `GetUserToken` after a password change,
    `RegisterUser` of an existing user);
  - the Translate and Dictionary steps failed in that environment because their web services
    could not answer; in a gbench run those are harness errors, not failures.

## Run
```bash
export GBENCH_APIBANK_DIR=/path/apibank/api-bank
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals api_bank --max-output-tokens 1024
```

Add `--eval-limit N` for a quick check (stratified across the level/track groups).

The run logs its planned workload in one line at its start (for example `[api_bank] --eval-limit
5: 5 sample(s) x 1 attempt = 5 generations`) and records it in the result as `eval_limit_plan`
(`limit`, `samples`, `per_setting`, `attempts_per_sample`, `generations`).

## Notes / caveats

- **Sampling:** the temperature is gbench's rule for every suite: 1.0 with `--thinking`, 0.0
  without. No other sampling field is sent, as `evaluator.py` calls the model
  (`chatgpt.call(messages)`; `utils.py` `ChatGPTWrapper` posts only `model` and `messages`, no
  temperature), recorded under `official_sampling` (`temperature_matches_official: false`).
  `--temperature`, or `GBENCH_API_BANK_TEMPERATURE` for this suite (note the underscore: the eval
  name `api_bank` slugs to `API_BANK`; it takes precedence), sets another. No `max_tokens` is sent (see above). No
  LLM judge: scoring is execution-based + ROUGE-L.
