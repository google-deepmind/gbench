# The endpoint relay: progress, request shape, smoke episodes, client closes

`EndpointRelay` (`gbench/runners/eval_suites/swebench_common.py`) sits between an agent harness and
the served model for the suites that cannot be pointed at the endpoint directly: swe_bench_live,
swe_bench_pro, swe_bench_multilingual, copilot_bench_swe, multi_swe_bench, terminal_bench,
skillsbench, tau2/tau3, toolbench, gaia2, wildclawbench, swe_lancer, ui_control_osworld, aider_polyglot,
livebench, bfcl_v4_agentic, and the re-ask relay of gpqa/causalbench/longbench_v2_code. It adds the
endpoint's key and the run's body fields (e.g. the thinking switch), refuses every path but the
model routes (and the server-root paths a caller names, `root_paths`: toolbench's POST
`/tokenize`, recorded as `root_paths`, `root_requests_relayed` and `root_path_errors`), and
counts what it sees in its `record()`, which each result keeps. A suite that fails after starting
relays keeps their records in its failed result as `relays_at_failure` (`runners/evals.py`),
which the smoke verdict reads.

This page covers five things the relay does for every relayed request since 2026-10-11:

1. A model request fails when it stops making **progress**, never because it took long.
2. It checks the **shape** of each model request against what the run expects (optional).
3. In a smoke run it ends each agent episode after its first reply (**smoke episodes**).
4. It records the **age of each request whose client gave up**, so client-side timeouts show.
5. For the agent suites whose scaffold sends no token budget (`swe_agentic.AGENT_MAX_TOKENS_SUITES`),
   it puts the run's `--max-output-tokens` as `max_tokens` on every model request that carries
   neither `max_tokens` nor `max_completion_tokens` (**max tokens from the run**; on by default,
   `GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0` turns it off).

Apart from item 5 and the opt-in `GBENCH_RELAY_STREAM_UPSTREAM` below (off by default), none of
them changes a request the agent sends or a reply it gets.

## Why

From the 2026-10-10 smoke run of the 84 suites at `--eval-limit 5`:

- SWE-agent's litellm 1.87 falls back to a 600 s client timeout per request, and the OpenAI SDK
  under it tries 3 times. The relay saw trimesh turns closed by the client at 600 s at least 12
  times: every reply that needed more than 600 s of decoding was thrown away and sampled again,
  a hidden cap on the measurement that no result recorded. Fixed 2026-10-11 for the SWE agents
  (swe_bench_live, swe_bench_pro, multi_swe_bench, and the mini-swe-agent suites
  copilot_bench_swe and swe_bench_multilingual): their model client now waits the relay's safety
  cap on a request plus 60 s (`swe_agentic.client_timeout_for_relay_limit`; tau2/tau3 size it the
  other way round, the relay from the client: relay limit = the agent's client timeout plus
  120 s, never below `TIMEOUT_S`), with no transport retries for the litellm/SDK
  clients (`AGENT_CLIENT_RETRIES` = 0; MSWE-agent keeps the SDK's 2, which only retry an error
  the relay answers). It is recorded in the result's `model_client_timeout`
  (`hidden_cap_removed` where a hidden cap was replaced).
- The relay itself read a non-streamed reply with a 3600 s socket timeout. vLLM sends a
  non-streamed reply whole at its end, so that was a 3600 s cap on every non-streamed turn
  (262,144 tokens at 120 tok/s take 2,185 s unloaded, and 1.4-2.4x longer when several suites
  share the server).
- A stuck request was only noticed when some timeout far away fired, hours later.

## 1. Progress, not duration

A model request (a `/chat/completions` or `/completions` path) is watched by a progress clock.
Progress is either:

- **an event of the request's own reply stream**: each `data:` event of a streamed reply. An SSE
  comment (vLLM's `: keep-alive`) is not progress; or
- **the server working**: the endpoint's `/metrics` (vLLM, at the server root), read per engine
  (one per data-parallel rank). An engine moved when its token counters
  `vllm:generation_tokens_total + vllm:prompt_tokens_total + vllm:iteration_tokens_total_sum +
  vllm:prefix_cache_queries_total` advanced, or its `vllm:kv_cache_usage_perc` changed (either
  way). The server is working when some engine moved and no engine that holds requests
  (`vllm:num_requests_running + vllm:num_requests_waiting > 0`) stood still.

  The KV-cache gauge is there for prefills. vLLM counts a request's prompt tokens only at its
  first output token, so the token counters do not move at all during a prefill. Measured
  2026-10-11 on 26B-A4B with DP=2: all of them stood still for 7.3 s through a 182K-token
  prefill and for 12.5 s through a 254K-token one. Meanwhile the KV-cache usage changed at every
  0.5 s read, because each prefill chunk allocates blocks. Before this was added, a live run of
  that 254K-token request through a relay with the windows scaled down (2 s idle, 3 s confirm)
  got a 504 `stalled_no_tokens` from a healthy server. With the gauge it returned 200 in two out
  of two runs. A longer prefill (a dense 31B model, a slower GPU, a recompute after preemption)
  would otherwise have outlasted the 30 s confirm window.

After `IDLE_TIMEOUT_S` (300 s) without an event of its own, the relay reads the counters twice,
`PROGRESS_CONFIRM_S` (30 s) apart:

| Engines | Verdict |
|---|---|
| some moved, and every engine holding requests moved | the server is working (the request is queued behind others under load, prefilling, or preempted): wait another 300 s |
| none moved, or an engine holding requests stood still | **stalled**: the endpoint's connection is shut down (vLLM aborts the request), the client gets a 504, counted as `stalled_no_tokens` |
| cannot be read (no `/metrics`, or all counters 0: a server that keeps no stats) | a streamed reply is stalled after 300 s without an event; a non-streamed reply (nothing to see before its end) waits for its safety cap |

So a stuck server is caught in about 330 s; a slow reply that keeps coming, a long non-streamed
reply on a working server, and a request waiting in a busy server's queue all complete.

`stalled_no_tokens` is an endpoint-side failure (`endpoint_side_failure`): it is counted in
`upstream_errors`, attributed to the rollout whose conversation it carried, and makes the run
`completed_with_errors`, like any other endpoint failure. A stall after a streamed reply started
ends the stream with an error event (`data: {"error": {..., "type": "stalled_no_tokens"}}`).

**Safety cap.** A model request's total time is bounded only by
`request_total_s` = max(`TIMEOUT_S` (3600 s, or the caller's `timeout_s`),
`base.request_timeout_s(budget)` + 120 s), where the budget is the request's
`max_completion_tokens` / `max_tokens`, or the served window (`max_model_len` run knob, else
262,144) when it sends none. At the default `GBENCH_REQUEST_TIMEOUT_S` 1200 and
`GBENCH_MIN_DECODE_TOK_S` 8: 9,512 s for a 65,536-token budget, 34,088 s without a budget. When the
cap is reached the request counts as `connection` (a reply the endpoint never finished) and the
client gets a 502. Other paths (the model list, `/tokenize`) keep `TIMEOUT_S`.

**Transport.** By default every request goes to the endpoint exactly as the client sent it (plus
`return_token_ids`, which the relay already asked of a vLLM endpoint for a non-streamed request and
takes out of the reply again): a non-streamed reply is the endpoint's own non-streamed reply,
byte for byte. With `GBENCH_RELAY_STREAM_UPSTREAM=1` (or `stream_upstream=True`) the relay sends a
non-streamed chat request to a vLLM endpoint as a stream (`stream: true`,
`stream_options.include_usage`) and rebuilds the non-streamed reply from its events, so the
request shows its own progress. It is off by default because vLLM's streaming path does not always
produce the same reply. The 2026-10-11 live check against vLLM 0.26.1rc1 (gemma-4-26B-A4B-it,
temperature 0, the same request both ways):

| Request | Same token ids | Rebuilt reply equals the non-streamed reply |
|---|---|---|
| no thinking, plain | yes | yes |
| thinking, plain | yes | yes |
| no thinking, two parallel tool calls | yes | yes |
| thinking, tool call with quotes in its arguments | yes | yes |
| thinking, tool call with nested JSON arguments | yes | yes |
| no thinking, content before a tool call | yes | **no**: the stream keeps the content's trailing `"\n\n"`, the non-streamed reply strips it |
| thinking, cut off at max_tokens inside the reasoning | yes | yes |
| cut off at max_tokens inside a tool call | yes | yes |
| thinking, unicode and a code block | yes | yes |
| thinking, after a tool result | yes | yes |
| no thinking, after a tool result | yes | yes |
| no token ids asked for | yes | yes |
| thinking, `tool_choice: "required"` | yes | **no**: the stream finishes `"stop"`, the non-streamed reply `"tool_calls"` |

("equals" ignores the per-request `id`, `created` and tool call ids.) A request that asks for
logprobs, `echo`, `n` > 1 or sends `stream_options`, a `/v1/completions` request and a server that
is not vLLM keep the non-streamed transport even with the opt-in. Setting the opt-in changes the
replies the agent sees in cases like the two above, so a run that sets it is not comparable with
one that does not.

**Knobs.**

| Variable / argument | Default | What it does |
|---|---|---|
| `GBENCH_RELAY_IDLE_TIMEOUT_S` / `idle_timeout_s` | 300 | Seconds without progress before the counters are read |
| `EndpointRelay.PROGRESS_CONFIRM_S` | 30 | Seconds between the two counter reads |
| `GBENCH_RELAY_STREAM_UPSTREAM` / `stream_upstream` | off | Send non-streamed chat requests as streams and rebuild the reply (changes replies, see above) |
| `timeout_s` (caller) | 3600 | The floor of the safety cap |

A malformed `GBENCH_RELAY_IDLE_TIMEOUT_S`, `GBENCH_RELAY_EXPECTED_SHAPE` or
`GBENCH_SMOKE_RELAY_REPLIES` raises `RelayEnvError`, which names the variable. It is
deliberately not a `ValueError`: suites rewrite a `ValueError` from the relay constructor as
"the endpoint URL cannot be relayed" (tau_common) or "could not listen on" (skillsbench), which
would hide the typo. A bad argument from a caller is still a `ValueError`.

## 2. Request shape

`expected_shape={...}` (or `GBENCH_RELAY_EXPECTED_SHAPE`, the same as a JSON object, used when the
caller gives none) compares each model request's JSON body **as it goes to the endpoint** (after
the relay's `body_fields`, before its transport fields) with what the run expects. It never blocks
a request; it counts.

| Field | Value | A request matches when |
|---|---|---|
| `temperature` | a number x | it sends `temperature` equal to x (absent is a violation: the server's default would apply) |
| `temperature` | `null` | it sends no temperature |
| `enable_thinking` | true / false | `chat_template_kwargs.enable_thinking`, read as the chat template reads it (a truth value; absent is false), equals it (chat requests only: a `/v1/completions` prompt is already rendered) |
| `max_tokens` | an integer n | its `max_completion_tokens` (else `max_tokens`) is n |
| `max_tokens` | `null` | it sends no token budget (absent or null: the served window bounds the reply; the SWE harnesses canonically send none) |
| `max_tokens_at_most` | an integer n | it sends no budget, or one of at most n (`gbench --smoke`'s rule) |

A field not named is not checked. Recorded under `shape_check`: `expected`, `source` (`caller` or
`GBENCH_RELAY_EXPECTED_SHAPE`), `requests_checked`, `requests_with_violations`, `violations` (per
field), and the first 5 `examples` (`field`, `expected`, `sent`, `conversation`; `sent` is
`"<absent>"` for a field not sent). `shape_violations` repeats the per-field counts at the top of
the record.

## 3. Smoke episodes

A relay runs smoke episodes when it gets `smoke_replies=N`, or when `GBENCH_SMOKE_RELAY_REPLIES=N`
is set and the caller didn't pass `smoke_from_env=False`. `gbench --smoke` sets that variable
only for its agent suites (`smoke.AGENT_SUITES`). tau2/tau3's relay passes
`smoke_from_env=False` (a domain whose every simulation stopped would have nothing to evaluate),
so their episodes run to their end; their follow-up is seen on the smoke tap. `GBENCH_SMOKE=1`
is set for every suite of a smoke run, and on its own it changes no relay. So a non-agent relay (bfcl's agentic part,
aider_polyglot, livebench) runs its one sample to its own end under `--smoke`, as docs/smoke.md
says. In a smoke episode:

- A conversation is the `conversation_fingerprint` of a request's messages (the first user
  message), the key the relay already uses to attribute failures to a rollout.
- The relay serves the conversation's first N model replies normally. A reply counts as served
  once it is a 2xx reply sent in full to the client (a streamed one when its stream ended).
- The conversation's next model request is answered by the relay itself, never sent to the
  endpoint: HTTP 400 with
  `{"error": {"message": "GBENCH_SMOKE_STOP", "type": "gbench_smoke_stop", "param": null, "code": 400}}`.
  It is not an endpoint failure and is not counted in `upstream_errors` or `requests_relayed`.
- That next request is the **follow-up** when its messages grew past those of the last served
  request: the agent read the reply, ran the action it asked for and sent the result back. It is
  recorded once per conversation (`follow_up_arrived`) and counted in `smoke_second_requests`. A
  stopped request that did not grow (a retry, or another rollout with the same task prompt) is
  counted in `stopped_without_growth`.
- A request with no user message has no conversation and is never stopped (`unkeyed_requests`).
- A reply the endpoint failed is not served: the conversation's next request is sent on.

Recorded under `smoke`: `replies_per_conversation`, `source` (`caller` or `GBENCH_SMOKE_RELAY_REPLIES`),
`conversations` (per fingerprint: `requests`, `replies_served`, `served_messages`,
`shape_violations`, `stopped_requests`, `follow_up_arrived`, `stopped_without_growth`),
`conversations_served`, `smoke_stops`, `smoke_second_requests`, `unkeyed_requests`; and
`smoke_second_requests` at the top of the record. `gbench --smoke` reads these
(`smoke._relay_follow_up`, docs/smoke.md).

The re-ask relay of gpqa, causalbench and longbench_v2_code (`upstream_reask_relay`) is never a smoke relay: base's
single-turn requests are not agent episodes. Outside a smoke run, without `smoke_replies`, nothing
changes.

Limit: an agent whose first user message changes on every turn (one that rebuilds its prompt
instead of growing a conversation) is a new conversation each turn and is never stopped by the
relay; its suite ends its episode with its own step limit.

## 4. Client closes

Every request whose client closed the connection before the relay had sent the whole reply is
counted in `client_closed_streams` (as before: the relay shuts the endpoint's connection, so
vLLM stops generating, and it is not an endpoint failure). Its age at that moment (from the end of
the request body) is now kept:

- `client_close_ages`: `closes_with_age`, `histogram_s` (`<60`, `60-300`, `300-590`, `590-610`,
  `610-1790`, `1790-1810`, `1810-3600`, `>=3600`), `max_s`, `suspect_timeouts_s`,
  `suspect_window_s`, `client_timeouts_suspected`, `client_timeout_suspect_ages` (the first 5).
- `client_timeouts_suspected` (also at the top of the record): closes within 10 s of a known
  client-side timeout, by default 600 s (litellm's fallback, also the OpenAI SDK's default),
  1800 s (three tries of it), and 3660 s (MSWE-agent's driver fallback
  `MSWE_AGENT_CLIENT_TIMEOUT_S` = `TIMEOUT_S + 60`, used only when a spec names no client
  timeout; gbench's specs always name one). A caller whose client has another fixed timeout
  names it with `client_timeout_suspect_s`; gbench's own agent callers pass their client timeout
  (`AgentTimeouts.suspect_s()`), so a close at it is flagged. A non-zero count means a client
  gave up on replies at its own timeout: the run measured a capped model.

A client that stops reading a stream at a stop sequence it applies itself also closes early; its
age is whatever it was, and only closes near a suspect timeout are flagged.

## 5. Max tokens from the run

When the suite that runs is an agent suite (`swe_agentic.AGENT_MAX_TOKENS_SUITES`, read from the
run's `suite` knob) and `GBENCH_AGENT_MAX_TOKENS_FROM_RUN` is on (the default, the user's decision
of 2026-10-11), the relay sets `max_tokens` = the run's `--max-output-tokens` on every completions
request that carries neither `max_tokens` nor `max_completion_tokens`; a request that carries one
is left as it is. The upstream scaffolds of these suites send no per-turn budget, so this is a
recorded deviation. Recorded under `max_tokens_from_run`: `value`, `requests_filled`; the eval
runner (`runners/evals.py`, `_stamp_agent_max_tokens`) sums them into the result's
`agent_max_tokens_from_run` and a `leaderboard_comparable_reason` entry when `requests_filled` is
above 0. swe_bench_live, swe_bench_pro and multi_swe_bench take a second route: the run's value
goes out as one of the relay's `body_fields` (`swe_agentic.with_run_max_tokens`), set on every
completions request, so the fill sees a `max_tokens` already there, `requests_filled` stays 0 and
the eval runner records nothing for them; their own result records
`official_sampling.agent_max_tokens_from_run` and the reason whenever the run's value is sent.
`gbench --smoke` checks the same rule on the tap: an agent
request that reaches the endpoint with no `max_tokens` FAILs `request shape` (docs/smoke.md).

## The record, at a glance

New keys in `record()` (all earlier keys keep their meaning):

```text
client_timeouts_suspected      int
client_close_ages              {closes_with_age, histogram_s, max_s, suspect_timeouts_s,
                                suspect_window_s, client_timeouts_suspected,
                                client_timeout_suspect_ages}
upstream_transport             {streams_non_streamed_chat_requests, requests_streamed_upstream,
                                replies_assembled, idle_timeout_s, progress_confirm_s,
                                idle_waits_tolerated, longest_tolerated_idle_s,
                                timeout_floor_s, largest_request_cap_s}
upstream_errors.stalled_no_tokens   (when a request stalled)
shape_check, shape_violations       (with expected_shape)
smoke, smoke_second_requests        (with smoke_replies / GBENCH_SMOKE_RELAY_REPLIES)
max_tokens_from_run                 {value, requests_filled}  (agent suites, unless
                                    GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0; requests_filled
                                    stays 0 where the value is a body field)
root_paths, root_requests_relayed, root_path_errors   (with root_paths)
```

`idle_waits_tolerated` counts the 300 s windows a streamed reply went without an event while the
server's engines moved (a queued, prefilling or preempted request); `longest_tolerated_idle_s` is the
longest such silence.

## Verifying it

- `tests/test_relay_progress_shape_smoke.py`: loopback fakes, time scaled (an idle timeout of
  1 s for 300 s): a stuck server fails a request in about 1.3 s and its connection is shut down;
  a 30-minute-equivalent slow stream, a long non-streamed reply on a working server and a request
  queued behind keep-alive comments complete; one stuck data-parallel engine stalls; the safety
  cap sizes; the rebuilt reply's layout; shape rules; smoke stops; client close ages.
- Live (2026-10-11, the vLLM server on 127.0.0.1:8000, data-parallel 2): `server_progress()` read
  both engines; with an idle timeout of 3 s, a non-streamed 2,500-token reply that took 11.5 s and
  a streamed 2,702-event reply that took 14.5 s both completed with no failure; a smoke follow-up
  got `GBENCH_SMOKE_STOP`; the shape check counted the one request that sent a budget.
