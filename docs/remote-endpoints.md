# Remote endpoint mode (`--remote-endpoint`)

Benchmarking any OpenAI-compatible `/v1` server, and the preflight, smoke and parallel-suite modes.

[Back to the README](../README.md)

## Remote endpoint mode (`--remote-endpoint`)

Remote Endpoint Mode decouples the client evaluation harness from backend serving engine internals:

- No local GPU checks: local GPU detection (`check_gpu_ready`) is skipped, so a CPU-only machine can drive a GPU server.
- Served model resolution: `--remote-endpoint` is normalized once to end in `/v1`. Every pillar checks the endpoint's `/v1/models` against `--models` and runs against the id it matches: the first of the model's HuggingFace id, path, name and short name that equals a served id, exactly or case-insensitively, else the one served id whose last path component equals one of them (case-insensitive; `google/gemma-4-E4B-it` matches a served `gemma-4-e4b-it`, or `other-org/gemma-4-E4B-it`, when no other served id ends in that name); an endpoint whose ids match none is an error that names `--served-model-id`, which then selects the id (it must be listed). Without `--models`, the endpoint must serve exactly one model (or `--served-model-id` names one). Results record the served id beside `model_name`.
- Multimodal transport: local images are sent as Base64 data URIs (`image_url`) and audio as Base64 in `input_audio` parts.
- Framework support: any OpenAI-compatible `/v1` server; validated on vLLM and Ollama.
- Reasoning and tool-call parsers (evals): serve the model with its serving recipe's parsers (for Gemma 4 on vLLM: `--reasoning-parser gemma4 --enable-auto-tool-choice --tool-call-parser gemma4`). Every request of gBench's shared eval client (the suites that run through `base.run_eval_suite`) and gaia2's agent requests ask for the generated token ids (`return_token_ids`), so text a parser takes out of a reply is recovered as the model's output: on a request sent with thinking off, text the reasoning parser put in `reasoning`, and a reply the tool-call parser deleted (a native tool call on a request that declared no tools). The recovery decodes the ids with the server's `/detokenize` (vLLM serves it next to `/tokenize`), else with `--tokenizer`; a deleted reply that cannot be recovered is a harness error, left out of the score and named. Suites with their own model client follow their harness. See the server-parser policy in [docs/evals/env-knobs.md](evals/env-knobs.md). A `--eval-thinking` reply that stops with its answer inside the thinking channel (reasoning, no content, no tool call) is the model's outcome in every suite: scored as an upstream harness reading `content` scores an empty answer, never a failed request, and counted per result in `answer_in_thinking`, `answer_in_thinking_of` and `answer_in_thinking_rate` (also in summary.json); see [docs/evals/answer_in_thinking.md](evals/answer_in_thinking.md).
- Cloud Run endpoints: a Cloud Run service that allows unauthenticated calls is measured like any other endpoint. IAM-protected services (Cloud Run's default) are not supported yet: the endpoint check that starts every remote run sends no identity token, so the run stops there, before a `GBENCH_GCP_AUTH=1` identity token (used by the golden and quality pillars) can apply.

```bash
# Run serving benchmarks against a local Ollama endpoint (or any remote /v1 endpoint; on Cloud
# Run, only a service that allows unauthenticated calls, see Cloud Run endpoints above)
gbench --models gemma4-qat:4b --remote-endpoint http://localhost:11434/v1 --tokenizer google/gemma-4-E4B-it --preset quick
```

---

### Preflight, smoke test and parallel suites

Against a `--remote-endpoint`, three flags check and run a large sweep (details, verdict rules and every knob in [docs/smoke.md](smoke.md)):

- `--preflight`: runs every selected suite's own prerequisite check (images, checkouts, datasets, keys) plus disk, Docker, the endpoint from this host and from a bridge container, all at once, without a model request; prints a PASS/FAIL table with each failure's reason and exits 2 on any FAIL (about 30 s for `--evals all`).
- `--smoke`: the preflight, then one sample of every suite (`--eval-limit 1`) as a pool of 12 suite processes, agent episodes behind a gBench smoke relay ending after their first action (tau2/tau3, mcp_bench, mcp_atlas and agent_dojo run their one episode to its own end), each suite's endpoint behind a recording tap. A suite PASSes when a reply came back, every model request carried the run's temperature, `--thinking` switch and at most `--max-output-tokens` (the agent suites' requests must carry it), and the agent's follow-up request arrived or the sample was scored. A model request that stays silent for `GBENCH_SMOKE_HANG_S` (300 s) while the server generates nothing FAILs its suite as a hang, and the whole run stops at `GBENCH_SMOKE_DEADLINE_S` (default 3900 s). The summary is the PASS/FAIL table, with no scores.
- `--parallel-suites N`: runs the sweep as a pool of up to N `gbench --evals-only --evals <suite>` processes, longest first, families that share images, files or ports kept apart, with one merged `summary.json` and `eval_summary.csv`; each suite's `eval_*.json` is the one a serial run writes, plus the concurrency it ran under (`pool`).

```bash
gbench --smoke --evals all --remote-endpoint http://127.0.0.1:8000/v1 \
  --tokenizer google/gemma-4-26B-A4B-it --thinking --max-output-tokens 65536
gbench --evals all --evals-only --eval-limit 5 --parallel-suites 8 \
  --remote-endpoint http://127.0.0.1:8000/v1 --tokenizer google/gemma-4-26B-A4B-it \
  --thinking --max-output-tokens 65536
```
