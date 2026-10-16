# What gbench is and who it is for

## The problem

Evaluating an open-weights model well means answering two questions at once: how fast does it
serve, and how well does it reason under that serving configuration. In practice these are measured
by separate tools. Performance tools report tokens per second; academic harnesses report accuracy on
static datasets. Stitching them together to answer "if I serve this model at high concurrency, does
it still answer correctly?" is left to the user.

Three things make that harder:

1. Performance and quality are measured by different, disconnected tools.
2. Benchmark scripts often bind to one serving framework, so testing another means rewriting them.
3. Without fixed resource rules and workload shapes, numbers from different GPUs are not comparable.

## What gbench does

gbench measures serving performance and model capability through the same OpenAI-compatible `/v1`
HTTP interface, so any endpoint that speaks `/v1` can be tested without changing the harness. It has:

* Serving performance: TTFT, TPOT, inter-token latency, and throughput/knee under fixed workload
  shapes, with a recommended GPU tier per model size that is validated against the required
  `--num-gpus`, so runs are comparable across hardware.
* Native academic evaluations: reasoning, math, coding, long-context, tool-use, and multimodal
  suites, each reconciled against its upstream benchmark (see `docs/RESOURCES.md` and
  `--eval-provenance`), with thinking-on support.
* A golden set: deterministic capability checks for code execution, tool use, structured output
  and chat-template serialization.
* Agentic quality: multi-turn agent scenarios (session recall, tool/plugin routing).

Every suite reports honestly: a suite that cannot produce a trustworthy score hard-errors rather than
emitting a misleading 0, and a run that deviates from a benchmark's canonical protocol is marked
`leaderboard_comparable = false` with the reason.

## Who it is for

* Application developers: check whether a model meets latency targets under realistic traffic while
  still answering correctly, using predefined workload campaigns.
* Hardware partners: a vendor-neutral standard with recommended resource tiers (validated against
  the `--num-gpus` a run passes) and fixed workload shapes, so a platform can be compared without a
  custom harness.
* Serving-framework maintainers: benchmark a new optimization over the same `/v1` endpoint and
  check for throughput or correctness regressions. gbench works with any OpenAI-compatible `/v1`
  server and is validated on vLLM and Ollama.
* Model researchers: reproducible baselines; gbench derives model metadata from the model config, so
  there is less manual setup between a checkpoint and a number.
