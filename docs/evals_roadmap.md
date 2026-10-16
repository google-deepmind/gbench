# gbench Evaluation Roadmap

gbench's evaluation suite is a living system, not a frozen list. New benchmarks appear, existing
ones publish fixes and new splits, and "state of the art" for what a *faithful, reproducible* eval looks
like keeps moving. This document is the standing plan for that continuous improvement: the standards
every eval must meet, the engineering playbook for adding or hardening one, and the prioritized
roadmap of evals to build next. It also tracks gbench tooling that is not shipped: section 8
covers the web results dashboard, its run launcher and their hosted deployment, which were taken
out of this release.

It is a technical contract, not marketing. If an eval is listed here as *planned*, it means the
canonical harness is not yet implemented, so there is no runnable suite for it: it is either
deregistered (absent from the registry and `--evals all`) or, if a placeholder exists, it
hard-errors (`infra_required`) rather than emit an approximate or fabricated number.

---

## 1. Non-negotiable standards (definition of done)

Every gbench eval, existing or new, must satisfy all of the following. These are hard rules;
"partially done" is not done.

1. **Canonical or nothing.** A suite is either a faithful measurement of the benchmark
   ("canonical-when-provisioned") or it hard-errors via `swebench_common.infra_required(...)`.
   There is no skip, no quarantine, no partial, and no fabricated/proxy number. A keyword-overlap
   or "first-action match" score published under a canonical suite name is *worse than no number* (it
   silently poisons the leaderboard) and is therefore forbidden.
2. **Delegate to the pinned upstream harness** when one exists, at a pinned commit (a benchmark
   with no release tags, pin a 40-char SHA). Do not re-implement scoring that upstream already
   defines; reproduce the exact split, decoding (greedy unless the benchmark says otherwise), metric,
   and answer extraction.
3. **Local by design.** Heavy or graph-conflicting dependencies go in a LOCAL
   `gbench/docker/<suite>.Dockerfile` built on the host; never `docker pull` an eval image into
   gbench's own build, and never a side venv. Additive pure-Python deps may go in the canonical env
   only after a `pip install --dry-run` shows no version changes to the pinned stack.
4. **Scoring: deterministic verifier preferred.** If the benchmark ships an execution-based / rule
   verifier, use it (no LLM judge). If it requires an LLM judge, reuse gbench's Gemini cascade
   (`base.judge_generate_cascade`) with the upstream judge prompts verbatim; swap only the model
   call, never the rubric.
5. **`leaderboard_comparable` is earned, not assumed.** It is `True` only for a full, greedy,
   canonical-protocol run scored by the canonical judge/agent. Any deviation (Gemini judge instead of
   the canonical one, a subset, a single trial where the leaderboard averages N, a self-hosted agent)
   sets it `False` with a specific machine-readable reason.
6. **Honest denominators.** The headline metric divides by the number of *selected* tasks; a
   missing/errored/timed-out task counts as 0, never dropped from the denominator (dropping
   inflates the score). Surface `n_scored` / `n_missing` / judge-fallback counts.
7. **Complete docs + tests.** Ship `docs/evals/<suite>.md` with the full provisioning recipe
   (checkouts + commit, `hf download`s, images to load, keys, toolchains, the exact run command, and
   precisely what hard-errors), plus deterministic unit tests and a smoke/E2E to the maximum extent
   the environment allows.

## 2. Engineering playbook (how to add or harden an eval)

**Files & registration.** A suite `foo` is: `gbench/third_party/eval_suites/foo.py` (the runner),
optionally `gbench/docker/foo.Dockerfile` + `foo_run.py` (launcher) + `foo_entrypoint.sh` (adapter),
`docs/evals/foo.md`, `tests/test_foo.py`. Register in `gbench/runners/eval_suites/__init__.py`
(import + `SUITES` map + `__all__`) and add the name to its pillar in `gbench/runners/evals.py`.

**Container-delegation pattern** (reference: `skillsbench`, `wildclawbench`, `mcp_bench`,
`ui_control_osworld`). The runner builds a `docker run` of the local orchestrator image; the
orchestrator drives the upstream harness. Two sub-shapes exist:
- **docker-out-of-docker (DooD):** the harness spawns *sibling* task containers on the host daemon
  (mount `/var/run/docker.sock`). Identity-mount pitfall: any path the harness bind-mounts into a
  sibling (`-v <path>`) is resolved on the host, so that path must be identical inside the
  orchestrator and on the host: identity-mount the workdir/checkout (`-v H:H`) and set `TMPDIR`
  there, or siblings get empty mounts. This bit skillsbench, wildclawbench, and OSWorld's qcow2.
- **In-process simulator** (reference: `gaia2`): no sockets, no siblings: the harness is a pure
  Python simulator; the orchestrator just needs `--network host`.

**Networking.** The model-under-test is served at gbench `/v1` (OpenAI-compatible). The orchestrator
runs `--network host` and reaches it at `127.0.0.1`. On a hardened host (`iptables FORWARD DROP`)
the docker bridge cannot reach host ports, so sibling task containers must also use `--network
host` (then the fixed agent ports may collide across parallel siblings, so serialize them, or use bridge
mode `172.17.0.1` on an unfirewalled host).

**Model wiring.** Point the upstream agent at gbench `/v1` via whatever it supports: an OpenAI base
URL (`OPENAI_BASE_URL`), a custom-provider config (OpenClaw `my_api.json`), or LiteLLM's `local`
provider (`--model openai/<name> --endpoint`). If the harness routes by model-name prefix or
hardcodes a name, and vLLM strictly rejects unknown names, pass a routing alias and monkeypatch the
outgoing request `model` field to the served name (reference: OSWorld's `gpt-4o` alias + payload
rewrite).

**Judge.** Reuse `gbench/docker/wildclawbench_cascade_judge.py`, a stdlib, OpenAI-compatible HTTP
proxy that runs the Gemini cascade (same model list / rounds / backoff as `base.judge_generate_cascade`)
and forwards the upstream prompt verbatim, overriding only model + temperature (0.0) and stripping
provider-only fields (`thinking`/`reasoning`, non-`function` tools). It has been reused unchanged
across wildclawbench and gaia2.

**Shared-runner hooks** (`gbench/runners/eval_suites/base.py`; every one is opt-in and the default
keeps the house behavior):
- `run_eval_suite(official_sampling=OfficialSampling(...))`: the upstream harness's temperature (or
  `send_temperature=False` when it sends none) and other request fields become the suite default;
  the operator's `GBENCH_<SUITE>_TEMPERATURE` / `--temperature` still wins and the result records
  `official_sampling` with `temperature_matches_official`. Suites outside `run_eval_suite` call
  `resolve_official_sampling` (which checks `params` the same way). `send_max_tokens=False` is for
  an upstream that sends no max_tokens: each request then asks for the room left in the context
  window (the field is omitted when the window is unknown), `--max-output-tokens` is not applied,
  and the result says so in `max_output_tokens_note`.
- `ToolLoopProtocol(invalid_arguments_as_error=True)`: a tool call whose arguments are not valid
  JSON gets `Error: <parse error>` back instead of running the tool with `{}`;
  `max_sample_wall_s`: a per-sample wall clock (started when the sample's first request gets a
  slot) after which the sample stops with status `WALL_CLOCK_LIMIT` and scores 0.
- `continue_final_message` / `add_generation_prompt` in a sample's meta are sent on the sample's
  first request and to `/tokenize`, so a suite can prefill the assistant turn as upstream does.
- `apply_shard` / `record_shard`: `--shard I/N` for a suite that builds its own item list; apply
  it to the full ordered list before the `--eval-limit` draw and record it on the result.

**Verify-don't-assume.** Pin the commit and read the real upstream (repo/wheel), not the gbench
docstring: the docstring's *hypothesis* has been wrong more than once (wildclawbench was assumed to
be a BenchFlow benchmark; it was InternLM's own harness). Verify CLI flags from `cli.py` /
`shared_params.py`, not from a README. Live-validate keys at gate time (an auth rejection is a hard error;
network flakiness is inconclusive and passes). Reap sibling containers by run-scoped label on
timeout/cleanup.

## 3. Roadmap (prioritized)

### #1: `lmarena_web_agent`, to become WebArena (status: planned; effort: XL)

**What it should measure.** Canonical WebArena (`web-arena-x/webarena`): 812 tasks in which
an agent drives a real Chromium browser (Playwright, `observation_type="accessibility_tree"`, the CoT
id+actree agent `p_cot_id_actree_2s`) across self-hosted websites and is scored by programmatic
per-task evaluators. The suite is *misnamed*: it is WebArena, not the LMArena human-preference arena.

**Current state.** Removed a fabricated number (keyword overlap between a text "plan" and a
checklist, over the wrong dataset `WPRM/annotated_webarena_checklist`, never opening a browser). The
suite is now deregistered (roadmap-only): it is not in the registry, so `--evals all` never runs
it and `--evals lmarena_web_agent` exits with an error pointing here (`ROADMAP_SUITES` in
`gbench/runners/eval_suites/base.py`). WebArena will be (re)introduced as a runnable suite (name
TBD, see Open decisions) when the canonical harness lands.

**Sites & ports** (all self-hosted): OneStopShop shopping (Magento, `:7770`), shopping-admin CMS
(`:7780/admin`), Postmill reddit (`:9999`), GitLab (`:8023`), OpenStreetMap map + tile/routing
backend (`:3000`), Wikipedia (kiwix `.zim`, `:8888`), a Flask homepage (`:4399`).

**Scoring** (`evaluation_harness/evaluators.py`): deterministic-dominant, `URLEvaluator`
(`url_match`), `HTMLContentEvaluator` (`program_html`), `StringEvaluator` (`exact_match` /
`must_include`, a substring check, *not* an LLM call). A minority use an LLM:
`StringEvaluator.fuzzy_match` (which calls `llm_fuzzy_match`) and the unachievable-task `ua_match` (`llm_ua_match`)
(upstream `gpt-4-1106-preview` @ temp 0).

**Buildable path** (mirror wildclawbench). `gbench/docker/lmarena_web_agent.Dockerfile` bakes a pinned
webarena checkout + Playwright (`playwright install --with-deps chromium`) in the *orchestrator*. The
runner `docker run --network host -v /var/run/docker.sock:...`; the orchestrator exports the
`SHOPPING/REDDIT/GITLAB/...` env pointing at the operator-provisioned sibling site containers, runs
`browser_env/auto_login.py` to mint `.auth` cookies, invokes `run.py --provider openai --model
<served> --temperature ...` over `--test_start_idx/--test_end_idx`, and reads the per-task
programmatic scores into the success rate. Model-under-test wired by pointing the legacy `openai` module at
gbench `/v1` (`openai.api_base`/`OPENAI_API_BASE`). Judge separation is mandatory: the agent and
the fuzzy/ua judge share the legacy `openai` module global, so the fuzzy judge must be
monkeypatched onto the gbench Gemini cascade (upstream `gpt-4-1106-preview` prompts verbatim); else
the model-under-test grades itself.

**External prerequisites (`infra_required`).** The site Docker images are un-bakeable and huge:
`shopping_final`, `shopping_admin_final`, `postmill-populated`, `gitlab-populated` (each multi-GB),
the Wikipedia `.zim` (tens of GB), and especially the map backend (~180 GB); upstream's own
recommended host is a 1 TB AMI. gbench must not pull or bake these; the operator loads them,
exposes the ports, and sets the site URL env vars. `GEMINI_API_KEY` is required only when the selected
task set includes fuzzy/ua tasks.

**In-sandbox feasibility.** A *partial deterministic* smoke is possible: `docker load` the two small
sites (shopping + reddit), run a handful of `url_match`/`program_html` tasks (which need no judge)
with Playwright in the orchestrator against the served model; this exercises the full spine (site
container + accessibility-tree agent loop + programmatic evaluator + model wiring). The full 812
(GitLab + Wikipedia + the ~180 GB map + the fuzzy-judge subset) is not reproducible without the ~1 TB
host, and WebArena is stateful: sites must be reset between task batches or scores silently
corrupt.

**Open decisions.** (a) Rename the suite to `webarena` vs keep `lmarena_web_agent` for continuity.
(b) Full-812 (requires the 1 TB host + stateful reset) vs an honest deterministic site-subset
denominator. (c) Judge: reuse the Gemini cascade for the fuzzy/ua subset (leaderboard_comparable
False) vs a deterministic-only task subset (no judge, no GEMINI). (d) Agent scaffold: canonical
`run.py` CoT accessibility-tree agent (leaderboard-shaped) vs BrowserGym/AgentLab.

**Risks.** Un-bakeable/huge stateful sites; statefulness/reset correctness (a "ran but wrong number"
trap); legacy `openai<1.0` agent/judge self-grading contamination; hardcoded URLs + `.auth` cookies
fragile to host/port drift; map-backend build time (60-90 min).

### #2: `swe_lancer`, run on a disposable Docker host (status: implemented; registered, not in `--evals all`; effort: S, infra only)

**What it measures.** SWE-Lancer (now `openai/frontier-evals` `project/swelancer`; the old
`openai/SWELancer-Benchmark` repository is archived and points there): real freelance Expensify
tasks, 198 IC SWE tasks graded by their hidden end-to-end tests and 265 SWE Manager tasks graded by
the proposal chosen, headlines pass@1 and dollars earned per task type.

**gbench side is DONE.** The earlier blocker (a task image built from Expensify/App at HEAD, which no
longer builds) and the one-shot patch protocol are gone. The suite runs upstream's own agent,
`SimpleAgentSolver`, on both task types through `openai/frontier-evals@51052ced` in the locally built
`gbench-swe-lancer-harness:51052ced` image, with the task images SWE-Lancer publishes
(`swelancer/swelancer_x86_<task>:releasev1`, and `swelancer/swelancer_x86_monolith:releasev1` for
SWE Manager). See `docs/evals/swe_lancer.md`.

**Why it stays out of `--evals all` (host requirements).** The harness edits the Docker host's
`DOCKER-USER` iptables rules while it runs (upstream recommends an ephemeral VM) and pulls about
1.3 TB of task images.

**Registration.** `swe_lancer` is registered but left out of `--evals all` (`NAMED_ONLY_SUITES` in
`gbench/runners/evals.py`): `--evals swe_lancer` runs it and hard-errors `infra_required` naming
whatever is missing (Docker, the docker socket, the `gbench-swe-lancer-harness:51052ced` image, or iptables
access to the host's `DOCKER-USER` chain). Once gbench is routinely run on a disposable host with
room for the images, remove it from `NAMED_ONLY_SUITES` so `--evals all` includes it.

### #3: `ui_control_osworld`, run on a nested-virt host (status: implemented; needs a KVM host; effort: S, infra only)

**What it measures.** Canonical OSWorld (`github.com/xlang-ai/OSWorld`, pinned `fc31a90`): 369
execution-scored desktop computer-use tasks; a CU agent drives an Ubuntu-desktop VM (screenshots +
`pyautogui`) and each task is graded by a deterministic per-task evaluator. No LLM judge.

**gbench side is DONE.** The suite delegates to the pinned OSWorld harness inside a LOCAL orchestrator
image via the docker provider (sibling QEMU-VM container); model wiring is verified against the served
`/v1` (`pong`). It is left out of `--evals all` only because of a host hardware requirement, not
a code gap: it is registered, and `--evals ui_control_osworld` runs it and hard-errors `infra_required`
naming `/dev/kvm` (or any other missing prerequisite) on a host that lacks it.

**The blocker (host hardware).** OSWorld's Ubuntu VM only boots within its 300s timeout with
hardware virtualization (`/dev/kvm`). GCP accelerator instances (A2/A3/G2) cannot enable nested
virtualization, and no container can conjure `/dev/kvm` the host kernel does not expose - so the model
serving box (an H100 A3) can never run it in place.

**Buildable path (no code change).** Run the orchestrator on a nested-virt CPU VM (GCP N2/N2D/C3
with `--enable-nested-virtualization`) and point `--remote-endpoint` at the model still served on the
GPU host - OSWorld needs no GPU of its own. The full split-topology recipe (VM creation, image build,
qcow2 + task image provisioning, firewall) is in `docs/evals/ui_control_osworld.md`. Once gbench is
routinely run on such a host, remove it from `NAMED_ONLY_SUITES` (`gbench/runners/evals.py`) so
`--evals all` includes it.

### #4: `cybergym`, to become CyberGym, execution-verified vulnerability reproduction (status: planned, deregistered; effort: XL)

**What it should measure.** Canonical CyberGym (Wang et al., arXiv:2506.02548; code
`github.com/sunblaze-ucb/cybergym`, data `huggingface.co/datasets/sunblaze-ucb/cybergym`): a
defensive-security benchmark of 1,507 real, already-fixed vulnerabilities in open-source C/C++
projects (1,368 from ARVO, 139 from OSS-Fuzz). For each task an agent works inside the vulnerable
repository and submits one proof-of-concept (PoC) input. The score is decided by execution only:
the PoC counts as a reproduction when it triggers the sanitizer-detected crash on the vulnerable build
and does not on the fixed build. The headline is the reproduction rate over tasks. There is no
LLM judge anywhere in the canonical metric.

**Current state.** Deregistered in this release. The suite that previously shipped under this name
was an LLM-judged text proxy: the model saw only a task's vulnerability description and crash
trace, answered with a root cause plus a PoC or a patch, and a judge graded the text against the
reference patch. That measured something different from CyberGym (text reasoning instead of an
executed reproduction), accepted answers CyberGym would reject (a patch, a PoC that is never run),
and could not be compared with any published CyberGym number, so it was removed rather than shipped
under the benchmark's name. `cybergym` is not in the registry, `--evals all` never runs it, and
`--evals cybergym` exits with an error pointing here. It returns as a runnable suite only when the
canonical harness below lands; no approximate number is published in the meantime.

**Upstream sources to pin.** Code `sunblaze-ucb/cybergym` at `c6fe2027d39471375920b92cf1025e23a99ffda5`
(the revision reviewed when the suite was deregistered; re-check for a newer release before building)
and dataset `sunblaze-ucb/cybergym` at `bde190ded494e52bc684b66073b436c9d992c7c6` (1,507 task
directories). Both pins go into the loader and `canonical_sync.py`, and the harness verifies them at
run time (a mismatch is a hard error, not a warning).

**Task protocol.**
- **Difficulty levels.** CyberGym defines several levels that differ only in which task files the
  agent receives (the repository snapshot, the vulnerability description, the sanitizer crash trace,
  and at the highest level the patch and fixed repository). Implement exactly the level(s) the
  official leaderboard reports, and report each level as its own setting if several are official.
  Files from a higher level must never reach the agent: a leaked patch turns the task into copying.
- **Agent.** Agentic, like the SWE family: the model explores and builds inside an isolated per-task
  container holding the vulnerable repository, then submits a single PoC file through the upstream
  submission interface. Drive the served model through the upstream reference agent setup (the same
  approach as `swe_agentic.py`: the OpenAI-compatible endpoint, the run knobs `--thinking`,
  `--temperature`, `--max-output-tokens`, and a recorded trajectory per task).
- **Network.** None inside the task container. Upstream's FAQ states network access is not needed to
  solve tasks and enabling it invites reward hacking (for example fetching a public reproducer);
  any deviation would have to be disclosed in the result.
- **Official budgets.** Step, time and cost limits exactly as the upstream reference runs specify.

**Scoring.** Use upstream's own verification server and runner logic unchanged: each submitted PoC is
executed against the vulnerable and the fixed build of its task, and the per-task verdict is the
upstream verdict. The headline is the official reproduction rate over the full 1,507 tasks; any
secondary metric the leaderboard reports is added alongside, never substituted. Infrastructure
failures (a runner that did not start, a build that could not be loaded) are harness errors that make
the run `completed_with_errors`, never "not reproduced".

**gbench build plan (mirror the other harness suites).**
1. LOCAL Dockerfiles for the CyberGym verification server and runner, built from the pinned code
   (`docker/cybergym*.Dockerfile`); gbench never pulls a prebuilt benchmark image.
2. A runner module that provisions a per-task agent container (`--network none`), runs the agent,
   collects the submission, sends it to the local verification server, and records the verdict.
   Concurrency is bounded by `--sandboxes`. Model-generated PoC inputs are untrusted data: they are
   only ever executed inside the upstream runner containers, never on the host.
3. Data under a configurable directory (for example `GBENCH_CYBERGYM_DATA_DIR`) holding the task
   dataset and the server's runner data. A missing directory, image or file is `infra_required`
   with exact download and build instructions.
4. Traces carry the trajectory, the agent exit status, the submission's size and hash, and the
   upstream verdict for each build; the result carries the pins, the level(s) run and
   `leaderboard_comparable` computed from them.
5. `docs/evals/cybergym.md` rewritten for the canonical suite (prerequisites, disk, build and run
   commands, how to read the result), `canonical_sync.py` entry, and tests: a stubbed agent and
   stubbed server for the control flow, plus an end-to-end run of one or two real tasks.

**External prerequisites (`infra_required`).** Docker; the task dataset (about 240 GB); the
verification server's runner data (about 130 GB in upstream's binary-only mode, which needs no
compilation, or about 10 TB for the full per-task images with the compilation environment). Plan for
a host with at least 500 GB free for the binary-only mode. No judge key is needed.

**Definition of done.** Everything in section 1, plus: the pins verified at run time; one real task
reproduced end to end through the local server; the full 1,507-task run completing on a provisioned
host with zero harness errors; then re-register `cybergym` and remove it from `ROADMAP_SUITES`
(`gbench/runners/eval_suites/base.py`).

**Risks.** Leaking higher-level files (the patch) to the agent; network access enabling reward
hacking; flaky crash reproduction (follow upstream's own retry and timeout policy, never add one);
very large data and image footprint; running untrusted inputs outside the sandbox (must never happen).

### Backlog: candidate evals (unordered)

New candidates are added here as they are identified. Inclusion criteria: (i) a widely-cited,
actively-maintained benchmark with a reproducible canonical harness; (ii) fills a capability gap
in the current pillars (Reasoning, Coding, Knowledge/QA, Tool-Use & Agentic, Long-Context,
Multimodal Vision & Grounding, Safety); (iii) has a deterministic verifier or a well-specified
judge rubric (avoid benchmarks scorable only by human preference). Each promoted item gets a full
technical spec in this section (like WebArena above) before implementation begins.

## 4. Recently landed (process in action)

A recent harness-build pass produced these canonical-when-provisioned suites, which
double as reference implementations of the patterns above: `gaia2` (in-process ARE simulator, hybrid
deterministic + Gemini-cascade judge), `ui_control_osworld` (DooD QEMU-VM siblings, deterministic
verifier), `wildclawbench` (DooD InternLM harness, in-orchestrator Gemini-cascade judge proxy),
`skillsbench` (DooD BenchFlow, deterministic verifier), `mcp_bench`, `toolbench`,
`multipl_e`, `complexfuncbench` (self-contained). See each suite's `docs/evals/<suite>.md`.

## 5. Metric fidelity refinements (tracked)

These suites already produce a canonical or conservative number for their primary metric, but the
official benchmark defines an additional dimension or a stricter variant that is not yet implemented.
Each entry names the canonical target, the current behavior, and the concrete prerequisite. None is a
skip: the suite reports its primary metric today; these refinements widen fidelity, not enable a
score. A suite listed here still carries its `canonical_sync` stamp (its `method` note points back to
this section). A change here moves the headline number, so each one requires a full dataset re-baseline
at temperature 0.0 before the change lands.

- **`livebench` agentic_coding category.** Current runs all 6 core categories (coding, data_analysis,
  instruction_following, language, math, reasoning) through LiveBench's own `run_livebench.py` and
  ground-truth scorers inside the `gbench-livebench` image, with no LLM judge. The 7th category,
  `agentic_coding`, is excluded (LiveBench's default benchmark set excludes it too). Prerequisite:
  LiveBench's agentic code runner plus its Multi-SWE-Bench task containers (~150 GB).

## 6. Model portability refinements (tracked)

The model registry is fully dynamic (no built-in model table, no model-identity or weights paths in
shipping code, no "model engine"); `vllm-gke.yaml` and the scripts name no model either (the model
and its serving recipe come from their flags). The per-image token count is model-derived: a
model's `vision_tokens_per_image` defaults to None and is read from its config keys or its
processor config (`image_seq_length`, `image_seq_len`, `num_image_tokens`), with the source recorded
in `vision_tokens_source`; the serving and stress pillars use the count the preflight measures on
the server, offline throughput the model's own processor, and the summary prints the recorded
`per_image_tokens` with its source. One multimodal token-budgeting constant remains:

- **`IMAGE_SOFT_TOKENS` estimate (`gbench/runners/eval_suites/base.py`).** Defaults to 1120 (gemma-4's
  max per-image soft tokens), env-overridable via `GBENCH_IMAGE_SOFT_TOKENS`, but with no per-model
  derivation in the eval engine, so an image-heavy prompt for a non-gemma VLM can be mis-estimated
  (mitigated today by a conservative-high default plus a reactive 400/over-context retry).
  Prerequisite: plumb the served model's per-image token count into `_estimate_prompt_tokens`.

## 7. Packaging / operational (parked)

Lower-priority operational polish, tracked but not scheduled.

- **Local container image size.** The per-suite images under `docker/` (SWE-bench family,
  `toolbench`, `putnam_formal` with its multi-GB Mathlib cache, etc.) are large. Parked idea:
  shrink them with multi-stage builds, slimmer bases, and shared layers so a swarm of workers pulls
  less. Not a correctness issue; deferred until it becomes a throughput bottleneck.
- **gbench CLI container image.** The Container Mode in [deployment.md](deployment.md#container-mode-docker-roadmap-not-shipped) (`docker run ... gbench:latest`) is
  a design sketch: no file in this repository builds a gbench CLI image. A real one needs a
  Dockerfile with the CLI as its entrypoint, a documented build command, a choice between bundling
  vLLM for local serving and a lighter client-only image for `--remote-endpoint`, and access to a
  Docker daemon for the quality pillar and the containerised evals (which start their own
  containers). Until then, install gbench and run the CLI.

## 8. Web dashboard and hosted deployment (not shipped)

Tooling, not an eval: a browser front end over gbench's results, a run launcher behind it, and a
Google Cloud deployment of both plus a model-serving service. All of it was taken out of this
release; gbench ships as a CLI. This section records what existed, why it left, and the spec a
re-introduction must meet. Status: removed from the release - effort: L.

**What was removed.**
- **`service/`**, a FastAPI backend started with `uvicorn service.main:app`. Routes:
  `GET /api/health`; `GET /api/models` (the model registry, plus the directories under the local
  models directory, or the top-level prefixes of a GCS models bucket in hosted mode);
  `GET /api/scenarios` (the GemmaClaw `qa/scenarios` tree, read from a depth-1 clone in gbench's
  cache); `GET /api/runs` and `GET /api/runs/{run_id}`; `POST /api/runs` (the launcher);
  `POST /api/runs/{run_id}/cancel`;
  `GET /api/runs/{run_id}/stream-logs` (Server-Sent Events over the run's `master.log`);
  `GET /api/analytics/summary` and `GET /api/analytics/{run_id}` (the quality pass rate over time,
  per-scenario pass/fail, and a compact serving/throughput block); and a catch-all route that served
  the built SPA from `ui/dist`. Two modes, chosen by `GBENCH_ENV`: `local` (the default; results
  under `results/run_<8 hex>/`, CORS open) and `prod` (results uploaded to a GCS bucket, the model
  endpoint resolved from a Cloud Run service). It read `GBENCH_RESULTS_BUCKET`,
  `GBENCH_MODELS_BUCKET`, `GBENCH_REMOTE_ENDPOINT`, `GBENCH_LOCAL_REMOTE_ENDPOINT` (default: an
  Ollama port on localhost), `GEMMACLAW_CACHE_PATH`, `VLLM_SERVICE_NAME`, `GCP_PROJECT` and
  `GCP_REGION`, and, in the maintainers' copy (see "Where the code is"), `GBENCH_CACHE_DIR` as the
  base directory of its GemmaClaw clone. The `[ui]` extra installed `fastapi`, `uvicorn`,
  `google-cloud-storage` and `google-api-python-client`; nothing else in gbench imports them.
- **The launcher.** `POST /api/runs` started `python -u -m gbench.cli ...` with
  `subprocess.Popen` on whatever host ran the service, one run at a time (409 while another run was
  live, tracked in `active_run.json`). The child's stdout and stderr went to the run's
  `master.log`; when it exited, in `prod` mode, the run folder was uploaded to GCS. Only the
  maintainers' copy opens `master.log` for appending and passes the run folder through
  `redact_tree` before the upload.
- **`ui/`**, a React 19 + Vite 8 + MUI single-page app (Recharts charts, axios calls to same-origin
  `/api/*`, a Vite dev proxy to port 8000). Pages: Dashboard (`/`), New Evaluation
  (`/new-evaluation`, a GemmaClaw quality-run form: model, tags, scenarios), Compare (`/compare`)
  and Analytics (`/analytics`).
- **`deploy/`**, the dashboard image and its deploy scripts. `deploy/Dockerfile`: a Node 22 stage
  built the SPA; a Python 3.11 stage added Node 22, pnpm and a depth-1 GemmaClaw clone, installed
  gbench with `--no-deps`, set `GBENCH_ENV=prod` and ran uvicorn on port 8080. `deploy.sh` and
  `cloudbuild.yaml` built and pushed the image to Artifact Registry, then ran `terraform init` and
  `terraform apply`.
- **`terraform/`**, a `google-beta` stack with a GCS state backend: an Artifact Registry
  repository; `gbench-eval-service`, a Cloud Run service on one NVIDIA L4 that ran `vllm serve` from
  a `gemma-serving` image that nothing in the repository built (an unreferenced `docker/Dockerfile`,
  a wrapper over the Vertex AI vLLM serving image, was apparently meant for it; also removed), with
  `roles/run.invoker` granted to a configured list; and
  `gbench-dashboard`, a Cloud Run v2 service with IAP enabled for a configured list of members.
- **`.github/workflows/deploy.yml`**, a push-to-main job that authenticated with Workload Identity
  Federation, installed gcloud and terraform 1.8.5, and ran the deploy script.
- **The quality pillar's `GBENCH_ENV=prod` path**, which took the GemmaClaw commit from the
  dashboard image's baked-in checkout (`HEAD`, no fetch) instead of the pinned commit. The quality
  pillar now always resolves the pinned `gemmaclaw_commit`, or uses `--gemmaclaw-path` as given.

**What stays.** Everything the CLI does: `--remote-endpoint` against a model server that speaks
the OpenAI-compatible `/v1` API, `--stage-to-gcs`, and the results directory with its credential
redaction (`summary.json`, `metadata.json` with the scaffold conditions and tags, the per-pillar
JSON files, `eval_summary.csv`, traces and logs). A future viewer reads exactly these files. On
Cloud Run, only a service that allows unauthenticated calls can be measured today. The endpoint
check that starts every remote run sends no token, so an IAM-protected service (Cloud Run's
default) stops the run there; the shared eval and performance request paths send none either.
`GBENCH_GCP_AUTH=1` makes the golden and quality pillars fetch a GCP identity token for a
`*.run.app` endpoint, but a run never reaches them. Sending that token from the endpoint check and
the shared eval and performance request paths is open CLI work, separate from this section.

**Why it left the release.**
1. **The launcher could not run anything useful.** In local mode, and in hosted mode whenever an
   endpoint was configured, every request exited at argument parsing with status 2: it never
   passed the `--tokenizer` that `--remote-endpoint` requires, and it passed `--format`, which the
   CLI rejects together with `--remote-endpoint` unless the value is `both` (the UI always sent
   `hf`). A performance run was mapped to `--no-quality`, which the CLI does not define, so it
   failed in every mode. In hosted mode with no endpoint configured, a quality or `all` run parsed
   and fell through to starting a local vLLM server on the dashboard host.
2. **Unsafe input handling.** The committed SPA catch-all joined the request path onto `ui/dist`
   without a containment check, so an encoded `..` read any file the service could read
   (`GET /..%2F..%2Fpyproject.toml` returned the repository's own file). In local mode the service
   was unauthenticated, CORS reflected any origin with credentials, and request fields (`model_id`,
   `format`, `tags`) went into the gbench argv unvalidated, so any web page the operator visited
   could start a run with injected options. `{run_id}` path parameters accepted `..`, which made the
   service walk the whole repository and fail with a 500. Cancel signalled only the gbench parent,
   so GemmaClaw's children kept running, and its restart fallback signalled a PID read from
   `active_run.json` without checking which process it was.
3. **The deploy paths could not complete.** `deploy.sh` and `cloudbuild.yaml` ran `docker build .`
   at the repository root, where there is no Dockerfile (it was `deploy/Dockerfile`, and the ignore
   files in `deploy/` were never applied). Both pushed the image before terraform created the
   Artifact Registry repository. The workflow called `./deploy.sh` (the script was
   `deploy/deploy.sh`) without the required state bucket, and because the access lists live in a
   gitignored file, a CI apply against shared state would have planned their IAM bindings for
   destruction. `cloudbuild.yaml` did not pass the region to terraform, and neither script passed
   the repository or service-name overrides (terraform hard-coded the repository id and the image
   names). The dashboard ran as the default compute service account, with no `run.invoker` on
   the serving service and no `GBENCH_GCP_AUTH`, so it could not call its own model endpoint. The
   L4 was set as a template annotation instead of the `node_selector` Cloud Run reads. The serving
   service account and image were created and documented nowhere, and the service served one
   hard-coded model name.
   The image baked an unpinned GemmaClaw `HEAD` while run metadata recorded the pinned commit.
4. **Docs and CI did not match the code.** The documented start command (`python -m service.main`)
   exited at once; the `[ui]` extra and the service variables were undocumented; the README
   described pages and charts that did not exist; the model picker listed bucket prefixes that did
   not match the layout `--stage-to-gcs` writes; the scenarios endpoint seeded a depth-1 GemmaClaw
   clone in the quality runner's cache, which a pinned checkout can fail on; and no CI job built or
   linted the SPA or validated the terraform.
5. **The hosting target no longer fits.** The quality pillar runs GemmaClaw's QA suite in gbench's
   own `gbench-quality` Docker container (built on first use, run with `--network none`), so a
   launcher must run on a host that can start containers. A Cloud Run container cannot, so a
   Cloud Run-hosted launcher would fail every quality run at the Docker preflight. Viewing results
   needs no Docker at all.

**Where the code is.** Not in this repository; this section is meant to be the self-contained spec
for rebuilding it. The maintainers keep a copy of the removed files. That copy also has fixes that
were never released: the SPA route's path containment check; `master.log` opened for appending,
streamed redacted while the run is live, and passed through `redact_tree` with the rest of the run
folder before the upload; a withheld `pass_rate` kept `null` by both analytics endpoints and shown
as withheld on the Analytics page; and `GBENCH_CACHE_DIR` honoured for the scenarios clone. Every
fix but the last has tests there (the SPA containment tests in the service's own tests, and the
stream redaction, redact-before-upload, withheld `pass_rate` and Analytics page checks taken out of
gbench's shared test files); the `GBENCH_CACHE_DIR` change has none. Every other problem listed above is still open in that copy, so it is reference
material, not something to restore as it is.

**Re-introduction spec.** Viewer first, launcher second, hosting last; each step ships on its own.
1. **Read-only results viewer.** Serves gbench's result files as the CLI writes them and nothing
   else: no subprocess, no Docker, no gbench import beyond reading. One configured results root;
   every path resolved and rejected if it falls outside the root; `run_id` checked against a strict
   pattern on every route; static files served by a containment-checked handler (for example
   Starlette's `StaticFiles(html=True)`). A withheld `pass_rate` (`null`) is shown as withheld,
   never as 0; `leaderboard_comparable` and its reason are shown next to each number; the
   performance pillars are shown as well as quality. Model outputs and traces are rendered as text,
   never as HTML.
2. **Launcher.** A typed request (enums for run type and preset; allowlisted characters for model
   ids, tags and scenario ids; model ids may not start with `-`; values passed as `--opt=value`).
   The argv is built from and checked by `gbench.cli.create_parser()` plus the CLI's own post-parse
   validation before any process starts, and a contract test feeds every request shape through that
   path, so a CLI change that breaks the launcher fails CI. The child starts in its own session and
   cancel signals the whole process group; after a restart, a PID's command line is checked before
   it is signalled. GemmaClaw is set up through the quality runner's own code (the pinned commit,
   under its lock), never a second clone. `master.log` is streamed redacted, and the docs say that
   tokens a run registers at run time (such as `GBENCH_GCP_AUTH` identity tokens) are unknown to the
   service process.
3. **Authentication and authorization.** Local: bind to 127.0.0.1 by default, CORS limited to the
   dev origin without credentials, and a per-session token (or equivalent) on every
   state-changing route. Hosted: an identity-aware proxy in front (IAP or equivalent), launcher
   routes limited to an explicit group, and a dedicated service account holding only what it needs
   (`run.invoker` on the model endpoint, object access on the results bucket).
4. **Hosting.** A viewer can run anywhere, Cloud Run included. A launcher that runs the quality
   pillar needs a Docker host: a VM (for example GCE) or a GKE node pool with a Docker daemon it may
   use; not Cloud Run. The model-serving stack stays separate from the dashboard stack, and gbench
   reaches it only through `--remote-endpoint`.
5. **Deployment.** `docker build -f <dashboard Dockerfile>` with the ignore rules at the build
   context's root; bootstrap order (registry, then build and push, then the full apply); every
   override passed through to terraform; the GPU requested with `node_selector` (or
   `google_cloud_run_v2_service`'s `node_selector.accelerator`); service accounts created in
   terraform; GemmaClaw pinned to the commit gbench records, with a hard error on a mismatch; a CI
   workflow with real paths and a real state bucket that refuses to apply with empty access lists;
   the `.gitignore` rules for `.terraform/`, `*.tfstate*` and `*.tfvars` kept (they are still in
   this release, for clones that have files from the removed deployment).
6. **Tests, CI and docs.** Service tests for the argv contract, path containment (including an
   encoded `..`), `run_id` validation, CORS and cancel; `npm ci && npm run lint && npm run build`
   and `terraform fmt -check && terraform validate` in CI; a `docs/deploy.md` covering the required
   APIs, buckets, IAM, the start command, every service variable and the extra to install; and a
   README that describes only what ships.

**Definition of done.** Every spec item implemented and tested; every problem under "Why it left
the release" closed, with a regression test wherever one can be written; the viewer, the launcher
and the deployment each documented and exercised end to end, including a quality run launched from
the dashboard on a Docker host, scored, and shown with its pinned GemmaClaw commit. Then the
README's roadmap pointer is replaced by real documentation.

**Open decisions.** (a) Ship the viewer alone first, or wait for the launcher. (b) A local launcher
only, or a hosted one too (and then GCE or GKE). (c) Whether a model-serving stack belongs in gbench
at all, or only in the docs, since gbench measures endpoints it does not own.

**Risks.** A launcher executes commands by design: any gap in argv validation or authentication is
remote code execution on the host. Hosted mode multiplies the credentials in play (Gemini keys for
the quality pillar and judged suites, endpoint tokens), which must stay out of images, logs and the
browser. A viewer that trusts result files can be fed a crafted `metadata.json`; parse strictly.
