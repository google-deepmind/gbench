# mobilegym setup

Canonical **MobileGym** ([github.com/Purewhiter/mobilegym](https://github.com/Purewhiter/mobilegym),
commit `5fbd3f34be0e9780ae4b1e9d5089963ddbd05d38`), a multi-app mobile GUI environment and
benchmark across 24 simulated web applications. `mobilegym` runs the 256-task held-out test split
(`bench_env/splits/test.txt`) with the canonical `GenericAgentV2` pure-GUI agent
(`coord_space="norm_0_1000"`, `eval_mode="grounded"`, `judge_mode="state"`, `sample_seed=42`).

Every task is scored deterministically by its programmatic Python/Playwright state verifier in
`bench_env/tasks/` (`judge_mode="state"`), with zero LLM/VLM judge calls and no external API keys
required (`GEMINI_API_KEY` is never read).

For the companion 215-task hybrid app-tools + GUI benchmark (**MobileGym++**), see
[mobilegym_pp.md](mobilegym_pp.md). Both suites share the locally built
`gbench-mobilegym:5fbd3f34` image (`gbench/docker/mobilegym.Dockerfile`).

## Prerequisites

1. **Docker** reachable on the host (`docker info`).
2. **Cloud metadata server firewall rules** on the Docker `DOCKER-USER` chain
   (`container_guard.require_metadata_blocked`), blocking `169.254.169.254/32` (IPv4) and
   `fd20:ce::254/128` (IPv6) on non-DNS TCP ports:
   ```bash
   sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP
   sudo ip6tables -I DOCKER-USER -d fd20:ce::254/128 -p tcp ! --dport 53 -j DROP
   ```
3. **Host inotify instance ceiling** of at least `1024` (`8192` recommended so Playwright Chromium
   and Vite watchers do not exhaust low default limits of 128):
   ```bash
   sudo sysctl -w fs.inotify.max_user_instances=8192
   ```
4. **Locally built `gbench-mobilegym:5fbd3f34` image** (gbench never pulls it):
   ```bash
   docker build -t gbench-mobilegym:5fbd3f34 -f gbench/docker/mobilegym.Dockerfile gbench/docker
   ```
   The image carries `LABEL org.gbench.mobilegym-launcher=2`, both pinned repositories
   (`Purewhiter/mobilegym@5fbd3f34be0e9780ae4b1e9d5089963ddbd05d38` and
   `OpenMobile2/MobileGym-plusplus@cdde4ad06ed67176b56964dd329b8cb8e0b0001c`), the SHA256-verified
   offline data bundle (`mobilegym-data-v0.1.0.tar.gz`,
   `sha256:933a8499c71450f135eda70f9107af72b56b42181fc62de76ed1aa4c513a603c`), pre-built offline
   Map assets, Playwright Chromium, and CJK/Emoji fonts.

## Running

```bash
# Full 256-task canonical MobileGym evaluation
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-26B-A4B-it \
       --evals mobilegym --max-output-tokens 4096

# Difficulty-stratified subset (--eval-limit) with reasoning enabled
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-26B-A4B-it \
       --evals mobilegym --eval-thinking --eval-limit 20 --max-output-tokens 32768
```

## Protocol and gbench wiring

- **Canonical split and agent**: runs the 256-task test split (`bench_env/splits/test.txt`,
  disjoint from the 38-task `val` split) with `GenericAgentV2` (`coord_space="norm_0_1000"`,
  `eval_mode="grounded"`, `judge_mode="state"`, `sample_seed=42`). Each task uses its own canonical
  `max_steps` (`task.max_steps`).
- **Frozen simulator clock (`2026-07-12 09:00:00`)**: both the React simulator (`SIMULATED_TIME` in
  `os/data/simulatorConfig.ts`) and Python's `default_tomorrow()` in `bench_env/tasks/base.py`
  (`2026-07-13`) are pinned to the fixed date `2026-07-12 09:00:00`, eliminating wall-clock drift on
  relative-date tasks (`tomorrow` in Ctrip / Railway12306).
- **Offline execution**: the container runs on Docker's `bridge` network with outbound traffic
  restricted by in-container `iptables` (`mobilegym_entrypoint.sh`) to loopback (`127.0.0.1`) and
  the host bridge gateway (`EndpointRelay` and `DetokenizeRelay`). Offline Map tiles/routes and the
  offline Railway12306 station-stop fallback (`trains_stops.json`) are served locally inside the
  container.
- **Reasoning and server-parser recovery**: `GenericAgentV2._parse_llm_output` strips closed
  `<think>...</think>` blocks and unclosed `<think>` prefixes before extracting `Thought:` and
  `Action:` lines, and falls back to `reasoning_content` / `reasoning` when `content` is empty.
  Every model request sends `extra_body={"return_token_ids": True, "chat_template_kwargs":
  {"enable_thinking": <bool>}}` and applies gbench's `recover_server_parser_text` /
  `server_parser_residue` policy via `DetokenizeRelay`.
- **Metrics and harness error exclusion**: the headline `accuracy` (0-100%) is the task Success Rate
  (`SR = passed / n_scored * 100`) where `passed = goal_success and clean`. Secondary metrics
  `progress_rate`, `false_complete_rate`, `overdue_termination_rate`,
  `unexpected_side_effect_rate`, `difficulty_breakdown` (`L1`, `L2`, `L3`) and
  `app_count_breakdown` (`1_app`, `2_app`, `3_app`, `4_app`) are recorded on the result. Tasks that
  hit an infrastructure or endpoint error (`_wait_ready` timeout, browser crash, HTTP 5xx, connection
  error, or `server_parser_text_lost`) are marked with `harness_error` and `scoring_excluded=True`,
  excluded from the accuracy denominator, and set `status="completed_with_errors"`.

## Environment knobs

- `GBENCH_MOBILEGYM_IMAGE`: local Docker image tag (default `gbench-mobilegym:5fbd3f34`).
- `GBENCH_MOBILEGYM_WORKERS`: parallel Playwright browser workers inside the container (default:
  `min(sandboxes, 32)` when `--sandboxes` is set, else `min(concurrency, 16)`).
- `GBENCH_MOBILEGYM_TEMPERATURE`: sampling temperature override (default: `0.0` without
  `--thinking`, `1.0` with `--thinking`).
- `GBENCH_MOBILEGYM_TIMEOUT`: container wall-clock timeout in seconds (default `86400`).

