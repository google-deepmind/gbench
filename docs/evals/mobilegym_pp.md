# mobilegym_pp setup

Canonical **MobileGym++** ([github.com/OpenMobile2/MobileGym-plusplus](https://github.com/OpenMobile2/MobileGym-plusplus),
commit `cdde4ad06ed67176b56964dd329b8cb8e0b0001c`), the 215-task hybrid app-tools + GUI mobile
benchmark (`bench215/official.json`). `mobilegym_pp` runs the 215 official tasks with the canonical
`Qwen35SessionAgent` hybrid session runtime (`interaction_mode="hybrid"`,
`coord_space="norm_0_1000"`, `judge_mode="state"`, `sample_seed=42`).

Every task is scored deterministically by its programmatic Python/Playwright state verifier in
`bench_env/tasks/` (`judge_mode="state"`), with zero LLM/VLM judge calls and no external API keys
required (`GEMINI_API_KEY` is never read).

For the companion 256-task pure-GUI benchmark (**MobileGym**), see
[mobilegym.md](mobilegym.md). Both suites share the locally built
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

## Running

```bash
# Full 215-task canonical MobileGym++ evaluation
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-26B-A4B-it \
       --evals mobilegym_pp --max-output-tokens 4096

# Difficulty-stratified subset (--eval-limit) with reasoning enabled
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-26B-A4B-it \
       --evals mobilegym_pp --eval-thinking --eval-limit 20 --max-output-tokens 32768
```

## Protocol and gbench wiring

- **Canonical 215-task manifest (`bench215/official.json`)**: loads the 215 tasks and their fixed
  instruction templates from `bench215/official.json` by overriding `task.templates =
  [official_instruction]` while keeping `task._instruction_override = None`, so `task.reset(seed)`
  runs its full parameterized state setup (`random.seed(seed)`, target selection, and environment
  seeding) while presenting the exact official instruction string.
- **Hybrid session agent (`Qwen35SessionAgent`, `interaction_mode="hybrid"`)**: exposes both
  structured app tools (`list_apps`, `list_app_tools`, `call_app_tool`, plus per-app tools) and GUI
  actions (`click`, `double_click`, `long_press`, `type`, `swipe`, `scroll`, `press_back`,
  `press_home`, `open_app`, `wait`, `finish`, `fail`).
- **Model family routing for Gemma**: `Qwen35SessionAgent._detect_model_family` routes `"gemma"`
  model names to the `"gemini"` session family (`session_history_mode="FunctionCall"`), which keeps
  pure text assistant turns (`content` + `tool_calls`, no assistant-turn images) and sends each new
  screenshot in a fresh user turn after the `role="tool"` output turns, matching Gemma 4's chat
  template constraints (`tools` declared in the request, text-only assistant messages, and images in
  user turns).
- **Frozen simulator clock (`2026-07-12 09:00:00`)**: both `SIMULATED_TIME` in
  `os/data/simulatorConfig.ts` and Python's `default_tomorrow()` in `bench_env/tasks/base.py`
  (`2026-07-13`) are pinned to `2026-07-12 09:00:00`.
- **Metrics**: headline `accuracy` (0-100%) is the task Success Rate (`SR = passed / n_scored *
  100`), reported alongside `progress_rate`, `false_complete_rate`, `overdue_termination_rate`,
  `unexpected_side_effect_rate`, `difficulty_breakdown`, `app_count_breakdown`, and per-task
  `tool_calls` / `app_tool_calls` traces.

## Environment knobs

- `GBENCH_MOBILEGYM_IMAGE`: local Docker image tag (default `gbench-mobilegym:5fbd3f34`).
- `GBENCH_MOBILEGYM_WORKERS`: parallel Playwright browser workers inside the container (default:
  `min(sandboxes, 32)` when `--sandboxes` is set, else `min(concurrency, 16)`).
- `GBENCH_MOBILEGYM_PP_TEMPERATURE`: sampling temperature override (default: `0.0` without
  `--thinking`, `1.0` with `--thinking`).
- `GBENCH_MOBILEGYM_TIMEOUT`: container wall-clock timeout in seconds (default `86400`).

