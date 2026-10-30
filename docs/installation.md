# Installation

The full installation guide: requirements, a virtual environment or conda, and the optional installs per use (remote endpoints, local serving, eval suites, tests).

[Back to the README](../README.md)

## Cross-platform requirements

gBench is cross-platform and supports both client-only orchestration workstations and GPU serving hosts.

| Component | Minimum Specification | Recommended Specification | Supported Operating Systems |
| :--- | :--- | :--- | :--- |
| **Client Orchestration Mode** | 2 CPU Cores, 4GB RAM | 4 CPU Cores, 8GB RAM | Linux, macOS (Apple Silicon M1-M4 / Intel), Windows (WSL2 / Native Python) |
| **Local GPU Serving Mode** | 1x NVIDIA GPU (16GB VRAM) | 8x NVIDIA H100/L4 GPUs | Linux OS (Ubuntu 22.04+, Debian 12+, RHEL 9+) |
| **Python Environment** | Python 3.11 | Python 3.12 (tau2-bench and Harbor, which need 3.12, run in their own local images; gBench itself needs 3.11 or later) | All supported OS environments |
| **CUDA Driver** | CUDA 12.1 | CUDA 12.4+ / Driver 535+ | Linux GPU hosts |

### Virtual environment installation

```bash
# Clone repository
git clone https://github.com/google-deepmind/gbench.git
cd gbench

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the lightweight base package (enough for laptops and remote endpoints such as Ollama)
pip install -e .
# With --remote-endpoint, also install the tokenizer library (torch is not needed): the serving
# and stress pillars size their prompts with the served model's tokenizer (--tokenizer), and the
# Golden Set's chat-template cases and the mmlu, wmdp, ruler and longbench_v2_code suites read it;
# all of them hard-error without it
pip install transformers

# For LOCAL GPU serving (a run without --remote-endpoint), provision a vLLM serving stack
# for your hardware YOURSELF - gbench does not install vllm/torch, to avoid overwriting a
# validated CUDA build. On a fresh machine `pip install vllm` pulls a matching torch/CUDA;
# otherwise use your existing build. (Runs against --remote-endpoint need neither, except
# real-dataset workloads: `--campaign chat-like` (ShareGPT), `--dataset custom` and `--dataset hf`
# are loaded with vLLM's dataset loaders.)
pip install vllm

# OR install the extra dependencies for execution-based eval suites
# (pycocoevalcap, swebench, docker, multi-swe-bench, mini-swe-agent; each suite
#  hard-errors with an install/provision hint if its dependency is absent)
pip install -e ".[evals]"

# terminal_bench runs Harbor 0.20.0 in its local image (docs/evals/terminal_bench.md); nothing
# to install here:
#   docker build -t gbench-terminal-bench:harbor-0.20.0 -f gbench/docker/terminal_bench.Dockerfile gbench/docker

# OR install what the unit tests need, in a test environment (then run `pytest`; see
# "Developer guide" in ../README.md and tests/README.md)
pip install -e ".[dev,evals,omnidocbench]" transformers
python -m nltk.downloader punkt punkt_tab
```

> **Notes on eval-suite dependencies**
> - Running the **test suite** requires the `[dev]` extra (`pytest-asyncio` powers the two
>   async claim-scoring tests in `tests/test_search_tool.py`). Bare `pytest` alone leaves those
>   2 tests erroring. The `[dev]` extra also carries `bfcl-eval==2026.3.23`, only for the BFCL
>   unit tests: the BFCL suites run it in the local image gbench-bfcl:2026.3.23. It pins
>   `numpy==1.26.4` and `filelock==3.20.0`, so install the test extras in a test environment,
>   not in the one that serves your models (tests/README.md). The unit tests also import
>   `transformers` (the serving stack's tokenizer library, which gBench does not declare) and
>   need nltk's punkt data; the system tools they exercise are bubblewrap, a Java runtime and
>   Docker. The `omnidocbench` extra carries the TEDS and formula edit-distance metrics that the
>   OmniDocBench unit tests exercise; the suite itself scores in its evaluator image and does not
>   need it.
> - The **SWE family** (`copilot_bench_swe`, `swe_bench_multilingual`, `swe_bench_live`,
>   `multi_swe_bench`, `swe_bench_pro`) runs the agentic protocol by default.
>   `swe_bench_live` runs every tab of the SWE-bench-Live leaderboard (Lite, Full and Verified;
>   C/C++, C#, Java, TS/JS, Go and Rust from SWE-bench-Live/MultiLang, scored by upstream's
>   multi-language harness in the same local image; Windows, which runs on a Windows Docker host
>   only after its own cloud metadata server check (a PowerShell probe in one of its instance
>   images) passes and is reported `infra_required` in its own entry otherwise; a tab whose
>   prerequisite is missing is reported `infra_required` in its own entry while the others run, in
>   both directions: the Windows tab off a Windows Docker host, the Linux tabs on one; the suite
>   hard-errors when no selected tab can run or when the metadata server check of the Linux tabs
>   fails), each with its own headline.
>   `swe_bench_live` rolls out with SWE-agent-for-eval (the SWE-agent fork its evaluation
>   README names) in the local image `gbench-sweagent-for-eval:82422422-rex3` (Win-Agent on the Windows
>   tab), and `multi_swe_bench`
>   with MSWE-agent in the local image `gbench-mswe-agent:88217624`. `copilot_bench_swe` and
>   `swe_bench_multilingual` run mini-swe-agent 2.4.2 / 2.4.6 in the local image
>   `gbench-mini-swe-agent-v2:2.4.2-2.4.6` and score
>   with swebench 5.0.2 in `gbench-swebench-v5:5.0.2`; `swe_bench_pro` runs SWE-bench Pro V2 under
>   Harbor 0.22.0 in `gbench-swe-bench-pro-v2:harbor-0.22.0-66f92766-sidecar-rt` (build commands in each suite doc).
>   `copilot_bench_swe`, `swe_bench_multilingual`, `swe_bench_live` and `multi_swe_bench` pull
>   their instance images with the host's Docker CLI (your `docker login` applies) before a
>   scaffold or harness container starts task containers from them; those containers get no
>   Docker client configuration.
>   `GBENCH_<SUITE>_MODE=oneshot` restores the single-request path, which is never
>   leaderboard-comparable.
> - The **Agentic Quality suite** (`--quality-only`) needs **Docker** (the CLI and a daemon this
>   user can reach), git, a Gemini API key (`GEMINI_API_KEYS` or `GEMINI_API_KEY`) and gBench's
>   source checkout (for `docker/quality.Dockerfile`): its QA suite always runs in gBench's local
>   `gbench-quality` container, which gBench builds on first use from the GemmaClaw commit
>   (GemmaClaw installed and built inside it, on GemmaClaw's own Node v22.19-or-newer base image;
>   several minutes and about 8 GB, cached per commit). No Node.js, pnpm or bubblewrap is needed on
>   the host for it. Eval (`--evals`), golden (`--golden`) and performance runs do not need the
>   quality pillar's prerequisites; execution-based evals list their own in the note below.

> **Note on execution-based evals.** Suites that run code/patches in a sandbox
> (`bigcodebench`, `multipl_e`, the SWE-bench family, `spider2`, `ojbench`, ...) need
> the `[evals]` extra plus, in some cases, Docker or extra assets; see each suite's
> guide under [`docs/evals/`](evals/). The competition-math suites graded by MathArena's
> own grader (`hmmt`, `aime`) need Docker and the locally built `gbench-matharena:b89f2f0a` image.
> `terminal_bench` runs Harbor 0.20.0 in the local image `gbench-terminal-bench:harbor-0.20.0`
> (`docker build -t gbench-terminal-bench:harbor-0.20.0 -f gbench/docker/terminal_bench.Dockerfile gbench/docker`;
> gBench's own interpreter needs no Harbor). LLM-judged suites (`simpleqa`, `frames`,
> `browsecomp`, `humanitys_last_exam`, `putnam_formal` (no-answer answer review), `acebench` (agent_multi_turn user simulator; without the key only that category is reported infra_required and the other categories run), ...) require `GEMINI_API_KEY` (or a key list in
> `GEMINI_API_KEYS`) and **hard-error** (never skip) if it is absent, so a judged suite is
> never silently dropped or scored on fabricated data.

> **Note for cloud VMs (GCE and other hosts with a metadata server).** A docker container on a
> bridge or user-defined network, or on the host network, can reach the VM's metadata server at
> `169.254.169.254`, which hands out the VM's service-account token. The suites that start a
> container running model-controlled code with network access call gBench's guard
> (`require_metadata_blocked`) first: `mcp_atlas`, `mcp_bench`, `wildclawbench`, `bfcl`,
> `bfcl_v4_agentic`, `tau2`, `tau3`, `livebench`, `aider_polyglot`, `bigcodebench`, the SWE family,
> `terminal_bench`, `skillsbench`, `swe_lancer` and `ui_control_osworld` (the list, with the
> network each one uses, is in toolchains.md). The guard probes that network with a short-lived
> container of the suite's own local image (a TCP connection to `169.254.169.254:80`, and to its
> IPv6 address `[fd20:ce::254]:80`, opened and closed, no request sent) and hard-errors
> `infra_required` if a connection opens; a suite can also have the same container resolve the
> host names its containers need, and a name that does not resolve is a hard error too. Block the
> metadata server once on the host; gBench never runs sudo or changes the host's firewall itself:
>
> ```bash
> sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP
> sudo apt-get install -y iptables-persistent && sudo netfilter-persistent save   # keep it
> ```
>
> On GCE the VPC DNS resolver is the same address, so the rule spares port 53 (a plain `-j DROP`
> cuts DNS in every container). It covers docker's bridge networks (the default bridge and
> user-defined bridge networks), not the host network, which Docker's `DOCKER-USER` chain never
> sees: on a cloud VM a container that runs model-controlled code on the host network hard-errors
> until it runs on a bridge or user-defined network (see the suite's guide). `--network none`
> needs no probe. See
> [docs/evals/toolchains.md](evals/toolchains.md#cloud-vms-the-metadata-server).

### Conda Environment Installation

```bash
conda create -n gbench-env python=3.11 -y
conda activate gbench-env

# Lightweight installation (Remote endpoints / Ollama), plus the tokenizer library the Golden
# Set's chat-template cases and the mmlu, wmdp, ruler and longbench_v2_code suites need
pip install -e .
pip install transformers

# For local GPU serving, provision a vLLM stack for your hardware yourself (gbench does not
# install it, to avoid overwriting a validated CUDA build):
pip install vllm
```
