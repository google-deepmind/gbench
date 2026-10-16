# Optional toolchains

Most suites need nothing beyond the base install (`pip install -e .` in the gbench checkout;
gbench is installed from its source checkout, not from PyPI). The ones below execute code,
drive containers, or wrap a third-party harness, and **hard-error** when their
prerequisite is missing: the error names the exact missing piece and points here.
Nothing is scored on a partial environment.

Install only what you intend to run. `gbench --list evals` names every suite;
`gbench --dry-run --evals <name> --max-output-tokens 65536 --models <hf-id> --num-gpus 1`
(add `--tool-call-parser <parser>` for a suite `--list evals` marks `[tools]`) shows what a
run would execute without generating.

## Python extras

| Install | Needed by |
| --- | --- |
| `pip install -e ".[evals]"` (from the checkout) | pulls `docker`, `pytest`, `swebench` 4.1 (the main env; copilot_bench_swe and swe_bench_multilingual score with swebench 5.0.2 in the local image `gbench-swebench-v5:5.0.2`), plus `mini-swe-agent>=1.15,<2` (the in-process 1.x runner of swe_agentic.py; copilot_bench_swe and swe_bench_multilingual run 2.x in `gbench-mini-swe-agent-v2:2.4.2-2.4.6`, swe_bench_live and multi_swe_bench run their reference scaffolds in their own images) with `litellm` (mini-swe-agent's model client, checked by the agentic SWE runner), `multi-swe-bench==1.1.2` (`multi_swe_bench`), `google-genai` (the Gemini judge cascade and every other Gemini model role), `openai>=2.31` + `tiktoken` (`mmlu_pro`, `mrcr`), `pycocoevalcap` (`coco_caption`), `pycocotools==2.0.11` (`bundled_detection`), `sacrebleu==2.6.0` + `portalocker` (`i18n_translate`; portalocker is its cache lock), `h5py` + `scipy` + `sympy` + `matplotlib` (`scicode`; one of its test problems imports mpl_toolkits), `math-verify` + `latex2sympy2_extended` (`math_equiv`), `pyyaml` (`hmmt` / `aime` read MathArena's vendored configs; `cimemories` reads the release's eval/prompts.yaml), `loguru` + `regex` (the vendored MathArena grader and parser of `hmmt` / `aime`), `rapidfuzz` (`omnidocbench`), `tree-sitter==0.21.3` + `tree-sitter-python==0.21.0` (`gorilla_apibench`), `scikit-learn` (`nestful`), `openpyxl` + `pymupdf` + `xlrd` (`gaia`; openpyxl and pymupdf also `gdpval`), `python-pptx` + `python-docx` + `soundfile` (`gdpval`), `json_repair` (the mcp_bench judge helper the host imports), `langdetect` + `nltk` (`ifeval`, and `nltk` for `causalbench`; then `python -m nltk.downloader punkt punkt_tab`), `jsonschema` (`mcp_atlas`), `networkx` (`t_eval`) and `agentdojo==0.1.35` (`agent_dojo`, which runs in-process); `datasets` and `pandas` come with the base install; covers most of this table |
| `pip install multi-swe-bench==1.1.2` | `multi_swe_bench` (any other version hard-errors) |
| `pip install sentence-transformers networkx` (networkx also in the `evals` extra) + torch from your serving stack | `t_eval`: the vendored T-Eval evaluators (BERTScore embedders all-mpnet-base-v2 / gte-large-zh and the plan graph matching); checked before the first request, infra_required naming a missing package; not needed to list or run other suites. See [t_eval.md](t_eval.md) |
| `pip install pycocoevalcap` (already in the `evals` extra) + a Java runtime (`apt install default-jre-headless`) | `coco_caption` (PTB tokenizer, BLEU-1..4, METEOR, ROUGE-L, CIDEr; the tokenizer and METEOR run Java). The suite runs every scorer on a small fixture first and hard-errors (infra_required) if the package, Java or a scorer is missing |
| `pip install rapidfuzz` (already in the `evals` extra) | `omnidocbench` interim per-page pass flag; the suite hard-errors (infra_required) without it. Its reported scores come from the evaluator image below, not from this package |
| `pip install sacrebleu==2.6.0` (in the `evals` extra) | `i18n_translate`: sacrebleu's `wmt19` test-set definition (the suite downloads data.statmt.org/wmt19/translation-task/test.tgz itself, sha256-checked) and sacreBLEU BLEU / chrF; the suite hard-errors (infra_required) without it |
| `pip install pycocotools==2.0.11` (in the `evals` extra) | `bundled_detection`: pycocotools COCOeval; the official COCO zips (about 1 GB) download from images.cocodataset.org on first use. `beam_128k` needs no json_repair install: it uses the vendored json_repair 0.44.1 |
| `pip install google-genai` | every LLM-judged suite and acebench's user simulator (also needs `GEMINI_API_KEY`) |
| `pip install openai tiktoken` (in the `evals` extra) + network access to openaipublic.blob.core.windows.net on the first run (or `TIKTOKEN_CACHE_DIR` holding cl100k_base) | `mmlu_pro`: TIGER-Lab evaluate_from_apiX.py streams through openai AsyncOpenAI (2.31 or later) and counts tokens with tiktoken cl100k_base; the suite hard-errors (infra_required) without them. See [mmlu_pro.md](mmlu_pro.md) |

## System toolchains

| Install (Debian/Ubuntu) | Needed by |
| --- | --- |
| Docker CLI + running daemon | the quality runner, `copilot_bench_swe`, `swe_bench_*`, `multi_swe_bench`, `bigcodebench`, `cyberseceval`, `api_bank`, `terminal_bench`, `putnam_formal`, `omnidocbench`, `multipl_e`, `hmmt`, `aime`, `mcp_atlas`, `mcp_bench`, `toolbench`, `bfcl`, `bfcl_live`, `bfcl_v4_agentic`, `ojbench`, `livebench`, `aider_polyglot`, `skillsbench`, `wildclawbench`, `gaia2`, `tau2`, `tau3`, `swe_lancer`, `ui_control_osworld` |
| `docker build -t gbench-swe-bench-live:a145aa87 -f gbench/docker/swe_bench_live.Dockerfile gbench/docker` | `swe_bench_live` (the SWE-bench-Live fork a145aa87 and upstream's multi-language harness, microsoft/SWE-bench-Live 62d02936 with RepoLaunch c4b623d9, run in that image, not the main env; ubuntu:24.04 pinned by digest with packages from the Ubuntu archive snapshot 20260925T120000Z, docker-ce-cli 5:29.8.0-1~ubuntu.24.04~noble, Python dependencies pinned by gbench/docker/swe_bench_live.constraints.txt; the prerequisite check requires the labels org.gbench.swe_bench_live_fork (Python tabs) and org.gbench.swe_bench_live_main / org.gbench.repolaunch (multi-language tabs), so rebuild an image built without them). See [swe_bench_live.md](swe_bench_live.md) |
| `docker build -t gbench-sweagent-for-eval:82422422-rex3 -f gbench/docker/sweagent_for_eval.Dockerfile gbench/docker` | `swe_bench_live` Python and MultiLang tabs, agentic: njukenanli/SWE-agent-for-eval 82422422 installed as its README says (Python 3.12 venv, editable install, vendored swerex) on ubuntu:24.04 pinned by digest with packages from the Ubuntu archive snapshot 20260925T120000Z, docker-ce-cli 5:29.8.0-1~ubuntu.24.04~noble, Python dependencies pinned by gbench/docker/sweagent_for_eval.constraints.txt, and gbench's fix of the task containers' runtime (gbench/docker/sweagent_for_eval_swerex_heredoc.patch: pinned swe-rex 1.4.0 in each task container's runtime layer, heredocs run whole) and of SWE-agent's read of the model's edited files at submit (gbench/docker/sweagent_for_eval_review_decode.patch: read with errors="replace", and a path that is not a regular file, such as a git submodule's directory, is not read; the patch itself is untouched); the prerequisite check requires the labels org.gbench.sweagent_for_eval, org.gbench.sweagent_for_eval_config_sha256, org.gbench.sweagent_for_eval_swerex_runtime and org.gbench.sweagent_for_eval_agents. See [swe_bench_live.md](swe_bench_live.md) |
| `docker build -t gbench-mswe-agent:88217624 -f gbench/docker/mswe_agent.Dockerfile gbench/docker` | `multi_swe_bench`, agentic: multi-swe-bench/MSWE-agent 88217624 on python:3.9.21-slim-bookworm pinned by digest with packages from the Debian archive snapshot 20260925T120000Z, docker-ce-cli 5:29.8.0-1~debian.12~bookworm, requirements plus dataclasses-json pinned by gbench/docker/mswe_agent.constraints.txt, and the six Java repository definitions of multi-swe-bench e0dcb3c4; the prerequisite check requires the labels org.gbench.mswe_agent, org.gbench.mswe_agent_config_sha256, org.gbench.mswe_agent_java_repos and org.gbench.mswe_agent_driver=4 (gbench's driver, with the archive.debian.org fallback of MSWE-agent's reset `apt-get update` for instance images of a Debian release the mirrors no longer serve, recording which setup step of an instance's reset failed and the exception an instance's rollout raised, and giving MSWE-agent's OpenAI client a 3,660 s timeout instead of the SDK's 600 s; an image built with an older driver has to be rebuilt). See [multi_swe_bench.md](multi_swe_bench.md) |
| `docker build -t gbench-mini-swe-agent-v2:2.4.2-2.4.6 -f gbench/docker/mini_swe_agent_v2.Dockerfile gbench/docker` | `copilot_bench_swe`, `swe_bench_multilingual` (mini-swe-agent 2.4.2 / 2.4.6 rollouts); each rollout container runs on docker's default bridge network with OPENAI_API_KEY=EMPTY and reaches the served endpoint through gbench's relay on that network's gateway (e.g. 172.17.0.1), which the bridge must be able to reach (checked before the first rollout, infra_required otherwise); it mounts the docker socket (mini-swe-agent's DockerEnvironment starts the instance container with docker run and runs each command with docker exec) |
| `docker build -t gbench-swebench-v5:5.0.2 -f gbench/docker/swebench_v5.Dockerfile gbench/docker` | `copilot_bench_swe`, `swe_bench_multilingual` (swebench 5.0.2 scoring on ubuntu:24.04 pinned by digest with packages from the Ubuntu archive snapshot 20260925T120000Z; the image carries the Docker CLI, pinned to docker-ce-cli 5:29.8.1-1~ubuntu.24.04~noble, that swebench 5.x uses to stop and remove task containers, and installs swebench under gbench/docker/swebench_v5.constraints.txt; rebuild an image built without them) |
| `docker build -t gbench-swe-bench-pro-v2:harbor-0.22.0-66f92766-sidecar-rt -f gbench/docker/swe_bench_pro_v2.Dockerfile gbench/docker` | `swe_bench_pro` (SWE-bench Pro V2 under Harbor 0.22.0). See [swe_bench_pro.md](swe_bench_pro.md) |
| `docker build -t gbench-bfcl:2026.3.23 -f gbench/docker/bfcl.Dockerfile gbench/docker` | `bfcl`, `bfcl_live` and `bfcl_v4_agentic`: python:3.10-slim with `bfcl-eval==2026.3.23` (byte-identical to gorilla@6ea5797; wheel sha256-checked) and its own pins, the resolved rest pinned by bfcl.constraints.txt, soundfile for qwen_agent (gorilla #1338), all-MiniLM-L6-v2@1110a243 for memory_vector; the single-turn checker, the multi-turn harness and the agentic harness run in it; no Docker, no image or an image for another release hard-errors; about 14 GB. gbench's own environment does not import bfcl_eval; the `dev` extra installs the same release only for the unit tests |
| `docker build -t gbench-terminal-bench:harbor-0.20.0 -f gbench/docker/terminal_bench.Dockerfile gbench/docker` (Harbor 0.20.0 with the TB2.1 leaderboard lock's package versions, gbench/docker/terminal_bench.constraints.txt; the Docker CLI with compose and buildx; built locally, never pulled) | `terminal_bench`: Harbor runs in that image, docker-out-of-docker (the docker socket, the jobs directory at the same path, gbench's Harbor cache `$GBENCH_CACHE_DIR/harbor`), Harbor's telemetry off (HARBOR_TELEMETRY=0 in every harness container and in the image, recorded as harness_telemetry); gbench's own interpreter needs no Harbor; a missing image, or one whose labels name another Harbor release or lock, is infra_required with this command. See [terminal_bench.md](terminal_bench.md) |
| `docker build -t gbench-api-bank:googletrans-3.1.0a0 -f gbench/docker/api_bank.Dockerfile gbench/docker` | `api_bank`: its Translate API runs upstream's pinned googletrans==3.1.0a0 (httpx 0.13.3; the resolved set pinned by gbench/docker/api_bank.constraints.txt) on python:3.10-slim, one container per call on docker's bridge network, read-only, unprivileged, no key or token. The prerequisite check requires the labels org.gbench.api-bank.googletrans=3.1.0a0 and org.gbench.api-bank.httpx=0.13.3 and a worker identical to gbench/docker/api_bank_translate_worker.py; no Docker or no image hard-errors (infra_required). See [api_bank.md](api_bank.md) |
| `docker build -t gbench-cyberseceval:114d02388213-semgrep-1.51.0 -f gbench/docker/cyberseceval.Dockerfile gbench/docker` | `cyberseceval`: Meta's Insecure Code Detector (PurpleLlama CodeShield at 114d0238, regex + semgrep 1.51.0 with PyYAML 6.0.1, the releases upstream's CybersecurityBenchmarks/requirements.txt pins, on python:3.10-slim pinned by digest; the rest pinned by gbench/docker/cyberseceval.constraints.txt) scores every row in that image, one container per probe, prompt-file read and scoring pass, with --network none, a read-only root, no capabilities and no host variable; the prompts are the image's instruct-v2.json (sha256-checked); a missing or mismatched image (the check also compares the constraints file's sha256, probe `constraints_sha256`, and the base-image label org.gbench.cyberseceval.base with the Dockerfile's `ARG BASE_IMAGE`), or a semgrep self-test with no finding, is infra_required. No semgrep and no PurpleLlama checkout on the host (semgrep 1.51.0 pins urllib3~=1.26 and ruamel.yaml<0.18, which would downgrade the serving environment's urllib3 2.x and ruamel.yaml, and upstream requires Python 3.10); see [cyberseceval.md](cyberseceval.md) |
| `gbench-mcp-bench:7a8eaeae` image (local build: `docker build -t gbench-mcp-bench:7a8eaeae -f gbench/docker/mcp_bench.Dockerfile gbench/docker`) | `mcp_bench`: MCP-Bench's runner, its MCP client pinned to mcp==1.16.0 (`gbench/docker/mcp_bench_constraints.txt`: the release upstream's mcp[cli]>=1.9.0 resolved to on the pinned commit's date), on the README's Python 3.10 and all 28 servers; the host needs no `mcp` package; the image carries the label `org.gbench.model-request-policy=2` (it applies the run's thinking switch and `--thinking` budget, with a request timeout sized to that budget, to the model under test, and counts failed model requests as harness errors); it also carries `org.gbench.mcp-bench-harness=3` (its judge takes gbench's Gemini relay placeholder, so no Gemini key enters the container; the five server keys go only to a second container of the image, the credentialed-servers container, which runs the five servers that need them behind gbench's bridge and relay, and the model container refuses a server key; a credentialed session answers only once its server is up and ends with an end frame, and the bridge records every session that failed outside the model; its network check resolves the servers' API host names; its launcher keeps each task's harness errors out of the averages and writes harness_log.json with the Gemini models that answered each task's judge calls), and gbench refuses an image without either label |
| `gbench-toolbench:aa4ed9f4-v5` image (local build: `docker build -t gbench-toolbench:aa4ed9f4-v5 -f docker/toolbench.Dockerfile docker`) | `toolbench`: StableToolBench at aa4ed9f, its Python packages pinned by `docker/toolbench_constraints.txt`; carries the label `org.gbench.model-request-policy=2` (the same request policy as `gbench-mcp-bench:7a8eaeae`; an image without it is refused) and `org.gbench.toolbench-harness=5` (its cache-miss simulator is sent upstream's `max_tokens` 1024 plus the Gemini thinking allowance of 8192 and counts the replies still cut there; the image also runs gbench's per-group query-id selection, `TB_QUERY_IDS_FILE`; its cached server records every tool call and the failure stage, its model client records every DFSDT step with a model request that failed at the endpoint, or that the endpoint rejected for its content (HTTP 400/422, the model's outcome, not an endpoint failure), with its query, the entrypoint takes the cascade from `GBENCH_JUDGE_MODELS` and only relay placeholders; an image without it is refused) |
| `git -C /path/to/LiveBench checkout 0baa4aa7a78dc0276c0ccf28779dba1113580e18 && git -C /path/to/LiveBench lfs pull`, then `docker build -t gbench-livebench:0baa4aa7 -f docker/livebench.Dockerfile /path/to/LiveBench` (run from the gbench repository root; the build context is the LiveBench checkout, with its .git kept) | `livebench` (LiveBench at 0baa4aa7 with gbench's recorded grading patch; a missing image is infra_required). See [livebench.md](livebench.md) |
| `docker build -t gbench-gaia2:are-1.2.0 -f gbench/docker/gaia2.Dockerfile gbench/docker` | `gaia2` (Meta ARE 1.2.0; gbench refuses an image without the label org.gbench.gaia2-harness-errors=7). See [gaia2.md](gaia2.md) |
| `docker build -t gbench-wildclawbench:316334cc-v9 -f gbench/docker/wildclawbench.Dockerfile gbench/docker`, plus the task image: `hf download internlm/WildClawBench Images/wildclawbench-ubuntu_v1.3.tar --repo-type dataset --local-dir /srv/wcb-images && docker load -i /srv/wcb-images/Images/wildclawbench-ubuntu_v1.3.tar` (gives `wildclawbench-ubuntu:v1.3`) | `wildclawbench` (gbench refuses an image without the label org.gbench.wildclawbench-harness-errors=9; OpenClaw's web_search timeout is 60 s, and the search proxy's deadline is 57 s). See [wildclawbench.md](wildclawbench.md) |
| `docker build -t gbench-skillsbench:9a1f4dd5-h7 -f gbench/docker/skillsbench.Dockerfile gbench/docker` | `skillsbench` (BenchFlow with gbench's transport and compose-build-retry patches; gbench refuses an image without the label org.gbench.skillsbench-harness=7). See [skillsbench.md](skillsbench.md) |
| gbench builds `gbench-quality:<commit[:12]>-<digest[:12]>` from `docker/quality.Dockerfile` itself on first use (network access to GitHub, Docker Hub, the npm registry and dl.google.com at build time; about 8 GB) | the quality runner: GemmaClaw at the commit, installed and built, with every bundled plugin's runtime dependencies in an offline npm cache and Google Chrome; its QA suite always runs in a container of it (no network, read-only root, the suite run as a user without capabilities). See [../quality.md](../quality.md) |
| `docker build -t gbench-matharena:b89f2f0a -f docker/matharena.Dockerfile docker` | `hmmt`, `aime` (MathArena's grader). See [hmmt.md](hmmt.md) |
| `docker build -t gbench-bigcodebench:09dd993f -f docker/bigcodebench.Dockerfile docker` | `bigcodebench` (the pinned harness). See [bigcodebench.md](bigcodebench.md) |
| `docker build -t gbench-mcp-atlas:f24ba3fb -f docker/mcp_atlas.Dockerfile docker` | `mcp_atlas` (the release's agent environment without brave-search, whose tools are all search tools; gbench serves them, and the search tools of ddg-search, exa and oxylabs, with Gemini Google Search grounding, while ddg-search, exa and oxylabs run their page tools as upstream, exa and oxylabs with their credentials). See [mcp_atlas.md](mcp_atlas.md) |
| `docker build --target mongo -t gbench-mcp-atlas-mongo:f24ba3fb -f docker/mcp_atlas.Dockerfile docker` | `mcp_atlas`: the release's MongoDB sample data in a local mongod, used when `MONGODB_CONNECTION_STRING` is unset. See [mcp_atlas.md](mcp_atlas.md) |
| `docker build -t gbench-putnam-formal:97b5e945 -f docker/putnam_formal.Dockerfile docker` | `putnam_formal` (Lean v4.33.1 + Mathlib + PutnamBench at 97b5e945). See [putnam_formal.md](putnam_formal.md) |
| `docker build -t gbench-tau2:b7ea9074 -f docker/tau2.Dockerfile docker/tau2_srt` | `tau2`, `tau3` (tau2-bench at b7ea9074 installed from its own uv.lock with litellm 1.81.11, plus Anthropic sandbox-runtime 0.0.23, ripgrep, bubblewrap and socat for the banking_knowledge shell tool). See [tau2.md](tau2.md) |
| Local image `gbench-multipl-e:3025a531`, built with MultiPL-E's own evaluation/Dockerfile at 3025a531, which `docker/multipl_e.Dockerfile` downloads and sha256-checks (`docker build -t gbench-multipl-e-src:3025a531 -f docker/multipl_e.Dockerfile docker` then `docker run --rm gbench-multipl-e-src:3025a531 \| docker build -t gbench-multipl-e:3025a531 -`; network access to codeload.github.com and upstream's toolchain hosts at build time) | `multipl_e`: the evaluator image ships every language toolchain and runs the tests; the suite hard-errors if Docker or the image is missing. See [multipl_e.md](multipl_e.md) |
| Local image `gbench-omnidocbench:193627ae`, built from `docker/omnidocbench.Dockerfile` with an OmniDocBench git checkout at the pinned commit as context | `omnidocbench`: OmniDocBench's own evaluator (TEDS, CDM with TeX Live + ImageMagick 7 + Ghostscript, edit distance) scores the composite inside it; the suite hard-errors if the image is missing or holds another commit. See [omnidocbench.md](omnidocbench.md) |
| `docker build -t aider-benchmark:5dc9490b -f docker/aider_polyglot.Dockerfile docker` (the image fetches aider at 5dc9490bb35f9729ef2c95d00a19ccd30c26339c and carries the label `org.gbench.aider.commit`, checked before every run), the exercises cloned from https://github.com/Aider-AI/polyglot-benchmark into `$GBENCH_PREREQS_DIR/aider-bench/polyglot-benchmark` and checked out at 7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f, unmodified (both checked, else infra_required), and `GBENCH_AIDER_BENCHMARK_DIR=$GBENCH_PREREQS_DIR/aider-bench` | `aider_polyglot`: aider's own benchmark image (upstream's benchmark/Dockerfile plus a setuptools-scm version pin and jest 29.7.0) carries every language toolchain and runs the exercises' tests with upstream's memory limit (`--memory 12g --memory-swap 12g`, shared by the run's `--threads`; the OOM kills in it are recorded as `container_oom_kills`); the host needs only Docker. See [aider_polyglot.md](aider_polyglot.md) |
| `docker build -t gbench-ojbench:5e94480b -f docker/ojbench.Dockerfile docker` + `GBENCH_OJBENCH_TESTDATA` (He-Ren/OJBench_testdata with NOI/ and ICPC/) | `ojbench`: OJBench at 5e94480b with the DMOJ judge-server at f098cd3a, PyPy3 and g++ in the image. See [ojbench.md](ojbench.md) |
| `apt install git` | `ruler` (verifies its RULER checkout's commit); cloning the `aider_polyglot` exercises |
| `apt install git-lfs` **or** `pip install git-lfs`, then `git lfs install` | `ruler`: the NVIDIA/RULER checkout (at the pinned commit, verified at run time) LFS-tracks its `*.json` data (`git clone`, `git checkout <pinned commit>`, then `git lfs pull`); see [ruler.md](ruler.md) |
| `ffmpeg` on PATH | `gaia` test setting (`GBENCH_GAIA_SPLIT=test`): decodes the .m4a attachment to wav |
| Network access to huggingface.co on the first run | `toolbench`: downloads the SoWR reference answers (stabletoolbench/baselines data_baselines.zip, 38 MB, pinned and sha256-checked) and converts them in the gbench-toolbench:aa4ed9f4-v5 image; or set `GBENCH_TOOLBENCH_REFERENCE_DIR`. See [toolbench.md](toolbench.md) |
| Network access to the ToolBench real-API proxy `http://8.130.32.149:8080` | `toolbench`: StableToolBench's server/config.yml toolbench_url, which the cached server asks for every uncached call before its simulator; checked before the run (infra_required) (from docker's bridge network, where the container runs; the patched server gives the call a 120 s read timeout, so a proxy that never answers is a recorded harness error) |
| `docker build -t gbench-ui-control-osworld:fc31a904 -f gbench/docker/ui_control_osworld.Dockerfile gbench/docker`, the task VM image happysixd/osworld-docker@sha256:0e6497a9295647cf05bf2b2af522fdd79bdeba2737595259cab310a3bcf6baa9 (docker pull by digest, then docker tag it as happysixd/osworld-docker), the OSWorld Ubuntu qcow2 under `$GBENCH_OSWORLD_VM_DIR/docker_vm_data/` and a host with `/dev/kvm` | `ui_control_osworld` (not in `--evals all`; runs when named and hard-errors naming whichever is missing). See [ui_control_osworld.md](ui_control_osworld.md) |
| `docker build -t gbench-swe-lancer-harness:51052ced -f gbench/docker/swe_lancer_harness.Dockerfile gbench/docker` | `swe_lancer` (not in `--evals all`; runs when named): openai/frontier-evals 51052ced project/swelancer with its uv.lock, run docker-out-of-docker on the host network with NET_ADMIN (the harness edits the host's DOCKER-USER iptables rules); the task images (swelancer/*, about 1.3 TB) are pulled by the harness. See [swe_lancer.md](swe_lancer.md) |
| `bubblewrap` (`bwrap`) | every code-executing suite (`codeforces`, `cruxeval`, `lcb`, `scicode`), `nestful`, whose Win Rate runs the model's calls of NESTFUL's reference functions (some eval, exec or run their arguments) in a bubblewrap container of its own (without a working bubblewrap the Win Rate alone is infra_required), and `acebench`, whose BinOp arguments are evaluated with Python eval in a bubblewrap container of its own, as upstream evals them; see below |
| Windows host with Docker in Windows-containers mode + checkouts of njukenanli/Win-Agent 47d85cff and microsoft/SWE-bench-Live 62d02936 (with submodule) + one of the tab's instance images on the host (its metadata server check probes with it) | `swe_bench_live` Windows tab only (elsewhere it is reported `infra_required` in its own entry when other tabs run, and hard-errors when selected alone). See [swe_bench_live.md](swe_bench_live.md) |

`aider_polyglot`, `multipl_e` and `ojbench` need none of the language toolchains on the host:
their images carry them.

## Sandboxing

Suites that execute model-written code on the host (`codeforces`, `cruxeval`, `lcb`, `scicode`) run
it through `bwrap` (`apt install bubblewrap`): a read-only root, private `/tmp` and `/run` (so no
Docker socket), no network, new PID, IPC and UTS namespaces with a `/proc` of its own, no
capabilities, and credential files hidden (the home directory is replaced by an empty one with only
the interpreter's directories, the Hugging Face hub cache and `GBENCH_SCICODE_TEST_DATA` bound back
read-only, and for scicode the vendored `scicode` package its test cases import
(`gbench/third_party/_scicode_vendor`), also where a symlink inside the home leads to them;
a home that is one of the interpreter's prefixes or lies inside one is not emptied: it stays, with a
warning, and only its credential files are hidden; credential files elsewhere, such as
`$GOOGLE_APPLICATION_CREDENTIALS`, `$CLOUDSDK_CONFIG`, `$DOCKER_CONFIG` and huggingface_hub's token
files, hidden; see [env-knobs.md](env-knobs.md)). In every mode, `none` included, the program is
given an environment without the credential variables (see [env-knobs.md](env-knobs.md)) and without
any variable whose value carries a credential; with `none` nothing else is isolated. `acebench`'s
BinOp evaluator always runs in a bubblewrap container of its own, whatever `GBENCH_SANDBOX` says (no
`/proc`, no `/run`, only the interpreter's directories on its root), and hard-errors when bubblewrap
is unavailable. `nestful`'s Win Rate worker also always runs in a bubblewrap container of its own,
whatever `GBENCH_SANDBOX` says: no network, only `PATH` and `HOME=/tmp` in its environment, no
`/proc` or `/run`, and only the interpreter's directories and the reference functions, read-only, on
its root. The worker reports what it can see and gbench refuses it unless it is isolated. Its
reference functions get no network (loopback alone; where bubblewrap offers `--ro-bind-data` an
`/etc/hosts` that names `localhost` alone): a Win Rate call that does not win after a failed use of
the network is a harness error, left out of the Win Rate and counted in
`win_rate_network_blocked_calls`, and the run is then `completed_with_errors` and not
leaderboard-comparable. Without a working bubblewrap the Win Rate alone is `infra_required` and the
other NESTFUL metrics run. `multipl_e` executes inside its Docker image instead (see the Docker rows
above) and does not use `bwrap`. Isolation is mandatory by default: there is no "degrade and run
unsandboxed" policy. `GBENCH_SANDBOX` selects the policy:

- `required` (default): require `bwrap`; the suite **hard-errors** (infra_required) before it runs
  if bubblewrap is unavailable or blocked
- `bwrap`: require it and **fail loudly** (raise) if unavailable, a strict CI gate
- `none`: run on the host without isolation (explicit opt-out)

Model code never runs unsandboxed unless you set `GBENCH_SANDBOX=none`. On a shared or
untrusted host keep the default, or use `bwrap` to skip the up-front check and fail at the
first sandboxed execution instead.

### Enabling bubblewrap

1. Install it:
   ```bash
   sudo apt-get install bubblewrap        # Debian/Ubuntu
   ```
2. **Ubuntu 23.10+/24.04 only.** These releases ship
   `kernel.apparmor_restrict_unprivileged_userns=1`, which blocks bubblewrap from creating
   the user namespace it needs; the suite hard-errors (infra_required) with *"bubblewrap IS
   installed but is BLOCKED"*. Allow unprivileged user namespaces:
   ```bash
   sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0
   # persist across reboot:
   echo 'kernel.apparmor_restrict_unprivileged_userns=0' | sudo tee /etc/sysctl.d/60-apparmor-userns.conf
   ```
   (A locked-down host can instead ship a per-binary AppArmor profile granting `bwrap` the
   `userns` permission: more surgical, but more setup.)
3. Verify the jail gbench probes (the probe also binds the interpreter's directories and the Hugging
   Face hub cache back into the emptied home when they live there, and hides credential files). This
   must print `1`:
   ```bash
   bwrap --ro-bind / / --tmpfs /run --dev /dev --unshare-pid --proc /proc --unshare-ipc \
         --unshare-uts --cap-drop ALL --tmpfs /tmp --die-with-parent --new-session \
         --unshare-net --tmpfs "$HOME" python3 -c "print(1)"
   ```
   The command binds nothing back, so run it with a `python3` that lives outside `$HOME` (for
   example `/usr/bin/python3`); with a conda, pyenv or uv interpreter under `$HOME` it prints
   nothing even where gbench's own probe, which binds its interpreter back, passes.

If you cannot (or prefer not to) enable unprivileged user namespaces, run the code-executing
suites unsandboxed **on a host you trust** with `GBENCH_SANDBOX=none`.

## Cloud VMs: the metadata server

On a cloud VM (a GCE VM, for example) the metadata server at `169.254.169.254` hands the VM's
service-account token to any process that can reach it, and a docker container on a bridge or
user-defined network reaches it like any other address. Block it once on the host; gBench never
runs sudo or changes the host's firewall itself:

```bash
sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP
# keep it across reboots (Debian/Ubuntu):
sudo apt-get install -y iptables-persistent && sudo netfilter-persistent save
```

On GCE the VPC DNS resolver answers at the same address, so the rule drops TCP to every port
except 53: the metadata HTTP endpoints are blocked and DNS (TCP and UDP 53) keeps working. A plain
`-j DROP` for the address also cuts name resolution in every container (docker builds, pip,
anything a task resolves). If you added one, remove it with
`sudo iptables -D DOCKER-USER -d 169.254.169.254/32 -j DROP`.

Docker sends the traffic of its bridge networks (the default bridge and user-defined bridge
networks) through its `DOCKER-USER` chain, so the rule covers them all. It does not cover the host
network (`--network host`), which shares the VM's own network stack: on a cloud VM a container
that runs model-controlled code must use a bridge or user-defined network. Do not drop the address
for the whole VM (an `OUTPUT` rule): that cuts the VM's own DNS, its guest agent (SSH keys) and its
credentials. On a dual-stack VM whose docker network has IPv6, the metadata server also answers at
`[fd20:ce::254]`: block it with
`sudo ip6tables -I DOCKER-USER -d fd20:ce::254/128 -p tcp ! --dport 53 -j DROP` (kept across
reboots the same way), or keep IPv6 off that docker network.

The gate is `require_metadata_blocked(eval_name, image, network, docs_url=None, *, resolve=())` in
`gbench/runners/eval_suites/container_guard.py`. The suites listed below call it before they start a
container that runs model-controlled code with network access. It starts one short-lived container
of the suite's own local image on the same network (`--pull never`, `--read-only`, `--cap-drop ALL`)
that only opens and closes a TCP connection to `169.254.169.254:80`, and to `[fd20:ce::254]:80` (a
container without an IPv6 route gets "unreachable" there, not an error), with no request, so no
token or other metadata is read, with the image's `python3`, else its `bash` (`/dev/tcp`) under
coreutils `timeout`. If a connection opens, the suite hard-errors (infra_required) naming the
command above (the `ip6tables` one for IPv6). A suite can also pass the host names its containers
need (`resolve`): the same container then resolves each of them (`python3`'s `socket.getaddrinfo`,
else `getent hosts` or `nslookup`), and a name that does not resolve is a hard error saying that
containers on that network cannot resolve host names, that on GCE the VPC DNS resolver is
`169.254.169.254` so a rule dropping all traffic to it drops DNS too, and naming the rule above. A
name the resolver answers does not exist (getaddrinfo's EAI_NONAME or EAI_NODATA, nslookup's
NXDOMAIN) is a hard error asking to check the URL or host name instead (DNS itself answered, so no
firewall rule is involved). If the probe cannot give a clear verdict (neither program in the image,
no resolver program for a name it has to check, a docker error or timeout), the suite hard-errors
too: no container starts on a verdict the probe could not establish. `--network none` needs no
probe. Each network's verdict, and each name that resolved there, is cached for the rest of the run.
No knob turns the gate off.

The suites that call it:

- `mcp_atlas`, `mcp_bench` and `wildclawbench`, on the network of their environment or task
  containers (see each suite's guide): mcp_atlas on both of its networks (the environment
  container's and the credentialed-servers container's), and for an environment given with
  `GBENCH_MCP_ATLAS_SANDBOX_URL` through that environment's own code tool; mcp_atlas's probe
  containers also resolve the host names each container's servers need; mcp_bench also has the
  probe container resolve the servers' API host names and a named model endpoint, and checks
  its credentialed-servers container's own network the same way (the credentialed servers'
  API hosts, pypi.org and files.pythonhosted.org).
- The SWE family (`copilot_bench_swe`, `swe_bench_multilingual`, `swe_bench_live`,
  `multi_swe_bench`, `swe_bench_pro`), `terminal_bench`, `skillsbench`, `swe_lancer` and
  `ui_control_osworld`, through `swebench_common.require_task_network_blocked`, before their task
  containers start. Their harnesses start those containers on docker's default bridge or on networks
  they create at run time (docker compose, Harbor, BenchFlow, alcatraz), so gbench probes the
  default bridge and a user-defined bridge network it creates for the probe
  (`gbench-metadata-probe-<id>`, label `org.gbench.metadata-probe=1`, removed right after). The
  probe runs in an image the suite already has on the host, and nothing is pulled for it:
  `terminal_bench` probes with a cached task image (a `--shard` run's own tasks first), and
  `multi_swe_bench`'s harness pass without the MSWE-agent image with a submitted instance's
  published image or a local `mswebench/*` image; when none is present, the gate hard-errors naming
  the `docker pull` of one of the run's task images. `terminal_bench` and `skillsbench` also pass
  `resolve` with the package hosts their task verifiers install from at run time
  (`archive.ubuntu.com`, `deb.debian.org`, `astral.sh`, `github.com`, `pypi.org`,
  `files.pythonhosted.org`), `swe_bench_live`'s agentic Linux tabs with the hosts SWE-ReX's
  runtime build fetches from (`deb.debian.org`, `archive.ubuntu.com`, `www.python.org`, `pypi.org`,
  `files.pythonhosted.org`), `swe_bench_pro` with the hosts every sandbox installs the agent from
  (`deb.debian.org`, `security.debian.org`, `archive.debian.org`, `archive.ubuntu.com`,
  `astral.sh`, `github.com`, `pypi.org`, `files.pythonhosted.org`), and `multi_swe_bench` with the
  package hosts its instance containers' builds fetch from at test time, for the run's languages
  (`repo.maven.apache.org`, `oss.sonatype.org`, `plugins.gradle.org`, `services.gradle.org`,
  `registry.npmjs.org`, `proxy.golang.org`, `index.crates.io`, `static.crates.io`, `github.com`).
  A `terminal_bench` or `skillsbench` task whose verifier still could not resolve its package hosts
  before its tests ran, and a `multi_swe_bench` instance whose harness test run captured no test
  result for that reason, is a harness error, left out of the score, when something outside its
  task container corroborates the failure (another such task in the run, or a fresh probe
  container after the run that cannot resolve one of the hosts either,
  `swebench_common.corroborated_dns_failures`); otherwise it stays the model's outcome.
  `swe_bench_live`'s Windows tab runs only after its own check on the Windows-containers engine
  (`swebench_common.check_windows_task_network`: a PowerShell TCP connect to `169.254.169.254:80`
  and to `[fd20:ce::254]:80`, nothing sent, in one of the tab's instance images already on the
  host, on docker's default network) found the metadata server unreachable over both; otherwise,
  and on a Linux engine, it is reported `infra_required` with the reason.
- `bfcl` (multi-turn part), `bfcl_v4_agentic` (`bfcl generate`), `tau2`, `tau3`, `livebench` (the
  run), `aider_polyglot` and `bigcodebench` (sanitize/evaluate), on docker's bridge network. The
  ones that talk to the model endpoint directly from there (`bfcl`'s multi-turn part and
  `livebench`, when the endpoint has no key) reach an endpoint on this host at
  `host.docker.internal` (`--add-host host.docker.internal:host-gateway`). Serve the model on an
  address the docker bridge reaches: vLLM as gbench starts it listens on every interface; a server
  bound to `127.0.0.1` only does not. gbench first checks that a TCP connection to it opens from
  that network and hard-errors (infra_required) otherwise. The same probe container also resolves
  the host names those containers look up themselves, so a network without working DNS is a hard
  error before any item runs: for `bfcl`'s multi-turn part and `livebench` without a key, the model
  endpoint's host when its URL names one (for `bfcl`, the proxy's host instead when it is a name
  and the request goes through a proxy; `livebench`'s container gets no proxy variables);
  `huggingface.co` for `livebench` and `bigcodebench`; and for `aider_polyglot` the hosts
  the exercises' tests download from, per language of the run (`services.gradle.org` and
  `repo.maven.apache.org` for Java, `index.crates.io` and `static.crates.io` for Rust). The model's
  code in them keeps general internet access on the bridge network, as upstream runs it; only the
  metadata server is blocked. With an endpoint key (`VLLM_API_KEY`, else `OPENAI_API_KEY`), `bfcl`'s
  multi-turn part and `livebench` reach the endpoint through gbench's endpoint relay
  (`swebench_common.EndpointRelay`) on the docker bridge gateway instead, so the key never enters
  their containers. `aider_polyglot` (since 2026-10-11) and `bfcl generate` (the agentic part of
  `bfcl`, and `bfcl_v4_agentic`), `tau2` and `tau3` (since 2026-10-13) always reach it through that
  relay, keyless too; the relay adds the key only when there is one. For `bfcl`'s multi-turn part
  with a key (`bfcl_v4_agentic.bridge_network_check`), `bfcl generate`, `tau2` and `tau3`, gbench
  checks that a TCP connection to the relay on the gateway opens from that network. For
  `aider_polyglot` and keyed `livebench` it does not: their probe
  (`tau_common.require_endpoint_network`) opens a TCP connection only when it rewrote a loopback
  URL to `host.docker.internal`, and the relay's gateway URL is none, so a gateway the bridge
  cannot reach shows up as failed model requests. For every relayed run the endpoint's own
  reachability and host name are not checked from the container. `terminal_bench`'s Terminus-2 runs
  in Harbor's process in its harness container on the host network and always uses such a relay on
  `127.0.0.1`, and the container gets no credential (only OPENAI_API_KEY=EMPTY, OPENAI_BASE_URL,
  HARBOR_TELEMETRY=0 and the other set HARBOR_* and proxy variables). The web_search `bfcl generate`
  also has the guard's probe resolve `vertexaisearch.cloud.google.com` and searches through gbench's
  Gemini relay on the same gateway. `tau2` and `tau3` reach the Gemini API only through gbench's
  Gemini relay on the docker bridge gateway (their container holds its per-run placeholder, never a
  key; the relay resolves `generativelanguage.googleapis.com` on this host), and always reach the
  endpoint through gbench's endpoint relay there; gbench checks that a TCP connection to each relay
  opens from that network.
- `gaia2` and `toolbench`, on docker's bridge network: their containers run the model's tool
  calls (ARE's in-process apps; StableToolBench's cached server, which passes an uncached call on
  to the ToolBench real-API proxy and the Gemini simulator). They reach the model endpoint and the
  Gemini API only through gbench's relays on the bridge gateway (`swebench_common.EndpointRelay`,
  `gemini_relay.GeminiRelay`), and the containers hold the Gemini relay's placeholder, no key.
  `gaia2` resolves no name either (it reads its datasets offline from the image and reaches
  gbench's relays and token-id decoder at the gateway address);
  `toolbench` resolves no name (the real-API proxy is an IP address).
- `quality`, whose container runs with `--network none`, so the gate needs no probe
  ([quality.md](../quality.md)).

These run their model-code containers with `--network none` and need no probe: the `ojbench`
judge, the `multipl_e` evaluator, the `omnidocbench` evaluator (its CDM metric compiles the model's
LaTeX with pdflatex), `putnam_formal`'s proof checker, `bfcl_live`, and the BFCL image probe,
single-turn checker and `bfcl evaluate`.

## Harnesses and gates

| Variable | Purpose |
| --- | --- |
| `GBENCH_SWE_BENCH_PRO_HARNESS_DIR` | clone of `scaleapi/SWE-bench_Pro-os`, used only by the separately maintained engine route (`GBENCH_SWEBENCH_PRO_AGENTIC=1`); the default V2 path uses the pinned tree inside `gbench-swe-bench-pro-v2:harbor-0.22.0-66f92766-sidecar-rt`. See [swe_bench_pro.md](swe_bench_pro.md) |
| `SCICODE_TEST_DATA` | path to SciCode's ~1 GB `test_data.h5` reference outputs; see [scicode.md](scicode.md). Without it the suite hard-errors (infra_required) rather than reporting a structural 0% |
| `GBENCH_SWE_LANCER_IMAGE` | `swe_lancer`: name of the locally built harness image (default gbench-swe-lancer-harness:51052ced). See [swe_lancer.md](swe_lancer.md) |
| `GBENCH_BFCL_PROJECT_ROOT` | where `bfcl-eval` keeps generations and scores (default `$GBENCH_CACHE_DIR/bfcl`); must hold no `.env` (upstream's `bfcl generate` / `bfcl evaluate` load it into the harness process; see [bfcl_v4_agentic.md](bfcl_v4_agentic.md#credentials-and-the-execution-guard)) |
| `GEMINI_API_KEY` (or the key list `GEMINI_API_KEYS`, which wins when both are set) | required by every judged suite, by acebench (its agent_multi_turn user simulator; without it only that category is reported infra_required and the other categories run) and by tau2 / tau3 (user simulator, judge, and tau3's retrieval embeddings with gemini-embedding-001: upstream's alltools, the only retrieval config, with its dense search on `gemini-embedding-001`; it stays on the host: gbench's Gemini relay adds it to the gbench-tau2 container's requests, which hold only the relay's placeholder; `GBENCH_TAU2_OPENAI_API_KEY`, `GBENCH_TAU2_OPENAI_BASE_URL` and `GBENCH_TAU2_RETRIEVAL_CONFIG` no longer change anything, see [tau3.md](tau3.md)); they hard-error (infra_required) without it rather than falling back to substring matching or another provider. The Web Search setting (the web_search categories) of `bfcl` and `bfcl_v4_agentic` needs it too: BFCL's web_search tool is served by Gemini Google Search grounding, and without a key, with one the Gemini API rejects, with a model of gbench's Gemini cascade the API does not serve to it, or when a search no grounding model served during the run, that setting alone is reported infra_required while the other BFCL categories run. Every model role a harness has (judge, grader or rater, labeler, user or API simulator, web search / grounding, embeddings) goes through the Gemini API: no model role reads a SerpAPI, Brave, RapidAPI or OpenAI embedding key. A benchmark's non-model tool backends run as upstream, with upstream's own keys: complexfuncbench's `RAPID_API_KEY`, mcp_bench's five server keys and mcp_atlas' environment credentials (rows below) |
| `NPS_API_KEY`, `NASA_API_KEY`, `HF_TOKEN`, `NCI_API_KEY`, `GOOGLE_MAPS_API_KEY` | `mcp_bench`: the keys the MCP-Bench README requires (exactly the ones the `env` lists of the pinned `mcp_servers/commands.json` name), passed only to gbench's credentialed-servers container, whose servers hold them while serving the model's tool calls (an accepted residual risk: see [mcp_bench.md](mcp_bench.md#the-server-keys-the-credentialed-servers-container)). The suite hard-errors (infra_required) before any container starts when one is unset or empty, naming each missing key and its server. The NPS, NASA, Hugging Face (a Read token) and NCI keys are free; the Google Maps key needs a Google Cloud billing account with the Geocoding, Elevation, Places, Directions and Distance Matrix APIs enabled. Each set key is also checked live before any container starts (one read-only request to the service its server calls; Google Maps one per Maps Platform API its server calls); a key the service refuses, or one that gets no answer of the service's own after 3 attempts, is infra_required naming the key and the service's answer; [mcp_bench.md](mcp_bench.md#checking-the-keys) lists the requests and [mcp_bench.md](mcp_bench.md#the-keys) says where to get each key |
| `RAPID_API_KEY` | `complexfuncbench`: upstream's response-based call-match tier (utils/rapidapi.py) sends both calls of a pair to the live booking-com15 API on RapidAPI with it. It is the key of a RapidAPI account subscribed to <https://rapidapi.com/DataCrawler/api/booking-com15> (on 2026-09-28 the free BASIC plan allows 50 requests a month; paid plans start at 8.99 USD a month). The suite hard-errors (infra_required) before the run when it is unset or empty, or when a live check with it gets an outage reply (401/403: rejected or not subscribed; 429: quota). It is never replaced by the Gemini judge. See [complexfuncbench.md](complexfuncbench.md#requirements) |
| `AIRTABLE_API_KEY`, `ALCHEMY_API_KEY`, `E2B_API_KEY`, `EXA_API_KEY`, `GITHUB_TOKEN`, `GOOGLE_MAPS_API_KEY`, `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` / `GOOGLE_REFRESH_TOKEN`, `LARA_ACCESS_KEY_ID` / `LARA_ACCESS_KEY_SECRET`, `NPS_API_KEY`, `NOTION_TOKEN`, `OXYLABS_USERNAME` / `OXYLABS_PASSWORD`, `SLACK_MCP_XOXC_TOKEN` / `SLACK_MCP_XOXD_TOKEN`, `TWELVE_DATA_API_KEY`, `WEATHER_API_KEY` (optional `MONGODB_CONNECTION_STRING`) | `mcp_atlas`: the release environment's credentials for its 15 credentialed servers, passed only to gbench's credentialed-servers container, whose servers hold them while serving the model's tool calls (an accepted residual risk: see [mcp_atlas.md](mcp_atlas.md#agent-environment-the-mcp-servers)), never to the container where the model's shell and code tools run. Every one is required: an unset one is infra_required before any container starts, named per server with the sample data it needs. The release's pre-flight makes one real call per server, so a rejected credential is infra_required before the run. During the run, a page-tool reply saying Exa or Oxylabs could not be reached, or refused the run's account (Exa HTTP 401/402/429; Oxylabs' reply with no JSON body, which is how its API refuses credentials and how a provider-side error page reads, its error reply with no job record (`Error: 'job'`), or HTTP 401/403/407/429), is a harness error, never the model's outcome. Where to get each and what it costs: [mcp_atlas.md](mcp_atlas.md#requirements) |

## Checking what is available

There is no separate probe command: run the suite. Anything whose prerequisite is missing
returns `status: error` with the exact missing piece and a docs pointer, costs no
generation, and is reported as an **error**, never as 0%. Read the run summary and
install from the tables above.
