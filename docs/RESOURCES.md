# gbench resources directory

## 1. Core gbench suite
* **gbench GitHub repository**: source code, runners and docs (`https://github.com/google-deepmind/gbench`)

## 2. Example serving backends
gbench works with any OpenAI-compatible `/v1` server and is validated on vLLM and Ollama. The eval
suites' recovery of text a server parser removed uses vLLM's `return_token_ids` and `/detokenize`
(other servers fall back as described in the server-parser policy of
[env-knobs.md](evals/env-knobs.md)).

* **vLLM** (validated; the server gbench launches for a local run): `https://github.com/vllm-project/vllm` | `https://docs.vllm.ai/`
* **Ollama** (validated with `--remote-endpoint`): `https://github.com/ollama/ollama` | `https://ollama.com/`
* **SGLang**, **TGI (Text Generation Inference)** and **TensorRT-LLM** also serve an OpenAI-compatible API; gbench has not been validated on them: `https://github.com/sgl-project/sglang` | `https://github.com/huggingface/text-generation-inference` | `https://github.com/NVIDIA/TensorRT-LLM`
* **Google Cloud Run** (`https://cloud.google.com/run`): only a service that allows unauthenticated calls. IAM-protected services (Cloud Run's default) are not supported yet, because the endpoint check that starts every remote run sends no token (see [Remote endpoint mode](remote-endpoints.md)).

## 3. Academic and domain evaluation suites

86 suites run in `--evals all`; `swe_lancer` and `ui_control_osworld` are registered but run only
when named, and the two deregistered suites are listed as roadmap entries.

> **Dataset licenses, gated access, and redistribution:** see
> [THIRD_PARTY_DATASETS.md](THIRD_PARTY_DATASETS.md). gbench pulls each dataset at runtime from its
> upstream source and redistributes none of it; some are gated (need an HF token) or carry
> non-commercial / copyleft / undeclared licenses. Running a benchmark is subject to that dataset's
> terms.

Dataset links below reflect the exact source each suite's loader reads. Suites with a
Setup Guide need extra tooling (Docker, gold data, or an opt-in) and hard-error
until it is provisioned.

### Pillar 1: General Knowledge & Scientific Reasoning
* **ARC-AGI**: Abstraction and Reasoning Corpus, the public evaluation set of grid-transformation puzzles (`https://github.com/fchollet/ARC-AGI` @3990304, data/evaluation | `https://arcprize.org/`)
* **CausalBench**: Corr2Cause, causal inference from stated correlations (`https://huggingface.co/datasets/causal-nlp/corr2cause` | `https://github.com/causalNLP/corr2cause`)
* **CIMemories (`cimemories`)**: Violation@5 / Completeness, release pipeline with a Gemini-cascade labeler and REVEAL judge ([Setup Guide](evals/cimemories.md) | `https://huggingface.co/datasets/facebook/CIMemories` | `https://github.com/facebookresearch/CIMemories`)
* **CyberSecEval**: CyberSecEval-instruct (Insecure Code), Meta's Insecure Code Detector over generated code ([Setup Guide](evals/cyberseceval.md) | `https://github.com/meta-llama/PurpleLlama/tree/114d02388213ae791650b10fa61c8785661c2602/CybersecurityBenchmarks` datasets/instruct/instruct-v2.json)
* **CyberGym**: execution-verified vulnerability reproduction, **deregistered (roadmap-only, not runnable)**: the text proxy that shipped under this name was removed, see [roadmap](evals_roadmap.md) (`https://github.com/sunblaze-ucb/cybergym`)
* **FRAMES**: Factuality, Retrieval, And reasoning MEasurement Set, 824 multi-hop questions over Wikipedia (`https://huggingface.co/datasets/google/frames-benchmark`)
* **GPQA**: graduate-level science multiple choice, full split (`https://huggingface.co/datasets/Idavidrein/gpqa`)
* **GPQA Diamond**: graduate-level science multiple choice, the 198-question diamond split (`https://huggingface.co/datasets/Idavidrein/gpqa`)
* **HealthBench**: 5,000 health conversations scored against physician-written rubrics (`https://huggingface.co/datasets/openai/healthbench`)
* **Humanity's Last Exam (HLE)**: 2,500 expert-written closed-ended questions across many subjects (`https://huggingface.co/datasets/cais/hle` | `https://lastexam.ai/`)
* **I18N Translation**: WMT19 news test sets (newstest2019, 19 directions) via sacrebleu (`https://data.statmt.org/wmt19/translation-task/test.tgz` | `https://github.com/mjpost/sacrebleu`)
* **IFEval**: instruction following with verifiable constraints, 541 prompts (`https://huggingface.co/datasets/google/IFEval`)
* **LAB-Bench**: all 8 categories (literature, databases, supplementary material, figures, tables, protocols, sequences, cloning) (`https://huggingface.co/datasets/futurehouse/lab-bench`)
* **LiveBench**: a monthly-refreshed benchmark; gbench runs its 6 core categories at the pinned release 2024-11-25 (`https://huggingface.co/datasets/livebench/coding` | `https://huggingface.co/datasets/livebench/data_analysis` | `https://huggingface.co/datasets/livebench/instruction_following` | `https://huggingface.co/datasets/livebench/language` | `https://huggingface.co/datasets/livebench/math` | `https://huggingface.co/datasets/livebench/reasoning`; harness `https://github.com/LiveBench/LiveBench` at 0baa4aa7a78dc0276c0ccf28779dba1113580e18, release 2024-11-25, in the local image gbench-livebench:0baa4aa7 with one recorded patch to gen_ground_truth_judgment.py so instruction_following is graded)
* **MedXpertQA**: Expert-level medical exam QA, Text + MM subsets (`https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA`)
* **MMLU**: Massive Multitask Language Understanding, canonical split, scored by answer-letter log-likelihood (hendrycks/test evaluate.py) (`https://huggingface.co/datasets/cais/mmlu`)
* **MMLU-Pro**: multi-discipline professional reasoning, 14 subjects and 10 options (`https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro`)
* **MMLU-Redux-2.0**: Re-annotated MMLU, 57 subjects, scored with HELM MMLU 5-shot on error_type ok rows; in-context examples from cais/mmlu dev (`https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux-2.0`)
* **Multilingual MMLU**: human-translated MMLU in 14 languages (`https://huggingface.co/datasets/openai/MMMLU`)
* **SimpleQA**: 4,326 short-form factual questions that reward abstaining over a wrong answer (`https://huggingface.co/datasets/basicv8vc/SimpleQA`)
* **WildBench (`wildbench`, renamed from `lmsys_noncoding_hard`)**: Real-user multi-turn tasks, checklist-judged WB_score.task_macro (headline) and WB_score plus pairwise WB-Reward vs gpt-4-turbo-2024-04-09 / claude-3-haiku-20240307 / Llama-2-70b-chat-hf, and WB-Elo (Raw/LC) over the 63-model leaderboard pool re-graded by the Gemini judge; all 1,024 v2 tasks ([Setup Guide](evals/wildbench.md) | `https://huggingface.co/datasets/allenai/WildBench` | `https://huggingface.co/datasets/allenai/WildBench-V2-Model-Outputs` | `https://github.com/allenai/WildBench` | `https://huggingface.co/spaces/allenai/WildBench`)
* **WMDP**: Weapons of Mass Destruction Proxy, hazardous knowledge in biosecurity, chemical security and cybersecurity (`https://huggingface.co/datasets/cais/wmdp`)

### Pillar 2: Mathematics & Proofs
* **AIME**: American Invitational Mathematics Examination, AIME 2026 as MathArena runs it ([Setup Guide](evals/aime.md) | `https://huggingface.co/datasets/MathArena/aime_2026` | grader: `https://github.com/eth-sri/matharena` at b89f2f0a)
* **GSM8K**: grade-school arithmetic word problems, lm-evaluation-harness gsm8k_cot (`https://huggingface.co/datasets/openai/gsm8k`)
* **HMMT**: Harvard-MIT Mathematics Tournament February 2026 as MathArena runs it ([Setup Guide](evals/hmmt.md) | `https://huggingface.co/datasets/MathArena/hmmt_feb_2026` | grader: `https://github.com/eth-sri/matharena` at b89f2f0a)
* **IMO-AnswerBench**: 400 olympiad problems with short final answers (`https://github.com/google-deepmind/superhuman/blob/955558ad67ef24dccd27545c326988bdad1b4471/imobench/answerbench_v2.csv`, v2)
* **AMC / AIME Combined**: a gbench composite of AMC 12 2022-2023 and AIME 2022-2024 (not canonical) (`https://huggingface.co/datasets/AI-MO/aimo-validation-amc` | `https://huggingface.co/datasets/AI-MO/aimo-validation-aime`)
* **PutnamBench**: Putnam Competition problems, informal track (a gbench protocol, not leaderboard-comparable) (`https://github.com/trishullab/PutnamBench` @97b5e945, informal/putnam.json, 673 problems)
* **Putnam-Formal**: PutnamBench Lean 4 track, both leaderboard settings (Lean with answer / Lean no answer), kernel-checked on Lean v4.33.1 + Mathlib v4.33.1; no-answer answers reviewed by kernel proof, else by the gbench Gemini judge cascade (`https://github.com/trishullab/PutnamBench` @97b5e945, lean4/src, 672 problems)

### Pillar 3: Coding & Algorithmic Design
* **Aider Polyglot**: 225 Exercism exercises in 6 languages, edited in aider's native edit format (`https://github.com/Aider-AI/polyglot-benchmark`)
* **BigCodeBench**: 1,140 practical Python tasks that call real libraries, Instruct and Complete splits plus BigCodeBench-Hard ([Setup Guide](evals/bigcodebench.md) | `https://huggingface.co/datasets/bigcode/bigcodebench`)
* **Codeforces**: the open-r1/codeforces verifiable test split, scored by pass rate (`https://huggingface.co/datasets/open-r1/codeforces`)
* **CopilotBench / SWE-bench Verified**: 500 human-validated GitHub issues from 12 Python repositories, resolved by an agent ([Setup Guide](evals/copilot_bench_swe.md) | `https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified` (pinned `78f471bf`, v5 harness))
* **CRUXEval**: Code Reasoning: Input and Output Prediction (CRUXEval-I / CRUXEval-O) (`https://huggingface.co/datasets/cruxeval-org/cruxeval`)
* **LiveCodeBench (LCB)**: code generation (headline), code execution, test output prediction and self-repair (`https://huggingface.co/datasets/livecodebench/code_generation_lite`)
* **Multi-SWE-bench**: issue resolution in 7 languages, 1,632 instances ([Setup Guide](evals/multi_swe_bench.md) | `https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-bench`)
* **MultiPL-E**: HumanEval translated into 23 programming languages, scored by execution ([Setup Guide](evals/multipl_e.md) | `https://huggingface.co/datasets/nuprl/MultiPL-E`)
* **OJBench**: 232 NOI and ICPC problems in Python and C++, judged in the DMOJ sandbox ([Setup Guide](evals/ojbench.md) | `https://huggingface.co/datasets/He-Ren/OJBench_testdata`)
* **SciCode**: research-level scientific computing problems, scored by execution ([Setup Guide](evals/scicode.md) | `https://huggingface.co/datasets/SciCode1/SciCode` | reference outputs `https://huggingface.co/datasets/Srimadh/Scicode-test-data-h5` (test_data.h5, pinned to 72c247d3a8410921b2e848e046d71ed63d9a0ddb))
* **SWE-bench Live**: Live GitHub issue resolution, every leaderboard tab (Lite, Full, Verified; C/C++, C#, Java, TS/JS, Go, Rust; Windows) ([Setup Guide](evals/swe_bench_live.md) | `https://huggingface.co/datasets/SWE-bench-Live/SWE-bench-Live`, `https://huggingface.co/datasets/SWE-bench-Live/MultiLang`, `https://huggingface.co/datasets/SWE-bench-Live/Windows`)
* **SWE-bench Multilingual**: 300 GitHub issues from 41 repositories in 9 languages ([Setup Guide](evals/swe_bench_multilingual.md) | `https://huggingface.co/datasets/SWE-bench/SWE-bench_Multilingual`)
* **SWE-bench Pro**: Scale AI's long-horizon software engineering tasks, public V2 set ([Setup Guide](evals/swe_bench_pro.md) | `https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro` (V2 = config `default` + HARD-51 = config `hard`, pinned `2d52cb3d`; Harbor tasks and locked-protocol tooling `https://github.com/scaleapi/SWE-bench_Pro-os` `v2/` at `66f92766`))
* **SWE-Lancer**: real-world freelance software engineering tasks, run with SWE-Lancer's own agent and harness; not in `--evals all`, runs when named; hard-errors `infra_required` without Docker, the locally built `gbench-swe-lancer-harness:51052ced` image or iptables access to the host's DOCKER-USER chain ([Setup Guide](evals/swe_lancer.md) | `https://github.com/openai/frontier-evals/tree/main/project/swelancer`)

### Pillar 4: Long Context & Retrieval
* **AA-LCR**: Artificial Analysis Long Context Reasoning v1.1, 100 questions over document sets of about 100k tokens (`https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR`)
* **BEAM 128k**: BEAM's 128K tier, long-term memory questions over multi-session conversations (`https://huggingface.co/datasets/Mohammadta/BEAM`)
* **LOFT SciFact retrieval (128k)** (`loft_scifact`, renamed from `loft_x_arxiv`): in-context retrieval, LOFT recall@1 ([Setup Guide](evals/loft_scifact.md) | `https://github.com/google-deepmind/loft`)
* **LongBench-v2 Code (`longbench_v2_code`, renamed from `culer`)**: LongBench-v2 Code Repository Understanding domain ([Setup Guide](evals/longbench_v2_code.md) | `https://huggingface.co/datasets/zai-org/LongBench-v2`)
* **MRCR**: multi-round co-reference resolution, 8 bins from 4K to 1M tokens; the bins that fit the served window are evaluated (`https://huggingface.co/datasets/openai/mrcr`)
* **RULER**: 13 long-context tasks at 6 bands from 4k to 128k; data generated per tokenizer by NVIDIA RULER's own prepare.py at commit c3f5e3b4 (`https://github.com/NVIDIA/RULER`)

### Pillar 5: Tool Use & Agentic Workflows
* **ACEBench**: function calling and tool use: Normal, Special and Agent in English (headline) and Chinese at 56dd66cf; the agent_multi_turn user simulator is the Gemini judge cascade ([Setup Guide](evals/acebench.md) | `https://github.com/chenchen0103/ACEBench` | `https://chenchen0103.github.io/ACEBench/`)
* **AgentDojo**: prompt-injection security of tool-calling agents (`https://github.com/ethz-spylab/agentdojo`, agentdojo==0.1.35, benchmark v1.2.2)
* **API-Bank**: multi-level API calling, tool retrieval and response synthesis (`https://huggingface.co/datasets/liminghao1630/API-Bank`)
* **BFCL (Berkeley Function Calling Leaderboard)**: all 22 v4 scoring categories: single-turn data from gorilla@6ea5797 (`https://github.com/ShishirPatil/gorilla/tree/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard/bfcl_eval/data`), single-turn checker, multi-turn data, backends and checker from `bfcl-eval==2026.3.23` in the local image gbench-bfcl:2026.3.23 (`gbench/docker/bfcl.Dockerfile`; byte-identical to 6ea5797; every file used is checked at run time; `https://pypi.org/project/bfcl-eval/2026.3.23/`), agentic via bfcl_v4_agentic ([Setup Guide](evals/bfcl.md) | `https://gorilla.cs.berkeley.edu/leaderboard.html`)
* **BFCL v4 Agentic**: Agentic memory and web-search tool use (BFCL Agentic score); data and harness from the `bfcl-eval==2026.3.23` package in the local image gbench-bfcl:2026.3.23 (memory_vector encoder sentence-transformers/all-MiniLM-L6-v2@1110a243; the web_search tool is served by Gemini Google Search grounding in place of BFCL's DuckDuckGo via SerpAPI search, recorded and not leaderboard-comparable) (`https://github.com/ShishirPatil/gorilla/tree/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard/bfcl_eval/data` | `https://pypi.org/project/bfcl-eval/2026.3.23/`)
* **BFCL Live (`bfcl_live`, renamed from `bfcl_v3_live`)**: Live (real-world) single-turn function-calling slice ([Setup Guide](evals/bfcl_live.md) | `https://github.com/ShishirPatil/gorilla/tree/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard/bfcl_eval/data`)
* **BrowseComp**: 1,266 hard web-research questions with short answers (`https://huggingface.co/datasets/smolagents/browse_comp`)
* **ComplexFuncBench**: THUDM multi-step, multi-constraint function calling over the Booking.com API (`https://huggingface.co/datasets/THUDM/ComplexFuncBench` at 5dc7739343f8c87aee465931ba66db13a0fb5dd4, sha256-checked at load; evaluation ported from `https://github.com/THUDM/ComplexFuncBench` at c37b284e2f2e03ee456115b7c4b7e537f534be37 with upstream's live-RapidAPI response comparison (booking-com15, RAPID_API_KEY), and both GPT-4o judge roles on the Gemini judge cascade)
* **DeepSearchQA**: Google DeepMind multi-step web research with set answers ([Setup Guide](evals/deepsearch_qa.md) | `https://huggingface.co/datasets/google/deepsearchqa`)
* **GAIA**: General AI Assistants, questions answered with web research, files and multi-step reasoning ([Setup Guide](evals/gaia.md) | `https://huggingface.co/datasets/gaia-benchmark/GAIA`); the test split is scored by the leaderboard from gbench's submission file (`https://huggingface.co/spaces/gaia-benchmark/leaderboard`)
* **GAIA-2**: Meta's Agents Research Environments (ARE) scenarios (canonical `are-benchmark` harness; hard-errors until provisioned; ARE's LLM judge runs on the gbench Gemini cascade in place of Llama-3.3-70B (recorded, not leaderboard-comparable)) ([Setup Guide](evals/gaia2.md) | `https://huggingface.co/datasets/meta-agents-research-environments/gaia2`)
* **GDPval**: economic knowledge-work deliverables, wins-or-ties rate against the expert reference (`https://huggingface.co/datasets/openai/gdpval`)
* **Gorilla APIBench**: one-line calls to real machine-learning APIs (TorchHub, TensorHub, HuggingFace) (`https://huggingface.co/datasets/gorilla-llm/APIBench` | `https://github.com/ShishirPatil/gorilla/tree/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/gorilla/eval`, all three hubs; 0-shot (headline) and the BM25, GPT-Index and Oracle retriever settings)
* **WebArena (formerly `lmarena_web_agent`)**: web agents in sandboxed web applications; the canonical harness is on the roadmap and the suite is **deregistered (roadmap-only, not runnable)**, see [roadmap](evals_roadmap.md) (`https://github.com/web-arena-x/webarena`)
* **MCP-Atlas**: multi-server Model Context Protocol tool use ([Setup Guide](evals/mcp_atlas.md) | `https://huggingface.co/datasets/ScaleAI/MCP-Atlas`; harness and agent environment `https://github.com/scaleapi/mcp-atlas` at f24ba3fb0bfa484c86acb28431fad6d7282455f9 (built locally as docker/mcp_atlas.Dockerfile; the search tools of its four search-engine servers (brave-search, ddg-search, exa, oxylabs) are served by the Gemini API behind the same tool names and schemas (Google Search grounding); their page tools (ddg-search fetch_content, exa web_fetch_exa, oxylabs universal_scraper and amazon_product_scraper) run as upstream with the release's credentials))
* **MCP-Bench**: live agentic tasks over 28 real Model Context Protocol servers ([Setup Guide](evals/mcp_bench.md) | `https://github.com/Accenture/mcp-bench`); all 28 servers (built, reachable, and connected with tools per upstream utils/collect_mcp_info.py) and the 5 README-required keys needed; MCP client pinned to mcp==1.16.0 (the release upstream's mcp[cli]>=1.9.0 resolved to on the pinned commit's date, 2025-10-07); the image runs the README's Python 3.10
* **MobileGym (`mobilegym`)**: 256 held-out test tasks across 24 simulated mobile web apps, run with the canonical `GenericAgentV2` pure-GUI agent (`coord_space="norm_0_1000"`, `eval_mode="grounded"`, `judge_mode="state"`, `sample_seed=42`) and scored by deterministic state verifiers in the local image `gbench-mobilegym:5fbd3f34` ([Setup Guide](evals/mobilegym.md) | `https://github.com/Purewhiter/mobilegym` at `5fbd3f34be0e9780ae4b1e9d5089963ddbd05d38`)
* **MobileGym++ (`mobilegym_pp`)**: 215 official hybrid app-tools + GUI mobile tasks (`bench215/official.json`), run with the canonical `Qwen35SessionAgent` hybrid session runtime (`interaction_mode="hybrid"`, `coord_space="norm_0_1000"`) and scored by deterministic state verifiers in the local image `gbench-mobilegym:5fbd3f34` ([Setup Guide](evals/mobilegym_pp.md) | `https://github.com/OpenMobile2/MobileGym-plusplus` at `cdde4ad06ed67176b56964dd329b8cb8e0b0001c`)
* **NESTFUL**: nested function calling, where one call's output feeds the next call's arguments (`https://github.com/IBM/NESTFUL` at fc2c4123e73500a56185a5fb354f05d1c8b4890c; the checkout's `data_v2/nestful_data.jsonl` is the only data source)
* **NexusRaven API Evaluation (`nexusraven_api_eval`, renamed from `nexus_function_calling`)**: Zero-shot single-call function calling, NexusRaven_API_evaluation's 5 tasks ([Setup Guide](evals/nexusraven_api_eval.md) | `https://huggingface.co/datasets/Nexusflow/NexusRaven_API_evaluation` | `https://github.com/nexusflowai/NexusRaven`)
* **SEAL-Tools**: tool selection and parameter filling from a retrieved API list, including multi-tool and nested calls (`https://huggingface.co/datasets/casey-martin/Seal-Tools`)
* **SkillsBench**: 87 execution-based agentic tasks, each in its own Docker environment and scored by a pytest verifier ([Setup Guide](evals/skillsbench.md) | `https://github.com/benchflow-ai/skillsbench` at 9a1f4dd5f7659f75707435da3ce854b6e48321d1, built into the local image gbench-skillsbench:9a1f4dd5-h7)
* **Spider 2.0**: Spider 2.0-lite text-to-SQL scored by execution accuracy; its 135 local (SQLite) instances run ([Setup Guide](evals/spider2.md) | `https://github.com/xlang-ai/Spider2` at cafb867313aab4e674652054198f383cf4018943, `spider2-lite/`, a local checkout)
* **T-Eval**: six tool-use sub-skills (instruct, plan, reason, retrieve, understand, review) scored deterministically (`https://huggingface.co/datasets/lovesnowbest/T-Eval` | English and Chinese settings; Plan/Reason embedders sentence-transformers/all-mpnet-base-v2@e8c3b32e (EN) and thenlper/gte-large-zh@64c364e5 (ZH), as test.py)
* **TAU-2 Bench**: airline, retail and telecom customer-service agents that follow a domain policy ([Setup Guide](evals/tau2.md) | `https://github.com/sierra-research/tau2-bench` @ `b7ea9074c1cba482b30687fecdb5c8425fd6f619` (runs in the local gbench-tau2 image, docker/tau2.Dockerfile))
* **TAU-3 Bench**: tau3-bench banking_knowledge RAG domain (full simulator) ([Setup Guide](evals/tau3.md) | `https://github.com/sierra-research/tau2-bench` @ `b7ea9074c1cba482b30687fecdb5c8425fd6f619` (runs in the local gbench-tau2 image, docker/tau2.Dockerfile); retrieval: upstream's alltools with its dense search on gemini-embedding-001 through the Gemini API (upstream: OpenAI text-embedding-3-large); recorded, not leaderboard-comparable)
* **TerminalBench**: Terminal-Bench 2.1 tasks in sandboxed Linux containers ([Setup Guide](evals/terminal_bench.md) | `https://hub.harborframework.com/datasets` `terminal-bench/terminal-bench-2-1@sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a` (the TB2.1 leaderboard's DATASET_REF, 89 tasks); harness Harbor 0.20.0 (PyPI `harbor==0.20.0`) in the local image gbench-terminal-bench:harbor-0.20.0, gbench/docker/terminal_bench.Dockerfile)
* **ToolBench**: StableToolBench DFSDT agent loop, SoPR Average over 6 groups ([Setup Guide](evals/toolbench.md) | `https://github.com/THUNLP-MT/StableToolBench`); SoWR vs the published ChatGPT-CoT answers (`https://huggingface.co/datasets/stabletoolbench/baselines`, revision ea70398f); uncached calls go to the ToolBench real-API proxy, then the API simulator on the gbench Gemini judge cascade (Gemini API only; gpt-4-turbo upstream)
* **WildClawBench**: 60 long-horizon real-world agent tasks in 6 categories, English and Chinese (`https://huggingface.co/datasets/internlm/WildClawBench`; harness `https://github.com/internlm/WildClawBench` at 316334ccc4a87b9b5635ad73da99b4dfc0b3887e; the judge and the agent's web_search tool, OpenClaw's Brave tool, run on the Gemini API, the search on Gemini Google Search grounding; the image's DuckDuckGo and Google Custom Search skills are removed, so web_search is the agent's only search tool)

### Pillar 6: Multimodal Vision & Grounding
* **Bundled Detection**: zero-shot object detection on COCO 2017 val, pycocotools COCOeval (`http://images.cocodataset.org/zips/val2017.zip` | `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`)
* **ChartQA**: question answering over real charts (`https://huggingface.co/datasets/ahmed-masry/ChartQA`)
* **CharXiv**: academic chart understanding, descriptive and reasoning questions (`https://huggingface.co/datasets/princeton-nlp/CharXiv`)
* **COCO Caption**: image captioning (CIDEr) on the Karpathy 5k test split, lmms-eval coco_karpathy_test protocol ([Setup Guide](evals/coco_caption.md) | `https://huggingface.co/datasets/yerevann/coco-karpathy` | `https://huggingface.co/datasets/lmms-lab/COCO-Caption`)
* **DocVQA**: question answering over scanned document pages (`https://huggingface.co/datasets/lmms-lab-encoder/DocVQA`)
* **InfographicVQA**: question answering over infographics (`https://huggingface.co/datasets/lmms-lab-encoder/DocVQA` (config InfographicVQA))
* **MMMU Pro**: multimodal multidisciplinary reasoning, Standard-10 and Vision settings (`https://huggingface.co/datasets/MMMU/MMMU_Pro`)
* **OmniDocBench v1.6**: document page parsing to Markdown, with tables as HTML and formulas as LaTeX (`https://huggingface.co/datasets/opendatalab/OmniDocBench`)
* **ScreenSpot V2 (`screenspot`)**: GUI grounding with a mouse-click tool on the official ScreenSpot-v2 release ([Setup Guide](evals/screenspot.md) | `https://huggingface.co/datasets/OS-Copilot/ScreenSpot-v2`)
* **ScreenSpot-v2 pointing prompt (`screenspot_pointing`, renamed from `semantic_keypoint`)**: ScreenSpot-v2 (official release) with a gbench pointing prompt ([Setup Guide](evals/screenspot_pointing.md) | `https://huggingface.co/datasets/OS-Copilot/ScreenSpot-v2`)
* **TextVQA**: question answering that requires reading text in natural images (`https://huggingface.co/datasets/lmms-lab/textvqa`)
* **OSWorld Desktop**: desktop computer use in an Ubuntu VM, 369 tasks scored by OSWorld's per-task evaluators; not in `--evals all`, runs when named; hard-errors `infra_required` without /dev/kvm (nested virtualization), Docker, the images and the Ubuntu qcow2 ([Setup Guide](evals/ui_control_osworld.md) | `https://github.com/xlang-ai/OSWorld` @ `fc31a9049664292fcb35d6e501ee1dc839f2cf6d`)

## 4. Agentic quality and capabilities (Tiers 2 and 3)
* **GemmaClaw / OpenClaw QA suite**: multi-turn agent scenarios (session recall, MCP plugin routing): the QA scenarios of the pinned GemmaClaw commit, 60 at d8bc6989, with the scenarios that need another model provider listed apart and not scored; its alternate, image-generation and embedding models run on the Gemini API through a gbench key relay, and the suite runs in gbench's own Docker container without network access ([docs/quality.md](quality.md))
* **Golden Set**: 16 deterministic capability checks (`gbench/golden_dataset/`); a healthy model passes every case that applies to it (a case for a modality the model lacks is not applicable and leaves the denominator)

## 5. Gemma models and datasets
* **Gemma on Hugging Face**: Google's Gemma model weights (`https://huggingface.co/google`)
* **ShareGPT V3 dataset**: multi-turn conversations for the `chat-like` workload campaign (`https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered`)
* **MLPerf Inference**: gbench's stress sweep drives the server with the same open-loop Poisson arrivals as MLPerf's server scenario (`https://mlcommons.org/benchmarks/inference/`)
