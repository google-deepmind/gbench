# Academic and custom eval suites (`--evals`)

Tier 1 of the capability evaluation: every built-in suite, custom plugins and JSONL suites. Each suite has its own setup and scoring guide in [evals/](evals/).

[Back to the README](../README.md)

## Tier 1: academic and custom evals (`--evals`)

Runs 86 canonical Python benchmarks under `--evals all` (plus `ui_control_osworld` and `swe_lancer`, which run only when named), and custom plugins, against local or remote `/v1` endpoints, in 6 pillars:

| Pillar | Suite ID | Task name | Description | Special options |
| :--- | :--- | :--- | :--- | :--- |
| **1. General Knowledge & Scientific Reasoning** | `arc_agi` | ARC-AGI | ARC-AGI-1 public evaluation set: infer a grid transformation from a few examples and produce the output grids | `--eval-thinking` |
| | `causalbench` | CausalBench | Corr2Cause: causal inference from stated correlations (F1 of the valid class) | Standard |
| | `cimemories` | CIMemories | CIMemories Violation@5 and Completeness via the release pipeline (gold labels and REVEAL judge on the Gemini cascade) ([Setup Guide](evals/cimemories.md)). The accuracy column is Completeness; the summary's Headline Metrics column (eval_summary.csv, and a `headline:` line in the console table) shows both official numbers, Violation@5 (lower is better) and Completeness, which the result JSON carries as `headline_metrics`. | Standard |
| | `cyberseceval` | CyberSecEval | CyberSecEval-instruct (Insecure Code): the generated code for instruct-v2.json's 1,681 prompts scored by Meta's Insecure Code Detector (regex + semgrep 1.51.0) in the local image gbench-cyberseceval:114d02388213-semgrep-1.51.0 (Docker) ([Setup Guide](evals/cyberseceval.md)) | Standard |
| | `frames` | FRAMES | 824 multi-hop factual questions whose answers combine facts from 2-15 Wikipedia articles | `--eval-thinking` |
| | `gpqa` | GPQA | Graduate-level science multiple choice, full split (simple-evals protocol, 10 repeats) | `--eval-thinking` |
| | `gpqa_diamond` | GPQA Diamond | Graduate-level science multiple choice, the 198-question Diamond set (simple-evals protocol, 10 repeats) | `--eval-thinking` |
| | `healthbench` | HealthBench | 5,000 health conversations scored against physician-written rubrics | Standard |
| | `humanitys_last_exam` | Humanity's Last Exam | 2,500 expert-written closed-ended questions across many subjects, 342 of them with an image | `--eval-thinking` |
| | `i18n_translate` | I18N Translation | WMT19 news test sets (newstest2019), all 19 directions, sacreBLEU BLEU (headline = mean over directions) and chrF | Standard |
| | `ifeval` | IFEval | 541 prompts with verifiable instructions (length, format, forbidden words and similar) | Standard |
| | `lab_bench` | LAB-Bench | LAB-Bench: all 8 categories (LitQA2, DbQA, SuppQA, FigQA, TableQA, ProtocolQA, SeqQA, CloningScenarios), zero-shot CoT baseline; FigQA/TableQA need an image-capable endpoint | `--eval-thinking` |
| | `livebench` | LiveBench | LiveBench's 6 core categories at the pinned release 2024-11-25, scored by LiveBench's own ground-truth graders (runs the served model's own LiveBench config when LiveBench ships one, e.g. gemma-4-31b-it) | `--eval-thinking` |
| | `medxpertqa` | MedXpertQA | Expert-level medical exam QA, Text + MM subsets (official two-turn zero-shot CoT) | `--eval-thinking` |
| | `mmlu` | MMLU | MMLU (hendrycks/test 5-shot, answer-letter next-token log-probabilities via /v1/completions logprobs=100; a vLLM server needs --max-logprobs 100, which the eval server gBench starts is given; thinking is not applied) | `--eval-n-shot` |
| | `mmlu_pro` | MMLU-Pro | Multi-discipline professional reasoning, 14 subjects and 10 options ( TIGER-Lab evaluate_from_apiX.py, 5-shot CoT, top_p 0.95) | `--eval-thinking`, `--eval-n-shot` |
| | `mmlu_redux` | MMLU-Redux | MMLU-Redux-2.0 error_type ok rows, HELM MMLU 5-shot protocol (as the MMLU-Redux authors score) | Standard |
| | `multilingual_mmlu` | Multilingual MMLU | Human-translated MMLU in 14 languages (openai/MMMLU, simple-evals protocol) | Standard |
| | `simpleqa` | SimpleQA | 4,326 short-form factual questions; not answering scores better than a wrong answer | Standard |
| | `wildbench` | WildBench | Real-user multi-turn tasks: checklist-judged WB_score.task_macro (headline) and WB_score, pairwise WB-Reward against the three leaderboard reference models, and WB-Elo (Raw and LC) over the leaderboard's 63-model pool re-graded by the Gemini judge (all 1,024 WildBench v2 tasks; renamed from `lmsys_noncoding_hard`) ([Setup Guide](evals/wildbench.md)) | Standard |
| | `wmdp` | WMDP | Weapons of Mass Destruction Proxy: hazardous knowledge in biosecurity, chemical security and cybersecurity, zero-shot log-likelihood (lm-eval v0.4.2 request: echoed prompt logprobs via /v1/completions, max_tokens=0) | Standard |
| **2. Mathematics & Proofs** | `aime` | AIME | AIME 2026 on MathArena's protocol (MathArena set and prompt, 4 runs, last-chance reprompt, MathArena's own grader in the local `gbench-matharena:b89f2f0a` image; integer 0-999) | `--eval-thinking` |
| | `gsm8k` | Grade School Math 8K | 1,319 grade-school arithmetic word problems (lm-eval gsm8k_cot, 8-shot chain of thought) | `--eval-n-shot` |
| | `hmmt` | HMMT | HMMT February 2026 on MathArena's protocol (MathArena set and prompt, 4 runs, last-chance reprompt, MathArena's own grader in the local `gbench-matharena:b89f2f0a` image) | `--eval-thinking` |
| | `imo_answer_bench` | IMO-AnswerBench | 400 olympiad problems with a short final answer (Google DeepMind IMO-Bench) | `--eval-thinking` |
| | `amc_aime` | AMC / AIME Combined | gBench composite: AMC 12 2022-2023 + AIME 2022-2024 (contaminated reference, not canonical) | `--eval-thinking` |
| | `putnam` | PutnamBench | Putnam Competition problems solved in natural language (a gBench protocol over the pinned PutnamBench release's informal problems; not leaderboard-comparable) | `--eval-thinking` |
| | `putnam_formal` | Putnam-Formal | PutnamBench Lean 4 track, Lean with answer + Lean no answer (672 each), kernel-checked; no-answer answers the kernel cannot decide are reviewed by the Gemini judge cascade (needs `GEMINI_API_KEY`) ([Setup Guide](evals/putnam_formal.md)) | `--eval-thinking` |
| **3. Coding & Algorithmic Design** | `aider_polyglot` | Aider Polyglot | 225 Exercism exercises in 6 languages, edited in aider's native edit format ([Setup Guide](evals/aider_polyglot.md)) | `--eval-thinking` |
| | `bigcodebench` | BigCodeBench | Instruct (headline) and Complete, plus BigCodeBench-Hard from the same run; prefilled per upstream's default path on no-think runs; `--eval-thinking` runs use the openai backend's request for a served model (top_p 0.95, the run's token budget; temperature 1.0, as every `--eval-thinking` run) ([Setup Guide](evals/bigcodebench.md)) | `--eval-thinking` |
| | `codeforces` | Codeforces | open-r1/codeforces verifiable test split (natural CF/ICPC/IOI mix; pass rate, not Elo) | `--eval-thinking` |
| | `copilot_bench_swe`| SWE-bench Verified | 500 human-validated GitHub issues from 12 Python repositories, resolved by an agent and scored by the swebench harness ([Setup Guide](evals/copilot_bench_swe.md)) | `--eval-thinking` |
| | `cruxeval` | CRUXEval | Code reasoning: CRUXEval-I + CRUXEval-O (input and output prediction, execution-checked; 10 samples; pass@1 at T=0.2, pass@5 at T=0.8) | Standard |
| | `lcb` | LiveCodeBench | Code generation headline (lcb_runner protocol, 10 samples per problem) + code execution + test output prediction + self-repair, reported separately | `--eval-thinking` |
| | `multi_swe_bench` | Multi-SWE-bench | Issue resolution in 7 languages (1,632 instances) ([Setup Guide](evals/multi_swe_bench.md)) | `--eval-thinking` |
| | `multipl_e` | MultiPL-E | HumanEval translated into 23 programming languages, scored by execution (pass@1) ([Setup Guide](evals/multipl_e.md)) | Standard |
| | `ojbench` | OJBench | 232 NOI and ICPC problems, each posed in Python and C++, judged in the DMOJ sandbox ([Setup Guide](evals/ojbench.md)) | `--eval-thinking` |
| | `scicode` | SciCode | Research-level scientific computing problems (65 main problems, 288 scored sub-steps), scored by execution ([Setup Guide](evals/scicode.md)) | `--eval-thinking` |
| | `swe_bench_live` | SWE-bench Live | Live GitHub issue resolution: every leaderboard tab, Lite (300), Full (1,319), Verified (500), C/C++ (253), C# (136), Java (160), TS/JS (219), Go (138), Rust (171) and Windows (66; needs a Windows Docker host and a passing Windows metadata server check, reported `infra_required` in its own entry otherwise), each with its own resolved-rate headline (`headline_metrics`) ([Setup Guide](evals/swe_bench_live.md)) | `--eval-thinking`, `--eval-categories` |
| | `swe_bench_multilingual` | SWE-bench Multilingual | 300 GitHub issues from 41 repositories in 9 languages, resolved by an agent ([Setup Guide](evals/swe_bench_multilingual.md)) | `--eval-thinking` |
| | `swe_bench_pro` | SWE-bench Pro V2 | Scale AI's public set of 642 long-horizon software engineering tasks (Go, Python, JavaScript, TypeScript) ([Setup Guide](evals/swe_bench_pro.md)) | `--eval-thinking` |
| | `swe_lancer` | SWE-Lancer | Real-world freelance software engineering tasks run with SWE-Lancer's own agent (SimpleAgentSolver) and harness (openai/frontier-evals) on the official task set, 198 IC SWE and 265 SWE Manager Diamond tasks, headlines pass@1 and dollars earned per task type; not in `--evals all` (runs when named: the harness edits the Docker host's iptables rules and pulls about 1.3 TB of task images); hard-errors `infra_required` without Docker, the docker socket, the locally built `gbench-swe-lancer-harness:51052ced` image or iptables access to the host's `DOCKER-USER` chain ([Setup Guide](evals/swe_lancer.md)) | `--eval-thinking` |
| **4. Long Context & Retrieval** | `aa_lcr` | AA-LCR | Artificial Analysis Long Context Reasoning v1.1: 100 questions, each over a document set of about 100k tokens | Standard |
| | `beam_128k` | BEAM 128k | BEAM's 128K tier: long-term memory questions over long multi-session conversations | Standard |
| | `loft_scifact` | LOFT SciFact (128k) | In-context retrieval over a 128k passage corpus (LOFT recall@1; renamed from `loft_x_arxiv`) ([Setup Guide](evals/loft_scifact.md)) | Standard |
| | `longbench_v2_code` | LongBench-v2 Code | LongBench-v2 Code Repository Understanding domain (long-context MCQ, non-CoT and w/ CoT settings; renamed from `culer`) ([Setup Guide](evals/longbench_v2_code.md)) | Standard |
| | `mrcr` | MRCR | Multi-round co-reference resolution, 8 bins from 4K to 1M tokens (those that fit the served window) | Standard |
| | `ruler` | RULER (4k-128k) | 13 retrieval, tracking and aggregation tasks at 6 context bands (4k, 8k, 16k, 32k, 64k, 128k) | Standard |
| **5. Tool Use & Agentic Workflows** | `acebench` | ACEBench | Function calling and tool use: Normal, Special and Agent (multi-step, and multi-turn with a user simulator answered by the Gemini judge cascade, needs `GEMINI_API_KEY`; without it agent_multi_turn alone is reported infra_required and the other categories run), English (headline) and Chinese settings; headline the leaderboard overall ([Setup Guide](evals/acebench.md)) | `--eval-thinking` |
| | `agent_dojo` | AgentDojo | Prompt-injection security of tool-calling agents in 4 task suites (workspace, travel, banking, slack) | Standard |
| | `api_bank` | API-Bank | Multi-level API calling, tool retrieval and response synthesis; its Translate API runs upstream's pinned googletrans 3.1.0a0 in the local image gbench-api-bank:googletrans-3.1.0a0 (Docker) ([Setup Guide](evals/api_bank.md)) | Standard |
| | `bfcl` | Berkeley Function-Call | All 22 BFCL v4 scoring categories: 13 single-turn (Non-Live + Live AST, irrelevance, relevance), 4 multi-turn run in bfcl-eval's own harness, 5 agentic (via bfcl_v4_agentic); BFCL FC protocol, headline BFCL Overall Acc (10/10/10/30/40); needs Docker with the local harness image gbench-bfcl:2026.3.23 (gbench/docker/bfcl.Dockerfile), and its Web Search setting a `GEMINI_API_KEY` (the web search is served by Gemini Google Search grounding in place of BFCL's DuckDuckGo via SerpAPI search, recorded and not leaderboard-comparable; without a working key that setting alone is reported infra_required and the rest is scored) ([Setup Guide](evals/bfcl.md)) | `--eval-thinking`, `--eval-categories` |
| | `bfcl_v4_agentic` | BFCL v4 Agentic | Agentic memory and web-search tool use (BFCL Agentic score); web_search is served by Gemini Google Search grounding in place of BFCL's DuckDuckGo via SerpAPI search and needs a `GEMINI_API_KEY` (recorded, not leaderboard-comparable; without a key, with a rejected one, or when a search no grounding model served during the run, the Web Search setting is reported infra_required and the memory categories are still scored); runs bfcl-eval in the local image gbench-bfcl:2026.3.23 | `--eval-categories` |
| | `bfcl_live` | BFCL Live | BFCL Live categories (AST + irrelevance/relevance); headline Live Overall Acc, scored by bfcl-eval in the local image gbench-bfcl:2026.3.23 (renamed from `bfcl_v3_live`) ([Setup Guide](evals/bfcl_live.md)) | `--eval-thinking` |
| | `browsecomp` | BrowseComp | 1,266 hard web-research questions with short answers, graded by BrowseComp's LLM grader | Standard |
| | `complexfuncbench` | ComplexFuncBench | THUDM multi-step, multi-constraint function calling over the Booking.com API; needs `RAPID_API_KEY` (booking-com15 on RapidAPI, upstream's response-based tier) and `GEMINI_API_KEY` (the judges) ([Setup Guide](evals/complexfuncbench.md)) | Standard |
| | `deepsearch_qa` | DeepSearchQA | Google DeepMind multi-step web research with set answers (Fully Correct, F1) | Standard |
| | `gaia` | GAIA | General AI assistants: web research, file and media reading and multi-step reasoning, each question with one short answer; the validation split is scored locally, and GBENCH_GAIA_SPLIT=test writes the leaderboard submission file for the test split ([Setup Guide](evals/gaia.md)) | Standard |
| | `gaia2` | GAIA-2 | Meta's Agents Research Environments (ARE) scenarios; canonical when provisioned; hard-errors `infra_required` if unprovisioned ([Setup Guide](evals/gaia2.md)) | Standard |
| | `gdpval` | GDPval | Wins-or-Ties Rate (GDPval win rate) vs Expert Reference Deliverables, 3 samples per task ([Setup Guide](evals/gdpval.md)) | Standard |
| | `gorilla_apibench` | Gorilla APIBench | One-line calls to real machine-learning APIs (TorchHub, TensorHub, HuggingFace); 0-shot (headline) and the BM25, GPT-Index and Oracle retriever settings ([Setup Guide](evals/gorilla_apibench.md)) | Standard |
| | `mcp_atlas` | MCP-Atlas | Multi-server Model Context Protocol tool use in the release's agent environment (locally built image running 35 of its 36 servers; only the search tools of its four search-engine servers (brave-search, ddg-search, exa, oxylabs) are served by the Gemini API behind the same tool names and argument schemas (Google Search grounding; brave-search, whose tools are all search tools, does not run), so results are not leaderboard-comparable; their page tools run as upstream on the release's servers (ddg-search's fetch_content with no key, exa's web_fetch_exa with EXA_API_KEY, oxylabs' universal_scraper and amazon_product_scraper with OXYLABS_USERNAME / OXYLABS_PASSWORD); the mongodb server's data comes from a locally built MongoDB image unless MONGODB_CONNECTION_STRING is set; hard-errors `infra_required`, naming each server's unset variables and sample data, until every server credential is set; they go only to a container of their own that runs no local shell, code or file tool, never to the container the model's shell and code tools run in; a page-tool reply saying Exa or Oxylabs could not be reached or refused the run's account is a harness error, never the model's outcome) ([Setup Guide](evals/mcp_atlas.md)) | Standard |
| | `mcp_bench` | MCP-Bench | Live agentic tasks over 28 real Model Context Protocol servers; all 104 tasks, which needs the 5 server keys (each accepted by its service in a live check before the run) and every server built, reachable and passing MCP-Bench's own connection check (28/28 connected, each listing tools), else hard-errors `infra_required`; the judge's Gemini key stays in gBench (the container gets a relay placeholder), the five server keys go only to a separate credentialed-servers container (never to the container where model text runs), and tasks a harness error fell in are left out of the score ([Setup Guide](evals/mcp_bench.md)) | Standard |
| | `mobilegym` | MobileGym | 256 held-out test tasks (`bench_env/splits/test.txt`) across 24 simulated mobile web apps, run with the canonical `GenericAgentV2` pure-GUI agent (`coord_space="norm_0_1000"`, `eval_mode="grounded"`, `judge_mode="state"`, `sample_seed=42`) in the local image `gbench-mobilegym:5fbd3f34` and scored by deterministic state verifiers ([Setup Guide](evals/mobilegym.md)) | `--eval-thinking` |
| | `mobilegym_pp` | MobileGym++ | 215 official tasks (`bench215/official.json`) across 24 simulated mobile web apps, run with the canonical `Qwen35SessionAgent` hybrid app-tools + GUI session runtime (`interaction_mode="hybrid"`, `coord_space="norm_0_1000"`, `judge_mode="state"`, `sample_seed=42`) in the local image `gbench-mobilegym:5fbd3f34` and scored by deterministic state verifiers ([Setup Guide](evals/mobilegym_pp.md)) | `--eval-thinking` |
| | `nestful` | NESTFUL | Nested function calling: call sequences in which one call's output feeds the next call's arguments (IBM's scorer) | Standard |
| | `nexusraven_api_eval` | NexusRaven API Evaluation | Zero-shot single-call function calling on NexusRaven_API_evaluation's 5 tasks (Generic and Cybersecurity domain accuracies as the NexusRaven repo publishes them, headline their mean; renamed from `nexus_function_calling`) ([Setup Guide](evals/nexusraven_api_eval.md)) | Standard |
| | `seal_tools` | SEAL-Tools | Tool selection and parameter filling from a retrieved API list, including multi-tool and nested calls | Standard |
| | `skillsbench` | SkillsBench | 87 execution-based agentic tasks, each in its own Docker environment and scored by a pytest verifier ([Setup Guide](evals/skillsbench.md)) | Standard |
| | `spider2` | Spider 2.0 | Spider 2.0-lite text-to-SQL scored by execution accuracy; the 135 local (SQLite) instances run ([Setup Guide](evals/spider2.md)) | Standard |
| | `t_eval` | T-Eval | Six tool-use sub-skills (instruct, plan, reason, retrieve, understand, review) scored deterministically; English (headline) and Chinese settings ([Setup Guide](evals/t_eval.md)) | Standard |
| | `tau2` | TAU-2 Bench | Airline, retail and telecom customer-service agents that follow a domain policy; full tau2-bench simulator at a pinned revision in the local gbench-tau2 image, pass^1 over 4 trials (tau2-bench leaderboard Overall) ([Setup Guide](evals/tau2.md)) | Standard |
| | `tau3` | TAU-3 Bench | tau3-bench banking_knowledge RAG domain (full tau2-bench simulator in the local gbench-tau2 image, upstream's alltools retrieval with its dense search on gemini-embedding-001 in place of OpenAI text-embedding-3-large, recorded and not leaderboard-comparable; needs `GEMINI_API_KEY`), pass^1 over 4 trials ([Setup Guide](evals/tau3.md)) | Standard |
| | `terminal_bench` | TerminalBench | Terminal-Bench 2.1 tasks in sandboxed Linux containers, run with Harbor ([Setup Guide](evals/terminal_bench.md)) | `--eval-thinking` |
| | `toolbench` | ToolBench | StableToolBench DFSDT agent loop; SoPR Average over the 6 test groups, SoWR against the published ChatGPT-CoT answers ([Setup Guide](evals/toolbench.md)) | Standard |
| | `ui_control_osworld` | OSWorld | Desktop computer use: OSWorld's 369 tasks in an Ubuntu VM, scored by its deterministic per-task evaluators; not in `--evals all` (runs when named), hard-errors `infra_required` without `/dev/kvm`, Docker, the locally built orchestrator image, the task VM image and the Ubuntu qcow2 ([Setup Guide](evals/ui_control_osworld.md)) | Standard |
| | `wildclawbench` | WildClawBench | 60 long-horizon real-world agent tasks in 6 categories (English and Chinese), each in its own Docker container ([Setup Guide](evals/wildclawbench.md)) | Standard |
| **6. Multimodal Vision & Grounding**| `bundled_detection`| Object Detection | Zero-shot COCO val2017 detection on all 5,000 images, scored by pycocotools COCOeval (AP@[.50:.95]; gBench prompt, not leaderboard-comparable) | `--eval-max-soft-tokens` |
| | `chartqa` | ChartQA | Question answering over real charts, 2,500 questions (original PNG bytes) | Standard |
| | `charxiv` | CharXiv | Academic chart understanding over 1,000 charts: descriptive and reasoning questions, graded with the upstream rubrics | Standard |
| | `coco_caption` | COCO Caption | Image Captioning, COCO Karpathy 5k test split with the lmms-eval coco_karpathy_test protocol (CIDEr headline; BLEU-1..4, METEOR, ROUGE-L) ([Setup Guide](evals/coco_caption.md)) | Standard |
| | `docvqa` | DocVQA | Question answering over scanned document pages, 5,349 validation questions (original image bytes) | Standard |
| | `infographicvqa` | InfographicVQA | Question answering over infographics, 2,801 validation questions (original image bytes) | `--eval-max-soft-tokens` |
| | `mmmu_pro` | MMMU-Pro | Standard-10 + Vision settings, CoT + Direct (official MMMU-Pro prompts and parser) | `--eval-thinking`, `--eval-max-soft-tokens` |
| | `omnidocbench` | OmniDocBench v1.6 | Document page parsing to Markdown, with tables as HTML and formulas as LaTeX | Standard |
| | `screenspot` | ScreenSpot V2 | ScreenSpot-v2 (official OS-Copilot release) with a gBench mouse-click tool prompt: click accuracy plus the six Mobile/Desktop/Web x Text/Icon cells (not leaderboard-comparable: gBench prompt); a local ScreenSpot-Pro dir only when GBENCH_SCREENSPOT_DATA_DIR names one ([Setup Guide](evals/screenspot.md)) | `--eval-max-soft-tokens` |
| | `screenspot_pointing` | ScreenSpot-v2 pointing prompt | ScreenSpot-v2 (official OS-Copilot release) with a gBench text-pointing prompt: click accuracy plus the six Mobile/Desktop/Web x Text/Icon cells (not leaderboard-comparable: gBench prompt; renamed from `semantic_keypoint`) | `--eval-max-soft-tokens` |
| | `textvqa` | TextVQA | Reading text in natural images (scene OCR), 5,000 validation questions (original image bytes) | `--eval-max-soft-tokens` |
| **Meta targets** | `all` | All built-in suites | Runs the 86 suites of `--evals all` across the 6 pillars (not `swe_lancer` or `ui_control_osworld`) | All built-in suites |
| | `plugins` | All plugins | Runs every plugin discovered in `--eval-plugins-dir` | Discovered plugins |

```bash
# 1. Run all standard built-in academic evals with Chain-of-Thought reasoning
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 --tokenizer google/gemma-4-E4B-it --evals all --eval-thinking --max-output-tokens 65536

# 2. Run vision evals (with --remote-endpoint the image soft-token budget is fixed server-side;
#    --eval-max-soft-tokens applies only when gbench launches its own vLLM server)
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 --tokenizer google/gemma-4-E4B-it --evals mmmu_pro screenspot textvqa --max-output-tokens 65536

# 3. Discover and run custom plugin suites from a directory
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 --tokenizer google/gemma-4-E4B-it \
       --eval-plugins-dir ./my_plugins/ --evals plugins --max-output-tokens 65536

# 4. Ingest and evaluate generic JSONL benchmark datasets
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 --tokenizer google/gemma-4-E4B-it \
       --eval-custom-jsonl ./custom_jsonl_data/my_eval.jsonl --evals custom_jsonl --max-output-tokens 65536
```

### Custom evaluation plugins and JSONL suites

A plugin adds your own benchmark or dataset as an eval suite without changing gBench's code:

> **Ready-to-run example:** [`examples/custom_evals/`](../examples/custom_evals/) ships a complete, dependency-free plugin (`custom_qa_eval.py`) and a `custom_jsonl` dataset (`sample_benchmark.jsonl`), with a short README explaining when to use each. See also [`docs/evals/plugins.md`](evals/plugins.md).

#### 1. Writing a Python evaluation plugin (`--eval-plugins-dir <dir>`)

To create a new evaluation suite, place a Python file (e.g. `my_domain_eval.py`) in your custom plugins directory. gBench dynamically registers it as a full CLI target:

```python
# my_plugins/my_domain_eval.py
from typing import Any, Dict, List, Optional, Tuple
from gbench.runners.eval_suites.base import run_eval_suite

# Define the reporting pillar header in CLI results table
PILLAR = "Custom Domain Knowledge"

def _load_my_domain_samples(
    enable_thinking: bool = False,
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load evaluation samples from local files, databases, or APIs."""
    raw_data = [
        {"q": "What is the primary law of thermodynamics?", "a": "First Law", "cat": "Physics"},
        {"q": "What is the capital of Australia?", "a": "Canberra", "cat": "Geography"},
    ]
    samples = []
    for item in raw_data:
        prompt = f"Question: {item['q']}\nAnswer in one short phrase."
        if enable_thinking:
            prompt += "\nThink step by step before answering."
        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, item["a"], {"category": item["cat"]}))
    return samples[:limit] if limit else samples

def _eval_my_domain(response_text: str, gold_answer: str) -> bool:
    """Return True if model response matches gold reference."""
    return gold_answer.strip().lower() in response_text.strip().lower()

def run_my_domain(
    model_name: str,
    base_url: str,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Entrypoint called by gbench (must match 'run_<eval_name>')."""
    samples = _load_my_domain_samples(enable_thinking=enable_thinking, limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="my_domain",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_my_domain,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),  # the run's --max-output-tokens
    )
```

**Run your plugin:**
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 --tokenizer google/gemma-4-E4B-it \
       --eval-plugins-dir ./my_plugins/ --evals my_domain --max-output-tokens 65536
```

#### 2. JSONL datasets without code (`--eval-custom-jsonl <file>`)

For a simple prompt/answer dataset without Python code, write one `.jsonl` file (one JSON object
per line) and pass it with `--eval-custom-jsonl`, which takes a single file:

```jsonl
{"prompt": "What is the speed of light (m/s)?", "gold": "299792458", "eval_type": "numeric", "category": "Physics"}
{"messages": [{"role": "user", "content": "15 * 14 = ?"}], "gold": "210", "eval_type": "numeric", "category": "Math"}
{"prompt": "Which planet is the Red Planet? (A) Venus (B) Mars", "gold": "B", "eval_type": "multiple_choice", "category": "Astronomy"}
{"prompt": "What is the capital of Australia?", "gold": "Canberra", "category": "Geography"}
```

Each row needs the request (`prompt`, or a chat-style `messages` list; `question` and `input` are
read as the prompt too; a line with none of them stops the run with an error naming the line) and the expected
answer (`gold`, else `answer`, `gold_answer` or `target`). A row with no gold cannot be scored and
counts as incorrect. `category` (or `domain`) groups rows in `category_accuracy`.

**Supported `eval_type` values** (deterministic scoring, per row):

| `eval_type` | Passes when |
| :--- | :--- |
| `contains` (default) | the response equals the gold or contains it as a whole token (case-insensitive; no letter, digit or underscore right before or after it, so `4` is not found in `14`); failing that, a one-letter gold passes when the model's stated choice is that letter, and a numeric gold when some number in the response equals it within 1e-4 |
| `exact` | the whole response, surrounding whitespace stripped, equals the gold (case-insensitive) |
| `numeric` | some number in the response equals the gold within 1e-4; a number with commas is read both ways, as one number with comma-grouped thousands (`299,792,458`, `1,234.5`) and as a list (`85,90,100`), and passes when either reading holds the gold |
| `multiple_choice` (also `mc`, `letter`) | the model's stated choice (a boxed letter, else the letter after an answer anchor, else the last standalone letter A-J) is the gold letter |

Any other `eval_type` value stops the run with an error naming the line and the allowed values;
it is never scored as `contains`. Under `contains` a one-letter gold still passes a response that
uses that letter as a word (`B` in `plan B`), so give such rows `multiple_choice`; give numeric
golds `numeric`. Format and scoring details:
[docs/evals/custom_jsonl.md](evals/custom_jsonl.md).

**Run your JSONL datasets:**
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 --tokenizer google/gemma-4-E4B-it \
       --eval-custom-jsonl ./custom_jsonl_data/my_eval.jsonl --evals custom_jsonl --max-output-tokens 65536
```

### Continuous evaluation improvement and roadmap

The eval suite is a living system: new benchmarks are added and existing ones hardened over
time, and every suite is held to a hard standard: it is either a faithful *canonical-when-provisioned*
measurement or it hard-errors (`infra_required`); it never emits a skipped, guessed, or
fabricated number. The standing plan, which lays out the non-negotiable standards, the engineering playbook for
adding/hardening an eval (container-delegation, docker-out-of-docker identity-mount, judge-cascade
reuse, model wiring), and the prioritized roadmap of evals to build next, lives in
[`docs/evals_roadmap.md`](evals_roadmap.md). The current first item is a full canonical
WebArena harness (formerly the `lmarena_web_agent` suite). Two suites are **deregistered and
tracked roadmap-only** rather than shipping runnable placeholders: `lmarena_web_agent` (to be replaced by the
WebArena harness, never built) and `cybergym` (planned canonical execution-verified CyberGym; the text
proxy that shipped under this name was removed). Naming one with `--evals` exits with an error
that points at its roadmap entry. Two implemented suites are registered but left out of `--evals all`
and run only when named: `ui_control_osworld` (needs a `/dev/kvm` nested-virt host, absent on
accelerator serving boxes) and `swe_lancer` (its harness edits the Docker host's iptables rules while it runs and pulls
about 1.3 TB of task images); each hard-errors `infra_required` naming the missing prerequisite. See the
roadmap for each one's status and buildable path. The same document also tracks tooling that is not
part of this release: the web results dashboard, its run launcher and its hosted deployment
([section 8](evals_roadmap.md#8-web-dashboard-and-hosted-deployment-not-shipped)).
