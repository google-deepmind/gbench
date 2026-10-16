# Third-party datasets: licenses, gated access, redistribution

gbench does **not** vendor or redistribute benchmark data. Each eval suite pulls its dataset at
runtime from its upstream source (mostly the Hugging Face Hub via `datasets.load_dataset`), into
your local HF cache. Running a benchmark and publishing a score is therefore subject to **that
dataset's own license and terms of use**, not gbench's Apache-2.0 license.

This inventory covers the datasets gbench loads via `load_dataset` / `hf_hub_download`, plus the
pinned upstream data files a few suites download from GitHub (listed with their repo). Suites that
instead stand up a container or clone an upstream harness (the SWE-bench family, `spider2`,
`terminal_bench`, `mcp_bench`, `wildclawbench`, `gaia2`, `skillsbench`, `tau2` / `tau3` (tau2-bench,
MIT, cloned into the local gbench-tau2 image), `mobilegym` / `mobilegym_pp` (MobileGym and
MobileGym++, Apache-2.0, cloned into the local `gbench-mobilegym:5fbd3f34` image), `swe_lancer`,
`ui_control_osworld`, etc.) carry their own prerequisites and upstream licenses in their
`docs/evals/<suite>.md`.

> Snapshot taken 2026-09-14 by querying the HF Hub (the `openai/gdpval` and `bigcode/bigcodebench*`
> entries on 2026-09-28). **License tags and gated status can change
> upstream** - re-verify before a release, and have counsel confirm that evaluation use and the
> publication of accuracy numbers are permitted for each dataset your run touches.

## Requires action before running / publishing

### Gated: need a Hugging Face token and acceptance of the dataset's terms
A gated dataset returns HTTP 401/403 until you accept its terms on the dataset page and run with a
token (`huggingface-cli login` or `HF_TOKEN`). `--evals all` will hard-error on these until then.

| Dataset | Suite(s) | License | Note |
|---|---|---|---|
| `Idavidrein/gpqa` | `gpqa`, `gpqa_diamond` | cc-by-4.0 | Ships a canary GUID; terms forbid training on / leaking the data. |
| `cais/hle` | `humanitys_last_exam` | mit | Gated access; "Humanity's Last Exam". |
| `gaia-benchmark/GAIA` | `gaia` | none declared | Gated; treat as all-rights-reserved (see below). The suite downloads 2023/validation, and with GBENCH_GAIA_SPLIT=test also 2023/test (gated; answers hidden). |

### Non-commercial (NC): evaluation/research use only, no commercial use
| Dataset | Suite(s) | License |
|---|---|---|
| `MathArena/hmmt_feb_2026` | `hmmt` | cc-by-nc-sa-4.0 |
| `MathArena/aime_2026` | `aime` | cc-by-nc-sa-4.0 |
| `facebook/CIMemories` | `cimemories` | cc-by-nc-4.0 (the suite also uses prompts and persona texts from GitHub `facebookresearch/CIMemories` @e39806e9, CC BY-NC 4.0, fetched at run time) |
| `Nexusflow/NexusRaven_API_evaluation` (incl. the `raw_queries` config) | `nexusraven_api_eval` | cc-by-nc-4.0 (stated in the dataset card; no Hub license tag) |
| GitHub `Purewhiter/mobilegym` release `data-v0.1.0`: `mobilegym-data-v0.1.0.tar.gz` (sha256 `933a8499c71450f135eda70f9107af72b56b42181fc62de76ed1aa4c513a603c`, baked into the local `gbench-mobilegym:5fbd3f34` image) | `mobilegym`, `mobilegym_pp` | cc-by-nc-4.0 (`LICENSE-DATA`; the simulator and harness code in `Purewhiter/mobilegym` @5fbd3f34 and `OpenMobile2/MobileGym-plusplus` @cdde4ad0 are Apache-2.0 / MIT) |

### Copyleft (GPL): redistribution/derivative implications
| Dataset | Suite(s) | License |
|---|---|---|
| `ahmed-masry/ChartQA` | `chartqa` | gpl-3.0 |

### No license declared / unknown: treat as all-rights-reserved
No SPDX tag on the Hub means **no grant of rights**; do not assume redistribution or even
evaluation use is permitted. Confirm with the upstream authors or counsel before relying on these.

`He-Ren/OJBench_testdata` (`ojbench`),
`allenai/WildBench` (`wildbench`; the card text says CC BY but there is
no Hub license tag), `allenai/WildBench-V2-Model-Outputs` (`wildbench` WB-Reward reference replies and
the WB-Elo pool replies of 63 models; no license tag and no license text on the card), GitHub `trishullab/PutnamBench` @97b5e945: `informal/putnam.json`
(`putnam`; no license file covers it, and the upstream README says the informal statements are
available with permission from the MAA), `causal-nlp/corr2cause` (`causalbench`),
COCO 2017 val (images.cocodataset.org; `bundled_detection`): annotations CC BY 4.0 (COCO
Consortium); images under the Flickr Terms of Use (COCO does not own the image copyright), `lmms-lab/COCO-Caption` (`coco_caption`
images), `yerevann/coco-karpathy` (`coco_caption` split and references), `lmms-lab/textvqa`
(`textvqa`), `opendatalab/OmniDocBench` (`omnidocbench`),
`livebench/{coding,data_analysis,instruction_following,language,math,reasoning}`
(`livebench`, fetched by LiveBench itself), WMT19 news test sets (data.statmt.org/wmt19/translation-task/test.tgz, downloaded by the suite and
sha256-checked; no license declared; `i18n_translate`),
`Srimadh/Scicode-test-data-h5` (`scicode` reference outputs; a third-party mirror of SciCode's
test_data.h5, read at commit 72c247d3 and sha256-checked), `openai/gdpval` @11e7900c (`gdpval`
tasks and their reference files, read with `load_dataset` and `hf_hub_download`; no license tag on
the Hub), `bigcode/bigcodebench-hard` @298d2cc7 (`bigcodebench`, read only for the task ids of its
Hard subset; no license tag on the Hub).

## Permissive: attribution as the license requires
apache-2.0 / mit / cc-by-4.0 / cc-by-sa-4.0. These still require attribution (and share-alike for
cc-by-sa); comply with each license's notice terms.

| Dataset | License |
|---|---|
| `AI-MO/aimo-validation-aime`, `AI-MO/aimo-validation-amc` | apache-2.0 |
| `ArtificialAnalysis/AA-LCR`, `MMMU/MMMU_Pro`, `SciCode1/SciCode` | apache-2.0 |
| `google/IFEval`, `google/frames-benchmark`, `google/deepsearchqa` | apache-2.0 |
| `gorilla-llm/APIBench` | apache-2.0 |
| `lovesnowbest/T-Eval` (`t_eval`: `data/*_v2.json` and `data/*_v2_zh.json`) | apache-2.0 |
| `THUDM/ComplexFuncBench` @5dc7739343f8c87aee465931ba66db13a0fb5dd4 (`complexfuncbench`: `ComplexFuncBench.jsonl`, sha256-checked at load; the Hub now redirects the repository to `zai-org/ComplexFuncBench`) | apache-2.0 (Hub license tag) |
| GitHub `ShishirPatil/gorilla` @6ea5797: `berkeley-function-call-leaderboard/bfcl_eval/data` (`bfcl`, `bfcl_live`; downloaded from raw.githubusercontent.com, cached under `$GBENCH_CACHE_DIR/bfcl_data/<commit>/`; the multi-turn files (BFCL_v4_multi_turn_*.json, possible_answer/, multi_turn_func_doc/) and the agentic files are read from bfcl-eval 2026.3.23 in the local image gbench-bfcl:2026.3.23 (`bfcl`, `bfcl_v4_agentic`)) and `gorilla/eval/eval-data/questions/*_{0_shot,bm25,gpt_index,oracle}.jsonl` (`gorilla_apibench`) | apache-2.0 (repo LICENSE) |
| Hugging Face `sentence-transformers/all-MiniLM-L6-v2` @1110a243fdf4706b3f48f1d95db1a4f5529b4d41 (memory_vector encoder of `bfcl` / `bfcl_v4_agentic`, downloaded into the locally built gbench-bfcl image) | apache-2.0 (HF model card) |
| GitHub `IBM/NESTFUL` @fc2c4123: `data_v2/nestful_data.jsonl` (`nestful`) | apache-2.0 (repo LICENSE) |
| GitHub `fchollet/ARC-AGI` @399030444e0ab0cc8b4e199870fb20b863846f34: `data/evaluation` (`arc_agi`; fetched at run time from codeload.github.com, verified against git tree 4a19f832, cached, not redistributed) | apache-2.0 (the repository's LICENSE) |
| GitHub `trishullab/PutnamBench` @97b5e945: `lean4/src` (`putnam_formal`; downloaded as the commit tarball, cached under `$GBENCH_CACHE_DIR/putnambench/<commit>/`, git-tree-checked) | apache-2.0 (lean4/LICENSE) |
| `smolagents/browse_comp`, `zai-org/LongBench-v2` | apache-2.0 |
| `bigcode/bigcodebench` @b74c0d0b, split v0.1.4 (`bigcodebench` prompts) | apache-2.0 (Hub license tag) |
| `lmms-lab-encoder/DocVQA` (`docvqa`, `infographicvqa`) | apache-2.0 |
| `OS-Copilot/ScreenSpot-v2` @5efbb1f1 (`screenspot` and `screenspot_pointing`: `screenspot_{mobile,desktop,web}_v2.json` and `screenspotv2_image.zip`) | apache-2.0 (Hub license tag) |
| `stabletoolbench/baselines` @ea70398f955e7b58f8807187936818e794ce5140: `data_baselines.zip` (`toolbench`: the SoWR reference answers, `chatgpt_cot` by default; downloaded at run time, sha256-checked, converted and cached under `$GBENCH_CACHE_DIR/toolbench_baselines/`, not redistributed) | apache-2.0 |
| GitHub `allenai/WildBench` @d6b8dcaf: `leaderboard/data_dir/all_stat.json` Arena Elo values (`wildbench` WB-Elo starting points, carried in wildbench.py as WB_ELO_INIT) | apache-2.0 (repo LICENSE) |
| `TIGER-Lab/MMLU-Pro`, `TsinghuaC3I/MedXpertQA`, `basicv8vc/SimpleQA` | mit |
| `cais/mmlu`, `cais/wmdp`, `casey-martin/Seal-Tools`, `cruxeval-org/cruxeval` | mit |
| `nuprl/MultiPL-E`, `openai/MMMLU`, `openai/gsm8k`, `openai/healthbench`, `openai/mrcr` | mit |
| GitHub `meta-llama/PurpleLlama` @114d0238: `CybersecurityBenchmarks/datasets/instruct/instruct-v2.json` (`cyberseceval`; read from the locally built gbench-cyberseceval image, not redistributed) | mit (CybersecurityBenchmarks/LICENSE; the ICD in CodeShield/ is MIT too, CodeShield/LICENSE) |
| `liminghao1630/API-Bank` (`api_bank`: `test-data/level-3*.json`) | mit |
| GitHub `chenchen0103/ACEBench` @56dd66cf: `data_all/data_en` and `data_all/data_zh`, all 17 categories including the agent files (`acebench`) | mit (repo LICENSE) |
| GitHub `Purewhiter/mobilegym` @5fbd3f34be0e9780ae4b1e9d5089963ddbd05d38 (`mobilegym`: `bench_env/splits/test.txt`, 256 tasks, baked into the local `gbench-mobilegym:5fbd3f34` image; companion `mobilegym-data` bundle is CC BY-NC 4.0 above) | apache-2.0 (`LICENSE`) |
| GitHub `OpenMobile2/MobileGym-plusplus` @cdde4ad06ed67176b56964dd329b8cb8e0b0001c (`mobilegym_pp`: `bench215/official.json`, 215 tasks, baked into the local `gbench-mobilegym:5fbd3f34` image; companion `mobilegym-data` bundle is CC BY-NC 4.0 above) | apache-2.0 (`LICENSE`) |
| `edinburgh-dawg/mmlu-redux-2.0`, `ScaleAI/MCP-Atlas`, `open-r1/codeforces` | cc-by-4.0 (the mcp_atlas MongoDB image also restores the release's data_exports MongoDB dump from github.com/scaleapi/mcp-atlas, MIT, fetched at build time) |
| GitHub `google-deepmind/superhuman` @955558ad: `imobench/answerbench_v2.csv` (`imo_answer_bench`) | cc-by-4.0 (non-software materials, per the repo README) |
| `Mohammadta/BEAM`, `futurehouse/lab-bench`, `princeton-nlp/CharXiv` | cc-by-sa-4.0 |
| `livecodebench/code_generation_lite`, `livecodebench/execution-v2`, `livecodebench/test_generation` | cc |

## Serving workload dataset
- **ShareGPT** (`serving.py`, `--campaign chat-like`): `anon8231489123/ShareGPT_Vicuna_unfiltered` is
  downloaded just-in-time (to `$GBENCH_CACHE_DIR`, default `~/.cache/gbench/`, or `SHAREGPT_PATH`) for the ShareGPT serving
  workload. Note it is a scraped dataset with contested license/provenance; `--dataset random` needs
  no external data.
