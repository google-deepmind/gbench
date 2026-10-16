# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""gbench native built-in runner for wildbench (WildBench, allenai/WildBench).

Renamed from `lmsys_noncoding_hard`: the tasks are WildBench's (collected from WildChat, not
LMSYS), so the old name did not say what runs; the old name still runs this suite, with a
deprecation warning.

Runs all 1,024 tasks of the `v2` test split, the set the WildBench leaderboard scores (the
dataset card's `load_dataset("allenai/WildBench", "v2", split="test")`), and reports the
leaderboard's columns as the official pipeline computes them (allenai/WildBench src/eval.py
--batch_mode, src/openai_batch_eval/batch_results_format.py, leaderboard/data_dir/_create_tables.py
and _merge_results.py, leaderboard/wb_elo.py, commit d6b8dcaf377d; the leaderboard Space
allenai/WildBench @4b94d1d8):

WB-Score (individual scoring, `--mode score --max_words_to_eval -1`). The judge gets
eval_template.score.v2.md verbatim (history / current query split, the task checklist, the
analysis then strengths / weaknesses / score steps) in JSON mode and rates each reply 1-10;
every reply is judged, an empty one included (the batch pipeline submits it too).
  * WB_score.task_macro, the leaderboard's score column and the headline `accuracy`: every task
    is mapped to the five task groups of its primary and secondary tags (task_group_new), each
    group's mean raw score is adjusted to (mean - 5) * 2, and the groups are averaged with the
    upstream weights (Creative Tasks 0.5, Planning & Reasoning 1.25, Math & Data Analysis 1,
    Information/Advice seeking 0.75, Coding & Debugging 1.25) divided by their sum, times 10.
    Replies that are empty or end with "... (truncated)" stay out of the group means (and the
    Length column), as _create_tables.py skips them there. The per-group values are
    WB_score.<group>.
  * WB_score = (mean raw score - 5) * 20 over all tasks (upstream's `adjusted_score` * 10,
    range -80..+100), reported as `wb_score`.

WB-Reward (pairwise, `--mode pairwise --max_words_to_eval 1000`). Each reply is compared with
the published reply of each of the three reference models (gpt-4-turbo-2024-04-09,
claude-3-haiku-20240307, Llama-2-70b-chat-hf; allenai/WildBench-V2-Model-Outputs, pinned). The
judge gets eval_template.pairwise.v2.md verbatim, with the A/B order src/eval.py draws
(random.seed(42), one draw per task in v2 order), history, query and both replies cut to 1,000
words by shorten(), and "[This model response is empty.]" for an empty reply; it picks one of
A++ / A+ / A=B / B+ / B++ in JSON mode. The reward is +1 / +0.5 / 0 / -0.5 / -1 for the tested
model (much better ... much worse). As _create_tables.py does, a comparison in which either
reply was truncated or empty counts 0 (it stays in the denominator), a judge reply that is not
JSON with one of the five choices is dropped, and with a length margin K >= 0 a slight win is a
tie when the winner's reply is more than K characters longer. Reported per reference
(`<ref>_reward.K=`, per group and task-macro), as the mixture of the three
(`mixture_of_rewards.K=`, `task_macro_reward.K=`) and as RewardScore_Avg (the average of
WB_score.task_macro and task_macro_reward), for every K _merge_results.py computes; K=-1 (no
length penalty, the leaderboard default) is the top-level `wb_reward_*` fields.

WB-Elo (leaderboard/wb_elo.py; the leaderboard's "WB-Elo (Raw)" and "WB-Elo (LC)" columns). The
published replies of the 63 models with WildBench score results (WB_ELO_POOL) are score-graded
by the same judge and prompt, the tested model's WB-Score verdicts are added, wb_elo.py turns
every pair of scores on a task into a virtual vote (a win when the scores differ by at least 3,
a tie when by at most 2, no vote otherwise; empty or truncated replies cast none) and simulates
Elo from LMSYS Arena Elo starting points (WB_ELO_INIT) with the published settings (K=4, 128
rounds, dynamic with interval 100). WB_Elo uses no length rule; WB_Elo_LC counts a vote as a tie
when the replies differ by more than 1,500 characters. Upstream runs its rounds in forked
ProcessPoolExecutor workers, whose `random` state Python reseeds from OS entropy at fork, so its
rounds cannot be reproduced; gbench runs them in one process after seeding with 42
(compute_wb_elo's default seed), so a run is deterministic. The pool verdicts are cached under
$GBENCH_CACHE_DIR, keyed by the judge cascade and the judge config. The pool files are read one
model at a time, both when the gradable replies are listed before generation and when they are
graded (a pool reply's judge prompt is built only then), so the run holds at most one pool model's
replies while listing and two while grading, never the pool's prompts, and the votes are streamed
into the simulation.

The judge model is gbench's Gemini cascade (WildBench uses gpt-4o-2024-05-13 for scores and
gpt-4-turbo-2024-04-09 for rewards), so the result is flagged leaderboard_comparable=False and
names the models that graded it.

Sampling: WildBench's inference scripts decode at temperature 0 with top_p 1.0
(base.OfficialSampling, recorded). top_p 1.0 is sent in both modes; the temperature follows
gbench's rule (base.resolve_temperature): 1.0 with `--thinking`, 0.0 without. Override the temperature for a whole
run with `--temperature`, or for this suite alone with `GBENCH_WILDBENCH_TEMPERATURE` (the
pre-rename `GBENCH_LMSYS_NONCODING_HARD_TEMPERATURE` is still read as a deprecated alias), which
takes precedence over both; the result records `official_sampling`. Without `--thinking` every
request carries the scripts' MAX_TOKENS=4096, so `--max-output-tokens` is not applied and a reply
that stops there is judged as it arrived (`replies_at_official_max_tokens` counts them). The scripts
define no reasoning-model settings, so with `--thinking` that cap, written for non-reasoning
generation, is not applied: each request is sent the run's `--max-output-tokens` (the token-limit
policy, base.resolve_token_budget; the result records it). The judge calls follow src/eval.py:
temperature 0, JSON mode and its `--max_tokens` default of 1024 for the verdict
(`judge_sampling`). The Gemini judge counts its thinking against max_output_tokens (gpt-4o and
gpt-4-turbo do not think), so it is sent 1024 + JUDGE_THINKING_ALLOWANCE (8192): its verdict keeps
upstream's 1024 tokens. A judge reply that still stops there is read as it arrived (dropped when it
is not a readable verdict, as the batch pipeline drops it) and counted in
`judge_replies_at_max_output_tokens`.

Judge-side failures are harness errors, never the tested model's outcome: a judge outage (every
model of the cascade failed, JUDGE_OUTAGE) and a judge refusal (the Gemini API blocked the prompt,
JUDGE_REFUSED, recorded with its block reasons) are left out of every score, counted per mode
(WB-Score, WB-Reward, WB-Elo pool), make the run `completed_with_errors` and are named, with the
refused task ids, in `leaderboard_comparable_reason`. A request that got no reply from the served
endpoint is a harness error too: it is not judged in either mode and stays out of WB_score,
WB_score.task_macro, WB-Reward (it is not in any reference's total) and WB-Elo (base marks it
`scoring_excluded`, counts it in failed_requests and names it in status_reason).
"""

import hashlib
import json
import logging
import os
import random
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple
from .base import (run_eval_suite, gemini_required_skip, is_scored, judge_cascade, judge_config,
                   judge_generate_cascade_detailed, gbench_cache_dir, apply_shard,
                   OfficialSampling)
from .gemini_budget import GEMINI_THINKING_ALLOWANCE, gemini_output_limit
from .metrics import record_budget_cutoffs
from .sampling import limit_dataset
from .fc_common import base_suite_plan
from .limit_plan import base_attempts
from .simpleqa import withhold_if_nothing_scored

logger = logging.getLogger(__name__)

#: The registry name (renamed from `lmsys_noncoding_hard`; the stratified-sample seed keeps the
#: old string).
EVAL_NAME = "wildbench"
DOCS_URL = f"docs/evals/{EVAL_NAME}.md"

#: allenai/WildBench `v2` test split: 1,024 tasks (the WildBench leaderboard set).
_DATASET = "allenai/WildBench"
_CONFIG = "v2"
_REVISION = "26c49eb39d7d5ce2099b0bbafed5a88dcce954ec"
N_TASKS = 1024

#: allenai/WildBench-V2-Model-Outputs (src/eval.py HF_RESULTS_PATH): the published outputs of the
#: three WB-Reward reference models, pinned. Column prefix (_merge_results.py) -> (config, sha256
#: of <config>/train-00000-of-00001.parquet, the HF LFS oid at the pinned revision).
_REF_DATASET = "allenai/WildBench-V2-Model-Outputs"
_REF_REVISION = "d6755bc68220df853c0825a733430f73f5af2501"
REFERENCE_MODELS = {
    "gpt4t": ("gpt-4-turbo-2024-04-09",
              "0e94087c32882ff8687f9a5f0a13a762ba74a77311fac58d07eb750dfe9b44d2"),
    "haiku": ("claude-3-haiku-20240307",
              "6bfda71a23bcb86d76279f6b23040af82c8e5100c3c3e0cf22885f97d40caafa"),
    "llama": ("Llama-2-70b-chat-hf",
              "2234f3794abc4b86fa52e11e62a983b7ffad7459b80185ef1d3f2f5b93c8d133"),
}
#: evaluation/run_eval_v2_batch.sh runs the pairwise judge with `--max_words_to_eval 1000`.
PAIRWISE_MAX_WORDS = 1000
#: src/eval.py `--seed` default: random.seed(42), then one random.random() per task in v2 test
#: order; the tested model is Response A when the draw is below 0.5.
PAIRWISE_SEED = 42
#: What src/eval.py writes for an empty reply, and what shorten() appends to a cut one.
EMPTY_RESPONSE_PLACEHOLDER = "[This model response is empty.]"
TRUNCATED_SUFFIX = "... (truncated)"
#: The length margins K (characters) _merge_results.py computes. K=-1 disables the length penalty
#: and is the leaderboard's default (DEFAULT_K on the WildBench leaderboard space).
LENGTH_MARGINS = (-1, 100, 300, 500, 1000, 1500, 2000, 3000)
DEFAULT_LENGTH_MARGIN = -1
#: batch_results_format.py: judge choice -> (winning side, extent).
PAIRWISE_CHOICES = {"A=B": (None, 0), "A+": ("A", 1), "A++": ("A", 2),
                    "B+": ("B", 1), "B++": ("B", 2)}

#: WildBench's inference settings (scripts/_common_vllm.sh and the per-model scripts @
#: d6b8dcaf377d: TEMP=0; TOP_P=1.0; MAX_TOKENS=4096), also the reference outputs' `configs`.
UPSTREAM_MAX_TOKENS = 4096
OFFICIAL_SAMPLING = OfficialSampling(
    reference=("allenai/WildBench scripts/_common_vllm.sh @d6b8dcaf377d (TEMP=0; TOP_P=1.0; "
               "MAX_TOKENS=4096)"),
    temperature=0.0, params={"top_p": 1.0}, max_tokens=UPSTREAM_MAX_TOKENS)
#: src/eval.py @ d6b8dcaf377d: the judge request of both modes (batch body and openai_args).
JUDGE_REFERENCE = ("allenai/WildBench src/eval.py @d6b8dcaf377d: temperature 0 (--temperature), "
                   "response_format json_object, max_tokens 1024 (--max_tokens)")
#: src/eval.py `--max_tokens` default (the batch body's max_tokens; no run script overrides it):
#: the room upstream's judges (gpt-4o, gpt-4-turbo, which do not think) have for the verdict.
JUDGE_MAX_TOKENS = 1024
#: The Gemini judge thinks before it answers and its thinking counts against max_output_tokens,
#: so at 1024 it stopped mid-verdict (2026-10-08 smoke run: 55 judge replies stopped at 1024 across
#: WB-Reward and the WB-Elo pool; 54 pool verdicts dropped). It gets this allowance on top, so its
#: verdict keeps upstream's 1024 tokens: gemini_budget's GEMINI_THINKING_ALLOWANCE, the one value
#: every Gemini role written for a non-thinking model gets.
JUDGE_THINKING_ALLOWANCE = GEMINI_THINKING_ALLOWANCE
#: What the judge calls send as max_output_tokens (gemini_budget.gemini_output_limit).
JUDGE_OUTPUT_LIMIT = gemini_output_limit(JUDGE_MAX_TOKENS)
#: The judge grade of a call the Gemini API refused (blocked the prompt) for every model.
JUDGE_REFUSED = "JUDGE_REFUSED"


def judge_sampling_record() -> Dict[str, Any]:
    """The `judge_sampling` block of a WildBench result (score, pairwise and pool calls)."""
    return {"reference": JUDGE_REFERENCE, "temperature": 0.0,
            "response_mime_type": "application/json",
            "max_output_tokens": JUDGE_OUTPUT_LIMIT,
            "max_output_tokens_note": (
                f"src/eval.py's max_tokens={JUDGE_MAX_TOKENS} for the verdict plus "
                f"{JUDGE_THINKING_ALLOWANCE} for the Gemini judge's thinking, which counts "
                "against max_output_tokens (gpt-4o and gpt-4-turbo do not think); a reply that "
                "still stops there is read as it arrived and counted in "
                "judge_replies_at_max_output_tokens")}


def judge_refusal(info: Dict[str, Any]) -> Optional[List[str]]:
    """The block reasons of a judge call the Gemini API refused (the cascade's `refused`), or
    None when the call was not refused. A refusal is a property of the prompt, so it is not an
    outage a rerun would recover from."""
    if not (info or {}).get("refused"):
        return None
    return [str(r) for r in (info.get("block_reasons") or [])] or ["blocked"]

#: allenai/WildBench leaderboard/data_dir/_create_tables.py @ d6b8dcaf377d, verbatim:
#: WildBench tag -> task group.
TASK_GROUP = {
    "Information seeking": "Information/Advice seeking",
    "Creative Writing": "Creative Tasks",
    "Coding & Debugging": "Coding & Debugging",
    "Reasoning": "Planning & Reasoning",
    "Editing": "Creative Tasks",
    "Math": "Math & Data Analysis",
    "Planning": "Planning & Reasoning",
    "Brainstorming": "Creative Tasks",
    "Role playing": "Creative Tasks",
    "Advice seeking": "Information/Advice seeking",
    "Data Analysis": "Math & Data Analysis",
    "Others": "Creative Tasks"
}
#: The same file's weights_by_task (score and pairwise modes use the same weights).
TASK_MACRO_WEIGHTS = {
    "Creative Tasks": 0.5,
    "Planning & Reasoning": 1.25,
    "Math & Data Analysis": 1,
    "Information/Advice seeking": 0.75,
    "Coding & Debugging": 1.25
}
#: _merge_results.py all_task_types (merged version), the per-group reward columns.
ALL_TASK_GROUPS = ['Creative Tasks', 'Planning & Reasoning', 'Math & Data Analysis',
                   'Information/Advice seeking', 'Coding & Debugging']

#: WB-Elo pool: every model with published WB-Score results at d6b8dcaf377d
#: (eval_results/v2.0625/score.v2/eval=gpt-4o-2024-05-13/<model>.json, 63 files, the scores
#: leaderboard/wb_elo.py reads). Each is re-graded by the gbench judge from its published replies in
#: allenai/WildBench-V2-Model-Outputs at _REF_REVISION (config = the file name). Value: sha256 of
#: <config>/train-00000-of-00001.parquet (the HF LFS oid at the pinned revision).
WB_ELO_POOL = {
    "Athene-70B": "a67b69c4b175bfaaea14f5ac81150cb7fcec8d382c002e6fe8c4b39b38b67525",
    "Hermes-2-Theta-Llama-3-8B": "ab7ce2178fb7cc2979b1bbbfffb0b621bd85ca88d218ab9b5b815c5463a59656",
    "Llama-2-70b-chat-hf": "2234f3794abc4b86fa52e11e62a983b7ffad7459b80185ef1d3f2f5b93c8d133",
    "Llama-2-7b-chat-hf": "a7b73a38d492817b434ddf0253711b63ce4a867661667b725094796964d52156",
    "Llama-3-8B-Magpie-Align-v0.1": "05affd09d7e7650ba3f07caee9a5c853e580f89effa1f4bc7199ad113e144821",
    "Llama-3-Instruct-8B-SimPO": "804f4f82323304f557e4cbc8abd075ced26c6dbc54cc6a1eead112a23ae6970f",
    "Llama-3-Instruct-8B-SimPO-ExPO": "5ced4163680d53194fa286f85ee1991d5607bc7ffc188efa3616957d8a532019",
    "Llama-3-Instruct-8B-SimPO-v0.2": "4ae93c02fe716428d02d576bd921b278bebf7bbfa474907f09229b8f4f21ac23",
    "Meta-Llama-3-70B-Instruct": "94c6b161c7403e056f4f69a0ff5c4bf003f83bf36c467029ed71c3c7867ce8f3",
    "Meta-Llama-3-8B-Instruct": "74ad3de2d49a46b72685e1e137e910cebee25d4c4694d9640c21bca49b648d3c",
    "Mistral-7B-Instruct-v0.2": "1216bcf7471ab06b77371b2c4b26d4e479bc83de9df28ee3cde858c30ef6b22d",
    "Mistral-Large-2": "7aa5a545222b38794f11c128b7d7ccd059cbc290db8e0e7854af0dac32e8beda",
    "Mistral-Nemo-Instruct-2407": "6da3ae135183e480a9f95bdad12aabe87c0dd3152c17fe03b40ddeaf9742b071",
    "Mixtral-8x7B-Instruct-v0.1": "ee3648357808170d8a0b43200db89ba48b613cff2409b689dcd526d7f2ad3667",
    "Nous-Hermes-2-Mixtral-8x7B-DPO": "a4fd0cc90fc583385269bb80186da659b5b2c11288373bd49f28e4bd20599c9f",
    "Phi-3-medium-128k-instruct": "d75cb9e946f670a7e4937051428f9eaeb3be5a56fbcc6162a2b7f334ad4c7db4",
    "Phi-3-mini-128k-instruct": "197739257b87c5ae49bebfb3960289d3915ded935542946d94b266b9665d14eb",
    "Qwen1.5-72B-Chat-greedy": "d298493c80b18482d88781ddb1e0703b29ec5a614d529fcafe2c7c99882a7bf3",
    "Qwen1.5-7B-Chat@together": "316ac2c0064c6091dd80821601771ee569a881f66a9eb02da8012a75997fdea5",
    "Qwen2-72B-Instruct": "164f95c3f3350e2a65746fea6e933b6a7cc9c7c91698bcbff2a72ee9a65b5cc2",
    "SELM-Llama-3-8B-Instruct-iter-3": "60d78ffb5295d7980bb9b439c133a6af9e9ae1e433e80a73bd587f75527d0298",
    "SELM-Zephyr-7B-iter-3": "7d5d5d52bdbc4a3233f4b03d1b72679ffedd7f41f49592bacae5e10990948c41",
    "Starling-LM-7B-beta": "5b16ecf3c7e7a8e5bad8f1dcf7910596d8e80b88806de65f7c55d39caf0bb1b6",
    "Starling-LM-7B-beta-ExPO": "6bb903432a6fcd13a38cf1c39e2116b10ce7fe83464347b4e5815c978634a052",
    "Yi-1.5-34B-Chat": "e287a63f4084bddc2be97dd6c2360d0bb3806b3055da7d541ae060a1888942aa",
    "Yi-1.5-6B-Chat": "60da698b4c8f9a4643b9869d1bb668740dff3c6223a79738d3abc7e180aa8b47",
    "Yi-1.5-9B-Chat": "b749183c9e04fcc17aaaba3e9c60e8da4de1bc184d599b801d900fcf61e65221",
    "claude-3-5-sonnet-20240620": "00e48a379d75de0c9019cb730b701a977d3ce0aa2ec28bd04879feea0fb95813",
    "claude-3-haiku-20240307": "6bfda71a23bcb86d76279f6b23040af82c8e5100c3c3e0cf22885f97d40caafa",
    "claude-3-opus-20240229": "9b20cf6cb68e4ba5fa47d7f77128f7d57759bd356a4349b422fc7a1f6fac79e4",
    "claude-3-sonnet-20240229": "c143ee99cc29e516c1657d14e278d9c560bddd9b2e43eebc104f3ef57962f503",
    "command-r": "aae35e740b91a4a6da785aa08ffbc9689881b31c9dcb029ec8910b3844b9b1cc",
    "command-r-plus": "b8717e6a370c237923344c682bfa90fdad255cddc13db5e0016704c509c74eeb",
    "dbrx-instruct@together": "bbb2445029057ff23d51c58a086b311d0ca8ad16688eec2a409e49c41b67a4d0",
    "deepseek-coder-v2": "1a5641e4d1576807eefb94a3443488ae1a93f4200e54c26d638512ec621543ce",
    "deepseek-v2-chat-0628": "d7ff24723b97bc484eacb7b3eed48be28837138304de3cbbb47cebe0dc0d6be1",
    "deepseek-v2-coder-0628": "ca8a044a114c12081cd4de10d6dca8660fc3ec1d448b362e099072d55294e917",
    "deepseekv2-chat": "ee0c564e0ea3558860cb5b310e4a97806266fb37f558a860d1a5857e176a6b08",
    "gemini-1.5-flash": "3bfe9cccb17e2c540e4b1509cf6b818b44b3e028c6e284144f6298f86942f247",
    "gemini-1.5-pro": "4da63f7d877b39736d93f4e9b2dda08075ed14ab1bee597e6c2f427ee464dcd2",
    "gemma-2-27b-it@together": "9d943338f657669733421507c19bbe3851deec95a6ef28591be1a0f9c9845151",
    "gemma-2-2b-it": "4d5f803d5245ccc43710fde5558a86acf6d3f43f5d7a277370df58eff27f087f",
    "gemma-2-9b-it": "02da844bbbdb8014ca8462f50da1144e0624d35a7197406dfec2cb1ba9b729fc",
    "gemma-2-9b-it-DPO": "ea4e5fd1ac9298afccea7c65762f6d07e1200f2e5f013d62d0356f61ff991538",
    "gemma-2-9b-it-SimPO": "bf62878ca3cf11798d5746067b0ec01064030b204e57ee761ecd714ed14ded28",
    "gemma-2b-it": "83e20342d25504ba55ebcb76c8cbb5710e331219f65d7a1f8c7714b346fdcdea",
    "gemma-7b-it": "993fba5367bf9ac95ef14bffd7cebddfd58afdc453db05c92d4cf9b749f9e277",
    "glm-4-9b-chat": "995e5eae4444b51a1afe1dc91b3a69d67be7ccebf4b99145ab938419ca18acc2",
    "gpt-3.5-turbo-0125": "be2960f5a9b13796080b1770d99bdbd386a50dc7011d5e76dedccc907c333f96",
    "gpt-4-0125-preview": "d787ff5310ead97e411ed0fc00e60da9a2ba978d6cc3b7d7684daf512fb80dc5",
    "gpt-4-turbo-2024-04-09": "0e94087c32882ff8687f9a5f0a13a762ba74a77311fac58d07eb750dfe9b44d2",
    "gpt-4o-2024-05-13": "3f624567bfa7a0f50d1754cd3fda413d7fbc78cb00e319e5ac87097c3b2164e8",
    "gpt-4o-mini-2024-07-18": "e446d812ac8c74a34202239f855a96292ac77a54065f29b9ffe82103b94f4e78",
    "mistral-large-2402": "dc59f8d29df5b9ce647fd5ca346f0f5d2aca2827f15ce536b7de2be77e818d82",
    "nemotron-4-340b-instruct": "45ba2dbf81092cd87f80765695c7fc98b99626410c43f8fb143afbf1b0f0d194",
    "neo_7b_instruct_v0.1": "34bb998315e2894b34b91334b133876fa2835588442c8b25f5e052308b4bbacd",
    "neo_7b_instruct_v0.1-ExPO": "971c1682a9a0c3cca5a7f8cc4cb43cc693668aefa2707d0ee66de40d4b85ddbc",
    "reka-core-20240501": "f59a73e717ddb34d3238368778f292a40f353ec427378f1da52a3a02ce23480a",
    "reka-edge": "9960e3906d4238b990f4c0f60d2202950ae51a0fe9b0470e90f4c5ea83af2409",
    "reka-flash-20240226": "83bddb70378ccc29fc36f6a14bb21b221c3304f26cb68f051ce5cd6ed88b9639",
    "tulu-2-dpo-70b": "3ad1ea7c85eaa8ec429f8fda364bcbc745dd043591f531abc6f69046d21bcd57",
    "yi-large": "5ee68a9315f7309cb315359afd761eb56694d894b758c5fb8f9af35298ebd139",
    "yi-large-preview": "673e2a9e02b8e015afbc4a2eb6933b520b4e986ef3a4c7a69c42231ba569da56",
}
#: The Elo each model starts from: leaderboard/data_dir/all_stat.json "Arena Elo (hard-en) - latest"
#: as wb_elo.load_init_elo reads it (the "@together" suffix dropped). The file is the same at
#: d6b8dcaf377d and on the leaderboard Space at 4b94d1d8 (sha256 da75f4b2...), and these are the
#: init_elo values the Space's published wb_elo_results.LM=-1.json / LM=1500.json record. Four
#: of them (Llama-2-13b-chat-hf, Mistral-7B-Instruct-v0.1, Yi-34B-Chat, zephyr-7b-beta) have no
#: WildBench results, so they cast no vote and keep their Arena Elo, as upstream.
WB_ELO_INIT = {
    "Llama-2-13b-chat-hf": 1049.0, "Llama-2-70b-chat-hf": 1071.0, "Llama-2-7b-chat-hf": 1012.0,
    "Meta-Llama-3-70B-Instruct": 1212.0, "Meta-Llama-3-8B-Instruct": 1144.0,
    "Mistral-7B-Instruct-v0.1": 1004.0, "Mistral-7B-Instruct-v0.2": 1072.0,
    "Mixtral-8x7B-Instruct-v0.1": 1114.0, "Nous-Hermes-2-Mixtral-8x7B-DPO": 1046.0,
    "Phi-3-mini-128k-instruct": 1037.0, "Qwen1.5-72B-Chat-greedy": 1142.0,
    "Qwen1.5-7B-Chat": 1057.0, "Qwen2-72B-Instruct": 1182.0, "Starling-LM-7B-beta": 1113.0,
    "Yi-1.5-34B-Chat": 1160.0, "Yi-34B-Chat": 1098.0, "claude-3-5-sonnet-20240620": 1271.0,
    "claude-3-haiku-20240307": 1170.0, "claude-3-opus-20240229": 1230.0,
    "claude-3-sonnet-20240229": 1188.0, "command-r": 1106.0, "command-r-plus": 1155.0,
    "dbrx-instruct": 1111.0, "deepseek-v2-chat-0628": 1233.0, "deepseek-v2-coder-0628": 1204.0,
    "gemini-1.5-flash": 1212.0, "gemini-1.5-pro": 1245.0, "gemma-2-27b-it": 1185.0,
    "gemma-2-9b-it": 1155.0, "gemma-2b-it": 977.0, "gemma-7b-it": 1046.0,
    "gpt-3.5-turbo-0125": 1107.0, "gpt-4-0125-preview": 1237.0,
    "gpt-4-turbo-2024-04-09": 1247.0, "gpt-4o-2024-05-13": 1280.0, "mistral-large-2402": 1158.0,
    "nemotron-4-340b-instruct": 1185.0, "reka-core-20240501": 1175.0,
    "reka-flash-20240226": 1127.0, "tulu-2-dpo-70b": 1101.0, "yi-large": 1198.0,
    "yi-large-preview": 1229.0, "zephyr-7b-beta": 1028.0,
}
#: The settings of the two published WB-Elo columns, as the leaderboard Space (allenai/WildBench
#: @4b94d1d8) records them in the "config" of data_dir/wb_elo_results.LM=-1.json and
#: LM=1500.json: K=4, num_rounds=128, margin=3, tie_margin=2, dynamic=True, interval=100,
#: use_regressed_as_init=False. seed: gbench seeds its in-process rounds with compute_wb_elo's
#: default 42; upstream's own rounds run in forked workers reseeded from OS entropy, so no seed
#: reproduces them.
WB_ELO_CONFIG = {"K": 4, "num_rounds": 128, "margin": 3, "tie_margin": 2, "dynamic": True,
                 "interval": 100, "seed": 42}
#: Column -> length_margin: WB_Elo ("WB-Elo (Raw)", no length rule) and WB_Elo_LC ("WB-Elo (LC)":
#: a vote is a tie when the two replies differ by more than 1,500 characters).
WB_ELO_LENGTH_MARGINS = {"WB_Elo": -1, "WB_Elo_LC": 1500}
#: The tested model's name in the vote pool; it cannot collide with a pool or Arena Elo name,
#: so the tested model never inherits another model's starting Elo.
WB_ELO_TARGET = "gbench:model-under-test"


def task_groups(primary_tag: Any, secondary_tags: Any) -> List[str]:
    """_create_tables.py task_mapping: the groups of [primary_tag] + secondary_tags, de-duplicated."""
    tags = [primary_tag] + list(secondary_tags or [])
    return sorted({TASK_GROUP[tag] for tag in tags})


# Copied verbatim from allenai/WildBench evaluation/eval_template.score.v2.md (commit
# d6b8dcaf377d): the whole file as src/eval.py reads it, trailing spaces included.
EVAL_TEMPLATE = """\
# Instruction 

You are an expert evaluator. Your task is to evaluate the quality of the responses generated by AI models. 
We will provide you with the user query and an AI-generated responses.
You should first read the user query and the conversation history carefully for analyzing the task, and then evaluate the quality of the responses based on and rules provided below.

# Conversation between User and AI

## History
<|begin_of_history|>

{$history}

<|end_of_history|> 

## Current User Query
<|begin_of_query|>

{$user_query}

<|end_of_query|>

## AI Response
<|begin_of_response|>

{$model_output}

<|end_of_response|>
 

# Evaluation   

## Checklist 

<|begin_of_checklist|>

{$checklist}

<|end_of_checklist|>

Please use this checklist to guide your evaluation, but do not limit your assessment to the checklist.

## Rules 

You should compare the above response based on your analysis of the user queries and the conversation history.
You should first write down your analysis and the checklist that you used for the evaluation, and then provide your assessment according to the checklist.
The scores are in the range of 1~10, where 1 means the response is very poor and 10 means the response is perfect.
Here are more detailed criteria for the scores:

- Score 1~2: The response is very poor and does not make sense at all.
- Score 3~4: The response is poor and does help user solve the problem in a meaningful way.
- Score 5~6: The response is fair but has some issues (e.g., factual errors, hallucinations, missing key information).
- Score 7~8: The response is good enough but could be improved in some ways.
- Score 9~10: The response is perfect and provides helpful information that can help user solve the problem.

## Output Format 
First, please output your analysis for the model response, and then summarize your assessment to two aspects: "strengths" and "weaknesses"; Finally, please write down your rating for the assessment.

Please provide your evaluation results in the following json format by filling in the placeholders in []:
```
{
    "strengths": "[analysis for the strengths of the response]",
    "weaknesses": "[analysis for the weaknesses of the response]",
    "score": "[1~10]"
}
```
"""

# Copied verbatim from allenai/WildBench evaluation/eval_template.pairwise.v2.md (commit
# d6b8dcaf377d): the whole file as src/eval.py reads it, trailing spaces included.
PAIRWISE_TEMPLATE = """\
# Instruction 

You are an expert evaluator. Your task is to evaluate the quality of the responses generated by two AI models. 
We will provide you with the user query and a pair of AI-generated responses (Response A and Response B). 
You should first read the user query and the conversation history carefully for analyzing the task, and then evaluate the quality of the responses based on and rules provided below.

# Conversation between User and AI

## History
<|begin_of_history|>

{$history}

<|end_of_history|> 

## Current User Query
<|begin_of_query|>

{$user_query}

<|end_of_query|>

## Response A
<|begin_of_response_A|>

{$candidate_A}

<|end_of_response_A|>

## Response B
<|begin_of_response_B|>

{$candidate_B}

<|end_of_response_B|>

# Evaluation   

## Checklist 

<|begin_of_checklist|>

{$checklist}

<|end_of_checklist|>

Please use this checklist to guide your evaluation, but do not limit your assessment to the checklist.

## Rules 

You should compare the above two responses based on your analysis of the user queries and the conversation history.
You should first write down your analysis and the checklist that you used for the evaluation, and then provide your assessment according to the checklist.
There are five choices to give your final assessment: ["A++", "A+", "A=B", "B+", "B++"], which correspond to the following meanings:

- `A++`: Response A is much better than Response B.
- `A+`: Response A is only slightly better than Response B.
- `A=B`: Response A and B are of the same quality. Please use this choice sparingly.
- `B+`: Response B is only slightly better than Response A.
- `B++`: Response B is much better than Response A.


## Output Format 
First, please output your analysis for each model response, and then summarize your assessment to three aspects: "reason A=B", "reason A>B", and "reason B>A", and finally make your choice for the final assessment.

Please provide your evaluation results in the following json format by filling in the placeholders in []:
```
{
    "analysis of A": "[analysis of Response A]",
    "analysis of B": "[analysis of Response B]",
    "reason of A=B": "[where Response A and B perform equally well]",
    "reason of A>B": "[where Response A is better than Response B]",
    "reason of B>A": "[where Response B is better than Response A]",
    "choice": "[A++ or A+ or A=B or B+ or B++]",
}
```
"""


def pairwise_assignment(session_ids: List[str]) -> Dict[str, bool]:
    """{session_id: tested model is Response A}, drawn as src/eval.py does for the full set.

    main() seeds `random` with 42 and placeholder_generation() draws once per task, in the
    order of the v2 test split, before any other draw; the same sequence serves all three
    references (each is a separate run with the same seed).
    """
    rng = random.Random(PAIRWISE_SEED)
    return {sid: rng.random() < 0.5 for sid in session_ids}


def _load_wildbench_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]]:
    """Load the 1,024 WildBench v2 tasks ('allenai/WildBench', config 'v2', test, pinned)."""
    from datasets import load_dataset

    ds = load_dataset(_DATASET, _CONFIG, split="test", revision=_REVISION)
    raw_samples = list(ds)
    if len(raw_samples) != N_TASKS:
        raise RuntimeError(f"wildbench: {_DATASET}@{_REVISION} ({_CONFIG}, test) has "
                           f"{len(raw_samples)} tasks, expected {N_TASKS}")
    # The A/B order is drawn over the whole split, so a limited run keeps upstream's order.
    target_is_a = pairwise_assignment([item.get("session_id") for item in raw_samples])
    # The stratified --eval-limit seed keeps the suite's pre-rename name.
    raw_samples = limit_dataset(raw_samples, limit, 'primary_tag', seed="lmsys_noncoding_hard")
    logger.info(f"Loaded {len(raw_samples)} WildBench v2 tasks from HF Hub ('{_DATASET}').")

    samples = []
    for item in raw_samples:
        conv = item.get("conversation_input") or []
        if not conv:
            raise RuntimeError(f"wildbench: task {item.get('session_id')!r} has no "
                               "conversation_input; refusing to drop it")
        messages = [{"role": msg.get("role", "user"), "content": msg.get("content", "")} for msg in conv]
        checklist = list(item.get("checklist") or [])
        tag = item.get("primary_tag", "general")
        samples.append((messages, checklist, {
            "category": tag,
            "session_id": item.get("session_id"),
            "wb_task_groups": task_groups(tag, item.get("secondary_tags")),
            "wb_pairwise_target_is_a": target_is_a[item.get("session_id")],
        }))

    return samples


def _fetch_model_outputs(config: str, sha256: str, purpose: str) -> Dict[str, str]:
    """{session_id: reply} of one model of allenai/WildBench-V2-Model-Outputs at _REF_REVISION.

    <config>/train-00000-of-00001.parquet must match `sha256`; the reply is `output[0]` (src/eval.py
    placeholder_generation). A file that cannot be fetched is infra_required (`purpose` says what
    needs it); a checksum or size mismatch raises.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from .swebench_common import infra_required

    filename = f"{config}/train-00000-of-00001.parquet"
    try:
        path = hf_hub_download(_REF_DATASET, filename, repo_type="dataset",
                               revision=_REF_REVISION)
    except Exception as e:                                              # noqa: BLE001
        raise infra_required(
            EVAL_NAME, f"the published outputs {_REF_DATASET}@{_REF_REVISION} ({filename}) could "
            f"not be downloaded ({type(e).__name__}: {e}); {purpose}", DOCS_URL)
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != sha256:
        raise RuntimeError(f"wildbench: {_REF_DATASET}@{_REF_REVISION} {filename} has sha256 "
                           f"{digest.hexdigest()}, expected {sha256}")
    rows = pq.read_table(path, columns=["session_id", "output"]).to_pylist()
    outputs: Dict[str, str] = {}
    for row in rows:
        out = row["output"]
        outputs[row["session_id"]] = str(out[0] if isinstance(out, list) else out)
    if len(rows) != N_TASKS or len(outputs) != N_TASKS:
        raise RuntimeError(f"wildbench: {_REF_DATASET}@{_REF_REVISION} {filename} has "
                           f"{len(rows)} rows / {len(outputs)} tasks, expected {N_TASKS}")
    return outputs


def _load_reference_outputs() -> Dict[str, Dict[str, str]]:
    """{column prefix: {session_id: reply}} for the three WB-Reward reference models."""
    return {prefix: _fetch_model_outputs(config, sha256,
                                         "WB-Reward compares every reply with them")
            for prefix, (config, sha256) in REFERENCE_MODELS.items()}


def _load_wb_elo_pool_outputs() -> Iterator[Tuple[str, Dict[str, str]]]:
    """(pool config, {session_id: reply}) for the 63 WB-Elo pool models (pinned, sha256-checked),
    one model at a time: each model's file is read when the caller asks for it, so a caller that
    drops a model's replies before taking the next holds one model's replies at a time."""
    for config, sha256 in WB_ELO_POOL.items():
        yield config, _fetch_model_outputs(config, sha256,
                                           "WB-Elo grades every pool model's replies")


def _output_chars(trace: Dict[str, Any]) -> int:
    """Length of the reply as _create_tables.py measures it (the stripped text in the prompt)."""
    if "wb_output_chars" in trace:
        return int(trace["wb_output_chars"])
    return len((trace.get("response_text") or "").strip())


def _output_truncated(trace: Dict[str, Any]) -> bool:
    """The reply as it appears in the score prompt ends with TRUNCATED_SUFFIX."""
    if "wb_output_truncated" in trace:
        return bool(trace["wb_output_truncated"])
    return (trace.get("response_text") or "").strip().endswith(TRUNCATED_SUFFIX)


def _in_group_tally(trace: Dict[str, Any]) -> bool:
    """_create_tables.py score mode: after `scores.append`, a reply that ends with
    "... (truncated)" or is empty is skipped (`continue`) before the length and group tallies."""
    return (trace.get("wb_score_raw") is not None and not trace.get("scoring_excluded")
            and not _output_truncated(trace) and _output_chars(trace) > 0)


def wb_task_macro(traces: List[Dict[str, Any]]) -> Tuple[Optional[float], Dict[str, float]]:
    """WB_score.task_macro and WB_score.<group>, as _create_tables.py (score) + _merge_results.py.

    Each group's mean raw score is adjusted to (mean - 5) * 2; the groups present are weighted
    and divided by the sum of ALL five weights, then * 10. A trace counts toward its groups
    when it has a raw score and a reply that is neither empty nor ends with "... (truncated)"
    (upstream `continue`s on both before the group tally, after counting the score in
    WB_score); outages and grading failures carry no score.
    """
    by_group: Dict[str, List[float]] = {}
    for t in traces:
        if not _in_group_tally(t):
            continue
        for group in (t.get("extra_payload") or {}).get("wb_task_groups") or []:
            by_group.setdefault(group, []).append(float(t["wb_score_raw"]))
    if not by_group:
        return None, {}
    adjusted = {g: (sum(v) / len(v) - 5) * 2 for g, v in by_group.items()}
    macro = (sum(adjusted[g] * TASK_MACRO_WEIGHTS[g] for g in adjusted)
             / sum(TASK_MACRO_WEIGHTS.values()))
    return round(macro * 10, 4), {g: round(a * 10, 4) for g, a in sorted(adjusted.items())}


def task_groups_absent(per_group: Optional[Dict[str, Any]]) -> List[str]:
    """The task groups of TASK_MACRO_WEIGHTS with no scored task (not in wb_task_macro's
    per-group scores), sorted: upstream's task macro counts them as 0 over all five weights, so
    a subset without them is capped below the full scale."""
    return sorted(set(TASK_MACRO_WEIGHTS) - set(per_group or {}))


def wb_score_by_category(traces: List[Dict[str, Any]]) -> Dict[str, float]:
    """WB_score per category (the task's primary tag), with run_wildbench's WB_score rules: every
    scored reply counts, and a request that got no reply from the served endpoint (a harness
    error), a judge outage, refusal or grading failure is left out; the mean adjusted score
    times 10, the headline's scale."""
    by_cat: Dict[str, List[float]] = {}
    for t in traces:
        if not is_scored(t) or t.get("response_text") is None:
            continue
        score = t.get("wb_score")
        if isinstance(score, (int, float)):
            by_cat.setdefault(str(t.get("category")), []).append(float(score))
    return {cat: round(sum(v) / len(v) * 10.0, 2) for cat, v in by_cat.items() if v}


def wb_length(traces: List[Dict[str, Any]]) -> Optional[float]:
    """The leaderboard's Length column: mean characters of the scored replies that are neither
    empty nor end with "... (truncated)" (_create_tables.py `lengths`)."""
    lengths = [_output_chars(t) for t in traces if _in_group_tally(t)]
    return round(sum(lengths) / len(lengths), 2) if lengths else None


# --------------------------------------------------------------------------- #
# WB-Elo (leaderboard/wb_elo.py @ d6b8dcaf377d)
# --------------------------------------------------------------------------- #
def _int_score(score: Any) -> int:
    """wb_elo.load_scores `int(score)`; a score int() cannot read directly (e.g. "7.5") is
    int(float())."""
    try:
        return int(score)
    except (TypeError, ValueError):
        return int(float(score))


def wb_elo_scores(models: List[Tuple[str, List[Tuple[str, Any, str]]]]
                  ) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, str]]]:
    """wb_elo.load_scores: (all_scores_by_id_model, all_outputs_by_id_model).

    `models` lists (model name, [(session_id, score, reply as in the score prompt), ...]) in load
    order; a session enters the dicts when first seen, and each score is `int()`-ed and each reply
    stripped, as upstream. A score that int() cannot read directly (e.g. "7.5") is int(float()).
    """
    scores: Dict[str, Dict[str, int]] = {}
    outputs: Dict[str, Dict[str, str]] = {}
    for model_name, items in models:
        for session_id, score, model_output in items:
            if session_id not in scores:
                scores[session_id] = {}
                outputs[session_id] = {}
            scores[session_id][model_name] = _int_score(score)
            outputs[session_id][model_name] = str(model_output).strip()
    return scores, outputs


def vote_length(model_output: str) -> Optional[int]:
    """What get_all_votes reads from a reply: its length (stripped, as load_scores stores it), or
    None when the reply cannot vote (empty, or ending with "... (truncated)")."""
    text = str(model_output).strip()
    if len(text) == 0 or text.endswith(TRUNCATED_SUFFIX):
        return None
    return len(text)


def wb_elo_score_lengths(models: List[Tuple[str, List[Tuple[str, Any, Optional[int]]]]]
                         ) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, Optional[int]]]]:
    """wb_elo_scores with each reply already reduced to its vote_length (same order, same int())."""
    scores: Dict[str, Dict[str, int]] = {}
    lengths: Dict[str, Dict[str, Optional[int]]] = {}
    for model_name, items in models:
        for session_id, score, length in items:
            if session_id not in scores:
                scores[session_id] = {}
                lengths[session_id] = {}
            scores[session_id][model_name] = _int_score(score)
            lengths[session_id][model_name] = length
    return scores, lengths


def wb_elo_votes(scores_by_session: Dict[str, Dict[str, int]],
                 outputs_by_session: Dict[str, Dict[str, str]],
                 margin: int, tie_margin: int) -> List[Dict[str, Any]]:
    """wb_elo.get_all_votes: virtual pairwise votes from the scores, same logic and order.

    For every session and every unordered pair of models (in load order): skipped when either
    reply is empty or ends with "... (truncated)"; the higher score wins when it leads by at least
    `margin`; a tie when the scores differ by at most `tie_margin`; otherwise no vote. Upstream
    walks ordered pairs and keeps the first of each unordered pair; every test is symmetric, so
    that is the pair (i, j), i < j, in load order.
    """
    lengths = {session_id: {m: vote_length(o) for m, o in per_model.items()}
               for session_id, per_model in outputs_by_session.items()}
    return wb_elo_votes_from_lengths(scores_by_session, lengths, margin, tie_margin)


def wb_elo_votes_from_lengths(scores_by_session: Dict[str, Dict[str, int]],
                              lengths_by_session: Dict[str, Dict[str, Optional[int]]],
                              margin: int, tie_margin: int) -> List[Dict[str, Any]]:
    """wb_elo_votes on vote_length values: a reply whose length is None casts no vote. The only
    things get_all_votes reads from a reply are whether it is empty or truncated and its length,
    so the pool's votes need no reply text."""
    return list(iter_wb_elo_votes(scores_by_session, lengths_by_session, margin, tie_margin))


def iter_wb_elo_votes(scores_by_session: Dict[str, Dict[str, int]],
                      lengths_by_session: Dict[str, Dict[str, Optional[int]]],
                      margin: int, tie_margin: int) -> Iterator[Dict[str, Any]]:
    """wb_elo_votes_from_lengths one vote at a time, in the same order (wb_elo_summary streams
    the ~2 million votes of a full run into wb_elo_ratings instead of keeping them all)."""
    for session_id, scores_by_model in scores_by_session.items():
        names = list(scores_by_model)
        for i, model_1 in enumerate(names):
            score_1 = scores_by_model[model_1]
            length_1 = lengths_by_session[session_id][model_1]
            for model_2 in names[i + 1:]:
                score_2 = scores_by_model[model_2]
                length_2 = lengths_by_session[session_id][model_2]
                if length_1 is None or length_2 is None:
                    continue
                if score_1 > score_2 and score_1 - score_2 >= margin:
                    winner = model_1
                elif score_2 > score_1 and score_2 - score_1 >= margin:
                    winner = model_2
                elif abs(score_1 - score_2) <= tie_margin:
                    winner = "tie"
                else:
                    continue
                yield {"session_id": session_id, "model_1": model_1,
                       "model_2": model_2, "score_1": score_1, "score_2": score_2,
                       "length_1": length_1, "length_2": length_2,
                       "winner": winner}


def wb_elo_single_round(votes: List[Dict[str, Any]], K: float, init_elos: Dict[str, float],
                        dynamic: bool, interval: int, length_margin: int,
                        rng: random.Random) -> Dict[str, float]:
    """wb_elo.compute_single_round (use_regressed_as_init=False), same updates.

    The votes are shuffled with rng.sample; every model starts at its Arena Elo or 1000. A vote
    between two models that both have an Arena Elo is skipped. A model without one moves by
    K * (score - expected); a model with one moves only when `dynamic`, and is pulled halfway back
    to its Arena Elo on its 1st, (interval+1)th, ... processed vote. With length_margin > 0 a vote
    whose replies differ by more than that many characters counts as a tie (upstream rewrites
    vote["winner"] in place, which gives the same outcome every round).
    """
    elo = dict(init_elos)
    sample_votes = rng.sample(votes, len(votes))
    for vote in sample_votes:
        if vote["model_1"] not in elo:
            elo[vote["model_1"]] = 1000
        if vote["model_2"] not in elo:
            elo[vote["model_2"]] = 1000
    vote_update_cnt: Dict[str, int] = defaultdict(int)
    for vote in sample_votes:
        model_1 = vote["model_1"]
        model_2 = vote["model_2"]
        if model_1 in init_elos and model_2 in init_elos:
            continue
        elo_1 = elo[model_1]
        elo_2 = elo[model_2]
        expected_1 = 1 / (1 + 10 ** ((elo_2 - elo_1) / 400))
        expected_2 = 1 / (1 + 10 ** ((elo_1 - elo_2) / 400))
        winner = vote["winner"]
        if length_margin > 0 and abs(vote["length_1"] - vote["length_2"]) > length_margin:
            winner = "tie"
        if winner == model_1:
            score_1, score_2 = 1, 0
        elif winner == model_2:
            score_1, score_2 = 0, 1
        else:
            score_1, score_2 = 0.5, 0.5
        if model_1 not in init_elos:
            elo[model_1] += K * (score_1 - expected_1)
        elif dynamic:
            elo[model_1] += K * (score_1 - expected_1)
            if interval > 0 and vote_update_cnt[model_1] % interval == 0:
                elo[model_1] = (elo[model_1] + init_elos[model_1]) / 2
        if model_2 not in init_elos:
            elo[model_2] += K * (score_2 - expected_2)
        elif dynamic:
            elo[model_2] += K * (score_2 - expected_2)
            if interval > 0 and vote_update_cnt[model_2] % interval == 0:
                elo[model_2] = (elo[model_2] + init_elos[model_2]) / 2
        vote_update_cnt[model_1] += 1
        vote_update_cnt[model_2] += 1
    return elo


def wb_elo_ratings(votes: Iterable[Dict[str, Any]], init_elos: Dict[str, float], *, K: float,
                   num_rounds: int, dynamic: bool, interval: int, length_margins: List[int],
                   seed: int) -> Dict[int, Dict[str, Dict[str, Any]]]:
    """wb_elo.compute_elo_based_on_votes + compute_wb_elo's elo_stat, for each length margin.

    Each margin is its own upstream run with the same rounds: `num_rounds` shuffles drawn in turn
    from one random.Random(seed). Upstream draws them in forked ProcessPoolExecutor workers (even
    with num_processes=1), and Python reseeds `random` from OS entropy in a forked child, so its
    random.seed(seed) does not fix them and no two upstream runs are alike; seeding in-process
    makes gbench's run deterministic. rng.sample's draws depend only on the number of votes, so
    every margin sees the same shuffles; they are drawn once per round and each margin replays
    wb_elo_single_round's updates on them (same arithmetic, same order). Per
    model: avg (the leaderboard value), std, median, the 2.5/97.5 percentile interval and
    init_elo ("-" without one), sorted by avg as upstream.
    """
    import numpy as np

    models: List[str] = []
    seen = set()
    compact = []
    for vote in votes:
        model_1, model_2 = vote["model_1"], vote["model_2"]
        for m in (model_1, model_2):
            if m not in seen:
                seen.add(m)
                models.append(m)
        base_code = 1 if vote["winner"] == model_1 else 2 if vote["winner"] == model_2 else 0
        codes = tuple(0 if lm > 0 and abs(vote["length_1"] - vote["length_2"]) > lm
                      else base_code for lm in length_margins)
        init_1, init_2 = model_1 in init_elos, model_2 in init_elos
        compact.append((model_1, model_2, init_1 and init_2, init_1, init_2, codes))
    rng = random.Random(seed)
    ratings: Dict[int, Dict[str, List[float]]] = {lm: defaultdict(list) for lm in length_margins}
    for _ in range(num_rounds):
        sample_votes = rng.sample(compact, len(compact))
        for index, lm in enumerate(length_margins):
            elo = dict(init_elos)
            for m in models:
                if m not in elo:
                    elo[m] = 1000
            vote_update_cnt: Dict[str, int] = {}
            for model_1, model_2, both_init, init_1, init_2, codes in sample_votes:
                if both_init:
                    continue
                elo_1 = elo[model_1]
                elo_2 = elo[model_2]
                expected_1 = 1 / (1 + 10 ** ((elo_2 - elo_1) / 400))
                expected_2 = 1 / (1 + 10 ** ((elo_1 - elo_2) / 400))
                code = codes[index]
                if code == 1:
                    score_1, score_2 = 1, 0
                elif code == 2:
                    score_1, score_2 = 0, 1
                else:
                    score_1, score_2 = 0.5, 0.5
                cnt_1 = vote_update_cnt.get(model_1, 0)
                cnt_2 = vote_update_cnt.get(model_2, 0)
                if not init_1:
                    elo[model_1] += K * (score_1 - expected_1)
                elif dynamic:
                    elo[model_1] += K * (score_1 - expected_1)
                    if interval > 0 and cnt_1 % interval == 0:
                        elo[model_1] = (elo[model_1] + init_elos[model_1]) / 2
                if not init_2:
                    elo[model_2] += K * (score_2 - expected_2)
                elif dynamic:
                    elo[model_2] += K * (score_2 - expected_2)
                    if interval > 0 and cnt_2 % interval == 0:
                        elo[model_2] = (elo[model_2] + init_elos[model_2]) / 2
                vote_update_cnt[model_1] = cnt_1 + 1
                vote_update_cnt[model_2] = cnt_2 + 1
            for model, rating in elo.items():
                ratings[lm][model].append(rating)
    out: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for lm, per_model in ratings.items():
        stat: Dict[str, Dict[str, Any]] = {}
        for model, values in per_model.items():
            avg = sum(values) / len(values)
            stat[model] = {
                "avg": avg,
                "std": (sum((v - avg) ** 2 for v in values) / len(values)) ** 0.5,
                "median": float(np.median(values)),
                "ci": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
                "init_elo": init_elos.get(model, "-"),
            }
        out[lm] = dict(sorted(stat.items(), key=lambda kv: kv[1]["avg"], reverse=True))
    return out


def wb_elo_name(config: str) -> str:
    """The model name wb_elo.py uses: the results file name without "@together" (data_utils)."""
    return config.replace("@together", "")


def _score_prompt_output(messages: List[Dict[str, Any]], checklist: List[str], reply: str) -> str:
    """The reply as batch_results_format.py stores it (read back from the score prompt)."""
    return score_output_from_prompt(build_wb_judge_prompt(messages, checklist, reply))


def _prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def wb_elo_pool_items(samples: List[Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]],
                      pool_outputs: Any) -> List[Dict[str, Any]]:
    """The pool replies whose score can reach a vote, one grading job each.

    `pool_outputs` is {config: {session_id: reply}}, or an iterable of (config, replies) pairs
    such as _load_wb_elo_pool_outputs(), which reads one model at a time: the items are built
    model by model and a model's replies are dropped before the next model is read.

    A reply that is empty or ends with "... (truncated)" in the score prompt never enters a vote
    (get_all_votes skips it before looking at the score), so it is not sent to the judge; its
    score could not change WB-Elo. An item keeps only what the run needs later: the pool model,
    the task, the sha256 of the judge prompt (the cache key; grade_wb_elo_pool rebuilds the
    prompt when it grades the item) and the reply's vote length.
    """
    pairs = pool_outputs.items() if isinstance(pool_outputs, dict) else pool_outputs
    items = []
    for config, replies in pairs:
        for messages, checklist, meta in samples:
            sid = meta.get("session_id")
            reply = replies.get(sid)
            if reply is None:
                raise RuntimeError(f"wildbench: WB-Elo pool model {config!r} has no reply for "
                                   f"task {sid!r}")
            prompt = build_wb_judge_prompt(messages, checklist, reply)
            length = vote_length(score_output_from_prompt(prompt))
            if length is None:
                continue
            items.append({"config": config, "session_id": sid,
                          "prompt_sha256": _prompt_sha256(prompt), "output_chars": length})
        replies = reply = None    # this model's replies are not held while the next is read
    return items


def _pool_judge_config() -> Any:
    """The judge config of every call, score and pairwise (upstream's batch body: temperature
    0, response_format json_object, max_tokens 1024), plus JUDGE_THINKING_ALLOWANCE for the
    Gemini judge's thinking."""
    return judge_config(response_mime_type="application/json",
                        max_output_tokens=JUDGE_OUTPUT_LIMIT)


def _config_fingerprint(cfg: Any) -> Any:
    """The judge config as data, for the cache key."""
    if cfg is None:
        return None
    dump = getattr(cfg, "model_dump", None)
    return dump(mode="json", exclude_none=True) if callable(dump) else repr(cfg)


def wb_elo_cache_dir() -> str:
    """Where WB-Elo pool verdicts are kept: $GBENCH_CACHE_DIR/wildbench/wb_elo_pool/<key>.

    <key> hashes the judge cascade, the judge config the score calls send (temperature, JSON
    mode), the pinned outputs revision and the score template, so a different judge, judge
    setting or template never reuses another's verdicts.
    """
    key = hashlib.sha256(json.dumps({
        "judge_cascade": judge_cascade(), "outputs": f"{_REF_DATASET}@{_REF_REVISION}",
        "template_sha256": hashlib.sha256(EVAL_TEMPLATE.encode("utf-8")).hexdigest(),
        "judge_config": _config_fingerprint(_pool_judge_config())},
        sort_keys=True).encode("utf-8")).hexdigest()
    return os.path.join(gbench_cache_dir(), "wildbench", "wb_elo_pool", key[:16])


def _read_pool_cache(path: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    cached: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not os.path.exists(path):
        return cached
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue                   # a line cut by an interrupted run: graded again
            cached[(rec.get("session_id"), rec.get("prompt_sha256"))] = rec
    return cached


def _load_wb_elo_pool_replies(config: str) -> Dict[str, str]:
    """{session_id: reply} of one WB-Elo pool model (pinned, sha256-checked), read again when its
    replies are graded."""
    return _fetch_model_outputs(config, WB_ELO_POOL[config],
                                "WB-Elo grades every pool model's replies")


async def grade_wb_elo_pool(
        items: List[Dict[str, Any]],
        tasks: Dict[str, Tuple[List[Dict[str, Any]], List[str]]],
        load_replies: Callable[[str], Dict[str, str]] = _load_wb_elo_pool_replies,
        judge_concurrency: int = 32) -> Dict[str, Any]:
    """Score-grade the pool replies with the gbench judge (eval_template.score.v2, JSON mode).

    Each verdict is appended to the cache as it arrives (one JSONL file per pool model under
    wb_elo_cache_dir()), so a rerun or an interrupted run only grades what is missing; a cached
    verdict is reused only for the same prompt (sha256). A reply the batch pipeline would drop
    (no JSON "score") is kept as score None, so it casts no vote, as upstream. A judge outage and
    a judge refusal are not cached and are counted. Returns {"grades": {config: {session_id:
    record}}, "outages", "refusals", "refused_sessions" ({session_id: count}), "block_reasons"
    ({reason: count}), "failures", "cached", "judge_models"}.

    The judge prompt of an item is built when the item is graded, from `tasks` ({session_id:
    (messages, checklist)}) and the pool model's replies, which `load_replies(config)` reads the
    first time one of that model's items is graded and which are dropped after its last one. The
    items are graded model by model, so at most two models' replies are held at a time.
    """
    import asyncio

    cache_dir = wb_elo_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    caches: Dict[str, Dict[Tuple[str, str], Dict[str, Any]]] = {}
    grades: Dict[str, Dict[str, Dict[str, Any]]] = {}
    todo: List[Dict[str, Any]] = []
    state = {"outages": 0, "refusals": 0, "failures": 0, "cached": 0, "outages_at_cap": 0}
    refused_sessions: Counter = Counter()
    block_reasons: Counter = Counter()
    judge_models: Counter = Counter()
    for item in items:
        config = item["config"]
        if config not in caches:
            caches[config] = _read_pool_cache(os.path.join(cache_dir, f"{config}.jsonl"))
        rec = caches[config].get((item["session_id"], item["prompt_sha256"]))
        if rec is not None:
            grades.setdefault(config, {})[item["session_id"]] = rec
            state["cached"] += 1
        else:
            todo.append(item)
    caches.clear()
    # Model by model: asyncio.Semaphore wakes its waiters in order, so the items run in this order.
    todo.sort(key=lambda item: item["config"])
    remaining: Counter = Counter(item["config"] for item in todo)
    loaded: Dict[str, Dict[str, str]] = {}
    semaphore = asyncio.Semaphore(judge_concurrency)
    cfg = _pool_judge_config()

    def _prompt(item: Dict[str, Any]) -> str:
        config, sid = item["config"], item["session_id"]
        if config not in loaded:
            loaded[config] = load_replies(config)
        messages, checklist = tasks[sid]
        prompt = build_wb_judge_prompt(messages, checklist, loaded[config][sid])
        remaining[config] -= 1
        if remaining[config] == 0:
            loaded.pop(config, None)
        if _prompt_sha256(prompt) != item["prompt_sha256"]:
            raise RuntimeError(f"wildbench: the WB-Elo judge prompt for {config!r} task {sid!r} "
                               "changed between loading and grading")
        return prompt

    async def _grade(item: Dict[str, Any]) -> None:
        async with semaphore:
            text, judge_used, info = await judge_generate_cascade_detailed(_prompt(item),
                                                                           config=cfg)
        if text is None:
            refusal = judge_refusal(info)
            if refusal is not None:
                state["refusals"] += 1
                refused_sessions[item["session_id"]] += 1
                block_reasons.update(refusal)
            else:
                state["outages"] += 1
                if info.get("max_tokens_without_text"):
                    # every answer stopped at max_output_tokens with no text (the thinking
                    # took the whole budget)
                    state["outages_at_cap"] += 1
            return
        rec = {"session_id": item["session_id"], "prompt_sha256": item["prompt_sha256"],
               "judge_model": judge_used, "judge_reply": text.strip(),
               "judge_finish_reason": info.get("finish_reason"),
               "score": parse_wb_score(text)}
        path = os.path.join(cache_dir, f"{item['config']}.jsonl")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        grades.setdefault(item["config"], {})[item["session_id"]] = rec

    if todo:
        logger.info("wildbench: grading %d WB-Elo pool replies (%d cached) with the judge.",
                    len(todo), state["cached"])
        await asyncio.gather(*[_grade(item) for item in todo])
    for per_model in grades.values():
        for rec in per_model.values():
            if rec.get("score") is None:
                state["failures"] += 1
            if rec.get("judge_model"):
                judge_models[rec["judge_model"]] += 1
    return {"grades": grades, "judge_models": dict(judge_models),
            "refused_sessions": dict(refused_sessions), "block_reasons": dict(block_reasons),
            **state}


def wb_elo_summary(traces: List[Dict[str, Any]], session_order: List[str],
                   pool_items: List[Dict[str, Any]], pool: Dict[str, Any],
                   config: Optional[Dict[str, Any]] = None,
                   init_elos: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """WB-Elo (Raw) and WB-Elo (LC) of the tested model, with the pool's table.

    Scores enter in a fixed order: the pool models in sorted name order, then the tested model,
    each with its sessions in `session_order` (upstream reads the results directory in
    os.listdir order and each file in batch-result order, neither of which is fixed). The tested
    model's score is its WB-Score raw score and its reply the one in its score prompt; a failed
    request, an outage or a dropped judge reply casts no vote.
    """
    config = dict(config or WB_ELO_CONFIG)
    init_elos = dict(WB_ELO_INIT if init_elos is None else init_elos)
    lengths_by_key = {(it["config"], it["session_id"]): it["output_chars"] for it in pool_items}
    order = {sid: i for i, sid in enumerate(session_order)}
    models: List[Tuple[str, List[Tuple[str, Any, Optional[int]]]]] = []
    for pool_config in sorted(pool.get("grades") or {}):
        per_model = pool["grades"][pool_config]
        items = [(sid, rec["score"], lengths_by_key[(pool_config, sid)])
                 for sid, rec in sorted(per_model.items(), key=lambda kv: order.get(kv[0], -1))
                 if rec.get("score") is not None and (pool_config, sid) in lengths_by_key]
        models.append((wb_elo_name(pool_config), items))
    target_items = []
    for t in traces:
        if t.get("wb_score_raw") is None or t.get("scoring_excluded"):
            continue
        sid = (t.get("extra_payload") or {}).get("session_id")
        output = _score_prompt_output(t.get("messages") or [], t.get("gold_answer") or [],
                                      t.get("response_text") or "")
        target_items.append((sid, t["wb_score_raw"], vote_length(output)))
    target_items.sort(key=lambda item: order.get(item[0], -1))
    models.append((WB_ELO_TARGET, target_items))
    scores, lengths = wb_elo_score_lengths(models)
    counts = {"votes": 0, "non_tie_votes": 0, "target_votes": 0}

    def _counted(votes):
        for v in votes:
            counts["votes"] += 1
            counts["non_tie_votes"] += v["winner"] != "tie"
            counts["target_votes"] += WB_ELO_TARGET in (v["model_1"], v["model_2"])
            yield v

    stats = wb_elo_ratings(
        _counted(iter_wb_elo_votes(scores, lengths, config["margin"], config["tie_margin"])),
        init_elos, K=config["K"], num_rounds=config["num_rounds"], dynamic=config["dynamic"],
        interval=config["interval"], length_margins=list(WB_ELO_LENGTH_MARGINS.values()),
        seed=config["seed"])
    target_votes = counts["target_votes"]
    out: Dict[str, Any] = {"votes": counts["votes"], "non_tie_votes": counts["non_tie_votes"],
                           "target_votes": target_votes, "config": config, "columns": {},
                           "tables": {}}
    for column, length_margin in WB_ELO_LENGTH_MARGINS.items():
        stat = stats[length_margin]
        target = stat.get(WB_ELO_TARGET)
        out["columns"][column] = None if target is None or not target_votes else {
            "avg": round(target["avg"], 4), "std": round(target["std"], 4),
            "median": round(target["median"], 4),
            "ci": [round(target["ci"][0], 4), round(target["ci"][1], 4)],
            "length_margin": length_margin}
        out["tables"][column] = {m: round(s["avg"], 4) for m, s in stat.items()}
    return out


def build_wb_judge_prompt(messages: List[Dict[str, Any]], checklist: List[str],
                          model_output: str) -> str:
    """The WildBench score-mode judge prompt, as src/eval.py builds it.

    compose_eval_item(): every turn but the last becomes `USER: ...` / `ASSISTANT: ...` history,
    the last turn is the current query. placeholder_generation(): the placeholders are filled
    in upstream's order, with no word truncation (the official score runs pass
    --max_words_to_eval -1), and the checklist as `- item` lines.
    """
    history, last_query = _history_and_query(messages)
    prompt = EVAL_TEMPLATE
    prompt = prompt.replace("{$history}", history)
    prompt = prompt.replace("{$user_query}", last_query)
    prompt = prompt.replace("{$model_output}", model_output)
    prompt = prompt.replace("{$checklist}", _checklist_markdown(checklist))
    return prompt


def _history_and_query(messages: List[Dict[str, Any]]) -> Tuple[str, str]:
    """src/eval.py compose_eval_item(): the `USER:` / `ASSISTANT:` history and the last query."""
    history = ""
    for x in messages[:-1]:
        if x.get("role") == "user":
            history += "USER: " + str(x.get("content", "")) + "\n\n"
        elif x.get("role") == "assistant":
            history += "ASSISTANT: " + str(x.get("content", "")) + "\n\n"
    last_query = str(messages[-1].get("content", "")) if messages else ""
    return history, last_query


def _checklist_markdown(checklist: List[str]) -> str:
    checklist_mardkdown = ""
    for checklist_item in checklist or []:
        checklist_mardkdown += f"- {checklist_item}\n"
    return checklist_mardkdown


def shorten(text, K=-1):
    """src/eval.py shorten(), verbatim: keep the first K space-separated words."""
    if K > 0 and len(text.split(" ")) > K:
        text = " ".join(text.split(" ")[:K]) + "... (truncated)"
    return text


def build_wb_pairwise_prompt(messages: List[Dict[str, Any]], checklist: List[str],
                             candidate_a: str, candidate_b: str,
                             max_words: int = PAIRWISE_MAX_WORDS) -> str:
    """The WildBench pairwise judge prompt, as src/eval.py placeholder_generation() builds it.

    History, query and both replies are cut to `max_words` words by shorten(); an empty reply
    (after the cut) is shown as EMPTY_RESPONSE_PLACEHOLDER; the checklist is not cut.
    """
    history, last_query = _history_and_query(messages)
    prompt = PAIRWISE_TEMPLATE
    prompt = prompt.replace("{$history}", shorten(history, max_words))
    prompt = prompt.replace("{$user_query}", shorten(last_query, max_words))
    a_output_str = shorten(candidate_a, max_words)
    b_output_str = shorten(candidate_b, max_words)
    if a_output_str.strip() == "":
        a_output_str = EMPTY_RESPONSE_PLACEHOLDER
    if b_output_str.strip() == "":
        b_output_str = EMPTY_RESPONSE_PLACEHOLDER
    prompt = prompt.replace("{$candidate_A}", a_output_str)
    prompt = prompt.replace("{$candidate_B}", b_output_str)
    prompt = prompt.replace("{$checklist}", _checklist_markdown(checklist))
    return prompt


def pairwise_outputs_from_prompt(prompt: str) -> Tuple[str, str]:
    """(Response A, Response B) read back from the prompt, as batch_results_format.py does."""
    model_a_output = prompt.split("<|begin_of_response_A|>\n")[1].split("<|end_of_response_A|>\n")[0]
    model_b_output = prompt.split("<|begin_of_response_B|>")[1].split("<|end_of_response_B|>\n")[0]
    return model_a_output.strip(), model_b_output.strip()


def score_output_from_prompt(prompt: str) -> str:
    """The reply read back from a score-mode prompt, as batch_results_format.py does."""
    return prompt.split("<|begin_of_response|>\n")[1].split("<|end_of_response|>\n")[0].strip()


def _parse_judge_json(result_str: Optional[str]) -> Optional[Dict[str, Any]]:
    """batch_results_format.py: `json.loads` of the judge reply; anything else is dropped."""
    try:
        parsed = json.loads(result_str or "")
    except Exception:                                                   # noqa: BLE001
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_wb_score(result_str: Optional[str]) -> Optional[float]:
    """The judge's 1-10 score as the batch pipeline reads it, None when it drops the reply.

    batch_results_format.py keeps a reply only when `json.loads` succeeds and the object has a
    "score"; _create_tables.py then applies float().
    """
    parsed = _parse_judge_json(result_str)
    if parsed is None or "score" not in parsed:
        return None
    try:
        return float(parsed["score"])
    except (TypeError, ValueError):
        return None


def parse_wb_choice(result_str: Optional[str]) -> Optional[str]:
    """The pairwise choice as batch_results_format.py reads it: one of the five, else None."""
    parsed = _parse_judge_json(result_str)
    if parsed is None or "choice" not in parsed:
        return None
    choice = parsed["choice"]
    return choice if isinstance(choice, str) and choice in PAIRWISE_CHOICES else None


def pairwise_outcome(choice: str, target_is_a: bool, target_output: str,
                     reference_output: str) -> Dict[str, Any]:
    """One parsed comparison, with what _create_tables.py reads from it.

    `target_output` / `reference_output` are the replies as they appear in the prompt (see
    pairwise_outputs_from_prompt): lengths are their characters, a reply is truncated when it
    ends with TRUNCATED_SUFFIX and empty when it is EMPTY_RESPONSE_PLACEHOLDER.
    """
    side, extent = PAIRWISE_CHOICES[choice]
    target_side = "A" if target_is_a else "B"
    winner = "tie" if side is None else ("target" if side == target_side else "reference")
    return {"choice": choice, "winner": winner, "extent": extent,
            "target_chars": len(target_output), "reference_chars": len(reference_output),
            "target_truncated": target_output.endswith(TRUNCATED_SUFFIX),
            "reference_truncated": reference_output.endswith(TRUNCATED_SUFFIX),
            "target_empty": target_output == EMPTY_RESPONSE_PLACEHOLDER,
            "reference_empty": reference_output == EMPTY_RESPONSE_PLACEHOLDER}


def wb_reward_table(outcomes: List[Dict[str, Any]], K: int = DEFAULT_LENGTH_MARGIN
                    ) -> Dict[str, Any]:
    """One reference's row, as _create_tables.py pairwise mode computes it.

    `outcomes` are the parsed comparisons (upstream's eval_result), each a pairwise_outcome()
    plus `groups` (the task's groups). Truncated or empty comparisons are skipped from the
    counts but stay in `total`; with K >= 0 a slight win (loss) is a tie when the winner's reply
    is more than K characters longer. reward = (win_much + 0.5 win - 0.5 lose - lose_much) /
    total; per group the same over the group's counted comparisons; task_macro_reward weights
    the groups present by TASK_MACRO_WEIGHTS over the sum of all five weights.
    """
    counts = {"win_much": 0, "win": 0, "tie": 0, "lose": 0, "lose_much": 0}
    lengths: List[int] = []
    task_cat_results: Dict[str, Dict[str, int]] = {}
    for item in outcomes:
        if item["target_truncated"] or item["reference_truncated"]:
            continue
        if item["target_empty"] or item["reference_empty"]:
            continue
        extent, winner = item["extent"], item["winner"]
        result_label = None
        if winner == "target":
            if extent == 2:
                result_label = "win_much"
            elif extent == 1:
                if K >= 0 and item["target_chars"] > item["reference_chars"] + K:
                    result_label = "tie"
                else:
                    result_label = "win"
        elif winner == "reference":
            if extent == 2:
                result_label = "lose_much"
            elif extent == 1:
                if K >= 0 and item["reference_chars"] > item["target_chars"] + K:
                    result_label = "tie"
                else:
                    result_label = "lose"
        elif winner == "tie":
            result_label = "tie"
        if result_label is None:
            raise ValueError(f"wildbench: unreadable pairwise outcome {item!r}")
        counts[result_label] += 1
        lengths.append(item["target_chars"])
        for tag in item.get("groups") or []:
            task_cat_results.setdefault(
                tag, {"win_much": 0, "win": 0, "tie": 0, "lose": 0, "lose_much": 0})
            task_cat_results[tag][result_label] += 1

    def _reward(c: Dict[str, int], n: int) -> float:
        return (c["win"] * 0.5 + c["win_much"] * 1 + c["tie"] * 0 - c["lose"] * 0.5
                - c["lose_much"] * 1) / n

    task_cat_reward = {tag: _reward(c, sum(c.values())) for tag, c in task_cat_results.items()}
    total = len(outcomes)
    return {**counts, "total": total,
            "avg_len": (sum(lengths) / len(lengths)) if lengths else None,
            "task_categorized_results": task_cat_results,
            "task_categorized_rewards": task_cat_reward,
            "reward": _reward(counts, total) if total else None,
            "task_macro_reward": (sum(task_cat_reward[tag] * TASK_MACRO_WEIGHTS[tag]
                                      for tag in task_cat_reward)
                                  / sum(TASK_MACRO_WEIGHTS.values())) if total else None,
            "K": K}


def wb_reward_columns(tables: Dict[str, Dict[str, Any]], K: int,
                      wb_score_task_macro: Optional[float]) -> Dict[str, Optional[float]]:
    """The leaderboard's reward columns for one K, named and scaled as _merge_results.py.

    `tables` maps a reference prefix (gpt4t / haiku / llama) to its wb_reward_table(); rewards
    are x100. A mixture needs all three references; a missing value is None.
    """
    def _x100(v: Optional[float]) -> Optional[float]:
        return None if v is None else round(v * 100, 4)

    def _mean(vals: List[Optional[float]]) -> Optional[float]:
        return None if any(v is None for v in vals) else round(sum(vals) / len(vals), 4)

    cols: Dict[str, Optional[float]] = {}
    for prefix in REFERENCE_MODELS:
        cols[f"{prefix}_reward.K={K}"] = _x100((tables.get(prefix) or {}).get("reward"))
    for tag in ALL_TASK_GROUPS:
        tagged = []
        for prefix in REFERENCE_MODELS:
            rewards = (tables.get(prefix) or {}).get("task_categorized_rewards") or {}
            value = _x100(rewards.get(tag)) if tag in rewards else None
            cols[f"{prefix}_reward.{tag}.K={K}"] = value
            tagged.append(value)
        cols[f"mixture_of_rewards.{tag}.K={K}"] = _mean(tagged)
    for prefix in REFERENCE_MODELS:
        cols[f"{prefix}_reward.task_macro.K={K}"] = _x100(
            (tables.get(prefix) or {}).get("task_macro_reward"))
    cols[f"mixture_of_rewards.K={K}"] = _mean(
        [cols[f"{prefix}_reward.K={K}"] for prefix in REFERENCE_MODELS])
    cols[f"task_macro_reward.K={K}"] = _mean(
        [cols[f"{prefix}_reward.task_macro.K={K}"] for prefix in REFERENCE_MODELS])
    tm = cols[f"task_macro_reward.K={K}"]
    cols["RewardScore_Avg"] = (None if tm is None or wb_score_task_macro is None
                               else round((wb_score_task_macro + tm) / 2, 4))
    return cols


def wb_reward_summary(traces: List[Dict[str, Any]], wb_score_task_macro: Optional[float]
                      ) -> Dict[str, Any]:
    """Every reference table and leaderboard reward column, for each K in LENGTH_MARGINS.

    A request that got no reply from the served endpoint (base's `harness_error`) is left out:
    it is not a comparison, so it is in no reference's total. A WB-Score grading failure or
    refusal on the same task does not remove its comparisons, which are judged on their own
    (upstream's pairwise tables do not read the score mode)."""
    outcomes: Dict[str, List[Dict[str, Any]]] = {prefix: [] for prefix in REFERENCE_MODELS}
    grades: Dict[str, Counter] = {prefix: Counter() for prefix in REFERENCE_MODELS}
    judge_models: Counter = Counter()
    for t in traces:
        if _failed_request(t):
            continue
        groups = (t.get("extra_payload") or {}).get("wb_task_groups") or []
        for prefix, rec in (t.get("wb_pairwise") or {}).items():
            grades[prefix][rec.get("judge_grade")] += 1
            if rec.get("judge_model"):
                judge_models[rec["judge_model"]] += 1
            if rec.get("outcome"):
                outcomes[prefix].append({**rec["outcome"], "groups": groups})
    by_k: Dict[str, Any] = {}
    for K in LENGTH_MARGINS:
        tables = {prefix: wb_reward_table(outcomes[prefix], K) for prefix in REFERENCE_MODELS}
        by_k[str(K)] = {"columns": wb_reward_columns(tables, K, wb_score_task_macro),
                        "references": tables}
    return {"by_k": by_k,
            "judge_failures": {p: grades[p].get("JUDGE_FAILED", 0) for p in REFERENCE_MODELS},
            "judge_outages": {p: grades[p].get("JUDGE_OUTAGE", 0) for p in REFERENCE_MODELS},
            "judge_refusals": {p: grades[p].get(JUDGE_REFUSED, 0) for p in REFERENCE_MODELS},
            "judge_models": dict(judge_models)}


def _failed_request(trace: Dict[str, Any]) -> bool:
    """A request that got no reply from the served endpoint: base's `harness_error` (a harness
    error, never the model's outcome), or no response text at all."""
    return trace.get("response_text") is None or bool(trace.get("harness_error"))


async def _async_judge_wildbench(
    sample_traces: List[Dict[str, Any]],
    judge_concurrency: int = 32,
    references: Optional[Dict[str, Dict[str, str]]] = None,
) -> None:
    """WildBench judges: WB-Score for every trace, and WB-Reward against each reference.

    Score mode: eval_template.score.v2 in JSON mode, 1-10 per reply; each trace gets
    ``wb_score_raw`` (the judge's 1-10) and ``wb_score`` = (raw - 5) * 2, upstream's per-item
    adjusted score (-8..+10); run_wildbench reports mean(wb_score) * 10. An empty reply is
    judged like any other (the batch pipeline submits it too); a request that got no reply from
    the served endpoint is a harness error: not judged, marked `scoring_excluded` and left out of
    every score.

    Pairwise mode (when `references` is given, {prefix: {session_id: reply}}): each trace gets
    ``wb_pairwise[prefix]`` with the judge's choice and the outcome wb_reward_table reads. A
    request that got no reply is not compared (no judge call, no outcome).
    """
    import asyncio
    from tqdm import tqdm

    semaphore = asyncio.Semaphore(judge_concurrency)
    # Upstream's judge request: temperature 0, json_mode=True (response_format json_object),
    # max_tokens 1024.
    cfg = _pool_judge_config()

    def _record(trace: Dict[str, Any], score: float) -> None:
        trace["wb_score"] = round((score - 5.0) * 2.0, 4)
        trace["wb_score_raw"] = score
        trace["is_correct"] = score >= 7.0            # secondary pass_rate only

    async def _judge_single(trace: Dict[str, Any], pbar: tqdm) -> None:
        resp_text = trace.get("response_text")
        if _failed_request(trace):
            # No reply from the served endpoint: a harness error, never the model's outcome.
            # Nothing to judge; left out of every score (base counts it in failed_requests).
            trace["is_correct"] = False
            trace["scoring_excluded"] = True
            trace["judge_grade"] = "not_judged_failed_request"
            pbar.update(1)
            return
        prompt = build_wb_judge_prompt(trace.get("messages", []), trace.get("gold_answer") or [],
                                       resp_text)
        in_prompt = score_output_from_prompt(prompt)
        trace["wb_output_chars"] = len(in_prompt)
        trace["wb_output_truncated"] = in_prompt.endswith(TRUNCATED_SUFFIX)
        async with semaphore:
            text, judge_used, info = await judge_generate_cascade_detailed(prompt, config=cfg)
        if trace.get("status") == "PENDING_JUDGE":   # keep EMPTY_RESPONSE etc.
            trace["status"] = "OK"
        if text is None:
            refusal = judge_refusal(info)
            if refusal is not None:
                # The Gemini API refused to grade this task's prompt: a judge-side failure that
                # a rerun would not recover, so it is not an outage. Left out of the mean and
                # the denominator (scoring_excluded) and counted as a refusal.
                trace["judge_grade"] = JUDGE_REFUSED
                trace["judge_block_reasons"] = refusal
                trace["scoring_excluded"] = True
                trace["is_correct"] = False
                pbar.update(1)
                return
            # Judge cascade exhausted (infra outage): excluded from the mean, NOT scored.
            trace["judge_grade"] = "JUDGE_OUTAGE"
            if info.get("max_tokens_without_text"):
                # answers that stopped at max_output_tokens with no text
                trace["judge_max_tokens_without_text"] = int(info["max_tokens_without_text"])
            pbar.update(1)
            return
        trace["judge_model"] = judge_used
        trace["judge_reply"] = text.strip()
        trace["judge_finish_reason"] = info.get("finish_reason")
        score = parse_wb_score(text)
        if score is None:
            # The batch pipeline drops a reply it cannot read: a grading failure, counted in
            # judge_failures and kept out of the mean (scoring_excluded), so a grader-side
            # format error neither moves the score nor poses as an outage.
            trace["is_correct"] = False
            trace["judge_grade"] = "JUDGE_FAILED"
            trace["scoring_excluded"] = True
            pbar.update(1)
            return
        _record(trace, score)
        trace["judge_grade"] = f"SCORE_{int(round(score))}"
        pbar.update(1)

    async def _judge_pair(trace: Dict[str, Any], prefix: str, reference: str,
                          pbar: tqdm) -> None:
        target_is_a = bool((trace.get("extra_payload") or {}).get("wb_pairwise_target_is_a"))
        if _failed_request(trace):
            # No reply from the served endpoint: a harness error, not a comparison.
            trace.setdefault("wb_pairwise", {})[prefix] = {
                "reference_model": REFERENCE_MODELS[prefix][0],
                "target_side": "A" if target_is_a else "B",
                "judge_grade": "not_judged_failed_request"}
            pbar.update(1)
            return
        target = trace.get("response_text") or ""
        cand_a, cand_b = (target, reference) if target_is_a else (reference, target)
        prompt = build_wb_pairwise_prompt(trace.get("messages", []),
                                          trace.get("gold_answer") or [], cand_a, cand_b)
        async with semaphore:
            text, judge_used, info = await judge_generate_cascade_detailed(prompt, config=cfg)
        rec: Dict[str, Any] = {"reference_model": REFERENCE_MODELS[prefix][0],
                               "target_side": "A" if target_is_a else "B"}
        refusal = judge_refusal(info) if text is None else None
        if refusal is not None:
            rec["judge_grade"] = JUDGE_REFUSED
            rec["judge_block_reasons"] = refusal
        elif text is None:
            rec["judge_grade"] = "JUDGE_OUTAGE"
            if info.get("max_tokens_without_text"):
                rec["judge_max_tokens_without_text"] = int(info["max_tokens_without_text"])
        else:
            rec["judge_model"] = judge_used
            rec["judge_reply"] = text.strip()
            rec["judge_finish_reason"] = info.get("finish_reason")
            choice = parse_wb_choice(text)
            if choice is None:
                rec["judge_grade"] = "JUDGE_FAILED"      # dropped, as the batch pipeline does
            else:
                out_a, out_b = pairwise_outputs_from_prompt(prompt)
                target_out, reference_out = (out_a, out_b) if target_is_a else (out_b, out_a)
                rec["judge_grade"] = f"CHOICE_{choice}"
                rec["outcome"] = pairwise_outcome(choice, target_is_a, target_out, reference_out)
        trace.setdefault("wb_pairwise", {})[prefix] = rec
        pbar.update(1)

    jobs = []
    with tqdm(total=len(sample_traces) * (1 + len(references or {})),
              desc="Judging [WILDBENCH]") as pbar:
        for t in sample_traces:
            jobs.append(_judge_single(t, pbar))
            for prefix, outputs in (references or {}).items():
                sid = (t.get("extra_payload") or {}).get("session_id")
                jobs.append(_judge_pair(t, prefix, outputs[sid], pbar))
        await asyncio.gather(*jobs)


def run_wildbench(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run WildBench v2: headline WB_score.task_macro, plus WB_score, WB-Reward and WB-Elo."""
    import asyncio
    from .metrics import finalize_mean_metric
    skip = gemini_required_skip(EVAL_NAME, model_name)
    if skip is not None:
        return skip
    limit = kwargs.get("limit")
    samples = _load_wildbench_samples(limit=limit)
    references = _load_reference_outputs()
    for _, _, meta in samples:
        missing = [p for p, outs in references.items() if meta.get("session_id") not in outs]
        if missing:
            raise RuntimeError(f"wildbench: task {meta.get('session_id')!r} has no reference "
                               f"reply for {missing}")
    # WB-Elo: every pool model's published replies to the same tasks (the GBENCH_SHARD shard
    # run_eval_suite will run, when one is set), graded by the same judge while the tested
    # model's replies are judged (fetched first, so a missing file fails fast).
    run_samples, _shard = apply_shard(samples, EVAL_NAME)
    pool_items = wb_elo_pool_items(run_samples, _load_wb_elo_pool_outputs())
    pool_tasks = {meta.get("session_id"): (messages, checklist)
                  for messages, checklist, meta in run_samples}
    pool: Dict[str, Any] = {}

    async def _judge(traces: List[Dict[str, Any]]) -> None:
        async def _pool() -> None:
            pool.update(await grade_wb_elo_pool(pool_items, pool_tasks,
                                                _load_wb_elo_pool_replies))
        await asyncio.gather(_async_judge_wildbench(traces, references=references), _pool())

    # One generation per sample: base runs a batch-scored suite (async_eval_fn, no
    # supports_attempts) at 1 attempt whatever --attempt-count asks.
    limit_plan = base_suite_plan("wildbench", samples, limit, base_attempts(batch_scored=True))
    result = run_eval_suite(
        eval_name=EVAL_NAME,
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        async_eval_fn=_judge,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=limit,
        max_output_tokens=kwargs.get("max_output_tokens"),
        official_sampling=OFFICIAL_SAMPLING,   # its max_tokens follows the token-limit policy
    )
    result["eval_limit_plan"] = limit_plan
    # Requests that got no reply from the served endpoint (harness errors), grading failures and
    # refusals are marked scoring_excluded and judge outages JUDGE_OUTAGE: finalize_mean_metric
    # leaves all of them out of the mean.
    result = finalize_mean_metric(
        result,
        metric_name=("WB_score.task_macro (the leaderboard's score column: per task-group mean "
                     "raw score adjusted to (mean-5)*2, weighted, x10); WB_score = (mean_raw-5)*20 "
                     "in wb_score; WB-Reward in wb_reward_*; WB-Elo in wb_elo / wb_elo_lc"),
        score_key="wb_score",
        secondary_key="pass_rate_at_7",
        # Headline = mean(adjusted wb_score) * 10 = (mean_raw-5)*20, the canonical WildBench WB-Score.
        scale=10.0,
    )
    traces = result.get("sample_traces", []) or []
    result["wb_score"] = result.get("accuracy")
    macro, per_group = wb_task_macro(traces)
    result["wb_score_task_macro"] = macro
    result["wb_score_task_groups"] = per_group
    result["wb_length"] = wb_length(traces)
    # Headline: WB_score.task_macro (None when no reply could enter a task group).
    result["accuracy"] = macro

    reward = wb_reward_summary(traces, macro)
    default = reward["by_k"][str(DEFAULT_LENGTH_MARGIN)]["columns"]
    k = DEFAULT_LENGTH_MARGIN
    result["wb_reward_task_macro"] = default[f"task_macro_reward.K={k}"]
    result["wb_reward_mix"] = default[f"mixture_of_rewards.K={k}"]
    result["wb_reward_by_reference"] = {p: default[f"{p}_reward.K={k}"] for p in REFERENCE_MODELS}
    result["wb_reward_score_avg"] = default["RewardScore_Avg"]
    result["wb_reward_length_margin"] = k
    result["wb_reward"] = reward["by_k"]
    result["wb_reward_judge_failures"] = reward["judge_failures"]
    result["wb_reward_judge_outages"] = reward["judge_outages"]
    result["wb_reward_judge_refusals"] = reward["judge_refusals"]
    result["wb_reward_judge_models"] = reward["judge_models"]
    result["wb_reward_references"] = {
        p: f"{_REF_DATASET}@{_REF_REVISION} ({cfg})" for p, (cfg, _) in REFERENCE_MODELS.items()}
    reward_outages = sum(reward["judge_outages"].values())
    reward_refusals = sum(reward["judge_refusals"].values())

    logger.info("wildbench: simulating WB-Elo (%d rounds, 2 length margins).",
                WB_ELO_CONFIG["num_rounds"])
    session_order = [meta.get("session_id") for _, _, meta in run_samples]
    elo = wb_elo_summary(traces, session_order, pool_items, pool)
    raw, lc = elo["columns"].get("WB_Elo"), elo["columns"].get("WB_Elo_LC")
    result["wb_elo"] = raw["avg"] if raw else None
    result["wb_elo_lc"] = lc["avg"] if lc else None
    result["wb_elo_detail"] = {
        "columns": elo["columns"], "votes": elo["votes"], "non_tie_votes": elo["non_tie_votes"],
        "target_votes": elo["target_votes"], "config": elo["config"],
        "length_margins": dict(WB_ELO_LENGTH_MARGINS), "target_name": WB_ELO_TARGET,
        "pool_models": len(WB_ELO_POOL), "pool_replies_graded": len(pool_items),
        "pool_verdicts_from_cache": pool.get("cached", 0),
        "pool_judge_failures": pool.get("failures", 0),
        "pool_judge_outages": pool.get("outages", 0),
        "pool_judge_refusals": pool.get("refusals", 0),
        "pool_judge_refused_sessions": pool.get("refused_sessions", {}),
        "pool_judge_models": pool.get("judge_models", {}),
        "pool_cache_dir": wb_elo_cache_dir(),
        "pool_outputs": f"{_REF_DATASET}@{_REF_REVISION}",
        "init_elos": "leaderboard/data_dir/all_stat.json 'Arena Elo (hard-en) - latest' "
                     "@d6b8dcaf377d"}
    result["wb_elo_tables"] = elo["tables"]
    pool_outages = int(pool.get("outages", 0))
    pool_refusals = int(pool.get("refusals", 0))

    # The per-category (primary tag) rows show the category's WB_score, on the headline's scale,
    # instead of base's pass@7 tally, which moves to the cell's `pass_rate_at_7`.
    by_category = wb_score_by_category(traces)
    for cat, cell in (result.get("category_accuracy") or {}).items():
        if isinstance(cell, dict):
            cell["pass_rate_at_7"] = cell.get("accuracy")
            cell["accuracy"] = by_category.get(cat)

    # Judge-side failures in the WB-Score, WB-Reward and WB-Elo calls: harness errors, left out
    # of every score, counted, named below, and the run is completed_with_errors.
    score_outages = sum(1 for t in traces if t.get("judge_grade") == "JUDGE_OUTAGE")
    refused = [t for t in traces if t.get("judge_grade") == JUDGE_REFUSED]
    refused_ids = sorted({str((t.get("extra_payload") or {}).get("session_id")) for t in refused}
                         | {str((t.get("extra_payload") or {}).get("session_id")) for t in traces
                            for rec in (t.get("wb_pairwise") or {}).values()
                            if rec.get("judge_grade") == JUDGE_REFUSED}
                         | set(pool.get("refused_sessions") or {}))
    block_reasons: Counter = Counter(pool.get("block_reasons") or {})
    for t in traces:
        block_reasons.update(t.get("judge_block_reasons") or [])
        for rec in (t.get("wb_pairwise") or {}).values():
            block_reasons.update(rec.get("judge_block_reasons") or [])
    result["judge_refusals"] = {"wb_score": len(refused), "wb_reward": reward_refusals,
                                "wb_elo_pool": pool_refusals, "tasks": refused_ids,
                                "block_reasons": dict(block_reasons)}
    at_cap = (sum(1 for t in traces if t.get("judge_finish_reason") == "MAX_TOKENS")
              + sum(1 for t in traces for rec in (t.get("wb_pairwise") or {}).values()
                    if rec.get("judge_finish_reason") == "MAX_TOKENS")
              + sum(1 for per_model in (pool.get("grades") or {}).values()
                    for rec in per_model.values()
                    if rec.get("judge_finish_reason") == "MAX_TOKENS"))
    result["judge_replies_at_max_output_tokens"] = at_cap
    # Judge outages where every answer stopped at max_output_tokens with no text: the thinking
    # took the whole budget (WB-Score traces, WB-Reward comparisons, WB-Elo pool verdicts).
    outages_at_cap = (sum(1 for t in traces if t.get("judge_max_tokens_without_text"))
                      + sum(1 for t in traces for rec in (t.get("wb_pairwise") or {}).values()
                            if rec.get("judge_max_tokens_without_text"))
                      + int(pool.get("outages_at_cap") or 0))
    result["judge_outages_after_max_output_tokens"] = outages_at_cap
    # WB-Reward comparisons and WB-Elo pool verdicts whose judge reply could not be read (no
    # JSON choice / score): dropped as upstream's batch_results_format drops them, and counted
    # here as the judge-side harness errors they are (WB-Score's own are counted by base).
    reward_failures = sum(reward["judge_failures"].values())
    pool_failures = int(pool.get("failures", 0))
    reward_failures_at_cap = sum(
        1 for t in traces if not _failed_request(t)
        for rec in (t.get("wb_pairwise") or {}).values()
        if rec.get("judge_grade") == "JUDGE_FAILED"
        and rec.get("judge_finish_reason") == "MAX_TOKENS")
    pool_failures_at_cap = sum(1 for per_model in (pool.get("grades") or {}).values()
                               for rec in per_model.values()
                               if rec.get("score") is None
                               and rec.get("judge_finish_reason") == "MAX_TOKENS")
    result["judge_failures_dropped"] = {
        "wb_reward": reward_failures, "wb_elo_pool": pool_failures,
        "at_max_output_tokens": reward_failures_at_cap + pool_failures_at_cap}
    causes = []
    if reward_outages:
        causes.append(f"wb_reward_judge_outages {reward_outages}")
    if pool_outages:
        causes.append(f"wb_elo_pool_judge_outages {pool_outages}")
    if reward_failures:
        causes.append(f"wb_reward_judge_failures {reward_failures} (comparisons dropped: judge "
                      "reply not readable JSON)")
    if pool_failures:
        causes.append(f"wb_elo_pool_judge_failures {pool_failures} (pool verdicts dropped: "
                      "judge reply not readable JSON)")
    if refused or reward_refusals or pool_refusals:
        causes.append(f"judge_refusals {len(refused)} WB-Score, {reward_refusals} WB-Reward, "
                      f"{pool_refusals} WB-Elo pool (tasks {', '.join(refused_ids)})")
    if causes:
        if result.get("status") == "success":
            result["status"] = "completed_with_errors"
        if result.get("status") == "completed_with_errors":
            result["status_reason"] = "; ".join(
                [r for r in [result.get("status_reason")] if r] + causes)

    result["dataset"] = f"{_DATASET}@{_REVISION} ({_CONFIG}, test)"
    result["judge"] = "gbench-gemini-cascade"
    # One score-mode judge call per graded task, so this is also the answered calls per model;
    # the pairwise calls are counted in wb_reward_judge_models.
    result["judge_models"] = dict(Counter(t["judge_model"] for t in traces
                                          if t.get("judge_model")))
    result["judge_cascade"] = judge_cascade()
    result["judge_sampling"] = judge_sampling_record()
    reasons = ["graded by gbench's standard Gemini cascade instead of WildBench's "
               "gpt-4o-2024-05-13 (WB-Score) and gpt-4-turbo-2024-04-09 (WB-Reward) judges "
               "(a gbench convention); WB-Elo re-grades the 63 pool models' published replies "
               "with the same cascade"]
    if limit:
        reasons.append(f"--eval-limit subset ({len(samples)} of the {N_TASKS:,} tasks; the "
                       "WB-Elo pool is graded on the same tasks)")
    absent = task_groups_absent(result.get("wb_score_task_groups"))
    result["wb_task_groups_absent"] = absent
    if absent:
        present = sum(w for g, w in TASK_MACRO_WEIGHTS.items() if g not in absent)
        reasons.append(f"task groups {', '.join(absent)} absent from the scored tasks count as 0 "
                       f"in WB_score.task_macro and task_macro_reward (upstream's formula "
                       f"divides by all five weights: cap = {present:g}/"
                       f"{sum(TASK_MACRO_WEIGHTS.values()):g} of the full scale)")
    if os.environ.get("GBENCH_SHARD"):
        reasons.append(f"GBENCH_SHARD {os.environ['GBENCH_SHARD']} shard (the WB-Elo pool is "
                       "graded on the same tasks)")
    n_tasks = len(traces)
    if score_outages:
        reasons.append(f"{score_outages} WB-Score repl(ies) not graded (judge outage); the "
                       f"headline covers the tasks the judge graded, not all {n_tasks}")
    if refused:
        reasons.append(f"{len(refused)} WB-Score repl(ies) not graded because the Gemini judge "
                       f"refused the task (block reasons: "
                       f"{', '.join(sorted(block_reasons)) or 'none given'}); the headline "
                       f"covers the tasks the judge graded, not all {n_tasks}")
    if reward_outages:
        reasons.append(f"{reward_outages} WB-Reward comparison(s) lost to a judge outage")
    if reward_refusals:
        reasons.append(f"{reward_refusals} WB-Reward comparison(s) lost to a judge refusal")
    if pool_outages:
        reasons.append(f"{pool_outages} WB-Elo pool repl(ies) not graded (judge outage)")
    if pool_refusals:
        reasons.append(f"{pool_refusals} WB-Elo pool repl(ies) not graded (judge refusal)")
    if reward_failures or pool_failures:
        at_cap_failures = reward_failures_at_cap + pool_failures_at_cap
        reasons.append(
            f"{reward_failures} WB-Reward comparison(s) and {pool_failures} WB-Elo pool "
            "verdict(s) dropped: judge reply not readable JSON"
            + (f" ({at_cap_failures} stopped at max_output_tokens {JUDGE_OUTPUT_LIMIT})"
               if at_cap_failures else ""))
    if refused_ids:
        reasons.append(f"tasks the Gemini judge refused: {', '.join(refused_ids)}")
    if at_cap:
        reasons.append(f"{at_cap} judge repl(ies) stopped at max_output_tokens "
                       f"{JUDGE_OUTPUT_LIMIT} (src/eval.py's {JUDGE_MAX_TOKENS} plus "
                       f"{JUDGE_THINKING_ALLOWANCE} for the Gemini judge's thinking)")
    if outages_at_cap:
        reasons.append(f"{outages_at_cap} judge outage(s) where every answer stopped at "
                       f"max_output_tokens {JUDGE_OUTPUT_LIMIT} with no text")
    sampling = result.get("official_sampling") or {}
    if sampling and sampling.get("temperature_matches_official") is False:
        reasons.append(f"temperature {result.get('temperature')} (WildBench decodes at 0)")
    cut = record_budget_cutoffs(result)
    if cut:
        reasons.append(cut)
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = "; ".join(reasons)
    # Every task a judge outage, refusal or grading failure: nothing was measured.
    withhold_if_nothing_scored(result, "task")
    return result
