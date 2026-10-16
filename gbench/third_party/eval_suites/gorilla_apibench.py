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

# Evaluation Suite: gorilla_apibench
# Description: Gorilla APIBench (UC Berkeley API call generation) - all three hubs (TorchHub, HuggingFace, TensorFlow Hub), 0-shot and the BM25 / GPT-Index / Oracle retriever settings, AST sub-tree matching

"""gbench native built-in runner for gorilla_apibench (Tool Use & Function Calling).

Runs Gorilla's own APIBench evaluation (ShishirPatil/gorilla `eval/`) in the four settings the
repo ships question files for and the paper reports: 0-shot and the BM25, GPT-Index and Oracle
retriever settings (`eval-data/questions/<hub>/questions_<hub>_{0_shot,bm25,gpt_index,oracle}.jsonl`;
the retriever files already carry the retrieved API document in the question text, "Use this API
documentation for reference: ...", so no retriever runs here). Every question of the three hubs in
every setting is sent through `get_llm_responses.encode_question` (system + user prompt that lists
the hub's domains and leaves the model to pick one), and each reply is scored by the hub's
`eval-scripts/ast_eval_{th,hf,tf}.py`: the `api_call` field is cut out of the reply, parsed with
tree-sitter, and matched by AST sub-tree against the hub's whole API database. A reply is correct
when the matched database entry's domain equals the question's; a reply that matches no entry is a
hallucination. The headline is the 0-shot setting.

Sampling: one completion and temperature 0.0 without `--thinking`, 1.0 with it (gbench's rule for
every suite, base.resolve_temperature; get_llm_responses.py sends `n=1, temperature=0`), and no
max_tokens (the server grants the room left
in the context window; --max-output-tokens is not applied and `max_output_tokens_note` says so).
Override the temperature for a whole run with `--temperature`, or for this suite alone with
`GBENCH_GORILLA_APIBENCH_TEMPERATURE`, which takes precedence over both; the result records
upstream's value under `official_sampling`. No LLM judge.
"""

import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import urllib.request
from typing import Any, Dict, List, Optional, Tuple
from .base import OfficialSampling, gbench_cache_dir, run_eval_suite, strip_thinking_tags
from .metrics import scorable_traces
from .sampling import parse_shard, stratified_sample
from .fc_common import base_suite_plan, run_attempts
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Function Calling"
DOCS_URL = "docs/evals/gorilla_apibench.md"

#: ShishirPatil/gorilla revision the prompts, questions and scorers are taken from.
GORILLA_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
#: gorilla-llm/APIBench revision; its eval/api files are byte-identical to the repo's
#: data/apibench/*_eval.json and data/api/*_api.jsonl at GORILLA_COMMIT.
APIBENCH_REPO = "gorilla-llm/APIBench"
APIBENCH_REVISION = "ac21e1892e634dfa25f8ad75f16cbdbfb0a5736d"

#: hub -> (encode_question api_name, question file under eval/eval-data/questions/, git blob of
#: that file at GORILLA_COMMIT, APIBench qa-pair file, APIBench API database file)
HUBS = {
    "torchhub": ("torchhub", "torchhub/questions_torchhub_0_shot.jsonl",
                 "1eb3068a46b63221dd4a2e47f42a3863e1413faf",
                 "torchhub_eval.json", "torchhub_api.jsonl"),
    "huggingface": ("huggingface", "huggingface/questions_huggingface_0_shot.jsonl",
                    "2885757192819c6f9dd79c717ad292f0cf9ffdf8",
                    "huggingface_eval.json", "huggingface_api.jsonl"),
    "tensorflowhub": ("tensorhub", "tensorflowhub/questions_tensorflowhub_0_shot.jsonl",
                      "b3a08ceedcec7a426ae88590563262f7302f301f",
                      "tensorflow_eval.json", "tensorflowhub_api.jsonl"),
}
#: Question counts at GORILLA_COMMIT (APIBench's published denominators; every setting's file has
#: the same question ids).
HUB_SIZES = {"torchhub": 186, "huggingface": 904, "tensorflowhub": 685}

#: The settings Gorilla ships question files for (get_llm_responses.py --question_data), in the
#: paper's order; 0-shot is the headline.
SETTINGS = ("0_shot", "bm25", "gpt_index", "oracle")
HEADLINE_SETTING = "0_shot"
#: (hub, retriever setting) -> git blob of eval-data/questions/<hub>/questions_<hub>_<setting>.jsonl
#: at GORILLA_COMMIT (the 0-shot blobs are in HUBS).
RETRIEVER_QUESTION_BLOBS = {
    ("torchhub", "bm25"): "b4908b0d31f790619c3e6d3508eab7eb2f5ec334",
    ("torchhub", "gpt_index"): "2b4675a89bdbf79c74ebd0b3bdb2884c2db48edb",
    ("torchhub", "oracle"): "51cc9bdfd5700ccc2ba45b4e8f8d5b97ad34107d",
    ("huggingface", "bm25"): "636e850937fe8e3bda9bc78b361088dcd309fe6c",
    ("huggingface", "gpt_index"): "6a1dd34fe30a2946cb7de285293222a2a62a2ff3",
    ("huggingface", "oracle"): "8ad9bcbff250ff8cf0848d80ce3bf55fb29a8157",
    ("tensorflowhub", "bm25"): "c4dae8bcb437996e8fffe8c0cac1fa27bf49f41d",
    ("tensorflowhub", "gpt_index"): "d1529d937bb1c2978c631e41f1282a3501294478",
    ("tensorflowhub", "oracle"): "07e5350f2dbe7b8a6b5ab2d6b04838e22206d8d2",
}


def question_file(hub: str, setting: str) -> Tuple[str, str]:
    """(path under eval-data/questions/, git blob) of one hub's question file in `setting`."""
    if setting == "0_shot":
        return HUBS[hub][1], HUBS[hub][2]
    return f"{hub}/questions_{hub}_{setting}.jsonl", RETRIEVER_QUESTION_BLOBS[(hub, setting)]

_QUESTIONS_URL = ("https://raw.githubusercontent.com/ShishirPatil/gorilla/{commit}/gorilla/eval/"
                  "eval-data/questions/{rel}")

#: gorilla/eval/get_llm_responses.py get_response: openai.ChatCompletion.create(model, messages,
#: n=1, temperature=0), nothing else (no max_tokens).
GORILLA_SAMPLING = OfficialSampling(
    reference=(f"ShishirPatil/gorilla@{GORILLA_COMMIT[:12]} gorilla/eval/get_llm_responses.py "
               "get_response (n=1, temperature=0)"),
    temperature=0.0,
    send_max_tokens=False,
)


# --------------------------------------------------------------------------- #
# Prompt: gorilla/eval/get_llm_responses.py encode_question, verbatim (the domain lists keep their
# backslash continuations, so the runs of spaces inside them are part of the prompt).
# --------------------------------------------------------------------------- #
def encode_question(question, api_name):
    """Encode multiple prompt instructions into a single string."""

    prompts = []
    if api_name == "torchhub":
        domains = "1. $DOMAIN is inferred from the task description and should include one of {Classification, Semantic Segmentation, Object Detection, Audio Separation, Video Classification, Text-to-Speech}."
    elif api_name == "huggingface":
        domains = "1. $DOMAIN should include one of {Multimodal Feature Extraction, Multimodal Text-to-Image, Multimodal Image-to-Text, Multimodal Text-to-Video, \
        Multimodal Visual Question Answering, Multimodal Document Question Answer, Multimodal Graph Machine Learning, Computer Vision Depth Estimation,\
        Computer Vision Image Classification, Computer Vision Object Detection, Computer Vision Image Segmentation, Computer Vision Image-to-Image, \
        Computer Vision Unconditional Image Generation, Computer Vision Video Classification, Computer Vision Zero-Shor Image Classification, \
        Natural Language Processing Text Classification, Natural Language Processing Token Classification, Natural Language Processing Table Question Answering, \
        Natural Language Processing Question Answering, Natural Language Processing Zero-Shot Classification, Natural Language Processing Translation, \
        Natural Language Processing Summarization, Natural Language Processing Conversational, Natural Language Processing Text Generation, Natural Language Processing Fill-Mask,\
        Natural Language Processing Text2Text Generation, Natural Language Processing Sentence Similarity, Audio Text-to-Speech, Audio Automatic Speech Recognition, \
        Audio Audio-to-Audio, Audio Audio Classification, Audio Voice Activity Detection, Tabular Tabular Classification, Tabular Tabular Regression, \
        Reinforcement Learning Reinforcement Learning, Reinforcement Learning Robotics }"
    elif api_name == "tensorhub":
        domains = "1. $DOMAIN is inferred from the task description and should include one of {text-sequence-alignment, text-embedding, text-language-model, text-preprocessing, text-classification, text-generation, text-question-answering, text-retrieval-question-answering, text-segmentation, text-to-mel, image-classification, image-feature-vector, image-object-detection, image-segmentation, image-generator, image-pose-detection, image-rnn-agent, image-augmentation, image-classifier, image-style-transfer, image-aesthetic-quality, image-depth-estimation, image-super-resolution, image-deblurring, image-extrapolation, image-text-recognition, image-dehazing, image-deraining, image-enhancemenmt, image-classification-logits, image-frame-interpolation, image-text-detection, image-denoising, image-others, video-classification, video-feature-extraction, video-generation, video-audio-text, video-text, audio-embedding, audio-event-classification, audio-command-detection, audio-paralinguists-classification, audio-speech-to-text, audio-speech-synthesis, audio-synthesis, audio-pitch-extraction}"
    else:
        raise ValueError(f"unsupported APIBench api_name {api_name!r}")

    prompt = question + "\nWrite a python program in 1 to 2 lines to call API in " + api_name + ".\n\nThe answer should follow the format: <<<domain>>> $DOMAIN, <<<api_call>>>: $API_CALL, <<<api_provider>>>: $API_PROVIDER, <<<explanation>>>: $EXPLANATION, <<<code>>>: $CODE}. Here are the requirements:\n" + domains + "\n2. The $API_CALL should have only 1 line of code that calls api.\n3. The $API_PROVIDER should be the programming framework used.\n4. $EXPLANATION should be a step-by-step explanation.\n5. The $CODE is the python code.\n6. Do not repeat the format in your answer."
    prompts.append({"role": "system", "content": "You are a helpful API writer who can write APIs based on requirements."})
    prompts.append({"role": "user", "content": prompt})
    return prompts


# --------------------------------------------------------------------------- #
# Scoring: gorilla/eval/eval-scripts/ast_eval_{hf,th,tf}.py. The three scripts share the sub-tree
# walk and the match rule and differ in which argument nodes of a database call must appear in
# the reply (get_args) and in torchhub's handling of a reply with no `api_call` field.
# --------------------------------------------------------------------------- #
def _tree_sitter_parser():
    """A tree-sitter Python parser. Upstream loads its own compiled grammar
    (codebleu/parser/my-languages.so, tree-sitter-python 0.20.2); gbench uses the
    tree-sitter-python 0.21.0 wheel, the release that matches the tree-sitter 0.21 runtime."""
    from tree_sitter import Language, Parser
    import tree_sitter_python
    parser = Parser()
    parser.set_language(Language(tree_sitter_python.language(), "python"))
    return parser


#: The tree-sitter releases the scorer is built for: _tree_sitter_parser uses the 0.21 runtime
#: API (Language(ptr, name), Parser.set_language; 0.22 removed both), and the grammar release
#: decides the parse trees the sub-tree match walks.
_TREE_SITTER_SERIES = "0.21."
_TREE_SITTER_PYTHON_VERSION = "0.21.0"
_PIP_HINT = "pip install 'tree-sitter==0.21.3' 'tree-sitter-python==0.21.0'"


def _dist_version(dist: str) -> Optional[str]:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_prereqs() -> None:
    """Hard-error before the rollout if the AST scorer cannot run as pinned."""
    missing = [m for m in ("tree_sitter", "tree_sitter_python") if importlib.util.find_spec(m) is None]
    if missing:
        raise infra_required(
            "gorilla_apibench",
            f"the APIBench AST scorer needs {', '.join(missing)} ({_PIP_HINT}).",
            DOCS_URL,
        )
    runtime, grammar = _dist_version("tree-sitter"), _dist_version("tree-sitter-python")
    if not (runtime or "").startswith(_TREE_SITTER_SERIES) or grammar != _TREE_SITTER_PYTHON_VERSION:
        raise infra_required(
            "gorilla_apibench",
            f"the APIBench AST scorer needs tree-sitter 0.21.x with tree-sitter-python "
            f"{_TREE_SITTER_PYTHON_VERSION}; installed: tree-sitter {runtime}, tree-sitter-python "
            f"{grammar} ({_PIP_HINT}).",
            DOCS_URL,
        )


def get_all_sub_trees(root_node) -> List[list]:
    """ast_eval_*.py get_all_sub_trees: [sexp, depth, node, first child text] per node with
    children (the sexp slot is unused by the match and left empty)."""
    node_stack = []
    sub_tree_sexp_list = []
    depth = 1
    node_stack.append([root_node, depth])
    while len(node_stack) != 0:
        cur_node, cur_depth = node_stack.pop()
        if cur_node.child_count > 0:
            sub_tree_sexp_list.append([None, cur_depth, cur_node, cur_node.children[0].text])
        else:
            sub_tree_sexp_list.append([None, cur_depth, cur_node, None])
        for child_node in cur_node.children:
            if len(child_node.children) != 0:
                depth = cur_depth + 1
                node_stack.append([child_node, depth])
    return sub_tree_sexp_list


def get_args(node, hub: str) -> List[bytes]:
    """ast_eval_*.py get_args: the argument texts of a database call that a reply must contain."""
    if node.child_count == 0:
        return []
    args_list = []
    for child in node.children[0].children[0].children[1].children:
        text = child.text.decode()
        if hub == "torchhub":
            # ast_eval_th.py keeps only these keyword values and no positional arguments.
            if "repo_or_dir" in text or "model" in text:
                args_list.append(child.children[2].text)
            continue
        if hub == "huggingface":
            keyword = "=" in text
        else:
            keyword = "model=" in text or "model =" in text
        if keyword:
            args_list.append(child.children[2].text)
        elif text != "(" and text != ")" and text != ",":
            args_list.append(child.text)
    return args_list


def ast_check(candidate_subtree_list: List[list], base_tree_list: List[Any],
              base_args: List[Any]) -> int:
    """ast_eval_*.py ast_check: index of the first database call whose callee appears in the reply
    and whose arguments all occur in that sub-tree's text, else -1. `base_args[idx]` caches
    get_args(base_tree_list[idx]) (a pure function of the database tree)."""
    for idx, base_tree in enumerate(base_tree_list):
        if base_tree.children[0].children[0].child_count == 0:
            continue
        api_name = base_tree.children[0].children[0].children[0].text
        for candidate_tree in candidate_subtree_list:
            if candidate_tree[3] == api_name:
                break
        # As upstream: with no callee match the loop variable is left on the last sub-tree.
        candidate_tree = candidate_tree[2]
        args_list = base_args[idx]
        if isinstance(args_list, Exception):
            raise args_list
        if len(args_list) == 0:
            continue
        ast_match = True
        for arg in args_list:
            if arg.decode().lstrip("'").rstrip("'") not in candidate_tree.text.decode():
                ast_match = False
                break
        if ast_match:
            return idx
    return -1


def extract_api_call(text: str, hub: str) -> Optional[str]:
    """ast_eval_*.py: the text between `api_call` + ':' and the last ')' before `api_provider`.
    torchhub scores a reply with no `api_call` field as neither correct nor hallucinated (None);
    the other hubs parse the whole reply."""
    output = text.split("api_call")
    if len(output) == 1:
        return None if hub == "torchhub" else output[0]
    output = output[1].split("api_provider")[0]
    start = 0 if ":" not in output else output.index(":")
    end = -2 if ")" not in output else output.rindex(")")
    return output[start + 2:end + 1]


class HubDatabase:
    """One hub's API database, parsed once."""

    def __init__(self, hub: str, api_rows: List[Dict[str, Any]], parser=None):
        self.hub = hub
        self.parser = parser or _tree_sitter_parser()
        self.api_rows = api_rows
        self.trees = [self.parse(r["api_call"]) for r in api_rows]
        self.args: List[Any] = []
        for tree in self.trees:
            try:
                self.args.append(get_args(tree, hub))
            except Exception as e:                           # noqa: BLE001 - re-raised on use
                self.args.append(e)

    def parse(self, code: str):
        return self.parser.parse(bytes(code, "utf8")).root_node

    def score(self, response_text: str, gold_domain: str) -> Tuple[bool, bool]:
        """(correct, hallucination) for one reply, as ast_eval_*.py counts it."""
        api_call = extract_api_call(response_text, self.hub)
        if api_call is None:
            return False, False
        idx = ast_check(get_all_sub_trees(self.parse(api_call)), self.trees, self.args)
        if idx == -1:
            return False, True
        return self.api_rows[idx]["domain"] == gold_domain, False


#: hub -> HubDatabase, filled by the loader (the per-sample scorer runs inside the rollout).
_DATABASES: Dict[str, HubDatabase] = {}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _cache_dir() -> str:
    root = gbench_cache_dir()
    return os.path.join(root, "gorilla_apibench", GORILLA_COMMIT)


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _questions_file(rel: str, blob: str) -> str:
    """One pinned question file (downloaded once, checked against its git blob id)."""
    path = os.path.join(_cache_dir(), "questions", rel)
    if os.path.isfile(path):
        with open(path, "rb") as f:
            if _git_blob_sha1(f.read()) == blob:
                return path
    url = _QUESTIONS_URL.format(commit=GORILLA_COMMIT, rel=rel)
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
    except Exception as e:
        raise infra_required("gorilla_apibench", f"the question file {url} could not be "
                             f"downloaded ({e}).", DOCS_URL) from e
    if _git_blob_sha1(data) != blob:
        raise infra_required("gorilla_apibench", f"{url} does not match the pinned file "
                             f"(git blob {blob}).", DOCS_URL)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_gorilla_apibench_samples(
    limit: Optional[int] = None,
    parser=None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Every APIBench question of the three hubs in all four settings (4 x 1775), with each hub's
    API database parsed for scoring."""
    from huggingface_hub import hf_hub_download
    rows: List[Dict[str, Any]] = []
    for hub, (api_name, _q_rel, _q_blob, eval_file, api_file) in HUBS.items():
        try:
            qa_pairs = [r["api_data"] for r in _read_jsonl(hf_hub_download(
                APIBENCH_REPO, eval_file, repo_type="dataset", revision=APIBENCH_REVISION))]
            api_rows = _read_jsonl(hf_hub_download(
                APIBENCH_REPO, api_file, repo_type="dataset", revision=APIBENCH_REVISION))
        except Exception as e:
            raise infra_required("gorilla_apibench", f"{APIBENCH_REPO}@{APIBENCH_REVISION} "
                                 f"({eval_file}, {api_file}) could not be downloaded ({e}).",
                                 DOCS_URL) from e
        if not api_rows:
            raise RuntimeError(f"gorilla_apibench: {api_file} holds no APIs")
        _DATABASES[hub] = HubDatabase(hub, api_rows, parser)
        _score.cache_clear()
        for setting in SETTINGS:
            questions = _read_jsonl(_questions_file(*question_file(hub, setting)))
            if len(questions) != HUB_SIZES[hub]:
                raise RuntimeError(f"gorilla_apibench: unexpected {hub} {setting} data "
                                   f"({len(questions)} questions)")
            for q in questions:
                # ast_eval_*.py: qa_pairs[question_id - 1] is the question's reference entry.
                gold_domain = qa_pairs[q["question_id"] - 1]["domain"]
                rows.append({"hub": hub, "setting": setting, "api_name": api_name,
                             "question_id": q["question_id"], "text": q["text"],
                             "domain": gold_domain})

    # Stratified by hub and setting, not a contiguous head.
    rows = stratified_sample(rows, limit, lambda r: f"{r['hub']}/{r['setting']}",
                             seed="gorilla_apibench")

    samples = []
    for r in rows:
        gold = json.dumps({"hub": r["hub"], "setting": r["setting"],
                           "question_id": r["question_id"], "domain": r["domain"]})
        samples.append((encode_question(r["text"], r["api_name"]), gold,
                        {"category": f"{r['hub']}/{r['setting']}"}))
    logger.info(f"Loaded {len(samples)} gorilla_apibench samples.")
    return samples


@functools.lru_cache(maxsize=None)
def _score(response_text: str, gold_target: str) -> Tuple[bool, bool]:
    """(correct, hallucination); cached because the run scores each reply twice (per-sample
    verdict during the rollout, per-hub tallies after it)."""
    g = json.loads(gold_target)
    return _DATABASES[g["hub"]].score(response_text or "", g["domain"])


def _hub_numbers(c: Dict[str, int]) -> Dict[str, Any]:
    acc = c["correct"] / c["n"] * 100.0
    hal = c["hallucination"] / c["n"] * 100.0
    return {"n": c["n"], "accuracy": round(acc, 2), "hallucination": round(hal, 2),
            "error": round(100.0 - acc - hal, 2)}


def _eval_gorilla_apibench(response_text: str, gold_target: str) -> bool:
    """Per-sample verdict = ast_eval's functionality check for this reply."""
    if not response_text:
        return False
    return _score(response_text, gold_target)[0]


def run_gorilla_apibench(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute gorilla_apibench: per-hub functionality accuracy and hallucination rate as
    Gorilla's ast_eval scripts report them, in each of the four settings; the headline is the
    0-shot setting's unweighted mean over the hubs."""
    check_prereqs()
    samples = _load_gorilla_apibench_samples(limit=kwargs.get("limit"))
    limit_plan = base_suite_plan("gorilla_apibench", samples, kwargs.get("limit"),
                                 run_attempts(kwargs))
    result = run_eval_suite(
        eval_name="gorilla_apibench",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_gorilla_apibench,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
        official_sampling=GORILLA_SAMPLING,
    )
    result["eval_limit_plan"] = limit_plan
    tally = {setting: {hub: {"n": 0, "correct": 0, "hallucination": 0} for hub in HUBS}
             for setting in SETTINGS}
    for t in scorable_traces(result):
        g = json.loads(t["gold_answer"])
        hub, setting = g["hub"], g.get("setting", HEADLINE_SETTING)
        # The same reasoning-stripped text the per-sample verdict saw.
        correct, halluc = _score(strip_thinking_tags(t.get("response_text") or ""), t["gold_answer"])
        t["apibench_hallucination"] = halluc
        tally[setting][hub]["n"] += 1
        tally[setting][hub]["correct"] += int(correct)
        tally[setting][hub]["hallucination"] += int(halluc)
    per_setting: Dict[str, Dict[str, Any]] = {}
    for setting, hubs in tally.items():
        numbers = {hub: _hub_numbers(c) for hub, c in hubs.items() if c["n"]}
        if not numbers:
            continue
        per_setting[setting] = {
            "per_hub": numbers,
            "mean_accuracy": round(sum(h["accuracy"] for h in numbers.values()) / len(numbers), 2),
            "mean_hallucination": round(sum(h["hallucination"] for h in numbers.values())
                                        / len(numbers), 2),
        }
    per_hub = (per_setting.get(HEADLINE_SETTING) or {}).get("per_hub", {})
    result["per_hub"] = per_hub
    result["per_setting"] = per_setting
    # the 0-shot questions the headline rests on, and the hubs they came from
    result["headline_n"] = sum(int(h.get("n") or 0) for h in per_hub.values())
    n_hubs = len(HUB_SIZES)
    hubs_used = ", ".join(f"{hub}, n={h.get('n')}" for hub, h in sorted(per_hub.items()))
    if per_hub:
        result["accuracy"] = per_setting[HEADLINE_SETTING]["mean_accuracy"]
        result["hallucination"] = per_setting[HEADLINE_SETTING]["mean_hallucination"]
        result.pop("accuracy_withheld_reason", None)
    elif result.get("accuracy") is not None or scorable_traces(result):
        # No 0-shot question in the draw: base's pooled accuracy would be the retriever
        # settings' number under the 0-shot headline's name, so the headline is withheld.
        result["accuracy"] = None
        result["accuracy_withheld_reason"] = (
            "no 0-shot question was scored (the --eval-limit draw holds none, or none was "
            "answered); the headline is the 0-shot setting's hub mean, and per_setting holds the "
            "retriever settings' numbers")
    scope = (f"mean over the {n_hubs} hubs" if len(per_hub) == n_hubs
             else f"mean over {len(per_hub)} of {n_hubs} hubs ({hubs_used})" if per_hub
             else "withheld: no 0-shot question scored")
    result["metric"] = (f"0-shot setting: {scope} of APIBench functionality accuracy "
                        "(ast_eval 'Final Functionality accuracy'); per_hub accuracy / hallucination "
                        "/ error are the per-hub 0-shot numbers Gorilla reports, per_setting the same "
                        "for 0-shot and the BM25, GPT-Index and Oracle retriever settings")
    result["dataset_revision"] = (f"{APIBENCH_REPO}@{APIBENCH_REVISION}; questions and scorer "
                                  f"ShishirPatil/gorilla@{GORILLA_COMMIT}")
    reasons = ["Gorilla reports each hub separately; the headline is gbench's unweighted mean of "
               "the three 0-shot hub accuracies (per_hub and per_setting hold the comparable "
               "numbers)"]
    if per_hub and len(per_hub) < n_hubs:
        reasons.append(f"the headline covers {len(per_hub)} of the {n_hubs} hubs ({hubs_used}; "
                       f"headline_n {result['headline_n']})")
    if kwargs.get("limit"):
        reasons.append(f"--eval-limit {kwargs.get('limit')} (APIBench scores all "
                       f"{sum(HUB_SIZES.values())} questions in each of the {len(SETTINGS)} "
                       f"settings)")
    if parse_shard(os.environ.get("GBENCH_SHARD")):
        reasons.append("sharded run (a subset of the questions)")
    if (result.get("official_sampling") or {}).get("temperature_matches_official") is False:
        reasons.append(f"temperature {result.get('temperature')} (upstream samples at 0)")
    if result.get("decoding_penalties"):
        reasons.append("decoding penalties set")
    if ((result.get("attempts") or {}).get("attempts_per_sample") or 1) > 1:
        reasons.append("--attempt-count > 1 (counts pool every attempt)")
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = "; ".join(reasons)
    return result
