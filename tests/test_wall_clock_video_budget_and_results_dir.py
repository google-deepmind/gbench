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

"""Base-runner properties, test isolation, docs and provenance in step.

  * a sample stopped by its per-sample wall clock scores 0 and stays in the denominator, whatever
    the batch scorer writes into its trace;
  * OfficialSampling.params carries sampling fields only;
  * a `video_url` part is a fixed soft-token budget, never counted as text or sent to /tokenize;
  * tests start without GBENCH_RESULTS_DIR and write run directories under a temporary root;
  * README, env-knobs, toolchains, RESOURCES, THIRD_PARTY_DATASETS, NOTICE, pyproject, uv.lock
    and canonical_sync describe what the code does.

No model server, judge API or dataset hub is contacted.
"""

import ast
import asyncio
import json
import os
import re
import tomllib
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import base

REPO = Path(__file__).resolve().parents[1]
SUITE_DIR = REPO / "gbench" / "runners" / "eval_suites"
DOCS = REPO / "docs"


# The README was split: these docs hold sections moved verbatim out of the README, so a
# negative check that used to scan the whole README scans README.md plus all of them.
_README_SPLIT_DOCS = (
    "docs/installation.md", "docs/environment-variables.md", "docs/performance.md",
    "docs/eval-suites.md", "docs/golden-set.md", "docs/remote-endpoints.md",
    "docs/results.md", "docs/deployment.md", "docs/cli-reference.md",
)
_README_AND_SPLIT_DOCS = ("README.md", *_README_SPLIT_DOCS)


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _run(send, samples, **kw):
    with mock.patch.object(base, "_send_single_request", send):
        return base.run_eval_suite(eval_name="aime", model_name="m", base_url="http://x",
                                   concurrency=kw.pop("concurrency", 1), samples=samples, **kw)


def _timed_send(durations):
    async def _send(**k):
        async with k["semaphore"]:
            await asyncio.sleep(durations[k["messages"][-1]["content"]])
        return base.Reply("x", None, "stop", completion_tokens=1)
    return _send


def _slow_and_fast():
    return [([{"role": "user", "content": "slow"}], "x", {"category": "a"}),
            ([{"role": "user", "content": "fast"}], "x", {"category": "a"})]


# --------------------------------------------------------------------------- #
# per-sample wall clock: 0, whatever the batch scorer writes
# --------------------------------------------------------------------------- #
def test_a_scorer_that_rewrites_the_status_cannot_score_a_wall_clock_stopped_sample():
    proto = base.ToolLoopProtocol(reference="t", max_sample_wall_s=0.2)

    async def rewrites_status(traces):
        for t in traces:
            t["is_correct"] = not t.get("tool_calls")
            t["status"] = "CORRECT" if t["is_correct"] else "INCORRECT"

    res = _run(_timed_send({"slow": 3.0, "fast": 0.0}), _slow_and_fast(), concurrency=2,
               async_eval_fn=rewrites_status, tool_loop=proto)
    by_text = {t["messages"][0]["content"]: t for t in res["sample_traces"]}
    assert by_text["slow"]["is_correct"] is False
    assert by_text["slow"]["status"] == "WALL_CLOCK_LIMIT"
    assert by_text["fast"]["is_correct"] is True
    assert res["correct_answers"] == 1 and res["wall_clock_limited"] == 1
    assert res["total_questions"] == 2
    assert res["category_accuracy"]["a"]["total"] == 2


@pytest.mark.parametrize("marker", [{"judge_grade": "JUDGE_OUTAGE"}, {"scoring_excluded": True},
                                    {"empty_is_answer": True, "is_correct": True}])
def test_a_scorer_cannot_take_a_wall_clock_stopped_sample_out_of_the_denominator(marker):
    proto = base.ToolLoopProtocol(reference="t", max_sample_wall_s=0.2)

    async def excludes(traces):
        for t in traces:
            t["is_correct"] = t["messages"][0]["content"] == "fast"
            if t["messages"][0]["content"] == "slow":
                t.update(marker)

    res = _run(_timed_send({"slow": 3.0, "fast": 0.0}), _slow_and_fast(), concurrency=2,
               async_eval_fn=excludes, tool_loop=proto)
    slow = next(t for t in res["sample_traces"] if t["messages"][0]["content"] == "slow")
    assert slow["is_correct"] is False and slow["status"] == "WALL_CLOCK_LIMIT"
    assert "scoring_excluded" not in slow and "empty_is_answer" not in slow
    assert slow.get("judge_grade") is None
    assert slow.get("scorer_judge_grade") == marker.get("judge_grade")
    assert res["correct_answers"] == 1 and res["accuracy"] == pytest.approx(50.0)
    assert res["judge_outages"] == 0 and res["scoring_excluded"] == 0


def test_samples_that_finish_in_time_keep_the_scorers_verdict_and_markers():
    proto = base.ToolLoopProtocol(reference="t", max_sample_wall_s=5.0)

    async def scorer(traces):
        for t in traces:
            t["is_correct"] = False
            t["judge_grade"] = "JUDGE_OUTAGE"

    res = _run(_timed_send({"slow": 0.0, "fast": 0.0}), _slow_and_fast(), concurrency=2,
               async_eval_fn=scorer, tool_loop=proto)
    assert res["wall_clock_limited"] == 0 and res["judge_outages"] == 2
    assert all(t["judge_grade"] == "JUDGE_OUTAGE" for t in res["sample_traces"])


# --------------------------------------------------------------------------- #
# OfficialSampling.params: sampling fields only
# --------------------------------------------------------------------------- #
def _recorder():
    sent = []

    async def _send(**k):
        sent.append(k)
        return base.Reply("x", None, "stop", completion_tokens=1)
    return sent, _send


@pytest.mark.parametrize("params", [{"max_tokens": 1}, {"tools": []}, {"n": 2},
                                    {"chat_template_kwargs": {}}, {"stream": True},
                                    {"continue_final_message": True}, {"temperature": 0.3}])
def test_non_sampling_request_fields_are_not_accepted_as_official_sampling(params):
    sent, send = _recorder()
    with pytest.raises(ValueError, match="official sampling params"):
        _run(send, [([{"role": "user", "content": "q"}], "x", {})], eval_fn=lambda r, g: True,
             official_sampling=base.OfficialSampling(reference="r", params=params))
    assert not sent


def test_every_sampling_field_is_accepted_and_sendable():
    assert base.OFFICIAL_SAMPLING_PARAM_KEYS <= base.ALLOWED_API_KEYS
    assert "temperature" not in base.OFFICIAL_SAMPLING_PARAM_KEYS
    sent, send = _recorder()
    params = {"top_p": 0.9, "top_k": 20, "min_p": 0.05, "seed": 1, "stop": ["\n"],
              "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0}
    res = _run(send, [([{"role": "user", "content": "q"}], "x", {})], eval_fn=lambda r, g: True,
               official_sampling=base.OfficialSampling(reference="r", temperature=0.0,
                                                       params=params))
    assert res["official_sampling"]["params"] == params
    assert all(sent[0]["extra_payload"][k] == v for k, v in params.items())


def test_every_suite_declares_only_sampling_fields():
    bad = {}
    for path in SUITE_DIR.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call)
                    and getattr(node.func, "id", getattr(node.func, "attr", None))
                    == "OfficialSampling"):
                continue
            for kw in node.keywords:
                if kw.arg == "params" and isinstance(kw.value, ast.Dict):
                    keys = {ast.literal_eval(k) for k in kw.value.keys}
                    if keys - base.OFFICIAL_SAMPLING_PARAM_KEYS:
                        bad[path.stem] = sorted(keys - base.OFFICIAL_SAMPLING_PARAM_KEYS)
    assert not bad, bad


# --------------------------------------------------------------------------- #
# video parts: a fixed soft-token budget
# --------------------------------------------------------------------------- #
def _video_msgs(text="What happens in the clip?", nbytes=4_000_000):
    return [{"role": "user", "content": [
        {"type": "text", "text": text},
        {"type": "video_url", "video_url": {"url": "data:video/quicktime;base64," + "A" * nbytes}},
    ]}]


def test_the_video_budget_is_gemma4s_per_video_maximum():
    assert base.VIDEO_SOFT_TOKENS == 32 * (70 + 2 + 6) == 2496
    assert base._media_soft_tokens(0, 0) == 0
    assert base._media_soft_tokens(1, 1, 2) == (base.IMAGE_SOFT_TOKENS + base.AUDIO_SOFT_TOKENS
                                                + 2 * base.VIDEO_SOFT_TOKENS)


def test_a_video_part_is_not_counted_as_text(monkeypatch):
    monkeypatch.setattr(base, "_count_tokens", lambda text: None)   # character heuristics only
    msgs = _video_msgs()
    text, images, n_msgs, audios = base._prompt_text_and_images(msgs)
    assert "base64" not in text and "video_url" not in text
    assert (images, n_msgs, audios) == (0, 1, 0)
    clean, images, audios = base._text_only_messages(msgs)
    assert clean[0]["content"] == [{"type": "text", "text": "What happens in the clip?"}]
    assert base._video_parts(msgs) == 1 and not base._is_text_only(msgs)
    low, high, mode = base._prompt_token_bounds(msgs)
    assert mode == "estimate"
    assert base.VIDEO_SOFT_TOKENS <= low <= high < base.VIDEO_SOFT_TOKENS + 200


def test_a_video_is_not_uploaded_to_tokenize(monkeypatch):
    bodies = []

    class _Resp:
        def __init__(self, body):
            self._body = body

        def read(self):
            return json.dumps({"count": 40}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout=None):
        bodies.append(json.loads(req.data))
        return _Resp(req.data)

    base.set_run_knobs(tokenize_url="http://tok.invalid", served_model="m")
    monkeypatch.setattr(base.urllib.request, "urlopen", fake_urlopen)
    assert base._server_prompt_tokens(_video_msgs()) == 40 + base.VIDEO_SOFT_TOKENS
    assert "video_url" not in json.dumps(bodies[0]) and len(json.dumps(bodies[0])) < 1000


def test_a_long_video_prompt_is_not_refused_as_over_context(monkeypatch):
    monkeypatch.setattr(base, "_count_tokens", lambda text: None)
    max_tokens, reason = base.clamp_to_context(_video_msgs(nbytes=30_000_000), 4096,
                                               max_model_len=131072)
    assert reason is None and max_tokens == 4096


# --------------------------------------------------------------------------- #
# test isolation: no run artifacts outside temporary directories
# --------------------------------------------------------------------------- #
def test_tests_start_without_an_inherited_results_dir():
    assert "GBENCH_RESULTS_DIR" not in os.environ


def test_a_default_results_dir_lands_under_a_temporary_root(_isolate_results_dir):
    from gbench.utils.log_manager import LogManager
    lm = LogManager()
    assert str(lm.results_dir).startswith(str(_isolate_results_dir))
    assert not str(lm.results_dir).startswith(str(REPO / "results"))
    # LogManager exports the run directory; the fixture removes it again after the test.
    assert os.environ["GBENCH_RESULTS_DIR"] == str(lm.results_dir)


def test_an_absolute_results_dir_is_kept(tmp_path):
    from gbench.utils.log_manager import LogManager
    lm = LogManager(tmp_path / "runs")
    assert str(lm.results_dir).startswith(str(tmp_path / "runs"))


# --------------------------------------------------------------------------- #
# docs, packaging and provenance in step with the code
# --------------------------------------------------------------------------- #
def test_removed_opt_in_gates_are_not_documented():
    for rel in (*_README_AND_SPLIT_DOCS, "docs/evals/env-knobs.md", "docs/evals/toolchains.md"):
        text = _read(rel)
        assert "SWE_BENCH_PRO_RUN" not in text, rel
        assert "TAU2_ENV_RUN=1" not in text and "`GBENCH_TAU2_ENV_RUN` (legacy" not in text, rel
    toolchains = _read("docs/evals/toolchains.md")
    for gone in ("python3-pysrt", "pip install rank-bm25", "audioop-lts",
                 "pip install -e ./tau2-bench"):
        assert gone not in toolchains, gone
    assert "docker build -t gbench-tau2:b7ea9074 -f docker/tau2.Dockerfile docker/tau2_srt" \
        in toolchains


def test_the_tau_image_knob_is_documented_as_the_code_defines_it():
    from gbench.runners.eval_suites import tau_common
    assert tau_common.TAU2_IMAGE_DEFAULT == "gbench-tau2:b7ea9074"
    # The README "Environment variables" section moved to docs/environment-variables.md.
    for rel in ("docs/environment-variables.md", "docs/evals/env-knobs.md"):
        text = _read(rel)
        assert "GBENCH_TAU2_IMAGE" in text and tau_common.TAU2_IMAGE_DEFAULT in text, rel


def test_the_swe_image_knobs_are_in_the_readme_env_table():
    from gbench.runners.eval_suites import swe_agentic, swebench_common
    # The README env-var table moved to docs/environment-variables.md.
    readme = _read("docs/environment-variables.md")
    for env, default in ((swe_agentic.AGENT_V2_IMAGE_ENV, swe_agentic.AGENT_V2_IMAGE_DEFAULT),
                         (swebench_common.SWEBENCH_V5_IMAGE_ENV,
                          swebench_common.SWEBENCH_V5_IMAGE_DEFAULT)):
        assert f"`{env}`" in readme and f"`{default}`" in readme


def test_the_bfcl_eval_pin_is_the_release_the_suites_accept():
    from gbench.runners.eval_suites import bfcl_v4_agentic
    release = bfcl_v4_agentic.BFCL_EVAL_RELEASE
    pyproject = tomllib.loads(_read("pyproject.toml"))
    # The suites run the release in the local harness image; the host copy is for unit tests only.
    extras = pyproject["project"]["optional-dependencies"]
    assert f"bfcl-eval=={release}" in extras["dev"]
    assert not any(d.startswith("bfcl-eval") for d in extras["evals"])
    lock = _read("uv.lock")
    block = lock[lock.index('name = "bfcl-eval"'):][:200]
    assert f'version = "{release}"' in block
    for rel in ("docs/RESOURCES.md", "docs/evals/toolchains.md"):
        text = _read(rel)
        assert f"bfcl-eval=={release}" in text and "bfcl-eval==2025.12.17.1" not in text, rel


def test_the_readme_lists_every_deregistered_suite():
    # The Tier 1 suite catalog moved from the README to docs/eval-suites.md.
    readme = _read("docs/eval-suites.md")
    para = readme[readme.index("suites are **deregistered and"):][:900]
    assert para.startswith("suites are") and "Two suites are **deregistered" in readme
    for name in base.ROADMAP_SUITES:
        assert f"`{name}`" in para, name
    assert "SWELANCER_RUN" not in readme
    for rel in _README_AND_SPLIT_DOCS:   # README.md and every doc split out of it
        assert "SWELANCER_RUN" not in _read(rel), rel


def test_every_official_sampling_suite_is_listed_in_env_knobs():
    knobs = _read("docs/evals/env-knobs.md")
    section = knobs[knobs.index("- Suites that declare official sampling"):
                    knobs.index("- Renamed suites still read")]
    shared = {"fc_common": ("bfcl", "bfcl_live", "bfcl_v4_agentic"),
              "matharena_common": ("hmmt", "aime"),
              "gpqa": ("gpqa", "gpqa_diamond", "multilingual_mmlu", "simpleqa", "healthbench",
                       "browsecomp"),
              "vqa_common": ("chartqa", "docvqa", "infographicvqa", "textvqa", "coco_caption")}
    missing = []
    for path in SUITE_DIR.glob("*.py"):
        src = path.read_text(encoding="utf-8")
        if "OfficialSampling(" not in src or path.stem == "base":
            continue
        for name in shared.get(path.stem, (path.stem,)):
            if not re.search(rf"\b{re.escape(name)}\b", section):
                missing.append(name)
    assert not missing, missing


def test_new_knobs_are_documented():
    knobs = _read("docs/evals/env-knobs.md")
    for knob in ("GBENCH_VIDEO_SOFT_TOKENS", "GBENCH_GAIA_SPLIT", "GBENCH_MCP_ATLAS_MONGO_IMAGE",
                 "GBENCH_TAU2_IMAGE", "cimemories_release/", "arc_agi/<commit>/evaluation/"):
        assert knob in knobs, knob
    assert "GBENCH_REPETITION_PENALTY` now\n  reaches the server" in knobs \
        or "GBENCH_REPETITION_PENALTY` now reaches the server" in knobs.replace("\n  ", " ")


def test_notice_lists_the_code_vendored_from_upstream():
    notice = _read("NOTICE")
    for needle in ("gbench/third_party/eval_suites/omnidoc_metrics.py", "ibm-aur-nlp/PubTabNet",
                   "gbench/third_party/eval_suites/causalbench.py", "causalNLP/corr2cause",
                   "zhijing-jin/efficiency", "evaluation/utils_execute.py (vendored verbatim",
                   "test.py split_special_tokens", "eval_template.pairwise.v2.md",
                   "lm-evaluation-harness\n    at commit d6de8164",
                   "mongo_dump_video_game_store-UNZIP-FIRST.zip", "fchollet/ARC-AGI"):
        assert needle in notice, needle
    assert "at commit 740312ad" not in notice
    assert not [c for c in notice if ord(c) > 127]


def test_third_party_datasets_follow_the_loaders():
    text = _read("docs/THIRD_PARTY_DATASETS.md")
    for needle in ("`lovesnowbest/T-Eval`", "`liminghao1630/API-Bank`",
                   "`allenai/WildBench-V2-Model-Outputs`", "GitHub `fchollet/ARC-AGI`",
                   "GBENCH_GAIA_SPLIT=test", "`tau2` / `tau3` (tau2-bench"):
        assert needle in text, needle
    assert "dataartist/arc-agi" not in text
    assert "dataartist/arc-agi" not in _read("docs/RESOURCES.md")


def test_canonical_sync_follows_the_new_protocols():
    from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC as CS
    assert "gbench-tau2 image" in CS["tau2"]["method"]
    assert "gbench-tau2 image" in CS["tau3"]["method"]
    assert "Gemini judge cascade" in CS["putnam_formal"]["method"]
    assert "no opt-in variable" in CS["swe_bench_pro"]["method"]
    assert "make_run_report" in CS["swe_bench_live"]["method"] and "--rewrite_reports" not in json.dumps(CS)
    assert "2026.3.23" in CS["bfcl"]["upstream"] and "2026.3.23" in CS["bfcl_v4_agentic"]["upstream"]
    assert "fchollet/ARC-AGI" in CS["arc_agi"]["dataset"]
    assert "WildBench-V2-Model-Outputs" in CS["wildbench"]["dataset"]
    for name, entry in CS.items():
        text = json.dumps(entry)
        assert not [c for c in text if ord(c) > 127], name
        assert not re.search(r"\b[Ww]ave ?\d|\bW2[a-e]?\b", text), name


def test_the_renamed_putnam_tests_are_the_ones_the_doc_cites():
    doc = _read("docs/evals/putnam_formal.md")
    for name in ("test_putnambench_release.py", "test_putnam_formal_review.py"):
        assert (REPO / "tests" / name).exists() and f"tests/{name}" in doc


def test_new_text_in_the_shared_docs_is_ascii():
    for rel in ("docs/evals/env-knobs.md", "docs/evals/toolchains.md",
                "docs/THIRD_PARTY_DATASETS.md", "docs/evals/large-sweep-setup.md"):
        text = _read(rel)
        assert not [c for c in text if ord(c) > 127], rel


def test_the_cache_roots_added_since_the_last_check_follow_gbench_cache_dir(monkeypatch, tmp_path):
    from gbench.runners.eval_suites import arc_agi, bfcl_v4_agentic, cimemories, swebench_common
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))
    assert arc_agi._cache_dir().startswith(str(tmp_path / "arc_agi"))
    assert cimemories._cache_root() == tmp_path
    assert swebench_common._snapshot_root() == str(tmp_path / "swebench_snapshots")
    code = bfcl_v4_agentic._CHILD_BOOTSTRAP.format(model="m", args=["evaluate"])
    assert 'os.environ["BFCL_PROJECT_ROOT"]' in code and ".cache" not in code
