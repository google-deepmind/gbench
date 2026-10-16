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

"""Resource limits and endpoint failures in the code-execution suites.

* aider_polyglot: the benchmark container has upstream's memory limit (benchmark/docker.sh:
  --memory=12g --memory-swap=12g), recorded on the result; a container killed with SIGKILL is
  named as such in the hard error.
* A request that got no reply from the served endpoint is a harness error, never the model's
  outcome, in every base-run code suite: ojbench names it apart from a missing judge verdict and
  withholds the C++ headline when no C++ generation was scored; bigcodebench withholds the
  Instruct headline when every Instruct request failed, withholds the harness's own pass@1 of a
  split with such a task and names the failures; omnidocbench leaves such a page out of the
  evaluator's input (it would score it as an empty transcription); multipl_e names them as left
  out of the score.

Docker, the endpoint and the graders are stubbed; nothing is contacted.
"""

import asyncio
import json
import os
import re
import types
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import aider_polyglot as A
from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import bigcodebench as B
from gbench.runners.eval_suites import multipl_e as M
from gbench.runners.eval_suites import ojbench as OJ
from gbench.runners.eval_suites import omnidocbench as O
from gbench.runners.eval_suites import tau_common as TC

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs" / "evals"


def _flat(text):
    return " ".join(text.split())


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in ("GBENCH_AIDER_LANGUAGES", "GBENCH_AIDER_MODEL", "GBENCH_AIDER_EDIT_FORMAT",
                 "GBENCH_AIDER_EXTRA_ARGS", "GBENCH_OJBENCH_TEMPERATURE",
                 "GBENCH_BIGCODEBENCH_SUBSET", "GBENCH_BIGCODEBENCH_TEMPERATURE",
                 "GBENCH_OMNIDOCBENCH_TEMPERATURE", "GBENCH_SHARD"):
        monkeypatch.delenv(name, raising=False)


def _serve(lost):
    """A stub for base's request call: no reply (after base's retries) for a prompt in `lost`."""
    def _text(content):
        return content if isinstance(content, str) else " ".join(
            p.get("text", "") for p in content or [] if isinstance(p, dict))

    async def _send(**kw):
        # every turn (bigcodebench's no-think path ends with a prefilled assistant turn)
        text = " ".join(_text(m.get("content")) for m in kw["messages"])
        if any(key in text for key in lost):
            return base.Reply(text=None, tool_calls=None, error="HTTP 500: server error")
        return base.Reply(text="```cpp\nint main(){}\n```", tool_calls=None,
                          finish_reason="stop", completion_tokens=5)
    return _send


# --------------------------------------------------------------------------- #
# aider_polyglot: upstream's memory limit                                      #
# --------------------------------------------------------------------------- #
def test_the_aider_container_has_upstreams_memory_limit():
    cmd = A._build_command("aider-benchmark:5dc9490b", "/b", "gbench", "openai/m",
                           "http://host.docker.internal:8000/v1", 4, A._LANGUAGES, "", None, "",
                           settings_dir="/s")
    # the network arguments stay where they were, the limit follows them, before the image
    assert cmd[3:7] == list(TC.ENDPOINT_NETWORK_ARGS)
    assert cmd[7:11] == ["--memory", "12g", "--memory-swap", "12g"]
    assert cmd.index("--memory-swap") < cmd.index("aider-benchmark:5dc9490b")
    assert A.CONTAINER_MEMORY_ARGS == ("--memory", "12g", "--memory-swap", "12g")


def test_the_limit_is_the_one_in_upstreams_docker_sh():
    # benchmark/docker.sh as upstream ships it (Aider-AI/aider); the prereqs checkout, when this
    # host has one, must say the same.
    upstream = "--memory=12g \\\n       --memory-swap=12g \\"
    flags = dict(re.findall(r"--(memory(?:-swap)?)=(\S+)", upstream))
    assert flags == {"memory": A.CONTAINER_MEMORY, "memory-swap": A.CONTAINER_MEMORY}
    checkout = Path(os.environ.get("GBENCH_PREREQS_DIR", REPO.parent / "gbench-prereqs"))
    docker_sh = checkout / "aider" / "benchmark" / "docker.sh"
    if docker_sh.is_file():
        found = dict(re.findall(r"--(memory(?:-swap)?)=(\S+)", docker_sh.read_text()))
        assert found == flags


def _aider_run(monkeypatch, tmp_path, returncode=0, exercises=None):
    langs = A._LANGUAGES
    n = A._EXPECTED_FULL if exercises is None else exercises

    def fake_run(cmd, **_kw):
        fake_run.cmd = cmd
        root = tmp_path / "2026-run1--gbench"
        for i in range(n):
            d = root / langs[i % len(langs)] / "exercises" / "practice" / f"x{i}"
            d.mkdir(parents=True, exist_ok=True)
            (d / ".aider.results.json").write_text(json.dumps(
                {"language": langs[i % len(langs)], "testcase": f"x{i}",
                 "tests_outcomes": [True]}))
            (d / ".aider.chat.history.md").write_text("")
        report = {"files_loaded": ["/gbench-aider/model-settings.yml"], "extra_params": {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "timeout": A.model_request_timeout_s(), "max_retries": 0,
            "temperature": 0.0}}  # a no-think run's temperature (base.resolve_temperature)
        return types.SimpleNamespace(returncode=returncode, stderr="",
                                     stdout=A._CHECK_MARKER + json.dumps(report) + "\n")
    monkeypatch.setenv("GBENCH_AIDER_BENCHMARK_DIR", str(tmp_path))
    monkeypatch.setenv("GBENCH_AIDER_MODEL", "openai/m")
    for i in range(n):      # the exercises checkout a limited run selects from
        (tmp_path / "polyglot-benchmark" / langs[i % len(langs)] / "exercises" / "practice"
         / f"x{i}").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(A, "_check_prereqs", lambda image, bench: None)
    monkeypatch.setattr(TC, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC.container_url(url))
    monkeypatch.setattr(A.subprocess, "run", fake_run)
    return fake_run


def test_the_result_records_the_limit_and_the_threads_that_shared_it(monkeypatch, tmp_path):
    fake = _aider_run(monkeypatch, tmp_path)
    r = A.run_aider_polyglot("m", "http://127.0.0.1:8000/v1", concurrency=6)
    assert r["status"] == "success"
    # this fake container prints no enforced limit: recorded as unreadable
    assert r["container_memory_limit"] == {"memory": "12g", "memory_swap": "12g", "threads": 6,
                                           "enforced_bytes": None}
    assert "--threads 6" in fake.cmd[-1]
    assert fake.cmd[fake.cmd.index("--memory") + 1] == "12g"


@pytest.mark.parametrize("limit", [None, 5])
def test_a_container_killed_with_sigkill_is_named(monkeypatch, tmp_path, limit):
    _aider_run(monkeypatch, tmp_path, returncode=137, exercises=10)
    with pytest.raises(RuntimeError) as ei:
        A.run_aider_polyglot("m", "http://127.0.0.1:8000/v1", concurrency=1, limit=limit)
    msg = str(ei.value)
    assert "137: its main process was killed with SIGKILL" in msg
    assert "12g memory limit (upstream's benchmark/docker.sh limit)" in msg
    assert A._exit_hint(1) == "" and A._exit_hint(0) == ""


def test_the_aider_doc_states_the_limit():
    doc = _flat((DOCS / "aider_polyglot.md").read_text(encoding="utf-8"))
    assert "`docker run --memory 12g --memory-swap 12g`" in doc
    assert "`--memory=12g --memory-swap=12g` of `benchmark/docker.sh`" in doc
    assert "`container_memory_limit`" in doc
    src = (REPO / "gbench" / "runners" / "eval_suites" / "aider_polyglot.py").read_text()
    assert "--memory=12g --memory-swap=12g" in src


# --------------------------------------------------------------------------- #
# ojbench: endpoint failures are named apart from missing judge verdicts       #
# --------------------------------------------------------------------------- #
def _oj(monkeypatch, problems, no_verdict=()):
    """problems: (id, language) pairs; the judge returns no verdict for an id in `no_verdict`."""
    monkeypatch.setenv("GBENCH_OJBENCH_SAMPLES", "1")
    monkeypatch.setattr(OJ, "check_ojbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(OJ, "served_model_recommended_sampling",
                        lambda model_name=None: (dict(OJ.GEMMA4_RECOMMENDED), "stub"))
    monkeypatch.setattr(OJ, "_load_ojbench_samples", lambda limit=None: [
        ([{"role": "user", "content": f"solve p{pid} in {lang}"}], pid,
         {"category": lang, "row": {"id": pid, "dataset": "NOI", "language": lang,
                                    "difficulty": "easy"}}) for pid, lang in problems])

    def judge(records, workers):
        return [dict(r) if r["id"] in no_verdict else {**r, "is_passed": True} for r in records]
    monkeypatch.setattr(OJ, "_judge_in_container", judge)


def _run_oj(lost):
    with mock.patch.object(base, "_send_single_request", side_effect=_serve(lost)):
        return OJ.run_ojbench("m", "http://x/v1", 2)


def test_ojbench_names_endpoint_failures_apart_from_missing_verdicts(monkeypatch):
    _oj(monkeypatch, [(1, "cpp"), (2, "cpp"), (3, "python"), (4, "python")], no_verdict={4})
    r = _run_oj(lost=["p2 "])
    assert r["status"] == "completed_with_errors"
    assert r["scoring_excluded"] == 2                      # base: every left-out generation
    assert r["judge_no_verdict"] == 1
    assert r["request_harness_errors"] == {"endpoint_failure": 1}
    reason = r["leaderboard_comparable_reason"]
    assert "1 generation(s) got no judge verdict and were excluded" in reason
    assert ("1 generation request(s) got no reply from the served endpoint (harness errors, "
            "left out of the score; request_harness_errors)") in reason
    assert "2 generation(s) got no judge verdict" not in reason
    # the headline is C++ Pass@1 over the one C++ generation that was scored
    assert r["accuracy"] == 100.0 and r["total_questions"] == 1
    assert r["per_language"]["cpp"]["generations"] == 1
    assert r["per_language"]["python"]["generations"] == 1


def test_ojbench_withholds_the_cpp_headline_when_no_cpp_generation_was_scored(monkeypatch):
    _oj(monkeypatch, [(1, "cpp"), (2, "cpp"), (3, "python")])
    r = _run_oj(lost=["in cpp"])
    assert r["status"] == "completed_with_errors"
    assert r["accuracy"] is None
    assert r["correct_answers"] == 0 and r["total_questions"] == 0
    assert "no C++ generation was scored" in r["accuracy_withheld_reason"]
    assert "cpp" not in r["per_language"] and r["per_language"]["python"]["pass_at_1"] == 100.0
    assert r["pooled_pass_at_1"] == 100.0                  # base's pooled figure, not the headline
    assert r["leaderboard_comparable"] is False
    assert "pooled" not in r["metric"].split(";")[0]


def test_ojbench_with_every_request_failed_reports_no_score(monkeypatch):
    _oj(monkeypatch, [(1, "cpp"), (2, "python")])
    r = _run_oj(lost=["solve"])
    assert r["status"] == "failed"
    assert r["accuracy"] is None and r["accuracy_withheld_reason"]
    assert r["judge_no_verdict"] == 0
    assert r["request_harness_errors"] == {"endpoint_failure": 2}


def test_ojbench_doc_names_the_new_fields():
    doc = _flat((DOCS / "ojbench.md").read_text(encoding="utf-8"))
    assert "`judge_no_verdict`" in doc and "`request_harness_errors`" in doc
    assert "`endpoint_failure`, `endpoint_timeout`, `over_context`" in doc
    assert "no C++ generation scored (the headline is then withheld)" in doc


# --------------------------------------------------------------------------- #
# bigcodebench                                                                 #
# --------------------------------------------------------------------------- #
def _bcb(monkeypatch, tmp_path, tasks=("BigCodeBench/0", "BigCodeBench/1")):
    monkeypatch.setattr(B, "check_bigcodebench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(B, "_prefill_supported", lambda: True)
    monkeypatch.setattr(B, "harness_cpus", lambda: [0, 1])
    monkeypatch.setattr(B, "_load_bigcodebench_samples", lambda limit=None, subset="full": [
        ([{"role": "user", "content": f"{split} task {tid}"}], tid,
         {"category": split, "task_id": tid, "split": split, "hard": False})
        for tid in tasks for split in B.SPLITS])
    monkeypatch.setattr(B.subprocess, "run", lambda cmd, **k: types.SimpleNamespace(
        returncode=0, stdout="", stderr=""))
    # the harness passes every task in the samples file and reports its own pass@1 over them
    monkeypatch.setattr(B, "_parse_bcb_results", lambda workdir: (
        {t: True for t in tasks}, {}, {"bigcodebench_pass_at_1": 0.5}))


def _run_bcb(lost):
    with mock.patch.object(base, "_send_single_request", side_effect=_serve(lost)):
        return B.run_bigcodebench("m", "http://x/v1", 2)


def test_bigcodebench_withholds_instruct_when_every_instruct_request_failed(monkeypatch,
                                                                           tmp_path):
    _bcb(monkeypatch, tmp_path)
    r = _run_bcb(lost=["instruct task"])
    assert r["status"] == "completed_with_errors"
    assert "instruct" not in r["splits"] and r["splits"]["complete"]["full"]["total"] == 2
    # never the Complete figure under the Instruct headline, never 0
    assert r["accuracy"] is None and r["blended_splits_accuracy"] == 100.0
    assert "every Instruct request got no reply from the served endpoint" in \
        r["accuracy_withheld_reason"]
    assert r["bigcodebench_pass_at_1"] is None
    assert r["bigcodebench_pass_at_1_by_split"] == {"instruct": None, "complete": 0.5}
    assert "2 task(s) whose request got no reply" in r["bigcodebench_pass_at_1_withheld"]["instruct"]
    assert ("2 request(s) got no reply from the served endpoint (harness errors, left out of the "
            "score; request_harness_errors)") in r["leaderboard_comparable_reason"]


def test_bigcodebench_scores_the_rest_and_names_one_failed_request(monkeypatch, tmp_path):
    _bcb(monkeypatch, tmp_path)
    r = _run_bcb(lost=["instruct task BigCodeBench/1"])
    assert r["status"] == "completed_with_errors"
    assert r["accuracy"] == 100.0 and r["total_questions"] == 1
    assert r["bigcodebench_pass_at_1_by_split"]["complete"] == 0.5
    assert r["bigcodebench_pass_at_1_by_split"]["instruct"] is None
    assert "1 request(s) got no reply from the served endpoint" in \
        r["leaderboard_comparable_reason"]


def test_bigcodebench_clean_run_keeps_the_harness_figure(monkeypatch, tmp_path):
    _bcb(monkeypatch, tmp_path)
    r = _run_bcb(lost=[])
    assert r["bigcodebench_pass_at_1"] == 0.5
    assert "bigcodebench_pass_at_1_withheld" not in r
    assert "got no reply" not in (r.get("leaderboard_comparable_reason") or "")


# --------------------------------------------------------------------------- #
# omnidocbench: a page with no reply is not handed to the evaluator            #
# --------------------------------------------------------------------------- #
def _odb(monkeypatch, tmp_path, pages):
    gt = tmp_path / "OmniDocBench.json"
    gt.write_text(json.dumps([{"page_info": {"image_path": f"images/{p}.jpg"}, "layout_dets": []}
                              for p in pages]))
    monkeypatch.setattr(O, "_check_prereqs", lambda image: None)
    monkeypatch.setattr(O, "hf_hub_download", lambda **k: str(gt))
    monkeypatch.setattr(O, "_load_omnidocbench_samples", lambda limit=None: [
        ([{"role": "user", "content": [{"type": "text", "text": f"transcribe {p}"}]}],
         json.dumps({"page_name": f"{p}.jpg"}), {"category": "page"}) for p in pages])
    seen = {}

    def fake_docker(cmd, **kw):
        mounts = {c.split(":")[1]: c.split(":")[0] for c in cmd
                  if isinstance(c, str) and c.count(":") >= 1 and c.startswith("/")}
        seen["gt"] = json.load(open(mounts["/workspace/gt/gt.json"]))
        seen["preds"] = sorted(os.listdir(mounts["/workspace/data_md/predictions"]))
        with open(os.path.join(mounts["/workspace/result"],
                               "predictions_quick_match_run_summary.json"), "w") as f:
            json.dump({"notebook_metric_summary": {"overall_notebook": 50.0, "metrics": {}}}, f)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(O.subprocess, "run", fake_docker)
    return seen


def _run_odb(lost):
    with mock.patch.object(base, "_send_single_request", side_effect=_serve(lost)):
        return O.run_omnidocbench("m", "http://x/v1")


def test_omnidocbench_leaves_a_page_with_no_reply_out_of_the_evaluators_input(monkeypatch,
                                                                              tmp_path):
    seen = _odb(monkeypatch, tmp_path, ["p1", "p2", "p3"])
    r = _run_odb(lost=["transcribe p2"])
    assert seen["preds"] == ["p1.md", "p3.md"]
    assert [a["page_info"]["image_path"] for a in seen["gt"]] == ["images/p1.jpg",
                                                                  "images/p3.jpg"]
    assert r["pages_scored"] == 2 and r["pages_left_out"] == 1
    assert r["accuracy"] == 50.0
    assert r["status"] == "completed_with_errors"
    assert r["request_harness_errors"] == {"endpoint_failure": 1}
    assert r["leaderboard_comparable"] is False
    assert ("1 page request(s) got no reply from the served endpoint (harness errors, left out "
            "of the score; request_harness_errors)") in r["leaderboard_comparable_reason"]
    assert "scored as wrong" not in r["leaderboard_comparable_reason"]


def test_omnidocbench_with_every_page_failed_reports_no_score(monkeypatch, tmp_path):
    seen = _odb(monkeypatch, tmp_path, ["p1", "p2"])
    r = _run_odb(lost=["transcribe"])
    assert "gt" not in seen                                  # the evaluator never ran
    assert r["status"] == "failed"
    assert r["accuracy"] is None and r["accuracy_withheld_reason"]
    assert r["pages_scored"] == 0 and r["pages_left_out"] == 2
    assert r["omnidocbench_overall"] is None
    assert r["leaderboard_comparable"] is False


def test_omnidocbench_clean_run_is_unchanged(monkeypatch, tmp_path):
    seen = _odb(monkeypatch, tmp_path, ["p1", "p2"])
    r = _run_odb(lost=[])
    assert seen["preds"] == ["p1.md", "p2.md"] and len(seen["gt"]) == 2
    assert r["pages_left_out"] == 0 and r["pages_scored"] == 2
    assert r["status"] == "success" and r["leaderboard_comparable"] is True


def test_omnidocbench_doc_no_longer_scores_failed_pages():
    doc = _flat((DOCS / "omnidocbench.md").read_text(encoding="utf-8"))
    assert "written as an empty prediction and scored" not in doc
    assert "`pages_left_out`" in doc and "scored as wrong" not in doc


# --------------------------------------------------------------------------- #
# multipl_e                                                                    #
# --------------------------------------------------------------------------- #
def test_multipl_e_names_requests_with_no_response_as_left_out():
    src = (REPO / "gbench" / "runners" / "eval_suites" / "multipl_e.py").read_text()
    assert "scored as empty programs\")" not in src
    assert "got no response from the \"\n                        \"served endpoint" in src
    doc = _flat((DOCS / "multipl_e.md").read_text(encoding="utf-8"))
    assert "`multipl_e_no_response`: requests that got no response from the served endpoint" in doc
    assert "requests that got no response, scored as empty programs" not in doc


def test_multipl_e_run_with_a_failed_request_leaves_it_out(monkeypatch):
    monkeypatch.setattr(M, "check_multipl_e_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(M, "_langs", lambda: ["py"])
    monkeypatch.setenv("GBENCH_MULTIPL_E_SAMPLES", "1")
    gold = json.dumps({"name": "HumanEval_0", "language": "py", "prompt": "", "tests": "",
                       "stop_tokens": []})
    monkeypatch.setattr(M, "_load_multipl_e_samples", lambda langs=None, limit=None: [
        ([{"role": "user", "content": f"write q{i}"}], gold, {"category": "py"})
        for i in range(2)])

    def scorer(image, workers, metrics):
        async def _score(traces):
            metrics["multipl_e_no_response"] = sum(1 for t in traces
                                                   if t.get("response_text") is None)
            for t in traces:
                t["is_correct"], t["status"] = t.get("response_text") is not None, "OK"
        return _score
    monkeypatch.setattr(M, "_make_scorer", scorer)
    with mock.patch.object(base, "_send_single_request", side_effect=_serve(["write q1"])):
        r = M.run_multipl_e("m", "http://x/v1", 2)
    assert r["status"] == "completed_with_errors"
    assert r["accuracy"] == 100.0                           # the failed request is not a 0
    assert r["request_harness_errors"] == {"endpoint_failure": 1}
    assert ("1 request(s) got no response from the served endpoint (harness errors, left out of "
            "the score; request_harness_errors)") in r["leaderboard_comparable_reason"]
