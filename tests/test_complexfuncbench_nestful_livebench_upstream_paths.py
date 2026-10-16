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

"""complexfuncbench, nestful and livebench on their upstream paths.

complexfuncbench: upstream's call-matching path, its live RapidAPI (Booking) response comparison
included (rule_based -> value_checker -> response_based -> Gemini llm_based): a non-model tool
backend runs as upstream, with RAPID_API_KEY; the result records it and is not
leaderboard-comparable because of the Gemini judges. The Gemini judge key follows GEMINI_API_KEYS
first, then GEMINI_API_KEY.

nestful: one data source, the IBM/NESTFUL checkout (data_v2/nestful_data.jsonl +
data_v2/executable_functions); no Hugging Face copy; no checkout -> whole-suite infra_required; the
Win Rate alone infra_required when only the executable functions are missing.

livebench: the pinned commit plus one minimal recorded patch for the upstream
instruction_following grading crash (list(set(...)) over question dicts -> de-duplicated by
question_id), applied by docker/livebench.Dockerfile at build time, checked in the image before
each run and recorded on every result.
"""
import ast
import asyncio
import json
import re
import shlex
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import complexfuncbench as C
from gbench.runners.eval_suites import livebench as LB
from gbench.runners.eval_suites import nestful as N
from gbench.runners.eval_suites import tau_common as TC

ROOT = Path(__file__).resolve().parent.parent
SUITES = ROOT / "gbench" / "runners" / "eval_suites"


def _env_keys_read(path):
    """Every string key a module reads with os.environ.get / os.getenv / os.environ[...]."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            f = ast.unparse(node.func)
            if f in ("os.environ.get", "os.getenv", "environ.get", "getenv"):
                keys.add(node.args[0].value)
        if isinstance(node, ast.Subscript) and ast.unparse(node.value) == "os.environ":
            if isinstance(node.slice, ast.Constant):
                keys.add(node.slice.value)
    return keys


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


# =========================================================================== #
# complexfuncbench
# =========================================================================== #
class _LiveAPI:
    """A stand-in for the booking-com15 API: each call answers its own response, so two calls
    that differ get different responses and the response-based tier leaves the pair to the
    judge."""

    def __init__(self):
        self.stats = Counter()
        self.calls = []

    async def call(self, func_call):
        self.calls.append(func_call)
        return {"response": {"echo": json.dumps(func_call, sort_keys=True)}, "raised": False,
                "labels": ["ok"]}


def _cmp(usage=None, api=None):
    return C._CompareFC(C._load_exact_match_dict(), C._get_embedder(), usage,
                        api if api is not None else _LiveAPI())


def test_complexfuncbench_reads_rapid_api_key_for_the_response_based_tier():
    src = SUITES / "complexfuncbench.py"
    assert "RAPID_API_KEY" in src.read_text(encoding="utf-8")
    assert C.RAPID_API_KEY_ENV == "RAPID_API_KEY"
    assert hasattr(C, "_RapidAPICall") and hasattr(C._CompareFC, "response_based")
    assert "requests" in src.read_text(encoding="utf-8")   # utils/rapidapi.py's HTTP client
    assert "tool_info.json" in src.read_text(encoding="utf-8")


def test_complexfuncbench_call_matching_is_upstreams_path(monkeypatch):
    assert C.CALL_MATCH_TIERS == ["rule_based", "value_checker", "response_based", "llm_based"]
    usage = Counter()
    api = _LiveAPI()
    cmp = _cmp(usage, api)
    seen = []

    async def judge(functions, history, predict, golden):
        seen.append((predict, golden))
        return True
    cmp.llm_based = judge
    # value_checker passes (no critical params for this name), rule_based does not -> the
    # response-based tier (different responses) -> the judge.
    pred = {"name": "Search_Foo", "arguments": {"q": "New York"}}
    gold = {"name": "Search_Foo", "arguments": {"q": "NY"}}
    ok, err = asyncio.run(cmp.compare_single_call([], [], pred, gold))
    assert ok is True and err is None and seen == [(pred, gold)]
    assert api.calls == [pred, gold]
    # rule_based matches: neither the live API nor the judge is asked.
    seen.clear()
    api.calls.clear()
    ok, _ = asyncio.run(cmp.compare_single_call([], [], dict(gold), dict(gold)))
    assert ok is True and seen == [] and api.calls == []


def test_complexfuncbench_judge_usage_counts_models_and_outages(monkeypatch):
    replies = iter([('{"is_equal": false, "reason": "x"}', "gemini-a"), (None, "JUDGE_OUTAGE"),
                    (None, "JUDGE_OUTAGE")])

    async def fake_cascade(contents, *, config=None):
        return next(replies)
    monkeypatch.setattr(C, "judge_generate_cascade", fake_cascade)
    usage = Counter()
    cmp = _cmp(usage)
    a = {"name": "F", "arguments": {"x": 1}}
    b = {"name": "F", "arguments": {"x": 2}}
    assert asyncio.run(cmp.llm_based([], [], a, b)) is False
    # an outage gives no verdict: None, as upstream's llm_based returns when GPT fails ...
    assert asyncio.run(cmp.llm_based([], [], a, b)) is None
    assert cmp.judge_failure == "call_judge_outage"
    # ... and a harness error, never "not equivalent": the sample is not scored
    with pytest.raises(C._CallJudgeNoVerdict) as ei:
        asyncio.run(cmp.compare_single_call([], [], a, b))
    assert ei.value.kind == "call_judge_outage"
    assert usage == Counter({"gemini-a": 1, "JUDGE_OUTAGE": 2})


def _run_cfb(monkeypatch, usage):
    monkeypatch.setattr(C, "check_complexfuncbench_prerequisites", lambda: (True, ""))

    async def fake_drive(*a, **k):
        return {"success_rate": 50.0, "call_accuracy": 60.0, "completeness": 1.5,
                "correctness": 1.0, "n_scored": 2, "n_request_failed": 0, "n_success": 1,
                "per_domain_success_rate": None, "_is_full": False, "_n_items": 2,
                "_embed_backend": "transformers", "_judge_usage": usage}
    monkeypatch.setattr(C, "_drive_all", fake_drive)
    return C.run_complexfuncbench("m", "http://x/v1", 1, limit=2)


def test_complexfuncbench_result_records_the_tier_and_the_gemini_judges(monkeypatch):
    monkeypatch.setenv("RAPID_API_KEY", "must-not-appear-0123456789")
    r = _run_cfb(monkeypatch, Counter({"gemini-x": 5, "JUDGE_OUTAGE": 2}))
    assert r["leaderboard_comparable"] is False
    reason = r["leaderboard_comparable_reason"]
    assert "not run" not in reason
    assert "Gemini" in reason and "gpt-4o-2024-05-13" in reason and "gpt-4o-2024-08-06" in reason
    assert r["call_match_tiers"] == ["rule_based", "value_checker", "response_based",
                                     "llm_based"]
    assert r["response_based_tier"]["run"] is True
    assert r["response_based_tier"]["tool"] == "booking-com15"
    upstream = {s["upstream"] for s in r["judge_substitution"]}
    assert upstream == {"gpt-4o-2024-05-13 (OpenAI API)", "gpt-4o-2024-08-06 (OpenAI API)"}
    assert all("Gemini API" in s["gbench"] for s in r["judge_substitution"])
    assert r["judge_models"] == {"gemini-x": 5} and r["judge_outages"] == 2
    # a Gemini judge outage is a harness error of the run, named in the reason
    assert r["status"] == "completed_with_errors"
    assert "2 Gemini judge call(s) failed on every cascade model" in reason
    assert r["judge_cascade"] == base.judge_cascade()
    assert r["upstream_revision"] == f"THUDM/ComplexFuncBench@{C.UPSTREAM_COMMIT}"
    assert "must-not-appear-0123456789" not in json.dumps(r)


def test_complexfuncbench_gemini_keys_list_wins(monkeypatch):
    seen = []
    monkeypatch.setattr(C, "gemini_key_live_valid", lambda k: (seen.append(k), (True, ""))[1])
    monkeypatch.setenv("RAPID_API_KEY", "rapid-key-0123456789")
    monkeypatch.setattr(C, "rapidapi_live_check", lambda key: None)
    monkeypatch.setenv("GEMINI_API_KEYS", "k1,k2")
    monkeypatch.setenv("GEMINI_API_KEY", "single")
    ok, why = C.check_complexfuncbench_prerequisites()
    assert ok, why
    assert seen == ["k1,k2"]
    monkeypatch.delenv("GEMINI_API_KEY")
    seen.clear()
    assert C.check_complexfuncbench_prerequisites()[0] and seen == ["k1,k2"]


def test_complexfuncbench_no_or_rejected_key_names_both_variables_not_the_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    ok, why = C.check_complexfuncbench_prerequisites()
    assert not ok and "GEMINI_API_KEYS" in why and "GEMINI_API_KEY" in why
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value-123")
    monkeypatch.setattr(C, "gemini_key_live_valid", lambda k: (False, "HTTP 403"))
    ok, why = C.check_complexfuncbench_prerequisites()
    assert not ok and "rejected" in why and "secret-value-123" not in why
    with pytest.raises(RuntimeError) as e:
        C.run_complexfuncbench("m", "http://x/v1", 1)
    assert "[INFRA REQUIRED] complexfuncbench" in str(e.value)
    assert "secret-value-123" not in str(e.value)


def test_complexfuncbench_embedder_and_assignment_are_single_paths():
    src = SUITES / "complexfuncbench.py"
    assert not {"FlagEmbedding", "sentence_transformers"} & _imports(src)
    assert C._Embedder().backend == "transformers"
    # scipy's optimal assignment for every size (a greedy pick of the 10 would give 16 here)
    m = np.zeros((8, 8))
    m[0, 0], m[0, 1], m[1, 0] = 10, 9, 9
    for i in range(2, 8):
        m[i, i] = 1
    r, c = C._assign_max(m)
    assert sum(m[i, j] for i, j in zip(r, c)) == 24
    assert all(isinstance(x, int) for x in r + c)
    assert C._assign_max(np.zeros((0, 4))) == ([], [])


def test_complexfuncbench_doc_matches_the_fixed_path():
    doc = (ROOT / "docs" / "evals" / "complexfuncbench.md").read_text(encoding="utf-8")
    assert doc.isascii()
    assert "`RAPID_API_KEY` is not read" not in doc and "`RAPID_API_KEY`" in doc
    assert '`["rule_based", "value_checker", "response_based", "llm_based"]`' in doc
    assert "OPTIONAL" not in doc and "graceful degradation" not in doc
    assert "GEMINI_API_KEYS" in doc and "gpt-4o-2024-05-13" in doc and "gpt-4o-2024-08-06" in doc
    for field in ("call_match_tiers", "response_based_tier", "judge_substitution",
                  "judge_cascade", "judge_models", "judge_outages", "upstream_revision"):
        assert f"`{field}`" in doc, field
    assert "FlagEmbedding` ->" not in doc and "sentence-transformers` ->" not in doc


# =========================================================================== #
# nestful
# =========================================================================== #
_TOOLS = [{"name": "add", "description": "", "parameters": {},
           "output_parameters": {"result": {}}}]
_ROWS = [{"sample_id": "s1", "input": "1+2?", "tools": _TOOLS, "gold_answer": 3,
          "output": [{"name": "add", "label": "$var_1", "arguments": {"a": 1, "b": 2}}]}]


def _clear_nestful_env(monkeypatch):
    monkeypatch.delenv(N._FUNC_DIR_ENV, raising=False)
    monkeypatch.delenv("GBENCH_PREREQS_DIR", raising=False)


def _forbid_hf(monkeypatch):
    import huggingface_hub

    def _no(*a, **k):
        raise AssertionError("nestful must never download from Hugging Face")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _no)


def _endpoint(answer):
    async def _reply(**kw):
        return base.Reply(answer(kw["messages"]), None, "stop", completion_tokens=5)
    return _reply


def test_nestful_module_has_no_hugging_face_data_path():
    src = SUITES / "nestful.py"
    text = src.read_text(encoding="utf-8")
    assert not {"huggingface_hub", "pyarrow", "datasets"} & _imports(src)
    assert "ibm-research" not in text and "hf_hub_download" not in text
    for name in ("_hf_rows", "HF_DATASET", "HF_REVISION", "HF_FILE", "HF_SHA256"):
        assert not hasattr(N, name), name


def test_nestful_empty_prereqs_tree_is_a_whole_suite_infra_required(tmp_path, monkeypatch):
    _clear_nestful_env(monkeypatch)
    monkeypatch.setenv("GBENCH_PREREQS_DIR", str(tmp_path))          # no NESTFUL/ under it
    _forbid_hf(monkeypatch)

    async def _no_request(**kw):
        raise AssertionError("nothing may be generated without the checkout")
    with mock.patch.object(base, "_send_single_request", _no_request):
        with pytest.raises(RuntimeError) as e:
            N.run_nestful("m", "http://x", 1)
    assert str(e.value).startswith("[INFRA REQUIRED] nestful cannot run")
    assert "no IBM/NESTFUL checkout" in str(e.value)


def test_nestful_only_the_functions_missing_reports_the_win_rate_alone(tmp_path, monkeypatch):
    data_v2 = tmp_path / "NESTFUL" / "data_v2"
    data_v2.mkdir(parents=True)
    path = data_v2 / "nestful_data.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in _ROWS))
    monkeypatch.setitem(N._PINNED_BLOBS, "nestful_data.jsonl", N._git_blob_sha1(str(path)))
    _clear_nestful_env(monkeypatch)
    monkeypatch.setenv("GBENCH_PREREQS_DIR", str(tmp_path))
    _forbid_hf(monkeypatch)
    reply = json.dumps(_ROWS[0]["output"])
    with mock.patch.object(base, "_send_single_request", _endpoint(lambda m: reply)):
        r = N.run_nestful("m", "http://x", 1)
    assert r["full_match_accuracy"] == 100.0 and r["accuracy"] == 100.0
    assert r["win_rate"] is None and r["metrics_infra_required"] == ["win_rate"]
    assert r["win_rate_metric"]["status"] == "infra_required"
    assert r["dataset_source"] == "checkout"
    assert r["dataset_revision"] == f"IBM/NESTFUL@{N.NESTFUL_COMMIT} data_v2/nestful_data.jsonl"
    assert r["status"] == "completed_with_errors" and r["leaderboard_comparable"] is False


def test_nestful_row_without_gold_answer_is_refused(tmp_path):
    path = tmp_path / "nestful_data.jsonl"
    row = dict(_ROWS[0])
    row.pop("gold_answer")
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(RuntimeError, match="unexpected row schema"):
        N._load_nestful_samples(data_path=str(path))


def test_nestful_doc_states_the_single_source():
    doc = (ROOT / "docs" / "evals" / "nestful.md").read_text(encoding="utf-8")
    assert doc.isascii()
    assert "ibm-research" not in doc and "HF copy" not in doc
    assert "only data source" in doc
    assert "infra_required" in doc and "data_v2/executable_functions" in doc


# =========================================================================== #
# livebench
# =========================================================================== #
DOCKERFILE = ROOT / "docker" / "livebench.Dockerfile"


def _patch_run_step():
    """The Dockerfile's RUN step that applies the patch, with continuation lines joined."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    joined = re.sub(r"\\\n", "", text)
    steps = [ln for ln in joined.splitlines() if ln.startswith("RUN ") and "sha256sum" in ln]
    assert len(steps) == 1
    return steps[0][len("RUN "):]


def test_livebench_patch_record_matches_the_dockerfile():
    step = _patch_run_step()
    p = LB.LIVEBENCH_PATCH
    assert p["file"] == "livebench/gen_ground_truth_judgment.py" and p["line"] == 480
    assert f"echo \"{p['sha256_upstream']}  {p['file']}\"" in step
    assert f"echo \"{p['sha256_patched']}  {p['file']}\"" in step
    assert step.index(p["sha256_upstream"]) < step.index("python -c") < step.index(
        p["sha256_patched"])
    assert p["upstream"] in step
    assert p["patched"].replace('"', '\\"') in step or p["patched"] in step
    text = DOCKERFILE.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    assert lines.index("COPY . /opt/LiveBench") < next(
        i for i, ln in enumerate(lines) if ln.startswith("RUN echo") and p["sha256_upstream"] in ln)
    # the build context is still the pinned checkout (no extra context needed for the patch)
    assert LB.BUILD_COMMAND.split()[-1] == "/path/to/LiveBench"


def _python_of_patch_step():
    """The `python -c` program of the patch step, as the shell would pass it."""
    step = _patch_run_step()
    words = shlex.split(step)
    return words[words.index("python") + 2]


def test_livebench_patch_step_replaces_exactly_the_one_line(tmp_path):
    code = _python_of_patch_step()
    target = tmp_path / "livebench" / "gen_ground_truth_judgment.py"
    target.parent.mkdir()
    before = ("x = 1\n        " + LB.LIVEBENCH_PATCH["upstream"] + "\n"
              "        task_name = if_questions[0]['task']\n")
    target.write_bytes(before.encode())
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)
    after = target.read_text()
    assert after == before.replace(LB.LIVEBENCH_PATCH["upstream"], LB.LIVEBENCH_PATCH["patched"])
    # the patched text de-duplicates by question_id, keeping each question once
    qs = [{"question_id": "a"}, {"question_id": "b"}, {"question_id": "a"}]
    matches = [types.SimpleNamespace(question=q) for q in qs]
    ns = {"old_instruction_following_matches": matches}
    exec(LB.LIVEBENCH_PATCH["patched"], {}, ns)
    assert [q["question_id"] for q in ns["if_questions"]] == ["a", "b"]
    # the line absent (or twice) fails the build step
    target.write_text("nothing to patch\n")
    assert subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                          capture_output=True).returncode != 0


def _fake_docker(monkeypatch, stdout, seen=None):
    monkeypatch.setattr(LB.shutil, "which", lambda _x: "/usr/bin/docker")

    def fake_run(cmd, **_kw):
        if seen is not None:
            seen.append(cmd)
        out = stdout if cmd[:2] == ["docker", "run"] else ""
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")
    monkeypatch.setattr(LB.subprocess, "run", fake_run)


def test_livebench_prereqs_accept_the_patched_image(monkeypatch):
    seen = []
    _fake_docker(monkeypatch, LB._PATCH_MARKER + LB.LIVEBENCH_PATCH["sha256_patched"] + "\n", seen)
    LB._check_prereqs("img:tag")
    run = [c for c in seen if c[:2] == ["docker", "run"]]
    assert len(run) == 1 and run[0][2:5] == ["--rm", "--network", "none"]
    assert "img:tag" in run[0]


@pytest.mark.parametrize("stdout, needle", [
    (LB._PATCH_MARKER + LB.LIVEBENCH_PATCH["sha256_upstream"] + "\n", "unpatched"),
    (LB._PATCH_MARKER + "ab" * 32 + "\n", "ab" * 32),
    ("", "could not be checked"),
])
def test_livebench_prereqs_refuse_an_image_without_the_patch(monkeypatch, stdout, needle):
    _fake_docker(monkeypatch, stdout)
    with pytest.raises(RuntimeError) as e:
        LB._check_prereqs("img:tag")
    msg = str(e.value)
    assert msg.startswith("[INFRA REQUIRED] livebench cannot run")
    assert needle in msg
    assert "docker build -t img:tag -f docker/livebench.Dockerfile /path/to/LiveBench" in msg


def _write_answers(tmp_path, category, task, n):
    d = tmp_path / LB._ANSWER_DIR / "data" / "live_bench" / category / task / "model_answer"
    d.mkdir(parents=True, exist_ok=True)
    row = {"question_id": "q", "choices": [{"index": 0, "turns": ["ok"]}],
           "total_output_tokens": 10}
    (d / "m.jsonl").write_text("".join(json.dumps(dict(row, question_id=f"q{i}")) + "\n"
                                       for i in range(n)), encoding="utf-8")


def _run_lb(monkeypatch, tmp_path, if_rows):
    base._RUN_KNOBS.clear()
    for name in ("GBENCH_LIVEBENCH_RELEASE", "GBENCH_LIVEBENCH_IMAGE", "GBENCH_LIVEBENCH_CATEGORIES",
                 "GBENCH_LIVEBENCH_TEMPERATURE", "GBENCH_LIVEBENCH_MODEL",
                 "GBENCH_LIVEBENCH_EXTRA_ARGS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(LB, "_check_prereqs", lambda image: None)
    # the network checks before the run container are docker calls of their own
    # (tests/test_guard_code_images.py)
    monkeypatch.setattr(TC, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC.container_url(url))
    monkeypatch.setattr(LB, "upstream_model_config", lambda image, names: {
        "name": None, "config": None, "commit": LB.LIVEBENCH_COMMIT})
    monkeypatch.setattr(LB.tempfile, "mkdtemp", lambda prefix="": str(tmp_path))

    def fake_run(cmd, **_kw):
        cols = LB._CATEGORIES
        (tmp_path / "all_groups.csv").write_text(
            "model,average," + ",".join(cols) + "\nm,50.0," + ",".join("50.0" for _ in cols)
            + "\n", encoding="utf-8")
        for c in cols:
            if c != "instruction_following":
                _write_answers(tmp_path, c, "t", 2)
        for task, n in if_rows.items():
            _write_answers(tmp_path, "instruction_following", task, n)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(LB.subprocess, "run", fake_run)
    return LB.run_livebench("m", "http://h/v1", 4, max_output_tokens=4096)


def test_livebench_result_records_the_patch(monkeypatch, tmp_path):
    r = _run_lb(monkeypatch, tmp_path, {"paraphrase": 50, "simplify": 50,
                                        "story_generation": 50, "summarize": 50})
    assert r["livebench_patch"] == LB.LIVEBENCH_PATCH
    assert r["livebench_image_commit"] == LB.LIVEBENCH_COMMIT
    assert r["leaderboard_comparable"] is True, r.get("leaderboard_comparable_reason")
    assert r["status"] == "success"


def test_livebench_unequal_instruction_following_tasks_are_not_comparable(monkeypatch, tmp_path):
    r = _run_lb(monkeypatch, tmp_path, {"paraphrase": 50, "simplify": 3})
    assert r["leaderboard_comparable"] is False
    reason = r["leaderboard_comparable_reason"]
    assert "instruction_following tasks answered in unequal numbers" in reason
    assert "{'paraphrase': 50, 'simplify': 3}" in reason
    assert LB.instruction_following_rows(str(tmp_path)) == {"paraphrase": 50, "simplify": 3}


def test_livebench_doc_describes_the_recorded_patch():
    doc = (ROOT / "docs" / "evals" / "livebench.md").read_text(encoding="utf-8")
    assert doc.isascii()
    assert "## Recorded patch: instruction_following grading at `0baa4aa7`" in doc
    assert "Known upstream defect" not in doc
    p = LB.LIVEBENCH_PATCH
    assert p["sha256_upstream"] in doc and p["sha256_patched"] in doc
    assert "-        " + p["upstream"] in doc and "+        " + p["patched"] in doc
    assert "`livebench_patch`" in doc
    assert "instruction_following tasks answered in equal numbers" in doc
