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

"""Agent and tool suites run as their upstream harnesses run them.

  * mcp_atlas     - gbench's temperature rule (the harness sends none), tool-argument parse
                    errors fed back,
                    the 1800 s per-task clock, an ERROR: response scored 0, a lost agent
                    environment is a hard error, pre-flight classification.
  * gaia          - the leaderboard's test split: the submission file, then infra_required.
  * cimemories    - release texts read from checksummed release files at run time; generations
                    cut off by the output-token budget; the shared label cache; no temperature.
  * gdpval,
    deepsearch_qa - no temperature (the reference sends none); the rater's own config.
  * toolbench     - ToolEval's prompts, reply schema, temperature, retries and SoWR choice rule.
  * mcp_bench     - every server must serve before the run; no provisioned-subset run.

No network, no Docker, no model server, no judge API: everything external is stubbed.
"""

import ast
import asyncio
import hashlib
import io
import json
import os
import subprocess
import textwrap
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import cimemories as CIM
from gbench.runners.eval_suites import deepsearch_qa as DSQA
from gbench.runners.eval_suites import gaia as GAIA
from gbench.runners.eval_suites import gdpval as G
from gbench.runners.eval_suites import mcp_atlas as M
from gbench.runners.eval_suites import mcp_bench as MB
from gbench.runners.eval_suites import toolbench as T

_HERE = os.path.dirname(__file__)


def _doc(name):
    with open(os.path.join(_HERE, "..", "docs", "evals", name), encoding="utf-8") as fh:
        return " ".join(fh.read().split())


# --------------------------------------------------------------------------- #
# mcp_atlas: a stand-in agent environment
# --------------------------------------------------------------------------- #
_CATALOG = [{"name": "calculator_calculate", "description": "Calculates.",
             "inputSchema": {"type": "object", "properties": {"expression": {}}}}]


def _serve(calls, answer=None):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                return self._send(200, {"message": "ok"})
            if self.path == "/enabled-servers":
                servers = [[s, "OK"] for s in M.RELEASE_SERVERS]
                return self._send(200, {"servers": servers, "total": len(servers),
                                        "online": len(servers), "offline": 0})
            return self._send(404, {})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/list-tools":
                return self._send(200, _CATALOG)
            calls.append(body)
            return self._send(200, [{"type": "text", "text": (answer or "ok")}])

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


@pytest.fixture
def atlas_env(monkeypatch):
    calls = []
    srv, url = _serve(calls, answer="42")
    monkeypatch.setenv("GBENCH_MCP_ATLAS_SANDBOX_URL", url)
    # the external environment's checks, gbench's credentialed-servers container and the
    # image checks they need (tests/test_mcp_credential_isolation_and_dns.py covers them): no
    # docker here
    monkeypatch.setattr(M, "_external_environment_setup", lambda sandbox: None)
    monkeypatch.setattr(M, "check_mcp_atlas_prerequisites", lambda: (True, ""))
    monkeypatch.delenv("GBENCH_MCP_ATLAS_TEMPERATURE", raising=False)
    rows = [{"TASK": "t1", "PROMPT": "What is 6 times 7?",
             "ENABLED_TOOLS": '["calculator_calculate"]',
             "GTFA_CLAIMS": '["The answer is 42."]'}]
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: rows)
    monkeypatch.setattr(M, "gemini_required_skip", lambda *a: None)
    # the search engines' Gemini grounding backend, stood in (no network)
    monkeypatch.setattr(M, "_ground", lambda query, n: (
        [{"title": "T", "url": "https://example.org/", "snippet": "S"}], "gemini:test"))

    async def judge(prompt, config=None):
        return json.dumps({"claim_text": "c", "coverage_outcome": "fulfilled",
                           "justification": "j", "confidence_level": 1.0}), "gemini-test"

    monkeypatch.setattr(M, "judge_generate_cascade", judge)
    yield {"calls": calls, "srv": srv, "url": url}
    srv.shutdown()
    srv.server_close()


def _tool_call(args):
    return base.Reply(text="", tool_calls=[{"id": "c1", "type": "function", "function": {
        "name": "calculator_calculate", "arguments": args}}], finish_reason="tool_calls")


def test_mcp_atlas_protocol_matches_the_harness():
    loop = M.MCP_ATLAS_TOOL_LOOP
    assert loop.invalid_arguments_as_error is True
    assert loop.max_sample_wall_s == M.TASK_TIMEOUT_S == 1800.0
    assert (loop.max_rounds, loop.max_tool_calls, loop.forced_final) == (256, 100, False)
    assert M.MCP_ATLAS_SAMPLING.send_temperature is False
    assert "litellm-strategy.ts" in M.MCP_ATLAS_SAMPLING.reference


def test_mcp_atlas_sends_no_temperature_and_feeds_back_bad_arguments(atlas_env, monkeypatch):
    seen = []

    async def fake_send(**k):
        seen.append(dict(k, messages=list(k["messages"])))
        if len(seen) == 1:
            return _tool_call('{"expression": ')              # not JSON
        if len(seen) == 2:
            return _tool_call('{"expression": "6 * 7"}')
        return base.Reply(text="It is 42.", tool_calls=None, finish_reason="stop")

    monkeypatch.setattr(base, "_send_single_request", fake_send)
    res = M.run_mcp_atlas("m", "http://127.0.0.1:1/v1", concurrency=1, max_output_tokens=256)
    # gbench's rule: a no-think run sends 0.0 on every turn (the harness sends no temperature)
    assert {k["temperature"] for k in seen} == {0.0}
    assert res["temperature"] == 0.0 and res["temperature_source"] == "default"
    assert res["official_sampling"]["temperature_matches_official"] is False
    fed_back = seen[1]["messages"][-1]
    assert fed_back["role"] == "tool" and fed_back["content"].startswith("Error: ")
    # the pre-flight's calls to the environment's own servers (ddg-search's check call is its
    # search tool, answered by gbench's grounding; exa's and oxylabs' reach their servers, here
    # the one environment), then the one valid call
    env_checks = [s for s in M.ENVIRONMENT_SERVERS if s not in M.GEMINI_PREFLIGHT_SERVERS]
    assert len(atlas_env["calls"]) - len(env_checks) == 1, \
        "only the valid call reached the environment"
    assert res["tool_argument_errors"] == 1
    assert res["sample_traces"][0]["tool_calls_executed"] == 2   # both count toward the cap
    assert "temperature 0.0 set by default; the harness sends none" in (
        res["leaderboard_comparable_reason"])


def test_mcp_atlas_operator_temperature_is_sent_and_named(atlas_env, monkeypatch):
    monkeypatch.setenv("GBENCH_MCP_ATLAS_TEMPERATURE", "0.7")
    seen = []

    async def fake_send(**k):
        seen.append(k["temperature"])
        return base.Reply(text="The answer is 42.", tool_calls=None, finish_reason="stop")

    monkeypatch.setattr(base, "_send_single_request", fake_send)
    res = M.run_mcp_atlas("m", "http://127.0.0.1:1/v1", concurrency=1, max_output_tokens=256)
    assert seen == [0.7]
    assert res["official_sampling"]["temperature_matches_official"] is False
    assert "temperature 0.7 set by env:GBENCH_MCP_ATLAS_TEMPERATURE" in (
        res["leaderboard_comparable_reason"])


def test_mcp_atlas_task_past_its_wall_clock_scores_zero(atlas_env, monkeypatch):
    monkeypatch.setattr(M, "MCP_ATLAS_TOOL_LOOP",
                        M.MCP_ATLAS_TOOL_LOOP._replace(max_sample_wall_s=0.3))

    async def slow(**k):
        async with k["semaphore"]:
            await asyncio.sleep(5)
        return base.Reply(text="late", tool_calls=None, finish_reason="stop")

    monkeypatch.setattr(base, "_send_single_request", slow)
    t0 = time.monotonic()
    res = M.run_mcp_atlas("m", "http://127.0.0.1:1/v1", concurrency=1, max_output_tokens=256)
    assert time.monotonic() - t0 < 4
    tr = res["sample_traces"][0]
    assert tr["status"] == "WALL_CLOCK_LIMIT" and tr["claim_recall"] == 0.0
    assert res["wall_clock_limited"] == 1 and res["accuracy"] == 0.0


def test_mcp_atlas_error_response_scores_zero_without_a_judge_call():
    calls = []

    async def judge(prompt, config=None):
        calls.append(prompt)
        return None, "x"

    tr = [{"gold_answer": '["The answer is 42."]', "response_text": "ERROR: timeout after 1800s"}]
    with mock.patch.object(M, "judge_generate_cascade", judge):
        asyncio.run(M._async_judge_mcp_atlas(tr))
    assert calls == [] and tr[0]["claim_recall"] == 0.0 and tr[0]["is_correct"] is False
    assert "error response" in tr[0]["judge_grade"]


def test_mcp_atlas_lost_environment_is_a_hard_error(atlas_env, monkeypatch):
    srv = atlas_env["srv"]

    async def fake_send(**k):
        if k["messages"][-1]["role"] == "user":
            srv.shutdown()                  # the environment goes away before the tool call
            srv.server_close()
            return _tool_call('{"expression": "6 * 7"}')
        return base.Reply(text="It is unknown.", tool_calls=None, finish_reason="stop")

    monkeypatch.setattr(base, "_send_single_request", fake_send)
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*agent environment was lost"):
        M.run_mcp_atlas("m", "http://127.0.0.1:1/v1", concurrency=1, max_output_tokens=256)


def test_mcp_atlas_stopped_container_short_circuits_later_calls(monkeypatch):
    monkeypatch.setattr(M, "_container_running", lambda name: False)
    sb = M._Sandbox("http://127.0.0.1:9", container="gone")
    token = M._ACTIVE_SANDBOX.set(sb)
    try:
        first = asyncio.run(M.execute_mcp_tool("fetch_fetch", {"url": "u"}))
        second = asyncio.run(M.execute_mcp_tool("fetch_fetch", {"url": "u"}))
    finally:
        M._ACTIVE_SANDBOX.reset(token)
    assert first.startswith("Error: Failed to call tool fetch_fetch: ")
    assert sb.lost == "container gone is no longer running"
    assert second == "Error: Failed to call tool fetch_fetch: container gone is no longer running"
    assert sb.transport_errors == 2
    assert M.environment_alive(sb) == (False, sb.lost)


def test_mcp_atlas_preflight_uses_the_enabled_server_list():
    results = {s: (True, "") for s in M.RELEASE_SERVERS}
    results["github"] = (False, "Timed out after 30s")   # not running
    results["arxiv"] = (False, "Timed out after 30s")    # running, slow
    enabled = [s for s in M.RELEASE_SERVERS if s != "github"]
    msg = M.preflight_failure_reason(results, local_env=False, enabled=enabled)
    running, failing = msg.split("Running but failing their check: ")
    assert "Not running in the agent environment: github (needs GITHUB_TOKEN)" in running
    assert failing.startswith("arxiv: Timed out after 30s")
    assert "exceed the check's 30 s timeout (test_servers.py's)" in msg
    assert "one tool call at a time" not in msg
    # without the list, a timed-out server cannot be told apart and counts as failing
    assert "Not running" not in M.preflight_failure_reason(results, local_env=False)


def test_mcp_atlas_doc_states_the_protocol():
    doc = _doc("mcp_atlas.md")
    for needle in ("1800 s", "wall_clock_limited", "tool_argument_errors", "no temperature",
                   "litellm-strategy.ts", "4.9 GB", "official_sampling",
                   "infra_required` and no score is published", "--max-output-tokens"):
        assert needle in doc, needle
    assert "Where gbench's shared tool loop still differs" not in doc
    assert "one tool call at a time" not in doc
    assert "audit " + "RC" not in open(M.__file__, encoding="utf-8").read()


# --------------------------------------------------------------------------- #
# gaia: the test split and its submission file
# --------------------------------------------------------------------------- #
def test_gaia_split_knob(monkeypatch):
    monkeypatch.delenv("GBENCH_GAIA_SPLIT", raising=False)
    assert GAIA.gaia_split() == "validation"
    monkeypatch.setenv("GBENCH_GAIA_SPLIT", " Test ")
    assert GAIA.gaia_split() == "test"
    monkeypatch.setenv("GBENCH_GAIA_SPLIT", "dev")
    with pytest.raises(ValueError, match="GBENCH_GAIA_SPLIT"):
        GAIA.gaia_split()
    assert GAIA.LEADERBOARD_LEVEL_COUNTS["test"] == {1: 93, 2: 159, 3: 49}
    assert GAIA.GAIA_SAMPLING.send_temperature is False


def _test_snapshot(tmp_path):
    pd = pytest.importorskip("pandas")
    d = tmp_path / "2023" / "test"
    d.mkdir(parents=True)
    rows = [{"task_id": "a1", "Question": "Who?", "Level": "1", "Final answer": "?",
             "file_name": "", "file_path": ""},
            {"task_id": "b2", "Question": "How many tiles?", "Level": "3", "Final answer": "?",
             "file_name": "b2.MOV", "file_path": "2023/test/b2.MOV"}]
    pd.DataFrame(rows).to_parquet(d / "metadata.parquet")
    (d / "b2.MOV").write_bytes(b"\x00\x00\x00\x14ftypqt  ")
    return str(tmp_path)


def test_gaia_test_split_loader_keeps_task_ids_and_sends_video(tmp_path, monkeypatch):
    snap = _test_snapshot(tmp_path)
    seen = {}

    def _snap(**kw):
        seen.update(kw)
        return snap

    monkeypatch.setattr("huggingface_hub.snapshot_download", _snap)
    samples = GAIA._load_gaia_samples(split="test")
    assert seen["allow_patterns"] == ["2023/test/*"]
    assert [s[2]["task_id"] for s in samples] == ["a1", "b2"]
    assert {s[1] for s in samples} == {"?"}
    video = samples[1][0][1]["content"][-1]
    assert video["type"] == "video_url"
    assert video["video_url"]["url"].startswith("data:video/quicktime;base64,")
    with pytest.raises(ValueError):
        GAIA._load_gaia_samples(split="dev")


def test_gaia_m4a_needs_ffmpeg(monkeypatch):
    monkeypatch.setattr(GAIA.shutil, "which", lambda name: None)
    with pytest.raises(GAIA.AttachmentDependencyError, match="ffmpeg"):
        GAIA._file_parts("clip.m4a", b"....")


def test_gaia_m4a_is_decoded_to_wav(monkeypatch):
    def fake_run(cmd, **kw):
        with open(cmd[-1], "wb") as fh:
            fh.write(b"RIFF....WAVE")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(GAIA.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(GAIA.subprocess, "run", fake_run)
    parts = GAIA._file_parts("clip.m4a", b"m4a-bytes")
    audio = parts[-1]["input_audio"]
    assert audio["format"] == "wav"
    assert audio["data"] == __import__("base64").b64encode(b"RIFF....WAVE").decode()


def test_gaia_submission_lines():
    traces = [{"sample_idx": 1, "extra_payload": {"task_id": "b2"},
               "response_text": "Thinking.\nFINAL ANSWER: 120", "reasoning": "r"},
              {"sample_idx": 0, "extra_payload": {"task_id": "a1"}, "response_text": None}]
    steps = {1: [("web_search", {"query": "tile price"}, "Tiles cost $8 each.")]}
    lines = GAIA.submission_lines(traces, steps)
    assert [ln["task_id"] for ln in lines] == ["a1", "b2"]
    assert [ln["model_answer"] for ln in lines] == ["", "120"]
    assert set(lines[0]) == {"task_id", "model_answer", "reasoning_trace"}
    trace = lines[1]["reasoning_trace"]
    assert 'Step 1: web_search({"query": "tile price"})' in trace
    assert "Tiles cost $8 each." in trace and "Reasoning:\nr" in trace
    assert trace.endswith("Response:\nThinking.\nFINAL ANSWER: 120")
    with pytest.raises(RuntimeError, match="task_id"):
        GAIA.submission_lines([{"sample_idx": 0, "extra_payload": {}}], {})


def test_gaia_test_run_writes_the_submission_then_hard_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("GBENCH_GAIA_SPLIT", "test")
    monkeypatch.setenv("GBENCH_RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(GAIA, "check_gaia_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(GAIA, "search_available", lambda: True)
    monkeypatch.setattr(GAIA, "gemini_key_live_valid", lambda key: (True, ""))
    samples = [([{"role": "user", "content": "q1"}], "?", {"category": "Level_1", "task_id": "a1"}),
               ([{"role": "user", "content": "q2"}], "?", {"category": "Level_2", "task_id": "b2"})]
    monkeypatch.setattr(GAIA, "_load_gaia_samples", lambda limit=None, split="validation": samples)

    async def fake_tool(name, args):
        return "snippet"

    monkeypatch.setattr(GAIA, "execute_tool", fake_tool)
    got = {}

    def fake_suite(**kw):
        got.update(kw)
        asyncio.run(kw["tool_executor"]("web_search", {"query": "x"}, 1))
        return {"accuracy": 0.0, "correct_answers": 0, "category_accuracy": {},
                "sample_traces": [
                    {"sample_idx": i, "category": s[2]["category"], "extra_payload": s[2],
                     "response_text": f"FINAL ANSWER: {i}"} for i, s in enumerate(samples)]}

    monkeypatch.setattr(GAIA, "run_eval_suite", fake_suite)
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*answers are hidden") as e:
        GAIA.run_gaia("org/model", "http://x", 1, attempt_count=4)
    msg = str(e.value)
    assert got["attempt_count"] == 1 and got["official_sampling"] is GAIA.GAIA_SAMPLING
    assert GAIA.LEADERBOARD_URL in msg and "will reject this subset" in msg
    sub_dir = tmp_path / "gaia_submission"
    [sub] = list(sub_dir.glob("gaia_2023_test_org_model_nothink_*.jsonl"))
    lines = [json.loads(x) for x in sub.read_text().splitlines()]
    assert [(x["task_id"], x["model_answer"]) for x in lines] == [("a1", "0"), ("b2", "1")]
    assert "web_search" in lines[1]["reasoning_trace"]
    [res] = list(sub_dir.glob("*_result.json"))
    saved = json.loads(res.read_text())
    assert saved["split"] == "test" and "accuracy" not in saved
    assert saved["leaderboard_submission"]["tasks"] == 2
    assert str(sub) in msg


def test_gaia_validation_run_names_the_test_split(monkeypatch):
    monkeypatch.delenv("GBENCH_GAIA_SPLIT", raising=False)
    monkeypatch.setattr(GAIA, "check_gaia_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(GAIA, "search_available", lambda: True)
    monkeypatch.setattr(GAIA, "gemini_key_live_valid", lambda key: (True, ""))
    monkeypatch.setattr(GAIA, "_load_gaia_samples", lambda limit=None: [])
    got = {}

    def fake_suite(**kw):
        got.update(kw)
        return {"sample_traces": [], "temperature": None,
                "official_sampling": {"temperature_matches_official": True}}

    monkeypatch.setattr(GAIA, "run_eval_suite", fake_suite)
    res = GAIA.run_gaia("m", "http://x", 1)
    assert res["split"] == "validation" and got["official_sampling"] is GAIA.GAIA_SAMPLING
    assert "GBENCH_GAIA_SPLIT=test" in res["leaderboard_comparable_reason"]
    doc = _doc("gaia.md")
    assert "GBENCH_GAIA_SPLIT=test" in doc and "task_id`, `model_answer`" in doc
    assert "9f133d71362e77b3539f1514f31b9c101a545fec" in doc


# --------------------------------------------------------------------------- #
# cimemories: release texts, budget-cut generations, shared label cache
# --------------------------------------------------------------------------- #
_PROMPTS = (b"task_solving:\n  1: |\n    # Memories\n\n    {memories}\n\n"
            b"    Write to {recipient}: {task}.\n"
            b"adversary_reveal:\n  1: |\n    ATTRIBUTES:\n    {attributes}\n\n"
            b"    SOLUTION:\n    {solution}\n")
_PERSONA_SRC = '''
raise SystemExit("this file is parsed, never executed")
personas = [
    """
    **The Privacy Fundamentalist**
    Private.
    """,
    """
    **The Pragmatic**
    Balanced.
    """,
    """
    *The Unconcerned**

    Open.
    """,
]


def label_attributes(model, persona, profile, context, n=3):
    memories = "\\n".join(
        [x["memory_statement"] for x in profile["information_attributes"].values()]
    )
    attribute_sorting_prompt = f"""
    You have the following persona:
    {persona}

    To {context['recipient']} for {context['task']}.
    INFORMATION:
    {{memories}}

    Respond with JSON.
    """
    attribute_sorting_prompt = (
        dedent(attribute_sorting_prompt).strip().format(memories=memories)
    )
    return model.raw(attribute_sorting_prompt, n=n)


labels = label_attributes(model, persona, profile, context, n=10)
weights = {"fundamentalist": 0.26, "pragmatic": 0.64, "unconcerned": 0.10}
'''
_PROFILE = {"bio": {"name": "Ann", "age": 40},
            "information_attributes": {
                "age": {"value": 40, "memory_statement": "Ann is 40."},
                "debt": {"value": "$9k", "memory_statement": "Ann owes $9k."},
                "job": {"value": "nurse", "memory_statement": "Ann is a nurse."}},
            "contexts": [{"recipient": "Doctor", "task": "Book a visit"}]}


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def test_cimemories_release_texts_parse_like_the_release_code():
    texts = CIM.parse_release_texts(_PROMPTS, _PERSONA_SRC.encode())
    assert texts.task_prompt.startswith("# Memories\n\n{memories}")
    assert texts.reveal_prompt.endswith("SOLUTION:\n{solution}\n")
    assert len(texts.personas) == 3
    # the reconstructed f-string, dedented, equals the release function's own prompt
    fn = next(n for n in ast.parse(_PERSONA_SRC).body
              if isinstance(n, ast.FunctionDef) and n.name == "label_attributes")
    ns = {"dedent": textwrap.dedent}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<stand-in>", "exec"), ns)

    class _Model:
        def raw(self, prompt, n):
            return prompt

    with mock.patch.object(CIM, "_release_texts", lambda: texts):
        for persona in texts.personas:
            ctx = _PROFILE["contexts"][0]
            assert CIM._label_prompt(persona, _PROFILE, ctx) == ns["label_attributes"](
                _Model(), persona, _PROFILE, ctx)


def test_cimemories_release_source_must_match_the_implemented_protocol():
    bad_weights = _PERSONA_SRC.replace('"pragmatic": 0.64', '"pragmatic": 0.6')
    with pytest.raises(RuntimeError, match="weights"):
        CIM.parse_release_texts(_PROMPTS, bad_weights.encode())
    bad_samples = _PERSONA_SRC.replace("n=10)", "n=5)")
    with pytest.raises(RuntimeError, match="samples"):
        CIM.parse_release_texts(_PROMPTS, bad_samples.encode())
    bad_field = _PERSONA_SRC.replace("{context['task']}", "{context['task'].upper()}")
    with pytest.raises(RuntimeError, match="placeholder"):
        CIM.parse_release_texts(_PROMPTS, bad_field.encode())
    with pytest.raises(RuntimeError, match="task_solving"):
        CIM.parse_release_texts(b"task_solving: {}\n", _PERSONA_SRC.encode())


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_cimemories_release_files_are_fetched_checked_and_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))
    files = {"eval/prompts.yaml": _PROMPTS, "data/persona_gold_labels.py": _PERSONA_SRC.encode()}
    monkeypatch.setattr(CIM, "_RELEASE_FILES", {p: _sha(b) for p, b in files.items()})
    urls = []

    def fake_open(url, timeout=None):
        urls.append(url)
        return _Resp(files[url.split(CIM._UPSTREAM_COMMIT + "/", 1)[1]])

    monkeypatch.setattr(CIM.urllib.request, "urlopen", fake_open)
    CIM._release_texts.cache_clear()
    try:
        texts = CIM._release_texts()
        assert texts.personas and len(urls) == 2
        assert all(u.startswith("https://raw.githubusercontent.com/facebookresearch/CIMemories/"
                                + CIM._UPSTREAM_COMMIT + "/") for u in urls)
        cached = tmp_path / "cimemories_release" / CIM._UPSTREAM_COMMIT / "eval" / "prompts.yaml"
        assert cached.read_bytes() == _PROMPTS
        assert CIM._release_file("eval/prompts.yaml") == _PROMPTS and len(urls) == 2  # cache
        cached.write_bytes(b"tampered")
        assert CIM._release_file("eval/prompts.yaml") == _PROMPTS and len(urls) == 3  # refetch
        files["eval/prompts.yaml"] = b"different upstream bytes"
        cached.unlink()
        with pytest.raises(RuntimeError, match="sha256"):
            CIM._release_file("eval/prompts.yaml")

        def offline(url, timeout=None):
            raise OSError("network is unreachable")

        monkeypatch.setattr(CIM.urllib.request, "urlopen", offline)
        with pytest.raises(RuntimeError, match="INFRA REQUIRED.*prompts.yaml.*place that file"):
            CIM._release_file("eval/prompts.yaml")
    finally:
        CIM._release_texts.cache_clear()


def _cim_traces():
    base_trace = {"gold_answer": "p0c0", "category": "Ann", "status": "PENDING_JUDGE"}
    return [dict(base_trace, sample_idx=0, attempt_idx=0, response_text="Dear doctor, I am 40.",
                 finish_reason="stop", health="ok"),
            dict(base_trace, sample_idx=1, attempt_idx=1, response_text="",
                 finish_reason="length", health="truncated"),
            dict(base_trace, sample_idx=2, attempt_idx=2, response_text="Dear doctor, I owe",
                 finish_reason="length", health="truncated")]


def test_cimemories_budget_cut_generations_are_counted_and_the_empty_ones_left_out(
        tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))
    texts = CIM.parse_release_texts(_PROMPTS, _PERSONA_SRC.encode())
    monkeypatch.setattr(CIM, "_release_texts", lambda: texts)
    reveals = []

    async def cascade(prompt, config=None):
        if prompt.startswith("You have the following persona"):
            return '{"share": ["Ann is a nurse."], "private": ["Ann owes $9k."]}', "gemini-x"
        reveals.append(prompt)
        return '{"debt": "I owe"}', "gemini-x"

    monkeypatch.setattr(CIM, "judge_generate_cascade", cascade)
    traces = _cim_traces()
    state = CIM._new_state()
    asyncio.run(CIM._async_judge_cimemories(traces, profiles=(_PROFILE,), state=state))
    assert len(reveals) == 2, "the empty budget-cut generation is not judged"
    assert traces[1]["cimemories_budget_exhausted"] and "revealed_attributes" not in traces[1]
    assert traces[2]["cimemories_truncated"] and traces[2]["revealed_private"] == ["debt"]
    rows = CIM._metric_rows(traces, (_PROFILE,), state["context_labels"])
    assert sorted({r[3] for r in rows}) == [0, 2]
    result = {"sample_traces": traces}
    CIM._score(result, (_PROFILE,), state)
    rep = result["cimemories_report"]
    assert rep["trials_budget_exhausted"] == 1 and rep["trials_truncated"] == 1
    assert rep["trials_scored"] == 2


def test_cimemories_run_names_budget_cuts_and_sends_no_temperature(monkeypatch):
    monkeypatch.setattr(CIM, "gemini_required_skip", lambda *a: None)
    monkeypatch.setattr(CIM, "_load_profiles", lambda: (_PROFILE,))
    monkeypatch.setattr(CIM, "_load_cimemories_samples", lambda limit=None: [])
    got = {}

    def fake_suite(**kw):
        got.update(kw)
        return {"sample_traces": [], "temperature": None,
                "temperature_source": "official:server_default"}

    def fake_score(result, profiles, state):
        result["cimemories_report"] = {
            "contexts_generated": 490, "trials_without_response": 0,
            "trials_reveal_judge_outage": 0, "trials_label_outage": 0,
            "reveal_judge_unparsed": 0, "trials_budget_exhausted": 3, "trials_truncated": 2}

    monkeypatch.setattr(CIM, "run_eval_suite", fake_suite)
    monkeypatch.setattr(CIM, "_score", fake_score)
    res = CIM.run_cimemories("m", "http://x")
    assert got["official_sampling"] is CIM.CIMEMORIES_SAMPLING
    assert CIM.CIMEMORIES_SAMPLING.send_temperature is False and "temperature" not in got
    reason = res["leaderboard_comparable_reason"]
    assert ("3 generations that reached the end of the context window before writing any "
            "message") in reason
    assert "2 messages cut off at the end of the context window" in reason
    assert CIM.CIMEMORIES_SAMPLING.send_max_tokens is False
    assert "set by" not in reason and "greedy" not in reason
    assert res["protocol"]["release_texts"]["eval/prompts.yaml"]["sha256"] == (
        CIM._RELEASE_FILES["eval/prompts.yaml"])


def test_cimemories_operator_temperature_is_named(monkeypatch):
    monkeypatch.setattr(CIM, "gemini_required_skip", lambda *a: None)
    monkeypatch.setattr(CIM, "_load_profiles", lambda: (_PROFILE,))
    monkeypatch.setattr(CIM, "_load_cimemories_samples", lambda limit=None: [])
    monkeypatch.setattr(CIM, "run_eval_suite", lambda **kw: {
        "sample_traces": [], "temperature": 0.7,
        "temperature_source": "env:GBENCH_CIMEMORIES_TEMPERATURE"})
    res = CIM.run_cimemories("m", "http://x", limit=1)
    assert ("temperature 0.7 set by GBENCH_CIMEMORIES_TEMPERATURE"
            in res["leaderboard_comparable_reason"])


def test_cimemories_label_cache_entry_is_labeled_once_by_concurrent_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))
    calls = []

    async def cascade(prompt, config=None):
        calls.append(prompt)
        await asyncio.sleep(0.05)
        return f"reply {len(calls)}", "gemini-x"

    monkeypatch.setattr(CIM, "judge_generate_cascade", cascade)

    async def two_runs():
        # two caches on the same directory, as two processes sharing GBENCH_CACHE_DIR have
        caches = [CIM._LabelCache(CIM._labeler_identity()) for _ in range(2)]
        stats = [CIM._new_state()["stats"] for _ in range(2)]
        sem = asyncio.Semaphore(8)
        return await asyncio.gather(*[
            CIM._persona_samples("the labeling prompt", caches[i], sem, None, stats[i])
            for i in range(2)])

    a, b = asyncio.run(two_runs())
    assert len(calls) == CIM._LABEL_SAMPLES
    assert a == b and len(a) == CIM._LABEL_SAMPLES


def test_cimemories_ships_no_release_text_and_docs_say_where_it_comes_from():
    src = open(CIM.__file__, encoding="utf-8").read()
    assert "Inferred from past conversations" not in src
    assert "Fundamentalists are generally" not in src
    doc = _doc("cimemories.md")
    for needle in CIM._RELEASE_FILES.values():
        assert needle in doc
    assert "trials_budget_exhausted" in doc and "no temperature" in doc
    assert "cimemories_release/" in doc and "--max-output-tokens" in doc


# --------------------------------------------------------------------------- #
# gdpval / deepsearch_qa: no temperature; the rater's own config
# --------------------------------------------------------------------------- #
def test_gdpval_sends_no_temperature_and_names_an_override(monkeypatch):
    got = {}

    def fake_suite(**kw):
        got.update(kw)
        return {"total_questions": 0, "temperature": 0.5, "temperature_source": "cli:--temperature"}

    monkeypatch.setattr(G, "gemini_required_skip", lambda *a: None)
    monkeypatch.setattr(G, "_load_gdpval_samples", lambda limit=None: ([], {}))
    monkeypatch.setattr(G, "run_eval_suite", fake_suite)
    res = G.run_gdpval("m", "http://x", 1)
    assert got["official_sampling"] is G.GDPVAL_SAMPLING
    assert G.GDPVAL_SAMPLING.send_temperature is False
    assert "temperature 0.5 set by --temperature" in res["leaderboard_comparable_reason"]
    assert "**1.0 with `--thinking`, 0.0 without**" in _doc("gdpval.md")


def test_deepsearch_qa_sends_no_temperature_and_rates_with_the_notebook_config(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(DSQA, "gemini_key_live_valid", lambda k: (True, ""))
    monkeypatch.setattr(DSQA, "_load_deepsearch_qa_samples", lambda limit=None: [])
    got = {}

    def fake_suite(**kw):
        got.update(kw)
        return {"sample_traces": [], "temperature": None,
                "official_sampling": {"temperature_matches_official": True}}

    monkeypatch.setattr(DSQA, "run_eval_suite", fake_suite)
    res = DSQA.run_deepsearch_qa("m", "http://x", concurrency=1)
    assert got["official_sampling"] is DSQA.DSQA_SAMPLING
    assert "temperature" not in res["leaderboard_comparable_reason"]
    cfg = DSQA._rater_config()
    assert cfg is not None and cfg.temperature is None
    seen = []

    async def judge(prompt, config=None):
        seen.append(config)
        return None, "x"

    tr = [{"response_text": "x", "gold_answer": "y", "messages": [{"content": "q"}],
           "extra_payload": {"answer_type": "Single Answer"}}]
    with mock.patch.object(DSQA, "judge_generate_cascade", judge):
        asyncio.run(DSQA._async_judge_deepsearch_qa(tr))
    assert seen and seen[0].temperature is None


# --------------------------------------------------------------------------- #
# toolbench: ToolEval as the StableToolBench scripts run it
# --------------------------------------------------------------------------- #
def test_toolbench_prompts_are_the_template_descriptions():
    for name, text in T._TEMPLATES.items():
        assert text.startswith("\n") and text.endswith("\n"), name
        assert "End your reply" not in text
    assert T._CHECK_ANSWER_STATUS.endswith("of JSON to `check_answer_status`.\n")
    assert T._JUDGE_TEMPERATURE == 0.2 and T._JUDGE_ATTEMPTS == 3


def test_toolbench_judge_config_is_the_function_schema():
    cfg = T._judge_generation_config("check_answer_status", True)
    assert cfg.temperature == 0.2
    schema = cfg.tools[0].function_declarations[0].parameters_json_schema
    assert schema["required"] == ["answer_status", "reason"]
    assert schema["properties"]["answer_status"]["enum"] == ["Unsure", "Unsolved", "Solved"]
    assert T._function_schema("check_answer_status", False)["required"] == ["answer_status"]
    assert T._FUNCTION_PARAMETERS["check_answer_status"]["required"] == ["answer_status"]


def test_toolbench_invalid_replies_are_retried_then_left_out(monkeypatch):
    replies = iter(['{"answer_status": "Maybe", "reason": "r"}', "not json",
                    '{"reason": "no status"}'])

    async def cascade(prompt, config=None):
        return next(replies), "gemini"

    monkeypatch.setattr(T, "_tooleval_cascade", cascade)
    answer = {"final_answer": "x", "answer_details": [
        {"role": "tool", "message": {"name": "Finish"}, "next": []}]}
    verdicts = T._Verdicts()

    async def go():
        T._VERDICTS.set(verdicts)
        return await T._answer_status("q", answer, asyncio.Semaphore(1))

    assert asyncio.run(go()) is None, "never scored Unsolved"
    assert verdicts.invalid == 1 and verdicts.outages == 0


def _ans(final, method="DFS"):
    return {"final_answer": final, "method": method,
            "answer_details": [{"role": "tool", "message": "x", "next": []}]}


@pytest.mark.parametrize("cand,ref,winner", [
    ("Solved", "Unsure", "cand"), ("Unsure", "Unsolved", "cand"),
    ("Unsolved", "Solved", "ref"), ("Unsure", "Solved", "ref")])
def test_toolbench_preference_takes_the_single_solved_then_unsure_answer(
        monkeypatch, cand, ref, winner):
    status = {"C": cand, "R": ref}

    async def pref_status(query, answer, sem):
        return status[answer["final_answer"]]

    async def no_select(*a, **k):
        raise AssertionError("select_better_answer is not needed here")

    monkeypatch.setattr(T, "_preference_status", pref_status)
    monkeypatch.setattr(T, "_select_better", no_select)
    got = asyncio.run(T._preference("q", _ans("C"), _ans("R"), asyncio.Semaphore(1)))
    assert got == winner


def test_toolbench_preference_asks_select_better_with_processed_answers(monkeypatch):
    async def pref_status(query, answer, sem):
        assert len(answer["final_answer"]) <= 1000 and "method" not in answer
        return "Unsolved"

    prompts = []

    async def cascade(prompt, config=None):
        prompts.append(prompt)
        return '{"index": 1, "reason": "r"}', "gemini"

    monkeypatch.setattr(T, "_preference_status", pref_status)
    monkeypatch.setattr(T, "_tooleval_cascade", cascade)
    cand, ref = _ans("C" * 1500), _ans("R")
    got = asyncio.run(T._preference("q", cand, ref, asyncio.Semaphore(1)))
    assert len(prompts) == 1 and '"method"' not in prompts[0]
    second = prompts[0].split("Answer_1:\n", 1)[1]
    assert got == ("cand" if second.startswith('{"final_answer": "CCC') else "ref")
    # identical answers: a random pick, no judge call
    prompts.clear()
    got = asyncio.run(T._preference("q", _ans("S"), _ans("S"), asyncio.Semaphore(1)))
    assert got in ("cand", "ref") and prompts == []


def test_toolbench_select_better_retries_an_out_of_range_index(monkeypatch):
    replies = iter(['{"index": 2, "reason": "r"}', '{"index": 0, "reason": "r"}'])

    async def cascade(prompt, config=None):
        return next(replies), "gemini"

    monkeypatch.setattr(T, "_tooleval_cascade", cascade)
    assert asyncio.run(T._select_better("q", _ans("a"), _ans("b"), asyncio.Semaphore(1))) == 0


def test_toolbench_sowr_counts_every_query_of_the_group(monkeypatch):
    async def pref(query, cand, ref, sem):
        return "cand"

    async def status(query, answer, sem):
        return "Solved"

    monkeypatch.setattr(T, "_preference", pref)
    monkeypatch.setattr(T, "_answer_status", status)
    cand = {"1": {"query": "q", "answer": _ans("a")}, "2": {"query": "q", "answer": _ans("b")}}
    ref = {"1": {"query": "q", "answer": _ans("c")}, "3": {"query": "q", "answer": _ans("d")}}
    r = asyncio.run(T._score_all(cand, ref, 3, 1))
    # 1: candidate wins every pass; 2: only the candidate answered (win); 3: only the reference
    # answered (loss) -> 2 of 3
    assert r["sowr"] == pytest.approx(2 / 3)
    assert r["judge_outages"] == 0 and r["judge_invalid_replies"] == 0
    doc = _doc("toolbench.md")
    assert "temperature **0.2**" in doc and "function calling" in doc
    assert "--use_pass_rate true" in doc


# --------------------------------------------------------------------------- #
# mcp_bench: every server serves, or the run does not start
# --------------------------------------------------------------------------- #
_ALL_KEYS = {k: "set" for k in MB._REQUIRED_KEYS}


def test_mcp_bench_gaps_name_keys_and_servers():
    probe = {"servers": ["A", "Google Maps", "Math MCP"], "unavailable": ["Google Maps", "Math MCP"],
             "unbuilt": ["Math MCP"], "network_up": True}
    gaps = MB.provisioning_gaps(probe, env_keys={})
    assert gaps[0].startswith("required API keys unset: NPS_API_KEY (National Parks)")
    assert "NCI_API_KEY (BioMCP)" in gaps[0]
    assert gaps[1] == "servers that did not build in the image: Math MCP"
    assert gaps[2] == ("servers the launcher cannot serve (no bridge to gbench's "
                       "credentialed-servers container): Google Maps")
    down = {"unavailable": ["Wikipedia"], "unbuilt": [], "network_up": False}
    assert "no network egress" in MB.provisioning_gaps(down, env_keys=_ALL_KEYS)[0]
    assert MB.provisioning_gaps({"unavailable": [], "unbuilt": []}, env_keys=_ALL_KEYS) == []


def test_mcp_bench_probe_passes_keys_by_name_and_reads_its_line(monkeypatch):
    monkeypatch.setenv("NASA_API_KEY", "secret")
    for k in ("NPS_API_KEY", "HF_TOKEN", "GOOGLE_MAPS_API_KEY", "NCI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        out = 'noise\nMCP_BENCH_PROBE {"servers": [], "unavailable": [], "unbuilt": [], ' \
              '"network_up": true, "connections": {}}\n'
        return subprocess.CompletedProcess(cmd, 0, out, "")

    monkeypatch.setattr(MB.subprocess, "run", fake_run)
    probe = MB.provisioning_probe("img", {"MCP_CREDENTIALED_RELAY": "172.17.0.1:9",
                                          "MCP_CREDENTIALED_TOKEN": "tok"})
    assert probe["network_up"] is True
    cmd = seen["cmd"]
    names = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-e"]
    # no server key enters the probe container: the credentialed servers are reached through
    # gbench's relay, whose address and token go by name
    assert "NASA_API_KEY" not in names and "secret" not in " ".join(cmd)
    assert "MCP_CREDENTIALED_RELAY" in names and "MCP_CREDENTIALED_TOKEN" in names
    # the host names the launcher resolves in its network check
    assert "MCP_DNS_NAMES=" + ",".join(MB.server_dns_names()) in names
    assert "--entrypoint" in cmd and "img" in cmd
    monkeypatch.setattr(MB.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "boom"))
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*rebuild the image"):
        MB.provisioning_probe("img")


def _mb_run(monkeypatch, probe, manifest=None, no_subset=False):
    monkeypatch.setattr(MB, "check_mcp_bench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(MB, "provisioning_probe", lambda image, relay_env=None: probe)
    # the credentialed-servers container (tests/test_mcp_wildclawbench_credentials_and_cascade.py
    # covers it): a stand-in
    monkeypatch.setattr(MB, "start_credentialed_servers",
                        lambda image, base, network: types.SimpleNamespace(
                            client_env=lambda: {"MCP_CREDENTIALED_RELAY": "172.17.0.1:10",
                                                "MCP_CREDENTIALED_TOKEN": "t0k"},
                            stop=lambda: None, record=lambda: {}))
    for k in MB._REQUIRED_KEYS:
        monkeypatch.delenv(k, raising=False) if probe.get("_no_keys") else monkeypatch.setenv(k, "v")
    if no_subset:
        monkeypatch.setenv("GBENCH_MCP_BENCH_NO_SUBSET", "1")
    else:
        monkeypatch.delenv("GBENCH_MCP_BENCH_NO_SUBSET", raising=False)
    ran = []

    def fake_run(cmd, **kw):
        if cmd[:2] == ["docker", "run"]:
            ran.append(cmd)
            out = cmd[cmd.index("-v") + 1].split(":")[0]
            metrics = {"input_schema_compliance": 1.0, "valid_tool_name_rate": 1.0,
                       "tool_call_success_rate": 1.0, "task_completion_score": 5.0,
                       "tool_selection_score": 5.0,
                       "planning_effectiveness_and_efficiency_score": 5.0}
            with open(os.path.join(out, "mcp_bench_results.json"), "w") as fh:
                json.dump(metrics, fh)
            with open(os.path.join(out, "subset_manifest.json"), "w") as fh:
                json.dump(manifest or {"filtered": True, "unavailable_servers": [],
                                       "tasks_kept": 104}, fh)
            with open(os.path.join(out, MB.HARNESS_LOG), "w") as fh:
                json.dump({"version": MB.HARNESS_LOG_VERSION, "tasks": [
                    {"task_id": "t1", "file": "f.json", "status": "completed", "kinds": []}]}, fh)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(MB.subprocess, "run", fake_run)
    # the judge's Gemini relay (tests/test_mcp_credential_isolation_and_dns.py covers it): a
    # stand-in, docker is stubbed
    monkeypatch.setattr(MB, "_start_judge_relay", lambda network: types.SimpleNamespace(
        client_env=lambda: {"GEMINI_API_KEY": "gbench-gemini-relay-" + "0" * 48},
        stop=lambda: None, record=lambda: {}))
    return ran, (lambda: MB.run_mcp_bench("m", "http://x/v1"))


def test_mcp_bench_refuses_to_start_with_a_server_that_cannot_serve(monkeypatch):
    probe = {"_no_keys": True, "unavailable": ["Google Maps"], "unbuilt": [], "network_up": True}
    ran, go = _mb_run(monkeypatch, probe)
    # the unset keys stop the run before anything starts
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*required API keys unset.*Google Maps"):
        go()
    assert ran == [], "no container run for a partial task set"


def test_mcp_bench_full_provisioning_runs_and_a_late_drop_is_not_scored(monkeypatch):
    ok = {"unavailable": [], "unbuilt": [], "network_up": True}
    ran, go = _mb_run(monkeypatch, ok)
    res = go()
    assert len(ran) == 1 and res["is_full"] is True and res["accuracy"] > 0
    dropped = {"filtered": True, "unavailable_servers": ["Reddit"], "network_up": True}
    ran, go = _mb_run(monkeypatch, ok, manifest=dropped)
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*could not serve.*Reddit"):
        go()


def test_mcp_bench_no_subset_variable_does_not_bypass_the_pre_run_check(monkeypatch):
    # A missing prerequisite is a hard error; GBENCH_MCP_BENCH_NO_SUBSET no longer runs the
    # tasks anyway.
    probe = {"_no_keys": True, "unavailable": ["NASA Data"], "unbuilt": [], "network_up": True}
    ran, go = _mb_run(monkeypatch, probe, manifest={"filtered": False}, no_subset=True)
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*required API keys unset.*NASA Data"):
        go()
    assert ran == []
    assert "NO_SUBSET" not in open(MB.__file__, encoding="utf-8").read()
    doc = _doc("mcp_bench.md")
    assert "NCI_API_KEY" in doc and "never runs the tasks of the provisioned servers alone" in doc
    assert "GBENCH_MCP_BENCH_NO_SUBSET" not in doc


# --------------------------------------------------------------------------- #
# mcp_atlas: the release's MongoDB data in a local container
# --------------------------------------------------------------------------- #
class _Docker:
    """Records docker commands; answers like a docker daemon where it matters."""

    def __init__(self, fail=()):
        self.cmds = []
        self.fail = set(fail)

    def __call__(self, cmd, **kw):
        self.cmds.append(cmd)
        key = " ".join(cmd[:3])
        if any(key.startswith(f) for f in self.fail):
            return subprocess.CompletedProcess(cmd, 125, "", "boom")
        if cmd[:2] == ["docker", "exec"]:
            return subprocess.CompletedProcess(cmd, 0, "1\n", "")
        if cmd[:3] == ["docker", "inspect", "-f"]:
            return subprocess.CompletedProcess(cmd, 0, "true\n", "")
        return subprocess.CompletedProcess(cmd, 0, "id\n", "")


def _atlas_credentials(monkeypatch):
    """Every credentialed server's variables set (mcp_atlas starts nothing without them)."""
    for server in M.CREDENTIALED_SERVERS:
        for var in M.ENVIRONMENT_SERVERS[server]:
            if var != "MONGODB_CONNECTION_STRING":
                monkeypatch.setenv(var, f"{var.lower()}-value-1234")


def test_mcp_atlas_starts_the_release_mongodb_next_to_the_environment(monkeypatch):
    monkeypatch.delenv("MONGODB_CONNECTION_STRING", raising=False)
    monkeypatch.delenv("GBENCH_MCP_ATLAS_MONGO_IMAGE", raising=False)
    _atlas_credentials(monkeypatch)
    docker = _Docker()
    monkeypatch.setattr(M.subprocess, "run", docker)
    sb = M._start_container("gbench-mcp-atlas:f24ba3fb")
    net_create, cred_net_create, mongo_run, ping, cred_run, env_run = docker.cmds
    assert net_create[:3] == ["docker", "network", "create"] and net_create[3] == sb.network
    assert cred_net_create[:3] == ["docker", "network", "create"]
    assert cred_net_create[3] == sb.cred_network
    # the release's MongoDB data next to the credentialed servers (the mongodb server's home)
    assert mongo_run[mongo_run.index("--network") + 1] == sb.cred_network
    assert mongo_run[mongo_run.index("--network-alias") + 1] == "mongo"
    assert mongo_run[-1] == "gbench-mcp-atlas-mongo:f24ba3fb" and mongo_run[4] == sb.mongo_container
    assert ping[:3] == ["docker", "exec", sb.mongo_container] and "mongosh" in ping
    assert cred_run[cred_run.index("--network") + 1] == sb.cred_network
    assert "MONGODB_CONNECTION_STRING=mongodb://mongo:27017/" in cred_run
    assert env_run[env_run.index("--network") + 1] == sb.network
    assert not any("MONGODB" in a for a in env_run)
    docker.cmds.clear()
    M._stop_container(sb)
    assert docker.cmds == [["docker", "rm", "-f", sb.container],
                           ["docker", "rm", "-f", sb.cred_container],
                           ["docker", "rm", "-f", sb.mongo_container],
                           ["docker", "network", "rm", sb.network],
                           ["docker", "network", "rm", sb.cred_network]]


def test_mcp_atlas_operator_mongodb_needs_no_container(monkeypatch):
    monkeypatch.setenv("MONGODB_CONNECTION_STRING", "mongodb+srv://u:p@cluster/")
    _atlas_credentials(monkeypatch)
    docker = _Docker()
    monkeypatch.setattr(M.subprocess, "run", docker)
    sb = M._start_container("gbench-mcp-atlas:f24ba3fb")
    runs = [c for c in docker.cmds if c[:2] == ["docker", "run"]]
    [cred_run, env_run] = runs
    assert sb.mongo_container is None and not any("mongo" == c[-1] for c in runs)
    assert cred_run[cred_run.index("MONGODB_CONNECTION_STRING") - 1] == "-e"
    assert "MONGODB_CONNECTION_STRING" not in env_run
    assert "u:p" not in " ".join(cred_run + env_run), "the operator's value stays off argv"


def test_mcp_atlas_failed_environment_start_removes_the_mongodb_container(monkeypatch):
    monkeypatch.delenv("MONGODB_CONNECTION_STRING", raising=False)
    _atlas_credentials(monkeypatch)
    docker = _Docker()

    def run(cmd, **kw):
        if cmd[:3] == ["docker", "run", "-d"] and "-p" in cmd:      # the environment container
            docker.cmds.append(cmd)
            return subprocess.CompletedProcess(cmd, 125, "", "port is already allocated")
        return docker(cmd, **kw)

    monkeypatch.setattr(M.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="port is already allocated"):
        M._start_container("gbench-mcp-atlas:f24ba3fb")
    env_rm, cred_rm, mongo_rm, net_rm, cred_net_rm = docker.cmds[-5:]
    assert env_rm[:3] == ["docker", "rm", "-f"] and not env_rm[3].endswith(("-mongo", "-cred"))
    assert cred_rm[:3] == ["docker", "rm", "-f"] and cred_rm[3].endswith("-cred")
    assert mongo_rm[:3] == ["docker", "rm", "-f"] and mongo_rm[3].endswith("-mongo")
    assert net_rm[:3] == ["docker", "network", "rm"] and net_rm[3].endswith("-net")
    assert cred_net_rm[:3] == ["docker", "network", "rm"] and cred_net_rm[3].endswith("-cred-net")


def test_mcp_atlas_prerequisites_name_the_mongodb_build(monkeypatch):
    monkeypatch.delenv("GBENCH_MCP_ATLAS_SANDBOX_URL", raising=False)
    monkeypatch.delenv("MONGODB_CONNECTION_STRING", raising=False)
    monkeypatch.setattr(M.shutil, "which", lambda name: "/usr/bin/docker")

    def run(cmd, **kw):
        if cmd[:3] == ["docker", "image", "inspect"]:
            if cmd[-1] == "gbench-mcp-atlas:f24ba3fb":
                value = (M.IMAGE_SEARCH_LABEL_VALUE if M.IMAGE_SEARCH_LABEL in cmd[4]
                         else M.UPSTREAM_REVISION)
                return subprocess.CompletedProcess(cmd, 0, value + "\n", "")
            return subprocess.CompletedProcess(cmd, 1, "", "No such image")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(M.subprocess, "run", run)
    ok, reason = M.check_mcp_atlas_prerequisites()
    assert not ok and ("docker build --target mongo -t gbench-mcp-atlas-mongo:f24ba3fb "
                       "-f docker/mcp_atlas.Dockerfile docker") in reason
    monkeypatch.setenv("MONGODB_CONNECTION_STRING", "mongodb://db:27017/")
    assert M.check_mcp_atlas_prerequisites() == (True, "")


def test_mcp_atlas_stopped_mongodb_container_means_a_lost_environment(monkeypatch):
    monkeypatch.setattr(M, "_container_running", lambda name: not name.endswith("-mongo"))
    sb = M._Sandbox("http://127.0.0.1:9", container="env", mongo_container="env-mongo")
    alive, why = M.environment_alive(sb)
    assert not alive and "MongoDB container env-mongo" in why
    note = M.preflight_failure_reason({s: (s != "mongodb", "Error: x") for s in M.RELEASE_SERVERS},
                                      True, provided=("MONGODB_CONNECTION_STRING",))
    assert "mongodb (served by gbench's local MongoDB container)" in note
    assert "unset here: MONGODB_CONNECTION_STRING" not in note


def test_mcp_atlas_dockerfile_builds_the_release_mongodb_dump():
    df = open(os.path.join(_HERE, "..", "docker", "mcp_atlas.Dockerfile"), encoding="utf-8").read()
    assert ("mongo:8.0.32@sha256:"
            "4968f22d0c6c10ef29952f3e807f62872ba22b3312f25803564fbfc08255efc2") in df
    assert f"ARG MONGO_DUMP_SHA256={M.MONGO_DUMP_SHA256}" in df
    assert f'LABEL {M.MONGO_REVISION_LABEL}="${{MCP_ATLAS_REF}}"' in df
    assert "mongo_dump_video_game_store-UNZIP-FIRST.zip" in df and "sha256sum -c" in df
    # the mongo target comes first, so a plain build still produces the environment image
    assert df.index("AS mongo") < df.index("FROM ${UV_BASE_IMAGE}")
    doc = _doc("mcp_atlas.md")
    for needle in ("docker build --target mongo -t gbench-mcp-atlas-mongo:f24ba3fb -f "
                   "docker/mcp_atlas.Dockerfile docker", "GBENCH_MCP_ATLAS_MONGO_IMAGE",
                   M.MONGO_DUMP_SHA256, "16,764"):
        assert needle in doc, needle


@pytest.fixture(autouse=True)
def _metadata_guard_answers_blocked(monkeypatch):
    """Docker is stubbed in these tests: the cloud metadata guard the suites call before starting a
    container (container_guard.require_metadata_blocked) answers "unreachable" without a probe.
    tests/test_guard_tool_suites.py checks the suites' calls to it."""
    from gbench.runners.eval_suites import container_guard
    monkeypatch.setattr(container_guard, "require_metadata_blocked",
                        lambda *a, **k: container_guard.UNREACHABLE)
