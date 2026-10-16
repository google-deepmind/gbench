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

"""Agent and tool suites: tool-loop options, attachment sizing, SoWR and ToolEval references, and
the mcp_bench connection check.

  * mcp_atlas - the harness's tool-loop options and sampling; the pre-flight hint and the doc
                make no claim that the environment serializes tool calls (it does not).
  * gaia      - a video attachment is sized as media, not as its base64; the submission note
                tells over-context tasks apart from failed requests.
  * toolbench - SoWR votes over the test ids the run covered (the first N under --eval-limit);
                ToolEval's function (name, description, parameters) reaches the judge through
                forced function calling; the published ChatGPT-CoT answers are the default
                reference, pinned, checked, converted and cached.
  * mcp_bench - the pre-run check forwards GBENCH_MCP_BENCH_ASSUME_NETWORK and runs MCP-Bench's
                own connection check (28/28 servers connected, each with tools).

No network, no model server, no judge API: docker, the hub and the Gemini client are stubbed.
"""

import asyncio
import hashlib
import json
import os
import subprocess
import threading
import types
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import gaia as GAIA
from gbench.runners.eval_suites import mcp_atlas as M
from gbench.runners.eval_suites import mcp_bench as MB
from gbench.runners.eval_suites import toolbench as T

_HERE = os.path.dirname(__file__)


def _doc(name):
    with open(os.path.join(_HERE, "..", "docs", "evals", name), encoding="utf-8") as fh:
        return " ".join(fh.read().split())


# --------------------------------------------------------------------------- #
# mcp_atlas
# --------------------------------------------------------------------------- #
def test_mcp_atlas_uses_the_release_harness_loop_options_and_sampling():
    loop = M.MCP_ATLAS_TOOL_LOOP
    assert loop.invalid_arguments_as_error is True
    assert loop.max_sample_wall_s == 1800.0 == M.TASK_TIMEOUT_S
    assert (loop.max_rounds, loop.max_tool_calls, loop.forced_final) == (256, 100, False)
    assert M.MCP_ATLAS_SAMPLING.send_temperature is False
    doc = _doc("mcp_atlas.md")
    for needle in ("not valid JSON is not sent", "**1800 s**", "no temperature",
                   "`wall_clock_limited`", "`tool_argument_errors`"):
        assert needle in doc, needle


def test_mcp_atlas_preflight_timeout_hint_names_the_check_timeout_not_serialization():
    results = {s: (True, "") for s in M.RELEASE_SERVERS}
    results["arxiv"] = (False, "Timed out after 30s")
    msg = M.preflight_failure_reason(results, local_env=False, enabled=list(M.RELEASE_SERVERS))
    assert "arxiv: Timed out after 30s" in msg
    assert "exceed the check's 30 s timeout (test_servers.py's), as it would upstream" in msg
    assert "one tool call at a time" not in msg and "queued" not in msg
    # no timeout, no hint
    results["arxiv"] = (False, '{"results":[]}')
    assert "timeout" not in M.preflight_failure_reason(
        results, local_env=False, enabled=list(M.RELEASE_SERVERS))


def test_mcp_atlas_doc_drops_the_serialization_claim_and_the_machine_anecdote():
    doc = _doc("mcp_atlas.md")
    assert "one tool call at a time" not in doc
    assert "HTTP 429" not in doc and "On the machine this suite was checked on" not in doc
    assert "serves tool calls concurrently" in doc
    assert "wikipedia" in doc and '{"results": []}' in doc
    src = open(M.__file__, encoding="utf-8").read()
    assert "one tool call at a time" not in src


@pytest.fixture
def atlas_env(monkeypatch):
    """A stand-in agent environment (every release server online, one calculator tool)."""
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/enabled-servers":
                servers = [[s, "OK"] for s in M.RELEASE_SERVERS]
                return self._send({"servers": servers, "total": len(servers),
                                   "online": len(servers), "offline": 0})
            return self._send({"message": "ok"})

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.path == "/list-tools":
                return self._send([{"name": "calculator_calculate", "description": "Calculates.",
                                    "inputSchema": {"type": "object"}}])
            return self._send([{"type": "text", "text": "42"}])

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("GBENCH_MCP_ATLAS_SANDBOX_URL", f"http://127.0.0.1:{srv.server_address[1]}")
    # the external environment's checks, gbench's credentialed-servers container and the
    # image checks they need (tests/test_mcp_credential_isolation_and_dns.py covers them): no
    # docker here
    monkeypatch.setattr(M, "_external_environment_setup", lambda sandbox: None)
    monkeypatch.setattr(M, "check_mcp_atlas_prerequisites", lambda: (True, ""))
    monkeypatch.delenv("GBENCH_MCP_ATLAS_TEMPERATURE", raising=False)
    rows = [{"TASK": "t1", "PROMPT": "What is 6 times 7?",
             "ENABLED_TOOLS": '["calculator_calculate"]', "GTFA_CLAIMS": '["The answer is 42."]'}]
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: rows)
    monkeypatch.setattr(M, "gemini_required_skip", lambda *a: None)
    # the search engines' Gemini grounding backend, stood in (no network)
    monkeypatch.setattr(M, "_ground", lambda query, n: (
        [{"title": "T", "url": "https://example.org/", "snippet": "S"}], "gemini:test"))

    async def judge(prompt, config=None):
        return json.dumps({"claim_text": "c", "coverage_outcome": "fulfilled",
                           "justification": "j", "confidence_level": 1.0}), "gemini-test"

    monkeypatch.setattr(M, "judge_generate_cascade", judge)
    yield
    srv.shutdown()
    srv.server_close()


def test_mcp_atlas_sends_no_max_tokens_and_counts_window_cut_tasks(atlas_env, monkeypatch):
    seen = []

    async def cut(**k):
        seen.append(k.get("max_output_tokens"))
        return base.Reply(text="The answer is 42 and furthermore", tool_calls=None,
                          finish_reason="length", completion_tokens=256)

    monkeypatch.setattr(base, "_send_single_request", cut)
    res = M.run_mcp_atlas("m", "http://127.0.0.1:1/v1", concurrency=1, max_output_tokens=256)
    # the harness sends no max_tokens: the run's --max-output-tokens is not applied (the
    # window is unknown here, so the field is left out)
    assert seen and set(seen) == {None}
    assert M.MCP_ATLAS_SAMPLING.send_max_tokens is False
    assert res["official_sampling"]["send_max_tokens"] is False
    assert res["max_output_tokens_note"].startswith("not sent")
    assert res["truncated_responses"] == 1
    assert "output budget" not in res["leaderboard_comparable_reason"]
    doc = _doc("mcp_atlas.md")
    assert "the harness sends no `max_tokens`" in doc and "`truncated_responses`" in doc
    assert "`max_output_tokens_note`" in doc


# --------------------------------------------------------------------------- #
# gaia
# --------------------------------------------------------------------------- #
def test_gaia_video_attachment_is_sized_as_a_video_part_not_as_its_base64(monkeypatch):
    parts = GAIA._file_parts("clip.mov", b"\x00" * 3_000_000)
    assert parts[1]["type"] == "video_url"
    messages = [{"role": "system", "content": GAIA._GAIA_SYSTEM},
                {"role": "user", "content": [{"type": "text", "text": "What happens?"}] + parts}]
    monkeypatch.setattr(base, "get_run_knob",
                        lambda name: 131072 if name == "max_model_len" else None)
    low, high, _mode = base._prompt_token_bounds(messages, [GAIA.WEB_SEARCH_TOOL])
    assert high < 10_000, "the 4 MB of base64 must not be counted as text"
    assert low >= base.VIDEO_SOFT_TOKENS
    budget, refused = base.clamp_to_context(messages, 16384, tools=[GAIA.WEB_SEARCH_TOOL])
    assert refused is None and budget == 16384


def _gaia_test_run(tmp_path, monkeypatch, traces):
    monkeypatch.setenv("GBENCH_GAIA_SPLIT", "test")
    monkeypatch.setenv("GBENCH_RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(GAIA, "check_gaia_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(GAIA, "search_available", lambda: True)
    monkeypatch.setattr(GAIA, "gemini_key_live_valid", lambda key: (True, ""))
    monkeypatch.setattr(GAIA, "_load_gaia_samples", lambda limit=None, split="validation": [])
    monkeypatch.setattr(GAIA, "run_eval_suite", lambda **kw: {"sample_traces": traces})
    with pytest.raises(RuntimeError, match="INFRA REQUIRED") as e:
        GAIA.run_gaia("m", "http://x", 1)
    [res] = list((tmp_path / "gaia_submission").glob("*_result.json"))
    return str(e.value), json.loads(res.read_text())["leaderboard_submission"]


def test_gaia_submission_note_tells_over_context_tasks_apart_from_failed_requests(
        tmp_path, monkeypatch):
    traces = [
        {"sample_idx": 0, "category": "Level_1", "extra_payload": {"task_id": "a1"},
         "response_text": "FINAL ANSWER: 3", "status": "CORRECT"},
        {"sample_idx": 1, "category": "Level_2", "extra_payload": {"task_id": "mov"},
         "response_text": None, "status": "OVER_CONTEXT"},
        {"sample_idx": 2, "category": "Level_3", "extra_payload": {"task_id": "c3"},
         "response_text": None, "status": "TIMEOUT"},
    ]
    msg, sub = _gaia_test_run(tmp_path, monkeypatch, traces)
    assert sub["failed_requests"] == 1 and sub["over_context_tasks"] == ["mov"]
    assert "1 task(s) do not fit the served model's context window" in msg
    assert "larger --max-model-len" in msg
    assert "1 request(s) failed and have an empty model_answer; re-run" in msg
    doc = _doc("gaia.md")
    assert "over_context_tasks" in doc and "GBENCH_VIDEO_SOFT_TOKENS" in doc


def test_gaia_submission_note_without_failures_names_neither(tmp_path, monkeypatch):
    traces = [{"sample_idx": 0, "category": "Level_1", "extra_payload": {"task_id": "a1"},
               "response_text": "FINAL ANSWER: 3", "status": "CORRECT"}]
    msg, sub = _gaia_test_run(tmp_path, monkeypatch, traces)
    assert sub["failed_requests"] == 0 and sub["over_context_tasks"] == []
    assert "context window" not in msg and "request(s) failed" not in msg


# --------------------------------------------------------------------------- #
# toolbench: the SoWR query set
# --------------------------------------------------------------------------- #
def _ans(final):
    return {"final_answer": final, "answer_details": [{"role": "tool", "message": "x", "next": []}]}


def _stub_judge(monkeypatch):
    async def fake_tooleval(func, args, sem, *, return_reason, accept=None):
        return ({"answer_status": "Solved"}
                if "answer_status" in T._FUNCTION_PARAMETERS[func]["properties"] else {"index": 0})
    monkeypatch.setattr(T, "_tooleval", fake_tooleval)
    monkeypatch.setattr(T, "_final_step_has_finish", lambda a: True)


def test_toolbench_sowr_under_eval_limit_votes_only_on_the_queries_that_ran(monkeypatch):
    """The review's reproduction: 2 of 10 reference queries ran, the candidate wins both."""
    _stub_judge(monkeypatch)
    reference = {str(i): {"query": f"q{i}", "answer": _ans("")} for i in range(10)}
    candidate = {str(i): {"query": f"q{i}", "answer": _ans("solved it")} for i in range(2)}
    r = asyncio.run(T._score_all(candidate, reference, 3, 4, test_ids=["0", "1"]))
    assert r["sopr"] == 1.0 and r["sowr"] == 1.0
    # a query that ran but produced no candidate answer is the reference's vote
    r = asyncio.run(T._score_all(candidate, reference, 3, 4, test_ids=["0", "1", "2"]))
    assert r["sowr"] == pytest.approx(2 / 3)
    # SoPR scores the candidate's queries among the test ids, as eval_pass_rate.py does
    extra = dict(candidate, x={"query": "qx", "answer": _ans("")})
    r = asyncio.run(T._score_all(extra, reference, 1, 4, test_ids=["0", "1"]))
    assert r["n_candidate"] == 2 and r["sopr"] == 1.0


def test_toolbench_test_ids_come_from_the_image_and_follow_the_entrypoint_slice(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        data = {"G1_tool": {"test_ids": ["5", "3", "9"], "instruction_order": ["3", "9", "5"]},
                "G3_instruction": {"test_ids": ["1", "2"], "instruction_order": ["1", "2"]}}
        return subprocess.CompletedProcess(cmd, 0, "noise\nTB_TEST_IDS " + json.dumps(data), "")

    monkeypatch.setattr(T.subprocess, "run", fake_run)
    full = T._group_test_ids("img", ["G1_tool", "G3_instruction"], None)
    assert full == {"G1_tool": ["5", "3", "9"], "G3_instruction": ["1", "2"]}
    cmd = seen["cmd"]
    assert cmd[:3] == ["docker", "run", "--rm"] and "none" in cmd and "img" in cmd
    assert cmd[-2:] == ["G1_tool", "G3_instruction"]
    assert "test_query_ids" in T._TEST_IDS_PROBE and "test_instruction" in T._TEST_IDS_PROBE
    # --eval-limit 1 per group: the first query of test_instruction, kept in test-id order
    assert T._group_test_ids("img", ["G1_tool", "G3_instruction"], 2) == {
        "G1_tool": ["3", "9"], "G3_instruction": ["1", "2"]}
    monkeypatch.setattr(T.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "boom"))
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*test_query_ids"):
        T._group_test_ids("img", ["G1_tool"], None)


def test_toolbench_probe_script_runs_against_a_query_tree(tmp_path):
    root = tmp_path / "stb" / "solvable_queries"
    (root / "test_query_ids").mkdir(parents=True)
    (root / "test_instruction").mkdir()
    (root / "test_query_ids" / "G1_tool.json").write_text(json.dumps({"7": 1, "3": 1}))
    (root / "test_instruction" / "G1_tool.json").write_text(
        json.dumps([{"query_id": 3}, {"query_id": 7}]))
    script = T._TEST_IDS_PROBE.replace("/stb/solvable_queries", str(root))
    out = subprocess.run(["python3", "-c", script, "G1_tool"], capture_output=True, text=True,
                         check=True).stdout
    data = json.loads(out.split("TB_TEST_IDS ", 1)[1])
    assert data == {"G1_tool": {"test_ids": ["7", "3"], "instruction_order": ["3", "7"]}}


# --------------------------------------------------------------------------- #
# toolbench: ToolEval's function reaches the judge
# --------------------------------------------------------------------------- #
def test_toolbench_judge_request_carries_the_config_yaml_function():
    cfg = T._judge_generation_config("select_better_answer", True)
    assert cfg.temperature == 0.2
    [tool] = cfg.tools
    [decl] = tool.function_declarations
    assert decl.name == "select_better_answer"
    assert decl.description.endswith("You should ignore the impact of the order of candidate "
                                     "answers.")
    assert decl.parameters_json_schema["required"] == ["index", "reason"]
    fcc = cfg.tool_config.function_calling_config
    assert str(fcc.mode).endswith("ANY") and fcc.allowed_function_names == ["select_better_answer"]
    assert cfg.response_json_schema is None and cfg.response_mime_type is None
    for func in ("check_answer_status", "parse_answer_status"):
        decl = T._judge_generation_config(func, False).tools[0].function_declarations[0]
        assert decl.description == ("Parse the json answer with layerd nodes and return the "
                                    "answer_status about the answer")
        assert decl.parameters_json_schema == T._FUNCTION_PARAMETERS[func]
    # the prompt names the function the judge is given
    assert T._SELECT_BETTER_ANSWER.rstrip().endswith("select a random answer.")
    assert "`select_better_answer`" in T._SELECT_BETTER_ANSWER


class _Client:
    """A google-genai client stand-in: `replies` per model, in call order."""

    def __init__(self, replies):
        self.calls = []
        self.replies = replies
        outer = self

        class _Models:
            async def generate_content(self, model, contents, config):
                outer.calls.append((model, contents, config))
                reply = outer.replies[model].pop(0)
                if isinstance(reply, Exception):
                    raise reply
                return reply

        self.aio = types.SimpleNamespace(models=_Models())


def _call(name, args):
    return types.SimpleNamespace(function_calls=[types.SimpleNamespace(name=name, args=args)],
                                 text=None)


def _patch_cascade(monkeypatch, client, models=("m1", "m2"), rounds=1):
    monkeypatch.setattr(base, "gemini_api_keys", lambda: ["k"])
    monkeypatch.setattr(base, "_judge_clients", lambda: [client])
    monkeypatch.setattr(base, "_next_judge_client", lambda: client)
    monkeypatch.setattr(base, "judge_cascade", lambda: list(models))
    monkeypatch.setattr(base, "_judge_cascade_rounds", lambda: rounds)
    monkeypatch.setattr(base, "_JUDGE_BACKOFF", 0.0)


def test_toolbench_cascade_reads_the_forced_call_and_moves_on_without_one(monkeypatch):
    cfg = T._judge_generation_config("check_answer_status", True)
    client = _Client({"m1": [types.SimpleNamespace(function_calls=None, text="Solved")],
                      "m2": [_call("check_answer_status",
                                   {"answer_status": "Solved", "reason": "r"})]})
    _patch_cascade(monkeypatch, client)
    text, model = asyncio.run(T._tooleval_cascade("the prompt", config=cfg))
    assert model == "m2" and json.loads(text) == {"answer_status": "Solved", "reason": "r"}
    assert [c[0] for c in client.calls] == ["m1", "m2"]
    assert client.calls[0][1] == "the prompt" and client.calls[0][2] is cfg
    # every model failing every round is an outage
    client = _Client({"m1": [RuntimeError("429")] * 2, "m2": [RuntimeError("429")] * 2})
    _patch_cascade(monkeypatch, client, rounds=2)
    assert asyncio.run(T._tooleval_cascade("p", config=cfg)) == (None, "JUDGE_OUTAGE")
    assert len(client.calls) == 4
    with pytest.raises(ValueError, match="forced function"):
        asyncio.run(T._tooleval_cascade("p", config=types.SimpleNamespace()))


def test_toolbench_select_better_parses_the_function_call_index(monkeypatch):
    client = _Client({"m1": [_call("select_better_answer", {"index": 1.0, "reason": "r"})]})
    _patch_cascade(monkeypatch, client, models=("m1",))
    got = asyncio.run(T._select_better("q", _ans("a"), _ans("b"), asyncio.Semaphore(1)))
    assert got == 1
    [(_, prompt, cfg)] = client.calls
    assert prompt.startswith("\nQuery:\nq\n\nAnswer_0:\n")
    assert cfg.tool_config.function_calling_config.allowed_function_names == [
        "select_better_answer"]


# --------------------------------------------------------------------------- #
# toolbench: the published reference answers
# --------------------------------------------------------------------------- #
def _baselines_zip(path, groups, model="chatgpt_cot"):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("data_baselines/", "")
        zf.writestr("__MACOSX/data_baselines/._chatgpt_cot", "junk")
        for g in groups:
            for qid in ("11", "12"):
                zf.writestr(f"data_baselines/{model}/{g}/{qid}_CoT@1.json", json.dumps({"q": qid}))
        zf.writestr("data_baselines/gpt-4-0613_cot/G1_tool/1_CoT@1.json", "{}")
    return str(path)


def test_toolbench_reference_is_the_published_chatgpt_cot_converted_and_cached(tmp_path,
                                                                               monkeypatch):
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path / "cache"))
    zip_path = _baselines_zip(tmp_path / "data_baselines.zip",
                              ["G1_tool", "G3_instruction", "G2_category"])
    downloads, runs = [], []
    monkeypatch.setattr(T, "_download_baselines", lambda: downloads.append(1) or zip_path)
    # the image's test ids, which every converted file must cover
    monkeypatch.setattr(T, "_group_test_ids", lambda image, groups, limit: {
        g: ["11", "12"] for g in groups})

    def fake_run(cmd, **kw):
        runs.append(cmd)
        mount = cmd[cmd.index("-v") + 1].split(":")[0]
        g = cmd[cmd.index("--answer_dir") + 1].rsplit("/", 1)[1]
        files = sorted(os.listdir(os.path.join(mount, "in", g)))
        with open(os.path.join(mount, f"{g}.json"), "w") as fh:
            json.dump({f.split("_")[0]: {"query": f, "answer": {}} for f in files}, fh)
        return subprocess.CompletedProcess(cmd, 0, "Converted 2 answers", "")

    monkeypatch.setattr(T.subprocess, "run", fake_run)
    assert T._REFERENCE_DEFAULT == "chatgpt_cot"
    root = T._baseline_reference_root("chatgpt_cot", ["G1_tool", "G3_instruction"], "img")
    ref = T._load_groups(os.path.join(root, "chatgpt_cot"), ["G1_tool", "G3_instruction"])
    assert {g: sorted(v) for g, v in ref.items()} == {"G1_tool": ["11", "12"],
                                                      "G3_instruction": ["11", "12"]}
    assert root.startswith(str(tmp_path / "cache")) and T.BASELINES_REVISION[:12] in root
    cmd = runs[0]
    assert "img" in cmd and "--network" in cmd and "convert_to_answer_format.py" in cmd
    assert cmd[cmd.index("--method") + 1] == "CoT@1" and cmd[cmd.index("-w") + 1] == \
        "/stb/toolbench/tooleval"
    # cached: no second download or conversion
    T._baseline_reference_root("chatgpt_cot", ["G1_tool"], "img")
    assert len(downloads) == 1 and len(runs) == 2
    # a model that is not a published baseline
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*not one of the published baselines"):
        T._baseline_reference_root("virtual_chatgpt_cot", ["G1_tool"], "img")
    # a failed conversion is a hard error, never a missing SoWR
    monkeypatch.setattr(T.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "boom"))
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*converting the published"):
        T._baseline_reference_root("chatgpt_cot", ["G2_category"], "img")


def test_toolbench_baselines_download_is_pinned_and_checksummed(tmp_path, monkeypatch):
    blob = tmp_path / "data_baselines.zip"
    blob.write_bytes(b"not the release")
    seen = {}

    def fake_download(repo, filename, repo_type, revision):
        seen.update(repo=repo, filename=filename, repo_type=repo_type, revision=revision)
        return str(blob)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*not the pinned"):
        T._download_baselines()
    assert seen == {"repo": "stabletoolbench/baselines", "filename": "data_baselines.zip",
                    "repo_type": "dataset", "revision": T.BASELINES_REVISION}
    monkeypatch.setattr(T, "BASELINES_SHA256", hashlib.sha256(b"not the release").hexdigest())
    assert T._download_baselines() == str(blob)


def test_toolbench_run_records_the_reference_source_and_sowr_query_counts(tmp_path, monkeypatch):
    async def status(query, answer, sem):
        return "Solved"

    async def pref(query, cand, ref, sem):
        return "cand"

    monkeypatch.setattr(T, "_answer_status", status)
    monkeypatch.setattr(T, "_preference", pref)
    monkeypatch.setattr(T, "check_toolbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(T, "_served_model_id", lambda endpoint, fallback: fallback)
    # the bridge network's gateway (after the metadata guard) and the image's records of its
    # tool calls: stubbed, no docker (the relays listen on loopback)
    monkeypatch.setattr(T, "_bridge_gateway", lambda image: "127.0.0.1")
    monkeypatch.setattr(T, "read_server_calls", lambda path: [])
    monkeypatch.setattr(T, "read_observations", lambda path: {})
    # gbench's Gemini relay (started by the run) needs a key; it never reaches the container
    # Split so that no source line matches the leak gate (scripts/check_no_leaks.sh).
    monkeypatch.setenv("GEMINI_API_KEY", "AIza" + "TESTfixtureKEY000000000000000000000")
    for k in ("GBENCH_TOOLBENCH_GROUPS", "GBENCH_TOOLBENCH_REFERENCE_DIR",
              "GBENCH_TOOLBENCH_REFERENCE", "GBENCH_TOOLBENCH_EVAL_TIMES"):
        monkeypatch.delenv(k, raising=False)
    ref_root = tmp_path / "ref"
    (ref_root / "chatgpt_cot").mkdir(parents=True)
    for g in T._GROUPS:
        (ref_root / "chatgpt_cot" / f"{g}.json").write_text(json.dumps(
            {str(i): {"query": "q", "answer": _ans("r")} for i in range(5)}))
    seen = {}
    monkeypatch.setattr(T, "_baseline_reference_root",
                        lambda ref, groups, image: seen.setdefault("ref", ref) and str(ref_root))
    monkeypatch.setattr(T, "_group_test_ids", lambda image, groups, limit: (seen.setdefault(
        "limit", limit), {g: ["0", "1"] for g in groups})[1])

    def fake_run(cmd, *a, **k):
        if cmd[:2] == ["docker", "run"]:
            out = cmd[cmd.index("-v") + 1].split(":")[0]
            os.makedirs(os.path.join(out, "candidate"), exist_ok=True)
            for g in T._GROUPS:
                with open(os.path.join(out, "candidate", f"{g}.json"), "w") as fh:
                    json.dump({"0": {"query": "q", "answer": _ans("a")},
                               "1": {"query": "q", "answer": _ans("b")}}, fh)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(T.subprocess, "run", fake_run)
    r = T.run_toolbench("m", "http://x:8000/v1", 2, limit=12)
    # every test id is read (limit None); --eval-limit 12 then draws 12 of the 12 (group, id)
    # pairs on the host (select_query_ids), never a per-group head
    assert seen == {"ref": "chatgpt_cot", "limit": None}
    assert r["sowr"] == 100.0, "under --eval-limit the unrun reference queries are not losses"
    assert r["sowr_queries_by_group"] == {g: 2 for g in T._GROUPS}
    assert r["reference_source"].startswith("stabletoolbench/baselines@" + T.BASELINES_REVISION)
    assert r["reference_model"] == "chatgpt_cot"


def test_toolbench_doc_states_the_reference_the_function_and_process_answer():
    doc = _doc("toolbench.md")
    for needle in ("`answer_details` cut to its first 3,000 entries", "function calling",
                   "You should ignore the impact of the order of candidate answers.",
                   T.BASELINES_REVISION, T.BASELINES_SHA256, "`chatgpt_cot`",
                   "drawn stratified by group", "sowr_queries_by_group",
                   "reference_source", "--max-output-tokens"):
        assert needle in doc, needle
    assert "structured output" not in doc and "image's shipped data" not in doc
    assert "virtual_chatgpt_cot" not in doc


# --------------------------------------------------------------------------- #
# mcp_bench
# --------------------------------------------------------------------------- #
_CONNECTED = {f"S{i}": {"status": "success", "tools": 3, "error": ""} for i in range(28)}


def test_mcp_bench_probe_forwards_the_assume_network_knob(monkeypatch):
    monkeypatch.setenv("GBENCH_MCP_BENCH_ASSUME_NETWORK", "1")
    for k in MB._REQUIRED_KEYS:
        monkeypatch.delenv(k, raising=False)
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"], seen["timeout"] = cmd, kw.get("timeout")
        net = any(a == "MCP_ASSUME_NETWORK=1" for a in cmd)
        line = {"servers": ["Wikipedia"], "unavailable": [] if net else ["Wikipedia"],
                "unbuilt": [], "network_up": net, "connections": _CONNECTED}
        return subprocess.CompletedProcess(cmd, 0, "MCP_BENCH_PROBE " + json.dumps(line), "")

    monkeypatch.setattr(MB.subprocess, "run", fake_run)
    probe = MB.provisioning_probe("img")
    assert probe["network_up"] is True and probe["unavailable"] == []
    assert seen["timeout"] == MB.PROBE_TIMEOUT_S
    monkeypatch.delenv("GBENCH_MCP_BENCH_ASSUME_NETWORK")
    assert MB.provisioning_probe("img")["network_up"] is False
    assert not any("MCP_ASSUME_NETWORK" in a for a in seen["cmd"])


def test_mcp_bench_probe_runs_upstreams_connection_check_with_the_keys_file():
    probe = MB._PROBE
    compile(probe, "probe", "exec")
    assert "from utils.collect_mcp_info import MCPServerInfoCollector" in probe
    assert "MCPServerInfoCollector('individual').collect_all_info()" in probe
    # the keys file stays empty: no server key is in the model container, the credentialed
    # servers are reached through the bridge the launcher installs
    assert "open(os.path.join('mcp_servers', 'api_key'), 'w').close()" in probe
    assert "R.install_credentialed_bridges()" in probe
    for key in MB._SERVER_KEYS:
        assert key not in probe
    assert "os.chdir(R.MCPBENCH_DIR)" in probe


def test_mcp_bench_probe_without_a_connection_check_is_a_hard_error(monkeypatch):
    line = {"servers": [], "unavailable": [], "unbuilt": [], "network_up": True}
    monkeypatch.setattr(MB.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 0, "MCP_BENCH_PROBE " + json.dumps(line), ""))
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*connect to"):
        MB.provisioning_probe("img")


def test_mcp_bench_connection_check_gaps_name_each_server():
    keys = {k: "set" for k in MB._REQUIRED_KEYS}
    ok = {"servers": sorted(_CONNECTED), "unavailable": [], "unbuilt": [], "network_up": True,
          "connections": dict(_CONNECTED)}
    assert MB.provisioning_gaps(ok, env_keys=keys) == []
    bad = dict(_CONNECTED)
    bad["S1"] = {"status": "success_no_tools", "tools": 0, "error": ""}
    bad["S2"] = {"status": "failed", "tools": 0, "error": "Connection timeout after 30 seconds"}
    del bad["S3"]
    probe = dict(ok, connections=bad)
    [gap] = MB.provisioning_gaps(probe, env_keys=keys)
    assert gap.startswith("servers that failed MCP-Bench's own connection check "
                          "(utils/collect_mcp_info.py")
    assert "28/28 connected" in gap
    assert "S1 (connected, but listed no tools)" in gap
    assert "S2 (failed: Connection timeout after 30 seconds)" in gap
    assert "S3 (not checked)" in gap and "27 servers checked, where MCP-Bench has 28" in gap


def test_mcp_bench_run_refuses_a_server_that_lists_no_tools_and_records_the_check(monkeypatch):
    monkeypatch.setattr(MB, "check_mcp_bench_prerequisites", lambda: (True, ""))
    for k in MB._REQUIRED_KEYS:
        monkeypatch.setenv(k, "v")
    monkeypatch.delenv("GBENCH_MCP_BENCH_NO_SUBSET", raising=False)
    conns = dict(_CONNECTED)
    conns["Game Trends"] = {"status": "success_no_tools", "tools": 0, "error": ""}
    del conns["S0"]
    probe = {"servers": sorted(conns), "unavailable": [], "unbuilt": [], "network_up": True,
             "connections": conns}
    monkeypatch.setattr(MB, "provisioning_probe", lambda image, relay_env=None: probe)
    # the credentialed-servers container (tests/test_mcp_wildclawbench_credentials_and_cascade.py
    # covers it): a stand-in
    stopped = []
    monkeypatch.setattr(MB, "start_credentialed_servers",
                        lambda image, base, network: types.SimpleNamespace(
                            client_env=lambda: {"MCP_CREDENTIALED_RELAY": "172.17.0.1:10",
                                                "MCP_CREDENTIALED_TOKEN": "t0k"},
                            stop=lambda: stopped.append(1), record=lambda: {}))
    ran = []
    monkeypatch.setattr(MB.subprocess, "run", lambda cmd, **kw: ran.append(cmd))
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*Game Trends \\(connected, but listed "
                                           "no tools\\)"):
        MB.run_mcp_bench("m", "http://x/v1")
    # no benchmark container ran; the credentialed-servers container was stopped again
    assert not any(c[:2] == ["docker", "run"] for c in ran) and stopped == [1]

    ok = dict(probe, servers=sorted(_CONNECTED), connections=dict(_CONNECTED))

    def fake_run(cmd, **kw):
        if cmd[:2] == ["docker", "run"]:
            out = cmd[cmd.index("-v") + 1].split(":")[0]
            with open(os.path.join(out, "mcp_bench_results.json"), "w") as fh:
                json.dump({"input_schema_compliance": 1.0, "valid_tool_name_rate": 1.0,
                           "tool_call_success_rate": 1.0, "task_completion_score": 5.0,
                           "tool_selection_score": 5.0,
                           "planning_effectiveness_and_efficiency_score": 5.0}, fh)
            with open(os.path.join(out, "subset_manifest.json"), "w") as fh:
                json.dump({"filtered": True, "unavailable_servers": [], "tasks_kept": 104}, fh)
            with open(os.path.join(out, MB.HARNESS_LOG), "w") as fh:
                json.dump({"version": MB.HARNESS_LOG_VERSION, "tasks": [
                    {"task_id": "t1", "file": "f.json", "status": "completed", "kinds": []}]}, fh)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(MB, "provisioning_probe", lambda image, relay_env=None: ok)
    monkeypatch.setattr(MB.subprocess, "run", fake_run)
    # the judge's Gemini relay (tests/test_mcp_credential_isolation_and_dns.py covers it): a
    # stand-in, docker is stubbed
    monkeypatch.setattr(MB, "_start_judge_relay", lambda network: types.SimpleNamespace(
        client_env=lambda: {"GEMINI_API_KEY": "gbench-gemini-relay-" + "0" * 48},
        stop=lambda: None, record=lambda: {}))
    res = MB.run_mcp_bench("m", "http://x/v1")
    assert res["server_connection_check"] == {"servers_connected_with_tools": 28,
                                              "servers_checked": 28, "tools_listed": 84}
    assert res["is_full"] is True


def test_mcp_bench_doc_states_the_connection_check():
    doc = _doc("mcp_bench.md")
    for needle in ("utils/collect_mcp_info.py", "28/28 servers connected",
                   "All successfully connected servers returned tools!",
                   "`server_connection_check`", "Neither part of the image check sees a key",
                   "in the pre-run check and in the run"):
        assert needle in doc, needle


@pytest.fixture(autouse=True)
def _metadata_guard_answers_blocked(monkeypatch):
    """Docker is stubbed in these tests: the cloud metadata guard the suites call before starting a
    container (container_guard.require_metadata_blocked) answers "unreachable" without a probe.
    tests/test_guard_tool_suites.py checks the suites' calls to it."""
    from gbench.runners.eval_suites import container_guard
    monkeypatch.setattr(container_guard, "require_metadata_blocked",
                        lambda *a, **k: container_guard.UNREACHABLE)
