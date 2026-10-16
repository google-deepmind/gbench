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
"""SWE family: the endpoint key stays out of trajectories, logs, command lines and task
sandboxes, and SWE-bench-Live runs and reports its three leaderboard settings.

* mini-swe-agent 1.x (swe_bench_live, multi_swe_bench): the key is sent with each request, never
  written into model_kwargs, which mini-swe-agent's save_traj serializes into every trajectory
  file. Credential values found in the files mini-swe-agent writes are replaced as a backstop,
  and the per-run minisweagent.log handler redacts too.
* The fork harness container gets HF_TOKEN by name only.
* swe_bench_pro: the agent gets OPENAI_API_KEY=EMPTY as a literal argument (no shell expansion);
  an endpoint with a key is reached through a forwarder that adds it.
* SWE-bench-Live: Lite, Full (1,319 rows at the leaderboard's commit) and Verified, each with its
  own rollouts, harness pass and headline; Verified divides by its 500 rows.
* `split=` no longer collides with the executor's positional split.

Hermetic: no container, no docker daemon, no network beyond loopback test servers; every cache
and output is under tmp_path.
"""

import asyncio
import http.server
import json
import os
import sys
import threading
import types

import pytest

from gbench.runners.eval_suites import copilot_bench_swe as CB
from gbench.runners.eval_suites import swe_agentic as A
from gbench.runners.eval_suites import swe_bench_live as LV
from gbench.runners.eval_suites import swe_bench_multilingual as ML
from gbench.runners.eval_suites import swe_bench_pro as P
from gbench.runners.eval_suites import swebench_common as C


@pytest.fixture(autouse=True)
def _no_metadata_probe(monkeypatch):
    """The cloud metadata server gate (swebench_common.require_task_network_blocked) starts probe
    containers; tests/test_guard_swe_agents.py covers it, here it is stubbed."""
    monkeypatch.setattr(C, "require_task_network_blocked", lambda *a, **k: None)


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECRET = "sk-test-endpoint-key-0123456789"


@pytest.fixture(autouse=True)
def _scratch(monkeypatch, tmp_path):
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.delenv("GBENCH_RESULTS_DIR", raising=False)
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path / "cache"))
    for var in ("OPENAI_API_KEY", "HF_TOKEN", "GBENCH_SWEBENCH_V5_IMAGE",
                "GBENCH_SWE_OMP_THREADS", "GBENCH_SHARD", "GBENCH_SWE_BENCH_LIVE_MODE",
                "GBENCH_SWE_BENCH_LIVE_IMAGE", "GBENCH_COPILOT_BENCH_SWE_MODE",
                "GBENCH_SWE_BENCH_MULTILINGUAL_MODE", "GBENCH_SWE_BENCH_LIVE_TEMPERATURE",
                "GBENCH_SWE_BENCH_PRO_TASK_HOST", "GBENCH_SWE_BENCH_PRO_TASK_ENDPOINT",
                "GBENCH_SWE_BENCH_PRO_TRAJ_DIR", "GBENCH_REPETITION_PENALTY",
                "GBENCH_FREQUENCY_PENALTY", "GBENCH_PRESENCE_PENALTY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GBENCH_SWE_OMP_THREADS", "0")


def _v1_base():
    return ({"agent": {"step_limit": 250, "cost_limit": 3.0, "instance_template": "/testbed"},
             "environment": {"timeout": 60, "cwd": "/testbed"},
             "model": {"model_name": "anthropic/x",
                       "model_kwargs": {"drop_params": True, "temperature": 0.0}}},
            A.LEADERBOARD_CONFIG_SHA256, "1.17.5")


class _Recorder(http.server.BaseHTTPRequestHandler):
    """A loopback OpenAI-compatible stand-in that records each request's headers and body."""
    seen = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        type(self).seen.append({"path": self.path, "auth": self.headers.get("Authorization"),
                                "body": json.loads(body or b"{}")})
        reply = json.dumps({
            "id": "c1", "object": "chat.completion", "created": 1, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "hello"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *a):
        return


@pytest.fixture
def upstream():
    _Recorder.seen = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Recorder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, _Recorder.seen
    server.shutdown()
    server.server_close()


# ============================================================================================
# mini-swe-agent 1.x: the key is not in the config, the trajectory or the agent log
# ============================================================================================
def test_v1_config_carries_no_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    base = _v1_base()[0]
    base["model"]["model_kwargs"]["api_key"] = "from-the-bundled-file"
    cfg = A.build_agent_config(base, model_name="m", base_url="http://h:1", temperature=0.0,
                               thinking=False, max_output_tokens=None)
    kw = cfg["model"]["model_kwargs"]
    assert "api_key" not in kw and SECRET not in json.dumps(cfg)
    assert kw["api_base"] == "http://h:1/v1"


def test_v1_trajectory_file_written_by_mini_swe_agent_has_no_key(monkeypatch, tmp_path):
    pytest.importorskip("minisweagent")
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    base, _, _ = A.load_agent_config()
    cfg = A.build_agent_config(base, model_name="m", base_url="http://127.0.0.1:9/v1",
                               temperature=0.0, thinking=False, max_output_tokens=None)
    model = A._make_model(cfg["model"])
    agent = types.SimpleNamespace(model=model, messages=[{"role": "user", "content": "x"}],
                                  config={}, env=types.SimpleNamespace(config={}))
    path = str(tmp_path / "i" / "i.traj.json")
    assert A._save_traj(agent, path, "Submitted", "diff\n", None, "i") == path
    text = open(path, encoding="utf-8").read()
    assert SECRET not in text
    saved = json.loads(text)["info"]["config"]["model"]["model_kwargs"]
    assert "api_key" not in saved and saved["api_base"] == "http://127.0.0.1:9/v1"


def test_v1_model_sends_the_key_with_each_request(monkeypatch, upstream):
    pytest.importorskip("minisweagent")
    pytest.importorskip("litellm")
    server, seen = upstream
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "1")
    base, _, _ = A.load_agent_config()
    cfg = A.build_agent_config(base, model_name="m",
                               base_url=f"http://127.0.0.1:{server.server_address[1]}",
                               temperature=0.0, thinking=False, max_output_tokens=None)
    model = A._make_model(cfg["model"])
    reply = model.query([{"role": "user", "content": "hi"}])
    assert reply["content"] == "hello"
    assert seen and seen[-1]["auth"] == f"Bearer {SECRET}"
    assert "api_key" not in model.config.model_kwargs
    assert SECRET not in json.dumps(reply["extra"], default=str)
    # an endpoint without a key gets the placeholder the OpenAI client needs
    monkeypatch.delenv("OPENAI_API_KEY")
    model.query([{"role": "user", "content": "hi"}])
    assert seen[-1]["auth"] == "Bearer EMPTY"


def test_files_mini_swe_agent_writes_are_scrubbed_of_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    traj = tmp_path / "t.traj.json"
    traj.write_text(json.dumps({"messages": [{"role": "user", "content": f"env: {SECRET}"}]}))
    clean = tmp_path / "c.json"
    clean.write_text('{"a": 1}')
    before = clean.stat().st_mtime_ns
    A.scrub_secrets_in_files(str(traj), str(clean), str(tmp_path / "missing.json"), None)
    assert SECRET not in traj.read_text() and "[REDACTED]" in traj.read_text()
    assert json.loads(traj.read_text())["messages"][0]["content"] == "env: [REDACTED]"
    assert clean.stat().st_mtime_ns == before          # nothing to replace: not rewritten


def test_v2_rollout_scrubs_the_runner_files_before_reading_them(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    task = A.AgentTask("o__r-1", "issue", "img", "o/r")

    def _fake_run(cmd, **kw):
        if cmd[:2] == ["docker", "run"]:
            inst = tmp_path / "o__r-1"
            (inst / "result.json").write_text(json.dumps(
                {"exit_status": "Submitted", "submission": "d\n", "error": None,
                 "trajectory": [{"role": "tool", "content": f"OPENAI_API_KEY={SECRET}"}]}))
            (inst / "o__r-1.traj.json").write_text(json.dumps({"k": SECRET}))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(A.subprocess, "run", _fake_run)
    tr = A._rollout_v2(0, task, {"model": {}}, str(tmp_path), image="img", version="2.4.6")
    assert SECRET not in json.dumps(tr)
    for name in ("result.json", "o__r-1.traj.json", "spec.json"):
        assert SECRET not in (tmp_path / "o__r-1" / name).read_text()


def test_v1_agent_log_file_is_redacted(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setenv("GBENCH_SWE_BENCH_LIVE_TRAJ_DIR", str(tmp_path / "trajs"))
    monkeypatch.setattr(A, "load_agent_config", _v1_base)

    def _rollouts(tasks, config, concurrency, out_dir, eval_name="swe", rollout=None):
        import logging
        logging.getLogger("minisweagent.agent").warning("request failed for key %s", SECRET)
        assert SECRET not in json.dumps(config)
        return [{"sample_idx": i, "category": t.category, "agent_patch": "d\n",
                 "extra_payload": {"instance_id": t.instance_id}, "agent_exit_status": "Submitted",
                 "agent_infra_error": False, "agent_steps": 1, "truncated_turns": 0,
                 "context_retry_turns": 0, "is_correct": False, "status": "OK"}
                for i, t in enumerate(tasks)]
    monkeypatch.setattr(A, "run_rollouts", _rollouts)

    async def _score(traces):
        return None
    res = A.run_agentic("swe_bench_live", "m", "http://h:1/v1", 1,
                        [A.AgentTask("a__a-1", "i", "img", "o/r")], _score, {})
    log = os.path.join(res["agent_output_dir"], "minisweagent.log")
    text = open(log, encoding="utf-8").read()
    assert "request failed for key [REDACTED]" in text and SECRET not in text
    assert SECRET not in json.dumps(res, default=str)


# ============================================================================================
# fork harness container: HF_TOKEN by name only
# ============================================================================================
def test_harness_container_gets_hf_token_by_name_only(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value_123456")
    captured = {}

    def _fake_run(cmd, **kw):
        captured.setdefault("cmd", list(cmd))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(C.subprocess, "run", _fake_run)
    scorer = C.make_swebench_scorer("swe_bench_live", "m", "SWE-bench-Live/SWE-bench-Live",
                                    "lite", "starryzhang", 1, {},
                                    harness_image="gbench-swe-bench-live:a145aa87")
    asyncio.run(scorer([{"extra_payload": {"instance_id": "a__a-1"}, "agent_patch": "d\n"}]))
    cmd = captured["cmd"]
    assert "hf_secret_value_123456" not in " ".join(cmd)
    assert cmd[cmd.index("HF_TOKEN") - 1] == "-e"


# ============================================================================================
# swe_bench_pro: no key on a command line or in a sandbox
# ============================================================================================
def test_pro_orchestrator_has_no_shell_and_no_key(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    cmd = P._orchestrator_cmd(str(tmp_path), "n1", ["-p", "x"])
    assert SECRET not in " ".join(cmd) and "sh" not in cmd and "-c" not in cmd
    assert cmd[-2:] == ["--ae", "OPENAI_API_KEY=EMPTY"]
    assert "OPENAI_API_KEY" not in cmd[:cmd.index(P.v2_image())]


def test_pro_run_harbor_passes_no_key_in_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    seen = {}

    def _fake_run(cmd, **kw):
        seen.setdefault("cmd", list(cmd))
        seen.setdefault("env", kw.get("env"))
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(P.subprocess, "run", _fake_run)
    P._run_harbor(str(tmp_path), ["-p", "x"], "agent")
    assert SECRET not in " ".join(seen["cmd"])
    # the docker CLI inherits the environment, but the container gets no OPENAI_API_KEY
    assert "OPENAI_API_KEY" not in seen["cmd"][:seen["cmd"].index(P.v2_image())]


def test_endpoint_api_key_treats_the_placeholder_as_none(monkeypatch):
    assert P.endpoint_api_key() is None
    monkeypatch.setenv("OPENAI_API_KEY", "EMPTY")
    assert P.endpoint_api_key() is None
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    assert P.endpoint_api_key() == SECRET


def test_key_forwarder_adds_the_key_and_relays_the_reply(upstream):
    import urllib.request
    server, seen = upstream
    fwd = P.KeyForwarder(f"http://127.0.0.1:{server.server_address[1]}/v1", SECRET,
                         "127.0.0.1").start()
    try:
        assert fwd.url.startswith("http://127.0.0.1:") and fwd.url.endswith("/v1")
        req = urllib.request.Request(fwd.url + "/chat/completions", method="POST",
                                     data=json.dumps({"model": "m", "messages": []}).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer EMPTY"})
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read())
            assert r.status == 200 and r.headers["Content-Type"] == "application/json"
        assert body["choices"][0]["message"]["content"] == "hello"
        assert seen[-1]["path"] == "/v1/chat/completions"
        assert seen[-1]["auth"] == f"Bearer {SECRET}"
        assert seen[-1]["body"] == {"model": "m", "messages": []}
    finally:
        fwd.stop()


def _pro_v2_stubs(monkeypatch, tmp_path):
    monkeypatch.setenv("GBENCH_SWE_BENCH_PRO_TRAJ_DIR", str(tmp_path / "runs"))
    rows = [{"instance_id": "instance_a", "problem_statement": "p", "docker_image": "img",
             "repo_language": "go", "repo": "o/r"}]
    monkeypatch.setattr(P, "load_v2_rows", lambda: rows)
    monkeypatch.setattr(P, "load_hard51_ids", lambda ids: set())
    monkeypatch.setattr(P, "read_locked_agent_config",
                        lambda: ("model:\n  model_kwargs: {}\n", P.AGENT_CONFIG_SHA256))
    jobs = []

    def _harbor(workdir, args, job):
        jobs.append((job, list(args)))
        return 0, ""
    monkeypatch.setattr(P, "_run_harbor", _harbor)
    return jobs


def test_pro_v2_with_a_key_routes_the_sandboxes_through_the_forwarder(monkeypatch, tmp_path,
                                                                      upstream):
    server, seen = upstream
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setenv("GBENCH_SWE_BENCH_PRO_TASK_HOST", "127.0.0.1")
    jobs = _pro_v2_stubs(monkeypatch, tmp_path)
    started = []
    real_start = P.KeyForwarder.start

    def _start(self):
        started.append(self)
        return real_start(self)
    monkeypatch.setattr(P.KeyForwarder, "start", _start)
    base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    res = P.run_swe_bench_pro_v2("m", base, 1, enable_thinking=True)
    agent_args = jobs[0][1]
    flags = list(zip(agent_args, agent_args[1:]))
    fwd_url = res["harness"]["task_endpoint"]
    assert fwd_url.startswith("http://127.0.0.1:") and fwd_url != base
    assert ("--ae", f"OPENAI_API_BASE={fwd_url}") in flags
    assert ("--allow-agent-host", "127.0.0.1") in flags
    assert res["harness"]["key_forwarder"]["relays_to"] == base
    assert res["harness"]["sandbox_api_key"] == "EMPTY"
    assert SECRET not in json.dumps(jobs) and SECRET not in json.dumps(res, default=str)
    assert started and started[0]._server is None          # stopped when the run ended


def test_pro_v2_without_a_key_goes_through_a_relay_that_adds_nothing(monkeypatch, tmp_path):
    # Without a key the sandboxes still reach the endpoint through gbench's relay, which counts
    # the endpoint's failures (they are harness errors); it adds no Authorization header.
    monkeypatch.setenv("GBENCH_SWE_BENCH_PRO_TASK_HOST", "127.0.0.1")
    jobs = _pro_v2_stubs(monkeypatch, tmp_path)
    monkeypatch.setattr(P.KeyForwarder, "start",
                        lambda self: pytest.fail("no key forwarder without a key"))
    res = P.run_swe_bench_pro_v2("m", "http://127.0.0.1:8000/v1", 1, enable_thinking=True)
    flags = list(zip(jobs[0][1], jobs[0][1][1:]))
    relay_url = res["harness"]["task_endpoint"]
    assert relay_url.startswith("http://127.0.0.1:") and relay_url != "http://127.0.0.1:8000/v1"
    assert ("--ae", f"OPENAI_API_BASE={relay_url}") in flags
    assert "key_forwarder" not in res["harness"]
    assert res["harness"]["endpoint_relay"]["relays_to"] == "http://127.0.0.1:8000/v1"
    assert res["relay"]["relays_to"] == "http://127.0.0.1:8000/v1"
    assert "adds" not in res["relay"]


def test_pro_v2_forwarder_that_cannot_listen_is_infra_required(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    _pro_v2_stubs(monkeypatch, tmp_path)

    def _fail(self):
        raise OSError(99, "Cannot assign requested address")
    monkeypatch.setattr(P.KeyForwarder, "start", _fail)
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*GBENCH_SWE_BENCH_PRO_TASK_HOST") as e:
        P.run_swe_bench_pro_v2("m", "http://127.0.0.1:8000/v1", 1, enable_thinking=True)
    assert SECRET not in str(e.value)


# ============================================================================================
# SWE-bench-Live: the three leaderboard settings
# ============================================================================================
def _fake_datasets(monkeypatch, by_split):
    calls = []
    fake = types.ModuleType("datasets")

    def _load(name, *a, split=None, revision=None, **k):
        calls.append({"name": name, "split": split, "revision": revision})
        return [dict(r) for r in by_split[split]]
    fake.load_dataset = _load
    monkeypatch.setitem(sys.modules, "datasets", fake)
    return calls


def _live_rows(split, n, dup=False):
    rows = [{**{c: "x" for c in LV.SNAPSHOT_COLUMNS}, "repo": "o/r",
             "instance_id": f"{split}__r-{i}", "problem_statement": f"issue {i}"}
            for i in range(n)]
    if dup:
        rows[-1] = {**rows[-2], "test_cmds": "other"}
    return rows


def _live_setup(monkeypatch, dup_verified=True):
    by_split = {"lite": _live_rows("lite", 3), "full": _live_rows("full", 4),
                "verified": _live_rows("verified", 5, dup=dup_verified)}
    for s, rows in by_split.items():
        monkeypatch.setitem(LV.SPLIT_ROWS, s, len(rows))
        monkeypatch.setitem(LV.SPLIT_SHA256, s, C.snapshot_sha256(rows))
    calls = _fake_datasets(monkeypatch, by_split)
    monkeypatch.setattr(LV, "check_swe_bench_live_prerequisites", lambda settings: (True, ""))
    monkeypatch.setattr(LV, "check_sweagent_prerequisites", lambda: (True, ""))
    return by_split, calls


def _stub_agent(monkeypatch, resolve):
    """SWE-agent rollouts that submit a patch per task; the scorer resolves the ids in
    `resolve`."""
    def _rollouts(tasks, rows, model_name, relay_url, concurrency, out_dir, relay=None):
        return [{"sample_idx": i, "source_sample_idx": i, "category": t.category,
                 "agent_patch": "d\n", "extra_payload": {"instance_id": t.instance_id, **t.meta},
                 "agent_exit_status": "submitted", "agent_infra_error": False,
                 "agent_steps": 1, "truncated_turns": 0, "context_retry_turns": 0,
                 "is_correct": False, "status": "OK"} for i, t in enumerate(tasks)]
    monkeypatch.setattr(LV, "run_sweagent", _rollouts)
    harness = []

    def _scorer(eval_name, model_name, dataset, split, namespace, workers, metrics, **k):
        harness.append({"dataset": dataset, "split": split, "image": k.get("harness_image")})

        async def _score(traces):
            ids = [t["extra_payload"]["instance_id"] for t in traces]
            rows = C._read_jsonl(dataset)
            metrics["swebench_report"] = {
                "total_instances": sum(1 for r in rows if r["instance_id"] in ids),
                "submitted_instances": len(ids), "completed_instances": len(ids),
                "resolved_instances": len(set(ids) & resolve), "error_ids": []}
            for t in traces:
                t["is_correct"] = t["extra_payload"]["instance_id"] in resolve
        return _score
    monkeypatch.setattr(C, "make_swebench_scorer", _scorer)
    return harness


def test_live_runs_every_leaderboard_setting_with_its_own_headline(monkeypatch, tmp_path):
    by_split, calls = _live_setup(monkeypatch)
    harness = _stub_agent(monkeypatch, {"lite__r-0", "full__r-1", "full__r-2", "verified__r-3"})
    # the three Python tabs (the multi-language and Windows tabs have their own tests)
    res = LV.run_swe_bench_live("m", "http://h:1/v1", 1, eval_categories="lite,full,verified")
    assert res["leaderboard_settings"] == ["lite", "full", "verified"]
    # each setting read its own split at its own commit, and had its own harness pass
    assert [(c["split"], c["revision"]) for c in calls] == [
        ("lite", LV._REVISION), ("full", LV._FULL_REVISION), ("verified", LV._REVISION)]
    assert [h["split"] for h in harness] == ["lite", "full", "verified"]
    assert all(h["image"] == LV._image() for h in harness)
    assert harness[1]["dataset"].endswith(f"SWE-bench-Live__SWE-bench-Live@{LV._FULL_REVISION}"
                                          "/full.jsonl")
    s = res["settings"]
    assert (s["lite"]["correct_answers"], s["lite"]["total_questions"]) == (1, 3)
    assert (s["full"]["correct_answers"], s["full"]["total_questions"]) == (2, 4)
    # verified: 5 rows, 4 ids (one repeated); the headline divides by rows, as the board does
    assert s["verified"]["total_questions"] == 5 and s["verified"]["total_instance_ids"] == 4
    assert s["verified"]["accuracy"] == 20.0
    assert s["verified"]["dataset_repeated_instance_ids"] == {"verified__r-3": 2}
    assert res["headline_metrics"] == {"lite_resolved_rate": 33.33, "full_resolved_rate": 50.0,
                                       "verified_resolved_rate": 20.0}
    assert res["headline_setting"] == "lite" and res["accuracy"] == 33.33
    assert (res["correct_answers"], res["total_questions"]) == (1, 3)
    assert set(res["category_accuracy"]) == {"lite", "full", "verified"}
    assert res["category_accuracy"]["verified"] == {"correct": 1, "total": 5, "accuracy": 20.0}
    assert res["status"] == "success"
    traces = res["sample_traces"]
    assert len(traces) == 3 + 4 + 4
    assert [t["sample_idx"] for t in traces] == list(range(11))
    assert [t["setting"] for t in traces] == ["lite"] * 3 + ["full"] * 4 + ["verified"] * 4
    assert "sample_traces" not in s["lite"]
    # SWE-agent-for-eval is a reference scaffold: the only reasons are the stand-in splits' row
    # counts, which no board row has
    reason = res["leaderboard_comparable_reason"]
    assert res["leaderboard_comparable"] is False
    assert "reference scaffolds" not in reason and "no Lite leaderboard row scores" in reason
    assert res["dataset"].startswith("lite: SWE-bench-Live/SWE-bench-Live@" + LV._REVISION)


def test_live_split_kwarg_runs_one_setting_through_the_real_executor(monkeypatch, tmp_path):
    _live_setup(monkeypatch)
    harness = _stub_agent(monkeypatch, {"verified__r-0"})
    res = LV.run_swe_bench_live("m", "http://h:1/v1", 1, split="verified")
    assert res["leaderboard_settings"] == ["verified"] and [h["split"] for h in harness] == [
        "verified"]
    assert res["headline_setting"] == "verified"
    assert res["total_questions"] == 5 and res["total_instance_ids"] == 4
    assert res["accuracy"] == 20.0


def test_live_oneshot_split_kwarg_reaches_the_real_executor(monkeypatch, tmp_path):
    _live_setup(monkeypatch)
    monkeypatch.setenv("GBENCH_SWE_BENCH_LIVE_MODE", "oneshot")
    seen = {}

    def _suite(**k):
        seen["samples"] = k["samples"]
        traces = [{"extra_payload": s[2], "is_correct": s[2]["instance_id"] == "verified__r-3"}
                  for s in k["samples"]]
        return {"status": "success", "temperature": 0.0, "total_questions": len(traces),
                "correct_answers": 1, "accuracy": 25.0, "sample_traces": traces,
                "category_accuracy": {}}
    monkeypatch.setattr(C, "run_eval_suite", _suite)
    monkeypatch.setattr(C, "make_swebench_scorer", lambda *a, **k: (lambda traces: None))
    res = LV.run_swe_bench_live("m", "http://h:1/v1", 1, split="verified")
    assert len(seen["samples"]) == 4
    v = res["settings"]["verified"]
    assert v["total_questions"] == 5 and v["total_instance_ids"] == 4 and v["accuracy"] == 20.0
    assert res["mode"] == A.ONESHOT and res["leaderboard_comparable"] is False
    assert "one-shot mode" in res["leaderboard_comparable_reason"]


def test_live_eval_categories_pick_settings_in_leaderboard_order(monkeypatch):
    assert LV.resolve_settings() == list(LV.SETTINGS)
    assert LV.resolve_settings(None, "verified,lite") == ["lite", "verified"]
    assert LV.resolve_settings("full", "lite") == ["full"]
    with pytest.raises(ValueError, match="not leaderboard settings"):
        LV.resolve_settings(None, "test")
    with pytest.raises(ValueError, match="not leaderboard settings"):
        LV.run_swe_bench_live("m", "http://h:1/v1", 1, split="test")


def _labels(fork):
    return types.SimpleNamespace(returncode=0, stdout=json.dumps(
        {"org.gbench.swe_bench_live_fork": fork}))


def test_live_prereq_refuses_an_image_of_another_fork_commit(monkeypatch):
    monkeypatch.setattr(LV.shutil, "which", lambda _x: "/usr/bin/docker")
    monkeypatch.setattr(LV, "pinned_swe_snapshot",
                        lambda *a, **k: pytest.fail("no fetch for a wrong image"))
    for stdout in ("null", json.dumps({"org.gbench.swe_bench_live_fork": "0" * 40})):
        monkeypatch.setattr(LV.subprocess, "run", lambda cmd, **kw: types.SimpleNamespace(
            returncode=0, stdout=stdout))
        ok, why = LV.check_swe_bench_live_prerequisites(LV.PYTHON_SETTINGS)
        assert not ok and LV.FORK_COMMIT in why and "docker build" in why


def test_live_image_pins_the_fork_the_suite_expects():
    text = open(os.path.join(REPO, "gbench", "docker", "swe_bench_live.Dockerfile"),
                encoding="utf-8").read()
    assert f"ARG SWE_BENCH_LIVE_REF={LV.FORK_COMMIT}" in text
    assert 'org.gbench.swe_bench_live_fork="${SWE_BENCH_LIVE_REF}"' in text
    assert "ARG DOCKER_CLI_VERSION=5:29.8.0-1~ubuntu.24.04~noble" in text
    assert "-c /opt/swe-bench-live-constraints.txt" in text
    assert "FROM ubuntu:24.04@sha256:" in text
    constraints = open(os.path.join(REPO, "gbench", "docker", "swe_bench_live.constraints.txt"),
                       encoding="utf-8").read().splitlines()
    pins = [c for c in constraints if c and not c.startswith("#")]
    assert "datasets==5.0.1" in pins and "docker==7.2.0" in pins
    assert all("==" in c for c in pins) and not any(c.startswith("swebench") for c in pins)
    v5 = open(os.path.join(REPO, "gbench", "docker", "swebench_v5.Dockerfile"),
              encoding="utf-8").read()
    assert "FROM ubuntu:24.04@sha256:" in v5


def test_live_prereq_checks_every_selected_setting(monkeypatch):
    monkeypatch.setattr(LV.shutil, "which", lambda _x: "/usr/bin/docker")
    monkeypatch.setattr(LV.subprocess, "run", lambda cmd, **kw: _labels(LV.FORK_COMMIT))
    checked = []
    monkeypatch.setattr(LV, "pinned_swe_snapshot",
                        lambda d, split, rev, name, **k: checked.append(
                            (split, rev, k["rows"], k["sha256"])) or "p")
    assert LV.check_swe_bench_live_prerequisites(LV.PYTHON_SETTINGS) == (True, "")
    assert checked == [(s, LV.SETTING_REVISION[s], LV.SPLIT_ROWS[s], LV.SPLIT_SHA256[s])
                       for s in ("lite", "full", "verified")]
    checked.clear()
    assert LV.check_swe_bench_live_prerequisites(["verified"]) == (True, "")
    assert [c[0] for c in checked] == ["verified"]


def test_live_run_checks_the_settings_it_will_run(monkeypatch):
    seen = []
    monkeypatch.setattr(LV, "check_swe_bench_live_prerequisites",
                        lambda settings: seen.append(list(settings)) or (False, "no image"))
    with pytest.raises(RuntimeError, match="INFRA REQUIRED"):
        LV.run_swe_bench_live("m", "http://h:1/v1", 1, eval_categories="full")
    assert seen == [["full"]]


def test_live_pins_the_leaderboard_full_set():
    assert LV.PYTHON_SETTINGS == ("lite", "full", "verified") == LV.SETTINGS[:3]
    assert LV.SETTING_REVISION == {"lite": LV._REVISION, "full": LV._FULL_REVISION,
                                   "verified": LV._REVISION}
    assert LV._REVISION == "b51a86422e10cfd403beb4773e5a2947953e36ec"
    assert LV._FULL_REVISION == "3da151be953883d3cd4e1ea0dc7a042ff2abf8f4"
    assert LV.SPLIT_ROWS == {"lite": 300, "full": 1319, "verified": 500}
    assert LV.SPLIT_SHA256["full"] == (
        "b231d1932ce614df07b6b4237db2a6f4eb6020213d0d9e919d95a62b976cf95e")


def test_live_one_setting_failing_keeps_the_others(monkeypatch):
    monkeypatch.setattr(LV, "check_swe_bench_live_prerequisites", lambda settings: (True, ""))
    monkeypatch.setattr(LV, "check_sweagent_prerequisites", lambda: (True, ""))

    def _execute(eval_name, model_name, base_url, concurrency, dataset, split, namespace, **k):
        if split == "full":
            raise RuntimeError("harness image vanished")
        return {"status": "success", "total_questions": 2, "correct_answers": 1,
                "accuracy": 50.0, "leaderboard_comparable": False,
                "leaderboard_comparable_reason": "subset run (--eval-limit 2)", "mode": A.AGENTIC,
                "sample_traces": [{"sample_idx": 0}, {"sample_idx": 1}]}
    monkeypatch.setattr(LV, "execute_sweagent_python", _execute)
    res = LV.run_swe_bench_live("m", "http://h:1/v1", 1, eval_categories="lite,full,verified")
    assert res["status"] == "completed_with_errors" and res["settings_with_errors"] == ["full"]
    assert "harness image vanished" in res["settings"]["full"]["error"]
    assert res["headline_metrics"]["full_resolved_rate"] is None
    assert res["headline_metrics"]["lite_resolved_rate"] == 50.0 and res["accuracy"] == 50.0
    assert "setting could not run (full)" in res["leaderboard_comparable_reason"]
    assert set(res["category_accuracy"]) == {"lite", "verified"}


def test_live_all_settings_failing_is_an_error(monkeypatch):
    results = {s: {"status": "error", "error": f"{s} broke", "leaderboard_comparable": False}
               for s in LV.SETTINGS}
    res = LV.combine_settings(results)
    assert res["status"] == "error" and "lite: lite broke" in res["error"]
    assert res["accuracy"] is None


def test_row_total_headline_only_when_an_evaluated_id_repeats():
    rows = [{"instance_id": "a"}, {"instance_id": "b"}, {"instance_id": "b"}, {"instance_id": "c"}]
    res = {"total_questions": 2, "correct_answers": 1, "accuracy": 50.0,
           "sample_traces": [{"extra_payload": {"instance_id": "a"}},
                             {"extra_payload": {"instance_id": "c"}}]}
    C.row_total_headline(res, rows)
    assert res == {"total_questions": 2, "correct_answers": 1, "accuracy": 50.0,
                   "sample_traces": res["sample_traces"]}
    res["sample_traces"].append({"extra_payload": {"instance_id": "b"}})
    res["total_questions"] = 3
    C.row_total_headline(res, rows)
    assert (res["total_questions"], res["total_instance_ids"], res["accuracy"]) == (4, 3, 25.0)
    # several attempts per id: left alone
    res2 = {"total_questions": 6, "correct_answers": 2,
            "sample_traces": [{"extra_payload": {"instance_id": i}} for i in "abcabc"]}
    C.row_total_headline(res2, rows)
    assert res2["total_questions"] == 6 and "total_instance_ids" not in res2


# ============================================================================================
# copilot_bench_swe / swe_bench_multilingual: split kwarg
# ============================================================================================
@pytest.mark.parametrize("mod,run,check", [
    (CB, "run_copilot_bench_swe", "check_copilot_bench_swe_prerequisites"),
    (ML, "run_swe_bench_multilingual", "check_swe_bench_multilingual_prerequisites"),
])
def test_v5_suites_accept_the_split_kwarg(monkeypatch, mod, run, check):
    monkeypatch.setattr(mod, check, lambda: (True, ""))
    monkeypatch.setattr(A, "require_agent_v2", lambda *a: None)
    seen = []

    def _execute(eval_name, model_name, base_url, concurrency, dataset, split, namespace, **k):
        seen.append((split, k))
        return {"status": "success"}
    monkeypatch.setattr(mod, "execute_swebench_agentic", _execute)
    getattr(mod, run)("m", "http://h:1/v1", 1, split="test")
    assert seen[0][0] == "test" and "split" not in seen[0][1]
    assert seen[0][1]["snapshot_rows"] == mod.SPLIT_ROWS["test"]
    with pytest.raises(ValueError, match="not a pinned split"):
        getattr(mod, run)("m", "http://h:1/v1", 1, split="dev")


# ============================================================================================
# docs
# ============================================================================================
def _doc(name):
    return " ".join(open(os.path.join(REPO, "docs", "evals", f"{name}.md"),
                         encoding="utf-8").read().split())


def test_live_doc_matches_the_settings():
    doc = _doc("swe_bench_live")
    for s in LV.PYTHON_SETTINGS:
        assert f"`{s}` @ `{LV.SETTING_REVISION[s]}` | {LV.SPLIT_ROWS[s]} | " \
               f"`{LV.SPLIT_SHA256[s]}`" in doc
    for key in ("lite_resolved_rate", "full_resolved_rate", "verified_resolved_rate",
                "headline_metrics", "total_instance_ids", "--eval-categories",
                "`total: 1319`", "`total: 500`", "settings_with_errors"):
        assert key in doc
    assert "count instance ids (499" not in doc
    assert "`[REDACTED]`" in doc and "`OPENAI_API_KEY`" in doc


def test_pro_doc_describes_the_forwarder():
    doc = _doc("swe_bench_pro")
    assert "--ae OPENAI_API_KEY=EMPTY" in doc and "harness.key_forwarder" in doc
    assert "harness.sandbox_api_key" in doc
    assert "passed into the Harbor container through the environment" not in doc


def test_no_internal_label_in_the_earlier_swe_test_docstring():
    text = open(os.path.join(REPO, "tests", "test_swe_report_retry_and_split_fingerprints.py"),
                encoding="utf-8").read()
    assert "reviewer" not in text.lower()
