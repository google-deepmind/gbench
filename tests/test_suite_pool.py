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

"""The --parallel-suites pool (gbench.pool) and the union of unsharded runs (merge.union_runs).

2026-10-09: the serial --eval-limit 5 sweep of the 84 built-in suites took 783 min. 2026-10-10:
three hand-made lanes ran, and one lane sat at 0 of 28 suites after 3 h 29 min behind
swe_bench_live, while gaia, gaia2 and wildclawbench collided on ports 18789-18795 when two of
them ran at once. The pool runs one gbench process per suite, longest first, with families kept
apart, an admission gate on the server's in-flight requests and one merged summary.

No network, no docker: processes, the clock and /metrics are faked.
"""

import json
from pathlib import Path

import pytest

from gbench import merge, pool
from gbench.cli import create_parser


# --------------------------------------------------------------------------- scheduling
def test_longest_first_with_unknown_suites_at_the_median():
    order, est = pool.order_longest_first(["a", "b", "c", "d"], {"a": 10, "b": 300, "c": 50})
    assert est["d"] == 50
    assert order == ["b", "c", "d", "a"]          # ties keep the given order (c before d)


def test_seed_table_covers_most_builtin_suites_and_orders_swe_live_first():
    durations = pool.load_durations(5, overlay_path=Path("/nonexistent/overlay.json"))
    assert len(durations) >= 80
    plan = pool.build_plan(["gsm8k", "swe_bench_live", "aime", "skillsbench"], 4, durations,
                           env={})
    assert plan.order[0] == "swe_bench_live" and plan.order[-1] == "gsm8k"


def test_family_caps_block_a_second_member_and_allow_two_swe():
    plan = pool.build_plan(["gaia", "gaia2", "wildclawbench", "swe_bench_live", "swe_bench_pro",
                            "copilot_bench_swe", "mrcr", "ruler", "aa_lcr"], 9, {}, env={})
    assert pool.blocked_by("gaia2", ["gaia"], plan) == "family gaia_ports at its cap of 1"
    assert pool.blocked_by("swe_bench_pro", ["swe_bench_live"], plan) is None
    assert pool.blocked_by("copilot_bench_swe", ["swe_bench_live", "swe_bench_pro"], plan)
    assert pool.blocked_by("aa_lcr", ["mrcr", "ruler"], plan)
    assert pool.next_suite(["gaia2", "aa_lcr", "swe_bench_pro"], ["gaia", "mrcr", "ruler"],
                           plan) == "swe_bench_pro"


def test_every_family_member_is_a_builtin_suite():
    from gbench.runners.evals import builtin_all_suites
    known = set(builtin_all_suites())
    for name, members, cap in pool.FAMILIES:
        assert members <= known, (name, members - known)
        assert cap >= 1
    assert pool.SEARCH_SUITES <= known


def test_search_rate_is_split_between_the_search_suites_allowed_at_once():
    plan = pool.build_plan(["browsecomp", "deepsearch_qa", "gaia", "gsm8k"], 8, {},
                           env={"GBENCH_SEARCH_MAX_RPS": "6"})
    assert plan.search_rps == {"browsecomp": 2.0, "deepsearch_qa": 2.0, "gaia": 2.0}
    fam = [f for f in plan.families if f[0] == "search"][0]
    assert fam[2] == 3
    # never more than SEARCH_MAX_CONCURRENT at once, each with 1/k of the rate
    many = ["browsecomp", "deepsearch_qa", "gaia", "bfcl", "bfcl_v4_agentic", "mcp_atlas"]
    plan = pool.build_plan(many, 8, {}, env={})
    k = pool.SEARCH_MAX_CONCURRENT
    assert set(plan.search_rps.values()) == {round(8.0 / k, 3)}
    assert [f for f in plan.families if f[0] == "search"][0][2] == k


# --------------------------------------------------------------------------- admission gate
METRICS = """# HELP vllm:num_requests_running Number of requests
vllm:num_requests_running{engine="0",model_name="m"} 30.0
vllm:num_requests_waiting{engine="0",model_name="m"} 12.0
vllm:num_requests_running{engine="1",model_name="m"} 7.0
vllm:gpu_cache_usage_perc{engine="0"} 0.5
"""


def test_parse_inflight_sums_running_and_waiting_over_engines():
    assert pool.parse_inflight(METRICS) == 49
    assert pool.parse_inflight("nothing here") is None


def test_admission_gate_closes_at_the_limit_and_opens_on_unreadable_metrics():
    clock = [0.0]
    gate = pool.AdmissionGate("http://x/metrics", 48, fetch=lambda _u: METRICS,
                              clock=lambda: clock[0])
    assert gate.open() is False and gate.last_inflight == 49
    gate = pool.AdmissionGate("http://x/metrics", 50, fetch=lambda _u: METRICS)
    assert gate.open() is True
    gate = pool.AdmissionGate("http://x/metrics", 1, fetch=lambda _u: None)
    assert gate.open() is True


def test_metrics_url_is_the_endpoint_root():
    assert pool.metrics_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/metrics"


# --------------------------------------------------------------------------- child arguments
def test_strip_dests_removes_pool_flags_with_values_and_abbreviations():
    parser = create_parser()
    argv = ["--evals", "gsm8k", "aime", "--remote-endpoint", "http://h:1/v1", "--thinking",
            "--parallel=4", "--smoke", "--results-dir", "/r", "--max-output-tokens", "65536",
            "--dry-run", "--tokenizer", "t"]
    out = pool.strip_dests(parser, argv, pool.POOL_OWNED_DESTS)
    assert out == ["--remote-endpoint", "http://h:1/v1", "--thinking", "--max-output-tokens",
                   "65536", "--tokenizer", "t"]
    child = pool.child_argv(out, "aime", Path("/r/aime"), ["--eval-limit", "1"])
    args = parser.parse_args(child)
    assert args.evals == ["aime"] and args.evals_only and args.eval_limit == 1
    assert str(args.results_dir) == "/r/aime" and args.eval_thinking
    assert args.max_output_tokens == 65536 and not args.smoke and args.parallel_suites is None


# --------------------------------------------------------------------------- run_pool
class FakeProc:
    def __init__(self, suite, life, registry, writes_result=True, code=0, directory=None):
        self.suite, self.life, self.registry = suite, life, registry
        self.pid = 1000 + len(registry["started"])
        self.code, self.writes, self.dir = code, writes_result, directory
        self.returncode = None

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        self.life -= 1
        if self.life > 0:
            return None
        self.returncode = self.code
        self.registry["running"].discard(self.suite)
        if self.writes:
            run = self.dir / "10-11-000000"
            run.mkdir(parents=True, exist_ok=True)
            (run / f"eval_{self.suite}_m_remote-endpoint.json").write_text(json.dumps({
                "benchmark_type": "eval", "eval_name": self.suite, "model_name": "m",
                "model_short": "m", "format": "remote-endpoint", "status": "success",
                "accuracy": 50.0, "total_questions": 2, "correct_answers": 1,
                "duration_s": 1.0}))
        return self.code


def _spawner(lives, registry, no_result=(), codes=None):
    def spawn(argv, stdout=None, stderr=None, start_new_session=None, env=None):
        suite = argv[argv.index("--evals") + 1]
        d = Path(argv[argv.index("--results-dir") + 1])
        registry["started"].append(suite)
        registry["env"][suite] = env
        registry["running"].add(suite)
        registry["max"] = max(registry["max"], len(registry["running"]))
        registry["overlaps"].append(set(registry["running"]))
        return FakeProc(suite, lives.get(suite, 2), registry, suite not in no_result,
                        (codes or {}).get(suite, 0), d)
    return spawn


def _registry():
    return {"started": [], "running": set(), "max": 0, "overlaps": [], "env": {}}


def _opts(n):
    return pool.PoolOptions(parallel=n, stall_s=0, poll_s=0, scan_s=1e9, reap_containers=False)


def test_run_pool_keeps_families_apart_and_the_pool_size(tmp_path):
    suites = ["gaia", "gaia2", "wildclawbench", "gsm8k", "ifeval", "aime", "hmmt", "swe_bench_live",
              "swe_bench_pro", "copilot_bench_swe"]
    durations = {s: float(i) for i, s in enumerate(suites)}
    plan = pool.build_plan(suites, 4, durations, env={})
    reg = _registry()
    lives = {s: 3 + i for i, s in enumerate(suites)}
    manifest = pool.run_pool(plan, ["--thinking"], tmp_path, _opts(4),
                             spawn=_spawner(lives, reg), clock=lambda: 0.0,
                             sleep=lambda _s: None, install_signals=False)
    assert sorted(reg["started"]) == sorted(suites)
    assert reg["started"][0] == "copilot_bench_swe"           # the longest first
    assert reg["max"] <= 4
    for running in reg["overlaps"]:
        assert len(running & {"gaia", "gaia2", "wildclawbench"}) <= 1
        assert len(running & {"aime", "hmmt"}) <= 1
        assert len(running & pool.SWE_FAMILY) <= 2
    assert pool.pool_exit_code(manifest) == 0
    rec = manifest["suites"]["gsm8k"]
    assert rec["result_files"] and rec["concurrency"]["parallel_suites"] == 4
    assert json.loads((tmp_path / "pool.json").read_text())["parallel_suites"] == 4
    env = reg["env"]["gsm8k"]
    record = json.loads(env["GBENCH_POOL_RECORD"])
    assert record["parallel_suites"] == 4 and isinstance(record["running_at_start"], list)
    assert env["GBENCH_POOL_ACTIVITY_FILE"].endswith("gsm8k/.activity")


def test_run_pool_passes_the_search_rate_and_extra_env(tmp_path):
    plan = pool.build_plan(["browsecomp", "deepsearch_qa"], 2, {}, env={})
    reg = _registry()
    pool.run_pool(plan, [], tmp_path, _opts(2), spawn=_spawner({}, reg),
                  child_env=lambda s, _o: {"GBENCH_SMOKE": "1"},
                  clock=lambda: 0.0, sleep=lambda _s: None, install_signals=False)
    assert reg["env"]["browsecomp"]["GBENCH_SEARCH_MAX_RPS"] == "4.0"
    assert reg["env"]["deepsearch_qa"]["GBENCH_SMOKE"] == "1"


def test_admission_gate_holds_new_suites_while_the_server_is_full(tmp_path):
    plan = pool.build_plan(["a1", "a2", "a3"], 3, {"a1": 3, "a2": 2, "a3": 1}, env={})
    reg = _registry()
    gate = pool.AdmissionGate("http://x/metrics", 10, fetch=lambda _u: METRICS)
    pool.run_pool(plan, [], tmp_path, _opts(3), gate=gate, spawn=_spawner({}, reg),
                  clock=lambda: 0.0, sleep=lambda _s: None, install_signals=False)
    # the first suite always starts; the closed gate keeps the others for when it is alone
    assert reg["max"] == 1 and reg["started"] == ["a1", "a2", "a3"]


def test_a_suite_without_result_makes_the_pool_fail_and_is_in_the_merged_summary(tmp_path):
    plan = pool.build_plan(["gsm8k", "ifeval"], 2, {}, env={})
    reg = _registry()
    manifest = pool.run_pool(plan, [], tmp_path, _opts(2),
                             spawn=_spawner({}, reg, no_result={"ifeval"}, codes={"ifeval": 2}),
                             clock=lambda: 0.0, sleep=lambda _s: None, install_signals=False)
    assert pool.pool_exit_code(manifest) == 2
    rec = manifest["suites"]["ifeval"]
    missing = pool._missing_result_record("ifeval", rec)
    merge.union_runs([tmp_path / "gsm8k"], tmp_path, link=True, extra_failures=[missing])
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["successful_configurations"] == 1 and summary["failed_configurations"] == 1
    assert "exited 2 without writing a result" in summary["failures"][0]["error"]
    link = tmp_path / "eval_gsm8k_m_remote-endpoint.json"
    assert link.is_symlink() and json.loads(link.read_text())["accuracy"] == 50.0
    csv = (tmp_path / "eval_summary.csv").read_text()
    assert "GSM8K" in csv and "IFEVAL" in csv


# --------------------------------------------------------------------------- containers
def test_container_attribution_prefers_the_longest_suite_name():
    names = {"gbench-gaia-1", "gbench-gaia2-abc", "gbench-bfcl-x", "gbench-bfcl-live-y",
             "other"}
    assert pool.attribute_names(names, "gaia", ["gaia", "gaia2"]) == {"gbench-gaia-1"}
    assert pool.attribute_names(names, "bfcl", ["bfcl", "bfcl_live"]) == {"gbench-bfcl-x"}
    assert pool.attribute_names(names, "bfcl_live", ["bfcl", "bfcl_live"]) == \
        {"gbench-bfcl-live-y"}


# --------------------------------------------------------------------------- durations
def test_record_durations_overlay_is_keyed_by_eval_limit(tmp_path):
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps({"suites": {"a": {"duration_s": 10}, "b": {"duration_s": 20}}}))
    overlay = tmp_path / "overlay.json"
    pool.record_durations({"a": 99.0}, 1, overlay_path=overlay, source="x")
    assert pool.load_durations(1, seed, overlay) == {"a": 99.0, "b": 20.0}
    assert pool.load_durations(5, seed, overlay) == {"a": 10.0, "b": 20.0}


# --------------------------------------------------------------------------- merge
def _result(d: Path, suite: str, shard=None, acc=40.0):
    d.mkdir(parents=True, exist_ok=True)
    rec = {"benchmark_type": "eval", "eval_name": suite, "model_name": "m", "model_short": "m",
           "format": "remote-endpoint", "status": "success", "accuracy": acc,
           "total_questions": 5, "correct_answers": 2,
           "sample_traces": [{"is_correct": True}, {"is_correct": False}]}
    if shard:
        rec["shard"] = shard
    name = f"eval_{suite}_m_remote-endpoint" + (f".shard-{shard.replace('/', 'of')}" if shard
                                                else "") + ".json"
    (d / name).write_text(json.dumps(rec))
    (d / "metadata.json").write_text(json.dumps({"gbench": {"commit": "abc"}}))


def test_merge_joins_unsharded_runs_of_different_suites(tmp_path):
    _result(tmp_path / "r1", "gsm8k")
    _result(tmp_path / "r2", "ifeval", acc=80.0)
    written = merge.merge_runs([tmp_path / "r1", tmp_path / "r2"], tmp_path / "out")
    assert sorted(p.name for p in written) == ["eval_gsm8k_m_remote-endpoint.json",
                                               "eval_ifeval_m_remote-endpoint.json"]
    out = tmp_path / "out"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["successful_configurations"] == 2
    assert all("sample_traces" not in r for r in summary["models"])
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["gbench"] == {"commit": "abc"} and set(meta["suites"]) == {"gsm8k", "ifeval"}
    # the files are unchanged copies
    assert json.loads((out / "eval_ifeval_m_remote-endpoint.json").read_text())["accuracy"] == 80


def test_merge_refuses_a_suite_found_twice_and_a_mix_of_shards(tmp_path):
    _result(tmp_path / "r1", "gsm8k")
    _result(tmp_path / "r2", "gsm8k")
    with pytest.raises(merge.MergeError, match="more than one run"):
        merge.merge_runs([tmp_path / "r1", tmp_path / "r2"], tmp_path / "out")
    _result(tmp_path / "r3", "ifeval", shard="1/2")
    with pytest.raises(merge.MergeError, match="mix sharded and unsharded"):
        merge.merge_runs([tmp_path / "r1", tmp_path / "r3"], tmp_path / "out2")
    assert not (tmp_path / "out").exists() and not (tmp_path / "out2").exists()


# --------------------------------------------------------------------------- processes
def test_first_signal_sends_sigterm_to_each_child_group_and_the_second_kills(tmp_path):
    import signal
    import subprocess
    import sys
    code = ("import signal,sys,time\n"
            "signal.signal(signal.SIGTERM, lambda *a: sys.exit(7))\n"
            "print('ready', flush=True)\n"
            "time.sleep(60)\n")
    stubborn = "import signal,time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n" \
               "print('ready', flush=True)\ntime.sleep(60)\n"
    procs = [subprocess.Popen([sys.executable, "-c", c], start_new_session=True,
                              stdout=subprocess.PIPE) for c in (code, stubborn)]
    for p in procs:
        assert p.stdout.readline().strip() == b"ready"
    children = [pool.Child(f"s{i}", p, tmp_path, tmp_path / "log", 0.0, "", [])
                for i, p in enumerate(procs)]
    sig = pool._Signals(lambda: children)
    sig._handler(signal.SIGINT, None)
    assert sig.stop == signal.SIGINT
    assert procs[0].wait(timeout=10) == 7                 # SIGTERM, handled by the child
    assert procs[1].poll() is None                         # ignores SIGTERM
    sig._handler(signal.SIGINT, None)
    assert procs[1].wait(timeout=10) == -signal.SIGKILL


def test_docker_run_names_are_read_from_the_child_process_group(tmp_path):
    import subprocess
    import sys
    import time
    # a process whose command line reads `docker run --name gbench-tau3-abc` (the interpreter
    # runs the script file named `run`)
    (tmp_path / "run").write_text("import time\ntime.sleep(30)\n")
    p = subprocess.Popen(["docker", "run", "--name", "gbench-tau3-abc"], executable=sys.executable,
                         cwd=tmp_path, start_new_session=True)
    try:
        names = set()
        for _ in range(100):
            names = pool._docker_names_in_group(p.pid)
            if names:
                break
            time.sleep(0.05)
        assert p.poll() is None and names == {"gbench-tau3-abc"}
        assert pool._docker_names_in_group(p.pid + 999999) == set()
    finally:
        p.kill()
        p.wait()


def test_stall_alarm_warns_once_per_silent_interval_and_rearms_on_activity(tmp_path, caplog):
    import logging
    import os
    import time
    d = tmp_path / "aime"
    d.mkdir()
    (d / "gbench.log").write_text("x")
    old = time.time() - 3600
    for p in (d / "gbench.log", d):
        os.utime(p, (old, old))

    class _P:
        pid = None
    c = pool.Child("aime", _P(), d, d / "gbench.log", 0.0, "", [], last_activity=old)
    opts = pool.PoolOptions(parallel=1, stall_s=600, reap_containers=False)
    with caplog.at_level(logging.WARNING, logger="gbench.pool"):
        pool._scan(c, 0.0, opts)
        pool._scan(c, 0.0, opts)                 # the same silent interval: no second alert
    assert c.stall_alerts == 1 and "STALL? aime" in caplog.text and "60 min" in caplog.text
    (d / "gbench.log").write_text("progress")    # activity resets the clock
    pool._scan(c, 0.0, opts)
    assert c.stall_alerts == 1 and time.time() - c.last_activity < 60


def test_custom_jsonl_flag_reaches_only_the_custom_jsonl_process():
    parser = create_parser()
    per = pool.suite_argv(parser, ["--evals", "all", "--eval-custom-jsonl", "/d.jsonl",
                                   "--parallel-suites", "4", "--thinking"])
    assert per("gsm8k") == ["--thinking"]
    assert per("custom_jsonl") == ["--eval-custom-jsonl", "/d.jsonl", "--thinking"]


def test_cli_refuses_a_pool_that_also_selects_performance_pillars(capsys):
    from gbench.cli import main
    with pytest.raises(SystemExit):
        main(["--evals", "gsm8k", "--parallel-suites", "2", "--input-lengths", "128",
              "--output-lengths", "128", "--max-output-tokens", "100",
              "--remote-endpoint", "http://127.0.0.1:1/v1", "--tokenizer", "t"])
    assert "run eval suites only" in capsys.readouterr().err


# --------------------------------------------------------------------------- 2026-10-11 review
def test_merge_of_a_pool_directory_counts_each_linked_result_once(tmp_path):
    # 2026-10-11 review: `gbench --merge <pool-dir>` was refused, each suite appearing twice
    # (the pool's top-level link next to <suite>/<ts>/eval_*.json)
    for name, suites in (("pool-a", ["gsm8k", "ifeval"]), ("pool-b", ["aime"])):
        root = tmp_path / name
        for s in suites:
            _result(root / s / "10-11-000000", s)
        merge.union_runs([root / s for s in suites], root, link=True)
        assert (root / "eval_gsm8k_m_remote-endpoint.json").is_symlink() or name == "pool-b"
    written = merge.merge_runs([tmp_path / "pool-a"], tmp_path / "out1")
    assert sorted(p.name for p in written) == ["eval_gsm8k_m_remote-endpoint.json",
                                               "eval_ifeval_m_remote-endpoint.json"]
    written = merge.merge_runs([tmp_path / "pool-a", tmp_path / "pool-b"], tmp_path / "out2")
    assert len(written) == 3
    meta = json.loads((tmp_path / "out2" / "metadata.json").read_text())
    assert meta["gbench"] == {"commit": "abc"}     # the suite's own metadata, by its real path


def test_reap_removes_only_containers_created_after_the_child_started(monkeypatch):
    # 2026-10-11 review: a gsm8k child's exit removed an unrelated `gbench-gsm8k-otherrun-...`
    # container, and a putnam child would remove putnam_formal's `gbench-putnam-<hex>` checkers
    present = {"gbench-gsm8k-otherrun", "gbench-gsm8k-mine", "gbench-putnam-1a2b", "named-run",
               "gbench-gsm8k-before-pool"}
    removed = []
    monkeypatch.setattr(pool, "container_names", lambda: set(present))
    monkeypatch.setattr(pool, "_docker", lambda args, timeout=60.0: (removed.extend(args[2:]),
                                                                     (0, ""))[1])
    c = pool.Child("gsm8k", None, Path("/x"), Path("/x/log"), 0.0, "", [],
                   docker_names={"named-run"},
                   containers_before={"gbench-gsm8k-otherrun", "gbench-putnam-1a2b"})
    pool._reap(c, ["gsm8k", "putnam"], baseline={"gbench-gsm8k-before-pool"})
    assert sorted(removed) == ["gbench-gsm8k-mine", "named-run"]
    removed.clear()
    p = pool.Child("putnam", None, Path("/x"), Path("/x/log"), 0.0, "", [],
                   containers_before={"gbench-putnam-1a2b"})
    pool._reap(p, ["putnam", "putnam_formal"], baseline=set())
    assert removed == []


def _write(d: Path, suite: str, **kw):
    d.mkdir(parents=True, exist_ok=True)
    rec = {"benchmark_type": "eval", "eval_name": suite, "status": "success"}
    rec.update(kw)
    path = d / f"eval_{suite}_m_remote-endpoint.json"
    path.write_text(json.dumps(rec))
    return path


def test_a_failed_or_stopped_suite_only_raises_its_duration(tmp_path):
    # 2026-10-11 review: a swe_bench_live record that failed after 9 s overwrote its estimate
    # and the next pool started the longest suite last
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps({"suites": {"swe_bench_live": {"duration_s": 13000},
                                           "gsm8k": {"duration_s": 16}, "aime": {"duration_s": 1200},
                                           "tau2": {"duration_s": 300}}}))
    overlay = tmp_path / "overlay.json"
    root = tmp_path / "pool"
    files = {
        "swe_bench_live": _write(root / "swe_bench_live", "swe_bench_live", status="error",
                                 failed=True, error="stopped by SIGTERM"),
        "gsm8k": _write(root / "gsm8k", "gsm8k"),
        "aime": _write(root / "aime", "aime", status="completed_with_errors"),
        "tau2": _write(root / "tau2", "tau2", partial=True),
    }
    rel = lambda s: [str(files[s].relative_to(root))]                       # noqa: E731
    manifest = {"suites": {
        "swe_bench_live": {"exit_code": 0, "duration_s": 9.0,
                           "result_files": rel("swe_bench_live")},
        "gsm8k": {"exit_code": 0, "duration_s": 30.0, "result_files": rel("gsm8k")},
        "aime": {"exit_code": 0, "duration_s": 2000.0, "result_files": rel("aime")},
        "tau2": {"exit_code": 0, "duration_s": 100.0, "result_files": rel("tau2")},
        "ifeval": {"not_started": True, "exit_code": None, "duration_s": None},
    }}
    clean, lower = pool.duration_updates(manifest, root)
    assert clean == {"gsm8k": 30.0}
    assert lower == {"swe_bench_live": 9.0, "aime": 2000.0, "tau2": 100.0}
    pool.record_durations(clean, 5, overlay_path=overlay, raise_only=lower, seed_path=seed)
    got = pool.load_durations(5, seed, overlay)
    assert got == {"swe_bench_live": 13000.0, "gsm8k": 30.0, "aime": 2000.0, "tau2": 300.0}
    plan = pool.build_plan(["gsm8k", "swe_bench_live", "aime"], 8, got, env={})
    assert plan.order[0] == "swe_bench_live"
    # a suite the pool stopped (deadline, hang) is a lower bound even when it exited 0
    manifest["suites"]["gsm8k"]["stop_reason"] = "deadline"
    assert pool.duration_updates(manifest, root)[0] == {}


def test_smoke_durations_have_their_own_key(tmp_path):
    overlay = tmp_path / "overlay.json"
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps({"suites": {"a": {"duration_s": 500}}}))
    pool.record_durations({"a": 5.0}, pool.SMOKE_KEY, overlay_path=overlay)
    assert pool.load_durations(1, seed, overlay) == {"a": 500.0}
    assert pool.load_durations(pool.SMOKE_KEY, seed, overlay) == {"a": 5.0}


def test_seed_entries_from_unclean_runs_are_marked_and_terminal_bench_is_not_short():
    # 2026-10-11 review: terminal_bench's seed was 326 s from a run whose Harbor trials were
    # cancelled (complete runs: 9638 s for 3 samples), so it was scheduled 31st of 84
    data = json.loads(pool.DURATIONS_FILE.read_text())
    for name, entry in data["suites"].items():
        if entry.get("status") != "success":
            assert entry.get("estimate"), name
    durations = pool.load_durations(5, overlay_path=Path("/nonexistent/overlay.json"))
    top = pool.order_longest_first(sorted(durations), durations)[0][:3]
    assert "terminal_bench" in top and "swe_bench_live" in top


class _SignalProc(FakeProc):
    """A child that exits a few polls after a signal, recording it."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.signals = []

    def send_signal(self, sig):
        self.signals.append(sig)
        self.life = min(self.life, 2)


def _signal_spawner(lives, registry, procs):
    def spawn(argv, stdout=None, stderr=None, start_new_session=None, env=None):
        suite = argv[argv.index("--evals") + 1]
        d = Path(argv[argv.index("--results-dir") + 1])
        registry["started"].append(suite)
        registry["running"].add(suite)
        proc = _SignalProc(suite, lives.get(suite, 2), registry, True, 0, d)
        procs[suite] = proc
        return proc
    return spawn


def _no_killpg(monkeypatch):
    def fake(_pid, _sig):
        raise OSError("fake pid")                  # falls back to proc.send_signal
    monkeypatch.setattr(pool.os, "killpg", fake)


def test_deadline_stops_the_running_suites_and_starts_no_more(tmp_path, monkeypatch):
    _no_killpg(monkeypatch)
    import signal
    plan = pool.build_plan(["a", "b", "c"], 2, {"a": 3, "b": 2, "c": 1}, env={})
    reg, procs = _registry(), {}
    t = {"now": 0.0}

    def clock():
        t["now"] += 1.0
        return t["now"]
    opts = _opts(2)
    opts.deadline_s = 5.0
    manifest = pool.run_pool(plan, [], tmp_path, opts,
                             spawn=_signal_spawner({"a": 1000, "b": 1000}, reg, procs),
                             clock=clock, sleep=lambda _s: None, install_signals=False)
    assert reg["started"] == ["a", "b"]
    assert procs["a"].signals == [signal.SIGTERM] and procs["b"].signals == [signal.SIGTERM]
    assert manifest["deadline_reached"] is True
    assert manifest["suites"]["a"]["stop_reason"].startswith("deadline")
    assert manifest["suites"]["c"]["not_started"] and \
        manifest["suites"]["c"]["not_started_by"] == "the deadline"
    assert pool.pool_exit_code(manifest) == 2
    assert "deadline" in pool._missing_result_record("c", manifest["suites"]["c"])["error"]


def test_a_watch_reason_stops_that_suite_and_kills_it_after_the_grace(tmp_path, monkeypatch):
    _no_killpg(monkeypatch)
    import signal

    class Stubborn(_SignalProc):
        def send_signal(self, sig):                  # ignores SIGTERM, dies on SIGKILL
            self.signals.append(sig)
            if sig == signal.SIGKILL:
                self.life = 1

    procs = {}

    def spawn(argv, stdout=None, stderr=None, start_new_session=None, env=None):
        suite = argv[argv.index("--evals") + 1]
        d = Path(argv[argv.index("--results-dir") + 1])
        reg["started"].append(suite)
        procs[suite] = (Stubborn if suite == "hung" else _SignalProc)(
            suite, 10**6 if suite == "hung" else 3, reg, True, 0, d)
        return procs[suite]
    reg = _registry()
    t = {"now": 0.0}

    def clock():
        t["now"] += 10.0
        return t["now"]
    plan = pool.build_plan(["hung", "ok"], 2, {"hung": 2, "ok": 1}, env={})
    manifest = pool.run_pool(plan, [], tmp_path, _opts(2), spawn=spawn,
                             watch=lambda c: "hang: test" if c.suite == "hung" else None,
                             clock=clock, sleep=lambda _s: None, install_signals=False)
    assert procs["hung"].signals == [signal.SIGTERM, signal.SIGKILL]
    assert procs["ok"].signals == []
    assert manifest["suites"]["hung"]["stop_reason"] == "hang: test"
    assert manifest["suites"]["ok"]["stop_reason"] is None


def test_cli_refuses_shard_with_a_pool_and_skip_existing_with_smoke(capsys, monkeypatch):
    from gbench.cli import main
    base = ["--evals", "gsm8k,ifeval", "--evals-only", "--max-output-tokens", "100",
            "--remote-endpoint", "http://127.0.0.1:9/v1", "--tokenizer", "x"]
    with pytest.raises(SystemExit) as e:
        main([*base, "--parallel-suites", "2", "--shard", "1/2"])
    assert e.value.code == 2 and "--shard / GBENCH_SHARD cannot" in capsys.readouterr().err
    monkeypatch.setenv("GBENCH_SHARD", "1/2")
    with pytest.raises(SystemExit):
        main([*base, "--smoke"])
    assert "--shard / GBENCH_SHARD cannot" in capsys.readouterr().err
    monkeypatch.delenv("GBENCH_SHARD")
    with pytest.raises(SystemExit):
        main([*base, "--smoke", "--skip-existing"])
    assert "--skip-existing cannot be combined with --smoke" in capsys.readouterr().err


def test_skip_existing_in_a_pool_child_searches_the_runs_results_dir(tmp_path, monkeypatch):
    # 2026-10-11 review: each pool child's --results-dir is a new pool-*/<suite>/ directory, so
    # --skip-existing found nothing a restarted pool could reuse
    from types import SimpleNamespace
    from gbench.runners.evals import EvalsBenchmarkRunner
    name = "eval_gsm8k_m_remote-endpoint.json"
    old = _write(tmp_path / "pool-old" / "gsm8k" / "10-10-000000", "gsm8k",
                 finished_at="2026-10-10T00:00:00+00:00")
    (tmp_path / "pool-old" / name).symlink_to(old)          # the old pool's top-level link
    child_dir = tmp_path / "pool-new" / "gsm8k"
    child_dir.mkdir(parents=True)
    seen = []
    stub = SimpleNamespace(config=SimpleNamespace(log_manager=SimpleNamespace(base_dir=child_dir)),
                           _reuse_mismatch=lambda e, d, s, p: seen.append(d) or None)
    out = child_dir / "10-11-000000" / name
    monkeypatch.delenv("GBENCH_REUSE_ROOT", raising=False)
    assert EvalsBenchmarkRunner._reusable_result(stub, "gsm8k", name, out, {}, {}) is None
    monkeypatch.setenv("GBENCH_REUSE_ROOT", str(tmp_path))
    hit = EvalsBenchmarkRunner._reusable_result(stub, "gsm8k", name, out, {}, {})
    assert hit is not None and hit[0].resolve() == old.resolve()
    assert len(seen) == 1                          # the link and its target are one candidate


def test_a_real_run_never_reuses_a_smoke_result(monkeypatch):
    from types import SimpleNamespace
    from gbench.runners.evals import EvalsBenchmarkRunner
    monkeypatch.delenv("GBENCH_SMOKE", raising=False)
    why = EvalsBenchmarkRunner._reuse_mismatch(SimpleNamespace(), "gsm8k",
                                               {"smoke": True, "status": "success"}, {}, {})
    assert "smoke result" in why
