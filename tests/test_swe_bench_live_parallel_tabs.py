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

"""swe_bench_live runs its selected tabs at the same time (tab_schedule / run_tab_schedule).

2026-10-09 smoke run: --eval-limit 5 over Lite, Verified, C/C++, Go and Rust ran five SWE-agent
batches and harness passes one after the other, ~3 h. The tabs now run in lanes at once (the
Python tabs, whose instance ids overlap, one after the other in one lane), within the
--sandboxes budget, each tab exactly as alone. These tests drive run_swe_bench_live with fake
tabs: the lanes overlap (a barrier only the concurrent schedule can pass), the Python tabs never
do, the budget is never exceeded, the combined result is the serial run's, and a missing
prerequisite or a stop still ends the suite after the running tabs end.
"""

import threading
import time

import pytest

from gbench.runners.eval_suites import swe_agentic as SA
from gbench.runners.eval_suites import swe_bench_live as LV

LINUX = [s for s in LV.SETTINGS if s != "windows"]
SMOKE_TABS = ["lite", "verified", "ccpp", "go", "rust"]   # the 2026-10-09 --eval-limit 5 draw


def test_lanes_put_the_python_tabs_together_and_every_other_tab_alone():
    assert LV.tab_lanes(LINUX) == [["lite", "full", "verified"], ["ccpp"], ["csharp"], ["java"],
                                   ["tsjs"], ["go"], ["rust"]]
    assert LV.tab_lanes(["go", "verified", "rust", "lite"]) == [["go"], ["verified", "lite"],
                                                                ["rust"]]
    assert LV.tab_lanes(["windows"]) == [["windows"]]


def test_the_smoke_run_shape_runs_four_lanes_at_once_one_worker_each():
    sched = LV.tab_schedule(SMOKE_TABS, {t: 1 for t in SMOKE_TABS}, 16)
    assert sched["lanes"] == [["lite", "verified"], ["ccpp"], ["go"], ["rust"]]
    assert sched["concurrent_lanes"] == 4
    assert sched["workers"] == {t: 1 for t in SMOKE_TABS}


def test_the_budget_is_dealt_over_the_lanes_and_never_exceeded():
    rows = {s: LV.setting_rows(s) for s in LINUX}
    for budget in (1, 3, 7, 8, 16, 64, 500):
        sched = LV.tab_schedule(LINUX, rows, budget)
        lane_workers = [sched["workers"][lane[0]] for lane in sched["lanes"]]
        assert all(sched["workers"][t] == sched["workers"][lane[0]]
                   for lane in sched["lanes"] for t in lane)
        assert min(lane_workers) >= 1
        # what can run at one time: the `concurrent_lanes` largest lanes
        at_once = sum(sorted(lane_workers, reverse=True)[:sched["concurrent_lanes"]])
        assert at_once <= budget, (budget, sched)
        if budget >= len(sched["lanes"]):
            assert sum(lane_workers) == min(budget, sum(max(rows[t] for t in lane)
                                                        for lane in sched["lanes"]))
        else:
            assert sched["concurrent_lanes"] == budget and set(lane_workers) == {1}
    # the Python lane (1,319 + 500 + 300 instances one after the other) gets the largest share
    sched = LV.tab_schedule(LINUX, rows, 16)
    assert sched["workers"]["lite"] == max(sched["workers"].values()) > sched["workers"]["go"]
    # a lane never gets more workers than its largest tab has instances
    assert LV.tab_schedule(["go", "rust"], {"go": 1, "rust": 50}, 16)["workers"] == {
        "go": 1, "rust": 15}


def _suite(monkeypatch, run_setting):
    monkeypatch.setattr(LV.swe_agentic, "resolve_mode", lambda name: (SA.AGENTIC, "default"))
    monkeypatch.setattr(LV, "missing_prerequisites",
                        lambda settings, mode: {"windows": "not a Windows host"})
    monkeypatch.setattr(LV.swebench_common, "require_task_network_blocked",
                        lambda *a, **k: None)
    monkeypatch.setattr(LV, "_run_setting", run_setting)
    # no stop requested (another test may have left the process-wide flag set)
    monkeypatch.setattr(LV, "_raise_if_stop_requested", lambda where: None)


def _tab_result(setting, kwargs):
    n = kwargs.get("limit") or LV.setting_rows(setting)
    return {"status": "success", "accuracy": 100.0 * (len(setting) % 3) / 3,
            "total_questions": n, "correct_answers": len(setting) % 3,
            "leaderboard_comparable": False,
            "leaderboard_comparable_reason": kwargs.get("limit_note") or "full",
            "tab": LV.TAB_LABELS[setting],
            "sample_traces": [{"sample_idx": i, "instance_id": f"{setting}-{i}"}
                              for i in range(n)]}


def _seeded_limit_5(monkeypatch):
    """--eval-limit 5 dealt as on 2026-10-09: Lite, Verified, C/C++, Go, Rust one each."""
    def shares(limit, runnable):
        got = {s: (1 if s in SMOKE_TABS else 0) for s in runnable}
        return got, {"limit": limit, "samples": 5, "per_setting": got}
    monkeypatch.setattr(LV, "limit_shares", shares)


def test_the_lanes_overlap_and_the_python_tabs_do_not(monkeypatch):
    _seeded_limit_5(monkeypatch)
    # 4 lanes: the first tab of each must be running at the same time to pass this barrier
    barrier = threading.Barrier(4, timeout=20)
    lock = threading.Lock()
    active, peak, python_active, python_peak, conc_seen = [0], [0], [0], [0], {}
    windows = {}

    def run_setting(setting, mode, model, url, conc, kwargs):
        with lock:
            active[0] += conc
            peak[0] = max(peak[0], active[0])
            if LV.setting_family(setting) == "python":
                python_active[0] += 1
                python_peak[0] = max(python_peak[0], python_active[0])
            conc_seen[setting] = conc
        start = time.monotonic()
        try:
            if setting != "verified":       # verified waits for lite (same lane)
                barrier.wait()
            time.sleep(0.2)
            return _tab_result(setting, kwargs)
        finally:
            with lock:
                active[0] -= conc
                if LV.setting_family(setting) == "python":
                    python_active[0] -= 1
            windows[setting] = (start, time.monotonic())
    _suite(monkeypatch, run_setting)
    res = LV.run_swe_bench_live("m", "http://h/v1", 16, limit=5)
    assert barrier.broken is False and set(windows) == set(SMOKE_TABS)
    # every MultiLang tab overlapped every other one, and Lite
    for a in ("lite", "ccpp", "go", "rust"):
        for b in ("lite", "ccpp", "go", "rust"):
            if a != b:
                assert windows[a][0] < windows[b][1] and windows[b][0] < windows[a][1], (a, b)
    # the Python tabs ran one after the other
    assert python_peak[0] == 1 and windows["lite"][1] <= windows["verified"][0]
    assert peak[0] <= 16 and conc_seen == {t: 1 for t in SMOKE_TABS}
    assert res["tab_schedule"]["lanes"] == [["lite", "verified"], ["ccpp"], ["go"], ["rust"]]
    assert res["tab_schedule"]["concurrent_lanes"] == 4


def test_the_combined_result_is_the_serial_runs(monkeypatch):
    """Run the same fake tabs with a budget of 1 (one lane at a time, the old order) and of 16
    (all lanes at once, finishing in another order): every tab's result, the suite headline, the
    settings' order and every other field are the same; only tab_schedule differs."""
    _seeded_limit_5(monkeypatch)
    order = []

    def run_setting(setting, mode, model, url, conc, kwargs):
        # the later a tab comes, the sooner it finishes when the lanes overlap
        time.sleep(0.05 * (len(LINUX) - LINUX.index(setting)))
        order.append(setting)
        return _tab_result(setting, kwargs)
    _suite(monkeypatch, run_setting)
    serial = LV.run_swe_bench_live("m", "http://h/v1", 1, limit=5)
    serial_order = list(order)
    order.clear()
    parallel = LV.run_swe_bench_live("m", "http://h/v1", 16, limit=5)
    assert serial_order == SMOKE_TABS and order != serial_order    # really another order
    s_sched, p_sched = serial.pop("tab_schedule"), parallel.pop("tab_schedule")
    assert s_sched["concurrent_lanes"] == 1 and p_sched["concurrent_lanes"] == 4
    assert serial == parallel
    assert list(parallel["settings"]) == list(serial["settings"])
    assert parallel["settings"]["windows"]["status"] == "infra_required"


def test_a_failed_tab_keeps_the_others(monkeypatch):
    _seeded_limit_5(monkeypatch)

    def run_setting(setting, mode, model, url, conc, kwargs):
        if setting == "go":
            raise ValueError("go broke")
        return _tab_result(setting, kwargs)
    _suite(monkeypatch, run_setting)
    res = LV.run_swe_bench_live("m", "http://h/v1", 16, limit=5)
    assert res["settings"]["go"]["status"] == "error"
    assert res["settings"]["go"]["error"] == "ValueError: go broke"
    assert all(res["settings"][t]["status"] == "success" for t in SMOKE_TABS if t != "go")


@pytest.mark.parametrize("fatal", ["infra", "stop"])
def test_a_missing_prerequisite_or_a_stop_waits_for_the_running_tabs_and_starts_no_more(
        monkeypatch, fatal):
    """Budget 2, four lanes: Lite and C/C++ start; C/C++ hits a missing prerequisite (or the run
    is stopped); Lite, already running, is waited for; Verified (Lite's lane), Go and Rust never
    start; the suite raises what C/C++ raised."""
    _seeded_limit_5(monkeypatch)
    started, finished = [], []
    lite_running = threading.Event()
    stopped = []

    def run_setting(setting, mode, model, url, conc, kwargs):
        started.append(setting)
        if setting == "lite":
            lite_running.set()
            time.sleep(0.5)
            finished.append(setting)
            return _tab_result(setting, kwargs)
        assert lite_running.wait(10)
        if fatal == "infra":
            raise LV.infra_required("swe_bench_live", "the docker socket vanished", LV.DOCS_URL)
        stopped.append(True)
        raise SystemExit(143)
    _suite(monkeypatch, run_setting)
    monkeypatch.setattr(LV, "_raise_if_stop_requested",
                        lambda where: (_ for _ in ()).throw(SystemExit(143)) if stopped else None)
    expected = RuntimeError if fatal == "infra" else SystemExit
    with pytest.raises(expected) as e:
        LV.run_swe_bench_live("m", "http://h/v1", 2, limit=5)
    if fatal == "infra":
        assert str(e.value).startswith("[INFRA REQUIRED]")
    else:
        assert e.value.code == 143
    assert sorted(started) == ["ccpp", "lite"] and finished == ["lite"]


def test_a_signal_under_suite_timeout_reaches_every_running_tab(monkeypatch, tmp_path):
    """--suite-timeout runs the suite in a worker thread, and the CLI's SIGTERM handler raises
    SystemExit in the main thread only: evals._run_with_timeout hands the signal to the suite's
    thread. With two tabs running (Lite and C/C++, each in its own tab thread with a real
    SweagentWatch, as run_sweagent does), a real SIGTERM must reach both tab threads: both
    watchdogs stop their batches, neither tab scores its rollouts (the real
    _raise_if_stop_requested after the batch), Verified (next in Lite's lane) never starts, and
    the suite ends with SystemExit(143). Before the handover the tab threads never saw it."""
    import os
    import signal

    from gbench import cli
    from gbench.runners import evals as E
    from gbench.utils import process_groups
    monkeypatch.setattr(LV, "TAB_STOP_POLL_S", 0.05)
    stopped, started, scored, tab_idents = {}, [], [], {}
    both_running = threading.Barrier(3, timeout=20)
    lock = threading.Lock()

    def fake_stop(self, reason):          # the real stop removes docker containers
        self.stopped = reason
        with lock:
            stopped[self.container] = reason

    monkeypatch.setattr(LV.SweagentWatch, "stop", fake_stop)

    def run_setting(setting, mode, model, url, conc, kwargs):
        with lock:
            started.append(setting)
            tab_idents[setting] = threading.get_ident()
        watch = LV.SweagentWatch(str(tmp_path), {}, f"batch-{setting}", poll_s=0.05).start()
        both_running.wait()
        deadline = time.monotonic() + 20
        while watch.stopped is None and time.monotonic() < deadline:
            time.sleep(0.02)
        watch.halt()
        LV._raise_if_stop_requested("after the SWE-agent batch")
        with lock:
            scored.append(setting)
        return _tab_result(setting, kwargs)

    monkeypatch.setattr(LV, "_run_setting", run_setting)
    sched = LV.tab_schedule(["lite", "verified", "ccpp"], {"lite": 1, "verified": 1, "ccpp": 1},
                            2)
    assert sched["lanes"] == [["lite", "verified"], ["ccpp"]] and sched["concurrent_lanes"] == 2
    outcome, done = {}, threading.Event()

    def suite():
        outcome["thread"] = threading.get_ident()
        try:
            outcome["result"] = LV.run_tab_schedule(sched, SA.AGENTIC, "m", "http://h/v1", {},
                                                    None, {})
        except BaseException as e:
            outcome["exc"] = e
        finally:
            done.set()
        return {}

    def kill():
        both_running.wait()
        os.kill(os.getpid(), signal.SIGTERM)

    process_groups.clear_stop()
    killer = threading.Thread(target=kill)
    with pytest.raises(SystemExit) as exc:
        with cli._termination_handlers():
            killer.start()
            E._run_with_timeout("swe_bench_live", suite, {}, 3600, "m")
    killer.join(10)
    assert exc.value.code == 128 + signal.SIGTERM
    assert process_groups.stop_requested() is None          # the run-wide record ended
    assert done.wait(30)
    assert sorted(stopped) == ["batch-ccpp", "batch-lite"], stopped
    assert all("signal 15" in why for why in stopped.values())
    assert scored == [] and sorted(started) == ["ccpp", "lite"]          # Verified never starts
    assert isinstance(outcome.get("exc"), SystemExit) and outcome["exc"].code == 143
    # the handover is forgotten once the tabs end (pool thread idents are reused)
    assert all(process_groups.stop_requested(t) is None for t in tab_idents.values())


def _scoring_suite(monkeypatch, scoring_tabs, release):
    """Fake tabs: each tab in `scoring_tabs` sits in harness scoring (no stop check) until
    `release` is set, then returns a full result; every other tab returns at once."""
    started, finished, at_scoring = [], [], threading.Barrier(len(scoring_tabs) + 1, timeout=20)
    lock = threading.Lock()

    def run_setting(setting, mode, model, url, conc, kwargs):
        with lock:
            started.append(setting)
        if setting in scoring_tabs:
            at_scoring.wait()
            assert release.wait(20)
        with lock:
            finished.append(setting)
        return _tab_result(setting, kwargs)
    monkeypatch.setattr(LV, "_run_setting", run_setting)
    return started, finished, at_scoring


def test_a_signal_under_suite_timeout_while_the_tabs_score_still_stops_the_suite(
        monkeypatch):
    """2026-10-10 round-3: a stop handed over while every running tab is in harness scoring
    (which never looks for a stop, so no tab raises) made run_tab_schedule return normally with
    Verified silently missing. A real SIGTERM under --suite-timeout while Lite and C/C++ score:
    the scoring tabs are waited for, Verified never starts, and the suite thread ends with
    TabScheduleStopped (SystemExit 143) whose record names the signal and every halted tab."""
    import os
    import signal

    from gbench import cli
    from gbench.runners import evals as E
    from gbench.utils import process_groups
    monkeypatch.setattr(LV, "TAB_STOP_POLL_S", 0.05)
    release = threading.Event()
    started, finished, at_scoring = _scoring_suite(monkeypatch, {"lite", "ccpp"}, release)
    sched = LV.tab_schedule(["lite", "verified", "ccpp"], {"lite": 1, "verified": 1, "ccpp": 1},
                            2)
    assert sched["lanes"] == [["lite", "verified"], ["ccpp"]]
    outcome, done = {}, threading.Event()

    def suite():
        outcome["thread"] = threading.get_ident()
        try:
            outcome["result"] = LV.run_tab_schedule(sched, SA.AGENTIC, "m", "http://h/v1", {},
                                                    None, {})
        except BaseException as e:
            outcome["exc"] = e
        finally:
            done.set()
        return {}

    def kill():
        at_scoring.wait()
        os.kill(os.getpid(), signal.SIGTERM)
        # scoring goes on past the handover (TAB_STOP_POLL_S) and then ends normally
        time.sleep(0.5)
        release.set()

    process_groups.clear_stop()
    killer = threading.Thread(target=kill)
    with pytest.raises(SystemExit) as exc:
        with cli._termination_handlers():
            killer.start()
            E._run_with_timeout("swe_bench_live", suite, {}, 3600, "m")
    killer.join(10)
    assert exc.value.code == 128 + signal.SIGTERM
    assert done.wait(30)
    assert "result" not in outcome, "the suite returned normally after the stop"
    stop = outcome["exc"]
    assert isinstance(stop, LV.TabScheduleStopped) and isinstance(stop, SystemExit)
    assert stop.code == 143 and stop.signum == 15
    assert sorted(started) == ["ccpp", "lite"] and sorted(finished) == ["ccpp", "lite"]
    rec = stop.record
    assert rec["status"] == "error" and rec["failed"] is True and rec["accuracy"] is None
    assert rec["stopped_by_signal"] == 15 and "signal 15 (SIGTERM)" in rec["error"]
    assert rec["tabs_stopped"] == ["lite", "verified", "ccpp"]
    assert rec["tabs_cut_short"] == ["lite", "ccpp"] and rec["tabs_not_started"] == ["verified"]
    assert rec["tabs_completed"] == [] and stop.results == {}
    assert all(LV.TAB_LABELS[t] in rec["error"] for t in ("lite", "verified", "ccpp"))
    assert all(process_groups.stop_requested(t) is None for t in (outcome["thread"],))


def test_a_signal_in_the_suite_thread_while_a_tab_scores_names_the_completed_and_halted_tabs(
        monkeypatch):
    """Without --suite-timeout the CLI's SIGTERM handler raises SystemExit in the thread running
    the schedule. Lite has completed; Verified (after it) and C/C++ are scoring when SIGTERM
    arrives: the run stops with TabScheduleStopped naming Verified and C/C++ as cut short, Lite
    as completed (its result kept on the exception, not reported as the suite's)."""
    import os
    import signal

    from gbench import cli
    from gbench.utils import process_groups
    monkeypatch.setattr(LV, "TAB_STOP_POLL_S", 0.05)
    monkeypatch.setattr(LV, "_raise_if_stop_requested", lambda where: None)
    release = threading.Event()
    started, finished, at_scoring = _scoring_suite(monkeypatch, {"verified", "ccpp"}, release)
    sched = LV.tab_schedule(["lite", "verified", "ccpp"], {"lite": 1, "verified": 1, "ccpp": 1},
                            2)

    def kill():
        at_scoring.wait()
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.3)
        release.set()

    process_groups.clear_stop()
    killer = threading.Thread(target=kill)
    with pytest.raises(LV.TabScheduleStopped) as exc:
        with cli._termination_handlers():
            killer.start()
            LV.run_tab_schedule(sched, SA.AGENTIC, "m", "http://h/v1", {}, None, {})
    killer.join(10)
    stop = exc.value
    assert stop.code == 128 + signal.SIGTERM
    assert sorted(finished) == ["ccpp", "lite", "verified"]     # the running tabs were waited for
    rec = stop.record
    assert rec["tabs_completed"] == ["lite"] and list(stop.results) == ["lite"]
    assert rec["tabs_cut_short"] == ["verified", "ccpp"] and rec["tabs_not_started"] == []
    assert rec["tabs_stopped"] == ["verified", "ccpp"]
    assert isinstance(stop.__cause__, SystemExit)


def test_a_signal_as_soon_as_the_first_tab_starts_still_names_every_tab(monkeypatch):
    """The tabs are submitted inside the schedule's signal handling: a SIGTERM that arrives while
    the later lanes are still being submitted (here the first tab sends it the moment it starts)
    still ends in TabScheduleStopped naming every tab. With the lanes submitted before that
    handling, 20 of 20 runs raised a bare SystemExit naming no tab."""
    import os
    import signal

    from gbench import cli
    from gbench.utils import process_groups
    monkeypatch.setattr(LV, "TAB_STOP_POLL_S", 0.05)
    monkeypatch.setattr(LV, "_raise_if_stop_requested", lambda where: None)
    tabs = ["lite", "ccpp", "go", "rust", "java"]

    def run_setting(setting, mode, model, url, conc, kwargs):
        if setting == "lite":
            os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.1)
        return _tab_result(setting, kwargs)
    monkeypatch.setattr(LV, "_run_setting", run_setting)
    sched = LV.tab_schedule(tabs, {t: 1 for t in tabs}, 5)
    for _ in range(20):
        process_groups.clear_stop()
        with pytest.raises(LV.TabScheduleStopped) as exc:
            with cli._termination_handlers():
                LV.run_tab_schedule(sched, SA.AGENTIC, "m", "http://h/v1", {}, None, {})
        rec = exc.value.record
        assert exc.value.code == 143 and "lite" in rec["tabs_cut_short"]
        assert sorted(rec["tabs_stopped"] + rec["tabs_completed"]) == sorted(tabs)


def test_tab_stop_record_lists_every_tab_once_in_schedule_order():
    rec = LV.tab_stop_record(1, ["lite", "full", "verified", "go"], ["lite", "go"], ["go"])
    assert rec["tabs_completed"] == ["lite"] and rec["tabs_cut_short"] == ["go"]
    assert rec["tabs_not_started"] == ["full", "verified"]
    assert rec["tabs_stopped"] == ["full", "verified", "go"]
    assert "signal 1 (SIGHUP)" in rec["error"] and rec["leaderboard_comparable"] is False


def test_the_thread_cap_shares_the_box_over_every_lane_as_the_serial_run_did(monkeypatch):
    """The per-container thread cap (swe_thread_cap.resolve_threads: min(4, cpus // containers))
    is computed from the whole suite's budget in every tab, even though each tab is handed only
    its lane's share, and it reaches the harness's scorer through asyncio.run/asyncio.to_thread
    as the scorers call it. 208 CPUs, --sandboxes 100 over 4 lanes: 2 threads per container (the
    serial run's 208 // 100), not 4 (208 // 25), which would put 400 capped threads on 208
    CPUs."""
    import asyncio

    from gbench.runners.eval_suites import swe_thread_cap
    monkeypatch.delenv(swe_thread_cap.ENV_VAR, raising=False)
    monkeypatch.setattr(swe_thread_cap.os, "cpu_count", lambda: 208)
    monkeypatch.setattr(LV, "_raise_if_stop_requested", lambda where: None)
    tabs = ["lite", "ccpp", "go", "rust"]
    caps, lock = {}, threading.Lock()

    def run_setting(setting, mode, model, url, conc, kwargs):
        async def scorer():
            return await asyncio.to_thread(swe_thread_cap.resolve_threads, conc)
        with lock:
            caps[setting] = (conc, swe_thread_cap.resolve_threads(conc), asyncio.run(scorer()))
        return _tab_result(setting, kwargs)

    monkeypatch.setattr(LV, "_run_setting", run_setting)
    sched = LV.tab_schedule(tabs, {t: 500 for t in tabs}, 100)
    assert sched["concurrent_lanes"] == 4 and set(sched["workers"].values()) == {25}
    LV.run_tab_schedule(sched, SA.AGENTIC, "m", "http://h/v1", {}, None, {})
    serial = swe_thread_cap.resolve_threads(100)
    assert serial == 2 and swe_thread_cap.resolve_threads(25) == 4    # outside a tab: unchanged
    assert caps == {t: (25, serial, serial) for t in tabs}
    assert sum(conc * cap for conc, cap, _ in caps.values()) <= 208
