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

"""Per-turn output budgets for tool-loop suites, and the terminal_bench turn cap.

The bug these guard against is subtle: `max_output_tokens` reads like a bound on the run,
but in a tool loop it is sent on every turn. browsecomp at 65536 produced 3/3 samples that
spent the whole budget on one turn and never answered; terminal_bench left `max_tokens`
unset entirely, so a turn could run to `max_model_len` and eat the agent's whole clock.
"""

import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gbench.runners.eval_suites.base import (  # noqa: E402
    TOOL_LOOP_MAX_OUTPUT_TOKENS, tool_loop_turn_ceiling)
from gbench.runners.eval_suites import terminal_bench as tb  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("GBENCH_TOOL_LOOP_MAX_OUTPUT_TOKENS", "GBENCH_TB_TURN_MAX_TOKENS",
              "GBENCH_TB_TIMEOUT_MULTIPLIER", "GBENCH_TERMINAL_BENCH_TEMPERATURE"):
        monkeypatch.delenv(k, raising=False)


# --- the tool-loop ceiling -------------------------------------------------------------
def test_ceiling_caps_a_tool_loop_suite():
    assert tool_loop_turn_ceiling(True, 65536) == TOOL_LOOP_MAX_OUTPUT_TOKENS


def test_ceiling_leaves_single_shot_suites_alone():
    """aime spends 31,252 tokens on one turn and that turn IS the answer."""
    assert tool_loop_turn_ceiling(False, 65536) is None


def test_ceiling_never_raises_a_smaller_budget():
    assert tool_loop_turn_ceiling(True, 4096) is None


def test_ceiling_is_a_no_op_for_the_other_tool_loop_suites():
    """Largest turn observed on the 2026-08-18 run: mcp_atlas 3,014 tokens."""
    for observed_max in (1063, 2866, 3014, 1335):
        assert observed_max < TOOL_LOOP_MAX_OUTPUT_TOKENS


def test_ceiling_can_be_disabled(monkeypatch):
    monkeypatch.setenv("GBENCH_TOOL_LOOP_MAX_OUTPUT_TOKENS", "0")
    assert tool_loop_turn_ceiling(True, 65536) is None


def test_ceiling_honours_an_override(monkeypatch):
    monkeypatch.setenv("GBENCH_TOOL_LOOP_MAX_OUTPUT_TOKENS", "2048")
    assert tool_loop_turn_ceiling(True, 65536) == 2048


def test_ceiling_ignores_a_malformed_override(monkeypatch):
    monkeypatch.setenv("GBENCH_TOOL_LOOP_MAX_OUTPUT_TOKENS", "not-a-number")
    assert tool_loop_turn_ceiling(True, 65536) == TOOL_LOOP_MAX_OUTPUT_TOKENS


def test_ceiling_handles_an_unset_budget():
    assert tool_loop_turn_ceiling(True, None) is None


# --- terminal_bench per-turn cap -------------------------------------------------------
def test_terminal_bench_sends_no_turn_cap_by_default():
    """Terminus-2 and LiteLLM send no max_tokens (the canonical setting), in both thinking
    modes; GBENCH_TB_TURN_MAX_TOKENS is an opt-in backstop."""
    assert tb._turn_token_cap(True) == 0
    assert tb._turn_token_cap(False) == 0


def _report_at(rate_tok_s, cap=None):
    """One trial that generated `rate_tok_s` tokens per second of model time."""
    cap = cap or tb._TURN_MAX_TOKENS_THINKING
    turns = 20
    seconds = 100.0
    return tb._turn_budget_report(
        [{"output_tokens": int(rate_tok_s * seconds * turns),
          "mean_turn_latency_s": seconds, "agent_turns": turns}], cap, True)


def test_the_turn_budget_is_reported_from_measurement_not_a_constant():
    """The first version of this test asserted `74 tok/s` - this box's throughput - as
    though it were a property of the benchmark. It would have passed green on a machine
    10x slower while that machine timed out on every trial. The second version hardcoded
    numbers derived from an 8192 cap and broke the moment the cap moved, which is the same
    mistake wearing a different hat. Derive everything from the constants."""
    rate, cap = 20.0, tb._TURN_MAX_TOKENS_THINKING
    report = _report_at(rate)
    assert report["observed_tok_s"] == rate
    assert report["seconds_per_turn"] == round(cap / rate, 1)
    assert report["shortest_agent_budget_s"] == int(
        tb._SHORTEST_TASK_AGENT_TIMEOUT_S * float(tb._timeout_multiplier(True)))
    assert report["turns_in_shortest_budget"] < 10, "a slow endpoint must read as unhealthy"


def test_a_fast_endpoint_reports_a_healthy_turn_budget():
    """The rate that yields 25 turns follows from the cap, so compute it rather than
    guessing a number that only holds for one cap value."""
    budget = tb._SHORTEST_TASK_AGENT_TIMEOUT_S * float(tb._timeout_multiplier(True))
    needed = tb._TURN_MAX_TOKENS_THINKING / (budget / 25.0)
    assert _report_at(needed * 1.2)["turns_in_shortest_budget"] >= 25


def test_the_turn_budget_report_is_absent_rather_than_invented():
    """No trials, or trials Harbor gave no token counts for, must not produce a number."""
    assert tb._turn_budget_report([], 8192, True) is None
    assert tb._turn_budget_report([{"output_tokens": 0}], 8192, True) is None
    assert tb._turn_budget_report(
        [{"output_tokens": 100, "mean_turn_latency_s": None, "agent_turns": None}],
        8192, True) is None


def test_terminal_bench_turn_cap_is_overridable(monkeypatch):
    monkeypatch.setenv("GBENCH_TB_TURN_MAX_TOKENS", "1234")
    assert tb._turn_token_cap(True) == 1234
    monkeypatch.setenv("GBENCH_TB_TURN_MAX_TOKENS", "0")
    assert tb._turn_token_cap(True) == 0


def test_terminal_bench_timeout_multiplier_is_overridable(monkeypatch):
    # The canonical 1.0 (leaderboard/SUBMIT.md) in both modes unless the knob sets one.
    assert tb._timeout_multiplier(True) == "1.0"
    assert tb._timeout_multiplier(False) == "1.0"
    monkeypatch.setenv("GBENCH_TB_TIMEOUT_MULTIPLIER", "1.5")
    assert tb._timeout_multiplier(True) == "1.5"


def test_terminal_bench_llm_call_kwargs_are_valid_json(monkeypatch):
    """Harbor parses `--agent-kwarg k=v` values; a hand-built brace string got this wrong."""
    monkeypatch.setenv("GBENCH_TB_TURN_MAX_TOKENS", str(tb._TURN_MAX_TOKENS_THINKING))
    payload = {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
        "max_tokens": tb._turn_token_cap(True),
    }
    parsed = json.loads(json.dumps(payload))
    assert parsed["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert parsed["max_tokens"] == tb._TURN_MAX_TOKENS_THINKING


# --- terminal_bench transcript recovery -------------------------------------------------
def test_trajectory_text_reads_terminus_atif_files(tmp_path):
    """Harbor writes the conversation to agent/trajectory.json, NOT into result.json."""
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "trajectory.json").write_text(json.dumps({"steps": [
        {"reasoning_content": "thinking about ls", "message": "I will list files",
         "tool_calls": [{"arguments": {"keystrokes": "ls -la\n"}}]},
        {"message": "done"},
    ]}))
    text, steps = tb._trajectory_text(str(tmp_path))
    assert steps == 2
    assert "thinking about ls" in text and "ls -la" in text


def test_trajectory_text_merges_summarization_continuations(tmp_path):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "trajectory.json").write_text(json.dumps({"steps": [{"message": "first"}]}))
    (agent_dir / "trajectory.cont-1.json").write_text(
        json.dumps({"steps": [{"message": "second"}]}))
    text, steps = tb._trajectory_text(str(tmp_path))
    assert steps == 2 and "first" in text and "second" in text


def test_trajectory_text_tolerates_a_missing_or_broken_file(tmp_path):
    assert tb._trajectory_text(str(tmp_path)) == ("", None)
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "trajectory.json").write_text("{not json")
    assert tb._trajectory_text(str(tmp_path)) == ("", None)


def test_agent_transcript_fallback_reads_store_all_messages():
    agent = {"metadata": {"all_messages": [{"content": "hello"}, {"content": "world"}]}}
    assert tb._agent_transcript(agent, {}) == "hello\nworld"


def test_agent_transcript_does_not_invent_a_transcript():
    """AgentContext carries token counts only; probing it for messages must return ''."""
    agent = {"n_input_tokens": 10, "n_output_tokens": 20, "metadata": {"n_episodes": 3}}
    assert tb._agent_transcript(agent, {}) == ""


# --- the empty-reasoned turn ------------------------------------------------------------
def test_a_turn_that_reasons_itself_out_of_budget_is_asked_again():
    """A truncated turn with no content and no tool call used to end the sample with
    nothing: the judge saw an empty response and recorded JUDGE_FAILED, which reads like
    an outage. Measured 2026-08-18 on browsecomp - 2/3 samples reasoned 28,538 and 25,618
    characters into an 8192-token turn and emitted no answer at all."""
    import inspect
    from gbench.runners.eval_suites import base
    src = inspect.getsource(base._run_suite_async)
    assert 'reply.finish_reason == "length"' in src
    assert "You ran out of room to think" in src


def test_the_recovery_turn_disables_thinking():
    """Re-asking with thinking still on just spends another whole turn reasoning; the
    point is to force the content channel open."""
    import inspect, re
    from gbench.runners.eval_suites import base
    src = inspect.getsource(base._run_suite_async)
    block = src.split("You ran out of room to think")[1].split("break")[0]
    assert "thinking=False" in block


def test_the_recovery_turn_withdraws_the_tools():
    import inspect
    from gbench.runners.eval_suites import base
    src = inspect.getsource(base._run_suite_async)
    block = src.split("You ran out of room to think")[1].split("break")[0]
    assert '"tools", "tool_choice"' in block


def test_truncation_advice_differs_for_tool_loop_suites():
    """'Raise --max-output-tokens' is the wrong advice once the budget bounds one turn."""
    import inspect
    from gbench.runners.eval_suites import base
    src = inspect.getsource(base._run_suite_async)
    assert "In a tool loop" in src
    assert "GBENCH_TOOL_LOOP_MAX_OUTPUT_TOKENS if the turn is genuinely too" in src


# --- the assumptions must be self-checking on unfamiliar hardware -----------------------
def test_the_decode_rate_is_measured_and_published():
    """Every timeout here is sized from an ASSUMED rate. Publishing the observed one is
    what makes a timeout on unfamiliar hardware diagnosable instead of a bare failure."""
    import inspect
    from gbench.runners.eval_suites import base
    src = inspect.getsource(base._run_suite_async)
    assert '"assumed_floor": MIN_DECODE_TOK_S' in src
    assert '"observed_median"' in src


def test_a_slow_endpoint_is_warned_about_not_left_to_time_out():
    import inspect
    from gbench.runners.eval_suites import base
    src = inspect.getsource(base._run_suite_async)
    assert "BELOW the %.1f tok/s the request" in src


def test_the_turn_cap_is_denominated_in_tokens_not_seconds():
    """Token budgets port across hardware; second budgets do not. Anything time-shaped
    has to be derived at runtime, not frozen into a constant."""
    from gbench.runners.eval_suites.base import TOOL_LOOP_MAX_OUTPUT_TOKENS
    assert isinstance(TOOL_LOOP_MAX_OUTPUT_TOKENS, int)
    assert isinstance(tb._TURN_MAX_TOKENS_THINKING, int)
    assert isinstance(tb._TURN_MAX_TOKENS_PLAIN, int)


def test_every_hardware_shaped_knob_has_an_env_override():
    """The escape hatch matters more than the default: the defaults were tuned on one
    8xA100 box and will be wrong somewhere."""
    import inspect
    from gbench.runners.eval_suites import base
    for src, var in ((inspect.getsource(base), "GBENCH_MIN_DECODE_TOK_S"),
                     (inspect.getsource(base), "GBENCH_REQUEST_TIMEOUT_S"),
                     (inspect.getsource(base), "GBENCH_TOOL_LOOP_MAX_OUTPUT_TOKENS"),
                     (inspect.getsource(tb), "GBENCH_TB_TURN_MAX_TOKENS"),
                     (inspect.getsource(tb), "GBENCH_TB_TIMEOUT_MULTIPLIER")):
        assert var in src, f"{var} must remain overridable"


def test_decode_rate_is_computed_from_a_real_response(monkeypatch):
    """Exercises the hot path with a stubbed endpoint: a change here breaks every suite,
    and it cannot be checked against the live server without perturbing the very latency
    the report exists to measure."""
    import asyncio, time
    from gbench.runners.eval_suites import base

    class _Resp:
        status = 200
        async def json(self):
            time.sleep(0.05)          # 100 tokens over >=0.05s => <=2000 tok/s
            return {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 100}}
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    class _Session:
        def post(self, *a, **k): return _Resp()

    reply = asyncio.run(base._send_single_request(
        session=_Session(), api_url="http://x/v1/chat/completions",
        model_name="m", messages=[{"role": "user", "content": "q"}],
        extra_payload=None, semaphore=asyncio.Semaphore(1), pbar=None,
        max_output_tokens=128, temperature=1.0, thinking=False))
    assert reply.completion_tokens == 100
    assert reply.decode_tok_s is not None and 0 < reply.decode_tok_s <= 2000


def test_decode_rate_is_none_when_the_server_reports_no_usage():
    """No token count means no rate - inventing one would make the assumption check lie."""
    import asyncio
    from gbench.runners.eval_suites import base

    class _Resp:
        status = 200
        async def json(self):
            return {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]}
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    class _Session:
        def post(self, *a, **k): return _Resp()

    reply = asyncio.run(base._send_single_request(
        session=_Session(), api_url="http://x/v1/chat/completions",
        model_name="m", messages=[{"role": "user", "content": "q"}],
        extra_payload=None, semaphore=asyncio.Semaphore(1), pbar=None,
        max_output_tokens=128, temperature=1.0, thinking=False))
    assert reply.decode_tok_s is None


# --- the livelock that a too-tight cap causes -------------------------------------------
def _log(tmp_path, lines):
    (tmp_path / "trial.log").write_text("\n".join(lines) + "\n")
    return str(tmp_path)


def test_a_truncation_livelock_is_detected(tmp_path):
    """9 truncations with no step between them: what write-compressor did on 2026-08-18
    while reporting nothing but `AgentTimeoutError`."""
    d = _log(tmp_path, ["Trajectory dumped to x"] + ["Output length exceeded: ..."] * 9)
    r = tb._truncation_livelock(d)
    assert r["truncated_turns"] == 9
    assert r["unresolved_truncations"] == 9
    assert r["livelocked"] is True


def test_recovered_truncations_are_not_a_livelock(tmp_path):
    """torch-tensor-parallelism hit the cap 3 times and recovered each time across 21
    steps. That is a healthy trial and must not be flagged."""
    d = _log(tmp_path, sum(
        [["Output length exceeded: ...", "Trajectory dumped to x"] for _ in range(3)], []))
    r = tb._truncation_livelock(d)
    assert r["truncated_turns"] == 3
    assert r["longest_truncation_run"] == 1
    assert r["livelocked"] is False


def test_a_clean_trial_reports_no_truncation_fields(tmp_path):
    d = _log(tmp_path, ["Trajectory dumped to x"] * 5)
    assert tb._truncation_livelock(d) == {}


def test_livelock_detection_tolerates_a_missing_log(tmp_path):
    assert tb._truncation_livelock(str(tmp_path)) == {}


# --- the parser has to be able to salvage a truncated turn ------------------------------
def test_the_agent_uses_a_parser_that_can_salvage_truncation(monkeypatch):
    """`json` (terminus-2's default, the canonical parser) has no `salvage_truncated_response`,
    so a capped turn is unrecoverable and gets re-asked forever. No cap is sent by default, so
    the default is json; GBENCH_TB_PARSER=xml is the opt-in parser for a run that sets a cap."""
    monkeypatch.delenv("GBENCH_TB_PARSER", raising=False)
    assert tb.agent_parser() == "json" == tb.CANONICAL_PARSER
    monkeypatch.setenv("GBENCH_TB_PARSER", "xml")
    assert tb.agent_parser() == "xml"


def test_the_turn_cap_clears_the_observed_maximum_with_headroom():
    """8192 sat 7% above the largest turn ever observed (7,644) and livelocked two trials.
    A backstop has to be well clear of the working distribution, not adjacent to it."""
    observed_max = 7644
    assert tb._TURN_MAX_TOKENS_THINKING >= 4 * observed_max


# --- terminal_bench must honour the documented temperature default ----------------------
def test_terminal_bench_resolves_its_temperature_like_every_other_suite():
    """It shells out to Harbor instead of using run_eval_suite, so it never reached
    resolve_temperature: Harbor received the literal `temperature=None` and terminus-2
    skipped it, while the module header promised a 1.0 default."""
    import inspect
    src = inspect.getsource(tb.run_terminal_bench)
    # the TB2.1 leaderboard's temperature 1, through the official-sampling resolver
    assert "resolve_official_sampling(" in src
    assert '"terminal_bench"' in src


# --- container thread cap ---------------------------------------------------------------
def test_container_threads_default_to_one(monkeypatch):
    """No cap by default: the TB2.1 leaderboard requires no resource overrides, and the cap
    reaches the verifier's environment too. GBENCH_TB_CONTAINER_THREADS=1 is the opt-in value
    for the case it was built for: `cpus = 1` is the modal AND minimum task declaration, and
    nothing rewrites /proc/cpuinfo, so an uncapped runtime spawns one thread per HOST core."""
    monkeypatch.delenv("GBENCH_TB_CONTAINER_THREADS", raising=False)
    assert tb.container_thread_env() == {}
    monkeypatch.setenv("GBENCH_TB_CONTAINER_THREADS", "1")
    assert tb.container_thread_env()["OMP_NUM_THREADS"] == "1"


def test_container_threads_cover_the_same_runtimes_as_the_swe_cap(monkeypatch):
    from gbench.runners.eval_suites.swe_thread_cap import THREAD_VARS
    monkeypatch.setenv("GBENCH_TB_CONTAINER_THREADS", "1")
    assert set(tb.container_thread_env()) == set(THREAD_VARS)


def test_container_threads_can_be_disabled(monkeypatch):
    for off in ("0", "off", "none", "false"):
        monkeypatch.setenv("GBENCH_TB_CONTAINER_THREADS", off)
        assert tb.container_thread_env() == {}


def test_container_threads_honour_an_override(monkeypatch):
    monkeypatch.setenv("GBENCH_TB_CONTAINER_THREADS", "4")
    assert tb.container_thread_env()["MKL_NUM_THREADS"] == "4"


def test_container_threads_ignore_a_malformed_override(monkeypatch):
    # a malformed value falls back to the default: nothing injected
    monkeypatch.setenv("GBENCH_TB_CONTAINER_THREADS", "lots")
    assert tb.container_thread_env() == {}


def test_thread_env_reaches_BOTH_the_agent_and_the_verifier():
    """They are different Harbor code paths: the agent gets `extra_env` (which becomes
    `tmux new-session -e K=V`), the verifier gets `--verifier-env` (applied as
    `override_env` on its exec). Setting only the first leaves the SCORING run thrashing,
    and a verifier that times out marks a correct solution failed."""
    import inspect
    src = inspect.getsource(tb.run_terminal_bench)
    assert "extra_env=" in src
    assert "--verifier-env" in src


def test_the_container_cpu_limit_itself_is_never_overridden():
    """`cpus` in task.toml is the benchmark's published environment spec. Raising it would
    make the numbers incomparable with every other Terminal-Bench result."""
    import inspect
    src = inspect.getsource(tb)
    for flag in ("--cpus", "nano_cpus", "NanoCpus", "--memory"):
        assert flag not in src, f"{flag} must not be set by gbench"


def test_the_budget_report_prices_truncated_turns(monkeypatch):
    """Replays the real 2026-08-18 torch-tensor-parallelism trial: 22 completed turns,
    36,349 useful tokens, 513s of recorded model time, and 4 truncated turns that
    `api_request_times_msec` never saw. Uncorrected the report called it 22% model-bound;
    the truncations had actually burned 78% of every token generated."""
    monkeypatch.setenv("GBENCH_TB_TURN_MAX_TOKENS", "32768")
    cap = tb._turn_token_cap(True)
    r = tb._turn_budget_report(
        [{"output_tokens": 36349, "mean_turn_latency_s": 513 / 22, "agent_turns": 22,
          "truncated_turns": 4}], cap, True)
    assert r["tokens_useful"] == 36349
    assert r["tokens_discarded"] == 4 * cap
    assert r["discarded_token_pct"] > 75, "the headline waste must be visible"
    assert r["seconds_lost_to_truncation"] > 1500
    assert r["observed_tok_s"] == round(36349 / 513, 1), "rate must use COMPLETED turns only"


def test_a_clean_trial_reports_zero_waste():
    r = tb._turn_budget_report(
        [{"output_tokens": 20000, "mean_turn_latency_s": 25.0, "agent_turns": 20,
          "truncated_turns": 0}], 32768, True)
    assert r["tokens_discarded"] == 0 and r["discarded_token_pct"] == 0.0
    assert r["seconds_lost_to_truncation"] == 0


# --- the harness must not throw away its own results ------------------------------------
def test_job_stats_fallback_reads_the_keys_harbor_actually_emits():
    """Harbor 0.20.0 emits `n_completed_trials` and `evals[<name>].metrics[0].mean`. The
    fallback read `n_trials` / `n_passed` / `n_success`, none of which exist, so on
    2026-08-18 a COMPLETED 2h40 job published `total_questions: 0`."""
    stats = {"n_completed_trials": 3, "n_errored_trials": 0, "n_cancelled_trials": 0,
             "evals": {"terminus-2__m__d": {"n_trials": 3, "n_errors": 0,
                                            "metrics": [{"mean": 0.0}]}}}
    assert tb._job_stats_totals(stats) == (3, 0)


def test_job_stats_fallback_converts_a_mean_reward_to_a_count():
    stats = {"n_completed_trials": 4,
             "evals": {"e": {"n_trials": 4, "metrics": [{"mean": 0.5}]}}}
    assert tb._job_stats_totals(stats) == (4, 2)


def test_job_stats_fallback_falls_back_to_the_nested_trial_count():
    stats = {"evals": {"e": {"n_trials": 7, "metrics": [{"mean": 0.0}]}}}
    assert tb._job_stats_totals(stats)[0] == 7


def test_job_stats_fallback_is_empty_not_wrong_when_harbor_says_nothing():
    assert tb._job_stats_totals({}) == (0, 0)


def test_parse_failures_are_warnings_not_debug():
    """A swallowed exception on the only path that turns work into a score is not a
    debug detail - it is how 2h40 of compute became `nothing was measured`."""
    import inspect
    src = inspect.getsource(tb.run_terminal_bench)
    assert "parse_failures.append" in src
    assert 'logger.warning("Could not parse' in src


def test_the_jobs_dir_is_kept_when_a_run_yields_nothing(monkeypatch, tmp_path):
    """An unconditional TemporaryDirectory deleted Harbor's result.json, trial.log and
    agent/trajectory.json on exit - twice in one day that destroyed the only evidence of
    why a run failed."""
    monkeypatch.delenv("GBENCH_TB_KEEP_JOBS_DIR", raising=False)
    with tb._jobs_dir() as d:
        open(os.path.join(d, "trial.log"), "w").write("evidence")
        tb._JOBS_DIR_OK[d] = False          # parser found nothing
    assert os.path.isdir(d), "a failed run must keep its evidence"
    import shutil as _sh; _sh.rmtree(d, ignore_errors=True)


def test_the_jobs_dir_is_cleaned_up_when_the_run_was_healthy(monkeypatch):
    monkeypatch.delenv("GBENCH_TB_KEEP_JOBS_DIR", raising=False)
    with tb._jobs_dir() as d:
        open(os.path.join(d, "trial.log"), "w").write("fine")
        tb._JOBS_DIR_OK[d] = True
    assert not os.path.exists(d), "a healthy run must not litter /tmp"


def test_the_jobs_dir_survives_an_exception(monkeypatch):
    monkeypatch.delenv("GBENCH_TB_KEEP_JOBS_DIR", raising=False)
    kept = {}
    try:
        with tb._jobs_dir() as d:
            kept["d"] = d
            tb._JOBS_DIR_OK[d] = True       # even a "healthy" flag must not win here
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert os.path.isdir(kept["d"])
    import shutil as _sh; _sh.rmtree(kept["d"], ignore_errors=True)


# --- the agent-kwarg / agent-env collision ----------------------------------------------
def _tb_cmd():
    """The harbor argv gbench builds, captured without running anything.

    Stubs the Docker precheck too: an earlier version of this helper patched
    `subprocess.Popen` globally, which broke the `subprocess.run(["docker","info"])`
    precheck, so the suite skipped, the argv was never built, and the assertions below
    passed against an EMPTY list - green while the bug was live.

    The task list `limit=3` draws from is a fixed list too: the real one comes from Harbor's
    task registry over the network, and a unit test neither needs Harbor installed (Python
    3.11 cannot install Harbor 0.20.0) nor queries a live service.
    """
    from unittest import mock
    seen = {}
    task_names = [f"terminal-bench/task-{i}" for i in range(8)]

    def _popen(cmd, *a, **k):
        seen["cmd"] = cmd
        raise RuntimeError("argv captured")

    with mock.patch.object(tb, "check_terminal_bench_prerequisites",
                           return_value=(True, "")), \
         mock.patch.object(tb, "_dataset_task_names", return_value=task_names), \
         mock.patch.object(tb, "require_task_metadata_blocked", return_value=None), \
         mock.patch.object(tb, "prepare_task_images", return_value=({}, {})), \
         mock.patch.object(tb.subprocess, "Popen", _popen):
        try:
            tb.run_terminal_bench("m", "http://x:8000/v1", 4,
                                  enable_thinking=True, limit=3)
        except Exception:
            pass
    assert seen.get("cmd"), "argv was never built - the capture harness is broken"
    return seen["cmd"]


def test_thinking_run_resolves_temperature_to_one():
    """Regression (value-level, not a source grep): enable_thinking is a NAMED param and is never
    present in **kwargs, so the old `bool(kwargs.get('enable_thinking', False))` was always False and
    a --thinking run silently decoded at 0.0 greedy. _tb_cmd() builds the argv with
    enable_thinking=True, so the temperature handed to Harbor must be 1.0 (DEFAULT_TEMPERATURE)."""
    cmd = _tb_cmd()
    kwargs = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--agent-kwarg"]
    temps = [k for k in kwargs if k.startswith("temperature=")]
    assert temps == ["temperature=1.0"], f"expected temperature=1.0 for a --thinking run, got {temps}"


def test_extra_env_is_never_passed_as_an_agent_kwarg():
    """AgentFactory does `create_agent_from_name(..., extra_env=extra_env, **agent_kwargs)`,
    so an `extra_env` agent-kwarg is a guaranteed TypeError: got multiple values for
    keyword argument. That killed a whole job 8s in on 2026-08-19."""
    cmd = _tb_cmd()
    kwargs = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--agent-kwarg"]
    assert not any(k.startswith("extra_env=") for k in kwargs), \
        "extra_env is a reserved AgentFactory parameter; use --agent-env"


def test_thread_env_goes_to_both_agent_env_and_verifier_env(monkeypatch):
    # opt-in (none injected by default); when set it must reach both channels
    monkeypatch.setenv("GBENCH_TB_CONTAINER_THREADS", "1")
    cmd = _tb_cmd()
    agent_env = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--agent-env"]
    ver_env = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--verifier-env"]
    want = {f"{k}={v}" for k, v in tb.container_thread_env().items()}
    assert want and set(agent_env) == want, f"agent env {agent_env}"
    assert set(ver_env) == want, f"verifier env {ver_env}"


def test_no_agent_kwarg_collides_with_a_reserved_factory_parameter():
    """Generalises the bug: any agent-kwarg whose name AgentFactory also passes
    positionally/by-name will blow up the same way."""
    reserved = {"extra_env", "logs_dir", "model_name", "logger", "registry_spec"}
    cmd = _tb_cmd()
    kwargs = [cmd[i + 1].split("=", 1)[0]
              for i, a in enumerate(cmd) if a == "--agent-kwarg"]
    clash = reserved & set(kwargs)
    assert not clash, f"agent kwargs collide with reserved factory params: {clash}"


def test_trial_parsing_has_every_name_it_uses():
    """`repetition_run` / `repetition_onset` were used in the trial parser but never
    imported, so EVERY per-trial parse raised NameError. At debug level that was silent and
    a completed 2h40 job published `total_questions: 0`; the WARNING made it obvious on the
    next run. Exercise the real parse body against a realistic Harbor result.json."""
    import json, os, tempfile
    with tempfile.TemporaryDirectory() as d:
        trial = os.path.join(d, "job", "some-task__abc")
        os.makedirs(os.path.join(trial, "agent"))
        json.dump({"steps": [{"message": "ls -la", "reasoning_content": "thinking",
                              "metrics": {"completion_tokens": 100}}]},
                  open(os.path.join(trial, "agent", "trajectory.json"), "w"))
        open(os.path.join(trial, "trial.log"), "w").write("Trajectory dumped to x\n")
        json.dump({"trial_name": "some-task__abc",
                   "verifier_result": {"rewards": {"reward": 0.0}},
                   "agent_result": {"n_output_tokens": 500,
                                    "metadata": {"n_episodes": 7,
                                                 "api_request_times_msec": [1000.0, 2000.0]}}},
                  open(os.path.join(trial, "result.json"), "w"))
        text, steps = tb._trajectory_text(trial)
        # the two names the parser needs but did not have
        assert tb.repetition_run(text) is not None
        assert tb.repetition_onset(text) is None or isinstance(tb.repetition_onset(text), dict)
        assert steps == 1


def test_the_module_imports_the_base_helpers_it_calls():
    import inspect, re
    src = inspect.getsource(tb)
    used = set(re.findall(r"\b(repetition_run|repetition_onset|resolve_temperature)\s*\(", src))
    for name in used:
        assert hasattr(tb, name), f"{name} is called but not importable from the module"


# --- recovering a forced-final answer the tool-call parser ate ----------------------------
import asyncio as _asyncio                                                    # noqa: E402
from unittest import mock as _mock                                            # noqa: E402
from gbench.runners.eval_suites import base as B                              # noqa: E402


def _discarded():
    """What vLLM returns when gemma-4 emits a tool call on a request declaring no `tools`:
    real tokens generated, `content` null, nothing bound. Measured live 2026-08-20."""
    return B.Reply(text="", tool_calls=None, finish_reason="stop", reasoning=None,
                   completion_tokens=34, stop_reason=50)


def _answered(txt="Final Answer: 7 years"):
    return B.Reply(text=txt, tool_calls=None, finish_reason="stop", completion_tokens=12)


def _retry_kwargs(**over):
    kw = dict(convo=[{"role": "user", "content": "q"}], session=None, api_url="u",
              model_name="m", payload={}, semaphore=None, budget=100000, temperature=1.0)
    kw.update(over)
    return kw


def test_discarded_forced_final_is_classified_as_such():
    assert B.classify_reply(_discarded()) == "output_discarded"


def test_discarded_forced_final_is_retried_and_recovered(monkeypatch):
    """35 of 152 forced finals (23%) on the 2026-08-20 mcp_atlas run came back discarded -
    7% of the suite graded as silence on answers the model had already generated."""
    monkeypatch.setattr(B, "RETRY_DISCARDED_FINAL", True)
    monkeypatch.setattr(B, "clamp_to_context", lambda convo, budget, *rest: (4096, False))
    with _mock.patch.object(B, "_send_single_request",
                            new=_mock.AsyncMock(return_value=_answered())) as send:
        out = _asyncio.run(B._retry_discarded_final(reply=_discarded(), **_retry_kwargs()))
    assert send.await_count == 1
    assert out.text == "Final Answer: 7 years"


def test_a_healthy_final_is_never_retried(monkeypatch):
    monkeypatch.setattr(B, "RETRY_DISCARDED_FINAL", True)
    with _mock.patch.object(B, "_send_single_request", new=_mock.AsyncMock()) as send:
        out = _asyncio.run(B._retry_discarded_final(reply=_answered(), **_retry_kwargs()))
    assert send.await_count == 0 and out.text == "Final Answer: 7 years"


def test_retry_that_comes_back_empty_keeps_the_original(monkeypatch):
    """Never return something emptier than what we were handed."""
    monkeypatch.setattr(B, "RETRY_DISCARDED_FINAL", True)
    monkeypatch.setattr(B, "clamp_to_context", lambda convo, budget, *rest: (4096, False))
    orig = _discarded()
    with _mock.patch.object(B, "_send_single_request",
                            new=_mock.AsyncMock(return_value=_discarded())):
        assert _asyncio.run(B._retry_discarded_final(reply=orig, **_retry_kwargs())) is orig


def test_retry_survives_a_transport_error(monkeypatch):
    monkeypatch.setattr(B, "RETRY_DISCARDED_FINAL", True)
    monkeypatch.setattr(B, "clamp_to_context", lambda convo, budget, *rest: (4096, False))
    orig = _discarded()
    with _mock.patch.object(B, "_send_single_request",
                            new=_mock.AsyncMock(side_effect=OSError("reset"))):
        assert _asyncio.run(B._retry_discarded_final(reply=orig, **_retry_kwargs())) is orig


def test_retry_skipped_when_the_convo_no_longer_fits(monkeypatch):
    monkeypatch.setattr(B, "RETRY_DISCARDED_FINAL", True)
    monkeypatch.setattr(B, "clamp_to_context", lambda convo, budget, *rest: (0, True))
    orig = _discarded()
    with _mock.patch.object(B, "_send_single_request", new=_mock.AsyncMock()) as send:
        assert _asyncio.run(B._retry_discarded_final(reply=orig, **_retry_kwargs())) is orig
    assert send.await_count == 0


def test_retry_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(B, "RETRY_DISCARDED_FINAL", False)
    orig = _discarded()
    with _mock.patch.object(B, "_send_single_request", new=_mock.AsyncMock()) as send:
        assert _asyncio.run(B._retry_discarded_final(reply=orig, **_retry_kwargs())) is orig
    assert send.await_count == 0


# --- the tally that would have caught the silent outage -----------------------------------
def test_backend_tally_counts_what_actually_answered(monkeypatch):
    """`search_backend` records what was CONFIGURED (the whole cascade). The tally records
    what actually SERVED - on 2026-08-20 the configured name read grounding on a run where
    every lookup 429'd, which is why the per-model tally exists."""
    from gbench.runners.eval_suites import search_tool as ST
    ST.reset_backend_tally()
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    for knob in ("GBENCH_SEARCH_MODELS", "GBENCH_SEARCH_MODEL", "GBENCH_JUDGE_MODELS",
                 "GBENCH_JUDGE_MODEL"):
        monkeypatch.delenv(knob, raising=False)
    first, second = ST._search_cascade()[:2]
    # first model 429s, second serves -> tally credits the model that actually answered
    def cascade(q, n=8, model=None):
        if model == first:
            return [{"title": "error", "url": "", "snippet": "429"}]
        return [{"title": "t", "url": "", "snippet": "s"}]
    monkeypatch.setattr(ST, "gemini_search", cascade)
    for _ in range(3):
        ST.search_with_fallback("q")
    assert ST.backend_tally() == {f"gemini:{second}": 3}
    ST.reset_backend_tally()
    assert ST.backend_tally() == {}
