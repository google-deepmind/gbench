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

"""toolbench's gbench-native Gemini SoPR/SoWR judge (ported verbatim from StableToolBench ToolEval).

Covers the deterministic pre-checks, verdict parsing, and the check_answer_status ->
parse_answer_status escalation, with the Gemini cascade stubbed (no network).
"""
import asyncio
import warnings

warnings.filterwarnings("ignore")

import pytest
import gbench.runners.eval_suites.toolbench as T


def _finish_answer(final="the answer"):
    return {"final_answer": final, "answer_details": [
        {"role": "assistant", "message": "thinking", "next": [
            {"role": "tool", "message": {"name": "Finish", "arguments": {}, "response": ""},
             "next": []}]}]}


def test_status_pre_checks_are_deterministic_no_judge(monkeypatch):
    # Empty answer, give_up, and no-Finish must be Unsolved WITHOUT any judge call.
    async def _boom(*a, **k):
        raise AssertionError("judge must not be called on a deterministic pre-check")
    monkeypatch.setattr(T, "_tooleval_cascade", _boom)
    sem = asyncio.Semaphore(1)

    async def go():
        assert await T._answer_status("q", {"final_answer": "", "answer_details": []}, sem) == "Unsolved"
        assert await T._answer_status(
            "q", {"final_answer": "give_up_and_restart", "answer_details": []}, sem) == "Unsolved"
        # a valid final answer but no Finish node in the chain -> Unsolved
        no_finish = {"final_answer": "x", "answer_details": [
            {"role": "tool", "message": {"name": "get_x"}, "next": []}]}
        assert await T._answer_status("q", no_finish, sem) == "Unsolved"
    asyncio.run(go())


def test_status_uses_primary_then_escalates_on_unsure(monkeypatch):
    calls = []

    async def _fake(prompt, *a, **k):
        calls.append(prompt)
        # first call = check_answer_status -> Unsure; second = parse_answer_status -> Solved
        # (the judge answers the function's arguments as JSON, `reason` included)
        status = "Unsure" if len(calls) == 1 else "Solved"
        return '{"answer_status": "%s", "reason": "r"}' % status, "gemini"
    monkeypatch.setattr(T, "_tooleval_cascade", _fake)
    sem = asyncio.Semaphore(1)
    status = asyncio.run(T._answer_status("q", _finish_answer(), sem))
    assert status == "Solved"                       # escalation verdict wins
    assert len(calls) == 2
    assert "check_answer_status" in calls[0] and "parse_answer_status" in calls[1]


def test_status_judge_outage_returns_none(monkeypatch):
    async def _outage(*a, **k):
        return None, "JUDGE_OUTAGE"
    monkeypatch.setattr(T, "_tooleval_cascade", _outage)
    sem = asyncio.Semaphore(1)
    assert asyncio.run(T._answer_status("q", _finish_answer(), sem)) is None


def test_prereqs_hard_error_without_docker_image(monkeypatch):
    # Gate is now the locally-built container (not a checkout you provision): no docker -> hard-error
    # pointing at the local Dockerfile build.
    import gbench.runners.eval_suites.toolbench as _t
    monkeypatch.setattr(_t.shutil, "which", lambda _x: None)
    ok, reason = T.check_toolbench_prerequisites()
    assert ok is False and "docker" in reason.lower() and "toolbench.Dockerfile" in reason
    with pytest.raises(RuntimeError, match="docs/evals/toolbench.md"):
        T.run_toolbench("m", "http://x:8000/v1", 1)


def test_score_mapping_and_leaderboard_is_not_comparable():
    # SoPR maps Solved/Unsure/Unsolved -> 1/0.5/0; the Gemini judge is never leaderboard-comparable.
    assert T._STATUS_SCORE == {"Solved": 1.0, "Unsure": 0.5, "Unsolved": 0.0}
    import inspect
    src = inspect.getsource(T.run_toolbench)
    assert '"leaderboard_comparable": False' in src
