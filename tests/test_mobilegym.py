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

"""Unit tests for the `mobilegym` and `mobilegym_pp` suites and their in-container launcher."""

import json
import os
import random
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from gbench.merge import merge_suite
from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import mobilegym as MG
from gbench.runners.eval_suites import mobilegym_pp as MGPP
from gbench.runners.eval_suites.sampling import stratified_indices

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "gbench" / "docker") not in sys.path:
    sys.path.insert(0, str(REPO / "gbench" / "docker"))
import mobilegym_run as MGRUN  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        "GBENCH_SHARD",
        "GBENCH_MOBILEGYM_IMAGE",
        "GBENCH_MOBILEGYM_WORKERS",
        "GBENCH_MOBILEGYM_TEMPERATURE",
        "GBENCH_MOBILEGYM_PP_TEMPERATURE",
        "GBENCH_MOBILEGYM_TIMEOUT",
        "OPENAI_API_KEY",
        "VLLM_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    base._RUN_KNOBS.clear()
    yield
    base._RUN_KNOBS.clear()


# =========================================================================== #
# Launcher parity with base.py and sampling.py
# =========================================================================== #
def test_launcher_stratified_indices_matches_sampling():
    rnd = random.Random(42)
    for _ in range(100):
        n = rnd.randint(0, 40)
        keys = [rnd.choice(["L1", "L2", "L3", None]) for _ in range(n)]
        limit = rnd.choice([None, 0, 1, 3, 7, n, n + 2])
        seed = rnd.choice(["mobilegym", "mobilegym_pp", None])
        assert MGRUN.stratified_indices(keys, limit, seed) == stratified_indices(keys, limit, seed)


def test_launcher_server_parser_functions_match_base():
    cases = [
        ("answer", None, None, False, 5, "answer"),
        ("print(1)\n", "def f():", None, False, 12, "def f():<channel|>print(1)\n"),
        (None, "Thought: click", None, False, 8, None),
        (None, None, None, False, 14, "<|tool_call>call:x{}<tool_call|>"),
        (None, None, None, True, 14, None),
        ("B", "reasoning text", None, True, 10, "<|thought>reasoning text<channel|>B"),
        ("", "thought", [{"id": "1"}], False, 9, None),
    ]
    for content, reasoning, tool_calls, thinking, comp_tokens, gen_text in cases:
        assert MGRUN.server_parser_signal(
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            thinking=thinking,
            completion_tokens=comp_tokens,
        ) == base.server_parser_signal(
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            thinking=thinking,
            completion_tokens=comp_tokens,
        )
        r_m = MGRUN.recover_server_parser_text(
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            thinking=thinking,
            completion_tokens=comp_tokens,
            generation_text=gen_text,
        )
        r_b = base.recover_server_parser_text(
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            thinking=thinking,
            completion_tokens=comp_tokens,
            generation_text=gen_text,
        )
        assert (r_m.event, r_m.text, r_m.recovered_from) == (
            r_b.event,
            r_b.text,
            r_b.recovered_from,
        )
        s_m = MGRUN.server_parser_residue(
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            thinking=thinking,
            generation_text=gen_text,
        )
        s_b = base.server_parser_residue(
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            thinking=thinking,
            generation_text=gen_text,
        )
        assert (s_m.event, s_m.text, s_m.recovered_from) == (
            s_b.event,
            s_b.text,
            s_b.recovered_from,
        )


def test_launcher_select_tasks_sharding_and_stratified_limit(tmp_path):
    items = [
        {"task_id": f"task_{i:02d}", "difficulty": ["L1", "L2", "L3"][i % 3]}
        for i in range(12)
    ]
    (tmp_path / "bench215").mkdir()
    (tmp_path / "bench215" / "official.json").write_text(
        json.dumps({"tasks": items}), encoding="utf-8"
    )
    selected, sel_info, shard_info = MGRUN._select_tasks_mobilegym_pp(
        tmp_path, limit=2, shard_index=2, num_shards=3
    )
    assert shard_info == {"spec": "2/3", "n_full": 12, "n_selected": 2}
    assert sel_info["limit"] == 2
    assert sel_info["limit_draw"] == "stratified_indices(key=difficulty, seed=mobilegym_pp)"
    assert len(selected) == 2

    # Empty shard when shard_index > n_full
    (tmp_path / "bench215" / "official.json").write_text(
        json.dumps({"tasks": items[:2]}), encoding="utf-8"
    )
    empty_sel, _, empty_shard = MGRUN._select_tasks_mobilegym_pp(
        tmp_path, limit=None, shard_index=3, num_shards=3
    )
    assert empty_sel == []
    assert empty_shard == {"spec": "3/3", "n_full": 2, "n_selected": 0}


# =========================================================================== #
# Gemma 4 & benchmark fidelity patches in mobilegym_run.py
# =========================================================================== #
def _setup_fake_bench_env(monkeypatch):
    fake_task_utils = types.ModuleType("bench_env.task.utils")
    fake_task_utils.default_tomorrow = lambda: "2099-01-01"

    class FakeMobileGymEnv:
        delay_after_action = 0.5

        @staticmethod
        def get_launch_args(headless: bool = True, proxy: Optional[str] = None, browser_type: str = "chromium"):
            return {"headless": headless, "args": ["--ignore-certificate-errors"]}

        def _attach_page_listeners(self):
            pass

        async def reset(self, app_ids: Optional[List[str]] = None):
            return None

    fake_mg_mod = types.ModuleType("bench_env.env.mobile_gym")
    fake_mg_mod.MobileGymEnv = FakeMobileGymEnv
    fake_mg_mod.random = random

    class FakeLLMClient:
        default_args = {}
        max_retries = 0
        total_timeout_s = 30.0
        model = "m"

        def chat(self, *, messages, args=None):
            return None

    fake_openai_chat = types.ModuleType("bench_env.llm.openai_chat")
    fake_openai_chat.LLMClient = FakeLLMClient
    fake_openai_chat.ChatResult = types.SimpleNamespace

    class FakeGenericAgentV2:
        def _parse_llm_output(self, raw_text: str):
            thought = ""
            action = {}
            for line in raw_text.splitlines():
                if line.startswith("Thought:"):
                    thought = line[len("Thought:") :].strip()
                elif line.startswith("Action:"):
                    action = {"raw": line[len("Action:") :].strip()}
            return thought, action

    fake_generic_v2 = types.ModuleType("bench_env.agent.generic_v2")
    fake_generic_v2.GenericAgentV2 = FakeGenericAgentV2

    import re as _re

    fake_q35 = types.ModuleType("bench_env.hybrid_agent.qwen35_session")
    fake_q35._TOOL_CALL_RE = _re.compile(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", _re.IGNORECASE)
    fake_q35.session_family = lambda model_name: "gemini" if "gemini" in model_name.lower() else "qwen"
    fake_q35.parse_session_tool_call = lambda text: None
    fake_q35._parse_model_output = lambda raw, content, *, prefer_xml=False: (
        fake_q35.parse_session_tool_call(content),
        None,
        {"role": "assistant", "content": content},
    )

    def _orig_transform(action: str, arguments: Dict[str, Any], width: int, height: int) -> Dict[str, Any]:
        if action == "terminate":
            st = str(arguments.get("status") or "").lower()
            if st == "success":
                return {"action_type": "status", "goal_status": "complete", "text": ""}
            if st == "failure":
                return {"action_type": "status", "goal_status": "infeasible", "text": ""}
            raise ValueError(f"Unknown terminate status: {st}")
        return {"action_type": action}

    fake_q35.transform_mobile_use = _orig_transform

    class FakeHybridArtifactRecorder:
        def _write_screenshot(self, stem, obs):
            return ""

    fake_artifacts = types.ModuleType("bench_env.hybrid_agent.artifacts")
    fake_artifacts.HybridArtifactRecorder = FakeHybridArtifactRecorder

    for name, mod in (
        ("bench_env", types.ModuleType("bench_env")),
        ("bench_env.task", types.ModuleType("bench_env.task")),
        ("bench_env.task.utils", fake_task_utils),
        ("bench_env.env", types.ModuleType("bench_env.env")),
        ("bench_env.env.mobile_gym", fake_mg_mod),
        ("bench_env.llm", types.ModuleType("bench_env.llm")),
        ("bench_env.llm.openai_chat", fake_openai_chat),
        ("bench_env.agent", types.ModuleType("bench_env.agent")),
        ("bench_env.agent.generic_v2", fake_generic_v2),
        ("bench_env.hybrid_agent", types.ModuleType("bench_env.hybrid_agent")),
        ("bench_env.hybrid_agent.qwen35_session", fake_q35),
        ("bench_env.hybrid_agent.artifacts", fake_artifacts),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return fake_task_utils, FakeGenericAgentV2, fake_q35


def test_frozen_clock_and_agent_patches(monkeypatch):
    fake_task_utils, FakeGenericAgentV2, fake_q35 = _setup_fake_bench_env(monkeypatch)
    MGRUN._install_common_patches(
        suite="mobilegym", temperature=0.0, max_tokens=4096, enable_thinking=True
    )
    assert fake_task_utils.default_tomorrow() == "2026-07-13"

    agent = FakeGenericAgentV2()
    thought, action = agent._parse_llm_output(
        "<think>\nLet me inspect the screen.\nThought: fake inside think\n</think>\n"
        "Thought: Click the search bar\nAction: click(x=500, y=200)"
    )
    assert thought == "Click the search bar"
    assert action == {"raw": "click(x=500, y=200)"}

    MGRUN._install_common_patches(
        suite="mobilegym_pp", temperature=0.0, max_tokens=4096, enable_thinking=False
    )
    assert fake_q35.session_family("google/gemma-4-26B-A4B-it") == "gemini"
    assert fake_q35.session_family("gemma-4-31b-it") == "gemini"
    assert fake_q35.session_family("Qwen/Qwen3.5-27B") == "qwen"

    # Gemma 4 native <|tool_call>call:NAME{...}<tool_call|> & normalization
    assert fake_q35.parse_session_tool_call(
        '<|tool_call>call:mobile_use{action: "terminate", status: "success"}<tool_call|>'
    ) == {"name": "mobile_use", "arguments": {"action": "terminate", "status": "success"}}
    assert fake_q35.parse_session_tool_call(
        '<|tool_call>call:ctrip__open_hotel{hotel_id: <|"|>h_quanji<|"|>}<tool_call|>'
    ) == {"name": "ctrip__open_hotel", "arguments": {"hotel_id": "h_quanji"}}
    assert fake_q35.parse_session_tool_call(
        '<|channel>thought\n<channel|><|tool_call>call:mobile_use:mobile_use'
        '{action: "mobile_use", parameters: {action: "swipe", coordinate: [500, 500], coordinate2: [500, 200]}}'
        "<tool_call|>"
    ) == {
        "name": "mobile_use",
        "arguments": {"action": "swipe", "coordinate": [500, 500], "coordinate2": [500, 200]},
    }
    assert fake_q35.parse_session_tool_call(
        '<|tool_call>call:mobile_use{action: "calendar__get_busy_intervals", '
        '"arguments": {"end": "2025-07-13T03:31:00", "start": "2025-07-13T03:30:00"}}<tool_call|>'
    ) == {
        "name": "calendar__get_busy_intervals",
        "arguments": {"end": "2025-07-13T03:31:00", "start": "2025-07-13T03:30:00"},
    }
    assert fake_q35.parse_session_tool_call(
        '<tool_call>\n{"name": "mobile_use", "arguments": {"action": "Home"}}\n</tool_call>'
    ) == {"name": "mobile_use", "arguments": {"action": "system_button", "button": "Home"}}
    assert fake_q35.parse_session_tool_call(
        '<tool_call>\n{"name": "mobile_use", "arguments": {"action": "terminate"}}\n</tool_call>'
    ) == {"name": "mobile_use", "arguments": {"action": "terminate", "status": "success"}}
    assert fake_q35.transform_mobile_use("terminate", {"action": "terminate"}, 1000, 1000) == {
        "action_type": "status",
        "goal_status": "complete",
        "text": "",
    }


_MOBILEGYM_LAUNCHER_SHA256 = {
    "2": "2f7890df51db5ad0b475a0272a99e38076fec54e69f9517d88220b90ffa39ff0",
}


def test_the_mobilegym_label_follows_the_launcher():
    import hashlib

    df = (REPO / "gbench" / "docker" / "mobilegym.Dockerfile").read_text(encoding="utf-8")
    assert f"LABEL {MG.LAUNCHER_LABEL}={MG.LAUNCHER_VERSION}" in df
    digest = hashlib.sha256()
    for name in ("mobilegym_run.py", "mobilegym_entrypoint.sh"):
        digest.update((REPO / "gbench" / "docker" / name).read_bytes())
    assert _MOBILEGYM_LAUNCHER_SHA256.get(MG.LAUNCHER_VERSION) == digest.hexdigest(), (
        "the mobilegym launcher changed: bump mobilegym.LAUNCHER_VERSION and the "
        "Dockerfile label, and record the new fingerprint here"
    )
    doc = (REPO / "docs" / "evals" / "mobilegym.md").read_text(encoding="utf-8")
    assert f"`LABEL {MG.LAUNCHER_LABEL}={MG.LAUNCHER_VERSION}`" in doc


def test_official_json_templates_override_preserves_parameterized_reset():
    class DummyTask:
        task_id = "meituan_01"
        templates = ["Order {dish} from {shop}"]
        _instruction_override = None

        def __init__(self):
            self.seeded = False

        def set_instruction(self, text: str) -> None:
            self._instruction_override = text

        def reset(self, seed: Optional[int] = None) -> str:
            if self._instruction_override is not None:
                return self._instruction_override
            self.seeded = True
            return self.templates[0]

    task = DummyTask()
    official_instruction = "Order Kung Pao Chicken from Sichuan Bistro"
    task.templates = [official_instruction]
    task._instruction_override = None
    out = task.reset(seed=42)
    assert out == official_instruction
    assert task.seeded is True


# =========================================================================== #
# Host runner prerequisites, execution, harness errors, and shard merge
# =========================================================================== #
def test_prerequisites_check_image_label_and_inotify(monkeypatch):
    monkeypatch.setattr(MG.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(MG, "_docker_image_exists", lambda _: True)
    monkeypatch.setattr(MG, "_check_inotify_ceiling", lambda: None)

    # Wrong launcher label refused
    monkeypatch.setattr(
        MG.subprocess,
        "run",
        lambda cmd, **kw: types.SimpleNamespace(
            returncode=0, stdout="0\n" if "--format" in cmd else "", stderr=""
        ),
    )
    ok, reason = MG.check_mobilegym_prerequisites()
    assert not ok and f"{MG.LAUNCHER_LABEL}={MG.LAUNCHER_VERSION}" in reason

    # Matching launcher label passes
    monkeypatch.setattr(
        MG.subprocess,
        "run",
        lambda cmd, **kw: types.SimpleNamespace(
            returncode=0,
            stdout=f"{MG.LAUNCHER_VERSION}\n" if "--format" in cmd else "",
            stderr="",
        ),
    )
    assert MG.check_mobilegym_prerequisites() == (True, "")
    assert MGPP.check_mobilegym_pp_prerequisites() == (True, "")


def _fake_summary(
    suite: str,
    tasks: List[Dict[str, Any]],
    *,
    shard: Optional[str] = None,
    limit: Optional[int] = None,
    n_full: int = 256,
) -> Dict[str, Any]:
    scored = [t for t in tasks if not t.get("harness_error")]
    n_scored = len(scored)
    passed = sum(1 for t in scored if t.get("success"))
    return {
        "suite": suite,
        "model": "google/gemma-4-26B-A4B-it",
        "simulated_time": MG.SIMULATED_TIME,
        "n_full": n_full,
        "n_selected": len(tasks),
        "n_scored": n_scored,
        "n_harness_errors": len(tasks) - n_scored,
        "empty_selection": len(tasks) == 0,
        "success_rate": (passed / n_scored) if n_scored else None,
        "progress_rate": 0.5 if n_scored else 0.0,
        "false_complete_rate": 0.0,
        "overdue_termination_rate": 0.0,
        "unexpected_side_effect_rate": 0.0,
        "difficulty_breakdown": {"L1": {"accuracy": 1.0, "n": 1, "passed": 1}},
        "app_count_breakdown": {"1_app": {"accuracy": 1.0, "n": 1, "passed": 1}},
        "selection": {
            "n_full": n_full,
            "n_after_shard": len(tasks),
            "limit": limit,
            "n_selected": len(tasks),
            "limit_draw": f"stratified_indices(key=difficulty, seed={suite})" if limit else None,
            "task_ids": [t["task_id"] for t in tasks],
        },
        "shard": (
            {"spec": shard, "n_full": n_full, "n_selected": len(tasks)} if shard else None
        ),
        "server_parser": {
            "server_parser_moved_text": 0,
            "server_parser_dropped_text": 0,
            "server_parser_text_lost": 0,
            "stopped_in_reasoning": 0,
        },
        "token_usage": {"llm_calls": 4, "prompt_tokens": 1000, "completion_tokens": 200},
        "per_task": tasks,
    }


def _stub_run(monkeypatch, summary: Dict[str, Any]):
    monkeypatch.setattr(MG, "_check_common_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(MG.container_guard, "require_metadata_blocked", lambda *a, **k: None)
    monkeypatch.setattr(MG.gemini_relay, "network_gateway", lambda *a, **k: "127.0.0.1")

    seen: Dict[str, Any] = {}

    def fake_subprocess_run(cmd, *a, **kw):
        if cmd[:2] == ["docker", "run"]:
            seen["cmd"] = list(cmd)
            seen["env"] = dict(kw.get("env") or {})
            mount = cmd[cmd.index("-v") + 1]
            workdir = mount.split(":", 1)[0]
            suite_name = cmd[cmd.index("--suite") + 1]
            with open(os.path.join(workdir, f"{suite_name}_summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(MG.subprocess, "run", fake_subprocess_run)
    return seen


def test_run_mobilegym_and_mobilegym_pp_exclude_harness_errors(monkeypatch):
    tasks = [
        {
            "task_id": "t1",
            "difficulty": "L1",
            "apps": ["calendar"],
            "app_count": 1,
            "success": True,
            "goal_success": True,
            "clean": True,
            "progress": 1.0,
            "steps": 3,
            "max_steps": 15,
            "stop_reason": "finish",
        },
        {
            "task_id": "t2",
            "difficulty": "L2",
            "apps": ["chat"],
            "app_count": 1,
            "success": False,
            "goal_success": False,
            "clean": True,
            "progress": 0.0,
            "steps": 0,
            "max_steps": 15,
            "stop_reason": "error",
            "error": "Timeout 60000ms exceeded in _wait_ready",
            "harness_error": True,
            "harness_error_reason": "harness_Or_endpoint_error: Timeout 60000ms exceeded",
        },
    ]
    summary = _fake_summary("mobilegym", tasks, limit=2, n_full=256)
    seen = _stub_run(monkeypatch, summary)
    res = MG.run_mobilegym(
        model_name="google/gemma-4-26B-A4B-it",
        base_url="http://127.0.0.1:8000/v1",
        limit=2,
        concurrency=2,
        enable_thinking=True,
        max_output_tokens=8192,
    )
    assert "--enable-thinking" in seen["cmd"]
    assert res["eval_name"] == "mobilegym"
    assert res["status"] == "completed_with_errors"
    assert res["n_selected"] == 2
    assert res["n_scored"] == 1
    assert res["correct_answers"] == 1
    assert res["category_accuracy"] == {"t1": {"total": 1, "correct": 1, "accuracy": 100.0}}
    assert res["n_endpoint_failed"] == 1
    assert res["scoring_excluded"] == 1
    assert res["accuracy"] == 100.0
    assert res["leaderboard_comparable"] is False
    assert res["sample_traces"][1]["scoring_excluded"] is True


def test_empty_shard_and_merge_shards(monkeypatch):
    monkeypatch.setenv("GBENCH_SHARD", "3/3")
    empty = _fake_summary("mobilegym_pp", [], shard="3/3", n_full=2)
    _stub_run(monkeypatch, empty)
    res_empty = MGPP.run_mobilegym_pp(
        model_name="google/gemma-4-26B-A4B-it",
        base_url="http://127.0.0.1:8000/v1",
    )
    assert res_empty["empty_shard"] is True
    assert res_empty["accuracy"] is None

    monkeypatch.setenv("GBENCH_SHARD", "1/2")
    s1 = _fake_summary(
        "mobilegym_pp",
        [{"task_id": "t1", "difficulty": "L1", "apps": ["a"], "app_count": 1, "success": True, "goal_success": True, "clean": True, "progress": 1.0}],
        shard="1/2",
        n_full=2,
    )
    _stub_run(monkeypatch, s1)
    r1 = MGPP.run_mobilegym_pp("m", "http://127.0.0.1:8000/v1")

    monkeypatch.setenv("GBENCH_SHARD", "2/2")
    s2 = _fake_summary(
        "mobilegym_pp",
        [{"task_id": "t2", "difficulty": "L2", "apps": ["b", "c"], "app_count": 2, "success": False, "goal_success": False, "clean": True, "progress": 0.5}],
        shard="2/2",
        n_full=2,
    )
    _stub_run(monkeypatch, s2)
    r2 = MGPP.run_mobilegym_pp("m", "http://127.0.0.1:8000/v1")

    monkeypatch.setattr(MG, "MOBILEGYM_PP_FULL_COUNT", 2)
    merged = merge_suite("mobilegym_pp", [r1, r2])
    assert merged["n_selected"] == 2
    assert merged["n_scored"] == 2
    assert merged["correct_answers"] == 1
    assert merged["category_accuracy"] == {
        "t1": {"total": 1, "correct": 1, "accuracy": 100.0},
        "t2": {"total": 1, "correct": 0, "accuracy": 0.0},
    }
    assert merged["accuracy"] == 50.0
    assert merged["progress_rate"] == 75.0
    assert merged["status"] == "success"
    assert merged["leaderboard_comparable"] is True

