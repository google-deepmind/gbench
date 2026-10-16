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

"""Turn counts of the mini-swe-agent 2.x path (swe_agentic._rollout_v2 / _v2_trace).

mini-swe-agent 2.4.x (LitellmModel.query -> parse_toolcall_actions) raises FormatError for a
reply without a tool call before the assistant message is built, and DefaultAgent.run then adds
only a role "user" format-error message whose extra.response is the reply. The messages below
copy the shape DefaultAgent.save writes for such replies (litellm 1.102.1 answering a
vLLM-shaped reply: `reasoning` arrives as `reasoning_content`, and also in
provider_specific_fields). No container, network or GPU is used: the runner container is a
stubbed subprocess.run.
"""

import asyncio
import json
import os
import types

import pytest

from gbench.runners.eval_suites import swe_agentic as A
from gbench.runners.eval_suites import swebench_common as C


@pytest.fixture(autouse=True)
def _no_metadata_probe(monkeypatch):
    """The cloud metadata server gate (swebench_common.require_task_network_blocked) starts probe
    containers; tests/test_guard_swe_agents.py covers it, here it is stubbed."""
    monkeypatch.setattr(C, "require_task_network_blocked", lambda *a, **k: None)


_FORMAT_ERROR_TEXT = ("Tool call error:\n\n<error>\nNo tool calls found in the response. Every "
                      "response MUST include at least one tool call.\n</error>")


def _response(message, finish="stop"):
    return {"id": "x", "created": 0, "model": "m", "object": "chat.completion",
            "system_fingerprint": None,
            "choices": [{"finish_reason": finish, "index": 0, "message": message,
                         "provider_specific_fields": {"stop_reason": None}}],
            "usage": {"completion_tokens": 5, "prompt_tokens": 10, "total_tokens": 15}}


def _reply(content=None, reasoning=None, tool_calls=None, legacy=False):
    msg = {"content": content, "role": "assistant", "tool_calls": tool_calls,
           "function_call": None, "provider_specific_fields": {"refusal": None}}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
        if not legacy:
            msg["provider_specific_fields"]["reasoning"] = reasoning
    return msg


def _format_error(reply, finish="stop"):
    """The message 2.x adds for a reply with no tool call: role user, the reply in extra."""
    return {"role": "user", "content": _FORMAT_ERROR_TEXT,
            "extra": {"interrupt_type": "FormatError", "cost": 0.0,
                      "response": _response(reply, finish)}}


def _tool_turn(reasoning="plan", command="ls"):
    calls = [{"function": {"arguments": json.dumps({"command": command}), "name": "bash"},
              "id": "call_1", "type": "function"}]
    reply = _reply(None, reasoning, calls)
    return dict(reply, extra={"actions": [{"command": command, "tool_call_id": "call_1"}],
                              "response": _response(reply, "tool_calls"), "cost": 0.0})


def _messages():
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        _format_error(_reply(None, "plan 1")),                      # empty after reasoning
        _tool_turn("plan 2"),                                        # a tool call
        {"role": "tool", "content": "<returncode>0</returncode>", "tool_call_id": "call_1",
         "extra": {"raw_output": "", "returncode": 0}},
        _format_error(_reply(None, "plan 3"), "length"),            # empty and cut off
        _format_error(_reply("THOUGHT: no call", "plan 4")),        # content, no tool call
        _format_error(_reply("", "plan 5", legacy=True)),           # empty (reasoning_content)
        _format_error(_reply("", None)),                             # empty, no reasoning
        {"role": "exit", "content": "", "extra": {"exit_status": "Submitted", "submission": ""}},
    ]


def _condensed(messages):
    """The 2.x runner's condensed_trajectory (gbench/docker/mini_swe_agent_v2_run.py), which
    cannot be imported here (it imports mini-swe-agent 2.x)."""
    out = []
    for m in messages:
        entry = {"role": m.get("role"), "content": m.get("content")}
        extra = m.get("extra") or {}
        choices = (extra.get("response") or {}).get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            if msg.get("reasoning_content") or msg.get("reasoning"):
                entry["reasoning"] = msg.get("reasoning_content") or msg.get("reasoning")
            if choices[0].get("finish_reason"):
                entry["finish_reason"] = choices[0]["finish_reason"]
        if extra.get("actions"):
            entry["actions"] = [a.get("command") for a in extra["actions"]]
        out.append(entry)
    return out


def _proc(rc=0, out="", err=""):
    return types.SimpleNamespace(returncode=rc, stdout=out, stderr=err)


def _task(iid="o__r-1"):
    return A.AgentTask(iid, f"issue {iid}", A.swebench_image(iid), "o/r")


def _write_traj(path, messages, api_calls):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"info": {"model_stats": {"instance_cost": 0.0, "api_calls": api_calls},
                            "exit_status": "Submitted", "submission": ""},
                   "messages": messages, "trajectory_format": "mini-swe-agent-1.1"}, f)


def _stub_runner(monkeypatch, *, result=None, traj=None, api_calls=6):
    """subprocess.run stand-in for the runner container: writes result.json (unless None) and
    the trajectory file (unless None) where the 2.x runner writes them."""
    monkeypatch.setattr(C, "host_pull_images",
                        lambda images, workers=1: {i: {"state": "present"} for i in images})

    def _run(cmd, **kw):
        if cmd[:2] == ["docker", "run"]:
            spec = json.load(open(cmd[-1]))
            iid = spec["instance"]["instance_id"]
            d = os.path.join(spec["output_dir"], iid)
            if traj is not None:
                _write_traj(os.path.join(d, f"{iid}.traj.json"), traj, api_calls)
            if result is not None:
                res = dict(result)
                if traj is not None:
                    res.setdefault("trajectory_file", os.path.join(d, f"{iid}.traj.json"))
                with open(os.path.join(d, "result.json"), "w", encoding="utf-8") as f:
                    json.dump(res, f)
        return _proc()
    monkeypatch.setattr(A.subprocess, "run", _run)


def _driver_result(messages, n_calls=6, truncated=1):
    return {"exit_status": "Submitted", "submission": "diff\n", "error": None,
            "infra_error": False, "n_calls": n_calls, "context_retry_turns": 0,
            "truncated_turns": truncated, "trajectory": _condensed(messages)}


# --------------------------------------------------------------------------------------------
# counting from the replies
# --------------------------------------------------------------------------------------------
def test_reply_counts_read_each_reply_whatever_role_holds_it():
    counts = A.v2_reply_counts(_messages())
    # replies: 1 (empty), 2 (tool call), 3 (empty, length), 4 (content), 5 (empty, legacy), 6.
    # Reply 3 stopped at the length limit: a truncated turn only, not one that ended with its
    # answer inside the thinking channel (base's ANSWER_IN_THINKING_POLICY).
    assert counts == {"model_replies": 6, "truncated_turns": 1,
                      "empty_after_reasoning_turns": 2}


def test_the_role_based_rule_misses_format_error_replies():
    # The 2.x runner's condensed trajectory files the reasoning-only replies under role user
    # with the format-error text as content; the 1.x rule (role assistant) sees none of them.
    assert A.empty_after_reasoning(_condensed(_messages())) == 0
    assert A.v2_reply_counts(_messages())["empty_after_reasoning_turns"] == 2


@pytest.mark.parametrize("message,empty", [
    (_reply(None, None), 0),
    (_reply("   ", "r"), 1),
    (dict(_reply(None, None), reasoning="r"), 1),
    (dict(_reply(None, None), provider_specific_fields={"reasoning": "r"}), 1),
    (_reply([{"type": "text", "text": "x"}], "r"), 0),
])
def test_reply_counts_reasoning_fields_and_content(message, empty):
    counts = A.v2_reply_counts([_format_error(message)])
    assert counts["model_replies"] == 1 and counts["empty_after_reasoning_turns"] == empty


def test_a_tool_call_with_empty_content_is_not_empty_after_reasoning():
    assert A.v2_reply_counts([_tool_turn("r")])["empty_after_reasoning_turns"] == 0


def test_an_unreadable_reply_makes_the_counts_unmeasured():
    bad = _format_error(_reply(None, "r"))
    bad["extra"]["response"] = "ModelResponse(id='x', ...)"      # upstream's repr() fallback
    assert A.v2_reply_counts(_messages() + [bad]) is None
    assert A.v2_reply_counts(None) is None and A.v2_reply_counts({"messages": []}) is None
    assert A.v2_reply_counts([{"role": "system", "content": "s"}]) == {
        "model_replies": 0, "truncated_turns": 0, "empty_after_reasoning_turns": 0}


def test_read_v2_trajectory(tmp_path):
    path = tmp_path / "t.traj.json"
    assert A.read_v2_trajectory(str(path)) is None and A.read_v2_trajectory(None) is None
    path.write_text("{not json")
    assert A.read_v2_trajectory(str(path)) is None
    path.write_text(json.dumps(["not", "an", "object"]))
    assert A.read_v2_trajectory(str(path)) is None
    _write_traj(str(path), _messages(), 6)
    assert A.read_v2_trajectory(str(path)) == {
        "model_replies": 6, "truncated_turns": 1, "empty_after_reasoning_turns": 2,
        "bash_env_conda_turns": 0, "api_calls": 6}
    _write_traj(str(path), _messages(), None)
    assert A.read_v2_trajectory(str(path))["api_calls"] is None


# --------------------------------------------------------------------------------------------
# the rollout's trace
# --------------------------------------------------------------------------------------------
def test_rollout_v2_counts_empty_replies_from_the_trajectory_file(monkeypatch, tmp_path):
    msgs = _messages()
    _stub_runner(monkeypatch, result=_driver_result(msgs), traj=msgs)
    tr = A._rollout_v2(0, _task(), {"model": {}}, str(tmp_path), image="img", version="2.4.6")
    assert tr["empty_after_reasoning_turns"] == 2
    assert (tr["agent_steps"], tr["truncated_turns"], tr["context_retry_turns"]) == (6, 1, 0)
    assert tr["unmeasured_counters"] == []
    assert tr["trajectory_file"] == str(tmp_path / "o__r-1" / "o__r-1.traj.json")
    # the condensed trajectory on the trace still shows the format-error message (role user)
    assert [m["role"] for m in tr["trajectory"]][2] == "user"
    assert tr["trajectory"][2]["reasoning"] == "plan 1"


def test_rollout_v2_with_calls_but_no_trajectory_file_leaves_the_count_unmeasured(
        monkeypatch, tmp_path):
    _stub_runner(monkeypatch, result=_driver_result(_messages()), traj=None)
    tr = A._rollout_v2(0, _task(), {"model": {}}, str(tmp_path), image="img", version="2.4.6")
    assert tr["empty_after_reasoning_turns"] is None
    assert tr["unmeasured_counters"] == ["empty_after_reasoning_turns"]
    assert (tr["agent_steps"], tr["truncated_turns"], tr["context_retry_turns"]) == (6, 1, 0)


def test_rollout_v2_without_model_calls_measures_zero(monkeypatch, tmp_path):
    res = dict(_driver_result([], n_calls=0, truncated=0), infra_error=True,
               exit_status="RuntimeError", error="container did not start",
               failed_phase="environment", trajectory_file=None)
    _stub_runner(monkeypatch, result=res, traj=None)
    tr = A._rollout_v2(0, _task(), {"model": {}}, str(tmp_path), image="img", version="2.4.6")
    assert tr["empty_after_reasoning_turns"] == 0 and tr["unmeasured_counters"] == []
    assert tr["agent_infra_error"] is True and tr["trajectory_file"] is None


def test_rollout_v2_without_a_result_reads_the_trajectory_file(monkeypatch, tmp_path):
    _stub_runner(monkeypatch, result=None, traj=_messages(), api_calls=6)
    tr = A._rollout_v2(0, _task(), {"model": {}}, str(tmp_path), image="img", version="2.4.6")
    assert tr["agent_infra_error"] is True and tr["agent_exit_status"] == "RunnerError"
    assert (tr["agent_steps"], tr["truncated_turns"], tr["empty_after_reasoning_turns"]) == \
        (6, 1, 2)
    # no per-turn cap was sent, so the runner resent no turn
    assert tr["context_retry_turns"] == 0 and tr["unmeasured_counters"] == []
    assert tr["trajectory_file"] == str(tmp_path / "o__r-1" / "o__r-1.traj.json")


def test_rollout_v2_without_a_result_under_a_cap_cannot_know_the_resends(monkeypatch, tmp_path):
    _stub_runner(monkeypatch, result=None, traj=_messages(), api_calls=6)
    tr = A._rollout_v2(0, _task(), {"model": {"model_kwargs": {"max_tokens": 512}}},
                       str(tmp_path), image="img", version="2.4.6")
    assert tr["context_retry_turns"] is None
    assert tr["unmeasured_counters"] == ["context_retry_turns"]


def test_rollout_v2_without_a_result_or_trajectory_measures_nothing(monkeypatch, tmp_path):
    _stub_runner(monkeypatch, result=None, traj=None)
    tr = A._rollout_v2(0, _task(), {}, str(tmp_path), image="img", version="2.4.6")
    assert tr["agent_infra_error"] is True
    assert (tr["agent_steps"], tr["truncated_turns"], tr["empty_after_reasoning_turns"]) == \
        (None, None, None)
    assert tr["context_retry_turns"] == 0
    assert tr["unmeasured_counters"] == ["agent_steps", "truncated_turns",
                                         "empty_after_reasoning_turns"]


def test_an_image_that_does_not_pull_measures_zero(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "host_pull_images", lambda images, workers=1: {
        i: {"state": "failed", "error": "denied"} for i in images})
    monkeypatch.setattr(A.subprocess, "run", lambda *a, **k: pytest.fail("no container"))
    tr = A._rollout_v2(0, _task(), {}, str(tmp_path), image="img", version="2.4.6")
    assert tr["agent_exit_status"] == "ImagePullFailed"
    assert (tr["agent_steps"], tr["truncated_turns"], tr["empty_after_reasoning_turns"],
            tr["context_retry_turns"], tr["unmeasured_counters"]) == (0, 0, 0, 0, [])


# --------------------------------------------------------------------------------------------
# the run's result
# --------------------------------------------------------------------------------------------
def _desc():
    return {"version": "2.4.6", "config_name": A.AGENT_V2_CONFIG,
            "config_sha256": A.AGENT_V2_CONFIG_SHA256["2.4.6"],
            "config": {"agent": {"step_limit": 250, "cost_limit": 3.0},
                       "environment": {"timeout": 60},
                       "model": {"model_kwargs": {"drop_params": True}}}}


async def _resolve_all(traces):
    for t in traces:
        t["is_correct"] = True


def _run(monkeypatch, tmp_path, tasks):
    monkeypatch.setenv("GBENCH_SWE_BENCH_MULTILINGUAL_TRAJ_DIR", str(tmp_path / "trajs"))
    monkeypatch.setattr(A, "describe_agent_v2", lambda v: _desc())
    # the relay the 2.x containers reach the endpoint through listens on loopback here
    monkeypatch.setattr(A, "bridge_gateway", lambda: ("127.0.0.1", ""))
    monkeypatch.setattr(A, "require_relay_reachable", lambda *a, **k: None)   # no docker
    return A.run_agentic("swe_bench_multilingual", "m", "http://h:1/v1", 1, tasks,
                         _resolve_all, {}, agent_v2="2.4.6")


def test_run_counts_format_error_replies_as_the_models_outcome(monkeypatch, tmp_path):
    msgs = _messages()
    _stub_runner(monkeypatch, result=_driver_result(msgs), traj=msgs)
    res = _run(monkeypatch, tmp_path, [_task("a__a-1"), _task("b__b-2")])
    assert res["empty_after_reasoning_turns"] == 4 and res["truncated_turns"] == 2
    assert res["agent_steps_total"] == 12 and "unmeasured_counters" not in res
    # The relay (the only place a reply's token ids are seen) saw none of these replies, so the
    # empty turns cannot be told apart from a native tool call the server's parser removed:
    # unclassified, never answer_in_thinking, and a comparability reason. The relay-routed case
    # is test_empty_reply_classification_and_upstream_reask.py.
    assert (res["answer_in_thinking"], res["answer_in_thinking_of"]) == (0, 0)
    assert res["unclassified_empty_turns"] == 4 and res["tool_call_dropped_turns"] == 0
    reason = res["leaderboard_comparable_reason"]
    assert res["leaderboard_comparable"] is False
    assert "2 model turn(s) cut off at the end of the served context window" in reason
    assert "4 model turn(s) that came back with reasoning and no content or tool call " \
        "without token ids" in reason


def test_run_with_clean_replies_stays_comparable(monkeypatch, tmp_path):
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"},
            _tool_turn("plan", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")]
    _stub_runner(monkeypatch, result=_driver_result(msgs, n_calls=1, truncated=0), traj=msgs)
    res = _run(monkeypatch, tmp_path, [_task()])
    assert (res["empty_after_reasoning_turns"], res["truncated_turns"]) == (0, 0)
    # the only deviation: the temperature gbench always sends (the bundled config sends none)
    assert res["leaderboard_comparable"] is False
    assert res["leaderboard_comparable_reason"].startswith("temperature=0.0 sent by the no-think rule")
    assert ";" not in res["leaderboard_comparable_reason"].split("; the bundled", 1)[1]


def test_run_with_an_unmeasured_count_is_not_comparable(monkeypatch, tmp_path):
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"},
            _tool_turn("plan", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")]
    _stub_runner(monkeypatch, result=_driver_result(msgs, n_calls=1, truncated=0), traj=None)
    res = _run(monkeypatch, tmp_path, [_task()])
    assert res["empty_after_reasoning_turns"] == 0
    assert res["unmeasured_counters"] == {"empty_after_reasoning_turns": 1}
    assert res["leaderboard_comparable"] is False
    assert ("counts not measured on some instance(s) (runner result or trajectory file missing "
            "or unreadable): empty_after_reasoning_turns on 1") in \
        res["leaderboard_comparable_reason"]


def test_run_totals_skip_unmeasured_steps(monkeypatch, tmp_path):
    _stub_runner(monkeypatch, result=None, traj=None)
    res = _run(monkeypatch, tmp_path, [_task()])
    assert res["status"] == "error" and res["agent_steps_total"] == 0
    assert res["unmeasured_counters"] == {"agent_steps": 1, "empty_after_reasoning_turns": 1,
                                          "truncated_turns": 1}
    assert res["sample_traces"][0]["agent_steps"] is None


# --------------------------------------------------------------------------------------------
# docs
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["copilot_bench_swe.md", "swe_bench_multilingual.md"])
def test_docs_say_what_is_counted(name):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    doc = " ".join(open(os.path.join(root, "docs", "evals", name), encoding="utf-8").read()
                   .split())
    for phrase in ("`empty_after_reasoning_turns` model replies that stopped on their own (not "
                   "at `length`) with reasoning (`reasoning_content` or `reasoning`) but neither "
                   "content nor a tool call",
                   "the model's outcome, counted with its share of the relay's replies "
                   "(`answer_in_thinking`, `answer_in_thinking_rate` over the relay's "
                   "`completion_replies`, numerator and denominator the same replies) and not a "
                   "comparability reason",
                   "a native tool call the server's tool-call parser removed "
                   "(`tool_call_dropped_turns`, a server-parser artifact)",
                   "`unclassified_empty_turns`",
                   "files a reply without a tool call as a role `user` format-error message "
                   "whose `extra.response` is that reply",
                   "`empty_after_reasoning_turns` is counted from the responses in the "
                   "instance's `.traj.json`",
                   "is `null` on its trace, never 0, and named in the trace's "
                   "`unmeasured_counters`",
                   "every count measured"):
        assert phrase in doc, (name, phrase)
