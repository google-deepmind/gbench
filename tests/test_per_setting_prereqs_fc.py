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

"""One official metric or category whose prerequisite is missing is reported infra_required in
its own entry while the rest of the suite runs and is scored.

nestful: only the executable Win Rate needs IBM/NESTFUL's reference functions
(data_v2/executable_functions); without them F1 Intent / F1 Slot / Partial / Full Match are
computed (upstream's calculate_scores(win_rate_flag=False) path) and the Win Rate is null with its
own entry. The rows come only from the checkout's data_v2/nestful_data.jsonl; without it the whole
suite is infra_required.

acebench: only agent_multi_turn needs its user simulator (the Gemini judge cascade); without a
usable key the other 16 categories run and agent_multi_turn is reported infra_required, with the
aggregates that weigh it (agent_summary, the Summary) null rather than computed with a 0.
"""
import json
from types import SimpleNamespace
from unittest import mock

import pytest

from gbench.runners.eval_suites import acebench as AB
from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import nestful as N


def _endpoint(answer):
    """A fake server: `answer(messages)` -> reply text."""
    async def _reply(**kw):
        return base.Reply(answer(kw["messages"]), None, "stop", completion_tokens=5)
    return _reply


def _through_reports(res, name, tmp_path, monkeypatch, summary=True):
    """Send a suite result through the real evals.py result block and the cli CSV summary (and
    the printed summary when `summary`)."""
    from gbench import cli
    from gbench.runners import evals as E
    monkeypatch.setattr(E, "_resolve_remote_model_id", lambda url, fallback: "m")
    monkeypatch.setattr(E, "_server_max_model_len", lambda url: None)
    monkeypatch.setattr(E.EvalsBenchmarkRunner, "_run_single_eval",
                        lambda self, eval_name, **k: dict(res))
    cfg = SimpleNamespace(evals=[name], eval_custom_jsonl=None, eval_plugins_dir=None,
                          batch_sizes=[1], remote_endpoint="http://127.0.0.1:1/v1", tokenizer="t",
                          eval_max_output_tokens=1024, skip_existing=False,
                          log_manager=SimpleNamespace(results_dir=tmp_path, base_dir=tmp_path,
                                                      save_result=lambda p, r: None))
    model = SimpleNamespace(name="m", short_name="m", hf_model_id="org/m")
    out = E.EvalsBenchmarkRunner(cfg).run(model, SimpleNamespace(value="remote-endpoint"))
    assert [r["eval_name"] for r in out] == [name] and out[0].get("status") != "error"
    if summary:
        cli.print_results_summary(out, SimpleNamespace(results_dir=tmp_path))
    csv = cli._save_eval_summary_csv(tmp_path, out, [], [("5. Tool Use", [name])], {})
    assert csv is not None and name.upper() in csv.read_text()
    return out[0], csv.read_text()


# --------------------------------------------------------------------------- #
# nestful
# --------------------------------------------------------------------------- #
_TOOLS = [{"name": "add", "description": "", "parameters": {},
           "output_parameters": {"result": {}}}]
_OUT = [{"name": "add", "label": "$var_1", "arguments": {"a": 1, "b": 2}}]
_ROWS = [
    {"sample_id": "s1", "input": "1+2?", "tools": _TOOLS, "output": _OUT, "gold_answer": 3},
    {"sample_id": "s2", "input": "2+2?", "tools": _TOOLS,
     "output": [{"name": "add", "label": "$var_1", "arguments": {"a": 2, "b": 2}}],
     "gold_answer": 4},
]


def _reply_nestful(messages):
    # s1 answered right, s2 wrong (an empty call list)
    return json.dumps(_OUT) if messages[-1]["content"] == "1+2?" else "[]"


class _NoWorker:
    def __init__(self, *a, **k):
        raise AssertionError("the Win Rate worker must not start without the reference functions")


def _forbid_hf(monkeypatch):
    """nestful has no Hugging Face data path: any download attempt fails the test."""
    import huggingface_hub

    def _no_hf(*a, **k):
        raise AssertionError("nestful must never read a Hugging Face copy of the data")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _no_hf)


def _checkout_data(tmp_path, monkeypatch):
    """A checkout's data_v2/nestful_data.jsonl (pinned blob set to this file)."""
    data_v2 = tmp_path / "NESTFUL" / "data_v2"
    data_v2.mkdir(parents=True)
    path = data_v2 / "nestful_data.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in _ROWS))
    monkeypatch.setitem(N._PINNED_BLOBS, "nestful_data.jsonl", N._git_blob_sha1(str(path)))
    return data_v2


def _no_checkout_env(monkeypatch):
    monkeypatch.delenv(N._FUNC_DIR_ENV, raising=False)
    monkeypatch.delenv("GBENCH_PREREQS_DIR", raising=False)


def _assert_win_rate_infra(r, reason_part):
    assert r["win_rate"] is None
    assert r["win_rate_killed_calls"] is None
    assert r["metrics_infra_required"] == ["win_rate"]
    entry = r["win_rate_metric"]
    assert entry["status"] == "infra_required" and entry["metric"] == "win_rate"
    assert entry["value"] is None and entry["leaderboard_comparable"] is False
    assert N._FUNC_DIR_ENV in entry["prerequisite"] and N.NESTFUL_COMMIT in entry["prerequisite"]
    assert reason_part in entry["infra_required"]
    assert entry["error"].startswith("[INFRA REQUIRED] nestful: NESTFUL's Win Rate was not scored")
    # the other metrics are computed, the headline among them
    assert r["full_match_accuracy"] == 50.0 and r["accuracy"] == 50.0
    assert r["partial_match_accuracy"] == 50.0
    assert isinstance(r["f1_intent"], float) and isinstance(r["f1_slot"], float)
    assert r["parsing_errors"] == 0
    assert r["status"] == "completed_with_errors"
    assert r["leaderboard_comparable"] is False
    assert "Win Rate was not scored" in r["leaderboard_comparable_reason"]
    assert "not scored (infra_required" in r["metric"]
    assert all("nestful_win" not in t for t in r["sample_traces"])


def test_nestful_without_functions_scores_the_other_metrics(tmp_path, monkeypatch):
    data_v2 = _checkout_data(tmp_path, monkeypatch)
    _no_checkout_env(monkeypatch)
    # the functions directory the variable names does not exist; the data next to it does
    monkeypatch.setenv(N._FUNC_DIR_ENV, str(data_v2 / "executable_functions"))
    monkeypatch.setattr(N, "_WinRateWorker", _NoWorker)
    _forbid_hf(monkeypatch)
    with mock.patch.object(base, "_send_single_request", _endpoint(_reply_nestful)):
        r = N.run_nestful("m", "http://x", 1)
    _assert_win_rate_infra(r, "the executable Win Rate needs IBM/NESTFUL's reference functions")
    assert r["dataset_source"] == "checkout"
    assert r["dataset_revision"] == f"IBM/NESTFUL@{N.NESTFUL_COMMIT} data_v2/nestful_data.jsonl"
    # the gold still carries the typed gold_answer of the checkout
    assert json.loads(r["sample_traces"][0]["gold_answer"])["gold_answer"] in ("3", "4")


def test_nestful_prereqs_dir_data_without_functions(tmp_path, monkeypatch):
    _checkout_data(tmp_path, monkeypatch)
    _no_checkout_env(monkeypatch)
    monkeypatch.setenv("GBENCH_PREREQS_DIR", str(tmp_path))
    monkeypatch.setattr(N, "_WinRateWorker", _NoWorker)
    _forbid_hf(monkeypatch)
    with mock.patch.object(base, "_send_single_request", _endpoint(_reply_nestful)):
        r = N.run_nestful("m", "http://x", 1)
    _assert_win_rate_infra(r, "reference functions")
    assert r["dataset_source"] == "checkout"


def test_nestful_functions_at_another_revision_only_drop_the_win_rate(tmp_path, monkeypatch):
    data_v2 = _checkout_data(tmp_path, monkeypatch)
    fd = data_v2 / "executable_functions"
    fd.mkdir()
    (fd / "basic_functions.py").write_text("def add(a, b):\n    return a + b\n")
    (fd / "func_file_map.json").write_text("{}")
    _no_checkout_env(monkeypatch)
    monkeypatch.setenv(N._FUNC_DIR_ENV, str(fd))
    monkeypatch.setattr(N, "_WinRateWorker", _NoWorker)
    _forbid_hf(monkeypatch)
    with mock.patch.object(base, "_send_single_request", _endpoint(_reply_nestful)):
        r = N.run_nestful("m", "http://x", 1)
    _assert_win_rate_infra(r, N.NESTFUL_COMMIT)
    assert "basic_functions.py does not match" in r["win_rate_metric"]["infra_required"]


def test_nestful_func_dir_raises_the_win_rate_error(monkeypatch):
    _no_checkout_env(monkeypatch)
    with pytest.raises(N.WinRateUnavailable) as e:
        N._func_dir()
    assert isinstance(e.value, RuntimeError)
    assert "executable Win Rate needs" in e.value.reason
    assert str(e.value).startswith("[INFRA REQUIRED] nestful: NESTFUL's Win Rate cannot be scored")


def test_nestful_checkout_data_at_another_revision_is_still_a_hard_error(tmp_path, monkeypatch):
    data_v2 = tmp_path / "NESTFUL" / "data_v2"
    data_v2.mkdir(parents=True)
    (data_v2 / "nestful_data.jsonl").write_text(json.dumps(_ROWS[0]) + "\n")
    _no_checkout_env(monkeypatch)
    monkeypatch.setenv(N._FUNC_DIR_ENV, str(data_v2 / "executable_functions"))
    _forbid_hf(monkeypatch)
    with pytest.raises(RuntimeError, match=N.NESTFUL_COMMIT) as e:
        N.run_nestful("m", "http://x", 1)
    assert "[INFRA REQUIRED] nestful cannot run" in str(e.value)


def _no_rollout(**kw):
    raise AssertionError("no request may be sent when the suite is infra_required")


def test_nestful_without_any_checkout_is_infra_required_for_the_whole_suite(monkeypatch):
    # The checkout is NESTFUL's only data source: without it nothing runs (no HF copy, no rollout).
    _no_checkout_env(monkeypatch)
    _forbid_hf(monkeypatch)
    monkeypatch.setattr(N, "_WinRateWorker", _NoWorker)
    with mock.patch.object(base, "_send_single_request", _no_rollout):
        with pytest.raises(RuntimeError) as e:
            N.run_nestful("m", "http://x", 1)
    text = str(e.value)
    assert text.startswith("[INFRA REQUIRED] nestful cannot run")
    assert "no IBM/NESTFUL checkout" in text and "data_v2/nestful_data.jsonl" in text
    assert N.NESTFUL_COMMIT in text
    assert not isinstance(e.value, N.WinRateUnavailable)     # the whole suite, not the Win Rate


def test_nestful_rows_come_only_from_the_checkout(tmp_path, monkeypatch):
    # The removed Hugging Face path (IBM's str()-coerced parquet copy) is gone, not dormant.
    for name in ("_hf_rows", "HF_DATASET", "HF_REVISION", "HF_FILE", "HF_SHA256"):
        assert not hasattr(N, name), name
    import inspect
    assert "rows" not in inspect.signature(N._load_nestful_samples).parameters
    data_v2 = _checkout_data(tmp_path, monkeypatch)
    ck = N._load_nestful_samples(data_path=str(data_v2 / "nestful_data.jsonl"))
    for (_m, g, _c), row in zip(ck, _ROWS):
        g = json.loads(g)
        assert g["output"] == row["output"] and g["tools"] == row["tools"]
        assert json.loads(g["gold_answer"]) == row["gold_answer"]      # typed, from the checkout


def test_nestful_functions_dir_without_its_data_file_is_a_hard_error(tmp_path, monkeypatch):
    # GBENCH_NESTFUL_FUNC_DIR names a checkout whose data_v2/ has no nestful_data.jsonl.
    data_v2 = tmp_path / "NESTFUL" / "data_v2"
    data_v2.mkdir(parents=True)
    _no_checkout_env(monkeypatch)
    monkeypatch.setenv(N._FUNC_DIR_ENV, str(data_v2 / "executable_functions"))
    _forbid_hf(monkeypatch)
    with pytest.raises(RuntimeError) as e:
        N.run_nestful("m", "http://x", 1)
    text = str(e.value)
    assert "[INFRA REQUIRED] nestful cannot run" in text
    assert str(data_v2 / "nestful_data.jsonl") + " is missing" in text


def test_nestful_no_data_anywhere_names_the_checkout(monkeypatch):
    _no_checkout_env(monkeypatch)
    _forbid_hf(monkeypatch)
    with pytest.raises(RuntimeError) as e:
        N.run_nestful("m", "http://x", 1)
    text = str(e.value)
    assert "[INFRA REQUIRED] nestful cannot run" in text
    assert N._FUNC_DIR_ENV in text and "GBENCH_PREREQS_DIR" in text
    assert "github.com/IBM/NESTFUL" in text and N.NESTFUL_COMMIT in text
    assert "huggingface" not in text.lower() and "ibm-research" not in text


class _InProcWorker:
    def __init__(self, func_dir, *a, **k):
        self.func_dir, self.kills = func_dir, 0

    def score(self, pred, gold, tools):
        return N._win_score(pred, gold, tools, self.func_dir)

    def close(self):
        pass


def test_nestful_with_functions_scores_the_win_rate_and_reports_no_infra(tmp_path, monkeypatch):
    data_v2 = _checkout_data(tmp_path, monkeypatch)
    fd = data_v2 / "executable_functions"
    fd.mkdir()
    (fd / "basic_functions.py").write_text("def add(a, b):\n    return a + b\n")
    (fd / "func_file_map.json").write_text("{}")
    monkeypatch.setattr(N, "_func_dir", lambda: str(fd))
    monkeypatch.setattr(N, "_WinRateWorker", _InProcWorker)
    _forbid_hf(monkeypatch)
    with mock.patch.object(base, "_send_single_request", _endpoint(_reply_nestful)):
        r = N.run_nestful("m", "http://x", 1)
    assert r["win_rate"] == 50.0 and r["win_rate_killed_calls"] == 0
    assert "win_rate_metric" not in r and "metrics_infra_required" not in r
    assert r["status"] == "success" and r["dataset_source"] == "checkout"


def test_nestful_win_rate_infra_result_through_reports(tmp_path, monkeypatch):
    data_v2 = _checkout_data(tmp_path, monkeypatch)
    _no_checkout_env(monkeypatch)
    monkeypatch.setenv(N._FUNC_DIR_ENV, str(data_v2 / "executable_functions"))
    monkeypatch.setattr(N, "_WinRateWorker", _NoWorker)
    _forbid_hf(monkeypatch)
    with mock.patch.object(base, "_send_single_request", _endpoint(_reply_nestful)):
        r = N.run_nestful("m", "http://x", 1)
    out, _csv = _through_reports(r, "nestful", tmp_path, monkeypatch)
    assert out["status"] == "completed_with_errors"
    assert out["win_rate_metric"]["status"] == "infra_required"


# --------------------------------------------------------------------------- #
# acebench
# --------------------------------------------------------------------------- #
_ACE_FNS = [{"name": "Api", "parameters": {"type": "dict", "properties": {"q": {"type": "string"}},
                                           "required": ["q"]}}]


def _ace_sample(sub, row_id, reply_key, gt, language="en"):
    gold = json.dumps({"id": row_id, "sub": sub, "language": language, "ground_truth": gt,
                       "functions": _ACE_FNS, "question": "q"})
    return ([{"role": "user", "content": reply_key}], gold,
            {"category": AB._category_key(sub, language)})


_ACE_REPLIES = {"good": "[Api(q='hello')]", "bad": "[Api(q='bye')]",
                "irr": "Due to the limitations of the function, I cannot solve this problem."}
_GOOD = {"Api": {"q": "hello"}}


def _ace_singles(language="en"):
    if language == "en":
        return [_ace_sample("normal_atom_bool", "normal_atom_bool_0", "good", _GOOD),
                _ace_sample("normal_atom_bool", "normal_atom_bool_1", "bad", _GOOD),
                _ace_sample("special_irrelevant", "special_irrelevant_0", "irr", "x")]
    return [_ace_sample("normal_atom_bool", "normal_atom_bool_0", "good", _GOOD, "zh")]


def _ace_agent_items(language="en"):
    return [{"language": language, "category": sub, "key": AB._category_key(sub, language),
             "row": {"id": f"{sub}_0"}, "answer": {"ground_truth": [], "mile_stone": []}}
            for sub in AB.AGENT_CATEGORIES]


def _agent_trace(it):
    return {"category": it["key"], "acebench_language": it["language"],
            "question_id": it["row"]["id"], "attempt_idx": it.get("attempt_idx", 0),
            "gold_answer": json.dumps({"id": it["row"]["id"], "sub": it["category"],
                                       "language": it["language"], "ground_truth": [],
                                       "mile_stone": []}),
            "status": "OK", "error": None, "agent_dialogue": [], "agent_result": [],
            "agent_process": [], "model_requests": 1, "truncated_replies": 0,
            "replies_at_official_max_tokens": 0, "completion_tokens": 5, "user_sim_requests": 0,
            "user_sim_models": {}, "user_sim_replies_cut_at_max_tokens": 0,
            "user_sim_empty_at_max_tokens": 0, "window_limited_requests": 0,
            "window_over_steps": 0, "window_uncounted_requests": 0,
            "execution_guard_refusals": [], "dialogue_steps": 1, "is_correct": True,
            "agent_error_type": None, "process_accuracy": 1.0, "response_text": "done"}


def _ace_setup(monkeypatch, keys, live=(True, ""), singles=_ace_singles, agents=_ace_agent_items):
    monkeypatch.setattr(AB, "_load_acebench_samples",
                        lambda language="en", **k: singles(language))
    monkeypatch.setattr(AB, "_load_agent_items", lambda language="en": agents(language))
    monkeypatch.setattr(AB, "check_binop_evaluator", lambda: None)
    monkeypatch.setattr(AB, "gemini_api_keys", lambda: list(keys))
    monkeypatch.setattr(AB, "gemini_key_live_valid", lambda key: live)
    ran: list = []

    async def fake_agents(items, *a, **k):
        ran.extend(items)
        return [_agent_trace(it) for it in items]
    monkeypatch.setattr(AB, "_run_agent_items", fake_agents)
    return ran


def _run_ace():
    reply = _endpoint(lambda m: _ACE_REPLIES[m[-1]["content"]])
    with mock.patch.object(base, "_send_single_request", reply):
        return AB.run_acebench("m", "http://x", 1)


def _assert_agent_multi_turn_infra(r, ran, reason_part):
    # only agent_multi_step conversations ran
    assert sorted((it["language"], it["category"]) for it in ran) == [
        ("en", "agent_multi_step"), ("zh", "agent_multi_step")]
    assert r["categories_infra_required"] == ["agent_multi_turn"]
    entry = r["agent_multi_turn_category"]
    assert entry["status"] == "infra_required" and entry["category"] == "agent_multi_turn"
    assert entry["languages"] == ["en", "zh"]
    assert entry["conversations_not_run"] == {"en": 1, "zh": 1}
    assert "GEMINI_API_KEY" in entry["prerequisite"]
    assert reason_part in entry["infra_required"]
    assert entry["error"].startswith("[INFRA REQUIRED] acebench: agent_multi_turn was not run")
    assert entry["accuracy"] is None and entry["leaderboard_comparable"] is False
    for key in ("agent_multi_turn", "agent_multi_turn_zh"):
        cat = r["category_accuracy"][key]
        assert cat == {"correct": 0, "total": 0, "accuracy": None, "status": "infra_required"}
    for per_sub in (r["per_sub_category"], r["per_sub_category_zh"]):
        mt = per_sub["agent_multi_turn"]
        assert mt["status"] == "infra_required" and mt["accuracy"] is None
        assert mt["process_accuracy"] is None and mt["total"] == 0
        assert per_sub["agent_multi_step"]["accuracy"] == 100.0
    # never scored, not even as 0: every aggregate that weighs it is null
    assert r["agent_summary"] is None and r["accuracy"] is None and r["overall_zh"] is None
    assert r["leaderboard_columns"]["agent"] is None
    assert r["leaderboard_columns"]["overall"] is None
    assert r["per_language"]["zh"]["overall"] is None
    # what ran is scored as upstream aggregates it
    assert r["normal_summary"] == round(round(0.5 / 12, 3) * 100, 2)
    assert r["special_summary"] == round(round(1 / 3, 3) * 100, 2)
    assert r["leaderboard_columns"]["atom"] == round(round(0.5 / 6, 3) * 100, 2)
    assert r["per_language"]["zh"]["normal_summary"] == round(round(1 / 12, 3) * 100, 2)
    assert r["category_accuracy"]["normal_atom_bool"] == {"correct": 1, "total": 2,
                                                          "accuracy": 50.0}
    # counts cover what ran: 4 single-turn rows + 2 agent_multi_step conversations
    assert r["total_questions"] == 6
    assert r["acebench_report"]["agent_conversations"] == 2
    assert r["acebench_report"]["agent_conversations_infra_required"] == 2
    assert reason_part in r["user_simulator"]["infra_required"]
    assert r["user_simulator"]["requests"] == 0
    assert r["status"] == "completed_with_errors"
    assert r["leaderboard_comparable"] is False
    assert "agent_multi_turn (2 conversations) was not run" in r["leaderboard_comparable_reason"]
    assert "agent_multi_turn was not run (infra_required" in r["metric"]


def test_acebench_without_key_runs_the_other_categories(monkeypatch):
    ran = _ace_setup(monkeypatch, keys=[])
    r = _run_ace()
    _assert_agent_multi_turn_infra(r, ran, "needs GEMINI_API_KEY (or GEMINI_API_KEYS)")


def test_acebench_rejected_key_is_reported_the_same_way(monkeypatch):
    ran = _ace_setup(monkeypatch, keys=["k"], live=(False, "HTTP 401"))
    r = _run_ace()
    _assert_agent_multi_turn_infra(r, ran, "was rejected (HTTP 401)")


def test_acebench_with_a_key_runs_agent_multi_turn(monkeypatch):
    ran = _ace_setup(monkeypatch, keys=["k"])
    r = _run_ace()
    assert sorted((it["language"], it["category"]) for it in ran) == [
        ("en", "agent_multi_step"), ("en", "agent_multi_turn"),
        ("zh", "agent_multi_step"), ("zh", "agent_multi_turn")]
    assert "categories_infra_required" not in r and "agent_multi_turn_category" not in r
    assert "infra_required" not in r["user_simulator"]
    assert r["per_sub_category"]["agent_multi_turn"]["accuracy"] == 100.0
    assert r["agent_summary"] == 100.0 and isinstance(r["accuracy"], float)
    assert r["status"] == "success"


def test_acebench_agent_multi_turn_alone_without_key_is_a_hard_error(monkeypatch):
    def only_multi_turn(language="en"):
        return [it for it in _ace_agent_items(language) if it["category"] == "agent_multi_turn"]
    ran = _ace_setup(monkeypatch, keys=[], singles=lambda language="en": [],
                     agents=only_multi_turn)
    with pytest.raises(RuntimeError, match=r"\[INFRA REQUIRED\] acebench cannot run"):
        _run_ace()
    assert ran == []


def test_aggregate_nulls_every_aggregate_that_weighs_an_unavailable_category():
    rows = [("normal_atom_bool", "normal_atom_bool_0", True, 0),
            ("special_irrelevant", "special_irrelevant_0", True, 0)]
    agent = {"agent_multi_step": [(True, 1.0)], "agent_multi_turn": []}
    plain = AB.aggregate(rows, agent)
    held = AB.aggregate(rows, agent, unavailable=["agent_multi_turn"])
    # upstream's reading of a category with no result is unchanged: it counts 0
    assert plain["agent_summary"] == 0.5 and plain["summary"] is not None
    assert "agent_multi_turn" not in plain["per_sub_category"]
    # a category reported infra_required does not count 0
    assert held["agent_summary"] is None and held["summary"] is None
    assert held["leaderboard_columns"]["agent"] is None
    assert held["leaderboard_columns"]["overall"] is None
    assert held["per_sub_category"]["agent_multi_turn"] == {
        "status": "infra_required", "accuracy": None, "correct": 0, "total": 0,
        "process_accuracy": None, "conversations": 0}
    for key in ("normal_summary", "special_summary"):
        assert held[key] == plain[key]
    for col in ("atom", "single", "multi", "api", "pref", "summary", "special"):
        assert held["leaderboard_columns"][col] == plain["leaderboard_columns"][col]


def test_acebench_agent_multi_turn_infra_result_through_reports(tmp_path, monkeypatch):
    _ace_setup(monkeypatch, keys=[])
    r = _run_ace()
    # the printed summary is left out here: it formats accuracy as a number
    out, csv = _through_reports(r, "acebench", tmp_path, monkeypatch, summary=False)
    assert out["status"] == "completed_with_errors" and out["accuracy"] is None
    lines = [ln for ln in csv.splitlines() if "ACEBENCH" in ln]
    overall = next(ln for ln in lines if ",OVERALL," in ln)
    assert ",-," in overall
    mt = next(ln for ln in lines if ",agent_multi_turn," in ln)
    assert ",-," in mt
