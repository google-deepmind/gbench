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

"""Canonical fidelity of the agent / simulator suites.

  * mcp_bench  - Schema Understanding includes Execution Success (paper Table 3).
  * gdpval     - headline is GDPval's win rate: wins + ties.
  * cimemories - the release pipeline (labels, 5 generations per prompt, REVEAL judge,
    Violation@5 / Completeness); full coverage in tests/test_cimemories_release_pipeline.py.
  * toolbench  - SoPR/SoWR per test group, headline = their unweighted Average.
  * tau2/tau3  - upstream agent sampling (temperature 0.0 only), every substitution named in
                 leaderboard_comparable_reason, no partial runs, no retrieval fallback.
                 The pooled headline and gbench's retry patches are named gaps, and
                 per-domain pass^1 is reported as the comparable number.
  * gdpval     - a missing converter or failed download is infra_required; only formats
                 with no rendering are excluded; the paper's 3 samples x 3 graders is named.

No network, no Docker, no model server, no judge API: everything external is stubbed.
"""

import asyncio
import json
import os
import subprocess
import tempfile
from unittest import mock

import pytest

from gbench.runners.eval_suites import cimemories as CIM
from gbench.runners.eval_suites import gdpval as G
from gbench.runners.eval_suites import mcp_bench as M
from gbench.runners.eval_suites import tau_common as TC
from gbench.runners.eval_suites import toolbench as T


# --------------------------------------------------------------------------- #
# mcp_bench (#85)
# --------------------------------------------------------------------------- #
def _table3_row(vtn, sc, es, tf, ig, ta, pa, da, pe):
    """One arXiv:2508.20453 Table 3 row in the upstream results_aggregator units (judge 1-10)."""
    return {"valid_tool_name_rate": vtn, "input_schema_compliance": sc,
            "tool_call_success_rate": es,
            "task_completion_score": (tf + ig) / 2 * 10,
            "tool_selection_score": (ta + pa) / 2 * 10,
            "planning_effectiveness_and_efficiency_score": (da + pe) / 2 * 10}


@pytest.mark.parametrize("row, printed", [
    # llama-3-1-8b-instruct, gpt-4o-mini, claude-sonnet-4 (Table 3)
    ((0.961, 0.894, 0.909, 0.261, 0.295, 0.352, 0.310, 0.221, 0.141), 0.428),
    ((0.975, 0.981, 0.939, 0.374, 0.500, 0.555, 0.544, 0.352, 0.201), 0.557),
    ((1.000, 0.998, 0.988, 0.554, 0.676, 0.689, 0.671, 0.541, 0.328), 0.681),
])
def test_mcp_bench_overall_reproduces_the_paper_table(row, printed):
    r = M.compute_overall_score(_table3_row(*row))
    assert round(r["overall"], 3) == printed


def test_mcp_bench_schema_dimension_is_the_three_rule_based_metrics():
    r = M.compute_overall_score({"valid_tool_name_rate": 0.9, "input_schema_compliance": 0.6,
                                 "tool_call_success_rate": 0.3})
    assert r["dimensions"]["schema_understanding"] == pytest.approx(0.6)


# --------------------------------------------------------------------------- #
# gdpval (#44)
# --------------------------------------------------------------------------- #
_MARK = "__MODEL__"


def _gd_trace(task_id):
    return {"extra_payload": {"task_id": task_id},
            "gold_answer": json.dumps([{"score": 1, "criterion": "c"}]),
            "response_text": f"answer {_MARK}",
            "messages": [{"role": "user", "content": task_id}]}


def _gd_score(traces, verdicts):
    """verdicts[task] = (verdict with the model as A, verdict with the model as B), each one of
    'model' / 'ref' / 'tie'."""
    refs = {t["extra_payload"]["task_id"]: {"text": "expert", "note": None, "n_files": 1}
            for t in traces}
    metrics = {}

    async def fake(prompt):
        task = prompt.split("# TASK\n", 1)[1].split("\n", 1)[0]
        a = prompt.split("# DELIVERABLE A", 1)[1].split("# DELIVERABLE B", 1)[0]
        model_is_a = _MARK in a
        who = verdicts[task][0 if model_is_a else 1]
        if who == "tie":
            w = "tie"
        elif who == "model":
            w = "A" if model_is_a else "B"
        else:
            w = "B" if model_is_a else "A"
        return json.dumps({"winner": w}), "gemini-x"

    with mock.patch.object(G, "judge_generate_cascade", fake):
        asyncio.run(G._make_scorer("j", 4, metrics, refs)(traces))
    return metrics["gdpval_report"]


def test_gdpval_headline_counts_ties_as_gdpval_does():
    traces = [_gd_trace(t) for t in ("w", "t", "l", "split_win", "split_tie", "split_loss")]
    rep = _gd_score(traces, {
        "w": ("model", "model"),          # 1.0  -> win
        "t": ("tie", "tie"),              # 0.5  -> tie
        "l": ("ref", "ref"),              # 0.0  -> loss
        "split_win": ("model", "tie"),    # 0.75 -> win
        "split_tie": ("model", "ref"),    # 0.5  -> tie
        "split_loss": ("tie", "ref"),     # 0.25 -> loss
    })
    assert (rep["wins"], rep["ties"], rep["losses"]) == (2, 2, 2)
    assert rep["win_rate"] == rep["wins_or_ties_rate"] == pytest.approx(4 / 6, abs=1e-4)
    assert rep["wins_only_rate"] == pytest.approx(2 / 6, abs=1e-4)
    assert rep["mean_pairwise_score"] == pytest.approx(3.0 / 6, abs=1e-4)
    assert rep["judge_models_used"] == {"gemini-x": 12}
    outcome = {t["extra_payload"]["task_id"]: t["pairwise_outcome"] for t in traces}
    assert outcome == {"w": "win", "t": "tie", "l": "loss", "split_win": "win",
                       "split_tie": "tie", "split_loss": "loss"}
    # is_correct == "better than or as good as"
    assert [t["is_correct"] for t in traces] == [True, True, False, True, True, False]


def test_gdpval_run_headlines_the_wins_or_ties_rate(monkeypatch):
    rep = {"win_rate": 0.5, "wins_or_ties_rate": 0.5, "wins_only_rate": 0.25,
           "mean_pairwise_score": 0.375, "tasks_in_win_rate": 4,
           "tasks_no_reference": 0, "tasks_judge_outage": 0}

    def fake_scorer(judge_model, conc, metrics, references):
        metrics["gdpval_report"] = rep
        return None

    monkeypatch.setattr(G, "gemini_required_skip", lambda *a: None)
    monkeypatch.setattr(G, "_load_gdpval_samples", lambda limit=None: ([], {}))
    monkeypatch.setattr(G, "_make_scorer", fake_scorer)
    monkeypatch.setattr(G, "run_eval_suite", lambda **kw: {"accuracy": 12.0, "total_questions": 4})
    r = G.run_gdpval("m", "http://x", 1)
    assert r["accuracy"] == 50.0 and r["win_rate_pct"] == 50.0
    assert r["wins_only_pct"] == 25.0 and r["mean_pairwise_score_pct"] == 37.5
    assert r["leaderboard_comparable"] is False
    assert "wins-or-ties" in r["leaderboard_comparable_reason"]
    assert r["judge"] == "gbench-gemini-cascade"


def test_gdpval_loads_the_pinned_revision():
    seen = {}

    def fake_load(repo, split=None, revision=None):
        seen["revision"] = revision
        return []

    with mock.patch("datasets.load_dataset", fake_load):
        with pytest.raises(RuntimeError):
            G._load_gdpval_samples()
    assert seen["revision"] == G._REVISION and len(G._REVISION) == 40


# --------------------------------------------------------------------------- #
# cimemories (#06, #69)
# --------------------------------------------------------------------------- #
#: Stand-ins for the release's prompt files, which the suite reads at run time.
_CIM_RELEASE = CIM.parse_release_texts(
    b"task_solving:\n  1: |\n    # User Knowledge Memories:\n\n    {memories}\n\n"
    b"    Write to {recipient}: {task}.\n"
    b"adversary_reveal:\n  1: |\n    ATTRIBUTES:\n    {attributes}\n\n    SOLUTION:\n"
    b"    {solution}\n",
    b'personas = ["\\n    **The Privacy Fundamentalist**\\n    ", '
    b'"\\n    **The Pragmatic**\\n    ", "\\n    *The Unconcerned**\\n    "]\n'
    b"def label_attributes(model, persona, profile, context, n=3):\n"
    b'    attribute_sorting_prompt = f"""\n    You have the following persona:\n    {persona}\n'
    b"    To {context['recipient']}: {context['task']}.\n    INFORMATION:\n    {{memories}}\n"
    b'    """\n'
    b"labels = label_attributes(model, persona, profile, context, n=10)\n"
    b"weights = {'fundamentalist': 0.26, 'pragmatic': 0.64, 'unconcerned': 0.10}\n")


def _cim_profiles():
    profs = []
    for name in ("Ann", "Bob"):
        attrs = {f"a{i}": {"value": i, "memory_statement": f"{name} fact {i}."} for i in range(3)}
        profs.append({"bio": {"name": name}, "information_attributes": attrs,
                      "contexts": [{"recipient": "Doctor", "task": f"task {t}"}
                                   for t in range(3)]})
    return tuple(profs)


def test_cimemories_generates_each_prompt_once(monkeypatch):
    monkeypatch.setattr(CIM, "_load_profiles", _cim_profiles)
    monkeypatch.setattr(CIM, "_release_texts", lambda: _CIM_RELEASE)
    samples = CIM._load_cimemories_samples()
    assert len(samples) == 6                               # one sample per (profile, context)
    assert {s[2]["category"] for s in samples} == {"Ann", "Bob"}
    assert all(set(s[2]) == {"category"} for s in samples)  # nothing else leaks into the request
    assert samples[0][0][0]["content"].startswith("# User Knowledge Memories:")   # release prompt
    sub = CIM._load_cimemories_samples(limit=2)
    assert sorted(s[2]["category"] for s in sub) == ["Ann", "Bob"]


def _cim_judge(monkeypatch, tmp_path, reply):
    """Ann's task 0 labeled a0 = share and a1 = private (every persona sample agrees)."""
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))

    async def fake(prompt, config=None):
        if prompt.startswith("You have the following persona"):
            return '{"share": ["Ann fact 0."], "private": ["Ann fact 1."]}', "gemini-x"
        return reply, "gemini-x"

    monkeypatch.setattr(CIM, "judge_generate_cascade", fake)
    monkeypatch.setattr(CIM, "_release_texts", lambda: _CIM_RELEASE)
    tr = {"response_text": "Dear doctor, ...", "gold_answer": "p0c0", "attempt_idx": 0}
    asyncio.run(CIM._async_judge_cimemories([tr], profiles=_cim_profiles()))
    return tr


def test_cimemories_appropriate_sharing_passes(monkeypatch, tmp_path):
    tr = _cim_judge(monkeypatch, tmp_path, '{"a0": "Ann fact 0."}')
    assert tr["is_correct"] is True and tr["revealed_share"] == ["a0"]


def test_cimemories_inappropriate_disclosure_fails(monkeypatch, tmp_path):
    tr = _cim_judge(monkeypatch, tmp_path, '{"a1": "Ann fact 1."}')
    assert tr["is_correct"] is False and tr["revealed_private"] == ["a1"]


def test_cimemories_unfinished_purpose_fails(monkeypatch, tmp_path):
    # Withholding the must-share attribute is no violation; it lowers Completeness instead.
    tr = _cim_judge(monkeypatch, tmp_path, "{}")
    assert tr["revealed_share"] == [] and tr["judge_grade"] == "NO_VIOLATION"


def test_cimemories_unparseable_verdict_is_a_judge_failure(monkeypatch, tmp_path):
    tr = _cim_judge(monkeypatch, tmp_path, "GRADE: CORRECT")
    # left out of both numbers (the substitute judge's failure), never "revealed nothing"
    assert tr["judge_grade"] == "JUDGE_FAILED" and "revealed_attributes" not in tr
    assert tr["scoring_excluded"] is True and tr["judge_raw"] == "GRADE: CORRECT"


def test_cimemories_judge_has_no_offline_fallback(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        asyncio.run(CIM._async_judge_cimemories([{"response_text": "x"}]))
    assert not hasattr(CIM, "_eval_cimemories")


def test_cimemories_result_says_it_is_not_the_papers_metric(monkeypatch):
    # It is the paper's metric now, but the labels and reveals come from the Gemini cascade.
    monkeypatch.setattr(CIM, "gemini_required_skip", lambda *a: None)
    monkeypatch.setattr(CIM, "_load_profiles", _cim_profiles)
    monkeypatch.setattr(CIM, "_load_cimemories_samples", lambda limit=None: [])
    monkeypatch.setattr(CIM, "run_eval_suite", lambda **kw: {"accuracy": 0.0,
                                                              "sample_traces": []})
    r = CIM.run_cimemories("m", "http://x")
    assert "violation_at_5" in r and "completeness" in r
    assert r["leaderboard_comparable"] is False
    reason = r["leaderboard_comparable_reason"]
    assert "instead of GPT-5" in reason and "instead of DeepSeek-R1-0528" in reason
    assert "withhold scenarios" not in r["metric"]


def test_cimemories_docs_list_the_judge_key():
    doc = open(os.path.join(os.path.dirname(__file__), "..", "docs", "evals", "cimemories.md"),
               encoding="utf-8").read()
    assert "GEMINI_API_KEY" in doc and "None beyond" not in doc
    assert "7a2bde79f747d011b23095e6bb035c6f64127d9f" in doc


# --------------------------------------------------------------------------- #
# toolbench (#65)
# --------------------------------------------------------------------------- #
def _tb_answer(solved):
    return {"query": "q", "answer": {"final_answer": "SOLVED" if solved else "NOPE",
                                     "answer_details": []}}


def _patch_status(monkeypatch):
    async def status(query, answer, sem):
        return "Solved" if answer.get("final_answer") == "SOLVED" else "Unsolved"
    monkeypatch.setattr(T, "_answer_status", status)
    # the preference comparison computes its own statuses (eval_preference.py)
    monkeypatch.setattr(T, "_preference_status", status)


def test_toolbench_sopr_is_the_unweighted_mean_of_the_groups(monkeypatch):
    _patch_status(monkeypatch)
    cand = {
        "G1_instruction": {str(i): _tb_answer(True) for i in range(9)},   # 9/9 -> 1.0
        "G3_instruction": {"1": _tb_answer(False)},                        # 0/1 -> 0.0
    }
    s = asyncio.run(T._score_groups(cand, {}, 1, 4))
    assert s["by_group"]["G1_instruction"]["sopr"] == 1.0
    assert s["by_group"]["G3_instruction"]["sopr"] == 0.0
    assert s["sopr"] == pytest.approx(0.5)          # Average, not the pooled 9/10
    assert s["sowr"] is None and s["n_candidate"] == 10


def test_toolbench_groups_are_not_merged_by_qid(tmp_path):
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "G1_tool.json").write_text(json.dumps({"7": _tb_answer(True)}))
    (root / "G2_category.json").write_text(json.dumps({"7": _tb_answer(False)}))
    groups = T._load_groups(str(root), ["G1_tool", "G2_category", "G3_instruction"])
    assert groups["G1_tool"]["7"]["answer"]["final_answer"] == "SOLVED"
    assert groups["G2_category"]["7"]["answer"]["final_answer"] == "NOPE"
    assert groups["G3_instruction"] == {}


def _run_toolbench(monkeypatch, groups_written, reference=None, groups_env=None):
    """Drive run_toolbench with docker stubbed: the fake `docker run` writes the converted
    answers into the mounted /out the way docker/toolbench_entrypoint.sh does."""
    _patch_status(monkeypatch)
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
    if groups_env:
        monkeypatch.setenv("GBENCH_TOOLBENCH_GROUPS", groups_env)
    else:
        monkeypatch.delenv("GBENCH_TOOLBENCH_GROUPS", raising=False)
    monkeypatch.delenv("GBENCH_TOOLBENCH_REFERENCE_DIR", raising=False)
    monkeypatch.delenv("GBENCH_TOOLBENCH_REFERENCE", raising=False)
    # The published reference answers (converted) and the image's test ids, stubbed. SoWR needs a
    # reference answer for every test id, so by default each group gets an unsolved one per id.
    if reference is None:
        run_groups = (groups_env or " ".join(T._GROUPS)).split()
        reference = {g: {q: _tb_answer(False) for q in (groups_written.get(g) or {"1": None})}
                     for g in run_groups}
    monkeypatch.setattr(T, "_group_test_ids", lambda image, groups, limit: {
        g: sorted(set(groups_written.get(g, {})) | set((reference or {}).get(g, {})))
        for g in groups})

    def fake_run(cmd, *a, **k):
        if cmd[:2] == ["docker", "run"]:
            out = cmd[cmd.index("-v") + 1].split(":")[0]
            os.makedirs(os.path.join(out, "candidate"), exist_ok=True)
            for g, answers in groups_written.items():
                with open(os.path.join(out, "candidate", f"{g}.json"), "w") as fh:
                    json.dump(answers, fh)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(T.subprocess, "run", fake_run)
    with tempfile.TemporaryDirectory(prefix="gbench_test_tbref_") as ref_root:
        for g, answers in reference.items():
            d = os.path.join(ref_root, T._REFERENCE_DEFAULT)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"{g}.json"), "w") as fh:
                json.dump(answers, fh)
        monkeypatch.setattr(T, "_baseline_reference_root", lambda ref, groups, image: ref_root)
        return T.run_toolbench("m", "http://x:8000/v1", 2)


def test_toolbench_run_headlines_the_group_average(monkeypatch):
    written = {g: {"1": _tb_answer(i % 2 == 0), "2": _tb_answer(i % 2 == 0)}
               for i, g in enumerate(T._GROUPS)}
    written["G3_instruction"] = {"1": _tb_answer(True)}    # smallest group, all solved
    r = _run_toolbench(monkeypatch, written)
    per = r["sopr_by_group"]
    assert set(per) == set(T._GROUPS)
    assert r["accuracy"] == pytest.approx(sum(per.values()) / len(per), abs=0.01)
    assert r["queries_by_group"]["G3_instruction"] == 1
    assert "unweighted" in r["aggregation"]
    assert "Average" in r["leaderboard_comparable_reason"]
    assert "subset" not in r["leaderboard_comparable_reason"]


def test_toolbench_missing_group_is_a_hard_error_not_a_partial_average(monkeypatch):
    written = {g: {"1": _tb_answer(True)} for g in T._GROUPS if g != "G2_category"}
    with pytest.raises(RuntimeError, match="G2_category"):
        _run_toolbench(monkeypatch, written)


def test_toolbench_group_subset_is_named_in_the_reason(monkeypatch):
    written = {"G1_tool": {"1": _tb_answer(True)}}
    r = _run_toolbench(monkeypatch, written, groups_env="G1_tool")
    assert "subset" in r["leaderboard_comparable_reason"]


def test_toolbench_sowr_reads_only_the_named_reference_model(monkeypatch):
    written = {g: {"1": _tb_answer(True)} for g in T._GROUPS}
    reference = {g: {"1": _tb_answer(False)} for g in T._GROUPS}
    r = _run_toolbench(monkeypatch, written, reference=reference)
    # candidate Solved vs reference Unsolved -> the one Solved answer wins every query
    assert r["sowr"] == 100.0 and set(r["sowr_by_group"].values()) == {100.0}


# --------------------------------------------------------------------------- #
# tau2 / tau3 (#64)
# --------------------------------------------------------------------------- #
def _clear_tau_env(monkeypatch):
    for k in ("GBENCH_TAU2_TEMPERATURE", "TAU2_TEMPERATURE", "GBENCH_TAU3_TEMPERATURE",
              "GBENCH_TAU2_TOP_P", "TAU2_TOP_P", "GBENCH_TAU2_TOP_K", "TAU2_TOP_K",
              "GBENCH_TAU2_USER_LLM", "TAU2_USER_LLM", "GBENCH_TAU2_EVAL_LLM", "TAU2_EVAL_LLM",
              "GBENCH_TAU2_NUM_TRIALS", "TAU2_NUM_TRIALS", "GBENCH_TAU2_USER_TEMPERATURE",
              "TAU2_USER_TEMPERATURE"):
        monkeypatch.delenv(k, raising=False)


def test_tau_agent_defaults_to_upstream_sampling(monkeypatch):
    _clear_tau_env(monkeypatch)
    args = TC._build_agent_args("http://x/v1", enable_thinking=False, eval_name="tau2")
    assert args["temperature"] == 0.0
    assert "top_p" not in args and "extra_body" not in args
    # --thinking: upstream's 0.0 does not apply to a reasoning run; the agent samples at 1.0
    think = TC._build_agent_args("http://x/v1", enable_thinking=True, eval_name="tau3")
    assert think["temperature"] == 1.0
    assert think["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}


def test_tau_top_p_top_k_and_temperature_are_opt_in(monkeypatch):
    _clear_tau_env(monkeypatch)
    monkeypatch.setenv("GBENCH_TAU2_TOP_P", "0.95")
    monkeypatch.setenv("GBENCH_TAU2_TOP_K", "64")
    monkeypatch.setenv("GBENCH_TAU2_TEMPERATURE", "1.0")
    args = TC._build_agent_args("http://x/v1", enable_thinking=False, eval_name="tau2")
    assert args["temperature"] == 1.0 and args["top_p"] == 0.95
    assert args["extra_body"] == {"top_k": 64}


def _run_env(monkeypatch, domains, canned, limit=None):
    monkeypatch.setattr(TC, "check_tau_env_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(TC, "check_banking_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(TC, "_install_robustness_patches", lambda *a, **k: None)
    monkeypatch.setattr(TC, "_configure_evaluator_llm", lambda *a, **k: None)
    monkeypatch.setattr(TC, "_quiet_tau2_noise", lambda: None)
    # A limited or sharded run lists the domain's tasks through tau2, which the CI install does
    # not have (it runs in its own image): a fixed task list stands in for it.
    ids = [f"task-{i}" for i in range(10)]
    monkeypatch.setattr(TC, "_shard_task_ids",
                        lambda domain, shard, limit, seed="tau2": (ids[:limit] if limit else ids,
                                                                   len(ids)))

    def one(domain, *a, **k):
        TC._build_agent_args("http://x/v1", False, a[-1] if a else "tau2")
        if domain == "banking_knowledge":
            TC._LAST_RETRIEVAL.update({"retrieval_config": "alltools",
                                       "retrieval_variant": "alltools-gemini",
                                       "embedders": ["gemini:gemini-embedding-001"],
                                       "embed_model": "gemini:gemini-embedding-001"})
        TC._DOMAIN_PASS_HAT_K[domain] = {"pass^1": 0.5}
        out = canned[domain]
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(TC, "_run_one_domain", one)
    eval_name = "tau3" if domains == ["banking_knowledge"] else "tau2"
    return TC.run_tau_env(eval_name, domains, "m", "http://x", 4, limit, "docs/evals/tau2.md")


def test_tau_reason_names_every_substitution(monkeypatch):
    _clear_tau_env(monkeypatch)
    monkeypatch.setenv("GBENCH_TAU2_NUM_TRIALS", "1")   # the default is now the leaderboard's 4
    r = _run_env(monkeypatch, ["banking_knowledge"], {"banking_knowledge": (4, 2.0, 2, 0)})
    reason = r["leaderboard_comparable_reason"]
    assert r["leaderboard_comparable"] is False
    assert "1 trial(s)" in reason and "4+" in reason
    assert "user simulator gemini/" in reason and "gpt-4.1-2025-04-14" in reason
    assert "NL-assertion judge" in reason
    assert "retrieval config" not in reason          # upstream's alltools is the config
    assert "retrieval embeddings by gemini-embedding-001 (upstream: text-embedding-3-large)" \
        in reason                                   # its dense embedder is the substitution
    assert "agent sampled" not in reason            # default sampling is upstream's
    rep = r["tau2_report"]
    assert rep["pass_hat_k"] == {"banking_knowledge": {"pass^1": 0.5}}
    assert rep["agent_sampling"] == {"temperature": 0.0, "top_p": None, "top_k": None}
    assert rep["retrieval_config"] == "alltools"
    assert rep["retrieval_variant"] == "alltools-gemini"
    assert rep["embed_model"] == "gemini:gemini-embedding-001"
    assert "tau2_version" in rep


def test_tau_reason_names_nondefault_sampling_and_limit(monkeypatch):
    _clear_tau_env(monkeypatch)
    monkeypatch.setenv("GBENCH_TAU2_TEMPERATURE", "1.0")
    r = _run_env(monkeypatch, ["airline"], {"airline": (2, 1.0, 1, 1)}, limit=2)
    reason = r["leaderboard_comparable_reason"]
    assert "agent sampled at temperature 1.0" in reason
    assert "--eval-limit" in reason
    assert "infrastructure errors" in reason


def test_tau_matching_setup_can_be_leaderboard_comparable(monkeypatch):
    _clear_tau_env(monkeypatch)
    monkeypatch.setenv("GBENCH_TAU2_USER_LLM", "gpt-4.1-2025-04-14")
    monkeypatch.setenv("GBENCH_TAU2_EVAL_LLM", "gpt-4.1-2025-04-14")
    monkeypatch.setenv("GBENCH_TAU2_NUM_TRIALS", "4")
    # one domain (no pooled headline) and gbench's retry patches not active
    monkeypatch.setattr(TC, "_RETRY_PATCHES", {"completion": False, "generate": False})
    r = _run_env(monkeypatch, ["retail"], {"retail": (8, 4.0, 4, 0)})
    assert r["leaderboard_comparable"] is True and r["leaderboard_comparable_reason"] is None


def test_tau_failed_domain_is_a_hard_error_not_a_partial_score(monkeypatch):
    _clear_tau_env(monkeypatch)
    with pytest.raises(RuntimeError, match="retail"):
        _run_env(monkeypatch, ["airline", "retail"],
                 {"airline": (2, 1.0, 1, 0), "retail": RuntimeError("simulator boom")})
    with pytest.raises(RuntimeError, match="telecom"):
        _run_env(monkeypatch, ["airline", "telecom"],
                 {"airline": (2, 1.0, 1, 0), "telecom": (0, 0.0, 0, 0)})


def test_the_removed_retrieval_knob_is_ignored_by_the_prerequisite(monkeypatch):
    # GBENCH_TAU2_RETRIEVAL_CONFIG no longer selects anything: the one config (alltools with
    # Gemini embeddings) is checked, and only the Gemini key decides.
    pytest.importorskip("tau2.domains.banking_knowledge.retrieval")
    monkeypatch.delenv("TAU2_RETRIEVAL_CONFIG", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setenv("GBENCH_TAU2_RETRIEVAL_CONFIG", "alltools-gemini")
    monkeypatch.setenv("OPENAI_API_KEY", "already-set")
    ok, why = TC.check_retrieval_credentials()
    assert not ok and "GEMINI_API_KEY" in why and "alltools" in why
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-for-the-check")
    assert TC.check_retrieval_credentials() == (True, "")


def test_tau_docs_name_upstreams_embedder_and_the_v1_0_1_data_commit():
    here = os.path.dirname(__file__)
    tau3 = open(os.path.join(here, "..", "docs", "evals", "tau3.md"), encoding="utf-8").read()
    tau2 = open(os.path.join(here, "..", "docs", "evals", "tau2.md"), encoding="utf-8").read()
    assert "text-embedding-3-large" in tau3 and "gemini-embedding-001" in tau3
    assert "reference tau2 config" not in tau2 and "1.0/0.95" not in tau2
    assert "fc0055dc4e0a316c3f83133267fbd6faaa770992" in tau2


# --------------------------------------------------------------------------- #
# tau2 / tau3 comparability is honest about the headline and the retries
# --------------------------------------------------------------------------- #
def test_tau_three_domain_run_names_the_pooled_headline_and_reports_pass1(monkeypatch):
    """With the leaderboard's own user simulator, judge and 4 trials, a three-domain run's
    headline is the leaderboard's: the unweighted mean of the per-domain pass^1 (the tau2-bench
    track Overall), not the mean reward pooled over the simulations (50/114/114 tasks)."""
    _clear_tau_env(monkeypatch)
    monkeypatch.setenv("GBENCH_TAU2_USER_LLM", "openai/gpt-4.1-2025-04-14")
    monkeypatch.setenv("GBENCH_TAU2_EVAL_LLM", "openai/gpt-4.1-2025-04-14")
    monkeypatch.setenv("GBENCH_TAU2_NUM_TRIALS", "4")
    monkeypatch.setattr(TC, "_RETRY_PATCHES", {"completion": False, "generate": False})
    pass1 = {"airline": 0.40, "retail": 0.70, "telecom": 0.60}
    counts = {"airline": 200, "retail": 456, "telecom": 456}
    canned = {d: (counts[d], pass1[d] * counts[d], int(pass1[d] * counts[d]), 0) for d in pass1}
    monkeypatch.setattr(TC, "check_tau_env_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(TC, "_install_robustness_patches", lambda *a, **k: None)
    monkeypatch.setattr(TC, "_configure_evaluator_llm", lambda *a, **k: None)
    monkeypatch.setattr(TC, "_quiet_tau2_noise", lambda: None)

    def one(domain, *a, **k):
        TC._build_agent_args("http://x/v1", False, "tau2")
        TC._DOMAIN_PASS_HAT_K[domain] = {"pass^1": pass1[domain], "pass^4": pass1[domain] / 2}
        return canned[domain]

    monkeypatch.setattr(TC, "_run_one_domain", one)
    r = TC.run_tau_env("tau2", ["airline", "retail", "telecom"], "m", "http://x", 4, None,
                       "docs/evals/tau2.md")
    assert r["accuracy"] == 56.67                      # the tau2-bench track Overall
    assert r["tau2_report"]["mean_reward"] == 0.605    # pooled, weighted by task count
    assert r["pass1_by_domain"] == {"airline": 40.0, "retail": 70.0, "telecom": 60.0}
    assert r["pass1_domain_mean"] == 56.67
    assert r["leaderboard_comparable"] is True and r["leaderboard_comparable_reason"] is None


def test_tau_active_retry_patches_are_a_named_gap(monkeypatch):
    # The retry patches now act only on gemini/ user-simulator and judge calls, never on the
    # agent, so with upstream's user simulator and judge they are no comparability gap.
    _clear_tau_env(monkeypatch)
    monkeypatch.setenv("GBENCH_TAU2_USER_LLM", "gpt-4.1-2025-04-14")
    monkeypatch.setenv("GBENCH_TAU2_EVAL_LLM", "gpt-4.1-2025-04-14")
    monkeypatch.setenv("GBENCH_TAU2_NUM_TRIALS", "4")
    TC._build_agent_args("http://x/v1", False, "tau2")
    for patches in ({"completion": True, "generate": False},
                    {"completion": False, "generate": True},
                    {"completion": False, "generate": False}):
        monkeypatch.setattr(TC, "_RETRY_PATCHES", patches)
        assert TC._comparability_gaps(["retail"], 4, None, 0) == []


def test_tau_docs_do_not_overclaim_comparability():
    here = os.path.dirname(__file__)
    tau2 = open(os.path.join(here, "..", "docs", "evals", "tau2.md"), encoding="utf-8").read()
    tau3 = open(os.path.join(here, "..", "docs", "evals", "tau3.md"), encoding="utf-8").read()
    assert "lists every difference found" not in tau2
    assert "pass1_by_domain" in tau2 and "pass1_domain_mean" in tau2 and "pooled" in tau2
    assert "robustness patches" in tau2 and "pass1_by_domain" in tau3
    assert "--max-output-tokens" in tau2 and "--max-output-tokens" in tau3
    import inspect
    assert "matching the reference" not in inspect.getsource(TC)


# --------------------------------------------------------------------------- #
# gdpval converters, downloads and exclusions
# --------------------------------------------------------------------------- #
def test_gdpval_missing_converter_is_infra_required_naming_the_package(monkeypatch):
    monkeypatch.setitem(G._CONVERTER_PACKAGES, "pptx", ("gbench_no_such_module", "python-pptx"))

    def no_download(**kw):
        raise AssertionError("the package check must come before the download")

    with mock.patch("huggingface_hub.hf_hub_download", no_download):
        with pytest.raises(RuntimeError, match="INFRA REQUIRED.*python-pptx"):
            G.render_attachment("deliverable_files/a/Deck.pptx")
        with pytest.raises(RuntimeError, match="INFRA REQUIRED.*python-pptx"):
            G.render_reference(["deliverable_files/a/Deck.pptx"])
        with pytest.raises(RuntimeError, match="INFRA REQUIRED.*python-pptx"):
            G.attachment_parts(["reference_files/a/Deck.pptx"])


def test_gdpval_failed_download_is_infra_required_everywhere():
    with mock.patch("huggingface_hub.hf_hub_download", side_effect=OSError("offline")):
        for call in (lambda: G.render_reference(["deliverable_files/a/Out.docx"]),
                     lambda: G.attachment_parts(["reference_files/a/In.pdf"]),
                     lambda: G.attachment_parts(["reference_files/a/pic.png"])):
            with pytest.raises(RuntimeError, match="INFRA REQUIRED.*could not download"):
                call()


def test_gdpval_unrenderable_formats_get_a_note_without_a_download():
    with mock.patch("huggingface_hub.hf_hub_download",
                    side_effect=AssertionError("must not fetch an unrenderable file")):
        text, note = G.render_reference(["deliverable_files/a/Reel.mp4"])
        parts, rendered, unavailable = G.attachment_parts(["reference_files/a/part.step"])
    assert text is None and ".mp4 file" in note
    assert (rendered, unavailable) == (0, 1) and "cannot render" in parts[0]["text"]


def test_gdpval_code_and_notebook_deliverables_render_as_text(tmp_path):
    files = {"HeatConduction.py": "import numpy as np\nprint(1)\n",
             "api.yaml": "openapi: 3.0.0\n",
             "abq okc query.overpassql": "[out:json];node(1);out;\n",
             "Pricing.ipynb": json.dumps({"cells": [
                 {"cell_type": "markdown", "source": ["# American option"]},
                 {"cell_type": "code", "source": "price = 4.2",
                  "outputs": [{"output_type": "stream", "text": ["4.2\n"]},
                              {"output_type": "display_data", "data": {"image/png": "AAAA"}}]}]})}
    for name, body in files.items():
        (tmp_path / name).write_text(body, encoding="utf-8")
    for name in files:
        text, note = G.render_local_file(str(tmp_path / name), name)
        assert text and note is None, name
    nb, _ = G.render_local_file(str(tmp_path / "Pricing.ipynb"), "Pricing.ipynb")
    assert "# American option" in nb and "price = 4.2" in nb and "4.2" in nb
    assert "1 image(s) not rendered" in nb and "AAAA" not in nb


def _pdf_without_text(path):
    import pymupdf   # an evals-extra dependency (gdpval and gaia render PDFs with it)
    doc = pymupdf.open()
    page = doc.new_page()
    page.draw_rect(pymupdf.Rect(50, 50, 300, 200), color=(0, 0, 0), fill=(0.2, 0.4, 0.8))
    doc.save(str(path))
    return str(path)


def test_gdpval_pdf_without_a_text_layer(tmp_path):
    local = _pdf_without_text(tmp_path / "Safety Circuit.pdf")
    text, note = G.render_local_file(local, "Safety Circuit.pdf")
    assert text is None and "no text layer" in note
    with mock.patch.object(G, "_fetch", return_value=local):
        ref_text, ref_note = G.render_reference(["deliverable_files/a/Safety Circuit.pdf"])
        parts, rendered, unavailable = G.attachment_parts(["reference_files/a/Safety Circuit.pdf"])
    assert ref_text is None and "no text layer" in ref_note
    assert (rendered, unavailable) == (1, 0), "an input scan goes to the model as page images"
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_gdpval_images_keep_their_own_mime_type(tmp_path):
    p = tmp_path / "chart.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    with mock.patch.object(G, "_fetch", return_value=str(p)):
        parts, rendered, _ = G.attachment_parts(["reference_files/a/chart.jpg"])
    assert rendered == 1 and parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_gdpval_no_reference_reasons_are_reported():
    traces = [_gd_trace(t) for t in ("none", "video", "ok")]
    refs = {"none": {"text": None, "note": "[no reference deliverable shipped for this task]",
                     "n_files": 0},
            "video": {"text": None, "note": "[... .mp4 ...]", "n_files": 1},
            "ok": {"text": "expert", "note": None, "n_files": 1}}
    metrics = {}

    async def fake(prompt):
        return json.dumps({"winner": "tie"}), "gemini-x"

    with mock.patch.object(G, "judge_generate_cascade", fake):
        asyncio.run(G._make_scorer("j", 2, metrics, refs)(traces))
    rep = metrics["gdpval_report"]
    assert rep["tasks_no_reference"] == 2 and rep["tasks_in_win_rate"] == 1
    assert rep["tasks_no_reference_by_reason"] == {"no_deliverable_shipped": 1,
                                                   "no_text_rendering": 1}


def test_gdpval_reason_names_the_papers_three_samples_three_graders(monkeypatch):
    monkeypatch.setattr(G, "gemini_required_skip", lambda *a: None)
    monkeypatch.setattr(G, "_load_gdpval_samples", lambda limit=None: ([], {}))
    monkeypatch.setattr(G, "run_eval_suite", lambda **kw: {"total_questions": 0})
    r = G.run_gdpval("m", "http://x", 1)
    reason = r["leaderboard_comparable_reason"]
    # gbench now samples 3 times per task, as the paper does; only the 3 graders per sample
    # remain a difference
    assert "3 generations per task" in reason and "each judged once" in reason
    assert "3 human experts" in reason and "footnote 2" in reason
    doc = open(os.path.join(os.path.dirname(__file__), "..", "docs", "evals", "gdpval.md"),
               encoding="utf-8").read()
    assert "footnote 2" in doc and "**3 times**" in doc and "--max-output-tokens" in doc
    assert "Each task ships an **expert reference deliverable**" not in doc
