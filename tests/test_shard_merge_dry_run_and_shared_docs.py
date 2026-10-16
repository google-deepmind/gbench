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

"""Dry runs, shard naming and merges, and the shared docs and provenance, checked against the code:

* a dry run creates no results directory; --shard names the non-eval pillars it does not split;
* the merge of livebench leaves an empty shard out of the headline and recomputes the reasons a
  shard's part of the questions causes; toolbench's merge records every shard's Gemini models,
  harness errors and counts; mmlu and mmlu_redux keep identical rows apart in the merge; a
  missing OmniDocBench annotation file is a MergeError;
* the shared docs say what the code does (server bind address and endpoint key, the key tables,
  mcp_atlas's page tools on the release's servers, the Golden Set's derived leak tokens);
* the provenance records the decisions (synced bumps, verdict-preserving declarations, label
  literals equal to the Dockerfiles).

No test contacts a network, a model server or Docker.
"""

import json
import re
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _flat(text: str) -> str:
    return " ".join(text.split())


# The README was split: these docs hold sections moved verbatim out of the README, so a
# negative check that used to scan the whole README scans README.md plus all of them.
_README_SPLIT_DOCS = (
    "docs/installation.md", "docs/environment-variables.md", "docs/performance.md",
    "docs/eval-suites.md", "docs/golden-set.md", "docs/remote-endpoints.md",
    "docs/results.md", "docs/deployment.md", "docs/cli-reference.md",
)
_README_AND_SPLIT_DOCS = ("README.md", *_README_SPLIT_DOCS)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main(monkeypatch, argv):
    from gbench import cli
    # The CLI exports GBENCH_SHARD for the suites: setenv first records the variable's state,
    # so monkeypatch restores it after the test (delenv alone records nothing when it is unset).
    monkeypatch.setenv("GBENCH_SHARD", "")
    monkeypatch.delenv("GBENCH_SHARD")
    monkeypatch.setattr(sys, "argv", ["gbench", *argv])
    try:
        return cli.main()
    except SystemExit as e:
        return e.code


def test_a_dry_run_creates_no_results_directory(monkeypatch, tmp_path, caplog):
    target = tmp_path / "results"
    caplog.set_level("INFO")
    code = _main(monkeypatch, ["--models", "org/m", "--evals", "gsm8k", "--remote-endpoint",
                               "http://127.0.0.1:9/v1", "--tokenizer", "org/m",
                               "--max-output-tokens", "1024", "--dry-run",
                               "--results-dir", str(target)])
    assert code == 0
    assert not target.exists()
    assert "(dry run: not created)" in caplog.text


def test_the_log_manager_resolves_without_creating(tmp_path):
    from gbench.utils import LogManager
    lm = LogManager(tmp_path / "r", create_dirs=False)
    assert lm.results_dir.parent == tmp_path / "r" and not (tmp_path / "r").exists()


def test_a_shard_names_the_pillars_it_does_not_split(monkeypatch, tmp_path, caplog):
    caplog.set_level("WARNING")
    code = _main(monkeypatch, ["--models", "org/m", "--evals", "gsm8k", "--golden",
                               "--shard", "1/2", "--remote-endpoint", "http://127.0.0.1:9/v1",
                               "--tokenizer", "org/m", "--max-output-tokens", "1024",
                               "--dry-run", "--results-dir", str(tmp_path / "r")])
    assert code == 0
    assert ("--shard 1/2 splits the eval suites only: golden run in full on this node and "
            "record shard_applied=false") in caplog.text
    src = _read("gbench/cli.py")
    assert "result['shard_applied'] = False" in src and '"shard_not_applied"' in src


def test_the_host_owner_text_follows_the_bind_choice():
    from gbench.core import config as C
    text = C.GBENCH_OWNED_SERVE_ARGS["--host"]
    assert "a run that selects no eval suite, bind 127.0.0.1" in text
    assert "with eval suites selected the server binds all interfaces" in text


def test_the_provenance_table_fits_a_same_day_serial():
    src = _read("gbench/cli.py")
    assert "{'synced':<13}" in src and "{s:<13}" in src and "{'synced':<11}" not in src


# --------------------------------------------------------------------------- #
# merges
# --------------------------------------------------------------------------- #
def _lb_trace(key, task, score=1.0, category="instruction_following"):
    return {"sample_key": key, "question_id": key, "task": task, "category": category,
            "score": score}


def _lb_shard(i, n, traces, reasons=(), **extra):
    from gbench.runners.eval_suites.base import SHARD_REASON
    shard_reason = SHARD_REASON.format(label=f"{i}/{n}", index=i, count=n)
    res = {"eval_name": "livebench", "shard": f"{i}/{n}", "status": "success",
           "accuracy": 50.0, "sample_traces": traces, "selected_question_ids": [
               t["question_id"] for t in traces], "server_aborted_requests": 0,
           "leaderboard_comparable": False,
           "leaderboard_comparable_reason": "; ".join([shard_reason, *reasons])}
    res.update(extra)
    return res


def test_livebench_an_empty_shard_does_not_withhold_the_merged_score():
    from gbench.runners.eval_suites import livebench as LB
    scored = _lb_shard(1, 2, [_lb_trace("a", "t1"), _lb_trace("b", "t2", 0.0)])
    empty = {"eval_name": "livebench", "shard": "2/2", "status": "success", "accuracy": None,
             "accuracy_withheld_reason": "empty shard 2/2: no items on this node (2 in the full "
             "set)", "empty_shard": True, "sample_traces": [], "selected_question_ids": [],
             "leaderboard_comparable": False,
             "leaderboard_comparable_reason": "empty shard 2/2: no items on this node"}
    merged = LB.merge_shards([scored, empty])
    assert merged["accuracy"] == 50.0 and "accuracy_withheld_reason" not in merged
    assert merged["server_aborted_requests"] == 0
    assert "empty_shard" not in merged or not merged["empty_shard"]
    assert merged["leaderboard_comparable"] is True
    # a withheld scored shard still withholds
    failed = _lb_shard(1, 2, [_lb_trace("a", "t1")], accuracy=None,
                       accuracy_withheld_reason="1 request(s) failed")
    assert LB.merge_shards([failed, empty])["accuracy"] is None


def test_livebench_shard_only_reasons_are_recomputed_over_the_union():
    from gbench.runners.eval_suites import livebench as LB
    unequal = "instruction_following tasks answered in unequal numbers {'t1': 2, 't2': 1}: x"
    missing = ("categories with no question run (none among the selected questions, or none "
               "active at this release): ['math']")
    s1 = _lb_shard(1, 2, [_lb_trace("a", "t1"), _lb_trace("b", "t1"), _lb_trace("c", "t2")],
                   reasons=[unequal, missing])
    s2 = _lb_shard(2, 2, [_lb_trace("d", "t2"), _lb_trace("e", "m1", category="math")])
    merged = LB.merge_shards([s1, s2])
    assert merged["leaderboard_comparable"] is True, merged["leaderboard_comparable_reason"]
    # a gap the union still has is kept, with the union's counts
    s2b = _lb_shard(2, 2, [_lb_trace("d", "t2"), _lb_trace("f", "t2")])
    merged = LB.merge_shards([s1, s2b])
    reason = merged["leaderboard_comparable_reason"]
    assert "unequal numbers {'t1': 2, 't2': 3}" in reason
    assert "none active at this release): ['math']" in reason
    assert merged["leaderboard_comparable"] is False


def test_livebench_the_run_and_the_merge_word_the_unequal_reason_alike():
    from gbench.runners.eval_suites import livebench as LB
    src = _read("gbench/runners/eval_suites/livebench.py")
    assert "reasons.append(_if_unequal_reason(if_rows))" in src
    assert LB._if_unequal_reason({"a": 1}).startswith(LB._SHARD_ONLY_REASONS[0])


def _tb_shard(i, models, sim, errors, replies):
    return {"eval_name": "toolbench", "shard": f"{i}/2", "status": "completed_with_errors",
            "sample_traces": [{"group": "G1", "query_id": str(i), "harness_errors": [],
                               "sopr_passes": [{"pass": 0, "status": "Solved"}],
                               "sowr_rounds": [{"vote": "cand"}]}],
            "judge_models_used": models, "judge_models_used_by_group": {"G1": models},
            "simulator_models_used": sim,
            "gemini_models_answered": {m: models.get(m, 0) + sim.get(m, 0)
                                       for m in set(models) | set(sim)},
            "harness_error_queries_by_group": {"G1": errors},
            "harness_error_query_kinds": {"model_endpoint": len(errors)},
            "harness_errors": [{"error": f"shard {i} error"}], "model_replies": replies,
            "simulator_usage": {"tool_calls": 2}, "tool_calls_by_type": {"fake_response": 1},
            "server_errors_by_stage": {}, "leaderboard_comparable_reason": None}


def test_toolbench_merge_records_every_shards_gemini_models_and_errors():
    from gbench.runners.eval_suites import toolbench as T
    a = _tb_shard(1, {"gemini-3.8-flash": 3}, {"gemini-3.8-flash": 1}, ["7"],
                  {"replies": 4, "cut_at_max_tokens": 1})
    b = _tb_shard(2, {"gemini-3.7-flash": 2}, {}, ["9", "10"], {"replies": 5})
    m = T.merge_shards([a, b])
    assert m["judge_models_used"] == {"gemini-3.7-flash": 2, "gemini-3.8-flash": 3}
    assert m["gemini_models_answered"] == {"gemini-3.7-flash": 2, "gemini-3.8-flash": 4}
    assert m["simulator_models_used"] == {"gemini-3.8-flash": 1}
    assert m["judge_models_used_by_group"] == {"G1": {"gemini-3.7-flash": 2,
                                                      "gemini-3.8-flash": 3}}
    assert m["harness_error_queries_by_group"] == {"G1": ["7", "9", "10"]}
    assert m["harness_error_query_kinds"] == {"model_endpoint": 3}
    assert [e["error"] for e in m["harness_errors"]] == ["shard 1 error", "shard 2 error"]
    assert m["model_replies"] == {"cut_at_max_tokens": 1, "replies": 9}
    assert m["simulator_usage"] == {"tool_calls": 4}
    assert m["accuracy"] == 100.0
    # a shard without a request log leaves the merged reply count unknown, never partial
    b["model_replies"] = None
    assert T.merge_shards([a, b])["model_replies"] is None


def _fake_datasets(monkeypatch, rows_by_config):
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda name, config=None, split=None, revision=None: [
        dict(r) for r in rows_by_config(name, config, split)]
    monkeypatch.setitem(sys.modules, "datasets", fake)


def test_mmlu_identical_rows_keep_their_own_position(monkeypatch):
    from gbench.runners.eval_suites import mmlu
    row = {"subject": "anatomy", "question": "Q?", "choices": ["a", "b", "c", "d"], "answer": 0}
    _fake_datasets(monkeypatch, lambda name, config, split: [row, row, dict(row, question="R?")])
    items = mmlu._load_mmlu_items(limit=None, n_shot=0)
    assert [it["row_index"] for it in items] == [0, 1, 2]
    from gbench.runners.eval_suites.base import sample_key
    keys = {sample_key({"subject": it["subject"], "row_index": it["row_index"],
                        "question": it["question"], "gold": it["gold"]}) for it in items}
    assert len(keys) == 3
    assert '"row_index": items[idx].get("row_index")' in _read(
        "gbench/runners/eval_suites/mmlu.py")


def test_mmlu_redux_identical_rows_keep_their_own_position(monkeypatch):
    from gbench.runners.eval_suites import mmlu_redux as R
    row = {"question": "Q?", "choices": ["a", "b", "c", "d"], "answer": 1, "error_type": "ok"}

    def rows(name, config, split):
        if name == R.DATASET:
            return [row, row] if config == R.MMLU_REDUX_SUBJECTS[0] else []
        return []
    _fake_datasets(monkeypatch, rows)
    monkeypatch.setattr(R, "_dev_blocks", lambda: {R.MMLU_REDUX_SUBJECTS[0]: []})
    monkeypatch.setattr(R, "helm_prompt", lambda subject, blocks, q, choices: f"{subject}:{q}")
    items = R._load_mmlu_redux_items(limit=None)
    assert [it["row_index"] for it in items] == [0, 1]
    assert items[0]["prompt"] == items[1]["prompt"]
    assert '"row_index": it.get("row_index")' in _read("gbench/runners/eval_suites/mmlu_redux.py")


def test_omnidocbench_merge_names_a_missing_annotation_file(monkeypatch):
    from gbench.merge import MergeError
    from gbench.runners.eval_suites import omnidocbench as O
    from huggingface_hub.errors import LocalEntryNotFoundError
    monkeypatch.setattr(O, "_check_prereqs", lambda image: None)

    def _offline(**k):
        raise LocalEntryNotFoundError("offline and not in the cache")
    monkeypatch.setattr(O, "hf_hub_download", _offline)
    trace = {"sample_key": "k:a", "global_idx": 0, "attempt_idx": 0, "is_correct": False,
             "category": "v1.5", "gold_answer": json.dumps({"page_name": "a.png"}),
             "response_text": "# t", "finish_reason": "stop", "health": "ok"}
    shard = {"eval_name": "omnidocbench", "shard": "1/1", "status": "success", "accuracy": 1.0,
             "sample_traces": [trace]}
    with pytest.raises(MergeError, match="OmniDocBench.json"):
        O.merge_shards([shard])


def test_the_decode_rate_warning_counts_timeouts_as_harness_errors():
    src = _flat(_read("gbench/runners/eval_suites/base.py"))
    assert "which biases the score downward" not in src
    assert ("Long generations on this endpoint will time out: those are harness errors, left "
            "out of the score") in src.replace('" "', "")


# --------------------------------------------------------------------------- #
# docs
# --------------------------------------------------------------------------- #
def test_readme_and_env_knobs_describe_the_bind_address_and_the_endpoint_key():
    readme = _read("docs/environment-variables.md")
    host = next(l for l in readme.splitlines() if l.startswith("| `GBENCH_SERVER_HOST` |"))
    assert "a run that selects no eval suite" in host and "`result['server']['host_source']`" \
        in host
    key = next(l for l in readme.splitlines() if l.startswith("| `VLLM_API_KEY` |"))
    assert "to a remote endpoint or the server gBench launched" in key
    assert "`server.endpoint_key_enforced`" in key
    knobs = _flat(_read("docs/evals/env-knobs.md"))
    assert "for the quality / Golden Set server of a run that selects no eval suite" in knobs
    assert "host_source, endpoint_key_enforced (never the key)" in knobs
    assert "a placeholder `VLLM_API_KEY` is removed from its environment" in knobs


def test_the_key_table_lists_every_backend_key_with_where_to_get_it():
    tools = _read("docs/evals/toolchains.md")
    rapid = next(l for l in tools.splitlines() if l.startswith("| `RAPID_API_KEY` |"))
    assert "rapidapi.com/DataCrawler/api/booking-com15" in rapid
    assert "unset or empty" in rapid and "never replaced by the Gemini judge" in rapid
    mcp = next(l for l in tools.splitlines() if l.startswith("| `NPS_API_KEY`"))
    assert "unset or empty" in mcp and "mcp_bench.md#the-keys" in mcp
    gem = next(l for l in tools.splitlines() if l.startswith("| `GEMINI_API_KEY`"))
    assert "no model role reads a SerpAPI" in gem and "no suite reads a SerpAPI" not in gem
    # the anchors the rows link to exist
    for doc, heading in (("mcp_bench", "### The keys"), ("mcp_bench", "### Checking the keys"),
                         ("complexfuncbench", "## Requirements")):
        assert f"\n{heading}\n" in _read(f"docs/evals/{doc}.md"), (doc, heading)


def test_the_shared_docs_run_mcp_atlas_page_tools_on_the_release_servers():
    from gbench.runners.eval_suites import mcp_atlas
    readme = _read("docs/eval-suites.md")
    row = next(l for l in readme.splitlines() if "| `mcp_atlas` |" in l)
    assert f"{len(mcp_atlas.ENVIRONMENT_SERVERS)} of its 36 servers" in row
    for name in ("EXA_API_KEY", "OXYLABS_USERNAME", "OXYLABS_PASSWORD"):
        assert name in row
    for rel in (*_README_AND_SPLIT_DOCS, "docs/evals/env-knobs.md", "docs/evals/toolchains.md",
                "docs/RESOURCES.md"):
        assert "URL context" not in _read(rel), rel
    from gbench.runners.eval_suites import search_tool
    assert "url_context (its own request in mcp_atlas.py" not in search_tool.__doc__


def test_the_golden_docs_describe_the_derived_leak_tokens():
    from gbench.runners import golden
    readme = _flat(_read("README.md") + "\n" + _read("docs/golden-set.md"))
    assert "Only Gemma-family delimiters are listed" not in readme
    assert "`forbid_chat_template_tokens`" in readme and "`chat_template_leak_check`" in readme
    assert "pytest tools/template_checks" in readme and "tests/test_chat_template_gemma4.py" \
        not in readme
    for rel in _README_AND_SPLIT_DOCS:   # README.md and every doc split out of it
        text = _flat(_read(rel))
        assert "Only Gemma-family delimiters are listed" not in text, rel
        assert "tests/test_chat_template_gemma4.py" not in text, rel
    versioning = _flat(_read("docs/scaffold-versioning.md"))
    assert f"`SCORING_VERSION` is {golden.SCORING_VERSION}." in versioning
    tests_readme = _read("tests/README.md")
    assert "tests/test_chat_template_gemma4.py" not in tests_readme
    assert "tools/template_checks/README.md" in tests_readme


def test_the_distributed_page_follows_the_scripts():
    dist = _flat(_read("docs/distributed-gbench.md"))
    assert "--reasoning-parser NAME --pool POOL" in dist
    assert "`ui_control_osworld` and the SWE family" in dist
    assert "`terminal_bench`, `toolbench`, `ui_control_osworld`, `wildclawbench`" in dist
    assert "`shard_not_applied`" in dist


def test_the_sandbox_page_names_the_swe_bench_live_cleanup_label():
    from gbench.runners.eval_suites import swe_bench_live
    page = _read("docs/evals/docker-sandboxes.md")
    assert f"label={swe_bench_live.SWEAGENT_BATCH_LABEL}" in page


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def test_the_provenance_records_the_page_tool_and_merge_decisions():
    from gbench.runners.eval_suites import canonical_sync as cs
    C, S = cs.CANONICAL_SYNC, cs.SHARED_SYNC
    assert C["mcp_atlas"]["synced"] >= "2026-09-28"
    atlas = C["mcp_atlas"]["method"]
    assert "their page tools run as upstream on the release's servers (UPSTREAM_SEARCH_TOOLS)" \
        in atlas and "label gbench.mcp_atlas.search_servers=removed:brave-search;" in atlas
    for name in ("swe_bench_live", "ui_control_osworld", "mcp_bench", "mmlu", "mmlu_redux"):
        assert C[name]["synced"] >= "2026-09-28", name
    for name in ("skillsbench", "wildclawbench", "gaia2", "livebench"):
        assert C[name]["synced"] >= "2026-09-28.2", name
    # terminal_bench was bumped again when Harbor moved into its local image, then for
    # harbor run --max-retries 3; tau_common when USER_ERROR became a harness error
    assert C["terminal_bench"]["synced"] >= "2026-09-28.3"
    assert S["tau_common"]["synced"] >= "2026-09-28.2"
    assert "swe_bench_live.SweagentWatch" in C["swe_bench_live"]["method"]
    assert "row_index" in C["mmlu"]["method"] and "row_index" in C["mmlu_redux"]["method"]
    # verdict-preserving changes are declared with exactly the current fingerprint
    fps = cs.code_fingerprints()
    # swe_lancer, toolbench and omnidocbench changed a verdict since (the 2026-10-07 thinking
    # temperature, the 2026-10-09 --eval-limit rule): their synced moved, and their earlier
    # verdict-preserving declarations are gone
    for name in ("swe_lancer", "toolbench", "omnidocbench"):
        assert C[name]["synced"] >= "2026-10-07" and name not in cs.VERDICT_PRESERVING, name
    # base's context-retry, pass^k and status changes can move a verdict: bumped, not declared
    assert S["base"]["synced"] >= "2026-09-29" and "base" not in cs.VERDICT_PRESERVING


def test_label_literals_in_the_provenance_equal_the_dockerfiles():
    from gbench.runners.eval_suites import canonical_sync as cs
    labels = {}
    for path in list((REPO / "gbench" / "docker").glob("*Dockerfile")) + list(
            (REPO / "docker").glob("*Dockerfile")):
        if ".bak" in path.name:
            continue
        text = path.read_text(encoding="utf-8")
        args = dict(re.findall(r"(?m)^ARG\s+(\w+)=(\S+)", text))
        for key, value in re.findall(r'((?:org\.)?gbench\.[\w.\-]+)="?([^"\s\\]+)"?', text):
            value = re.sub(r"\$\{(\w+)\}", lambda m: args.get(m.group(1), m.group(0)), value)
            labels.setdefault(key, set()).add(value)
    checked = 0
    for name, entry in {**cs.SHARED_SYNC, **cs.CANONICAL_SYNC}.items():
        text = " ".join(str(v) for v in entry.values())
        for key, value in re.findall(r"((?:org\.)?gbench\.[\w\-]+(?:\.[\w\-]+)*)=([\w.:,\-]+)",
                                     text):
            value = value.rstrip(".,;")
            if key in labels and not value.startswith("<"):
                assert value in labels[key], (name, key, value, labels[key])
                checked += 1
    assert checked >= 5
