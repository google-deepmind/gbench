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

"""Canonical protocols of the knowledge and reasoning suites: cruxeval's grader-side verdict,
arc_agi on the official ARC-AGI-1 data, --shard for the suites that run outside
base.run_eval_suite (mmlu, mmlu_redux, wmdp), gsm8k as lm-evaluation-harness gsm8k_cot,
causalbench as the Corr2Cause release protocol, the lm-eval tokenization of mmlu/wmdp, and the
official sampling settings each suite now sends by default. No network and no model server:
datasets and endpoints are faked; the cruxeval grader runs for real (unsandboxed in tests,
see conftest)."""

import base64
import hashlib
import io
import json
import logging
import tarfile
import zlib
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import (arc_agi, base, causalbench, cruxeval, gpqa,
                                        gpqa_diamond, gsm8k, lab_bench, longbench_v2_code, mmlu,
                                        mmlu_pro, mmlu_redux, multilingual_mmlu, wmdp)
from gbench.runners.eval_suites.sampling import parse_shard, shard_select

REPO = Path(__file__).resolve().parent.parent


def _doc(suite):
    return (REPO / "docs" / "evals" / f"{suite}.md").read_text(encoding="utf-8")


def _fake_hf(rows_by_key):
    """load_dataset stand-in keyed by (path, config, split), (path, config) or path; asserts a
    pinned revision is requested."""
    def _load(path, *args, **kwargs):
        assert kwargs.get("revision"), f"{path} loaded without a pinned revision"
        name = args[0] if args else kwargs.get("name")
        for k in ((path, name, kwargs.get("split")), (path, name), path):
            if k in rows_by_key:
                return list(rows_by_key[k])
        raise AssertionError(f"unexpected dataset request {(path, name, kwargs.get('split'))}")
    return mock.patch("datasets.load_dataset", side_effect=_load)


class _CharTok:
    """One token per character, no BOS."""
    bos_token_id = None

    def encode(self, s):
        return [ord(c) for c in s]


# =========================================================================== #
# cruxeval: the verdict comes from the grader's comparison, not the exit status
# =========================================================================== #
_CODE = "def f(x):\n    return x + 1"
_OUT = {"code": _CODE, "input": "1", "output": "2", "mode": "output", "cot": False}
_IN = {"code": _CODE, "input": "1", "output": "2", "mode": "input", "cot": False}


def test_cruxeval_vendors_upstream_utils_execute_verbatim():
    # facebookresearch/cruxeval evaluation/utils_execute.py at 190faf16 (git blob 9eec77cd).
    data = cruxeval._UTILS_EXECUTE.encode()
    assert hashlib.sha256(data).hexdigest() == (
        "3da60293cb6d75115920acd7c890445810108334d6f787a6e6d6955aef166a2f")
    assert hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == (
        "9eec77cd6e6ad80d32ab4f452a4907f3f4446f80")


@pytest.mark.parametrize("reply,gold,verdict", [
    ("assert f(1) == 2", _OUT, "passed"),
    ("assert f(1) == 3", _OUT, "failed"),
    ("assert f(1) == 2.0", _OUT, "passed"),          # Python equality, as upstream's assert
    ("assert f(1) == '2'", _OUT, "failed"),
    ("assert f(1) == 2", _IN, "passed"),
    ("assert f(5) == 2", _IN, "failed"),
    ("assert f(1) == 7", {**_OUT, "input": "7"}, "failed"),
])
def test_cruxeval_honest_answers_are_scored_by_comparison(reply, gold, verdict):
    assert cruxeval._check(reply, gold) == verdict


@pytest.mark.parametrize("reply,gold", [
    ("assert f(1) == exit()", _OUT),
    ("assert f(1) == quit()", _OUT),
    ("assert f(1) == __import__('sys').exit(0)", _OUT),
    ("assert f(1) == (_ for _ in ()).throw(SystemExit)", _OUT),
    ("assert f(1) == __import__('os')._exit(0)", _OUT),
    ("assert f(5) or exit() == 2", _IN),
    ("assert f(__import__('os')._exit(0)) == 2", _IN),
])
def test_cruxeval_an_answer_that_ends_the_program_early_fails(reply, gold):
    """Upstream records these as failed (reliability_guard sets exit/quit to None, SystemExit is
    caught as a failure, a dead worker records no "passed"); so must gbench."""
    assert cruxeval._check(reply, gold) == "failed"


def test_cruxeval_needs_exactly_one_true_comparison():
    # `or 2` makes upstream's assert true; here the single comparison (f(5) == 2) is false.
    assert cruxeval._check("assert f(5) or 2 == 2", _IN) == "failed"


@pytest.mark.parametrize("code,inp,out,reply", [
    ("def f(s):\n    return s.encode()", "'a'", "b'a'", "assert f('a') == b'a'"),
    ("def f(x):\n    return {x: [x, (x,)]}", "1", "{1: [1, (1,)]}", "assert f(1) == {1: [1, (1,)]}"),
    ("def f(x):\n    return None", "1", "None", "assert f(1) == None"),
    ("def f(x):\n    return x > 0", "1", "True", "assert f(1) == True"),
])
def test_cruxeval_plain_data_gold_types_cross_to_the_grader(code, inp, out, reply):
    gold = {"code": code, "input": inp, "output": out, "mode": "output", "cot": False}
    assert cruxeval._check(reply, gold) == "passed"


def test_cruxeval_grader_self_check_passes_here():
    assert cruxeval._grader_self_check() is None


def test_cruxeval_run_stops_when_the_grader_is_broken():
    with mock.patch.object(cruxeval, "sandbox_skip_reason", return_value=None), \
            mock.patch.object(cruxeval, "_run_program", return_value=(False, "broken")), \
            mock.patch.object(cruxeval, "run_eval_suite") as run, \
            pytest.raises(RuntimeError, match="INFRA REQUIRED.*known answer"):
        cruxeval.run_cruxeval("m", "http://x/v1", 1)
    run.assert_not_called()


def _crux_run(tokens5=20, **kwargs):
    rows = [{"code": "def f(x):\n    return x", "input": "1", "output": "1", "id": "s0"}]
    with mock.patch("datasets.load_dataset", return_value=rows):
        samples = cruxeval._load_cruxeval_samples()
    calls = []

    def fake_run(**kw):
        calls.append(kw)
        temp, source = base.resolve_official_sampling("cruxeval", kw["official_sampling"])
        tokens = tokens5 if kw["official_sampling"].temperature == 0.8 else 20
        traces = [{"sample_idx": i * 10 + a, "source_sample_idx": i, "gold_answer": gold,
                   "cruxeval_check": "passed", "completion_tokens": tokens, "is_correct": True}
                  for i, (_, gold, _) in enumerate(kw["samples"]) for a in range(10)]
        return {"temperature": temp, "temperature_source": source,
                "attempts": {"attempts_per_sample": 10}, "sample_traces": traces,
                "failed_requests": 0, "status": "success"}
    with mock.patch.object(cruxeval, "sandbox_skip_reason", return_value=None), \
            mock.patch.object(cruxeval, "_grader_self_check", return_value=None), \
            mock.patch.object(cruxeval, "_load_cruxeval_samples", return_value=samples), \
            mock.patch.object(cruxeval, "run_eval_suite", side_effect=fake_run):
        return cruxeval.run_cruxeval("m", "http://x/v1", 1, **kwargs), calls


def test_cruxeval_column_flag_covers_the_pass_at_5_pass(monkeypatch):
    res, calls = _crux_run()
    assert [c["official_sampling"].temperature for c in calls] == [0.2, 0.8]
    # gbench's no-think temperature 0.0 on both passes: neither column is as published.
    col = res["cruxeval_report"]["CRUXEval-O"]
    assert col["pass_at_1_temperature"] == 0.0 and col["pass_at_5_temperature"] == 0.0
    assert col["leaderboard_comparable"] is False
    # Both passes at 0.2: pass@1 is as published, pass@5 is not.
    monkeypatch.setenv("GBENCH_CRUXEVAL_TEMPERATURE", "0.2")
    res, _ = _crux_run()
    col = res["cruxeval_report"]["CRUXEval-I"]
    assert col["pass_at_1_temperature"] == 0.2 and col["pass_at_5_temperature"] == 0.2
    assert col["leaderboard_comparable"] is False
    assert "pass@5 sampled at temperature 0.2" in col["leaderboard_comparable_reason"]
    assert "pass@1 sampled" not in col["leaderboard_comparable_reason"]


def test_cruxeval_long_direct_replies_in_the_pass_at_5_pass_are_flagged():
    res, _ = _crux_run(tokens5=150)
    col = res["cruxeval_report"]["CRUXEval-O"]
    assert col["leaderboard_comparable"] is False
    assert "in the pass@5 pass ran past the upstream chat path's 100-token cap" in col[
        "leaderboard_comparable_reason"]


# =========================================================================== #
# arc_agi: official fchollet/ARC-AGI data at a pinned commit, verified and cached
# =========================================================================== #
# The official files of two tasks upstream corrected in 2024 (zlib + base64 of the exact bytes
# of data/evaluation/<id>.json at the pinned commit; their git blob ids are the entries of the
# pinned data/evaluation tree).
_OFFICIAL = {
    "310f3251": (
        "9dc25ada9652a98a8e89bcc5923de8a6a848b2d1",
        "eNrtVTsOgzAMvUtmD+ZbxFVQBgYqMZRWbZgQd28UApjEINoOFVWFEj/izzMvieiEupd1I/KiE3Vza5VGBQJK0PNJ"
        "ShDXVtnlEBDsGNxgh357zyd7WNLaCIRU29S8L1oYCpF5DAcyj6l01cR9I5f5xKlmtoKRwd5OIGlogYzi2bTqIUoH"
        "DKK03LPD/+/vk/7YI2OzEppvULJ+Sxw7lwDHEkkcO1O4meRW/AbPluwuCrhb+fJ+Tx6v2ZkCHLuj2rF4ZK9FVNVD"
        "ef8gUjJ29I95/beOiU2hdrNfJn7XcTwaj9G/KS+VyEUU4DkKk0D0T83TtCY="),
    "79fb03f4": (
        "3417440c6fb03f157cf592483d224fb9810d76be",
        "eNrtWMtuGzEM/Jc970FGS5TNrxg5pEAC5FC3aDenwP9ex15p+RhSMpAGQVoINtZaiuSMSEr087T8uns8TDf75+nx"
        "8PNpOT3ty/wyeP3Y5zZu572ecxJ15UnS6mD3XXUiCaBfSGo/jW0jibFtOnfQGsLuZxmhb9Y58JYhooxTzTz2c9W9"
        "IuL+LkHrmi0O/WTMvpGM4mNjnjt4CuSzDCLqxxJDTOx0cr7nZ5238/TjafnwKbU7jzyl6uAZP5+HSKmqszSpv5tS"
        "F+ussMjnInzhFVGK5TJ3ti79l7jqZ8OepVTVWYTOdUbprd5szHPoKyvmt5kCdO4U8xrHG6bUcTbHFIGyXAfV94Bd"
        "aiUn0kBGFpdhKUlJzMmjhYwGcp650jEOU6cLtTAgkGx13sI0KdeBqaRFupOxRkqW4HZCi5rQRiTBd/IXqMCx+tW1"
        "BoBEXm7vlPqO+8Udbe7ioGvzNe43JnU5uka9cp90CIIQpTAB/NApQ1YzSL0WXS627VaN+kJJvuh1JOZJPg1gGMFO"
        "ap3a1teivcasDA5qb6gFhy0FmvYaVOTWWV9k+L0V7RKDH6ROUanvsg5la3ChBb/VDSeQD28d0VUwUrWBcoc4eN8x"
        "fTwZX+5/L77F44FWg+HdlEeu35ib3hnfKSyp3WSVXM/hySyR8Zz67W7VQyNtSUKPTbPJIy3NIBuoadqlTWCPjahz"
        "wE0f6i/yfeewAeRORGW8BfEI23yEkV+ZeR6NSNcxDuQkjF6+Og4HK0cn5nF/gHCx85eT/MorTL/vHc9lTiPQWPVn"
        "wUcuw/io3sqw6B3cHwUsWlL2zbVrnBnaK/7K8i7KcOYhGzYs8hE2ZKJEtv6dMmz/zIiihMM/rDoRKXD1bP0vw++i"
        "DL/ciw933++nm+nL14dv5dPD5+n4B9GQ7rM="),
}
# Their corrected test output (the stale mirror had row 2 of 310f3251 all zeros, and a
# different 79fb03f4 output); 79fb03f4's 26x26 grid is pinned by its sha256.
_GOLD_310F3251 = [[0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0], [0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0],
                  [0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2], [4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0]] * 3
_GOLD_79FB03F4_SHA256 = "28fea83e9286531cf30ef449fc758f6cc2aa0f5acd18dadca8a689ddf7a55138"


def _official_files():
    return {f"{tid}.json": zlib.decompress(base64.b64decode(b64))
            for tid, (_, b64) in _OFFICIAL.items()}


def _tarball(files, root=f"ARC-AGI-{arc_agi.ARC_COMMIT}"):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in list(files.items()) + [("README.md", b"not a task")]:
            path = f"{root}/data/evaluation/{name}" if name.endswith(".json") else f"{root}/{name}"
            info = tarfile.TarInfo(path)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_arc_agi_pins_the_official_repository_commit_and_tree():
    assert arc_agi.ARC_REPO == "fchollet/ARC-AGI"
    assert arc_agi.ARC_COMMIT == "399030444e0ab0cc8b4e199870fb20b863846f34"
    assert arc_agi.ARC_EVALUATION_TREE == "4a19f8327d45dda5e1809ee47be08bf1484b493b"
    assert arc_agi._DATASET_REVISION == arc_agi.ARC_COMMIT
    assert arc_agi._git_tree_id({}) == "4b825dc642cb6eb9a060e54bf8d69288fbee4904"   # git's empty tree


def test_arc_agi_official_files_carry_the_corrected_golds():
    files = _official_files()
    for tid, (blob, _) in _OFFICIAL.items():
        data = files[f"{tid}.json"]
        assert hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == blob
    assert json.loads(files["310f3251.json"])["test"][0]["output"] == _GOLD_310F3251
    gold = json.loads(files["79fb03f4.json"])["test"][0]["output"]
    assert hashlib.sha256(json.dumps(gold).encode()).hexdigest() == _GOLD_79FB03F4_SHA256


def test_arc_agi_loader_verifies_caches_and_builds_samples(monkeypatch):
    files = _official_files()
    monkeypatch.setattr(arc_agi, "ARC_EVALUATION_TREE", arc_agi._git_tree_id(files))
    downloads = []
    monkeypatch.setattr(arc_agi, "_download_tarball",
                        lambda: downloads.append(1) or _tarball(files))
    samples = arc_agi._load_arc_agi_samples()
    golds = {m["task_id"]: g for _, g, m in samples if m["pair_index"] == 0}
    assert golds["310f3251"] == _GOLD_310F3251 and len(downloads) == 1
    cache = Path(arc_agi._cache_dir())
    root = Path(base.gbench_cache_dir())
    assert root != Path.home() / ".cache" / "gbench"            # conftest's temporary cache
    assert cache.is_relative_to(root) and arc_agi.ARC_COMMIT in cache.parts
    assert sorted(p.name for p in cache.iterdir()) == ["310f3251.json", "79fb03f4.json"]
    # Second run: from the cache, no download.
    monkeypatch.setattr(arc_agi, "_download_tarball",
                        mock.Mock(side_effect=AssertionError("no download expected")))
    assert arc_agi._load_arc_agi_samples() == samples
    # A cached file that no longer matches the tree is not used: it is downloaded again.
    (cache / "310f3251.json").write_bytes(b"{}")
    monkeypatch.setattr(arc_agi, "_download_tarball", lambda: _tarball(files))
    assert arc_agi._load_arc_agi_samples() == samples


def test_arc_agi_mismatched_or_unreachable_data_is_infra_required(monkeypatch):
    files = _official_files()
    monkeypatch.setattr(arc_agi, "ARC_EVALUATION_TREE", arc_agi._git_tree_id(files))
    # The stale mirror's copy of 310f3251: the third row of the test output all zeros.
    task = json.loads(files["310f3251.json"])
    task["test"][0]["output"][2] = [0] * 12
    stale = {**files, "310f3251.json": json.dumps(task, separators=(",", ":")).encode()}
    monkeypatch.setattr(arc_agi, "_download_tarball", lambda: _tarball(stale))
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*does not match the pinned commit"):
        arc_agi._load_arc_agi_samples()
    monkeypatch.setattr(arc_agi, "_download_tarball", mock.Mock(side_effect=OSError("offline")))
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*could not be downloaded"):
        arc_agi._load_arc_agi_samples()


def _arc_tasks(n):
    return [{"id": f"t{i}", "train": [{"input": [[0]], "output": [[1]]}],
             "test": [{"input": [[i]], "output": [[1]]}] * (1 + i % 2)} for i in range(n)]


def test_arc_agi_shards_whole_tasks_before_the_limit(monkeypatch):
    monkeypatch.setenv("GBENCH_SHARD", "2/2")
    with mock.patch.object(arc_agi, "_load_arc_agi_tasks", return_value=_arc_tasks(6)):
        samples = arc_agi._load_arc_agi_samples()
        limited = arc_agi._load_arc_agi_samples(limit=1)
    assert {m["task_id"] for _, _, m in samples} == {"t1", "t3", "t5"}
    assert len(samples) == 6                                     # every pair of each task
    assert len({m["task_id"] for _, _, m in limited}) == 1
    assert {m["task_id"] for _, _, m in limited} <= {"t1", "t3", "t5"}
    # base's own GBENCH_SHARD selection of the laid-out list gives back exactly the samples.
    assert shard_select(arc_agi._for_base(samples), parse_shard("2/2")) == samples


def test_arc_agi_sends_no_temperature_by_default():
    assert arc_agi.OFFICIAL_SAMPLING.send_temperature is False
    seen = []

    def fake_run(**kw):
        seen.append(kw)
        return {"attempts": {"attempts_per_sample": 2}, "sample_traces": [],
                "official_sampling": {"temperature_matches_official": False},
                "temperature": 0.3, "temperature_source": "cli:--temperature"}
    with mock.patch.object(arc_agi, "_load_arc_agi_samples", return_value=[]), \
            mock.patch.object(arc_agi, "run_eval_suite", side_effect=fake_run):
        res = arc_agi.run_arc_agi("m", "http://x/v1")
    assert seen[0]["official_sampling"] is arc_agi.OFFICIAL_SAMPLING
    assert "temperature" not in seen[0]
    assert "temperature 0.3 (cli:--temperature; arc-agi-benchmarking sends none" in res[
        "leaderboard_comparable_reason"]


# =========================================================================== #
# --shard for mmlu, mmlu_redux and wmdp (they do not go through run_eval_suite)
# =========================================================================== #
_MMLU_TEST = [{"question": f"Q{i}?", "choices": ["a", "b", "c", "d"], "answer": i % 4,
               "subject": ("anatomy", "virology")[i % 2]} for i in range(8)]
_MMLU_DEV = [{"question": f"D{i}?", "choices": ["a", "b", "c", "d"], "answer": 0,
              "subject": s} for s in ("anatomy", "virology") for i in range(5)]


def _mmlu_run(**kwargs):
    async def fake(base_url, model_name, enc, samples, concurrency,
                   temperature=0.0):
        return [{"loglikelihoods": [0.0, -1, -1, -1], "pred": 0, "is_correct": s["gold"] == 0,
                 "status": "OK", "error": None} for s in samples]
    with _fake_hf({("cais/mmlu", "all", "test"): _MMLU_TEST,
                   ("cais/mmlu", "all", "dev"): _MMLU_DEV}), \
            mock.patch.object(mmlu, "_load_tokenizer", return_value=_CharTok()), \
            mock.patch.object(mmlu, "_score_all", side_effect=fake), \
            mock.patch.object(mmlu, "get_run_knob", return_value=None):
        return mmlu.run_mmlu("m", "http://x/v1", 1, **kwargs)


def test_mmlu_honours_shard_before_the_limit(monkeypatch):
    full = _mmlu_run()
    assert full["total_questions"] == 8 and "shard" not in full
    monkeypatch.setenv("GBENCH_SHARD", "1/4")
    res = _mmlu_run()
    assert res["total_questions"] == 2 and res["shard"] == "1/4"
    assert [t["prompt"].rsplit("Q", 1)[1][:2] for t in res["sample_traces"]] == ["0?", "4?"]
    assert _mmlu_run(limit=1)["total_questions"] == 1


def _redux_rows(subject, n):
    return [{"question": f"{subject} q{i}?", "choices": ["a", "b", "c", "d"], "answer": 0,
             "error_type": "ok"} for i in range(n)]


def _redux_run(**kwargs):
    rows = {(mmlu_redux.DATASET, s, "test"): _redux_rows(s, 2)
            for s in mmlu_redux.MMLU_REDUX_SUBJECTS}
    dev = [{"question": f"d{i}", "choices": ["a", "b", "c", "d"], "answer": 0, "subject": s}
           for s in mmlu_redux.MMLU_REDUX_SUBJECTS for i in range(5)]
    rows[(mmlu_redux.MMLU_DATASET, "all", "dev")] = dev
    sent = []

    async def fake_generate(base_url, model_name, prompts, concurrency, temperature):
        sent.append(temperature)
        return [base.Reply("A", None, "length") for _ in prompts]
    with _fake_hf(rows), mock.patch.object(mmlu_redux, "_generate_all", side_effect=fake_generate):
        return mmlu_redux.run_mmlu_redux("m", "http://x/v1", 1, **kwargs), sent


def test_mmlu_redux_honours_shard_and_records_helm_sampling(monkeypatch):
    res, sent = _redux_run()
    assert res["total_questions"] == 114 and sent == [0.0]
    assert res["official_sampling"]["temperature"] == 0.0
    assert res["official_sampling"]["temperature_matches_official"] is True
    monkeypatch.setenv("GBENCH_SHARD", "2/4")
    res, _ = _redux_run()
    assert res["total_questions"] == 29 and res["shard"] == "2/4"     # rows 1, 5, ..., 113


def test_mmlu_redux_warns_when_replies_are_not_letters(caplog):
    rows = {(mmlu_redux.DATASET, s, "test"): _redux_rows(s, 1)
            for s in mmlu_redux.MMLU_REDUX_SUBJECTS}
    rows[(mmlu_redux.MMLU_DATASET, "all", "dev")] = [
        {"question": "d", "choices": ["a", "b", "c", "d"], "answer": 0, "subject": s}
        for s in mmlu_redux.MMLU_REDUX_SUBJECTS for _ in range(5)]

    async def fake_generate(base_url, model_name, prompts, concurrency, temperature):
        return [base.Reply("", None, "length") for _ in prompts]
    with _fake_hf(rows), mock.patch.object(mmlu_redux, "_generate_all", side_effect=fake_generate), \
            caplog.at_level(logging.WARNING):
        res = mmlu_redux.run_mmlu_redux("m", "http://x/v1", 1)
    assert res["non_letter_replies"] == 57
    assert "are not a bare option letter" in caplog.text


def test_wmdp_honours_shard_before_the_limit(monkeypatch):
    rows = {("cais/wmdp", sub): [{"question": f"{sub} {i}", "choices": ["w", "x", "y", "z"],
                                  "answer": 0} for i in range(4)] for sub, _ in wmdp._SUBSETS}

    async def fake(base_url, model_name, enc, samples, concurrency,
                   temperature=0.0):
        return [{"loglikelihoods": [0.0] * 4, "pred": 0, "is_correct": True, "status": "OK",
                 "error": None} for _ in samples]
    monkeypatch.setenv("GBENCH_SHARD", "3/3")
    with _fake_hf(rows), mock.patch.object(wmdp, "_load_tokenizer", return_value=_CharTok()), \
            mock.patch.object(wmdp, "_score_all", side_effect=fake):
        res = wmdp.run_wmdp("m", "http://x/v1")
    assert res["total_questions"] == 4 and res["shard"] == "3/3"      # items 2, 5, 8, 11 of 12


# =========================================================================== #
# mmlu / wmdp tokenization: the tokenizer's own defaults (lm-eval), no forced BOS
# =========================================================================== #
def test_encoder_uses_the_tokenizers_defaults():
    class NoBos(_CharTok):          # like Gemma 4: has a BOS token, adds none by default
        bos_token_id = 2

    class AddsBos(_CharTok):        # a tokenizer whose default encoding starts with BOS
        bos_token_id = 1

        def encode(self, s):
            return [1] + super().encode(s)
    enc = wmdp._Encoder(NoBos())
    assert enc.encode("ab") == [97, 98] and enc.bos_prepended is False
    enc = wmdp._Encoder(AddsBos())
    ctx, cont = enc.pair("Answer:", " A")
    assert ctx[0] == 1 and cont == [32, 65] and enc.bos_prepended is True
    assert mmlu._Encoder is wmdp._Encoder


def test_mmlu_and_wmdp_record_whether_bos_was_present():
    res = _mmlu_run()
    assert res["bos_prepended"] is False and "BOS" not in res["metric"]


# =========================================================================== #
# gsm8k: lm-evaluation-harness gsm8k_cot
# =========================================================================== #
def test_gsm8k_exemplars_and_settings_are_the_task_yaml():
    # gsm8k-cot.yaml at d6de8164: 8 samples, each target ending "The answer is N."
    assert len(gsm8k.GSM8K_COT_SAMPLES) == 8 and gsm8k.N_SHOT == 8
    assert all(t.endswith(f"The answer is {n}.") for (_, t), n in zip(
        gsm8k.GSM8K_COT_SAMPLES, (6, 5, 39, 8, 9, 29, 33, 8)))
    digest = hashlib.sha256(json.dumps(gsm8k.GSM8K_COT_SAMPLES).encode()).hexdigest()
    assert digest == hashlib.sha256(json.dumps(_YAML_SAMPLES).encode()).hexdigest()
    assert gsm8k.UNTIL == ["Q:", "</s>", "<|im_end|>"]
    assert gsm8k.STRICT_PATTERN == "The answer is (\\-?[0-9\\.\\,]+)."
    assert gsm8k.FLEXIBLE_PATTERN == "(-?[$0-9.,]{2,})|(-?[0-9]+)"
    assert gsm8k.REGEXES_TO_IGNORE == [",", "\\$", "(?s).*#### ", "\\.$"]
    assert gsm8k.OFFICIAL_SAMPLING.temperature == 0.0
    assert gsm8k.OFFICIAL_SAMPLING.params == {"seed": 1234}


_YAML_SAMPLES = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After "
     "they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were planted. So "
     "there must have been 21 - 15 = 6. The answer is 6."),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the "
     "parking lot?", "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have "
     "left in total?", "Originally, Leah had 32 chocolates. Her sister had 42. So in total they "
     "had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many "
     "lollipops did Jason give to Denny?", "Jason started with 20 lollipops. Then he had 12 after "
     "giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8."),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many "
     "toys does he have now?", "Shawn started with 5 toys. If he got 2 toys each from his mom and "
     "dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9."),
    ("There were nine computers in the server room. Five more computers were installed each day, "
     "from monday to thursday. How many computers are now in the server room?", "There were "
     "originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 "
     "computers were added. 9 + 20 is 29. The answer is 29."),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. "
     "How many golf balls did he have at the end of wednesday?", "Michael started with 58 golf "
     "balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had "
     "35 - 2 = 33 golf balls. The answer is 33."),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has "
     "23 - 15 dollars left. 23 - 15 is 8. The answer is 8."),
]


def test_gsm8k_messages_are_multiturn_few_shot():
    msgs = gsm8k.fewshot_messages("How many?")
    assert [m["role"] for m in msgs] == ["user", "assistant"] * 8 + ["user"]
    assert msgs[1]["content"] == _YAML_SAMPLES[0][1]
    assert msgs[-1]["content"] == "Q: How many?\nA:"
    assert gsm8k.fewshot_messages("x", 0) == [{"role": "user", "content": "Q: x\nA:"}]
    assert gsm8k.doc_to_target("steps <<2*3=6>>6\n#### 1,200") == "1,200"


@pytest.mark.parametrize("reply,ref,strict,flexible", [
    ("She has 6 left. The answer is 6.", "6", True, True),
    ("So 1,200 in total. The answer is 1,200.", "1200", True, True),
    # "$" is outside strict-match's character class; the stop sequence cuts at "Q:".
    ("The answer is $18.\n\nQ: next question The answer is 18.", "18", False, True),
    ("The answer is 18.\n\nQ: next question The answer is 99.", "18", True, True),
    ("Final Answer: 42", "42", False, True),              # strict needs "The answer is"
    (r"\boxed{42}", "42", False, True),
    ("The answer is 42", "42", False, True),              # the pattern needs one more char
    ("The answer is 41. Actually 42.", "42", False, True),  # strict: first; flexible: last
    ("no number", "7", False, False),
])
def test_gsm8k_filters_and_exact_match_follow_lm_eval(reply, ref, strict, flexible):
    s = gsm8k.score_reply(reply, ref)
    assert s["strict_match_correct"] is strict and s["flexible_extract_correct"] is flexible


def test_gsm8k_stop_is_sent_without_thinking_and_applied_afterwards_with_it():
    rows = [{"question": "q", "answer": "#### 3"}]
    with _fake_hf({"openai/gsm8k": rows}):
        direct = gsm8k._load_gsm8k_samples()
        think = gsm8k._load_gsm8k_samples(enable_thinking=True)
    # local-chat-completions sends `until` as stop and max_gen_toks (256) as max_tokens; the 256
    # is the suite's official cap (OFFICIAL_SAMPLING.max_tokens), applied per mode by base.
    assert direct[0][2] == {"stop": ["Q:", "</s>", "<|im_end|>"]}
    assert gsm8k.OFFICIAL_SAMPLING.max_tokens == 256
    assert think[0][2] == {}
    s = gsm8k.score_reply("The answer is 3. Q: The answer is 4.", "3")
    assert s["strict_match"] == "3." and s["strict_match_correct"] is True     # "\\.$" ignored


def _gsm8k_run(traces, **kwargs):
    seen = {}

    def fake_run(**kw):
        seen.update(kw)
        return {"temperature": 0.0, "temperature_source": "official",
                "official_sampling": {"temperature_matches_official": True},
                "sample_traces": traces, "failed_requests": 0}
    with mock.patch.object(gsm8k, "_load_gsm8k_samples", return_value=[]), \
            mock.patch.object(gsm8k, "run_eval_suite", side_effect=fake_run):
        return gsm8k.run_gsm8k("m", "http://x/v1", 1, **kwargs), seen


def test_gsm8k_scores_both_filters_and_flags_replies_past_max_gen_toks():
    traces = [{"status": "PENDING_JUDGE", "response_text": "x The answer is 5.", "gold_answer": "5",
               "completion_tokens": 40},
              {"status": "PENDING_JUDGE", "response_text": "It is 7", "gold_answer": "7",
               "completion_tokens": 40}]
    import asyncio
    asyncio.run(gsm8k._async_score_gsm8k(traces))
    assert [t["is_correct"] for t in traces] == [True, False]
    res, seen = _gsm8k_run(traces)
    assert seen["official_sampling"] is gsm8k.OFFICIAL_SAMPLING and "temperature" not in seen
    assert seen["limit"] is None and seen["async_eval_fn"] is gsm8k._async_score_gsm8k
    assert res["accuracy"] == 50.0 and res["flexible_extract_accuracy"] == 100.0
    assert res["gsm8k_cot_report"]["strict-match"]["exact_match"] == 50.0
    assert res["leaderboard_comparable"] is True
    long = [dict(traces[0], completion_tokens=300)]
    res, _ = _gsm8k_run(long)
    assert res["leaderboard_comparable"] is False
    assert "past lm-eval's 256-token max_gen_toks" in res["leaderboard_comparable_reason"]


# =========================================================================== #
# causalbench: the Corr2Cause release protocol
# =========================================================================== #
def test_causalbench_prompt_is_run_model_py_direct_template():
    premise, hypothesis = causalbench._split_input(
        "Premise: Suppose there is a closed system of 2 variables, A and B. A correlates with "
        "B.\nHypothesis: A directly causes B.")
    assert causalbench.build_prompt(premise, hypothesis) == (
        "Question: Suppose there is a closed system of 2 variables, A and B. A correlates with "
        "B.\nCan we deduct the following: A directly causes B? Just answer \"Yes\" or \"No.\""
        "\n\nAnswer:")
    assert causalbench.SYSTEM_PROMPT == ("You are a highly intelligent question-answering bot "
                                         "with profound knowledge of causal inference.")
    with pytest.raises(RuntimeError, match="unexpected 'input' format"):
        causalbench._split_input("no premise here")


@pytest.mark.parametrize("reply,relation", [
    ("Yes", "entailment"), ("No.", "contradiction"), ('"Yes".', "entailment"),
    ("Yes, because A and B are dependent.", "entailment"), ("Necessarily true", "entailment"),
    ("Not enough information", "contradiction"), ("that there is yes", "entailment"),
    ("", "contradiction"), ("I think yes", "contradiction"), ("**Yes**", "contradiction"),
])
def test_causalbench_reads_replies_with_rel_normalize(reply, relation):
    assert causalbench.predicted_relation(reply) == relation


def test_causalbench_metric_is_the_minority_label_f1():
    golds = ["entailment"] * 2 + ["contradiction"] * 6
    preds = ["entailment", "contradiction", "entailment"] + ["contradiction"] * 5
    rep = causalbench.perf_report(golds, preds)
    assert rep["minority_label"] == "entailment"
    assert (rep["P"], rep["R"], rep["F1"], rep["Acc"]) == (50.0, 50.0, 50.0, 75.0)


def test_causalbench_run_is_comparable_only_as_published():
    traces = [{"gold_answer": "entailment", "response_text": "Yes", "completion_tokens": 2,
               "extra_payload": {"num_variables": 3, "relation": "parent"}},
              {"gold_answer": "contradiction", "response_text": "No, because the premise "
               "says A is independent of B given C and so on", "completion_tokens": 20,
               "extra_payload": {"num_variables": 3, "relation": "child"}}]
    seen = {}

    def fake_run(**kw):
        seen.update(kw)
        return {"sample_traces": traces, "official_sampling": {"temperature_matches_official": True}}
    with mock.patch.object(causalbench, "_load_causalbench_samples", return_value=[]), \
            mock.patch.object(causalbench, "run_eval_suite", side_effect=fake_run):
        res = causalbench.run_causalbench("m", "http://x/v1", 1)
        think = causalbench.run_causalbench("m", "http://x/v1", 1, enable_thinking=True)
    assert seen["official_sampling"] is causalbench.OFFICIAL_SAMPLING
    assert causalbench.OFFICIAL_SAMPLING.temperature == 0.0
    assert res["accuracy"] == 100.0 and res["leaderboard_comparable"] is True
    rep = res["causalbench_report"]
    assert rep["replies_over_upstream_cap"] == 1 and set(rep["by_relation"]) == {"parent", "child"}
    assert think["leaderboard_comparable"] is False and "--thinking" in think[
        "leaderboard_comparable_reason"]


def test_causalbench_requires_nltk_punkt():
    import nltk
    with mock.patch.object(nltk, "word_tokenize", side_effect=LookupError("Resource punkt_tab")), \
            pytest.raises(RuntimeError, match="INFRA REQUIRED.*punkt"):
        causalbench._require_nltk()


# =========================================================================== #
# official sampling: each suite sends its upstream's settings by default
# =========================================================================== #
def _capture(module, run, loader, loader_value, **kwargs):
    seen = {}

    def fake_run(**kw):
        seen.update(kw)
        return {"sample_traces": [], "temperature": None}
    with mock.patch.object(module, loader, return_value=loader_value), \
            mock.patch.object(module, "run_eval_suite", side_effect=fake_run):
        getattr(module, run)("m", "http://x/v1", 1, **kwargs)
    return seen


@pytest.mark.parametrize("module,run", [(gpqa, "run_gpqa"), (gpqa_diamond, "run_gpqa_diamond")])
def test_gpqa_samples_like_simple_evals_samplers(module, run):
    # Both variants run through gpqa._run_simple_evals_gpqa.
    # A served model takes simple-evals' chat sampler in both modes (T=0.5).
    for thinking, temp, send in ((False, 0.5, True), (True, 0.5, True)):
        seen = {}
        with mock.patch.object(gpqa, "_load_gpqa_items", return_value=[]), \
                mock.patch.object(gpqa, "run_eval_suite",
                                  side_effect=lambda **kw: seen.update(kw) or {}):
            getattr(module, run)("m", "http://x/v1", 1, enable_thinking=thinking)
        # One declaration for both modes; base picks the mode's settings (for_mode).
        official = seen["official_sampling"].for_mode(thinking)
        assert official.temperature == temp and official.send_temperature is send
        assert "temperature" not in seen


def test_multilingual_mmlu_samples_like_simple_evals_samplers():
    # A served model takes simple-evals' chat sampler in both modes.
    for thinking, official in ((False, gpqa.SIMPLE_EVALS_SAMPLING),
                               (True, gpqa.SIMPLE_EVALS_SAMPLING)):
        seen = _capture(multilingual_mmlu, "run_multilingual_mmlu",
                        "_load_multilingual_mmlu_samples", [], enable_thinking=thinking)
        # One declaration for both modes; base picks the mode's settings (for_mode).
        assert seen["official_sampling"] is gpqa.SIMPLE_EVALS_SAMPLING
        assert seen["official_sampling"].for_mode(thinking) is official
        assert "temperature" not in seen


def test_simple_evals_flag_names_an_operator_temperature():
    result = {"temperature": 0.0, "temperature_source": "cli:--temperature",
              "official_sampling": {"temperature_matches_official": False,
                                    "send_temperature": True, "temperature": 0.5}}
    res = gpqa._flag_comparability(result, None, 448, 448, 10)
    assert res["leaderboard_comparable"] is False
    assert "temperature 0.0 (cli:--temperature; simple-evals' sampler sends temperature 0.5)" in (
        res["leaderboard_comparable_reason"])


def test_mmlu_pro_samples_like_evaluate_from_apix():
    # evaluate_from_apiX.py streaming_request sends the same sampling for every model; gbench
    # keeps its top_p in both modes and samples at its rule's temperature (1.0 / 0.0), recording
    # upstream's 0.1.
    assert mmlu_pro.OFFICIAL_SAMPLING.temperature == 0.1
    assert mmlu_pro.OFFICIAL_SAMPLING.params == {"top_p": 0.95}
    for thinking in (False, True):
        with mock.patch.object(mmlu_pro, "_load_mmlu_pro_samples", return_value=[]):
            res = mmlu_pro.run_mmlu_pro("m", "http://x/v1", 1, enable_thinking=thinking)
        assert (res["temperature"], res["temperature_source"]) == (
            (1.0, "thinking") if thinking else (0.0, "default"))
        assert res["official_sampling"]["params"] == {"top_p": 0.95}
        assert res["official_sampling"]["temperature"] == 0.1


def test_lab_bench_sends_no_temperature_like_the_baseline_harness():
    seen = {}
    with mock.patch.object(lab_bench, "_load_lab_bench_samples", return_value=[]), \
            mock.patch.object(lab_bench, "run_eval_suite",
                              side_effect=lambda **kw: seen.update(kw) or {"sample_traces": []}):
        lab_bench.run_lab_bench("m", "http://x/v1")
    assert seen["official_sampling"].send_temperature is False and "temperature" not in seen


def test_official_sampling_reaches_the_request(monkeypatch):
    """End to end through base: a send_temperature=False setting and a fixed one both send
    gbench's rule temperature (0.0 without --thinking), the fixed one with its params, and the
    result records the official values."""
    sent = []

    async def fake_send(**kw):
        sent.append((kw.get("temperature"), dict(kw.get("extra_payload") or {})))
        return base.Reply("x", None, "stop")
    monkeypatch.setattr(base, "_send_single_request", fake_send)
    samples = [([{"role": "user", "content": "q"}], "g", {})]
    res = base.run_eval_suite(eval_name="arc_agi", model_name="m", base_url="http://x",
                              concurrency=1, samples=samples, eval_fn=lambda a, b: False,
                              official_sampling=arc_agi.OFFICIAL_SAMPLING)
    assert sent[-1][0] == 0.0 and res["official_sampling"]["temperature_matches_official"] is False
    res = base.run_eval_suite(eval_name="mmlu_pro", model_name="m", base_url="http://x",
                              concurrency=1, samples=samples, eval_fn=lambda a, b: False,
                              official_sampling=mmlu_pro.OFFICIAL_SAMPLING)
    assert sent[-1][0] == 0.0 and sent[-1][1]["top_p"] == 0.95
    assert res["temperature_source"] == "default"
    assert res["official_sampling"]["temperature"] == 0.1


def test_longbench_v2_code_samples_at_0_1_with_and_without_thinking():
    assert longbench_v2_code.OFFICIAL_SAMPLING.temperature == 0.1


# =========================================================================== #
# docs match the code
# =========================================================================== #
def test_docs_name_the_new_pins_and_protocols():
    arc = _doc("arc_agi")
    assert arc_agi.ARC_COMMIT in arc and arc_agi.ARC_EVALUATION_TREE in arc
    assert "dataartist" not in arc and "codeload.github.com" in arc
    g = _doc("gsm8k")
    assert gsm8k.LM_EVAL_COMMIT in g and "strict-match" in g and "flexible-extract" in g
    assert "seed: 1234" in g and "256" in g
    c = _doc("causalbench")
    assert causalbench.CORR2CAUSE_COMMIT in c and "69d191bc1009be5d4f1320388367ff6a0733433b" in c
    assert "Can we deduct the following" in c and "punkt" in c
    crux = _doc("cruxeval")
    assert "exit status plays no part" in crux and "numpy" in crux
    for suite in ("mmlu", "wmdp"):
        assert "`bos_prepended`" in _doc(suite) and "gets the BOS" not in _doc(suite)
    assert "nan" in _doc("mmlu") and "14,032" in _doc("mmlu")
    for suite in ("mmlu", "mmlu_redux", "wmdp"):
        assert "--shard I/N" in _doc(suite)
    for suite in ("gpqa", "gpqa_diamond", "multilingual_mmlu"):
        assert "default 0.5" in _doc(suite) and "1.0 with `--eval-thinking`" in _doc(suite)
    for suite in ("arc_agi", "lab_bench"):
        assert "no temperature" in _doc(suite)
    for suite in ("arc_agi", "gsm8k", "causalbench", "cruxeval", "mmlu", "mmlu_redux", "wmdp",
                  "gpqa", "gpqa_diamond", "mmlu_pro", "lab_bench", "longbench_v2_code"):
        doc = _doc(suite)
        run = doc.split("## Run", 1)[1]
        assert f"--evals {suite}" in run and "--max-output-tokens" in run, suite
        assert doc.isascii(), suite
