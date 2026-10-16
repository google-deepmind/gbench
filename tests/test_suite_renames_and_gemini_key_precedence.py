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

"""Suite renames, provenance, Gemini key precedence and shared helpers.

  * renames lmsys_noncoding_hard -> wildbench, bfcl_v3_live -> bfcl_live and
    loft_x_arxiv -> loft_scifact (old names resolve with a warning, old temperature variables
    are deprecated aliases, --eval-limit draws the same rows), and cimemories keeps its
    canonical name (the proxy names are gone, not aliases; cybergym is deregistered, see
    tests/test_roadmap_deregistration_and_base_hooks.py).
  * canonical_sync: the documented `checksum` key with full values, SWE entries that match the
    module pins, charxiv's constants byte-identical to upstream's runtime strings.
  * GEMINI_API_KEYS wins over GEMINI_API_KEY everywhere.
  * beam_128k parses judge replies with upstream's json_repair fallback; finalize_mean_metric
    skips scoring_excluded traces; the evals extra pins the multi-swe-bench harness.

No model server, judge API or dataset hub is contacted.
"""

import hashlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from gbench.runners import evals as E
from gbench.runners.eval_suites import base

REPO = Path(__file__).resolve().parents[1]
SUITE_DIR = REPO / "gbench" / "runners" / "eval_suites"
DOCS = REPO / "docs" / "evals"

NEW_RENAMES = {"lmsys_noncoding_hard": "wildbench", "bfcl_v3_live": "bfcl_live",
               "loft_x_arxiv": "loft_scifact"}
REVERTED = {"cimemories_ci_proxy": "cimemories"}
#: Proxy names that never shipped: unknown suites, not aliases.
PROXY_NAMES = ("cybergym_text_proxy", "cimemories_ci_proxy")


class _Stop(Exception):
    pass


def _cli(monkeypatch, *argv):
    """Run gbench.cli.main through validation only; returns the parsed args."""
    from gbench import cli
    captured = {}
    real = cli.create_parser

    def _parser():
        p = real()
        orig = p.parse_args
        p.parse_args = lambda a=None: captured.setdefault("args", orig(a))
        return p

    def _stop(*a, **k):
        raise _Stop

    monkeypatch.setattr(cli, "create_parser", _parser)
    monkeypatch.setattr(cli, "setup_logging", _stop)
    with pytest.raises(_Stop):
        cli.main(["--remote-endpoint", "http://127.0.0.1:1/v1", "--tokenizer", "t",
                  "--max-output-tokens", "1024", *argv])
    return captured["args"]


# --------------------------------------------------------------------------- #
# renames
# --------------------------------------------------------------------------- #
def test_rename_map_has_the_new_renames_and_not_the_reverted_proxy_names():
    from gbench.runners.eval_suites import RENAMED_SUITES
    assert {"semantic_keypoint": "screenspot_pointing", **NEW_RENAMES}.items() <= \
        RENAMED_SUITES.items()
    for proxy in PROXY_NAMES:
        assert proxy not in RENAMED_SUITES and proxy not in RENAMED_SUITES.values()
    for name in REVERTED.values():
        assert name not in RENAMED_SUITES


def test_renamed_and_reverted_suites_are_registered_documented_and_reconciled():
    from gbench.runners.eval_suites import SUITES
    from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC
    builtin = {s for _, su in E.BUILTIN_PILLARS for s in su}
    for old, new in {**NEW_RENAMES, **REVERTED}.items():
        assert new in SUITES and new in builtin and new in CANONICAL_SYNC, new
        assert old not in SUITES and old not in builtin and old not in CANONICAL_SYNC, old
        assert (DOCS / f"{new}.md").exists() and (SUITE_DIR / f"{new}.py").exists(), new
        assert not (DOCS / f"{old}.md").exists() and not (SUITE_DIR / f"{old}.py").exists(), old


def test_modules_carry_their_registry_names():
    from gbench.runners.eval_suites import bfcl_live, cimemories, loft_scifact, wildbench
    for mod, name in ((wildbench, "wildbench"), (bfcl_live, "bfcl_live"),
                      (loft_scifact, "loft_scifact"), (cimemories, "cimemories")):
        assert mod.EVAL_NAME == name
        assert callable(getattr(mod, f"run_{name}"))
    for mod in (wildbench, loft_scifact, cimemories):
        assert mod.DOCS_URL == f"docs/evals/{mod.EVAL_NAME}.md"


def test_old_names_resolve_in_the_cli(monkeypatch):
    args = _cli(monkeypatch, "--evals", "lmsys_noncoding_hard,bfcl_v3_live", "loft_x_arxiv",
                "cimemories")
    assert args.evals == ["wildbench", "bfcl_live", "loft_scifact", "cimemories"]


def test_proxy_names_are_unknown_suites(capsys):
    from gbench import cli
    for proxy in PROXY_NAMES:
        with pytest.raises(SystemExit):
            cli.main(["--remote-endpoint", "http://127.0.0.1:1/v1", "--tokenizer", "t",
                      "--max-output-tokens", "1024", "--evals", proxy])
        assert proxy in capsys.readouterr().err


def test_runner_resolves_each_old_name(monkeypatch, tmp_path):
    ran = []
    monkeypatch.setattr(E, "_resolve_remote_model_id", lambda url, fallback: "m")
    monkeypatch.setattr(E, "_server_max_model_len", lambda url: None)
    monkeypatch.setattr(E.EvalsBenchmarkRunner, "_run_single_eval",
                        lambda self, eval_name, **k: ran.append(eval_name) or {"accuracy": 0.0})
    cfg = SimpleNamespace(evals=list(NEW_RENAMES), eval_custom_jsonl=None, eval_plugins_dir=None,
                          batch_sizes=[1], remote_endpoint="http://127.0.0.1:1/v1", tokenizer="t",
                          eval_max_output_tokens=1024, skip_existing=False,
                          log_manager=SimpleNamespace(results_dir=tmp_path, base_dir=tmp_path,
                                                      save_result=lambda p, r: None))
    model = SimpleNamespace(name="m", short_name="m", hf_model_id="org/m")
    results = E.EvalsBenchmarkRunner(cfg).run(model, SimpleNamespace(value="remote-endpoint"))
    assert ran == list(NEW_RENAMES.values())
    assert all(r["canonical_sync"]["status"] == "reconciled" for r in results)


@pytest.mark.parametrize("old,new", sorted(NEW_RENAMES.items()))
def test_old_temperature_variables_are_deprecated_aliases(monkeypatch, old, new):
    old_var, new_var = base.temperature_env_var(old), base.temperature_env_var(new)
    monkeypatch.delenv(new_var, raising=False)
    monkeypatch.setenv(old_var, "0.3")
    assert base.resolve_temperature(new) == (0.3, f"env:{old_var}")
    monkeypatch.setenv(new_var, "0.7")
    assert base.resolve_temperature(new) == (0.7, f"env:{new_var}")


def test_proxy_temperature_variables_are_not_read():
    """The proxy names never shipped, so their variables are not aliases."""
    assert base.legacy_temperature_env_vars("cimemories") == ()


def test_limit_seeds_keep_the_pre_rename_strings():
    for module, seed in (("wildbench", 'seed="lmsys_noncoding_hard"'),
                         ("loft_scifact", 'seed="loft_x_arxiv"'),
                         ("cimemories", 'seed="cimemories"'),
                         ("bfcl_live", 'limit_seed="bfcl_v3_live"')):
        assert seed in (SUITE_DIR / f"{module}.py").read_text(encoding="utf-8"), module


def test_run_eval_suite_limit_seed_draws_the_pre_rename_subset(monkeypatch):
    """bfcl_live's --eval-limit is applied by run_eval_suite: seeded on the old name it draws
    exactly the rows the same call drew as bfcl_v3_live."""
    from unittest import mock
    from gbench.runners.eval_suites.sampling import stratified_sample
    samples = [([{"role": "user", "content": f"q{i}"}], "g", {"category": f"c{i % 3}"})
               for i in range(60)]

    def key(s):
        return s[2]["category"]

    sent = []

    async def _reply(**kw):
        sent.append(kw["messages"][-1]["content"])
        return base.Reply("x", None, "stop", completion_tokens=1)

    def run(**extra):
        sent.clear()
        with mock.patch.object(base, "_send_single_request", _reply):
            base.run_eval_suite(eval_name="bfcl_live", model_name="m", base_url="http://x",
                                concurrency=1, samples=samples, eval_fn=lambda r, g: True,
                                limit=9, **extra)
        return sorted(sent)

    old = sorted(s[0][0]["content"] for s in stratified_sample(samples, 9, key, seed="bfcl_v3_live"))
    new = sorted(s[0][0]["content"] for s in stratified_sample(samples, 9, key, seed="bfcl_live"))
    assert old != new                                   # the seed matters for this set
    assert run(limit_seed="bfcl_v3_live") == old
    assert run() == new                                 # default: seeded on eval_name


def test_bfcl_live_passes_the_old_seed_and_runs_bfcls_live_group(monkeypatch):
    from gbench.runners.eval_suites import bfcl_live
    seen = {}

    def spy(**kw):
        seen.update(kw)
        return {"accuracy": 0.0, "category_accuracy": {}, "sample_traces": []}

    monkeypatch.setattr(bfcl_live, "_load_bfcl_live_samples", lambda: ([], {}))
    monkeypatch.setattr(bfcl_live, "run_eval_suite", spy)
    # The checker's image probe (a container of the locally built gbench-bfcl image) is not what
    # this test checks, and a CI runner has no such image: it answers as the pinned image does.
    probed = []
    monkeypatch.setattr(bfcl_live._bfcl, "_harness_probe",
                        lambda *a, **k: probed.append(a) or (bfcl_live._bfcl.BFCL_EVAL_RELEASE, []))
    bfcl_live.run_bfcl_live("m", "http://x", 1, limit=5)
    assert probed == [(bfcl_live.EVAL_NAME, [], True)]   # probed before anything is generated
    assert seen["eval_name"] == "bfcl_live" and seen["limit_seed"] == "bfcl_v3_live"
    mapping = pytest.importorskip("bfcl_eval.constants.category_mapping")
    assert set(mapping.TEST_COLLECTION_MAPPING["live"]) == set(bfcl_live.LIVE_CATEGORIES)


def test_loft_metric_and_docs_name_the_task_it_runs():
    from gbench.runners.eval_suites import loft_scifact
    src = (SUITE_DIR / "loft_scifact.py").read_text(encoding="utf-8")
    assert "the suite name is historical" not in src
    assert "retrieval_scifact" in src
    doc = (DOCS / "loft_scifact.md").read_text(encoding="utf-8")
    assert "The suite name is historical" not in doc and "loft_scifact" in doc
    assert loft_scifact._CONTEXT_LENGTH == "128k"


@pytest.mark.parametrize("name", sorted({**NEW_RENAMES, **REVERTED}.values()))
def test_renamed_docs_run_the_new_name_with_the_mandatory_budget(name):
    doc = (DOCS / f"{name}.md").read_text(encoding="utf-8")
    assert doc.splitlines()[0].startswith(f"# {name}")
    run = doc.split("## Run", 1)[1]
    assert f"--evals {name}" in run and "--max-output-tokens" in run
    assert all(ord(ch) < 128 for ch in doc), name


def test_catalogs_use_the_current_suite_ids():
    for rel in ("README.md", "docs/eval-suites.md", "docs/RESOURCES.md", "docs/THIRD_PARTY_DATASETS.md",
                "docs/evals_roadmap.md", "docs/evals/env-knobs.md"):
        text = (REPO / rel).read_text(encoding="utf-8")
        for proxy in PROXY_NAMES:
            assert proxy not in text, (rel, proxy)
        for old in NEW_RENAMES:
            rest = text.replace(f"renamed from `{old}`", "").replace(
                f"GBENCH_{old.upper()}_TEMPERATURE", "")
            assert f"`{old}`" not in rest, (rel, old)
    # The suite catalog moved from README.md to docs/eval-suites.md, so its
    # Setup Guide links are now relative to docs/ (evals/<name>.md).
    catalog = (REPO / "docs" / "eval-suites.md").read_text(encoding="utf-8")
    for name in (*NEW_RENAMES.values(), *REVERTED.values()):
        assert f"`{name}`" in catalog and f"](evals/{name}.md)" in catalog, name


# --------------------------------------------------------------------------- #
# canonical_sync
# --------------------------------------------------------------------------- #
def test_checksum_key_carries_full_values_that_match_the_modules():
    from gbench.runners.eval_suites import api_bank, imo_answer_bench, loft_scifact, scicode
    from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC as C
    expect = {
        "api_bank": "sha256:" + api_bank._CHECKOUT_DIGEST,
        "imo_answer_bench": "sha256:" + imo_answer_bench._CSV_SHA256,
        "loft_scifact": "sha256:" + loft_scifact._SCIFACT_SHA256,
        "scicode": "sha256:" + scicode._TEST_DATA_SHA256,
        "simpleqa": "sha256:feee3f7e7db3617e94e8fcf1977b756ec420ef8568f4e0fcbbe0e92e9d5fc032",
        "healthbench": "sha256:e99dd3c6372c10d6fcc5e385c5fae69d0dd40392dae56836ef9493ae324ecd2f",
    }
    nestful_src = (SUITE_DIR / "nestful.py").read_text(encoding="utf-8")
    assert '"nestful_data.jsonl": "ab49f52a8a11289247f7f4c1b5265c01d449c1aa"' in nestful_src
    expect["nestful"] = "git-blob:ab49f52a8a11289247f7f4c1b5265c01d449c1aa"
    for name, value in expect.items():
        assert C[name]["checksum"].split(" ")[0] == value, name
    for name, entry in C.items():
        # no truncated 12-hex digest folded into the method string any more
        assert "sha256 feee3f7e7db3 " not in entry["method"]
        assert "ab49f52a)" not in entry["method"] and "66978bd75f70 verified" not in entry["method"]
        assert set(entry) <= {"synced", "upstream", "revision", "dataset", "checksum",
                              "method"}, name


def test_swe_entries_record_the_module_pins():
    from gbench.runners.eval_suites import (copilot_bench_swe, multi_swe_bench, swe_bench_live,
                                            swe_bench_multilingual, swe_bench_pro)
    from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC as C
    for name, mod in (("copilot_bench_swe", copilot_bench_swe),
                      ("swe_bench_multilingual", swe_bench_multilingual),
                      ("swe_bench_live", swe_bench_live)):
        src = (SUITE_DIR / f"{name}.py").read_text(encoding="utf-8")
        rev = C[name]["dataset"].split("@", 1)[1].split(" ", 1)[0]
        assert len(rev) == 40 and rev in src, name
        assert "loaded by name" not in C[name]["dataset"], name
    assert swe_bench_pro._REVISION in C["swe_bench_pro"]["dataset"]
    assert "V2" in C["swe_bench_pro"]["dataset"] and "unpinned" not in C["swe_bench_pro"]["dataset"]
    assert multi_swe_bench._REVISION in C["multi_swe_bench"]["dataset"]
    assert "multi-swe-bench 1.1.2" in C["multi_swe_bench"]["upstream"]


def test_charxiv_enum_icl_is_upstreams_runtime_string():
    from gbench.runners.eval_suites.charxiv_constants import DESCRIPTIVE_GRADING_ICL
    from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC as C
    enum = DESCRIPTIVE_GRADING_ICL["enum"]
    assert len(enum) == 1064
    assert hashlib.sha256(enum.encode()).hexdigest() == (
        "6b73413c2f4f88a4e41689654753eb9ca612aa1c1f02eb8c0d9f064f2b820834")
    assert "\x07lpha" in enum and "\x08eta" in enum and "\\gamma" in enum
    assert "escape fix" not in C["charxiv"]["method"]


# --------------------------------------------------------------------------- #
# GEMINI_API_KEYS precedence
# --------------------------------------------------------------------------- #
def _keys_after_import(env):
    code = ("import os, gbench; from gbench.runners.eval_suites import base; "
            "print(os.environ.get('GEMINI_API_KEYS')); print(os.environ.get('GEMINI_API_KEY')); "
            "print(','.join(base.gemini_api_keys()))")
    clean = {k: v for k, v in os.environ.items() if k not in ("GEMINI_API_KEY", "GEMINI_API_KEYS")}
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env={**clean, **env},
                         capture_output=True, text=True, timeout=120, check=True).stdout
    return out.strip().splitlines()[-3:]


def test_gemini_api_keys_wins_when_both_are_set():
    assert _keys_after_import({"GEMINI_API_KEY": "a", "GEMINI_API_KEYS": "b; c"}) == ["b,c", "b", "b,c"]


def test_gemini_api_key_alone_and_keys_alone():
    assert _keys_after_import({"GEMINI_API_KEY": "a,b"}) == ["a,b", "a", "a,b"]
    assert _keys_after_import({"GEMINI_API_KEYS": "x;y"}) == ["x,y", "x", "x,y"]


def test_gemini_api_keys_reads_the_list_first(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "a")
    monkeypatch.setenv("GEMINI_API_KEYS", "b,c,b")
    assert base.gemini_api_keys() == ["b", "c"]
    monkeypatch.setenv("GEMINI_API_KEYS", " ")
    assert base.gemini_api_keys() == ["a"]


def test_docs_state_the_same_precedence():
    env_doc = (REPO / "docs" / "environment-variables.md").read_text(encoding="utf-8")
    row = next(l for l in env_doc.splitlines() if l.startswith("| `GEMINI_API_KEY` |"))
    assert "`GEMINI_API_KEYS` wins when both are set" in row and row.isascii()
    knobs = (DOCS / "env-knobs.md").read_text(encoding="utf-8")
    assert "wins over `GEMINI_API_KEY` when both are" in knobs
    for doc in DOCS.glob("*.md"):
        text = doc.read_text(encoding="utf-8")
        assert "GEMINI_API_KEY wins" not in text and "GEMINI_API_KEY is used" not in text, doc.name


# --------------------------------------------------------------------------- #
# beam_128k json_repair, metrics, deps, docs
# --------------------------------------------------------------------------- #
def test_beam_judge_reply_parse_uses_upstreams_json_repair_fallback():
    pytest.importorskip("json_repair")
    from gbench.runners.eval_suites import beam_128k as B
    assert B.parse_judge_reply('```json\n{"score": 0.5}\n```') == {"score": 0.5}
    assert B.parse_json_response('{"score": 1.0, "reason": "ok"') is None
    assert B.parse_judge_reply('{"score": 1.0, "reason": "ok"')["score"] == 1.0
    with pytest.raises((TypeError, KeyError, IndexError)):
        B.parse_judge_reply("no json here")["score"]


def test_beam_requires_json_repair(monkeypatch):
    # json_repair 0.44.1 is vendored: the installed package plays no part, and the check refuses
    # vendored files that are not the pinned release's.
    from gbench.runners.eval_suites import beam_128k as B
    monkeypatch.setitem(sys.modules, "json_repair", None)
    B._require_json_repair()
    monkeypatch.setitem(B.JSON_REPAIR_VENDOR_SHA256, "json_parser.py", "0" * 64)
    with pytest.raises(RuntimeError, match="json_repair"):
        B._require_json_repair()


def test_finalize_mean_metric_skips_scoring_excluded_traces():
    from gbench.runners.eval_suites.metrics import finalize_mean_metric, scorable_traces
    result = {"accuracy": 50.0, "sample_traces": [
        {"response_text": "a", "s": 1.0},
        {"response_text": "b", "scoring_excluded": True, "judge_grade": "JUDGE_FAILED"},
        {"response_text": None},
    ]}
    finalize_mean_metric(result, metric_name="m", score_key="s", failure_score=-1.0, scale=1.0)
    assert result["accuracy"] == 0.0 and result["scored_samples"] == 2   # (1.0 + -1.0) / 2
    assert [t["response_text"] for t in scorable_traces(result)] == ["a"]


def test_evals_extra_pins_the_harness_and_declares_suite_packages():
    with open(REPO / "pyproject.toml", "rb") as f:
        evals = tomllib.load(f)["project"]["optional-dependencies"]["evals"]
    for req in ("multi-swe-bench==1.1.2", "json_repair", "python-docx", "soundfile"):
        assert req in evals, req
    assert "multi-swe-bench" not in evals
    doc = (DOCS / "multi_swe_bench.md").read_text(encoding="utf-8")
    assert "multi-swe-bench==1.1.2" in doc and "24f493f8a103e72312ded4f6b9c89f081d69cb09" in doc


def test_shared_docs_follow_ups():
    knobs = (DOCS / "env-knobs.md").read_text(encoding="utf-8")
    for s in ("swebench_snapshots/", "GBENCH_SWE_BENCH_PRO_TRAJ_DIR", "GBENCH_GDPVAL_PDF_PAGE_DPI",
              "attack_model_label", "max_generated_tests"):
        assert s in knobs, s
    resources = (REPO / "docs" / "RESOURCES.md").read_text(encoding="utf-8")
    agentic = next(l for l in resources.splitlines() if l.startswith("* **BFCL v4 Agentic**"))
    assert "huggingface.co" not in agentic and "bfcl-eval==2026.3.23" in agentic
    assert "Srimadh/Scicode-test-data-h5" in resources
    ruler = (DOCS / "ruler.md").read_text(encoding="utf-8")
    assert "cap plus\na few tokens" not in ruler and "`vt`" in ruler
    notice = (REPO / "NOTICE").read_text(encoding="utf-8")
    for s in ("Vendored third-party code", "lcb.py", "multipl_e.py", "scicode.py",
              "bigcodebench.py", "28fef95e", "3025a531"):
        assert s in notice, s
    third = (REPO / "docs" / "THIRD_PARTY_DATASETS.md").read_text(encoding="utf-8")
    assert "Srimadh/Scicode-test-data-h5" in third


def test_eval_n_shot_help_states_the_accepted_ranges():
    from gbench import cli
    action = next(a for a in cli.create_parser()._actions if "--eval-n-shot" in a.option_strings)
    assert "gsm8k 0-8" in action.help and "0-5" in action.help
