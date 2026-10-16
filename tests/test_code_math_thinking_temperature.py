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

"""The harness-owned paths of the code and math suites sample the model under test at gbench's
temperature (base.resolve_temperature): 1.0 with --thinking, 0.0 without, unless --temperature or
GBENCH_<SUITE>_TEMPERATURE sets one, which reaches the request in both modes.

* aider_polyglot: aider's benchmark sets temperature 0 itself (Model.send_completion); the run's
  temperature goes in the aider/extra_params settings file in both modes, which send_completion
  applies with kwargs.update after its own temperature. The fake container below echoes the file
  gbench mounted as aider's pre-flight would, so the value checked is the one written.
* livebench: the run's temperature is written into the one-off model config in both modes (never
  --force-temperature, which stops LiveBench on a question with a required temperature).
* scicode: a sub-step reply cut off while looping in its reasoning is counted as a loop
  (base.looping_channel), as base's classify_reply counts it.
"""

import json
import types

import pytest

from gbench.runners.eval_suites import aider_polyglot as A
from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import livebench as LB
from gbench.runners.eval_suites import scicode as SC
from gbench.runners.eval_suites import tau_common as TC_GUARD


# --------------------------------------------------------------------------- #
# aider_polyglot                                                               #
# --------------------------------------------------------------------------- #
def test_aider_settings_carry_the_given_temperature(tmp_path):
    thinking = A._write_model_settings(str(tmp_path), True, timeout_s=100, temperature=1.0)
    assert thinking["temperature"] == 1.0
    written = json.loads((tmp_path / A._SETTINGS_FILE).read_text())
    assert written == [{"name": "aider/extra_params", "extra_params": thinking}]
    nothink = A._write_model_settings(str(tmp_path), False, timeout_s=100, temperature=0.0)
    assert nothink["temperature"] == 0.0
    assert json.loads((tmp_path / A._SETTINGS_FILE).read_text())[0]["extra_params"][
        "temperature"] == 0.0


def _aider_run(monkeypatch, tmp_path, thinking, env_temperature=None):
    monkeypatch.setenv("GBENCH_AIDER_BENCHMARK_DIR", str(tmp_path))
    monkeypatch.setenv("GBENCH_AIDER_MODEL", "openai/m")
    for var in ("GBENCH_AIDER_LANGUAGES", "GBENCH_AIDER_EDIT_FORMAT", "GBENCH_AIDER_EXTRA_ARGS",
                "GBENCH_AIDER_POLYGLOT_TEMPERATURE"):
        monkeypatch.delenv(var, raising=False)
    if env_temperature is not None:
        monkeypatch.setenv("GBENCH_AIDER_POLYGLOT_TEMPERATURE", env_temperature)
    monkeypatch.setattr(A, "_check_prereqs", lambda image, bench: None)
    monkeypatch.setattr(TC_GUARD, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC_GUARD.container_url(url))
    seen = {}

    def fake_run(cmd, **_kw):
        # the settings file gbench mounted read-only for aider, read back as aider's pre-flight
        # (models.register_models, then Model(...).extra_params) reports it
        mount = next(v for v in cmd if isinstance(v, str) and v.endswith(f":{A._SETTINGS_MOUNT}:ro"))
        settings_dir = mount[:-len(f":{A._SETTINGS_MOUNT}:ro")]
        with open(f"{settings_dir}/{A._SETTINGS_FILE}", encoding="utf-8") as f:
            extra = json.load(f)[0]["extra_params"]
        seen["extra_params"] = extra
        d = tmp_path / "2026-run--gbench" / "python" / "exercises" / "practice" / "x0"
        d.mkdir(parents=True, exist_ok=True)
        (d / ".aider.results.json").write_text(json.dumps({"language": "python",
                                                           "tests_outcomes": [True]}))
        report = {"files_loaded": [f"{A._SETTINGS_MOUNT}/{A._SETTINGS_FILE}"],
                  "extra_params": extra}
        return types.SimpleNamespace(returncode=0, stderr="",
                                     stdout=A._CHECK_MARKER + json.dumps(report) + "\n"
                                     + A._MEMORY_MARKER + "12884901888\n" + A._OOM_MARKER + "0\n")

    monkeypatch.setattr(A.subprocess, "run", fake_run)
    (tmp_path / "polyglot-benchmark" / "python" / "exercises" / "practice" / "x0").mkdir(
        parents=True, exist_ok=True)
    r = A.run_aider_polyglot("m", "http://127.0.0.1:1/v1", concurrency=1, limit=1,
                             enable_thinking=thinking)
    return r, seen["extra_params"]


def test_aider_thinking_run_sends_temperature_one_in_extra_params(monkeypatch, tmp_path):
    r, extra = _aider_run(monkeypatch, tmp_path, thinking=True)
    assert extra["temperature"] == 1.0
    assert extra["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert r["model_extra_params"]["temperature"] == 1.0
    assert (r["temperature"], r["temperature_source"]) == (1.0, "thinking")
    assert r["temperature_applied"] is True and "temperature_note" not in r
    rec = r["official_sampling"]
    assert rec["temperature"] == 0.0 and rec["temperature_matches_official"] is False
    assert "temperature 1.0 (thinking)" in r["leaderboard_comparable_reason"]
    assert "aider's benchmark samples at temperature 0" in r["leaderboard_comparable_reason"]


def test_aider_thinking_run_honours_the_operators_temperature(monkeypatch, tmp_path):
    r, extra = _aider_run(monkeypatch, tmp_path, thinking=True, env_temperature="0.7")
    assert extra["temperature"] == 0.7
    assert (r["temperature"], r["temperature_source"]) == (
        0.7, "env:GBENCH_AIDER_POLYGLOT_TEMPERATURE")


def test_aider_no_think_run_sends_temperature_zero_in_extra_params(monkeypatch, tmp_path):
    r, extra = _aider_run(monkeypatch, tmp_path, thinking=False)
    assert extra["temperature"] == 0.0
    assert extra["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert r["model_extra_params"]["temperature"] == 0.0
    assert (r["temperature"], r["temperature_source"]) == (0.0, "default")
    assert r["temperature_applied"] is True and "temperature_note" not in r
    assert r["official_sampling"]["temperature_matches_official"] is True
    assert "temperature" not in str(r.get("leaderboard_comparable_reason"))


def test_aider_no_think_run_honours_the_operators_temperature(monkeypatch, tmp_path):
    base.set_run_knobs(temperature=0.5)
    r, extra = _aider_run(monkeypatch, tmp_path, thinking=False)
    assert extra["temperature"] == 0.5
    assert (r["temperature"], r["temperature_source"]) == (0.5, "cli:--temperature")
    assert "temperature 0.5 (cli:--temperature)" in r["leaderboard_comparable_reason"]


# --------------------------------------------------------------------------- #
# livebench                                                                    #
# --------------------------------------------------------------------------- #
def _livebench(monkeypatch, tmp_path, **kwargs):
    monkeypatch.setenv("GBENCH_LIVEBENCH_RELEASE", "2025-04-02")
    for var in ("GBENCH_LIVEBENCH_CATEGORIES", "GBENCH_LIVEBENCH_EXTRA_ARGS",
                "GBENCH_LIVEBENCH_TEMPERATURE", "GBENCH_LIVEBENCH_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(LB, "_check_prereqs", lambda image: None)
    monkeypatch.setattr(TC_GUARD, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC_GUARD.container_url(url))
    monkeypatch.setattr(LB, "upstream_model_config",
                        lambda image, names: {"name": None, "config": None,
                                              "commit": LB.LIVEBENCH_COMMIT})
    monkeypatch.setattr(LB.tempfile, "mkdtemp", lambda prefix="": str(tmp_path))
    seen = {}

    def fake_run(cmd, **_kw):
        seen["inner"] = cmd[-1]
        cols = LB._CATEGORIES
        (tmp_path / "all_groups.csv").write_text(
            "model,average," + ",".join(cols) + "\ngoogle-gemma-4-31b-it,50.0,"
            + ",".join("50.0" for _ in cols) + "\n", encoding="utf-8")
        for cat in ("math", "instruction_following"):
            d = tmp_path / "model_answer" / "data" / "live_bench" / cat / "x" / "model_answer"
            d.mkdir(parents=True, exist_ok=True)
            (d / "google-gemma-4-31b-it.jsonl").write_text(json.dumps(
                {"question_id": "q", "choices": [{"index": 0, "turns": ["ok"]}],
                 "total_output_tokens": 10}) + "\n", encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(LB.subprocess, "run", fake_run)
    base.set_run_knobs(max_output_tokens=32768)
    r = LB.run_livebench("google/gemma-4-31B-it", "http://h/v1", 4, max_output_tokens=32768,
                         **kwargs)
    cfg = json.load(open(tmp_path / LB._MODEL_CONFIG_FILE))
    return r, seen["inner"], cfg["api_kwargs"]["default"]


def test_livebench_thinking_temperature_goes_into_the_config(monkeypatch, tmp_path):
    r, inner, default = _livebench(monkeypatch, tmp_path, enable_thinking=True)
    assert default["temperature"] == 1.0
    assert "--force-temperature" not in inner
    assert (r["temperature"], r["temperature_source"]) == (1.0, "thinking")


def test_livebench_no_think_temperature_zero_goes_into_the_config(monkeypatch, tmp_path):
    r, inner, default = _livebench(monkeypatch, tmp_path)
    assert default["temperature"] == 0.0
    assert "--force-temperature" not in inner
    assert (r["temperature"], r["temperature_source"]) == (0.0, "default")
    assert r["official_sampling"]["send_temperature"] is False


def test_livebench_no_think_operator_temperature_goes_into_the_config(monkeypatch, tmp_path):
    base.set_run_knobs(temperature=0.4)
    r, inner, default = _livebench(monkeypatch, tmp_path)
    assert default["temperature"] == 0.4
    assert "--force-temperature" not in inner
    assert (r["temperature"], r["temperature_source"]) == (0.4, "cli:--temperature")


# --------------------------------------------------------------------------- #
# scicode                                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("content,reasoning,loops", [
    ("", " ".join(["the same twenty words go round and round again"] * 40), True),
    (" ".join(["the same twenty words go round and round again"] * 40), "", True),
    ("def f():\n    return 1\n", "a short, finished thought", False),
])
def test_scicode_counts_a_loop_in_either_channel(content, reasoning, loops):
    assert SC._loops(content, reasoning) is loops
