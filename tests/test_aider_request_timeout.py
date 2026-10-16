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

"""aider_polyglot sizes aider's request timeout for a reply to the end of the served window.

aider gives each request 600 s and sends no max_tokens. At temperature 0 a reply longer than that
timed out on every retry, and benchmark.py retries for 24 h, so the run hung (2026-10-06 smoke
test: two exercises, 12 timeouts each, 6 h 40 min). gbench now passes base's timeout for a request
without max_tokens in the settings file's extra_params, still sends no max_tokens, and treats a
request that times out anyway as an endpoint failure.

No network, no docker: the container is faked.
"""

import json

from gbench.runners.eval_suites import aider_polyglot as A
from gbench.runners.eval_suites import base

HEADER = "# aider chat started at 2026-10-06\n\n#### Fix the code.  \n"
TIMEOUT = ("litellm.Timeout: APITimeoutError - Request timed out. Error_str: Request timed out.")


def _tool(*lines):
    return "".join(f"> {ln}  \n" for ln in lines)


def test_the_timeout_covers_the_served_window(monkeypatch):
    monkeypatch.setitem(base._RUN_KNOBS, "max_model_len", 131072)
    assert A.model_request_timeout_s() == base.request_timeout_s(131072)
    assert A.model_request_timeout_s() > 600


def test_without_a_reported_window_the_timeout_assumes_the_uncapped_length(monkeypatch):
    monkeypatch.delitem(base._RUN_KNOBS, "max_model_len", raising=False)
    assert A.model_request_timeout_s() == base.request_timeout_s(base.UNCAPPED_TIMEOUT_TOKENS)


def test_the_settings_file_carries_the_timeout_and_no_max_tokens(tmp_path, monkeypatch):
    monkeypatch.setitem(base._RUN_KNOBS, "max_model_len", 262144)
    returned = A._write_model_settings(str(tmp_path), True)
    (entry,) = json.loads((tmp_path / A._SETTINGS_FILE).read_text())
    assert entry["name"] == "aider/extra_params"
    assert entry["extra_params"] == returned
    assert returned["timeout"] == base.request_timeout_s(262144)
    assert returned["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert returned["max_retries"] == 0
    assert "max_tokens" not in returned
    assert A._write_model_settings(str(tmp_path), False, timeout_s=1234)["timeout"] == 1234


def test_a_request_that_still_times_out_is_an_endpoint_failure():
    retries = "".join(_tool(TIMEOUT) + _tool(f"Retrying in {d} seconds...")
                      for d in ("0.2", "0.5", "1.0"))
    history = HEADER + retries + _tool(TIMEOUT) + "```\nFAILED\n```"
    (give_up,) = A.request_give_ups(history)
    assert give_up["cause"] == "Timeout" and give_up["endpoint"] is True
    assert A.exercise_harness_error(history)["kind"] == "model_endpoint"


def test_the_preflight_must_confirm_the_timeout(monkeypatch, tmp_path):
    """A container whose aider did not take the timeout (it reports extra_params without it)
    stops the run before any result is read."""
    import types
    from gbench.runners.eval_suites import tau_common as TC

    def fake_run(cmd, **_kw):
        report = {"files_loaded": ["/gbench-aider/model-settings.yml"], "extra_params": {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}}
        return types.SimpleNamespace(returncode=0, stderr="",
                                     stdout=A._CHECK_MARKER + json.dumps(report) + "\n")
    for name in ("GBENCH_AIDER_LANGUAGES", "GBENCH_AIDER_MODEL", "GBENCH_AIDER_EDIT_FORMAT",
                 "GBENCH_AIDER_EXTRA_ARGS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GBENCH_AIDER_BENCHMARK_DIR", str(tmp_path))
    monkeypatch.setattr(A, "_check_prereqs", lambda image, bench: None)
    monkeypatch.setattr(TC, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC.container_url(url))
    monkeypatch.setattr(A.subprocess, "run", fake_run)
    try:
        A.run_aider_polyglot("m", "http://127.0.0.1:9/v1", concurrency=2)
    except RuntimeError as e:
        assert "did not confirm that it loaded gbench's model settings" in str(e)
    else:
        raise AssertionError("the run went on without the timeout confirmed")


def test_the_result_records_the_timeout(monkeypatch, tmp_path):
    import types
    from gbench.runners.eval_suites import tau_common as TC
    langs = A._LANGUAGES
    monkeypatch.setitem(base._RUN_KNOBS, "max_model_len", 262144)

    def fake_run(cmd, **_kw):
        root = tmp_path / "2026-run1--gbench"
        for i in range(A._EXPECTED_FULL):
            d = root / langs[i % len(langs)] / "exercises" / "practice" / f"x{i}"
            d.mkdir(parents=True, exist_ok=True)
            (d / ".aider.results.json").write_text(json.dumps(
                {"language": langs[i % len(langs)], "testcase": f"x{i}",
                 "tests_outcomes": [True]}))
        report = {"files_loaded": ["/gbench-aider/model-settings.yml"], "extra_params": {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "timeout": A.model_request_timeout_s(), "max_retries": 0,
            "temperature": 0.0}}  # a no-think run's temperature (base.resolve_temperature)
        return types.SimpleNamespace(returncode=0, stderr="", stdout=(
            A._CHECK_MARKER + json.dumps(report) + "\n" + A._MEMORY_MARKER + "12884901888\n"
            + A._OOM_MARKER + "0\n"))
    for name in ("GBENCH_AIDER_LANGUAGES", "GBENCH_AIDER_MODEL", "GBENCH_AIDER_EDIT_FORMAT",
                 "GBENCH_AIDER_EXTRA_ARGS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GBENCH_AIDER_BENCHMARK_DIR", str(tmp_path))
    monkeypatch.setattr(A, "_check_prereqs", lambda image, bench: None)
    monkeypatch.setattr(TC, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC.container_url(url))
    monkeypatch.setattr(A.subprocess, "run", fake_run)
    r = A.run_aider_polyglot("m", "http://127.0.0.1:9/v1", concurrency=2)
    assert r["model_request_timeout_s"] == base.request_timeout_s(262144)
    assert "max_tokens" not in r["model_extra_params"]
