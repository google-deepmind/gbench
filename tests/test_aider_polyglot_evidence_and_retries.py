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

"""aider_polyglot reads aider's chat history as aider writes it, and keeps its evidence.

* A test run's output block follows one of aider's own lines ("> " + message + two spaces); a
  fence after a blockquote line the model wrote is the model's text, never read as test output.
* benchmark.py raises aider's RETRY_TIMEOUT to 24 h, so a request is given up on after its last
  retry line, whatever the delay, and the cause is the last failure logged.
* The container reports the memory limit the kernel enforced; the result records it and a limit
  other than 12 GiB (or an unreadable one) makes the run not comparable.
* The run directory is recorded and handed to the host user; a model-id override is checked.

No network, no docker: the container is faked.
"""

import subprocess

import pytest

from gbench.runners.eval_suites import aider_polyglot as A

HEADER = "# aider chat started at 2026-09-29\n\n#### Fix the code.  \n"
RUSTUP = ("error: rustup could not choose a version of cargo to run, because one wasn't "
          "specified explicitly, and no default is configured.")
CONN = "litellm.APIConnectionError: APIConnectionError: OpenAIException - Connection error."
TOKENS = "Tokens: 1.6k sent, 2.5k received."


def _tool(*lines):
    return "".join(f"> {ln}  \n" for ln in lines)


def _reply(text):
    return "\n" + text.strip() + "\n\n"


def _tests(output):
    return f"```\n{output}\n```"


def test_a_fence_after_the_models_own_blockquote_is_not_test_output():
    reply = _reply("> note: you may see this\n```\n" + RUSTUP + "\n```\nDone.")
    history = HEADER + reply + _tool(TOKENS, "Applied edit to src/lib.rs") + _tests("ok")
    assert A.exercise_test_outputs(history) == ["ok"]
    assert A.exercise_harness_error(history) is None
    # aider's own line right before the fence still opens a test block
    assert A.exercise_harness_error(HEADER + _tool("Applied edit to x") + _tests(RUSTUP))[
        "kind"] == "toolchain"


def _retries(delays, error=CONN):
    return "".join(_tool(error) + _tool(f"Retrying in {d} seconds...") for d in delays)


DELAYS = ["0.2", "0.5", "1.0", "2.0", "4.0", "8.0", "16.0", "32.0", "64.0", "128.0"]


def test_retries_past_32_seconds_that_then_succeed_are_scored():
    history = HEADER + _retries(DELAYS) + _reply("fixed") + _tool(TOKENS) + _tests("ok")
    assert A.request_give_ups(history) == []
    assert A.exercise_harness_error(history) is None


def test_a_request_is_given_up_after_its_last_retry_whatever_the_delay():
    # the run ended while aider was still retrying (no reply, no final failure logged)
    history = HEADER + _retries(DELAYS) + _tests("FAILED")
    (give_up,) = A.request_give_ups(history)
    assert give_up["cause"] == "APIConnectionError" and give_up["endpoint"] is True
    assert give_up["detail"] == f"{CONN} (Retrying in 128.0 seconds...)"
    # the final attempt's failure, logged after the last retry, is the cause
    rate = "litellm.RateLimitError: RateLimitError: OpenAIException - slow down"
    history = HEADER + _retries(DELAYS) + _tool(rate) + _tests("FAILED")
    (give_up,) = A.request_give_ups(history)
    assert give_up["cause"] == "RateLimitError" and give_up["detail"].startswith(rate)
    assert A.exercise_harness_error(history)["kind"] == "model_endpoint"


def test_an_unread_oom_count_still_names_the_threads():
    reason = A._oom_reason(None, 16)
    assert "could not be read" in reason
    assert "shared by the 16 exercise(s) benchmark.py runs at once (--threads)" in reason
    assert "more than the 10 of upstream's documented run" in reason
    assert "more than the 10" not in A._oom_reason(None, 4)


def test_the_report_prints_the_enforced_limit_before_the_oom_count():
    out = subprocess.run(["bash", "-c", "(exit 2)" + A._OOM_REPORT], capture_output=True,
                         text=True, timeout=30)
    assert out.returncode == 2
    lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
    assert lines[-1].startswith(A._OOM_MARKER) and lines[-2].startswith(A._MEMORY_MARKER)
    value = lines[-2][len(A._MEMORY_MARKER):].strip()
    parsed = A.container_memory_limit(out.stdout)
    assert parsed == (int(value) if value.isdigit() else (value or None))


@pytest.mark.parametrize("stdout, expected", [
    (A._MEMORY_MARKER + "12884901888\n" + A._OOM_MARKER + "0\n", 12884901888),
    (A._MEMORY_MARKER + "max\n", "max"),
    (A._MEMORY_MARKER + "\n", None),
    ("", None),
])
def test_the_enforced_limit_is_parsed(stdout, expected):
    assert A.container_memory_limit(stdout) == expected


def test_a_limit_other_than_upstreams_makes_the_run_not_comparable():
    assert A._memory_reason(12884901888) is None
    assert "not the 12g" in A._memory_reason("max")
    assert "could not be read" in A._memory_reason(None)


def test_the_run_directory_goes_back_to_the_host_user_and_keeps_the_exit_status():
    cmd = A._build_command("aider-benchmark:5dc9490b", "/b", "gbench", "openai/m",
                           "http://host.docker.internal:8000/v1", 4, ["python"], "", None, "",
                           settings_dir="/s", owner="1000:1001")
    script = cmd[-1]
    assert "chown -R 1000:1001" in script and "ls -td /benchmarks/*--gbench" in script
    assert script.endswith(A._OOM_REPORT)
    # no /benchmarks here: nothing to hand over, and the status is still benchmark.py's
    handover = script[script.index("; rc=$?; d="):script.index(A._OOM_REPORT)]
    out = subprocess.run(["bash", "-c", "(exit 5)" + handover + "; echo status=$?"],
                         capture_output=True, text=True, timeout=30)
    assert "status=5" in out.stdout
    # without an owner nothing is handed over
    plain = A._build_command("aider-benchmark:5dc9490b", "/b", "gbench", "openai/m",
                             "http://x/v1", 4, ["python"], "", None, "", settings_dir="/s")[-1]
    assert "chown" not in plain


def test_a_model_override_must_be_a_served_openai_id(monkeypatch):
    from gbench.utils import endpoint as EP
    monkeypatch.setenv("GBENCH_AIDER_MODEL", "anthropic/claude")
    with pytest.raises(RuntimeError, match="must be openai/<served id>"):
        A._litellm_model("http://h/v1", "m")
    monkeypatch.setenv("GBENCH_AIDER_MODEL", "openai/other")

    def refuse(url, candidates, override=None, **kw):
        raise EP.EndpointError("serves only m")
    monkeypatch.setattr(EP, "resolve_served_model", refuse)
    with pytest.raises(RuntimeError, match="names a model the endpoint does not serve"):
        A._litellm_model("http://h/v1", "m")
    monkeypatch.setattr(EP, "resolve_served_model",
                        lambda url, c, o=None, **kw: EP.ServedModel("other", "listed", None))
    assert A._litellm_model("http://h/v1", "m") == ("openai/other",
                                                    "GBENCH_AIDER_MODEL (listed)")
    # the served id gbench passed needs no lookup
    monkeypatch.setattr(EP, "resolve_served_model", refuse)
    monkeypatch.setenv("GBENCH_AIDER_MODEL", "openai/m")
    assert A._litellm_model("http://h/v1", "m")[0] == "openai/m"


def test_the_result_records_the_run_directory_and_the_enforced_limit(monkeypatch, tmp_path):
    import json
    import types
    from gbench.runners.eval_suites import tau_common as TC
    langs = A._LANGUAGES

    def fake_run(cmd, **_kw):
        fake_run.cmd = cmd
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
            A._CHECK_MARKER + json.dumps(report) + "\n" + A._MEMORY_MARKER + "max\n"
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
    assert r["aider_run_dir"] == str(tmp_path / "2026-run1--gbench")
    assert r["request_model_id"] == "m"
    assert r["container_memory_limit"]["enforced_bytes"] == "max"
    assert r["leaderboard_comparable"] is False
    assert "ran under a memory limit of max" in r["leaderboard_comparable_reason"]
    assert "chown -R" in fake_run.cmd[-1]
