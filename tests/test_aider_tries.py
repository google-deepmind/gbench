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

"""Aider Polyglot's two tries: pass@2 is delegated to aider's own benchmark.py `--tries 2` in the
aider-benchmark:5dc9490b container, and the headline is pass@2 (not pass@1) on a run where the two
differ. The build-command and result-parsing contract lives in
test_aider_polyglot_canonical.py."""

import json
import re
import types

import gbench.runners.eval_suites.aider_polyglot as A
import gbench.runners.eval_suites.tau_common as TC


def test_pass_at_2_is_delegated_to_aider_benchmark_tries_2():
    cmd = A._build_command("aider-benchmark:5dc9490b", "/bench", "gbench", "openai/m",
                           "http://h/v1", 4, A._LANGUAGES, "", None, "")
    joined = " ".join(cmd)
    assert "--tries 2" in joined, "pass@2 = aider benchmark.py --tries 2 (its native retry loop)"
    assert "benchmark.py" in joined


def test_headline_is_pass_at_2(monkeypatch, tmp_path):
    """A full run where a third of the exercises pass on the first try, a third only on the
    second and a third never: accuracy is pass@2 (66.67), pass@1 (33.33) is reported beside it."""
    monkeypatch.setenv("GBENCH_AIDER_BENCHMARK_DIR", str(tmp_path))
    monkeypatch.setenv("GBENCH_AIDER_MODEL", "openai/m")   # skip the served-id lookup
    monkeypatch.setattr(A, "_check_prereqs", lambda image, bench: None)
    monkeypatch.setattr(TC, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC.container_url(url))
    langs = ["cpp", "go", "java", "javascript", "python", "rust"]
    outcomes = ([True], [False, True], [False, False])

    def fake_run(cmd, **_kw):
        m = re.search(r"--num-tests (\d+)", " ".join(cmd))
        count = int(m.group(1)) if m else A._EXPECTED_FULL
        base = tmp_path / "2026-run--gbench"
        for i in range(count):
            d = base / langs[i % len(langs)] / "exercises" / "practice" / f"x{i}"
            d.mkdir(parents=True, exist_ok=True)
            (d / ".aider.results.json").write_text(json.dumps(
                {"language": langs[i % len(langs)], "tests_outcomes": outcomes[i % 3]}))
        report = {"files_loaded": ["/gbench-aider/model-settings.yml"], "extra_params": {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "timeout": A.model_request_timeout_s(), "max_retries": 0,
            "temperature": 0.0}}  # a no-think run's temperature (base.resolve_temperature)
        return types.SimpleNamespace(returncode=0, stderr="",
                                     stdout=A._CHECK_MARKER + json.dumps(report) + "\n"
                                     + A._MEMORY_MARKER + "12884901888\n" + A._OOM_MARKER + "0\n")

    monkeypatch.setattr(A.subprocess, "run", fake_run)
    r = A.run_aider_polyglot("m", "http://127.0.0.1:9/v1", concurrency=1)
    assert r["tries"] == 2
    assert r["pass_rate_2"] == round(100.0 * 150 / 225, 2)
    assert r["pass_rate_1"] == round(100.0 * 75 / 225, 2)
    assert r["accuracy"] == r["pass_rate_2"] != r["pass_rate_1"]
