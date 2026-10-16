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

"""The tau container never fetches LiteLLM's model cost map from the network."""
from gbench.runners.eval_suites import tau_common as TC


def test_the_tau_container_uses_litellms_bundled_cost_map(tmp_path):
    cmd = TC._docker_run_command("n", "gbench-tau2:x", ["retail"], str(tmp_path / "io"),
                                 str(tmp_path / "cache"), None, [])
    pairs = [cmd[i + 1] for i, a in enumerate(cmd[:-1]) if a == "-e"]
    assert "LITELLM_LOCAL_MODEL_COST_MAP=True" in pairs


def test_the_doc_says_why(tmp_path):
    from pathlib import Path
    doc = (Path(__file__).resolve().parents[1] / "docs" / "evals" / "tau2.md").read_text()
    assert "LITELLM_LOCAL_MODEL_COST_MAP=True" in doc
