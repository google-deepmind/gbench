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

"""api_bank's ToolSearcher encoder loads on the CPU, not on the GPU the model server holds.

Upstream's apis/tool_search.py calls SentenceTransformer('sentence-transformers/paraphrase-
MiniLM-L3-v2') with no device, which takes cuda:0 when a GPU is visible. On 2026-10-10 the
model server held all four GPUs and api_bank failed (CUDA out of memory) before its first
request. Inside _in_checkout the class now defaults to TOOL_SEARCHER_DEVICE; outside it the
package holds its own class again.

sentence_transformers is faked: no model, no GPU. The real upstream ToolSearcher was run the
same way with all GPUs full (75 encodes, all on cpu).
"""

import sys
import types

from gbench.runners.eval_suites import api_bank as A


def _fake_st(monkeypatch):
    class SentenceTransformer:
        def __init__(self, name, device=None):
            self.name, self.device = name, device
    st = types.ModuleType("sentence_transformers")
    st.SentenceTransformer = SentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)
    return st, SentenceTransformer


def test_the_checkout_encoder_defaults_to_the_cpu_and_is_restored(monkeypatch, tmp_path):
    st, real = _fake_st(monkeypatch)
    with A._in_checkout(str(tmp_path)):
        model = st.SentenceTransformer(A.TOOL_SEARCHER_MODEL)
        assert model.device == A.TOOL_SEARCHER_DEVICE == "cpu"
        assert isinstance(model, real)
        assert st.SentenceTransformer(A.TOOL_SEARCHER_MODEL, device="cuda:1").device == "cuda:1"
        with A._in_checkout(str(tmp_path)):                  # nested: still one subclass level
            assert st.SentenceTransformer.__mro__[1] is real
    assert st.SentenceTransformer is real


def test_an_already_imported_tool_search_module_is_rebound(monkeypatch, tmp_path):
    st, real = _fake_st(monkeypatch)
    for name in ("apis.tool_search", "lv3_apis.tool_search"):
        mod = types.ModuleType(name)
        mod.SentenceTransformer = real                       # bound by an earlier import
        monkeypatch.setitem(sys.modules, name, mod)
    with A._in_checkout(str(tmp_path)):
        for name in ("apis.tool_search", "lv3_apis.tool_search"):
            assert sys.modules[name].SentenceTransformer("m").device == "cpu"


def test_the_prerequisite_probe_loads_on_the_cpu(monkeypatch):
    st, real = _fake_st(monkeypatch)
    loaded = []
    monkeypatch.setattr(real, "__init__", lambda self, name, device=None: loaded.append(device))
    monkeypatch.setattr(A, "check_translate_backend", lambda: {})
    A.check_backend_prerequisites("unused")
    assert loaded == ["cpu"]
