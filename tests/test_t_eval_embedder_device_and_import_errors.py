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

"""t_eval: the embedder's device is recorded, and a setting whose embedder did not run on the GPU
(a CPU fallback, a forced CPU, no GPU) is named as not comparable; an installed but broken torch /
sentence-transformers is infra_required naming the failed import, not a crash; a file with no
sample in the shard is not blamed on the endpoint. No model is loaded; no network."""

import builtins
import sys
import types

import pytest

from gbench.runners.eval_suites import t_eval as T


def test_a_non_gpu_embedder_is_named():
    info = {"en": {"device": "cpu", "fallback": True, "forced": None},
            "zh": {"device": "cuda", "fallback": False, "forced": None}}
    [reason] = T.embedder_device_reasons(info)
    assert reason.startswith("the en setting's Plan/Reason embedder ran on cpu (the GPU load "
                             "failed, fallback)")
    assert "forced" not in reason
    assert "GBENCH_T_EVAL_DEVICE=cpu" in T.embedder_device_reasons(
        {"en": {"device": "cpu", "forced": "cpu"}})[0]
    assert "no GPU visible" in T.embedder_device_reasons({"en": {"device": "cpu"}})[0]
    assert T.embedder_device_reasons({}) == []


def test_the_cpu_fallback_is_recorded(monkeypatch):
    calls = []

    class _ST:
        def __init__(self, name, device, revision):
            calls.append(device)
            if device == "cuda":
                raise RuntimeError("CUDA out of memory")
    fake_st = types.ModuleType("sentence_transformers")
    fake_st.SentenceTransformer = _ST
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv("GBENCH_T_EVAL_DEVICE", raising=False)
    monkeypatch.setattr(T, "_SHARED_ST", {})
    monkeypatch.setattr(T, "_ST_INFO", {})
    T._shared_st("en")
    assert calls == ["cuda", "cpu"]
    assert T._ST_INFO["en"]["device"] == "cpu" and T._ST_INFO["en"]["fallback"] is True


def test_a_broken_install_is_infra_required_naming_the_import(monkeypatch):
    real_import = builtins.__import__

    def failing(name, globals=None, locals=None, fromlist=(), level=0):
        if "_teval_vendor" in name or (fromlist and "_teval_vendor" in (fromlist or ())) \
                or (level and name == "" and "_teval_vendor" in (fromlist or ())):
            raise OSError("libtorch_cuda.so: cannot open shared object file")
        return real_import(name, globals, locals, fromlist, level)
    monkeypatch.setattr(T, "_VENDOR", None)
    monkeypatch.setattr(builtins, "__import__", failing)
    with pytest.raises(Exception) as err:
        T._vendor()
    text = str(err.value)
    assert "[INFRA REQUIRED]" in text and "could not be imported" in text
    assert "OSError: libtorch_cuda.so" in text and "reinstall torch" in text


def test_an_empty_shard_file_is_not_blamed_on_the_endpoint():
    text = T._unscored_text("English", ["plan", "review"], ["plan"])
    assert "no sample of the English file(s) plan in this shard or limit" in text
    assert "no scored reply in the English file(s) review (every request got no reply" in text
    assert "harness errors" not in T._unscored_text("English", ["plan"], ["plan"])


def test_only_the_loaded_embedders_are_recorded(monkeypatch):
    monkeypatch.setattr(T, "_ST_INFO", {"en": {"device": "cpu"}, "zh": {"device": "cuda"}})
    monkeypatch.setattr(T, "_SHARED_ST", {"zh": object()})
    assert T.embedders_in_use() == {"zh": {"device": "cuda"}}
    monkeypatch.setattr(T, "_SHARED_ST", {})
    assert T.embedders_in_use() == {}
