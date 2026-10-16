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

"""gaia's live Gemini key gate, putnam's key source, and t_eval's lazy evaluator import.

No network: the Gemini gate is stubbed, and the networkx-free import of the eval package runs in
a subprocess whose import system refuses networkx.
"""

import inspect
import os
import subprocess
import sys
import textwrap

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import gaia as GAIA
from gbench.runners.eval_suites import putnam as P
from gbench.runners.eval_suites import t_eval as T

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _StopError(Exception):
    """Raised by a stub to end a run once the code under test got past the gate."""


# --------------------------------------------------------------------------- #
# gaia: presence check, then a live check of the key and the cascade models
# --------------------------------------------------------------------------- #

def _gaia_gate(monkeypatch, verdict):
    """Configure a key, stub the HF check and the live gate (recording the key it was given), and
    make the dataset load raise _StopError so a run that passes the gate stops there."""
    monkeypatch.delenv("GBENCH_GAIA_SPLIT", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEYS", "AIzaFAKEKEY000000000001,AIzaFAKEKEY000000000002")
    monkeypatch.setattr(GAIA, "check_gaia_prerequisites", lambda: (True, ""))
    seen = {"keys": [], "loaded": 0}

    def gate(key):
        seen["keys"].append(key)
        return verdict

    def load(*a, **k):
        seen["loaded"] += 1
        raise _StopError()

    monkeypatch.setattr(GAIA, "gemini_key_live_valid", gate)
    monkeypatch.setattr(GAIA, "_load_gaia_samples", load)
    return seen


def test_gaia_rejected_key_is_infra_required_before_the_download(monkeypatch):
    seen = _gaia_gate(monkeypatch, (False, "HTTP 401"))
    with pytest.raises(RuntimeError) as exc:
        GAIA.run_gaia("m", "http://x", concurrency=1)
    msg = str(exc.value)
    assert "INFRA REQUIRED" in msg and "gaia" in msg
    assert "rejected" in msg and "HTTP 401" in msg
    assert "docs/evals/gaia.md" in msg
    # every configured key is offered to the gate (it passes when any one is live) ...
    assert seen["keys"] == ["AIzaFAKEKEY000000000001,AIzaFAKEKEY000000000002"]
    # ... and nothing was downloaded or generated
    assert seen["loaded"] == 0


def test_gaia_cascade_model_not_served_names_the_model_and_knob(monkeypatch):
    why = (f"{base.MODELS_NOT_SERVED_PREFIX} gemini-9.9-flash (from GBENCH_JUDGE_MODELS) to "
           "this key")
    seen = _gaia_gate(monkeypatch, (False, why))
    with pytest.raises(RuntimeError) as exc:
        GAIA.run_gaia("m", "http://x", concurrency=1)
    msg = str(exc.value)
    assert "gemini-9.9-flash" in msg and "GBENCH_JUDGE_MODELS" in msg
    assert "every model of the Gemini cascade" in msg
    assert "rejected" not in msg
    assert seen["loaded"] == 0


@pytest.mark.parametrize("verdict", [(True, ""), (True, "inconclusive URLError")])
def test_gaia_live_or_inconclusive_gate_lets_the_run_start(monkeypatch, verdict):
    seen = _gaia_gate(monkeypatch, verdict)
    with pytest.raises(_StopError):
        GAIA.run_gaia("m", "http://x", concurrency=1)
    assert seen["loaded"] == 1 and len(seen["keys"]) == 1


def test_gaia_test_split_goes_through_the_same_gate(monkeypatch):
    seen = _gaia_gate(monkeypatch, (False, "HTTP 403"))
    monkeypatch.setenv("GBENCH_GAIA_SPLIT", "test")
    with pytest.raises(RuntimeError, match="HTTP 403"):
        GAIA.run_gaia("m", "http://x", concurrency=1)
    assert seen["loaded"] == 0


def test_gaia_without_a_key_fails_on_presence_without_a_live_probe(monkeypatch):
    seen = _gaia_gate(monkeypatch, (True, ""))
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEYS or GEMINI_API_KEY"):
        GAIA.run_gaia("m", "http://x", concurrency=1)
    assert seen["keys"] == [] and seen["loaded"] == 0


def test_gaia_doc_describes_the_live_check():
    with open(os.path.join(REPO, "docs", "evals", "gaia.md"), encoding="utf-8") as fh:
        doc = fh.read()
    assert "base.gemini_key_live_valid" in doc
    assert "an inconclusive check" in doc


# --------------------------------------------------------------------------- #
# putnam: the gate reads the key only where the judge reads it
# --------------------------------------------------------------------------- #

def test_putnam_prerequisites_take_no_key_argument():
    assert list(inspect.signature(P.check_putnam_prerequisites).parameters) == []


def test_putnam_gate_ignores_a_gemini_api_key_kwarg_and_checks_the_env_keys(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEYS", "AIzaFAKEKEY000000000003")
    seen = []
    monkeypatch.setattr(P, "gemini_key_live_valid", lambda k: seen.append(k) or (True, ""))

    def load(*a, **k):
        raise _StopError()

    monkeypatch.setattr(P, "_load_putnam_samples", load)
    with pytest.raises(_StopError):
        P.run_putnam("m", "http://x", 1, gemini_api_key="AIzaSOMEOTHERKEY0000000")
    # the gate validated the key the judge uses (gemini_api_keys), not the kwarg
    assert seen == ["AIzaFAKEKEY000000000003"]


def test_putnam_gate_without_env_key_fails_even_if_a_kwarg_is_passed(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setattr(P, "gemini_key_live_valid",
                        lambda k: pytest.fail("no key: the live gate must not be reached"))
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        P.run_putnam("m", "http://x", 1, gemini_api_key="AIzaSOMEOTHERKEY0000000")


# --------------------------------------------------------------------------- #
# t_eval: the vendored evaluators (networkx, sentence-transformers) load only when it runs
# --------------------------------------------------------------------------- #

_BLOCK_NETWORKX = textwrap.dedent("""
    import importlib.abc, sys

    class _Refuse(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "networkx" or name.startswith("networkx."):
                raise ModuleNotFoundError("No module named %r" % name, name=name)
            return None

    sys.meta_path.insert(0, _Refuse())
""")


def _run_py(code, tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-c", code], cwd=str(tmp_path), env=env,
                          capture_output=True, text=True, timeout=600)


def test_eval_package_and_list_evals_work_without_networkx(tmp_path):
    code = _BLOCK_NETWORKX + textwrap.dedent("""
        import gbench.runners.eval_suites as es
        assert "t_eval" in es.SUITES, sorted(es.SUITES)
        assert "networkx" not in sys.modules
        from gbench import cli
        try:
            rc = cli.main(["--list", "evals"])
        except SystemExit as e:
            rc = e.code
        print("RC=%r" % (rc,))
        print("NETWORKX_LOADED=%r" % ("networkx" in sys.modules,))
        from gbench.runners.eval_suites import t_eval as T
        print("MISSING=%r" % (T.missing_packages(),))
    """)
    proc = _run_py(code, tmp_path)
    assert proc.returncode == 0, proc.stderr[-4000:]
    out = proc.stdout
    assert "RC=0" in out or "RC=None" in out, out[-2000:]
    assert "t_eval" in out
    assert "NETWORKX_LOADED=False" in out
    assert "MISSING=['networkx']" in out


def test_run_t_eval_without_networkx_is_infra_required_before_any_request(monkeypatch):
    monkeypatch.setitem(sys.modules, "networkx", None)
    monkeypatch.setattr(T, "run_eval_suite",
                        lambda **kw: pytest.fail("no request may be sent without networkx"))
    monkeypatch.setattr(T, "_load_file_samples",
                        lambda *a, **k: pytest.fail("no data may be loaded without networkx"))
    assert T.missing_packages() == ["networkx"] or "networkx" in T.missing_packages()
    with pytest.raises(RuntimeError) as exc:
        T.run_t_eval("m", "http://x", 1)
    msg = str(exc.value)
    assert "INFRA REQUIRED" in msg and "t_eval" in msg
    assert "requires networkx" in msg and "docs/evals/t_eval.md" in msg


def test_run_t_eval_without_sentence_transformers_names_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    monkeypatch.setattr(T, "run_eval_suite",
                        lambda **kw: pytest.fail("no request may be sent without the embedder"))
    with pytest.raises(RuntimeError, match="requires sentence-transformers"):
        T.run_t_eval("m", "http://x", 1)


def test_vendor_import_failure_is_infra_required_naming_the_module(monkeypatch):
    # A fresh import of the vendored package with networkx refused.
    monkeypatch.setattr(T, "_VENDOR", None)
    for name in [n for n in list(sys.modules)
                 if n.startswith("gbench.third_party._teval_vendor")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "networkx", None)
    with pytest.raises(RuntimeError) as exc:
        T._vendor()
    assert "networkx" in str(exc.value) and "INFRA REQUIRED" in str(exc.value)
    assert T._VENDOR is None


def test_vendor_is_loaded_once_with_the_embedder_constructor_neutered(monkeypatch):
    monkeypatch.setattr(T, "_VENDOR", None)
    vendor = T._vendor()
    assert T._vendor() is vendor
    from gbench.third_party._teval_vendor import planning_evaluator as pe
    from gbench.third_party._teval_vendor import (
        reason_retrieve_understand_evaluator as rru)
    assert pe.SentenceTransformer("any-model") is None
    assert rru.SentenceTransformer("any-model") is None
    for cls in ("InstructEvaluator", "PlanningEvaluator", "ReasonRetrieveUnderstandEvaluator",
                "ReviewEvaluator"):
        assert hasattr(vendor, cls)


def test_t_eval_module_no_longer_imports_the_vendor_at_load():
    with open(os.path.join(REPO, "gbench", "runners", "eval_suites", "t_eval.py"),
              encoding="utf-8") as fh:
        src = fh.read()
    head = src[:src.index("def _vendor(")]
    # the only import of the vendored package is inside _vendor()
    assert ("from ._teval_vendor" not in head and "from . import _teval_vendor" not in head
            and "gbench.third_party" not in head)


def test_t_eval_doc_names_both_packages_and_the_run_time_check():
    with open(os.path.join(REPO, "docs", "evals", "t_eval.md"), encoding="utf-8") as fh:
        doc = fh.read()
    assert "pip install sentence-transformers networkx" in doc
    assert "gbench --list evals" in doc
    assert "not in any gbench extra" not in doc
