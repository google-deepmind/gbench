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

"""tau2's user-simulator / nl-assertion judge must survive an overloaded gemini model.

On the 2026-08-21 sweep tau3 tasks failed after their same-model retries when the single
judge/user model (gemini/gemini-3.6-flash) returned 500 throttling::OVERLOADED. Grounding survives this
by cascading to the next model; the judge/user path did not. `_build_gemini_cascade` gives it
the SAME cascade. These tests exercise the wrapper directly (no global litellm mutation).
"""
import pytest

from gbench.runners.eval_suites import tau_common


_CASCADE = ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash",
            "gemini-3-flash-preview", "gemini-2.5-flash"]


def _no_sleep(monkeypatch):
    monkeypatch.setattr(tau_common.time, "sleep", lambda *_a, **_k: None)


def test_overloaded_gemini_model_falls_over_to_the_next(monkeypatch):
    _no_sleep(monkeypatch)
    seen = []

    def inner(*args, **kwargs):
        m = kwargs["model"]
        seen.append(m)
        if m == "gemini/gemini-3.6-flash":              # the overloaded alias
            raise RuntimeError("500 throttling::OVERLOADED")
        return {"ok": m}

    casc = tau_common._build_gemini_cascade(inner, backoff=0.0, cascade_fn=lambda: _CASCADE)
    # requested model is the one that overloads -> must fall over, not fail the task
    out = casc(model="gemini/gemini-3.6-flash", messages=[])
    assert out == {"ok": "gemini/gemini-3.7-flash"}
    assert seen == ["gemini/gemini-3.6-flash", "gemini/gemini-3.7-flash"]


def test_requested_model_is_tried_first_then_the_rest_of_the_cascade(monkeypatch):
    _no_sleep(monkeypatch)
    seen = []

    def inner(*args, **kwargs):
        seen.append(kwargs["model"])
        raise RuntimeError("overloaded")

    casc = tau_common._build_gemini_cascade(inner, backoff=0.0, cascade_fn=lambda: _CASCADE)
    try:
        casc(model="gemini/gemini-3.6-flash", messages=[])
    except RuntimeError:
        pass
    # requested (3.6) first, then the remaining cascade in order, each exactly once
    assert seen == ["gemini/gemini-3.6-flash", "gemini/gemini-3.7-flash",
                    "gemini/gemini-3.5-flash", "gemini/gemini-3-flash-preview",
                    "gemini/gemini-2.5-flash"]


def test_all_models_overloaded_reraises_the_last_error(monkeypatch):
    _no_sleep(monkeypatch)

    def inner(*args, **kwargs):
        raise RuntimeError(f"down:{kwargs['model']}")

    casc = tau_common._build_gemini_cascade(inner, backoff=0.0, cascade_fn=lambda: _CASCADE)

    with pytest.raises(RuntimeError):
        casc(model="gemini/gemini-3.6-flash", messages=[])


def test_agent_model_is_never_cascaded(monkeypatch):
    """The model under test is routed as openai/<local>; it must pass straight through with no
    fallover (cascading it to a gemini model would grade a different model entirely)."""
    _no_sleep(monkeypatch)
    seen = []

    def inner(*args, **kwargs):
        seen.append(kwargs["model"])
        raise RuntimeError("endpoint down")

    casc = tau_common._build_gemini_cascade(inner, backoff=0.0, cascade_fn=lambda: _CASCADE)

    with pytest.raises(RuntimeError):
        casc(model="openai/google/gemma-4-26B-A4B-it", messages=[])
    assert seen == ["openai/google/gemma-4-26B-A4B-it"], "agent must be tried once, not cascaded"


def test_first_model_success_makes_no_extra_calls(monkeypatch):
    _no_sleep(monkeypatch)
    seen = []

    def inner(*args, **kwargs):
        seen.append(kwargs["model"])
        return {"ok": kwargs["model"]}

    casc = tau_common._build_gemini_cascade(inner, backoff=0.0, cascade_fn=lambda: _CASCADE)
    out = casc(model="gemini/gemini-3.6-flash", messages=[])
    assert out == {"ok": "gemini/gemini-3.6-flash"} and seen == ["gemini/gemini-3.6-flash"]


def test_cascade_uses_the_same_chain_as_grounding_by_default(monkeypatch):
    """No cascade_fn override -> it pulls gbench's one Gemini cascade (base.judge_cascade(), the
    order every Gemini text role uses), never the search cascade."""
    from gbench.runners.eval_suites import base
    for name in ("GBENCH_JUDGE_MODELS", "GBENCH_JUDGE_MODEL", "GBENCH_TAU2_USER_LLM",
                 "GBENCH_TAU2_EVAL_LLM", "TAU2_USER_LLM", "TAU2_EVAL_LLM", "GEMINI_API_KEY",
                 "GEMINI_API_KEYS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GBENCH_SEARCH_MODELS", "gemini-search-only")
    seen = []

    def inner(*args, **kwargs):
        seen.append(kwargs["model"])
        raise RuntimeError("x")

    casc = tau_common._build_gemini_cascade(inner, backoff=0.0)  # default cascade_fn

    with pytest.raises(RuntimeError):
        casc(model="gemini/" + base.judge_cascade()[0], messages=[])
    expected = ["gemini/" + m for m in base.judge_cascade()]
    assert seen == expected and base.judge_cascade() == list(base.GEMINI_CASCADE)
