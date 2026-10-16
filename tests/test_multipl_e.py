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

"""multipl_e regression: the canonical language list must contain only real
nuprl/MultiPL-E configs (MultiPL-E translates FROM Python, so there is no
`humaneval-py` -- requesting it raises BuilderConfig-not-found and 0-questions)."""

from gbench.runners.eval_suites import multipl_e

# The configs nuprl/MultiPL-E actually publishes (humaneval-<lang> / mbpp-<lang>).
_VALID = set("adb clj cpp cs d dart elixir go hs java jl js lua ml php pl r rb "
             "rkt rs scala sh swift ts".split())


def test_no_python_config():
    assert "py" not in multipl_e._CANONICAL_LANGS
    assert "python" not in multipl_e._CANONICAL_LANGS


def test_all_canonical_langs_are_real_configs():
    bad = [l for l in multipl_e._CANONICAL_LANGS if l not in _VALID]
    assert not bad, f"invalid MultiPL-E language configs: {bad}"
    assert len(multipl_e._CANONICAL_LANGS) >= 20  # sanity: full language coverage
