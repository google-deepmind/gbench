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
"""SciCode's `scicode` package, as far as its test cases import it.

SciCode's test cases import `scicode.compare.cmp` (90 of them in the test split, e.g.
`from scicode.compare.cmp import cmp_tuple_or_list`), and upstream's
eval/scripts/test_generated_code.py runs them where the `scicode` package is installed. This
directory holds that package's `scicode/__init__.py`, `scicode/compare/__init__.py` and
`scicode/compare/cmp.py` verbatim from github.com/scicode-bench/SciCode at commit e3158ea0
(src/scicode/, Apache-2.0), each under an attribution header. The scicode suite puts this
directory on the PYTHONPATH of every scored sub-step, so `import scicode` resolves here; it is
not imported by gbench itself. The rest of the upstream package (gen, parse, utils) pulls
dependencies the test cases do not need and is not vendored.
"""
