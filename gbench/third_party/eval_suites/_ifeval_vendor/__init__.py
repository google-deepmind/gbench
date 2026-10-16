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
# Vendored from google-research/google-research, instruction_following_eval/ at commit
# e6890f85757dd84e27ca6df2dd30651dafad28e0 (Apache-2.0; each file keeps its license header):
# instructions.py, instructions_util.py, instructions_registry.py, evaluation_lib.py.
# Verbatim except the imports: package-relative instead of `instruction_following_eval.*`;
# stdlib `logging` instead of absl's (same logging.error calls); `langdetect` imported in a
# try/except so importing gbench does not need it (the ifeval runner hard-errors when it is
# missing); and `types.MappingProxyType` instead of `immutabledict` for the read-only
# LANGUAGE_CODES table. Do NOT edit the checkers.
# IFEval: Zhou et al., 2023 (arXiv:2311.07911).
