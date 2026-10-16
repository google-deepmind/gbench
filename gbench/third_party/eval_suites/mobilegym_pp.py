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

# Evaluation Suite: mobilegym_pp
# Description: MobileGym++ - 215-task hybrid app-tools + GUI mobile benchmark

"""gbench native built-in runner for `mobilegym_pp` (MobileGym++).

Delegates to `mobilegym.py` inside the shared `gbench-mobilegym:5fbd3f34` local image
(`gbench/docker/mobilegym.Dockerfile`), running the 215-task `bench215/official.json` benchmark
from `OpenMobile2/MobileGym-plusplus@cdde4ad06ed67176b56964dd329b8cb8e0b0001c` with
`Qwen35SessionAgent` (`interaction_mode="hybrid"`).

Sampling: default temperature 0.0 (1.0 with --thinking); override with --temperature or
GBENCH_MOBILEGYM_PP_TEMPERATURE.
"""

from __future__ import annotations

from .mobilegym import (
    BUILD_HINT,
    LAUNCHER_LABEL,
    LAUNCHER_VERSION,
    MOBILEGYM_PP_FULL_COUNT,
    PILLAR,
    SIMULATED_TIME,
    UPSTREAM_DATA_SHA256,
    _IMAGE_DEFAULT,
    _image,
    check_mobilegym_pp_prerequisites,
    merge_shards,
    run_mobilegym_pp,
)

EVAL_NAME = "mobilegym_pp"
DOCS_URL = "docs/evals/mobilegym_pp.md"
UPSTREAM_REPO = "https://github.com/OpenMobile2/MobileGym-plusplus"
UPSTREAM_COMMIT = "cdde4ad06ed67176b56964dd329b8cb8e0b0001c"
UPSTREAM_PP_REPO = UPSTREAM_REPO
UPSTREAM_PP_COMMIT = UPSTREAM_COMMIT

__all__ = [
    "BUILD_HINT",
    "DOCS_URL",
    "EVAL_NAME",
    "LAUNCHER_LABEL",
    "LAUNCHER_VERSION",
    "MOBILEGYM_PP_FULL_COUNT",
    "PILLAR",
    "SIMULATED_TIME",
    "UPSTREAM_DATA_SHA256",
    "UPSTREAM_PP_COMMIT",
    "UPSTREAM_PP_REPO",
    "_IMAGE_DEFAULT",
    "_image",
    "check_mobilegym_pp_prerequisites",
    "merge_shards",
    "run_mobilegym_pp",
]
