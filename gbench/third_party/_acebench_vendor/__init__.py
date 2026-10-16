# -*- coding: utf-8 -*-
# Vendored from chenchen0103/ACEBench (github.com/chenchen0103/ACEBench) at commit
# 56dd66cf6439b0d9655ee1b353e4cd745c6f664e for gbench's acebench suite: the Chinese prompts and
# the agent category's prompts and simulated environment. Each file starts with a three-line
# header (four for the two *_prompts.py files, then a blank line) naming its upstream source:
#
#   model_inference/prompt_en.py, prompt_zh.py            -> prompt_en.py, prompt_zh.py (verbatim)
#   model_inference/multi_turn/scenarios{en,zh}/travel.py -> scenarios{en,zh}/travel.py (verbatim)
#   model_inference/multi_turn/scenarios{en,zh}/phone_platform/{base_api,message,reminder,
#     food_services}.py -> scenarios{en,zh}/phone_platform/ (verbatim except that the
#     `from model_inference.multi_turn.scenarios<lang>.phone_platform.base_api import BaseApi`
#     line reads `from .base_api import BaseApi`)
#   the module-level prompt constants of model_inference/multi_turn/common_agent.py (lines
#     8-41) and multi_turn/APIModel_user.py (lines 4-64) -> multi_turn_prompts.py, and of
#     model_inference/multi_step/common_agent_step.py (lines 5-58) -> multi_step_prompts.py
#
# model_inference/multi_step/scenarios{en,zh}/ holds a second copy of the ten scenario files
# (multi_step_utils.py imports from there); it differs from the multi_turn copy only in comments
# and blank lines (identical ASTs), so both agent sub-categories use the copy here.
#
# UPSTREAM_BLOBS / PROMPT_SEGMENT_SHA256 record what each file was copied from, so a test can
# rebuild the upstream bytes (strip the header, undo the import shim) and check them offline.
#
# Upstream license (LICENSE.md at that commit):
#
#     MIT License
#
#     Copyright (c) Microsoft Corporation.
#
#     Permission is hereby granted, free of charge, to any person obtaining a copy
#     of this software and associated documentation files (the "Software"), to deal
#     in the Software without restriction, including without limitation the rights
#     to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#     copies of the Software, and to permit persons to whom the Software is
#     furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included in all
#     copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#     SOFTWARE

ACEBENCH_COMMIT = "56dd66cf6439b0d9655ee1b353e4cd745c6f664e"

#: vendored file -> (upstream path, git blob of the upstream file at ACEBENCH_COMMIT, import shim
#: as (upstream line, line here) or None).
UPSTREAM_BLOBS = {
    "prompt_en.py": ("model_inference/prompt_en.py",
                     "48a4318bb66da926536524b944516f8e2f06409e", None),
    "prompt_zh.py": ("model_inference/prompt_zh.py",
                     "92c436cfa3dd0a0199618666c13b91b1a6a92a71", None),
    "scenariosen/travel.py": ("model_inference/multi_turn/scenariosen/travel.py",
                              "2a4238cd6ec506800bc86d744fd699213f911d7b", None),
    "scenariosen/phone_platform/base_api.py": (
        "model_inference/multi_turn/scenariosen/phone_platform/base_api.py",
        "e3b86f398bf13210a9ea17f69274e3b20b582b4e", None),
    "scenariosen/phone_platform/message.py": (
        "model_inference/multi_turn/scenariosen/phone_platform/message.py",
        "82b0bd3122a56f05bc8e166a5e134a4b8c7a0f0b",
        ("from model_inference.multi_turn.scenariosen.phone_platform.base_api import BaseApi",
         "from .base_api import BaseApi")),
    "scenariosen/phone_platform/reminder.py": (
        "model_inference/multi_turn/scenariosen/phone_platform/reminder.py",
        "3b66663386eaa11618a286b6bdbd4f1afb113c29",
        ("from model_inference.multi_turn.scenariosen.phone_platform.base_api import BaseApi",
         "from .base_api import BaseApi")),
    "scenariosen/phone_platform/food_services.py": (
        "model_inference/multi_turn/scenariosen/phone_platform/food_services.py",
        "482469a2c5a8f4e79ae294579117f48e316a71be",
        ("from model_inference.multi_turn.scenariosen.phone_platform.base_api import BaseApi",
         "from .base_api import BaseApi")),
    "scenarioszh/travel.py": ("model_inference/multi_turn/scenarioszh/travel.py",
                              "6cda18dbbc8fea9e2c1a18742c51704299eb1756", None),
    "scenarioszh/phone_platform/base_api.py": (
        "model_inference/multi_turn/scenarioszh/phone_platform/base_api.py",
        "1fe4daa5bbcadf6d39c2f96743ebfb6401f36ea0", None),
    "scenarioszh/phone_platform/message.py": (
        "model_inference/multi_turn/scenarioszh/phone_platform/message.py",
        "ec0ee701dc8118fc0c58bf9013227d07ae764d7c",
        ("from model_inference.multi_turn.scenarioszh.phone_platform.base_api import BaseApi",
         "from .base_api import BaseApi")),
    "scenarioszh/phone_platform/reminder.py": (
        "model_inference/multi_turn/scenarioszh/phone_platform/reminder.py",
        "16f35e968da791747b509503c6175d08cf58d993",
        ("from model_inference.multi_turn.scenarioszh.phone_platform.base_api import BaseApi",
         "from .base_api import BaseApi")),
    "scenarioszh/phone_platform/food_services.py": (
        "model_inference/multi_turn/scenarioszh/phone_platform/food_services.py",
        "67a317c4155f15a2d12354fa5f5e3102791687f5",
        ("from model_inference.multi_turn.scenarioszh.phone_platform.base_api import BaseApi",
         "from .base_api import BaseApi")),
}

#: sha256 of each *_prompts.py file after its header and blank line (the copied segments with
#: their "# <path> lines a-b, verbatim." markers), taken from the upstream files at the commit.
PROMPT_SEGMENT_SHA256 = {
    "multi_turn_prompts.py": "1ec31917d4ad8392056437b5979124798ff2585203bec886f74f5b276d22a35c",
    "multi_step_prompts.py": "ac7f3a3a93b0a775a601506f52c0c999fc63d0f6e1401e701cdba7fed31c1b91",
}
