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

"""gbench - Open Model Performance Benchmark Suite.

A production-grade, modular benchmarking tool for comprehensive evaluation
of open models served by vLLM or any OpenAI-compatible endpoint (text and multimodal).
"""

__version__ = "0.5.0"
__author__ = "Luciano Martins"

import os
import re
import sys

# Process-wide Gemini API key multi-key support & auto-sanitization.
# GEMINI_API_KEYS (a comma/semicolon list) wins when both it and GEMINI_API_KEY are set, as in
# base.gemini_api_keys(); otherwise GEMINI_API_KEY (one key or a list) is used. The chosen list is
# written back to GEMINI_API_KEYS, and GEMINI_API_KEY is set to its first key for third-party
# libraries (LiteLLM, OpenAI SDK) that read a single key.
def _split_gemini_keys(raw: str) -> list:
    return [k.strip() for k in re.split(r"[;,]", raw or "") if k.strip()]


_GEMINI_KEYS = (_split_gemini_keys(os.environ.get("GEMINI_API_KEYS", ""))
                or _split_gemini_keys(os.environ.get("GEMINI_API_KEY", "")))
if _GEMINI_KEYS:
    os.environ["GEMINI_API_KEYS"] = ",".join(_GEMINI_KEYS)
    os.environ["GEMINI_API_KEY"] = _GEMINI_KEYS[0]

# Auto-patch google.genai.Client globally so any eval or plugin can pass raw or multi-keys safely
try:
    from google import genai
    _orig_genai_client_init = genai.Client.__init__

    def _patched_genai_client_init(self, *args, **kwargs):
        if "api_key" in kwargs and kwargs["api_key"]:
            keys = _split_gemini_keys(str(kwargs["api_key"]))
            if len(keys) > 1:
                kwargs["api_key"] = keys[0]
        elif not kwargs.get("api_key"):
            keys = (_split_gemini_keys(os.environ.get("GEMINI_API_KEYS", ""))
                    or _split_gemini_keys(os.environ.get("GEMINI_API_KEY", "")))
            if keys:
                kwargs["api_key"] = keys[0]
        _orig_genai_client_init(self, *args, **kwargs)

    genai.Client.__init__ = _patched_genai_client_init
except Exception:
    pass

