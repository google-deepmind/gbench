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

"""Credential redaction for code outside the eval suites (LogManager, the CLI and the runners).

The implementation is the credential-redaction section of gbench.runners.eval_suites.base,
which has to stay self-contained there: the tau2 and BFCL harness images load that module on
its own, without the gbench package. Names are resolved from it on first use, so importing
this module does not load the eval suites; import it inside the function that needs it.
"""

from typing import Any


def __getattr__(name: str) -> Any:
    from ..runners.eval_suites import base
    try:
        return getattr(base, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
