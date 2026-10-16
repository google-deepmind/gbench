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

"""aider_polyglot starts gbench's relay on a keyless endpoint too.

2026-10-11 smoke run: the relay started only when the endpoint had a key, so on the user's
keyless vLLM aider's requests went straight to the server: no max_tokens from the run's
--max-output-tokens (one reply ran 1,738 s, to the end of the 262,144-token window), no smoke
stop and no request-shape check. The relay now starts either way and adds a key only when there
is one.
"""

from gbench.runners.eval_suites import aider_polyglot as A
from gbench.runners.eval_suites import base, gemini_relay


def test_the_relay_starts_without_an_endpoint_key(monkeypatch):
    monkeypatch.setattr(base, "endpoint_api_key", lambda: None)
    monkeypatch.setattr(gemini_relay, "network_gateway", lambda *a, **k: "127.0.0.1")
    monkeypatch.setitem(base._RUN_KNOBS, "suite", "aider_polyglot")
    monkeypatch.setitem(base._RUN_KNOBS, "max_output_tokens", 4321)
    relay = A._endpoint_relay.original("http://127.0.0.1:9/v1")
    try:
        assert relay is not None and relay._key is None
        assert relay.record()["max_tokens_from_run"]["value"] == 4321
    finally:
        relay.stop()
