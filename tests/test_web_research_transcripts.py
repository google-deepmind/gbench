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

"""gaia / deepsearch_qa keep every web_search call, a crashed one included, per run.

* A search whose tool raised is recorded as a failed search (backend `tool-raised`), so the
  sample is a search_backend_failure harness error; the exception still reaches base.
* Each run has its own store: a run in another thread never reads this run's calls.

No network: the search tool is faked.
"""

import asyncio
import threading

import pytest

from gbench.runners.eval_suites import gaia as G


def test_a_search_that_raised_is_recorded_as_a_failed_search(monkeypatch):
    async def boom(name, args):
        raise ConnectionError("grounding proxy down")
    monkeypatch.setattr(G, "execute_tool", boom)
    G.reset_tool_transcripts()
    with pytest.raises(ConnectionError):
        asyncio.run(G.record_web_search("web_search", {"query": "q"}, 3))
    traces = [{"sample_idx": 3, "status": "OK"}]
    G.attach_tool_transcripts(traces)
    (entry,) = traces[0]["tool_transcript"]
    assert entry["backend"] == "tool-raised" and entry["search_failed"] is True
    assert entry["exception"] == "ConnectionError" and "grounding proxy down" in entry["reply"]
    assert G.web_research_harness_error(traces[0])["kind"] == "search_backend_failure"


def test_each_run_keeps_its_own_calls(monkeypatch):
    async def ok(name, args):
        return '{"backend": "gemini:m", "results": [{"title": "t"}]}'
    monkeypatch.setattr(G, "execute_tool", ok)
    G.reset_tool_transcripts()
    asyncio.run(G.record_web_search("web_search", {"query": "mine"}, 0))
    other = {}

    def another_run():
        G.reset_tool_transcripts()
        asyncio.run(G.record_web_search("web_search", {"query": "theirs"}, 0))
        traces = [{"sample_idx": 0}]
        G.attach_tool_transcripts(traces)
        other["calls"] = traces[0]["tool_transcript"]
    t = threading.Thread(target=another_run)
    t.start()
    t.join()
    mine = [{"sample_idx": 0}]
    G.attach_tool_transcripts(mine)
    assert [c["arguments"]["query"] for c in mine[0]["tool_transcript"]] == ["mine"]
    assert [c["arguments"]["query"] for c in other["calls"]] == ["theirs"]
