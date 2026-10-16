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

"""livebench tells server-ended requests apart from replies the model finished empty.

LiveBench records a request that a vLLM server aborted (for example while draining at shutdown)
exactly like an empty reply the model finished. vLLM counts the requests it ended itself in its
Prometheus counter request_success_total (finished_reason "abort" or "error"), so the suite reads
that counter before and after the run. These tests serve fake /metrics pages from a local stub on
an ephemeral port; no model server is used.
"""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from gbench.runners.eval_suites import livebench

PAGE = """# HELP vllm:request_success_total Count of successfully processed requests.
# TYPE vllm:request_success_total counter
vllm:request_success_total{engine="0",finished_reason="stop",model_name="m"} 24.0
vllm:request_success_total{engine="0",finished_reason="length",model_name="m"} 2.0
vllm:request_success_total{engine="0",finished_reason="abort",model_name="m"} %(abort)s
vllm:request_success_total{engine="0",finished_reason="error",model_name="m"} %(error)s
vllm:request_success_total{engine="0",finished_reason="abort",model_name="other"} 7.0
"""


def _serve(body, status=200):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"


def test_counts_abort_and_error_for_the_served_model_only():
    srv, url = _serve(PAGE % {"abort": "3.0", "error": "1.0"})
    try:
        assert livebench.server_aborted_requests(url, ["m"]) == 4
    finally:
        srv.shutdown()


def test_counts_every_model_when_no_line_names_the_served_one():
    srv, url = _serve(PAGE % {"abort": "3.0", "error": "1.0"})
    try:
        assert livebench.server_aborted_requests(url, ["served-alias"]) == 11
    finally:
        srv.shutdown()


def test_an_endpoint_without_vllm_metrics_gives_none():
    srv, url = _serve("not prometheus", status=200)
    try:
        assert livebench.server_aborted_requests(url, ["m"]) is None
    finally:
        srv.shutdown()
    srv, url = _serve("gone", status=404)
    try:
        assert livebench.server_aborted_requests(url, ["m"]) is None
    finally:
        srv.shutdown()


def test_the_empty_replies_note_points_at_the_abort_count():
    assert "server_aborted_requests" in livebench._EMPTY_REPLIES_NOTE
