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

"""--throughput-only with --remote-endpoint is refused with a clear message.

2026-10-14 smoke run: the throughput pillar needs a local vLLM engine, so against a remote
endpoint it plans nothing, and `--throughput-only --remote-endpoint ...` exited 0 after printing
only "No benchmark results to summarize".
"""

import pytest

from gbench import cli


def test_throughput_only_against_a_remote_endpoint_is_an_argument_error(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--throughput-only", "--remote-endpoint", "http://127.0.0.1:9/v1",
                  "--tokenizer", "google/gemma-4-26B-A4B-it"])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "--throughput-only cannot be used with --remote-endpoint" in err
    assert "--serving-only or --stress-test" in err
