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

"""swe_bench_pro's image removes the 15 s read timeout of Harbor's egress-control sidecar.

In the agent phase the sandbox's model requests pass through Harbor 0.22.0's egress sidecar, a
gost transparent proxy (red handler, HTTP sniffing). gost waits at most `readTimeout` for the
upstream's response headers and uses 15 s when it is unset (go-gost/x v0.10.9,
handler/redirect/tcp/metadata.go). A non-streamed completion sends its headers only when the
reply is complete. 2026-10-11 validation run (5 tasks, shared server): 128 of 198 model requests
were closed at <= 15 s (relay client_close_ages max_s 15.0), every agent logged "Server
disconnected without sending a response", and 4 of 5 rollouts were agent_infra. A probe through
the sidecar to an upstream that answers after 30 s failed at 15.0 s with the original gost.yaml
and returned 200 after 30.0 s with the patched one.

These tests run the Dockerfile's own sed command on Harbor 0.22.0's gost.yaml and check that
the runner requires and records the patched image. No network, no docker.
"""

import hashlib
import json
import re
import subprocess
from pathlib import Path

import yaml

from gbench.runners.eval_suites import swe_agentic
from gbench.runners.eval_suites import swe_bench_pro as P

DOCKERFILE = (Path(__file__).resolve().parents[1] / "gbench" / "docker"
              / "swe_bench_pro_v2.Dockerfile")

#: harbor 0.22.0: harbor/environments/docker/harbor-docker-egress-control-sidecar/gost.yaml
HARBOR_GOST_YAML = """\
# addr, so_mark, and file.path must stay in sync with the constants in
# bin/network-policy (GOST_PORT, GOST_MARK, ALLOWLIST); the `reload` interval
# must stay below RELOAD_WAIT there.
services:
  - name: transparent-egress
    addr: ":12345"
    bypass: allowlist
    metadata:
      so_mark: 114514
    handler:
      type: red
      metadata:
        sniffing: true
        sniffing.timeout: 5s
        sniffing.fallback: true
    listener:
      type: red

bypasses:
  - name: allowlist
    whitelist: true
    reload: 1s
    file:
      path: /opt/egress-sidecar/allowlist.txt
"""


def _arg(name):
    m = re.search(rf"^ARG {name}=(\S+)$", DOCKERFILE.read_text(), re.M)
    assert m, name
    return m.group(1)


def test_dockerfile_checks_the_original_file_and_labels_the_timeout():
    text = DOCKERFILE.read_text()
    assert _arg("EGRESS_SIDECAR_GOST_YAML_SHA256") == hashlib.sha256(
        HARBOR_GOST_YAML.encode()).hexdigest()
    assert _arg("EGRESS_SIDECAR_READ_TIMEOUT") == P.EGRESS_SIDECAR_READ_TIMEOUT
    assert ('LABEL org.gbench.egress_sidecar_read_timeout="${EGRESS_SIDECAR_READ_TIMEOUT}"'
            in text)


def test_dockerfile_sed_adds_read_timeout_to_the_handler_metadata(tmp_path):
    m = re.search(r'&& (sed -i "s/\^ +sniffing\.fallback.*?" "\$f")', DOCKERFILE.read_text())
    assert m
    f = tmp_path / "gost.yaml"
    f.write_text(HARBOR_GOST_YAML)
    env = {"f": str(f), "EGRESS_SIDECAR_READ_TIMEOUT": P.EGRESS_SIDECAR_READ_TIMEOUT,
           "PATH": "/usr/bin:/bin"}
    subprocess.run(["sh", "-c", m.group(1)], env=env, check=True)
    doc = yaml.safe_load(f.read_text())
    md = doc["services"][0]["handler"]["metadata"]
    assert md == {"sniffing": True, "sniffing.timeout": "5s", "sniffing.fallback": True,
                  "readTimeout": P.EGRESS_SIDECAR_READ_TIMEOUT}
    # nothing else in the sidecar's config changes
    del md["readTimeout"]
    assert doc == yaml.safe_load(HARBOR_GOST_YAML)


def test_read_timeout_exceeds_every_agent_client_timeout():
    h = int(re.fullmatch(r"(\d+)h", P.EGRESS_SIDECAR_READ_TIMEOUT).group(1))
    assert P.EGRESS_SIDECAR_READ_TIMEOUT_S == h * 3600
    # the largest request the relay sizes for: a 1M-token window at the 1 tok/s floor
    from gbench.runners.eval_suites import base
    floor = base.MIN_DECODE_TOK_S
    try:
        base.MIN_DECODE_TOK_S = 1.0
        client_s = swe_agentic.agent_request_timeouts(1 << 20).client_s
    finally:
        base.MIN_DECODE_TOK_S = floor
    assert client_s > 1 << 20
    assert P.EGRESS_SIDECAR_READ_TIMEOUT_S > 10 * client_s


def _prereqs(monkeypatch, labels):
    class R:
        def __init__(self, stdout=""):
            self.returncode, self.stdout, self.stderr = 0, stdout, ""

    def run(cmd, **kw):
        return R(json.dumps(labels)) if cmd[:3] == ["docker", "image", "inspect"] else R()

    monkeypatch.delenv("GBENCH_SWE_BENCH_PRO_IMAGE", raising=False)
    monkeypatch.setattr(P.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(P.subprocess, "run", run)
    monkeypatch.setattr(P.os.path, "exists", lambda p: True)
    return P.check_swe_bench_pro_v2_prerequisites()


def test_prerequisites_reject_an_image_without_the_sidecar_patch(monkeypatch):
    old = {"org.gbench.harbor": P.HARBOR_VERSION, "org.gbench.swe_bench_pro_os": P.HARNESS_COMMIT}
    ok, msg = _prereqs(monkeypatch, old)
    assert not ok and "org.gbench.egress_sidecar_read_timeout" in msg and "rebuild" in msg
    ok, msg = _prereqs(monkeypatch, {
        **old, "org.gbench.egress_sidecar_read_timeout": P.EGRESS_SIDECAR_READ_TIMEOUT})
    assert ok and msg == ""


def test_default_image_is_the_rebuilt_tag():
    assert P.V2_IMAGE_DEFAULT == "gbench-swe-bench-pro-v2:harbor-0.22.0-66f92766-sidecar-rt"
    assert P.v2_image() == P.V2_IMAGE_DEFAULT or P.os.environ.get("GBENCH_SWE_BENCH_PRO_IMAGE")
