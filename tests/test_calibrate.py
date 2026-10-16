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

"""`gbench --calibrate` hardware recommendation + the pre-run over-subscription guard."""

import pytest

import gbench.utils.calibrate as C


@pytest.fixture(autouse=True)
def _fake_docker(monkeypatch):
    """No test here talks to a real Docker daemon: docker is reported present with the built-in
    address pools and no user-defined network."""
    monkeypatch.setattr(C, "_docker_ok", lambda: True)
    monkeypatch.setattr(C, "_run", lambda cmd, timeout=15: "null" if "info" in cmd else "")


def test_detect_hardware_shape():
    hw = C.detect_hardware()
    for k in ("gpus", "gpu_total_vram_gb", "gpu_free_vram_gb", "cpu_cores",
              "ram_total_gb", "ram_available_gb", "docker"):
        assert k in hw, f"missing {k}"
    assert isinstance(hw["cpu_cores"], int) and hw["cpu_cores"] >= 1
    assert isinstance(hw["docker"], bool)


def test_max_safe_sandboxes():
    assert C.max_safe_sandboxes({"docker": False, "cpu_cores": 64, "ram_available_gb": 512}) == 0
    # min(cores//2, ram//8): min(4, 4) = 4
    assert C.max_safe_sandboxes({"docker": True, "cpu_cores": 8, "ram_available_gb": 32}) == 4
    # RAM-bound: min(32, 16//8=2) = 2
    assert C.max_safe_sandboxes({"docker": True, "cpu_cores": 64, "ram_available_gb": 16}) == 2
    # always at least 1 when docker present
    assert C.max_safe_sandboxes({"docker": True, "cpu_cores": 1, "ram_available_gb": 1}) == 1


def test_recommend_scales_with_hardware():
    rec = C.recommend({"gpus": 8, "cpu_cores": 96, "ram_available_gb": 700, "docker": True})
    assert rec["num_gpus"] == 8
    assert rec["sandboxes"] == min(96 // 2, 700 // 8)
    assert rec["eval_concurrency"] == 64          # capped at 64
    rec2 = C.recommend({"gpus": 0, "cpu_cores": 4, "ram_available_gb": 16, "docker": False})
    assert rec2["num_gpus"] is None and rec2["sandboxes"] == 0
    assert rec2["eval_concurrency"] == 16         # 4 cores * 4


def test_check_oversubscription(monkeypatch):
    monkeypatch.delenv(C.GUARD_BYPASS_ENV, raising=False)
    hw = {"docker": True, "cpu_cores": 8, "ram_available_gb": 32}   # rec 4, hard_cap max(8, 12)=12
    ok, _ = C.check_oversubscription(4, hw)
    assert ok
    ok, msg = C.check_oversubscription(50, hw)
    assert not ok and "--sandboxes 50" in msg and "over-subscribes" in msg and "--calibrate" in msg
    # None / 0 / non-int are no-ops
    assert C.check_oversubscription(None, hw)[0]
    assert C.check_oversubscription(0, hw)[0]
    # no docker -> guarded elsewhere, so this returns ok
    assert C.check_oversubscription(999, {"docker": False, "cpu_cores": 2, "ram_available_gb": 4})[0]


def test_bypass_env_disables_the_guard(monkeypatch):
    monkeypatch.setenv(C.GUARD_BYPASS_ENV, "1")
    ok, _ = C.check_oversubscription(9999, {"docker": True, "cpu_cores": 8, "ram_available_gb": 32})
    assert ok


def test_format_report_is_str():
    hw = C.detect_hardware()
    r = C.format_report(hw, C.recommend(hw))
    assert isinstance(r, str) and "Recommended flags:" in r and "--sandboxes" in r


def test_calibrate_cli_flag_prints_and_exits_zero(capsys):
    from gbench.cli import main
    rc = main(["--calibrate"])
    assert rc == 0
    out = capsys.readouterr().out
    # The eval client's HTTP concurrency is recommended as --eval-concurrency, never as
    # --batch-sizes (which the pre-run guard reads as a sandbox count for containerized suites
    # and which also sets the performance batch sizes).
    assert "hardware calibration" in out and "--eval-concurrency" in out
    assert "--batch-sizes" not in out
