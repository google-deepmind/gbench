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

"""Credentials stay out of reach of model-written code and of the files a run leaves.

A fake host is built in a temporary directory under /var/tmp (the jail already empties /tmp,
so files there would be hidden whatever the sandbox did): a home directory with the usual
credential files (gcloud ADC and token database, Docker config, ~/.netrc, an SSH key, the
huggingface_hub token files, an arbitrary key file), and credential files outside it named by
$GOOGLE_APPLICATION_CREDENTIALS, $CLOUDSDK_CONFIG, $DOCKER_CONFIG and a symlinked
$HF_TOKEN_PATH. A program run through run_sandboxed must read none of them, while it still
imports from the interpreter and reads the dataset files it legitimately needs (the Hugging
Face hub cache, an operator-configured SciCode test_data.h5, a caller's `readable` path).

The results backstop (base.redact_tree) must also redact a key another process appended while
it rewrote a file, instead of recording the file as checked.

No model server or network is used; every file is written under a temporary directory.
"""

import json
import os
import sys
import tempfile
from types import SimpleNamespace

import pytest

from gbench.runners.eval_suites import sandbox

SECRETS = {
    "adc": "ADC-refresh-token-0001",
    "gcloud_db": "GCLOUD-access-token-0002",
    "docker": "DOCKER-auth-0003",
    "netrc": "NETRC-password-0004",
    "ssh": "SSH-private-key-0005",
    "hf_token": "hf_FAKEtoken0006",
    "hf_stored": "hf_FAKEstored0007",
    "home_key": "HOME-key-file-0008",
    "gac": "GAC-service-account-0009",
    "cloudsdk": "CLOUDSDK-token-0010",
    "docker_cfg": "DOCKER-CONFIG-auth-0011",
    "hf_link": "hf_FAKElinked0012",
}


def _write(path: str, text: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def _outside_tmp_base() -> str:
    base = "/var/tmp"
    if not (os.path.isdir(base) and os.access(base, os.W_OK)):
        pytest.skip("needs a writable temporary directory outside /tmp (/var/tmp)")
    return base


@pytest.fixture
def host(monkeypatch):
    with tempfile.TemporaryDirectory(dir=_outside_tmp_base(), prefix="gbench-sbx-") as root:
        root = os.path.realpath(root)
        home = os.path.join(root, "home")
        out = os.path.join(root, "outside")
        creds = {
            "adc": _write(f"{home}/.config/gcloud/application_default_credentials.json",
                          SECRETS["adc"]),
            "gcloud_db": _write(f"{home}/.config/gcloud/access_tokens.db", SECRETS["gcloud_db"]),
            "docker": _write(f"{home}/.docker/config.json", SECRETS["docker"]),
            "netrc": _write(f"{home}/.netrc", SECRETS["netrc"]),
            "ssh": _write(f"{home}/.ssh/id_ed25519", SECRETS["ssh"]),
            "hf_token": _write(f"{home}/.cache/huggingface/token", SECRETS["hf_token"]),
            "hf_stored": _write(f"{home}/.cache/huggingface/stored_tokens", SECRETS["hf_stored"]),
            "home_key": _write(f"{home}/keys/service-account.json", SECRETS["home_key"]),
            "gac": _write(f"{out}/sa.json", SECRETS["gac"]),
            "cloudsdk": _write(f"{out}/gcloud-config/credentials.db", SECRETS["cloudsdk"]),
            "docker_cfg": _write(f"{out}/docker-config/config.json", SECRETS["docker_cfg"]),
        }
        linked_target = _write(f"{out}/real/hf-token", SECRETS["hf_link"])
        link = f"{out}/hf/token"
        os.makedirs(os.path.dirname(link))
        os.symlink(linked_target, link)
        creds["hf_link"] = link
        data = {
            "hub": _write(f"{home}/.cache/huggingface/hub/datasets--x/snapshots/r/rows.txt",
                          "hub-rows"),
            "scicode": _write(f"{home}/data/test_data.h5", "h5-bytes"),
            "readable": _write(f"{home}/extra/input.txt", "caller-input"),
            "outside": _write(f"{out}/public/notes.txt", "public-notes"),
        }
        for name in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME",
                     "XDG_CONFIG_HOME", "SCICODE_TEST_DATA", "AWS_SHARED_CREDENTIALS_FILE",
                     "AWS_CONFIG_FILE", "KUBECONFIG", "NETRC", "BOTO_CONFIG",
                     "AZURE_CONFIG_DIR", "GH_CONFIG_DIR"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("HOME", home)
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", creds["gac"])
        monkeypatch.setenv("CLOUDSDK_CONFIG", os.path.dirname(creds["cloudsdk"]))
        monkeypatch.setenv("DOCKER_CONFIG", os.path.dirname(creds["docker_cfg"]))
        monkeypatch.setenv("HF_TOKEN_PATH", link)
        monkeypatch.setenv("GBENCH_SCICODE_TEST_DATA", data["scicode"])
        yield SimpleNamespace(root=root, home=home, out=out, creds=creds, data=data,
                              linked_target=linked_target)


@pytest.fixture
def jail(monkeypatch):
    """GBENCH_SANDBOX=required with a fresh probe; skipped only where bubblewrap cannot run."""
    monkeypatch.setenv("GBENCH_SANDBOX", "required")
    sandbox.sandbox_available.cache_clear()
    sandbox._probe_bwrap.cache_clear()
    try:
        if not sandbox.sandbox_available():
            pytest.skip("bubblewrap is unavailable or blocked on this host")
        yield
    finally:
        sandbox.sandbox_available.cache_clear()
        sandbox._probe_bwrap.cache_clear()


_READER = r"""
import json, os, sys
out = {}
for name, path in json.loads(sys.argv[1]).items():
    try:
        if os.path.isdir(path):
            out[name] = {"dir": sorted(os.listdir(path))}
        else:
            with open(path, encoding="utf-8", errors="replace") as fh:
                out[name] = {"text": fh.read()}
    except OSError as e:
        out[name] = {"error": type(e).__name__}
import numpy
out["_numpy"] = numpy.__name__
print(json.dumps(out))
"""


def test_no_credential_file_is_readable_inside_the_sandbox(host, jail):
    targets = dict(host.creds)
    targets.update({"hf_link_target": host.linked_target,
                    "gcloud_dir": os.path.dirname(host.creds["adc"]),
                    "cloudsdk_dir": os.path.dirname(host.creds["cloudsdk"]),
                    "ssh_dir": os.path.dirname(host.creds["ssh"])})
    targets.update({f"data_{k}": v for k, v in host.data.items()})
    r = sandbox.run_sandboxed([sys.executable, "-c", _READER, json.dumps(targets)],
                              readable=[host.data["readable"]], capture_output=True, text=True,
                              timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    seen = json.loads(r.stdout.strip().splitlines()[-1])
    for secret in SECRETS.values():
        assert secret not in r.stdout, secret
    for name in list(host.creds) + ["hf_link_target"]:
        got = seen[name]
        assert "error" in got or got.get("text") == "", (name, got)
    for name in ("gcloud_dir", "cloudsdk_dir", "ssh_dir"):
        assert seen[name].get("dir", []) == [] or "error" in seen[name], (name, seen[name])
    # what a program legitimately reads is still there
    assert seen["data_hub"] == {"text": "hub-rows"}
    assert seen["data_scicode"] == {"text": "h5-bytes"}
    assert seen["data_readable"] == {"text": "caller-input"}
    assert seen["data_outside"] == {"text": "public-notes"}
    assert seen["_numpy"] == "numpy"


def test_the_credential_files_are_readable_outside_the_sandbox(host):
    """Control: the fixture's files are real and readable by this process."""
    for name, path in host.creds.items():
        with open(path, encoding="utf-8") as fh:
            assert fh.read() == SECRETS[name]


def test_the_jail_empties_the_home_and_hides_credentials_by_their_target(host):
    argv = sandbox.wrap_argv(["python3", "-c", "pass"], force=True)
    pairs = list(zip(argv, argv[1:]))
    triples = list(zip(argv, argv[1:], argv[2:]))
    assert ("--tmpfs", host.home) in pairs
    assert ("--ro-bind", "/dev/null", os.path.realpath(host.creds["gac"])) in triples
    assert ("--ro-bind", "/dev/null", host.linked_target) in triples
    assert host.creds["hf_link"] not in argv            # a symlink is hidden by its target
    assert ("--tmpfs", os.path.dirname(host.creds["cloudsdk"])) in pairs
    assert ("--tmpfs", os.path.dirname(host.creds["docker_cfg"])) in pairs
    hub = os.path.join(host.home, ".cache", "huggingface", "hub")
    assert ("--ro-bind", hub, hub) in triples
    # the emptied home comes before what is bound back, which comes before the masks
    assert argv.index(host.home) < argv.index(hub) < argv.index(host.linked_target)
    # nothing inside the emptied home is masked: the home itself is already empty
    assert not any(a == "/dev/null" and b.startswith(host.home + os.sep)
                   for a, b in zip(argv, argv[1:]))
    assert argv[-3:] == ["python3", "-c", "pass"]


def test_a_home_holding_the_interpreter_is_not_emptied(monkeypatch):
    monkeypatch.setenv("HOME", os.path.realpath(sys.prefix))
    assert os.path.realpath(sys.prefix) not in sandbox.masked_homes()
    assert os.path.realpath(sys.prefix) not in sandbox._home_rebinds(sandbox.masked_homes())


def test_nothing_bound_back_contains_a_home(host):
    homes = sandbox.masked_homes()
    for path in sandbox._home_rebinds(homes, readable=[host.home, os.path.dirname(host.home)]):
        assert not any(h == path or h.startswith(path + os.sep) for h in homes), path


def test_the_module_docstring_says_what_is_hidden():
    doc = sandbox.__doc__
    assert "The program never sees a credential" not in doc
    for needle in ("home directory", "credential_paths", "GOOGLE_APPLICATION_CREDENTIALS",
                   "`none` included"):
        assert needle in doc, needle


# --------------------------------------------------------------------------- #
# credential values a writer appends while the results backstop rewrites a file
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("live", [True, False])
def test_bytes_appended_during_a_rewrite_are_checked_again(tmp_path, monkeypatch, live):
    """redact_file keeps what another process appended between its read and its rewrite, as
    written; redact_tree checks the file again before it records it as clean, so a key in
    that tail is redacted now instead of being skipped by every later sweep."""
    from gbench.runners.eval_suites import base
    # Split so that no source line matches the leak gate (scripts/check_no_leaks.sh).
    key = "sk-" + "FAKEappend000000key0002"
    monkeypatch.setenv("OPENAI_API_KEY", key)
    monkeypatch.setattr(base, "_TREE_SEEN", {})
    log = tmp_path / "harness.log"
    log.write_text(f"first {key}\n", encoding="utf-8")
    if not live:
        old = os.stat(log).st_mtime - 3600
        os.utime(log, (old, old))
    real = base._rewrite_in_place
    appended = []

    def rewrite_with_a_concurrent_append(fh, data, new, keep_length):
        if not appended:                    # another process appends after the read
            with open(log, "ab") as writer:
                writer.write(f"late {key}\n".encode())
            appended.append(True)
        return real(fh, data, new, keep_length)

    monkeypatch.setattr(base, "_rewrite_in_place", rewrite_with_a_concurrent_append)
    report = base.redact_tree(tmp_path)
    text = log.read_text(encoding="utf-8")
    assert appended and key not in text, text
    assert text.count(base.REDACTED) == 2 and "late " in text
    assert report["redacted"] == [str(log)]
    assert str(log) in base._TREE_SEEN


# --------------------------------------------------------------------------- #
# paths reached through a symlink inside the emptied home
# --------------------------------------------------------------------------- #
@pytest.fixture
def linked_host(monkeypatch):
    """A home whose Hugging Face directory and data directory are symlinks to another disk: the
    emptied home no longer holds the links, so the jail must bind the targets back at the paths
    the program uses, and still hide the token that sits next to the hub cache."""
    with tempfile.TemporaryDirectory(dir=_outside_tmp_base(), prefix="gbench-sbx-link-") as root:
        root = os.path.realpath(root)
        home, disk = os.path.join(root, "home"), os.path.join(root, "disk")
        os.makedirs(os.path.join(home, ".cache"))
        _write(f"{disk}/hf/hub/datasets--y/snapshots/r/rows.txt", "linked-hub-rows")
        _write(f"{disk}/hf/token", SECRETS["hf_link"])
        _write(f"{disk}/data/test_data.h5", "linked-h5-bytes")
        os.symlink(f"{disk}/hf", f"{home}/.cache/huggingface")
        os.symlink(f"{disk}/data", f"{home}/linkdata")
        for name in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME",
                     "HF_TOKEN_PATH", "SCICODE_TEST_DATA", "GOOGLE_APPLICATION_CREDENTIALS",
                     "CLOUDSDK_CONFIG", "DOCKER_CONFIG"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("HOME", home)
        monkeypatch.setenv("GBENCH_SCICODE_TEST_DATA", f"{home}/linkdata/test_data.h5")
        yield SimpleNamespace(home=home, disk=disk)


def test_paths_through_a_symlink_in_the_home_still_resolve(linked_host, jail):
    h = linked_host.home
    targets = {"hub": f"{h}/.cache/huggingface/hub/datasets--y/snapshots/r/rows.txt",
               "scicode": f"{h}/linkdata/test_data.h5",
               "token_as_written": f"{h}/.cache/huggingface/token",
               "token_real": f"{linked_host.disk}/hf/token"}
    r = sandbox.run_sandboxed([sys.executable, "-c", _READER, json.dumps(targets)],
                              capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    seen = json.loads(r.stdout.strip().splitlines()[-1])
    assert seen["hub"] == {"text": "linked-hub-rows"}
    assert seen["scicode"] == {"text": "linked-h5-bytes"}
    for name in ("token_as_written", "token_real"):
        assert "error" in seen[name] or seen[name].get("text") == "", (name, seen[name])
    assert SECRETS["hf_link"] not in r.stdout


def test_a_linked_path_is_bound_at_both_its_target_and_the_path_as_written(linked_host):
    homes = sandbox.masked_homes()
    binds = sandbox._home_binds(homes)
    written = os.path.join(linked_host.home, ".cache", "huggingface", "hub")
    assert (os.path.realpath(written), written) in binds
    assert (os.path.realpath(f"{linked_host.home}/linkdata/test_data.h5"),
            f"{linked_host.home}/linkdata/test_data.h5") in binds
    # destinations are ordered parents first, so a child bind lands inside its parent
    dests = [d for _, d in binds]
    assert dests == sorted(dests, key=lambda p: (p.count(os.sep), p))
