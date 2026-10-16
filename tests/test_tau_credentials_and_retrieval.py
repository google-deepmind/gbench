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

"""tau2 / tau3: credentials never reach the model under test or the saved files, and tau3's
retrieval is upstream's configuration with its dense search on Gemini embeddings.

  * secrets    - which values count as credentials, and how text, JSON values and whole trace
                 directories are redacted.
  * shell env  - the banking_knowledge shell tool runs its commands without any credential in
                 their environment (a real subprocess checks it), and the probe fails when one
                 is visible.
  * agent key  - the endpoint key is added to the agent's calls only, never to llm_args, and
                 agent-bound messages are redacted.
  * forwarding - no host credential enters the image, whatever the roles and retrieval config
                 (the Gemini relay's placeholder is its only Gemini key); the host's
                 OPENAI_API_KEY (the endpoint key) never becomes the image's.
  * end to end - run_tau_env over a stand-in tau2 that writes results.json the way tau2 does:
                 every produced file and the result are scanned for the key values.
  * retrieval  - upstream's alltools, with its dense search on gemini-embedding-001, is the one
                 configuration; the Gemini key is a prerequisite; the removed retrieval-config
                 knob changes nothing.
  * docs       - the pages state the configuration and the credential handling as coded.

No Docker, no network, no model server.
"""

import contextlib
import json
import os
import subprocess
import sys
import types
from unittest import mock

import pytest

from gbench.runners.eval_suites import tau_common as TC

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

ENDPOINT_KEY = "endpoint-secret-1111aaaa"
OPENAI_KEY = "openai-secret-2222bbbb"
GEMINI_KEY = "gemini-secret-3333cccc"
PROXY_PASS = "proxypass-4444dddd"
ALL_SECRETS = (ENDPOINT_KEY, OPENAI_KEY, GEMINI_KEY, PROXY_PASS)


def _read(*parts):
    with open(os.path.join(_REPO, *parts), encoding="utf-8") as fh:
        return fh.read()


def _files_with_secrets(root, secrets=ALL_SECRETS):
    hits = []
    for dirpath, _dirs, files in os.walk(root):
        for fname in files:
            with open(os.path.join(dirpath, fname), "rb") as fh:
                data = fh.read()
            hits += [(fname, s) for s in secrets if s.encode() in data]
    return hits


@pytest.fixture
def clean_state(monkeypatch):
    monkeypatch.setattr(TC, "_SECRETS", [])
    monkeypatch.setattr(TC, "_AGENT_ROUTE", {})
    for name in ("GBENCH_TAU2_USER_LLM", "TAU2_USER_LLM", "GBENCH_TAU2_EVAL_LLM", "TAU2_EVAL_LLM",
                 "GBENCH_TAU2_RETRIEVAL_CONFIG", "TAU2_RETRIEVAL_CONFIG",
                 "GBENCH_TAU2_OPENAI_API_KEY", "GBENCH_TAU2_OPENAI_BASE_URL", TC._AGENT_KEY_ENV):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# secrets and redaction
# --------------------------------------------------------------------------- #
def test_secret_values_cover_credential_names_lists_and_url_passwords():
    env = {"GEMINI_API_KEYS": f"{GEMINI_KEY},second-key-5555eeee", "HF_TOKEN": "hf_tok_abcdefgh",
           "DB_PASSWORD": "pw-long-enough", "GPG_KEY": "0123456789ABCDEF",
           "HTTPS_PROXY": f"http://proxyuser:{PROXY_PASS}@proxy:3128",
           "GBENCH_TOKENIZER": "google/gemma-4-E4B-it", "OPENAI_API_KEY": "EMPTY",
           "PATH": "/usr/bin:/bin", "GBENCH_TAU2_TOP_K": "64"}
    vals = TC._secret_values(env, extra=(ENDPOINT_KEY, "short"))
    for v in (GEMINI_KEY, "second-key-5555eeee", "hf_tok_abcdefgh", "pw-long-enough",
              PROXY_PASS, f"proxyuser:{PROXY_PASS}", ENDPOINT_KEY, "0123456789ABCDEF"):
        assert v in vals, v
    for v in ("google/gemma-4-E4B-it", "EMPTY", "/usr/bin:/bin", "64", "short"):
        assert v not in vals, v
    assert vals == sorted(vals, key=lambda v: (-len(v), v)), "longest first"


def test_redaction_of_text_values_and_trace_trees(tmp_path):
    secrets = TC._secret_values({"X_API_KEY": 'key"with\\quote1'}, extra=(GEMINI_KEY,))
    assert TC._redact_text(f"a {GEMINI_KEY} b", secrets) == "a [REDACTED] b"
    as_json = json.dumps({"v": 'key"with\\quote1'})
    assert 'quote1' not in TC._redact_text(as_json, secrets)
    obj = {"a": [GEMINI_KEY, {"b": f"x{GEMINI_KEY}y"}], "n": 3, "t": (GEMINI_KEY,)}
    assert TC._redact(obj, secrets) == {"a": ["[REDACTED]", {"b": "x[REDACTED]y"}], "n": 3,
                                        "t": ("[REDACTED]",)}
    assert TC._redact(obj, []) is obj
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "results.json").write_text(json.dumps({"k": GEMINI_KEY}))
    (tmp_path / "clean.txt").write_text("nothing here")
    rewritten = TC._redact_tree(str(tmp_path), secrets)
    assert rewritten == [str(tmp_path / "d" / "results.json")]
    assert _files_with_secrets(str(tmp_path)) == []
    assert TC._redact_tree(str(tmp_path / "missing"), secrets) == []


# --------------------------------------------------------------------------- #
# the shell tool's environment
# --------------------------------------------------------------------------- #
def test_tool_commands_run_without_credentials(monkeypatch, clean_state):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.setenv("HTTPS_PROXY", f"http://u:{PROXY_PASS}@proxy:3128")
    monkeypatch.setenv("GBENCH_TAU2_NUM_TRIALS", "4")
    monkeypatch.setattr(TC, "_SECRETS", TC._secret_values(os.environ, extra=(ENDPOINT_KEY,)))
    monkeypatch.setenv("SMUGGLED", f"prefix-{ENDPOINT_KEY}")      # a value holding a secret
    proxy = TC._ScrubbedSubprocess(subprocess)
    out = proxy.run([sys.executable, "-c", "import os, json; print(json.dumps(dict(os.environ)))"],
                    capture_output=True, text=True, timeout=60)
    seen = json.loads(out.stdout)
    assert not any(s in out.stdout for s in ALL_SECRETS)
    assert "GEMINI_API_KEY" not in seen and "OPENAI_API_KEY" not in seen
    assert "HTTPS_PROXY" not in seen and "SMUGGLED" not in seen
    assert seen["GBENCH_TAU2_NUM_TRIALS"] == "4" and "PATH" in seen
    assert proxy.TimeoutExpired is subprocess.TimeoutExpired   # the rest is the real module
    explicit = proxy.run([sys.executable, "-c", "import os; print(os.environ.get('K'))"],
                         capture_output=True, text=True, env={"K": "given"}, timeout=60)
    assert explicit.stdout.strip() == "given", "an explicit env is left alone"


def test_scrub_is_installed_on_tau2s_sandbox_manager_once():
    sm = types.ModuleType("tau2.knowledge.sandbox_manager")
    sm.subprocess = subprocess
    mods = {"tau2": types.ModuleType("tau2"), "tau2.knowledge": types.ModuleType("tau2.knowledge"),
            "tau2.knowledge.sandbox_manager": sm}
    mods["tau2"].knowledge = mods["tau2.knowledge"]
    mods["tau2.knowledge"].sandbox_manager = sm
    with mock.patch.dict(sys.modules, mods):
        TC._install_tool_env_scrub()
        first = sm.subprocess
        TC._install_tool_env_scrub()
    assert isinstance(first, TC._ScrubbedSubprocess) and sm.subprocess is first


class _ProbeSandbox:
    output = ""
    installed = []

    def __init__(self, allow_writes=False, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def export_documents(self, documents, file_format="txt"):
        pass

    def run_command(self, command, timeout=30):
        assert command == TC._SHELL_PROBE_COMMAND
        return 0, "Title: gbench probe\n\ngbench sandbox probe\n" + self.output, ""


def _probe_modules(cls):
    sm = types.ModuleType("tau2.knowledge.sandbox_manager")
    sm.SandboxManager = cls
    sm.subprocess = subprocess
    mods = {"tau2": types.ModuleType("tau2"), "tau2.knowledge": types.ModuleType("tau2.knowledge"),
            "tau2.knowledge.sandbox_manager": sm}
    mods["tau2"].knowledge = mods["tau2.knowledge"]
    mods["tau2.knowledge"].sandbox_manager = sm
    return mods, sm


def test_shell_probe_fails_when_the_sandbox_shows_a_credential(monkeypatch, clean_state):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    monkeypatch.setattr(TC, "_SECRETS", TC._secret_values(os.environ))
    leaky = type("Leaky", (_ProbeSandbox,), {"output": f"PATH=/bin\nGEMINI_API_KEY={GEMINI_KEY}\n"})
    mods, sm = _probe_modules(leaky)
    with mock.patch.dict(sys.modules, mods):
        ok, why = TC.check_shell_sandbox()
    assert not ok and "GEMINI_API_KEY" in why and GEMINI_KEY not in why
    assert "must never see" in why and "docs/evals/tau3.md" in why
    assert isinstance(sm.subprocess, TC._ScrubbedSubprocess), "scrub installed before probing"
    clean = type("Clean", (_ProbeSandbox,), {"output": "PATH=/bin\nHOME=/tmp\n"})
    mods, _ = _probe_modules(clean)
    with mock.patch.dict(sys.modules, mods):
        assert TC.check_shell_sandbox() == (True, "")


# --------------------------------------------------------------------------- #
# the agent's endpoint key
# --------------------------------------------------------------------------- #
def test_agent_guard_adds_the_endpoint_key_to_agent_calls_only(monkeypatch, clean_state):
    calls = []
    guarded = TC._agent_guard(lambda *a, **k: calls.append(k) or "ok")
    monkeypatch.setattr(TC, "_SECRETS", [GEMINI_KEY])
    TC._AGENT_ROUTE.update({"model": "openai/m", "api_base": "http://x/v1", "api_key": ENDPOINT_KEY})
    msgs = [{"role": "tool", "content": f"env: GEMINI_API_KEY={GEMINI_KEY}"}]
    assert guarded(model="openai/m", api_base="http://x/v1", messages=msgs) == "ok"
    assert calls[-1]["api_key"] == ENDPOINT_KEY
    assert calls[-1]["messages"] == [{"role": "tool", "content": "env: GEMINI_API_KEY=[REDACTED]"}]
    assert msgs[0]["content"].endswith(GEMINI_KEY), "the caller's list is not mutated"
    guarded(model="gemini/gemini-3.6-flash", messages=msgs)          # user simulator / judge
    assert "api_key" not in calls[-1] and calls[-1]["messages"] is msgs
    guarded(model="openai/m", api_base="http://elsewhere/v1", messages=[])
    assert "api_key" not in calls[-1]
    TC._AGENT_ROUTE.clear()
    guarded(model="openai/m", api_base="http://x/v1", messages=[])
    assert "api_key" not in calls[-1]
    assert guarded._gbench_agent_guard is True


def test_agent_args_carry_no_key(monkeypatch, clean_state):
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    args = TC._build_agent_args("http://x/v1", False, "tau2")
    assert "api_key" not in args and args["api_base"] == "http://x/v1"
    assert OPENAI_KEY not in json.dumps(args)


# --------------------------------------------------------------------------- #
# what enters the image
# --------------------------------------------------------------------------- #
_HOST = {"OPENAI_API_KEY": ENDPOINT_KEY, "GBENCH_TAU2_OPENAI_API_KEY": OPENAI_KEY,
         "GEMINI_API_KEY": GEMINI_KEY, "ANTHROPIC_API_KEY": "anthropic-secret-6666",
         "OPENROUTER_API_KEY": "openrouter-secret-7777", "DEEPSEEK_API_KEY": "deepseek-secret-8",
         "GBENCH_TAU2_OPENAI_BASE_URL": "http://gateway/v1", "OPENAI_BASE_URL": "http://local/v1",
         "GBENCH_TAU2_EMBED_API_KEY": "old-gemini-embed-key", "GBENCH_TAU2_EMBED_MODEL": "g-e-1",
         "GBENCH_TAU2_SECRET_TOKEN": "x" * 12, "PATH": "/usr/bin"}
_CREDS = {"GEMINI_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE",
          "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_OPENAI_BASE_URL"}


@pytest.mark.parametrize("domains, knobs", [
    (["airline"], {}),
    (["banking_knowledge"], {}),
    (["banking_knowledge"], {"GBENCH_TAU2_RETRIEVAL_CONFIG": "alltools-qwen"}),
    (["banking_knowledge"], {"GBENCH_TAU2_RETRIEVAL_CONFIG": "bm25"}),
    (["banking_knowledge"], {"GBENCH_TAU2_RETRIEVAL_CONFIG": "qwen_embeddings_reranker"}),
    (["retail"], {"GBENCH_TAU2_USER_LLM": "gpt-4.1-2025-04-14",
                  "GBENCH_TAU2_EVAL_LLM": "anthropic/claude-x"}),
])
def test_only_the_gemini_key_is_forwarded(domains, knobs):
    """Whatever the domains, roles or retrieval config, no provider credential of the host enters
    the image (a non-Gemini role or a config that calls another provider is refused before or
    inside the container, never given that provider's key); the only Gemini "key" it gets is the
    placeholder of gbench's Gemini relay, from the launcher's bridge variables."""
    env = dict(_HOST, **knobs)
    names = set(TC._forwarded_env(env, domains=domains))
    assert names & _CREDS == set()
    placeholder = {"GEMINI_API_KEY": "gbench-gemini-relay-" + "0" * 48}
    assert set(TC._forwarded_env(env, domains=domains, bridge_env=placeholder)) & _CREDS == \
        {"GEMINI_API_KEY"}
    assert TC._launch_env(env, bridge_env=placeholder)["GEMINI_API_KEY"] == \
        placeholder["GEMINI_API_KEY"]
    assert TC._AGENT_KEY_ENV in names
    assert "GBENCH_TAU2_OPENAI_API_KEY" not in names and "GBENCH_TAU2_SECRET_TOKEN" not in names
    assert "GBENCH_TAU2_OPENAI_BASE_URL" not in names
    assert not names & set(TC._REMOVED_KNOBS)


def test_launch_env_maps_the_endpoint_and_gemini_keys():
    """No real key reaches the docker CLI's environment for the container: a real endpoint key
    becomes EMPTY (gbench's endpoint relay adds it), the Gemini keys are left out (gbench's
    Gemini relay adds them) and GBENCH_TAU_AGENT_KEY is always EMPTY."""
    env = TC._launch_env(dict(_HOST, GEMINI_API_KEYS=f"{GEMINI_KEY};other-key-9",
                              GEMINI_API_KEY="single-key-0000"))
    assert env[TC._AGENT_KEY_ENV] == "EMPTY"
    assert "OPENAI_API_KEY" not in env, "the image gets no OPENAI_API_KEY"
    assert "OPENAI_BASE_URL" not in env and "OPENAI_API_BASE" not in env
    assert "GEMINI_API_KEY" not in env and "GEMINI_API_KEYS" not in env
    assert not any(s in v for v in env.values() for s in (ENDPOINT_KEY, GEMINI_KEY))
    bare = TC._launch_env({"OPENAI_API_KEY": ENDPOINT_KEY, "OPENAI_BASE_URL": "http://local/v1",
                           "PATH": "/usr/bin"})
    assert bare[TC._AGENT_KEY_ENV] == "EMPTY"
    assert "OPENAI_API_KEY" not in bare and "OPENAI_BASE_URL" not in bare
    assert TC._launch_env({"PATH": "/usr/bin"})[TC._AGENT_KEY_ENV] == "EMPTY"
    # a placeholder is no key: the agent sends EMPTY
    assert TC._launch_env({"OPENAI_API_KEY": "dummy"})[TC._AGENT_KEY_ENV] == "EMPTY"
    assert "GEMINI_API_KEY" not in TC._launch_env({"GEMINI_API_KEY": GEMINI_KEY})


def test_streamed_container_output_is_redacted(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", ENDPOINT_KEY)
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    child = ("import os, sys, time\n"
             "k = os.environ['GBENCH_TAU_AGENT_KEY']\n"
             "sys.stdout.write('agent key ' + k[:9]); sys.stdout.flush(); time.sleep(0.3)\n"
             "sys.stdout.write(k[9:] + '\\r progress\\n'); sys.stdout.flush()\n"
             "print('gemini ' + os.environ['GEMINI_API_KEY'] + ' no newline at end', end='')\n")
    env = dict(os.environ, GBENCH_TAU_AGENT_KEY=ENDPOINT_KEY)
    code, tail = TC._launch_image([sys.executable, "-c", child], env)
    err = capsys.readouterr().err
    assert code == 0
    assert "agent key [REDACTED]" in err and "gemini [REDACTED] no newline at end" in err
    assert not any(s in err or s in tail for s in ALL_SECRETS)


# --------------------------------------------------------------------------- #
# end to end over a stand-in tau2 that writes its files the way tau2 does
# --------------------------------------------------------------------------- #
class _StandInTau2:
    """The tau2 surface the credential guards patch, with upstream's attribute names:
    utils.llm_utils.completion, data_model.simulation.Results.save (+ TextRunConfig),
    runner.checkpoint.json, environment.environment.Environment.get_response,
    knowledge.sandbox_manager.subprocess, run.run_domain, metrics.agent_metrics.compute_metrics,
    and a litellm stand-in."""

    def __init__(self, gemini_key):
        self.gemini_key = gemini_key
        self.configs, self.agent_calls, self.tool_outputs = [], [], []
        self.tool_contents, self.on_disk = [], []

    def modules(self, drop=()):
        me = self
        names = ["tau2", "tau2.utils", "tau2.utils.llm_utils", "tau2.data_model",
                 "tau2.data_model.simulation", "tau2.runner", "tau2.runner.checkpoint",
                 "tau2.environment", "tau2.environment.environment", "tau2.knowledge",
                 "tau2.knowledge.sandbox_manager", "tau2.run", "tau2.metrics",
                 "tau2.metrics.agent_metrics", "litellm"]
        m = {n: types.ModuleType(n) for n in names if n not in drop}
        for n in list(m):
            if "." in n:
                parent, child = n.rsplit(".", 1)
                if parent in m:
                    setattr(m[parent], child, m[n])

        def agent_completion(**kwargs):
            me.agent_calls.append(kwargs)
            return {"choices": [{"message": {"content": "ok"}}]}

        m["litellm"].completion = agent_completion
        if "tau2.utils.llm_utils" in m:
            m["tau2.utils.llm_utils"].completion = agent_completion

        class TextRunConfig:
            def __init__(self, **kw):
                self.__dict__.update(kw)

        class Results:
            def __init__(self, info):
                self.info = info

            def model_dump_json(self, indent=None):
                return json.dumps({"info": self.info, "simulations": []}, indent=indent)

            def save(self, path, format="json"):
                with open(path, "w") as fh:
                    fh.write(self.model_dump_json(indent=2))

        m["tau2.data_model.simulation"].TextRunConfig = TextRunConfig
        m["tau2.data_model.simulation"].Results = Results
        if "tau2.runner.checkpoint" in m:
            m["tau2.runner.checkpoint"].json = json
        sm = m["tau2.knowledge.sandbox_manager"]
        sm.subprocess = subprocess

        class ToolMessage:
            def __init__(self, content):
                self.content = content

        class Environment:
            def make_tool_call(self, name, **kwargs):
                out = sm.subprocess.run([sys.executable, "-c", "import os; print(dict(os.environ))"],
                                        capture_output=True, text=True, timeout=60)
                me.tool_outputs.append(out.stdout)
                return out.stdout + f"\nError: upstream said key={me.gemini_key} is invalid"

            def get_response(self, message):
                return ToolMessage(self.make_tool_call(message))

        m["tau2.environment.environment"].Environment = Environment

        def run_domain(config):
            me.configs.append(config)
            save_path = os.path.join(config.save_to, "results.json")
            os.makedirs(config.save_to, exist_ok=True)
            Results({"agent_info": {"llm": config.llm_agent, "llm_args": config.llm_args_agent},
                     "note": f"retried after: invalid key {me.gemini_key}"}).save(save_path)
            me.on_disk.append(open(save_path).read())         # as written, before any sweep
            tool = Environment().get_response("shell")
            me.tool_contents.append(tool.content)
            m["tau2.utils.llm_utils"].completion(
                model=config.llm_agent, api_base=config.llm_args_agent["api_base"],
                messages=[{"role": "tool", "content": tool.content},
                          {"role": "user", "content": f"my key is {me.gemini_key}"}])
            ckpt = {"simulations": [{"messages": [tool.content, f"error: {me.gemini_key}"]}]}
            with open(save_path, "w") as fh:
                m["tau2.runner.checkpoint"].json.dump(ckpt, fh, indent=2)
            me.on_disk.append(open(save_path).read())
            # a file written by another tau2 path (sim_status.json keeps the raw error text)
            with open(os.path.join(config.save_to, "sim_status.json"), "w") as fh:
                fh.write(json.dumps({"error": f"auth failed for {me.gemini_key}"}))
            sims = [types.SimpleNamespace(reward_info=types.SimpleNamespace(reward=1.0),
                                          termination_reason="agent_stop")]
            return types.SimpleNamespace(simulations=sims, num_trials=1)

        m["tau2.run"].run_domain = run_domain
        m["tau2.metrics.agent_metrics"].compute_metrics = (
            lambda results: types.SimpleNamespace(pass_hat_ks={1: 1.0}))
        # A dropped module maps to None, so importing it fails whatever an earlier test
        # imported (leaving it out of the dict would keep the real module in sys.modules).
        m.update({n: None for n in drop})
        return m


@pytest.fixture
def stand_in(monkeypatch, tmp_path, clean_state):
    monkeypatch.setenv(TC._AGENT_KEY_ENV, ENDPOINT_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    monkeypatch.setenv("HTTPS_PROXY", f"http://user:{PROXY_PASS}@proxy:3128")
    monkeypatch.setenv("GBENCH_TAU2_SAVE_TRACES", str(tmp_path / "traces"))
    monkeypatch.setenv("GBENCH_TAU2_NUM_TRIALS", "1")
    for name, value in (("check_tau_env_prerequisites", lambda: (True, "")),
                        ("check_banking_prerequisites", lambda: (True, "")),
                        ("_install_robustness_patches", lambda *a, **k: None),
                        ("_configure_evaluator_llm", lambda *a, **k: None),
                        ("_quiet_tau2_noise", lambda: None),
                        ("_prune_stale_search_sandboxes", lambda: None),
                        ("_refuse_resume_prompts", lambda: None),
                        ("_tau2_version", lambda: "1.0.1"),
                        # the Gemini retrieval setup has its own end-to-end credential test
                        # (tests/test_tau3_gemini_embeddings.py)
                        ("_setup_banking_retrieval", lambda: ("alltools-gemini", None))):
        monkeypatch.setattr(TC, name, value)
    return _StandInTau2(GEMINI_KEY)


def test_no_credential_reaches_the_agent_or_any_produced_file(stand_in, tmp_path):
    with mock.patch.dict(sys.modules, stand_in.modules()):
        r = TC.run_tau_env("tau3", ["banking_knowledge"], "m", "http://x/v1", 1, None,
                           "docs/evals/tau3.md")
    # the shell tool's command saw no credential, and the tool result was redacted at the source
    assert stand_in.tool_outputs and not any(s in stand_in.tool_outputs[0] for s in ALL_SECRETS)
    assert "key=[REDACTED] is invalid" in stand_in.tool_contents[0]
    # results.json is redacted as it is written (the initial save and each checkpoint append)
    assert len(stand_in.on_disk) == 2 and "invalid key [REDACTED]" in stand_in.on_disk[0]
    assert not any(s in text for text in stand_in.on_disk for s in ALL_SECRETS)
    # the agent authenticated with the endpoint key, and saw no credential in its messages
    call = stand_in.agent_calls[0]
    assert call["api_key"] == ENDPOINT_KEY
    assert not any(s in json.dumps(call["messages"]) for s in ALL_SECRETS)
    assert "[REDACTED]" in json.dumps(call["messages"])
    assert "api_key" not in stand_in.configs[0].llm_args_agent
    # every file tau2 wrote, and the result, hold none of the values
    traces = str(tmp_path / "traces")
    written = [f for _d, _s, fs in os.walk(traces) for f in fs]
    assert {"results.json", "sim_status.json"} <= set(written)
    assert _files_with_secrets(traces) == []
    assert not any(s in json.dumps(r) for s in ALL_SECRETS)
    assert r["tau2_report"]["credentials_redacted"] is True
    # nothing is kept once the run is over
    assert TC._AGENT_KEY_ENV not in os.environ and TC._AGENT_ROUTE == {} and TC._SECRETS == []


def test_missing_guard_targets_are_a_hard_error(stand_in):
    with mock.patch.dict(sys.modules, stand_in.modules(drop=("tau2.runner.checkpoint",))):
        with pytest.raises(RuntimeError) as ei:
            TC.run_tau_env("tau2", ["airline"], "m", "http://x/v1", 1, None, "docs/evals/tau2.md")
    assert str(ei.value).startswith("[INFRA REQUIRED]") and "airline" in str(ei.value)
    assert stand_in.configs == [], "tau2 never ran without the guards"


def test_image_main_redacts_the_error_it_writes(monkeypatch, tmp_path, clean_state):
    monkeypatch.setenv(TC._AGENT_KEY_ENV, ENDPOINT_KEY)
    with open(tmp_path / "job.json", "w") as fh:
        json.dump({"eval_name": "tau2", "domains": ["airline"], "model_name": "m",
                   "base_url": "http://x/v1", "concurrency": 1, "docs_url": "d"}, fh)

    def boom(*a, **k):
        os.environ.pop(TC._AGENT_KEY_ENV, None)
        raise RuntimeError(f"401 for bearer {ENDPOINT_KEY}")

    monkeypatch.setattr(TC, "run_tau_env", boom)
    assert TC._image_main(str(tmp_path)) == 1
    text = (tmp_path / "result.json").read_text()
    assert ENDPOINT_KEY not in text and "401 for bearer [REDACTED]" in text


def test_host_redacts_what_comes_back_from_the_image(monkeypatch, tmp_path, clean_state):
    monkeypatch.setenv("OPENAI_API_KEY", ENDPOINT_KEY)
    monkeypatch.setenv("GBENCH_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(TC, "check_tau_image", lambda: (True, ""))
    # the network checks before the container are docker calls of their own
    # (tests/test_guard_code_images.py)
    monkeypatch.setattr(TC, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC.container_url(url))
    monkeypatch.setattr(TC.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, b"", b""))
    # a real endpoint key: only the metadata guard runs on the bridge (gbench's endpoint relay
    # reaches the endpoint), then the relays start (tests/test_tau_gemini_cascade_and_relays.py
    # covers both)
    from gbench.runners.eval_suites import container_guard
    monkeypatch.setattr(container_guard, "require_metadata_blocked", lambda *a, **k: None)

    @contextlib.contextmanager
    def bridge(eval_name, image, base_url, agent_base_url, domains, docs_url):
        b = TC._TauBridge("http://172.17.0.1:4001/v1")
        b.env = {"GEMINI_API_KEY": "gbench-gemini-relay-" + "0" * 48, TC._AGENT_KEY_ENV: "EMPTY"}
        yield b
    monkeypatch.setattr(TC, "_bridge_services", bridge)

    def launch(cmd, env=None):
        assert not any(ENDPOINT_KEY in v for v in (env or {}).values()), "no key for the image"
        io_dir = next(cmd[i + 1].split(":")[0] for i, a in enumerate(cmd)
                      if a == "-v" and cmd[i + 1].split(":")[1] == "/gbench_io")
        job = open(os.path.join(io_dir, "job.json")).read()
        assert ENDPOINT_KEY not in job, "job.json is readable inside the container"
        assert json.loads(job)["base_url"] == "http://172.17.0.1:4001/v1"
        assert not any("=" in cmd[i + 1] and ENDPOINT_KEY in cmd[i + 1]
                       for i, a in enumerate(cmd) if a == "-e")
        with open(os.path.join(io_dir, "result.json"), "w") as fh:
            json.dump({"ok": False, "error_type": "ValueError",
                       "error": f"bad key {ENDPOINT_KEY}"}, fh)
        return 1, f"tail {ENDPOINT_KEY}"

    monkeypatch.setattr(TC, "_launch_image", launch)
    with pytest.raises(RuntimeError) as ei:
        TC.run_tau_suite("tau2", ["airline"], "m", "http://x/v1", 1, None, "docs/evals/tau2.md")
    assert ENDPOINT_KEY not in str(ei.value) and "bad key [REDACTED]" in str(ei.value)


def test_removed_embedder_knobs_are_reported_not_used(monkeypatch, caplog, clean_state):
    monkeypatch.setenv("GBENCH_TAU2_EMBED_MODEL", "text-embedding-004")
    monkeypatch.setenv("GBENCH_TAU2_OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.setattr(TC, "check_tau_image", lambda: (False, "no image"))
    with caplog.at_level("WARNING"), pytest.raises(RuntimeError):
        TC.run_tau_suite("tau3", ["banking_knowledge"], "m", "http://x/v1", 1, None,
                         "docs/evals/tau3.md")
    assert "GBENCH_TAU2_EMBED_MODEL, GBENCH_TAU2_OPENAI_API_KEY no longer change anything" \
        in caplog.text
    assert "gemini-embedding-001" in caplog.text and OPENAI_KEY not in caplog.text


# --------------------------------------------------------------------------- #
# tau3 retrieval: upstream's configuration, dense search on Gemini embeddings
# --------------------------------------------------------------------------- #
def _spec(kind, embedder=None, model=None, reranker=False):
    return types.SimpleNamespace(type=kind, embedder_type=embedder, embedder_model=model,
                                 reranker=reranker)


def _registry():
    variants = {
        "alltools": types.SimpleNamespace(kb_search=None, kb_search_bm25=_spec("bm25"),
                                          kb_search_dense=_spec("embedding", "openai",
                                                                "text-embedding-3-large"),
                                          shell=object()),
        "alltools-qwen": types.SimpleNamespace(kb_search=None, kb_search_bm25=_spec("bm25"),
                                               kb_search_dense=_spec("embedding", "openrouter",
                                                                     "qwen3-embedding-8b"),
                                               shell=object()),
        "bm25_reranker": types.SimpleNamespace(kb_search=_spec("bm25", reranker=True), shell=None),
        "bm25": types.SimpleNamespace(kb_search=_spec("bm25"), shell=None),
    }
    retrieval = types.ModuleType("tau2.domains.banking_knowledge.retrieval")
    retrieval.RETRIEVAL_VARIANTS = variants
    retrieval.RETRIEVAL_VARIANT_ALIASES = {"AllTools": "alltools"}
    names = ("tau2", "tau2.domains", "tau2.domains.banking_knowledge", "tau2.knowledge",
             "tau2.knowledge.document_preprocessors",
             "tau2.knowledge.document_preprocessors.embedding_indexer",
             "tau2.knowledge.input_preprocessors",
             "tau2.knowledge.input_preprocessors.embedding_encoder")
    mods = {n: types.ModuleType(n) for n in names}
    mods["tau2.domains.banking_knowledge.retrieval"] = retrieval
    for n in list(mods):
        parent, _, child = n.rpartition(".")
        if parent in mods:
            setattr(mods[parent], child, mods[n])
    for n in ("tau2.knowledge.document_preprocessors.embedding_indexer",
              "tau2.knowledge.input_preprocessors.embedding_encoder"):
        mods[n].EMBEDDER_REGISTRY = {"openai": object, "openrouter": object}
    return mods


# The removed GBENCH_TAU2_RETRIEVAL_CONFIG changes nothing: the Gemini key decides.
@pytest.mark.parametrize("config, env, fragment", [
    (None, {}, "GEMINI_API_KEY"),
    ("AllTools", {}, "GEMINI_API_KEY"),
    (None, {"OPENAI_API_KEY": OPENAI_KEY}, "GEMINI_API_KEY"),      # no OpenAI stand-in
    (None, {"GEMINI_API_KEY": GEMINI_KEY}, None),
    ("AllTools", {"GEMINI_API_KEYS": f"{GEMINI_KEY},second-key-5555"}, None),
    ("alltools-qwen", {"GEMINI_API_KEY": GEMINI_KEY, "OPENROUTER_API_KEY": "r" * 10}, None),
    ("bm25_reranker", {"GEMINI_API_KEY": GEMINI_KEY}, None),
    ("bm25", {}, "GEMINI_API_KEY"),
    ("alltools-gemini", {"GEMINI_API_KEY": GEMINI_KEY}, None),
])
def test_retrieval_credentials_follow_the_config(monkeypatch, clean_state, config, env, fragment):
    for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "GEMINI_API_KEYS"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    if config:
        monkeypatch.setenv("GBENCH_TAU2_RETRIEVAL_CONFIG", config)
    with mock.patch.dict(sys.modules, _registry()):
        ok, why = TC.check_retrieval_credentials()
    if fragment is None:
        assert (ok, why) == (True, "")
    else:
        assert not ok and fragment in why and "docs/evals/tau3.md" in why


def test_default_retrieval_is_alltools_with_gemini_embeddings(monkeypatch, clean_state):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    monkeypatch.setattr(TC, "_LAST_RETRIEVAL", {})
    monkeypatch.setattr(TC, "_warm_gemini_document_embeddings", lambda: 698)
    before = dict(os.environ)
    mods = _registry()
    with mock.patch.dict(sys.modules, mods):
        name, kwargs = TC._setup_banking_retrieval()
    assert (name, kwargs) == ("alltools-gemini", None)
    variants = mods["tau2.domains.banking_knowledge.retrieval"].RETRIEVAL_VARIANTS
    ours, upstream = variants["alltools-gemini"], variants["alltools"]
    assert ours.shell is not None, "alltools' shell tool is kept"
    assert (ours.kb_search_dense.embedder_type, ours.kb_search_dense.embedder_model) == (
        "gemini", "gemini-embedding-001")
    assert (upstream.kb_search_dense.embedder_type, upstream.kb_search_dense.embedder_model) == (
        "openai", "text-embedding-3-large"), "upstream's alltools is left as it is"
    for n in ("tau2.knowledge.document_preprocessors.embedding_indexer",
              "tau2.knowledge.input_preprocessors.embedding_encoder"):
        assert mods[n].EMBEDDER_REGISTRY["gemini"] is TC.GeminiEmbedder
    assert TC._LAST_RETRIEVAL["retrieval_config"] == "alltools"
    assert TC._LAST_RETRIEVAL["retrieval_variant"] == "alltools-gemini"
    assert TC._LAST_RETRIEVAL["embedders"] == ["gemini:gemini-embedding-001"]
    assert TC._LAST_RETRIEVAL["embedder"]["upstream_embedder"] == "openai:text-embedding-3-large"
    assert dict(os.environ) == before, "no environment is rewritten"
    assert TC._UPSTREAM_RETRIEVAL_CONFIG == "alltools"


@pytest.mark.parametrize("config", [None, "alltools", "AllTools", "bm25", "alltools-qwen"])
def test_retrieval_gap_is_the_embedding_substitution_whatever_the_removed_knob(
        monkeypatch, clean_state, config):
    monkeypatch.setenv("GBENCH_TAU2_USER_LLM", "gpt-4.1-2025-04-14")
    monkeypatch.setenv("GBENCH_TAU2_EVAL_LLM", "gpt-4.1-2025-04-14")
    if config:
        monkeypatch.setenv("GBENCH_TAU2_RETRIEVAL_CONFIG", config)
    monkeypatch.setattr(TC, "_LAST_AGENT_TEMPERATURE", {"value": 0.0})
    monkeypatch.setattr(TC, "_LAST_RETRIEVAL", {"retrieval_config": config or "alltools"})
    gaps = TC._comparability_gaps(["banking_knowledge"], 4, None, 0)
    assert gaps == ["retrieval embeddings by gemini-embedding-001 "
                    "(upstream: text-embedding-3-large)"]


def test_the_gemini_embeddings_are_a_named_gap(monkeypatch, clean_state):
    monkeypatch.setenv("GBENCH_TAU2_USER_LLM", "gpt-4.1-2025-04-14")
    monkeypatch.setenv("GBENCH_TAU2_EVAL_LLM", "gpt-4.1-2025-04-14")
    monkeypatch.setenv("GBENCH_TAU2_OPENAI_BASE_URL", "http://gateway/v1")    # no effect
    monkeypatch.setattr(TC, "_LAST_AGENT_TEMPERATURE", {"value": 0.0})
    monkeypatch.setattr(TC, "_LAST_RETRIEVAL", {"retrieval_config": "alltools",
                                                "embedders": ["gemini:gemini-embedding-001"]})
    gaps = TC._comparability_gaps(["banking_knowledge"], 4, None, 0)
    assert gaps == ["retrieval embeddings by gemini-embedding-001 "
                    "(upstream: text-embedding-3-large)"]


@pytest.mark.parametrize("user, env, fragment", [
    ("openai/gpt-4.1-2025-04-14", {"OPENAI_API_KEY": OPENAI_KEY}, "'openai'"),
    ("gpt-5.2", {"OPENAI_API_KEY": OPENAI_KEY}, "'openai'"),
    ("anthropic/claude-x", {"ANTHROPIC_API_KEY": "a" * 10}, "'anthropic'"),
    ("claude-sonnet-x", {"ANTHROPIC_API_KEY": "a" * 10}, "'anthropic'"),
    ("openrouter/qwen", {"OPENROUTER_API_KEY": "r" * 10}, "'openrouter'"),
    ("gemini/gemini-3.6-flash", {}, "GEMINI_API_KEY"),
    ("gemini/gemini-3.6-flash", {"GEMINI_API_KEY": GEMINI_KEY}, None),
])
def test_role_credentials(monkeypatch, clean_state, user, env, fragment):
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY",
                 "GEMINI_API_KEYS"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("GBENCH_TAU2_USER_LLM", user)
    monkeypatch.setenv("GBENCH_TAU2_EVAL_LLM", user)
    ok, why = TC.check_role_credentials()
    if fragment is None:
        assert (ok, why) == (True, "")
    else:
        assert not ok and fragment in why
        if "'" in fragment:
            assert "only through the Gemini API" in why


def test_no_other_provider_path_remains():
    """The OpenAI / OpenRouter / Anthropic credential paths are gone: no such key can enter the
    image, and the former embedder patches are not back."""
    import inspect
    src = inspect.getsource(TC)
    for gone in ("_PROVIDER_CREDENTIALS", "_RETRIEVAL_PROVIDERS", "_needed_providers",
                 "_openai_in_use", "_HOST_KEY_NAMES", "_PROVIDER_KEY_VARS",
                 "_patch_openai_embedder_batch", "_patch_embedder_config_resolver",
                 "tau2-openai-gateway", "generativelanguage.googleapis.com/v1beta/openai/"):
        assert gone not in src, gone
    assert not hasattr(TC, "_AGENT_API_KEY")
    host = {"OPENAI_API_KEY": ENDPOINT_KEY, "GBENCH_TAU2_OPENAI_API_KEY": OPENAI_KEY,
            "OPENROUTER_API_KEY": "r" * 10, "ANTHROPIC_API_KEY": "a" * 10,
            "GBENCH_TAU2_OPENAI_BASE_URL": "http://gateway/v1", "GEMINI_API_KEY": GEMINI_KEY}
    env = TC._launch_env(host)
    passed = {name: env.get(name) or "" for name in TC._forwarded_env(host)}
    assert not any(s in v for v in passed.values() for s in (ENDPOINT_KEY, OPENAI_KEY, GEMINI_KEY))
    assert not hasattr(TC, "_credential_map"), "no host credential is mapped into the image"


# --------------------------------------------------------------------------- #
# docs
# --------------------------------------------------------------------------- #
def test_docs_state_the_retrieval_config_and_the_credential_handling():
    tau2 = " ".join(_read("docs", "evals", "tau2.md").split())
    tau3 = " ".join(_read("docs", "evals", "tau3.md").split())
    assert "upstream's **`alltools`**" in tau3 and "OpenAI `text-embedding-3-large`" in tau3
    assert "**`gemini-embedding-001`**" in tau3
    assert "retrieval embeddings by gemini-embedding-001 (upstream: text-embedding-3-large)" in tau3
    assert "GBENCH_TAU2_OPENAI_API_KEY" in tau3 and 'export GBENCH_TAU2_OPENAI_API_KEY' not in tau3
    assert "--max-output-tokens 65536" in tau3 and "--max-output-tokens 65536" in tau2
    assert f"`{TC._SHELL_PROBE_COMMAND}`" in tau3
    assert "### Credentials and the model under test" in _read("docs", "evals", "tau3.md")
    assert "### Credentials" in _read("docs", "evals", "tau2.md")
    assert "tau2_report.credentials_redacted" in tau2 and "tau2_report.credentials_redacted" in tau3
    assert "no longer change anything" in tau2 and "no longer change anything" in tau3
    assert "`alltools-gemini`" in tau3
    for doc in (tau2, tau3):
        assert "tau2-openai-gateway" not in doc and "OPENROUTER_API_KEY" not in doc
        assert "ANTHROPIC_API_KEY" not in doc
        assert "leaderboard's standard text entries use" not in doc
