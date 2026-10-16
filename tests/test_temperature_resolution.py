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

"""Temperature resolution: think-aware default (0.0 no-think / 1.0 --thinking),
--temperature everywhere, per-suite env override.

The default is the project baseline: a no-think run is greedy (0.0, canonical-comparable),
a --thinking run uses Gemma 4's shipped 1.0. The 1.0 side is measured - replaying the
2026-08-17 copilot_bench_swe prompts, 4 attempts each, only the temperature differing:

    prompt set          T=0.0          T=1.0        T=1.0 +top_p .95/top_k 64
    10 that looped      31/40 (78%)    16/40 (40%)  15/40 (38%)
    10 that did not     10/40 (25%)    -            3/40  (8%)

i.e. 0.0 loops far more - the no-think tradeoff, surfaced by the looping metrics, not
silently absorbed - while the reasoning pass makes 1.0 safe under --thinking. Override
either side with --temperature or the per-suite env var.

Precedence, highest first:
    GBENCH_<EVAL>_TEMPERATURE  >  --temperature  >  suite value  >  0.0 / 1.0 (think-aware)
"""

import glob
import os
import re
from unittest import mock

import pytest

from gbench.runners.eval_suites import base


@pytest.fixture(autouse=True)
def _clean_knobs():
    """Each test starts with no run knobs and no stray env overrides."""
    saved = dict(base._RUN_KNOBS)
    base._RUN_KNOBS.clear()
    drop = [k for k in os.environ if k.startswith("GBENCH_") and k.endswith("_TEMPERATURE")]
    with mock.patch.dict(os.environ, {k: "" for k in drop}, clear=False):
        for k in drop:
            os.environ.pop(k, None)
        yield
    base._RUN_KNOBS.clear()
    base._RUN_KNOBS.update(saved)


# --------------------------------------------------------------------------- #
# the default
# --------------------------------------------------------------------------- #
def test_default_is_think_aware():
    """Baseline: no-think defaults to 0.0 (greedy), --thinking to 1.0 (the shipped
    generation_config). 1.0 is what google/gemma-4-*-it ships."""
    assert base.DEFAULT_TEMPERATURE == 1.0
    assert base.NOTHINK_TEMPERATURE == 0.0
    assert base.resolve_temperature("aime") == (0.0, "default")
    assert base.resolve_temperature("aime", thinking=False) == (0.0, "default")
    assert base.resolve_temperature("aime", thinking=True) == (1.0, "thinking")


def test_thinking_uses_one_for_every_suite_whatever_its_official_temperature(monkeypatch):
    """The maintainer's rule: with thinking, 1.0 replaces a suite's own or official temperature
    (lcb's official 0.2 looped in 40 of 50 thinking replies); the operator still wins; no-think
    keeps the official value."""
    lcb = base.OfficialSampling(reference="lcb_runner", temperature=0.2, params={"top_p": 0.95})
    silent = base.OfficialSampling(reference="sends none", send_temperature=False)
    assert base.resolve_temperature("lcb", 0.2, thinking=True) == (1.0, "thinking")
    assert base.resolve_official_sampling("lcb", lcb, thinking=True) == (1.0, "thinking")
    assert base.resolve_official_sampling("lcb", silent, thinking=True) == (1.0, "thinking")
    # no-think: 0.0, whatever the official value (a temperature is always sent)
    assert base.resolve_temperature("lcb", 0.2, thinking=False) == (0.0, "default")
    assert base.resolve_official_sampling("lcb", lcb, thinking=False) == (0.0, "default")
    assert base.resolve_official_sampling("lcb", silent, thinking=False) == (0.0, "default")
    # the record says the thinking run did not match the official value
    assert lcb.record(1.0, "thinking")["temperature_matches_official"] is False
    # the operator's choice still wins in a thinking run
    base.set_run_knobs(temperature=0.6)
    assert base.resolve_official_sampling("lcb", lcb, thinking=True) == (0.6, "cli:--temperature")
    monkeypatch.setenv("GBENCH_LCB_TEMPERATURE", "0.3")
    assert base.resolve_official_sampling("lcb", lcb, thinking=True) == (
        0.3, "env:GBENCH_LCB_TEMPERATURE")


def test_a_suite_with_no_opinion_gets_the_default():
    assert base.resolve_temperature("bigcodebench", None)[0] == 0.0
    assert base.resolve_temperature("bigcodebench", None, thinking=True)[0] == 1.0


# --------------------------------------------------------------------------- #
# precedence
# --------------------------------------------------------------------------- #
def test_a_suite_value_never_applies():
    """The maintainer's rule: every suite samples at 0.0 (no-think) / 1.0 (thinking)."""
    assert base.resolve_temperature("screenspot", 0.6) == (0.0, "default")
    assert base.resolve_temperature("screenspot", 0.6, thinking=True) == (1.0, "thinking")


def test_cli_beats_the_suite_value():
    """--temperature is documented as applying across all benchmarks; a suite that
    encodes its own protocol must not silently ignore the operator."""
    base.set_run_knobs(temperature=0.3)
    assert base.resolve_temperature("mmmu_pro", 1.0) == (0.3, "cli:--temperature")


def test_cli_zero_is_honoured_and_not_mistaken_for_unset():
    """`--temperature 0.0` must restore greedy. This is why the CLI default is None."""
    base.set_run_knobs(temperature=0.0)
    assert base.resolve_temperature("aime", 1.0) == (0.0, "cli:--temperature")


def test_per_suite_env_beats_everything():
    base.set_run_knobs(temperature=0.3)
    with mock.patch.dict(os.environ, {"GBENCH_AIME_TEMPERATURE": "0.6"}):
        val, src = base.resolve_temperature("aime", 1.0)
    assert val == 0.6 and src == "env:GBENCH_AIME_TEMPERATURE"


def test_per_suite_env_only_affects_its_own_suite():
    with mock.patch.dict(os.environ, {"GBENCH_AIME_TEMPERATURE": "0.6"}):
        assert base.resolve_temperature("aime")[0] == 0.6
        assert base.resolve_temperature("hmmt")[0] == 0.0    # no-think default


def test_full_precedence_chain():
    chain = []
    chain.append(base.resolve_temperature("gsm8k"))                       # default
    chain.append(base.resolve_temperature("gsm8k", 0.2))                  # suite: not applied
    base.set_run_knobs(temperature=0.5)
    chain.append(base.resolve_temperature("gsm8k", 0.2))                  # cli
    with mock.patch.dict(os.environ, {"GBENCH_GSM8K_TEMPERATURE": "0.9"}):
        chain.append(base.resolve_temperature("gsm8k", 0.2))              # env
    assert [c[0] for c in chain] == [0.0, 0.0, 0.5, 0.9]   # default is no-think 0.0
    assert [c[1] for c in chain] == ["default", "default", "cli:--temperature",
                                     "env:GBENCH_GSM8K_TEMPERATURE"]


# --------------------------------------------------------------------------- #
# env var naming and robustness
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("eval_name,expected", [
    ("aime", "GBENCH_AIME_TEMPERATURE"),
    ("copilot_bench_swe", "GBENCH_COPILOT_BENCH_SWE_TEMPERATURE"),
    ("swe_bench_pro", "GBENCH_SWE_BENCH_PRO_TEMPERATURE"),
    ("mmmu_pro", "GBENCH_MMMU_PRO_TEMPERATURE"),
    ("bfcl_live", "GBENCH_BFCL_LIVE_TEMPERATURE"),
    ("perception-bench", "GBENCH_PERCEPTION_BENCH_TEMPERATURE"),
])
def test_env_var_name_is_derived_systematically(eval_name, expected):
    """No per-suite registry to maintain - the name follows from the eval name."""
    assert base.temperature_env_var(eval_name) == expected


def test_a_garbage_env_value_is_ignored_not_crashed_on():
    with mock.patch.dict(os.environ, {"GBENCH_AIME_TEMPERATURE": "hot"}):
        val, src = base.resolve_temperature("aime", 0.2)
    assert (val, src) == (0.0, "default")


def test_an_empty_env_value_is_treated_as_unset():
    with mock.patch.dict(os.environ, {"GBENCH_AIME_TEMPERATURE": "  "}):
        assert base.resolve_temperature("aime")[0] == 0.0    # falls to no-think default


# --------------------------------------------------------------------------- #
# it actually reaches the request, and is recorded
# --------------------------------------------------------------------------- #
def _run(**kw):
    sent = []

    async def _fake_send(**k):
        sent.append(k)
        return base.Reply(text="x", tool_calls=None, finish_reason="stop",
                          reasoning=None, error=None, completion_tokens=1, stop_reason=None)

    with mock.patch.object(base, "_send_single_request", _fake_send):
        res = base.run_eval_suite(
            eval_name="aime", model_name="m", base_url="http://x", concurrency=1,
            samples=[([{"role": "user", "content": "q"}], "x", {})],
            eval_fn=lambda r, g: True, **kw)
    return res, sent


def test_the_resolved_value_is_recorded_on_the_result():
    res, _ = _run()                            # no thinking -> greedy default
    assert res["temperature"] == 0.0
    assert res["temperature_source"] == "default"
    res_t, _ = _run(thinking=True)             # --thinking -> 1.0 default
    assert res_t["temperature"] == 1.0
    assert res_t["temperature_source"] == "thinking"


def test_the_source_is_recorded_so_a_run_can_say_what_it_measured():
    """A number produced at 1.0 is not comparable with a greedy leaderboard entry."""
    with mock.patch.dict(os.environ, {"GBENCH_AIME_TEMPERATURE": "0.6"}):
        res, _ = _run()
    assert (res["temperature"], res["temperature_source"]) == (
        0.6, "env:GBENCH_AIME_TEMPERATURE")


def test_env_override_reaches_the_actual_request_payload():
    with mock.patch.dict(os.environ, {"GBENCH_AIME_TEMPERATURE": "0.42"}):
        _, sent = _run()
    payload = sent[0].get("payload") or sent[0]
    assert payload["temperature"] == 0.42


# --------------------------------------------------------------------------- #
# no suite may re-introduce a hardcoded greedy fallback
# --------------------------------------------------------------------------- #
def test_no_suite_hardcodes_a_temperature_fallback():
    """`kwargs.get("temperature", 0.0)` pins a suite to greedy whenever it is called
    outside evals.py, which is how the old global default survived --temperature."""
    offenders = []
    for path in glob.glob(os.path.join(os.path.dirname(base.__file__), "*.py")):
        src = open(path, encoding="utf-8").read()
        if re.search(r'kwargs\.get\(\s*["\']temperature["\']\s*,\s*0\.0\s*\)', src):
            offenders.append(os.path.basename(path))
    assert not offenders, f"hardcoded greedy fallback in: {offenders}"


#: Data tables whose temperature fields are official sampling declarations, not request
#: payloads: (file, top-level name). hmmt.MATHARENA_MODEL_CONFIGS transcribes MathArena's
#: configs/models yaml files; hmmt.translate_matharena_config turns a config into an
#: OfficialSampling, and resolve_official_sampling lets --temperature and the env override win
#: (pinned by the test below).
OFFICIAL_SAMPLING_TABLES = {("hmmt.py", "MATHARENA_MODEL_CONFIGS")}


def _table_lines(path: str, name: str) -> set:
    import ast
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign) else [])
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return set(range(node.lineno, node.end_lineno + 1))
    raise AssertionError(f"{name} not found in {path}")


def test_no_suite_pins_temperature_in_a_per_sample_payload():
    """A `temperature` key in a sample's `extra` beats the resolved value, which would
    make --temperature and the env override inert for that suite."""
    offenders = []
    for path in glob.glob(os.path.join(os.path.dirname(base.__file__), "*.py")):
        name = os.path.basename(path)
        if name in ("base.py", "tau_common.py"):     # base builds it; tau2 judge stays 0.0
            continue
        skip = set()
        for table_file, table in OFFICIAL_SAMPLING_TABLES:
            if name == table_file:
                skip |= _table_lines(path, table)
        for lineno, line in enumerate(open(path, encoding="utf-8"), start=1):
            if lineno not in skip and re.match(r'\s*"temperature"\s*:\s*[0-9]', line):
                offenders.append(f"{name}: {line.strip()}")
    assert not offenders, f"per-sample temperature pin in: {offenders}"


def test_matharena_config_temperatures_yield_to_the_operator(monkeypatch):
    """The temperatures in hmmt.MATHARENA_MODEL_CONFIGS are declared as OfficialSampling, so the
    per-suite env override and --temperature still win over them."""
    from gbench.runners.eval_suites import hmmt
    path = "qwen/qwen3.5_27b"
    blob, _hf, fields = hmmt.MATHARENA_MODEL_CONFIGS[path]
    sampling, _extra, _na, _gaps = hmmt.translate_matharena_config(path, blob, fields, True)
    assert sampling.temperature == fields["temperature"] == 1.0
    assert base.resolve_official_sampling("hmmt", sampling, thinking=True) == (1.0, "thinking")
    base.set_run_knobs(temperature=0.3)
    assert base.resolve_official_sampling("hmmt", sampling, thinking=True)[0] == 0.3
    monkeypatch.setenv("GBENCH_HMMT_TEMPERATURE", "0.7")
    assert base.resolve_official_sampling("hmmt", sampling, thinking=True)[0] == 0.7


# --------------------------------------------------------------------------- #
# CLI wiring
# --------------------------------------------------------------------------- #
def test_cli_default_is_none_so_explicit_zero_is_distinguishable():
    from gbench.cli import create_parser
    p = create_parser()
    assert p.parse_args(["--evals-only"]).temperature is None
    assert p.parse_args(["--evals-only", "--temperature", "0.0"]).temperature == 0.0
    assert p.parse_args(["--evals-only", "--temperature", "1.0"]).temperature == 1.0


def test_cli_does_not_coerce_unset_to_zero():
    import inspect
    from gbench import cli
    assert 'config.temperature = getattr(args, "temperature", None)' in inspect.getsource(cli)


def test_evals_passes_none_through_rather_than_defaulting_to_zero():
    import inspect
    from gbench.runners import evals
    assert '"temperature": getattr(self.config, "temperature", None)' in inspect.getsource(evals)


# --------------------------------------------------------------------------- #
# every eval documents its own sampling contract
# --------------------------------------------------------------------------- #
def _builtin_evals():
    from gbench.runners.evals import BUILTIN_PILLARS
    return sorted({s for _, su in BUILTIN_PILLARS for s in su})


def test_every_builtin_eval_documents_its_sampling_in_the_module_header():
    """A reader opening any suite must see the default, the global override and the
    per-suite override without going to base.py."""
    import ast
    import importlib
    missing = []
    for name in _builtin_evals():
        mod = importlib.import_module(f"gbench.runners.eval_suites.{name}")
        doc = ast.get_docstring(ast.parse(open(mod.__file__, encoding="utf-8").read())) or ""
        want = ["Sampling:", "--temperature", base.temperature_env_var(name)]
        absent = [w for w in want if w not in doc]
        if absent:
            missing.append((name, absent))
    assert not missing, f"header missing sampling contract: {missing}"


def test_every_builtin_eval_header_names_its_own_env_var():
    """A copy-pasted header pointing at another suite's variable is worse than none."""
    import ast
    import importlib
    wrong = []
    for name in _builtin_evals():
        mod = importlib.import_module(f"gbench.runners.eval_suites.{name}")
        doc = ast.get_docstring(ast.parse(open(mod.__file__, encoding="utf-8").read())) or ""
        others = [base.temperature_env_var(o) for o in _builtin_evals()
                  if o != name and base.temperature_env_var(o) != base.temperature_env_var(name)]
        leaked = [o for o in others if o in doc]
        if leaked:
            wrong.append((name, leaked))
    assert not wrong, f"header names another suite's env var: {wrong}"


def test_headers_state_that_judges_are_pinned():
    import ast
    import importlib
    for name in _builtin_evals()[:5]:
        mod = importlib.import_module(f"gbench.runners.eval_suites.{name}")
        doc = ast.get_docstring(ast.parse(open(mod.__file__, encoding="utf-8").read())) or ""
        assert "judge" in doc.lower() and "0.0" in doc


# --------------------------------------------------------------------------- #
# judges are deterministic and independent of the run's sampling knob
# --------------------------------------------------------------------------- #
def test_judge_temperature_is_zero_and_not_tied_to_the_model_under_test():
    assert base.JUDGE_TEMPERATURE == 0.0
    base.set_run_knobs(temperature=1.0)
    cfg = base.judge_config()
    if cfg is not None:                       # None only when google-genai is absent
        assert cfg.temperature == 0.0


def test_every_judge_call_passes_the_deterministic_config():
    """Until 2026-08-17 all 15 sites passed no config, so grading ran at the Gemini API
    default of 1.0 - undocumented variance on top of every judge-scored suite."""
    import glob
    import os
    import re
    # A suite whose upstream grader fixes its own sampling follows it (the judge model is the only
    # allowed substitution): toolbench's ToolEval calls run at config.yaml's temperature 0.2 with
    # its max_tokens 1000, built by toolbench._judge_generation_config and passed as
    # `config=config`;
    # tests/test_agent_token_policy_and_mcp_bench_launcher.py::test_toolbench_tooleval_config_is_the_official_one
    # checks those values.
    official_grader_configs = {"toolbench.py": "config=config"}
    offenders = []
    for path in glob.glob(os.path.join(os.path.dirname(base.__file__), "*.py")):
        src = open(path, encoding="utf-8").read()
        calls = len(re.findall(r"generate_content\(", src))
        # The central judge cascade (base.judge_generate_cascade) passes `config=cfg`, where
        # cfg defaults to judge_config() - deterministic by construction. Count that too.
        configured = src.count("config=judge_config()") + src.count("config=cfg")
        official = official_grader_configs.get(os.path.basename(path))
        if official:
            configured += src.count(official)
        if calls and configured < calls:
            offenders.append((os.path.basename(path), calls, configured))
    assert not offenders, f"judge calls without judge_config(): {offenders}"
