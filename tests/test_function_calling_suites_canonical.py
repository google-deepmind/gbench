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

"""Canonical-fidelity regressions for gorilla_apibench, nestful, seal_tools,
nexusraven_api_eval (was nexus_function_calling), acebench, and the api_bank / t_eval / agent_dojo pins.

The expected hashes below were computed from the upstream code itself (Gorilla
get_llm_responses.encode_question, NESTFUL deepseek_prompt_input + get_icl_str, LangChain 0.0.294
StructuredChatAgent.create_prompt), so a drift from the verbatim prompts fails here."""
import hashlib
import json
import warnings

warnings.filterwarnings("ignore")

import pytest

from gbench.runners.eval_suites import acebench as AB
from gbench.runners.eval_suites import agent_dojo as AD
from gbench.runners.eval_suites import api_bank as API
from gbench.runners.eval_suites import gorilla_apibench as G
from gbench.runners.eval_suites import nestful as N
from gbench.runners.eval_suites import nexusraven_api_eval as NX
from gbench.runners.eval_suites import seal_tools as ST
from gbench.runners.eval_suites import t_eval as TE


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _fake_result(traces, **extra):
    return {"sample_traces": traces, "accuracy": 0.0, "temperature": 0.0, "thinking": False,
            "decoding_penalties": None, "attempts": None, "status": "success", **extra}


# --------------------------------------------------------------------------- #
# gorilla_apibench
# --------------------------------------------------------------------------- #
def test_gorilla_prompt_is_gorillas_encode_question():
    """No gold domain or exemplar call: the prompt lists the hub's domains and the model picks."""
    expected = {"huggingface": "204f19de7ab9915bd5b6cbc28aaf097d5c8ed57996159dda487f59a5a618908b",
                "torchhub": "018e7aef24018c5b815edd9ae34d3b599e44da278485514c2148840fa90b3f0b",
                "tensorhub": "d0624bcaaf7e4b7c9c82828853f6e81cb25ce4aae25c393fff05aad263835393"}
    for api_name, h in expected.items():
        assert _sha(repr(G.encode_question("QUESTION", api_name))) == h
    msgs = G.encode_question(" Classify tweets.\\n", "huggingface")
    assert msgs[0] == {"role": "system",
                       "content": "You are a helpful API writer who can write APIs based on requirements."}
    assert "AutoModel.from_pretrained('org/checkpoint-name')" not in msgs[1]["content"]


def test_gorilla_extracts_the_api_call_field_like_ast_eval():
    text = ("<<<domain>>> NLP, <<<api_call>>>: pipeline('sentiment-analysis', model='x'), "
            "<<<api_provider>>>: Hugging Face")
    assert G.extract_api_call(text, "huggingface") == "pipeline('sentiment-analysis', model='x')"
    assert G.extract_api_call("pipeline('a')", "huggingface") == "pipeline('a')"
    # ast_eval_th.py: no `api_call` field -> neither correct nor hallucinated
    assert G.extract_api_call("torch.hub.load('a', 'b')", "torchhub") is None


class _Node:
    def __init__(self, text, children=()):
        self.text, self.children = text.encode(), list(children)
        self.child_count = len(self.children)


def _call_tree(fn, args):
    """Hand-built module -> expression_statement -> call tree shaped like tree-sitter's."""
    kids = [_Node("(")]
    for i, (k, v) in enumerate(args):
        if i:
            kids.append(_Node(","))
        val = _Node(f"'{v}'", [_Node("'"), _Node(v), _Node("'")])
        kids.append(_Node(f"{k}='{v}'", [_Node(k), _Node("="), val]) if k else val)
    kids.append(_Node(")"))
    src = f"{fn}(" + ", ".join(f"{k}='{v}'" if k else f"'{v}'" for k, v in args) + ")"
    call = _Node(src, [_Node(fn, [_Node(fn)]), _Node(src[len(fn):], kids)])
    return _Node(src, [_Node(src, [call])])


def test_gorilla_get_args_differs_per_hub_like_the_three_scripts():
    tree = _call_tree("torch.hub.load", [(None, "pytorch/vision"), ("model", "resnet18"),
                                         ("pretrained", "True")])
    assert G.get_args(tree, "huggingface") == [b"'pytorch/vision'", b"'resnet18'", b"'True'"]
    # ast_eval_tf.py: only `model=` keywords are unwrapped; other keywords fall to the
    # positional branch and must appear whole
    assert G.get_args(tree, "tensorflowhub") == [b"'pytorch/vision'", b"'resnet18'", b"pretrained='True'"]
    assert G.get_args(tree, "torchhub") == [b"'resnet18'"]


def test_gorilla_scores_domain_match_and_hallucination_against_the_database():
    """Correct = the matched database entry's domain equals the question's; no match = hallucination."""
    rows = [{"api_call": "pipeline('asr', model='openai/whisper-tiny')", "domain": "Audio ASR"},
            {"api_call": "pipeline('text-classification', model='distilbert')", "domain": "NLP TC"}]
    trees = {r["api_call"]: _call_tree("pipeline", [(None, r["api_call"].split("'")[1]),
                                                    ("model", r["api_call"].split("'")[3])])
             for r in rows}

    class _Parser:
        def parse(self, data):
            text = data.decode()
            return type("T", (), {"root_node": trees.get(text, _Node(text))})()

    db = G.HubDatabase("huggingface", rows, parser=_Parser())
    ok = "<<<api_call>>>: pipeline('asr', model='openai/whisper-tiny'), <<<api_provider>>>: HF"
    other = "<<<api_call>>>: pipeline('text-classification', model='distilbert'), <<<api_provider>>>: HF"
    assert db.score(ok, "Audio ASR") == (True, False)
    assert db.score(other, "Audio ASR") == (False, False)
    assert db.score("I would use Whisper.", "Audio ASR") == (False, True)


def test_gorilla_real_grammar_matches_upstream_behaviour():
    import tree_sitter_python  # noqa: F401  (pinned in the evals extra; the suite requires it)
    rows = [{"api_call": "AutoModel.from_pretrained('princeton-nlp/unsup-simcse-roberta-base')",
             "domain": "NLP Sentence Similarity"},
            {"api_call": "pipeline('translation_en_to_fr', model='t5-base')", "domain": "NLP Translation"}]
    db = G.HubDatabase("huggingface", rows)
    good = ("<<<api_call>>>: `AutoModel.from_pretrained('princeton-nlp/unsup-simcse-roberta-base')`, "
            "<<<api_provider>>>: Hugging Face Transformers")
    assert db.score(good, "NLP Sentence Similarity") == (True, False)
    assert db.score(good.replace("simcse", "other"), "NLP Sentence Similarity") == (False, True)


def test_gorilla_hard_errors_without_the_ast_scorer(monkeypatch):
    import importlib.util
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "tree_sitter_python" else real(name, *a))
    with pytest.raises(RuntimeError, match="INFRA REQUIRED"):
        G.check_prereqs()


def test_gorilla_reports_per_hub_and_flags_the_mean(monkeypatch):
    monkeypatch.setattr(G, "check_prereqs", lambda: None)
    monkeypatch.setattr(G, "_load_gorilla_apibench_samples", lambda limit=None: [])
    traces = [{"gold_answer": json.dumps({"hub": h, "question_id": 1, "domain": "d"}),
               "response_text": r} for h, r in [("torchhub", "ok"), ("torchhub", "bad"),
                                                ("huggingface", "ok"), ("tensorflowhub", "hal")]]
    monkeypatch.setattr(G, "run_eval_suite", lambda **k: _fake_result(traces))
    verdict = {"ok": (True, False), "bad": (False, False), "hal": (False, True)}
    monkeypatch.setattr(G, "_score", lambda text, gold: verdict[text])
    r = G.run_gorilla_apibench("m", "http://x", 1)
    assert r["per_hub"]["torchhub"] == {"n": 2, "accuracy": 50.0, "hallucination": 0.0, "error": 50.0}
    assert r["per_hub"]["tensorflowhub"]["hallucination"] == 100.0
    assert r["accuracy"] == round((50.0 + 100.0 + 0.0) / 3, 2)
    assert r["leaderboard_comparable"] is False
    assert "per_hub" in r["leaderboard_comparable_reason"]


# --------------------------------------------------------------------------- #
# nestful
# --------------------------------------------------------------------------- #
_NEST_TOOLS = [{"name": "divide", "description": "divides two numbers",
                "parameters": {"arg_0": {"description": "The first number", "type": "int or float"}},
                "output_parameters": {"result": {"description": "q", "type": "float"}}}]


def test_nestful_prompt_is_byte_identical_to_deepseek_prompt_input():
    """eval.py json-encodes the tools and deepseek_prompt_input encodes them again; the ICL block
    and trailing newline come from get_icl_str + deepseek_prompt_input."""
    assert _sha(N._icl_str()) == "cfcfba97aa57d4c76744cab83843e5896ae77c12b4ca2de38762b749b3ecfe95"
    assert _sha(N._system_prompt(_NEST_TOOLS)) == \
        "27f31249a056d83a5016c5ea83a13553faba61468ac8c557f043c935b33e0f65"
    assert json.dumps(json.dumps(_NEST_TOOLS)) in N._system_prompt(_NEST_TOOLS)


def test_nestful_parses_the_whole_reply_as_json_like_parse_deepseek_output():
    gold = [{"name": "f", "label": "$var_1", "arguments": {"a": 1}}]
    good = json.dumps([{"name": "f", "label": "$var_1", "arguments": {"a": 1}}])
    pc, gc, pd, err = N._ibm_parse(good, gold)
    assert not err and pc == gc == ['{"name": "f", "arguments": {"a": 1}}']
    pc, _, pd, err = N._ibm_parse("```json\n" + good + "\n```", gold)
    assert err and pc == [] and pd == []
    # a bare number: upstream's scorer would crash; here it is a parse failure with no calls
    pc, _, pd, err = N._ibm_parse("5", gold)
    assert err and pc == [] and pd == 5


def test_nestful_no_longer_rewrites_labels():
    assert not hasattr(N, "_ibm_canon_label")
    # A dangling reference to an unknown label is compared verbatim.
    gold = [{"name": "f", "label": "$var_1", "arguments": {"x": "$var_9.result$"}}]
    pred = [{"name": "f", "label": "$var_1", "arguments": {"x": "$var9.result$"}}]
    pc, gc, _, _ = N._ibm_parse(json.dumps(pred), gold)
    assert N._ibm_score_example(pc, gc)["accuracy_combined"] == 0


def _func_dir(tmp_path):
    d = tmp_path / "data_v2" / "executable_functions"
    d.mkdir(parents=True)
    (d / "basic_functions.py").write_text(
        "def add(a, b):\n    return a + b\n\ndef pow_(a, b):\n    return a ** b\n\ndef truth(a):\n    return a > 0\n")
    (d / "func_file_map.json").write_text("{}")
    return str(d)


def test_nestful_win_rate_matches_calculate_win_score(tmp_path):
    fd = _func_dir(tmp_path)
    tools = [{"name": n, "output_parameters": {"result": {}}} for n in ("add", "pow_", "truth")]
    seq = [{"name": "add", "label": "$var_1", "arguments": {"a": 1, "b": 2}},
           {"name": "add", "label": "$var_2", "arguments": {"a": "$var_1.result$", "b": 0.5}}]
    assert N._win_score(seq, json.dumps(3.5), tools, fd) is True
    # gold types survive (the HF parquet stored str(gold) and lost them)
    assert N._win_score([{"name": "truth", "label": "$v", "arguments": {"a": 1}}],
                        json.dumps(True), tools, fd) is True
    assert N._win_score([{"name": "add", "label": "$v", "arguments": {"a": 50, "b": 1}}],
                        json.dumps("51"), tools, fd) is False
    # upstream builds str(arg) for every argument; a >4300-digit int makes that raise
    big = [{"name": "pow_", "label": "$var_1", "arguments": {"a": 10, "b": 5000}},
           {"name": "add", "label": "$var_2", "arguments": {"a": "$var_1.result$", "b": 0}}]
    assert N._win_score(big, json.dumps(1), tools, fd) is False


def test_nestful_requires_the_pinned_checkout(tmp_path, monkeypatch):
    fd = _func_dir(tmp_path)
    monkeypatch.setenv(N._FUNC_DIR_ENV, fd)
    with pytest.raises(RuntimeError, match=N.NESTFUL_COMMIT):
        N._func_dir()


def test_nestful_comparability_reasons():
    r = _fake_result([], temperature=1.0, thinking=True)
    reasons = N._noncanonical_reasons(r, limit=20)
    assert any("eval-limit" in x for x in reasons) and any("temperature" in x for x in reasons)
    assert any("thinking" in x for x in reasons)
    assert N._noncanonical_reasons(_fake_result([]), limit=None) == []


# --------------------------------------------------------------------------- #
# seal_tools
# --------------------------------------------------------------------------- #
_SEAL_GOLD = "[{'api': 'getWeather', 'parameters': {'city': 'Paris', 'days': 5}, 'responses': ['API_call_0']}]"


def test_seal_tools_compares_str_values_exactly():
    gold = ST._gold_calls(_SEAL_GOLD)
    lower = ST._transform_output('[{"api": "getWeather", "parameters": {"city": "paris", "days": "5"}, '
                                 '"responses": ["API_call_0"]}]')
    c = ST._sample_counts(gold, lower)
    assert (c["correct_param"], c["predict_param"], c["gold_param"]) == (1, 2, 2)   # '5' == str(5)


def test_seal_tools_reads_only_the_api_list():
    assert ST._transform_output("I will call getWeather(city) now.") == -1
    assert ST._transform_output('[{"name": "getWeather", "arguments": {"city": "Paris"}}]') == -1
    assert ST._transform_output('[{"api": "getWeather", "parameters": {"city": "Paris"}}]') == -1
    got = ST._transform_output("Sure:\n[{'api': 'getWeather', 'parameters': {'city': 'Paris'}, "
                               "'responses': ['API_call_0']}] done")
    assert got == [{"api": "getWeather", "parameters": {"city": "Paris"}, "responses": ["API_call_0"]}]


def test_seal_tools_pools_counts_like_calculate_score_toolLearning():
    gold = ST._gold_calls(_SEAL_GOLD)
    exact = ST._sample_counts(gold, ST._transform_output(json.dumps(gold)))
    missing = ST._sample_counts(gold, -1)
    s = ST._scores([exact, missing])
    assert s["format_accuracy"] == 0.5
    assert s["parameter_precision"] == 1.0 and s["parameter_recall"] == 0.5
    assert s["tool_f1"] == pytest.approx(2 * 1.0 * 0.5 / 1.5)


def test_seal_tools_loads_the_pinned_revision(monkeypatch):
    seen = {}

    def fake_load(repo, split=None, revision=None):
        seen.update(repo=repo, split=split, revision=revision)
        return [{"id": "x", "domain": "in-domain",
                 "conversations": [{"from": "human", "value": "Q \n"}, {"from": "gpt", "value": _SEAL_GOLD}]}]
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    samples = ST._load_seal_tools_samples()
    assert seen == {"repo": ST._REPO, "split": "test", "revision": ST._REVISION}
    assert samples[0][0] == [{"role": "user", "content": "Q \n"}]   # sent verbatim


# --------------------------------------------------------------------------- #
# nexusraven_api_eval (formerly nexus_function_calling)
# --------------------------------------------------------------------------- #
_NX_ROWS = [{"name": "search_cves", "description": "Search CVEs.\n\n    Args:\n    keyword (str): term",
             "args_dicts": [{"name": "limit", "type": "int", "required": False, "default": "10", "description": ""},
                            {"name": "keyword", "type": "str", "required": True, "default": None, "description": ""},
                            {"name": "verbose", "type": "None", "required": False, "default": "None",
                             "description": ""}]}]


def _nx_task():
    task = NX._Task("t", NX.build_functions(_NX_ROWS))
    _, ref = task.functions["search_cves"](**{"keyword": "log4j", "limit": 10})
    task.samples[0] = (["search_cves"], "search_cves", ref)
    NX._TASKS["t"] = task
    return task


def test_nexus_prompt_is_langchain_structured_chat_verbatim():
    task = _nx_task()
    prompt = NX.agent_prompt([task.tools["search_cves"]], "Find log4j CVEs")
    assert _sha(prompt) in (
        "aa406fbca357c0c1428d01917d4d93c42805de147b127624f483e55638e33c90",
        "1d21af354d7d8b662a76e5a84b6197f1cc4a15d4eb054fd0b4ae9c868a014347",
    )
    assert prompt.startswith("System: Respond to the human")
    assert "args: {{'keyword': {{'title': 'Keyword', 'type': 'string'}}" in prompt


def test_nexus_binds_defaults_and_compares_full_dicts():
    _nx_task()
    blob = lambda inp: "Action:\n```\n" + json.dumps({"action": "search_cves", "action_input": inp}) + "\n```"
    assert NX.score("t", 0, blob({"keyword": "log4j"})) == (True, False)          # limit default 10
    assert NX.score("t", 0, blob({"keyword": "log4j", "verbose": False})) == (False, False)
    assert NX.score("t", 0, blob({"keyword": "log4j", "limit": "10"})) == (True, False)  # pydantic v1
    assert NX.score("t", 0, "search_cves(keyword='log4j', limit=10.0)") == (True, False)
    assert NX.score("t", 0, blob({"keyword": "x"}).replace("search_cves", "nope", 1)) == (False, False)
    assert NX.score("t", 0, "```\n{broken\n```") == (False, True)


def test_nexus_fix_prompt_matches_output_fixing_parser():
    p = NX.fix_prompt("oops")
    assert p.startswith("Instructions:\n--------------\n" + NX.FORMAT_INSTRUCTIONS)
    assert "OutputParserException('Could not parse LLM output: oops')" in p


def test_nexus_runs_one_repair_call_for_rejected_replies(monkeypatch):
    _nx_task()
    gold = json.dumps({"task": "t", "key": 0})
    monkeypatch.setattr(NX, "_load_nexusraven_api_eval_samples", lambda limit=None: [])
    calls = []

    def fake_run(**kw):
        calls.append(kw)
        if len(calls) == 1:
            assert kw["extra_payload"]["stop"] == ["Observation:"]
            return _fake_result([{"gold_answer": gold, "response_text": "```\n{bad\n```"},
                                 {"gold_answer": gold, "response_text": "search_cves(keyword='log4j')"}])
        assert "stop" not in (kw.get("extra_payload") or {})
        fixed = "```\n" + json.dumps({"action": "search_cves", "action_input": {"keyword": "log4j"}}) + "\n```"
        return _fake_result([{"sample_idx": 0, "gold_answer": gold, "response_text": fixed,
                              "is_correct": True}])
    monkeypatch.setattr(NX, "run_eval_suite", fake_run)
    r = NX.run_nexusraven_api_eval("m", "http://x", 1)
    assert len(calls) == 2 and r["output_fixing"]["repair_calls"] == 1
    assert r["per_task"]["t"] == {"n": 2, "accuracy": 100.0}
    assert r["leaderboard_comparable"] is False


# --------------------------------------------------------------------------- #
# acebench
# --------------------------------------------------------------------------- #
def test_acebench_scores_multi_turn_per_conversation_and_macro_averages():
    rows = [("normal_multi_turn_user_adjust", "normal_multi_turn_user_adjust_0_0", True, 0),
            ("normal_multi_turn_user_adjust", "normal_multi_turn_user_adjust_0_1", False, 0),
            ("normal_multi_turn_user_adjust", "normal_multi_turn_user_adjust_1_0", True, 0),
            ("normal_atom_bool", "normal_atom_bool_0", True, 0),
            ("normal_atom_bool", "normal_atom_bool_1", True, 0),
            ("normal_atom_bool", "normal_atom_bool_2", False, 0),
            ("special_irrelevant", "special_irrelevant_0", True, 0)]
    agg = AB.aggregate(rows)
    mt = agg["per_sub_category"]["normal_multi_turn_user_adjust"]
    assert mt["accuracy"] == 0.5 and mt["conversations"] == 2 and mt["process_accuracy"] == 0.75
    assert agg["per_sub_category"]["normal_atom_bool"]["accuracy"] == 0.667
    # unweighted over all 12 / 3 sub-categories; missing ones count 0 (generate_result_csv)
    assert agg["normal_summary"] == round((0.5 + 0.667) / 12, 3)
    assert agg["special_summary"] == round(1.0 / 3, 3)
    # generate_result_csv's Summary: an agent category that did not run counts 0
    assert agg["agent_summary"] == 0
    assert agg["summary"] == round(0.2676 * agg["special_summary"] + 0.578 * agg["normal_summary"], 3)


def test_acebench_special_rules_are_the_exact_substrings():
    gt = {"Api_x": ["a", "b"]}
    ok = '["Missing necessary parameters (a, b) for the api (Api_x)"]'
    assert AB.special_row_valid(ok, gt, "special_incomplete") is True
    assert AB.special_row_valid(ok.lower(), gt, "special_incomplete") is False
    assert AB.special_row_valid('["The request is missing a and b for Api_x"]', gt, "special_incomplete") is False
    assert AB.special_row_valid("sorry, due to the limitations of the function", "x", "special_irrelevant")
    assert not AB.special_row_valid("Due to the Limitations of the Function", "x", "special_irrelevant")
    assert AB.special_row_valid("There is incorrect value (7x) for the parameters (id)", {"id": ["7x"]},
                                "special_error_param")


def test_acebench_normal_extraction_differs_single_vs_multi_turn():
    fns = [{"name": "Api", "parameters": {"type": "dict", "properties": {"q": {"type": "string"}},
                                          "required": ["q"]}}]
    gt = {"Api": {"q": "hello world"}}
    prose = "Sure: [Api(q='hello world')] done"
    assert AB.normal_row_valid(prose, fns, gt, "", "normal_single_turn_single_function") is True
    assert AB.normal_row_valid(prose, fns, gt, "", "normal_multi_turn_user_adjust") is False
    assert AB.normal_row_valid("[Api(q='hello world')]", fns, gt, "", "normal_multi_turn_user_adjust") is True


def test_acebench_prompt_renders_functions_as_python_list():
    row = {"function": [{"name": "Api", "description": "d", "parameters": {}}], "time": "", "question": "q"}
    sp = AB._system_prompt("normal_atom_bool", row)
    assert "[{'name': 'Api', 'description': 'd', 'parameters': {}}]" in sp


def test_acebench_pinned_file_is_checked(monkeypatch, tmp_path):
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"id": "x"}\n'
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    with pytest.raises(RuntimeError, match="does not match the pinned file"):
        AB._pinned_file("data_normal_atom_bool.json", AB._PINNED_BLOBS["normal_atom_bool"][0])


def test_acebench_result_is_flagged_with_the_real_scope(monkeypatch):
    monkeypatch.setattr(AB, "_load_acebench_samples", lambda **k: [])
    monkeypatch.setattr(AB, "_load_agent_items", lambda language="en": [])
    traces = [{"gold_answer": json.dumps({"sub": "special_irrelevant", "id": "special_irrelevant_0"}),
               "response_text": "x", "is_correct": True}]
    monkeypatch.setattr(AB, "run_eval_suite", lambda **k: _fake_result(traces))
    r = AB.run_acebench("m", "http://x", 1)
    assert r["special_summary"] == round(round(1 / 3, 3) * 100, 2)
    # the headline is the leaderboard's Summary over all three categories (agent counted 0 here)
    assert r["accuracy"] == round(round(0.2676 * round(1 / 3, 3), 3) * 100, 2)
    assert "lower bound" not in (r["leaderboard_comparable_reason"] or "")
    assert "agent category" not in (r["leaderboard_comparable_reason"] or "")


# --------------------------------------------------------------------------- #
# api_bank / t_eval / agent_dojo pins
# --------------------------------------------------------------------------- #
def test_api_bank_checkout_digest_ignores_bytecode_and_catches_edits(tmp_path):
    d = tmp_path / "api-bank"
    (d / "apis" / "__pycache__").mkdir(parents=True)
    (d / "lv1-lv2-samples").mkdir()
    (d / "evaluator.py").write_text("x = 1\n")
    (d / "apis" / "a.py").write_text("y = 2\n")
    base = API.checkout_digest(str(d))
    (d / "apis" / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\0")
    assert API.checkout_digest(str(d)) == base
    (d / "apis" / "a.py").write_text("y = 3\n")
    assert API.checkout_digest(str(d)) != base


def test_api_bank_rejects_an_unpinned_checkout(tmp_path, monkeypatch):
    d = tmp_path / "api-bank"
    (d / "apis").mkdir(parents=True)
    (d / "tool_manager.py").write_text("")
    monkeypatch.setenv("GBENCH_APIBANK_DIR", str(d))
    with pytest.raises(RuntimeError, match=API.APIBANK_COMMIT):
        API._checkout_dir()


@pytest.mark.parametrize("mod,call", [(API, lambda: API._hf("test-data/level-3.json")),
                                      (TE, lambda: TE._load_file_samples("instruct", "instruct_v2"))])
def test_hf_downloads_are_revision_pinned(monkeypatch, tmp_path, mod, call):
    seen = {}
    f = tmp_path / "d.json"
    f.write_text(json.dumps([{"origin_prompt": [{"role": "user", "content": "q"}]}]))

    def fake(repo, filename=None, repo_type=None, revision=None):
        seen["revision"] = revision
        return str(f)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    call()
    assert seen["revision"] == mod._HF_REVISION


def _dojo_suite():
    """One AgentDojo suite: user task u, injection task i."""
    task = type("T", (), {})
    user, inj = task(), task()
    user.ID, inj.ID = "u", "i"
    return type("S", (), {"user_tasks": {"u": user}, "injection_tasks": {"i": inj},
                          "get_user_task_by_id": lambda self, t: user,
                          "get_injection_task_by_id": lambda self, t: inj})()


def test_agent_dojo_flags_a_different_package_or_override(monkeypatch):
    fake = {"version": "0.1.99",
            "openai": type("O", (), {"OpenAI": staticmethod(lambda **k: object())}),
            "OpenAILLM": lambda *a, **k: object(),
            "AgentPipeline": lambda els: type("P", (), {"name": None})(),
            "SystemMessage": lambda *a, **k: object(), "InitQuery": lambda *a, **k: object(),
            "ToolsExecutionLoop": lambda *a, **k: object(), "ToolsExecutor": lambda *a, **k: object(),
            "load_system_message": lambda *a, **k: "s",
            "get_suites": lambda v: {"s": _dojo_suite()},
            "load_attack": lambda *a, **k: object(),
            "OutputLogger": lambda *a, **k: type("C", (), {"__enter__": lambda s: s,
                                                           "__exit__": lambda s, *a: False})(),
            "run_without": lambda *a, **k: (True, True),
            "run_with": lambda *a, **k: ({("u", "i"): True}, {("u", "i"): False})}
    monkeypatch.setattr(AD, "_load_agentdojo", lambda: fake)
    r = AD.run_agent_dojo("m", "http://x", limit=None, concurrency=1)
    assert r["agentdojo_version"] == "0.1.99" and r["leaderboard_comparable"] is False
    assert AD.AGENTDOJO_VERSION in r["leaderboard_comparable_reason"]
    fake["version"] = AD.AGENTDOJO_VERSION
    monkeypatch.setattr(AD, "_ATTACK", "tool_knowledge")
    r = AD.run_agent_dojo("m", "http://x", limit=None, concurrency=1)
    assert r["leaderboard_comparable"] is False and "overridden" in r["leaderboard_comparable_reason"]


# --------------------------------------------------------------------------- #
# The report contract (real evals.py / cli.py code) for every suite
# --------------------------------------------------------------------------- #
import threading  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest import mock  # noqa: E402

from gbench.runners.eval_suites import base  # noqa: E402


def _through_reports(res, name, tmp_path, monkeypatch):
    """Send a suite result through the real evals.py result block and the cli summaries."""
    from gbench import cli
    from gbench.runners import evals as E
    monkeypatch.setattr(E, "_resolve_remote_model_id", lambda url, fallback: "m")
    monkeypatch.setattr(E, "_server_max_model_len", lambda url: None)
    monkeypatch.setattr(E.EvalsBenchmarkRunner, "_run_single_eval", lambda self, eval_name, **k: dict(res))
    cfg = SimpleNamespace(evals=[name], eval_custom_jsonl=None, eval_plugins_dir=None, batch_sizes=[1],
                          remote_endpoint="http://127.0.0.1:1/v1", tokenizer="t",
                          eval_max_output_tokens=1024, skip_existing=False,
                          log_manager=SimpleNamespace(results_dir=tmp_path, base_dir=tmp_path,
                                                      save_result=lambda p, r: None))
    model = SimpleNamespace(name="m", short_name="m", hf_model_id="org/m")
    out = E.EvalsBenchmarkRunner(cfg).run(model, SimpleNamespace(value="remote-endpoint"))
    assert [r["eval_name"] for r in out] == [name] and out[0].get("status") != "error"
    cli.print_results_summary(out, SimpleNamespace(results_dir=tmp_path))
    csv = cli._save_eval_summary_csv(tmp_path, out, [], [("5. Tool Use", [name])], {})
    assert csv is not None and name.upper() in csv.read_text()
    return out[0]


def _assert_category_shape(res, multi=True):
    ca = res.get("category_accuracy") or {}
    if multi:
        assert 1 < len(ca) <= 25, "the per-category report block must actually run"
    for cat, st in ca.items():
        assert isinstance(st, dict), cat
        assert isinstance(st["correct"], int) and isinstance(st["total"], int), cat
        assert isinstance(st["accuracy"], (int, float)), cat


def _endpoint(answer):
    """A fake server: `answer(messages)` -> reply text."""
    async def _reply(**kw):
        return base.Reply(answer(kw["messages"]), None, "stop", completion_tokens=5)
    return _reply


def test_report_contract_catches_the_float_shape(monkeypatch, tmp_path):
    """The contract check itself: the pre-fix {task: float} shape aborts the evals.py loop."""
    bad = {"accuracy": 50.0, "status": "success", "total_questions": 2, "correct_answers": 1,
           "category_accuracy": {"a": 50.0, "b": 50.0}}
    with pytest.raises(AttributeError):
        _through_reports(bad, "nexusraven_api_eval", tmp_path, monkeypatch)


def test_gorilla_result_through_reports(monkeypatch, tmp_path):
    monkeypatch.setattr(G, "check_prereqs", lambda: None)
    hubs = (("torchhub", "torchhub"), ("huggingface", "huggingface"), ("tensorflowhub", "tensorhub"))
    samples = [(G.encode_question("q", api), json.dumps({"hub": hub, "question_id": 1, "domain": "d"}),
                {"category": hub}) for hub, api in hubs]
    monkeypatch.setattr(G, "_load_gorilla_apibench_samples", lambda limit=None: samples)
    seen = []

    def fake_score(text, gold):
        seen.append(text)
        return text == "ok", text == "hal"
    monkeypatch.setattr(G, "_score", fake_score)
    replies = {"torchhub.": "ok", "huggingface.": "hal", "tensorhub.": "<thought>hal</thought>ok"}

    def answer(msgs):
        return next(v for k, v in replies.items() if f"call API in {k}" in msgs[-1]["content"])
    with mock.patch.object(base, "_send_single_request", _endpoint(answer)):
        r = G.run_gorilla_apibench("m", "http://x", 1)
    # the per-hub tally scores the reasoning-stripped text, like the per-sample verdict
    assert "<thought>hal</thought>ok" not in seen
    assert r["per_hub"]["tensorflowhub"]["accuracy"] == 100.0
    assert r["category_accuracy"]["tensorflowhub"] == {"correct": 1, "total": 1, "accuracy": 100.0}
    _assert_category_shape(r)
    _through_reports(r, "gorilla_apibench", tmp_path, monkeypatch)


def test_seal_tools_result_through_reports(monkeypatch, tmp_path):
    gold = "[{'api': 'getWeather', 'parameters': {'city': 'Paris'}, 'responses': ['API_call_0']}]"
    samples = [([{"role": "user", "content": f"{dom}"}], gold, {"category": dom})
               for dom in ("in-domain", "out-domain")]
    monkeypatch.setattr(ST, "_load_seal_tools_samples", lambda limit=None: samples)
    call = '[{"api": "getWeather", "parameters": {"city": "Paris"}, "responses": ["API_call_0"]}]'
    answer = (lambda msgs: call if msgs[-1]["content"] == "in-domain"
              else "<thought>[{'api': 'x', 'parameters': {}, 'responses': []}]</thought>" + call)
    with mock.patch.object(base, "_send_single_request", _endpoint(answer)):
        r = ST.run_seal_tools("m", "http://x", 1)
    assert r["parameter_f1"] == 100.0 and r["exact_match_rate"] == 100.0   # tags stripped in both
    _assert_category_shape(r)
    _through_reports(r, "seal_tools", tmp_path, monkeypatch)


def test_nexus_result_through_reports(monkeypatch, tmp_path):
    t = _nx_task()
    NX._TASKS["u"] = t
    samples = [([{"role": "user", "content": task}], json.dumps({"task": task, "key": 0}), {"category": task})
               for task in ("t", "u")]
    monkeypatch.setattr(NX, "_load_nexusraven_api_eval_samples", lambda limit=None: samples)
    blob = "```\n" + json.dumps({"action": "search_cves", "action_input": {"keyword": "log4j"}}) + "\n```"

    def answer(msgs):
        content = msgs[-1]["content"]
        if content.startswith("Instructions:"):
            return blob                                           # the repair call
        return "<thought>```\n{no\n```</thought>" + blob if content == "t" else "```\n{bad\n```"
    with mock.patch.object(base, "_send_single_request", _endpoint(answer)):
        r = NX.run_nexusraven_api_eval("m", "http://x", 1)
    # "t": the fenced block inside the reasoning tags is not parsed; "u": repaired
    assert r["output_fixing"]["repair_calls"] == 1
    assert r["category_accuracy"] == {"t": {"correct": 1, "total": 1, "accuracy": 100.0},
                                      "u": {"correct": 1, "total": 1, "accuracy": 100.0}}
    assert r["per_task"]["t"] == {"n": 1, "accuracy": 100.0}
    _assert_category_shape(r)
    _through_reports(r, "nexusraven_api_eval", tmp_path, monkeypatch)


_ACE_FNS = [{"name": "Api", "parameters": {"type": "dict", "properties": {"q": {"type": "string"}},
                                           "required": ["q"]}}]


def _ace_sample(sub, row_id, reply_key, gt):
    gold = json.dumps({"id": row_id, "sub": sub, "ground_truth": gt, "functions": _ACE_FNS,
                       "question": "q"})
    return ([{"role": "user", "content": reply_key}], gold, {"category": sub})


def test_acebench_result_through_reports(monkeypatch, tmp_path):
    mt = "normal_multi_turn_user_adjust"
    good, bad = {"Api": {"q": "hello"}}, {"Api": {"q": "hello"}}
    samples = [_ace_sample(mt, f"{mt}_0_0", "good", good), _ace_sample(mt, f"{mt}_0_1", "bad", bad),
               _ace_sample(mt, f"{mt}_1_0", "good", good),
               _ace_sample("normal_atom_bool", "normal_atom_bool_0", "good", good),
               _ace_sample("normal_atom_bool", "normal_atom_bool_1", "bad", bad),
               _ace_sample("special_irrelevant", "special_irrelevant_0", "irr", "x")]
    monkeypatch.setattr(AB, "_load_acebench_samples",
                        lambda language="en", **k: samples if language == "en" else [])
    monkeypatch.setattr(AB, "_load_agent_items", lambda language="en": [])
    replies = {"good": "[Api(q='hello')]", "bad": "[Api(q='bye')]",
               "irr": "Due to the limitations of the function, I cannot solve this problem."}
    with mock.patch.object(base, "_send_single_request", _endpoint(lambda m: replies[m[-1]["content"]])):
        r = AB.run_acebench("m", "http://x", 1)
    ca = r["category_accuracy"]
    assert ca[mt] == {"correct": 1, "total": 2, "accuracy": 50.0}              # conversations
    assert ca["normal_atom_bool"] == {"correct": 1, "total": 2, "accuracy": 50.0}
    assert ca["special_irrelevant"] == {"correct": 1, "total": 1, "accuracy": 100.0}
    assert r["per_sub_category"][mt]["rows"] == 3 and r["per_sub_category"][mt]["process_accuracy"] == 75.0
    _assert_category_shape(r)
    _through_reports(r, "acebench", tmp_path, monkeypatch)


class _InProcWorker:
    def __init__(self, func_dir, *a, **k):
        self.func_dir, self.kills = func_dir, 0

    def score(self, pred, gold, tools):
        return N._win_score(pred, gold, tools, self.func_dir)

    def close(self):
        pass


def test_nestful_result_through_reports(monkeypatch, tmp_path):
    fd = _func_dir(tmp_path)
    data = tmp_path / "data_v2" / "nestful_data.jsonl"
    tools = [{"name": "add", "description": "", "parameters": {}, "output_parameters": {"result": {}}}]
    out = [{"name": "add", "label": "$var_1", "arguments": {"a": 1, "b": 2}}]
    data.write_text(json.dumps({"sample_id": "s1", "input": "1+2?", "tools": tools, "output": out,
                                "gold_answer": 3}) + "\n")
    monkeypatch.setattr(N, "_func_dir", lambda: fd)
    monkeypatch.setattr(N, "_data_file", lambda d: str(data))
    monkeypatch.setattr(N, "_WinRateWorker", _InProcWorker)
    reply = "<thought>[]</thought>" + json.dumps(out)
    with mock.patch.object(base, "_send_single_request", _endpoint(lambda m: reply)):
        r = N.run_nestful("m", "http://x", 1)
    # the corpus metrics score the reasoning-stripped text, like the per-sample verdict
    assert r["full_match_accuracy"] == 100.0 and r["parsing_errors"] == 0 and r["win_rate"] == 100.0
    assert r["win_rate_killed_calls"] == 0
    _assert_category_shape(r, multi=False)
    _through_reports(r, "nestful", tmp_path, monkeypatch)


def test_api_bank_result_through_reports(monkeypatch, tmp_path):
    monkeypatch.setattr(API, "_checkout_dir", lambda: "unused")
    samples = []
    for level in (1, 2):
        for track in ("api", "response"):
            for i in range(2):
                gold = json.dumps({"level": level, "track": track, "sample_id": i})
                samples.append(([{"role": "user", "content": f"{level}{track}{i}"}], gold,
                                {"category": f"api_bank::L{level}_{track}"}))
    monkeypatch.setattr(API, "_load_l12_samples", lambda d: samples)
    monkeypatch.setattr(API, "_load_l3_samples", lambda: [])

    def fake_score(d, traces):
        for t in traces:
            g = json.loads(t["gold_answer"])
            t["api_bank_level"], t["api_bank_track"] = g["level"], g["track"]
            ok = g["sample_id"] == 0
            t["api_bank_score"] = 1.0 if ok else 0.0
            t["is_correct"] = ok
        return set()
    monkeypatch.setattr(API, "_score_traces", fake_score)
    with mock.patch.object(base, "_send_single_request", _endpoint(lambda m: "[Api()]")):
        r = API.run_api_bank("m", "http://x", 1)
    # the scored call-track verdicts, not run_eval_suite's placeholder zeros
    assert r["category_accuracy"] == {"api_bank::L1_api": {"correct": 1, "total": 2, "accuracy": 50.0},
                                      "api_bank::L2_api": {"correct": 1, "total": 2, "accuracy": 50.0}}
    assert r["dimension_scores"]["L1_call"] == 50.0
    _assert_category_shape(r)
    _through_reports(r, "api_bank", tmp_path, monkeypatch)


def test_t_eval_result_through_reports(monkeypatch, tmp_path):
    monkeypatch.setattr(TE, "_load_file_samples", lambda dim, stem, limit=None: [
        ([{"role": "user", "content": dim}], json.dumps({"dimension": dim, "sample": {}}),
         {"category": f"t_eval::{dim}"})])
    monkeypatch.setattr(TE, "_score_file", lambda dim, traces, language="en": {})
    with mock.patch.object(base, "_send_single_request", _endpoint(lambda m: "x")):
        r = TE.run_t_eval("m", "http://x", 1)
    _assert_category_shape(r, multi=False)
    _through_reports(r, "t_eval", tmp_path, monkeypatch)


def _fake_dojo(version):
    return {"version": version,
            "openai": type("O", (), {"OpenAI": staticmethod(lambda **k: object())}),
            "OpenAILLM": lambda *a, **k: object(),
            "AgentPipeline": lambda els: type("P", (), {"name": None})(),
            "SystemMessage": lambda *a, **k: object(), "InitQuery": lambda *a, **k: object(),
            "ToolsExecutionLoop": lambda *a, **k: object(), "ToolsExecutor": lambda *a, **k: object(),
            "load_system_message": lambda *a, **k: "s",
            "get_suites": lambda v: {"s": _dojo_suite()},
            "load_attack": lambda *a, **k: object(),
            "OutputLogger": lambda *a, **k: type("C", (), {"__enter__": lambda s: s,
                                                           "__exit__": lambda s, *a: False})(),
            "run_without": lambda *a, **k: (True, True),
            "run_with": lambda *a, **k: ({("u", "i"): True}, {("u", "i"): False})}


def test_agent_dojo_model_label_override_is_not_comparable(monkeypatch, tmp_path):
    monkeypatch.setattr(AD, "_load_agentdojo", lambda: _fake_dojo(AD.AGENTDOJO_VERSION))
    r = AD.run_agent_dojo("m", "http://x", limit=None, concurrency=1)
    # a full run at the pinned label differs only in the temperature gbench always sends (0.0
    # without --thinking; the vllm_parsed provider sends none)
    assert r["leaderboard_comparable"] is False and r["attack_model_label"] == "vllm_parsed"
    assert r["leaderboard_comparable_reason"] == (
        "temperature 0.0 set by default (the vllm_parsed provider sends none)")
    assert r["temperature"] == 0.0
    _through_reports(r, "agent_dojo", tmp_path, monkeypatch)
    monkeypatch.setattr(AD, "_MODEL_LABEL", "gemini-2.0-flash-001")
    r = AD.run_agent_dojo("m", "http://x", limit=None, concurrency=1)
    assert r["leaderboard_comparable"] is False
    assert "model label" in r["leaderboard_comparable_reason"]
    assert r["attack_model_label"] == "gemini-2.0-flash-001"


# --------------------------------------------------------------------------- #
# Hard errors go through infra_required
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self.data


def _offline(*a, **k):
    raise OSError("network unreachable")


def test_gorilla_question_download_errors_are_infra_required(monkeypatch, tmp_path):
    import urllib.request
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))
    rel, blob = G.HUBS["torchhub"][1], G.HUBS["torchhub"][2]
    monkeypatch.setattr(urllib.request, "urlopen", _offline)
    with pytest.raises(RuntimeError, match=r"\[INFRA REQUIRED\] gorilla_apibench.*could not be downloaded"):
        G._questions_file(rel, blob)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp(b"{}\n"))
    with pytest.raises(RuntimeError, match=r"\[INFRA REQUIRED\] gorilla_apibench.*does not match"):
        G._questions_file(rel, blob)
    assert not list(tmp_path.rglob("*.jsonl"))


def test_gorilla_dataset_download_error_is_infra_required(monkeypatch):
    import huggingface_hub

    def boom(*a, **k):
        raise OSError("hub unreachable")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    with pytest.raises(RuntimeError, match=r"\[INFRA REQUIRED\] gorilla_apibench.*APIBench"):
        G._load_gorilla_apibench_samples()


def test_gorilla_rejects_other_tree_sitter_releases(monkeypatch):
    import importlib.util
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: object())
    versions = {"tree-sitter": "0.22.3", "tree-sitter-python": "0.21.0"}
    monkeypatch.setattr(G, "_dist_version", lambda d: versions[d])
    with pytest.raises(RuntimeError, match=r"INFRA REQUIRED.*tree-sitter 0\.22\.3"):
        G.check_prereqs()
    versions.update({"tree-sitter": "0.21.3", "tree-sitter-python": "0.23.6"})
    with pytest.raises(RuntimeError, match=r"INFRA REQUIRED.*tree-sitter-python 0\.23\.6"):
        G.check_prereqs()
    versions["tree-sitter-python"] = "0.21.0"
    G.check_prereqs()


def test_acebench_download_error_is_infra_required(monkeypatch, tmp_path):
    import urllib.request
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(urllib.request, "urlopen", _offline)
    with pytest.raises(RuntimeError, match=r"\[INFRA REQUIRED\] acebench.*could not be downloaded"):
        AB._pinned_file("data_normal_atom_bool.json", AB._PINNED_BLOBS["normal_atom_bool"][0])


@pytest.mark.parametrize("mod,call,name", [
    (ST, lambda: ST._load_seal_tools_samples(), "seal_tools"),
    (NX, lambda: NX._load_nexusraven_api_eval_samples(), "nexusraven_api_eval")])
def test_hf_dataset_errors_are_infra_required(monkeypatch, mod, call, name):
    import datasets

    def boom(*a, **k):
        raise OSError("hub unreachable")
    monkeypatch.setattr(datasets, "load_dataset", boom)
    with pytest.raises(RuntimeError, match=rf"\[INFRA REQUIRED\] {name}"):
        call()


@pytest.mark.parametrize("call,name", [
    (lambda: API._hf("test-data/level-3.json"), "api_bank"),
    (lambda: TE._load_file_samples("instruct", "instruct_v2"), "t_eval")])
def test_hf_file_errors_are_infra_required(monkeypatch, call, name):
    import huggingface_hub

    def boom(*a, **k):
        raise OSError("hub unreachable")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    with pytest.raises(RuntimeError, match=rf"\[INFRA REQUIRED\] {name}"):
        call()


def test_api_bank_error_gives_the_pinned_checkout_command(tmp_path, monkeypatch):
    monkeypatch.setenv("GBENCH_APIBANK_DIR", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError) as e:
        API._checkout_dir()
    assert f"git fetch --depth 1 origin {API.APIBANK_COMMIT} && git checkout FETCH_HEAD" in str(e.value)
    d = tmp_path / "api-bank"
    (d / "apis").mkdir(parents=True)
    (d / "tool_manager.py").write_text("")
    monkeypatch.setenv("GBENCH_APIBANK_DIR", str(d))
    with pytest.raises(RuntimeError, match="git checkout FETCH_HEAD"):
        API._checkout_dir()


# --------------------------------------------------------------------------- #
# acebench's literal evaluator is bounded
# --------------------------------------------------------------------------- #
def test_acebench_binop_refuses_huge_results_quickly():
    t0 = time.time()
    for expr in ("[F(x=10**10**10)]", "[F(x=(2**60000)**2)]", "[F(s='a'*10**7)]", "[F(s=10**7*'ab')]"):
        with pytest.raises(ValueError, match="too large"):
            AB.ast_parse(expr)
    assert AB.normal_row_valid("[Api(q=10**10**10)]", _ACE_FNS, {"Api": {"q": "x"}}, "", "normal_atom_bool") is False
    assert time.time() - t0 < 5
    # ordinary arithmetic still evaluates to the value upstream's eval gives
    assert AB.ast_parse("[F(x=2**10, y=-3*2, z='ab'*2, w=1.5**2)]") == [{"F": {"x": 1024, "y": -6, "z": "abab", "w": 2.25}}]


# --------------------------------------------------------------------------- #
# The nestful Win Rate cap holds off the main thread
# --------------------------------------------------------------------------- #
def test_nestful_win_rate_worker_caps_calls_on_any_thread(tmp_path):
    d = tmp_path / "data_v2" / "executable_functions"
    d.mkdir(parents=True)
    (d / "basic_functions.py").write_text(
        "import time\n"
        "def add(a, b):\n    return a + b\n\n"
        "def slow(a):\n    time.sleep(60)\n    return a\n\n"
        "def spin(a):\n    return sum(range(10 ** 13))\n\n"
        "def noisy(a):\n    print('reference output')\n    return a\n")
    (d / "func_file_map.json").write_text("{}")
    fd = str(d)
    tools = [{"name": n, "output_parameters": {"result": {}}} for n in ("add", "slow", "spin", "noisy")]
    add = [{"name": "add", "label": "$var_1", "arguments": {"a": 1, "b": 2}}]
    got = {}

    def run():
        w = N._WinRateWorker(fd, alarm_s=1, backstop_s=6)
        try:
            got["add"] = (w.score(add, json.dumps(3), tools), N._win_score(add, json.dumps(3), tools, fd))
            got["noisy"] = w.score([{"name": "noisy", "label": "$v", "arguments": {"a": 4}}],
                                   json.dumps(4), tools)
            t0 = time.time()
            got["slow"] = (w.score([{"name": "slow", "label": "$v", "arguments": {"a": 1}}],
                                   json.dumps(1), tools), time.time() - t0, w.kills)
            t0 = time.time()
            got["spin"] = (w.score([{"name": "spin", "label": "$v", "arguments": {"a": 1}}],
                                   json.dumps(1), tools), time.time() - t0, w.kills)
            got["after"] = w.score(add, json.dumps(3), tools)          # a fresh worker
        finally:
            w.close()
        got["closed"] = w.proc is None
    th = threading.Thread(target=run)
    th.start()
    th.join(timeout=120)
    assert not th.is_alive()
    assert got["add"] == (True, True)
    assert got["noisy"] is True                     # a print cannot break the protocol
    ok, secs, kills = got["slow"]
    assert ok is False and secs < 5 and kills == 0, got["slow"]   # the alarm fired in the child
    ok, secs, kills = got["spin"]
    assert ok is False and 5 <= secs < 30 and kills == 1, got["spin"]   # stuck in C: killed
    assert got["after"] is True and got["closed"] is True
