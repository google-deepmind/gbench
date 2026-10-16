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

"""Upstream protocols of the knowledge suites: mmlu (Hendrycks log-likelihood), gpqa and gpqa_diamond
(simple-evals 10 repeats), multilingual_mmlu (failure gate), mmlu_redux (HELM MMLU on the
"ok" rows), arc_agi (ARCTester retries and the Attempt model), lab_bench (all 8 categories,
LAB-Bench's own parse) and cruxeval (the published sampling). No network and no model server:
datasets are faked and endpoints are local stubs."""

import asyncio
import random
import threading
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import (arc_agi, cruxeval, gpqa, gpqa_diamond, lab_bench, mmlu,
                                        mmlu_redux, multilingual_mmlu)

REPO = Path(__file__).resolve().parent.parent


def _doc(suite):
    return (REPO / "docs" / "evals" / f"{suite}.md").read_text(encoding="utf-8")


def _fake_hf(rows_by_key):
    """load_dataset stand-in keyed by (path, config) or path; asserts a pinned revision."""
    def _load(path, *args, **kwargs):
        assert kwargs.get("revision"), f"{path} loaded without a pinned revision"
        name = args[0] if args else kwargs.get("name")
        key = (path, name, kwargs.get("split"))
        for k in (key, (path, name), path):
            if k in rows_by_key:
                return rows_by_key[k]
        raise AssertionError(f"unexpected dataset request {key}")
    return mock.patch("datasets.load_dataset", side_effect=_load)


def _serve(routes):
    """aiohttp app on an ephemeral port in a thread; returns (base_url, stop)."""
    from aiohttp import web
    loop = asyncio.new_event_loop()
    app = web.Application()
    for path, handler in routes.items():
        app.router.add_post(path, handler)
    runner = web.AppRunner(app)
    ready, port = threading.Event(), {}

    def run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, "127.0.0.1", 0)
        loop.run_until_complete(site.start())
        port["p"] = site._server.sockets[0].getsockname()[1]
        ready.set()
        loop.run_forever()

    threading.Thread(target=run, daemon=True).start()
    ready.wait(10)

    def stop():
        asyncio.run_coroutine_threadsafe(runner.cleanup(), loop).result(10)
        loop.call_soon_threadsafe(loop.stop)
    return f"http://127.0.0.1:{port['p']}/v1", stop


class _CharTok:
    """Character-level stand-in tokenizer: ' X' is two tokens (' ', 'X'), no BOS."""
    bos_token_id = None

    def encode(self, s):
        return [ord(c) for c in s]


# =========================================================================== #
# mmlu: hendrycks/test evaluate.py, 5-shot, answer-letter log-likelihood
# =========================================================================== #
_MMLU_DEV = [{"question": f"Dev {i}?", "choices": ["w", "x", "y", "z"], "answer": i % 4,
              "subject": "anatomy"} for i in range(5)]
_MMLU_TEST = [{"question": f"Test {i}?", "choices": ["0", "1", "2", "3"], "answer": i % 4,
               "subject": "anatomy"} for i in range(4)]


def _mmlu_hf():
    return _fake_hf({("cais/mmlu", "all", "test"): _MMLU_TEST,
                     ("cais/mmlu", "all", "dev"): _MMLU_DEV})


def _letter_server(prefer, seen):
    """/v1/completions: in the top list of the one generated token, the letter
    `prefer(prompt_text)` scores highest."""
    from aiohttp import web

    async def handler(req):
        body = await req.json()
        seen.append(body)
        best = prefer(body["prompt"])
        top = {f" {c}": (-0.2 if c == best else -3.0) for c in "ABCD"}
        top["\n"] = -4.0
        return web.json_response({"choices": [{"index": 0, "text": f" {best}", "logprobs": {
            "tokens": [f" {best}"], "token_logprobs": [-0.2], "top_logprobs": [top]}}]})
    return {"/v1/completions": handler}


def test_mmlu_scores_the_next_token_letter_loglikelihood_end_to_end():
    seen = []
    gold = {f"Test {i}?": "ABCD"[i % 4] for i in range(4)}
    # The model is right on Test 0 and Test 1 only (it answers "A" otherwise).
    url, stop = _serve(_letter_server(
        lambda p: gold[p.rsplit("\n\n", 1)[-1].split("\n")[0]]
        if p.rsplit("\n\n", 1)[-1].split("\n")[0] in ("Test 0?", "Test 1?") else "A", seen))
    try:
        with _mmlu_hf(), mock.patch.object(mmlu, "_load_tokenizer", return_value=_CharTok()), \
                mock.patch.object(mmlu, "get_run_knob", return_value=None):
            res = mmlu.run_mmlu("m", url, 2, enable_thinking=True)
    finally:
        stop()
    # evaluate.py's request, without echo: text prompt, one token, its top-100 log-probs.
    assert all("echo" not in b and b["logprobs"] == 100 and b["max_tokens"] == 1
               and b["temperature"] == 0 and isinstance(b["prompt"], str) for b in seen)
    prompt = seen[0]["prompt"]
    # The Hendrycks prompt: 5 dev exemplars, raw question, "Answer:".
    assert prompt.startswith("The following are multiple choice questions (with answers) about "
                             " anatomy.\n\nDev 0?\nA. w\nB. x\nC. y\nD. z\nAnswer: A\n\n")
    assert prompt.count("Answer:") == 6 and prompt.endswith("\nAnswer:")
    assert res["total_questions"] == 4 and res["correct_answers"] == 2 and res["accuracy"] == 50.0
    assert res["scoring_mode"] == "loglikelihood" and res["n_shot"] == 5
    assert res["leaderboard_comparable"] is True and "leaderboard_comparable_reason" not in res
    assert res["thinking"] is False and res["thinking_requested"] is True
    assert res["thinking_applied"] is False and "not applied" in res["thinking_not_applied_reason"]
    trace = res["sample_traces"][2]
    assert trace["predicted"] == "A" and trace["gold_answer"] == "C"


def test_mmlu_endpoint_without_top_logprobs_is_infra_required():
    from aiohttp import web

    async def handler(req):
        return web.json_response({"choices": [{"index": 0, "logprobs": {"token_logprobs": [-1.0]}}]})
    url, stop = _serve({"/v1/completions": handler})
    try:
        with _mmlu_hf(), mock.patch.object(mmlu, "_load_tokenizer", return_value=_CharTok()), \
                mock.patch.object(mmlu, "get_run_knob", return_value=None), \
                pytest.raises(RuntimeError, match="INFRA REQUIRED.*mmlu.*--max-logprobs 100"):
            mmlu.run_mmlu("m", url, 1)
    finally:
        stop()


def test_mmlu_crop_rule_drops_trailing_exemplars_to_fit_the_window():
    item = {"subject": "anatomy", "header": "H\n\n", "shots": ["S0\n\n", "S1\n\n", "S2\n\n"],
            "question": "Q?\nAnswer:", "gold": 0}
    enc = mmlu._Encoder(_CharTok())
    full = mmlu._prompt(item)
    assert mmlu._fit_to_window(item, enc, None) == (full, 3)
    # Room for the header, the first exemplar, the question and the one generated token.
    window = len(mmlu._prompt(item, 1)) + 1
    prompt, k = mmlu._fit_to_window(item, enc, window)
    assert k == 1 and prompt == "H\n\nS0\n\nQ?\nAnswer:"          # the LAST shots go first


def test_mmlu_flags_failed_requests_and_rejects_more_than_five_shots():
    def some_fail(samples):
        return [{"loglikelihoods": None, "pred": None, "is_correct": False, "status": "FAILED",
                 "error": "HTTP 500"}] + [
            {"loglikelihoods": [0.0, -1.0, -1.0, -1.0], "pred": 0, "is_correct": s["gold"] == 0,
             "status": "OK", "error": None} for s in samples[1:]]

    async def fake(base_url, model_name, enc, samples, concurrency,
                   temperature=0.0):
        return some_fail(samples)
    with _mmlu_hf(), mock.patch.object(mmlu, "_load_tokenizer", return_value=_CharTok()), \
            mock.patch.object(mmlu, "_score_all", side_effect=fake), \
            mock.patch.object(mmlu, "get_run_knob", return_value=None):
        res = mmlu.run_mmlu("m", "http://x/v1", 1)
    assert res["leaderboard_comparable"] is False and res["failed_requests"] == 1
    # a harness error, left out of the score (never a wrong answer)
    assert "1 question(s) got no usable reply from the served endpoint" in \
        res["leaderboard_comparable_reason"]
    assert res["status"] == "completed_with_errors"
    with mock.patch.object(mmlu, "_load_tokenizer") as tok, pytest.raises(ValueError):
        mmlu.run_mmlu("m", "http://x/v1", 1, eval_n_shot=6)
    tok.assert_not_called()


def test_mmlu_shares_wmdps_lm_eval_encoder_with_tokenizer_defaults():
    # A tokenizer that adds no BOS by default (Gemma 4's) gets none: the reference sends text.
    class GemmaTokenizer(_CharTok):
        bos_token_id = 2
    enc = mmlu._Encoder(GemmaTokenizer())
    ctx, cont = enc.pair("Answer:", " B")
    assert ctx[0] != 2 and "".join(map(chr, cont)) == " B" and enc.bos_prepended is False


# =========================================================================== #
# gpqa / gpqa_diamond: simple-evals GPQAEval, n_repeats=10
# =========================================================================== #
def _simple_evals_permutations(n_examples, n_repeats):
    """GPQAEval.__init__ at 652c89d0, verbatim apart from the example payload."""
    examples = list(range(n_examples))
    rng = random.Random(0)
    examples = examples * n_repeats
    return [rng.sample(range(4), 4) for _ in examples]


def _gpqa_rows(n, sub=("Physics", "Chemistry")):
    return [{"Question": f"Q{i}?", "Correct Answer": f"right{i}", "Incorrect Answer 1": f"w1{i}",
             "Incorrect Answer 2": f"w2{i}", "Incorrect Answer 3": f"w3{i}",
             "Subdomain": sub[i % len(sub)]} for i in range(n)]


def test_gpqa_permutations_are_simple_evals_own_sequence():
    assert gpqa._permutations(198, 10) == _simple_evals_permutations(198, 10)
    assert gpqa.N_REPEATS == 10


def test_gpqa_copies_follow_examples_times_n_repeats_and_limit_keeps_full_run_orders():
    rows = _gpqa_rows(6)
    items = gpqa._gpqa_items(rows, 10, None, "gpqa")
    assert len(items) == 60
    assert [(it["question_idx"], it["repeat_idx"]) for it in items[:7]] == [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (0, 1)]
    msgs, gold, meta = items[7]["sample"]
    perm = items[7]["permutation"]
    choices = [rows[1][k] for k in ("Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2",
                                    "Incorrect Answer 3")]
    choices = [choices[i] for i in perm]
    assert gold == "ABCD"[choices.index("right1")]
    # The chat sampler's 2048 is declared once (gpqa.SIMPLE_EVALS_SAMPLING), not per sample.
    assert meta == {"category": "Chemistry"}
    assert gpqa.SIMPLE_EVALS_SAMPLING.max_tokens == gpqa.SIMPLE_EVALS_CHAT_MAX_TOKENS
    assert msgs[-1]["content"] == gpqa.QUERY_TEMPLATE_MULTICHOICE.format(
        A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question="Q1?")
    sub = gpqa._gpqa_items(rows, 10, 2, "gpqa")
    kept = {it["question_idx"] for it in sub}
    assert len(kept) == 2 and len(sub) == 20
    full = {(it["question_idx"], it["repeat_idx"]): it["permutation"] for it in items}
    assert all(full[(it["question_idx"], it["repeat_idx"])] == it["permutation"] for it in sub)


def _run_gpqa(module, run, result, rows=6, **kwargs):
    items = gpqa._gpqa_items(_gpqa_rows(rows), max(10, int(kwargs.get("attempt_count") or 1)),
                             kwargs.get("limit"), "gpqa")
    seen = {}

    def fake_run(**kw):
        seen.update(kw)
        traces = [{"sample_idx": i, "is_correct": i % 2 == 0} for i in range(len(kw["samples"]))]
        return {**result, "sample_traces": traces}
    with mock.patch.object(gpqa, "_load_gpqa_items", return_value=items), \
            mock.patch.object(gpqa, "run_eval_suite", side_effect=fake_run):
        return getattr(module, run)("m", "http://x/v1", 1, **kwargs), seen


@pytest.mark.parametrize("module,run", [(gpqa, "run_gpqa"), (gpqa_diamond, "run_gpqa_diamond")])
def test_gpqa_runs_ten_repeats_and_maps_every_copy(module, run):
    res, seen = _run_gpqa(module, run, {"temperature": 1.0})
    assert seen["attempt_count"] == 1 and seen["limit"] is None and len(seen["samples"]) == 60
    assert res["n_repeats"] == 10 and res["repeats"]["generations"] == 60
    assert res["repeats"]["questions"] == 6 and len(res["repeats"]["per_repeat_accuracy"]) == 10
    t = res["sample_traces"][13]
    assert (t["source_sample_idx"], t["attempt_idx"]) == (1, 2) and sorted(t["permutation"]) == [
        0, 1, 2, 3]
    assert res["repeats"]["score_std"] == 0.5
    assert res["leaderboard_comparable"] is True
    more, seen = _run_gpqa(module, run, {}, attempt_count=12)
    assert more["n_repeats"] == 12 and len(seen["samples"]) == 72
    assert "12 repeats per question (simple-evals averages 10)" in more[
        "leaderboard_comparable_reason"]


@pytest.mark.parametrize("module,run,total", [
    (gpqa, "run_gpqa", 448), (gpqa_diamond, "run_gpqa_diamond", 198),
    (multilingual_mmlu, "run_multilingual_mmlu", 14 * 14042)])
@pytest.mark.parametrize("counts", [{"failed_requests": 3},
                                    {"timed_out_requests": 2, "over_context_prompts": 1}])
def test_simple_evals_suites_flag_requests_that_failed_or_timed_out(module, run, total, counts):
    if module is multilingual_mmlu:
        with mock.patch.object(module, "_load_multilingual_mmlu_samples",
                               return_value=[("m", "A", {})]), \
                mock.patch.object(module, "run_eval_suite", return_value=dict(counts)):
            res = module.run_multilingual_mmlu("m", "http://x/v1", 1)
    else:
        res, _ = _run_gpqa(module, run, dict(counts))
    assert res["leaderboard_comparable"] is False
    # A request with no reply from the endpoint is a harness error, left out of the score.
    assert "3 request(s) got no reply from the served endpoint" in res[
        "leaderboard_comparable_reason"]
    assert "harness errors, left out of the score; simple-evals' samplers ask a failed request " \
           "again" in res["leaderboard_comparable_reason"]


def test_gpqa_traces_map_to_copies_under_sharding(monkeypatch):
    # --shard selects questions (each keeps all 10 repeats); base keeps exactly those copies.
    from gbench.runners.eval_suites.sampling import parse_shard, shard_select
    monkeypatch.setenv("GBENCH_SHARD", "2/3")
    items = gpqa._gpqa_items(_gpqa_rows(6), 10, None, "gpqa")
    assert {it["question_idx"] for it in items} == {1, 4} and len(items) == 20

    def fake_run(**kw):
        shown = shard_select(kw["samples"], parse_shard("2/3"))
        return {"sample_traces": [{"sample_idx": i, "is_correct": True}
                                  for i in range(len(shown))]}
    with mock.patch.object(gpqa, "_load_gpqa_items", return_value=items), \
            mock.patch.object(gpqa, "run_eval_suite", side_effect=fake_run):
        res = gpqa.run_gpqa("m", "http://x/v1", 1)
    t = res["sample_traces"][4]
    assert (t["source_sample_idx"], t["attempt_idx"]) == (items[4]["question_idx"],
                                                           items[4]["repeat_idx"])
    assert res["repeats"]["generations"] == 20 and res["repeats"]["questions"] == 2


# =========================================================================== #
# mmlu_redux: HELM MMLU (multiple_choice_joint, 5-shot, max_tokens=1) on error_type "ok"
# =========================================================================== #
# The MMLU dev questions of two subjects and the in-context order HELM's own v1.3.0 GPT-4o run
# used for them (display_requests.json). college_physics has a repeated label ("4").
_AA_DEV = [
    ("Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.", ["0", "1", "2", "3"], 1),
    ("Statement 1 | If aH is an element of a factor group, then |aH| divides |a|. Statement 2 | "
     "If H and K are subgroups of G then HK is a subgroup of G.",
     ["True, True", "False, False", "True, False", "False, True"], 1),
    ("Statement 1 | Every element of a group generates a cyclic subgroup of the group. "
     "Statement 2 | The symmetric group S_10 has 10 elements.",
     ["True, True", "False, False", "True, False", "False, True"], 2),
    ("Statement 1| Every function from a finite set onto itself must be one to one. Statement 2 "
     "| Every subgroup of an abelian group is abelian.",
     ["True, True", "False, False", "True, False", "False, True"], 0),
    ("Find the characteristic of the ring 2Z.", ["0", "3", "12", "30"], 0),
]
_AA_HELM_ORDER = [1, 0, 4, 3, 2]
_TELESCOPE = ("A refracting telescope consists of two converging lenses separated by 100 cm. "
              "The eye-piece lens has a focal length of 20 cm. The angular magnification of the "
              "telescope is")
_CP_DEV = [
    (_TELESCOPE, ["4", "5", "6", "20"], 0),
    ("For which of the following thermodynamic processes is the increase in the internal energy "
     "of an ideal gas equal to the heat added to the gas?",
     ["Constant temperature", "Constant volume", "Constant pressure", "Adiabatic"], 1),
    ("One end of a Nichrome wire", ["2.4 V", "3.3 V", "4.5 V", "5.7 V"], 0),
    (_TELESCOPE, ["4", "5", "6", "20"], 0),
    ("The muon decays", ["charge", "mass", "energy and momentum", "lepton number"], 3),
]
_CP_HELM_QUESTIONS = [_TELESCOPE, "One end of a Nichrome wire", "The muon decays",
                      "For which of the following thermodynamic processes is the increase in the "
                      "internal energy of an ideal gas equal to the heat added to the gas?",
                      _TELESCOPE]


def _train(dev):
    return [{"question": q, "choices": c, "answer": a, "label": c[a]} for q, c, a in dev]


def test_helm_sample_examples_reproduces_helms_own_order():
    picked = mmlu_redux.sample_examples(_train(_AA_DEV), seed=0)
    assert [p["question"] for p in picked] == [_AA_DEV[i][0] for i in _AA_HELM_ORDER]
    picked = mmlu_redux.sample_examples(_train(_CP_DEV), seed=0)
    assert [p["question"] for p in picked] == _CP_HELM_QUESTIONS


def test_helm_prompt_matches_helms_request_text():
    blocks = [mmlu_redux._example_block(q, c, a) for q, c, a in
              (_AA_DEV[i] for i in _AA_HELM_ORDER)]
    prompt = mmlu_redux.helm_prompt(
        "abstract_algebra", blocks,
        "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.",
        ["0", "4", "2", "6"])
    # The head and tail of HELM v1.3.0's GPT-4o request for this test question.
    assert prompt.startswith(
        "Answer with only a single letter.\n\nThe following are multiple choice questions (with "
        "answers) about abstract algebra.\n\nQuestion: Statement 1 | If aH is an element of a "
        "factor group, then |aH| divides |a|. Statement 2 | If H and K are subgroups of G then HK "
        "is a subgroup of G.\nA. True, True\nB. False, False\nC. True, False\nD. False, True\n"
        "Answer: B\n\nQuestion: Find all c in Z_3")
    assert prompt.endswith(
        "D. False, True\nAnswer: C\n\nQuestion: Find the degree for the given field extension "
        "Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\nA. 0\nB. 4\nC. 2\nD. 6\nAnswer:")
    assert mmlu_redux._csv_text("I.\r\nII.\rIII.") == "I.\nII.\nIII."


def test_mmlu_redux_request_is_helms_one_token_greedy_answer():
    from aiohttp import web
    seen = []

    async def handler(req):
        body = await req.json()
        seen.append(body)
        return web.json_response({"choices": [{"message": {"content": "B"},
                                               "finish_reason": "length"}],
                                  "usage": {"completion_tokens": 1}})
    url, stop = _serve({"/v1/chat/completions": handler})
    items = [{"subject": "anatomy", "choices": ["a", "b", "c", "d"], "answer": a, "prompt": "P"}
             for a in (1, 2)]
    try:
        with mock.patch.object(mmlu_redux, "_load_mmlu_redux_items", return_value=items):
            res = mmlu_redux.run_mmlu_redux("m", url, 2, max_output_tokens=65536)
            think = mmlu_redux.run_mmlu_redux("m", url, 2, enable_thinking=True,
                                              max_output_tokens=65536)
    finally:
        stop()
    assert len(seen) == 4
    for body in seen[:2]:
        assert body["max_tokens"] == 1 and body["temperature"] == 0.0 and "stop" not in body
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["messages"] == [{"role": "user", "content": "P"}]
    assert res["accuracy"] == 50.0 and res["subject_mean_accuracy"] == 50.0
    assert res["thinking_applied"] is False and res["thinking_requested"] is False
    assert res["leaderboard_comparable"] is True and res["status"] == "success"
    assert res["max_tokens_sent"] == [1]
    # The token-limit policy: HELM's one-token cap is not applied to reasoning; a thinking run
    # thinks and is sent the run's budget, at temperature 1.0 (not HELM's 0).
    for body in seen[2:]:
        assert body["max_tokens"] == 65536 and body["temperature"] == 1.0
        assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert think["thinking_applied"] is True and think["max_tokens_sent"] == [65536]
    assert think["official_sampling"]["max_tokens_applied"] is False
    assert think["leaderboard_comparable"] is False
    assert "--thinking run" in think["leaderboard_comparable_reason"]
    assert "temperature 1.0 (thinking; HELM decodes at 0)" in think["leaderboard_comparable_reason"]
    assert think["official_sampling"]["temperature_matches_official"] is False


def test_mmlu_redux_temperature_override_and_failures_are_flagged(monkeypatch):
    monkeypatch.setenv("GBENCH_MMLU_REDUX_TEMPERATURE", "0.7")
    item = {"subject": "anatomy", "choices": ["a", "b", "c", "d"], "answer": 0, "prompt": "P"}
    replies = [mock.Mock(text="A", finish_reason="length", error=None),
               mock.Mock(text=None, finish_reason=None, error="HTTP 500")]

    async def fake(base_url, model_name, prompts, concurrency, temperature):
        assert temperature == 0.7
        return replies
    with mock.patch.object(mmlu_redux, "_load_mmlu_redux_items", return_value=[item, item]), \
            mock.patch.object(mmlu_redux, "_generate_all", side_effect=fake):
        res = mmlu_redux.run_mmlu_redux("m", "http://x/v1", 1)
    reason = res["leaderboard_comparable_reason"]
    assert res["leaderboard_comparable"] is False and res["failed_requests"] == 1
    # a request with no reply is a harness error, left out of the score
    assert "temperature 0.7" in reason and "1 request(s) got no reply from the served " \
        "endpoint" in reason


def test_mmlu_redux_scores_only_ok_rows_and_counts_them():
    assert mmlu_redux.SCORED_ROWS == 5330
    doc = _doc("mmlu_redux")
    assert "5,330" in doc and "5,168" in doc and "14,042" in doc
    assert "Answer with only a single letter." in doc and "max_tokens=1" in doc


# =========================================================================== #
# arc_agi: Attempt model coercion and ARCTester --retry_attempts 2
# =========================================================================== #
def test_arc_agi_attempt_answer_is_pydantics_lax_list_of_int_lists():
    p = arc_agi._parse_grid
    assert p("[[1.0, 2]]") == [[1, 2]]
    assert p('[["1", "2"]]') == [[1, 2]]
    assert p("[[true, 0]]") == [[1, 0]]
    assert p("[[1.5]]") is None and p('[["a"]]') is None and p("[[null]]") is None
    assert p("\\boxed{[]}") == [] and p("no grid") is None and p(None) is None
    assert arc_agi._eval_arc_agi("[[1.0, 2]]", [[1, 2]]) is True
    assert arc_agi._RETRY_ATTEMPTS == 2


def _arc_sample(tid):
    return ([{"role": "user", "content": "p"}], [[1]],
            {"category": "grid_transformation", "task_id": tid, "pair_index": 0})


def _arc_trace(idx, src, text, status="OK"):
    # base marks a request that got no reply from the endpoint as a harness error
    no_reply = ({"harness_error": {"kind": "endpoint_failure", "detail": "x"},
                 "scoring_excluded": True} if status in ("FAILED", "TIMEOUT") else {})
    return {"sample_idx": idx, "source_sample_idx": src, "attempt_idx": idx % 2,
            "extra_payload": {"task_id": f"t{src}"}, "response_text": text,
            "is_correct": arc_agi._eval_arc_agi(text, [[1]]), "status": status, **no_reply}


def test_arc_agi_retries_each_failed_try_once_and_leaves_null_attempts():
    calls = []

    def fake_run(**kw):
        calls.append(kw)
        if len(calls) == 1:
            return {"attempts": {"attempts_per_sample": 2}, "sample_traces": [
                _arc_trace(0, 0, "no grid"), _arc_trace(1, 0, "still no grid"),
                _arc_trace(2, 1, None, "FAILED"), _arc_trace(3, 1, "[[1]]")]}
        assert [s[2]["task_id"] for s in kw["samples"]] == ["t0", "t0", "t1"]
        return {"sample_traces": [_arc_trace(0, 0, "[[1]]"), _arc_trace(1, 0, "nothing"),
                                  _arc_trace(2, 0, None, "TIMEOUT")]}
    with mock.patch.object(arc_agi, "_load_arc_agi_samples",
                           return_value=[_arc_sample("t0"), _arc_sample("t1")]), \
            mock.patch.object(arc_agi, "run_eval_suite", side_effect=fake_run):
        res = arc_agi.run_arc_agi("m", "http://x/v1")
    assert len(calls) == 2                                   # 2 tries per attempt, no more
    rep = res["arc_agi_report"]
    assert rep["retried_attempts"] == 3 and rep["recovered_by_retry"] == 1
    # the attempt with no reply in either try is a harness error, not a null attempt
    assert rep["unparsed_attempts"] == 1 and rep["tries_per_attempt"] == 2
    assert res["accuracy"] == 100.0                          # both pairs solved by one attempt
    traces = res["sample_traces"]
    assert traces[0]["tries"] == 2 and traces[0]["earlier_try"]["response_text"] == "no grid"
    assert (traces[2]["source_sample_idx"], traces[2]["attempt_idx"]) == (1, 0)
    # A null attempt from unparseable replies is upstream's own outcome; one with no reply in
    # either try measures the serving stack: a harness error, left out and flagged.
    assert res["leaderboard_comparable"] is False and res["failed_requests"] == 1
    assert "1 attempt(s) got no reply from the served endpoint on their last try" in res[
        "leaderboard_comparable_reason"]
    assert res["scoring_excluded"] == 1 and res["status"] == "completed_with_errors"
    assert "harness errors left out of the score: 1 attempt(s)" in res["status_reason"]


def test_arc_agi_retry_batch_is_not_resharded(monkeypatch):
    # The loader draws this shard's tasks; base must keep all of them, and GBENCH_SHARD is
    # never touched (another suite may read it concurrently).
    from gbench.runners.eval_suites.sampling import parse_shard, shard_select
    monkeypatch.setenv("GBENCH_SHARD", "1/2")
    seen_env, seen_samples = [], []

    def fake_run(**kw):
        import os
        seen_env.append(os.environ.get("GBENCH_SHARD"))
        seen_samples.append(shard_select(kw["samples"], parse_shard("1/2")))
        if len(seen_env) == 1:
            return {"attempts": {"attempts_per_sample": 2}, "sample_traces": [
                _arc_trace(0, 0, "x"), _arc_trace(1, 0, "[[1]]")]}
        return {"sample_traces": [_arc_trace(0, 0, "[[1]]")]}
    samples = [_arc_sample("t0"), _arc_sample("t1"), _arc_sample("t2")]
    with mock.patch.object(arc_agi, "_load_arc_agi_samples", return_value=samples), \
            mock.patch.object(arc_agi, "run_eval_suite", side_effect=fake_run):
        arc_agi.run_arc_agi("m", "http://x/v1")
    assert seen_env == ["1/2", "1/2"]
    assert seen_samples == [samples, [samples[0]]]
    import os
    assert os.environ.get("GBENCH_SHARD") == "1/2"


def test_arc_agi_docs_drop_the_unreachable_model_assisted_extraction():
    doc = _doc("arc_agi")
    for text in (doc, arc_agi.__doc__):
        assert "model-assisted" not in text and "asks the model to extract" not in text
    assert "--retry_attempts 2" in doc and "never runs upstream" in doc


# =========================================================================== #
# lab_bench: all 8 categories, zero-shot baseline, LAB-Bench's own parse
# =========================================================================== #
def _lab_row(i, **extra):
    return {"id": f"id{i}", "question": f"Q{i}?", "ideal": f"right{i}",
            "distractors": [f"d{i}a", f"d{i}b"], "canary": "c", "source": None,
            "subtask": "sub-v1-public", **extra}


def _image_bytes(fmt):
    """A real 4x4 image file in `fmt` (the figures are opened with PIL, as upstream does)."""
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (200, 30, 30)).save(buf, format=fmt)
    return buf.getvalue()


_PNG = _image_bytes("PNG")
_JPG = _image_bytes("JPEG")


def _lab_rows():
    rows = {c: [_lab_row(0)] for c, _ in lab_bench.CATEGORIES}
    rows["ProtocolQA"] = [_lab_row(0, protocol="Step 1. Mix.\n")]
    rows["SuppQA"] = [_lab_row(0, source="https://doi.org/x", **{"paper-title": "T"})]
    rows["FigQA"] = [_lab_row(0, figure={"bytes": _JPG, "path": None})]
    rows["TableQA"] = [_lab_row(0, tables=[{"bytes": _PNG, "path": None},
                                           {"bytes": _JPG, "path": None}])]
    return rows


def test_lab_bench_loads_all_eight_categories_as_upstream_task_py_builds_them():
    rows = _lab_rows()
    with mock.patch.object(lab_bench, "_load_category", side_effect=lambda c: rows[c]):
        samples = lab_bench._load_lab_bench_samples()
    by = {s[2]["category"]: s for s in samples}
    assert set(by) == {c for c, _ in lab_bench.CATEGORIES} and len(samples) == 8
    assert "Question: Step 1. Mix.\nQ0?\n\nOptions:" in by["ProtocolQA"][0][0]["content"]
    assert "Question: Paper title: T\nDOI: https://doi.org/x\nQ0?\n" in by["SuppQA"][0][0][
        "content"]
    fig = by["FigQA"][0][0]["content"]
    assert [p["type"] for p in fig] == ["text", "image_url"]        # text first, as upstream
    assert fig[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    table = by["TableQA"][0][0]["content"]
    assert [p["type"] for p in table] == ["text", "image_url", "image_url"]
    assert table[1]["image_url"]["url"].startswith("data:image/png;base64,")
    gold = by["LitQA2"][1]
    assert gold["category"] == "LitQA2" and gold["letters"] == "ABCD"
    assert f"({gold['letter']}) right0" in by["LitQA2"][0][0]["content"]
    assert f"({gold['refuse']}) {lab_bench.REFUSE_CHOICE}" in by["LitQA2"][0][0]["content"]
    assert lab_bench.PUBLIC_SIZE == 1967


def test_lab_bench_scores_the_raw_reply_with_chembench_parse():
    g = {"letter": "B", "letters": "ABCD", "refuse": "C", "category": "SeqQA", "subtask": "s"}
    assert lab_bench._answer_letter("[ANSWER]B[/ANSWER]", "ABCD") == ("B", "tag")
    assert lab_bench._answer_letter("B, D", "ABCD") == ("B", "letters_only")
    for reply in ("Final Answer: B", " B", "B\n", "", None, "\\boxed{B}"):
        assert lab_bench._answer_letter(reply, "ABCD") == (None, "none"), repr(reply)
    traces = [{"response_text": " B", "gold_answer": g, "status": "PENDING_JUDGE"},
              {"response_text": "B", "gold_answer": g, "status": "PENDING_JUDGE",
               "health": "truncated"},
              {"response_text": None, "gold_answer": g, "status": "FAILED"}]
    asyncio.run(lab_bench._async_score_lab_bench(traces))
    assert [t["is_correct"] for t in traces] == [False, True, False]
    assert [t["status"] for t in traces] == ["OK", "TRUNCATED", "FAILED"]


def test_lab_bench_report_is_per_category_with_the_mean_as_headline():
    def t(cat, reply, correct, sub="s"):
        return {"gold_answer": {"letter": "A", "letters": "ABCD", "refuse": "B", "category": cat,
                                "subtask": sub}, "response_text": reply, "is_correct": correct}
    # A prompt over the context window got no reply: base marks it a harness error, and it is
    # left out of every metric (never scored as upstream's unsure answer).
    traces = [t("SeqQA", "[ANSWER]A[/ANSWER]", True), t("SeqQA", "[ANSWER]B[/ANSWER]", False),
              t("DbQA", "[ANSWER]A[/ANSWER]", True, "x"), t("DbQA", "nothing", False, "y"),
              {**t("DbQA", None, False, "y"), "status": "OVER_CONTEXT", "scoring_excluded": True,
               "harness_error": {"kind": "over_context", "detail": "prompt too long"}}]
    rep = lab_bench._lab_bench_report(traces)
    seq, db = rep["categories"]["SeqQA"], rep["categories"]["DbQA"]
    assert (seq["accuracy"], seq["precision"], seq["coverage"]) == (50.0, 100.0, 50.0)
    assert (db["accuracy"], db["n_sure"]) == (50.0, 2)
    assert db["subtasks"]["y"]["n_total"] == 1
    assert list(rep["categories"]) == ["SeqQA", "DbQA"]                 # labbench.Eval order
    assert rep["mean_category_accuracy"] == 50.0
    with mock.patch.object(lab_bench, "_load_lab_bench_samples", return_value=[]), \
            mock.patch.object(lab_bench, "run_eval_suite",
                              return_value={"sample_traces": traces, "failed_requests": 1}):
        res = lab_bench.run_lab_bench("m", "http://x/v1")
    assert res["accuracy"] == rep["mean_category_accuracy"]
    assert res["leaderboard_comparable"] is False
    assert "1 request(s) got no reply from the served endpoint" in \
        res["leaderboard_comparable_reason"]


def _image_server(tokens_per_image, status=200):
    from aiohttp import web
    seen = []

    async def handler(req):
        body = await req.json()
        seen.append(body)
        if status != 200:
            return web.json_response({"error": "image input is not supported"}, status=status)
        images = sum(1 for p in body["messages"][0]["content"] if p["type"] == "image_url")
        return web.json_response({"choices": [{"message": {"content": "x"}}],
                                  "usage": {"prompt_tokens": 10 + images * tokens_per_image}})
    url, stop = _serve({"/v1/chat/completions": handler})
    return url, stop, seen


def _figure_sample():
    rows = _lab_rows()
    with mock.patch.object(lab_bench, "_load_category", side_effect=lambda c: rows[c]):
        return [s for s in lab_bench._load_lab_bench_samples() if s[2]["category"] == "FigQA"]


@pytest.mark.parametrize("tokens,status,ok", [(260, 200, True), (0, 200, False),
                                              (0, 400, False)])
def test_lab_bench_requires_an_endpoint_that_processes_images(tokens, status, ok):
    url, stop, seen = _image_server(tokens, status)
    try:
        if ok:
            lab_bench._require_image_input(url, "m", _figure_sample())
        else:
            with pytest.raises(RuntimeError, match="INFRA REQUIRED.*lab_bench.*image"):
                lab_bench._require_image_input(url, "m", _figure_sample())
        lab_bench._require_image_input(url, "m", [([{"role": "user", "content": "text"}], {},
                                                   {"category": "SeqQA"})])
    finally:
        stop()
    assert len(seen) in (1, 2) and seen[0]["max_tokens"] == 1   # text-only samples: no probe


def test_lab_bench_run_uses_the_raw_reply_scorer():
    seen = {}
    with mock.patch.object(lab_bench, "_load_lab_bench_samples", return_value=[]), \
            mock.patch.object(lab_bench, "run_eval_suite",
                              side_effect=lambda **kw: seen.update(kw) or {"sample_traces": []}):
        lab_bench.run_lab_bench("m", "http://x/v1")
    assert seen["async_eval_fn"] is lab_bench._async_score_lab_bench
    assert seen["declared_scoring_mode"] == "deterministic" and "eval_fn" not in seen


# =========================================================================== #
# cruxeval: 10 samples, pass@1 at 0.2 and pass@5 at 0.8, both columns reported
# =========================================================================== #
def _crux_traces(kw, passing=True, tokens=20):
    out = []
    for i, (_, gold, meta) in enumerate(kw["samples"]):
        for a in range(kw["attempt_count"]):
            out.append({"sample_idx": i * kw["attempt_count"] + a, "source_sample_idx": i,
                        "gold_answer": gold, "cruxeval_check": "passed" if passing else "failed",
                        "completion_tokens": tokens, "is_correct": passing})
    return out


def _crux_samples():
    rows = [{"code": "def f(x):\n    return x", "input": "1", "output": "1", "id": "s0"}]
    with mock.patch("datasets.load_dataset", return_value=rows):
        return cruxeval._load_cruxeval_samples(), cruxeval._load_cruxeval_samples(
            enable_thinking=True)


def _run_crux(enable_thinking=False, tokens=20, **kwargs):
    direct, cot = _crux_samples()
    calls = []

    def fake_run(**kw):
        calls.append(kw)
        temp = kw["official_sampling"].temperature
        return {"temperature": temp, "attempts": {"attempts_per_sample": kw["attempt_count"]},
                "sample_traces": _crux_traces(kw, passing=temp < 0.5, tokens=tokens),
                "failed_requests": 0, "status": "success"}
    with mock.patch.object(cruxeval, "sandbox_skip_reason", return_value=None), \
            mock.patch.object(cruxeval, "_grader_self_check", return_value=None), \
            mock.patch.object(cruxeval, "_load_cruxeval_samples",
                              return_value=cot if enable_thinking else direct), \
            mock.patch.object(cruxeval, "run_eval_suite", side_effect=fake_run):
        return cruxeval.run_cruxeval("m", "http://x/v1", 1, enable_thinking=enable_thinking,
                                     **kwargs), calls


def test_cruxeval_no_think_runs_the_two_published_passes():
    res, calls = _run_crux()
    assert [c["official_sampling"].temperature for c in calls] == [0.2, 0.8]
    assert all("temperature" not in c for c in calls)
    assert all(c["attempt_count"] == 10 for c in calls)
    rep = res["cruxeval_report"]
    for label in ("CRUXEval-O", "CRUXEval-I"):
        assert rep[label]["pass_at_1"] == 100.0 and rep[label]["pass_at_1_temperature"] == 0.2
        assert rep[label]["pass_at_5"] == 0.0 and rep[label]["pass_at_5_temperature"] == 0.8
        assert rep[label]["leaderboard_comparable"] is True
    assert res["accuracy"] == 100.0 and res["pass_at_5_run"]["temperature"] == 0.8
    assert res["leaderboard_comparable"] is False
    assert res["leaderboard_comparable_reason"].startswith("headline is the mean")


def test_cruxeval_thinking_runs_the_same_two_passes_and_is_flagged():
    # openai_run.py samples the CoT prompts at 0.2 and 0.8 too.
    res, calls = _run_crux(enable_thinking=True)
    assert [c["official_sampling"].temperature for c in calls] == [0.2, 0.8]
    col = res["cruxeval_report"]["CRUXEval-O"]
    assert col["leaderboard_comparable"] is False and "--thinking run" in col[
        "leaderboard_comparable_reason"]
    assert "pass_at_5" in col and col["pass_at_5_temperature"] == 0.8


def test_cruxeval_flags_direct_replies_past_upstreams_token_cap():
    res, _ = _run_crux(tokens=150)
    col = res["cruxeval_report"]["CRUXEval-I"]
    assert col["leaderboard_comparable"] is False
    assert "ran past the upstream chat path's 100-token cap" in col[
        "leaderboard_comparable_reason"]


# =========================================================================== #
# docs
# =========================================================================== #
@pytest.mark.parametrize("suite,module", [
    ("mmlu", mmlu), ("mmlu_redux", mmlu_redux), ("gpqa", gpqa), ("gpqa_diamond", gpqa_diamond),
    ("multilingual_mmlu", multilingual_mmlu), ("arc_agi", arc_agi), ("lab_bench", lab_bench),
    ("cruxeval", cruxeval)])
def test_docs_name_the_pinned_revision_and_a_runnable_command(suite, module):
    doc = _doc(suite)
    rev = (getattr(module, "DATASET_REVISION", None) or getattr(module, "_DATASET_REVISION"))
    assert rev in doc
    run = doc.split("## Run", 1)[1]
    blocks = [b.split("```", 1)[0] for b in run.split("```bash")[1:]]
    assert blocks and all(f"--evals {suite}" in b and "--max-output-tokens" in b for b in blocks)
    if suite != "multilingual_mmlu":            # quotes its localized answer anchors
        assert doc.isascii()
