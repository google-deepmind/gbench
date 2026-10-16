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

"""Canonical fidelity: ruler, omnidocbench, loft_scifact, i18n_translate, toolchains doc.

Every expectation here was checked against the upstream source it cites (RULER c3f5e3b4,
OmniDocBench 193627ae, LOFT 219f68e, sacrebleu 2.6.0); nothing contacts a model server.
"""
import hashlib
import json
import os
import subprocess
import sys
import types
import warnings

warnings.filterwarnings("ignore")

import pytest

import gbench.runners.eval_suites.i18n_translate as I
import gbench.runners.eval_suites.loft_scifact as L
import gbench.runners.eval_suites.omnidocbench as O
import gbench.runners.eval_suites.ruler as R

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _doc(name):
    with open(os.path.join(_REPO, "docs", "evals", name), encoding="utf-8") as f:
        return f.read()


# --------------------------------------------------------------------------- #
# ruler
# --------------------------------------------------------------------------- #

def test_ruler_window_equal_to_the_128k_band_is_enough(monkeypatch):
    # RULER's max_seq_length already holds the template, prompt and generation cap.
    monkeypatch.setattr(R, "_get_server_max_model_len", lambda u: 131072)
    R._require_context("http://x/v1", 131072)
    monkeypatch.setattr(R, "_get_server_max_model_len", lambda u: 131071)
    with pytest.raises(RuntimeError, match="131072"):
        R._require_context("http://x/v1", 131072)


def _fake_git(head, dirty="", head_rc=0):
    def run(cmd, **kw):
        if "rev-parse" in cmd:
            return types.SimpleNamespace(returncode=head_rc, stdout=head + "\n", stderr="fatal")
        return types.SimpleNamespace(returncode=0, stdout=dirty, stderr="")
    return run


def test_ruler_revision_is_verified(monkeypatch):
    monkeypatch.setattr(R.shutil, "which", lambda _x: "/usr/bin/git")
    monkeypatch.setattr(R.subprocess, "run", _fake_git(R._RULER_REVISION))
    assert R._verify_revision("/r") == R._RULER_REVISION

    monkeypatch.setattr(R.subprocess, "run", _fake_git("0" * 40))
    with pytest.raises(RuntimeError, match=R._RULER_REVISION):
        R._verify_revision("/r")

    monkeypatch.setattr(R.subprocess, "run", _fake_git(R._RULER_REVISION, dirty=" M scripts/data/prepare.py\n"))
    with pytest.raises(RuntimeError, match="local changes"):
        R._verify_revision("/r")

    monkeypatch.setattr(R.subprocess, "run", _fake_git("", head_rc=128))
    with pytest.raises(RuntimeError, match="cannot read the revision"):
        R._verify_revision("/r")

    monkeypatch.setattr(R.shutil, "which", lambda _x: None)
    with pytest.raises(RuntimeError, match="git not found"):
        R._verify_revision("/r")


def test_ruler_served_chat_template_is_rendered_around_the_slot(monkeypatch):
    class Tok:
        def apply_chat_template(self, msgs, tokenize, add_generation_prompt, enable_thinking):
            assert tokenize is False and add_generation_prompt is True
            sys_turn = "<sys think>" if enable_thinking else ""
            return f"<bos>{sys_turn}<user>{msgs[0]['content']}<end><model>"

        def tokenize(self, text):
            return list(text)

    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda name: Tok())
    text, n = R._served_chat_template("any/tok", enable_thinking=False)
    assert text == "<bos><user>{task_template}<end><model>" and n == len(text)
    assert "<sys think>" in R._served_chat_template("any/tok", enable_thinking=True)[0]

    def boom(name):
        raise OSError("no such tokenizer")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", boom)
    with pytest.raises(RuntimeError, match="docs/evals/ruler.md"):
        R._served_chat_template("missing/tok", enable_thinking=False)


def test_ruler_bootstrap_registers_the_template_and_runs_prepare_unmodified(tmp_path):
    """The real mechanism against a stand-in data dir: template.py + a prepare.py that records
    what it received (the real prepare.py reads Templates[args.model_template_type])."""
    (tmp_path / "template.py").write_text("Templates = {'base': '{task_template}'}\n")
    (tmp_path / "prepare.py").write_text(
        "import sys, json\n"
        "from template import Templates\n"
        "t = sys.argv[sys.argv.index('--model_template_type') + 1]\n"
        "json.dump({'argv': sys.argv, 'template': Templates[t]}, open('seen.json', 'w'))\n")
    chat = "<bos><|turn>user\n{task_template}<turn|>\n<|turn>model\n"
    proc = subprocess.run(
        [sys.executable, "-c", R._PREPARE_BOOTSTRAP, R._TEMPLATE_NAME, chat,
         "--task", "vt", "--model_template_type", R._TEMPLATE_NAME, "--prepare_for_ns"],
        cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    seen = json.loads((tmp_path / "seen.json").read_text())
    assert seen["template"] == chat
    assert seen["argv"][0] == "prepare.py"
    assert seen["argv"][1:] == ["--task", "vt", "--model_template_type", R._TEMPLATE_NAME,
                                "--prepare_for_ns"]


def test_ruler_prepare_task_uses_ns_mode_and_a_revision_keyed_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(R.Path, "home", classmethod(lambda cls: tmp_path))
    seen = {}

    def fake_run(cmd, cwd=None, env=None, capture_output=None, text=None):
        seen["cmd"] = cmd
        save = cmd[cmd.index("--save_dir") + 1]
        out = os.path.join(save, "niah_single_1")
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "validation.jsonl"), "w") as f:
            for i in range(seen.get("rows", 3)):
                f.write(json.dumps({"input": "x", "answer_prefix": " y", "outputs": ["1"]}) + "\n")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(R.subprocess, "run", fake_run)
    rows = R._prepare_task("/ruler", "org/tok", "niah_single_1", 4096, 3,
                           chat_template="<t>{task_template}</t>", revision=R._RULER_REVISION)
    assert len(rows) == 3
    cmd = seen["cmd"]
    assert cmd[1] == "-c" and cmd[2] == R._PREPARE_BOOTSTRAP
    assert cmd[3] == R._TEMPLATE_NAME and cmd[4] == "<t>{task_template}</t>"
    assert "--prepare_for_ns" in cmd
    assert cmd[cmd.index("--model_template_type") + 1] == R._TEMPLATE_NAME
    assert cmd[cmd.index("--max_seq_length") + 1] == "4096"
    save = cmd[cmd.index("--save_dir") + 1]
    tmpl_hash = hashlib.sha256("<t>{task_template}</t>".encode()).hexdigest()[:12]
    assert f"{R._RULER_REVISION[:12]}-{tmpl_hash}" in save

    # prepare.py exits 0 even when its generator fails: a short file is an error, not data.
    seen["rows"] = 2
    with pytest.raises(RuntimeError, match="2/5 rows"):
        R._prepare_task("/ruler", "org/tok", "niah_single_1", 8192, 5,
                        chat_template="<t>{task_template}</t>", revision=R._RULER_REVISION)


def _run_ruler(monkeypatch, temperature=0.0, status="OK", limit=None, thinking=False):
    import gbench.runners.eval_suites.base as B
    monkeypatch.setattr(B, "_RUN_KNOBS", {})        # no window known: the pre-flight has nothing to check
    monkeypatch.setattr(R, "_ruler_dir", lambda: "/ruler")
    monkeypatch.setattr(R, "_verify_revision", lambda d: R._RULER_REVISION)
    monkeypatch.setattr(R, "_require_context", lambda u, b: None)
    monkeypatch.setattr(R, "_served_chat_template", lambda tok, think: ("<t>{task_template}", 7))
    monkeypatch.delenv("GBENCH_RULER_LENGTHS", raising=False)
    monkeypatch.delenv("GBENCH_RULER_NUM_SAMPLES", raising=False)

    def fake_prepare(ruler_dir, tokenizer, task, band, num_samples, chat_template=None, revision=None):
        assert chat_template == "<t>{task_template}" and revision == R._RULER_REVISION
        return [{"input": f"CTX {task} {band}.", "answer_prefix": " Answer: the value is",
                 "outputs": ["42"]}]
    monkeypatch.setattr(R, "_prepare_task", fake_prepare)
    captured = {}

    def fake_suite(**kw):
        captured.update(kw)
        traces = [{"gold_answer": g, "response_text": "42", "category": m["category"],
                   "status": status} for _msgs, g, m in kw["samples"]]
        return {"sample_traces": traces, "temperature": temperature}
    monkeypatch.setattr(R, "run_eval_suite", fake_suite)
    kwargs = {"tokenizer": "org/tok"}
    if limit:
        kwargs["limit"] = limit
    return R.run_ruler("m", "http://x/v1", 4, enable_thinking=thinking, **kwargs), captured


def test_ruler_sends_input_plus_answer_prefix(monkeypatch):
    res, captured = _run_ruler(monkeypatch)
    msgs, _gold, meta = captured["samples"][0]
    assert msgs == [{"role": "user", "content": "CTX niah_single_1 4096. Answer: the value is"}]
    assert meta["max_tokens"] == 128
    assert res["ruler_revision"] == R._RULER_REVISION and res["ruler_template_tokens"] == 7


def test_ruler_full_greedy_matrix_is_comparable(monkeypatch):
    res, _ = _run_ruler(monkeypatch, temperature=0.0)
    assert res["accuracy"] == 100.0
    assert res["leaderboard_comparable"] is True
    assert "leaderboard_comparable_reason" not in res


def test_ruler_comparability_checks_the_resolved_temperature(monkeypatch):
    res, _ = _run_ruler(monkeypatch, temperature=0.7)       # e.g. GBENCH_RULER_TEMPERATURE=0.7
    assert res["leaderboard_comparable"] is False
    assert "temperature 0.7" in res["leaderboard_comparable_reason"]


def test_ruler_unsent_prompts_are_an_infra_error_not_a_zero(monkeypatch):
    # RULER data is built to fit: an OVER_CONTEXT trace is a serving mismatch, never scored.
    with pytest.raises(RuntimeError, match=r"INFRA REQUIRED.*did not fit the served context window"):
        _run_ruler(monkeypatch, status="OVER_CONTEXT")


def test_ruler_limit_is_named_and_thinking_caps_follow_the_token_limit_policy(monkeypatch):
    res, captured = _run_ruler(monkeypatch, temperature=1.0, thinking=True, limit=10)
    reason = res["leaderboard_comparable_reason"]
    assert "--eval-limit" in reason and "temperature 1.0" in reason
    # the per-task caps ride in each sample's meta in both modes; base's token-limit policy
    # replaces them with the run's budget on a --thinking run (not a comparability reason)
    assert "--thinking" not in reason
    assert all(m["max_tokens"] in (128, 30, 120, 50, 32) for _msgs, _g, m in captured["samples"])


#: Room RULER leaves at a band beyond the generation cap: prepare.py counts the literal
#: `{task_template}` slot inside the chat template, which the served prompt does not contain.
_SLOT_TOKENS = 5


def _ruler_on_a_131072_window(monkeypatch, served_count, tokenize=True, thinking=False,
                              local_count=None):
    """run_ruler over all 13 tasks x 6 bands with base's real pre-send check (clamp_to_context)
    and a simulated server: /tokenize answers `served_count(task, band)` for each prompt, or is
    absent (404) so base counts locally with `local_count(task, band)` text tokens."""
    import io
    import urllib.error
    import gbench.runners.eval_suites.base as B
    monkeypatch.setattr(B, "_RUN_KNOBS", {"max_model_len": 131072, "served_model": "m",
                                          "tokenize_url": "http://tokenize.invalid"})
    monkeypatch.setattr(B, "_COUNT_CACHE", {})
    monkeypatch.setattr(B, "_NO_TOKENIZE_ENDPOINT", set())
    monkeypatch.setattr(R, "_get_server_max_model_len", lambda u: 131072)
    monkeypatch.setattr(R, "_ruler_dir", lambda: "/ruler")
    monkeypatch.setattr(R, "_verify_revision", lambda d: R._RULER_REVISION)
    monkeypatch.setattr(R, "_served_chat_template", lambda tok, think: ("<t>{task_template}", 7))
    monkeypatch.delenv("GBENCH_RULER_LENGTHS", raising=False)
    monkeypatch.delenv("GBENCH_RULER_NUM_SAMPLES", raising=False)
    monkeypatch.setattr(R, "_prepare_task", lambda d, tok, task, band, n, chat_template=None,
                        revision=None: [{"input": f"CTX {task} {band}", "answer_prefix": " A:",
                                         "outputs": ["42"]}])
    bodies = []

    def urlopen(req, timeout=None):
        body = json.loads(req.data)
        bodies.append(body)
        if not tokenize:
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, io.BytesIO(b"no route"))
        _ctx, task, band = body["messages"][0]["content"].split()[:3]
        return io.BytesIO(json.dumps({"count": served_count(task, int(band))}).encode())

    monkeypatch.setattr(B.urllib.request, "urlopen", urlopen)
    if local_count is not None:
        monkeypatch.setattr(B, "_count_tokens",
                            lambda text: local_count(text.split()[1], int(text.split()[2])))
    sent = []

    def fake_suite(**kw):
        # What base's _fetch_sample does per sample: the sample's max_tokens (the task cap), the
        # run's chat_template_kwargs, then clamp_to_context against the served window.
        for msgs, gold, meta in kw["samples"]:
            ctk = B.effective_chat_template_kwargs(None, kw["thinking"])
            budget = meta.get("max_tokens", 32768)
            sent.append((meta["category"], budget, B.clamp_to_context(msgs, budget, None, None, ctk)))
        traces = [{"gold_answer": g, "response_text": "42", "category": m["category"],
                   "status": "OK"} for _m, g, m in kw["samples"]]
        return {"sample_traces": traces, "temperature": 0.0}

    monkeypatch.setattr(R, "run_eval_suite", fake_suite)
    res = R.run_ruler("m", "http://x/v1", 4, enable_thinking=thinking, tokenizer="org/tok")
    return res, sent, bodies


def test_ruler_every_band_passes_the_pre_send_check_on_a_131072_window(monkeypatch):
    # RULER's budget: served prompt (template + input + answer prefix) + cap <= band. Simulated
    # /tokenize counts at that bound (and at the slot-token margin the real data leaves).
    for margin in (0, _SLOT_TOKENS):
        def served(task, band):
            return band - R._GEN_CAP[R._base_type(task)] - margin
        res, sent, bodies = _ruler_on_a_131072_window(monkeypatch, served)
        assert res["leaderboard_comparable"] is True and res["accuracy"] == 100.0
        assert len(sent) == len(R._TASKS) * len(R._DEFAULT_LENGTHS) == 78
        for cell, budget, (max_tokens, refused) in sent:
            assert refused is None, (cell, refused)
            assert max_tokens == budget == R._GEN_CAP[R._base_type(cell.split("@")[0])]
        assert {c for c, _b, _r in sent if c.endswith("@128k")} == {f"{t}@128k" for t in R._TASKS}
        # the count is exact for the request: same template kwargs, generation prompt added
        assert all(b["chat_template_kwargs"] == {"enable_thinking": False}
                   and b["add_generation_prompt"] is True for b in bodies)


def test_ruler_prompt_over_the_band_budget_is_refused_before_sending(monkeypatch):
    def served(task, band):
        return band - R._GEN_CAP[R._base_type(task)] + (1 if band == 131072 else 0)
    with pytest.raises(RuntimeError, match=r"INFRA REQUIRED.*longest \w+@128k prompt would not be sent"):
        _ruler_on_a_131072_window(monkeypatch, served)


def test_ruler_without_tokenize_the_128k_band_is_an_infra_error(monkeypatch):
    # No /tokenize: base counts text tokens locally plus template allowances and a 64-token
    # slack, which a window equal to the band cannot absorb. Hard error, not scored zeros.
    def local(task, band):
        return band - R._GEN_CAP[R._base_type(task)] - 7 - _SLOT_TOKENS
    with pytest.raises(RuntimeError, match=r"INFRA REQUIRED.*@128k prompt would not be sent.*/tokenize"):
        _ruler_on_a_131072_window(monkeypatch, None, tokenize=False, local_count=local)


def test_ruler_thinking_on_a_window_equal_to_the_band_is_an_infra_error(monkeypatch):
    # --thinking lifts the caps, so the check wants 256 tokens of room; RULER leaves about the cap.
    def served(task, band):
        return band - R._GEN_CAP[R._base_type(task)] - _SLOT_TOKENS
    with pytest.raises(RuntimeError, match=r"INFRA REQUIRED.*@128k prompt would not be sent"):
        _ruler_on_a_131072_window(monkeypatch, served, thinking=True)


def test_ruler_docs_describe_the_hard_error_not_a_zero():
    doc = _doc("ruler.md")
    assert "Caveat" not in doc and "scored 0," not in doc
    assert "hard-errors" in doc and "OVER_CONTEXT" in doc and "/tokenize" in doc
    assert "prefill" in doc                          # the answer-prefix placement is explained


def test_ruler_docs_match_the_pin():
    doc = _doc("ruler.md")
    assert R._RULER_REVISION in doc
    assert "140k" not in doc and "140000" not in doc     # the old over-estimate is gone
    assert "--prepare_for_ns" in doc and "answer_prefix" in doc


# --------------------------------------------------------------------------- #
# omnidocbench
# --------------------------------------------------------------------------- #

#: sha256 of the prompt string in opendatalab/OmniDocBench tools/model_infer/Qwen2VL_img2md.py
#: @ 193627ae (identical in gemini25/gemini3_flash/gemini3_pro/internvl2_test_img2md.py).
_OMNIDOC_PROMPT_SHA256 = "7527266003c4b68cb4fd2e963f4e6ce2afeec11c94c96de411cb484cc3f243d0"


def test_omnidocbench_prompt_is_the_official_vlm_instruction_verbatim():
    assert hashlib.sha256(O._PROMPT.encode("utf-8")).hexdigest() == _OMNIDOC_PROMPT_SHA256
    assert O._PROMPT.startswith("You are an AI assistant specialized in converting PDF images")
    assert "Wrap the entire table with <table> and </table>." in O._PROMPT
    assert r"Enclose inline formulas with \( \)." in O._PROMPT


class _FakeDS:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]

    def cast_column(self, col, feature):
        return _FakeDS([{"image": {"path": r["_path"], "bytes": None}} for r in self.rows])


def test_omnidocbench_loader_pins_revision_and_builds_image_then_prompt(monkeypatch, tmp_path):
    import datasets
    calls = []

    def fake_load(name, split=None, revision=None, **kw):
        calls.append(("load", name, revision))
        return _FakeDS([{"image": b"\x89PNG-bytes", "_path": "/c/images/p1.jpg"},
                        {"image": b"\x89PNG-art", "_path": "/c/data_diversity.png"}])
    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    gt = tmp_path / "OmniDocBench.json"
    gt.write_text(json.dumps([{"page_info": {"image_path": "images/p1.jpg",
                                             "page_attribute": {"data_source": "book"}},
                               "layout_dets": [{"category_type": "text_block", "text": "hi",
                                                "order": 0}]}]))

    def fake_hub(repo_id, filename, repo_type, revision=None):
        calls.append(("hub", repo_id, revision))
        return str(gt)
    monkeypatch.setattr(O, "hf_hub_download", fake_hub)
    samples = O._load_omnidocbench_samples()
    assert all(c[2] == O._DATASET_REVISION for c in calls) and len(calls) == 2
    assert len(samples) == 1                       # the unannotated artwork image is excluded
    content = samples[0][0][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[1] == {"type": "text", "text": O._PROMPT}


def test_omnidocbench_evaluator_revision_is_verified_inside_the_image(monkeypatch):
    def fake(stdout, rc=0):
        return lambda cmd, **kw: types.SimpleNamespace(returncode=rc, stdout=stdout, stderr="")

    monkeypatch.setattr(O.subprocess, "run", fake(O._EVALUATOR_REVISION + "\n"))
    O._verify_evaluator_revision("img", "")
    monkeypatch.setattr(O.subprocess, "run", fake("f133a71e9e91c3621c7ce8994200a7b394a06eb3\n"))
    with pytest.raises(RuntimeError, match=O._EVALUATOR_REVISION):
        O._verify_evaluator_revision("img", "")
    monkeypatch.setattr(O.subprocess, "run", fake(O._EVALUATOR_REVISION + "\n M metrics/cal_metric.py\n"))
    with pytest.raises(RuntimeError, match="local"):
        O._verify_evaluator_revision("img", "")
    monkeypatch.setattr(O.subprocess, "run", fake("", rc=128))
    with pytest.raises(RuntimeError, match="cannot read the OmniDocBench revision"):
        O._verify_evaluator_revision("img", "")


def test_omnidocbench_requires_rapidfuzz(monkeypatch):
    real = O.importlib.util.find_spec
    monkeypatch.setattr(O.importlib.util, "find_spec",
                        lambda name: None if name == "rapidfuzz" else real(name))
    with pytest.raises(RuntimeError, match="rapidfuzz"):
        O._check_prereqs(O._SCORER_IMAGE_DEFAULT)


def _run_omnidoc(monkeypatch, tmp_path, temperature, limit=None, thinking=False, mot=None):
    monkeypatch.setattr(O, "_check_prereqs", lambda image: None)
    gt = tmp_path / "OmniDocBench.json"
    gt.write_text(json.dumps([{"page_info": {"image_path": "images/p1.jpg"}, "layout_dets": []}]))
    monkeypatch.setattr(O, "hf_hub_download", lambda **k: str(gt))
    monkeypatch.setattr(O, "_load_omnidocbench_samples", lambda limit=None: [("m", "g", {})])
    captured = {}

    def fake_suite(**kw):
        captured.update(kw)
        return {"accuracy": 0.0, "temperature": temperature, "sample_traces": [
            {"response_text": "# hi", "gold_answer": json.dumps({"page_name": "p1.jpg"})}]}
    monkeypatch.setattr(O, "run_eval_suite", fake_suite)

    def fake_docker(cmd, **kw):
        out = [c.split(":")[0] for c in cmd if isinstance(c, str) and c.endswith(":/workspace/result")][0]
        with open(f"{out}/predictions_quick_match_run_summary.json", "w") as f:
            json.dump({"notebook_metric_summary": {"overall_notebook": 50.0, "metrics": {}}}, f)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(O.subprocess, "run", fake_docker)
    r = O.run_omnidocbench("m", "http://x", limit=limit, enable_thinking=thinking,
                           max_output_tokens=mot)
    return r, captured


def test_omnidocbench_full_greedy_run_is_comparable(monkeypatch, tmp_path):
    r, captured = _run_omnidoc(monkeypatch, tmp_path, temperature=0.0)
    assert r["leaderboard_comparable"] is True
    assert r["omnidocbench_evaluator_revision"] == O._EVALUATOR_REVISION
    # Qwen2VL_img2md.py max_new_tokens, applied by the token-limit policy
    assert captured["official_sampling"].max_tokens == 32000


def test_omnidocbench_comparability_checks_the_resolved_temperature(monkeypatch, tmp_path):
    r, _ = _run_omnidoc(monkeypatch, tmp_path, temperature=1.0)   # no --thinking, but sampled
    assert r["leaderboard_comparable"] is False
    assert "temperature 1.0" in r["leaderboard_comparable_reason"]


def test_omnidocbench_passes_the_flag_to_the_token_limit_policy(monkeypatch, tmp_path):
    # the flag reaches run_eval_suite, whose policy sends it only with --thinking; without it
    # every request carries the reference script's 32000
    _r, captured = _run_omnidoc(monkeypatch, tmp_path, temperature=0.0, mot=48000)
    assert captured["max_output_tokens"] == 48000
    assert captured["official_sampling"].max_tokens == 32000


def test_omnidocbench_docs_match_the_code():
    doc = _doc("omnidocbench.md")
    assert O._DATASET_REVISION in doc and O._EVALUATOR_REVISION in doc
    assert "32000" in doc and "Qwen2VL_img2md.py" in doc
    with open(os.path.join(_REPO, "docker", "omnidocbench.Dockerfile"), encoding="utf-8") as f:
        assert O._EVALUATOR_REVISION in f.read()


# --------------------------------------------------------------------------- #
# loft_scifact
# --------------------------------------------------------------------------- #

_CORPUS = [
    {"pid": "900", "title_text": " Alpha title ", "passage_text": "Alpha body about childhood abuse cases."},
    {"pid": "17", "title_text": "", "passage_text": "Beta body."},
    {"pid": "5", "title_text": "Gamma", "passage_text": "Gamma body."},
]
_FEW = [{"qid": "f1", "query_text": "Few claim.", "metadata": {"qrels": [["5", 1]]},
         "answers": [["5", 1]]}]
_TEST = [{"qid": "t1", "query_text": " Test claim. ", "metadata": {"qrels": [["17", 1]]},
          "answers": [["17", 1]]}]

#: What LOFT's registry renders for this corpus (prompts/utils.py chunk semantics: each chunk is
#: its text + "\n"; the prompt is the chunks concatenated and stripped).
_EXPECTED = (
    "You will be given a list of passages. You need to read carefully and understand all of "
    "them. Then you will be given a claim, and your goal is to find all passages from the list "
    "that can help verify the claim as true of false. Print out the ID and TITLE of each passage.\n"
    "\n"
    "Your final answer should be a list of IDs, in the following format:\n"
    "Final Answer: [id1, id2, ...]\n"
    "If there is only one ID, it should be in the format:\n"
    "Final Answer: [id1]\n"
    "\n"
    "If there is no perfect answer output the closest one. Do not give an empty final answer.\n"
    "\n"
    "ID: 0 | TITLE: Alpha title | CONTENT: Alpha body about  cases. | END ID: 0\n"
    "ID: 1 | TITLE: None | CONTENT: Beta body. | END ID: 1\n"
    "ID: 2 | TITLE: Gamma | CONTENT: Gamma body. | END ID: 2\n"
    "====== Example 1 ======\n"
    "Which passage is most relevant to the claim, and can help verify the claim as true or false? "
    "Print out the TITLE and ID of the document. Then format the IDs into a list.\n"
    "If there is no perfect answer output the closest one. Do not give an empty final answer.\n"
    "claim: Few claim.\n"
    "The following passages can help verify this sentence:\n"
    "TITLE: Gamma | ID: 2\n"
    "Final Answer: ['2']\n"
    "====== Now let's start! ======\n"
    "Which passage is most relevant to the claim, and can help verify the claim as true or false? "
    "Print out the TITLE and ID of the document. Then format the IDs into a list.\n"
    "If there is no perfect answer output the closest one. Do not give an empty final answer.\n"
    "claim: Test claim.\n"
    "The following passages can help verify this sentence:"
)


def test_loft_prompt_matches_the_registry_rendering():
    prompts, pid_mapper = L.build_loft_prompts(_CORPUS, _FEW, _TEST)
    assert pid_mapper == {"900": "0", "17": "1", "5": "2"}
    assert prompts == [_EXPECTED]
    # load_data_from_file cleans titles/passages only; queries are just stripped.
    q = [{"qid": "t2", "query_text": "About childhood abuse.", "answers": [["17", 1]]}]
    prompt = L.build_loft_prompts(_CORPUS, _FEW, q)[0][0]
    assert prompt.endswith("claim: About childhood abuse.\n"
                           "The following passages can help verify this sentence:")


@pytest.mark.parametrize("text,expected", [
    # each pair checked against LOFT utils.extract_prediction @ 219f68e
    ("Final Answer: ['12']", ["12"]),
    ("TITLE: x | ID: 12\nFinal Answer: [12, 7]", [12, 7]),
    ("**Final Answer:** [`3`]", [3]),
    ("I think [maybe] it is\nFinal Answer: [5]", []),         # first bracket line only
    ("Final Answer: ['a', 'the devil's sleep']", ["a", "the devil's sleep"]),
    ("Final Answer: [1] and [2]", []),
    ("no brackets", []),
    ("", []),
])
def test_loft_extract_prediction_is_upstream(text, expected):
    assert L._extract_prediction(text) == expected


def test_loft_recall_at_1_with_the_id_mapping():
    gold = {"ids": ["1"], "pids": ["17"], "num_docs": 3}
    assert L._eval_loft_scifact("TITLE: Beta | ID: 1\nFinal Answer: ['1']", gold) is True
    assert L._eval_loft_scifact("Final Answer: [1, 2]", gold) is True
    assert L._eval_loft_scifact("Final Answer: [2, 1]", gold) is False      # recall@1: first only
    # LOFT maps every predicted ID; one outside the corpus drops the whole response.
    assert L._eval_loft_scifact("Final Answer: [1, 7]", gold) is False
    assert L._eval_loft_scifact("Final Answer: ['17']", gold) is False      # a raw pid is not an ID
    assert L._eval_loft_scifact("Final Answer: []", gold) is False


def test_loft_bundle_hash_mismatch_is_a_hard_error(monkeypatch, tmp_path):
    monkeypatch.setattr(L, "_CACHE_DIR", str(tmp_path))

    def fake_retrieve(url, dest):
        with open(dest, "wb") as f:
            f.write(b"not the pinned bundle")
    monkeypatch.setattr(L.urllib.request, "urlretrieve", fake_retrieve)
    with pytest.raises(RuntimeError, match=L._SCIFACT_SHA256):
        L._ensure_scifact("128k")


def _run_loft(monkeypatch, temperature, limit=None, thinking=False):
    monkeypatch.setattr(L, "_load_loft_scifact_samples", lambda limit=None: [("m", {}, {})] * (limit or 100))
    monkeypatch.setattr(L, "run_eval_suite", lambda **kw: {"accuracy": 50.0, "temperature": temperature})
    kwargs = {"limit": limit} if limit else {}
    return L.run_loft_scifact("m", "http://x", 4, enable_thinking=thinking, **kwargs)


def test_loft_full_greedy_run_is_comparable_and_labelled_scifact(monkeypatch):
    r = _run_loft(monkeypatch, 0.0)
    assert r["leaderboard_comparable"] is True
    assert r["benchmark"] == "LOFT retrieval_scifact_128k_test:few_shot_with_cot"
    r = _run_loft(monkeypatch, 1.0, limit=10)
    assert r["leaderboard_comparable"] is False
    assert "temperature 1.0" in r["leaderboard_comparable_reason"]
    assert "--eval-limit" in r["leaderboard_comparable_reason"]


def test_loft_docs_match_the_pin():
    doc = _doc("loft_scifact.md")
    assert L._SCIFACT_SHA256 in doc and "219f68e470260c22fc53cc8cc2416b78313db774" in doc
    assert "recall@1" in doc and "SciFact" in doc


# --------------------------------------------------------------------------- #
# i18n_translate
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("hyp,ref,expected", [
    # sacrebleu 2.6.0: sacrebleu.sentence_chrf(hyp, [ref]).score
    ("Munich 1856: Four maps that change your perspective of the city",
     "Munich 1856: Four maps that will change your view of the city", 72.42536317122938),
    ("Yes", "Yes.", 68.86227544910179),
    ("ok", "No", 25.0),
    ("", "Hello there", 0.0),
    ("The cat sat on the mat.", "the cat sat on the mat", 91.72483478088577),
    ("Sie sind hier", "They are here", 7.727272727272727),
])
def test_i18n_sentence_chrf_equals_sacrebleu(hyp, ref, expected):
    assert I.chrf_score(hyp, ref) == pytest.approx(expected, abs=1e-9)


def test_i18n_corpus_chrf_equals_sacrebleu():
    hyps = ["Munich 1856: Four maps that change your perspective of the city", "Yes", "ok", "",
            "The cat sat on the mat.", "Sie sind hier"]
    refs = ["Munich 1856: Four maps that will change your view of the city", "Yes.", "No",
            "Hello there", "the cat sat on the mat", "They are here"]
    # sacrebleu 2.6.0: sacrebleu.corpus_chrf(hyps, [refs]).score
    assert I.chrf_corpus(hyps, refs) == pytest.approx(63.04138659464644, abs=1e-9)


def _fake_wmt(monkeypatch, per_direction=None):
    """A stand-in for sacrebleu's WMT19 test set: `per_direction` segments per direction."""
    class _DS:
        langpairs = dict(I.DIRECTIONS)

        def source(self, direction):
            return [f"{direction} seg {i}" for i in range(counts[direction])]

        def references(self, direction):
            return [(f"ref {direction} {i}",) for i in range(counts[direction])]

    counts = {d: (per_direction or n) for d, n in I.DIRECTIONS.items()}
    monkeypatch.setattr(I, "DIRECTIONS", counts)
    monkeypatch.setattr(I, "_wmt19_dataset", lambda: _DS())


def test_i18n_default_is_every_segment_of_every_wmt19_direction(monkeypatch):
    _fake_wmt(monkeypatch, per_direction=31)
    samples = I._load_i18n_translate_samples()
    assert len(samples) == 19 * 31                      # no per-direction cap, no filter
    first = samples[0]
    assert first[0][0]["content"] == ("Translate the following text from Czech into German. "
                                      "Output only the translation:\n\ncs-de seg 0")
    assert first[1] == "ref cs-de 0" and first[2] == {"category": "cs-de", "segment": 0}


def test_i18n_limit_is_spread_across_directions(monkeypatch):
    _fake_wmt(monkeypatch, per_direction=31)
    samples = I._load_i18n_translate_samples(limit=38)
    cats = [s[2]["category"] for s in samples]
    assert len(samples) == 38 and all(cats.count(d) == 2 for d in I.DIRECTIONS)


def test_i18n_subsampling_is_named_only_when_limited(monkeypatch):
    monkeypatch.setattr(I, "_load_i18n_translate_samples", lambda limit=None: [])
    monkeypatch.setattr(I, "run_eval_suite", lambda **kw: {"accuracy": 1.0, "sample_traces": [
        {"response_text": "Hello", "gold_answer": "Hello", "category": d,
         "extra_payload": {"category": d, "segment": 0}} for d in I.DIRECTIONS]})
    full = I.run_i18n_translate("m", "http://x", 4)
    assert full["leaderboard_comparable"] is False
    assert "--eval-limit" not in full["leaderboard_comparable_reason"]
    assert "averages the directions' BLEU" in full["leaderboard_comparable_reason"]
    assert set(full["i18n_bleu"]) == set(I.DIRECTIONS)
    limited = I.run_i18n_translate("m", "http://x", 4, limit=40)
    assert "--eval-limit 40 subset of the 31,387 segments" in limited["leaderboard_comparable_reason"]


def test_i18n_docs_match_the_code():
    doc = _doc("i18n_translate.md")
    assert I.TEST_SET_SHA256 in doc and "31,387" in doc and "sacrebleu==2.6.0" in doc
    assert "12,962" not in doc and "newstest2018" not in doc


# --------------------------------------------------------------------------- #
# docs/evals/toolchains.md
# --------------------------------------------------------------------------- #

def test_toolchains_doc_describes_only_real_scoring_paths():
    doc = _doc("toolchains.md")
    assert "single global text edit distance" not in doc
    assert "pip install apted lxml Distance pylatexenc" not in doc
    assert "`omnidocbench` / `mrcr`" not in doc          # mrcr scores with difflib, not rapidfuzz
    assert "gbench-omnidocbench:193627ae" in doc


# --------------------------------------------------------------------------- #
# Every documented run command passes the CLI-mandatory --max-output-tokens
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ["ruler", "loft_scifact", "i18n_translate", "omnidocbench",
                                  "chartqa", "textvqa", "medxpertqa", "mmmu_pro", "coco_caption",
                                  "bundled_detection", "screenspot_pointing"])
def test_documented_run_commands_pass_max_output_tokens(name):
    lines = _doc(f"{name}.md").splitlines()
    commands = []
    for i, line in enumerate(lines):
        if line.startswith("gbench --evals-only"):
            cmd = [line]
            while cmd[-1].rstrip().endswith("\\") and i + len(cmd) < len(lines):
                cmd.append(lines[i + len(cmd)])      # backslash-continued lines
            commands.append(" ".join(cmd))
    assert commands, name
    for cmd in commands:
        assert "--max-output-tokens" in cmd, (name, cmd)
