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

"""Canonical API-Bank: execution-based correctness against the checkout backend."""
import json
import os
import warnings

warnings.filterwarnings("ignore")

import pytest
import gbench.runners.eval_suites.api_bank as A

_CO = os.environ.get("GBENCH_APIBANK_DIR")
_HAVE = bool(_CO and os.path.isfile(os.path.join(_CO, "tool_manager.py")))
needs_checkout = pytest.mark.skipif(not _HAVE, reason="GBENCH_APIBANK_DIR checkout not present")


def test_api_bank_hard_errors_without_checkout(monkeypatch):
    # No-skip policy: absent backend -> hard-error with the clone command, not a silent skip.
    monkeypatch.delenv("GBENCH_APIBANK_DIR", raising=False)
    with pytest.raises(RuntimeError, match="docs/evals/api_bank.md"):
        A._checkout_dir()


def test_build_messages_maps_roles_as_evaluator_py():
    ch = [
        {"role": "User", "text": "hi"},
        {"role": "AI", "text": "ok"},
        {"role": "API", "api_name": "X", "param_dict": {"a": "1"}, "result": {"output": "R"}},
    ]
    msgs = A._build_messages("SYS", ch)
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert msgs[1] == {"role": "user", "content": "hi"}
    assert msgs[2] == {"role": "assistant", "content": "ok"}
    # evaluator.py sends an earlier API result as a system message
    assert msgs[3] == {"role": "system", "content": "[X(a='1')] Response: R"}


def test_build_messages_keeps_consecutive_same_role_turns_apart():
    ch = [{"role": "User", "text": "a"}, {"role": "User", "text": "b"}]
    msgs = A._build_messages("SYS", ch)
    assert msgs == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "a"},
                    {"role": "user", "content": "b"}]


@needs_checkout
def test_l1_execution_scores_gold_and_rejects_garbage():
    """Feeding the GOLD call reproduces the gold result via real backend execution -> mostly
    correct (some APIs have strict checks, canonically counted wrong); garbage -> wrong."""
    d = A._checkout_dir()
    api_traces, resp_trace = [], None
    with A._in_checkout(d):
        import evaluator as ev
        ddir = os.path.join("lv1-lv2-samples", "level-1-given-desc")
        files = sorted(f for f in os.listdir(ddir) if f.endswith(".jsonl"))[:4]
        for fname in files:
            with open(os.path.join(ddir, fname), encoding="utf-8") as f:
                hist = [json.loads(line) for line in f]
            for sid, s in enumerate(ev.Sample.from_chat_history(hist)):
                gt = s.ground_truth
                if gt.get("role") == "API":
                    params = ", ".join("{}='{}'".format(k, v) for k, v in (gt.get("param_dict") or {}).items())
                    call = "[{}({})]".format(gt.get("api_name"), params)
                    api_traces.append({"gold_answer": json.dumps(
                        {"level": 1, "track": "api", "file": fname, "sample_id": sid}),
                        "response_text": call})
                elif gt.get("role") == "AI" and resp_trace is None:
                    resp_trace = {"gold_answer": json.dumps(
                        {"level": 1, "track": "response", "file": fname, "sample_id": sid}),
                        "response_text": gt.get("text", "")}

    assert api_traces, "expected some L1 API steps"
    garbage = {"gold_answer": api_traces[0]["gold_answer"], "response_text": "no api call here"}
    A._score_traces(d, api_traces + [garbage] + ([resp_trace] if resp_trace else []))

    correct = sum(1 for t in api_traces if t.get("is_correct"))
    assert correct / len(api_traces) >= 0.5, f"gold calls should mostly execute-correct ({correct}/{len(api_traces)})"
    assert garbage["is_correct"] is False  # no parseable call -> wrong
    if resp_trace:
        assert resp_trace["api_bank_score"] >= 0.99  # identical text -> ROUGE-L ~1.0
