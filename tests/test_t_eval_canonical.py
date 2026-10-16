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

"""Canonical T-Eval: 6 dimensions scored by the vendored upstream evaluators."""
import hashlib
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import gbench.runners.eval_suites.t_eval as T


class _FakeST:
    """Deterministic stand-in for all-mpnet-base-v2: identical text -> cosine 1."""
    def encode(self, texts, convert_to_tensor=True):
        import torch

        def vec(s):
            h = np.frombuffer(hashlib.md5(str(s).encode()).digest(), dtype=np.uint8).astype("float32")
            t = torch.tensor(h)
            return t / (t.norm() + 1e-9)

        if isinstance(texts, list):
            return torch.stack([vec(t) for t in texts])
        return vec(texts)


def _trace(sample: dict, response: str) -> dict:
    return {"gold_answer": json.dumps({"sample": sample}), "response_text": response}


def _use_fake_st(monkeypatch):
    monkeypatch.setattr(T, "_shared_st", lambda language="en": _FakeST())


def test_to_chat_maps_function_to_user_one_message_per_turn():
    op = [{"role": "system", "content": "S"}, {"role": "user", "content": "U1"},
          {"role": "function", "content": "F"}, {"role": "user", "content": "U2"},
          {"role": "assistant", "content": "A"}]
    msgs = T._to_chat(op)
    # function->user (its api_role in upstream's meta templates); every turn stays its own message
    assert [m["role"] for m in msgs] == ["system", "user", "user", "user", "assistant"]
    assert [m["content"] for m in msgs] == ["S", "U1", "F", "U2", "A"]


def test_instruct_scoring_json_correct():
    sample = {"template": {"thought": "goal", "action": "name", "args": "args"},
              "meta_data": {"response_format": "json"},
              "ground_truth": {"action": "FilmDouban.print_detail", "args": {"film_name": "x"}}}
    pred = '{"goal": "t", "name": "FilmDouban.print_detail", "args": {"film_name": "x"}}'
    agg = T._score_file("instruct", [_trace(sample, pred)])
    assert agg["json_format_metric"] == 1.0 and agg["json_args_em_metric"] == 1.0


def test_instruct_scoring_wrong_args():
    sample = {"template": {"thought": "goal", "action": "name", "args": "args"},
              "meta_data": {"response_format": "json"},
              "ground_truth": {"action": "A.b", "args": {"k": "v"}}}
    pred = '{"goal": "t", "name": "A.b", "args": {"k": "WRONG"}}'
    agg = T._score_file("instruct", [_trace(sample, pred)])
    assert agg["json_format_metric"] == 1.0
    assert agg["json_args_em_metric"] == 0.5  # action matches (+1), arg wrong; 1/(1+1)


def test_retrieve_str_name_match():
    sample = {"meta_data": {"response_format": "str"},
              "ground_truth": {"thought": "t", "name": "FileOperation.read", "args": "{}"}}
    agg = T._score_file("retrieve_str", [_trace(sample, "FileOperation.read")])
    assert agg["name"] == 1.0


def test_understand_str_args_exact():
    sample = {"meta_data": {"response_format": "str"},
              "ground_truth": {"args": "researchteam@example.com"}}
    agg_ok = T._score_file("understand_str", [_trace(sample, "researchteam@example.com")])
    agg_no = T._score_file("understand_str", [_trace(sample, "other@example.com")])
    assert agg_ok["args"] == 1.0 and agg_no["args"] == 0.0


def test_review_str_letter_match():
    sample = {"template": "", "meta_data": {"response_format": "str"},
              "ground_truth": {"answer": "B", "thought": "t"}}
    agg_ok = T._score_file("review_str", [_trace(sample, "Answer: B")])
    agg_no = T._score_file("review_str", [_trace(sample, "Answer: C")])
    assert agg_ok["review_quality"] == 1.0 and agg_no["review_quality"] == 0.0


def test_reason_str_thought_cosine(monkeypatch):
    _use_fake_st(monkeypatch)
    sample = {"meta_data": {"response_format": "str"},
              "ground_truth": {"thought": "read the movies file"}}
    agg = T._score_file("reason_str", [_trace(sample, "read the movies file")])
    assert agg["thought"] >= 0.99  # identical text -> cosine 1


def test_plan_json_f1_identical(monkeypatch):
    _use_fake_st(monkeypatch)
    plan = [{"id": 0, "prev": [], "name": "A.search", "args": {"q": "x"}},
            {"id": 1, "prev": [0], "name": "A.detail", "args": {"id": "1"}}]
    sample = {"meta": {"prompt_type": "json", "API_list": ["A.search", "A.detail"]},
              "ground_truth": plan}
    agg = T._score_file("plan_json", [_trace(sample, json.dumps(plan))])
    assert agg["f1_score"] >= 0.99  # identical plan -> perfect match


def test_rru_json_all_three_metrics(monkeypatch):
    _use_fake_st(monkeypatch)
    gt = {"thought": "get reviews", "name": "Airbnb.reviews", "args": {"id": "II-7"}}
    sample = {"meta_data": {"response_format": "json"}, "ground_truth": gt}
    agg = T._score_file("rru_json", [_trace(sample, json.dumps(gt))])
    # canonical args = matched/(len(gt_args)+1e-5) -> ~0.99999 for a perfect 1-key match
    assert agg["thought"] >= 0.99 and agg["name"] == 1.0 and agg["args"] >= 0.999


def test_safe_mean_drops_nan():
    assert T._safe_mean([float("nan"), 1.0]) == 1.0
    assert T._safe_mean([float("nan")]) == 0.0
    assert abs(T._safe_mean([0.5, 1.0]) - 0.75) < 1e-9
