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

"""`--attempt-count`: repeated attempts per sample, reported as avg@k / pass@k / pass^k.

Most published numbers for small benchmarks are not single-sample - AIME is avg@4-64,
GPQA-Diamond avg@10, ARC-AGI pass@2 by rule, tau-bench pass^k by definition. Single
greedy sampling is also not reproducible on this server: on 2026-08-17
`astropy__astropy-14309` flipped RESOLVED -> unresolved between two identical
temperature=0.0 runs.

Nothing here touches the network.
"""

import itertools
from unittest import mock

import pytest

from gbench.runners.eval_suites import base


def _reply(text):
    return base.Reply(text=text, tool_calls=None, finish_reason="stop",
                      reasoning=None, error=None, completion_tokens=5, stop_reason=None)


def _run(answers, samples, *, attempt_count=None, async_eval_fn=None,
         supports_attempts=False, eval_fn=None):
    """Drive run_eval_suite with a scripted sequence of model answers."""
    seq = itertools.cycle(answers) if not isinstance(answers, itertools.cycle) else answers
    sent = []

    async def _fake_send(**k):
        sent.append(k)
        return _reply(next(seq))

    with mock.patch.object(base, "_send_single_request", _fake_send):
        res = base.run_eval_suite(
            eval_name="aime", model_name="m", base_url="http://x", concurrency=4,
            samples=samples,
            eval_fn=eval_fn if async_eval_fn is None else None,
            async_eval_fn=async_eval_fn,
            attempt_count=attempt_count, supports_attempts=supports_attempts)
    return res, sent


ONE = [([{"role": "user", "content": "q"}], "yes", {})]
TWO = ONE + [([{"role": "user", "content": "q2"}], "yes", {"category": "b"})]
CORRECT = lambda resp, gold: gold in (resp or "")


# --------------------------------------------------------------------------- #
# default is unchanged
# --------------------------------------------------------------------------- #
def test_default_is_one_attempt_and_changes_nothing():
    res, sent = _run(["yes"], TWO, eval_fn=CORRECT)
    assert len(sent) == 2                 # one request per sample, as before
    assert res["attempts"] is None        # no @k bookkeeping when k == 1
    assert res["total_questions"] == 2 and res["accuracy"] == 100.0


def test_attempt_count_one_is_explicitly_a_no_op():
    res, sent = _run(["yes"], TWO, attempt_count=1, eval_fn=CORRECT)
    assert len(sent) == 2 and res["attempts"] is None


# --------------------------------------------------------------------------- #
# k > 1 on a per-response-scored suite
# --------------------------------------------------------------------------- #
def test_k_attempts_produce_k_generations_per_sample():
    res, sent = _run(["yes"], TWO, attempt_count=3, eval_fn=CORRECT)
    assert len(sent) == 6                                   # 2 samples x 3
    assert res["total_questions"] == 6
    assert res["attempts"]["samples"] == 2
    assert res["attempts"]["generations"] == 6
    assert res["attempts"]["attempts_per_sample"] == 3


def test_accuracy_is_avg_at_k():
    """One sample, 4 attempts, 3 of them right -> avg@4 = 75%."""
    res, _ = _run(["yes", "yes", "no", "yes"], ONE, attempt_count=4, eval_fn=CORRECT)
    a = res["attempts"]
    assert res["accuracy"] == 75.0 and a["avg_at_k"] == 75.0


def test_pass_at_k_is_any_attempt_correct():
    """ARC-AGI's rule: the task scores 1 if ANY attempt matches."""
    res, _ = _run(["no", "no", "yes"], ONE, attempt_count=3, eval_fn=CORRECT)
    a = res["attempts"]
    assert a["pass_at_k"] == 100.0        # one attempt landed
    assert a["avg_at_k"] == pytest.approx(33.33, abs=0.01)


def test_pass_hat_k_requires_every_attempt_correct():
    """tau-bench's reliability metric: all k trials must succeed."""
    res, _ = _run(["yes", "yes", "no"], ONE, attempt_count=3, eval_fn=CORRECT)
    assert res["attempts"]["pass_hat_k"] == 0.0
    res, _ = _run(["yes"], ONE, attempt_count=3, eval_fn=CORRECT)
    assert res["attempts"]["pass_hat_k"] == 100.0


def test_the_three_metrics_are_ordered_and_distinct():
    """pass^k <= avg@k <= pass@k always; a run where they differ is the useful case."""
    res, _ = _run(["yes", "no", "no", "yes"], ONE, attempt_count=4, eval_fn=CORRECT)
    a = res["attempts"]
    assert a["pass_hat_k"] <= a["avg_at_k"] <= a["pass_at_k"]
    assert (a["pass_hat_k"], a["avg_at_k"], a["pass_at_k"]) == (0.0, 50.0, 100.0)


def test_unstable_samples_counts_disagreement_across_attempts():
    """This is the run measuring its own reproducibility - the reason to use @k."""
    res, _ = _run(["yes", "no"], ONE, attempt_count=2, eval_fn=CORRECT)
    assert res["attempts"]["unstable_samples"] == 1
    res, _ = _run(["yes"], ONE, attempt_count=2, eval_fn=CORRECT)
    assert res["attempts"]["unstable_samples"] == 0


def test_traces_identify_which_sample_and_which_attempt():
    res, _ = _run(["yes"], TWO, attempt_count=3, eval_fn=CORRECT)
    traces = res["sample_traces"]
    assert len(traces) == 6
    assert sorted(t["source_sample_idx"] for t in traces) == [0, 0, 0, 1, 1, 1]
    for src in (0, 1):
        got = sorted(t["attempt_idx"] for t in traces if t["source_sample_idx"] == src)
        assert got == [0, 1, 2]


def test_single_attempt_traces_still_carry_the_fields():
    """Downstream analysis must not need to special-case k == 1."""
    res, _ = _run(["yes"], TWO, eval_fn=CORRECT)
    for t in res["sample_traces"]:
        assert t["attempt_idx"] == 0
        assert t["source_sample_idx"] == t["sample_idx"]


# --------------------------------------------------------------------------- #
# batch-scored suites must not fabricate an @k
# --------------------------------------------------------------------------- #
def test_batch_scored_suite_refuses_attempts_unless_it_opts_in():
    """A scorer keyed by instance id would collapse k attempts into one prediction and
    hand the same verdict to all k traces. That is a fabricated @k, so we run 1."""
    async def scorer(traces):
        for t in traces:
            t["is_correct"] = True

    res, sent = _run(["yes"], TWO, attempt_count=5, async_eval_fn=scorer)
    assert len(sent) == 2                                  # NOT 10
    assert res["attempts"]["attempts_per_sample"] == 1
    assert res["attempts"]["attempts_requested"] == 5
    assert "supports_attempts" in res["attempts"]["not_applied"]


def test_batch_scored_suite_can_opt_in():
    async def scorer(traces):
        for t in traces:
            t["is_correct"] = True

    res, sent = _run(["yes"], TWO, attempt_count=5, async_eval_fn=scorer,
                     supports_attempts=True)
    assert len(sent) == 10
    assert res["attempts"]["attempts_per_sample"] == 5


def test_refusal_is_recorded_not_silent():
    """The result must never look like it measured an @k it did not."""
    async def scorer(traces):
        for t in traces:
            t["is_correct"] = True

    res, _ = _run(["yes"], ONE, attempt_count=8, async_eval_fn=scorer)
    assert res["attempts"]["attempts_requested"] == 8
    assert res["attempts"]["attempts_per_sample"] == 1
    assert res["attempts"].get("avg_at_k") is None          # no @k claimed


# --------------------------------------------------------------------------- #
# the knob must not reach the server
# --------------------------------------------------------------------------- #
def test_attempt_count_is_client_side_and_never_sent_to_the_api():
    """Replication is k independent REQUESTS, not the server-side `n` parameter: separate
    requests get separate batch composition, so the samples are genuinely independent."""
    assert "attempt_count" not in base.ALLOWED_API_KEYS if hasattr(base, "ALLOWED_API_KEYS") else True
    res, sent = _run(["yes"], ONE, attempt_count=3, eval_fn=CORRECT)
    for call in sent:
        payload = call.get("payload") or call
        assert "attempt_count" not in str(payload)


def test_temperature_is_recorded_on_the_result():
    """An @k number is uninterpretable without the temperature it was drawn at."""
    res, _ = _run(["yes"], ONE, attempt_count=2, eval_fn=CORRECT)
    assert "temperature" in res


# --------------------------------------------------------------------------- #
# run-knob fallback and CLI wiring
# --------------------------------------------------------------------------- #
def test_run_knob_reaches_suites_that_do_not_forward_it():
    """Same rule as --temperature: a suite cannot lose the knob by omission."""
    base.set_run_knobs(attempt_count=3)
    try:
        res, sent = _run(["yes"], ONE, eval_fn=CORRECT)   # attempt_count NOT passed
        assert len(sent) == 3 and res["attempts"]["attempts_per_sample"] == 3
    finally:
        base._RUN_KNOBS.pop("attempt_count", None)


def test_cli_exposes_the_flag():
    import inspect
    from gbench import cli
    src = inspect.getsource(cli)
    assert '"--attempt-count"' in src
    assert 'dest="attempt_count"' in src


def test_cli_plumbs_it_into_the_config_and_run_knobs():
    import inspect
    from gbench import cli
    from gbench.core import config as cfg
    from gbench.runners import evals
    assert "attempt_count" in inspect.getsource(cfg)
    assert 'config.attempt_count = getattr(args, "attempt_count", 1)' in inspect.getsource(cli)
    assert "attempt_count=kwargs.get(\"attempt_count\")" in inspect.getsource(evals)
