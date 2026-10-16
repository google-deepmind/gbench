# Chat-template checks

`test_chat_template_gemma4.py` checks one model artifact: a Gemma 4
`chat_template.jinja`. It is not a gbench test. Its pass rate is a fact
about the template file you point it at, not about gbench's code, so it
lives here, outside `tests/`, and is never part of gbench's unit-test
gate. A plain `pytest` run collects `tests/` only (`testpaths` in
`pyproject.toml`), so it never picks this directory up.

## What it checks

Gemma 4 chat-template conformance. The module renders the template
locally with Jinja (a sandboxed environment) and compares what it emits
with `google-deepmind/dialog`, the training-time serialization of Gemma
conversations: turn wrappers and BOS, content trimming and turn merging
(system and consecutive turns), content ordering, tool calls, tool
responses and tool declarations, thinking and reasoning channels,
generation prompts, multimodal placeholders, and history robustness
(turn closure, continuation prefill). It needs no GPU, no model
weights and no endpoint, so it can check a change to a template before
any model exists to serve it.

Calibration: of the 48 checks, 29 assert behaviour the shipped template
already has. The other 19 assert dialog-correct behaviour it does not,
and are marked `xfail(strict=True)`. A template that closes one of those
gaps reports an XPASS, which fails the run until the stale marker is
removed, so the markers cannot quietly go out of date. Every reason
string quotes what the template actually emitted. The calibration was
recorded against `google/gemma-4-31B-it` and `google/gemma-4-E4B-it`,
which give identical results.

The Golden Set (`gbench --golden-only`) covers a different question for
any model family: its `chat_template` cases send a served endpoint the
structures a template has to serialize and fail on a leaked
chat-template token of that model. That says a served template is
broken; this module says which part of a Gemma 4 template diverges from
the training-time format.

## How to run it

It never downloads anything. With no template named, every check skips.
Point `GEMMA4_TEMPLATE_PATH` at a template file and run it from the
repository root:

```bash
curl -L -o /tmp/gemma4.jinja \
  https://huggingface.co/google/gemma-4-31B-it/resolve/main/chat_template.jinja
GEMMA4_TEMPLATE_PATH=/tmp/gemma4.jinja pytest tools/template_checks
```

A healthy run of the shipped template ends in `29 passed, 19 xfailed`.
Point `GEMMA4_TEMPLATE_PATH` at a candidate template to check a fix
before it ships.

Prerequisites: gbench installed with the `dev` extra
(`pip install -e ".[dev]"`), which provides `pytest` and `jinja2`. The
module itself imports nothing from gbench, but the repository's root
`conftest.py` loads when you run it from the root, and that needs the
gbench install.

## Where it came from

The template half of the vLLM Gemma 4 reproduction suite
(https://gist.github.com/lucianommartins/c449b600647d207408a799a42870a561).
The rest of that suite drives vLLM parser internals (the Gemma 4
reasoning and tool parsers) and belongs next to that code. The upstream
suite describes its calibration in prose but ships without the xfail
markers, so it is red out of the box; the markers here are the
difference. The module used to sit at `tests/test_chat_template_gemma4.py`
and moved here unchanged apart from its run instructions.
