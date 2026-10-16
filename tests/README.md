# Unit tests

Everything in this directory tests **gbench itself**.

Three things live in the repo and are easy to confuse, so they are kept
apart on purpose:

| | Unit tests (`tests/`) | Evals (`gbench/golden_dataset/`) | Template checks (`tools/template_checks/`) |
|---|---|---|---|
| Subject under test | the gbench code | whatever model you point gbench at | a Gemma 4 `chat_template.jinja` artifact |
| Run by | `pytest` | `gbench --golden-only --remote-endpoint <url>/v1 --tokenizer <model>` | `GEMMA4_TEMPLATE_PATH=... pytest tools/template_checks` |
| Needs a model? | no | yes | no, just the template file |
| Deterministic? | yes, always | no, that is the point | yes |
| Gates merges? | yes, via CI | no | no: it is not part of gbench's test gate |
| Ships in the wheel? | no | yes | no |

## Running them

Install gbench with its declared dependencies and the `dev`, `evals` and
`omnidocbench` extras (`omnidocbench` holds the TEDS and edit-distance
libraries of the in-process omnidocbench helpers the tests exercise), plus
the one library the tests import that gbench deliberately does not declare
(transformers, the serving stack's tokenizer library), and the nltk
tokenizer data the ifeval and causalbench scorers read:

```bash
pip install -e ".[dev,evals,omnidocbench]" transformers
python -m nltk.downloader punkt punkt_tab
pytest
```

Do this in a test environment, not in the environment that serves your
models. The `dev` extra carries `bfcl-eval==2026.3.23` (the release the
gbench-bfcl harness image holds) only for the BFCL unit tests, which run
the image's checker functions in-process; bfcl-eval pins
`numpy==1.26.4` and `filelock==3.20.0`, so installing it next to vLLM
would move those packages. When the bfcl-eval the tests import is not
the pinned release, the session summary says so (`tests/conftest.py`),
since those tests then checked another release.

The harnesses whose own pins conflict with gbench's run in local images.
tau2-bench (tau2, tau3), Harbor 0.20.0 (terminal_bench) and semgrep 1.51.0
with Meta's Insecure Code Detector (cyberseceval) are not installed by any
extra. The BFCL suites run bfcl-eval in the local gbench-bfcl image; the
dev extra installs the same release only for the unit tests that run its
checker functions in-process. The unit tests do not need these images:
they stub the container calls. A few tests run a real
image and are opt-in, so they skip where it is not built (CI builds none):

* `GBENCH_TEST_BFCL_IMAGE=1`: the BFCL harness image
  (`gbench-bfcl:2026.3.23`) against local stub servers;
* `GBENCH_TEST_LEAN_IMAGE=<tag>`: the putnam_formal Lean kernel tests;
* the gbench-matharena:b89f2f0a, gbench-multipl-e-src:3025a531 and
  gbench-livebench:0baa4aa7 tests run when that image is present.

The quality pillar's isolation tests build a small probe image of their
own (`FROM python:3.12-slim`), so they need a Docker daemon that can pull
that base image.

Some tests also run system tools: bubblewrap (`bwrap`, allowed to create
unprivileged user namespaces) for acebench's isolated evaluation and the
code sandbox, a Java runtime for pycocoevalcap, and Docker for the
container suites' guards.

This is exactly what CI does (`.github/workflows/test.yml`, on Python
3.11 and 3.12), except that CI installs torch's CPU build first so the
install stays small.

No GPU, no model weights and no model server are needed. The Golden Set
tests stand up a fake in-process OpenAI-compatible endpoint with scripted
answers, so they assert that the runner *scores* correctly rather than
that a model *answers* correctly. Tests may not contact a local model
server: `tests/conftest.py` refuses loopback connections to ports
8000-8010, and the model registry's HuggingFace lookups are stubbed for
the whole session (the repository-level `conftest.py`), so no test
registers a model against the real Hub.

Network: most tests are hermetic, but some still fetch pinned upstream
files and will fail without network access. Run offline (for example in a
network namespace with an empty `HOME`), these fail:

* multipl_e's scoring tests (`test_multipl_e_assembly.py`, the `test_mpe_*`
  tests in `test_code_exec_suites_canonical.py`, and one each in
  `test_builtin_scoring_regressions.py` and
  `test_code_exec_upstream_extraction_and_judging.py`): the suite downloads MultiPL-E's pinned `chat_completions.py` from GitHub
  (sha256-checked) into `GBENCH_CACHE_DIR`;
* dataset loaders reached through the Hub: `test_amc_aime`
  (`test_eval_suites_hybrid.py`), `test_infographicvqa_eval_fn`
  (`test_eval_suites_extended.py`) and the MathArena config tests of
  `test_math_code_token_limit_policy.py`;
* `test_complexfuncbench.py::test_embedder_reproduction_and_multicall_alignment`
  (the embedding model).

Making these hermetic (a vendored copy of each pinned file served by a
URL-scoped fake, or a separate integration job) is tracked work; until it
lands, CI runs with network access. Without the nltk data above, the
ifeval and causalbench tests fail with a LookupError naming `punkt`.

CI also fails a run in which any test skipped because a dependency that
`pyproject.toml` declares is missing (`scripts/ci_skip_gate.py`, loaded
with `PYTHONPATH=scripts pytest -p ci_skip_gate --skip-gate-extras
dev,evals,omnidocbench`): such a skip means the install is incomplete and
the test protects nothing. The gate reads `pytest.importorskip` skips and
"No module named" errors, so a test skips a missing package only through
`pytest.importorskip(<module>)`, never with a message of its own
(`tests/test_cluster_scripts_and_ci_helpers.py` checks it). Skips from modules nothing declares (a harness that
runs in a suite's own image, such as tau2) are listed but do not fail the
run; so are skips from a requirement whose environment marker is false on
the running interpreter, since pip installs nothing for it there.
`tests/test_dependency_policy.py` keeps the declarations honest: every
third-party module gbench's host-side code imports is declared or listed
there with the reason it is not, `uv.lock` records exactly what
`pyproject.toml` declares, and one resolve holds every extra CI installs.

## Layout

Tests are named after the module or suite they cover
(`test_<module>.py`, `test_<suite>*.py`, `test_cli*.py`,
`test_scaffold*.py`, ...); a file that covers several suites or modules
names them, or the behaviour they share. A new test for a module goes in
that module's own file. No file or test is named after the batch of
changes that added it (`tests/test_repo_gates_and_shared_docs.py` checks
it). Registry tests are in `test_models.py`,
the sizing helpers of `gbench/core/config.py` in `test_config.py`, and
the cluster scripts, the GKE manifest and the CI helpers in
`test_cluster_scripts_and_ci_helpers.py`.

Every test file starts with the Apache license header
(`tests/test_run_directories_merge_and_license_headers.py::test_every_test_file_carries_the_license_header`
checks it; the maintainer's plugin tests are not checked). The shipping
code under `gbench/`, `scripts/` and `examples/` carries it too
(`tests/test_repo_gates_and_shared_docs.py`); vendored code keeps its upstream license.

## Other CI gates

Separate workflows, each its own named check:

* `leak-gate.yml`: no internal identifier or secret pattern in a tracked
  file (`scripts/check_no_leaks.sh`), and no tracked backup file
  (`*.bak`, `*.bak_*`, `*.bak-*`, `*.orig`).
* `golden-changelog.yml`: a changed Golden Set case must say why
  (`scripts/check_golden_changelog.py`).
* `canonical-sync-gate.yml`: a change to an eval's scoring code must come
  with a recorded decision in `gbench/runners/eval_suites/canonical_sync.py`
  (`--check-bump`).

## Why the split matters

`gbench/golden_dataset/` is an eval: a deterministic smoke test over
basic capabilities, shipped so other repos can point `--golden-only` at
their own endpoint while developing a model. Its pass rate is a fact
about that model on that day, and it is expected to go red sometimes.
That is a signal to the team developing the model, not a reason to
block a gbench merge.

CI must therefore never run the eval. If it did, a bad checkpoint
somewhere else would turn this repo's `main` red, and the eval's real
audience would learn to ignore it.

The unit tests are the opposite: they must be green on every pull
request. `tests/test_golden.py` in particular carries the regression
guard for the defect that motivated this split. An adversarial fixture
that returns factually inverted and actively unsafe answers once scored
6/6 at 100% under substring matching. `test_adversarial_responses_all_fail`
now pins every one of those to `failed`.

## The chat-template checks

`tools/template_checks/test_chat_template_gemma4.py` checks one model's artifact (a Gemma 4
`chat_template.jinja`), not gbench, so it lives outside this directory. A plain `pytest`
collects `tests/` only (`testpaths`) and never runs it.
[`tools/template_checks/README.md`](../tools/template_checks/README.md) says what it checks and
how to run it. The Golden Set's own chat-template leak check is model-agnostic. It is
unit-tested here with fake tokenizers shaped like a Gemma 4, a Gemma 3 and a Llama 3 tokenizer
(`tests/test_golden.py`, `tests/test_golden_chat_template_leak.py`); none of those tests loads a
real tokenizer.
