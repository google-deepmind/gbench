# Contributing to gbench

Thank you for your interest in contributing to `gbench`. This suite provides performance benchmarking and capability evaluation for open large language models (LLMs) and vision-language models (VLMs).

We welcome contributions from developers, researchers, hardware partners, and framework maintainers. Please read these guidelines before opening an issue or submitting a pull request.

## 1. Development environment setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/google-gemma/gbench.git
   cd gbench
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install the package in editable mode with what the unit tests need (the same install the
   Unit Tests workflow runs):
   ```bash
   pip install -e ".[dev,evals,omnidocbench]" transformers
   python -m nltk.downloader punkt punkt_tab
   ```
   The `omnidocbench` extra carries the TEDS and formula edit-distance metrics the OmniDocBench
   unit tests exercise. The system tools the tests exercise are bubblewrap, a Java runtime and
   Docker. `transformers`
   (the serving stack's tokenizer library) is not a gbench dependency; torch's CPU build is
   enough for it. Keep this test environment apart from the one that serves your models: the
   `dev` extra's `bfcl-eval` pins `numpy==1.26.4` and `filelock==3.20.0` (see
   [tests/README.md](tests/README.md)).
4. When you change a dependency in `pyproject.toml`, run `uv lock` and commit `uv.lock` with it;
   `tests/test_dependency_policy.py` fails on a stale lock and on a module the code imports that no
   extra declares.

## 2. Code formatting and type checking

Before submitting code changes, verify formatting and type safety:

* Run the linter and auto-formatter:
  ```bash
  ruff check gbench/
  ```
* Run static type checks:
  ```bash
  mypy gbench/
  ```

## 3. Running unit tests and capability verification

All pull requests must pass the existing test suite and preserve core evaluation invariants:

* Run unit and integration tests:
  ```bash
  pytest tests/
  ```
* **CI checks** (`.github/workflows/`): **Unit Tests** (the install above, then pytest with the
  `ci_skip_gate` plugin, which fails the run when a test skipped because a declared dependency is
  missing: a `pytest.importorskip` of it, or a skip whose reason says "No module named" it; a skip
  with a reason of its own is not seen by the gate), **Leak Gate** (no internal names or credentials, and no tracked backup files),
  **Golden Changelog** (a Golden Set change comes with its changelog entry) and **Canonical Sync
  Gate** (`canonical_sync.py --check-bump`: a scoring-code change needs a moved anchor or a
  `VERDICT_PRESERVING` declaration, then `python gbench/runners/eval_suites/canonical_sync.py
  --refresh-code-fp`).
* **Definition of done for a change to an eval suite or a shared eval module**: reconcile it
  against upstream; when a verdict can change, bump the suite's `canonical_sync` entry
  (`synced`, and `revision` / `checksum` / `dataset` when the pin moved; a shared module's entry
  is in `SHARED_SYNC`), else declare the change in `VERDICT_PRESERVING` with the new code
  fingerprint and the reason; run `python gbench/runners/eval_suites/canonical_sync.py
  --refresh-code-fp` (the refresh refuses, and writes nothing, while a moved fingerprint has
  neither; when the entry was already bumped the same day, use a serial such as `2026-09-28.2`);
  update `docs/evals/<suite>.md`; run `tests/test_canonical_sync.py` and
  `tests/test_scaffold.py`. A new environment knob is declared run-shaping
  (`base.RUN_SHAPING_SUITE_ENV`) or not (`base.NON_RUN_SHAPING_ENV`, with its reason). See
  [docs/scaffold-versioning.md](docs/scaffold-versioning.md).
* **Golden Set**: the pull-request gate is the unit-test suite, not a Golden Set run. The Golden
  Set measures a model (a healthy model passes every case that applies to it) and CI never runs it
  (see [tests/README.md](tests/README.md)). A change to it must not weaken the two-sided presence
  matchers or bypass the unit-test execution assertions in `gbench/runners/golden.py`, and a change
  to its cases comes with its changelog entry (the Golden Changelog check).
* **Statistical repeatability**: a change to runner execution loops or timing logic must not add
  run-to-run variance of its own; the repeatability check (`--max-cv-percent`, default 5.0) judges
  every multi-iteration performance run.

## 4. Documentation style

Documentation (the `README` and everything under `docs/`) is written to read as human authored:

* **Plain ASCII punctuation.** No em dashes and no en dashes. Use a comma, parentheses, a colon, or
  two sentences instead; for a numeric or token range use a hyphen (for example `4k-128k`).
* **No LaTeX in prose.** Write `O(n)` and `x >= 0` in plain text; keep math notation only where it is
  a benchmark's actual data.
* **Plain vocabulary.** Avoid machine-tell filler (delve, leverage, seamless, boasts, realm,
  tapestry, testament, and the like). Say what a thing does directly.
* **Code identifiers stay in backticks.** Do not spell `snake_case` or `camelCase` names as prose
  words; wrap real identifiers in backticks and use ordinary words otherwise.
* **Accurate policy language.** A suite that lacks a prerequisite hard-errors (`infra_required`)
  rather than emitting a fabricated score; it does not report a `skipped` status (a code-execution
  suite without a working sandbox hard-errors too). Do not describe behavior the suites no longer
  have.

The same rules apply to code: docstrings, comments and printed strings use plain ASCII
punctuation too (vendored upstream code is left as upstream wrote it), and the plain-vocabulary and
accurate-policy points apply there as well.

## 5. Pull request guidelines

* **Keep diffs small and focused**: Avoid combining formatting cleanups with functional logic changes. Small, atomic pull requests are easier to review and merge quickly.
* **Provide clear reproducer commands**: When reporting a bug or submitting a fix, include the exact command line arguments used so reviewers can reproduce the behavior.
* **Add Apache 2.0 copyright headers**: All new source and configuration files (Python, Shell, Docker, JavaScript) must include the standard Apache 2.0 Google LLC copyright notice at the top of the file.
* **Update documentation**: If you add new CLI arguments, dataset providers, or evaluation metrics, update `README.md` and the relevant documentation files in `docs/`.

## 6. License and contributor license agreement

By contributing to `gbench`, you agree that your contributions will be licensed under the Apache License, Version 2.0.

### Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License Agreement (CLA). You (or your employer) retain the copyright to your contribution; this simply gives us permission to use and redistribute your contributions as part of the project. Head over to <https://cla.developers.google.com/> to see your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one (even if it was for a different project), you probably don't need to do it again.

### Code reviews

All submissions, including submissions by project members, require review. We use GitHub pull requests for this purpose. Consult [GitHub Help](https://help.github.com/articles/about-pull-requests/) for more information on using pull requests.

## 7. Community guidelines

This project follows [Google's Open Source Community Guidelines](https://opensource.google/conduct/). See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
