# Vendored SciCode data

`13.6.txt`, `62.1.txt`, `76.3.txt` are copied verbatim from
[scicode-bench/SciCode](https://github.com/scicode-bench/SciCode)
(`eval/data/`), licensed Apache-2.0.

Canonical SciCode does **not** ask the model to generate these three sub-steps
(problem 13 step 6, problem 62 step 1, problem 76 step 3): each depends on
prior human-authored scaffolding that the benchmark supplies directly. The
reference runner injects these snippets as the "previous step" context for the
*later* steps of those problems and excludes them from scoring (see
`gencode.py` / `test_generated_code.py` upstream). gbench mirrors that behavior,
so these files are shipped as package data.
