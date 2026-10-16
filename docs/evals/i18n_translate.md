# i18n_translate

Multilingual machine translation on the official **WMT19 news translation test sets**
(newstest2019), in all 19 WMT19 directions, into and out of English and between other languages,
scored with **sacreBLEU**, the scorer WMT reports automatic scores with.

## Dataset (pinned)

| What | Pin |
| --- | --- |
| sacrebleu test set `wmt19` = `https://data.statmt.org/wmt19/translation-task/test.tgz` (the WMT19 news test sets) | sha256 `5ef66d76e183ed40450086ad6fa7beccda1fbb290c5556b93273d515fd6683fb`, checked by the suite (the md5 sacrebleu lists for it is `84de7162d158e28403103b01aeefc39a`) |
| Scorer | `sacrebleu==2.6.0` |

The 19 directions and their segments (one reference each): cs-de 1,997, de-cs 1,997, de-en
2,000, de-fr 1,701, en-cs 1,997, en-de 1,997, en-fi 1,997, en-gu 998, en-kk 998, en-lt 998,
en-ru 1,997, en-zh 1,997, fi-en 1,996, fr-de 1,701, gu-en 1,016, kk-en 1,000, lt-en 1,000,
ru-en 2,000, zh-en 2,000: **31,387 segments**. A default run translates every segment; nothing is
filtered out. The suite downloads the tarball once to
`$GBENCH_CACHE_DIR/sacrebleu/wmt19/raw/wmt19.test.tgz` (default `~/.cache/gbench/...` when
`GBENCH_CACHE_DIR` is unset), with a 120 s read timeout, into a `.part` file that becomes the
tarball only when its sha256 matches; it then extracts the SGML files next to it, and sacrebleu
turns them into the text files it reads the segments from. sacrebleu's own downloader, which
ends the process (`sys.exit`) on a TLS error or a checksum mismatch, is never asked to download.
- A download that fails (network, TLS, HTTP) is a hard error (`infra_required`) for this suite
  only; the rest of a `--evals` run continues.
- A download with another sha256 raises (the file stays as `wmt19.test.tgz.part`); a cached
  tarball with another sha256 (for example one cut short by an interrupted download) is
  downloaded again.
- The download, the extraction and the text preparation run under a file lock,
  `$GBENCH_CACHE_DIR/sacrebleu/wmt19/.gbench.lock` (portalocker, which sacrebleu depends on), so
  processes that share the cache on a first run (for example `GBENCH_SHARD` shards) wait for one
  another instead of racing on the same files; a process that waits more than 30 minutes raises.
- An extraction is marked complete (`raw/.extracted-<sha256 prefix>`) only after it finishes, so a
  tarball that was downloaded but never extracted, or whose extraction was cut short, is extracted
  again. A text file cut short is rebuilt from the SGML before its segments are counted; when the
  count is still wrong (an SGML file cut short), the tarball is extracted again and the text
  rebuilt once more, and only then does the count check raise.
- A direction list other than the 19, or a segment count other than the table, raises.

`--eval-limit N` takes a seeded subset of N segments spread evenly across the directions.

## Protocol

- **Prompt** (gbench's own; WMT defines no prompt for a language model), one user turn per
  segment: `Translate the following text from <Source> into <Target>. Output only the
  translation:\n\n<source segment>`, with the language names Czech, German, English, Finnish,
  French, Gujarati, Kazakh, Lithuanian, Russian and Chinese.
- **Decoding:** gbench's temperature rule (0.0 greedy for a no-think run, 1.0 with
  `--eval-thinking`; `--temperature` or `GBENCH_I18N_TRANSLATE_TEMPERATURE` replaces it); WMT
  defines none.
- **Post-processing:** none beyond removing surrounding whitespace. The reply is scored as the
  system output (quotes are kept: many references are quoted sentences).

## Scoring

Per direction, over all of its segments in test-set order (sacrebleu 2.6.0):
- corpus **BLEU**: `sacrebleu.metrics.BLEU(trg_lang=<target>)`, i.e. the target language's
  default tokenizer (`zh` for Chinese, `13a` otherwise), mixed case, exponential smoothing
  (signature `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`, `tok:zh` for en-zh);
- corpus **chrF**: `sacrebleu.metrics.CHRF()` defaults (chrF2; signature
  `nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0`).

An empty generation stays in its direction's corpus as an empty hypothesis, so a model that
fails on hard sentences cannot inflate a score. A request that got no reply from the served
endpoint (failed, timed out, or over the context window) is a harness error, never the model's
translation: its segment is left out of the corpus (and of `pass_rate_chrf40`), counted, and
named in `status_reason`; when no segment is scored, `accuracy` is null with
`accuracy_withheld_reason`. Checked: the per-direction values equal `sacrebleu.corpus_bleu` /
`sacrebleu.corpus_chrf` on the same hypotheses for all 19 directions.

## Reading the result

- `accuracy` (headline) = the **mean of the 19 directions' BLEU**, taken over the exact
  (unrounded) corpus scores and rounded once. WMT19 reports each direction on its own; the
  average is gbench's single number. `headline_metrics` holds it with the mean chrF.
- `i18n_bleu` / `i18n_chrf`: BLEU and chrF per direction (the values comparable with WMT19
  systems' sacreBLEU scores); `mean_chrf`: their chrF mean (over the exact scores, rounded
  once); `i18n_directions`: per direction the BLEU and chrF (2 decimals, and `bleu_exact` /
  `chrf_exact` unrounded), segment count and both sacrebleu signatures; `sacrebleu_version`.
- `category_accuracy`: per direction its corpus BLEU (`total` = its segments), so the summary's
  rows carry the headline metric.
- `pass_rate_chrf40`: share of segments at sentence chrF >= 40 (a secondary; per direction in
  `category_pass_rate_chrf40`, the count in `segments_chrf40`; `correct_answers` is `null`, with
  `correct_answers_note`), and a per-trace sentence `chrf` for inspection.
- `leaderboard_comparable` is always `false`, and `leaderboard_comparable_reason` says why: the
  headline averages the directions (no WMT number is such an average) and the prompt is
  gbench's; it also names an `--eval-limit` subset, a `GBENCH_SHARD` shard, directions missing
  from the scored traces, a sacrebleu release other than 2.6.0, requests that got no reply
  (failed, timed out or over the context window; harness errors, left out of the score), and
  replies cut
  off by the run's token budget rather than at an official cap (`replies_cut_off_by_budget`;
  `replies_cut_off_in_reasoning` counts those that ran out inside the reasoning channel with no
  answer).

## Requirements

- A running OpenAI-compatible `/v1` endpoint.
- `sacrebleu==2.6.0` (declared in the `evals` extra: `pip install -e '.[evals]'`); without it
  the suite hard-errors (`infra_required`) before generating.
- Network access to `data.statmt.org` on the first run (the test set, about 3.5 MB), then the cache.
- No API key and no judge.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals i18n_translate \
       --max-output-tokens 65536
```
Add `--eval-limit 95` for a quick smoke run (5 segments per direction).

The run logs its planned workload in one line at its start (for example `[i18n_translate]
--eval-limit 5: 5 sample(s) x 1 attempt = 5 generations`) and records it in the result as
`eval_limit_plan` (`limit`, `samples`, `per_setting`, `attempts_per_sample`, `generations`).

## Sharding

`--shard I/N` (or `GBENCH_SHARD=I/N`) runs shard I of N of the suite's full, ordered segments
(direction by direction) (round-robin: positions I, I+N, I+2N, ...), selected before the
`--eval-limit` draw, which is then taken within the shard (N shards run up to N x `--eval-limit`
segments). The result records `shard`, `shard_index` and `shard_count`, and a shard of more than
one is not leaderboard-comparable (the shard reason is named in `leaderboard_comparable_reason`);
`gbench --merge` combines the shards' results as docs/distributed-gbench.md describes: it
re-counts the union of their traces, and refuses a suite whose headline is not a correct / scored
count over them unless the suite defines its own merge.
