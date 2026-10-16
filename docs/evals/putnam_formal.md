# putnam_formal setup

PutnamBench's Lean 4 track: William Lowell Putnam Competition problems (1962-2025) formalized
in Lean 4, where a problem counts as solved only if the Lean kernel accepts the model's proof.
Both Lean settings of the [PutnamBench leaderboard](https://trishullab.github.io/PutnamBench/leaderboard.html)
are run and reported by default. For the informal, judge-graded track see [putnam](putnam.md).
SANDBOX_EVAL.

## Dataset and toolchain (pinned)
- **Statements**: `trishullab/PutnamBench` at commit
  `97b5e9458b04e3d1044e8f8afe1724ff133f5a53` (main on 2026-09-24), directory `lean4/src`:
  672 files, one theorem each, the count the leaderboard uses. 346 of them have an answer
  placeholder. Tags come from `informal/putnam.json` of the same commit.
- **Download**: the GitHub tarball of that commit (about 1 MB) on first use, cached under
  `$GBENCH_CACHE_DIR/putnambench/<commit>/` (default `~/.cache/gbench`). Every load checks that
  `lean4/src` hashes to its git tree id `f10ab3f77620e127ca58cae55924f6449da403ce` and
  `informal/putnam.json` to its git blob id `77675840ebadc2bc9b5a93f390bb0a678f33dbbc`, so the
  statements are exactly the release's (a modified cache is refused).
- **Lean and Mathlib**: what that commit pins. `lean4/lean-toolchain` is
  `leanprover/lean4:v4.33.1`; `lean4/lakefile.lean` requires Mathlib `v4.33.1`, which
  `lean4/lake-manifest.json` locks to `0df444a360eaa60ab8c11dca51a86af692955474`. The package
  option `autoImplicit false` is applied to every compile (`-DautoImplicit=false`).
  PutnamBench's CI built all 672 statements and the answer-inserted variant at this commit, and
  the image build below repeats that, so no statement is unverifiable under this toolchain.

## The two settings
Every problem is run in both settings, so a full run is 1,344 generations (672 x 2).

- **Lean with answer** (results key `lean-wsolution`): answer placeholders are pre-filled with
  the reference answer. The statement is exactly what the release's
  `lean4/scripts/rewrite_solutions.py` produces (gbench ports it verbatim): the
  `abbrev <name>_solution : T := sorry` line is dropped and every use of the placeholder is
  replaced by `((<answer>) : T)`, where `<answer>` is the `-- ...` comment on the next line.
- **Lean no answer** (results key `lean-nosolution`): the model sees the release file with the
  answer comment removed, and must replace both `sorry`s: the placeholder's value (the answer)
  and the proof. This is the paper's Task 1 ("first identify the (closed-form) solution, and then
  provide a proof of correctness"). The supplied answer must also be faithful (below).

The 326 problems without a placeholder are the same task in both settings; each setting still
runs all 672 problems, as each leaderboard column counts them.

**Prompt.** PutnamBench does not publish a complete generation prompt (the paper's Figure 16
prints only parts of a GPT-4 system prompt for a step-wise search tool), so gbench uses its own
instruction: "Complete the Lean 4 formal proof below." (or, for a no-answer problem with a
placeholder, an instruction to replace the `sorry` in the placeholder with an explicit
closed-form answer, not a restatement of the problem, and the theorem's `sorry` with a proof,
and to output the whole file), the Lean and Mathlib versions, a request for a single
` ```lean ` block, and the statement. The prompt is the same with and without `--eval-thinking`.

## What counts as solved
A problem is solved only if a **trusted checker** accepts it. Three short-lived, network-less
containers of the Lean image are used per reply (a fourth prints the answers of a no-answer reply
that goes to the judge review, below):

1. **Reference**: the dataset statement alone (with its `sorry`) is compiled once and cached per
   image under `$GBENCH_CACHE_DIR/putnam_formal_ref`. For a no-answer problem the reference
   answer is compiled right after the placeholder, in its scope, as
   `<name>_solution.gbench_reference`.
2. **Solution**: the reply's last ` ```lean ` (or ` ```lean4 `) block is compiled. Fences are
   paired line by line: a line of three or more backticks opens a block and a bare line of at
   least as many backticks closes it; a reply that opens a new ` ```lean ` before closing the
   previous block abandoned that block, so the new opening starts over (its final code block is
   compiled, not the prose between the two openings). A reply whose fences are not on lines of
   their own is read with an inline pattern; a reply with no closed block is compiled whole.
   A reply that declares the
   theorem is compiled as written, under `import Mathlib` and the statement's own `open` and
   `set_option` lines (its own imports are dropped); a proof-body-only reply is placed under the
   dataset statement. The file is compiled inside a `noncomputable section`, which affects code
   generation only: several release placeholders are plain `abbrev`s whose answers use
   noncomputable real-number operations, and filling them in verbatim would otherwise not
   compile. The reply's code can write only to this container's output directory.
3. **Checker**: a gbench Lean program, in a fresh container where no reply code runs, reads the
   solution's declarations straight from its compiled file (the solution module is never
   imported), replays them through the Lean kernel on a clean `import Mathlib` environment
   (`Lean.Environment.replay` of the pinned Lean v4.33.1), and requires that:
   - the theorem exists, is a theorem, and has the reference theorem's type, compared with every
     definition declared in each file unfolded (so a reply cannot give a dataset name or a
     generated auxiliary such as `<theorem>.match_1` another meaning);
   - every other dataset declaration (helper definitions, inductive types, fixed answers such as
     `:= True`) is identical in kind, type and value;
   - an answer placeholder keeps its name and type (its value is the model's answer);
   - the theorem depends on no axiom beyond `propext`, `Classical.choice` and `Quot.sound`
     (so `sorry`, a declared `axiom`, or `native_decide` fail);
   - in the no-answer setting, the answer is faithful (below).

The verdict is the checker's own output, never the solution compile's. After the solution is
compiled, the harness draws a random 128-bit nonce and passes it only to the checker (on its
command line). The checker prints `GBENCH_CHECKER <nonce> start` and then exactly one line
`GBENCH_VERDICT <nonce> <ok|fail|review|usage> <JSON>` (`usage`, exit code 2, only for a
malformed command line); every control and line-separator character in the JSON (for example a
newline inside a declaration name or a kernel message) is written as a `\uXXXX` escape, so
nothing the solution declares can begin a line. The harness captures the output as bytes,
decodes it as UTF-8 without newline translation (so a carriage return stays a carriage return)
and splits it at newlines only, and accepts `ok` only with exit code 0 and exactly one line
carrying the nonce (`review` needs exit code 3; its JSON is `{"different": <tactic or "">}`).
Output without the start line means the checker never ran (it does not compile under the image,
or `lake` failed): that is a harness error, not a failed proof. This defeats replies that comment out or reinterpret the statement
(`variable`/`include`, `macro_rules`, namespaces, redefined helpers), forge an axiom report or
a verdict line, or exit early.

### Answer faithfulness (Lean no answer)
The leaderboard asks every no-answer result to state its answer-faithfulness review procedure
([docs/leaderboard.html](https://github.com/trishullab/PutnamBench/blob/97b5e9458b04e3d1044e8f8afe1724ff133f5a53/docs/leaderboard.html)
at the pinned commit: "Results should identify the benchmark commit, any modified inputs, and
the answer-faithfulness review procedure"); the one full no-answer entry in `docs/results.json`
used an LLM "placeholder-faithfulness review, followed by manual checking". gbench's procedure
has two stages.

**Stage one: kernel proof, inside the checker.** With every definition of the solution (and of
the reference) unfolded, so both answers mention only Mathlib, the checker tries these tactics
in order on the goal `<model answer> = <reference answer>` and accepts the first that closes it:

`with_reducible rfl`, `norm_num`, `simp`, `ext <;> simp`, `ext <;> norm_num`,
`ext <;> simp <;> tauto`, `ext <;> ring_nf`, `rfl`

If none does, it tries `norm_num`, `simp` and `decide` on `<model answer> != <reference
answer>`. Each attempt has its own budget of 400000 heartbeats; a proof it finds is checked by
the kernel and may use only the standard axioms. A proof of equality accepts the answer (for
example `100 + 1` for `101`, `{8, 7, 4, 3}` for `{3, 4, 7, 8}`, `fun p => p.1 - p.2` for
`fun (a, b) => a - b`); a proof of difference rejects it as `answer not faithful`. An answer that
mentions a local declaration without a definitional value (a local inductive type, `opaque`,
`axiom`) is also rejected, since it cannot be compared.

**Stage two: judge review.** An answer that stage one did not decide (an equal value in a form
the tactics cannot bridge, such as `1 / Real.sqrt 2` for `Real.sqrt 2 / 2`, or a restatement of
the problem such as `sInf {p | ...}`, a `Classical.choose`, or the theorem's own condition) goes
to the gbench Gemini judge cascade (PutnamBench names no review model; the cascade is gbench's
substitute for the LLM reviewer). The judge gets the informal statement, the Lean statement the
model saw, the release's reference answer comment, and both answers as Lean elaborates them in
the clean Mathlib environment (the model's with its own definitions unfolded), and answers
`Grade: CORRECT` (faithful: the same object, stated explicitly, in any equivalent form) or
`Grade: INCORRECT` (a different object, or a restatement of the problem). Its last `Grade:` line
decides; a reply without one is asked again, 3 times in all. Judge temperature is 0.0.

The two answers and their type are printed by the checker's **print mode** in a fourth
container, with its own fresh nonce (it prints its start line and then exactly one line
`GBENCH_PRINTED <nonce> <JSON>` with `answer`, `reference` and `type`, or with `error`). Like the
checker it reads the compiled files directly and runs no reply code. Before printing, it
renames every bound variable to `x` and drops expression metadata; neither changes the value.
The printer then makes the names distinct (`x`, `x_1`, ...) with shadowing turned off
(`pp.safeShadowing`), so no name hides another that is still used (Lean's `if h : c then t else e`
notation would otherwise reuse the hypothesis name of its first branch in the second, where it
can hide a variable). This keeps binder names, free text a model can choose (a Lean name can hold
any character, line breaks and backticks included), out of the prompt, and the prompt tells the
judge that bound variables are renamed. Literals are part of the answer's value and
are printed as Lean writes them: numbers, characters such as `'G'`, and strings such as `"abc"`,
quoted, with quotes, backslashes, line breaks and other control characters escaped (so
`"abc".length - 3` is shown as it is, and it is 0). Lean's own quoting leaves U+0080 to U+009F, U+2028 and U+2029 as
they are; the printer writes those as `\uXXXX` escapes too, which Lean reads as the same
character. A `Name` literal is printed as the applications it is built from, each component a
string literal (`` `Grade.CORRECT `` as `Lean.Name.mkStr2 "Grade" "CORRECT"`): Lean's quoted-name
form writes each component as it is, so a component with a line break or the closing escape
bracket U+00BB would print across lines or read as a different name (Lean's own `Name.toString`
notes that such names do not round-trip). To get this form the printer wraps the arguments of
`Name.str`, `Name.num` and `Name.mkStr1` to `Name.mkStr8` in empty metadata, which does not change
the value either. Text a model writes inside a literal is thus shown like the rest of the value:
each block is fenced (below) and the prompt says the answer is data to compare, not instructions.
No reference answer or answer type in the release contains a string literal or a `Name`, so none
of them prints differently because of it.

Every notation that `import Mathlib` loads prints through this printer, and a few would write data
from the term as raw text, so the printer stops them:
- ProofWidgets' HTML notation writes the text of `ProofWidgets.Html.text` and the tag and attribute
  names of `ProofWidgets.Html.element` as they are (across lines, unquoted, re-indented). The
  printer wraps every string literal in empty metadata, which that notation does not match, so
  such a term prints as the applications it is built from, with its strings quoted.
- Qq's `q(...)` notation rebuilds constant and binder names from `Lean.Expr` data and prints them
  as identifiers. The printer turns it off (`pp.qq`, Qq's own option), so a `Qq.Quoted.unsafeMk`
  term prints as the applications it is built from.
- The `do` notation lays a bind out over several lines, with nothing but a line break between
  its items. The printer wraps the continuation of each `Bind.bind` in empty metadata, so a bind
  prints as `>>=` on one line.

Whatever notation printed it, the result is checked before it is laid out: every atom outside a
literal must be a token of Lean's grammar or a keyword (spaces around a token are layout), every
identifier must be free of line-breaking characters and of the escape brackets U+00AB and U+00BB,
string and character literals are written anew from their value (as above), numbers and field
indices from theirs, a decimal such as `3.5` must be plain ASCII, and a quoted name must read back
as the same name. A line break left in the laid-out text is then layout, which the printer joins
(below). The printed text must then hold no character that can break a line (a control character,
U+0080 to U+009F, U+2028 or U+2029). An answer that fails any of this is refused as a printer error
(below), never shown misstated; the harness also refuses any printed text with such a character.
These rules were checked against every printing extension the pinned `import Mathlib` registers:
292 delaborator entries (185 functions) and 437 app unexpanders (355 of them generated by
`notation`-style commands, which only rearrange the printed arguments). Apart from the ones above
and the quoted-name printer, none writes term data as raw text (Lean's `Std.Tactic.Do` goal
notation prints hypothesis names kept in expression metadata, which the printer drops). With these
rules all 346 release answers and their types print identically, except that 19 answers now name
inner binders `x_1`, `x_2`, ... where an inner `x` used to hide an outer one.

Lean lays a few notations out over several lines whatever the width: it breaks the line after the
`;` of a `let` or `have` (and so after `show T from e`, which elaborates to a `have`) and before
each alternative of a `match` (in Lean v4.33.1, `optSemicolon` and `matchAlts` in
`src/Lean/Parser/Term.lean` use `ppLine`, which `src/Lean/Parser/Extra.lean` formats as a hard line
break). The printer joins each line break, with the spaces around it, into a single space:
`fun n => let m := n ^ 4; Nat.sqrt m` prints as `fun x => have x_1 := x ^ 4; x_1.sqrt`. It keeps the
joined text only when Lean's term parser reads it as the same syntax as the laid-out text (a
laid-out text the parser cannot read at all is kept joined). When the two readings differ, the line
breaks carried meaning, as between the items of a `do` block, and the answer is refused. A term
that ends in a block which only indentation closes (the alternatives of `match`, of `fun` or of
Mathlib's `Finsupp` notation, `fun` with a subscript 0; the items of `do`; the tactics of `by` or
`show ... by`) is first put in parentheses, so that on one line the block still ends where it did:
a `match` in the first alternative of another prints as
`(match x with | 0 => (match x with | 0 => 1 | x_2.succ => x_2) | x_1.succ => x_1 + 2)`.
Parentheses do not change the value. Lean's own layout of a `Finsupp` term nested in another has
no line break to join (Mathlib separates its alternatives with spaces) and reads as if the inner
term had every later alternative; the parentheses fix that too. No release answer or type uses any
of these notations, and all 346 print as before.

The answers are printed in full on one line. Lean's own printing limits are all lifted there:
by default the printer elides subterms nested deeper than 50, or met after 5000 steps, as an
ellipsis, stops at 200000 heartbeats and a recursion depth of 512, and breaks lines at 120
columns (proofs inside an answer stay elided; they do not change its value). What remains are
the printer container's limits, `GBENCH_PUTNAM_FORMAL_CHECK_TIMEOUT_S` (default 900 s) and
8 GB of memory. An answer that cannot be printed within them, or that the printer fails on (an
error, a crash, or output without exactly one result line), gets **no review and no score**: its
`answer_review` outcome is `not printed` (with the reason in `verification`), the attempt is
left out of the accuracy, and the run is not `leaderboard_comparable`. It is
never scored as a failed proof, and no placeholder text is sent to the judge in its place. On
the pinned image, a sum nested 12500 deep prints in 14 s at a 7.4 GB peak and one nested 20000
deep runs out of memory; a shared term that unfolds into 2^22 summands prints 46 MB, on one line,
in 11 minutes at a 6.4 GB peak; all 346 release answers print in at most 502 characters. Each block in the prompt is
fenced with a run of backticks longer than any inside it, so nothing in an answer can close its
fence, and the prompt says the answer is data to compare, not instructions. Every printed answer
is sent, whatever its length; a request that no cascade model answers is a judge outage (below).

Five release files note that the problem has several correct answers (`putnam_1973_b4`,
`putnam_1984_b1`, `putnam_2005_b1`, `putnam_2011_a3`, `putnam_2015_a2`). There a proof of
difference does not reject the answer: it goes to the judge, which is told that any explicit
answer is faithful (the kernel has already proved it satisfies the statement) and only a
restatement is not.

Each no-answer trace with a placeholder that reached the comparison records `answer_review`:
`{"stage": "kernel", "faithful": true|false}`, or
`{"stage": "judge", "faithful": true|false|null, "judge_model": ..., "outcome": ...}` with
`outcome` one of `faithful`, `not faithful`, `judge outage`, `no verdict`, `not printed`. A
trace the judge answered also has `judge_model` and `judge_grade`. A judge outage (every
cascade model failed) is `judge_grade: "JUDGE_OUTAGE"` and is left out of the accuracy, as in
every judged gbench suite. A judge that answers without a `Grade:` verdict line is asked again
(3 calls in all); markdown or LaTeX around the line (`**Grade:** CORRECT`,
`Grade: **CORRECT**`, `**Grade**: INCORRECT`) is removed before it is read, and a line that
restates the format (`Grade: CORRECT / INCORRECT`) is not a verdict. When no call gives a
verdict the trace is `JUDGE_FAILED`: a grading failure, not the model's outcome, so it is
`scoring_excluded` (left out of the accuracy), counted in base's `judge_failures` and
`answer_review.judge_no_verdict`, and the run is `completed_with_errors`. An answer that could
not be printed is `scoring_excluded` and left out of the accuracy too.

## Reading the result
- `accuracy` (headline): **Lean with answer**, the share of attempts kernel-checked as solved
  (pass@1 with one attempt). `accuracy_lean_no_answer` is the same for **Lean no answer**
  (attempts with a judge outage, a review without a verdict or an answer that could not be
  printed are left out).
- `putnambench_report.<setting>`: `leaderboard_column`, `results_key`, `problems` (672),
  `attempts_per_problem`, `solved` (problems with at least one accepted attempt, the
  leaderboard's "solved out of 672" at the run's budget), `solved_pct` and `accuracy`. `solved` and `solved_pct` count only scored attempts: a problem whose every attempt was a
  harness error or a judge outage has no verdict, so it is left out of `problems` and
  `solved_pct` (which is then over the problems with a scored attempt, null when none was
  scored) and counted in `problems_unscored`, named in `leaderboard_comparable_reason`. The
  no-answer entry adds `answer_placeholder_attempts`, `answer_not_faithful` (attempts that
  passed every other check but whose answer was rejected, by either stage), `answer_review`
  (counts: `kernel_faithful`, `kernel_not_faithful`, `judge_faithful`, `judge_not_faithful`,
  `judge_outage`, `judge_no_verdict`, `not_printed`) and `judge_outages`.
- `compute_budget`: `pass@k` for k attempts per problem (`--attempt-count k`; default 1).
  Leaderboard entries state their own budgets.
- `category_accuracy` has one entry per setting. `total_questions`, `correct_answers` and
  `effective_n` (attempts scored) describe the headline setting, "Lean with answer", so the
  Questions / Correct / Accuracy printed together agree; the counts over both settings (1,344
  generations per attempt) are `total_questions_all_settings` / `correct_answers_all_settings`.
  A setting with no scored attempt reports `accuracy` null (the headline then carries
  `accuracy_withheld_reason`), never 0.
- `answer_faithfulness`: the two-stage procedure above, recorded in the result, with `judge`
  (`gbench-gemini-cascade`), `judge_cascade` (the ordered models) and `judge_models` (answered
  review calls per model).
- `lean_image`: the image reference, its local id, and the toolchain, Mathlib revision and
  PutnamBench commit read from it; `reproducible` is true when all three match the pins (the
  startup check requires it).
- `leaderboard_comparable` is true for a full run. It is false, with the reason, for an
  `--eval-limit` subset (N samples in total, spread over the two settings; each setting's problems stratified by tag), if any
  statement failed to compile (`unverifiable_statements`, which should be 0 under the pinned
  image), if any reply was not judged because of harness errors, if any answer in the
  no-answer setting did not get its review (judge outage or no verdict), if any answer could
  not be printed for the review (the reason names the causes; each trace's `verification` has
  its own), or if the run's token budget cut off any reply (below).
- `replies_cut_off_by_budget` and `replies_cut_off_in_reasoning`: replies that stopped at the
  token limit (`finish_reason` `length`), including an empty answer after reasoning (the second
  count). PutnamBench publishes no generation harness, so there is no official cap and the
  suite keeps gbench's token budget; every such stop is this run's budget: the
  `max_tokens` gbench sends (`--max-output-tokens`, else gbench's default floor for the mode), a
  `max_tokens` the context window lowered, or the end of the window when none was sent. A reply
  cut off mid-answer is checked on what arrived; an empty one is scored as a failure and counted
  as a failed request (`empty_responses`), so the run reports `completed_with_errors`. Either way
  it measures the budget rather than the model, so any of them makes the run not
  `leaderboard_comparable`.
- `verification_harness_errors`: replies never judged because Docker or the host failed (or the
  checker did not start). They are harness errors, not the model's outcome: left out of the
  accuracy (numerator and denominator), counted as failed requests (`judge_grade`
  `JUDGE_FAILED`), so the run reports `completed_with_errors`, and a run where every reply hit
  one is `failed` with its score withheld. Fix the host and re-run.
- Each trace records the checker's reason in `verification`.

## Requirements
- **Docker** with a reachable daemon.
- **The bundled Lean image**, built locally (it is never pulled). Build it once:
  ```bash
  docker build -t gbench-putnam-formal:97b5e945 -f docker/putnam_formal.Dockerfile docker
  ```
  It is PutnamBench's `lean4/` project at the pinned commit (checked against its git tree id),
  on `leanprovercommunity/lean4` pinned by digest (Debian 12, elan 3.1.1), and repeats the
  project's CI: `lake exe cache get`, `lake build`, `python3 scripts/rewrite_solutions.py`,
  `lake build putnam_with_solutions`. The build needs network access (Debian `python3`, the
  Lean toolchain, the Mathlib olean cache, the PutnamBench tarball) and about 25 GB of disk for
  the image. If `lake exe cache get` fails with "too many open files", add
  `--ulimit nofile=1048576:1048576` to the build command.
- **`GEMINI_API_KEY`** (or a key list in `GEMINI_API_KEYS`) and the `google-genai` package, for
  the stage-two answer review of the Lean no answer setting. The Lean with answer headline uses
  no judge. Judge cascade knobs: `GBENCH_JUDGE_MODELS`, `GBENCH_JUDGE_MODEL`,
  `GBENCH_JUDGE_CASCADE_ROUNDS`, `GBENCH_JUDGE_BACKOFF` (see [env-knobs](env-knobs.md)).
- At startup the suite checks that the image exists (else it hard-errors with the build command),
  that `/opt/putnamenv/lean-toolchain` is `leanprover/lean4:v4.33.1`, that Mathlib is locked to
  `0df444a360eaa60ab8c11dca51a86af692955474`, that `/opt/putnamenv/.gbench-putnambench-revision`
  is the pinned commit, that `import Mathlib` compiles, that the checker compiles and runs (it
  must print its nonce-tagged start and usage lines), and that a Gemini key is set and not
  rejected by the judge endpoint. Any failure is `infra_required` (`status: "error"`), never a
  skipped or partial run.
- `GBENCH_PUTNAM_FORMAL_LEAN_IMAGE` (legacy alias `PUTNAM_LEAN_IMAGE`) selects another image
  that passes the same check; `GBENCH_PUTNAM_FORMAL_LEAN_PROJECT` is its Lake project (default
  `/opt/putnamenv`). Every compile runs `lake env lean` from there.
- Network access to `codeload.github.com` on the first run (the dataset tarball), unless the
  cache directory is already populated, and to the Gemini API for the answer review.
- Knobs: `GBENCH_PUTNAM_FORMAL_COMPILE_TIMEOUT_S` (default 300; legacy alias
  `PUTNAM_COMPILE_TIMEOUT_S`) caps each compile, `GBENCH_PUTNAM_FORMAL_CHECK_TIMEOUT_S`
  (default 900) caps the checker and, separately, the answer printer, `GBENCH_PUTNAM_FORMAL_TEMPERATURE` sets the sampling
  temperature for this suite. A reply whose compile or check runs out of time is a failed proof;
  the container is killed. The answer printer gets its own `GBENCH_PUTNAM_FORMAL_CHECK_TIMEOUT_S`;
  running out of it is `not printed`, not a failed proof. A check takes roughly 20 to 30
  seconds, and printing the answers of a reply sent to review about 10 more.

## Run
```bash
export GEMINI_API_KEY=...
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals putnam_formal \
       --sandboxes 8 --max-output-tokens 32768
```
Verification runs after generation finishes, `--sandboxes` checks at a time (default: the eval
concurrency); the answer reviews follow. Add `--attempt-count k` for a pass@k budget, or
`--eval-limit 20` for a quick, non-comparable subset: 20 samples in total, spread over the two
settings (`sampling.allocate_limit`, seeded `putnam_formal`: 10 Lean with answer, 10 Lean no answer),
never 20 problems in each. Both settings draw from the same tag-stratified problem order, so the
setting with fewer samples runs a subset of the other's problems; with `--eval-limit 1` the headline setting
(Lean with answer) gets the one sample and Lean no answer does not run. The run logs its plan on its first line and records it as
`eval_limit_plan`.

## Tests
The suite's tests are the `tests/test_*putnam*.py` files: `tests/test_putnam_formal_verifier.py`,
`tests/test_putnambench_release.py`, `tests/test_putnam_formal_review.py`, the review-payload
tests (`tests/test_putnam_formal_payload.py`), the answer-display tests
(`tests/test_putnam_formal_display.py`, and `tests/test_putnam_formal_binders.py` for `let`,
`have`, `show ... from` and `match`), the answer-printer tests
(`tests/test_putnam_formal_printer.py`) and the string and name literal tests
(`tests/test_putnam_formal_literals.py`). The pure tests run anywhere:
`python -m pytest tests/test_*putnam*.py -q`.
The kernel tests (every known attack including crafted declaration names, honest proofs, the
answer-faithfulness cases, the review payload and how every notation prints) need the image and
run with the same command plus `GBENCH_TEST_LEAN_IMAGE=gbench-putnam-formal:97b5e945`. The
verifier, review, payload, display and binders kernel tests take about 50 minutes; the printer and
literals kernel tests add to that. They use a stand-in judge, so they need no Gemini key.
