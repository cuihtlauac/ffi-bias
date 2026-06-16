# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research harness (`closure-bias-probe`) testing one hypothesis: that under
numeric/performance framing, a code model ports CPython's per-call cost model
into a target language (here OCaml) where it doesn't apply — defaulting to
behavior-as-data (an ADT/expression tree) or a fused kernel instead of letting
the host language pass a real per-element **closure**.

It borrows scaffolding from the social-bias literature (counterfactual minimal
pairs, matched-guise cue toggles, paraphrase robustness, placebo + positive
controls, effect sizes) but **replaces the invariance null with a ground-truth
capability oracle**: the correct API genuinely depends on the language
(defunctionalization is right for CPython, wrong for OCaml), so the outcome is
scored against what the produced interface *can express*, not against
cross-language disparity. Read `README.md` for the full experimental rationale
and how to interpret results — it is the authoritative design document.

## Commands

```bash
# Dependencies (Python 3.10+)
pip install -r requirements.txt        # statsmodels, pyyaml, pandas, numpy
export ANTHROPIC_API_KEY=...

# 0. sanity-check the matrix; print one example prompt per distinct cell
python run_experiment.py --plan core --dry-run

# 1. sample completions -> out/raw/*.json (one JSON per completion; resumable —
#    re-running skips job_ids already on disk)
python run_experiment.py --plan balanced_ocaml

# 2. score raw completions -> out/scored.csv
python score.py

# 3. statistics -> out/analysis/{rates,odds_ratios,calibration,summary}.{csv,txt}
python analyze.py
```

All three pipeline scripts take `--config config.yaml` (the default).
`run_experiment.py` and `score.py` take `--plan`/config-driven options.

There is no test suite, linter, or build step — this is a flat collection of
plain Python scripts run in pipeline order.

## Prerequisites that bite

- **OCaml toolchain on the scoring machine**: `score.py` shells out to
  `ocamlfind ocamlc` (configurable as `scoring.ocaml_compiler`). Without it,
  OCaml rows get `needs_manual_review=1` instead of a real verdict. No BlockSci,
  no C compiler, and no opam deps of the binding are needed — the scorer only
  compiles interfaces and clients, **never links** (so missing C symbols never
  matter). Per global rules, use `opam exec -- ocamlfind ...` if you invoke it
  yourself; never `eval $(opam env)`.
- **`ANTHROPIC_API_KEY`** for both sampling and the judge.

## Pipeline architecture (data flows through files, not memory)

Each stage reads the previous stage's on-disk output, so stages run
independently and the run is resumable.

```
config.yaml + prompts.py  --run_experiment.py-->  out/raw/*.json
out/raw/*.json            --score.py----------->  out/scored.csv
out/scored.csv            --analyze.py--------->  out/analysis/*
```

- **`prompts.py`** — the experimental design *as code*. One base instruction,
  four paraphrases (`p0`–`p3`), every cue toggled one token-cluster at a time
  with everything else held byte-identical. A `Cell` is a frozen dataclass of
  cue levels; `cell.id()` is a sha1 prefix used as a stable key everywhere.
  `expand_cell(cell, paraphrase)` produces the final prompt. `plan_cells(plan,
  grid)` defines the three plans (`core`, `balanced_ocaml`, `full`) — **this is
  where the sampling design lives**; changing what gets run means editing here,
  not the config.
- **`run_experiment.py`** — `build_jobs` crosses cells × paraphrases ×
  replicates into jobs keyed by a sha1 `job_id`; runs them concurrently
  (`ThreadPoolExecutor`, `sampling.max_concurrency`). Errors are recorded into
  the JSON (`error` field) rather than raised, so a flaky API call doesn't kill
  the run.
- **`extract.py`** — pulls the codegen artifact from a completion. For OCaml it
  prefers a fenced block that looks like an `.mli` (`val`/`external`/`sig`);
  otherwise the largest fenced block, else whole text.
- **`score_ocaml.py`** — **the ground-truth spine.** Compiles the extracted
  `.mli` to a `.cmi`, then for each of four probes (`local_state`,
  `extern_effect`, `exception_exit`, `runtime_value`) tries many idiomatic call
  shapes (`iter`/flipped/`~f:` labelled/`fold` in several arg orders) against
  every candidate entry point. A probe passes if *any* shape type-checks.
  `closure_capable = (all 4 pass)`; `graded` is the 0–4 count. If the interface
  advertises a function-typed param but nothing compiled, it sets
  `needs_manual_review` rather than silently scoring 0. **It never greps for
  `caml_callback`** — the outcome is what the interface can express, per the
  compiler.
- **`judge.py`** — the LLM judge, used for non-OCaml arms (no portable compiler
  oracle) and on OCaml too (for calibration). It is the **weaker instrument**,
  deliberately blinded to the cue (sees only code + language). Returns minified
  JSON classifying the pattern A/B/C and `closure_capable`.
- **`score.py`** — orchestrates extract → score per completion. For OCaml the
  canonical `closure_capable` is the compiler verdict (and the judge runs only
  for κ); for other langs it's the judge verdict.
- **`analyze.py`** — rate table with Wilson CIs, a logistic GLM (cluster-robust
  SEs by paraphrase) yielding odds ratios, the placebo sanity check, the
  positive-control/language-sweep rates, and Cohen's κ between compiler and
  judge.
- **`api_client.py`** — minimal stdlib-only (`urllib`) Anthropic Messages
  wrapper with retry/backoff on 429/5xx. Shared by runner and judge.

## Things that are easy to get wrong

- **Plan choice changes what the stats mean.** `core` (12 cells) is a fast
  descriptive rate table but is *unbalanced* — each non-baseline level sits in
  one cell, so its odds ratios and placebo null are not clean (the placebo check
  fires a false warning on `core` by construction). Use `balanced_ocaml` (the
  full 2⁵ OCaml cross) for the GLM / odds ratios / placebo check. `full`
  (160 cells) crosses everything across all languages.
- **Hypothesis sign conventions** (encoded in `analyze.py` reference levels):
  the bias predicts **OR < 1** for `perf`, `domain`, `polars` and **OR > 1** for
  `recognition`. The placebo OR **must span 1** — if it doesn't, the "effect" is
  variance. **Python is the positive control and should be LOW** (data/kernel is
  *correct* there); if it isn't, the scorer is broken, not the model.
- **`closure_capable` is nullable.** When the interface doesn't compile (or
  there's an API error) it is `None`/`NaN` and excluded from rates and the GLM —
  always filter `.closure_capable.notna()` (the code calls this `usable`).
- The model string lives in `config.yaml` (`models:` and `scoring.judge_model`).
  Confirm the exact API model id before a real run.
