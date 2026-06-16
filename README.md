# closure-bias-probe

[![GitHub](https://img.shields.io/badge/GitHub-cuihtlauac%2Fffi--bias-181717?logo=github)](https://github.com/cuihtlauac/ffi-bias)

Code models tend to carry one language's habits into another. This project tests a
specific case: asked under performance framing to write OCaml bindings around a C++
iterator, does the model import **Python's cost model** — where you hand the native
side a *description* of the work instead of a per-item function — and produce an
OCaml API that compiles and looks fine but can't actually accept a real closure? We
measure this objectively by asking the **real OCaml compiler** whether the produced
interface can express basic per-element tasks (counting while looping, calling other
libraries, early exit), and we isolate the cause by toggling one framing cue at a
time across many samples, with a Python positive control and a placebo to keep the
signal honest. The failure matters precisely because it's **invisible** — the binding
type-checks — and because existing work catches wrong-language or ugly-but-correct
output, not valid code that's architecturally crippled by a borrowed assumption.

The rest of this README makes that precise. In one sentence, the hypothesis under
test:

> Under numeric/performance framing, a code model ports **CPython's per-call
> cost model** into a target language where it doesn't apply (here OCaml),
> defaulting to *behavior-as-data* (an ADT/expression tree) or a *fused kernel*
> instead of letting the host language pass a real per-element **closure**.

> Under numeric/performance framing, a code model ports **CPython's per-call
> cost model** into a target language where it doesn't apply (here OCaml),
> defaulting to *behavior-as-data* (an ADT/expression tree) or a *fused kernel*
> instead of letting the host language pass a real per-element **closure**.

It borrows the experimental scaffolding of the social-bias literature
(counterfactual minimal pairs, matched-guise cue toggles, paraphrase
robustness, placebo + positive controls, effect sizes) but **replaces its
invariance null**. The correct API genuinely depends on the language —
defunctionalization is right for CPython and wrong for OCaml — so we score
against a **ground-truth capability oracle**, not against cross-language
disparity.

## The outcome measure (why it isn't a grep)

We never pattern-match for `caml_callback`. We ask what the produced *public
interface* can express, using the real OCaml compiler:

`ocamlfind ocamlc -c blocksci.mli` produces `blocksci.cmi` with **no
implementation, no C stubs, and no real BlockSci checkout**. Client probes
type-check against that `.cmi` alone (we compile, never link, so missing C
symbols never matter). Four probes each *require* genuine closure power:

1. `local_state` — close over caller-local mutable state (`ref`)
2. `extern_effect` — perform an arbitrary host-library effect per element
3. `exception_exit` — propagate an OCaml exception for early exit
4. `runtime_value` — capture a value known only at runtime

A closure-passing API compiles 4/4. A defunctionalized ADT compiles 0–1. A
fused kernel compiles 0. `closure_capable = (4/4)` is the primary binary
outcome; the 0–4 count is the graded score. The scorer tries several idiomatic
call shapes (`iter` / flipped / `~f:` labelled / `fold` in a few arg orders) per
entry point, so a perfectly good `fold`- or label-style API is **not**
penalised. If the interface clearly exposes a function-typed parameter yet
nothing compiled, the row is flagged `needs_manual_review` instead of being
silently scored 0.

## What transfers from the bias literature, and what doesn't

Transfers: counterfactual minimal-pair substitution; template + paraphrase
robustness; factorial cue toggles; placebo and positive controls; black-box
sampling over many completions; effect-size reporting (odds ratios).

Does **not** transfer: the invariance null (replaced by the capability oracle —
the one change that makes the study valid); WEAT/SEAT embedding tests (wrong
tool — the bias is a multi-line design choice, so behavioral generation probes
are right); protected-attribute lexicons (no analog — the "attribute" is a
framing cue you author).

## The cue grid (factors)

| factor        | levels                | tests                                   |
|---------------|-----------------------|-----------------------------------------|
| `lang`        | ocaml, python, julia, lua, c | python = positive control; others = target-conditionality |
| `perf`        | absent / present      | perf-framing trigger                    |
| `domain`      | neutral / numeric     | perf-numeric subculture as the source   |
| `polars`      | absent / present      | most on-the-nose causal probe           |
| `recognition` | default / offered     | recognition-vs-generation split         |
| `placebo`     | a / b                 | nuisance factor — its OR must span 1    |

Primary pre-registered contrasts are **within `lang=ocaml`, across cues**, with
Python as the ceiling. General FFI-callback caution (JNI/P-Invoke) is real and
cross-ecosystem, so a uniform `lang` effect would *not* localise the bias to the
Python-numeric subculture; the `domain` and `polars` sensitivity plus a
*non-uniform* `lang` effect are what pull "Python-specific prior" apart from
"generic FFI caution."

## Plans

- `core` (12 cells) — one-factor-at-a-time sweep off a clean baseline + Python
  control + language sweep. Best for the **descriptive rate table** and a fast
  first look. Not balanced: each non-baseline level sits in one cell, so don't
  read its odds ratios or placebo null as clean.
- `balanced_ocaml` (37 cells) — the full 2⁵ cross of OCaml cues (every main
  effect, incl. the placebo null, cleanly estimable) + Python ceiling + language
  sweep. **Use this for the GLM / odds ratios / placebo check.**
- `full` (160 cells) — everything crossed across all languages.

## Prerequisites

- Python 3.10+ with `pip install statsmodels pyyaml pandas numpy`
- `export ANTHROPIC_API_KEY=...`
- **OCaml toolchain on the scoring machine**: `ocamlfind` + `ocamlc` on PATH
  (e.g. `opam install ocamlfind`). No BlockSci, no C compiler, no opam deps of
  the binding are needed — we only compile interfaces and clients.

## Run order

```bash
# 0. sanity-check the matrix and read a few prompts
python run_experiment.py --plan core --dry-run

# 1. sample completions (one JSON per completion in out/raw/; resumable)
python run_experiment.py --plan balanced_ocaml

# 2. score: real compiler for OCaml (+ judge for calibration), judge for others
python score.py

# 3. statistics: rates + Wilson CIs, odds ratios, placebo check, judge kappa
python analyze.py
```

Set the exact model string(s) in `config.yaml` (`models:`). Raise
`sampling.n_per_cell` for small effects; rotate more `paraphrases` — the
literature's most repeated scar is effects that evaporate under rewording.

## Reading the output (`out/analysis/summary.txt`)

- **Rate table** — closure-capable rate per cue level with Wilson 95% CIs.
- **Odds ratios** — from a logistic GLM (`closure_capable ~ perf+domain+polars+
  recognition+placebo + paraphrase`) with cluster-robust SEs by paraphrase.
  The hypothesis predicts **OR < 1 for `perf`, `domain`, `polars`** and **OR > 1
  for `recognition`** (knows closures work, just doesn't propose them).
- **Placebo check** — its OR must span 1. If the placebo moves the outcome as
  much as a real cue, your "effect" is variance, not bias. (On `core` this fires
  a false warning by construction — use `balanced_ocaml`.)
- **Positive control** — Python should be **low** (data/kernel dominates and is
  *correct* there). If it isn't, the scorer is broken, not the model.
- **Judge calibration** — Cohen's κ between the OCaml compiler (ground truth)
  and the LLM judge. ≥0.8 → trust the cross-language judge; 0.6–0.8 → caution;
  <0.6 → hand-code a subsample. The judge is the weak instrument and is only the
  primary measure off the OCaml spine.

## Inference limits (read before claiming a result)

A clean result shows the model *ports CPython's cost model into the target under
numeric/perf framing*. It does **not**, by itself, separate a Python-specific
prior from generic FFI-callback caution — that's what the `domain` and `lang`
arms are for. If the effect is domain- and Polars-cue-sensitive but **not**
uniformly target-sensitive, you've localised it to the subculture rather than to
FFI in general.

For a true mixed-effects logistic (random intercepts for paraphrase and
model/template), fit in R/`pymer4`:
`glmer(closure_capable ~ perf+domain+polars+recognition+placebo + (1|paraphrase), family=binomial)`.
The cluster-robust GLM here is a pragmatic stand-in.

## Files

| file | role |
|------|------|
| `config.yaml`      | cue grid, models, sampling, paths |
| `prompts.py`       | templates, 4 paraphrases, cue expansion, plans |
| `run_experiment.py`| expand plan → sample completions → `out/raw/*.json` |
| `extract.py`       | pull the produced interface from a completion |
| `score_ocaml.py`   | **ground-truth compiler battery** |
| `judge.py`         | LLM judge for non-OCaml arms (calibrated vs compiler) |
| `api_client.py`    | Anthropic Messages API wrapper |
| `score.py`         | raw → `out/scored.csv` |
| `analyze.py`       | rates, odds ratios, placebo check, κ → `out/analysis/` |

## Validation status

Pure logic (prompt expansion, entry-point enumeration, probe-client generation,
the entire stats pipeline incl. Wilson CIs / cluster-robust GLM / κ) is verified
on fixtures and synthetic hypothesis-consistent data. The compiler battery
itself requires `ocamlfind` on the scoring machine — it is not exercised in an
environment without the OCaml toolchain.

## License

Open by default, so the methodology and results can be replicated and reused
with minimal friction:

- **Code** (everything but `lab/results/`) — MIT (`SPDX-License-Identifier: MIT`),
  see `LICENSE`.
- **Experimental data & figures** (recorded runs under `lab/results/`) —
  Creative Commons Attribution 4.0 (`CC-BY-4.0`). Reuse freely with attribution.

Copyright © 2026 Cuihtlauac ALVARADO (see `LICENSE`).
