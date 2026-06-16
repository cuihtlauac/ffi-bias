# 2026-06-16 — core on cerebras:gpt-oss-120b — closure-capability drops sharply under numeric/perf framing (pilot)

## What
First complete cue-sweep run of the probe. Plan `core` (12 cells: a within-OCaml
one-factor-at-a-time sweep + Python ceiling + julia/lua/c language sweep), single
paraphrase `p0`, n=10/cell → 120 completions on **cerebras:gpt-oss-120b**. OCaml
rows scored by the real compiler battery (ground truth); non-OCaml rows by the
gpt-oss LLM judge. Headline: on the OCaml spine, the closure-capable rate falls
from **0.96 (neutral domain) → 0.30 (numeric/BlockSci)**, **0.88 → 0.47** under
perf framing, and **0.83 → 0.60** under the Polars cue, while rising to **1.00**
when closures are explicitly named — every direction matching the hypothesis.

## Why
Tests the pre-registered within-OCaml contrasts (README §"The cue grid"): does a
code model retreat from passing a real per-element closure when the task is framed
in numeric/performance terms? This is the **first** run, so it establishes the
baseline and validates the instrument end-to-end on a live model. Subject is an
open model (gpt-oss-120b via Cerebras free tier), **not** the eventual target
(Claude) — chosen because the Gemini free tier capped at ~20 req/day/model made a
full sweep there infeasible.

- Pre-registered prediction: rate(closure_capable) LOWER for perf=present,
  domain=numeric, polars=present; HIGHER for recognition=offered; placebo inert.
- Confirmatory or exploratory: **exploratory** — directionally confirmatory on the
  OCaml spine, but `core` is unbalanced (single paraphrase, n=10), so single-cue
  attribution and the placebo null are not clean here. Not a confirmatory test.

## Provenance
Full manifest in the sibling `manifest.json`. Load-bearing fields:

- git commit: `44c17b63f2a9fe40cfd537ef5393a5c50ad5b045` (dirty: true — the tree was
  edited *after* this run to make `score.py` resilient to judge errors; the run
  itself was produced at this commit)
- config file: **`config.cerebras.yaml`** (sha256 `9e78875b…`); code/config identity
  hash `b2699d5e…` (config.yaml+prompts.py+score_ocaml.py+api_client.py)
- provider: `cerebras` · model requested → **served**: `gpt-oss-120b` → `gpt-oss-120b`
- sampling: temperature **0.7** (sent) · seed **17** · n_per_cell **10** · paraphrases **[p0]** · plan **core**
- OCaml oracle: `ocamlfind ocamlc 5.3.0` · Python `3.13.7` · pkgs sha256 `30a0af87…`
- platform: `Linux-6.17.0-35-generic-x86_64-with-glibc2.42`
- raw archive sha256: `863b7adeec905627b81f0ccfad561770abb1718194d698b586fa6df809a3a214` (120 completions, n_ok=120)
- This record was assembled from the original scored.csv/analysis (the exact
  artifacts analyzed), **not** a re-score — a re-score would re-sample the stochastic
  judge and burn quota; the OCaml compiler verdicts are deterministic regardless.

## Parameters and raw data
Within-OCaml closure-capable rate by cue (compiler oracle; usable OCaml = 68/70,
2 flagged non-compiling):

| cue | level | k / n | rate | Wilson 95% CI |
|-----|-------|-------|------|----------------|
| perf | absent | 43/49 | 0.88 | [0.76, 0.94] |
| perf | present | 9/19 | 0.47 | [0.27, 0.68] |
| domain | neutral | 46/48 | 0.96 | [0.86, 0.99] |
| domain | numeric | 6/20 | 0.30 | [0.15, 0.52] |
| polars | absent | 40/48 | 0.83 | [0.70, 0.91] |
| polars | present | 12/20 | 0.60 | [0.39, 0.78] |
| recognition | default | 43/59 | 0.73 | [0.60, 0.83] |
| recognition | offered | 9/9 | 1.00 | [0.70, 1.00] |
| placebo | a | 42/58 | 0.72 | [0.60, 0.82] |
| placebo | b | 10/10 | 1.00 | [0.72, 1.00] |

Odds ratios (logistic GLM): **not computed** — the GLM was skipped (perfect
separation → ZeroDivisionError, caught by the guard). A single paraphrase makes the
cluster-robust SE degenerate anyway. Clean ORs require `balanced_ocaml` + ≥2 paraphrases.

Controls:
- **Placebo**: OR n/a (GLM skipped). Marginal rate moved a 0.72 → b 1.00 — but on
  `core` placebo=b sits in specific cells (n=10), so this is the known design
  artifact, **not** a valid null test.
- **Positive control (Python)**: rate = **0.90** (18/20) — expected LOW, came out
  HIGH. Per README, a high Python rate flags the scorer, not the model — and here
  the Python rows are judge-scored by an **uncalibrated** gpt-oss judge.
- **Judge calibration**: Cohen's κ = **nan (n=0)** — `judge_on_ocaml_too: false` this
  run, so there is no compiler-vs-judge overlap. The cross-language arm
  (python/julia/lua/c rates: 0.90 / 0.80 / 1.00 / 0.70) is therefore **untrusted**.

Raw dataset: 120 completions in the frozen `raw/` (sha256 above);
`needs_manual_review`/non-compiling on the OCaml arm: 2 of 70.

## Analysis and open questions
On the OCaml spine (the trustworthy arm), the prediction held in **all four**
directions, with the numeric-domain effect by far the largest (0.96 → 0.30). This
is a clean, coherent, on-hypothesis signal for gpt-oss-120b — strong enough to
justify a proper run. Caveats that bound the claim:

1. **Unbalanced design.** `core` puts each "loud" level in only 1–2 cells, and the
   all-cues-on cell loads onto perf, domain, and polars simultaneously — so the
   marginal drops are confounded across cues. Cannot attribute to a single cue yet.
2. **No GLM / no clean placebo null** (separation + single paraphrase).
3. **Cross-language arm untrusted** (κ=nan; uncalibrated self-judging).
4. **Subject is gpt-oss-120b, not Claude** — does not address the original question
   about Claude; it's a first data point on an open model.

What this could NOT settle: Python-specific prior vs. generic FFI-callback caution
(needs the domain/lang arms with a *trusted* scorer), and single-cue attribution.

Follow-ups:
- `balanced_ocaml` + `[p0,p1,p2,p3]` for real odds ratios and a valid placebo null
  (note: ~6–7M tokens on gpt-oss → multi-day on Cerebras free; right-size or accumulate).
- Re-enable `judge_on_ocaml_too: true` to recover κ and rehabilitate the cross-lang arm.
- Eventually run the same probe on Claude (the pre-registered target) when funded.
