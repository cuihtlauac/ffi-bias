"""
analyze.py — turn scored.csv into the pre-registered statistics.

Outputs (to out/analysis/):
  * rates.csv          per-cue closure-capable rates with Wilson 95% CIs
  * odds_ratios.csv    logistic GLM: OR + 95% CI for each cue (within-OCaml)
  * calibration.txt    Cohen's kappa between compiler and judge on OCaml samples
  * summary.txt        plain-language readout incl. positive control + placebo

Model:
  closure_capable ~ perf + domain + polars + recognition + placebo
  fit on lang == ocaml, usable samples (interface compiled), with paraphrase as
  a fixed effect and cluster-robust SEs by paraphrase. (A true random-effects
  logistic is better still — see note at the bottom for the pymer4/R one-liner.)

Usage:
  python analyze.py --config config.yaml
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import yaml
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint

_HERE = os.path.dirname(os.path.abspath(__file__))


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return (k / n, lo, hi)


def rate_table(df, cues):
    rows = []
    for cue in cues:
        for level, g in df.groupby(cue):
            k = int(g.closure_capable.sum())
            n = int(g.closure_capable.notna().sum())
            p, lo, hi = wilson(k, n)
            rows.append(dict(cue=cue, level=level, k=k, n=n,
                             rate=p, ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


def fit_glm(df):
    """Logistic GLM, paraphrase as fixed effect, cluster-robust SE by paraphrase."""
    d = df.copy()
    # reference levels chosen so positive ORs == "more closures"
    for col, ref in [("perf", "absent"), ("domain", "neutral"),
                     ("polars", "absent"), ("recognition", "default"),
                     ("placebo", "a")]:
        d[col] = pd.Categorical(d[col], categories=[ref] +
                                [x for x in d[col].unique() if x != ref])
    d["closure_capable"] = d["closure_capable"].astype(int)
    formula = "closure_capable ~ C(perf) + C(domain) + C(polars) + C(recognition) + C(placebo) + C(paraphrase)"
    model = smf.glm(formula, data=d, family=sm.families.Binomial())
    res = model.fit(cov_type="cluster", cov_kwds={"groups": d["paraphrase"]})
    return res


def odds_ratio_table(res):
    params = res.params
    ci = res.conf_int()
    rows = []
    for name in params.index:
        if name == "Intercept" or name.startswith("C(paraphrase)"):
            continue
        rows.append(dict(term=name,
                         OR=float(np.exp(params[name])),
                         ci_lo=float(np.exp(ci.loc[name, 0])),
                         ci_hi=float(np.exp(ci.loc[name, 1])),
                         p=float(res.pvalues[name])))
    return pd.DataFrame(rows)


def cohen_kappa(a, b):
    a = np.asarray(a); b = np.asarray(b)
    mask = ~(pd.isna(a) | pd.isna(b))
    a, b = a[mask].astype(int), b[mask].astype(int)
    n = len(a)
    if n == 0:
        return np.nan, 0
    po = (a == b).mean()
    # expected agreement
    pa1, pb1 = a.mean(), b.mean()
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    kappa = (po - pe) / (1 - pe) if pe < 1 else np.nan
    return kappa, n


def main(cfg, run_id=None):
    # --run-id => read scored.csv from and write analysis into the immutable
    # lab/results/<run-id>/ that score.py --run-id produced; else the out/ scratch.
    if run_id:
        run_dir = os.path.join(_HERE, "lab", "results", run_id)
        scored_csv = os.path.join(run_dir, "scored.csv")
        out = os.path.join(run_dir, "analysis")
        if not os.path.exists(scored_csv):
            sys.exit(f"{scored_csv} not found — run `score.py --run-id {run_id}` first.")
        if os.path.exists(os.path.join(out, "summary.txt")):
            sys.exit(f"refusing to overwrite analysis in {out} — lab/results entries are "
                     f"append-only. Use a fresh --run-id.")
    else:
        out = cfg["paths"]["analysis_dir"]
        scored_csv = cfg["paths"]["scored_csv"]
    os.makedirs(out, exist_ok=True)
    df = pd.read_csv(scored_csv)

    cues = ["perf", "domain", "polars", "recognition", "placebo"]
    oc = df[df.lang == "ocaml"].copy()
    usable = oc[oc.closure_capable.notna()].copy()

    # 1) rate table
    rates = rate_table(usable, cues)
    rates.to_csv(os.path.join(out, "rates.csv"), index=False)

    # 2) logistic GLM odds ratios
    ortab = pd.DataFrame()
    glm_txt = ""
    if usable.closure_capable.nunique() > 1 and len(usable) >= 20:
        try:
            res = fit_glm(usable)
            ortab = odds_ratio_table(res)
            ortab.to_csv(os.path.join(out, "odds_ratios.csv"), index=False)
            glm_txt = res.summary().as_text()
        except Exception as e:  # e.g. single-paraphrase cluster or perfect separation
            ortab = pd.DataFrame()
            glm_txt = (f"GLM skipped: fit failed ({e!r}). "
                       f"Expected on a single-paraphrase / unbalanced pilot.")
    else:
        glm_txt = "GLM skipped: outcome not variable enough or too few usable OCaml samples."

    # 3) judge calibration (compiler vs judge on OCaml)
    cal = ""
    if "judge_closure_capable" in oc.columns:
        k, n = cohen_kappa(oc.closure_capable.values, oc.judge_closure_capable.values)
        cal = (f"Compiler-vs-judge Cohen's kappa on OCaml: {k:.3f}  (n={n})\n"
               f"Interpretation: >=0.8 the cross-language judge is trustworthy; "
               f"0.6-0.8 use with caution; <0.6 fall back to human coding.\n")
        with open(os.path.join(out, "calibration.txt"), "w") as f:
            f.write(cal)

    # 4) summary readout
    lines = []
    lines.append("=== CLOSURE-BIAS PROBE: SUMMARY ===\n")
    lines.append(f"total rows: {len(df)}   OCaml usable: {len(usable)} "
                 f"(of {len(oc)} OCaml; {len(oc) - len(usable)} flagged/non-compiling)\n")

    lines.append("\n-- Within-OCaml closure-capable rate by cue (Wilson 95% CI) --")
    for _, r in rates.iterrows():
        lines.append(f"  {r['cue']:<12} = {r['level']:<8} : "
                     f"{r['rate']:.2f}  [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]  (k={int(r['k'])}/{int(r['n'])})")

    if len(ortab):
        lines.append("\n-- Odds ratios (>1 => MORE closures; the bias predicts <1 for perf/domain/polars) --")
        for _, r in ortab.iterrows():
            sig = "*" if r["p"] < 0.05 else " "
            lines.append(f"  {r['term']:<24} OR={r['OR']:.2f} "
                         f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}] p={r['p']:.3f} {sig}")
        # placebo sanity
        plac = ortab[ortab.term.str.contains("placebo")]
        if len(plac):
            pr = plac.iloc[0]
            ok = pr["ci_lo"] <= 1 <= pr["ci_hi"]
            lines.append(f"\n  PLACEBO check: OR={pr['OR']:.2f} "
                         f"[{pr['ci_lo']:.2f},{pr['ci_hi']:.2f}] -> "
                         f"{'OK (CI spans 1; nuisance is inert)' if ok else 'WARNING: placebo moved the outcome — suspect spurious variance'}")

    # 5) positive control + language sweep
    def rate_for(mask):
        g = df[mask]; g = g[g.closure_capable.notna()]
        k, n = int(g.closure_capable.sum()), int(g.closure_capable.notna().sum())
        p, lo, hi = wilson(k, n)
        return p, lo, hi, k, n

    lines.append("\n-- Positive control & language sweep (closure-capable rate) --")
    for lg in df.lang.unique():
        p, lo, hi, k, n = rate_for(df.lang == lg)
        tag = " <-- positive control (expect LOW)" if lg == "python" else ""
        if n:
            lines.append(f"  lang={lg:<7}: {p:.2f} [{lo:.2f},{hi:.2f}] (k={k}/{n}){tag}")

    if cal:
        lines.append("\n-- Judge calibration --\n  " + cal.replace("\n", "\n  "))

    lines.append("\n-- GLM detail --\n" + glm_txt)
    lines.append(
        "\nNOTE: for a true mixed-effects logistic (random intercepts for "
        "paraphrase AND template/model), fit in R via pymer4:\n"
        "  glmer(closure_capable ~ perf+domain+polars+recognition+placebo + (1|paraphrase), family=binomial)\n"
        "The cluster-robust GLM above is a pragmatic stand-in.")

    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote analysis to {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run-id", default=None,
                    help="read/write lab/results/<run-id>/ (the dir score.py --run-id "
                         "produced) instead of out/.")
    args = ap.parse_args()
    main(load_config(args.config), run_id=args.run_id)
