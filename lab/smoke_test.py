#!/usr/bin/env python3
"""
lab/smoke_test.py — zero-token smoke test of the experimental setup.

Validates the parts of the pipeline that need no API tokens, no network, and no
key — i.e. everything except the model-generation and judge calls:

  1. Prompt expansion — cues appear / disappear as designed; the plan has the
     expected shape.
  2. Interface extraction + the OCaml compiler battery (the ground-truth oracle)
     discriminates the three API patterns, and the integrity flag fires on a
     higher-order interface the battery cannot drive.
  3. analyze.py end-to-end on synthetic, hypothesis-consistent data — rates,
     GLM odds ratios, and a placebo null that spans 1. Skipped (not failed) if
     statsmodels / numpy are not installed.

Run with the OCaml toolchain on PATH (never `eval $(opam env)`):

    opam exec -- python3 lab/smoke_test.py

Exit codes: 0 all checks passed · 1 a check failed · 2 cannot run the oracle
(ocamlfind missing). Nothing is written outside a temp dir.
"""

import os, sys, shutil, tempfile, itertools, hashlib, csv as csvmod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_PASS, _FAIL, _SKIP = [], [], []


def check(name, cond, detail=""):
    (_PASS if cond else _FAIL).append(name)
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail and not cond else ""))
    return cond


def skip(name, why):
    _SKIP.append(name)
    print(f"  [SKIP] {name} — {why}")


# --- 1. prompt expansion -----------------------------------------------------
def test_prompts():
    print("prompt expansion:")
    from prompts import Cell, expand_cell, plan_cells
    import yaml
    base = dict(perf="absent", domain="neutral", polars="absent",
                recognition="default", placebo="a")
    off = expand_cell(Cell(lang="ocaml", **base), "p0")
    on = expand_cell(Cell(lang="ocaml", **{**base, "perf": "present"}), "p0")
    rec = expand_cell(Cell(lang="ocaml", **{**base, "recognition": "offered"}), "p0")
    check("perf cue absent by default", "performance-critical" not in off)
    check("perf cue appears when perf=present", "performance-critical" in on)
    check("recognition clause appears when offered", "one option among others" in rec)
    grid = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))["grid"]
    cells = plan_cells("balanced_ocaml", grid)
    check("balanced_ocaml expands to 37 cells", len(cells) == 37, f"got {len(cells)}")


# --- 2. the compiler-battery oracle ------------------------------------------
# completion text -> expected (graded, closure_capable, needs_manual_review)
_FIXTURES = {
    "closure-passing": (
        "```ocaml\ntype t\ntype block\n"
        "val iter : t -> (block -> unit) -> unit\n"
        "val fold : t -> 'a -> ('a -> block -> 'a) -> 'a\n```",
        dict(graded=4, closure_capable=1, needs_manual_review=0)),
    "defunctionalized-ADT": (
        "```ocaml\ntype t\ntype op = Count | Sum_size | Filter_nonempty\n"
        "type result = { n : int; total : int }\n"
        "val run : t -> op list -> result\n```",
        dict(graded=0, closure_capable=0, needs_manual_review=0)),
    "fused-kernel": (
        "```ocaml\ntype t\ntype stats = { count : int; mean : float }\n"
        "val compute_stats : t -> stats\n```",
        dict(graded=0, closure_capable=0, needs_manual_review=0)),
    "predicate-only-HOF": (
        "```ocaml\ntype t\ntype block\n"
        "val count_where : t -> (block -> bool) -> int\n```",
        dict(graded=0, closure_capable=0, needs_manual_review=1)),
}


def test_oracle():
    """Return True iff the oracle could actually be exercised (ocamlfind present)."""
    print("compiler battery (the ground-truth oracle):")
    if shutil.which("ocamlfind") is None:
        print("  ocamlfind not on PATH — cannot smoke-test the oracle.")
        print("  re-run with: opam exec -- python3 lab/smoke_test.py")
        return False
    from extract import extract_interface
    from score_ocaml import score_mli
    for name, (text, exp) in _FIXTURES.items():
        code = extract_interface(text, "ocaml")["code"]
        s = score_mli(code)
        got = dict(graded=s.graded, closure_capable=s.closure_capable,
                   needs_manual_review=s.needs_manual_review)
        check(f"{name}: graded={s.graded} capable={s.closure_capable} "
              f"review={s.needs_manual_review}", got == exp, f"expected {exp}")
    return True


# --- 3. analyze.py on synthetic data -----------------------------------------
def _write_synthetic_csv(path):
    """Hypothesis-consistent OCaml rows + a few Python controls. Deterministic.
    Outcome falls with perf/polars/domain and rises with recognition; placebo is
    independent of the outcome (so its OR should span 1)."""
    levels = [("perf", ["absent", "present"]), ("domain", ["neutral", "numeric"]),
              ("polars", ["absent", "present"]), ("recognition", ["default", "offered"]),
              ("placebo", ["a", "b"])]
    keys = [k for k, _ in levels]
    rows, i = [], 0
    for combo in itertools.product(*[v for _, v in levels]):
        c = dict(zip(keys, combo))
        for para in ("p0", "p1", "p2", "p3"):
            for rep in range(2):
                score = 70
                score -= 30 if c["perf"] == "present" else 0
                score -= 20 if c["polars"] == "present" else 0
                score -= 20 if c["domain"] == "numeric" else 0
                score += 15 if c["recognition"] == "offered" else 0
                # Deterministic Bernoulli draw keyed on every factor EXCEPT placebo,
                # so the outcome is genuinely independent of placebo (its OR must
                # span 1). PYTHONHASHSEED-independent (hashlib, not builtin hash()).
                key = f"{c['perf']}|{c['domain']}|{c['polars']}|{c['recognition']}|{para}|{rep}"
                r = int(hashlib.sha1(key.encode()).hexdigest(), 16) % 100
                i += 1
                cc = 1 if r < score else 0
                rows.append(dict(job_id=f"j{i}", model="m", cell_id="x", paraphrase=para,
                                 replicate=rep, lang="ocaml", **c, closure_capable=cc,
                                 judge_closure_capable=(cc if i % 7 else 1 - cc),
                                 needs_manual_review=0, interface_compiles=1))
    for k in range(4):   # Python positive control (expected low)
        rows.append(dict(job_id=f"py{k}", model="m", cell_id="p", paraphrase="p0",
                         replicate=k, lang="python", perf="present", domain="numeric",
                         polars="present", recognition="default", placebo="a",
                         closure_capable=0, judge_closure_capable=0,
                         needs_manual_review=0, interface_compiles=1))
    with open(path, "w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def test_analyze():
    print("analyze.py on synthetic hypothesis-consistent data:")
    try:
        import numpy, statsmodels  # noqa: F401
    except ImportError as e:
        skip("analyze.py end-to-end", f"missing dep ({e.name}); pip install statsmodels numpy")
        return
    import importlib, pandas as pd
    analyze = importlib.import_module("analyze")
    work = tempfile.mkdtemp(prefix="cbp_smoke_")
    try:
        csv_path = os.path.join(work, "scored.csv")
        andir = os.path.join(work, "analysis")
        _write_synthetic_csv(csv_path)
        analyze.main({"paths": {"scored_csv": csv_path, "analysis_dir": andir}})
        check("rates.csv produced", os.path.exists(os.path.join(andir, "rates.csv")))
        check("odds_ratios.csv produced (GLM ran)",
              os.path.exists(os.path.join(andir, "odds_ratios.csv")))
        check("summary.txt produced", os.path.exists(os.path.join(andir, "summary.txt")))
        orp = os.path.join(andir, "odds_ratios.csv")
        if os.path.exists(orp):
            ortab = pd.read_csv(orp)
            plac = ortab[ortab.term.str.contains("placebo")]
            if len(plac):
                r = plac.iloc[0]
                check("placebo OR spans 1 (null control behaves)",
                      r.ci_lo <= 1 <= r.ci_hi, f"OR CI=[{r.ci_lo:.2f}, {r.ci_hi:.2f}]")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    print("=== closure-bias-probe smoke test (zero tokens, no network) ===\n")
    test_prompts()
    oracle_ran = test_oracle()
    test_analyze()
    print(f"\n{len(_PASS)} passed, {len(_FAIL)} failed, {len(_SKIP)} skipped")
    if _FAIL:
        sys.exit(1)
    if not oracle_ran:
        print("INCOMPLETE: the oracle was not exercised (ocamlfind missing).")
        sys.exit(2)
    print("OK: setup smoke-tested without spending a token.")
    sys.exit(0)
