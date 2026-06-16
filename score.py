"""
score.py — walk out/raw/*.json, extract the produced interface, score it.

For lang == ocaml:   real compiler battery (ground truth) + (optional) judge.
For other langs:     LLM judge only.

Emits one row per completion to out/scored.csv.

Usage:
  python score.py --config config.yaml
"""

import os, json, glob, argparse
import pandas as pd
import yaml

from extract import extract_interface
from score_ocaml import score_mli, score_to_row, OcamlScore
from judge import judge_api, judge_to_row


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def score_one(rec, cfg):
    c = rec["cell"]
    lang = c["lang"]
    row = dict(job_id=rec["job_id"], model=rec["model"], cell_id=rec["cell_id"],
               paraphrase=rec["paraphrase"], replicate=rec["replicate"],
               lang=lang, perf=c["perf"], domain=c["domain"], polars=c["polars"],
               recognition=c["recognition"], placebo=c["placebo"],
               error=rec.get("error"))

    if rec.get("error") or not rec.get("completion"):
        row.update(dict(extract_kind="none", interface_compiles=0,
                        closure_capable=None, needs_manual_review=1))
        return row

    ex = extract_interface(rec["completion"], lang)
    row["extract_kind"] = ex["kind"]
    row["n_blocks"] = ex["n_blocks"]
    code = ex["code"]

    # ground-truth compiler battery (OCaml only)
    if lang == "ocaml":
        try:
            s = score_mli(code, ocamlfind=cfg["scoring"]["ocaml_compiler"],
                          timeout=cfg["scoring"]["ocaml_timeout_s"])
        except RuntimeError as e:
            s = OcamlScore(needs_manual_review=1, compiler_error=str(e))
        row.update(score_to_row(s))
        # the canonical outcome for OCaml is the compiler verdict
        if s.interface_compiles:
            row["closure_capable"] = s.closure_capable
        else:
            row["closure_capable"] = None    # don't score; flag for review

    # LLM judge (all non-ocaml; and ocaml too if configured, for calibration)
    if lang != "ocaml" or cfg["scoring"].get("judge_on_ocaml_too"):
        d = judge_api(code, lang, model=cfg["scoring"]["judge_model"])
        row.update(judge_to_row(d))
        if lang != "ocaml":
            row["closure_capable"] = row["judge_closure_capable"]
            row["needs_manual_review"] = int(d.get("confidence", 0) < 0.5)

    return row


def main(cfg):
    files = sorted(glob.glob(os.path.join(cfg["paths"]["raw_dir"], "*.json")))
    print(f"scoring {len(files)} completions...")
    rows = []
    for i, fp in enumerate(files, 1):
        with open(fp) as f:
            rec = json.load(f)
        rows.append(score_one(rec, cfg))
        if i % 25 == 0:
            print(f"  {i}/{len(files)}")
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(cfg["paths"]["scored_csv"]), exist_ok=True)
    df.to_csv(cfg["paths"]["scored_csv"], index=False)
    print(f"wrote {cfg['paths']['scored_csv']}  ({len(df)} rows)")
    # quick console summary
    oc = df[df.lang == "ocaml"]
    if len(oc):
        usable = oc[oc.closure_capable.notna()]
        print("\nOCaml closure_capable rate by cue (usable samples):")
        for cue in ("perf", "domain", "polars", "recognition", "placebo"):
            print(f"  {cue}:")
            print(usable.groupby(cue).closure_capable.mean().to_string())
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    main(load_config(args.config))
