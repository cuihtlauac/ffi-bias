#!/usr/bin/env python3
"""
lab/cost.py — tally measured token cost from a pilot and extrapolate to a full run.

Run a cheap pilot first (real API calls — needs a key):

    python run_experiment.py --plan core --n-per-cell 1
    python score.py
    python lab/cost.py

It reads the per-call token usage recorded in out/raw/*.json (generation) and
out/scored.csv (judge), applies a pricing table, and prints the pilot total plus
a linear extrapolation to the full balanced_ocaml / full plans.

Pricing is a CACHED table (USD per 1M tokens) — confirm current rates before
quoting a number. Gemini free tier is $0. Use --in-price/--out-price to override.
"""

import os, sys, json, glob, argparse

# USD per 1M tokens (input, output). Verify before quoting — see the claude-api skill.
PRICES = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5": (10.0, 50.0),
    "gemini": (0.0, 0.0),          # free tier
}

# Full-run call counts = cells * paraphrases * n_per_cell (generation), and the
# judge runs once per completion when judge_on_ocaml_too is set.
PLAN_CELLS = {"core": 12, "balanced_ocaml": 37, "full": 160}


def price_for(model):
    if not model:
        return None
    for prefix, p in PRICES.items():
        if model.startswith(prefix):
            return p
    return None


def _avg(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return (sum(vals) / len(vals)) if vals else 0.0


def main(raw_dir, scored_csv, paraphrases, in_price, out_price):
    # --- generation: from out/raw provenance -------------------------------
    gen_in, gen_out, gen_model = [], [], None
    for fp in glob.glob(os.path.join(raw_dir, "*.json")):
        rec = json.load(open(fp))
        prov = rec.get("provenance") or {}
        if rec.get("error"):
            continue
        gen_in.append(prov.get("input_tokens"))
        gen_out.append(prov.get("output_tokens"))
        gen_model = gen_model or prov.get("model")
    n_gen = len([x for x in gen_in if x is not None])
    if not n_gen:
        sys.exit(f"no usage found in {raw_dir}/*.json — run a pilot first "
                 f"(run_experiment.py + score.py) with a recent api_client that records usage.")
    avg_gi, avg_go = _avg(gen_in), _avg(gen_out)

    # --- judge: from scored.csv --------------------------------------------
    avg_ji = avg_jo = 0.0
    n_judge = 0
    judge_model = None
    if os.path.exists(scored_csv):
        import csv
        ji, jo = [], []
        rows = list(csv.DictReader(open(scored_csv)))
        for r in rows:
            v = r.get("judge_input_tokens")
            if v not in (None, "", "nan"):
                ji.append(float(v)); jo.append(float(r.get("judge_output_tokens") or 0))
        n_judge = len(ji)
        avg_ji, avg_jo = _avg(ji), _avg(jo)

    # --- prices ------------------------------------------------------------
    gp = (in_price, out_price) if in_price is not None else (price_for(gen_model) or (0, 0))
    # judge model isn't in scored.csv; price it at the generation model's rate unless overridden
    jp = (in_price, out_price) if in_price is not None else gp

    def dollars(n, ai, ao, price):
        return (n * ai * price[0] + n * ao * price[1]) / 1_000_000

    pilot_gen = dollars(n_gen, avg_gi, avg_go, gp)
    pilot_judge = dollars(n_judge, avg_ji, avg_jo, jp)

    print(f"PILOT (measured)")
    print(f"  generation: {n_gen} calls   avg in/out = {avg_gi:.0f}/{avg_go:.0f} tok   "
          f"model={gen_model}   price={gp}  ->  ${pilot_gen:.4f}")
    print(f"  judge:      {n_judge} calls   avg in/out = {avg_ji:.0f}/{avg_jo:.0f} tok"
          f"   price={jp}  ->  ${pilot_judge:.4f}")
    print(f"  pilot total: ${pilot_gen + pilot_judge:.4f}")

    print(f"\nEXTRAPOLATION (avg tokens x full call counts; paraphrases={paraphrases})")
    print(f"  {'plan':<16}{'gen calls':>10}{'judge calls':>12}{'gen $':>10}{'judge $':>10}{'total $':>10}")
    for plan, cells in PLAN_CELLS.items():
        # full run uses config n_per_cell (default 40); judge once per completion
        for n in (40,):
            gcalls = cells * paraphrases * n
            jcalls = gcalls if n_judge else 0
            g = dollars(gcalls, avg_gi, avg_go, gp)
            j = dollars(jcalls, avg_ji, avg_jo, jp)
            print(f"  {plan+' (n=40)':<16}{gcalls:>10}{jcalls:>12}{g:>10.2f}{j:>10.2f}{g+j:>10.2f}")
    print("\nNote: linear in n_per_cell. Pricing is cached — confirm current rates "
          "(claude-api skill) before quoting.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="out/raw")
    ap.add_argument("--scored", default="out/scored.csv")
    ap.add_argument("--paraphrases", type=int, default=4, help="paraphrases per cell (config).")
    ap.add_argument("--in-price", type=float, default=None, help="override $/1M input tokens.")
    ap.add_argument("--out-price", type=float, default=None, help="override $/1M output tokens.")
    args = ap.parse_args()
    main(args.raw_dir, args.scored, args.paraphrases, args.in_price, args.out_price)
