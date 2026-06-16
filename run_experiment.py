"""
run_experiment.py — expand the plan into (cell × paraphrase × replicate) jobs,
query the model, and save one JSON per completion under out/raw/.

Usage:
  export ANTHROPIC_API_KEY=...
  python run_experiment.py --config config.yaml --plan core
  python run_experiment.py --config config.yaml --plan core --dry-run   # just print the matrix
"""

import os, json, argparse, hashlib, random
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

from prompts import plan_cells, expand_cell, Cell
from api_client import call_model_meta


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_jobs(cfg, plan):
    cells = plan_cells(plan, cfg["grid"])
    paraphrases = cfg["sampling"]["paraphrases"]
    n = cfg["sampling"]["n_per_cell"]
    jobs = []
    for model in cfg["models"]:
        for cell in cells:
            for para in paraphrases:
                prompt = expand_cell(cell, para)
                for rep in range(n):
                    jid = hashlib.sha1(
                        f"{model}|{cell.id()}|{para}|{rep}".encode()).hexdigest()[:16]
                    jobs.append(dict(job_id=jid, model=model, cell=asdict(cell),
                                     cell_id=cell.id(), paraphrase=para,
                                     replicate=rep, prompt=prompt))
    return jobs


def run(cfg, plan, dry_run=False, n_per_cell=None):
    if n_per_cell is not None:           # CLI override, e.g. --n-per-cell 1 for a cost pilot
        cfg["sampling"]["n_per_cell"] = n_per_cell
    raw_dir = cfg["paths"]["raw_dir"]
    os.makedirs(raw_dir, exist_ok=True)
    jobs = build_jobs(cfg, plan)
    random.Random(cfg["sampling"]["seed"]).shuffle(jobs)  # spread cells over time

    print(f"plan={plan}  cells={len(plan_cells(plan, cfg['grid']))}  "
          f"paraphrases={len(cfg['sampling']['paraphrases'])}  "
          f"n_per_cell={cfg['sampling']['n_per_cell']}  "
          f"models={len(cfg['models'])}  ->  {len(jobs)} total completions")
    if dry_run:
        # show one example prompt per distinct cell
        seen = set()
        for j in jobs:
            if j["cell_id"] in seen:
                continue
            seen.add(j["cell_id"])
            c = j["cell"]
            print(f"\n--- cell {j['cell_id']}  ({c['lang']}, perf={c['perf']}, "
                  f"domain={c['domain']}, polars={c['polars']}, "
                  f"recog={c['recognition']}, placebo={c['placebo']}) [{j['paraphrase']}]")
            print(j["prompt"])
        return

    temp = cfg["api"].get("temperature")   # None/absent -> omitted (required for Opus 4.8/4.7)
    maxtok = cfg["api"]["max_tokens"]
    def _needs_run(j):
        # Run if the file is missing OR a previous attempt recorded an error, so a
        # second pass recovers jobs lost to transient throttling (429s) while
        # leaving successful completions untouched (keeps the run idempotent).
        p = os.path.join(raw_dir, j["job_id"] + ".json")
        if not os.path.exists(p):
            return True
        try:
            with open(p) as f:
                return bool(json.load(f).get("error"))
        except Exception:
            return True          # unreadable/partial -> redo
    todo = [j for j in jobs if _needs_run(j)]
    print(f"{len(jobs) - len(todo)} already done; running {len(todo)}")

    def work(j):
        try:
            txt, meta = call_model_meta(model=j["model"], user=j["prompt"],
                                        temperature=temp, max_tokens=maxtok)
            j["completion"] = txt
            j["provenance"] = meta   # provider, resolved model, endpoint, exact params sent
            j["error"] = None
        except Exception as e:  # noqa: BLE001 — record and continue
            j["completion"] = ""
            j["provenance"] = None
            j["error"] = repr(e)
        with open(os.path.join(raw_dir, j["job_id"] + ".json"), "w") as f:
            json.dump(j, f)
        return j["job_id"], j["error"]

    done = 0
    with ThreadPoolExecutor(max_workers=cfg["sampling"]["max_concurrency"]) as ex:
        futs = [ex.submit(work, j) for j in todo]
        for fut in as_completed(futs):
            jid, err = fut.result()
            done += 1
            flag = "ERR" if err else "ok"
            if done % 10 == 0 or err:
                print(f"[{done}/{len(todo)}] {jid} {flag}{(' '+err) if err else ''}")
    print("done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--plan", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n-per-cell", type=int, default=None,
                    help="override sampling.n_per_cell (use 1 for a cheap cost pilot).")
    args = ap.parse_args()
    cfg = load_config(args.config)
    plan = args.plan or cfg.get("default_plan", "core")
    run(cfg, plan, dry_run=args.dry_run, n_per_cell=args.n_per_cell)
