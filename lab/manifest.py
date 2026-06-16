#!/usr/bin/env python3
"""
lab/manifest.py — emit a reproducibility manifest for a completed run.

This is the provenance a recorded experiment's record.md should reference. It
reads an out/ directory (default ./out) produced by run_experiment.py (+ score.py)
and prints a JSON manifest to stdout. Persisting is a human act (lab-notebook
rule 2): pipe this into the run-scoped results dir WHEN YOU DECIDE to record:

    python3 lab/manifest.py out > lab/results/2026-06-16-balanced_ocaml-<sha>/manifest.json

Captures, so two runs can be proven driven identically except for the model:
  * git commit (+ dirty flag) and a hash of the outcome-determining source/config
  * providers, requested models, and the SERVED model version (alias drift!) plus
    the exact sampling params sent — pulled from each completion's `provenance`
  * OCaml-oracle + Python toolchain versions, the run seed, completion counts
  * a content hash over the whole raw dataset, so the archive can be verified later

Deliberately omits hostname and absolute paths (kernel/OS/arch are fine — they are
routine reproducibility provenance).
"""

import os, sys, json, glob, hashlib, subprocess, platform


def _run(cmd):
    """Best-effort subprocess capture; None on any failure (tool missing, timeout)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return r.stdout.strip() or None
    except Exception:
        return None


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build(out_dir="out", repo="."):
    raw_dir = os.path.join(out_dir, "raw")
    files = sorted(glob.glob(os.path.join(raw_dir, "*.json")))

    # --- code/config identity ---------------------------------------------
    commit = _run(["git", "-C", repo, "rev-parse", "HEAD"])
    dirty = bool(_run(["git", "-C", repo, "status", "--porcelain"]))
    # hash the files that actually determine the outcome
    cfg = hashlib.sha256()
    for cf in ("config.yaml", "prompts.py", "score_ocaml.py", "api_client.py"):
        p = os.path.join(repo, cf)
        if os.path.exists(p):
            cfg.update(cf.encode()); cfg.update(_sha256_file(p).encode())
    config_sha256 = cfg.hexdigest()

    # --- aggregate per-call provenance + content-hash the raw dataset ------
    providers, requested, served, sampling = set(), set(), set(), set()
    n_ok = n_err = 0
    archive = hashlib.sha256()
    for fp in files:
        archive.update(_sha256_file(fp).encode())   # order is stable (sorted glob)
        try:
            rec = json.load(open(fp))
        except Exception:
            n_err += 1
            continue
        n_err += 1 if rec.get("error") else 0
        n_ok += 0 if rec.get("error") else 1
        prov = rec.get("provenance") or {}
        if prov.get("provider"):      providers.add(prov["provider"])
        if prov.get("model"):         requested.add(prov["model"])
        if prov.get("model_served"):  served.add(prov["model_served"])
        sampling.add((prov.get("temperature_sent"), prov.get("temperature")))

    # --- run knobs ---------------------------------------------------------
    seed = None
    try:
        import yaml
        seed = yaml.safe_load(open(os.path.join(repo, "config.yaml"))) \
                   .get("sampling", {}).get("seed")
    except Exception:
        pass
    pkgs = _run([sys.executable, "-m", "pip", "freeze"]) or ""

    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "config_sha256": config_sha256,
        "providers": sorted(providers),
        "model_requested": sorted(requested),
        "model_served": sorted(served),       # the version actually billed/served
        "sampling_params": [{"temperature_sent": s[0], "temperature": s[1]}
                            for s in sorted(sampling, key=str)],
        "seed": seed,
        "n_completions": len(files),
        "n_ok": n_ok,
        "n_error": n_err,
        "raw_archive_sha256": archive.hexdigest() if files else None,
        "ocaml": _run(["ocamlfind", "ocamlc", "-version"]),
        "python": platform.python_version(),
        "python_packages_sha256": hashlib.sha256(pkgs.encode()).hexdigest(),
        "platform": platform.platform(),      # kernel/OS/arch — no hostname
    }


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "out"
    print(json.dumps(build(out), indent=2))
