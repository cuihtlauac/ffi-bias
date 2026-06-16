# Lab notebook — closure-bias-probe

Append-only index of recorded experimental runs. One row per recorded run, newest
first; the row links to that run's directory under `lab/results/`. **No analysis
lives here** — it's an index. See `/home/.../skills/lab-notebook` for the discipline.

## Conventions for this project (stochastic LLM probe)

- **A run is sampled, so replication is data.** Re-running a cell is not a redundant
  confirmation — it quantifies variance. Keep repeated runs and report rates with
  Wilson CIs / odds ratios, never a bare number. A result that *fails* to replicate
  is the most important kind to record.
- **Results are immutable, run-scoped directories**, never overwritten:
  `lab/results/<date>-<plan>-<short-sha>/` containing
  - `record.md` — filled from `lab/results/TEMPLATE.md`
  - `manifest.json` — provenance, emitted by `python3 lab/manifest.py out > .../manifest.json`
  - the raw/scored archive (or its location + the `raw_archive_sha256` from the manifest
    if too large to keep in-repo).
- **`score.py` / `analyze.py` overwrite `out/`** in place today. Before recording a run,
  copy/freeze its `out/` into the run directory (or archive it elsewhere and record the
  hash) so the record can't be clobbered by the next run.
- **Persistence is a human decision.** The drivers print; you decide what's worth a row here.

## Index

| date | plan | model (served) | n/cell | headline finding | record |
|------|------|----------------|--------|------------------|--------|
| _(no runs recorded yet)_ | | | | | |
