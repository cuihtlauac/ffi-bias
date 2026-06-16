# Lab notebook — closure-bias-probe

Append-only index of recorded experimental runs. One row per recorded run, newest
first; the row links to that run's directory under `lab/results/`. **No analysis
lives here** — it's an index. See `/home/.../skills/lab-notebook` for the discipline.

## Conventions for this project (stochastic LLM probe)

- **A run is sampled, so replication is data.** Re-running a cell is not a redundant
  confirmation — it quantifies variance. Keep repeated runs and report rates with
  Wilson CIs / odds ratios, never a bare number. A result that *fails* to replicate
  is the most important kind to record.
- **Results are immutable, run-scoped directories**, never overwritten. Produce one with
  `score.py --run-id <id> --freeze-raw` then `analyze.py --run-id <id>`, giving
  `lab/results/<id>/` (use `<date>-<plan>-<short-sha>` for `<id>`):
  - `scored.csv` — the analyzable dataset
  - `analysis/` — rates, odds ratios, calibration, summary
  - `manifest.json` — provenance (auto-written by `score.py --run-id`)
  - `raw/` — the frozen raw completions (`--freeze-raw`); the manifest's
    `raw_archive_sha256` verifies them. Omit `--freeze-raw` only for very large runs
    you'd rather archive out-of-repo, keeping just the hash.
- Both scripts **refuse to overwrite** an existing record — re-recording needs a fresh
  `--run-id`. The plain `out/` scratch (no `--run-id`) stays freely overwritable.
- **Persistence is a human decision.** The drivers print; you decide what's worth a row here.

## Index

| date | plan | model (served) | n/cell | headline finding | record |
|------|------|----------------|--------|------------------|--------|
| _(no runs recorded yet)_ | | | | | |
