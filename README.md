# b2t-drift

Studying how an intracortical speech-decoding model degrades across recording
sessions spanning 20 months, using the Brain-to-Text '25 (T15) dataset
(512 neural features per 20 ms bin, 45 sessions).

Status: scaffolding only. No data has been inspected and no models have been
trained yet.

## Setup

```bash
uv sync
```

See `CLAUDE.md` for repository conventions.

## Layout

- `src/b2t_drift/` — library code
- `scripts/` — thin CLI entry points
- `configs/` — YAML configs for data, models, and experiments
- `splits/` — versioned session split manifests (JSON)
- `data/` — local data (gitignored)
- `runs/` — training run outputs (gitignored)
- `figures/` — generated plots
