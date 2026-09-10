# b2t-drift

Measuring and mitigating cross-session degradation of an intracortical
speech decoder on the Brain-to-Text '25 (T15) dataset. T15 spans 45
recording sessions over 20 months; this project trains a GRU/CTC phoneme
decoder on early sessions and studies how its accuracy changes as
evaluation data moves further in time from the training data, with the
eventual goal of characterizing that degradation and testing cheap ways to
recalibrate the decoder to a new session.

## Research questions

1. **Measurement** — How does phoneme error rate change as a function of
   the time (or session count) between training data and evaluation data?
2. **Attribution** — What drives that degradation: nonstationary neural
   statistics, vocabulary/sentence coverage differences across sessions,
   or something else?
3. **Recalibration budget** — How much labeled data from a new session
   (how many trials, blocks, or minutes) is needed to recover most of the
   lost accuracy, and what does that cost/accuracy tradeoff look like?

## Setup

```bash
uv sync
```

Requires Python >=3.11 and a CUDA-capable GPU (see `pyproject.toml` for the
PyTorch CUDA index configuration).

Plain `uv sync` installs only the base dependencies and yields **177
passed / 1 failed / 1 skipped** under `uv run pytest` — one test and one
skip require the optional `tensorboard` extra. To match the documented
**179-passed** baseline, run:

```bash
uv sync --extra tensorboard   # or --extra tracking for wandb+tensorboard+mlflow
```

### Data

This repo does not include the dataset. `data/` is gitignored and must be
populated separately:

1. Download the T15 HDF5 release from Dryad:
   https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq
2. Extract it so the layout matches:

```
data/raw/hdf5_data_final/
├── t15.2023.08.11/
│   └── data_train.hdf5
├── t15.2023.08.13/
│   ├── data_train.hdf5
│   ├── data_val.hdf5
│   └── data_test.hdf5
└── ... (45 session directories total)
```

3. Run `uv run scripts/00_verify_data.py` to build the session manifest at
   `data/interim/session_manifest.csv` and sanity-check the download.

See `CLAUDE.md` for repository conventions and `GLOSSARY.md` for dataset
facts and terminology.

## Layout

- `src/b2t_drift/` — library code
- `scripts/` — thin CLI entry points
- `configs/` — YAML configs for data, models, and experiments
- `splits/` — versioned session split manifests (JSON)
- `data/` — local data (gitignored, obtained from Dryad — see Setup)
- `runs/` — training run outputs (gitignored)
- `figures/` — generated plots

## Status

Environment, data inventory, and label-encoding verification complete (see
`GLOSSARY.md` "Dataset facts"); pipeline built and validated; architecture
settled (`docs/FINDINGS.md`: "Day layers are unstable when trained jointly
across many sessions"). The cut-point drift sweep has not been run yet.

**On absolute PER.** Our pipeline reaches val PER **0.296** on the official
split, against the published baseline's **0.101**. The remaining gap is
attributable to a deliberately smaller architecture (3 GRU layers × 512
hidden vs. the baseline's 5 × 768) and to remaining implementation
differences — not to a pipeline defect: the gradient-clipping audit found
no error, and the single-batch overfit gate and phoneme-decode tests pass.
The drift results this project exists to produce are **relative**
measurements — `effective_signal` (real vs. shuffled-input PER) and the
day-layer recalibration gap — and do not depend on matching the published
absolute PER.
