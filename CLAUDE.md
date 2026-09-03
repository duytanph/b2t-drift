# CLAUDE.md

Guidance for working in this repository.

## Project layout

- Source lives in `src/b2t_drift/` and is installed editable via `uv sync`.
- Run all commands with `uv run` (e.g. `uv run pytest`, `uv run scripts/02_train.py`).
- `scripts/` are thin CLI wrappers: they parse a config, call library code in
  `src/b2t_drift/`, and do nothing else. No logic belongs in `scripts/`.

## Data and run artifacts

- Never commit anything under `data/` or `runs/`.
- Split manifests are versioned JSON artifacts in `splits/`, never computed
  inline in a script or notebook.
- Temporal splits only — never shuffle sessions across time. Session order
  matters for this project; a random train/test split across sessions would
  invalidate the drift analysis.

## Known pitfalls

- `collate_trials` returns `input_lengths` BEFORE any model-side temporal
  downsampling. Any model that strides, patches, or pools the time axis
  MUST divide `input_lengths` accordingly before passing them to
  `nn.CTCLoss`. Getting this wrong produces a model that trains and
  converges while emitting garbage, with no error raised. This is the
  most likely location for a silent bug in the pipeline.

## Design philosophy

- No premature abstraction: no registries, no `BaseTrainer`, no plugin
  systems. One model family, one dataset. Concrete over flexible.
- Do not invent dataset details. The HDF5 structure has not been inspected
  yet — treat any assumption about field names, shapes, or session metadata
  as a TODO until confirmed against the real files.
