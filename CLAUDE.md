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
- Do not invent dataset details. Confirm field names, shapes, and session
  metadata against `docs/DATASET.md` or the real files before relying on
  them.

## Documentation

`docs/DESIGN.md` holds durable decisions and their justifications. It
states what is true now and why, with no chronology. Target length is
under 400 lines; if it exceeds that, content belongs elsewhere. When a
decision changes, replace the old text with the new decision and its
measured justification, plus a one-line pointer to the FINDINGS.md entry
recording the change. Do not narrate the revision here. DESIGN.md states
what is true now; FINDINGS.md is the append-only record of how it came
to be true. If DESIGN.md exceeds 400 lines, the excess is almost always
revision narrative that belongs in FINDINGS.md.

`docs/FINDINGS.md` holds the dated empirical record — investigations,
measurements, and what each changed. Append-only. Entries are titled by
what was found, never by task number.

`docs/DATASET.md` holds facts about the data itself.

When recording work: a DECISION goes to DESIGN.md, a MEASUREMENT goes to
FINDINGS.md. Most work produces both — record the decision briefly in
DESIGN.md and link to the FINDINGS.md entry for the evidence. Never
record local machine state, drive paths, or environment troubleshooting
in either; those go to README.md if reproducible and nowhere if not.
