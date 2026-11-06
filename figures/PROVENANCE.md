# Figure provenance

Several figures in this directory read the raw T15 HDF5 recordings directly
(`data/raw/hdf5_data_final`) or other files under `data/`, none of which are
committed (`CLAUDE.md`: never commit anything under `data/`). The underlying
dataset is de-identified single-participant data, publicly available from its
own Dryad deposit (https://doi.org/10.5061/dryad.dncjsxm85, `docs/DATASET.md`
section 1) rather than redistributed through this repository. A third party
who obtains the dataset from Dryad and runs this project's data pipeline
(`scripts/00_verify_data.py` and `src/b2t_drift/data/`) can regenerate those
figures; without it, the committed PNG is the only copy.

| Figure | Generating script | Inputs | Reproducible without T15 data? | Referenced from |
|---|---|---|---|---|
| `attribution_overview.png` | `scratch/task_f_combined_figure.py` (untracked) | raw T15 HDF5; gitignored `scratch/*.npz` caches; `results/s05/per_session_metrics.csv` | no, needs raw HDF5 | unreferenced |
| `attribution_verdict.png` | `scratch/task_f_attribution_verdict.py` (untracked) | raw T15 HDF5; `splits/temporal_s01_s05_switchboard.json`; gitignored `scratch/tuning_matrices/*.npz`; `results/s05/per_session_metrics.csv` | no, needs raw HDF5 | unreferenced |
| `blocks_per_session.png` | `scratch/investigate_blocks_and_checkpoint.py` (untracked) | `data/interim/session_manifest.csv`; `data/external/t15_pretrained_rnn_baseline/checkpoint/*` | no, needs uncommitted `data/` | `docs/FINDINGS.md:152` |
| `corpus_composition_over_time.png` | `tools/investigate_corpus_composition.py` (tracked) | `data/external/t15_copyTaskData_description.csv`; `data/interim/session_manifest.csv` | no, needs uncommitted `data/` | `docs/FINDINGS.md:67,2487` |
| `drift_curve.png` | `scripts/04_drift_curve.py` (tracked) | `results/{s05,s10,s15,s20}/per_session_metrics.csv` (committed; `scripts/03_evaluate.py` output) | yes | `README.md:247,250`; `docs/FINDINGS.md:1258,1341` |
| `error_breakdown.png` | `scripts/04_drift_curve.py` (tracked) | same as `drift_curve.png` | yes | `docs/FINDINGS.md:1341` |
| `feature_layout_correlation.png` | `scratch/investigate_feature_layout.py` (untracked) | raw T15 HDF5 | no, needs raw HDF5 | `docs/FINDINGS.md:42` |
| `headline.png` | `tools/headline_figure.py` (tracked) | `results/drift_curve.csv`, `results/recalibration_recovery_fraction.csv` (both committed) | yes | `README.md:28,251` |
| `normalization_unit_check.png` | `scratch/investigate_normalization_unit.py` (untracked) | raw T15 HDF5 | no, needs raw HDF5 | `docs/FINDINGS.md:171` |
| `recalibration_arms.png` | `tools/task3_make_figure.py` + `tools/task3_analysis_tables.py` (tracked) | `results/recalibration_grid.csv` (committed) | yes | `docs/FINDINGS.md:1772` |
| `recalibration_recovery_fraction.png` | `tools/recovery_figure.py` (tracked) | `results/recalibration_grid.csv` (committed) | yes | `docs/FINDINGS.md:1825,2115,2947` |
| `sentence_length_over_time.png` | `scratch/investigate_sentence_length.py` (untracked) | raw T15 HDF5; `data/external/t15_copyTaskData_description.csv` | no, needs raw HDF5 | `docs/DESIGN.md:361` |
| `subspace_angles.png` | `scratch/task_d_subspace_geometry.py` (untracked) | raw T15 HDF5; `splits/temporal_s01_s05_switchboard.json`; `results/s05/per_session_metrics.csv` | no, needs raw HDF5 | `docs/FINDINGS.md:1692` |
| `switchboard_trials_per_session.png` | `scratch/switchboard_survivor_table.py` (untracked; confirmed byte-identical producer -- `scratch/investigate_switchboard_viability.py` is an earlier, non-matching draft) | raw T15 HDF5; `data/external/t15_copyTaskData_description.csv` | no, needs raw HDF5 | `docs/FINDINGS.md:231` |
| `tuning_correlation_decay.png` | `scratch/task_bc_correlation_analysis.py` (untracked) | raw T15 HDF5; gitignored `scratch/*.npz` caches | no, needs raw HDF5 | `docs/FINDINGS.md:1647` |
| `tx_sbp_pair_correlation.png` | `scratch/task1_tx_sbp_pair_correlation.py` (untracked) | raw T15 HDF5 | no, needs raw HDF5 | `docs/FINDINGS.md:1752` |

Six other figures (`channel_mortality.png`, `channels_below_2hz_vs_date.png`,
`saturation_vs_date.png`, `saturation_vs_firing_rate.png`,
`tx_vs_sbp_crossing_channels.png`, `tx_vs_sbp_degradation.png`) were deleted
as unreferenced internal diagnostics; their generating scripts remain in
`scratch/` if needed again.

`drift_curve.png`/`error_breakdown.png` also read an optional sibling
`evaluation_summary.json` per input (not committed) for the training-val-PER
reference line and nicer panel labels; `scripts/04_drift_curve.py` falls back
to the directory name and omits that reference line if it's absent, so this
does not affect reproducibility from the committed CSVs alone.
