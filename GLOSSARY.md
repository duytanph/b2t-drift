# Glossary

Working definitions for terms used throughout this repo, followed by a
record of dataset facts established by direct inspection of
`data/raw/hdf5_data_final/`.

## Terms

- **Session**: A single recording day, one subdirectory under
  `data/raw/hdf5_data_final/`, named `t15.YYYY.MM.DD`. This project studies
  model behavior across 45 sessions.
- **Block**: A contiguous sub-unit of a recording session, identified by
  the `block_num` trial attribute. Sessions are partitioned into blocks;
  the official train/val/test split assigns whole blocks to `train`, and
  splits the remaining held-out block(s) at the trial level between `val`
  and `test` (see Dataset facts below).
- **Trial**: A single attempted utterance within a session, stored as one
  `trial_NNNN` HDF5 group. `trial_num` is per-file (restarts at 0 in each
  of `data_train.hdf5` / `data_val.hdf5` / `data_test.hdf5`), not a global
  identifier.
- **n_time_steps**: HDF5 trial attribute giving the number of 20 ms time
  bins in that trial's `input_features` array. `input_features` is
  variable-length and not padded, so this is also
  `input_features.shape[0]`.
- **seq_len**: HDF5 trial attribute giving the true (unpadded) length of
  `seq_class_ids` for that trial. Only present when labels are released
  for the partition (absent on `test`).
- **Drift**: Degradation in decoding performance as a function of distance
  (in time or session count) between training data and evaluation data.
- **Temporal split**: A train/eval split that assigns whole sessions (or,
  within a session, whole blocks) to each partition ordered by time,
  never shuffling sessions across time. Required for measuring
  generalization to future days rather than measuring i.i.d. performance.
- **Day-specific input layer**: A per-session input transformation (e.g.
  an affine layer) used to adapt a shared decoder to session-specific
  neural statistics. TODO: confirm whether this project ends up using
  per-session day layers, and if so, their exact parameterization.
- **TX (threshold crossings)**: A neural feature counting how many times,
  within a time bin, the raw voltage on a given electrode crosses a
  (typically negative) threshold -- a proxy for multi-unit spiking
  activity. *From published documentation on this dataset family, not
  independently verified against this HDF5 schema* -- the HDF5 files
  carry no per-channel feature-identity metadata.
- **SBP (spike band power)**: A neural feature giving the power of the raw
  signal in the spike-frequency band within a time bin -- a continuous
  complement to TX. Same provenance caveat as TX. Prior work on this
  dataset family (e.g. `WillettTFRecordDataset`, 256 channels) is
  consistent with the 512 `input_features` channels here being 256
  electrodes x {TX, SBP}, but this specific pairing is *not* verified
  from the HDF5 files themselves.
- **CTC (Connectionist Temporal Classification)**: Loss function used to
  train sequence models without frame-level alignment between neural
  features and phoneme labels.
- **Blank token**: The CTC "no output" symbol. For this dataset, verified
  to be label index **0** (see Dataset facts).
- **PER (phoneme error rate)**: Levenshtein edit distance between
  predicted and reference phoneme-index sequences, normalized by
  reference length. The primary metric for this project, computable
  directly from `seq_class_ids`.
- **WER (word error rate)**: Levenshtein edit distance between predicted
  and reference word sequences, normalized by reference word count.
  Standard ASR metric; not directly computable from phoneme output alone
  without a phoneme-to-word decoding step (e.g. a language model).
- **CER (character error rate)**: Levenshtein edit distance between
  predicted and reference character sequences, normalized by reference
  character count. Computable directly from `sentence_label` /
  `transcription` text without going through phonemes.

## Dataset facts

Empirically verified against `data/raw/hdf5_data_final/`, via
`scripts/00_verify_data.py` (session manifest) and
`scratch/investigate_labels.py` (label encoding investigation), as of
2026-08-30. All of the following are direct observations from the data
files, not taken from any published documentation.

**Data inventory**
- 45 sessions, dated 2023-08-11 to 2025-04-13 (spanning 611 days).
- 10,948 trials total: 8,072 train / 1,426 val / 1,450 test.
- `test` partitions are **unlabeled** -- `seq_len` and `sentence_label`
  attributes are entirely absent on test trials. Only the 9,498
  train+val trials are usable for supervised work.
- 4 sessions have only `data_train.hdf5` (no val/test files):
  `t15.2023.08.11`, `t15.2024.03.03`, `t15.2024.04.25`, `t15.2024.04.28`.
- Two large recording gaps: ~58 days (2023-12-29 to 2024-02-25) and ~166
  days (2024-07-28 to 2025-01-10).
- Total labeled speech: 2,792 minutes (train + val, from
  `total_time_bins * 0.02 / 60`).

**Trial structure**
- `input_features`: `(n_time_steps, 512)` float32, variable length, not
  padded. `n_time_steps` ranges 138-1025+ bins at 20 ms/bin; median ~527
  bins (~10.5 s).
- `seq_class_ids`: `(500,) int32`, padded with `0`; real length given by
  the `seq_len` attribute. `seq_len` ranges 5-47, median ~16.
- `transcription`: `(500,) int32`, ASCII character codes, null-terminated
  (0), then 0-padded to length 500. Verified identical (after decoding)
  to the `sentence_label` attribute in 200/200 sampled trials ->
  **redundant with `sentence_label`; do not load `transcription` in the
  data pipeline.**

**Label encoding**
- `seq_class_ids` values inside `seq_len` are exactly the integers
  `1..40` (40 distinct classes); `0` never appears inside `seq_len`.
- `0` serves as both the CTC blank token and the padding value in
  `seq_class_ids` -- they are indistinguishable without `seq_len`.
- CTC output layer size is therefore **41** (blank + 40 classes), not 40.
- The integer -> phoneme-string mapping is **not present** in the HDF5
  files, but is now known (sourced from the Brain-to-Text '25 Kaggle
  competition data page) and lives in `src/b2t_drift/phonemes.py` as
  `LOGIT_TO_PHONEME`: index 0 is the CTC blank, indices 1-39 are the 39
  standard ARPABET phonemes (alphabetical), and index 40 is a silence
  token.

**Block structure**
- In all 41 three-file sessions, `val` and `test` draw from an
  **identical** `block_num` set, with **zero** overlap with `train`'s
  block set. `val` and `test` are trial-level splits of the *same*
  held-out block(s) -- they are not independent samples of the session.
- The held-out block(s) are not necessarily the last block(s) of the
  session (observed: session `t15.2023.08.13` has train blocks
  `[1,2,3,4,5,6,7,11,12]` and val/test blocks `[8,9]`, which sit in the
  middle of the block sequence). The official split therefore
  interpolates temporally even within a single day, and cannot by itself
  be used to measure cross-day drift.
- `trial_num` restarts at 0 independently in each of the three files; it
  is per-file, not a global trial identifier.

## Known unknowns

- **Whether the pretrained baseline checkpoint can be evaluated with our
  code.** Not yet determined -- depends on whether our model
  architecture, feature preprocessing, and phoneme vocabulary ordering
  match what the official baseline expects.
