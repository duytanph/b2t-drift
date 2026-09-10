# Findings — the empirical record

This file is the dated, append-only record of investigations and
measurements behind `docs/DESIGN.md`'s decisions. It grows as
investigations complete; entries are titled by **what was found**, never
by task number, and are never rewritten in place — a later correction is
a new entry that cross-references the one it supersedes. Read
`docs/DESIGN.md` for what is true now and why; come here when a specific
claim needs to see the evidence behind it. Investigation scripts live in
`scratch/` (gitignored, not committed); figures referenced below are
committed under `figures/`.

---

### Feature column layout verified empirically (2026-09-01)

**Question:** Are the published 512-column TX/SBP boundary and the
64-wide per-array boundaries correct?

**Method:** A discreteness test (distinct-value count per column, which
survives affine z-scoring) located the TX/SBP transition. A boundary-shift
scan — testing all integer shifts of the three internal 64-wide
boundaries from −16 to +16 against within- vs. across-block correlation —
tested the array boundaries.

**Result:** TX/SBP transition confirmed exactly at column 256 (columns
0–255: ~0.6% as many distinct values as samples; 256–511: ~76%). 64-wide
array boundaries (64/128/192) confirmed: the correlation gap peaks
exactly at shift 0 (0.0129) vs. next-best neighbors (0.0127, 0.0125),
falling off monotonically outward — the official README's literal
0–64/65–128/... boundaries (off by one) are not supported. All 8
hypothesized 64-wide blocks have per-column mean ≈ 0, variance ≈ 1
(expected under normalization, not independent boundary evidence).
Unplanned finding: `input_features` are already z-scored — every column
mean ≈ 0, std ≈ 1, and TX columns repeat small sets of exact float
values.

**Consequence:** `docs/DATASET.md` section 7's column table is the
confirmed layout. The z-scoring finding reopened the normalization
question, resolved by two entries below and by author correspondence
(`docs/DATASET.md` section 9 item 7). Figure:
`figures/feature_layout_correlation.png`.

---

### Corpus composition is a serious training/eval confound (2026-09-01)

**Question:** Does sentence-corpus composition vary enough across
sessions to confound a drift measurement?

**Method:** Joined `data/external/t15_copyTaskData_description.csv`
block-level corpus labels against session dates.

**Result:** Six corpora, not five (Harvard sentences, 98 trials, was
missing from the dataset reference's original five-corpus list and has
since been added). The planned train block (sessions 1–15) is 86.9%
Switchboard / 8.5% 50-Word / 3.7% Openwebtext, zero Freq words or
Harvard. The 30 held-out sessions are 56.0% Switchboard / 31.2% Freq
words / 10.7% 50-Word / 1.5% Harvard. 8 of 30 held-out sessions have zero
Switchboard trials. Freq words is confined to 2023-10-22 through
2023-12-29; Harvard to 2023-10-20 through 2023-11-03; Openwebtext
entirely inside the train window (2023-09-29 through 2023-10-15). Only
Switchboard and 50-Word recur across the full 20-month span.

**Consequence:** Adopted Switchboard-only as the primary train+eval
analysis (`docs/DESIGN.md` section 7). Figure:
`figures/corpus_composition_over_time.png`.

---

### GPU compute is not a constraint on model size (2026-09-01)

**Question:** Can this machine train the full-size published
architecture, or must model size be cut to fit the GPU?

**Method:** Synthetic-tensor benchmark, RTX 2080 SUPER (8 GB VRAM):
batch=32, input_size=512, output_size=41, CTC loss,
forward+backward+optimizer step, 10 warmup + 50 timed steps, across model
size (baseline 5L/1024H vs. small 3L/512H), sequence length (527 vs. 132
via 4x downsampling), and precision (fp32 vs. fp16).

**Result:**

| model | seq_len | precision | ms/step | peak VRAM | h @ 10k steps |
|---|---|---|---:|---:|---:|
| baseline (5L,1024H) | 527 | fp32 | 461.0 | 3.4 GB | 1.28 |
| baseline (5L,1024H) | 527 | fp16 | 203.9 | 2.0 GB | 0.57 |
| baseline (5L,1024H) | 132 (4x) | fp32 | 120.7 | 1.3 GB | 0.34 |
| baseline (5L,1024H) | 132 (4x) | fp16 | 57.8 | 0.9 GB | 0.16 |
| small (3L,512H) | 527 | fp32 | 104.9 | 1.1 GB | 0.29 |
| small (3L,512H) | 527 | fp16 | 80.3 | 0.7 GB | 0.22 |
| small (3L,512H) | 132 (4x) | fp32 | 28.1 | 0.4 GB | 0.08 |
| small (3L,512H) | 132 (4x) | fp16 | 21.6 | 0.3 GB | 0.06 |

No OOM at batch 32 or 64 in any configuration (batch=64 peak 6.1 GB,
baseline/fp32/no-downsampling). fp16 roughly halves step time (Turing has
no native bf16). The official baseline's real architecture (patched
input, GRU dim 7,168) is not directly comparable to this 512-dim grid.
The fp16 timings above were measured without a `torch.amp.GradScaler` —
acceptable for a pure speed benchmark on random targets, but a real fp16
training run needs one to avoid silent gradient underflow.

**Consequence:** No GPU-driven reason to shrink the model or the
four-cut-point sweep (`docs/DESIGN.md` section 7) — the sweep's real
bottleneck turned out to be the dataloader, not GPU compute (see
"Storage medium, not DataLoader configuration..." below). Confirmed
`scripts/02_train.py` actually constructs a `GradScaler` whenever
`amp: true` (the configured default,
`configs/experiment/switchboard_baseline.yaml`) and `train_one_epoch`
uses it to scale/unscale before clipping and stepping — this benchmark's
missing-GradScaler caveat does not apply to real training runs.

---

### Block count decreases over the recording; pretrained checkpoint uses patched input and per-session layers (2026-09-01)

**Question:** (a) How many blocks per session, and does block count
trend over time? (b) What is the pretrained baseline checkpoint's actual
architecture?

**Method:** (a) Read block counts per session from
`data/interim/session_manifest.csv`. (b) Loaded
`data/external/t15_pretrained_rnn_baseline/checkpoint/best_checkpoint`
and inspected state_dict shapes and `args.yaml`.

**Result:** (a) Median 6 blocks/session (range 3–11), median ~42
trials/block (range ~22–50). Block count correlates negatively with
session order (r = −0.40) — earlier sessions have more blocks
(`t15.2023.08.13`: 11) than later ones (many: 3–6).

(b) `torch.load(weights_only=True)` failed (`UnpicklingError` —
`numpy.core` renamed to `numpy._core`, breaking torch's safe-globals
string match against the official file); fell back to
`weights_only=False` for this specific, documented-source file. 5 GRU
layers, hidden size 768. GRU input size is 7,168, not 512: the official
model patches its input (`patch_size=14, patch_stride=4`, 14×512=7,168,
stepped every 4 bins), matching the 80 ms real-time phoneme rate. Output
size 41, confirming `ctc_output_size=41` independently. Per-session
day-specific input layers confirmed present, as 45 separately-keyed
tensors `day_weights.0..44` (512×512) and `day_biases.0..44` (1×512) —
not found via a naive "first dim == 45" heuristic. `args.yaml` confirms
`dataset.sessions` lists all 45 sessions; `dataset_probability_val`'s 4
zero-flagged sessions match the 4 known train-only sessions. Checkpoint's
recorded `val_PER = 0.101`, `val_loss = 22.45` on the official split.
`use_amp: true`.

**Consequence:** (a) No design change; recorded for the
normalization-exposure caveat (`docs/DESIGN.md` section 11). (b) Confirms
the pretrained checkpoint cannot be used for the drift curve
(`docs/DESIGN.md` section 7); the day-specific-layer architecture
directly informs `DayLayers` and Arm 2 (`docs/DESIGN.md` section 5).
Figure: `figures/blocks_per_session.png`.

---

### Normalization unit not resolved by statistical testing alone (2026-09-01)

**Question:** Is the pre-applied z-scoring computed per block, per
session, or globally?

**Method:** Recomputed per-column mean/std at block, session (train-only
and full), and 3-session-pooled granularity, from 18 blocks across 3
sessions spanning the recording.

**Result:** Per-block: rejected (5th–95th percentile spread ~0.06 across
columns; an exact per-block fit would be ~1e-6). Per-session: tighter but
not confirmed (~0.019–0.026 spread, still an order of magnitude above
float precision). Global: no predicted trend materialized — 6
representative columns across 12 sessions over 20 months stay within
±0.08 of zero with no monotonic drift
(`figures/normalization_unit_check.png`). Confirmed via the official
repo's `dataset.py`/`rnn_model.py` that no further normalization happens
downstream of the HDF5 files, and that `day_weights`/`day_biases` is a
learned parameter (identity/zero init), not a normalization step.

**Consequence:** Statistically inconclusive between session and global;
superseded by the next entry and fully resolved by author correspondence
(`docs/DATASET.md` section 9 item 7).

---

### Normalization is not a simple affine transform of integer counts (2026-09-01)

**Question:** Can the exact per-block (mu, sigma) be recovered by
treating TX columns as integer spike counts on an evenly-spaced lattice?

**Method:** Tested whether each (session, block, column) triple's
distinct released values sit on a lattice of spacing 1/sigma, across 3
sessions × 18 blocks × 256 TX columns (4,608 triples). SBP columns tested
as a continuous-data control.

**Result:** Only 3 of 4,608 triples showed evenly-spaced values, and all
3 are float32-precision artifacts (recovered sigma exactly 2^24 or 2^25,
inverting to absurd counts ~350M). 99.9% showed no constant spacing. The
most "discrete-looking" case (212 distinct values / 40,727 samples)
failed a robustness check: no dominant zero-count value, and
consecutive-difference sizes span ~5 orders of magnitude with no bimodal
gap (best percentile fit: 17.5% of diffs matching an integer multiple,
far short of ~99% for a real lattice). Control: SBP columns show no
lattice either (0/108 cases), confirming the method isn't simply broken.

**Consequence:** Stopped rather than report a spurious mu/sigma.
Explained in full once author correspondence confirmed a different
divisor per SENTENCE (a rolling window) — no shared per-block or
per-session lattice could ever have existed (`docs/DATASET.md` section 9
item 7).

---

### Switchboard survivor count corrected to 17 of 30 sessions (2026-09-01)

**Question:** How many held-out sessions have enough labeled Switchboard
trials (≥50) to be usable eval points?

**Method:** Rebuilt the viability table using per-(session, block) trial
counts read directly from HDF5 (train+val partitions), joined to the
CSV's block-level corpus label — the method the next entry shows is
necessary.

**Result:** 17 of 30 held-out sessions survive (down from an earlier,
less precise 19/30). Day-offset range of survivors unchanged: 5–546 days.
All four requested curve endpoints pass: 2023-10-20 (+5d, 98 trials),
2023-10-22 (+7d, 83), 2025-03-14 (+516d, 83), 2025-04-13 (+546d, 94).
Train block (sessions 1–15): 3,501 labeled Switchboard trials of 3,969
total (88.2%). Two sessions flip from PASS to FAIL under the corrected
method: `2023-11-03` (+19d, actually 25 labeled Switchboard trials, not
the CSV's 50) and `2023-12-29` (+75d, similarly overcounted).

**Consequence:** 17/30 is the authoritative Switchboard survivor count
used everywhere else in this project. Figure (redrawn):
`figures/switchboard_trials_per_session.png`.

---

### Corpus CSV trial counts are unreliable; the manifest (HDF5) is authoritative (2026-09-01)

**Question:** Why did the CSV's per-block trial counts disagree with
actual HDF5 trial counts?

**Method:** Compared CSV block-level counts against actual per-block HDF5
counts, 265 (session, block, split) rows, the same granularity the CSV
itself reports.

**Result:** Disagreement goes both directions (30 rows CSV > actual,
+249 excess; 4 rows actual > CSV, −35; net +214) — a "QC drops during
packaging" explanation predicts only one direction and is incomplete.
Roughly proportional to corpus/partition size, not concentrated
(Switchboard: 24/34 mismatched rows, 174/214 of the discrepancy, close to
its ~69% trial-count base rate). Mismatched blocks skew smaller (mean
32.7 actual trials vs. 42.6 matched) with several suspicious round-number
mismatches (±10, ±5) in both directions. Worked example
(`t15.2023.11.03`): CSV lists block 6 as 50 Switchboard sentences; actual
HDF5 block 6 has 50 trials total across val+test **combined** (25 each)
— the CSV's 50 is right for the block, but only 25 are in the labeled
(val) partition. Confirms partition misattribution as a real, not just
hypothesized, contributor (doesn't explain every mismatch; packaging
drops remain plausible for the remainder). 0/265 blocks had a missing
corpus lookup in the CSV.

**Consequence:** HDF5/manifest is authoritative for trial counts; CSV is
authoritative for block-to-corpus labels only. Correct method: join CSV
corpus label to HDF5 per-block, per-partition trial count. Used to
produce the corrected survivor table above; fixed at least two wrong
PASS/FAIL calls.

---

### Raw firing rates are recoverable per sentence via the value lattice (2026-09-03)

**Question:** Can genuine raw spike counts be recovered from the
released, normalized `input_features`?

**Method:** Per-channel minima distributions (TX vs. SBP) checked
separately, then an exact per-(sentence, channel) lattice test — since
normalization is per-sentence, each sentence has exactly one divisor per
channel, unlike a whole block. Tested on 30 trials × 8 TX channels = 240
pairs (501 trials sampled across 3 sessions).

**Result:** TX and SBP per-channel minima are different distributions
(TX median −0.85, range to −1.44; SBP median −2.12, range to −3.51) — a
previously-cited global minimum of −3.5107 is an SBP column (278), not
TX, so a Poisson spike-count inversion never applied to it. Every TX
channel's minimum recurs >100 times (median 13,207) — plateaus, not
one-off outliers, consistent with genuine zero-count bins. The
per-sentence lattice is confirmed in 230/240 (95.8%) of tested pairs,
giving an exact recovered sigma per sentence and channel (median 0.56,
range 0.23–0.88). Exact recovery: `sigma = 1/spacing`,
`mu = -min*sigma`, `raw = released*sigma + mu` — median implied firing
rate ~24 Hz. A cruder pooled-minimum-plus-Poisson approximation (median
~36 Hz, 83.2% of channels in the 20–100 Hz plausible range) disagrees by
~1.5x and is conceptually unsound (it averages over many
sentence-specific windows as if there were one fixed per-channel scale).

**Consequence:** Use the exact per-(sentence, channel) route for any
firing-rate analysis (`docs/DATASET.md` section 9 item 9); keep the
Poisson figure only as a rough cross-check. Feeds the Week 2
channel-mortality analysis (`docs/DESIGN.md` section 6).

---

### Cut-point sweep manifests built and validated; a 126-day coverage gap (2026-09-03)

**Question:** What do the five split manifests (four Switchboard cut
points + one all-corpora fallback) actually contain once built?

**Method:** Built and validated all five manifests; read trial/session
counts directly from the written manifests.

**Result:**

| cut | cut date | train | val | eval sessions | passing | offset range |
|---|---|---:|---:|---:|---:|---|
| s01–s05 | 2023-08-25 | 908 | 101 | 40 | 27 | +2 to +597 |
| s01–s10 | 2023-09-29 | 2,098 | 235 | 35 | 22 | +2 to +562 |
| s01–s15 | 2023-10-15 | 3,150 | 351 | 30 | 17 | +5 to +546 |
| s01–s20 | 2023-11-17 | 3,336 | 371 | 25 | 15 | +100 to +513 |
| s15 all-corpora | 2023-10-15 | 3,571 | 398 | 30 | 30 | +5 to +546 |

The 17 passing s01–s15 day-offsets are exactly: 5, 7, 133, 145, 152, 154,
208, 243, 278, 280, 287, 453, 455, 516, 518, 532, 546 — a **126-day hole
between +7 and +133** (the "frequent words" corpus era, which doesn't
clear the 50-Switchboard-trial threshold at this cut). s01–s20 is poor
value: +186 training trials (5.9%) over s01–s15, for −5 eval sessions and
the loss of **both** early time points (+5, +7, +133-to-+287 gone
entirely). Corpus-restriction cost at s15: 3,969 (all-corpora) → 3,501
(Switchboard) = **11.8%** less training data (corrects an earlier ~13%
estimate).

**Consequence:** The 126-day gap and the s20 poor-value finding are
recorded as measurement limitations, not engineered around
(`docs/DESIGN.md` section 7). 11.8% is now the single number cited
everywhere this cost is mentioned.

---

### Pre-committed cut-point selection rule (2026-09-03)

**Question:** How will the primary cut point be chosen, decided before
any PER value exists?

**Method:** N/A — a decision, recorded in advance per the validation-set
rule (`docs/DESIGN.md` section 7), before any PER had been computed on
any of the five manifests above.

**Result:** Use the largest cut point whose baseline validation PER is
acceptable, preferring earlier cut points when PER is comparable (within
2 absolute percentage points), on the grounds that earlier cuts give
better curve resolution. If no cut point yields a usable decoder under
Switchboard-only training, fall back to `temporal_s01_s15_allcorpora` and
report the corpus confound as a stated limitation rather than
restricting evaluation.

**Consequence:** Binds the eventual primary-cut selection. Recording the
rule before seeing any PER — not after — is what makes the selection
principled rather than a post-hoc rationalization of whichever cut looks
best.

**Addendum (2026-09-08): the numeric "acceptable" threshold, defined now
while only one cut point (s15) has been trained.** This rule named
"acceptable" and a "2 absolute percentage point" comparable band but never
gave either a number. s15's 100-epoch diagnostic run gives exactly one
anchor -- `best_val_per = 0.353` (epoch 92) -- and this addendum is written
using only that anchor, before s05/s10/s20 have been trained.

- **Acceptable validation PER: ≤ 0.45.** Reasoning against the s15 anchor:
  (1) the margin above the anchor (0.45 − 0.353 = 0.097, ~0.10) is roughly
  11x the measured best-epoch-selection noise (`std(val_per)` over the last
  20 epochs = 0.0091 -- see the timing/noise entry below), so a cut failing
  this threshold is failing for a real reason (underfitting / a floor
  effect from less training data), not selection variance -- see the next
  bullet for where the 0.0091 noise figure comes from; (2) it sits far
  below the diagnostic run's observed "blank phase" ceiling (val PER ~0.83
  in early epochs, before the model commits to any phoneme structure --
  see the earlier diagnostic-run analysis), so a cut at 0.45 is still
  unambiguously a functioning decoder, nowhere near collapse into that
  regime; (3) it does not pre-judge the untrained cuts: s05 (908 training
  trials, 28.8% of s15's 3,150) is expected to be worse than s15 via the
  floor effect `docs/DESIGN.md` section 7 already names as a risk, and a
  tighter threshold (e.g. requiring PER within a few points of 0.353) would
  effectively decide s05/s10 fail before they are even run, defeating the
  purpose of a pre-committed rule that is supposed to be decided ahead of
  the evidence, not shaped to it.
- **The existing 2-percentage-point "comparable" band, read against
  measured selection noise.** `std(val_per)` over the diagnostic run's last
  20 epochs is **0.0091**; best epoch 92 (0.353) vs. final epoch 99 (0.371)
  differ by **0.018**, roughly two standard deviations of that noise. The
  0.02 "comparable" band is therefore about **2.2x** the measured
  selection noise (0.02 / 0.0091 ≈ 2.2) -- tight enough to still mean
  something (a difference this size is unlikely to be pure epoch-selection
  luck) but not so tight that ordinary run-to-run noise would make two
  genuinely-similar cut points look different. Recorded explicitly so a
  reader can see the band was derived from a measurement, not chosen by
  eye.
- **Why this was written before any other cut point had a PER value.**
  Exactly one anchor (s15 = 0.353) exists at the time of writing. Waiting
  until s05/s10/s20 also have PER values before fixing this threshold would
  let those numbers silently influence where the line gets drawn -- the
  same failure mode the original rule (written before any PER existed) was
  designed to prevent. Fixing the number now, against only the s15 anchor,
  keeps the whole rule (both the original "largest acceptable cut" logic
  and this threshold) pre-committed rather than post-hoc.

---

### Storage medium, not DataLoader configuration, was the training bottleneck (2026-09-03 to 2026-09-04)

**Question:** What `num_workers` setting is fastest, and can it be tuned
from a synthetic benchmark?

**Method:** Two rounds of synthetic DataLoader benchmarking, then a real
measurement on the full 3,150-trial `temporal_s01_s15_switchboard` train
set (3 real epochs, `num_workers` in {0, 1, 2, 4}) — first on the
original spinning disk, then again after the dataset was moved to an
SSD.

**Result:** Round 1 (200 synthetic items) picked `num_workers=0` —
wrong: Windows `spawn` worker-startup overhead swamped the signal at
that scale. Round 2 (2,048 synthetic fixed-length items) measured 35
ms/batch at `num_workers=2`, concluding it cleared the ~230 ms/step fp16
compute budget — also wrong at real scale: that same config still spent
74% of steady-state epoch time waiting on the dataloader (217.4s of
292.8s). Disk read before/after the storage change: 70–93 ms/trial vs.
4.43 ms/trial, 371.7 MB/s. Steady-state epoch time at `num_workers=2`
before/after: 292.8s at 74% wait vs. 41.9s at under 1% wait. Post-move
`num_workers` comparison (3rd epoch): 0 → 62.33s (24.8% wait), 1 →
42.92s (0.9%), 2 → 41.86s (0.9%), 4 → 43.12s (1.6%) — 1/2/4 effectively
tied, 0 slower because loading and compute no longer overlap. Peak GPU
memory unchanged at 1749.7 MB across every configuration.

**Consequence:** A benchmark must reproduce the real access pattern —
many distinct files, real variable lengths, no page-cache reuse — or it
measures something else; both synthetic rounds passed a scale test but
failed the access-pattern test. `num_workers=2` is the config default;
I/O is no longer the bottleneck at any `num_workers >= 1`, so no further
(e.g. packed-cache) dataloader work is needed. Extrapolating the
`num_workers=2` rate linearly by train-batch count across all four cut
points: 3.50h for 100 epochs of every cut point, summed — superseding
the earlier 18.76–25.50h estimate and the original GPU-compute-only
"~6 hours" estimate.

---

### Batch composition changed decoded output; fixed by masking padding before smoothing (2026-09-04)

**Question:** Does a trial's decoded phoneme sequence depend on which
other trials happen to share its batch?

**Method:** Compared `log_probs` for one trial decoded alone vs. padded
inside a larger batch, before and after training (`DayLayers`' bias moves
off its zero-init once trained).

**Result:** Before fix: 0.0755 max absolute `log_probs` difference for a
trained model — large enough to flip a marginal argmax. Mechanism:
`DayLayers` runs on the batch's zero-padded region too; once its bias is
nonzero, a padded timestep evaluates to `bias` (not zero), which
`_smooth`'s Gaussian convolution then mixes into the last ~6 real
timesteps of any trial shorter than its batch's max length. A fresh
(identity-init) model showed ~0 difference, pinning the mechanism to
`DayLayers`' bias specifically. After the fix (`GRUCTC.forward` now zeros
every timestep at or past each item's `input_lengths[i]` immediately
after `DayLayers`, before `_smooth`; `input_lengths` is now a mandatory
forward argument): 4.77e-7 (fresh) and 9.54e-7 (trained), both below the
1e-5 threshold
(`tests/test_gru_ctc.py::test_smoothing_batch_composition_sensitivity_below_threshold_after_fix`).

**Consequence:** PER is now provably independent of batch composition
(`docs/DESIGN.md` section 8).

---

### Combined train+eval epoch timing does not yet resolve against the train-only 41.9s figure; instrumentation added (2026-09-08)

**Question:** The s15 100-epoch diagnostic run's `metrics.csv` `seconds`
column averaged 21.3 s/epoch, far below the "Storage medium, not DataLoader
configuration..." entry's 41.9 s/epoch. Is this the same page-cache-warming
effect that entry itself names as a risk, or something else?

**Method:** Read `metrics.csv`'s `seconds` column directly (no retraining).
**This column is `train_seconds + eval_seconds` combined** (`fit()`'s row
dict, `training/loop.py`), while the 41.9s figure above was **train-only**
(`train_one_epoch`, no `evaluate()` call) -- the two were never a like-for-
like comparison, and this could not be resolved from the existing artifact
alone, since `metrics.csv` had no way to separate the two. `training/loop.py`'s
`fit()` now additionally logs `train_seconds` and `eval_seconds` as their own
columns (keeping `seconds` as their sum, unchanged, for anything already
reading it) -- this resolves the ambiguity starting with the NEXT run, not
retroactively for the diagnostic run already on disk.

**Result (from the existing, combined-only column):** epoch 0: 55.68s;
epoch 1: 47.19s; epoch 2: 41.34s; steady-state (epochs 10-99): median
**18.36 s/epoch**, IQR 1.35s (Q1 17.90, Q3 19.25); first epoch within 10% of
that steady-state median: **epoch 22**. The monotonic decline from a cold
55.68s down to a steady ~18.36s over the first ~22 epochs is consistent
with the ~5 GB s15 training set (3,150 trials) gradually becoming
OS-page-cache resident against 16 GB system RAM, mirroring the mechanism
the original entry already documented for the initial spinning-disk-to-SSD
comparison -- but **this cannot yet be stated as confirmed**: because the
column is combined, the same decline could also be explained by eval-side
caching, cuDNN algorithm re-selection settling, or some mix of all three.
**The train-only figure needed to compare directly against the recorded
41.9s is not computable from this run's artifacts and is not reported
here** -- no subtracted or estimated number is given; the next run (which
will have separate `train_seconds`/`eval_seconds` columns) resolves this
directly.

**Consequence:** Revised four-cut sweep wall-clock estimate, using the
measured 18.36 s/epoch **combined train+eval** rate, scaled linearly by
train-trial count (908 / 2,098 / 3,150 / 3,336) at 100 epochs each per
`configs/experiment/switchboard_baseline.yaml`'s new `max_epochs: 100`
(see "Early stopping disabled for the cut-point sweep" above): **~1.54h**,
superseding the previously recorded 3.50h. Flagged explicitly as a
train+eval combined figure, not a train-only one, and therefore not
directly comparable to the 41.9s train-only figure it revises the context
around.

---

### Early stopping disabled for the cut-point sweep (2026-09-08)

**Question:** Is `switchboard_baseline.yaml`'s `patience: 5` (no `min_delta`)
safe to use for the four-cut sweep?

**Method:** Offline replay of `fit()`'s exact early-stopping comparison
(strict `<`, no `min_delta`) against the 100-epoch diagnostic run's real
`val_per` trajectory (`runs/diag_s15_100ep_nostop/metrics.csv`), at
`patience` in `{5, 10, 15}` and `min_delta` in `{0.000, 0.005, 0.010, 0.015,
0.020, 0.030}` -- pure arithmetic over the recorded column, no retraining.

**Result:** Every `patience=5` row stops between epoch 19 and 22 regardless
of `min_delta` -- not threshold-sensitive, since there is a noisy stretch
around epochs 14-22 (a transient grad-norm spike, `n_inf_batches` still 0)
that any short patience walks into. At `min_delta=0.000, patience=5`
(today's config, before this entry): stops at epoch 22, selects epoch 17 at
`val_per=0.5937`, vs. `0.3529` achievable at epoch 92 -- a **+0.241
absolute PER cost**. The boundary is otherwise sharp and run-specific:
`min_delta=0.005` never stops within 100 epochs (selects epoch 92,
`0.3529`), while `min_delta=0.010` at `patience=10` stops at epoch 60
(`0.3954`).

**Consequence:** `configs/experiment/switchboard_baseline.yaml` now sets
`max_epochs: 100, patience: 100` (early stopping effectively disabled --
no 100-epoch run can accumulate 100 epochs-since-improvement). Early
stopping is a compute-saving device; compute is not the binding constraint
here (~1.5h for the full four-cut sweep at the measured steady-state rate,
see the timing entry above), so disabling it costs nothing and removes a
documented, real hazard. `fit()`'s early-stopping comparison itself
(`training/loop.py`) is deliberately left untouched -- no `min_delta` was
added to it. Calibrating a `min_delta` against this single run, from a
single cut point (s15, 3,150 training trials), and then applying it to cut
points ranging 908-3,336 training trials with their own, unmeasured noise
profiles, is not defensible: the sharp boundary above (`0.005` vs. `0.010`
producing a 26-epoch difference in stop point) shows how sensitive the
choice would be to exactly the kind of single-run overfitting this project's
validation philosophy (`CLAUDE.md`) exists to avoid. `checkpoint_best.pt`
already selects the best epoch independent of run length, so there is no
accuracy cost to running the full 100 epochs instead.

---

### Model A's initial result and divergence (2026-09-08)

**Question:** Does the current pipeline, trained on the official split
(all 45 sessions, all corpora), reproduce the published baseline's
`val_PER=0.101` (`docs/FINDINGS.md`: "Block count decreases over the
recording...")?

**Method:** Trained Model A (`docs/DESIGN.md` section 3) on
`splits/official.json`, current architecture unchanged (3 layers, 512
hidden, no patching), 100 epochs, early stopping disabled.

**Result:** Best val PER **0.5009 at epoch 18**, rising to **0.5692 by
epoch 99** -- not a plateau. Train loss itself ROSE from **1.49 to 1.64
between epochs 50 and 99** (not overfitting -- overfitting is falling
train loss with rising val loss; rising train loss is divergence).
`grad_norm_mean` (the mean PRE-clip gradient norm -- see the next entry
for why this is the correct number to read): **283 / 1176 / 349 / 1006**
at epochs **70 / 72 / 96 / 99** respectively.

**Consequence:** `0.5009` is far above both the "pipeline correct,
architecture nearly matches" band (0.10-0.15) and the "meaningfully
weaker architecture" band (0.20-0.30) -- squarely in "something is
wrong" territory. The exploding-then-shrinking `grad_norm_mean` pattern
and the late rise in train loss motivated the gradient-clipping audit
(next entry) and the day-layer divergence investigation this round's
Task 1/3/4 resolve -- see "This round's Tasks 1-4 results" below for the
diagnosis (unregularized day layers) and the retrained result.

---

### The gradient-clipping audit found no defect (2026-09-09)

**Question:** Model A's `grad_norm_mean` reaching four figures despite
`grad_clip=1.0` -- is gradient clipping actually being applied, or is
`grad_norm_mean` silently reporting something else?

**Method:** Read `train_one_epoch` line by line: (a) is
`clip_grad_norm_` called every step, or conditionally; (b) does
`scaler.unscale_(optimizer)` precede it when AMP is in use; (c) is
`grad_norm_mean` the return value of `clip_grad_norm_` (documented as the
PRE-clip norm) or something recomputed after clipping; (d) is
`grad_clip` actually read from config and threaded through, or silently
defaulting. Then verified empirically: forced a large gradient (a
`nn.Linear` with weights deliberately scaled up) through the exact
`scale -> backward -> unscale_ -> clip -> step` sequence with AMP
enabled, and measured the actual gradient norm immediately before
`optimizer.step()`.

**Result:** All four code-reading questions came back clean: (a)
`clip_grad_norm_` runs on every non-skipped step, no condition beyond
the (unrelated) non-finite-loss skip; (b) `scaler.unscale_(optimizer)`
correctly precedes it; (c) `grad_norm_mean` IS `clip_grad_norm_`'s
return value, which PyTorch documents as the PRE-clip norm by design --
a large logged value is expected, not evidence of broken clipping; (d)
`grad_clip=1.0` is read from config and passed through
`02_train.py -> fit() -> train_one_epoch()` explicitly, no silent
default. Empirically: with a forced pre-clip norm of **102.59**, the
ACTUAL gradient norm immediately before `optimizer.step()` measured
**0.99999994** (~1.0, i.e. clipping worked exactly as intended).
Separately confirmed `scaler.step()` correctly no-ops on a genuine fp16
overflow (model weights unchanged, `GradScaler`'s scale factor shrinks
on the next `update()`).

**Consequence:** No defect. Model B's drift-curve numbers (the 0.353
s15 baseline and everything built on it) are unaffected -- they were
never in question, but are recorded as unaffected here explicitly, since
a reader seeing four-figure gradient norms in `metrics.csv` without this
entry would otherwise reasonably suspect the opposite.

---

### The confounded substitution rate and its correction (2026-09-09)

**Question:** Deletion rate rose `+0.049`/100d while (reference-length-
normalized) substitution rate FELL `-0.021`/100d across the 22-session
drift curve -- read at the time as evidence for "channels going quiet"
over "phoneme-to-channel mapping shifting" (`docs/DESIGN.md` section 4).
Does that comparison actually distinguish the two hypotheses?

**Method:** Noted that `substitution_rate` (substitutions / reference
length) can only fall as EMISSION falls, mechanically, regardless of
whether tuning changed: a substitution can only occur where the model
emitted a token, and `decoded_to_reference_length_ratio` (emission) falls
with elapsed time on this curve -- so both rates move together by
construction, confounding the comparison. Added
`substitution_rate_among_emitted = substitutions / (substitutions +
n_correct)` (`metrics.per_session_metrics`) -- substitutions as a
fraction of COMMITMENTS (right or wrong), independent of how often the
model commits at all -- and recomputed the slope.

**Result:** Corrected `substitution_rate_among_emitted` slope is
**flat**: `-0.0027`/100d over all 22 sessions, `+0.0010`/100d over the 17
passing-threshold sessions -- both an order of magnitude smaller than
the old confounded `substitution_rate` slope (`-0.021`/100d) and than
`deletion_rate`'s slope (`+0.049`/100d), and opposite in sign between
the two subsets (noise, not a trend). Mean
`substitution_rate_among_emitted` across the 22 sessions: **0.428** (std
**0.031**) -- roughly 43% of committed phonemes are wrong, essentially
constant across the whole 546-day range.

**Consequence:** The attribution conclusion survives the correction
(deletions rising while substitution-among-emitted stays flat still
points at channels going quiet / SNR loss over tuning change), but now
rests on defensible footing instead of a mechanically confounded
comparison. `substitution_rate_among_emitted` and `n_correct` are now
standard columns in `03_evaluate.py`'s CSV and `per_session_metrics`'s
output, not a one-off recomputation.

---

### The effective-signal control: the drift curve stops measuring degradation partway through (2026-09-09)

**Question:** At +546 days, real-feature PER (0.857) and shuffled-feature
PER (0.869, features replaced with a length-compatible donor trial from
the same session, targets unchanged) differ by only 0.012 -- does the
decoder ever meaningfully stop using neural information within the
observed 5-to-546-day range, and if so, when?

**Method:** `effective_signal = PER_shuffled - PER_real`, computed
per session across all 22 evaluable held-out sessions
(`scripts/03_evaluate.py --shuffled-control`), with a paired bootstrap
CI (`bootstrap_paired_difference_ci`) since real and shuffled PER are
computed on the identical trials.

**Result:** `effective_signal` is **0.1478 at +5d**, falls to **0.0248
by +75d** (first session below the 0.05 threshold), and is
**statistically indistinguishable from zero by +453d** (`-0.0029`, 95%
CI `[-0.0149, 0.0089]`). Descriptive linear fit: **-0.0140** per 100
days. Degenerate baselines establish the actual PER ceiling for
context: an empty hypothesis (all deletions) scores exactly **1.000**
(confirmed exactly, both tested sessions); a constant-silence hypothesis
(collapses to one token under greedy CTC decoding) scores
**0.962-0.966**; a genuinely non-functional decoder on this task
therefore scores roughly **0.87**, not 1.0 -- which is why
`effective_signal` (shuffled vs. real), not raw PER against 1.0, is the
correct null to compare against.

**Consequence:** **The drift curve stops measuring degradation and
starts measuring failure at roughly day 75** -- 471 of the 546 observed
days (86% of the range) show a decoder gaining essentially nothing from
receiving the correct brain data over a wrong trial's. Every PER value
reported for sessions beyond ~day 75 should be read as "the decoder has
stopped extracting usable signal here," not as a point on a graceful
degradation curve. `docs/DESIGN.md` records `effective_signal` as a
standard reported metric and the s15 cut point's coverage-gap
implication -- see that file.

---

### The identity-day-layer redirect control: the missing day layer explains the +5-day gap (2026-09-09)

**Question:** Validation PER on held-in sessions is 0.353; PER on the
earliest held-out session, only 5 days later, is 0.652 -- a 0.299 gap
too large to be electrode drift over 5 days. Two confounded causes: no
trained day layer (identity fallback) AND the session being unseen.

**Method:** For 3 training sessions with substantial Switchboard trial
counts, evaluated each session's own validation trials twice: normally
(trained day-layer slice), and with `session_indices` redirected (at
evaluation time only, never touching the saved checkpoint) to a held-out
session's index -- verified exact identity, zero bias. Same trials,
same everything else; only which day-layer slice applies differs.

**Result:** Average PER difference (identity minus trained), across the
3 sessions: **+0.313** -- essentially the ENTIRE 0.299 val-to-+5-day gap
(the identity-redirect penalty slightly exceeds it). Checkpoint file
confirmed byte-identical (sha256) before and after.

**Consequence:** **The missing day layer, not the session being unseen,
accounts for the +5-day gap.** This directly informs Question 3
(`docs/DESIGN.md` section 5): fitting a new day layer (Arm 2) on a
handful of labeled sentences should recover most of the gap, making that
the headline Arm 2 result rather than a supporting one -- see
`docs/DESIGN.md`'s Question 3 section for the consequence of this being
LARGER than the observed 18-month drift itself.

---

### The full 22-session drift curve: rapid early rise, then a flat plateau (2026-09-09)

**Question:** What does the s15 checkpoint's PER actually look like
across all 22 evaluable held-out sessions (not just the 2 endpoints), and
is the shape consistent with smooth continuous decay?

**Method:** `scripts/03_evaluate.py` (default: all held-out sessions,
`passes_threshold` recorded per row, not baked into which sessions run)
against the s15 diagnostic checkpoint (`checkpoint_best.pt`, epoch 92,
val PER 0.353), then `04_drift_curve.py`.

**Result:** PER at the earliest passing session (+5d) is **0.6522**;
at the latest (+546d), **0.8567**. Absolute change **+0.2045**,
descriptive linear-fit slope **+0.0278** PER per 100 days across all 22
points. **The shape is NOT smooth decay**: an early segment (days <=150,
n=8) has slope **+0.114**/100d; the late segment (days >150, n=14) has
slope **+0.013**/100d -- an order of magnitude flatter. Mean PER
<=150d: 0.750; >150d: 0.842 (std 0.036).

**Consequence:** Most of the observed PER change happens in the first
~150 days; the remaining ~400 days are a noisy plateau, not continued
decline. Read alongside the effective-signal entry above: the plateau
region substantially overlaps the region where `effective_signal` has
already collapsed toward zero, consistent with the plateau being a
floor (decoder no longer extracting much signal) rather than a slowing
but still-real decline.

---

### Choosing the identity-anchored day-layer penalty's lambda (2026-09-09)

**Question:** `DayLayers.identity_penalty()` (`L_day`, `docs/DESIGN.md`
section 5) needs a weighting lambda. Weight decay pulls toward zero, the
wrong no-op point for a per-session correction whose no-op is the
identity matrix, motivating this penalty -- but the identity-anchoring
formula still needs a real magnitude, not a guess.

**Method:** Instrumented one real training run on
`temporal_s01_s15_switchboard` (`scratch/task1_lambda_diagnostic.py`):
computed `raw_penalty = model.day_layers.identity_penalty()` every step
WITHOUT applying it to the loss, logging it alongside the real CTC loss
for the first 20 steps and mean values at epochs 1, 5, 10. Since the
penalty is never applied in this diagnostic run, `day_layers` drift
unconstrained -- the measured raw_penalty is an UPPER BOUND on what a run
that actually applies the penalty would see (an anchored run should hold
weights closer to identity, producing a smaller raw_penalty at the same
epoch).

**Result:** At epoch 5, mean CTC loss **2.8091**, mean raw (unweighted)
penalty **3.154027e-04**. Steps 0-4 show `raw_penalty` exactly 0.0 (a
freshly-initialized `DayLayers` starts exactly at identity). Percentage
of total loss contributed by the penalty term at candidate lambda values,
computed from these raw magnitudes: lambda=300 gives **~0.11%** of total
loss at epoch 1, **~3.26%** at epoch 5, **~7.53%** at epoch 10 -- growing
but not dominating, and landing inside the targeted 1-10%-at-epoch-5
band.

**Consequence:** **lambda = 300**, recorded in
`configs/experiment/switchboard_baseline.yaml` (`training.day_layer_penalty`,
default 0.0 -- 300 is the value used for the Task 3 comparison's B and D
runs specifically, not a change to the checked-in baseline default this
round). See the next entry for how this alt performed against the
non-instrumented alternatives.

---

### Task 3 architecture comparison: removing day layers entirely outperforms every regularization/patching variant on s15 (2026-09-09)

**Question:** Given Model A's divergence (unregularized day layers,
mean max|W-I| reaching 0.806) and the published baseline's patched input
(`patch_size=14, patch_stride=4`, 7,168-dim GRU input vs. this project's
plain 512-dim), which combination of (a) the identity-anchored day-layer
penalty (lambda=300, previous entry) and (b) input patching actually
helps, on `temporal_s01_s15_switchboard` -- a configuration tuned against
the 8,072-trial official split may not suit s15's 3,150 trials, so this
comparison was run on s15 specifically before committing anything to the
official-split retrain.

**Method:** Five 100-epoch runs, early stopping disabled, otherwise
identical hyperparameters: (A) current baseline (day layers,
unconstrained, no patching) -- already measured, see "Model A's initial
result and divergence" above (that entry is the official-split case;
here A is the analogous s15-scale run, `runs/diag_s15_100ep_nostop`); (B)
+ identity-anchored day-layer penalty only; (C) + patching only
(`patch_size=14, patch_stride=4`); (D) + both; (E) day layers removed
entirely (`use_day_layers: false`) as a control.

**Result:**

| config | best val PER (epoch) | final val PER (@99) | train loss trend @99 | grad_norm_mean @25/50/99 | day_penalty/mean\|W-I\|/max\|W-I\| @99 | total_seconds |
|---|---|---|---|---|---|---:|
| A baseline | 0.3529 (92) | 0.3712 | falling to ep92, slight uptick to 99 | 3.26 / 2.55 / 5.53 | 0.005352 / 0.1588 / 0.5692 | ~2076.5 |
| B +penalty | 0.3027 (81) | 0.3082 | flat | 3.72 / 2.79 / 3.22 | 0.0014 / 0.1787 / 0.7207 | 2162.8 |
| C +patching | 0.2823 (75) | 0.3014 | flat (val_loss/grad_norm creeping up) | 2.82 / 4.45 / 5.98 | 0.0058 / 0.2122 / 0.8673 | 1505.6 |
| D +both | 0.2641 (32) | 0.2896 | **rising** (+29% ep50->99) | 3.07 / 3.45 / 6.39 | 0.0020 / 0.1994 / 0.8069 | 1485.6 |
| E no day layers | **0.2001 (98)** | 0.2081 | falling | 3.99 / 1.89 / 2.84 | n/a | 1786.9 |

E's best epoch (98) lands almost at the very end of training with train
loss still falling and the lowest, flattest `grad_norm_mean` of any run
-- not an early peak followed by drift, unlike D (best epoch 32, then
train loss rises 29% while it keeps training, though its val PER stays
roughly flat/mildly worse). Notably, on s15, B's day-layer deviation
magnitude (mean max|W-I|=0.1787) is comparable to or slightly *larger*
than A's (0.1588) -- the penalty's benefit over A here is optimization
stability, not simply a smaller deviation magnitude. (s15's baseline A
never diverged as severely as the official-split Model A to begin with:
only ~14 of 45 day-layer slices train on s15's ~14 active sessions,
against all 45 on the official split.)

**Consequence:** **Configuration E (day layers removed) is the winner:
best PER by a wide margin (24% relative improvement over the next-best,
D) and the most stable training of the five runs.** Used for
`splits/official.json`'s Model A retrain (next entry). This finding
requires flagging plainly: **removing day layers eliminates the
mechanism Question 3's Arm 2 (day-layer-based recalibration) depends on**
-- every day-layer variant (A/B/C/D) underperforms the no-day-layer
control on s15, deepening the open question from "which day-layer
parameterization" to "whether day layers earn their complexity in this
architecture at all." This is recorded as an open question for
`docs/DESIGN.md` section 5, not resolved unilaterally here --
`configs/experiment/switchboard_baseline.yaml`'s checked-in default
(`use_day_layers: true`) is deliberately left unchanged this round; this
comparison used dedicated `scratch/` configs.

---

### Task 4: Model A retrained without day layers on the official split (2026-09-09)

**Question:** Given Task 3's result, does retraining Model A
(`splits/official.json`, all 45 sessions) with day layers removed close
the gap toward the published baseline's `val_PER=0.101`, and away from
the original divergent attempt's `best_val_per=0.5009`?

**Method:** Retrained on `splits/official.json` with Task 3's winning
configuration (`use_day_layers: false`, no patching, no day-layer
penalty), 100 epochs, early stopping disabled, otherwise identical
hyperparameters to the original attempt.

**Result:** Best val PER **0.2963 at epoch 47**, final (epoch 99)
**0.4045**. Train loss and `grad_norm_mean` both degrade substantially
after epoch 47 (train loss 0.628 -> 1.044; `grad_norm_mean` climbs from
single digits to spiking as high as 439.8 at epoch 91) -- real, but far
milder than the original with-day-layers divergence (`grad_norm_mean`
previously reached 283-1176). `checkpoint_best.pt` correctly recovers
the good epoch, so 0.2963 is the number that stands, but training on the
full 45-session official split is evidently noisier for this
configuration than s15 was (where run E was stable through epoch 99) --
plausibly because s15 only exercises ~14 of 45 sessions.

**Consequence:** **0.2963 lands in the 0.20-0.30 band: pipeline correct,
architecture meaningfully weaker than published.** Roughly halves the
gap to published (0.101) relative to the original divergent attempt
(0.5009) and eliminates the catastrophic divergence, but does not reach
the 0.10-0.15 "architecture nearly matches" band. Largest remaining
candidates, not implemented this round: (1) this project's GRU is
unidirectional by explicit design choice (`docs/DESIGN.md` section 10,
`[principled]`) while published baselines in this space are typically
bidirectional; (2) patching (`patch_size=14, patch_stride=4`, already
implemented and tested this round) has never been tried *combined with*
day layers off -- an essentially free experiment, since both pieces
already exist independently and pass their own tests, that Task 3's
A-E matrix did not include.

---

### Pre-committed rule for keeping or dropping day layers (2026-09-10)

**This entry was written and committed BEFORE the round-5 four-way
comparison's results were read** -- the same discipline as the
"Pre-committed cut-point selection rule" above, and for the same reason:
the decision to keep or drop day layers has consequences for the whole
project (Question 3's Arm 2 and the identity-day-layer-redirect finding
both depend on the mechanism existing), so the threshold that decides it
must not be shaped by whichever configuration happens to win.

**Context.** The round-4 s15 comparison (`docs/FINDINGS.md`: "Task 3
architecture comparison") found `use_day_layers: false` beat every
day-layer variant, but that verdict was unreliable for two reasons: (1)
the no-day-layer run's best epoch was 98 of 100 -- it had not converged,
so its `0.2001` is an upper bound; (2) it differed from the best
day-layer run in TWO ways at once (no day layers AND no patching), and
the patching-with-no-day-layers combination was never run. Round 5 runs
four configs at 150 epochs (early stopping disabled) to resolve both:
(1) patching + no day layers; (2) patching + diagonal day layers,
lambda 300; (3) patching + full day layers, lambda 3000; (4) no
patching + no day layers, 150-epoch rerun of round-4's E.

**The rule.** *Keep day layers if the best day-layer configuration's val
PER is within 0.03 of the best no-day-layer configuration's. Rationale:
0.03 is roughly 3x the measured checkpoint-selection noise (`std` 0.0091
over the last 20 epochs of the s15 diagnostic run -- `docs/FINDINGS.md`:
"Pre-committed cut-point selection rule" addendum), so a smaller gap is
not a reliable difference, and preserving the day-layer mechanism is
worth a small, reliably-measured PER cost because Question 3's Arm 2 and
the project's central finding (`docs/DESIGN.md` section 5: the day-layer
gap of 0.313 exceeds 18 months of drift) both depend on it. If the gap
exceeds 0.03, drop day layers from the primary architecture and redesign
Arm 2 around a different adaptation mechanism.*

**Outcome:** recorded in the next entry ("Day layers are unstable when
trained jointly across many sessions"), which also carries the four-way
comparison's numbers. The rule above is fixed as written; that entry
only applies it.

---

### Day layers are unstable when trained jointly across many sessions (2026-09-10)

**Question:** Model A (official split, all 45 sessions) diverged with
`"full"` day layers. Is per-session adaptation itself the problem, or
the way it is parameterized and trained?

**Method:** (1) Recap of the divergence and the round-4 mitigations.
(2) A round-5 four-way comparison on `temporal_s01_s15_switchboard`,
150 epochs each, early stopping disabled, to separate the confounds
round-4 left open (round-4's winning no-day-layer run had not converged
-- best epoch 98 of 100 -- and differed from the best day-layer run in
two ways at once, patching AND day layers, with the patching +
no-day-layer combination never run):

  1. patching (14/4) + `use_day_layers=False`
  2. patching (14/4) + `"diagonal"` day layers, `λ=300`
  3. patching (14/4) + `"full"` day layers, `λ=3000` (10x the round-4 anchor)
  4. no patching + `use_day_layers=False` (a fair 150-epoch rerun of round-4's E)

**Result -- the instability is a parameterization failure, not a
mechanism failure.**

*The divergence.* Model A's train loss ROSE from 1.49 to 1.64 between
epochs 50 and 99 (rising train loss is divergence, not overfitting);
`grad_norm_mean` (pre-clip) hit 283 / 1176 / 349 / 1006 at epochs
70 / 72 / 96 / 99; the 45 `"full"` slices reached `mean max|W−I| = 0.806`
with nothing anchoring them (`docs/FINDINGS.md`: "Model A's initial
result and divergence").

*The round-4 mitigations.* Identity-anchored regularization improved s15
val PER from 0.353 (unregularized `"full"`) to 0.303; a plain
`weight_decay=1e-4` diagnostic did **not** help, because weight decay
pulls toward `0` and the day-layer no-op point is `I`, not `0` -- pulling
`W` toward `0` annihilates the input (`y = x·0 + b`). Removing day layers
entirely was stable and slightly better still on PER.

*The round-5 four-way comparison:*

| # | config | best val PER (epoch) | converged? | final PER @149 | train-loss trend @149 | grad_norm_mean @149 | day-layer `mean|W−I| / max|W−I|` @149 | wall clock |
|---|---|---|---|---|---|---|---|---|
| 1 | patch, no day layers | **0.1697 (102)** | yes | 0.1803 | flat (mild rise) | 1.66 | — | 35.9 min |
| 2 | patch, `"diagonal"` day layers, λ300 | **0.16932 (106)** | yes | 0.1776 | flat (mild rise) | 1.85 | 0.0098 / 0.0410 | 40.2 min |
| 3 | patch, `"full"` day layers, λ3000 | 0.2509 (38) | best early, then **diverges** | 0.3317 | **rising** (1.4→3.5 from ep25) | 10.70 | 0.1481 / 0.5919 | 42.1 min |
| 4 | no patch, no day layers | 0.19575 (147) | **no — best in last 10%, upper bound** | 0.1990 | flat | 2.06 | — | 61.6 min |

Reading the table: (a) `"diagonal"` day layers (run 2, 1,024
params/session) train to full PER parity with a no-day-layer model
(run 1) -- gap 0.16932 vs 0.1697, i.e. **−0.0004** -- and the slices
barely leave identity (`max|W−I| ≈ 0.04` after 150 epochs,
`day_penalty ≈ 7e-5`). (b) `"full"` day layers (run 2 of round 4, and
run 3 here) destabilize joint training regardless of anchor strength:
`λ=3000` (measured to contribute 46% of total loss by epoch 10 --
dominating, not anchoring) did not fix it, it starved CTC instead --
train loss never fell below ~1.3 and then rose to 3.5, and the slices
*still* drifted to `max|W−I| ≈ 0.59` because `identity_penalty`
normalizes by `n_features²` so the per-element pull stays weak even as
the scalar loss term dominates. (c) Patching is what moves PER off the
~0.196–0.200 no-patch plateau (run 4, itself not even converged at
ep149) down to ~0.169–0.170.

**The tension, stated plainly.** A *trained* `"full"` day layer helps
its own session by **+0.313 PER** (the identity-day-layer redirect
control, `docs/FINDINGS.md`) -- a large, real effect. Yet training all
45 `"full"` slices *jointly* HURTS overall PER and destabilizes the run.
Both are true: the per-session correction is worth having, but the
262,656-parameter-per-session `"full"` parameterization is the wrong way
to carry it at this scale. The 1,024-parameter `"diagonal"`
parameterization keeps the benefit (session-parity PER, and it is what
Arm 2 fits from 10–50 sentences) without the instability.

**This is a real engineering problem for deployed speech BCIs, not a
quirk of this codebase.** Per-session / per-day input adaptation is
standard practice in this field (the published baseline ships 45
per-session `day_weights`/`day_biases`). That a full-rank version of it
*destabilizes joint training across sessions* -- and that the fix is to
constrain the parameterization, not to regularize harder -- is not, to
our knowledge, reported.

**Consequence:** applying the pre-registered rule (previous entry):
best day-layer PER 0.16932 (run 2) − best no-day-layer PER 0.1697
(run 1) = **−0.0004**, far inside the 0.03 band. **Day layers are KEPT**,
with `day_layer_parameterization: "diagonal"` and `day_layer_penalty:
300`. `configs/experiment/switchboard_baseline.yaml` now sets
`patch_size: 14, patch_stride: 4, use_day_layers: true,
day_layer_parameterization: diagonal, day_layer_penalty: 300.0,
max_epochs: 150, patience: 150`. `docs/DESIGN.md` sections 5 and 10 are
updated to the resolved decision; the round-4 "open question about
whether to keep day layers at all" is closed.
