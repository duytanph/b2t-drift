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
