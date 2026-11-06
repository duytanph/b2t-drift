# Findings: the empirical record

This file is the dated, append-only record of investigations and
measurements behind `docs/DESIGN.md`'s decisions. It grows as
investigations complete; entries are titled by **what was found**, never
by task number, and are never rewritten in place. A later correction is
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
scan (testing all integer shifts of the three internal 64-wide
boundaries from −16 to +16 against within- vs. across-block correlation)
tested the array boundaries.

**Result:** TX/SBP transition confirmed exactly at column 256 (columns
0–255: ~0.6% as many distinct values as samples; 256–511: ~76%). 64-wide
array boundaries (64/128/192) confirmed: the correlation gap peaks
exactly at shift 0 (0.0129) vs. next-best neighbors (0.0127, 0.0125),
falling off monotonically outward. The official README's literal
0–64/65–128/... boundaries (off by one) are not supported. All 8
hypothesized 64-wide blocks have per-column mean ≈ 0, variance ≈ 1
(expected under normalization, not independent boundary evidence).
Unplanned finding: `input_features` are already z-scored: every column
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
The fp16 timings above were measured without a `torch.amp.GradScaler`:
acceptable for a pure speed benchmark on random targets, but a real fp16
training run needs one to avoid silent gradient underflow.

**Consequence:** No GPU-driven reason to shrink the model or the
four-cut-point sweep (`docs/DESIGN.md` section 7). The sweep's real
bottleneck turned out to be the dataloader, not GPU compute (see
"Storage medium, not DataLoader configuration..." below). Confirmed
`scripts/02_train.py` actually constructs a `GradScaler` whenever
`amp: true` (the configured default,
`configs/experiment/switchboard_baseline.yaml`) and `train_one_epoch`
uses it to scale/unscale before clipping and stepping: this benchmark's
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
session order (r = −0.40): earlier sessions have more blocks
(`t15.2023.08.13`: 11) than later ones (many: 3–6).

(b) `torch.load(weights_only=True)` failed (`UnpicklingError`:
`numpy.core` renamed to `numpy._core`, breaking torch's safe-globals
string match against the official file); fell back to
`weights_only=False` for this specific, documented-source file. 5 GRU
layers, hidden size 768. GRU input size is 7,168, not 512: the official
model patches its input (`patch_size=14, patch_stride=4`, 14×512=7,168,
stepped every 4 bins), matching the 80 ms real-time phoneme rate. Output
size 41, confirming `ctc_output_size=41` independently. Per-session
day-specific input layers confirmed present, as 45 separately-keyed
tensors `day_weights.0..44` (512×512) and `day_biases.0..44` (1×512),
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
float precision). Global: no predicted trend materialized. 6
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
divisor per SENTENCE (a rolling window): no shared per-block or
per-session lattice could ever have existed (`docs/DATASET.md` section 9
item 7).

---

### Switchboard survivor count corrected to 17 of 30 sessions (2026-09-01)

**Question:** How many held-out sessions have enough labeled Switchboard
trials (≥50) to be usable eval points?

**Method:** Rebuilt the viability table using per-(session, block) trial
counts read directly from HDF5 (train+val partitions), joined to the
CSV's block-level corpus label: the method the next entry shows is
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
+249 excess; 4 rows actual > CSV, −35; net +214): a "QC drops during
packaging" explanation predicts only one direction and is incomplete.
Roughly proportional to corpus/partition size, not concentrated
(Switchboard: 24/34 mismatched rows, 174/214 of the discrepancy, close to
its ~69% trial-count base rate). Mismatched blocks skew smaller (mean
32.7 actual trials vs. 42.6 matched) with several suspicious round-number
mismatches (±10, ±5) in both directions. Worked example
(`t15.2023.11.03`): CSV lists block 6 as 50 Switchboard sentences; actual
HDF5 block 6 has 50 trials total across val+test **combined** (25 each):
the CSV's 50 is right for the block, but only 25 are in the labeled
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
separately, then an exact per-(sentence, channel) lattice test. Since
normalization is per-sentence, each sentence has exactly one divisor per
channel, unlike a whole block. Tested on 30 trials × 8 TX channels = 240
pairs (501 trials sampled across 3 sessions).

**Result:** TX and SBP per-channel minima are different distributions
(TX median −0.85, range to −1.44; SBP median −2.12, range to −3.51). A
previously-cited global minimum of −3.5107 is an SBP column (278), not
TX, so a Poisson spike-count inversion never applied to it. Every TX
channel's minimum recurs >100 times (median 13,207): plateaus, not
one-off outliers, consistent with genuine zero-count bins. The
per-sentence lattice is confirmed in 230/240 (95.8%) of tested pairs,
giving an exact recovered sigma per sentence and channel (median 0.56,
range 0.23–0.88). Exact recovery: `sigma = 1/spacing`,
`mu = -min*sigma`, `raw = released*sigma + mu`: median implied firing
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
208, 243, 278, 280, 287, 453, 455, 516, 518, 532, 546: a **126-day hole
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

**Method:** N/A. A decision, recorded in advance per the validation-set
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
rule before seeing any PER (not after) is what makes the selection
principled rather than a post-hoc rationalization of whichever cut looks
best.

**Addendum (2026-09-08): the numeric "acceptable" threshold, defined now
while only one cut point (s15) has been trained.** This rule named
"acceptable" and a "2 absolute percentage point" comparable band but never
gave either a number. s15's 100-epoch diagnostic run gives exactly one
anchor (`best_val_per = 0.353` at epoch 92), and this addendum is written
using only that anchor, before s05/s10/s20 have been trained.

- **Acceptable validation PER: ≤ 0.45.** Reasoning against the s15 anchor:
  (1) the margin above the anchor (0.45 − 0.353 = 0.097, ~0.10) is roughly
  11x the measured best-epoch-selection noise (`std(val_per)` over the last
  20 epochs = 0.0091, see the timing/noise entry below), so a cut failing
  this threshold is failing for a real reason (underfitting / a floor
  effect from less training data), not selection variance (see the next
  bullet for where the 0.0091 noise figure comes from); (2) it sits far
  below the diagnostic run's observed "blank phase" ceiling (val PER ~0.83
  in early epochs, before the model commits to any phoneme structure,
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
  selection noise (0.02 / 0.0091 ≈ 2.2): tight enough to still mean
  something (a difference this size is unlikely to be pure epoch-selection
  luck) but not so tight that ordinary run-to-run noise would make two
  genuinely-similar cut points look different. Recorded explicitly so a
  reader can see the band was derived from a measurement, not chosen by
  eye.
- **Why this was written before any other cut point had a PER value.**
  Exactly one anchor (s15 = 0.353) exists at the time of writing. Waiting
  until s05/s10/s20 also have PER values before fixing this threshold would
  let those numbers silently influence where the line gets drawn: the
  same failure mode the original rule (written before any PER existed) was
  designed to prevent. Fixing the number now, against only the s15 anchor,
  keeps the whole rule (both the original "largest acceptable cut" logic
  and this threshold) pre-committed rather than post-hoc.

**Addendum (2026-09-11): the rule applied, now that all four cut points have
been trained** (`docs/FINDINGS.md`: "The cut-point sweep: calibration-data
advantage is front-loaded and fades by ~100 days").

All four validation PERs (s05 0.15765, s10 0.16054, s15 0.16932, s20
0.16566) are acceptable under the 0.45 threshold and fall within a
**0.012-wide band**, well inside the 0.02 comparability band. **The rule
selects the earliest acceptable cut point: s05.** Decision recorded: **s05
is the PRIMARY analysis**, with s15 reported alongside as a robustness
check, and the curve's shape (step-then-plateau) confirmed consistent
across all four cut points independently.

**The rule's own limitation, recorded plainly rather than papered over.**
It weighs only validation PER and curve resolution, and is silent on
generalization to unseen sessions and on training stability. Had it
weighed those, s10 or s15 would have been preferred: s05 has the worst
unseen-session generalization of the four (a 0.12 PER gap between its
validation set and its earliest held-out session, at +2d, vs. s15's 0.008
at +5d) and the least stable late training (train loss and gradient norms
rise sharply after ~epoch 115, where s10/s15/s20 stay flat or mildly
noisy). **The rule was followed rather than overridden, because it was
fixed before any of these values existed**: overriding it after seeing
the outcome would make the selection post-hoc, exactly the failure mode
recording it in advance exists to prevent. This is the same reasoning
that governed the original rule's timing (previous bullet), applied
symmetrically to an outcome the rule-writer might not have chosen by eye.

---

### Storage medium, not DataLoader configuration, was the training bottleneck (2026-09-03 to 2026-09-04)

**Question:** What `num_workers` setting is fastest, and can it be tuned
from a synthetic benchmark?

**Method:** Two rounds of synthetic DataLoader benchmarking, then a real
measurement on the full 3,150-trial `temporal_s01_s15_switchboard` train
set (3 real epochs, `num_workers` in {0, 1, 2, 4}), first on the
original spinning disk, then again after the dataset was moved to an
SSD.

**Result:** Round 1 (200 synthetic items) picked `num_workers=0`:
wrong. Windows `spawn` worker-startup overhead swamped the signal at
that scale. Round 2 (2,048 synthetic fixed-length items) measured 35
ms/batch at `num_workers=2`, concluding it cleared the ~230 ms/step fp16
compute budget: also wrong at real scale, since that same config still spent
74% of steady-state epoch time waiting on the dataloader (217.4s of
292.8s). Disk read before/after the storage change: 70–93 ms/trial vs.
4.43 ms/trial, 371.7 MB/s. Steady-state epoch time at `num_workers=2`
before/after: 292.8s at 74% wait vs. 41.9s at under 1% wait. Post-move
`num_workers` comparison (3rd epoch): 0 → 62.33s (24.8% wait), 1 →
42.92s (0.9%), 2 → 41.86s (0.9%), 4 → 43.12s (1.6%): 1/2/4 effectively
tied, 0 slower because loading and compute no longer overlap. Peak GPU
memory unchanged at 1749.7 MB across every configuration.

**Consequence:** A benchmark must reproduce the real access pattern
(many distinct files, real variable lengths, no page-cache reuse) or it
measures something else; both synthetic rounds passed a scale test but
failed the access-pattern test. `num_workers=2` is the config default;
I/O is no longer the bottleneck at any `num_workers >= 1`, so no further
(e.g. packed-cache) dataloader work is needed. Extrapolating the
`num_workers=2` rate linearly by train-batch count across all four cut
points: 3.50h for 100 epochs of every cut point, summed, superseding
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
trained model, large enough to flip a marginal argmax. Mechanism:
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
(`train_one_epoch`, no `evaluate()` call): the two were never a like-for-
like comparison, and this could not be resolved from the existing artifact
alone, since `metrics.csv` had no way to separate the two. `training/loop.py`'s
`fit()` now additionally logs `train_seconds` and `eval_seconds` as their own
columns (keeping `seconds` as their sum, unchanged, for anything already
reading it). This resolves the ambiguity starting with the NEXT run, not
retroactively for the diagnostic run already on disk.

**Result (from the existing, combined-only column):** epoch 0: 55.68s;
epoch 1: 47.19s; epoch 2: 41.34s; steady-state (epochs 10-99): median
**18.36 s/epoch**, IQR 1.35s (Q1 17.90, Q3 19.25); first epoch within 10% of
that steady-state median: **epoch 22**. The monotonic decline from a cold
55.68s down to a steady ~18.36s over the first ~22 epochs is consistent
with the ~5 GB s15 training set (3,150 trials) gradually becoming
OS-page-cache resident against 16 GB system RAM, mirroring the mechanism
the original entry already documented for the initial spinning-disk-to-SSD
comparison, but **this cannot yet be stated as confirmed**: because the
column is combined, the same decline could also be explained by eval-side
caching, cuDNN algorithm re-selection settling, or some mix of all three.
**The train-only figure needed to compare directly against the recorded
41.9s is not computable from this run's artifacts and is not reported
here**: no subtracted or estimated number is given; the next run (which
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
0.020, 0.030}`: pure arithmetic over the recorded column, no retraining.

**Result:** Every `patience=5` row stops between epoch 19 and 22 regardless
of `min_delta`: not threshold-sensitive, since there is a noisy stretch
around epochs 14-22 (a transient grad-norm spike, `n_inf_batches` still 0)
that any short patience walks into. At `min_delta=0.000, patience=5`
(today's config, before this entry): stops at epoch 22, selects epoch 17 at
`val_per=0.5937`, vs. `0.3529` achievable at epoch 92: a **+0.241
absolute PER cost**. The boundary is otherwise sharp and run-specific:
`min_delta=0.005` never stops within 100 epochs (selects epoch 92,
`0.3529`), while `min_delta=0.010` at `patience=10` stops at epoch 60
(`0.3954`).

**Consequence:** `configs/experiment/switchboard_baseline.yaml` now sets
`max_epochs: 100, patience: 100` (early stopping effectively disabled:
no 100-epoch run can accumulate 100 epochs-since-improvement). Early
stopping is a compute-saving device; compute is not the binding constraint
here (~1.5h for the full four-cut sweep at the measured steady-state rate,
see the timing entry above), so disabling it costs nothing and removes a
documented, real hazard. `fit()`'s early-stopping comparison itself
(`training/loop.py`) is deliberately left untouched: no `min_delta` was
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
epoch 99**: not a plateau. Train loss itself ROSE from **1.49 to 1.64
between epochs 50 and 99** (not overfitting: overfitting is falling
train loss with rising val loss; rising train loss is divergence).
`grad_norm_mean` (the mean PRE-clip gradient norm, see the next entry
for why this is the correct number to read): **283 / 1176 / 349 / 1006**
at epochs **70 / 72 / 96 / 99** respectively.

**Consequence:** `0.5009` is far above both the "pipeline correct,
architecture nearly matches" band (0.10-0.15) and the "meaningfully
weaker architecture" band (0.20-0.30): squarely in "something is
wrong" territory. The exploding-then-shrinking `grad_norm_mean` pattern
and the late rise in train loss motivated the gradient-clipping audit
(next entry) and the day-layer divergence investigation this round's
Task 1/3/4 resolve. See "This round's Tasks 1-4 results" below for the
diagnosis (unregularized day layers) and the retrained result.

---

### The gradient-clipping audit found no defect (2026-09-09)

**Question:** Model A's `grad_norm_mean` reaching four figures despite
`grad_clip=1.0`: is gradient clipping actually being applied, or is
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
return value, which PyTorch documents as the PRE-clip norm by design:
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
s15 baseline and everything built on it) are unaffected: they were
never in question, but are recorded as unaffected here explicitly, since
a reader seeing four-figure gradient norms in `metrics.csv` without this
entry would otherwise reasonably suspect the opposite.

---

### The confounded substitution rate and its correction (2026-09-09)

**SUPERSEDED by "Substitution rate among emitted phonemes rises under the settled architecture: the old flatness was a ceiling artifact" (2026-09-11).** The flat-slope conclusion below was measured on the old architecture (val PER 0.353); under the settled architecture the slope rises. The methodological correction in this entry (substitution rate must be normalized against commitments, not reference length) is NOT superseded and still stands. See the new entry for what changed and what didn't.

**Question:** Deletion rate rose `+0.049`/100d while (reference-length-
normalized) substitution rate FELL `-0.021`/100d across the 22-session
drift curve, read at the time as evidence for "channels going quiet"
over "phoneme-to-channel mapping shifting" (`docs/DESIGN.md` section 4).
Does that comparison actually distinguish the two hypotheses?

**Method:** Noted that `substitution_rate` (substitutions / reference
length) can only fall as EMISSION falls, mechanically, regardless of
whether tuning changed: a substitution can only occur where the model
emitted a token, and `decoded_to_reference_length_ratio` (emission) falls
with elapsed time on this curve, so both rates move together by
construction, confounding the comparison. Added
`substitution_rate_among_emitted = substitutions / (substitutions +
n_correct)` (`metrics.per_session_metrics`): substitutions as a
fraction of COMMITMENTS (right or wrong), independent of how often the
model commits at all, and recomputed the slope.

**Result:** Corrected `substitution_rate_among_emitted` slope is
**flat**: `-0.0027`/100d over all 22 sessions, `+0.0010`/100d over the 17
passing-threshold sessions, both an order of magnitude smaller than
the old confounded `substitution_rate` slope (`-0.021`/100d) and than
`deletion_rate`'s slope (`+0.049`/100d), and opposite in sign between
the two subsets (noise, not a trend). Mean
`substitution_rate_among_emitted` across the 22 sessions: **0.428** (std
**0.031**): roughly 43% of committed phonemes are wrong, essentially
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

**SUPERSEDED by "Effective signal does not collapse under the settled architecture" (2026-09-11).** The collapse-by-+75d / statistically-zero-by-+453d conclusion below was measured on the old architecture (val PER 0.353); under the settled architecture (val PER 0.169) effective signal never reaches statistical zero in three of the four cut points. The control ITSELF, and the reasoning for why raw PER cannot substitute for it, are NOT superseded. See the new entry for what changed and why the control still mattered.

**Question:** At +546 days, real-feature PER (0.857) and shuffled-feature
PER (0.869, features replaced with a length-compatible donor trial from
the same session, targets unchanged) differ by only 0.012: does the
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
therefore scores roughly **0.87**, not 1.0, which is why
`effective_signal` (shuffled vs. real), not raw PER against 1.0, is the
correct null to compare against.

**Consequence:** **The drift curve stops measuring degradation and
starts measuring failure at roughly day 75**: 471 of the 546 observed
days (86% of the range) show a decoder gaining essentially nothing from
receiving the correct brain data over a wrong trial's. Every PER value
reported for sessions beyond ~day 75 should be read as "the decoder has
stopped extracting usable signal here," not as a point on a graceful
degradation curve. `docs/DESIGN.md` records `effective_signal` as a
standard reported metric and the s15 cut point's coverage-gap
implication: see that file.

---

### The identity-day-layer redirect control: the missing day layer explains the +5-day gap (2026-09-09)

**Question:** Validation PER on held-in sessions is 0.353; PER on the
earliest held-out session, only 5 days later, is 0.652: a 0.299 gap
too large to be electrode drift over 5 days. Two confounded causes: no
trained day layer (identity fallback) AND the session being unseen.

**Method:** For 3 training sessions with substantial Switchboard trial
counts, evaluated each session's own validation trials twice: normally
(trained day-layer slice), and with `session_indices` redirected (at
evaluation time only, never touching the saved checkpoint) to a held-out
session's index: verified exact identity, zero bias. Same trials,
same everything else; only which day-layer slice applies differs.

**Result:** Average PER difference (identity minus trained), across the
3 sessions: **+0.313**, essentially the ENTIRE 0.299 val-to-+5-day gap
(the identity-redirect penalty slightly exceeds it). Checkpoint file
confirmed byte-identical (sha256) before and after.

**Consequence:** **The missing day layer, not the session being unseen,
accounts for the +5-day gap.** This directly informs Question 3
(`docs/DESIGN.md` section 5): fitting a new day layer (Arm 2) on a
handful of labeled sentences should recover most of the gap, making that
the headline Arm 2 result rather than a supporting one. See
`docs/DESIGN.md`'s Question 3 section for the consequence of this being
LARGER than the observed 18-month drift itself.

---

### The full 22-session drift curve: rapid early rise, then a flat plateau (2026-09-09)

**SUPERSEDED by "The full drift curve under the settled architecture: steeper rise, same shape" (2026-09-11).** The specific numbers below (absolute change, slope, segment slopes) were measured on the old architecture (val PER 0.353) and are superseded by larger values under the settled architecture. The step-then-plateau SHAPE conclusion is NOT superseded: it survives and is now confirmed independently across three separately-trained models, not just this one.

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
slope **+0.013**/100d: an order of magnitude flatter. Mean PER
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
identity matrix, motivating this penalty, but the identity-anchoring
formula still needs a real magnitude, not a guess.

**Method:** Instrumented one real training run on
`temporal_s01_s15_switchboard` (`scratch/task1_lambda_diagnostic.py`):
computed `raw_penalty = model.day_layers.identity_penalty()` every step
WITHOUT applying it to the loss, logging it alongside the real CTC loss
for the first 20 steps and mean values at epochs 1, 5, 10. Since the
penalty is never applied in this diagnostic run, `day_layers` drift
unconstrained: the measured raw_penalty is an UPPER BOUND on what a run
that actually applies the penalty would see (an anchored run should hold
weights closer to identity, producing a smaller raw_penalty at the same
epoch).

**Result:** At epoch 5, mean CTC loss **2.8091**, mean raw (unweighted)
penalty **3.154027e-04**. Steps 0-4 show `raw_penalty` exactly 0.0 (a
freshly-initialized `DayLayers` starts exactly at identity). Percentage
of total loss contributed by the penalty term at candidate lambda values,
computed from these raw magnitudes: lambda=300 gives **~0.11%** of total
loss at epoch 1, **~3.26%** at epoch 5, **~7.53%** at epoch 10: growing
but not dominating, and landing inside the targeted 1-10%-at-epoch-5
band.

**Consequence:** **lambda = 300**, recorded in
`configs/experiment/switchboard_baseline.yaml` (`training.day_layer_penalty`,
default 0.0: 300 is the value used for the Task 3 comparison's B and D
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
helps, on `temporal_s01_s15_switchboard` (a configuration tuned against
the 8,072-trial official split may not suit s15's 3,150 trials). So this
comparison was run on s15 specifically before committing anything to the
official-split retrain.

**Method:** Five 100-epoch runs, early stopping disabled, otherwise
identical hyperparameters: (A) current baseline (day layers,
unconstrained, no patching), already measured, see "Model A's initial
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
loss still falling and the lowest, flattest `grad_norm_mean` of any run:
not an early peak followed by drift, unlike D (best epoch 32, then
train loss rises 29% while it keeps training, though its val PER stays
roughly flat/mildly worse). Notably, on s15, B's day-layer deviation
magnitude (mean max|W-I|=0.1787) is comparable to or slightly *larger*
than A's (0.1588): the penalty's benefit over A here is optimization
stability, not simply a smaller deviation magnitude. (s15's baseline A
never diverged as severely as the official-split Model A to begin with:
only ~14 of 45 day-layer slices train on s15's ~14 active sessions,
against all 45 on the official split.)

**Consequence:** **Configuration E (day layers removed) is the winner:
best PER by a wide margin (24% relative improvement over the next-best,
D) and the most stable training of the five runs.** Used for
`splits/official.json`'s Model A retrain (next entry). This finding
requires flagging plainly: **removing day layers eliminates the
mechanism Question 3's Arm 2 (day-layer-based recalibration) depends on**:
every day-layer variant (A/B/C/D) underperforms the no-day-layer
control on s15, deepening the open question from "which day-layer
parameterization" to "whether day layers earn their complexity in this
architecture at all." This is recorded as an open question for
`docs/DESIGN.md` section 5, not resolved unilaterally here:
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
single digits to spiking as high as 439.8 at epoch 91): real, but far
milder than the original with-day-layers divergence (`grad_norm_mean`
previously reached 283-1176). `checkpoint_best.pt` correctly recovers
the good epoch, so 0.2963 is the number that stands, but training on the
full 45-session official split is evidently noisier for this
configuration than s15 was (where run E was stable through epoch 99),
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
day layers off, an essentially free experiment, since both pieces
already exist independently and pass their own tests, that Task 3's
A-E matrix did not include.

---

### Pre-committed rule for keeping or dropping day layers (2026-09-10)

**This entry was written and committed BEFORE the round-5 four-way
comparison's results were read**: the same discipline as the
"Pre-committed cut-point selection rule" above, and for the same reason:
the decision to keep or drop day layers has consequences for the whole
project (Question 3's Arm 2 and the identity-day-layer-redirect finding
both depend on the mechanism existing), so the threshold that decides it
must not be shaped by whichever configuration happens to win.

**Context.** The round-4 s15 comparison (`docs/FINDINGS.md`: "Task 3
architecture comparison") found `use_day_layers: false` beat every
day-layer variant, but that verdict was unreliable for two reasons: (1)
the no-day-layer run's best epoch was 98 of 100, it had not converged,
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
over the last 20 epochs of the s15 diagnostic run, `docs/FINDINGS.md`:
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
round-4 left open (round-4's winning no-day-layer run had not converged,
best epoch 98 of 100, and differed from the best day-layer run in
two ways at once, patching AND day layers, with the patching +
no-day-layer combination never run):

  1. patching (14/4) + `use_day_layers=False`
  2. patching (14/4) + `"diagonal"` day layers, `λ=300`
  3. patching (14/4) + `"full"` day layers, `λ=3000` (10x the round-4 anchor)
  4. no patching + `use_day_layers=False` (a fair 150-epoch rerun of round-4's E)

**Result: the instability is a parameterization failure, not a
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
pulls toward `0` and the day-layer no-op point is `I`, not `0`: pulling
`W` toward `0` annihilates the input (`y = x·0 + b`). Removing day layers
entirely was stable and slightly better still on PER.

*The round-5 four-way comparison:*

| # | config | best val PER (epoch) | converged? | final PER @149 | train-loss trend @149 | grad_norm_mean @149 | day-layer `mean|W−I| / max|W−I|` @149 | wall clock |
|---|---|---|---|---|---|---|---|---|
| 1 | patch, no day layers | **0.1697 (102)** | yes | 0.1803 | flat (mild rise) | 1.66 | n/a | 35.9 min |
| 2 | patch, `"diagonal"` day layers, λ300 | **0.16932 (106)** | yes | 0.1776 | flat (mild rise) | 1.85 | 0.0098 / 0.0410 | 40.2 min |
| 3 | patch, `"full"` day layers, λ3000 | 0.2509 (38) | best early, then **diverges** | 0.3317 | **rising** (1.4→3.5 from ep25) | 10.70 | 0.1481 / 0.5919 | 42.1 min |
| 4 | no patch, no day layers | 0.19575 (147) | **no, best in last 10%, upper bound** | 0.1990 | flat | 2.06 | n/a | 61.6 min |

Reading the table: (a) `"diagonal"` day layers (run 2, 1,024
params/session) train to full PER parity with a no-day-layer model
(run 1): gap 0.16932 vs 0.1697, i.e. **−0.0004**, and the slices
barely leave identity (`max|W−I| ≈ 0.04` after 150 epochs,
`day_penalty ≈ 7e-5`). (b) `"full"` day layers (run 2 of round 4, and
run 3 here) destabilize joint training regardless of anchor strength:
`λ=3000` (measured to contribute 46% of total loss by epoch 10:
dominating, not anchoring) did not fix it. It starved CTC instead:
train loss never fell below ~1.3 and then rose to 3.5, and the slices
*still* drifted to `max|W−I| ≈ 0.59` because `identity_penalty`
normalizes by `n_features²` so the per-element pull stays weak even as
the scalar loss term dominates. (c) Patching is what moves PER off the
~0.196–0.200 no-patch plateau (run 4, itself not even converged at
ep149) down to ~0.169–0.170.

**The tension, stated plainly.** A *trained* `"full"` day layer helps
its own session by **+0.313 PER** (the identity-day-layer redirect
control, `docs/FINDINGS.md`): a large, real effect. Yet training all
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
*destabilizes joint training across sessions* (and that the fix is to
constrain the parameterization, not to regularize harder) is not, to
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

---

### The identity-day-layer redirect control, repeated under diagonal day layers: the gap is gone (2026-09-10)

**Question:** "The identity-day-layer redirect control" (above) measured
a `+0.313` PER cost to redirecting a trained day layer to an untrained
(identity) slice, on the OLD architecture's `"full"` parameterization.
The settled architecture uses `"diagonal"` instead (`docs/FINDINGS.md`:
"Day layers are unstable when trained jointly across many sessions").
Does the same gap hold?

**Method:** Repeated the control exactly as before
(`scratch/r6_redirect_control.py`), against the settled checkpoint
(`runs/r5_2_patch_diag_l300/checkpoint_best.pt`: diagonal day layers,
λ300, patch 14/4). Same three training sessions (`t15.2023.09.03`,
`t15.2023.09.01`, `t15.2023.08.20`), same redirect target
(`t15.2023.10.20`, held-out, verified exact identity: `max|scale−1|=0`,
`max|bias|=0`), same method: evaluate each session's own validation
trials twice, once with its trained day-layer slice and once with
`session_indices` redirected to the identity slice. Checkpoint file
confirmed byte-identical (sha256) before and after.

**Result:** Average difference (identity minus trained) across the 3
sessions: **−0.0038** (individually: `t15.2023.09.03` −0.0011,
`t15.2023.09.01` −0.0020, `t15.2023.08.20` −0.0084): essentially zero,
and if anything slightly *negative* (identity is marginally better than
the trained slice, within noise). The trained diagonal slices barely
leave identity in the first place: `max|scale−1|` ranges 0.024–0.026,
`max|bias|` ranges 0.024–0.031 across the three sessions: an order of
magnitude smaller than the old `"full"` architecture's per-session
deviations.

**Consequence:** **With diagonal day layers, per-session calibration is
close to unnecessary**: a trained slice changes its own session's PER
by essentially nothing. This is a substantive finding about BCI
calibration practice (`docs/DESIGN.md` section 5): constraining the
per-session correction to per-channel scaling makes the day layer nearly
free to omit, and the old `+0.313` figure was a property of the more
expressive `"full"` parameterization's capacity to memorize a session,
not evidence that per-session correction is inherently valuable. This
directly reframes Question 3. See `docs/DESIGN.md` section 5 for the
revised framing and the open question it leaves for Arm 2's own
single-session, frozen-backbone fitting regime, which this
joint-training measurement does not settle either way.

---

### Effective signal does not collapse under the settled architecture (2026-09-11)

**Supersedes** "The effective-signal control: the drift curve stops
measuring degradation partway through" (2026-09-09), whose collapse
conclusion was measured on the OLD architecture (val PER 0.353). All
numbers below are corrections driven by a better decoder, not errors in
the original measurement: the old numbers were the correct reading of
the model that existed when they were taken.

**Question:** The settled architecture (patch 14/4, diagonal day layers,
val PER 0.169, roughly half the old error rate) reaches this project's
drift curve differently from the old one. Does it still show the old
decoder's effective-signal collapse, or was that collapse a property of a
weak decoder rather than of the data?

**Method:** `scripts/03_evaluate.py --shuffled-control` re-run against
the settled architecture's checkpoints for all four cut points
(`docs/FINDINGS.md`: "The cut-point sweep: calibration-data advantage is
front-loaded and fades by ~100 days"), reading `effective_signal` per
session exactly as the superseded entry did.

**Result:** On s15, `effective_signal` is **0.755 at +5d**, **0.213 at
+75d** (vs. the old architecture's 0.148 and 0.025 at the same offsets),
and **0.072 at +546d with a 95% CI of [0.041, 0.106]**: clear of zero.
Every one of the 22 s15 sessions is CI-separated from zero. Across the
full sweep, `effective_signal` never reaches statistical zero in s10,
s15, or s20; only marginally in s05, at +504d (2 sessions, both with a CI
barely spanning zero). Descriptive slopes per 100 days: s05 −0.078, s10
−0.093, s15 −0.074, s20 −0.030.

**Consequence:** **The collapse was a property of a weak decoder, not of
the data.** The drift curve now measures genuine degradation across
close to the full ~18-month range this project set out to characterize,
not degradation for ~75 days followed by ~470 days of noise about a
failed decoder. This also demonstrates directly why the effective-signal
control was worth building: raw PER alone (0.857 real vs. 0.869 shuffled
at +546d, OLD architecture) could not distinguish a genuinely degraded
decoder from one that had stopped using neural information: only the
shuffled-input comparison could, and it is this same control that now
shows the settled architecture has NOT stopped using neural information
anywhere in the observed range. `docs/DESIGN.md` sections 7 and 9 are
updated to match.

---

### The full drift curve under the settled architecture: steeper rise, same shape (2026-09-11)

**Supersedes** "The full 22-session drift curve: rapid early rise, then
a flat plateau" (2026-09-09), whose numbers were measured on the OLD
architecture (val PER 0.353). A correction driven by a better decoder,
not an error in the original measurement.

**Question:** Does the settled architecture's (val PER 0.169) drift
curve show the same step-then-plateau shape as the old one, or does a
stronger decoder degrade differently?

**Method:** Same method as the superseded entry, re-run against the
settled s15 checkpoint and, this round, independently against three more
checkpoints (s05, s10, s20; `docs/FINDINGS.md`: "The cut-point sweep:
calibration-data advantage is front-loaded and fades by ~100 days").
The shape claim is now checked across four separately-trained models,
not asserted from one.

**Result:** s15: PER **0.1768 at +5d** rising to **0.7536 at +546d**,
absolute **+0.577**, relative **+326%**. Descriptive slope **+0.069**
PER per 100 days. Early segment (≤150d) **+0.326**/100d, late segment
(>150d) **+0.021**/100d: ratio **15.7×**. Independently: s05's ratio is
**~23×**, s10's is **~24×**; s20 is consistent everywhere it has
coverage, though its own held-out sessions start at +100d (past the
steep segment), so it cannot check the ratio directly.

**Consequence:** The absolute numbers above are superseded (the old
+0.205/+0.028/+0.114/+0.013 were correct for the old decoder, not wrong
measurements). **The qualitative shape claim survives and is
strengthened**: most of the observed change happens in the first ~150
days, then a noisy plateau: now confirmed independently in three
separately-trained models (s05, s10, s15), not asserted from one. The
plateau is steeper and the total drift larger under the stronger decoder
(+0.577 vs. +0.205): a better decoder starts from a much lower PER, has
more room to lose across 18 months, and loses more of it in absolute
terms even though it stays far ahead of the old decoder throughout.

---

### Substitution rate among emitted phonemes rises under the settled architecture: the old flatness was a ceiling artifact (2026-09-11)

**Supersedes** "The confounded substitution rate and its correction"
(2026-09-09)'s flat-slope conclusion (−0.0027/100d), measured on the OLD
architecture. Its methodological correction (that substitution rate
must be normalized against commitments, not reference length, because
the latter is confounded with emission rate) is NOT superseded and
still stands; this entry only revises what the corrected quantity shows
once measured on a decoder good enough to have room to move on it.

**Question:** Does `substitution_rate_among_emitted` still look flat
under the settled architecture, or does the attribution conclusion built
on its flatness ("channels going quiet, not tuning change") change?

**Method:** Same quantity, same method, re-measured on the settled
architecture across all four cut points.

**Result:** `substitution_rate_among_emitted` **RISES** in every cut
point: s05 +0.058, s10 +0.082, s15 +0.062, s20 +0.036 per 100 days,
alongside `deletion_rate`, which also rises everywhere (s05 +0.062, s10
+0.033, s15 +0.035, s20 +0.011). On s15 specifically it goes from
**0.126 at +5d to 0.632 at +546d**.

Why it looked flat before: the old decoder was already at ~43%
wrong-per-commitment at +5 days (mean 0.428 across the old 22-session
curve), near its own floor on that axis, with almost nowhere further
to degrade except into deletions. **The flatness was a ceiling artifact
of a decoder that started out already bad on this axis, not a real
property of drift itself.** The settled decoder starts
at 12.6% (a real margin above its own floor) and visibly degrades on
both axes as elapsed time grows.

**Consequence:** **The attribution revises: BOTH SNR loss (channels
going quiet, reflected in rising deletions) and tuning change (a
shifting phoneme-to-channel mapping, reflected in rising
substitution-among-emitted) contribute, not SNR loss alone.** This
makes Phase 2's planned diagnostics (channel mortality, phoneme-tuning
matrices, subspace-angle measurements; `docs/DESIGN.md` section 6)
**more necessary, not less**: there are now two measured changes to separate
and attribute, rather than one to confirm. The methodological point from
the superseded entry is unchanged: normalizing substitutions by
reference length is confounded with emission rate, and
`substitution_rate_among_emitted` is the correct quantity to track:
that correction was right, and it is precisely what let this round
detect the rise the old, ceiling-limited architecture could not show.

---

### The cut-point sweep: calibration-data advantage is front-loaded and fades by ~100 days (2026-09-11)

**SUPERSEDED by "Per-session Switchboard counts measured directly; s20's active-session count corrected from 19 to 17" (2026-10-28).** Only the "generalization gap scales with active session count" paragraph below is affected: s20's active Switchboard-contributing session count (19) was wrong — the correct count, measured directly from the HDF5 manifest, is 17 (s20 loses three zero-Switchboard sessions, `t15.2023.08.11`/`t15.2023.11.04`/`t15.2023.11.17`, not one). s05=4, s10=9, and s15=14 are confirmed correct and are NOT superseded, and neither is this entry's headline front-loading/fades-by-~100-days result below, which does not depend on session counts. See the new entry for the full per-session breakdown.

**Question:** With the architecture settled (patch 14/4, diagonal day
layers, λ300), does the s15 drift curve's shape depend on the arbitrary
choice of a 1-15 training cut, and how much does more calibration data
actually buy against drift?

**Method:** Trained and evaluated the three remaining cut points
(s05 = sessions 1-5, s10 = 1-10, s20 = 1-20) with the identical settled
config, `--shuffled-control` against every held-out session, alongside
s15 (already trained in round 5). `scripts/04_drift_curve.py` combined
all four into one figure (`figures/drift_curve.png`).

**Result:**

*Training.* s05 best val PER **0.15765 @ epoch 111** (908 trials), s10
**0.16054 @ 105** (2,098), s15 **0.16932 @ 106** (3,150), s20 **0.16566 @
87** (3,336). No best epoch falls in the last 10% of its 150-epoch run
(111/150=74%, 105/150=70%, 106/150=71%, 87/150=58%), so none is an
upper bound. All four fall in a **0.012-wide band**.

*Late-training stability scales with training-set size.* s05
destabilizes after ~epoch 115 (train loss and gradient norms rise
sharply); its checkpoint at epoch 111 precedes this and is unaffected.
s10 and s20 show only mild late upticks; s15 is flat. Smaller training
sets overfit and destabilize: a real, if secondary, cost of choosing a
smaller cut.

**THE HEADLINE RESULT: calibration-data advantage is front-loaded and
largely erased by ~100 days.** At matched elapsed time near the
training boundary, s05 (908 trials) gives PER 0.277-0.370 at +2 to +9
days and s10 (2,098 trials) gives 0.277-0.281 at the same offsets, while
s15's (3,150 trials) +5d point is 0.177: a real, large spread. By
~100 days elapsed, s05 is 0.675, s10 0.669 (at +91d), s20 (3,336
trials) 0.643: all within **0.03 PER** of each other despite a
**3.7×** range in training-set size. **Front-loading calibration has a
short half-life: additional participant time at implant buys roughly
three months of advantage, not durable performance.** This reframes
Question 3 from "how much [calibration data] to collect once" to "how
often, and how cheaply, to top up."

**The generalization gap scales with active session count, not just
trial count.** Every cut loses session 1 to the corpus filter (100%
50-Word, `docs/DESIGN.md` section 7), so active Switchboard-contributing
sessions are: s05 4, s10 9, s15 14, s20 19. The gap between each cut's
own validation PER and its earliest held-out session's PER: s05 0.12
(0.158 val vs. 0.277 at +2d), s10 ~0.116 (0.161 vs. 0.277 at +2d), s15
0.008 (0.169 vs. 0.177 at +5d). The gap barely moves from 4 to 9 active
sessions (908→2,098 trials, +131%) then collapses from 9 to 14 (2,098→
3,150 trials, +50%), consistent with a session-count effect rather
than a trial-count one, though the s05-to-s10 step alone doesn't
cleanly isolate it. **s20 (19 active sessions, MORE trials than s15)
would be the clean test, but its behavior is AMBIGUOUS on this
question**: its earliest held-out session is +16d (non-passing, 24
trials) and its earliest PASSING session is +100d: neither sits close
enough to its own cut boundary to measure a comparable early gap. So
s20 neither confirms nor refutes the session-count reading.

**Important confound, recorded explicitly.** The same-calendar-session
comparison across cut points (the same recording evaluated under
different cuts) cannot separate "more calibration data" from "less
elapsed time," because a larger cut point is also chronologically later:
the same session sits closer to a bigger cut's boundary by
construction. The headline result above avoids this by comparing PER at
MATCHED elapsed time (not matched calendar session) across cuts of
different size, which is the unconfounded comparison.

**Shape consistency.** s05, s10, and s15 agree closely: early
(≤150d)/late (>150d) PER slope ratio 15-24×, `effective_signal` starting
0.6-0.76 and declining −0.07 to −0.09/100d, plateau PER settling
0.74-0.80. s20 appears flatter (ratio ~9×) but its first passing session
is at +100d, so it has no coverage of the steep early segment at all.
**Stated precisely rather than forced either way: the shape is
consistent everywhere s20 has coverage to check it; s20 simply cannot
observe the early climb, and this is a coverage gap, not evidence the
shape differs.**

**`t15.2025.03.16` is a confirmed outlier, not a bug in full.** Anomalous
shuffled-control PER in three of the four cut points, at three different
day offsets (+534d under s10, +518d under s15, +485d under s20), from
three separately trained models: decisive evidence it is a property of
the session, not of any one model or cut. A donor-selection bug that
ignored the model's post-patch length accounted for PART of it
(`per_shuffled` 0.998 → 0.991 on s15 after the fix, committed as
`6a30120`); the remainder is real: the session uniquely combines
`seq_len` up to 110 with recordings down to 149 frames. Flagged as an
outlier on the combined figure; not corrected further.

**Consequence:** applying the pre-committed cut-point selection rule
(`docs/FINDINGS.md`: "Pre-committed cut-point selection rule" addendum,
2026-09-11) selects **s05** as the primary cut point, with s15 reported
alongside as a robustness check. See that addendum for the rule's own
acknowledged blind spot (generalization and stability, both of which
favor s15 or s10) and why the rule was followed rather than overridden.
`figures/drift_curve.png`, `figures/error_breakdown.png`, and the four
`results/*/per_session_metrics.csv` files are the artifacts this entry
describes.

---

### The pretrained baseline checkpoint verified as a drift-independent alignment tool (2026-09-17)

**Question:** Phase 2's tuning-matrix analysis needs an alignment tool
whose own quality does not trend with session date: our own decoders
degrade across sessions, which would contaminate any tuning measurement
with the very drift being measured. Is the officially released pretrained
checkpoint (`data/external/t15_pretrained_rnn_baseline/`) usable for this,
and can it actually be loaded and run?

**Method:** Built an explicit adapter
(`src/b2t_drift/models/pretrained_baseline.py`) mapping the checkpoint's
113 top-level parameters into a `GRUCTC` subclass matching its architecture
(5 GRU layers, hidden 768, patch 14/4, `"full"` day layers): asserted
every parameter accounted for, no silent skipping. Ran inference on the
official val partition of 3 sessions, then 10 sessions spread across the
full 611-day range, correlating PER against session date.

**Result:** All 113 checkpoint parameters mapped: 45 day-weight/day-bias
pairs stacked into `DayLayers`' `"full"` representation, 20 GRU tensors
direct-copied, 2 output-layer tensors renamed, and `h0` (see the
correction below). The day-weight matrix orientation (`x @ W` vs.
`x @ W.T`) was ambiguous from shape and identity-init alone; resolved
empirically by comparing decode quality under each convention: `x @ W`
untransposed reproduces sane PER, the transposed convention does not.
3-session val PER: **0.109**, against the checkpoint's recorded 0.101
(close, within per-session sampling noise at n=20-35 trials/session).
10-session date-spread PER shows **no significant trend**: Pearson
r=0.40 (p=0.25), Spearman rho=0.28 (p=0.43), linear slope 0.013 PER/100d
(p=0.25).

**The `h0` discovery: a correction to "Block count decreases over the
recording; pretrained checkpoint uses patched input and per-session
layers" (2026-09-01).** That entry's checkpoint inspection reported 5 GRU
layers, hidden 768, patched 7,168-dim input, and 45 per-session
day-weight/day-bias pairs, but missed one further learned parameter: an
initial GRU hidden state `h0`, shape `(1, 1, 768)`, broadcast across every
layer and batch item, `max|h0| = 0.435` (genuinely nonzero, not left at
some zero default). `GRUCTC` has no equivalent (PyTorch's default
zero-initialized hidden state); `PretrainedBaselineRNN` adds it via a
minimal `_run_gru` extract-method hook on `GRUCTC` (no behavior change for
existing code). Effect on PER: real learned `h0` vs. zeroed, 0.0894 vs.
0.0904 on a 35-trial sample: **~0.001**, negligible. Nothing downstream
(the checkpoint's `val_PER`, the drift-curve models, or any reported
result) changes; this is a correction to a committed claim about the
checkpoint's architecture, not to any measurement.

**Consequence:** The checkpoint is confirmed a valid, date-independent
alignment tool. Phase 2's tuning-matrix and subspace analyses ("Tuning
change confirmed via CTC forward-backward..." and "Subspace rotation
confirmed..." below) use it via `src/b2t_drift/models/pretrained_baseline.py`,
never a model we trained.

---

### Channel death across 45 sessions: ruled out (2026-09-17)

**Question:** Does channel death (electrodes crossing the >=2 Hz "has
spiking activity" threshold, Willett et al. 2023, and staying below it)
explain a meaningful share of the drift curve?

**Method:** Recovered raw firing rates via the exact per-(trial, channel)
lattice route (`docs/DATASET.md` section 9 item 9: `sigma = 1/spacing`,
`mu = -min*sigma`, `raw = released*sigma + mu`), NOT the Poisson
approximation, across every labeled trial of all 45 sessions:
2,431,488 (trial, channel) pairs (`src/b2t_drift/analysis/firing_rate.py`,
tested against synthetic lattices). Counted, per session, TX channels with
median recovered rate below 2 Hz.

**Result:** Lattice recovery succeeded on **94.4%** of 2,431,488 pairs
(2,295,388), consistent with the earlier 95.8%-on-240-pairs estimate at
much larger scale. **19 of 11,520 (session, channel) cells had zero
successful trials** (flagged explicitly and excluded from that cell's
median, not silently dropped). Population trend in below-2Hz channel count
vs. date: **+0.425 channels/100d, p=0.131**: not significant. First
session (`t15.2023.08.11`): 2 channels below 2 Hz (135, 168, both array
55b); **both recovered** by later sessions. Last session
(`t15.2025.04.13`): 2 different channels below 2 Hz (63 ventral 6v, 211
dorsal 6v). **Exactly 2 of 256 channels cross from above to below 2 Hz and
stay there, and both cross at the final session** (2025-03-30 ->
2025-04-13, the last 14-day step in the whole 611-day range). There is
no accumulating population of dead channels across the recording, only
these two very-late crossings. Per-array breakdown: only dorsal 6v shows a
significant trend (+0.420/100d, p=0.023); ventral 6v, area 4, and 55b do
not (p=0.29, 0.093, 0.985). **Caveat, stated plainly: one significant
array of four at p=0.023 is roughly what chance alone would produce (a
Bonferroni correction at 4 comparisons would need p<0.0125): suggestive,
not an established array-specific effect.**

**Consequence:** Channel death is **ruled out** as a driver of the drift
curve. `docs/DESIGN.md` section 6 records this as settled.

---

### Clip saturation is a concurrent marker of low firing rate, not confirmed as a leading indicator (2026-09-17)

**Question:** `docs/DATASET.md` section 9 item 8 flagged clip-saturation
rate (fraction of values exactly at the +10 z-score clip) as a *candidate*
signature of channel death, untested. Does it hold up, and can it predict
death before it happens?

**Method:** Per-session, per-column saturation rate across every labeled
trial, all 45 sessions. Correlated TX-column saturation against the same
session's recovered firing rate (previous entry), within session. For the
2 channels confirmed crossing below 2 Hz and staying (63, 211), compared
saturation rate before, at, and after their crossing session.

**Result:** Global saturation rate (all 512 columns): 0.0070%, consistent
with the documented 0.009% baseline from a smaller sample. **Pooled
Spearman(saturation, firing rate) = -0.793, p approximately 0, and the
correlation is individually significant and negative in all 45 of 45
sessions**: the strongest, most consistent result in this round. Mean
saturation rate also rises with date: **+1.21e-5/100d, p=0.0023**. The
leading-indicator test is **underpowered**: both crossing channels (63,
211) cross at the very last session, leaving exactly one post-crossing
data point each. Channel 211's saturation is concurrent-to-lagging
(0.000244 mean before crossing -> 0.0000059 AT the last-alive session ->
0.000741 after; it does not rise in advance of the crossing). Channel 63
is ambiguous (0.000400 before, 0.000341 at, 0.000514 after: a small,
inconsistent change).

**Consequence:** The Week 0 hypothesis is **supported as a
cross-sectional/concurrent marker** of low firing rate (very strong,
universal, statistically overwhelming) and **untested as a leading
indicator**: not enough crossing events with enough post-crossing
sessions to say whether saturation predicts death before it happens. **Do
not upgrade this beyond what was measured**: it identifies channels that
are already low-firing, not channels about to become so.

---

### SNR loss confirmed at the feature level: TX declines 26x faster than SBP (2026-09-17)

**Question:** The project's one mechanistic prediction made before any
data was examined: TX should degrade faster than SBP, since TX reports
zero once spike amplitude falls below -4.5xRMS while SBP still registers
reduced energy in that regime. Does the data show this?

**Method:** Per session, mean recovered TX firing rate across all 256
channels (previous entries) and mean SBP magnitude (mean absolute
z-scored value across all 256 SBP columns), both normalized to their
session-1 value for comparability. Linear slope vs. date for each; a
direct test of whether the gap between them widens.

**Result:** TX normalized-rate slope **-0.0399/100d (p=4.9e-7)** vs. SBP
normalized-magnitude slope **-0.00154/100d (p=0.045)**: a **26x ratio**.
Over the observed 611 days that is roughly a **24% TX decline against 1%
for SBP**. The gap (SBP_norm - TX_norm) widens significantly: slope
+0.0383/100d, **p=3.0e-6**. TX rate tracks the s05 PER curve at
**Spearman rho=-0.750 (p=6.8e-6)**; SBP does not (**rho=-0.191, p=0.34**).
Electrode 211 (dorsal 6v, one of the two channels confirmed crossing below
2 Hz and staying, previous entries) is the clean single-electrode
mechanism case: TX collapses from 1.0x to 0.05-0.2x normalized by day 200,
while SBP holds flat near 1.0x throughout, even ticking slightly above it.
**Caveat: SBP's own p=0.045 is borderline (not comfortably below 0.05):
the robust claim is the RATIO between the two slopes and the significantly
widening gap, not that SBP itself is confirmed to be declining.**

**Consequence:** SNR loss is confirmed at the feature level. **See
"SBP-only is not more drift-resistant than TX-only, despite SNR loss's
feature-level prediction: TX/SBP redundancy is why" (2026-09-18) for why
this does not extend cleanly to a decoder-level prediction.**

---

### Which channels the decoder depends on differs between an early and a late held-out session, with no rank correlation between them (2026-09-17)

**SUPERSEDED in its interpretation by "Late-session channel dependence is
unmeasurable at this trial count, not diffuse" (2026-09-17).** The
conclusion below, that late-session dependence is "diffuse and
non-reproducible," overstates what a single ablation run on ~94 trials can
show. The superseding entry's split-half test demonstrates the late-session
measurement itself is too noisy to distinguish "genuinely diffuse" from
"unmeasurable." The per-channel delta-PER numbers below are not wrong and
are not superseded: only the interpretive claim about what the
early-vs-late difference MEANS is corrected.

**Question:** Firing rate measures whether a channel is active, not
whether the decoder uses it. Using the s05 checkpoint (the primary cut
point), which channels does the decoder actually depend on, at an early
vs. a late held-out session?

**Method:** Per-channel ablation (zero both TX and SBP columns for that
electrode, re-evaluate) across all 256 electrodes, on one early held-out
session (`t15.2023.08.27`, +2d) and one late one (`t15.2025.04.13`,
+597d). The full 256-channel sweep was not prohibitive (92s and 51s
respectively).

**Result:** Early session (PER=0.276): 4/256 channels cost >0.01 PER
(max: channel 169 at 0.023), 83/256 negligible (<0.001). Late session
(PER=0.795): **0/256 channels cost >0.01 PER**, 59/256 negligible.
Weak/no correlation between firing rate and per-channel dependence in
either session (early rho=0.083 p=0.18 n.s.; late rho=0.126 p=0.044, small
effect): high-firing channels are not reliably the ones the decoder
depends on. Spearman(delta-PER_early, delta-PER_late) across all 256
channels: **rho=-0.023, p=0.71**, no relationship; top-20-most-depended-
on-channel overlap is 2/20 (chance baseline ~1.6).

**Consequence:** Channel-level dependence appeared diffuse and
non-reproducible by the late session, read at the time as consistent with
the previously-reported effective-signal collapse. **See the superseding
entry for why this reading does not survive a direct test of whether it is
measurable at all.**

---

### Late-session channel dependence is unmeasurable at this trial count, not diffuse (2026-09-17)

**Supersedes** the interpretation (not the numbers) in "Which channels the
decoder depends on differs between an early and a late held-out session,
with no rank correlation between them" (2026-09-17), which read the
early-vs-late rank-correlation gap as evidence that channel-level
dependence becomes diffuse by the late session.

**Question:** That gap has two possible causes the single-session analysis
cannot distinguish: dependence genuinely diffused across channels, or
single-session ablation is too noisy to measure dependence at all in the
late regime. These support very different claims.

**Method:** Split each of the same two sessions' trials into two random
halves (fixed seed), ran the full 256-channel ablation independently on
each half, and correlated per-channel delta-PER between halves within the
same session.

**Result:** Early session (+2d): half A (87 trials) vs. half B (88
trials), Spearman rho=0.116 (p=0.064, marginal on the full 256-channel
correlation) but **top-20-most-depended-on-channel overlap = 9/20**,
against a hypergeometric chance baseline of ~1.6, **P(overlap >= 9 by
chance) = 1.6e-6**. Late session (+597d): half A (47 trials) vs. half B
(47 trials), Spearman rho=0.054 (p=0.39), **top-20 overlap = 2/20,
P(overlap >= 2 by chance) = 0.48**, indistinguishable from chance.

**Consequence:** The early session's halves agree (a concentrated,
reproducible set of ~9-10 channels the decoder depends on, even though the
full-256 rank correlation is diluted by mostly-noise channels); the late
session's halves do not agree at all. Per the decision rule fixed before
running this: **late-session channel dependence is UNMEASURABLE at ~94
trials, not evidence that dependence became diffuse.** The earlier
entry's "diffuse and non-reproducible late" language is retracted; the
corrected statement is that this method cannot resolve channel-level
dependence at the late session's trial count, one way or the other.

---

### Tuning change confirmed via CTC forward-backward, and it survives the healthy-channel control (2026-09-17)

**Question:** Does the phoneme-to-channel mapping itself shift over 20
months, independent of channels simply going quiet (SNR loss)?

**Method:** Implemented CTC forward-backward
(`src/b2t_drift/evaluation/ctc_forward_backward.py`) to get per-timestep
posterior probabilities over the 41 classes from the pretrained
checkpoint's emissions and each trial's known target: alpha/beta
recursion in log space, verified against `nn.CTCLoss`'s marginal
likelihood to numerical tolerance (8 unit tests, including a
forced-alignment case where T equals the raw target length with no
consecutive repeats, which has exactly one valid path regardless of the
model's actual output). Patched-resolution posteriors (one frame per 4 raw
bins) mapped back to 20ms bins by repeating each frame's posterior across
its `patch_stride`, covering every raw bin exactly once with no gaps or
double-assignment. Built one 41x512 tuning matrix per session
(posterior-weighted mean feature vector per class, all labeled trials, all
45 sessions): **0 infeasible trials** across every trial processed.
Correlated tuning matrices across all 990 session pairs (mean PER-PHONEME
correlation of the two matrices' same-class rows, not the flattened-matrix
correlation, chosen because flattening conflates the across-phoneme
pattern with the within-phoneme channel structure this analysis actually
wants; see `src/b2t_drift/analysis/tuning.py`) against days separating the
pair.

**Result:** Correlation vs. days-apart: **slope=-0.0564/100d,
p=1.4e-163** across 990 pairs. **THE LOAD-BEARING CONTROL: recomputed
excluding channels 63 and 211 (confirmed dying) and the bottom decile of
mean firing rate (27 electrodes total, both TX and SBP columns):
slope=-0.0562/100d, p=3.8e-162, essentially unchanged.** This control
matters because a dying channel's tuning would appear to change simply
because its signal is vanishing into noise: if the decay were SNR loss
in disguise, excluding the degraded channels would flatten it. It does not
move. Per-phoneme breakdown: fastest-decaying are silence (class 40,
-0.1005/100d), Z, IY, AY, ER; slowest are G, OY, K, UH, JH. Silence decays
at roughly double the real-phoneme mean (-0.0553/100d, std 0.0115),
consistent with silence's different acoustic/motor character, not a red
flag for the aggregate (each class contributes equally to the reported
mean, not weighted by token frequency, so silence's faster decay does not
mechanically inflate the headline number).

**The alignment-free cross-check (Task C), independent of any model.**
For each session, weighted each trial's mean feature vector by how often
each phoneme appears in that trial's target, no alignment at all. Same
correlation-vs-days measurement: **slope=-0.0365/100d, p=5.5e-148**,
same direction, **~65% the magnitude** of the forward-backward figure. The
smaller magnitude is expected: every trial average mixes ALL of its
phonemes together with its silence, diluting any one phoneme's specific
signature: coarser by construction, not a disagreement. **The
forward-backward figure (-0.0564/100d) is the better magnitude estimate;
the alignment-free cross-check is the better evidence that the effect is
real, since it cannot be an artifact of the model used for alignment.**

**Consequence:** Tuning change is confirmed, independent of SNR loss
(survives the healthy-channel control) and independent of the alignment
method (survives the alignment-free cross-check). Figure:
`figures/tuning_correlation_decay.png`.

---

### Subspace rotation confirmed, and it tracks PER as tightly as tuning change (2026-09-17)

**Question:** Has the population-activity manifold rotated in 256-d
space, independent of the per-channel affine changes the rolling z-score
already removes? This is the failure mode Arm 1 (unsupervised
distribution alignment, `docs/DESIGN.md` section 5) is built to correct,
if it exists.

**Method:** Per session, PCA on the 512-dim features pooled across all
labeled trials (via an incrementally-accumulated 512x512 covariance
matrix, never materializing the full per-bin data at once). Principal
angles (via SVD of the two subspaces' cross-product) between the top-20
subspaces of every one of the 990 session pairs, mean angle vs. days
separating the pair. Repeated on healthy channels only (same exclusion as
the tuning-change control). Separately, built a reference subspace from
the s05 training block (sessions 1-5) and measured each held-out
session's angle to it, against the s05 PER curve.

**Result:** Dimensionality needed for 90% variance: mean ~270 components
(range 248-287), **no significant trend with date** (slope -0.567/100d,
p=0.511): the manifold's dimensionality itself is stable; what moves is
its ORIENTATION. Principal angles across all 990 pairs: **slope=3.17
deg/100d, p=2.5e-121**. **Healthy-channel-only control: slope=3.05
deg/100d, p=1.7e-111, barely reduced from the all-channel figure**, the
same pattern as the tuning-change control. Angle-to-s05-reference vs. the
s05 PER curve (27 matched sessions): **Spearman rho=0.908, p=6.3e-11**.

**Stated precisely rather than overclaimed: this is NOT tighter than
tuning change's own correlation with PER (rho=-0.898 to -0.903, depending
on channel restriction, previous entry). With 27 sessions, 0.908 and
0.90 are statistically indistinguishable.** Both track PER equally
tightly; this is part of why the underlying measured changes cannot be ranked
against each other by correlation strength (`docs/DESIGN.md` section 6).

**Consequence for Arm 1.** Rotation is large, statistically robust, and,
because it survives the healthy-channel control, **independent of
channel health**, meaning it is not simply a symptom of dying electrodes
distorting the population geometry. This gives Arm 1 (unsupervised linear
alignment, no labels needed) a real, MEASURED target rather than a
hopeful premise: there is a genuine rotation of this magnitude for a
linear map to correct. `docs/DESIGN.md` section 5 is updated. Figure:
`figures/subspace_angles.png`.

---

### SBP-only is not more drift-resistant than TX-only, despite SNR loss's feature-level prediction: TX/SBP redundancy is why (2026-09-18)

**Question:** SNR loss predicts an SBP-only decoder should be more
drift-resistant than a TX-only one (TX declines 26x faster: "SNR loss
confirmed at the feature level" above). Does training and evaluating both
confirm this?

**Method:** Trained TX-only (columns 0-255) and SBP-only (columns
256-511) decoders on `temporal_s01_s05_switchboard`, settled architecture
otherwise unchanged, 150 epochs, early stopping disabled (a
`feature_subset` option added to `TrialDataset`/`collate_trials`/
`build_shuffled_batches` and a `--feature-subset` flag to
`02_train.py`/`03_evaluate.py`, tested). Evaluated both with
`--shuffled-control` across all held-out sessions.

**Result: level and slope point in OPPOSITE directions; state both, do
not read one as confirming the other.** SBP-only reached better val PER
than TX-only (**0.16306 vs. 0.18238**) and better PER in **24 of 27**
held-out sessions: essentially the whole 611-day range, not just at the
start. But its PER slope is **+0.0693/100d against TX-only's +0.0507:
37% STEEPER, not flatter**, and its `effective_signal` erodes faster too
(-0.0848 vs. -0.0634/100d). **SBP is the better feature; it is not the
more drift-resistant one. The decoder-level test does not confirm the
feature-level prediction; the clean "route around degrading TX by
weighting SBP" story is not supported end to end.**

**Redundancy evidence.** Full-feature (512-dim) beats TX-only by 0.0247
PER (13.6% relative) but beats SBP-only by only 0.0054 (3.3% relative):
adding TX on top of SBP buys almost nothing, while adding SBP on top of TX
buys a great deal. Caveat: confounded with input width (256 vs. 512
columns), not cleanly separable from "more features help" in general.

**The direct test: per-electrode TX-vs-SBP correlation.** Column i (TX)
and i+256 (SBP) are the same physical electrode. Per session, correlated
each of the 256 electrodes' TX time series against its own SBP time
series across all bins of that session's labeled trials, took the median
across electrodes. **Median correlation stays high throughout the full
611 days: 0.666 to 0.782, overall median 0.717**, never drops to a low
or moderate level. There IS a small, statistically significant declining
trend (slope=-0.00933/100d, p=0.0002, driven mostly by the first ~250
days, then roughly flat): a real, modest weakening, not a divergence:
correlation never falls out of the 0.66-0.78 band.

**This explains the contradiction.** TX and SBP are, throughout the whole
recording, substantially redundant views of the same underlying spiking
activity at each electrode (median r~0.72): both derive from the same
shrinking spikes as an electrode encapsulates, through different
transforms (a hard count vs. a continuous power). Because they are
watching the same underlying process rather than two independent ones,
whatever degrades one's DECODE quality over time (tuning change, subspace
rotation, both confirmed to affect the whole 512-dim population, not a
TX-specific effect) degrades the other's too, even though TX's own raw
FEATURE MAGNITUDE happens to decline faster. The feature-level SNR-loss
prediction is about raw signal amplitude; the decoder-level result is
dominated by the other two measured changes (tuning change, subspace rotation),
which do not distinguish TX from SBP. Figure:
`figures/tx_sbp_pair_correlation.png`.

**Consequence:** Confirms `docs/DESIGN.md` section 6's multi-causal
verdict from the other direction: SNR loss alone cannot be the whole
story, because a decoder built entirely from the feature that degrades
least (SBP) is not thereby protected from drift.

---

### Full fine-tuning is the best use of a fixed participant-time budget, at every elapsed time measured (2026-09-23)

**Question:** Of the four recalibration arms (0: no adaptation, 1:
unsupervised alignment, 2: day layer only, 3b: full fine-tune; 3a:
output layer only, added as an intermediate-cost point), which recovers
the most PER per minute of participant time, and does the answer change
with elapsed time?

**Method:** The recalibration grid: 5 evaluation sessions (+9/+42/+58/
+329/+597d), N in {0,10,25,50,100} calibration sentences, 5 random
calibration/evaluation splits per (arm, session, N) cell, identical
partitions across arms. `figures/recalibration_arms.png`;
`results/recalibration_grid.csv`; `results/recalibration_summary.csv`.
Sentence-de-duplicated PER (`docs/FINDINGS.md`: "Sentence-level leakage
in the calibration split, and its effect") used throughout below.

**Result:** Arm 3b (full fine-tune) wins every (session, N) cell **except
+9d at N=10**, where it is worse than doing nothing on 4/5 splits (mean
PER 0.3892 vs. Arm 0's 0.3706). See the companion entry on the minimum
viable budget. Per-session recovery (mean PER over 5 splits; minutes =
mean calibration time for that N):

| Session | Arm 0 | N=10 (~2.6min) | N=25 (~7min) | N=50 (~14min) | N=100 (~27min) |
|---|---|---|---|---|---|
| +9d | 0.3706 | 0.3892 (worse) | 0.3335 | 0.3032 | 0.2753 |
| +42d | 0.5779 | 0.5375 | 0.4632 | 0.4313 | 0.3754 |
| +58d | 0.5549 | 0.5026 | 0.4426 | 0.3892 | impossible (83 trials) |
| +329d | 0.7923 | 0.7683 | 0.7543 | 0.7342 | 0.7072 |
| +597d | 0.7961 | 0.7733 | 0.7431 | 0.7182 | impossible (94 trials) |

Arm 3b beats the best FREE option (min of Arm 0, 1a, 1b, i.e. doing
nothing or aligning without labels) from N=10 (~2.5–3min) at +42d, +58d,
+329d, and +597d, and from N=25 at +9d. Arm 3a (output layer only) needs
N=10–50 depending on session; Arm 2 (day layer only) needs N=100 or never
beats it at all (fails outright at +329d; see "A diagonal day layer
lacks the capacity to correct accumulated drift" below).

**Consequence:** For this decoder and this drift regime, the DECISION
question (`docs/DESIGN.md` section 4) has a single answer across the
whole +9d–+597d range: spend the participant-time budget on full
fine-tuning, not a cheaper partial arm: the intermediate arms' lower
per-minute cost never translates into a per-minute PER advantage large
enough to close the gap to Arm 3b. The one exception (+9d, N=10) is its
own finding, below.

---

### Recalibration slows the decline but does not restore function (2026-09-28)

**PARTIALLY SUPERSEDED by "Recovery fraction's 'two sharp steps' withdrawn: three measured levels, two transitions of unresolved sharpness" (2026-11-05).** Only the characterization of the decline as happening "in two sharp steps" is withdrawn. The recovery-fraction values themselves, the -15.5 (N=25) / -23.1 (N=50) point difference between the +58d and +184d readings, and every other number in this entry are NOT superseded.

**Question:** How much of the accumulated drift does the best arm (3b)
actually recover, relative to what the decoder could achieve with no
drift at all, and does that recovered share change with elapsed time?

**Method:** `recovery_fraction = (Arm0_PER − Arm3b_PER) / (Arm0_PER −
reference_PER)`, `reference_PER = 0.15765`: the s05 checkpoint's
validation PER, measured WITHIN its own training sessions, not a
held-out day-zero session. This makes `reference_PER` a slightly
optimistic ceiling, so every fraction below is a LOWER BOUND on true
recoverability. Computed on sentence-de-duplicated PER, at 5 calibration
splits per cell, across 7 sessions: the original 5 (+9/+42/+58/+329/
+597d) plus two added specifically to close the +58d–+329d gap, +184d
and +259d (`docs/FINDINGS.md`: "Task 1's gap-fill..." below).
`figures/recalibration_recovery_fraction.png`;
`results/recalibration_recovery_fraction.csv`.

**Result:** Recovery fraction (mean ± sd over 5 splits), N=25/N=50:

| Session | N=25 | N=50 |
|---|---|---|
| +9d | 16.8% ± 4.1 | 31.2% ± 3.3 |
| +42d | 26.7% ± 4.0 | 35.3% ± 1.7 |
| +58d | 28.4% ± 2.0 | 42.6% ± 5.4 (29 eval trials, too small to trust) |
| +184d | 12.8% ± 1.7 | 19.5% ± 2.8 |
| +259d | 12.8% ± 1.1 | 19.9% ± 2.7 |
| +329d | 6.0% ± 1.9 | 9.3% ± 1.5 |
| +597d | 8.2% ± 2.1 | 12.2% ± 1.1 |

Descriptive regression on the 7 session means (labeled DESCRIPTIVE, not a
fitted drift model): N=25 slope −3.14 points/100d (p=0.050); N=50 slope
−4.86 points/100d (p=0.025). With the two gap-fill sessions added, these
are significant where the original 5-session slopes were not (N=50 was
p=0.074). The decline is NOT smooth: it happens in two sharp steps
(+58d→+184d, and again +259d→+329d) separated by two flat plateaus
(+9–+58d high, +184–+259d mid, +329–+597d low): the steepest single step
(+58d→+184d) accounts for −15.5 (N=25) / −23.1 (N=50) points of the total
decline by itself.

**Practical statement, with corrected (de-duplicated) numbers:** at
+597d, ~15 minutes of labeled speech buys PER 0.718 against Arm 0's
0.796. At +329d, ~26 minutes buys 0.707 against 0.792 (headline
unchanged by the leak correction; see the leak entry). Even the best
cell measured (+9d, ~26 minutes: PER 0.275) only matches a +2d decoder's
performance (0.277), not the 0.158 the same checkpoint reaches inside its
own training sessions. Neither +329d nor +597d reaches PER 0.70 at any
measured budget.

**Consequence:** Recalibration (full fine-tuning) slows the PER decline;
it does not restore day-0 function, at any tested budget, once drift has
substantially accumulated. `docs/DESIGN.md` section 5's Question 3 answer
is updated accordingly.

---

### There is a minimum viable calibration budget, and it depends on accumulated drift (2026-09-23)

**Question:** Is more labeled data always at least as good as less, or can
a small labeled budget make the decoder worse than doing nothing?

**Method:** Same recalibration grid as above; recovery_fraction at N=10
on the least-drifted session (+9d).

**Result:** At +9d, N=10 gives recovery_fraction **−8.8% ± 7.1%**: ten
labeled sentences on the EASIEST session make the decoder worse than no
adaptation at all (4/5 splits). This is catastrophic forgetting appearing
exactly where intuition says it should not: least drift, least to gain,
most to lose. At +9d, N=25 is needed to break even (+16.8% ± 4.1%). At
+329d, by contrast, even N=10 helps (+3.8% ± 1.2%): the same arm, same
step count, same learning rate.

**Consequence:** When Arm 0 is already close to the achievable floor for
that session, 10 fine-tuning sentences carry more optimization noise than
correctable signal, and full fine-tuning has enough capacity to overfit
to that noise. The minimum viable calibration budget is not a fixed
number of sentences: it depends on how much accumulated drift there is
to correct. A recalibration protocol that always spends a small fixed
budget (e.g. "10 sentences on every new day") would harm performance on
early, lightly-drifted days.

---

### Subspace rotation is a correlate of drift, not a correctable cause (2026-09-23)

**Question:** Phase 2 (`docs/FINDINGS.md`: "Subspace rotation confirmed,
and it tracks PER as tightly as tuning change") found subspace rotation
tracking PER at Spearman ρ=0.908 (the tightest relationship in the
project), and `docs/DESIGN.md` section 5 concluded Arm 1's premise "is no
longer hypothetical." Does correcting that rotation, without labels,
actually recover the PER it predicts?

**Method:** Arm 1a (CORAL) and Arm 1b (Procrustes) in the recalibration
grid use only the calibration trials' FEATURES (no transcripts) to fit an
unsupervised linear alignment, targeting exactly the rotation Phase 2
measured. Correlated each arm's PER improvement over Arm 0 against that
session's measured principal angle (`docs/FINDINGS.md`'s Phase 2 entry),
across the grid's 5 sessions (3 at N=100).

**Result:** Measured angles: 51.9° / 51.7° / 55.9° / 65.0° / 64.0° at
+9/+42/+58/+329/+597d. Arm 1b's improvement vs. angle: Spearman
**−0.9 to −1.0** across N=10/25/50/100: OPPOSITE of "more rotation, more
benefit": Arm 1b gains +0.004 to +0.014 PER at ≤+58d (smaller angles) and
LOSES −0.006 to −0.014 at +329d/+597d (larger angles). Arm 1a (CORAL)
shows no coherent relationship (ρ≈−0.1 to 0) and is erratic in sign
across sessions. At +597d, no unlabeled arm beats doing nothing. Arm 1's
best measured gains (+0.004 to +0.014 PER) are an order of magnitude
below Arm 3b's.

**Reconciliation with Phase 2, stated explicitly:** these results are
compatible, not contradictory. Rotation is a CORRELATE of drift, not a
correctable CAUSE. Array micromotion through tissue plausibly produces
both the coherent reorientation (rotation) and the information loss
(reduced spike amplitude, altered neuron sampling; `docs/DESIGN.md`
section 6) simultaneously, as two consequences of one physical process.
Undoing the rotation does not restore the information, because the
information was not lost BECAUSE OF the rotation. **General lesson: a
measurement that predicts an outcome almost perfectly can still be
useless as an intervention target.** Prediction and correction are
different claims, and confirming one does not confirm the other.

**Caveat, stated honestly:** n=5 sessions (3 at N=100), and the measured
angle is confounded with elapsed time (larger angles occur at later
sessions): the negative correlation could partly reflect "later
sessions are harder for Arm 1 for any reason," not rotation specifically.
The result is weak evidence on its own. It nonetheless directly
contradicts the current DESIGN.md claim and is sufficient to withdraw it;
`docs/DESIGN.md` section 5 is updated to reflect this.

---

### A diagonal day layer lacks the capacity to correct accumulated drift (2026-09-24)

**Question:** `docs/DESIGN.md` section 5 (Phase 1) left open whether a
dedicated diagonal day-layer fit on one heavily-drifted session (frozen
backbone, no joint-training anchor from the other 44 sessions) has the
capacity to recover accumulated drift.

**Method:** Arm 2 in the recalibration grid, at a learning rate (1e-5)
measured to be safe on the hardest session (`docs/FINDINGS.md`: "A
learning rate measured for one arm does not transfer to another").
Measured the fitted layer's own deviation from identity (`max|scale-1|`,
the diagonal parameterization's only weight term) and swept
`day_layer_penalty` λ over {30, 300, 3000} at this learning rate, at both
N=10 and N=100.

**Result:** At the safe learning rate, the fitted layer barely leaves
identity: `max|scale-1| ≈ 0.004` at every λ tested, at both N=10 and
N=100 (versus ≈0.09–0.10 at the old, unsafe lr=1e-3: an order of
magnitude more movement, and still constrained enough to be governed by
lr rather than λ). λ is irrelevant across this 100x range at the safe lr:
mean PER differs only in the 4th decimal between λ=30/300/3000.

**Answer: no, 1,024 parameters is not enough.** It is enough to remain
stable when trained jointly across 45 sessions (`docs/DESIGN.md` section
10's day-layer keep/drop rule), and not enough to correct a single new,
heavily-drifted day from a small labeled budget: the mechanism's own
capacity is the limit, not the optimizer.

**Caveat, stated honestly:** the safe learning rate was chosen
specifically to avoid harm on the hardest session (+329d), so this
capacity ceiling is measured at a THROTTLED setting: Arm 2 is further
below its own best-case ceiling on easier sessions than this number
suggests. At the old, unsafe lr=1e-3, Arm 2 on +42d reached PER
improvements of +0.061 (N=10) and +0.106 (N=100), against the safe grid's
+0.0095 and +0.0178 there. Arm 3b still wins at +42d regardless (+0.040 /
+0.200), so the cross-arm ranking is unaffected, but "Arm 2 is a no-op"
is a claim about a globally-safe learning rate, not a universal capacity
limit independent of lr.

---

### A learning rate measured for one arm does not transfer to another (2026-09-24)

**Question:** Arm 3b's learning rate (1e-3) was measured by a dedicated
sweep specifically to avoid baking in a guessed value. That measured
value was then reused, unmeasured, for Arm 2 (1,024 trainable parameters)
and Arm 3a (21,033): four and three orders of magnitude smaller than Arm
3b's 16.6M. Does it still hold?

**Method:** Diagnosed via Arm 2's behavior on the hardest session
(+329d): first tested whether stronger identity-anchoring (λ) could fix
an observed failure (the "too-weak regularization" hypothesis), then
swept learning rate directly, then re-swept λ at the corrected rate.

**Result:** No. It does not transfer, and the failure mode is
specifically consistent with the anchor being overpowered, not with
insufficient anchor strength. At lr=1e-3, N=10, sweeping λ over
{30,300,3000}: the fitted layer's `max|scale-1|` landed at **0.0996 /
0.0992 / 0.0918** (essentially unchanged across a 100x range), while the
penalty's own share of the total loss rose from **0.16% to 9.41%**. The
CTC gradient at this lr overpowers the identity anchor regardless of how
hard the anchor pulls. Consequently, STRONGER anchoring made Arm 2's
+329d failure WORSE, not better (4/5, 4/5, 5/5 splits worse than Arm 0 at
λ=30/300/3000 respectively): the opposite of what the "too-weak
regularization" hypothesis predicted, refuting it. Arm 3a was harmed too,
in a different way: at lr≥1e-4, its N=10 improvement REVERSES between
step 200 and step 400 (starts better than Arm 0, ends worse): classic
overfitting to a tiny calibration set, on the arm with 21,000 parameters
to fit from 10 sentences.

**Consequence:** A Task-1-style learning-rate sweep is required per arm,
not per project: the whole point of measuring lr 1e-3 for Arm 3b was to
avoid a guessed value, and that discipline was silently dropped the
moment the value was reused elsewhere. Arm 2 and Arm 3a's final grid
values (lr=1e-5/400 steps and lr=1e-4/200–400 steps respectively) were
each measured the same way Arm 3b's was, on a middle session and verified
on the pathological one, before being used for the grid
(`docs/FINDINGS.md`: "A diagonal day layer lacks the capacity...").

---

### Sentence-level leakage in the calibration split, and its effect (2026-09-24)

**Question:** The recalibration grid's stated protocol was "never
evaluate on a calibration TRIAL." Is that the right unit of leakage?

**Method:** Checked whether any evaluation trial's `sentence_label`
(case/whitespace-normalized) also appeared among that same split's
calibration trials, across every (session, N, split) cell in the grid.

**Result:** No. The right unit is the SENTENCE, not the trial: the same
sentence text can occur as two distinct trials within one session, so a
model fine-tuned on one trial could be scored on the other and get credit
for memorizing a phoneme sequence rather than decoding it. 32 of 90
labeled cells were affected, all at +58d (13 cells) and +329d (19 cells);
zero overlap at +9d, +42d, +597d. Overlap was 0.8–10.9% of eval trials
per affected cell (mean 10.9% at +58d N=50; mean 5.5%, max 8.5%, at +329d
N=100).

Fitted weights for Arm 2/3a/3b were never saved during the grid run, so
Arm 3a and Arm 3b were REFIT on the 32 affected cells (identical recipe)
and each refit model scored on both the original and the
sentence-de-duplicated evaluation set, isolating the leak's effect from
ordinary refit-to-refit training noise (dropout is unseeded; two refits
of the "same" cell differ by ≈0.001–0.003 PER on that basis alone). Arm
0/1a/1b needed no refit (deterministic, dropout-off evaluation) and were
re-scored exactly (recomputed full-eval PER matched the recorded grid
values to 0.00e+00). Arm 2 was not refit: analytically unaffected, since
a transform that stays within 0.004 of identity (see the capacity entry
above) has no capacity to memorize a specific sentence's phoneme
sequence.

**Within-model leak effect on Arm 3b's improvement over Arm 0** (negative
= the leak inflated it): +58d: −0.0037 / −0.0062 / −0.0066 (N=10/25/50);
+329d: −0.0004 / −0.0005 / −0.0005 / **+0.0014 (sd 0.0024)** (N=10/25/50/
100). At +329d N=100 (the headline late-session cell) the effect is
indistinguishable from zero: the "~26 minutes recovers 0.081 PER" claim
becomes **0.0866** de-duplicated, unchanged in substance. At +58d the
leak inflated Arm 3b by 0.004–0.007, small next to gains of 0.05–0.17
there. Arm 1a/1b change by at most 0.0006 at +329d; Arm 3a by at most
0.0004. One cell is too small to trust regardless of the leak: +58d,
N=50, drops to 29 evaluation trials after de-duplication.

**Prevention:** `split_calibration_trials` (`src/b2t_drift/evaluation/
calibration.py`) gained an explicit `dedup_sentences` option
(`split_off_sentence_overlap` does the work), covered by 6 new tests. The
default remains `False`: the Phase 3 grid used `False` throughout, and
that choice is now explicit and self-documented rather than an implicit
gap in the stated protocol. `results/recalibration_grid.csv` gained
`n_eval_overlap`, `n_eval_dedup`, `arm0_per_dedup`, `per_dedup`,
`per_refit_full`, and `dedup_source` columns; every original column is
byte-for-byte unchanged.

**Consequence:** The Phase 3 headline (full fine-tuning wins at every
elapsed time, and slows but does not restore function) survives
unchanged. The protocol statement itself was wrong and is corrected here
and in `docs/DESIGN.md`; any future N-vs-eval split in this codebase
should default to asking whether sentence-level (not just trial-level)
independence is required.

---

### Task 1's gap-fill: two intermediate sessions turn a descriptive slope into a significant one (2026-09-28)

**PARTIALLY SUPERSEDED by "Recovery fraction's 'two sharp steps' withdrawn: three measured levels, two transitions of unresolved sharpness" (2026-11-05).** This entry's "roughly HALFWAY between" reasoning, used here as evidence FOR a discrete step, in fact argues the opposite: landing between two endpoints is what a linear decline predicts. The measured recovery-fraction values (19.5%/19.9% at N=50, 12.8%/12.8% at N=25) and the significance results for the N=25/N=50 slopes are NOT superseded.

**Question:** The recovery-fraction curve had no measured session between
+58d and +329d (a 271-day window chosen by session selection, not a data
limitation: s05 has passing sessions at +184d through +294d). Does
filling it change the shape or the statistical support of "recovery
falls with elapsed time"?

**Method:** Ran Arm 3b (+ matched Arm 0, identical partition) at N=25 and
N=50, 5 splits, on the two s05 passing sessions in +150d–+300d that
maximize the minimum spacing among {+58d, x, y, +329d}: **+184d
(t15.2024.02.25, 216 trials)** and **+259d (t15.2024.05.10, 135
trials)**: spacings of 126/75/70 days, the most even split the available
sessions (184/196/203/205/259/294d) allow. Same lr/step schedule and
calibration-split seeds as the main grid. UNLIKE the main grid, used
`dedup_sentences=True` for these new cells (recorded explicitly, per the
prevention above); measured overlap was 0% at both sessions, all N, all
splits, so de-duplication changed nothing here in practice.

**Result:** Recovery fraction at N=50: +184d 19.5% ± 2.8, +259d 19.9% ±
2.7, almost identical to each other, and roughly HALFWAY between the
+58d plateau (~36–43%) and the +329d/+597d plateau (~9–12%). With these
two points added (7 sessions total, 3 at N=100), the N=25/N=50
recovery-fraction slopes become statistically significant (p=0.050,
p=0.025) where the original 5-session slopes were not (best: N=50,
p=0.074). The decline is NOT smooth: essentially all of it happens in two
sharp steps, +58d→+184d and +259d→+329d, separated by flat plateaus
before, between, and after, not a continuous decay with elapsed time.
40 new rows appended to `results/recalibration_grid.csv`, marked
`is_gapfill=1`; all 545 pre-existing rows are byte-for-byte unchanged.
`figures/recalibration_recovery_fraction.png` regenerated across all 7
sessions.

**Consequence:** The gap was worth filling: it changed a
just-shy-of-significant descriptive slope into a significant one, and
revealed that the underlying shape is step-wise, not smooth. Confirms
`docs/FINDINGS.md`: "Recalibration slows the decline but does not restore
function" with stronger statistical support and a corrected description
of the curve's shape.

---

### Length-distribution documentation errors traced to one session's statistics; padding waste re-measured (2026-10-21)

**Question:** `docs/DATASET.md` section 6 stated `n_time_steps` spans
138-1025+ bins, median ~527 (~10.5s); `docs/DESIGN.md` section 8 stated it
spans 138-2179+ bins. The two disagreed, and neither number carried a
sample size. What is the true distribution, and where did the two wrong
numbers come from?

**Method:** Read `n_time_steps` directly from every labeled trial's HDF5
attributes (`iter_trial_ids`, train+val partitions, all 45 sessions),
not from the per-session `session_manifest.csv` aggregates alone. Then
checked whether 1025 or 2179 matched any single session's, cut point's,
or corpus subset's own max.

**Result:** True distribution across all 9,498 labeled trials: min 138,
max **2475**, median **845**, p99 **1806**. In seconds (20ms bins): 2.8
to 49.5s, median 16.9s, p99 36.1s. 47 trials (0.49%) exceed 2000 bins; a
thin but real populated tail, not a single outlier. 2,651 trials (27.9%
of the dataset) exceed the old "1025" figure. **1025 and "median ~527"
are exactly `t15.2023.08.11` (session 1)'s own `max_time_bins` and
`median_time_bins`** from `session_manifest.csv`, mistaken for the
dataset's. **2179 matched no session, cut point, or corpus subset** in
the manifest; its source could not be identified. Two numbers derived
from the wrong max were consequently also wrong: the seconds range
(2.8-20.5s, derived from 1025 x 20ms) and the "7.4x" length spread
(1025/138); the true spread is **17.9x** (2475/138).

Re-measured the padding-waste claim this error sat beside
(`docs/DESIGN.md` section 8: "35.5% vs. 13.7% bucketed, at a 32-trial
pool"), since no entry here documented it. Computed `1 - (sum of true
lengths) / (batch_size * sum of batch maxima)` at `batch_size=32`,
`drop_last=True` (matching `configs/experiment/switchboard_baseline.yaml`
and `scripts/02_train.py`'s train-partition DataLoader) over the same
9,498-trial population, for (a) random batching and (b) the real
`LengthBucketedSampler` (`src/b2t_drift/data/collate.py`), averaged over
50 shuffles/seeds. Random: mean **47.6%** (std 0.19%, range
47.2-47.9%). Bucketed: mean **9.3%** (std 0.06%, range 9.2-9.5%). Both
differ materially from the original 35.5%/13.7%, in opposite directions,
consistent with the true spread being more than double what the
original (wrong) range implied. "Pool" is the same thing as
`batch_size` here; `LengthBucketedSampler` has no separate pool
parameter. This measurement is now a tracked, re-runnable script,
`tools/measure_padding_waste.py` (`uv run tools/measure_padding_waste.py`),
not a one-off: the prior version lived in `scratch/` and was therefore
exactly as untraceable as the number it was replacing.

**Consequence:** `docs/DATASET.md` section 6's distribution line and
section 14's length-spread bullet, and `docs/DESIGN.md` section 8, are
corrected to the measured values above, each stating its
sample size. **The corrected 17.9x spread makes the existing
length-bucketing decision MORE justified, not less: no design decision
changes.** The lesson is procedural, not architectural: a per-session
manifest aggregate (one session's min/max/median) was read as a
dataset-wide fact without checking its scope, and an unsourced number sat
in `docs/DESIGN.md` for over a month before being traced. Future
distribution claims in these docs should name their sample size in the
line itself, which is now also done here.

---

### At the real median length, batch=64 does not fit in 8 GB; it silently spills through WDDM (2026-10-22)

**PARTIALLY SUPERSEDED by "The ceiling was data-limited, not signal-limited; Arm 3b has real headroom" (2026-11-05)'s companion fix to `docs/DESIGN.md` section 8.** Only the implication that batch=64 is unusable FOR THIS PROJECT'S MODEL is affected: the benchmark's `batch=64` spill figures below (9,527 MB, the 22.3x slowdown) were measured on the "baseline" 5L/1024H config -- the published paper's architecture, not this project's own 3L/512H one, which uses 3,411 MB at batch=64 in fp32 and does not spill (re-measured, same entry). The spill MECHANISM itself (WDDM silently spilling to system RAM with no `OutOfMemoryError`, rather than raising one) and the N=100 full-batch observation in `training/loop.py`'s docstring (26x slowdown, N=100 vs. N=10, during the real recalibration grid) are NOT superseded -- both are real and independent of which architecture this entry's own benchmark happened to measure.

**Question:** "GPU compute is not a constraint on model size" (2026-09-01)
benchmarked at `seq_len=527`, the session-1 statistic since corrected to
the true median 845 (previous entry). Its headline, "no OOM risk even at
batch=64," is quoted by `docs/DESIGN.md` section 8. Does that claim
survive at the real median, measured, not estimated?

**Method:** Reran the GPU benchmark (then in `scratch/`, now tracked as
`tools/benchmark_gru.py`; same RTX 2080 SUPER, same
8,192 MiB reported VRAM, same baseline 5L/1024H and small 3L/512H
configs, same fp32/fp16 precisions) with `SEQ_LENS` changed from
`{527, 132}` to `{845, 211}` (845/4, keeping the 4x-downsample arm
proportional). Confirmed first that the original table's `461.0 ms/step`
/ `3.4 GB` figures were measured at `batch=32` ("GPU compute is not a
constraint on model size"'s own Method line states `batch=32` for the
whole grid; its `batch=64` row is a separate fit-only check). Measured
both `batch=32` and `batch=64` at the real lengths, all other factors
unchanged.

**Result:** At `seq_len=845`, baseline (5L,1024H), fp32: `batch=32`
measured **673.70 ms/step, 5029 MB peak** (vs. the superseded entry's
461.0 ms/step / 3.4 GB at `seq_len=527`, confirming the expected
slowdown). `batch=64` reported **"fits," 9527 MB peak, at 15023.81
ms/step**: 22.3x slower than its own `batch=32` row, not the ~1.8x
`batch=32`->`batch=64` scaling every other configuration in this run
shows. **9527 MB exceeds this card's 8,192 MiB.** The reported "fits"
and the absence of a raised `torch.cuda.OutOfMemoryError` do not mean
it ran on-card: this is the same WDDM driver-level spill-to-system-RAM
path already documented in `src/b2t_drift/training/loop.py`'s
`train_one_step_accumulated` docstring (observed there: 26x slowdown,
N=100 vs. N=10, during the recalibration grid). fp16 at the same
`seq_len=845`/`batch=64` reports 5130 MB (genuinely under 8,192 MiB) at
404.05 ms/step, only 1.3x its own `batch=32` row (301.05 ms/step),
consistent with a real, non-spilling fit. The small model (3L,512H)
fits cleanly at `batch=64`/`seq_len=845` in both precisions (3412 MB
fp32, 1872 MB fp16), with proportionate step times.

| model | seq_len | precision | batch | ms/step | peak (reported) MB |
|---|---:|---|---:|---:|---:|
| baseline (5L,1024H) | 845 | fp32 | 32 | 673.70 | 5029 |
| baseline (5L,1024H) | 845 | fp32 | 64 | **15023.81** | **9527 (spilled)** |
| baseline (5L,1024H) | 845 | fp16 | 32 | 301.05 | 2826 |
| baseline (5L,1024H) | 845 | fp16 | 64 | 404.05 | 5130 |
| small (3L,512H) | 845 | fp32 | 64 | 269.24 | 3412 |
| small (3L,512H) | 845 | fp16 | 64 | 130.81 | 1872 |

**Consequence:** **`docs/DESIGN.md` section 8's "no OOM risk even at
batch=64" does not survive at the real median length, for the baseline
architecture in fp32.** It held only at the wrong, 38%-shorter `seq_len`
the original benchmark used. It does survive in fp16, and for the small
architecture in both precisions. `docs/DESIGN.md`'s wording is left
unchanged pending a decision on how to state this (flagged separately,
not resolved here). Two further echoes of the same session-1-derived
`2,179` figure were found and corrected while investigating this:
`src/b2t_drift/analysis/subspace_alignment.py`'s `CovarianceAccumulator`
docstring (described the s05 training block specifically; measured
1,735, not 2,179 or the dataset-wide 2,475) and
`src/b2t_drift/training/loop.py`'s `train_one_step_accumulated`
docstring (describes the recalibration grid's 7 evaluation sessions
specifically; measured 2,355, not 2,179). Neither of those two was a
quantitative claim anything downstream depended on; both were
illustrative figures inside comments.

---

### Model A diverges with or without day layers at 45-session scale; the day-layer mechanism is not implicated (2026-10-22)

**PARTIALLY SUPERSEDED by "Per-session Switchboard counts measured directly; s20's active-session count corrected from 19 to 17" (2026-10-28).** Every reference below to "19 active sessions" / "19 sessions" as Model B's maximum (the question, and the Interpretation and "What this does and does not call into question" paragraphs) should read 17, not 19 — s20 loses three zero-Switchboard sessions, not one. This is descriptive context inherited from the now-corrected "cut-point sweep" entry's count, not a number measured in this entry. NOT superseded: this entry's own result (both runs fail nearly identically, day-layer mechanism ruled out) does not depend on the exact session count and is unaffected.

**Question:** Model A (`docs/DESIGN.md` section 3) has two prior attempts
on record, "Model A's initial result and divergence" (2026-09-08,
diverged to 0.5009 with full day layers, no patching) and "Task 4: Model
A retrained without day layers on the official split" (2026-09-09,
reached 0.2963, day layers off, no patching). Neither used today's
settled architecture (patching on, diagonal day layers). Does the
shipping configuration pass, and if not, is the day-layer mechanism
(never run past 19 active sessions) or something else implicated?

**Pre-registered bands, fixed before either run, not revised after
seeing a result:** PASS (same scale as published) val PER <= 0.20;
EQUIVOCAL 0.20-0.25; FAIL val PER > 0.25, OR no clear improvement on the
prior 0.2963, OR train loss rising at any point during the run.
Divergence is an automatic fail independent of val PER, per the
mechanism that killed the 2026-09-08 attempt.

**Method:** Two sequential runs (never in parallel), both on
`splits/official.json` (8,072 train trials, 1,426 val trials, 45
sessions, all corpora), both `max_epochs=150`/`patience=150` (Model B's
convention, not the prior attempts' 100: both prior attempts' best
epochs, 18 and 47, sit well inside either budget, so the comparison
stands), both logging val PER every epoch.

- **Run 1, the shipping architecture**
  (`configs/experiment/switchboard_baseline.yaml`): patching (14/4),
  diagonal day layers, lambda=300, 3 layers, hidden 512. This is
  simultaneously a pipeline check and the first test of diagonal day
  layers at 45 sessions (previously validated only up to 19).
- **Run 2, the day-layer control**
  (`configs/experiment/switchboard_baseline_no_day_layers.yaml`): same
  split, same architecture, same epoch budget, patching ON, day layers
  OFF entirely: isolating the one variable Run 1 alone cannot separate,
  and filling the "patching combined with day layers off" gap the
  2026-09-09 entry named as untried.

**Result: both runs fail, nearly identically, regardless of day layers.**

| | Run 1 (diagonal) | Run 2 (no day layers) |
|---|---|---|
| Best val PER | **0.3129** @ epoch 23 | **0.3149** @ epoch 24 |
| Final val PER (epoch 149) | 0.4699 | 0.4837 |
| Train loss | falls to ~0.82 by ep17-24, then rises for ~126 consecutive epochs to 2.23 | falls to ~0.78-0.81 by ep23-24, then rises for ~125 consecutive epochs to 1.60 |
| Total wall time | 6,221.4 s (103.7 min) | 5,661.9 s (94.4 min) |
| Peak GPU memory | 2,017.2 MB | 2,015.2 MB |

Both best-epoch PERs are **worse than the prior 0.2963** (day layers off,
no patching), both are far above the published checkpoint's 0.101 and
our own inference reproduction of it (0.109), and both diverge by the
same train-loss-rising criterion that failed the 2026-09-08 attempt.
**FAIL on all three independent criteria, for both runs.**

**Interpretation.** Removing day layers entirely (Run 2) did not fix the
divergence: best PER, final PER, and the shape and magnitude of the
train-loss rise are nearly identical with or without the mechanism Run 1
was specifically testing. **This rules out the day-layer mechanism as the
cause.** The common factor across both runs, absent from every prior
successful run (Model B's cut points, s05-s20, and the two 2026-09
Model A attempts, neither of which combined patching with the full
45-session/all-corpora/8,072-trial population) is patching at this scale
and corpus composition, which has only previously been validated stable
up to 19 active, Switchboard-only sessions. Per the standing instruction
for this case, **no further investigation was conducted; this is reported
as evidence, not diagnosed.**

**What this does and does not call into question for Model B.** Model B
never trains on more than 19 active sessions or on more than 3,336
trials, and never on all-corpora data: the two cut points that failed
here (45 sessions, all corpora, 8,072 trials) sit well outside every
regime Model B's drift curve, attribution analysis, or recalibration grid
actually exercise. Nothing here threatens the drift curve. It does limit
how far the "diagonal day layers are stable, full day layers are not"
conclusion generalizes: it is now confirmed only up to 19 sessions, not
45, and the untested gap between them is larger than previously assumed.

**Wall-clock: the per-epoch model was accurate; the apparent discrepancy
was a unit mismatch, not an error in the rate.** The pre-registered
estimate (5.10 ms/trial-epoch from the s15 round-5 comparison, scaled to
8,072 trials: 41.2 s/epoch) was quoted as 69 min against a 100-epoch
budget, but the epoch budget was then set to 150 to match Model B's
convention, and the estimate was never re-scaled to match. Re-scaled
correctly: 41.2 s/epoch x 150 epochs = 6,175 s. Run 1 actual: 6,221.4 s
(0.7% over). Run 2 actual: 5,661.9 s (8% under). The per-epoch rate was
accurate to within 1%; there was no wall-clock discrepancy to explain.
Separately, a real but small (~2%) effect is worth noting: epoch 0 took
172 s against a ~37 s steady state, an excess of ~135 s, consistent with
cold page-cache across 8,072 distinct trial files spanning 45 session
directories warming up over the first ~10 epochs. This does not explain
any meaningful fraction of total wall time and is recorded only because
it is a real, reproducible effect.

**Consequence:** `docs/DESIGN.md` section 3 is updated to state Model A's
result and what it does and does not establish. No result is added to
`README.md`'s evidence-of-correctness framing, since neither run passed.
This entry corrects and extends "Model A's initial result and divergence"
(2026-09-08) and "Task 4: Model A retrained without day layers on the
official split" (2026-09-09); those entries are left unchanged.

---

### Model A's divergence is a learning-rate transfer problem, not the pipeline; the result is equivocal (2026-10-23)

**PARTIALLY SUPERSEDED by "Per-session Switchboard counts measured directly; s20's active-session count corrected from 19 to 17" (2026-10-28).** "Model B's val PER at 4-19 sessions" below should read "4-17 sessions" — s20's active-session count was 19, corrected to 17. NOT superseded: the session-count-trend extrapolation itself (0.158/0.161/0.169 at s05/s10/s15 → 0.219 predicted at 45 sessions) uses only s05, s10, and s15 (4, 9, 14 — all confirmed correct), never s20, so the predicted 0.219 and the EQUIVOCAL verdict are unaffected.

**Question:** The previous entry ruled out the day-layer mechanism.
Model A takes ~252 optimizer steps/epoch (8,072 trials / batch 32)
against ~28 for Model B at s05: ~37,800 steps over 150 epochs vs. ~4,200.
`learning_rate=0.001` was validated only in the smaller-step regime
(the same class of error already seen once in this project: a rate
measured for Arm 3b's 16.6M parameters, reused for Arm 2's 1,024, made
Arm 2 worse than no recalibration until measuring it per-arm resolved
it). Is Model A's divergence a learning-rate transfer problem?

**Configuration facts, read from the code before running anything.**
`learning_rate: 0.001` is a flat constant (`configs/experiment/
switchboard_baseline.yaml`); no scheduler or warmup is configured at all:
`scripts/02_train.py` calls `training.loop.fit` without a `scheduler`
argument, and `fit`'s default is `None`, confirmed by every epoch's
logged `lr=0.0010` staying identical across all 150 epochs of both
runs in the previous entry. `grad_clip=1.0`. `grad_norm_mean` was logged
throughout both failing runs: in Run 1, it climbs steadily from ~3.0
(epoch 14) to ~4.5+ (epoch 29) in the window before its eventual
divergence, continuing into the hundreds by epoch 130+; Run 2 shows the
same growth pattern, independent of day layers.

**Pre-registered decision rule, fixed before running:** if diagnostic B
does not diverge AND its best val PER is below 0.25, run ONE full-length
confirmation at that learning rate and report it against the original
bands (pass <=0.20, equivocal 0.20-0.25, fail >0.25), with no further
tuning. If B still diverges or lands at or above 0.25, stop and document
the failure.

**Method:** Two 30-epoch diagnostics, sequential, both on
`splits/official.json`, patched, diagonal day layers, lambda=300 (both
failing runs peaked at epoch 23-24, so 30 epochs captures the useful
signal). Diagnostic A: `learning_rate=0.001` (control). Diagnostic B:
`learning_rate=0.0001` (1/10th).

**Result, diagnostics:** Diagnostic A reproduces Run 1's trajectory
exactly through the shared epochs (best val PER **0.3090** @ epoch 28,
`grad_norm_mean` climbing 1.50->3.88 across the 30 epochs, consistent
with the slower-building instability already seen). Diagnostic B shows
**no divergence**: train loss falls monotonically every single epoch
(4.47->0.08) and `grad_norm_mean` falls monotonically too (2.67->0.89),
the opposite trend from A. Best val PER **0.2463** @ epoch 28, val PER
at epoch 29 **0.2482**. Per the decision rule: B does not diverge and
its best PER is below 0.25, so one full-length run follows.

**Result, the full confirmation run** (`learning_rate=0.0001`, 150
epochs, otherwise identical to Run 1): **no divergence anywhere across
all 150 epochs**: train loss falls monotonically from 4.47 to ~0.01,
`grad_norm_mean` falls monotonically from 2.67 to ~0.4. Best val PER
**0.2211** @ epoch 135, final (epoch 149) **0.2237**. Wall time 6,762.7 s
(112.7 min). Peak GPU memory 2,017.2 MB.

**Band: EQUIVOCAL (0.2211, between the 0.20 pass threshold and the 0.25
fail threshold).** Not reinterpreted as a pass. Both FAIL sub-conditions
are independently checked and neither applies: train loss never rose,
and 0.2211 is a clear improvement over the prior 0.2963 (not "no clear
improvement"). The result is squarely equivocal, stated as such.

**Consequence:** The divergence seen in both prior full-length runs was
a learning-rate transfer problem, not an issue with the pipeline, the
GRU/CTC core, patching, or the day-layer mechanism: `learning_rate=0.001`
was tuned in a ~4,200-step regime and does not transfer to a ~37,800-step
one, the same class of error this project has now hit twice. At the
transferred rate (0.0001), training is stable for the full 150 epochs,
but the resulting PER (0.2211) is equivocal against the pre-registered
bands, not a pass: still roughly double the published checkpoint's 0.101
and our own reproduction of it (0.109). Per the standing instruction,
this is the last round on Model A: no further learning rates, schedulers,
or architecture changes are proposed. `docs/DESIGN.md` section 3 is
updated to its final state. This entry corrects and extends "Model A
fails at 45-session scale regardless of day layers" (2026-10-22); that
entry is left unchanged.

**Post-hoc observation; the band is NOT revised.** Model B's own
validation PER scales with active session count: 0.158 at 4 sessions
(s05), 0.161 at 9 (s10), 0.169 at 14 (s15), a rate of 0.0016/session
across the 9-to-14 step. Extrapolated linearly 31 further sessions to
Model A's 45, this predicts 0.219 against the measured 0.2211, a
difference of 0.0025. **This is observed after the fact. The
pre-registered band is NOT revised and the verdict stays EQUIVOCAL.**
The agreement is a linear extrapolation over 31 sessions from a
5-session step (more than 6x the span it was measured over), so its
closeness may be partly coincidental and is not claimed as confirmation.
What it suggests: the <=0.20 pass band was anchored on Model B's val PER
at 4-19 sessions without carrying the measured session-count trend out
to 45, and the miss is better explained by a mis-set band than by a
broken pipeline. The pre-registration was kept exactly as made so that
this claim could be checked rather than assumed, not adjusted to fit it.

**Process lesson: `grad_norm_mean` is the leading indicator, and went
unread.** It was climbing steadily from the earliest epochs in both
failing full-length runs this round and last (`docs/FINDINGS.md`: "Model
A fails at 45-session scale regardless of day layers"), logged to
`metrics.csv` every epoch throughout, and was not examined until after
four full-length runs and roughly 3.5 hours of GPU time had already been
spent. A rising `grad_norm_mean` precedes train-loss divergence by many
epochs; for this failure mode specifically, it is the column to check
first, ahead of val PER or train loss. This is not a new instrumentation
gap: `src/b2t_drift/training/loop.py`'s `train_one_epoch` already
special-cases a transient all-non-finite `grad_norm_mean` (an AMP
warm-up overflow, returning `nan` rather than a misleading number) from
an earlier round. The metric has been instrumented correctly both times;
it simply was not read until the failure it would have flagged early had
already happened.

**Cross-reference, not reopened.** "The cut-point sweep: calibration-data
advantage is front-loaded and fades by ~100 days" (2026-09-11) reports
s05 "destabilizes after ~epoch 115 (train loss and gradient norms rise
sharply)," attributed there to smaller training sets overfitting. Rising
TRAIN loss is not overfitting, and that is the same signature diagnosed
in this entry as a learning-rate transfer problem: the attribution in
that entry may describe this same mechanism rather than training-set
size. s05's selected checkpoint is epoch 111, before the destabilization
begins (~epoch 115), so best-checkpoint selection means nothing in any
of Model B's reported results is affected either way, regardless of
which explanation is correct. Not re-run, not re-analyzed here: recorded
as a cross-reference only.

---

### Per-session Switchboard counts measured directly; s20's active-session count corrected from 19 to 17 (2026-10-28)

**Question:** "The cut-point sweep: calibration-data advantage is
front-loaded and fades by ~100 days" (2026-09-11) states "every cut
loses session 1 to the corpus filter," giving active Switchboard-
contributing session counts of s05=4, s10=9, s15=14, s20=19. A user
cross-check against `figures/corpus_composition_over_time.png` (which
appeared to show session 1 as ~82% Switchboard, not 0%) and a pytest
warning naming `t15.2023.11.17` as a zero-trial training session (not
session 1) conflicted with this claim. Are the per-cut active-session
counts correct, and is session 1 really the only session any cut loses?

**Method:** Measured labeled (train+val) Switchboard trial counts for
all 45 sessions directly from the HDF5 manifest (`iter_trial_ids`
joined against `load_corpus_labels`), independent of the CSV's
per-block sentence counts. Cross-checked session 1's composition two
ways: from `data/external/t15_copyTaskData_description.csv` directly,
and from the HDF5-manifest join; both must agree given the CSV's
label-vs-count rule (`docs/FINDINGS.md`: "Corpus CSV trial counts are
unreliable; the manifest is authoritative"). Recomputed each cut
point's active-session count directly from `build_temporal_split`'s own
`train_sessions` output (count of `n_trials > 0` entries), not by
reasoning about it. Separately rendered `figures/corpus_composition_
over_time.png` and inspected it directly to locate the source of the
apparent contradiction.

**Result:** Session 1 (`t15.2023.08.11`) is 100% 50-Word, 0%
Switchboard, confirmed identically by the CSV (7/7 blocks labeled
`50-Word`) and the HDF5 manifest (288/288 trials). The "session 1 is
100% 50-Word" claim itself is CORRECT. Two zero-Switchboard sessions
exist that the original entry missed: `t15.2023.11.04` (index 18,
session 19) and `t15.2023.11.17` (index 19, session 20) — both fall
inside the s20 (1-20) training block, both 0 of their labeled trials
Switchboard (95 and 125 total trials respectively, all unlabeled by any
tracked corpus). Recomputed active-session counts directly from
`build_temporal_split`: s05=4, s10=9, s15=14 (all confirmed correct —
each of these three blocks loses only session 1), **s20=17, not 19**
(loses session 1, 19, and 20). The figure discrepancy that prompted this
check was a rendering bug, not a data error: `scratch/investigate_
corpus_composition.py` draws each corpus's stacked bars at true session
dates with a hardcoded `width=3.0` days; session 1 (`08-11`) and session
2 (`08-13`) are only 2 days apart, so session 2's bar (15.4% 50-Word,
84.6% Switchboard — almost exactly what the apparent "first bar" showed)
overlapped and was drawn on top of session 1's, since Switchboard sorts
last and is drawn last. Fixed separately (see `tools/investigate_corpus_
composition.py`, moved from `scratch/` so the committed figure has a
reproducible, version-controlled source).

**Consequence:** The pytest warning naming `t15.2023.11.17` was correct
and not actually in conflict with "session 1 is 100% 50-Word" — both are
true; the entry's "every cut loses only session 1" was an unwarranted
generalization from s05/s10/s15 to s20. **What does NOT change:** the
generalization-gap argument in the superseded entry is built entirely on
s05 (4), s10 (9), and s15 (14) — all three confirmed correct here — and
s20 was already treated there as ambiguous and excluded from that
argument ("s20 neither confirms nor refutes the session-count reading").
Nothing about that argument's conclusion moves. Likewise `docs/DESIGN.md`
section 7's "at least 8 of 30 held-out sessions have zero Switchboard
trials" (the s15 held-out set) is independently confirmed correct by
this same per-session count (8 zero-Switchboard sessions among indices
15-44). `docs/DESIGN.md` section 7's table and prose are corrected to
17 in place, since DESIGN.md states current decisions rather than a
dated record.

**Addendum: two more entries carried the same inherited figure.** A
broader grep (not limited to the original "session 1" wording) found
"19 active sessions" / "19 sessions" also used, as descriptive context
rather than a measured result, in "Model A diverges with or without day
layers at 45-session scale; the day-layer mechanism is not implicated"
(2026-10-22) and "Model A's divergence is a learning-rate transfer
problem, not the pipeline; the result is equivocal" (2026-10-23). Both
are marked `PARTIALLY SUPERSEDED` in place, scoped the same way as
above: in both cases the entry's own measured result does not depend on
the session count and is unaffected, only a descriptive mention of
Model B's max active-session count is wrong (19 -> 17).

---

### Raw cross-measure correlation is confounded by a shared time trend; it disappears once days-since-cut is partialled out (2026-11-03)

**Question:** `docs/DESIGN.md` section 6 and `README.md`'s Attribution
paragraph both state the three confirmed measured changes (SNR loss,
tuning change, subspace rotation) are "inter-correlated at |ρ|=0.70-0.90,"
read as "the signature of one shared physical cause... producing three
effects at once, not sloppy measurement." Where does that figure actually
come from, and does it survive controlling for the one obvious confound:
all three measures are themselves strongly monotone in elapsed time, so
two time-trending series correlate with each other whether or not they
share a cause.

**Source of the original figure.** No FINDINGS entry computed a true
pairwise inter-measure correlation under this framing. The number was
assembled from three SEPARATE entries, each reporting one measured change's own
Spearman correlation AGAINST THE s05 PER CURVE, not against each other:
"SNR loss confirmed at the feature level..." (TX vs PER, rho=-0.750,
p=6.8e-6), "Tuning change confirmed via CTC forward-backward..." (tuning
vs PER, rho=-0.898 to -0.903, stated inline in the next entry), and
"Subspace rotation confirmed, and it tracks PER as tightly as tuning
change" (rotation vs PER, rho=0.908, p=6.3e-11). The per-session arrays
needed to compute the ACTUAL pairwise correlation existed only in a
gitignored `scratch/task_f2_summary.json` (byproduct of
`scratch/task_f_attribution_verdict.py`), never surfaced as its own
result.

**Method.** `tools/attribution_partial_correlation.py` (new, reading that
same per-session array: tx_norm, corr_all i.e. tuning-correlation-to-s05-
reference, and angle_values i.e. subspace-angle-to-s05-reference, all
matched to the s05 PER curve's 27 passing held-out sessions, days 2-597).
Computed Spearman correlation for each of the three measured-change pairs,
raw and with days-since-cut partialled out (rank-transform all series,
regress each measured change's ranks on ranked days, Pearson-correlate the
residuals — cross-checked against the standard partial-correlation
formula applied to the raw Spearman rhos; the two methods agree exactly).
Unit of observation: per held-out session (n=27 for every pair, same
index throughout — TX/SBP normalized to session 1's own value, tuning
and rotation each measured as distance to an s05-training-block-pooled
reference; different baselines, but the same 27-session index for all
three, so this is not the per-session-PAIR vs. per-session mismatch that
affects some of this project's other correlation measurements).

**Result: raw correlations reproduce the quoted range exactly.**

| Pair | Raw Spearman ρ | p | 95% CI | Partial ρ (controlling for days) | p | 95% CI |
|---|---|---|---|---|---|---|
| SNR (TX) vs tuning | +0.739 | 1.1e-5 | [0.500, 0.874] | +0.110 | 0.586 | [-0.290, 0.477] |
| SNR (TX) vs rotation | -0.699 | 5.0e-5 | [-0.853, -0.434] | -0.070 | 0.728 | [-0.445, 0.326] |
| Tuning vs rotation | -0.903 | 1.2e-10 | [-0.955, -0.796] | -0.322 | 0.102 | [-0.630, 0.075] |

n=27 for every cell. **All three partial correlations collapse toward
zero and lose significance** (p=0.586, 0.728, 0.102): two of three have
|partial ρ| < 0.3 outright, and the third (tuning vs rotation, 0.322) is
not significant either, independently satisfying the same "no residual
association" reading. None of the three partial 95% CIs excludes zero.

**Consequence: this is a BRANCH B result, not Branch A.** The raw
|ρ|=0.70-0.90 range is real and reproducible, but it is exactly what
three measured changes that are each strongly monotone in elapsed time would
produce whether or not they share a cause — it establishes that the
three measured changes are collinear and cannot be ranked against each other,
which is the claim this design can actually support; it is weak evidence
for "one shared physical cause," which the raw number alone cannot
distinguish from three concurrent, independently-time-trending processes.
`README.md`'s Attribution paragraph and `docs/DESIGN.md` section 6 are
both corrected to this weaker, now partial-correlation-informed reading;
neither previously stated only "cannot be ranked" (both asserted the
shared-cause reading as the signature the data showed), so both needed
the correction, not just a confirmation.

---

### Late-period ceiling measured directly: the signal itself, not the recalibration method, limits the +597d result (2026-11-04)

**PARTIALLY SUPERSEDED by "The ceiling was data-limited, not signal-limited; Arm 3b has real headroom" (2026-11-05).** Only this entry's "Consequence" paragraph's claim that 0.6901 is a ceiling Arm 3b's +597d result should be judged against, and the ~73% implied recovery fraction built on treating it as one, are affected. The measured point estimate (0.6901), its bootstrap CI, the pool composition, and the split/training method are NOT superseded -- they are a real, reproducible number, just not a ceiling on what the late-period signal supports. The first addendum below (the Model A data-scarcity check, MIDDLE case) and caveat (b) of the second addendum (architecture/data-quantity-specific, not an information-theoretic bound) already anticipated this; see the new entry for the resolution.

**Question:** Recovery-fraction's reference (`docs/DESIGN.md` section 11)
is s05's training-block validation PER (0.15765) -- an optimistic
ceiling, since it is measured with no elapsed time at all between
calibration and evaluation. How well can ANY decoder do on a LATE
session's signal, with drift removed from the problem entirely, by
training and evaluating on the same late sessions? This is the real
upper bound Arm 3b's late-session results should be judged against.

**Pool.** The six Switchboard sessions past +500d from the s05 cut,
their labeled (train+val partition) Switchboard trials only:

| Session | Day offset | Labeled Switchboard trials |
|---|---:|---:|
| `t15.2025.01.10` | +504 | 129 |
| `t15.2025.01.12` | +506 | 210 |
| `t15.2025.03.14` | +567 | 83 |
| `t15.2025.03.16` | +569 | 125 |
| `t15.2025.03.30` | +583 | 195 |
| `t15.2025.04.13` | +597 | 94 |

836 trials pooled, 92% of s05's 908-trial training block -- a fair-size
comparison, no window-widening needed.

**Method.** Per-session 90/10 split (`val_fraction=0.1`, seed 0, same
convention as `build_temporal_split`), then the recalibration grid's
stricter rule on top: any val trial whose sentence text also appears
anywhere in train is dropped (`b2t_drift.evaluation.calibration.
normalize_sentence`) -- 83 val trials pre-dedup, 31 dropped to sentence
overlap, 52 remaining. 753 train trials. Train and val both come from
the same six late sessions, so there is no elapsed-time gap between
them: this isolates signal quality from drift by construction. Manifest:
`splits/late_period_ceiling_switchboard.json` (`kind: "official"`, built
by `scratch/build_late_period_ceiling_split.py`). Architecture
config-identical to s05's run (`configs/experiment/
switchboard_baseline.yaml`: 3 layers, hidden 512, patch 14/4, diagonal
day layers, λ=300, lr=1e-3, batch 32, amp on, max_epochs/patience 150,
seed 0) -- the only difference from s05's run is which six sessions are
in the training block. Trained via `scripts/02_train.py`, evaluated via
`scratch/evaluate_late_period_ceiling.py` (bootstrap CI on the pooled
post-dedup val set, `bootstrap_error_rate_ci`, 10,000 resamples, seed 0).

**PRE-REGISTERED, fixed before training:** late-period val PER <= 0.208
(within 0.05 of s05's 0.158) means the signal is largely intact and
recalibration METHODS are the bottleneck; PER > 0.30 means the signal
itself has degraded and Arm 3b should be judged against this ceiling
rather than against 0.158; 0.208–0.30 is partial. Bands evaluated
against the bootstrap CI, not the point estimate alone; a CI straddling
either boundary is INCONCLUSIVE at this sample size, not assignable to a
band.

**Result: late-period val PER = 0.6901, 95% CI [0.6757, 0.7040] (n=52,
10,000 resamples) -- SIGNAL DEGRADED, not inconclusive.** The CI sits
entirely above 0.30, with no overlap with either boundary. Best epoch
31 of 150 (full budget ran; patience=150 never triggers early stopping
by construction). No divergence: train loss fell from 7.305 to 0.632
over 150 epochs with the longest consecutive rising streak being 2
epochs (ordinary noise, not the Model A signature of a sustained,
many-epoch rise) -- the run was allowed to finish in full per the
pre-registered protocol regardless. Wall-clock 680.1s (~11.3 min),
well under the ~39 min fp32-benchmark-based estimate because this run
used `amp=True` (fp16, the shipped default) and the fp32 benchmark the
estimate was built from did not account for that. Peak GPU memory
1,346.4 MB (`torch.cuda.max_memory_allocated`), also far under the
~5,029 MB fp32 estimate for the same reason. Max GPU temperature during
the run: 69°C (idle baseline ~45°C).

**Consequence: the signal itself, not the recalibration method, is the
binding constraint at late sessions -- Arm 3b's +597d result should be
read against this ceiling, not against 0.158.** At `t15.2025.04.13`
(+597d, N=50): arm0_per=0.7960, arm3b_per=0.7182
(`results/recalibration_recovery_fraction.csv`). Against the old
denominator (reference_per=0.15765): drift_gap=0.6383,
recovery_fraction=12.2%. Against the new, measured ceiling (0.6901):
drift_gap=0.1059, recovered (unchanged, =arm0_per−arm3b_per)=0.0777,
implied recovery_fraction≈73.4% -- a 6x difference driven entirely by
which denominator is used. Arm 3b's point estimate (0.7182) sits only
0.014 above the ceiling's CI upper bound (0.7040), i.e. close to the
measured ceiling, not meaningfully below it: full fine-tuning is nearly
exhausting what this late session's degraded signal supports. This is a
measured implication, not a restatement -- the published recovery
fractions in `results/recalibration_recovery_fraction.csv`, README, and
`docs/DESIGN.md` section 5 are NOT recomputed against this new ceiling
in this entry; whether to restate them against it is a separate
decision.

**Addendum (2026-11-04): ruling out data scarcity as the explanation,
via Model A's checkpoint.** The ceiling trained on 752 trials; a live
alternative explanation is that a noisier signal just needs more data,
so 0.6901 could reflect data scarcity rather than genuine late-period
signal degradation. Model A (`runs/modelA_run3_lr1e4_full/
checkpoint_best.pt`, `docs/DESIGN.md` section 3, val PER 0.2211) trains
on each session's own train partition and scores on that session's own
val partition -- the SAME within-session arrangement the ceiling run
used, at 8,072 trials across all 45 sessions and all 6 corpora (9.6x the
ceiling's 836). No training was needed: just re-scoring its existing
checkpoint on `splits/official.json`'s val partition, broken down
(`scratch/modelA_breakdown.py`). Overall val PER reproduced exactly
(0.2211, n=1,426), confirming the methodology. By corpus: Freq words
0.1682 (n=225), Harvard 0.2433 (n=45, small-N), Openwebtext 0.2267
(n=80), Random 0.3253 (n=39, small-N), Switchboard 0.2270 (n=1,037);
50-Word has no official-split val trials. Early sessions (all corpora):
0.2023 (n=1,253). The six late-pool sessions (all corpora, which turned
out to be 100% Switchboard in the official val partition): **0.3564,
bootstrap 95% CI [0.3347, 0.3788] (n=173)** -- the cell that matters,
not small-N. **This is the MIDDLE case, not either boundary**: the
pre-registered bands were >=0.55 (data scarcity ruled out, ceiling
corroborated) or <=0.35 (ceiling was data-limited); 0.3564 sits just
above 0.35, and its own CI lower bound (0.3347) sits just below it. With
9.6x the data, Model A still cannot reach late-session Switchboard PER
anywhere near its own 0.2211 average or near early sessions' 0.2023 --
more data clearly helps (0.3564 is well below the ceiling's 0.6901) but
does not fully rescue late-session performance, and the result is too
close to the low-case boundary to call cleanly. Per the pre-registered
protocol: both readings are reported, item 4 (README) is skipped, and
this round stops here on the data-scarcity question -- it is not
resolved in either direction. Note: Model A is itself equivocal against
its own pre-registered band (`docs/DESIGN.md` section 3), so it is a
conservative probe for this purpose, not a clean one.

**Addendum (2026-11-04): what the bootstrap CI resamples.** The ceiling
CI ([0.6757, 0.7040], width 0.028 at n=52) resamples whole TRIALS, not
tokens or phonemes: `bootstrap_error_rate_ci` (`evaluation/metrics.py`)
draws `n` trials with replacement per resample and recomputes
`aggregate_error_rate` -- pooled edit distance over pooled reference
length, a ratio estimator, not a mean of per-trial PERs -- on each
resample (confirmed by reading the function and its docstring, not
assumed). This is the same function `results/drift_curve.csv`'s
per-session CIs use. A ratio-of-sums estimator legitimately has lower
variance than a mean-of-per-trial-PERs one, so the CI's width is not by
itself evidence of an error. No recomputation needed, and the finding is
unaffected either way: the conclusion only required the CI to clear
0.30, and it does by a wide margin regardless of which of the two
reasonable resampling conventions is used.

**Addendum (2026-11-04): two caveats.** (a) The late pool spans 93 days
(+504 to +597); s05's training block spans about two weeks. The pooled
model fits data that is itself internally drifting, which inflates its
validation PER for reasons unrelated to signal quality at any single
day -- **0.6901 is a CONSERVATIVE ceiling**: a true single-day ceiling
would likely be lower (better), and Arm 3b's margin wider than the
measured 0.0281 (0.7182 − 0.6901). Magnitude, estimated (not measured
directly on the ceiling model -- its 52 val trials split six ways are
too thin for a reliable per-session slope): s15's measured late-segment
(>150d) plateau slope is **+0.021 PER/100d** (this entry's own
pre-existing text above); over a 93-day span that predicts roughly
**+0.02 PER** of within-pool drift inflation, the same order of
magnitude as Arm 3b's margin itself. This is a proxy from a different
model's plateau, not a direct measurement of the ceiling run -- stated
as a direction and an order-of-magnitude, not a precise correction. (b)
0.6901 is the ceiling for THIS architecture (3 layers, hidden 512,
patch 14/4, diagonal day layers) at THIS data quantity (752 training
trials), not an information-theoretic bound on what the late-period
signal contains -- a larger model or more late-period data might do
better.

---

### The ceiling was data-limited, not signal-limited; Arm 3b has real headroom (2026-11-05)

**Question:** The previous entry's data-scarcity check (Model A's
late-pool Switchboard cell, PER=0.3564) landed in the pre-registered
MIDDLE band and was left unresolved. Two questions remained open: does
more training data genuinely rescue late-session Switchboard decoding,
and separately, was the 836-trial ceiling run's 0.6901 actually a
ceiling on the late-period signal, or an artifact of its own small
training set?

**Method:** Re-scored Model A's checkpoint once more for the matched
comparison the previous entry's cells (b)/(c) were missing: EARLY
sessions, Switchboard trials only -- same checkpoint, same
within-session train/val protocol, same corpus filter as the late cell,
differing only in recording period. Bootstrap CI computed identically
to every other cell (`scratch/modelA_breakdown.py`).

**Result: Model A's early-session Switchboard PER is 0.1993, 95% CI
[0.1900, 0.2086] (n=864).** Matched comparison, model/data
quantity/corpus/protocol all held constant, period the only variable:
early 0.1993 (n=864) vs. late 0.3564 (n=173, previous entry's cell (c))
-- absolute difference **+0.1571**, relative increase **+78.8%**. This
is clean, single-model evidence that late-period signal genuinely
degrades relative to early sessions.

That is a different question from whether 0.6901 is a ceiling, though.
Comparing Model A's late-Switchboard cell (0.3564, 8,072 training
trials across all 45 sessions) against the dedicated ceiling run
(0.6901, 752 training trials from only the six late sessions): same
late sessions, same corpus, 10.7x the training data, and PER falls by
nearly half.

**Consequence, five points:**

**(a) The 836-trial ceiling run was substantially DATA-LIMITED.** Model
A reaches 0.3564 on the same late Switchboard material with 8,072
training trials, against the ceiling run's 0.6901 with 752. 0.6901 is
therefore NOT a ceiling on what the late signal supports -- it is a
small-data artifact of that specific run.

**(b) Arm 3b's +597d result of 0.718 is NOT near-optimal.** It sits
well above 0.3564 (the best late-Switchboard PER actually demonstrated,
by any model, on this data), so meaningful headroom exists and the
limiting factor is not the signal alone.

**(c) Methodology note: the pre-registered bands were poorly designed.**
They anchored on whether the Model A cell was high or low in absolute
terms (>=0.55 vs. <=0.35), when the informative comparison was always
that cell against 0.6901 directly -- a relative test, not an absolute
one. An absolute band cannot distinguish "the late signal is this bad"
from "this checkpoint under-extracts the late signal," which is exactly
the ambiguity that produced the inconclusive MIDDLE case. Recorded here
as a mis-set-band note, the same way `docs/DESIGN.md` section 3 keeps
Model A's own mis-set equivocal band on record rather than erasing it.

**(d) What this experiment DID establish:** signal degradation measured
within a single model on matched material -- Model A's own Switchboard
PER rises from 0.1993 (early) to 0.3564 (late), +78.8% relative, with
data quantity, corpus, model, and evaluation protocol all held
constant. This is real and not superseded by (a)/(b) above.

**(e) What remains UNMEASURED: the ceiling that matters for Question
3** -- the best achievable PER from a FROZEN decoder plus N new-day
labeled sentences, which is what every recalibration arm actually has
available. Model A's 0.3564 required training on the late sessions
with 8,072 trials, which no recalibration protocol can do (Arm 3b's
richest cell uses N=100, a few dozen minutes of speech). Neither 0.6901
(the data-limited 752-trial ceiling) nor 0.3564 (Model A's
8,072-trial, all-session result) should be substituted for this
unmeasured number; both are off this experiment's achievable axis in
opposite directions.

---

### Recovery fraction's "two sharp steps" withdrawn: three measured levels, two transitions of unresolved sharpness (2026-11-05)

**Question:** `docs/DESIGN.md` section 5 and two entries above ("Recalibration
slows the decline but does not restore function", 2026-09-28; "Task 1's
gap-fill...", 2026-09-28) describe the N=25/N=50 recovery-fraction curve as
declining in "two sharp steps" rather than smoothly. Does the data support
an abrupt transition, or only three measured levels with two untested gaps
between them?

**What is actually measured**, re-verified directly against
`results/recalibration_grid.csv` via `scratch/recovery_analysis.py` (the
module `scratch/recovery_figure.py` itself imports, so this is the same
data the figure reads, not a re-derivation):

| Elapsed time | N=50 recovery fraction |
|---|---|
| +9d | 31.2% |
| +42d | 35.3% |
| +58d | 42.6% (29 eval trials, flagged unreliable, previous entries) |
| +184d | 19.5% |
| +259d | 19.9% |
| +329d | 9.3% |
| +597d | 12.2% |

Three groups, not two points joined by two drops: an early group of three
rising points (+9/+42/+58), a middle group of two nearly-equal points 75
days apart (+184/+259), and a late group of two points (+329/+597). No
session was ever evaluated in the 126 days between +58d and +184d, or the
70 days between +259d and +329d.

**Method:** To test whether the two transitions look like discrete steps or
are simply unmeasured, fit a straight line between the two endpoints
bracketing each gap and compare its prediction at +184d/+259d against what
the gap-fill sessions actually measured.

**Result, part 1 -- the step claim is unsupported.** A straight line from
the +58d value (42.6%) to the +329d value (9.3%) predicts **27.1% at +184d
and 17.9% at +259d**. The measured values are **19.5% and 19.9%** -- 7.6
points below the line at +184d and 2.0 points above it at +259d (N=25 shows
the same pattern: predicted 18.0%/11.8% vs. measured 12.8%/12.8%). The
gap-fill entry above read "roughly HALFWAY between" the outer levels as
evidence FOR a discrete step. **That is backwards: landing between the two
endpoints is exactly what a smooth, linear decline predicts across an
unmeasured interval -- it is not distinguishing evidence either way.** With
only two points inside what was a 271-day gap, the shape of either
transition cannot be recovered: a sharp drop and a gradual one through the
same window produce the same two endpoint readings.

**Result, part 2 -- what the data does support.** +184d and +259d are
nearly equal to EACH OTHER (19.5% vs. 19.9%, N=50; 12.8% vs. 12.8%, N=25)
despite being 75 days apart. Continuing the same linear decline used above
predicts a drop of **9.2 points (N=50) / 6.2 points (N=25)** between them;
the observed change is **+0.4 / 0.0 points**. That IS real evidence for a
distinct, comparatively flat middle level sitting between the early and
late groups -- just not evidence about how sharply either boundary
transition happens. Right conclusion (three levels, not one smooth
single-rate decline), wrong argument (the earlier entry's "halfway"
reasoning, which actually argues the opposite of what it was read as
showing).

**Consequence:** "Two sharp steps" is withdrawn. What replaces it: recovery
fraction sits at three measured levels -- roughly 31-42% through +58d,
~19.5-19.9% at +184d-+259d, ~9.3-12.2% from +329d on -- and the two
transitions between them, spanning 126 and 70 unmeasured days respectively,
could be abrupt or gradual; this design cannot tell which. The -15.5-point
(N=25) / -23.1-point (N=50) difference between the +58d and +184d readings,
reported in "Recalibration slows the decline but does not restore
function" above, is unaffected: it is the gap between two measured values
and stands regardless of the shape in between. Only the word "step"
attached to it, and the "roughly halfway" argument in the gap-fill entry,
are withdrawn.

`README.md`'s own "What I'd do next" section already asked, before this
retraction, whether the drop was "itself sudden or just under-resolved" --
the repo contained this contradiction before today: one passage asserted a
shape, another already flagged that the sampling couldn't establish it.
That bullet is reworded to drop "steps down sharply" while keeping its
point. `docs/DESIGN.md` sections 5 and 11 are updated accordingly;
`figures/recalibration_recovery_fraction.png` is retitled, and its shaded
bands now mark the two unmeasured intervals instead of the apparent drops.
