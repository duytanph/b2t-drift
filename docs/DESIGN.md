# Design decisions — the durable record of WHY

This file records design decisions and their justification. `docs/DATASET.md`
records WHAT the data is; `docs/FINDINGS.md` records the dated empirical
investigations behind these decisions; this file records why we chose to
work with the data the way we do, as it stands now. When a decision
changes, update this file and say why the old reasoning no longer holds —
don't just delete it. Target length: under 400 lines. No chronology here —
that belongs in `docs/FINDINGS.md`.

## Summary

Speech BCI decoders lose accuracy across days; this project measures that
loss, attributes it to specific causes, and compares recalibration
strategies under a fixed budget of participant time.

- **Measurement** — how fast does PER degrade with days since calibration?
- **Attribution** — why: channel death, tuning change, or subspace
  rotation? (Marginal drift is already removed upstream — section 6.)
- **Decision** — given a fixed budget of new-day participant time, what's
  the best use of it?

**Split:** train on an early block of sessions, swept at four cut points
(1–5/1–10/1–15/1–20); evaluate each later session individually as a
PER-vs-days-elapsed curve. Temporal only, never shuffled. Switchboard-only
is the primary analysis.

**Status:** data verified, pipeline built (a batch-composition bug found
and fixed), hyperparameters set, dataloader bottleneck resolved — no
drift-curve model has been trained yet.

---

## 1. Thesis

Speech BCI decoders lose accuracy across days; this repo quantifies the
loss, attributes it to specific changes in the recorded features, and
compares recalibration strategies under a fixed budget of user time (see
Summary above for the one-paragraph version).

---

## 2. Terminology

- **"Our split"**: the temporal train/held-out boundary this project
  defines, cutting the 45 sessions by date into a training block (starting
  at session 1, `t15.2023.08.11`) and everything after it. Base case:
  sessions 1–15 train / 16–45 held out, but section 7 sweeps this cut
  point across four values (1–5, 1–10, 1–15, 1–20) — "our split" means
  this general design, not one fixed pair of session counts. Entirely our
  own construction; not shipped as files.
- **"Official split"**: the `data_train.hdf5` / `data_val.hdf5` /
  `data_test.hdf5` files shipped for the Brain-to-Text '25 competition.
  Relevant here only because (1) its `test` partitions are unlabeled
  (1,450 trials, 13% of the dataset, unusable regardless of split), and
  (2) the provided pretrained checkpoint was trained on this split's
  `train`/`val` data spanning **all 45 sessions** (`docs/FINDINGS.md`:
  "Block count decreases..."), so it has no held-out future and cannot
  measure drift.
- **"Held-out"** always means *withheld from our model* (not in our
  split's training block) unless stated otherwise — never the official
  split's `val`/`test` files, which are a different withholding and (per
  section 7) not independent of each other or useful for drift.

---

## 3. Two models

This project trains two distinct models. Conflating their numbers is a
common, easy mistake — keep them separate in every results table.

- **Model A — official-split pipeline check.** Trained on the official
  split's `train` partition (all 45 sessions). Confirms our data loading,
  model, and training code produce a PER on the same scale as published
  results — a correctness check, not a contribution. Run once; not part of
  the drift experiment; not comparable to Model B (different training
  data, different scale of task).
- **Model B — our-split drift model.** Trained on **our split** (section
  2) — the temporal cut-point sweep. Its PER curve across held-out
  sessions **is the project**.
- **The paper's published numbers (`docs/DATASET.md` section 2) are not a
  drift comparison, for either model.** The published system was
  continuously, periodically recalibrated; Model B is calibrated once and
  frozen. "Maintained performance under active maintenance" and
  "performance decay under no maintenance" are different quantities — a
  gap between our curve and the paper's flat one is not evidence of a
  worse decoder; it's the point of the study.

---

## 4. Three research questions

- **MEASUREMENT** — How fast does PER degrade as a function of days since
  calibration?
- **ATTRIBUTION** — Why does it degrade? See section 6 for the current set
  of candidate causes (revised once from four to three, after the
  normalization mechanism was confirmed by author correspondence).
- **DECISION** — Given a fixed budget of participant time on a new day,
  what is the best thing to spend it on? See section 5 for the candidate
  interventions ("arms").

---

## 5. Recalibration strategies (Question 3 arms)

Candidate interventions for the DECISION question, increasing order of
participant-time cost. Each targets a different subset of section 6's
attribution causes.

**A fourth candidate arm — "recompute per-session z-score statistics" —
is retired.** The released data already does something strictly better: a
continuous rolling per-channel z-score over the previous up-to-20
sentences, recomputed sentence by sentence (`docs/DATASET.md` section 9
item 7). Recomputing one statistic per session would be *coarser*, and
isn't a participant-time cost at all — it happens for free, as a property
of the release. Removed rather than kept as a strawman.

| Arm | Labels needed | Targets | Notes |
|---|---|---|---|
| **0 — no adaptation (baseline)** | None | — | Day-0 calibrated decoder, unmodified. Zero participant time. This is what the drift curve itself measures. |
| **1 — unsupervised distribution alignment** (e.g. CORAL, Procrustes) | None (unlabeled new-day data only) | Subspace rotation | If the population-activity manifold has rotated in 256-d space but per-channel marginals are already handled by the rolling z-score, a linear alignment from unlabeled data alone can in principle undo the rotation. |
| **2 — new day-specific input layer, rest of network frozen** | N labeled sentences | Tuning change + subspace rotation | Structurally identical to the `day_weights`/`day_biases` layers found in the official checkpoint (`docs/FINDINGS.md`). More degrees of freedom than Arm 1; can partially compensate for a shifted phoneme-to-channel mapping. |
| **3 — fine-tune the whole network** | N labeled sentences | Everything the other arms target, plus any deeper change | Highest participant-time and compute cost. **Catastrophic-forgetting risk at small N**: too little new-day data and whole-network fine-tuning can overfit to the N calibration sentences, losing generalization the frozen-backbone arms would have kept. |

**The day-layer gap (0.313) exceeds 18 months of observed drift (0.205)
— recalibration is the DOMINANT effect here, not a secondary one.**
Measured directly (`docs/FINDINGS.md`: "The identity-day-layer redirect
control"): redirecting a training session's day-layer index to an
untrained (identity) slice costs +0.313 PER on average, while the
entire observed drift from +5d to +546d is +0.205 PER
(`docs/FINDINGS.md`: "The full 22-session drift curve"). A single day's
Arm 2 recalibration therefore has more headroom to recover than 18
months of drift has room to lose — this reframes Question 3 from "how
much does drift cost, and can recalibration offset some of it" to
"recalibration recovers more than drift ever takes away, so the DECISION
question is really about cost-efficiency (how little labeled data
achieves that recovery), not whether recalibration helps at all."

**Arm 2 uses the `"diagonal"` parameterization; `"full"` is confirmed
unworkable when trained jointly.** Model A trained all 45 `"full"`
day-layer slices (`docs/DATASET.md`; 11.8M of 16.6M total parameters at
this scale) simultaneously, each against hundreds of training trials,
with no anchor toward identity — and diverged (`docs/FINDINGS.md`:
"Model A's initial result and divergence", "Day layers are unstable when
trained jointly across many sessions"). The round-5 s15 comparison
confirmed this is a **parameterization** failure, not a mechanism
failure or an anchor-strength failure: `"full"` diverged again even with
the identity-anchored penalty at `λ=3000` (train loss rose 1.4 → 3.5
over training), while `"diagonal"` (1,024 params/session — per-channel
gain and offset only) trained stably to full PER parity with a
no-day-layer model, its slices never leaving `max|W−I| ≈ 0.04` of
identity. `"diagonal"` is therefore the parameterization Arm 2 ships;
`"full"` is retained only as an UPPER-BOUND comparison (PER recovery
available with unlimited per-session capacity), and `"lowrank"`
(16,896 params/session) is implemented but untested. That the lean
parameterization is both the stable one AND the one fittable from 10–50
labelled sentences is a direct de-risking of Arm 2 at small N.

**Day layers are KEPT in the primary architecture (decided by a
pre-registered rule).** Round-4's five-way s15 comparison appeared to
show a no-day-layer model beating every day-layer variant, which would
have undercut Arm 2's premise entirely. Round 5 re-ran the comparison
properly — at 150 epochs (round-4's no-day-layer run had not converged,
its best epoch was 98 of 100) and including the previously-missing
patching + `"diagonal"` day-layer combination — under a keep/drop rule
fixed and committed *before* the results were read (`docs/FINDINGS.md`:
"Pre-committed rule for keeping or dropping day layers"). Outcome: the
best day-layer config (patched, `"diagonal"`, λ300) reached val PER
**0.16932** vs. **0.1697** for the best no-day-layer config — a
**−0.0004** gap, i.e. the day-layer model was marginally *better*, and
the gap is ~1/75th of the rule's 0.03 threshold. The day-layer
mechanism costs nothing once parameterized correctly, so it stays, and
Question 3's Arm 2 and the +0.313 day-layer-gap finding both stand.

---

## 6. What a rising PER curve means

The release pipeline applies **three independent adaptive corrections, at
three different timescales**, before any data reaches our model. This
determines what the ATTRIBUTION question (section 4) can and cannot
conclude, and motivates the arms in section 5.

| # | Correction | Timescale | Removes |
|---|---|---|---|
| 1 | RMS spike-detection threshold (`docs/DATASET.md` section 9 item 3) | ~50 sentences (recomputed from the preceding block) | Slow drift in a channel's noise floor |
| 2 | Rolling per-channel z-score (`docs/DATASET.md` section 9 item 7) | 20 sentences (rolling) | Per-channel gain/offset (affine) changes |
| 3 | Day-specific input layer (`docs/FINDINGS.md`: "Block count decreases...") | Trained per session | Whatever a learned linear map can capture — starts at identity, contributes nothing until trained |

- **Stage 1** tracks noise, not signal: as an electrode becomes
  encapsulated, spike amplitude shrinks while the noise floor holds
  steady, so SNR falls and crossing rate genuinely declines — the
  threshold has no mechanism to compensate. **Caveat:** a channel can be
  actively dying while its crossing rate *appears* stable if the noise
  floor happens to move the same direction. **Channel-mortality analysis
  must use recovered raw spike counts (Hz, via `docs/DATASET.md` section 9
  item 9) and the +10 clip-saturation signature (item 8), not crossing
  rate alone.**
- **Stage 2** removes per-channel affine changes only. It does **not**
  remove: which channels respond to which phonemes, inter-channel
  correlation structure (scale-invariant), the geometry of population
  activity (a 256-d rotation isn't undone by 256 independent rescalings),
  or temporal structure within a sentence.
- **Stage 3** is our own intervention (Arm 2, section 5), not something
  upstream — do not confuse it with stages 1–2.

**Attribution reduces from four candidates to three:**
- **Channel death** — an electrode stops producing usable signal.
  Non-obvious interaction: z-scoring may *amplify* channel death rather
  than hide it (dividing a near-zero-variance channel by its own near-zero
  std inflates residual noise back to unit variance; the +10 clip then
  caps it). Check empirically before relying on this as an explanation.
- **Tuning change** — phoneme-to-channel mapping shifts (e.g. tissue/
  electrode geometry changes). Not fixable by rescaling. Targeted by Arm 2.
- **Subspace rotation** — the population-activity manifold's orientation
  changes. Correctable in principle by a linear map (Arm 1, and within
  reach of Arm 2).
- **Marginal drift (gain/offset, noise-floor drift) is removed from view
  by stages 1–2 and cannot be studied with this release** — it is no
  longer a fourth candidate.

**What this buys the measurement.** Any degradation the drift curve shows
has already survived three independent, causal (past/current data only)
adaptive corrections. It therefore cannot be attributed to scaling, gain,
offset, or noise-floor drift — whatever is left is SNR loss, tuning
change, or subspace-geometry change. The curve is meaningful precisely
because the easy explanations are already ruled out by the data itself.

---

## 7. Split design and rationale

**Cut-point sweep, not one fixed cut.** Trained separately at sessions
1–5, 1–10, 1–15, and 1–20, producing four drift curves. Two reasons: (1)
**robustness** — if the curve's *shape* is stable across cuts, the result
doesn't hinge on an arbitrary choice of "15"; if it isn't, that's itself a
finding about sensitivity to calibration-set size; (2) **a question of its
own** — PER at a fixed day-offset across the four cuts measures how much
calibration data buys how much drift-resistance. **Tradeoff:** more
training sessions → stronger baseline but a shorter, more recent held-out
time axis (training through session 15 leaves +5 to +546 days; through
session 30 would cut that to roughly +5 to +166, discarding the six 2025
sessions). This is why the sweep stops at 1–20.

- **Why temporal, not random.** The question is extrapolation into the
  future, not interpolation within a shuffled pool — a randomly-held-out
  session tells you nothing about a decoder that's aging.
- **Bootstrap CIs are mandatory.** Early held-out sessions have as few as
  80 trials; PER on 80 sentences is noisy enough to look like structure by
  chance. Every point on every curve needs an interval.
- **Floor effect risk.** The 1–15 training block is 3,969 labeled trials
  (~17.6 hours of speech) — comparable to Willett et al. 2023's 16.8-hour
  budget (23.8% WER, 125K-word vocabulary), so a usable decoder is
  plausible. The 1–20 cut (more training data) serves as a check on
  whether floor effects distort the 1–5/1–10 curves.
- **Validation set rule.** Early-stopping/hyperparameter data must come
  from WITHIN the training sessions, never from held-out sessions —
  peeking at the drift curve to tune hyperparameters would fit the model
  to the test data by the back door. Examined once, at the end, per
  cut-point model. Session 16 is a valid validation source for the 1–15
  cut but must never be used for the 1–5 cut's evaluation of session 16.

**s15's 126-day coverage gap (days +7 to +133) sits exactly across the
region where effective signal collapses — a consequence of the sweep
design, not a flaw to engineer around.** The gap itself
(`docs/FINDINGS.md`: "Cut-point sweep manifests built and validated; a
126-day coverage gap") means s15 has no eval point between +7d and
+133d — exactly the window where `effective_signal` falls from ~0.15
(day +5/+7) toward the ~0.02–0.03 range it holds from +133 onward
(`docs/FINDINGS.md`: "The effective-signal control"). s15 alone can
therefore only establish that the collapse has ALREADY mostly happened
by +133d, not resolve where within that gap it actually occurs. **s05**
(27 passing sessions, day-offset range starting at +2 — `docs/FINDINGS.md`'s
cut-point table) is the cut point better positioned to characterize the
collapse itself, once trained, since its early eval points sit inside
the exact window s15 skips. This is precisely why the sweep runs at
multiple cut points (this section's opening rationale) rather than
committing to s15 alone.

**Corpus stratification.** Switchboard-only, both train and eval, is the
**primary analysis** (`splits/temporal_s01_s*_switchboard.json`);
all-corpora is a named fallback (`temporal_s01_s15_allcorpora.json`,
1–15 only) if Switchboard-only proves too small. Pooling all corpora
would conflate task-distribution shift with neural drift — the training
block is ~87–88% Switchboard while at least 8 of 30 held-out sessions
have **zero** Switchboard trials (`docs/FINDINGS.md`: "Corpus composition
is a serious confound"). Cost: 3,501 Switchboard vs. 3,969 all-corpora
trials at the 1–15 cut, **11.8% less training data**. The Switchboard
1–15 block is effectively 14 sessions: session 1 is 100% 50-Word,
contributing zero Switchboard trials (recorded per session in every
manifest, cut DATE unaffected). The ≥50-labeled-Switchboard-trial
viability threshold (`passes_threshold` in every manifest) was fixed
before seeing any PER value.

**Primary cut-point selection follows a rule fixed in advance of any PER
value** — see `docs/FINDINGS.md`: "Pre-committed cut-point selection
rule" for the exact rule and why recording it in advance matters.

**Why the official split cannot measure drift, and why the pretrained
checkpoint cannot be used for the drift curve:** see `docs/DATASET.md`
sections 12–13 for the full mechanism (val/test are trial-level splits of
the same held-out blocks, not independent; every session, including the
most recent, contributes training data, so the official split shows
near-zero drift by construction). The pretrained checkpoint was trained
on all 45 sessions (`docs/FINDINGS.md`: "Block count decreases..."), so
it has no held-out future — useful only for validating our evaluation
code and for inspecting its learned per-session day-layers.

---

## 8. Pipeline constraints derived from data inspection

- **Lazy HDF5 loading.** The dataset is ~12 GB against a 16 GB system RAM
  budget — the Dataset class reads lazily per `__getitem__`, never holds
  everything in memory.
- **Length-bucketed batching** (`LengthBucketedSampler`,
  `data/collate.py`). `n_time_steps` spans 138–2179+ bins (~7.4x within
  one session, more across the dataset); random batching wastes most of
  each batch on padding. Sorting before chunking cuts padding waste from
  35.5% (random) to 13.7% (bucketed) at a 32-trial pool, batch_size 8.
  Only batch ORDER is shuffled per epoch (with jitter so the same trials
  aren't grouped identically every epoch) — batch CONTENTS are not
  randomized. **Cost, accepted deliberately:** batches are non-i.i.d. in
  sentence length (a narrow duration band per batch, not a uniform
  sample) — standard practice in speech recognition, not a bug.
- **DataLoader: `num_workers=2`, `persistent_workers=True`,
  `pin_memory=True`.** Two rounds of synthetic benchmarking picked wrong
  values that didn't survive contact with real data; only a real
  measurement against the actual dataset settled it, and moving the
  dataset to an SSD — not a `num_workers` change — is what actually
  resolved the bottleneck (`docs/FINDINGS.md`: "Storage medium, not
  DataLoader configuration, was the training bottleneck"). **Methodology
  lesson:** a benchmark must reproduce the real access pattern (many
  distinct files, real variable lengths, no page-cache reuse) or it
  measures something else — an early benchmark reported ~1 ms/item on a
  small, cache-resident single-session set; real training walks 3,150
  distinct trials across 14 files and measured 70–93 ms/item on the
  original spinning disk. Dataloader wait is now under 2% of epoch time
  at `num_workers >= 1`; GPU compute is the dominant cost.
- **Normalization is already applied upstream** — a rolling per-channel,
  per-sentence window, NOT per-block/session/global
  (`docs/DATASET.md` section 9 item 7). `data/normalize.py` must not
  implement any block/session/global z-score of its own; the values are
  already normalized by a mechanism outside our control.
- **GPU budget is not a constraint on model size.** The full baseline
  architecture (5 layers, 1024 hidden) trains with no OOM risk even at
  batch=64 (`docs/FINDINGS.md`: "GPU compute is not a constraint");
  real-training peak VRAM for the actual small/default config is 1.75 GB
  (`docs/FINDINGS.md`: "Storage medium, not DataLoader configuration...").
  No need to shrink the model to
  fit a GPU memory budget.
- **PER is independent of batch composition.** `GRUCTC.forward` zeros
  every padded timestep immediately after `DayLayers` and before the
  smoothing convolution; `input_lengths` is a mandatory forward argument.
  Fixed 2026-09-04 after a real bug was found and measured — see
  `docs/FINDINGS.md`: "Batch composition changed decoded output" for the
  mechanism and the before/after numbers.

---

## 9. Metrics

**PER (phoneme error rate) is primary.** Measures the neural decoder in
isolation, directly from `seq_class_ids`, with no downstream language
model in the loop.

**WER is out of reach on this machine — a stated limitation, not silently
skipped.** The published pipeline's LM stage needs ~60 GB RAM (3-gram,
300 GB for 5-gram) and ~12.4 GB VRAM for OPT-6.7B rescoring
(`docs/DATASET.md` section 15). On a 16 GB Windows machine, PER is
achievable and WER is not, short of a heavily pruned n-gram or a rented
large-memory instance.

**Silence is scored, not excluded.** CTC blanks are removed and
consecutive repeats collapsed per standard decoding, but the silence
token (class 40) is scored as a regular token in the Levenshtein
alignment — the field convention for this dataset lineage. **Why:**
silence marks word boundaries; a decoder that gets every phoneme right
but misplaces silence produces text a downstream LM can't use for word
segmentation, so excluding silence from PER would flatter the decoder
beyond what it can actually support. **Visualization caveat only, not a
PER change:** silence is ~25% of tokens and will dominate any confusion
matrix at a shared scale — plot it separately or normalize per row.

**`effective_signal` is a reported metric alongside PER, not a
replacement for it.** `effective_signal = PER_shuffled − PER_real`,
where `PER_shuffled` replaces each trial's `input_features` with a
length-compatible donor trial's from the SAME session (targets
unchanged) — `scripts/03_evaluate.py --shuffled-control`,
`b2t_drift.evaluation.shuffled_control`. **Why raw PER cannot do this
job alone:** a degenerate, non-functional decoder does not score 1.0 on
this task — the empty-hypothesis ceiling is exactly 1.000 (confirmed),
but a constant-silence hypothesis already scores 0.962–0.966
(`docs/FINDINGS.md`: "The effective-signal control"), and greedy CTC
decoding on genuinely uninformative input tends to land near there too
— so a real PER of ~0.85–0.90 is consistent with EITHER "still degrading
gracefully" or "already extracting nothing," and raw PER alone cannot
tell the two apart. `effective_signal` can: it is the PER gap between
the model's real decode and its decode of scrambled input for the SAME
session, i.e. how much the model gains from receiving genuine neural
data. It falls to near-zero (`docs/FINDINGS.md`'s measured slope:
−0.0140 per 100 days, reaching statistical zero by +453d) exactly where
the decoder has stopped using neural information, regardless of what raw
PER happens to read there. Report it alongside every drift-curve PER
value from a cut point onward, not only at the two endpoints.

---

## 10. Model hyperparameters

`src/b2t_drift/models/gru_ctc.py`'s `GRUCTC.__init__` and
`src/b2t_drift/models/day_layers.py`'s `DayLayers` are this project's only
model family (`CLAUDE.md`: no premature abstraction). Tags: `principled`
(physical/architectural argument), `convention` (standard in this
literature), `judgment call` (reasonable given constraints, not derived),
`fixed by data` (not a free hyperparameter).

| Hyperparameter | Value | vs. published baseline | Reason | Tag |
|---|---|---|---|---|
| `n_layers` | 3 | 5 | Our Switchboard-only train set (3,150 trials) is much smaller than the baseline's (~8,000); depth is the first thing to cut when data is limited. 3 is also the common floor before RNNs underfit temporal structure. | judgment call — revisit if underfitting |
| `hidden_size` | 512 | 768 | Same data-budget reason. Matches input dim, keeping the first layer square. | judgment call — sweep this |
| `dropout` | 0.4 | — | High by general standards, standard in this literature (small, noisy datasets need heavy regularization). | convention |
| `bidirectional` | False | — | A bidirectional GRU needs the whole utterance before emitting — fine offline, impossible in the real-time system this dataset models (emits a phoneme every 80 ms while still speaking). Causal is the honest default; bidirectional is a legitimate ablation, not the default. | principled |
| `smoothing_sigma_bins` | 2.0 | — | From the published pipeline (2 bins = 40 ms). Articulatory gestures last ~50–150 ms, so 40 ms of smoothing averages away per-bin variance without blurring the gestures. | principled |
| `n_classes` | 41 | — | CTC blank (0) + 39 ARPABET phonemes (1–39) + silence (40); verified against `seq_class_ids`. | fixed by data |
| `input_dropout` | 0.0 | — | Off by default; channel dropout simulating electrode failure is a later ablation, not a training default — enabling it now would confound the baseline. | judgment call |
| `n_features` | 512 | — | 256 threshold-crossing + 256 spike-band-power channels at 20 ms bins; not tunable. | fixed by data |
| `patch_size` | 14 | 14 (7,168-dim GRU input) | Concatenate 14 consecutive 20 ms bins into one GRU input vector, matching the published pipeline. `unfold`-based, applied after smoothing, before the GRU (`tests/test_patching.py`); `ctc_input_lengths` divides lengths to the patched axis before `nn.CTCLoss` (`CLAUDE.md`'s "known pitfalls"), verified against every s15 train+val trial. Round-5 s15 comparison: patching cut val PER from the ~0.196–0.200 no-patch plateau to ~0.169–0.170 (`docs/FINDINGS.md`: "Day layers are unstable when trained jointly across many sessions"). `1` (no-op) remains available for an ablation. | principled |
| `patch_stride` | 4 | 4 | Step the 14-bin window every 4 bins (80 ms output stride — the published real-time phoneme rate). Paired with `patch_size`; both read from `self`, so `_patch` and `ctc_input_lengths` cannot disagree. | principled |
| `day_layer_parameterization` | `"diagonal"` | `"full"` (512×512 per session) | Per-channel gain+offset only, 1,024 params/session vs. `"full"`'s 262,656. `"full"` **diverges** when all 45 session slices train jointly, even at `day_layer_penalty=3000` (`docs/FINDINGS.md`: "Day layers are unstable when trained jointly across many sessions"); `"diagonal"` is stable and reached full PER parity with the no-day-layer config on s15, with slices staying within `max|W−I| ≈ 0.04` of identity after 150 epochs. It is also the parameterization Arm 2 (`docs/DESIGN.md` section 5) can realistically fit from 10–50 labelled sentences. `"lowrank"` remains implemented and untested. | judgment call — `"full"` matches the published checkpoint but is unworkable jointly at this scale |
| `day_layer_penalty` (λ) | 300.0 | n/a (published checkpoint's day layers are unregularized) | Weight on `DayLayers.identity_penalty()`, added to the CTC loss. The penalty is **Ridge-style in form** (a squared Frobenius norm of the per-session weight deviation plus a squared-L2 bias term) **but its shrinkage target is the identity matrix `I`, not `0`** — pulling a day layer's weight toward `0` would annihilate its input (`y = x·0 + b`), whereas the no-op transform is `W = I, b = 0`. A plain `weight_decay=1e-4` diagnostic on Model A barely helped for exactly this reason: it regularized toward the wrong point. `λ=300` was measured (not guessed) to contribute ~0.11% / 3.05% / 7.89% of total loss at epochs 1 / 5 / 10 on real s15 data — an anchor, not a dominator; `λ=3000` reaches 46% by epoch 10 and starved the CTC objective (`docs/FINDINGS.md`: "Choosing the identity-anchored day-layer penalty's lambda", "Day layers are unstable…"). With `"diagonal"` the penalty is only lightly load-bearing (slices stay near identity on their own) but is kept for the cut-point sweep's smaller train sets. | judgment call — λ measured against real runs |
| `use_day_layers` | True, identity init | Confirmed from pretrained checkpoint (`docs/FINDINGS.md`) | A session with no trained layer falls back to a no-op, not a random projection — makes "how much calibration data does a new day need" well-posed (zero data = identity transform). **KEPT by the pre-registered keep/drop rule (`docs/FINDINGS.md`: "Pre-committed rule for keeping or dropping day layers"):** the round-5 s15 comparison's best day-layer config (`"diagonal"`, λ300, patched) reached val PER 0.16932 vs. 0.1697 for the best no-day-layer config — a −0.0004 gap, far inside the rule's 0.03 band (≈3× checkpoint-selection noise). Round-4's apparent no-day-layer advantage was an artifact of comparing an unconverged run against `"full"` day layers; with `"diagonal"` and 150 epochs the mechanism costs nothing. Question 3's Arm 2 and the +0.313 day-layer-gap finding both depend on it. | principled |

**Memory cost of `DayLayers`.** Dense `(n_sessions, n_features,
n_features)` float32 — 45×512×512×4 bytes ≈ 47 MB, a large fraction of
total parameter count. Sized deliberately, not a negligible add-on.

**Per-session freezing uses a gradient hook, not `requires_grad`**, since
`requires_grad` can't mark one session's slice of a shared
`(n_sessions, ...)` tensor trainable while the rest stay frozen —
`DayLayers.freeze_all_except(session_index)` / `unfreeze_all()` register a
backward-time hook instead. Caveat: Adam's momentum can still apply a few
steps of decaying updates to a newly-frozen slice if it had nonzero
optimizer state before freezing — freeze before the fine-tuning
optimizer's first step, or use a fresh optimizer for that phase.

---

## 11. Stated limitations

- **Single participant.** Everything here is one person's brain. Nothing
  generalizes across subjects.
- **All results offline.** Closed-loop performance differs — the
  participant adapts to the decoder's behavior in real use, and nothing
  here captures that adaptation.
- **Speaking strategy is unrecoverable.** Some blocks are vocalized, some
  silent speech; the authors don't publish which, so it can't be
  controlled for.
- **Test partitions are unlabeled.** 1,450 trials (13%) are unusable for
  supervised work.
- **Normalization is sentence-indexed, not time-indexed.** Sessions with
  the same day-offset don't necessarily have the same number of
  normalization-window updates (median ~42 trials/block,
  `docs/FINDINGS.md`). Check whether per-session PER correlates with
  session trial count before attributing an effect to elapsed time.
- **The first ~20 sentences of each session are normalized on a
  still-filling window** and are noisier as a result. Elevated PER there
  is expected to be a normalization artifact, not drift — see section 12
  for the resulting warm-up-trials decision.
- **ALS progression may change what the participant can attempt to
  articulate, independent of electrode health — indistinguishable from
  electrode degradation in this dataset.** Neither this dataset nor this
  project's design can resolve that ambiguity; state it alongside any
  drift-curve result.
- **Best-epoch selection carries roughly ±0.01 PER of noise.** Measured on
  the s15 100-epoch diagnostic run: `std(val_per)` over the last 20 epochs
  is 0.0091, and the selected best epoch (92, PER 0.353) differs from the
  final epoch (99, PER 0.371) by 0.018 — about two standard deviations of
  that noise (`docs/FINDINGS.md`: "Pre-committed cut-point selection
  rule" addendum). **Differences below about 0.02 between cut points
  should not be treated as meaningful** — that gap is comparable to
  checkpoint-selection noise alone, before any real difference between
  cuts is considered.
- **Sentence length correlates with session date; Switchboard-only
  reduces but does not eliminate this confound.** Session level (n=45):
  Pearson r=0.27 (p=0.074, n.s.) but Spearman ρ=0.47 (p=0.0012) — driven
  by several much-shorter 50-Word-corpus sessions scattered
  non-monotonically through the timeline (train median seq_len 26.0 vs.
  28.25 for the 30 held-out sessions). Trial level, Switchboard only
  (n=6,491): still a significant, weak upward trend (r=0.137,
  p=1.3e-28). **Verdict: mild-to-moderate, not eliminated by
  Switchboard-only alone** — report per-session median seq_len alongside
  PER on every drift curve (`figures/sentence_length_over_time.png`).

---

## 12. Decisions fixed in advance

A standing record of decisions made — and recorded here — before the
evidence that could have biased them: before any implementation in
`src/b2t_drift/data`/`models`/`training`, and (for anything touching
evaluation) before looking at any PER value, per section 7's validation
rule.

**Warm-up trials: DECIDED — keep all trials, exclude none.**

- *Case against keeping them:* the first ~20 trials of each session are
  normalized on a partially-filled window (as few as 1 prior sentence)
  and are noisier as a result (section 11).
- *Case for keeping them, which won:* excluding them discards data from
  already-small sessions (as few as 80 trials); the exclusion threshold N
  would itself be an arbitrary, undocumented parameter; and every session
  is affected equally by this window-filling effect regardless of its
  calendar date, so it cannot masquerade as drift — at worst it adds
  uniform noise, not a directional bias.
- **Consequence:** expect noisier per-trial behavior at the start of each
  session's evaluated trials; the mandatory bootstrap confidence
  intervals (section 7) account for this rather than trial exclusion.
