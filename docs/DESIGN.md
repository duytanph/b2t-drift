# Design decisions: the durable record of WHY

This file records design decisions and their justification, as they
stand now. `docs/DATASET.md` records WHAT the data is; `docs/FINDINGS.md`
records the dated empirical investigations behind these decisions. When
a decision changes, replace the old text with the new decision and its
measured justification, plus a one-line pointer to the FINDINGS.md entry
recording the change. Do not narrate the revision here. If this file
exceeds 400 lines, the excess is almost always revision narrative that
belongs in FINDINGS.md. Target length: under 400 lines.

## Summary

Speech BCI decoders lose accuracy across days; this project measures that
loss, attributes it to specific causes, and compares recalibration
strategies under a fixed budget of participant time.

- **Measurement**: how fast does PER degrade with days since calibration?
- **Attribution**: settled (section 6). SNR loss, tuning change, and
  subspace rotation confirmed and confounded; channel death ruled out.
- **Decision**: settled (section 5). Full fine-tuning wins throughout;
  recalibration slows decline but does not restore function.

**Split:** train on an early block of sessions, swept at four cut points
(1–5/1–10/1–15/1–20); evaluate each later session individually as a
PER-vs-days-elapsed curve. Temporal only, never shuffled. Switchboard-only
is the primary analysis.

**Status:** architecture settled, the four-cut drift sweep is complete,
and the primary drift curve has been produced (`docs/FINDINGS.md`: "The
cut-point sweep: calibration-data advantage is front-loaded and fades by
~100 days"). s05 is the primary analysis per the pre-committed
cut-point-selection rule; s15 is reported alongside as a robustness
check.

---

## 1. Thesis

Speech BCI decoders lose accuracy across days; this repo quantifies the
loss, attributes it to specific changes in the recorded features, and
compares recalibration strategies under a fixed budget of user time (see
Summary above for the one-paragraph version).

---

## 2. Terminology

- **"Our split"**: the temporal train/held-out boundary this project
  defines, cutting the 45 sessions by date into a training block
  (starting at session 1, `t15.2023.08.11`) and everything after it.
  Swept across four cut points (1–5, 1–10, 1–15, 1–20, section 7); "our
  split" means this general design, not one fixed session count.
  Entirely our own construction, not shipped as files.
- **"Official split"**: the `data_train.hdf5` / `data_val.hdf5` /
  `data_test.hdf5` files shipped for the Brain-to-Text '25 competition.
  Relevant only because its `test` partitions are unlabeled (13% of the
  dataset) and the provided pretrained checkpoint was trained on all 45
  sessions (`docs/FINDINGS.md`: "Block count decreases..."), so it has no
  held-out future and cannot measure drift.
- **"Held-out"** always means *withheld from our model*, never the
  official split's `val`/`test` files, which are a different withholding
  and (section 7) not useful for drift.

---

## 3. Two models

This project trains two distinct models. Conflating their numbers is a
common, easy mistake: keep them separate in every results table.

- **Model A: official-split pipeline check, EQUIVOCAL (2026-10-23).** Trained on the official split's `train` partition (all 45 sessions, all corpora,
  8,072 trials); exists to confirm our data loading, GRU/CTC core, patching, and training loop produce a PER on the same scale as published results.
  The shipping learning rate (0.001) diverges at this scale (~37,800 optimizer steps over 150 epochs vs. ~4,200 for Model B at s05); a day-layer-off
  control reproduced the same divergence, ruling out the diagonal-for-full day-layer substitution as the cause. At 10x lower (0.0001), training is
  stable for the full 150 epochs but reaches only val PER **0.2211**: equivocal against the pre-registered bands (pass <=0.20, fail >0.25), not a pass
  (`docs/FINDINGS.md`: "Model A's divergence is a learning-rate transfer problem, not the pipeline; the result is equivocal"). **The failure mode is
  an optimization instability, not a pipeline defect: localized to the 45-session/all-corpora regime, independent of day layers, outside every regime Model B exercises.** Not comparable to Model B.
- **Model B: our-split drift model.** Trained on **our split** (section
  2): the temporal cut-point sweep. Its PER curve across held-out
  sessions is the project.
- **The paper's published numbers (`docs/DATASET.md` section 2) are not a
  drift comparison, for either model.** The published system was
  continuously, periodically recalibrated; Model B is calibrated once and
  frozen. A gap between our curve and the paper's flat one is therefore
  expected by construction, not evidence of a worse decoder.

---

## 4. Three research questions

- **MEASUREMENT**: How fast does PER degrade as a function of days since
  calibration?
- **ATTRIBUTION**: Why does it degrade? See section 6 for the three
  candidate causes.
- **DECISION**: Given a fixed budget of participant time on a new day,
  what is the best thing to spend it on? See section 5 for the candidate
  interventions ("arms").

---

## 5. Recalibration strategies (Question 3 arms)

Candidate interventions for the DECISION question, increasing order of
participant-time cost. Phase 3 split Arm 3 into 3a/3b below.

| Arm | Labels needed | Targets | Notes |
|---|---|---|---|
| **0: no adaptation (baseline)** | None | n/a | Day-0 calibrated decoder, unmodified. Zero participant time. This is what the drift curve itself measures. |
| **1: unsupervised distribution alignment** (CORAL, Procrustes) | None (unlabeled new-day data only) | Subspace rotation | Linear alignment fit from unlabeled new-day features alone. **REFUTED as a correction**: verdict below. |
| **2: new day-specific input layer, rest of network frozen** | N labeled sentences | Tuning change + subspace rotation | Structurally identical to the `day_weights`/`day_biases` layers in the official checkpoint (`docs/FINDINGS.md`). lr=1e-5, measured separately from Arm 3b's (`docs/FINDINGS.md`: "A learning rate measured for one arm does not transfer to another"). **CLOSED: insufficient capacity**: verdict below. |
| **3a: output projection layer only** | N labeled sentences | Whatever a 41-class linear readout can absorb | Intermediate cost between Arm 2 and full fine-tune. lr=1e-4, also measured separately. |
| **3b: fine-tune the whole network** | N labeled sentences | Everything the other arms target, plus any deeper change | Highest cost, lr=1e-3. **Wins at every elapsed time measured** bar one exception: verdict below. Catastrophic-forgetting risk at small N on lightly-drifted sessions. |

A fourth arm (recomputing per-session z-score statistics) is retired:
the data already does this via a rolling per-channel z-score (`docs/DATASET.md` section 9 item 7).

**Arm 2's day layer uses the `"diagonal"` parameterization at λ=300**
(1,024 params/session; see section 10 for the full justification), nearly
free to omit per-session: the identity-redirect control costs **−0.0038**
PER on average vs. **+0.313** under `"full"` (`docs/FINDINGS.md`: "The
identity-day-layer redirect control, repeated under diagonal day layers").

**Recalibration verdict (Phase 3 complete):**

| Arm | Verdict | FINDINGS.md pointer |
|---|---|---|
| 1: unsupervised alignment | **REFUTED**: benefit runs OPPOSITE to measured rotation magnitude (Procrustes, ρ=−0.9 to −1.0); CORAL shows nothing coherent. Rotation is a correlate of drift, not a correctable cause. | "Subspace rotation is a correlate of drift, not a correctable cause" |
| 2: day layer, backbone frozen | **CLOSED**: at a safe lr the layer barely leaves identity (max\|scale−1\|≈0.004); 1,024 params is enough to be stable under joint training, not enough to correct one new day | "A diagonal day layer lacks the capacity to correct accumulated drift" |
| 3b: full fine-tune | **WINS** at every elapsed time measured except +9d at N=10 (worse than nothing there: a minimum viable budget exists below which recalibration is counterproductive) | "Full fine-tuning is the best use of a fixed participant-time budget...", "There is a minimum viable calibration budget..." |

**Answer to Question 3:** spend the budget on full fine-tuning. It slows
the PER decline but does not restore day-0 function, recovery-fraction
falls in two sharp steps rather than smoothly, and a minimum viable
budget exists below which recalibration hurts (`docs/FINDINGS.md`:
"Recalibration slows the decline but does not restore function").

---

## 6. What a rising PER curve means

The release pipeline applies three independent adaptive corrections
before any data reaches our model: this bounds what the ATTRIBUTION
question (section 4) can conclude.

| # | Correction | Timescale | Removes |
|---|---|---|---|
| 1 | RMS spike-detection threshold (`docs/DATASET.md` section 9 item 3) | ~50 sentences | Slow drift in a channel's noise floor |
| 2 | Rolling per-channel z-score (`docs/DATASET.md` section 9 item 7) | 20 sentences (rolling) | Per-channel gain/offset changes |
| 3 | Day-specific input layer (`docs/FINDINGS.md`: "Block count decreases...") | Trained per session | Whatever a learned linear map can capture |

Stage 1 tracks noise, not signal: a channel can appear stable while
dying if its noise floor moves with it, so mortality analysis must use
recovered raw spike counts and clip-saturation (`docs/DATASET.md`
section 9 items 8–9), not crossing rate. Stage 2 removes per-channel
affine changes only. Stage 3 is our own intervention (Arm 2). Marginal
drift is removed from view by stages 1–2 and cannot be studied here.

**Attribution verdict (Phase 2 complete):**

| Mechanism | Verdict | FINDINGS.md pointer |
|---|---|---|
| Channel death | **RULED OUT**: 2/256 channels crossed below 2 Hz and stayed, both at the final session; population trend n.s. (p=0.131) | "Channel death across 45 sessions: ruled out" |
| SNR loss | **CONFIRMED at the feature level** (TX declines 26× faster than SBP, p=3.0e-6 gap widening) **but not at the decoder level**: TX/SBP are redundant (median pair r=0.72, stable across 611d), so an SBP-only decoder is not more drift-resistant | "SNR loss confirmed...", "SBP-only is not more drift-resistant..." |
| Tuning change | **CONFIRMED, survives the healthy-channel control**: correlation decays −0.056/100d regardless of excluding dying/low-rate channels | "Tuning change confirmed via CTC forward-backward..." |
| Subspace rotation | **CONFIRMED, survives the control, tracks PER at ρ=0.908**: as tightly as tuning change, not more | "Subspace rotation confirmed..." |

**Drift is multi-causal; this design cannot rank the three confirmed
mechanisms.** They are inter-correlated at |ρ| 0.70–0.90, not
independent evidence, but not sloppy measurement: one physical process,
array micromotion through tissue over months, plausibly produces
reduced spike amplitude, altered neuron sampling, and coherent
reorientation simultaneously. SBP magnitude alone stands apart,
uncorrelated with everything else (all p>0.23): it resists the shared
process better at the feature level, though a decoder built from it
alone is not thereby protected.

---

## 7. Split design and rationale

**Cut-point sweep, not one fixed cut.** Trained separately at sessions
1–5, 1–10, 1–15, and 1–20 (each losing session 1 to the corpus filter:
100% 50-Word, zero Switchboard trials), producing four drift curves:
tests whether the result depends on an arbitrary cut, and measures how
much calibration data buys drift-resistance. More training sessions
gives a stronger baseline but a shorter, more recent held-out axis,
which is why the sweep stops at 1–20. Temporal, not random:
extrapolation into the future, not interpolation within a shuffled pool.
Bootstrap CIs are mandatory (early held-out sessions have as few as 80
trials), and early-stopping data comes from WITHIN the training
sessions, never held-out ones (`docs/FINDINGS.md`: "Cut-point sweep
manifests built and validated; a 126-day coverage gap").

| cut | train trials | active Switchboard sessions | held-out coverage |
|---|---:|---:|---|
| s05 | 908 | 4 | +2 to +597d |
| s10 | 2,098 | 9 | +2 to +562d |
| s15 | 3,150 | 14 | +5 to +546d |
| s20 | 3,336 | 19 | +100 to +513d |

**The +2-to-+133-day region is a smooth ramp, not a step**: s05's early
sessions show PER rising from 0.277 at +2d to 0.746 at +126d with no
discontinuity, corroborated by s10 (`docs/FINDINGS.md`: "The cut-point
sweep: calibration-data advantage is front-loaded and fades by ~100 days").

**Primary cut point: s05**, per the pre-committed rule (largest cut
point with acceptable validation PER, preferring earlier cuts when
comparable within 0.02, per `docs/FINDINGS.md`: "Pre-committed cut-point
selection rule"). **s15 is reported alongside as a robustness check**:
the rule is silent on generalization/stability, both of which s05
scores worst on (same entry's addendum, and "The cut-point sweep...").

**Corpus stratification: Switchboard-only** is the primary analysis;
all-corpora is a fallback, since pooling would conflate task-distribution
shift with neural drift: the training block is ~87–88% Switchboard
while at least 8 of 30 held-out sessions have zero Switchboard trials
(`docs/FINDINGS.md`: "Corpus composition is a serious training/eval
confound"); this is an 11.8% training-data cost at s15. The ≥50-trial viability
threshold was fixed before any PER value existed. The official split
cannot measure drift for the same reason (`docs/DATASET.md` sections
12–13): every session contributes training data, so it shows near-zero
drift by construction; the pretrained checkpoint has no held-out future
and is useful only for validating our evaluation code.

---

## 8. Pipeline constraints derived from data inspection

- **Lazy HDF5 loading.** The dataset is ~12 GB against a 16 GB system RAM
  budget: the Dataset class reads lazily per `__getitem__`, never holds
  everything in memory.
- **Length-bucketed batching** (`LengthBucketedSampler`, `data/collate.py`).
  `n_time_steps` spans 138–2475 bins; random batching wastes most of each
  batch on padding (47.6% vs. 9.3% bucketed, `batch_size=32`/`drop_last=True`
  as shipped; `tools/measure_padding_waste.py`). Only batch ORDER is shuffled
  per epoch. **Cost, accepted deliberately:** batches are non-i.i.d. in
  sentence length, standard practice in speech recognition, not a bug.
- **DataLoader: `num_workers=2`, `persistent_workers=True`, `pin_memory=True`.** Settled by a real measurement on SSD storage; wait is
  under 2% of epoch time at `num_workers >= 1`, GPU compute dominates. **Benchmarking rule, violated twice:** a synthetic benchmark
  must reproduce the real access pattern AND the real data dimensions, and any parameter standing in for a measured property must cite
  its source, so correcting the property corrects the benchmark: violated here (page-cache hits, not real SSD reads) and by the GPU
  benchmark below (`seq_len=527`, an uncited copy of one session's statistic) (`docs/FINDINGS.md`: "Storage medium, not DataLoader
  configuration, was the training bottleneck").
- **Normalization is already applied upstream**: a rolling per-channel,
  per-sentence window, NOT per-block/session/global (`docs/DATASET.md`
  section 9 item 7). `data/normalize.py` must not implement its own.
- **GPU headroom at the real sequence length is 61%, not 42%; batch=64 does not fit in fp32.** `seq_len=845` (real median):
  batch=32/fp32 is 5,029 MB/673.70 ms per step (61% of 8,192 MiB); batch=64/fp32 reports 9,527 MB but does NOT fit, since WDDM silently
  spills to system RAM with no `OutOfMemoryError` raised; watch step time (22.3x, not ~1.8x elsewhere), not exceptions. fp16/batch=64
  genuinely fits (5,130 MB). Same mechanism as the N=100 recalibration-grid slowdown (26x, `loop.py`), predictable from a benchmark at
  the real length (`docs/FINDINGS.md`: "At the real median length, batch=64 does not fit in 8 GB; it silently spills through WDDM").
- **PER is independent of batch composition.** `GRUCTC.forward` zeros
  every padded timestep right after `DayLayers`, before smoothing;
  `input_lengths` is a mandatory forward argument (`docs/FINDINGS.md`:
  "Batch composition changed decoded output").

---

## 9. Metrics

**PER (phoneme error rate) is primary**: measures the neural decoder in
isolation, directly from `seq_class_ids`, with no downstream language
model in the loop. **WER is out of reach on this machine:** the LM
stage's resource requirements exceed a 16 GB Windows machine
(`docs/DATASET.md` section 15). PER is achievable, WER is not, short of
a heavily pruned n-gram or a rented large-memory instance.

**Silence is scored, not excluded.** CTC blanks are removed and
consecutive repeats collapsed per standard decoding, but the silence
token (class 40) is scored as a regular token in the Levenshtein
alignment: the field convention. Silence marks word boundaries; a
decoder that gets every phoneme right but misplaces silence produces
text a downstream LM can't use for word segmentation, so excluding it
would flatter the decoder beyond what it can support. Silence is ~25% of
tokens and will dominate any confusion matrix at a shared scale: plot
it separately or normalize per row.

**`effective_signal` is a reported metric alongside PER, not a
replacement for it.** `effective_signal = PER_shuffled − PER_real`,
where `PER_shuffled` replaces each trial's `input_features` with a
length-compatible donor trial's from the SAME session (targets
unchanged): `scripts/03_evaluate.py --shuffled-control`. **Why raw PER
cannot do this job alone:** a degenerate, non-functional decoder does
not score 1.0 on this task: the empty-hypothesis ceiling is exactly
1.000, but a constant-silence hypothesis already scores 0.962–0.966, so
a real PER of ~0.85–0.90 is consistent with EITHER "still degrading
gracefully" or "already extracting nothing." `effective_signal`
distinguishes them: the PER gap between the model's real decode and its
decode of scrambled input for the SAME session. It is CI-separated from
zero across nearly the full observed range under the settled
architecture (`docs/FINDINGS.md`: "Effective signal does not collapse
under the settled architecture"). Report it alongside every drift-curve
PER value from a cut point onward, not only at the two endpoints.

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
| `n_layers` | 3 | 5 | Our Switchboard-only train set (3,150 trials) is much smaller than the baseline's (~8,000); depth is the first thing to cut when data is limited. 3 is also the common floor before RNNs underfit temporal structure. | judgment call: revisit if underfitting |
| `hidden_size` | 512 | 768 | Same data-budget reason. Matches input dim, keeping the first layer square. | judgment call: sweep this |
| `dropout` | 0.4 | n/a | High by general standards, standard in this literature (small, noisy datasets need heavy regularization). | convention |
| `bidirectional` | False | n/a | A bidirectional GRU needs the whole utterance before emitting: fine offline, impossible in the real-time system this dataset models (emits a phoneme every 80 ms while still speaking). Causal is the honest default; bidirectional is a legitimate ablation, not the default. | principled |
| `smoothing_sigma_bins` | 2.0 | n/a | From the published pipeline (2 bins = 40 ms). Articulatory gestures last ~50–150 ms, so 40 ms of smoothing averages away per-bin variance without blurring the gestures. | principled |
| `n_classes` | 41 | n/a | CTC blank (0) + 39 ARPABET phonemes (1–39) + silence (40); verified against `seq_class_ids`. | fixed by data |
| `input_dropout` | 0.0 | n/a | Off by default; channel dropout simulating electrode failure is a later ablation, not a training default: enabling it now would confound the baseline. | judgment call |
| `n_features` | 512 | n/a | 256 threshold-crossing + 256 spike-band-power channels at 20 ms bins, not tunable. | fixed by data |
| `patch_size` | 14 | 14 (7,168-dim GRU input) | Concatenate 14 consecutive 20 ms bins into one GRU input vector, matching the published pipeline. `unfold`-based, applied after smoothing, before the GRU (`tests/test_patching.py`); `ctc_input_lengths` divides lengths to the patched axis before `nn.CTCLoss` (`CLAUDE.md`'s "known pitfalls"), verified against every s15 train+val trial. Patching cuts val PER from the ~0.196–0.200 no-patch plateau to ~0.169–0.170 (`docs/FINDINGS.md`: "Day layers are unstable when trained jointly across many sessions"). `1` (no-op) remains available for an ablation. | principled |
| `patch_stride` | 4 | 4 | Step the 14-bin window every 4 bins (80 ms output stride, the published real-time phoneme rate). It is paired with `patch_size`. Both read from `self`, so `_patch` and `ctc_input_lengths` cannot disagree. | principled |
| `day_layer_parameterization` | `"diagonal"` | `"full"` (512×512 per session) | Per-channel gain+offset only, 1,024 params/session vs. `"full"`'s 262,656. `"full"` **diverges** when all 45 session slices train jointly, even at `day_layer_penalty=3000` (`docs/FINDINGS.md`: "Day layers are unstable when trained jointly across many sessions"); `"diagonal"` is stable and reached full PER parity with the no-day-layer config on s15, with slices staying within `max|W−I| ≈ 0.04` of identity after 150 epochs. It is also the parameterization Arm 2 (`docs/DESIGN.md` section 5) can realistically fit from 10–50 labelled sentences. `"lowrank"` remains implemented and untested. | judgment call: `"full"` matches the published checkpoint but is unworkable jointly at this scale |
| `day_layer_penalty` (λ) | 300.0 | n/a (published checkpoint's day layers are unregularized) | Weight on `DayLayers.identity_penalty()`, added to the CTC loss. The penalty is **Ridge-style in form** (a squared Frobenius norm of the per-session weight deviation plus a squared-L2 bias term) **but its shrinkage target is the identity matrix `I`, not `0`**: pulling a day layer's weight toward `0` would annihilate its input (`y = x·0 + b`), whereas the no-op transform is `W = I, b = 0`. A plain `weight_decay=1e-4` diagnostic on Model A barely helped for exactly this reason: it regularized toward the wrong point. `λ=300` was measured (not guessed) to contribute ~0.11% / 3.05% / 7.89% of total loss at epochs 1 / 5 / 10 on real s15 data: an anchor, not a dominator. `λ=3000` reaches 46% by epoch 10 and starved the CTC objective (`docs/FINDINGS.md`: "Choosing the identity-anchored day-layer penalty's lambda", "Day layers are unstable…"). With `"diagonal"` the penalty is only lightly load-bearing (slices stay near identity on their own) but is kept for the cut-point sweep's smaller train sets. | judgment call: λ measured against real runs |
| `use_day_layers` | True, identity init | Confirmed from pretrained checkpoint (`docs/FINDINGS.md`) | A session with no trained layer falls back to a no-op, not a random projection: makes "how much calibration data does a new day need" well-posed (zero data = identity transform). **KEPT by the pre-registered keep/drop rule** (`docs/FINDINGS.md`: "Pre-committed rule for keeping or dropping day layers"): the best day-layer config (`"diagonal"`, λ300, patched) reached val PER 0.16932 vs. 0.1697 for the best no-day-layer config, a −0.0004 gap, inside the rule's 0.03 band. Question 3's Arm 2 mechanism depends on it staying available (section 5). | principled |

**Memory cost of `DayLayers`.** Dense `(n_sessions, n_features,
n_features)` float32: 45×512×512×4 bytes ≈ 47 MB, a large fraction of
total parameter count. Sized deliberately, not a negligible add-on.

**Per-session freezing uses a gradient hook, not `requires_grad`**, since
`requires_grad` can't mark one session's slice of a shared
`(n_sessions, ...)` tensor trainable while the rest stay frozen.
`DayLayers.freeze_all_except(session_index)` / `unfreeze_all()` register a
backward-time hook instead. Caveat: Adam's momentum can still apply a few
steps of decaying updates to a newly-frozen slice if it had nonzero
optimizer state before freezing. Freeze before the fine-tuning
optimizer's first step, or use a fresh optimizer for that phase.

---

## 11. Stated limitations

- **Single participant.** Everything here is one person's brain; nothing
  generalizes across subjects.
- **All results offline.** Closed-loop performance differs: the
  participant adapts to the decoder's behavior in real use, uncaptured
  here.
- **Speaking strategy is unrecoverable.** Some blocks are vocalized, some
  silent speech; the split isn't released, so it can't be controlled for.
- **Test partitions are unlabeled.** 1,450 trials (13%) are unusable for
  supervised work.
- **Normalization is sentence-indexed, not time-indexed.** Sessions at
  the same day-offset don't necessarily have the same number of
  normalization-window updates (median ~42 trials/block); check whether
  per-session PER correlates with session trial count before attributing
  an effect to elapsed time.
- **The first ~20 sentences of each session are normalized on a
  still-filling window** and are noisier as a result (expected to be a
  normalization artifact, not drift: section 12's warm-up-trials
  decision).
- **ALS progression may change what the participant can attempt to
  articulate, independent of electrode health**: indistinguishable from
  electrode degradation in this dataset. State it alongside any
  drift-curve result.
- **Best-epoch selection carries roughly ±0.01 PER of noise**
  (`docs/FINDINGS.md`: "Pre-committed cut-point selection rule"
  addendum). **Differences below about 0.02 between cut points should
  not be treated as meaningful.**
- **Sentence length correlates with session date; Switchboard-only
  reduces but does not eliminate this confound** (trial-level r=0.137,
  p=1.3e-28). Report per-session median seq_len alongside PER on every
  drift curve (`figures/sentence_length_over_time.png`).
- **SNR loss, tuning change, and subspace rotation are confirmed but
  cannot be separated or ranked by this design** (section 6; |ρ| 0.70–0.90
  inter-correlated).
- **Recovery-fraction's reference (section 5) is validation PER inside
  the training sessions, not a true day-zero session** (an optimistic
  ceiling, so reported fractions are lower bounds); its 5-split sd reflects
  calibration-subset variability only, and calibration/evaluation trials
  are interleaved within a session, not temporally ordered.

---

## 12. Decisions fixed in advance

A standing record of decisions made before the evidence that could have
biased them: before any implementation, and (for evaluation) before
looking at any PER value, per section 7's validation rule.

**Warm-up trials, DECIDED: keep all trials, exclude none.**

- *Case against keeping them:* the first ~20 trials of each session are
  normalized on a partially-filled window (as few as 1 prior sentence)
  and are noisier as a result (section 11).
- *Case for keeping them, which won:* excluding them discards data from
  already-small sessions (as few as 80 trials); the exclusion threshold N
  would itself be an arbitrary, undocumented parameter; and every session
  is affected equally by this window-filling effect regardless of its
  calendar date, so it cannot masquerade as drift: at worst it adds
  uniform noise, not a directional bias.
- **Consequence:** expect noisier per-trial behavior at the start of each
  session's evaluated trials; the mandatory bootstrap confidence
  intervals (section 7) account for this rather than trial exclusion.
