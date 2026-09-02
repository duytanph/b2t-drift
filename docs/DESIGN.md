# Design decisions — the durable record of WHY

This file records design decisions and their justification. `docs/DATASET.md`
records WHAT the data is; this file records why we chose to work with it the
way we do. When a decision changes, update this file and say why the old
reasoning no longer holds — don't just delete it.

---

## 1. Thesis

Speech BCI decoders lose accuracy across days; this repo quantifies the
loss, attributes it to specific changes in the recorded features, and
compares recalibration strategies under a fixed budget of user time.

---

## 2. Terminology

Precise definitions, because "split" and "held-out" get used loosely
elsewhere and that looseness causes real mistakes here.

- **"Our split"**: the temporal train/held-out boundary this project
  defines, cutting the 45 sessions by date into a training block (starting
  at session 1, `t15.2023.08.11`) and everything after it. The base case is
  sessions 1–15 train / sessions 16–45 held out, but section 5 below sweeps
  this cut point across several values (1–5, 1–10, 1–15, 1–20) rather than
  fixing it at 15 — "our split" refers to this general temporal-split
  design, not one single fixed pair of session counts. This is entirely our
  own construction; it does not exist as files anywhere.
- **"Official split"**: the `data_train.hdf5` / `data_val.hdf5` /
  `data_test.hdf5` files actually shipped with the dataset for the
  Brain-to-Text '25 competition. Relevant to us for two reasons only: (1)
  its `test` partitions are unlabeled, so 1,450 trials (13% of the dataset)
  are unusable for any supervised purpose regardless of which split we use;
  and (2) the provided pretrained checkpoint was trained on the official
  split's `train`/`val` data spanning **all 45 sessions** (confirmed,
  `args.yaml`'s `dataset.sessions` — Task 4b), so it has no held-out future
  and cannot measure drift.
- **"Held-out"** always means *withheld from our model* (i.e. not in our
  split's training block) unless explicitly stated otherwise. It never
  means the official split's `val`/`test` files unless said so explicitly
  — those are a different withholding, from the official baseline, and (per
  section 5) are not independent of each other or useful for drift.

---

## 3. Two models

This project trains two distinct models. Conflating their numbers is a
common and easy mistake — keep them separate in every results table.

- **Model A — official-split pipeline check.** Trained on the official
  split's `train` partition (all 45 sessions). Its only purpose is to
  confirm our data loading, model, and training code produce a PER on the
  same scale as published results — a correctness check, not a
  contribution. **Run once.** Not part of the drift experiment, not
  referenced in the drift curve, not comparable to Model B's numbers
  (different training data, different scale of task).
- **Model B — our-split drift model.** Trained on **our split** (section 2)
  — the temporal cut-point sweep. Its PER curve across held-out sessions
  **is the project**.
- **The paper's published numbers (`docs/DATASET.md` section 2) are not a
  comparison for drift, for either model.** The published system was
  continuously, periodically recalibrated across its whole deployment;
  Model B is calibrated once and then frozen. These measure different
  quantities — "maintained performance under an active maintenance
  regimen" vs. "performance decay under no maintenance at all." A gap
  between our drift curve and the paper's flat curve is not evidence of a
  worse decoder; it's the whole point of the study.

---

## 4. Three research questions

- **MEASUREMENT** — How fast does PER degrade as a function of days since
  calibration?
- **ATTRIBUTION** — Why does it degrade? Candidate causes: channel
  mortality (electrodes going quiet), marginal statistics drift with
  preserved tuning (the signal shifts but the underlying code doesn't),
  or genuine change in the neural code (the tuning itself changes). These
  have different fixes — you don't renormalize your way out of channel
  death, and you don't replace electrodes to fix a normalization offset.
- **DECISION** — Given a fixed budget of participant time on a new day,
  what is the best thing to spend it on: recalibrating a day-specific
  input layer, collecting more labeled data, or something else?

---

## 5. Split design and rationale

### Cut-point sweep, not a single fixed cut

**Revised.** Earlier drafts of this document fixed the training cut at
sessions 1–15. **The cut point is now swept over sessions 1–5, 1–10, 1–15,
and 1–20**, training a separate Model B for each and producing **four**
drift curves, not one.

**Why this is affordable.** The Task 3 GPU benchmark (section 9) showed
the full baseline architecture trains in ~1.3 hours per 10,000 steps on
this machine. Four training runs cost roughly 6 hours total — affordable on
a single machine, not a reason to compromise on model size or cut count.

**Why sweep instead of picking one cut and defending it.** Two distinct
benefits:
1. **Robustness.** If the drift curve's *shape* (not necessarily its exact
   PER values, since baseline strength differs by cut) is stable across
   cut points, the result does not depend on an arbitrary choice of "15."
   If it isn't stable, that's itself an important finding about how
   sensitive the measurement is to calibration-set size.
2. **A question of its own.** Comparing PER at a fixed day-offset across
   the four cuts directly measures how much calibration data buys how much
   resistance to drift — a decision-relevant result independent of the
   measurement/attribution questions in section 4.

**The tradeoff, stated explicitly.** More training sessions gives a
stronger baseline decoder (more calibration data) but a shorter time axis
(fewer, more recent held-out sessions). Example: training through session
15 leaves a held-out range of +5 to +546 days; training through session 30
(illustrative, not one of the four swept cuts) would cut that to roughly
+5 to +166 days — discarding the six 2025 sessions, the most valuable tail
of the curve. This tradeoff is exactly why the sweep only goes up to 1–20,
not further: beyond that point the shrinking time axis starts costing more
than the stronger baseline buys.

**Held-out spacing** for the 1–15 cut (the base case) is roughly
logarithmic: +5 to +546 days, dense in the first two months, sparse in the
tail — see `docs/DATASET.md` section 13 for the full list. Other cut
points have correspondingly fewer, later-starting held-out sessions.

**Why temporal, not random.** The question is extrapolation into the
future, not interpolation within a shuffled pool. Shuffling sessions across
time destroys the thing we're trying to measure — a randomly-held-out
session tells you nothing about what happens to a decoder that's aging.

**Bootstrap confidence intervals are mandatory, not optional.** Early
held-out sessions have as few as 80 trials. PER computed on 80 sentences is
noisy enough that random wiggle will look like structure without an
interval around it. Every point on every one of the four curves needs one.

### Floor effect risk

At the base 1–15 cut, the training block is 3,969 labeled trials — roughly
**17.6 hours of speech** (`total_minutes` summed over those 15 sessions,
`data/interim/session_manifest.csv`). For context, Willett et al. 2023
(the T12 predecessor dataset, `docs/DATASET.md` section 16) reached 23.8%
WER on a 125,000-word vocabulary with 16.8 hours of training data —
a comparable budget, so a usable decoder is plausible at this cut. But if
the baseline PER is poor, the drift signal sits on top of an already-weak
decoder and is harder to interpret (a curve that starts bad and stays bad
is a different finding than a curve that starts good and degrades). The
cut-point sweep partly mitigates this: the 1–20 cut has substantially more
training data and serves as a check on whether floor effects are
distorting the 1–5/1–10 curves.

### Validation set rule

**The validation set for early stopping and hyperparameter selection must
come from WITHIN the training sessions** — held-out blocks or a random
subset of trials from sessions in the training block — **never from our
held-out (evaluation) sessions.** Inspecting the drift curve and then
adjusting hyperparameters in response would fit the model to the test data
by the back door; the whole point of the curve is that it's an honest,
unpeeked-at estimate. **The drift curve is examined once, at the end**, for
every one of the four cut-point models. This applies independently to each
cut in the sweep — session 16 is a legitimate validation source for the
1–15 cut's model but must never be for the 1–5 cut's held-out evaluation of
session 16.

### Corpus stratification decision

**Switchboard-only is the PRIMARY analysis; all-trials (every corpus
pooled) is reported as a secondary curve.** Justification: the training
block is ~87–88% Switchboard by trial count (`docs/DESIGN.md` section 9,
Task 2 finding) while at least 8 of the 30 base-case held-out sessions
contain **zero** Switchboard trials at all — an all-trials curve conflates
task-distribution shift (evaluating on corpora the model never trained on)
with neural drift, which is exactly the confound this project exists to
avoid mistaking for its own subject. Switchboard-only removes the worst of
that confound while preserving the full day-offset range and, critically,
all six of the 2025 sessions (see Task 2 findings in section 9).

**The ≥50-labeled-Switchboard-trial viability threshold was fixed in
advance of seeing any PER values** (during the Task 2 feasibility check,
section 9) — a session below it is excluded from the primary curve because
PER on fewer than 50 sentences is too noisy to plot as a meaningful curve
point, not because of how it happens to score.

### Why the official split cannot measure drift

Within each session, `val` and `test` are trial-level splits of the *same*
held-out block(s) — not independent generalization estimates — and every
session (including the most recent ones) contributes training data. A
model trained on the official split has seen data from every era of the
recording, so it shows near-zero drift by construction: there is no
temporal extrapolation anywhere in that design. It measures general
decoding accuracy, which is the right target for the Brain-to-Text '25
competition and the wrong one for this project. (This is why Model A,
section 3, is a pipeline check only.)

### Why the provided pretrained checkpoint cannot be used for the drift curve

`data/external/t15_pretrained_rnn_baseline/checkpoint/args.yaml` lists all
45 sessions under `dataset.sessions` — confirmed by direct inspection
(Task 4b, section 9). It has seen every session in the dataset and has no
held-out future. It remains useful for (a) validating our evaluation code
against a known-good model, and (b) inspecting its 45 learned per-session
day-layers as a record of how much per-session correction the official
baseline needed — see Task 4 findings in section 9.

---

## 6. Pipeline constraints derived from data inspection

- **Lazy HDF5 loading.** The dataset is ~12 GB against a 16 GB system RAM
  budget. `T15Dataset` must not hold everything in memory; it has to read
  lazily from HDF5 per `__getitem__`.
- **Length-bucketed batching.** `n_time_steps` spans 138–1025+ bins, a
  7.4x range. Random batching wastes most of each batch on padding —
  batch by similar length instead.
- **Block-level normalization.** Array statistics shift within a single
  session, not just across sessions, so z-scoring must be computed per
  block, not per session. **Superseded as of the normalization-unit
  follow-up (2026-09-01), and further clarified by the Task 1 arithmetic
  investigation (2026-09-01, section 9):** the raw `input_features` values
  in the released HDF5 are already z-scored (every sampled column has mean
  ≈ 0, std ≈ 1), and the official training code applies no further
  normalization at load time or in the model — see `docs/DATASET.md`
  section 9. Block-level is ruled out as the normalization unit; an attempt
  to settle session-vs-global arithmetically found that TX values do not
  even form the evenly-spaced integer lattice the arithmetic method
  depends on, so the question could not be settled at all by that method —
  see Task 1 findings in section 9. **Do not implement `data/normalize.py`
  as a from-scratch per-block (or any-unit) z-score** — the values may
  already be adequately normalized, and we do not yet understand the
  transform well enough to safely reverse or duplicate it. Revisit only
  after a genuinely new source of evidence (e.g. contacting the dataset
  authors, or finding pipeline code we haven't yet located).
- **GPU budget is not a constraint on model size.** The Task 3 benchmark
  (section 9) showed the full baseline architecture (5 layers, 1024
  hidden) trains in ~1.3 hours per 10,000 steps at fp32 on this machine,
  with no OOM risk even at batch=64. This is also what makes the section
  5 cut-point sweep affordable (~6 hours for all four runs). Ablations and
  the main drift-curve runs can use the full-size model throughout — there
  is no need to fall back to the "small" (3 layer, 512 hidden) config to
  fit time or memory budgets.

---

## 7. Metrics

**PER (phoneme error rate) is primary.** It measures the neural decoder in
isolation, directly from `seq_class_ids`, with no downstream language model
in the loop.

**WER (word error rate) is out of reach on this machine and is a stated
limitation, not silently skipped.** The published pipeline's language-model
stage needs roughly 60 GB RAM for the 3-gram model (300 GB for 5-gram) and
~12.4 GB VRAM for OPT-6.7B rescoring (`docs/DATASET.md` section 15). On a
16 GB Windows machine, PER is achievable and WER is not, short of a heavily
pruned n-gram or a rented large-memory instance. Report PER as primary and
say plainly that WER was not computed, rather than reporting only PER
without saying why.

---

## 8. Stated limitations

- **Single participant.** Everything here is one person's brain. Nothing
  generalizes across subjects.
- **All results offline.** Closed-loop performance differs, because in
  real use the participant adapts to the decoder's behavior — nothing in
  this dataset captures that adaptation.
- **Speaking strategy is unrecoverable.** Some blocks are attempted
  *vocalized* speech and some are attempted *silent* speech; the authors
  deliberately do not publish which is which, so it cannot be controlled
  for.
- **Test partitions are unlabeled.** 1,450 trials (13% of the dataset) are
  unusable for supervised work — neither trainable nor scorable.

---

## 9. Findings from Tasks 1–4

Investigation scripts live in `scratch/` (gitignored, not committed).
Figures referenced below are committed under `figures/`.

### Task 1 — feature layout (2026-09-01)

- **TX/SBP boundary at column 256: CONFIRMED**, but not by the originally
  planned test. Threshold-crossing values are *not* raw integer counts in
  the released files — see the normalization finding below — so an
  integrality test found nothing. Instead, a **discreteness** test
  (distinct-value count per column, which survives an affine z-scoring
  transform) gives an extremely sharp result: columns 0–255 take ~0.6% as
  many distinct values as samples (small integer counts, quantized even
  after normalization); columns 256–511 take ~76% as many distinct values
  as samples (continuous). The transition is exactly at column 256.
- **64-wide array boundaries (64/128/192) within TX: CONFIRMED, weak but
  real signal.** A naive 3-way comparison of candidate boundary sets
  (0/64/128/192 vs the README-literal 0/65/129/193 vs 0/63/127/191) gave
  within-vs-across correlation gaps too close to call (0.0129 vs 0.0127 vs
  0.0125). A wider test settles it: scanning all integer shifts of the
  three internal boundaries from −16 to +16, the gap is a smooth,
  unimodal curve that peaks *exactly* at shift 0 (0.0129, vs. the next
  best neighbors at 0.0127 and 0.0125, falling off monotonically further
  out). A single 3-way comparison would have been weak evidence; the full
  scan is not. In absolute terms the correlation is still small (~0.018
  mean within-block r vs ~0.005 across-block) — electrodes on the same
  array are only weakly more correlated than electrodes on different
  arrays, consistent with TX being sparse, mostly-independent spike
  counts rather than a shared local field signal.
- **Per-block homogeneity (part c):** every one of the 8 hypothesized
  64-wide blocks (4 TX + 4 SBP) has per-column mean ≈ 0 and variance ≈ 1.
  This is expected given the normalization finding below, and by itself
  provides no additional boundary evidence (z-scoring makes all blocks
  look homogeneous by construction).
- **Verdict:** the `docs/DATASET.md` section 7 table (64-wide blocks,
  columns 0–63/64–127/.../448–511) is confirmed on two independent tests.
  The README's literal 0–64/65–128/... boundaries are not supported.
- **Unplanned but important finding — input_features appear
  pre-normalized.** Spot-checking raw HDF5 values directly (not just the
  sampled-block statistics) shows every column has mean ≈ 0, std ≈ 1, and
  TX columns take on a small repeated set of exact float values (e.g.
  `-0.5859305` recurring hundreds of times in a single trial) —
  consistent with small integer counts having already been passed through
  a fixed per-column affine transform before release. This contradicted
  the earlier draft of `docs/DATASET.md` section 9 ("Left to us:
  Normalization"), which has since been corrected there.
- Figure: `figures/feature_layout_correlation.png`

### Task 2 — corpus composition confound check (2026-09-01)

- **Corpora found: 6, not 5.** Switchboard (7,672 trials), Freq words
  (2,050), 50-Word (1,091), Openwebtext (172), Harvard (98), Random (79).
  `docs/DATASET.md` section 10 names only five corpora (Switchboard,
  OpenWebText2, 50-word, frequent-word, random); **Harvard sentences are
  an additional corpus not mentioned there** and have since been added.
- **Verdict: SERIOUS CONFOUND, not mild.** The planned TRAIN block
  (sessions 1–15) is 86.9% Switchboard / 8.5% 50-Word / 3.7% Openwebtext
  and contains **zero** Freq words or Harvard trials. The 30 held-out
  sessions are 56.0% Switchboard / 31.2% Freq words / 10.7% 50-Word / 1.5%
  Harvard — a 31-percentage-point jump in Freq words alone, from a corpus
  the training data never saw at all. Worse: **8 of the 30 held-out
  sessions have zero Switchboard trials** (7 are 100% Freq words, 1 is
  100% 50-Word). For those 8 sessions, every evaluated trial comes from a
  corpus absent from training — any PER measured there conflates neural
  drift with pure out-of-distribution task shift and cannot be
  interpreted as a clean drift estimate. This is why section 5 adopts
  Switchboard-only as the primary analysis.
- **Era concentration:** Freq words is confined to 2023-10-22 through
  2023-12-29 (11 sessions); Harvard to 2023-10-20 through 2023-11-03 (3
  sessions); Openwebtext to 2023-09-29 through 2023-10-15 (entirely
  *inside* the train window). Only Switchboard and 50-Word recur across
  the full 20-month span.
- Figure: `figures/corpus_composition_over_time.png` (stacked bars at true
  session dates, not a continuously-interpolated area chart — the two
  multi-month recording gaps would otherwise render as a fictitious smooth
  trend).

### Task 3 — GPU timing benchmark (2026-09-01)

RTX 2080 SUPER, 8 GB VRAM, synthetic tensors (batch=32, `input_size=512`,
`output_size=41`, CTC loss, forward+backward+optimizer step; 10 warmup + 50
timed steps).

| model | seq_len | precision | ms/step | peak VRAM | hours @ 10k steps |
|---|---|---|---|---|---|
| baseline (5L, 1024H) | 527 (none) | fp32 | 461.0 | 3.4 GB | 1.28 h |
| baseline (5L, 1024H) | 527 (none) | fp16 | 203.9 | 2.0 GB | 0.57 h |
| baseline (5L, 1024H) | 132 (4x) | fp32 | 120.7 | 1.3 GB | 0.34 h |
| baseline (5L, 1024H) | 132 (4x) | fp16 | 57.8 | 0.9 GB | 0.16 h |
| small (3L, 512H) | 527 (none) | fp32 | 104.9 | 1.1 GB | 0.29 h |
| small (3L, 512H) | 527 (none) | fp16 | 80.3 | 0.7 GB | 0.22 h |
| small (3L, 512H) | 132 (4x) | fp32 | 28.1 | 0.4 GB | 0.08 h |
| small (3L, 512H) | 132 (4x) | fp16 | 21.6 | 0.3 GB | 0.06 h |

- **No configuration OOM'd**, at batch=32 or batch=64. Peak VRAM at
  batch=64 topped out at 6.1 GB (baseline, fp32, no downsampling) — still
  under the 8 GB budget, but with limited headroom once the OS/desktop's
  own ~0.7 GB VRAM baseline usage (observed via `nvidia-smi`) is
  accounted for.
- fp16 autocast (`torch.float16`, not bf16 — Turing has no native bf16
  support) roughly **halves** step time across the board, as expected.
  4x temporal downsampling gives a comparable or larger speedup than fp16
  alone.
- The official baseline's real architecture is not directly comparable to
  this benchmark grid — see Task 4b below: it uses input patching
  (`patch_size=14`, `patch_stride=4`), making its GRU input dimension
  7,168, not the 512 used here. This benchmark bounds our own from-scratch
  architecture's cost, not the official baseline's.
- No `GradScaler` was used for the fp16 runs (acceptable for a pure speed
  benchmark on random targets; a real fp16 training run should use one).
- **This benchmark is why section 5's four-cut sweep is affordable** — see
  the sweep rationale above.

### Task 4 — block structure and pretrained checkpoint (2026-09-01)

**4a — blocks per session** (from `data/interim/session_manifest.csv`
only): median 6 blocks/session (range 3–11), median ~42 trials/block
(range ~22–50). Block count correlates negatively with session order
(r = −0.40): the earliest sessions tend to have more blocks (e.g.
`t15.2023.08.13` has 11) than later ones (many late sessions have 3–6).
Figure: `figures/blocks_per_session.png`.

**4b — pretrained checkpoint**
(`data/external/t15_pretrained_rnn_baseline/checkpoint/best_checkpoint`,
loaded with `torch.load(map_location="cpu", weights_only=True)` as
instructed — **this failed** with an `UnpicklingError` on
`numpy.core.multiarray.scalar`, not because the checkpoint contains
anything unsafe, but because newer numpy renamed `numpy.core` to
`numpy._core`, and torch's safe-globals allowlist matches by the pickled
module/qualname *string*, which no longer lines up with the renamed
module. Fell back to `weights_only=False` for this specific,
documented-source file (the official `nejm-brain-to-text` pretrained
baseline release).

From the state_dict shapes:
- **5 GRU layers**, hidden size **768** (`weight_ih_l0` etc. shaped
  `(2304, ...)` = 3×768 gate-concatenated).
- **GRU input size is 7,168, not 512** — because the official model
  patches the input: `args.yaml` gives `patch_size: 14, patch_stride: 4`,
  so each GRU input step is 14 consecutive 20 ms bins concatenated
  (14 × 512 = 7,168), stepped every 4 bins (80 ms) — consistent with
  `docs/DATASET.md` section 5's note that the real-time system outputs
  one phoneme every 80 ms.
- **Output size 41** (`out.weight` shaped `(41, 768)`), confirming
  `ctc_output_size=41` independently from the checkpoint.
- **Per-session day-specific input layers: CONFIRMED present**, but not
  via the "first dimension == 45" heuristic (that check found nothing).
  They're stored as 45 *separately-keyed* tensors —
  `day_weights.0` .. `day_weights.44` (each `512x512`) and `day_biases.0`
  .. `day_biases.44` (each `1x512`) — one per-session affine input
  transform, applied before the shared GRU. This is a direct architectural
  confirmation of the "day-specific input layer" concept in
  `GLOSSARY.md`.
- **`args.yaml` confirms `dataset.sessions` lists all 45 sessions** —
  direct confirmation that this checkpoint cannot be used for the drift
  curve (see section 5 above). It also lists `dataset_probability_val`, a
  0/1 flag per session; the 4 sessions flagged `0` are exactly the 4
  train-only sessions we already knew about (`t15.2023.08.11`,
  `t15.2024.03.03`, `t15.2024.04.25`, `t15.2024.04.28`) — independent
  cross-validation of that fact.
- Checkpoint's recorded `val_PER = 0.101` (10.1%), `val_loss = 22.45`, on
  the official split (i.e. not comparable to Model B's numbers, per
  section 3 — different training data, different question).
- `use_amp: true` in `args.yaml` — the official baseline was trained with
  mixed precision, same choice our Task 3 benchmark evaluates.

### Follow-up — normalization unit, statistical approach (2026-09-01)

Direct follow-up to the Task 1 finding above that `input_features` are
already z-scored. Sampled every train block from 3 sessions spanning early
2023 / mid 2024 / early 2025 (18 blocks total), recomputed per-column
mean/std at block, session (train-only and full train+val+test), and
3-session-pooled granularity (`scratch/investigate_normalization_unit.py`).

- **Per-block: rejected.** Recomputed block-level column means/stds have a
  5th–95th-percentile spread of ~0.06 across the 512 columns. If a block's
  own data were exactly what was normalized against, recomputing its mean
  on itself is an exact identity and must land within floating-point
  precision (~1e-6), not 0.06.
- **Per-session: tighter, but not confirmed.** Pooling a session's own
  train blocks gives ~0.025 spread; adding that session's val+test blocks
  (i.e. the *whole* session) tightens it further to ~0.019–0.026 depending
  on the session — closer to zero than block-level, but still an order of
  magnitude above floating-point precision. Exact per-session
  normalization, recomputed on the same population it was supposedly fit
  on, is not confirmed.
- **Global (fixed, dataset-wide statistics): the diagnostic prediction did
  not materialize.** If a single fixed mean/std were computed once (e.g.
  early in the deployment) and never updated, later sessions should show
  growing deviation from 0/1 as genuine neural drift accumulates —
  that's the residual signal the whole project is trying to measure.
  Plotting per-block column means for 6 representative columns across 12
  sessions spread over the full 20 months
  (`figures/normalization_unit_check.png`) shows **no visible trend**: all
  values stay within roughly ±0.08 of zero throughout, with no monotonic
  drift.
- **Verdict at the time: block-level rejected with high confidence.
  Session vs. global not distinguished by this evidence.** Superseded (not
  contradicted) by the Task 1 arithmetic follow-up below, which explains
  *why* this statistical test couldn't cleanly resolve session-vs-global:
  the premise the arithmetic method depends on (a simple fixed affine
  transform of integer counts) does not hold cleanly enough to test either
  hypothesis to machine precision.
- Confirmed via the official repo's `model_training/dataset.py` and
  `model_training/rnn_model.py` (fetched from GitHub) that no further
  normalization is applied downstream of the HDF5 files — whatever
  produced the observed 0/1 statistics happened entirely upstream of what
  either the official pipeline or ours currently touches. Also confirmed
  there that the per-session `day_weights`/`day_biases` layer is a
  *learned* parameter initialized to the identity transform (`torch.eye`)
  and zero bias — not a fixed normalization step, and not the source of
  the pre-release z-scoring.

### Task 1 (this round) — settling the normalization unit arithmetically (2026-09-01)

Attempted to replace the statistical test above with an exact one: TX
columns are integer spike counts before release, so if
`released = (raw - mu) / sigma`, the distinct released values in a column
must sit on an evenly-spaced lattice with spacing `1/sigma`, letting
`(mu, sigma)` be solved exactly rather than estimated statistically. Same 3
sessions, all train blocks (18 blocks), all 256 TX columns
(`scratch/recover_normalization_params.py`).

- **Result: the lattice does not exist.** Only 3 of 4,608 (session, block,
  column) triples showed evenly-spaced values under the test — and all 3
  are float32-precision artifacts, not real lattices: their recovered
  `sigma` values are exactly `2^24` and `2^25` (16,777,216 and
  33,554,432), the float32 mantissa-precision limit, and "inverting" them
  gives absurd raw counts up to ~350 million. **99.9% of cases showed no
  constant spacing at all.**
- **This was checked for robustness, not just accepted at face value.**
  One column/block (session `t15.2023.08.13`, block 1, column 0) has 212
  distinct values over 40,727 samples (0.52% — consistent with the
  "discrete-looking" TX signature from Task 1 of the prior round), but:
  - The most common single value covers only 2.7% of samples — not the
    heavily zero-dominated distribution a sparse spike count would
    produce.
  - Consecutive-difference sizes span ~5 orders of magnitude (1e-5 to
    ~1) with **no bimodal gap** separating "same level" from "different
    level" — ruled out a min-diff artifact by sweeping candidate spacings
    at several percentiles of the diff distribution (p1/p5/p10/p25/p50);
    the best fit found any of them was 17.5% of diffs matching an integer
    multiple, far short of the ~99% a real lattice would show.
- **Control (step 5): SBP columns show no lattice either** (0/108
  (block, column) cases), as expected for genuinely continuous data —
  this validates that the method correctly distinguishes lattice from
  non-lattice structure when a non-lattice case is actually presented to
  it, so the TX non-result is not simply a broken detector.
- **Verdict, per the decision rule: NOT A SIMPLE AFFINE TRANSFORM OF
  INTEGER COUNTS. Stopping rather than reporting mu/sigma.** Steps 3
  (compare recovered params across blocks/sessions) and 4 (invert on a
  full session, report firing-rate range) do not apply — there was nothing
  to compare or invert. `scratch/recovered_normalization.csv` was still
  saved with the per-(session, block, column) test results for the record,
  including the 3 float32-artifact false positives, but it is not a usable
  parameter table.
- **What this means for the open question:** the normalization unit
  (global vs. per-session) remains unresolved, but for a more fundamental
  reason than "the statistical test wasn't sharp enough" — the simplest
  hypothesis about *what kind* of transform produced these values (a fixed
  per-column affine rescaling of integer counts) is itself not well
  supported by the data. Plausible alternatives, none confirmed:
  time-varying/adaptive (e.g. rolling or exponentially-weighted) online
  normalization; smoothing or filtering applied to the counts before
  normalization; or a transform other than z-scoring entirely. Do not
  guess further without new evidence (e.g. contacting the dataset authors
  or finding source code we have not yet located).

### Task 2 (this round) — Switchboard survivor table, corrected (2026-09-01)

Rebuilt the Switchboard-viability table from Task 2 (prior round) using the
method the Task 3 discrepancy investigation (below) showed is necessary:
per-(session, block) trial counts read directly from the HDF5 files
(train+val partitions only), joined to the CSV's block-level corpus label,
rather than trusting the CSV's own per-block trial counts
(`scratch/switchboard_survivor_table.py`).

- **17 of 30 held-out sessions survive** the ≥50-labeled-Switchboard-trial
  threshold (down from the prior round's less precise estimate of 19/30 —
  see the Task 3 correction below for why). Day-offset range of survivors:
  **5 to 546 days**, unchanged — the full range is still preserved.
- **All four requested curve endpoints PASS:** `2023-10-20` (+5d, 98
  trials), `2023-10-22` (+7d, 83 trials), `2025-03-14` (+516d, 83 trials),
  `2025-04-13` (+546d, 94 trials).
- **TRAIN block (sessions 1–15):** 3,501 labeled Switchboard trials out of
  3,969 total labeled trials (88.2%) — consistent with, and a slightly
  more precise version of, the prior round's ~87–88% estimate.
- Two sessions flip from the prior round's PASS to FAIL under the
  corrected method: `2023-11-03` (+19d, was counted as exactly 50 using
  the CSV's raw count, actually only 25 labeled Switchboard trials — see
  the worked example in the Task 3 finding below) and `2023-12-29` (+75d,
  similarly overcounted by the CSV).
- Figure (redrawn with the corrected numbers): `figures/switchboard_trials_per_session.png`.

### Task 3 (this round) — resolving the CSV-vs-manifest discrepancy (2026-09-01)

Compared CSV block-level trial counts against actual per-block HDF5 trial
counts (not just per-session totals, as the prior round's check did) —
265 (session, block, split) rows total, at the same granularity the CSV
itself reports (`scratch/investigate_csv_manifest_discrepancy.py`).

- **(a) Direction: goes both ways, not uniformly CSV ≥ actual.** 30 rows
  have CSV > actual (total excess +249), but 4 rows have actual > CSV
  (total excess −35), netting to the previously-found +214. The earlier
  "QC drops during packaging" hypothesis predicts only the first direction
  and is therefore incomplete as a full explanation.
- **(b) Concentration: roughly proportional to corpus/partition size, not
  disproportionately concentrated.** Switchboard accounts for 24/34
  mismatched rows and 174/214 of the total absolute discrepancy — but
  Switchboard is also ~69% of all trials, so this is close to its base
  rate, not special concentration. `train`-partition rows are mismatched
  somewhat more than `val`/`test` (27 vs. 7 rows), but again roughly
  tracks partition size.
- **(c) Shared property: mismatched blocks skew toward smaller blocks
  (mean 32.7 actual trials vs. 42.6 for matched blocks), and several
  mismatches are suspicious round numbers (several instances of exactly
  ±10, ±5) in BOTH directions.** A concrete worked example
  (`t15.2023.11.03`): the CSV lists block 6 as Switchboard with 50
  sentences; the actual HDF5 block 6 has 50 trials **total across val+test
  combined (25 in each)** — so the CSV's 50 is right for the *block*, but
  only 25 of those are in the *labeled* (val) partition, the other 25
  being in unlabeled test. A count computed straight from the CSV without
  partition-awareness silently mixes labeled and unlabeled trials. This
  specific mechanism doesn't explain every mismatched row (some really are
  short by more trials than test/val would account for), so packaging-time
  drops are still a plausible contributor to the remainder, but partition
  misattribution is now a *confirmed*, not just hypothesized, contributor.
- **(d) Authoritative source, and how the Switchboard analysis must handle
  this:** the **HDF5 files (and therefore the manifest) are authoritative
  for trial counts** — they are what will actually be loaded and trained
  or evaluated on. The CSV is authoritative for **block-to-corpus
  labeling** (every block has exactly one corpus label there, and no
  boundary/attribution problems were found in the corpus assignment
  itself — 0/265 blocks had a missing corpus lookup). The correct method,
  now used in Task 2's corrected table above, is: **join the CSV's
  per-block corpus label to the HDF5's actual per-block, per-partition
  trial count** (train+val only for "labeled"), rather than using either
  source's trial counts alone. Using the CSV's counts directly (as the
  prior round did, even with a session-level cap) understated the risk of
  partition misattribution and produced at least two wrong PASS/FAIL
  calls, corrected above.
