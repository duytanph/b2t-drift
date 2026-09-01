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

## 2. Three research questions

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

## 3. Split design and rationale

**Train** on sessions 1–15 (`t15.2023.08.11` through `t15.2023.10.15`),
**3,969 labeled trials** [verified against `data/interim/session_manifest.csv`:
`train` + `val` trials across those 15 sessions sum to exactly 3,969].

**Evaluate** each of the 30 subsequent sessions **individually**, producing a
PER-vs-days-elapsed curve rather than one pooled number.

**Held-out spacing is roughly logarithmic**: +5 to +546 days — dense in the
first two months, sparse in the tail. See `docs/DATASET.md` section 13 for
the full list of gaps.

**Why temporal, not random.** The question is extrapolation into the
future, not interpolation within a shuffled pool. Shuffling sessions across
time destroys the thing we're trying to measure — a randomly-held-out
session tells you nothing about what happens to a decoder that's aging.

**Why 15 sessions, not 5 or 25.** A tradeoff between baseline decoder
strength and number of held-out evaluation points. Fewer training sessions
means a weaker baseline and risks a floor effect masking drift; more
training sessions leaves fewer held-out days and less dynamic range on the
x-axis. Fifteen sessions gives ~42% of labeled data for training and still
leaves 30 evaluation points spanning 18 months.

**Bootstrap confidence intervals are mandatory, not optional.** Early
held-out sessions have as few as 80 trials. PER computed on 80 sentences is
noisy enough that random wiggle will look like structure without an
interval around it. Every point on the curve needs one.

**Why the official split cannot measure drift.** Within each session, `val`
and `test` are trial-level splits of the *same* held-out block(s) — not
independent generalization estimates — and every session (including the
most recent ones) contributes training data. A model trained on the
official split has seen data from every era of the recording, so it shows
near-zero drift by construction: there is no temporal extrapolation
anywhere in that design. It measures general decoding accuracy, which is
the right target for the Brain-to-Text '25 competition and the wrong one
for this project.

**Why the provided pretrained checkpoint cannot be used for the drift
curve.** `data/external/t15_pretrained_rnn_baseline/checkpoint/args.yaml`
lists all 45 sessions under `dataset.sessions` — confirmed by direct
inspection (Task 4b, 2026-09-01). It has seen every session in the dataset
and has no held-out future. It remains useful for (a) validating our
evaluation code against a known-good model, and (b) inspecting its 45
learned per-session day-layers as a record of how much per-session
correction the official baseline needed — see Task 4 findings below.

---

## 4. Pipeline constraints derived from data inspection

- **Lazy HDF5 loading.** The dataset is ~12 GB against a 16 GB system RAM
  budget. `T15Dataset` must not hold everything in memory; it has to read
  lazily from HDF5 per `__getitem__`.
- **Length-bucketed batching.** `n_time_steps` spans 138–1025+ bins, a
  7.4x range. Random batching wastes most of each batch on padding —
  batch by similar length instead.
- **Block-level normalization.** Array statistics shift within a single
  session, not just across sessions, so z-scoring must be computed per
  block, not per session. **Caveat as of Task 1 (2026-09-01):** the raw
  `input_features` values in the released HDF5 already appear to be
  z-scored (every sampled column has mean ≈ 0, std ≈ 1) — see Findings
  below. This needs reconciling before implementing our own normalization
  step: either the release already did block-level normalization upstream
  and "normalize per block" means "trust the released values," or there
  is a second, un-normalized feature source we haven't found. Do not
  implement `data/normalize.py` until this is resolved.

---

## 5. Metrics

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

## 6. Stated limitations

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

## 7. Findings from Tasks 1–4

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
  a fixed per-column affine transform before release. **This appears to
  contradict `docs/DATASET.md` section 9's framing** ("Left to us:
  Normalization (z-scoring). The baseline does this per block, not per
  session"), which implies the shipped `input_features` are raw
  counts/power. This needs resolving — either DATASET.md's section 9
  needs correcting, or there's a raw (non-normalized) feature source we
  have not found — before implementing `data/normalize.py`. Flagging here
  rather than editing DATASET.md directly, since that file is the
  project's verified-facts reference and this contradiction should be
  reviewed before it's changed.
- Figure: `figures/feature_layout_correlation.png`

### Task 2 — corpus composition confound check (2026-09-01)

- **Corpora found: 6, not 5.** Switchboard (7,672 trials), Freq words
  (2,050), 50-Word (1,091), Openwebtext (172), Harvard (98), Random (79).
  `docs/DATASET.md` section 10 names only five corpora (Switchboard,
  OpenWebText2, 50-word, frequent-word, random); **Harvard sentences are
  an additional corpus not mentioned there** and should be added.
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
  interpreted as a clean drift estimate.
- **Era concentration:** Freq words is confined to 2023-10-22 through
  2023-12-29 (11 sessions); Harvard to 2023-10-20 through 2023-11-03 (3
  sessions); Openwebtext to 2023-09-29 through 2023-10-15 (entirely
  *inside* the train window). Only Switchboard and 50-Word recur across
  the full 20-month span.
- **Required stratification:** the pooled PER-vs-days curve should not be
  reported as the headline result on its own. At minimum, report PER
  stratified by corpus (Switchboard vs. everything else), and treat the 8
  fully-non-Switchboard held-out sessions as measuring generalization to
  an unseen task, not drift, or exclude them from the primary drift curve
  and analyze them separately.
- **Cross-check vs. `data/interim/session_manifest.csv` (part e):** trial
  counts mostly agree but **24 of 86 (session, train-vs-val/test) cells
  disagree**, and the CSV's total (11,162 trials across all sessions) is
  214 higher than the manifest's (10,948). The direction is mostly
  CSV > manifest (more "intended" sentences listed than actual trials in
  the released HDF5), but three cells go the other way
  (`2023-12-17`, `2024-03-15`, `2024-06-14`), so it isn't a uniform
  offset — consistent with some trials being dropped during HDF5 packaging
  on a per-block basis (failed trials, QC exclusions), not a simple
  systematic recount. Noted as a real discrepancy; the cause is not
  determined from the data alone.
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
  curve (see Split design, section 3 above). It also lists
  `dataset_probability_val`, a 0/1 flag per session; the 4 sessions
  flagged `0` are exactly the 4 train-only sessions we already knew about
  (`t15.2023.08.11`, `t15.2024.03.03`, `t15.2024.04.25`,
  `t15.2024.04.28`) — independent cross-validation of that fact.
- Checkpoint's recorded `val_PER = 0.101` (10.1%), `val_loss = 22.45`, on
  the official split (i.e. not comparable to our held-out-session PER
  numbers once we have them, per the "two baselines are not comparable"
  note in `docs/DATASET.md` section 13).
- `use_amp: true` in `args.yaml` — the official baseline was trained with
  mixed precision, same choice our Task 3 benchmark evaluates.
