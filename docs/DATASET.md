# Brain-to-Text '25 / T15 Dataset — Reference Notes

*Compiled for the `b2t-drift` project. Facts marked **[verified]** were confirmed by direct inspection of the downloaded files; everything else comes from published documentation.*

---

## 1. Orientation — what this dataset is in one paragraph

This is intracortical neural data recorded from a single human participant with ALS while he attempted to speak prompted sentences, collected across 45 sessions spanning 20 months. Each trial pairs a time series of 512 neural features (sampled in 20 ms bins from 256 implanted electrodes) with the ground-truth sentence he was trying to say and its phoneme transcription. It is the largest publicly available intracortical speech dataset, and it is the substrate for the Brain-to-Text '25 competition. The reason it matters for a chronic-stability project specifically is the 20-month span — most BCI datasets cover weeks.

---

## 2. Provenance

**Who produced it:** The Neuroprosthetics Lab at UC Davis (Sergey Stavisky and David Brandman, co-senior authors), in collaboration with Stanford (Francis Willett, Jaimie Henderson) and Brown/MGH (Leigh Hochberg). Part of the BrainGate2 clinical trial lineage.

**Primary paper:** Card, N.S. *et al.* "An Accurate and Rapidly Calibrating Speech Neuroprosthesis." *New England Journal of Medicine* 391(7):609–618, August 15, 2024. DOI: 10.1056/NEJMoa2314132

**What they were trying to accomplish:** Prior speech BCIs had two problems that blocked real-world use — accuracy too low for reliable communication, and calibration requirements too heavy. Earlier work reported word error rates of 25.5% on a 1,024-word vocabulary and 23.8% on a 125,000-word vocabulary, and needed at least 13 days of recording (roughly 17 hours of training data) to reach those numbers. The stated goal here was high accuracy on a full-size vocabulary with drastically less calibration.

**Key findings:**

- On the very first day of use — 25 days after surgery — the system reached 99.6% accuracy on a 50-word vocabulary.
- Calibration took 30 minutes of recorded attempted speech, followed by offline processing.
- Word error rate fell to 2.5% (95% CI 1.0–4.5) by session 15, on the full 125,000-word vocabulary.
- Accuracy held at or near 97.5% through session 84, more than 8 months post-implant.
- Decoded text was spoken aloud by text-to-speech software built to approximate his pre-ALS voice.

**Why this matters for our project:** the paper's headline is that performance was *maintained* over months — but that was with an actively maintained, periodically recalibrated system. Our question is what happens to a decoder that is *frozen* on day one. Those are different questions, and the gap between them is roughly the value of the maintenance effort.

---

## 3. Participant and recording setup

**Participant:** A 45-year-old man with amyotrophic lateral sclerosis (ALS), with tetraparesis (weakness in all four limbs) and severe dysarthria (impaired speech articulation). Implanted five years after symptom onset. Referred to as T15.

Single participant. Everything in this dataset comes from one brain. Nothing here supports claims about across-subject generalization.

**Implant:** Four Utah-style high-density microelectrode arrays, 64 electrodes each = 256 total, placed in the left ventral precentral gyrus. Signals were acquired through two percutaneous connector pedestals using the Blackrock Neurotech NeuroPort system.

**The four array locations** (this is the anatomical breakdown that matters for interpreting the feature layout):

| Region | Common name | Role |
|---|---|---|
| **Ventral 6v** | Ventral premotor cortex | Speech motor planning / articulator control |
| **Area 4** | Primary motor cortex | Direct motor output |
| **55b** | Middle precentral gyrus | Speech-selective region, language-adjacent |
| **Dorsal 6v** | Dorsal premotor cortex | Motor planning |

All four sit in or adjacent to speech motor cortex. This is *attempted articulation*, not inner speech or semantic representation — the participant tries to produce the sentence and the arrays capture the motor commands.

---

## 4. What the signals are

**Signal class:** Intracortical multiunit activity. Not EEG, not ECoG. Penetrating microelectrodes recording from within cortical tissue, roughly 1–2 mm deep, each electrode sensing spiking from a small population of nearby neurons.

**Two feature types per electrode**, both derived from the same high-pass-filtered broadband signal:

**Threshold crossings (TX)** — a count of how many times the filtered voltage dips below −4.5 × RMS in each bin. An extracellular action potential appears as a sharp *negative* deflection (the spiking neuron acts as a current sink), which is why the threshold is negative. The −4.5 multiplier is a signal-detection convention: shallower thresholds count noise as spikes, deeper ones miss all but the largest units.

**Spike band power (SBP)** — the mean squared amplitude of the same filtered signal per bin. Continuous rather than discrete.

**Why both:** They fail differently. TX is high-signal-to-noise when units are well isolated, but reports zero once spike amplitudes shrink below threshold — which is exactly what happens as electrodes degrade through micromotion, glial encapsulation, and rising impedance. SBP still registers reduced energy in that regime. SBP also needs less bandwidth than raw voltage, which matters for wireless implant power budgets.

**Note on spike sorting:** No sorting into individual neurons. Unit identity is unstable across days as arrays shift microns in tissue, and sorting costs compute an implanted device doesn't have. Multiunit threshold crossings decode nearly as well as sorted units for BCI purposes.

---

## 5. Channel count, sampling, and binning

- **256 electrodes** → **512 features** (TX and SBP per electrode)
- **Raw acquisition:** 30 kHz broadband, high-pass filtered (roughly >250–300 Hz) to isolate the spike band
- **Released resolution:** 20 ms bins (50 Hz effective sampling rate)
- **Why 20 ms:** Speech articulatory gestures last roughly 50–150 ms, so bins need to be well below that to resolve them. But shorter bins mean fewer spikes per bin and higher count variance. 20 ms is the compromise, and it's what the authors released, which is why the field standardized on it.
- **Real-time system output:** one phoneme every 80 ms — meaning the deployed model downsamples 4× internally relative to the input bin rate. Useful to know when sizing your own architecture.

**We never see raw 30 kHz voltage.** The dataset ships pre-binned. This is a large practical advantage — no filtering, spike detection, or artifact rejection to implement — but it also means feature-extraction choices are fixed and not something we can ablate.

---

## 6. How the data is structured

### Directory layout

```
data/raw/
├── t15_copyTask.pkl              # online decoder results (Figure 2 of paper)
├── t15_personalUse.pkl           # conversation-mode data (Figure 4)
└── hdf5_data_final/
    ├── t15.2023.08.11/
    │   └── data_train.hdf5
    ├── t15.2023.08.13/
    │   ├── data_train.hdf5
    │   ├── data_val.hdf5
    │   └── data_test.hdf5
    └── ...                        # 45 session folders total
```

Session folders are named `t15.YYYY.MM.DD`, so **dates are parseable directly from folder names** — no metadata lookup needed to build temporal splits.

### Hierarchy

`session → block → trial`

A **block** is a contiguous recording run within a session, typically tens of trials. A **trial** is one sentence attempt. Blocks matter more than they look: array statistics shift measurably *within* a single day, which is why the official pipeline normalizes per block rather than per session.

### Per-trial contents **[verified]**

**Datasets:**

| Name | Shape | Type | Notes |
|---|---|---|---|
| `input_features` | (n_time_steps, 512) | float32 | **Variable length**, not padded |
| `seq_class_ids` | (500,) | int32 | Phoneme IDs, **padded with 0** |
| `transcription` | (500,) | int32 | ASCII codes, null-terminated |

**HDF5 attributes** (these are attributes, *not* datasets — easy to miss):

| Name | Example | Notes |
|---|---|---|
| `n_time_steps` | 1001 | True length of `input_features` |
| `seq_len` | 21 | True length of `seq_class_ids` |
| `sentence_label` | `'Hope you are not hungry.'` | Ground truth text |
| `session` | `t15.2023.08.11` | Session ID |
| `block_num` | 6 | Recording block |
| `trial_num` | 9 | Index **within file**, restarts at 0 per file |

**Trial group keys** are named `trial_NNNN` and are separate from the `trial_num` attribute. Don't treat them as interchangeable.

### Observed distributions **[verified]**

- `n_time_steps`: 138–1025+ bins → 2.8–20.5 seconds, median ~527 bins (~10.5 s)
- `seq_len`: 5–47 phonemes, median ~16
- Trials are far longer than the speech itself — each contains substantial pre- and post-speech buffer. Most bins are silence.
- Input:target length ratio is roughly 33:1 at the median, so CTC's constraint (input ≥ target) is satisfied with enormous margin.

---

## 7. Feature layout — which column is which electrode

The 512 columns are ordered by array and feature type:

| Columns | Array | Feature |
|---|---|---|
| 0–63 | Ventral 6v | Threshold crossings |
| 64–127 | Area 4 | Threshold crossings |
| 128–191 | 55b | Threshold crossings |
| 192–255 | Dorsal 6v | Threshold crossings |
| 256–319 | Ventral 6v | Spike band power |
| 320–383 | Area 4 | Spike band power |
| 384–447 | 55b | Spike band power |
| 448–511 | Dorsal 6v | Spike band power |

> ⚠️ **The official README states these boundaries as 0–64, 65–128, 129–192, etc.**, which is internally inconsistent (0–64 is 65 values, not 64). The intent is clearly 64-electrode blocks as written above. **Verify empirically before relying on it** — a shifted feature index would silently corrupt any per-array analysis.

**Two consequences worth remembering:**

1. Column *i* (TX) and column *i + 256* (SBP) are the **same physical electrode**. Any channel-health analysis should treat them as a pair.
2. Array-level analysis is possible without extra metadata — you can ask whether 55b degraded faster than area 4, for instance, purely by slicing columns.

---

## 8. Label encoding **[verified]**

`seq_class_ids` uses 41 classes, indexed 0–40:

- **0 = CTC blank.** Never appears inside `seq_len`. Also used as the padding value, so blank and pad are indistinguishable from the array alone — `seq_len` is the only way to find the real end.
- **1–39 = the 39 standard ARPABET phonemes**, alphabetical: AA, AE, AH, AO, AW, AY, B, CH, D, DH, EH, ER, EY, F, G, HH, IH, IY, JH, K, L, M, N, NG, OW, OY, P, R, S, SH, T, TH, UH, UW, V, W, Y, Z, ZH
- **40 = silence token** (`' | '`, with surrounding spaces). Not a phoneme.

**CTC output layer size is 41, not 40.** Sizing it 40 produces a model that trains, converges to something plausible, and is quietly wrong.

The `transcription` array decodes to exactly the same string as `sentence_label` (verified 200/200 sampled trials), so it's **redundant — don't load it**.

Source of the mapping: the Brain-to-Text '25 Kaggle data page, as `LOGIT_TO_PHONEME`. It is not in the HDF5 files.

---

## 9. Preprocessing already done upstream

Applied by the authors before release:

1. Broadband 30 kHz acquisition
2. High-pass filtering to isolate the spike band
3. Threshold crossing detection at −4.5 × RMS
4. Spike band power computation
5. Binning to 20 ms
6. De-identification (the Dryad deposit states it contains no personal or identifiable information)
7. **Normalization (z-scoring). [verified]** Corrected from an earlier draft
   of this document, which claimed normalization was left to us. It is not.
   Direct inspection of raw `input_features` values (`scratch/investigate_feature_layout.py`,
   `scratch/investigate_normalization_unit.py`, 2026-09-01) shows every
   sampled column already has mean ≈ 0, std ≈ 1 — the released values are
   already z-scored, not raw counts/power. Confirmed independently by
   reading the official repo's `model_training/dataset.py` and
   `model_training/rnn_model.py`: neither applies any further
   normalization, z-scoring, or standardization at load time or in the
   model — `input_features` is used exactly as loaded from HDF5, and the
   only per-input transform in the model (`day_weights`/`day_biases`) is a
   *learned*, per-session affine layer initialized to the identity matrix
   and zero bias, not a fixed normalization step.
   **The normalization *unit* (global / per-session / per-block) is not
   fully resolved.** Recomputing mean/std over an exact candidate unit's
   own data must reproduce ≈0/≈1 to within floating-point precision if
   that unit is what was actually normalized against; that only cleanly
   rules things out, it doesn't cleanly confirm the true unit:
     - **Per-block: ruled out with high confidence.** Recomputed per-block
       column means/stds scatter with a 5th–95th-percentile spread of
       ~0.06 — far too large to be exact-block normalization recomputed on
       itself (that would show ~1e-6 spread).
     - **Per-session: not confirmed, evidence favors it only moderately.**
       Recomputed per-session statistics (even pooling train+val+test) are
       tighter (~0.02 spread) than block-level but still not near machine
       precision, so exact per-session normalization is not fully
       confirmed either.
     - **Global (one fixed mean/std for the whole 20-month dataset):** the
       clearest prediction of this hypothesis — visible drift in per-block
       column means over the 20-month timeline — was **not observed**
       (`figures/normalization_unit_check.png`: values stay within a
       roughly ±0.08 band across the entire timeline with no monotonic
       trend).
   **Net: block-level is rejected; session vs. global is not distinguished
   by the evidence collected so far.** Do not assume block-level
   normalization when implementing `data/normalize.py` — if further
   normalization is applied at all, treat the existing values as already
   adequately normalized rather than re-deriving per-block statistics from
   scratch, pending a follow-up investigation.

**Left to us:**

- Any temporal smoothing or downsampling
- Batching and padding strategy
- Data augmentation, if any

---

## 10. What the participant was actually saying — and the confound this creates

Sentences were drawn from **six** different corpora [verified against `data/external/t15_copyTaskData_description.csv`, 2026-09-01]: **Switchboard** (conversational telephone speech, 7,672 trials — the large majority), **Freq words** (a frequent-word corpus, 2,050), **50-Word** (1,091), **Openwebtext** (172), **Harvard** (Harvard sentences, a standard phonetically-balanced speech corpus, 98), and **Random** (random word sequences, 79). Blocks differ in which corpus they used.

> Earlier drafts of this document (and the official repo's own README) name only five corpora, omitting **Harvard**. It is a small corpus (98 trials, confined to `2023-10-20` through `2023-11-03`) but it is present in the released description CSV and should not be forgotten when stratifying by corpus.

Additionally — and this is the important part — **most data is attempted *vocalized* speech, but some is attempted *silent* speech**, and the authors deliberately **do not publish which blocks are which**.

> 🚨 **This is a genuine threat to the validity of a drift analysis.** If corpus composition or speaking strategy varies systematically over the 20 months, then apparent "degradation over days" could partly be distribution shift in the *task*, not degradation of the *neural signal*. A decoder trained mostly on 50-word-corpus blocks will do worse on Switchboard blocks regardless of electrode health.
>
> **Mitigation:** the official repo ships `data/t15_copyTaskData_description.csv`, giving block-by-block session date, block number, trial count, corpus, and split assignment. Download it. Before drawing any conclusion from the drift curve, check whether corpus composition is stable across the timeline — and if it isn't, stratify by corpus or restrict the analysis to a consistent subset.
>
> Speaking strategy (vocalized vs. silent) cannot be controlled for, since it isn't released. Note it as a stated limitation.

---

## 11. Verified inventory

| Quantity | Value |
|---|---|
| Sessions | 45 |
| Date range | 2023-08-11 → 2025-04-13 (611 days) |
| Total trials | 10,948 |
| — train | 8,072 |
| — val | 1,426 |
| — test | 1,450 (**unlabeled**) |
| **Usable labeled trials** | **9,498** |
| Total labeled speech | 2,792 minutes (~46.5 hours) |
| Sessions with only `data_train.hdf5` | 4 (2023.08.11, 2024.03.03, 2024.04.25, 2024.04.28) |
| Approximate on-disk size | ~12 GB |

**Recording gaps:**
- ~58 days: 2023-12-29 → 2024-02-25
- ~166 days (5.5 months): 2024-07-28 → 2025-01-10

**Session density is very uneven:** 27 sessions in Aug–Dec 2023 (roughly weekly), 12 across Feb–Jul 2024, 6 in Jan–Apr 2025.

---

## 12. The official split, and why we can't use it

**How the official split works [verified]:** Within each session, certain blocks are held out. In all 41 sessions that have three files, `data_val.hdf5` and `data_test.hdf5` draw from an **identical set of blocks**, with no overlap with train. So val and test are trial-level partitions of the *same* held-out recording blocks.

**Two problems for us:**

1. **Val and test are not independent.** Any block-specific artifact appears in both. Treating them as separate generalization estimates overstates confidence.
2. **Held-out blocks are not the last blocks of a session.** In one checked session, train used blocks 1–7, 11, 12 while val/test used 8–9. The split therefore *interpolates* in time even within a single day — and across the dataset, it interpolates across the whole 20 months, since every session contributes training data.

**Consequence:** a model trained on the official split has seen data from every era of the recording. Evaluating it shows near-zero drift, because there is no temporal extrapolation anywhere in the design. The official split measures general decoding accuracy, which is the right target for the competition and the wrong one for us.

The Kaggle terms explicitly permit repartitioning the train and validation data.

---

## 13. Our split design

**Training block:** sessions 1–15 (2023-08-11 → 2023-10-15), **3,969 labeled trials**, about two months of calibration data.

**Evaluation:** each of the 30 subsequent sessions scored *individually*, giving a PER-versus-days-elapsed curve.

**Held-out day spacing:** +5, +7, +19, +20, +33, +35, +42, +49, +54, +56, +63, +75, +133, +140, +145, +152, +154, +193, +196, +208, +243, +278, +280, +287, +453, +455, +516, +518, +532, +546 days.

### Reasoning behind these choices

**Why train on the first 15 sessions rather than a random sample.** The scientific question is temporal *extrapolation* — what happens to a decoder as its calibration data recedes into the past. Any shuffling across time destroys the question. Sessions 1–15 also mirror the real deployment situation: you calibrate at the start and then live with what you have.

**Why 15 and not 5 or 25.** A tradeoff. Fewer training sessions means a weaker baseline decoder, and a floor effect could mask drift. More training sessions means fewer held-out days and less dynamic range on the x-axis. Fifteen gives ~42% of the labeled data for training and still leaves 30 evaluation points spanning 18 months.

**Why the spacing is good.** It's roughly logarithmic — dense in the first two months, sparse in the tail. That is exactly the sampling you want for a curve expected to change fast early and slowly later.

**Why per-session evaluation rather than pooling.** Pooling destroys the x-axis. The curve *is* the result.

**Why bootstrap confidence intervals are mandatory.** Early held-out sessions are small — 107, 167, 95, 80 trials. PER computed on 80 sentences is noisy enough that random wiggle will look like structure. Every point on the curve needs an interval.

### Things this design cannot do

- **Test partitions are unusable.** Unlabeled, so neither trainable nor scorable. 1,450 trials (13% of the dataset) are simply out of reach.
- **The two comparison baselines are not comparable to each other.** A model trained on the official split (8,072 trials) will outperform one trained on sessions 1–15 (3,969 trials). Report both; never compare across them.
- **The provided pretrained checkpoint cannot be used for the drift curve.** It was trained on train partitions from all 45 sessions, so it has seen every session in the dataset and has no held-out future. It is still useful for (a) validating our evaluation code against a known-good model and (b) inspecting its learned per-day input layers as a record of how much per-session correction was needed.

---

## 14. Traps and gotchas

- **`ctc_output_size` is 41.** Blank + 39 phonemes + silence.
- **Always slice `seq_class_ids` with `seq_len`.** Blank and padding are both 0; the array alone cannot tell you where the target ends.
- **Attributes, not datasets.** `n_time_steps` and `seq_len` live in `.attrs`. Listing `f[trial].keys()` will not show them.
- **`trial_num` is per-file and restarts at 0.** Not a global identifier.
- **Data will not fit in RAM.** ~12 GB against 16 GB system memory. The Dataset class must read lazily from HDF5.
- **Length spread is 7.4×.** Random batching wastes most of each batch on padding. Use length-bucketed batching.
- **Normalize per block, not per session.**
- **Feature index boundaries in the official README are off by one.** Verify empirically.
- **Silence (class 40) will dominate any confusion matrix.** Most bins in a 10-second trial are silence. Exclude or plot separately.
- **Offline metrics overstate closed-loop performance.** In real use the participant adapts to the decoder's behavior. Nothing in this dataset can capture that.

---

## 15. Language model context (for the WER question)

The full published pipeline goes: neural features → GRU + CTC → phoneme sequence → n-gram language model → OPT-6.7B rescoring → final text.

**PER is measured before the language model; WER after.** This matters because a stronger LM improves WER without touching the decoder at all. PER is the honest measure of neural decoding quality.

**Resource requirements for the LM stage:** the 3-gram model needs roughly 60 GB of RAM, the 5-gram roughly 300 GB, and OPT-6.7B needs about 12.4 GB of VRAM. A pretrained 1-gram model ships in the repo. The Kaldi-based implementation requires a separate conda environment with a different torch version, and building it needs CMake and gcc on Linux.

**Practical implication:** on a 16 GB Windows machine, PER is achievable and WER is not, short of a heavily pruned n-gram or renting a large-memory instance. Report PER as the primary metric and state the limitation explicitly.

---

## 16. Links

- **Paper (NEJM):** https://doi.org/10.1056/NEJMoa2314132
- **Preprint (medRxiv, open access):** https://doi.org/10.1101/2023.12.26.23300110
- **Data (Dryad):** https://doi.org/10.5061/dryad.dncjsxm85
- **Official code:** https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text
- **Competition + data documentation:** https://www.kaggle.com/competitions/brain-to-text-25
- **Pretrained 3-gram / 5-gram LMs:** https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq

**Related prior work worth reading:** Willett, F.R. *et al.* "A high-performance speech neuroprosthesis." *Nature* 620:1031–1036 (2023). This is the T12 dataset — same task, different participant, 128 electrodes, ~4-month span. Useful as the immediate predecessor and for comparing what changed.

---

## 17. Open questions

- [x] Verify the 512-feature column boundaries empirically (official docs are off by one) — confirmed 64-wide blocks; see `docs/DESIGN.md` section 7, Task 1.
- [x] Download `t15_copyTaskData_description.csv` and check corpus composition across the 20-month timeline — done; serious confound found, see `docs/DESIGN.md` section 7, Task 2.
- [x] Confirm whether the pretrained checkpoint can be loaded and evaluated with our own code — loaded and inspected (not yet run for inference); see `docs/DESIGN.md` section 7, Task 4b.
- [x] Determine typical block sizes and whether block count per session varies systematically over time — done; see `docs/DESIGN.md` section 7, Task 4a.
- [ ] Resolve the normalization unit (per-session vs. global) — block-level is ruled out but session vs. global is not distinguished by the evidence so far; see section 9 above and `docs/DESIGN.md` section 7.
- [ ] Determine Switchboard-only viability for the drift curve as a mitigation for the corpus confound — investigated (19/30 held-out sessions survive a 50-trial threshold, full day-offset range preserved), decision on whether to adopt it is still open; see `docs/DESIGN.md` section 7.
