# M7 SENTINEL-Lite, architecture round v2 — code readings of the amendment text

Written Tue 10 Nov 2026 19:47 EST by the M7 workstream agent, before the lead's stamp, before any fit on a real row
and before any validation row was loaded by v2 code. Asked for by the lead (message of 19:38 EST, item 1).

**What this file is.** Where amendment v0.3 (`lite_arch_v2_amendment.md`, sha256 `2b369018…8a56`) or a ruling of
`reviews/arch_v2_20261110/M7_verify.json` left a detail open, the code had to pick one reading. Each such reading is
listed here with the file that holds it. Nothing below was chosen with a validation number: none exists.

**What this file is not.** It is not a deviation record. `lite_arch_v2_deviations.md` (sha256 `23e6fd87…642c`) is
not changed. Section 7 lists the four readings that, in my view, touch a stamped number or sentence; the lead decides
whether any of them becomes a deviation entry, which would then go in a new file.

Code state: `sentinel/models/litev2/{tabular,harness,nets,fit}.py`, `scripts/train/litev2_{comparators,train,
tabpfn_memtest}.py`. Tests (synthetic tensors only): `tests/test_litev2_{tabular,harness,nets}.py`.

## 1. Tabular models — `sentinel/models/litev2/tabular.py`

| # | Text | Reading in the code |
|---|---|---|
| T1 | S6 / anchor, turbidity | One pooled turbidity model over NTU, FNU and NTRU rows with two unit indicator columns; a unit is predicted by setting its indicator. TSS and SSC are separate models. A GBM never uses a censored row. |
| T2 | Anchors (§4.2) | Training rows: out-of-fold over five water-body folds. Evaluation rows: the model refitted on all v2 training rows. |
| T3 | S8, "the other targets" | A per-target model gets the out-of-fold S6 predictions of every other quantity (the other targets and NTRU) and of the four teacher means. The pooled turbidity model gets them for Secchi, TSS, SSC, chl-a and the four teacher means (the three turbidity units are its own targets). |
| T4 | S8, "chip-mean teacher μ" | The trust-weighted mean of the teacher cells that the student's KD loss uses (`lite.distill.load_kd_tensors`). Missing where the chip has no pairing, no trusted cell, or the pairing is masked by ruling F4 (M2 Test-S). |
| T5 | S9 (XGBoost AFT) | The listed settings; package defaults for everything else. Three seeds are kept although the default settings make them identical. Targets = the registered list (`lite/s9_targets`: secchi, tss, chla), from the count rule "≥ 20 censored training rows". |
| T6 | P2a, "water pixels" | The frozen `features.water_mask`. With fewer than five water pixels the statistics are taken over the whole chip (as the frozen chip features do). "Percent water" is the mask fraction. |
| T7 | S10 TabPFN (F2), row cap | All rows if ≤ cap. Otherwise whole water bodies in ascending order of sha256("SENTINEL-Lite-tabpfn\|" + water body) until the next one would pass the cap. Cap 10,000 unless the memory test lowers it (lead order item 6). Package defaults, uncensored rows, 3 seeds averaged, prediction in chunks of 2,000 rows. |
| T8 | S10 memory test | Per candidate cap (10,000 / 8,000 / 6,000), in a child process: the S10 fit of the target with the most uncensored training rows, seed 0, and the prediction of one 2,000-row chunk of *training* rows. Predictions are discarded; no metric. The first cap that completes is registered. Runs only after the stamp (lead, 19:38). |
| T9 | S11 GBM-stacked (DS-06), "pooled embedding" | The 48-channel decoder output of the config-3 network, pooled with the frozen centre-window water weights; seed-matched (seed s of S11 uses seed s of the network); 16 principal components, PCA fitted on v2 training rows only. The training-row embeddings are in-sample for the network (it was trained on those rows); this is in the comparator's favour and is stated with the result. |

## 2. Harness and registry — `sentinel/models/litev2/harness.py`

| # | Reading |
|---|---|
| H1 | Registry: workstream `M7-lite`; experiments `litev2_val_national` and `litev2_val_p2`; model `litev2_<name>`; fits are logged under `litev2_fit` with split `train` and metric `train_loss_final`. |
| H2 | National validation block (`val` ∪ `val_n`): the frozen primary metric and population for the four original targets. FNU and SSC on this block are descriptive (all classes, MdSA) and are not required by the gate. P2 block: FNU MdSA over all `val_p2` rows. Clusters are water bodies. A score on fewer than 10 rows is flagged as not interpretable (`lite_national_protocol.md` N2). |
| H3 | Comparators-first gate: 25 names — S1–S9, every P1 form, P2a, P2b, S10, S11, and four DenseNet names (P3a and P3b at each backbone learning rate). S9 is required only for its registered targets; a P1 form only for the targets it is defined for. The gate is called for both blocks before a candidate loads any validation row. |
| H4 | A name is evaluated on validation once: the evaluation stage refuses a second run of the same name. |
| H5 | A neural arm is scored as the 3-seed ensemble (mean μ; σ² = mean σ² + variance of μ). Per-seed scores go in the registry row's `extra.per_seed`. |
| H6 | Teacher means and KD tensors are built for training roles only. A value on an F4-masked pairing raises an error instead of being dropped silently. P2 chips have no teacher pairing and are never looked up. |

## 3. Networks — `sentinel/models/litev2/nets.py`

| # | Text | Reading in the code |
|---|---|---|
| N1 | "the leader's dense 4-target map head is kept as an auxiliary head" | The head reads the FiLM-modulated decoder output, the same map the pooling reads. At initialisation FiLM is the identity, so the head returns the leader's map exactly. |
| N2 | "restricted to dark-NIR water cells" | A stride-4 output cell is a water cell when at least half of its 4 × 4 pixels are in the frozen `features.water_mask`. |
| N3 | a·r + b·r², r in metres | a and b are stored per 100 m and per (100 m)², so one optimiser step moves the logits by a comparable amount. Same function family; initial value 0. |
| N4 | "The Δ output is zero-initialised" | Only the Δ row of each head's last layer is zeroed. The log σ² row keeps default weights and gets the §4.3 bias. |
| N5 | [s, c] and a | s = 50 scene statistics + GSD (51), c = 5; standardised with train-row statistics kept in buffers; a missing scene value is the train mean. The anchor a enters unstandardised (log10 units). |
| N6 | Image input | The leader's input statistics (`x_mean`, `x_std`) and its `y_mean`, `y_std` for the KD head are kept as loaded. |
| N7 | Measured parameter counts | ANCHOR-Lite 1,404,616 (leader 1,323,880 + 80,736 new); A1 1,404,232; A3 1,391,144; A4 1,383,620; A5 1,386,133; ANCHOR-Set 307,341; P3a 6,969,293; P3b 7,243,537. |

## 4. Training — `sentinel/models/litev2/fit.py`

| # | Text | Reading in the code |
|---|---|---|
| F1 | Σ_t w_t · (…) | The target terms are summed, not averaged over targets. |
| F2 | L_t | Mean over the batch rows that carry a term for target t: β-NLL of an uncensored value; Tobit term of a censored row × detached σ^(2β); for the NTU head, β-NLL of an NTRU value against μ + one learned offset. |
| F3 | "1e-3 for new modules, 5e-4 for backbone and decoder" | Encoder, decoder and the leader's map head train at 5e-4; every new module at 1e-3. Weight decay 1e-4, 2 warm-up epochs + cosine, gradient clip 5.0, AMP: as config 3. |
| F4 | Augmentation of the scene statistics | With f the per-chip gamma / gain / offset map applied to the pixels: a quantile q becomes f(q); the mean becomes f(mean); the std becomes (f(mean + std) − f(mean − std)) / 2. The noise and block-resampling steps leave them unchanged. Fractions, GSD and c are unchanged. |
| F5 | A1 | μ_t = train median + Δ_t; log σ² bias = log of the train variance; the 0.01 · mean(Δ²) term stays (the loss is "unchanged"). |
| F6 | DenseNet heads (P3a, P3b) | Not standardised; bias = train median and log train variance. Input statistics from the training chips. |
| F7 | P3b "auxiliary chip-level KD head on the teacher μ, weight 0.3" | 0.5 · squared error on the chip-mean teacher μ (T4), standardised with train-row statistics; F4-masked. |
| F8 | ANCHOR-Set | No radiometric augmentation (§4.4 names only the pixel resampling). Schedule as the other arms. Pixel logarithms are log10(DN + 1). Pixel features standardised with training-chip statistics. Shore distance = Euclidean distance transform in the frozen water mask. Evaluation sample seed = sha256("SENTINEL-Lite-set\|" + uid). Water pixels are drawn first at random; the fill to 64 is by distance to the chip centre. |
| F9 | A4 | Uncensored labels of the four original targets only (no censored row, no NTRU row). Anchors refitted on those reduced rows. FNU is scored through the NTU head. The KD term is kept. |
| F10 | No-KD twins (F5 of the verifier) | Same arm with KD weight 0, started from the no-KD leader, seed-matched; run name suffix `_nokd`. |

## 5. Run script — `scripts/train/litev2_train.py`

| # | Reading |
|---|---|
| R1 | Two stages. `fit` reads training rows only and trains the fixed number of epochs; one checkpoint and one registry row per seed; a seed with a checkpoint is never retrained. `eval` loads the validation blocks once and writes the validation rows. Both need the stamp. |
| R2 | The gate of §5 is on the candidate's *validation evaluation*. A candidate `fit` reads no validation row and is not blocked by the gate in the code. Whether candidate fits may run beside the comparators is asked of the lead; until answered, comparators go first. |
| R3 | `--init-run` has no default: the leader run an arm starts from is always named on the command line and written in the run record. |
| R4 | DenseNet (ruling F8): both backbone learning rates × 3 seeds = 12 runs for P3a + P3b, one seed per job, names `<arm>_lr1e-4` and `<arm>_lr3e-4`. No learning rate is picked on validation. |
| R5 | One training run serves both tracks (national and P2). |

## 6. What has not been done

No fit on a real row. No validation row loaded by v2 code. No v2 registry row. The stamp guard still refuses
(`lite_arch_v2_lead_rulings.md` does not exist yet). The DenseNet121 weight file has not been loaded.

## 7. Readings that touch a stamped number or sentence — for the lead's decision

1. **ANCHOR-Set size (N7).** §4.4 says "About 0.15 M parameters". The architecture as written measures 307,341. The
   architecture was not changed to meet the estimate. It is far inside the 3 M budget.
2. **DenseNet compute (R4).** §5 and §10 of the amendment count 8 DenseNet runs (≈ 11 GPU-h). Ruling F8 makes it
   12 runs; my estimate is ≈ 14 GPU-h for them and ≈ 29 GPU-h for the whole round. The ceiling is an open point of
   the stamp package.
3. **Candidate fits and the gate (R2).** The stamped sentence covers validation evaluation only. The code follows
   the sentence; the stricter order (comparators first for fits too) is kept until the lead rules.
4. **A1 keeps the Δ² penalty (F5) and ANCHOR-Set has no radiometric augmentation (F8).** Both are readings of "same
   loss" and of §4.4's silence. Either could be read the other way; they are fixed here before any number exists.
