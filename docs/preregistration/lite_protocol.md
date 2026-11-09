# SENTINEL-Lite (M7) — Pre-registration protocol

**Status:** v1.3, 2026-11-08 (v1.2 2026-10-24). The split (§3) was frozen and hashed **before any model was fitted or selected**
(splits v1 and v2 were superseded before any non-smoke fit, after the pre-training review; §11).
The test roles are scored once, after `docs/preregistration/FREEZE_lite.json` is written (§8); that file does
not exist yet. This file is part of the hashed scoring code (`sentinel/models/lite/guard.py::frozen_files`), so
any edit after the freeze invalidates the freeze. Changes before the freeze are listed in §11 with dates.

**Authors:** Bryan Cheng (AI-assisted). **Implements:** RESEARCH_PLAN.md §10 (G6), §14, §15.2 H-G6, §15.3 Lite soft thresholds.

## 1. Objective, scope and hypotheses
We test whether a miniature 4-band (R, G, B, NIR) student network can estimate optical water-quality variables on
**water bodies that were never seen in training**, scored against **in-situ measurements** (never against the
teacher), and whether distillation from the M2 SENTINEL-Orbit Sentinel-2 teacher (with red-edge) helps.

**Scope rule (RESEARCH_PLAN §10).** Concentration/depth regression only for Secchi depth, turbidity and TSS.
Chlorophyll-a is reported **only as trophic classes** (Carlson 1977 TSI(chl) edges 40/50/70 → chl-a 2.6, 7.3, 56 µg/L;
classes oligo/meso/eutro/hypereutrophic). Not predicted: phycocyanin, CDOM, DO, pH, conductivity, TN/TP, microcystin.

- **H1 (primary, H-G6).** On Test-A, the student's MdSA is lower than that of (a) the RGBN band-ratio baseline and
  (b) the GBM-on-RGBN-statistics baseline for **Secchi**, and for **turbidity or TSS** (both reported), with paired
  water-body-bootstrap 95% CIs of the MdSA difference excluding 0 and the one-sided ("student better") bootstrap
  p-value rejected after Holm correction across the 6 comparisons (3 targets × 2 baselines). A comparison where the
  baseline is better never counts as a rejection.
- **H2 (primary, H-G6).** Distillation gain: the distilled student's MdSA is lower than the same student trained
  without the teacher (same seeds, same data) for Secchi and turbidity/TSS (paired CI excluding 0, Holm over 3).
  If the M2 teacher checkpoint is unavailable, H2 is reported as *not tested*.
- **Secondary (descriptive).** Red-edge gap (M2 teacher on S2 vs student on NAIP, same chips with an S2 chip ≤ ±1 d);
  chl-a trophic-class accuracy vs the band-ratio class baseline (soft threshold §15.3); robustness (§7);
  Test-B (future NAIP cycle at seen-region water bodies); lake/reservoir-only subset; GLORIA-pretraining ablation;
  RGB-only vs RGBN ablation; INT8 vs FP32 Δ.

## 2. Data and labels
- **Imagery.** NAIP 4-band DOQQ chips, 256 × 256 px at native GSD (0.3/0.6/1.0 m), uint8 DN, vendor colour-balanced
  per state-year, **not reflectance** (`processed/naip_chips/wqp`, manifest `naip_matchups`). Per-DOQQ statistics
  (`naip_scene_stats.parquet`) provide the per-scene normalisation (§5).
- **In situ.** WQP surface samples within ±1 day of the NAIP acquisition (MN, FL, WI, MI; 2010–2023), JRC water test
  passed (`docs/datasheets/naip_matchups.md`).
- **Label rules** (`sentinel/models/lite/data.py`): units converted only by exact factors (Secchi m/cm/ft → m;
  TSS mg/L, ppm; chl-a µg/L, mg/m³, ppb). **Turbidity = NTU only** (FNU/NTRU/FNRU/FNMU/JTU are other variables and
  are dropped; 271 results). Plausibility bounds: Secchi 0.05–30 m, turbidity 0.05–1000 NTU, TSS 0.1–1000 mg/L,
  chl-a 0.05–1000 µg/L. Censored and missing values dropped (counted). SSC is not TSS and is dropped.
  One label per (chip, target) = median over sample dates within ±1 d; targets modelled as log10.
- **Counts** (from `lite_split_report.json`): 10,672 matched samples → 10,113 usable → 5,998 chips at 4,228 sites,
  1,616 water bodies (3,015 sites inside a HydroLAKES lake). Censored results (46 Secchi, 1 turbidity, 8 TSS, 17 chl-a; all have no value in this source)
  are dropped and counted. Right-censored Secchi (disk on the bottom) are the clearest waters, so the clear end of the
  Secchi distribution is truncated; stated as a limitation.
- **Label hash.** The label table's content hash (`splits.labels_sha256`, over chip ids, groups and log10 labels) is
  part of the split and of its sha256; `data.role_frame` refuses a label table that differs.
- **Water-body proxy** (`data.water_body_groups`; no lake polygons are available locally). Union of three rules:
  (a) sites ≤ 2 km apart; (b) same normalised name (`data.water_body_name`: e.g. "Lake Winnebago - G21" →
  "winnebago", "Mississippi R off Boom Island" → "mississippi river", station-code prefixes "JUP122" → "jup")
  within 40 km; (c) river/estuary/canal/other sites in the same HUC8 (flow-connected stations together, R4);
  (d) the same HydroLAKES v1.0 lake (polygon containing the site, else nearest ≤ 100 m; lakes ≥ 10 ha; the same
  rule as M2 Orbit; `processed/hydrolakes`, CC BY 4.0). False merges only make groups larger (conservative).

## 3. Split (frozen 2026-10-24, before model selection)
`docs/preregistration/lite_split.json` (version 2 format, third build), canonical sha256 (`splits.split_sha256`,
includes the labels hash): **`46ed6e1f8724a2ef4aded2f4251c151d322a920d3c1df0d90874bb81a3397475`** (pinned in
`guard.FROZEN_SPLIT_SHA256`; `load_split` refuses any other split). Labels sha256
`bb3e6ec90ecd2976699308dd00c71db65c8f29e1ae0528ce1faab216f5bfd72a`.

- Water bodies are assigned to the 25 × 25 km block of their site centroid (`sentinel.eval.splits.spatial_blocks`);
  blocks are randomly assigned to test (20%) / val (25%) / train, weighted by chip count, seed 20261024.
- **Year cut:** train and val use NAIP years ≤ 2021. Test-A (`test`) = test-block water bodies, **all years**
  (headline: unseen water bodies). Test-B (`test_time`) = train/val-block water bodies in NAIP 2022–2023.
  Unseen-water-body × future-year (test ∩ 2022–2023) is reported as a secondary slice.
- **Buffers:** 2 km (train/val water bodies with a site ≤ 2 km from a test site, and train water bodies ≤ 2 km from
  a val site, are `excluded`). Because water bodies are already 2-km-linked, this excludes nothing; distinct water
  bodies in adjacent blocks can be 2–5 km apart. A 5-km buffer was tried on label counts and excluded 20–24% of
  chips (47% of turbidity labels), so it was dropped (§11). **Pre-registered secondary slice:** Test-A restricted to
  chips > 10 km from every train site. Checked in code (`splits.check_split`): labels hash; water bodies and sites
  disjoint across train/val/test; years; distances (measured train→test min 2.05 km, median 13.8 km; val→train
  min 2.12 km, median 10.9 km). No HydroLAKES lake appears in more than one of train/val/test.

| role | chips | water bodies | Secchi (lake/res.) | turbidity (lake) | TSS (lake) | chl-a (lake) |
|---|---|---|---|---|---|---|
| train | 2,795 | 725 | 2,526 (2,069) | 575 (180) | 544 (178) | 994 (617) |
| val | 1,278 | 378 | 1,163 (1,083) | 132 (50) | 161 (57) | 428 (329) |
| test (A) | 1,178 | 420 | 1,094 (1,065) | 92 (58) | 98 (62) | 310 (271) |
| test_time (B) | 747 | 327 | 602 (480) | 190 (51) | 166 (43) | 361 (160) |

**Primary populations (review W12).** Secchi: lakes/reservoirs (as RESEARCH_PLAN H-G6 "held-out lakes"); all site
classes reported as secondary. Turbidity and TSS: **all water bodies** (lakes, reservoirs, rivers, estuaries) —
a stated deviation from the plan's "held-out lakes" wording (**Deviation D1**, §11), because lake-only Test-A counts
(turbidity 58, TSS 62 lake/reservoir rows; table above) give no usable power (test turbidity 92 / TSS 98 even over all classes — H1 for these targets is low-powered; stated
in advance); lake-only results are reported as secondary. Chl-a classes: lakes/reservoirs. Results are also
reported by site class. Test-A contains more estuarine/Florida chips than train (distribution shift; reported).

## 4. Metrics and statistics
- Regression (log10 space): **MdSA (%) primary**; SSPB, log-RMSE, log-R², log-slope; for models with σ, 80% interval
  coverage. Classes (chl-a): **balanced accuracy primary**; macro-F1, quadratic-weighted κ.
- CIs: cluster bootstrap over **water bodies**, B = 2,000 (`sentinel.eval.bootstrap`); paired comparisons on the
  identical chips; Holm across the primary family. Seeds: ≥ 3 per student configuration; the reported student
  prediction is the mean over seeds of μ (ensemble), with per-seed metrics and their spread also reported.
- Only chips where every compared model has a prediction are scored (all baselines predict every chip).

## 5. Models
**Baselines** (all fitted on train only, identical chips; `sentinel/models/lite/baselines.py`):
1. *Location/season-only:* (i) location kNN — median of the k = 10 nearest train sites' labels; (ii) GBM on lat, lon,
   day-of-year.
2. *RGBN band-ratio/index regressions* (OLS on train; band values = centre 9 × 9 median DN or the median over
   dark-NIR water pixels in the ~30 m window, each divided by the DOQQ band median). Forms (`baselines.BAND_RATIO_FORMS`):
   Secchi `log SD ~ B/R + B` (Kloiber et al. 2002), its log form `~ log(B/R) + log B`, and the log form on water
   pixels; turbidity/TSS `log y ~ log R + log NIR` (Nechad/Dogliotti single band + NIR) on centre/water pixels and
   `~ log R + log(R/G)` on water pixels; chl-a `log chl ~ log(B/G) + NDVI` (centre/water) → class by thresholds.
   **The best form per target on val is the comparator** (selection in the baseline's favour; review W3).
3. *GBM on RGBN statistics:* LightGBM (fixed hyper-parameters: 400 trees, lr 0.03, 15 leaves, min 20 per leaf,
   subsample 0.8, colsample 0.8, 6 threads; mean of 3 seeds, matching the 3-seed student ensemble) on centre 9 × 9 / 31 × 31 medians and dark-NIR water-pixel percentiles
   of each band (raw and scene-normalised), band ratios, dark-NIR fractions, GSD. Variant +context adds lat, lon, DOY.
4. *Student without distillation* (H2 comparator).
5. *M2 teacher* (S2 with red-edge) on chips with an S2 chip ≤ ±1 d — when the M2 checkpoint exists. The teacher
   must not have trained on Lite `test`/`test_time` water bodies (checked with `assert_disjoint` on site ids and
   coordinates ≤ 2 km when teacher targets are generated, review W8); otherwise the red-edge gap is not reported.
   Teacher targets are only ever computed on Lite train-role chips.
   **W8 result (2026-11-08, v1.3):** the stock M2 teacher (`docs/model_cards/orbit_teacher.json`, `satwq_base5`) fails
   this check: its fit rows (AquaCorr matchups incl. inner early-stopping rows; GLORIA pretraining rows) overlap
   131 / 420 Test-A, 105 / 327 Test-B and 106 / 378 val water bodies (same WQP site id, ≤ 2 km, or same HydroLAKES
   lake; `distill.flag_overlap`). It is therefore **not used** for distillation or the red-edge gap. The KD teacher
   is a *Lite-safe* retrain of the identical M2 recipe (code, config `configs/orbit/satwq.yaml`, fit scheme `base`,
   seeds 0–4) with every overlapping row removed (2,394 / 9,149 matchup rows, 27 / 2,631 GLORIA rows;
   `scripts/train/lite_teacher_litesafe.py`, `distill.litesafe_drop_mask`); val is removed too so KD gains on
   val are not inflated. `lite_teacher_targets.py` refuses any teacher whose seen rows overlap a test/test_time
   water body. Limitation: river reaches > 2 km from every Lite site are not matched (no HUC8 on teacher rows).

**Student** (`sentinel/models/lite/model.py`): timm backbone (MobileNetV4-Conv-Small primary; EfficientNet-Lite0
alternative), 4-channel stem, ImageNet init (in_chans = 4 adaptation), light FPN decoder (48 ch, stride 4) with
per-pixel μ and log σ² for 4 targets. Chip-level prediction = water-weighted mean of the per-pixel μ over the centre
window (~30 m), water weights from a dark-NIR (NIR < G) mask. Losses: Gaussian NLL on in-situ labels (direct
supervision) + optional distillation NLL of S2-cell block means against teacher (μ_T, σ_T² + σ_scale²) weighted by
teacher trust (`losses.py`). Radiometric augmentation (per-band gain ±10%, offset, gamma, noise, flips/rot90).
Optional GLORIA band-simulation pretraining of the spectral branch (`gloria.py`). Budget: ≤ 3 M params,
≤ 2 GMACs / 512² tile (`torch.utils.flop_counter`, conv/matmul only), INT8 ≤ 4 MB, latency ≤ 100 ms per tile.

**Model selection** uses val only, with one pre-specified score (review W4): S = mean over {Secchi, turbidity, TSS}
of (student val MdSA / GBM-RGBN val MdSA) on the primary populations; lower is better; ties (|ΔS| < 0.01) go to the
smaller model. Candidate changes: backbone, input normalisation (raw DN/255 vs scene-normalised), augmentation,
GLORIA pretraining, multi-scale context, epochs/lr, distillation weight, ensembling. Following the RESEARCH_PLAN
§15.5 amendment there is no fixed cap; **every configuration tried is logged (`TRAINING_LOG.md`, registry) and the
count is reported next to any claim**, and the val-selection optimism is stated. The val curve recorded every 5
epochs is diagnostic only (fixed epoch budget, no val early stopping).

## 6. Comparators and claims
H-G6 claim requires H1 (Secchi + turbidity-or-TSS vs both baselines) and H2. "First-in-class" wording follows
RESEARCH_PLAN §10 (*not* "first drone water-quality model") and requires a dated novelty search in
`docs/novelty_checks/`. Otherwise: "competitive" or "negative".

## 7. Robustness suite (val during development; test once at freeze)
Applied to input chips at inference (`augment.py::perturb`), MdSA degradation relative to clean:
sun glint (additive specular blobs covering 10/25/40% of pixels), haze (blue-weighted additive offset 5/10/20 DN),
resolution (block-average down-sampling ×2/×4/×8, then nearest up-sampling), per-band calibration gain ±5%
(random sign per band, 5 fixed draws), NIR misregistration (1/3/10 px shift). Soft threshold: degradation ≤ 20% under
moderate glint (25%), haze (10 DN) and ±5% gain.

## 8. Freeze and single test scoring
`FREEZE_lite.json` = {created, split_sha256, scoring_code_sha256 (guard.scoring_code_sha256), weights_sha256
{model: sha256}, configurations_tried}. Test roles (`test`, `test_time`) are loaded only by
`scripts/eval/lite_score_test.py --allow-test` with a valid freeze; training code refuses them
(`guard.assert_no_protected`).

## 9. Export and latency
ONNX opset 17 → `onnxruntime.quantization.shape_inference.quant_pre_process` → static QDQ INT8 (per-channel weights)
calibrated on train chips. Latency: ORT CPU EP, 1 intra-op thread, batch 1, 512 × 512 × 4, 5 warm-ups, median/p90 of
≥ 30 runs, CPU model logged — **"desktop proxy", not device latency**. INT8 Δ MdSA vs FP32 reported on val.

**Deployable unit and freeze candidate (lead decision 2026-11-08, pre-declared before any test access).**
- The frozen deployable artefact is a single model: **seed 0** of the selected configuration. Seed 0 was declared in
  advance and is not the best seed on val.
- The budgets (§5: ≤ 3 M params, ≤ 2 GMACs/512², INT8 ≤ 4 MB, proxy latency ≤ 100 ms) apply to this model. The
  3-seed ensemble (3.4 GMACs, 4.9 MB INT8) exceeds them.
- At the single test scoring, all 3 seeds are scored individually and as the §4 ensemble. The 3-seed spread is
  reported, and the RESEARCH_PLAN §15.1 rule (a gain must hold in each of ≥ 3 seeds) is applied across the 3 seeds.
- Robustness at the freeze is reported for the seed-0 model and the ensemble.

**Glint augmentation (lead decision 2026-11-08, after the val glint failure).** One pre-declared configuration,
`mnv4s_raw_crop128_kd03_glint`: the selected configuration plus training-time glint blobs (`augment.glint_aug`,
p = 0.5 per chip, coverage U(2%, 40%), amplitude U(40, 150) DN). It replaces the selected configuration only if its
val S ≤ S(selected) + 0.01 (the §5 tie margin). The augmentation uses the same blob family as the §7 glint
perturbation, so glint robustness of an augmented model is optimistic, and this is stated with any result. No glint
mask input is added: it would change the export interface.
- **Result (2026-11-08, val):** S = 0.898 vs 0.868, so the configuration is **not kept** and kd03 stays selected.
  Glint at 25% degrades Secchi by +0.3% (kd03: +124%). This is optimistic, same blob family. The glint failure is a
  stated limitation of the selected model.
- **Secondary robustness variant (pre-declared 2026-11-08, before any further selection).** Seed 0 of
  `mnv4s_raw_crop128_kd03_glint_20261108-201455` is scored once at the single test scoring, alongside the primary,
  on clean Test-A/Test-B and on the §7 robustness grid.
  - It is reported only as a secondary robustness comparison: robust variant vs primary, paired, with the
    same-family caveat.
  - It is outside the H1/H2 Holm families and never replaces the primary.

## 10. Known limitations (pre-stated)
NAIP DN are not reflectance and differ from drone Rrs; the domain gap to the proposed drone camera is untested
until flights or NEON AOP simulation; the water-body proxy is distance-based; site-location error vs a 0.3–1 m pixel;
NAIP acquisition time-of-day unknown (±1 d window).

## 11. Change log
- 2026-10-24 v1.0: initial protocol; split v1 frozen (sha256 767571a9…) before any model fitting.
- 2026-10-24 v1.1 (before any non-smoke fit; only 2-epoch smoke runs on train/val subsets had run, never
  registered): pre-training review (REVIEW_REPORT_PRE_TRAINING.md, "M7 Lite") found large lakes/bays split across
  roles (W1). Water-body rule extended (name + HUC8 merging); split v2 rebuilt and re-frozen (sha256 above). Split
  fractions were changed from 25/20 to 20/25 (test/val) and the train↔val buffer to 2 km after inspecting **label
  counts only** (no model output): the 5-km train↔val buffer excluded 24% of chips. Also: one-sided Holm (W2),
  band-ratio form family (W3), selection score (W4), labels hash (W6), primary populations (W12).
- 2026-10-24 v1.2 (still before any non-smoke fit): HydroLAKES lake identity added as merge rule (d) (real lake
  polygons, as in M2) and the test-side buffer reduced to 2 km after inspecting label counts only; split v3
  re-frozen (sha256 above). Splits v1 (767571a9…) and v2 (73cd7de8…) were never used for any fit beyond 2-epoch
  smoke runs on train/val subsets and nothing from them was registered.
- 2026-11-08 v1.3 (before any test scoring; split unchanged): W8 overlap check implemented and failed for the stock
  teacher; Lite-safe teacher defined (§5). **Deviation D1** from RESEARCH_PLAN H-G6 wording ("held-out lakes"):
  turbidity and TSS primary populations are all water bodies (Test-A n = 92 / 98) instead of lakes only
  (Test-A lake/reservoir n = 58 / 62); lake-only results are reported as secondary. This choice was made on label
  counts before any model was fitted (v1.1, review W12); the lead will amend the plan. The earlier text quoted
  lake-only counts 59 / 68 (from an intermediate split build); the frozen split's counts are 58 / 62.

## 12. National track (added 2026-11-08, v1.3; RESEARCH_PLAN §15.5 amendment items 5–7)
**Why.** M7 is not saturated in the §15.5 sense on Secchi (all models far from the ceiling, val log-R² ≤ 0.4), but
turbidity and TSS are: every method ties within CIs because Test-A has 44 water bodies per target. The national track
adds data and power; it is **in addition to** the original 4-state track, whose split, labels, results and claims are
unchanged and always reported. Claims name their track.

- **Data.** `build_naip_matchups.py --national` (unchanged code; manifest `naip_matchups_national`, files in
  `processed/naip_matchups/national/`): 21,455 matched samples, 11,286 NAIP chips, 3,607 S2 teacher chips, all states
  in the WQP table. The canonical 4-state files and manifest were restored byte-identical; the chip stores are
  supersets (all 6,195 existing NAIP and 2,130 S2 chip ids and shards unchanged, checked). Disk added ≈ 0.8 GB.
- **Labels** (`processed/lite_labels/labels_national.parquet`, manifest `lite_labels_national`): frozen 4-state rows
  unchanged + 4,579 new chips with the same label and water-body rules (`sentinel/models/lite/national.py`).
- **Split** `docs/preregistration/lite_split_national.json`, sha256
  **`8e3b1ee00c8a5d2b6c9f9b291c15ae0b3d329abaa98c1d22887f2ec874cf7d19`** (pinned in
  `guard.FROZEN_NATIONAL_SPLIT_SHA256`), frozen before any model output on new data was seen. Rules in the
  `national.py` docstring: frozen chips keep their role; new chips in a national water body that contains a frozen
  chip become `train` only if all its frozen chips are `train` (else `excluded`); other new water bodies use the
  25-km block of their centroid (frozen block roles kept; new blocks test 20% / val 25%, seed 20261108); year cut as
  §3; 2-km buffers to every test-like and val-like site. Protected: `test`, `test_time`, `test_n`, `test_time_n`.

| role | chips | Secchi (wb) | turbidity (wb) | TSS (wb) | chl-a |
|---|---|---|---|---|---|
| train | 4,831 | 3,667 (1,146) | 1,555 (436) | 1,251 (407) | 2,059 |
| val (frozen) | 1,278 | 1,163 (370) | 132 (26) | 161 (37) | 428 |
| val_n | 780 | 365 (121) | 483 (136) | 303 (102) | 364 |
| test (A, frozen) | 1,178 | 1,094 (392) | 92 (44) | 98 (44) | 310 |
| **test_n (Test-N)** | 773 | 379 (152) | 486 (135) | 270 (105) | 422 |
| test_time (B) | 747 | 602 | 190 | 166 | 361 |
| test_time_n | 426 | 205 | 258 | 146 | 223 |
| excluded | 564 | 413 | 146 | 54 | 114 |

- **Headline (national track):** Test-N = unseen water bodies in 42 states, all years. Secondary: Test-A ∪ Test-N;
  test_time_n; by state and site class. Primary populations, metrics, statistics, baselines (refitted on the national
  train role with the same fixed hyper-parameters; band-ratio form chosen on national val) and the selection score S
  (§5, on val ∪ val_n) as in §3–§6. H1/H2 are re-tested on this track with their own Holm family.
- **Power check (counts only, before selection).** Scaling the val paired-CI widths of the 4-state student-vs-GBM
  comparison by √(water bodies_val / water bodies_test): expected 95% CI half-width on Test-N ≈ 13 MdSA points
  (turbidity), 17 (TSS), 9 (Secchi), vs ≈ 22 / 26 on Test-A. Differences smaller than this stay undetectable.
- **KD on this track** needs a teacher disjoint from the national val/test roles (a second Lite-safe retrain with the
  national evaluation sites excluded).
