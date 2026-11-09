# Model card — SENTINEL-Lite (M7): 4-band (R, G, B, NIR) student for drone/aerial water quality

**Status:** development, **validation only**. No test-set result exists (`FREEZE_lite.json` not written; the test
roles have never been loaded). Every number below is a registry entry (`results/registry.jsonl`, workstream
`M7-lite`; experiments `lite_val`, `lite_val_compare`, `lite_export`, `lite_robustness_val`, `lite_val_national`)
or is marked `[PENDING]`. Validation numbers are **selection-optimistic** (the selected configuration is the best of
16 tried on the same val role; the original track).

- Protocol: `docs/preregistration/lite_protocol.md` v1.3 (deviation D1 and the W8 teacher result in §3/§5/§11;
  national track §12).
- Splits: original 4-state `lite_split.json` (sha256 `46ed6e1f…7475`); national `lite_split_national.json`
  (sha256 `8e3b1ee0…7d19`).

## Intended use
- Chip-level (≈ 30 m centre window) estimates with a predictive σ of **Secchi depth (m), turbidity (NTU), TSS (mg/L)**
  and **chl-a trophic class** (Carlson TSI edges; chl-a is reported as classes only, scope rule) from 4-band
  uint8 imagery of inland and coastal water, as a starting point for a drone camera with an NIR band.
- **Not for:** phycocyanin, CDOM, DO, pH, conductivity, nutrients, toxins; regulatory decisions; imagery that is not
  NAIP-like 4-band DN (the drone domain gap is untested, see Limitations).

## Model
- `sentinel/models/lite/model.py`: MobileNetV4-Conv-Small (timm, ImageNet init, 4-channel stem), light FPN
  decoder (48 ch, stride 4), per-pixel μ and log σ² for 4 targets (log10); chip value = water-weighted mean of the
  per-pixel μ over the centre window (dark-NIR water mask).
- **Selected configuration (pre-registered rule, protocol §5):** `mnv4s_raw_crop128_kd03`:
  raw DN/255 input, centre 128-px training/inference crop, 40 epochs, Gaussian β-NLL on in-situ labels
  + 0.3 × distillation NLL against the **Lite-safe M2 teacher** (see Distillation). Reported prediction: mean μ
  over 3 seeds (0, 1, 2). Run: `$SENTINEL_RUNS/lite/student_mnv4s_raw_crop128_kd03_20261108-152950`.
- Size (per seed model): 1,323,880 parameters; 1.128 GMACs per 512 × 512 tile; ONNX FP32 5.29 MB, INT8 (static
  QDQ, per-channel) 1.65 MB. The 3-seed ensemble is 3× these (3.38 GMACs, 4.9 MB INT8), which **exceeds** the
  2-GMAC and 4-MB budgets; the budget-compliant deployable unit is one seed model (see Freeze candidate).
- Desktop-proxy latency (ONNX Runtime CPU EP, Xeon Silver 4214, 1 thread, 512² tile, median of 30;
  **not device latency**; shared host at load ≈ 30): INT8 32–41 ms, FP32 60–63 ms (seeds 0–2).

## Training data
- NAIP 4-band DOQQ chips (256 × 256, native GSD 0.3–1 m) × WQP surface samples within ±1 day (MN, FL, WI, MI;
  2010–2021 for train/val). Train role: 2,795 chips / 725 water bodies (Secchi 2,526; turbidity 575; TSS 544).
  Datasheet `docs/datasheets/naip_matchups.md`; labels manifest `lite_labels`.
- National track (additional, §12): 4,831 train chips in 46 states (`naip_matchups_national`,
  `lite_labels_national`).

## Distillation (H2) and the W8 leakage check
- The stock M2 teacher (`docs/model_cards/orbit_teacher.json`) had fitted rows at 131/420 Lite Test-A,
  105/327 Test-B and 106/378 val water bodies, so it is **not used**.
- Instead, M7 retrained the identical M2 SatWQ-Net recipe (5 seeds) after removing every teacher row that shares a
  site id, lies ≤ 2 km from, or is in the same HydroLAKES lake as a Lite val/test/test_time site (2,394/9,149
  matchup rows, 27/2,631 GLORIA rows). Artifacts are in `$SENTINEL_RUNS/lite/teacher_litesafe/` (manifest with
  checkpoint sha256, `seen_points.parquet`, `exclusion.json`).
- `lite_teacher_targets.py` re-checks disjointness: it found 0 overlapping water bodies for val, test and test_time.
- Teacher inner-set selection MdSA is 45.0–47.4 for the Lite-safe teacher vs 41.2–42.1 for the stock teacher. These
  are on different inner rows and are not comparable as quality numbers.
- Targets are the imagery-only (context-off) per-pixel teacher medians on 892 S2 chips paired with 917 Lite train
  chips, using the teacher's chip-level variance.

## Validation results (original 4-state track; MdSA %, lower is better; 95% water-body cluster-bootstrap CI)
Primary populations: Secchi lakes/reservoirs (n = 1,083 chips, 366 water bodies); turbidity (n = 132, 26 wb) and
TSS (n = 161, 37 wb) all water bodies (deviation D1).

| Model | Secchi | Turbidity | TSS |
|---|---|---|---|
| **Student, selected (crop128 + KD 0.3)** | **46.6** | **72.6** | 107.1 |
| Student, same without KD (crop128) | 49.1 | 77.4 | 115.1 |
| GBM-RGBN (comparator) | 41.2 | 112.1 | 129.5 |
| GBM-RGBN + lat/lon/DOY | 35.4 | 87.6 | 122.4 |
| Band-ratio (best form on val) | 61.0 | 83.3 | 105.3 |
| Train median | 73.3 | 100.8 | 97.7 |

Paired differences (selected minus comparator; negative = student better):
- vs band-ratio: Secchi −14.4 [−20.4, −9.1] (Holm-rejected); turbidity −10.7 [−33.4, +25.9]; TSS +1.8 [−24.0, +21.9].
- vs GBM-RGBN: Secchi **+5.4 [+0.5, +9.6] (student worse)**; turbidity −39.5 [−71.6, +4.2]; TSS −22.4 [−38.1, +7.7].
- **H2, KD vs no KD:** Secchi −2.6 [−5.2, −0.4] (one-sided p = 0.01); turbidity −4.8 [−36.4, +12.1];
  TSS −8.0 [−28.0, +2.5].
- **H-G6 on val: not met.** The student loses to GBM-RGBN on Secchi. Turbidity and TSS CIs include 0 against both
  baselines, and no method clearly beats the train median on TSS.
- Single-target Secchi training gave val Secchi 45.5 [40.6, 52.9]. It was not selected: the selection score covers
  all three targets.

## Robustness (val, protocol §7; `lite_robustness_val`; relative MdSA change vs clean, 3-seed ensemble)
Soft threshold: ≤ 20% degradation under 25% glint, 10 DN haze and ±5% gain.

| Perturbation | Secchi | Turbidity | TSS |
|---|---|---|---|
| glint 10 / 25 / 40% of pixels | +75% / **+124%** / +138% | +9% / **+24%** / +22% | −6% / −13% / −17% |
| haze 5 / 10 / 20 DN | +1% / +0.3% / +3% | +7% / +12% / +12% | −8% / −11% / −14% |
| resolution ×2 / ×4 / ×8 | +2% / +10% / +28% | +4% / +7% / +15% | −0% / −7% / −1% |
| gain ±5% (5 draws) | −1% | +3% | −3% |
| NIR misregistration 1 / 3 / 10 px | 0% / 0% / 0% | +2% / +3% / 0% | 0% / 0% / +5% |

- **Soft threshold failed for glint:** Secchi degrades by 124% and turbidity by 24%. The haze and ±5% gain
  thresholds are met.
- TSS "improves" under several perturbations. This is consistent with TSS predictions carrying little skill, since
  clean val TSS is near the train median.
- Weighted-median pooling (inference-time variant, same weights; `student_mnv4s_raw_crop128_kd03_medpool`) gives
  clean val Secchi 47.1, turbidity 73.5, TSS 105.1, S = 0.870. That ties the selected model under the selection rule,
  so it is not selected. It does **not** fix glint: Secchi +118% and turbidity +18% at 25% glint.
- **Glint augmentation** (pre-declared, protocol §9: `student_mnv4s_raw_crop128_kd03_glint_20261108-201455`).
  - Val MdSA: 47.1 / 81.9 / 106.3, S = 0.898. The keep rule needed S ≤ 0.878, so it was **not kept**.
  - Paired vs the selected model: Secchi +0.5 [−1.5, +2.8]; turbidity +9.4 [−18.3, +18.1]; TSS −0.8 [−12.0, +22.6].
  - It removes the glint failure: Secchi +0.3% and turbidity −9% at 25% glint; haze 10 DN +3% / +5%; gain ≤ 3%.
    This result is optimistic, because training and test use the same blob family.
  - An override in its favour would need the lead's decision, recorded as a deviation.

## INT8 vs FP32 (val; `lite_export`, one registry model per seed)
ONNX-FP32 reproduces torch exactly on the 128-px centre crop (max |Δ| = 0).

INT8 − FP32 val MdSA, per seed:

| Seed | Secchi | Turbidity | TSS |
|---|---|---|---|
| 0 | −0.45 | −6.41 | +5.82 |
| 1 | −0.24 | −0.04 | −1.33 |
| 2 | −0.66 | +0.70 | −1.90 |

Proxy latency (corrected run, INT8, 1 thread): 30.1–31.4 ms. An earlier export had compared ONNX on the full 256-px
chip against torch on the crop; the corrected rows supersede those rows.

## National track (protocol §12; experiments `lite_val_national`, `lite_val_compare_national_*`)
Baselines and students are refit on the national train role (4,831 chips). Val = frozen val ∪ val_n
(Secchi n_wb 491, turbidity 162, TSS 139).

| Model (national train) | Secchi | Turbidity | TSS |
|---|---|---|---|
| Student crop128 (no KD) | 49.8 | 107.9 | 101.0 |
| Student crop128 + KD 0.3 (national Lite-safe teacher) | `[PENDING]` | | |
| GBM-RGBN | 45.2 | 116.8 | 97.9 |
| GBM-RGBN + context | 38.7 | 111.9 | 102.7 |
| Band-ratio (best form on national val) | 69.3 | 129.6 | 100.4 |
| Train median | 84.3 | 151.9 | 107.7 |

- Student vs GBM-RGBN: Secchi +4.6 [+0.8, +8.8] (student worse); turbidity −8.8 [−28.8, +15.2]; TSS +3.0 [−12.5, +21.3].
- Student vs train median, turbidity: −44.0 [−101.2, −18.2]. Turbidity has real skill nationally. TSS has none for
  any method: the CIs against the median include 0.
- On val_n only (new states), Secchi gets much harder for everyone: student 72.1, GBM 60.8, band-ratio 99.6, median 155.2.

**More data on the original track.** Scored on the frozen 4-state val:
- the national-trained student gives Secchi 46.7, turbidity 88.5, TSS 94.3, vs 49.1 / 77.4 / 115.1 with 4-state
  training (paired CIs include 0);
- the national-trained GBM gives Secchi 42.3, turbidity 110.2, TSS 96.5;
- the train median gives TSS 75.0.

More data helps GBM about as much as the student, so the Secchi gap remains.

## Limitations
- NAIP DN are not reflectance; colour balancing differs by state and year. The domain gap to a drone camera is
  untested.
- The water-body proxy is distance/name/HUC8/HydroLAKES based. River reaches more than 2 km from every Lite site are
  not matched in the W8 check.
- Turbidity and TSS have low power on the original track (Test-A: 44 water bodies each). The national track adds
  power (Test-N: 135 / 105 water bodies).
- The clear end of the Secchi distribution is truncated (disk-on-bottom results are dropped).
- Validation selection over 13 configurations makes the val numbers optimistic.
- Orbit turbidity may mix NTU and FNU in the teacher; Lite labels are NTU only.

## Red-edge gap (descriptive, val; dry run of the test-scoring script)
These are the same chips, scored on primary populations; the Lite-safe teacher never saw val water bodies.

| | n | Lite-safe teacher on S2, all bands, no context | student ensemble on NAIP |
|---|---|---|---|
| Secchi | 335 | 27.9 | 40.3 |
| Turbidity | 30 | 105.3 | 71.1 |
| TSS | 44 | 67.8 | 96.1 |

## Other configurations tried (val, not selected)
Secchi-only + KD 0.3: 46.3 / 96.0 / 136.3. There is no Secchi gain over Secchi-only without KD (+0.9 [−1.6, +3.5]).

## Freeze candidate (for the lead's pre-test review; nothing frozen, test not scored)
- File: `docs/preregistration/FREEZE_lite_candidate.json`. It is not `FREEZE_lite.json`, and the guard ignores it.
- Contents:
  - split sha256;
  - scoring-code sha256 (to be recomputed when the real freeze is written);
  - weights sha256 for the primary seeds 0–2, the no-KD comparator seeds 0–2 (H2), the robust-variant seed 0, and
    the deployable seed-0 ONNX FP32/INT8;
  - the val-selected band-ratio forms;
  - the Lite-safe teacher manifest hash;
  - configurations tried: 16.
- Deployable unit: **seed 0** of `mnv4s_raw_crop128_kd03` (pre-declared; val 48.0 / 76.3 / 109.8). All 3 seeds and
  the ensemble are scored, and the §15.1 seed rule is applied.
- Seed rule on val (dry run): all 3 seeds beat band-ratio on Secchi; beat GBM on turbidity and on TSS (point
  estimates); lose to GBM on Secchi.
- Single test run: `python -B scripts/eval/lite_score_test.py --allow-test`, only after `FREEZE_lite.json` exists.
- Dry run on val (`--dry-run-val`, `$SENTINEL_RUNS/lite/smoke/dryrun_test_scoring_20261109-092043`) reproduces every
  registered val number exactly. Nothing was registered and no test row was loaded.
- The national-track freeze is separate and later (its KD student is still training).
