# SENTINEL-Lite (M7): single test look, original 4-state track (2026-11-09)

**Sources.** This file is written only from the sealed look outputs and the registry:
- `$SENTINEL_RUNS/lite/test_look/test_results.json` and the predictions sealed 0444 with `SHA256SUMS`;
- registry experiments `lite_test*`: 238 + 12 + 12 + 6 + 36 + 1 rows, plus one `lite_test_claim` and one
  `lite_test_sealed` row;
- the hashed export report of the deployable model.

It is a post-look write-up. It changes no frozen file (`lite_protocol.md` v1.7, the scorer and the code are
untouched).

| | |
|---|---|
| Freeze | `docs/preregistration/FREEZE_lite.json` (sha256 `a066b57a…`; status FROZEN; authorised 2026-11-09T22:05:18Z) |
| Scoring code | `a7a8be82…` (80 files) |
| Look | 2026-11-09 22:05–22:30 UTC, exit 0. Claim, seal and score each ran once. Nothing was "not run". |
| Selected model | `mnv4s_raw_crop128_kd03` (val selection score S = 0.868 over **16** configurations). Ensemble = seeds 0–2. Deployable / H-G6 carrier = INT8 ONNX of seed 0. |
| Test-A (`test`, unseen water bodies, all years) | 1,178 chips. Secchi on lakes: n = 1,065 / 387 water bodies. Turbidity, all water bodies: 92 / 44. TSS, all water bodies: 98 / 44. |
| Test-B (`test_time`, NAIP 2022–23 at train/val water bodies) | 747 chips, **descriptive only**. |

All metrics are MdSA (%; lower is better) with 95% water-body cluster-bootstrap CIs (B = 2000). "Δ" = student −
comparator, so a negative Δ favours the student.

## Headline verdicts (computed in code, protocol §8a)
| Verdict | Result |
|---|---|
| H1 (3-seed ensemble; Holm-6; rejection needs Holm ∧ CI excluding 0) | **not met.** No comparison rejected. |
| H1-deployable (INT8 seed 0; its own Holm-6) | **not met.** No comparison rejected. |
| H2 (distillation; Holm-3 + every-seed rule) | **not met.** |
| Budgets (INT8 seed 0) | **met.** 1,323,880 params; 1.128 GMACs / 512² tile; INT8 1.645 MB; proxy latency median 30.2 ms (p90 30.5; ONNX Runtime CPU EP, 1 thread, desktop proxy, **not device latency**). |
| Red-edge gap reported | yes |
| **H-G6** | **FALSE** |
| §15.1 label (carrier) | **negative.** GBM-RGBN is significantly better on Secchi. |

## Test-A scores (primary populations)
| Model | Secchi | Turbidity | TSS | Chl-a class (bal. acc.) |
|---|---|---|---|---|
| Student ensemble (KD) | 54.4 [48.0, 59.1] | 125.9 [68.9, 188.2] | 74.7 [47.1, 113.8] | 0.334 |
| **Deployable INT8 seed 0** | 55.0 [46.0, 60.2] | 125.5 [83.7, 203.7] | 78.0 [53.2, 151.1] | 0.337 |
| FP32 seed 0 (= torch seed 0) | 53.1 [48.7, 59.9] | 132.3 [83.5, 205.2] | 80.5 [55.6, 111.1] | 0.343 |
| Seeds 0 / 1 / 2 | 53.1 / 54.2 / 51.3 | 132.3 / 133.8 / 105.4 | 80.5 / 83.9 / 69.1 | |
| No-KD ensemble (H2 comparator) | 52.9 | 113.6 | 67.7 | 0.338 |
| Band-ratio (val-selected form) | 54.2 [48.6, 61.7] | 117.3 [83.6, 194.0] | 84.2 [59.4, 119.3] | 0.311 |
| **GBM-RGBN** | **44.2 [39.7, 49.5]** | **87.2 [67.8, 156.8]** | 75.2 [58.4, 103.6] | 0.381 |
| GBM-RGBN + lat/lon/DOY | 39.6 | 89.3 | 63.8 | 0.446 |
| Location kNN | 50.0 | 183.4 | 83.3 | 0.342 |
| Location/season GBM | 54.3 | 155.1 | 114.8 | 0.370 |
| Train median | 66.7 | 158.9 | 114.2 | 0.250 |

**Lake/reservoir-only turbidity and TSS** (secondary, deviation D1; turbidity n = 58 / 26 water bodies, TSS
62 / 30). MdSA [95% CI]:

| Model | Turbidity | TSS |
|---|---|---|
| Ensemble | 124.3 [63.0, 227.6] | 69.3 [42.0, 116.7] |
| INT8 | 116.9 [59.3, 231.6] | 66.6 [40.1, 111.4] |
| Band-ratio | 121.8 [85.7, 268.8] | 69.0 [53.8, 118.2] |
| GBM-RGBN | 87.2 [63.2, 229.5] | 69.5 [45.0, 97.4] |
| GBM + context | 91.9 [62.7, 196.3] | 54.6 [44.9, 99.9] |

**Secchi, all site classes (secondary):** ensemble 55.8, GBM-RGBN 45.2 (n = 1,094).

**By site class (point MdSA).** The non-lake classes are small.

| Class / target | n | Ensemble | GBM-RGBN |
|---|---|---|---|
| Secchi, river/stream | 21 | 146.3 | 124.8 |
| Turbidity, estuary/coastal | 13 | 155.9 | 80.8 |
| Turbidity, river/stream | 21 | 126.0 | 105.6 |
| TSS, river/stream | 29 | 105.9 | 125.6 |

**Chl-a trophic class** (soft threshold §15.3, balanced accuracy, lakes, n = 271):
- ensemble 0.334 [0.282, 0.389] vs band-ratio 0.311 [0.267, 0.356]. That is above on the point estimate only;
  there is no paired test and the CIs overlap;
- GBM-RGBN 0.381 [0.319, 0.451]; GBM + context 0.446.

## Confirmatory families (Test-A only)
**H1, student ensemble.** Holm over 6; a rejection needs Holm AND the CI to exclude 0.

| Comparison | Δ [95% CI] | one-sided p | Rejected |
|---|---|---|---|
| Secchi vs band-ratio | +0.2 [−6.0, +4.5] | 0.42 | no |
| Secchi vs GBM-RGBN | **+10.2 [+4.4, +14.2]** (student worse) | 0.9995 | no |
| Turbidity vs band-ratio | +8.5 [−34.6, +47.8] | 0.67 | no |
| Turbidity vs GBM-RGBN | +38.6 [−9.2, +79.7] | 0.93 | no |
| TSS vs band-ratio | −9.5 [−40.3, +18.8] | 0.32 | no |
| TSS vs GBM-RGBN | −0.5 [−28.9, +33.0] | 0.39 | no |

Seed rule, Δ vs GBM-RGBN on Secchi: +8.9 / +10.0 / +7.1 (seeds 0/1/2), so every seed is worse.

**H1-deployable, INT8 seed 0.**

| Comparison | Δ [95% CI] |
|---|---|
| Secchi vs band-ratio | +0.8 [−6.9, +5.5] |
| Secchi vs GBM-RGBN | **+10.8 [+3.1, +15.3]** |
| Turbidity vs band-ratio | +8.2 [−26.1, +61.2] |
| Turbidity vs GBM-RGBN | +38.2 [−3.8, +91.2] |
| TSS vs band-ratio | −6.3 [−33.6, +33.6] |
| TSS vs GBM-RGBN | +2.8 [−24.8, +49.5] |

None is rejected.

**H2, KD ensemble vs the no-KD ensemble.** The comparator is the re-run under current code, bit-identical to the
2026-10-24 run.

| Target | Δ [95% CI] | Per-seed Δ (seeds 0/1/2) |
|---|---|---|
| Secchi | +1.6 [−1.7, +3.5] | −0.1 / +2.0 / −3.3 |
| Turbidity | +12.3 [−15.5, +52.6] | +40.1 / +18.4 / −10.5 |
| TSS | +7.0 [−8.6, +26.7] | +8.0 / +6.3 / −3.0 |

None is rejected, and the every-seed rule fails for all targets. **The val distillation gain (Secchi −2.6
[−5.2, −0.4]) did not replicate on Test-A.**

**Every applicable baseline vs the carrier** (secondary, no Holm):

| Target | GBM-RGBN + context | Location kNN | Location/season GBM | Train median |
|---|---|---|---|---|
| Secchi | +15.4 [+7.1, +20.3] | +5.0 [−4.9, +13.5] | +0.7 [−10.5, +9.6] | −11.7 [−26.1, −5.5] |
| Turbidity | +36.2 [+3.0, +101.5] | −57.9 [−124.2, +42.1] | −29.6 [−99.1, +53.1] | −33.4 [−62.5, +56.2] |
| TSS | +14.1 [−12.8, +66.8] | −5.3 [−95.9, +59.7] | −36.8 [−72.6, +14.4] | −36.2 [−93.3, +14.0] |

## Secondary slices (Test-A; point MdSA)
| Slice | Ensemble | INT8 | Band-ratio | GBM-RGBN |
|---|---|---|---|---|
| Secchi, chips > 10 km from every train site | 54.8 | 55.0 | 54.2 | 47.4 |
| Secchi, future years (Test-A ∩ NAIP 2022–23; n = 148 / 98) | 43.1 | 42.5 | 51.3 | 41.0 |
| Turbidity, future years (n = 8 chips / 6 water bodies; not interpretable) | 110.0 | 102.1 | 42.8 | 14.1 |
| TSS, future years (n = 3 chips / 2 water bodies; not interpretable) | 32.8 | 39.8 | 25.4 | 71.5 |
| Secchi, water bodies with no teacher row ≤ 5 km (W8) | 50.7 | 49.3 | 53.0 | 42.3 |

W8 check: H2 restricted to water bodies with no teacher row within 5 km is also null (Secchi +0.3 [−4.0, +3.0]).

## Test-B (descriptive only; NAIP 2022–23 at train/val water bodies)
**Scores** (Secchi n = 480 / 303 water bodies; turbidity 190 / 28; TSS 166 / 35):

| Model | Secchi | Turbidity | TSS |
|---|---|---|---|
| Ensemble | 54.6 | 138.8 | 98.4 |
| INT8 | 52.5 | 141.6 | 90.5 |
| Band-ratio | 49.6 | 220.5 | 115.6 |
| GBM-RGBN | 58.7 | 116.3 | 54.5 |
| GBM + context | 49.3 | 88.2 | 45.3 |
| Train median | 80.1 | 250.2 | 144.2 |

**Ensemble Δ:**

| Comparison | Δ [95% CI] |
|---|---|
| vs GBM-RGBN, Secchi | −4.1 [−10.4, +3.5] |
| vs band-ratio, turbidity | −81.7 [−150.3, −14.2] |
| vs band-ratio, TSS | −17.2 [−50.0, +1.3] |
| KD vs no-KD, Secchi | −3.3 [−7.1, +0.8] |

Against GBM-RGBN, the ensemble is worse on Test-B turbidity and TSS (scores above). No Test-B Δ carries a claim,
and Test-B has no Holm flags.

## Robustness (§7; Test-A, primary populations; soft threshold ≤ 20% degradation)
| Model | Clean Secchi | Glint 10 / 25 / 40% | Haze 10 DN | Gain ±5% | Resolution ×8 | NIR misreg 1–10 px |
|---|---|---|---|---|---|---|
| Deployable (seed 0) | 53.1 | 80.5 / **86.1 (+62%)** / 89.7 | 54.7 (+3%) | 54.9 (+3%) | 61.9 (+17%) | 52.6–53.3 |
| Ensemble | 54.4 | 72.2 / **90.7 (+67%)** / 97.3 | 54.0 | 52.8 | 60.6 | 53.9–54.4 |
| Robust variant (glint-augmented seed 0; secondary) | 52.1 | 52.4 / 52.5 / 52.0 | 54.0 | 51.1 | 58.6 | 52.4–52.6 |

- **The glint soft threshold fails** for the selected model, as on val. On Test-B, seed-0 Secchi goes from 54.9 to
  123.4 (+125%) at 25% glint.
- Haze, ±5% gain and NIR misregistration pass.
- Robust variant vs primary seed 0, paired: clean Secchi −1.0 [−4.3, +1.7]; at 25% glint −33.6 [−45.3, −23.7]
  (Test-B: −72.5 [−84.3, −53.3]).
- Caveat (protocol §9): training and test use the same glint blob family, so the variant's glint robustness is
  optimistic.

## Red-edge gap (descriptive; same chips with an S2 chip ≤ ±1 d)
The Lite-safe M2 teacher runs on S2 (all bands, no context); the students run on NAIP.

| Role / target | n | Teacher (S2) | Student ensemble (NAIP) | INT8 (NAIP) |
|---|---|---|---|---|
| Test-A Secchi | 321 | 36.2 | 48.9 | 46.0 |
| Test-A turbidity | 25 | 65.8 | 78.1 | 95.8 |
| Test-A TSS | 19 | 22.2 | 54.0 | 46.3 |
| Test-B Secchi | 247 | 33.3 | 55.6 | 54.3 |
| Test-B turbidity | 148 | 177.1 | 161.6 | 176.8 |
| Test-B TSS | 109 | 44.0 | 121.7 | 112.1 |

On Secchi the S2 teacher with red-edge is better than the 4-band student on the same chips by 10–13 points on Test-A (n = 321) and 21–22 points on Test-B (n = 247). For TSS the teacher is also better. For turbidity the result is mixed (Test-B: the teacher is worse, 177.1 vs 161.6). These are descriptive point values with no CI, and turbidity/TSS n on Test-A is small (25 / 19).

## Ablations (Test-A, paired, ensembles)
| Ablation | Secchi | Turbidity | TSS |
|---|---|---|---|
| RGB-only vs RGBN | −0.5 [−3.0, +3.0] | +14.4 [−20.8, +31.8] | −1.3 [−17.5, +19.0] |
| GLORIA pretraining vs none (2026-10-24 runs, same code) | −1.0 [−3.3, +0.9] | −13.8 [−47.8, +23.4] | −12.1 [−35.2, +2.3] |
| INT8 vs FP32, seed 0 | +1.9 [−2.7, +3.3] | −6.8 [−13.0, +12.6] | −2.5 [−8.8, +10.4] |

- **RGB-only vs RGBN:** the NIR band gives no measurable benefit (W-f caveat: KD-path code parity is not shown).
- **GLORIA:** no CI excludes 0.
- **INT8 vs FP32:** on Test-B, Secchi is −2.4 [−5.2, −0.3].

## Event track E4a (algal bloom, chl-a > 20 µg/L): DESCRIPTIVE ONLY (protocol §13 v1.7)
No decision and no claim rests on E4a. `test_results.json` contains `"confirmatory": true` under `events.E4a` for
Test-A. That is a stale role label (Test-A vs Test-B) from the code path. **Nothing reads it.** The scorer computes
no event decision, and the post-E-patch check confirmed this. E4a is not confirmatory anywhere.

**Test-A lakes:** n = 271, 50 positives, 28 water bodies with positives (prevalence 0.185). Probabilities are
calibrated with Platt fits frozen on val.

| Route | AUPRC | Brier |
|---|---|---|
| Derived (student chl-a μ/σ) | 0.553 [0.401, 0.715] | 0.125 [0.093, 0.161] |
| Direct head | 0.498 [0.354, 0.675] | 0.129 [0.098, 0.166] |
| GBM-RGBN | 0.494 [0.319, 0.671] | 0.130 [0.093, 0.173] |
| Band-ratio | 0.415 [0.272, 0.636] | 0.138 [0.102, 0.176] |
| Climatology | 0.393 [0.233, 0.579] | 0.138 [0.099, 0.183] |

**Derived − X, Test-A:**

| X | ΔBrier | ΔAUPRC |
|---|---|---|
| Climatology | −0.013 [−0.038, +0.008] | +0.16 [−0.002, +0.33] |
| Band-ratio | −0.013 [−0.033, +0.005] | +0.14 [−0.08, +0.30] |
| Direct | −0.004 [−0.013, +0.004] | +0.055 [−0.010, +0.111] |
| GBM | −0.005 [−0.020, +0.009] | +0.06 [−0.06, +0.20] |

All CIs include 0.

**Test-B lakes:** n = 160, 58 positives. Derived AUPRC 0.836, Brier 0.141.

| X | ΔBrier |
|---|---|
| Climatology | −0.079 [−0.126, −0.036] |
| Band-ratio | −0.080 [−0.125, −0.038] |
| GBM | −0.002 [−0.034, +0.028] |
| Direct | +0.005 [−0.006, +0.018] |

**Tiers and E3** (descriptive):
- E4a > 12 µg/L, Test-A: AUPRC 0.64 (77 positives).
- E4a ≥ 50 µg/L, Test-A: AUPRC 0.35 (24 positives).
- E3 (TSS > 80, turbidity > 50): 0 positives on Test-A and 1 on Test-B, so it is not evaluable.

## Interpretation (honest reading)
- **Negative result for H-G6.** The 4-band student is a budget-compliant edge model (1.3 M params, 1.65 MB INT8,
  about 30 ms proxy). It beats the train median on Secchi, but it does not beat GBM on RGBN chip statistics.
  - GBM-RGBN is better by about 10 MdSA points on Secchi. The ensemble CI excludes 0, and the point difference
    favours GBM in each of the 3 seeds (+8.9 / +10.0 / +7.1).
  - The student only ties the band-ratio baseline.
  - The location-aware GBM is better still.
- **Val → test.**
  - Student Secchi went from 46.6 to 54.4, and GBM from 41.2 to 44.2. The student's gap to GBM widened (+5.4 on
    val, +10.2 on test), consistent with the selection optimism of 16 configurations and with Test-A's
    distribution shift (more Florida/estuarine chips, §3).
  - The val distillation gain did not replicate.
- **Turbidity and TSS** remain power-limited on Test-A (44 water bodies each). Every comparison CI is wide.
- **Robustness:** the selected model is fragile to sun glint (+62–125% Secchi degradation). A glint-augmented
  variant removes this on the synthetic perturbation (optimistic).
- **Red-edge gap:** S2 with red-edge is more informative than NAIP RGBN for Secchi on the same water (descriptive, no CI). For turbidity and TSS the comparison is small-n and mixed.
- The national track (protocol §12) is unaffected by this look: Test-A is not part of any national claim.
