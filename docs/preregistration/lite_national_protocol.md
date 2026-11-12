# SENTINEL-Lite (M7): national-track protocol additions

**Status:** v0.1, 2026-11-09, written **before the national freeze**. No national test role (`test_n`,
`test_time_n`) has been loaded. No national val result exists for any stratum or event defined here.

**Base rules.** The national track is defined in `docs/preregistration/lite_protocol.md` §12 (frozen with the 4-state
track and not edited):
- split `lite_split_national.json`, sha256 `8e3b1ee0…7d19`;
- headline Test-N;
- secondary test_time_n (Test-A ∪ Test-N dropped, v1.4);
- populations, metrics and statistics as §3–§6 and §8a;
- the v1.7 single-look hardening.

This file adds the items the lead required on 2026-11-09 ("SOTA campaign", `docs/plans/sota_campaign.md`). Each item
was fixed before any result on it existed.

## N1. A-priori difficulty strata (T-X; RESEARCH_PLAN §15.5 item 5)
Thresholds come from the **national-1 train role only** (`national.role_frame(split, ("train",))`, computed
2026-11-09). They are fixed here and are not recomputed on later data.

| Stratum | Definition | Threshold |
|---|---|---|
| X1 small water bodies (drone use case: farm and stormwater ponds) | lake/reservoir site with no HydroLAKES polygon within 100 m (`hylak_id` missing; HydroLAKES holds lakes ≥ 10 ha) | — |
| X2 high turbidity (plume conditions) | turbidity label > train P90 | > 32.38 NTU (log10 1.5103; train n = 1,555) |
| X3 high TSS | TSS label > train P90 | > 50.0 mg/L (log10 1.6990; train n = 1,251) |
| X4 low Secchi (eutrophic or turbid) | Secchi label < train P10 | < 0.549 m (log10 −0.2607; train n = 3,667) |

- **Scoring.** Each stratum is scored in the single national look: per-model MdSA with water-body cluster-bootstrap
  CIs, plus paired Δ vs band-ratio and GBM-RGBN.
- **Secondary, no Holm.** A stratum result is a scoped claim only if it meets the §15.1 conditions within the
  stratum and N2's minimum n.
- **Selection effect.** Strata X2–X4 select on the label, so MdSA within them is conditional on the true value.
  Comparisons are paired on identical chips, so model-vs-model differences remain fair, but absolute errors are
  not comparable to the full population.

## N2. Minimum n for any interpretation (post-results review, 2026-11-09)
- Every slice, stratum and site-class result needs **≥ 10 chips per target**, counted per target and not per slice,
  before it is interpreted.
- Below that it is printed with its n and marked "not interpretable".
- A paired comparison additionally needs ≥ 5 water bodies.
- The national scorer computes these n per target and flags them in its output.

## N3. Event track on the national track (continuing lite_protocol.md §13)
- **E4a** (chl-a > 20 µg/L, lakes): descriptive only, as on the original track.
- **E3a_turb** (turbidity > 50 NTU, snapshot; all water bodies): `val_n` has 48 positives in 18 water bodies, so it
  is feasible.
  - Pre-declared here, before any national event result: ONE endpoint, ΔBrier(derived − climatology) on Test-N.
    "Derived better" iff the paired water-body 95% CI lies below 0.
  - The derived route is P(turbidity > 50) from the national student's turbidity μ/σ, with one Platt fit on
    val ∪ val_n (cross-fitted for val metrics).
  - Climatology is the 10-nearest-train-site positive rate.
  - Everything else (ΔAUPRC, band-ratio, GBM, direct head) is descriptive.
  - If Test-N has fewer than 30 positives or fewer than 10 water bodies with positives (events plan §1.2), E3a_turb
    is reported as descriptive only.
- **E3a_tss:** `val_n` has 31 positives / 8 water bodies, so it is underpowered and descriptive only.

## N4. Data extension before the freeze (lead decision (b), 2026-11-09)
- The national freeze is held until SOTA levers 1–3 finish (≤ 9 working days).
- Any new labelled data enters as split version `national-2`, pre-registered before any model sees the new rows:
  1. every national-1 chip keeps its role;
  2. new chips in a national water body that holds any evaluation chip (val, val_n, test, test_n, test_time,
     test_time_n) are `excluded`;
  3. new water bodies follow the national-1 block roles where the block exists, otherwise the salted random rule;
  4. 2-km buffers, as in national-1.
- **Comparator parity:** every baseline is refitted on exactly the student's training rows.
- The **±7-day Secchi window** (Secchi only; turbidity and TSS stay ±1 d) applies to all models. Counts are declared
  in N5 before any build.
- SSL pretraining uses only unlabelled chips outside every val, test and buffer water body of both splits (by
  coordinates), with an exclusion manifest.

## N5. Counts and power, recorded before any build (2026-11-09; `scripts/eval/national_lite_counts.py`)
Counts only, from the cached national inputs (WQP surface samples, JRC flags, NAIP STAC items).
`$SENTINEL_RUNS/lite/counts_national2/counts.json`.

| Target / window | Samples | (site, DOQQ) pairs | Sites | Lake samples | States | New NAIP chips needed |
|---|---|---|---|---|---|---|
| Secchi ±1 d (current; reproduces the national build exactly) | 8,135 | 8,016 | 5,689 | 6,788 | 48 | 0 |
| **Secchi ±7 d (proposed)** | **38,061** | **30,647** | **14,086** | 31,109 | 49 | **22,601 (≈ 3.2 GB at 140 KB/chip)** |
| Turbidity ±1 d (unchanged) | 4,433 | 4,138 | 3,380 | 1,781 | 49 | 0 |
| TSS ±1 d (unchanged) | 2,879 | 2,822 | 2,300 | 1,162 | 46 | 0 |

**N4 refinement (2026-11-09, before any build): evaluation blocks stay unchanged.** Under `national-2`, new ±7-day
chips may enter **training only**:
- they must belong to a water body whose (national-1 rule) role is train;
- new chips in val/val_n/test/test_n/test_time/test_time_n blocks or water bodies, or within the 2-km buffers, are
  `excluded`.

So val, val_n, Test-N and test_time_n keep exactly their national-1 chips and ±1-day labels. Model comparisons
across levers stay on identical evaluation rows, and evaluation labels keep one matching window.

**SSL corpus (lever 2) estimate:** 20–30 k unlabelled water chips × 140 KB ≈ 2.8–4.2 GB.
**Total new data for levers 1–2:** about 6–8 GB, including S2 teacher chips for the new pairs (≈ 0.3 GB at the
current pairing rate). That is under the 10 GB cap. With 100 GB free now, the 60 GB floor would not be reached.

LAGOS-US LIMNO: the package id and size are not yet identified (EDI). Much of its Secchi comes from the same
agencies as WQP, so its added value is uncertain; it is only pursued after ±7 d. NLA (local `nars_lakes`): the
NAIP-date match counts need STAC queries for NLA sites and are pending.

## N6. National-2 split, frozen 2026-11-10 before any model saw the new rows
`docs/preregistration/lite_split_national2.json`, sha256 **`976b5be7ed7c76e9417d526f41e7b6899f88e49e3dc00b7dba077bf78f40ebe2`**
(pinned in `sentinel/models/lite/national2.py`). Labels are in `processed/lite_labels/labels_national2.parquet`.
Built by `national2.build_labels` / `build_split` from `naip_matchups_national_pm7` with the N4 train-only rules.

| Role | Chips | Note |
|---|---|---|
| train | 10,535 (national-1: 4,831) | Secchi labels 9,371 / 2,822 water bodies (national-1: 3,667). Turbidity 1,555, TSS 1,251, chl-a 2,059 (unchanged: new chips carry Secchi only). |
| val / val_n | 1,278 / 780 | identical to national-1 (chip ids and labels checked) |
| test / test_n / test_time / test_time_n | 1,178 / 773 / 747 / 426 | identical to national-1 |
| excluded | 17,103 | new chips in evaluation water bodies or blocks, in 2022–23, or ≤ 2 km from evaluation sites |

**Comparator parity.** Every baseline is refitted on exactly these 10,535 train rows
(`scripts/train/national2_run.py baselines --track national`).

## N7. Numeric chl-a and prediction intervals (user requirement, lead 2026-11-10; before the national freeze)
- **Chl-a in µg/L** (lakes), from the existing chl-a head (log10 μ/σ). Reported as a **secondary regression
  target**:
  - MdSA with a water-body cluster-bootstrap CI, SSPB, log-RMSE, log-R² and log-slope;
  - paired MdSA Δ vs band-ratio, GBM-RGBN and GBM + context (no Holm).
  - The trophic-class balanced accuracy stays reported. **Chl-a never enters selection**: S uses Secchi,
    turbidity and TSS only.
  - Power: national val has chl-a lakes 428 (val) + 364 (val_n) labels. It stays secondary in any case, since no
    test of it was pre-registered before val results existed.
- **80% prediction intervals for every target** (Secchi, turbidity, TSS, chl-a): μ ± 1.2816 σ in log10.
  - Ensemble σ² = mean seed σ² + between-seed variance of μ; seed 0 uses its own σ.
  - Reported: empirical coverage (nominal 0.80) and median interval width.
- N2 applies: ≥ 10 chips per target, otherwise "not interpretable".
