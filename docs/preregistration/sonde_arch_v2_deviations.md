# M1 SENTINEL-Sonde — architecture round v2: deviations log (lead ruling L4)

**Stamped amendment:** `sonde_arch_v2_amendment.md` (sha256 `6c7b169d…60226`, registry `2921bb1ca000`), with
`sonde_arch_v2_lead_rulings.md` (sha256 `7cbf3987…ac55`). Both files are read-only and unchanged.

**Rule (L4).** Each entry below is dated, gives its reason, and needs the lead's stamp **before** any affected run
produces a validation number. Entries marked *interpretation* make a stamped sentence concrete without changing its
intent. They are listed so the reviewer can reject any of them.

**Status:** DRAFT v0.1 (2026-11-10, ~18:00 UTC), written by M1. No validation number exists on any affected arm.

| # | Date | Item | Stamped text | Implementation | Reason | Affects |
|---|---|---|---|---|---|---|
| D1 | 2026-11-10 | GEFS bias correction (interpretation) | §2c: "for each product and lead day, a per-site linear map … fitted on train-block overlap only: GEFS reforecast: WY2008–2021-09-30 ∩ product availability" | Reforecast (P2) and operational (P3) are corrected by **one** GEFSv12 map per site × lead × field (× month), fitted on their combined train-block overlap (reforecast to 2019-12-31 ∪ operational 2020-09-23..2021-09-30). The pooled HUC2 fallback is unchanged. GFS (P1) has its own map on 2021-03-24..2021-09-30, unchanged. | P2 and P3 are the same GEFSv12 model and resolution. The stamped bias-rule sentence already names a single GEFS window running to 2021-09-30. Fitting P3 alone would rest on about one year of overlap. | every NWP-using arm |
| D2 | 2026-11-10 | GBM-NWP quantiles (interpretation) | §2b: "Quantile LightGBM (19 levels), not homoscedastic" | Config **Q**: LightGBM quantile objectives at 7 levels (5, 10, 25, 50, 75, 90, 95 %), sorted, then interpolated linearly in normal-score space (with linear tail extrapolation) to the 19 levels. Config **LS**: an L1 location model plus an input-dependent scale model, using the empirical standardised residual quantiles of the calibration slice. Both are among the reference's ≤ 5 parity configurations. | 19 quantile boosters per variable × horizon would multiply the CPU cost by about 2.7 over 7 levels, for no gain in the CRPS approximation at the inner levels. Both variants are heteroscedastic. | GBM-NWP, PF-GBM, E2-B |
| D3 | 2026-11-10 | NWP input encoding (interpretation) | §2a/§2c: masked mean over the available products, with no zero-fill | A product-lead is "available" when its temperature forecast exists. A field missing *within* an available product (GFS precipitation and shortwave before 2024-01-20) enters as 0 in standardised units, together with a per-field availability flag. A product that is unavailable at a lead is excluded from the masked mean. | GFS (P1) has temperature only before 2024. Dropping P1 whenever precipitation is missing would discard all of its 2021–2023 temperature information. | SONDE-HF, R-NH, Chronos-2-cov (covariate rows carry NaN, as Chronos-2 allows) |
| D4 | 2026-11-10 | Auxiliary maxima transform (interpretation) | §2d: SC daily max and turbidity daily max as numeric targets | Modelled in the same log space as their means (offsets 1.0 µS/cm and 0.1 FNU). They are registered in-process in `LOG_VARS`; the frozen `data.py` is unchanged, and no existing variable is affected. | These are heavy-tailed. Every model gets the same transform. Reports are in original units. | auxiliary targets only (secondary) |
| D5 | 2026-11-10 | Stage-2 / GBM training input (interpretation) | §2c: "stage 2 fine-tunes on real reforecasts over the train block" | Stage 2 and the GBM-type references use the bias-corrected real GEFS for train-block issue dates where it exists. Where it does not (the GEFSv11 gap 2020-01-01..2020-09-22), they use perfect-prog with the stage-1 error model. Early stopping uses the same stage's inputs at the inner hold-out. | Without this, the gap would remove 9 months of training rows from every model. The same rule applies to all comparators (parity). | all NWP-using arms |

## Log

| Date | Change | Reason |
|---|---|---|
| 2026-11-10 | v0.1 draft, D1–D5, written before any validation number on an affected arm | lead ruling L4 |
