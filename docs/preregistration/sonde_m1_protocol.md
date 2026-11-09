# M1 SENTINEL-Sonde — pre-registration protocol (forecasting, Test-A / Test-B)

**Status:** DRAFT v0.1 (2026-11-08). Implements RESEARCH_PLAN.md §5.1, §14, §15.2 (M1 primary).
Frozen together with, or after, the EW-v2 record (`ew_v2_protocol.md`); test blocks are scored once,
only after the lead authorises it. **Authors:** Austin Jin, Bryan Cheng (AI-assisted).

## 1. Benchmark (original track — always reported)
- Split `docs/preregistration/sonde_split.json` (sha256 `fd4bb14fe7304d05310e286e9aced22675fd6d07aa4dcac6415ed039a1e06a80`); stream sites (ST, ST-TS, ST-CA, ST-DCH); blocks Train WY2008–21, Val WY2022–23, **Test-A WY2024–25**, Test-B WY2026 (Approved values only), 42-day embargo.
- Cells: seen × Test-A, unseen (test-role) × train-period, **unseen × Test-A (headline)**, NEON × Test-A/B (if accessible).
- Variables: temperature, DO mean, DO min, SC, pH, turbidity (mean, median), NO3+NO2-N; horizons 1, 2, 3, 5, 7, 10, 14 d; 19 quantiles clipped to `PLAUSIBLE`.
- **Primary (§15.2):** median per-site CRPS skill of the frozen SENTINEL model vs the best of the plan references {damped anomaly, climatology, persistence, GBM (with weather), LSTM (with weather), zero-shot Chronos-2 uni/multi, zero-shot TimesFM-2.5} on unseen × Test-A, basin-group cluster bootstrap (B = 2,000), lower 95% bound > 0 at every h ∈ {1, 2, 3, 5, 7} for ≥ 3 of {turbidity, SC, nitrate, DO, temperature}, in each of 3 seeds. Holm across the five variables.
- **Strict control (reported next to the primary):** the same test against equal-weight ensembles of baselines only (GBM-w + LSTM(-w) + zero-shot Chronos-2 multi), so that an ensembling gain is not credited to SENTINEL.
- N-b virtual sensing: KGE improvement over GBM for ≥ 2 variables (basin-group CI > 0).

## 2. Freeze candidate
Candidate prepared 2026-11-08 (`sonde_m1_freeze_candidate.json`, NOT frozen): `ens3w` = equal-weight Chronos-2-FT + LSTM-weather + zero-shot Chronos-2 multi, seeds s0–s2, member checkpoint SHA-256s listed. On val the bar is **not** met on the unseen cell (SC 4–5/5 horizons, turbidity mean 4–5/5, nitrate 2–3/5, DO mean 2/5, DO min 0/5, temperature 0/5; 8–13 basin groups per variable). Freeze for Test-A is a lead decision.

## 3. Harder track M1-X (§15.5 amendment items 5–7; added 2026-11-08, before any result on it)
Saturation note: on familiar (train-role) sites, temperature and DO at h = 1 d are near ceiling (N-b temperature KGE ≈ 0.89 for both GBM and LSTM; all strong methods within CIs). The unseen-station cell is not saturated. M1-X adds, without replacing the original track:
- **X1 storm windows:** issue days whose trailing 3-day observed precipitation (gridMET, lagged as in `common.load_weather`) ≥ the site's training-block P95 of that quantity.
- **X2 heat windows:** issue days whose trailing 3-day mean tmax ≥ the site's training-block P95.
- **X3 extremes:** threshold-weighted CRPS with weight 1{target beyond the site × DOY training climatology by > 1.645 SD} (DO: lower tail). This is a proper score, not CRPS conditioned on the outcome.
- **X4 longer horizons (21, 30 d):** requires retraining, so it is planned only after the Test-A freeze.
- Same frozen split and blocks (the strata are row subsets defined from training-block forcings or climatology; definition code `scripts/eval/sonde_strata_power.py`, hashed at freeze). Power check on val **before** any model is scored on a stratum (2026-11-08, `sentinel_runs/sonde/m1x_strata_power_val/strata_power_val.csv`, h = 3 d; counts only): unseen (val-role) sites / basin groups in X1 storm — DO min 40/8, SC 97/12, temperature 166/13, turbidity mean 16/9, nitrate 7/3; X2 heat similar; X3 extremes — DO min 35/8, SC 65/12, temperature 157/13, turbidity mean 15/8, nitrate 0/0. Pre-specified minimum for a stratum claim: ≥ 5 basin groups with stratum rows at unseen sites → nitrate (all strata) and turbidity median are descriptive only. No model has been scored on any stratum. Claims name the track.
