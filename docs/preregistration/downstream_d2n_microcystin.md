# D2n — national lake microcystin track (enlarged track of D2 ToxinWatch)

**Status:** v1.0, written 2026-11-08 **before the split was frozen and before any model was fitted** on this
track. **Why:** RESEARCH_PLAN §15.5 amendment 2026-11-08 — on the Washington track (D2) the GBM challenger
ties the stronger INLA-class incumbent on known lakes (+0.010 [−0.001, 0.055]) and the new-lake set has
51 exceedances. This track is **added**; the WA track and its results are unchanged and always reported.
**Code:** `sentinel/models/downstream/wqp_track.py`, `sentinel/models/downstream/d2n_microcystin.py`,
`scripts/train/downstream_d2n_microcystin.py`. **Data:** `processed/wqp_discrete` (public domain),
gridMET `forcings_daily`, WBD `wbd_huc12`. No new download. NLA microcystin stays a separate,
cross-sectional check (D2 §5).

## 1. Labels and unit
- WQP `microcystins` (total microcystins / microcystins + nodularins, µg/L), accepted, non-QC, sites with
  `site_class == lake`; **Washington excluded** (D2 track's test lakes/years). Site-day label = any result
  > **8 µg/L** (EPA 2019 recreational value); '<DL' decides only if DL ≤ 8, '>DL' only if DL ≥ 8; other
  censored days are dropped.
- Lake cluster = union-find of stations ≤ 2.0 km; rows = cluster-days. Spatial group = HUC4 of the nearest
  HUC12 centroid; bootstrap unit = HUC8.

## 2. Split (frozen with `--freeze-split`; sha256 in §8)
Years: train 2006–2017, validation 2018–2020, test 2021–2025; 2026 reserve. HUC4 roles and row rules
exactly as D1n §2 (counts only; test 0.20, val 0.15, giant groups > 10 % → train); `val_known` primary rows
= clusters with ≥ 1 training row; `val_new` = lakes in never-seen HUC4s.

## 3. Decision time and features
Decision on the morning of sampling day *d*; toxin results take days → own and network results dated
≤ *d* − 7 only (as D2). Own history as D1n (EB rate, last value/exceedance/age, 30/365-d, 365-d max,
same-year), network: other clusters ≤ 50 km with results in [d−36, d−7]; gridMET windows ending d − 1.

## 4. Models and comparators (same 4-configuration budget each)
- Simple baselines: climatology, location-only, lake history rate (EB ≤ d − 7), persistence (last result
  ≤ 30 days old at d − 7).
- **Incumbent (re-implemented): INLA-class spatiotemporal model**, both D2 variants (`st_glmm_incumbent`
  v1 and `_v2`, identical terms to `toxinwatch.StGlmm`, sparse lake random intercept), L2 C ∈ {0.01, 0.1,
  1, 10}; the stronger on validation (per set) is the incumbent for the claim.
- Challengers: L2 logistic, LightGBM (grids as D1n).
- Selection: inner split of training years (fit 2006–2014, score 2015–2017), AUROC.

## 5. Endpoints and statistics
As D1n §5 (AUROC on `val_known` and `val_new`; HUC8 paired bootstrap; Holm; claim on both test sets vs the
incumbent and the strongest simple baseline, ≥ 3 seeds). A-priori strata: cold-start lakes (no prior
result ≤ d − 7), hot weeks (7-d Tmax anomaly > training P90).

## 6. Power check (from `--build-only`, before any fit) — §8

## 7. Iteration
As D2 §6. Test scored once per frozen configuration; 2026 reserve.

## 8. Frozen split, counts and power check
- **Split frozen 2026-11-08:** `downstream_d2n_microcystin_split.json`, **sha256 `0d9eb684…d4176295`**
  (full: 0d9eb684285155c7d73a8454b3b211aea626e5bb6b1bc809970d55ded4176295); HUC4 roles train 70 / val 61 /
  test 62; 4,567 lake stations → 3,160 clusters (2 km); 49,935 eligible site-days 2003–2026.
- **Phase-1 frame:** train 12,357 cluster-days / 751 exceedances / 851 clusters / 70 HUC4; `val_known`
  5,997 / 314 / 405 clusters / 118 HUC8; `val_new` 1,273 / 57 / 164 clusters / 68 HUC8. gridMET missing 29 %.
- **Power check:** AUROC SE at 0.75 = 0.016 (`val_known`), 0.038 (`val_new`); paired MDE ≈ 0.041 / 0.095.
  `val_known` is ~1.5× the WA known-lake set; `val_new` is no larger than WA's (57 vs 51) → new-lake
  validation remains **underpowered**; the test block (2021–2025, all states but WA) is the larger one.
- **Smoke test disclosure:** code exercised once on TRAINING years only (fit 2006–2014, score 2015–2017,
  the inner block) before phase 1; no validation row scored; no change afterwards.

## 9. Iteration log (validation only)
- **phase 1** (run `d2n_microcystin/phase1`, 2026-11-08; 16 configurations, 4 per learned model).
  Selected: logistic C 0.01; GBM 7 leaves / min_child 100; incumbent v1 C 0.1, v2 C 0.01. AUROC (HUC8
  paired CIs):
  | set (n / pos) | history rate | persistence* | incumbent v1 | **incumbent v2** | logistic | GBM | GBM − v2 |
  |---|---|---|---|---|---|---|---|
  | val_known (5,404 / 289) | 0.830 | 0.899 | 0.877 | **0.905** | 0.894 | 0.898 | −0.007 [−0.020, 0.000] |
  | val_new (1,273 / 57) | 0.754 | 0.919 | 0.831 | **0.901** | 0.898 | 0.902 | +0.001 [−0.019, 0.023] |
  *persistence scored on rows with a result ≤ 30 d old at d − 7 only. GBM beats climatology, location,
  history rate (all sets) and v1 (new lakes +0.071 [0.035, 0.117]); GBM − persistence +0.030
  [0.012, 0.053] (known), +0.006 [−0.007, 0.027] (new). Strongest simple baseline (all rows): history rate.
  A-priori strata: hot weeks (7-d Tmax anomaly > train P90) known 212 / 7 exceedances, GBM 0.940 vs
  v2 0.944; cold-start new lakes 170 / 3 (uninformative). **Verdict: competitive** — the national
  track reproduces the WA tie with the AR(1)/spline INLA-class incumbent rather than resolving it.
