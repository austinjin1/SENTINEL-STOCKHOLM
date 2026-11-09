# SENTINEL-Orbit (M2) — Pre-registration protocol

**Status:** v0.4, 2026-11-09 (v0.4 adds the event track E4a/E3, §12, and the code-level single-look test procedure, §9b; both written before any E4a > 20 µg/L validation metric was computed and before any test look). v0.3, 2026-11-08 (v0.2 of 2026-10-24 was written before any SENTINEL-Orbit model was trained
or selected; only smoke tests had run). v0.3 changes, all made before any test scoring and before any model
result on the new tracks: the turbidity unit rule (§2.1), data versions dev_v4/dev_v5 (§2.3), the power
analysis (§5a), the Test-B cell + pooled analysis (§9a) and the harder tracks U and E (§11, RESEARCH_PLAN §15.5 amendment of 2026-11-08). The
headline cell, H1 and the split rule are unchanged.
- The **split rule** is frozen as **v3**: `docs/preregistration/orbit_split_v3.json`, sha256
  **`ffe288f2fafecada964fc8125dcd35ddd9b332d3a04f782d0298e01f097a5e32`**.
- v3 supersedes v2 (`orbit_split_v2.json`, sha256 `06e5067d…b70e`), which superseded v1
  (`orbit_split.json`, sha256 `27854499…6437`). All three were written on 2026-10-24, before any model
  selection, and only smoke tests had run. The changes follow the pre-training review
  (`REVIEW_REPORT_PRE_TRAINING.md`, M2 Orbit: C1 and N1). Lake-block roles are identical in all three
  versions. Superseded files are kept for provenance.
- The **test configuration** is frozen later, when `docs/preregistration/FREEZE_orbit.json` is written
  (§9). That file does not exist yet. This protocol is part of the hashed frozen code
  (`sentinel/models/orbit/guard.py::frozen_files`), so any edit after the freeze invalidates the freeze.

**Authors:** Bryan Cheng (AI-assisted). **Implements:** RESEARCH_PLAN.md §5.2, §14 and §15 (M2 primary
endpoint of H-G1; §15.5 amendment of 2026-10-24).

## 1. Objective and hypotheses
We test whether a model fed Sentinel-2 L2A imagery retrieves in-situ water quality at **lakes it has
never seen, in years after its training data**. It must also beat the strongest published retrieval
method retrained on the same data and the strongest simple baselines.

- **H1 (primary).** On the headline cell (**in-lake rows** of test-role water bodies × 2024–2025),
  SatWQ-Net's chl-a MdSA
  is lower than **each** of three comparators:
  - (a) MDN retrained on our matchups (MDN_B);
  - (b) the best GBM on imagery + season/location;
  - (c) location kNN.

  Each comparison uses a paired **role-block** bootstrap 95% CI of the MdSA difference, and every CI must
  exclude 0 (the water-body bootstrap is a reported sensitivity) (intersection–union test, so no multiplicity correction is needed within H1). The gain
  must also hold for each of ≥ 3 seeds individually (point estimate), as §15.1 requires.
- **H2 (secondary).** The same test for TSS, turbidity and Secchi depth, with Holm correction across
  these three targets.
- **H3 (secondary tracks, descriptive).** Within-lake temporal skill, bloom-class detection with a
  placebo, seen-lake × future-year skill, interval calibration, and a GLORIA sanity check (§7).

## 2. Data and split
### 2.1 Matchups
**Sources.**
- AquaCorr-Global S2 rows with SENTINEL's own L2A chips (`processed/s2_chips/aquacorr`, extraction
  still in progress; in-situ values from AquaCorr).
- WQP WQX3 surface samples matched to S2 (`processed/s2_wqp_matchups`, chips
  `processed/s2_chips/wqp`; the national 2024–2025 rebuild is pending).

**Rules** (`sentinel/models/orbit/data.py`).
- One row per (location, in-situ date, S2 acquisition), **same day only** (primary window).
- Image QC uses imagery only (no labels):
  - chip extracted, with the same acquisition as the AquaCorr scene;
  - ≥ 5 SCL-water pixels in the centre 3 × 3;
  - JRC water test passed;
  - no radiometric `offset_suspect` flag;
  - no SCL cloud, cirrus or shadow in the centre 3 × 3;
  - water-median B02 ≤ 0.15 and B11 ≤ 0.05 (glint/haze/land screen).

  AquaCorr's own `passed_qc` is not trusted, since it does not screen cloud or land.
- **Targets.**

  | Target | Unit | Valid range | Notes |
  |---|---|---|---|
  | chl-a | µg/L | 0.05–3,000 | no in-vivo/sensor values; AquaCorr outlier flags removed; WQP lab names only |
  | TSS | mg/L | 0.1–5,000 | |
  | turbidity | NTU (v0.3) | 0.05–5,000 | see the unit rule below |
  | Secchi | m | 0.05–40 | |

  Values outside the valid range are set missing, never clipped.
- **Turbidity unit rule (v0.3; replaces the v0.2 pooled NTU/FNU target).** NTU (EPA 180.1 white light) and
  FNU (ISO 7027, 860 nm) are not interconvertible by a fixed factor, so they are never mixed
  (CONTRIBUTING_V2). `data.apply_turbidity_unit_policy` classes each sample-date as NTU family (NTU, ratio
  NTRU), FNU, mixed (both units on one sample-date) or unspecified (AquaCorr states no unit).
  - The modelled and scored target `turbidity` = NTU family + AquaCorr-unspecified values. Unspecified values
    remain a documented limitation; they occur only in AquaCorr rows, and the headline test cell is WQP-only,
    so its turbidity rows are all NTU family.
  - FNU values are moved to `turbidity_fnu`. They are never fitted and are reported as a secondary stratum
    (cross-unit transfer, descriptive). `turbidity_all_units` keeps the pooled v0.2 value as a sensitivity.
  - Effect on development: dev_v3/dev_v4 contain only AquaCorr turbidity (unspecified), so their results are
    unchanged. The teacher manifest for M7 labels turbidity "NTU family; AquaCorr unit unstated".
- **De-duplication.**
  - Distance-based (review W5). AquaCorr rows within 20 m of each other on the same date and platform
    are one sample: they are merged, and each target takes the median.
  - A WQP row is preferred over an AquaCorr row within 100 m on the same date.
  - Multiple stations of one lake on one date stay together automatically, because roles are per
    water body.

### 2.2 Split rule v3 (frozen; `sentinel/models/orbit/split.py`)
1. **Lake water bodies.** A point inside a HydroLAKES v1.0 polygon, or within **500 m** of one, belongs to
   that lake (`L<Hylak_id>`).
   - The anchor is the lake's pour point.
   - The role block is `spatial_blocks(anchor, 50 km)`. Every station of a lake shares one block, so
     Lake Erie is a single unit.
   - Lake roles are identical to v1, because the salt and the block size are unchanged.
2. **Non-lake points** (rivers, estuaries, ponds < 10 ha, coast).
   - The role block is a **200-km** block of the point, prefixed `N`, so estuaries and river reaches
     stay together. In v1, Chesapeake and Tampa Bay were split across roles.
   - The unit name is `P<200-km block>|<10-km cell>`, nested in its block.
3. **Buffer.** These rows are `buffer_excluded`, never fitted and never scored:
   - (a) a non-lake point within 1 km of a HydroLAKES lake of a different role;
   - (b) (v3, review N1) a lake point outside its polygon (captured within 500 m) whose 200-km
     non-lake block has a different role from the lake. Estuary stations next to tiny polygons can
     no longer carry another role's water into training.
4. **Role.** u = SHA-256(salt | block) mapped to [0, 1):
   - u < 0.2 → **test**
   - u < 0.4 → **val**
   - otherwise → **train**
5. **Years.**
   - ≤ 2021 → train
   - 2022–2023 → val
   - 2024–2025 → **test**
   - ≥ 2026 → test_b
6. **Cells** (in-lake rows mirror the headline; buffer rows excluded everywhere).

   | Cell | Definition |
   |---|---|
   | Fit | train role × train years (in-lake and non-lake), minus inner blocks |
   | Inner (early stopping, baseline tuning) | 15% of train-role blocks (salted hash) × train years |
   | **Model selection** (`valxval_lake`) | in-lake rows, val role × val years |
   | TSS/turbidity selection (`valxany_lake`) | in-lake rows, val role × (train + val years); the val-year cell holds < 50 rows for these targets |
   | Secondary: non-lake (`valxval_nonlake`) | val role × val years, non-lake rows |
   | Secondary: seen lakes, future years (`trainxval_lake`) | train role × val years, minus inner blocks |
   | **Headline** | in-lake rows, test role × test years |
   | Secondary (test) | non-lake test rows; train role × test years |
   | Sensitivity (test) | headline rows more than 10 km from any train-role point |

   **Protected:** every test-role row (in any year) and every row from 2024 onward.
7. `spatial_blocks` is vendored into `split.py`. `verify_split_file` refuses to run if the rule code
   hash differs from the frozen file, and `data.load_dev` recomputes every row's role and block and
   asserts that they equal the cache (review C3).

The rule depends on coordinates, dates and HydroLAKES only, never on labels. Rows added later (the
national WQP 2024–2025 rebuild, more AquaCorr chips) therefore receive their roles from the same
frozen rule. Distance reports (nearest train point for val and test points) are saved in
`docs/preregistration/orbit_split_report_<data version>.json`.

### 2.3 Data versions
- The development snapshot `dev_v3` (`$SENTINEL_RUNS/orbit/cache/dev_v3`, manifest `orbit_dev_v3`;
  `dev_v1`/`dev_v2` used the superseded splits and were only used for smoke tests) contains train/val roles × train/val years only. Protected rows are dropped right after split
  assignment and are never written.
- If more chips arrive (the AquaCorr extraction is still running), a new version `dev_vN` may be built
  under the same rule. All models in a comparison use the identical version.
- The data version used for the frozen configuration is written to FREEZE_orbit.json.
- **dev_v4** (2026-11-08): all 117,889 extracted AquaCorr chips (dev_v3 used 68,102) plus the national WQP
  2024–2025 matchups (`s2_wqp_matchups`). Every WQP row of that build is in test years, so it is protected and
  only its counts enter the split report.
- **dev_v5** (pending): dev_v4 plus national WQP 2015–2023 matchups (`s2_wqp_matchups_hist`, built by the
  same script and rules with `--out-name s2_wqp_matchups_hist`). This adds same-source (WQP) rows to fit and
  val, which addresses review W4 (the dev source differed from the WQP-only test source). The final
  configuration is selected on the val cell of the latest version, and all comparators are refitted on it.

## 3. Inputs (enforced in code)
- **Imagery.**
  - 12 L2A surface-reflectance bands (B01–B12 without B10; harmonised BOA offset).
  - SCL water mask; centre water pixels (up to 64 nearest valid SCL-water pixels in the centre
    15 × 15 = 150 m).
  - Water-masked centre medians (3 × 3, 7 × 7).
  - Sun/view geometry and AOT.
  - The JRC distance to land.
- **Context (metadata).** DOY (sin/cos) and location (3-D unit vector of lat/lon).
  - In SatWQ-Net the whole metadata vector is dropped at random (p = 0.5) during training.
  - An imagery-only variant is also reported.
- **Never used:**
  - in-situ values of the scored row or of any val/test-role water body;
  - lake identity;
  - site means of unseen lakes.

  The seen-lake site mean is used only in the secondary seen-lake track.
- **Normalisation, climatologies, kNN neighbours and index calibrations** are fitted on the fit cell
  only.

## 4. Models
### 4.1 SatWQ-Net (SENTINEL) — `sentinel/models/orbit/satwq.py`
- **Spectral-token encoder.**
  - Each band of each water pixel becomes a token: an embedding of the asinh-scaled reflectance plus a
    sinusoidal embedding of the band's SRF-centroid wavelength (per platform, S2A/S2B; S2C uses the
    S2A values).
  - A Transformer runs over the band tokens; missing bands are masked.
  - Bands are dropped at random during training, including an explicit RGB+NIR (B02/B03/B04/B08)
    subset mode, so the same weights can run on drone bands.
- **Attention pooling** over valid water pixels, with a pixel-offset embedding. During training,
  random pixel subsets (1…64 pixels) are sampled, so per-pixel inference is in distribution.
- **Metadata branch** with whole-vector dropout.
- **Head.** Mixture-density head: K = 5 Gaussians per target in log10 space, with a masked negative
  log-likelihood loss.
- **Training.**
  1. Pretrain on GLORIA (S2A-simulated bands as reflectance ρ = π·Rrs). Augmentations: multiplicative
     band noise and additive offsets that mimic atmospheric-correction error. GLORIA samples in
     val/test-role water bodies, inner blocks or buffer zones, or dated ≥ 2022, are excluded.
  2. Fine-tune on the fit cell.
  3. Early stopping on the inner split: 15% of train-role water bodies, chosen by a seeded hash of
     the block (`inner_seed = 0`).
- **Evaluated model.** The ensemble of ≥ 3 seeds; the reporting target is 5.
  - Seed mixtures are averaged with equal weights.
  - Point estimate = median of the predictive mixture (log10).

### 4.2 Baseline ladder (all fitted on the identical fit cell and scored on identical rows)
1. **Trivial context.**
   - Global training median.
   - **Location kNN**: k = 8, inverse-distance weighted over log10 values of train-role points
     (haversine). k is chosen on the inner split from {4, 8, 16, 32}.
   - **Season + location GBM** (LightGBM on lat/lon and DOY sin/cos, no imagery).
   - **Seen-lake site mean** (secondary track only).
2. **Index algorithms**, calibrated on the fit cell ("-opt"; polynomial in log space):
   - chl-a: OC3-MSI (max(B1, B2)/B3), NDCI ((B5 − B4)/(B5 + B4)), the 2-band ratio B5/B4, MCI (705 nm line height; Salls et al. 2024);
   - TSS and turbidity: single-band red (B4) and NIR (B8A) log-log fits;
   - Secchi: blue/red and green/red log-log fits.

   The best index per target on the inner split is the representative.
3. **MDN.** All variants below use our re-implementation of the Pahlevan et al. (2020) architecture
   with the preprocessing of the official MDN-STREAM MSI configuration: 5 × 100 ReLU layers, K = 5,
   RobustScaler on reflectance, ln + MinMax(−1, 1) on the target, 10,000 iterations, batch 128,
   10 inits; the estimate is the top component, taken as the median over inits.
   - **MDN_GLORIA**: retrained on our GLORIA pretraining rows with S2 B1–B7 Rrs input, and applied to
     L2A water-median ρ/π.
   - **MDN_B**: retrained on the fit cell, with L2A water-median ρ as input. The 7-band (B01–B07) or
     12-band variant is chosen per target on the inner split. This is the strongest-published-method
     comparator.
   - **mdn_official**: the official MDN-STREAM MSI weights (joint chl/TSS/CDOM, 10 rounds), applied
     to L2A ρ/π of B01–B07, for chl-a and TSS only. The weights were obtained on 2026-10-25 (datasheet
     `mdn_stream_weights.md`) and are never scored on GLORIA.
4. **GBM** (LightGBM) on band statistics (3 × 3 and 7 × 7 water medians, band ratios, geometry, AOT):
   - **imagery-only**;
   - **imagery + season/location** (comparator b).

   Hyperparameters come from a grid on the inner split. **Tuning parity:** every learned baseline
   family gets at least as many inner-split configurations as SatWQ-Net gets.
5. **Ablations** (secondary): SatWQ-Net without GLORIA pretraining; imagery-only; RGB+NIR-only;
   no pixel set (centre median only).

## 5. Metrics (`sentinel/models/orbit/evaluate.py`, built on `sentinel.eval.metrics`)
- **Primary:** MdSA (%) = 100 × (exp(median |ln(e/o)|) − 1).
- **Secondary:** SSPB, log10 slope, RMSLE (log10), log-MAE, log R², N and n_waterbodies.
- **Intervals:** 68% and 95% central-interval coverage and width for distributional models.
- **CIs:** the **role-block** cluster bootstrap is primary (B = 2,000, percentile; the 50-km lake
  block or the 200-km non-lake block is the unit the split assigns, RESEARCH_PLAN §15.1). The
  water-body cluster bootstrap is a sensitivity (review W1).
- **Paired comparisons:** paired bootstrap of the MdSA difference on the same rows and the same
  clusters. Role blocks are primary; water bodies are a sensitivity.
- **Scored rows:** rows where every compared model has a finite prediction. Each model's dropped rows
  are counted.

## 5a. Power analysis (v0.3, review W4)
`scripts/train/orbit_power.py` uses only validation per-row predictions and the **counts** of the protected
headline cell (`protected_cell_counts` in the split report; rows, water bodies and role blocks per target).
Pseudo-test samples of B_test role blocks are drawn from the val blocks. The SD of the paired MdSA difference
across them is the expected test SE; the script reports the MDE at 80% power (2.80 SE) and the power at the
val point difference, per comparator and jointly. It is registered as `orbit_power_<version>` (a design
quantity, not a result).

## 6. Model selection and iteration (validation only)
- All selection and iteration (RESEARCH_PLAN §15.5 amendment) uses the val cell; the headline cell
  is never touched.
- Every configuration evaluated on val is appended to `$SENTINEL_RUNS/orbit/configs_tried.jsonl`. The
  count is reported next to any claim.
- **Data-level changes** (QC, window, inputs) are applied to every model, and the baselines are
  re-fitted on the same data version.
- **Comparators (tuning parity, review W2).** Each comparator family is selected on the same val cell
  as SatWQ-Net:
  - MDN_B: grid of {7, 12 bands} × {5k, 10k, 20k iterations} × {linear, log input};
  - GBM imagery + context: 16-configuration grid, with the inner-best config refitted as a 5-seed ensemble;
  - location kNN: k ∈ {4, 8, 16, 32}.

  Within each family, the member with the lowest val MdSA is the H1 comparator. This is optimistic
  for the baselines and conservative for SENTINEL. The inner-selected representatives are also
  reported. The selected comparators are written to FREEZE_orbit.json `comparators` before test scoring.

## 7. Secondary tracks
- **T, within-lake temporal skill.**
  - Eligible: water bodies with ≥ 8 rows in the scored cell.
  - Metric: pooled Spearman ρ of (obs − lake mean obs) vs (pred − lake mean pred) in log10.
  - Comparators: the same models; season-only models are the reference.
- **B, bloom classes.**
  - Chl-a ≥ 10 µg/L and ≥ 50 µg/L (WHO 2003 recreational guidance: low/moderate probability of
    adverse effects, ≈ 20,000 / 100,000 cyanobacterial cells/mL).
  - Score: the predictive P(chl-a ≥ threshold).
  - Metrics: AUROC, AUPRC and recall at the FPR = 10% threshold, with the threshold frozen on val.
  - **Placebo:** shuffle the imagery among rows of the same water body and month. The score must drop.
- **S, seen lakes × future years** (train role × val years, later × test years). The seen-lake site
  mean and location kNN are the references.
- **R, GLORIA sanity.** Leave-dataset-out MDN vs the SatWQ-Net spectral encoder on the GLORIA
  pretraining rows.

## 7a. Claim labels
These follow RESEARCH_PLAN §15.1.
- **"State of the art"** requires all of:
  - H1 holds on the headline cell;
  - the gain holds for ≥ 3 seeds;
  - the test was scored once after the freeze.
- Otherwise the result is "competitive" (CI overlaps) or "negative".
- Val-only results are labelled **"validation (development)"**. They are never claims.

## 8. Teacher for M7 (SENTINEL-Lite)
- The frozen SatWQ-Net ensemble provides per-chip and per-pixel predictive distributions (log10 mean,
  variance and quantiles) for all S2 bands and for the RGB+NIR subset.
- The interface and checkpoint hash are in `docs/model_cards/orbit.md`.
- Teacher predictions use imagery only, never labels.

## 9. Test access
- `sentinel/models/orbit/guard.py`: training code refuses protected rows unconditionally.
- Evaluation code needs `--allow-test` **and** `FREEZE_orbit.json` with these keys:
  - `created`;
  - `split_sha256` (= split v3);
  - `data_version` and `data_manifest_sha256`;
  - `scoring_code_sha256`;
  - `weights_sha256` (per model and seed);
  - `comparators`.
- The headline cell is scored **once**.

## 9b. Single-look enforcement in code (v0.4, 2026-11-09)
`scripts/eval/orbit_test.py` with `sentinel/models/orbit/testlook.py` (pattern of `sentinel/models/hydromet/testlook.py`).
Looks: `M2/test` (headline + Test-B + pooled + track E strata + event track §12) and `M2U/test` (track U).
1. **Only `docs/preregistration/FREEZE_orbit.json` unlocks protected rows.** `guard.load_freeze` refuses any other
   path (including `orbit_freeze_candidate.json`), refuses `status` other than `FROZEN`, requires `authorised_by`
   and `authorised`, and requires `candidate_sha256` = sha256 of the reviewed candidate file, whose keys (except
   `status`) must equal the freeze's. Only the lead writes FREEZE_orbit.json.
2. **Registry refusal.** A look refuses to start if `results/registry.jsonl` already holds any row of its
   experiment (`orbit_TEST_<version>` / `orbit_TEST_<version>_trackU`).
3. **Content-hash data pins** (`data_pins` in the freeze): sha256 over (relative path, file sha256) of every
   source file the protected loader reads (WQP matchup tables, AquaCorr request/plan tables, chip-store indexes and
   shards, HydroLAKES, WBD HUC12). Verified **before** the claim, so a data mismatch never burns the look.
4. **Claim.** An O_CREAT|O_EXCL, 0444 marker `$SENTINEL_RUNS/orbit/test_looks/<look>/TEST_LOOK.json` plus a
   `look_claimed` registry row. Protected rows are loadable only while an unscored claim of this freeze exists.
5. **Sealed predictions.** All per-row predictions (all models, all cells, event probabilities) are written once,
   chmod 0444, with `SHA256SUMS` and an O_EXCL `PREDICTIONS_SAVED.json`. Scoring reads only the sealed,
   sha-checked files.
6. **SCORED marker.** After the registry rows, an O_EXCL `SCORED.json`; afterwards nothing can be re-scored.
   Crash recovery between 5 and 6: `--rescore-from-saved` scores the sealed predictions (never re-predicts).
7. The development dry run (`--dry-run-dev`) runs steps 4–6 on the val cell under a separate namespace
   (`test_looks_dryrun/`), without a freeze and without protected rows.

## 9a. Test-B cell and pooled analysis (v0.3, declared 2026-11-08, before any test look)
**Why.** The power analysis (§5a) shows that the headline chl-a cell is under-powered: 669 rows in 87 blocks,
MDE about 10–14 MdSA points. A second test cell is therefore declared now, at the lead's direction and
before any test look. **chl-a stays the pre-registered primary target; Secchi stays secondary.**
- **Test-B cell.** In-lake, non-buffer rows of **test-role** water bodies dated 2026-01-01 to 2026-09-30.
  Under the frozen split rule these rows are in the `test_b` time block.
  - This is the part of water year 2026 not already in the 2024–2025 headline. Oct–Dec 2025 rows of WY2026
    already belong to the headline cell and are not counted twice.
  - Source: WQP national, built with the same script and rules
    (`build_s2_wqp_matchups.py --start 2026-01-01 --end 2026-09-30 --out-name s2_wqp_matchups_2026`).
  - All of its rows are protected. Only counts are read before the freeze.
- **Reporting.** Each is reported separately, with the same models, comparators, metrics, strata and CIs as
  the headline. The order is fixed: (1) the headline (H1 as registered), (2) Test-B, (3) the pre-declared
  pooled cell (headline ∪ Test-B, role-block bootstrap over the union).
  - H1 is decided on the headline cell as registered.
  - The pooled result is a pre-declared secondary analysis. It is never substituted for H1.
  - If the pooled analysis is also under-powered, the claim says so.
- **Projected sizes** (row counts only). Headline count × (WQP Jan–Sep 2026 rows / WQP 2024–2025 rows in
  `processed/wqp_discrete`, all sites):

  | Target | WQP ratio | Test-B rows | Pooled rows |
  |---|---:|---:|---:|
  | chl-a | 0.106 | ≈ 71 | ≈ 740 |
  | Secchi | 0.096 | ≈ 138 | |
  | TSS | 0.097 | ≈ 21 | |
  | turbidity (NTU) | 0.106 | ≈ 32 | |

  WQP 2026 reporting is incomplete (reporting lag; latest record 2026-10-20). The pooled cell lowers the
  chl-a SE by at most about 5%, so it is expected to remain under-powered for gains of about 5 MdSA
  points. Actual counts go in the dev_v5 split report (`protected_cell_counts`, key `testxtest_b_lake`).

## 10. Known limitations at registration
- AquaCorr units are unstated (assumed: chl-a µg/L, TSS mg/L, Secchi m, turbidity NTU or FNU).
- AquaCorr chip extraction is incomplete and US-only so far.
- The national WQP 2024–2025 test matchups are not built yet (MN only).
- The official MDN weights are TensorFlow/Keras files. They are read as data, and the forward pass
  is re-implemented in NumPy; on GLORIA it reproduces the published accuracy (not independent).
- GLORIA contains some US lakes. Only train-role, ≤ 2021 samples are used.

## 11. Harder tracks (v0.3, 2026-11-08; RESEARCH_PLAN §15.5 items 5–7)
**Why.** On the v0.2 benchmark every strong method ties on the val cell: chl-a, about 800–2,000 rows in
about 55 role blocks, with CIs of about ±10 MdSA points. Under the §15.5 definition that makes the task
saturated by a small, easy cell.
- The national WQP data enlarge the benchmark (§2.3).
- Two tracks are added. The original headline (§1) is always reported alongside, and every claim names its
  track.
- Both tracks were defined and frozen before any model result on them was seen. The rule file is
  `docs/preregistration/orbit_track_u.json`, sha256
  `953948b7f0251057728adfc4d9a75065e1d0324c9ec3536eb6dae99ae957b2cf`.
- Neither track has its own val cell, so **no model selection happens on either**. The configuration and
  the comparator members selected on the main val cell are used unchanged (track U refits them on U-fit).

### 11.1 Track E: bloom extremes (a-priori difficulty strata)
- **Strata.** Headline-cell rows (and U-test rows) with observed chl-a ≥ 20 µg/L, ≥ 50 µg/L, and ≥ the
  fit-cell 95th percentile of dev_v4 (54.9 µg/L, frozen in the rule file).
- **Metrics.** MdSA and SSPB within each stratum (H1 comparators, paired role-block bootstrap), and bloom
  detection at 20 and 50 µg/L (AUROC, AUPRC, recall at FPR = 10% with the cut frozen on val).
- **Projected sizes.** These use the headline count × the val-cell exceedance fraction, so no test label is
  used: about 179 rows at ≥ 20 µg/L and about 43 at ≥ 50 µg/L. The ≥ 50 stratum is labelled preliminary
  (Clopper–Pearson for detection counts < 30).

### 11.2 Track U: unseen hydrologic regions × unseen years (stricter extrapolation)
- **Rule** (`sentinel/models/orbit/track_u.py`). A row's region is the HUC2 of the WBD HUC12 centroid
  nearest to its split anchor (the lake pour point for lakes).
- **Held-out regions.** The 4 CONUS regions with the lowest salted hash: **02 Mid-Atlantic, 06 Tennessee,
  09 Souris-Red-Rainy, 18 California**.
- **U-fit.** The main fit cell minus held-out-region rows and a 25-km buffer: 8,029 of 11,661 dev_v4 fit rows.
- **U-test.** In-lake rows of held-out regions × 2024–2025, any main role.
- **Counts** (`orbit_tracks_counts_dev_v4.json`, counts only):

  | Target | Rows | Water bodies | Role blocks |
  |---|---:|---:|---:|
  | chl-a | 617 | 185 | 69 |
  | Secchi | 876 | 264 | 70 |
  | turbidity (NTU) | 279 | 68 | 40 |
  | TSS | 145 | 50 | 34 |

- **Endpoint** (secondary, named "track U"). The same H1 comparison: SatWQ-Net vs MDN_B, GBM imagery +
  context and location kNN, all refitted on U-fit, ≥ 3 seeds, scored once after the freeze.

### 11.3 Track L: cross-sensor Landsat 8/9 transfer (proposed, not started)
- **Design.** Apply the band-agnostic (wavelength-tagged) SatWQ-Net trained on S2 to Landsat 8/9 OLI C2 L2
  chips at the same matchups, against MDN/GBM retrained on Landsat.
- **Prerequisite.** A Landsat chip extraction (Planetary Computer C2 L2): about 35 kB per chip, so about
  1–3 GB for 30–80k matchups. It needs its own protocol section, frozen split reference and power check
  before any model selection.


## 12. Event track (v0.4, 2026-11-09): (B) harmful-event detection from the same model
**Why.** User requirement (2026-11-09): every model outputs (A) water quality and (B) which harmful event is
happening. Shared definitions: `docs/plans/event_layer_plan.md` §1 and `sentinel/eval/events.py` (imported,
never copied). The track extends §7 B / §11.1 (bloom classes) rather than duplicating it. **State only, no onset**
(S2 revisit ≥ 5 d, cloud gaps).

**Pre-specification status.** Written before any E4a (> 20 µg/L) Brier/AUPRC/ECE was computed on any val cell.
Already seen before this section (disclosed): AUROC/AUPRC of median scores at 10 and 50 µg/L on dev_v3/dev_v4
val (`compare_*/bloom.csv`), and the label counts in the event plan §3.2.

### 12.1 Events
| Event | Definition | Population | Status (dev_v4 val) |
|---|---|---|---|
| **E4a** (primary event) | chl-a > 20 µg/L (`events.EVENTS["E4a"]`) | in-lake rows | F: 309 positives / 90 lakes / 36 blocks in `valxval_lake` |
| E4a tiers (sensitivity) | > 12 µg/L (WHO 2021 AL1), ≥ 50 µg/L | in-lake | F |
| **E3** sediment plume (descriptive) | TSS > 80 mg/L; turbidity > 50 NTU (NTU family only, §2.1) | in-lake, then all rows | **U**: 0 val positives. Derived route only, counts + metrics if any positives; never a claim |

### 12.2 Routes (both pre-registered; fixed before val results)
- **Primary: derived.** P(chl-a > 20) = exceedance of the frozen SatWQ-Net equal-weight seed-ensemble mixture
  (the same mixture that gives the (A) median; context on), `events.exceedance_probability` on the mixture.
- **Challenger: direct head.** L2 logistic regression on the standardised 64-d fused embedding (`model.fuse`
  output, read by a forward hook; the network is not modified) of each frozen seed, fitted on fit-cell E4a labels;
  C ∈ {0.01, 0.1, 1, 10} chosen by inner-cell log-loss; P averaged over seeds.
- **Recalibration.** Every route and every comparator gets one Platt recalibration (`events.recalibrate`) fitted
  on the full `valxval_lake` cell and frozen into the event artifact. Val metrics are reported **uncalibrated** and
  **cross-fitted** (5 folds grouped by water body; each fold's calibrator fitted on the other folds). Test uses the
  frozen full-val calibrators.

### 12.3 Comparators (all fitted on the fit cell; tuning on inner; same recalibration)
1. **Band-ratio / OC chl**: `index_opt` (best calibrated OC3/NDCI/B5:B4/MCI polynomial on inner) → P via a
   Gaussian in log10 with the inner-cell residual SD.
2. **Strongest ML regression baselines' P**: MDN_B (val-selected chl member) mixture exceedance; GBM imagery +
   context (val-selected member) → P via a Gaussian with its inner residual SD.
3. **GBM direct classifier** (LightGBM binary, imagery + context features; 8-configuration grid chosen by inner
   log-loss): the strongest direct ML comparator.
4. **Spectral cyanobacteria-index analogue** (labelled analogue): CI_S2 = −SS(665) = −[ρB4 − ρB3 − (ρB5 − ρB3) ·
   (665 − 560)/(705 − 560)], the Wynne et al. (2008) line-height form moved to S2 bands (S2 has no 681/620 nm
   band, so the Lunetta et al. (2015) SS(665;620,681) cyanobacteria discriminator cannot be computed). Also
   MCI (Gower et al. 2005; SS at B5 between B4/B6) and NDCI (Mishra & Mishra 2012). Each raw index (water median
   3 × 3) → univariate logistic fitted on the fit cell → P.
5. **Climatology**: LightGBM binary on DOY sin/cos + location unit vector (fit cell; the event analogue of
   `gbm_season_location`), plus the constant fit-cell prevalence.
6. **CyAN CI-cyano** (NOAA/EPA): **not compared** — needs an Earthdata login the user has not provided
   (`docs/datasheets/labels_cyan_availability.md`).

### 12.4 Metrics, CIs, endpoints
- `events.state_metrics`: AUPRC (prevalence = chance), Brier, BSS vs climatology (5) and vs constant prevalence,
  ECE (10 bins), reliability table; plus AUROC and recall at FPR = 10% (cut frozen on val) for continuity with §7 B.
- CIs: **water-body cluster bootstrap** (B = 2,000) primary for the event track (user direction 2026-11-09);
  role-block bootstrap as a sensitivity.
- **E-H1 (event skill, the H-G1 "disaster-relevant target" check):** on the headline cell, recalibrated derived
  P has BSS vs climatology > 0 with the CI excluding 0.
- **E-H2 (event comparison, secondary):** paired ΔBrier and ΔAUPRC (derived − comparator) vs comparators 1–4,
  each CI must exclude 0 in SatWQ's favour for an event "SOTA" label (intersection–union, no multiplicity
  correction); derived vs direct is reported (Holm across E4a tiers for the route comparison).
- Cells: headline (primary), Test-B and pooled (secondary, as §9a), track U (secondary; event artifacts refitted on
  U-fit with the same specification, calibrators from the main val cell).
- Power rule (`events.power_flag`): < 30 positives or < 10 clusters with positives → descriptive only.

### 12.5 Code and freeze
- Adapter `sentinel/events/orbit.py`, development script `scripts/eval/events_orbit.py`, config
  `configs/events/orbit.yaml`. They are outside the orbit freeze globs (event plan §4.1) and are **added
  deliberately** to the orbit scoring-code hash (`guard.EVENT_CODE`, together with `sentinel/eval/events.py`).
- The fitted event objects (direct heads, comparator classifiers, residual SDs, calibrators, frozen FPR cuts) are
  one artifact `events_orbit_<version>.pkl`, sha-pinned in `weights_sha256["events_orbit"]`. The event metrics
  are scored in the same `M2/test` look as the headline.
- **Shared module pin (2026-11-09).** `sentinel/eval/events.py` v1.0 (released, immutable), sha256
  `4279d2427d777405ce1cb140cab995567a8086fee5fd4776950e4de1065954ae`; the adapter refuses any other file, and the
  event artifact records the hash. Event labels are computed from the **linear** in-situ values (never from
  back-transformed log10, since 10^log10(20) > 20); predictive distributions are passed with `space="log10"`.
  Runs made earlier on 2026-11-09 against a pre-release draft were discarded unregistered.
