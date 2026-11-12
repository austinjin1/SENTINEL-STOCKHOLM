# EW-v3-N — pre-registration: hazard-onset early warning at USGS stations outside every v2 split (new-site block)

**Status:** v0.4 (2026-11-10; v0.2 §2a exposure audit, v0.3 §2b review fixes, v0.4 §2c delta-review fixes), written by M1 **before any value of the new-site block was examined** (lead decision
2026-11-09). It is for independent review; the freeze record `docs/preregistration/FREEZE_ew_v3.json` is written by
the lead only. **Not registered on OSF**: frozen in a public GitHub commit (addendum A1 convention).
**Authors:** Austin Jin, Bryan Cheng (AI-assisted). **Implements:** RESEARCH_PLAN §12, §15.1, §15.5 (items 1–7);
`docs/plans/sota_campaign.md`.

## 0. Why this track, decided when, and prior knowledge
**Why.** EW-v2 Test-A could not be powered: 43 H1 onsets against the ≥ 50 required, within a 20 %-of-basins ×
2-year test cell (protocol §9 projected ≈ 44).
- A metadata census of the USGS series catalogue (2026-11-09; values not read) found 1,430 USGS sites with any DO
  record in WY2008–26. The EW split holds 861 of them (≈ 75 % of DO site-years).
- **729 surface-water sites outside the EW split and outside every other workstream's pre-registration file** remain:
  `sonde_newsites_list.json`. Correction (v0.2): 113 of them are nominal members of the DEV smoke-test split, and
  other exposure tiers exist; see the §2a audit.

They give a large **spatial** test population, outside every v2 split; their v1 and other-workstream exposure is audited in §2a–§2b. Practical justification: forecasting at **newly deployed
stations with only two years of their own record**. This is the operational situation of a new sonde; it is a
"sparse-data sites" track in the sense of sota_campaign rule 2.

**When and on what basis.**
- **Chosen (2026-11-09) from metadata power only.** The site-specific 2-year reference design (B) was chosen over the
  EW-v2-like design (A: reference ≤ WY2021, issues WY2022–25). Series begin/end dates (no values) give:

  | Design | DO sites | Issue site-years | HUC4 |
  |---|---|---|---|
  | A | 85 | ≈ 306 | 39 |
  | B | 243 | ≈ 914 | 66 |

  With the train/val onset rate this projects ≈ 46 vs ≈ 135–147 H1 onsets (upper bounds).
- **Prior knowledge (disclosed).**
  - EW-v2 val and Test-A results (RESULTS.md 2026-11-09).
  - The live-trial power projection (train/val-role sites).
  - Metadata of the new sites: types, coordinates, HUC, series begin/end.
  - Integrity counts of the pull: per-site file counts, date coverage and observation counts, which may be read
    before freeze.
- **Never computed before the single look:** any onset, exceedance, event count, value statistic or plot of a
  new-site value.

## 1. Hypotheses (per hazard in the fixed family {H1, H2, H4}; Holm, m = 3)
- **N-a (primary):** at h = 3 d, the frozen EW-v3 selected method has BSS > 0 against the identically recalibrated
  site × season climatology on the new-site block.
- **N-b (secondary):** paired ΔBS vs persistence, the lag logistic, the event GBM and the **frozen EW-v2 stack**, as an
  original-model reference applied unchanged.
- **N-c (secondary):** POD (≥ 24 h) at the frozen alarm budget exceeds the season-matched placebo POD.

## 2. Population (frozen list; metadata-only rule)
- **Sites.** `docs/preregistration/sonde_newsites_list.json`, sha256
  `6200a77cbdb0bb7963e6373a962374a764a6577bd7b945a42530a2b2e4e8b720`, built by `scripts/data/build_newsite_list.py`:
  - 729 sites: 424 stream, 153 lake/reservoir, 133 estuary, 19 wetland;
  - by variable: DO 393, SC 661, NO3+NO2 36 (series covering ≥ 1 year in WY2008–25);
  - excluded: 1,541 split sites, 101 sites named in other workstreams' pre-registration files, 298 of other or unknown
    type.
- **Data.** `processed/sonde_newsites`: legacy DV (frozen `sonde_daily` QC) and legacy IV (frozen `sonde_hourly` QC),
  WY2008–2025, Approved values for labels.
  - **Daily tier.** USGS DV where the (site, variable, statistic) has a DV series. Otherwise it is derived from QC'd IV
    by the rule in `scripts/data/download_newsites.py`: local standard-time day; min / mean / max / median; day kept
    if ≥ 80 % of the water-year median daily count; provisional if any value is.
  - **Validation of that rule (val only).** The IV-derived daily rule is compared with DV on **train-role split
    sites, train block only**: agreement of daily min DO and an H1-onset concordance. This is reported before the
    freeze; if agreement < 99 % of days within 0.1 mg/L, the IV-only sites form a separate secondary cell.
- **Site-specific reference window.** Reference = the first 730 days from each site's first observation of the hazard
  variable (on or after 2007-10-01). It is used for:
  - climatology (smoothed DOY mean/std with `min_years = 2`);
  - the site's climatology-model deviation, fitted on reference-window labels;
  - feature history.

  **Issue window** = reference end + 42-day embargo → 2025-09-30. Sites with < 365 observed reference days are
  unscorable (reported).
- **Clusters.** HUC4 from the monitoring-location metadata (outer bootstrap stage); site-month inner stage.
- **Cells.**
  - **Primary: all classes.**
  - Secondary:
    - streams;
    - lakes/reservoirs;
    - estuaries;
    - new sites in HUC4s that contain no train/val-role split site (stricter spatial independence);
    - IV-derived vs DV-based sites.

## 2a. Exposure audit of the new-site block (added 2026-11-09, v0.2, before any model saw a new site)
**Trigger.** M6 reported that some reserved sites are also its track-N WQP stations. Audit record (metadata only):
`docs/preregistration/sonde_newsites_overlap.json`, sha256 `9c72139d…`, built by `scripts/data/newsite_overlap.py`.

**Findings:**
- **Exact identifier matches.**
  - `hydromet_track_n_stations.json` (`WQP:USGS-<no>` keys): **0**. The site-list builder had already excluded the 21
    track-N USGS stations named there.
  - Any other split or pre-registration file: **113 sites in `sonde_split_dev.json`**, the DEV split for smoke tests
    only (75 train / 33 test / 5 val roles there; all are estuaries, lakes or wetlands). The builder skipped every
    `sonde_split*` file instead of only the final split. This is an oversight, disclosed here.
- **Actually loaded by any recorded run** (155 `data_info.json` site lists): **3 sites** (smoke runs).
- **Co-located M6 track-N WQP stations** (another organisation's station at nearly the same place; M6N/test scored
  temperature/DO predictions there; no M1/EW model saw them):
  - within 100 m: 4 sites (1 with a test-role station);
  - within 500 m: 11 (3);
  - within 1 km: 24 (4).
- **Other workstreams' tables (catalogues, not splits).**
  - Powell metabolism labels (DO-derived): **48 sites**.
  - Orbit S2–WQP matchups: 94 sites (177 historical). These are satellite/grab-sample pairs, not sonde series.
  - WQP discrete catalogue: 401 sites.

**Tiers:**
- **Hard exposure: 72 sites.** Loaded by a run, OR a track-N station within 1 km, OR Powell DO-derived labels.
- **Full exposure set: 180 sites.** The hard tier plus nominal DEV-split membership.

**Populations (chosen from metadata power only; design B; train/val rate; ×0.5 discount):**

| Population | Sites | DO issue site-years | HUC4 | H1 projected onsets |
|---|---|---|---|---|
| all 729 | 243 DO | ≈ 914 | 66 | ≈ 68–74 |
| minus hard tier (657) | 208 DO | ≈ 676 | 62 | ≈ 50–54 |
| minus full set (549) | 170 DO | ≈ 449 | 55 | ≈ 33–36 |

**Pre-declared:**
- **Primary population = all sites minus the hard-exposure tier (657 sites).**
- **Sensitivity S1:** minus the full exposure set (549 sites).
- **Descriptive S2:** all 729 sites.

**Lead decision (2026-11-09): primary = all sites minus the hard-exposure tier (657 sites); S1 (minus the full set,
549 sites) is a pre-declared sensitivity; S2 (all 729) is descriptive.**

**Rationale (recorded as given).** Exposure means **actual data access or derived labels**:
- loaded by a recorded run;
- Powell DO-derived labels;
- a co-located M6 track-N station within 1 km (where M6N/test scored predictions).

A nominal entry in an unused smoke-test split with no recorded load is **not** exposure. The decision was made **on the
metadata counts above and this rationale, before any model or any target statistic touched the new sites**.

**Unchanged by the decision:**
- The builder oversight (DEV split not scanned) stays disclosed as stated above.
- The frozen site list (sha `6200a77c…`) is unchanged.
- The builder is fixed for future use: only the final split and the list's own files are skipped
  (`build_newsite_list.SKIP_FILES`; test `tests/test_newsite_list.py`).

Every cell named in §2 is formed within the primary population.

## 2b. v0.3 changes (2026-11-10; pre-registration review "EW-v3-N and M1-N", C-N1–C-N3, W-N1–W-N5; before any model or target statistic touched the new sites)

**C-N1. v1 exposure: the "legacy" tier (lead ruling).** Exposure tiers ask whether **v2 development could have been
informed** by a site. Audit record: `docs/preregistration/sonde_newsites_v1_audit.json` (built by
`scripts/data/newsite_v1_audit.py`; site IDs only, no value).
- **Part A: no v2 artifact derives from v1. Verified.**
  - No v2 code, config, registry row or run record reads the v1 mirror (`LEGACY_ROOT`), with one exception: the
    microcystin benchmark builder `build_mc_benchmark_v2.py`. Its output names **0** new sites, and its manifest is
    the only data manifest that mentions v1.
  - No v2 model loads a v1 checkpoint: every `torch.load` / `load_state_dict` path is a v2 run output.
  - No v2 threshold, Platt map, selection or protocol uses a v1 result. EW-v2 reports v1 as retracted, and the v1
    CUSUM is not re-scored.
- **Part B: v1 named 471 of the 729 sites, in 64 files.** v1 used the **values** of 69 sites:
  - its sensor-model normalisation statistics (60);
  - v1 results: salinisation events and detection, USGS anomaly, cross-network early warning;
  - v1 prospective predictions.

  Of these, **57 are outside the hard tier**. The reviewer counted 58; the difference is one v1 deployment script that
  hard-codes a site.
- **Tiering (the lead's ruling, applied because Part A holds).**
  - The 57 sites form a separate **legacy tier**. They are disclosed, **kept in the primary population**, and
    **excluded in the pre-declared sensitivity analysis S1b**.
  - **S1b is reported alongside the primary with equal prominence.**
  - Had any v2 artifact touched them, they would have joined the hard tier.

**H2 regression disclosure.** The October chloride builder had already fitted SC→Cl regressions at **90 new sites**,
1 of them eligible, before the reservation; that site's derived chloride series is in the shared `chloride_derived`
tier. Consequences:
- **Hard tier.** That eligible site is added to the hard tier: its derived H2 labels exist in a shared table, as with
  the Powell labels. The hard tier is now 73 sites; the primary population is 656 sites.
- **Fits.** EW-v3 never uses the existing fits. H2 regressions are re-fitted **inside the claimed look, through the
  guard**, from WQP pairs dated in each site's reference window.

**Populations (metadata projections; design B; train/val H1 rate 0.148–0.161 per observed site-year):**

| Population | DO sites | DO issue site-years | HUC4 | H1 projected (full coverage) | H1 projected (×0.5) |
|---|---|---|---|---|---|
| ~~Primary: minus hard tier~~ (superseded by §2c) | 208 | ≈ 676 | 62 | ≈ 100–109 | ≈ 50–54 |
| ~~S1b~~ (superseded by §2c) | 169 | ≈ 497 | 51 | ≈ 74–80 | ≈ 37–40 |
| ~~S1~~ (superseded by §2c) | 170 | ≈ 449 | 55 | ≈ 66–72 | ≈ 33–36 |

**Power, stated honestly (W-N3).**
- **H1 underpowered is about as likely as not** in the primary population, and more likely than not in S1b and S1.
  The ×0.5 discount is a guess, and stream onset rates are applied to lakes and estuaries.
- **Low power at EW-v2-like effects.** The SOTA bar (beat every comparator, with CIs excluding 0, in 3 seed sets) has
  low power at effects like Test-A's (BSS +0.013).
- **Not comparable to EW-v2.** The reference is a 2-year climatology, which is weaker than EW-v2's 14-year one, so BSS
  values are not comparable with EW-v2 and are labelled as such.
- **H2 and H4** are expected to be underpowered.

**C-N2. Archived forecasts in validation.** **Perfect-prognosis covariates (observed gridMET for days d+1..d+7) are used
for TRAINING ONLY**, i.e. the train block.
- **Every validation step uses the ARCHIVED ISSUED forecasts** (Open-Meteo Previous Runs, GFS `temperature_2m`, lead
  days 1–7; available from 2021-03, covering all of WY2022–23):
  - model selection;
  - Platt maps, stack weights and alarm thresholds;
  - quantile calibration and ensemble weights;
  - for every forecast-using model **and every covariate-taking comparator alike**.
- **Bias correction.** Before use as inputs, the archived forecasts are bias-corrected against gridMET by a per-site,
  lead-day linear map fitted on **train-block overlap only** (2021-03-24..2021-09-30). Where a site has < 60 overlap
  days, a pooled HUC2 map is used. A product mismatch therefore cannot be learned from validation data.
- **Secondary mode.** Precipitation and radiation forecasts (2024+) form an explicitly labelled secondary mode,
  validated perfect-prog, because no archived precipitation forecast exists for WY2022–23.

**C-N3. Guard, now enforcing** (`sentinel/data/newsite_guard.py`; tests `tests/test_newsite_guard.py`).
- `open_site` enforces: purpose = score; a lead freeze with no placeholders; content and list pins; and an **open
  claim of the calling process, read from disk** (`CLAIM.json` with this pid and host; no SCORED/ABORTED). There is no
  caller-supplied callback.
- **Lock.** `processed/sonde_newsites` is set to mode 000 (data files) and 0300 (directories) by
  `scripts/data/lock_newsites.py --lock` once the manifest is built. Unlocking happens only in-process inside the open
  claim, and the scorer re-locks in `finally`.
- **Block list.** `assert_not_newsite` / `exclude_reserved` (pinned list sha) are required in every non-frozen loader
  of the shared tiers and in every EW-v3 / M1-N development entry point.
- **Shared-tier copies.** They exist in `processed/sonde_daily` (both snapshot directories), `sonde_hourly` and raw IV.
  The permission plan for them is the lead's decision and is not yet applied. Until it is, **other code paths CAN read
  the shared-tier copies of the new sites**; the protection is the block-list helper plus the pre-claim audit.
- **Pre-claim audit.** Before the claim, every registry row and every `data_info.json` / run record written after
  2026-11-09 is scanned for reserved IDs. Any hit refuses the claim and is reported.

**W-N2. Reference window (unified with the M1-N protocol).** For each site and variable:
- **Reference** = the first **730 calendar days** from the site's first observation of that variable on or after
  2007-10-01. At least 365 observed days are required inside it; otherwise the site is unscorable for that variable.
- **Issue window** = reference end + 42 days → 2025-09-30.

The power numbers above use this definition.

**W-N4. Hold-out type and other overlaps.**
- **Spatial-only primary.** New-site issue years (WY2010–25) overlap the training years at other sites, so the primary
  is a **contemporaneous spatial hold-out** and is labelled as such.
- **Pre-declared sensitivity S-T** (spatial **and** temporal): issue dates ≥ 2023-10-01 only.
- **Grab samples.** Grab samples at new sites exist in the shared `wqp_discrete` (reviewer's counts): DO 239,
  NO3+NO2 208, chloride 164, SC 348 sites. Other workstreams use them (downstream, Fuse, HydroMet track N). The
  pre-claim audit extends to those workstreams' training-row site lists where they are written.
- **Turbidity and chlorophyll overlap.** Orbit's AquaCorr / S2 matchups (206 / 95 sites) and Lite's national labels
  (3 sites) carry turbidity and chlorophyll at new sites. This is disclosed for M1-N turbidity.

**W-N5. Data notes.**
- **IV gaps.** The legacy IV service rejects the spring-forward hour, so about 4 h per site-year are missing on each
  spring DST day (logged "503 on DST piece"). IV-derived daily values on those days fall under the 80 %-of-median
  count rule.
- **IV-derived daily rule: validation FAILED its pre-specified acceptance (2026-11-10).**
  - Method: `scripts/data/validate_iv_daily_rule.py` on 54 train-role split sites, train block only; result in
    `sentinel_runs/sonde/iv_daily_rule_validation/`.
  - DO daily min: 94.1 % of 117,856 common days within 0.1 mg/L, against the required ≥ 99 %. Per site: median
    98.6 %; 13 of 44 sites ≥ 99 %; 2 sites < 80 %.
  - H1 onsets: IV-derived 30 vs DV 39, with 29 in common.
- **Consequence as pre-specified.** IV-only new sites form a **separate secondary cell**, and the primary uses
  DV-based sites only.
- **Effect on power** (series metadata only): primary DO sites with a DV daily-min series are 53, with ≈ 237 issue
  site-years and 26 HUC4. H1 projects ≈ 18–19 onsets (×0.5), or ≈ 35–38 at full coverage, so it is **certainly
  underpowered**. The IV-only cell (154 sites, ≈ 432 site-years) is descriptive.
- **Lead ruling (2026-11-10): option (ii).**
  - **Split, frozen first.** `docs/preregistration/sonde_ivrule_split.json` (sha `e4143b74…`) holds the seed, the
    rule, the cap and the lists, all written before any revision. 428 train-role split sites with raw IV, train block
    only, no new-site data: 244 development and 184 confirmation sites.
  - **Revisions (≤ 3), developed on the development half only** (`scripts/data/ivrule_revision.py`):
    - R1: per-series daily statistics, longest series primary;
    - R2: R1 with DST-aware local time;
    - R3: R1 with plausibility-only QC.

    Each is logged (registry `ew_v3_m1n/ivrule_dev`). Selection: best development-half agreement.
  - **Confirmation, run ONCE** on the confirmation half (O_EXCL marker; registry `ew_v3_m1n/ivrule_confirm`), at the
    **unchanged** bar: DO-min within 0.1 mg/L ≥ 99 % pooled **and** H1 onset Jaccard ≥ 0.90.
  - **If the confirmation fails, option (i) applies automatically: EW-v3-N H1 is NON-CONFIRMATORY** (IV-only sites
    stay a descriptive cell).
  - **If it passes,** the new-site daily-IV tier is rebuilt with the confirmed rule, which needs a re-pull of the
    deleted raw IV, before the freeze.
- **Integrity table and manifest:** built when the pull finishes.

- **Shared-tier guard, applied (lead ruling 2026-11-10).**
  - **Locked:** the 27 + 21 new-site directories in `raw/usgs_iv{,_refresh_20261108}` are mode 000 (lock records in
    those trees). `sonde_daily` / `sonde_hourly` are untouched, because frozen scorers content-hash them.
  - **Shared helper:** `sentinel/data/reserved.py` (hash-pinned list), added to every NON-frozen loader of the sonde
    tiers:
    - M1: `events_feasibility_*`, `sonde_onset_counts`, `sonde_virtual`, `scripts/live/live_data.py`;
    - Sim: `sim/data.py`, `causal/diel.py`;
    - Downstream: `killalert.py`;
    - Fuse: `fuse_tier_w`, `fuse_tier_wp1`, `fuse_wp1_power`;
    - Platform: `data_access.py`.

    Each file was first checked against every FREEZE / freeze-pin record. `sonde_score.py` (hashed by the M1
    candidate) was not edited; it is split-gated. The M3 and NEON loaders read only NEON tiers and need no edit.
  - **Tests:** `tests/test_reserved_loaders.py`.
  - **Pre-claim audit:** `reserved.assert_preclaim_clean`, called through `newsite_guard.preclaim_checks(look)`. On
    2026-11-10 it found 0 registry and 0 run-record hits.

**W-N1.** Both protocols and all audit files are re-snapshotted in the registry (`ew_v3_m1n/prereg_snapshot`).

## 2c. v0.4 changes (2026-11-10; delta review C-F1, C-F2 and warnings; before any model or target statistic touched the new sites)

**C-F1 (lead ruling): v1 outcome-level use counts as exposure.** Under the lead's own rule, v1 outcome-level results
count as exposure: derived labels, episode or onset results, predictions, and v1 build results. Audit v2:
`docs/preregistration/sonde_newsites_v1_audit_v2.json` (sha `42b1c76b…`).
- **Classification fix.** The first audit's value-level pattern missed `data/processed/trackA_*/_build_results.json`.
  With it, 70 sites have v1 value-level use, matching the reviewer.
- **Moves to the hard tier.** **12 sites** with v1 outcome-level use that were not already hard:
  01304200, 01465798, 01467042, 01467086, 01653560, 01669000, 04118997, **07288048**, 07380245, 073802512, 11455508,
  412240071574700.
- **Remaining legacy tier: 46 sites,** whose only v1 use is the sensor-model normalisation statistics. They stay in the
  primary and are excluded in S1b, which is reported with equal prominence.
- **Machine-readable tiers:** `docs/preregistration/sonde_newsites_tiers_v2.json` (sha `cee6b019…`). Hard 85,
  legacy 46.

**Populations and power (metadata projections; design B; train/val H1 rate 0.148–0.161 per observed site-year):**

| Population | Sites | DO sites | DO issue site-years | HUC4 | H1 (full coverage) | H1 (×0.5) |
|---|---|---|---|---|---|---|
| **Primary: minus hard (85)** | 644 | 199 | ≈ 627 | 61 | ≈ 93–101 | **≈ 46–50** |
| **S1b: minus hard + legacy** | 598 | 167 | ≈ 478 | 51 | ≈ 71–77 | ≈ 35–38 |
| S1: minus hard + §2a full set | 538 | — | — | — | — | below S1b |

**If the IV-derived daily rule is not confirmed (§2b),** only DO sites with a USGS DV series stay in the primary:
51 sites, ≈ 225 site-years, 25 HUC4, H1 ≈ 17–18 (×0.5) or ≈ 33–36 at full coverage.

**Stated plainly:**
- **Even at the projection's centre, the primary is at or below the 50-onset threshold.** EW-v3-N H1 is more likely
  than not to be underpowered.
- **Without a confirmed IV rule it is certainly underpowered,** and EW-v3-N is then descriptive only.
- **S1b is more likely than not underpowered.**
- **H2 and H4 are underpowered.**
- **M1-N** (CRPS, many rows per site) is far less affected. Its variable cells still need ≥ 10 HUC4 each.

**C-F2: IV-rule split v2.** `docs/preregistration/sonde_ivrule_split_v2.json` (sha `f2ebe52f…`). The first
development run under split v1 was **stopped before it wrote any result**: no output was seen and no registry row
exists.
- **Confirmation half:** the 54 sites evaluated in the failed validation (results seen) are removed. 25 were in the
  confirmation half, which now has 159 sites; the development half keeps 244.
- **Selection:** a variant is eligible only if it meets **both** criteria on the development half. If none does, no
  confirmation is run and option (i) applies.

**IV-derived daily rule, development-phase outcome (2026-11-10; split v2; registry `ew_v3_m1n/ivrule_dev`, 4 rows;
`sentinel_runs/sonde/ivrule_revision/dev_selection.json`).** Development half: 244 train-role sites, train block.

| Variant | DO-min days within 0.1 mg/L | H1 onsets IV / DV / both | Onset Jaccard |
|---|---|---|---|
| v0 (failed rule) | 0.964 | 249 / 296 / 227 | 0.714 |
| R1 (primary series) | 0.976 | 247 / 296 / 227 | 0.718 |
| R2 (R1 + DST) | 0.963 | 250 / 296 / 225 | 0.701 |
| R3 (R1 + plausibility-only QC) | 0.976 | 247 / 296 / 227 | 0.718 |

**No variant met both criteria** (≥ 0.99 and ≥ 0.90), so **no confirmation was run**; the confirmation half is untouched.
**Option (i) applies automatically, as pre-specified.**
- **EW-v3-N H1 is NON-CONFIRMATORY.** The primary uses DV-based DO sites only: ≈ 51 sites, ≈ 17–18 projected H1
  onsets, certainly underpowered.
- **The IV-only new sites form a descriptive secondary cell.** H2 and H4 were already underpowered, so **EW-v3-N as a
  whole is descriptive.**
- **M1-N** uses DV-based sites in its primary as well; IV-only sites form a secondary cell. It remains viable
  (≥ 10 HUC4 per variable):

  | Variable | DV-based sites | Issue site-years | HUC4 |
  |---|---|---|---|
  | DO | 51 | ≈ 230 | 24 |
  | SC | 155 | ≈ 1,068 | 54 |

  Nitrate stays descriptive. Temperature and turbidity counts follow from the integrity table.
- **No further revision is permitted** under the cap and ruling.

**Warnings.**
- **New-site data built and locked (2026-11-10).**
  - **Pull:** 729 of 729 sites complete, **0 failed**. An earlier "17 failed" was a mistaken count of progress-log lines
    reading "… 0 failed".
  - **Manifest:** `processed/sonde_newsites/manifest.json`, **content pin `e615a10c743628e9f1148f556d06b42e69dc630f9a0acdaea7a0acef046c6591`**
    over 1,829 files (DV 499, hourly 665, IV-derived daily 665); 633 MB. The raw IV was deleted per site.
  - **Integrity table:** `integrity.parquet` (counts and coverage only).
  - **Locked:** `lock_newsites.py --lock`: data files 000, directories 0300. A read of a reserved file raises
    PermissionError. The manifest and the integrity table stay readable.
  - **Pin for the freezes:** this content pin is what `FREEZE_ew_v3.json` / `FREEZE_m1n.json` pin as
    `newsite_data.content_sha256`.
- **Bias-correction limitation.** The train-block overlap of archived GFS temperature with gridMET is only
  2021-03-24..2021-09-30, about 6 months of spring–summer. The correction is therefore poorly identified for autumn and
  winter, and residual seasonal bias in the forecast covariates is possible. Every comparator shares it.
- **Repo-wide static test** (`test_no_unguarded_reader_of_the_shared_tiers`). Every file in `sentinel/` and `scripts/`
  that references the shared sonde tiers must import the helper, be hashed by a freeze or candidate record, or be in a
  documented allowlist (builders, NEON-only, C-E1 live issuing, via-helper callers).
- **Unlock.** `lock_newsites.py --unlock` refuses: unlocking is scorer-only, through `unlock_in_process(look)` inside
  the scorer's own claim, followed by `relock()`.
- **W-N1.** Re-snapshot in the registry, covering the protocols, both split files, both audit files, both tiers files,
  the tools and the tests.

## 3. Hazards, forecast task, horizons
- **Hazards and onset rules.** Exactly the frozen `sentinel/eval/hazards.py` rules: H1 DO-min < 3 mg/L; H2 chronic
  chloride; H4 NO3+NO2 > 10 mg/L. The < 4 mg/L tier (the only Test-A cell with ≥ 50 onsets) is **not** promoted.
- **H2.** Needs a per-site SC→Cl regression from WQP pairs dated **inside the reference window** (≥ 20 pairs,
  R² ≥ 0.9, b > 0). With 2-year windows few or no sites are expected to qualify, so H2 is expected to be
  underpowered and to enter Holm with p = 1. It stays in the family (m = 3).
- **Task.** Daily issue at the end of day d. Leads 1, 3, 7; primary h = 3. Approved labels; label validity as
  `hazards.issue_labels`.
- **Inputs at issue.** Daily and hourly tiers up to d, and lagged observed gridMET (lag 1). In forecast mode, also
  **archived NWP forecasts** (§5).

## 4. Power (train/val rates only)
- **H1 rate:** 0.148–0.161 onsets per observed site-year, from train/val-role EW split sites (`ew_v2_live/power`).
- **Design B:**
  - all classes: ≈ 914 DO issue site-years → ≤ 135–147 onsets (span-based upper bound);
  - with a 50 % discount for gaps, Approved-only labels and class differences: **≈ 68–74**;
  - streams only: ≈ 477 site-years → ≤ 70–77, discounted ≈ 36.
- **Outlook by hazard:**
  - **H1** is expected to meet the power rule in the primary (all-class) cell;
  - **H4**: 15 sites, ≈ 36 site-years → ≈ 5 projected, so underpowered;
  - **H2**: expected 0 eligible sites.
- **Power rule (unchanged):** < 50 evaluable onsets or < 2 HUC4 groups with onsets → underpowered → p = 1 in Holm, no
  claim. The observed count is reported whatever it is.

## 5. Methods and comparators (identical inputs, rows and splits)
- **Training and selection data.** Training uses EW split **train-role sites × train block** only. Selection,
  calibration (Platt), stacking and thresholds use the **val block × train/val roles** (as EW-v2).
  - **Excluded from all training and tuning:** new sites, test-role sites, Test-A, Test-B, and WY2024–25 at any role
    (all-role Test-A cells were scored by the EW-v2 pass).
- **Regime parity.** Every feature-based model is trained with climatologies estimated from **2-year windows** of the
  training sites (rolling, causal). This gives the same noise level as at new sites. Disclosed as a model choice.
- **Comparators.** Each gets the same rows and inputs:
  - climatology (site × DOY harmonic logistic, 2-year reference deviations);
  - persistence / current state;
  - threshold-on-current-value (POD only);
  - lag logistic;
  - event GBM (5-bag);
  - the **frozen EW-v2 stack** (unchanged artifacts; reference only).
- **Candidates:** the monotone GBM, the logistic stack, and up to two information levers (below).
- **Archived weather forecasts (given to every comparator).**
  - Source: Open-Meteo Previous Runs API (`scripts/data/download_nwp_archive.py`). GFS `temperature_2m` lead days
    1–7 exist from 2021-03; precipitation and shortwave radiation from 2024-01 (GFS) and 2024-02 (ECMWF).
  - **Perfect-prognosis protocol, TRAINING ONLY (§2b C-N2 supersedes "and validation"):** in training the "forecast" covariates are **observed** gridMET
    values for days d+1..d+7. At test they are the archived **issued** forecasts (valid ≤ initialisation + lead, as
    verified by the downloader's `semantics` step).
  - **Primary forecast mode** uses NWP temperature only, for all issue dates ≥ 2021-03-30. Precipitation/radiation NWP
    is a secondary mode for issue dates ≥ 2024-02-10.
  - Issue dates before NWP temperature exists use the past-weather-only variant of the same frozen model; the split
    is reported.
- **Selection rule (pre-specified).** Per hazard, the highest val out-of-fold (by water year) BSS at h = 3 among
  non-baseline candidates, as `scripts/eval/sonde_ew_freeze_candidate.py`.
- **Configuration cap:** **≤ 12 calibrate configurations in total**, all counted and logged. Levers, in this order,
  stopping at the cap:
  1. 2-year-climatology parity;
  2. NWP temperature, perfect-prog;
  3. upstream-sensor context;
  4. LSTM-encoder head.

## 6. Endpoints and decision (as `ew_v2_protocol.md` §7, re-stated)
- **Primary statistic.** Pooled BSS of the selected method vs the identically recalibrated climatology, h = 3 d,
  primary cell. Two-stage HUC4 > site-month bootstrap, B = 2,000; one-sided p.
- **Multiplicity.** Holm over {H1, H2, H4}, m = 3.
- **Evaluable onsets.** One definition (`ew_scoring.evaluable_onsets`) for the gate, POD and every control.
- **Success:** not underpowered and Holm rejection.
- **Reporting.** Basin-group counts next to every p; CIs flagged as approximate with < 10 groups. Realised alarm
  rates reported as they are; thresholds are never re-tuned.
- **SOTA / first-in-class label (§15.1).**
  - The primary must beat **every** comparator (incl. the frozen EW-v2 stack and the event GBM) with paired CI
    excluding 0, under Holm.
  - "First-in-class" additionally needs a dated novelty search in `docs/novelty_checks/` before the look.
  - 3 seeds: event GBM bag seeds 0–4, 5–9 and 10–14, each meeting the bar.
  - Otherwise the result is reported as "competitive" or "negative". Claims name the track ("EW-v3-N, new USGS
    stations with 2-year records").
- **Controls** (as EW-v2 §8, on frozen predictions):
  - season-matched placebo (cluster bootstrap);
  - random alarms;
  - circular shift (labelled: no season control);
  - label permutation;
  - negative-control sites (training-reference DO min ≥ 6 mg/L);
  - covariate shuffle;
  - input-truncation audit;
  - Molchan, reliability, REV, Diebold–Mariano.

## 7. Single-look scorer requirements (hardened; sota_campaign rule 7)
1. **Before the claim** (a failed check never burns the look), the scorer must verify:
   - the lead-written `FREEZE_ew_v3.json` (not a candidate);
   - the full import-closure code hash of the scorer entry points;
   - the calibration and threshold file SHA-256 and every model artifact;
   - the new-site list sha and `processed/sonde_newsites` content pin (`sentinel.data.newsite_guard`);
   - the split sha;
   - that the registry holds no row of experiment `ew_v3_newsite_score` and no marker exists.
2. **Claim:** an O_EXCL, 0444 marker `sentinel_runs/sonde/.ew_v3_scored_newsite.json` holding pid, host, freeze sha and
   argv, plus a `look_claimed` registry row with the same pid and host.
3. **Data access** only through `newsite_guard.open_site(purpose="score", claim_check=…)`, which checks that this
   pid/host holds the open claim. Until the shared-tier permission plan (§2b, C-N3) is applied, the shared-tier copies remain readable; protection is the block-list helper plus the pre-claim audit.
4. **Abort:** any `BaseException` after the claim → `ABORTED.json` + `look_aborted` registry row; access closed. At
   most one logged resume, under the same claim, before predictions are sealed.
5. **Seal:** predictions are written once, sealed (`SHA256SUMS`, 0444, `predictions_sealed` row), then scored.
   `SCORED.json` + a sealing registry row close access. **Rescore is refused after sealing**; only a re-score from the
   sealed predictions after a crash is allowed, and it is logged.
6. **Write-up:** `ew_v3_test_results.md` (never edits to frozen files).

## 8. Deviations log
| Date | Change | Reason |
|---|---|---|
| 2026-11-09 | v0.1 written before any new-site value was examined | lead decision |
| 2026-11-09 | v0.2: §2a exposure audit (M6 report); primary = minus hard-exposure tier (657 sites); S1/S2 sensitivities; builder oversight (DEV split not scanned) disclosed | M6 finding; no new-site value examined |
| 2026-11-09 | Lead decision recorded in §2a: primary = minus hard tier (657); S1 sensitivity; S2 descriptive; exposure = actual access or derived labels; builder fixed (only the final split skipped) + test | lead decision on metadata counts, before any model or target statistic touched the new sites |
| 2026-11-10 | v0.3 (§2b): legacy tier (57) + S1b with equal prominence (lead ruling; v2-independence verified); 1 chloride-eligible site added to the hard tier (656 primary); archived forecasts in validation (perfect-prog for training only) + train-block bias correction; enforcing guard + lock + block-list + pre-claim audit; unified reference window; spatial-only primary + S-T sensitivity; grab-sample and Orbit/Lite overlaps disclosed; honest power; DST IV gaps | pre-registration review 2026-11-09 evening |
| 2026-11-10 | v0.4 (§2c): v1 outcome-level sites (12) moved to the hard tier (lead ruling C-F1); tiers v2 (hard 85, legacy 46; primary 644); power restated; IV-rule split v2 (C-F2); bias-correction limitation; repo-wide static test; unlock scorer-only | delta review 2026-11-10 |
| 2026-11-10 | IV-rule development phase: no variant met both criteria; option (i) applied automatically (EW-v3-N H1 non-confirmatory; DV-based primary; IV-only sites secondary) | pre-specified fallback (lead ruling) |
