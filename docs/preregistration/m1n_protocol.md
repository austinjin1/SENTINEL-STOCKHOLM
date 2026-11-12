# M1-N — pre-registration: sensor forecasting and virtual sensing at USGS stations outside every v2 split (new-site block)

**Status:** v0.5 (2026-11-10; v0.2 §2a exposure audit, v0.3 §2b review fixes, v0.4 §2c delta-review fixes, v0.5 §4a numeric-forecast reporting), written by M1 **before any value of the new-site block was examined** (lead decision
2026-11-09). It is for independent review. The freeze record `docs/preregistration/FREEZE_m1n.json` is lead-written.
It is frozen in a public GitHub commit, not on OSF.
**Authors:** Austin Jin, Bryan Cheng (AI-assisted). **Implements:** RESEARCH_PLAN §5.1, §15.1, §15.2 (M1 primary
structure), §15.5 items 1–7; `docs/plans/sota_campaign.md`.
**Relation to other tracks:**
- The original M1 track (`sonde_m1_protocol.md`; test-role × Test-A) is unchanged and reported on its own.
- M1-N is a **new, additional** track. A SOTA label from it names the track and never replaces the original.

## 0. Why this track, decided when, and prior knowledge
**Why the original track is not SOTA on val.** On the unseen val cell, ens3w beats the 9 references:
- turbidity mean and SC at 4–5/5 horizons;
- nitrate at 2–3/5;
- DO mean at 2/5;
- DO min and temperature at 0/5.

Causes:
- the val-role cell has only 8–13 basin groups (nitrate 3), so it is power-limited;
- at 1–7 d, lag-based learners add no information beyond persistence/damped-anomaly and a GBM with past weather.

**The practical questions this track answers (justification, sota_campaign rule 2).** All are fixed here, before any
result on them:
- **(i)** Forecasting at **newly deployed stations** with two years of record (sparse-data sites).
- **(ii)** **Multi-day horizons with operational weather forecasts.** A learned model can use future covariates; lag
  baselines cannot.
- **(iii)** **Unsensed-variable nowcasting** (virtual sensing) at new stations.

**Prior knowledge.**
- Val results of M1 cycles 1–3 and the EW-v2 Test-A result.
- Metadata of the new sites, and integrity counts of the pull (file counts, coverage, observation counts).
- No forecast, value statistic or plot of a new-site value is made before the single look.

## 1. Population and data
- **Site list.** The same frozen list as EW-v3-N: `sonde_newsites_list.json`, sha256 `6200a77c…`; 729 sites.
- **Data.** `processed/sonde_newsites`, WY2008–2025. Daily tier = DV where it exists, else IV-derived, with the rule
  validated against DV on train-role split sites as in `ew_v3_protocol.md` §2. Approved targets only.
- **Reference window** = the first 730 calendar days from the site's first observation of the variable (≥ 365 observed days required; unified with EW-v3, §2b). **Issue window** = reference end + 42 d →
  2025-09-30.
- **Clusters:** HUC4 (outer), site-month (inner).
- **Primary cell:** all classes. Secondary cells: streams, lakes, estuaries, HUC4s with no train/val-role split site,
  and IV-derived vs DV.
- **Variables:** temperature (mean), DO (mean, min), SC (mean), turbidity (mean), NO3+NO2 (mean).
  - A variable enters the primary only if it has **≥ 10 HUC4 groups** with scorable issue rows at h = 7; otherwise it
    is descriptive.
  - Metadata projections (design B): SC 442 sites / 95 HUC4; DO 243 / 66; NO3 15 / 10 (borderline, probably
    descriptive).
  - Temperature and turbidity counts come from the integrity table before the freeze. Counts only; they may be read.

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

## 2. Tracks and horizons
- **M1-N-F (primary). Forecasting.**
  - Horizons h ∈ {1, 2, 3, 5, 7, 10, 14} d, 19 quantiles. Primary horizons {1, 2, 3, 5, 7}.
  - **Forecast mode** (archived NWP, §3) for issue dates ≥ 2021-03-30. Past-weather mode before that, reported
    separately.
- **M1-N-X4 (secondary). Long horizons.** h ∈ {21, 30} d. NWP days 1–7, then training-climatology weather beyond.
- **M1-N-Nb (secondary). Virtual sensing.**
  - Predict each target from co-located other sensors + discharge + weather + StreamCat/NLDI catchment attributes
    (where the site is matched), with no target history.
  - Metric: KGE and CRPS. Comparator: the published nitrate KGE ≈ 0.18 and the GBM.
- **Extremes stratum (secondary, as M1-X X3).** Threshold-weighted CRPS beyond the reference-window climatology
  ± 1.645 SD.

## 3. Comparators (identical rows, inputs and data; tuning parity)
- **References:**
  - damped anomaly, climatology, persistence (naive);
  - global LightGBM (lags, rolls, weather, attributes);
  - LSTM with weather (multi-station);
  - zero-shot Chronos-2 (uni and multi);
  - zero-shot TimesFM-2.5;
  - the **frozen original-track candidate ens3w**, unchanged, as a reference.
- **Equal-weight ensembles of baselines only** (strict control): GBM + LSTM-w + Chronos-2 multi.
- **Archived weather forecasts for every comparator.** Open-Meteo Previous Runs:
  - **NWP temperature** (GFS `temperature_2m`, lead days 1–7) from 2021-03;
  - precipitation and shortwave radiation from 2024-01/02 (secondary mode).

  **Perfect-prognosis TRAINING ONLY (§2b C-N2: validation, selection and calibration use archived issued forecasts).** Training uses observed gridMET for the target days, aligned to the issue
  date exactly as the NWP lead days. The test uses the archived issued forecasts. The same rule applies to every
  comparator that can take covariates. Univariate zero-shot models are reported as they are, and a covariate-enabled
  Chronos-2 is added.
- **Training data.**
  - Allowed: EW/M1 split train-role sites × train block, with selection on val-block × val-role. Also the shared EA
    data (rule-3 cell only) and NEON train sites, if approved, given to every comparator.
  - Never: new-site data, test-role sites, Test-A/Test-B, or WY2024–25 at any role.
  - All models are trained with 2-year-window climatologies (regime parity with new sites).

## 4. Primary endpoint and decision (structure of RESEARCH_PLAN §15.2, M1)
- **Statistic.** Median per-site CRPS skill of the frozen SENTINEL model vs the **best** reference at each h.
  HUC4-cluster bootstrap, B = 2,000.
- **Bar.** The lower 95 % bound must be > 0 at **every** h ∈ {1, 2, 3, 5, 7}, for **≥ 3 of {turbidity, SC, nitrate,
  DO, temperature}**, in each of 3 seeds, with Holm across the 5 variables. The strict control is reported next to it.
- **Underpowered variables.** A variable with < 10 HUC4 groups counts as **not passing**. The bar is not lowered.
- **Labels.** "SOTA on M1-N (new USGS stations, 2-year records, operational weather forecasts)" only if the bar holds.
  Otherwise "competitive" or "negative". N-b and X4 are reported with their own CIs, as secondary tracks; each claim
  names its track.

## 4a. Numeric forecasts, prediction intervals and error metrics (required; added v0.5, 2026-11-10, before any freeze)
User requirement relayed by the lead: every model predicts specific numeric values, not only events. For M1-N
forecasting (N-a and every secondary forecasting track), the single look reports, for **every** model (frozen
SENTINEL model, each seed, every reference and strict control, plus persistence and climatology), every variable and
every horizon h ∈ {1, 2, 3, 5, 7}:
1. **Numeric forecasts.** The point forecast is the predictive median, in original units. Per-row values (site, issue
   date, h, observation, q05/q10/q25/q50/q75/q90/q95) are written for the frozen model's seeds and sealed with the
   predictions.
2. **Prediction intervals with coverage.** The 50 % (q25–q75), 80 % (q10–q90) and 90 % (q05–q95) intervals are read
   from the model's 19 quantiles. For an ensemble these are the equal-weight averaged member quantiles (fixed in the
   freeze record), not a spread heuristic. Empirical coverage and mean width are reported per variable × h, pooled,
   and as the median over sites. Nominal-vs-empirical gaps are reported as they are, with no recalibration on test.
3. **Error metrics.** MAE, RMSE and bias of the median, plus MAE and RMSE skill vs persistence and vs climatology
   (1 − metric/reference), with HUC4-cluster bootstrap 95 % CIs (B = 2,000) for the frozen model. Per-site values
   (sites with ≥ 30 scored rows) are reported, with their medians over sites for coverage, MAE, RMSE and MAE skill.
   Implemented 2026-11-10 in `sonde_pi_report.py` (`--n-boot`, `--ci-for`, `--min-site-n`). The cluster bootstrap
   is computed on per-group sums with the same draws as the frozen `bootstrap_ci` (unit-tested). The cluster unit
   for M1-N is HUC4, as in §4.

These are secondary, descriptive endpoints. The §4 decision is unchanged and is not re-weighted by them. EW-v3-N event
probabilities are reported in addition, never instead. Implementation: `scripts/eval/sonde_pi_report.py`, which
reads the sealed predictions. It is in the import-closure code hash, and the same script runs on val/CV before the
freeze so its outputs on held-out data are known in format only.

## 5. Selection and caps
- **Configuration cap.** ≤ 16 configurations per variable group (forecasting), ≤ 8 for N-b, ≤ 20 GPU runs per cycle,
  each ≤ 2 h on GPU 0/1 (sonde runner's own lock). All are counted and reported next to any claim.
- **Levers, in order.**
  1. Basin-group 5-fold CV over train+val-role sites × val block, for selection power.
  2. NWP temperature (perfect-prog) for all models.
  3. Upstream-sensor context (NLDI edges) for all models.
  4. Pretraining a Chronos-2 fine-tune on the hourly tier of train-role sites (+ EA rule-3 data if approved).
- **Final model.** Selected on CV/val by mean CRPS skill over the primary horizons and variables, before the freeze.
  Members and weights are fixed in the freeze record.

## 6. Single-look scorer requirements (hardened; identical to `ew_v3_protocol.md` §7)
- **Before the claim:**
  - the lead-written `FREEZE_m1n.json`;
  - import-closure code hash;
  - checkpoint SHA-256 for every member and seed;
  - new-site list sha and content pin through `sentinel.data.newsite_guard`;
  - registry refusal (experiment `m1n_newsite_score`).
- **Claim:** O_EXCL 0444 marker + `look_claimed` row holding pid and host. Data only through `newsite_guard`
  (`purpose="score"`, claim check).
- **After the claim:** `BaseException` → `ABORTED.json` + registry row. Sealed predictions (`SHA256SUMS`) before
  scoring. Access closed and rescore refused after sealing.
- **Write-up:** `m1n_test_results.md`.

## 7. Deviations log
| Date | Change | Reason |
|---|---|---|
| 2026-11-09 | v0.1 written before any new-site value was examined | lead decision |
| 2026-11-09 | v0.2: §2a exposure audit (M6 report); primary = minus hard-exposure tier (657 sites); S1/S2 sensitivities; builder oversight (DEV split not scanned) disclosed | M6 finding; no new-site value examined |
| 2026-11-09 | Lead decision recorded in §2a: primary = minus hard tier (657); S1 sensitivity; S2 descriptive; exposure = actual access or derived labels; builder fixed (only the final split skipped) + test | lead decision on metadata counts, before any model or target statistic touched the new sites |
| 2026-11-10 | v0.3 (§2b): legacy tier (57) + S1b with equal prominence (lead ruling; v2-independence verified); 1 chloride-eligible site added to the hard tier (656 primary); archived forecasts in validation (perfect-prog for training only) + train-block bias correction; enforcing guard + lock + block-list + pre-claim audit; unified reference window; spatial-only primary + S-T sensitivity; grab-sample and Orbit/Lite overlaps disclosed; honest power; DST IV gaps | pre-registration review 2026-11-09 evening |
| 2026-11-10 | v0.4 (§2c): v1 outcome-level sites (12) moved to the hard tier (lead ruling C-F1); tiers v2 (hard 85, legacy 46; primary 644); power restated; IV-rule split v2 (C-F2); bias-correction limitation; repo-wide static test; unlock scorer-only | delta review 2026-11-10 |
| 2026-11-10 | v0.5 (§4a): numeric forecasts (median), 50/80/90 % prediction intervals with coverage and width, MAE/RMSE/bias and skill vs persistence/climatology required for every model × variable × horizon; secondary/descriptive; EW-v3-N probabilities additional | user requirement relayed by the lead; before any freeze or new-site model output |
| 2026-11-10 | IV-rule development phase: no variant met both criteria; option (i) applied automatically (EW-v3-N H1 non-confirmatory; DV-based primary; IV-only sites secondary) | pre-specified fallback (lead ruling) |
