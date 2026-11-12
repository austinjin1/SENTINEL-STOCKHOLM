# EW-v2 — Amendment L1: live prospective trial (Test-C)

**Status:** DRAFT v0.4 — **ON HOLD (user decision 2026-11-10; see §12)**; not frozen. Written 2026-11-09 by M1; v0.2 the same evening, after the pre-test
review "EW-v2 live trial — pre-test review (2026-11-09)" in REVIEW_REPORT_PRE_TRAINING.md (C-L1..C-L3, W-L1..W-L13).

**How it takes effect.** The lead writes `docs/preregistration/FREEZE_ew_live.json` (§11). The lead then pushes it,
together with this file, the config, the series map and the issuing code, in one dated public GitHub commit before
2026-12-01. At the freeze this file is set read-only and **never edited again**: post-freeze deviations go to
`docs/preregistration/ew_v2_live_deviations.md`.

It is **not** registered on OSF (addendum A1). Reports must say "pre-specified and frozen in a public GitHub commit
before the trial started (not registered on OSF)".

**Authors:** Austin Jin, Bryan Cheng (AI-assisted). **Amends:** `ew_v2_protocol.md` §11. `ew_v2_protocol.md` and
`FREEZE_ew.json` are hashed and unchanged. Nothing in their §1–§10 depends on this amendment.

## 0. Prior knowledge: Test-A results were known when this was written
This amendment was written **after** the single EW-v2 Test-A pass (2026-11-09, 12:05–14:01 EST; registry
`ew_v2_test_score`; post-results review "EW-v2 Test-A — post-results review (2026-11-09)"). The authors knew:
- **Primary.** H1 had 43 evaluable onsets in 11 basin groups, H2 9 in 3, H4 9 in 2. All three were underpowered and no
  claim was possible.
- **Descriptive BSS of the stack vs climatology:** H1 +0.013 [−0.004, +0.038]; H2 −0.015; H4 +0.019.
- **H1 alarms.** Realised **1.99** alarm-days per 30 d against a budget of 1. POD 0.86 vs a placebo POD of 0.33 and
  a climatology POD of 0.49.
- **H1 tiers.** The < 4 mg/L tier was null: BSS −0.002 on 56 onsets.
- **H2.** Its alarms carried no information beyond season.

**What the Test-A results were NOT used for:**

| Choice | Basis (no Test-A value used) |
|---|---|
| Hazards {H1 < 3 mg/L, H2 chronic, H4} and their definitions | The frozen EW-v2 primary family. No tier substitution: the < 4 mg/L tier (the only Test-A cell with ≥ 50 onsets) is **not** promoted. |
| H1/H4 model | `ew_live_variant_cal_nosubdaily` (no sub-daily features), calibrated 2026-11-08 19:25–19:42 EST, **before** the Test-A pass. **How the file is bound to the pre-Test-A run (review W-L12):** the file's own SHA-256 (`942440b3…`) appears in no pre-Test-A public record. It is bound through: <br>• its `config_hash 89f6ef25…`, which equals that of the 402 registry rows `ew_v2_val_oof` written 2026-11-09T00:44Z; <br>• the TRAINING_LOG entry (overrides `use_subdaily=false`, H1/H4); <br>• protocol v0.3 §5 ("9th calibrate run … 2026-11-08 19:25"); <br>• the H1 stack val-OOF BSS 0.0213 [−0.010, 0.041] quoted in the handoff. <br>All of these are in public commit cef72b6, pushed 10:03 EST, before the Test-A marker (12:05). The current file reproduces that metric (0.021286 [−0.0103, 0.0408]). The method (stack for H1 and H4) follows the pre-existing rule in `sonde_ew_freeze_candidate.py:59`. |
| H2 model | The frozen EW-v2 H2 stack (thresholds sha `3c243c1b…`), unchanged. |
| Platt maps, stacks, **alarm thresholds** | Read byte-identically from those files, SHA-256-verified at every issue. **Not re-tuned.** The Test-A overshoot of the H1 budget is **not** corrected, and realised rates are reported as they are. |
| Sites | Mechanical metadata rule (§3); all roles, all-or-none. |
| Alarm budget, leads, primary lead, endpoint, power rule, Holm family | As in `ew_v2_protocol.md` §6–§7. |

**Decisions made after Test-A, with their basis:**
1. **Issue span frozen now: 2026-12-01 → 2028-09-30, one continuous live block.** The confirmatory window (WY2027
   alone, or WY2027+WY2028 pooled) is chosen by the pre-specified count-only H1 rule (§8). Basis: USGS water-year
   approval timing and the train/val power projection. The whole span is frozen up front so the extension needs no
   mid-trial re-freeze (review C-L2).
2. **H2's primary window is the whole trial, not the winter.** Basis: train/val month shares (§8): only 8 % of H2
   onsets fall in Dec–Mar, most in Jul–Sep. This **cannot affect any confirmatory outcome**: H2 projects < 1 onset in
   winter and ≈ 9–20 over the whole span, against a threshold of 50.
3. **All roles in the live population.** This is a temporal hold-out, as RESEARCH_PLAN §5.1 names "Test-C live" for the
   benchmark sites. The test-role cell is reported separately.

**Departures from RESEARCH_PLAN §11 item 1 (disclosed):**
- no OSF registration (addendum A1);
- no "one analysis date per season": the looks of §7–§8 replace it;
- the external challenge scoring (NEON / VERA / usgsrc4cast) is not part of this amendment;
- the models were frozen 2026-11-08/09, not by 2026-11-15.

## 1. Objective and hypotheses
The trial tests, prospectively, whether frozen EW forecasts issued daily from **provisional** real-time inputs have
skill over the identically recalibrated site × season climatology for hazard onset within 3 days. It runs at the
EW-v2 benchmark sites that report in WY2027–28. WY2027–28 values did not exist at calibration.
- **L-a (primary, per hazard in {H1, H2, H4}):** BSS of the issued stack vs the issued climatology at h = 3 d > 0, on
  Approved labels.
- **L-b (secondary):** ΔBS vs persistence, the lag logistic and the event GBM (paired, basin-group bootstrap;
  Diebold–Mariano).
- **L-c (secondary):** POD ≥ 24 h at the frozen budget vs the season-matched placebo POD (cluster bootstrap).

**Stated expectation (review W-L7).**
- **H2 and H4 cannot be confirmed:** even over both years they project ≈ 9–20 and ≈ 11–15 onsets.
- **At best, a confirmatory claim is possible for H1 alone, most likely only after WY2028.** Power at the disclosed
  Test-A effect (BSS +0.013, used for power only) is about 0.19 with 50 onsets and about 0.30 with 80. At the variant's
  val-OOF effect (+0.021) it is about 0.50 and 0.71.
- **A non-rejection is not evidence of no skill.**

## 2. Models (frozen; no fitting during the trial)
| Hazard | Calibration file | SHA-256 | Selected | Inputs |
|---|---|---|---|---|
| H1 DO-min < 3 mg/L | `sentinel_runs/sonde/ew_live_variant_cal_nosubdaily/event_calibration.json` | `942440b3697c325918e03ae8b641cb9afa0680c2a637702f3e665daa8fa8351d` | stack | daily tier (12 variables), no weather, no sub-daily |
| H4 NO3+NO2 > 10 mg/L | same | same | stack | daily tier + lagged gridMET (lag 1 d) |
| H2 chloride 4-d > 230 mg/L | `sentinel_runs/sonde/ew_v2_cand_bag5_wxrunoff/event_calibration.json` (frozen EW-v2) | `3c243c1b98f1fea5f741915a9f9c54477e4c84658de661e391ba17be34d0aa6e` | stack | daily tier + derived chloride (frozen regressions) + lagged gridMET + snow proxies |

- **Issued methods.** All methods in the files are issued: climatology, persistence, threshold, lag logistic, event
  GBM, monotone GBM and stack. The q-methods are not issued.
- **Verification and archiving.** Every artifact is SHA-256-verified at every issue. Both calibration directories are
  made read-only at the freeze and mirrored publicly with the issue files at scoring, so third parties can recompute
  the forecasts.
- **This trial does not test the frozen Test-A H1/H4 models**, which use sub-daily features.

## 3. Sites (`configs/live/ew_v2_live.json`; metadata only, no data value read)
A site enters for a hazard when all of the following hold:
- it is in `sonde_split.json` (any role);
- its hazard variable has data starting on or before 2020-09-30;
- its last value is on or after 2026-10-01 (2026-11-08 pull);
- H2 only: it has an eligible frozen SC→Cl regression.

| Hazard | Sites | train / val / test role | Basin groups |
|---|---|---|---|
| H1 | 379 | 220 / 33 / 126 | 51 / 10 / 13 |
| H4 | 43 | 21 / 6 / 16 | — |
| H2 | 25 | 9 / 7 / 9 | — |

At each issue, sites without training climatology are reported as unforecastable, not imputed.

## 4. Inputs
- **USGS.** Provisional daily values from the USGS Water Data OGC API (`daily` collection), fetched at issue time for
  [max(D − 45 d, 2026-10-01), D − 1]. Each fetch is stored read-only with its SHA-256. The key is read only by the
  existing loader.
- **QC rule (frozen; reviews W-L6, W-D1).** `sonde_daily.clean_long` with the same plausibility ranges and clipping
  as the training tier. Each OGC qualifier's handling is **derived from agreement with the legacy training tier** on
  val-block data (`derive_qualifier_map.py`). The data are the WY2022–23 cached OGC records at train/val-role sites,
  single-series keys, single-qualifier values. Rule: mask if ≥ 50 % of the values are absent in legacy; keep if ≥ 90 %
  are equal; otherwise mask. Exact names are matched.

  | Handling | Qualifiers (evidence) |
  |---|---|
  | **masked** | ESTIMATED (100 % absent in legacy); LESSTHAN / GREATERTHAN (100 %); **EQUIP** (71 %; 87 % on hazard variables); **MAINT** (95 %); and, not observed on val data, INCOMPLETE, MAINTENANCE, BACKWATER, FLOOD, PARTIAL |
  | **kept** | **FORCEINTERPOLATION** (100 % equal); **REVISED** (90 % equal, 0 % absent on hazard variables); ICE, DRY, SEASONAL, UNAVAIL, DISCONTINUED |
  | **masked and counted** | any other (unknown) qualifier |

  The first derivation run counted values carrying several qualifiers. It was revised to single-qualifier values,
  with the thresholds unchanged, because co-occurring qualifiers confounded the attribution. The table is enforced by
  `test_qualifier_table` / `test_qualifier_map_matches_derivation`.
- **Series choice (review C-L3).** The frozen map `configs/live/ew_v2_live_series_map.json` picks the OGC series
  matching the 2026-11-08 snapshot over the pre-trial window 2026-10-01..11-07. If a mapped series has **no record in
  the last 7 days** of the fetch window (sensor replaced, relocated or discontinued), all series of that key are kept
  and the frozen `splice_site` rule decides. Those keys are listed in the manifest (`fallback_keys`). The same map and
  fallback are used for the label snapshots (§6).
- **Cube.**
  - **Contents.** The training block (WY2008–21) comes from the pinned physical snapshot directory
    `processed/sonde_daily_refresh_20261108`, with a per-site SHA-256. The live window starts 2026-10-01.
  - **Pins.** A missing pinned site file is refused. The chloride regressions, `sites.parquet`, the gridMET cells and
    the split file are SHA-256-verified at every issue and refused if missing or empty (review W-L1).
  - **Excluded years.** WY2022–WY2026 never enter the cube, and the builder asserts this. Features need at most 34
    days, so every issue depends only on WY2027–28 inputs and the training block.
- **Derived chloride (H2)** is computed exactly as `build_chloride_derived.py`: float32, the 4-day mean over stored
  rows, and no `cl_mgl_max` without an SC-max series.
- **Weather.** gridMET, with the same product, cells and conversions as `forcings_daily`. Recent days are
  "provisional/early" and are revised later. **If the gridMET fetch fails, the issue goes ahead with NaN weather**:
  this affects H2/H4; H1 uses no weather.
- **Environment (reviews W-L2, C-D2).** Issuing and scoring run in a **dedicated** conda environment,
  `sentinel2-live`. It is a clone of `sentinel2` with the same library versions, so the replay is identical, plus
  `opentimestamps-client` 0.8.0, and no other workstream uses it. Lockfiles
  `configs/live/sentinel2-live.{conda-explicit,pip-freeze}.txt` and the SHA-256 of the interpreter plus every
  installed `name==version` are in the freeze record. Live issuing refuses on a mismatch, and every manifest and
  hash-log row records the environment hash.

## 5. Issue procedure and issue-time evidence (`scripts/live/`; review C-L1)
1. **Schedule.** The owner's crontab runs, with `CRON_TZ=UTC`, `0 12 * * * …/run_daily.sh live` daily at 12:00 UTC
   (review W-L5). Issue day D uses data through d = D − 1 and forecasts P(onset in (d, d+L]) for L = 1, 3, 7.
   **On-time** means issued ≤ D 23:59 UTC. Inputs are capped at d, so a later run on the same day gains no
   information.
2. **Guard.** `issue.py --mode live` refuses without `FREEZE_ew_live.json` (which must contain no placeholders), and
   on any mismatch of:
   - the config, series map or issuing code (import closure);
   - the environment lock;
   - the calibration files and every artifact;
   - the static pins or the snapshot site files.

   It never issues for a future day. Past days need `--late`; a late issue is flagged and is never primary.
3. **Issue rows.** The issue-time rule (provisional values allowed):
   - day d is observed and is not an event day;
   - an onset in (d, d+L] is not excluded by the refractory rule;
   - the variable is forecastable (last value ≤ 3 d old, training climatology exists);
   - H2 also needs an SC issue row.

   Test `test_issue_rows_superset_of_frozen_event_rows_on_val` shows that these rows contain the frozen label-valid
   rows on the val block (review W-L10).
4. **Atomic day (review W-L4).** Files are built in a temporary directory, made read-only, then renamed to
   `issues/<D>/`, after which the `issued` row is appended. Stale temporary directories of crashed runs are removed.
   An issue directory without an `issued` row is an **orphan**: `verify_hashlog.py` lists it, and it is never primary.
5. **A row for every scheduled day.** `run_daily.sh` traps EXIT, so every scheduled day ends with a chained row in
   `results/live/ew_v2_live_hashlog.jsonl`:
   - `issued`: file SHA-256s, on-time flag, code, config, freeze and environment hashes;
   - or `failed`: stage, exit code, exception class and the orphan flag.

   At most one `issued` row exists per day. A scheduled day with **no row at all** is "missing" and is reported.
6. **Independent same-day timestamp: OpenTimestamps only (reviews C-L1, C-D1).**
   - **Stamping.** Right after the row is written, `stamp.py` writes the stamped file (the canonical row bytes, whose
     SHA-256 is `row_sha256`). It then runs the **official client** `ots stamp` against three public calendars. Only a
     hash, with a client nonce, leaves the host.
   - **Activation.** The backend is set **in the freeze record from the start** (`timestamp_backends: ["ots"]`), after
     the project owner approves hash-only submission and the one-shot `ots_selftest.py` succeeds. The record is never
     edited afterwards. Dry runs never submit.
   - **No GitHub backend.** GitHub comments can be deleted without trace and grant no eligibility.
7. **Deletion-proof evidence record.**
   - Every evidence file (row file, `.ots` proof, `.evidence.json`) is listed with its SHA-256 in the **next** hash-log
     row (`evidence_refs`). That row is itself chained and stamped.
   - If a referenced evidence file is later missing or altered, the scorer **restores it from the public pushed copy**
     when that copy matches the committed SHA-256. Otherwise the file is reported as unrecoverable, and its row counts as
     **"issued but not eligible"** (review C-E2). The look is never refused for this.
   - **Why proofs are hash-committed rather than fetched from calendars by the row hash** (as the review proposed): the
     OTS protocol retrieves upgrades by the commitment inside each proof, which is derived from the calendar's reply
     and the client nonce. A proof cannot be found from the row hash alone, so its file must be preserved, and
     committing its hash makes deletion detectable.
8. **Primary eligibility (scorer only; `ots_verify.py`, outside the issuing closure; review W-D2).**
   - **Third-party verification.** For each row, a temporary copy of its proof is upgraded from the calendars. The
     proof's file digest must equal the row's SHA-256. Every Bitcoin block-header attestation's merkle root and block
     time are checked against two independent explorers (blockstream.info, mempool.space), which must agree.
   - **Existence time** of row i = min(own attestation time, attestation time of any later row), since later rows chain
     to it.
   - **Rule.** An issue is primary-eligible iff `issued`, `on_time`, and existence time ≤ D 23:59:59 UTC. Every other
     issued row goes only into the "all on-time" sensitivity analysis.
   - **Refusal.** The scorer refuses to claim a look if `ots_verify.self_test()` fails: missing client, < 2 calendars
     reachable, explorers unreachable, or explorers disagreeing on the genesis merkle root. A broken verifier therefore
     cannot silently empty the primary.
   - **Offline check.** `verify_hashlog.py` checks the chain, files, evidence references, missing/failed days and
     orphans offline, and never uses git dates.
   - **Residual (reviews W-D7, C-E2).**
     - **Deleted proof.** A day whose proof is deleted before the next row references it (up to about 24 h of outcomes
       in view; day-D values appear around 04:00 UTC on D+1) is **"issued but not eligible"**: its row exists, its
       attestation does not.
     - **Deleted row.** A day whose row is deleted counts as **"missing"**.
     - **Reporting.** Every look reports both kinds of day (§7).
     - **Monitoring.** The share of rows attested before D 23:59 is monitored during December 2026, and the lead
       reports it.
9. **Code changes during the trial.** Allowed only for operational faults, as dated entries in the deviations file.
   Each must pass the val replay identity check (§10) before use. Issues made before a fix stay as issued.

## 6. Labels (one frozen path; review W-L9)
- **Label snapshot.** For each look, `build_label_snapshot.py` fetches the OGC `daily` values of all configured sites
  for [2026-10-01, targets end + 8 d] **after the look's cut-off**, and processes them through:
  1. `ogc_long` (§4 qualifier rule);
  2. the frozen series map with fallback;
  3. `clean_long` + `splice_site`;
  4. per-site frames, with a manifest content SHA-256.

  **Pins (review W-D3).** The scorer freeze `FREEZE_ew_live_score.json` is written, OTS-stamped and pushed **before** the
  first label snapshot. The label builder refuses without it and records its SHA-256. After retrieval, the lead writes a
  **separate, append-only** `FREEZE_ew_live_labels_<look>.json` holding the label content SHA-256, the manifest
  SHA-256 and the scorer-freeze SHA-256. The scorer freeze is never rewritten. `build_label_snapshot.py` and
  `ots_verify.py` are part of the scorer code hash.

  The training block comes from the pinned snapshot.
- **Labels** are computed with the frozen `events.hazard_labels`, on Approved values only (provisional = missing),
  under the frozen `hazards.py` rules.
- **Path identity check (no outcome data; val block, train/val roles; `build_label_snapshot.py --identity-check`).**
  The OGC path is compared with legacy-snapshot daily values and onset days for WY2022–23. Result: §10.
- **Series choice at multi-sonde sites (review W-D5).** Labels at multi-sonde sites depend on which series is chosen.
  Applying the 2026-era map to 2022–23 moved the onsets at 2 sites: 0 of 8 legacy onsets coincide there. The label
  snapshot applies the 7-day fallback **once**, at the end of its window, whereas issuing applies it daily, so after a
  sensor swap the inputs and the labels may use different series. A pre-specified sensitivity analysis therefore
  excludes the multi-series hazard site-variables listed in the label manifest (`multi_series_keys`).
- **Scored rows** must be label-valid **and** have an issued forecast. Rows where any issued method has a non-finite p
  are dropped for every method and counted. Label-valid rows without an issue are counted by site and month. Late
  issues are never primary. A site missing from a day's fetch has no row that day.

## 7. Looks and endpoints (each look scored once; `score_live.py`)
| Look | Not before | Issues | Targets | Labels | Content |
|---|---|---|---|---|---|
| `prov_interim` | 2027-10-15 | 2026-12-01..2027-09-30 | ≤ 2027-09-30 | provisional | **operational only:** issuance coverage (issued / failed / missing days), realised alarm-days per 30 d, evaluable onset counts. **No probability is read; no BSS/DM/POD.** |
| `count_only` | 2028-06-30 (WY2027 approval cut-off) | 2026-12-01..2027-09-30 | ≤ 2027-09-30 | Approved | evaluable H1 onsets + basin groups in the primary population (key columns only). Decides the final look (§8). |
| `final_wy2027` | 2028-06-30 | 2026-12-01..2027-09-30 | ≤ 2027-09-30 | Approved | confirmatory, **only if** count_only found H1 powered |
| `final_pooled` | 2029-06-30 (WY2028 approval cut-off) | 2026-12-01..2028-09-30 | ≤ 2028-09-30 | Approved | confirmatory, **only if** count_only found H1 underpowered |

**Common to all looks.**
- **One continuous `live` block** (2026-10-01..2028-09-30). Its 42-day embargo applies only at its start, before
  the first issue; there is **no internal embargo at 2027-10-01**.
- **Unused issues.** If WY2027 is decided confirmatory, WY2028 issues are not scored. Issuing still continues to
  2028-09-30 for operational reporting.

**Primary population.** All configured sites (all roles) × **primary-eligible** issues (§5.8), h = 3 d.

**Day classes (every look; review C-E2).**
- **Classes.** Each issue day in the look's window is exactly one of:
  - issued and eligible;
  - issued on time but not eligible (no qualifying attestation, or an unrecoverable proof);
  - issued late;
  - failed only;
  - missing (no row).
- **Reporting.** Each look reports the counts and dates of each class, and, for H1 at L = 3, the mean onsets per day in
  each class's forecast window. Issue day D, with data through d = D − 1, covers onset days **[D, D + L − 1]** =
  (d, d + L]. This fixes the earlier off-by-one window.

**Pre-specified bound (review C-E2).** If **more than 3 %** of the on-time issued days in a final look's window lack a
qualifying attestation, the "all on-time issued rows" population becomes **co-primary**. A hazard is then claimed only
if it succeeds (not underpowered, Holm rejection) in **both** populations.

**Primary statistic, test and decision.**
- Pooled BSS of the issued stack vs the issued climatology.
- Two-stage basin-group > site-month bootstrap, B = 2,000; one-sided p.
- **Holm over {H1, H2, H4}, m = 3, in both final looks** (H2/H4 pooled exactly like H1).
- **Power rule:** < 50 evaluable onsets or < 2 basin groups → p = 1, no claim.
- **Success:** not underpowered, and rejected by Holm.
- Basin-group counts are reported next to every p; with < 10 groups the CI is labelled approximate.

**Secondaries (final looks, descriptive):**
- Cells and populations:
  - test-role cell (spatial + temporal hold-out), train/val-role cells;
  - "all on-time issued rows" population.
- Comparisons and alarm performance:
  - L-b and L-c;
  - POD ≥ 24 h with **realised** alarm-days per 30 d and the share of sites over budget.
- Probabilistic diagnostics: Molchan, reliability + ECE, REV.
- Controls of `ew_v2_protocol.md` §8 where they apply:
  - placebo;
  - random alarms;
  - circular shift (labelled: no season control);
  - label permutation;
  - negative-control sites.
- H2 Dec–Mar.

**Sensitivity analyses (pre-specified):**
- (a) all on-time issued rows, whatever their timestamp evidence;
- (b) "strictly ahead": positive rows whose onset falls on the issue day D are dropped;
- (c) sensor-fault audit: at alarm and onset rows, the provisional issue-time value vs the Approved value (fraction
  deleted, fraction revised);
- (e) multi-series site-variables excluded (§6);
- (d) input-revision audit: issued vs re-issued probabilities from Approved USGS inputs, and, for H2/H4 only, gridMET
  rows whose `gridmet_status` is "permanent". Separate descriptive script; H1 uses no weather, so for H1 this is a
  USGS-only audit.

## 8. Power and the look decision (a-priori, train/val only; `power_projection.py`, registry `ew_v2_live/power`)
**Method.** Onsets are counted with the frozen rules on Approved values at **train/val-role** configured sites, over the
train block (WY2008–21) and val block (WY2022–23). This gives a rate per observed site-year and a month share. Both are
scaled to all configured sites. These are **full-coverage projections**: real coverage is lower, rates can differ, and
evaluable onsets ≤ raw onsets.

| Hazard | Onsets / observed site-yr (train; val) | Dec–Sep share (Dec–Mar) | WY2027 (Dec–Sep) | Pooled to 2028-09-30 |
|---|---|---|---|---|
| H1 | 0.161; 0.148 | 0.87 (0.08) | 49–53 | ≈ 105–114 |
| H4 | 0.189; 0.146 | 0.82 (0.18) | 5–7 | ≈ 11–15 |
| H2 | 0.425; 0.191 | 0.81 (0.08) | 4–9 | ≈ 9–20 |

**Look decision (count-only, mechanical).** At the `count_only` look, if H1 meets the power rule in the WY2027 primary
population, `final_wy2027` is the single confirmatory look. Otherwise `final_pooled` is. The scorer enforces this:
- a final look refuses without a scored `count_only`;
- a final look refuses if it is not the look decided by `count_only`.

## 9. Scorer (`scripts/live/score_live.py`; hardened single look; frozen separately)
The scorer and the label builder sit **outside** the issuing closure (review W-L13). Their import-closure hash goes into
a lead-written `FREEZE_ew_live_score.json` **before** any label snapshot is built for any look. Per look, the scorer:
1. **Checks before the claim**, so a failed check never burns the look:
   - both freeze records (no candidates, no placeholders) and the scorer code hash;
   - the calendar gate of the look (§7) and, for final looks, the count_only decision;
   - calibration, static pins, the hash chain and every issued file;
   - the look's label snapshot against its separate pin record (§6), made under the current scorer freeze;
   - for final looks, the `count_only` result against its sealed `SHA256SUMS` and `SCORED.json` (review W-D6);
   - `ots_verify.self_test()` and the evidence references (§5.7–5.8);
   - no registry row of experiment `ew_v2_live_<look>_score`;
   - no CLAIM / SCORED / ABORTED marker.
2. **Claims:** an O_EXCL, 0444 `CLAIM.json` holding pid, host, argv and the freeze hashes, plus a `look_claimed`
   registry row with the same pid and host.
3. **Aborts on any `BaseException`:** `ABORTED.json` + `look_aborted` row; the look is closed. An aborted look is never
   resumed without a dated deviation.
4. **Seals:** results are written once, with `SHA256SUMS`, read-only. Metric rows are registered, then `SCORED.json`
   + `look_sealed` are written. A rerun is refused.
5. **Is rehearsed** with `--rehearsal` on the dry-run tree (scratch registry and markers; train/val-role sites only).

## 10. Dry-run evidence on the final code (v0.4; dedicated `sentinel2-live` environment; train/val-role sites only)

**v0.4 final issuing code `9ee1a2ac69b1cd4b…` (52 files).** Its compute code is identical to `27496e20…`: v0.4
reverted the reserved-helper import (C-E1) and added `push_evidence.py` plus its `run_daily.sh` call.
- **Val replay on `9ee1a2ac…`.** Rows are identical in all 9 cells, and max |Δp| is 3.9e-6.
- **Dry daily cycle.** `run_daily.sh dryrun 2026-11-11` wrote row 8 as `issued`, with code `9ee1a2ac…` and
  environment `e950cae0…`. `stamp.py` and `push_evidence.py` stayed dormant; verify was clean, with 0 orphans.
  - **This row is NOT an on-time example.** It is a **future-dated dry run** built from the data available on
    2026-11-10; the "on time" flag only compares the issue clock with the end of D.
  - **Dry vs live.** Dry mode accepts any date, past or future. **Live mode refuses future days** and flags past days
    as late (`--late`).
- **Zero-row day (review S-1).** Row 9, D = 2026-11-12, was built from a clearly labelled empty USGS fetch fixture
  that simulates late-arriving values. The issuer wrote a zero-row issue file with no columns. The fixed scorer (S-1,
  `score_live.py` only; issuing hash unchanged) behaves as follows:
  - it test-loads every issue file before the claim;
  - it skips zero-row files when loading;
  - it classifies days from the hash-log rows, so the zero-row day is issued, not missing.

  All three rehearsals were re-run on the fixed scorer `ef598860…` with this day included.
  - Window 2026-11-05..11-12; 8 issued days, **2 of them zero-row**: the fixture day, and the future-dated 11-11 run,
    for which USGS values of 11-10 did not yet exist, i.e. the real late-data case.
  - `prov_interim`, `count_only` and `final_wy2027` each claimed and sealed.
  - The pre-claim test-load found 6 files with rows and 2 zero-row days.
  - **Issuing hash unchanged: `9ee1a2ac…`.**

The v0.3 evidence below remains valid for the unchanged compute path.

v0.3 issuing code (import closure) `27496e202d9abf15…`; config `7693424f866239cf…`; environment lock
`e950cae026f8ca21…`; scorer code (outside the issuing closure) `313275676f36cdd3…`. Dry-run hash-log rows 6–7 were written on
this code. Earlier rows (0–5) and the `superseded_v0_1/` tree are kept for provenance.

- **Val replay (`issue.py --mode replay`).** Rows are identical in all 9 hazard × lead cells (H1 L3 58,887 = 58,887;
  H2 L3 2,619 = 2,619), with 0 NaN mismatches. Max |Δp| over all methods ≤ 3.9e-6 (H1), ≤ 6.2e-10 (H4), ≤ 5.3e-12
  (H2).
- **End-to-end daily cycle (`run_daily.sh dryrun 2026-11-10`).**
  - OGC fetch for all train/val-role sites, gridMET fetched, then issue: row 6 `issued`, on time.
  - `stamp.py` submitted nothing (disabled without the freeze).
  - `verify_hashlog.py`: chain, files and evidence references OK; 0 orphans.
- **Forced failure (`run_daily.sh dryrun 2026-11-01`).** Row 7 is `failed`, stage `issue`, error class
  "SystemExit: issue day too early …".
- **Guards.**
  - Live mode refused without the freeze.
  - A second issue for the same day was refused.
  - A real look was refused without the freeze, before any claim.
  - A sealed look's rerun was refused.
  - An abort wrote `ABORTED.json` and a registry row.
  - Deletion of a referenced proof is detected (`test_evidence_refs_detect_deleted_proof`).
- **Scorer rehearsal (final scorer; `--rehearsal`; dry-run days; provisional labels; scratch registry).**
  `prov_interim`, `count_only` and `final_wy2027` each claimed, sealed and registered (6 claim/seal rows). Issued rows
  joined label-valid rows 1:1. **Not exercised:** November has no onsets, so the control, strictly-ahead, sensor-fault
  and multi-series branches ran on zero positive rows. The OTS verification path cannot be exercised before the
  owner-approved `ots_selftest.py`.
- **Label-path identity (`build_label_snapshot.py --identity-check`, WY2022–23, 279 train/val-role sites).**
  - Single-series site-variables: DO-min values identical (0 of 152,732 days differ), H1 onsets 46 = 46.
  - H4: values and onsets identical (6 = 6).
  - Multi-series sites under the frozen splice rule alone: H1 onsets identical (12 = 12).
  - **Applying the 2026-era series map to 2022–23 changes the series at 2 sites, where 0 of 8 legacy onsets coincide
    with the map path (6 vs 2 on each side).** The map is era-specific (built from and used for WY2027–28), and a
    multi-series sensitivity analysis is pre-specified (§6, §7 e).
  - The REVISED qualifier is kept, following the §4 derivation.
- **Tests.** `tests/test_live.py` passes 33 tests in `sentinel2-live`, including:
  - status rows and hash-chain tamper detection;
  - the table-driven qualifier map and its match with the derivation;
  - series-map fallback;
  - no submission without a freeze;
  - deletion of referenced evidence detected;
  - the issuing closure excludes the scorer;
  - a cube past 2027-09-30 through issue and score;
  - `issue_rows` ⊇ frozen `event_rows`.

## 11. Freeze procedure (lead; the owner's sign-off is needed for steps 1, 7 and 8)
1. **Owner approval** of daily hash-only OpenTimestamps submissions.
2. **One-shot test.** In the `sentinel2-live` environment, run
   `python -B scripts/live/ots_selftest.py --i-have-owner-approval`. It must report a Bitcoin block time verified by
   both explorers within 12 h.
3. **Frozen series map:** `python -B scripts/live/build_series_map.py --roles train,val,test --out frozen`.
4. **Draft the record:** `python -B scripts/live/freeze_draft.py` (it prints the draft and writes nothing). The draft
   holds:
   - the SHA-256s of this amendment, the config, the series map and `FREEZE_ew.json`;
   - the issuing code hash and file list;
   - the environment lock and the lockfile hashes;
   - the calibration SHA-256s;
   - **`timestamp_backends: ["ots"]`**.
5. **Make read-only:** this amendment and both calibration directories.
6. **Write `docs/preregistration/FREEZE_ew_live.json`** (read-only). **It is never edited afterwards.** It includes
   `auto_push.enabled`, true only if the owner approves push access. `push_evidence.py`, run after every stamp, then
   commits and pushes only the hash log and proofs; until enabled it is dormant.
6b. **Make the environment read-only:** `chmod -R a-w /home/bcheng/miniconda3/envs/sentinel2-live`, so no install can
    change it during the trial.
7. **Commit and push before 2026-12-01:** this amendment, `configs/live/*` (including the lockfiles),
   `scripts/live/*`, the freeze record, an empty hash log and an empty `ew_v2_live_deviations.md`. A commit cannot
   contain its own hash, so the commit hash is recorded in the deviations file.
8. **Owner's crontab:** `CRON_TZ=UTC` / `0 12 * * * …/scripts/live/run_daily.sh live` (the script uses `sentinel2-live`).
9. **Push regularly:** commit and push `results/live/` (log + OTS proofs) at least weekly.
10. **Scorer freeze.** Before the first label snapshot (by 2027-10-15), the lead writes, stamps and pushes
    `FREEZE_ew_live_score.json` (scorer code hash). Label pin records follow per look (§6).

## 12. Deviations log (pre-freeze only; post-freeze entries go to `ew_v2_live_deviations.md`)
| Date | Change | Reason |
|---|---|---|
| 2026-11-09 (pre-freeze) | H2 primary window → whole trial; Dec–Mar descriptive | Train/val month shares (§8) |
| 2026-11-09 (pre-freeze) | Chloride derivation aligned to the frozen builder | Val replay (H2 max Δp 0.01 → ≤ 1e-11) |
| 2026-11-09 (pre-freeze) | Frozen series map for multi-sonde sites | Dry-run fetch identity (1.2 % → 0.16 % differing values) |
| 2026-11-10 (ON HOLD, user decision) | **The live trial is ON HOLD until all individual models are good.** It is not frozen. The following are NOT done: live mode, cron, the OTS self-test, any external submission. Everything stays ready and freezable: issuing code `9ee1a2ac69b1cd4b…` (52 files), scorer `ef598860ac8a3b86…`, config `7693424f…`, environment `e950cae0…`. **Any later start date requires a dated amendment of the trial window** (§7 issue windows, look dates and approval cut-offs, `configs/live/ew_v2_live.json` `trial.*`, code `LIVE_BLOCK`). The current dates (issues from 2026-12-01) are therefore void unless re-confirmed by such an amendment. | User decision via the lead |
| 2026-11-10 (pre-freeze, v0.4) | C-E1: the reserved-helper import was reverted from `live_data.py`; the issuing compute code is again identical to `27496e20…`, and a test checks that config sites never overlap the reserved list. C-E2: a deleted-proof day is "issued but not eligible", not "missing"; four day classes are reported in every look; the forecast-window off-by-one is fixed; the > 3 % co-primary bound is added. Recommended: proofs restored from the public pushed copy instead of refusing; dormant auto-push of the log and proofs; `sentinel2-live` read-only after the freeze. Final issuing code hash `9ee1a2ac…` (adds `push_evidence.py` and the `run_daily.sh` call). **First-run qualifier table (superseded; it counted values carrying several qualifiers):** BACKWATER n=79 missing 1.000 → mask; DISCONTINUED 4, 0.000 → keep; DRY 174, 0.305 → mask (flag); EQUIP 917, 0.767 → mask; ESTIMATED 8917, 1.000 → mask; FORCEINTERPOLATION 494, 0.012 → keep; GREATERTHAN 78, 1.000 → mask; ICE 642, 0.880 → mask; LESSTHAN 70, 1.000 → mask; MAINT 125, 0.960 → mask; REVISED 6302, 0.104 (equal 0.896) → mask (flag); SEASONAL 648, 0.008 → keep; UNAVAIL 9, 0.000 → keep. Final derivation: single-qualifier values (§4). | Delta review 2026-11-10 |
| 2026-11-09/10 (pre-freeze, v0.3) | C-D1: OTS-only eligibility, verified by third parties (calendar upgrade + two block explorers); evidence files hash-committed in the next row, so deletion is detected; self-test or refuse; GitHub backend dropped; eligibility moved to `ots_verify.py`, outside the issuing closure (W-D2). C-D2: dedicated `sentinel2-live` env with the OTS client and lockfiles; backend set in the freeze record from the start; one-shot `ots_selftest.py` (not run). W-D1: qualifier map re-derived from legacy agreement (FORCEINTERPOLATION kept; EQUIP/MAINT masked), table-driven test. W-D3: separate label-pin records; label builder and verifier in the scorer hash. W-D4: §10 re-run on the final code, wording corrected. W-D5: multi-series sensitivity analysis + fallback difference stated. W-D6: count_only checked against its seal. W-D7: missing days vs provisional onsets; attestation-share monitoring | Delta review 2026-11-09 evening |
| 2026-11-09 (pre-freeze, v0.2) | C-L1: a status row for every scheduled day; hash-only independent timestamps (OTS / GitHub, pluggable, disabled until approval); primary = evidence ≤ D 23:59 UTC; no git dates. C-L2: span frozen to 2028-09-30; four looks defined. C-L3: series-map fallback + manifest list + test. W-L1 static pins and missing files refused; W-L2 environment lock; W-L3 deviations file; W-L4 atomic day and orphans; W-L5 UTC cron; W-L6 unknown qualifiers masked; W-L7 honest power; W-L8 interim fenced; W-L9 label path, non-finite and late rules, identity check; W-L10 test; W-L11 re-runs on the final code; W-L12 §0 disclosures; W-L13 scorer outside the issuing closure + CI test | Pre-test review 2026-11-09 |

## 13. Limitations
- **Public record.** A git commit plus OTS/GitHub timestamps is weaker than a registry.
- **Tail deletion.** Deleting the last day before its next-day stamp shows only as a missing day.
- **Provisional inputs.** USGS and gridMET inputs differ from the training data.
- **Power.** Confirmation is possible for H1 only, with low to moderate power.
- **Timing.** Confirmatory results cannot exist before mid-2028 (WY2027) or mid-2029 (pooled).
- **Population.** Train/val-role sites were used for fitting or calibration, so the all-roles population is a
  temporal hold-out; only the test-role cell is spatially unseen.
