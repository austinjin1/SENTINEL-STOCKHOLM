# M1 SENTINEL-Sonde — architecture round v2 amendment (M1-N v0.6, M1-H, M1-SP, forcing strata)

**Status:** DRAFT v0.1 (2026-11-10, ~17:40 UTC), written by M1 for the lead's review and stamp. AI-assisted.
**Stamp:** pending. The lead records this file's SHA-256 in a `sonde/prereg_snapshot` registry row and commits it.
**Implements:** `docs/plans/architecture_research/LEAD_DECISIONS_20261110.md` (rules 1–6 and the M1 section, which
override the plan where they differ) and `docs/plans/architecture_research/M1_architecture_plan.md` §8 step 0,
items (a)–(g), plus H-2, H-3, H-5 and H-6.
**Relation to frozen files.** This is a new file. It edits nothing that is frozen or hashed. It amends
`m1n_protocol.md` v0.5 as **M1-N v0.6** by reference: where the two differ, this file wins for M1-N. The original M1
track (`sonde_m1_protocol.md`) and its report-only Test-A look are unchanged and are not delayed.
**Bars.** Nothing here lowers a bar. RESEARCH_PLAN §15.1 and `sota_campaign.md` rules 1–8 are unchanged. The M1-N
primary endpoint (`m1n_protocol.md` §4) is unchanged; adding references only makes it harder.

## 0. What exists at the time of writing (and what does not)

- **No validation number exists on any arm or track in this file.** In particular, **no NWP-based M1-N validation
  number exists.** None has been computed by M1 or by any other script reading M1 data.
  - The archived-GFS pull (`download_nwp_archive.py`, train/val split sites, `temperature_2m`, ≤ 2023-09-30) only
    writes raw hourly forecasts. It has no scoring step. The `aggregate` step, not yet run, writes forecast values and
    coverage counts only.
  - No M1 model has been trained with NWP inputs. No M1 scorer reads NWP files.
  - **Guarantee until the stamp:** no M1 script that combines NWP inputs with a target is run on any block except
    synthetic data or train-role × train-block smoke tests (rule 2). Code and unit tests only.
- **What is known** (all val-block, original track, selected on val; caveat W6): plan §1. The ens3w candidate leads
  damped anomaly by 3–12 % CRPS. A baseline-only ensemble is within ±0.01. An oracle with observed future weather
  (SENTINEL-Sim) reaches +0.25 to +0.28 for temperature at h = 3–7 d. These numbers motivate this amendment, and they
  are disclosed as prior knowledge.
- **New-site block:** locked; no value read. Every count below comes from the frozen site list's series metadata
  (`sonde_newsites_list.json`, sha `6200a77c…`, begin/end dates per variable) and tiers v2
  (`sonde_newsites_tiers_v2.json`, sha `cee6b019…`). They are metadata only.

## 1. Practical reasons (stated before any result)

| Arm / track | Practical reason |
|---|---|
| NWP forecast branch (SONDE-HF, R-NH, GBM-NWP, PF-GBM, Chronos-2-cov) | Operational sonde forecasting is issued with weather forecasts. The current models see only past weather, and the val evidence shows they sit at the ceiling of that task. |
| GEFS forecast mode (H-1) | New stations need forecasts from their first issue day. Today only issues after 2021-03-30 have archived forecasts, temperature only. Rain forecasts drive turbidity, SC dilution and nitrate flushes. |
| Auxiliary numeric targets | Users act on T max (thermal stress), DO max/amplitude and % saturation, SC/turbidity maxima, and discharge. The user asked for more predictive numeric targets. |
| M1-H (H-2) | Fish kills and permit violations happen at night-time DO minima lasting a few hours. Operators read IV (sub-daily) data. Daily means cannot say how long hypoxia lasts. |
| Forcing strata (H-3) | Warnings matter most in heatwaves and storms, where tree models cannot extrapolate. |
| M1-SP (H-5) | A different operator network and QC regime, and the research streams where metabolism is measured. This is a transfer test M1 has never run. |
| EW-H1h (H-6) | An hourly-native hypoxia onset definition gives an EW test that does not depend on the failed IV→daily rule. Descriptive by default. |

## 2. M1-N v0.6 (amends `m1n_protocol.md`; same locked block, same single hardened look)

### 2a. Candidate families and configuration allocation (item a)
Cap: ≤ 16 configurations per variable group, counted as below. Every configuration that is trained, including a
failed one, counts, and is reported next to any claim.

| Family | Status (lead decisions) | Configs | Configuration list (pre-declared) |
|---|---|---|---|
| **SONDE-HF** (plan §4.1): daily 365-d hindcast LSTM + optional hourly 14-d hindcast LSTM → state hand-off → 14-step forecast LSTM over a masked-mean NWP embedding; multi-task monotone 19-quantile heads with ordering constraints and the damped-anomaly anchor; AR inputs with observed/missing flags | approved, rank 1 | 5 | C1 base (hourly on, multi-task, quantile head, hidden 256/128); C2 hourly branch off; C3 primary-only loss; C4 CMAL head; C5 hidden 384/192 |
| **E2**: CV-weighted SENTINEL ensemble | approved | 2 | W1 weights per variable × horizon block {h ≤ 2, 3 ≤ h ≤ 7, h ≥ 10}; W2 one weight vector per variable |
| **PhiDA-Lite** (plan §4.3 with changes C1–C8) | gated (step-1 rule, §2f, and gate C3) | 4 | residual bound c ∈ {0.5, 1.0} × latent-loss weight {0.1, 0.3}. E3 (KalmanNet gain) is a 1-seed ablation, not a config |
| **Chronos-2-NWP** FT with known-future NWP covariates | gated (step-1 rule) | 2 | full FT with small batch (gradient accumulation); LoRA |
| reserve | — | 3 | used only for a failed-run repeat or a pre-declared ablation; the use is logged |

### 2b. References and strict control, with parity (item b)
- **New mandatory references** (approved to build), added to the M1-N reference set:
  - **R-NH**: Nearing et al. 2024-style hindcast–forecast LSTM, i.e. 365-d daily hindcast → state hand-off → 14-d
    forecast LSTM over NWP; CMAL head mapped to 19 quantiles; multi-site; one model per target; no hourly branch, no
    multi-task, no physics.
  - **GBM-NWP**: the current GBM-w plus per-lead NWP features. These are forecast Tair mean/max/min (P and SW where
    present), cumulative and exponentially weighted forecast degree-days over leads 1..h, GEFS member mean and
    spread, and product-availability flags. Quantile LightGBM (19 levels), not homoscedastic.
  - **PF-GBM**: GBM-NWP plus uncalibrated physics features: a regional-default air2stream water-temperature forecast,
    DO_sat(T_forecast) (Benson–Krause), Raymond K600 from Q and slope, and a two-end-member SC mixing estimate.
  - **Chronos-2-cov ZS**: zero-shot Chronos-2 with NWP as known-future covariates (already required by M1-N §3).
  - The existing references are unchanged: frozen ens3w, LSTM-w, ZS Chronos-2 uni/multi, TimesFM-2.5, damped
    anomaly, climatology, persistence, and the strict controls of v0.5.
- **Strict control E2-B**: an ensemble of baseline members only (GBM-NWP, PF-GBM, R-NH, Chronos-2-cov ZS), weighted
  by **exactly the E2 rule** (same weighting variant, same out-of-fold predictions, same horizon blocks).
  - *Ruling requested (R1):* the plan's E2-B list names an "LSTM-w-NWP" member, which was not approved to build.
    R-NH is the LSTM-with-NWP reference, so this file uses R-NH in its place.
- **Parity.** Every model that can take an input gets the same:
  - NWP products and members;
  - bias correction (§2c);
  - static attributes;
  - training sites and blocks;
  - extra training networks;
  - 2-year reference climatologies;
  - auxiliary targets, where the model can take them.

  **Tuning parity:** R-NH, GBM-NWP and PF-GBM may each use up to 5 CV configurations, the same as the best-tuned
  SENTINEL family. Their configuration lists are declared in the parity log before their CV runs.

### 2c. Forecast-mode extension with GEFSv12, and the bias-correction rule (item c; H-1)
- **Products**, combined in SONDE-HF / R-NH by a masked mean over the available products, with no zero-fill:
  - (P1) Open-Meteo Previous Runs GFS (pull running; T2m from 2021-03-24; P/SW from 2024-01-20);
  - (P2) the **GEFSv12 reforecast** (AWS `noaa-gefs-retrospective`, 2000–2019, 00 UTC, c00 + p01–p04);
  - (P3) the **GEFSv12 operational archive** (AWS `noaa-gefs-pds`, from the v12 implementation 2020-09-23; 00 UTC).

  GBM-type models get per-product features and availability flags.
- **Model-version rule.** Only GEFSv12 is used. Operational files dated 2020-01-01..2020-09-22 are GEFSv11 (a
  different model), so they are **not used**. Issues whose lead days fall in that gap use the other available
  products, else past-weather mode, and are reported as such.
- **HEAD-verified sizes** (2026-11-10, one file each):
  - Reforecast: one GRIB2 per variable × init × member, 0.25°, 80 three-hourly steps to day 10, with `.idx` byte
    index. `tmp_2m` 39.0 MB, `tmax_2m` 34.6 MB, `tmin_2m` 44.8 MB, `apcp_sfc` 24.8 MB, `dswrf_sfc` 26.4 MB per init.
    Byte ranges for lead days 1–7 (56 of 80 steps) take about 70 %.
  - Operational: `pgrb2sp25` per member × step, 14.4 MB, `.idx` with TMP/TMAX/TMIN/APCP/DSWRF messages
    (≈ 0.4–0.5 MB each).
- **Transfer estimate (c00 only; four variables tmax, tmin, apcp, dswrf; lead days 1–7 by byte range):**
  - reforecast WY2008–2019: 4,475 inits × ≈ 91 MB ≈ **0.41 TB**;
  - operational 2020-09-23..2025-09-30: 1,834 inits × 56 steps × 4 messages × ≈ 0.45 MB ≈ **0.18 TB**;
  - total ≈ **0.6 TB** for c00. Each extra member adds about the same again.
  - **Default: c00 only**, members p01–p04 later only if the lead approves after the c00 extraction.
  - Stream-extract-delete per file; nothing global is kept.
- **Kept extraction:** point values at the split train/val sites (train and val blocks) and the 729 new sites (whole
  window), aggregated to local-standard-time days per lead (Tmax, Tmin, P sum, SW mean). Stored as int16 / 0.1-unit
  parquet, **< 1 GB** (≈ 0.75 GB raw estimate before compression); `df -h /` ≥ 60 GB is checked before every batch.
  The new-site point extractions are forcing data, not targets; they are written under the reserved new-site guard
  directory.
- **Bias correction** (replaces the v0.3/v0.4 rule for GEFS; the GFS P1 rule is unchanged). For each product and
  lead day, a per-site linear map from forecast to gridMET is fitted on **train-block overlap only**:
  - GEFS reforecast: WY2008–2021-09-30 ∩ product availability;
  - GFS: 2021-03-24..2021-09-30.

  The map is per month-of-year where the site has ≥ 60 overlap days in that month, else a pooled HUC2 map. New sites
  use their **reference window** (first 730 days; design B) for the site map when the product covers it, else the
  pooled HUC2 map fitted on train-role sites. No validation or test target is used.
- **Training regimes** (fixed): stage 1 is perfect-prog with a lead- and season-dependent error model (GEFS
  reforecast-minus-gridMET residuals at train-role sites, WY2008–2019). Stage 2 fine-tunes on real reforecasts over
  the train block. Validation, selection, calibration and ensemble weights use **archived issued forecasts only**
  (M1-N C-N2), for every model alike.
- **Forecast-mode coverage (metadata projection, primary population 644, design-B issue windows):**

  | Variable | Sites | HUC4 | Issue site-years | Share with P1 (issues ≥ 2021-03-30) | Share with GEFSv12 (P2 ∪ P3) |
  |---|---|---|---|---|---|
  | DO | 199 | 61 | ≈ 627 | 0.46 | 0.93 |
  | SC | 392 | 86 | ≈ 1,869 | 0.37 | 0.95 |
  | NO3 | 11 | 9 | ≈ 17 | 0.55 | 0.96 (descriptive) |

  The DV-based primary cells of v0.4 (DO 51 sites / 24 HUC4; SC 155 / 54) are unchanged.
- **Reporting.** Forecast mode with GEFS is the primary mode. GFS-only mode (P1 only) and past-weather mode are
  reported as separate cells. A missing product is reported, never imputed with observed weather at test time.
- **If the lead does not approve the GEFS transfer:** M1-N stays as v0.5 (forecast mode only for issues ≥ 2021-03-30),
  and the rest of this file applies unchanged.

### 2d. Auxiliary numeric targets (item d), secondary and descriptive
Each is a numeric 19-quantile forecast in original units, reported through `sonde_pi_report.py` (M1-N §4a: median,
50/80/90 % PI with coverage and width, MAE/RMSE/bias and skill vs persistence and climatology, per-site medians,
cluster-bootstrap CIs):
- T max and min (and 7-day average of daily max);
- DO max, DO diel amplitude and DO % saturation (deterministic transform; Benson–Krause with pressure from elevation);
- SC daily max and turbidity daily max;
- log discharge mean and gage height (auxiliary inputs and outputs, never a decision).

They never enter or re-weight the §4 decision. The loss weights are fixed per-target inverse-scale normalisations,
set before training. A "primary-only" configuration (C3) is the declared ablation.

### 2e. Cells and strata (item e; H-3), secondary
- **Forcing-defined strata**, selected by forcing only, never by outcome:
  - **X3-heat**: issue rows whose bias-corrected NWP Tair at lead h exceeds the site's reference-window P95 of daily
    Tair (gridMET);
  - **X1-storm**: NWP 3-day precipitation sum over leads h−2..h above the reference-window P95.

  A stratum is analysed only if it has ≥ 5 HUC4 with stratum rows at the variable; otherwise it is descriptive.
  Metric: median per-site CRPS skill vs the best reference, with a HUC4 bootstrap.
- **Cells:** streams-only (site class from metadata); has-Q (co-located discharge in the issue window); GEFS vs
  GFS-only vs past-weather mode; DV vs IV-only (v0.4).

### 2f. Gates and drop rules (item f)
- **Step-1 information-use test.** It is pre-declared, runs after the stamp, and uses **train-role sites × train
  block only**.
  - Inner split: 5-fold basin-group split of the train-role sites (seed 20261110); fold 0 is held out.
  - Inputs: perfect-prog gridMET for the target days plus the stage-1 NWP error model.
  - Contrast: smoke SONDE-HF (config C1, 1 seed) vs GBM-NWP (default config).
  - Statistic: for T mean and DO mean, the median per-site CRPS skill of SONDE-HF vs GBM-NWP, averaged over
    h ∈ {3, 5, 7}, on the held-out fold × train-block issue dates (WY2016–2021).
  - **Rule:** if the statistic is < 0.01 for **both** T and DO, PhiDA-Lite and Chronos-2-NWP are **not built**, and
    their GPU budget goes to M1-H. Otherwise both proceed, PhiDA-Lite still subject to C3.
  - The result is reported whichever way it falls.
- **Gate C3 (PhiDA-Lite)**, on train-block inner CV (same split), before any val use. All three must hold:
  - (i) the physics-only branch beats damped anomaly for T at h ≥ 3 (median per-site CRPS skill, basin-group
    bootstrap CI > 0 at each of h = 3, 5, 7);
  - (ii) Sim's metamorphic checks: M1-consistent ≥ 0.70 and M4-night ≥ 0.80;
  - (iii) latent GPP Spearman ≥ 0.5 vs Powell labels at inner-held-out Powell sites.

  **If any check fails, PhiDA-Lite is dropped** and reported as dropped, with the three numbers.
- **Upstream-context GAT:** not approved in this round; the metadata gate of plan §4.5 is recorded only.

### 2g. Upstream-input date rule (item g)
Upstream-sensor features, if a later amendment approves them, may come only from **train/val-role split sites** and
only for dates **≤ 2023-09-30**. They may never come from test-role sites, reserved lists, the new-site block or
WY2024–25 data. After 2023-09-30 the feature is missing, with a flag. They would be given to every comparator.
*Ruling requested (R2):* confirm this rule, or the stricter "no upstream inputs in M1-N".

### 2h. Selection, early stopping and the freeze candidate
- **Selection data:** 5-fold basin-group CV over train+val-role sites × val block. Folds are fixed before any CV run
  (seed 20261110); archived issued forecasts are used in every validation step.
- **Early stopping** on an inner 15 % of train-role basin groups, never on the val block (fixes W6).
- **Within a family:** the config with the best mean out-of-fold CRPS skill vs damped anomaly over the primary
  horizons {1, 2, 3, 5, 7} and the primary variables (T, DO mean, DO min, SC, turbidity mean).
- **E2 / E2-B weights:** non-negative CRPS-optimal weights on out-of-fold predictions (W1 or W2, selected by the same
  criterion), fixed in the freeze record. For the comparison between E2 and SONDE-HF, E2's score is **cross-fitted**:
  weights are fitted on 4 folds' out-of-fold predictions and evaluated on the 5th.
- **SENTINEL candidate for the M1-N primary:** whichever of {selected SONDE-HF, E2} has the higher cross-fitted mean
  CV score. If they are within 0.002, SONDE-HF is chosen (fewer parts). One network serves every horizon; there is no
  per-horizon switching.
- **Final:** 3 seeds of the selected configuration(s), trained on train-role sites × train block with inner early
  stopping. The members, weights, checkpoints and the code hash go in `FREEZE_m1n.json` (lead-written).
- **Primary endpoint and decision:** unchanged (`m1n_protocol.md` §4). The reference set now includes R-NH, GBM-NWP,
  PF-GBM and Chronos-2-cov ZS. E2-B is reported next to it as the strict control.

## 3. M1-H — sub-daily DO forecasting and hours of hypoxia (H-2): protocol

- **Targets: hourly IV values.** M1-H forecasts the **hourly values themselves**: the hourly means of USGS
  instantaneous (IV) DO and water-temperature readings, as built in the hourly tier. **The failed IV→daily rule
  (`m1n_protocol.md` §2b–§2c) does not apply.** That rule tried to reproduce the USGS *DV daily statistics* from IV
  data, and it failed that equivalence test. M1-H never claims or needs DV equivalence. Every M1-H target is defined
  on the hourly IV values and is named as such ("hourly-IV DO"), never as a DV statistic.
- **Targets and horizons:**
  - **Primary:** hourly-IV DO (mg/L) at lead times +6, +12, +24, +48 and +72 h from the issue time (00:00 local
    standard time);
  - **secondary:** hourly-IV water temperature at the same leads;
  - **secondary:** daily hypoxia hours (number of hourly values < 3 mg/L and < 5 mg/L) for days 1–7 after the issue;
  - **secondary:** daily minimum of hourly-IV DO (an IV-defined statistic, not a DV-equivalent).

  All are 19-quantile forecasts; the numeric-forecast report applies.
- **Data.** Training is on train-role split sites' hourly tier × train block (hourly tier: 423 train-role and 76
  val-role sites with any hourly series; DO subset counted before the CV, counts only). Selection is on 5-fold
  basin-group CV over train+val roles × val block. Never: new-site data, test-role sites, Test-A/B, WY2024–25. Only
  Approved values are scored.
- **Unseen test block:** the locked new-site block, primary population minus the hard tier (644 sites), hourly-IV DO.
  - Reference window: the first 730 days of hourly DO at the site (design B); issue window = reference end + 42 d →
    2025-09-30.
  - Scored in the **same single hardened M1-N look** (one claim, one sealing). It is a separate track with its own
    decision and is not in the M1-N Holm family.
- **Comparators (parity: every one gets hourly inputs, the same NWP and the same statics):**
  - naive: hourly persistence; diel persistence (same hour yesterday + the daily-mean anomaly); hourly climatology
    (reference window, hour-of-day × month);
  - GBM-hourly: quantile LightGBM with hourly lags, hour-of-day, NWP;
  - LSTM-hourly (multi-site);
  - zero-shot Chronos-2 (uni and multi) on hourly context;
  - R-NH-hourly: R-NH with an hourly forecast branch.

  The candidate is SONDE-HF-hourly: the SONDE-HF code with hourly heads.
  - **Caps:** ≤ 8 configurations for the candidate family; ≤ 4 for each learned reference.
  - **Selection:** as §2h, with hourly-IV DO CRPS over the primary leads.
- **Primary decision:** median per-site CRPS skill of the frozen candidate vs the best reference at each primary
  lead, HUC4 bootstrap B = 2,000. The lower bound must be > 0 at **every** lead in {6, 12, 24, 48, 72} h, in each of 3
  seeds. There is one primary variable, so no Holm correction.
- **Metadata-only power check** (pre-declared, computed from the site list's series metadata; counts only):
  - Primary DO sites with ≥ 1 summer (Jun–Sep) in the design-B issue window: **195 sites, 60 HUC4, ≈ 712
    site-summers** (stream 78, lake 88, estuary 29). This counts DO series of any type; IV availability per site is
    confirmed from the integrity table (row counts only) before the freeze.
  - **Rule:** M1-H is confirmatory only if **≥ 10 HUC4** have scorable hourly-IV DO issue rows at +72 h in that
    count. Otherwise it is descriptive. Projection: about 60 HUC4, so likely powered for CRPS.
  - Hypoxia-hours > 0 rows are rare. The conditional cell is descriptive.
- **EW-H1h (H-6): descriptive by default.**
  - Definition: an hourly-native hypoxia onset day is the first day with **≥ 2 consecutive hourly-IV DO values
    < 3 mg/L**, with the EW-v2 H1 refractory and clean-pre-window rules (`ew_v2_protocol.md`) applied unchanged with
    this criterion. Probabilities come from the M1-H predictive trajectories, with no separate classifier.
  - **Disclosure (always):** EW-H1h is defined **after** the IV→daily rule failed (2026-11-10). It was recommended
    earlier (`docs/research/09` §0 item 4, 2026-10-22), but it is adopted here after that failure. This is a
    forking-paths risk and is stated wherever EW-H1h is reported.
  - **Confirmatory only if** the metadata-only power check passes before the freeze. Projected onsets = (H1h onset
    rate per observed DO site-year on train-role sites × train block, from the hourly tier: training data, counts
    only) × (primary new-site hourly-IV DO issue site-years, from metadata and integrity counts) × 0.5, and must be
    ≥ 50, with ≥ 2 basin groups (the EW-v2 power rule). At scoring it must also meet the EW-v2 rule of ≥ 50 evaluable
    onsets. Otherwise descriptive.
  - Plan §7 projects ≈ 46–50 onsets before the ≥ 2 h rule, so failure is likely.

## 4. M1-SP — StreamPULSE new-source block (H-5): protocol before download

- **Practical reason:** see §1. Secondary track.
- **Source:** StreamPULSE bulk sensor data (https://data.streampulse.org/), ODC-By 1.0, about 0.5 GB. Licence terms
  are followed: attribution; derivative-work notice to contributors; the 1-year embargo on new sites is respected.
- **Eligible sites** (decided from portal *metadata* only, before any sensor value is downloaded):
  - ≥ 1 year of DO or water temperature within WY2008–2025;
  - not within 1 km of any USGS split site, reserved site, new-site block site or NEON site;
  - not a RiverPulse (downstream D8) test-role site;
  - not in any other `docs/preregistration/*.json` (conservative token match, as for the new-site list).
- **Exposure audit before download.** `processed/labels_streampulse` (metabolism labels, 133 sites) is already on
  disk. Sites whose labels any workstream has read for training or evaluation form a **hard tier**, excluded from
  the M1-SP primary and reported as a sensitivity. The audit uses file and registry metadata only.
- **Lock-on-download** (as `sonde_newsites`):
  - the downloader writes only into `processed/sonde_streampulse/` under an enforcing guard;
  - integrity checks are counts only, with no target statistics;
  - then manifest + content pin, and files 000 / directories 0300;
  - the scorer-only unlock requires the lead's freeze and an open claim.

  StreamPULSE metabolism labels at M1-SP sites are **never** used for PhiDA-Lite latent supervision (only labels at
  USGS train-role split sites may be).
- **Block, targets, decision:**
  - all eligible sites are test-only, and none is used in training;
  - targets: daily DO mean/min and T mean at h ∈ {1, 2, 3, 5, 7}; design-B reference window;
  - clusters: HUC4 for US sites, else the Level-II ecoregion;
  - a variable is confirmatory only if it has **≥ 10 clusters** with scorable issue rows at h = 7 (metadata
    projection computed from portal metadata before download; counts only); otherwise descriptive;
  - decision structure as the M1-N primary (median per-site CRPS skill vs the best reference at every h, 3 seeds,
    Holm over the confirmatory variables);
  - models: the M1-N frozen candidate and references, unchanged (transfer test);
  - its own hardened single look and claim, separate from M1-N.
- **Not downloaded before the stamp.**

## 5. Compute, disk and execution rules
- **GPUs 0–9** through `sentinel/core/gpu.py`: idle GPUs only (`--min-free-mb` 10000, so a GPU used by another user
  is never taken); ≤ 2 h per process, with checkpoint/resume; parallel where GPUs are free.
- **CPU** ≤ 6 threads per job; CPU jobs are sequential within M1.
- **Disk floor** 60 GB free on `/`, checked before each download batch. M1's new-data cap (10 GB) applies to kept
  data; GEFS is stream-extract-delete with < 1 GB kept.
- **Registry:** a row for every run (failed ones included); `prereg_snapshot` rows at each stamp.
- **Order after the stamp:**
  1. step 1 (§2f);
  2. references (R-NH, GBM-NWP, PF-GBM, Chronos-2-cov ZS);
  3. SONDE-HF CV;
  4. M1-H models;
  5. gated families if allowed;
  6. finals, E2 / E2-B;
  7. numeric-forecast report on CV;
  8. freeze candidate → rehearsal → pre-test review → lead freeze → single look.

## 6. Rulings requested
- **R1** E2-B member list: R-NH in place of the unapproved "LSTM-w-NWP" (§2b).
- **R2** Upstream-input date rule (§2g), or no upstream inputs at all.
- **R3** GEFS transfer: c00, four variables, lead days 1–7, ≈ 0.6 TB streamed, < 1 GB kept (§2c). Default c00
  only; members later by ruling.
- **R4** M1-H in the same single look as M1-N (one claim), with its own decision outside the M1-N Holm family (§3).
- **R5** EW-H1h stays descriptive unless the §3 projection is ≥ 50 onsets (expected to fail).

## 7. Deviations log
| Date | Change | Reason |
|---|---|---|
| 2026-11-10 | v0.1 draft written before any result on any arm or track in this file; no NWP-based M1-N validation number exists | lead dispatch, architecture round v2 |
