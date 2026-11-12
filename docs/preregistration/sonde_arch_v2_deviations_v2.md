# M1 SENTINEL-Sonde — architecture round v2: consolidated deviations from the stamped amendment (v2)

**Written by:** the M1 engineer (AI-assisted), generated 2026-11-11 18:58 UTC. **Status:** NOT stamped. It takes
effect only when the lead stamps it with a registry row (`M1-sonde` / `sonde/prereg_snapshot`, written by the lead;
§7). This file is read-only (0444) from the moment it is written and is not edited afterwards; a later change is a new
dated deviation file that the lead stamps before the affected run (ruling O `DR:73`).

**What it is.** The one deviations file of this round (ruling A `DR:45`). It supersedes
`docs/preregistration/sonde_arch_v2_deviations.md` and `docs/preregistration/sonde_arch_v2_deviations_part2.md`, which
stay on disk unchanged as the reviewed record and are never stamped. It records every point where the code of §6
differs from, or adds detail to, the stamped amendment and the lead rulings R1–R5 / L1–L5, with the same rule for the
candidate and for every reference.

**No result exists.** No step-1 number, no cross-validation number and no validation number of any architecture-v2 arm
or reference has been produced (§1). The test blocks and the new-site block are not read by any code of §6
(`prep.py:96, 102`).

**Public record.** Per lead clarification L5 (`LR:73`), reports describe the record as "pre-specified and frozen in a
public GitHub commit before testing (not registered on OSF)".


## 0. Citation convention and files referred to

A citation `X:n` means line n of file X as it is on disk at the sha256 below. Every citation and every quoted sentence
in this file was looked up in the file by a script when this file was generated; a pattern that was not found would
have stopped the generation. Short names: `AM` = `docs/preregistration/sonde_arch_v2_amendment.md`; `LR` =
`docs/preregistration/sonde_arch_v2_lead_rulings.md`; `DR` = `docs/preregistration/sonde_arch_v2_dev_lead_rulings.md`;
`TR` = `docs/preregistration/sonde_arch_v2_gbm_timing_rule.md`; `GB` =
`docs/preregistration/sonde_arch_v2_gbm_lead_ruling_20261110.md`; `PN` = `docs/preregistration/m1n_protocol.md`. A
bare file name (`prep.py`, `sonde_v2_cv.py`, `test_sonde_v2.py`) is a file of the code list in §6
(`sentinel/models/sonde_v2/`, `scripts/train/`, `tests/`). `sonde/x.py` = `sentinel/models/sonde/x.py`, a file of the
original track that this round does not edit. Lead rulings are cited as "ruling X `DR:n`"; only rulings that are on
disk are cited.

| File | sha256 | mode |
|---|---|---|
| `docs/preregistration/sonde_arch_v2_amendment.md` | `6c7b169d60ad74aa941965c67641d9a9e72faea804db6ac5b5b7b77d6bd60226` | 444 |
| `docs/preregistration/sonde_arch_v2_lead_rulings.md` | `7cbf398793a6345b8c0e2aef349f82db821ef821c3c58dd0a5542bddee8aac55` | 444 |
| `docs/preregistration/sonde_arch_v2_dev_lead_rulings.md` | `9bd515416928d31194cb7bcc0c46e69c1202932764a6f3ada5dbd8cb58e547fe` | 444 |
| `docs/preregistration/sonde_arch_v2_gbm_timing_rule.md` | `9d627602f3868c4b8547bea128862c23effd9a56d7019c47459e24ef253ea658` | 444 |
| `docs/preregistration/sonde_arch_v2_gbm_lead_ruling_20261110.md` | `3028fc1bd0509152e7428ad8a0ccbe27dece086027da9713bc9dae20bbf32c5b` | 444 |
| `docs/preregistration/sonde_arch_v2_gbm_lead_ruling_20261110.md` | `3028fc1bd0509152e7428ad8a0ccbe27dece086027da9713bc9dae20bbf32c5b` | 444 |
| `docs/preregistration/m1n_protocol.md` | `e71770e5de850dbb4a4f80833aa6a91c9c52de17cbae78764c9af5afebba2b51` | 664 |
| `docs/preregistration/sonde_arch_v2_deviations.md` | `031ddddc8bb8fecb4788ae639bc358f9c28ad042b08cb4ea3d8fd9b1ab2fcd92` | 664 |
| `docs/preregistration/sonde_arch_v2_deviations_part2.md` | `b17f3ec67af92809f2c2c2b31dca210275d7712c45d9a7a3d8e9930e788a63f4` | 664 |
| `docs/preregistration/reviews/arch_v2_20261110/M1dev_review.json` | `ef0f26bd755a73727c31af30e1b17133381fd60ec874f9a69f7b09446af14c7d` | 664 |
| `docs/preregistration/reviews/arch_v2_20261110/M1p2_review_integrity.json` | `3735bb1566ef7193ce423ed878c220c8599bc7e3e273ec17fe50336ca0b2b7c3` | 664 |
| `docs/preregistration/reviews/arch_v2_20261110/M1p2_review_code.json` | `c7ff02e5a9362bc0b68b4fcbb70e75679763534a17f8e189311ba5de3f7fdba1` | 664 |
| `docs/preregistration/reviews/arch_v2_20261110/M1p2_verify.json` | `9ba443d93358cd7c7f69ce8deffa84d3faf6fc60cf89f6081a44171d137fb14a` | 664 |
| `sentinel/models/sonde/gbm.py` | `3ded64d5623e8d28d71b120657cd1fee707f96fae8b0449c1a5f4de195487df5` | 664 |
| `sentinel/models/sonde/common.py` | `38524bf14e43f0e8b4679389499269948eb34cf459c1b92d140ba9dd1a2a6281` | 444 |

## 1. State when this file was generated

- Registry rows of this round by experiment: `sonde_arch_v2/gefs_pilot` 7. Rows of `sonde_arch_v2/step1`: 0; of
  `sonde_arch_v2/cv`: 0; of `sonde_arch_v2/declaration`: 0.

- Run directories under `sentinel_runs/sonde/`: `v2_step1` does not exist; `v2_cv` does not exist; `v2_smoke` holds
  train-block smoke checkpoints (`rnh/`, `sonde_hf/`), the counts records of §4 and the timing records of §5. No
  forecast file of a val-block row exists.

- The drivers cannot produce a prediction or a registry row before the stamp: every phase of both drivers calls the
  stamp guard (V20).

- GEFS point extraction on disk: reforecast months 2007-10 … 2019-12 complete; operational months 2020-09 … 2023-09
  complete.


## 2. Deviations and recorded details

Each entry gives the stamped text, what the code of §6 does, the reason, and the effect on the candidate (SONDE-HF)
and on the references (R-NH, GBM-NWP, PF-GBM, Chronos-2-cov ZS). "Same for every arm" means the same code path or the
same array. Counts are in §4.

### V01. One NWP product for every arm: GEFSv12 control member; no GFS product, no ensemble members

- **Stamped text.** "Products, combined in SONDE-HF / R-NH by a masked mean over the available products, with no
  zero-fill" (`AM:88`); the three products P1 (Open-Meteo GFS), P2 (GEFSv12 reforecast, "c00 + p01–p04"), P3 (GEFSv12
  operational) at `AM:89–91`; "GBM-type models get per-product features and availability flags." (`AM:93`) "GFS-only
  mode (P1 only) and past-weather mode are reported as separate cells." (`AM:135–136`) Lead ruling R3: "GEFSv12,
  control member c00 only, variables tmax / tmin / apcp / dswrf, lead days 1–7" (`LR:34–35`) and "Members p01–p04 need
  a new ruling." (`LR:46`)
- **What the code does.** Exactly one declared product, `gefs_c00` (`declared.py:20`; `products.py:34`). The
  reforecast (issue days to 2019-12-31) and the operational archive (issue days from 2020-09-23) are one tensor on the
  date axis: reforecast where it is finite, else operational (`products.py:72`); required month files `prep.py:49–51`.
  Fields: `nwp.py:32`; forecast mean temperature = (Tmax + Tmin) / 2 because R3 extracts no mean (`prep.py:182–183`),
  and the observed mean is formed the same way (`prep.py:134`). A list of products or a second product is refused by
  the network tensors (`data_v2.py:100–102`), the model builder (`train_v2.py:58–59`) and the Chronos reference
  (`chronos_cov.py:27–28`); tests `test_sonde_v2.py:788, 1435`. No code of §6 reads the Open-Meteo GFS table. The
  network keeps its masked-mean module with one product; where no product is available the embedding is a learned
  vector (`nets.py:42, 53`). The GBM arms get the features and availability flags of the one product (V07).
- **Reason.** ruling §2 G `DR:92`: GEFS-only for every arm, because the GFS train/val table cannot exist before about
  11-28 and its train-block overlap is 191 days with 2-m temperature only. Members: ruling R3 (control member only).
- **Effect on candidate and references.** Same for every arm. No model learns from or is evaluated with GFS, so this
  round has no "GFS-only mode" cell; the cells are forecast mode with GEFS and past-weather mode (V05). GBM-NWP and
  PF-GBM lose the stamped "GEFS member mean and spread" features (V07): information removed from a reference, to be
  disclosed wherever GBM-NWP or PF-GBM is reported (ruling M `DR:69`).


### V02. Bias-correction map: one map for GEFSv12, fitted on reforecast pairs only; full record of the map

- **Stamped text.** The bias-correction rule at `AM:114–121`, in particular "For each product and lead day, a per-site
  linear map from forecast to gridMET is fitted on train-block overlap only" (`AM:114–115`), "GEFS reforecast:
  WY2008–2021-09-30 ∩ product availability" (`AM:116`) and "The map is per month-of-year where the site has ≥ 60
  overlap days in that month, else a pooled HUC2 map." (`AM:119`)
- **What the code does.** One map for the GEFSv12 product, fitted on reforecast-minus-gridMET pairs only, with valid
  day ≤ 2021-09-30 (`products.py:78`; `nwp.py:147`); it is applied unchanged to the operational archive
  (`products.py:79`). The map is per site × lead day (1–7) × field × month of the VALID day (`nwp.py:106`). Order of
  sources per cell (`nwp.py:100`): (source 1) the site-month map where the site has ≥ 60 finite pairs in that month
  (`nwp.py:166–168`, `nwp.py:134`); else (source 3) the pooled HUC2-month map over all loaded sites of the HUC2
  (`nwp.py:169–171`), which exists when the pool has ≥ 3 finite pairs (`nwp.py:35, 119`); else (source 2) the site's
  all-months map with ≥ 60 pairs (`nwp.py:172–174`); else (source 0) identity, the forecast passes through
  uncorrected. Temperature and shortwave: linear map a + b·x, slope clipped to [0.5, 1.5] (`nwp.py:36, 129`), shift
  only when the forecast has no variance (`nwp.py:126–127`). Precipitation: ratio of means b = mean(obs) /
  mean(forecast), NOT clipped, identity when the mean forecast is ≤ 1e-6 mm (`nwp.py:122–124`; `nwp.py:150`);
  corrected precipitation is floored at 0 (`nwp.py:113`). The pooled maps are fitted over the sites a build loads:
  train-role sites in step 1, train- and val-role split sites in CV (weather pairs only, valid day ≤ 2021-09-30). So
  the site maps are identical in step 1 and CV, and the pooled maps are not (the module text at `products.py:12–13`
  speaks of the site maps only). The new-site rule of the stamped text (reference-window site map, else the pooled map
  of train-role sites) is not implemented in the code of §6; no new-site data is read in this round.
- **Reason.** "For each product": the reforecast and the operational archive are the same model version (GEFSv12) and
  are one product here (V01); the operational train-block overlap (2020-09-23 … 2021-09-30) cannot give 60 pairs in a
  site-month, so the map comes from the reforecast alone. The detail beyond the stamped sentence (3-pair minimum,
  clip, precipitation ratio, identity) is recorded in full as ruled (rulings I `DR:61`, item 7 `DR:104`).
- **Effect on candidate and references.** The same corrected array is the input of every arm (its sha256 is in each
  parity record, V20). No water-quality value and no val-block weather pair enters the map. A weather-only train-block
  diagnostic of the corrected forecasts is in §4.3.
- **Counts.** §4.2: sources 0–3 per field, the precipitation-ratio distribution.


### V03. Stage-1 error model and the perfect-prog sample

- **Stamped text.** "stage 1 is perfect-prog with a lead- and season-dependent error model (GEFS
  reforecast-minus-gridMET residuals at train-role sites, WY2008–2019)" (`AM:122–123`); step 1: "Inputs: perfect-prog
  gridMET for the target days plus the stage-1 NWP error model." (`AM:167`)
- **What the code does.** Residuals = BIAS-CORRECTED reforecast minus gridMET at train-role sites (`products.py:80,
  82`), issue days with issue + 7 ≤ 2019-09-30 (`products.py:31, 81`). Per lead × field × season (DJF, MAM, JJA, SON;
  `nwp.py:205`) a mean and a standard deviation, from that season's residuals where there are ≥ 30 (`nwp.py:245`),
  else from all seasons (`nwp.py:248`). The noise is AR(1) across leads with one correlation per field, clipped to [0,
  0.99] (`nwp.py:257`; `nwp.py:215–216`), independent across sites, issue days and fields; precipitation is floored at
  0 (`nwp.py:222`). The perfect-prog input = observed gridMET of the target days, leads 1–7 (`products.py:67`), plus
  ONE noise sample per build, seed 20261110 (`products.py:35, 83`).
- **Reason.** The stamped text fixes the fit set and "lead- and season-dependent"; the noise structure (AR(1) across
  leads, independence across sites and days) and the use of the corrected forecast are details it does not give. The
  window ends 2019-09-30 as stamped (ruling N `DR:71`).
- **Effect on candidate and references.** One array for every arm of a build; its sha256 is written to each parity
  record, and the step-1 evaluation refuses to run unless the SONDE-HF phase and the GBM phase carry the same hash
  (`sonde_v2_step1.py:292–293`; `test_sonde_v2.py:1369`). The sample of step 1 and the sample of CV are different
  arrays, because the date axes differ (V22).
- **Counts.** §4.2 (fit sites, issue days, standard deviations, correlations, sha256).


### V04. Network NWP input: zero for a missing field inside an available lead; leads 8–14 have no product

- **Stamped text.** "Products, combined in SONDE-HF / R-NH by a masked mean over the available products, with no
  zero-fill" (`AM:88`); SONDE-HF row: "14-step forecast LSTM over a masked-mean NWP embedding" (`AM:52`).
- **What the code does.** A lead is available when its mean temperature is finite (`data_v2.py:180`). Inside an
  available lead, a missing other field enters as 0 after standardisation (the train mean) with a per-field
  availability flag (`data_v2.py:179`). An unavailable lead is never zero-filled: the embedding is the learned
  no-product vector (V01). The product has leads 1–7 (R3), so leads 8–14 of the 14-step forecast LSTM always get the
  no-product vector plus the deterministic inputs of V17. Standardisation: per field from the fit sites × train block
  of the stage-1 input, precipitation as log1p (`data_v2.py:108–113`); the same statistics are used for the stage-2
  and evaluation inputs.
- **Reason.** A lead with temperature but without precipitation or shortwave would otherwise be dropped whole. The
  field flag tells the network which value is a fill.
- **Effect on candidate and references.** Identical batch code for SONDE-HF and R-NH (`data_v2.py:177`); the GBM arms
  and Chronos-2-cov get NaN, not 0, for a missing value (V07, V19).
- **Counts.** §4.3: in the step-1 build no GEFS cell with temperature lacks precipitation or shortwave.


### V05. GEFSv11 gap and missing forecasts: past-weather mode, no substitution; hard errors for missing data

- **Stamped text.** "Issues whose lead days fall in that gap use the other available products, else past-weather mode,
  and are reported as such." (`AM:95–96`) "A missing product is reported, never imputed with observed weather at test
  time." (`AM:136`)
- **What the code does.** No operational file dated before 2020-09-23 is read (`products.py:32`; `prep.py:51`). With
  one product (V01) there is no "other available product": issue days 2020-01-01 … 2020-09-22, and any other issue day
  without a forecast, are in past-weather mode for every arm (networks: no-product vector; GBM arms: NaN features and
  availability 0; Chronos-2-cov: NaN future covariates, `chronos_cov.py:47`). Observed weather is never substituted
  for a missing archived forecast in stage 2, in a GBM fit of CV or in any out-of-fold forecast
  (`sonde_v2_cv.py:183–184`; `sonde_v2_cv.py:303`; `sonde_v2_cv.py:369`). Missing data are hard errors: a missing
  required month file (`prep.py:158–160`; `test_sonde_v2.py:422`); a share of site × issue-day cells with a finite
  lead-1 temperature below 0.90 in any required window (`products.py:33`, windows `products.py:55–58`, error
  `products.py:74–76`); the same threshold on the held-out rows of each arm in CV (`sonde_v2_cv.py:143`).
- **Reason.** rulings §2 D5 `DR:95`, G `DR:57`. The threshold 0.90 is declared here, as ruling G asks.
- **Effect on candidate and references.** Same for every arm. In CV the GBM arms are fitted on the archived forecasts
  over the whole train block, so their fit rows issued 2020-01-01 … 2020-09-22 are past-weather rows; the networks see
  the same days without a product in stage 2. Stage 1 of the networks and both step-1 arms use the perfect-prog sample
  (V03), which exists for every day.
- **Counts.** §4.3.


### V06. Step-1 information-use test: details not fixed by the stamped text

- **Stamped text.** The step-1 rule at `AM:164–173`, in particular "Contrast: smoke SONDE-HF (config C1, 1 seed) vs
  GBM-NWP (default config)." (`AM:168`) and the statistic on "the held-out fold × train-block issue dates
  (WY2016–2021)." (`AM:170`)
- **What the code does.** Sites: fold 0 of the 5-fold basin-group split of the train-role sites, seed 20261110, is
  held out (`sonde_v2_step1.py:80`); 15 % of the in-fold basin groups (seed 20261110 + 0) are the early-stopping sites
  and are not fit sites (`sonde_v2_step1.py:83–84`). Data: train-role sites × train block only
  (`sonde_v2_step1.py:94`). Comparator: GBM-NWP configuration Q19 ("default config" is read as the 19-level
  configuration; `declared.py:58`); no Q or LS run in step 1. Both arms are fitted and evaluated on the same
  perfect-prog sample (`sonde_v2_step1.py:154, 223, 227`); SONDE-HF is trained by stage 1 only (no stage 2 in step 1).
  Evaluation rows: held-out sites × train-block issue days ≥ 2015-10-01 with the target day inside the train block
  (`declared.py:56`; `sonde_v2_step1.py:109`) that pass the issue rule of V09 (`sonde_v2_step1.py:114`). Statistic:
  CRPS of a row = 2 × mean pinball loss over the 19 levels (`sonde_v2_step1.py:258–262`); per site with ≥ 30 scored
  rows the skill 1 − mean CRPS(SONDE-HF) / mean CRPS(GBM-NWP); the median over sites per horizon; the mean over h = 3,
  5, 7 (`sonde_v2_step1.py:265–280`). A row without a finite forecast from either arm is dropped for both and counted
  (`sonde_v2_step1.py:271`). Rule as stamped, threshold 0.01 (`sonde_v2_step1.py:314–316`). The SONDE-HF phase runs in
  jobs of ≤ 2 h and resumes from its checkpoint (`sonde_v2_step1.py:182`).
- **Reason.** ruling B `DR:47` (Q19 is the step-1 comparator; a later Q or LS run in step 1 would make the gate use
  the best-scoring GBM configuration). The other details are needed to compute the statistic and are fixed here before
  any step-1 number.
- **Effect on candidate and references.** The comparator is the strongest stamped form of GBM-NWP (19 fitted levels).
  Both arms use identical fit sites, early-stopping sites, evaluation rows and inputs; the evaluation checks equal
  rows and equal input hashes (`sonde_v2_step1.py:294–297`).


### V07. GBM-NWP features

- **Stamped text.** "These are forecast Tair mean/max/min (P and SW where present), cumulative and exponentially
  weighted forecast degree-days over leads 1..h, GEFS member mean and spread, and product-availability flags."
  (`AM:63–65`) "Quantile LightGBM (19 levels), not homoscedastic." (`AM:65`)
- **What the code does.** For horizon h the features use leads 1..min(h, 7) (`nwp.py:283`): per-lead forecast Tmean,
  Tmax, Tmin (`nwp.py:287`); mean Tmean; cumulative degree-days above 0 °C and exponentially weighted degree-days with
  weight 0.7^(min(h, 7) − lead) (`nwp.py:296–299`; `gbm_nwp.py:69–70`); precipitation sum and maximum and mean
  shortwave over those leads, as summaries, not per lead (`nwp.py:300–301, 303`); the Tmean of the target day, which
  for h > 7 is the lead-7 Tmean (`nwp.py:304`); availability shares of temperature and precipitation
  (`nwp.py:305–306`). When the forecast is missing every summary is NaN, never 0 (`nwp.py:298`;
  `test_sonde_v2.py:79`). No member mean or spread. Deterministic target-day clear-sky shortwave and day length are
  added for both GBM arms (V17). The context features of the frozen builder cover every cube variable except the one
  no model reads (`gbm_nwp.py:133`).
- **Reason.** rulings M `DR:69`, item 6 `DR:103`. Members do not exist under R3. The base temperature, the decay, the
  summaries of P and SW and the reading of "leads 1..h" are details the stamped sentence does not fix.
- **Effect on candidate and references.** GBM-NWP and PF-GBM share these features. The missing member mean and spread
  remove information from two references (not from the candidate, which never had member inputs) and are disclosed
  wherever GBM-NWP or PF-GBM is reported.


### V08. PF-GBM physics features: fixed constants

- **Stamped text.** "GBM-NWP plus uncalibrated physics features: a regional-default air2stream water-temperature
  forecast, DO_sat(T_forecast) (Benson–Krause), Raymond K600 from Q and slope, and a two-end-member SC mixing
  estimate." (`AM:66–67`)
- **What the code does.** Four features, none fitted or tuned on any block (`physics_features.py:24–25`): (1) water
  temperature of the target day (for h > 7, of lead 7) from the last observed water temperature relaxed towards max(A0
  + A1·Tair(forecast), 0) with rate κ per day, A0 = 3.0 °C, A1 = 0.8, κ = 0.25 d⁻¹ (`physics_features.py:29–39`;
  `gbm_nwp.py:159`); (2) Benson–Krause DO saturation at that temperature and the site elevation (`gbm_nwp.py:162`);
  (3) the Raymond K600 prior from the LAST OBSERVED discharge and the NHDPlus channel slope, NaN where either is
  missing (`physics_features.py:50–58`); (4) SC of the last observation diluted by runoff = 0.2 × forecast
  precipitation sum over leads 1..min(h, 7) × drainage area, as a mean flow over the h days, runoff SC taken as 0
  (`physics_features.py:61–67`).
- **Reason.** The stamped text names the four features and "uncalibrated"; the constants and the exact formulas are
  fixed here before any result. Discharge is not forecast, so K600 and the SC end member use the last observation.
- **Effect on candidate and references.** PF-GBM only. The physics features are additional columns; PF-GBM has every
  GBM-NWP feature. `test_sonde_v2.py:108`.


### V09. Reference-window climatology and the issue rule, for every model and every use

- **Stamped text.** "Reference window = the first 730 calendar days from the site's first observation of the variable
  (≥ 365 observed days required; unified with EW-v3, §2b)." (`PN:39`) Amendment parity list: "2-year reference
  climatologies" (`AM:81`).
- **What the code does.** For every site-variable the reference window is the 730 calendar days from its first finite
  value on or after the cube start 2007-10-01 (`prep.py:45–47`). With < 365 observed days in the window the
  site-variable has no climatology and no row (`prep.py:234–236`). A window that is not wholly inside the dates a
  build loads gives no climatology in that build (`prep.py:229–230`). The climatology (day-of-year mean and standard
  deviation, ±15-day circular smoothing, window mean/variance where < 2 years support a day of year) replaces the
  training-years climatology of the original track for anomalies, inputs and naive references of this round
  (`prep.py:263`). Issue rule: a row of variable v at site s is used for training, early stopping, calibration,
  ensemble weights or scoring only if its issue day ≥ window end + 42 days (`prep.py:63–66`). It is applied to the
  common scored rows (`SondeDataV2.rows`), to the GBM fit and early-stopping rows (`gbm_nwp.py:203`), to the network
  labels and training rows (`data_v2.py:134, 155`) and to the step-1 evaluation rows (V06). The climatology of a
  hindcast INPUT variable is not blanked before its own window end: a row of variable v issued after v's window end
  may have, among its inputs, anomalies of another variable w whose reference window ends later, i.e. w's climatology
  then contains days after the issue day (train-block or loaded-range days of the same site only). Tests:
  `test_sonde_v2.py:250, 362`.
- **Reason.** rulings F `DR:55`, §2 F `DR:88`. The original-track climatology used all training years, which the
  new-site design cannot have.
- **Effect on candidate and references.** Same build and same mask for the candidate, every reference and the naive
  references, in step 1 and in CV. The set of site-variables with a climatology differs between step 1 (dates to
  2021-09-30) and CV (dates to 2023-09-30): a window that ends after 2021-09-30 exists only in CV, where its rows are
  val-block rows (out-of-fold scoring only). Train-block rows are the same in both builds.
- **Counts.** §4.1 (site-variables and rows kept or lost, per variable, role and block; rows issued inside the window
  of another input variable).


### V10. Training rows: one rule for networks and tree models; the frozen GBM fit window is not used

- **Stamped text.** Lead clarification L3: "In each fold, network weights and tree models are fitted on the in-fold
  sites × train block only, with the 42-day embargo at the block edge." (`LR:66–67`) Amendment parity list: "training
  sites and blocks" (`AM:79`).
- **What the code does.** Training rows of every learned arm = fit sites × issue days of the train block whose target
  day is ≤ 2021-09-30 by the frozen issue-date rule (`declared.py:83`): GBM arms, target day issue + h
  (`gbm_nwp.py:18–21`; `sonde/gbm.py:106`); networks, issue + 14 for all leads (`data_v2.py:141`). The fit window
  `fit_end = 2019-09-30` of the original-track GBM (`sonde/gbm.py:49, 63`) is not applied by the v2 subclass, and no
  calibration slice is held back inside the train block. No embargo is added inside the train block; the 42-day
  embargo of the frozen issue-date rule is at the block edges. `sonde/gbm.py` is not edited (sha256 in §0). The fit
  window (first and last issue day, last target day) of every run is written to the parity log (`sonde_v2_cv.py:149`;
  `sonde_v2_step1.py:233`).
- **Reason.** rulings D `DR:51`, §2 D `DR:89`. In CV the GBM arms are fitted on the archived forecasts of the whole
  train block, operational GEFS of 2020-09-23 … 2021-09-30 included, as the networks are in stage 2.
- **Effect on candidate and references.** The references are fitted on the same years as the candidate (two more water
  years than the original-track GBM). `test_sonde_v2.py:987`.


### V11. Early-stopping rows: 15 % of the in-fold basin groups; in CV these include val-role sites

- **Stamped text.** "Early stopping on an inner 15 % of train-role basin groups, never on the val block (fixes W6)."
  (`AM:192`)
- **What the code does.** Early-stopping sites = 15 % of the basin groups of the in-fold sites, drawn with seed
  20261110 + fold (`declared.py:19`; CV `sonde_v2_cv.py:123–126`; step 1 V06). They are removed from the fit sites.
  Their rows follow the training-row rule of V10 (train block only). In CV the folds are over train- and val-role
  split sites (`sonde_v2_cv.py:100–102`), as stamped for the selection data, so the early-stopping groups contain
  val-role sites; only their train-block rows are used. The same sites serve every arm of a fold
  (`sonde_v2_cv.py:115`). GBM arms: early-stopping rows are drawn after the finite-target mask, at most 400,000 per
  variable and horizon, by the seeded site-stratified rule (`gbm_nwp.py:261–262`), and the effective counts go to the
  parity log. Networks: at most 30,000 early-stopping samples, drawn once per run (`train_v2.py:43, 137`).
- **Reason.** "Train-role basin groups" cannot be taken literally in CV, where the in-fold sites are train- and
  val-role sites; the rule keeps the stamped intent (site-wise inner hold-out, never a val-block row). rulings D
  `DR:51`, item 3 `DR:100`.
- **Effect on candidate and references.** Same sites and same rule for the candidate and every learned reference; no
  time slice is used for early stopping by any arm. `test_sonde_v2.py:1305`.


### V12. GBM-NWP and PF-GBM: configurations, LightGBM settings, fit-row rule

- **Stamped text.** "Quantile LightGBM (19 levels), not homoscedastic." (`AM:65`) "R-NH, GBM-NWP and PF-GBM may each
  use up to 5 CV configurations, the same as the best-tuned SENTINEL family." (`AM:84–85`) GBM-NWP is "the current
  GBM-w plus per-lead NWP features." (`AM:63`)
- **What the code does.** Configurations of both arms (`declared.py:41–46`): **Q19** = 19 quantile boosters, one per
  reported level, the mandatory configuration of each arm and the step-1 comparator (`declared.py:49`), the 19
  predictions of a row sorted (`gbm_nwp.py:311`); **Q** = 7 fitted levels (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
  interpolated to 19 (`gbm_nwp.py:58`); **LS** = an L1 location booster and an L2 booster for the log absolute scaled
  residual, with residual quantiles taken from the early-stopping rows (`gbm_nwp.py:323`). Q and LS are additional
  configurations inside the limit of 5 (3 are declared per arm). LightGBM recipe of the original track, unchanged:
  learning rate 0.05, ≤ 2000 rounds, early stopping 100 rounds, 63 leaves, ≥ 200 rows per leaf, feature fraction 0.8,
  bagging 0.8 every round, L2 1.0 (`sonde/gbm.py:39–47`; the same values written out in `declared.py:70–72` and
  asserted against the frozen file by `test_sonde_v2.py:675`), as GBM ruling G-5 `GB:75` rules; "Proposal E(5)
  (learning rate 0.10, 1000 rounds) is not adopted." (`GB:78–79`) 6 threads, max_bin 255, seed 0, early-stopping rows
  ≤ 400,000 (`declared.py:74`). One model per scored variable and horizon; a variable-horizon with < 1000 fit rows or
  < 200 early-stopping rows is skipped and recorded (`gbm_nwp.py:59–60, 273`). Each fit logs row counts, the fit
  window, the number of rounds and, in the quantile modes, the share of early-stopping labels (train-block rows)
  between the lowest and the highest level (`gbm_nwp.py:312`); no value of a val-block row. Fit rows: every labelled
  row of V10, with no cap (the seeded site-stratified subsample `gbm_nwp.py:209`, `test_sonde_v2.py:972`, is used for
  the early-stopping rows only). The timing rule `TR` §2–§4 would have set a cap from the benchmark (the first cap of
  the ladder full / 1,500,000 / 1,000,000 / 500,000 / 250,000 whose unit time is ≤ 2 h for PF-GBM on the variable and
  horizon with most labelled fit rows, used by BOTH arms); §4 of that rule is not applied (§5.5). The cap ladder is
  not used: GBM ruling G-2 `GB:52` rules full rows for every configuration of both arms, in step 1 and in CV — "no
  driver applies a row cap to fit rows" (`GB:54`) (`max_fit_rows` is `None` for Q19, Q and LS, `declared.py:75`; the
  fit path refuses any cap, `gbm_nwp.py:226`; a test recomputes the masked rows of a unit and compares them with the
  fitted rows, `test_sonde_v2.py:1150`). The ladder stays in `GBM_SETTINGS` because the timing script reads it
  (`sonde_v2_gbm_timing.py:117`); no driver and no model code reads it, which a test asserts (`test_sonde_v2.py:701`).
  Device, resume and the GPU-hour account are in V12b. Ruling reference in the code:
  `docs/preregistration/sonde_arch_v2_gbm_lead_ruling_20261110.md`, sha256
  `3028fc1bd0509152e7428ad8a0ccbe27dece086027da9713bc9dae20bbf32c5b` (§0; `declared.py:69`). No GBM arm can run before
  the ruling is in the code (`declared.py:135–137`; `test_sonde_v2.py:657`).
- **Reason.** rulings B `DR:47`, E `DR:53`, §2 E(1) `DR:90`, item 2 `DR:99` and GBM ruling G-2 `GB:52`, G-5 `GB:75`.
  The ≤ 2 h per job rule of the project is met by resuming, not by a cap (V12b); the timing benchmark and its records
  are in §5.
- **Effect on candidate and references.** Both GBM arms have identical settings and rows and neither is capped; PF-GBM
  differs only by the four physics columns. The sentence of the superseded `sonde_arch_v2_deviations.md` (row D2) that
  19 boosters bring "no gain in the CRPS approximation at the inner levels" is withdrawn: no evidence for it exists
  (ruling B `DR:47`).
- **Counts.** §5 (labelled fit rows per variable, timings, projected hours).


### V12b. GBM boosters: one file per booster, resume at booster level, GPU device, GPU-hour account

- **Stamped text.** "≤ 2 h per process, with checkpoint/resume; parallel where GPUs are free." (`AM:300`) (amendment
  §5) and "CPU ≤ 6 threads per job; CPU jobs are sequential within M1." (`AM:301`) The stamped text fixes no device
  for the GBM references and no booster-level record.
- **What the code does.** Every booster of both GBM arms is fitted whole inside one process and written to its own
  file as soon as it is finished, with its key, stopping round, rounds actually run, wall time, process id, device and
  GPU index (`gbm_resume.py:364`, `gbm_resume.py:400`). The file is re-read at once and is the model that forecasts
  (the fitted booster in memory is used for nothing after the check that the two agree on the early-stopping rows,
  `gbm_resume.py:445`); the saved model holds the best iteration only, so every forecast is a plain `predict`
  (`gbm_nwp.py:344`). A booster whose record exists is never fitted again; a record whose key differs from the key of
  this process (fit rows, early-stopping rows, targets, objective, quantile level, seed, every LightGBM setting,
  LightGBM version) stops the run and is never silently refitted (`gbm_resume.py:73`). A process starts a new booster
  only while its elapsed wall time plus 1.5 × the longest booster time it has seen (40 min while it has seen none) is
  ≤ 115 min, and otherwise ends with exit code 75 for the same command to continue (`gbm_resume.py:122`;
  `sonde_v2_step1.py:248`, `sonde_v2_cv.py:290`). "Seen" counts the boosters this process fitted and the booster
  records it read. Each unit directory (arm, seed, fold, variable, horizon) holds a record naming every process that
  worked on it, its boosters and how it ended (`gbm_resume.py:374`), and a non-blocking lock keeps two processes off
  the same unit (`gbm_resume.py:347`). Because the booster files of one arm, seed and fold are named by what they are
  (`q05` … `q95`, `ls_loc`, `ls_scale`), configuration Q re-uses the seven boosters of Q19 when their keys are
  identical (`test_sonde_v2.py:1132`). Device: every GBM fit runs with LightGBM `device_type=gpu`, `gpu_use_dp=False`,
  max_bin 255, one GPU per process through the lock helper and ≤ 6 CPU threads (`gbm_nwp.py:140`,
  `gbm_resume.py:279`); before its first booster a process checks with a short synthetic fit, polling `nvidia-smi`
  during the fit, that its own process is on the locked GPU and on no other and that no other user's process is on it
  (`gbm_resume.py:220`, `gbm_resume.py:214`); the same check runs before every further booster, and a foreign process
  appearing on the GPU ends the process (exit 75) instead of sharing it. There is no CPU fallback anywhere in the GBM
  path. Every process writes its GPU wall time (from the lock to the end of the process), its boosters and its device
  to a registry row `sonde_arch_v2/gbm_gpu_hours` (`gbm_resume.py:329`), and the parity record of a run carries the
  ruling, the device settings, every process and the stopping round of every booster (`sonde_v2_step1.py:239`). Before
  any CV GBM process the projected CV GPU-hours, re-computed from the step-1 stopping rounds and the fit rows of every
  CV unit (`sonde_v2_cv.py:404`, `gbm_resume.py:511`), must be on disk and must, with the hours already spent, stay
  within 300 GPU-h, which only a lead-written registry row can raise (`gbm_resume.py:569`; `sonde_v2_cv.py:475`). Two
  readings the ruling leaves to the code, both only protecting the 2 h limit: a booster still running when its process
  is 119 min old is abandoned unsaved and fitted again from its start by the next process (a second abandonment of the
  same booster stops the work), and the forecast step after the fits starts only while the process is ≤ 105 min old
  (`gbm_resume.py:56–57`). A booster whose own wall time passes 2 h stops the work and is reported
  (`gbm_resume.py:69`).
- **Reason.** GBM ruling G-3 `GB:59`, G-6 `GB:80`, G-7 `GB:90`, which the lead wrote after the benchmark of §5; "each
  booster is fitted whole inside one process and saved to its own file as soon as it is finished, with its stopping
  round and wall time; a booster that has a file is never fitted again" (`GB:61–62`) and "no fit falls back to the CPU
  without a new lead ruling" (`GB:88`). The resume keeps the 2 h process limit of the stamped amendment without
  reducing the reference by a row cap.
- **Effect on candidate and references.** Identical for GBM-NWP and PF-GBM and for every configuration of both;
  nothing of this changes a fitted model: the rows, seeds, settings and the number of rounds of a resumed unit are
  those of an uninterrupted one, which a test shows by refitting an interrupted unit and comparing its forecasts
  (`test_sonde_v2.py:1085`). The networks, Chronos-2 and the baselines are untouched. GPU-hour rows are a compute
  account and carry no score.
- **Counts.** §5 (benchmark), §6 (`gbm_resume.py` is in the code list).


### V13. R-NH: configurations, own-target rows, stage 2, inputs

- **Stamped text.** "R-NH: Nearing et al. 2024-style hindcast–forecast LSTM, i.e. 365-d daily hindcast → state
  hand-off → 14-d forecast LSTM over NWP; CMAL head mapped to 19 quantiles; multi-site; one model per target; no
  hourly branch, no multi-task, no physics." (`AM:60–62`) "R-NH, GBM-NWP and PF-GBM may each use up to 5 CV
  configurations, the same as the best-tuned SENTINEL family." (`AM:84–85`)
- **What the code does.** Five configurations (`declared.py:38–40`): R1 CMAL / hidden 128; R2 CMAL / 256; R3
  monotone-quantile head / 128; R4 quantile / 256; R5 CMAL / 128 with learning rate 3e-4. R3 and R4 replace the
  stamped CMAL head by the candidate's quantile head and pinball loss (a head the stamped row does not name; it gives
  the reference the candidate's loss). One model per scored target (`sonde_v2_cv.py:172`). Training and early-stopping
  rows are the rows labelled for the model's OWN target (`train_v2.py:102–105`; `sonde_v2_cv.py:185–187`;
  `test_sonde_v2.py:849`), and the parity record's row count is that number. Inputs: the same daily hindcast channels
  (all modelled variables, weather, flags), statics, NWP slot and four deterministic future inputs as SONDE-HF
  (`nets.py:165–166`); the static embedding also enters the head (`nets.py:176`). No hourly branch, no anchor, no
  ordering constraint, one target. Training budget identical to SONDE-HF: one `TrainConfig` (`train_v2.py:37–43`) and
  one stage-2 rule: learning rate / 3, ≤ 10 epochs, patience 3, started from the stage-1 best weights
  (`declared.py:53`; `sonde_v2_cv.py:190–191`), with AdamW weight decay 1e-5 in both stages, set from the
  configuration and not from a saved optimiser state (`train_v2.py:130–131`; `test_sonde_v2.py:929`); the optimiser
  settings are written to each stage's record.
- **Reason.** rulings J `DR:63`, L `DR:67`, item 5 `DR:102`. The quantile-head configurations use the tuning allowance
  of the stamped parity rule.
- **Effect on candidate and references.** The reference has 5 configurations, as many as the candidate (C1–C5), the
  same row rule, budget and inputs. The deterministic future inputs and the static embedding in the head are additions
  to the stamped description; the reference gets them exactly as the candidate does.


### V14. Auxiliary targets (§2d): who writes what; DO % saturation; gage height

- **Stamped text.** §2d at `AM:140–150`: "DO max, DO diel amplitude and DO % saturation (deterministic transform;
  Benson–Krause with pressure from elevation);" (`AM:145`) "log discharge mean and gage height (auxiliary inputs and
  outputs, never a decision)." (`AM:147`) Parity list: "auxiliary targets, where the model can take them" (`AM:82`).
- **What the code does.** Declared outputs per arm (`declared.py:32–33`). SONDE-HF (multi-task configurations) is
  trained on 20 targets, every cube variable except observed DO % saturation (`prep.py:35–36`): the 8 scored
  variables, 9 auxiliary series of §2d, and pH mean, chlorophyll and fDOM, which are targets of the original track and
  are trained but not written. It writes the 8 scored variables and the auxiliary variables of `declared.py:27–28` (9
  from their own heads, `sonde_v2_cv.py:239`; DO % saturation by the transform below). "Log discharge mean" is
  `discharge_cfs_mean`, modelled in log space with the offset of the original track. C3 trains and writes the five
  primary variables only (`sonde_v2_cv.py:237`). R-NH, GBM-NWP, PF-GBM and Chronos-2-cov fit and write the 8 scored
  variables only (`sonde_v2_cv.py:305, 371`). DO % saturation is not a trained head and not an input of any model
  (`prep.py:38`; `sonde_v2_cv.py:77`): SONDE-HF's forecast is 100 × its DO-mean quantiles / DO_sat(its T-mean median,
  pressure at site elevation) (`prep.py:266–274`; `sonde_v2_cv.py:253–266`; `test_sonde_v2.py:387`), which ignores the
  uncertainty of T. The observed series for scoring is 100 × DO mean / DO_sat(T mean); DO amplitude = DO max − DO min;
  T 7DADM = 7-day mean of daily max with all 7 days present (`prep.py:296–301`). Gage height (`prep.py:34`, raw feet,
  no log transform) is a cube variable: an input of every model that reads the cube (network hindcast channels, GBM
  context features, Chronos-2 context) and an output of SONDE-HF only.
- **Reason.** rulings C `DR:49`, H `DR:59`, §2 H `DR:93`. The superseded statement that no gage-height series exists
  was false (review findings P2-01, F02); it is withdrawn and the target restored.
- **Effect on candidate and references.** The §2d auxiliary report of this round is SONDE-HF against persistence and
  climatology only; it supports NO claim against R-NH, GBM-NWP, PF-GBM or Chronos-2-cov, and the report header must
  say so. The auxiliary targets never enter the §4 decision (stamped). The references receive the auxiliary series as
  inputs where their feature builder can take them, exactly as the candidate does.


### V15. Loss weights of the multi-task loss; log offsets of the new maxima

- **Stamped text.** "The loss weights are fixed per-target inverse-scale normalisations, set before training."
  (`AM:149–150`) "A "primary-only" configuration (C3) is the declared ablation." (`AM:150`)
- **What the code does.** Every target is trained in standardised anomaly space (anomaly from the reference-window
  climatology divided by its scale), which is the inverse-scale normalisation. No further per-target weight is
  applied: the loss is the mean pinball loss over all OBSERVED target-lead entries of a batch (`nets.py:124–134`; the
  trainer passes no weights, `train_v2.py:99`), so a target with more observed days carries more of the loss. C3 sets
  the labels of every non-primary target to missing, i.e. weight 0 for 15 of the 20 trained targets
  (`train_v2.py:96–98`). SC daily max and turbidity daily max use the log offsets of their daily means (1.0 µS/cm; 0.1
  FNU), registered in the transform table of the original track at import time and checked equal (`prep.py:39–43`).
- **Reason.** The stamped sentence does not say whether the weights equalise targets or entries; the code's reading is
  recorded here before any run. The registration adds two keys and changes no existing key; only the v2 drivers and
  tests import the v2 package.
- **Effect on candidate and references.** Candidate only (the references are single-target or tree models). R-NH's
  loss on its one target is the same function.


### V16. SONDE-HF configuration C4 (CMAL head) is implemented

- **Stamped text.** SONDE-HF configurations: "C1 base (hourly on, multi-task, quantile head, hidden 256/128); C2
  hourly branch off; C3 primary-only loss; C4 CMAL head; C5 hidden 384/192" (`AM:52`).
- **What the code does.** C4 builds a CMAL head for all targets and trains it with the CMAL negative log-likelihood;
  its 19 quantiles come from the mixture CDF by bisection, 60 iterations (`nets.py:213–215`; `nets.py:109`;
  `test_sonde_v2.py:875`). The anchor and the ordering constraints apply with every head: for C4 the anchor shifts the
  location of every mixture component and the ordering is applied to the quantiles, not inside the likelihood
  (`nets.py:230–231`). Hidden sizes: daily hindcast LSTM 256, hourly LSTM 128, forecast LSTM 128 (C5: daily 384,
  forecast 192; hourly 128) (`train_v2.py:34–36`).
- **Reason.** ruling K `DR:65`. In the reviewed state C4 was the same model as C1 (findings U5, P2-09, F09).
- **Effect on candidate and references.** The candidate has its five stamped configurations; R-NH has five (V13).


### V17. Inputs and structure beyond the stamped description

- **Stamped text.** SONDE-HF row: "daily 365-d hindcast LSTM + optional hourly 14-d hindcast LSTM → state hand-off →
  14-step forecast LSTM over a masked-mean NWP embedding; multi-task monotone 19-quantile heads with ordering
  constraints and the damped-anomaly anchor; AR inputs with observed/missing flags" (`AM:52`). Parity list: "static
  attributes" (`AM:78`).
- **What the code does.** (a) Hourly branch: five variables (`data_v2.py:34`), the 336 hours of the 14 local days that
  end with the issue day, the last day of the daily hindcast (`data_v2.py:191`), approved (non-provisional) values
  only, as levels standardised with the site's daily statistics and clipped to ±8, each with an observed flag, plus
  hour-of-day sine and cosine in local standard time from the site registry's UTC offset (`data_v2.py:38, 210, 228,
  231`). The pH slot holds the hourly pH MEAN standardised with the daily pH-median statistics (`data_v2.py:36`). The
  daily LSTM runs to day −14, hands its state to the hourly LSTM and continues over the last 14 days; a site without
  hourly data gets a learned vector (`nets.py:236–246`). (b) Deterministic future inputs for both networks: clear-sky
  shortwave, day length and day-of-year sine and cosine of each lead day, from latitude and date only
  (`data_v2.py:158–161`); the GBM arms get clear-sky shortwave and day length of the target day
  (`gbm_nwp.py:178–187`). (c) Statics for every model: the NHDPlus channel slope (log10) and the StreamCat catchment
  table in addition to the original-track statics (`prep.py:113–117`); the list of static columns is in every parity
  record and must be equal across arms (V20). (d) Anchor: the last observed anomaly within 4 days, damped by a learned
  weight per target and lead initialised at 0.9^lead, for all SONDE-HF targets (`nets.py:220–221`; `data_v2.py:164`).
  (e) Ordering constraints in original units, per quantile level: DO min ≤ DO mean ≤ DO max, T min ≤ T mean ≤ T max
  (`train_v2.py:51–52`; `nets.py:287–289`). (f) Hindcast channels of both networks and context features of the GBM
  arms cover every cube variable except observed DO % saturation (V14).
- **Reason.** The stamped row names the blocks; these are the details that fix them.
- **Effect on candidate and references.** (b), (c) and (f) are the same for the candidate and the learned references.
  (a), (d) and (e) belong to the candidate only: the stamped R-NH row has no hourly branch and one target per model,
  and the R-NH code has no anchor and no ordering step.
- **Counts.** §4.4 (static columns; sites with slope).


### V18. Default values for missing site attributes

- **Stamped text.** Not specified. §2d: "Benson–Krause with pressure from elevation" (`AM:145`).
- **What the code does.** A missing site elevation is taken as 0 m (sea-level pressure) in the observed DO %
  saturation, in the transform of V14 and in the PF-GBM DO-saturation feature (`prep.py:272, 297`;
  `physics_features.py:45`). A missing latitude is taken as 40.0° for the deterministic solar inputs (`data_v2.py:85`;
  `gbm_nwp.py:183`). A site without a UTC offset in the site registry is taken as UTC−6 for the hourly branch
  (`data_v2.py:217`).
- **Reason.** A value is needed to compute the quantity; the same default is used for the observed series and the
  forecasts.
- **Effect on candidate and references.** DO % saturation at a site without elevation is biased low or high in the
  same way for the observation and for SONDE-HF's transform. The latitude default is not used by any loaded site in
  step 1.
- **Counts.** §4.4.


### V19. Chronos-2-cov ZS: covariates and configuration

- **Stamped text.** "Chronos-2-cov ZS: zero-shot Chronos-2 with NWP as known-future covariates (already required by
  M1-N §3)." (`AM:68`)
- **What the code does.** One configuration, zero-shot, no fitting (`declared.py:47`): the multivariate mode of the
  original-track Chronos-2 runner (context 512 days, climatology of the target days as a known-future input), plus
  three weather covariates: mean temperature, precipitation (log1p) and shortwave (`chronos_cov.py:20`), with observed
  gridMET as the past values and the archived bias-corrected GEFS of leads 1–7 as the known-future values; leads 8–14
  and issue days without a forecast are NaN (`chronos_cov.py:46–55`). Observed DO % saturation is not given to the
  model (`chronos_cov.py:61`; `test_sonde_v2.py:757`). Jobs stop at 95 minutes and resume (`sonde_v2_cv.py:387`).
- **Reason.** The stamped sentence fixes "NWP as known-future covariates"; which fields and how missing leads are
  passed is recorded here.
- **Effect on candidate and references.** Reference only. It reads the same corrected GEFS array as the other arms
  (hash in its parity record) and writes the 8 scored variables on the same held-out rows.


### V20. Declaration of configuration lists, parity log and its check, stamp guard

- **Stamped text.** "Their configuration lists are declared in the parity log before their CV runs." (`AM:85`) Parity
  list of the amendment at `AM:76–82`.
- **What the code does.** (a) Stamp guard: every phase of both drivers, the declaration and the registry row of the
  counts script first call `require_stamp` (`parity.py:150`; `sonde_v2_cv.py:96, 468, 524, 539`;
  `test_sonde_v2.py:569`). It passes only if this file exists and is read-only, every file matching
  `docs/preregistration/sonde_arch_v2_deviations_v2*.md` is read-only, the code list of §6 names every file of the
  three code globs with its current sha256, and a registry row `M1-sonde` / `sonde/prereg_snapshot` written by the
  lead carries the sha256 of each of those files and the code hash of §7 (`parity.py:95–143`; `test_sonde_v2.py:479,
  527`). Any later edit of a listed file closes the drivers. (b) Declaration: `--phase declare` writes the
  configuration lists of all five arms, with every setting that defines a run (configurations, outputs, seeds, row
  rule, product, horizons, GBM settings, network defaults, stage-2 rule, ordering constraints; `declared.py:93–107`),
  once, to a read-only file, to the parity log and to the registry (`sonde_arch_v2/declaration`) with its sha256
  (`parity.py:159–186`). Every run refuses an undeclared configuration, a list that differs from the declared one, or
  a declaration that is not in the registry (`parity.py:190–207`; `test_sonde_v2.py:582, 618`). (c) Parity record of
  every run, Chronos-2-cov included: arm, configuration, fold, declaration hash, numbers of training rows and sites,
  hashes of the fit, early-stopping and held-out site sets, site types, feature columns, static columns, hashes of the
  input arrays, fit window (`parity.py:225–237`). (d) Check, not only record: before the step-1 statistic and before
  any combination of CV folds, `check` raises unless all records of a fold agree on held-out sites, static columns,
  number of loaded sites, declaration and shared input hashes, and all fitted arms agree on fit and early-stopping
  sites (`parity.py:252`; `sonde_v2_cv.py:526, 542`; `test_sonde_v2.py:1257`).
- **Reason.** rulings L `DR:67`, O `DR:73`, R `DR:131`, item 4 `DR:101`. In the reviewed state no guard and no
  declaration existed (findings U7, U8, P2-10, F04, F15).
- **Effect on candidate and references.** Same mechanism for the candidate and every reference; the candidate's
  configuration list is declared with the references' lists, before the first run of any arm. "Extra training
  networks" of the parity list: no arm uses any site outside the split's train- and val-role sites in this round.


### V21. Code in the list of §6 that no driver of this round calls

- **Stamped text.** Amendment: E2 / E2-B ensembles (`AM:53, 71`) and the M1-H hourly track (lead ruling R4, `LR:47`).
- **What the code does.** `ensemble.py` (ensemble weights and their cross-fitted score) and `hourly.py` (hourly
  targets and naive references for M1-H) are imported by the unit tests only; `masked_mean_products` (`nwp.py:263`) is
  not called by any model, because there is one product. `sonde_v2_smoke.py` and `sonde_v2_gbm_timing.py` read
  train-role sites × train block only and write no forecast of a val row and no registry row of a result. No CV
  scorer, no ensemble driver and no M1-H model exist yet.
- **Reason.** ruling §2 O `DR:94`: the M1-H model code, its CV driver, the CV scorer and the ensemble driver come
  later as NEW files under their own dated, stamped deviation; M1-H is done before the freeze.
- **Effect on candidate and references.** None on this round's runs. The CV scorer must use the reference-window
  climatology for the naive references (V23).


### V22. Date axes, seeds and what differs between step 1 and CV

- **Stamped text.** "pre-declared, runs after the stamp, and uses train-role sites × train block only." (`AM:164–165`)
  "5-fold basin-group CV over train+val-role sites × val block." (`AM:190`) "3 seeds of the selected configuration(s),
  trained on train-role sites × train block with inner early stopping." (`AM:201–202`)
- **What the code does.** Step 1 loads train-role sites and dates 2007-10-01 … 2021-09-30; CV loads train- and
  val-role sites and dates to 2023-09-30 (`sonde_v2_step1.py:94`; `sonde_v2_cv.py:72`). Consequences: the pooled HUC2
  bias maps (V02), the noise sample of the perfect-prog input (V03), the set of site-variables with a reference
  climatology (V09) and the standardisation statistics (computed from the fit sites of each run, `data_v2.py:78–79`)
  are not the same arrays in step 1 and in CV. Within a use they are the same for every arm. The error-model fit set
  is train-role sites in both. One seed (0) per configuration in step 1 and in CV (`declared.py:50`); the 3 seeds of
  the stamped final training are not run in this round. A CV fold of an arm that has no held-out val row for a
  variable writes no file for it and records the fact (`sonde_v2_cv.py:555`).
- **Reason.** Step 1 must not load val-block data (stamped), so its build cannot be the CV build. The superseded
  statement that both drivers use the same date axes was wrong (finding F07).
- **Effect on candidate and references.** No comparison is made across the two uses; each contrast is inside one use.


### V23. Paths of the original track that still use all training years; corrected wording

- **Stamped text.** Parity list: "2-year reference climatologies" (`AM:81`).
- **What the code does.** (a) Site keep/drop: `prepare_v2` calls the unedited loader of the original track, which
  drops a site only when it has no finite training-years climatology for ANY variable (`sonde/common.py:225–231`);
  that climatology is then replaced (V09) and decides nothing else. A site kept by it but without any reference-window
  climatology has no row. (b) The naive references of the original-track module
  (`sentinel/models/baselines/sonde_baselines.py`: damping factors and residual quantiles fitted on all training
  years) are not called by any code of §6, which imports from that module only the constant of the 19 quantile levels
  (`gbm_nwp.py:49`); the CV scorer, a later file, must fit them inside the same row rule (V21). (c) Standardisation
  floors and site statistics of the network tensors come from the unedited tensor class, restricted to the fit sites
  of the run (V22). (d) Wording corrections to the superseded files: "6,040 of 6,516 registry sites" described the
  StreamCat site table (6,516 rows, 6,040 with a slope; read again today), not the Sonde site registry; the counts for
  the loaded sites are in §4.4.
- **Reason.** The frozen files are not edited (ruling D `DR:51`). Finding F14.
- **Effect on candidate and references.** Same for every arm.



## 3. Review findings and verifier items: where each is resolved

One line per finding of the four review files of §0 (rulings P `DR:75`, §2 order `DR:106`). "Resolved" = the code of
§6 was changed and the entry describes the present code; "recorded" = kept as a deviation in the entry.

| Finding | Severity | Where |
|---|---|---|
| D1 | should_fix | recorded — one GEFSv12 map from reforecast pairs only, applied to both archives (V02). |
| D2 | blocking | resolved — Q19 (19 fitted levels) is the mandatory configuration and the step-1 comparator; Q and LS are extra (V12, V06). |
| D3 | should_fix | recorded — zero for a missing field inside an available lead, with a flag (V04). |
| D4 | should_fix | recorded — log offsets of SC max and turbidity max (V15). |
| D5 | should_fix | resolved — no perfect-prog substitution anywhere outside stage 1 and step 1; gap rows are past-weather rows (V05). |
| U1 | blocking | resolved — one product for every arm; a second product is a hard error (V01). |
| U2 | should_fix | resolved — GBM arms fit the whole train block; frozen fit_end not used (V10). |
| U3 | blocking | resolved — issue rule for every model and use; window must lie inside the loaded dates; counts in §4.1 (V09). |
| U4 | should_fix | resolved — missing month file or coverage < 0.90 is a hard error (V05). |
| U5 | should_fix | resolved — C4 implemented and tested (V16). |
| U6 | should_fix | resolved — gage height restored; derived targets and the DO % saturation transform recorded (V14). |
| U7 | should_fix | resolved — declaration of all lists before the first run, enforced (V20). |
| U8 | blocking | resolved — this file is generated from the code at the state of §6; the guard closes the drivers on any later change (V20, §6). |
| U9 | should_fix | recorded — deterministic inputs, static embedding in the R-NH head, statics for every model (V13, V17). |
| U10 | note | recorded — early-stopping groups include val-role sites in CV, train-block rows only (V11). |
| U11 | note | recorded — degree-day definitions; no member mean or spread (V07). |
| P2-01 | blocking | resolved — gage height restored (V14). |
| P2-02 | blocking | resolved — as U2 (V10). |
| P2-03 | blocking | resolved — full-rows rung in the cap ladder, cap rule written before timing, lead ruling in the code before any GBM run (V12, §5). |
| P2-04 | blocking | resolved — as U3 (V09). |
| P2-05 | should_fix | resolved — as U1; no GFS slot exists (V01). |
| P2-06 | should_fix | resolved — outputs per arm stated as the code writes them (V14). |
| P2-07 | should_fix | resolved — complete record of the bias map; counts in §4.2 (V02). |
| P2-08 | should_fix | resolved — R-NH rows are those labelled for its own target (V13). |
| P2-09 | should_fix | resolved — as U5 (V16). |
| P2-10 | should_fix | resolved — as U7 (V20). |
| P2-11 | should_fix | resolved/recorded — degree-days are exponentially weighted degree-days, NaN when missing; no members (V07). |
| P2-12 | should_fix | resolved — error model ends 2019-09-30 as stamped (V03). |
| P2-13 | note | resolved — only rulings on disk are cited (§0). |
| P2-14 | note | resolved — citations are generated from the files at the state of §6 (§0). |
| F01 | blocking | resolved — as U3 (V09). |
| F02 | blocking | resolved — as P2-01 (V14). |
| F03 | blocking | resolved — counts in §4, timing in §5; no PENDING item. |
| F04 | should_fix | resolved — stamp guard in every phase (V20). |
| F05 | should_fix | resolved — as U2 (V10). |
| F06 | should_fix | resolved — as P2-06 (V14). |
| F07 | should_fix | resolved/recorded — different date axes stated; error-model window fixed (V22, V03). |
| F08 | should_fix | resolved — weight decay and learning rate come from the configuration in both stages (V13). |
| F09 | should_fix | resolved — as U5 (V16). |
| F10 | should_fix | resolved — early-stopping rows drawn after the finite-target mask by one seeded rule in both drivers; counts in the parity log (V11, V12). |
| F11 | should_fix | resolved — as P2-07 (V02). |
| F12 | note | resolved — as P2-14. |
| F13 | note | recorded — features stop at lead 7; the target-day Tmean is the lead-7 value for h > 7 (V07). |
| F14 | note | recorded — remaining paths of the original track (V23). |
| F15 | note | resolved — parity check enforced; Chronos-2-cov writes a record (V20). |
| F16 | note | recorded — loss weights per observed entry; C3 trains and writes primary targets only; elevation default (V15, V14, V18). |
| F17 | note | resolved/recorded — slope counts corrected (§4.4); hourly pH slot recorded (V17). |

`M1p2_verify.json`: its 31 verified entries refer to findings of the two part-2 reviews (F01, F02, F03, F04, F05, F06,
F07, F08, F09, F10, F11, F12, F13, F14, F15, F16, F17, P2-01, P2-02, P2-03, P2-04, P2-05, P2-06, P2-07, P2-08, P2-09,
P2-10, P2-11, P2-12, P2-13, P2-14); each `condition_for_rulings` is met through the line of that finding above. Its
five `missed` items:

1. no model in CV learns from GFS — recorded, with the true reason (V01).

2. the unit of "one fit" — defined in `TR` §1 (Q19 = 19 boosters, Q = 7, LS = 2) before the timing (V12, §5).

3. networks trained on operational GEFS in stage 2 and GBM arms not — resolved: the GBM arms are fitted on the
   archived forecasts of the whole train block (V05, V10).

4. cumulative degree-days 0 instead of NaN for a missing product — resolved (V07).

5. audit of the reviewers' checked-ok lists (the "two CV slots" pass was misleading) — resolved: one product (V01).

Verifier items 1–7 of `DR:98, 104`: 1 → V01, V04, V05; 2 → V12, §5; 3 → V10, V11; 4 → V20; 5 → V13; 6 → V07; 7 → V02,
§4.2. Notes D1, D3, D4, D5, U6, U9 of ruling P `DR:75`: lines above.


## 4. Counts

Counts and weather only: no water-quality label value, no label statistic and no forecast enters this section (rulings
F `DR:55`, I `DR:61`). The records were written by `sonde_v2_counts.py`; after the stamp its `log` mode writes the
weather-only diagnostic to the registry (`sonde_arch_v2/p3_bias_diagnostic`).

### 4.1 Reference window and issue rule (V09)

**4.1a Use `step1`** — record `sentinel_runs/sonde/v2_smoke/counts_step1.json`, written 2026-11-11T18:34:32Z; 1,610
loaded sites (train-role 1,610); dates 2007-10-01 … 2021-09-30; window 730 days, ≥ 365 observed, embargo 42 days. Code
of the record = the code of §6.

Site-variable pairs by first usable issue day (a = pairs with data; b = no climatology, < 365 observed days in the
window or window not inside the loaded dates; c = first usable issue day in the train block; d = after the train
block, inside the loaded dates; e = after the loaded dates):

| Variable | role | a | b | c | d | e |
|---|---|---:|---:|---:|---:|---:|
| `temp_c_mean` | train | 1,325 | 200 | 1,104 | 0 | 21 |
| `do_mgl_mean` | train | 394 | 87 | 302 | 0 | 5 |
| `do_mgl_min` | train | 390 | 75 | 311 | 0 | 4 |
| `spcond_uscm_mean` | train | 769 | 154 | 602 | 0 | 13 |
| `ph_median` | train | 303 | 67 | 231 | 0 | 5 |
| `turb_fnu_mean` | train | 276 | 63 | 208 | 0 | 5 |
| `turb_fnu_median` | train | 192 | 49 | 138 | 0 | 5 |
| `no3no2_mgl_mean` | train | 46 | 9 | 37 | 0 | 0 |
| `temp_c_max` | train | 1,275 | 178 | 1,075 | 0 | 22 |
| `temp_c_min` | train | 1,275 | 178 | 1,075 | 0 | 22 |
| `temp_c_7dadm` | train | 1,273 | 217 | 1,044 | 0 | 12 |
| `do_mgl_max` | train | 391 | 75 | 312 | 0 | 4 |
| `do_amp_mgl` | train | 389 | 74 | 311 | 0 | 4 |
| `do_pctsat` | train | 390 | 84 | 301 | 0 | 5 |
| `spcond_uscm_max` | train | 716 | 135 | 567 | 0 | 14 |
| `turb_fnu_max` | train | 381 | 93 | 279 | 0 | 9 |
| `discharge_cfs_mean` | train | 1,271 | 70 | 1,195 | 0 | 6 |
| `gage_ft_mean` | train | 468 | 19 | 447 | 0 | 2 |

Position of the reference window (f = first finite value on the cube start 2007-10-01, so the true record may start
earlier; g = window ends in the train block, with climatology; h = ends in the val block, with climatology; i = window
cut by the end of the loaded dates and a climatology was built — must be 0; j = cut by the end of the loaded dates, no
climatology; k = complete window, < 365 observed days, no climatology):

| Variable | role | f | g | h | i | j | k |
|---|---|---:|---:|---:|---:|---:|---:|
| `temp_c_mean` | train | 437 | 1,125 | 0 | 0 | 91 | 109 |
| `do_mgl_mean` | train | 121 | 307 | 0 | 0 | 39 | 48 |
| `do_mgl_min` | train | 115 | 315 | 0 | 0 | 41 | 34 |
| `spcond_uscm_mean` | train | 270 | 615 | 0 | 0 | 75 | 79 |
| `ph_median` | train | 92 | 236 | 0 | 0 | 37 | 30 |
| `turb_fnu_mean` | train | 46 | 213 | 0 | 0 | 23 | 40 |
| `turb_fnu_median` | train | 37 | 143 | 0 | 0 | 29 | 20 |
| `no3no2_mgl_mean` | train | 0 | 37 | 0 | 0 | 6 | 3 |
| `temp_c_max` | train | 459 | 1,097 | 0 | 0 | 92 | 86 |
| `temp_c_min` | train | 461 | 1,097 | 0 | 0 | 91 | 87 |
| `temp_c_7dadm` | train | 0 | 1,056 | 0 | 0 | 102 | 115 |
| `do_mgl_max` | train | 114 | 316 | 0 | 0 | 40 | 35 |
| `do_amp_mgl` | train | 114 | 315 | 0 | 0 | 40 | 34 |
| `do_pctsat` | train | 120 | 306 | 0 | 0 | 35 | 49 |
| `spcond_uscm_max` | train | 246 | 581 | 0 | 0 | 76 | 59 |
| `turb_fnu_max` | train | 76 | 288 | 0 | 0 | 48 | 45 |
| `discharge_cfs_mean` | train | 926 | 1,201 | 0 | 0 | 29 | 41 |
| `gage_ft_mean` | train | 318 | 449 | 0 | 0 | 10 | 9 |

Rows of the common row set (site × issue day with at least one scored horizon) after the issue rule: kept, dropped by
the issue rule, sites with kept rows, and kept rows issued inside the reference window of ANOTHER input variable of
the same site (V09):

| Variable | block | rows kept | rows dropped | sites | inside another variable's window |
|---|---|---:|---:|---:|---:|
| `temp_c_mean` | train | 2,318,852 | 804,968 | 1,058 | 61,580 |
| `do_mgl_mean` | train | 641,457 | 215,608 | 283 | 24,802 |
| `do_mgl_min` | train | 683,408 | 223,063 | 296 | 33,132 |
| `spcond_uscm_mean` | train | 1,202,769 | 434,324 | 560 | 33,557 |
| `ph_median` | train | 469,401 | 168,233 | 220 | 24,627 |
| `turb_fnu_mean` | train | 312,173 | 147,443 | 202 | 15,492 |
| `turb_fnu_median` | train | 243,551 | 98,794 | 129 | 12,382 |
| `no3no2_mgl_mean` | train | 36,282 | 23,891 | 32 | 5,416 |
| `temp_c_max` | train | 2,594,765 | 790,597 | 1,047 | 129,338 |
| `temp_c_min` | train | 2,595,538 | 790,805 | 1,047 | 128,873 |
| `temp_c_7dadm` | train | 2,477,197 | 752,078 | 1,018 | 122,040 |
| `do_mgl_max` | train | 683,844 | 223,808 | 297 | 33,133 |
| `do_amp_mgl` | train | 683,266 | 223,002 | 296 | 33,117 |
| `do_pctsat` | train | 632,246 | 214,069 | 282 | 22,702 |
| `spcond_uscm_max` | train | 1,273,124 | 415,950 | 541 | 67,727 |
| `turb_fnu_max` | train | 489,265 | 200,017 | 264 | 22,827 |
| `discharge_cfs_mean` | train | 4,076,145 | 857,518 | 1,178 | 346,514 |
| `gage_ft_mean` | train | 1,492,544 | 338,295 | 442 | 127,039 |

Scored rows (finite target) per horizon, kept / dropped by the issue rule:

| Variable | block | h=1 | h=2 | h=3 | h=5 | h=7 | h=10 | h=14 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `temp_c_mean` | train | 2,289,796 / 793,987 | 2,285,713 / 792,710 | 2,282,187 / 791,626 | 2,275,996 / 789,769 | 2,270,503 / 788,211 | 2,262,891 / 786,141 | 2,253,527 / 783,707 |
| `do_mgl_mean` | train | 625,010 / 209,562 | 623,066 / 208,910 | 621,495 / 208,371 | 618,897 / 207,511 | 616,734 / 206,842 | 613,976 / 206,023 | 610,804 / 205,040 |
| `do_mgl_min` | train | 667,291 / 216,934 | 665,337 / 216,266 | 663,738 / 215,711 | 661,069 / 214,820 | 658,813 / 214,133 | 655,845 / 213,298 | 652,471 / 212,324 |
| `spcond_uscm_mean` | train | 1,166,428 / 421,969 | 1,162,350 / 420,678 | 1,159,032 / 419,621 | 1,153,708 / 417,890 | 1,149,389 / 416,534 | 1,143,981 / 414,861 | 1,137,953 / 413,026 |
| `ph_median` | train | 458,684 / 164,232 | 457,490 / 163,821 | 456,479 / 163,471 | 454,728 / 162,888 | 453,268 / 162,412 | 451,328 / 161,784 | 448,966 / 161,059 |
| `turb_fnu_mean` | train | 291,761 / 137,066 | 289,853 / 136,160 | 288,379 / 135,448 | 286,131 / 134,401 | 284,485 / 133,657 | 282,573 / 132,838 | 280,471 / 132,067 |
| `turb_fnu_median` | train | 221,874 / 90,439 | 219,997 / 89,707 | 218,772 / 89,187 | 217,226 / 88,450 | 216,136 / 87,922 | 214,796 / 87,372 | 213,521 / 86,892 |
| `no3no2_mgl_mean` | train | 34,547 / 22,389 | 34,358 / 22,248 | 34,200 / 22,134 | 33,915 / 21,970 | 33,674 / 21,850 | 33,355 / 21,689 | 32,978 / 21,533 |
| `temp_c_max` | train | 2,564,921 / 780,719 | 2,560,738 / 779,550 | 2,557,106 / 778,550 | 2,550,729 / 776,847 | 2,545,086 / 775,408 | 2,537,247 / 773,445 | 2,527,670 / 771,131 |
| `temp_c_min` | train | 2,565,415 / 780,743 | 2,561,215 / 779,558 | 2,557,579 / 778,552 | 2,551,219 / 776,837 | 2,545,580 / 775,402 | 2,537,781 / 773,445 | 2,528,235 / 771,124 |
| `temp_c_7dadm` | train | 2,438,574 / 739,738 | 2,428,325 / 736,691 | 2,418,262 / 733,691 | 2,401,882 / 728,809 | 2,392,822 / 726,155 | 2,383,168 / 723,480 | 2,372,619 / 720,777 |
| `do_mgl_max` | train | 667,708 / 217,722 | 665,769 / 217,052 | 664,168 / 216,497 | 661,496 / 215,611 | 659,245 / 214,913 | 656,283 / 214,065 | 652,922 / 213,087 |
| `do_amp_mgl` | train | 666,854 / 216,737 | 664,886 / 216,059 | 663,273 / 215,498 | 660,594 / 214,603 | 658,331 / 213,910 | 655,360 / 213,076 | 651,984 / 212,104 |
| `do_pctsat` | train | 615,491 / 207,747 | 613,553 / 207,078 | 611,991 / 206,525 | 609,381 / 205,638 | 607,225 / 204,938 | 604,482 / 204,081 | 601,343 / 203,077 |
| `spcond_uscm_max` | train | 1,239,609 / 404,931 | 1,235,770 / 403,738 | 1,232,617 / 402,756 | 1,227,558 / 401,187 | 1,223,420 / 399,963 | 1,218,233 / 398,465 | 1,212,364 / 396,893 |
| `turb_fnu_max` | train | 452,819 / 186,272 | 449,592 / 185,016 | 447,290 / 184,098 | 444,138 / 182,769 | 441,812 / 181,846 | 439,149 / 180,881 | 436,437 / 179,939 |
| `discharge_cfs_mean` | train | 3,975,510 / 836,767 | 3,963,435 / 834,425 | 3,954,021 / 832,554 | 3,939,301 / 829,730 | 3,927,541 / 827,497 | 3,912,585 / 824,617 | 3,895,583 / 821,580 |
| `gage_ft_mean` | train | 1,474,801 / 333,262 | 1,473,097 / 332,881 | 1,471,744 / 332,568 | 1,469,576 / 332,079 | 1,467,705 / 331,709 | 1,465,203 / 331,278 | 1,462,263 / 330,859 |

**4.1b Use `cv`** — record `sentinel_runs/sonde/v2_smoke/counts_cv.json`, written 2026-11-11T18:46:12Z; 1,919 loaded
sites (train-role 1,610, val-role 309); dates 2007-10-01 … 2023-09-30; window 730 days, ≥ 365 observed, embargo 42
days. Code of the record = the code of §6.

Site-variable pairs by first usable issue day (a = pairs with data; b = no climatology, < 365 observed days in the
window or window not inside the loaded dates; c = first usable issue day in the train block; d = after the train
block, inside the loaded dates; e = after the loaded dates):

| Variable | role | a | b | c | d | e |
|---|---|---:|---:|---:|---:|---:|
| `temp_c_mean` | train | 1,399 | 186 | 1,104 | 106 | 3 |
| `temp_c_mean` | val | 276 | 24 | 239 | 13 | 0 |
| `do_mgl_mean` | train | 418 | 76 | 302 | 40 | 0 |
| `do_mgl_mean` | val | 83 | 18 | 62 | 3 | 0 |
| `do_mgl_min` | train | 415 | 63 | 311 | 41 | 0 |
| `do_mgl_min` | val | 81 | 19 | 58 | 4 | 0 |
| `spcond_uscm_mean` | train | 819 | 133 | 602 | 82 | 2 |
| `spcond_uscm_mean` | val | 172 | 12 | 150 | 10 | 0 |
| `ph_median` | train | 328 | 59 | 231 | 38 | 0 |
| `ph_median` | val | 74 | 11 | 61 | 2 | 0 |
| `turb_fnu_mean` | train | 290 | 54 | 208 | 27 | 1 |
| `turb_fnu_mean` | val | 40 | 4 | 34 | 2 | 0 |
| `turb_fnu_median` | train | 203 | 35 | 138 | 28 | 2 |
| `turb_fnu_median` | val | 22 | 6 | 14 | 2 | 0 |
| `no3no2_mgl_mean` | train | 51 | 9 | 37 | 4 | 1 |
| `no3no2_mgl_mean` | val | 8 | 0 | 8 | 0 | 0 |
| `temp_c_max` | train | 1,341 | 155 | 1,075 | 108 | 3 |
| `temp_c_max` | val | 238 | 21 | 205 | 12 | 0 |
| `temp_c_min` | train | 1,341 | 156 | 1,075 | 107 | 3 |
| `temp_c_min` | val | 237 | 21 | 204 | 12 | 0 |
| `temp_c_7dadm` | train | 1,336 | 187 | 1,044 | 103 | 2 |
| `temp_c_7dadm` | val | 235 | 25 | 198 | 12 | 0 |
| `do_mgl_max` | train | 416 | 64 | 312 | 40 | 0 |
| `do_mgl_max` | val | 81 | 19 | 58 | 4 | 0 |
| `do_amp_mgl` | train | 414 | 63 | 311 | 40 | 0 |
| `do_amp_mgl` | val | 81 | 19 | 58 | 4 | 0 |
| `do_pctsat` | train | 415 | 78 | 301 | 36 | 0 |
| `do_pctsat` | val | 81 | 18 | 61 | 2 | 0 |
| `spcond_uscm_max` | train | 767 | 115 | 567 | 83 | 2 |
| `spcond_uscm_max` | val | 150 | 14 | 126 | 10 | 0 |
| `turb_fnu_max` | train | 404 | 72 | 279 | 50 | 3 |
| `turb_fnu_max` | val | 55 | 8 | 44 | 3 | 0 |
| `discharge_cfs_mean` | train | 1,274 | 45 | 1,195 | 34 | 0 |
| `discharge_cfs_mean` | val | 240 | 9 | 228 | 3 | 0 |
| `gage_ft_mean` | train | 473 | 14 | 447 | 12 | 0 |
| `gage_ft_mean` | val | 81 | 6 | 74 | 1 | 0 |

Position of the reference window (f = first finite value on the cube start 2007-10-01, so the true record may start
earlier; g = window ends in the train block, with climatology; h = ends in the val block, with climatology; i = window
cut by the end of the loaded dates and a climatology was built — must be 0; j = cut by the end of the loaded dates, no
climatology; k = complete window, < 365 observed days, no climatology):

| Variable | role | f | g | h | i | j | k |
|---|---|---:|---:|---:|---:|---:|---:|
| `temp_c_mean` | train | 437 | 1,125 | 88 | 0 | 74 | 112 |
| `temp_c_mean` | val | 79 | 239 | 13 | 0 | 8 | 16 |
| `do_mgl_mean` | train | 121 | 307 | 35 | 0 | 24 | 52 |
| `do_mgl_mean` | val | 16 | 62 | 3 | 0 | 2 | 16 |
| `do_mgl_min` | train | 115 | 315 | 37 | 0 | 25 | 38 |
| `do_mgl_min` | val | 19 | 58 | 4 | 0 | 3 | 16 |
| `spcond_uscm_mean` | train | 270 | 615 | 71 | 0 | 50 | 83 |
| `spcond_uscm_mean` | val | 47 | 150 | 10 | 0 | 2 | 10 |
| `ph_median` | train | 92 | 236 | 33 | 0 | 25 | 34 |
| `ph_median` | val | 22 | 61 | 2 | 0 | 3 | 8 |
| `turb_fnu_mean` | train | 46 | 213 | 23 | 0 | 14 | 40 |
| `turb_fnu_mean` | val | 5 | 34 | 2 | 0 | 1 | 3 |
| `turb_fnu_median` | train | 37 | 143 | 25 | 0 | 11 | 24 |
| `turb_fnu_median` | val | 5 | 14 | 2 | 0 | 1 | 5 |
| `no3no2_mgl_mean` | train | 0 | 37 | 5 | 0 | 5 | 4 |
| `no3no2_mgl_mean` | val | 0 | 8 | 0 | 0 | 0 | 0 |
| `temp_c_max` | train | 459 | 1,097 | 89 | 0 | 66 | 89 |
| `temp_c_max` | val | 77 | 205 | 12 | 0 | 9 | 12 |
| `temp_c_min` | train | 461 | 1,097 | 88 | 0 | 66 | 90 |
| `temp_c_min` | val | 77 | 204 | 12 | 0 | 9 | 12 |
| `temp_c_7dadm` | train | 0 | 1,056 | 93 | 0 | 63 | 124 |
| `temp_c_7dadm` | val | 0 | 198 | 12 | 0 | 6 | 19 |
| `do_mgl_max` | train | 114 | 316 | 36 | 0 | 25 | 39 |
| `do_mgl_max` | val | 19 | 58 | 4 | 0 | 3 | 16 |
| `do_amp_mgl` | train | 114 | 315 | 36 | 0 | 25 | 38 |
| `do_amp_mgl` | val | 19 | 58 | 4 | 0 | 3 | 16 |
| `do_pctsat` | train | 120 | 306 | 31 | 0 | 25 | 53 |
| `do_pctsat` | val | 15 | 61 | 2 | 0 | 2 | 16 |
| `spcond_uscm_max` | train | 246 | 581 | 71 | 0 | 51 | 64 |
| `spcond_uscm_max` | val | 47 | 126 | 10 | 0 | 3 | 11 |
| `turb_fnu_max` | train | 76 | 288 | 44 | 0 | 23 | 49 |
| `turb_fnu_max` | val | 8 | 44 | 3 | 0 | 2 | 6 |
| `discharge_cfs_mean` | train | 926 | 1,201 | 28 | 0 | 3 | 42 |
| `discharge_cfs_mean` | val | 158 | 228 | 3 | 0 | 2 | 7 |
| `gage_ft_mean` | train | 318 | 449 | 10 | 0 | 5 | 9 |
| `gage_ft_mean` | val | 28 | 74 | 1 | 0 | 1 | 5 |

Rows of the common row set (site × issue day with at least one scored horizon) after the issue rule: kept, dropped by
the issue rule, sites with kept rows, and kept rows issued inside the reference window of ANOTHER input variable of
the same site (V09):

| Variable | block | rows kept | rows dropped | sites | inside another variable's window |
|---|---|---:|---:|---:|---:|
| `temp_c_mean` | train | 2,898,630 | 1,025,813 | 1,291 | 104,560 |
| `temp_c_mean` | val | 664,925 | 22,248 | 1,111 | 13,275 |
| `do_mgl_mean` | train | 747,239 | 269,564 | 344 | 28,811 |
| `do_mgl_mean` | val | 166,454 | 13,185 | 296 | 2,335 |
| `do_mgl_min` | train | 787,749 | 275,559 | 354 | 38,589 |
| `do_mgl_min` | val | 172,720 | 13,477 | 305 | 2,787 |
| `spcond_uscm_mean` | train | 1,552,032 | 580,305 | 707 | 60,618 |
| `spcond_uscm_mean` | val | 365,195 | 17,695 | 613 | 10,558 |
| `ph_median` | train | 604,265 | 221,263 | 281 | 32,151 |
| `ph_median` | val | 141,212 | 11,294 | 249 | 3,497 |
| `turb_fnu_mean` | train | 368,089 | 180,383 | 233 | 17,188 |
| `turb_fnu_mean` | val | 97,397 | 6,856 | 185 | 1,811 |
| `turb_fnu_median` | train | 261,878 | 116,294 | 143 | 14,651 |
| `turb_fnu_median` | val | 64,743 | 10,297 | 121 | 911 |
| `no3no2_mgl_mean` | train | 53,111 | 30,061 | 40 | 6,290 |
| `no3no2_mgl_mean` | val | 16,566 | 1,591 | 30 | 0 |
| `temp_c_max` | train | 3,069,494 | 988,519 | 1,248 | 173,760 |
| `temp_c_max` | val | 670,927 | 21,239 | 1,103 | 15,164 |
| `temp_c_min` | train | 3,069,622 | 987,449 | 1,247 | 173,296 |
| `temp_c_min` | val | 669,791 | 20,985 | 1,101 | 15,167 |
| `temp_c_7dadm` | train | 2,932,286 | 947,263 | 1,214 | 164,617 |
| `temp_c_7dadm` | val | 637,030 | 19,159 | 1,071 | 14,455 |
| `do_mgl_max` | train | 788,131 | 276,009 | 355 | 38,587 |
| `do_mgl_max` | val | 172,807 | 13,087 | 305 | 2,787 |
| `do_amp_mgl` | train | 787,542 | 274,984 | 354 | 38,571 |
| `do_amp_mgl` | val | 172,567 | 13,087 | 304 | 2,787 |
| `do_pctsat` | train | 733,268 | 265,405 | 342 | 26,538 |
| `do_pctsat` | val | 164,019 | 11,420 | 289 | 2,335 |
| `spcond_uscm_max` | train | 1,541,536 | 544,739 | 666 | 97,673 |
| `spcond_uscm_max` | val | 353,918 | 17,571 | 589 | 10,312 |
| `turb_fnu_max` | train | 558,229 | 246,662 | 306 | 26,836 |
| `turb_fnu_max` | val | 141,868 | 16,029 | 260 | 3,101 |
| `discharge_cfs_mean` | train | 4,857,330 | 1,039,377 | 1,403 | 437,797 |
| `discharge_cfs_mean` | val | 832,065 | 5,229 | 1,321 | 24,131 |
| `gage_ft_mean` | train | 1,649,033 | 398,348 | 513 | 143,381 |
| `gage_ft_mean` | val | 310,399 | 2,680 | 466 | 2,779 |

Scored rows (finite target) per horizon, kept / dropped by the issue rule:

| Variable | block | h=1 | h=2 | h=3 | h=5 | h=7 | h=10 | h=14 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `temp_c_mean` | train | 2,862,569 / 1,011,656 | 2,857,525 / 1,009,951 | 2,853,153 / 1,008,496 | 2,845,504 / 1,005,952 | 2,838,773 / 1,003,840 | 2,829,430 / 1,001,034 | 2,818,044 / 997,728 |
| `temp_c_mean` | val | 655,432 / 19,241 | 653,504 / 19,220 | 651,761 / 19,204 | 648,530 / 18,859 | 645,556 / 18,834 | 641,256 / 18,805 | 635,731 / 18,771 |
| `do_mgl_mean` | train | 727,506 / 262,092 | 725,129 / 261,207 | 723,186 / 260,464 | 719,921 / 259,242 | 717,207 / 258,223 | 713,712 / 257,008 | 709,693 / 255,586 |
| `do_mgl_mean` | val | 162,029 / 12,953 | 161,339 / 12,936 | 160,750 / 12,923 | 159,642 / 12,902 | 158,668 / 12,233 | 157,336 / 12,204 | 155,680 / 12,177 |
| `do_mgl_min` | train | 768,727 / 267,802 | 766,369 / 266,876 | 764,425 / 266,095 | 761,136 / 264,812 | 758,363 / 263,747 | 754,698 / 262,482 | 750,522 / 261,039 |
| `do_mgl_min` | val | 168,155 / 13,222 | 167,438 / 13,205 | 166,831 / 13,193 | 165,695 / 13,173 | 164,694 / 12,502 | 163,343 / 12,472 | 161,646 / 12,448 |
| `spcond_uscm_mean` | train | 1,508,112 / 563,378 | 1,503,073 / 561,609 | 1,498,937 / 560,165 | 1,492,240 / 557,750 | 1,486,830 / 555,816 | 1,479,979 / 553,475 | 1,472,317 / 550,821 |
| `spcond_uscm_mean` | val | 353,448 / 15,979 | 351,906 / 15,952 | 350,574 / 15,930 | 348,283 / 15,575 | 346,316 / 15,237 | 343,670 / 15,202 | 340,414 / 14,523 |
| `ph_median` | train | 590,434 / 216,007 | 588,868 / 215,407 | 587,527 / 214,887 | 585,177 / 213,953 | 583,191 / 213,226 | 580,526 / 212,259 | 577,302 / 211,167 |
| `ph_median` | val | 137,482 / 11,125 | 136,956 / 11,112 | 136,495 / 11,102 | 135,625 / 10,774 | 134,885 / 10,754 | 133,805 / 10,727 | 132,472 / 10,701 |
| `turb_fnu_mean` | train | 344,545 / 167,753 | 342,319 / 166,645 | 340,592 / 165,768 | 337,929 / 164,477 | 335,969 / 163,567 | 333,659 / 162,569 | 331,140 / 161,545 |
| `turb_fnu_mean` | val | 90,150 / 5,120 | 89,355 / 5,088 | 88,727 / 5,060 | 87,706 / 5,033 | 86,890 / 5,000 | 85,845 / 4,978 | 84,701 / 4,969 |
| `turb_fnu_median` | train | 238,247 / 106,444 | 236,148 / 105,533 | 234,777 / 104,884 | 233,006 / 103,969 | 231,720 / 103,283 | 230,174 / 102,506 | 228,757 / 101,751 |
| `turb_fnu_median` | val | 60,293 / 8,645 | 59,863 / 8,600 | 59,534 / 8,565 | 59,058 / 8,522 | 58,678 / 8,489 | 58,183 / 8,443 | 57,593 / 7,637 |
| `no3no2_mgl_mean` | train | 50,643 / 28,246 | 50,388 / 28,072 | 50,172 / 27,932 | 49,788 / 27,725 | 49,463 / 27,565 | 49,020 / 27,349 | 48,502 / 27,133 |
| `no3no2_mgl_mean` | val | 16,006 / 870 | 15,921 / 861 | 15,848 / 856 | 15,725 / 849 | 15,614 / 845 | 15,471 / 846 | 15,291 / 841 |
| `temp_c_max` | train | 3,034,321 / 975,841 | 3,029,400 / 974,294 | 3,025,110 / 972,961 | 3,017,580 / 970,631 | 3,010,935 / 968,689 | 3,001,717 / 966,058 | 2,990,523 / 962,945 |
| `temp_c_max` | val | 661,579 / 18,238 | 659,695 / 18,217 | 657,992 / 18,204 | 654,820 / 17,864 | 651,879 / 17,842 | 647,627 / 17,817 | 642,183 / 17,790 |
| `temp_c_min` | train | 3,034,160 / 974,566 | 3,029,219 / 973,002 | 3,024,926 / 971,661 | 3,017,414 / 969,319 | 3,010,773 / 967,381 | 3,001,600 / 964,758 | 2,990,437 / 961,639 |
| `temp_c_min` | val | 660,438 / 17,983 | 658,559 / 17,962 | 656,858 / 17,949 | 653,691 / 17,609 | 650,765 / 17,587 | 646,527 / 17,563 | 641,094 / 17,535 |
| `temp_c_7dadm` | train | 2,886,227 / 930,711 | 2,874,030 / 926,547 | 2,862,058 / 922,447 | 2,842,640 / 915,844 | 2,831,888 / 912,262 | 2,820,324 / 908,651 | 2,807,808 / 905,085 |
| `temp_c_7dadm` | val | 623,203 / 17,312 | 618,916 / 17,182 | 614,699 / 17,054 | 608,039 / 16,892 | 604,104 / 16,851 | 599,397 / 16,812 | 593,893 / 16,750 |
| `do_mgl_max` | train | 769,031 / 268,262 | 766,685 / 267,329 | 764,735 / 266,546 | 761,435 / 265,261 | 758,665 / 264,188 | 755,001 / 262,926 | 750,837 / 261,485 |
| `do_mgl_max` | val | 168,217 / 12,834 | 167,498 / 12,816 | 166,885 / 12,804 | 165,746 / 12,784 | 164,743 / 12,112 | 163,391 / 12,079 | 161,687 / 12,053 |
| `do_amp_mgl` | train | 768,105 / 266,902 | 765,728 / 265,959 | 763,765 / 265,167 | 760,458 / 263,871 | 757,674 / 262,799 | 754,001 / 261,546 | 749,823 / 260,099 |
| `do_amp_mgl` | val | 167,977 / 12,832 | 167,259 / 12,814 | 166,649 / 12,802 | 165,510 / 12,782 | 164,511 / 12,109 | 163,166 / 12,077 | 161,474 / 12,050 |
| `do_pctsat` | train | 713,189 / 257,655 | 710,823 / 256,765 | 708,888 / 256,014 | 705,616 / 254,785 | 702,913 / 253,756 | 699,447 / 252,536 | 695,466 / 251,125 |
| `do_pctsat` | val | 159,532 / 11,212 | 158,855 / 11,199 | 158,276 / 11,189 | 157,188 / 11,172 | 156,232 / 10,506 | 154,924 / 10,483 | 153,297 / 10,462 |
| `spcond_uscm_max` | train | 1,501,944 / 529,429 | 1,497,341 / 527,791 | 1,493,529 / 526,447 | 1,487,340 / 524,238 | 1,482,302 / 522,464 | 1,475,927 / 520,335 | 1,468,738 / 517,990 |
| `spcond_uscm_max` | val | 342,688 / 15,854 | 341,239 / 15,826 | 339,993 / 15,807 | 337,835 / 15,455 | 335,978 / 15,118 | 333,488 / 15,088 | 330,400 / 14,413 |
| `turb_fnu_max` | train | 517,643 / 229,864 | 514,014 / 228,318 | 511,391 / 227,161 | 507,729 / 225,466 | 504,968 / 224,266 | 501,813 / 222,979 | 498,617 / 221,652 |
| `turb_fnu_max` | val | 132,953 / 12,800 | 131,961 / 12,733 | 131,180 / 12,677 | 129,959 / 12,613 | 128,977 / 12,555 | 127,713 / 12,494 | 126,372 / 11,965 |
| `discharge_cfs_mean` | train | 4,738,713 / 1,014,015 | 4,724,188 / 1,011,068 | 4,712,856 / 1,008,726 | 4,695,158 / 1,005,185 | 4,681,106 / 1,002,414 | 4,663,404 / 998,851 | 4,643,715 / 995,125 |
| `discharge_cfs_mean` | val | 809,020 / 5,051 | 805,573 / 5,038 | 802,708 / 5,027 | 797,770 / 5,012 | 793,459 / 4,992 | 787,651 / 4,967 | 780,537 / 4,940 |
| `gage_ft_mean` | train | 1,629,175 / 392,462 | 1,627,187 / 391,971 | 1,625,604 / 391,567 | 1,623,050 / 390,921 | 1,620,856 / 390,426 | 1,617,923 / 389,827 | 1,614,552 / 389,219 |
| `gage_ft_mean` | val | 307,314 / 2,643 | 306,620 / 2,641 | 305,988 / 2,639 | 304,827 / 2,641 | 303,712 / 2,640 | 302,155 / 2,639 | 300,128 / 2,635 |

### 4.2 Bias map, error model, product hashes (V02, V03)

**Use `step1`.** Product `gefs_c00`; coverage of the required windows (share of site × issue-day cells with a finite
lead-1 temperature): `reforecast_2007-10-01_2019-09-30` 1.0000 (threshold 0.9). GEFS issue days on the date axis:
4,475 of 5,114. sha256 of the corrected product array
`f06d68f7fd730dabed3560c0ea60d750b17b540cb80edbaa7cacc0d49a06fb1f`; of the perfect-prog sample
`232be7b72e73e6ff6b3013921e896bc6ea15cd9ac77ff7cba91096dd88ce1814` (seed 20261110).

Bias map (reforecast only, valid day <= 2021-09-30; site-month minimum 60 pairs, pooled minimum 3 pairs, slope clip
[0.5, 1.5]). Number of site × lead × month cells by source, per field:

| Field | 0 identity | 1 site-month | 2 site all-months | 3 pooled HUC2-month |
|---|---:|---:|---:|---:|
| `tmax_c` | 0 | 134,981 | 0 | 259 |
| `tmin_c` | 0 | 134,981 | 0 | 259 |
| `tmean_c` | 0 | 134,981 | 0 | 259 |
| `precip_mm` | 0 | 134,981 | 0 | 259 |
| `srad_wm2` | 0 | 134,981 | 0 | 259 |

Precipitation ratio b over 135,240 cells: min 0.0, 1st percentile 0.3063387289643288, median 0.9170671701431274, 99th
percentile 2.026843712329851, max 17.145160675048828. A ratio of 0 sets the corrected precipitation of that site, lead
and month to 0 (observed mean 0 in the fit pairs); the ratio is not clipped (V02). Sites whose mean-temperature map is
identity for every cell: 0.

Error model: 1,610 train-role fit sites, 4,376 issue days, last valid day 2019-09-30; standard deviation of the
mean-temperature residual (°C; DJF, MAM, JJA, SON) at lead 1: 1.84, 1.49, 1.21, 1.39; at lead 7: 3.70, 3.18, 2.31,
3.01; lead-to-lead correlation (Tmax, Tmin, Tmean, P, SW): 0.420, 0.485, 0.503, 0.073, 0.190; sha256 of the model
`5109a147a5c91e71177da86c5ab61d44151bf6b5459b420df469f1d6c240777f`.

**Use `cv`.** Product `gefs_c00`; coverage of the required windows (share of site × issue-day cells with a finite
lead-1 temperature): `reforecast_2007-10-01_2019-09-30` 1.0000; `operational_train_2020-09-23_2021-09-23` 1.0000;
`operational_val_2021-10-01_2023-09-23` 1.0000 (threshold 0.9). GEFS issue days on the date axis: 5,578 of 5,844.
sha256 of the corrected product array `19e580e8f2aaeaf3b755ccbb43dc0e1b3e0f9eeed731a9f04a81a5d234a1e86f`; of the
perfect-prog sample `172e3ac234655a8333850541659d255882318a8f8a86252e289e732a3253c79f` (seed 20261110).

Bias map (reforecast only, valid day <= 2021-09-30; site-month minimum 60 pairs, pooled minimum 3 pairs, slope clip
[0.5, 1.5]). Number of site × lead × month cells by source, per field:

| Field | 0 identity | 1 site-month | 2 site all-months | 3 pooled HUC2-month |
|---|---:|---:|---:|---:|
| `tmax_c` | 0 | 160,657 | 0 | 539 |
| `tmin_c` | 0 | 160,657 | 0 | 539 |
| `tmean_c` | 0 | 160,657 | 0 | 539 |
| `precip_mm` | 0 | 160,657 | 0 | 539 |
| `srad_wm2` | 0 | 160,657 | 0 | 539 |

Precipitation ratio b over 161,196 cells: min 0.0, 1st percentile 0.3149218544363976, median 0.9181854724884033, 99th
percentile 2.0115250349044773, max 17.145160675048828. A ratio of 0 sets the corrected precipitation of that site,
lead and month to 0 (observed mean 0 in the fit pairs); the ratio is not clipped (V02). Sites whose mean-temperature
map is identity for every cell: 0.

Error model: 1,610 train-role fit sites, 4,376 issue days, last valid day 2019-09-30; standard deviation of the
mean-temperature residual (°C; DJF, MAM, JJA, SON) at lead 1: 1.84, 1.49, 1.21, 1.39; at lead 7: 3.70, 3.18, 2.31,
3.01; lead-to-lead correlation (Tmax, Tmin, Tmean, P, SW): 0.420, 0.484, 0.503, 0.073, 0.190; sha256 of the model
`d223d06469b8f51e91e6b46c97223b5dc0c6981a80b547b225c4e6ab31178da0`.

### 4.3 GEFS availability and the weather-only diagnostic (V04, V05)

**Use `step1`.** Cells = site × issue day × lead (1–7) of the archived, bias-corrected product:

| Block | cells | with Tmean | share | Tmean but no P | Tmean but no SW | issue days | issue days without GEFS |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 57,634,780 | 50,433,250 | 0.8750 | 0 | 0 | 5,114 | 639 |

Runs of issue days without any GEFS forecast: 2020-01-01 … 2021-09-30 (639 days). Sites without any GEFS forecast: 0.
In step 1 only the reforecast is loaded (issue days to 2019-12-31) and both arms use the perfect-prog sample, which
exists for every day; the archived product is not an input of step 1.

Weather-only diagnostic `reforecast_fit_2007-10-01_2019-09-23` (bias-corrected GEFS minus observed gridMET, train
block; no water-quality value): mean / mean absolute difference per lead.

| Field | lead 1 | lead 2 | lead 3 | lead 4 | lead 5 | lead 6 | lead 7 | n at lead 1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `tmax_c` | -0.000 / 1.42 | -0.000 / 1.56 | -0.002 / 1.74 | -0.002 / 1.98 | -0.002 / 2.26 | -0.002 / 2.55 | -0.000 / 2.82 | 6,932,956 |
| `tmin_c` | -0.001 / 1.46 | -0.001 / 1.53 | -0.001 / 1.64 | -0.001 / 1.79 | -0.002 / 1.98 | -0.001 / 2.21 | -0.002 / 2.42 | 6,932,956 |
| `tmean_c` | -0.001 / 1.12 | -0.001 / 1.22 | -0.001 / 1.36 | -0.002 / 1.55 | -0.002 / 1.79 | -0.002 / 2.05 | -0.001 / 2.31 | 6,932,956 |
| `precip_mm` | +0.000 / 2.31 | +0.002 / 2.52 | +0.001 / 2.76 | -0.001 / 3.02 | -0.008 / 3.28 | -0.001 / 3.55 | +0.003 / 3.76 | 6,932,956 |
| `srad_wm2` | +0.027 / 21.83 | +0.018 / 23.29 | +0.015 / 25.04 | +0.010 / 27.00 | +0.024 / 29.01 | +0.014 / 30.79 | +0.024 / 32.41 | 6,932,956 |

**Use `cv`.** Cells = site × issue day × lead (1–7) of the archived, bias-corrected product:

| Block | cells | with Tmean | share | Tmean but no P | Tmean but no SW | issue days | issue days without GEFS |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 68,696,362 | 65,123,184 | 0.9480 | 0 | 0 | 5,114 | 266 |
| val | 9,806,090 | 9,806,090 | 1.0000 | 0 | 0 | 730 | 0 |

Runs of issue days without any GEFS forecast: 2020-01-01 … 2020-09-22 (266 days). Sites without any GEFS forecast: 0.

Weather-only diagnostic `reforecast_fit_2007-10-01_2019-09-23` (bias-corrected GEFS minus observed gridMET, train
block; no water-quality value): mean / mean absolute difference per lead.

| Field | lead 1 | lead 2 | lead 3 | lead 4 | lead 5 | lead 6 | lead 7 | n at lead 1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `tmax_c` | -0.000 / 1.43 | -0.000 / 1.57 | -0.002 / 1.75 | -0.002 / 1.98 | -0.001 / 2.26 | -0.001 / 2.55 | +0.001 / 2.83 | 8,228,281 |
| `tmin_c` | -0.001 / 1.45 | -0.002 / 1.53 | -0.001 / 1.64 | -0.001 / 1.79 | -0.002 / 1.99 | -0.001 / 2.21 | -0.002 / 2.42 | 8,228,281 |
| `tmean_c` | -0.000 / 1.11 | -0.001 / 1.22 | -0.001 / 1.36 | -0.001 / 1.55 | -0.002 / 1.79 | -0.001 / 2.06 | -0.001 / 2.31 | 8,228,281 |
| `precip_mm` | +0.000 / 2.36 | +0.001 / 2.58 | +0.000 / 2.82 | +0.000 / 3.10 | -0.007 / 3.37 | +0.001 / 3.65 | +0.004 / 3.87 | 8,228,281 |
| `srad_wm2` | +0.031 / 22.04 | +0.021 / 23.57 | +0.019 / 25.37 | +0.013 / 27.36 | +0.027 / 29.44 | +0.017 / 31.29 | +0.027 / 32.90 | 8,228,281 |

Weather-only diagnostic `operational_train_2020-09-23_2021-09-23` (bias-corrected GEFS minus observed gridMET, train
block; no water-quality value): mean / mean absolute difference per lead.

| Field | lead 1 | lead 2 | lead 3 | lead 4 | lead 5 | lead 6 | lead 7 | n at lead 1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `tmax_c` | +0.651 / 1.62 | +0.677 / 1.78 | +0.694 / 1.98 | +0.694 / 2.21 | +0.673 / 2.48 | +0.582 / 2.75 | +0.476 / 3.03 | 695,232 |
| `tmin_c` | +0.034 / 1.59 | +0.087 / 1.66 | +0.116 / 1.76 | +0.147 / 1.92 | +0.141 / 2.08 | +0.085 / 2.28 | +0.022 / 2.49 | 695,232 |
| `tmean_c` | +0.345 / 1.24 | +0.387 / 1.35 | +0.414 / 1.49 | +0.436 / 1.68 | +0.425 / 1.89 | +0.353 / 2.14 | +0.265 / 2.40 | 695,232 |
| `precip_mm` | -0.108 / 2.55 | -0.112 / 2.79 | -0.083 / 3.04 | -0.075 / 3.28 | -0.010 / 3.61 | +0.047 / 3.83 | -0.043 / 4.06 | 695,232 |
| `srad_wm2` | -0.377 / 21.50 | -0.524 / 23.08 | -0.585 / 25.07 | -0.592 / 26.71 | -0.924 / 29.36 | -0.548 / 31.11 | -0.074 / 32.68 | 695,232 |

### 4.4 Statics (V17, V18)

- Use `step1`, train-role sites: 1,610 sites; with NHDPlus slope 1,539; without elevation (default 0 m) 168; without
  latitude (default 40°) 0.

- Use `step1`: 36 static columns: `lat`, `lon`, `huc2`, `elev_m`, `log_drain_area_km2`, `slope`, `sc_log10_nhd_slope`,
  `sc_pctcrop2019ws`, `sc_pcthay2019ws`, `sc_pctimp2019ws`, `sc_pcturbhi2019ws`, `sc_pctow2019ws`, `sc_fertws`,
  `sc_manurews`, `sc_naniws`, `sc_pctagdrainagews`, `sc_popden2010ws`, `sc_wwtpalldensws`, `sc_rddensws`,
  `sc_pctcarbresidws`, `sc_caows`, `sc_na2ows`, `sc_clayws`, `sc_sandws`, `sc_permws`, `sc_kffactws`, `sc_rckdepws`,
  `sc_bfiws`, `sc_runoffws`, `sc_precip9120ws`, `sc_tmean9120ws`, `sc_wetindexws`, `sc_elevws`, `sc_wsareasqkm`,
  `sc_pctforest2019ws`, `sc_pctwetland2019ws`.

- Use `cv`, train-role sites: 1,610 sites; with NHDPlus slope 1,539; without elevation (default 0 m) 168; without
  latitude (default 40°) 0.

- Use `cv`, val-role sites: 309 sites; with NHDPlus slope 283; without elevation (default 0 m) 12; without latitude
  (default 40°) 0.

- Use `cv`: 36 static columns: `lat`, `lon`, `huc2`, `elev_m`, `log_drain_area_km2`, `slope`, `sc_log10_nhd_slope`,
  `sc_pctcrop2019ws`, `sc_pcthay2019ws`, `sc_pctimp2019ws`, `sc_pcturbhi2019ws`, `sc_pctow2019ws`, `sc_fertws`,
  `sc_manurews`, `sc_naniws`, `sc_pctagdrainagews`, `sc_popden2010ws`, `sc_wwtpalldensws`, `sc_rddensws`,
  `sc_pctcarbresidws`, `sc_caows`, `sc_na2ows`, `sc_clayws`, `sc_sandws`, `sc_permws`, `sc_kffactws`, `sc_rckdepws`,
  `sc_bfiws`, `sc_runoffws`, `sc_precip9120ws`, `sc_tmean9120ws`, `sc_wetindexws`, `sc_elevws`, `sc_wsareasqkm`,
  `sc_pctforest2019ws`, `sc_pctwetland2019ws`.


## 5. GBM timing benchmark (V12)

Timing only. "No skill, no early-stopping loss and no held-out loss is computed. No forecast file and no registry row
is written." (`TR:54`) The rule that turns a time into a cap (`TR`, sha256 in §0) was written before the first timing:
"T(cap) = t_build(cap) + n_boosters × 2000 × s(cap)" (`TR:27`); "The first cap on the ladder with T(cap) ≤ 2 h,
evaluated for the more expensive arm (PF-GBM, which has the physics features) on the scored variable and horizon with
the most labelled fit rows. The cap is used by BOTH GBM arms, in step 1 and in CV." (`TR:33–35`) Script:
`sonde_v2_gbm_timing.py` (in the list of §6).

### 5.1 Deviations from the timing rule

- **Load.** "The benchmark starts only if the 1-minute load average is ≤ 36." (`TR:63`) The 1-minute load average of
  the shared server stayed above 36 all evening. Under ruling Q `DR:118` the benchmark was started above the limit,
  with the load in the record: 50.87 at the start and 62.43 at the end of the complete run. By that ruling such a
  record supports ONE conclusion only ("full rows on that device" if the PF-GBM unit at full rows is ≤ 2 h); it may
  not be used to choose a cap below full rows, to choose a device, or to support a change of the recipe.

- **Record file.** `TR` names `gbm_timing.json` as the record; that name is kept for a record taken at a load ≤ 36.
  The loaded record is `sentinel_runs/sonde/v2_smoke/gbm_timing_loaded.json`.

- **GPU index.** `TR:49` names GPUs 7, 8, 9. The script asks the lock helper for an idle GPU with ≥ 10,000 MiB free,
  preferring 7 (`sonde_v2_gbm_timing.py:211`); the helper returned GPU 0 (record field `gpu.gpu_index`). The
  engineer's `nvidia-smi` reading at 20:24 EST showed memory of other jobs on GPUs 7, 8 and 9; that reading is not a
  file on disk. The lead allowed any idle GPU through the lock helper in the message reproduced as Annex B of the GBM
  ruling (`GB:150`) and requires the disclosure of this point in GBM ruling G-8 `GB:108` (i).

- **Two starts.** The benchmark was started twice (5.2, 5.3). The first start stopped with an error before any record
  was written; its log is kept read-only and its five timings are reported below. The second start was ordered by the
  lead in the message reproduced as Annex A of the GBM ruling (`GB:136`), which also fixes this disclosure; GBM ruling
  G-8 `GB:108` (ii) repeats it.

### 5.2 First start (stopped with an error)

Log `sentinel_runs/sonde/v2_smoke/gbm_timing_loaded_crash1.log`, sha256
`6ec70cf28fa3f791f83c37f8bf83deb90e2bd39d32060ee7f203498503b9143f`, read-only. The CPU full-rows fits completed
(1,597,042 rows × 106 features in the log); then the script raised `KeyError: '500k'` in its projection step, which
ran after the first block and needed the 500,000-row block that had not been timed yet. No record file, no GPU timing.
The load at this start was not written to the log (the load is written to the record file only); the engineer's shell
showed a 1-minute load of about 60–62, which is not a file on disk. Fix: one line, the projection is computed only
when both blocks of a device exist (`sonde_v2_gbm_timing.py:112`). sha256 of the script after the fix
`193f0439f742eecf50c984028f6a72188168b74d116f0144dcaf591eb8814f50` (= the file in §6); before the fix
`72e40f3a7ba5bb5291395a33ba62ab4c662055028fde3fcab6ec16d13d1fcd01` (noted by the engineer when the fix was made; the
earlier file is not on disk). The fix changes no fit, no row and no timing.

| CPU, full rows: s/round | q05 | q50 | q95 | L1 | L2 | quantile mean |
|---|---:|---:|---:|---:|---:|---:|
| first start (stopped) | 1.102 | 1.199 | 0.828 | 0.961 | 0.803 | 1.043 |
| second start (complete) | 0.5798 | 0.7932 | 0.6451 | 0.6239 | 0.5995 | 0.6727 |
| first start slower by | 90 % | 51 % | 28 % | 54 % | 34 % | 55 % |

The two starts differ by more than 20 % in every booster; both are reported. With the first start's quantile mean the
Q19 unit at full rows on CPU is 19 × 2000 × 1.043 s = 11.0 h (binning time not in that log, a few seconds).

### 5.3 Second start (complete record)

Record `sentinel_runs/sonde/v2_smoke/gbm_timing_loaded.json`, sha256
`21807d51bb9c2c9193887755437733ba7bd8fa48455527f8338604c8948af4ec`; log `gbm_timing_loaded.log`, sha256
`ade5a57e6fc4cea721f630b0420cfecd188a435895218c1635e9d92d4e3ec000`; both read-only. Start 2026-11-11T01:08:43Z, end
2026-11-11T01:20:19Z; load average (1, 5, 15 min) at the start [50.87, 58.4, 60.42], at the end [62.43, 57.15, 57.86].
Unit timed: `temp_c_mean`, h = 1, PF-GBM features, 50 rounds per booster, 6 threads, max_bin 255, LightGBM 4.7.0;
1,024 fit sites (step-1 fit sites). Labelled fit rows at h = 1 (counts only): `temp_c_mean` 1,597,042; `do_mgl_mean`
407,758; `do_mgl_min` 446,132; `spcond_uscm_mean` 722,454; `ph_median` 258,836; `turb_fnu_mean` 129,743;
`turb_fnu_median` 132,941; `no3no2_mgl_mean` 9,086. Ratio of train+val-role to train-role split sites used for the CV
projection: 1.1876.

Seconds per boosting round of one booster:

| Device | fit rows | binning s | q05 | q50 | q95 | L1 | L2 | quantile mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU | 1,597,042 | 4.9 | 0.5798 | 0.7932 | 0.6451 | 0.6239 | 0.5995 | 0.6727 |
| CPU | 500,000 | 2.2 | 0.2193 | 0.3171 | 0.2077 | 0.2990 | 0.2028 | 0.2480 |
| GPU (GPU 0) | 1,597,042 | 4.6 | 0.2250 | 0.2083 | 0.1841 | 0.1958 | 0.2221 | 0.2058 |
| GPU (GPU 0) | 500,000 | 2.4 | 0.1011 | 0.1950 | 0.3474 | 0.4309 | 0.1608 | 0.2145 |

Projected hours at learning rate 0.05 and 2000 rounds for every booster (upper bound, no credit for early stopping;
`TR` §3, §8): one unit of the largest variable; all of step 1 for one arm (Q19 only); one CV fold of one arm (8
variables × 7 horizons, sequential):

| Device | configuration | cap (rows) | unit h | step 1 h | CV fold·arm h |
|---|---|---|---:|---:|---:|
| CPU | Q19 | full | 7.10 | 27.7 | 146.9 |
| CPU | Q19 | 1,500,000 | 6.71 | 26.5 | 135.6 |
| CPU | Q19 | 1,000,000 | 4.66 | 20.4 | 121.3 |
| CPU | Q19 | 500,000 | 2.62 | 14.3 | 95.9 |
| CPU | Q19 | 250,000 | 1.31 | 7.9 | 57.7 |
| CPU | Q | full | 2.62 | — | 54.2 |
| CPU | Q | 1,500,000 | 2.47 | — | 50.0 |
| CPU | Q | 1,000,000 | 1.72 | — | 44.7 |
| CPU | Q | 500,000 | 0.97 | — | 35.3 |
| CPU | Q | 250,000 | 0.48 | — | 21.3 |
| CPU | LS | full | 0.68 | — | 14.8 |
| CPU | LS | 1,500,000 | 0.65 | — | 13.8 |
| CPU | LS | 1,000,000 | 0.46 | — | 12.5 |
| CPU | LS | 500,000 | 0.28 | — | 10.2 |
| CPU | LS | 250,000 | 0.14 | — | 6.2 |
| GPU | Q19 | full | 2.17 | 12.1 | 81.9 |
| GPU | Q19 | 1,500,000 | 2.18 | 12.1 | 82.1 |
| GPU | Q19 | 1,000,000 | 2.22 | 12.2 | 82.4 |
| GPU | Q19 | 500,000 | 2.26 | 12.3 | 82.9 |
| GPU | Q19 | 250,000 | 1.13 | 6.8 | 49.9 |
| GPU | Q | full | 0.80 | — | 30.2 |
| GPU | Q | 1,500,000 | 0.80 | — | 30.3 |
| GPU | Q | 1,000,000 | 0.82 | — | 30.4 |
| GPU | Q | 500,000 | 0.83 | — | 30.6 |
| GPU | Q | 250,000 | 0.42 | — | 18.4 |
| GPU | LS | full | 0.23 | — | 11.0 |
| GPU | LS | 1,500,000 | 0.24 | — | 11.2 |
| GPU | LS | 1,000,000 | 0.29 | — | 11.5 |
| GPU | LS | 500,000 | 0.33 | — | 12.1 |
| GPU | LS | 250,000 | 0.17 | — | 7.3 |

GPU settings that differ from the CPU fit: `{"device_type": "gpu", "gpu_device_id": 0, "gpu_use_dp": false}`.
Synthetic equality check (`TR` §6): relative difference of the pinball loss GPU vs CPU α = 0.1: 3.68e-03, α = 0.5:
0.00e+00, α = 0.9: 2.71e-09; within 1 %: True. Relative difference of the training loss after 50 rounds on the fixed
sample of 200,000 fit rows, GPU vs CPU: full rows ≤ 8.1e-09; 500,000 rows ≤ 4.4e-03. The GPU timing at 500,000 rows is
not faster than at full rows (quantile mean 0.2145 vs 0.2058 s/round): the timings are noisy under this load.

### 5.4 What the loaded record gives

PF-GBM, Q19, full rows (1,597,042), largest variable: unit time 7.10 h on CPU with 6 threads and 2.17 h on GPU 0; with
the first start's CPU timings 11.0 h. Both are above 2 h, so the record does not give "full rows". By ruling Q
`DR:118` it is kept and marked `not_used_for_cap`: no cap, no device and no recipe change is taken from it. The lead
re-computed the same unit times from the record and ruled the same way: GBM ruling G-1 `GB:50`, "The record is kept
and is `not_used_for_cap`, as ruling Q says." (`GB:50`) The engineer took no number of this benchmark into any
setting; the settings of V12 and V12b are those of the ruling.

### 5.5 Why no second benchmark, and the lead's ruling

**No repeat of the benchmark** (GBM ruling G-4 `GB:72`). `TR` §4 (the cap ladder), `TR` §7 and the clause of ruling Q
`DR:118` that calls for a second record at a load ≤ 36 ("the benchmark is repeated at a load of 36 or less; only that
second record enters §4 of the timing rule" (`DR:127`)) are not applied, because their only purpose was to pick a cap
and GBM ruling G-2 `GB:52` sets full rows for every configuration: "The benchmark is not started a third time."
(`GB:74`) No record at a load ≤ 36 exists and none will be taken; `gbm_timing.json` was never written. This is the
disclosure that GBM ruling G-8 `GB:108` (iii) requires.

**The ruling in the code.** `docs/preregistration/sonde_arch_v2_gbm_lead_ruling_20261110.md` (sha256 in §0, 0444,
registry row `6cad67b32813`), entered as `GBM_SETTINGS` (`declared.py:66–80`): device `gpu` with `{"device_type":
"gpu", "gpu_use_dp": false, "max_bin": 255}`; fit rows Q19: full rows; Q: full rows; LS: full rows; learning rate
0.05, ≤ 2000 rounds, early stopping 100, 63 leaves, ≥ 200 rows per leaf, feature fraction 0.8, bagging 0.8, L2 1.0,
max_bin 255, 6 threads, seed 0, early-stopping rows ≤ 400,000; resume {"process_limit_min": 115,
"first_booster_allowance_min": 40, "factor": 1.5, "booster_limit_h": 2, "abort_min": 119, "predict_latest_min": 105};
CV limit 300 GPU-h. The test that compares these values with the frozen `GBMConfig` and with the sha256 of the ruling
on disk is `test_sonde_v2.py:657`. V12 and V12b say what the code does with them; GBM ruling G-8 `GB:108` (iv) and (v)
are disclosed there.

**Projected GBM compute** (GBM ruling G-7 `GB:90`, upper bounds of the record at full rows on the GPU, 2000 rounds for
every booster, no credit for early stopping, as re-computed by the lead): "12.1 GPU-h for step 1 (Q19) and, per CV
fold and arm, 81.9 GPU-h for Q19, 30.2 for Q and 11.0 for LS; with 5 folds and 2 arms that is about 819 GPU-h for Q19
alone" (`GB:91–93`). The real cost depends on the stopping rounds, which no one knows before the first fit; the rule
that follows from it (GPU-hours per process in the registry, the CV projection before the first CV GBM process, and
the 300 GPU-h gate) is implemented as described in V12b, and "if that projection, or later the running CV total plus
the projected remainder, is above 300 GPU-h for the declared GBM configurations of both arms together, no further CV
GBM process starts until the lead has ruled." (`GB:97–99`) GBM ruling G-7 `GB:90` changes no declared configuration
list: Q19 stays mandatory and Q and LS stay the additional declared configurations of both arms (V12).


## 6. Code list (ruling O)

sha256 of every file matching `sentinel/models/sonde_v2/*.py`, `scripts/train/sonde_v2_*.py`,
`tests/test_sonde_v2*.py` at the moment this file was generated (rulings O `DR:73`, §2 O `DR:94`). The unit tests of
`tests/test_sonde_v2.py` passed on exactly these files before the list was written. From here on none of these files
is edited; the stamp guard (V20) closes every driver phase if one of them changes or a new file matches the globs
without being listed in a later stamped file `sonde_arch_v2_deviations_v2*.md`.

<!-- CODE-SHA256-BEGIN -->
```
a436f85ff9a0bf6689bc65ae312f0581bb45e1b302efbd320fd147751a19b257  scripts/train/sonde_v2_counts.py
a0b6228cb00ac0d200eeccef54f99266fb1ef635a7af4239aaef331f8e94b565  scripts/train/sonde_v2_cv.py
193f0439f742eecf50c984028f6a72188168b74d116f0144dcaf591eb8814f50  scripts/train/sonde_v2_gbm_timing.py
bd885d4761a3a89939e4df27ee21d6b6fe84a636e0f7a5fc202e5cde06c3a95f  scripts/train/sonde_v2_smoke.py
fe6e162386be0e7d22d7e43fa2092bc35e0dc09762a0bbe3e9e783738d56e278  scripts/train/sonde_v2_step1.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  sentinel/models/sonde_v2/__init__.py
cbf65f8ddfc31ca778a64e05af49ab3bebfad2d07a890028661a7d4dfd940574  sentinel/models/sonde_v2/chronos_cov.py
ddc54ccfcb28e9c789a249902b0dbe57d929336b9ad5900d5f4b9d55fc1c0919  sentinel/models/sonde_v2/data_v2.py
b2079072b81b995d4ac4e1c6b7c43ea48b702c24a810d1611d3f8ad5c038d3df  sentinel/models/sonde_v2/declared.py
83963935baaf03d80d5a8669d9e9abbc3d255e973075b31250a32bde2fcdd497  sentinel/models/sonde_v2/ensemble.py
12f182f85e9706a5804ab074c1b1001b04740eab959bc7db7722a986329b9f12  sentinel/models/sonde_v2/gbm_nwp.py
392d03da0d447d8b46d19b45484c75958ff65a4694b114bdc3da59220c1f9c91  sentinel/models/sonde_v2/gbm_resume.py
a76fd98bbf624d833ab1b58cf9703ac9b87d28e79328b2c384107cdf9036a58b  sentinel/models/sonde_v2/hourly.py
15e3bb7c12e6eed85481d54108b827ab822c6826e88d3094f01e0352d5190254  sentinel/models/sonde_v2/nets.py
922881b06b2477ee1a2494c524b72366f0db5aa5bf8d49d4f386bc7bbe8d3834  sentinel/models/sonde_v2/nwp.py
21c84591183c4624560cd546a9ae693e260e597dadc33505cb1eb781feffb8fb  sentinel/models/sonde_v2/parity.py
e46ca5116af51a49826bdfe44ece765369cebdfe99367e3b0345d0429d34b36c  sentinel/models/sonde_v2/physics_features.py
8f0994a5b1af55b823ecc9109320be6ecb6d104c469740b39ab74590d25877e9  sentinel/models/sonde_v2/prep.py
d881f63e9df252e5f8bc2727869573c3509218fb8d9c313d39ec137e69b25d5f  sentinel/models/sonde_v2/products.py
8f0a5dca1ffdb62440440282af395c3cc0598e09ea212bb9a4c2e0e517dff74f  sentinel/models/sonde_v2/train_v2.py
b10353bcaf41243da79c30041a7be05f0030febb2f93ec0f882e2e270fc33fb8  tests/test_sonde_v2.py
```
<!-- CODE-SHA256-END -->

`code_sha256` = `f4de13a0d71965c07946638c2319a2db2365ef89510cf1f2a3126ff04b015526` = sha256 of the UTF-8 text of the
21 lines above (`<sha256>`, two spaces, `<path>`; lines joined by a single line feed, no line feed after the last
line; `parity.py:77`).


## 7. Stamp row (ruling R)

This file is stamped by ONE registry row written by the lead (ruling R `DR:131`): workstream `M1-sonde`, experiment
`sonde/prereg_snapshot`, with `extra.written_by = "lead"`; `extra.sha256` = a mapping {repo-relative path: sha256}
that contains at least this file, `docs/preregistration/sonde_arch_v2_dev_lead_rulings.md`,
`docs/preregistration/sonde_arch_v2_gbm_lead_ruling_20261110.md` and the lead's rulings file on this file, and —
because the guard reads them — EVERY file on disk that matches `docs/preregistration/sonde_arch_v2_deviations_v2*.md`,
each read-only; and `extra.code_sha256` = the value of §6. The guard accepts nothing else (`parity.py:141–143`). The
sha256 of this file cannot be written inside it; the engineer sends it to the lead with the path.

Order after this file (ruling S `DR:137` and GBM ruling G-9 `GB:113`): independent delta review, asked to check G-2 to
G-7 against the code → stamp → step 1 (`--phase declare`, then `--phase hf` and `--phase gbm`, which may run side by
side on idle GPUs, `--phase gbm` repeated while it returns 75 (V12b), then `--phase eval`). Before the first CV GBM
process: `sonde_v2_cv.py --phase gbm_budget` and the projection to the lead (GBM ruling G-7 `GB:90` (c)). Until the
stamp row exists no step-1 and no validation number is produced.
