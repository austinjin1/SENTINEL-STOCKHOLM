# SENTINEL-Hydro (M6) — architecture round v2 amendment: tracks, arms, references, gates and claim rules

**Status:** DRAFT v0.1 (2026-11-10 12:57 EST), written by the M6 architecture-v2 engineer for the lead's review and
stamp. AI-assisted.
**Stamp:** pending. The lead records this file's SHA-256 in a `<ws>/prereg_snapshot` registry row and commits it.
**Implements:** `docs/plans/architecture_research/LEAD_DECISIONS_20261110.md` rules 1–6 and "M6 HydroMet: rulings
2026-11-10 12:26 EST"; plan `docs/plans/architecture_research/M6_architecture_plan.md`. Where the plan and the rulings
differ, the rulings win. Where this file is stricter than either, this file applies.
**Relation to frozen files.** This is a new file. It edits no frozen file: `hydromet_protocol.md`,
`FREEZE_hydromet.json`, the 46 hashed M6 files (code hash `5bd7d3292a29…fd60b`), `hydromet_sota_protocol.md` and
`hydromet_ea_protocol.md` are unchanged. All v1 results stay reported. Display name SENTINEL-Hydro; code identifiers,
paths and registry keys keep `hydromet`.
**Bars.** Nothing here lowers a bar. RESEARCH_PLAN §15.1 and `sota_campaign.md` rules 1–8 apply unchanged: a SOTA
label needs a paired CI excluding 0 against both the strongest published method and the strongest simple baseline,
Holm correction, 3 seeds with the every-seed rule, and one look at an unseen frozen test block.

Tags: [RV] = number taken from the plan, re-verified there. [M] = counted by me today from metadata or row counts only.
**UNVERIFIED** = not yet known; listed again in §13.

## 0. What exists at the time of writing (and what does not)
- Frozen M6 code hash, checked before any write of mine: `5bd7d3292a29aa9beb46cd1da5e480ca58b3856eeb909dd7fd43b0e3561fd60b`
  (`sentinel.models.hydromet.guard.scoring_code_sha256()`).
- Registry (`results/registry.jsonl`) at 12:50 EST: **no row** with an experiment name starting `M6SOTA`, `M6V2`,
  `M6EA` or `M6/test_b` [M]. `sentinel_runs/hydromet/configs_tried.jsonl`: 95 rows, one lever row (`sota_C2a`), **no
  architecture-v2 row** [M].
- No architecture-v2 code, config, test, data table, run directory or log exists: `sentinel/hydromet_sota/{gso_mt,
  thermox,stack}/`, `configs/hydromet_sota/v2/`, `sentinel_runs/hydromet/sota_v2/` are absent [M].
- EA England: the lever engineer's build is running (catalogue done, readings downloading; no value dated
  ≥ 2023-10-01 requested). `hydromet_ea_protocol.md` is DRAFT v0.1; `hydromet_ea_split.json` and M1's
  `sonde_ea_split.json` do not exist. I have read EA station metadata only (no reading).
- I have read no validation-block, test-block or EA value and computed no metric of any kind on any new arm, reference
  or track. Lever results: none read by me.
- **Guarantee until the stamp.** Only code, tests, data builds (protected rows: counts only) and smoke tests on
  synthetic data or train-role sites × train block are run. No validation, test or EA metric is computed, logged or
  printed for any arm, reference or track named here.

## 1. Practical reasons (stated before any result)
- **Why new arms at all.** On `M6/test` the EA-LSTM ties gradient-boosted trees; on `M6N/test` it is worse on
  temperature [RV]. The plan's diagnosis is that the binding limits are the number of independent training sites and
  test power, not model capacity. The arms below add independent sites (grab samples used at their sampling hour) and
  process structure, and the references get exactly the same additions.
- **D1 GSO-MT.** Most US reaches with any measurement have a few instantaneous grab samples, not a sensor. A model that
  can learn from a sample taken at 10:00 without treating it as a daily mean uses one to two orders of magnitude more
  reaches. Use: daily mean, maximum, minimum and 7-DADMax at unsensed reaches.
- **D2 ThermOx-T.** Heatwaves and a new country are outside the training range of forcings and attributes. A bounded
  energy-balance structure may extrapolate where a free sequence model does not. Use: thermal-stress screening.
- **D3 stack.** A cheap final-candidate option; its control stack shows whether the new arms add anything to it.
- **T-E (England, new geography):** a deployable unsensed-reach estimator has to transfer to another country's
  network, forcing product and regulator (`hydromet_ea_protocol.md` §1).
- **T-B / N2 (US future year WY2026):** the operational product always runs on years after development.
- **T-H (forcing-defined heatwave windows):** temperature standards for coldwater fish are set on 7-DADMax, and the
  days that matter are heatwave days. The stratum is defined from forcing only, never from the outcome.
- **T-D (historical logger reaches):** regulators assess 7-DADMax at reaches without a USGS gauge. Descriptive only.

## 2. Data: sources, role rules, data hygiene, pool pin and locks
### 2.1 Training-label sources (all models, identical rows)
| Code | Source | Rows used for fitting |
|---|---|---|
| L-sonde | FINAL split (sha `fd4bb14f…6a80`) train-role sonde sites × train block WY2008–2021 | daily mean/7-DADMax/DO mean/DO min; auxiliary daily T min, T max, DO max, discharge; hourly tables (749 sites [RV]) |
| L-usgs | USGS daily-value sites outside the FINAL split: the lever engineer's pinned `lever_sites.parquet` (sha `62c4bf53…`), rule = `fit` only (445 sites [RV]) | daily series as L-sonde |
| L-grab | WQP discrete samples WY2008–2021 (`processed/wqp_discrete`, temp and DO), **instantaneous rows with local solar hour** | value at hour h |
| L-logger | WQP non-USGS high-frequency rows aggregated to station-days (exact row rule fixed in the build script, pinned) | daily mean/min/max |
| L-oliver | Oliver et al. 2024 (doi:10.5066/P9EMWZ35, CC0), stations that are neither NWIS nor WQP | daily mean/min/max temperature |
| priors | Powell hydraulic-geometry coefficients at the 160 train-role Powell sites [RV] (D2 priors only; depth and velocity are not labels) | none |

Not used in this round: StreamPULSE (M1 locks an M1-SP block; avoided entirely), Powell val/test-role sites (37 / 117,
reserved for Sim H-G3), any Caravan/CAMELS data, any lake row (M2 Test-S is in-lake; no overlap is expected and any
overlap is reported as a count).

### 2.2 Grab-sample pool (two tiers; fixed by counts only)
Forcings and statics exist on disk only for the 12,771 track-N stations (stations active in WY2022+). The plan listed
the grab archive as "0 new data"; that holds for the labels, not for the inputs.
- **Tier 1 (inputs on disk).** Track-N-set stations with the lever rule `fit` (3,545 stations in 32 fit groups;
  256,628 temp / 230,992 DO station-days [RV]) for fitting, and those with rule `inner_group` (358 [M]) as the inner
  grab set. Samples are re-extracted from `wqp_discrete` with their time (the lever table holds daily means only).
- **Tier 2 (inputs must be built).** Other stream stations with ≥ 8 train-block sample-days of temperature or DO
  (46,598 / 41,387 candidates before role rules [RV]). Admission is deterministic and uses counts only: stations pass
  the role rules of §2.3, are ranked by train-block sample-day count (temp + DO, descending; ties by station key), and
  the first **N₂ ≤ 10,000** are admitted. Inputs are built with the track-N method (COMID by point-in-catchment,
  StreamCat by COMID, site-cell gridMET, upstream-basin forcing). Cost: ≈ 0.31 GB per 1,000 stations kept
  (≈ 3.1 GB at the cap), transient gridMET yearly files deleted after each pass. **UNVERIFIED:** build time, the
  number that pass the role rules, and how many get a complete input set. A station without a complete input set is
  dropped and counted.
- Filters as track N (`hydromet_protocol.md` §12): stream class, CONUS, unit match, accepted status, uncensored,
  discrete activity types, depth NaN or ≤ 2 m. Local solar hour h = (UTC hour + longitude/15) mod 24 (mean solar time;
  the equation of time, ≤ 16 min, is ignored). A sample without a time gets the uniform mixture over 08–17 h.
- Provider class ∈ {USGS, state, tribal, local, volunteer, other} from a fixed mapping of WQP provider, organisation
  type and tribal flag, written in the build script and pinned.

### 2.3 Role rules for every new location (lever rules, imported unchanged)
`sentinel.hydromet_sota.sites.apply_rules`, first match wins: reserved (729 M1 sites) → `reserved_comid` → in the FINAL
split → `new_region` → owning basin group is val- or test-role → owning group is one of the 20 inner groups (256 inner
sonde sites; fixed by `inner_frac` 0.15, seed 0, identical to the reference run) → flow-connected to a protected site
(NHDPlus V2 main path ≤ 500 km) → `fit`. In addition a WQP station with a track-N role of test, val or excluded
(roles sha `ffb41471…e2d0`) is never a fitting or inner station.

### 2.4 Data-hygiene rules (plan §6, mandatory)
1. Every new label table passes `sentinel.data.reserved.assert_no_reserved` and
   `sentinel.hydromet_sota.reserved.assert_no_reserved` at build time and again in every training loader.
2. No row from a FINAL val- or test-role group, a track-N test-, val- or new-region station, or a spent cell
   (test-role × WY2024–2025) enters fitting or selection. Training and selection loaders read no US row dated after
   2021-09-30 (row filter at read time). Validation-block rows (WY2022–2023, val-role only) are read only by the one
   validation go/no-go per track, after the stamp. No US row dated ≥ 2023-10-01 is read by arch-v2 code before a look.
3. A grab, logger or Oliver station whose COMID holds a FINAL-split site or a reserved site is dropped.
4. Role-assignment code and every label, input and role table are pinned by sha256 in
   `configs/hydromet_sota/v2/pins.json` **before the first post-stamp fit**; loaders refuse unpinned tables. The pin
   file's sha256 is reported to the lead and written in every run record.
5. Protected rows: counts only (stations, station-days, groups). No value, mean, range or plot.
- **One pool for all models.** The pinned pool is the training pool of every arm and reference in this round. A
  source that is not built and pinned before the first post-stamp fit (Tier 2, L-logger, L-oliver are the candidates)
  is **not** training data for any model in this round, unless a stamped deviation adds it for all models before any
  model has a validation number.
- Run records never contain a reserved site number (`sentinel.data.reserved.preclaim_audit` must stay clean).

### 2.5 Lock-at-extraction rules
- **WQP rows at protected stations:** station roles are computed from station metadata first; values are read only for
  `fit` and inner stations. Protected stations appear in my tables as counts.
- **Oliver et al. 2024:** HEAD size check first (size **UNVERIFIED**); stream-extract-delete inside the 20 GB cap and
  the 60 GB floor. The extractor assigns roles from station coordinates before it keeps any value. Rows of stations in
  protected groups go directly to a separate locked table (0440, outside every training path); only counts are logged.
  That table is the T-D block and is opened only by the T-D scoring step named in a FREEZE file.
- **WQP logger rows at protected stations** (the other T-D source) stay unread in `wqp_discrete` until that step.
- **EA:** I consume the lever engineer's build through one loader that drops every row dated ≥ 2023-10-01, every
  M6-EA test-role group (all years) and, once `sonde_ea_split.json` exists, every M1-test group. I download no EA data.
- **Sonde auxiliary labels and hourly tables** are loaded for train-role sites × train block only.

## 3. Inputs
- **US input set v2** (tracks T-B, N2, T-D and all inner criteria): the 26 dynamic channels and the frozen static set
  of the `ealstm_ws` reference; plus 3 NHDPlus VAA statics (`pathtimema`, `streamorde`, `lengthkm`); plus 4
  solar-geometry channels computed from latitude, day of year, elevation and `srad` (photoperiod, clear-sky radiation
  by the FAO-56 formula, cloud ratio = srad / clear-sky clipped to [0, 1.2], noon solar elevation). Every reference
  gets the same additions (as features for the GBMs). No in-situ input of any kind.
- **T-E common schema:** exactly `hydromet_ea_protocol.md` §3 (11 site-cell forcing channels by a fixed variable
  mapping, gridMET in the US and ERA5-Land in England; 39 RiverATLAS statics; calendar), plus deterministic functions
  of schema inputs only (the 4 solar channels; trailing means). **Every** model scored on T-E — claim model, R-TD, G1,
  G2, P and any lever model — is trained on and fed this schema and nothing else: no upstream-basin channel, no
  StreamCat or VAA attribute, no US-only input. In zero-shot mode no transformation is fitted on EA data (input and
  target standardisation come from US train-block data). A component that needs a non-schema input takes a fixed
  schema-derived substitute written in `configs/hydromet_sota/v2/schema_variant.yaml` before any EA number (D2: travel
  time becomes a bounded static parameter; the upstream source uses the site-cell equilibrium temperature).

## 4. Configuration budgets and counting (as ruled)
| Budget | Items | Cap |
|---|---|---|
| **Architecture v2 (neural networks)** | D1 grid 3 + 2 ablations; D2 temperature stage grid 3 + 2 ablations; D2 DO stage ≤ 1 (gated); R-TD 1 fixed | **≤ 12** |
| GBM references | G1: 3; G2: 3 | 3 per variant |
| **Levers** (lever engineer; `hydromet_sota_protocol.md` §4) | C1a, C1b, C2a, C2b, C12 (gated), lever 4 (≤ 3, gated) | **≤ 8** |

- Lever 3 (pretraining) is deferred and its 3 slots are retired, not transferred. Row A* of the lever protocol is
  replaced by the architecture budget. **Disclosed change:** the M6 total rises from ≤ 12 to ≤ 20.
- R-EA (frozen `ealstm_ws_full` recipe on the enlarged data) is optional, descriptive, outside the cap, last in queue.
- Every configuration is appended to `sentinel_runs/hydromet/configs_tried.jsonl` before it trains, failed runs
  included. Smoke runs are logged with `smoke: true` and not counted (v1 practice).
- **Counting rule (proposed; see §14 Q1).** A configuration is (arm, architecture hyperparameters, loss weights and
  grab share, ablation switch). Seeds are not configurations. Re-training an already selected configuration with
  unchanged hyperparameters (a) on the label set that gate G-D1a determines, (b) on the T-E common schema, or (c) in
  the EA fine-tuning cell is a **variant**: logged with `variant_of`, always reported, not counted. No hyperparameter is
  ever re-selected in a variant.

## 5. References and parity (built before or with the candidates)
All references get the same pinned label pool, auxiliary targets, input set, inner rows and 3-seed ensembling as the
candidates. They are fitted before or together with D1 and never after a candidate's validation number exists.
- **R-TD (strongest published method, re-run here).** The published top-down LSTM for unmonitored reaches (Rahmani et
  al. 2021; Willard et al. 2024; MHPI `hydroDL` style): one cuDNN LSTM layer over 365-day windows with the standardised
  static attributes concatenated to the dynamic channels at every step, independent Gaussian heads per daily target
  (mean, max, min, 7-DADMax, as in the published multi-target form), and the same auxiliary targets and grab-sample
  observation operator as D1 (§6.1; amplitude and phase heads feed the operator for grab and hourly rows only), so
  that it uses the same rows. One
  fixed recipe, no grid: hidden 256, dropout 0.5 [**UNVERIFIED** against the papers; transcription-checked before
  the first fit, any difference logged in the deviations file, still one configuration]. It gets D1's optimiser,
  schedule, sampling, static-noise augmentation, early stopping, refit and seeds, so that a difference is not due to
  training protocol.
- **G1 (simple baseline).** `forcing_gbm_bag` (LightGBM, 76 causal forcing features + statics + calendar), retrained on
  the pool, bagged over seeds 0–2, with the grab-sample formulation below.
- **G2 PF-GBM (strongest simple baseline).** G1 plus physics features (prior-parameter equilibrium temperature Te and
  its trailing 1–30-day means, DO saturation at Te, cloud ratio, photoperiod, the 3 VAA statics) plus out-of-fold
  stacked predictions (K = 5 folds by basin group; full-fit predictions elsewhere) of log discharge, T min, T max and DO
  max. In the T-E schema the VAA features are absent for every model.
- **GBM grab-sample formulation (mandatory parity).** For v ∈ {temperature, DO} one instantaneous model is trained on
  rows (station, date, hour h): every grab row that D1 uses; sonde hourly rows at fixed hours {1, 5, 9, 13, 17, 21};
  and every daily-mean row (L-sonde, L-usgs, L-logger, L-oliver) with a `kind = daily mean` flag and missing hour
  features. Features = the usual daily features + first and second hour harmonics and their products with tmax, tmin
  and srad + the kind flag. Target = the instantaneous (or daily-mean) value. Outputs: hourly prediction at h;
  daily mean = average of the 24 hourly predictions; daily max and min = daily mean ± an amplitude model trained on
  sonde min/max (G2: with the stacked features); 7-DADMax = trailing 7-day mean of the daily max; DO min from the
  amplitude model. The v1-style direct daily-target models are also refitted on the enlarged daily rows. **For each
  primary target the reference uses whichever of {derived, direct} has the lower inner MSE** (chosen on inner rows
  before any validation number; this can only strengthen the reference and is reported).
- **GBM tuning parity.** 3-point grid per variant, mirroring D1's grid: (`max_rows_per_site`, grab weight share) ∈
  {(3000, 0.5), (3000, 0.25), (1000, 0.5)}; selected per target on the same inner rows; refit on fit + inner with the
  selected iteration count (as lever protocol §5).
- **P (physics family).** `air2stream_regional`, `air2stream_regional_ws` and `dosat_ratio[…]` refitted on the pool's
  daily rows (donors with ≥ 365 train-block days and NSE ≥ 0.5); the member per endpoint is chosen on inner rows as in
  `hydromet_protocol.md` §6. In the T-E schema only non-`ws` members exist.

## 6. Candidates
### 6.1 D1 GSO-MT (rank 1)
- **Model.** Static encoder MLP [194 → 128 → 32] (GELU, LayerNorm; static noise σ = 0.5, attribute dropout 0.1); one
  cuDNN LSTM layer (hidden per grid) over 365-day windows of the dynamic channels, with FiLM of the static embedding on
  the input projection; loss on the last 90 days. Daily heads on [h_t ⊕ z_s] for v ∈ {T, DO}: mean μ_v, log diel
  amplitude, phase (sin, cos), skew weight κ_v ∈ [0, 0.5], σ_v (softplus). Diel curve on a 24-point grid:
  g_v(h) = A_v[cos(2π(h − φ_v)/24) + κ_v cos(4π(h − φ_v)/24)], centred to zero daily mean. Daily max/min = μ_v +
  max/min g_v; 7-DADMax = trailing 7-day mean of the daily max, in-model. Auxiliary heads: log discharge (μ, σ;
  weight 0.5); specific conductance, turbidity, nitrate (weight 0.25; never used for stopping or selection).
- **Observation model** (masked Gaussian NLL, target-balanced): sonde, USGS, logger and Oliver daily statistics;
  sonde hourly rows as μ_v + g_v(h); grab rows as μ_v + g_v(h) with variance σ_v² + σ_src², σ_src learned per provider
  class with a half-normal prior; rows without a time as the 08–17 h mixture.
- **Sampling.** A share s of each batch is grab/logger-station windows (grid), the rest dense-site windows; station
  weight ∝ √(observations), capped.
- **Grid (3; seed 0):** (hidden 192, s = 0.5); (192, 0.25); (128, 0.5).
- **Ablations (2):** (a) no grab, logger or Oliver stations (L-sonde + L-usgs only); (b) diel heads replaced by
  independent heads for mean, max, min and 7-DADMax (grab rows then use the mean head).

### 6.2 D2 ThermOx-T, temperature stage (rank 2; DO stage gated)
- **Model.** Static MLP [194 → 128 → 64 → 32] (noise σ = 0.5) → about 13 bounded static parameters with biases at
  prior medians (reference depth and its flow exponent; maximum shade, leaf-on and leaf-off day; wind sheltering;
  groundwater share, time scale and offset; reservoir share and time scale; upstream share; exchange scaling); cuDNN
  GRU (hidden 64) giving daily latents (log relative discharge, supervised with weight λ_Q; snowmelt share;
  groundwater modifier; residual r_t ∈ [−c, c]). Physics in fp32: surface energy balance solved by 4 vectorised Newton
  steps for equilibrium temperature Te (clamped to [−5, 45] °C) and the exchange coefficient; source mixing
  (atmosphere, groundwater kernel, reservoir kernel, upstream Te lagged by travel time, melt at 0 °C); exact
  exponential integrator for the daily mean; hourly linearised diel sub-model on loss days (2 periodic cycles × 24
  steps) giving daily max, min and the grab operator; 7-DADMax from the trailing mean of the hourly maximum. Output
  μ = physics + r_t, with σ heads. Hydraulic-geometry coefficients at the 160 train-role Powell sites enter as priors
  on the depth parameters.
- **Grid (3; seed 0):** {c = 0, λ_Q = 1}; {c = 1 °C, λ_Q = 1}; {c = 1 °C, λ_Q = 0.3}. Warm-up: residual frozen at 0
  for 3 epochs. The c = 0 point is always reported.
- **Ablations (2):** no diel sub-model; no source mixing (atmosphere only).
- **DO stage (≤ 1 configuration, only if G-DO passes):** oxygen balance on top of the temperature stage (saturation,
  reaeration, production and respiration terms); K600 from Powell rows of H confidence only, weight 0.1. The Raymond
  et al. 2012 reaeration coefficients are **UNVERIFIED** and must be transcription-checked before any DO-stage fit.
- A temperature-only D2 (G-DO failed) is a secondary arm on the temperature and heatwave endpoints and a member of
  D3 for those endpoints. It is not eligible as a claim model.

### 6.3 D3 SENTINEL-Stack (rank 3) and its control
- Non-negative convex stack per primary endpoint of {D1, D2, R-TD, G2}. Weights are fitted on inner-group predictions
  of the early-stopped runs and applied to the refitted members. No validation row is used.
- **Mandatory control:** a references-only stack {R-TD, G1, G2}, built by the same code.
- D3's inner criterion is computed by leave-one-inner-group-out cross-fitting of the weights (20 folds).
- Disclosure: D3 contains comparator models. When D3 is a track's claim model, the references-only stack joins that
  track's comparator set (§10), so a win means "the new arms add skill to a stack of the references".

## 7. Training protocol, selection criterion, seeds and refit
- **Common recipe (D1, D2, R-TD):** AdamW, lr 5e-4, weight decay 1e-4, batch 256, 200 steps per epoch, ≤ 60 epochs,
  patience 6, minimum improvement 1e-4, gradient clipping 1.0; per-epoch checkpoint and `--resume`; ≤ 110 min per job.
- **Inner rows (never fitted in an early-stopped run):** the 256 inner sonde sites in the 20 inner groups, fixed inner
  windows tiling the train block (as the lever `fit`); and the inner grab set (grab stations with rule `inner_group`).
- **Criterion (disclosed changes).**
  - J_sonde = mean over the **4 primary targets** (`temp_c_mean`, `temp_c_7dadm`, `do_mgl_mean`, `do_mgl_min`) of the
    standardised point MSE on the inner sonde sites. This is the lever protocol's `inner_primary_mse`, so arms, levers
    and the v1 reference are comparable on it. *Change from v1 (which used all targets): approved by the lead for
    arch-v2 arms only; v1 models stay as frozen.*
  - J_grab = mean over {temperature, DO} of the standardised point MSE of the hour-resolved prediction at the inner
    grab set (per-station weight capped at 30 sample-days).
  - **J = 0.8 · J_sonde + 0.2 · J_grab** is used for early stopping and for selecting the grid point within D1, D2 and
    the GBM grids. *Change from the plan, which proposed a grab-sample NLL term: a point MSE can be computed for every
    model class (GBM, physics), an NLL cannot.* Ablation (a) of D1 and any no-grab variant use J_sonde alone.
- **Selection.** Lowest J among the grid points at seed 0; ties within 1e-4 go to the earlier grid point. Ablations
  and variants reuse the selected point's hyperparameters.
- **Seeds and refit.** Seeds 0–2 of the selected point are early-stopped (fixing E_k = best epoch + 1), then refitted
  on fit + inner rows for E_k epochs with no validation. The evaluated model is the 3-seed ensemble of the refits. The
  GBMs are refitted on fit + inner rows with the selected iteration count and bagged over seeds 0–2.
- Inner numbers of every run go to the registry as `M6V2/inner_v1` (workstream `M6-hydromet`), with seed-level rows.

## 8. Gates and kill rules (all fixed here; every outcome is reported)
- **D2 smoke kill rule (train-only data, ≤ 50 train-role sites, before any D2 grid run).** Step time and VRAM are measured
  at batch 256 on one 11 GB GPU, with finite-gradient assertions. If the step time is > 0.5 s, the diel sub-model is
  restricted to loss days of June–September; if it is still > 0.5 s, D2 is dropped. The measurement is reported.
- **G-D1a (do grab samples help?).** Compare the selected D1 point with ablation (a), both as the mean over 3
  early-stopped seeds. Grab, logger and Oliver rows stay in the claim configurations of **all** models only if they
  lower the criterion by ≥ 3 %: for the sonde tracks (T-B, T-E, T-H) on J_sonde; for N2, whose endpoints are grab
  samples, on J_grab (proposed clause; §14 Q2). If the gate fails for a track, every model's claim configuration for
  that track is its no-grab variant (same hyperparameters; R-TD, G1, G2 and D2 included), and this is reported. The
  seed SD of the reference's J_sonde is ≈ 5 % [RV]; the gate is a decision rule, not an inference.
- **G-DO (D2 DO stage).** The DO stage is trained only if, on 3-seed means, the temperature stage's inner MSE on the
  two temperature targets is within 2 % of D1's **and** its inner-site heatwave-window RMSE (mean of the median
  per-site RMSE of `temp_c_mean` and `temp_c_7dadm`; windows as §9.3 inside the train block) is ≤ D1's.
- **G-D1b and its analogue for D2 and D3 (one validation go/no-go per track).** Each track's validation cell is
  scored **once**, for all arms and references together. A candidate can be frozen for a track only if its point Δ
  against R-TD and against each of G1 and G2 is ≥ 0 on at least 3 of the 4 primary endpoints of that cell (N2: on both
  endpoints). Otherwise it is reported and not frozen for that track.
- A killed or gated-out arm keeps its registry rows and its place in every table.

## 9. Tracks
Common to all sonde-type cells: a site is scored for a target if it has ≥ 60 approved (EA: valid) days in the cell
after a 42-day embargo, and the cell is restricted to sites finite for every compared model. Primary metric: median
per-site KGE. Δ = claim model − comparator on the sites scored by both. Paired bootstrap over **basin groups**
(B = 2,000, seed 0), 95 % CI and two-sided p.

**Minimum groups (all tracks).** For an endpoint's paired cell let G be the number of basin groups and
G_eff = (Σ n_g)² / Σ n_g² (n_g = scored sites in group g). **An endpoint is a claim endpoint only if G ≥ 15 and
G_eff ≥ 8; otherwise it is descriptive.** The check is made twice: from metadata before the freeze (stations expected
in the block), and on the realised cell inside the look. An endpoint that fails stays in its Holm family with p = 1
(the family is never shrunk). Basis: the weakest existing M6 claim cells have 19 groups (Test-B DO [RV]); all FINAL
test-role sites give G = 42, G_eff = 14.1 [M]. Below about half of that the group bootstrap is unreliable and the
detectable effect exceeds ≈ 1.3× the US value. The v1 rule "fewer than 3 groups → p = 1" stays in force.

### 9.1 T-E — EA England, new geography (primary opportunity)
- **Data, eligibility, groups, roles, blocks, strata, single-look requirements:** `hydromet_ea_protocol.md` (lever
  engineer). This file fixes the arms, the claim rule, the families and the minimum groups.
- **Two tracks, two families.** *Zero-shot:* models fitted on US data only, in the common schema. *Fine-tuned:* each
  model continues from its zero-shot state on EA train-role groups (EA-train and EA-val blocks), with early stopping
  on EA val-role groups; neural arms use one fixed recipe (all weights, lr 1e-4, ≤ 20 epochs, patience 4, batches half
  EA and half US), GBMs and P as the EA protocol §4 states. Fine-tuned runs are variants (§4).
- **Unseen test block:** EA test-role groups × 2023-10-01 … 2026-09-30, never downloaded before the look.
- **Validation cell (one go/no-go per mode):** EA val-role groups × EA-val block. No EA fit or validation number is
  computed before `hydromet_ea_split.json` is frozen **and** M1's `sonde_ea_split.json` exists or the lead has ruled
  on it (M1-test groups are excluded from every M6 EA fit and validation cell).
- **Endpoints (6 per family):** the 4 primary endpoints + 2 heatwave endpoints (§9.3).
- **Near-discharge stratum:** as EA protocol §5 (public UWWTR table only); reported for every model; descriptive.
- **Power (metadata only; assumes US-like between-group variance — UNVERIFIED and probably optimistic for
  zero-shot).** MDE ≈ MDE₀ · √(14 / G_eff), MDE₀ = 0.014 / 0.022 / 0.07 / 0.06 KGE (temperature mean, 7-DADMax, DO
  mean, DO min; 80 % power, α = 0.05 [RV]); 14 is the G_eff of all FINAL test-role sites [M], used for every endpoint
  (conservative for the DO cells); × 1.24 for the worst Holm-6 step. At G_eff = 8: 0.019 / 0.029 / 0.09 / 0.08
  (0.023 / 0.036 / 0.12 / 0.10 with Holm-6); heatwave RMSE ≈ 0.6 °C.
- **Metadata facts [M] (station catalogue only; provisional until the split is frozen):** 1,687 of 1,964 stations
  snap ≤ 1 km, in 54 groups; **one group holds 622 of them (37 %)**, G_eff = 6.4 over all groups. Stations open
  after 2023-11-12: 359 in 41 groups, 110 of them in that one group, G_eff = 8.5 over all roles together. Applying the
  draft role rule to snapped stations (not yet to eligible stations) puts the large group in the test role: test share
  ≈ 61 %, **no val-role group**, and a test cell of ≈ 189 open stations in 18 groups with G_eff ≈ 3.
  **Consequence, stated plainly:** under the draft split rule the T-E endpoints would be descriptive by the minimum
  above, and the validation go/no-go could not be run. This has to be resolved in `hydromet_ea_protocol.md` before the
  split is frozen (§14 Q4). The final counts and the claim/descriptive status of each T-E endpoint are fixed in a
  stamped entry of the deviations file after the split freeze and before any EA number.

### 9.2 T-B and N2 — US future year WY2026 (one joint look)
- **Joint look (ruled).** The v1 `M6/test_b` models and the arch-v2 claim models are frozen together and scored in
  **one** look after 2027-06-30. v1 models are secondary members. There is no earlier, separate v1 `test_b` freeze.
- **T-B.** FINAL test-role sites × WY2026, approved data only. Endpoints: 4 primary + 2 heatwave = **6** (the plan's
  list said 4; T-H is ruled a claim endpoint inside T-B, so it is counted in the family — the stricter reading).
  Validation cell for the go/no-go: FINAL val-role sites × val block WY2022–2023 (used in six earlier M6 selection
  cycles; here it is a go/no-go only). Power [RV]: 317 / 315 / 155 / 164 sites in 33 / 31 / 19 / 19 groups; MDE ≈
  0.017 / 0.027 / 0.09 / 0.08 KGE at α = 0.05 (≈ 0.021 / 0.034 / 0.11 / 0.10 with Holm-6); heatwave ≈ 0.44 °C
  (≈ 0.55 °C). Observed v1 gaps are ≈ 0.006 KGE: only a large effect is detectable.
- **N2.** Track-N test-role stations (frozen roles; a station first sampled in WY2026 gets its role by the frozen rule
  from metadata before its values are read) × WY2026, ≥ 8 sample days. Endpoints (**2**): median per-station RMSE of
  temperature and of DO, each sample compared with the model's prediction **at the sample's local solar hour** (no
  time: 08–17 h mixture mean). Models without an hour-resolved output (v1 models, P) use their daily mean. The v1 form
  (daily-mean prediction vs sample mean) is reported for every model as secondary. Validation cell: track-N val-role
  stations × val block. Power [RV]: 38 / 36 test groups; MDE ≈ 0.12 °C and 0.09 mg/L with Holm-2.
- WY2026 values of either cell are not read by any arch-v2 code before the look.

### 9.3 T-H — forcing-defined heatwave windows (claim endpoints inside T-E and T-B)
- Window: runs of ≥ 3 consecutive days with site-cell Tmax above the site's own train-block P95 (US: gridMET, train
  block WY2008–2021; England: ERA5-Land, EA-train block), plus the 3 following days
  (`sentinel.models.hydromet.tracks.heatwave_mask`, q = 95, run = 3, after = 3). Never the outcome-defined P90.
- Endpoints: RMSE of `temp_c_mean` and of `temp_c_7dadm` on window days; per-site RMSE for sites with ≥ 10 scored
  window days; median over sites; Δ = comparator − claim model (positive = better); same bootstrap and minimum groups.

### 9.4 T-D — historical logger reaches (secondary, descriptive only)
- Rows: Oliver non-NWIS/non-WQP stations and WQP logger stations in FINAL test-role and new-region-test groups,
  WY2008–2021; locked at extraction (§2.5); count of stations by group **UNVERIFIED**. Scored once, with models already
  frozen for the joint US look or for an earlier US-model freeze written by the lead. Labelled "historical
  reconstruction". No Holm family, no §15.1 label: its groups are the spent Test-A groups and its years are training
  years. Dropped (and reported as dropped) if the Oliver size check or the input build for its stations fails.

T-M and T-K are deferred. Track R is unchanged and carries no new claim.

## 10. Claim model, comparators, families and the label rule
- **Candidates per track:** D1, D2 (only with a passed G-DO), D3, each in the track's input set and gate-determined
  label set. A lever model is a candidate only if the lever engineer supplies, before the track's validation scoring,
  3-seed inner predictions on the identical inner rows in the track's input set; otherwise it is a secondary arm.
- **Claim-model rule (inner-CV and validation data only; fixed before any result).**
  1. *Inner ordering, before any validation number:* candidates are ordered by the inner criterion of their 3-seed
     early-stopped ensembles — J_sonde for T-B and both T-E modes (schema variants), J_grab for N2; D3 by its
     cross-fitted value. Lower is better; ties go to D1, then D2, then D3, then levers.
  2. *Validation go/no-go, once per track* (§8, G-D1b).
  3. *Choice:* the claim model is the first candidate in the inner ordering that passed the go/no-go. If none passed,
     the track has **no arch-v2 claim model**: T-E is not frozen in this round and its block stays unseen; the joint
     T-B/N2 look runs with the v1 models and all arch-v2 arms as secondary members.
  The choice, the ordering and the validation table are written to the freeze candidate before any test access.
- **Comparator set (IUT).** {R-TD, G1, G2, P}; when D3 is the claim model, also the references-only stack. p_IUT is the
  maximum p over the set. Requiring both G1 and G2 is at least as strict as "the better of G1/G2" and removes the
  choice of which is better (§14 Q5). Comparators are read only from the freeze file.
- **Families (Holm–Bonferroni, family-wise α = 0.05):** T-E zero-shot 6; T-E fine-tuned 6; T-B 6; N2 2. One claim
  model per family. All other arms, ablations, v1 models, lever models and R-EA are secondary and uncorrected.
- **An endpoint is won only if:** (1) Holm rejects; (2) Δ > 0 with a 95 % CI excluding 0 against **every** member of
  the comparator set; (3) every-seed rule: the point Δ is > 0 for each of the 3 single-seed claim models against every
  comparator; (4) the endpoint meets the minimum-groups rule. A label is scoped to its track and to the endpoints won
  ("SOTA on England zero-shot temperature mean"), never generalised. Every arm and track is reported, wins and losses.

## 11. Freeze and single look
- T-E: own FREEZE file written by the lead; requirements of `hydromet_ea_protocol.md` §8 and `sota_campaign.md` rule 7
  (sealed registry row, rescore refused, access closed at sealing, artifacts verified before the claim, BaseException
  aborts, claim tied to pid/host, full import-closure code hash). Registry prefix **`M6EA/test`** (never `M6E`).
- T-B/N2: `FREEZE_hydromet_test_b.json`, written by the lead before 2027-06-30, holding the v1 models and the arch-v2
  claim models; same hardening. The commit and public push of this amendment must exist before any freeze.
- I write freeze *candidates* only, under `sentinel_runs/hydromet/sota_v2/`. I write no `FREEZE_*.json`.
- Validation rows: `M6V2/val_v1`, `M6V2/trackn_val_v1`, `M6EA/val_v1`. Write-ups go to a `*_test_results.md` file.

## 12. Compute, disk and execution order
- GPUs 0–9 only through `python -m sentinel.core.gpu --name <job> --min-free-mb 10000 -- <cmd>`; ≤ 2 h per process
  (jobs stop at 110 min and resume); never a GPU another process occupies. CPU: one heavy job at a time, ≤ 6 threads.
- Disk: `df -h /` before every build or download; stop and report below 60 GB free. M6 new data ≤ 20 GB in total,
  shared with the EA build; planned here: Tier 2 ≤ 3.1 GB, Oliver extract (size UNVERIFIED), run artifacts.
- Open-Meteo: this round makes **no** Open-Meteo call (the fixed variable mapping needs no ERA5-Land at US sites; the
  EA pull is the lever engineer's, sized in the EA protocol §7).
- **Before the stamp:** code and tests; builds of L-grab Tier 1, the VAA join, the solar channels, Powell priors,
  L-logger; Oliver HEAD check; synthetic and train-only smoke (step time, VRAM, D2 kill rule). Tier 2 only after the
  lead answers §14 Q3.
- **After the stamp:** (1) pin the pool; (2) references: G1, G2, P on CPU, R-TD on GPU; (3) D1 grid → seeds →
  ablation (a) → G-D1a → ablation (b) → refits; (4) D2 grid on the gate-determined label set → seeds → ablations →
  G-DO → optional DO stage; (5) D3 and its control; (6) schema variants for T-E; (7) one validation go/no-go per
  track; (8) freeze candidates. Estimated ≈ 30 GPU-h without the DO stage, ≈ 37 with it [plan]; schema variants of all
  candidates add ≈ 6–10 GPU-h (UNVERIFIED).
- Every run gets a registry row through `sentinel.core.registry.log_result`, failed and killed runs included.

## 13. Not yet known (UNVERIFIED)
1. EA eligible-station and group counts per role and endpoint; whether any T-E endpoint meets the minimum (§9.1).
2. Size of the Oliver et al. 2024 release (no HEAD sent yet); count of its new stations; T-D stations by group.
3. Grab-sample counts after role rules (Tier 1 instantaneous rows with a time; Tier 2 pool); Tier 2 build time.
4. L-logger row rule and counts after role rules.
5. D1, D2 and R-TD step time and VRAM; whether D2 survives its kill rule.
6. R-TD published hyperparameters (hidden size, dropout) against the cited papers.
7. Raymond et al. 2012 coefficients (DO stage only).
8. Novelty of the diel observation operator: no novelty wording is used until a dated search exists in
   `docs/novelty_checks/`.
9. Between-group variance in England (all T-E MDEs assume the US value).
10. GPU-hours for the schema variants and fine-tuned runs.

## 14. Questions for the lead
- **Q1 Counting.** Is the variant rule of §4 accepted (re-training a selected configuration on the gate-determined
  label set, on the T-E schema or in the EA fine-tuning cell is logged and reported but not counted against the 12)?
  Without it the 12 slots cannot cover T-E. In particular R-TD needs a no-grab variant if G-D1a fails.
- **Q2 G-D1a for N2.** Proposed: the same ≥ 3 % rule evaluated on J_grab for N2. Alternative: one gate on J_sonde for
  all tracks.
- **Q3 Tier 2.** Approve the forcing build for ≤ 10,000 extra grab stations (transient gridMET yearly files, StreamCat
  API by COMID, ≈ 3.1 GB kept)? Default until answered: Tier 1 only.
- **Q4 EA split (lever engineer's file; I cannot edit it).** By catalogue metadata one group holds 37 % of the snapped
  stations; the draft role rule would likely leave no val-role group and a test cell with G_eff ≈ 3 (§9.1). Options
  to fix before the split freeze, all metadata-only: (a) a pre-declared rule that a group above a share threshold is
  train-role; (b) a per-group station cap in the primary statistic; (c) keep the rule and accept T-E as descriptive.
- **Q5 Minimum groups and IUT.** Confirm G ≥ 15 and G_eff ≥ 8 (or set other values), and the IUT over both G1 and G2.
- **Q6 Family size of T-B:** 6 (4 + 2 heatwave), as written, or 4 as in the plan's list?
- **Q7 Lever protocol §8** says the lever candidate "enters the single `M6/test_b` look". Under the joint-look ruling
  it enters as a secondary member unless chosen by §10. The wording there is the lever engineer's to align.
- **Q8 M1's EA draw.** Default written in §9.1: no EA fit or validation number before `sonde_ea_split.json` exists
  or the lead rules. Confirm.
- **Q9 Ownership of two small files:** `configs/hydromet_sota/v2/pins.json` for the pool pins (my path) — accepted
  as the pin record? Who writes the dated novelty check for the diel observation operator?

## 15. Deviations log
Deviations after the stamp go in a separate `docs/preregistration/hydromet_arch_v2_deviations.md`, stamped before the
affected run produces a validation number.

| Date | Change | Reason |
|---|---|---|
| 2026-11-10 | v0.1 draft | initial |
