# M6 SENTINEL-HydroMet: architecture research and SOTA plan

**Date:** 2026-11-10. **Type:** read-only research synthesis for the lead's decision. AI-assisted.

**This file is not** a pre-registration, a result, or an edit to any frozen file. Every protocol item below is a
*proposal*. The lead writes it into the new amendment file `docs/preregistration/hydromet_arch_v2_amendment.md`
(LEAD_DECISIONS_20261110 rule 1) before any validation number exists on a new arm or track.

**Scope:** M6 estimates daily water temperature (mean, 7-DADMax) and DO (mean, min) at unsensed reaches from
weather forcings, catchment attributes and the calendar only. It never uses in-situ inputs.

**How the evidence was checked (read this first).**

| Tag | Meaning |
|---|---|
| **[RV]** | Repo value re-read or recomputed this session. I computed **counts only**. No protected test-role value and no reserved site was read. The 729 reserved sites (`sonde_newsites_list.json`) were dropped before every count. |
| **[S]** | Found by web search this session (URL or DOI given). |
| **[F]** | Page fetched directly this session (abstract, landing page or API documentation). |
| **[R01]** | Carried from `docs/research/01_sensor_timeseries.md` (verified 2026-10-22). **UNVERIFIED in this session.** |
| **[D]** | Taken from the workflow's design or data inputs; not re-verified by me. |
| **[U]** | Unverified. |

- **Truncated inputs.** The workflow's design, critique, literature and data inputs reached me truncated.
  - I saw one full design, ThermOx-dPL, with part of its fair-test plan. I saw no critic text at all.
  - Section 4 therefore applies my own critique, grounded in repo numbers and the literature fetched here. It names
    every change it makes to ThermOx-dPL.
- **A correction to the inputs.** The data input called the WQP logger and aggregate rows the "largest
  zero-download lever". A count this session does not support that (§6).
  - Non-USGS stream logger data come to only about **153k temperature station-days**, with **396 stations ≥ 60 days**.
  - The genuinely large on-disk lever is the WQP **grab-sample** archive in the training years: **2.16M temperature
    and 1.86M DO station-days**.
- **Compute.** The task assumed GPUs 0–1 only. `LEAD_DECISIONS_20261110.md` (12:10 EST today) now allows GPUs 0–9 under these
  conditions:
  - through the lock helper with `--min-free-mb`;
  - ≤ 2 h per job;
  - never on a GPU that has another user's process.

  CPU stays ≤ 6 threads, and the host is at about 40 of 48 cores, so GPU work is preferred. The plan below fits
  inside a 2-GPU fair share either way.

---

## 0. Decision summary

1. **Simple ML ties because the two models see the same information and the test has little power. Architecture is not the main cause.**
   - **Same information.** The GBM's hand-built causal features (trailing 1–365-day means, API, snow bucket, DO_sat(air)) already encode
     the memory an LSTM would learn. Both models sit at temperature KGE ≈ 0.90.
   - **Power.** The binding constraints are independent sites (memorisation: train MSE 0.06 vs inner 0.21) and test
     power: 33 basin groups give a minimal detectable effect (MDE) of 0.014 KGE against observed gaps of about 0.006.
   - **Literature.** The same picture appears elsewhere:
     - XGBoost was the most accurate single model for unmonitored stream temperature (Varadharajan et al. 2025 [F]).
     - Six ML models were within 0.08 °C of each other (Feigl et al. 2021 [S]).
     - Going from 782 to 1,346 training sites "saturates" an LSTM (Willard et al. 2024 [F]).
2. **The frozen EA-LSTM is adequate but not optimal.** Five concrete defects:
   - a Python-loop cell about 10–30× slower than cuDNN;
   - an early-stopping criterion averaged over 7 targets, three of them near-zero skill, which gives noisy E_k (23 vs 9/10 epochs);
   - mean, 7-DADMax and DO-min heads that are structurally uncoupled;
   - about 4.0M grab-sample station-days (2.16M temperature + 1.86M DO) and the discharge, min/max and DO-max labels left unused;
   - no structural prior for heat extremes.

   Swapping in a Transformer, Mamba, xLSTM or GNN backbone would not change the outcome. There is no information
   gap for a bigger encoder to exploit.
3. **What to build, ranked** (details in §4):

   | Rank | Design | Score /10 | Own GPU-h | Peak VRAM |
   |---|---|---|---|---|
   | 1 | **D1 GSO-MT**: cuDNN multi-task sequence model with a learned **diel observation operator**, trained on sonde daily stats + hourly + ~20k train-region grab-sample stations + Q | 6.5 | ≈ 10 | ≤ 2.5 GB (est.) |
   | 2 | **D2 ThermOx-T**: staged, revised ThermOx-dPL. A differentiable energy-balance and source-mixing temperature core with attribute-learned bounded parameters; its diel sub-model doubles as the grab-sample operator; the O2 budget is gated | 6.0 | ≈ 10 (+ 7 if the DO stage passes its gate) | ≤ 3 GB (est.) |
   | 3 | **D3 SENTINEL-Stack**: inner-group convex stack of {D1, D2, R-TD, G2}, with a references-only control stack | 4.5 | ≈ 0 (CPU) | – |
   | – | Deferred: D4 neighbour-informed network model (changes the input contract; easier, not harder) | 3.0 | – | – |

   **Required references, built first** (lead rule 3), ≈ 6–13 GPU-h plus CPU:
   - **R-TD**: the published top-down LSTM (Rahmani 2021 / Willard 2024 configuration), re-run on identical data.
   - **G2 PF-GBM**: forcing GBM plus physics features plus out-of-fold stacked auxiliary predictions. This is the
     strongest simple baseline.
   - The physics family.
4. **More data is the first lever, and it is mostly already on disk:**
   - **WQP grab samples** in the training years: 119,737 temperature / 110,322 DO stream stations, of which 46,598 / 41,387 have ≥ 8 sample-days. About half are expected to fall in train-role regions.
   - **USGS daily-value sites outside the FINAL split:** 775 temperature / 998 discharge / 119 DO sites with ≥ 365 days (about 405 / 642 / 79 in train-only HUC4s).
   - **Sonde auxiliary labels** at train-role sites: daily min/max at 1,551, DO max at 485, discharge at 1,357, and 749 hourly site tables.
   - **New download:** one, the CC0 USGS multi-agency compilation (Oliver et al. 2024). It already contains NorWeST and EcoSHEDS, so there is no need to download those separately. Size UNVERIFIED; size it with a HEAD request first.
5. **Harder tracks, to be pre-registered now:**

   | Track | Status | Why |
   |---|---|---|
   | **T-E** EA England new geography (zero-shot + fine-tuned) | Approved 2026-11-09 | The earliest unseen block (data exist now), probably more groups, a real domain shift |
   | **T-B/N2** US WY2026 (sonde + WQP stations) | One *joint* look after 2027-06-30 | — |
   | **T-H** forcing-defined heatwave stratum as a claim endpoint on T-E and T-B | — | Not the outcome-defined P90, which is biased by regression dilution (§1.4) |
   | **T-D** diel extremes at non-USGS logger reaches | Conditional on a lead ruling about historical blocks | — |
   | **T-M** metabolism | Deferred | Collides with Sim's H-G3 claim cell |

6. **Honest outlook** (§9):
   - Any scoped §15.1 SOTA label for M6 at the EA look (about 5–7 weeks): **≈ 18 % (12–25 %)**.
   - The same by the Test-B look (Q3 2027): **≈ 25 %**.
   - M6's original H-G1 criterion: **≤ 3 %**.

   The most likely outcome is "competitive" again, with better absolute skill and a much larger training set.
7. **Recommended first step (no GPU, no validation number).** The lead drafts and stamps
   `hydromet_arch_v2_amendment.md` (tracks, arms, references, parity, caps, gates; §8 step 0). In parallel, on CPU:
   - build the role-assigned label tables (counts only for protected rows);
   - send the HEAD request that sizes the Oliver et al. compilation.

---

## 1. Diagnosis: why simple ML wins or ties

### 1.1 The test numbers that define the problem [RV]

Source: `hydromet_test_results.md`, single looks of 2026-11-09.

**`M6/test`** (test-role sites × WY2024–25; Holm 28; 0 won). `ealstm_ws_full` median KGE is 0.902 / 0.890 / 0.789 / 0.800.

| Endpoint | Δ vs best GBM | Δ vs physics |
|---|---|---|
| Temperature mean | −0.006 [−0.020, 0.003] | +0.017 [0.006, 0.033] |
| 7-DADMax | +0.004 [−0.007, 0.018] | +0.014 [0.004, 0.031] |
| DO mean | −0.000 [−0.020, 0.035] | +0.090 [0.054, 0.142] |
| DO min | +0.013 [−0.018, 0.046] | +0.071 [0.038, 0.104] |

- On median per-site RMSE the GBM is ahead on temperature: 1.33 vs 1.44 °C.
- Four other standalone EA-LSTMs are *below* the GBM on temperature mean, with CIs excluding 0.

**`M6N/test`** (unsensed WQP stations; 38 groups): `ealstm_ws_full` is worse than the GBM on temperature by
**+0.161 °C RMSE [0.095, 0.246]**, every seed worse. This is a §15.1 "negative". On DO it is level.

### 1.2 No information gap: the GBM already has the memory

- **Same inputs.** Both models see the same 26 dynamic channels (site and basin gridMET plus DOY harmonics) and the same 191 statics.
- **Memory pre-computed.** `features.py` turns each forcing block into about 45 causal features, the slow memory an LSTM would otherwise learn:
  - trailing 1–365-day means and sums;
  - P − PET;
  - API;
  - a degree-day snow bucket;
  - the 30-day air-temperature anomaly;
  - DO_sat(air).
- **Same ceiling elsewhere.** The literature finds the same equivalence on similar tasks:
  - Feigl et al. 2021 (HESS 25:2951) [S]: 6 ML models in 10 Austrian catchments differ by a median of 0.08 °C RMSE; XGBoost and FNN are each best in 4 of 10.
  - Weierbach et al. 2022 (Water 14:1032) [S]: an XGBoost ensemble was best for unmonitored-basin monthly stream temperature (median RMSE 0.62 °C).
  - Varadharajan, Willard & Ciulla (SFS 2025 abstract) [F]: on CONUS unmonitored stream temperature, LSTM, GRU, TCN and XGB were compared. XGB had the highest single-model accuracy, and the authors "question whether deep learning is needed for this application".
  - Grinsztajn et al. 2022 (NeurIPS) [S]: on medium-sized tabular data, trees remain state of the art because they handle irregular functions and uninformative features well. With hand-built temporal features, M6 is in effect a tabular problem.

### 1.3 Independent sites and power, not capacity, are binding [RV]

- **Memorisation.** The EA-LSTM memorises sites: train MSE 0.06 vs inner MSE 0.21 (handoff). Each of the 1,456 fit sites is uniquely identifiable from 191 standardised statics.
  - Static noise σ = 0.5 helps.
  - A compact 45-attribute set was *worse* on test (−0.026 KGE vs GBM). The statics carry real information, so the fix is not fewer attributes but more sites.
- **Epoch count is noisy.** Seeds early-stopped at 23, 9 and 10 epochs. The early-stopping criterion averages inner MSE over all 7 targets, including SC, turbidity and nitrate, whose unseen-site median KGE is about 0 or negative (model card). These noisy targets help set E_k.
- **Power.** The protocol's own power check gives 80 %-power MDEs of 0.014 / 0.022 / 0.07 / 0.06 KGE on the four endpoints at 34 / 33 / 22 / 22 groups. The val gaps were 0.002 / −0.002 / −0.016 / +0.011. "Competitive" was a power outcome before any architecture choice.
- **More sites saturate.** Willard et al. 2024 [F] report the same pattern for a top-down CONUS LSTM. Going from 782 to 1,346 training sites "changes median RMSE only slightly (saturation)". More sites alone is therefore not a guaranteed lever (§9).

### 1.4 Warm tail: most of the "P90 cold bias" is regression dilution [RV]

Track S, `M6S/val_v2`, val cell, recomputed from `strata_val.csv`:

| Stratum (definition) | Sites / groups | `forcing_gbm` RMSE (bias) | `ealstm_full` | `ealstm_physdo` | `air2stream_regional` |
|---|---|---|---|---|---|
| `warm_p90` temp mean (**observed** > 25.8 °C) | 52 / 10 | 1.53 (−1.29) | 1.48 (−1.26) | 1.56 (−1.22) | 2.28 (−2.09) |
| `heatwave` temp mean (**forcing-defined**) | 106 / 12 | 1.85 (−0.80) | 1.84 (−0.72) | 1.72 (−0.61) | 2.42 (−1.43) |
| `heatwave` 7-DADMax | 88 / 10 | 1.95 (−0.28) | 2.05 (−0.08) | 2.13 (−0.28) | 2.74 (−1.66) |

- **Outcome-defined strata are biased by construction.** For any calibrated predictor of E[Y | X], days selected because Y exceeded a threshold carry a negative mean residual. Part of the −1.2 to −1.4 °C `warm_p90` bias is therefore guaranteed, and fixing it would push predictions too high elsewhere.
- **The real structural bias is smaller.** It is the forcing-defined heatwave bias of −0.6 to −0.8 °C for ML models (−1.4 for air2stream). This is the stratum a physics prior can legitimately target.
- **The stratum needs a large effect to be detected.** Model differences in it are small, and its CI is wide: the physdo−GBM heatwave Δ is −0.13 °C [−0.66, 0.41] at 12 groups.

### 1.5 Track N: a distribution shift the LSTM never trained on

Track-N stations are small, mostly non-USGS reaches. Their `elev_m` comes from StreamCat `elevcat`, since there is no gauge datum. No grab-sample station was ever a training row: the track-N table stores only WY2022 onward.

The LSTM's site-memorising statics transfer worse than trees, which extrapolate flat. The obvious untried fix is to
**train on that population**. Grab samples are in fact the training distribution of the published DO models: Zhi et al. 2021 [S] used sparse CAMELS-Chem samples.

### 1.6 DO and hypoxia: the predictive information is mostly absent

- **Detection fails.** On val, CSI for DO-min < 5 mg/L was ≤ 0.033 for every model, over 255 exceedance days. The summer low-DO stratum had 3 sites in 2 groups [RV].
- **Literature on drivers:**
  - DO dynamics are dominated by temperature, then light, with flow minor (Zhi et al. 2023, Nature Water [S]). The GBM already captures the temperature part through DO_sat(air).
  - Hypoxia (< 2 mg/L) occurs at 12.6 % of locations worldwide and is most likely in warmer, smaller, lower-gradient rivers. Maximum water temperature is its top predictor. Daytime-only sampling misses about 25 % of hypoxic rivers (Blaszczak et al. 2023 [S]).
- **Why the remainder is hard.** The rest of low-DO variance is organic loading, effluent and wetland respiration. These are site-specific and only weakly observed by StreamCat statics.
- **Expectation.** No architecture is expected to fix hypoxia detection. A metabolism-structured model can at most make low-DO risk *coherent* with temperature, depth and slope.

### 1.7 What this implies

| Lever | Expected payoff |
|---|---|
| New *information* (more and different sites; diel and hydrologic supervision; a physically structured prior that extrapolates) | Can move the numbers |
| New *test power* (more independent groups: a new geography) | Can move the label |
| Capacity, depth or a fashionable backbone | Cannot move either |

---

## 2. Literature SOTA summary (verified this session unless tagged)

| Method / paper | Setting | Reported result | Code / data | Use for M6 |
|---|---|---|---|---|
| **Willard, Ciulla, Weierbach, Kumar, Varadharajan 2024**, arXiv:2410.19865 (submitted to WRR) [S][F] https://arxiv.org/abs/2410.19865 | Top-down vs grouped vs bottom-up (MTL) LSTMs; 782 training / 580 "unmonitored" test sites; Daymet + 274 GAGES-II attributes ± observed Q | Median RMSE: top-down 1.43 °C; regional 1.98; cluster 2.02; MTL 2.00. Met only 1.77; met + location + Q 1.57; met + location + GAGES-II without Q 1.47 (1.43 with the extended set); 782 → 1,346 sites gives "saturation". No GBM baseline | doi:10.15485/2448016 (ESS-DIVE) | Defines the **published method class** (top-down LSTM). Statics matter more than Q. Grouped and MoE-style models are worse |
| **Varadharajan, Willard, Ciulla 2025**, SFS abstract [F] https://sfs-2025.p.asnevents.com.au/days/2025-05-19/abstract/118778 | LSTM, GRU, TCN, XGB for CONUS unmonitored stream temperature | XGB most accurate single model; multi-architecture ensembles best; no numbers | – | Independent confirmation that trees tie or win |
| **Rahmani, Shen, Oliver, Lawson, Appling 2021**, Hydrol. Process. 35:e14400 [S] https://pubs.usgs.gov/publication/70238318 ; https://par.nsf.gov/biblio/10299543 | > 400 CONUS basins, PUB | PUB RMSE 1.129 °C (NSF record); dammed-PUB 1.202 °C, R² 0.984; training on all basins preferred for PUB | USGS data release (on the USGS page) | Published PUB bar (context; different sites and years) |
| **Rahmani et al. 2021**, ERL 16:024025, doi:10.1088/1748-9326/abd501 [S] | 118 undammed basins, temporal test | Median RMSE 0.69 °C with observed Q; simulated Q "still helped". The 0.82 / 0.86 °C (simulated / no Q) values are [R01] | MHPI code; USGS release | Upper bound of the discharge-information value |
| **Rahmani, Appling, Feng, Lawson, Shen 2023**, WRR 59(12) e2023WR034420, doi:10.1029/2023WR034420 [S] | δ-SNTEMP: differentiable PRMS-SNTEMP + NN source fractions, 415 basins | Separate shallow-subsurface store helps; recency-weighted air temperature beats simple averaging for source temperature; **a single target cannot constrain internal fractions** | https://mhpi.github.io/codes/Rahmani_et_al_2023 | Template and warning for D2 |
| **Siddik, Turner, Gomez-Velez, Tanim, Kao 2025**, J. Hydrol. 666:134780, doi:10.1016/j.jhydrol.2025.134780 [S] | LSTM with reanalysis meteorology **and reanalysis streamflow**, inputs over thermally relevant upstream areas; 300 basins | Median MAE 1.1 °C, NSE 0.95; unregulated reaches only | Dataset + weights doi:10.21951/RiTHyMs_V1/2587036 (ORNL HydroSource) | Source of the upstream-area idea (null on M6 test). Its streamflow-reanalysis input (product not confirmed here) blocks re-running it on future blocks |
| **Diaz, Oliver, Gorski 2025**, Environ. Model. Softw., doi:10.1016/j.envsoft.2025.106655 [S] | National RGCN, 57,810 segments, 1979–2021 | Authors: does **not** outperform SOTA ML (RMSE ≤ 1.5 °C) because of a limited input set | Predictions doi:10.5066/P13DG5CA | Not usable as an input (trained on observations probably including M6 test-role sites; ends 2021) |
| **Li et al. (DPINN)**, EGU26-13760 [F] https://meetingorganizer.copernicus.org/EGU26/EGU26-13760.html | 0-D heat advection–dispersion + NN; 78 CONUS sites | MAE: DPINN 0.68 °C, MLP 0.79, LSTM 0.93; beats air2stream. Holdout design not stated; "works on single-site data" | Unknown | Evidence that a physics core can beat an LSTM (abstract; probably not unseen sites) |
| **Jia et al. 2021**, SDM, doi:10.1137/1.9781611976700.69 [S] | Physics-guided recurrent graph network, Delaware River Basin | +33 % over the physics model and +24 % over an LSTM on temperature with 0.1 % training data | arXiv:2009.12575 | Graph models shine with sparse in-network *observations*, which M6 does not have |
| **Topp et al. 2023**, WRR 59:e2022WR033880 [S] | RGCN vs Graph WaveNet under shift, DRB | Test RMSE 1.64 / 1.65 °C; Graph WaveNet better in 4 shift partitions | USGS release | Robustness under shift; same-basin training |
| **Sadler et al. 2022**, WRR 58:e2021WR030138, doi:10.1029/2021WR030138 [S] | Multi-task Q + temperature LSTM, 101 sites | Multi-task helped single-site models (56/101 on Q); a pooled model improved only 17 sites; **naive loss weighting hurt temperature** | doi:10.5066/P9U0TG8L | Q-auxiliary gains are small and weighting-sensitive |
| **Tsai et al. 2021**, Nat. Commun. 12:5988, doi:10.1038/s41467-021-26107-z [S] | Differentiable parameter learning | Generalises better as data grow; about 12.5 % of the data for similar performance | MHPI | The scaling argument for D2 |
| **Kratzert et al. 2019**, HESS 23:5089, doi:10.5194/hess-23-5089-2019 [S] | EA-LSTM, 531 CAMELS basins | Beats calibrated models | NeuralHydrology | The frozen M6 backbone |
| **Toffolon & Piccolroaz 2015**, ERL 10:114011, doi:10.1088/1748-9326/10/11/114011 [S] | air2stream | Daily RMSE < 1 °C at 3 rivers; the family beats statistical models at 38 Swiss rivers (Piccolroaz 2016) | Yes | M6 physics family |
| **Isaak et al. 2017**, WRR 53:9181, doi:10.1002/2017WR020969 [S] | NorWeST spatial stream-network models; > 22,700 sites, > 220M readings | Mean-August scenarios on 343,000 km of streams | Data on the USFS site | Published *regime* product. Its spatial-stream-network method needs neighbour observations (D4) |
| **Siegel et al. 2023**, PLOS Water, doi:10.1371/journal.pwat.0000119 [S] | Daily stream temperature, 261,200 free-flowing PNW reaches | RMSE 1.76, MAE 1.32 °C | Open access | Regional daily published product |
| **Letcher et al. 2016**, PeerJ 4:e1727, doi:10.7717/peerj.1727 [S] | Hierarchical Bayesian daily model (EcoSHEDS) | RMSE 0.59 °C in a small network | Yes | Regional published model (EcoSHEDS) |
| **Zhi et al. 2021**, ES&T 55:2357, doi:10.1021/acs.est.0c06783 [S] | LSTM DO, 236 CAMELS-Chem basins, sparse samples | Works in "chemically ungauged" basins; misses DO peaks and troughs | – | Published DO method class (LSTM) |
| **Zhi et al. 2023**, Nature Water, doi:10.1038/s44221-023-00038-z [S]; **Zhi et al. 2023**, Nat. Clim. Change, doi:10.1038/s41558-023-01793-3 [S] | 580 US (+ 216 European) rivers | Temperature > light >> flow; authors note **daytime grab samples bias training** | github.com/LiReactiveWater/WT-DO-US-CE-LSTM | Motivates the diel observation operator (D1/D2) |
| **Blaszczak et al. 2023**, L&O Letters 8:453 [S] https://pubs.usgs.gov/publication/70238795 (DOI 10.1002/lol2.10297 [D]) | 118M DO observations, 125k locations | Hypoxia in 12.6 % of locations; warmer, smaller, lower-gradient rivers; daytime sampling misses ~25 % | – | Low-DO expectations |
| **Appling et al. 2018**, Sci. Data 5:180292, doi:10.1038/sdata.2018.292 [S] | 356-river metabolism (GPP, ER, K600) | 490,907 site-days | doi:10.5066/F70864KX (on disk) | Auxiliary labels |
| "Overcoming equifinality" (Appling et al. 2018, JGR-B) [S title] https://pubs.usgs.gov/publication/70196033 | K600–ER equifinality and pooling | – | – | Down-weight K600 labels |
| **Guan et al.**, EGU26-12035 [F] | Deep-learning daily metabolism reconstruction for 293 US rivers | No holdout design or skill stated | – | Partial prior art for a DO/metabolism head |
| **Grinsztajn et al. 2022**, NeurIPS, arXiv:2207.08815 [S] | 45 tabular datasets | Trees state of the art at ~10k samples | Yes | Explains the GBM tie |

**Published comparator for §15.1(i).** The strongest published method for unmonitored stream temperature and DO is
a **top-down LSTM on forcings and statics** (Rahmani 2021; Willard 2024; Zhi 2021 for DO).

None of the released models can be run faithfully on M6's future test blocks:
- Willard needs GAGES-II attributes, which exist only at gauges.
- Siddik needs reanalysis streamflow.
- Diaz ends in 2021 and is likely leaky.

The §15.1-compliant route is therefore to **re-run the published configuration (R-TD, §4.0) on identical inputs and labels**. δ-SNTEMP is an optional secondary reference: it is not shown to beat a top-down LSTM at unmonitored sites, and its retraining cost is unknown.

---

## 3. Is the current architecture optimal? (honest)

**Verdict: adequate, not optimal. Closing the gap to optimal on the *frozen* task would not change the label.**

Concrete defects, each fixable:

| # | Defect | Evidence | Fix (in D1 and D2) |
|---|---|---|---|
| 1 | Python-loop EA-LSTM cell | 2.7 min per 200-step epoch [D]; fewer configurations, epochs and seeds possible | cuDNN GRU/LSTM; the static gate becomes FiLM/initial-state conditioning |
| 2 | Early stopping on mean inner MSE over **7** targets, 3 of them near-zero skill | E_k = 23 / 9 / 10 across seeds [D] | Criterion on the **4 primary** targets only (a disclosed change) |
| 3 | Mean, 7-DADMax and DO-min are independent Gaussian heads | No min ≤ mean ≤ max structure; 7-DADMax not tied to the daily max | Shared diel latent: amplitude and phase per day; 7-DADMax computed in-model |
| 4 | Unused labels | Q at 1,357 train sites; T min/max at 1,551; DO max at 485; 749 hourly sites; ~2M grab-sample station-days [RV] | Multi-source observation model (§4.1) |
| 5 | No structural extrapolation prior | Heatwave bias −0.6 to −0.8 °C (§1.4) | D2 energy-balance core |
| 6 | Statics identify sites (191-D, standardised) | train/inner MSE 0.06 / 0.21 | 32-D noisy static bottleneck + attribute dropout; mainly more sites |

**What does not need fixing.**
- Hidden size, depth and window length: there is no evidence they bind.
- Transformers: not competitive with LSTMs on CAMELS memory-dependent tasks (Liu et al. 2024; 2025 [R01, UNVERIFIED this session]).
- Mamba/xLSTM: no evidence for unmonitored water temperature found.
- Grouped or mixture-of-experts models: worse than top-down (Willard 2024 [F]).
- Graph routing of forcings: null on test (`ealstm_ws` − `ealstm` −0.001 / +0.001 / −0.003 / −0.015) [RV].

---

## 4. Ranked architecture proposals

**Scoring rubric (/10):** P(scoped §15.1 label on ≥ 1 pre-registered track) 50 %, cost 20 %, scientific novelty and value
15 %, implementation risk 15 %.

### 4.0 References built first (lead rule 3; not candidates)

All references get the same enlarged label set, auxiliary targets, input set (including the new VAA statics and
solar-geometry channels) and 3-seed ensembling as the candidates.

| Code | What | Role | Cost |
|---|---|---|---|
| **R-TD** | Published top-down LSTM, re-run: cuDNN LSTM with statics concatenated at every step (Rahmani 2021 / Willard 2024 configuration; MHPI-style), Gaussian heads, fixed recipe (no grid), trained on the enlarged labels with the same grab-sample operator as D1 | §15.1(i) strongest published method | ≈ 3 GPU-h (3 early-stopped + 3 refits, ~0.5 h each, est.) |
| R-EA (optional) | Frozen `ealstm_ws_full` recipe retrained on enlarged data | Continuity with v1; descriptive | ≈ 7 GPU-h (Python loop); drop if time-limited |
| **G1** | `forcing_gbm_bag` retrained on the enlarged labels, with the grab-sample formulation below | Simple baseline | CPU (or XGBoost GPU-hist; xgboost 3.1.3 is importable here, LightGBM is not in the system Python, so check the project env) |
| **G2 PF-GBM** | G1 plus physics features (prior-parameter Te, trailing 1–30-day means of Te, DO_sat(Te), clear-sky ratio, photoperiod, VAA `pathtimema`, `streamorde`, `lengthkm`) plus **grouped out-of-fold (K = 5 by basin group) stacked predictions** of log Q, T min/max, DO max (and GPP/ER where present) | **Strongest simple baseline** (parity with D1/D2's auxiliary information) | CPU |
| P | `air2stream_regional(_ws)`, `dosat_ratio[...]` | Physics family (kept for continuity) | CPU |

**GBM parity for grab samples (critical).**
- **Training.** The GBM trains on rows (station, date, local solar hour h). Features are its usual daily features plus hour harmonics and their interactions with tmax, tmin and srad. The target is the instantaneous value.
- **Daily outputs.**
  - Daily mean = the average of its 24 hourly predictions.
  - Daily max and min come from a G2-stacked amplitude model trained on sonde min/max.
  - 7-DADMax is computed from the daily max.

This gives the GBM the same information as D1's observation operator. Without it, any D1 win could be attributed to unequal data use.

### 4.1 Rank 1: D1 GSO-MT (Grab-Sample-Operator Multi-Task sequence model)

**Score 6.5 / 10. Own GPU-h ≈ 10. Peak VRAM ≤ 2.5 GB (est.; confirm in smoke).**

**Idea.** Attack the binding constraint (independent sites) with data the frozen model never used. The
architectural novelty is a **learned diel observation operator**. It lets one sequence model learn jointly from
four sources:
- sonde daily mean/min/max;
- hourly sonde shape;
- instantaneous grab samples taken at known hours, at ~10⁴ unsensed stations;
- logger daily statistics.

Joint training makes the daily-mean, daily-extreme and DO-min outputs coherent.

**Architecture.**
1. **Static encoder:** MLP [194 → 128 → 32], GELU, LayerNorm. Inputs are the 191 frozen statics plus VAA `pathtimema`, `streamorde` and `lengthkm`. Training adds static noise σ = 0.5 and attribute dropout 0.1. The 32-D bottleneck limits site identifiability.
2. **Dynamic encoder:** cuDNN `nn.LSTM` (1 layer, hidden 192) over 365-day windows. Inputs are the 26 frozen dynamic channels, 4 deterministic solar channels (photoperiod, clear-sky radiation, cloud ratio = srad / clear-sky, noon solar elevation) and the static embedding (FiLM on the input projection). Loss on the last 90 days.
3. **Daily heads** on [h_t ⊕ z_s], per variable v ∈ {T, DO}:
   - mean μ_v;
   - log diel amplitude a_v;
   - phase φ_v (sin/cos);
   - a skew weight κ_v ∈ [0, 0.5] for a second harmonic;
   - σ_v (softplus).

   The diel curve is g_v(h) = A_v[cos(2π(h − φ_v)/24) + κ_v cos(4π(h − φ_v)/24)] / norm, evaluated on a 24-point grid.
   - Daily max/min = μ_v + max/min g_v.
   - 7-DADMax = trailing 7-day mean of the daily max, computed inside the model.
4. **Auxiliary heads:** log Q (μ, σ); secondary SC, turbidity, NO3 (weight 0.25, never used for stopping).
5. **Observation model** (Gaussian NLL, masked, target-balanced):
   - **Sonde daily stats:** mean, min, max and 7-DADMax, with model σ.
   - **Sonde hourly** (749 site tables): μ + g(h), with model σ.
   - **WQP grab sample** at local solar hour h: μ + g(h), with variance σ_model² + σ_src².
     - σ_src is learned per provider class (USGS / state / tribal / local / volunteer / other), with a half-normal prior.
     - Rows with no time take a mixture over 08–17 h.
   - **Logger daily stats** (WQP aggregate, Oliver compilation): treated like sonde daily stats, plus σ_src for the logger class.
6. **Sampling:** a fixed mix of 50 % sonde-site windows and 50 % grab/logger-station windows. Each station's weight is capped, so 20k stations cannot swamp the sondes.

**What the GBM lacks** (honest about what parity removes):
- **(a) Shared representation.** Mean, extremes, Q and DO share one hydrologic and thermal state. The GBM gets these only as stacked out-of-fold features (G2).
- **(b) Coherent diel structure** for 7-DADMax and DO-min from one latent.
- **(c) Source-specific noise learning** across heterogeneous agencies.

With full parity, the realistic edge is (a) and (b), and that edge is not guaranteed. The main bet is that a
shared-state sequence model **scales better with 10× more stations** than per-target trees. Tsai et al. 2021 show
this scaling for parameter learning. Willard et al. 2024 show saturation for a plain LSTM going from 782 to 1,346 sites,
which is the counter-evidence.

**Novelty.** One search this session [S] found no stream-temperature or DO model that learns a diel observation
operator to train on time-stamped grab samples. Zhi et al. 2023 [S] name the daytime-sampling bias but, in the excerpt
I saw, do not model it. Novelty is plausible, **UNVERIFIED** until a dated `docs/novelty_checks/` search.

**Fair test plan.**
- **Selection.** Inner train-role groups only (`inner_frac` 0.15, seed 0, the §2 rule). The criterion is target-balanced point MSE over the 4 primary targets **on sonde sites**, plus grab-sample NLL at inner-group stations (pre-declared weights).
- **Grid.** 3 configurations:
  - (hidden 192, grab share 0.5);
  - (hidden 192, grab share 0.25);
  - (hidden 128, grab share 0.5).

  Seed 0 per point; ≤ 60 epochs, patience 6.
- **Final model.** Seeds 0–2 of the selected point early-stop to fix E_k, then refit on fit + inner for E_k epochs. The evaluated model is the 3-seed ensemble.
- **Comparators.** R-TD, G2 (and G1), and P, all on identical cells. IUT over {published, simple, physics}, every-seed rule, Holm over the track family (§7).
- **Ablations (counted against the cap):**
  - D1 without grab/logger stations;
  - D1 with diel heads replaced by independent heads;
  - D1 without auxiliary heads.
- **Kill / gate rules** (pre-declared; train-block inner groups only):
  - **G-D1a.** If adding grab-sample stations does not lower the inner-group primary MSE by ≥ 3 % vs the no-grab ablation, grab samples are dropped from *all* models' claim configurations. This is reported.
  - **G-D1b.** One validation go/no-go per track. D1 proceeds to a freeze only if its point Δ vs both R-TD and G2 is ≥ 0 on ≥ 3 of 4 endpoints in the track's val cell. Otherwise it is reported and not frozen.

### 4.2 Rank 2: D2 ThermOx-T (revised ThermOx-dPL, temperature first, DO behind a gate)

**Score 6.0 / 10. Own GPU-h ≈ 10 for the temperature stage, + ≈ 7 for the DO stage only if gate G-DO passes. Peak VRAM ≤ 3 GB (est.).**

**Idea.** Keep ThermOx-dPL's strongest element: a differentiable conservation-law core whose ~13 bounded reach
parameters are learned from attributes (parameter learning à la Tsai 2021). Water temperature relaxes exactly toward a
convex mix of source temperatures with a depth-proportional timescale. Drop or defer the parts the evidence does not
support.

**Temperature stage (kept from ThermOx-dPL, unchanged unless noted).**
- **Encoders.** Static MLP [194 → 128 → 64 → 32] with noise σ = 0.5. Dynamic encoder: cuDNN GRU (hidden 64) over the 26 channels ⊕ z_s.
- **Bounded static parameters.** Biases initialised at prior medians:
  - h_ref, f_h;
  - shade_max, leaf-on DOY, leaf-off DOY;
  - wind-sheltering w;
  - g_base, τ_gw, ΔT_gw;
  - w_res, τ_res;
  - w_up;
  - β (low flow raises the groundwater share).
- **Dynamic latents:** q_t = log(Q/Q̄) (supervised), snowmelt share m_t, groundwater modifier Δg_t, residual r_t ∈ [−c, c].
- **Physics core** (fp32):
  - a surface energy balance (shortwave with shade, atmospheric/vegetation longwave, LW_out, evaporation via a fixed wind function, Bowen sensible heat);
  - 4 vectorised Newton steps give the equilibrium temperature Te_t and the exchange coefficient;
  - source temperatures:
    - Te;
    - groundwater = recency kernel of basin air temperature + ΔT_gw (Rahmani 2023's recency-weighted prior);
    - reservoir kernel;
    - upstream Te lagged by `pathtimema`;
    - melt at 0 °C;
  - an exact exponential integrator: Tw̄_t = Te_eff + (Tw̄_{t−1} − Te_eff)·e^{−Δt/τ_t};
  - an hourly linearised diel sub-model on loss days (2 periodic cycles × 24 steps).
- **Outputs:** μ = physics + r_t; σ heads; 7-DADMax from the trailing mean of the hourly maximum.

**Required changes vs the submitted ThermOx-dPL (my critique):**
1. **Staged scope.** The O2 budget (K600 via the Raymond 2012 form, GPP light-saturation, Arrhenius ER, hourly O2 ODE, soft-min DO) runs **only if** gate G-DO passes:
   - temperature-stage inner primary MSE within 2 % of D1's; **and**
   - inner-group heatwave-stratum RMSE ≤ D1's.

   **Reason.** About 20 coupled parameters with a Newton solve, two ODEs and a diel sub-model carry high engineering risk. The DO payoff (§1.6) is the least likely to show.
2. **Drop Powell depth and velocity as "independent" labels.**
   - Powell `sites.parquet` carries hydraulic-geometry coefficients (`hydgeom_c/f/a/b`) [RV]. Depth and velocity there are therefore functions of discharge, and add no information beyond the log-Q label (Appling et al. 2018 methods; exact derivation **[U]**).
   - Use the coefficients only as **priors** on h_ref and f_h at the 160 train-role Powell sites.
   - K600 labels: H-confidence days only, low weight (0.1). The K600–ER equifinality is documented ("Overcoming equifinality" [S]).
3. **Re-scope the motivation.** Target the forcing-defined heatwave stratum, cross-domain transfer (T-E zero-shot) and future years (T-B). Do **not** target the outcome-defined P90 bias (§1.4).
4. **Integrate the grab-sample operator.** Tw at a sample's hour comes from the diel sub-model, so the ~20k grab-sample stations supervise the diel physics directly. This is shared with D1, and it partly answers Rahmani 2023's warning that a single target cannot constrain internal states.
5. **Published reference.** It is R-TD, not δ-SNTEMP (cost unknown, and not shown to beat a top-down LSTM for PUB). δ-SNTEMP stays an optional secondary reference if its released code runs within 3 GPU-h on a smoke subset.
6. **Budget trimmed** from 26 to ≈ 10 (+7) GPU-h. Comparators are shared with D1, so they are not double-counted.
7. **Numerical safety.**
   - fp32 physics: σT⁴ overflows fp16. Optional fp16 autocast is allowed on the GRU and MLPs only.
   - Te clamped to [−5, 45] °C.
   - finite-gradient assertions through the Newton and recurrence steps in smoke.
   - Raymond et al. 2012 coefficients are **[U]** and must be transcription-checked against the paper before the amendment is stamped.
8. **Equifinality control.**
   - c ∈ {0, 1 °C} stays a grid choice.
   - The c = 0 "pure bottleneck" variant is always reported as an ablation.
   - Learned parameters are reported with their ranges at inner sites (an interpretability deliverable).

**What the GBM lacks** (after parity, which gives G2 Te_prior, DO_sat(Te) and the clear-sky ratio):
- monotone, bounded responses and a 1/h radiative sensitivity under out-of-range forcing (heatwaves, a new country's forcing and attribute distributions);
- a ~13-number site summary in place of 191 free statics, which limits memorisation.

On in-distribution US cells, parity may let G2 match it. Its realistic winning ground is **T-E zero-shot and T-H**.

**Novelty.**
- Rahmani 2023 [S] (δ-SNTEMP) and DPINN [F] are the closest prior art.
- New elements: a full energy balance with attribute-learned parameters, supervised with Q, diel and grab-sample operators; and a zero-shot cross-country test.
- Search coverage: one search for differentiable metabolism models [S] found only per-site Bayesian inversion, RF upscaling and a DL reconstruction abstract (Guan et al. [F]). The DO-stage novelty is plausible, **UNVERIFIED**.

**Fair test plan.**
- **Grid** (temperature stage, 3 points; seed 0 each):
  - {c = 0, λ_Q = 1};
  - {c = 1 °C, λ_Q = 1};
  - {c = 1 °C, λ_Q = 0.3}.
- **Selection.** Inner selection, seeds and refit as in D1. Primary-only criterion.
- **Warm-up.** 3 epochs with residual frozen at 0.
- **Comparators.** R-TD, G2, P. Same IUT, every-seed rule and Holm.
- **Ablations:**
  - no diel sub-model;
  - no source mixing (atmosphere only);
  - c = 0 (already in the grid).
- **Smoke.** ≤ 50 train-role sites. Measure s/step and VRAM before stamping.
- **Kill rule.** If the smoke step time exceeds 0.5 s/step at batch 256, cut the diel sub-model to loss days of summer months only. If still too slow, drop D2.

### 4.3 Rank 3: D3 SENTINEL-Stack (cheap final-candidate option)

**Score 4.5 / 10. GPU-h ≈ 0 (CPU).**

**What.** A non-negative convex stack, per primary endpoint, of {D1, D2, R-TD, G2}. Weights are fitted on the
inner-group predictions of the early-stopped runs (no val use) and then applied to the refit members.

**Matched control (mandatory).** A **references-only stack** {R-TD, G1, G2}, as in M1's E2-B control. If D3 beats the
GBM only by as much as the control does, the gain is ensembling, not SENTINEL.

**Why.** Varadharajan et al. 2025 [F] found multi-architecture ensembles most accurate. M6's own 50/50 blend tied the GBM.

**Limits.** D3 contains its comparators (disclosed, as with `ealstm_gbmblend`). It is the claim model only if
pre-declared as such before val.

### 4.4 Deferred: D4 neighbour-informed river-network model (track K)

**Score 3.0 / 10.**

A flow-network attention model over anomalies at *other* sensors (NorWeST-style spatial stream-network information;
Isaak et al. 2017 [S]) would probably cut errors at unsensed reaches. But it changes M6's input contract (no in-situ
data), and it makes the task **easier**, not harder.

Fund it only if the lead wants an "operational interpolation" product. It would need its own track. Its comparators would be:
- a GBM with identical neighbour features;
- a spatial-stream-network regression.

### 4.5 Rejected (with reasons)

| Idea | Why rejected |
|---|---|
| Transformer / Mamba / xLSTM backbone swap | No information gap. LSTM ≥ Transformer on memory tasks [R01, UNVERIFIED] |
| Regime experts (groundwater / dam / snow MoE) | Grouped models are worse than top-down (regional 1.98 / cluster 2.02 vs 1.43 °C) [F] |
| River-network GNN routing of forcings | Basin-forcing lever null on test [RV]. National RGCN below SOTA ML [S]. Graph gains need in-network observations [S] |
| Caravan / CAMELS discharge pretraining | Q is worth less than statics (Willard: met + location + Q 1.57 vs met + location + statics 1.47) [F]. A pooled multi-task Q model helped 17/101 sites [S]. Caravan is 24.8 GB [R01], beyond the disk headroom |
| NWM v3 retrospective discharge as an input | The archive covers Feb 1979–Jan 2023 [S], so there is no input for WY2024+ test blocks. Calibration leakage risk |
| Distilling or using the Diaz 2025 RGCN predictions | Trained on observations probably including M6 test-role sites; ends 2021 [S] |
| Tabular foundation models (TabPFN-type) for M6 | Millions of rows vs small-context design; not searched for M6 this session [U] |
| Loss reweighting toward outcome-defined P90 days | Fights regression dilution (§1.4); worsens other days |

---

## 5. New numeric predictive targets (multi-task)

All counts are [RV], computed this session, excluding the 729 reserved sites. "Train-role" means FINAL split
train role. Counts use ≥ 365 days of sonde data unless stated.

| Target | Role in D1/D2 | Label source | Train-role count | Notes |
|---|---|---|---|---|
| Daily T min, T max (→ diel range) | Auxiliary + structural (diel heads) | `sonde_daily` `temp_c_min/max` | **1,551 sites** | Val 272, test 506 |
| Hourly T shape | Auxiliary (shape) | `sonde_hourly` | 749 site tables in total (roles not counted) | Identical to the 2026-10-24 snapshot up to 2023-09-30 |
| Daily DO max (→ diel DO range) | Auxiliary + structural | `sonde_daily` `do_mgl_max` | **485 sites** | DO min 485, DO mean 487 |
| log Q (daily mean discharge) | Auxiliary target (never an input) | `sonde_daily` `discharge_cfs_mean` | **1,357 sites** + ≈ 642 not-in-split sites in train-only HUC4s | Not-in-split: 998 sites, ≈ 201 touch test HUC4s, so they are excluded |
| Instantaneous T and DO at sample hour | Main new-data target via the observation operator | `processed/wqp_discrete` (stream, accepted) WY2008–21 | 119,737 T / 110,322 DO stations in all regions; **46,598 / 41,387 with ≥ 8 days**; 2.16M / 1.86M station-days | Roles not yet assigned. Track-N's ratio (6,244 of 12,768 train-region) suggests ≈ 50 % train-region. Includes NWIS discrete; dedupe against DV sites |
| Logger daily mean/min/max (non-USGS) | Auxiliary + more sites | Raw WQP physchem (QC step-3 rows) | 6,067 T / 5,175 DO stations; **396 / 253 with ≥ 60 days** in total | Approximate classifier (see §6). **Small lever** |
| GPP, ER (daily) | D2 DO stage only; G2 stacked | `labels_powell_metabolism` | **160 sites, 234,978 site-days** (H 176,180, M 32,178, L 26,620) | Val 37, test 117 (reserved by Sim's H-G3 claim; never use) |
| K600 | D2 DO stage, weight 0.1, H-confidence only | Powell | Same | Equifinality with ER |
| GPP, ER (StreamPULSE) | D2 DO stage | `labels_streampulse` | 62,383 site-days, 738 site records; roles unassigned | Train-role groups only, after role assignment |
| Depth, velocity | **Not targets** | Powell | – | Q-derived through hydraulic geometry |
| SC, turbidity, NO3 | Secondary (weight 0.25, never used for stopping) | `sonde_daily` | SC 931, turbidity 312, NO3 59 | Near-zero unseen-site skill |
| DO % saturation, pH | **Not targets** (too sparse) | `sonde_daily` | 22, 31 sites | – |

**Derived regime outputs** (computed from the predictive distribution; scored descriptively unless a track lists them):
- annual maximum 7-DADMax;
- days > 20 and > 24 °C;
- mean August temperature (NorWeST-comparable);
- thermal sensitivity (slope of water vs air temperature);
- P(DO-min < 5) and P(DO-min < 2 mg/L).

These numeric targets feed the event track. They do not replace it.

---

## 6. More data: verified sources, licence and size

Disk on `/` is at **86 GB free** [RV, `df -h` today]. The floor is 60 GB, so there is about 26 GB of headroom for
*all* workstreams. M6's cap is 20 GB, mostly earmarked for EA England.

| Source | Status | Licence | Size | Content | Verdict |
|---|---|---|---|---|---|
| **WQP grab samples, WY2008–21** (`processed/wqp_discrete`) | On disk [RV] | US public domain | 0 new; processed table < 0.5 GB (est.) | §5 counts; `activity_start_time` + tz available | **Use first.** Assign roles with the frozen rules: FINAL basin-group + track-N role logic, test-role and reserved exclusion, new sha |
| **USGS DV sites outside the FINAL split** | On disk [RV] | Public domain | 0 | 775 T / 998 Q / 119 DO sites ≥ 365 d | **Use** after the role assignment (HUC4 + flow-connection rule) and `sentinel.data.reserved` exclusion |
| **Sonde min/max/DO-max/Q and hourly** | On disk [RV] | Public domain | 0 | §5 | **Use** |
| **WQP logger/aggregate rows** (raw physchem) | On disk [RV] | Public domain | 0 | My approximate re-classification of QC step 3 (activity type contains Logger/Continuous/Time Series/Composite, or non-empty time basis or statistical base, or > 12 distinct times per site-date), stream site types, non-USGS: 153,376 T and 91,384 DO station-days; 396 / 78 T stations with ≥ 60 / ≥ 365 days | **Use, but small.** It corrects the "largest lever" claim |
| **Oliver et al. 2024**, multi-agency compilation, doi:10.5066/P9EMWZ35 [S][F] https://www.usgs.gov/data/compilation-multi-agency-water-temperature-observations-us-streams-1894-2022 | Not on disk | **CC0 1.0** (USGS page) | **UNVERIFIED** (not stated; HEAD request first) | Daily mean/min/max from NWIS, WQP, **EcoSHEDS and NorWeST**; 1894-08-30 to 2022-03-31 | **Download** (stream, extract, delete the zip). It replaces separate NorWeST/EcoSHEDS pulls. Only the non-NWIS, non-WQP stations are new. Crosswalk NHM segments → COMID by point-in-catchment |
| NorWeST, USFS RDS-2016-0032 [S] (Isaak 2017: > 22,700 sites, > 220M readings) | – | "Without additional permissions" [D] | [U] | Mostly summer loggers, western US | Covered by Oliver et al.; do not download separately |
| EcoSHEDS / NE stream temperature [D] (model: Letcher 2016 [S]) | – | [U] | Small [D] | NE loggers | Covered by Oliver et al. |
| **EA Hydrology API**, water-quality sondes [F] https://environment.data.gov.uk/hydrology/doc/reference | Approved; not downloaded | **OGL v3.0** (API reference: "All content is available under the Open Government Licence v3.0") | Within M6's 20 GB cap (handoff); stream to daily stats | 15-min temperature, DO (% and mg/L), conductivity, pH, turbidity | **Use** (T-E). The ~1,960-station count is from the 2026-11-09 data check [D] |
| **HydroATLAS** (RiverATLAS/BasinATLAS), Linke et al. 2019 Sci. Data 6:283, doi:10.1038/s41597-019-0300-6 [S] | – | **CC BY 4.0** | [U]; take regional subsets only | 281 attributes, 8.5M reaches, global | **Conditional.** Only if T-E zero-shot uses a global static schema for *both* domains. The plan of record (handoff) is StreamCat-schema matching |
| ERA5-Land through Open-Meteo [S] https://open-meteo.com/en/docs/historical-weather-api | Partly on disk (`raw/openmeteo_era5`) | ERA5 / Open-Meteo terms [U] | Small per point; **shared daily quota** | Daily forcings, snow depth, soil temperature (the exact deep-layer variable names are [U]) | Needed for EA. For zero-shot, US training also needs ERA5-Land-derived channels at sonde sites. Size the quota cost first; the puller is capped at 60 % |
| TempER (Grundmann et al., EGU26-7752) [F] | – | **Dataset licence/repository not stated** | – | 4,757 European stations, up to 72 years | **Not now.** A later new-geography candidate once a citable release exists |
| Caravan, Kratzert et al. 2023 Sci. Data 10:61 [S] | – | CC BY 4.0 (4TU listing) | 24.8 GB [R01] | Q + ERA5-Land, 6,830 basins; no water quality | **Reject** (low value, over the disk headroom) |
| NWM v3.0 retrospective [S] https://registry.opendata.aws/nwm-archive/ | – | AWS open data | Multi-TB; subset size [U] | Simulated Q per reach, Feb 1979–Jan 2023 | **Reject:** no coverage of the test blocks |
| Diaz et al. 2025 predictions, doi:10.5066/P13DG5CA [S] | – | USGS [U] | Multi-GB [D] | 57,810 segments, 1979–2021 | **Reject** as an input (leakage). Descriptive only, on historical cells |

**Data-hygiene rules** (proposed for the amendment):
1. Every new label row passes `sentinel.data.reserved.exclude_reserved` / `assert_no_reserved`.
2. No row from a FINAL test-role group, from any track-N test-role or new-region-test station, or from the spent cells (test-role × WY2024–25) enters training or selection.
3. Grab-sample stations whose COMID holds a FINAL-split or reserved site are dropped.
4. Role-assignment code and outputs are pinned by sha before any fit.
5. For protected rows: counts only.

---

## 7. Harder task tracks (practical justification first; unseen block; power)

**Power method** (metadata only): MDE(G) ≈ MDE₀ · √(G₀/G), using the protocol's group-SE scaling.
- The primary test cell gives MDE₀ (80 % power, α = 0.05) = 0.014 / 0.022 / 0.07 / 0.06 KGE at G₀ = 34 / 33 / 22 / 22 groups.
- Track N's group SE comes from the `M6N/test` CI widths:
  - temperature half-width 0.075 °C → SE ≈ 0.038 → MDE ≈ 0.11 °C (α = 0.05) or ≈ 0.13 °C (Holm-4);
  - DO: SE ≈ 0.029 → MDE ≈ 0.08 / 0.10 mg/L.
- The heatwave stratum: the val CI half-width is ≈ 0.5 °C at 12 groups, so SE ≈ 0.16 at 33 groups and MDE ≈ 0.44 °C.

| Track | Practical justification (stated before any result) | Unseen test block | Power (metadata only) | Recommendation |
|---|---|---|---|---|
| **T-E** EA England new geography: **zero-shot** (US-trained) and **fine-tuned** (EA train block; comparators fine-tuned identically). Endpoints: temperature mean, 7-DADMax, DO mean, DO min (median per-site KGE) + heatwave-stratum RMSE (T-H) | A deployable unsensed-reach estimator must transfer to a new country's network, forcing product and regulator. England has thousands of WFD reaches without sondes | EA values ≥ 2023-10-01 at M6-EA test-role groups (reserved since 2026-11-09, `ea_england_allocation.md`); roles from `hydromet_ea_split.json`, frozen before any fit | **Unknown until the metadata count.** If G ≈ 60 → MDE(temp mean) ≈ 0.011; if G ≈ 120 → ≈ 0.007 (assumes US-like between-group variance; zero-shot heterogeneity is likely larger) [U] | **Primary SOTA opportunity.** Zero-shot needs a common input schema for *every* model (forcing: ERA5-Land-derived channels in both domains or a fixed variable mapping; statics: StreamCat-matched or HydroATLAS). Pre-declare the "near consented discharge" stratum. EA sondes skew to impacted waters |
| **T-B / N2** US future year WY2026: FINAL test-role sondes + track-N test-role WQP stations | The operational product runs on years after all development | Sondes: test-role × WY2026 (Approved after 2027-06-30). WQP: test-role stations × WY2026, including new stations | As Test-A: 0.014 / 0.022 / 0.07 / 0.06 KGE; N2: 0.13 °C / 0.10 mg/L (Holm-4) | **One joint look**, see the note below |
| **T-H** forcing-defined heatwave windows (≥ 3 days of site Tmax > train-block site P95, + 3 following days; §11 thresholds) as a **claim endpoint** on T-E and T-B | Thermal-stress management for coldwater fish; 7-DADMax criteria | As the parent track | MDE ≈ 0.44 °C at ~33 US groups; T-E unknown | Pre-register. Only a structural gain (D2) is detectable. **Not** the outcome-defined P90 (regression dilution) |
| **T-D** diel extremes (7-DADMax, daily min/max) at **non-USGS logger reaches** (Oliver compilation + WQP loggers) | Regulators set temperature standards on 7-DADMax at reaches without USGS gauges | Logger stations in FINAL test-role and new-region-test groups. Sources end 2022-03-31, so the block is **historical (WY2008–21) and contemporaneous with training years**. These values were never read | Unknown: needs a station-by-group count (CPU, counts only) | **Conditional on a lead ruling.** The data are unseen, but the groups overlap spent Test-A groups. If approved: label it "historical reconstruction" and make it secondary unless the count gives ≥ 30 groups |
| T-M metabolism (GPP/ER at unseen groups) | Ecosystem-function mapping | Test-role Powell sites | 117 sites | **Deferred.** `sim_protocol.md` reserves the test-role Powell sites for Sim's H-G3 final claim |
| T-K neighbour-informed interpolation | Operational gap-filling between gauges | – | – | **Deferred** (§4.4; easier, not harder) |
| Track R (leave-region-out) | Already pre-registered | Same cells as Test-A | – | Unchanged; cannot carry a new claim |

**Note on T-B: one joint look.** v1 already plans a `M6/test_b` look of the frozen models. Under §15.5(1), once Test-B
is scored, an improved model cannot claim on it. **The v1 `test_b` look and the arch-v2 claim model must therefore be
frozen and scored in a single look**, with v1 models as secondary members. Otherwise the arch-v2 model needs another
block. This has to be decided before the v1 `test_b` freeze.

**Holm families** (proposed):
- **One claim model per track** (D1, D2 or D3, chosen before val by a pre-declared rule), × the track's endpoints.
  - T-E zero-shot: 4 endpoints + 2 heatwave = 6 hypotheses. T-E fine-tuned: 6.
  - T-B: 4. N2: 2.
- Every family is IUT over {R-TD, best of G1/G2, P}, with the every-seed rule.
- Other arms are reported as secondary.

---

## 8. Sequencing within the shared GPU budget and configuration caps

**Configuration caps** (counted in `configs_tried.jsonl`). The handoff's ≤ 12 neural-network configurations total is kept:

| Arm | Configurations |
|---|---|
| D1 | 3 + 3 ablations |
| D2 temperature stage | 3 + 2 ablations (c = 0 is in the grid) |
| D2 DO stage (gated) | ≤ 1 |
| R-TD | 1 fixed |
| **Total** | **13** |

13 exceeds 12 by one. To stay within cap, **drop the D1 "no auxiliary heads" ablation** (total 12). R-EA is an
optional descriptive re-run outside the cap only if the lead agrees. The GBMs get 3 per variant (G1, G2).

**GPU plan** (2080 Ti; ≤ 110 min per job with per-epoch checkpoint and `--resume`; one job per GPU via the lock helper):

| Phase | Items | GPU-h (est.) |
|---|---|---|
| Smoke (train-only, ≤ 50 sites) | D1, D2, R-TD: s/step, VRAM, gradient health | 1.0 |
| References | R-TD 3 early-stopped + 3 refits | 3.0 |
| D1 | 3-point grid (≈ 0.7 h each) + 2 early-stopped seeds + 3 refits + 2 ablations | ≈ 7.5 |
| D2 temperature stage | 3-point grid (≈ 0.8 h each) + 2 + 3 + 2 ablations | ≈ 8.5 |
| D2 DO stage | Only if G-DO passes | ≈ 7 |
| T-E | Claim model + R-TD in global-schema / ERA5 inputs (zero-shot) and fine-tuned, 3 seeds each | ≈ 6 |
| Inference | US val, track N, EA | 1.0 |
| Contingency | – | 3.0 |
| **Total** | Without the DO stage / with it | **≈ 30 / 37** |
| Optional | R-EA continuity re-run | + 7 |

- **On 2 shared GPUs.** A 1/7 fair share is ≈ 6.9 GPU-h per day, so the plan takes ≈ 5–6 working days of GPU time
  over about 3 weeks.
- **Under the 0–9 rule** of LEAD_DECISIONS it fits in about 1 week.
- **CPU.** The CPU work (label builds, GBM parity, stacking) runs **sequentially**, ≤ 6 threads, because the host is at about 40 of 48 cores.

**Order of work.** Steps 0–2 are allowed before the stamp under LEAD_DECISIONS rule 2. There must be no validation number before the stamp.

| Step | When | Work |
|---|---|---|
| **0** | Now; no GPU, no val number | The lead drafts and stamps `docs/preregistration/hydromet_arch_v2_amendment.md` (contents below) |
| **1** | CPU, before the stamp | Label builds:<br>(a) WQP grab samples WY2008–21 with local solar hour, provider class and roles;<br>(b) USGS not-in-split sites;<br>(c) logger daily stats;<br>(d) VAA join and solar channels;<br>(e) Powell priors.<br>HEAD request to size the Oliver et al. compilation; download it only if it fits under the floor. EA build per the approved plan; count EA groups from metadata, then the T-E power check |
| **2** | GPU smoke, train-only data | Measure s/step and VRAM; apply the D2 kill rule |
| **3** | After the stamp | References first: R-TD, G1, G2, P on the enlarged data. Then D1 grid → seeds → refits → ablations; gate G-D1a |
| **4** | After step 3 | D2 temperature grid → seeds → refits; gate G-DO → optional DO stage |
| **5** | CPU | D3 stack + references-only control |
| **6** | Once per track | Val go/no-go: US val cell, track-N val stations, EA val groups. Gate G-D1b (and its D2 analogue). The claim model per track is chosen by the pre-declared rule |
| **7** | — | EA freeze (own FREEZE file, full import-closure code hash, hardened single-look guard per `sota_campaign.md` rule 7) → single look `M6EA/test`. Do not reuse the `M6E` prefix, which is the event track. Earliest about 5–7 weeks from now, gated by the EA data build |
| **8** | — | Joint Test-B / N2 freeze before 2027-06-30 → single look after Approval |

**Step 0: what the amendment must contain.**
- Tracks T-E, T-B/N2, T-H (and T-D if ruled in), each with its dated practical justification, unseen block, power method and group counts from metadata.
- Arms D1, D2 (with the G-DO gate), D3 and its control. References R-TD, G1, G2, P, with the parity rules of §4.0, including the GBM grab-sample formulation.
- Grids, caps, the selection criterion (4 primary targets; disclosed change from v1), gates G-D1a, G-D1b and G-DO, kill rules, and the claim-model rule per track.
- Label sources and role-assignment rules with their sha.
- The Holm families.
- The joint Test-B decision.

**Code location.** Only unhashed paths:
- `sentinel/hydromet_sota/{gso_mt,thermox,stack}/`;
- `scripts/{train,eval}/sota_hydromet_*.py`;
- `configs/hydromet_sota/`.

Physics helpers are imported read-only from the frozen module. The frozen code hash `5bd7d329…` must not change.

---

## 9. Honest outlook (probability of a §15.1 SOTA label, per track)

These are subjective estimates from the evidence above, not computed. "SOTA" means: CI excluding 0 vs **both** R-TD and
the best of G1/G2 (plus physics), ≥ 3 seeds every-seed, Holm within the track, single look.

| Track / endpoint | P(SOTA) | Reasoning |
|---|---|---|
| T-E zero-shot, ≥ 1 temperature endpoint | **15 % (10–25 %)** | Genuine domain shift is where a physics bottleneck (D2) and broader training (D1) can beat flat-extrapolating trees. Probably more groups. Risks: forcing/attribute-schema mismatch hurts every model; impacted EA sites |
| T-E fine-tuned, temperature | 8 % | With EA training data the GBM recovers; back toward the information ceiling |
| T-E, DO endpoints | 5–8 % | DO at impacted sites is driven by effluent; no input sees it |
| T-H heatwave (EA or US) | 6 % | MDE ≈ 0.4 °C (US) vs a structural bias of 0.6–0.8 °C; needs D2 to remove most of it while G2 does not |
| T-B US primary (any temperature endpoint) | 5 % | Same 33 groups; MDE 0.014 KGE vs gaps ~0.006; the GBM gets the same new data |
| N2 temperature | 5 % | Must turn a 0.16 °C deficit into a ≥ 0.13 °C lead while the GBM also gains from grab samples |
| N2 DO | 6 % | Level now; MDE 0.10 mg/L |
| Low-DO events (CSI / Brier) | ≤ 3 % | Information absent (§1.6) |
| **Any scoped SOTA label at the EA look (~5–7 weeks)** | **≈ 18 % (12–25 %)** | Mostly T-E zero-shot temperature |
| **Any scoped SOTA label by Test-B (Q3 2027)** | **≈ 25 %** | – |
| M6 original H-G1 (both temperature and DO won on the primary) | **≤ 3 %** | – |

**Most likely outcome.**
- "Competitive" on most tracks, with better absolute skill. A speculative guess is a 0.1–0.2 °C lower track-N temperature RMSE *for both* the NN and the GBM, from grab-sample training.
- Coherent diel and 7-DADMax/DO-min outputs, and a physically interpretable parameter map (D2).
- A transparent negative or positive answer on cross-country transfer.

**What would change these numbers.**
- An EA group count ≥ 100 (raises T-E power).
- Gate G-D1a showing a > 5 % inner-MSE drop from grab samples for D1 but not for G2.
- D2 smoke timing.

---

## Appendix A: provenance of the [RV] numbers (this session; read-only)

- **Test, track N and track S numbers:**
  - `docs/preregistration/hydromet_test_results.md`;
  - `RESULTS.md` (2026-11-08/09 M6 lines);
  - `$SENTINEL_RUNS/hydromet/strata_val_v2/strata_val.csv` (val cell, selection-affected).
- **Sonde label counts:** `processed/sonde_daily/index.parquet`, from `n_days` metadata only, joined to `sonde_split.json` roles. Reserved sites (`sonde_newsites_list.json`) were dropped first.
- **Not-in-split roles:** approximate, by HUC4 from `processed/sites/sites.parquet`. NLDI flow merges were not applied.
- **Powell counts:** `processed/labels_powell_metabolism/daily_metabolism.parquet` (`nwis_id`, `model_confidence`), joined to roles. No metabolism value was summarised.
- **WQP grab samples:** `processed/wqp_discrete/variable={temp,do}`. `site_class = stream`, `status_class = accepted`, WY from `activity_start_date`; distinct station-days counted. No values were read beyond the filters.
- **WQP logger rows:** `raw/wqp/physchem/*.parquet`, with an approximate re-implementation of QC step 3 as described in §6. Stream site types come from `raw/wqp/stations.parquet`; `USGS-` stations were excluded. A first pass mis-flagged every row (empty strings were counted as set); the reported numbers are from the corrected second pass.
- **Disk:** `df -h /` showed 86 GB free.
- **Libraries:** xgboost 3.1.3 importable; LightGBM not importable under `python -I` (the project environment may differ).
- **Other inputs:** `LEAD_DECISIONS_20261110.md` was read for the compute and amendment rules. `sim_protocol.md` was read for the Powell test-role reservation.

## Appendix B: references and verification status

[S] = found by web search this session; [F] = fetched directly this session; [R01] = from research report 01, unverified here; [D] = from the workflow inputs; [U] = unverified.

**Stream-temperature models:**
- Willard, J.D., Ciulla, F., Weierbach, H., Kumar, V., Varadharajan, C. (2024). Evaluating deep learning approaches for predictions in unmonitored basins with continental-scale stream temperature models. arXiv:2410.19865. https://arxiv.org/abs/2410.19865 ; data doi:10.15485/2448016 [S][F]
- Varadharajan, C., Willard, J., Ciulla, F. (2025). The case for using low-complexity models and ensembling approaches for hydrological predictions. SFS 2025 oral abstract. https://sfs-2025.p.asnevents.com.au/days/2025-05-19/abstract/118778 [F]
- Rahmani, F., Shen, C., Oliver, S., Lawson, K., Appling, A. (2021). Deep learning approaches for improving prediction of daily stream temperature in data-scarce, unmonitored, and dammed basins. *Hydrol. Process.* 35:e14400. https://pubs.usgs.gov/publication/70238318 [S]
- Rahmani, F., Lawson, K., Ouyang, W., Appling, A., Oliver, S., Shen, C. (2021). Exploring the exceptional performance of a deep learning stream temperature model and the value of streamflow data. *ERL* 16:024025. doi:10.1088/1748-9326/abd501 [S]
- Rahmani, F., Appling, A., Feng, D., Lawson, K., Shen, C. (2023). Identifying structural priors in a hybrid differentiable model for stream water temperature modeling. *WRR* 59(12):e2023WR034420. doi:10.1029/2023WR034420 [S]
- Siddik, Turner, Gomez-Velez, Tanim, Kao (2025). Advancing stream temperature prediction with a generalizable large-sample framework across CONUS river reaches. *J. Hydrol.* 666:134780. doi:10.1016/j.jhydrol.2025.134780 ; dataset doi:10.21951/RiTHyMs_V1/2587036 [S]. Author initials [U].
- Diaz, J.A., Oliver, S.K., Gorski, G. (2025). Evaluation of daily stream temperature predictions (1979–2021) across the contiguous United States using a spatiotemporal aware machine learning algorithm. *Environ. Model. Softw.* doi:10.1016/j.envsoft.2025.106655 ; data doi:10.5066/P13DG5CA [S]
- Li, G., Shi, P., Kong, L., White, J., Zhu, S., Sun, Y., Qu, S., Yang, Q. (2026). A Differentiable Physics-Informed Neural Network (DPINN) for national-scale river temperature modelling. EGU26-13760. https://meetingorganizer.copernicus.org/EGU26/EGU26-13760.html [F]
- Jia, X. et al. (2021). Physics-guided recurrent graph model for predicting flow and temperature in river networks. *SDM 2021*, 612–620. doi:10.1137/1.9781611976700.69 [S]
- Topp, S.N. et al. (2023). Stream temperature prediction in a shifting environment: explaining the influence of deep learning architecture. *WRR* 59:e2022WR033880. doi:10.1029/2022WR033880 [S]
- Sadler, J.M. et al. (2022). Multi-task deep learning of daily streamflow and water temperature. *WRR* 58:e2021WR030138. doi:10.1029/2021WR030138 ; data doi:10.5066/P9U0TG8L [S]
- Feigl, M., Lebiedzinski, K., Herrnegger, M., Schulz, K. (2021). Machine-learning methods for stream water temperature prediction. *HESS* 25:2951–2977. https://hess.copernicus.org/articles/25/2951/2021/ [S]
- Weierbach, H. et al. (2022). Stream temperature predictions for river basin management in the Pacific Northwest and Mid-Atlantic regions using machine learning. *Water* 14(7):1032. https://www.mdpi.com/2073-4441/14/7/1032 [S]
- Toffolon, M., Piccolroaz, S. (2015). A hybrid model for river water temperature as a function of air temperature and discharge. *ERL* 10:114011. doi:10.1088/1748-9326/10/11/114011 [S]
- Isaak, D.J. et al. (2017). The NorWeST summer stream temperature model and scenarios for the western U.S. *WRR* 53(11):9181–9205. doi:10.1002/2017WR020969 [S]
- Siegel, J.E., Fullerton, A.H., FitzGerald, A.M., Holzer, D., Jordan, C.E. (2023). Daily stream temperature predictions for free-flowing streams in the Pacific Northwest, USA. *PLOS Water* 2(8). doi:10.1371/journal.pwat.0000119 [S]
- Letcher, B. et al. (2016). A hierarchical model of daily stream temperature using air-water temperature synchronization, autocorrelation, and time lags. *PeerJ* 4:e1727. doi:10.7717/peerj.1727 [S]

**General ML and hydrology methods:**
- Kratzert, F. et al. (2019). Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets. *HESS* 23:5089–5110. doi:10.5194/hess-23-5089-2019 [S]
- Tsai, W.-P. et al. (2021). From calibration to parameter learning: harnessing the scaling effects of big data in geoscientific modeling. *Nat. Commun.* 12:5988. doi:10.1038/s41467-021-26107-z [S]
- Grinsztajn, L., Oyallon, E., Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? NeurIPS 2022. arXiv:2207.08815 [S]

**Dissolved oxygen and metabolism:**
- Zhi, W. et al. (2021). From hydrometeorology to river water quality: can a deep learning model predict dissolved oxygen at the continental scale? *ES&T* 55:2357–2368. doi:10.1021/acs.est.0c06783 [S]
- Zhi, W., Ouyang, W., Shen, C., Li, L. (2023). Temperature outweighs light and flow as the predominant driver of dissolved oxygen in US rivers. *Nature Water*. doi:10.1038/s44221-023-00038-z [S]
- Zhi, W., Klingler, C., Liu, J., Li, L. (2023). Widespread deoxygenation in warming rivers. *Nat. Clim. Change* 13. doi:10.1038/s41558-023-01793-3 [S]
- Blaszczak, J.R. et al. (2023). Extent, patterns, and drivers of hypoxia in the world's streams and rivers. *L&O Letters* 8(3):453–463. https://pubs.usgs.gov/publication/70238795 [S]; DOI 10.1002/lol2.10297 [D]
- Appling, A.P. et al. (2018). The metabolic regimes of 356 rivers in the United States. *Sci. Data* 5:180292. doi:10.1038/sdata.2018.292 [S]
- Appling, A.P. et al. (2018). Overcoming equifinality: leveraging long time series for stream metabolism estimation. *JGR-Biogeosciences*. https://pubs.usgs.gov/publication/70196033 [S, title and record only]
- Guan, Q. et al. (2026). Declining predictions of net ecosystem production in US rivers and streams throughout the 21st century. EGU26-12035. https://meetingorganizer.copernicus.org/EGU26/EGU26-12035.html [F]

**Datasets:**
- Oliver, S.K., Appling, A.P., Watkins, D.D., Atshan, R.A., Read, J.S. (2024). Compilation of multi-agency water temperature observations for U.S. streams, 1894–2022. USGS data release. doi:10.5066/P9EMWZ35 (CC0) [S][F]
- Kratzert, F. et al. (2023). Caravan – a global community dataset for large-sample hydrology. *Sci. Data* 10:61 (4TU listing: CC BY 4.0) [S]; size 24.8 GB [R01]
- Linke, S. et al. (2019). Global hydro-environmental sub-basin and river reach characteristics at high spatial resolution. *Sci. Data* 6:283. doi:10.1038/s41597-019-0300-6 (CC BY 4.0) [S]
- NOAA National Water Model CONUS Retrospective v3.0, AWS Open Data Registry. https://registry.opendata.aws/nwm-archive/ [S]
- Environment Agency Hydrology API reference (water-quality measures; OGL v3.0). https://environment.data.gov.uk/hydrology/doc/reference [F]
- Grundmann, M.H. et al. (2026). TempER (Temperatures in European Rivers). EGU26-7752. https://meetingorganizer.copernicus.org/EGU26/EGU26-7752.html [F]

**Not verified this session:**
- Liu et al. 2024 / 2025 (Transformer vs LSTM) [R01];
- Raymond et al. 2012 k600 coefficients [U];
- Caissie 2006 [D];
- the NorWeST RDS licence wording [D];
- EcoSHEDS database counts [D].
