# M1 SENTINEL-Sonde: architecture research and SOTA plan

**Date:** 2026-11-10. **Type:** read-only research synthesis, for the lead's decision. AI-assisted.
**Not:** a pre-registration, a result, or an edit to any protocol. Every protocol change below is a *proposal*
that the lead writes as a dated amendment before any result on it exists (`sota_campaign.md` rule 2).
**Scope:** M1 forecasting (1–14 d numeric forecasts of USGS sonde series) and its early-warning use (EW).

**How the evidence was checked (read this first).**
- **Repo numbers.** Every M1 number was re-read this session from run files. The main sources:
  - `sentinel_runs/sonde/cyc3_score_members_ens3w/{m1_bar_val.csv, median_site_skill_ci_val.csv}`;
  - `sentinel_runs/sim/eval/base/hindcast_summary.json`;
  - `raw/openmeteo_nwp/availability.json`;
  - split and tier indexes, read for counts only.

  No new-site value was read. The only new-site figures used are metadata counts already in the protocols.
- **Web search.** The session's shared search budget ran out after **one** query (Nearing 2024). The other
  references were checked by **fetching the publisher page, the Crossref DOI record or the dataset landing page
  directly** this session. That was a direct fetch, not a search. Tags:

  | Tag | Meaning |
  |---|---|
  | **[S]** | Found by web search this session |
  | **[F]** | Fetched directly this session (publisher page, DOI record or landing page) |
  | **[R01]** | Carried from `docs/research/01_sensor_timeseries.md` (verified 2026-10-22 in an earlier session). **UNVERIFIED in this session.** |
  | **[U]** | Unverified |
- **Design inputs truncated.** The workflow's design and critique inputs reached me truncated: I saw one full design
  (PhiDA-Cast) and none of the critics' text. Section 4 therefore applies my own critique, grounded in the repo
  evidence and the fetched literature. It names each change it makes to PhiDA-Cast.

---

## 0. Decision summary

1. **Simple ML ties because the task, as posed, is close to its information ceiling. It is not mainly an
   architecture problem.**
   - Every M1 model sees only lagged daily data and lag-1 *observed* weather.
   - On val, the candidate ens3w beats damped-anomaly persistence by only **3–12 % CRPS**.
   - An equal-weight ensemble of *baselines* (GBM-w + LSTM-w + zero-shot Chronos-2) lands within **±0.01** of it on
     every variable, and is ahead on temperature.
   - An oracle with **observed future weather** (SENTINEL-Sim) reaches **+0.25 to +0.28** over damped anomaly for
     temperature at h = 3–7 d, against ens3w's +0.03 to +0.05.
   - The missing ingredient is information about the future (weather forecasts) and about sub-daily dynamics, not
     model capacity.
2. **The current architecture is reasonable but not optimal for the task the user wants.**
   - It has no forecast branch, cannot take known-future inputs, and has no sub-daily DO context in the forecaster.
   - It weights members equally across variables, which hurts temperature.
   - It is not over- or under-parameterised. A bigger or more exotic encoder (Mamba, xLSTM, Transformer) would not
     help without new inputs.
3. **What to build, ranked** (details in §4):

   | Rank | Design | Score /10 | GPU-h |
   |---|---|---|---|
   | 1 | **SONDE-HF**: multi-resolution hindcast–forecast multi-task network with masked-mean NWP embedding | 7.0 | ≈ 18 |
   | 2 | **E2**: CV-weighted SENTINEL ensemble (with a matched baseline-ensemble control) | 6.0 | ≈ 2 |
   | 3 | **PhiDA-Lite**: physics-anchored heat + DO forecast branch on the SONDE-HF backbone, gated | 5.5 | ≈ 20 |
   | 4 | **Chronos-2-NWP**: fine-tuned with known-future NWP covariates | 4.5 | ≈ 12 |
   | 5 | **Upstream context**, as features for every model (GAT only if the metadata gate passes) | 3.0 | ≈ 6 |

   Two mandatory new references come first:
   - a **Nearing-style hindcast–forecast LSTM (R-NH)**, the strongest *published* method for NWP-driven multi-day
     hydrologic forecasting;
   - a **physics-feature GBM (PF-GBM)**.

   Without them no SOTA label is possible (§15.1(i)).
4. **Data lever #1: GEFSv12 archived ensemble forecasts.**
   - Sources: the reforecast (2000–2019) plus the operational archive (2020–2025).
   - They give real forecast error in the training years and turn the *whole* new-site issue window into
     forecast mode. Today only issues on or after 2021-03-30 are in forecast mode, with temperature only.
   - Cost: transfer is roughly 0.5–2 TB, streamed and deleted file by file (**UNVERIFIED**: size one file with a
     HEAD request first). Point extractions kept on disk: < 1 GB.
5. **Harder tracks worth pre-registering now, before any result:**
   - **(a)** M1-N forecast mode over the full issue window, with GEFS;
   - **(b) M1-H**: sub-daily DO and hours-of-hypoxia forecasting at new sites;
   - **(c)** forcing-defined heat and storm strata;
   - **(d)** a StreamPULSE new-source block.
6. **Honest outlook.**
   - M1-N primary bar (≥ 3 of 5 variables at every h ∈ {1, 2, 3, 5, 7}, 3 seeds, Holm): **≈ 10 %** with GEFS,
     ≈ 6 % without.
   - Any §15.1 label for M1 within about 6 weeks: **≈ 30 % (20–40 %)**. The most likely labels are SC on M1-N, M1-H
     DO, or a first-in-class numeric metabolism forecast.
7. **Recommended first step (no GPU).**
   - The lead writes the **M1-N v0.6 amendment** and the **M1-H protocol**, from the items in §8 step 0. This must
     happen before any NWP-based M1-N validation number exists. The val-role NWP pull finishes in about 1.8 days,
     so this is time-critical.
   - In parallel, size one GEFS reforecast file by HEAD request and decide the transfer.

---

## 1. Diagnosis: why simple ML wins or ties

All numbers are **VAL** (WY2022–23) and were selected on val (caveat W6: FT-Chronos and the LSTM early-stopped on val).
The metric is median per-site CRPS skill with a basin-group bootstrap.

### 1.1 The learned models add little over a two-parameter baseline

Skill vs damped anomaly, "all" cell (all roles, val block, 44–67 basin groups), seed s1.

| Variable | ens3w h1 / h3 / h7 | Baseline-only ensemble control3w h1 / h3 / h7 | GBM-w h1 / h3 / h7 | LSTM-w h1 / h3 / h7 |
|---|---|---|---|---|
| Temperature | .090 / .045 / .031 | **.096 / .046 / .035** | .074 / .019 / .015 | .071 / .036 / .022 |
| DO mean | .099 / .073 / .054 | .090 / .069 / .051 | .051 / .026 / .020 | .062 / .039 / .027 |
| DO min | .104 / .080 / .055 | .101 / .079 / .056 | .062 / .036 / .026 | .089 / .057 / .034 |
| SC | .120 / .111 / .095 | .117 / .094 / .077 | .063 / .049 / .036 | .048 / .050 / .041 |
| Turbidity mean | .080 / .062 / .052 | .085 / .056 / .047 | .060 / .035 / .026 | .052 / .034 / .028 |
| Nitrate (6 groups) | .099 / .106 / .105 | .056 / .077 / .079 | −.015 / −.025 / −.043 | −.036 / .013 / .028 |

**Reading:**
- Most of ens3w's lead over any single model comes from **ensembling**. A baseline-only ensemble recovers it:
  - margin over control3w ≤ +0.017 for SC;
  - about 0 for DO;
  - negative for temperature.
- Only **SC** (and nitrate, on 6 groups) keeps a margin over the baseline ensemble. That margin comes from Chronos-FT.

### 1.2 The bar is not met where it counts

Val-role ("unseen") cell, 8–13 basin groups per variable (nitrate 3). Horizons passing out of 5 (h ∈ {1, 2, 3, 5, 7})
against the best of all references and strict controls:

| Seed | SC | Turbidity mean | pH | Nitrate | DO mean | DO min | Temperature |
|---|---|---|---|---|---|---|---|
| s0 | 4 | 1 | 0 | 1 | 0 | 0 | 0 |
| s1 | 5 | 2 | 2 | 1 | 0 | 0 | 0 |
| s2 | 5 | 2 | 2 | 1 | 0 | 0 | 0 |

- **Temperature, h1, s1:** −0.014 [−0.018, −0.010] vs control3w, i.e. *significantly worse*. The equal-weight
  Chronos members hurt temperature.
- **High-power cell** ("all" roles, s1): bar met for SC 5/5, nitrate 4/5 (6 groups), pH 4/5, turbidity mean 2/5,
  DO mean 1/5, DO min 0/5, temperature 0/5.

### 1.3 Where the missing information is

**Oracle future weather.** SENTINEL-Sim (`hindcast_summary.json`, 133 val sites, 3 seeds) is driven by **observed**
future weather and discharge, so it is an oracle. Pooled CRPSS vs damped anomaly:

| Target | h3 | h7 |
|---|---|---|
| Temperature | **+0.249 [0.203, 0.326]** | **+0.280 [0.233, 0.372]** |
| DO mean | +0.113 [−0.006, 0.159] | +0.094 [−0.021, 0.129] |

Sim uses pooled CRPSS and M1 uses the median per-site skill, so the comparison is indicative only. Even so, the
temperature headroom with future weather is about **5–9×** what ens3w achieves.

**Why the oracle overstates what forecasts deliver.**
- Operational NWP is not the oracle. **Konold et al. 2026 (BiasCast)** [F] measured the loss on 451 basins:
  - median NSE fell from **0.58 to 0.33** when a reanalysis-trained LSTM was fed real forecasts;
  - a sequential forecast LSTM recovered to **0.63**;
  - adding recent observations reached **0.71**, above the reanalysis baseline of 0.69;
  - feeding archived forecasts into the *hindcast* did not help (0.62 vs 0.63).
- So the gain depends on *how* the forecasts enter the model (forecast branch, forecast-aware training,
  observation anchoring).

**What is on disk now** (`availability.json`):
- GFS 2-m temperature from 2021-03-24;
- GFS precipitation and shortwave only from 2024-01-20;
- ECMWF from 2024-02.

The current M1-N pull is temperature only, so rain-driven targets (turbidity, SC dilution, nitrate flushes) get almost
no forecast information before 2024.

### 1.4 Why the GBM keeps up

- **Same inputs.** At 1–7 d the target is close to "seasonal climatology plus a decaying anomaly plus
  weather-driven departures".
  - With lag and rolling features, LightGBM represents the first two as well as any sequence model.
  - With only *past* weather, the third is unpredictable for every model.
- **Daily resolution hides the dynamics where sequence models help.**
  - DO minimum and hypoxia are nocturnal: median hypoxia event length about 3 h (Blaszczak et al. 2023, [R01]).
  - The forecaster's LSTM is daily-only. Only the EW stack sees 6 hand-made hourly features.
- **Small, noisy test cells.**
  - 8–13 val-role basin groups give CI half-widths of about ±0.01–0.03, larger than any architectural effect seen.
  - The primary bar needs a CI above 0 at **every** h, *including h = 1*. That is where persistence is near its
    ceiling and future weather adds least.
- **The reference set is the maximum over 9+ models and 3 baseline ensembles.** Beating the pointwise best
  reference at every h is a much stricter test than beating any one of them.

**Conclusion** (`sota_campaign` rule 5, "diagnose before adding capacity"):
- No new encoder will pass the bar on the current inputs.
- Levers that can help, roughly in order of evidence:
  1. future-weather information, properly integrated;
  2. sub-daily context and targets;
  3. multi-task coherence;
  4. physically constrained extrapolation in extremes.

---

## 2. Literature: what the state of the art says

### 2.1 Architecture evidence most relevant to M1

| # | Work | What it shows | Use for M1 |
|---|---|---|---|
| L1 | **Nearing et al. 2024**, *Nature* 627:559–563, doi:10.1038/s41586-024-07145-1 [S] | Hindcast LSTM (about 1 yr of weather) hands its state to a forecast LSTM (7 d of NWP), with a mixture-density head; 5,680 gauges; reliable extreme-event prediction out to 5 d; operational in 80+ countries | Template for the forecast branch, and the **strongest published method** to re-implement as reference R-NH |
| L2 | **Konold, Feigl, Podest, Klingler & Schulz 2026**, "BiasCast", *HESS* 30:5067–5096, https://hess.copernicus.org/articles/30/5067/2026/ [F] | 451 LamaH-CE basins. Median NSE: reanalysis → forecast drop 0.58 → 0.33; sequential forecast LSTM 0.63; encoder–decoder 0.57; transfer learning 0.44; **+recent discharge 0.71**; archived forecasts in hindcast no help | Train forecast-aware (real or realistically perturbed forecasts), keep the recent target observations, use a sequential forecast branch |
| L3 | **Gauch, Kratzert, Klotz, Nearing, Cohen & Gilon 2025**, *HESS* 29:6221–6235, https://hess.copernicus.org/articles/29/6221/2025/ [F] | 531 CAMELS basins: a **masked mean** of per-product embeddings handles missing forcing products, best "by a small margin" | Mix GFS-T (2021+), GFS-P/SW (2024+), ECMWF (2024+) and GEFS members without zero-fill |
| L4 | **Nearing et al. 2022**, *HESS* 26:5493–5513, doi:10.5194/hess-26-5493-2022 [F] | Autoregression on recent observations: median NSE 0.796 → 0.879. A **missing-data flag** is "transformative"; AR with the flag ≥ data assimilation, at a quarter of the cost | Prefer AR inputs plus observed/missing flags over a separate learned-gain DA module |
| L5 | **Klotz et al. 2022**, *HESS* 26:1673–1693, doi:10.5194/hess-26-1673-2022 [F] | CMAL best overall; UMAL sharpest but over-confident at high flows; GMM under-confident; MC-dropout worst | CMAL head as an ablation against the existing monotone 19-quantile head |
| L6 | **Gauch et al. 2021**, *HESS* 25:2045–2062, doi:10.5194/hess-25-2045-2021 [F] | Multi-timescale LSTM (hourly + daily): best median NSE 0.81 (daily) / 0.81 (hourly); about 40× faster inference than a naive hourly LSTM | Template for the hourly branch (DO min, hypoxia hours) |
| L7 | **Kratzert, Gauch, Klotz & Nearing 2024**, *HESS* 28:4187–4201, doi:10.5194/hess-28-4187-2024 [F] | Train on all basins: performance rises with every basin added up to 531; HUC- or cluster-grouped training is worse than all-basin | Train on every train-role site plus extra networks (parity), never per site or per region |
| L8 | **Liu, Shen, O'Donncha, Song, Zhi, Beck, Bindas, Kraabel & Lawson 2025**, *HESS* 29:6811–6828, https://hess.copernicus.org/articles/29/6811/2025/ [F] | 11 Transformers vs LSTM: LSTM wins regression (global streamflow KGE +0.11) and 1-d forecasts (0.89 vs 0.81). **DO: LSTM KGE 0.70, best attention model within about 0.01**, and attention better on low extremes. Attention catches up and overtakes on autoregression (30–60 d), spatial CV and zero-shot | No reason to switch the core to a Transformer at 1–7 d. Attention becomes interesting for long horizons and zero-shot transfer |
| L9 | **Baste, Klotz, Acuña Espinoza, Bardossy & Loritz 2025**, *HESS* 29:5871–5891, doi:10.5194/hess-29-5871-2025 [F] | 196 CAMELS-CH catchments: LSTM theoretical ceiling 73 mm/d vs a training maximum of 183. A hybrid rose about 51 % from the 50-yr to the 300-yr event, the LSTM about 6 %. Median NSE: LSTM 0.84, hybrid 0.79 | Physics helps *extrapolation* and costs average skill. Use it for extreme strata, not the headline |
| L10 | **Frame et al. 2022**, *HESS* 26:3377–3392, doi:10.5194/hess-26-3377-2022 [F] | On held-out CAMELS extremes the LSTM stayed accurate; **mass-balance constraints reduced skill** | Counter-evidence: avoid hard conservation constraints on the main output path |
| L11 | **Mosaffa, Pappenberger, Prudhomme, Chantry, Rüdiger & Cloke 2026**, *HESS* 30:2079–2092, doi:10.5194/hess-30-2079-2026 [F] | 530 Upper-Danube subbasins: LSTM–GAT NSE 0.61 / KGE 0.65 vs LSTM 0.46 / 0.49 (discharge, dense nested network) | Graph routing helps on dense nested networks. Most M1 sites are not nested (median upstream-edge distance 116 km) |
| L12 | **Ansari et al. 2025**, "Chronos-2: From Univariate to Universal Forecasting", arXiv:2510.15821 [F] | Group attention across series and covariates; in-context learning; SOTA on fev-bench, GIFT-Eval and Chronos Bench II (author-reported); large gains on covariate tasks | Known-future NWP covariates in a fine-tune: the Chronos-2-NWP member and a mandatory covariate-enabled reference |
| L13 | **Das, Kong, Leach, Mathur, Sen & Yu**, TiDE, arXiv:2304.08424 [F] | MLP encoder–decoder with covariates; matches or beats Transformers on LTSF; 5–10× faster | Optional cheap covariate-aware deep reference |

### 2.2 Physics-guided and water-quality forecasting evidence

| # | Work | What it shows | Use for M1 |
|---|---|---|---|
| P1 | **Zwart, Oliver, Watkins, Sadler, Appling, Corson-Dosch, Jia, Kumar & Read 2023**, *JAWRA* 59(2):317–337, doi:10.1111/1752-1688.13093 [F] | Process-guided DL plus DA, 7-d forecasts of daily-max stream temperature (Delaware River Basin, DRB): MAE about 1.1–1.4 °C (d1), 1.4–1.9 °C (d7); DA "modestly" reduces error; **65–82 % coverage of the 90 % interval** | Closest WQ forecasting analogue. It is temperature only, and its under-dispersion warns us to calibrate |
| P2 | **Rahmani, Appling, Feng, Lawson & Shen 2023**, *WRR* 59(12), doi:10.1029/2023WR034420 [F] | Differentiable stream-temperature model with structural priors. **Internal variables poorly constrained by a single target**; recency-weighted air-temperature averaging helps | Keep the physics small (heat, DO) and supervise latents (Powell GPP/ER) |
| P3 | **Read et al. 2019**, *WRR* 55(11):9173–9190, doi:10.1029/2019WR024922 [F] | Process-guided deep learning (PGDL) for lake temperature, 68 lakes: median RMSE 1.65 °C vs DL 1.78 vs process-based 2.03. The extrapolation gain appears only when pretraining data are more variable than training data | Physics pretraining helps modestly |
| P4 | **Appling et al. 2018a**, *Sci. Data* 5:180292, doi:10.1038/sdata.2018.292 [F] | GPP, ER and K600 for 356 US rivers, 490,907 site-days, 2007-10 to 2017-01 | Latent supervision and a new numeric target (on disk) |
| P5 | **Appling, Hall, Yackulic & Arroita 2018b**, *JGR-Biogeosci.* 123(2):624–645, doi:10.1002/2017JG004140 [F] | Multi-day state-space metabolism with K600 pooled on discharge; up to 70 % error reduction against equifinality | Pool K600 on Q; use long windows |
| P6 | **Olsson et al. 2025**, *Ecol. Appl.* 35(1):e70004, doi:10.1002/eap.70004 [F] | 100,000+ NEON lake forecasts: **10/29 temperature and 6/15 DO models beat the null**; air-temperature covariates and process models did best | At longer horizons climatology wins. Do not expect X4 (21–30 d) wins |
| P7 | **Zhi, Feng, Tsai, Sterle, Harpold, Shen & Li 2021**, *ES&T* 55(4):2357–2368, doi:10.1021/acs.est.0c06783 [F] (metadata only; numbers not re-verified) | Continental DO LSTM from hydrometeorology | Prior art for DO from weather |
| P8 | **Revach et al. 2022**, KalmanNet, *IEEE TSP* 70:1532–1547, doi:10.1109/TSP.2022.3158588 [F] | Learned Kalman gain for partially known dynamics | Basis of PhiDA's E3 gain. Demoted to an ablation, because L4 shows AR with flags is as good |

**[R01] items used below (UNVERIFIED this session; verified 2026-10-22 per `docs/research/01`):**

| Item | What it reports |
|---|---|
| Blaszczak et al. 2023, *L&O Lett.* 8:453 | Hypoxia median event 3 h, mostly nocturnal |
| Pandit et al. 2025, *WRR* | Nitrate KGE 0.60 temporal, 0.18 in ungauged basins |
| Smith et al. 2024, *ES&T* | SC KGE: random forest 0.67 vs RGCN 0.64 (DRB) |
| Sadler et al. 2022, *WRR* | Multi-task helps discharge at 56/101 sites single-site, only 17 multi-site; naive loss weights hurt temperature |
| Kemper et al. 2025, *JAWRA* | LightGBM plus NWM flow forecasts "relatively skilful" for 1–3 d turbidity |
| Song et al. 2024, *J. Hydrol.* | SSC R² 0.55 in new basins |
| TIME "Water_Quality_Darwin" (Qiao et al., ICML 2026) | TSFMs vs persistence on 15-min sondes |

Also **[U]** in this session: Guan et al. 2022 (*MWR* 150:647, GEFSv12 reforecast paper), Raymond et al. 2012 (K600),
Toffolon & Piccolroaz 2015 (air2stream), Benson–Krause DO saturation.

### 2.3 What the literature implies

1. **The best-evidenced design for multi-day forecasting with weather forecasts** is a multi-site hindcast →
   forecast LSTM, trained on many sites, forecast-aware, anchored on recent observations, with a mixture or
   quantile head (L1, L2, L4, L5, L7). **M1 has none of the forecast-side pieces.**
2. **Architecture swaps without new information do not help** at 1–7 d. LSTM ≥ Transformers on regression and short
   forecasts, DO included (L8). Mamba, xLSTM and similar have no WQ evidence, and their kernels fail on Turing
   (`docs/research/01` a.3).
3. **Physics is a bet on extrapolation and interpretability, not on headline CRPS** (L9, L10, P2, P3). Its best use
   is the extremes stratum and numeric GPP/ER forecasts.
4. **Graphs help on dense nested networks** (L11) and lose to tree models on regional SC (Smith 2024, [R01]).
5. **Multi-day WQ forecasts are under-dispersed in the best published system** (P1: 65–82 % in a 90 % PI). M1's
   intervals are near nominal on val, a strength to keep.

---

## 3. Is the current architecture optimal? (honest)

**For the task it was given (daily, lag-only inputs, 1–7 d): close to optimal.**
- It is an equal-weight ensemble of a fine-tuned TSFM, a multi-site LSTM and a zero-shot TSFM.
- It already sits at the information ceiling of that task.
- Nothing in the literature suggests that another encoder, given the same inputs, would beat the baseline ensemble
  by the +0.015 to +0.03 that the M1-N bar needs at 24–54 HUC4 clusters.

**For the task the user wants (beat strong baselines on a hard, operationally meaningful forecasting problem): not
optimal.** Specific gaps, each fixable:

| # | Gap | Evidence | Fix |
|---|---|---|---|
| G1 | No known-future inputs; no forecast branch | §1.3; L1, L2 | Hindcast → forecast hand-off with NWP members (SONDE-HF) |
| G2 | Weather enters as lag-1 *observed* gridMET only; training never sees forecast error | L2: 0.58 → 0.33 NSE collapse | Forecast-aware training: real reforecasts (GEFS) or a lead/season-dependent error model |
| G3 | Daily-only forecaster; DO min / hypoxia are sub-daily | Blaszczak [R01]; EW needs hand-made hourly features | Hourly hindcast branch (L6) plus hourly-derived targets |
| G4 | Separate heads per variable × horizon; no cross-variable coherence (DOmin ≤ DOmean ≤ DOmax; DO ↔ T) | Design | Multi-task heads with ordering constraints; shared forecast state |
| G5 | Equal member weights across variables; Chronos members hurt temperature at h1 | §1.2: −0.014 [−0.018, −0.010] | Per-variable weights fixed by CV, **and the same rule for the baseline-ensemble control** (parity) |
| G6 | Early stopping on the val block (W6); selection on 8–13 groups | Handoff | Inner-train early stopping plus 5-fold basin-group CV (already M1-N lever 1) |
| G7 | GBM uncertainty is homoscedastic | Diagnosis | Not a candidate problem, but the GBM reference needs quantile LightGBM so it is not under-tuned |
| G8 | No physically constrained extrapolation (heatwaves past the training range) | L9 | PhiDA-Lite, scored on a forcing-defined heat stratum |

**Bottom line.** The "simple ML ties" result is real, informative, and should be **reported, not engineered away**.
New architecture work is justified only together with new information (G1–G3). That is also where comparator parity
bites: every baseline gets the same NWP. The open question is whether sequence models use forecast *trajectories*
better than a GBM given summary features. Section 8 schedules a cheap, train-only test of exactly that before the
expensive designs.

---

## 4. Ranked architecture proposals

**Scoring.** Each design gets a score out of 10 for expected contribution to a §15.1 label per GPU-hour, adjusted for
risk. GPU-hours assume RTX 2080 Ti jobs of ≤ 2 h on GPU 0/1 via the sonde runner's own lock, fp16 autocast plus
GradScaler for the LSTMs, physics in fp32, no bf16, and **≤ 5 GB VRAM** (GPU 0 was at 10.7/11 GB used by co-tenants
when checked).

**Common to every design.**
- Train on train-role split sites × train block only, plus extra networks given to every comparator.
- Reference climatologies come from the site's first 2 years (M1-N regime parity).
- Selection: 5-fold basin-group CV over train+val roles × val block.
- **Validation uses archived *issued* forecasts** (M1-N C-N2).
- Every loader goes through `sentinel/data/reserved.py`.

### 4.0 Mandatory new references (built first; they are not candidates)

| Ref | What | Why mandatory | GPU-h |
|---|---|---|---|
| **R-NH** | Nearing-style hindcast–forecast LSTM (NeuralHydrology lineage): 365-d daily hindcast LSTM → state hand-off → 14-d forecast LSTM over NWP inputs; CMAL head mapped to 19 quantiles; multi-site; per-target. Same inputs, NWP products and statics as the candidates; daily only, single-task, no physics | The strongest *published* method for this problem class (L1). §15.1(i) needs it. Any SENTINEL design must beat it, so it is the bar for "architecture adds something" | ≈ 5 (2 configs × 5 folds × 0.4 h, plus 3 seeds) |
| **GBM-NWP** | Current GBM-w plus NWP features per lead: forecast Tair mean/max/min (and P, SW where present), **cumulative and exponentially weighted forecast degree-days over 1..h**, GEFS member mean/spread, product flags. Quantile LightGBM (not homoscedastic) | Strongest simple baseline with parity. Without trajectory summaries it would be under-tuned (§15.5 item 4) | CPU |
| **PF-GBM** | GBM-NWP plus uncalibrated physics features: regional-default air2stream water-temperature forecast, DO_sat(T_forecast), Raymond K600 from Q and slope, two-end-member SC mixing estimate | Tests whether end-to-end physics beats physics-as-features. Must be beaten for PhiDA-Lite to mean anything | CPU |
| **Chronos-2-cov ZS** | Zero-shot Chronos-2 with NWP as known-future covariates | Already required by M1-N §3 | ≈ 2 |
| Frozen ens3w, LSTM-w, ZS Chronos-2 uni/multi, TimesFM-2.5, damped anomaly, climatology, persistence | Unchanged | M1-N §3 | prediction only |
| **Strict control E2-B** | Baseline-only ensemble (GBM-NWP + LSTM-w-NWP + R-NH + Chronos-2-cov), weighted by **the same per-variable CV rule as E2** | So that CV-weighted ensembling is not credited to SENTINEL | ≈ 0.5 |

### 4.1 Rank 1: SONDE-HF, a multi-resolution hindcast–forecast multi-task network. Score 7.0/10

**Idea.** Combine three published pieces in a form not published for water quality (novelty **UNVERIFIED**, since the
search budget is exhausted):
- the hindcast–forecast hand-off (L1);
- the multi-timescale hourly + daily hindcast (L6);
- masked-mean multi-product NWP (L3).

On top of them, a **multi-task, ordering-constrained head** covers every co-located variable and the derived
sub-daily targets of §5.

**Architecture** (about 1.3 M parameters).

*Hindcast inputs.*
- **Daily hindcast:** LSTM 1×256 over 365 d. Per variable v ∈ {T, DOmean, DOmin, DOmax, SC, SCmax, turb, turbmax,
  NO3, pH, logQ, gage}:
  - anomaly vs the 2-yr reference climatology, site-normalised level, observed/missing flag (L4);
  - plus gridMET lag-1 (8 channels), DOY sin/cos and day length.
  - Statics (≈ 24: lat/lon, elevation, drainage area, NHDPlus slope, % urban/ag/forest, impervious, baseflow index,
    site type, HUC2 embedding) pass through an MLP (24 → 32) concatenated at every step.
- **Hourly hindcast** (sMTS-style): LSTM 1×128 over the last 14 d × 24 h of T, DO, SC, pH and turbidity (+ masks,
  solar elevation). It is initialised from the daily LSTM's state at d − 14. Sites without an hourly tier take a
  learned "no-hourly" state plus a flag, so every DV-only site still works.

*Forecast side.*
- **NWP embedding:** per product p (Open-Meteo GFS-T; GEFS members; ECMWF; climatology fallback) and per lead k:
  Linear(fields_p,k) → 32. Then a **masked mean over available products** (L3). Sources disagree in period, so there
  is no zero-fill.
- **Forecast branch:** LSTM 1×128 over k = 1..14, initialised by Linear(h_daily ‖ h_hourly) → (h0, c0), the
  hand-off (L1).
  - Inputs per k: NWP embedding, clear-sky shortwave and day length (deterministic), and a k one-hot.
  - With M GEFS members, the branch runs per member (batched). Member quantiles are pooled as a mixture and
    re-quantiled to 19 levels, so uncertainty propagates through the nonlinear dynamics.

*Heads.*
- One head per target t and lead k, giving 19 monotone quantiles (location plus softplus increments, as today), with
  the existing learnable damped-anomaly anchor.
- **Ordering constraints:**
  - DOmin = DOmean − softplus(·) and DOmax = DOmean + softplus(·);
  - Tmin ≤ Tmean ≤ Tmax the same way;
  - hypoxia hours ∈ [0, 24] via 24·sigmoid.
- CMAL head as a pre-declared ablation (L5).

**Loss.**
- Σ_t,k w_t · pinball₁₉. w_t is per-target inverse-scale normalisation, fixed before training (Sadler: naive weights
  hurt temperature [R01]).
- Plus a per-target "primary-only" ablation.

**Training.**
- **Stage 1, perfect-prog with realistic error.** Observed gridMET for the target days, plus lead- and
  season-dependent NWP error drawn from:
  - (a) GEFS reforecast-minus-gridMET residuals over WY2008–19 at train-role sites, if GEFS is approved; or
  - (b) an AR(1) error model fitted on the 2021-03-24..2021-09-30 GFS overlap at train-role sites (spring–summer
    only; the protocol already records this limitation).
- **Stage 2, if GEFS is approved.** Fine-tune on real reforecasts over the train block. BiasCast-style training on
  real forecasts is the evidence-backed fix (L2).
- Optimiser: AdamW lr 1e-3, batch 256, grad clip 1, fp16 autocast. Early stopping on an **inner** 15 % of train-role
  basin groups (fixes W6).

**Inductive bias a GBM with identical inputs lacks.**
1. It integrates the *whole* forecast trajectory with learned, site-conditioned memory. Water temperature responds to
   an exponentially weighted history of air temperature (P2: recency weighting helps). GBM-NWP gets only
   pre-specified cumulative summaries.
2. Its sub-daily state feeds DOmin and hypoxia-hours forecasts.
3. Shared multi-task representation across about 1,600 train-role sites and 12+ targets, with coherent ordered
   outputs.
4. Member-wise propagation of ensemble NWP.

**Honest caveat.** At h = 1–2, persistence dominates and the GBM should tie. The design wins only if the gains at
h = 3–7 are large *and* h = 1 is not lost. Mitigations:
- the damped-anomaly anchor;
- AR inputs with missing flags;
- a pre-declared rule: the candidate for h = 1 is the same network. There is no per-horizon model switching, which
  would be selection on the bar.

**Configurations** (≤ 5 of the M1-N cap of 16):
- hourly branch on/off;
- CMAL vs quantile head;
- multi-task vs primary-only;
- two residual or hidden sizes.

**Compute.**
- One CV fold is about 35–45 min (LSTM-w took 17–23 min for 0.93 M parameters and 6 M samples; the hourly branch and
  14-step forecast branch add about 1.5–2×).
- Total: 5 configs × 5 folds × 0.6 h ≈ 15 h (one 20-run cycle plus 5), plus 3 final seeds about 2.5 h, plus
  prediction about 1 h. **≈ 18 GPU-h. Peak VRAM ≈ 3–4 GB** at batch 256.

**Fair test (shared by all designs, §4.6).** SONDE-HF must beat **R-NH, GBM-NWP, PF-GBM and the frozen ens3w** at every
primary h. The R-NH comparison isolates what the multi-resolution, multi-task additions buy.

### 4.2 Rank 2: E2, a CV-weighted SENTINEL ensemble. Score 6.0/10

- **Members:** SONDE-HF (3 seeds) + Chronos-2-NWP FT (if built) + PhiDA-Lite (only if it passes its gate, §4.3).
  ens3w members are optional.
- **Weights:** per variable × horizon block (h ≤ 2, h 3–7), by CRPS-optimal non-negative weights on **out-of-fold CV
  predictions**, fixed in the freeze record. This fixes G5.
- **Parity:** the strict control E2-B (§4.0) uses the identical weighting rule on baseline-only members. Without this,
  part of the E2 gain is a weighting gain, not a SENTINEL gain.
- **Why rank 2.** It is the most likely to have the best raw CRPS (ensembles win freshwater challenges: P6, and
  Olsson 2024 [R01]). Its *credited* margin over E2-B equals the margin of its non-baseline members, which is why it
  ranks below SONDE-HF in information value.
- **Compute:** prediction and weighting only, **≈ 2 GPU-h** (CPU for the weights). Counts 1–2 configurations.

### 4.3 Rank 3: PhiDA-Lite, a physics-anchored heat + DO forecast branch. Score 5.5/10

This is the critique-revised **PhiDA-Cast**, built on the SONDE-HF backbone.

**Kept from PhiDA-Cast:**
- **Hypernetwork E2** over [hindcast state ‖ statics], producing bounded physical parameters.
- **Heat core:** equilibrium-temperature form.
  - T_k = T_e,k + (T_{k−1} − T_e,k)·exp(−κ_k), with T_e = c0 + c1·Tair + c_sw·SW + groundwater mixing.
  - κ depends on depth via Q.
  - Exact exponential update, fp32.
- **One-station DO core**, reusing `sentinel/models/sim/physics.integrate_do` by import:
  - hourly GPP = P_max·tanh(SW/I_k);
  - ER = ER20·θ^(T−20);
  - K600 = Raymond prior × learned multiplier;
  - DO_sat(T, P) from the M6 Benson–Krause routine.
  - Outputs: DO mean/min/max, %sat and soft hypoxia hours.
- **Bounded residual** r = c·σ_site·tanh(·) from the SONDE-HF forecast branch, so physics carries the extrapolation.
- **Latent supervision:** Gaussian NLL of GPP / ER / log K600 vs Powell (and StreamPULSE) labels, using their CIs, at
  train-role sites only. The 48 new-site Powell overlaps are never used.

**Changes to PhiDA-Cast (my critique; each is required):**

| # | Change | Reason |
|---|---|---|
| C1 | **Drop** the SC mixing, sediment-supply/turbidity, nitrate end-member and two-reservoir Q modules. SC, turbidity and NO3 stay learned (SONDE-HF heads) | These add about 10 weakly identified parameters. Their forcing (precip forecasts) does not exist before 2024 without GEFS. P2 warns that internals are unidentifiable from one target. Sim's free-regression analogues failed |
| C2 | **Demote the KalmanNet gain (E3) to an ablation.** Default anchoring = the AR inputs with flags already in SONDE-HF | L4: AR with flags ≥ DA at a quarter of the cost. Removes one module and one failure mode |
| C3 | **Pre-declared physics gate before any M1-N val use**, on train-block inner CV (not the val block). (i) The physics-only branch beats damped anomaly for T at h ≥ 3 (CRPS skill CI > 0). (ii) Sim's metamorphic checks M1-consistent ≥ 0.70 and M4-night ≥ 0.80. (iii) Latent GPP Spearman ≥ 0.5 at inner-held-out Powell sites. **If any check fails, PhiDA-Lite is dropped and reported as such** | Sim failed exactly these (0.56–0.72; 0.30–0.71; GPP ρ 0.51). Building on an unvalidated core would spend GPU on a known failure mode |
| C4 | **Streams-only physics.** For lake, estuary and wetland types the core is gated off by type (deterministic, from metadata, not learned), and the output equals SONDE-HF. Report a pre-declared streams-only cell | Hydraulic-geometry depth and one-station metabolism do not apply to lakes or estuaries |
| C5 | **Depth without Q:** drainage-area hydraulic-geometry prior plus a flag, and a pre-declared "has-Q" cell | Not every new site has co-located discharge |
| C6 | **No separate Stage A of 120-d simulation windows.** Instead, the physics-only pinball term (0.5·pinball(μ_phys)) runs throughout SONDE-HF-style training | Halves the GPU cost. Keeps the physics predictive on its own (the gate) |
| C7 | **The claim is relative to SONDE-HF and PF-GBM**, not only to the M1-N references. PhiDA-Lite = SONDE-HF + physics, so SONDE-HF is the exact no-physics ablation | Isolates what physics buys; avoids crediting information gains to physics |
| C8 | **Primary use: two secondary claims.** (a) The X3-heat stratum (forcing-defined, §7). (b) First-in-class *numeric forecasts of GPP/ER with intervals*, scored vs Powell labels at held-out train-split sites. That needs a dated novelty check in `docs/novelty_checks/` and must beat GPP persistence and GPP climatology. Headline CRPS is reported, but is not expected to improve (L9: hybrid NSE 0.79 vs LSTM 0.84) | Matches the evidence: physics helps extrapolation and interpretability, and costs average skill |

**Inductive bias a GBM lacks:**
- linear-in-forcing heat physics and the exact DO_sat(T) curve, which extrapolate past training heat extremes;
- DO minimum from hourly integration of night respiration against reaeration;
- latent processes supervised by 490,907 Powell site-days, which a GBM cannot use as constraints at sites without
  labels.

**Configurations** (≤ 4): residual bound c ∈ {0.5, 1.0} × latent loss weight {0.1, 0.3}. E3 on/off as a 1-seed
ablation only.

**Compute.**
- Gate: inner-CV 1 config × 3 folds × 0.8 h ≈ 2.5 h.
- If passed: 4 configs × 5 folds × 0.8 h = 16 h, plus 3 seeds ≈ 2.5 h.
- **≈ 20 GPU-h total. VRAM ≈ 4–5 GB** (the hourly affine scan over 336 steps × M members; batch 128 plus gradient
  checkpointing as fallback).

The original PhiDA-Cast estimate (52 GPU-h) is cut by C1, C2 and C6.

### 4.4 Rank 4: Chronos-2-NWP, a fine-tuned TSFM with known-future NWP covariates. Score 4.5/10

- **What:** the existing `chronos_ft.py` full fine-tune in "multi" mode. NWP fields per lead are added as **known-future
  covariates** (Chronos-2 supports them natively, L12). Where an hourly tier exists, hourly context is aggregated
  into daily tokens.
- **Why it ranks here:**
  - It is the only route by which the SC advantage (which comes from Chronos-FT) gains weather information.
  - It is cheap to specify, because the code exists.
  - Expected standalone gain over SONDE-HF is small: FT-Chronos already performs about like the LSTM, and
    TSFM-vs-LSTM evidence in hydrology favours trained multi-site LSTMs (Kratzert 2024, L7; [R01] Sun & Sun 2026).
- **Risk:** 120 M parameters in fp32 on Turing. Full FT at batch 128 likely exceeds the ≤ 5 GB co-tenancy budget.
  Use **LoRA plus fp16 autocast**, or batch 32 with gradient accumulation (VRAM ≈ 5–6 GB, **UNVERIFIED**; measure in
  the smoke run).
- **Configurations** (≤ 2): full-FT-small-batch vs LoRA.
- **Compute:** 2 configs × 3 folds (not 5, to save GPU; pre-declared) × 1.2 h ≈ 7 h, plus 3 seeds × 1.2 h,
  plus prediction. **≈ 12 GPU-h.**
- **Also needed as a reference:** the zero-shot covariate Chronos-2 (§4.0).

### 4.5 Rank 5: upstream context (features for all; GAT only behind a gate). Score 3.0/10

- **What:** recent anomalies at flow-connected upstream sensors, travel-time aligned via the NLDI edges on disk
  (`site_network/edges.parquet`: 39,453 edges, median distance 116 km). They are given to **every** comparator as
  features (parity). A GAT message-passing variant of SONDE-HF (L11) is built only if a metadata gate passes:
  ≥ 30 % of M1-N primary sites have an upstream sensor within ≤ 2 d travel time with data in the issue window.
- **Evidence is mixed:**
  - positive for discharge on a dense nested network (L11);
  - negative for regional SC, where a random forest beat RGCN (Smith 2024 [R01]).
- **Governance problem (lead ruling needed).** At new-site issue dates in WY2024–25, upstream sensors are often split
  sites whose WY2024–25 values are test-block data of the original track (Test-A). Using them as *inputs* at M1-N
  look time is not training on them, but it touches a test block.
  - Recommended rule: upstream inputs only for issue dates ≤ 2023-09-30.
  - Or: only from sites outside every split and reserved list.
- **Compute:** ≈ 6 GPU-h if the gate passes; otherwise CPU only (features).

### 4.6 Fair-test plan (shared)

- **Venue:** the pre-registered **M1-N** track (unseen 729-site block; primary population 644 minus hard tier;
  DV-based primary cells DO 51 sites / 24 HUC4 and SC 155 / 54; temperature and turbidity from the integrity table;
  nitrate descriptive). The single look uses the hardened scorer (§6 of `m1n_protocol.md`).
  - **No spent block is touched.** EW-v2 Test-A is spent. M1 original Test-A is report-only, with DO labels exposed.
- **Primary (unchanged, M1-N §4):** median per-site CRPS skill vs the **best** reference at each h; HUC4 bootstrap,
  B = 2,000; lower bound > 0 at every h ∈ {1, 2, 3, 5, 7} for ≥ 3 of 5 variables, in 3 seeds; Holm over 5.
  - The reference set grows by R-NH, GBM-NWP, PF-GBM and Chronos-2-cov. **This only makes the bar harder.**
  - The strict control E2-B is reported alongside.
- **Comparator parity:** identical NWP products, members, bias correction, static attributes, training sites and
  extra networks for every model that can take them. Tuning parity: each learned reference gets as many CV configs
  as the best-tuned SENTINEL family (≤ 5), declared in the amendment.
- **Selection:** 5-fold basin-group CV (train + val roles × val block), archived issued forecasts in every
  validation step, inner-train early stopping. All configurations are counted against M1-N's ≤ 16 per variable
  group: SONDE-HF 5 + PhiDA-Lite 4 + Chronos-2-NWP 2 + E2 weighting 2 + reserve 3.
- **Secondary, with their own CIs:**
  - the numeric-forecast report (M1-N §4a), for auxiliary targets too;
  - streams-only and has-Q cells;
  - forcing-defined strata;
  - S1b / S-T sensitivities;
  - GPP/ER numeric forecasts (PhiDA-Lite).

---

## 5. New numeric predictive targets (multi-task)

Purpose:
- make the task harder and more useful (the user asked for "more predictive targets");
- give the shared encoder more supervision.

All are **numeric 19-quantile forecasts in original units**. Medians and 50/80/90 % PIs go through the existing
`sonde_pi_report.py`. All are **secondary/descriptive** unless a protocol promotes one before any result.

**Counts.** Train/val counts are sites with ≥ 365 days in `processed/sonde_daily/index.parquet`, by split role. They
are counts only, and include every year in the index; the train block is a subset.

| Target | Why it matters | Label source | Train / val sites | Notes |
|---|---|---|---|---|
| **T max daily**, and **7-day average of daily max (7-DADMax)** | Thermal stress (salmonids), regulatory metric | USGS DV `temp_c_max` | 1,551 / 273 | Derived 7-DADMax from DV max |
| **T min daily** | Diel range, cold-water refugia | USGS DV `temp_c_min` | ≈ 1,550 / ≈ 270 | |
| **DO max** and **DO diel amplitude** (max − min) | Amplitude ∝ metabolism; supersaturation risk | DV `do_mgl_max`, `do_mgl_min` | 485 / 91 | Ordering constraint with DO mean/min |
| **DO % saturation** | Removes the temperature effect; the standard criterion in some states | Derived: DO mean + T + barometric pressure from elevation (Benson–Krause [U]) | 487 / 90 | Deterministic transform; no new data |
| **Hours/day with DO < 3 and < 5 mg/L** | The quantity that kills fish; daily min misses duration | Hourly tier (15-min → hourly) | Hourly tier: 423 train / 76 val sites (all variables; DO subset to be counted) | CRPS on counts. Feeds an EW probability descriptively. M1-H primary (§7) |
| **SC daily max** | Proxy for acute chloride (860 mg/L, 1-h) | DV `spcond_uscm_max` | 872 / 162 | |
| **Turbidity daily max** | Treatment-plant intake shocks | DV `turb_fnu_max` | 447 / 60 | More sites than turbidity mean (312 / 44) |
| **log discharge mean** (auxiliary) | Drives every WQ response; helps the encoder | DV `discharge_cfs_mean` | 1,357 / 266 | Auxiliary target and input; never a primary |
| **Gage height mean** (auxiliary) | Flood state where Q is absent | DV `gage_ft_mean` | 519 / 84 | |
| **GPP, ER (g O₂ m⁻² d⁻¹), K600 (d⁻¹)**, with 80 % intervals | Ecosystem function; eutrophication signal (GPP/ER) | Powell Center (Appling 2018a [F]); StreamPULSE | Powell 356 sites / 490,907 site-days; StreamPULSE (processed) 133 sites / 62,383 site-days | Train-role sites only; 48 new-site overlaps never used. Scored by Spearman and CRPS at held-out train-split sites (PhiDA-Lite) |
| **Derived hazard probabilities**: P(DOmin < 3 within h), P(NO3 > 10), P(derived Cl > 230) | Turns numeric forecasts into EW without a separate classifier | From the predictive trajectories | — | Descriptive only (EW-v3-N is non-confirmatory) |

**Not recommended as targets:**
- chlorophyll / phycocyanin at USGS stream sites: few sites, unit inconsistency (RFU vs µg/L, [R01]);
- fDOM: 96 sites nationally [R01];
- ammonium: only via EA, not until M6's split freeze.

**Evidence and risk.**
- Multi-task gains in multi-site hydrology models are **modest and can hurt** with naive loss weights (Sadler 2022 [R01]).
- Hence:
  - fixed per-target normalisation;
  - a "primary-only" ablation counted as a config;
  - the rule that auxiliary targets never re-weight the primary decision.

---

## 6. More data: verified sources, licence and size

M1's new-data cap is **10 GB**. Used so far: about 0.63 GB (new-site block) plus about 1 GB (NWP pull). The host has
87 GB free against a 60 GB floor. The **Test-A look's ≈ 13 GB** of sealed forecasts is a separate lead decision and
competes with this cap.

| # | Source | Verified | Licence | Size (kept / transfer) | What it adds | Parity / blocks |
|---|---|---|---|---|---|---|
| D1 | **NOAA GEFSv12 reforecast**, AWS `noaa-gefs-retrospective`, https://registry.opendata.aws/noaa-gefs-reforecast/ ; layout PDF https://noaa-gefs-retrospective.s3.amazonaws.com/Description_of_reforecast_data.pdf | **[F]**: 2000–2019; once daily 00 UTC; **5 members** (c00, p01–p04); weekly 11 members to +35 d; GRIB2 **one file per variable × init × member**; **0.25° 3-hourly to day 10, 0.5° 6-hourly after**; global grid | NOAA open data: free use; attribution requested; no implied endorsement; modified data not presented as original [F] | Kept: < 1 GB point extractions. **Transfer: ≈ 0.5–2 TB for c00 + 4–5 surface variables, WY2008–19 (UNVERIFIED; HEAD one file first)**, streamed and deleted per file. The global grid means no per-site quota | Real forecast error in **training years** (fixes G2, L2) and forecast mode for new-site issues < 2020. Members give ensemble spread | Given to every comparator. Bias-correct to gridMET on train-block overlap. Use the run from the previous day's 00 UTC for a same-day issue (latency **UNVERIFIED**) |
| D2 | **NOAA GEFS operational archive**, AWS `noaa-gefs-pds`, https://registry.opendata.aws/noaa-gefs/ | Partly **[F]**: the reforecast PDF states GEFSv12 real-time is 31 members, 4×/day to +16 d, and points to this bucket. **Archive start date and holdings UNVERIFIED** (registry page not fetched) | NOAA NODD open data [U] | Similar per-file sizes; kept < 0.5 GB | Precipitation and shortwave forecasts for 2020–2025, which Open-Meteo lacks before 2024. Same GEFSv12 model as D1 | Same rules |
| D3 | **Open-Meteo Previous Runs GFS** (on disk; pull running) | Local `availability.json`: T2m from 2021-03-24; P/SW from 2024-01-20; ECMWF from 2024-02 | Per Open-Meteo terms (see `docs/licenses.md`) | ≈ 1 GB | Already M1-N's forecast mode | Keep as a product inside the masked mean |
| D4 | **USGS hourly tier** (on disk, under-used) | Local: 749 sites (423 train, 76 val, 209 test-role, 41 not in split) | US public domain | 2.0 GB, already counted | Sub-daily DO and T context; hypoxia-hours labels | Test-role and reserved sites excluded by `reserved.py` |
| D5 | **Powell Center metabolism** (on disk) | **[F]** Appling 2018a: 356 rivers, 490,907 site-days | US public domain (USGS release) [R01] | Small | GPP/ER/K600 labels | Train-role only; 48 new-site overlaps excluded |
| D6 | **StreamPULSE** sensor data (bulk), https://data.streampulse.org/ | **[F]**: 741 sites on the portal. ODC-By 1.0 and bulk 509.2 MB from the local policy snapshot and research 01 **[R01]** | ODC-By 1.0; notify contributors of derivative works; 1-yr embargo on new sites [R01] | ≈ 0.5 GB | About 100–200 usable non-USGS/non-NEON sites (estimate) with DO, T, light, depth, SC; a **new-source block** M1 has never loaded (§7 H-6) | De-duplicate within 1 km against USGS split and reserved sites; exclude RiverPulse (D8) test-role sites |
| D7 | **NEON DP1.20288.001** water quality (+ SUNA nitrate) (on disk) | **[F]**: SC, DO, pH, turbidity, fDOM, chl-a (lakes/rivers); 1-min (wadeable) / 5-min (lake, river); collocated SUNA nitrate. Site count and release DOI not read this session | CC BY 4.0 since 2026-06-30 [R01] | On disk (≈ 0.25 GB processed) | 26 usable sites after excluding Fuse Tier-D test domains | **Recommend as training data** (parity), not a test block: about 15 domains is borderline for clusters |
| D8 | **UK EA Hydrology API** sondes, via M6's `processed/ea_wq_sonde` (not built yet), https://environment.data.gov.uk/hydrology/doc/reference | **[F]**: DO (% and mg/L), conductivity, turbidity, ammonium, pH, temperature, chlorophyll at **15-min**; **OGL v3.0** | OGL v3.0 | Shared (M6 cap) | Training data under rule 3, and an **M1-test GB block** (≤ 20 % of non-M6-test groups, pre-2023-10-01), as already approved | Wait for M6's split freeze. No M1 download |

**Not recommended:**
- NERRS SWMP: no redistribution, IP registration [R01];
- FCR/BVR: CC BY-NC [R01];
- HRRR: about 150 MB per hourly CONUS file [R01], too large for the cap and bandwidth;
- larger TSFM weights (Timestral, TimesFM-3) beyond zero-shot references: licence clauses [R01].

**Ranked by expected gain per GB and per hour:**
1. **D1 + D2 (GEFS)**: the only source of real forecast error in training years and of rain forecasts before 2024.
2. **D4 (hourly tier)**: already on disk, zero cost.
3. **D5 (Powell)**: on disk.
4. **D6 (StreamPULSE)**: a new source block.
5. **D8 (EA)**: after M6.
6. **D7 (NEON)**: training only.

---

## 7. Harder tracks: practical justification, unseen block, power

The rules (`sota_campaign.md`):
- each track is pre-registered with its practical reason **before** any result on it;
- the original tracks stay reported;
- each claim names its track;
- spent blocks are never reused;
- each track has its own cap and hardened single look.

| ID | Track | Practical justification (stated now, before results) | Unseen test block | Power (metadata only; to be confirmed before freeze) | Recommend |
|---|---|---|---|---|---|
| **H-1** | **M1-N-F over the full issue window with GEFS** (amend M1-N, v0.6). Forecast mode for *all* new-site issues: reforecast to 2019, operational 2020+, Open-Meteo GFS kept as a product | Operational sonde forecasting uses weather forecasts; new stations need them from day one. Today, issues before 2021-03-30 fall back to past-weather mode, the saturated task of §1 | The existing locked 729-site block (primary 644), WY2010–25, Approved | Forecast-mode issue site-years grow from the post-2021-03 share to all issues. **The lead should compute both shares from the integrity table (counts only) before deciding.** HUC4 counts are unchanged (DO 24, SC 54) | **Yes**, if D1/D2 are approved. Otherwise keep M1-N as is |
| **H-2** | **M1-H: sub-daily DO forecasting and hours-of-hypoxia** (new protocol). Targets: hourly DO and T at +6/12/24/48/72 h, daily hypoxia hours (< 3, < 5 mg/L) at h = 1–7 d, daily DO min | Fish kills and permit violations happen at night-time minima of a few hours (Blaszczak [R01]). Daily means and daily minima from DV cannot say *how long*. Operators read IV, not DV | New-site **hourly IV values** (665 hourly files in the block; DO listed at 393 sites). The failed IV→daily rule (`m1n_protocol.md` §2b) is about deriving DV-equivalent daily minima; **M1-H targets are the hourly IV values themselves**, so that failure does not apply. This must be stated explicitly and approved by the lead | Need ≥ 10 HUC4 with scorable summer rows. The v0.4 table puts primary DO at 199 sites / 61 HUC4 (mostly IV) → **likely powered for CRPS**. Hypoxia-hours > 0 rows are rare: report the conditional cell descriptively | **Yes**. The most "structurally hard for daily baselines" task, with parity: baselines get hourly inputs too (hourly persistence, diel persistence = yesterday's hour + daily anomaly, GBM-hourly, LSTM-hourly, Chronos-2 ZS hourly) |
| **H-3** | **Forcing-defined strata inside M1-N**: X3-heat (NWP Tair at lead h > site reference-window P95) and X1-storm (NWP 3-day precip > P95) | Warnings matter most during heatwaves and storms, where trees cannot extrapolate (L9) | Rows of the M1-N block, selected by forcing only (never by outcome) | Small: the DO primary has 24 HUC4 in total. Pre-declare ≥ 5 HUC4 with stratum rows, else descriptive (as M1-X) | **Yes** (secondary; the PhiDA-Lite claim lives here) |
| **H-4** | **M1-GB**: transfer to the EA England M1-test groups | A new country and network; sewage-impacted waters | M1-test groups drawn after M6's freeze; pre-2023-10-01 | Unknown until M6 publishes. EA quality is mostly "Unchecked" [R01] | Later (blocked on M6) |
| **H-5** | **M1-SP**: StreamPULSE new-source block | A different operator network and QC; research streams with metabolism data, a different population | StreamPULSE sites not within 1 km of any USGS split, reserved or NEON site; locked on download (as `sonde_newsites`) | About 100–200 sites with ≥ 1 yr (estimate). Clusters = HUC4 or ecoregion. Needs ≥ 10 clusters per variable: probably DO and T only | **Yes** (secondary), if D6 is approved. Pre-register before download, with the lock-on-download pattern of the new-site block |
| **H-6** | **EW-H1h: hourly-native hypoxia onsets at new sites** (≥ 2 consecutive hours < 3 mg/L; recommended in `docs/research/09` §0 item 4 on 2026-10-22, before Test-A) | Restores a powered EW test without the failed IV→DV rule | New-site IV, as H-2 | Daily-min rate × 0.5 gives ≈ 46–50 onsets; the ≥ 2 h rule lowers it. **About as likely as not to be underpowered** | Optional. Lead decision: it postdates the IV-rule failure, so it must be disclosed as such (forking-paths risk) |
| H-7 | X4 long horizons (21–30 d) with GEFS weekly 35-d members | Seasonal planning | M1-N block | Powered, but expected null (P6: no model beat climatology at 15–25 d) | Keep secondary; no GPU priority |
| H-8 | N-b virtual sensing, with extra training networks | Unmonitored stations; published PUB nitrate KGE 0.18 [R01] | M1-N block | As registered (≤ 8 configs) | Keep as registered. On val, the N-b GBM was not beaten; more data is the only lever |

**Why these and not "just harder".** A harder task does not automatically favour deep models: climatology wins at
long horizons (P6). The tracks above are hard in ways that need what sequence and physics models add:
- integrating forecast trajectories (H-1);
- sub-daily dynamics (H-2);
- extrapolation (H-3);
- transfer across networks (H-4, H-5).

This matches Liu et al. 2025 (L8): attention and sequence models gain as tasks get harder.

---

## 8. Sequencing within the shared 2-GPU budget and config caps

**Budget assumptions.**
- 2 shared GPUs across about 7 workstreams gives **about 7 GPU-h per day** for M1 on average.
- Jobs ≤ 2 h each, checkpointed every 10 min with `--resume`.
- ≤ 20 GPU runs per cycle; ≤ 16 configurations per variable group (M1-N); CPU ≤ 6 threads.
- **Total plan: about 60 GPU-h**, about 9 GPU-days, about 3–4 weeks of wall time including lock waits.

| Step | When | What | GPU-h | Gate / output |
|---|---|---|---|---|
| **0. Pre-register (lead; no compute)** | **Now, before the val-role NWP pull completes (≈ 1.8 d)** | **M1-N v0.6 amendment** (dated, public commit) containing: (a) candidate families SONDE-HF / E2 / PhiDA-Lite / Chronos-2-NWP with the config allocation (5/2/4/2, reserve 3 ≤ 16); (b) new references R-NH, GBM-NWP (with trajectory features, quantile LightGBM), PF-GBM, Chronos-2-cov ZS; strict control E2-B with the same weighting rule; (c) GEFS forecast-mode extension (if approved) and the bias-correction rule; (d) auxiliary numeric targets as secondary; (e) X3-heat / X1-storm forcing strata, streams-only and has-Q cells; (f) the PhiDA-Lite gate (C3) and its drop rule; (g) the upstream-input date rule. **M1-H protocol** (H-2) with its metadata power check. **M1-SP protocol** (H-5) if D6 is approved | 0 | No M1-N NWP val number exists yet. Registry re-snapshot |
| 0b. Data decisions | Day 0–1 | HEAD-request one GEFS reforecast file (size × count → transfer estimate); lead decides D1/D2. If yes: CPU stream-extract per file (≤ 6 threads) into point extractions < 1 GB | 0 | `df -h` before each step; stop at 60 GB free |
| 1. Information-use test (train only) | Week 1 | On **train-block inner CV only** (not the val block): GBM-NWP vs a smoke SONDE-HF (1 config, 1 fold), perfect-prog plus error model. Purpose: does the sequence model use the forecast trajectory better than GBM summaries? Pre-declared rule: if SONDE-HF's inner-CV gain over GBM-NWP at h = 3–7 is < 0.01 CRPS skill for T and DO, skip PhiDA-Lite and Chronos-2-NWP and put the GPU into H-2 | ≈ 2 | Go / no-go for steps 4–5 |
| 2. References | Week 1–2 | GBM-NWP, PF-GBM (CPU); R-NH CV (2 configs × 5 folds); Chronos-2-cov ZS | ≈ 7 | Reference parity log |
| 3. SONDE-HF CV | Week 2 | 5 configs × 5 folds (one cycle of 20 runs, plus 5 in the next) | ≈ 15 | CV selection by mean CRPS skill over primary h and variables |
| 4. PhiDA-Lite gate, then CV | Week 2–3 | Gate on inner CV (≈ 2.5 h). If passed: 4 configs × 5 folds | ≈ 2.5–18 | Drop rule C3 |
| 5. Chronos-2-NWP | Week 3 (if step 1 says go) | 2 configs × 3 folds + seeds | ≈ 12 | VRAM measured in the smoke run |
| 6. M1-H models | Week 2–3, interleaved | SONDE-HF hourly-target variant and hourly references (same code, hourly heads) | ≈ 6 | Own protocol caps |
| 7. Finals and E2 | Week 3–4 | 3 seeds per selected family; out-of-fold weights for E2 and E2-B; numeric-forecast report on CV | ≈ 8 | Freeze candidate (lead) |
| 8. Rehearsal, review, single look | Week 4 | Streaming val rehearsal; pre-test review; lead's `FREEZE_m1n.json`; single look | ≈ 2 | `m1n_test_results.md` |

**Must be pre-registered first (before any result on it):**
1. the M1-N v0.6 amendment items (a)–(g);
2. the M1-H protocol;
3. the M1-SP protocol and its lock-on-download (if D6);
4. the EW-H1h option (if the lead wants it), with disclosure that it postdates the IV-rule failure.

The **information-use test (step 1)** is fixed in advance, with its decision rule. It runs on train-block inner CV
only, so it cannot leak validation or test information.

**Interactions to watch:**
- The M1 Test-A look (report-only) still needs a disk decision (about 13 GB).
- The live trial is on hold.
- None of the above depends on either.

---

## 9. Honest outlook (probability of earning a §15.1 label)

These are subjective estimates from the val evidence above and the literature. They are not computed power.

| Track | P(SOTA label) | Main reasons |
|---|---|---|
| **M1-N-F primary** (≥ 3 of 5 variables, every h, 3 seeds, Holm) **with GEFS** | **≈ 10 % (5–15 %)** | Needs 3 variables. Per variable: SC ≈ 40 % (already ahead of the controls on val); temperature ≈ 20 % (big h3–7 headroom, but h1 is at the ceiling and the reference set now includes NWP-GBM, PF-GBM and R-NH); DO ≈ 15 % (only 24 HUC4 → CI ±0.01–0.015); turbidity ≈ 15–20 % (needs rain forecasts, so only with GEFS); nitrate 0 (descriptive) |
| M1-N-F primary, **without GEFS** | ≈ 6 % | Forecast mode only for post-2021 issues, temperature only; perfect-prog domain shift (L2) |
| **M1-H** (sub-daily DO / hypoxia hours) | **≈ 20–25 %** | Structural advantage of a model with an hourly hindcast. Likely powered on CRPS. But diel persistence is a strong baseline, and parity gives the GBM hourly features too |
| X3-heat stratum (PhiDA-Lite) | ≈ 10–15 % | Few clusters; the physics gate may fail (Sim's record) |
| **First-in-class: numeric GPP/ER forecasts with intervals** (PhiDA-Lite, held-out Powell sites) | ≈ 25–35 % *if* the gate passes (≈ 50 %), so ≈ 15 % overall | Needs a dated novelty check, and must beat GPP persistence and climatology. Equifinality risk (P2, P5) |
| M1-SP (StreamPULSE block) | ≈ 10–15 % | Heterogeneous QC; cluster count uncertain |
| N-b virtual sensing | ≈ 10 % | The GBM was not beaten on val; more data helps all comparators equally |
| EW (any confirmatory hazard claim in 2026–27) | ≈ 5–10 % | EW-v3-N is descriptive; the pooled Test-A ∪ B pass is secondary; the live trial is on hold; EW-H1h is about 50/50 on power |
| **At least one §15.1 label for M1 within about 6 weeks** | **≈ 30 % (20–40 %)** | Tracks are correlated (shared encoder, shared NWP). Most likely: SC on M1-N, then M1-H DO |

**What would change these numbers.**
- *Up:* GEFS approved (turbidity and SC become forecastable).
- *Up:* the step-1 information-use test shows a > 0.02 sequence-model gain over GBM-NWP at h = 3–7.
- *Down:* SONDE-HF loses h = 1 to persistence-like references.
- *Down:* R-NH alone matches SONDE-HF. Then the honest label is "competitive", and the deliverable becomes the
  first open, pre-registered benchmark of NWP-driven multi-variable sonde forecasting
  (`docs/research/01` (c) #1, realism "high").

**What not to do:**
- lower the bar;
- reuse EW-v2 Test-A or the DO-exposed M1 Test-A for any new claim;
- pick strata or targets after seeing val wins;
- give the SENTINEL ensemble a weighting rule the baseline control does not get;
- spend GPU on bigger encoders before step 1 says the information is there.
