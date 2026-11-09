# 01: In-situ sensor time series (sondes): SOTA, benchmarks, data access, and a leakage-free plan

*Prepared 2026-10-22 for the SENTINEL v2 rebuild (Austin Jin & Bryan Cheng).*

**Scope.** This note covers high-frequency in-situ sensors: dissolved oxygen (DO), water temperature, pH, specific conductance (SC), turbidity, nitrate, fDOM, chlorophyll and phycocyanin fluorescence (fChl/fPC), and discharge/stage. For these it reports:
- the state of the art in forecasting, virtual sensing and event early warning;
- the public benchmarks that publish numbers;
- current data access;
- derived ecological features (stream metabolism);
- a concrete benchmark and model plan built to avoid the leakage failures found in the October 2026 audit.

**Tags.**
- **[V]** We verified it on 2026-10-22, by a live API call or file download, or from the Crossref / arXiv / DataCite / Hugging Face record plus the abstract or full text.
- **[D]** We derived it ourselves from public data. These numbers are not peer-reviewed.
- **[U]** Unverified. Do not cite without checking.

Numbers attributed to papers are the authors' own, with their split type.

**Process note.** The session-wide web-search quota ran out mid-research. After that, sources were checked by direct fetches: the arXiv API, Crossref, publisher pages, HF/PyPI/GitHub and S3 listings. Remaining gaps are listed in Appendix C.

---

## 0. Executive summary: the decisions that matter

1. **Build on the new USGS API now.** Use `api.waterdata.usgs.gov/ogcapi/v1` (`continuous`, `daily`, `time-series-metadata`) with `dataretrieval.waterdata` (v1.4.0, 2026-10-02) [V].
   - Legacy WaterServices (`/nwis/iv`, `/nwis/dv`) starts degrading in **November 2026**, has planned outages in **January 2027**, and is **decommissioned 2027-02-22** [V]. The legacy `dataretrieval.nwis` module is scheduled for removal on or after 2027-05-06 [V].
   - **Register a free API key.** Our anonymous quota on this host's IP ran out (`x-ratelimit-limit: 1000`, HTTP 429) [V]. A full SENTINEL pull is about 40–60 k requests.
2. **NEON downloads now require a free account plus an API token** (since 2026-06-30), and NEON data are now **CC BY 4.0** [V]. NEON metadata endpoints remain anonymous. EDI, which hosts LTER/GLEON lake buoys, has required sign-in since 2026-07-30; the DataONE mirror still serves files [V].
3. **Plenty of data exist.** USGS continuous series by site count [V]:

   | Variable | Sites |
   |---|---|
   | DO | 1,551 (758 active) |
   | SC | 2,558 |
   | Turbidity | 1,444 |
   | pH | 1,298 |
   | In-situ nitrate | 280 |
   | fDOM | 96 |
   | Phycocyanin | 133 |

   About **610 stream sites have at least 5 years of co-located DO + SC** (period-of-record upper bound). NEON adds 34 sites (2016–2026, 1/5-min). Strong add-ons are GLOS Lake Erie HAB buoys, the UK Environment Agency (≈1,900 DO sondes, ≈1,800 with ammonium), NERRS SWMP (hypoxia-rich), CDEC and StreamPULSE [V].
4. **Naive baselines are very strong for temperature and DO.**
   - In our pilot on 7 USGS supersites, a two-parameter damped-anomaly persistence forecast reaches **14-day NSE 0.95 for water temperature and 0.86 for DO** [D].
   - In the NEON Forecasting Challenge, only 10/29 temperature and 6/15 DO models beat day-of-year climatology, and none beat it on average at 15–25 days [V].
   - **Never report raw NSE/KGE without skill against persistence and climatology at each horizon.**
5. **Where deep models can genuinely win:**
   - **(i) Event-driven variables** (turbidity, SC, nitrate, phycocyanin) at 2–7 days, when operational weather forecasts are used. Naive skill collapses there: pilot 7-day turbidity NSE is 0.17 [D].
   - **(ii) Virtual sensing at unmonitored stations.** Published numbers are weak: nitrate KGE 0.18 in ungauged basins, SSC R² 0.55 [V].
   - **(iii) Event early warning.** We found **no** water-quality study reporting event-level POD/FAR/lead time with placebo or negative controls [V, search date-bounded].
6. **Time-series foundation models (TSFMs) are now serious baselines.**
   - Zero-shot Chronos-2 and TiRex already **beat persistence** in the live EFI–USGS river-chlorophyll challenge: 1-day CRPS 1.65 vs 2.03 µg/L [D, n = 2,357 matched forecasts].
   - Timestral-209M (Mistral, released 2026-10-15) tops GIFT-Eval and fev-bench [D].
   - Trained multi-site LSTMs still beat zero-shot TSFMs on CAMELS streamflow [V].
7. **Recommended model.** Fine-tune **Chronos-2** (Apache-2.0; covariate-aware; its official `fit()` trains in fp32 on Turing). Feed it past multi-sensor context, *operational* weather forecasts and catchment attributes. Alongside it:
   - a NeuralHydrology-style multi-station LSTM;
   - a **differentiable metabolism head** for DO, which yields interpretable GPP/ER/K600;
   - a calibrated event head.

   Ensemble these with GBM. Run Timestral, TimesFM-3/2.5, t0, Toto-2, TiRex, Moirai-2, Sundial and TabPFN-TS as zero-shot baselines.
8. **Benchmark ("SENTINEL-Sonde").**
   - Data: USGS 15-min data, with NEON as an *external* network.
   - Horizons: 1 h to 14 d.
   - Time splits: water-year blocks with a ≥42-day embargo.
   - Station splits: HUC4 groups, with flow-connected stations kept together.
   - Test periods: a post-release test year (WY2026) and a prospective live test.
   - Weather inputs: "oracle" (observed) and operational (forecast) kept strictly separate.
   - **Mandatory placebo and negative-control tests for every event claim.**
9. **External comparability.**
   - Score on neon4cast aquatics. Its live scoring has been stalled since August 2026, so recompute scores with its code.
   - Score on VERA and usgsrc4cast.
   - Replicate the Rahmani 2021 stream-temperature split (median RMSE 0.69 °C to beat).
   - Compare against LakeBeD-US / LakeFM for lakes, and TIME "Water_Quality_Darwin" for 15-min sondes.
10. **Data traps we found:**
    - 250 USGS DO series carry `SuppressData` lower thresholds at ≥ 2 mg/L, which may censor hypoxia [V; effect U].
    - Station records split across `time_series_id`s.
    - Fluorescence reported in RFU vs µg/L.
    - Legacy −999999 no-data codes, which corrupted v1 normalization (see 10_local_inventory.md). The new API returns nulls.
    - Hypoxia is short and nocturnal (median event 3 h; Blaszczak et al. 2023), so daily means miss it.

---

## (a) SOTA methods and reported metrics

### a.0 Calibration pilot: how strong are naive baselines on real USGS sonde data? [D]

**Why this matters.** Most published WQ-forecasting skill is reported as NSE, R² or RMSE without a naive reference. We measured the reference on real data before choosing claims.

**Daily pilot (method).**
- **Data:** USGS `daily` means (statistic 00003), pulled 2026-10-22, for 8 long-record supersites. Seven contributed:
  - USGS-01389005 (NJ)
  - 03353200 (IN)
  - 03612600 (Ohio R. at Olmsted, IL)
  - 05586300 (Illinois R. at Florence, IL)
  - 06892350 (KS)
  - 07374000 (Mississippi R. at Baton Rouge, LA)
  - 11447650 (Sacramento R. at Freeport, CA; temperature only)

  01646500 (Potomac) dropped out because its records switch to new `time_series_id`s in July 2019, which is itself a splicing pitfall.
- **Split:** climatology and AR coefficients fitted on WY2008–2019; scored on WY2022–2025.
- **Baselines:**
  - persistence;
  - smoothed day-of-year climatology;
  - damped anomaly persistence, `clim(t+h) + φ_h·(y_t − clim_t)`, with φ_h fitted on training data.
- Turbidity is scored in log space. Code is in Appendix B.

| Variable (sites) | Persistence / climatology / damped-anomaly NSE at **h = 1 d** | **3 d** | **7 d** | **14 d** |
|---|---|---|---|---|
| Water temperature (7) | 0.996 / 0.942 / 0.996 | 0.975 / 0.944 / 0.977 | 0.931 / 0.945 / 0.951 | 0.868 / 0.946 / **0.948** |
| DO (6) | 0.977 / 0.836 / 0.978 | 0.898 / 0.843 / 0.920 | 0.798 / 0.840 / 0.873 | 0.696 / 0.845 / **0.857** |
| Specific conductance (6) | 0.910 / −0.013 / 0.910 | 0.698 / −0.009 / 0.707 | 0.445 / 0.015 / 0.480 | 0.254 / −0.001 / 0.316 |
| Turbidity, log (5) | 0.809 / 0.028 / 0.817 | 0.350 / 0.043 / 0.438 | −0.192 / 0.048 / **0.167** | −0.398 / 0.045 / 0.117 |
| Nitrate in situ (4) | 0.896 / 0.098 / 0.900 | 0.621 / 0.092 / 0.673 | 0.361 / 0.071 / 0.501 | 0.053 / 0.068 / 0.373 |

Values are medians across sites.

**Sub-daily pilot.** Illinois River at Florence (USGS-05586300), 15-min data, 2024-06-01 to 2024-09-01. NSE is computed within one summer, a harder reference than multi-year NSE. Values are persistence NSE at +1 h / +6 h / +24 h / +72 h:

| Variable | +1 h | +6 h | +24 h | +72 h |
|---|---|---|---|---|
| DO | 0.94 | 0.80 | 0.61 | −0.21 |
| Water temperature | 0.998 | 0.96 | 0.87 | 0.34 |
| SC | 0.997 | 0.96 | 0.76 | 0.52 |
| pH | 0.96 | 0.91 | 0.79 | 0.36 |
| log-turbidity | 0.88 | 0.82 | 0.56 | −0.09 |
| fChl | 0.94 | 0.85 | 0.72 | 0.26 |
| fPC | 0.83 | 0.72 | 0.44 | 0.20 |

Nitrate is excluded from this table: the 50,000-row page cap truncated its series after one week.

**Implications.**
1. "NSE 0.9 for DO forecasting" is meaningless on its own. A two-parameter baseline reaches 0.86 at 14 days.
2. Short-horizon (≤6 h) skill for temperature, SC and pH is near the persistence ceiling.
3. Turbidity, nitrate and SC lose naive skill within 3–7 days, and DO within a single summer loses it by 72 h. This is where learned models with weather forecasts can add value.

These are pilots on 1–7 sites, not benchmark results. Reference works on the benchmark issue: Seibert 2001; Schaefli & Gupta 2007; Knoben et al. 2019.

### a.1 Hydrology / water-quality deep learning (multi-site prior art)

| Paper | Task | Data | Split | Reported metric (as published) | Code / data |
|---|---|---|---|---|---|
| Kratzert et al. 2019 HESS (EA-LSTM) [V] | Daily streamflow | CAMELS-US, 531 basins | Temporal | Median NSE 0.71 (single model) / 0.74 (ensemble); beats calibrated SAC-SMA, VIC, mHM | NeuralHydrology |
| Kratzert et al. 2019 WRR [V] | Ungauged streamflow | CAMELS-US, 531 | k-fold PUB | Median NSE: LSTM 0.69, SAC-SMA 0.64, NWM 0.58 | – |
| Gauch et al. 2021 HESS (MTS-LSTM) [V] | Hourly + daily streamflow | 516 basins | Temporal (test 2008–18) | Hourly median NSE ≈0.75; ≈+0.19 NSE over NWM hourly | NeuralHydrology |
| Feng, Fang & Shen 2020 WRR [V] | 1-day forecast with lagged discharge ("data integration") | CAMELS | Temporal | Median NSE 0.86 | – |
| Nearing et al. 2022 HESS [V] | Autoregression vs data assimilation of recent observations | CAMELS, 531 | Temporal | Autoregression: median NSE 0.796 → 0.879; DA +8%; a missing-data flag fixes sensitivity to gaps | – |
| Klotz et al. 2022 HESS [V] | Probabilistic LSTM heads | CAMELS, 531 | Temporal | CMAL best calibrated; MC-dropout weakest | NeuralHydrology |
| Liu, Bian, Lawson & Shen 2024 J. Hydrol. [V] | Transformer vs LSTM | CAMELS | Temporal | Vanilla Transformer not competitive, especially at high flows | Zenodo 13664154 |
| Liu, Shen, O'Donncha et al. 2025 HESS 29:6811 [V] | 11 Transformers + DLinear vs LSTM, including DO | 236 CAMELS-Chem basins (DO) | Temporal | LSTM DO median KGE 0.70; best Transformer within 0.01; LSTM wins memory-dependent tasks | – |
| Zhi et al. 2021 ES&T [V] | Daily DO from hydrometeorology | 236 CAMELS-Chem basins, **sparse grab samples** | Temporal + chemically ungauged | NSE not re-verified [U]; misses DO peaks and troughs | – |
| Zhi et al. 2023 *Nature Water* / *Nat. Clim. Change* [V] | DO drivers / 1981–2019 reconstruction | 580 US rivers (+216 European) | – | Temperature > light >> flow; deoxygenation in 70% of rivers. Training on daytime grab samples biases results | – |
| Rahmani et al. 2021 ERL [V] | Daily stream temperature | 118 undammed basins | Temporal (test WY2015–16) | Median RMSE 0.69 °C, NSE 0.986 with observed discharge; 0.86 °C without; local AR(2) 1.86 °C | doi:10.5066/P97CGHZH (CC0) |
| Rahmani et al. 2021 Hydrol. Process. [V] | Stream temperature, data-scarce or dammed basins | > 400 basins | PUB | RMSE 1.13 °C; dammed PUB 1.20 °C | – |
| Read et al. 2019 WRR (PGDL) [V] | Lake temperature profiles | 68 lakes | Temporal | Median RMSE: PGDL 1.65 °C, DL 1.78, process model 2.03 | doi:10.5066/P9AQPIVD |
| Willard et al. 2022 L&O Lett. [V] | Lake surface temperature | 185,549 lakes (12,227 observed) | Held-out lakes | Median per-lake RMSE 1.24 °C (overall 1.61) | doi:10.5066/P9CEMS0M |
| Jia et al. 2021 SDM (PGRNN/RGrN) [V] | River-network flow + temperature | Delaware River Basin (DRB) | Sparse labels | 24% / 14% better than LSTM (temperature / flow) | – |
| Topp et al. 2023 WRR [V] | Stream temperature under distribution shift | DRB | In-distribution + 5 shift tests | RMSE ≈1.64 °C; Graph WaveNet more robust (4/5 shift tests) | doi:10.5066/P9HU7BLR |
| **Zwart et al. 2023 JAWRA** [V] | **1–7-day forecasts** of daily max stream temperature | DRB | Operational | RMSE 1.1–1.4 °C (day 1), 1.4–1.9 °C (day 7); DA lowers RMSE 0–14%; only 65–82% of obs inside the 90% PI | doi:10.5066/P9GD8I7A |
| Sadler et al. 2022 WRR [V] | Multi-task discharge + temperature | 101 sites | Temporal | Multi-task helps discharge at 56/101 sites single-site, but at only 17 in the multi-site model; naive loss weights hurt temperature | USGS release |
| Saha et al. 2023 STOTEN [V] | Daily nitrate, data-sparse sites | 42 Iowa sensor sites | Temporal | Median NSE 0.75, RMSE 1.53 mg/L; neighbour-site nitrate is the key input | – |
| Pandit et al. 2025 WRR [V] | Daily riverine nitrate | Mississippi River Basin sensor sites (n [U]) | Temporal / PUB | Median KGE **0.60 temporal, 0.18 PUB** | – |
| Song et al. 2024 J. Hydrol. [V] | Daily SSC | 377 sites | Temporal / spatial | Median R² 0.63 (CONUS model), 0.55 in new basins | – |
| Smith et al. 2024 ES&T [V] | Daily SC (salinization) | DRB, 459 segments | – | **KGE: random forest 0.67 vs RGCN 0.64** | USGS release |
| Kermorvant et al. 2021 IJERPH [V] | 15-min nitrate gap-filling (GAM + ARIMA) | NEON ARIK | Synthetic gaps | Median RMSE 0.25 µmol/L (10-day gaps) to 0.75 (10-week gaps) | – |

**Reviews [V].** All were verified, and they agree on the points below.
- Zhi, Appling, Golden, Podgorski & Li 2024, *Nature Water* 2:228–241, "Deep learning for water quality".
- Varadharajan et al. 2022, *Hydrol. Process.* 36:e14565.
- Shen et al. 2023, *Nat. Rev. Earth Environ.* 4:552–567 (differentiable modelling).
- Willard et al. 2025, *Environ. Data Sci.* 4:e7 (prediction at unmonitored sites).
- Lofton et al. 2023, *Global Change Biology* 29:1691–1714 (near-term freshwater-quality forecasting).
- Rousso et al. 2020, *Water Research* 182:115959 (cyanobacteria forecasting).

They agree on four points:
- Most WQ-DL studies are single-site, use random or unblocked splits, and compare only to weak baselines.
- Multi-site LSTMs with static attributes are the strongest prior. Single-basin training is an anti-pattern (Kratzert et al. 2024).
- Accuracy drops at dam- or groundwater-influenced sites, for reactive solutes in ungauged basins, and in extremes.
- Predictive intervals tend to be over-confident.

### a.2 Near-term WQ forecasting: what actually beats persistence and climatology

| Study | Variable / horizon | Key finding |
|---|---|---|
| **Olsson et al. 2025, *Ecol. Appl.* 35:e70004** [V]: NEON Forecasting Challenge synthesis ("100,000 freshwater forecasts") | Temperature and DO at 7 NEON lakes, 1–30 d, CRPS (calendar 2023) | **10/29** temperature and **6/15** DO models beat day-of-year climatology. Persistence beat climatology on days 1–3. **No model beat climatology on average at 15–25 d** (temperature). Best model was process-based (FLARE-GLM). Chl-a was omitted for data-quality reasons |
| Thomas et al. 2023b, *Front. Ecol. Environ.* 21:220 [V] | FLARE, 6 NEON lakes | Temperature RMSE 0.61 °C at 1 d, 2.17 °C at 35 d; better than the null model out to 35 d |
| Olsson et al. 2024 WRR [V] | Lake temperature profiles | An ensemble of baseline and process models improved probabilistic skill by 8–30%; no single model was best everywhere |
| Thomas et al. 2020 WRR (FLARE) [V] | Reservoir temperature, 1–16 d | RMSE 1.13 °C (7 d), 1.62 °C (16 d); fall turnover predicted 4–14 d ahead |
| Woelmer et al. 2022 *Ecol. Appl.* [V] | Chl-a, 1–14 d | Beat the null model "under most conditions"; coarser model time steps were better at 7–14 d |
| Lofton et al. 2022 *Ecol. Appl.* [V] | Cyanobacteria (*Gloeotrichia*), 1–4 wk | No covariate model beat the previous-week model at 1 week |
| **Fournier et al. 2024, *Water Res.* 267:122553** [V] | Phycocyanin (alarm > 10 µg/L), 4–28 d | Alarm accuracy 0.7–0.9, but **MAPE skill vs persistence was negative for every model at 4–16 d** |
| Howard et al. 2026 *JGR-Biogeosciences* [V] | fDOM, 1–34 d, 3 reservoirs | Only one model beat both day-of-year climatology and persistence across the full horizon |
| Kemper et al. 2025 JAWRA [V] | Turbidity, 1–3 d (NYC water supply) | LightGBM with National Water Model flow forecasts was "relatively skillful" |
| Kim et al. 2021 WRR [V] | DO depletion in urban streams, 24 h | Hypoxic events forecast with "lower accuracy"; 8:2 split with no blocking stated (leakage risk) |
| EFI challenge score databases (see b.3) [D] | neon4cast / VERA / usgsrc4cast | Chronos-2 and TiRex (zero-shot) beat persistence on river chlorophyll; nothing reliably beats climatology for VERA bloom probability at ≥ 7 d |

### a.3 General time-series architectures and state-space models

| Model | Citation | Evidence | Relevance for WQ |
|---|---|---|---|
| DLinear / NLinear | Zeng et al., AAAI 2023, arXiv 2205.13504 [V] | A single linear layer beat LTSF Transformers | Mandatory sanity baseline (GIFT-Eval normalized CRPS 0.846 [D]) |
| PatchTST | Nie et al., ICLR 2023, arXiv 2211.14730 [V] | Patching + channel independence | GIFT 0.587 [D]; original code has the `drop_last` test bug |
| iTransformer | Liu et al., ICLR 2024, arXiv 2310.06625 [V] | Attention over variate tokens | Captures cross-sensor coupling (DO–T–pH); GIFT 0.620 [D] |
| TimeMixer / TimeMixer++ | Wang et al., ICLR 2024 (2405.14616) / ICLR 2025 (2410.16032) [V] | Multiscale mixing | Evidence only on saturated LTSF sets |
| TSMixer | Chen et al., TMLR 2023, arXiv 2303.06053 [V] | All-MLP with auxiliary covariates; strong on M5 | TSMixerx (neuralforecast) takes weather covariates |
| TimesNet | Wu et al., ICLR 2023, arXiv 2210.02186 [V] | 2-D period modelling; 5 task types | Anomaly-detection baseline |
| N-HiTS / N-BEATS(x) | Challu et al., AAAI 2023 (2201.12886); Oreshkin et al., ICLR 2020 (1905.10437) [V] | Fast multi-horizon models | Cheap baselines (N-BEATS GIFT 0.816 [D]) |
| TFT | Lim et al., *IJF* 37:1748–1764 (2021) [V] | Static + known-future + observed inputs; quantiles; variable selection | Classic fit for NWP covariates plus catchment attributes (GIFT 0.605 [D]) |
| S4; SpaceTime | Gu et al., ICLR 2022 (2111.00396); Zhang et al., ICLR 2023 (2303.09489) [V] | Structured SSMs | Historical |
| Mamba for time series: S-Mamba (2403.11144), TimeMachine (ECAI 2024, 2403.09898), Bi-Mamba+ (2404.15772), Chimera (2406.04320), Mamba4Cast (2410.09385) [V records] | – | LTSF-only evidence | **No robust evidence they beat TSFMs** |
| xLSTM-Mixer | Kraus et al., NeurIPS 2025, arXiv 2410.16928 [V] | Best per-dataset-trained deep model on GIFT (0.510 [D]) | sLSTM CUDA kernels need CC ≥ 8.0; use the vanilla backend on Turing |

**Benchmark pitfalls [V].**
- Per-dataset deep models lose to TSFMs on GIFT-Eval: top TSFMs score 0.45–0.49 normalized CRPS, versus 0.59–0.62 for PatchTST, TFT and iTransformer [D].
- Informer-lineage `drop_last=True` test loaders silently drop test windows. This affects PatchTST, DLinear (AAAI version), Informer and Autoformer.
- Brigato et al. (TMLR; arXiv 2502.14045), "There are no Champions in Supervised Long-Term Time Series Forecasting", trained about 5,000 networks and found that small setup changes flip rankings.
- TFB (Qiu et al., PVLDB 2024) documents pipeline inconsistencies.
- ETT/Weather/ILI results are **not** evidence for WQ.

**Hardware facts.** Two come from our own tests on this host [V]:
- bf16 matmul runs at **7.1 TFLOPS vs 51.3 TFLOPS for fp16** (11.5 for fp32), even though `torch.cuda.is_bf16_supported()` returns True.
- The FlashAttention SDPA backend is unavailable on sm_75; the memory-efficient and math backends work.

From the TSFM sub-study:
- Triton officially supports only compute capability ≥ 8.0, so avoid `torch.compile`.
- Mamba-2's Triton kernels crash on Turing in fp32 (mamba issue #361).
- Prebuilt mamba-ssm 2.2.5 / 2.2.6.post3 wheels exist for cu11/torch 2.7.
- flashrnn/xLSTM CUDA kernels need CC ≥ 8.0.
- torch 2.7.1+cu118 is the newest cu118 build (2.7.0 and 2.7.1 cu118 wheels verified).

### a.4 Time-series foundation models (state as of 2026-10-22)

**Sources for the leaderboard columns.**
- GIFT-Eval: recomputed from the Salesforce/GIFT-Eval Space at commit 23078cb (2026-10-17), using its own aggregation. Values are CRPS / MASE normalized to seasonal naive, geometric mean over 97 configs [D].
- fev-bench: autogluon/fev at commit f9e2c7d3 (2026-10-16), 100 tasks, SQL skill / win rate [D].
- Ten model cards were checked via the HF API on 2026-10-22 for existence, creation date and license [V]: Timestral-209M, TimesFM-3.0, Chronos-2, TiRex-2, TimesFM-2.5, t0-gamma, Toto-2.0-22m, Moirai-2.0-small, Sundial-base and FlowState. The other rows rely on the TSFM sub-study.

| Model (release) | Params | License | Covariates / multivariate | GIFT CRPS / MASE | fev skill / win | Runs on 2080 Ti, CUDA 11.8, torch 2.7? | Hydrology in pretraining corpus? |
|---|---|---|---|---|---|---|---|
| **Timestral-209M** (Mistral, 2026-10-15) | 209M (also 5M–123M) | Modified MIT (HF "other"; revenue clause) | Past + future + categorical / yes | **0.446 / 0.658** (#1 pretrained) | **0.504 / 0.890** (#1 open) | Likely (pure torch ≥ 2.4, fp32, SDPA). No official fine-tuning code | Mostly synthetic; real data unnamed |
| t0-gamma (The Forecasting Company, 2026-10) | ~256M | Apache-2.0 | Yes / yes | 0.452 / 0.661 | 0.494 / 0.867 | Likely | Undisclosed |
| TimesFM-3.0 (Google, 2026-08) | 331M | TimesFM Non-Commercial | Covariates / yes | 0.456 / 0.667 | 0.490 / 0.815 | Yes (disable `torch.compile`) | GiftEvalPretrain (minus fev overlap), Trends, Wiki, synthetic; no USGS |
| **Chronos-2** (AWS, 2025-10) | 120M | **Apache-2.0** | Past + known-future (real/categorical) / yes (group attention); context 8,192, horizon 1,024 | 0.485 / 0.698 | 0.473 / 0.740 | **Yes.** Official full and LoRA `fit()`; trains in fp32 on Turing (bf16 only if sm80) | Chronos datasets, including Saugeen river flow, USHCN, WeatherBench; **no USGS / CAMELS / NEON** |
| Toto-2.0 (Datadog, 2026) | 4M–2.5B | Apache-2.0 | ? / yes | 2.5B: 0.476 / 0.696 | 2.5B: 0.444 / 0.702 | Up to 1B (2.5B: fp16 inference only) | Observability telemetry + synthetic |
| TiRex-2 (NX-AI, 2026) | 82.5M | Apache-2.0 | Past + future / yes | 0.478 / 0.697 | 0.455 / 0.699 | **No on GPU** (needs torch ≥ 2.8; flashrnn needs CC ≥ 8.0); CPU works | Chronos datasets + LOTSA |
| TiRex 1.x (NeurIPS 2025) | 35M | NXAI Community | No / no | 0.488 / 0.716 | 0.426 / 0.625 | Likely (torch backend) | Chronos datasets + GiftEvalPretrain |
| TimesFM-2.5 (2025-09) | 231M | Apache-2.0 | XReg / no; 16k context | 0.490 / 0.705 | 0.422 / 0.589 | Yes (LoRA example) | As TimesFM-3 |
| FlowState r1.1 (IBM, ICML 2026) | 9.1M | Apache-2.0 | SSM encoder, sampling-rate-equivariant | 0.487 / 0.701 | 0.399 / 0.531 | Yes (granite-tsfm ≤ 0.3.4) | [U] |
| Moirai-2.0-small (Salesforce) | 11.4M | CC-BY-NC-4.0 | Univariate decoder | 0.516 / 0.728 | 0.393 / 0.464 | Yes, in a separate torch 2.4 env (uni2ts needs torch < 2.5) | GIFT pretrain + synthetic; no USGS |
| Sundial-base (ICML 2025) | 128M | Apache-2.0 | No / no (generative) | 0.559 / 0.750 | 0.334 / 0.309 | Yes | TimeBench (incl. ERA5, LOTSA) |
| Chronos-Bolt (2024-11) | 9–205M | Apache-2.0 | No | 0.574 (leak-flagged) | 0.389 / 0.456 | Yes (fp32 safer) | Chronos datasets |
| TabPFN-TS (arXiv 2501.02945) | TabPFN-v2/3 | v2 open; v3 non-commercial | Yes (tabular features) | 0.544 / 0.771 | 0.415–0.431 | Yes | Synthetic prior |
| TTM r2 (NeurIPS 2024) | 0.8–5M | Apache-2.0 | Exogenous inputs in fine-tuning | r3-FT 0.511 | – | Yes, even on CPU | Monash + LibCity |
| Also evaluated | Time-MoE (ICLR 2025, Apache-2.0); Timer / Timer-XL (Apache-2.0); MOMENT (MIT; better for imputation/anomaly detection); Lag-Llama (GIFT 0.880); Timer-S1 (8.3B, too large for 11 GB); Moirai-1.1 (CC-BY-NC) | | | | | | Time-300B / UTSD include ERA5 and Saugeen; no USGS |

**Pretraining contamination.**
- The disclosed corpora contain **no USGS, CAMELS, Caravan or NEON series**. These are the Chronos datasets, LOTSA, GiftEvalPretrain, Time-300B and UTSD (HF dataset trees listed by our TSFM sub-study) and TimeBench (from the Sundial paper). Their hydrology content is Saugeen river flow plus weather and climate reanalyses.
- The Toto, TimesFM, Timestral, t0 and Moirai-2 corpora are only partly disclosed.
- SENTINEL should therefore score every TSFM on a test year after the model's release (WY2026) plus held-out stations, and ideally prospectively.

**TSFM evidence on hydrology and WQ.**

| Evidence | Setting | Result |
|---|---|---|
| **TIME benchmark "Water_Quality_Darwin"**: Qiao et al., *It's TIME*, ICML 2026, arXiv 2602.12147 [V]. Official per-window predictions re-scored against persistence by our sub-study [D] | 15-min sonde (conductivity, DO, salinity, temperature, turbidity, chl), 2015–2018, 7 deployments; likely tidal coastal water [U] | MAE relative to persistence (< 1 = better) at 4 h / 24 h / 72 h: Timestral 0.58 / 0.58 / 0.72; TimesFM-3 0.62 / 0.61 / 0.77; Chronos-2 0.69 / 0.64 / 0.79; TiRex 0.70 / 0.71 / 0.86; Moirai-2 0.75 / 0.75 / 0.94; **Sundial 0.81 / 0.92 / 1.09**. Tides favour models over persistence; about 15 test days per deployment |
| Sun & Sun 2026, *Machine Learning: Earth* 2:010501 [V] | CAMELS-US, 531 basins, daily, 1 day ahead, zero-shot univariate | Median NSE: **trained LSTM 0.593**, Sundial 0.564, Chronos-Bolt 0.495, TTM 0.437, Moirai 0.258. Fine-tuned TTM with static attributes 0.753. No persistence baseline |
| He et al. 2025, arXiv 2511.11849 [V record] | CAMELS-US | A multivariate LSTM beats zero-shot Chronos-Bolt and Sundial on every stream |
| Zheng et al. 2026, *Water Research X* 30:100469 [V record] | Chronos for total nitrogen, 12 Chinese stations, 4–48 h | Fine-tuning beats zero-shot (numbers [U]) |
| Rangaraj et al. 2025, arXiv 2505.01415 [V record] | Everglades water level | Chronos-Bolt-small beats 12 task-specific models and 4 other FMs |
| Zhu & Liu, GitHub preprint [U] | 6,569 catchments, 1-day | Persistence (0.705) beats TimesFM (0.677) on LamaH; AR(2) beats both on CAMELS-CL |
| Chronos-2 on CAMELS-DE [U; venue unknown] | 1,582 catchments | Zero-shot with meteorological covariates NSE 0.68; univariate 0.15; LSTM ensemble 0.90; fine-tuned Chronos-2 reportedly matches the ensemble |
| **LakeFM** (Neog et al., KDD 2026, arXiv 2606.11268) [V record] | Lake WQ foundation model on LakeBeD-US (21 lakes; 15 in-distribution / 6 out-of-distribution held out); 30-day context, 14-day horizon | 7M parameters; matches or beats Chronos-2, Moirai, MOMENT and iTransformer on MSE/CRPS. **Direct prior art for lakes** |
| HydroGEM (arXiv 2512.14106) [V record] | Self-supervised TCN-Transformer on 3,724 USGS discharge stations, for QC | F1 0.792 on held-out stations (synthetic anomalies) |
| GIFT-Eval Saugeen subset [D] | One daily river-flow series | Normalized CRPS: FlowState 0.542, TimesFM-3 0.551, Timestral 0.556, Chronos-2 0.567, PatchTST 0.727, DLinear 1.019. Monash archive MASE: CatBoost 1.41 beats zero-shot TimesFM 2.23 |

### a.5 Event / anomaly detection and early warning

| Study | Topic / data | Result | Placebo / negative control? |
|---|---|---|---|
| Leigh et al. 2019 STOTEN; Talagala et al. 2019 WRR (`oddwater`) [V] | Technical anomalies (spikes, shifts, drift) in turbidity/SC/level, Great Barrier Reef rivers | Regression and feature-based methods work best combined; feature-based has low false positives but high false negatives | No |
| **Jones, Jones & Horsburgh 2022 EMS (PyHydroQC)** [V] | Technician-labelled anomalies in temperature/SC/pH/DO, Logan River, 6 sites | **ARIMA F2 0.889 beats all 4 LSTM variants (0.769–0.827)**; DO at one site only 0.497 | No (point metric after window widening, a point-adjust-like inflation risk) |
| Rodriguez-Perez et al. 2020 ES&T; Russo et al. 2020 EMS / 2021 WR [V] | ANN / active-learning QC | Semi-supervised best for spikes and shifts, supervised for drift; expert labels very valuable | No |
| EPA CANARY: Hart et al. 2007; Burkhardt et al. 2022 *Env. Sci.: Adv.*; Cronin et al. 2025 MethodsX [V] | Multi-parameter event detection (spills, illicit discharges, turbidity events) | > 90% of manually identified events (urban watershed); 47/47 and 38/39 turbidity events (all partial matches). **Repo archived 2026-06-24 (v4.3.3)** | No |
| **Blaszczak et al. 2023, *L&O Letters* 8:453** [V] | Hypoxia (DO < 2 mg/L), > 125k river sites | 12.6% of sites; **median event duration 3 h; most start at night** | n/a |
| Aulenbach & McKee 2020, USGS OFR 2020-1048 [V] | E. coli > 235 CFU/100 mL nowcast from turbidity + flow + season | Adj. R² 0.795 / 0.686; misclassification 3.5–11.3% | No |
| Sorensen et al. 2018 STOTEN [V] | Fecal contamination via tryptophan-like fluorescence (TLF) | At 1.3 ppb TLF: 15% false negatives, 18% false positives | No |
| Moore et al. 2020 ES&T [V] | Chloride-criteria exceedances from SC, 93 sites | Frequent where impervious cover > 9–10% | n/a |
| Nearing et al. 2024 *Nature* [V] (template) | Flood return-period events, 5,680 gauges | Per-gauge precision/recall/F1 with a ±2-day hit window, a 1-year temporal buffer and 10-fold spatial CV | Design template |
| TSAD critiques [V]: Wu & Keogh (TKDE 2021); Kim et al. AAAI 2022; Paparrizos et al. PVLDB 2022 (VUS); Huet et al. KDD 2022; Liu & Paparrizos NeurIPS 2024 (TSB-AD) | Evaluation methodology | Under point-adjust, even random scores look SOTA. Simple and statistical detectors often win. Prefer range-based or threshold-free metrics | – |

**Conclusions from a.5.**
1. We found no WQ early-warning study that ran placebo or negative-control tests.
2. Event-level POD/FAR/lead-time reporting is essentially absent in WQ.
3. Persistence is very hard to beat at short horizons.
4. For QC, ARIMA beats LSTMs.
5. Sensor faults (fouling, calibration steps) mimic real events and must be screened before labelling.

### a.6 Virtual sensing / nowcasting

- **Unseen sites (PUB) are the weakest area.**
  - Nitrate: KGE 0.18 (Pandit 2025).
  - SSC: R² 0.55 in new basins (Song 2024).
  - Stream temperature: RMSE 1.13–1.20 °C (Rahmani 2021 HP).
  - SC: KGE 0.67, where a random forest beat a graph network (Smith 2024).
- **Seen-site gap filling is easier.** Nitrate NSE 0.75 from neighbour sites (Saha 2023); 15-min nitrate RMSE 0.25 µmol/L for 10-day gaps (Kermorvant 2021).
- **Classical surrogate regressions are the baselines any virtual sensor must beat:**
  - turbidity → SSC (Rasmussen et al. 2009, USGS TM 3-C4);
  - SC → chloride (Granato & Smith 1999; Moore et al. 2020; data release doi:10.5066/P9YN2QST);
  - turbidity + flow + season → E. coli (Aulenbach & McKee 2020).

---

## (b) Benchmarks and datasets, with verified access details

### b.1 Core raw sources: USGS and NEON (verified by live calls, 2026-10-22)

| Name | URL | Size / coverage | Format | License | Variables (resolution) | Account? |
|---|---|---|---|---|---|---|
| **USGS `continuous`** (formerly NWIS IV) | https://api.waterdata.usgs.gov/ogcapi/v1/collections/continuous | Site counts in the table below. Archive effectively **starts 2007-10-01**: no values for 1990 at a series whose metadata begins in 1987; 2007-10-01 present. About 35k rows per series-year at 15 min; 3 months × 8 parameters at one site = 5.4 MB CSV | GeoJSON (`f=json`), CSV (`f=csv`), JSON-LD, HTML | US federal public domain. Per-value `approval_status` (Approved / Provisional) and `qualifier` | DO 00300, DO %sat 00301, temperature 00010, pH 00400, SC 00095, turbidity 63680 (FNU), NO3+NO2 99133 / nitrate 99137, fDOM 32295, fChl 32316 (µg/L) / 32315 (RFU), fPC 32319 (µg/L) / 32321 (RFU), discharge 00060, stage 00065, tidally filtered Q 72137, salinity 00480, SSC surrogate 99409. Mostly 15-min | Not for small use. **A free key is needed in practice**: "If you're going to make more than a few queries per hour, you'll need to provide an API key" |
| USGS `daily` (formerly DV) | …/collections/daily | Daily mean/min/max computed from continuous data | same | same | statistic_id 00001 max, 00002 min, 00003 mean | same |
| USGS metadata: `time-series-metadata`, `combined-metadata`, `monitoring-locations`, `field-measurements` | …/collections/… | Begin/end per series, site type, HUC, drainage area, **operational thresholds incl. `SuppressData`**, field measurements for sensor checks | same | same | – | same |
| **NEON DP1.20288.001 Water quality** | https://data.neonscience.org/data-products/DP1.20288.001 | **34 aquatic sites**, 3,297 site-months, 2016-03 to 2026-09 (RELEASE-2021 to 2026 + provisional) | Zipped CSV packages (basic/expanded) | **CC BY 4.0 since 2026-06-30** (was CC0) | SC, DO, DO %sat, pH, turbidity, fDOM; chl-a at lakes/rivers only (YSI EXO2 incl. "chlorophyll-PC" probe; whether a PC channel is published is [U]). 1-min at streams, 5-min at lakes/rivers; 4-hourly profiles at BARC, CRAM, LIRO, PRLA, TOOK | **Yes for data** (free account + API token); metadata anonymous |
| NEON DP1.20033.001 Nitrate | …/DP1.20033.001 | 34 sites, 3,102 site-months, 2016-12 to 2026-09 | same | same | SUNA V2; 15-min burst means | Yes |
| NEON DP1.20264.001 Temperature at specific depth | …/DP1.20264.001 | 10 sites (lakes/rivers), 706 site-months | same | same | T-chain; 1-min and 30-min | Yes |
| NEON DP1.20053.001 Temperature (PRT) in surface water | …/DP1.20053.001 | 24 stream sites, 2,536 site-months | same | same | 1-min and 30-min | Yes |
| NEON DP4.00130.001 Continuous discharge | …/DP4.00130.001 | 28 sites, 2,853 site-months, 2016-08 to 2026-09 | same | same | 1- and 15-min, with 95% uncertainty | Yes |
| NEON companions | DP1.20016.001 water elevation (34 sites); DP1.20042.001 PAR at surface (34); DP1.20261.001 PAR below surface (10); DP1.20097.001 dissolved gases (34); DP1.20190.001 reaeration (24); DP1.20093.001 grab-sample chemistry (34); DP1.20163.001 periphyton/seston/phytoplankton chemistry (34) | | | | Useful for metabolism and as labels | Yes |

**NEON sites.**
- 7 lakes: BARC, CRAM, LIRO, PRLA, PRPO, SUGG, TOOK.
- 3 large rivers: BLWA, FLNT, TOMB.
- 24 wadeable streams.

**USGS site counts.** From `time-series-metadata`, Instantaneous series, period-of-record (POR) spans [V]. "Active" means the series ends on or after 2026-10-01.

| Parameter | Sites with data | Active | POR ≥ 5 y | POR ≥ 10 y |
|---|---|---|---|---|
| DO (00300) | 1,551 | 758 | 785 | 470 |
| SC (00095) | 2,558 | 1,220 | 1,403 | 864 |
| Turbidity FNU (63680) | 1,444 | 678 | 686 | 295 |
| pH (00400) | 1,298 | 556 | 643 | 367 |
| NO3+NO2 in situ (99133) | 280 | 141 | 124 | 57 |
| fDOM (32295) | 96 | 50 | 22 | 7 |
| fChl µg/L / RFU (32316 / 32315) | 119 / 132 | 38 / 58 | 27 / 18 | 8 / 1 |
| fPC µg/L / RFU (32319 / 32321) | 61 / 110 | 20 / 53 | 11 / 22 | 3 / 10 |

- Phycocyanin in either unit: 133 sites (73 streams, 41 lakes).
- DO sites by type: streams 1,106, lakes 202, estuaries 91, tidal streams 46, wells 40, springs 22.

**Co-located POR overlap at stream sites** [V; these are upper bounds, because real completeness is lower due to seasonal removal and gaps]:

| Variable set | ≥ 5 y | ≥ 10 y |
|---|---|---|
| DO + SC | 610 | 381 |
| + turbidity | 377 | 206 |
| + pH | 339 | 184 |
| + nitrate | 61 | 32 |
| + fDOM | 7 | – |

Lakes with DO + SC ≥ 5 y: 48.

**USGS access facts** [V unless marked]:
- **Paging and limits:**
  - Pages hold up to **50,000 records** and are followed via the `next` link (JSON).
  - **CSV responses carry no `next` link**, so chunk by time.
  - `continuous` allows **at most 3 years per query**. If `time` is blank it returns the last year.
  - Use `properties=` to trim fields; about 107 bytes/row in CSV with coordinates, about 50 without.
- **Key and rate limits:**
  - The key (api.data.gov) goes in `api_key=` or the `X-Api-Key` header; sign up at https://api.waterdata.usgs.gov/signup/.
  - After about 110 anonymous calls from this agent (the IP is shared with other processes), we got **HTTP 429 `OVER_RATE_LIMIT`** with `x-ratelimit-limit: 1000`, `x-ratelimit-remaining: 0`, `retry-after: 1315`.
  - The api.data.gov default for keyed use is 1,000 requests/hour. Ask wdfn@usgs.gov for more.
- **Query quirks:**
  - CQL2 numeric filters on `value` failed: HTTP 500 for `<`, and `between` compares as strings (values are strings).
  - `state_name`/`state_code`/HUC filters on `daily` returned 0 rows in our tests.
  - So list sites from metadata, query per site, and filter locally.
- **Missing values** are `null` in the new API. The legacy service used −999999, which leaked into v1 normalization statistics (see `10_local_inventory.md`).
- **Thresholds that may censor hypoxia.** 1,270 of 1,715 DO series carry a lower threshold with `SuppressData: true`. **250 have it at ≥ 2 mg/L** (values seen: 2, 2.5, 3, 4, 5, 6).
  - How this affects publicly served values is **[U]**. At three well-oxygenated test sites no value fell below the threshold, which is uninformative.
  - Treat it as potential censoring for hypoxia labels, and check against `field-measurements`.
- **Series splitting.** A station's record for one parameter can split into several `time_series_id`s after sublocation or equipment changes (USGS-01646500 switches in July 2019). Splice only after an overlap or offset check.
- **Timeline:**
  - Data for parameter codes released after June 2026 are **not** served by legacy services (WDFN blog, 2026-06-30).
  - v0 `/time-series-metadata` and related endpoints stay online until June 2027 and are then redirected to v1 (blog, 2026-09-04).
  - WaterServices is decommissioned **2027-02-22**, with degradation from November 2026 and outages in January 2027 (USGS bulletin, 2026-09-17).
- **`dataretrieval` 1.4.0** (Python ≥ 3.10; httpx, pandas ≥ 2):
  - The `dataretrieval.waterdata` module provides `get_continuous`, `get_daily`, `get_latest_continuous`, `get_latest_daily`, `get_time_series_metadata`, `get_combined_metadata`, `get_monitoring_locations`, `get_field_measurements`, `get_peaks`, `get_samples`, `get_stats_por`, `get_stats_date_range`, `get_cql`, `get_reference_table`, `get_nearest_continuous` and `get_ratings`.
  - Key via the `API_USGS_PAT` env var, `~/.dataretrieval/config.toml` or `configure()`.
  - `dataretrieval.nwis` (`get_iv`, `get_dv`, `get_record`) is deprecated, with removal on or after 2027-05-06.
- **Discrete samples** (for validating proxies: chl-a, microcystin, E. coli). The Water Quality Portal's legacy WQX 2.2 profiles exclude USGS data added after 2024-03-11; use WQX 3.0 at https://www.waterqualitydata.us/beta/ or the USGS Samples API.
- **Pull volume.** About 610 sites × ~7 parameters × ~10 years ≈ 43k series-years ≈ **43k one-year requests (~43 h at 1,000 requests/h)** and about 1.5 B rows. That is roughly 6–10 GB as Parquet.

**NEON access facts [V]:**
- `/api/v0/products/...` works anonymously (headers `x-ratelimit-limit: 200`, `x-ratelimit-reset: 1`). `/api/v0/data/...` and `/data/query` return **HTTP 403 "Access Denied"** without a token.
- Python `neonutilities` 2.0.2 (2026-09-09) provides `load_by_product`, `zips_by_product`, `stack_by_table`, `dataset_query`, `read_table_neon`, `token_date`, `get_citation` and `get_issue_log`. It sends the token in the `X-API-TOKEN` header, and its error text reads: "As of June 2026, NEON requires an API token for data download."
- For R, use neonUtilities ≥ 4.0.1. Latency is about 2 weeks after month end.

### b.2 Public benchmarks / benchmark-style datasets with published numbers

| Name | URL | Size | Format | License | Coverage | Variables / resolution | Account? | Published metric / split / numbers |
|---|---|---|---|---|---|---|---|---|
| **NEON Forecasting Challenge, Aquatics (neon4cast)** [V] | https://projects.ecoforecast.org/neon4cast-ci/targets.html; targets `https://sdsc.osn.xsede.org/bio230014-bucket01/challenges/targets/project_id=neon4cast/duration=P1D/aquatics-targets.csv.gz` | Targets 2,026,711 B gz, 265,410 rows (last modified 2026-08-15); scores parquet up to ~370 MB per model/variable | CSV.gz (long format); S3 parquet scores | Challenge data [U]; Olsson data on Zenodo CC BY 4.0 | 34 NEON sites; data 2016-03-04 to **2026-07-31** | Daily mean DO and temperature (34 sites); chl-a at 10 sites (BARC, BLWA, CRAM, FLNT, LIRO, PRLA, PRPO, SUGG, TOMB, TOOK) | None for data; a registration form for a scored model_id | Olsson 2025 (see a.2); derived numbers in b.3 |
| **VERA (vera4cast)** [V] | https://www.ltreb-reservoirs.org/vera4cast/; targets `https://amnh1.osn.mghpcc.org/bio230121-bucket01/vera4cast/targets/project_id=vera4cast/duration=P1D/daily-insitu-targets.csv.gz` | 1.76 MB gz (222k rows) | CSV.gz with depth | [U] | Falling Creek and Beaverdam Reservoirs + inflow, 2013 to 2026-10-21 | Daily temperature, DO, chl-a, fDOM, SC, turbidity, fluxes; **`Bloom_binary` (chl-a > 20 µg/L)** | None (registration to submit) | No synthesis paper found; derived numbers in b.3 |
| **EFI–USGS River Chlorophyll (usgsrc4cast)** [V] | https://projects.ecoforecast.org/usgsrc4cast-ci/targets.html; `…/project_id=usgsrc4cast/duration=P1D/river-chl-targets.csv.gz` | 380 KB gz (54k rows) | CSV.gz | [U] | 15 rivers (10 USGS + 5 CA DWR), to 2026-10-21 | Daily mean chl-a; horizon ≤ 30 d | None | Derived numbers in b.3 |
| Rahmani 2021 stream temperature [V] | https://doi.org/10.5066/P97CGHZH | ~40 MB | CSV/zip | CC0 | 118 GAGES-II sites, WY2011–2016 | Daily water temperature, Q, forcings, attributes, predictions | None | Test WY2015–16; median RMSE / NSE recomputed from the release: LSTM + observed Q 0.69 °C / 0.986; simulated Q 0.82 / 0.983; no Q 0.86 / 0.980; local AR(2) 1.86 / 0.914 |
| Delaware River Basin temperature (Oliver et al. 2021, for Zwart 2023) [V] | https://doi.org/10.5066/P9GD8I7A | ~950 MB | CSV/zip | USGS [U] | 456 reaches + 2 reservoirs, 1980–2021 | Temperature, flow, reservoir releases, **archived GEFS forecasts** | None | 1–7-d forecasts: RMSE 1.1–1.4 °C (1 d), 1.4–1.9 °C (7 d) |
| **LakeBeD-US** (CS Edition) [V] | https://huggingface.co/datasets/eco-kgml/LakeBeD-US-CSE; Ecology Edition doi:10.6073/pasta/c56a204a65483790f6277de4896d7140; ESSD 17:3141 (2025) | 1.14 GB | Parquet | CC BY 4.0 | 21 US lakes (NTL-LTER, NEON lakes, Niwot, Virginia reservoirs) | 17 WQ parameters; high-frequency sensors + profiles + samples | None | LakeFM (KDD 2026) reports MSE/CRPS vs Chronos-2 / Moirai / MOMENT / iTransformer on held-out lakes |
| TIME benchmark, Water_Quality_Darwin [V record] | hf.co/datasets/Real-TSF/TIME (predictions: Real-TSF/TIME-Output) | small | – | **CC-BY-NC** | 7 deployments, 2015–2018 | 15-min CNDC, DO, PSAL, TEMP, TURB, CPHL | None | TSFM MAE relative to persistence (a.4) [D] |
| CAMELS-Chem [V] | https://www.hydroshare.org/resource/841f5e85085c423f889ac809c1bed4ac | 1.1 GB total (flat CSV 16 MB) | CSV, xlsx, SQL | CC BY 4.0 | 516 CAMELS catchments, 1980–2018 | 18 constituents (DO, pH, T, NO3, DOC, ions); **discrete samples** + Q; NADP deposition | None | Zhi 2021 DO LSTM (236 basins; NSE [U]); Liu et al. 2025 HESS: LSTM DO median KGE 0.70 |
| Lake temperature PGDL releases [V] | doi:10.5066/P9AQPIVD (Read 2019); doi:10.5066/P9CEMS0M (Willard 2022) | 3.2 GB; 7.3 GB | CSV/zip | USGS [U] | 68 Midwest lakes; 185,549 lakes | Temperature by depth; daily surface temperature | None | Median RMSE 1.65 °C (PGDL); 1.24 °C (held-out lakes) |
| LamaH-CE / LamaH-Ice [V] | doi:10.5281/zenodo.4525244; doi:10.4211/hs.86117a5f36cc4b7c90a5d54e18161c91 | 16.3 GB / ~10 GB | CSV / zip | CC BY 4.0 | 859 / 107 catchments | Streamflow + meteorology; **no WQ** | None | – |
| Caravan v1.6 + extensions [V] | doi:10.5281/zenodo.6522634 | 24.8 GB (netCDF) | netCDF/CSV | CC BY 4.0 | 6,830 basins (+ GRDC 5,357, national extensions, MultiMet) | Daily streamflow, ERA5-Land; **no WQ in any extension** | None | No model benchmark in the paper |
| WaterBench-Iowa [V] | doi:10.5281/zenodo.7087806 | 51 MB | CSV | CC BY 4.0 | 125 Iowa gauges, hourly, WY2012–2018 | Streamflow, precipitation, ET | None | 120-h forecasts; median NSE 0.74 / KGE 0.79 |
| GRQA / GEMStat / GLORICH [V] | doi:10.5281/zenodo.5097436; zenodo.org/records/18459694; doi:10.1594/PANGAEA.902360 | 3.1 GB / 201 MB / 150 MB | CSV | CC BY 4.0 / CC BY 4.0 (open part) / **CC BY-NC-SA** | Global | **Discrete samples only** | None (GEMStat portal: form) | None |
| Monash / GIFT-Eval "Saugeen" [V] | zenodo.org/record/4656058 | 29 KB | .tsf / arrow | CC BY 4.0 | 1 daily river-flow series | Flow | None | Monash MASE: CatBoost 1.41 best; TimesFM zero-shot 2.23 [D]. GIFT-Eval normalized CRPS in a.4 |
| fev-bench, TFB, LTSF sets | – | – | – | – | – | **No hydrology or WQ tasks** (fev task lists checked) | – | – |

**Unverified** [U]: CAMELS-CH-Chem; Kaggle WQ sensor datasets (e.g., Brisbane); any VERA or usgsrc4cast synthesis paper.

### b.3 EFI forecasting challenges: mechanics and current numbers

**Mechanics [V].**
- **Targets:** daily means of NEON 30-min sensor data in the top 1 m.
- **Horizon:** the documentation says 30 days; `challenge_configuration.yaml` sets `max_horizon` to 35.
- **Submission format:** a file `aquatics-YYYY-MM-DD-<model_id>.csv.gz` with columns `project_id, model_id, datetime, reference_datetime, duration, site_id, family, parameter, variable, prediction`. Families are ensemble, normal, lognormal, …, and bernoulli for binary targets.
- **How to submit:** anonymous S3 `put_object` to bucket `submissions` at `s3-west.nrp-nautilus.io` (R `neon4cast::submit()`). The model_id must be registered.
- **Weather drivers:** NOAA GEFS 35-day forecasts are provided.
- **Scoring:** CRPS and log score (scoringRules).
- **Score location:** public parquet at `s3://bio230014-bucket01/challenges/scores/bundled-parquet/project_id=neon4cast/...` on `sdsc.osn.xsede.org`. Anonymous reads work.
- **Baselines:**
  - `climatology`: normal distribution from day-of-year mean and SD, using prior years only.
  - `persistenceRW`: random walk with 200 members.
  - Organizer `tg_*` models (ARIMA, ETS, TBATS, lm, lasso, random forest, XGBoost, bagged MLP, BRNN, SVM), which ran 2023 to July 2025.

**Current state [V].**
- The aquatics targets were last modified 2026-08-15 and end 2026-07-31.
- The target-generation job failed on each run from 2026-10-18 through 2026-10-21, so live neon4cast scoring is stalled.
- Non-baseline aquatics submissions mostly stopped after 2025-07.
- No LSTM, Transformer or TSFM has a sustained neon4cast aquatics record.

**neon4cast skill vs climatology [D]** (1 − CRPS/CRPS_clim; 2023-01 to 2025-07):

| Model | Target, site type | 1 d | 7 d | 14 d | 30 d |
|---|---|---|---|---|---|
| tg_arima | DO, lakes | +0.40 | – | ~0 | −0.27 |
| tg_arima | DO, streams | +0.30 | – | ~0 | −0.57 |
| tg_randfor | DO, lakes | – | +0.07 | +0.13 | +0.12 |
| FLARE-GLM | Temperature, lakes | +0.44 | +0.25 | −0.03 | −0.21 |
| tg_randfor / lasso / lm | Temperature, rivers and streams | below 0 at nearly all horizons | | | |
| Any model | Chl-a, lakes | negative | | | |

- Climatology CRPS, 2024–2025: DO 0.68 / 0.52 / 0.44 mg/L (lakes / rivers / streams); temperature 1.13 / 0.90 / 1.09 °C; chl-a 3.2 / 3.9 µg/L.
- persistenceRW is already worse than climatology at 1 d, probably because of target latency (≥ 3 d) and the wide random-walk spread.

**usgsrc4cast (river chlorophyll) [D].** Skill vs climatology at 1 / 7 / 14 / 30 d:

| Model | 1 d | 7 d | 14 d | 30 d |
|---|---|---|---|---|
| Chronos-2 zero-shot (from 2026-08-31) | +0.75 | +0.54 | +0.43 | +0.28 |
| TiRex zero-shot (from 2026-02) | +0.56 | +0.31 | +0.21 | +0.13 |
| USGS LSTM + EnKF | +0.28 | +0.09 | +0.03 | ≈0 |
| USGS random forest | +0.46 | +0.08 | −0.08 | −0.19 |

On 2,357 matched forecasts (2026-08-31 to 2026-10-20), mean CRPS in µg/L:

| Horizon | Chronos-2 | TiRex | Persistence | USGS LSTM | Climatology |
|---|---|---|---|---|---|
| 1 d | **1.65** | 1.72 | 2.03 | 3.69 | 6.14 |
| 7 d | 3.08 | **2.75** | 3.15 | 4.63 | 6.78 |

**VERA [D]** (sample sizes ≤ ~250 for newer models):
- Chronos-2 DO skill: +0.56 (1 d), +0.43 (7 d), +0.17 (30 d).
- Chronos-2 chl-a skill: +0.40 (1 d), ≈0 (30 d).
- FLARE-AED: chl-a +0.63 at 1 d; DO negative beyond 1 d.
- `Bloom_binary` (Brier score): Chronos-2 +0.22 and FLARE +0.30 at 1 d, but **nothing reliably beats climatology at ≥ 7 d**.

**What this means for SENTINEL.** These challenges give the most defensible external SOTA test: forecasts made before the observations, a proper scoring rule, documented baselines and public competitor scores.
- Zero-shot Chronos-2 / TiRex is the bar on usgsrc4cast and VERA.
- ARIMA (DO, 1–3 d) and FLARE (lake temperature, 1–7 d) are the bar on neon4cast.
- Climatology is the bar for chl-a and for horizons ≥ 15 d.
- Because neon4cast live scoring is stalled, either recompute scores with the published scoring code on new NEON data (token needed), or prioritise VERA and usgsrc4cast, which are current.

### b.4 Other open high-frequency sources

Verified by test requests on 2026-10-22; status codes are from our requests.

| Source | Access (verified) | Account? | License | Sites / variables / resolution / period | Notes |
|---|---|---|---|---|---|
| **GLOS Seagull ERDDAP** (Lake Erie HAB network) | https://seagull-erddap.glos.org/erddap/ → 200; e.g. `tabledap/GLERLWE2_archive.csv?...` → 200, 15-min | None | Free use and redistribution ("not for legal use") | 3,625 datasets, **202 with phycocyanin** (GLERL WE2/4/8 buoys, Toledo crib, Cleveland/Ohio intakes); PC/BGA RFU, chl, DO, T, turbidity, fDOM, SC; QARTOD flags; 2015 to 2026 (seasonal) | **Best open US HAB source.** CF units: DO in kg/m³, T in K |
| **UK Environment Agency Hydrology API** | `https://environment.data.gov.uk/hydrology/id/measures/KEWPON-do-i-subdaily-mgL/readings.csv?...` → 200 | None | OGL v3.0 | DO 1,912 (154 active), conductivity 1,942, pH 1,852, **ammonium 1,818**, turbidity 1,737, chl 553, BGA 104 stations; 15-min; e.g. Thames 2007 onwards | Largest non-USGS sonde network. Ammonium is a sewage proxy. Many short incident deployments; mostly "Unchecked" quality |
| **NOAA NERRS SWMP (CDMO)** | https://cdmo.baruch.sc.edu/aqs/ (zip form asks email/org/purpose); web services → "Invalid ip" (needs static-IP registration) | Email form / IP registration | Cite doi:10.25921/vw8a-8031; **"should not be archived by outside sources for distribution"** | 29 reserves, 173 WQ stations; T, SC, salinity, DO, pH, turbidity, chl fluorescence; 15-min since ~1995 (30-min before 2007) | Rich in hypoxia. Freshwater or tidal-fresh sites: Old Woman Creek (Lake Erie), Lake Superior, Hudson, Jug Bay |
| **NTL-LTER buoys** (Mendota knb-lter-ntl.129; Trout Lake etc.) | DataONE mirror `https://gmn.lternet.edu/mn/v2/object/<pid>` → 200 (hourly CSV 10 MB) | **EDI API needs a free account since 2026-07-30** (PASTA → 403); mirror is open | CC BY 4.0 | Chl/PC RFU, DO, pH, fDOM, turbidity, SC, pCO2, PAR, met; 1-min / hourly / daily; 2006–2025 | Long lake records with phycocyanin |
| Virginia Reservoirs FCR/BVR (edi.271) + FLARE GitHub | GMN mirror → 206; GitHub → 200 | None via mirror | **CC BY-NC 4.0** | Multi-depth T/DO, EXO chl, BGA-PC, fDOM, turbidity, SC; 10-min; 2018 to real time | Experimental hypolimnetic oxygenation confounds DO |
| **StreamPULSE** | https://data.streampulse.org/download_bulk → 200 zips (core 508 MB) | None (tick the policy box) | ODC-By 1.0 | **741 sites** (356 Powell Center, 274 leveraged, 68 NEON, 43 core); DO, T, light, depth/Q, SC, pH, turbidity; 5–15-min; metabolism outputs | About 317 sites outside USGS/NEON; dedupe |
| CA CDEC + DWR CEMP (edi.1177) | `https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?Stations=RRI&SensorNums=61&dur_code=E&...` → 200 | None | CDEC [U]; CEMP CC BY 4.0 | 15-min DO, EC, T, turbidity, chl, pH; CEMP 15 stations, 2005–2023 QA'd | RRI (Stockton) is a recurrent hypoxia site; EC tracks salinity intrusion. Don't use CNRA links with embedded `?key=` |
| IOOS Sensors ERDDAP (Axiom) | https://erddap.sensors.ioos.us/erddap/ → 200 (1,474 DO datasets) | None | Free; cite the regional association | MD DNR (29), CBIBS (19), some NERRS, USGS | Mostly coastal; mirror for MD DNR / CBIBS |
| ECCC Automated Fresh Water Quality | `https://data-donnees.az.ec.gc.ca/api/path_contents?path=...` → 200 | None | OGL-Canada | 23 stations; T, DO, pH, SC, turbidity; hourly | Transboundary rivers |
| Logan River Observatory (HydroShare) | `https://www.hydroshare.org/hsapi/resource/728566f42906412698b09f6fe2f7cd02/files/` → 200 | None | CC BY 4.0 | ~6+ aquatic sites; DO, SC, turbidity, pH, T, stage, Q; 15-min; 2013–2026 | Also the PyHydroQC labelled-anomaly source |
| Others verified | NDBC `.ocean` (170 stations, mostly marine); TWDB coastal (163 estuarine stations); Rijkswaterstaat (2,500 locations, POST catalogue); Hub'Eau temperature (872 stations, hourly); Queensland WMIP (CC BY 4.0; WQ codes [U]) | None | various | | Pretraining or transfer tests |
| Pretraining / priors (discrete) | Jane et al. 2021 lake O2/T profiles (edi.698, 393 lakes); Pilla et al. (edi.705, 153 lakes) | EDI | [U] | Profiles | Priors only |

**Blocked or unverified** [U]:
- Iowa IWQIS (host not responding; its USGS-run sites are in NWIS anyway).
- SFWMD DBHYDRO (client ID/secret needed).
- TCEQ CWQMN (web app only).
- BoM KiWIS (504).
- NZ Hilltop (unreachable).
- Bavaria GKD (403).
- China CNEMC (no open archive).
- **Zenodo returned 403 "unusual traffic" to this server**, so GRQA/LamaH downloads may need another network.

**Recommended additions, ranked by value for SENTINEL's disaster targets:**
1. GLOS (HABs, drinking-water intakes; about 1 day of work).
2. UK EA (sewage/ammonium, scale, transfer test; 1–2 days).
3. NTL-LTER + FCR/BVR + VERA (lakes, phycocyanin, a live benchmark; about 2 days).
4. NERRS SWMP (hypoxia; 1–2 days plus registration; no redistribution).
5. CDEC/CEMP (hypoxia, salinity intrusion; under 1 day).
6. StreamPULSE (metabolism, DO sag; under 1 day).

### b.5 Derived ecological features: stream metabolism (GPP / ER / K600)

| Item | What | Numbers | Access |
|---|---|---|---|
| **Appling et al. 2018, *Sci. Data* 5:180292**, "The metabolic regimes of 356 rivers in the United States" [V] | Daily GPP, ER, K600 for 356 US rivers, Oct 2007 to Jan 2017, from USGS + NASA inputs | 490,907 site-days; model confidence H 299 / M 63 / L 71 of 433 runs | Data release doi:10.5066/F70864KX (ScienceBase item 59bff507e4b091459a5e0982, 9 child items). Small files downloadable without an account (site_data.tsv 0.12 MB; diagnostics.zip 0.03 MB). daily_predictions.zip 101 MB, fits.zip 176 MB, inputs.zip 1.23 GB and timeseries.zip 7.47 GB sit behind manager links; scripted bulk download [U] |
| streamMetabolizer (Appling, Hall, Yackulic & Arroita 2018, *JGR-Biogeosci.* 123:624) [V] | Hierarchical state-space model; K600 pooled against discharge; MLE and Bayesian (Stan) | Daily estimation error reduced by up to 70% | github.com/DOI-USGS/streamMetabolizer (R + rstan; limited-maintenance archive package) |
| StreamPULSE / Bernhardt et al. 2022, *PNAS* 119:e2121976119 [V] | Annual metabolic regimes; light and flow stability dominate | > 200 rivers vs > 150 terrestrial sites | data.streampulse.org (741 sites; bulk zips) |
| Savoy et al. 2019 L&O; Koenig et al. 2019 L&O Lett.; Bertuzzo et al. 2022 L&O; Battin et al. 2023 *Nature*; Ulseth et al. 2019 *Nat. Geosci.*; Hall & Ulseth 2020 WIREs Water [V] | Regime typologies, ER partitioning, network GPP, gas-exchange regimes, review | e.g. Bertuzzo: 53–74% of ER dynamics explained at 5 streams | – |
| NEON [V] | **No metabolism data product** (203-product catalog scanned). Inputs exist: DP1.20190.001 reaeration (24 sites), DP1.20097.001 dissolved gases, PAR, DO, temperature, discharge | – | github.com/NEONScience/NEON-reaeration (reaRate) |
| ML / DL / differentiable metabolism | Appling et al. 2022 (ML for inland waters, chapter); Sadler 2022 multi-task DL as a precedent | – | **We found no verified DL, differentiable or hybrid GPP/ER/K600 estimator** before the search budget ran out, so novelty is [U] |

**How SENTINEL should use metabolism.**
1. **As interpretable latent variables.** Use a differentiable one-station DO model inside the DO forecaster; see d.6.
2. **As validation targets.** Out-of-sample agreement with the Powell Center and StreamPULSE estimates at H/M-confidence sites.
3. **As features for downstream ecology** (goal 4), e.g. GPP/ER ratios, which indicate eutrophication.

### b.6 Covariates and static attributes (verified)

| Source | Access | Use |
|---|---|---|
| Open-Meteo Historical Forecast API | `https://historical-forecast-api.open-meteo.com/v1/forecast?...` → 200. The docs say it stitches the first hours of each run, from 2022 | Near-analysis weather |
| Open-Meteo Previous Runs API (described on the same docs page) | Fixed leads of day 1–7; "Data starts from January 2024 (GFS from March 2021, JMA from 2018)" | **Operational** (forecast) covariates for test periods |
| NOAA HRRR on AWS (`noaa-hrrr-bdp-pds`) | Bucket listing → 200; about 150 MB per CONUS surface file per hour | Hourly NWP (large) |
| NOAA GEFS v12 reforecast on AWS (`noaa-gefs-retrospective`) | Bucket listing → 200 | Operational-style covariates for training years |
| neon4cast GEFS drivers | `bio230014-bucket01/neon4cast-drivers/noaa/gefs-v12` | Ready-made 35-d ensembles at NEON sites |
| USGS NLDI | `https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-05586300` → COMID 3607116; `/basin` → polygon | Site → NHDPlus COMID, basins, upstream/downstream navigation (for station grouping) |
| EPA StreamCat API | `https://api.epa.gov/StreamCat/streams/metrics?name=pcturbhi2019,pctcrop2019,elev&aoi=ws&comid=3607116` → crop 65.66%, high-intensity urban 1.49%, elevation 206 m | Hundreds of watershed attributes (land cover, imperviousness, dams, soils) |

---

## (c) Gaps where a SOTA or first-in-class claim is realistic (and where it is not)

The searches behind this table are date-bounded (2026-10-22) and partly limited by the search quota. Re-check every "first" claim before submission.

| # | Claim | Literature status | What must be shown | Realism |
|---|---|---|---|---|
| 1 | **First open benchmark for multi-variable, sub-daily sonde forecasting, virtual sensing and event warning, with station × time holdouts** (USGS + NEON, external networks) | Not found. Closest: TIME Darwin (coastal, 7 deployments, NC license), LakeBeD-US (21 lakes), EFI challenges (daily; 34 / 2 / 15 sites) | Release data manifests, splits, baselines and scoring code; reproduce baseline numbers | **High.** A data/benchmark contribution that doesn't depend on winning |
| 2 | **SOTA on EFI challenges** (VERA DO/chl-a/bloom, usgsrc4cast chl-a, neon4cast DO/temperature) | Current bars: Chronos-2/TiRex zero-shot, FLARE, ARIMA, climatology | Prospective submissions, or matched hindcasts with archived GEFS and the official scoring code | **Medium.** Chronos-2 is already strong; lake chl-a and ≥ 15 d are near-unpredictable |
| 3 | **First fine-tuned, covariate-aware TSFM for multi-site river WQ** | Only zero-shot challenge entries; Zheng 2026 (total N, 12 Chinese stations); LakeFM (lakes); HESS 2025 (daily DO, LSTM ≈ Transformers) | Fine-tuned Chronos-2 beats (i) itself zero-shot, (ii) a multi-site LSTM, (iii) GBM, (iv) naive baselines, on unseen stations × future time | **High** for first-in-class; medium for clear SOTA |
| 4 | **Event early warning with event-level metrics and placebo controls** (hypoxia, salinization, turbidity, bloom proxy) | Not found in WQ (a.5). Hypoxia characterized but not forecast at scale (Blaszczak 2023) | BSS > 0 vs base rate and persistence-of-state; POD/FAR/lead time at ≤ 2 false alarms per station-year; placebo alarm rate ≈ nominal | **High** for first-in-class; skill beyond 7 d likely small (VERA bloom) |
| 5 | **Virtual sensing at unmonitored stations** (nitrate, fPC, turbidity/SSC, DO) | Weak published PUB: nitrate KGE 0.18, SSC R² 0.55 | Beat surrogate regressions and GBM on HUC-grouped held-out stations; ideally replicate the Pandit/Song settings | **High** (no persistence to hide behind; largest headroom) |
| 6 | **Differentiable metabolism-informed DO forecasting**, yielding GPP/ER/K600 | Novelty [U] | Matches or beats a pure DL model on DO (especially extremes); metabolism estimates agree with Powell Center/StreamPULSE at H-confidence sites | **Medium** (research risk; strong interpretability payoff) |
| 7 | **Cross-network / cross-country transfer** (train USGS → test NEON, UK EA) | Not found for WQ sondes | Zero-shot and few-shot skill vs locally trained baselines | **Medium** |
| 8 | Sewage-pollution event detection (UK EA ammonium + DO + turbidity) | Not searched [U] | Event-level metrics against EA incident records | Medium |

**Not realistic, and should not be claimed:**
- large gains over persistence at ≤ 24 h for temperature, SC or pH;
- beating day-of-year climatology at 15–30 d for lake temperature or chl-a;
- skilful bloom-probability forecasts beyond about 7 d;
- "SOTA" from random-split or single-site results.

---

## (d) Recommended approach for SENTINEL's sensor model

### d.1 Benchmark: "SENTINEL-Sonde" (working name)

**Networks.**
- **Primary:** USGS continuous stream and river sites.
- **External test network:** NEON (34 sites; 1/5-min resampled to 15-min means). It has different sensors, QA and site types.
- **Optional tracks:**
  - lakes/HAB: GLOS, NTL-LTER, VERA/LakeBeD-US;
  - estuary hypoxia: NERRS SWMP (respect its no-redistribution clause);
  - transfer and sewage: UK EA.

**Site selection.** Apply these rules from metadata, then verify actual completeness:
- At least 5 years POR overlap of DO + SC + temperature (USGS: about 610 stream sites by metadata).
- At least 60% 15-min completeness in every split period.
- **Subsets:**
  - turbidity: about 377 sites;
  - nitrate: about 124 sites with ≥ 5 y; about 61 with the full DO + SC + turbidity + pH + NO3 set;
  - algal: 133 sites with any PC series.
- Record every `time_series_id`, sublocation and unit. Splice only after an overlap or offset check.

**Targets.**
- Water temperature
- DO: mg/L, plus daily minimum
- SC, pH
- Turbidity: log
- Nitrate: NO3+NO2-N
- fDOM
- fChl and fPC: **RFU and µg/L never mixed**
- log-discharge or stage

**Horizons and issue time.**
- Hourly product: +1, 3, 6, 12, 24, 48, 72 h.
- Daily product (mean / min / max): +1 … +14 d.
- Issue at 00 UTC daily, with a context cutoff of issue time minus 2 h (telemetry latency).
- Report skill at every horizon.

**Tasks.**
- **F (forecast):** probabilistic forecasts at every horizon for all available targets.
- **N (virtual sensing):** estimate a withheld variable from co-located sensors plus covariates.
  - N-a: the variable is withheld for a period at a seen station.
  - N-b: unseen stations.
- **E (event early warning):** probability of event onset (d.4) within 0–24 h, 1–3 d and 4–7 d.

**Inputs.**
- Past sensors, with missingness masks and qualifier flags as features.
- Past meteorology.
- Static attributes (NLDI → StreamCat).
- Calendar and solar geometry (clear-sky PAR).
- Future weather in two **strictly separated regimes:**
  - **Oracle (O):** observed future meteorology. An upper bound only, never called a "forecast".
  - **Operational (Op):** archived NWP forecasts, i.e. Open-Meteo Previous Runs (GFS from 2021-03, most models from 2024-01) or the GEFS v12 reforecast and neon4cast GEFS drivers for earlier years. **Only Op results may be called forecasts or early warnings.**

### d.2 Leakage-free splits (by station and by time)

**Time blocks** (water years, WY):

| Block | Water years | Dates | Purpose |
|---|---|---|---|
| Train | WY2008–WY2019 | 2007-10-01 to 2019-09-30 | Training |
| Validation | WY2020–WY2021 | | Model selection |
| Test-A | WY2022–WY2025 | | Main test |
| **Test-B (post-release)** | WY2026 | 2025-10-01 to 2026-09-30 | Used for every pretrained TSFM |
| **Test-C (prospective)** | – | from 2026-11 | Forecasts issued live from a frozen model and scored as data arrive (mirrors the EFI challenges); the cleanest guard against contamination and hindsight |

**Embargo.** Drop every sample whose context or forecast window crosses a block boundary: at least 42 days (28-day context + 14-day horizon). Nearing et al. 2024 used a 1-year buffer, which is also acceptable.

**Station folds.**
- Five folds, grouped by HUC4.
- Flow-connected stations (via NLDI upstream/downstream navigation) go in the same fold.
- **Report three cells:**
  1. seen stations × Test-A;
  2. unseen stations × training period;
  3. **unseen stations × Test-A, the headline**.
- Report NEON (an unseen network) × Test-A/B separately.

**Fit on training data only:** scalers and transforms, climatologies, imputers, QA thresholds, percentile event thresholds and conformal calibration (calibrated on validation). For unseen stations, per-station statistics may use only that station's training-period history; otherwise use global statistics.

**Freeze and audit.**
- Store raw pulls with retrieval time, `approval_status` and `last_modified`.
- Score on Approved values (Provisional-only results reported separately).
- Never re-pull test data after model selection.
- Commit the scoring script and record its hash before unblinding.
- The previous audit found 11/17 event stations overlapping pretraining data (10_local_inventory.md D5). Enforce **station-ID disjointness tests** in code.

### d.3 Baselines (re-run on the frozen benchmark, per variable and horizon)

1. **Persistence:** last value, plus diel persistence (same clock time, last observed day).
2. **Station climatology:** day-of-year × hour-of-day mean and quantiles, training years only.
3. **Damped anomaly persistence:** NSE 0.95 / 0.86 at 14 d for temperature / DO in our pilot.
4. **statsforecast:** AutoETS, AutoARIMA, Theta (seasonal 96 at 15-min, 365 daily).
5. **LightGBM / XGBoost**, global, on lags, rolling statistics, covariates and attributes. This is probably the hardest non-deep baseline (Smith 2024; NEON tg_randfor).
6. **Multi-station LSTM** (NeuralHydrology-style): static attributes, CMAL or quantile head, recent observations with a missing-data flag (Nearing 2022).
7. **Deep TS models** via neuralforecast or Time-Series-Library: N-HiTS / NBEATSx, TFT, PatchTST, iTransformer, TimeMixer, TSMixerx, TiDE. Fix the `drop_last` bug.
8. **Zero-shot TSFMs**, scored on Test-B/C:
   - Timestral-209M (strongest public; no official fine-tuning)
   - Chronos-2 (+ covariates)
   - TimesFM-2.5 (XReg) and TimesFM-3 (non-commercial license)
   - t0-gamma
   - Toto-2.0 (≤ 313M)
   - TiRex 1.x (torch backend)
   - Moirai-2 (separate env)
   - Sundial
   - TabPFN-TS
9. **Physics references:**
   - DO_sat(T, P) × typical saturation ratio;
   - streamMetabolizer forward model with the last estimated GPP/ER/K600;
   - an air → water temperature regression;
   - surrogate regressions for N tasks (turbidity → SSC, SC → Cl).

### d.4 Event definitions (forecast targets, labelled from QC'd Approved data)

**Rules that apply to every event.**
- Use absolute thresholds from authorities where they exist (see the table).
- Use relative thresholds as training-period station percentiles.
- **Onset** requires the trigger condition for at least the minimum duration D.
- Exceedances separated by gaps under G = 24 h are merged.
- A new onset needs ≥ 7 d of non-event first.
- **QA exclusion:** no onset when trigger values carry equipment, ice or maintenance qualifiers (e.g., `EQUIP`, `ICE`, `Eqp`, `Mnt`, `Dis`, `***`), or within 24 h after a sensor service.
- **Cross-variable coherence check:** for example, a DO crash should agree with the diel DO–T–light structure, to reject fouling artefacts.

| Event | Sensor rule (proposal) | Authority / basis | Caveats |
|---|---|---|---|
| **Hypoxia** | DO < 2 mg/L for ≥ 1 h (4 consecutive 15-min values). Also report tiers: < 3 mg/L (EPA warm-water 1-day minimum, other life stages) and < 5 mg/L (warm-water early-life-stage 1-day minimum) | Blaszczak et al. 2023 (DO < 2 convention; median event 3 h, nocturnal); EPA 1986 Gold Book | Needs sub-daily data. Check USGS `SuppressData` thresholds (250 series ≥ 2 mg/L). Our pilot: 5/18 random long-record stream sites had ≥ 1 day with daily-min DO < 2 mg/L in WY2016–2025 (55 onsets) [D] |
| Cold-water thermal stress | 7-day average of daily maxima (7-DADMax) > 16 °C (core salmonid) / 20 °C (warm-water tier) | Washington WAC 173-201A-200 (tiers 12–20 °C) | Defined on 7-DADMax, not instantaneous values |
| **Salinization pulse** | Site-regressed chloride > 230 mg/L (chronic) or > 860 mg/L (acute), held ≥ 6 h; fallback SC > training P99.5 for ≥ 6 h | EPA chloride criteria (1988); SC–Cl regressions (Moore 2020; Granato & Smith 1999) | SC also reflects non-salt ions. The v1 manifest used max(median + 4·IQR/1.349, P99.5) for ≥ 6 h, a reasonable relative rule |
| **Turbidity / sediment event** | Turbidity > max(background + 10 FNU, 1.2 × background) for ≥ 6 h, background = trailing 7-d median; also > training P95 | Washington turbidity criteria (+5 to +10 NTU or +10 to 20% over background); USGS TM 3-C4 for SSC | FNU ≠ NTU; rain-driven events are real events, so don't filter them out |
| **Algal-bloom proxy** | fChl > 20 µg/L (VERA `Bloom_binary` convention) or fChl/fPC > training P95 for ≥ 24 h (lakes, large rivers) | WHO 2021: Alert 1 at 12–24 µg/L chl-a when cyanobacteria dominate; EPA 2019 MC 8 µg/L / CYN 15 µg/L (needs grab samples) | **No consensus PC trigger** (e.g., 10 µg PC/L in Fournier 2024); RFU is instrument-specific; validate against WQP toxin/chl samples |
| Nitrate exceedance | NO3-N > 10 mg/L for ≥ 6 h | EPA drinking-water MCL (10 mg/L as N) | Relevant to intakes (e.g., Iowa) |
| pH excursion | pH < 6.5 or > 9.0 for ≥ 2 h | EPA Gold Book range | Diel pH > 9 in productive rivers, so require a duration |
| Fecal-contamination proxy | Nowcast of E. coli > 235 CFU/100 mL from turbidity + Q + season | EPA Beach Action Value; Aulenbach & McKee 2020 | Standard fDOM sensors are **not** TLF sensors, so TLF thresholds (1.3 / 6.9 ppb) don't transfer |

**Scoring** (unit = station-issue-day):
- **Label:** onset within the lead window, given no event at issue time. Issue times during an ongoing event are scored separately as "continuation".
- **Probabilistic metrics:**
  - Brier score and BSS vs (i) station × month climatological base rate and (ii) persistence-of-state;
  - reliability diagrams;
  - PR-AUC (base-rate-aware), with ROC-AUC secondary;
  - SEDI for rare events (Ferro & Stephenson 2011).
- **Operational metrics** at an operating point fixed on validation (≤ 2 false alarms per station-year): POD, FAR, CSI with a stated tolerance window, false alarms per station-year, and the lead-time distribution of the first correct alarm. No point adjustment.
- **Mandatory baselines:**
  - climatology;
  - persistence-of-state;
  - current-value rules (e.g., alarm if DO < 4 mg/L now);
  - GBM / logistic regression on the same features;
  - random alarms at matched rate;
  - an untrained / random-score detector (Kim et al. 2022).
- **Mandatory negative controls**, given the previous Track-A failure:
  1. **Placebo dates:** the alarm rate in windows before random season- and station-matched clean dates must not exceed the nominal false-alarm rate; report lift = rate before onsets ÷ rate before placebos.
  2. Labels permuted in time within each station: skill must drop to about 0.
  3. Labels swapped between stations.
  4. Covariates shuffled.
  5. **Input-truncation audit:** re-score with inputs cut 1 h earlier. Skill must change smoothly; a cliff signals look-ahead.
- **Counts and intervals:** report events per split and per station-year. With fewer than 30 test events, give exact (Clopper–Pearson) CIs and call the result preliminary.
- **Never evaluate only on observed extremes** (the forecaster's dilemma; Lerch et al. 2017).

### d.5 Metrics (forecast and virtual-sensing tasks)

- **Deterministic, per station:** RMSE, MAE, NSE, KGE. Report the median and IQR across stations, and **always** the skill relative to the best naive baseline (MSE-SS vs persistence, diel persistence and damped anomaly).
- **Probabilistic:**
  - CRPS from 19 quantiles (5–95%), i.e. the weighted quantile loss used by GIFT-Eval and fev;
  - CRPSS vs climatology and vs a persistence-error ensemble;
  - 50/90% interval coverage and width;
  - PIT histograms.
- **Comparability:** also report GIFT/fev-style scaled MASE and SQL relative to seasonal naive, with geometric-mean aggregation and win rates.
- **Tails:** threshold-weighted CRPS (Gneiting & Ranjan 2011; Allen et al. 2023) at the event thresholds.
- **Uncertainty and tests:** station-level block bootstrap (resample stations; monthly blocks within stations) for 95% CIs; Diebold–Mariano tests with HAC variance for paired comparisons; win/tie/loss counts across stations.

### d.6 Architecture most likely to win

Run a staged bake-off with a fixed protocol rather than committing to a single architecture. Model selection uses the validation block only.

**(A) TSFM-FT: Chronos-2 fine-tuned** (primary; TimesFM-2.5 + LoRA as the Apache-2.0 second choice).
- **Why Chronos-2:**
  - Apache-2.0 license.
  - Native past and known-future covariates and multivariate group attention, which suits co-located sensors with missing channels.
  - An official full and LoRA `fit()` that runs in fp32 on Turing.
  - Already the strongest zero-shot entry in the EFI river-chlorophyll and VERA challenges.
- **Setup:**
  - Context: 2,048 steps at 15-min (≈ 21 d), or 1,024–2,048 at hourly (43–85 d), plus daily-aggregate tokens for long memory.
  - Covariates: operational NWP, solar geometry, catchment embeddings.
  - Output: quantiles.
  - fp32 full fine-tuning (≈ 120M params fits in 11 GB with gradient checkpointing), or fp16 AMP with GradScaler for LoRA.
  - **Avoid bf16** (7 TFLOPS on these GPUs).

**(B) Hydro-LSTM (NeuralHydrology-style)**, the strongest hydrology prior.
- Multi-station LSTM with static attributes and recent-observation autoregression with missing-data flags.
- MTS-LSTM for 15-min + daily inputs.
- CMAL or quantile head.
- Optionally a Mamba encoder variant (mamba-ssm 2.2.x; fp16 only on Turing).

**(C) Differentiable metabolism head for DO** (first-in-class candidate; novelty [U]).
- The network predicts daily GPP, ER and K600, with positivity constraints and priors from the Powell Center / StreamPULSE estimates.
- DO is integrated by the one-station model:
  `dDO/dt = GPP·PAR(t)/ΣPAR(day)/z − ER/z + K_O2(T)·(DO_sat(T,P) − DO)`
  where z is mean depth (from NEON or the USGS rating), and K_O2 is converted from K600 via the Schmidt number.
- Trained end-to-end on 15-min DO.
- Benefits: physically constrained DO extremes (where pure DL misses peaks and troughs; Zhi 2021), interpretable GPP/ER/K600 for SENTINEL goal 3, and a bridge to downstream ecology.
- Validate the latent estimates against independent metabolism estimates at held-out sites.

**(D) Event head.**
- Exceedance probabilities derived from the predictive distribution, plus a directly trained onset classifier (focal loss).
- Calibrated with adaptive conformal inference (Gibbs & Candès 2021) on validation.

**Final model:** a validation-weighted ensemble of A, B, C (for DO) and GBM. Multi-model ensembles are consistently the most robust in freshwater forecasting challenges (Olsson 2024, 2025).

**Honest expectations:**
- **1–24 h (temperature, DO, SC, pH):** small gains over persistence; prove them with DM tests.
- **3–14 d (temperature, DO):** damped-anomaly and climatology baselines are already near 0.95 / 0.86 NSE. Gains need operational weather, and **15–30 d is climatology territory**.
- **Turbidity, SC, nitrate, phycocyanin at 2–7 d:** the most room for gains, limited by precipitation-forecast error.
- **N-b virtual sensing at unseen stations:** the most likely large, defensible win.
- **Events:** first-in-class rigour is achievable; skill beyond 7 d will be small.

### d.7 Interfaces to the rest of SENTINEL

- **Fusion (goal 2).** Expose:
  - a per-station, per-time latent state, with per-variable embeddings and an explicit "missing" token;
  - a calibrated predictive distribution per variable.

  The model must accept any subset of sensors, which Chronos-2 group attention supports natively, so fusion can drop modalities.
- **Drone and satellite (goals 1 and 6).** Sonde values (turbidity, fChl, fPC, SC, temperature) are the ground truth for RGB+NIR and Sentinel-2 matchups. Store exact timestamps, not nominal dates, which was a v1 defect.
- **Simulation and interpretability (goal 3):**
  - metabolism latents;
  - attribution of DO to temperature / light / flow, to compare with Zhi et al. 2023 (temperature > light >> flow);
  - counterfactual weather scenarios, run through the Op-regime model only.
- **Applications (goal 4):**
  - fish-kill risk (hypoxia + thermal stress);
  - HAB / drinking-water-intake risk (GLOS, intakes);
  - recreational E. coli nowcasts;
  - salinization impacts on aquatic life.

### d.8 Execution plan and compute

| Phase | Work | Duration |
|---|---|---|
| P0 | Get USGS and NEON keys/tokens (and EDI if lakes are used). Build the site manifest from `time-series-metadata` / `combined-metadata`; NLDI COMIDs and flow-connected groups; StreamCat attributes; HUC4 folds. **Freeze the manifest** | Week 1 |
| P1 | Bulk pull 2007-10 → 2026-09: ≈ 40–60k one-year requests, ≈ 2–3 days at 1,000/h; ask USGS for a higher limit. Store Parquet partitioned by site/parameter/year with approval, qualifier and last-modified. NEON via `neonutilities`. QC pipeline: qualifier masks, plausible-range tables, flatline/spike screens, splicing | Weeks 1–2 |
| P2 | Evaluation harness (all metrics, bootstrap, DM tests); baselines 1–5; event catalogs with placebo sampling; automated leakage tests | Week 3 |
| P3 | Models A–D; hyperparameter search on validation; zero-shot TSFM runs (Test-B) | Weeks 4–6 |
| P4 | Single scoring of Test-A/B; submissions to VERA and usgsrc4cast, neon4cast recomputation; start prospective Test-C from November 2026 | Week 7+ |

- **Compute.** This host currently exposes 10 RTX 2080 Ti GPUs (driver 470.256.02, 11 GB each); SENTINEL plans to use up to 5. LSTM and GBM baselines are trivial. Chronos-2 fp32 fine-tuning is about 1 GPU-day per configuration (estimate [U]). Data loading is CPU-bound (48 cores).
- **Environments.** Keep separate envs:
  - torch 2.7.1+cu118 for Chronos-2, TimesFM, Toto and LSTMs;
  - torch 2.4.1+cu118 for Moirai / uni2ts;
  - CPU for TiRex-2.

  Pin granite-tsfm ≤ 0.3.4 and neuralforecast ≤ 3.1.2.

---

## (e) Risks and fallbacks

| # | Risk | Likelihood / impact | Mitigation / fallback |
|---|---|---|---|
| 1 | **Deep models don't beat naive baselines** for temperature, DO, SC and pH at 1–24 h, or for temperature and DO at 7–14 d | High / medium | Pre-register skill-vs-baseline reporting. Put the headline claims on tasks where naive baselines are weak: event-driven variables at 2–7 d, unseen-station virtual sensing, events. An honest null at short horizons is publishable |
| 2 | **Too few events** (hypoxia and bloom onsets at river sites) | High / high for track E | Pool stations. Add NERRS, GLOS, lakes and UK EA. Use tiered thresholds. Report exact CIs; with fewer than 30 test events, label results preliminary |
| 3 | **USGS `SuppressData` censoring** at ≥ 2 mg/L (250 DO series) | Medium / high for hypoxia | Inspect thresholds per series. Flag events near thresholds. Cross-check against `field-measurements` and NEON. Treat gaps adjacent to low values as informative missingness |
| 4 | **Sensor artefacts labelled as events** (fouling drift, post-service steps, ice, burial) | High / high | Qualifier masks, PyHydroQC-style QC, service-window exclusion, cross-variable coherence. Hand-audit 100 random events; validate against independent records (fish-kill reports, beach advisories, toxin samples, spill logs) |
| 5 | **Leakage**, the cause of the previous failure: overlapping windows, scalers fit on all data, flow-connected stations split, observed future weather passed off as a forecast, TSFM contamination, re-pulled revised data, station overlap with pretraining | Medium / fatal | d.2 protocol plus automated tests: station-ID disjointness, embargo checks, shuffled-label control, input-truncation audit, placebo dates. External review before unblinding. Prospective Test-C |
| 6 | **Access changes:** WaterServices off 2027-02-22 (degraded from November 2026); USGS v0 partly retired mid-2027; NEON tokens (2026-06-30); EDI sign-in (2026-07-30); Zenodo 403 from this server; rate limits | Certain / medium | Use only `ogcapi/v1` + `dataretrieval.waterdata` ≥ 1.4.0. Get free keys/accounts. Mirror raw pulls locally with metadata. Chunk requests at ≤ 1 y and ≤ 50k rows |
| 7 | **Licensing.** NEON is now CC BY 4.0 (attribution). FCR/BVR, TIME, Moirai and TimesFM-3 are non-commercial. NERRS forbids redistribution. Timestral has a revenue clause | Medium / medium for a public platform | Keep a license register per source and model. The public platform serves derived products and attributions; non-commercial models are used only as research baselines |
| 8 | **Series splicing and units** (new `time_series_id`, RFU vs µg/L, FNU vs NTU, CF units in ERDDAP) | High / medium | Unit is part of target identity; never mix RFU and µg/L; unit-conversion tests |
| 9 | **Provisional data revised after scoring** | Medium / medium | Score on Approved values; store `last_modified`; re-score once as a robustness check |
| 10 | **Hardware:** no native bf16, FlashAttention, Triton or Mamba-2 fp32 on Turing; TiRex-2 needs torch ≥ 2.8 | Certain / low–medium | fp32 or fp16 AMP, SDPA (memory-efficient), eager mode; TiRex-2 on CPU; models ≤ 300M params for full fine-tuning, LoRA above |
| 11 | **External benchmark instability** (neon4cast live scoring stalled since August 2026) | Realized / low | Recompute with the published scoring code; prioritise VERA and usgsrc4cast; keep our own prospective Test-C |
| 12 | **"First" claims overturned** (literature search was quota-limited) | Medium / medium | Full re-search before writing (Scopus/Google Scholar; HESS, WRR, ES&T, *Water Research*, ESSD, NeurIPS/ICML/KDD); phrase as "to our knowledge, as of [date]" |

---

## (f) References

Records were verified via Crossref, the arXiv API, DataCite or the publisher on 2026-10-22, unless marked [U].

### f.1 Hydrology / water-quality deep learning
- Kratzert, F., Klotz, D., Brenner, C., Schulz, K., Herrnegger, M. (2018). Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks. *HESS* 22:6005–6022. doi:10.5194/hess-22-6005-2018
- Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G. (2019). Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets. *HESS* 23:5089–5110. doi:10.5194/hess-23-5089-2019
- Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A.K., Hochreiter, S., Nearing, G.S. (2019). Toward improved predictions in ungauged basins: exploiting the power of machine learning. *WRR* 55:11344–11354. doi:10.1029/2019WR026065
- Gauch, M., Kratzert, F., Klotz, D., Nearing, G., Lin, J., Hochreiter, S. (2021). Rainfall–runoff prediction at multiple timescales with a single Long Short-Term Memory network. *HESS* 25:2045–2062. doi:10.5194/hess-25-2045-2021
- Klotz, D., Kratzert, F., Gauch, M., et al. (2022). Uncertainty estimation with deep learning for rainfall–runoff modeling. *HESS* 26:1673–1693. doi:10.5194/hess-26-1673-2022
- Frame, J.M., Kratzert, F., Klotz, D., et al. (2022). Deep learning rainfall–runoff predictions of extreme events. *HESS* 26:3377–3392. doi:10.5194/hess-26-3377-2022
- Kratzert, F., Gauch, M., Klotz, D., Nearing, G. (2024). HESS Opinions: Never train a Long Short-Term Memory (LSTM) network on a single basin. *HESS* 28:4187–4201. doi:10.5194/hess-28-4187-2024
- Kratzert, F., Gauch, M., Nearing, G., Klotz, D. (2022). NeuralHydrology — a Python library for deep learning research in hydrology. *JOSS* 7:4050. doi:10.21105/joss.04050
- Feng, D., Fang, K., Shen, C. (2020). Enhancing streamflow forecast and extracting insights using long-short term memory networks with data integration at continental scales. *WRR* 56:e2019WR026793. doi:10.1029/2019WR026793
- Nearing, G.S., Klotz, D., Frame, J.M., Gauch, M., et al. (2022). Technical note: Data assimilation and autoregression for using near-real-time streamflow observations in long short-term memory networks. *HESS* 26:5493–5513. doi:10.5194/hess-26-5493-2022
- Nearing, G., Cohen, D., Dube, V., et al. (2024). Global prediction of extreme floods in ungauged watersheds. *Nature* 627:559–563. doi:10.1038/s41586-024-07145-1
- Liu, J., Bian, Y., Lawson, K., Shen, C. (2024). Probing the limit of hydrologic predictability with the Transformer network. *J. Hydrol.* 637:131389. doi:10.1016/j.jhydrol.2024.131389
- Liu, J., Shen, C., O'Donncha, F., et al. (2025). From RNNs to Transformers: benchmarking deep learning architectures for hydrologic prediction. *HESS* 29:6811–6828. doi:10.5194/hess-29-6811-2025
- Zhi, W., Feng, D., Tsai, W.-P., Sterle, G., Harpold, A., Shen, C., Li, L. (2021). From hydrometeorology to river water quality: can a deep learning model predict dissolved oxygen at the continental scale? *ES&T* 55:2357–2368. doi:10.1021/acs.est.0c06783
- Zhi, W., Ouyang, W., Shen, C., Li, L. (2023). Temperature outweighs light and flow as the predominant driver of dissolved oxygen in US rivers. *Nature Water* 1:249–260. doi:10.1038/s44221-023-00038-z
- Zhi, W., Klingler, C., Liu, J., Li, L. (2023). Widespread deoxygenation in warming rivers. *Nat. Clim. Change* 13:1105–1113. doi:10.1038/s41558-023-01793-3
- Zhi, W., Appling, A.P., Golden, H.E., Podgorski, J., Li, L. (2024). Deep learning for water quality. *Nature Water* 2:228–241. doi:10.1038/s44221-024-00202-z
- Rahmani, F., Lawson, K., Ouyang, W., Appling, A., Oliver, S., Shen, C. (2021). Exploring the exceptional performance of a deep learning stream temperature model and the value of streamflow data. *ERL* 16:024025. doi:10.1088/1748-9326/abd501 (data doi:10.5066/P97CGHZH)
- Rahmani, F., Shen, C., Oliver, S., Lawson, K., Appling, A. (2021). Deep learning approaches for improving prediction of daily stream temperature in data-scarce, unmonitored, and dammed basins. *Hydrol. Process.* 35:e14400. doi:10.1002/hyp.14400
- Willard, J.D., Ciulla, F., Weierbach, H., Kumar, V., Varadharajan, C. (2024). Evaluating deep learning approaches for predictions in unmonitored basins with continental-scale stream temperature models. arXiv:2410.19865
- Willard, J., Varadharajan, C., Jia, X., Kumar, V. (2025). Time series predictions in unmonitored sites: a survey of machine learning techniques in water resources. *Environ. Data Sci.* 4:e7. doi:10.1017/eds.2024.14
- Read, J.S., Jia, X., Willard, J., et al. (2019). Process-guided deep learning predictions of lake water temperature. *WRR* 55:9173–9190. doi:10.1029/2019WR024922
- Willard, J.D., Read, J.S., Appling, A.P., Oliver, S.K., Jia, X., Kumar, V. (2021). Predicting water temperature dynamics of unmonitored lakes with meta-transfer learning. *WRR* 57:e2021WR029579. doi:10.1029/2021WR029579
- Willard, J.D., Read, J.S., Topp, S., Hansen, G.J.A., Kumar, V. (2022). Daily surface temperatures for 185,549 lakes in the conterminous United States estimated using deep learning (1980–2020). *L&O Letters* 7:287–301. doi:10.1002/lol2.10249
- Jia, X., Zwart, J., Sadler, J., et al. (2021). Physics-guided recurrent graph model for predicting flow and temperature in river networks. *SDM 2021*:612–620. doi:10.1137/1.9781611976700.69
- Chen, S., Appling, A., Oliver, S., et al. (2021). Heterogeneous stream-reservoir graph networks with data assimilation. *IEEE ICDM 2021*:1024–1029. doi:10.1109/ICDM51629.2021.00117
- Topp, S.N., Barclay, J., Diaz, J., Sun, A.Y., Jia, X., Lu, D., Sadler, J.M., Appling, A.P. (2023). Stream temperature prediction in a shifting environment: explaining the influence of deep learning architecture. *WRR* 59:e2022WR033880
- Zwart, J.A., Oliver, S.K., Watkins, W.D., Sadler, J.M., et al. (2023). Near-term forecasts of stream temperature using deep learning and data assimilation in support of management decisions. *JAWRA* 59:317–337. doi:10.1111/1752-1688.13093
- Sadler, J.M., Appling, A.P., Read, J.S., et al. (2022). Multi-task deep learning of daily streamflow and water temperature. *WRR* 58:e2021WR030138. doi:10.1029/2021WR030138
- Saha, G.K., Rahmani, F., Shen, C., Li, L., Cibin, R. (2023). A deep learning-based novel approach to generate continuous daily stream nitrate concentration for nitrate data-sparse watersheds. *STOTEN* 878:162930
- Saha, G.K., Shen, C., Duncan, J., Cibin, R. (2024). Performance evaluation of deep learning based stream nitrate concentration prediction model to fill stream nitrate data gaps at low-frequency nitrate monitoring basins. *J. Environ. Manage.* 357:120721
- Pandit, A., Golden, H., Christensen, J., Lane, C., Husic, A. (2025). Deep learning prediction and interpretation of riverine nitrate export across the Mississippi River Basin. *WRR* 61:e2024WR039207
- Song, Y., Chaemchuen, P., Rahmani, F., et al. (2024). Deep learning insights into suspended sediment concentrations across the conterminous United States: strengths and limitations. *J. Hydrol.* 639:131573
- Smith, J.C., Koenig, L.E., Sleckman, M.J., Appling, A.P., et al. (2024). Predictive understanding of stream salinization in a developed watershed using machine learning. *ES&T* 58:18822–18833. doi:10.1021/acs.est.4c05004
- Kermorvant, C., Liquet, B., Litt, G., et al. (2021). Reconstructing missing and anomalous data collected from high-frequency in-situ sensors in fresh waters. *IJERPH* 18:12803
- Varadharajan, C., Appling, A.P., Arora, B., et al. (2022). Can machine learning accelerate process understanding and decision-relevant predictions of river water quality? *Hydrol. Process.* 36:e14565
- Shen, C., Appling, A.P., Gentine, P., et al. (2023). Differentiable modelling to unify machine learning and physical models for geosciences. *Nat. Rev. Earth Environ.* 4:552–567
- Sterle, G., Perdrial, J., Kincaid, D.W., et al. (2024). CAMELS-Chem: augmenting CAMELS (Catchment Attributes and Meteorology for Large-sample Studies) with atmospheric and stream water chemistry data. *HESS* 28:611–630. doi:10.5194/hess-28-611-2024

### f.2 Near-term freshwater forecasting and challenges
- Thomas, R.Q., Boettiger, C., Carey, C.C., et al. (2023). The NEON Ecological Forecasting Challenge. *Front. Ecol. Environ.* 21:112–113. doi:10.1002/fee.2616
- Thomas, R.Q., McClure, R.P., Moore, T.N., et al. (2023). Near-term forecasts of NEON lakes reveal gradients of environmental predictability across the US. *Front. Ecol. Environ.* 21:220–226. doi:10.1002/fee.2623
- Olsson, F., Carey, C.C., Boettiger, C., Harrison, G., et al. (2025). What can we learn from 100,000 freshwater forecasts? A synthesis from the NEON Ecological Forecasting Challenge. *Ecol. Appl.* 35:e70004. doi:10.1002/eap.70004 (data doi:10.5281/zenodo.11087208)
- Olsson, F., Moore, T.N., Carey, C.C., Breef-Pilz, A., Thomas, R.Q. (2024). A multi-model ensemble of baseline and process-based models improves the predictive skill of near-term lake forecasts. *WRR* 60:e2023WR035901. doi:10.1029/2023WR035901
- Thomas, R.Q., Figueiredo, R.J., Daneshmand, V., et al. (2020). A near-term iterative forecasting system successfully predicts reservoir hydrodynamics and partitions uncertainty in real time. *WRR* 56:e2019WR026138. doi:10.1029/2019WR026138
- Lofton, M.E., Howard, D.W., Thomas, R.Q., Carey, C.C. (2023). Progress and opportunities in advancing near-term forecasting of freshwater quality. *Glob. Change Biol.* 29:1691–1714. doi:10.1111/gcb.16590
- Lofton, M.E., Brentrup, J.A., Beck, W.S., et al. (2022). Using near-term forecasts and uncertainty partitioning to inform prediction of oligotrophic lake cyanobacterial density. *Ecol. Appl.* 32:e2590. doi:10.1002/eap.2590
- Woelmer, W.M., Thomas, R.Q., Lofton, M.E., et al. (2022). Near-term phytoplankton forecasts reveal the effects of model time step and forecast horizon on predictability. *Ecol. Appl.* 32:e2642. doi:10.1002/eap.2642
- Woelmer, W.M., Thomas, R.Q., Olsson, F., Steele, B.G., Weathers, K.C., Carey, C.C. (2024). Process-based forecasts of lake water temperature and dissolved oxygen outperform null models, with variability over time and depth. *Ecol. Inform.* 83:102825 (results beyond title [U])
- Howard, D.W., Lofton, M.E., Thomas, R.Q., et al. (2026). Near-term forecasts of dissolved organic matter exhibit consistent patterns of accuracy across multiple freshwater reservoirs. *JGR-Biogeosci.* 131:e2025JG009064. doi:10.1029/2025JG009064
- Fournier, C., Fernandez-Fernandez, R., Cirés, S., et al. (2024). LSTM networks provide efficient cyanobacterial blooms forecasting even with incomplete spatio-temporal data. *Water Res.* 267:122553. doi:10.1016/j.watres.2024.122553
- Kemper, J.T., Underwood, K.L., Hamshaw, S.D., et al. (2025). Leveraging high-frequency sensor data and U.S. National Water Model output to forecast turbidity in a drinking water supply basin. *JAWRA* 61:e70011. doi:10.1111/1752-1688.70011
- Kim, Y.W., Kim, T.H., Shin, J., et al. (2021). Forecasting abrupt depletion of dissolved oxygen in urban streams using discontinuously measured hourly time-series data. *WRR* 57:e2020WR029188. doi:10.1029/2020WR029188
- Thomas, M.K., Fontana, S., Reyes, M., Kehoe, M., Pomati, F. (2018). The predictability of a lake phytoplankton community, over time-scales of hours to years. *Ecol. Lett.* 21:619–628. doi:10.1111/ele.12927
- Lewis, A.S.L., et al. (2022). Increased adoption of best practices in ecological forecasting enables comparisons of forecastability. *Ecol. Appl.* 32:e2500. doi:10.1002/eap.2500
- Carey, C.C., Woelmer, W.M., Lofton, M.E., et al. (2022). Advancing lake and reservoir water quality management with near-term, iterative ecological forecasting. *Inland Waters* 12:107–120. doi:10.1080/20442041.2020.1816421
- Rousso, B.Z., Bertone, E., Stewart, R., Hamilton, D.P. (2020). A systematic literature review of forecasting and predictive models for cyanobacteria blooms in freshwater lakes. *Water Res.* 182:115959. doi:10.1016/j.watres.2020.115959

### f.3 Benchmarks and datasets
- McAfee, B.J., Pradhan, A., Neog, A., Fatemi, S., et al. (2025). LakeBeD-US: a benchmark dataset for lake water quality time series and vertical profiles. *ESSD* 17:3141–3165. doi:10.5194/essd-17-3141-2025
- Neog, A., Fatemi, S., Sawhney, M., Mehrab, K.S., Pradhan, A., McAfee, B.J., et al. (2026). LakeFM: Toward a foundation model for aquatic ecosystems using irregular multivariate multi-depth time series data. KDD 2026. arXiv:2606.11268
- Qiao, Z., Pan, S., Wang, A., Zhukova, V., Liu, Y., Jiang, X., et al. (2026). It's TIME: Towards the next generation of time series forecasting benchmarks. ICML 2026. arXiv:2602.12147
- Klingler, C., Schulz, K., Herrnegger, M. (2021). LamaH-CE: LArge-SaMple DAta for Hydrology and Environmental Sciences for Central Europe. *ESSD* 13:4529–4565. doi:10.5194/essd-13-4529-2021
- Helgason, H.B., Nijssen, B. (2024). LamaH-Ice. *ESSD* 16:2741–2771. doi:10.5194/essd-16-2741-2024
- Kratzert, F., Nearing, G., Addor, N., et al. (2023). Caravan – A global community dataset for large-sample hydrology. *Sci. Data* 10:61. doi:10.1038/s41597-023-01975-w
- Shalev, G., Kratzert, F. (2024). Caravan MultiMet: Extending Caravan with multiple weather nowcasts and forecasts. arXiv:2411.09459
- Demir, I., Xiang, Z., Demiray, B., Sit, M. (2022). WaterBench-Iowa: a large-scale benchmark dataset for data-driven streamflow forecasting. *ESSD* 14:5605–5616. doi:10.5194/essd-14-5605-2022
- Virro, H., Amatulli, G., Kmoch, A., Shen, L., Uuemaa, E. (2021). GRQA: Global River Water Quality Archive. *ESSD* 13:5483–5507. doi:10.5194/essd-13-5483-2021
- Hartmann, J., Lauerwald, R., Moosdorf, N. (2014). A brief overview of the GLObal RIver CHemistry Database, GLORICH. *Procedia Earth Planet. Sci.* 10:23–27. doi:10.1016/j.proeps.2014.08.005
- Godahewa, R., Bergmeir, C., Webb, G.I., Hyndman, R.J., Montero-Manso, P. (2021). Monash Time Series Forecasting Archive. NeurIPS Datasets & Benchmarks. arXiv:2105.06643
- Vlah, M.J., Rhea, S., Bernhardt, E.S., Slaughter, W., Gubbins, N., DelVecchia, A.G., et al. (2023). MacroSheds: A synthesis of long-term biogeochemical, hydroclimatic, and geospatial data from small watershed ecosystem studies. *L&O Letters* 8:419–452. doi:10.1002/lol2.10325 (daily data only; not high-frequency)
- Jane, S.F., Hansen, G.J.A., Kraemer, B.M., Leavitt, P.R., Mincer, J.L., North, R.L., et al. (2021). Widespread deoxygenation of temperate lakes. *Nature* 594:66–70. doi:10.1038/s41586-021-03550-y (data: EDI edi.698)
- Moore, T.N., Mesman, J.P., Ladwig, R., Feldbauer, J., Olsson, F., Pilla, R.M., et al. (2021). LakeEnsemblR: An R package that facilitates ensemble modelling of lakes. *EMS* 143:105101. doi:10.1016/j.envsoft.2021.105101
- Golub, M., Thiery, W., Marcé, R., Pierson, D., Vanderkelen, I., Mercado-Bettin, D., et al. (2022). A framework for ensemble modelling of climate change impacts on lakes worldwide: the ISIMIP Lake Sector. *GMD* 15:4597–4623. doi:10.5194/gmd-15-4597-2022

### f.4 General time-series architectures and SSMs (arXiv records verified)
- Nie, Y., Nguyen, N.H., Sinthong, P., Kalagnanam, J. A time series is worth 64 words: long-term forecasting with Transformers. ICLR 2023. arXiv:2211.14730
- Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., et al. iTransformer: Inverted Transformers are effective for time series forecasting. ICLR 2024. arXiv:2310.06625
- Wang, S., Wu, H., Shi, X., et al. TimeMixer: Decomposable multiscale mixing for time series forecasting. ICLR 2024. arXiv:2405.14616
- Wang, S., Li, J., Shi, X., et al. TimeMixer++: A general time series pattern machine for universal predictive analysis. ICLR 2025. arXiv:2410.16032
- Chen, S.-A., Li, C.-L., Yoder, N., Arik, S.O., Pfister, T. TSMixer: An all-MLP architecture for time series forecasting. TMLR 2023. arXiv:2303.06053
- Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., Long, M. TimesNet: Temporal 2D-variation modeling for general time series analysis. ICLR 2023. arXiv:2210.02186
- Challu, C., Olivares, K.G., Oreshkin, B.N., Garza, F., Mergenthaler-Canseco, M., Dubrawski, A. N-HiTS: Neural hierarchical interpolation for time series forecasting. AAAI 2023. arXiv:2201.12886
- Oreshkin, B.N., Carpov, D., Chapados, N., Bengio, Y. N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. ICLR 2020. arXiv:1905.10437
- Lim, B., Arık, S.Ö., Loeff, N., Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *Int. J. Forecast.* 37:1748–1764. doi:10.1016/j.ijforecast.2021.03.012
- Zeng, A., Chen, M., Zhang, L., Xu, Q. Are Transformers effective for time series forecasting? AAAI 2023. arXiv:2205.13504
- Gu, A., Goel, K., Ré, C. Efficiently modeling long sequences with structured state spaces. ICLR 2022. arXiv:2111.00396
- Zhang, M., Saab, K.K., Poli, M., Dao, T., Goel, K., Ré, C. Effectively modeling time series with simple discrete state spaces. ICLR 2023. arXiv:2303.09489
- Wang, Z., Kong, F., Feng, S., et al. Is Mamba effective for time series forecasting? arXiv:2403.11144 (venue [U])
- Ahamed, M.A., Cheng, Q. TimeMachine: A time series is worth 4 Mambas for long-term forecasting. ECAI 2024. arXiv:2403.09898
- Liang, A., Jiang, X., Sun, Y., Shi, X., Li, K. Bi-Mamba+: Bidirectional Mamba for time series forecasting. arXiv:2404.15772
- Behrouz, A., Santacatterina, M., Zabih, R. Chimera: Effectively modeling multivariate time series with 2-dimensional state space models. arXiv:2406.04320 (venue [U])
- Bhethanabhotla, S.K., Swelam, O., Siems, J., Salinas, D., Hutter, F. Mamba4Cast: Efficient zero-shot time series forecasting with state space models. arXiv:2410.09385
- Alharthi, M., Mahmood, A. xLSTMTime: Long-term time series forecasting with xLSTM. arXiv:2407.10240
- Kraus, M., Divo, F., Dhami, D.S., Kersting, K. xLSTM-Mixer: Multivariate time series forecasting by mixing via scalar memories. NeurIPS 2025. arXiv:2410.16928
- Qiu, X., Hu, J., Zhou, L., et al. TFB: Towards comprehensive and fair benchmarking of time series forecasting methods. PVLDB 2024. arXiv:2403.20150
- Brigato, L., Morand, R., Strømmen, K., Panagiotou, M., Schmidt, M., Mougiakakou, S. There are no champions in supervised long-term time series forecasting. TMLR. arXiv:2502.14045

### f.5 Time-series foundation models and TS benchmarks (arXiv records verified)
- Ansari, A.F., Stella, L., Turkmen, C., et al. Chronos: Learning the language of time series. TMLR 2024. arXiv:2403.07815
- Ansari, A.F., Shchur, O., Küken, J., Auer, A., et al. (2025). Chronos-2: From univariate to universal forecasting. arXiv:2510.15821 (weights hf.co/amazon/chronos-2)
- Das, A., Kong, W., Sen, R., Zhou, Y. A decoder-only foundation model for time-series forecasting. ICML 2024. arXiv:2310.10688 (TimesFM; 2.5 and 3.0 weights on hf.co/google)
- Woo, G., Liu, C., Kumar, A., Xiong, C., Savarese, S., Sahoo, D. Unified training of universal time series forecasting Transformers. ICML 2024. arXiv:2402.02592 (Moirai)
- Liu, X., Liu, J., Woo, G., et al. Moirai-MoE: Empowering time series foundation models with sparse mixture of experts. arXiv:2410.10469
- Liu, C., Aksu, T., Liu, J., et al. (2025). Moirai 2.0: When less is more for time series forecasting. arXiv:2511.11698
- Auer, A., Podest, P., Klotz, D., Böck, S., Klambauer, G., Hochreiter, S. TiRex: Zero-shot forecasting across long and short horizons with enhanced in-context learning. NeurIPS 2025. arXiv:2505.23719
- Podest, P., Pichler, M., Bürger, E., et al. (2026). TiRex-2: Generalizing TiRex to multivariate data and streaming. arXiv:2607.01204
- Cohen, B., Khwaja, E., Doubli, Y., et al. (2025). This time is different: An observability perspective on time series foundation models. arXiv:2505.14766 (Toto)
- Khwaja, E., Lettieri, C., Woo, G., et al. (2026). Toto 2.0: Time series forecasting enters the scaling era. arXiv:2605.20119
- Liu, Y., Qin, G., Shi, Z., et al. Sundial: A family of highly capable time series foundation models. ICML 2025. arXiv:2502.00816
- Shi, X., Wang, S., Nie, Y., et al. Time-MoE: Billion-scale time series foundation models with mixture of experts. ICLR 2025. arXiv:2409.16040
- Liu, Y., Zhang, H., Li, C., Huang, X., Wang, J., Long, M. Timer: Generative pre-trained Transformers are large time series models. ICML 2024. arXiv:2402.02368
- Liu, Y., Qin, G., Huang, X., Wang, J., Long, M. Timer-XL: Long-context Transformers for unified time series forecasting. ICLR 2025. arXiv:2410.04803
- Goswami, M., Szafer, K., Choudhry, A., Cai, Y., Li, S., Dubrawski, A. MOMENT: A family of open time-series foundation models. ICML 2024. arXiv:2402.03885
- Rasul, K., Ashok, A., Williams, A.R., et al. Lag-Llama: Towards foundation models for probabilistic time series forecasting. arXiv:2310.08278
- Ekambaram, V., Jati, A., Dayama, P., et al. Tiny Time Mixers (TTMs): Fast pre-trained models for enhanced zero/few-shot forecasting of multivariate time series. NeurIPS 2024. arXiv:2401.03955
- Graf, L., Ortner, T., Woźniak, S., Pantazi, A. FlowState: Sampling-rate-equivariant time-series forecasting. ICML 2026 (PMLR 306). arXiv:2508.05287
- Hoo, S.B., Müller, S., Salinas, D., Hutter, F. From tables to time: Extending TabPFN-v2 to time series forecasting. arXiv:2501.02945 (TabPFN-TS)
- Mistral AI (2026). Timestral (model release; no paper). hf.co/mistralai/Timestral-209M; mistral.ai/news/timestral
- The Forecasting Company (2026). t0-gamma (model release). hf.co/theforecastingcompany/t0-gamma
- Aksu, T., Woo, G., Liu, J., et al. GIFT-Eval: A benchmark for general time series forecasting model evaluation. arXiv:2410.10393
- Shchur, O., Ansari, A.F., Turkmen, C., Stella, L., Erickson, N., Guerron, P., et al. fev-bench: A realistic benchmark for time series forecasting. arXiv:2509.26468

### f.6 TSFM evidence in hydrology / WQ
- Sun, A.Y., Sun, A.A. (2026). Zero-shot forecasting of streamflow using time series foundation models: are we there yet? *Machine Learning: Earth* 2:010501. doi:10.1088/3049-4753/ae4982
- He, J., Fox, J., Jafari, A., Chen, Y.-J., Fox, G. (2025). Leveraging exogenous signals for hydrology time series forecasting. arXiv:2511.11849
- Rangaraj, R., Shi, J., Shirali, A., Paudel, R., Wu, Y., Narasimhan, G. (2025). How effective are large time series models in hydrology? A study on water level forecasting in Everglades. arXiv:2505.01415
- Zheng, K., Cui, G., Xie, Y., Liu, Y., Du, X. (2026). Time series foundation model chronos enhances nitrogen forecasting under data scarcity. *Water Research X* 30:100469. doi:10.1016/j.wroa.2025.100469 (numbers [U])
- Haq, I.U., Lee, B.S., Perdrial, J.N., Baude, D. (2025). HydroGEM: A self supervised zero shot hybrid TCN transformer foundation model for continental scale streamflow quality control. arXiv:2512.14106
- [U] Zhu & Liu, "hydro-foundation-model" (GitHub preprint); Chronos-2 on CAMELS-DE (ResearchGate, venue unknown)

### f.7 Anomaly / event detection and early warning
- Leigh, C., Alsibai, O., Hyndman, R.J., et al. (2019). A framework for automated anomaly detection in high frequency water-quality data from in situ sensors. *STOTEN* 664:885–898. doi:10.1016/j.scitotenv.2019.02.085
- Talagala, P.D., Hyndman, R.J., Leigh, C., Mengersen, K., Smith-Miles, K. (2019). A feature-based procedure for detecting technical outliers in water-quality data from in situ sensors. *WRR* 55:8547–8568. doi:10.1029/2019WR024906
- Rodriguez-Perez, J., Leigh, C., Liquet, B., et al. (2020). Detecting technical anomalies in high-frequency water-quality data using artificial neural networks. *ES&T* 54:13719–13730. doi:10.1021/acs.est.0c04069
- Russo, S., Lürig, M., Hao, W., Matthews, B., Villez, K. (2020). Active learning for anomaly detection in environmental data. *EMS* 134:104869. doi:10.1016/j.envsoft.2020.104869
- Russo, S., Besmer, M.D., Blumensaat, F., et al. (2021). The value of human data annotation for machine learning based anomaly detection in environmental systems. *Water Res.* 206:117695. doi:10.1016/j.watres.2021.117695
- Jones, A.S., Jones, T.L., Horsburgh, J.S. (2022). Toward automating post processing of aquatic sensor data. *EMS* 151:105364. doi:10.1016/j.envsoft.2022.105364
- Almuhtaram, H., Zamyadi, A., Hofmann, R. (2021). Machine learning for anomaly detection in cyanobacterial fluorescence signals. *Water Res.* 197:117073. doi:10.1016/j.watres.2021.117073
- Hart, D., McKenna, S.A., Klise, K., Cruz, V., Wilson, M. (2007). CANARY: A water quality event detection algorithm development tool. World Environmental & Water Resources Congress 2007 (ASCE). doi:10.1061/40927(243)517
- Burkhardt, J.B., Sahoo, D., Hammond, B., et al. (2022). Near real-time event detection for watershed monitoring with CANARY. *Environ. Sci.: Adv.* 1:170–181. doi:10.1039/d2va00014h
- Cronin, L., Taylor, C.M., Briciu Burghina, C., et al. (2025). An open-source workflow for identifying hydrodynamic water quality events in rivers … using R and US EPA CANARY. *MethodsX* 15:103538. doi:10.1016/j.mex.2025.103538
- Mounce, S.R., Machell, J., Boxall, J. (2012). Water quality event detection and customer complaint clustering analysis in distribution systems. *Water Supply* 12:580–587. doi:10.2166/ws.2012.030
- Wu, R., Keogh, E. (2021). Current time series anomaly detection benchmarks are flawed and are creating the illusion of progress. *IEEE TKDE*. doi:10.1109/TKDE.2021.3112126
- Kim, S., Choi, K., Choi, H.-S., et al. (2022). Towards a rigorous evaluation of time-series anomaly detection. *AAAI* 36:7194–7201. doi:10.1609/aaai.v36i7.20680
- Paparrizos, J., Boniol, P., Palpanas, T., et al. (2022). Volume under the surface: a new accuracy evaluation measure for time-series anomaly detection. *PVLDB* 15:2774–2787. doi:10.14778/3551793.3551830
- Huet, A., Navarro, J.M., Rossi, D. (2022). Local evaluation of time series anomaly detection algorithms. *KDD 2022*:635–645. doi:10.1145/3534678.3539339
- Schmidl, S., Wenig, P., Papenbrock, T. (2022). Anomaly detection in time series: a comprehensive evaluation. *PVLDB* 15:1779–1797. doi:10.14778/3538598.3538602
- Liu, Q., Paparrizos, J. (2024). The elephant in the room: towards a reliable time-series anomaly detection benchmark. NeurIPS 2024 Datasets & Benchmarks
- Blaszczak, J.R., Koenig, L.E., Mejia, F.H., et al. (2023). Extent, patterns, and drivers of hypoxia in the world's streams and rivers. *L&O Letters* 8:453–463. doi:10.1002/lol2.10297
- Vaquer-Sunyer, R., Duarte, C.M. (2008). Thresholds of hypoxia for marine biodiversity. *PNAS* 105:15452–15457. doi:10.1073/pnas.0803833105
- Moore, J., Fanelli, R.M., Sekellick, A.J. (2020). High-frequency data reveal deicing salts drive elevated specific conductance and chloride along with pulsed toxicity to aquatic life in northern Virginia, USA. *ES&T* 54:778–789. doi:10.1021/acs.est.9b04316 (data doi:10.5066/P9YN2QST)
- Granato, G.E., Smith, K.P. (1999). Estimating concentrations of road-salt constituents in highway-runoff from measurements of specific conductance. USGS WRIR 99-4077. doi:10.3133/wri994077
- Aulenbach, B.T., McKee, A.M. (2020). Monitoring and real-time modeling of *Escherichia coli* bacteria for the Chattahoochee River…, 2000–2019. USGS OFR 2020-1048. doi:10.3133/ofr20201048
- Francy, D.S., Brady, A.M.G., Carvin, R.B., et al. (2013). Developing and implementing the use of predictive models for estimating water quality at Great Lakes beaches. USGS SIR 2013-5166. doi:10.3133/sir20135166
- Sorensen, J.P.R., Baker, A., Cumberland, S.A., et al. (2018). Real-time detection of faecally contaminated drinking water with tryptophan-like fluorescence: defining threshold values. *STOTEN* 622–623:1250–1257. doi:10.1016/j.scitotenv.2017.11.162
- Sorensen, J.P.R., Lapworth, D.J., Marchant, B.P., et al. (2015). In-situ tryptophan-like fluorescence: a real-time indicator of faecal contamination in drinking water supplies. *Water Res.* 81:38–46. doi:10.1016/j.watres.2015.05.035

### f.8 Criteria, thresholds and sensor methods
- US EPA (1986). Quality Criteria for Water (Gold Book). https://www.epa.gov/sites/default/files/2018-10/documents/quality-criteria-water-1986.pdf
- US EPA. National recommended water quality criteria, aquatic life table (chloride acute 860 / chronic 230 mg/L). https://www.epa.gov/wqc/national-recommended-water-quality-criteria-aquatic-life-criteria-table
- US EPA (2013; 822-R-18-002). Aquatic life criteria for ammonia, freshwater. https://www.epa.gov/wqc/aquatic-life-criteria-ammonia
- US EPA (2019). Recommended human health recreational ambient water quality criteria for microcystins and cylindrospermopsin (EPA 822-R-19-001). https://www.epa.gov/sites/default/files/2019-05/documents/hh-rec-criteria-habs-document-2019.pdf
- US EPA. National Primary Drinking Water Regulations (nitrate-N MCL 10 mg/L; turbidity treatment technique). https://www.epa.gov/ground-water-and-drinking-water/national-primary-drinking-water-regulations
- WHO (2021). Guidelines on recreational water quality, Vol. 1: coastal and fresh waters. https://www.who.int/publications/i/item/9789240031302
- WHO (2020). Cyanobacterial toxins: microcystins (WHO/HEP/ECH/WSH/2020.6)
- Washington State, WAC 173-201A-200 (freshwater criteria: temperature 7-DADMax tiers, turbidity). https://app.leg.wa.gov/wac/default.aspx?cite=173-201A-200
- Rasmussen, P.P., Gray, J.R., Glysson, G.D., Ziegler, A.C. (2009). Guidelines and procedures for computing time-series suspended-sediment concentrations and loads from in-stream turbidity-sensor and streamflow data. USGS TM 3-C4. doi:10.3133/tm3C4
- Wagner, R.J., Boulger, R.W., Oblinger, C.J., Smith, B.A. (2006). Guidelines and standard procedures for continuous water-quality monitors. USGS TM 1-D3. doi:10.3133/tm1D3
- Downing, B.D., Pellerin, B.A., Bergamaschi, B.A., et al. (2012). Seeing the light: the effects of particles, dissolved materials, and temperature on in situ measurements of DOM fluorescence in rivers and streams. *L&O Methods* 10:767–775. doi:10.4319/lom.2012.10.767
- Watras, C.J., Hanson, P.C., Stacy, T.L., et al. (2011). A temperature compensation method for CDOM fluorescence sensors in freshwater. *L&O Methods* 9:296–301. doi:10.4319/lom.2011.9.296
- Bertone, E., Burford, M.A., Hamilton, D.P. (2018). Fluorescence probes for real-time remote cyanobacteria monitoring: a review of challenges and opportunities. *Water Res.* 141:152–162. doi:10.1016/j.watres.2018.05.001
- Macário, I.P.E., Castro, B.B., Nunes, M.I.S., et al. (2015). New insights towards the establishment of phycocyanin concentration thresholds considering species-specific variability of bloom-forming cyanobacteria. *Hydrobiologia* 757:155–165. doi:10.1007/s10750-015-2248-7

### f.9 Stream metabolism
- Appling, A.P., Read, J.S., Winslow, L.A., et al. (2018). The metabolic regimes of 356 rivers in the United States. *Sci. Data* 5:180292. doi:10.1038/sdata.2018.292 (data release doi:10.5066/F70864KX)
- Appling, A.P., Hall, R.O., Yackulic, C.B., Arroita, M. (2018). Overcoming equifinality: leveraging long time series for stream metabolism estimation. *JGR-Biogeosci.* 123:624–645. doi:10.1002/2017JG004140
- Bernhardt, E.S., Savoy, P., Vlah, M.J., et al. (2022). Light and flow regimes regulate the metabolism of rivers. *PNAS* 119:e2121976119. doi:10.1073/pnas.2121976119
- Bernhardt, E.S., Heffernan, J.B., Grimm, N.B., et al. (2018). The metabolic regimes of flowing waters. *L&O* 63:S99–S118. doi:10.1002/lno.10726
- Savoy, P., Appling, A.P., Heffernan, J.B., et al. (2019). Metabolic rhythms in flowing waters: an approach for classifying river productivity regimes. *L&O* 64:1835–1851. doi:10.1002/lno.11154
- Koenig, L.E., Helton, A.M., Savoy, P., et al. (2019). Emergent productivity regimes of river networks. *L&O Letters* 4:173–181. doi:10.1002/lol2.10115
- Bertuzzo, E., Hotchkiss, E.R., Argerich, A., et al. (2022). Respiration regimes in rivers: partitioning source-specific respiration from metabolism time series. *L&O* 67:2374–2388. doi:10.1002/lno.12207
- Battin, T.J., Lauerwald, R., Bernhardt, E.S., et al. (2023). River ecosystem metabolism and carbon biogeochemistry in a changing world. *Nature* 613:449–459. doi:10.1038/s41586-022-05500-8
- Ulseth, A.J., Hall, R.O., Boix Canadell, M., et al. (2019). Distinct air–water gas exchange regimes in low- and high-energy streams. *Nat. Geosci.* 12:259–263. doi:10.1038/s41561-019-0324-8
- Hall, R.O., Ulseth, A.J. (2020). Gas exchange in streams and rivers. *WIREs Water* 7:e1391. doi:10.1002/wat2.1391
- Appling, A.P., Oliver, S.K., Read, J.S., et al. (2022). Machine learning for understanding inland water quantity, quality, and ecology. In *Encyclopedia of Inland Waters* (2nd ed.):585–606. doi:10.1016/B978-0-12-819166-8.00121-3

### f.10 Evaluation methodology
- Nash, J.E., Sutcliffe, J.V. (1970). River flow forecasting through conceptual models part I — A discussion of principles. *J. Hydrol.* 10:282–290. doi:10.1016/0022-1694(70)90255-6
- Seibert, J. (2001). On the need for benchmarks in hydrological modelling. *Hydrol. Process.* 15:1063–1064. doi:10.1002/hyp.446
- Schaefli, B., Gupta, H.V. (2007). Do Nash values have value? *Hydrol. Process.* 21:2075–2080. doi:10.1002/hyp.6825
- Gupta, H.V., Kling, H., Yilmaz, K.K., Martinez, G.F. (2009). Decomposition of the mean squared error and NSE performance criteria: implications for improving hydrological modelling. *J. Hydrol.* 377:80–91. doi:10.1016/j.jhydrol.2009.08.003
- Knoben, W.J.M., Freer, J.E., Woods, R.A. (2019). Technical note: Inherent benchmark or not? Comparing Nash–Sutcliffe and Kling–Gupta efficiency scores. *HESS* 23:4323–4331. doi:10.5194/hess-23-4323-2019
- Diebold, F.X., Mariano, R.S. (1995). Comparing predictive accuracy. *JBES* 13:253–263. doi:10.1080/07350015.1995.10524599
- Brier, G.W. (1950). Verification of forecasts expressed in terms of probability. *Mon. Weather Rev.* 78:1–3
- Jolliffe, I.T., Stephenson, D.B. (eds.) (2012). *Forecast Verification: A Practitioner's Guide*, 2nd ed. Wiley. doi:10.1002/9781119960003
- Gneiting, T., Raftery, A.E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA* 102:359–378. doi:10.1198/016214506000001437
- Gneiting, T., Balabdaoui, F., Raftery, A.E. (2007). Probabilistic forecasts, calibration and sharpness. *JRSS-B* 69:243–268. doi:10.1111/j.1467-9868.2007.00587.x
- Gneiting, T., Ranjan, R. (2011). Comparing density forecasts using threshold- and quantile-weighted scoring rules. *JBES* 29:411–422. doi:10.1198/jbes.2010.08110
- Allen, S., Ginsbourger, D., Ziegel, J. (2023). Evaluating forecasts for high-impact events using transformed kernel scores. *SIAM/ASA JUQ* 11:906–940. doi:10.1137/22M1532184
- Lerch, S., Thorarinsdottir, T.L., Ravazzolo, F., Gneiting, T. (2017). Forecaster's dilemma: extreme events and forecast evaluation. *Stat. Sci.* 32:106–127. doi:10.1214/16-STS588
- Ferro, C.A.T., Stephenson, D.B. (2011). Extremal dependence indices: improved verification measures for deterministic forecasts of rare binary events. *Weather Forecast.* 26:699–713. doi:10.1175/WAF-D-10-05030.1
- Pappenberger, F., Ramos, M.H., Cloke, H.L., et al. (2015). How do I know if my forecasts are better? Using benchmarks in hydrological ensemble prediction. *J. Hydrol.* 522:697–713. doi:10.1016/j.jhydrol.2015.01.024
- Gibbs, I., Candès, E. (2021). Adaptive conformal inference under distribution shift. NeurIPS 34. arXiv:2106.00170
- Angelopoulos, A.N., Candès, E.J., Tibshirani, R.J. (2023). Conformal PID control for time series prediction. arXiv:2307.16895 (venue not re-verified)

### f.11 Data-access documentation (all fetched 2026-10-22)
- USGS Water Data APIs: https://api.waterdata.usgs.gov/ ; OGC v1: https://api.waterdata.usgs.gov/ogcapi/v1/ ; index: https://api.waterdata.usgs.gov/llms.txt
- USGS keys: https://api.waterdata.usgs.gov/docs/ogcapi/keys/ ; signup: https://api.waterdata.usgs.gov/signup/
- USGS migration (iv → continuous, dv → daily, 3-year limit): https://api.waterdata.usgs.gov/docs/ogcapi/migration/
- USGS efficient queries (50k/page): https://api.waterdata.usgs.gov/docs/ogcapi/efficiency/ ; versioning: https://api.waterdata.usgs.gov/docs/ogcapi/versioning/ ; CQL2: https://api.waterdata.usgs.gov/docs/ogcapi/complex-queries/
- USGS bulletin (2026-09-17), NWISWeb decommissioning webinar (WaterServices off 2027-02-22): https://content.govdelivery.com/accounts/USDOIGS/bulletins/42b03d7
- WDFN blog: https://waterdata.usgs.gov/blog/api-updates-2026/ (2026-06-18, updated 2026-10-17); https://waterdata.usgs.gov/blog/api-v1-release/ (2026-09-04); https://waterdata.usgs.gov/blog/nwisweb-no-params (2026-06-30)
- api.data.gov developer manual (keyed default 1,000 req/h; DEMO_KEY 30/h, 50/day): https://api.data.gov/docs/developer-manual/
- dataretrieval-python 1.4.0: https://pypi.org/project/dataretrieval/ ; https://github.com/DOI-USGS/dataretrieval-python
- Water Quality Portal (WQX 3.0 beta): https://www.waterqualitydata.us/beta/
- NEON API: https://data.neonscience.org/api/v0/products/DP1.20288.001 ; NEON blog, logins and CC BY (2026-06-30): https://www.neonscience.org/impact/observatory-blog/logins-required-and-data-licensing-updated ; announcement (2026-03-30): https://www.neonscience.org/impact/observatory-blog/upcoming-required-logins-and-data-licensing-updates ; token tutorial: https://www.neonscience.org/resources/learning-hub/tutorials/neon-api-tokens-tutorial ; neonutilities 2.0.2: https://pypi.org/project/neonutilities/
- EFI challenges: https://projects.ecoforecast.org/neon4cast-ci/targets.html ; https://projects.ecoforecast.org/usgsrc4cast-ci/targets.html ; https://www.ltreb-reservoirs.org/vera4cast/targets.html
- EDI sign-in notice: https://edirepository.org/news/news-20260727.00 ; DataONE mirror: https://gmn.lternet.edu/mn/v2/object/
- GLOS ERDDAP: https://seagull-erddap.glos.org/erddap/ ; IOOS ERDDAP: https://erddap.sensors.ioos.us/erddap/ ; UK EA Hydrology: https://environment.data.gov.uk/hydrology/doc/reference ; NERRS CDMO: https://cdmo.baruch.sc.edu/ ; CDEC: https://cdec.water.ca.gov/ ; StreamPULSE: https://data.streampulse.org/
- Open-Meteo: https://open-meteo.com/en/docs/historical-forecast-api ; HRRR: https://noaa-hrrr-bdp-pds.s3.amazonaws.com/ ; GEFS reforecast: https://noaa-gefs-retrospective.s3.amazonaws.com/
- NLDI: https://api.water.usgs.gov/nldi/ ; StreamCat API: https://api.epa.gov/StreamCat/streams/metrics
- PyTorch cu118 wheels: https://download.pytorch.org/whl/cu118/torch/

---

## Appendix A: verification log (selected requests, 2026-10-22)

| Request | Result |
|---|---|
| `GET https://api.waterdata.usgs.gov/ogcapi/v1/collections?f=json` | 200; 39 collections (continuous, daily, latest-*, field-measurements, time-series-metadata, combined-metadata, peaks, …) |
| `GET …/ogcapi/v1/collections/continuous/items?f=csv&monitoring_location_id=USGS-05586300&parameter_code=00300,00010,00095,63680,99133,32316,32319,00400&datetime=2024-06-01T00:00:00Z/2024-09-01T00:00:00Z&limit=50000&properties=time,value,parameter_code,approval_status,qualifier,time_series_id` | 200; 5.4 MB; 50,000 rows (page cap reached; no `next` link in CSV) |
| `GET …/collections/continuous/items?...USGS-04193500&parameter_code=00060&datetime=1990-01-01/1990-01-03` | 200; 0 features (metadata says POR from 1987); 2007-10-01 → 25 records |
| `GET …/collections/time-series-metadata/items?parameter_code=00300&computation_identifier=Instantaneous&limit=50000&properties=…` | 200; 1,864 series / 1,551 sites with data |
| `POST …/collections/daily/items` with CQL2 `value < 2` | 500 "A database error occurred" |
| 22nd per-site `daily` request in a burst (after ~110 total) | **429 OVER_RATE_LIMIT**; `x-ratelimit-limit: 1000`, `x-ratelimit-remaining: 0`, `retry-after: 1315` |
| `GET https://waterservices.usgs.gov/nwis/iv/?sites=01646500&parameterCd=00010&period=P1D&format=json` | 200 (legacy still up; decommission 2027-02-22) |
| `GET https://data.neonscience.org/api/v0/products/DP1.20288.001` | 200; 34 sites; 3,297 site-months; `x-ratelimit-limit: 200` |
| `GET https://data.neonscience.org/api/v0/data/DP1.20288.001/PRLA/2024-07` | **403 Access Denied** (token required) |
| `GET https://sdsc.osn.xsede.org/bio230014-bucket01/challenges/targets/project_id=neon4cast/duration=P1D/aquatics-targets.csv.gz` | 200; 2,026,711 B; 265,410 rows; DO/temperature 34 sites, chl-a 10 sites; 2016-03-04 → 2026-07-31 |
| `GET …/project_id=usgsrc4cast/duration=P1D/river-chl-targets.csv.gz` | 200; 380,325 B |
| `GET https://seagull-erddap.glos.org/erddap/info/index.csv?...` | 200 |
| `GET https://environment.data.gov.uk/hydrology/id/measures/KEWPON-do-i-subdaily-mgL/readings.csv?mineq-date=2024-08-01&max-date=2024-08-02` | 200 |
| `GET https://edirepository.org/news/news-20260727.00` | 200 (EDI sign-in notice) |
| `GET https://historical-forecast-api.open-meteo.com/v1/forecast?latitude=39.63&longitude=-90.61&start_date=2024-07-01&end_date=2024-07-03&hourly=temperature_2m,precipitation,shortwave_radiation` | 200 |
| `GET https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-05586300?f=json` | 200; COMID 3607116 |
| `GET https://api.epa.gov/StreamCat/streams/metrics?name=pcturbhi2019,pctcrop2019,elev&aoi=ws&comid=3607116` | 200; pctcrop2019ws 65.66, pcturbhi2019ws 1.49, elevws 206.3 |
| HF API `models/{amazon/chronos-2, mistralai/Timestral-209M, google/timesfm-3.0-pytorch, NX-AI/TiRex-2, …}` | 200; licenses as in a.4 |
| Local GPU micro-benchmark (4096² matmul, RTX 2080 Ti, torch 2.7.1+cu118) | fp32 11.5, fp16 51.3, bf16 7.1 TFLOPS; FlashAttention SDPA: "No available kernel" |

## Appendix B: pilot code (daily naive baselines, Section a.0)

Data: `GET https://api.waterdata.usgs.gov/ogcapi/v1/collections/daily/items?f=csv&monitoring_location_id=<site>&parameter_code=00010,00300,00095,63680,99133&statistic_id=00003&datetime=2007-10-01/2026-09-30&limit=50000&properties=time,value,parameter_code,approval_status,time_series_id`

```python
# per site & parameter: keep the longest time_series_id; daily array y (NaN gaps); log for turbidity
TRAIN_END, TEST = date(2019, 9, 30), (date(2021, 10, 1), date(2025, 9, 30))
clim_raw[k] = mean(y[train & doy == k])          # k = 1..365, training years only
clim[k]     = nanmean(clim_raw[k-15 .. k+15])     # circular +-15-day smoothing
anom = y - clim[doy]
for h in (1, 3, 7, 14):
    phi = dot(anom[t], anom[t+h]) / dot(anom[t], anom[t])  # pairs fully inside training
    # forecasts issued at t (only y<=t used), scored at t+h inside TEST:
    pers  = y[t];  climf = clim[doy[t+h]];  damp = clim[doy[t+h]] + phi * anom[t]
    NSE = 1 - sum((y[t+h] - f)**2) / sum((y[t+h] - mean(y[t+h]))**2)
# report the median across sites per variable and horizon
```

The hypoxia pilot (d.4) drew a random sample (seed 20261022) of long-record DO sites and pulled daily-minimum DO (statistic 00002) for WY2016–2025. Of the 21 sites downloaded before the rate limit, 18 had data; 5 of those 18 had at least one day below 2 mg/L, giving 55 onsets (an onset = a day < 2 mg/L after ≥ 7 days ≥ 2 mg/L).

## Appendix C: unverified items and open questions

- **Zhi et al. 2021 DO numbers.** The NSE values in Zhi et al. 2021 were not visible: the abstract has none and the full text returned 403.
- **Effects and units still unconfirmed:**
  - how USGS `SuppressData` thresholds affect publicly served values;
  - whether NEON publishes a phycocyanin channel;
  - NEON challenge inconsistencies: stream-temperature product ID (DP1.20035 vs 20053), 30- vs 35-day horizon, latency.
- **Licenses not confirmed:** neon4cast, VERA and usgsrc4cast targets; CDEC; Hub'Eau; Rijkswaterstaat.
- **Claims with unknown provenance:** the Chronos-2 CAMELS-DE results; the Zhu & Liu numbers; the Zheng 2026 numbers; Timestral/t0 training data; whether TiRex's torch backend and t0 actually run on Turing; how Triton behaves on driver 470.
- **Novelty is unconfirmed** for:
  - DL/differentiable metabolism estimators;
  - chloride-exceedance forecasting;
  - placebo-controlled WQ event benchmarks;
  - UK EA ammonium event forecasting;
  - multi-site fine-tuned TSFMs for river WQ.

  A full literature re-search is required before using "first".
- **Paths not followed up:** NERRS `-p`/`-s` station suffixes; StreamPULSE account policy for non-bulk API use; bulk scripted download of the large ScienceBase metabolism files.
- **"Derived" numbers are ours, not peer-reviewed.** These are the EFI challenge skill/CRPS values, the GIFT/fev recomputations, TIME vs persistence, and our USGS pilots. Sample sizes are small for the 2026 TSFM challenge entries.
