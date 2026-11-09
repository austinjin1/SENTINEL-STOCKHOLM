# Landscape Survey — AI for freshwater water quality, disaster early warning and ecological insight

**Date:** 2026-10-22 · **Scope:** the seven SENTINEL v2 goals · **Method:** ten parallel research
investigations with live data-access checks; full reports (≈ 9,000 lines, ≈ 1,000 references
checked against Crossref/arXiv/OpenAlex/publisher records) are in `docs/research/01–10`. This
file is the synthesis. Items the reports marked *unverified* are flagged here as such; novelty
statements are provisional until the pre-writing literature re-search (several agents hit
web-search quotas).

---

## 1. Cross-cutting picture

### 1.1 Current dominant methods
- **Hydrology/WQ time series:** multi-site LSTMs (NeuralHydrology lineage: EA-LSTM, MTS-LSTM, CMAL heads) remain the strongest hydrology-specific deep baselines; gradient-boosted trees on lag features are the hardest non-deep baseline; general time-series foundation models (Chronos-2, TimesFM-2.5/3, Toto, Moirai-2, Sundial, TiRex, Timestral — the last currently tops GIFT-Eval/fev-bench) are entering environmental forecasting, mostly zero-shot.
- **Satellite water quality:** Mixture Density Networks (Pahlevan et al.) trained on in-situ hyperspectral reflectance resampled to sensor bands are the state of the art for chl-a/TSS from Sentinel-2/3 and Landsat on in-situ spectra; on real atmospherically corrected imagery they degrade sharply (chl-a MdSA ≈ 195%); band-ratio algorithms remain common; EO foundation models (Prithvi, Galileo, Presto, DOFA, TerraMind, OlmoEarth, AlphaEarth embeddings) are rarely validated on water.
- **Microbiome ML:** RF/Ridge/XGBoost on CLR-transformed abundances are hard to beat in-distribution (Li et al. 2025, 83 cohorts; Karwowska et al. 2025); deep/transformer models win mainly cross-study (Pope et al.: 0.805 vs RF 0.530 on an external cohort).
- **Ecotoxicology:** QSAR with RF/XGBoost on fingerprints; deep models (TRIDENT, *Sci Adv* 2024; a 2026 multitask MMoE model; GTGT; species-aware models) on ECOTOX-derived sets; ADORE is the benchmark with fixed splits but only RF/XGBoost-class results (micro-RMSE 0.927).
- **Bioassessment:** RIVPACS-style O/E models and random forests (Hill et al. 2017: StreamCat predictors, 75% OOB, AUC 0.84, random holdout); no deep model among the 74 works citing Hill et al. 2017.
- **Fusion:** Perceiver/HEALNet-style latent bottlenecks, modality dropout, MoE (Flex-MoE, FuseMoE), ImageBind-style hub alignment; in tabular-scale problems GBM fusion is usually competitive or better.
- **Early warning:** threshold/CUSUM detectors, EPA CANARY (archived 2026-06-24), anomaly scores evaluated with point-adjust or NAB windows — practices now known to inflate skill (Kim et al. 2022).

### 1.2 Known open problems
1. **Evaluation inflation:** random splits, same-station tests, look-back detection windows, point-adjusted F1. A 45-day look-back plus 11 false alarms per 30 clean days makes any random date "detected" with probability ≈ 1 (the v1 failure, docs/research/09 §b.1).
2. **Strong naive baselines:** damped-anomaly persistence reaches 14-day NSE 0.95 (temperature) / 0.86 (DO) on USGS data; only 10/29 temperature and 6/15 DO models beat climatology in the NEON challenge; none beyond 15 days.
3. **Unmonitored locations:** virtual sensing at unseen sites is weak (nitrate KGE ≈ 0.18; SSC R² 0.55; stream-temperature RMSE 1.13–1.20 °C in new basins).
4. **Modality co-location:** public modalities are rarely co-located; NEON's 34 aquatic sites are the only verified network with sensors, chemistry, 16S, metagenomes, macroinvertebrates, periphyton, fish and airborne imagery at the same sites (≤ 616 site-months with 5 modalities).
5. **Event labels:** most incident catalogues are complaint- or effort-driven (only 23% of 2024 NRC water reports have coordinates); landmark incidents (Toledo, Elk River, East Palestine, Dan River, Gold King) were largely invisible to public sensors at the time.
6. **Physical consistency of ML scenarios:** published LSTMs give physically wrong warming responses (Wi & Steinschneider 2022/2024; Reichert 2024).
7. **Causal discovery on environmental series:** confounding by diel/seasonal drivers, autocorrelation and missing light variables (the v1 PCMCI+ run found 0 of 3 chains for exactly these reasons).

### 1.3 Recent breakthroughs (last 1–2 years)
- Covariate-aware, multivariate TSFMs with official fine-tuning (Chronos-2, Apache-2.0; runs in fp32 on our GPUs, 120M parameters).
- Live, independently scored ecological forecasting challenges (NEON/neon4cast — live scoring stalled since Aug 2026; VERA reservoirs with a bloom target; usgsrc4cast river chlorophyll).
- Continental river-WQ deep learning (CAMELS-Chem; Zhi et al. DO/temperature analyses) and differentiable/process-guided hybrid models.
- Large open biomonitoring archives (UK EA: 280,992 invertebrate samples, 30,735 sites, 1965–2026).
- NEON metagenomes processed in NMDC; national-scale biology + chemistry surveys (NLA 2022: 981 lakes with microcystin/cylindrospermopsin).

### 1.4 Standard datasets & benchmarks (verified access)
| Domain | Dataset / benchmark | Notes |
|---|---|---|
| Sensors | USGS continuous WQ (DO 1,551 sites; SC 2,558; turbidity 1,444; nitrate 280; phycocyanin 133) | New OGC API (key recommended); legacy WaterServices off 2027-02-22 |
| Sensors | NEON aquatic (34 sites) | Token required since 2026-06-30; CC BY 4.0 |
| Forecast challenges | neon4cast aquatics; VERA; usgsrc4cast | Daily targets, CRPS scoring vs climatology/persistence |
| Satellite | GLORIA (7,572 Rrs + WQ), AquaCorr-Global (1.98M matchups; S2 chl-a 113,720), AquaSat, CyAN, WQP matchups, Tick Tick Bloom (SeaBASS, 23,570 points) | new matchup set needs audit |
| Microbial | NEON 16S/ASV + metagenomes; EMP (freshwater 4,915 samples, 14 studies) | EMP cannot carry a freshwater WQ claim |
| Ecotox | ADORE (fixed splits; noise floor 0.383 log units) | RF/XGB micro-RMSE 0.927 |
| Bioassessment | EPA NRSA/NLA, UK EA, NEON | blocked RF baselines measured in research |
| Health/ecology labels | WQP beach FIB, Chicago beaches, WA toxic algae, Iowa fish kills, SDWIS/UCMR, USGS NAS, Appling 2018 metabolism | see docs/research/07 |
| Aerial | NAIP 4-band, NEON AOP | no public drone RGB+NIR × in-situ dataset exists |

### 1.5 What everyone is already doing (avoid as a headline)
- Zero-shot TSFM forecasting of generic benchmarks; "transformers for water quality" on single-site random splits.
- Satellite chl-a regressions with random-pixel splits; band-ratio tuning.
- Microbiome classifiers on random splits; DEG-based toxicogenomic classifiers.
- "Species-aware deep toxicity model" (taken: TRIDENT, GTGT, MMoE 2026).
- "First drone water-quality model" / "first RGB water-quality model" (prior work: Su & Chou 2015, Cheng 2020, HydroColor, HydroVision, Belcore 2019).
- Point-adjusted anomaly-detection leaderboards.

### 1.6 Gaps identified (SENTINEL v2 targets)
1. A **leakage-free, station- and time-blocked multi-variable sonde benchmark with an event track** (none found).
2. **Event-onset forecasting with placebo/surrogate controls, frozen thresholds and pre-registered live evaluation** for freshwater hazards (no WQ study found doing this).
3. **Virtual sensing at unseen stations** with covariate-aware foundation models + physics (metabolism) heads.
4. **Differentiable metabolism heads** inside multi-site DO models, validated against independent metabolism estimates (novelty unverified).
5. **Deep bioassessment under ecoregion blocking** (no deep model in the citing literature of the main RF benchmark).
6. **Deep regression on ADORE's recommended and cross-taxa splits** (only RF/XGB published; no cross-taxa regression baseline).
7. **Sequence-level microbial models that transfer across primer sets/releases** on a sensor-co-located network.
8. **Any-subset deep fusion tested against GBM fusion under spatial blocking** for water quality (none found).
9. **A national NAIP–in-situ matchup set and an openly released RGB+NIR student distilled from a red-edge satellite teacher** (none found in two searches).
10. **Fish-kill risk models with reported discrimination** (none found; Iowa DNR API provides labels).

---

## 2. Per-goal summaries

### 2.1 In-situ sensors (docs/research/01)
Baselines are strong at short horizons and for temperature/DO; room for gains is in event-driven variables (turbidity, SC, nitrate, phycocyanin) at 2–7 days with operational weather, unseen-station virtual sensing, and rigorous events. Recommended bake-off: Chronos-2 fine-tuned with covariates, NeuralHydrology-style LSTM, differentiable metabolism head for DO, calibrated event head, GBM; ensemble. Access: USGS OGC API (anonymous 1,000 req/h; key needed in practice), legacy DV still serves multi-decade multi-site requests (being used for the daily tier), legacy IV limited to ≈ 2-month windows. Traps: USGS `SuppressData` thresholds at ≥ 2 mg/L on 250 DO series (possible hypoxia censoring), split time-series IDs, RFU vs µg/L, nocturnal ~3-h hypoxia missed by daily means.

### 2.2 Satellite (docs/research/02)
MDN is SOTA on in-situ spectra (chl-a MdSA ≈ 26–30%; ≈ 46–48% with held-out datasets) but collapses on real atmospherically corrected Sentinel-2 pixels (≈ 195% chl-a, ≈ 110% TSS; Saranathan et al. 2024). Location and season dominate: in a Minnesota pilot the site mean explained 74% of log chl-a variance and location-only kNN (86–109%) beat MDN on real imagery; GBM on imagery reached 73% (58% with context). The open problem is imagery → in-situ WQ on unseen lakes in unseen years versus a retrained MDN and GBM. New data: AquaCorr-Global (Aug 2026, 1.98M matchups, 113,720 S2 chl-a; needs audit). Access: Earth Search `sentinel-2-c1-l2a` (consistent processing) and Planetary Computer (beware the processing-baseline 04.00 offset and double-correction). Recommended model: spectral-token encoder with wavelength tags + water-pixel pooling + metadata dropout + MDN-style head, GLORIA-pretrained; foundation models (OlmoEarth, TerraMind, Galileo, Copernicus-FM) as ablations; TESSERA masks water (unusable).

### 2.3 Microbial / molecular (docs/research/03)
NEON is the only true sample-level co-location of 16S with sensors and chemistry (token needed; pin one release — AY26 OTU set incompatible with AY25; primers changed twice). Don't expect to beat RF in-distribution; the defensible claim is cross-primer/cross-release transfer by a sequence-level set model under leave-one-domain-out, with modest absolute skill (Hermans et al. 2024: R² 0.02 for stream N). Toxicogenomics should be dropped as a headline (cross-study DEG overlap is low). Secondary: deep microbial source tracking (benchmark designs exist), ARG burden from NEON metagenomes in NMDC.

### 2.4 Ecotox & bioassessment (docs/research/04)
Behavioural field biomonitors keep data private — v1 "BioMotion" cannot be rebuilt as claimed. Primary replacement: community bioassessment (NRSA, NLA, UK EA, NEON) under ecoregion blocking; blocked RF baseline R² 0.34 (TP), 0.39 (TN), 0.33 (conductivity), 0.33 (chloride), 0.39 (turbidity), 0.29 (pH); geography-only control collapses from 0.50 (random CV) to 0.06. Secondary: ChemTox on ADORE + spill severity from NRC incident files (CAS, amount in water, lat/lon). Optional: zebrafish photomotor lab model (beware plate-position shortcuts).

### 2.5 Fusion (docs/research/05)
Honest fusion gains are modest; GBM and persistence are the real opponents at 10³–10⁴ labelled rows. Recommended: Perceiver/HEALNet hybrid with absent-token missingness, Δt embeddings, per-target × horizon queries, modality dropout incl. single-modality episodes, unimodal heads, frozen small encoders, hub alignment; pre-registered paired evaluation vs best single modality and GBM-concatenated under spatial blocking; exact Shapley over modality subsets.

### 2.6 Simulation, causal, interpretability (docs/research/06)
Replace latent neural-ODE twins with physics-guided multi-site LSTMs whose physics module fixes DO saturation and gas exchange while the network predicts GPP/ER/K600; perturb only weather/flow/light; physics checks and train-cool/test-hot tests before any scenario. Causal claims follow a validation ladder (simulation recovery → positive/negative controls → replication → natural experiments) with Bagged-PCMCI+/LPCMCI/J-PCMCI+ including radiation and discharge. Insight bets: warming-driven oxygen loss decomposition (solubility vs metabolism, night vs day), upstream→downstream propagation and lead time, groundwater-fed resilience.

### 2.7 Downstream applications (docs/research/07)
Eight applications with real labels and real incumbents (beach FIB, cyanotoxins, fish kills, drinking-water risk, biological condition, invasive mussels, cold-water fish, metabolism). Cross-fit upstream predictions; decision-time feature rules (FIB culture lag 18–24 h); case-crossover for fish kills. Chicago beach bar ≈ AUROC 0.74 (same-day lab average) vs city model 0.70 and persistence 0.645.

### 2.8 Drone RGB+NIR (docs/research/08)
RGB+NIR supports clarity/Secchi, turbidity/TSS, water masks and surface-scum/plume detection; chl-a only as trophic classes; no subsurface phycocyanin, CDOM or non-optical variables. Losing red-edge costs ≈ 20 MdSA points for chl-a on held-out GLORIA datasets (research pilot); calibration error matters as much. Train on NAIP 4-band matchups (8-bit, uncalibrated — main risk) with an S2 red-edge teacher; deploy on Raspberry Pi 5 + Hailo-8L; budget ≤ 3M params, ≤ 4 MB INT8, ≤ 100 ms/tile.

### 2.9 Events & early-warning evaluation (docs/research/09)
Frame early warning as onset forecasting scored at every issue time (BSS vs climatology primary), thresholds frozen on a calibration period (≤ 1 false-alarm episode per site per 30 d), mandatory placebo/surrogate controls, Molchan diagrams, ≥ 50 test onsets per hazard, pre-registered live trial (OSF by ≈ 2026-11-15; external scoring via challenges). Event definitions anchored to regulatory criteria (EPA DO 3.0 mg/L 1-day minimum; chloride 230/860 mg/L; nitrate MCL 10 mg/L; EPA microcystin 8 µg/L; E. coli BAV 235 cfu/100 mL; WHO chl-a alert levels).

### 2.10 Local assets (docs/research/10)
Little raw data survives locally; reusable: v1 microcystin benchmark (3,649 rows; satellite features need re-extraction — processing-baseline offset and band-footprint mismatch found), raw WQP toxin/pigment tables, E. coli results, event catalogues and site lists, NHDPlus site→COMID map, EMP taxonomy matrices. No v1 checkpoint is worth using as an initialiser.
