# SENTINEL v2 — Research Plan

**Honest, multimodal deep learning for freshwater water-quality forecasting, disaster early warning, and ecological insight**

| | |
|---|---|
| **Field** | Environmental machine learning — freshwater water quality, hydroinformatics, aquatic ecology |
| **Authors** | Austin Jin & Bryan Cheng (Stockholm Junior Water Prize). AI-assisted: substantial code, analysis and drafting by Claude (Anthropic); see §21 for the disclosure this requires |
| **Constraints** | Data: public/open sources only (US-centred; UK/EU where useful); ~350 GB free local disk. Compute: ≤ 5 × RTX 2080 Ti (11 GB; fp32/fp16, no bf16; NVIDIA driver 470 → CUDA ≤ 11.8, torch 2.7.1+cu118), 48 CPU cores, 500 GB RAM; no wall-clock limit |
| **Date** | 2026-10-22 (plan v1.0) |
| **License** | MIT — Bryan Cheng, 2026 |
| **Branch** | `v2-rebuild` (local; nothing pushed) |
| **Inputs** | `REVIEW_REPORT.md` (v1 integrity audit, 2026-10-21); `LANDSCAPE_SURVEY.md`; ten research reports in `docs/research/` |

> **Status legend used throughout:** every quantitative target below is a *goal*, not a result. Results exist only in `results/registry.jsonl` and are rendered into `RESULTS.md`. Placeholders read `[PENDING]`.

---

## 0. How this plan was made, and deviations from the standard workflow

1. **Research first.** Ten parallel literature/data investigations (Oct 22, 2026) produced `docs/research/01–10_*.md` (≈ 9,000 lines, ≈ 1,000 verified references, live API checks). `LANDSCAPE_SURVEY.md` synthesises them.
2. **No proposal tournament.** The research-engine workflow normally generates 5–8 competing proposals and asks the user to pick one. Here the user specified the goals (G1–G7, §1), so the "selected proposal" is the user's brief. Competing *designs* were still generated and critiqued per component (each report's "recommended approach" and "risks" sections), and the cheapest feasibility probes were run during research (baseline pilots on real USGS data, ADORE noise floor, bioassessment blocked baseline, Chicago beach baseline, NEON co-location audit, Chronos-2 on our GPUs).
3. **Package layout.** The engine's template uses `src/`; v2 keeps the importable package name `sentinel/` (see §20).
4. **User interaction.** Execution is autonomous. The only user requests are free credentials that only the user can create (§13.4): a USGS Water Data API key and a NEON API token. Work proceeds without them on everything else.

---

## 1. Goals (the user's brief, verbatim intent) and how each is addressed

| # | Goal (user) | v2 deliverable | Plan § |
|---|---|---|---|
| **G1** | 5+ SOTA (or first-in-class) deep models, different data modalities, that predict water-quality disasters **and** the water-quality indicators themselves | Seven modality models M1–M7 (+ optional M8), each with a frozen benchmark, the strongest published/operational baseline, and a pre-registered SOTA/first-in-class test | §5 |
| **G2** | Multimodal fusion of 5+ models predicting ALL water-quality features + disasters better than single models; plug-and-play when data types are missing | **SENTINEL-Fuse**: any-subset Perceiver/HEALNet-style fusion over M1–M7 tokens, modality dropout, per-target × horizon queries, adapter protocol for new modalities; pre-registered paired tests vs best single modality *and* vs GBM on concatenated features | §6 |
| **G3** | Future-simulation platform using derived features + model interpretation identifying causal chains, ecological interactions, novel insights | **SENTINEL-Sim** (physics-guided multi-site LSTM with differentiable DO metabolism; validated scenario engine), causal-discovery "claim ladder", interpretability suite with fidelity tests, three pre-specified insight studies (E1–E3) | §7 |
| **G4** | Downstream applications: disease, species health, other ecological factors (more than v1) | Eight applications with real labels: SafeSwim, ToxinWatch, KillAlert, TapRisk, BioIntegrity, MusselGuard, TroutRefuge, RiverPulse (+ stretch) | §8 |
| **G5** | A usable platform | **SENTINEL Console**: FastAPI service + web app (map, forecasts, alerts, scenario simulator, causal explorer, downstream dashboards, upload-and-predict for every modality), model cards, Docker | §9 |
| **G6** | Distilled/miniature drone model using only NIR + visible cameras | **SENTINEL-Lite (M7)**: RGB+NIR student distilled from the satellite/fusion teachers, trained on real 4-band aerial (NAIP) and Sentinel-2 RGBN matchups, INT8/ONNX, latency- and robustness-tested; ROS2 integration path | §10 |
| **G7** | Anything else valuable | Early-Warning v2 + **pre-registered live prospective trial** + external challenge submissions (NEON, VERA, usgsrc4cast); **SENTINEL-Bench** open benchmark release; calibrated/conformal uncertainty; monitoring-network design (where to add sensors); equity analysis (SVI); model cards/datasheets; honest LaTeX paper | §11 |
| — | "Fix the early-warning claims" | **EW-v2 protocol** (onset forecasting, frozen thresholds, placebo/surrogate controls, BSS vs climatology, live trial) replaces v1 Track-A claims | §12 |

---

## 2. Background & motivation

Freshwater disasters — cyanobacterial blooms (Toledo 2014), hypoxic fish kills, road-salt salinisation, chemical spills (Elk River 2014, East Palestine 2023), sewage/fecal contamination, sediment pulses after storms and wildfires — are monitored by sparse grab sampling, a few thousand continuous sondes, satellites, and biological surveys that rarely talk to each other. Machine learning promises to (i) estimate water quality where it is not measured, (ii) forecast it, and (iii) warn before thresholds that matter are crossed. The literature (see `LANDSCAPE_SURVEY.md`) shows three recurring weaknesses that SENTINEL v2 is designed around:

1. **Weak evaluation.** Random splits, same-site train/test, point-adjusted anomaly scores and look-back "detection windows" inflate skill. Naive baselines are strong: on real USGS data a two-parameter damped-anomaly baseline already reaches 14-day NSE 0.95 (temperature) and 0.86 (DO) (docs/research/01 §a.0), and in the NEON forecasting challenge only 10/29 temperature and 6/15 DO models beat climatology.
2. **Modal silos.** Sensor, satellite, microbial, chemical and biological evidence are modelled separately; honest multimodal gains are small and only appear for genuinely complementary signals (docs/research/05).
3. **No prospective validation.** Early-warning systems are almost never scored on pre-registered, live forecasts against placebo controls (docs/research/09).

### 2.1 Lessons from SENTINEL v1 (REVIEW_REPORT.md) → v2 design rules

| v1 failure (audit ref) | v2 rule |
|---|---|
| Fig 7 "SOTA" bars were typed-in literals; figures showed retracted numbers (C1) | Figures/tables are generated only from `results/registry.jsonl` through `registry.lookup()`; a CI check fails on numeric literals in figure scripts (R2) |
| Track-A "25/32 at ≤5% FPR" failed placebo (95–99% of clean dates alarmed); FPR in-sample; 45-day look-back window (C2) | EW-v2 (§12): onset forecasting, thresholds frozen on a calibration period, placebo/surrogate controls mandatory, BSS vs climatology primary |
| −999999 missing codes in normalisation stats; reconstruction head never trained (C2) | Missing-code masking at ingest (tested), manifests with statistics, training-only normalisation, run logger records every run incl. failures |
| Headline numbers came from GBM/logistic baselines or ImageNet ViT, credited to SENTINEL models (C4) | Attribution rule (R5): every number names the model that produced it; baselines are first-class rows; if GBM wins, the paper says so |
| Synthetic/circular data presented as real: twin on 300 synthetic samples, GNN label = threshold of its input, behaviour data that did not exist (C5) | `synthetic` flag in manifests; synthetic data allowed only in unit tests and declared simulation-recovery studies; label-circularity check in the pre-training review |
| Day-of-year + latitude baseline beat HydroViT (R² 0.87 vs 0.729); "strict spatial holdout" was a random split (C6) | Mandatory "trivial-context" baselines (DOY + location + site mean) in every table; spatial splits built and verified by `sentinel.eval.splits` with distance reports |
| 10/30 references wrong, 3 nonexistent (C7) | Bibliography triple-pass verification before any paper draft is circulated |
| Acknowledgements contradicted repo (A100s, CSHL, AI co-author trailers stripped) (C8) | Accurate disclosure (§21); AI co-author trailers kept; compute logged automatically |
| Parameter counts wrong (C9) | Parameter counts computed from checkpoints and logged to the registry |
| Public artifacts stale (README, HF, demo with `Math.random` data labelled OBSERVED) (C10) | Platform shows only registry-backed numbers and live data with provenance; demo data must be labelled |

---

## 3. Integrity rules (apply to every workstream)

- **R1 Zero fabrication.** No number appears anywhere unless code computed it and wrote it to the registry. `[PENDING]` is the only allowed placeholder.
- **R2 Registry-only reporting.** `sentinel.core.registry.log_result()` is the single write path; `lookup()` the single read path for figures/tables/paper/platform. Entries are append-only; corrections supersede.
- **R3 Provenance.** Every dataset has a manifest (`sentinel.core.manifest`: sources, timestamps, sha256, `synthetic` flag). Every run snapshots code (package hash + script sha256) and config hash.
- **R4 Leakage tests in code.** `assert_disjoint` on station/lake/chemical/study IDs; embargo checks; normalisers/PCA/CLR/climatologies fit on training data only; flow-connected stations kept in the same fold; no observed future weather in anything called a "forecast".
- **R5 Attribution.** Each result names its model; deep models must be compared with the strongest simple baseline under the identical split; "SOTA" / "first-in-class" labels follow §15.1 only.
- **R6 Test sets are touched once.** Model selection uses validation/calibration data. Final test scoring happens after the scoring script and thresholds are frozen and hashed (`docs/preregistration/`). The number of configurations tried is reported.
- **R7 Uncertainty.** Every headline metric gets a block/cluster-bootstrap CI with the resampling unit stated (site, lake, chemical, ecoregion); paired comparisons use paired bootstrap / Diebold–Mariano; multiplicity corrected with Holm across primary endpoints; ≥ 3 seeds (5 for fusion).
- **R8 Negative results are results.** Failures are logged (`TRAINING_LOG.md` is written automatically by `TrainingRun`, including crashes) and reported.
- **R9 Synthetic data only where declared.** Unit tests and simulation-recovery experiments (e.g. causal discovery on simulated physics) — always labelled as such.
- **R10 Review gates.** Pre-training review (code, splits, leakage) before each workstream's first real training run; post-results review before any number leaves the repo (`REVIEW_REPORT_PRE_TRAINING.md`, `REVIEW_REPORT_POST_RESULTS.md`).
- **R11 Licenses.** A license register per data source and model (`docs/licenses.md`); non-commercial models used only as research baselines.
- **R12 Disclosure.** AI assistance and compute disclosed accurately (§21).

---

## 4. System architecture

```
                 ┌──────────────────────── DATA HUB (sites × time) ────────────────────────┐
 USGS sondes ──► │ M1 SENTINEL-Sonde   (forecast / virtual-sense / event-onset)             │
 Sentinel-2/L8 ─►│ M2 SENTINEL-Orbit   (optical WQ retrieval + bloom/plume detection)       │
 NEON 16S/ASV ──►│ M3 SENTINEL-Microbe (sequence-level community → WQ, source, ARG)         │
 SMILES/CAS ────►│ M4 SENTINEL-ChemTox (structure → aquatic toxicity → spill severity)      │
 Inverts/fish ──►│ M5 SENTINEL-BioAssess (community ↔ water chemistry & condition)          │
 Weather+basin ─►│ M6 SENTINEL-HydroMet (forcing + catchment → WQ at unsensed reaches)      │
 RGB+NIR drone ─►│ M7 SENTINEL-Lite    (distilled RGBN student; edge INT8)                  │
 (opt) lab video►│ M8 ZebraMotion      (embryo photomotor behaviour → toxicity; lab only)   │
                 └───────────────┬──────────────────────────────────────────────────────────┘
                                 │ tokens + calibrated predictive distributions
                                 ▼
                 ┌─────────── SENTINEL-Fuse (any-subset fusion, all targets × horizons) ─────┐
                 └──────┬─────────────────────┬───────────────────────────┬──────────────────┘
                        ▼                     ▼                           ▼
              EW-v2 onset forecasts   SENTINEL-Sim (scenarios,     Downstream apps (D1–D8)
              + live trial            metabolism, causal, insights) health / species / ecology
                        └────────────── SENTINEL Console (API + web app) ─────────────────────┘
```

Shared conventions: site key = agency site ID + NHDPlus COMID + HUC12 (strings); time = UTC; units are part of a variable's identity (RFU ≠ µg/L; FNU ≠ NTU); every model exposes `predict(...) → distribution`, `embed(...) → tokens`, and a model card.

---

## 5. Goal 1 — Modality models (M1–M8)

Each model section lists: task · data · benchmark & splits · baselines (published + simple) · architecture · training · metrics · success thresholds (also collected in §15) · main risks. Detailed evidence lives in the referenced research report.

### 5.1 M1 SENTINEL-Sonde — in-situ sensor time series (docs/research/01)

- **Tasks.** (F) probabilistic multi-horizon forecasts (hourly +1…+72 h; daily mean/min/max +1…+14 d) of water temperature, DO (mg/L; daily min), specific conductance, pH, turbidity (log), nitrate (NO3+NO2-N), fDOM, fChl and fPC (RFU and µg/L never mixed), log-discharge; (N) virtual sensing: (N-a) withheld variable at a seen station, (N-b) at unseen stations; (E) event-onset probabilities (EW-v2 hazards H1–H4, H5b) within 1/3/7 d.
- **Data.** USGS continuous network: daily min/mean/max for all WQ sites via NWIS DV (downloading now), 15-min data for the core set via the Water Data OGC API (`continuous`, ≤ 3 yr/request, 50k rows/page; API key requested). Core site set: ≥ 5 yr overlap of DO + SC + temperature (~610 stream sites by metadata; 424 with DO+T+Q active ≥ 5 yr); subsets: turbidity (~377), nitrate (~124), algal PC (~133). External network: NEON (34 sites; token needed; EFI targets as token-free daily fallback). Covariates: Open-Meteo/ERA5 & gridMET past weather, **archived operational NWP** for forecast mode, clear-sky PAR, StreamCat/NLDI static attributes.
- **Benchmark "SENTINEL-Sonde"** (frozen manifest). Time blocks (shared with EW-v2, §14.1): Train WY2008–2021; Validation/Calibration WY2022–2023; **Test-A WY2024–2025**; **Test-B WY2026** (post-release guard for pretrained TSFMs); **Test-C live** from Nov 2026. 42-day embargo at block boundaries. Station folds: 5 folds grouped by HUC4 with flow-connected stations together; 20% of HUC4s never used in training or calibration (unseen-station headline). Cells reported: seen×Test-A, unseen×train-period, **unseen×Test-A (headline)**, NEON×Test-A/B.
- **Baselines** (same inputs/splits): persistence and diel persistence; station climatology (DOY × hour, training years); damped-anomaly persistence; statsforecast (AutoETS/AutoARIMA/Theta); global LightGBM/XGBoost on lags/rolls/covariates/attributes; NeuralHydrology-style multi-station LSTM (EA-LSTM/MTS-LSTM, CMAL head); PatchTST, iTransformer, N-HiTS, TFT, TimeMixer, TiDE; zero-shot TSFMs on Test-B/C: Chronos-2, TimesFM-2.5, Toto, Sundial, Moirai-2 (separate env), TabPFN-TS, Timestral-209M (if license permits research use); physics references (DO_sat × ratio, air→water temperature regression, surrogate regressions turbidity→SSC, SC→Cl).
- **Architecture (staged bake-off; selection on validation only).** (A) Chronos-2 (120M, Apache-2.0; verified to run in fp32 on our GPUs) fine-tuned with past/known-future covariates and group attention across co-located sensors; (B) multi-station hydro-LSTM; (C) **differentiable metabolism head for DO** — network predicts daily GPP, ER, K600 (positivity + priors), DO integrated through `dDO/dt = GPP·PAR(t)/ΣPAR/z − ER/z + K_O2(T)(DO_sat(T,P) − DO)` (shared with SENTINEL-Sim, §7); (D) event head: exceedance probabilities from the predictive CDF + directly trained onset classifier, conformal/isotonic calibration on validation. Final = validation-weighted ensemble of A–D + GBM.
- **Metrics.** CRPS (19 quantiles) and CRPSS vs climatology / persistence-error ensemble; MSE skill vs best naive baseline; NSE/KGE/RMSE (median & IQR across stations); 50/90% coverage & width; PIT; threshold-weighted CRPS at event thresholds; event metrics per §12. Station-block bootstrap CIs; Diebold–Mariano; win/tie/loss counts.
- **Realistic claims.** Largest defensible wins: event-driven variables (turbidity, SC, nitrate, PC) at 2–7 d with operational weather; **N-b virtual sensing at unseen stations** (published nitrate KGE ≈ 0.18); event-onset forecasting with first-in-class rigour. Small/no gains expected for T/DO/SC/pH at 1–24 h and T/DO beyond ~10 d — reported honestly.

### 5.2 M2 SENTINEL-Orbit — satellite optical water quality (docs/research/02)

- **Why the claim space is open.** The state of the art (Mixture Density Networks, Pahlevan et al.) reaches MdSA ≈ 26–30% for chl-a on in-situ spectra (≈ 46–48% with whole datasets held out), but on real atmospherically corrected Sentinel-2 imagery the same MDN degrades to ≈ 195% (chl-a) and ≈ 110% (TSS) (Saranathan et al. 2024, 3,053 matchups); Pahlevan et al. 2022 judged neither MDN nor OC3 viable there. **No published study evaluates satellite pixels → in-situ WQ on unseen lakes in unseen years against a retrained MDN.**
- **The location/season trap (research pilot, real data):** on 844 same-day Minnesota S2 matchups with 25-km block CV, the site mean explains 74% of log chl-a variance; a location-only kNN reaches MdSA 86–109% (already better than MDN on real imagery); GBM on imagery alone 73%, + season & location 58%; within-lake skill ρ = 0.33. **Plain GBM on imagery + context is the bar.**
- **Tasks.** Retrieve chl-a, TSS, turbidity, Secchi depth (and aCDOM, phycocyanin where labels allow — 10–1000× scarcer) from Sentinel-2 MSI (primary) and Landsat 8/9 (secondary); bloom detection (WHO/EPA chl-a classes; Tick Tick Bloom cell counts); per-lake time series + uncertainty as fusion tokens; teacher for M7. *Not claimed:* inland oil/chemical spills, direct hypoxia/fish-kill/salinisation detection, PACE inland products.
- **Data.** (i) **AquaCorr-Global** (Zenodo, Aug 2026, CC-BY; 1.98M same-day matchups; S2 subset 113,720 chl-a incl. 23,370 lab-extracted; ACOLITE reflectance) — new, not peer-reviewed, from the MDN group: audit (duplicates, matchup windows, QC) before use; (ii) fresh WQP (WQX 3.0) chl-a/TSS/turbidity/Secchi matched to S2 overpasses with exact acquisition times, scene IDs, SCL cloud/water masks, JRC water occurrence, and **processing-baseline offset handling** (subtract 1,000 DN for baseline ≥ 04.00 on Planetary Computer; beware already-corrected legacy Earth Search items); (iii) GLORIA for pretraining/sanity only; (iv) NLA 2022, Tick Tick Bloom (SeaBASS) for blooms. Imagery from Element 84 Earth Search `sentinel-2-c1-l2a` (consistent processing, no account); Planetary Computer for Landsat C2 L2, HLS, OLCI, JRC.
- **Benchmark "SENTINEL-Orbit".** Headline: **unseen lakes × future years (2024–2025)** with spatial blocks (≥ 25 km); secondary tracks: within-lake temporal skill (Spearman of anomalies), bloom detection (recall at fixed false-alarm rate + shuffled-imagery placebo), GLORIA held-out-dataset sanity check.
- **Baselines (all retrained on identical folds).** Location/season-only models incl. location kNN and site mean (when site is seen); index algorithms (MDN-STREAM repo: OC3, NDCI, band ratios); **MDN three ways** (official weights, retrained on GLORIA, retrained on our matchups); GBM on band statistics ± context; operational products where co-located (OLCI/Copernicus LWQ, Lakes CCI, CyAN, CyFi); EO foundation-model features (OlmoEarth, TerraMind, Galileo, Copernicus-FM; AlphaEarth annual embeddings as a site prior) as ablations.
- **Architecture "SatWQ-Net" (1–5M parameters).** Spectral-token encoder (each band a token tagged with its centre wavelength → transfers across sensors and to the drone's 4 bands), attention pooling over valid water pixels in the chip (+ adjacency/edge features), metadata (geometry, season, location) with random dropout so the model cannot lean on location, MDN-style mixture head for multiple targets with uncertainty; pretrained on GLORIA (simulated bands) then fine-tuned on satellite matchups; ensemble of 5 seeds.
- **Metrics.** MdSA, SSPB, slope, log-RMSE (MDN-literature comparable), interval coverage; bloom AUPRC/recall at fixed FAR; all with lake-cluster bootstrap CIs.

### 5.3 M3 SENTINEL-Microbe — microbial community sequencing (docs/research/03)

- **Tasks.** From NEON surface-water 16S/ASV community profiles (co-collected with grab chemistry near the sensor): predict co-located water chemistry and sonde summaries (TN, TP, DOC, nitrate, SC, turbidity, DO, chl-a), eutrophic/hypoxia-risk classes, and **antibiotic-resistance-gene (ARG) burden** derived from co-located NEON metagenomes (NMDC); secondary: deep microbial source tracking (fecal contamination) on the established blinded-spike/FEAST design.
- **Data.** NEON DP1.20282.001 / DP1.20141.002 (16S/ASV; 34 sites; 2014–2026; pin one release), DP1.20281.001 metagenomes (via NMDC), DP1.20093.001 chemistry, DP1.20288.001 sonde. EMP freshwater (4,915 samples, 14 studies) only for unlabeled pretraining / OOD testing (it cannot carry a WQ claim). **Requires NEON token** (except NMDC route).
- **Benchmark.** Leave-one-NEON-domain-out (primary); leave-one-site-out; **cross-primer** (NEON changed primers twice) and **cross-release** transfer (train on one primer set/release, test on another).
- **Baselines.** DOY + latitude + site-mean; RF/XGBoost/Ridge on CLR genus abundances; TabPFN; MGM / published microbiome transformers where runnable; permutation null; site-identity probe (does the model just recognise sites?).
- **Architecture.** Sequence-level set model: ASV sequences embedded by a pretrained DNA encoder (k-mer/CNN or small DNA LM), attention over CLR-weighted abundances (Set Transformer), multi-target heads — applicable across primer sets and OTU releases where table models cannot even be applied.
- **Realistic claim.** First molecular→WQ model evaluated under leave-one-domain-out on a sensor-co-located continental network, with cross-primer/release transfer; absolute skill expected to be modest (published comparable: R² ≈ 0.02 for stream N under random split). No daily early warning from monthly molecular data — stated explicitly.
- **Dropped:** toxicogenomics headline (cross-study signal not reproducible: docs/research/03 §6).

### 5.4 M4 SENTINEL-ChemTox — chemical structure → aquatic toxicity → spill severity (docs/research/04 Track A)

- **Tasks.** Predict acute LC50/EC50 (log10 mol/L) for fish, crustaceans and algae from structure + experimental conditions; build model-based species sensitivity distributions (HC5); score real spill incidents (USCG NRC reports: CAS, amount in water, lat/lon) by risk quotient with gauged dilution — a chemical-hazard channel for fusion and alerts.
- **Benchmark.** ADORE (Schür et al. 2023): `t-F2F`, `t-C2C`, `a-CA2F-same`, `a-CA2F-diff`; primary `split_occurrence`, secondary `split_scaffold-murcko`; folds 0–4 for selection; test scored once; chemical-overlap counts printed.
- **Baselines.** Global/species/species+duration means; published RF/XGBoost (micro-RMSE 0.927 / 0.904 / 0.938 / 0.845, Gasser et al.) reproduced first; TabPFN v2 on fingerprints; TRIDENT compared only on its released held-out per-SMILES errors (its training overlaps ADORE); replicate noise floor 0.383 log units.
- **Architecture.** Chemprop v2 D-MPNN with CheMeleon foundation initialisation + species/taxonomy embedding + duration/endpoint-conditioned head; frozen ChemBERTa-2/MoLFormer embeddings as alternatives; ensemble.
- **Realistic claim.** SOTA on ADORE recommended splits (5–15% RMSE reduction over 0.927 plausible) and first regression results on the cross-taxa challenge (`a-CA2F-diff`); spill-severity validation limited to consistency with recorded response actions (labels are weak) — stated as such.

### 5.5 M5 SENTINEL-BioAssess — biological community bioassessment (docs/research/04 Track B)

- **Tasks.** (B2, first) community → stressor: predict log TP, log TN, log conductivity, log chloride, log turbidity, pH (and M4-predicted mixture toxicity) from benthic macroinvertebrate (and diatom where available) assemblages; (B1) environment → biology: predict assemblages and condition class from chemistry + habitat + StreamCat + climate (used by D5 BioIntegrity and Sim).
- **Data.** EPA NRSA (2008-09, 2013-14, 2018-19; 2023-24 when posted), NLA 2022 (zooplankton/phytoplankton), UK Environment Agency ecology archive (280,992 invertebrate samples, 30,735 sites, 1965–2026, OGL v3.0; diatom metabarcoding), NEON macroinvertebrates (token).
- **Benchmark.** Leave-one-ecoregion-out (NARS aggregated ecoregions) and ≥ 50 km/HUC8 blocked variant; leave-one-survey-cycle-out; UK↔US transfer as an external test; random-CV reported only to quantify the leakage gap.
- **Baselines.** The blocked RF measured during research: **R² 0.34 (TP), 0.39 (TN), 0.33 (conductivity), 0.33 (chloride), 0.39 (turbidity), 0.29 (pH)**; geography-only control (0.06–0.31) in the same table; ecoregion-mean, prevalence-only, nearest-reference-site; WA-PLS (diatoms); RIVPACS-style O/E; Hill et al. 2017-style RF condition model (75% OOB, random holdout).
- **Architecture.** Taxon-embedding set transformer over present taxa (embeddings initialised from taxonomy/traits; total-count offset; mixed-resolution handling); multi-target heads; deep compositional likelihood for task B1; sjSDM/HMSC as baselines.
- **Realistic claim.** First deep bioassessment model evaluated under ecoregion-blocked CV that beats the blocked RF on community→chemistry; cross-continent transfer reported whatever the outcome.

### 5.6 M6 SENTINEL-HydroMet — weather + catchment → water quality at unsensed reaches

- **Tasks.** Daily water temperature (incl. 7-DADMax), DO, SC, turbidity and nitrate at reaches **without any sensor**, from meteorological forcings (gridMET/Daymet/ERA5 via Open-Meteo), discharge where gauged, and StreamCat/NLDI catchment attributes; also the "hub" context token used by fusion for every site-day.
- **Benchmark.** Unseen-HUC4 stations × Test-A (same blocks as M1), CAMELS-Chem-style catchments as a comparability check.
- **Baselines.** Air→water temperature regression and air2stream-type hybrid; GBM on forcings + attributes; DOY + location; published unseen-basin numbers as context only (stream temperature RMSE 1.13–1.20 °C, SC KGE 0.67 where RF beat a GNN, SSC R² 0.55, nitrate KGE 0.18 — different data, so not head-to-head).
- **Architecture.** Entity-aware LSTM / MTS-LSTM with static attribute conditioning, optional graph message passing along NHDPlus flow lines (upstream context), heteroscedastic/CMAL heads; shares the metabolism head with M1/Sim for DO.
- **Realistic claim.** Beats GBM and physics-hybrid baselines at unseen HUC4s with CI excluding 0 for ≥ 2 of 5 targets; else reported as competitive/negative.

### 5.7 M7 SENTINEL-Lite — drone RGB+NIR student

Specified under G6, §10 (counts toward G1 as the aerial-imagery modality).

### 5.8 M8 ZebraMotion (optional, lab-only behaviour)

Only if time permits after M1–M7: 24-h zebrafish embryo photomotor traces (~365 environmental chemicals; PFAS file 30,582 embryos) → 120-h morphology/mortality; chemicals and plates held out; plate-position randomisation control (published shortcut: AUROC 0.97 → 0.84); baselines = per-interval GBM and KS/AUC test; comparison Green et al. 2024. **Never described as in-situ early warning.** This is the honest replacement for v1 "BioMotion" (whose behavioural data never existed).

---

## 6. Goal 2 — SENTINEL-Fuse: plug-and-play multimodal fusion (docs/research/05)

- **Design principles.** Missing modalities are *absent tokens*, never imputed raw values; every modality can stand alone (unimodal heads/teachers) so fusion cannot fall below unimodal; frozen pretrained encoders + small fusion core (≈ 5–20M parameters); the **context hub** (forcings, catchment, satellite) is present for every row and binds non-co-located modalities; GBM, stacking and TabPFN are first-class competitors.
- **Data tiers.** **Tier D (NEON, 34 sites):** sonde, nitrate sensor, discharge, lab chemistry (12–26/yr), 16S (same sampling event), metagenomes (1/yr), macroinvertebrates & periphyton (3/yr), fish, zooplankton, airborne hyperspectral/RGB + hub; ≤ 616 site-months with sonde+chemistry+16S+macroinverts+periphyton (upper bound; confirm after download). **Tier C (cyanotoxins):** NLA 2022 (981 lakes) + WQP microcystin + v1's microcystin benchmark (3,649 rows, 227 sites; satellite features to be re-extracted — offset + band-footprint defects) + satellite + forcings. **Tier W (wide):** USGS sondes + WQP discrete + satellite + forcings + catchment (pretraining; FIB/nutrient tasks).
- **Architecture.** Per-modality tokenizers (M1–M7 encoders, frozen or adapters) → tokens with modality, variable, continuous Δt (log-scaled sinusoidal), cadence/QC embeddings → HEALNet/Perceiver hybrid core (64 latents × 256; modality-specific cross-attention K/V; 4–6 latent blocks) → query decoder with one query per (target × horizon) + event queries → Student-t/Gaussian heads (β-NLL; censored likelihood below detection limits) and event probabilities from the CDF.
- **Training.** Stage 0 audit + baselines (no DL) → Stage 1 unimodal encoders (each must match/beat its own GBM baseline or is replaced by engineered features) → Stage 2 masked multimodal pretraining + hub-contrastive alignment → Stage 3 supervised multi-target fine-tuning with modality-dropout schedule (20% all; 20% single modality + hub; else independent drop p = 0.3–0.5), unimodal auxiliary heads, availability-stratified batches; 5 seeds (one per GPU) → Stage 4 isotonic recalibration + Mondrian conformal (pattern × region) → Stage 5 pre-registered evaluation.
- **Plug-and-play protocol.** New modality = featurizer + encoder + projection + new modality embedding + new K/V weights; core frozen; acceptance test = paired blocked ΔCRPS/ΔAUPRC > 0 (CI excludes 0) for ≥ 1 target with no target worse than a pre-set non-inferiority margin; when absent, predictions on old rows are bit-identical (verified numerically).
- **Pre-registered endpoints.** **P1 (NEON):** 7-day CRPS skill for DO, turbidity, nitrate, chl-a vs (a) best single modality (chosen on validation) and (b) GBM on concatenated features, leave-site-out. **P2 (cyanotoxins):** microcystin exceedance AUPRC/AUROC under 5-km and 20-km blocks and leave-region-out (HUC parsed as strings). **P3:** ≥ 5 modalities each with positive leave-one-modality-out ΔCRPS (CI excludes 0) on paired NEON rows. **P4:** graceful degradation — for every availability pattern with enough rows, fusion ≥ the model trained on that pattern. **P5:** conformal coverage within ±3 points per pattern and region.
- **Honest fallback headline.** If deep fusion only ties GBM fusion: "missing-modality-robust deep fusion matches strong GBM fusion and adds calibrated any-subset inference" — with the full table.

---

## 7. Goal 3 — Simulation, causal chains, ecological interactions, insights (docs/research/06)

### 7.1 SENTINEL-Sim (scenario engine)
- **Core model.** Multi-site LSTM (NeuralHydrology lineage) with a **physics module for oxygen**: fixed Benson–Krause DO saturation (T, pressure) and gas-exchange scaling; the network predicts only daily GPP, ER, K600; simulates observed variables only (water temperature, DO mean/min/max, SC, turbidity, nitrate where sensed). Replaces v1's latent neural-ODE twin (6/10 states never observed; trained on synthetic fallback).
- **Baseline gate.** Hindcast skill reported vs persistence, DOY climatology and physics-only baselines on site- and region-held-out tests (v1 twin: R² 0.07 vs persistence 0.92).
- **Counterfactual rules.** Perturb only air temperature, precipitation/discharge, and light; no nutrient scenarios and no latent-embedding perturbations. Each scenario must first pass physics checks (e.g. +2 °C must lower DO_sat by ≈ 0.35 mg/L at 20 °C; water warming bounded by the site's air–water sensitivity) and a **train-cool-years → test-hot-years** extrapolation test. Uncertainty from ensembles + conformal.
- **Derived features exported to the platform/downstream:** GPP, ER, K600, NEP, DO deficit, thermal sensitivity (air–water slope), turbidity–discharge hysteresis class, 7-DADMax, hypoxia hours, salinity pulses, forecastability score.
- **Validation of latents.** Against independent metabolism estimates (Appling et al. 2018 Powell Center 356 rivers; NEON/StreamPULSE) at held-out sites; symbolic regression on learned GPP/ER/K600 functions.

### 7.2 Causal-discovery claim ladder
1. Simulation recovery: Bagged-PCMCI+, LPCMCI, J-PCMCI+ (tigramite) recover known links in data simulated from the Sim physics (declared synthetic, R9).
2. Positive/negative controls on real data: recover temperature→DO-saturation deficit, discharge→turbidity, discharge→SC dilution, light→GPP; reject impossible links (river variables → air temperature, anything → solar geometry).
3. Cross-site replication (stability across HUC regions, bootstrap frequency ≥ 0.8).
4. Agreement with natural experiments (storms, dam releases, heatwaves) and out-of-sample predictive gain from discovered parents.
Only links passing all four are reported as "causal evidence"; others are hypotheses. Design fixes vs v1's PCMCI+ run: multi-year data, explicit radiation and discharge variables, analysis of DO deficit rather than raw DO, lags spanning diel cycles.

### 7.3 Ecological interactions & thresholds
Temperature–DO–metabolism coupling; nutrient→chl→DO cascades (lakes); C–Q hysteresis typology; conductivity thresholds for macroinvertebrates (national replication of EPA's 300 µS/cm benchmark with TITAN-style change-point analysis on NRSA/UK EA); microbial co-occurrence networks only as hypothesis generators.

### 7.4 Interpretability suite (fidelity before interpretation)
Synthetic ground-truth attribution tests and weight-randomisation sanity checks → integrated gradients/attention only after passing; LSTM state probes against independent metabolism; emulator-based Sobol sensitivity over scenario forcings; modality Shapley values from SENTINEL-Fuse (exact over subsets).

### 7.5 Pre-specified insight studies
- **E1** Continental, sensor-based decomposition of warming-driven oxygen loss into solubility vs metabolism components, day vs night (extends Zhi et al. 2023, which used daytime grab samples).
- **E2** How far and how fast signals propagate between upstream and downstream sensors, and the early-warning lead time this buys (checked against river travel time from NHDPlus velocities).
- **E3** Whether groundwater-fed (thermally buffered) sites resist oxygen loss and hypoxia (Hare et al. 2021 classes).
- Secondary: turbidity–discharge hysteresis map; "which sites are early-warnable" (forecastability map); salinity–community thresholds.
Novelty is provisional until the pre-writing literature re-search (docs/research/06 Appendix B).

---

## 8. Goal 4 — Downstream applications (docs/research/07)

All use real labels (measurements or field observations as primary; advisories/listings only as secondary), forward-chaining by year + spatial blocks (HUC4/50 km or leave-one-state-out), block-bootstrap CIs, location-only and climatology rows, and ablations (predicted variables vs embeddings vs observed values). **Cross-fitting rule:** upstream predictions used as features in a downstream test fold must come from upstream models that never saw those sites/times. Decision-time rule: culture-based FIB results arrive 18–24 h later and are never same-day inputs.

| ID | Application | Labels (verified sources) | Incumbent to beat |
|---|---|---|---|
| D1 | **SafeSwim** — beach E. coli > 235 cfu/100 mL nowcast/forecast | WQP beach-program results (e.g. Ohio 53,069 results/96 sites), Chicago beach lab data, NY State Parks, King County, EU bathing water | USGS nowcast goals (≥ 80% correct and ≥ 5 pts over persistence); Chicago: city model AUROC 0.70, persistence 0.645, same-day-lab bar ≈ 0.74 |
| D2 | **ToxinWatch** — microcystin > 8 µg/L / HAB advisory | WQP (73,928 microcystin results), Washington toxic-algae DB (16,575), NLA; CA/NY/OR advisories secondary | EPA cyanoHAB forecast (90% accuracy, 49% precision on 2021 holdout) |
| D3 | **KillAlert** — fish-kill risk | Iowa DNR fish-kill API (1,035 geocoded events since 1981, natural vs human cause); case-crossover design | No published fish-kill model with AUC found (open niche) |
| D4 | **TapRisk** — drinking-water nitrate / DBP-precursor risk | SDWIS violations; UCMR raw-water TOC & bromide (3,444 systems); EPA service-area boundaries | Pennino et al. 2020 (90% of catchments correct) |
| D5 | **BioIntegrity** — NRSA/NLA condition class | NRSA/NLA condition labels (≈ 2,000 visits/cycle) | Hill et al. 2017 RF (75%) |
| D6 | **MusselGuard** — zebra/quagga mussel invasion forecast | USGS NAS records; Minnesota infested-waters list (rolling forecast test) | Minnesota risk model AUC 0.78 |
| D7 | **TroutRefuge** — cold-water fish occupancy & thermal stress | Fish-surveyed sites only (NRSA fish, state surveys); predicted 7-DADMax from M1/M6/Sim | Thermal-threshold rules; RF occupancy |
| D8 | **RiverPulse** — stream metabolism (GPP/ER) at unsensed reaches | Appling et al. 2018 (356 rivers), StreamPULSE | GBM on attributes; Appling regime classes |
| D9* | Lake bottom-water hypoxia (NLA DO profiles); 303(d) impairment screening (ATTAINS layers); avian botulism (USGS wildlife mortality) | stretch | — |

Disease framing (judges): D1 (recreational illness risk), D2 (cyanotoxin exposure), D4 (drinking-water risk); CDC WBDOSS/NORS and OHHABS are state-month/state-year only → validation context, not training labels.

---

## 9. Goal 5 — SENTINEL Console (usable platform)

- **Backend:** FastAPI service (`sentinel/platform/api/`): `/sites`, `/forecast/{site}`, `/alerts`, `/simulate` (scenario engine), `/causal/{site}`, `/downstream/{app}`, `/predict/{modality}` (upload CSV sonde series, satellite chip/GeoTIFF, drone RGBN image, 16S table, SMILES/CAS list, taxa table), `/models` (model cards with registry-backed metrics), `/provenance/{id}`.
- **Frontend:** web app (Streamlit first for speed; React/Vite rebuild of v1 demo optional) with map (pydeck/folium), site pages (observations, forecast fans, event probabilities, frozen thresholds), alert feed (EW-v2 calibrated), scenario simulator (sliders constrained to validated perturbation ranges), causal-graph explorer (only ladder-passing links highlighted), downstream dashboards, "bring your own data" pages, honest-scope banners.
- **Live pipeline:** daily cron pulls USGS (API key) + weather → forecasts → EFI-standard files → public log with hashes (feeds the live trial, §11).
- **Quality:** every number shown comes from the registry or a live model call with provenance; demo/sample data clearly labelled; tests (API contract tests, UI smoke tests); Dockerfile + docker-compose; user guide.

---

## 10. Goal 6 — SENTINEL-Lite: distilled RGB+NIR drone model (docs/research/08)

- **What RGB+NIR can and cannot see (scope rule).** *Good:* Secchi depth/clarity, turbidity/TSS, water mask, detection of surface scum/floating algae, sediment plumes and discoloured water. *Moderate:* oil-sheen presence (glint-geometry dependent). *Weak:* chl-a concentration across sites → reported as **trophic classes**, not concentrations. *Not possible:* subsurface phycocyanin, CDOM, and all non-optical variables (pH, DO, conductivity, TN/TP, microcystin) — these leave the drone track entirely (v1 `download_drone_s2_pairs.py` targets DO/pH/Temp/SpCond and is retired).
- **Design input (research pilot, not a result):** on GLORIA with whole datasets held out, RGB+NIR vs with red-edge gave MdSA chl-a 71–91% vs 48–69%, Secchi 37–44% (≤ 4 points worse), TSS 77–84% (3–7 points worse), CDOM ≥ 85% (unusable); random 5-fold looked 1.5–2× better than held-out datasets; a 5% per-band calibration error cost as much as losing red-edge (noise augmentation recovers part).
- **Sensor spec.** RGB camera + NoIR camera with ≥ 780–800 nm long-pass, hardware-synchronised; RAW capture with fixed exposure/gains (not MJPEG + auto white balance); reflectance panel each flight; sky frames (40° off-nadir, 135° from sun) and NIR-based glint correction; measured spectral response of both cameras. These are changes requested of the `camera-model` branch (credited contributor).
- **Training data (real, public).** No public drone dataset pairs RGB+NIR imagery with in-situ WQ, so: **NAIP 4-band** (0.3–1 m; 8-bit, not reflectance-calibrated — biggest risk; per-scene normalisation + radiometric augmentation) matched to WQP/USGS in-situ within ±1 day (Minnesota alone: 1,105 Secchi and 424 chl-a station-days, unfiltered upper bound; a national NAIP–in-situ matchup set appears not to exist yet); Sentinel-2 B2/B3/B4/B8 chips (teacher domain); GLORIA (CC BY 4.0) for physics pretraining with camera-band simulation; NEON AOP (token) as a drone-scale external test. Hazard labels: MARIDA (sediment/foam), MADOS (oil, satellite), LADOS/De Kerf (drone RGB oil), FloodNet/Chesapeake (water masks).
- **Teacher → student.** Teacher = M2 SENTINEL-Orbit (Sentinel-2 with red-edge, distributional outputs, trained on GLORIA + matchups). Student = 4-band network trained on NAIP with S2 scenes within ±1 day: block-averaged, uncertainty-weighted distillation loss + direct in-situ labels; spectral reconstruction (RGB→red-edge) only as an auxiliary task (cannot recover missing information).
- **Student architecture & budget.** MobileNetV4-Conv-S / EfficientNet-Lite0 / RepViT backbones with light decoder (per-pixel maps + scalar heads); ≤ 3M parameters, ≤ 2 GMACs per 512×512 tile, ≤ 4 MB INT8, ≤ 100 ms per tile on device. Primary device: Raspberry Pi 5 + AI HAT+ (Hailo-8L); alternative Jetson Orin Nano Super; Coral avoided. Desktop proxy (ONNX Runtime, 1 thread, INT8, after `quant_pre_process`) measured during research: MobileNetV4-Conv-S 19 ms, EfficientNet-Lite0 42 ms per tile; MobileNetV3/EfficientViT slowed under INT8. Only on-device numbers are reported as device latency.
- **Evaluation.** Headline = student vs in-situ on held-out lakes/water bodies (never student vs teacher). Comparisons: teacher (S2 with red-edge) on the same lakes, RGB+NIR band-ratio/index baselines, GBM on RGBN statistics, student without distillation; robustness to glint, haze, resolution (0.1–10 m), band misregistration, exposure and ±5% calibration error.
- **Integration.** ROS2 node + on-board alerting (scum/plume detection) aligned with the `camera-model` branch; the dual-camera payload remains *proposed hardware* until flown and validated.
- **Honest first-in-class claim (to be re-checked in Scopus/Scholar before writing).** *Not* "first drone/RGB water-quality model" (prior work exists: Su & Chou 2015, Cheng 2020, HydroColor, HydroVision, Belcore 2019). *Defensible:* the first openly released RGB+NIR student distilled from a red-edge satellite teacher using only public data, tested on held-out lakes against in-situ measurements, reporting the red-edge gap, the distillation gain and on-device INT8 latency.

---

## 11. Goal 7 — Additional contributions

1. **Pre-registered live prospective trial** (strongest fix for v1's credibility gap): freeze models/thresholds by ≈ 2026-11-15; OSF registration; daily forecasts published with hashes; external scoring via NEON, VERA and usgsrc4cast challenges; winter season (Dec 2026–Mar 2027: salinisation, turbidity) and summer season (Apr–Sep 2027: hypoxia, nitrate, blooms, beaches); one pre-specified analysis date per season (§12.6).
2. **SENTINEL-Bench release:** frozen manifests, splits, scoring code, baselines and model cards for Sonde, Orbit, Microbe, ChemTox, BioAssess, HydroMet, Lite, Fuse and EW-v2 tasks.
3. **Calibrated uncertainty everywhere:** conformal (Mondrian by region/pattern) + coverage audits under shift.
4. **Monitoring-network design:** where would one new sensor most reduce forecast/EW uncertainty (value of information from SENTINEL-Fuse and the forecastability map)?
5. **Equity analysis:** model skill and monitoring coverage stratified by CDC SVI 2022; report gaps.
6. **Climate-scenario exploration** via SENTINEL-Sim (delta-change air temperature), only within validated perturbation ranges and labelled as scenario analysis.
7. **Documentation:** model cards, datasheets, license register, reproducibility scripts (`make`-style pipeline), CI tests.
8. **Honest LaTeX paper** (+ SJWP-format version ≤ 20 pages, abstract ≤ 180 words), verified bibliography.

---

## 12. Fixing the early-warning claims — EW-v2 protocol (docs/research/09)

### 12.1 What we will say about v1
"The v1 label-free detector raised a pre-onset alarm before random clean dates as often as before real events; we therefore report no validated discrimination or lead time for v1." The v1 detector is re-scored under EW-v2 as a baseline.

### 12.2 Hazards (criterion-anchored excursions, pre-registered exactly)
H1 hypoxia: DO < 3.0 mg/L (tiers < 4.0, < 2.0; cold-water < 4.0/5.0) for ≥ 2 h, onset after ≥ 72 h at ≥ threshold + 1 mg/L with coverage rules and QA exclusions. H2 salinisation: site-regressed chloride (≥ 20 WQP pairs, R² ≥ 0.9) 4-day mean > 230 mg/L or 1-h > 860 mg/L. H3 turbidity excursion (secondary). H4 nitrate > 10 mg/L N near drinking-water intakes. H5 lake cyanoHAB (CyAN lake-week chl-a ≥ 12 µg/L with cyano dominance); H5b in-situ chl-a > 20 µg/L (VERA); H5c microcystin > 8 µg/L (sample-based). H6 beach E. coli > 235 cfu/100 mL (secondary). H7 flood stage (context). Events are labelled "criterion-anchored excursions", not regulatory violations.

### 12.3 Forecast tasks
Daily issue (06:00 local) of P(onset within 1, 3, 7 d) for H1–H4 (H5 weekly, H6 same-day); inputs only with arrival ≤ issue time; weather forecasts only as issued; site eligible if ≥ 80% of previous 72 h present.

### 12.4 Baselines
Site × DOY climatology (harmonic logistic), persistence/current-state, threshold-on-current-value, logistic regression on lags, GBM, SENTINEL models, v1 CUSUM re-scored, operational forecasts where available (EPA cyanoHAB, NOAA Lake Erie, NWS).

### 12.5 Splits, calibration, freezing
Train ≤ WY2021; calibrate/validate WY2022–2023; **test once on WY2024–2025** (Approved data); ≈ 20% of sites by HUC4 fully held out. Thresholds set on calibration at ≤ 1 false-alarm episode per site per 30 days (sensitivity: 2/30 d), frozen and hashed before any test data are loaded (loader refuses test years without the freeze file).

### 12.6 Endpoints, controls, power
- **Primary:** BSS vs climatology at h = 3 d per hazard (H1, H2, H4; H5 for summer), pooled test site-days; success = lower 95% site-month block-bootstrap bound > 0, Holm-adjusted.
- **Secondary:** ΔBS vs persistence/threshold/GBM (paired; Diebold–Mariano); POD (≥ 24 h warning) at the frozen budget with realised false-alarm rate per site-month; **POD_real − POD_placebo** with permutation p; Molchan area skill; reliability; relative economic value.
- **Controls (all mandatory):** season-matched placebo onsets (K = 20 per onset), 1,000 circular-shift surrogates (≥ 30 d), random alarms at matched rate, negative-control hazard at clean sites, sensor-fault audit, label permutation, station-swap, covariate shuffle, input-truncation audit.
- **Power:** ≥ 50 test onsets per hazard (≥ 30 per prospective season) before any claim; otherwise "underpowered". Count calibration-period onsets before registration.
- Implementation: `sentinel/eval/early_warning.py` (event extraction with censoring, onset labels, threshold calibration, event-level detection with strictly-pre-onset alarms, season-matched placebo sampling, Fisher test) — unit-tested; to be extended with alarms-per-site-month calibration, circular-shift surrogates and Molchan diagrams.

---

## 13. Dataset strategy

### 13.1 Sources (access verified 2026-10-22 unless noted)

| Source | Use | Access | Size budget |
|---|---|---|---|
| USGS NWIS daily values (legacy WaterServices; decommissioned 2027-02-22, slowdowns from 2026-11-16) | M1/M6/EW daily tier, all WQ sites | No key; **downloading now** | ~2–5 GB |
| USGS Water Data OGC API `continuous` | 15-min core sites; live trial | Anonymous 1,000 req/h; **API key requested** | ~10 GB parquet |
| NEON DP1.20288/20033/20093/20282/20141/20281/20120/20163/20107/20219, AOP | M1 external, M3, M5, Fuse Tier D, M7 | **Token required since 2026-06-30** (CC BY 4.0); EFI targets token-free | ~50 GB (stream-aggregated) |
| Water Quality Portal (WQX 3.0 profiles) | discrete chemistry, chl-a/TSS/Secchi, microcystin, E. coli, chloride–SC pairs | No key | ~10 GB |
| Sentinel-2 L2A, Landsat C2 L2, NAIP (Planetary Computer / Earth Search STAC) | M2, M7 chips | No key (PC SAS signing) | ~60–120 GB chips |
| GLORIA, AquaSat, CyAN, NLA 2022 | M2 matchups; Tier C | Public (PANGAEA/figshare/EPA) | ~5 GB |
| Open-Meteo (ERA5 history + Previous Runs NWP), gridMET, Daymet | forcings, operational forecasts | No key (rate-limited) | ~5 GB |
| NLDI (`api.water.usgs.gov/nldi`), StreamCat/LakeCat, NHDPlus V2 | site linking, attributes, flow connectivity | No key | ~10 GB |
| ADORE, ECOTOX, CompTox/DSSTox, USCG NRC incident files | M4 + spill severity | Public | ~3 GB |
| EPA NRSA/NLA, UK EA ecology archive | M5, D5 | Public (OGL v3.0 for EA) | ~3 GB |
| NMDC (NEON metagenome products), EMP | M3 ARG labels, pretraining | Public | ~20 GB |
| Event catalogs: Iowa DNR fish kills, Chicago/Ohio beach data, WA toxic algae, CA FHAB, NY DEC, NRC, CyAN, NOAA Lake Erie | D1–D3, EW-v2 context | Public | <1 GB |
| v1 reusable assets (SENTINEL-HF): microcystin benchmark (3,649 rows), raw WQP toxin/pigment tables, E. coli results, event catalogs, site lists, NHDPlus site→COMID map, EMP taxonomy matrices | Tier C, D1/D2, bootstrap lists | Local (read-only) | — |

### 13.2 Pipeline and QC
Raw → interim → processed, each with a manifest. Ingest masks missing codes (−999999, non-numeric qualifiers), keeps qualifiers/approval status/last-modified, splices series only after overlap checks, enforces unit identity, applies plausible-range/flatline/spike screens (PyHydroQC-style), excludes service windows. Normalisation statistics computed on training blocks only and stored with the experiment.

### 13.3 Storage budget
Target ≤ 200 GB under `SENTINEL_DATA` (free: ~355 GB), chips stored as compressed uint16/uint8 NPZ/zarr; raw text never retained after parsing.

### 13.4 Credentials needed from the user (free; only the user can create them)
1. **USGS Water Data API key** (api.waterdata.usgs.gov) → env `API_USGS_PAT`. Without it: anonymous 1,000 requests/h (bulk 15-min pull ≈ 2–3 days; live trial fragile).
2. **NEON API token** (data.neonscience.org account) → env `NEON_TOKEN`. Without it: M3 and NEON parts of M1/M5/Fuse fall back to EFI targets/NMDC only.
Keys must be stored locally by the user (e.g. `~/.config/sentinel/keys.env`, chmod 600), never committed.

---

## 14. Common experimental design

### 14.1 Time blocks (sensor-derived tasks; water years)
Train WY2008–2021 · Validation/Calibration WY2022–2023 · Test-A WY2024–2025 · Test-B WY2026 · Test-C live (from Nov 2026). Embargo 42 d at block edges. Survey/sample tasks use their own documented blocks (e.g. NRSA cycle-out; ADORE fixed splits).

### 14.2 Spatial blocks
HUC4 groups with flow-connected stations together (NLDI), 20% HUC4s fully held out; lakes held out as wholes; 5-km/20-km clusters and leave-region-out for cyanotoxins; leave-one-ecoregion-out for bioassessment; leave-one-NEON-domain-out for microbes. Distance reports (nearest train→test km) saved with each split.

### 14.3 Baseline ladder (every table)
Trivial-context (DOY + location + site mean) → persistence/climatology → classical/physics references → GBM/RF/TabPFN → strongest published method → SENTINEL model(s).

### 14.4 Statistics
Cluster bootstrap (B = 2,000) by the stated unit; paired differences; Diebold–Mariano (HAC) for forecasts; Holm across primary endpoints; Clopper–Pearson for event counts < 30 ("preliminary"); ≥ 3 seeds (5 for Fuse) with seed variance in CIs; configurations-tried counts.

### 14.5 Ablations
M1: covariates (none / past weather / operational NWP / oracle), metabolism head on/off, foundation pretraining on/off, context length. M2: spectral vs spatial branch, foundation encoder vs from scratch, matchup window. M3: sequence embedding vs CLR table, pretraining on EMP. M4: foundation init, species embedding, conditioning head. M5: taxon embeddings vs one-hot, taxonomic resolution. Fuse: leave-one-modality-out, exact Shapley over subsets, shuffled-modality controls, dropout schedule. Lite: teacher vs no teacher, resolution, band subsets (RGB-only vs RGBN). EW: lead window, threshold budget.

---

## 15. Evaluation criteria & definition of success

### 15.1 Claim labels (applies to G1, G2, G6)
- **"State of the art"** only if (i) the comparison includes the strongest published method for the task (re-run on our split when code exists, or on the published benchmark split) **and** the strongest simple baseline, (ii) the paired block-bootstrap 95% CI of the primary-metric improvement excludes 0, (iii) the gain holds in ≥ 3 seeds, and (iv) the test set was evaluated once after freezing.
- **"First-in-class"** only if a dated literature search (documented in `docs/novelty_checks/`) finds no prior method for the task under comparable evaluation **and** the model beats every applicable baseline with CI excluding 0.
- Otherwise: "competitive" (CI overlaps best baseline) or "negative" (worse) — reported as such.

### 15.2 Hard thresholds (all must hold for the project to be called complete)
- **H-G1:** ≥ 5 of M1–M7 earn "SOTA" or "first-in-class" on their pre-registered primary endpoint; each of these five also predicts at least one disaster-relevant target with skill (BSS > 0 or AUPRC > prevalence with CI excluding chance).
  - M1 primary: median CRPS skill (unseen × Test-A) vs best of {damped anomaly, climatology, persistence, GBM, zero-shot Chronos-2, fine-tuned PatchTST/LSTM} > 0 with CI excluding 0 for ≥ 3 of {turbidity, SC, nitrate, DO, temperature} at 1–7 d **and** N-b virtual-sensing KGE improvement over GBM for ≥ 2 variables.
  - M2 primary: on unseen lakes × 2024–2025, chl-a MdSA lower than (a) MDN retrained on our matchups and (b) GBM on imagery + season/location, with lake-bootstrap paired CIs excluding 0; also lower than the location-only kNN baseline; secondary targets TSS/Secchi/turbidity reported with the same comparators.
  - M3 primary: leave-one-domain-out skill above both DOY+lat+site-mean and RF-on-CLR for ≥ 2 targets (CI excluding 0), or first-in-class cross-primer transfer that beats the trivial baseline.
  - M4 primary: micro-RMSE on ADORE `t-F2F` `split_occurrence` test below 0.927 with chemical-bootstrap CI excluding the published value, reproduced baseline within ±0.03 of 0.927.
  - M5 primary: leave-one-ecoregion-out R² above the blocked RF for ≥ 3 of {TP, TN, conductivity, chloride, turbidity, pH} with CI excluding 0.
  - M6 primary: unseen-HUC4 skill above GBM and physics-hybrid baselines for ≥ 2 of 5 targets (CI excluding 0).
  - M7 primary: see §15.2 H-G6.
- **H-G2:** P1 or P2 passes (fusion beats both best single modality and GBM-concatenated under spatial blocking, CI excluding 0), and P3 holds for ≥ 5 modalities on NEON (or, if NEON access is unavailable, on Tier C/W with ≥ 5 modalities).
- **H-G3:** SENTINEL-Sim hindcast beats persistence and climatology at ≥ 3-day horizons on held-out sites for DO and temperature (CI excluding 0) and passes all pre-registered physics checks; metabolism latents correlate with independent estimates at held-out sites (Spearman ρ ≥ 0.5 for GPP or ER); ≥ 1 insight study (E1–E3) completed with its pre-specified validation.
- **H-G4:** ≥ 6 of D1–D8 implemented with real labels and blocked evaluation; ≥ 4 beat their incumbent (§8) with CI excluding 0.
- **H-G5:** Console runs end-to-end (API + UI) on real data with tests passing; all displayed metrics resolve to registry IDs.
- **H-G6:** M7 INT8 model ≤ 4 MB and ≤ 3M parameters with measured latency ≤ 100 ms per 512×512 RGBN tile (on-device if hardware is available; otherwise the single-thread ONNX Runtime proxy, labelled as proxy); on held-out lakes vs in-situ it beats RGB+NIR band-ratio and GBM-on-RGBN baselines for Secchi and turbidity/TSS (CI excluding 0), distillation improves over the no-teacher student, and the red-edge gap vs the teacher is reported.
- **EW-v2:** primary BSS endpoint passes for ≥ 2 hazards on Test-A with all controls reported; live trial registered and running.

### 15.3 Soft thresholds (should meet most)
M1 CRPSS > 0 vs zero-shot Timestral/Chronos-2 on Test-B; external challenge scores above climatology; M3 ARG prediction above trivial baseline; M4 first results on `a-CA2F-diff`; M5 UK↔US transfer above geography-only; Fuse P4/P5; Sim counterfactual train-cool/test-hot test passed; D3 KillAlert AUROC > 0.7 under case-crossover; Lite robustness degradation ≤ 20% under moderate glint/haze and ±5% calibration error; Lite trophic-class (chl-a) accuracy above the band-ratio baseline; scum/plume detection AUPRC above an RGBN-index baseline; paper ≤ 20 pages SJWP format.

### 15.4 Failure criteria (trigger deep diagnosis)
Any SENTINEL model below its trivial-context baseline after 3 configurations; any leakage test failure; placebo detection rate within 1.5× of real detection for a hazard; fusion below best single modality on paired rows; Sim failing physics checks; inability to reproduce a published baseline within tolerance (stop and debug before proceeding).

### 15.5 Iteration budget
Up to 3 improvement cycles per workstream (diagnose → research → one change at a time → re-run), ≤ 20 additional training runs per cycle per workstream (GPU-bounded); after 3 cycles report best honest result and re-frame the claim.

**Amendment 2026-10-24 (user direction: every model must be SOTA or first-in-class; keep improving until it is).** The 3-cycle cap is lifted: each workstream keeps iterating (diagnose → literature search for the current best method → targeted change → re-run on validation/CV) until it earns a §15.1 label or every reasonable avenue is documented as exhausted in RESULTS.md. Integrity guards that do **not** change:
1. Iteration uses validation/CV only. A test block is scored once per frozen configuration; a test block already scored (M4 ADORE) cannot be re-used for a claim — an improved model needs a **new, pre-registered, previously unseen test set** (e.g. ECOTOX records/chemicals added after the ADORE cut-off), and the first test result stays reported.
2. Every iteration, including failed ones, is logged (TRAINING_LOG.md, registry); the number of configurations tried is reported next to the final claim, and Holm/selection effects are stated.
3. "First-in-class" requires a dated novelty search in `docs/novelty_checks/` and still requires beating every applicable baseline with CI excluding 0 — redefining a task only to escape a strong baseline is not allowed.
4. Baselines are never weakened, removed or under-tuned to make a model win (tuning parity as in each protocol).

**Amendment 2026-11-08 (user direction: if a task is saturated, enlarge the dataset / make the task harder and keep pushing).** A task counts as *saturated* when the strongest simple baseline is already near the ceiling (e.g. KGE ≥ 0.9, R² ≥ 0.9, AUROC ≥ 0.95) or when all strong methods tie within their CIs because the test is too small or too easy. Saturated tasks get a **harder / larger benchmark track** in addition to (never instead of) the original one:
5. Harder tracks are defined from information available before any result on them is seen: larger/national data (more sites, regions, years), stricter extrapolation (unseen regions, unseen years, ungauged/no-discharge, cross-sensor/cross-taxa), longer horizons, and **difficulty strata defined a priori** (e.g. extremes above the training-climatology 90th/95th percentile, heatwave/storm periods, event windows) — never strata chosen because the model happens to win there.
6. Each new track gets its own protocol section, frozen split (sha256) and power check before model selection on it; the original benchmark and its results are always reported alongside, and claims name the track they come from.
7. Data enlargement follows the data rules (manifests, datasheets, licences, QC) and the shared-disk budget; additions to a training set never include rows of any frozen test block.

---

## 16. Implementation roadmap (milestones, gates, GPU allocation)

GPU budget: 5 of 10 GPUs (IDs 0–4), enforced by `sentinel.core.gpu` locks. Default allocation — GPU0: M1/M6/Sim; GPU1: M2/M7; GPU2: M3/M4/M5; GPU3: Fuse; GPU4: shared (baselines, TSFM zero-shot, seeds).

| Milestone | Content | Gate / verification |
|---|---|---|
| **MS0 Infrastructure** | Reorganised repo (§20); `sentinel.core` (paths, registry, manifests, seeds, GPU locks, run logger) and `sentinel.eval` (metrics, splits, bootstrap, early_warning) with tests; env lock | Unit tests pass; registry/run-log round-trip |
| **MS1 Data foundation** | USGS DV (all WQ sites) + 15-min core; NEON (token) / EFI targets; WQP WQX3; StreamCat/NLDI; forcings + archived NWP; S2/Landsat/NAIP chip pipeline + matchups; ADORE; NRSA/NLA/UK EA; NMDC/EMP; event catalogs; v1 asset re-extraction | Manifests; QC reports; leakage-safe split files frozen (`docs/preregistration/splits_*.json` hashed) |
| **MS2 Baselines** | Baseline ladder for every task (incl. published-method reproductions: MDN, ADORE RF/XGB, blocked RF bioassessment, Chicago beach) | Baselines reproduce published values within tolerance |
| **Gate R-pre** | Pre-training review per workstream (`/review` skill) | No critical issues |
| **MS3 Modality models** | M1–M6 training + ablations; M8 optional | §15.2 per-model endpoints on validation; then single test scoring |
| **MS4 EW-v2 + pre-registration** | Event catalogs, calibration, freeze, OSF registration (≈ 2026-11-15), live pipeline | Freeze file + hashes; live forecasts publishing |
| **MS5 Fusion** | Stage 0–5 | P1–P5 |
| **MS6 Simulation & insights** | Sim, causal ladder, interpretability, E1–E3 | H-G3 |
| **MS7 Downstream** | D1–D8 | H-G4 |
| **MS8 Drone** | M7 distillation, quantisation, latency/robustness, ROS2 interface | H-G6 |
| **MS9 Platform** | Console | H-G5 |
| **Gate R-post** | Post-results review | No fabrication/leakage |
| **MS10 Paper & release** | Figures from registry, LaTeX paper (SJWP + full), bibliography triple-pass, SENTINEL-Bench release, model cards | Compiles; verified bib |

---

## 17. Risk register

| Risk | L | I | Mitigation |
|---|---|---|---|
| Deep models do not beat naive/GBM baselines on many tasks | H | M | Headline on tasks where baselines are weak (event-driven variables, unseen sites, transfer); honest "competitive/negative" labels; ensembles with GBM allowed but attributed |
| Too few events per hazard | H | H | Pool sites/networks; tiered thresholds; exact CIs; "underpowered" label |
| Leakage (cause of v1 failure) | M | Fatal | §3 rules + automated tests + review gates + prospective Test-C |
| Data access changes (USGS legacy shutdown, NEON token, rate limits) | Certain | M | Download DV now; OGC API client; request keys; mirror raw pulls with manifests |
| NEON token unavailable | M | H (M3, Tier D) | NMDC + EFI targets; Tier C/W fusion; M3 scope reduced and stated |
| Sensor artefacts labelled as events | H | H | QA qualifiers, service windows, coherence checks, hand audit of 100 events |
| Satellite matchup quality (clouds, adjacency, baseline offsets) | H | M | SCL/Fmask masks, offset correction, exact overpass times, water-pixel checks |
| Compute (11 GB, no bf16) | Certain | L–M | fp32/fp16 AMP, ≤ 300M-param fine-tunes, LoRA, gradient checkpointing |
| "First" claims overturned | M | M | Dated novelty searches before writing; "to our knowledge as of" phrasing |
| Licensing of models/data for public platform | M | M | License register; non-commercial models as baselines only |
| Over-scoping (8 models + fusion + sim + 8 apps + platform + drone) | H | M | Milestone gates; M8 and D9 optional; shared infrastructure; parallel workstreams |

---

## 18. Timeline (indicative; workstreams run in parallel)

| Phase | Duration | Dependencies |
|---|---|---|
| MS0 Infrastructure + reorganisation | 1–2 days | — |
| MS1 Data foundation | 1–3 weeks (USGS/NEON pulls rate-limited) | MS0; credentials |
| MS2 Baselines | 1 week (overlaps MS1) | MS1 per task |
| MS3 Modality models (+ iteration cycles) | 3–6 weeks | MS2 + review gate |
| MS4 EW-v2 freeze + live trial start | by ≈ 2026-11-15, then continuous to Oct 2027 | MS1 (sensor), MS3 (M1 event head) |
| MS5 Fusion | 2–4 weeks | MS3 encoders |
| MS6 Simulation & insights | 2–4 weeks | MS1, M1 metabolism head |
| MS7 Downstream | 2–3 weeks | MS3/MS5 cross-fitted predictions |
| MS8 Drone | 2–3 weeks | M2 teacher |
| MS9 Platform | 2 weeks (incremental) | models |
| Post-results review, paper, bibliography | 1–2 weeks | all |

---

## 19. Deliverables

- [ ] Reorganised, tested codebase (`sentinel/`, `scripts/`, `tests/`, `configs/`) with legacy v1 archived
- [ ] Data manifests + frozen split files + pre-registration documents
- [ ] Trained models M1–M7 (+ M8 optional), SENTINEL-Fuse, SENTINEL-Sim, D1–D8, with model cards
- [ ] `RESULTS.md` (rendered from registry) and `TRAINING_LOG.md` (automatic)
- [ ] `REVIEW_REPORT_PRE_TRAINING.md`, `REVIEW_REPORT_POST_RESULTS.md`
- [ ] SENTINEL Console (API + web app + Docker)
- [ ] Live prospective trial running + registration
- [ ] SENTINEL-Bench release package
- [ ] Publication-quality figures (PNG + PDF) generated from the registry
- [ ] LaTeX paper (full + SJWP format) with compiled PDF; verified `references.bib`; `BIBLIOGRAPHY_VERIFICATION.md`

---

## 20. Codebase reorganisation plan

Goals: separate retracted v1 material from v2, make v2 importable and testable, keep provenance.

```
SENTINEL-STOCKHOLM/  (branch v2-rebuild)
├── LICENSE  README.md  RESEARCH_PLAN.md  LANDSCAPE_SURVEY.md  RESULTS.md  TRAINING_LOG.md
├── pyproject.toml  requirements-v2.txt  environment-v2.yml
├── configs/                     # YAML per workstream / experiment
├── sentinel/                    # v2 package
│   ├── core/                    # paths, registry, manifest, seed, gpu, runlog, config
│   ├── eval/                    # metrics, splits, bootstrap, early_warning, calibration/conformal
│   ├── data/sources/            # usgs (legacy + OGC), neon, wqp, stac (S2/Landsat/NAIP), weather, nldi/streamcat,
│   │                            #   adore/ecotox/nrc, nrsa/nla/uk_ea, nmdc/emp, events
│   ├── data/                    # QC, matchups, datasets, feature builders
│   ├── models/                  # sonde/ orbit/ microbe/ chemtox/ bioassess/ hydromet/ lite/ fuse/ baselines/
│   ├── sim/  causal/  interpret/  insights/
│   ├── downstream/              # safeswim/ toxinwatch/ killalert/ taprisk/ biointegrity/ musselguard/ troutrefuge/ riverpulse/
│   └── platform/                # api/ app/ live/
├── scripts/                     # thin CLIs: data/, train/, eval/, figures/, live/
├── tests/                       # unit + leakage + contract tests
├── docs/                        # research/, preregistration/, model_cards/, datasheets/, novelty_checks/, licenses.md
├── results/                     # registry.jsonl + small JSON summaries (large outputs in SENTINEL_RUNS)
├── figures/  paper/  logs/ (gitignored)
└── legacy/                      # v1, archived for provenance — NOT maintained, results retracted
    ├── README.md                # what this is; pointer to REVIEW_REPORT.md
    ├── sentinel/  scripts/  tests/  configs/  results/  demo/  paper_v1/ (Stockholm Paper Final.*)
```

Large data in `SENTINEL_DATA` (`~/STOCKHOLM/sentinel_data`), runs/checkpoints in `SENTINEL_RUNS` (`~/STOCKHOLM/sentinel_runs`), v1 HF mirror read-only (`SENTINEL_LEGACY`). Reusable v1 code (clean downloaders and protocols listed in docs/research/10) is *ported* into v2 modules with tests, not imported from `legacy/`.

---

## 21. Disclosure and authorship (SJWP requirement)

SJWP rules require accurate disclosure of help received. v2 work is substantially AI-assisted (Claude, Anthropic: research synthesis, code, analysis, drafting) and runs on a shared multi-GPU server. Commits made in this effort carry `Co-Authored-By: Claude` trailers; the acknowledgements in any submission must state the AI assistance, the compute actually used (RTX 2080 Ti server; v1 logs also show A100 use), and contributors (incl. the ROS2/camera contributor). The students should be able to explain and defend every component; `docs/` will include walkthroughs for that purpose.

---

## Appendix A — Research inputs
`docs/research/01_sensor_timeseries.md` · `02_satellite_remote_sensing.md` · `03_microbial_molecular.md` · `04_ecotox_bioassessment.md` · `05_multimodal_fusion.md` · `06_simulation_causal_interp.md` · `07_downstream_applications.md` · `08_drone_rgb_nir.md` · `09_events_early_warning.md` · `10_local_inventory.md`

## Appendix B — v1 → v2 name map
AquaSSM → M1 SENTINEL-Sonde (new model; v1 results retracted) · HydroViT → M2 SENTINEL-Orbit · MicroBiomeNet → M3 SENTINEL-Microbe · ToxiGene → dropped as headline (M4 ChemTox replaces the "molecular" slot) · BioMotion → withdrawn (no behavioural data); M5 BioAssess (+ optional M8 ZebraMotion) · Perceiver fusion → SENTINEL-Fuse · Digital twin → SENTINEL-Sim · SENTINEL-Lite/HydroDenseNet → M7 SENTINEL-Lite (new) · Track-A detector → EW-v2 baseline only.
