# 02 — Satellite / Remote-Sensing Modality: state of the art, data access, and a leakage-free plan for SENTINEL

*Research date: 2026-10-22. Author: Claude (research agent) for the SENTINEL team (A. Jin, B. Cheng).*
*Scope: inland and coastal water quality (WQ) from Sentinel-2 MSI, Landsat 8/9 OLI/TIRS, Sentinel-3 OLCI (PACE OCI noted). Covers chlorophyll-a (Chl-a), TSS, turbidity, Secchi depth, CDOM/a<sub>CDOM</sub>(440), phycocyanin/cyanobacteria, surface temperature, and WQ-disaster detection.*

**How this was verified.** Every number in the tables below was read from the paper text (PDF or publisher HTML), a Crossref or Semantic Scholar record looked up by DOI, an arXiv record, or a live API call made during this session. Papers whose full text I could not open are marked *(abstract only)*. Anything I could not confirm is marked **unverified**. All STAC endpoints and chip reads in §b were run against the live services on 2026-10-22. The scripts are in `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/scripts/`. Downloaded samples (≈0.4 GB cumulative across this survey and its two sub-surveys, within the 500 MB budget; no imagery tiles) are in `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/research_samples/satellite/`. The session's web-search budget ran out partway through. After that point, sources were reached only by fetching known URLs and looking up DOIs or arXiv IDs.

*Companion reports in this folder (not reviewed by this agent) cover the drone RGB+NIR student (08), multimodal fusion (05) and events/early warning (09). This report covers only the satellite side and the teacher outputs it should provide.*

---

## 0. Key findings

1. **Reported in-situ accuracy hides the real difficulty.** The Mixture Density Network (MDN) reaches a median symmetric accuracy (MdSA) of about 26–30% for Chl-a on *in-situ, simulated-band* spectra under random splits. Under leave-one-dataset-out this rises to about 46–48%. Applied to *real atmospherically corrected Sentinel-2 imagery*, the same MDN reaches MdSA ≈ **195%** for Chl-a and 110% for TSS (ACOLITE, N = 3,053 matchups; Saranathan et al. 2024). Pahlevan et al. (2022) write that on ACOLITE MSI matchups "neither MDN nor OC3 is viable for generating Chla products from MSI".
2. **The open target is the satellite-matchup task.** That means satellite pixels → in-situ WQ, on held-out lakes and held-out years. No public, leakage-free Sentinel-2 benchmark of this kind exists. AquaCorr-Global (Aug 2026, below) supplies matchups but no splits, baselines or leaderboard. Published end-to-end models use random or partly random splits (e.g., Sherf et al. 2025; Topp et al. 2021). A defensible SOTA or first-in-class claim is realistic here. It is not realistic on GLORIA random splits.
3. **Location and season baselines are the main threat.** In the Tick Tick Bloom HAB challenge, the winner used a KNN on location for the West and South regions (private region-averaged RMSE 0.7608 in severity classes). In Topp et al. (2021), static lake and landscape variables carried much of the Secchi skill: optical-only R² was 0.50. On SENTINEL's own measurement (Minnesota 2023 WQP sample), an oracle site mean explains **74%** of log10 Chl-a variance, and the within-lake SD (0.28 log10) is smaller than typical satellite RMSLE (0.5–1.0). A **location-only KNN** reaches MdSA **86–109%** (RMSLE 0.45–0.50) on held-out sites and spatial blocks. That is already better than in-situ-trained MDN on ACOLITE imagery (~195%), so this baseline is mandatory. Every SENTINEL claim must beat metadata-only models on held-out lakes. It must also show *within-lake temporal* skill.
4. **Imagery is free and account-free**, apart from OLCI L1 and the CLMS products (which need a free CDSE account). Use Earth Search `sentinel-2-c1-l2a`, which has a consistent processing baseline. Use Planetary Computer for Landsat C2 L2, OLCI L2 WFR, HLS v2 and JRC GSW. Planetary Computer's `sentinel-2-l2a` mixes processing baselines (02.12 → 05.11), and the Element 84 legacy `sentinel-2-l2a` metadata invites double-applying the −1000 offset. Both traps were verified with real pixels.
5. **Matchups are plentiful in the US, and a ready-made open matchup set now exists.** **AquaCorr-Global** (Zenodo, 2026-08-14, CC-BY-4.0; NASA GSFC authors) pairs in-situ data with **ACOLITE-corrected Rrs**. It holds 1,984,387 same-day matchups (935,740 pass QC). The Sentinel-2 QC-passing set includes 113,720 Chl-a (23,370 strict lab-extracted, at 5,655 sites), 121,281 TSS, 79,368 Secchi and 92,007 turbidity matchups. My own Minnesota 2023 check gave a 14.3% same-day valid S2 yield, consistent with AquaCorr's 11.2% for WQP Chl-a. Phycocyanin, CDOM and microcystin are 10–1000× scarcer. HAB cell-count labels (CAML; 23,570 points; the Tick Tick Bloom labels) are public on SeaBASS. The WQP legacy endpoints silently drop USGS data after 2024-03-11, so use WQX3.
6. **Recommended architecture.** A wavelength-token spectral encoder with a masked spatial aggregator over water pixels and an MDN-style probabilistic multi-target head. Train it end-to-end on L2A matchups, and pretrain the spectral stem on GLORIA. Add frozen EO-foundation-model or embedding features as an ablation, not as the core. The same band-token design lets the model be distilled into an RGB+NIR student by dropping bands.

**Decisions to take now:**
- **D1. Headline benchmark:** Track S of §d.1 (S2 C1 L2A + Landsat 8/9 matchups; unseen lakes *and* future years), plus Track T (within-lake skill) and Track H (HAB detection with a placebo test). GLORIA (Track R) is only a sanity check.
- **D2. Fixed baseline set:** metadata-only and KNN-location; index algorithms; MDN three ways (official weights, GLORIA-retrained, matchup-retrained); GBM; operational products; foundation-model features.
- **D3. Imagery:** Earth Search `sentinel-2-c1-l2a`; MPC `landsat-c2-l2`, `sentinel-3-olci-wfr-l2-netcdf`, `hls2-*`, `jrc-gsw`. A free CDSE account is optional (for OLCI L1, ACOLITE/C2RCC baselines and CLMS LWQ).
- **D4. Pilot before scaling:** run one state and year end-to-end (matchups → all baselines) to measure real yield and the baseline gap before building the national set.
- **D5. Model:** SatWQ-Net (§d.4), with foundation models as an ablation branch. The drone student is the same network restricted to B2/B3/B4/B8, plus GLORIA-simulated drone bands (§d.7).

---

## (a) SOTA methods and reported metrics

### a.1 Metric definitions (use these exactly, for comparability with the MDN literature)

With estimate *e*, observation *o* and Q = e/o:

- **MdSA** (median symmetric accuracy, called ε in Pahlevan 2022): 100 × (exp(median |ln Q|) − 1).
- **SSPB** (symmetric signed percentage bias, β): 100 × sign(median ln Q) × (exp(|median ln Q|) − 1). Source: Morley et al. 2018, *Space Weather*, doi:10.1002/2017SW001669. Adopted with slight changes by Pahlevan et al. 2022, Eqs. 1–3.
- **Slope**: slope of a Type-I linear regression of log10(e) on log10(o).
- **RMSLE / RMSLD**: sqrt(mean((log10 e − log10 o)²)).
- **MAE (log)** = 10^mean|log10 e − log10 o| and **Bias (log)** = 10^mean(log10 e − log10 o). These are multiplicative factors: 1.0 is perfect, and a Bias of 1.5 means +50% (Pahlevan 2020, Eqs. 8–9; Seegers et al. 2018, *Opt. Express*, doi:10.1364/OE.26.007404).
- Older MDN papers (2020) also report **MAPE** (median absolute % error) and the pairwise **Model Win Rate**.
- Uncertainty metrics used by Werther 2022 and Saranathan 2024: **PICP / ρ<sub>UNC</sub>** (coverage of the prediction interval), **sharpness / σ<sub>UNC</sub>**, and **MACD** (mean absolute calibration difference).

### a.2 In-situ radiometry benchmarks (Rrs → WQ, sensor bands simulated from in-situ hyperspectral Rrs)

| Method (paper) | Bands | Data, N | Split | Chl-a | TSS | a<sub>CDOM</sub>(440) | Notes |
|---|---|---|---|---|---|---|---|
| **MDN** — Pahlevan et al. 2020, RSE 240:111604 | MSI (7 bands, 400–800 nm); OLCI (12) | 2,943 Rrs–Chl pairs | Random: 1,000 train; test 1,943 (MSI), 1,905 (OLCI) | MSI: MAPE 24.0%, RMSLE 0.616, Bias 0.995, MAE 1.275, slope 0.876. OLCI: 22.9%, 0.581, 1.003, 1.265, 0.865 (Table 2) | – | – | OC3 (MSI): MAPE 74.5%, RMSLE 1.276. Blend: 62.1%, 1.199. Architecture: 5 layers × 100 units, 5 Gaussians, median of 10 inits |
| **MDN, simultaneous** — Pahlevan et al. 2022, RSE 270:112860 | OLI, MSI, OLCI | Chl: 5,680 of 6,475 (OLI). TSS 3,847. a<sub>CDOM</sub> 2,910. Only 1,613 samples have all three | **Hold-out 50/50 random** | MdSA: MSI 29.2% (β −3.4%, slope 0.88, RMSLD 0.62). OLCI 26.2%. OLI 59.1% | MSI 32.1%, OLCI 29.9%, OLI 40.6% | MSI 34.5%, OLCI 26.8%, OLI 37.9% | Table 2. Missing targets handled by multiple imputation |
| same | same | 35 data sources (MSI Chl-a) | **Leave-one-dataset-out** | MSI: median MdSA **48.30%** (weighted 42.79%) vs 2nd-best 52.2% (GI2B; >850 invalid retrievals) / 62.7% | MSI: 48.23% (62.20%) vs SOLID 58.5% (73.0%) | MSI: 55.03% (91.45%) vs Ficek 54.2% (184.4%) | Table 3. Out-of-sample advantage over 2nd-best shrinks to about 4–10 points |
| **MDN vs BNN-MCD on GLORIA** — Saranathan et al. 2024, Front. Remote Sens. 5:1383147 | MSI, OLCI, HICO, PRISMA | GLORIA. Test N: Chl 2,544, TSS 2,338, a<sub>CDOM</sub> 2,228 | 50/50 random, plus LOO over 22 GLORIA sub-datasets | MDN MSI: MdSA **30.2%**, SSPB 6.6%, slope 0.85, RMSLE 0.72. OLCI 29.5%. BNN-MCD MSI 35.39% | MDN MSI 42.1%, OLCI 39.4%. BNN 44.89% | MDN MSI 40.3%, OLCI 34.8%. BNN 45.98% | Table 3. LOO error is higher than hold-out error for about 17–19 of 22 left-out sets. MDN interval coverage ρ<sub>UNC</sub> 68–78% vs BNN 35–48% |
| **BNN (MC dropout)** — Werther et al. 2022, RSE 283:113295 | OLCI, MSI | 1,755 obs, 178 oligo/mesotrophic systems (+167 eutrophic obs, 31 systems) | 50/50 random; LOO by region; Lake Geneva matchups | MSI BNN MdSA: oligotrophic 32.42% vs **MDN (original weights, not retrained)** 71.13%. Mesotrophic 19.14 vs 26.43. Eutrophic 19.77 vs 20.94 (Table 1) | – | – | Not like-for-like: the BNN was trained in-domain, while the MDN was used off-the-shelf |
| **MDN, Landsat-8** — Smith et al. 2021, Front. Remote Sens. 1:623678 | OLI | 4,354 in situ (Type A). 3,337–3,371 satellite matchups (Type B) | 50/50 random | In situ: "49% (median) accuracy with near-zero bias" (abstract) | – | – | See a.3 for Type B |
| **MDN, phycocyanin** — O'Shea et al. 2021, RSE 266:112693 | HICO, PRISMA (hyperspectral) | 939 samples | Random half | PC MdSA **44.3%**, "less than half" of all viable optimized multispectral PC algorithms | – | – | Inputs: band ratios and line heights. Code: STREAM-RS/MDN-Phycocyanin |
| **MDN, heritage sensors** — Balasubramanian et al. 2025, Front. Remote Sens. 6:1488565 *(abstract only)* | MODIS, MERIS, VIIRS | Augmented GLORIA, N = 9,956 (+ IOPs) | 80/20 + LOO | Median MdSR over 10 variables: 25.9% (MODIS), 24.5% (MERIS), 28.9% (VIIRS) | (TSS has the highest error) | – | SeaBASS satellite matchups: 60–75% (a.3) |
| **SOLID** (TSS) — Balasubramanian et al. 2020, RSE 246:111768 *(abstract only)* | multi | >3,500. Independent test N = 2,729 (0.1–2,626.8 g m⁻³) | Independent subset | – | "outperforms all other models … from 10 to >100%" vs Nechad 2010, Novoa 2017, etc. | – | Implemented in MDN-STREAM `benchmarks/tss/SOLID` |
| **B1D-CNN** (Salah, Higa, Ishizaka, Salem; IGARSS 2023) | MSI | "global in-situ data" | **unverified** | Claims RMSE improvement of 9.25–54.12% and MAE reduction of 3.45–59.53% vs MDN and classical algorithms | – | – | Conference abstract only. Split and whether MDN was retrained are **unverified**. Not a credible SOTA claim without those |

**Reading the table.**
- (i) Every headline MDN number above comes from a **random split of in-situ data**. Pahlevan et al. 2022 themselves note that such splits "may favorably impact the model performance". The leave-one-dataset-out numbers (about 46–55%) are the honest out-of-sample figures.
- (ii) Among strong models on GLORIA-like data, gains have been single-digit MdSA points. The label noise floor (lab methods, depth, timing) probably limits further gains.
- (iii) The officially released MDN weights "are trained using 100% of the in situ data" (MDN-STREAM README), and that data overlaps GLORIA. Any comparison on GLORIA must therefore **retrain** the MDN on the same folds.

### a.3 Satellite-matchup results (where the real difficulty lies)

| Study | Imagery / AC | Matchups | Time / space rule | Reported result |
|---|---|---|---|---|
| Pahlevan et al. 2020 | MSI (SeaDAS, POLYMER, ACOLITE); OLCI-A (SeaDAS, POLYMER) | "n < 800". Common sets 281 (MSI) and 219 (OLCI); >85% of MSI matchups from Chesapeake buoys | ±30 min (buoys), ±3 h (lakes); 3×3 window, ≥5 valid pixels, median | Best MSI products: **Bias 1.34, MAE 1.86, RMSLE 0.96** (vs 0.616 in situ). Authors: AC causes "at least 30% loss of performance". ACOLITE biased high |
| Pahlevan et al. 2022 | MSI and OLI with ACOLITE or SeaDAS | Mostly North American WQP/NWIS/Chesapeake/LPN | ±3 h coastal; same-day inland; ±3 days for a<sub>CDOM</sub>; 3×3 median | ACOLITE MSI Chl-a: "neither MDN nor OC3 is viable". MDN best for OLI TSS. a<sub>CDOM</sub> strongly affected by Rrs errors |
| Smith et al. 2021 | OLI, SeaDAS Rrs or Rayleigh-corrected ρ<sub>s</sub> | 3,371 (50/50 random) | Same day; 30 min for dynamic sites; 3×3 | MDN<sub>A</sub> (in-situ-trained) "error levels exceeding 100%". **MDN<sub>B</sub>, trained on the satellite matchups themselves, improves accuracy.** This is the key precedent for end-to-end training |
| Saranathan et al. 2024 | MSI, ACOLITE v20220222 | **3,053** (Chl-a, TSS; Chesapeake, Klamath, Great Lakes, Winnipeg, Simcoe) | ±3 h coastal / same-day inland; 3×3 ≥5 valid | MdSA **Chl-a ≈ 195%** (vs 30% on GLORIA). **TSS 110% (MDN), 150% (BNN-MCD)** (vs ~42%). MDN interval coverage ρ<sub>UNC</sub>: Chl-a 47.79% (vs 78.5% in situ), TSS 63.83% (vs 74.0%) (Tables 4 and 7) |
| Balasubramanian et al. 2025 *(abstract)* | MODIS-A/T, MERIS, VIIRS (SeaBASS matchups) | – | – | Median MdSR 63.21% / 63.15% / 60.45% / 75.19% |
| Werther et al. 2022 | OLCI/MSI, POLYMER or C2RCC, Lake Geneva | 21 OLCI + 12 MSI | – | OLCI-POLYMER: OC3/Blend 48.41% / 36.61% vs BNN 49.75%. With C2RCC, BNN "four times more accurate" than OC3/Blend |
| **Sherf et al. 2025**, arXiv:2510.24124 (v2 May 2026) | S2 Sen2Cor L2A (GEE) + **AquaMatch** (USGS, from WQP) | 13,626 matchups. Test 2,044 points, 867 water bodies | ±1 day; lakes; depth ≤1.5 m | XGBoost (lake-level 70/30 split): R² 0.69, slope 0.74, RMSLE 0.60. +Residual CNN (**random** 50/50 split, so possibly leaky): R² 0.79, slope 0.91, RMSLE 0.56, MAE 13.52 mg m⁻³ (Tables 3/5). No MDN baseline |
| Topp et al. 2021, ERL 16:055025 (Secchi) | Landsat 5/7/8 SR (AquaSat) | **250,760** (200,607 train / 50,153 random hold-out) | AquaSat ±1 day | MAE **0.71 m** (MAPE 38%), bias <0.01 m. **Optical-only** model: RMSE 1.3 m, R² 0.50. XGBoost with leave-location-leave-time-out CV used for feature selection. 11 features (3 optical + 8 static lake/landscape) |
| Cao et al. 2020, RSE 248:111974 (BST = XGBoost) | Landsat-8 Rayleigh-corrected | 225 (8 Chinese lakes) + 97 (SeaBASS). Test N = 102 | near-synchronous | R² 0.79, RMSD 7.1 µg L⁻¹, MAPE 24%, MAE 1.4, Bias 0.9 (abstract) |
| Tick Tick Bloom (DrivenData/NASA 2023) | S2 L1C/L2A, Landsat 8/9 (MPC/GEE), HRRR, Copernicus DEM | US-wide in situ cyanobacteria densities (count not stated on pages read) | **Geographic split**: "each geographic area is either entirely in the train set or entirely in the test set" | Region-averaged RMSE of severity class (1–5): **0.7608** (1st, private), 0.7616 (2nd), 0.7844 (3rd). The 1st place used GBDT on temperature + water colour for the Midwest/NE and **KNN on location** for the West/South. CNNs "performed poorly" |
| JAG 2026 "Pioneering Test-Time training … Chl-a from Sentinel-2 … China's lake eutrophication (2018–2025)" | S2 | 2,621 lakes | **unverified** (publisher page 403) | R² 0.79, RMSE 9.99 µg L⁻¹; claims 28–36% better than conventional ML (search snippet only) |

**Conclusion.** Real-imagery Chl-a errors span MdSA ≈ 60–200% for physics-based atmospheric correction followed by an MDN trained on in-situ data. Models trained *on matchups* (MDN<sub>B</sub>, XGBoost) do far better but have not been evaluated on clean held-out-lake/held-out-time splits against a fairly retrained MDN. This is the gap SENTINEL should target (§c).

### a.4 Other indicators: best published accuracy

| Indicator | Method | Reported accuracy | Protocol / caveat |
|---|---|---|---|
| Secchi depth | Lee et al. 2015, RSE 169:139 (mechanistic, K<sub>d</sub>-based) | ~18% mean absolute difference, R² 0.96 (N = 338) | Uses in-situ IOPs/K<sub>d</sub>, not satellite imagery |
| Secchi depth | Topp et al. 2021 (XGBoost, AquaSat Landsat) | MAE 0.71 m (MAPE 38%) | Random 20% hold-out; static covariates matter (optical-only R² 0.50) |
| Turbidity | Dogliotti et al. 2015, RSE 156:157 (single-band 645/859 nm, switching) | Mean relative error 13.7%, bias 4.8% (n = 106); valid up to ~1000 FNU | In-situ reflectance; coastal/estuarine waters |
| TSS | Nechad et al. 2010, RSE 114:854 (single-band semi-analytical) | – (full text not read) | Benchmark in MDN-STREAM |
| TSS | SOLID 2020; MDN 2022 | MDN MSI MdSA 32.1% (hold-out) / 48.23% (LOO) | In situ |
| a<sub>CDOM</sub>(440) | MDN 2022; Ficek (band-ratio); QAA-CDOM | MDN MSI 34.5% (hold-out) / 55.03% (LOO) | Few samples. Satellite a<sub>CDOM</sub> strongly affected by Rrs errors |
| Phycocyanin | Simis et al. 2005, L&O 50:237 (620/709 nm semi-analytical) | – | **Needs a ~620 nm band.** OLCI has one (Oa06); MSI and OLI do not. Sentinel-2 can at best proxy cyanobacteria via Chl-a, 705 nm and scum NIR |
| Phycocyanin | O'Shea et al. 2021 MDN-PC | MdSA 44.3% (hyperspectral, random half) | HICO/PRISMA, not S2 |
| Cyanobacteria Chl-a (CI-based) | Seegers et al. 2021, RSE 266:112685 (MERIS ChlBS) | Multiplicative bias 1.11, MAE 1.60 (60%) | >2,300 CONUS lakes; bootstrapped CI→Chl-a |
| Lake surface temperature | Schaeffer et al. 2018, IJRS 39:7789 (Landsat 5/7 ST vs WQP) | MAE **1.34 °C** for lake pixels >180 m from land; **4.89 °C** at the land–water boundary; 1.11 °C in estuaries | Shows adjacency matters; use a shore buffer ≥ 2 TIRS-resampled pixels |
| Lake surface temperature | ESA Lakes CCI v2.0.2 (Carrea et al. 2023, Sci. Data 10:30, Table 14) | Satellite minus in situ, quality level 5: median **−0.15 K**, robust SD 0.50 K (N = 30,080). QL4: −0.22 / 0.76 K. QL3: −0.30 / 1.02 K | 79 lakes, 207 sites; matchups within 3 km and 3 h. ~1 km sensors, so only large lakes. The paper says its Chl-a and turbidity validation (LIMNADES) "was not independent" because the same data tuned the algorithms |

### a.5 Methods from 2022–2026 that claim to match or beat MDN, and how credible the claims are

- **BNN-MCD** (Werther 2022; Saranathan 2024). On a fair, same-split comparison (Saranathan 2024) the MDN is slightly better in MdSA: MSI Chl 30.2 vs 35.4%. The BNN gives sharper but under-covering uncertainties. Werther's large gains came from in-domain training against off-the-shelf MDN weights. **Not a fair "beat".**
- **Simultaneous / multi-task MDN** (Pahlevan 2022; Saranathan 2024, Table 6). This is a small but consistent gain. On OLCI Chl, MDN-Sim vs single-target MDN gives about 1% MdSA. Simultaneous estimation also regularises uncertainty.
- **Hyperspectral IOP-based MDN** (O'Shea et al. 2023, RSE 295:113706; Pahlevan et al. 2021, RSE 253:112200). Pahlevan 2021 reports Chl-a errors cut by more than 2× vs band algorithms (N = 1,902), with a<sub>ph</sub> median error 20–30%. These are hyperspectral only.
- **End-to-end on satellite matchups** (Smith 2021 MDN<sub>B</sub>; Cao 2020 BST; Sherf 2025 XGBoost+RCNN; Topp 2021). These are the strongest real-imagery numbers. None used a held-out-lake *and* held-out-time split against a retrained MDN on the same images.
- **Foundation models for water colour.** Dawson et al. 2025, arXiv:2509.21273, "A Sentinel-3 foundation model for ocean colour", pretrains a Prithvi-EO ViT on OLCI and fine-tunes it for (marine) Chl-a and primary production. Quantitative results were not read here (abstract only). Lima & Martins 2026, arXiv:2606.17242, compare Swin and other ViTs for 30 m Landsat/S2 coastal floating-bloom mapping and report omission/commission errors of 8–65%. **No inland-water foundation-model paper with a leakage-free benchmark was found.** See §a.6 for model availability.
- **Transformers on spectra.** No peer-reviewed transformer that beats a retrained MDN on GLORIA under grouped splits was found (search budget limited; treat as **unverified absence**).

### a.6 Open-weight EO foundation models

This section comes from a dedicated sub-survey. Weights were checked with unauthenticated HEAD requests on Hugging Face (302 = downloadable without an account) on 2026-10-22. Memory figures assume ~16 bytes per parameter for AdamW mixed-precision fine-tuning plus activations; they are estimates. Bracketed numbers [FMx] refer to the foundation-model references in §f.

| Model (variants, params) | Paper | Weights / licence | Inputs relevant to water | Small-chip fit | RTX 2080 Ti (fp16) | WQ fit |
|---|---|---|---|---|---|---|
| **OlmoEarth v1.x**: Nano 1.4M, Tiny 6.2M, Base 89M, Large 308M (v1.2 adds Small 35.6M; v1.3 Base 128M) | arXiv:2511.13655 (2025); v1.2 arXiv:2605.20804 (2026) [FM14] | HF, open. "OlmoEarth Artifact License": free use, bans military/defence and extractive uses [FM15][FM16] | **S2 L2A 12 bands**, Landsat, S1; 1×1 to 128×128 px, patch 1–8, up to 12 monthly steps; pretraining data harmonized for the S2 offset | **Best**: native small inputs | All sizes full-FT (Large: small batch) | **High** |
| **TerraMind 1.0**: tiny / small / base / large | ICCV 2025; arXiv:2504.11171 [FM6] | HF, open; Apache-2.0 [FM7] | S2 L1C (13) or L2A (12), S1, DEM, LULC/NDVI tokens; 224 px, patch 16 | Patch 16, so small chips give few tokens (upsample or set `img_size`) | T/S/B full-FT; L LoRA | **High**. Best PANGAEA MADOS: L 75.57 / B 69.52 mIoU |
| **Galileo**: Nano / Tiny / Base (~1 / 6 / 86M, estimated) | ICML 2025; arXiv:2502.09356 [FM18] | HF; MIT [FM19] | Monthly S2 **L1C (GEE harmonized)** 10 bands + S1 + ERA5 + TerraClimate + SRTM + Dynamic World + location; flexible patch | Good | Full-FT | **High** (but TOA input) |
| **Prithvi-EO-2.0**: tiny-TL 5M, 100M-TL, 300M(-TL), 600M(-TL) | arXiv:2412.02732 [FM3] | HF; Apache-2.0 [FM4] | **HLS** 6 bands (B, G, R, NIR-narrow, SWIR1, SWIR2) at 30 m; 224 px; T = 4; TL variants add lat/lon/date. No red-edge | Medium | ≤300M full-FT; 600M LoRA/frozen | Medium. Closest analogue: GPP regression from 50×50 HLS chips at 37 flux towers, R² 0.74–0.88 (600M-TL) vs RF 0.64–0.78 |
| **Copernicus-FM** (ViT-B; ViT-L checkpoint) | ICCV 2025; arXiv:2503.11849 [FM22] | HF + TorchGeo; code Apache-2.0 (+ MIT/CC-BY-NC parts), weights CC-BY-4.0 | Any band given wavelength + bandwidth; pretrained on S1, S2 TOA, **S3 OLCI (21 bands)**, S5P, DEM; metadata (lon/lat, time, area) | Medium | Full-FT | Medium–high. **The only open foundation model pretrained on OLCI** (has 620 nm for phycocyanin) |
| **DOFA**: Base ~111M, Large ~337M | arXiv:2403.15356 [FM23] | HF + TorchGeo; MIT / CC-BY-4.0 | Any bands via centre wavelength; 224 px, patch 16 | Medium | Full-FT | Medium |
| **Panopticon** (ViT-B/14); **SMARTIES** (B 116.5M, L 335M) | CVPRW 2025 [FM24]; ICCV 2025 [FM25] | HF; Apache-2.0 / CC-BY-4.0 | Any-sensor, wavelength-aware; 224 px | Medium | Full-FT (B) | Medium |
| **Clay** v1 (201M) / v1.5 (632M; 311M encoder) | Docs only [FM26] | HF; Apache-2.0 (v1.5), OpenRAIL-M (v1) | S2 L2A, Landsat C2 L1/L2, S1, NAIP…; needs wavelengths, GSD, lat/lon, time | Patch 8 | v1 full-FT; v1.5 LoRA | Medium–high |
| **AnySat** (125M) | CVPR 2025 [FM21] | HF; MIT | 11 sensors incl. S2 time series; 1-px sub-patches | Good | Full-FT (`flash_attn=False`) | Medium–high |
| **CROMA** (B/L); **SSL4EO-S12** (ResNet/ViT-S/MAE); **DeCUR**; **Presto** (0.4M) | NeurIPS 2023; IEEE GRSM 2023; ECCV 2024; arXiv:2304.14065 [FM27][FM28][FM30][FM20] | Open; MIT / Apache-2.0 / CC-BY-4.0 | S2 L1C 12–13 bands (SSL4EO 2021 data, **pre-offset**, so subtract 1000 from newer scenes). CROMA README uses a per-batch min-max stretch, which **destroys absolute dark-water radiometry** | Medium | Full-FT | Medium (classic baselines). Presto suits pixel time series |
| SatMAE / SatMAE++ / Scale-MAE / SatlasPretrain / SpectralGPT / SSL4EO-L / DINOv3-SAT / SkySense | various [FM31–FM38] | Mixed (CC-BY-NC, 8-bit TCI, RGB-only, gated, or unclear licence) | RGB or 8-bit or uint8 inputs lose water radiometry | – | – | Low |
| **Hydro** (Swin-v2-B; RGB and 12-band versions) | GitHub "work in progress" [FM41] | HF; MIT | S2 L2A 12 bands; 100k water patches | Medium | Full-FT | Medium (water-specific, **no benchmarks**) |
| HyperSIGMA / SpectralEarth | TPAMI 2025 / JSTARS 2025 [FM39][FM40] | HF | Hyperspectral (EnMAP etc.) | – | B ok | Later (PRISMA/EnMAP/PACE work) |

**Pixel-embedding products (cheap feature baselines).**
- **AlphaEarth Foundations / Google Satellite Embedding V1** (Brown et al. 2025, arXiv:2507.22291) [FM43–FM47]: 64-d, 10 m, **annual** (2017–2025 in the bucket copies), CC-BY-4.0. Coverage explicitly includes "inland waterways". Anonymous COG access was verified on Source Cooperative (`source.coop/tge-labs/aef`; HEAD 200) and on the GCS bucket `alphaearth_foundations` (anonymous ranged GET worked even though its README says requester-pays).
  - **Use**: as a static site prior (trophic state, optical type) only. Annual embeddings cannot track within-season Chl-a.
- **TESSERA v1.1**: masks permanent water (NaN = water), so it is **unusable for lakes** [FM48][FM49].
- Patch-level embedding sets (Clay, Earth Index, Major TOM, Copernicus-Embed) are 320 m to 0.25°, too coarse for point samples.

**Water benchmarks and lessons.**
- PANGAEA (Marsocci et al., arXiv:2412.04204) [FM55]:
  - Sen1Floods11 is saturated. A UNet trained from scratch scores 91.42 mIoU, and every foundation model scores 87–91.
  - MADOS (marine pollutants) separates models: TerraMind-L 75.57, CROMA 67.55, DOFA 59.58, Prithvi-1.0 49.98.
  - PANGAEA reports that under a domain gap on regression "even supervised baselines do better than GFMs".
- Studies that applied general EO foundation models to inland water-quality *retrieval* with numbers: **none found** (search incomplete).
- Narula 2026 (*Remote Sensing* 18:2147; doi:10.3390/rs18132147) [FM65] benchmarked vision-language models for HAB detection on S2 in western Lake Erie. VLM F1 was 0.69–0.75, vs random forest 0.76, vs a **trivial "always bloom" predictor 0.77**. This is another warning about trivial baselines.
- CyFi (https://cyfi.drivendata.org/) [FM67] is DrivenData's open-source LightGBM-on-S2 cyanobacteria estimator, derived from Tick Tick Bloom. It is a natural Track H baseline.

**Recommendation for SENTINEL.**
- **(1) OlmoEarth Tiny/Small** (fine-tune Base): native 1–128 px inputs, S2 L2A + Landsat, harmonized pretraining, TorchGeo support. Its quickstart reads raw 2025 L2A without removing the offset, so harmonize inputs yourself.
- **(2) TerraMind Small/Base**: 12–13 bands including B1 and B5, best MADOS score, Apache-2.0.
- **(3) Galileo Tiny/Base**: monthly series + ERA5/terrain drivers, MIT. Expects harmonized TOA.
- Add **Copernicus-FM** if OLCI is used.
- Pixel-embedding baseline: **AlphaEarth**.
- Practical notes for the 2080 Ti: use fp16 autocast with GradScaler. FlashAttention-2 needs Ampere or newer, so use PyTorch SDPA. TorchGeo 0.10.1 needs Python ≥3.12 while TerraTorch 1.2.13 pins torchgeo <0.10, so separate environments may be needed.
- Foundation models should be an **ablation branch** of SatWQ-Net (§d.4), not its core. The evidence for gains on regression under domain shift is weak.

**S2 offset handling.**
- OlmoEarth, Galileo and AlphaEarth were trained on offset-harmonized data.
- SSL4EO-S12, Presto and Copernicus-FM were pretrained on mostly pre-2022 (pre-offset) data.
- For TerraMind, Clay, DOFA and Hydro, offset handling is undocumented.
- **Always convert DN to a common reflectance scale before normalizing.** A 0.1 reflectance error is several times typical water reflectance.
- The sub-survey reported that Earth Search keeps the offset in pixel values for both S2 collections. That holds for `sentinel-2-c1-l2a` and for legacy items flagged `boa_offset_applied: false`. **It does not hold for legacy items flagged `boa_offset_applied: true`**: there I measured the 1000-DN difference directly on the same pixel (§b.1).


### a.7 HAB and other WQ-disaster detection from satellites

| Task | Method / study | Data and protocol | Reported numbers |
|---|---|---|---|
| Cyanobacteria (CyanoHAB presence) | CI<sub>cyano</sub> (EPA CyAN; MERIS/OLCI 300 m). Mishra et al. 2021, *Sci. Total Environ.* 774:145462 | Same-day matchups with field microcystin (presence = MC detected), 11 states, 11 seasons (2005–2011, 2016–2019) | Overall accuracy **84%**, precision 87%, recall 90%. Bootstrap 95% CI for accuracy 77–87% |
| Cyanobacteria vs. state advisories | CI<sub>cyano</sub>. Whitman et al. 2022, *Harmful Algae* 115:102191 | 160 state recreation advisories, 1,343 state-reported events | True-positive rate **69%** (advisories) and **60%** (events). Reduction or absence detected after 76% of advisories ended |
| National bloom statistics | Coffer et al. 2020 (Ecol. Indic. 111:105976; 2,321 lakes); Coffer et al. 2021 (Ecol. Indic. 128:107822; 2,196 lakes); Schaeffer et al. 2022 (Ecol. Indic. 140:108990) | OLCI/MERIS frequency and extent | Descriptive only (no skill scores). Shows CyAN covers only ~2,000+ large, resolvable US lakes |
| Cyanobacteria severity (cells/mL classes) | Tick Tick Bloom 2023 (DrivenData) | Severity 1: <20,000; 2: 20,000–<100,000; 3: 100,000–<1,000,000; 4: 1,000,000–<10,000,000; 5: ≥10,000,000 cells/mL. Geographic split | Region-averaged RMSE 0.7608 (winner). GBMs on colour statistics + temperature + elevation/location. Competition data terms were "only for the competition", but the labels are public as **CAML** on SeaBASS (23,570 points; doi:10.5067/SeaBASS/CAML/DATA001) |
| Indices for Sentinel-2 / OLCI | NDCI (Mishra & Mishra 2012, RSE 117:394); MCI; MPH (Matthews et al. 2012, RSE 124:637; Matthews & Odermatt 2015, RSE 156:374); FAI (Hu 2009, RSE 113:2118) | Thresholds on red-edge (705 nm) or line heights; MPH needs OLCI/MERIS bands | Baselines to beat; implemented in MDN-STREAM `benchmarks/chl` |
| Operational S2 bloom product | CLMS Lake Water Quality 100 m v2 (S2-MSI; 10-daily), layer `fcbprob` (floating cyanobacteria probability) plus `chlamean`, `tsmmean`, `tmean` (turbidity), `tsi` | Verified in the CDSE STAC on 2026-10-22 (download needs a free CDSE account) | Operational baseline (no skill numbers read here) |
| Coastal floating algal blooms (30 m) | Lima & Martins 2026, arXiv:2606.17242 (Swin Transformer) | Global coastal bloom patches; L8/9 + S2A/B/C | Omission/commission errors 8–65% across models |
| Oil spills (SAR, marine) | Krestenitis et al. 2019, *Remote Sens.* 11:1762 | 1,112 Sentinel-1 images with EMSA CleanSeaNet-confirmed spills. 5 classes | DeepLabv3+ mIoU **65.06%**. Oil-spill IoU 53.38% (UNet 53.79%) |
| Marine pollutants (optical S2) | MARIDA (Kikaki et al. 2022, PLoS ONE 17:e0262247); MADOS (Kikaki et al. 2024, ISPRS J. 210:39) | S2 pixel annotations incl. sediment-laden water, turbid water, foam (MARIDA) and oil spills (MADOS) | Benchmark datasets (numbers not re-read here). **Marine**: transfer to rivers/lakes is unproven |
| Sediment / tailings events | Rotta et al. 2020, JAG 90:102119 (Brumadinho 2019) | Multispectral imagery + InSAR case study | Qualitative: abrupt SPM increase in the Paraopeba River. No detection skill score |

**What is realistically detectable from public imagery.**
- **Yes:** surface cyanobacterial blooms and scums (S2 at 20 m for lakes larger than ~1–4 ha; OLCI 300 m for large lakes); high-turbidity and sediment events and plumes (S2/Landsat; red/NIR saturate slowly); thermal anomalies in large rivers and lakes (Landsat 100 m TIRS); large floating oil slicks (S1 SAR, mainly marine).
- **Indirect only:** hypoxia (proxies: Chl-a, temperature, stratification season); fish kills; sewage or fecal contamination (turbidity and colour after rain, at best); salinization (no optical signature in fresh water).
- **Not feasible to claim:** inland chemical spills at scale (no labelled inland dataset; most spills are invisible).

### a.8 Atmospheric correction (AC) over water: evidence

| Study | Finding (exact) |
|---|---|
| ACIX-Aqua — Pahlevan et al. 2021, RSE 258:112366 *(abstract)* | 8 AC processors for L8/S2. ">1000 radiometric matchups". Best processors: median errors in ρ<sub>w</sub>(560) and ρ<sub>w</sub>(664) of **20–30%** (inland CVD) and 15–30% (AERONET-OC). Blue bands 25–60%. 20–30% ρ<sub>w</sub> errors give **25–70%** uncertainty in Chl-a and TSS. Calls for better absorbing-aerosol, glint and **adjacency** corrections |
| Warren et al. 2019, RSE 225:267 *(abstract)* | 6 processors on S2A (ACOLITE, C2RCC, iCOR, l2gen, POLYMER, **Sen2Cor**). 1,059–1,668 matchups per processor (521 common). Uncertainties often ">100% in the red and >1000% in the near-infra red". POLYMER and C2RCC best (RMSD ~0.0016 sr⁻¹; MAD ~40–60% blue/green) |
| Maciel et al. 2023, L&O Letters (doi:10.1002/lol2.10344) *(abstract)* | USGS Landsat SR (LaSRC/LEDAPS) vs in situ, N = 1,100. L8/9 green and red "marginally meet" 30%. Blue and coastal 48–110%. **L5/7 visible bias >+25%**, which can create **artificial downward trends** when merged with L8/9 |
| Pahlevan et al. 2020 | On MSI, POLYMER gave ~70% more valid matchups (noisier). ACOLITE (2019 version) biased high for Chl <15 mg m⁻³ but better in hypereutrophic Klamath |

**Implication.** Using L2A or LaSRC reflectance directly is acceptable for *end-to-end* learning, because the model learns the biases. It is not acceptable as input to an in-situ-trained physics or MDN model without accepting large errors. For trend work, never mix Landsat 5/7 with 8/9 without harmonization.

### a.9 Open code usable as baselines (verified)

- **MDN-STREAM** (https://github.com/STREAM-RS/MDN-STREAM; GPL-3.0; updated Nov 2025, pushed Sep 2026). This is the production MDN behind NASA STREAM (https://ladsweb.modaps.eosdis.nasa.gov/stream/). Its `Weights/` folders are MSI, OLI, OLCI, S3A, S3B, HICO, MERIS, MOD, VI and SD8-cc_base (the last three presumably MODIS, VIIRS and PlanetScope SuperDove: **unverified**). It also ships relative spectral response files (`Rrs_manipulation/Rsr/*.csv`) and a `gloria_processing_utils.py`. It also includes **benchmark implementations**: OC/OCx, NDCI, Gons, Gilerson/Gurlin/Moses 2-band and 3-band, Smith Blend, FLH, FAI, QAA/GIOP, CDOM (Ficek, Mannino, QAA-CDOM), TSS (Nechad, Novoa, Petus, Miller-McKee, SOLID) and PC (Simis 2005/2007, Hunter, Mishra, Schalles). It is TensorFlow-based.
- Legacy MDN: https://github.com/BrandonSmithJ/MDN (GPL-3.0, last push 2024-08-16). MDN-PC: https://github.com/STREAM-RS/MDN-Phycocyanin. AQUAVERSE tutorials (Ocean Optics 2026): https://github.com/STREAM-RS/aqv-tutorials (under development).
- Tick Tick Bloom winning solutions (MIT): https://github.com/drivendataorg/tick-tick-bloom (Zenodo doi:10.5281/zenodo.8329929).

---

## (b) Datasets and imagery sources with verified access

### b.1 Imagery access (all queried live on 2026-10-22; point = western Lake Erie, 41.70 N, 83.30 W)

| Source / endpoint | Relevant collections (verified) | Account? | Format | Verified behaviour and pitfalls |
|---|---|---|---|---|
| **Microsoft Planetary Computer (MPC)** — `https://planetarycomputer.microsoft.com/api/stac/v1` (139 collections) | `sentinel-2-l2a` (2015-06-27→), `landsat-c2-l2` (1982→), `sentinel-3-olci-wfr-l2-netcdf` (2017-11-01→), `sentinel-3-olci-lfr-l2-netcdf`, `sentinel-3-slstr-wst-l2-netcdf`, `hls2-s30` / `hls2-l30` (2020→), `jrc-gsw`, `esa-worldcover`, `cop-dem-glo-30`, `sentinel-1-rtc` / `-grd`, MODIS 061 | **No.** Anonymous SAS tokens via `planetary_computer.sign_inplace` | COG for S2, Landsat, HLS and JRC; NetCDF4 for S3 | **S2 L2A processing baselines are mixed.** Items over T17TLG/T17TKG were 02.12 (2018, 2019), 03.00 (2021), **05.10 reprocessed 2024-10** (2023) and 05.11 (2025). Assets carry **no scale/offset metadata**, so apply DN−1000 only when `s2:processing_baseline` ≥ 04.00. Decide **per item, not by date**: on 2022-01-24, tiles T17TLG/T17TKG were 03.00 while T16TGM was 04.00 (processed 2022-05-12). From 2022-01-29 all three were 04.00. **No S2 L1C.** Landsat assets do carry `raster:bands` (SR: scale 2.75e-5, offset −0.2; ST: scale 0.00341802, offset 149 K). OLCI WFR has one NetCDF per band (6–12 MB per granule), `geo-coordinates` 56 MB, `wqsf` flags 8.7 MB. Ready-made `chl-nn`, `chl-oc4me`, `tsm-nn`, `iop-nn` and `trsp` assets are useful baselines |
| **Element 84 Earth Search** — `https://earth-search.aws.element84.com/v1` (9 collections) | `sentinel-2-c1-l2a` (**ESA Collection-1 reprocessing**; at the test tile: 05.00 for 2018–2021, 05.09 for 2023, 05.11 for 2025), `sentinel-2-l2a` (legacy), `sentinel-2-pre-c1-l2a` (0 items at the test point), `sentinel-2-l1c` (JP2), `landsat-c2-l2`, `naip`, `cop-dem-glo-30/90`, `sentinel-1-grd` | **No** for S2 (public S3 over HTTPS). Landsat assets are **requester-pays** (`s3://usgs-landsat`), so anonymous reads fail (HTTP 403 verified) | COG, deflate, 1024-px blocks (S2 L2A); JP2 (L1C) | `c1-l2a`: `raster:bands` scale 1e-4, offset −0.1 (correct for raw DN). **Legacy `sentinel-2-l2a` trap:** items flagged `earthsearch:boa_offset_applied: true` already have the 1000 DN removed, yet still advertise offset −0.1. On the same pixel the DNs were 1734 (C1) vs 734 (legacy), a difference of exactly 1000. The legacy collection also holds duplicate `_0_`/`_1_` items (original vs reprocessed). **Use `sentinel-2-c1-l2a`.** L1C JP2 window reads worked anonymously but were slow (~10 s per band window) |
| **Copernicus Data Space Ecosystem (CDSE)** — STAC `https://stac.dataspace.copernicus.eu/v1` (427 collections) | `sentinel-2-l1c` / `-l2a`, **`sentinel-3-olci-1-efr-ntc`** (OLCI L1, needed for C2RCC/POLYMER/ACOLITE), `sentinel-3-olci-2-wfr-ntc`, SLSTR WST, **CLMS Lake Water Quality** (`clms_lwq-nrt_global_100m_10daily_v2_cog`, 300 m v1/v2, reprocessed 300 m v1), CLMS LSWT 1 km | Search is anonymous; **downloads need a (free) account token**. An unauthenticated GET returned HTTP 401 "Token not found" | SAFE/zip via OData; `s3://eodata` | The only account-gated source worth having: OLCI L1 for running AC yourself, and the CLMS LWQ layers (`chlamean`, `tsmmean`, `tmean`, `tsi`, `fcbprob`, rw443–rw2190) as operational baselines |
| **USGS LandsatLook STAC** — `https://landsatlook.usgs.gov/stac-server` (14 collections) | `landsat-c2l2-sr`, `-st`, `c2l1`, ARD, `landsat-c2l3-dswe` | Search is anonymous; data are requester-pays S3 | COG | Use MPC for Landsat pixels instead |
| **PACE OCI** (NASA OB.DAAC) | L1B/L1C/L2/L3 | Earthdata Search / OB.DAAC (account requirement not confirmed on the pages read; **unverified**) | NetCDF | OCI covers 315–895 nm at 2.5 nm resolution (5 nm bandwidth), has 7 SWIR bands, a **1.2 km** nadir footprint, a ~2,700 km swath and 2-day coverage (pace.oceansciences.org/oci.htm). Public data since 11 Apr 2024. Too coarse except for very large lakes and coastal waters |

**Measured chip-read performance** (from this server, single-threaded, using GDAL settings `GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR` and `GDAL_HTTP_MERGE_CONSECUTIVE_RANGES=YES`):

| Read | Time |
|---|---|
| S2 COG, 64×64 window, one band (MPC) | 0.26–1.34 s |
| S2 COG, same window (Earth Search C1) | ~1.4 s |
| Landsat C2 L2, 32×32 window, one band (MPC) | ~0.4 s |
| JRC GSW occurrence, 32×32 | ~1.0 s |
| S2 L1C JP2 window | ~10 s (avoid) |
| OLCI WFR, 16×16 via `fsspec` + `h5py` HTTP range reads | 0.2–1.6 s per variable |

The OLCI NetCDF variables are chunked 1023×1217 (reflectance) and 818×973 (full-resolution geo), gzip-compressed, so a chip needs one chunk per variable rather than the whole file. Locating the pixel needs the 1.2 MB tie-point grid and then a partial read of `geo-coordinates`.

**Recipe that scales:**
1. Group in-situ samples by (MGRS tile, date).
2. Run one STAC query per tile and date, not one per point.
3. Open each COG once and read all windows for that scene.
4. Store uint16 DN plus masks plus the `processing_baseline` and angles.

At ~14 bands × 64×64 × 2 bytes ≈ 115 KB per chip, 100k chips ≈ 11.5 GB, which fits the disk budget. With 24–32 parallel readers, the measured latencies suggest about 10⁴ chips per hour. That is an extrapolation, not a benchmark.

**Masks.** For S2, use SCL. It was verified present; the standard Sen2Cor classes that matter are 6 (water), 8 and 9 (cloud medium/high; 6, 8 and 9 were observed in the test chip), 3 (cloud shadow), 10 (thin cirrus) and 11 (snow/ice). SCL was designed for land. Treat it as a first filter only, and confirm water with the JRC occurrence layer and a reflectance-based water test (mislabelling of turbid or bloom water is a common complaint, but it was not verified here). For Landsat use QA_PIXEL, for HLS the Fmask layer, and for OLCI the WQSF flags.

**Empirical matchup yield.** I took the Water Quality Portal Chl-a records for Minnesota lakes and reservoirs, May–Oct 2023 (6,926 rows → **5,882 unique surface site-days at 1,161 sites**, depth ≤2 m, QC samples removed). These were matched against Earth Search `sentinel-2-c1-l2a`. A matchup counted as valid when ≥5 of the 3×3 20-m SCL pixels were water.

| Window | Same-day | ±1 day |
|---|---|---|
| Any S2 overpass | 31.0% | 80.7% |
| **Valid clear-water matchup** | **14.3%** (844) | **36.9%** (2,169) |

570 distinct sites had at least one same-day valid matchup (script: `scripts/yield_experiment.py`). Throughput: after 185 daily state-wide STAC queries, SCL screening of all 5,882 site-days took 48 s with 24 threads. This is a summer sample with lenient QA. No shore buffer or glint test was applied, so expect lower yields year-round and with stricter QA.

### b.2 In-situ and matchup datasets

This section comes from a dedicated sub-survey. Links and DOIs were checked on DataCite, Crossref and Zenodo, and by direct API calls, on 2026-10-22. [Dx] numbers refer to the dataset references in §f.

**A. In-situ and matchup datasets**

| Dataset (version) | Variables | N (samples / matchups) | Water bodies / sites | Coverage | IDs + coordinates | Licence | Access (no account?) | Size |
|---|---|---|---|---|---|---|---|---|
| **AquaCorr-Global** v1.0 (Zenodo, 2026-08-14; Ghazi Zadeh Hashemi, Bolten, Kumar, Seegers) [D13] | **ACOLITE-DSF Rrs** (6 harmonized bands + extra) for L5/7/8/9 and S2A/B; Chl-a, TSS, Secchi, a<sub>CDOM</sub>, turbidity, DOC, TOC | **1,984,387 same-calendar-day matchups** (935,740 pass QC). S2: 522,500; L8/9: 330,313. Built from 12.5 M in-situ records at 372,639 sites | 202,029 matched sites; US-dominated | 1984–2024 | `site_id`, lat/lon. **98,165 rows use lake centroids**, not sampling points | CC-BY-4.0 | Open (Zenodo) | Parquet 377 MB (csv.gz 505 MB; Rrs Zarr 89 MB) |
| **GLORIA** (PANGAEA 948492) [D1][D2] | Hyperspectral Rrs 350–900 nm (1 nm); Chl-a, TSS, a<sub>CDOM</sub>(440), Secchi, turbidity | 7,572 Rrs. Chl-a 5,132; TSS 4,623; a<sub>CDOM</sub> 4,393; Secchi 3,948; turbidity 1,252. After removing 1,779 QC-flagged rows: 3,746 / 3,460 / 3,344 / 3,099 / 895 | 486 site names (347 lakes), 30 countries. Lake Erie alone has 611 samples; the median lake has 3 | 1990–2022 | lat/lon (7,445 rows), `Site_name` (not a lake ID; e.g., "Lake Erie" spans US and Canada) | CC-BY-4.0 | Open | 56 MB zip |
| **AquaMatch** ("AquaSat v2", EDI) [D11][D12] | Harmonized WQP Chl-a, DOC, Secchi, TSS, CDOM (3 QA tiers); siteSR/lakeSR Landsat C2 SR/ST | Records: Chl-a 3,393,022; Secchi 2,739,358; TSS 4,606,035; CDOM 159,480. **No published matchups yet** | siteSR 395,307 sites; lakeSR covers NHD lakes >1 ha | 1970–2024/25 | WQP site IDs | **CC0** | EDI (scripted access blocked; DataONE mirror works) | Chl-a package 1.87 GB |
| **AquaSat v1** (figshare) [D8][D9][D10] | Landsat 5/7/8 SR + TSS, DOC, Chl-a, Secchi | ">600,000 matchups" (±1 day). Per-variable counts unverified. AquaCorr's same-day re-match of the same records: Secchi 271,616; Chl-a 100,713; TSS 90,934 | CONUS; 71% of data from FL/WI/MN | 1984–2019 | SiteID; LAGOS lakes as centroids | CC-BY-4.0 | Open | 291 MB matchup CSV |
| **Water Quality Portal** (WQX 2.2 legacy + **WQX 3.0**) [D27][D28] | All (characteristic names listed below) | Lakes 2016–2025 (legacy endpoint, activities): Chl-a 331,993; Secchi 545,183; TSS 162,883; turbidity 392,817; phycocyanin 23,209; CDOM+fDOM 1,870; microcystin 27,368 | Lake sites: Chl-a 20,513; Secchi 26,246 | US | `MonitoringLocationIdentifier`, lat/lon | USGS data public domain (cite doi:10.5066/P9QRKUVJ) | Open API | – |
| **EPA NLA 2007/2012/2017/2022** [D31] | Chl-a, Secchi, turbidity, colour, DOC, **microcystin**, cylindrospermopsin, phytoplankton counts | 2017: 1,210 visits / 1,113 lakes. 2022: 1,225 visits / 1,129 lakes | Probability sample of US lakes | Summer | `SITE_ID`, `UID`, lat/lon | Public domain (EPA) | Open CSV | Small |
| **LAGOS-US** LIMNO (edi.1439.5) / LOCUS (edi.854.1) / LANDSAT (edi.1427.1) [D16]–[D19] | 47 limnological variables; lake IDs and geometry; Landsat lake estimates | LIMNO: 3,511,020 observations, 20,329 lakes. LOCUS: 479,950 lakes ≥1 ha. LANDSAT: 723,206 matchup records (window "up to 7 calendar days") | CONUS | 1975–2021 | `lagoslakeid` | CC-BY | EDI / DataONE | 2.4–20.8 GB |
| **CAML** (cyanobacteria abundance; the Tick Tick Bloom labels) on SeaBASS [D36] | Cyanobacteria cells/mL → severity 1–5 | **23,570 points** | CONUS | 2013-06-13 → 2021-10-18 | lat/lon, `uid`, `region`, `data_provider` | SeaBASS: "all SeaBASS data are publicly available" (doi:10.5067/SeaBASS/CAML/DATA001) | Open | Small |
| NOAA GLERL/CIGLR western Lake Erie HAB monitoring [D40] | Extracted Chl-a and phycocyanin, microcystin, TSS, CDOM, turbidity, Secchi, CTD | Weekly May–Oct since 2012 | WLE stations | 2012– | (unverified) | (unverified) | NCEI (doi:10.25921/11da-3x54) | – |
| UNEP GEMStat (open subset) [D32] / GRQA v1.4 [D33] / EEA Waterbase [D34] | Mostly river chemistry; few optical variables | >50 M open entries / >17 M / – | 22,982 open stations (91 countries) / global / Europe | 1906–2025 / 1898–2023 / 1899–2025 | Station metadata | CC-BY-4.0 (open parts) | Zenodo / EEA (open) | 0.2–3 GB |
| SLU Miljödata-MVM (Sweden) [D45]; UK EA Water Quality [D46]; Canada ECCC/DataStream [D47] | Lake/stream chemistry and biology | MVM: 1,376,904 samples | National networks | – | Station IDs | MVM licence not found; UK OGL v3; Canada OGL | Open APIs | – |
| Valente et al. v3 [D39]; AERONET-OC V3 [D37]; HYPERNETS [D38] | Bio-optical (mostly ocean); water-leaving radiance (41 L2 sites, a few inland); hyperspectral | 148,432 rows / – / – | Global | 1997–2021 / – / – | lat/lon | CC-BY-4.0 / public / (unverified; site unreachable) | Open | – |
| LIMNADES (Stirling) [D21] | Bio-optical + WQ | Unverified | ">250 aquatic systems" (Spyrakos 2018) | – | – | **Not open access**; site does not resolve | – | – |
| HAB event sources: NY DEC HABs [D43]; California FHAB [D44]; CDC OHHABS [D42]; NOAA HABSOS [D41] | Bloom reports and advisories; lab results; event summaries; *K. brevis* counts (marine) | NY: 1,371 rows (2019–). OHHABS: summaries only | State / national | – | lat/lon (NY, CA) | CA: public domain; NY: not stated | Open (Socrata / CKAN) | Small |

**B. Satellite-derived products (weak labels or baselines, never ground truth)**
- **ESA Lakes CCI v3.0.0** (published 2026-03-18; DOI 10.5285/a56dba09df8a42ec9fba8b8c7a5e1f69) [D22] [D23].
  - Content: 2,024 lake IDs; 1992–2023; daily; 1/120°.
  - Variables: LSWT, ice, level, extent, Rw400–1020, Chl-a, TSM, a<sub>CDOM</sub>, K<sub>d</sub>, and **phycocyanin "from MDN algorithm by O'Shea et al. 2021"** (a leakage risk if MDN is a baseline). There is no turbidity variable in the v3 files.
  - Volume and terms: about 1 TB; terms say it "may be used by any user for any purpose" (acknowledgement required).
- **EPA CyAN** [D24] [D25].
  - Product: CI<sub>cyano</sub> from OLCI 300 m (2016–) and MERIS (2002–2012); Sentinel-2 has also been used since 2024. Covers ">2,000 of the largest" US lakes.
  - Access: an **unauthenticated API works**. `https://cyan.epa.gov/waterbody/search/?name=Okeechobee` and `.../data/?OBJECTID=...` returned weekly 256-bin histograms.
  - Conversion: CI<sub>cyano</sub> = 10^(DN·0.011714 − 4.1870866), about 10⁴–7×10⁶ cells/mL. DN 0 = below detection, 254 = land, 255 = no data.
  - NASA OB.DAAC file downloads redirect to an Earthdata login.
- **CLMS Lake Water Quality** (100 m S2 and 300 m OLCI) [D49].
  - Layers: `chlamean`, `tsmmean`, `tmean`, `tsi`, `fcbprob` and reflectances. The layer list was verified by me in the CDSE STAC; downloads need a CDSE account.
- **LimnoSat-US** [D15]: Landsat SR for 56,792 lakes, 1984–2020, CC-BY-4.0, 3.6 GB.

**C. Water masks and lake geometry**
- **HydroLAKES v1.0** [D51]: 1.4 M lakes ≥10 ha, `Hylak_id`, CC-BY-4.0, ~0.8 GB.
- **JRC GSW** [D52]: Earth Engine `JRC/GSW1_4`, 1984–2021; MPC `jrc-gsw`, 1984–2020, layers occurrence, change, seasonality, recurrence, transitions, extent (chip read verified by me). Licence: "free of charge, without restriction of use".
- **GLWD v2** [D53]: ~500 m, 33 classes, CC-BY-4.0.
- **SWOT Prior Lake Database** [D54]: ~6 M lakes ≥1 ha; PO.DAAC, Earthdata login usually required.
- **ESA WorldCover** 2020/2021 [D55]: 10 m, class 80 = permanent water, CC-BY-4.0, on MPC.
- Also GLAKES (3.4 M lakes) [D56] and GLCP [D57].

**Water Quality Portal: practical facts (verified with real queries by the sub-survey).**
- **The legacy WQX 2.2 endpoints silently omit USGS data added after 2024-03-11.** For Wisconsin Chl-a in July 2024, the legacy endpoint returned 404 rows with 0 USGS rows; WQX3 returned 429 rows with 25 USGS rows. Use `/wqx3/Result/search` (beta profiles) or the USGS Samples API for 2024+. The WQP counts in §b.1 came from the legacy endpoint, so their USGS share after March 2024 is undercounted.
- **characteristicName values:**
  - Chl-a: "Chlorophyll a"; "…, corrected for pheophytin"; "…, uncorrected for pheophytin"; "Chlorophyll a (probe)"; "… (probe relative fluorescence)"; "Chlorophyll a, free of pheophytin".
  - Phycocyanin: "Phycocyanin"; "Phycocyanin (probe)"; "… (probe relative fluorescence)".
  - Suspended solids: "Total suspended solids"; "Suspended Sediment Concentration (SSC)".
  - Turbidity: "Turbidity"; "Turbidity Field".
  - Secchi: "Depth, Secchi disk depth"; "Water transparency, Secchi disc".
  - Dissolved organic matter: "Colored dissolved organic matter (CDOM)"; "Fluorescent dissolved organic matter (FDOM)".
  - Temperature: "Temperature, water".
- **QA filters** (EPA TADA functions [D29]):
  - Drop QC activities; keep Final/Accepted results.
  - Handle censored "Not Detected" values; harmonize units.
  - Keep surface samples only; drop continuous-sensor rows.
  - Remove cross-organization duplicates; convert times to UTC.
  - Keep lab-extracted Chl-a separate from probe data.

**Biases and leakage traps.**
- **Geography.** The data are US-heavy (GLORIA 44% US; AquaSat 71% from FL/WI/MN) and skewed toward eutrophic, heavily sampled sites.
- **Duplicates across datasets:**
  - LIMNO = WQP + NLA + NEON.
  - AquaSat v1 = WQP + LAGOS-NE.
  - AquaMatch = WQP.
  - AquaCorr merges all of these plus GLORIA and SeaBASS (85,825 rows shared between AquaMatch and AquaSat v1).
  - GLORIA contains SeaBASS (1,206 rows) and LIMNADES (307 rows) samples.
- **De-duplication rule:** (~100 m, date, variable, value), then assign HydroLAKES/LAGOS IDs before splitting.
- **Published-model overlap.** The MDN development set shares most contributors with GLORIA (inferred). Lakes CCI phycocyanin is itself an MDN output.
- **Centroid coordinates.** AquaCorr's centroid rows must be dropped from pixel-level matchups.
- **Ready-made, no-account options.** AquaCorr-Global (matchups with ACOLITE Rrs) and CAML/SeaBASS (HAB labels) remove two big bottlenecks.


**Volume check on the Water Quality Portal** (my HTTP HEAD counts from the *legacy* `waterqualitydata.us/data/Result/search` endpoint, which misses USGS data added after 2024-03-11; siteType "Lake, Reservoir, Impoundment" unless noted; 2017-01-01 to 2024-12-31; run 2026-10-22):

| characteristicName | Sites | Results |
|---|---|---|
| Chlorophyll a | 6,802 | 115,762 |
| Chlorophyll a, corrected for pheophytin | 11,552 | 156,173 |
| Chlorophyll a, uncorrected for pheophytin | 7,628 | 100,183 |
| Chlorophyll a (probe relative fluorescence) | 2,043 | 83,592 |
| Depth, Secchi disk depth | 23,729 | 465,747 |
| Turbidity | 12,645 | 364,614 |
| Total suspended solids | 9,285 | 140,235 |
| Phycocyanin (lab) / Phycocyanin (probe RFU) | 393 / 170 | 4,683 / 25,673 |
| Microcystin | 2,881 | 26,348 |
| Colored dissolved organic matter (CDOM) | 156 | 2,016 |
| *Stream:* Turbidity / TSS / Chlorophyll a | 39,160 / 39,927 / 8,289 | 1,011,307 / 772,463 / 126,045 |
| *Estuary:* Chlorophyll a | 1,278 | 36,930 |

Results include depth profiles and duplicate characteristic names for the same sample. Unique surface site-days are fewer: 0.85× in the Minnesota sample, and lower for sonde profiles.

---

## (c) Gaps where a SOTA or first-in-class claim is realistic

| # | Gap | Why it is open (evidence) | Claim SENTINEL could make | Realism |
|---|---|---|---|---|
| G1 | **Satellite-matchup retrieval on held-out lakes and held-out years** (S2 C1 L2A; L8/9; OLCI) | Real-imagery MdSA is 60–200% for in-situ-trained MDN with physics AC (Saranathan 2024: Chl ≈195%, TSS 110%; Balasubramanian 2025: 60–75%). Matchup-trained models (MDN<sub>B</sub>, XGBoost) exist but use random or partly random splits and no retrained-MDN baseline (Smith 2021; Sherf 2025; Topp 2021) | "On a public, leakage-free S2/Landsat matchup benchmark (unseen lakes, unseen years), our model reduces MdSA by X% vs. the best of: retrained MDN (in-situ and matchup-trained), GBM, operational products, and metadata-only baselines" | **High.** This is the main target. AquaCorr-Global (Aug 2026) supplies matchups but defines no splits, baselines or leaderboard. Re-check the literature for new AquaCorr-based MDN papers before claiming "first" |
| G2 | **Within-lake temporal skill** (does the satellite see change, not just place?) | No paper reports de-meaned or within-lake metrics. Location and season baselines are strong (Tick Tick Bloom KNN; Topp static covariates; SENTINEL's own DOY+lat R² 0.87). **Measured here** on the Minnesota 2023 WQP sample (5,665 site-days, 1,012 sites): an oracle *site mean* explains **74%** of log10 Chl-a variance. Within-site SD is only **0.28 log10** (×1.9), smaller than published satellite RMSLE (0.5–1.0) | First benchmark with a within-lake anomaly metric (Spearman ρ of de-meaned log values; anomaly AUROC) | **High, first-in-class.** Cheap to add |
| G3 | **Calibrated uncertainty on real imagery** | MDN Chl interval coverage drops from 78.5% (in situ) to 47.8% on ACOLITE MSI matchups. BNN-MCD drops to 42.0% (Saranathan 2024, Tables 4 and 7) | Prediction intervals with near-nominal coverage (PICP) on held-out-lake satellite matchups | **Medium–high** |
| G4 | **Small-lake HAB detection at 10–20 m** | CyAN CI<sub>cyano</sub> is 300 m OLCI and covers ~2,000+ large US lakes (Coffer 2020/2021). Its validation is per-lake same-day (OA 84%; TPR 69% vs advisories). CLMS `fcbprob` exists but has little published skill. The Tick Tick Bloom winner used a location-only KNN in the South and West | S2-based bloom and exceedance probability for lakes smaller than CyAN resolves, evaluated on held-out lakes against CI<sub>cyano</sub> (where both exist), CLMS, NDCI/MCI thresholds and KNN-location | **Medium.** Labels are sparse and biased (sampling happens when blooms are visible) |
| G5 | **Plug-and-play multi-sensor fusion** (S2 + L8/9 + OLCI + metadata, any subset) | Published models are single-sensor. MDN harmonizes via separate per-sensor networks | One wavelength-token model that degrades gracefully when bands or sensors are missing | **Medium**, and matches SENTINEL goal (2) |
| G6 | **RGB+NIR drone student distilled from the satellite teacher** | No public study quantifies what a 4-band RGB+NIR camera can retrieve vs. a full S2 teacher on held-out lakes | "Distillation gap" per variable, plus a deployable student | **High, first-in-class** (with honest negative results for PC and CDOM) |
| G7 | **EO foundation models for inland WQ** | Only these were found: an ocean OLCI foundation model (Dawson 2025), coastal-bloom ViTs (Lima & Martins 2026), an unbenchmarked water foundation model (Hydro) and a HAB self-supervised framework with no reported numbers (SIT-FUSE). There is no leakage-free inland WQ evaluation | First systematic test of frozen and fine-tuned EO foundation models vs. a spectral model on G1 | **Medium.** Likely result is "modest or no gain", which is still publishable |

**Not realistic, or not worth claiming:**
- Beating MDN by a wide margin on GLORIA random splits (saturated and noise-limited; target a tie to small gain and treat it as a sanity check).
- Inland oil or chemical spill detection (no labelled inland data; marine SAR results do not transfer).
- Direct detection of hypoxia, fish kills, salinization or sewage from optical imagery. Use satellite outputs only as *features* for the fusion and forecast models.
- PACE-based inland products (1.2 km pixels).

---

## (d) Recommended approach

### d.1 Benchmark: "SENTINEL-SatWQ v1" (four tracks; freeze and pre-register the protocol before training)

**Track R — in-situ radiometry (comparability with the MDN literature).**
- Data: GLORIA. Resample Rrs to MSI, OLI and OLCI with the RSR files shipped in MDN-STREAM. Also resample to the drone camera bands (§d.7).
- Splits:
  - R1: 50/50 random. Only for reproducing published numbers (target: MDN MSI Chl MdSA ≈30%).
  - R2: 5-fold GroupKFold by water body.
  - R3: leave-one-dataset-out, using GLORIA source IDs as in Saranathan 2024.
- Retrain every baseline on every fold. Report MdSA, SSPB, slope, RMSLE, MAE(log), Bias(log) and N, broken down by trophic class and optical water type.

**Track S — satellite matchups (the main SOTA arena).**
- Imagery: Earth Search `sentinel-2-c1-l2a` (primary; consistent baseline), MPC `landsat-c2-l2` (L8/9 SR+ST; L5/7 only with harmonization), MPC `sentinel-3-olci-wfr-l2-netcdf` (large lakes). Optional: Earth Search `sentinel-2-l1c` for a TOA-input variant. AquaCorr already ships ACOLITE Rrs, so you do not need to run ACOLITE.
- In situ: **start from AquaCorr-Global** (CC-BY-4.0). It already provides same-day site–date matchups and ACOLITE Rrs, which makes the MDN baseline exactly reproducible. For each AquaCorr site-date (minus centroid-coordinate rows), extract SENTINEL's own S2 C1 L2A / Landsat chips. Extend to 2024–2025 with WQP **WQX3** pulls (the legacy endpoints miss post-2024-03 USGS data). Add EPA NLA 2017/2022 as an independent probability-sample test set and non-US open sets (§b.2) for transfer tests. Deduplicate across sources (~100 m, same day, same value). Keep one surface record (≤2 m) per site-day, using the median of replicates. Drop QC samples. Keep method covariates (corrected vs uncorrected Chl-a, HPLC vs fluorometric). Exclude probe RFU from the main targets.
- Matchup rules:
  - Time: **same day** (primary; matches MDN papers for inland waters), with a ±1-day secondary set; ±3 days for Secchi and a<sub>CDOM</sub> sensitivity only.
  - Space: 3×3 window at 20 m (S2), 30 m (Landsat) or 300 m (OLCI), with ≥5 valid water pixels (Pahlevan 2020; Werdell & Bailey 2005 median protocol).
  - Shore buffer: ≥2 pixels from land (JRC occurrence ≥90%). Schaeffer 2018 shows land-adjacent pixels are 3–4× worse for temperature.
  - Screening: drop cloud, shadow, cirrus, snow and ice (SCL / QA_PIXEL / WQSF). Add a glint screen (SWIR B11/B12 reflectance threshold; the SWIR-based sunglint correction of Harmel et al. 2018 as an optional fix). Keep the 3×3 coefficient of variation as a quality covariate.
- Splits (all grouped, all reported):
  - S1, **unseen lakes**: GroupKFold(5) on water-body ID (HydroLAKES `Hylak_id` / NHD COMID; river reaches grouped by HUC10), with groups assigned to folds by **spatial blocks** (~50–100 km or HUC8) so near-neighbour lakes do not straddle folds. Large lakes with many stations (e.g., Lake Erie) count as *one* group.
  - S2, **unseen region**: leave-one-EPA-Level-II-ecoregion (or HUC2) out.
  - S3, **future**: train 2016–2022, validate 2023, test 2024–2025.
  - S1∩S3, **strictest**: unseen lakes in future years. This is the headline number.
- Metrics: the Track R set plus R² (log space), skill relative to the best metadata-only baseline, and uncertainty metrics (PICP/coverage at 68% and 95%, sharpness, CRPS).

**Track T — within-lake temporal skill (anti-climatology).**
- Use lakes with ≥8 test matchups.
- Report Spearman ρ between predicted and observed log values after subtracting each lake's *training-period* mean (for unseen lakes: the mean of that lake's own test observations, used for scoring only).
- Report AUROC for "observation above the lake's 75th percentile".
- A model that only knows location and season scores ≈0 here, so this track defuses the DOY+latitude trap directly.

**Track H — HAB / exceedance detection.**
- Labels:
  - Chl-a exceedance at fixed thresholds (e.g., 10, 25 and 50 µg L⁻¹; thresholds to be justified from WHO/EPA documents: **unverified here**).
  - Cyanobacteria density classes from **CAML on SeaBASS** (23,570 points, 2013–2021; Tick Tick Bloom thresholds 20k / 100k / 1M / 10M cells mL⁻¹). Reuse its geographic regions as one held-out split.
  - Microcystin detection or exceedance (NLA and WQP).
  - State advisories (Whitman 2022 design).
- Same S1/S3 splits.
- Metrics: AUROC, AUPRC, F1 at fixed thresholds, **recall at 5% and 10% FPR**, Brier score, and region-averaged scores (Tick Tick Bloom style).
- Event-level metrics: share of advisory periods detected within ±7 days, and false alarms per lake-season outside advisories.
- **Placebo test:** shuffle imagery among matchups of the same lake and month. If the score does not drop, imagery is not being used. This is the test SENTINEL's previous detector failed.

### d.2 Baselines to beat (all retrained on identical folds and inputs)

1. **Metadata-only:**
   - (a) global and per-ecoregion median;
   - (b) GBM on lat/lon, DOY (sin/cos), elevation, lake area, depth, ecoregion and catchment land use (HydroLAKES/LAGOS), *no imagery*;
   - (c) **KNN on location** (+ season), as used by the Tick Tick Bloom winner;
   - (d) the old SENTINEL "DOY + latitude" model.
2. **Index and band-ratio algorithms on the same L2A pixels:**
   - Chl-a: OC3, NDCI, Gons, Gurlin/Gilerson 2- and 3-band, Moses, Blend, MCI, FAI;
   - TSS: Nechad, SOLID;
   - turbidity: Dogliotti;
   - Secchi: QAA-Lee 2015;
   - a<sub>CDOM</sub>: Ficek;
   - PC (OLCI only): Simis 2005.

   These are already coded in MDN-STREAM. Calibrate their coefficients on training folds ("-opt" versions, as in Werther 2022).
3. **MDN, three ways:**
   - (a) official MDN-STREAM weights on (i) **AquaCorr's ACOLITE Rrs** (no need to run ACOLITE yourself) and (ii) Sen2Cor L2A/π, with GLORIA overlap disclosed;
   - (b) MDN retrained on GLORIA folds that exclude test lakes, then applied to the same AC'd imagery;
   - (c) **MDN<sub>B</sub>-style**: the MDN architecture retrained on *our* L2A matchups. This is the fairest "strongest published method" comparison on Track S.
4. **GBM** (LightGBM/XGBoost) on 3×3 medians of all bands, indices, angles and AOT, with and without metadata (AquaSat/Topp/Sherf style).
5. **Operational products** at the same matchups:
   - OLCI WFR `chl-nn`, `chl-oc4me`, `tsm-nn` (MPC);
   - CLMS LWQ 100 m / 300 m (`chlamean`, `tsmmean`, `tmean`, `fcbprob`; CDSE account);
   - ESA Lakes CCI Chl-a and turbidity;
   - EPA CyAN CI<sub>cyano</sub> and DrivenData's open-source **CyFi** cyanobacteria estimator (Track H);
   - NASA STREAM MDN maps where available.
6. **Foundation-model features** (frozen embeddings + linear/GBM head) and **pixel-embedding products** (see §a.6). These test whether generic EO representations help.

### d.3 Leakage-control checklist (enforce in code; fail the run if violated)

- No water body (or HUC10 river reach) appears in both train and test. Spatial blocks keep neighbouring lakes together.
- Cross-source duplicates are removed before splitting: WQP ↔ LAGOS ↔ NLA ↔ AquaMatch, and GLORIA ↔ WQP for coincident samples.
- All lake-level statistics (climatologies, lake means, normalization constants) come from training folds and training years only.
- Hyperparameters are tuned on inner validation folds only. The test set is evaluated once, with a logged hash of the split file.
- The official MDN weights were trained on data overlapping GLORIA, so they are never scored on GLORIA as if independent.
- Processing baseline: use S2 C1 throughout. If MPC is used, harmonize by baseline and test that residuals do not depend on processing baseline or acquisition year.
- Report imagery-only, metadata-only and both, plus the placebo test (Track H).
- Multiple stations in one lake on one date stay in one fold (they share an image and a bloom).
- Attribution: every reported number names the exact model that produced it (SatWQ-Net vs GBM vs a frozen backbone + head). Never report an off-the-shelf backbone's score, or a GBM's score, as "the SENTINEL model". That is the failure mode of the previous ImageNet-ViT result.

### d.4 Architecture likely to win: "SatWQ-Net"

1. **Inputs (any subset):**
   - S2 C1 L2A chip, 32×32 px at 20 m (640 m), all 12 L2A bands (no B10), DN→reflectance by baseline;
   - SCL one-hot, JRC occurrence and distance to shore;
   - sun and view angles (glint geometry), AOT, WVP;
   - optional Landsat 8/9 chip (SR + ST);
   - optional OLCI 5×5 window (16 bands);
   - Δt (image minus sample);
   - metadata: 3-D unit vector of lat/lon, DOY sin/cos, HydroLAKES area, depth and elevation.
2. **Spectral pixel encoder (shared across sensors).** Each band becomes a token: reflectance value (asinh or log scaled) plus a sinusoidal embedding of centre wavelength and FWHM plus a sensor embedding. A 4-layer, 64-dimensional transformer runs over a pixel's band tokens. Missing bands simply drop out. This is the same idea as wavelength-conditioned patch embeddings in DOFA, Copernicus-FM and Clay, and it is what makes S2→drone distillation and multi-sensor fusion plug-and-play.
3. **Spatial aggregator.** Masked attention pooling over valid water pixels (centre-weighted), plus a 3-layer CNN on pixel embeddings for context (shoreline adjacency, plumes, scums, cloud edges).
4. **Metadata branch with modality dropout** (p ≈ 0.5), so the network cannot rely on location alone. Also train and report an imagery-only variant.
5. **Probabilistic multi-target head.** MDN-style Gaussian mixture (K = 5) in log10 space for Chl-a, TSS, turbidity, Secchi, a<sub>CDOM</sub> and PC, plus a Gaussian for temperature (Landsat ST input). The loss is a masked negative log-likelihood, so missing targets are skipped (MDN 2022 showed simultaneous retrieval helps). Exceedance probabilities for Track H come from the predictive distribution, or from a small auxiliary head.
6. **Pretraining:**
   - (i) Pretrain the spectral stem on GLORIA Rrs → WQ (Track R folds only), with noise augmentation drawn from observed Rrs-error statistics (e.g., ACIX-Aqua-scale 20–60% band errors).
   - (ii) Self-supervised masked-band and masked-pixel reconstruction on about 10⁶ unlabelled S2 water pixels and chips (cheap, label-free).
7. **Ensemble** of 5–10 seeds (median), as MDN does. This also gives epistemic uncertainty.
8. **Compute.** About 1–5 M parameters. Trains in hours on one RTX 2080 Ti with fp16 AMP. A frozen foundation-model branch, if used, is precomputed once per chip.

**Why it should beat the baselines.**
- (i) End-to-end training on matchups absorbs AC bias. MDN<sub>B</sub> > MDN<sub>A</sub> (Smith 2021) and XGBoost on L2A > MDN on ACOLITE (Sherf vs Saranathan, though on different data).
- (ii) Multi-target heads regularize (MDN 2022).
- (iii) Spatial context helps with adjacency and glint, which ACIX-Aqua names as unsolved.
- (iv) Calibrated mixtures address G3.

**Main risk.** A well-tuned GBM on 3×3 medians may match it on Chl-a. Plan the ablations (spectral-only MLP, +spatial, +metadata, +FM) so that whichever component wins is reported honestly.

### d.5 Realistic matchup counts

**Verified counts** (AquaCorr-Global, same-calendar-day, QC-passing, from the dataset sub-survey):

| Sensor | Chl-a (all methods) | Chl-a (strict lab-extracted) | TSS | Secchi | Turbidity | a<sub>CDOM</sub> |
|---|---|---|---|---|---|---|
| Sentinel-2 (2015–2024) | 113,720 | **23,370** at 5,655 sites | 121,281 | 79,368 | 92,007 | 238 |
| Landsat 8/9 (2016–2024) | 37,139 | 7,856 | 38,406 | 30,221 | 27,813 | 171 |

These counts include rivers and all data sources. Expect a further 20–50% loss after shore buffers, glint and reflectance-sanity filters, de-duplication and dropping centroid coordinates.

**Estimates for US lakes only** (sub-survey estimates from WQP counts × AquaCorr's empirically observed yields: S2 Chl-a same-day 11.2%, Secchi 7.5%; Landsat 8/9 4.4% / 3.0%). My Minnesota 2023 measurement (14.3% same-day, 36.9% ±1 day) is consistent with them.

| Variable (US lakes, 2016–2025) | S2 same-day | S2 ±1 day | L8/9 same-day | Notes |
|---|---|---|---|---|
| Chl-a (all methods) | ~37k | 56k–93k | ~15k | Lab-extracted is only ~20–25% of this |
| Secchi | ~41k | 61k–100k | ~16k | Slow-varying; ±3 days acceptable |
| Turbidity | ~29k | 44k–74k | ~12k | NTU vs FNU: keep the unit as a covariate |
| TSS | ~12k–18k | 18k–45k | ~5k–7k | Many more in wide rivers |
| Microcystin (HAB label) | ~2k–3k | 3k–7.5k | ~1k | Detection limits vary |
| Phycocyanin (mostly probe RFU) | ~1.7k–2.6k | 2.6k–6.5k | ~1k | RFU is not comparable across sondes |
| CDOM / fDOM | ~140–210 | 210–520 | <100 | Too few: use GLORIA plus multi-task transfer |
| NLA 2017 + 2022 (2,435 visits) | ~180–270 | 270–680 | – | Independent probability-sample test set |
| Water temperature (Landsat ST) | – | – | ≫10⁴ | S2 has no thermal band |

### d.6 Expected difficulty and honest targets

- **Track R:** matching a retrained MDN (±2 MdSA points) is likely. Beating it by more than 5 points on R2/R3 is uncertain. Report it as a sanity check, not as the SOTA claim.
- **The floor to beat (measured here).** On the Minnesota 2023 WQP Chl-a sample, inverse-distance KNN on *location only* (k = 8; script `scripts/knn_location_baseline.py`) scores as follows (log10 units):

  | Holdout | R² | RMSLE | MdSA |
  |---|---|---|---|
  | Held-out site | 0.34 | 0.446 | 87% |
  | Held-out 2-km blocks | 0.32 | 0.453 | 86% |
  | Held-out 25-km blocks | 0.17 | 0.503 | 103% |
  | Held-out 100-km blocks | 0.17 | 0.501 | 109% |

  The drop between 2 km and 25 km blocks is near-neighbour leakage; it is why the splits must be blocked spatially. A satellite model that cannot beat MdSA ≈ 100% on spatial blocks adds nothing over location.
- **Mini-pilot (does imagery add skill?).** Data: 844 same-day S2 C1 L2A matchups at 570 Minnesota lake sites, May–Oct 2023. Features: medians of 11 L2A bands over a ~60–80 m window, plus 5 band ratios. Split: 5-fold CV over **25-km spatial blocks**, untuned scikit-learn HistGradientBoosting models (`scripts/pilot_matchups.py`, `scripts/pilot_eval.py`; output in `research_samples/satellite/wqp_mn_2023/pilot_eval_output.txt`).

  | Model (log10 Chl-a) | R² | RMSLE | MdSA | SSPB |
  |---|---|---|---|---|
  | KNN location only | 0.29 | 0.463 | 98% | +6% |
  | GBM location + season (no imagery) | 0.22 | 0.487 | 101% | +7% |
  | NDCI linear (calibrated per fold) | −0.16 | 0.591 | 149% | +24% |
  | **GBM imagery only** | **0.49** | **0.391** | **73%** | +1% |
  | GBM imagery + season | 0.53 | 0.378 | 64% | +5% |
  | GBM imagery + location + season | 0.60 | 0.348 | 58% | −2% |

  Within-lake skill (imagery-only GBM, sites with ≥3 matchups: 61 sites, 196 observations): Spearman ρ of de-meaned observed vs predicted = **0.33** (p = 1.8×10⁻⁶).

  Caveats: one state, one season, small N, untuned, and no glint, adjacency or reflectance-sanity filters. 22 of 844 matchups had blue reflectance >0.15 (missed cloud or haze); 111 had SWIR1 >0.05; 11 had negative visible reflectance. So SCL alone is not enough. Even so, the pilot shows (i) imagery adds real skill beyond location under spatial blocking, (ii) a single-index algorithm is worse than location, and (iii) **a plain GBM sets a high bar (MdSA ≈ 58–73%)** that SatWQ-Net must beat.
- **Track S (S1∩S3):** no clean published reference exists. Published real-imagery Chl-a numbers range from RMSLE ≈0.5–0.6 (XGBoost on L2A, partly random splits) to MdSA ≈195% (in-situ MDN on ACOLITE). A plausible goal is Chl-a MdSA ≈45–55% on unseen lakes (the pilot GBM reached 58–73%), TSS and turbidity ≈40–60%, and Secchi MAPE ≈30–40%, with ≥15% relative improvement over the best retrained baseline. These are hypotheses to test, not promises.
- **Track T:** expect modest within-lake ρ (0.3–0.6) for Chl-a and higher for turbidity and Secchi. Any positive, significant value above the metadata baseline (≈0) is a real result.
- **Track H:** expect recall at 10% FPR well below CyAN's same-day per-lake accuracy, because the task is harder (small lakes, unseen lakes). The claim should be relative to location and season baselines and to CI<sub>cyano</sub> on the overlap set.

### d.7 Teacher outputs for the RGB+NIR drone student

- **Band correspondence.** S2 B2 (490 nm), B3 (560), B4 (665) and B8 (842) / B8A (865) (MPC band metadata) roughly match the drone's RGB and NIR. For example, DJI Mavic 3 Multispectral has G 560±16, R 650±16, RE 730±16 and NIR 860±26 nm, plus a 20 MP RGB camera (ag.dji.com/mavic-3-m/specs). Consumer RGB Bayer filters are broad and overlapping, so measure or obtain their SRFs.
- **What to distill** (the teacher is SatWQ-Net with all S2 bands):
  - (1) per-pixel predictive **mean and variance** (log space) for turbidity, TSS, Secchi and Chl-a;
  - (2) **exceedance and bloom probabilities** (Track H);
  - (3) quality flags (glint, adjacency, cloud shadow) as auxiliary targets;
  - (4) optionally intermediate pixel embeddings (same token encoder, so features align).

  The student is the *same network with only B2/B3/B4/B8 tokens*. Train it on millions of unlabelled S2 water pixels with teacher soft targets (KL or NLL to the teacher mixture; Hinton et al. 2015, arXiv:1503.02531), then fine-tune and evaluate on **real in-situ matchups in held-out lakes**. Never evaluate by agreement with the teacher alone.
- **Use GLORIA to simulate the exact drone bands.** Convolve its 1-nm Rrs with the drone SRFs. This gives an in-situ "drone-band" training and evaluation set (Track R for the student) with no flights needed.
- **Expected retrievability with RGB+NIR:**
  - turbidity, TSS and Secchi: good (red and NIR);
  - surface scum and bloom presence: good (NIR jump over the red);
  - Chl-a: moderate (no 705 nm red edge, which hurts eutrophic retrieval; OLI's lack of 700–800 nm bands gave 59% vs 29% MdSA in MDN 2022);
  - CDOM: weak;
  - phycocyanin: not retrievable (no 620 nm);
  - temperature: impossible without a thermal camera.

  Report this "distillation gap" table. It is itself a first-in-class result.
- **Domain gap to handle on the drone side:** reflectance calibration (panel or irradiance sensor), sun-glint and sky-reflection removal, and aggregation to ≥1–10 m before inference. Validate with a few same-day drone + S2 + in-situ campaigns if the team can fly.

---

## (e) Risks and fallbacks

| Risk | Likelihood | Mitigation / fallback |
|---|---|---|
| The deep model does not beat GBM on Track S | Medium–high | Report honestly. The contribution then shifts to the benchmark, Track T/H skill, calibrated uncertainty, multi-sensor fusion and distillation. Ensembling GBM and NN is allowed if declared |
| Spatial or temporal leakage inflates results (as in the old project) | High if unguarded | §d.3 checklist in code; S1∩S3 headline; placebo tests; external review before any claim |
| Location and season baselines win (DOY+lat trap) | High for HAB, medium for Chl-a | Track T metrics; modality dropout; report imagery-only models; claim only what beats metadata-only |
| Fewer matchups than estimated (clouds, strict QA, narrow rivers) | Medium | Pilot one state first; add ±1 day, Landsat 8/9 and OLCI; pool lakes and rivers; add non-US open sets |
| Label noise (Chl methods, depth, lab bias, RFU) | High | Method covariates; robust NLL; replicate-based noise ceiling; exclude RFU from headline targets |
| S2 radiometric inconsistencies (baseline offset; S2A/B/C; L5/7 vs L8/9 +25% bias) | Certain if ignored | Use `sentinel-2-c1-l2a`; sensor ID as a covariate; no L5/7 without harmonization (Maciel 2023) |
| AC errors, glint and adjacency in small lakes | High | End-to-end training; shore buffer; glint flag; learn from context; report by lake size |
| HAB labels biased (sampling only when blooms are visible) | High | Use NLA (probability survey) as an unbiased test subset; advisories as event labels; report by data source |
| Foundation-model input mismatch (L1C vs L2A, band sets, 224 px input) | Medium | Wavelength-agnostic models; frozen features; keep foundation models as ablations, not the core |
| Data access changes (MPC; requester-pays) | Low–medium | Cache chips; Earth Search mirror for S2; MPC for Landsat |
| Licences and data provenance (LIMNADES not open; CLMS/Copernicus terms; AquaCorr-Global new and not peer-reviewed) | Medium | Use CAML (SeaBASS, public) instead of the competition files. Cite every licence. Audit AquaCorr against its sources (WQP/AquaMatch/GLORIA) on a sample before relying on it |
| WQP legacy endpoint misses USGS data after 2024-03-11 | Certain if ignored | Use WQX3 / USGS Samples API for 2024+. De-duplicate NWIS vs USGS provider codes |
| Future overlap of the benchmark with MDN training data (AquaCorr comes from the NASA GSFC group) | Medium | Freeze and hash the test split; when new MDN versions appear, retrain them on your folds instead of using their weights |

---

## (f) References

**How these were checked.** Each DOI was resolved on Crossref (`api.crossref.org/works/<DOI>`) or Semantic Scholar on 2026-10-22, and title, first author, venue and year were confirmed. arXiv IDs were checked on export.arxiv.org. ✓ means the full text (or the cited table) was read. *(abs)* means only the abstract was read. **unverified** means the item was found only through a search snippet.

**Mixture density networks and in-situ radiometry**
1. Pahlevan N., Smith B., Schalles J., Binding C., Cao Z., Ma R., Alikas K., Kangro K., Gurlin D., Hà N., Matsushita B., Moses W., Greb S., Lehmann M.K., Ondrusek M., Oppelt N., Stumpf R. (2020). Seamless retrievals of chlorophyll-a from Sentinel-2 (MSI) and Sentinel-3 (OLCI) in inland and coastal waters: A machine-learning approach. *Remote Sensing of Environment* 240, 111604. doi:10.1016/j.rse.2019.111604 ✓ (Tables 2–3, §3.3)
2. Pahlevan N., Smith B., Alikas K., Anstee J., Barbosa C., Binding C., et al. (23 authors) (2022). Simultaneous retrieval of selected optical water quality indicators from Landsat-8, Sentinel-2, and Sentinel-3. *RSE* 270, 112860. doi:10.1016/j.rse.2021.112860 ✓ (Tables 2–3). Open-access PDF: https://plymsea.ac.uk/id/eprint/9508/
3. Smith B., Pahlevan N., Schalles J., Ruberg S., Errera R., Ma R., Giardino C., Bresciani M., Barbosa C., Moore T., Fernandez V., Alikas K., Kangro K. (2021). A Chlorophyll-a Algorithm for Landsat-8 Based on Mixture Density Networks. *Frontiers in Remote Sensing* 1, 623678. doi:10.3389/frsen.2020.623678 ✓ (partial)
4. Saranathan A.M., Werther M., Balasubramanian S.V., Odermatt D., Pahlevan N. (2024). Assessment of advanced neural networks for the dual estimation of water quality indicators and their uncertainties. *Front. Remote Sens.* 5, 1383147. doi:10.3389/frsen.2024.1383147 ✓ (Tables 3, 4, 7; §4.4)
5. Saranathan A.M., Smith B., Pahlevan N. (2023). Per-pixel uncertainty quantification and reporting for satellite-derived chlorophyll-a estimates via mixture density networks. *IEEE TGRS* 61. doi:10.1109/TGRS.2023.3234465 *(abs)*
6. Werther M., Odermatt D., Simis S.G.H., Gurlin D., Lehmann M.K., Kutser T., et al. (11 authors) (2022). A Bayesian approach for remote sensing of chlorophyll-a and associated retrieval uncertainty in oligotrophic and mesotrophic lakes. *RSE* 283, 113295. doi:10.1016/j.rse.2022.113295 ✓ (Tables 1–2). Open-access PDF: https://plymsea.ac.uk/id/eprint/9822/
7. O'Shea R.E., Pahlevan N., Smith B., Bresciani M., Egerton T., Giardino C., Li L., Moore T., Ruiz-Verdu A., Ruberg S., Simis S.G.H., Stumpf R., Vaičiūtė D. (2021). Advancing cyanobacteria biomass estimation from hyperspectral observations: Demonstrations with HICO and PRISMA imagery. *RSE* 266, 112693. doi:10.1016/j.rse.2021.112693 ✓ (abstract numbers)
8. O'Shea R.E., Pahlevan N., Smith B., Boss E., Gurlin D., Alikas K., et al. (9 authors) (2023). A hyperspectral inversion framework for estimating absorbing inherent optical properties and biogeochemical parameters in inland and coastal waters. *RSE* 295, 113706. doi:10.1016/j.rse.2023.113706 (Crossref only)
9. Pahlevan N., Smith B., Binding C., Gurlin D., Li L., Bresciani M., et al. (7 authors) (2021). Hyperspectral retrievals of phytoplankton absorption and chlorophyll-a in inland and nearshore coastal waters. *RSE* 253, 112200. doi:10.1016/j.rse.2020.112200 *(abs)*
10. Balasubramanian S.V., O'Shea R.E., Saranathan A.M., Begeman C.C., Gurlin D., Binding C., et al. (12 authors) (2025). Mixture density networks for re-constructing historical ocean-color products over inland and coastal waters: demonstration and validation. *Front. Remote Sens.* 6, 1488565. doi:10.3389/frsen.2025.1488565 *(abs)*
11. Balasubramanian S.V., Pahlevan N., Smith B., Binding C., Schalles J., Loisel H., et al. (20 authors) (2020). Robust algorithm for estimating total suspended solids (TSS) in inland and nearshore coastal waters. *RSE* 246, 111768. doi:10.1016/j.rse.2020.111768 *(abs)*
12. Salah M., Higa H., Ishizaka J., Salem S.I. (2023). B1D-CNN hybrid for Chl-a from Sentinel-2 MSI. IGARSS 2023. Exact title, DOI and protocol **unverified**.

**Satellite-matchup and end-to-end models**
13. Ross M.R.V., Topp S.N., Appling A.P., Yang X., Kuhn C., Butman D., Simard M., Pavelsky T.M. (2019). AquaSat: A data set to enable remote sensing of water quality for inland waters. *Water Resources Research* 55, 10012–10025. doi:10.1029/2019WR024883 *(abs)*
14. Topp S.N., Pavelsky T.M., Stanley E.H., Yang X., Griffin C.G., Ross M.R.V. (2021). Multi-decadal improvement in US Lake water clarity. *Environ. Res. Lett.* 16, 055025. doi:10.1088/1748-9326/abf002 ✓ (§2.2, §3.1)
15. Cao Z., Ma R., Duan H., Pahlevan N., Melack J., Shen M., et al. (7 authors) (2020). A machine learning approach to estimate chlorophyll-a from Landsat-8 measurements in inland lakes. *RSE* 248, 111974. doi:10.1016/j.rse.2020.111974 *(abs)*
16. Sherf Y., Efrati B., Rozman G., Harel M. (2025, v2 2026). Global Chlorophyll-a Retrieval algorithm from Sentinel 2 Using Residual Deep Learning and Novel Machine Learning Water Classification. arXiv:2510.24124 ✓ (HTML v2, Tables 3/5)
17. *Pioneering Test-Time training for robust Fine-Grained chlorophyll-a retrieval from Sentinel-2: Unveiling China's lake eutrophication dynamics (2018–2025)*. *Int. J. Appl. Earth Obs. Geoinf.* (2026), https://www.sciencedirect.com/science/article/pii/S1569843226004036. Authors and numbers **unverified** (page returned 403).
18. Dawson G., Vandaele R., Taylor A., Moffat D., Tamura-Wicks H., Jackson S., Lickorish R., Fraccaro P., Williams H., Luo C., Jones A. (2025). A Sentinel-3 foundation model for ocean colour. arXiv:2509.21273 *(abs)*
19. Lima T., Martins V., et al. (2026). Landsat-Sentinel-2 Algal Bloom Mapping Using Vision Transformers: Model Description, Implementation, and Examples. arXiv:2606.17242 *(abs)*

**Atmospheric correction and sensors**
20. Pahlevan N., Mangin A., Balasubramanian S.V., Smith B., Alikas K., Arai K., et al. (36 authors) (2021). ACIX-Aqua: A global assessment of atmospheric correction methods for Landsat-8 and Sentinel-2 over lakes, rivers, and coastal waters. *RSE* 258, 112366. doi:10.1016/j.rse.2021.112366 *(abs)*
21. Warren M., Simis S., Martinez-Vicente V., Poser K., Bresciani M., Alikas K., et al. (9 authors) (2019). Assessment of atmospheric correction algorithms for the Sentinel-2A MultiSpectral Imager over coastal and inland waters. *RSE* 225, 267–289. doi:10.1016/j.rse.2019.03.018 *(abs)*
22. Maciel D.A., Pahlevan N., Barbosa C.C.F., de Moraes Novo E.M.L., Paulino R.S., Martins V.S., et al. (8 authors) (2023). Validity of the Landsat surface reflectance archive for aquatic science: Implications for cloud-based analysis. *Limnology & Oceanography Letters*. doi:10.1002/lol2.10344 *(abs)*
23. Kuhn C., de Matos Valerio A., Ward N., Loken L., Sawakuchi H.O., Kampel M., et al. (13 authors) (2019). Performance of Landsat-8 and Sentinel-2 surface reflectance products for river remote sensing retrievals of chlorophyll-a and turbidity. *RSE* 224, 104–118. doi:10.1016/j.rse.2019.01.023 (title only)
24. Vanhellemont Q., Ruddick K. (2018). Atmospheric correction of metre-scale optical satellite data for inland and coastal water applications. *RSE* 216, 586–597. doi:10.1016/j.rse.2018.07.015 (Crossref)
25. Vanhellemont Q. (2019). Adaptation of the dark spectrum fitting atmospheric correction for aquatic applications of the Landsat and Sentinel-2 archives. *RSE* 225, 175–192. doi:10.1016/j.rse.2019.03.010 (Crossref)
26. Steinmetz F., Deschamps P.-Y., Ramon D. (2011). Atmospheric correction in presence of sun glint: application to MERIS (POLYMER). *Optics Express* 19, 9783. doi:10.1364/OE.19.009783 (Crossref)
27. Harmel T., Chami M., Tormos T., Reynaud N., Danis P.-A. (2018). Sunglint correction of the Multi-Spectral Instrument (MSI)-SENTINEL-2 imagery over inland and sea waters from SWIR bands. *RSE* 204, 308–321. doi:10.1016/j.rse.2017.10.022 (Crossref)
28. Pahlevan N., Sarkar S., Franz B., Balasubramanian S.V., He J. (2017). Sentinel-2 MultiSpectral Instrument (MSI) data processing for aquatic science applications: Demonstrations and validations. *RSE* 201, 47–56. doi:10.1016/j.rse.2017.08.033 (Crossref)
29. Werdell P.J., Bailey S.W. (2005). An improved in-situ bio-optical data set for ocean color algorithm development and satellite data product validation. *RSE* 98, 122–140. doi:10.1016/j.rse.2005.07.001 (Crossref)
30. PACE OCI specifications: https://pace.oceansciences.org/oci.htm. Data access: https://pace.oceansciences.org/access_pace_data.htm ✓

**Algorithms for other indicators**
31. O'Reilly J.E., Maritorena S., Mitchell B.G., Siegel D.A., Carder K.L., Garver S.A., et al. (1998). Ocean color chlorophyll algorithms for SeaWiFS. *JGR Oceans* 103, 24937–24953. doi:10.1029/98JC02160 (Crossref)
32. Mishra S., Mishra D.R. (2012). Normalized difference chlorophyll index (NDCI). *RSE* 117, 394–406. doi:10.1016/j.rse.2011.10.016 (Crossref)
33. Matthews M.W., Bernard S., Robertson L. (2012). An algorithm for detecting trophic status (chlorophyll-a), cyanobacterial-dominance, surface scums and floating vegetation in inland and coastal waters (MPH). *RSE* 124, 637–652. doi:10.1016/j.rse.2012.05.032 (Crossref)
34. Matthews M.W., Odermatt D. (2015). Improved algorithm for routine monitoring of cyanobacteria and eutrophication in inland and near-coastal waters. *RSE* 156, 374–382. doi:10.1016/j.rse.2014.10.010 (Crossref)
35. Hu C. (2009). A novel ocean color index to detect floating algae in the global oceans (FAI). *RSE* 113, 2118–2129. doi:10.1016/j.rse.2009.05.012 (Crossref)
36. Simis S.G.H., Peters S.W.M., Gons H.J. (2005). Remote sensing of the cyanobacterial pigment phycocyanin in turbid inland water. *Limnol. Oceanogr.* 50, 237–245. doi:10.4319/lo.2005.50.1.0237 (Crossref)
37. Dogliotti A.I., Ruddick K.G., Nechad B., Doxaran D., Knaeps E. (2015). A single algorithm to retrieve turbidity from remotely-sensed data in all coastal and estuarine waters. *RSE* 156, 157–168. doi:10.1016/j.rse.2014.09.020 *(abs)*
38. Nechad B., Ruddick K.G., Park Y. (2010). Calibration and validation of a generic multisensor algorithm for mapping of total suspended matter in turbid waters. *RSE* 114, 854–866. doi:10.1016/j.rse.2009.11.022 (Crossref)
39. Lee Z., Shang S., Hu C., Du K., Weidemann A., Hou W., et al. (8 authors) (2015). Secchi disk depth: A new theory and mechanistic model for underwater visibility. *RSE* 169, 139–149. doi:10.1016/j.rse.2015.08.002 *(abs)*
40. Hu C., Lee Z., Franz B. (2012). Chlorophyll a algorithms for oligotrophic oceans: A novel approach based on three-band reflectance difference. *JGR Oceans* 117, C01011. doi:10.1029/2011JC007395 (Crossref)
41. Spyrakos E., O'Donnell R., Hunter P.D., Miller C., Scott M., Simis S.G.H., et al. (25 authors) (2018). Optical types of inland and coastal waters. *Limnol. Oceanogr.* 63, 846–870. doi:10.1002/lno.10674 (Crossref)
42. Neil C., Spyrakos E., Hunter P.D., Tyler A.N. (2019). A global approach for chlorophyll-a retrieval across optically complex inland waters based on optical water types. *RSE* 229, 159–178. doi:10.1016/j.rse.2019.04.027 (Crossref)
43. Schaeffer B.A., Iiames J., Dwyer J., Urquhart E., Salls W., Rover J., et al. (7 authors) (2018). An initial validation of Landsat 5 and 7 derived surface water temperature for U.S. lakes, reservoirs, and estuaries. *Int. J. Remote Sens.* 39, 7789–7805. doi:10.1080/01431161.2018.1471545 *(abs)*
44. Carrea L., Crétaux J.-F., Liu X., Wu Y., Calmettes B., Duguay C.R., et al. (16 authors) (2023). Satellite-derived multivariate world-wide lake physical variable timeseries for climate studies. *Scientific Data* 10, 30. doi:10.1038/s41597-022-01889-z *(abs)*
45. Seegers B.N., Stumpf R.P., Schaeffer B.A., Loftin K.A., Werdell P.J. (2018). Performance metrics for the assessment of satellite data products: an ocean color case study. *Optics Express* 26, 7404. doi:10.1364/OE.26.007404 (Crossref)
46. Morley S.K., Brito T.V., Welling D.T. (2018). Measures of model performance based on the log accuracy ratio. *Space Weather* 16, 69–88. doi:10.1002/2017SW001669 (Crossref)
47. Gardner J.R., Yang X., Topp S.N., Ross M.R.V., Altenau E.H., Pavelsky T.M. (2021). The Color of Rivers. *GRL* 48, e2020GL088946. doi:10.1029/2020GL088946 (Crossref)

**Harmful algal blooms and disasters**
48. Wynne T.T., Stumpf R.P., Tomlinson M.C., Warner R.A., Tester P.A., Dyble J., et al. (7 authors) (2008). Relating spectral shape to cyanobacterial blooms in the Laurentian Great Lakes. *Int. J. Remote Sens.* 29, 3665–3672. doi:10.1080/01431160802007640 (title)
49. Wynne T.T., Stumpf R.P., Tomlinson M.C., Dyble J. (2010). Characterizing a cyanobacterial bloom in Western Lake Erie using satellite imagery and meteorological data. *Limnol. Oceanogr.* 55, 2025–2036. doi:10.4319/lo.2010.55.5.2025 (Crossref)
50. Lunetta R.S., Schaeffer B.A., Stumpf R.P., Keith D., Jacobs S.A., Murphy M.S. (2015). Evaluation of cyanobacteria cell count detection derived from MERIS imagery across the eastern USA. *RSE* 157, 24–34. doi:10.1016/j.rse.2014.06.008 (Crossref)
51. Urquhart E.A., Schaeffer B.A., Stumpf R.P., Loftin K.A., Werdell P.J. (2017). A method for examining temporal changes in cyanobacterial harmful algal bloom spatial extent using satellite remote sensing. *Harmful Algae* 67, 144–152. doi:10.1016/j.hal.2017.06.001 (Crossref)
52. Mishra S., Stumpf R.P., Schaeffer B.A., Werdell P.J., Loftin K.A., Meredith A. (2021). Evaluation of a satellite-based cyanobacteria bloom detection algorithm using field-measured microcystin data. *Sci. Total Environ.* 774, 145462. doi:10.1016/j.scitotenv.2021.145462 *(abs)*
53. Whitman P., Schaeffer B., Salls W., Coffer M., Mishra S., Seegers B., et al. (9 authors) (2022). A validation of satellite derived cyanobacteria detections with state reported events and recreation advisories across U.S. lakes. *Harmful Algae* 115, 102191. doi:10.1016/j.hal.2022.102191 *(abs)*
54. Coffer M.M., Schaeffer B.A., Darling J.A., Urquhart E.A., Salls W.B. (2020). Quantifying national and regional cyanobacterial occurrence in US lakes using satellite remote sensing. *Ecol. Indic.* 111, 105976. doi:10.1016/j.ecolind.2019.105976 *(abs)*
55. Coffer M.M., Schaeffer B.A., Salls W.B., Urquhart E., Loftin K.A., Stumpf R.P., et al. (8 authors) (2021). Satellite remote sensing to assess cyanobacterial bloom frequency across the United States at multiple spatial scales. *Ecol. Indic.* 128, 107822. doi:10.1016/j.ecolind.2021.107822 *(abs)*
56. Schaeffer B.A., Urquhart E., Coffer M., Salls W., Stumpf R.P., Loftin K.A., et al. (7 authors) (2022). Satellites quantify the spatial extent of cyanobacterial blooms across the United States at multiple scales. *Ecol. Indic.* 140, 108990. doi:10.1016/j.ecolind.2022.108990 *(abs)*
57. Schaeffer B.A., Bailey S.W., Conmy R.N., Galvin M., Ignatius A.R., Johnston J.M., et al. (13 authors) (2018). Mobile device application for monitoring cyanobacteria harmful algal blooms using Sentinel-3 satellite Ocean and Land Colour Instruments. *Environ. Model. Softw.* 109, 93–103. doi:10.1016/j.envsoft.2018.08.015 *(abs)*
58. Mishra S., Stumpf R.P., Schaeffer B.A., Werdell P.J., Loftin K.A., Meredith A. (2019). Measurement of cyanobacterial bloom magnitude using satellite remote sensing. *Scientific Reports* 9, 18310. doi:10.1038/s41598-019-54453-y *(abs)*
59. Seegers B.N., Werdell P.J., Vandermeulen R.A., Salls W., Stumpf R.P., Schaeffer B.A., et al. (10 authors) (2021). Satellites for long-term monitoring of inland U.S. lakes: The MERIS time series and application for chlorophyll-a. *RSE* 266, 112685. doi:10.1016/j.rse.2021.112685 *(abs)*
60. Stumpf R.P., Wynne T.T., Baker D.B., Fahnenstiel G.L. (2012). Interannual variability of cyanobacterial blooms in Lake Erie. *PLoS ONE* 7, e42444. doi:10.1371/journal.pone.0042444 (Crossref)
61. Clark J.M., Schaeffer B.A., Darling J.A., Urquhart E.A., Johnston J.M., Ignatius A.R., et al. (10 authors) (2017). Satellite monitoring of cyanobacterial harmful algal bloom frequency in recreational waters and drinking water sources. *Ecol. Indic.* 80, 84–95. doi:10.1016/j.ecolind.2017.04.046 (Crossref)
62. Kudela R.M., Palacios S.L., Austerberry D.C., Accorsi E.K., Guild L.S., Torres-Perez J. (2015). Application of hyperspectral remote sensing to cyanobacterial blooms in inland waters. *RSE* 167, 196–205. doi:10.1016/j.rse.2015.01.025 (Crossref)
63. Hu C., Muller-Karger F.E., Taylor C., Carder K.L., Kelble C., Johns E., et al. (7 authors) (2005). Red tide detection and tracing using MODIS fluorescence data: A regional example in SW Florida coastal waters. *RSE* 97, 311–321. doi:10.1016/j.rse.2005.05.013 (Crossref)
64. DrivenData (2023). Tick Tick Bloom: Harmful Algal Bloom Detection Challenge. https://www.drivendata.org/competitions/143/tick-tick-bloom/ ✓. Winners: https://drivendata.co/blog/tick-tick-bloom-challenge-winners ✓. Code (MIT): https://github.com/drivendataorg/tick-tick-bloom ✓ (doi:10.5281/zenodo.8329929)
65. Krestenitis M., Orfanidis G., Ioannidis K., Avgerinakis K., Vrochidis S., Kompatsiaris I. (2019). Oil spill identification from satellite images using deep neural networks. *Remote Sensing* 11, 1762. doi:10.3390/rs11151762 ✓ (Table 4)
66. Kikaki K., Kakogeorgiou I., Mikeli P., Raitsos D.E., Karantzalos K. (2022). MARIDA: A benchmark for Marine Debris detection from Sentinel-2 remote sensing data. *PLoS ONE* 17, e0262247. doi:10.1371/journal.pone.0262247 *(abs)*
67. Kikaki K., Kakogeorgiou I., Hoteit I., Karantzalos K. (2024). Detecting marine pollutants and sea surface features with deep learning in Sentinel-2 imagery (MADOS). *ISPRS J. Photogramm. Remote Sens.* 210, 39–54. doi:10.1016/j.isprsjprs.2024.02.017 (Crossref)
68. Silva Rotta L.H., Alcântara E., Park E., Negri R.G., Lin Y.N., Bernardo N., et al. (8 authors) (2020). The 2019 Brumadinho tailings dam collapse: Possible cause and impacts of the worst human and environmental disaster in Brazil. *JAG* 90, 102119. doi:10.1016/j.jag.2020.102119 *(abs)*

**Water masks, distillation and miscellaneous**
69. Pekel J.-F., Cottam A., Gorelick N., Belward A.S. (2016). High-resolution mapping of global surface water and its long-term changes. *Nature* 540, 418–422. doi:10.1038/nature20584 (Crossref)
70. Messager M.L., Lehner B., Grill G., Nedeva I., Schmitt O. (2016). Estimating the volume and age of water stored in global lakes using a geo-statistical approach (HydroLAKES). *Nat. Commun.* 7, 13603. doi:10.1038/ncomms13603 (Crossref)
71. Hinton G., Vinyals O., Dean J. (2015). Distilling the knowledge in a neural network. arXiv:1503.02531 ✓
72. DJI Mavic 3 Multispectral specifications: https://ag.dji.com/mavic-3-m/specs ✓

**Code and services (verified live 2026-10-22)**
73. MDN-STREAM (GPL-3.0): https://github.com/STREAM-RS/MDN-STREAM. Legacy MDN: https://github.com/BrandonSmithJ/MDN. MDN-PC: https://github.com/STREAM-RS/MDN-Phycocyanin. AQUAVERSE tutorials: https://github.com/STREAM-RS/aqv-tutorials. NASA STREAM viewer: https://ladsweb.modaps.eosdis.nasa.gov/stream/
74. Microsoft Planetary Computer STAC: https://planetarycomputer.microsoft.com/api/stac/v1. Element 84 Earth Search: https://earth-search.aws.element84.com/v1. CDSE STAC: https://stac.dataspace.copernicus.eu/v1. USGS LandsatLook STAC: https://landsatlook.usgs.gov/stac-server. Water Quality Portal: https://www.waterqualitydata.us/

**EO foundation models and embeddings (from the foundation-model sub-survey; numbering [FMx] matches §a.6; arXiv IDs re-checked on export.arxiv.org)**
- FM1. Jakubik J., Roy S., Phillips C.E., et al. (2023). Foundation Models for Generalist Geospatial Artificial Intelligence (Prithvi-EO-1.0). arXiv:2310.18660
- FM2. Prithvi-EO-1.0-100M model card. https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M
- FM3. Szwarcman D., Roy S., Fraccaro P., et al. (2024). Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications. arXiv:2412.02732 ✓
- FM4. Prithvi-EO-2.0 weights. https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL (and -tiny-TL, -600M, -300M-TL-Sen1Floods11); https://github.com/NASA-IMPACT/Prithvi-EO-2.0
- FM5. HLS Product User Guide v2.0. https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf
- FM6. Jakubik J., Yang F., Blumenstiel B., et al. (2025). TerraMind: Large-Scale Generative Multimodality for Earth Observation. ICCV 2025; arXiv:2504.11171 ✓
- FM7. TerraMind 1.0 weights. https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base ; https://github.com/IBM/terramind
- FM8. TerraTorch TerraMind registry. https://github.com/terrastackai/terratorch
- FM9. TerraMesh dataset card. https://huggingface.co/datasets/ibm-esa-geospatial/TerraMesh
- FM10. TerraTorch. https://pypi.org/project/terratorch/
- FM11. TorchGeo models API. https://torchgeo.readthedocs.io/en/stable/api/models.html
- FM12. TorchGeo embedding datasets. https://torchgeo.readthedocs.io/en/stable/api/datasets.html
- FM13. TorchGeo transforms and sources. https://github.com/torchgeo/torchgeo
- FM14. Herzog et al. (2025). OlmoEarth: Stable Latent Image Modeling for Multimodal Earth Observation. arXiv:2511.13655 ✓. Tseng et al. (2026). OlmoEarth v1.2. arXiv:2605.20804 ✓
- FM15. olmoearth_pretrain (licence, quickstart, S2 config). https://github.com/allenai/olmoearth_pretrain
- FM16. OlmoEarth weights. https://huggingface.co/allenai/OlmoEarth-v1-Base
- FM17. rslearn (offset harmonization code). https://github.com/allenai/rslearn
- FM18. Tseng G., Fuller A., Reil M., et al. (2025). Galileo: Learning Global & Local Features of Many Remote Sensing Modalities. ICML 2025 (PMLR 267); arXiv:2502.09356 ✓
- FM19. Galileo code and weights. https://github.com/nasaharvest/galileo ; https://huggingface.co/nasaharvest/galileo
- FM20. Tseng G., Cartuyvels R., Zvonkov I., et al. (2023). Lightweight, Pre-trained Transformers for Remote Sensing Timeseries (Presto). arXiv:2304.14065 ✓
- FM21. Astruc G., Gonthier N., Mallet C., Landrieu L. (2025). AnySat. CVPR 2025; arXiv:2412.14123 ✓. https://github.com/gastruc/AnySat
- FM22. Wang Y., Xiong Z., Liu C., et al. (2025). Towards a Unified Copernicus Foundation Model for Earth Vision. ICCV 2025; arXiv:2503.11849 ✓. https://github.com/zhu-xlab/Copernicus-FM
- FM23. Xiong Z., Wang Y., Zhang F., et al. (2024). Neural Plasticity-Inspired Multimodal Foundation Model for Earth Observation (DOFA). arXiv:2403.15356 ✓. https://github.com/zhu-xlab/DOFA
- FM24. Waldmann L., Shah A., Wang Y., et al. (2025). Panopticon: Advancing Any-Sensor Foundation Models for Earth Observation. CVPRW (EarthVision) 2025; arXiv:2503.10845
- FM25. Sumbul G., Xu C., Dalsasso E., et al. (2025). SMARTIES: Spectrum-Aware Multi-Sensor Auto-Encoder for Remote Sensing Images. ICCV 2025; arXiv:2506.19585
- FM26. Clay Foundation model docs and weights. https://clay-foundation.github.io/model/ ; https://huggingface.co/made-with-clay/Clay
- FM27. Fuller A., Millard K., Green J.R. (2023). CROMA. NeurIPS 2023. https://github.com/antofuller/CROMA
- FM28. Wang Y., Braham N.A.A., Xiong Z., et al. (2023). SSL4EO-S12. *IEEE GRSM* 11(3), 98–106. doi:10.1109/MGRS.2023.3281651 (Crossref ✓)
- FM29. Stewart A.J., Lehmann N., Corley I.A., et al. (2023). SSL4EO-L. NeurIPS 2023 Datasets & Benchmarks; arXiv:2306.09424
- FM30. Wang Y., Albrecht C.M., Braham N.A.A., et al. (2024). DeCUR. ECCV 2024. doi:10.1007/978-3-031-73397-0_17
- FM31–FM38. SatMAE (NeurIPS 2022), SatMAE++ (CVPR 2024), Scale-MAE (ICCV 2023), SatlasPretrain (ICCV 2023), SpectralGPT (TPAMI 2024, doi:10.1109/TPAMI.2024.3362475), DINOv3 (arXiv:2508.10104), SkySense (CVPR 2024) and successors. Licence and access details as in §a.6.
- FM39. Wang D., et al. (2025). HyperSIGMA. *IEEE TPAMI* 47(8). doi:10.1109/TPAMI.2025.3557581
- FM40. Braham N.A.A., et al. (2025). SpectralEarth. *IEEE JSTARS* 18. doi:10.1109/JSTARS.2025.3581451
- FM41. Hydro water foundation model. https://github.com/isaaccorley/hydro-foundation-model
- FM43. Brown C.F., Kazmierski M.R., Pasquarella V.J., et al. (2025). AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data. arXiv:2507.22291 ✓
- FM44. Earth Engine catalog: Satellite Embedding V1. https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
- FM45. AlphaEarth embeddings on Source Cooperative. https://source.coop/tge-labs/aef
- FM46. AlphaEarth GCS bucket README. https://storage.googleapis.com/alphaearth_foundations/satellite_embedding/v1/annual/README.md
- FM47. Major TOM AlphaEarth subset. https://huggingface.co/datasets/Major-TOM/Core-AlphaEarth-Embeddings
- FM48. Feng Z., et al. (2025). TESSERA: Temporal Embeddings of Surface Spectra for Earth Representation and Analysis. arXiv:2506.20380 ✓
- FM49. geotessera / TESSERA data. https://github.com/ucam-eo/geotessera ; https://source.coop/tessera/tessera
- FM55. Marsocci V., Jia Y., Le Bellier G., et al. (2024). PANGAEA: A Global and Inclusive Benchmark for Geospatial Foundation Models. arXiv:2412.04204 ✓
- FM56. Lacoste A., et al. (2023). GEO-Bench. NeurIPS 2023 Datasets & Benchmarks.
- FM64. LaHaye N., Luis K., Gierach M. (2025). Fusing Multi- and Hyperspectral Satellite Data for Harmful Algal Bloom Monitoring with Self-Supervised and Hierarchical Deep Learning (SIT-FUSE). arXiv:2510.02763 ✓
- FM65. Narula H.D.S. (2026). Bloom or Bluff? Benchmarking Vision–Language Models Against Classical Machine Learning for Harmful Algal Bloom Detection from Satellite Imagery. *Remote Sensing* 18, 2147. doi:10.3390/rs18132147 (Crossref ✓)
- FM67. CyFi (DrivenData). https://cyfi.drivendata.org/
- FM68. ESA SentiWiki, S2 processing. https://sentiwiki.copernicus.eu/web/s2-processing
- FM69. Earth Engine S2_SR_HARMONIZED catalog. https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
- FM73. FlashAttention README (Ampere+ requirement). https://github.com/Dao-AILab/flash-attention
**Datasets (from the dataset sub-survey; DOIs checked on DataCite/Crossref; [Dx] numbering matches §b.2)**
- D1. Lehmann M.K., Gurlin D., Pahlevan N., et al. (77 authors) (2023). GLORIA – A globally representative hyperspectral in situ dataset for optical sensing of water quality. *Scientific Data* 10, 100. doi:10.1038/s41597-023-01973-y (Crossref ✓). Full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC9935528/
- D2. GLORIA dataset (PANGAEA). doi:10.1594/PANGAEA.948492. Download: https://download.pangaea.de/dataset/948492/files/GLORIA-2022.zip (downloaded and analysed by the sub-survey)
- D7. Maciel et al. Secchi matchup dataset doi:10.1594/PANGAEA.961720; paper doi:10.1016/j.rse.2023.113889
- D8. Ross M.R.V., et al. (2019). AquaSat. *WRR* 55, 10012–10025. doi:10.1029/2019WR024883
- D9. AquaSat figshare collection. doi:10.6084/m9.figshare.c.4506140.v2
- D10. AquaSat code and manuscript. https://github.com/GlobalHydrologyLab/AquaSat
- D11. AquaMatch ("AquaSat v2"). https://github.com/AquaSat ; https://aquasat.github.io/AquaMatch_documentation/
- D12. AquaMatch EDI packages (CC0), e.g., Chl-a doi:10.6073/pasta/2f750544112e5408928dd9a61e6ace30
- D13. Ghazi Zadeh Hashemi M.S., Bolten J., Kumar S., Seegers B. (2026). *AquaCorr-Global: A Multi-Sensor, Multi-Decadal Water Quality Matchup Dataset for Inland and Coastal Waters.* Zenodo. doi:10.5281/zenodo.21927415 (record checked via the Zenodo API: CC-BY-4.0; parquet 377 MB)
- D15. LimnoSat-US. doi:10.5281/zenodo.4139695
- D16. LAGOS-US LOCUS. doi:10.6073/pasta/e5c2fb8d77467d3f03de4667ac2173ca ; Cheruvelil K.S., et al. (2021), *L&O Letters* 6, 270–292, doi:10.1002/lol2.10203
- D17. LAGOS-US GEO. doi:10.6073/pasta/0e443bd43d7e24c2b6abc7af54ca424a
- D18. LAGOS-US LIMNO. doi:10.6073/pasta/2c58f5a50ab813919f99cc1f265f271c
- D19. LAGOS-US LANDSAT. doi:10.6073/pasta/128700feb3bbc3ffe5800e7b232bd81f
- D21. LIMNADES. https://www.globolakes.ac.uk/limnades.html (limnades.stir.ac.uk does not resolve)
- D22. ESA Lakes CCI v3.0. https://catalogue.ceda.ac.uk/uuid/a56dba09df8a42ec9fba8b8c7a5e1f69 ; archive https://dap.ceda.ac.uk/neodc/esacci/lakes/data/lake_products/L3S/v3.0/
- D23. Carrea L., et al. (2023). *Scientific Data* 10, 30. doi:10.1038/s41597-022-01889-z (v2.0.2 dataset doi:10.5285/a07deacaffb8453e93d57ee214676304)
- D24. EPA CyAN. https://www.epa.gov/water-research/cyanobacteria-assessment-network-cyan ; API example https://cyan.epa.gov/waterbody/search/?name=Okeechobee
- D25. NASA CyAN. https://www.earthdata.nasa.gov/data/projects/cyan ; doi:10.5067/MERGED-S3/OLCI/L3M/CYAN/CI/6.0
- D27. Water Quality Portal. https://www.waterqualitydata.us/ ; WQX3 beta documentation https://www.waterqualitydata.us/beta/webservices_documentation/ ; doi:10.5066/P9QRKUVJ
- D28. USGS WQX3 transition and Samples API. https://waterdata.usgs.gov/blog/wqx3/ ; https://doi-usgs.github.io/dataRetrieval/articles/Status.html ; https://api.waterdata.usgs.gov/samples-data/docs
- D29. EPA TADA (WQP QA tools). https://usepa.github.io/EPATADA/reference/index.html
- D31. EPA National Aquatic Resource Surveys data. https://www.epa.gov/national-aquatic-resource-surveys/data-national-aquatic-resource-surveys
- D32. UNEP GEMStat. https://gemstat.org/ ; open archive doi:10.5281/zenodo.18459694
- D33. Virro H., et al. (2021). GRQA. *ESSD* 13, 5483–5507. doi:10.5194/essd-13-5483-2021 ; data doi:10.5281/zenodo.15335450
- D34. EEA Waterbase – Water Quality ICM. https://discodata.eea.europa.eu/
- D36. CAML (cyanobacteria abundance; Tick Tick Bloom labels), SeaBASS. https://seabass.gsfc.nasa.gov/experiment/CAML ; doi:10.5067/SeaBASS/CAML/DATA001
- D37. AERONET-OC. https://aeronet.gsfc.nasa.gov/new_web/ocean_color.html
- D38. HYPERNETS. https://www.hypernets.eu/ (unreachable on 2026-10-22)
- D39. Valente A., et al. (2022). *ESSD* 14, 5737–5770. doi:10.5194/essd-14-5737-2022
- D40. NOAA GLERL western Lake Erie HAB monitoring. doi:10.25921/11da-3x54
- D41. NOAA HABSOS. https://www.ncei.noaa.gov/products/harmful-algal-blooms-observing-system
- D42. CDC OHHABS. https://www.cdc.gov/ohhabs/data/index.html
- D43. NY DEC HABs. https://data.ny.gov/d/95my-wijm
- D44. California FHAB. https://data.ca.gov/dataset/surface-water-freshwater-harmful-algal-blooms
- D45. SLU Miljödata-MVM. https://miljodata.slu.se/mvm/ ; Open API https://miljodata.slu.se/MVM/OpenAPI
- D46. UK Environment Agency water quality API. https://environment.data.gov.uk/water-quality/api-docs
- D47. Canada: https://open.canada.ca/data/en/dataset/bf06d4e3-7a57-4ac4-bcfe-d3417e20b8fc ; https://datastream.org/
- D49. CLMS Lake Water Quality. https://land.copernicus.eu/en/products/water-bodies/lake-water-quality-v2-2-100m
- D51. HydroLAKES. https://www.hydrosheds.org/products/hydrolakes (Messager et al. 2016, doi:10.1038/ncomms13603)
- D52. JRC Global Surface Water. Pekel et al. 2016, doi:10.1038/nature20584 ; https://planetarycomputer.microsoft.com/api/stac/v1/collections/jrc-gsw
- D53. GLWD v2. https://www.hydrosheds.org/products/glwd ; Lehner B., et al. (2025), *ESSD*, doi:10.5194/essd-17-2277-2025
- D54. SWOT Prior Lake Database. Wang et al. (2025), *WRR* 61, e2023WR036896, doi:10.1029/2023WR036896
- D55. ESA WorldCover. doi:10.5281/zenodo.5571936 ; doi:10.5281/zenodo.7254221
- D56. GLAKES. Pi X., et al. (2022), doi:10.1038/s41467-022-33239-3
- D57. GLCP. Meyer M.F., et al. (2020), doi:10.1038/s41597-020-0517-4

---

## Appendix A — Chip extraction pattern that ran in this session

Environment: a scratch venv with `pystac-client 0.9.1`, `planetary-computer 1.0.0`, `rasterio 1.5.2` (GDAL 3.12.2), `pyproj`, `shapely`, `fsspec`, `h5py`.

```python
import os, pystac_client, planetary_computer, rasterio
from rasterio.windows import Window
from pyproj import Transformer
os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                  GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
                  AWS_NO_SIGN_REQUEST="YES")   # Earth Search S2 buckets are public

def chip(href, lat, lon, half=16):
    with rasterio.open(href) as ds:
        x, y = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True).transform(lon, lat)
        r, c = ds.index(x, y)
        return ds.read(1, window=Window(c - half, r - half, 2 * half, 2 * half),
                       boundless=True, fill_value=0)

# Sentinel-2 (consistent Collection-1 processing): Element 84 Earth Search
es = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
it = next(es.search(collections=["sentinel-2-c1-l2a"],
                    intersects={"type": "Point", "coordinates": [-83.30, 41.70]},
                    datetime="2023-08-27").items())
red = chip(it.assets["red"].href, 41.70, -83.30) * 1e-4 - 0.1   # C1: scale 1e-4, offset -0.1
scl = chip(it.assets["scl"].href, 41.70, -83.30, half=8)        # 20 m; class 6 = water

# Landsat C2 L2 / OLCI / HLS / JRC GSW: Planetary Computer (anonymous SAS signing)
mpc = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                modifier=planetary_computer.sign_inplace)
# Landsat SR = DN * 2.75e-5 - 0.2 ; ST [K] = DN * 0.00341802 + 149
# MPC sentinel-2-l2a: reflectance = (DN - 1000*[baseline >= 04.00]) / 10000  (no offset in metadata)
```

OLCI WFR (NetCDF4/HDF5) supports partial reads:

```python
import fsspec, h5py
f = fsspec.filesystem("http").open(href, block_size=256 * 1024, cache_type="blockcache")
with h5py.File(f, "r") as h:
    window = h["Oa08_reflectance"][r - 8:r + 8, c - 8:c + 8]
```

The scale factor is in the variable attributes. Values are ρ<sub>w</sub>, so Rrs = ρ<sub>w</sub>/π.

## Appendix B — Verification log (2026-10-22)

- **STAC collection listings:** MPC (139 collections), Earth Search (9), CDSE (427, paginated), USGS LandsatLook (14). Script: `scripts/stac_collections.py`.
- **Chip reads** for S2 L2A (MPC, Earth Search C1 and legacy), S2 L1C (Earth Search, JP2), Landsat C2 L2 (MPC OK; Earth Search requester-pays: 403), HLS v2 S30 (MPC; int16, scale 1e-4), JRC GSW occurrence (MPC), and OLCI WFR partial HTTP reads. Scripts: `stac_chip_test.py`, `earthsearch_test.py`, `offset_check.py`, `olci_partial.py`, `hls_check.py`.
- **Processing-baseline audit** (MPC vs Earth Search C1 vs legacy) and the same-pixel DN comparison: C1 1734 vs legacy 734, Δ = 1000.
- **CDSE:** search is anonymous. Product download returned 401 "Token not found". The CLMS LWQ 100 m v2 item listing was retrieved.
- **WQP:** HEAD counts for 14 characteristic/site-type combinations (`scripts/wqp_counts.sh`). Minnesota 2023 Chl-a download (4.6 MB) and the S2 yield experiment (`scripts/yield_experiment.py`).
- **References:** DOIs checked on Crossref (`scripts/crossref_check.py`, `crossref_authors.py`) and Semantic Scholar. arXiv IDs checked on export.arxiv.org. GitHub repositories checked through the GitHub API.
