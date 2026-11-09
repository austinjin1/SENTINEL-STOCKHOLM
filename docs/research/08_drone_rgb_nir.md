# 08: Drone RGB+NIR water-quality model (literature, data, distillation, edge deployment, plan)

*Research note for the SENTINEL rebuild, Goal 6: a distilled, miniature drone model that uses only visible (RGB) and near-infrared (NIR) cameras. Prepared 2026-10-22.*

**How to read the evidence tags**

- **[V]**: bibliographic metadata verified by identifier (Crossref, DataCite or the arXiv API).
- **[A]**: numbers quoted from the paper's abstract (the full text was not read).
- **[P]**: landing page, repository record or vendor page opened in this session.
- **[H]**: hands-on test run in this session. Scripts are listed in Appendix A.
- **[U]**: unverified (from memory or a single search snippet). Do not cite [U] items without checking them.

Untagged numbers in §a.4, §b.2 and §d.3 come from this session's own analyses.

**Coverage limits.** The shared web-search budget ran out twice during this session. The 2023–2026 sweep of UAV water-quality literature is therefore incomplete. Several MDPI full texts returned HTTP 403, so for those papers only abstract-level numbers are given. Run a Scopus or Google Scholar check before making any novelty claim (§e.11).

---

## 0. Decisions on one page

1. **What an RGB+NIR camera can honestly deliver**
   - **Good:** water clarity (Secchi depth), turbidity/TSS, water masks, and detection of surface scums, floating algae, sediment plumes and discoloured discharges.
   - **Moderate (presence only):** oil-sheen presence, which depends on glint geometry.
   - **Weak:** chl-a concentration across sites. It works only as a within-site or trophic-class product.
   - **Not retrievable:** subsurface phycocyanin, CDOM, and anything non-optical (pH, DO, conductivity, TN/TP, microcystin).
   - The old `download_drone_s2_pairs.py` targets (DO, pH, Turb, Temp, SpCond) should be cut to the optical set.
2. **Our band-sufficiency pilot (§a.4) puts numbers on the red-edge penalty.** It uses GLORIA's 7,572 in-situ hyperspectral spectra and holds out whole datasets in cross-validation.
   - **chl-a:** dropping red-edge costs about 22–27 percentage points of median symmetric accuracy (MdSA 71–91% vs 48–69%).
   - **Secchi and TSS:** the cost is small, about 3–7 points.
   - **Random vs. held-out splits:** random K-fold looks 1.5–2× better than held-out datasets. Only leave-water-body-out numbers may be reported.
   - **Radiometric noise:** a 5% band-gain error is as damaging as losing red-edge. Training with noise augmentation recovers part of the loss.
3. **No public drone dataset pairs RGB+NIR imagery with in-situ water quality** (none found; §b.4). The proxy imagery is therefore:
   - NAIP 4-band (0.3–1 m) for the student;
   - Sentinel-2 (with red-edge) for the teacher;
   - GLORIA for physics pre-training;
   - NEON AOP hyperspectral to simulate the drone camera.

   We tested NAIP↔Water Quality Portal matchups (±1 day) for Minnesota alone over three NAIP cycles. They yield 1,105 Secchi and 424 chl-a site-days. A national NAIP–in-situ matchup set looks feasible and may itself be novel.
4. **Distillation design.** The teacher is a Sentinel-2 pixel model with uncertainty, trained on in-situ data. The student is a 4-band dense network.
   - Distill on near-coincident S2↔NAIP pairs, using block-averaged, uncertainty-weighted losses, plus direct in-situ supervision.
   - Spectral super-resolution ("hallucinated red-edge") is an auxiliary task only, never a measurement.
5. **Edge target.**
   - **Primary:** Raspberry Pi 5 + AI HAT+ (Hailo-8L, 13 TOPS, from $70). This matches the Pi-based ROS 2 design already on the `camera-model` branch.
   - **Alternative:** Jetson Orin Nano Super (67 sparse / 33 dense INT8 TOPS, $249).
   - **Budget:** ≤3 M parameters, ≤2 GMACs per 512×512×4 tile, INT8 ≤4 MB, ≤100 ms per tile on the device.
   - **Desktop proxy (Xeon, ONNX Runtime, single thread, INT8):** MobileNetV4-Conv-S+FPN 19 ms and EfficientNet-Lite0+FPN 42 ms per 512² tile. MobileNetV3 and EfficientViT did *not* speed up under INT8 on CPU.
6. **The `camera-model` capture settings must change** before any quantitative use:
   - MJPEG with auto white balance → RAW with fixed exposure, gain and colour gains;
   - calibration panel and sky frames;
   - a second camera (NoIR + long-pass filter) synchronised with the RGB camera;
   - correct viewing geometry.
7. **Honest claim (§e.11).** Not "first drone water-quality model". Rather: the first openly released RGB+NIR student, distilled from a red-edge teacher using only public data, evaluated on held-out lakes against in-situ data, with the red-edge gap and the INT8 edge latency measured. Any "first" must say what was searched.

---

## (a) Literature and achievable accuracies

### a.1 What visible+NIR can see, by parameter

| Parameter | RGB+NIR (no red-edge) | Physical basis | Best evidence (as reported) | Confidence |
|---|---|---|---|---|
| **Secchi / clarity** | **Good** | Total attenuation drives brightness and hue. Blue/red ratio plus red magnitude is the classic Landsat recipe (Kloiber et al. 2002; Olmanson et al. 2008). Mechanistic visibility theory: Lee et al. 2015, 2016. | Smartphone RGB: Z_sd "accuracy 72.4%", 112 sites (Gao et al. 2022) [A]. PlanetScope 4-band ≈ Sentinel-2 for Secchi (Mansaray et al. 2021) [A]. Our pilot (held-out datasets): MdSA 37–44%; adding red-edge gains ≤3–4 points. | High |
| **Turbidity / TSS** | **Good–moderate** | Particulate backscatter. Red band for low–medium turbidity, NIR for high (Nechad et al. 2010; Dogliotti et al. 2015). | Dogliotti 645/859 nm: within 12–22% at five sites, mean relative error 13.7%, n=106 [A]. Parrot Sequoia NIR single band: R²=0.94, RMSE 7.8 mg/L over 0–180 mg/L (Olivetti et al. 2020) [A]. 4-band camera RMSE 9.8 vs hyperspectral 3.6 mg/L, N=53 (Olivetti et al. 2025) [A]. MicaSense: TSS 9% relative error (Windle & Silsbe 2021) [A]. HydroColor smartphone: turbidity within 24% (Leeuw & Boss 2018) [A]. Our pilot: MdSA 77–84%; red-edge gains 3–7 points. | High (NIR needed when turbid) |
| **Chlorophyll-a** | **Weak across sites; moderate within a site** | In eutrophic water the robust signal is the red-edge reflectance peak against red absorption (~705 vs 665 nm; e.g., NDCI, Mishra & Mishra 2012). RGB+NIR only offers blue/green (clear water) or NDVI-like proxies (blooms). | NDVI vs chl-a R²=0.88 at one stream, UAV+USV (Kim et al. 2021) [A]. Index R²=0.78 but **failed** independent-site validation (McEliece et al. 2020) [A]. Landsat-8 OLI (no red-edge) MDN: "49% (median) accuracy", their error metric (Smith et al. 2021) [A]. DJI P4M "consistently underperformed" MicaSense RedEdge-P, whose best R² was 0.867 (Salim et al. 2025) [A]. Our pilot: MdSA 71–91% without red-edge vs 48–69% with it. | Medium-low |
| **Cyanobacteria: surface scum / floating algae** | **Detectable (presence/extent)** | Floating biomass has vegetation-like high NIR reflectance (Floating Algae Index concept; Hu 2009). | UAV multispectral + transformer: >85% bloom detection (Song et al. 2022) [A]. RPi RGB+NoIR drone mapped water with NDWI (Belcore et al. 2019) [A]. | Medium-high for detection |
| **Cyanobacteria: subsurface phycocyanin** | **No** | Needs the ~620 nm absorption feature resolved, i.e., hyperspectral data. | Drone hyperspectral PC models R²=0.87–0.94 (Kwon et al. 2020) [A]. Phycocyanin MDN on HICO/PRISMA (O'Shea et al. 2021) [V]. | High that RGB+NIR can't |
| **CDOM** | **No (not transferable)** | Blue absorption confounded with chl-a and detritus. | Our pilot: MdSA 85–110% and R²_log ≤0.37 for every multispectral set on held-out datasets. HydroVision's R²=0.89 for CDOM comes from fixed USGS cameras (Deshmukh et al. 2025) [A]. | High that it is unreliable |
| **Oil sheen / slick** | **Presence: moderate. Thickness: no.** | Contrast depends on sun-glint geometry (Sun & Hu 2016: detection threshold on glint strength). Thick oil and emulsions also show NIR/visible features (Lu et al. 2019; review by Hu et al. 2021). | Drone RGB datasets: U-Net F1 0.71 (De Kerf et al. 2024) [A]; >66% mIoU, oil/emulsion >74% IoU (LADOS, Gkountakos et al. 2025) [A]. | Medium |
| **Sediment plumes / discoloured discharges** | **Detectable** | Strong red/NIR and hue anomalies against ambient water. | Turbidity plumes from UAV hyperspectral ML, R²=0.75 (Kieu et al. 2023) [A]. Sentinel-2 class labels for sediment-laden and turbid water (MARIDA; Kikaki et al. 2022) [P]. | Medium-high for detection |
| **Non-optical (pH, DO, SpCond, TN, TP, COD, microcystin)** | **No direct signal** | Any skill is a site-specific proxy. | Sequoia, n=60, one day: R² 0.97 TN / 0.90 TP / 0.70 COD (Long et al. 2025) [A]. MLR TP and TN R²=0.98 (Arango & Nairn 2020) [A]. Both are in-sample fits and should be read as overfitting warnings. | High that these won't transfer |

### a.2 Drone and consumer-camera studies

*R² and errors are as reported in the abstracts. "In-sample" means no independent site or date test was described in the abstract.*

| Study | Sensor / bands | Site, n | Parameters | Best result (reported) | Red-edge needed? | Validation | Tag |
|---|---|---|---|---|---|---|---|
| Su & Chou 2015, *Remote Sens.* 7:10078 | **Separate RGB and NIR cameras** on UAV | Tain-Pu reservoir (Kinmen), few points | Chl-a, TP, Secchi | Matching pixel-by-pixel beats window averaging. Map means: chl-a 179.7 µg/L, Secchi 1.4 m. R² not in abstract. | No | Small n, in-sample | [V][A] |
| Cheng, Chan & Lee 2020, *Mar. Pollut. Bull.* 152:110889 | **Digital RGB camera** on UAV | Hong Kong coast, one year of field data | Chl-a during blooms | Calibrated model, "validated against one-year field data". Numbers not in abstract. | No | Temporal (details unverified) | [V][A] |
| Ehmann et al. 2019, *Hydrol. Process.* 33:1013 | Two sUAS, multispectral | Creek, Syracuse NY | Turbidity | Red-band metrics strongly related to turbidity. Numbers not in abstract. | No | Several flights | [V][A] |
| Leeuw & Boss 2018, *Sensors* 18:256 | **Smartphone RGB** (HydroColor) | Various | Rrs, turbidity | Rrs within 26% of a radiometer; turbidity within 24% | No | Instrument comparison | [V][A] |
| Gao et al. 2022, *Remote Sens.* 14:1371 | **Smartphone RGB** + reference card | 7 water bodies, 112 sites | Rrs, Secchi, turbidity | Rrs error (AURE) 25.7/29.5/35.2% (R/G/B). Z_sd accuracy 72.4%, turbidity 60.2%. | No | Multi-site | [V][A] |
| Burggraaff et al. 2022, *Front. Remote Sens.* 3:940096 | **Smartphone RAW**, calibrated | Field intercomparison | Rrs | ~10% Rrs accuracy per RGB band, ~2% for band ratios. JPEG apps differ by up to 40%. | No | Instrument comparison | [V][A] |
| Belcore et al. 2019, *ISPRS Arch.* XLII-2/W13 | **RPi Camera v2 RGB + NoIR v2** on UAV | Two villages, SW Niger | Water bodies (NDWI) | Matched a Sony camera for water delineation | No | Qualitative | [V][A] |
| De Keukelaere et al. 2023, *Remote Sens.* 15:1345 (MapEO) | **DJI Phantom 4 Pro (RGB)** + MicaSense | European lakes | Rrs, turbidity, chl-a | R² 0.71 (turbidity), 0.93 (chl-a). The abstract does not say which camera gave which result. | Unclear | Multi-lake | [V][A] |
| Olivetti et al. 2020, *Remote Sens.* 12:1855 | Parrot Sequoia (G, R, RE, NIR) | Two reservoirs, Brazil | TSS | NIR single band R² 0.94, RMSE 7.8 mg/L (0–180) | No (NIR) | Dry and rainy seasons | [V][A] |
| Olivetti et al. 2025, *Drones* 9:173 | 276-band pushbroom vs 4-band global shutter | 15 campaigns, N=53, two biomes | TSS | RMSE 3.6 (hyperspectral) vs 9.8 mg/L (4-band). Sun-glint mitigation. | No (NIR) | Multi-campaign | [V][A] |
| Larson et al. 2018, *Int. J. Remote Sens.* 39:5472 | Sequoia (G, R, RE, NIR) | Maumee River, OH | SSC by depth | UAV R²adj 0.56 vs field spectroradiometer 0.71 | Red–NIR | In-sample | [V][A] |
| McEliece et al. 2020, *Remote Sens.* 12:2258 | 448/494/550/675 nm (no NIR) | 7 UK sites, n=35; validated in France | Chl-a, turbidity | Index (R550−R448)/(R550+R448): 78% / 74% variance explained | No | **Failed independent validation**; site-specific calibration needed | [V][A] |
| Arango & Nairn 2020, *Drones* 4:1 | sUAS multispectral | Oligotrophic ponds and eutrophic lagoons | TSS, Secchi, chl-a, TP, TN | R² 0.99 / 0.88 / 0.85 (and 0.98 TP/TN) | Visible bands | In-sample MLR | [V][A] |
| Kim et al. 2021, *Water* 13:1930 | UAV multispectral + USV | Stream, Korea | Chl-a | NDVI R² 0.88, RMSE 2.25. GNDVI R² 0.74. | No (NDVI) | Single system | [V][A] |
| Windle & Silsbe 2021, *Front. Environ. Sci.* 9:674247 | MicaSense RedEdge | Coastal waters | Rrs, chl-a, TSS | NIR pixel-based removal of surface reflection works best. Relative error 37% (chl-a) and 9% (TSS). | Uses 5 bands | Against in-situ hyperspectral Rrs | [V][A] |
| Cillero Castro et al. 2020, *Remote Sens.* 12:1514 | MicaSense RedEdge (8 cm) + S2 + L8 | Drinking-water reservoir with cyanobacterial blooms | Chl-a | Red/NIR combinations (2BDA, SABI, 3BDA) best. Blue/green separate two chl-a classes. Moderate agreement across sensors. | Yes (best) | Multi-sensor | [V][A] |
| Zhao et al. 2022, *Drones* 7:2 | Multispectral UAV | Erhai Lake, 155 samples | Chl-a | CNN best among SVM/RF/ELM/CNN | Unclear | Two-month campaign | [V][A] |
| Salim et al. 2025, *ISPRS Arch.* XLVIII-2/W11:263 | DJI P4M vs MicaSense RedEdge-P Dual | Ibirité reservoir, 4 campaigns (2024) | Chl-a, turbidity | MicaSense chl-a R² 0.867 (RMSE 6.72 µg/L), turbidity R² 0.712. P4M underperformed. | Helps | Multi-campaign | [P][A] |
| Kieu et al. 2023, *Mar. Pollut. Bull.* 196:115482 | UAV hyperspectral | Land-reclamation turbidity plumes | Turbidity | ML R² 0.75 | n/a | Large dataset | [V][A] |
| Kwon et al. 2020, *Remote Sens. Environ.* 236:111517 | Drone hyperspectral | Deep reservoir, Korea | Phycocyanin, chl-a | PC models R² 0.87–0.94 using cumulative pigment | Hyperspectral | — | [V][A] |
| Becker et al. 2019, *J. Great Lakes Res.* 45:444 | UAS spectroradiometer (Ocean Optics STS) | Lake Erie / Maumee, 10 sites | Cyanobacteria Index | Consistent with FluoroProbe and field data | Hyperspectral | Qualitative | [V][A] |
| Song et al. 2022, *Sensors* 22:4571 | UAV multispectral + transformer | Chinese rivers | Bloom detection | >85% detection accuracy | Vegetation indices | Multi-river | [V][A] |
| Román et al. 2023, *Remote Sens.* 15:237 | 5-band multispectral | Maltese coast | Chl-a, TSS | Photogrammetry fails over water, so metadata-based orthomosaic. Chl-a 0–3 mg/m³, TSS 10–20 mg/m³. | Yes | Spectral comparisons | [V][A] |
| Long et al. 2025, *Water* 17:1566 | Sequoia | Zhangshan reservoir, n=60, one day | TN, TP, COD | R² 0.970 / 0.902 / 0.695 | — | **In-sample**; a cautionary example | [P] (PDF read) |
| Deshmukh et al. 2025, arXiv:2509.01882 (HydroVision) | Fixed USGS HIVIS RGB cameras, >500k images | USGS monitoring sites, 2022–2024 | Chl-a, CDOM, phycocyanin, sediment, turbidity | DenseNet121 best, R² 0.89 for CDOM (validation) | No | Validation split (independent sites not confirmed) | [V][A] |

### a.3 Four-band satellite analogues (no red-edge)

- **PlanetScope (B, G, R, NIR) turbidity** (Vanhellemont 2019, *Opt. Express* 27:A1372) [V][A]. Red-band turbidity was good. NIR-band turbidity was "mediocre, likely due to calibration issues". RapidEye's red-edge band allowed chl-a absorption to be avoided.
- **PlanetScope vs Landsat-8 vs Sentinel-2** (Mansaray et al. 2021) [V][A]. Data: 13 Oklahoma reservoirs, 2017–2020. Landsat-8 and S2 explained more chl-a variance than PlanetScope. PlanetScope ≈ S2 for turbidity and Secchi.
- **Landsat-8 OLI chl-a MDN** (Smith et al. 2021) [V][A]. OLI has no red-edge band. Trained on N=4,354 in-situ Rrs–chl-a pairs; 49% median accuracy on held-out data. Multi-sensor MDNs (Pahlevan et al. 2020, 2022) [V][A] report large gains for MSI and OLCI, which do have red-edge.
- **Landsat clarity census of Minnesota lakes** (Olmanson et al. 2008; procedure in Kloiber et al. 2002) [V]. This is the precedent for blue/red-based Secchi at state scale. Our NAIP matchups (§b.2) are also in Minnesota.

### a.4 Band-sufficiency pilot (this session; GLORIA, CC BY 4.0)

**Setup** [H]
- **Data:** GLORIA (Lehmann et al. 2023): 7,572 in-situ Rrs spectra (350–900 nm, 1 nm) with co-located chl-a (5,132), TSS (4,623), aCDOM(440) (4,393) and Secchi (3,948), drawn from 54 contributing datasets.
- **QC:** spectra with QC flags removed (noisy red/blue, baseline shift, QWIP fail, suspect, negative UV slope).
- **Band simulation:** Gaussian spectral response functions from nominal centre/bandwidth. These are approximations, not measured camera curves.
- **Model:** gradient boosting on log band Rrs plus all log band ratios.
- **Metrics:** MdSA (median symmetric accuracy, %; lower is better) and R² of log values.
- **Splits:** "random" is shuffled 5-fold. **"Held-out" is GroupKFold-5 by GLORIA dataset**, a proxy for unseen water bodies.
- **Noise test (tables A/B):** a systematic per-band gain error (one draw per band per test fold, 5% 1σ) plus a per-spectrum additive offset (5×10⁻⁴ sr⁻¹, 1σ), applied to the test spectra only. This mimics camera calibration error plus residual sky glint.
- **Noise-augmentation test:** gain and offset noise drawn independently per spectrum and band; the training set is augmented 4×.
- **Two variants:**
  - A needs complete 400–900 nm spectra (2,137 spectra, NIR band = S2 B8);
  - B needs 400–800 nm (5,580 spectra, NIR proxy = 783 nm, S2 B7-like).

**Variant A: held-out MdSA % (R²_log)**

| Band set | chl-a (n=1,566) | TSS (n=1,602) | Secchi (n=1,206) | aCDOM (n=1,411) |
|---|---|---|---|---|
| RGB (S2 B2–B4) | 122 (0.34) | 94 (0.41) | 42 (0.69) | 109 (0.20) |
| **RGB+NIR (S2 B2, B3, B4, B8)** | **91 (0.48)** | **77 (0.52)** | **37 (0.74)** | 103 (0.19) |
| Broad "camera-like" RGB + 720–900 nm NIR (approx.) | 95 (0.44) | 83 (0.50) | 35 (0.77) | 109 (0.18) |
| P4M-like B, G, R, **RE 730**, NIR | 69 (0.59) | 74 (0.56) | 33 (0.80) | 92 (0.23) |
| S2 B2–B8 (incl. red-edge B5–B7) | **64 (0.60)** | 73 (0.55) | 35 (0.80) | 85 (0.31) |
| *Same RGB+NIR, random 5-fold* | *54 (0.71)* | *40 (0.75)* | *21 (0.87)* | *47 (0.59)* |
| *Same RGB+NIR, held-out + radiometric noise* | *120 (0.18)* | *81 (0.50)* | *40 (0.71)* | *111 (0.11)* |

**Variant B: held-out MdSA % (R²_log)**

| Band set | chl-a (n=3,759) | TSS (n=3,865) | Secchi (n=3,075) | aCDOM (n=3,397) |
|---|---|---|---|---|
| RGB | 98 (0.60) | 86 (0.28) | 53 (0.39) | 97 (0.33) |
| **RGB + NIR\* (783 nm)** | **71 (0.69)** | **84 (0.29)** | **44 (0.46)** | 97 (0.31) |
| Broad camera RGB + NIR\* 720–800 nm (approx.) | 78 (0.67) | 83 (0.28) | 45 (0.44) | 103 (0.32) |
| RGB + RE (704, 740) + NIR\* | **48 (0.80)** | 77 (0.32) | 41 (0.44) | 91 (0.37) |
| *RGB+NIR\*, held-out + noise* | *94 (0.59)* | *90 (0.26)* | *57 (0.27)* | *117 (0.21)* |
| *RGB+RE+NIR\*, held-out + noise* | *88 (0.37)* | *90 (0.28)* | *56 (0.33)* | *134 (0.17)* |

**Does training with noise help?** Variant B, held-out, MdSA %:

| Target / bands | Clean train → clean test | Clean train → noisy test | **Noise-augmented train → noisy test** | Noise-augmented train → clean test |
|---|---|---|---|---|
| chl-a, RGB+NIR\* | 70.7 | 97.0 | **85.7** | 72.2 |
| chl-a, RGB+RE+NIR\* | 47.5 | 89.7 | **62.4** | 52.2 |
| TSS, RGB+NIR\* | 83.6 | 89.9 | **78.9** | 74.0 |
| Secchi, RGB+NIR\* | 44.1 | 54.7 | **48.6** | 45.3 |

**Interpretation**

1. Secchi depth is the most transferable RGB+NIR product. TSS is usable. chl-a pays a large red-edge penalty. CDOM is not retrievable from any of these band sets.
2. NIR helps over RGB-only for chl-a and TSS.
3. The approximated broad camera bands cost a few more points than narrow bands.
4. Random splits overstate skill by about 1.5–2×.
5. Radiometric error is as damaging as losing red-edge. Noise-augmented training recovers part of the loss and sometimes even helps on clean data (TSS).

**Caveats**
- GLORIA Rrs are *in-situ* best-case spectra: no atmosphere, sky glint already removed.
- The spectral response functions are approximations.
- Gradient boosting with hand-made ratios was used. The hyperspectral rows in the raw tables lacked ratio features, so they are not an upper bound.
- The two variants use different subsets, which is why R² differs between them.

These are design inputs, not paper results. Re-run with measured camera response curves (§e.2) before publishing.

### a.5 Radiometry: the real bottleneck for drones

**Surface-reflected light (sky and sun glint)**

- **Mobley 1999** [V][A]. Use a view 40° from nadir and 135° in azimuth from the sun. ρ≈0.028 is acceptable only for wind <5 m/s; on overcast days use ρ≈0.028 at all wind speeds. Otherwise ρ depends on sun zenith and wind.
- **Lee et al. 2010** [V][A]. Effective ρ can differ by a factor of 8 between 400 and 800 nm within one scan, so a constant ρ can be highly inaccurate. They propose spectral optimisation.
- **Other methods** [V]: polarised surface model (Mobley 2015); spectral sky reflectance (Zhang et al. 2017); NIR "similarity spectrum" for turbid water (Ruddick et al. 2006); 3C model (Groetsch et al. 2017); glint removal in field radiometry (Kutser et al. 2013).
- **Image deglinting from NIR** [V]: regress each visible band on NIR over deep water (Hedley et al. 2005). Review of high- and low-resolution methods: Kay et al. 2009. Glint statistics: Cox & Munk 1954.
- **Drones specifically:** Windle & Silsbe 2021 [V][A] recommend a pixel-based NIR approach to remove surface reflection (MicaSense). Gray et al. 2022 [V][A] give a guide to viewing geometry, sky-reflection removal and uncertainty, with open code.
- **DroneWQ** (Windle et al. 2026, *JOSS*; MIT licence) [V][P] implements the pipeline:
  1. raw → radiance;
  2. sky-reflection removal (e.g., Hedley);
  3. Rrs from downwelling irradiance (DLS or panel);
  4. NIR>0.02 glint mask;
  5. chl-a (OCx/Hu/Gitelson) and TSM (Nechad);
  6. sky images taken at about 40° from zenith and 135° azimuth.

**Calibration**

- **Downwelling light sensors are unreliable.**
  - Parrot Sequoia (Olsson et al. 2021) [V][A]: camera sensitivity drifts with temperature, the atmosphere between ground and flight altitude matters, and sunshine-sensor orientation adds strong noise. Recommendation: warm up the camera and image panels at close to flight altitude.
  - MicaSense RedEdge-MX Dual (Daniels et al. 2023) [V][A]: multi-target empirical line calibration at mission altitude was best. The DLS2 did not improve bias or RMSE, even when overcast.
  - Background on lab and field protocols [V]: Kelcey & Lucieer 2012; Mamaghani & Salvaggio 2019; Cao et al. 2019.
- **Consumer and Pi cameras** (SPECTACLE; Burggraaff et al. 2019) [V][A]. RAW output is linear, JPEG is not. Pixel gain varies by >400%. Flat-field (vignetting) factors reach **2.79** across the field of view. Camera models differ widely.
  - With RAW plus calibration, Rrs ≈10% per band and ≈2% for band ratios (Burggraaff et al. 2022).
  - The Raspberry Pi v2 camera (IMX219) is linear in RAW, and its spectral response has been measured (Pagnutti et al. 2017) [V][A].

**Geometry**

- Structure-from-motion mosaicking fails over featureless, moving water (Román et al. 2023, 2024) [V][A]. Direct georeferencing from GNSS/IMU reached RMSE 4.52 m (SD 2.51 m) in the best case.
- Sun glint also breaks orthomosaics over uniform water (Olivetti et al. 2025) [A].

**Minimum radiometric protocol for SENTINEL's Pi cameras (recommended)**

1. Capture RAW (Bayer, 10–12 bit) with fixed exposure, gain and colour gains. Turn auto white balance off and record all settings. The `camera-model` branch currently uses MJPEG with auto white balance, which is unusable for radiometry.
2. Do dark-frame, linearity and flat-field calibration once per camera (SPECTACLE's DIY methods).
3. Measure the spectral response of both cameras, including the long-pass filter. Pagnutti et al. 2017 measured only the standard IMX219 V2, so NoIR + filter must be measured.
4. Image a grey reflectance panel before and after each flight at altitude. Optionally fly an upward irradiance sensor, but do not trust it alone.
5. Take sky frames at 40°/135°.
6. Point water frames away from the sun (about 30–40° off-nadir at about 135° relative azimuth, or nadir only at low sun).
7. Remove sky reflection using ρ (Mobley), then NIR-based residual correction (Hedley-type).
8. Mask glint, shadows, shoreline pixels within ~2 GSD, and vegetation.
9. Hardware-sync the two cameras and co-register them by homography.

### a.6 Reviews and syntheses (2018–2026)

**Verified**

- Kislik, Dronova & Kelly 2018, *Drones* 2:35. UAVs for algal-bloom research [V].
- Wu et al. 2019, *Environ. Monit. Assess.* 191:211. Drone-based HAB monitoring [V].
- Topp et al. 2020, *Water* 12:169. Trends in inland-water remote sensing [V].
- Kutser et al. 2020, *Remote Sens. Environ.* 240:111619. Shallow-water remote sensing over 50 years [V].
- Sibanda et al. 2021, *Drones* 5:84. Drones for surface water, global south [V].
- Yang Z. et al. 2022, *Sci. Total Environ.* 838:155939. UAV marine monitoring [V].
- Yang H. et al. 2022, *Remote Sens.* 14:1770. Remote sensing for water-quality retrieval [V].
- Gray, Larsen & Johnston 2022, *Front. Ecol. Environ.* 20:413. Drones for biological oceanography [V].
- Adjovu et al. 2023, *Remote Sens.* 15:1938. Remote sensing of water-quality parameters [V].
- Oil: Fingas & Brown 2018, *Sensors* 18:91 [V]; Hu et al. 2021, *J. Remote Sens.* 9141902, "what is really possible" [V].

**Gap:** no dedicated 2024–2026 *UAV* water-quality review was found with the limited search budget [U]. Check Scopus.

---

## (b) Datasets with verified access

### b.1 Inventory

| Dataset | Content / bands / GSD | Coverage, time | Size | License | Access (credentials?) | Role in plan | Tag |
|---|---|---|---|---|---|---|---|
| **NAIP** on Microsoft Planetary Computer (`naip`) | 4-band R, G, B, NIR, **uint8**, Cloud-Optimized GeoTIFF. GSD 1.0 m (≤2017), 0.6 m (2018+), **0.3 m in 2023** for FL, OH and MN in our queries. | CONUS. Planetary Computer extent 2010–2023. ≤3-year cycle, leaf-on season. | ~460 MB uncompressed per 0.6 m quarter-quad. Minnesota has 6,791 quads per cycle. | LILA's Chesapeake page says NAIP imagery is public domain, and the USGS EROS page marks its NAIP example "Public Domain". The Planetary Computer STAC licence field reads "proprietary". Confirm with USDA FSA before redistributing. | **Anonymous**: STAC + automatic SAS signing via `planetary-computer` (no account). Also USGS EarthExplorer (DOI 10.5066/F7QN651G). | Student imagery (drone proxy), in-situ matchups, scum and plume weak labels | [H][P] |
| NAIP caveats | **Not calibrated to reflectance**. Per-state vendor colour balancing. Acquisition time of day not in STAC. Some cycles flown outside summer (Utah 2021 in Nov; Florida 2021–23 in Jan–Feb). NIR response depends on sensor (not verified). | | | | | Needs per-scene normalisation (§e.3) | [H][U] |
| **Water Quality Portal** (WQP) | In-situ results: "Depth, Secchi disk depth", "Chlorophyll a" (+ ", corrected/uncorrected for pheophytin"), "Turbidity", "Total suspended solids". | USA, decades | API queries; Minnesota summer ≈1.3–1.7 MB zipped per year | US public data | **Anonymous** REST (`/data/Result/search`, `/data/Station/search`). We used the legacy WQX2 profile; new USGS data after 2024 sit in the WQX3 profiles [U]. | In-situ labels | [H] |
| **AquaMatch chl-a** (USGS/EDI) | Harmonised WQP chl-a with three quality tiers (restrictive / narrowed / inclusive), 1970–2024. Part of the forthcoming AquaSat v2. DOC also released. | USA | not stated | **CC0** | DOI 10.6073/pasta/4c795593504d927ac546a36e5dc9fdb0 | Clean chl-a labels | [P] |
| **AquaSat** (Ross et al. 2019) | >600k Landsat–in-situ matchups (TSS, DOC, chl-a, Secchi) within ±1 day, 1984–2019 | USA | — | see paper | DOI 10.1029/2019WR024883 | Matchup pipeline template, teacher training | [V] |
| **Sentinel-2 L2A** (Planetary Computer `sentinel-2-l2a`) | 13 bands. 10 m: B2 0.49, B3 0.56, B4 0.665, B8 0.842 µm. 20 m red-edge: B5 0.704, etc. Centres from STAC. | Global, 2015– | per tile | Copernicus free and open (STAC field "proprietary") | **Anonymous** via STAC. Overpasses found within ±1 day (Utah Lake) and −2/+3 days (Maumee Bay) of NAIP dates. | Teacher domain, cross-scale pairs | [H] |
| **GLORIA** (Lehmann et al. 2023, *Sci. Data* 10:100) | 7,572 in-situ hyperspectral Rrs (350–900 nm), plus Lt/Lsky/Lw/Es, chl-a, TSS, aCDOM, Secchi, turbidity, QC flags; 54 datasets | Global inland and coastal | **56 MB zip** | **CC BY 4.0** | Anonymous: PANGAEA DOI 10.1594/PANGAEA.948492, direct zip | Physics pre-training, band-sufficiency, noise model | [H] |
| **NEON AOP + aquatic sites** | AOP: 10 cm RGB camera (DP1/DP3.30010.001); 1 m imaging-spectrometer reflectance (DP1/DP3.30006.001). Aquatic: water-quality sonde DP1.20288.001 (turbidity, chlorophyll, fDOM), Secchi DP1.20252.001, phytoplankton chlorophyll DP1.20163.001, chemistry DP1.20093.001. | 7 NEON lakes (BARC, SUGG, CRAM, LIRO, PRLA, PRPO, TOOK) plus rivers and streams. AOP flights roughly every 1–2 years per site (e.g., OSBS: 9 flights 2014–2026; LIRO: 5 flights 2017–2025). | Hyperspectral tiles are large (hundreds of MB to GB; not downloaded) | **CC BY 4.0** since NEON's 2026-06-30 change (previously CC0) | **Free account + API token now required for downloads** (data calls returned 403 without one). Product metadata is open. | Drone-scale out-of-distribution test: simulate the camera from 1 m hyperspectral data; sonde values at flight time | [H][P] |
| PlanetScope (Planet Education & Research) | 3 m, 4- or 8-band | Global, daily | — | Non-commercial research. Derived figures publishable with attribution. | **University-affiliated only** (Basic: free, 3,000 km²/month, 30-day delay). High-school students are not eligible on their own. | Optional, via a university mentor | [P] |
| **NOAA GLERL/CIGLR western Lake Erie HAB monitoring** | Weekly physical, chemical and biological samples (chl-a, phycocyanin, toxins, etc.) since 2012 | Western Lake Erie | — | US Government (NCEI) | DOI 10.25921/11da-3x54 | Scum/bloom labels near NAIP OH/MI dates | [V] |
| **MARIDA** (Kikaki et al. 2022) | Sentinel-2, 1,381 patches (256×256), 15 classes: marine debris, dense/sparse Sargassum, natural organic material, ship, clouds, cloud shadows, marine/turbid/**sediment-laden**/shallow/mixed water, **foam**, waves, wakes. **No oil class.** | Global coastal | 1.16 GB zip (4.38 GB unzipped) | **CC BY 4.0** | Zenodo 10.5281/zenodo.5151941 | Plume, turbid-water and foam classes using only S2 B2/B3/B4/B8 | [P] |
| **MADOS** (Kikaki et al. 2024) | Sentinel-2 marine pollution: 174 scenes (2015–2022), ~1.5 M annotated pixels; **oil spill**, debris and look-alike classes | Global | 4.04 GB | **CC BY 4.0** | Zenodo 10.5281/zenodo.10664073 | Oil and look-alike detection at 10 m | [P][A] |
| **LADOS** (Gkountakos et al. 2025, *Data* 10:117) | Low-altitude aerial images, 3,388, pixel labels in 6 classes incl. oil, **emulsion, sheen** | — | — | Article CC BY 4.0. **Dataset licence not confirmed.** | Via article / Zenodo 15888341 (which holds the article PDF) | Drone-scale oil segmentation (RGB) | [V][P] |
| **De Kerf et al. 2024** (*Sci. Data*) | Drone RGB, 1,268 images, labels oil / water / other (port) | Antwerp port | — | Check record | DOI 10.1038/s41597-024-03993-8 | Drone-scale oil (RGB) | [V][A] |
| **Chesapeake Land Cover** (Robinson et al. 2019; LILA) | **NAIP 4-band 1 m** + Landsat-8 + 1 m land cover incl. **water** | 6 state-years in the Chesapeake watershed, 732 tiles | ~140 GB zip | NAIP/Landsat public domain. Labels: "credit the source" (no formal licence). | Anonymous (GCP/AWS/Azure) | Water mask for the 4-band student | [P] |
| **FloodNet** (Rahnemoonfar et al. 2021) | UAV (DJI Mavic Pro) RGB, 2,343 images, 10 classes incl. water, pool, flooded | Hurricane Harvey | — | **CDLA-Permissive 1.0** | Dropbox via GitHub | Drone-scale water masks (RGB) | [P][V] |
| OpenEarthMap (Xia et al. 2023) | 5,000 aerial/satellite RGB images, 0.25–0.5 m, 8 classes incl. water | 97 regions, 44 countries | — | not confirmed [U] | open-earth-map.org | Water-mask generalisation | [V][A] |
| LoveDA (Wang et al. 2021) | 5,987 high-resolution images (RGB), urban/rural domains, incl. water | 3 cities, China | — | not confirmed [U] | GitHub/Zenodo | Domain-shift test | [V][A] |
| S1S2-Water (Wieland et al. 2024); Sen1Floods11 (Bonafilia et al. 2020); UAVid (Lyu et al. 2020) | Water masks (S2) / UAV urban classes | — | — | check | — | Auxiliary | [V] |
| USGS HIVIS (via HydroVision) | Fixed-camera RGB stills, >500k images 2022–2024; co-located sondes [U] | USGS monitoring sites | — | USGS (public) [U] | apps.usgs.gov/hivis (JavaScript page) | Ground-level RGB domain (discolouration, turbidity) | [A][U] |
| Tick Tick Bloom (DrivenData, 2022–23) | Cyanobacteria severity labels for small US water bodies | US | — | **Rules restrict data use to the competition**. An "Open data" tab exists but was not verified. | Login | Do not use unless open terms are confirmed | [P] |

### b.2 How many NAIP-date in-situ matchups exist? (hands-on, Minnesota)

**Method** [H]
1. List all NAIP quarter-quads for Minnesota per cycle through STAC (6,791 per cycle). Acquisition months: 2019 Jul–Oct, 2021 Jun–Aug, 2023 Jun–Sep. GSD: 0.6 m in 2019 and 2021; 0.3 m in 2023.
2. Pull Water Quality Portal results and stations for June–October.
3. Assign each station to the quad whose bounding box contains it, then count unique station-days within ±d days of that quad's acquisition date.

**Unique station-days**

| Variable | ±0 d | **±1 d** | ±3 d | ±1 d, lakes/reservoirs only |
|---|---|---|---|---|
| Secchi depth | 427 | **1,105** | 2,431 | 1,091 |
| Chl-a (any variant) | 133 | **424** | 1,025 | 349 |
| TSS | 140 | 360 | 953 | 21 (mostly rivers) |
| Turbidity | 27 | 109 | 347 | 38 |

These are upper bounds. Not yet filtered for:
- station coordinates falling on water pixels;
- cloud, haze or glint;
- depth (Secchi is integrative; chl-a sampling depth is variable).

A single-lake spot check (Utah Lake, NAIP 2018-09-09) found **no** samples within ±7 days. Matchups should therefore cluster where volunteer lake monitoring is dense: Minnesota is confirmed; WI, MI and others are not yet checked [U]. Scaling this to other volunteer-rich states and older cycles (1 m) should give several thousand Secchi matchups and >1,000 chl-a matchups. That is enough for held-out-lake evaluation of a student, but not for training a deep model from scratch. Hence distillation.

### b.3 Camera band mapping (canonical 4-band input)

| Canonical band | Proposed SENTINEL drone camera | NAIP | Sentinel-2 (10 m) | DJI P4M | DJI Mavic 3M | Notes |
|---|---|---|---|---|---|---|
| Blue | Camera Module 3 (IMX708, IR-cut) Bayer B | Blue | B2 (0.49 µm) | 450±16 nm | — (no blue) | Bayer bands are broad and overlapping; measure the response. |
| Green | Bayer G | Green | B3 (0.56) | 560±16 | 560±16 | |
| Red | Bayer R | Red | B4 (0.665) | 650±16 | 650±16 | |
| NIR | **Second camera: NoIR (no IR-cut) + long-pass filter**; NIR = mean of the RAW Bayer channels | NIR (sensor-specific) [U] | B8 (0.842; 10 m) | 840±26 | 860±26 | Choose ≥780–800 nm long-pass to match the NAIP/S2 NIR. A 720 nm long-pass leaks red-edge (helps chl-a a little) but breaks proxy consistency. The pilot's broad 720–800 band was no better than a narrow 783 nm band. |
| (Red-edge) | not available | — | B5–B7 (20 m) | 730±16 | 730±16 | Teacher only |

Supporting facts:
- Raspberry Pi docs [P]: Camera Module 2 and 3 come in NoIR versions; HQ, Global Shutter and AI Camera do not. NoIR units ship with a blue gel for "plant health" (Infragram-style). The Pi 5 has two camera connectors and supports two cameras.
- DJI specs [P]: P4M bands 450/560/650/730/840 nm; Mavic 3M bands 560/650/730/860 nm.
- The single-camera "blue gel" trick makes the red channel ≈ NIR but sacrifices clean green/red bands. **Not recommended** for water quality.

### b.4 Searched but not found (gaps)

- **No public drone dataset** combining RGB+NIR (or multispectral) imagery, calibration frames and co-located in-situ chl-a/turbidity was found. Search queries were run against Zenodo- and MDPI-indexed results (limited budget) [U]. The closest are paper-only studies (e.g., Salim et al. 2025, no data statement), oil datasets (LADOS, De Kerf) and DroneWQ's software archive.
- **No peer-reviewed NAIP + in-situ water-quality matchup study** was found in two targeted searches [U].
- **No NEON-AOP water-quality retrieval study** was found in one search [U].
- **No RGB→hyperspectral reconstruction validated for water-quality retrieval** was found (§c.3) [U].

---

## (c) Distillation and cross-sensor transfer

### c.1 Framing

This is **learning with privileged information**: the teacher sees red-edge (and SWIR/hyperspectral) bands at training time; the student sees only R, G, B, NIR at test time (Vapnik & Vashist 2009; Lopez-Paz et al. 2016 unify distillation with privileged information) [V].

Three gaps must be bridged:
1. **Spectral:** the red-edge information is lost. §a.4 quantifies it, and no method can recreate information absent from the input.
2. **Radiometric:** satellite surface reflectance with atmospheric-correction error, vs NAIP DN, vs drone RAW with glint.
3. **Spatial:** 10–30 m teacher vs 0.05–1 m student.

### c.2 Methods that apply

| Method | Idea | Use here | Ref (tag) |
|---|---|---|---|
| Response KD (Hinton et al. 2015) | Match softened teacher outputs | Segmentation heads (scum, plume, sheen) | arXiv:1503.02531 [V] |
| Hint / feature KD (FitNets) | Regress student features onto teacher features | Only when teacher and student see the same scene at similar GSD (e.g., S2 teacher → S2-4-band student) | arXiv:1412.6550 [V] |
| Cross-modal distillation (Gupta, Hoffman & Malik 2016) | Supervision transfer between paired modalities | Paired S2-full ↔ S2-4band / NAIP | arXiv:1507.00448 [V] |
| Modality hallucination (Hoffman et al. 2016) | Student learns to hallucinate the missing modality's features | Auxiliary "red-edge feature" branch | CVPR 2016, DOI 10.1109/CVPR.2016.96 [V] |
| Structured KD for dense prediction (Liu et al. 2019) | Pairwise and holistic distillation | Segmentation maps | arXiv:1903.04197 [V] |
| Contrastive representation distillation (Tian et al. 2020) | Mutual-information-based KD | Pre-training the student encoder | arXiv:1910.10699 [V] |
| **Regression KD with teacher trust** (Saputra et al. 2019) | Weight imitation by the teacher's own error ("attentive imitation loss") | Down-weight teacher pixels or regions where the teacher's in-situ error is high | arXiv:1908.00858 [V] |
| Heteroscedastic uncertainty (Kendall & Gal 2017) | Predict mean and variance; NLL loss | Teacher and student both output (μ, σ); distil with σ-aware NLL | arXiv:1703.04977 [V] |
| Remote-sensing cross-modality learning (Hong et al. 2021; X-ModalNet, Hong et al. 2020) | One modality absent at test time | Design reference | DOI 10.1109/TGRS.2020.3016820; 10.1016/j.isprsjprs.2020.06.014 [V] |
| DisCoM-KD (Ienco & Fraga Dantas 2024) | Multimodal data → single-modal classifiers via disentanglement + adversarial learning, without a separate teacher. The remote-sensing benchmark list (EuroSAT-MS-SAR) comes from a search snippet [U]. | Design reference | arXiv:2408.07080 (BMVC 2024) [V] |
| On-board panoptic segmentation (Fernando et al. 2022) | Multimodal teacher → uni-modal (single-frame) student via online KD, aimed at on-board satellite processing | Edge-oriented KD reference | arXiv:2204.01952 [V][A] |
| Dual-teacher distillation (Wolf et al. 2026) | Multispectral teacher + optical VFM teacher → one student; +3.64 pp segmentation | Pre-training option | arXiv:2602.19863 (CVPR 2026) [V][A] |

### c.3 Spectral super-resolution (RGB → hyperspectral)

**What exists**

- Sparse recovery (Arad & Ben-Shahar 2016) [V]; HSCNN+ (Shi et al. 2018) [V]; AWAN, which uses the camera response as a prior (Li et al. 2020) [V].
- MST++ won NTIRE 2022 spectral reconstruction (Cai et al. 2022) [V].
- NTIRE 2020 (Arad et al.) has a "Real World" track with noisy JPEG input from an unknown camera [V][A].
- In remote sensing:
  - Sentinel-2 20 m bands (incl. red-edge) can be super-resolved to 10 m using the 10 m bands (Lanaras et al. 2018) [V];
  - a GAN-generated NIR band from RGB raised forest segmentation F1 from 0.914 to 0.947 (Illarionova et al. 2021) [V][A].

**Limits for water**
- Recovery from RGB is ill-posed (metamerism). It works when the scene's spectra lie on a low-dimensional manifold learned from training data, and degrades out of distribution.
- Water Rrs is small (often <0.01 sr⁻¹) and dominated by glint errors.
- The chl-a red-edge feature is a small perturbation that RGB+NIR channels barely constrain.
- In §a.4, the red-edge sets beat RGB+NIR by about 20+ points even with a flexible model. A hallucinated red-edge band can only reproduce correlations that the RGB+NIR bands already carry.
- We found **no** peer-reviewed validation of RGB→hyperspectral reconstruction for water-quality retrieval [U].

**Use spectral super-resolution only as an auxiliary pre-training target.** For example, predict S2 B5/B6 block means from student features. Report its error, and never present reconstructed bands as measurements.

### c.4 Foundation models as teachers or encoders

| Model | Inputs | Notes for this project | Ref |
|---|---|---|---|
| **DOFA** (Xiong et al. 2024) | Any bands, via a **wavelength-conditioned** dynamic hypernetwork | Natural fit for "RGB+NIR with known wavelengths". Weights on GitHub (licence not checked [U]). | arXiv:2403.15356 [V] |
| Copernicus-FM (Wang et al. 2025, ICCV) | Copernicus sensors, spectral/variable inputs | Teacher features for S2 | arXiv:2503.11849 [V] |
| Prithvi-EO-2.0 (Szwarcman et al. 2024) | HLS 6-band, multi-temporal, 30 m | Coarse teacher | arXiv:2412.02732 [V] |
| SpectralGPT (Hong et al. 2024, TPAMI) | S2 spectral tokens | S2 teacher | arXiv:2311.07113 [V] |
| Galileo; TerraMind; AnySat; CROMA; SatMAE | Multimodal, multi-scale | Optional | arXiv:2502.09356; 2504.11171; 2412.14123; 2311.00566; 2207.08051 [V] |
| Scale-MAE (Reed et al. 2023, ICCV) | GSD-aware positional encoding | **Cross-resolution** pre-training idea | arXiv:2212.14532 [V] |
| DINOv3 (Siméoni et al. 2025) | RGB (natural and satellite variants) | Dense RGB features; no NIR | arXiv:2508.10104 [V] |

None of these has a published, validated chl-a/TSS regression benchmark that we found. Use them for **segmentation and feature pre-training**. For concentrations, use a dedicated pixel teacher (MDN-style; Pahlevan et al. 2020, 2022) trained on in-situ data.

### c.5 Cross-resolution and cross-sensor transfer

- Water-leaving reflectance is an intensive quantity: away from shores, a 10 m block mean of a calibrated 0.5 m field ≈ the 10 m value. That makes **block-average consistency** a sound distillation constraint for open water.
- The constraint breaks near shores (adjacency effects, mixed pixels), in shallow water (bottom reflectance) and for sub-pixel features (scums, sheens).
- Cillero Castro et al. 2020 [A] found moderately good agreement among MicaSense (8 cm), S2 (10 m) and L8 (30 m) chl-a. MapEO frames drones as the link between satellites and in-situ data [A].
- Cross-sensor consistency of atmospheric correction is itself a large error source over water (ACIX-Aqua; Pahlevan et al. 2021) [V]. Use ACOLITE (dark spectrum fitting; Vanhellemont & Ruddick 2018) [V] for S2 teacher inputs.

### c.6 Concrete recipes for the 4-band student

**R1: Physics-first pixel distillation (cheapest, most honest)**
1. Convolve GLORIA spectra with (a) S2-MSI response curves and (b) the measured camera response curves.
2. Train teacher T on (a), with red-edge, outputting μ and σ for log chl-a, log TSS and log Secchi.
3. Train the pixel student S on (b) with a mixed loss: NLL against in-situ values + λ·σ-weighted NLL against teacher μ.
4. Apply radiometric-noise augmentation (§a.4).
5. Gives a per-pixel model (the "Pixel-MLP", 1.6k parameters) that later initialises the dense student's head.

**R2: Cross-scale pseudo-label distillation (main recipe)**
1. For each NAIP quad, find an S2 L2A scene within ±1 day (0 days preferred), ≤10% cloud.
2. Run T on ACOLITE-corrected S2 to get maps of μ_T, σ_T at 20 m.
3. The dense student S runs on NAIP (0.3–1 m). Average its outputs over water-only pixels in each S2 cell.
4. Loss: Gaussian NLL of the block mean against (μ_T, σ_T² + σ_scale²), plus **in-situ NLL at WQP points (±1 day)**, plus a total-variation prior.
5. Weight teacher terms by teacher trust: the teacher's held-out in-situ error for that lake type or region.

**R3: Spectral-hallucination auxiliary head.** S additionally predicts S2 B5/B6 block reflectance (L2 loss). This gives a red-edge-aware representation (Hoffman-style). The head is dropped at export, and the claim is limited to "auxiliary".

**R4: Detection heads via soft labels.**
1. Fine-tune a foundation teacher (DOFA or Copernicus-FM) on MARIDA/MADOS with S2 B2/B3/B4/B8 *and* full bands.
2. Distil its soft class maps into S's segmentation head on S2 4-band and on NAIP (upsampled).
3. Fine-tune on drone-scale RGB sets (LADOS, De Kerf, FloodNet) with **NIR-channel dropout**: the NIR channel is zeroed and flagged when missing.

**R5 (optional): Contrastive or dual-teacher pre-training** (CRD; Wolf et al. 2026) on unlabeled NAIP+S2 pairs before R2.

**Guarding against learning teacher artefacts**
- Headline metrics are always **student vs in-situ** on held-out lakes, never student vs teacher.
- Exclude teacher pixels flagged for glint, cloud, adjacency or shallow water.
- Report the teacher's own in-situ error on the same held-out lakes.
- Ablate: no-KD vs KD; block constraint on vs off.

---

## (d) Edge deployment

### d.1 Hardware (vendor pages opened unless noted)

| Device | Compute | Power | Price | Notes | Tag |
|---|---|---|---|---|---|
| **Raspberry Pi 5** | 4× Cortex-A76 @ 2.4 GHz; LPDDR4X 1–16 GB | 5 V / 5 A USB-C supply recommended | 16 GB listed at $305 (2026 page) | Two 4-lane MIPI camera/display connectors (two cameras supported); **PCIe 2.0 ×1** | [P] |
| **Raspberry Pi AI HAT+** | Hailo-8L **13 TOPS** or Hailo-8 **26 TOPS** | not stated | **from $70** | Pi 5 only; integrated with the Pi camera stack; in production until at least Jan 2030 | [P] |
| Raspberry Pi AI Camera | Sony IMX500 with on-sensor accelerator (12.3 MP, 78.3° FoV) | — | $70 | **No NoIR version**; TOPS and model-size limits not stated. Useful only for the RGB stream. | [P] |
| **Jetson Orin Nano Super Dev Kit** (Dec 2024) | **67 TOPS sparse / 33 TOPS dense INT8**; GPU 1,020 MHz; 102 GB/s | 7 / 15 / **25 W** modes | **$249** (was $499) | Existing Orin Nano kits get the Super mode via JetPack 6.1+ | [P] |
| Jetson Orin NX | up to 157 TOPS | — | — | Heavier payload | [P] |
| Coral Edge TPU (USB/M.2) | 4 TOPS [U] | ~2 W [U] | — | **coral.ai now redirects to developers.google.com/coral**, which describes a new RISC-V "Coral" platform. The status of the legacy Edge TPU tools (TF 2.x, full-INT8 TFLite only) is unclear. **Not recommended** for a new project. | [P][U] |

### d.2 Published accelerator benchmarks (for scale, not our model)

**Hailo-8L, Hailo Model Zoo** [P]
- Conditions: Dataflow Compiler v2.19.0, host i5-9400, **PCIe Gen3 ×4**. A Pi 5 link is PCIe 2.0 ×1 by default, so expect lower throughput on a Pi.
- Classification, batch 1: MobileNetV3 **1,935 FPS** (224², quantised top-1 71.8 vs float 72.2); EfficientNet-Lite0 **1,057 FPS** (75.0 → 73.9); ResNet-18 1,281 FPS.
- Segmentation, batch 1: unet_mobilenet_v2 256² **99.9 FPS**; deeplab_v3_mobilenet_v2 513² **51.8 FPS** (mIoU 76.0 → 74.6 after quantisation); segformer_b0_bn 512×1024 **11.4 FPS**.

**Coral Edge TPU, legacy benchmark page** [P]

| Model (input) | USB Accelerator | Dev Board | Xeon Gold 6154 CPU (same TFLite model) |
|---|---|---|---|
| MobileNet v2 (224²) | 2.6 ms | 2.6 ms | 51 ms |
| U-Net-MobileNetV2 (128²) | 3.3 ms | 5.7 ms | — |
| DeepLab v3 (513²) | 52 ms | 241 ms | — |

**Jetson / Pi 5 drone pipeline** (Rey et al. 2025, *Electronics* 14:638; arXiv:2502.15737) [V][A]: YOLOv8n reached 52 FPS on Orin NX and 65 FPS with INT8 PTQ. The **Raspberry Pi 5 CPU failed real-time requirements**.

### d.3 Desktop latency *proxy* for candidate students (this session) [H]

**Setup**
- **Input and models:** 4-channel 512×512 input. Each model is a timm encoder (features_only, `in_chans=4`) plus a 48-channel FPN head with 13 outputs (4 targets × (μ, log σ²) + 5 classes) at stride 4. Random weights; latency and size only.
- **Export and runtime:** ONNX opset 17 → ONNX Runtime 1.31 CPU execution provider, batch 1, sequential execution.
- **Timing:** median of 30 runs after 5 warm-ups.
- **Machine:** Intel **Xeon Silver 4214 @ 2.2 GHz** (AVX-512 VNNI), pinned with `taskset`, shared server.
- **INT8:** static QDQ, per-channel weights, after `quant_pre_process`, calibrated on 16 random tensors. "Rel. err" is ‖y_int8 − y_fp32‖ / ‖y_fp32‖ on random weights, a sanity check only.

| Student (encoder + FPN head) | Params (M) | GMACs @512²×4 | FP32 1-thr (ms) | FP32 4-thr | **INT8 1-thr** | INT8 4-thr | INT8 ONNX (MB) | Rel. err |
|---|---|---|---|---|---|---|---|---|
| Pixel-MLP (1×1 convs, 32 hidden) | 0.0016 | 0.41 | 25.4 | 10.6 | 17.7 | 6.1 | 0.01 | 0.006 |
| Tiny U-Net (16 base ch, depthwise-separable) | 0.03 | 0.28 | 21.0 | 7.4 | 20.8 | 9.0 | 0.07 | 0.006 |
| MobileNetV3-Small | 0.96 | 0.38 | 20.7 | 7.1 | 25.9 ✗ | 13.3 | 1.3 | 0.012 |
| MobileNetV3-Large | 3.03 | 1.22 | 58.0 | 17.4 | 52.4 | 23.3 | 3.5 | 0.020 |
| **MobileNetV4-Conv-Small** | 1.32 | 1.09 | 38.8 | 12.1 | **19.4** | **6.8** | 1.6 | 0.038 |
| **EfficientNet-Lite0** | 2.99 | 2.01 | 93.7 | 27.7 | **41.7** | 13.5 | 3.4 | 0.058 |
| EfficientViT-B0 (MIT) | 0.70 | 0.56 | 37.4 | 16.6 | 45.4 ✗ | 26.4 | 1.1 | 0.010 |
| RepViT-M0.9 | 4.76 | 4.40 | 197.6 | 60.7 | — | — | — | — |
| ResNet-18 (reference) | 11.23 | 9.83 | 314.2 | 88.2 | — | — | — | — |

*✗ = INT8 slower than FP32. RepViT and ResNet-18 were not re-run with pre-processing; without `quant_pre_process`, INT8 was slower than FP32 for every backbone model, e.g., MobileNetV3-L 113 ms vs 58 ms.*

**Takeaways**
1. ReLU/ReLU6 convolutional backbones (MobileNetV4-Conv, EfficientNet-Lite) get about 2× from INT8 on CPU. Hard-swish/SE (MobileNetV3) and attention (EfficientViT) do not, at least on ONNX Runtime CPU.
2. The INT8 pipeline matters: without BN-folding and pre-processing, INT8 was 1.0–2.4× *slower* than FP32.
3. These are x86 numbers. A Cortex-A76 core is slower, and the Hailo/Jetson numbers depend on their compilers, so **only on-device measurements count**.

### d.4 Toolchain notes (torch 2.7 / CUDA 11.8 / 2080 Ti workstation)

- **Tested here:** PyTorch → ONNX (TorchScript exporter, `dynamo=False`) and ORT quantisation work with torch 2.7.1+cu118 and timm 1.0.22 [H].
- **TensorRT 11.4** supports compute capability SM 7.5+ (Turing included) but is built against CUDA 12.x only [P]. This does not matter: TensorRT engines are device-specific and should be built *on the Jetson* (JetPack 6.x).
- **TFLite path:** LiteRT Torch (formerly AI Edge Torch; beta; Linux; torch ≥2.4; Apache-2.0) [P], or onnx2tf (MIT) [P]. Neither was tested here.
- **Hailo:** the Dataflow Compiler and HailoRT sit behind the Hailo Developer Zone login [P]. Version and OS requirements were not verified [U]. Hailo compiles ONNX to HEF and runs INT8 with its own calibration.

### d.5 Honest benchmarking protocol

1. **Report** the model card: params, MACs (state the counter: `torch.utils.flop_counter`, which counts conv/matmul only), input size, output stride, opset, runtime version and precision.
2. **On desktop:**
   - ORT CPU EP, `intra_op_num_threads ∈ {1, 4}`, `inter_op=1`, sequential execution, graph optimisation on;
   - batch 1, fixed shape, pinned cores (`taskset`), 5–10 warm-ups, ≥30 timed runs, report median and p90;
   - log CPU model, frequency governor and background load.

   Label these "desktop proxy". Never quote GPU fp16 throughput as edge latency.
3. **On device:**
   - **Pi 5 CPU:** ORT or LiteRT/XNNPACK with 4 threads and active cooler, measured after 10 minutes of thermal soak;
   - **Hailo:** `hailortcli benchmark` [U: tool name from Hailo docs, not opened];
   - **Jetson:** `trtexec` plus `tegrastats` for power.

   Measure end-to-end time (RAW capture → debayer/radiometric correction → tiling → inference → post-processing) separately from model-only time. Measure board power with a USB-C power meter or an INA-type sensor.
4. **Accuracy after quantisation:** evaluate the INT8 model on the *held-out lakes* and report Δ MdSA vs FP32. Keep the final regression 1×1 conv in FP16/FP32 if Δ > 2 points; Hailo and TensorRT allow mixed precision [U].

### d.6 Budget for the drone student

| Item | Budget |
|---|---|
| Model size | ≤3 M params; INT8 ≤4 MB |
| Compute | ≤2 GMACs per 512×512×4 tile; ReLU/ReLU6 conv backbone (MobileNetV4-Conv-S or EfficientNet-Lite0) |
| Latency | ≤100 ms per tile on device (Pi 5 + Hailo-8L, or Pi 5 CPU INT8 as fallback) |
| Rate | 0.5–2 frames/s is enough for mapping. Illustration: at 50 m altitude with an assumed ~66°×41° FoV [U], the footprint is ≈65 m × 37 m. At 5 m/s, one frame every 2–3 s advances 10–15 m, which still gives ≥60% forward overlap along the short side. |
| Power | ≤5–8 W added (AI HAT+ + Pi 5) [U: no vendor power figure found] |
| Fallback | Ground-station inference over the existing ROS 2/DDS link (current `camera-model` design) for the first flights |

---

## (e) Recommended plan

### e.1 Scope and outputs (what the student predicts)

**Per-pixel regression** (stride 4) with uncertainty, μ and log σ²:
- log₁₀ Secchi depth;
- log₁₀ turbidity/TSS;
- log₁₀ chl-a, reported as a **trophic class** unless held-out MdSA beats the baselines;
- optionally a Forel-Ule-type hue index.

**Per-pixel classes:**
- water (vs land, vegetation, shadow, glint);
- surface scum / floating algae;
- sediment plume / discoloured water;
- oil sheen / slick;
- "unknown/abstain".

**Not predicted:** DO, pH, SpCond, temperature, TN/TP, microcystin. Remove them from the drone track.

### e.2 Sensor and input spec

- **Cameras:**
  - **RGB:** Pi Camera Module 3 (IMX708, standard IR-cut; the current branch uses the Wide variant).
  - **NIR:** Camera Module 3 **NoIR** (preferred, same sensor and optics family) or the planned NoIR v2 (IMX219), fitted with an **800 nm (or 780–850 nm) long-pass** filter. NIR = mean of the four RAW Bayer channels. Pagnutti et al. 2017 measured the spectral response of the *standard* V2 module; the NoIR response beyond ~700 nm and the filter transmission must be measured.
  - Hardware-trigger or frame-sync both cameras on the Pi 5's two CSI ports.
  - Same lens field of view, if possible, to simplify co-registration.
- **Capture:** RAW 10/12-bit, fixed exposure, gain and colour gains; no auto white balance; lens-shading correction disabled (replaced by our flat-field).
- **Calibration:** dark/linearity/flat-field (SPECTACLE methods); spectral response measurement of both cameras (Pagnutti et al. 2017 gives the standard IMX219 V2 as a starting point); panel before and after each flight; sky frames at 40°/135°.
- **Model input:** 4 channels of **Rrs-like reflectance** (sr⁻¹) after ρ·Lsky removal and NIR residual correction, plus a per-pixel validity mask. Training-time inputs from NAIP, S2 and NEON are mapped into the same space (§e.3).

### e.3 Data pipeline (all public)

| Stage | Data | Purpose | Notes |
|---|---|---|---|
| S0 | **GLORIA** (CC BY 4.0) | Physics pre-training of pixel teacher and student; band-sufficiency; noise model | Measured camera response curves once available |
| S1 | **Sentinel-2 L2A/L1C** (ACOLITE) + **AquaMatch/WQP** in-situ (±1 day, later ±0) | Train and validate the **teacher** (red-edge) | Leave-lake-out and leave-ecoregion-out. Report teacher skill. |
| S2 | **NAIP** (0.3–1 m, 2010–2023) paired with S2 within ±1 day | **Cross-scale distillation** (R2) | Per-quad radiometric normalisation: fit DN→reflectance gains/offsets against the co-incident S2 over large homogeneous water and land targets, then add noise augmentation. Use the 2023 cycle as a temporal test. |
| S3 | NAIP ↔ WQP/AquaMatch (±1 day) | **Direct in-situ supervision and evaluation** of the student | ~1.1k Secchi / ~0.4k chl-a site-days in Minnesota alone. Add WI, MI, IA, NY, VT, ME [U]. |
| S4 | MARIDA, MADOS (S2); LADOS, De Kerf (drone RGB); FloodNet, Chesapeake (water masks); NOAA GLERL Lake Erie (bloom context) | **Hazard heads** (R4) | Use NIR-dropout for RGB-only sets. Never train or evaluate oil on S2-only look-alikes without drone-scale tests. |
| S5 | **NEON AOP** (free account + token): 1 m hyperspectral → simulate camera bands with measured response curves; 10 cm RGB; aquatic sonde at flight time | **Drone-scale out-of-distribution test** without flying | Lakes BARC, SUGG, CRAM, LIRO, PRLA, PRPO, TOOK. Confirm flight boxes cover each lake. Small n, high temporal fidelity. |
| S6 (opt.) | One or two validation flights: RAW + panel + sky frames + 20–40 in-situ samples (Secchi, turbidity meter, chl-a), ideally on an S2 overpass day | Real camera validation | Pre-register the analysis |

### e.4 Teachers

- **T1, S2 pixel/patch teacher** with red-edge.
  - Inputs: ACOLITE Rrs B1–B8A.
  - Head: MDN-style or heteroscedastic Gaussian.
  - Trained on GLORIA (MSI-convolved) + S2–in-situ matchups.
  - The published MDN code could serve as a baseline teacher (licence and weights not checked [U]).
- **T2, segmentation teacher.** DOFA or Copernicus-FM fine-tuned on MARIDA/MADOS. Use T2 only where it beats a U-Net trained from scratch.
- Compute: everything fits on 2080 Ti GPUs in fp16. DOFA ViT-B fine-tuning needs gradient checkpointing at batch ~16 [U].

### e.5 Student architectures (shortlist)

1. **Pixel-MLP** (1.6k params). Interpretable baseline and per-pixel fallback; exportable anywhere.
2. **MobileNetV4-Conv-Small + 48-ch FPN** (1.3 M, 1.1 GMACs, INT8 1.6 MB). Primary.
3. **EfficientNet-Lite0 + FPN** (3.0 M, 2.0 GMACs). Accuracy candidate; Hailo-proven backbone family.
4. **Tiny U-Net** (0.03 M). Ablation for "does context help?"

Avoid hard-swish/SE/attention backbones for the INT8 build unless the target compiler handles them; re-test on Hailo. Add a 4-channel stem with **per-band normalisation from calibrated reflectance**, not ImageNet statistics.

### e.6 Losses

| Term | Formula sketch | When |
|---|---|---|
| L_insitu | Gaussian NLL of log target at in-situ pixels (3×3 water-only median) | Wherever matchups exist |
| L_KD | Σ_cells w_trust · NLL(block-mean(μ_S); μ_T, σ_T² + σ_scale²) + KL on σ | S2↔NAIP pairs |
| L_aux-RE | ‖B5/B6 predicted − S2 B5/B6‖² on blocks | Pre-training only |
| L_seg | Focal + Dice; soft-label KD (T = 2–4) from T2 | Hazard heads |
| L_cons | ‖f(x) − f(aug(x))‖ for photometric/glint/haze augmentations, with σ allowed to grow | Always |
| L_TV | Total variation on μ maps inside water | Light weight |

### e.7 Evaluation protocol (pre-register)

- **Splits:**
  - by water body (GroupKFold);
  - by region (HUC-2 or ecoregion);
  - by time (train ≤2021 NAIP cycles, test 2023);
  - NEON lakes as a fully external test.

  Never use random pixel or random site-date splits for headline numbers.
- **Metrics:**
  - regression: MdSA, signed bias (SSPB), RMSE and R² in log space, slope, and **prediction-interval coverage** (calibration of σ);
  - classes: per-event precision/recall/F1, IoU, false alarms per km² of clean water.
- **Baselines, all on the same pixels and dates:**
  1. climatology (lake or regional median);
  2. band-ratio models: Secchi via blue/red + red (Kloiber/Olmanson form); turbidity via red/NIR single band (Nechad/Dogliotti); chl-a via blue/green OC-type and NDVI-type; scum via NDVI/FAI-like NIR index;
  3. gradient boosting on band ratios;
  4. **student trained without KD**;
  5. **teacher on S2** (upper bound with red-edge);
  6. S2-4-band student at 10 m (same bands, satellite scale).
- **Reported gaps:**
  - red-edge gap = teacher − S2-4-band student;
  - scale gap = S2-4-band − NAIP student;
  - KD gain = KD − no-KD;
  - INT8 Δ.

### e.8 Robustness suite (synthetic + real)

1. **Sun glint:** Cox–Munk-style specular fields injected at 0–40% of the frame. Also real midday NAIP glint patches.
2. **Sky glint / ρ error:** ρ ∈ [0.02, 0.05] with blue-weighted sky spectra.
3. **Haze / path radiance:** additive offsets (satellite-trained inputs).
4. **Calibration:** per-band gain ±5–20%, offset ±5×10⁻⁴ sr⁻¹, JPEG Q70–95, 8-bit vs 12-bit, vignetting up to 2.8× (SPECTACLE).
5. **Resolution:** 0.1/0.3/0.6/1/3/10 m, checking block-mean consistency.
6. **Band misregistration:** NIR shifted 1–10 px, rotated ±1°. Dual-camera timing and parallax make this real.
7. **Confounders:**
   - shoreline adjacency and tree shadows;
   - shallow bottoms;
   - emergent and floating vegetation (scum look-alikes);
   - foam, wakes and boats (oil look-alikes);
   - off-nadir view (30–40°).

Report error vs perturbation curves and check that σ increases or the model abstains rather than silently failing.

### e.9 Deployment target

- **v1:** Pi 5 (8 GB) + AI HAT+ 13 TOPS on the drone, running MobileNetV4-Conv-S-FPN as INT8 HEF. Radiometric pre-processing runs on the Pi CPU.
  - Messages: publish μ, σ and class maps over ROS 2 at 0.5–2 Hz; log RAW for re-processing.
- **v0 / fallback:** stream RAW to the ground computer (existing `sentinel_inference` node).
- **v2 (optional):** Jetson Orin Nano Super if heavier models are needed.

### e.10 Milestones and go/no-go

| Weeks | Deliverable | Go/no-go |
|---|---|---|
| 1–2 | Camera response curves measured; GLORIA band study with measured curves | Proceed with the chl-a regression head only if held-out MdSA ≤ 60% on GLORIA camera bands. Otherwise report trophic classes. The pilot's 71–91% makes "classes only" the likely outcome. |
| 3–6 | S2–in-situ and NAIP–in-situ matchup sets (≥5 states), water masks, QA flags; teacher trained | Teacher beats band-ratio baselines on held-out lakes |
| 7–12 | Student via R1→R2(+R3, R4); full evaluation and robustness suite | Student beats climatology and band ratios on held-out lakes for Secchi and turbidity |
| 13–14 | INT8 export; Pi 5 / Hailo benchmarks; NEON external test | INT8 Δ ≤2 MdSA points; ≤100 ms/tile |
| opt. | Validation flight | — |

### e.11 What "first-in-class" can honestly mean

**Defensible, if the results hold and a final Scopus/Scholar check finds no prior example:**

> "To our knowledge (searched: …, October 2026), this is the first openly released water-quality model for a low-cost dual RGB+NIR camera that is **distilled from a red-edge-capable Sentinel-2 teacher using only public data**. It is **evaluated on lakes held out from training against in-situ measurements**, with the information lost by dropping red-edge (X points MdSA for chl-a) and the gain from distillation reported separately. Its INT8 latency is measured on a $70 edge accelerator."

Optional dataset claim: "the first public NAIP–in-situ water-quality matchup dataset". Our two searches found none [U].

**Do not claim:**
- "first drone/UAV water-quality model" (many exist, e.g., Su & Chou 2015; Cheng et al. 2020);
- "first RGB water-quality model" (HydroColor, HydroVision);
- "first Raspberry Pi RGB+NoIR drone" (Belcore et al. 2019);
- any non-optical parameter from imagery;
- "drone-validated" without flights;
- skill from random splits;
- comparisons to satellites at different pixels or times;
- "early warning" without lead-time analysis and placebo tests (the SENTINEL Track-A lesson).

---

## (f) Risks and fallbacks

| Risk | Likelihood | Impact | Mitigation / fallback |
|---|---|---|---|
| NAIP radiometry (uncalibrated, vendor colour balancing) defeats regression transfer | High | High | Per-quad S2-anchored normalisation; ratio features; noise augmentation. Fallback: train the 4-band student on S2 B2/B3/B4/B8 at 10 m (same bands, satellite scale) and use NAIP only for masks and detection. |
| Too few clean NAIP–in-situ matchups for chl-a | Medium | Medium | Expand states and cycles; ±3-day sensitivity analysis; report trophic classes; Secchi and turbidity as primary products |
| Teacher errors (atmospheric correction over water) propagate | Medium | High | ACOLITE; trust weighting; exclude adjacency and glint pixels; always evaluate against in-situ |
| Camera response unknown or mismatched (IMX708 NoIR + filter) | Medium | Medium | Measure it (SPECTACLE DIY methods); start from the published IMX219 (V2) visible response; response-curve jitter augmentation |
| Glint and sky reflection dominate drone frames | High | High | Flight geometry, sky frames, NIR correction, glint masks, abstention via σ |
| Dual-camera misregistration and timing | Medium | Medium | Frame sync; homography from shoreline features; misregistration augmentation; pixel features at stride 4 |
| Hailo toolchain access or op support | Medium | Low | Pi 5 CPU INT8 (ORT/LiteRT) at 0.5 Hz; Jetson Orin Nano Super; ground inference |
| NEON account/token or age policy | Low–medium | Low | Skip S5 or have an adult co-author register |
| Dataset licences (LADOS, OpenEarthMap, LoveDA unconfirmed; Tick Tick Bloom restricted) | Medium | Medium | Use only confirmed licences in released weights; document in a data card |
| Over-claiming (the project's history) | — | Very high | Pre-registered splits; publish code, splits and negative results; tag every number with its source and split |

---

## (g) References

*Tag: [V] identifier metadata verified; [A] abstract read; [P] page opened. DOIs/arXiv IDs were checked in this session.*

### UAV, camera and in-situ water-quality studies
- Arango, J.G., Nairn, R.W. (2020). Prediction of optical and non-optical water quality parameters in oligotrophic and eutrophic aquatic systems using a small unmanned aerial system. *Drones* 4(1):1. doi:10.3390/drones4010001 [V][A]
- Becker, R.H., Sayers, M., Dehm, D., Shuchman, R., Quintero, K., Bosse, K., Sawtell, R. (2019). Unmanned aerial system based spectroradiometer for monitoring harmful algal blooms: A new paradigm in water quality monitoring. *J. Great Lakes Res.* 45(3):444–453. doi:10.1016/j.jglr.2019.03.006 [V][A]
- Belcore, E., Piras, M., Pezzoli, A., Massazza, G., Rosso, M. (2019). Raspberry Pi 3 multispectral low-cost sensor for UAV based remote sensing. Case study in South-west Niger. *ISPRS Arch.* XLII-2/W13:207–214. doi:10.5194/isprs-archives-XLII-2-W13-207-2019 [V][A]
- Burggraaff, O., Schmidt, N., Zamorano, J., Pauly, K., Pascual, S., Tapia, C., Spyrakos, E., Snik, F. (2019). Standardized spectral and radiometric calibration of consumer cameras. *Opt. Express* 27(14):19075. doi:10.1364/OE.27.019075 [V][A]
- Burggraaff, O., Werther, M., Boss, E.S., Simis, S.G.H., Snik, F. (2022). Accuracy and reproducibility of above-water radiometry with calibrated smartphone cameras using RAW data. *Front. Remote Sens.* 3:940096. doi:10.3389/frsen.2022.940096 [V][A]
- Cheng, K.H., Chan, S.N., Lee, J.H.W. (2020). Remote sensing of coastal algal blooms using unmanned aerial vehicles (UAVs). *Mar. Pollut. Bull.* 152:110889. doi:10.1016/j.marpolbul.2020.110889 [V][A]
- Cillero Castro, C., Domínguez Gómez, J.A., Delgado Martín, J., Hinojo Sánchez, B.A., Cereijo Arango, J.L., Cheda Tuya, F.A., Díaz-Varela, R. (2020). An UAV and satellite multispectral data approach to monitor water quality in small reservoirs. *Remote Sens.* 12(9):1514. doi:10.3390/rs12091514 [V][A]
- Daniels, L., Eeckhout, E., Wieme, J., Dejaegher, Y., Audenaert, K., Maes, W.H. (2023). Identifying the optimal radiometric calibration method for UAV-based multispectral imaging. *Remote Sens.* 15(11):2909. doi:10.3390/rs15112909 [V][A]
- De Keukelaere, L., Moelans, R., Knaeps, E., Sterckx, S., Reusen, I., De Munck, D., Simis, S.G.H., et al. (2023). Airborne drones for water quality mapping in inland, transitional and coastal waters—MapEO water data processing and validation. *Remote Sens.* 15(5):1345. doi:10.3390/rs15051345 [V][A]
- Deshmukh, S.L., Wilchek, M., Batarseh, F.A. (2025). HydroVision: Predicting optically active parameters in surface water using computer vision. arXiv:2509.01882 [V][A]
- Ehmann, K., Kelleher, C., Condon, L.E. (2019). Monitoring turbidity from above: Deploying small unoccupied aerial vehicles to image in-stream turbidity. *Hydrol. Process.* 33(6):1013–1021. doi:10.1002/hyp.13372 [V][A]
- Gao, M., Li, J., Wang, S., Zhang, F., Yan, K., Yin, Z., Xie, Y., Shen, W. (2022). Smartphone–camera–based water reflectance measurement and typical water quality parameter inversion. *Remote Sens.* 14(6):1371. doi:10.3390/rs14061371 [V][A]
- Gray, P.C., Windle, A.E., Dale, J., Savelyev, I.B., Johnson, Z.I., Silsbe, G.M., Larsen, G.D., Johnston, D.W. (2022). Robust ocean color from drones: Viewing geometry, sky reflection removal, uncertainty analysis, and a survey of the Gulf Stream front. *Limnol. Oceanogr. Methods* 20(10):656–673. doi:10.1002/lom3.10511 [V][A]
- Kieu, H.T., Pak, H.Y., Trinh, H.L., Pang, D.S.C., Khoo, E., Law, A.W.-K. (2023). UAV-based remote sensing of turbidity in coastal environment for regulatory monitoring and assessment. *Mar. Pollut. Bull.* 196:115482. doi:10.1016/j.marpolbul.2023.115482 [V][A]
- Kim, E.-J., Nam, S.-H., Koo, J.-W., Hwang, T.-M. (2021). Hybrid approach of unmanned aerial vehicle and unmanned surface vehicle for assessment of chlorophyll-a imagery using spectral indices in stream, South Korea. *Water* 13(14):1930. doi:10.3390/w13141930 [V][A]
- Kwon, Y.S., Pyo, J., Kwon, Y.-H., Duan, H., Cho, K.H., Park, Y. (2020). Drone-based hyperspectral remote sensing of cyanobacteria using vertical cumulative pigment concentration in a deep reservoir. *Remote Sens. Environ.* 236:111517. doi:10.1016/j.rse.2019.111517 [V][A]
- Larson, M.D., Simic Milas, A., Vincent, R.K., Evans, J.E. (2018). Multi-depth suspended sediment estimation using high-resolution remote-sensing UAV in Maumee River, Ohio. *Int. J. Remote Sens.* 39(15–16):5472–5489. doi:10.1080/01431161.2018.1465616 [V][A]
- Leeuw, T., Boss, E. (2018). The HydroColor app: Above water measurements of remote sensing reflectance and turbidity using a smartphone camera. *Sensors* 18(1):256. doi:10.3390/s18010256 [V][A]
- Long, C., Zhang, J., Xia, X., Liu, D., Chen, L., Yan, X. (2025). High-resolution water quality monitoring of small reservoirs using UAV-based multispectral imaging. *Water* 17:1566. doi:10.3390/w17111566 [P]
- McEliece, R., Hinz, S., Guarini, J.-M., Coston-Guarini, J. (2020). Evaluation of nearshore and offshore water quality assessment using UAV multispectral imagery. *Remote Sens.* 12(14):2258. doi:10.3390/rs12142258 [V][A]
- Olivetti, D., Roig, H., Martinez, J.-M., Borges, H., Ferreira, A., Casari, R., Salles, L., Malta, E. (2020). Low-cost unmanned aerial multispectral imagery for siltation monitoring in reservoirs. *Remote Sens.* 12(11):1855. doi:10.3390/rs12111855 [V][A]
- Olivetti, D., Roig, H.L., Martinez, J.-M., Ferreira, A.M.R., Marinho, R., et al. (2025). Inland water quality monitoring using airborne small cameras: Enhancing suspended sediment retrieval and mitigating sun glint effects. *Drones* 9(3):173. doi:10.3390/drones9030173 [V][A]
- Olsson, P.-O., Vivekar, A., Adler, K., Garcia Millan, V.E., Koc, A., Alamrani, M., Eklundh, L. (2021). Radiometric correction of multispectral UAS images: Evaluating the accuracy of the Parrot Sequoia camera and sunshine sensor. *Remote Sens.* 13(4):577. doi:10.3390/rs13040577 [V][A]
- Pagnutti, M.A., Ryan, R.E., Cazenavette, G.J., Gold, M.J., Harlan, R., Leggett, E., Pagnutti, J.F. (2017). Laying the foundation to use Raspberry Pi 3 V2 camera module imagery for scientific and engineering purposes. *J. Electron. Imaging* 26(1):013014. doi:10.1117/1.JEI.26.1.013014 [V][A]
- Pyo, J., Duan, H., Baek, S., Kim, M.S., Jeon, T., Kwon, Y.S., Lee, H., Cho, K.H. (2019). A convolutional neural network regression for quantifying cyanobacteria using hyperspectral imagery. *Remote Sens. Environ.* 233:111350. doi:10.1016/j.rse.2019.111350 [V]
- Román, A., Tovar-Sánchez, A., Gauci, A., Deidun, A., Caballero, I., Colica, E., D'Amico, S., Navarro, G. (2023). Water-quality monitoring with a UAV-mounted multispectral camera in coastal waters. *Remote Sens.* 15(1):237. doi:10.3390/rs15010237 [V][A]
- Román, A., Heredia, S., Windle, A.E., Tovar-Sánchez, A., Navarro, G. (2024). Enhancing georeferencing and mosaicking techniques over water surfaces with high-resolution UAV imagery. *Remote Sens.* 16(2):290. doi:10.3390/rs16020290 [V][A]
- Salim, D.H.C., Mello, C.C.S., Pereira, G., Maretto, R.V., Machado, F.S., Amorim, C.C. (2025). Chlorophyll-a and turbidity estimation using UAV imagery and machine learning in a tropical eutrophic reservoir. *ISPRS Arch.* XLVIII-2/W11-2025:263–268. doi:10.5194/isprs-archives-XLVIII-2-W11-2025-263-2025 [P]
- Shang, S., Lee, Z., Lin, G., Hu, C., et al. (2017). Sensing an intense phytoplankton bloom in the western Taiwan Strait from radiometric measurements on a UAV. *Remote Sens. Environ.* 198:85–94. doi:10.1016/j.rse.2017.05.036 [V]
- Song, Z., Xu, W., Dong, H., Wang, X., Cao, Y., Huang, P., Hou, D., Wu, Z., Wang, Z. (2022). Research on cyanobacterial-bloom detection based on multispectral imaging and deep-learning method. *Sensors* 22(12):4571. doi:10.3390/s22124571 [V][A]
- Su, T.-C., Chou, H.-T. (2015). Application of multispectral sensors carried on unmanned aerial vehicle (UAV) to trophic state mapping of small reservoirs: A case study of Tain-Pu reservoir in Kinmen, Taiwan. *Remote Sens.* 7(8):10078–10097. doi:10.3390/rs70810078 [V][A]
- Windle, A.E., Silsbe, G.M. (2021). Evaluation of unoccupied aircraft system (UAS) remote sensing reflectance retrievals for water quality monitoring in coastal waters. *Front. Environ. Sci.* 9:674247. doi:10.3389/fenvs.2021.674247 [V][A]
- Windle, A.E., Gray, P.C., Román, A., Heredia, S., Navarro, G., Enkhtamir, T., Kwan, K., Khiantani, N., et al. (2026). DroneWQ: A Python package for processing MicaSense multispectral drone imagery for aquatic remote sensing. *J. Open Source Softw.* doi:10.21105/joss.10900; code doi:10.5281/zenodo.14018788 (MIT) [V][P]
- Zeng, C., Richardson, M., King, D.J. (2017). The impacts of environmental variables on water reflectance measured using a lightweight UAV-based spectrometer system. *ISPRS J. Photogramm. Remote Sens.* 130:217–230. doi:10.1016/j.isprsjprs.2017.06.004 [V]
- Zhao, X., Li, Y., Chen, Y., Qiao, X., Qian, W. (2022). Water chlorophyll a estimation using UAV-based multispectral data and machine learning. *Drones* 7(1):2. doi:10.3390/drones7010002 [V][A]

### Radiometry and algorithms
- Cox, C., Munk, W. (1954). Measurement of the roughness of the sea surface from photographs of the sun's glitter. *JOSA* 44(11):838. doi:10.1364/JOSA.44.000838 [V]
- Dogliotti, A.I., Ruddick, K.G., Nechad, B., Doxaran, D., Knaeps, E. (2015). A single algorithm to retrieve turbidity from remotely-sensed data in all coastal and estuarine waters. *Remote Sens. Environ.* 156:157–168. doi:10.1016/j.rse.2014.09.020 [V][A]
- Groetsch, P.M.M., Gege, P., Simis, S.G.H., Eleveld, M.A., Peters, S.W.M. (2017). Validation of a spectral correction procedure for sun and sky reflections in above-water reflectance measurements. *Opt. Express* 25(16):A742. doi:10.1364/OE.25.00A742 [V]
- Hedley, J.D., Harborne, A.R., Mumby, P.J. (2005). Simple and robust removal of sun glint for mapping shallow-water benthos. *Int. J. Remote Sens.* 26(10):2107–2112. doi:10.1080/01431160500034086 [V]
- Hu, C. (2009). A novel ocean color index to detect floating algae in the global oceans. *Remote Sens. Environ.* 113(10):2118–2129. doi:10.1016/j.rse.2009.05.012 [V]
- Kay, S., Hedley, J.D., Lavender, S. (2009). Sun glint correction of high and low spatial resolution images of aquatic scenes: a review of methods for visible and near-infrared wavelengths. *Remote Sens.* 1(4):697–730. doi:10.3390/rs1040697 [V]
- Kelcey, J., Lucieer, A. (2012). Sensor correction of a 6-band multispectral imaging sensor for UAV remote sensing. *Remote Sens.* 4(5):1462–1493. doi:10.3390/rs4051462 [V]
- Kloiber, S.M., Brezonik, P.L., Olmanson, L.G., Bauer, M.E. (2002). A procedure for regional lake water clarity assessment using Landsat multispectral data. *Remote Sens. Environ.* 82(1):38–47. doi:10.1016/S0034-4257(02)00022-6 [V]
- Kutser, T., Vahtmäe, E., Paavel, B., Kauer, T. (2013). Removing glint effects from field radiometry data measured in optically complex coastal and inland waters. *Remote Sens. Environ.* 133:85–89. doi:10.1016/j.rse.2013.02.011 [V]
- Lee, Z., Ahn, Y.-H., Mobley, C., Arnone, R. (2010). Removal of surface-reflected light for the measurement of remote-sensing reflectance from an above-surface platform. *Opt. Express* 18(25):26313. doi:10.1364/OE.18.026313 [V][A]
- Lee, Z., Shang, S., Hu, C., Du, K., Weidemann, A., Hou, W., Lin, J., Lin, G. (2015). Secchi disk depth: A new theory and mechanistic model for underwater visibility. *Remote Sens. Environ.* 169:139–149. doi:10.1016/j.rse.2015.08.002 [V]
- Lee, Z., Shang, S., Qi, L., Yan, J., Lin, G. (2016). A semi-analytical scheme to estimate Secchi-disk depth from Landsat-8 measurements. *Remote Sens. Environ.* 177:101–106. doi:10.1016/j.rse.2016.02.033 [V]
- Mamaghani, B., Salvaggio, C. (2019). Multispectral sensor calibration and characterization for sUAS remote sensing. *Sensors* 19(20):4453. doi:10.3390/s19204453 [V]
- Cao, S., Danielson, B., Clare, S., Koenig, S., Campos-Vargas, C., Sanchez-Azofeifa, A. (2019). Radiometric calibration assessments for UAS-borne multispectral cameras: Laboratory and field protocols. *ISPRS J.* 149:132–145. doi:10.1016/j.isprsjprs.2019.01.016 [V]
- Mishra, S., Mishra, D.R. (2012). Normalized difference chlorophyll index: A novel model for remote estimation of chlorophyll-a concentration in turbid productive waters. *Remote Sens. Environ.* 117:394–406. doi:10.1016/j.rse.2011.10.016 [V]
- Mobley, C.D. (1999). Estimation of the remote-sensing reflectance from above-surface measurements. *Appl. Opt.* 38(36):7442–7455. doi:10.1364/AO.38.007442 [V][A]
- Mobley, C.D. (2015). Polarized reflectance and transmittance properties of windblown sea surfaces. *Appl. Opt.* 54(15):4828–4849. doi:10.1364/AO.54.004828 [V]
- Nechad, B., Ruddick, K.G., Park, Y. (2010). Calibration and validation of a generic multisensor algorithm for mapping of total suspended matter in turbid waters. *Remote Sens. Environ.* 114(4):854–866. doi:10.1016/j.rse.2009.11.022 [V]
- Olmanson, L.G., Bauer, M.E., Brezonik, P.L. (2008). A 20-year Landsat water clarity census of Minnesota's 10,000 lakes. *Remote Sens. Environ.* 112(11):4086–4097. doi:10.1016/j.rse.2007.12.013 [V]
- Ruddick, K.G., De Cauwer, V., Park, Y.-J., Moore, G. (2006). Seaborne measurements of near infrared water-leaving reflectance: The similarity spectrum for turbid waters. *Limnol. Oceanogr.* 51(2):1167–1179. doi:10.4319/lo.2006.51.2.1167 [V]
- Vanhellemont, Q., Ruddick, K. (2018). Atmospheric correction of metre-scale optical satellite data for inland and coastal water applications. *Remote Sens. Environ.* 216:586–597. doi:10.1016/j.rse.2018.07.015 [V]
- Vanhellemont, Q. (2019). Daily metre-scale mapping of water turbidity using CubeSat imagery. *Opt. Express* 27(20):A1372. doi:10.1364/OE.27.0A1372 [V][A]
- Van der Woerd, H.J., Wernand, M.R. (2015). True colour classification of natural waters with medium-spectral resolution satellites: SeaWiFS, MODIS, MERIS and OLCI. *Sensors* 15(10):25663. doi:10.3390/s151025663 [V]
- Zhang, X., He, S., Shabani, A., Zhai, P.-W., Du, K. (2017). Spectral sea surface reflectance of skylight. *Opt. Express* 25(4):A1–A13. doi:10.1364/OE.25.0000A1 [V]

### Satellite water-quality ML and datasets
- Lehmann, M.K., Gurlin, D., Pahlevan, N., Alikas, K., et al. (77 authors) (2023). GLORIA – A globally representative hyperspectral in situ dataset for optical sensing of water quality. *Sci. Data* 10:100. doi:10.1038/s41597-023-01973-y; data doi:10.1594/PANGAEA.948492 (CC BY 4.0) [V][H]
- Mansaray, A.S., Dzialowski, A.R., Martin, M.E., Wagner, K.L., Gholizadeh, H., Stoodley, S.H. (2021). Comparing PlanetScope to Landsat-8 and Sentinel-2 for sensing water quality in reservoirs in agricultural watersheds. *Remote Sens.* 13(9):1847. doi:10.3390/rs13091847 [V][A]
- O'Shea, R.E., Pahlevan, N., Smith, B., et al. (2021). Advancing cyanobacteria biomass estimation from hyperspectral observations: Demonstrations with HICO and PRISMA imagery. *Remote Sens. Environ.* 266:112693. doi:10.1016/j.rse.2021.112693 [V]
- Pahlevan, N., Smith, B., Schalles, J., Binding, C., et al. (2020). Seamless retrievals of chlorophyll-a from Sentinel-2 (MSI) and Sentinel-3 (OLCI) in inland and coastal waters: A machine-learning approach. *Remote Sens. Environ.* 240:111604. doi:10.1016/j.rse.2019.111604 [V]
- Pahlevan, N., Mangin, A., Balasubramanian, S.V., et al. (2021). ACIX-Aqua: A global assessment of atmospheric correction methods for Landsat-8 and Sentinel-2 over lakes, rivers, and coastal waters. *Remote Sens. Environ.* 258:112366. doi:10.1016/j.rse.2021.112366 [V]
- Pahlevan, N., Smith, B., Alikas, K., Anstee, J., Barbosa, C., et al. (2022). Simultaneous retrieval of selected optical water quality indicators from Landsat-8, Sentinel-2, and Sentinel-3. *Remote Sens. Environ.* 270:112860. doi:10.1016/j.rse.2021.112860 [V][A]
- Ross, M.R.V., Topp, S.N., Appling, A.P., Yang, X., Kuhn, C., Butman, D., Simard, M., Pavelsky, T.M. (2019). AquaSat: A data set to enable remote sensing of water quality for inland waters. *Water Resour. Res.* 55(11):10012–10025. doi:10.1029/2019WR024883 [V]
- Smith, B., Pahlevan, N., Schalles, J., Ruberg, S., Errera, R., Ma, R., Giardino, C., et al. (2021). A chlorophyll-a algorithm for Landsat-8 based on mixture density networks. *Front. Remote Sens.* 1:623678. doi:10.3389/frsen.2020.623678 [V][A]
- USGS (2024). AquaMatch chlorophyll a data from Water Quality Portal: ~1970–2024. doi:10.6073/pasta/4c795593504d927ac546a36e5dc9fdb0 (CC0) [P]
- NOAA GLERL & CIGLR (2019–). Physical, chemical, and biological water quality monitoring data to support detection of HABs in western Lake Erie since 2012. NCEI. doi:10.25921/11da-3x54 [V]
- Cheruvelil, K.S., Soranno, P.A., McCullough, I.M., Webster, K.E., Rodriguez, L.K., Smith, N.J., et al. (2021). LAGOS-US LOCUS v1.0. *Limnol. Oceanogr. Lett.* 6:270–292. doi:10.1002/lol2.10203 [V]

### Hazard and segmentation datasets
- Kikaki, K., Kakogeorgiou, I., Mikeli, P., Raitsos, D.E., Karantzalos, K. (2022). MARIDA: A benchmark for marine debris detection from Sentinel-2 remote sensing data. *PLoS ONE* 17(1):e0262247. doi:10.1371/journal.pone.0262247; data doi:10.5281/zenodo.5151941 (CC BY 4.0) [V][P]
- Kikaki, K., Kakogeorgiou, I., Hoteit, I., Karantzalos, K. (2024). Detecting marine pollutants and sea surface features with deep learning in Sentinel-2 imagery. *ISPRS J.* 210:39–54. doi:10.1016/j.isprsjprs.2024.02.017; data doi:10.5281/zenodo.10664073 (CC BY 4.0) [V][P]
- Gkountakos, K., Melitou, M., Ioannidis, K., Demestichas, K., Vrochidis, S., Kompatsiaris, I. (2025). LADOS: Aerial imagery dataset for oil spill detection, classification, and localization using semantic segmentation. *Data* 10(7):117. doi:10.3390/data10070117 [V][A]
- De Kerf, T., Sels, S., Samsonova, S., Vanlanduit, S. (2024). A dataset of drone-captured, segmented images for oil spill detection in port environments. *Sci. Data* 11:1180. doi:10.1038/s41597-024-03993-8 [V][A]
- Fingas, M., Brown, C. (2018; online Dec 2017). A review of oil spill remote sensing. *Sensors* 18(1):91. doi:10.3390/s18010091 [V]
- Hu, C., Lu, Y., Sun, S., Liu, Y. (2021). Optical remote sensing of oil spills in the ocean: What is really possible? *J. Remote Sens.* 2021:9141902. doi:10.34133/2021/9141902 [V][A]
- Lu, Y., Shi, J., Wen, Y., Hu, C., et al. (2019). Optical interpretation of oil emulsions in the ocean – Part I. *Remote Sens. Environ.* 230:111183. doi:10.1016/j.rse.2019.05.002 [V]
- Sun, S., Hu, C. (2016). Sun glint requirement for the remote detection of surface oil films. *Geophys. Res. Lett.* 43(1):309–316. doi:10.1002/2015GL066884 [V][A]
- Jackisch, R., Lorenz, S., Zimmermann, R., Möckel, R., Gloaguen, R. (2018). Drone-borne hyperspectral monitoring of acid mine drainage: An example from the Sokolov lignite district. *Remote Sens.* 10(3):385. doi:10.3390/rs10030385 [V]
- Rahnemoonfar, M., Chowdhury, T., Sarkar, A., Varshney, D., Yari, M., Murphy, R.R. (2021). FloodNet: A high resolution aerial imagery dataset for post flood scene understanding. *IEEE Access* 9:89644–89654. doi:10.1109/ACCESS.2021.3090981 [V][P]
- Robinson, C., Hou, L., Malkin, K., Soobitsky, R., Czawlytko, J., Dilkina, B., Jojic, N. (2019). Large scale high-resolution land cover mapping with multi-resolution data. CVPR 2019. (Chesapeake Land Cover, LILA) [P]
- Xia, J., Yokoya, N., Adriano, B., Broni-Bediako, C. (2023). OpenEarthMap. WACV 2023. arXiv:2210.10732 [V][A]
- Wang, J., Zheng, Z., Ma, A., Lu, X., Zhong, Y. (2021). LoveDA. NeurIPS Datasets & Benchmarks. arXiv:2110.08733 [V][A]
- Wieland, M., Fichtner, F., Martinis, S., Groth, S., Krullikowski, C., Plank, S., Motagh, M. (2024). S1S2-Water: A global dataset for semantic segmentation of water bodies from Sentinel-1 and Sentinel-2 satellite images. *IEEE JSTARS* 17:1084–1099. doi:10.1109/JSTARS.2023.3333969 [V][A]
- Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020). Sen1Floods11. CVPRW 2020, 835–845. doi:10.1109/CVPRW50498.2020.00113 [V]
- Lyu, Y., Vosselman, G., Xia, G.-S., Yilmaz, A., Yang, M.Y. (2020). UAVid: A semantic segmentation dataset for UAV imagery. *ISPRS J.* 165:108–119. doi:10.1016/j.isprsjprs.2020.05.009 [V]
- Mateo-Garcia, G., Veitch-Michaelis, J., Smith, L., et al. (2021). Towards global flood mapping onboard low cost satellites with machine learning. *Sci. Rep.* doi:10.1038/s41598-021-86650-z [V]

### Distillation, spectral reconstruction, foundation models
- Arad, B., Ben-Shahar, O. (2016). Sparse recovery of hyperspectral signal from natural RGB images. ECCV 2016. doi:10.1007/978-3-319-46478-7_2 [V]
- Arad, B., Timofte, R., Ben-Shahar, O., Lin, Y.-T., Finlayson, G.D., et al. (2020). NTIRE 2020 challenge on spectral reconstruction from an RGB image. CVPRW. arXiv:2005.03412 [V][A]
- Cai, Y., Lin, J., Lin, Z., Wang, H., et al. (2022). MST++: Multi-stage spectral-wise transformer for efficient spectral reconstruction. CVPRW 2022. arXiv:2204.07908 [V][A]
- Fernando, T., Fookes, C., Gammulle, H., Denman, S., et al. (2022). Towards on-board panoptic segmentation of multispectral satellite images. arXiv:2204.01952 [V][A]
- Gupta, S., Hoffman, J., Malik, J. (2016). Cross modal distillation for supervision transfer. CVPR 2016. arXiv:1507.00448 [V][A]
- Hinton, G., Vinyals, O., Dean, J. (2015). Distilling the knowledge in a neural network. arXiv:1503.02531 [V]
- Hoffman, J., Gupta, S., Darrell, T. (2016). Learning with side information through modality hallucination. CVPR 2016. doi:10.1109/CVPR.2016.96 [V]
- Hong, D., Gao, L., Yokoya, N., Yao, J., Chanussot, J., Du, Q., Zhang, B. (2021). More diverse means better: Multimodal deep learning meets remote-sensing imagery classification. *IEEE TGRS* 59(5):4340. doi:10.1109/TGRS.2020.3016820 [V]
- Hong, D., Yokoya, N., Xia, G.-S., Chanussot, J., Zhu, X.X. (2020). X-ModalNet. *ISPRS J.* 167:12–23. doi:10.1016/j.isprsjprs.2020.06.014 [V]
- Ienco, D., Fraga Dantas, C. (2024). DisCoM-KD: Cross-modal knowledge distillation via disentanglement representation and adversarial learning. BMVC 2024. arXiv:2408.07080 [V]
- Illarionova, S., Shadrin, D., Trekin, A., Ignatiev, V., Oseledets, I. (2021). Generation of the NIR spectral band for satellite images with convolutional neural networks. *Sensors* 21(16):5646. doi:10.3390/s21165646 [V][A]
- Kendall, A., Gal, Y. (2017). What uncertainties do we need in Bayesian deep learning for computer vision? NeurIPS. arXiv:1703.04977 [V]
- Lanaras, C., Bioucas-Dias, J., Galliani, S., Baltsavias, E., Schindler, K. (2018). Super-resolution of Sentinel-2 images: Learning a globally applicable deep neural network. *ISPRS J.* 146:305–319. doi:10.1016/j.isprsjprs.2018.09.018 [V]
- Li, J., Wu, C., Song, R., Li, Y., et al. (2020). Adaptive weighted attention network with camera spectral sensitivity prior for spectral reconstruction from RGB images. CVPRW. arXiv:2005.09305 [V]
- Liu, Y., Shun, C., Wang, J., Shen, C. (2019). Structured knowledge distillation for dense prediction. CVPR 2019. arXiv:1903.04197 [V]
- Lopez-Paz, D., Bottou, L., Schölkopf, B., Vapnik, V. (2016). Unifying distillation and privileged information. ICLR. arXiv:1511.03643 [V]
- Romero, A., Ballas, N., Kahou, S.E., Chassang, A., Gatta, C., Bengio, Y. (2015). FitNets: Hints for thin deep nets. ICLR. arXiv:1412.6550 [V]
- Saputra, M.R.U., de Gusmão, P.P.B., Almalioglu, Y., Markham, A., et al. (2019). Distilling knowledge from a deep pose regressor network. ICCV. arXiv:1908.00858 [V][A]
- Shi, Z., Chen, C., Xiong, Z., Liu, D., Wu, F. (2018). HSCNN+: Advanced CNN-based hyperspectral recovery from RGB images. CVPRW. doi:10.1109/CVPRW.2018.00139 [V]
- Tian, Y., Krishnan, D., Isola, P. (2020). Contrastive representation distillation. ICLR. arXiv:1910.10699 [V]
- Vapnik, V., Vashist, A. (2009). A new learning paradigm: Learning using privileged information. *Neural Netw.* 22(5–6):544–557. doi:10.1016/j.neunet.2009.06.042 [V]
- Wolf, F., Rolih, B., Čehovin Zajc, L. (2026). Brewing stronger features: Dual-teacher distillation for multispectral Earth observation. CVPR 2026. arXiv:2602.19863 [V][A]
- Xiong, Z., Wang, Y., Zhang, F., Stewart, A.J., et al. (2024). Neural plasticity-inspired multimodal foundation model for Earth observation (DOFA). arXiv:2403.15356 [V][A]
- Wang, Y., Xiong, Z., Liu, C., Stewart, A.J., et al. (2025). Towards a unified Copernicus foundation model for Earth vision. ICCV 2025. arXiv:2503.11849 [V]
- Szwarcman, D., Roy, S., Fraccaro, P., Gíslason, Þ.E., et al. (2024). Prithvi-EO-2.0. arXiv:2412.02732 [V]
- Hong, D., Zhang, B., Li, X., Li, Y., et al. (2024). SpectralGPT. *IEEE TPAMI*. arXiv:2311.07113 [V]
- Tseng, G., Fuller, A., Reil, M., Herzog, H., et al. (2025). Galileo. arXiv:2502.09356 [V]
- Jakubik, J., Yang, F., Blumenstiel, B., Scheurer, E., et al. (2025). TerraMind. ICCV 2025. arXiv:2504.11171 [V]
- Astruc, G., Gonthier, N., Mallet, C., Landrieu, L. (2025). AnySat. arXiv:2412.14123 [V]
- Reed, C.J., Gupta, R., Li, S., Brockman, S., et al. (2023). Scale-MAE. ICCV 2023. arXiv:2212.14532 [V][A]
- Cong, Y., Khanna, S., Meng, C., Liu, P., et al. (2022). SatMAE. NeurIPS. arXiv:2207.08051 [V]
- Fuller, A., Millard, K., Green, J.R. (2023). CROMA. NeurIPS. arXiv:2311.00566 [V]
- Siméoni, O., Vo, H.V., Seitzer, M., Baldassarre, F., et al. (2025). DINOv3. arXiv:2508.10104 [V]

### Edge architectures, quantisation, hardware
- Howard, A., Sandler, M., Chu, G., Chen, L.-C., et al. (2019). Searching for MobileNetV3. ICCV. arXiv:1905.02244 [V]
- Qin, D., Leichner, C., Delakis, M., Fornoni, M., et al. (2024). MobileNetV4: Universal models for the mobile ecosystem. arXiv:2404.10518 [V][A]
- Tan, M., Le, Q.V. (2019). EfficientNet. ICML. arXiv:1905.11946 [V] (EfficientNet-Lite variants via timm/TF; blog not re-opened [U])
- Cai, H., Li, J., Hu, M., Gan, C., Han, S. (2023). EfficientViT: Multi-scale linear attention for high-resolution dense prediction. ICCV. arXiv:2205.14756 [V]
- Liu, X., Peng, H., Zheng, N., Yang, Y., Hu, H., Yuan, Y. (2023). EfficientViT: Memory efficient vision transformer with cascaded group attention. CVPR. arXiv:2305.07027 [V]
- Mehta, S., Rastegari, M. (2022). MobileViT. ICLR. arXiv:2110.02178 [V]
- Wang, A., Chen, H., Lin, Z., Han, J., Ding, G. (2024). RepViT. CVPR. arXiv:2307.09283 [V]
- Vasu, P.K.A., et al. (2023). MobileOne (CVPR, arXiv:2206.04040) and FastViT (ICCV, arXiv:2303.14189) [V]
- Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J.M., Luo, P. (2021). SegFormer. NeurIPS. arXiv:2105.15203 [V]
- Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net. MICCAI. arXiv:1505.04597 [V]
- Jacob, B., Kligys, S., Chen, B., Zhu, M., et al. (2018). Quantization and training of neural networks for efficient integer-arithmetic-only inference. CVPR. arXiv:1712.05877 [V]
- Krishnamoorthi, R. (2018). Quantizing deep convolutional networks for efficient inference: A whitepaper. arXiv:1806.08342 [V]
- Nagel, M., Fournarakis, M., Amjad, R.A., Bondarenko, Y., van Baalen, M., Blankevoort, T. (2021). A white paper on neural network quantization. arXiv:2106.08295 [V]
- Rey, L., Bernardos, A.M., Dobrzycki, A.D., Carramiñana, D., et al. (2025). A performance analysis of YOLO models for deployment on constrained computational edge devices in drone applications. *Electronics* 14(3):638. arXiv:2502.15737 [V][A]
- Vendor pages [P]:
  - Raspberry Pi 5: https://www.raspberrypi.com/products/raspberry-pi-5/
  - AI HAT+: https://www.raspberrypi.com/products/ai-hat/
  - AI Camera: https://www.raspberrypi.com/products/ai-camera/
  - Camera documentation: https://www.raspberrypi.com/documentation/accessories/camera.html
  - NVIDIA Orin Nano Super blog (2024-12-17): https://developer.nvidia.com/blog/nvidia-jetson-orin-nano-developer-kit-gets-a-super-boost/
  - Jetson modules: https://developer.nvidia.com/embedded/jetson-modules
  - Hailo Model Zoo (HAILO8L classification and semantic segmentation tables): https://github.com/hailo-ai/hailo_model_zoo
  - Coral (legacy benchmarks; coral.ai now redirects to https://developers.google.com/coral)
  - TensorRT support matrix: https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html
  - LiteRT Torch: https://github.com/google-ai-edge/ai-edge-torch
  - onnx2tf: https://github.com/PINTO0309/onnx2tf
  - DJI P4 Multispectral specs: https://ag.dji.com/p4-multispectral/specs
  - DJI Mavic 3M specs: https://ag.dji.com/mavic-3-m/specs
  - Planet Education & Research: https://www.planet.com/industries/education-and-research/
  - NEON licence/login change (2026-06-30): https://www.neonscience.org/impact/observatory-blog/logins-required-and-data-licensing-updated
  - USGS NAIP: https://www.usgs.gov/centers/eros/science/usgs-eros-archive-aerial-photography-national-agriculture-imagery-program-naip
  - LILA Chesapeake: https://lila.science/datasets/chesapeakelandcover
  - FloodNet: https://github.com/BinaLab/FloodNet-Supervised_v1.0
  - Tick Tick Bloom rules: https://www.drivendata.org/competitions/143/tick-tick-bloom/

### Reviews (see §a.6)
- Kislik, C., Dronova, I., Kelly, M. (2018). *Drones* 2(4):35. doi:10.3390/drones2040035 [V]
- Wu, D., Li, R., Zhang, F., Liu, J. (2019). *Environ. Monit. Assess.* 191:211. doi:10.1007/s10661-019-7365-8 [V]
- Topp, S.N., Pavelsky, T.M., Jensen, D., Simard, M., Ross, M.R.V. (2020). *Water* 12(1):169. doi:10.3390/w12010169 [V]
- Kutser, T., Hedley, J., Giardino, C., Roelfsema, C., Brando, V.E. (2020). *Remote Sens. Environ.* 240:111619. doi:10.1016/j.rse.2019.111619 [V]
- Sibanda, M., Mutanga, O., Chimonyo, V.G.P., Clulow, A.D., Shoko, C., Mazvimavi, D., Dube, T., Mabhaudhi, T. (2021). *Drones* 5(3):84. doi:10.3390/drones5030084 [V]
- Yang, Z., Yu, X., Dedman, S., et al. (2022). *Sci. Total Environ.* 838:155939. doi:10.1016/j.scitotenv.2022.155939 [V]
- Yang, H., Kong, J., Hu, H., Du, Y., Gao, M., Chen, F. (2022). *Remote Sens.* 14(8):1770. doi:10.3390/rs14081770 [V]
- Gray, P.C., Larsen, G.D., Johnston, D.W. (2022). Drones address an observational blind spot for biological oceanography. *Front. Ecol. Environ.* 20:413–421. doi:10.1002/fee.2472 [V]
- Adjovu, G.E., Stephen, H., James, D., Ahmad, S. (2023). *Remote Sens.* 15(7):1938. doi:10.3390/rs15071938 [V]

---

## Appendix A: Reproducibility of this session's checks

The scripts and outputs live in the session scratchpad, which is temporary. Re-create them in the repo if they are needed later.

- **Scripts:** `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/drone_agent/scripts/`
  - `naip_probe.py`, `naip_sample.py`, `naip_state_items.py`, `s2_probe.py`: Planetary Computer STAC (anonymous SAS signing). Reads a 512×512 NAIP window over Utah Lake (2018-09-09). 8-bit DN means over water: R 133, G 160, B 128, NIR 98. These are DN, not reflectance.
  - `matchup_yield.py`, `matchup_unique.py`: WQP REST queries (statecode US:27, Jun–Oct, `narrowResult` profile) and NAIP-date matching.
  - `gloria_band_pilot.py`, `gloria_band_pilot_800.py`, `gloria_noise_aug.py`: §a.4.
  - `edge_bench.py`, `int8_variants.py`: §d.3.
- **Outputs:** `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/research_samples/drone/`
  - `pilot/*.csv`
  - `edge_bench/*.json*`
  - `wqp/*.zip`
  - `naip/mn_items.json`
  - `neon/*.json`
- **Environment:**
  - Python 3.13.2; torch 2.7.1+cu118; timm 1.0.22; onnx 1.23.2; onnxruntime 1.31.0; scikit-learn 1.7.1; rasterio 1.5.2; pystac-client 0.9.1 (in a scratch venv).
  - CPU: 2× Xeon Silver 4214 (shared machine).
- **API facts observed:**
  - Planetary Computer `naip` extent 2010–2023, with assets `image` (4-band COG), `thumbnail`, `tilejson` and `rendered_preview`.
  - NEON `/api/v0/products/...` is open, but `/api/v0/data/...` returned **403** without a token.
  - WQP legacy `/data/Result/search` works anonymously.
  - Zenodo record APIs confirmed the MARIDA and MADOS licences and sizes.
