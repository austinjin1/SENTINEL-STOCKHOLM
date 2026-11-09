# Novelty and literature check: causal discovery on river WQ sensor data, 2026-10-25

**Scope (docs/research/06 §b, §f.4).** PCMCI/PCMCI+/LPCMCI/J-PCMCI+, CCM and transfer entropy applied to river
water-quality sensor series (DO, turbidity, SpC, nitrate), validated with:
- positive controls (known physics, e.g. T → DOsat, upstream → downstream);
- negative controls (time reversal, disconnected pairs, placebo drivers);
- simulation recovery (twin output with known graph).

This fills the "[U] PCMCI, CCM or Granger applied directly to river WQ sensors" row in docs/research/06 §b.5.

**Method caveat.** Searches used a general web index, not Scholar or WoS. Many applied papers, especially Chinese
journals, may be missing from it.

## Searches (web, 2026-10-25)
1. "PCMCI river water quality dissolved oxygen causal discovery high-frequency sensor" (extended).
   - "Application of Causal Discovery of Factors Driving Dissolved Oxygen in Estuarine Environments"
     (ScienceDirect S1001074225007399; J. Environ. Sci. per the URL prefix [U]; authors not seen [U]).
     - **PCMCI+** on **monthly** data at 8 Long Island Sound stations, 1994–2022.
     - Surface temperature is the strongest driver of bottom DO; discharge has no direct link.
     - Estuary, monthly data, no reported controls [U].
   - Pearl River high-frequency DO periodicity (J. Hydrol., S0022169422002864): spectral analysis, not causal.
   - NEON nitrate–WQ links from sensors (PMC10313027): GAMMs, associational.
2. "convergent cross mapping river water quality nitrate turbidity dissolved oxygen causality".
   - **Shin & Cohen (UF Water Institute 2022 poster; P-Y-Shin.pdf, now 404).** CCM on long-term **metabolism**
     series at 39 US rivers: light → GPP is weak under dense canopy; discharge is not causal in dynamic-flow rivers;
     timescales differ. Peer-reviewed version not found [U].
   - A related frequency-domain paper on 87 rivers (L&O Letters, Oct 2024; doaj 9cc70ab5…) uses spectral analysis,
     not CCM.
   - EGU2020-5760: EDM on stream chemistry with discharge removed as a common driver; DOC → pH.
   - cCCM causalized-CCM caveat (par.nsf.gov 10400328). CCM noise sensitivity (Sci. Rep. 2022).
   - CEcBaN (Water 18:764): CCM + Bayesian network tested on synthetic data and **Lake Kinneret** (lake).
3. "transfer entropy stream water quality sensors nitrate conductance discharge information flow".
   - Lamprey River, NH: TE from wet deposition N → river N (par.nsf.gov, Shattuck et al. [U on authorship]).
     Long-term concentration series, not sensors.
   - Neri et al. 2022: forcing → streamflow TE at more than 200 catchments.
   - Dry Creek TE dataset (Dryad doi:10.6078/D1CH64; met/soil → Q).
   - Wax Lake Delta (Sendrowski & Passalacqua 2017; levels).
   - No TE on river WQ sensors.
4. "PCMCI OR PCMCI+ OR LPCMCI lake OR river OR catchment nitrate OR chlorophyll OR water quality Tigramite"
   (extended). No WQ applications. Hydrology uses:
   - Kumar et al. 2024 WRR (flash droughts, PCMCI+);
   - HESS preprint hess-2021-445 (hydrological connectivity by causal inference);
   - J-PCMCI+ on Greenland supraglacial lakes (arXiv:2510.15265).
5. "causal discovery eutrophication OR algal bloom OR chlorophyll time series PCMCI lake reservoir 2023–2025".
   - NeurIPS 2025 workshop (neurips.cc/virtual/2025/124090): TS-ICD vs PC on **Lake Mendota** daily WQ; flags
     confounding between same-day temperature and chlorophyll.
   - Dianchi Lake EDM 2025.
   - Chao Lake CCM (2026).
   - PNAS 2025 study of 24,452 lakes (algae causally related to climate in 34 %) [U on method].
   - Krich et al. 2020 PCMCI non-stationarity caveat.
6. "Granger causality OR causal inference high-frequency stream sensor data synthetic benchmark validation".
   - **Stein, Shadaydeh, Blunk, Penzel & Denzler 2025, CausalRivers (ICLR 2025; arXiv:2503.17452).**
     - 15-min **discharge** at 1,160 German stations, with ground-truth graphs from river topology.
     - Benchmarks PCMCI (linear CI test), VAR-Granger, VARLiNGAM, DYNOTEARS, CDMI and causal pretraining.
     - Quote: "our naive baselines often achieve scores similar to actual causal discovery approaches."
     - Discharge only, no WQ.
   - Trivariate Granger sensitivity on synthetic data (arXiv:1904.07920).
7. "PCMCI dissolved oxygen river China water quality monitoring stations causal network".
   - EGU26-21949 (Yu, Li & Qu): topology- and travel-time-constrained causal forecasting on more than 1,900 Chinese
     national stations. Algorithm not named; no validation details.
   - Yangtze River Delta VAR/Granger DO drivers (Frontiers 2021/2022; doaj 82081f69…).
   - Complex-network DO/NH3-N/CODMn at 91 Chinese stations, weekly (ESS Open Archive essoar.10512718.1).
8. "transfer entropy dissolved oxygen river OR stream sensor drivers metabolism". No TE-DO hit. Regier et al. 2026
   (SFS abstract 135678) is a KGML classifier, not causal discovery.
9. Seen: Langendorf, Lyubchich, Testa & Zhang 2021, ACS ES&T Water 1(8):1665–1675. Causal networks
   (empirical-dynamics-type [U]) for Chesapeake Bay DO-criterion attainment; estuary, long-term monitoring data.

## Conclusion (provisional)
- Causal-discovery applications to aquatic WQ exist but are **almost all**:
  - estuarine or lake systems (Long Island Sound PCMCI+, Chesapeake, Mendota, Kinneret, Dianchi, Chao);
  - at monthly, weekly or daily resolution;
  - or on derived metabolism series (Shin & Cohen CCM).
- We found **no** study applying PCMCI-family, CCM or TE to **high-frequency river WQ sensor** series (DO, SpC,
  turbidity, nitrate) **with explicit positive/negative controls or simulation recovery**.
- The only river benchmark with ground truth is CausalRivers (discharge). Its finding that naive cross-correlation
  baselines match causal methods is directly relevant: **we must report a cross-correlation baseline next to
  PCMCI+**.
- Classification: **partially novel (methodological extension)**. The application domain plus validation ladder is
  new. The methods and the general finding "temperature drives DO" are not.
- Permitted wording: "To our knowledge, causal discovery has not been applied to continuous river WQ sensor
  networks with physics-based positive controls, time-reversal negative controls and simulation recovery."
- Avoid claiming new causal *mechanisms*. Restrict claims to rungs of the docs/research/06 §b.3 ladder that are
  actually passed.
- To do: Scholar search "PCMCI" AND ("dissolved oxygen" OR turbidity OR nitrate) AND (river OR stream), focusing
  on Chinese-language and Water Research/STOTEN 2023–2026. Locate the peer-reviewed Shin & Cohen CCM paper and
  the full text of the estuarine PCMCI+ paper.
