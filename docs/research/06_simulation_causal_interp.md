# 06: Simulation and Digital Twins, Causal Discovery, Ecological Interactions, Interpretability

*SENTINEL rebuild, research note 06. Prepared 2026-10-22. Scope: rivers first (USGS and NEON), lakes second. This note serves Goal 3 (a future-simulation platform plus interpretation that finds causal chains, ecological interactions, and defensible new insights).*

---

## How to read this note (verification status)

Every reference in §(h) carries a flag:

- **[V]**: metadata confirmed (title, authors, venue, year, DOI or arXiv ID) through the Crossref, arXiv or DataCite APIs, **and** every claim quoted here was checked against the abstract or the publisher's page.
- **[M]**: metadata confirmed only. The note cites the paper for what its title and venue establish, and for nothing more specific.
- **[U]**: unverified. Treat it as a lead, not as a citation.

**Method caveat.** This session shared a web-search quota with other agents, and the quota ran out partway through the work. From that point on, verification used these sources:

- DOI lookups through the Crossref REST API (about 190 records), arXiv API lookups (about 45), and OpenAlex and Semantic Scholar for abstracts. DataCite covered USGS data releases.
- Direct fetches of publisher, agency and documentation pages.
- PyPI, CRAN and GitHub pages for software status.
- Live queries to the USGS Water Data API, the USGS NLDI and the NEON API for data facts. The site counts in §(e) and Appendix A come from these queries.

Discovery of the literature therefore leaned partly on prior knowledge. That means **the novelty ratings in §(e) are provisional.** Each candidate needs one dedicated literature search before anyone writes "first" or "novel". Appendix B gives the queries.

---

## Executive summary: the decisions that matter

1. **Replace the latent neural ODE twin with a multi-site "LSTM plus process head" hybrid.** Simulate only variables that are actually observed: water temperature, DO (daily mean, minimum and maximum, or hourly), specific conductance, turbidity, and nitrate where a sensor exists. The DO head should be a differentiable one-station metabolism model.
   - Solubility is hard-coded: Benson–Krause O₂ saturation.
   - Gas exchange is hard-coded: K₆₀₀ with Schmidt-number scaling.
   - Neural networks supply only daily GPP, ER and K₆₀₀.
   - This matches the published state of the art (process-guided and differentiable models; Read 2019, Jia 2021, Feng 2022, Rahmani 2023, Shen 2023). It also turns the twin's internal variables into checkable ecological quantities.
2. **Every simulator number is reported as skill against baselines:** persistence, day-of-year climatology, and a physics-only baseline. The old twin lost to persistence (R² 0.07 vs. 0.92). Any new result that fails this gate is not reported as a twin result.
3. **Counterfactuals are allowed only for physical forcings, and only after metamorphic tests pass.** Physical forcings are air temperature, precipitation or discharge, and shortwave radiation. Published LSTMs produced physically wrong warming responses (Wi & Steinschneider 2022, 2024; Reichert 2024), so "+2 °C" outputs need sign and magnitude tests plus a differential split-sample test (Klemeš 1986). Nutrient-reduction scenarios are out of scope unless nutrients are measured inputs. Only about 150 DO+T+Q sites have co-located nitrate sensors (Appendix A).
4. **Causal discovery must pass a validation ladder before any "novel chain" is claimed.**
   - The ladder runs: synthetic data from known physics, then positive and negative controls on real data, then cross-site replication, then natural experiments.
   - Use PCMCI+, LPCMCI or J-PCMCI+ in tigramite 5.2, with bootstrap aggregation (Bagged-PCMCI+).
   - Include radiation and discharge as explicit variables.
   - The earlier run used one month of 30-min lake data with no light variable. It did not recover temperature→DO even though the raw r was −0.82. Its own output file attributes this to "a shared diel solar cycle" (§b.6). That is a design flaw to fix, not proof that the method fails.
5. **Do not treat microbial co-occurrence networks or SHAP plots as evidence of interactions.** Correlation-network methods often do no better than Pearson correlation (Hirano & Takemoto 2019; Weiss 2016). Permutation and marginal-SHAP explanations extrapolate when inputs are correlated (Hooker 2021; Aas 2021).
6. **Interpretability starts with fidelity tests:** synthetic ground-truth attributions, model-randomization checks, and probe controls. Use attribution with climatology baselines, probes of LSTM states against independent metabolism estimates, symbolic regression on the learned GPP, ER and K₆₀₀ functions, and Sobol sensitivity on the scenario forcings.
7. **Best novel-insight bets (§e):**
   - (E1) A continental, **diel-resolved** partition of DO warming sensitivity into solubility and metabolism parts, using continuous sensors. Zhi et al. 2023 used daytime grab samples and flag nocturnal declines as likely underestimated.
   - (E2) How far, and with what lead time, WQ signals propagate between upstream and downstream USGS sensors. About half of the sampled continuous-temperature sites have a downstream partner within 100 km.
   - (E3) Groundwater (thermal-buffering) signatures as a control on DO resilience. This combines Hare et al. 2021 with E1.
8. **Data plumbing is time-critical.**
   - USGS WaterServices, the legacy NWIS web services, are scheduled for **decommission in early 2027**. Build loaders on the Water Data API v1 at `api.waterdata.usgs.gov/ogcapi/v1`.
   - Water Quality Portal legacy (WQX 2.2) profiles **lack USGS data added after 2024-03-11**. Use the WQX 3.0 "beta" profiles instead.
   - The NLDI has moved to `api.water.usgs.gov/nldi`.
9. **Get one independent, prospective validation:** the NEON Ecological Forecasting Challenge aquatics theme, which was still active on 2026-10-21. It covers daily DO and water temperature at NEON sites, scored with CRPS against climatology and random-walk persistence baselines. Judges cannot argue with a skill score earned on data that did not exist when the forecast was made.

---

## 0. Lessons from the October 2026 audit, turned into design rules

| Audit finding (REVIEW_REPORT.md) | Root cause | Design rule for the rebuild |
|---|---|---|
| Twin trained on a 300-sample synthetic fallback | Silent fallback when the loader failed | Loaders raise on failure; there is no synthetic fallback in any training path; a minimum-N assertion runs; a provenance hash of the training data goes in every result JSON |
| Twin lost to persistence (R² 0.07 vs. 0.92) | No baseline gate | Report skill relative to persistence, climatology and physics-only baselines; a result below baseline fails |
| 6/10 twin states never observed; the "turbidity" slot held SpC | No schema or unit checks | State vector = observed variables only; assert USGS parameter codes and units (00010, 00300, 00095, 63680, 99133, 00060) |
| USGS −999999 codes left unmasked (DO MAE 65.8 mg/L) | Sentinel values entered the statistics | Mask sentinel values and qualifier-flagged values (ice, equipment, estimated) before any statistic; enforce physical-range checks |
| GNN label was a threshold of its own input | Label leakage | Labels come from independent sources; keep the existing test that bans input-derived labels |
| River graph had 247 disconnected pieces | Graph not built from network topology | Build upstream/downstream relations from NLDI or NHDPlus navigation; report connectivity |
| PCMCI+ recovered 0/3 "known chains" | Chains defined at the wrong time scale; the diel light driver was omitted; one-month samples | Specify positive controls with time scale **and** conditioning set; include radiation and discharge; use multi-year daily data plus diel-specific analyses; first show recovery on simulated data |
| "Reduce P by 30 %" scenario acted on a variable that was zero everywhere | Scenario variables never observed | Scenario variables must be observed inputs with a validated response |
| Conformal scorer fit on its own calibration set | Leakage | Assert split discipline in code |

---

## (a) Simulation and digital-twin state of the art

### a.1 What a defensible water-quality twin is

In the hydrology and limnology literature, working "twins" for water quality are almost always one of two things.

- **Knowledge-guided ML (KGML).** These are neural networks constrained by physics: loss penalties, pretraining on process-model output, or physically structured architectures (Read 2019; Jia 2021; Willard 2023).
- **Differentiable process models.** Here the neural networks parameterize or replace modules inside a process model written in an autodiff framework (Tsai 2021; Feng 2022; Shen 2023).

The reviews agree on the promise and the caveats. Varadharajan 2022 and Zhi 2024 cover water quality specifically. Reichstein 2019 covers Earth-system science generally. Zhi 2024 frames the goal as moving from "black boxes" to "glass boxes" so that skill gains also yield process knowledge.

### a.2 Process-guided deep learning (lakes and stream temperature)

| Work | System | How physics enters | Verified result | Open resources |
|---|---|---|---|---|
| Read et al. 2019, WRR [V] | Depth-resolved lake temperature | LSTM, energy-conservation penalty, pretraining on GLM simulations | PGDL beat DL and PB at two study lakes "but only when pretraining data included greater variability than the training period". Across 68 lakes the median RMSE was 1.65 °C (DL 1.78, PB 2.03) | Data and code release, doi:10.5066/P9AQPIVD [V] |
| Jia et al. 2019 SDM; Jia et al. 2021 ACM/IMS TDS [V] | Lake temperature (PGRNN) | RNN with energy-conservation constraints and process-model knowledge | Over 20 % better than the physics-based model "even with very little training data", with outputs consistent with physical laws | — |
| Daw et al. 2020 SDM [M] | Lake temperature | Physics-guided architecture for uncertainty | — | — |
| Willard et al. 2021, WRR [V] | Unmonitored lakes | Meta-transfer learning from 145 source lakes (PB or PGDL) to 305 targets | Median RMSE 2.52 °C (uncalibrated GLM), 2.43 (PB-MTL), 2.16 (PGDL-MTL), 1.88 (9-source ensemble) | — |
| Willard et al. 2022, L&O Letters [V] | 185,549 CONUS lakes | LSTM optimized for unmonitored lakes | Held-out lakes: median lake-specific error 1.24 °C, overall RMSE 1.61 °C | Dataset described in the paper |
| Ladwig et al. 2024, JAMES [V] | 1-D lake hydrodynamics | **Modular compositional learning**: DL replaces only the vertical-diffusion module of a process model | Best for water temperature, Schmidt stability, buoyancy frequency and isotherm depths | — |
| Hanson et al. 2020, Ecol. Modelling [M] | Lake surface phosphorus | Process-guided ML | (title-level) | — |
| Jia et al. 2021 SDM [M]; Topp et al. 2023, WRR [V] | Delaware River Basin stream temperature | Recurrent graph models pretrained on process-model output | Topp: RGCN and Graph WaveNet tie in-distribution (RMSE 1.64 and 1.65 °C), but Graph WaveNet wins in 4 of 5 shifted-condition tests. XAI showed its spatial dependencies have "greater fidelity to physical processes" | github.com/USGS-R/river-dl, github.com/USGS-R/delaware-model-prep (both exist) |
| Zwart et al. 2023, JAWRA [V] | 7-day forecasts of daily maximum stream temperature | PGDL plus data assimilation | RMSE 1.1–1.4 °C (1 day) and 1.4–1.9 °C (7 days). DA cut RMSE by 0–14 %. **Only 65–82 % of observations fell inside the 90 % intervals**, so the intervals were too narrow | — |
| Chen et al. 2021 ICDM [M] | Stream–reservoir networks | Heterogeneous graph network plus DA | (title-level) | — |

**Lake DO.** Process-plus-ML DO work exists:

- Ladwig et al. 2021 (HESS) [V] coupled GLM-AED2 with random forests to explain 37 years of Lake Mendota anoxia. Physics dominated: stratification timing and stability.
- Ladwig et al. 2022 (L&O) [V] fit Bayesian metabolism to eight lakes.

I could not verify a dedicated PGDL-for-lake-DO paper in this session [U]. Treat lake DO as secondary for SENTINEL.

### a.3 Differentiable process models (δ models) and hybrid stream temperature

- **Tsai et al. 2021, Nat. Commun. [V]: differentiable parameter learning.** As training data grow, dPL gets "better performance, more physical coherence, and better generalizability". It needed about 12.5 % of the training data to match competitor methods.
- **Feng et al. 2022, WRR [V].** δHBV reached a median NSE of 0.732 against 0.748 for an LSTM on 671 basins (Daymet forcing). Unlike the LSTM, it outputs untrained variables: storages, ET, baseflow.
- **Feng et al. 2023, HESS [V].** In regional hold-outs (PUR), δ models beat LSTMs on daily metrics, and their "advantages in mean and high flow trends became prominent". Parameter fields stayed spatially stable.
- **Shen et al. 2023, Nat. Rev. Earth Environ. [V]: perspective.** Differentiable models give better "interpretability, generalizability, and extrapolation" at similar accuracy with less data. Future work must "verify the physical significance of outputs".
- **Rahmani et al. 2021a (ERL) [M] and 2021b (Hydrol. Process.) [V].** A CONUS stream-temperature LSTM on more than 400 basins achieved RMSE 1.129 °C and R² 0.983 in unmonitored basins, and a median of 1.202 °C even in unmonitored dammed basins.
- **Rahmani et al. 2023, WRR [V].** In a hybrid differentiable stream-temperature model, competing "structural priors" were distinguishable from data. However, internal variables (water-source fractions) "may not be adequately constrained by a single target variable". **This is directly relevant:** SENTINEL's GPP, ER and K₆₀₀ latents need independent checks (§a.8).
- **Bennett & Nijssen 2021, WRR [V].** Embedding DL parameterizations inside the SUMMA process model beat the calibrated standalone model.
- **Training scale.** "Never train an LSTM on a single basin" (Kratzert et al. 2024, HESS [V]). Multi-basin LSTMs beat per-basin-calibrated models, both out of sample (Kratzert et al. 2019, WRR [V]: median NSE 0.69 in ungauged mode against 0.64 for calibrated SAC-SMA) and with EA-LSTM catchment embeddings (Kratzert 2019, HESS [V]). Mass conservation: MC-LSTM (Hoedt et al. 2021, ICML) [V].

**Software that exists:**

- **NeuralHydrology** 1.13.0 (BSD-3). Its model zoo includes CudaLSTM, EA-LSTM, MC-LSTM, MTS-LSTM, ODE-LSTM, Transformer, xLSTM, Mamba and a Hybrid-Model. Output heads: Regression, GMM, CMAL, UMAL.
- **δMG** (`generic_deltaModel`, PyPI `dmg`), which describes itself as the successor to HydroDL.
- **δHBV 2.0.** Note that δMG and δHBV 2.0 both carry a **Non-Commercial (PSU)** license. That is fine for a student project, but record it.

### a.4 Continental deep learning for river water quality

**Zhi et al. 2021, ES&T [V].** An LSTM trained on sparse DO from 236 minimally disturbed CAMELS-chem basins "learns the theory of DO solubility". It also "misses some DO peaks and troughs when in-stream biogeochemical processes become important". It did *not* perform better where more data existed; it did better in basins with low flow and DO variability, a high runoff ratio (>0.45) and winter precipitation peaks.

**Zhi et al. 2023, Nature Climate Change: "Widespread deoxygenation in warming rivers" [V, publisher page].** This is the closest prior work to SENTINEL's simulator and insight goals.

- Coverage: 580 US and 216 Central European rivers.
- An LSTM reconstructed daily water temperature and DO for 1981–2019, using weather, water-quality data and static attributes; it beat Gaussian-process regression.
- Projections for 2020–2100 used NEX-GDDP-CMIP6 (10 GCMs; SSP2-4.5 and SSP5-8.5). Projected temperature and precipitation replaced historical inputs, long-term climate attributes were updated, and other inputs were held at historical values.
- Findings: 87 % of rivers warmed and 70 % lost DO; the mean rate was about −0.038 ± 0.026 mg L⁻¹ per decade, faster than in oceans and slower than in temperate lakes. Urban rivers warmed fastest; agricultural rivers lost oxygen fastest. Future rates were 1.6–2.5 times historical.
- **Key caveat for SENTINEL's opportunity:** training data were *daytime grab samples*, which miss nocturnal DO drops, so true rates may be higher.
- Code: github.com/LiReactiveWater/WT-DO-US-CE-LSTM. Data: github.com/LiReactiveWater/WT-DO-US-CE-dataset. Both exist.

**Other relevant work:**

- **CAMELS-Chem** (Sterle et al. 2024, HESS [V]): 516 catchments, 18 constituents including DO, 1980–2018, mostly grab samples.
- **Nitrate:** Saha et al. 2023 (STOTEN) [M] built a DL model that generates daily stream nitrate for data-sparse watersheds.
- **Reviews:** Li et al. 2024 (NCC) [M] on land–river connectivity controls on WQ. Zhi et al. 2024 (Nature Water) [V, partial page] on DL for WQ: gap filling, driver identification, and the "black box to glass box" agenda.

**Implication.** "An LSTM predicting DO across the US" is no longer novel. SENTINEL's distinct angle should be:

- continuous-sensor (diel) dynamics;
- a process head that separates solubility, metabolism and gas exchange;
- counterfactuals that pass explicit physical tests.

### a.5 Neural ODEs, universal differential equations, hybrid GCMs and world models

- **Neural ODEs and UDEs.** See Chen et al. 2018 (NeurIPS) [V-arXiv] and Rackauckas et al. 2020 (UDEs) [V-arXiv]. Höge et al. 2022 (HESS) [V] built hydrologic neural ODEs on 569 CONUS catchments. They perform "as well as state-of-the-art deep learning methods in stream flow prediction while maintaining the ease of interpretability of conceptual hydrologic models". This is because the ODE *structure* (the buckets) is fixed and neural networks replace only the process rates.
- **NeuralGCM** (Kochkov et al. 2024, Nature [V]). It combines a differentiable dynamical core with learned physics. It is competitive for weather and stable for decades with prescribed SST, but the authors state it "does not extrapolate to substantially different future climates". Even the best hybrid in Earth science admits a limited validity envelope. SENTINEL's platform must publish its own envelope.
- **Lesson for SENTINEL.** The old twin was a latent ODE over 10 states, 6 of them never observed, so it was **not identifiable**. If an ODE is used, make it a UDE in which the ODE is the *known* DO mass balance (§a.6). The state is then observed DO, and the neural network supplies only the interpretable rates (GPP, ER, K₆₀₀).

### a.6 Process components to hard-code (with sanity numbers)

**1. O₂ solubility.** Use Benson & Krause (1984) [V] for freshwater, with refits in Garcia & Gordon (1992) [V]. The USGS DOTABLES page (water.usgs.gov/water-resources/software/DOTABLES/) exists and gives the standard USGS implementation, including pressure and salinity corrections.

I computed the following values from the Benson–Krause freshwater equation at 1 atm, in the form used in APHA Standard Methods. Cross-check the implementation against DOTABLES in a unit test.

| T (°C) | 0 | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|
| DO_sat (mg/L) | 14.62 | 12.77 | 11.29 | 10.08 | 9.09 | 8.26 | 7.56 |

- The slope is −0.27 mg L⁻¹ °C⁻¹ at 10 °C (−2.4 %/°C), −0.18 at 20 °C (−2.0 %/°C) and −0.13 at 30 °C (−1.7 %/°C).
- **A +2 °C warming lowers saturation by 0.51 mg/L at 10 °C and 0.35 mg/L at 20 °C.** This is the "solubility-only" expectation for any +2 °C counterfactual.
- Pressure scales DO_sat roughly in proportion. At 1,500 m elevation, about 0.835 atm gives about 16 % lower DO_sat, so elevation must be an input.

**2. One-station metabolism (open-channel) model.** See Odum 1956 [V-meta], Van de Bogert et al. 2007 [M], Holtgrieve et al. 2010 [M], Hall & Hotchkiss 2017 [M], and Appling et al. 2018 (JGR-B) [V]:

dDO/dt = GPP·(PAR_t / ΣPAR_day)/z + ER/z + K_O2(T)·(DO_sat(T, P) − DO)

where z is mean depth, ER < 0, and K_O2 = K₆₀₀·(Sc_O2(T)/600)^(−0.5) (Wanninkhof 1992 [M] Schmidt-number relation).

- **Equifinality.** GPP, ER and K₆₀₀ trade off against each other when fitted from DO alone. Appling 2018's hierarchical model, which links K₆₀₀ to discharge, "reduces error by as much as 70 %" in daily K₆₀₀, GPP and ER. It is implemented in **streamMetabolizer** (R; github.com/DOI-USGS/streamMetabolizer).
- **K₆₀₀ priors.** Raymond et al. 2012 [V] showed slope and velocity predict *k* reasonably well, and gave hydraulic-geometry equations. See also Ulseth et al. 2019 [M] (distinct low- and high-energy regimes) and Hall & Ulseth 2020 [M] (review).
- **Schmidt-number effect, computed:** for a fixed K₆₀₀, K_O2 rises about 5 % for +2 °C at 20 °C. Warming therefore speeds re-aeration slightly, which partly offsets metabolic deficits.

**3. Temperature dependence of ecosystem respiration.** It is about **0.65 eV** for aquatic systems (Yvon-Durocher et al. 2012, Nature [V]). I computed that this means +19 % ER per +2 °C at 20 °C (Q₁₀ ≈ 2.3). Song et al. 2018 (Nat. Geosci.) [V] estimated that 1 °C of stream warming causes a 23.6 % decline in net ecosystem productivity across their streams.

**4. Other process models.** Streeter–Phelps BOD–DO sag (the 1925 US Public Health Service bulletin; exact citation details [U]) handles point-source scenarios. QUAL2K, WASP and SWAT are full process WQ models. My judgment: they are too heavy to calibrate at hundreds of sites for this team. Use them only as conceptual references.

### a.7 Counterfactual experiments with ML: what the literature found

- **Wi & Steinschneider 2022, WRR [V].** Under warming, a regional LSTM (15 California basins) reproduced the hydrograph shifts but "in some watersheds predicts an unrealistic increase in the runoff ratio".
  - A multi-output LSTM did not fix this.
  - Feeding process-model ET as an input helped only for one process model, so it depends on that model's fidelity.
  - An LSTM trained on more than 500 US basins gave more realistic warming responses.
- **Wi & Steinschneider 2024, HESS [V].** On 212 Great Lakes basins, process models lost −17 to −25 % of mean flow under temperature-based PET and −6 to −9 % under energy-based PET. The plain LSTM showed "unrealistically large water losses" (−20 %) even with energy-based PET. MC-LSTM was insensitive to PET. Only the physics-respecting MC-LSTM-PET behaved like the process models.
- **Reichert et al. 2024, HESS [V]: metamorphic testing.** ML models fit best, but "the magnitude, and even the sign of the response can depend on the training data". Testing responses to modified inputs is needed "beyond and in addition to the usual calibration–validation analysis".
- **Frame et al. 2022, HESS [V].** LSTMs stayed relatively accurate for extreme events left out of training. Adding mass-balance constraints *reduced* skill on extremes. Physics constraints are not automatically better, so test them.
- **Zhi et al. 2023 [V].** Forcing replacement with CMIP6, with all other inputs held at historical values (§a.4). This is a clean template, but the counterfactual itself was not metamorphically tested in the material I could read.

**Bottom line.** "+2 °C" from a pure LSTM is a hypothesis, not a result. It becomes a result only after it passes:

1. sign and magnitude tests against physics (§a.6);
2. a differential split-sample test (train cool or wet years, test hot or dry years; Klemeš 1986 [V-meta]);
3. multi-site training.

### a.8 How simulators are validated: a synthesis to adopt wholesale

1. **Benchmarks before metrics.**
   - NSE = 0 is the mean-flow benchmark, but for KGE the mean-flow benchmark is **1 − √2 ≈ −0.41** (Knoben et al. 2019 [V]). Report decomposed KGE (Gupta et al. 2009 [M]) and skill relative to explicit baselines (Schaefli & Gupta 2007 [M]; Pappenberger et al. 2015 [M]).
   - SENTINEL's baselines: persistence, DOY climatology, and "physics-only". For DO, physics-only means DO_sat(T_obs) × the site's median %saturation for that month. For temperature, it means an air–water regression.
2. **Splits.**
   - Temporal hold-out.
   - Site-disjoint k-fold (PUB; Kratzert 2019 [V]; Hrachowitz 2013 [M]).
   - Region hold-out (PUR; Feng 2023 [V]).
   - A **differential split-sample test** for counterfactual claims.
3. **Metamorphic and physical-consistency tests** (Reichert 2024; Wi 2022, 2024). Pass/fail criteria are in §f.3.
4. **Diagnostic signatures, "right answers for the right reasons"** (Kirchner 2006 [M]; Gupta et al. 2008 [M]). Check diel amplitude, C–Q slopes, hysteresis direction and thermal sensitivity, not just RMSE.
5. **Probabilistic calibration:** CRPS, PIT histograms, interval coverage. Zwart 2023 shows that 90 % intervals can cover only 65–82 %.
6. **Prospective, near-term forecasting** (Dietze et al. 2018 [V]; Lewis et al. 2022 [V], who found only 45 % of 178 forecasting papers reported uncertainty). The **NEON Forecasting Challenge** (Thomas et al. 2023 [M]; challenge pages [V]) is live, scores with CRPS and supplies climatology and persistenceRW baselines (§e, §f.3).
7. **Process-based evaluation with causal fingerprints.** Compare causal graphs estimated from observations with graphs estimated from twin simulations (Nowack et al. 2020, Nat. Commun. [V]: models whose fingerprints were closer to observations better reproduced precipitation patterns).
8. **Internal-variable checks.**
   - Compare learned GPP, ER and K₆₀₀ with the USGS Powell Center estimates for 356 rivers (Appling et al. 2018, Sci. Data [V]; ScienceBase release doi:10.5066/F70864KX [V]).
   - Compare with NEON metabolism for 27 streams and rivers (Marzolf et al. 2025, Sci. Data [V]).
   - These comparators are model estimates themselves, so use them as validation, not ground truth.

### a.9 Open-source inventory (simulation)

| Tool or dataset | URL | Language | What it gives | License or status |
|---|---|---|---|---|
| NeuralHydrology 1.13.0 | github.com/neuralhydrology/neuralhydrology | Python/PyTorch | Multi-basin LSTM, MC-LSTM, MTS-LSTM, Transformer, hybrid models; CMAL/GMM/UMAL uncertainty heads | BSD-3 (PyPI) |
| δMG (generic_deltaModel; PyPI `dmg`) | github.com/mhpi/generic_deltaModel | Python/PyTorch | Differentiable process models plus neural parameterization; successor to HydroDL | Non-Commercial (PSU) |
| δHBV 2.0 | github.com/mhpi/dHBV2.0 | Python | CONUS differentiable HBV, BMI-ready | Non-Commercial (PSU) |
| HydroDL | github.com/mhpi/hydroDL | Python | Older MHPI code (Zhi 2021 lineage) | see repo |
| river-dl; delaware-model-prep | github.com/USGS-R/river-dl; github.com/USGS-R/delaware-model-prep | Python | USGS process-guided graph models for stream temperature | see repo |
| PGDL lake data release | doi:10.5066/P9AQPIVD | data + code | Read 2019 inputs and outputs | USGS public domain |
| Zhi 2023 code and data | github.com/LiReactiveWater/WT-DO-US-CE-LSTM; …-dataset | Python | Continental DO and temperature LSTM, CMIP6 projections | see repo |
| streamMetabolizer | github.com/DOI-USGS/streamMetabolizer | R (Stan) | Hierarchical GPP/ER/K₆₀₀ estimation | GitHub only, not on CRAN |
| Powell Center metabolism, 356 rivers | doi:10.5066/F70864KX | data | Daily GPP/ER/K₆₀₀, 2007–2017 (490,907 site-days) | USGS |
| NEON metabolism, 27 sites | doi:10.1038/s41597-025-04710-9 | data | GPP/ER/K for NEON streams | open |
| CAMELS-Chem | doi:10.5194/hess-28-611-2024 | data | 516 basins, grab-sample chemistry | open |
| dataretrieval 1.4.0 / dataRetrieval 2.7.27 | github.com/DOI-USGS/dataretrieval-python; CRAN | Python / R | USGS Water Data API v1, WQP, NLDI | public domain |
| HyRiver (pynhd 0.20, pygridmet, pydaymet) | docs.hyriver.io; JOSS doi:10.21105/joss.03175 | Python | NLDI, NHDPlus, gridMET, Daymet access | MIT |
| neonutilities 2.0.2 / neonUtilities 4.0.2 | github.com/NEONScience/NEON-utilities-python; CRAN | Python / R | NEON downloads | AGPL / see CRAN |
| StreamCatTools 0.12.0 | github.com/USEPA/StreamCatTools; API docs usepa.github.io/StreamCatWebServices_Public/ | R / web API | More than 600 catchment and watershed metrics for about 2.65 M NHDPlusV2 segments | public |

---

## (b) Causal discovery on environmental time series

### b.1 Method families

The assumption and output columns for the tigramite methods follow the tigramite 5.2 README (verified 2026-10-22; PyPI tigramite 5.2.10.1, GPL-3.0).

| Method | Assumptions (beyond causal Markov and faithfulness) | Output | When to use in SENTINEL |
|---|---|---|---|
| PCMCI (Runge et al. 2019, Sci. Adv. [V]) | Causal stationarity; no contemporaneous links; no hidden variables | Directed lagged links | Sub-daily data where all effects are lagged |
| **PCMCI+** (Runge 2020, UAI [V-arXiv]) | Causal stationarity; no hidden variables | Lagged **and** contemporaneous links (time-series CPDAG) | **Default** for daily data, where many effects fall within one day |
| LPCMCI (Gerhardus & Runge 2020, NeurIPS [V-arXiv]) | Causal stationarity; latent confounders allowed | Time-series PAG | Robustness check when key drivers are unmeasured (e.g., nutrients) |
| RPCMCI (Saggioro et al. 2020, Chaos [V]) | Discrete persistent regimes | A graph per regime | Season, flow-regime or ice regimes |
| **J-PCMCI+** (Günther, Ninad & Runge 2023, UAI [V-arXiv]) | Multiple datasets; context variables (observed or latent) | Joint graph including context nodes | **Multi-site pooling.** The paper's own example is "river runoff from different catchments" sharing weather but differing in attributes such as altitude |
| Bagged-PCMCI+ (Debeire et al. 2024, CLeaR, PMLR 236:979–1007 [V-arXiv]) | As PCMCI+ | Link confidence from a time-series bootstrap | Always. It "improves in precision and recall" over PCMCI+, especially for short samples and high autocorrelation |
| Granger / VAR (Granger 1969 [M]) | Linear; lagged; all common drivers observed | Predictive causality | Baseline only |
| Transfer entropy (Schreiber 2000 [M]); process networks (Ruddell & Kumar 2009 [V]; Goodwell & Kumar 2017 [V]) | Nonparametric; pairwise unless conditioned | Information-flow networks | Timescale analyses; propagation lags (E2) |
| CCM / EDM (Sugihara et al. 2012 [V]; Ye et al. 2015 [V]; Chang et al. 2017 [V-meta]) | Deterministic attractor, weak to moderate coupling | Pairwise cross-map skill | Only with seasonal surrogates; not as primary evidence (b.4) |
| Neural: TCDF (Nauta 2019 [V]), Neural Granger (Tank et al., TPAMI [V-meta]), DYNOTEARS (Pamfil 2020 [V-arXiv]), ACD (Löwe 2022 [V-arXiv]), Rhino (Gong 2023 [V-arXiv]), CUTS (Cheng 2023 [V-arXiv]) | Varied; usually no formal error control | Graph scores | Comparison arm on synthetic benchmarks only |

**Conditional-independence tests in tigramite:** ParCorr, RobustParCorr, ParCorrWLS (heteroskedastic), GPDC/GPDCtorch, CMIknn (Runge 2018, AISTATS [V-arXiv]), Gsquared, CMIsymb, RegressionCI and CMIknnMixed (Popescu et al. [V-arXiv 2310.11132]).

**Verified features (source code, 2026-10-22):**

- masks and a `missing_flag` option;
- `analysis_mode='multiple'` for pooled multi-dataset analysis;
- `link_assumptions`, which encodes background knowledge such as "nothing in the river causes air temperature";
- FDR correction;
- a CausalEffects class for effect estimation with optimal adjustment sets (Runge 2021, NeurIPS [V-arXiv]) and mediation (Runge et al. 2015 [V]; Runge 2015, PRE [M]).

### b.2 What the benchmarks say

- **Assaad et al. 2022, JAIR [V].** "No family or method stands out in all situations".
- **Lawrence et al. 2021 [V-arXiv].** Their configurable data generator (code at github.com/causalens/cdml-neurips2020, which exists) showed "a notable degradation in performance when their assumptions are invalidated" and sensitivity to hyperparameters.
- **PCMCI.** Validated on "well-understood physical mechanisms in the climate system and the human heart" and on large synthetic datasets, where it beat other methods in detection power (Runge 2019).
- **PCMCI+.** Higher adjacency power and contemporaneous orientation recall, with false positives better controlled.
- **LPCMCI.** Much higher recall than FCI-type methods under autocorrelation, with false positives "at the desired level".
- **CauseMe** (Runge et al. 2019, Nat. Commun. [V]). Live at causeme.uv.es; causeme.net redirects there, and it is labelled beta. Its synthetic benchmarks cover "time delays, autocorrelation, nonlinearity, chaotic dynamics, extreme events, measurement error", plus real datasets with known structure. Registration is required.
- **SAVAR** (Tibau et al. 2022, Environ. Data Sci. [V]): a spatio-temporal benchmark generator (github.com/xtibau/savar).

### b.3 A validation ladder (adopt as SENTINEL's "claim ladder")

| Rung | Evidence | What may be claimed |
|---|---|---|
| 0 | Raw (lagged) correlation | "associated" |
| 1 | Predictive improvement (Granger / out-of-sample skill) | "predictive of" |
| 2 | PCMCI+ link that survives FDR and bagging at one site | "conditionally dependent given measured drivers" |
| 3 | Rung 2 replicated across sites of the same class, with consistent sign and lag; survives seasonal surrogates and preprocessing variants | "robust candidate causal link" |
| 4 | Rung 3 plus agreement with an intervention-like natural experiment (storm pulses, dam releases, heatwaves, outfall changes) or with physics | "causal (with stated assumptions)" |

**Positive and negative controls are mandatory at every rung** (§f.4). Known mechanisms must be recovered at their correct time scale. Impossible links must be rejected: for example river → air temperature, downstream → upstream, or links that only appear when the data are time-reversed. The observed rejection rate is the empirical false-positive rate of the whole pipeline.

### b.4 Failure modes and mitigations

| Failure mode | Evidence | Mitigation |
|---|---|---|
| Autocorrelation inflates false positives in pairwise and Granger-type tests | Runge 2018, Chaos [V]; Runge 2019 | PCMCI+ MCI step; Bagged-PCMCI+; report effective sample size |
| Seasonality or diel cycles act as a common driver | Krich et al. 2020 [V]: "the violation of the method's assumptions due to non-stationarities increases the likelihood to detect false links"; SENTINEL's own run (b.6) | Include the drivers (shortwave/PAR, air temperature, discharge) as variables; analyse anomalies against a *training-period* climatology; also run seasonal windows (summer only) or RPCMCI; never interpolate across gaps |
| Effects at daily resolution are contemporaneous | PCMCI+ paper (resolutions "too coarse to resolve time delays") | Use PCMCI+ (`tau_min=0`) with `link_assumptions` to orient physically forced edges (weather → water) |
| Subsampling or aggregation flips or hides links | Gong et al. 2015, ICML [U] | Analyse at two resolutions (hourly and daily); a claim must hold at both, or the reason it does not must be explained |
| CCM detects "causality" between independent oscillators with shared periodicity | Cobey & Baskerville 2016 [V]: "extremely sensitive to periodic fluctuations"; Baskerville & Cobey 2017 [M]; Mønster et al. 2017 [V]: fails under strong and even intermediate coupling; McCracken & Weigel 2014 [V]: inferred driver depends on system parameters | Use CCM only with seasonal-surrogate nulls (Deyle et al. 2016 [V] used EDM with such controls) and time-delay checks (Ye 2015); never as stand-alone evidence |
| Missing data and uneven sample sizes | Delforge et al. 2022 [V]: the multivariate nonlinear method was best on synthetic data but "unstable because of the uneven distribution of missing values" on real data | tigramite masks; minimum-coverage rules; report N per test |
| Hidden confounders (nutrients, upstream releases, management) | general | LPCMCI as a check; J-PCMCI+ context nodes for site attributes; natural experiments |
| Regime shifts and nonstationarity (ice, drought, dam operations) | Saggioro 2020; Krich 2020 | RPCMCI; split by hydrologic regime; record breakpoints |
| Measurement error, sensor drift, biofouling | Runge 2018 | QA flags; drift-corrected (approved) data only; sensitivity to noise injection |
| Multiple testing across hundreds of sites | — | Benjamini–Hochberg (Benjamini & Hochberg 1995 [V]) within site; replication-based inference across sites, not pooled p-values |
| Max-lag choice | — | Set from physics (travel time, response time), not only from the ACF; test τ_max sensitivity |
| Choice of method or CI test drives results | Assaad 2022; Lawrence 2021 | Pre-register a primary configuration; report alternatives as robustness checks |

### b.5 Published hydrology and WQ applications, and how strongly they were validated

| Study | Method | System | Finding | Validation strength |
|---|---|---|---|---|
| Ombadi et al. 2020, WRR [V] | Granger, TE, graph-based (PC), CCM | Synthetic hydrologic model plus shrubland ET | Compared the methods' assumptions and their sensitivity to sample size and noise; produced hypotheses on ET drivers by season | Synthetic ground truth: good. Real data: hypothesis-level |
| Delforge et al. 2022, HESS [V] | 4 CIMs (linear/nonlinear, bi-/multivariate) | Synthetic plus a real karst system | Multivariate nonlinear had low FPR on synthetic data; unstable on real data because of missing values | Good on synthetic data; the real-data caveats are stated honestly |
| Krich et al. 2020, Biogeosciences [V] | PCMCI | FLUXNET sites, GIMMS NDVI | Recovered lagged dependencies; non-stationarity raises false links | Synthetic data plus replicated sites |
| Sendrowski & Passalacqua 2017, WRR [V] | Transfer entropy | Wax Lake Delta water levels | Information-transfer timescales "compare well to physical travel times" from a hydrodynamic model, though they are longer | **Physics cross-check.** This is the template for E2 |
| Bennett et al. 2019, WRR [V] | Transfer entropy | Process networks of SUMMA, VIC and PRMS models | Models that fit similarly have different process connectivity | Model-to-model |
| Ruddell & Kumar 2009, WRR [V] | Transfer-entropy process networks | Corn–soy FLUXNET, normal vs. drought | The drought network was "substantially decoupled" | Contrast between natural states |
| McGowan et al. 2017, Ecology [V]; Ushio et al. 2018, Nature [M]; Deyle et al. 2013, PNAS [M] | EDM / CCM | Coastal phytoplankton blooms; fish community; sardine–climate | Bloom dynamics behaved as "stochastic chaos" and were predicted out of sample | **Out-of-sample prediction**, the strongest real-data check available to EDM |
| Nowack et al. 2020, Nat. Commun. [V] | PCMCI fingerprints | Climate models vs. reanalysis | Causal fingerprints rank model realism | Emergent-constraint style |
| PCMCI, CCM or Granger applied directly to river WQ sensors | — | — | I could not complete a systematic search this session [U] | Expect mostly weakly validated case studies; check before citing |

### b.6 Post-mortem of SENTINEL's genuine PCMCI+ run

This section is based on reading `scripts/experiments/causal_lowperf/causal-real.py` and `results/benchmarks/causal_lowperf/causal-real.json`.

**What was run:**

- Data: NEON 30-min data for one month at two lakes (BARC 2025-10, LIRO 2025-08), plus a longer stream record (LEWI).
- Method: PCMCI+ with ParCorr and BH-FDR (α = 0.05).
- τ_max was chosen from the ACF and **capped at 12 steps (6 h)**.
- Gaps of 3 h or less were linearly interpolated.
- **No light, weather or discharge variables** were included.
- 3 of 15 "known chains" were testable; 0 were recovered.

The JSON's own interpretation is correct: raw temperature→DO r = −0.82, but the link was "conditioned away" because the series were "dominated by a shared diel solar cycle and the target's own autocorrelation".

**Diagnosis and fixes:**

1. **Omitted common driver.** At diel scale, sunlight drives both water temperature and photosynthetic DO. **Fix:** include PAR or shortwave radiation (NEON DP1.20042.001 exists at all 34 aquatic sites) and discharge.
2. **Wrong formulation of a deterministic link.** "Temperature → DO (solubility)" is physics and does not need discovery. Compute DO_sat(T, P) and analyse the **deficit** D = DO − DO_sat, which isolates biology and gas exchange.
3. **Lags outside the window.** Several chains had lags of 24–168 h, but the window stopped at 6 h. One month cannot support multi-day lags or seasonal regimes.
4. **Interpolation** creates smooth artificial segments. Use masks instead.
5. **Some chains were untestable by design.** Chains that need satellite, microbial or molecular modalities do not belong in a single-sensor validation set.

---

## (c) Ecological interactions and thresholds relevant to WQ

### c.1 Temperature–DO–metabolism

- **Solubility versus biology.** In lakes, surface-water DO declines track solubility, while deep-water declines track stratification and clarity (Jane et al. 2021, Nature [V]: 393 lakes, 1941–2017; freshwater O₂ losses 2.75–9.3 times those of the oceans). In rivers, Zhi et al. 2023 attribute deoxygenation mostly to warming, with land use modulating the rate.
- **Stream metabolism regimes.**
  - Appling et al. 2018, Sci. Data [V]: 356 US rivers, Oct 2007 to Jan 2017, 490,907 site-days of daily GPP, ER and K₆₀₀.
  - Bernhardt et al. 2018, L&O [V]: research agenda for river metabolic regimes.
  - Savoy et al. 2019, L&O [V]: 47 rivers in two regimes, "Summer Peak" and "Spring Peak", with class membership predicted "with modest accuracy" from watershed area, temperature and discharge characteristics.
  - Bernhardt et al. 2022, PNAS [V]: more than 200 rivers; "for rivers the most important controls are annual light availability and flow stability", unlike temperature and precipitation for terrestrial biomes.
  - Koenig et al. 2019 [V]: emergent network-scale regimes.
  - Savoy & Harvey 2021, GRL [V]: a CONUS light model in which water-column processes limit productivity for "50 % of the nation's river length and 80 % of its surface area".
  - Marzolf et al. 2025 [V]: NEON metabolism.
- **Hypoxia** (Blaszczak et al. 2023, L&O Letters [V]):
  - 118 million paired DO and temperature observations from more than 125,000 river locations in 93 countries;
  - hypoxia (DO < 2 mg/L) at 12.6 % of sites, with no consistent trend since 1950;
  - a median event duration of 3 h, with events most likely to begin at night;
  - more likely in "warmer, smaller, and lower-gradient rivers, particularly those draining urban or wetland land cover";
  - river attributes predicted occurrence better than watershed land cover, topography and climate.
  - **High-frequency data matter: hypoxia is a night-time, hours-long phenomenon that daily means miss.**
- **Temperature dependence:** 0.65 eV for ER (Yvon-Durocher 2012) and the NEP decline with warming (Song 2018), as in §a.6. Demars et al. 2011 [M] is a further reference.

### c.2 Nutrient → chlorophyll → DO cascades

- Mechanisms are reviewed by Dodds & Smith 2016 (Inland Waters) [M].
- **The measurement reality:** continuous nitrate sensors exist at about 308 USGS locations (191 active). Only about 150 also have DO, temperature and discharge (Appendix A). Phosphorus is essentially never measured continuously.
- Pellerin et al. 2016 (JAWRA) [V] describe the vision of a national nitrate-sensor network.
- **Implication:** at continental scale, nutrient→DO links can be studied only as cross-site associations, using WQP or NRSA grab samples as site traits. They cannot be studied as time-series causal chains, except at the roughly 58 active DO+T+Q+nitrate sites (≥ 5 years of overlap).
- Chlorophyll fluorescence sensors exist at NEON (DP1.20288.001 includes chlorophyll) and at some USGS sites. I did not count them here.

### c.3 Discharge–turbidity hysteresis and C–Q behaviour

- **Hysteresis classes and indices:**
  - classic loop classes (Williams 1989 [M]);
  - the hysteresis index with uncertainty (Lloyd et al. 2016 [M]);
  - a versatile loop index tested on synthetic and field data (Zuecco et al. 2016 [V]);
  - a review of event metrics (Liu et al. 2021, Water Research [M]).
- **Hamshaw et al. 2018, WRR [V]:** more than 600 storm events, 14 hysteresis classes, and a restricted Boltzmann machine trained on images of hysteresis plots that classified 70 % of events correctly or into the next most similar class. This is the template for ML-based classification at scale.
- **Vaughan et al. 2017 [V]:** 126 storms with 15-min DOC and nitrate. DOC loops were generally anticlockwise; nitrate was clockwise at urban and forested sites and anticlockwise at the agricultural site. "Hysteresis indices did not improve predictions of storm nitrate yields". This is a useful negative result.
- **C–Q patterns:**
  - Godsey et al. 2009 [V]: chemostasis in 59 US catchments; weathering solutes vary 3–20× while Q varies by orders of magnitude.
  - Moatar et al. 2017 [V]: 293 French stations; 84 % chemodynamic over at least half the hydrograph, and 60 % nonlinear.
  - Musolff et al. 2015 [M].
  - Rose et al. 2018 [V]: geogenic solutes show dilution and clockwise loops; biologically associated solutes show enrichment and anticlockwise loops; particulates show positive slopes and clockwise loops.
- **Tools:** hydroEvents (R, CRAN 0.13.0; Wasko & Guo 2022 [V-meta]) and HydRun (MATLAB; Tang & Carey 2017 [V-meta]) for event extraction.
- **Scale check:** 916 USGS locations have continuous turbidity and discharge (521 with ≥ 5 years of overlap; 431 of those active).

### c.4 Salinity–community thresholds

- **TITAN** (Baker & King 2010 [V-meta]; King & Baker 2010 [M]; R package TITAN2 2.4.4, CRAN, 2025-10-16).
  - Critique: Cuffney & Qian 2013, Freshwater Science, doi:10.1899/12-056.1 [M], argued that indicator-species scores are unreliable for identifying thresholds.
  - Reply: Baker & King 2013 [M].
  - Alternative: **gradient forest** (Ellis et al. 2012, Ecology [V]), which builds compositional-turnover functions from random forests. It is on R-Forge, not CRAN.
- **EPA conductivity benchmark.**
  - The EPA's 2011 *Field-Based Aquatic Life Benchmark for Conductivity in Central Appalachian Streams* is cfpub.epa.gov record 233809 [V]. It sets **300 µS/cm**, derived as the level at which 5 % of macroinvertebrate genera are extirpated.
  - Cormier et al. 2013 (ETC) [V] give the derivation: 2,210 West Virginia samples and 163 genera, with a regional background of 116 µS/cm. They also give the method (Cormier & Suter 2013 [V]) and a weight-of-evidence causal assessment (Cormier et al. 2013 [V]).
  - Clements & Kotalik 2016 [M] tested the benchmark experimentally.
  - Olson & Cormier 2019 (ES&T) [M] modelled natural background specific conductance. This matters because thresholds are relative to background.
- **Salinization context.**
  - Kaushal et al. 2018, PNAS [V]: salinization and alkalinization affected 37 % and 90 % of contiguous-US drainage area over the past century; 66 % of 232 USGS sites showed rising pH.
  - Kaushal et al. 2021 [V] and Stets et al. 2020, ES&T [V]: "freshwaters are being salinized rapidly in all human-dominated land use types".
  - Cañedo-Argüelles et al. 2013 [M].
  - Hintz & Relyea 2019 [V]: effect concentrations for road salt "varied considerably".

### c.5 Microbial co-occurrence networks and their limits

- **Methods:**
  - SparCC (Friedman & Alm 2012 [V]). Compositional effects can be severe; on their reference network, standard correlation gave "3 spurious species-species interactions for each true interaction" and missed 60 %.
  - FastSpar (Watts et al. 2019 [V]), 2–3 orders of magnitude faster.
  - SPIEC-EASI (Kurtz et al. 2015 [V]).
  - FlashWeave (Tackmann et al. 2019 [V]), which integrates metadata and scales to more than 69,000 samples.
  - CoNet (Faust et al. 2012 [M]).
- **Limits:**
  - Hirano & Takemoto 2019 [V]: in realistic simulations, co-occurrence methods performed "almost equal to or less than" Pearson correlation, and poorly for predator–prey interactions.
  - Weiss et al. 2016 [V]: eight methods "vary widely in sensitivity and precision".
  - Carr et al. 2019 [V]: correlation "almost never provides reliable information on direct biotic interactions".
  - Blanchet et al. 2020 [V]: co-occurrence "is a poor proxy for ecological interactions".
  - Faust 2021 [V]; Röttjers & Faust 2018 [M]; Gloor et al. 2017 [M] (compositionality).
- **SENTINEL rule:** microbial networks may generate hypotheses only. Report them as "association networks". Relevant NEON surface-water microbe products include DP1.20282.001 (marker gene sequences, 34 sites) and DP1.20138.001 (cell counts).

### c.6 Critical transitions and early-warning signals

- **The theory and the whole-lake experiments:**
  - theory (Scheffer et al. 2009 [V-meta]; Dakos et al. 2012 [V-meta]);
  - whole-lake demonstrations: Carpenter et al. 2011 [V-meta]; Pace et al. 2017, PNAS [V], where a bloom was reversed after indicators passed a threshold;
  - Wilkinson et al. 2018 [V]: blooms in 5 of 6 lake-years, with at least one indicator warning 1–57 days before peak chlorophyll.
- **Caution:**
  - Burthe et al. 2016 (J. Appl. Ecol.) [V]: true positives were "rare", only 9 % (variance) and 13 % (autocorrelation), and false positives were more common (53 % and 47 %).
  - Boettiger & Hastings 2012 [V]: error rates "can be quite severe".
  - Gsell et al. 2016, PNAS [M].
  - Bury et al. 2021, PNAS [V]: a deep-learning detector was more sensitive with fewer false positives in their tests.
- **SENTINEL rule:** any early-warning claim must pass the placebo test that the earlier detector failed (alarms on random clean dates).

### c.7 Thermal sensitivity and groundwater signatures

- **Kelleher et al. 2012 [V].** Thermal sensitivity (the slope of water temperature against air temperature) ranged from 0.02 to 0.93 across 57 Pennsylvania sites. Stream order and baseflow index explained 43 % of its variance.
- **Hare et al. 2021, Nat. Commun. [V].** Paired multi-year air and stream temperature signals classify 1,729 CONUS sites as dam-influenced, shallow-groundwater, deep-groundwater or atmospheric. About 40 % of non-dam sites had substantial groundwater signatures, and shallow-groundwater sites showed "reduced baseflow and a higher proportion of warming trends".
- Method background: Briggs et al. 2018 (STOTEN) [V-meta].
- These are cheap, physically interpretable site traits.

### c.8 Derived ecological features: a catalogue for the platform

| Feature | Computation | Data | Validation |
|---|---|---|---|
| DO_sat, %sat, deficit D | Benson–Krause with pressure (elevation) correction | Continuous T, DO | Unit test against DOTABLES |
| Daily GPP, ER, K₆₀₀ | streamMetabolizer (Bayesian, K–Q pooling) or the twin's process head | DO, T, light (modelled or NEON PAR), depth (Raymond 2012 hydraulic geometry), Q | Agreement with Powell Center and NEON estimates at overlapping sites; GPP ≥ 0 and ER ≤ 0 constraints; K₆₀₀–Q coherence |
| Metabolic regime class | Clustering of annual GPP phenology (Savoy 2019) | GPP series | Stability across years; a predictable mapping to light and flow stability (Bernhardt 2022) |
| Thermal sensitivity and groundwater class | Air–water regression; Hare 2021 metrics | T, gridMET/Daymet air temperature | Agreement with Hare 2021 site classes where sites overlap |
| C–Q slope (low and high flow) | Segmented log–log regression (Moatar 2017) | SpC, turbidity, nitrate; Q | Bootstrap CI; sign stability across years |
| Hysteresis index and class | HI (Lloyd 2016) and the Zuecco index on extracted events; CNN or RBM classes | 15-min turbidity, Q | Agreement between methods; event bootstrap |
| Diel DO amplitude and nightly minimum | From the hourly series | DO | Sensor QA; relation to GPP |
| Hypoxia hours | Hours with DO < 2 or < 3 mg/L | DO | Night-time prevalence consistent with Blaszczak 2023 |
| Heatwave days | Percentile-based (Tassone et al. 2023 [V]: river heatwave frequency increased 1996–2021, with annual heatwave days doubling) | T | Reproduce Tassone's trends at overlapping sites |
| Intrinsic predictability | Weighted permutation entropy (Pennekamp 2019 [V]: PE correlates with forecast error across 461 series) | Any series | Correlation with realized CRPS skill |

---

## (d) Interpretability for hydrology and WQ deep models

### d.1 Hydrology-specific evidence that interpretation can work

- **Kratzert et al. 2019, "NeuralHydrology – Interpreting LSTMs in Hydrology"** (LNCS 11700, pp. 347–362) [V]. On one snow-influenced and one rain-dominated catchment, the network "internally learns to represent patterns that are consistent with our qualitative understanding of the hydrological system".
- **Lees et al. 2022, HESS [V].** Linear probes from the LSTM cell state to soil moisture and snow gave R² > 0.8. The results were "robust to different initial conditions" and benchmarking experiments. *This is the template for probing the twin's states against GPP, ER and K₆₀₀.*
- **Jiang et al. 2022, WRR [V].** Expected gradients on LSTMs for 160 catchments revealed three flood mechanisms: snowmelt 10.1 %, recent rainfall 60.9 % and historical rainfall 29.0 % of 20,908 peaks. Single mechanisms dominate 70.7 % of catchments.
- **Topp et al. 2023 [V].** XAI explained *why* one architecture generalized under shifted conditions. This shows interpretation can serve as a validation tool, not only as decoration.
- **Toms et al. 2020, JAMES [V]; McGovern et al. 2019 [V-meta]; Ebert-Uphoff & Hilburn 2020 [V-meta].** Interpretation is valuable as a discovery tool when combined with hypotheses.

### d.2 Gradient attributions and their fidelity

- **Methods.** Integrated gradients (Sundararajan et al. 2017 [V-arXiv]) are complete and baseline-dependent. Expected gradients (Erion et al. 2021 [V-meta]) average over a baseline distribution. Others: DeepLIFT (Shrikumar 2017 [V-arXiv]) and LRP (Bach 2015 [V-meta]). Implementation: Captum 0.9.0 (Kokhlikyan 2020 [V-arXiv]).
- **Baselines matter.** Sturmfels et al. 2020 (Distill) [V-meta]. For hydrology, use climatology or training-distribution baselines, not zeros (a "zero" air temperature is a real, cold value).
- **Fidelity:**
  - Mamalakis et al. 2022 (Environ. Data Sci.) [V] built attribution benchmarks from additively separable functions with known ground truth.
  - Mamalakis et al. 2022 (AIES) [V] found "gradient shattering, inability to distinguish the sign of attribution, and ignorance to zero input".
  - Adebayo et al. 2018 [V-arXiv] proposed model and data randomization sanity checks.
  - Kindermans et al. 2019 [V-meta]: input-shift unreliability.
  - Hooker et al. 2019 (ROAR) [V-arXiv]: retrain-after-removal.

### d.3 SHAP and permutation importance with correlated inputs (the core pitfall)

- **Hooker, Mentch & Zhou 2021 [V].** Permute-and-predict methods "can vastly over-emphasize correlated features", because breaking dependencies "forc[es] the original model to extrapolate". The recommendation is to refit or mute instead.
- **Aas et al. 2021 [V].** KernelSHAP assumes independent features. With correlation, "the explanations may be very misleading. This is the case even if a simple linear model is used".
- **Kumar et al. 2020 [V-arXiv]** (mitigations need causal reasoning) and **Janzing et al. 2020 [V-arXiv]** (interventional rather than observational conditioning). These two disagree about the right conditional distribution. Report both "true to the model" and "true to the data" variants (Chen et al. 2020 [V-arXiv]).
- **Causal-aware variants:** asymmetric Shapley values (Frye 2020 [V-arXiv]) and causal Shapley values (Heskes 2020 [V-arXiv]).
- **Global alternatives:** SAGE (Covert 2020 [V-arXiv]) and ALE plots (Apley & Zhu 2020 [V]), which "do not require this unreliable extrapolation with correlated predictors". Conditional importance: Strobl et al. 2008 [V-meta].
- **General pitfalls:** Molnar et al. 2022 [V], including "interpreting models that do not generalize well" and "unjustified causal interpretations".
- **Practical rule for SENTINEL's forcings,** where air temperature, radiation and vapour pressure co-vary strongly: compute attributions over **feature groups** (weather, hydrology, static attributes) and use ALE for dependence. Phrase results as "the model relies on X", never "X causes Y".

### d.4 Attention

- Attention weights are not faithful explanations by default (Jain & Wallace 2019 [V]). Faithfulness has to be tested, not assumed (Wiegreffe & Pinter 2019 [V]).
- The Temporal Fusion Transformer (Lim et al. 2021 [V]) offers built-in variable selection and attention patterns. Use them as hypotheses, and check them with deletion tests.

### d.5 Concept bottlenecks and physically structured latents

- Concept bottleneck models (Koh et al. 2020 [V-arXiv]) and TCAV (Kim et al. 2018 [V-arXiv]) make predictions pass through human-meaningful concepts.
- **In SENTINEL, the DO process head *is* a concept bottleneck:** DO must be explained through GPP, ER, K₆₀₀ and DO_sat.
- Rahmani 2023 warns that such latents can be under-constrained by a single target. Mitigations:
  - (i) compare with independent metabolism estimates;
  - (ii) use priors (K₆₀₀–Q from Raymond 2012 and Appling 2018);
  - (iii) use multi-target training (DO plus temperature, plus a weak GPP–light coherence penalty).

### d.6 Symbolic regression on learned components

- **Tools:** PySR (Cranmer 2023 [V-arXiv]; PySR 2.7.2, Apache-2.0), SINDy (Brunton et al. 2016 [V]; pysindy on GitHub), AI Feynman (Udrescu & Tegmark 2020 [V]). Cranmer et al. 2020 [V-arXiv] distilled graph-network components into equations.
- **Earth-science precedents** (climate rather than hydrology):
  - ocean mesoscale closures with conservation laws (Zanna & Bolton 2020 [V]);
  - a cloud-cover equation with R² = 0.94 and 13 parameters that "exhibits superior transferability" to ERA5 (Grundner et al. 2024 [V]).
- I could not verify a hydrology or WQ application of symbolic regression to *learned* differentiable-model components in this session [U]. This is an opening for SENTINEL (§f.5).

### d.7 Emulator-based global sensitivity analysis

- **Sobol' indices:** Sobol' 2001 [V-meta]; estimators in Saltelli et al. 2010 [V-meta]; SALib 1.6.0 (Herman & Usher 2017 [V-meta]; Iwanaga et al. 2022 [V-meta]).
- **Other methods:** Morris screening (Morris 1991 [V-meta]); VARS (Razavi & Gupta 2016 [V-meta]); field review in Razavi et al. 2021 [V-meta].
- **Dependent inputs:** Shapley effects (Owen 2014 [V-meta]; Song, Nelson & Staum 2016 [V-meta]).
- **Why it fits:** the twin is a fast emulator, so a Sobol design with tens of thousands of runs per site is cheap on a GPU. This answers questions such as "which scenario forcing dominates the change in nightly DO minimum, and where?"

### d.8 Uncertainty quantification

- Deep ensembles (Lakshminarayanan et al. 2017 [V-arXiv]) and MC dropout (Gal & Ghahramani 2016 [V-arXiv]).
- Mixture-density heads (CMAL/GMM/UMAL) were strong baselines in Klotz et al. 2022 (HESS) [V] and are available in NeuralHydrology.
- Conformal methods: adaptive conformal inference under distribution shift (Gibbs & Candès 2021 [V-arXiv]) and EnbPI (Xu & Xie 2021, ICML; extended in TPAMI 2023 [V-meta]).
- Data assimilation and autoregression for LSTMs: Nearing et al. 2022 [V-meta].

---

## (e) Novel-insight candidates with validation plans

### e.0 What the public data actually support (live queries, 2026-10-22)

These are USGS Water Data API v1 counts of `time-series-metadata` entries with `computation_period_identifier=Points`, meaning continuous series. Details are in Appendix A.

| Continuous parameter | Locations | Active (data into 2025+) | ≥ 10-yr span | ≥ 10-yr and active |
|---|---|---|---|---|
| DO (00300) | 1,680 | 905 | 483 | 441 |
| Water temperature (00010) | 5,011 | 2,747 | 1,553 | 1,421 |
| Specific conductance (00095) | 2,782 | 1,541 | 889 | 801 |
| Turbidity, FNU (63680) | 1,553 | 861 | 404 | 358 |
| In-situ nitrate+nitrite (99133) | 308 | 191 | 61 | 61 |
| Discharge (00060) | 14,441 | — | — | — |

| Co-located continuous set | Locations | ≥ 5-yr overlap | ≥ 5-yr overlap and active |
|---|---|---|---|
| DO + T + Q | 821 | 500 | 424 |
| + SpC | 750 | 466 | 391 |
| + SpC + turbidity | 526 | 299 | 246 |
| DO + T + Q + nitrate | 150 | 69 | 58 |
| Turbidity + Q | 916 | 521 | 431 |

**Caveats.** Counts include non-stream site types (lakes and estuaries; wells for temperature). "Span" means first-to-last date, not completeness, and some deployments are seasonal. Filter by site type and completeness before quoting any of these numbers.

**Other data facts:**

- **NEON:** 34 aquatic sites (DP1.20288.001 water quality at 34 sites, 2016-03 to 2026-09). Macroinvertebrates (DP1.20120.001), microbes (DP1.20282.001) and PAR (DP1.20042.001) are available at all 34. Discharge (DP4.00130.001) and reaeration (DP1.20190.001) cover 28 and 24 sites.
- **NLDI pairs:** among 250 randomly sampled USGS sites with active continuous temperature and discharge and ≥ 5 years of temperature record (population 1,374), **at least 129 (52 %) have another such site downstream on the main stem within 100 km, and at least 159 (64 %) within 300 km**. Twenty requests failed, so these are lower bounds.

### E1 (top pick). Diel-resolved warming sensitivity of DO: how much is solubility, and how much is metabolism?

- **Question.** Across about 400–500 USGS sites, and at NEON, how do daily mean DO, nightly minimum DO and the saturation deficit D respond to water temperature? Compare within-season, detrended anomalies with the pure-solubility slope (Benson–Krause), and ask which catchment traits explain departures from solubility (StreamCat, Hare groundwater class, light regime, size, slope, land use).
- **Why it is interesting.**
  - Zhi et al. 2023 found widespread deoxygenation from *daytime grab samples* and noted that nocturnal declines are probably underestimated.
  - Blaszczak et al. 2023 showed hypoxia is nocturnal and lasts hours.
  - A sensor-based, diel-resolved decomposition answers whether warming hits the **nightly minimum** harder than the mean, and where. That is the ecologically relevant quantity.
- **Method.**
  1. QA and masking of the data.
  2. Hourly DO and T, then daily mean, minimum and maximum; DO_sat with a pressure correction from elevation.
  3. For each site and season, regress anomalies (relative to a training-period climatology) of DO_min, DO_mean and D on T anomalies, controlling for discharge and shortwave radiation. Use mixed models with site random slopes.
  4. Partition using the metabolism model: ΔD ≈ Δ(NEP)/(K·z) plus the gas-exchange temperature effect (about +5 % K_O2 per +2 °C).
  5. Explain the slopes with ALE on catchment traits under spatial-block CV.
- **Validation.**
  - (i) Physics check: slopes for %saturation-stable sites must equal the solubility slope within CI.
  - (ii) Split-half replication by years.
  - (iii) Agreement in sign and magnitude with GPP and ER from the Powell Center dataset at overlapping sites.
  - (iv) Natural experiments: heatwave events (Tassone 2023) versus matched non-heatwave periods, using event-study or difference-in-differences against nearby sites.
  - (v) Sensitivity to sensor drift: exclude provisional and estimated data.
- **What would count as a finding.** A robust map showing that, for example, nightly DO minima decline faster than solubility predicts at warm, low-gradient, urban or wetland-draining, shallow-groundwater sites, with an explained-variance figure under spatial CV.
- **Ratings.** Novelty: Medium–High (provisional; check prior sensor-based DO–temperature studies). Feasibility: High. Main risks: sensor biofouling, and confounding by flow (handled with Q in the model and by stratifying by flow).

### E2 (top pick). How far and how fast do WQ signals propagate downstream, and what lead time does an upstream sensor buy?

- **Question.** For upstream–downstream sensor pairs from NLDI main-stem navigation, how do lag and attenuation of temperature, SpC and turbidity pulses scale with channel distance, travel time (NHDPlusV2 flow and velocity estimates [U: field names]), tributary dilution and dams? What is the *useful warning lead time* by river size?
- **Why it is interesting.** Sendrowski & Passalacqua 2017 showed that information-transfer timescales track physical travel times in one delta. A CONUS-scale WQ-sensor version would directly inform where early-warning sensors should go. I know of no such analysis [U].
- **Method.** Event-based cross-correlation and transfer entropy (or PCMCI+ on the pair plus weather drivers); estimate lag, gain and attenuation; regress on distance, travel time, ΔQ and dam presence (NLDI/NID).
- **Validation.**
  - (i) Physics: estimated lags must track computed travel times. Report the slope and R².
  - (ii) Negative controls: the reverse direction must show no lagged signal beyond the shared-weather null; disconnected pairs must show nothing beyond the weather-driven baseline.
  - (iii) Storm events as natural experiments.
  - (iv) Out-of-sample prediction of downstream pulses from upstream data, measured as skill over persistence.
- **Ratings.** Novelty: Medium–High (provisional). Feasibility: High (pairs exist). Risk: shared regional weather mimics propagation. Model weather explicitly and use time-reversal controls.

### E3. Groundwater buffering as a trait for DO and thermal resilience

Classify sites with Hare et al. 2021 metrics. Recompute them where Hare's classes do not cover a site, and validate against Hare's published classes. Then test whether shallow-groundwater sites show larger warming sensitivity of temperature and DO_min than deep-groundwater sites (E1 slopes), with spatial-block CV.

**Ratings.** Novelty: Medium–High as a *combination*. Feasibility: High. Risk: dam influence. Exclude dam-class sites, as Hare did.

### E4. A continental turbidity–discharge hysteresis typology

- **Method.** Run event extraction (hydroEvents) on about 431–521 turbidity+Q sites; compute HI and Zuecco indices and CNN or RBM classes (Hamshaw 2018); explain class frequencies by catchment traits, season and antecedent conditions.
- **Validation.** Agreement between methods; event bootstrap; consistency with Williams 1989 and Rose 2018 expectations (clockwise for proximal sources, anticlockwise for distal ones); temporal stability.
- **Ratings.** Novelty: Medium (methods are established; I could not confirm whether a CONUS-wide sensor typology exists [U]). Feasibility: High. Risk: turbidity-sensor saturation and fouling; FNU versus NTU method changes. Use pcode 63680 only.

### E5. Hypoxia-risk surfaces: temperature × low flow × catchment traits at diel resolution

- **Method.** Model the probability of hourly DO < 2 and < 3 mg/L as a function of water temperature, Q percentile, light and traits, using GAMs or gradient boosting with ALE and calibrated probabilities. Nitrate enters as a site trait from WQP WQX 3.0, and as a dynamic input only at the roughly 58 active nitrate-sensor sites.
- **Validation.** Spatial-block CV; calibration curves; check against the Blaszczak 2023 predictor ranking; a placebo test on random clean periods.
- **Ratings.** Novelty: Low–Medium (Blaszczak covered global drivers). The added value is the temperature × flow *interaction thresholds* at sub-daily resolution in the US. Feasibility: High.

### E6. Which sites are "early-warnable"?

- **Method.** Compute intrinsic predictability (weighted permutation entropy; Pennekamp 2019) and realized forecast skill horizons (CRPS skill over persistence by lead time; Petchey 2015 [V]; Lewis 2022 [V]) for DO, SpC and turbidity, and relate both to traits and to the availability of upstream sensors (E2).
- **Validation.** Correlation between PE and realized skill (Pennekamp's test); prospective scoring on the NEON Challenge.
- **Ratings.** Novelty: Medium. Feasibility: High once the twin exists.

### E7. Salinity–macroinvertebrate thresholds: replicating the 300 µS/cm benchmark nationally

- **Method.** Use NRSA site-level chemistry and macroinvertebrates (EPA NARS data pages verified) plus NEON's 34 sites. Run the extirpation-concentration method of Cormier & Suter 2013, TITAN2 and gradient forest, stratified by ecoregion and by natural background SpC (Olson & Cormier 2019).
- **Validation.** Agreement with the EPA 2011 and Cormier 2013 values in Appalachia (a positive control); methodological triangulation given the Cuffney & Qian critique; bootstrap CIs.
- **Ratings.** Novelty: Low (replication) but very defensible. NEON alone is underpowered (34 sites).

### E8. NEON metabolism regimes and DO extremes

Join the Marzolf 2025 metabolism estimates with NEON water quality and macroinvertebrates. Test whether metabolic regime class (Savoy 2019) predicts hypoxia hours and community metrics.

**Ratings.** Novelty: Low–Medium. Feasibility: High. Small N.

### Ranked shortlist

1. **E1:** high relevance to "WQ disasters", a clear gap left open by Zhi 2023, physics-anchored validation, and data that exist.
2. **E2:** directly informs early-warning design; physics gives a hard positive control (travel time).
3. **E3:** cheap to add to E1, and an interpretable mechanism.
4. **E4:** a large, clean, classifiable dataset with an ML angle (CNN classification).
5. **E6:** turns the twin and forecasts into a scientific statement about predictability.

E5 and E7 are good secondary or replication pieces.

---

## (f) Recommended design

### f.1 Architecture overview

```
 Data layer (QA'd, provenance-hashed)
   USGS Water Data API v1 (continuous T, DO, SpC, turbidity, nitrate, Q) · NEON (34 aquatic sites)
   gridMET/Daymet (air T, precip, shortwave, VPD, wind) · StreamCat · NLDI/NHDPlus topology · WQP WQX3.0
        │
        ▼
 Derived-feature layer (c.8): DO_sat, deficit, GPP/ER/K600, thermal sensitivity, GW class,
                              C–Q slopes, hysteresis class, heatwaves, permutation entropy
        │                                  │
        ▼                                  ▼
 SENTINEL-Twin v2 (f.3)            Causal & interaction pipeline (f.4)
 multi-site LSTM + process heads   controls → synthetic → PCMCI+/J-PCMCI+ → replication → natural expts
        │        ▲                         │
        ▼        └──── causal fingerprints (twin vs. obs) ◄┘
 Interpretability suite (f.5): fidelity tests → IG/EG (grouped) → probes → symbolic regression → Sobol
        │
        ▼
 Scenario platform: ΔT, P/Q scaling, shading (SW) scenarios with uncertainty and validity-envelope flags
```

### f.2 Data layer and QA (non-negotiable)

- **Sources.**
  - USGS: `api.waterdata.usgs.gov/ogcapi/v1` through `dataretrieval.waterdata` (≥ 1.4). Register an API key for rate limits; the team does this, and keys are never committed. **Do not build on `waterservices.usgs.gov`** (decommission is planned for early 2027).
  - WQP: use WQX 3.0 profiles (`waterqualitydata.us/beta/`), because the legacy profiles lack USGS data added after 2024-03-11.
  - NLDI: `api.water.usgs.gov/nldi/`.
  - NEON: neonutilities.
- **Masks.** Remove sentinel values (−999999 and similar), qualifier-flagged values (ice, equipment malfunction, estimated or provisional for analysis datasets), and values outside physical ranges. Examples: DO 0–25 mg/L; %saturation 0–300 %; water temperature −1 to 40 °C; SpC > 0; turbidity ≥ 0 and below sensor maximum.
- **Hard failures.** No fallback data paths. A minimum-N assertion runs. Record a provenance hash (site list, date range, API version, query timestamp) in every result JSON.
- **Splits.** Fit normalization statistics on training sites and years only (the `train_hydrovit_real.py` discipline already in the repo). Write site-disjoint and region-disjoint fold files once and freeze them.

### f.3 Simulation engine: SENTINEL-Twin v2

**Model.**

- **Backbone:** a multi-site LSTM in NeuralHydrology (CudaLSTM or EA-LSTM). Inputs are daily forcings (gridMET or Daymet: Tmin, Tmax, precipitation, shortwave, VPD, wind), observed or simulated Q, and static attributes (StreamCat, elevation, NLDI drainage area, Hare groundwater class).
- **Temperature head:** water temperature regression, optionally with Rahmani-2023-style source-temperature priors.
- **DO head (differentiable process module):** daily outputs GPP_d ≥ 0, ER_d ≤ 0 and K₆₀₀,d > 0 (softplus), integrated hourly within each day:

```
for each hour h of day d:
    DOsat_h = BensonKrause(T_h) * P_site/P0           # fixed physics, no learnable params
    KO2_h   = K600_d * (Sc_O2(T_h)/600)**-0.5        # Wanninkhof Schmidt scaling
    dDO     = GPP_d * PAR_h/sum(PAR_d)/z_d + ER_d/z_d * θ_ER(T_h) + KO2_h * (DOsat_h - DO_h)
    DO_{h+1} = DO_h + dDO * Δt
loss = masked MSE/CRPS on observed hourly DO (+ daily T) + weak priors:
       K600 ~ f(Q) (Raymond 2012 / Appling 2018 pooling), ER Arrhenius slope prior (~0.65 eV)
```

- Hourly light comes from NEON PAR, or from solar geometry × daily shortwave at USGS sites. Depth z comes from hydraulic geometry against Q (Raymond 2012).
- **SpC and turbidity heads:** data-driven, with monotone-in-Q inductive bias tested rather than imposed (some sites flush). Optional event-state features (antecedent Q).
- **Training set:**
  - phase 1: about 400–500 DO+T+Q sites with ≥ 5 years of overlap (Appendix A);
  - phase 2: add the roughly 2,700 active temperature-only sites to the temperature head;
  - NEON sites are held out entirely for the prospective test.
- **Precision:** fp16 autocast with gradient scaling. Turing GPUs (2080 Ti) have fp16 tensor cores but no bf16. Avoid architectures that need custom CUDA kernels (e.g., Mamba) unless they are verified on CUDA 11.8 and torch 2.7.

**Scenario forcings.**

| Allowed | Notes |
|---|---|
| Air temperature +ΔT (delta change, or CMIP6 via NEX-GDDP as in Zhi 2023) | Adjust humidity consistently (hold RH or VPD as a stated choice); report the choice |
| Precipitation scaling, or Q scaling (low-flow and drought scenarios) | Q scaling only within the observed Q-percentile envelope of each site; flag extrapolation |
| Shortwave and PAR scaling (riparian shading or turbidity-driven light loss) | Light enters GPP explicitly, so this is mechanistic |
| **Conditional:** land-use attributes (urban or agricultural %) | Space-for-time sensitivity only, not an "effect"; label it explicitly |
| **Not allowed:** nutrient loads (except at nitrate-sensor sites, as "model association"); any learned embedding from other SENTINEL encoders | Embeddings have no physical units and cannot be perturbed meaningfully. Use them only to condition the state, never as scenario levers |

**Validation battery: pass/fail gates, pre-registered before training.**

1. **Skill gates.** On temporal, PUB and PUR hold-outs, beat persistence, DOY climatology and physics-only baselines on RMSE, KGE (benchmark −0.41) and CRPS. Report the median and the interquartile range across sites.
2. **Differential split-sample test.** Train on the coolest 70 % of summers and test on the hottest 30 %, with the same for the driest years. Skill loss must be reported and must stay above the baselines.
3. **Metamorphic tests**, run on every site with the expected direction and band:
   - air temperature +2 °C gives a water-temperature rise between 0 and 2 °C, consistent with the site's historical thermal sensitivity (Kelleher range 0.02–0.93);
   - the DO_sat change equals the Benson–Krause value exactly (a unit test);
   - daily mean DO change ≤ 0 at sites with stable %saturation, and close to the solubility-only value (−0.35 mg/L at 20 °C) unless the metabolism head explains the departure;
   - nightly DO minimum change ≤ the mean change at heterotrophic sites (ER rises with Arrhenius sensitivity);
   - Q × 0.5 gives an SpC rise at dilution-type sites, with a sign that matches each site's own historical C–Q slope;
   - placebo forcings (another basin's weather, or time-shuffled forcings) give no skill.
   - The share of sites passing each test is a reported result, not a hidden check.
4. **Event hindcasts.** Pre-register expected change directions for storms, heatwaves and droughts (the old twin matched 1 of 19). Report direction accuracy with a binomial CI.
5. **Internal-variable checks.** GPP, ER and K₆₀₀ against the Powell Center and NEON metabolism estimates (correlation, bias); K₆₀₀–Q coherence.
6. **Causal fingerprint.** PCMCI+ graphs on twin output and on observations at the same sites should agree (Nowack 2020 approach). Report graph-distance distributions.
7. **Prospective.** Submit daily DO and water-temperature forecasts for NEON sites to the NEON Ecological Forecasting Challenge and report CRPS against its climatology and persistenceRW baselines.

**Uncertainty.**

- A deep ensemble (5–10 seeds, one per GPU) combined with CMAL heads.
- Split-conformal recalibration by site group, using calibration sites and years disjoint from training. Use ACI for running forecasts.
- Scenario uncertainty from a CMIP6 ensemble or a range of ΔT values.
- An **extrapolation flag** for scenario inputs outside the training envelope, measured as distance in forcing space fitted on training data only.

**Compute (my estimate).** About 500 sites × 10 years × 365 days ≈ 1.8 M site-days. An LSTM with about 256 hidden units fits easily in 11 GB. Hourly integration inside the DO head adds about 24× work in the head only. Expect hours, not days, per ensemble member on one 2080 Ti, and run five members in parallel.

### f.4 Causal and interaction pipeline (validated)

- **Stage 0: data.** Same QA as f.2. Build daily (and hourly) matrices per site with drivers included: shortwave/PAR, air temperature, precipitation, Q, water temperature, DO deficit D (not raw DO), SpC, turbidity, and nitrate where available. Apply masks, not interpolation.
- **Stage 1: synthetic benchmarks from known physics.**
  - Simulate with (a) the twin's process head and (b) a stand-alone metabolism + C–Q + air–water-temperature simulator.
  - Use realistic autocorrelated forcing, gaps, noise and drift at 15-min sampling, then aggregate.
  - Score PCMCI+, LPCMCI, Bagged-PCMCI+, VAR-Granger, TE, CCM (with surrogates) and one neural method (CUTS or Rhino) on precision, recall, FPR and orientation accuracy.
  - Optionally score on CauseMe.
  - **Gate:** the primary configuration must recover at least 80 % of the true links at the target resolution with an empirical FPR of 5 % or less. Otherwise change the configuration before touching real data.
- **Stage 2: positive and negative controls on real data.**
  - *Positive controls, with the right scale and conditioning:*
    - air temperature → water temperature (+, 0–2 d);
    - precipitation → Q (+, 0–3 d);
    - Q → turbidity (+, 0 d);
    - Q → SpC (− at dilution sites);
    - shortwave → D_max (+, same day);
    - water temperature → nightly D (−, via ER).
  - *Negative controls:* any river variable → air temperature or precipitation (forbidden); downstream → upstream; time-reversed data; pairs from different basins with weather included.
  - **Report recall on positives and the FPR on negatives across all sites.** This is the headline validation figure.
- **Stage 3: discovery.**
  - Per site: Bagged-PCMCI+ (ParCorr primary, CMIknn or GPDC on a subsample) with `link_assumptions` encoding the forbidden edges, τ_max from physics (7–10 d for daily data), and BH-FDR.
  - Seasonal windows and RPCMCI for regimes.
  - Pooled across sites: J-PCMCI+ with StreamCat traits and groundwater class as context nodes.
- **Stage 4: replication.** A link is "robust" only if it meets all of these:
  - (i) bootstrap frequency ≥ 0.8 at a site;
  - (ii) present in ≥ 60 % of sites of the same class, with consistent sign and a lag within ±1 step;
  - (iii) survives seasonal and phase-randomized surrogates (Theiler et al. 1992 [V-meta]; Schreiber & Schmitz 2000 [V-meta]);
  - (iv) stable at hourly and daily resolution;
  - (v) stable across the alternative CI test.
  - The thresholds are illustrative; **pre-register them.**
- **Stage 5: natural experiments.** Storm pulses (event studies with matched control sites); dam releases (sites below reservoirs, using NLDI and NID); heatwaves; documented outfall or treatment-plant changes if any can be sourced. Use Bayesian structural time series (CausalImpact; Brodersen et al. 2015 [V-meta]) or synthetic control (Abadie 2021 [V-meta]).
- **Stage 6: effect sizes.** For robust links, estimate effects with tigramite CausalEffects (optimal adjustment) and report CIs. Climb the claim ladder in b.3 only as far as the evidence goes.
- **Compute.** ParCorr PCMCI+ on about 10 variables × 3,650 days × τ_max 10 takes seconds to minutes per site on CPU, so 48 cores can process the whole network per configuration. CMIknn permutation tests are orders of magnitude slower; run them on subsamples or on a few hundred sites overnight.
- **Outputs.** Per-link "evidence cards" (rung, sites, sign, lag, bootstrap frequency, surrogate p, natural-experiment result) instead of a "number of chains".

### f.5 Interpretability suite (fidelity first)

1. **A synthetic ground-truth attribution benchmark.** Train the same architecture on data simulated by the process head with known additive or separable structure (Mamalakis 2022 approach). Score IG, EG, DeepLIFT, KernelSHAP (independent and dependent variants) and ALE against the ground truth. Choose the primary method from the scores.
2. **Sanity checks.** Model-weight randomization must destroy attributions (Adebayo 2018). Run deletion and insertion curves on feature groups, and ROAR-style retraining for the top groups.
3. **Production attributions.** IG or EG with climatology baselines, aggregated by **feature group** and **lag**, per site class. Report stability across ensemble members (Spearman ρ of rankings), not single-model maps.
4. **Probing.** Ridge probes from LSTM cell states to independent GPP, ER and K₆₀₀ (Powell Center, NEON), baseflow fraction and groundwater class, on held-out sites. Add a **control task** for selectivity (Hewitt & Liang 2019 [V-meta]; Belinkov 2022 [V]) and a randomly initialized network as a floor (Lees 2022 approach).
5. **Concept bottleneck.** The process head itself. Report how much DO variance runs through each concept: solubility, GPP, ER, gas exchange.
6. **Symbolic regression.** Fit PySR to the learned ER(T), K₆₀₀(Q, slope) and GPP(light) relations, pooled by site class.
   - Success means rediscovering known forms: Arrhenius at about 0.65 eV, power-law K₆₀₀–Q, saturating light curves.
   - Class-specific departures that replicate across held-out sites would be genuine insights.
   - Guard against overfitting with Pareto-front selection on held-out sites.
7. **Global sensitivity.** Run Sobol' (Saltelli designs, SALib) over the scenario forcings (ΔT, P or Q scaling, SW scaling) per site, with outputs of mean DO, nightly DO minimum and hypoxia hours. Use Shapley effects where scenario inputs are correlated. Map which lever dominates where.
8. **Rules.** No attention-as-explanation. No marginal permutation importance on correlated forcings. Every interpretive claim states whether it is "true to the model" or "true to the data".

### f.6 Integration with other SENTINEL models

- Features derived from other modality models (satellite, microbial, fusion embeddings) may enter the twin **only as state or conditioning inputs**, and only after an ablation shows they add hold-out skill. They are never scenario levers (§f.3).
- Derived ecological features (c.8) are first-class outputs of the platform. They feed the twin (as static traits) and the insight analyses (E1–E6).

### f.7 Milestones (suggested)

| Milestone | Deliverable | Gate |
|---|---|---|
| M1 | Loaders (Water Data API v1, NEON, gridMET, StreamCat, NLDI) with QA and provenance; data inventory | Unit tests for masks and units; Appendix-A-style inventory regenerated |
| M2 | Baselines plus a temperature-only LSTM | Beats all baselines under PUB |
| M3 | DO process head | Skill gate, metamorphic tests, metabolism agreement |
| M4 | Synthetic causal benchmark plus controls | Recall ≥ 0.8 and FPR ≤ 0.05 on synthetic data; controls reported on real data |
| M5 | E1 and E2 analyses | Pre-registered validations pass or are reported as negative |
| M6 | NEON Challenge submissions (rolling) | CRPS against climatology and persistence |
| M7 | Interpretability suite and platform | Fidelity benchmark reported first |

---

## (g) Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| USGS API transition (WaterServices off in early 2027; v0 redirects mid-2027) | High | Loaders break | Build only on v1 with dataretrieval ≥ 1.4; cache raw pulls with query metadata |
| WQP legacy profiles silently missing recent USGS data | High | Biased nutrient traits | Use WQX 3.0 profiles; assert date coverage |
| Sensor QA (fouling, drift, ice, seasonal deployments) | High | Spurious trends and links | Approved data only for science; qualifier masks; drift sensitivity checks |
| Sentinel or missing codes leak into statistics (repeat of the −999999 bug) | Medium | Meaningless results | Physical-range assertions in loaders and tests |
| Counterfactual extrapolation (physically wrong responses) | High | False headline claims | Metamorphic tests, DSST, extrapolation flags, multi-site training, process head |
| Unidentifiable latents (GPP, ER, K₆₀₀ equifinality) | Medium | Wrong ecological interpretation | K–Q priors; comparison with independent estimates; report identifiability diagnostics |
| Over-claiming causality from PCMCI, CCM or SHAP | High | Integrity risk (repeat of "375 chains") | Claim ladder; controls; replication; natural experiments; evidence cards |
| Confounding by shared regional weather (E2) | High | False propagation signals | Weather variables in the graph; time-reversal and unconnected-pair nulls |
| Novelty already published | Medium | Overstated novelty | Run the Appendix B searches before writing; frame findings as extension or replication where needed |
| Selection bias (sensor sites are not random) | High | Limited generality | State the population; weight by stream-order or region; avoid national extrapolation |
| Leakage (normalization, overlapping windows, sites in train and test) | Medium | Inflated skill | Frozen fold files; assertions; leakage tests in CI |
| Licensing (δMG and δHBV are non-commercial) | Low | Reuse limits | Use NeuralHydrology (BSD) as the core; cite and isolate any δMG use |
| Compute limits (CMIknn, Sobol on hourly heads) | Low–Medium | Delays | Subsample; ParCorr primary; daily-resolution Sobol |
| Time and attention spread across too many goals | High | Unfinished pieces | Ship E1 and E2 plus the twin before starting E4–E8 |

---

## (h) References

Flags: **[V]** metadata plus quoted claims verified; **[V-arXiv]** verified through the arXiv API (title, authors, and abstract where quoted); **[V-meta]** or **[M]** metadata verified only; **[U]** unverified. DOI links have the form https://doi.org/&lt;DOI&gt;.

### Simulation, digital twins and hybrid modelling
1. Read, J.S., Jia, X., Willard, J., Appling, A.P., Zwart, J.A., Oliver, S.K., et al. (2019). Process-guided deep learning predictions of lake water temperature. *Water Resources Research* 55(11):9173–9190. doi:10.1029/2019WR024922 [V]
2. Read, J.S., Jia, X., Willard, J., Appling, A.P., Zwart, J.A., et al. (2019). Data release: Process-guided deep learning predictions of lake water temperature. USGS. doi:10.5066/P9AQPIVD [V-meta, DataCite]
3. Jia, X., Willard, J., Karpatne, A., Read, J., Zwart, J., Steinbach, M., Kumar, V. (2019). Physics guided RNNs for modeling dynamical systems: A case study in simulating lake temperature profiles. *Proc. SIAM SDM 2019*, 558–566. doi:10.1137/1.9781611975673.63 [M]
4. Jia, X., Willard, J., Karpatne, A., Read, J.S., Zwart, J.A., Steinbach, M., Kumar, V. (2021). Physics-guided machine learning for scientific discovery: An application in simulating lake temperature profiles. *ACM/IMS Transactions on Data Science* 2(3):1–26. doi:10.1145/3447814 [V]
5. Daw, A., Karpatne, A., Watkins, W., Read, J., Kumar, V. (2017). Physics-guided Neural Networks (PGNN): An application in lake temperature modeling. arXiv:1710.11431. (The author list is as in the current arXiv version.) [V-arXiv]
6. Daw, A., Thomas, R.Q., Carey, C.C., Read, J.S., Appling, A.P., Karpatne, A. (2020). Physics-guided architecture (PGA) of neural networks for quantifying uncertainty in lake temperature modeling. *Proc. SIAM SDM 2020*, 532–540. doi:10.1137/1.9781611976236.60 [M]
7. Willard, J.D., Read, J.S., Appling, A.P., Oliver, S.K., Jia, X., Kumar, V. (2021). Predicting water temperature dynamics of unmonitored lakes with meta-transfer learning. *WRR* 57(7):e2021WR029579. doi:10.1029/2021WR029579 [V]
8. Willard, J., Jia, X., Xu, S., Steinbach, M., Kumar, V. (2023; online 2022). Integrating scientific knowledge with machine learning for engineering and environmental systems. *ACM Computing Surveys* 55(4):1–37. doi:10.1145/3514228 [V-meta]
9. Willard, J.D., Read, J.S., Topp, S., Hansen, G.J.A., Kumar, V. (2022). Daily surface temperatures for 185,549 lakes in the conterminous United States estimated using deep learning (1980–2020). *Limnology & Oceanography Letters* 7(4):287–301. doi:10.1002/lol2.10249 [V]
10. Ladwig, R., Daw, A., Albright, E.A., Buelo, C., Karpatne, A., Meyer, M.F., Neog, A., Hanson, P.C., Dugan, H.A. (2024). Modular compositional learning improves 1D hydrodynamic lake model performance by merging process-based modeling with deep learning. *JAMES* 16(1):e2023MS003953. doi:10.1029/2023MS003953 [V]
11. Ladwig, R., Hanson, P.C., Dugan, H.A., Carey, C.C., Zhang, Y., Shu, L., Duffy, C.J., Cobourn, K.M. (2021). Lake thermal structure drives interannual variability in summer anoxia dynamics in a eutrophic lake over 37 years. *HESS* 25(2):1009–1032. doi:10.5194/hess-25-1009-2021 [V]
12. Ladwig, R., Appling, A.P., Delany, A., Dugan, H.A., Gao, Q., Lottig, N., Stachelek, J., Hanson, P.C. (2022). Long-term change in metabolism phenology in north temperate lakes. *Limnology & Oceanography* 67(7):1502–1521. doi:10.1002/lno.12098 [V]
13. Hanson, P.C., Stillman, A.B., Jia, X., Karpatne, A., Dugan, H.A., Carey, C.C., et al. (2020). Predicting lake surface water phosphorus dynamics using process-guided machine learning. *Ecological Modelling* 430:109136. doi:10.1016/j.ecolmodel.2020.109136 [M]
14. Jia, X., Zwart, J., Sadler, J., Appling, A., Oliver, S., Markstrom, S., et al. (2021). Physics-guided recurrent graph model for predicting flow and temperature in river networks. *Proc. SIAM SDM 2021*, 612–620. doi:10.1137/1.9781611976700.69 [M]
15. Topp, S.N., Barclay, J., Diaz, J., Sun, A.Y., Jia, X., Lu, D., Sadler, J.M., Appling, A.P. (2023). Stream temperature prediction in a shifting environment: Explaining the influence of deep learning architecture. *WRR* 59(4):e2022WR033880. doi:10.1029/2022WR033880 [V]
16. Zwart, J.A., Oliver, S.K., Watkins, W.D., Sadler, J.M., Appling, A.P., Corson-Dosch, H.R., et al. (2023). Near-term forecasts of stream temperature using deep learning and data assimilation in support of management decisions. *JAWRA* 59(2):317–337. doi:10.1111/1752-1688.13093 [V]
17. Chen, S., Appling, A., Oliver, S., Corson-Dosch, H., Read, J., Sadler, J., Zwart, J., Jia, X. (2021). Heterogeneous stream-reservoir graph networks with data assimilation. *IEEE ICDM 2021*, 1024–1029. doi:10.1109/ICDM51629.2021.00117 [M]
18. Tsai, W.-P., Feng, D., Pan, M., Beck, H., Lawson, K., Yang, Y., Liu, J., Shen, C. (2021). From calibration to parameter learning: Harnessing the scaling effects of big data in geoscientific modeling. *Nature Communications* 12:5988. doi:10.1038/s41467-021-26107-z [V]
19. Feng, D., Liu, J., Lawson, K., Shen, C. (2022). Differentiable, learnable, regionalized process-based models with multiphysical outputs can approach state-of-the-art hydrologic prediction accuracy. *WRR* 58(10):e2022WR032404. doi:10.1029/2022WR032404 [V]
20. Feng, D., Beck, H., Lawson, K., Shen, C. (2023). The suitability of differentiable, physics-informed machine learning hydrologic models for ungauged regions and climate change impact assessment. *HESS* 27(12):2357–2373. doi:10.5194/hess-27-2357-2023 [V]
21. Shen, C., Appling, A.P., Gentine, P., Bandai, T., Gupta, H., Tartakovsky, A., et al. (2023). Differentiable modelling to unify machine learning and physical models for geosciences. *Nature Reviews Earth & Environment* 4(8):552–567. doi:10.1038/s43017-023-00450-9 [V]
22. Rahmani, F., Lawson, K., Ouyang, W., Appling, A., Oliver, S., Shen, C. (2021). Exploring the exceptional performance of a deep learning stream temperature model and the value of streamflow data. *Environmental Research Letters* 16:024025. doi:10.1088/1748-9326/abd501 [M]
23. Rahmani, F., Shen, C., Oliver, S., Lawson, K., Appling, A. (2021). Deep learning approaches for improving prediction of daily stream temperature in data-scarce, unmonitored, and dammed basins. *Hydrological Processes* 35(11):e14400. doi:10.1002/hyp.14400 [V]
24. Rahmani, F., Appling, A., Feng, D., Lawson, K., Shen, C. (2023). Identifying structural priors in a hybrid differentiable model for stream water temperature modeling. *WRR* 59(12):e2023WR034420. doi:10.1029/2023WR034420 [V]
25. Bennett, A., Nijssen, B. (2021). Deep learned process parameterizations provide better representations of turbulent heat fluxes in hydrologic models. *WRR* 57(5):e2020WR029328. doi:10.1029/2020WR029328 [V]
26. Kratzert, F., Gauch, M., Nearing, G., Klotz, D. (2022). NeuralHydrology — A Python library for deep learning research in hydrology. *JOSS* 7(71):4050. doi:10.21105/joss.04050 [M]
27. Hoedt, P.-J., Kratzert, F., Klotz, D., Halmich, C., Holzleitner, M., Nearing, G., Hochreiter, S., Klambauer, G. (2021). MC-LSTM: Mass-conserving LSTM. *ICML 2021* (PMLR 139). arXiv:2101.05186 [V-arXiv]
28. Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A.K., Hochreiter, S., Nearing, G.S. (2019). Toward improved predictions in ungauged basins: Exploiting the power of machine learning. *WRR* 55(12):11344–11354. doi:10.1029/2019WR026065 [V]
29. Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G. (2019). Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets. *HESS* 23(12):5089–5110. doi:10.5194/hess-23-5089-2019 [V]
30. Kratzert, F., Gauch, M., Klotz, D., Nearing, G. (2024). HESS Opinions: Never train a Long Short-Term Memory (LSTM) network on a single basin. *HESS* 28(17):4187–4201. doi:10.5194/hess-28-4187-2024 [V]
31. Gauch, M., Kratzert, F., Klotz, D., Nearing, G., Lin, J., Hochreiter, S. (2021). Rainfall–runoff prediction at multiple timescales with a single LSTM network. *HESS* 25(4):2045–2062. doi:10.5194/hess-25-2045-2021 [M]
32. Gauch, M., Mai, J., Lin, J. (2021). The proper care and feeding of CAMELS: How limited training data affects streamflow prediction. *Environmental Modelling & Software* 135:104926. doi:10.1016/j.envsoft.2020.104926 [M]
33. Nearing, G.S., Kratzert, F., Sampson, A.K., Pelissier, C.S., Klotz, D., Frame, J.M., Prieto, C., Gupta, H.V. (2021). What role does hydrological science play in the age of machine learning? *WRR* 57(3):e2020WR028091. doi:10.1029/2020WR028091 [V]
34. Nearing, G., Klotz, D., Frame, J.M., Gauch, M., Gilon, O., Kratzert, F., et al. (2022). Technical note: Data assimilation and autoregression for using near-real-time streamflow observations in long short-term memory networks. *HESS* 26(21):5493–5513. doi:10.5194/hess-26-5493-2022 [M]
35. Zhi, W., Feng, D., Tsai, W.-P., Sterle, G., Harpold, A., Shen, C., Li, L. (2021). From hydrometeorology to river water quality: Can a deep learning model predict dissolved oxygen at the continental scale? *Environmental Science & Technology* 55(4):2357–2368. doi:10.1021/acs.est.0c06783 [V]
36. Zhi, W., Klingler, C., Liu, J., Li, L. (2023). Widespread deoxygenation in warming rivers. *Nature Climate Change* 13(10):1105–1113. doi:10.1038/s41558-023-01793-3 [V, publisher page]
37. Zhi, W., Appling, A.P., Golden, H.E., Podgorski, J., Li, L. (2024). Deep learning for water quality. *Nature Water* 2(3):228–241. doi:10.1038/s44221-024-00202-z [V, partial page]
38. Li, L., Knapp, J.L.A., Lintern, A., Ng, G.-H.C., Perdrial, J., Sullivan, P.L., Zhi, W. (2024). River water quality shaped by land–river connectivity in a changing climate. *Nature Climate Change* 14(3):225–237. doi:10.1038/s41558-023-01923-x [M]
39. Varadharajan, C., Appling, A.P., Arora, B., Christianson, D.S., Hendrix, V.C., Kumar, V., et al. (2022). Can machine learning accelerate process understanding and decision-relevant predictions of river water quality? *Hydrological Processes* 36(4):e14565. doi:10.1002/hyp.14565 [V]
40. Sterle, G., Perdrial, J., Kincaid, D.W., Underwood, K.L., Rizzo, D.M., Haq, I.U., et al. (2024). CAMELS-Chem: augmenting CAMELS with atmospheric and stream water chemistry data. *HESS* 28(3):611–630. doi:10.5194/hess-28-611-2024 [V]
41. Saha, G.K., Rahmani, F., Shen, C., Li, L., Cibin, R. (2023). A deep learning-based novel approach to generate continuous daily stream nitrate concentration for nitrate data-sparse watersheds. *Science of the Total Environment* 878:162930. doi:10.1016/j.scitotenv.2023.162930 [M]
42. Reichstein, M., Camps-Valls, G., Stevens, B., Jung, M., Denzler, J., Carvalhais, N., Prabhat (2019). Deep learning and process understanding for data-driven Earth system science. *Nature* 566:195–204. doi:10.1038/s41586-019-0912-1 [M]
43. Karpatne, A., Atluri, G., Faghmous, J.H., Steinbach, M., Banerjee, A., Ganguly, A., et al. (2017). Theory-guided data science: A new paradigm for scientific discovery from data. *IEEE TKDE* 29(10):2318–2331. doi:10.1109/TKDE.2017.2720168 [M]
44. Raissi, M., Perdikaris, P., Karniadakis, G.E. (2019). Physics-informed neural networks. *Journal of Computational Physics* 378:686–707. doi:10.1016/j.jcp.2018.10.045 [M]
45. Jiang, S., Zheng, Y., Solomatine, D. (2020). Improving AI system awareness of geoscience knowledge: Symbiotic integration of physical approaches and deep learning. *GRL* 47(13):e2020GL088229. doi:10.1029/2020GL088229 [M]
46. Chen, R.T.Q., Rubanova, Y., Bettencourt, J., Duvenaud, D. (2018). Neural ordinary differential equations. *NeurIPS 2018*. arXiv:1806.07366 [V-arXiv]
47. Rackauckas, C., Ma, Y., Martensen, J., Warner, C., Zubov, K., Supekar, R., et al. (2020). Universal differential equations for scientific machine learning. arXiv:2001.04385 [V-arXiv]
48. Höge, M., Scheidegger, A., Baity-Jesi, M., Albert, C., Fenicia, F. (2022). Improving hydrologic models for predictions and process understanding using neural ODEs. *HESS* 26(19):5085–5102. doi:10.5194/hess-26-5085-2022 [V]
49. Kochkov, D., Yuval, J., Langmore, I., Norgaard, P., Smith, J., Mooers, G., et al. (2024). Neural general circulation models for weather and climate. *Nature* 632:1060–1066. doi:10.1038/s41586-024-07744-y [V]
50. Wi, S., Steinschneider, S. (2022). Assessing the physical realism of deep learning hydrologic model projections under climate change. *WRR* 58(9):e2022WR032123. doi:10.1029/2022WR032123 [V]
51. Wi, S., Steinschneider, S. (2024). On the need for physical constraints in deep learning rainfall–runoff projections under climate change: a sensitivity analysis to warming and shifts in potential evapotranspiration. *HESS* 28(3):479–503. doi:10.5194/hess-28-479-2024 [V]
52. Reichert, P., Ma, K., Höge, M., Fenicia, F., Baity-Jesi, M., Feng, D., Shen, C. (2024). Metamorphic testing of machine learning and conceptual hydrologic models. *HESS* 28(11):2505–2529. doi:10.5194/hess-28-2505-2024 [V]
53. Frame, J.M., Kratzert, F., Klotz, D., Gauch, M., Shalev, G., Gilon, O., Qualls, L.M., Gupta, H.V., Nearing, G.S. (2022). Deep learning rainfall–runoff predictions of extreme events. *HESS* 26(13):3377–3392. doi:10.5194/hess-26-3377-2022 [V]
54. Klemeš, V. (1986). Operational testing of hydrological simulation models. *Hydrological Sciences Journal* 31(1):13–24. doi:10.1080/02626668609491024 [V-meta]
55. Hrachowitz, M., Savenije, H.H.G., Blöschl, G., McDonnell, J.J., Sivapalan, M., Pomeroy, J.W., et al. (2013). A decade of Predictions in Ungauged Basins (PUB)—a review. *Hydrological Sciences Journal* 58(6):1198–1255. doi:10.1080/02626667.2013.803183 [M]
56. Knoben, W.J.M., Freer, J.E., Woods, R.A. (2019). Technical note: Inherent benchmark or not? Comparing Nash–Sutcliffe and Kling–Gupta efficiency scores. *HESS* 23(10):4323–4331. doi:10.5194/hess-23-4323-2019 [V]
57. Gupta, H.V., Kling, H., Yilmaz, K.K., Martinez, G.F. (2009). Decomposition of the mean squared error and NSE performance criteria. *Journal of Hydrology* 377(1–2):80–91. doi:10.1016/j.jhydrol.2009.08.003 [M]
58. Schaefli, B., Gupta, H.V. (2007). Do Nash values have value? *Hydrological Processes* 21(15):2075–2080. doi:10.1002/hyp.6825 [M]
59. Pappenberger, F., Ramos, M.H., Cloke, H.L., Wetterhall, F., Alfieri, L., Bogner, K., Mueller, A., Salamon, P. (2015). How do I know if my forecasts are better? Using benchmarks in hydrological ensemble prediction. *Journal of Hydrology* 522:697–713. doi:10.1016/j.jhydrol.2015.01.024 [M]
60. Kirchner, J.W. (2006). Getting the right answers for the right reasons. *WRR* 42(3):W03S04. doi:10.1029/2005WR004362 [M]
61. Gupta, H.V., Wagener, T., Liu, Y. (2008). Reconciling theory with observations: elements of a diagnostic approach to model evaluation. *Hydrological Processes* 22(18):3802–3813. doi:10.1002/hyp.6989 [M]
62. Beven, K. (2006). A manifesto for the equifinality thesis. *Journal of Hydrology* 320(1–2):18–36. doi:10.1016/j.jhydrol.2005.07.007 [M]
63. Dietze, M.C., Fox, A., Beck-Johnson, L.M., Betancourt, J.L., Hooten, M.B., Jarnevich, C.S., et al. (2018). Iterative near-term ecological forecasting: Needs, opportunities, and challenges. *PNAS* 115(7):1424–1432. doi:10.1073/pnas.1710231115 [V]
64. Thomas, R.Q., Boettiger, C., Carey, C.C., Dietze, M.C., Johnson, L.R., Kenney, M.A., et al. (2023). The NEON Ecological Forecasting Challenge. *Frontiers in Ecology and the Environment* 21(3):112–113. doi:10.1002/fee.2616 [M]. Challenge site (active 2026-10-21; CRPS scoring; climatology and persistenceRW baselines): https://projects.ecoforecast.org/neon4cast-ci/ [V]
65. Klotz, D., Kratzert, F., Gauch, M., Keefe Sampson, A., Brandstetter, J., Klambauer, G., Hochreiter, S., Nearing, G. (2022). Uncertainty estimation with deep learning for rainfall–runoff modeling. *HESS* 26(6):1673–1693. doi:10.5194/hess-26-1673-2022 [V]
66. Lakshminarayanan, B., Pritzel, A., Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS 2017*. arXiv:1612.01474 [V-arXiv]
67. Gal, Y., Ghahramani, Z. (2016). Dropout as a Bayesian approximation. *ICML 2016*. arXiv:1506.02142 [V-arXiv]
68. Gibbs, I., Candès, E. (2021). Adaptive conformal inference under distribution shift. *NeurIPS 2021*. arXiv:2106.00170 [V-arXiv]
69. Xu, C., Xie, Y. (2021). Conformal prediction interval for dynamic time-series. *ICML 2021* (PMLR 139); journal version: Conformal prediction for time series, *IEEE TPAMI* 45(10):11575–11587 (2023), doi:10.1109/TPAMI.2023.3272339; arXiv:2010.09107 [V-arXiv / V-meta]

### Process components (DO, metabolism, gas exchange)
70. Benson, B.B., Krause, D. (1984). The concentration and isotopic fractionation of oxygen dissolved in freshwater and seawater in equilibrium with the atmosphere. *Limnology & Oceanography* 29(3):620–632. doi:10.4319/lo.1984.29.3.0620 [V-meta]
71. Garcia, H.E., Gordon, L.I. (1992). Oxygen solubility in seawater: Better fitting equations. *Limnology & Oceanography* 37(6):1307–1312. doi:10.4319/lo.1992.37.6.1307 [V-meta]
72. USGS DOTABLES (dissolved-oxygen solubility tables): https://water.usgs.gov/water-resources/software/DOTABLES/ [page resolves]
73. Odum, H.T. (1956). Primary production in flowing waters. *Limnology & Oceanography* 1(2):102–117. doi:10.4319/lo.1956.1.2.0102 [V-meta]
74. Appling, A.P., Hall, R.O., Yackulic, C.B., Arroita, M. (2018). Overcoming equifinality: Leveraging long time series for stream metabolism estimation. *JGR Biogeosciences* 123(2):624–645. doi:10.1002/2017JG004140 [V]
75. Appling, A.P., Read, J.S., Winslow, L.A., Arroita, M., Bernhardt, E.S., Griffiths, N.A., et al. (2018). The metabolic regimes of 356 rivers in the United States. *Scientific Data* 5:180292. doi:10.1038/sdata.2018.292 [V]
76. USGS (2018). Metabolism estimates for 356 U.S. rivers (2007–2017). Data release, ScienceBase. doi:10.5066/F70864KX (creators include Appling, Read, Hall, Arroita and Stanley) [V-meta, DataCite]
77. Hall, R.O., Hotchkiss, E.R. (2017). Stream metabolism. In *Methods in Stream Ecology* (3rd ed.), Vol. 2, 219–233. doi:10.1016/B978-0-12-813047-6.00012-7 [M]
78. Van de Bogert, M.C., Carpenter, S.R., Cole, J.J., Pace, M.L. (2007). Assessing pelagic and benthic metabolism using free water measurements. *L&O: Methods* 5:145–155. doi:10.4319/lom.2007.5.145 [M]
79. Holtgrieve, G.W., Schindler, D.E., Branch, T.A., A'mar, Z.T. (2010). Simultaneous quantification of aquatic ecosystem metabolism and reaeration using a Bayesian statistical model of oxygen dynamics. *Limnology & Oceanography* 55(3):1047–1063. doi:10.4319/lo.2010.55.3.1047 [M]
80. Winslow, L.A., Zwart, J.A., Batt, R.D., Dugan, H.A., Woolway, R.I., Corman, J.R., Hanson, P.C., Read, J.S. (2016). LakeMetabolizer: an R package for estimating lake metabolism from free-water oxygen using diverse statistical models. *Inland Waters* 6(4):622–636. doi:10.1080/IW-6.4.883 [M]
81. Raymond, P.A., Zappa, C.J., Butman, D., Bott, T.L., Potter, J., Mulholland, P., et al. (2012). Scaling the gas transfer velocity and hydraulic geometry in streams and small rivers. *L&O: Fluids and Environments* 2(1):41–53. doi:10.1215/21573689-1597669 [V]
82. Ulseth, A.J., Hall, R.O., Boix Canadell, M., Madinger, H.L., Niayifar, A., Battin, T.J. (2019). Distinct air–water gas exchange regimes in low- and high-energy streams. *Nature Geoscience* 12:259–263. doi:10.1038/s41561-019-0324-8 [M]
83. Hall, R.O., Ulseth, A.J. (2020). Gas exchange in streams and rivers. *WIREs Water* 7(1):e1391. doi:10.1002/wat2.1391 [M]
84. Wanninkhof, R. (1992). Relationship between wind speed and gas exchange over the ocean. *JGR Oceans* 97(C5):7373–7382. doi:10.1029/92JC00188 [M] (source of the Schmidt-number polynomials)
85. Marzolf, N.S., Slaughter, W.M., Vlah, M.J., Rhea, S., DelVecchia, A.G., Bernhardt, E.S. (2025). Ecosystem metabolism estimates from the National Ecological Observatory Network (NEON) stream and river sites. *Scientific Data* 12:478. doi:10.1038/s41597-025-04710-9 [V]
86. Streeter, H.W., Phelps, E.B. (1925). A study of the pollution and natural purification of the Ohio River. US Public Health Service, Public Health Bulletin 146. [U: details not re-checked]

### Causal discovery
87. Runge, J., Nowack, P., Kretschmer, M., Flaxman, S., Sejdinovic, D. (2019). Detecting and quantifying causal associations in large nonlinear time series datasets. *Science Advances* 5(11):eaau4996. doi:10.1126/sciadv.aau4996 [V]
88. Runge, J. (2020). Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series datasets. *UAI 2020* (PMLR 124). arXiv:2003.03685 [V-arXiv]
89. Gerhardus, A., Runge, J. (2020). High-recall causal discovery for autocorrelated time series with latent confounders. *NeurIPS 33*. arXiv:2007.01884 [V-arXiv]
90. Günther, W., Ninad, U., Runge, J. (2023). Causal discovery for time series from multiple datasets with latent contexts. *UAI 2023* (PMLR 216). arXiv:2306.12896 [V-arXiv]
91. Saggioro, E., de Wiljes, J., Kretschmer, M., Runge, J. (2020). Reconstructing regime-dependent causal relationships from observational time series. *Chaos* 30(11):113115. doi:10.1063/5.0020538 [V]
92. Debeire, K., Runge, J., Gerhardus, A., Eyring, V. (2024). Bootstrap aggregation and confidence measures to improve time series causal discovery. *CLeaR 2024*, PMLR 236:979–1007. arXiv:2306.08946 [V-arXiv]
93. Runge, J. (2018). Causal network reconstruction from time series: From theoretical assumptions to practical estimation. *Chaos* 28(7):075310. doi:10.1063/1.5025050 [V]
94. Runge, J. (2018). Conditional independence testing based on a nearest-neighbor estimator of conditional mutual information. *AISTATS 2018* (PMLR 84). arXiv:1709.01447 [V-arXiv]
95. Runge, J., Bathiany, S., Bollt, E., Camps-Valls, G., Coumou, D., Deyle, E., et al. (2019). Inferring causation from time series in Earth system sciences. *Nature Communications* 10:2553. doi:10.1038/s41467-019-10105-3 [V]; CauseMe platform: https://causeme.uv.es/ (causeme.net redirects; beta) [V]
96. Runge, J., Gerhardus, A., Varando, G., Eyring, V., Camps-Valls, G. (2023). Causal inference for time series. *Nature Reviews Earth & Environment* 4(7):487–505. doi:10.1038/s43017-023-00431-y [M]
97. Runge, J., Petoukhov, V., Donges, J.F., Hlinka, J., Jajcay, N., Vejmelka, M., et al. (2015). Identifying causal gateways and mediators in complex spatio-temporal systems. *Nature Communications* 6:8502. doi:10.1038/ncomms9502 [V]
98. Runge, J. (2015). Quantifying information transfer and mediation along causal pathways in complex systems. *Physical Review E* 92(6):062829. doi:10.1103/PhysRevE.92.062829 [M]
99. Runge, J. (2021). Necessary and sufficient graphical conditions for optimal adjustment sets in causal graphical models with hidden variables. *NeurIPS 34*. arXiv:2102.10324 [V-arXiv]
100. Popescu, O.-I., Gerhardus, A., Runge, J. (2023). Non-parametric conditional independence testing for mixed continuous-categorical variables: A novel method and numerical evaluation. arXiv:2310.11132 [V-arXiv]
101. Massmann, A., Gentine, P., Runge, J. (2021). Causal inference for process understanding in Earth sciences. arXiv:2105.00912 [V-arXiv]
102. Camps-Valls, G., Gerhardus, A., Ninad, U., Varando, G., Martius, G., Balaguer-Ballester, E., et al. (2023). Discovering causal relations and equations from data. *Physics Reports* 1044:1–68. doi:10.1016/j.physrep.2023.10.005 [V]
103. Krich, C., Runge, J., Miralles, D.G., Migliavacca, M., Perez-Priego, O., El-Madany, T., Carrara, A., Mahecha, M.D. (2020). Estimating causal networks in biosphere–atmosphere interaction with the PCMCI approach. *Biogeosciences* 17(4):1033–1061. doi:10.5194/bg-17-1033-2020 [V]
104. Nowack, P., Runge, J., Eyring, V., Haigh, J.D. (2020). Causal networks for climate model evaluation and constrained projections. *Nature Communications* 11:1415. doi:10.1038/s41467-020-15195-y [V]
105. Kretschmer, M., Coumou, D., Donges, J.F., Runge, J. (2016). Using causal effect networks to analyze different Arctic drivers of midlatitude winter circulation. *Journal of Climate* 29(11):4069–4081. doi:10.1175/JCLI-D-15-0654.1 [M]
106. Ebert-Uphoff, I., Deng, Y. (2012). Causal discovery for climate research using graphical models. *Journal of Climate* 25(17):5648–5665. doi:10.1175/JCLI-D-11-00387.1 [M]
107. Tibau, X., Reimers, C., Gerhardus, A., Denzler, J., Eyring, V., Runge, J. (2022). A spatiotemporal stochastic climate model for benchmarking causal discovery methods for teleconnections. *Environmental Data Science* 1:e12. doi:10.1017/eds.2022.11 [V]
108. Sugihara, G., May, R., Ye, H., Hsieh, C., Deyle, E., Fogarty, M., Munch, S. (2012). Detecting causality in complex ecosystems. *Science* 338(6106):496–500. doi:10.1126/science.1227079 [V]
109. Ye, H., Deyle, E.R., Gilarranz, L.J., Sugihara, G. (2015). Distinguishing time-delayed causal interactions using convergent cross mapping. *Scientific Reports* 5:14750. doi:10.1038/srep14750 [V]
110. Chang, C.-W., Ushio, M., Hsieh, C. (2017). Empirical dynamic modeling for beginners. *Ecological Research* 32(6):785–796. doi:10.1007/s11284-017-1469-9 [V-meta]
111. Cobey, S., Baskerville, E.B. (2016). Limits to causal inference with state-space reconstruction for infectious disease. *PLoS ONE* 11(12):e0169050. doi:10.1371/journal.pone.0169050 [V]
112. Baskerville, E.B., Cobey, S. (2017). Does influenza drive absolute humidity? *PNAS* 114(12):E2270–E2271. doi:10.1073/pnas.1700369114 [M]
113. Deyle, E.R., Maher, M.C., Hernandez, R.D., Basu, S., Sugihara, G. (2016). Global environmental drivers of influenza. *PNAS* 113(46):13081–13086. doi:10.1073/pnas.1607747113 [V]
114. Mønster, D., Fusaroli, R., Tylén, K., Roepstorff, A., Sherson, J.F. (2017). Causal inference from noisy time-series data — Testing the Convergent Cross-Mapping algorithm in the presence of noise and external influence. *Future Generation Computer Systems* 73:52–62. doi:10.1016/j.future.2016.12.009 [V]
115. McCracken, J.M., Weigel, R.S. (2014). Convergent cross-mapping and pairwise asymmetric inference. *Physical Review E* 90(6):062903. doi:10.1103/PhysRevE.90.062903 [V]
116. Granger, C.W.J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica* 37(3):424–438. doi:10.2307/1912791 [M]
117. Schreiber, T. (2000). Measuring information transfer. *Physical Review Letters* 85(2):461–464. doi:10.1103/PhysRevLett.85.461 [M]
118. Ruddell, B.L., Kumar, P. (2009). Ecohydrologic process networks: 1. Identification. *WRR* 45(3):W03419. doi:10.1029/2008WR007279 [V]
119. Goodwell, A.E., Kumar, P. (2017). Temporal information partitioning: Characterizing synergy, uniqueness, and redundancy in interacting environmental variables. *WRR* 53(7):5920–5942. doi:10.1002/2016WR020216 [V]
120. Goodwell, A.E., Jiang, P., Ruddell, B.L., Kumar, P. (2020). Debates—Does information theory provide a new paradigm for Earth science? Causality, interaction, and feedback. *WRR* 56(2):e2019WR024940. doi:10.1029/2019WR024940 [M]
121. Jiang, P., Kumar, P. (2019). Using information flow for whole system understanding from component dynamics. *WRR* 55(11):8305–8329. doi:10.1029/2019WR025820 [M]
122. Tank, A., Covert, I., Foti, N., Shojaie, A., Fox, E.B. (2021/2022). Neural Granger causality. *IEEE TPAMI*. doi:10.1109/TPAMI.2021.3065601; arXiv:1802.05842 [V-meta]
123. Nauta, M., Bucur, D., Seifert, C. (2019). Causal discovery with attention-based convolutional neural networks. *Machine Learning and Knowledge Extraction* 1(1):312–340. doi:10.3390/make1010019 [V]
124. Pamfil, R., Sriwattanaworachai, N., Desai, S., Pilgerstorfer, P., Beaumont, P., Georgatzis, K., Aragam, B. (2020). DYNOTEARS: Structure learning from time-series data. *AISTATS 2020*. arXiv:2002.00498 [V-arXiv]
125. Löwe, S., Madras, D., Zemel, R., Welling, M. (2022). Amortized causal discovery: Learning to infer causal graphs from time-series data. *CLeaR 2022*. arXiv:2006.10833 [V-arXiv]
126. Gong, W., Jennings, J., Zhang, C., Pawlowski, N. (2023). Rhino: Deep causal temporal relationship learning with history-dependent noise. *ICLR 2023*. arXiv:2210.14706 [V-arXiv]
127. Cheng, Y., Yang, R., Xiao, T., Li, Z., Suo, J., He, K., Dai, Q. (2023). CUTS: Neural causal discovery from irregular time-series data. *ICLR 2023*. arXiv:2302.07458 [V-arXiv]
128. Assaad, C.K., Devijver, E., Gaussier, E. (2022). Survey and evaluation of causal discovery methods for time series. *JAIR* 73:767–819. doi:10.1613/jair.1.13428 [V]
129. Lawrence, A.R., Kaiser, M., Sampaio, R., Sipos, M. (2021). Data generating process to evaluate causal discovery techniques for time series data. arXiv:2104.08043 (NeurIPS 2020 CDML workshop) [V-arXiv]
130. Gong, M., Zhang, K., Schölkopf, B., Tao, D., Geiger, P. (2015). Discovering temporal causal relations from subsampled data. *ICML 2015* (PMLR 37). [U: not resolvable through Crossref/arXiv in this session]
131. Ombadi, M., Nguyen, P., Sorooshian, S., Hsu, K. (2020). Evaluation of methods for causal discovery in hydrometeorological systems. *WRR* 56(7):e2020WR027251. doi:10.1029/2020WR027251 [V]
132. Delforge, D., de Viron, O., Vanclooster, M., Van Camp, M., Watlet, A. (2022). Detecting hydrological connectivity using causal inference from time series: synthetic and real karstic case studies. *HESS* 26(8):2181–2199. doi:10.5194/hess-26-2181-2022 [V]
133. Sendrowski, A., Passalacqua, P. (2017). Process connectivity in a naturally prograding river delta. *WRR* 53(3):1841–1863. doi:10.1002/2016WR019768 [V]
134. Bennett, A., Nijssen, B., Ou, G., Clark, M., Nearing, G. (2019). Quantifying process connectivity with transfer entropy in hydrologic models. *WRR* 55(6):4613–4629. doi:10.1029/2018WR024555 [V]
135. McGowan, J.A., Deyle, E.R., Ye, H., Carter, M.L., Perretti, C.T., Seger, K.D., de Verneil, A., Sugihara, G. (2017). Predicting coastal algal blooms in southern California. *Ecology* 98(5):1419–1433. doi:10.1002/ecy.1804 [V]
136. Ushio, M., Hsieh, C., Masuda, R., Deyle, E.R., Ye, H., Chang, C.-W., Sugihara, G., Kondoh, M. (2018). Fluctuating interaction network and time-varying stability of a natural fish community. *Nature* 554:360–363. doi:10.1038/nature25504 [M]
137. Deyle, E.R., Fogarty, M., Hsieh, C., Kaufman, L., MacCall, A.D., Munch, S.B., et al. (2013). Predicting climate effects on Pacific sardine. *PNAS* 110(16):6430–6435. doi:10.1073/pnas.1215506110 [M]
138. Benjamini, Y., Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS-B* 57(1):289–300. doi:10.1111/j.2517-6161.1995.tb02031.x [M]
139. Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., Farmer, J.D. (1992). Testing for nonlinearity in time series: the method of surrogate data. *Physica D* 58(1–4):77–94. doi:10.1016/0167-2789(92)90102-S [M]
140. Schreiber, T., Schmitz, A. (2000). Surrogate time series. *Physica D* 142(3–4):346–382. doi:10.1016/S0167-2789(00)00043-9 [M]
141. Brodersen, K.H., Gallusser, F., Koehler, J., Remy, N., Scott, S.L. (2015). Inferring causal impact using Bayesian structural time-series models. *Annals of Applied Statistics* 9(1):247–274. doi:10.1214/14-AOAS788 [M]
142. Abadie, A. (2021). Using synthetic controls: Feasibility, data requirements, and methodological aspects. *Journal of Economic Literature* 59(2):391–425. doi:10.1257/jel.20191450 [M]
143. Hall, R.O., Yackulic, C.B., Kennedy, T.A., Yard, M.D., Rosi-Marshall, E.J., Voichick, N., Behn, K.E. (2015). Turbidity, light, temperature, and hydropeaking control primary productivity in the Colorado River, Grand Canyon. *Limnology & Oceanography* 60(2):512–526. doi:10.1002/lno.10031 [M] (an example of dam operations as a natural experiment)
144. Kennedy, T.A., Muehlbauer, J.D., Yackulic, C.B., Lytle, D.A., Miller, S.W., Dibble, K.L., et al. (2016). Flow management for hydropower extirpates aquatic insects, undermining river food webs. *BioScience* 66(7):561–575. doi:10.1093/biosci/biw059 [M]
145. tigramite 5.2.10.1 (PyPI, 2026-01-14), GPL-3.0: https://github.com/jakobrunge/tigramite [V: README and source checked]

### Ecological interactions and thresholds
146. Bernhardt, E.S., Heffernan, J.B., Grimm, N.B., Stanley, E.H., Harvey, J.W., Arroita, M., et al. (2018). The metabolic regimes of flowing waters. *Limnology & Oceanography* 63(S1):S99–S118. doi:10.1002/lno.10726 [V]
147. Bernhardt, E.S., Savoy, P., Vlah, M.J., Appling, A.P., Koenig, L.E., Hall, R.O., et al. (2022). Light and flow regimes regulate the metabolism of rivers. *PNAS* 119(8):e2121976119. doi:10.1073/pnas.2121976119 [V]
148. Savoy, P., Appling, A.P., Heffernan, J.B., Stets, E.G., Read, J.S., Harvey, J.W., Bernhardt, E.S. (2019). Metabolic rhythms in flowing waters: An approach for classifying river productivity regimes. *Limnology & Oceanography* 64(5):1835–1851. doi:10.1002/lno.11154 [V]
149. Savoy, P., Harvey, J.W. (2021). Predicting light regime controls on primary productivity across CONUS river networks. *GRL* 48(10):e2020GL092149. doi:10.1029/2020GL092149 [V]
150. Koenig, L.E., Helton, A.M., Savoy, P., Bertuzzo, E., Heffernan, J.B., Hall, R.O., Bernhardt, E.S. (2019). Emergent productivity regimes of river networks. *L&O Letters* 4(5):173–181. doi:10.1002/lol2.10115 [V]
151. Blaszczak, J.R., Koenig, L.E., Mejia, F.H., Gómez-Gener, L., Dutton, C.L., Carter, A.M., et al. (2023). Extent, patterns, and drivers of hypoxia in the world's streams and rivers. *L&O Letters* 8(3):453–463. doi:10.1002/lol2.10297 [V]
152. Jane, S.F., Hansen, G.J.A., Kraemer, B.M., Leavitt, P.R., Mincer, J.L., North, R.L., et al. (2021). Widespread deoxygenation of temperate lakes. *Nature* 594:66–70. doi:10.1038/s41586-021-03550-y [V, publisher page]
153. Yvon-Durocher, G., Caffrey, J.M., Cescatti, A., Dossena, M., del Giorgio, P., Gasol, J.M., et al. (2012). Reconciling the temperature dependence of respiration across timescales and ecosystem types. *Nature* 487:472–476. doi:10.1038/nature11205 [V, publisher page]
154. Song, C., Dodds, W.K., Rüegg, J., Argerich, A., Baker, C.L., Bowden, W.B., et al. (2018). Continental-scale decrease in net primary productivity in streams due to climate warming. *Nature Geoscience* 11:415–420. doi:10.1038/s41561-018-0125-5 [V, publisher page]
155. Demars, B.O.L., Russell Manson, J., Ólafsson, J.S., Gíslason, G.M., Gudmundsdóttir, R., Woodward, G., et al. (2011). Temperature and the metabolic balance of streams. *Freshwater Biology* 56(6):1106–1121. doi:10.1111/j.1365-2427.2010.02554.x [M]
156. Diamond, J.S., Pinay, G., Bernal, S., Cohen, M.J., Lewis, D., Lupon, A., Zarnetske, J., Moatar, F. (2023). Light and hydrologic connectivity drive dissolved oxygen synchrony in stream networks. *Limnology & Oceanography* 68(2):322–335. doi:10.1002/lno.12271 [M]
157. Dodds, W.K., Smith, V.H. (2016). Nitrogen, phosphorus, and eutrophication in streams. *Inland Waters* 6(2):155–164. doi:10.5268/IW-6.2.909 [M]
158. Pellerin, B.A., Stauffer, B.A., Young, D.A., Sullivan, D.J., Bricker, S.B., Walbridge, M.R., Clyde, G.A., Shaw, D.M. (2016). Emerging tools for continuous nutrient monitoring networks. *JAWRA* 52(4):993–1008. doi:10.1111/1752-1688.12386 [V]
159. Williams, G.P. (1989). Sediment concentration versus water discharge during single hydrologic events in rivers. *Journal of Hydrology* 111(1–4):89–106. doi:10.1016/0022-1694(89)90254-0 [M]
160. Lloyd, C.E.M., Freer, J.E., Johnes, P.J., Collins, A.L. (2016). Using hysteresis analysis of high-resolution water quality monitoring data, including uncertainty, to infer controls on nutrient and sediment transfer in catchments. *Science of the Total Environment* 543:388–404. doi:10.1016/j.scitotenv.2015.11.028 [M]
161. Zuecco, G., Penna, D., Borga, M., van Meerveld, H.J. (2016). A versatile index to characterize hysteresis between hydrological variables at the runoff event timescale. *Hydrological Processes* 30(9):1449–1466. doi:10.1002/hyp.10681 [V]
162. Vaughan, M.C.H., Bowden, W.B., Shanley, J.B., Vermilyea, A., Sleeper, R., Gold, A.J., et al. (2017). High-frequency dissolved organic carbon and nitrate measurements reveal differences in storm hysteresis and loading in relation to land cover and seasonality. *WRR* 53(7):5345–5363. doi:10.1002/2017WR020491 [V]
163. Hamshaw, S.D., Dewoolkar, M.M., Schroth, A.W., Wemple, B.C., Rizzo, D.M. (2018). A new machine-learning approach for classifying hysteresis in suspended-sediment discharge relationships using high-frequency monitoring data. *WRR* 54(6):4040–4058. doi:10.1029/2017WR022238 [V]
164. Liu, W., Birgand, F., Tian, S., Chen, C. (2021). Event-scale hysteresis metrics to reveal processes and mechanisms controlling constituent export from watersheds: A review. *Water Research* 200:117254. doi:10.1016/j.watres.2021.117254 [M]
165. Godsey, S.E., Kirchner, J.W., Clow, D.W. (2009). Concentration–discharge relationships reflect chemostatic characteristics of US catchments. *Hydrological Processes* 23(13):1844–1864. doi:10.1002/hyp.7315 [V]
166. Musolff, A., Schmidt, C., Selle, B., Fleckenstein, J.H. (2015). Catchment controls on solute export. *Advances in Water Resources* 86:133–146. doi:10.1016/j.advwatres.2015.09.026 [M]
167. Moatar, F., Abbott, B.W., Minaudo, C., Curie, F., Pinay, G. (2017). Elemental properties, hydrology, and biology interact to shape concentration-discharge curves for carbon, nutrients, sediment, and major ions. *WRR* 53(2):1270–1287. doi:10.1002/2016WR019635 [V]
168. Rose, L.A., Karwan, D.L., Godsey, S.E. (2018). Concentration–discharge relationships describe solute and sediment mobilization, reaction, and transport at event and longer timescales. *Hydrological Processes* 32(18):2829–2844. doi:10.1002/hyp.13235 [V]
169. Vercruysse, K., Grabowski, R.C., Rickson, R.J. (2017). Suspended sediment transport dynamics in rivers: Multi-scale drivers of temporal variation. *Earth-Science Reviews* 166:38–52. doi:10.1016/j.earscirev.2016.12.016 [M]
170. Tang, W., Carey, S.K. (2017). HydRun: A MATLAB toolbox for rainfall–runoff analysis. *Hydrological Processes* 31(15):2670–2682. doi:10.1002/hyp.11185 [M]
171. Wasko, C., Guo, D. (2022). Understanding event runoff coefficient variability across Australia using the hydroEvents R package. *Hydrological Processes* 36(4):e14563. doi:10.1002/hyp.14563 [M]; hydroEvents 0.13.0 on CRAN [V]
172. Baker, M.E., King, R.S. (2010). A new method for detecting and interpreting biodiversity and ecological community thresholds. *Methods in Ecology and Evolution* 1(1):25–37. doi:10.1111/j.2041-210X.2009.00007.x [M]; TITAN2 2.4.4 on CRAN (2025-10-16) [V]
173. King, R.S., Baker, M.E. (2010). Considerations for analyzing ecological community thresholds in response to anthropogenic environmental gradients. *JNABS* 29(3):998–1008. doi:10.1899/09-144.1 [M]
174. Cuffney, T.F., Qian, S.S. (2013). A critique of the use of indicator-species scores for identifying thresholds in species responses. *Freshwater Science* 32(2):471–488. doi:10.1899/12-056.1 [M]
175. Baker, M.E., King, R.S. (2013). Of TITAN and straw men: an appeal for greater understanding of community data. *Freshwater Science* 32(2):489–506. doi:10.1899/12-142.1 [M]
176. Ellis, N., Smith, S.J., Pitcher, C.R. (2012). Gradient forests: calculating importance gradients on physical predictors. *Ecology* 93(1):156–168. doi:10.1890/11-0252.1 [V]
177. US EPA (2011). A Field-Based Aquatic Life Benchmark for Conductivity in Central Appalachian Streams (Final Report), EPA/600/R-10/023F. https://cfpub.epa.gov/ncea/risk/recordisplay.cfm?deid=233809 [V: page states the 300 µS/cm benchmark]
178. Cormier, S.M., Suter, G.W. (2013). A method for deriving water-quality benchmarks using field data. *Environmental Toxicology and Chemistry* 32(2):255–262. doi:10.1002/etc.2057 [V]
179. Cormier, S.M., Suter, G.W., Zheng, L. (2013). Derivation of a benchmark for freshwater ionic strength. *Environmental Toxicology and Chemistry* 32(2):263–271. doi:10.1002/etc.2064 [V]
180. Cormier, S.M., Suter, G.W., Zheng, L., Pond, G.J. (2013). Assessing causation of the extirpation of stream macroinvertebrates by a mixture of ions. *Environmental Toxicology and Chemistry* 32(2):277–287. doi:10.1002/etc.2059 [V]
181. Clements, W.H., Kotalik, C. (2016). Effects of major ions on natural benthic communities: an experimental assessment of the US EPA aquatic life benchmark for conductivity. *Freshwater Science* 35(1):126–138. doi:10.1086/685085 [M]
182. Olson, J.R., Cormier, S.M. (2019). Modeling spatial and temporal variation in natural background specific conductivity. *ES&T* 53(8):4316–4325. doi:10.1021/acs.est.8b06777 [M]
183. Kaushal, S.S., Likens, G.E., Pace, M.L., Utz, R.M., Haq, S., Gorman, J., Grese, M. (2018). Freshwater salinization syndrome on a continental scale. *PNAS* 115(4):E574–E583. doi:10.1073/pnas.1711234115 [V]
184. Kaushal, S.S., Likens, G.E., Pace, M.L., Reimer, J.E., Maas, C.M., Galella, J.G., et al. (2021). Freshwater salinization syndrome: from emerging global problem to managing risks. *Biogeochemistry* 154(2):255–292. doi:10.1007/s10533-021-00784-w [V]
185. Stets, E.G., Sprague, L.A., Oelsner, G.P., Johnson, H.M., Murphy, J.C., Ryberg, K., et al. (2020). Landscape drivers of dynamic change in water quality of U.S. rivers. *ES&T* 54(7):4336–4343. doi:10.1021/acs.est.9b05344 [V]
186. Cañedo-Argüelles, M., Kefford, B.J., Piscart, C., Prat, N., Schäfer, R.B., Schulz, C.-J. (2013). Salinisation of rivers: An urgent ecological issue. *Environmental Pollution* 173:157–167. doi:10.1016/j.envpol.2012.10.011 [M]
187. Hintz, W.D., Relyea, R.A. (2019). A review of the species, community, and ecosystem impacts of road salt salinisation in fresh waters. *Freshwater Biology* 64(6):1081–1097. doi:10.1111/fwb.13286 [V]
188. Friedman, J., Alm, E.J. (2012). Inferring correlation networks from genomic survey data. *PLoS Computational Biology* 8(9):e1002687. doi:10.1371/journal.pcbi.1002687 [V]
189. Watts, S.C., Ritchie, S.C., Inouye, M., Holt, K.E. (2019). FastSpar: rapid and scalable correlation estimation for compositional data. *Bioinformatics* 35(6):1064–1066. doi:10.1093/bioinformatics/bty734 [V]
190. Kurtz, Z.D., Müller, C.L., Miraldi, E.R., Littman, D.R., Blaser, M.J., Bonneau, R.A. (2015). Sparse and compositionally robust inference of microbial ecological networks. *PLoS Computational Biology* 11(5):e1004226. doi:10.1371/journal.pcbi.1004226 [V]
191. Tackmann, J., Matias Rodrigues, J.F., von Mering, C. (2019). Rapid inference of direct interactions in large-scale ecological networks from heterogeneous microbial sequencing data. *Cell Systems* 9(3):286–296.e8. doi:10.1016/j.cels.2019.08.002 [V]
192. Faust, K., Sathirapongsasuti, J.F., Izard, J., Segata, N., Gevers, D., Raes, J., Huttenhower, C. (2012). Microbial co-occurrence relationships in the human microbiome. *PLoS Computational Biology* 8(7):e1002606. doi:10.1371/journal.pcbi.1002606 [M]
193. Weiss, S., Van Treuren, W., Lozupone, C., Faust, K., Friedman, J., Deng, Y., et al. (2016). Correlation detection strategies in microbial data sets vary widely in sensitivity and precision. *ISME Journal* 10(7):1669–1681. doi:10.1038/ismej.2015.235 [V]
194. Hirano, H., Takemoto, K. (2019). Difficulty in inferring microbial community structure based on co-occurrence network approaches. *BMC Bioinformatics* 20:329. doi:10.1186/s12859-019-2915-1 [V]
195. Blanchet, F.G., Cazelles, K., Gravel, D. (2020). Co-occurrence is not evidence of ecological interactions. *Ecology Letters* 23(7):1050–1063. doi:10.1111/ele.13525 [V]
196. Faust, K. (2021). Open challenges for microbial network construction and analysis. *ISME Journal* 15(11):3111–3118. doi:10.1038/s41396-021-01027-4 [V]
197. Röttjers, L., Faust, K. (2018). From hairballs to hypotheses–biological insights from microbial networks. *FEMS Microbiology Reviews* 42(6):761–780. doi:10.1093/femsre/fuy030 [M]
198. Carr, A., Diener, C., Baliga, N.S., Gibbons, S.M. (2019). Use and abuse of correlation analyses in microbial ecology. *ISME Journal* 13(11):2647–2655. doi:10.1038/s41396-019-0459-z [V]
199. Gloor, G.B., Macklaim, J.M., Pawlowsky-Glahn, V., Egozcue, J.J. (2017). Microbiome datasets are compositional: And this is not optional. *Frontiers in Microbiology* 8:2224. doi:10.3389/fmicb.2017.02224 [M]
200. Eiler, A., Heinrich, F., Bertilsson, S. (2012). Coherent dynamics and association networks among lake bacterioplankton taxa. *ISME Journal* 6(2):330–342. doi:10.1038/ismej.2011.113 [M]
201. Scheffer, M., Bascompte, J., Brock, W.A., Brovkin, V., Carpenter, S.R., Dakos, V., et al. (2009). Early-warning signals for critical transitions. *Nature* 461:53–59. doi:10.1038/nature08227 [M]
202. Carpenter, S.R., Cole, J.J., Pace, M.L., Batt, R., Brock, W.A., Cline, T., et al. (2011). Early warnings of regime shifts: A whole-ecosystem experiment. *Science* 332(6033):1079–1082. doi:10.1126/science.1203672 [M]
203. Pace, M.L., Batt, R.D., Buelo, C.D., Carpenter, S.R., Cole, J.J., Kurtzweil, J.T., Wilkinson, G.M. (2017). Reversal of a cyanobacterial bloom in response to early warnings. *PNAS* 114(2):352–357. doi:10.1073/pnas.1612424114 [V]
204. Wilkinson, G.M., Carpenter, S.R., Cole, J.J., Pace, M.L., Batt, R.D., Buelo, C.D., Kurtzweil, J.T. (2018). Early warning signals precede cyanobacterial blooms in multiple whole-lake experiments. *Ecological Monographs* 88(2):188–203. doi:10.1002/ecm.1286 [V]
205. Dakos, V., Carpenter, S.R., Brock, W.A., Ellison, A.M., Guttal, V., Ives, A.R., et al. (2012). Methods for detecting early warnings of critical transitions in time series illustrated using simulated ecological data. *PLoS ONE* 7(7):e41010. doi:10.1371/journal.pone.0041010 [M]
206. Boettiger, C., Hastings, A. (2012). Quantifying limits to detection of early warning for critical transitions. *J. R. Soc. Interface* 9(75):2527–2539. doi:10.1098/rsif.2012.0125 [V]
207. Burthe, S.J., Henrys, P.A., Mackay, E.B., Spears, B.M., Campbell, R., Carvalho, L., et al. (2016). Do early warning indicators consistently predict nonlinear change in long-term ecological data? *Journal of Applied Ecology* 53(3):666–676. doi:10.1111/1365-2664.12519 [V]
208. Gsell, A.S., Scharfenberger, U., Özkundakci, D., Walters, A., Hansson, L.-A., Janssen, A.B.G., et al. (2016). Evaluating early-warning indicators of critical transitions in natural aquatic ecosystems. *PNAS* 113(50):E8089–E8095. doi:10.1073/pnas.1608242113 [M]
209. Bury, T.M., Sujith, R.I., Pavithran, I., Scheffer, M., Lenton, T.M., Anand, M., Bauch, C.T. (2021). Deep learning for early warning signals of tipping points. *PNAS* 118(39):e2106140118. doi:10.1073/pnas.2106140118 [V]
210. Kelleher, C., Wagener, T., Gooseff, M., McGlynn, B., McGuire, K., Marshall, L. (2012). Investigating controls on the thermal sensitivity of Pennsylvania streams. *Hydrological Processes* 26(5):771–785. doi:10.1002/hyp.8186 [V]
211. Hare, D.K., Helton, A.M., Johnson, Z.C., Lane, J.W., Briggs, M.A. (2021). Continental-scale analysis of shallow and deep groundwater contributions to streams. *Nature Communications* 12:1450. doi:10.1038/s41467-021-21651-0 [V]
212. Briggs, M.A., Johnson, Z.C., Snyder, C.D., Hitt, N.P., Kurylyk, B.L., Lautz, L., et al. (2018). Inferring watershed hydraulics and cold-water habitat persistence using multi-year air and stream temperature signals. *Science of the Total Environment* 636:1117–1127. doi:10.1016/j.scitotenv.2018.04.344 [M]
213. Tassone, S.J., Besterman, A.F., Buelo, C.D., Ha, D.T., Walter, J.A., Pace, M.L. (2023). Increasing heatwave frequency in streams and rivers of the United States. *L&O Letters* 8(2):295–304. doi:10.1002/lol2.10284 [V]
214. Pennekamp, F., Iles, A.C., Garland, J., Brennan, G., Brose, U., Gaedke, U., et al. (2019). The intrinsic predictability of ecological time series and its potential to guide forecasting. *Ecological Monographs* 89(2):e01359. doi:10.1002/ecm.1359 [V]
215. Petchey, O.L., Pontarp, M., Massie, T.M., Kéfi, S., Ozgul, A., Weilenmann, M., et al. (2015). The ecological forecast horizon, and examples of its uses and determinants. *Ecology Letters* 18(7):597–611. doi:10.1111/ele.12443 [V]
216. Lewis, A.S.L., Woelmer, W.M., Wander, H.L., Howard, D.W., Smith, J.W., McClure, R.P., et al. (2022). Increased adoption of best practices in ecological forecasting enables comparisons of forecastability. *Ecological Applications* 32(2):e2500. doi:10.1002/eap.2500 [V]
217. van Vliet, M.T.H., Thorslund, J., Strokal, M., Hofstra, N., Flörke, M., Ehalt Macedo, H., et al. (2023). Global river water quality under climate change and hydroclimatic extremes. *Nature Reviews Earth & Environment* 4(10):687–702. doi:10.1038/s43017-023-00472-3 [M]
218. Rust, A.J., Hogue, T.S., Saxe, S., McCray, J. (2018). Post-fire water-quality response in the western United States. *International Journal of Wildland Fire* 27(3):203–216. doi:10.1071/WF17115 [V]
219. Abbott, B.W., Gruau, G., Zarnetske, J.P., Moatar, F., Barbe, L., Thomas, Z., et al. (2018). Unexpected spatial stability of water chemistry in headwater stream networks. *Ecology Letters* 21(2):296–308. doi:10.1111/ele.12897 [M]

### Interpretability and sensitivity analysis
220. Kratzert, F., Herrnegger, M., Klotz, D., Hochreiter, S., Klambauer, G. (2019). NeuralHydrology – Interpreting LSTMs in hydrology. In Samek, W., et al. (eds.), *Explainable AI*, LNCS 11700, Springer, 347–362. doi:10.1007/978-3-030-28954-6_19; arXiv:1903.07903 [V]
221. Lees, T., Reece, S., Kratzert, F., Klotz, D., Gauch, M., De Bruijn, J., et al. (2022). Hydrological concept formation inside long short-term memory (LSTM) networks. *HESS* 26(12):3079–3101. doi:10.5194/hess-26-3079-2022 [V]
222. Jiang, S., Zheng, Y., Wang, C., Babovic, V. (2022). Uncovering flooding mechanisms across the contiguous United States through interpretive deep learning on representative catchments. *WRR* 58(1):e2021WR030185. doi:10.1029/2021WR030185 [V]
223. Toms, B.A., Barnes, E.A., Ebert-Uphoff, I. (2020). Physically interpretable neural networks for the geosciences: Applications to Earth system variability. *JAMES* 12(9):e2019MS002002. doi:10.1029/2019MS002002 [V]
224. McGovern, A., Lagerquist, R., Gagne, D.J., Jergensen, G.E., Elmore, K.L., Homeyer, C.R., Smith, T. (2019). Making the black box more transparent. *BAMS* 100(11):2175–2199. doi:10.1175/BAMS-D-18-0195.1 [M]
225. Ebert-Uphoff, I., Hilburn, K. (2020). Evaluation, tuning, and interpretation of neural networks for working with images in meteorological applications. *BAMS* 101(12):E2149–E2170. doi:10.1175/BAMS-D-20-0097.1 [M]
226. Mamalakis, A., Ebert-Uphoff, I., Barnes, E.A. (2022). Neural network attribution methods for problems in geoscience: A novel synthetic benchmark dataset. *Environmental Data Science* 1:e8. doi:10.1017/eds.2022.7 [V]
227. Mamalakis, A., Barnes, E.A., Ebert-Uphoff, I. (2022). Investigating the fidelity of explainable artificial intelligence methods for applications of convolutional neural networks in geoscience. *Artificial Intelligence for the Earth Systems* 1(4):e220012. doi:10.1175/AIES-D-22-0012.1 [V]
228. Sundararajan, M., Taly, A., Yan, Q. (2017). Axiomatic attribution for deep networks. *ICML 2017* (PMLR 70). arXiv:1703.01365 [V-arXiv]
229. Erion, G., Janizek, J.D., Sturmfels, P., Lundberg, S.M., Lee, S.-I. (2021). Improving performance of deep learning models with axiomatic attribution priors and expected gradients. *Nature Machine Intelligence* 3:620–631. doi:10.1038/s42256-021-00343-w [M]
230. Shrikumar, A., Greenside, P., Kundaje, A. (2017). Learning important features through propagating activation differences. *ICML 2017*, PMLR 70:3145–3153. arXiv:1704.02685 [V-arXiv]
231. Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K.-R., Samek, W. (2015). On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation. *PLoS ONE* 10(7):e0130140. doi:10.1371/journal.pone.0130140 [M]
232. Kokhlikyan, N., Miglani, V., Martin, M., Wang, E., Alsallakh, B., Reynolds, J., et al. (2020). Captum: A unified and generic model interpretability library for PyTorch. arXiv:2009.07896 [V-arXiv]; Captum 0.9.0 (PyPI)
233. Sturmfels, P., Lundberg, S., Lee, S.-I. (2020). Visualizing the impact of feature attribution baselines. *Distill* 5(1). doi:10.23915/distill.00022 [M]
234. Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., Kim, B. (2018). Sanity checks for saliency maps. *NeurIPS 2018*. arXiv:1810.03292 [V-arXiv]
235. Hooker, S., Erhan, D., Kindermans, P.-J., Kim, B. (2019). A benchmark for interpretability methods in deep neural networks. *NeurIPS 2019*. arXiv:1806.10758 [V-arXiv]
236. Kindermans, P.-J., Hooker, S., Adebayo, J., Alber, M., Schütt, K.T., Dähne, S., Erhan, D., Kim, B. (2019). The (un)reliability of saliency methods. LNCS 11700, 267–280. doi:10.1007/978-3-030-28954-6_14 [M]
237. Lundberg, S.M., Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*. arXiv:1705.07874 [V-arXiv]; shap 0.54.1 (PyPI, MIT)
238. Lundberg, S.M., Erion, G., Chen, H., DeGrave, A., Prutkin, J.M., Nair, B., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence* 2:56–67. doi:10.1038/s42256-019-0138-9 [M]
239. Aas, K., Jullum, M., Løland, A. (2021). Explaining individual predictions when features are dependent: More accurate approximations to Shapley values. *Artificial Intelligence* 298:103502. doi:10.1016/j.artint.2021.103502 [V]
240. Kumar, I.E., Venkatasubramanian, S., Scheidegger, C., Friedler, S. (2020). Problems with Shapley-value-based explanations as feature importance measures. *ICML 2020*. arXiv:2002.11097 [V-arXiv]
241. Janzing, D., Minorics, L., Blöbaum, P. (2020). Feature relevance quantification in explainable AI: A causal problem. *AISTATS 2020*. arXiv:1910.13413 [V-arXiv]
242. Frye, C., Rowat, C., Feige, I. (2020). Asymmetric Shapley values: incorporating causal knowledge into model-agnostic explainability. *NeurIPS 2020*. arXiv:1910.06358 [V-arXiv]
243. Heskes, T., Sijben, E., Bucur, I.G., Claassen, T. (2020). Causal Shapley values. *NeurIPS 2020*. arXiv:2011.01625 [V-arXiv]
244. Hooker, G., Mentch, L., Zhou, S. (2021). Unrestricted permutation forces extrapolation: variable importance requires at least one more model, or there is no free variable importance. *Statistics and Computing* 31(6):82. doi:10.1007/s11222-021-10057-z [V]
245. Molnar, C., König, G., Herbinger, J., Freiesleben, T., Dandl, S., Scholbeck, C.A., et al. (2022). General pitfalls of model-agnostic interpretation methods for machine learning models. In *xxAI – Beyond Explainable AI*, LNAI 13200, 39–68. doi:10.1007/978-3-031-04083-2_4 [V]
246. Covert, I., Lundberg, S., Lee, S.-I. (2020). Understanding global feature contributions with additive importance measures (SAGE). *NeurIPS 2020*. arXiv:2004.00668 [V-arXiv]
247. Chen, H., Janizek, J.D., Lundberg, S., Lee, S.-I. (2020). True to the model or true to the data? arXiv:2006.16234 [V-arXiv]
248. Apley, D.W., Zhu, J. (2020). Visualizing the effects of predictor variables in black box supervised learning models. *JRSS-B* 82(4):1059–1086. doi:10.1111/rssb.12377 [V]
249. Strobl, C., Boulesteix, A.-L., Kneib, T., Augustin, T., Zeileis, A. (2008). Conditional variable importance for random forests. *BMC Bioinformatics* 9:307. doi:10.1186/1471-2105-9-307 [M]
250. Jain, S., Wallace, B.C. (2019). Attention is not explanation. *NAACL-HLT 2019*, 3543–3556. doi:10.18653/v1/N19-1357 [V]
251. Wiegreffe, S., Pinter, Y. (2019). Attention is not not explanation. *EMNLP-IJCNLP 2019*, 11–20. doi:10.18653/v1/D19-1002 [V]
252. Lim, B., Arık, S.Ö., Loeff, N., Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting* 37(4):1748–1764. doi:10.1016/j.ijforecast.2021.03.012 [V]
253. Ding, Y., Zhu, Y., Feng, J., Zhang, P., Cheng, Z. (2020). Interpretable spatio-temporal attention LSTM model for flood forecasting. *Neurocomputing* 403:348–359. doi:10.1016/j.neucom.2020.04.110 [M]
254. Koh, P.W., Nguyen, T., Tang, Y.S., Mussmann, S., Pierson, E., Kim, B., Liang, P. (2020). Concept bottleneck models. *ICML 2020*. arXiv:2007.04612 [V-arXiv]
255. Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viegas, F., Sayres, R. (2018). Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (TCAV). *ICML 2018*. arXiv:1711.11279 [V-arXiv]
256. Hewitt, J., Liang, P. (2019). Designing and interpreting probes with control tasks. *EMNLP-IJCNLP 2019*, 2733–2743. doi:10.18653/v1/D19-1275 [M]
257. Belinkov, Y. (2022). Probing classifiers: Promises, shortcomings, and advances. *Computational Linguistics* 48(1):207–219. doi:10.1162/coli_a_00422 [V]
258. Cranmer, M., Sanchez-Gonzalez, A., Battaglia, P., Xu, R., Cranmer, K., Spergel, D., Ho, S. (2020). Discovering symbolic models from deep learning with inductive biases. *NeurIPS 2020*. arXiv:2006.11287 [V-arXiv]
259. Cranmer, M. (2023). Interpretable machine learning for science with PySR and SymbolicRegression.jl. arXiv:2305.01582 [V-arXiv]; PySR 2.7.2 (PyPI, Apache-2.0)
260. Udrescu, S.-M., Tegmark, M. (2020). AI Feynman: A physics-inspired method for symbolic regression. *Science Advances* 6(16):eaay2631. doi:10.1126/sciadv.aay2631 [V]
261. Brunton, S.L., Proctor, J.L., Kutz, J.N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. *PNAS* 113(15):3932–3937. doi:10.1073/pnas.1517384113 [V]; pysindy: github.com/dynamicslab/pysindy
262. Zanna, L., Bolton, T. (2020). Data-driven equation discovery of ocean mesoscale closures. *GRL* 47(17):e2020GL088376. doi:10.1029/2020GL088376 [V]
263. Grundner, A., Beucler, T., Gentine, P., Eyring, V. (2024). Data-driven equation discovery of a cloud cover parameterization. *JAMES* 16(3):e2023MS003763. doi:10.1029/2023MS003763 [V]
264. Sobol', I.M. (2001). Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates. *Mathematics and Computers in Simulation* 55(1–3):271–280. doi:10.1016/S0378-4754(00)00270-6 [M]
265. Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M., Tarantola, S. (2010). Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index. *Computer Physics Communications* 181(2):259–270. doi:10.1016/j.cpc.2009.09.018 [M]
266. Herman, J., Usher, W. (2017). SALib: An open-source Python library for sensitivity analysis. *JOSS* 2(9):97. doi:10.21105/joss.00097 [M]; SALib 1.6.0 (PyPI)
267. Iwanaga, T., Usher, W., Herman, J. (2022). Toward SALib 2.0: Advancing the accessibility and interpretability of global sensitivity analyses. *Socio-Environmental Systems Modelling* 4:18155. doi:10.18174/sesmo.18155 [M]
268. Razavi, S., Jakeman, A., Saltelli, A., Prieur, C., Iooss, B., Borgonovo, E., et al. (2021). The future of sensitivity analysis: An essential discipline for systems modeling and policy support. *Environmental Modelling & Software* 137:104954. doi:10.1016/j.envsoft.2020.104954 [M]
269. Razavi, S., Gupta, H.V. (2016). A new framework for comprehensive, robust, and efficient global sensitivity analysis: 1. Theory. *WRR* 52(1):423–439. doi:10.1002/2015WR017558 [M]
270. Morris, M.D. (1991). Factorial sampling plans for preliminary computational experiments. *Technometrics* 33(2):161–174. doi:10.1080/00401706.1991.10484804 [M]
271. Owen, A.B. (2014). Sobol' indices and Shapley value. *SIAM/ASA JUQ* 2(1):245–251. doi:10.1137/130936233 [M]
272. Song, E., Nelson, B.L., Staum, J. (2016). Shapley effects for global sensitivity analysis: Theory and computation. *SIAM/ASA JUQ* 4(1):1060–1083. doi:10.1137/15M1048070 [M]

### Data sources and access (checked 2026-10-22)
273. USGS Water Data APIs v1 (OGC API), base `https://api.waterdata.usgs.gov/ogcapi/v1/` (resolves). Announcement: https://waterdata.usgs.gov/blog/api-v1-release/ [V: "WaterServices will be decommissioned in early 2027"; V0 redirects in mid-2027]
274. dataretrieval-python 1.4.0 (PyPI; NEWS: v1 default since 2026-09-22): https://github.com/DOI-USGS/dataretrieval-python [V]; dataRetrieval (R) 2.7.27 on CRAN [V]
275. Water Quality Portal: https://www.waterqualitydata.us/ (legacy WQX 2.2 profiles "do NOT contain USGS data added after March 11, 2024"; WQX 3.0 at https://waterqualitydata.us/beta/) [V]; Read, E.K., et al. (2017). Water quality data for national-scale aquatic research: The Water Quality Portal. *WRR* 53(2):1735–1745. doi:10.1002/2016WR019993 [V]
276. USGS NLDI: https://api.water.usgs.gov/nldi/ (main-stem navigation tested; the old labs.waterdata.usgs.gov path returned 404) [V]
277. NEON Data Portal API: https://data.neonscience.org/api/v0/products/DP1.20288.001 (and the other product IDs in §e.0) [V]; neonutilities (Python) 2.0.2 and neonUtilities (R) 4.0.2 [V]
278. Hill, R.A., Weber, M.H., Leibowitz, S.G., Olsen, A.R., Thornbrugh, D.J. (2016). The Stream-Catchment (StreamCat) Dataset. *JAWRA* 52(1):120–128. doi:10.1111/1752-1688.12372 [V]; https://www.epa.gov/national-aquatic-resource-surveys/streamcat-dataset ("over 600 metrics", about 2.65 M segments) [V]
279. EPA National Rivers and Streams Assessment: https://www.epa.gov/national-aquatic-resource-surveys/nrsa and data page https://www.epa.gov/national-aquatic-resource-surveys/data-national-aquatic-resource-surveys [pages resolve]
280. Abatzoglou, J.T. (2013). Development of gridded surface meteorological data for ecological applications and modelling (gridMET). *International Journal of Climatology* 33(1):121–131. doi:10.1002/joc.3413 [M]; https://www.climatologylab.org/gridmet.html
281. Thornton, P.E., Shrestha, R., Thornton, M., Kao, S.-C., Wei, Y., Wilson, B.E. (2021). Gridded daily weather data for North America with comprehensive uncertainty quantification (Daymet v4). *Scientific Data* 8:190. doi:10.1038/s41597-021-00973-0 [M]; https://daymet.ornl.gov/
282. Muñoz-Sabater, J., Dutra, E., Agustí-Panareda, A., Albergel, C., Arduini, G., Balsamo, G., et al. (2021). ERA5-Land. *ESSD* 13(9):4349–4383. doi:10.5194/essd-13-4349-2021 [M]; https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land
283. Chegini, T., Li, H.-Y., Leung, L.R. (2021). HyRiver: Hydroclimate data retriever. *JOSS* 6(66):3175. doi:10.21105/joss.03175 [M]
284. Addor, N., Newman, A.J., Mizukami, N., Clark, M.P. (2017). The CAMELS data set: catchment attributes and meteorology for large-sample studies. *HESS* 21(10):5293–5313. doi:10.5194/hess-21-5293-2017 [M]
285. Kratzert, F., Nearing, G., Addor, N., Erickson, T., Gauch, M., Gilon, O., et al. (2023). Caravan — A global community dataset for large-sample hydrology. *Scientific Data* 10:61. doi:10.1038/s41597-023-01975-w [M]
286. StreamPULSE data portal: https://data.streampulse.org/ [page resolves]

---

## Appendix A: Numbers computed for this note (reproducible)

- **DO saturation.** Benson–Krause freshwater equation at 1 atm: ln C* = −139.34411 + 1.575701×10⁵/T − 6.642308×10⁷/T² + 1.243800×10¹⁰/T³ − 8.621949×10¹¹/T⁴, with T in K and C* in mg/L. Results are in the §a.6 table. Validate against USGS DOTABLES before use.
- **Arrhenius.** E = 0.65 eV and k_B = 8.617×10⁻⁵ eV/K. The rate ratio for +2 °C is 1.205 at 10 °C, 1.190 at 20 °C and 1.184 at 25 °C. Q₁₀ is 2.48, 2.34 and 2.27.
- **Schmidt number.** Freshwater O₂ polynomial Sc = 1800.6 − 120.10T + 3.7818T² − 0.047608T³ (Wanninkhof 1992). Sc(20 °C) = 530.5, so K_O2/K₆₀₀ = 1.064 at 20 °C and 1.116 at 22 °C, about +4.9 % per +2 °C.
- **USGS counts.** `GET api.waterdata.usgs.gov/ogcapi/v1/collections/time-series-metadata/items?parameter_code=<pc>&computation_period_identifier=Points&properties=monitoring_location_id,begin,end&skipGeometry=true`, paged. Locations were aggregated by min(begin) and max(end). "Active" means end ≥ 2025-01-01. Overlap = min(end) − max(begin) across parameters, in years. Run unauthenticated on 2026-10-22. Scripts: /tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/usgs/count_ts.py and /tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/usgs/colocate.py (the scratchpad is session-local; copy them into the repo if they are needed).
- **NLDI pairs.** `api.water.usgs.gov/nldi/linked-data/nwissite/<id>/navigation/DM/nwissite?distance=100|300` for 250 random sites (seed 0) from the 1,374 USGS sites with active continuous temperature and discharge and a temperature span of at least 5 years. A hit is another site of the same set downstream. Result: ≥ 129 hits within 100 km and ≥ 159 within 300 km, with 20 request errors. Script: /tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/usgs/pairs.py.
- **NEON product availability.** NEON API `products/<DP>` siteCodes and availableMonths on 2026-10-22: DP1.20288.001 (34 sites, 2016-03 to 2026-09), DP1.20093.001 (34), DP1.20033.001 (34), DP1.20053.001 (24), DP1.20264.001 (10), DP1.20120.001 (34), DP1.20166.001 (34, to 2024-08), DP1.20107.001 (28), DP1.20190.001 (24), DP4.00130.001 (28), DP1.20282.001 (34), DP1.20138.001 (34), DP1.20042.001 (34), DP1.20261.001 (10), DP1.20097.001 (34), DP1.20163.001 (34), DP1.20016.001 (34), DP1.20048.001 (28), DP1.20280.001 (24), DP1.20126.001 (34).

## Appendix B: Searches to run before claiming novelty

Run these in Google Scholar or Web of Science, plus a forward-citation check of the anchor papers named. Record dates and hit lists in the repo.

- **E1:** "dissolved oxygen" AND ("diel" OR "nightly minimum" OR "high-frequency") AND (warming OR "temperature sensitivity") AND (rivers OR streams) AND (continental OR national OR CONUS), 2019–2026. Forward citations of Zhi et al. 2023 and Blaszczak et al. 2023.
- **E2:** ("travel time" OR "lag") AND ("water quality" OR conductance OR turbidity) AND ("upstream" "downstream") AND (sensor OR "high-frequency") AND (network OR national). Forward citations of Sendrowski & Passalacqua 2017.
- **E3:** forward citations of Hare et al. 2021 that mention "dissolved oxygen".
- **E4:** ("hysteresis" AND turbidity AND discharge) AND (national OR continental OR "hundreds of sites" OR CONUS). Forward citations of Hamshaw et al. 2018 and Lloyd et al. 2016.
- **E5:** forward citations of Blaszczak et al. 2023, plus "hypoxia" AND "temperature" AND "low flow" AND "rivers" AND "threshold".
- **E6:** forward citations of Pennekamp et al. 2019 that apply permutation entropy to water quality.
- **E7:** "specific conductance" AND (TITAN OR "extirpation concentration" OR "gradient forest") AND ("National Rivers and Streams Assessment" OR NRSA OR NEON).
- **Causal, WQ-specific:** (PCMCI OR "convergent cross mapping" OR "transfer entropy") AND ("dissolved oxygen" OR "water quality" OR nitrate OR turbidity) AND (river OR stream OR lake).
