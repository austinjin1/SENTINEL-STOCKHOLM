# Novelty and literature check: E7, national replication of the 300 µS/cm conductivity benchmark, 2026-10-25

**Scope (docs/research/06 §E7).**
- Use NRSA (plus NEON) macroinvertebrates and specific conductance (SC).
- Apply the Cormier & Suter 2013 extirpation-concentration (XC95/HC05) method, TITAN2 and gradient forest,
  stratified by ecoregion and by natural background SC (Olson & Cormier 2019).
- Use Central Appalachia as the positive control. Optionally replicate in the UK.

**Method caveat.** Searches used a general web index, not Scholar or WoS. The EPA documents below were downloaded
and read in full where quoted.

## Verified method details
### EPA 2011 benchmark, EPA/600/R-10/023F
Read via https://ordspub.epa.gov/ords/eims/eimscomm.getfile?p_download_id=502333 (linked from
assessments.epa.gov/risk/document/&deid=233809).

**Data.** 2,210 samples from Ecoregions 69 and 70 (WV WABbase). Quote: "sites were excluded with pH <6 before
identifying the XC95 values".

**Genus inclusion.** Quote: "a genus was excluded if it was never observed at reference sites or it was observed in
<25 samples". 220 genera occurred at reference sites, and "163 occurred at ≥25 sampling locations".

**XC95.**
- XC95 is the 95th centile of a **weighted** cumulative distribution of genus occurrences along log10 SC.
- Bins are "each 0.017 log10 conductivity units wide", 60 bins in total.
- Weight per sample = 1/n_bin. The two bracketing exposure levels are linearly interpolated.

**Result.** Quote: "The hazardous concentration value at the 5th centile of the SSDs is 295 μS/cm. Rounding the
HC05 to two significant figures yields a benchmark value of 300 μS/cm."
- Bootstrap: 1,000 resamples of 2,210 samples, giving 95 % CI 228–303 µS/cm.
- WV 295 vs KY 282 µS/cm.

**Background.** The 25th centile of probability samples was 72 µS/cm (Eco 69), 153 (Eco 70) and **116 combined**.
The reference-site 75th centile was 66, 214 and 150.

### EPA 2016 draft field-based methods (XCD)
Read at archive.epa.gov/epa/sites/production/files/2016-12/documents/field-based-conductivity-report.pdf.

**XC95 definition (eq. 3-1).** F(x) = Σ_i w_i Σ_j I(x_ij < x and G_ij) / Σ_i w_i Σ_j I(G_ij), with w_i = 1/M_i.
Quote: "The XC95 value is defined as the stressor value corresponding to F(x) = 0.95."
- Bins: 1/60 of the log10 SC range ("40 to 60 bins usually gives acceptable results").

**HC05.** Quote: "The cumulative proportion for each genus P is calculated as P = R/(N + 1) … The HC05 is derived by
two-point interpolation between the XC95 values bracketing P = 0.05". Genera coded ~ or > are kept in N.

**Qualifiers.** A GAM with a 90 % CI is used only to qualify XC95s as "~" or ">".

**Bootstrap inclusion.** Genera "that occur at least once in the original data set's reference sites and in more
than 24 sampled sites". 1,000 bootstrap replicates.

**Other rules.** pH < 6 is excluded by default. The B-C (background-to-criterion) model uses log HC05 against
log background (Appendix D: 24 XCDs, background 22–626 µS/cm, checked with LOOCV).

### Cormier & Suter 2013 (method paper)
Confirmed to exist: "A method for deriving water-quality benchmarks using field data", *Environ. Toxicol. Chem.*
32(2):255–262, doi:10.1002/etc.2057 (OUP ETC issue 32/2 listing).

Companions: "Derivation of a benchmark for freshwater ionic strength" (263–271, doi:10.1002/etc.2064); causation
(doi:10.1002/etc.2056); extirpation causation (doi:10.1002/etc.2059); confounding (doi:10.1002/etc.2054).

The **2013 paper text itself was not read**. The details above come from the EPA 2011 and 2016 documents that
implement it [U for any difference from the 2013 paper].

### TITAN2
Read from the CRAN manual for v2.4.4, https://cran.r-project.org/web/packages/TITAN2/TITAN2.pdf.

**Defaults.** `titan(env, txa, minSplt = 5, numPerm = 250, boot = TRUE, nBoot = 500, imax = FALSE, ivTot = FALSE,
pur.cut = 0.95, rel.cut = 0.95, ncpus = 1, memory = FALSE)`.

**Screening.** `txa.screen` checks "whether occurrence frequencies for any taxa fall below 3". The manual has no
hard minimum-occurrence default beyond this warning.

**Unverified [U]:** the p-value threshold defining "reliability" (sources disagree, 0.05 vs 0.01), and the
minimum-occurrence advice in Baker & King 2010 (*Methods Ecol. Evol.* 1:25–37, doi:10.1111/j.2041-210X.2009.00007.x;
not read directly).

## Searches (web, 2026-10-25)
1. "specific conductance extirpation concentration XC95 macroinvertebrate national NRSA benchmark" (extended).
   - EPA 2016 draft (above).
   - Ohio Cl⁻/SO4²⁻ study (Water 13:1815).
   - Chesapeake background-SC model (ACS EST Water, doi:10.1021/acsestwater.4c00589).
   - Victoria, Australia XC95 (ETC, doi:10.1002/etc.4400).
   - 2014 EPA peer review critical of the XC95 concept.
   - **No national NRSA-based XCD/HC05 was found.**
2. "TITAN specific conductance macroinvertebrate threshold NRSA OR NEON". TITAN on SC appears only regionally:
   - Mississippi Alluvial Plain (USGS 70246601; thresholds higher than in other ecoregions);
   - South Australia (Sultana et al. 2019, Ecol. Indic. 101:867).
   - No NRSA or NEON TITAN analysis was found.
3. "national scale conductivity benchmark … background-to-criterion regression Cormier 2018 2019".
   - **Cormier, Zheng, Novak & Flaherty 2018, STOTEN 633:1647–1656** (flow-chart for two field methods; cited in
     data.gov records; not read).
   - EPA data "Estimation of field-based benchmarks from a background specific conductivity" (24 state datasets).
   - **EPA has already produced multi-region HC05s and a B-C regression.**
4. "Olson Cormier 2019 natural background specific conductivity". Olson & Cormier 2019, *ES&T* 53(8):4316–4325,
   doi:10.1021/acs.est.8b06777. Random forest of monthly background SC for CONUS (1,785 segments; NSE 0.85).
5. "UK rivers electrical conductivity macroinvertebrate threshold XC95 OR TITAN … Environment Agency".
   **No UK study was found.**
   - Hits: Hun-Tai, China XC95 benchmark of 249 µS/cm (60 genera, 296 sites);
   - Australian TITAN vs HEA;
   - a Natural England organic-pollution report.
6. "Great Britain river invertebrates conductivity salinity tolerance national … RIVPACS". No UK hit. Only Kefford
   (Barwon, Australia, LC50s) and Queensland salinity indices.
7. "NRSA conductivity macroinvertebrate threshold salinity sensitive taxa".
   - NRSA 2013–14 Appendix B: assessment benchmarks of 500/1000 µS/cm (most ecoregions). These are condition
     classes, not extirpation-derived.
   - Portoviejo (Ecuador) TITAN.
   - Virginia family-level extirpation thresholds (366 / 391 / 136 µS/cm; source page title mismatch [U]).
8. "XC95 replication other region Europe OR China OR Australia OR Kentucky 2020–2024".
   - EPA 24-area study.
   - An Appalachian fish XC95 paper (PMC5886021; lowest fish XC95 322 µS/cm).
   - A consultant critique (Ecoregion 50, Minnesota) showing poor transfer of genus XC95s across ecoregions
     (not peer-reviewed).

## Conclusion (provisional)
- **This is a replication/extension, not a novel finding.** The XC95/HC05 method, the 300 µS/cm benchmark, a
  multi-region B-C model (24 XCDs) and CONUS background SC (Olson & Cormier 2019) all exist.
- **Not found:**
  - an HC05/XCD derived from the **NRSA probability survey** nationally;
  - a three-way triangulation (XCD vs TITAN2 vs gradient forest) on one national dataset;
  - any UK replication.
- Permitted wording: "an independent replication of the EPA field-based conductivity benchmark using NRSA data,
  with methodological triangulation". Use "novel" only for the UK application, if done and still unreported after a
  Scholar check.
- NEON (34 sites) is underpowered for an XCD. For the B-C model, the EPA draft says data requirements "were relaxed
  relative to those for calculating a HC05 using the XCD method (i.e., fewer than 90 genera across 500 sites)",
  which implies about 90 or more genera and 500 or more sites for a stand-alone XCD [U on the exact rule].
- **Pre-registration must fix:** pH < 6 exclusion; ≥ 25 occurrences plus presence at reference sites; 60 log10
  bins with w = 1/n; P = R/(N+1) interpolation; 1,000 bootstraps; TITAN2 defaults (nBoot 500, numPerm 250,
  purity/reliability 0.95).
