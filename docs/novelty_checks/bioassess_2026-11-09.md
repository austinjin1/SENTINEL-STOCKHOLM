# Novelty / published-method check — M5 SENTINEL-BioAssess (benthic community → co-located stream chemistry)

**Date:** 2026-11-09 (after the single test pass; expands the one-sentence check in protocol §10).
**Status:** web-search pass only. Before any "first-in-class" or "SOTA" wording leaves the repository, a
Scopus / Google Scholar pass is still required (RESEARCH_PLAN §15.1, §15.5 item 3). A novelty finding does
not earn a label by itself; the labels come from the frozen test (protocol §9.1) as scoped in the
post-results review (REVIEW_REPORT_POST_RESULTS.md, C-M5P-1) and RESULTS.md (2026-11-09).

## Task under test
Infer six co-located water-chemistry variables (log TP, log TN, log conductivity, log chloride, log
turbidity, lab pH) from a benthic macroinvertebrate sample, evaluated by leave-one-ecoregion-out ×
cycle-out on US EPA NRSA (train 2008-14, test 2018-19; replication 2023-24), with location-cluster
bootstrap CIs and Holm correction.

## Searches (2026-11-09, web search)
1. `bio.infer Yuan maximum likelihood inference environmental conditions benthic macroinvertebrate
   taxon-environment relationships` → the R package and its paper: Yuan (2007), "Maximum likelihood
   method for predicting environmental conditions from assemblage composition: the R package bio.infer",
   *J. Stat. Softw.* 22(3) (https://www.jstatsoft.org/article/view/v022i03); CRAN `bio.infer` 1.3-6
   (2020; https://cran.r-universe.dev/bio.infer); US EPA CADDIS Vol. 4, PECBO appendix
   (https://www.epa.gov/caddis-vol4/pecbo-appendix), which recommends ML inference over weighted
   averaging when full taxon–environment relationships are available.
2. `machine learning predict stream water chemistry nutrients conductivity from macroinvertebrate
   community composition` → studies predicting *biological condition* from landscape/chemistry
   (USGS Chesapeake small-stream RF models; gradient-boosted benthic condition, *JABES* 2021,
   https://link.springer.com/article/10.1007/s13253-021-00479-7), invertebrate pollution *classification*
   with RF (Alpine rivers), and a review of macroinvertebrate/fish modelling techniques. No study found
   that predicts the six chemistry variables from macroinvertebrate composition under an out-of-region test.
3. `deep learning set transformer species assemblage infer environmental variable biomonitoring` →
   transformer models on assemblages for *habitat* or assemblage prediction (Pl@ntBERT habitat mapping,
   CVPRW 2025; BioAnalyst foundation model, arXiv 2507.09080; STEM-LM preprint). None infers stream
   chemistry from benthic samples.
4. `diatom macroinvertebrate transfer function random forest weighted averaging inference stream total
   phosphorus conductivity National Rivers and Streams Assessment` → diatom WA / WA-PLS TP transfer
   functions (e.g. New Jersey coastal plain; Great Lakes), NRSA 2008-09 diatom–TP work using RF for taxon
   selection (Lee et al. 2019, reported R² ≈ 0.2), and RF models of *reference-condition* chemistry from
   landscape predictors on NRSA (Nelson, Courtwright & Hawkins, *Freshwater Science*;
   https://www.journals.uchicago.edu/doi/10.1086/736692) — chemistry from landscape, not from biota.
5. `Yuan 2006 Estimation and Application of Macroinvertebrate Tolerance Values` → EPA/600/P-04/116F:
   weighted-average and ML tolerance-value inference (temperature, fine sediment; western EMAP).
6. `Juggins quantitative reconstructions in palaeolimnology spatial autocorrelation transfer function
   cross-validation` → Juggins (2013, *Quat. Sci. Rev.*, "new paradigm or sick science?") and related work
   showing that random/leave-one-out CV overstates transfer-function skill under spatial autocorrelation;
   spatially blocked (h-block) validation gives more conservative estimates. This supports M5's
   leave-one-ecoregion-out design and explains why published in-sample/LOO transfer-function R² values are
   not comparable to M5's numbers.

## Findings
- **Published methods for this task family:** weighted averaging (WA), WA-PLS, modern-analogue (kNN)
  transfer functions; tolerance-value / ML inference (bio.infer); random forests. M5's frozen comparator
  set re-ran WA-PLS2, kNN-MAT, RF (several feature sets), LightGBM and TabPFN v2 plus simple baselines
  (ecoregion mean, nearest reference site, geography-only RF) on identical rows.
- **bio.infer was not in the frozen set and remains effectively NOT compared.** A post-hoc
  re-implementation (`sentinel/models/bioassess/bioinfer.py`; per-taxon quadratic logistic models, OTUs with
  ≥ 30 training occurrences, ML over a 200-point grid; addendum A3) scored Test A R² −0.17 to −1.27, and its
  development LOEO sanity check (NRSA 2008-14): R² −0.29 to −1.60, far below WA-PLS2 (0.18–0.30; registry e.g. `9ec21e854822`, `62b4cb574e36`) and kNN-MAT (0.13–0.31). Conclusion (addendum A4): our bio.infer-style re-implementation performed poorly; it may not reflect the EPA software; the SOTA label rests on the pre-registered comparators (WA-PLS2, kNN-MAT, RF, TabPFN, ensembles). A faithful comparison would need the EPA R package itself
  (its OTU taxonomy, taxon–environment fits and `mlsolve`), not attempted.
- **No published NRSA leave-one-ecoregion-out benchmark number exists** for this task as far as these
  searches show, so M5's comparison is to re-run methods on identical rows, not to published numbers.
- **Not claimed as novel:** predicting environmental variables from assemblages (a decades-old field);
  set/transformer encoders for species assemblages (prior art in habitat/assemblage modelling).
- **Possibly first-in-class (provisional, pending the database search):** a set-transformer benthic
  encoder, in an ensemble with the strongest re-run baselines, evaluated on NRSA by leave-one-ecoregion-out
  × cycle-out with a frozen single pass. The SOTA scope is narrow (RESULTS.md 2026-11-09).

## Literature pass 2 (2026-11-09, extended web search; requested "Scopus/Scholar pass")
**Scopus was not accessed** (institutional subscription; no credentials are handled by agents) and Google
Scholar cannot be queried programmatically from this host. Substitute: extended web searches restricted to
journal publishers (ScienceDirect, Wiley, Springer, UChicago, Frontiers, MDPI, Nature, T&F, PLOS, EPA). A
human Scopus/Scholar check remains required before any SOTA / first-in-class wording leaves the repository.

7. `predicting water chemistry nutrients conductivity from benthic macroinvertebrate assemblages random
   forest neural network cross-validation spatial` (publisher-restricted) → the closest prior work is
   **bioindication of chemistry with ANNs in Central Germany** (51 sites, 9 small streams; conductivity,
   nutrient forms, BOD5, pH predicted from benthic invertebrates; GRNN/MLP;
   https://link.springer.com/article/10.1023/A:1011433529239) — small, single-region, no out-of-region
   test reported in the abstract. Others model biotic *indices* from environment (Spanish river networks,
   RF, https://www.sciencedirect.com/science/article/abs/pii/S0048969716323749; Chesapeake RF with
   explainability, https://www.sciencedirect.com/science/article/pii/S0301479722016413; macrophyte-based
   quality, https://link.springer.com/article/10.1007/s11356-024-35497-8) or communities from environment
   (Zwalm ANN, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6009754/).
8. `macroinvertebrate inference model total phosphorus total nitrogen lakes National Lakes Assessment
   chironomid transfer function` → chironomid/diatom WA, PLS and WA-PLS TP inference models for lakes
   (e.g. https://hero.epa.gov/reference/3352828/; Yangtze, https://link.springer.com/article/10.1007/s11434-006-2062-8;
   independent TP validation, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11064217/); the NLA technical
   reports use TN/TP only for condition classification. No benthic-macroinvertebrate → lake chemistry model
   on NLA was found (relevant to the M5-X lake block).
9. `deep learning benthic invertebrate community composition predict environmental stressor river UK
   Environment Agency RIVPACS RICT machine learning` → RIVPACS/RICT predict *expected fauna* from
   environment (the inverse direction; https://en.wikipedia.org/wiki/RIVPACS; RICT,
   https://nora.nerc.ac.uk/id/eprint/517305/; composition predictions, bioRxiv 2024.06.14.599001); feed-forward
   ANN fauna prediction (Latium). No UK study inferring chemistry from EA invertebrate samples with ML under
   area-held-out validation was found (relevant to the M5-X UK block).

**Updated conclusion:** inferring chemistry from benthic invertebrates is established (WA/WA-PLS/ML transfer
functions; small-scale ANNs); no published benchmark matches NRSA leave-one-ecoregion-out, NLA lakes or EA
leave-area-out designs, so M5 compares against re-run methods on identical rows. The first-in-class wording
stays provisional pending a human Scopus/Scholar pass.
