# 03 — Microbial / eDNA / Molecular-Omics Modalities for Water Quality

**Scope.** Microbiome (16S amplicon, shotgun metagenome), microbial source tracking (MST), eDNA
metabarcoding, toxicogenomics, and antibiotic-resistance genes (ARGs) as SENTINEL modalities for
freshwater water-quality (WQ) indicators and WQ disasters.

**Status.** Research/scoping document. Every number below is either (i) read off a primary source
page that is cited with a URL/DOI, or (ii) computed by this session from a downloaded file or a
public API, in which case the query is stated. Anything not confirmed against a primary source is
marked **UNVERIFIED**. No numbers here are SENTINEL results.

**Verification date.** 2026-10-22. Scratch artifacts from this session (kept outside the repo):
`/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/research_samples/microbe/`
(NEON `products.json`, EMP mapping file, NEON user-guide PDFs, ENA/NMDC query output) and
`/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/refs/`
(Crossref verification logs).

**Why this document is cautious.** The 2026-10-21 integrity review
(`/home/bcheng/STOCKHOLM/REVIEW_REPORT.md`) found that the previous "MicroBiomeNet" headline came
from logistic regression on CLR taxonomic features (network scored 0.30–0.40 on a geographic
holdout), that its per-class numbers came from a random split in which ~63% of labels were MD5 coin
flips, and that the previous "ToxiGene" network scored 0.51 (chance) under leave-one-study-out. The
literature surveyed below says those outcomes were the *expected* outcomes, not bad luck. The
recommendation in §(d) is built so that the headline claim survives the evaluation that broke the
old ones.

---

## (a) SOTA methods and reported metrics

### a.1 Microbiome foundation models and deep learning on abundance profiles

| Method | Paper (year, venue) | Data / corpus | Evaluation protocol | Headline numbers vs baselines | Code |
|---|---|---|---|---|---|
| Microbiome language model (ELECTRA-style transformer over ASV tokens) | Pope et al. 2025, *PLOS Comput Biol* 21(5):e1011353 | Pretrain: 18,480 American Gut Project 16S-V4 samples (~16k train, 2,480 val), 26,726-ASV vocabulary; frozen 100-d GloVe taxa embeddings; 5 encoder blocks; abundance-rank positional encoding; 512-token truncation; 7.07M params | In-domain: 5× repeated 80/20 holdout on AGP. Cross-study: AGP-trained IBD model applied to HMP2 and Halfvarson with **no** fine-tuning | In-domain AUROC (transformer vs RF-on-raw-abundances "No-DR"): IBD **0.687±.04 vs 0.717±.05**; Fruit 0.619±.02 vs 0.655±.03; Vegetable 0.700±.02 vs 0.729±.01 — **RF wins all three**. Cross-study AUC: HMP2 transformer-ensemble 0.682±.02 vs RF 0.657±.01 vs CAEMatch 0.706±.04; Halfvarson transformer **0.805±.01 vs RF 0.530±.01**, CAEMatch 0.488±.04 | Dryad `10.5061/dryad.tb2rbp08p` |
| MGM (Microbial General Model) — causal-LM transformer over genus tokens | Zhang et al. 2026, *Advanced Science* 13(24):e13333 (DOI 10.1002/advs.202513333); preprint bioRxiv 2024.12.30.630825 | MicroCorpus-260K: **263,302** MGnify samples (to June 2023), **9,665** genus-level tokens; 8 layers, 8 heads, FF 1024, 512-token input | Biome classification: 5-fold CV. Cross-region disease: 50% of each region as test, other 50% to train/fine-tune | Biome classification mean ROC-AUC: MGM fine-tuned **0.99**, un-pretrained MGM 0.97, **RF 0.97**, LR 0.95, KNN 0.94, FEAST 0.68 (layer 1 only). On 43,528 newer MGnify samples the paper states RF was slightly better at lineage layers 1–2 and MGM better at 3–5 (no numbers given). Cross-region IBD Ireland→Canada 0.844, Canada→Ireland 0.829; IBS Australia 0.500 before fine-tuning → UK→Australia 0.833, US→UK 0.800. **No environmental/aquatic task** | MIT, `github.com/HUST-NingKang-Lab/MGM`, `pip install microformer-mgm` |
| TabPFN (tabular foundation model, in-context learning) | Hollmann et al. 2025, *Nature* 637:319–326 (DOI 10.1038/s41586-024-08328-6); v1 arXiv:2207.01848 | Synthetic-prior pretraining; applied at inference to the user's table | n/a (general-purpose) | Not a microbiome paper; relevant because it is the strongest "no-tuning" tabular baseline. Code Apache-2.0; TabPFN-2 weights under Prior Labs License (Apache + attribution), later checkpoint weights **non-commercial**; CPU sample cap 5,000 for recent versions; "~8 GB VRAM works well, 16 GB for some large datasets" | `github.com/PriorLabs/TabPFN` |
| Benchmark of foundation models vs classical, 83 cohorts | Mu, Tang & Chen 2026, Research Square preprint, DOI 10.21203/rs.3.rs-8912605/v1 (**preprint — not peer reviewed**) | 83 curated case–control cohorts, 20 diseases, 16S + shotgun | Three settings: intra-cohort CV, cross-cohort transfer, **leave-one-study-out (LOSO)** | From the abstract: GPT-derived semantic embeddings "consistently underperformed standard numerical representations"; TabPFN "achieved strong out-of-the-box performance and competitive cross-cohort robustness, but did not consistently outperform well-tuned classical baselines"; MGM "generally lagged behind the strongest tabular baselines"; batch-effect correction gave "limited and non-uniform" LOSO improvement. Conclusion: foundation models give "at best, modest gains" | — |
| SetBERT (set-level transformer over sequences in a sample) | Ludwig et al. 2025, *Bioinformatics* 41(7):btaf370 | High-throughput sequencing sets; pretraining over sequence sets | Taxonomic classification task | Genus-level taxonomic classification accuracy **95%**, reported as significantly outperforming other models. Note: the headline task is *taxonomic assignment*, not sample-level phenotype | `github.com/DLii-Research/setbert`; QIIME2 plugin `q2-deepdna` |
| DNABERT-S (species-aware DNA embeddings) | Zhou et al. 2025, *Bioinformatics* 41(Suppl):i255–i264 (DOI 10.1093/bioinformatics/btaf188); arXiv:2402.08777 | Genomic DNA | Species clustering/differentiation | Useful as a *sequence encoder* for ASVs; it is not a community-level WQ predictor | released by authors |
| DeepMicro (autoencoder representations + SVM/RF) | Oh & Zhang 2020, *Sci Rep* 10:6026 | 6 human metagenome datasets, 1,156 samples total (IBD 110, EW-T2D 96, C-T2D 344, Obesity 253, Cirrhosis 232, Colorectal 121) | Random 8:2 split × 5 seeds; 5-fold CV on train for hyperparameters | Best AUC (marker profile): IBD 0.955, EW-T2D 0.899, C-T2D 0.763, Obesity 0.659, Cirrhosis 0.940, Colorectal 0.803 (MetAML baseline 0.811 — baseline wins). "None of the autoencoders dominate across the datasets" | released by authors |
| PopPhy-CNN (phylogeny-embedded CNN) | Reiman, Metwally, Sun & Dai 2020, *IEEE JBHI* 24(10):2993–3001 | human metagenomes | random CV | phylogeny-aware CNN; see MetaPheno/benchmarks below for independent comparisons | — |
| Ph-CNN | Fioravanti et al. 2018, *BMC Bioinformatics* 19(S2):49 | human metagenomes | random CV | first phylogeny-aware CNN for metagenomics | — |
| MDeep | Wang et al. 2020, *Brief Bioinform* 22(3):bbaa073 | human microbiome | random CV | phylogeny-informed deep net | — |
| TaxoNN | Sharma, Paterson & Xu 2020, *Bioinformatics* 36(17):4544–4550 | human microbiome | random CV | phylum-stratified CNN ensemble | — |
| MetaNN | Lo & Marculescu 2019, *BMC Bioinformatics* 20(S12):314 | human microbiome | random CV | NN with data augmentation | — |
| MVIB (multimodal variational information bottleneck) | Grazioli et al. 2022, *PLOS Comput Biol* 18(4):e1010050 | human metagenomes (abundance + markers) | random CV | multimodal fusion for microbiome | — |
| GDmicro (GCN + deep adaptation network) | Liao, Shang & Sun 2023, *Bioinformatics* 39(12):btad747 | human gut metagenomes | includes cross-study setting | graph + domain-adaptation approach; closest published architecture to "handle study shift explicitly" | — |

### a.2 Benchmarks and meta-analyses where RF/XGBoost on CLR are hard to beat

| Study | Scale | Protocol | Result |
|---|---|---|---|
| Li, Li & Chen 2025, *Gut Microbes* 17(1):2489074 | 83 gut cohorts, 20 diseases, **156 tool-parameter-algorithm combinations** | internal and external AUC, stepwise optimisation of preprocessing → batch correction → algorithm | ComBat (sva) best for batch correction; **"Ridge and Random Forest ranked the best"** among 9 algorithms including neural networks, XGBoost and LightGBM |
| Karwowska, Aasmets et al. 2025, *Microbiome* 13:2 | **>8,500 samples, 24 shotgun datasets** | classification across transformations and algorithms | "minimal dependence on the choice of algorithm or transformation"; presence–absence performed comparably to abundance-based; only a small subset of predictors needed; **feature importance varied a lot** between transformations |
| Kubinski et al. 2022, *Front Genet* 13:784397 | **15 16S datasets, 7,707 samples** (North America) | **leave-one-dataset-out (LODO)** | compositional transformation + naive zero-centering batch correction best; models with non-linear decision boundaries generalised better than linearly constrained ones |
| Topçuoğlu et al. 2020, *mBio* 11(3):e00434-20 | n = 490 (261 controls, 229 cases), 16S | held-out test within a rigorous repeated pipeline | RF best: AUROC **0.695** (IQR 0.651–0.739) but 83.2 h training; **L2 logistic regression 0.680** (IQR 0.625–0.735) in 12 min and interpretable. Paper's own note: "the estimated performance associated with these models is likely overoptimistic" |
| Giliberti et al. 2022, *PLOS Comput Biol* 18(4):e1010066 | 4,128 metagenomes + 4,026 16S from 30 further studies | meta-analysis across classifiers | "it is the presence rather than the relative abundance of specific microbial taxa to be important"; robust to classifier choice |
| LaPierre, Ju, Zhou & Wang 2019, *Methods* 166:74–82 (MetaPheno) | T2D and obesity metagenomes | re-analysis across ML and DL methods | critical evaluation; T2D/obesity "eluded improved results"; code `github.com/nlapier2/metapheno` |
| Pasolli et al. 2016, *PLOS Comput Biol* 12(7):e1004977 (MetAML) | 8 large metagenomic datasets | cross-validation meta-analysis | the standard classical baseline suite for this field. (Specific per-dataset AUCs quoted elsewhere, e.g. 0.945 cirrhosis / 0.890 IBD, come from a secondary source and are **UNVERIFIED** against the paper here) |
| Vangay, Hillmann & Knights 2019, *GigaScience* 8(5):giz042 | ML Repo | standard task definitions | public repository of microbiome regression/classification tasks — useful template for how to publish a benchmark |

**Reading.** Across the two largest independent benchmarks (83 cohorts; 24 datasets) and the only
large LODO benchmark, deep learning does **not** beat regularised linear models or tree ensembles on
abundance tables. The single published result where a deep model clearly wins is Pope et al.'s
*cross-study* Halfvarson transfer (0.805 vs RF 0.530) — i.e. the advantage shows up only under
distribution shift, and even there the authors flag that pretraining saw the unlabeled evaluation
samples and that HMP2/Halfvarson share 51%/34% of ASVs with the training corpus.

### a.3 Microbial source tracking (fecal / sewage attribution)

| Method | Paper | Approach | Reported performance / protocol |
|---|---|---|---|
| SourceTracker | Knights et al. 2011, *Nat Methods* 8:761–763 (DOI 10.1038/nmeth.1650) | Bayesian mixture, Gibbs sampling | the field reference implementation |
| SourceTracker2 | `github.com/biota/sourcetracker2` | Gibbs sampler, estimates unknown-source fraction, leave-one-out source-class check | BSD-3-Clause, Python 3. **Maintenance is stale**: 90 commits on master, **no published releases**, 51 open issues, Travis CI badge still present (checked 2026-10-22). Default rarefaction 1,000 sequences; no automatic `alpha1` tuning |
| FEAST | Shenhav et al. 2019, *Nat Methods* 16:627–632 | fast expectation-maximization | Simulations built from **EMP** sources subsampled to 10,000 reads, K=20 known sources, 30 sinks per iteration; metric = squared Pearson between estimated and true proportions. Paper states higher accuracy than SourceTracker and than a random-forest classifier "across all levels of divergence" (numeric values appear only in figures — **not quoted here**); unknown-source estimation significantly better (P < 10⁻¹⁴); **30–300× faster**. ICU application AUC 0.91 vs weighted UniFrac 0.78 and JSD 0.87. `github.com/cozygene/FEAST` |
| STENSL | An et al. 2022, *mSystems* 7(5):e00995-21 | EM with sparsity, unsupervised source *selection* | designed for hundreds of candidate sources from repositories such as EMP |
| ONN4MST | Zha et al. 2022, *Genome Medicine* 14:43 | ontology-aware neural network | source tracking among **125,823 samples from 114 niches**, reported as "near-optimal accuracy" and outperforming other methods, with superior speed and interpretability (exact per-niche numbers not retrieved here — **partially verified**). `github.com/HUST-NingKang-Lab/ONN4MST` |
| EXPERT | Chong et al. 2022, *Brief Bioinform* 23(6):bbac396 | transfer-learning, context-aware community classification | companion to ONN4MST; in MGM's own benchmark EXPERT scored 0.54 ROC-AUC on the infant task vs MGM 0.90–0.91 |
| Random forest on host-preferred Clostridiales/Bacteroidales | Roguet, Eren, Newton & McLellan 2018, *Microbiome* 6:185 | supervised RF on taxon subsets | each classifier "correctly identified human and individual animal sources in approximately 90% of the fecal and environmental samples tested"; in-silico detection down to **~0.5%** of the assemblage (~0.04% of sequences matching the classifier); errors were mostly false-positive cat/dog signatures in hosts absent from training |
| FORENSIC | Roguet et al. 2020, *mSystems* 5(2):e00869-19 | online platform for the above | deployable MST service |
| SourceTracker double-blind evaluation | Staley et al. 2018, *Environ Sci Technol* 52(7):4207–4217 | blinded spikes of ≤5 fecal sources at 0.025–10% v/v into ambient freshwater | with a **local** source library: **91% accuracy (31/34)** at detecting presence of contamination, 2 false positives for sewage and 1 for horse; predictions correlated with spiked volume (Spearman ρ ≥ 0.88, P < 0.001); an initial 17-source **non-local** library "showed poor ability"; adding non-local sources to the local library changed nothing (P > 0.79). **Conclusion: geographically local libraries are required** |
| Library-composition sensitivity | Brown et al. 2019, *Environ Sci Technol* 53(1):60–68 | systematic library manipulation | confirms predictions are library-dependent |
| Coastal evaluation | Henry et al. 2016, *Water Research* 93:242–253 | field evaluation of SourceTracker | independent assessment in coastal waters |
| Review | Mathai, Staley & Sadowsky 2020, *J Microbiol Methods* 177:106050 | review of community-based MST with ML classification | the standard review of this exact design space |

**Reading.** There is a **real, documented benchmark design** for MST (blinded known mixtures, local
vs non-local libraries) and a clear classical baseline triple (SourceTracker2, FEAST, Roguet-RF). No
published deep-learning MST model is evaluated on blinded known-mixture freshwater data; ONN4MST and
EXPERT are evaluated on repository-niche classification, not on quantitative mixture recovery.

### a.4 Environmental microbiome → chemistry / water quality

| Study | System, n | Protocol | Numbers |
|---|---|---|---|
| Smith et al. 2015, *mBio* 6(3):e00326-15 — "Natural bacterial communities serve as quantitative geochemical biosensors" | 93 groundwater wells (Oak Ridge); plus Deepwater Horizon PhyloChip samples (7 pre-spill, 13 outside-plume, 39 plume) | **Random forest, 1,000 trees, out-of-bag error — no separate held-out test set.** Spatial confounding probed by retraining while excluding wells within 0–450 m radii | Uranium F1 **0.88** (specificity 0.71 / sensitivity 0.87 / accuracy 0.82), AUC 0.82; nitrate F1 0.73 (0.81/0.63/0.70), AUC 0.76; Deepwater Horizon oil F1 0.98, AUC 0.99; "previously contaminated" F1 0.98. Permutation control for uranium: real labels AUC 0.85 ± 0.008 vs shuffled 0.49 ± 0.12. Quantitative prediction significant for **26 of 38** geochemical parameters (18 at P < 0.0001, 8 at P < 0.01); correlations reported as Kendall's τ in figures, **no R² reported**. Performance declined as nearby wells were excluded, but less than a nearest-neighbour null (86% nitrate, 77% uranium) |
| Hermans et al. 2020, *Microbiome* 8:79 — soil | **>3,000 soil samples, 606 NZ sites**, ~1,300 km gradient | RF classification/regression | land-use class **85%** accuracy; grouping by key physico-chemical properties up to **83%**; individual variables (pH, nutrients, bulk density) predicted–true **R² 0.35–0.79**; correct soil-quality score assignment **50–95%** |
| Hermans et al. 2024, *Environmental Microbiome* 19:45 — **freshwater streams** | **600 biofilm samples, 204 NZ stream sites**, Feb–Mar 2010; 16S V4 (515f/806r), rarefied to 23,718 reads | RF; **random stratified 80/20 × 100 iterations — no spatial blocking** | catchment land use (rural/urban/native forest) **65%** overall accuracy; land-use-index categories 48.5%; environmental clusters 61.4% (disturbed) / 75.3% (undisturbed). Regression: **log stream N R² = 0.02** (weakest), annual mean **air** temperature R² = 0.69 (strongest); most disturbed-site adjusted R² between 0.02 and 0.24. SRA `PRJNA328535` |
| Pilgrim et al. 2022, *Water* 14(15):2361 (US EPA) | 25 wadeable streams, 342 periphyton samples, East Fork Little Miami R., OH, 2016 | TITAN (1,000 bootstraps), boosted regression trees with 10-fold CV, gradient forest | TP change points: low-P **136 µg/L** (110–152), high-P 183 (137–243); TN: low-N 467 (462–695), high-N 772 (665–812); gradient forest 278 ASVs with R² > 0 (max 0.57, mean 0.13, median 0.10); BRT deviance explained 54–67% observed, **45–56% cross-validated**. BioProject 592969; data DOI 10.23719/1504034 |
| Hilderbrand et al. 2020, *PLOS ONE* 15(8):e0236932 | headwater streams, water-column and sediment 16S, spring + summer | models trained to replicate the macroinvertebrate Benthic IBI, with a validation set | sediment communities much better than water column; accuracy "as high as 100% on training data … and **up to 80% on validation data**"; all-OTU models overfit, **order-level** models matched OTU-level on validation |
| Pearman et al. 2022, *Sci Total Environ* 812:152385 — Sediment Bacterial Trophic Index | **259 NZ non-saline lakes**, surface-sediment 16S; 96 lakes with monitoring data used to select 10,888 indicator ASVs, index then derived for the remaining 163 | correlation of SBTI with the Trophic Lake Index; no clearly separate held-out set described | **r² = 0.842, P < 0.001**. ⚠️ Citation details and numbers taken from the project's own summary page, not the publisher record — **partially verified** |
| Tromas et al. 2017, *ISME J* 11(8):1746–1763 | Lake Champlain, **135 samples, 2006–2013**, deep 16S | classification of bloom vs non-bloom; symbolic regression for bloom start date | bloom classification up to **92%** accuracy (**86% after excluding cyanobacterial sequences**); bloom start date predicted with **78–92%** accuracy depending on the training data used; **sequence data was a better predictor than environmental variables** |
| Staley et al. 2014, *Front Microbiol* 5:524 | Upper Mississippi River | correlative | community structure indicative of chemical inputs |
| Berry et al. 2017, *Environ Microbiol* 19(3):1149–1162 | western Lake Erie | observational | cyanobacterial HABs act as a biological disturbance to the bacterial community |
| Simonin et al. 2019, *Environ Microbiol* 21(10):3653–3668 | stream urbanisation gradient | indicator-taxon analysis | microbial indicator taxa along urbanisation |
| Price et al. 2018, *Sci Total Environ* 613–614:1104–1116; Mansfeldt et al. 2020, *Sci Total Environ* 709:135727 | WWTP-effluent-receiving streams | observational | effluent introduces recoverable community shifts in receiving streams |
| Ulrich et al. 2018, *Sci Rep* 8:5683 | **189 samples, 31 headwater stream sites, 2012–2016**, NW Pennsylvania | 5-year observational, hydraulic-fracturing exposure contrast | pH the main driver; methanotroph/methanogen enrichment in shale-activity-positive sites; "sentinel taxa" stable across 5 years |
| Sagova-Mareckova et al. 2021, *Water Research* 191:116767 | position paper | — | the community's own roadmap for integrating microorganisms into routine freshwater biomonitoring |

**Reading.** (i) Microbial communities genuinely carry quantitative chemical information — but every
strong number in this table comes from a protocol with either out-of-bag/random CV or no spatial
blocking. (ii) The one directly comparable *freshwater, many-site, regression* result with honest
scope (Hermans 2024) gives **R² = 0.02 for stream nitrogen**, while the best-predicted variable was
air temperature. (iii) No published study predicts NEON aquatic WQ from NEON aquatic 16S under
site-level or domain-level holdout. That is the gap in §(c).

### a.5 eDNA metabarcoding for biomonitoring and WQ

| Study | System, n | Protocol | Numbers / finding |
|---|---|---|---|
| Cordier et al. 2017, *Environ Sci Technol* 51(16):9118–9126 | marine aquaculture, foraminiferal eDNA | 3 supervised-ML approaches to infer macroinvertebrate-based biotic indices | taxonomy-free SML recovers ecological status comparable to macrofauna inventories; the founding paper of taxonomy-free SML bioassessment |
| Cordier et al. 2018, *Mol Ecol Resour* 18(6):1381–1391 | five ribosomal markers, marine aquaculture impact | models trained per marker, tested on **independent data sets** | all markers gave accurate predictive models, all **outperformed** assessment relying only on taxonomically assigned sequences; no significant difference between universal eukaryotic and prokaryotic markers |
| Frühe et al. 2021, *Mol Ecol* 30(13):2988–3006 | **152 samples, 7 Norwegian salmon farms**, bacterial + ciliate metabarcodes | SML (incl. random forest) vs indicator-value (IndVal) inference, referenced to standard macroinvertebrate monitoring | **SML outperformed IndVal**; RF less sensitive to noisy data and uneven class coverage; **bacteria gave more accurate status assessment than ciliates** |
| Apothéloz-Perret-Gentil et al. 2017, *Mol Ecol Resour* 17(6):1231–1242 | rivers, diatom eDNA | taxonomy-free molecular diatom index | the standard taxonomy-free diatom index |
| Feio et al. 2020, *Sci Total Environ* 722:137900 | **81 river sites (44 reference), central Portugal** | three ML techniques predict reference-condition OTUs, then Observed/Expected ratio → quality class; molecular (HYDGEN) vs morphological (HYDMORPH) models | both "accurate" (observed vs expected richness regression **R² ≈ 0.7**, intercept ≈ 0.8); sensitive to global disturbance (Rs² > 0.30, P < 0.006); the **molecular** model responded to more pressure types (land use, habitat quality) |
| Keck, Brantschen & Altermatt 2023, *Mol Ecol* 32(17):4791–4800 | **64 standard Swiss federal monitoring sites**, macroinvertebrate eDNA | ML classifier: reference vs impacted communities | the eDNA model was "significantly better than a naive model and performs similarly to a model trained on traditional data" |
| Kelly et al. 2020, *Ecol Indic* 118:106725 (+ corrigendum 10.1016/j.ecolind.2020.106890) | UK rivers, diatom DNA metabarcoding | new metric for WFD-style assessment | regulator-grade molecular diatom metric |
| Vasselon et al. 2017, *Ecol Indic* 82:1–12 | WFD monitoring network, Mayotte | scaling diatom metabarcoding to a regulatory network | operational scale-up |
| Flück et al. 2022, *Sci Rep* 12:10247 | tropical South American freshwater fish, 12S (60 bp) | CNN vs OBITools pipeline | taxonomic labels comparable to OBITools, high correlation for regional fauna composition; **~1M sequences/min, ~150× faster**. This is a *throughput* win, not an ecological-status win |
| Keck et al. 2022, *Mol Ecol* 31(7):1820–1835 | meta-analysis | DNA/eDNA vs traditional methods | congruence **and** complementarity — neither strictly dominates |
| Hänfling et al. 2016, *Mol Ecol* 25(13):3101–3119; Valentini et al. 2016, *Mol Ecol* 25(4):929–942; Pont et al. 2018, *Sci Rep* 8:10361 | lake and large-river fish eDNA | validation against long-term survey data | eDNA reflects established survey methods; quantitative patterns recoverable in large rivers |
| Carraro et al. 2020, *Nat Commun* 11:3585 (eDITH) | river networks | hydrological eDNA transport model | the model to use for turning point eDNA into upstream spatial distributions |
| Kuntke et al. 2020, *Ecol Indic* 111:105982 | streams | invertebrate metabarcoding | water-quality assessment from invertebrate metabarcoding |
| Pawlowski et al. 2018, *Sci Total Environ* 637–638:1295–1310; Blackman et al. 2024, *Mol Ecol* 33(11):e17355 | reviews | — | the framing references for (e)DNA in biological assessment |

**Reading.** eDNA bioassessment is the one sub-field where *supervised ML already is the published
SOTA* and has repeatedly beaten taxonomy-based indices. But the published models are random forests
on OTU tables with modest n (64–152 sites); deep learning appears only for **sequence annotation
speed** (Flück 2022), not for status prediction. A deep set-level model for sample→status is a real
opening, but the datasets that pair eDNA with chemistry are small and mostly European/marine.

### a.6 ARGs in water

| Item | Source | Content |
|---|---|---|
| Global sewage AMR (pilot) | Hendriksen et al. 2019, *Nat Commun* 10:1124 | metagenomic AMR monitoring of urban sewage |
| Global sewage AMR (scale-up) | Munk et al. 2022, *Nat Commun* 13:7251 | sewage from **101 countries**; the largest harmonised environmental resistome dataset |
| European WWTPs | Pärnänen et al. 2019, *Sci Adv* 5(3):eaau9124 | WWTP resistomes mirror clinical AMR prevalence patterns |
| Global ARG health risk | Zhang Z. et al. 2022, *Nat Commun* 13:1553 | global health-risk assessment of ARGs |
| Risk framework | Zhang A-N. et al. 2021, *Nat Commun* 12:4765 | omics-based ARG health-risk framework (mobility, host pathogenicity, enrichment) |
| WWTP resistome mechanism | Ju et al. 2019, *ISME J* 13:346–360 | resistomes shaped by community composition, gene exchange, expression |
| Anthropogenic resistome | Lee et al. 2020, *Microbiome* 8:2 | mobile resistome of gut/pathogens drives anthropogenic ARG bloom |
| Review | Larsson & Flach 2022, *Nat Rev Microbiol* 20:257–269 | the authoritative review of environmental AMR |
| ARG annotation (DL) | Arango-Argoty et al. 2018, *Microbiome* 6:23 (DeepARG) | deep-learning ARG prediction from metagenomes |
| ARG quantification pipelines | Yin et al. 2018, *Bioinformatics* 34(13):2263–2270 (ARGs-OAP v2.0); Yin et al. 2023, *Engineering* 27:234–241 (v3.0) | the standard environmental ARG quantification pipeline + SARG database |
| Reference database | Alcock et al. 2023, *Nucleic Acids Res* 51(D1):D690–D699 (CARD 2023) | curated AMR reference, ML support, resistome prediction |
| **ML source tracking of ARGs** | Li, Yin & Zhang 2018, *Microbiome* 6:93 | "Tracking antibiotic resistance gene pollution from different sources using machine-learning classification" — **this is the baseline any SENTINEL ARG-source model must beat** |
| ML source tracking, drinking water | Wang et al. 2023, *Water Research* 246:120682 | ML-based source tracking of ARG pollution in household drinking water |

**Reading.** ARG data at scale are **sewage and WWTP influent/effluent**, not ambient freshwater with
co-located WQ sensors. ARG abundance is also strongly collinear with fecal loading, so "predict ARGs"
and "predict fecal contamination" are partly the same task. Feasible ARG targets are listed in §(c.4).

### a.7 Toxicogenomics / stress transcriptomics

| Item | Source | What it means for a cross-study model |
|---|---|---|
| Zebrafish embryo chemical-exposure meta-analysis | Schüttler, Reiche, Altenburger & Busch 2017, *Toxicol Sci* 157(2):291–304 | **"overlap of differentially transcribed genes in response to chemical stress across independent studies is generally low and the most commonly differentially transcribed genes appear in less than 50% of all treatments across studies."** This is the single most important sentence in this section: it is a direct, published explanation of why a leave-one-study-out transcriptomic classifier lands near chance |
| Toxicogenomic "universe" / modelling | Schüttler et al. 2019, *GigaScience* 8(6):giz057 | time- and concentration-resolved fingerprints projected onto a self-organising map; argues comparability requires *designed* time/concentration resolution, which public GEO series lack |
| Daphnia stressor compendium | Orsini et al. 2016, *Sci Data* 3:160030 (erratum 10.1038/sdata.2017.6) | *D. magna* RNA-seq across **12 environmental stressors**, 2 inbred genotypes + a recombinant cross — the best-designed single aquatic stress compendium, but one lab, one species, 12 stressors |
| Daphnia genome | Colbourne et al. 2011, *Science* 331(6017):555–561 | "ecoresponsive genome" reference |
| EcoToxChip | Basu et al. 2019, *Environ Toxicol Chem* 38(2):279–288; Crump et al. 2023, *Environ Toxicol Chem* 42(8):1763–1771 | 384-well qPCR arrays for **fathead minnow, Xenopus laevis, Japanese quail**; the 2023 paper is explicit that it "represents an initial testing of only **24 EcoToxChips** for each of the model species" |
| EcoToxXplorer | Soufan et al. 2022, *Environ Toxicol Chem* 41(1):21–29; `ecotoxxplorer.ca` | analysis platform (volcano/heatmap/BMD/mods) with user accounts. **No public database of EcoToxChip results, and no chemical/sample counts, are stated on the site** (checked 2026-10-22) |
| CTD | Davis et al. 2023, *Nucleic Acids Res* 51(D1):D1257–D1262 | curated chemical–gene interactions. Current per-organism counts for zebrafish/fathead minnow/Daphnia **not verified here** (the data-status page redirected) |
| EPA HTTr | Harrill et al. 2021, *Toxicol Sci* 181(1):68–89 | high-throughput transcriptomics for chemical screening — human cell lines in this paper; an aquatic (e.g. RTgill-W1) public HTTr compendium of comparable size was **not found** in this session (**UNVERIFIED / likely does not exist publicly**) |
| ECOTOX Knowledgebase | Olker et al. 2022, *Environ Toxicol Chem* 41(6):1520–1539 | apical toxicity tests, **not molecular data** (the review flagged that SENTINEL previously mislabelled "ECOTOX molecular 268K") |
| Great Lakes AOP prioritisation | Corsi et al. 2019, *Sci Total Environ* 686:995–1009 | links measured tributary chemistry to HTS bioactivity and AOPs — the right template for a *downstream application*, not for a transcriptomic encoder |
| Zebrafish hazard screening | Truong et al. 2014, *Toxicol Sci* 137(1):212–233 | multidimensional in-vivo zebrafish hazard assessment |

**GEO scale (my own count, E-utilities `db=gds`, `gse[Entry Type]`, 2026-10-22).** Total GSE /
GSE matching a toxicant keyword set (`toxicity OR toxicant OR toxic OR exposure OR exposed OR
pollutant OR contaminant OR pesticide OR metal OR cadmium OR copper OR estrogen OR effluent`):
*Danio rerio* 3,175 / 1,415; *Pimephales promelas* 109 / 104; *Oncorhynchus mykiss* 276 / 137;
*Oryzias latipes* 87 / 35; *Daphnia magna* 80 / 71; *Daphnia pulex* 33 / 21; *Chlamydomonas
reinhardtii* 119 / 62; *Raphidocelis subcapitata* 4 / 4. These are **upper bounds** on usable series:
the keyword match is loose, series span many platforms (microarray generations and RNA-seq), dose and
duration are rarely harmonised, and many are not water-relevant exposures.

---

## (b) Datasets with verified access details

Legend: **n** figures are as verified. For NEON, the API exposes **site-months of availability**, not
sample counts — a site-month means ≥1 record exists that month. Sample counts require downloading the
tables; do not quote site-months as sample counts.

### b.1 NEON aquatic (the key asset: molecular data co-located with sensors and chemistry)

Verified from `https://data.neonscience.org/api/v0/products` (full JSON saved to the scratch dir).

| Product | Name | Sites | Site-months | Range |
|---|---|---|---|---|
| `DP1.20282.001` | Surface water microbe **marker gene sequences** (16S + ITS, raw) | 34 | 3,338 | 2014-08 → 2026-10 |
| `DP1.20141.002` | Surface water microbe **community taxonomy** (ASV/OTU tables) | 33 | 1,538 | 2015-02 → 2024-11 |
| `DP1.20141.001` | Surface water microbe community composition (**superseded** from RELEASE-2025) | 33 | 681 | 2014-08 → 2018-10 |
| `DP1.20281.001` | Surface water microbe **metagenome sequences** | 34 | 2,568 | 2014-08 → 2026-09 |
| `DP1.20278.001` | Surface water microbe **group abundances** (qPCR gene copies: bacteria/archaea/fungi) | 34 | 2,307 | 2014-08 → 2025-12 |
| `DP1.20138.001` | Surface water microbe **cell count** — **discontinued 1 Oct 2025** | 34 | 3,078 | 2014-08 → 2025-09 |
| `DP1.20280.001` | Benthic microbe marker gene sequences | 24 | 734 | 2014-07 → 2026-08 |
| `DP1.20086.002` | Benthic microbe community taxonomy | 24 | 423 | 2017-10 → 2024-11 |
| `DP1.20086.001` | Benthic microbe community composition (**superseded**) | 23 | 160 | 2014-07 → 2018-10 |
| `DP1.20279.001` | Benthic microbe metagenome sequences | 24 | 302 | 2014-07 → 2026-07 |
| `DP1.20277.001` | Benthic microbe group abundances (qPCR) | 24 | 594 | 2014-07 → 2025-11 |
| `DP1.20288.001` | **Water quality** sonde (SpCond, DO, pH, turbidity, fDOM, chlorophyll-a; YSI EXO2; 1-min streams, 5-min lakes/rivers) | 34 | 3,297 | 2016-03 → 2026-09 |
| `DP1.20093.001` | **Chemical properties of surface water** (grab: general chemistry, anions, cations, dissolved + total nutrients and carbon, ANC) | 34 | 4,083 | 2012-01 → 2026-10 |
| `DP1.20033.001` | Nitrate in surface water (SUNA) | 34 | 3,102 | 2016-12 → 2026-09 |
| `DP1.20163.001` | Periphyton/seston/phytoplankton chemical properties (chlorophyll, pheophytin, AFDM, C/N/P, isotopes) | 34 | 1,086 | 2014-05 → 2026-09 |
| `DP1.20097.001` | Dissolved gases in surface water (CO₂, CH₄, N₂O) | 34 | 3,848 | 2014-10 → 2026-10 |
| `DP1.20126.001` | **Macroinvertebrate metabarcoding** (CO1) | 34 | 1,066 | 2014-07 → 2026-08 |
| `DP1.20221.001` | **Zooplankton metabarcoding** (CO1) | 7 | 183 | 2017-10 → 2026-09 |
| `DP1.20105.001` | Fish sequences DNA barcode (COI) | 28 | 275 | 2017-11 → 2025-12 |
| `DP4.00130.001` | Continuous discharge | 28 | 2,853 | 2016-08 → 2026-09 |

**Co-location (quoted from the product design descriptions).** "Surface water microbe samples are
collected at the same time and location as surface water cell count samples and surface water
chemistry samples … once per month in wadeable streams (12 times per year), and every other month in
lakes and rivers (6 times per year). In wadeable streams, samples are collected near the downstream
**S2 sensor location**. In lakes and rivers, samples are collected near the **buoy sensor set**."
Metagenomes are collected **once per year** at approximate peak productivity, also alongside surface
water chemistry. This is the single most important fact in this document for fusion.

**Sites** (34 aquatic: 24 wadeable streams, 3 non-wadeable rivers, 7 lakes, across 19 ecoclimatic
domains): ARIK BARC BIGC BLDE BLUE BLWA CARI COMO CRAM CUPE FLNT GUIL HOPB KING LECO LEWI LIRO MART
MAYF MCDI MCRA OKSR POSE PRIN PRLA PRPO REDB SUGG SYCA TECR TOMB TOOK WALK WLOU.

**Access, license and gotchas (all verified 2026-10-22):**

1. **Authentication is now required.** `GET https://data.neonscience.org/api/v0/data/DP1.20141.002/ARIK/2021-07`
   returns `HTTP 403 {"error":{"status":403,"detail":"Access Denied"}}`. The `/products` and `/sites`
   endpoints remain open. Per NEON's 2026-06-30 notice, "NEON data users are now required to create a
   user account and either sign in or use an API token to download NEON data"
   (`https://www.neonscience.org/impact/observatory-blog/logins-required-and-data-licensing-updated`).
   Token goes in an `X-API-Token` header or `apiToken` query parameter
   (`https://data.neonscience.org/data-api/authentication/`).
2. **License changed from CC0 to CC BY 4.0** (same notice). Attribution is now required — this must
   appear in SENTINEL's data statement.
3. **OTU identifiers are not stable across releases.** Changelog 2025-11-25 for `DP1.20141.002` and
   `DP1.20086.002`: "a new set of Operational Taxonomic Units (OTUs) were clustered to include the
   AY2024 samples, so this new set will **not be compatible** with the OTUs from the AY25 release";
   per-sample taxon tables now include confidence estimates at all ranks. **Pin one release
   (RELEASE-2025 or RELEASE-2026) and record it in every artifact.**
4. **The 16S primer set changed twice**, and the change is confounded with time
   (`NEON_markerGenes_userGuide_vH.1`, 2026-06-15, author Lee Stanish; Table 1):
   - 16S V4, 515F `GTGCCAGCMGCCGCGGTAA` / 806R `GGACTACNVGGGTWTCTAAT` (Caporaso et al. 2011) — ~2013–2015
   - 16S V3V4, Pro341F `CCTACGGGNBGCASCAG` / Pro805R `GACTACNVGGGTATCTAATCC` (Takahashi et al. 2014) — ~2015–2018, 464 bp
   - 16S V4V5 **current**, 515F-Y `GTGYCAGCMGCCGCGGTAA` / 926R `CCGYCAATTYMTTTRAGTTT` (Parada et al. 2016) — from ~Nov 2018, 411 bp, EMP-aligned
   - ITS: ITS1f / ITS2r throughout
   - The guide says the authoritative per-sample record is the `forwardPrimer` / `reversePrimer` /
     `targetSubfragment` columns of the PcrAmplification table — **use those, not the date.**
5. **Bioinformatics pipeline** (`NEON_commTax_userGuide_vB`, 2026-04-01, author Hugh Cross): primers
   trimmed → QA → DADA2 denoising → ASVs clustered by abundance with VSEARCH → taxonomy by the RDP
   Classifier (Wang & Cole 2024) against RDP 16S or UNITE (ITS).
6. **Sequencing platform changed** to Illumina NovaSeq partway through, with different quality-score
   binning; the guide points to the DADA2 workaround (`dada2` issue #791).
7. Historic data impacts recorded in changelogs: COVID-19 reduced sampling 2020–2022; Toolik
   quarantine 2020–2022; YELL/BLDE inaccessible after the June 2022 Yellowstone flood (resumed
   2022-10-31); TOOK `littoral1`→`littoral3` relabelling; duplicate `amc_fieldSuperParent` records in
   2014–2018 (fixed); some early raw sequence files are run-level `.tar` archives in RELEASE-2021/22/23.

**NEON molecular data mirrored elsewhere (verified):**
- **NMDC**: surface water metagenomes study `nmdc:sty-11-hht5sb92` — **234 biosamples**
  (`https://api.microbiomedata.org/biosamples?filter=associated_studies:nmdc:sty-11-hht5sb92`),
  BioProject `PRJNA406977` (**185 SRA runs**); benthic `nmdc:sty-11-pzmd0x14` — **736 biosamples**,
  BioProject `PRJNA406976` (**544 SRA runs**). NMDC biosample records carry ENVO terms, lat/lon,
  collection date, temperature, depth, conductivity; **pH and dissolved oxygen were empty** in the
  record I inspected, so chemistry must come from NEON, not NMDC.
- **MGnify**: `MGYS00005075` "NEON Surface Water Microbe Marker Gene Sequences - 2014" — only
  **6 samples / 12 analyses**, pipeline 5.0, and the biome is mis-assigned as
  `root:Environmental:Aquatic:Marine:Oceanic`. Not a usable mirror.
- **GBIF/OBIS**: NEON marker-gene occurrences are published as DNA-derived occurrence datasets
  (e.g. GBIF dataset `eca31136-2ea8-4849-8d7f-fc0676b0cde3`). Occurrence form, not abundance tables.

### b.2 Earth Microbiome Project (quantified so the old MicroBiomeNet framing cannot recur)

Downloaded `https://ftp.microbio.me/emp/release1/mapping_files/emp_qiime_mapping_qc_filtered.tsv`
(19 MB) and computed the following directly:

- File: **23,828 samples × 76 columns**, **96 studies**. (`about_mapping_files.md`: release1 =
  27,751 samples, 97 studies; qc_filtered requires ≥1,000 observations in each of four OTU tables and
  excludes controls.)
- `empo_3 == "Water (non-saline)"`: **4,915 samples from 14 studies**.
- **Three studies are 89% of the freshwater set**: "Microbial diversity in arctic freshwaters…"
  1,922; "Bacterial community spatial and temporal variation in a north temperate bog lake" 1,341;
  "Long-term seasonal development in selected lakes of northeast Germany" 1,102 (= 4,365/4,915).
  Remainder: Shenzhen conference samples 188, Lake Mendota 91, Antarctic lake 89, slow sand filters
  74, Great Lakes 49, and six studies with ≤35.
- Countries: USA 3,605, Germany 1,102, Antarctica 89, Scotland 74, Sweden 26, Japan 10, Mexico 7,
  Brazil 2.
- Chemistry completeness **within those 4,915**: temperature 3,824; pH 2,530; oxygen_mg_per_l 1,230;
  nitrate 999; phosphate 347; ammonium 94; salinity 49. Geography is complete (lat/lon 4,915).
- `envo_biome_3`: freshwater lake biome 3,721; freshwater river biome **150**; dense settlement 76;
  blank 957. `Sediment (non-saline)` is a further 544 samples.
- Access: open HTTPS directory listing, no authentication (`otu_tables/`, `otu_info/`,
  `otu_distributions/`, `results/`, `emp_ontology/`). Primary citations: Thompson et al. 2017,
  *Nature* 551:457–463; Shaffer et al. 2022, *Nat Microbiol* 7:2128–2150 (EMP500). Qiita:
  Gonzalez et al. 2018, *Nat Methods* 15:796–798.

**Implication.** EMP freshwater is **150 river samples** and three dominant lake time series in two
countries, with phosphate on 7% of samples. It cannot support a river/stream WQ model, and an
"impacted vs not" label derived from habitat keywords is a habitat-string classifier, exactly as the
review found. Use EMP only as **unlabeled pretraining corpus** or as an out-of-distribution test set.

### b.3 Other freshwater microbiome datasets

| Dataset | Verified access | n | Metadata / co-located chemistry | License |
|---|---|---|---|---|
| **MGnify freshwater biome** | `https://www.ebi.ac.uk/metagenomics/api/v1/biomes/root:Environmental:Aquatic:Freshwater/samples` | **10,342 samples, 328 studies** in the Freshwater subtree (Lake 2,956; Lotic 554; plus Wetlands/Bog, Groundwater, Drinking water, Sediment branches). Analyses: **9,955 amplicon, 1,044 metagenomic** | per-study; sparse and non-standardised | open API; EMBL-EBI terms. Richardson et al. 2023, *NAR* 51(D1):D753–D759 |
| **NCBI BioSample "freshwater metagenome"** (`txid449393`, no-expand) | E-utilities | **195,517 BioSamples**; SRA **174,326 amplicon** + **14,043 WGS** runs. With attribute present: `temp` 13,376; `ph` 12,021; `diss_oxygen` 5,452; `chlorophyll` 2,307; `tot_phosp` 2,254 | attribute-name mining via `[Attribute Name]` works and is the route to a harmonised mined corpus | public |
| **GROWdb** (North American rivers) | Borton et al. 2025, *Nature* 637(8044):103–112; NMDC, KBase, GROWdb Explorer; Zenodo 10.5281/zenodo.11041178 and 10.5281/zenodo.8173286; WHONDRS geochemistry ESS-DIVE doi:10.15485/2439202 | **163 samples / 106 sites**; 158 metagenomes, 57 metatranscriptomes; ~3.8 Tb; ~90% of US watersheds (21 HUC-2); 3,825 MAGs → 2,093 at 99% ID | **up to 287 geochemical + geospatial measurements per site**; EPA StreamCat (600+ watershed metrics), NHDPlus V2, USGS gauges within 10 km, WWTP density, land use, NPP, aridity. WHONDRS collected over 6 weeks in summer 2019 | article CC BY-NC-ND; dataset license not separately stated (**UNVERIFIED**) |
| **LakePulse** (Canada) | 16S: ENA `PRJEB38100` (**210 runs**, first public 2020-07-24). Kraemer et al. 2020, *ISME J* 14:3011–3023; MAGs: Garner et al. 2023, *Nat Microbiol* 8:1920–1934 (+correction); network: Huot et al. 2019, *Sci Total Environ* 695:133668 | 220 lakes sampled Jul–Sep 2017, **212 sequenced, 167 in models**; 16S V4 (U515F/E786R), 11,510 ASVs at 15,000 reads. MAG paper: **308 lakes**, 1,008 genomospecies | DOC, DIC, TN, Mg, K, Na, Cl, SO₄, DO, lake area and depth; **pH and total phosphorus were missing for many lakes and excluded**; chlorophyll-a not available; Human Impact Index from watershed land use | ENA open. `lakepulse.ca/data/` returned **HTTP 502** on 2026-10-22 — portal availability **unverified** |
| **Lake Mendota long-term** | 16S: SRA `PRJNA846788` (**1,023 runs**); Rohwer et al. 2023, *PNAS* 120(11):e2211796120. Metagenomes: Rohwer et al. 2025, *Nat Microbiol* 10:246–257 (**471 metagenomes**, 2,855 species-rep genomes) | **961 16S samples over 496 dates, 2000–2019**, integrated top 12 m | temperature, DO, Secchi, ice cover, phosphorus loading (USGS Yahara), phytoplankton and zooplankton counts; **microcystin + anabaenopeptin on 246 samples, 2013–2018**. EDI packages `knb-lter-ntl.415.2`, `.416.1`, plus NTL-LTER physical/ice/Secchi/phytoplankton/zooplankton packages. Code `github.com/rrohwer/limony` | SRA open; NTL-LTER via EDI (the PASTA listing endpoint rejected anonymous calls here; use the portal) |
| **Laurentian Great Lakes** | SRA `PRJNA591360` (**364 runs**); Paver, Newton & Coleman 2020, *Environ Microbiol* 22(2):433–446 | **154 samples**, 5 lakes, 2012–2013 spring + summer, surface/DCL/hypolimnion, 16S V4 and V4–V5 | **EPA GLNPO** cruise chemistry: temperature, chlorophyll-a, total oxidized N, dissolved silica, phosphorus; CTD profiles; MODIS-Aqua light; flow cytometry counts | SRA open |
| **EPA Ohio periphyton nutrient study** | BioProject `592969` (**684 runs**); data DOI 10.23719/1504034; Pilgrim et al. 2022, *Water* 14(15):2361 | **342 periphyton samples, 25 streams**, weekly, 2016 | weekly **TP (n≈280) and TN (n≈281)** paired with each community sample — one of the best nutrient-paired freshwater 16S sets per sample | US EPA, open |
| **New Zealand stream biofilms** | SRA `PRJNA328535` (**612 runs**); Hermans et al. 2024, *Environ Microbiome* 19:45 | **600 samples, 204 sites** | land use, LUI, catchment variables, log stream N; **water chemistry limited** | SRA open |
| **MiCoDa** | GBIF dataset DOI 10.15468/ver9ne (Jurburg, Arboleda-Baena, Kazem, Frøslev & Jeppesen 2025); `micoda.idiv.de` | ">30,000 samples" of 16S, **all reprocessed over the same 515–806 region**; includes EMP; 12.2M GBIF occurrences | manually curated metadata; aquatic + host-associated + mineral | **CC BY 4.0** |
| **SPIRE** | Schmidt et al. 2024, *NAR* 52(D1):D777–D783; `spire.embl.de` | **99,146 metagenomes, 739 studies**, 16 Tbp assembly, 1.16M MAGs | manually curated contextual data; habitat labels | open |
| **Australian Microbiome (aquatic 16S)** | GBIF `227eb73f-9b53-4a66-b2ef-e12c960d76ba` | 34.7M occurrences (sequence-derived) | includes marine; freshwater subset must be filtered | **CC0** |

### b.4 Fecal / sewage and ARG datasets

| Dataset | Access | n | Co-located measurements | License |
|---|---|---|---|---|
| USGS Great Lakes optical ↔ bacteria | **DOI 10.5066/P9I0SPDQ** (ScienceBase item `5d011658e4b0573a18f77d6d`); paper Corsi et al. 2021, *Environ Sci Technol* 55(20):13770–13782 | **996 samples**; 8 watershed-scale sites, 5 subwatershed sites, **213 storm sewers / open channels** in 3 subwatersheds | absorbance + fluorescence optical properties, **human-associated bacteria (qPCR)**, fecal indicator bacteria, human viruses on a subset. Regressions worked at watershed/subwatershed scale, **not** at small scale | USGS public domain |
| USGS Menomonee River human markers + viruses | **DOI 10.5066/F7736P45**; Lenaker et al. 2018, *Environ Sci Technol* 52(21):12162–12171 | 6 sampling locations, Apr 2009 – Mar 2011 | human-associated bacteria and human-specific virus loads, sample volumes, drainage areas | USGS public domain |
| USGS host-specific flux models, 8 Great Lakes tributaries | **DOI 10.5066/F7VX0DRH**; Dila et al. 2018, *Environ Sci Technol* | 8 tributaries, 2011–2013 | regression models + inputs for Lachnospiraceae, human *Bacteroides*, ruminant *Bacteroides* flux; hydrology, precipitation, land use | USGS public domain |
| Milwaukee storm/CSO sewage survey | Olds et al. 2018, *PLOS Medicine* 15(7):e1002614; data in S1–S3 Supporting Information | **>2,000 samples collected, 755 analysed by qPCR**; 4 automated stations; Apr–Sep 2014 and 2015; 11 rain events, 4 low-flow periods, **2 CSO events** | human *Bacteroides* (HB), human Lachnospiraceae (Lachno2), ruminant marker at 2 sites, *E. coli*, enterococci, fecal coliforms. **CSO events ~10× higher** than heavy rain without overflow; ***E. coli* vs HB ρ = 0.29** | CC BY (PLOS) |
| Sewage microbiome reference | Newton et al. 2015, *mBio* 6(2):e02574-14 | — | the human-sewage 16S reference library used by MST classifiers | — |
| Global sewage resistome | Munk et al. 2022, *Nat Commun* 13:7251 (101 countries); Hendriksen et al. 2019, *Nat Commun* 10:1124 | large harmonised sewage metagenome collection | ARG abundances; country/socio-economic covariates. **No ambient-freshwater WQ sensors** | CC BY |
| European WWTP resistome | Pärnänen et al. 2019, *Sci Adv* 5(3):eaau9124 | European WWTPs | ARG abundances vs clinical AMR prevalence | CC BY-NC |
| CDC NWSS wastewater (SARS-CoV-2) | `data.cdc.gov` resource `j9g8-acpt` | **626,338 rows**, updated 2026-10-16; archived predecessor `2ew6-ywp6` has 837,382 rows across **1,162 WWTP IDs** (archived 2025-09-12) | per-sample concentration, flow rate, population served, county FIPS, method fields, recovery efficiency, human fecal normalisation target | **U.S. Government Works** |
| WastewaterSCAN | `wastewaterscan.org`, dashboard `data.wastewaterscan.org` | "nearly 150 locations across 40 states, serving more than 39 million people" | targets listed: influenza, RSV, SARS-CoV-2, HMPV, norovirus GII, EV-D68, mpox, *Candidozyma auris*, hepatitis A, measles, WNV, rotavirus, parvovirus. **H5/avian influenza not named on the page.** Bulk download link and license **not found on the homepage — UNVERIFIED** | unverified |

### b.5 eDNA datasets

| Dataset | Access | n | Chemistry pairing | License |
|---|---|---|---|---|
| **GBIF DNA-derived occurrences** | `api.gbif.org/v1/occurrence/search?isSequenced=true` | **165,544,817 occurrences** flagged `isSequenced`; 155,309,623 of them `MATERIAL_SAMPLE`; 10,589,626 in the US. Dataset search: **5,211** datasets match "eDNA", 251 match "metabarcoding" | occurrence records, **not abundance tables**; chemistry essentially absent | per dataset (CC0 / CC BY). Publishing guidance: Nilsson et al. 2022, *Metabarcoding and Metagenomics* 6:e84960 |
| **USFS Aquatic eDNAtlas** | `research.fs.usda.gov/rmrs/projects/ednatlas`; Research Data Archive **DOI 10.2737/RDS-2018-0010** (updated 2024-05-17) | "thousands of sites and dozens of species"; ~150 species codes listed, 57 species added May 2024; ~1 km sampling grid across much of the US plus a HighRes layer including Alaska | **species-specific assays only — "we do not accept data from metabarcoding studies."** No paired chemistry | described as open-access; **no license stated on the page**. Full geodatabase download listed as "currently unavailable" |
| **NEON macroinvertebrate / zooplankton metabarcoding** | `DP1.20126.001` (34 sites, 1,066 site-months), `DP1.20221.001` (7 sites, 183 site-months) | CO1 metabarcoding, mid-summer bout sent for sequencing; spring/fall archived at the NEON Biorepository | **collected at the same sites and bouts as NEON chemistry and sonde data** — this is the one eDNA-like product with sensor co-location | CC BY 4.0, token required |
| CALeDNA | Meyer et al. 2021, *California Agriculture* 75(1):20–32 | citizen-science multi-marker inventory | mostly terrestrial/coastal | — |
| Diat.barcode | Rimet et al. 2019, *Sci Rep* 9:15116 | curated diatom barcode library | reference library, not samples | open |

**Note.** NEON has **no eDNA product for fish or amphibians** and no fecal-indicator/human-marker
product. Its molecular products are 16S/ITS (surface water, benthic), shotgun metagenomes, qPCR group
abundances, CO1 metabarcoding (macroinvertebrate, zooplankton), and COI fish tissue barcodes.

---

## (c) Gaps where a SOTA or first-in-class claim is realistic

**c.1 First-in-class and defensible — NEON freshwater molecular → co-located WQ, with site- and
domain-level holdout.** I found **no** published model that predicts NEON aquatic water chemistry or
sonde state from NEON aquatic 16S/metagenomes, and no freshwater microbiome→WQ model at all that is
evaluated with leave-site-out or leave-domain-out splits. The nearest neighbours are soil
(Hermans 2020, 606 sites, random CV), NZ streams (Hermans 2024, 204 sites, **random** 80/20),
Oak Ridge groundwater (Smith 2015, 93 wells, out-of-bag), and a NZ lake-sediment index
(Pearman 2022, correlation, no clear holdout). "First molecular→WQ model evaluated under geographic
holdout on a continental sensor-co-located network" is a claim that is both novel and checkable.
Expect modest absolute skill: the honest prior from Hermans 2024 is R² ≈ 0.02–0.24 for chemistry and
~0.65 accuracy for land-use class under a *random* split, so a domain-level split will be lower.

**c.2 A genuine architectural gap that classical baselines structurally cannot fill — primer-,
release- and study-invariant sequence-level encoders.** RF/XGBoost/TabPFN/MGM all consume a fixed
feature table (ASV/OTU/genus columns). NEON changed 16S primers twice, and NEON explicitly states
OTU identifiers are **not compatible across releases**. A model that encodes each ASV's **sequence**
and aggregates with attention over CLR-weighted abundances (deep sets / set transformer, in the
SetBERT and Pope-et-al. lineage, optionally with DNABERT-S embeddings) can be trained on one primer
set/release and applied to another, where a table-feature model cannot be applied at all without
re-tabulation. This converts "my deep model is 0.02 AUROC better" (unwinnable, per §a.2) into "my
model works in a setting the baselines cannot enter" (winnable and useful). The transfer claims to
test: **cross-primer-set** (V3V4 ↔ V4V5), **cross-release** (RELEASE-2025 OTUs → RELEASE-2026 OTUs),
and **cross-dataset** (NEON → LakePulse / GROWdb / EPA-Ohio).

**c.3 Deep MST with an honest mixture benchmark.** The published MST evaluation standard already
exists (Staley 2018 blinded spikes at 0.025–10% v/v; FEAST's EMP-derived simulations) and the baseline
triple is well defined (SourceTracker2, FEAST, Roguet-RF ~90%). No deep model has been evaluated this
way. Caveat: Staley's central finding is that **local source libraries are required** — a deep model
claiming to remove that requirement (by learning transferable host signatures) would be genuinely
novel, and the experiment to prove it is well specified: train on non-local libraries, test on blinded
local mixtures. Risk: the public blinded-mixture data may not be redistributable in a form that
supports retraining (**UNVERIFIED** — needs a check of Staley 2018's data availability).

**c.4 ARG labels computed on co-located metagenomes — a new label, not a new dataset.** Instead of
borrowing a sewage ARG dataset with no WQ sensors, compute ARG abundance (ARGs-OAP v3.0 / DeepARG /
CARD-RGI) on the **234 NEON surface-water and 736 NEON benthic metagenomes** already in NMDC, then
predict that ARG burden from the **16S profile at the same site-date** plus sensors. That yields a
first-in-class, co-located ARG label for a continental US freshwater network. Predictable targets,
roughly in decreasing order of plausibility: total ARG relative abundance; resistome risk score
(Zhang A-N. 2021 framework); mobile-ARG fraction; source class (the baseline to beat is Li, Yin &
Zhang 2018, *Microbiome* 6:93). What is **not** predictable from these data: ARG *concentrations* in
clinically meaningful units, or human health risk.

**c.5 Disaster-relevant classification targets that are real labels, not keyword strings.** From the
same NEON tables: cyanobacterial dominance (relative abundance of *Microcystis* / *Dolichospermum* /
*Cyanobium* from the 16S table), hypoxia (sonde DO), salinization (SpCond), sediment events
(turbidity), and trophic/eutrophication state (chlorophyll-a + TP/TN). All are thresholds of measured
sensor or lab values at the same site-date, auditable from raw NEON tables, and **none** are derived
from habitat metadata strings. For microcystin specifically, the only co-located molecular+toxin
series I verified is **Lake Mendota, 246 samples, 2013–2018**.

**c.6 Where a SOTA claim is NOT realistic.**
- Human-disease microbiome prediction: three independent large benchmarks say classical wins (§a.2).
- Toxicogenomics cross-study: see §(e).
- eDNA ecological status: ML already is the SOTA and the datasets are 64–152 sites; beating a random
  forest on 100 samples is not a defensible headline.
- Anything evaluated on EMP habitat labels (§b.2).

---

## (d) Recommended approach

### d.1 Primary task — "MicrobeNet-WQ": multi-task molecular → water-quality state on NEON

**Inputs.** Per sample: (1) 16S ASV/OTU table from `DP1.20141.002` (surface water) with CLR
transform; (2) the ASV **sequences** from the same product, for the sequence-level encoder;
(3) optionally ITS; (4) qPCR gene-copy abundances (`DP1.20278.001`) as a biomass covariate;
(5) the PcrAmplification primer columns as an explicit batch indicator.

**Targets (multi-task, partially observed).** Same-site, same-bout values from
`DP1.20093.001` (nutrients, anions/cations, DOC/TN/TP, ANC), `DP1.20288.001` (DO, pH, SpCond,
turbidity, fDOM, chlorophyll-a), `DP1.20033.001` (nitrate), `DP1.20163.001` (chlorophyll/AFDM/CNP).
Masked loss per target so samples with partial labels still train. Secondary classification heads:
hypoxia (DO threshold), trophic class (chlorophyll-a + TP), cyanobacterial dominance, salinization,
turbidity event.

**Join rule (write it down before training, and ablate it).**
`namedLocation` + sampling bout. Grab chemistry: same `collectDate` at the same named location,
tolerance ±1 day, prefer the shared `fieldSuperParent`. Sonde: median over a stated window at the
same named location on the same date (propose ±2 h around collection time, with a ±24 h ablation).
Any sample whose label comes from a different named location is **excluded**, not interpolated. The
review's "same-day nearby-station leakage" finding for microcystin is the failure mode to avoid.

**Splits (the primary result is the domain split).**
1. **Leave-one-domain-out** (19 NEON ecoclimatic domains) — headline.
2. **Leave-one-site-out** (33–34 sites) — secondary.
3. **Temporal**: train ≤2021, test ≥2022.
4. **Leave-one-primer-set-out**: train V3V4, test V4V5 (and reverse) — the transfer claim.
5. **Cross-dataset**: NEON-trained → EPA-Ohio (nutrients), LakePulse (trophic), GROWdb.
Group-aware nested tuning: all preprocessing, feature selection and hyperparameter choice **inside**
each training fold. Report per-fold distributions, never a pooled number.

**Baselines (all of them, on identical folds).**
| Tier | Baseline | Why it must be there |
|---|---|---|
| 0 | Global mean / median; **site mean**; day-of-year + latitude + elevation + site mean | The review showed a day-of-year + latitude baseline reached R² 0.87 for water temperature, beating a custom encoder. Any target where this wins is reported as "not a molecular signal". |
| 0b | Label permutation null per split | Smith 2015's own control: shuffled labels gave AUC 0.49 ± 0.12 vs 0.85 ± 0.008. |
| 0c | **Site-identity probe** — can the model recover site ID from the community? | If yes (it will be), leave-site-out and leave-domain-out are the *only* valid splits, and this must be stated. |
| 1 | RF and XGBoost on CLR of top-k ASVs / genera | The empirically hardest baseline in this field (§a.2). |
| 1b | Elastic net / Ridge on CLR | Ridge tied for best in the 83-cohort benchmark. |
| 1c | Presence–absence RF | Giliberti 2022 and Karwowska 2025 both find P/A ≈ abundance. |
| 2 | **TabPFN** on genus-level CLR | Strongest no-tuning tabular foundation model; fits the ≤5,000-sample budget; watch the weight license (non-commercial for recent checkpoints). |
| 2b | **MGM** fine-tuned on genus profiles | The only published microbiome foundation model with an open pipeline; its own paper reports RF parity on biome classification. |
| 3 | DeepMicro, PopPhy-CNN (and/or MDeep) | Representative published microbiome DL. |
| 4 | Sensors-only model at the same site-dates | Establishes whether molecular data add anything over the cheap modality — required for the fusion claim. |

**Metrics.** Regression: per-target R² and MAE computed per held-out group, plus a **skill score**
relative to the tier-0 metadata baseline (negative skill reported as negative); Spearman ρ.
Classification: AUROC and AUPRC with prevalence stated, plus calibration (reliability curve, Brier).
All with per-fold spread and the permutation null alongside. Pre-register the thresholds.

**Architecture.** Two heads on one trunk, trained together:
- *Table branch*: CLR vector → MLP, with a primer-set embedding and a learned site-free positional
  prior. Purpose: parity with the strong baselines.
- *Sequence-set branch* (the novel part): each ASV → sequence embedding (small k-mer CNN trained from
  scratch, or frozen DNABERT-S), abundance-rank positional encoding as in Pope et al., truncate at
  the top 512 ASVs (Pope reports this affects <6% of samples in their corpus — re-measure on NEON),
  aggregate with a set transformer / attention pooling to a sample embedding, then multi-task heads.
- *Domain-shift handling*: primer-set and release adversarial/gradient-reversal head, or the
  GDmicro-style deep adaptation approach, as an ablation — not as part of the headline unless it helps
  on the domain split.
- Size: target ≤20M parameters. Report parameter counts from the actual state dict (the review found
  parameter counts misstated by ~2× and 20×). 5× RTX 2080 Ti at fp16 is ample; this is a small model.
- Optional self-supervised pretraining on unlabeled freshwater: MGnify freshwater (10,342 samples),
  EMP freshwater (4,915), MiCoDa freshwater subset, NCBI-mined freshwater amplicon runs. Pretraining
  corpus and the evaluation sets must be **disjoint by study** — Pope et al.'s own caveat is that
  their pretraining saw the unlabeled evaluation samples.

**Pre-registered honesty conditions (write these into the plan before any training).**
- If the tier-0 metadata baseline matches the model on a target, that target is reported as
  "explained by season and location", with the number.
- If the sequence-set branch does not beat the table branch on the domain split, say so and keep the
  primer-transfer result as the contribution.
- The deliverable includes a table of **all** targets, including those where skill is ≈0.
- Every reported number traces to a committed script + a saved fold-level results file; no literals
  in figure code (the review found Fig 7 values hard-coded in `paper_figures_v3.py`).

### d.2 Secondary task — deep MST for fecal/sewage contamination

Train a set-level classifier/regressor to recover source proportions from mixed communities. Sources
from published fecal/sewage 16S libraries (Newton 2015 sewage; Roguet 2018 host libraries); sinks from
simulated mixtures following FEAST's protocol (EMP-derived, 10,000 reads, K sources, many mixing
proportions) plus, if redistributable, Staley's blinded spike data. Baselines: SourceTracker2, FEAST,
Roguet-RF, ONN4MST. Metrics: squared Pearson / JSD between estimated and true proportions, unknown-source
error, and detection sensitivity at low spike fractions (Roguet's ~0.5% assemblage / ~0.04% sequence
detection is the bar). Headline claim to attempt: **accuracy retained with non-local source libraries**,
which is the stated weakness of SourceTracker.

### d.3 Fusion and disaster linkage

- **Fusion hook.** The molecular encoder emits a fixed-length embedding tagged with
  (`namedLocation`, `collectDate`). Because NEON collects microbial samples *at the sonde/buoy
  location on the chemistry sampling date*, that embedding joins without interpolation to sonde
  series, grab chemistry, discharge, satellite chips and macroinvertebrate metabarcoding at the same
  site-date. Train the fusion model with **modality dropout** so missing molecular data degrades
  gracefully; report single-modality, pairwise and all-modality scores on the same folds.
- **Cadence limit — state it explicitly.** 16S is monthly in wadeable streams and every other month
  in lakes/rivers; metagenomes are **annual**. Molecular data therefore cannot contribute to
  daily-resolution early warning. Its honest role is (i) a slowly varying site-state descriptor that
  conditions the fast sensor models, and (ii) detection of contamination types that are optically and
  chemically invisible (fecal source, resistome burden, cyanobacterial community composition).
- **Disaster targets, by data route.** Cyanobacterial dominance → NEON lake 16S + chlorophyll-a
  sonde; microcystin → Lake Mendota (246 samples, 2013–2018) only; hypoxia → NEON sonde DO;
  salinization → SpCond; sediment events → turbidity; eutrophication state → chlorophyll + TP/TN;
  **fecal contamination → requires the USGS Great Lakes qPCR releases (10.5066/P9I0SPDQ,
  F7736P45, F7VX0DRH) plus Olds 2018; NEON has no FIB or human-marker product.**
- **Interpretability.** Attention over ASVs gives per-taxon attribution; validate against independent
  indicator lists (Pilgrim 2022 TITAN indicators and change points; Kraemer 2020 indicator ASVs;
  Roguet 2018 host-preferred taxa). An attribution that recovers known indicators is evidence; an
  attribution that does not is a finding, not a failure to hide.

### d.4 Staged plan

1. **Access and provenance.** Obtain a NEON API token; pin one RELEASE; record the release, the
   product versions, and a SHA-256 of every downloaded table. Attribute under CC BY 4.0.
2. **Build the joined table.** Sample × (ASV table, ASV sequences, primer set, site, date, all
   labels). Publish the exact per-sample counts — *replace every site-month figure in §b.1 with real
   sample counts* before any modelling claim.
3. **Sanity audit before training.** Label distributions per site and domain; missing-value and
   sentinel-code screen (the review found unfiltered USGS −999999 codes producing pH MAE of 424
   units); site-identity probe; primer-set vs date confound quantified; duplicate-sample check.
4. **Baselines first, model second.** Tier 0–2 baselines on all five splits before writing the deep
   model. If tier 0 explains a target, drop that target from the headline.
5. **Model, ablations, transfer tests.** Then the fusion integration.
6. **Pre-registration.** Freeze targets, splits, thresholds and metrics in a committed file with a
   hash before the final runs.

---

## (e) Risks and fallbacks

**R1 — NEON access and license.** The data endpoint now returns 403 without a token, and the license
moved from CC0 to CC BY 4.0 (verified). *Mitigation*: create the account, store the token outside the
repo, add the required attribution. If token access is blocked, GROWdb (Zenodo/NMDC/KBase), LakePulse
(ENA `PRJEB38100`), EPA-Ohio (BioProject 592969) and Lake Mendota (`PRJNA846788`) are all open — but
none has NEON's per-sample sensor co-location, so the fusion claim weakens to site-level covariates.

**R2 — Effective sample size and group count.** 1,538 site-months for `DP1.20141.002` across 33 sites
and 19 domains means the domain split has **19 groups**, some with one site. *Mitigation*: report
leave-one-domain-out as a distribution over folds with confidence intervals; add leave-one-site-out;
do not report a single pooled number. If n per fold is too small for regression, fall back to the
classification targets in §c.5.

**R3 — OTU incompatibility across releases, and primer confounding with time.** Both verified from
NEON's own changelog and user guide. *Mitigation*: pin a release; use sequences, not OTU IDs, for the
transfer branch; include primer set as an explicit covariate and run leave-one-primer-set-out. This
risk is also the opportunity in §c.2.

**R4 — The honest result may be "community adds little beyond site and season."** Prior: Hermans 2024
got R² 0.02 for stream N; Kraemer 2020 found land use plus environment explained only ~9–18% of
community composition. *Mitigation*: pre-register this as a legitimate, publishable outcome with the
tier-0 baseline as the comparison, and keep the primer/release transfer result and the multi-task
masked-label protocol as the methodological contributions. This is the framing the review itself
recommends.

**R5 — Deep learning fails to beat RF/XGBoost on CLR.** This is the *expected* outcome from three
independent large benchmarks and from Pope et al.'s own in-domain table. *Mitigation*: the claim is
scoped to (a) the cross-primer / cross-release setting where table models cannot operate, and (b)
multi-task learning over partially observed targets. Report the in-distribution loss to the baselines
plainly if it happens.

**R6 — MST data redistribution.** Blinded known-mixture data may not be available in retrainable
form (**UNVERIFIED**). *Mitigation*: FEAST's simulation protocol is fully specified and reproducible
from EMP, so a simulation benchmark is always available; treat any real blinded data as a bonus
external test.

**R7 — Toxicogenomics.** **Recommendation: do not make this a headline SENTINEL model.** The
evidence: Schüttler et al. 2017 state directly that DEG overlap across independent zebrafish studies
is low and the most common DEGs appear in <50% of treatments across studies; EcoToxChip is 384 genes
across 3 non-aquatic-monitoring species with only 24 chips tested per species; EcoToxXplorer exposes
no public results database; no public aquatic HTTr compendium was found; and SENTINEL's own prior
network scored 0.51 under LOSO. The GEO counts in §a.7 look large but are an upper bound across
incompatible platforms and designs. *If it is pursued at all*, the only reframings with a chance are:
(i) a **single-species, single-platform** compendium (e.g. zebrafish, one microarray generation) with
**pathway-level** features (GSVA/Reactome scores) rather than genes, leave-one-study-out, and the
explicit expectation of near-chance; (ii) **regression of transcriptomic points of departure** within
a designed concentration–time series (Schüttler 2019's design) rather than cross-study classification;
(iii) treat it as a **downstream interpretive module** (predicted stress class → AOP, in the style of
Corsi et al. 2019), clearly labelled as not validated for prediction. Any of these must be presented
as exploratory with a stated chance-level risk.

**R8 — ARG compute and label quality.** ARGs-OAP/DeepARG over 234 surface-water metagenomes is
tractable on 48 cores; 736 benthic metagenomes is heavier and should be staged. ARG burden is
partly a proxy for fecal load, so report the correlation with any available fecal covariate and do
not present ARG prediction as independent of it.

**R9 — Reproducing the old failure modes.** Specific tripwires from the review, each with a guard
already placed above: random splits (→ five group-aware splits, §d.1); keyword-derived labels
(→ measured sensor/lab thresholds only, §c.5); baselines removed from figures (→ baseline table is
part of the deliverable); case studies scored on training samples (→ held-out folds only, with the
fold recorded); parameter counts misreported (→ counted from the state dict); hard-coded figure
values (→ figures read fold-level result files).

---

## (f) References

All entries below were resolved via Crossref, PubMed/PMC, arXiv, or the publisher/repository page on
2026-10-22. Items marked ⚠️ are partially verified as noted.

**Microbiome ML/DL and foundation models**
1. Pope Q, Varma R, Tataru C, David MM, Fern X. Learning a deep language model for microbiomes: The power of large scale unlabeled microbiome data. *PLOS Computational Biology* 21(5):e1011353, 2025. DOI 10.1371/journal.pcbi.1011353. Preprint: bioRxiv 2023.07.17.549267. Data/code: Dryad 10.5061/dryad.tb2rbp08p.
2. Tataru CA, David MM. Decoding the language of microbiomes using word-embedding techniques, and applications in inflammatory bowel disease. *PLOS Computational Biology* 16(5):e1007859, 2020. DOI 10.1371/journal.pcbi.1007859 (correction: 10.1371/journal.pcbi.1008423).
3. Zhang H, Zhang Y, Kang Z, Xiong J, Yang R, Ning K. MGM as a Large-Scale Pretrained Foundation Model for Microbiome Analyses in Diverse Contexts. *Advanced Science* 13(24):e13333, 2026. DOI 10.1002/advs.202513333. Preprint: bioRxiv 2024.12.30.630825. Code: github.com/HUST-NingKang-Lab/MGM (MIT).
4. Mu J, Tang Z-Z, Chen G. Systematic benchmarking of foundation models and classical baselines for microbiome-based disease prediction. Research Square, 2026. DOI 10.21203/rs.3.rs-8912605/v1. **Preprint, not peer reviewed.**
5. Hollmann N, Müller S, Purucker L, et al., Hutter F. Accurate predictions on small data with a tabular foundation model. *Nature* 637:319–326, 2025. DOI 10.1038/s41586-024-08328-6.
6. Hollmann N, Müller S, Eggensperger K, Hutter F. TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second. arXiv:2207.01848, 2022.
7. Ludwig DW 2nd, Guptil C, Alexander NR, et al., Phillips JL. SetBERT: the deep learning platform for contextualized embeddings and explainable predictions from high-throughput sequencing. *Bioinformatics* 41(7):btaf370, 2025. DOI 10.1093/bioinformatics/btaf370.
8. Zhou Z, Wu W, Ho H, et al., Liu H. DNABERT-S: pioneering species differentiation with species-aware DNA embeddings. *Bioinformatics* 41(Suppl):i255–i264, 2025. DOI 10.1093/bioinformatics/btaf188. arXiv:2402.08777.
9. Woloszynek S, Zhao Z, Chen J, Rosen GL. 16S rRNA sequence embeddings: Meaningful numeric feature representations of nucleotide sequences that are convenient for downstream analyses. *PLOS Computational Biology* 15(2):e1006721, 2019.
10. Oh M, Zhang L. DeepMicro: deep representation learning for disease prediction based on microbiome data. *Scientific Reports* 10:6026, 2020. DOI 10.1038/s41598-020-63159-5.
11. Reiman D, Metwally AA, Sun J, Dai Y. PopPhy-CNN: A Phylogenetic Tree Embedded Architecture for Convolutional Neural Networks to Predict Host Phenotype From Metagenomic Data. *IEEE Journal of Biomedical and Health Informatics* 24(10):2993–3001, 2020. DOI 10.1109/JBHI.2020.2993761.
12. Fioravanti D, Giarratano Y, Maggio V, Agostinelli C, Chierici M, Jurman G, Furlanello C. Phylogenetic convolutional neural networks in metagenomics. *BMC Bioinformatics* 19(S2):49, 2018. DOI 10.1186/s12859-018-2033-5.
13. Wang Y, Bhattacharya T, Jiang Y, Qin X, Wang Y, Liu Y, Saykin AJ, Chen L. A novel deep learning method for predictive modeling of microbiome data. *Briefings in Bioinformatics* 22(3):bbaa073, 2020. DOI 10.1093/bib/bbaa073.
14. Sharma D, Paterson AD, Xu W. TaxoNN: ensemble of neural networks on stratified microbiome data for disease prediction. *Bioinformatics* 36(17):4544–4550, 2020. DOI 10.1093/bioinformatics/btaa542.
15. Lo C, Marculescu R. MetaNN: accurate classification of host phenotypes from metagenomic data using neural networks. *BMC Bioinformatics* 20(S12):314, 2019. DOI 10.1186/s12859-019-2833-2.
16. Grazioli F, Siarheyeu R, Alqassem I, Henschel A, Pileggi G, Meiser A. Microbiome-based disease prediction with multimodal variational information bottlenecks. *PLOS Computational Biology* 18(4):e1010050, 2022. DOI 10.1371/journal.pcbi.1010050.
17. Liao H, Shang J, Sun Y. GDmicro: classifying host disease status with GCN and deep adaptation network based on the human gut microbiome data. *Bioinformatics* 39(12):btad747, 2023. DOI 10.1093/bioinformatics/btad747.

**Benchmarks, meta-analyses and evaluation practice**
18. Li P, Li M, Chen W-H. Best practices for developing microbiome-based disease diagnostic classifiers through machine learning. *Gut Microbes* 17(1):2489074, 2025. DOI 10.1080/19490976.2025.2489074.
19. Karwowska Z, Aasmets O, Estonian Biobank research team, Kosciolek T, Org E. Effects of data transformation and model selection on feature importance in microbiome classification data. *Microbiome* 13:2, 2025. DOI 10.1186/s40168-024-01996-6.
20. Kubinski R, Djamen-Kepaou J-Y, Zhanabaev T, et al., Martin RD. Benchmark of Data Processing Methods and Machine Learning Models for Gut Microbiome-Based Diagnosis of Inflammatory Bowel Disease. *Frontiers in Genetics* 13:784397, 2022. DOI 10.3389/fgene.2022.784397.
21. Topçuoğlu BD, Lesniak NA, Ruffin MT 4th, Wiens J, Schloss PD. A Framework for Effective Application of Machine Learning to Microbiome-Based Classification Problems. *mBio* 11(3):e00434-20, 2020. DOI 10.1128/mBio.00434-20.
22. Giliberti R, Cavaliere S, Mauriello IE, Ercolini D, Pasolli E. Host phenotype classification from human microbiome data is mainly driven by the presence of microbial taxa. *PLOS Computational Biology* 18(4):e1010066, 2022. DOI 10.1371/journal.pcbi.1010066.
23. LaPierre N, Ju CJ-T, Zhou G, Wang W. MetaPheno: A critical evaluation of deep learning and machine learning in metagenome-based disease prediction. *Methods* 166:74–82, 2019. DOI 10.1016/j.ymeth.2019.03.003.
24. Pasolli E, Truong DT, Malik F, Waldron L, Segata N. Machine Learning Meta-analysis of Large Metagenomic Datasets: Tools and Biological Insights. *PLOS Computational Biology* 12(7):e1004977, 2016. DOI 10.1371/journal.pcbi.1004977.
25. Vangay P, Hillmann BM, Knights D. Microbiome Learning Repo (ML Repo): A public repository of microbiome regression and classification tasks. *GigaScience* 8(5):giz042, 2019. DOI 10.1093/gigascience/giz042.
26. Dudek NK, Chakhvadze M, Kobakhidze S, Kantidze O, Gankin Y. Supervised machine learning for microbiomics: Bridging the gap between current and best practices. *Machine Learning with Applications* 18:100607, 2024. DOI 10.1016/j.mlwa.2024.100607.
27. Gloor GB, Macklaim JM, Pawlowsky-Glahn V, Egozcue JJ. Microbiome Datasets Are Compositional: And This Is Not Optional. *Frontiers in Microbiology* 8:2224, 2017. DOI 10.3389/fmicb.2017.02224.
28. Roberts DR, Bahn V, Ciuti S, et al., Dormann CF. Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 40:913–929, 2017. DOI 10.1111/ecog.02881.
29. Ploton P, Mortier F, Réjou-Méchain M, et al., Pélissier R. Spatial validation reveals poor predictive performance of large-scale ecological mapping models. *Nature Communications* 11:4540, 2020. DOI 10.1038/s41467-020-18321-y.

**Microbial source tracking**
30. Knights D, Kuczynski J, Charlson ES, et al., Kelley ST. Bayesian community-wide culture-independent microbial source tracking. *Nature Methods* 8:761–763, 2011. DOI 10.1038/nmeth.1650.
31. Shenhav L, Thompson M, Joseph TA, Briscoe L, Furman O, Bogumil D, Mizrahi I, Pe'er I, Halperin E. FEAST: fast expectation-maximization for microbial source tracking. *Nature Methods* 16:627–632, 2019. DOI 10.1038/s41592-019-0431-x. Code: github.com/cozygene/FEAST.
32. An U, Shenhav L, Olson CA, Hsiao EY, Halperin E, Sankararaman S. STENSL: Microbial Source Tracking with ENvironment SeLection. *mSystems* 7(5):e00995-21, 2022. DOI 10.1128/msystems.00995-21.
33. Zha Y, Chong H, Qiu H, Kang K, Dun Y, Chen Z, Cui X, Ning K. Ontology-aware deep learning enables ultrafast and interpretable source tracking among sub-million microbial community samples from hundreds of niches. *Genome Medicine* 14:43, 2022. DOI 10.1186/s13073-022-01047-5.
34. Chong H, Zha Y, Yu Q, et al., Ning K. EXPERT: transfer learning-enabled context-aware microbial community classification. *Briefings in Bioinformatics* 23(6):bbac396, 2022. DOI 10.1093/bib/bbac396.
35. Roguet A, Eren AM, Newton RJ, McLellan SL. Fecal source identification using random forest. *Microbiome* 6:185, 2018. DOI 10.1186/s40168-018-0568-3.
36. Roguet A, Esen ÖC, Eren AM, Newton RJ, McLellan SL. FORENSIC: an Online Platform for Fecal Source Identification. *mSystems* 5(2):e00869-19, 2020. DOI 10.1128/msystems.00869-19.
37. Staley C, Kaiser T, Lobos A, Ahmed W, Harwood VJ, Brown CM, Sadowsky MJ. Application of SourceTracker for Accurate Identification of Fecal Pollution in Recreational Freshwater: A Double-Blinded Study. *Environmental Science & Technology* 52(7):4207–4217, 2018. DOI 10.1021/acs.est.7b05401.
38. Brown CM, Mathai PP, Loesekann T, Staley C, Sadowsky MJ. Influence of Library Composition on SourceTracker Predictions for Community-Based Microbial Source Tracking. *Environmental Science & Technology* 53(1):60–68, 2019. DOI 10.1021/acs.est.8b04707.
39. Henry R, Schang C, Coutts S, et al., McCarthy D. Into the deep: Evaluation of SourceTracker for assessment of faecal contamination of coastal waters. *Water Research* 93:242–253, 2016. DOI 10.1016/j.watres.2016.02.029.
40. Mathai PP, Staley C, Sadowsky MJ. Sequence-enabled community-based microbial source tracking in surface waters using machine learning classification: A review. *Journal of Microbiological Methods* 177:106050, 2020. DOI 10.1016/j.mimet.2020.106050.
41. Ahmed W, Staley C, Sadowsky MJ, et al., Toze S. Toolbox Approaches Using Molecular Markers and 16S rRNA Gene Amplicon Data Sets for Identification of Fecal Pollution in Surface Water. *Applied and Environmental Microbiology* 81(20):7067–7077, 2015. DOI 10.1128/AEM.02032-15.
42. Newton RJ, McLellan SL, Dila DK, et al., Sogin ML. Sewage Reflects the Microbiomes of Human Populations. *mBio* 6(2):e02574-14, 2015. DOI 10.1128/mBio.02574-14.
43. SourceTracker2. github.com/biota/sourcetracker2 (BSD-3-Clause; 90 commits, no published releases, 51 open issues as of 2026-10-22).

**Fecal/sewage field datasets**
44. Olds HT, Corsi SR, Dila DK, Halmo KM, Bootsma MJ, McLellan SL. High levels of sewage contamination released from urban areas after storm events: A quantitative survey with sewage specific bacterial indicators. *PLOS Medicine* 15(7):e1002614, 2018. DOI 10.1371/journal.pmed.1002614.
45. Corsi SR, De Cicco LA, Hansen AM, Lenaker PL, Bergamaschi BA, Pellerin BA, Dila DK, Bootsma MJ, Spencer SK, Borchardt MA, McLellan SL. Optical Properties of Water for Prediction of Wastewater Contamination, Human-Associated Bacteria, and Fecal Indicator Bacteria in Surface Water at Three Watershed Scales. *Environmental Science & Technology* 55(20):13770–13782, 2021. DOI 10.1021/acs.est.1c02644. Data: USGS 10.5066/P9I0SPDQ.
46. Lenaker PL, Corsi SR, McLellan SL, et al., Baldwin AK. Human-Associated Indicator Bacteria and Human-Specific Viruses in Surface Water: A Spatial Assessment with Implications on Fate and Transport. *Environmental Science & Technology* 52(21):12162–12171, 2018. DOI 10.1021/acs.est.8b03481. Data: USGS 10.5066/F7736P45.
47. Dila DK, Corsi SR, Lenaker PL, Baldwin AK, Bootsma MJ, McLellan SL. Patterns of Host-Associated Fecal Indicators Driven by Hydrology, Precipitation, and Land Use Attributes in Great Lakes Watersheds. *Environmental Science & Technology*, 2018. DOI 10.1021/acs.est.8b01945. Data: USGS 10.5066/F7VX0DRH.

**Environmental microbiome → chemistry / water quality**
48. Smith MB, Rocha AM, Smillie CS, et al., Hazen TC. Natural Bacterial Communities Serve as Quantitative Geochemical Biosensors. *mBio* 6(3):e00326-15, 2015. DOI 10.1128/mBio.00326-15.
49. Hermans SM, Buckley HL, Case BS, Curran-Cournane F, Taylor M, Lear G. Using soil bacterial communities to predict physico-chemical variables and soil quality. *Microbiome* 8:79, 2020. DOI 10.1186/s40168-020-00858-1.
50. Hermans S, Gautam A, Lewis GD, Neale M, Buckley HL, Case BS, Lear G. Exploring freshwater stream bacterial communities as indicators of land use intensity. *Environmental Microbiome* 19:45, 2024. DOI 10.1186/s40793-024-00588-z. Data: SRA PRJNA328535.
51. Pilgrim EM, Smucker NJ, Wu H, Martinson J, Nietch CT, Molina M, Darling JA, Johnson BR. Developing Indicators of Nutrient Pollution in Streams Using 16S rRNA Gene Metabarcoding of Periphyton-Associated Bacteria. *Water* 14(15):2361, 2022. DOI 10.3390/w14152361. Data: BioProject 592969; 10.23719/1504034.
52. Hilderbrand RH, Keller SR, Laperriere SM, Santoro AE, Cessna J, Trott R. Microbial communities can predict the ecological condition of headwater streams. *PLOS ONE* 15(8):e0236932, 2020. DOI 10.1371/journal.pone.0236932.
53. ⚠️ Pearman JK, Wood SA, Vandergoes MJ, et al. A bacterial index to estimate lake trophic level: National scale validation. *Science of The Total Environment* 812:152385, 2022. DOI 10.1016/j.scitotenv.2021.152385. **Citation and the r² = 0.842 figure were read from the project's own summary page, not the publisher record — verify before citing.**
54. Tromas N, Fortin N, Bedrani L, Terrat Y, Cardoso P, Bird D, Greer CW, Shapiro BJ. Characterising and predicting cyanobacterial blooms in an 8-year amplicon sequencing time course. *The ISME Journal* 11(8):1746–1763, 2017. DOI 10.1038/ismej.2017.58.
55. Staley C, Gould TJ, Wang P, Phillips J, Cotner JB, Sadowsky MJ. Bacterial community structure is indicative of chemical inputs in the Upper Mississippi River. *Frontiers in Microbiology* 5:524, 2014. DOI 10.3389/fmicb.2014.00524.
56. Berry MA, Davis TW, Cory RM, et al., Denef VJ. Cyanobacterial harmful algal blooms are a biological disturbance to Western Lake Erie bacterial communities. *Environmental Microbiology* 19(3):1149–1162, 2017. DOI 10.1111/1462-2920.13640.
57. Steffen MM, Davis TW, McKay RML, et al., Wilhelm SW. Ecophysiological Examination of the Lake Erie *Microcystis* Bloom in 2014: Linkages between Biology and the Water Supply Shutdown of Toledo, OH. *Environmental Science & Technology* 51(12):6745–6755, 2017. DOI 10.1021/acs.est.7b00856.
58. Simonin M, Voss KA, Hassett BA, et al., Bernhardt ES. In search of microbial indicator taxa: shifts in stream bacterial communities along an urbanization gradient. *Environmental Microbiology* 21(10):3653–3668, 2019. DOI 10.1111/1462-2920.14694.
59. Price JR, Ledford SH, Ryan MO, Toran L, Sales CM. Wastewater treatment plant effluent introduces recoverable shifts in microbial community composition in receiving streams. *Science of The Total Environment* 613–614:1104–1116, 2018. DOI 10.1016/j.scitotenv.2017.09.162.
60. Mansfeldt C, Deiner K, Mächler E, et al., Altermatt F. Microbial community shifts in streams receiving treated wastewater effluent. *Science of The Total Environment* 709:135727, 2020. DOI 10.1016/j.scitotenv.2019.135727.
61. Ulrich N, Kirchner V, Drucker R, et al., Lamendella R. Response of Aquatic Bacterial Communities to Hydraulic Fracturing in Northwestern Pennsylvania: A Five-Year Study. *Scientific Reports* 8:5683, 2018. DOI 10.1038/s41598-018-23679-7.
62. Sagova-Mareckova M, Boenigk J, Bouchez A, et al., Stoeck T. Expanding ecological assessment by integrating microorganisms into routine freshwater biomonitoring. *Water Research* 191:116767, 2021. DOI 10.1016/j.watres.2020.116767.

**Freshwater microbiome datasets and resources**
63. Thompson LR, Sanders JG, McDonald D, et al., Knight R; Earth Microbiome Project Consortium. A communal catalogue reveals Earth's multiscale microbial diversity. *Nature* 551(7681):457–463, 2017. DOI 10.1038/nature24621. Data: ftp.microbio.me/emp/release1/.
64. Shaffer JP, Nothias L-F, Thompson LR, et al., Knight R; EMP500 Consortium. Standardized multi-omics of Earth's microbiomes reveals microbial and metabolite diversity. *Nature Microbiology* 7(12):2128–2150, 2022. DOI 10.1038/s41564-022-01266-x.
65. Gonzalez A, Navas-Molina JA, Kosciolek T, et al., Knight R. Qiita: rapid, web-enabled microbiome meta-analysis. *Nature Methods* 15:796–798, 2018. DOI 10.1038/s41592-018-0141-9.
66. Richardson L, Allen B, Baldi G, et al., Finn RD. MGnify: the microbiome sequence data analysis resource in 2023. *Nucleic Acids Research* 51(D1):D753–D759, 2023. DOI 10.1093/nar/gkac1080.
67. Schmidt TSB, Fullam A, Ferretti P, et al., Bork P. SPIRE: a Searchable, Planetary-scale mIcrobiome REsource. *Nucleic Acids Research* 52(D1):D777–D783, 2024. DOI 10.1093/nar/gkad943.
68. Borton MA, McGivern BB, Willi KR, et al., Wrighton KC. A functional microbiome catalogue crowdsourced from North American rivers. *Nature* 637(8044):103–112, 2025. DOI 10.1038/s41586-024-08240-z.
69. Rohwer RR, Hale RJ, Vander Zanden MJ, Miller TR, McMahon KD. Species invasions shift microbial phenology in a two-decade freshwater time series. *PNAS* 120(11):e2211796120, 2023. DOI 10.1073/pnas.2211796120. Data: SRA PRJNA846788; github.com/rrohwer/limony.
70. Rohwer RR, Kirkpatrick M, Garcia SL, Kellom M, McMahon KD, Baker BJ. Two decades of bacterial ecology and evolution in a freshwater lake. *Nature Microbiology* 10(1):246–257, 2025. DOI 10.1038/s41564-024-01888-3.
71. Paver SF, Newton RJ, Coleman ML. Microbial communities of the Laurentian Great Lakes reflect connectivity and local biogeochemistry. *Environmental Microbiology* 22(2):433–446, 2020. DOI 10.1111/1462-2920.14862. Data: SRA PRJNA591360.
72. Kraemer SA, Barbosa da Costa N, Shapiro BJ, Fradette M, Huot Y, Walsh DA. A large-scale assessment of lakes reveals a pervasive signal of land use on bacterial communities. *The ISME Journal* 14:3011–3023, 2020. DOI 10.1038/s41396-020-0733-0. Data: ENA PRJEB38100.
73. Garner RE, Kraemer SA, Onana VE, et al., Walsh DA. A genome catalogue of lake bacterial diversity and its drivers at continental scale. *Nature Microbiology* 8:1920–1934, 2023. DOI 10.1038/s41564-023-01435-6 (correction 10.1038/s41564-023-01515-7).
74. Huot Y, Brown CA, Potvin G, et al., Walsh DA. The NSERC Canadian Lake Pulse Network: A national assessment of lake health providing science for water management in a changing climate. *Science of The Total Environment* 695:133668, 2019. DOI 10.1016/j.scitotenv.2019.133668.
75. Jurburg S, Arboleda-Baena C, Kazem A, Frøslev T, Jeppesen T. Microbial Community Database (MiCoDa). A curated global 16S rRNA gene amplicon dataset from all environments. GBIF occurrence dataset, 2025. DOI 10.15468/ver9ne (CC BY 4.0). micoda.idiv.de.
76. Qin C, Bartelme R, Chung YA, et al., Zhu K. From DNA sequences to microbial ecology: Wrangling NEON soil microbe data with the *neonMicrobe* R package. *Ecosphere* 12(11):e03842, 2021. DOI 10.1002/ecs2.3842.
77. NEON data products, documents and notices (all accessed 2026-10-22): API `https://data.neonscience.org/api/v0/products`; authentication `https://data.neonscience.org/data-api/authentication/`; "Logins required and data licensing updated", 2026-06-30, `https://www.neonscience.org/impact/observatory-blog/logins-required-and-data-licensing-updated`; *NEON User Guide to Microbe Marker Gene Sequences (DP1.10108.001; DP1.20280.001; DP1.20282.001)*, rev. H.1, 2026-06-15, author Lee Stanish; *NEON User Guide to Microbial Community Taxonomy (DP1.10081.002; DP1.20141.002; DP1.20086.002)*, rev. B, 2026-04-01, author Hugh Cross.
78. Parker SM, Utz RM. Temporal design for aquatic organismal sampling across the National Ecological Observatory Network. *Methods in Ecology and Evolution* 13:1834–1848, 2022. DOI 10.1111/2041-210X.13944 (cited in the NEON 2026-03-20 changelog for revised sampling windows).
79. NMDC studies (accessed 2026-10-22): NEON surface water metagenomes `nmdc:sty-11-hht5sb92` (BioProject PRJNA406977); NEON benthic metagenomes `nmdc:sty-11-pzmd0x14` (PRJNA406976); API `https://api.microbiomedata.org/`.

**eDNA**
80. Cordier T, Esling P, Lejzerowicz F, Visco J, Ouadahi A, Martins C, Cedhagen T, Pawlowski J. Predicting the Ecological Quality Status of Marine Environments from eDNA Metabarcoding Data Using Supervised Machine Learning. *Environmental Science & Technology* 51(16):9118–9126, 2017. DOI 10.1021/acs.est.7b01518.
81. Cordier T, Forster D, Dufresne Y, Martins CIM, Stoeck T, Pawlowski J. Supervised machine learning outperforms taxonomy-based environmental DNA metabarcoding applied to biomonitoring. *Molecular Ecology Resources* 18(6):1381–1391, 2018. DOI 10.1111/1755-0998.12926.
82. Cordier T, Lanzén A, Apothéloz-Perret-Gentil L, Stoeck T, Pawlowski J. Embracing Environmental Genomics and Machine Learning for Routine Biomonitoring. *Trends in Microbiology* 27(5):387–397, 2019. DOI 10.1016/j.tim.2018.10.012.
83. Frühe L, Cordier T, Dully V, Breiner H-W, Lentendu G, Pawlowski J, Martins C, Wilding TA, Stoeck T. Supervised machine learning is superior to indicator value inference in monitoring the environmental impacts of salmon aquaculture using eDNA metabarcodes. *Molecular Ecology* 30(13):2988–3006, 2021. DOI 10.1111/mec.15434.
84. Apothéloz-Perret-Gentil L, Cordonier A, Straub F, Iseli J, Esling P, Pawlowski J. Taxonomy-free molecular diatom index for high-throughput eDNA biomonitoring. *Molecular Ecology Resources* 17(6):1231–1242, 2017. DOI 10.1111/1755-0998.12668.
85. Feio MJ, Serra SRQ, Mortágua A, Bouchez A, Rimet F, Vasselon V, Almeida SFP. A taxonomy-free approach based on machine learning to assess the quality of rivers with diatoms. *Science of The Total Environment* 722:137900, 2020. DOI 10.1016/j.scitotenv.2020.137900.
86. Keck F, Brantschen J, Altermatt F. A combination of machine-learning and eDNA reveals the genetic signature of environmental change at the landscape levels. *Molecular Ecology* 32(17):4791–4800, 2023. DOI 10.1111/mec.17073.
87. Keck F, Blackman RC, Bossart R, et al., Altermatt F. Meta-analysis shows both congruence and complementarity of DNA and eDNA metabarcoding to traditional methods for biological community assessment. *Molecular Ecology* 31(7):1820–1835, 2022. DOI 10.1111/mec.16364.
88. Kelly MG, Juggins S, Mann DG, et al., Walsh K. Development of a novel metric for evaluating diatom assemblages in rivers using DNA metabarcoding. *Ecological Indicators* 118:106725, 2020. DOI 10.1016/j.ecolind.2020.106725 (corrigendum 10.1016/j.ecolind.2020.106890).
89. Vasselon V, Rimet F, Tapolczai K, Bouchez A. Assessing ecological status with diatoms DNA metabarcoding: Scaling-up on a WFD monitoring network (Mayotte island, France). *Ecological Indicators* 82:1–12, 2017. DOI 10.1016/j.ecolind.2017.06.024.
90. Tapolczai K, Keck F, Bouchez A, Rimet F, Kahlert M, Vasselon V. Diatom DNA Metabarcoding for Biomonitoring: Strategies to Avoid Major Taxonomical and Bioinformatical Biases Limiting Molecular Indices Capacities. *Frontiers in Ecology and Evolution* 7:409, 2019. DOI 10.3389/fevo.2019.00409.
91. Rimet F, Gusev E, Kahlert M, et al., Bouchez A. Diat.barcode, an open-access curated barcode library for diatoms. *Scientific Reports* 9:15116, 2019. DOI 10.1038/s41598-019-51500-6.
92. Flück B, Mathon L, Manel S, et al., Pellissier L. Applying convolutional neural networks to speed up environmental DNA annotation in a highly diverse ecosystem. *Scientific Reports* 12:10247, 2022. DOI 10.1038/s41598-022-13412-w.
93. Pont D, Rocle M, Valentini A, et al., Dejean T. Environmental DNA reveals quantitative patterns of fish biodiversity in large rivers despite its downstream transportation. *Scientific Reports* 8:10361, 2018. DOI 10.1038/s41598-018-28424-8.
94. Valentini A, Taberlet P, Miaud C, et al., Dejean T. Next-generation monitoring of aquatic biodiversity using environmental DNA metabarcoding. *Molecular Ecology* 25(4):929–942, 2016. DOI 10.1111/mec.13428.
95. Hänfling B, Lawson Handley L, Read DS, et al., Winfield IJ. Environmental DNA metabarcoding of lake fish communities reflects long-term data from established survey methods. *Molecular Ecology* 25(13):3101–3119, 2016. DOI 10.1111/mec.13660.
96. Carraro L, Mächler E, Wüthrich R, Altermatt F. Environmental DNA allows upscaling spatial patterns of biodiversity in freshwater ecosystems. *Nature Communications* 11:3585, 2020. DOI 10.1038/s41467-020-17337-8.
97. Kuntke F, de Jonge N, Hesselsøe M, Nielsen JL. Stream water quality assessment by metabarcoding of invertebrates. *Ecological Indicators* 111:105982, 2020. DOI 10.1016/j.ecolind.2019.105982.
98. Pawlowski J, Kelly-Quinn M, Altermatt F, et al., Kahlert M. The future of biotic indices in the ecogenomic era: Integrating (e)DNA metabarcoding in biological assessment of aquatic ecosystems. *Science of The Total Environment* 637–638:1295–1310, 2018. DOI 10.1016/j.scitotenv.2018.05.002.
99. Blackman R, Couton M, Keck F, et al., Altermatt F. Environmental DNA: The next chapter. *Molecular Ecology* 33(11):e17355, 2024. DOI 10.1111/mec.17355.
100. Nilsson RH, Andersson AF, Bissett A, et al., Schigel D. Introducing guidelines for publishing DNA-derived occurrence data through biodiversity data platforms. *Metabarcoding and Metagenomics* 6:e84960, 2022. DOI 10.3897/mbmg.6.84960.
101. Meyer RS, Munguia Ramos M, Lin M, et al., Wayne RK. The CALeDNA program: Citizen scientists and researchers inventory California's biodiversity. *California Agriculture* 75(1):20–32, 2021. DOI 10.3733/ca.2021a0001.
102. US Forest Service Rocky Mountain Research Station. Aquatic eDNAtlas. Research Data Archive, DOI 10.2737/RDS-2018-0010 (updated 2024-05-17). `research.fs.usda.gov/rmrs/projects/ednatlas`.

**Antibiotic resistance genes**
103. Hendriksen RS, Munk P, Njage P, et al., Aarestrup FM. Global monitoring of antimicrobial resistance based on metagenomics analyses of urban sewage. *Nature Communications* 10:1124, 2019. DOI 10.1038/s41467-019-08853-3.
104. Munk P, Brinch C, Møller FD, et al., Aarestrup FM. Genomic analysis of sewage from 101 countries reveals global landscape of antimicrobial resistance. *Nature Communications* 13:7251, 2022. DOI 10.1038/s41467-022-34312-7.
105. Pärnänen KMM, Narciso-da-Rocha C, Kneis D, et al., Manaia CM. Antibiotic resistance in European wastewater treatment plants mirrors the pattern of clinical antibiotic resistance prevalence. *Science Advances* 5(3):eaau9124, 2019. DOI 10.1126/sciadv.aau9124.
106. Zhang Z, Zhang Q, Wang T, et al., Qian H. Assessment of global health risk of antibiotic resistance genes. *Nature Communications* 13:1553, 2022. DOI 10.1038/s41467-022-29283-8.
107. Zhang A-N, Gaston JM, Dai CL, et al., Zhang T. An omics-based framework for assessing the health risk of antimicrobial resistance genes. *Nature Communications* 12:4765, 2021. DOI 10.1038/s41467-021-25096-3.
108. Arango-Argoty G, Garner E, Pruden A, Heath LS, Vikesland P, Zhang L. DeepARG: a deep learning approach for predicting antibiotic resistance genes from metagenomic data. *Microbiome* 6:23, 2018. DOI 10.1186/s40168-018-0401-z.
109. Yin X, Jiang X-T, Chai B, et al., Zhang T. ARGs-OAP v2.0 with an expanded SARG database and Hidden Markov Models for enhancement characterization and quantification of antibiotic resistance genes in environmental metagenomes. *Bioinformatics* 34(13):2263–2270, 2018. DOI 10.1093/bioinformatics/bty053.
110. Yin X, Zheng X, Li L, Zhang A-N, Jiang X-T, Zhang T. ARGs-OAP v3.0: Antibiotic-Resistance Gene Database Curation and Analysis Pipeline Optimization. *Engineering* 27:234–241, 2023. DOI 10.1016/j.eng.2022.10.011.
111. Alcock BP, Huynh W, Chalil R, et al., McArthur AG. CARD 2023: expanded curation, support for machine learning, and resistome prediction at the Comprehensive Antibiotic Resistance Database. *Nucleic Acids Research* 51(D1):D690–D699, 2023. DOI 10.1093/nar/gkac920.
112. Li L-G, Yin X, Zhang T. Tracking antibiotic resistance gene pollution from different sources using machine-learning classification. *Microbiome* 6:93, 2018. DOI 10.1186/s40168-018-0480-x.
113. Wang C, Yang H, Liu H, Zhang X-X, Ma L. Anthropogenic contributions to antibiotic resistance gene pollution in household drinking water revealed by machine-learning-based source-tracking. *Water Research* 246:120682, 2023. DOI 10.1016/j.watres.2023.120682.
114. Ju F, Beck K, Yin X, et al., Bürgmann H. Wastewater treatment plant resistomes are shaped by bacterial composition, genetic exchange, and upregulated expression in the effluent microbiomes. *The ISME Journal* 13:346–360, 2019. DOI 10.1038/s41396-018-0277-8.
115. Lee K, Kim D-W, Lee D-H, et al., Cha C-J. Mobile resistome of human gut and pathogen drives anthropogenic bloom of antibiotic resistance. *Microbiome* 8:2, 2020. DOI 10.1186/s40168-019-0774-7.
116. Larsson DGJ, Flach C-F. Antibiotic resistance in the environment. *Nature Reviews Microbiology* 20:257–269, 2022. DOI 10.1038/s41579-021-00649-x.

**Wastewater surveillance**
117. Boehm AB, Wolfe MK, Wigginton KR, et al., Naughton CC. Human viral nucleic acids concentrations in wastewater solids from Central and Coastal California USA. *Scientific Data* 10:396, 2023. DOI 10.1038/s41597-023-02297-7.
118. CDC National Wastewater Surveillance System. "CDC Wastewater Data for SARS-CoV-2", data.cdc.gov resource `j9g8-acpt` (626,338 rows, updated 2026-10-16, U.S. Government Works); archived predecessor `2ew6-ywp6` (837,382 rows, 1,162 WWTP IDs, archived 2025-09-12). Accessed 2026-10-22.
119. WastewaterSCAN. `wastewaterscan.org` / `data.wastewaterscan.org`. ~150 locations in 40 states. **Bulk download and license not located — UNVERIFIED.**

**Toxicogenomics**
120. Schüttler A, Reiche K, Altenburger R, Busch W. The Transcriptome of the Zebrafish Embryo After Chemical Exposure: A Meta-Analysis. *Toxicological Sciences* 157(2):291–304, 2017. DOI 10.1093/toxsci/kfx045.
121. Schüttler A, Altenburger R, Ammar M, et al., Busch W. Map and model—moving from observation to prediction in toxicogenomics. *GigaScience* 8(6):giz057, 2019. DOI 10.1093/gigascience/giz057.
122. Orsini L, Gilbert D, Podicheti R, et al., Frilander MJ. Daphnia magna transcriptome by RNA-Seq across 12 environmental stressors. *Scientific Data* 3:160030, 2016. DOI 10.1038/sdata.2016.30 (erratum 10.1038/sdata.2017.6).
123. Colbourne JK, Pfrender ME, Gilbert D, et al., Boore JL. The Ecoresponsive Genome of *Daphnia pulex*. *Science* 331(6017):555–561, 2011. DOI 10.1126/science.1197761.
124. Basu N, Crump D, Head J, Hickey G, Hogan N, Maguire S, Xia J, Hecker M. EcoToxChip: A next-generation toxicogenomics tool for chemical prioritization and environmental management. *Environmental Toxicology and Chemistry* 38(2):279–288, 2019. DOI 10.1002/etc.4309.
125. Crump D, Hickey G, Boulanger E, et al., Basu N. Development and Initial Testing of EcoToxChip, a Novel Toxicogenomics Tool for Environmental Management and Chemical Risk Assessment. *Environmental Toxicology and Chemistry* 42(8):1763–1771, 2023. DOI 10.1002/etc.5676.
126. Soufan O, Ewald J, Zhou G, et al., Xia J. EcoToxXplorer: Leveraging Design Thinking to Develop a Standardized Web-Based Transcriptomics Analytics Platform for Diverse Users. *Environmental Toxicology and Chemistry* 41(1):21–29, 2022. DOI 10.1002/etc.5251.
127. Davis AP, Wiegers TC, Johnson RJ, Sciaky D, Wiegers J, Mattingly CJ. Comparative Toxicogenomics Database (CTD): update 2023. *Nucleic Acids Research* 51(D1):D1257–D1262, 2023. DOI 10.1093/nar/gkac833.
128. Harrill JA, Everett LJ, Haggard DE, et al., Judson RS. High-Throughput Transcriptomics Platform for Screening Environmental Chemicals. *Toxicological Sciences* 181(1):68–89, 2021. DOI 10.1093/toxsci/kfab009.
129. Olker JH, Elonen CM, Pilli A, et al., Hoff D. The ECOTOXicology Knowledgebase: A Curated Database of Ecologically Relevant Toxicity Tests to Support Environmental Research and Risk Assessment. *Environmental Toxicology and Chemistry* 41(6):1520–1539, 2022. DOI 10.1002/etc.5324.
130. Corsi SR, De Cicco LA, Villeneuve DL, et al., Baldwin AK. Prioritizing chemicals of ecological concern in Great Lakes tributaries using high-throughput screening data and adverse outcome pathways. *Science of The Total Environment* 686:995–1009, 2019. DOI 10.1016/j.scitotenv.2019.05.457.
131. Truong L, Reif DM, St Mary L, Geier MC, Truong HD, Tanguay RL. Multidimensional In Vivo Hazard Assessment Using Zebrafish. *Toxicological Sciences* 137(1):212–233, 2014. DOI 10.1093/toxsci/kft235.
