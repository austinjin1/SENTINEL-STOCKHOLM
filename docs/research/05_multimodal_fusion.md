# 05 — Multimodal fusion with missing modalities and multi-task outputs

*SENTINEL rebuild, research note 05. Prepared 2026-10-22.*

**Scope.** This note covers (a) state-of-the-art fusion under arbitrary missing modalities, including Earth-observation (EO) foundation models; (b) learning from modalities that are rarely co-located, and honest measurement of fusion gain; (c) sparse multi-target learning, distributional outputs, calibration and conformal prediction under shift; (d) environmental and water-quality (WQ) fusion examples and how honestly they were evaluated; (e) a concrete fusion design and evaluation protocol for SENTINEL; (f) risks and fallbacks; and (g) references.

**How claims were verified.** Each paper's metadata was checked against a primary record during this session: the arXiv API or abstract page, the Crossref DOI record, the official proceedings index (PMLR, NeurIPS, ICLR, MLSys, CVF), or the publisher or agency page. Code repositories and licenses were checked with the GitHub API. Quantitative results were taken from fetched abstracts, tables or READMEs, and the location is given (for example "Tab. 2"). NEON facts come from the official NEON API (`data.neonscience.org/api/v0/products/<code>`). Items marked **UNVERIFIED** could not be confirmed from a primary source in this session. Items marked "(search summary)" come only from a search-engine summary of the primary page, so treat them with caution. The web-search budget ran out partway through the session, so some secondary details rest on direct fetches only.

**Internal context.** These SENTINEL results are used as anchors. They come from `REVIEW_REPORT.md` and the files under `results/trackB_fusion/`:
- **Microcystin, gradient-boosted (GBM) 3-way fusion** (satellite band statistics + in-situ pigment + drivers): AUROC 0.823 with site-grouped CV, 0.755 with 5-km spatial blocking, 0.602 with 20-km blocking, and about 0.5 across regions. The actual Perceiver IO scores 0.754 [0.722, 0.782].
  - Leave-one-modality-out (LOMO) deltas: satellite +0.073 [0.044, 0.108], pigment +0.095 [0.062, 0.133].
  - n = 3,649 rows from 227 sites; prevalence of mc > 1.6 µg/L is 17.4% (`fusion_cycle2/modality_contribution.json`).
- **7-day DO new-onset task:** the best single modality reaches 0.946, fusion adds +0.003 (the CI includes 0), and persistence alone reaches 0.929. n = 10,172 rows from 361 sites (`complementary_fusion.json`).

---

## 0. Executive summary: decisions that matter

1. **Expect modest fusion gains, and design to detect them.** Verified fusion gains in strong, well-evaluated systems are small:
   - Combining three meteorological forcings in one LSTM raised median NSE from 0.77 to 0.82 [R159].
   - OmniSat gains +3.7 F1 over its best single modality, and on PASTIS-HD one modality alone *beats* "all" [R63].
   - HEALNet's margin over Perceiver on UCEC is 0.626 vs 0.623, and a unimodal model gets 0.630 [R29].
   - On SENTINEL's own DO task fusion adds +0.003.

   Gains are large only when the modalities are genuinely complementary for the target, as with microcystin.
2. **Gradient boosting and persistence are the opponents to beat, not the Perceiver.** Tree models remain state of the art on medium-sized tabular data (about 10K rows) [R90]. TabPFN beats tuned baselines for up to 10,000 samples and 500 features [R93]. The winners of the HAB-detection challenge used gradient boosting on colour statistics, and CNNs underperformed [R165]. SENTINEL's data are in exactly this regime (10³–10⁴ labelled rows).
3. **Architecture.** Use a Perceiver-style latent bottleneck [R1, R2], adding:
   - modality-specific tokenizers and **modality-specific cross-attention key/value (K/V) weights** (HEALNet-style [R29]), so that a missing modality simply contributes no tokens;
   - continuous-time embeddings;
   - **one query token per (target variable × horizon)**, following Perceiver IO's multitask queries, which matched task-specific tokens (GLUE 81.8 vs 81.0 single-task, [R2] Tab. 2);
   - modality dropout [R14, R15], unimodal auxiliary heads, and teacher distillation [R44] against modality laziness [R40, R41].

   Mixture-of-experts (FuseMoE about 341M parameters; Flex-MoE about 37M with one expert per modality combination [R23, R24]) and product-of-experts VAEs [R30–R33] should be baselines or ablations, not the core.
4. **Encoders.** Prefer frozen, pretrained, *small* encoders where strong ones exist:
   - Galileo nano/tiny/base (0.8/5.3/85M parameters [R60]) or Presto (0.4M parameters, handles masks natively [R61]) for satellite pixel time series;
   - AlphaEarth 64-d annual embeddings as a cheap, strong baseline token [R65].

   Validate *every* encoder against its own GBM or logistic baseline before fusion. The review showed that several of SENTINEL's earlier custom encoders lost to simple baselines.
5. **Data.** The design needs five or more genuinely co-located modalities.
   - **NEON** is the only public source verified here that co-locates sondes, grab chemistry, 16S, metagenomes, macroinvertebrates, periphyton/phytoplankton, fish and airborne imagery at the *same 34 sites*. An NEON API audit finds 616 site-months where sonde + chemistry + 16S + macroinvertebrate + periphyton data all appear in the same month (an upper bound; §e.2).
   - **EPA NLA** (981 lakes in 2022) co-locates cyanotoxins with chemistry, chl-a, benthic macroinvertebrates and zooplankton at a lake visit [R181].
   - **USGS/WQP** is "wide but shallow" and best for pretraining.
6. **Partially co-located data.** Use an always-available **context hub** (weather forcings + catchment attributes + satellite tokens exist for every site-time) as an ImageBind-style anchor [R6]. Combine it with masked multimodal pretraining that treats missing data as masked [R3, R61]. Emergent alignment through a hub is real but weaker than direct pairing: ImageBind scores 17.6 vs 28.4 mAP on AudioSet [R6] Tab. 2. All fusion claims must therefore be tested on genuinely co-located rows.
7. **Losses and uncertainty.** Plan the output side as follows:
   - masked multi-target NLL with **equal weights as the default**, since specialised multi-task optimizers rarely beat the plain sum [R116–R118];
   - Student-t/Gaussian heads on transformed scales with β-NLL [R123], or quantile/mixture heads (CMAL won the hydrology benchmark [R127]);
   - a censored (Tobit) likelihood for values below the detection limit [R130];
   - 5-seed deep ensembles [R128], one seed per GPU;
   - event probabilities derived from the predictive CDF, then isotonic recalibration [R132];
   - conformal intervals stratified by **missing-modality pattern** [R146] and by region, with adaptive conformal for deployment [R140].
8. **Evaluation must be pre-registered.**
   - Comparisons are paired: identical test rows, stratified by modality pattern.
   - Baselines: persistence and climatology, the best single modality, GBM on concatenated features with native NaN handling, out-of-fold stacking of unimodal models, and TabPFN where n ≤ 10k.
   - Splits: 5-km and 20-km spatial blocks, leave-region-out, and leave-years-out.
   - Statistics: site-cluster bootstrap CIs, Diebold–Mariano tests for forecasts, Holm correction across targets, LOMO and Shapley-over-modalities analyses, and shuffled-modality negative controls.

---

## (a) State of the art: fusion under arbitrary missing modalities

### a.1 Taxonomy

The 2024–2026 survey by Wu et al. (TMLR; arXiv v4, Feb 2026) organises missing-modality methods along two aspects, four types and twelve categories [R46]:

- **Data processing**
  - *Modality imputation:* composition (zero, random or retrieved values) or generation (autoencoders, GANs, diffusion).
  - *Representation-focused:* coordinated representations, composition, or generation.
- **Strategy design**
  - *Architecture-focused:* attention-based, distillation-based, graph-based, or multimodal LLMs.
  - *Model combinations:* ensembles, dedicated training, or discrete schedulers.

The survey's open problems include benchmarking and evaluation, streaming/temporal data with missing modalities, and "multimodal AI for natural science", which are exactly SENTINEL's issues.

### a.2 General missing-modality fusion methods

Results are as reported by the authors. Gains are on their own benchmarks and do not transfer automatically.

| Method (venue) | How missing modalities and multiple outputs are handled | Reported result (source) | Code / license | SENTINEL relevance |
|---|---|---|---|---|
| **Perceiver** (ICML 2021) [R1] | Asymmetric cross-attention distils any-length input arrays into a fixed latent bottleneck; cost is linear in input size | Competitive with ResNet-50/ViT on ImageNet without 2-D convolutions (abstract) | DeepMind research repo (Apache-2.0) | Core idea: a variable token set means absent modalities simply supply no tokens |
| **Perceiver IO** (ICLR 2022) [R2] | Adds output **queries**. Modalities get modality-specific embeddings padded to a common channel width (704 in the Kinetics autoencoder) and are concatenated. Queries combine position and task/modality features | Multitask GLUE: multitask query 81.8 vs single-task 81.0 (Tab. 2). Byte-level GLUE 81.0 vs byte-BERT 71.5 at matched FLOPs (Tab. 1). Sintel.final EPE 2.42 vs RAFT 2.57 (Tab. 3) | `google-deepmind/deepmind-research/perceiver` (Apache-2.0) | Use **(target × horizon) queries** for sparse multi-target outputs |
| **MultiMAE** (ECCV 2022) [R3] | Masked autoencoding across modalities and patches. Visible tokens per modality are drawn from Dirichlet(α=1), 98 visible tokens in total. Any subset can be used at fine-tuning | NYUv2 segmentation mIoU: RGB-D 56.0 vs MAE 49.3; depth-only 41.4 vs 23.4 (Tab. 2) | `EPFL-VILAB/MultiMAE` (custom license; check) | Template for masked pretraining on naturally incomplete data |
| **4M / 4M-21** (NeurIPS 2023 / 2024) [R4, R5] | Any-to-any masked modelling over tokenized modalities | (not extracted) | `apple-aiml-research/ml-4m` (Apache-2.0) | Pretraining at this scale is out of reach on 2080 Tis; use as a design reference |
| **ImageBind** (CVPR 2023) [R6] | Each modality is aligned **only to images** (the hub), using frozen OpenCLIP image/text encoders. Alignment between pairs never trained together "emerges" | Emergent zero-shot: NYU-D 54.0 vs 41.9 text-paired baseline; ESC 66.9 vs AudioCLIP 68.6 (AudioCLIP used class-name supervision); but AudioSet 17.6 vs 28.4 mAP for text-paired (Tab. 2). Paired data: about 5K depth, 12K thermal, 510K IMU clips | `facebookresearch/ImageBind` (custom license; check) | The **hub-binding** idea for non-co-located modalities (§b) |
| **LanguageBind** (ICLR 2024) [R7] | Same idea with language as the hub | (not extracted) | — | WQ has no natural language hub; our hub is site-time context |
| **Meta-Transformer** (arXiv 2023, no venue listed) [R8] | One **frozen** modality-shared encoder, modality-specific tokenizers, task heads; 12 modalities, unpaired data | (abstract claims only) | `invictus717/MetaTransformer` (Apache-2.0) | Supports a frozen-core plug-and-play design |
| **Ma et al.**, "Are Multimodal Transformers Robust to Missing Modality?" (CVPR 2022) [R10] | Shows sensitivity to missing modality and that the best fusion layer depends on the dataset. Fix: multi-task training on full/image-only/text-only inputs plus a searched fusion policy | ViLT on MM-IMDb: F1 55.3 with full text falls to 31.2 at 30% text, **below image-only 35.0** (Tab. 1). Method at 10% text: 37.3 vs 23.1 baseline (§5.4) | — | **Key warning:** a multimodal model can fall below the unimodal one when a modality is often missing. Train subset tasks explicitly |
| **SMIL** (AAAI 2021) [R11] | Bayesian meta-learning when up to 90% of training samples miss a modality | (abstract: qualitative) | `deep-real/SMIL` | Severe-missingness regime, as in SENTINEL |
| **Missing-aware prompts** (CVPR 2023) [R12]; **DCP** (NeurIPS 2024) [R13] | Learn prompts per missing pattern on a frozen multimodal transformer | Fewer than 1% of parameters are learnable (abstract) | `YiLunLee/missing_aware_prompts` | Cheap way to add patterns or modalities to a frozen core |
| **ModDrop** (TPAMI 2016) [R14] | Randomly drops whole modalities during training (after modality-wise pretraining) | (not extracted) | — | Canonical modality dropout |
| **HeMIS** (MICCAI 2016) [R15] | Per-modality encoders; fuses the **mean and variance** across whichever modalities are present (variance set to 0 if only one). "Pseudo-curriculum" drops modalities at random after warm-up | All modalities, BRATS DSC 83.15/71.46/75.37 vs mean-filling 82.43/71.46/72.08. Wins 14/15 subsets on BRATS core; not uniformly better (Tab. 3) | — | Simple, robust any-subset pooling; a good ablation |
| **RFNet** (ICCV 2021) [R16]; **mmFormer** (MICCAI 2022) [R17] | Region-aware or transformer fusion for incomplete MRI | (not extracted) | — | Medical analogues |
| **ShaSpec** (CVPR 2023) [R18] | Shared + modality-specific features with distribution-alignment and domain-classification auxiliary tasks; residual fusion | BraTS2018: over 3% (enhancing tumour), 5% (core), 3% (whole) above SOTA (abstract) | `billhhh/ShaSpec` (no license file) | Shared/specific split for the tokens |
| **Distillation / privileged information** [R19–R21] | Teacher on a rich modality transfers to a student on a common modality; MMANet adds margin-aware distillation and modality-aware regularisation | (not extracted) | MMANet (CVPR 2023) | Treat rare modalities (16S, metagenomes) as **privileged information** at training time |
| **MultiMoDN** (NeurIPS 2023) [R22] | Shared state updated **sequentially** by per-modality encoders; any task decoder reads any state; missing encoders are skipped (no imputation) | Matches parallel fusion on 10 tasks (Tab. 1). With 80% MNAR missingness, parallel fusion falls to AUROC 0.385 while MultiMoDN loses about 10% (Exp. 4) | `EPFLiGHT/MultiModN` (MIT) | Robust to **missing-not-at-random** data (sensors fail during storms) |
| **FuseMoE** (NeurIPS 2024) [R23] | Per-modality encoders; mTAND for irregular sampling; learnable "missing" embedding routed to under-used experts; Laplace gating | MIMIC-IV AUROC: in-hospital mortality 81.03 vs 79.65 (HAIM); length of stay 82.91 vs 82.58; phenotyping 71.23 vs 69.55, but phenotyping F1 31.33 vs 42.13 for XGBoost-based HAIM (Tab. 4) | `aaronhan223/FuseMoE` (no license file) | Irregular time series; note the XGBoost baseline wins one metric |
| **Flex-MoE** (NeurIPS 2024) [R24] | Missing-modality bank (one learned embedding per observed-combination × missing modality); each encoder trained only where its modality is observed; S-Router assigns one expert per combination | ADNI accuracy 66.11 vs 61.44 (MAG), but **loses on some subsets** (60.50 vs 63.68). MIMIC-IV 76.81 vs 64.62. 36.9M parameters vs FuseMoE 340.9M (Tabs. 1, 2, 4) | `UNITES-Lab/Flex-MoE` (no license file) | With 8 modalities there are 2⁸−1 = 255 combinations, too many for our sample sizes |
| **M3Care** (KDD 2022) [R25] | Imputes *task-relevant latent* information from **similar samples** (deep-kernel similarity graph + GCN), masking missing modalities | ODIR micro-AUC 0.849 vs 0.809 (SMIL); OV AUROC 0.800 vs 0.754 (HGMF) (Tabs. 2, 3) | `choczhang/M3Care` | Analogue: borrow from similar sites or neighbours |
| **MUSE** (ICLR 2024) [R26] | Patient–modality **bipartite graph** with contrastive objectives; uses samples **without labels** | "MUSE+ … elevates the absolute improvement to ~4%" (abstract) | not found | Missing modalities *and* labels, as in SENTINEL |
| **DrFuse** (AAAI 2024) [R27]; **MedFuse** (MLHC 2022) [R28] | Disentangled shared/specific representations; LSTM-based fusion of EHR + CXR | (not extracted) | — | Clinical analogues |
| **HEALNet** (NeurIPS 2024) [R29] | Shared latent array updated by **per-modality cross-attention** (modality-specific weights); a missing modality's update step is skipped | c-index BLCA 0.668 vs 0.631 (best multimodal) vs 0.606 (unimodal); UCEC 0.626 vs Perceiver 0.623 vs WSI-only 0.630 (Tab. 1) | `konst-int-i/healnet` (Apache-2.0) | **Closest template** for the recommended core |
| **PoE / multimodal VAEs** [R30–R33] | MVAE: product-of-experts posterior over any subset, trained with sub-sampling. MMVAE: mixture-of-experts. MoPoE generalises both | Daunhawer et al.: modality sub-sampling "enforces an undesirable upper bound on the multimodal ELBO" (abstract) | — | Precision-weighted (PoE) late fusion makes a principled baseline |
| **Gated / uncertainty-aware late fusion** [R34–R36] | Gated Multimodal Units (learned gates). TMC: Dirichlet evidence per view, combined with Dempster–Shafer. QMF: uncertainty-weighted dynamic fusion | (abstract claims only) | — | Learned gating over unimodal heads makes a strong, cheap baseline |
| **Incomplete multi-view learning** [R37–R39] | CPM-Nets: latent completion. COMPLETER: contrastive consistency + dual prediction to recover missing views. PVC: partially aligned views | (abstract claims) | — | Recovering missing views as an auxiliary objective |
| **UML, "Better Together"** (arXiv 2025) [R48] | One shared core network trained on **unpaired** samples from several modalities; modality-specific encoders | Stanford Cars fine-tuning 79.45 → 86.39; average 81.54 → 83.99; SUN397 66.20 → 66.03 (Tab. 2). No gain with semantically unrelated modalities (§E.9) | project page only | Unpaired data can help a unimodal target, but only if related |

### a.3 Why fusion often loses to the best single modality

These results apply directly to SENTINEL's earlier failure.

- **Wang, Tran & Feiszli** (CVPR 2020) [R40] report that "the best single-modal network always outperforms the multi-modal network" in their settings. They attribute this to overfitting and to different modalities generalising at different rates, and propose Gradient-Blending.
- **Wu et al.** (ICML 2022) [R41] describe a *greedy* reliance on one modality, measure it with a "conditional utilisation rate", and balance conditional learning speeds.
- **Huang et al.** (ICML 2022) [R43] prove that modality competition can make joint training fail.
- **OGM-GE** (CVPR 2022) [R42] modulates the gradient of the dominant modality on the fly. Code: MIT license.
- **Du et al.** (ICML 2023) [R44] find that late fusion often learns *uni-modal features insufficiently*. A Uni-Modal Ensemble (UME) or a Uni-Modal Teacher (UMT) distillation matches more complex fusion on VGG-Sound, Kinetics-400, UCF101 and ModelNet40.
- **MMPareto** (ICML 2024) [R45] adds unimodal auxiliary losses with Pareto-style gradient integration.

**Implication.** SENTINEL needs (i) unimodal auxiliary heads or teacher distillation, (ii) modality dropout that includes "single-modality-only" episodes, and (iii) the UME (average of unimodal models) as a mandatory baseline.

### a.4 EO multimodal foundation models that handle variable inputs

GPU-fit notes refer to an RTX 2080 Ti (11 GB, Turing): **no bf16**, and FlashAttention-2 does not support Turing. The FlashAttention README lists Ampere/Ada/Hopper for FlashAttention-2, points Turing users to a separate limited repository, and states that bf16 requires Ampere or newer [R58]. Fit statements are estimates and should be measured.

| Model (venue) | Inputs and variable/missing handling | Sizes and pretraining | Reported results (source) | Code, weights, license | 2080 Ti fit (estimate) |
|---|---|---|---|---|---|
| **Galileo** (ICML 2025) [R60] | Space-time inputs: S1, S2, NDVI. Space-only: SRTM, Dynamic World, World Cereal. Time-only: ERA5 precipitation/temperature, TerraClimate, VIIRS. Static: LandScan, location. Tokens are per **channel group**; pretraining samples 2–17 groups per view, with structured and unstructured masking; FlexiViT-style resizable patches | Nano 0.8M / tiny 5.3M / base 85.0M parameters; 200 / 259 / 573 H100-hours (Tab. 12) | Pixel time series, linear probe: Breizhcrops 73.0 vs Presto 63.0, but Togo 74.8 vs Presto 75.5 (Tab. 6). EuroSat kNN 93.0 vs CROMA 85.6 (Tab. 3) | `nasaharvest/galileo` (MIT); weights on GitHub and HF | Yes, all sizes (nano/tiny trivially) |
| **Presto** (arXiv 2023, v4 2024) [R61] | Monthly **pixel time series** of S1, S2 (10 bands), ERA5, NDVI, Dynamic World, SRTM and location, in 10 channel groups. Masking strategies: random, channel-group, contiguous-time, random-time (ratio 0.75). An explicit `mask` input at inference (README) | about 0.40M parameters | CropHarvest F1 0.835 vs random forest 0.441. **Tick Tick Bloom algae severity**, Midwest subset with a *geographically separated* test: RMSE 0.815 fine-tuned vs 0.850 linear vs 1.249 random forest (random-init Presto 0.955) (Tabs. 3, 10) | `nasaharvest/presto` (MIT) | Yes |
| **AnySat** (CVPR 2025) [R62] | JEPA with scale-adaptive patch encoders. Accepts a **dict of any subset** of 11 sensors (aerial 0.2 m, SPOT, NAIP, S2/S1/ALOS/L7/L8/MODIS time series with dates) | "base" variant; GeoPlex: 5 datasets, 11 sensors | State of the art on GeoPlex and 6 external datasets (abstract) | `gastruc/AnySat` (MIT); `flash_attn=False` option | Likely yes (base) |
| **OmniSat** (ECCV 2024) [R63] | Per-modality encoders (CNN for images, LTAE for time series) on georeferenced, aligned patches; random token masking across modalities | — | TreeSatAI-TS weighted F1, all 74.2 vs best single (VHR) 70.5. PASTIS-HD: **S2 alone 70.8 > all 69.9** (Tabs. 2, 3). Multimodal pretraining improves an S2-only linear probe to 44.7/46.3 vs 26.9/29.8 for S2-only pretraining (App. A-1) | `gastruc/OmniSat` (MIT) | Yes |
| **TerraMind** (ICCV 2025) [R64] | Any-to-any generative model with dual scale (token + pixel). Inputs: S2 L1C/L2A, S1 GRD/RTC, DEM, RGB, plus LULC/NDVI and coordinates (model card). "Thinking-in-Modalities" generates missing modalities | Tiny/small/base/large; TerraMesh 9M samples, 500B tokens (HF card) | State of the art on PANGAEA (abstract) | `IBM/terramind` (Apache-2.0); `ibm-esa-geospatial` on HF; TerraTorch | Base for inference: probably; large: tight |
| **AlphaEarth Foundations** (arXiv 2025) [R65] | Embedding field with continuous time. Inference uses S2, Landsat and S1 only; clouds and gaps handled implicitly | About 480M parameters (not needed: embeddings are precomputed). **64-d unit-norm annual embeddings at 10 m, 2017–2024**, in Earth Engine `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` (CC-BY 4.0) | About 23.9% average error reduction vs the next-best featurization (max-trial); 10.4% at ten-shot; 4.18% at one-shot; 15 tasks from 11 datasets | Earth Engine dataset | No GPU needed |
| CROMA (NeurIPS 2023) [R66]; DOFA (arXiv 2024) [R67]; SatMAE (NeurIPS 2022) [R68]; Prithvi-EO-2.0 (arXiv 2024) [R69]; Copernicus-FM (ICCV 2025) [R70]; SkySense (CVPR 2024) [R71] | CROMA: radar–optical contrastive masked autoencoder. DOFA and Copernicus-FM: wavelength/spectrum-conditioned dynamic patch embeddings for arbitrary bands. SatMAE: temporal and spectral masking. Prithvi: multi-temporal HLS | — | (not extracted) | various | Mostly ViT-B/L scale |

**Benchmark reality check.**
- **PANGAEA** [R72] concludes that GFMs "do not consistently outperform supervised models".
  - A UNet trained from scratch is best on HLS Burn Scars (84.51 mIoU), Sen1Floods11 (91.42) and AI4SmallFarms (46.34) (Tab. 5).
  - Optical-only input generally beats SAR or optical+SAR, so multimodal fusion "remains a significant challenge".
  - GFMs gain relative advantage as labels shrink.
- GEO-Bench (NeurIPS 2023 D&B) [R73] is the other standard benchmark.

**Water-specific, non-foundation baselines.**
- **Mixture Density Network (MDN)** chl-a retrieval for Sentinel-2/3 [R74]: trained on 2,943 co-located in-situ matchups. Log-MAE and bias fall 40–60%, and RMSLE and MAPE improve 2–3× vs the baselines (abstract).
- A multi-parameter MDN for Landsat-8, Sentinel-2 and Sentinel-3 [R75].
- **AquaSat**: over 600,000 Landsat–in-situ matchups (TSS, DOC, chl-a, Secchi; ±1 day; 1984–2019) [R77].
- **GLORIA**: 7,572 hyperspectral Rrs spectra from 450 water bodies with co-located WQ [R76].

**Recommendation for the satellite tokenizer.** Use a frozen **Galileo-tiny** or **Presto** over pixel time series around each site (both handle missing inputs by masking), plus the **AlphaEarth annual embedding** as one extra token and an MDN chl-a retrieval as a "physics-informed derived-product" token. Always compare against band-statistics + GBM, which is what won Tick Tick Bloom [R165].

---

## (b) Learning from partially co-located data, and measuring fusion gain honestly

### b.1 The structure of the problem

Write the data as a sparse **sample × modality matrix**. A sample is (site *s*, issue time *t*), and an entry is filled if modality *m* was observed in the look-back window. Public WQ data form a "block-sparse" matrix:
- NEON fills many columns at 34 sites.
- NLA fills many columns at one visit per lake.
- USGS fills sensors and chemistry at thousands of sites.
- EMP-type microbiome studies fill only the microbial column [R176].

**Crucially, three columns are filled everywhere**:
- meteorological forcings (Daymet 1 km daily since 1980; gridMET about 4 km daily since 1979; ERA5-Land about 9 km hourly since 1950) [R187];
- catchment attributes (StreamCat: about 2.65M stream segments and over 600 metrics; LakeCat: 378,088 lakes) [R186];
- satellite imagery and embeddings [R65].

These form a natural **hub**.

### b.2 Strategies, with evidence

1. **Hub binding (ImageBind-style).** Align each modality only with the hub; cross-modal relations "emerge" [R6].
   - *Evidence of limits:* emergent alignment is weaker than direct pairing (AudioSet 17.6 vs 28.4 mAP) [R6] Tab. 2, and contrastive spaces show a "modality gap" [R9].
   - *Use:* pretraining and representation alignment only. Never claim zero-shot cross-modal transfer without testing it on co-located rows.
2. **Unpaired shared-core training (UML).** One shared network, modality-specific encoders, alternating unpaired batches [R48].
   - Helps a unimodal target when modalities share semantics, not otherwise (§E.9).
   - *Use:* lets 16S-only datasets shape the shared core.
3. **Masked modelling with natural missingness.** MultiMAE's Dirichlet modality sampling [R3]; Presto/Galileo channel-group masks, which also allow missing inputs at inference [R60, R61]; AlphaEarth applies valid-data masks only in the loss [R65].
   - *Use:* the self-supervised objective in Stage 2 (§e.4). The real missing pattern is just additional masking.
4. **Modular training where each encoder only sees its own data.**
   - Flex-MoE trains each modality encoder "only on samples where that modality is observed" [R24].
   - MultiMoDN skips missing encoders and is robust under MNAR [R22].
   - HEALNet skips the update step for a missing modality [R29].
5. **Borrowing from similar samples or neighbours.**
   - M3Care imputes task-relevant latents from similar patients [R25].
   - MUSE builds a sample–modality bipartite graph and also uses unlabelled samples [R26].
   - River-network graph models propagate information between sites: a physics-guided recurrent graph model was 33% better (temperature) and 14% better (flow) than a process model, and beat LSTMs by 24% and 14% using 0.1% of the data [R85] (search summary of the abstract). Graph WaveNet generalised better than RGCN in 4 of 5 unseen-condition tests [R86].
   - *Caveat:* SENTINEL's review found that same-day exceedances at other sites alone give AUROC 0.754 (FIB). Neighbour inputs must be timestamped before issue time, and they must be excluded in "unmonitored-site" evaluations.
6. **Neural processes and set-convolution encoders for off-grid, heterogeneous observations.**
   - ConvGNPs map arbitrary context points to a joint Gaussian at arbitrary target locations [R80] (DeepSensor library, MIT).
   - Aardvark Weather encodes satellite, station, ship and radiosonde data with **SetConv layers** that output a gridded value plus a **density channel** marking where data exist. Gridded inputs get mask channels [R81].
   - *Use:* an optional "neighbourhood field" tokenizer so that observations from other sites (any modality) enter as a field, not as row-aligned features.
7. **Statistical data integration.** Integrated species-distribution models fit *one shared latent spatial field* with a joint likelihood over heterogeneous data types collected at different places, using weighted joint likelihoods to balance very different data volumes [R83, R84].
   - This is the classical analogue of hub binding and a good conceptual justification for reviewers.
8. **Incomplete multi-view learning.** Contrastive consistency plus dual prediction to recover missing views [R38]; latent completion [R37]; partially aligned views [R39].
9. **Privileged information and distillation.** Rare modalities can be used *only during training* to improve a model that runs on common modalities [R19–R21].
   - A natural fit for 16S and metagenomes, which are rarely available at prediction time.
10. **Pool heterogeneous data.** In hydrology, deep models trained on pooled, characteristically different regions beat region-specific models ("data synergy") [R87]. Continental single-model ("top-down") training beat bottom-up and grouped models for stream temperature in unmonitored basins [R158].

### b.3 Evaluating fusion gain honestly: the evidence base

- **Random CV inflates spatial skill.**
  - A random forest that "explains more than half" of forest-biomass variance under random CV showed "quasi-null predictive power" under spatial validation [R95].
  - Block CV is "nearly universally more appropriate" for prediction to new data [R94].
  - Tools: blockCV [R96]; area of applicability (AOA) via a dissimilarity index [R97]; kNNDM CV, which matches the nearest-neighbour distances of the prediction task [R99].
  - Counterpoint: when a *probability sample* exists, spatial CV can be pessimistic for map accuracy [R98]. Choose the CV scheme to match the deployment question.
  - SENTINEL's own microcystin AUROC falls from 0.823 (site-grouped) to 0.755 (5-km blocks) to 0.602 (20-km blocks).
- **Strong tabular baselines are mandatory.**
  - Trees win on about 10K-row data across 45 datasets [R90]. Deep learning is "not all you need" [R91]. Results on when neural nets win are mixed [R92].
  - TabPFN outperforms prior methods for up to 10,000 samples and 500 features, at most 10 classes, and handles missing values natively. In 2.8 s it "outperforms an ensemble of the strongest baselines tuned for 4 h" [R93].
- **Variance and testing.**
  - Account for seed, split and hyperparameter variance [R100].
  - Use Wilcoxon or Friedman tests across datasets [R101], the corrected resampled t-test for repeated CV [R102], the Diebold–Mariano test for forecast loss differentials [R103], and block or stationary bootstrap for dependent data [R104, R105].
- **Stacking baselines.** Stacked generalisation [R106] and Super Learner (cross-validated weights over candidate learners) [R107] are the right "late-fusion" comparators.
- **Per-modality contribution.**
  - Perceptual score (permutation-based modality reliance) [R49]; MM-SHAP (Shapley over modalities) [R50]; information-decomposition (redundant, unique, synergistic) [R51].
  - SENTINEL already computes exact Shapley over 3! orders with site-clustered CIs; extend this to more modalities.

---

## (c) Multi-task outputs with sparse targets, uncertainty, calibration and conformal prediction

### c.1 Sparse multi-target training

- **Masked losses.** Compute the loss only on observed (target, horizon) cells. Perceiver IO-style queries make this natural: decode only the queries that have labels [R2].
- Partially-annotated multi-task learning adds cross-task consistency between labelled and unlabelled tasks [R119].
- **Hydrology evidence is mixed.** Multi-task streamflow and temperature helped 56 of 101 sites with single-site models, but only 17 sites with a 101-site model. A naïve auxiliary weight made 47 of 53 sites significantly worse when temperature was the primary variable [R120]. Single-task comparisons are therefore required for each headline target.

### c.2 Loss balancing

| Method | Idea | Evidence |
|---|---|---|
| Uncertainty weighting (CVPR 2018) [R110] | Learn per-task homoscedastic variances that weight the losses | Beat separately trained single-task models (abstract) |
| GradNorm (ICML 2018) [R111] | Equalise gradient norms; one asymmetry hyperparameter α | "Matches or surpasses" exhaustive grid search (abstract) |
| PCGrad (NeurIPS 2020) [R112]; CAGrad (NeurIPS 2021) [R113]; Nash-MTL (ICML 2022) [R114]; FAMO (NeurIPS 2023) [R115] | Resolve conflicting gradients, bargaining, fast adaptive weights | Benchmark gains |
| **Skeptics** [R116–R118] | Unitary scalarisation (plain sum) plus standard regularisation **matches or beats** specialised optimizers. Optimizer methods "do not yield any performance improvements beyond what is achievable via traditional optimization" (Xin et al.). Scalarisation at scale (Royer et al.) | NeurIPS 2022 ×2, NeurIPS 2023 |

**Recommendation.** Use per-target normalisation (each target standardised on its transformed scale), an equal-weight sum as the default, and Kendall uncertainty weighting as the only adaptive variant. Report both. Skip PCGrad-family methods unless a pre-registered ablation shows a gain.

### c.3 Distributional outputs

- **Heteroscedastic Gaussian NLL** [R121, R122] can give "very poor but stable" fits, because high-variance points are down-weighted. β-NLL (loss weighted by σ^{2β}) fixes much of this [R123]. "Faithful" heteroscedastic regression keeps the mean fit equal to an MSE fit [R124].
- **Proper scoring.** Evaluate with CRPS and log score [R125]. Simultaneous quantile regression is one model for all quantiles [R126].
- **Hydrology benchmark.** Among GMM, CMAL (countable mixture of asymmetric Laplacians), UMAL and MC-dropout on 531 CAMELS basins, **CMAL performed best overall and MC-dropout gave the weakest uncertainty estimates** [R127]. Deep ensembles are a simple, strong default [R128]; MC dropout is available [R129] but weaker.
- **Censoring.** Toxins and FIB often fall below detection limits. Use a Tobit-style censored likelihood [R130] (P(y < DL) for left-censored rows), or censored quantile regression networks [R131] (developed for right-censoring; apply by symmetry). Do not substitute DL/2.

### c.4 "Disaster" probabilities

- **Consistency.** Derive P(y > τ) = 1 − F̂(τ) from each regression head's predictive CDF, so that event and concentration predictions agree. Optionally add a dedicated event head and penalise disagreement.
- **Calibration.** Recalibrate with isotonic regression of the CDF on a blocked calibration set [R132]. For classifier-style heads, temperature scaling is a baseline [R133].
- **Scoring.** Use the Brier score [R134] with Murphy's reliability, resolution and uncertainty decomposition [R135], reliability diagrams, and AUPRC for rare events.

### c.5 Conformal prediction under shift and missingness

| Method | Guarantee or idea | SENTINEL use |
|---|---|---|
| Split conformal [R136]; CQR (NeurIPS 2019) [R137]; tutorial [R138] | Finite-sample marginal coverage under exchangeability; CQR adapts to heteroscedasticity | Base layer on top of the quantile heads |
| **Conformal with missing values** (ICML 2023) [R146] | Marginal coverage holds after imputation, but **coverage varies with the missing pattern** and can under-cover some patterns. Missing-data augmentation targets pattern-conditional validity | Calibrate **per modality-availability pattern** (Mondrian groups, merging rare patterns) |
| Weighted conformal under covariate shift (NeurIPS 2019) [R139]; beyond exchangeability (Ann. Stat. 2023) [R142] | Likelihood-ratio or weighted quantiles that are robust to drift | New regions and new years |
| Adaptive conformal inference (NeurIPS 2021) [R140]; AgACI (ICML 2022) [R141]; EnbPI (ICML 2021) [R143] | Long-run coverage under arbitrary shift (ACI) | Online deployment at monitored sites |
| Local spatial conformal (JASA 2024) [R144]; localized conformal (Biometrika 2023) [R145] | Valid intervals without stationarity, from local approximate exchangeability | Unmonitored-site prediction |
| Conformal risk control (ICLR 2024) [R147]; multi-target copula conformal (arXiv 2021) [R148]; Mondrian conformal [R149] | Control expected loss (for example a false-negative rate); joint coverage over targets; group-conditional validity | Alarm-rate control; joint coverage across WQ variables |

### c.6 Recipe

1. Transform targets: log1p or asinh for concentrations and counts, with a censored likelihood below the detection limit.
2. Heads: Student-t or Gaussian with β-NLL (β = 0.5), or 9 quantiles; try CMAL for heavy-tailed toxins.
3. Use an equal-weight masked loss with per-target normalisation, plus a Kendall-weighted variant.
4. Train a 5-seed ensemble, one seed per 2080 Ti.
5. Derive event probabilities from the CDF, recalibrate with isotonic regression, and report Brier/reliability and AUPRC.
6. Apply CQR conformal with Mondrian groups by (modality pattern × region), check coverage per group, and use ACI in deployment.

---

## (d) Environmental and WQ examples of multi-source fusion, with evaluation quality

| Study | Sources fused → target | Validation | Reported added-source or fusion effect | Assessment |
|---|---|---|---|---|
| Kratzert et al., HESS 2021 [R159] | Three forcing products (NLDAS, Maurer, Daymet) + catchment attributes → streamflow (LSTM) | Temporal (test 1989–99, train 1999–2008), 531 basins | Median NSE **0.82 vs 0.77** for the best single product; better in 351/531 basins; Wilcoxon-significant | **Credible, modest fusion gain** over largely redundant sources |
| Rahmani et al., ERL 2021 [R154] | Forcings + attributes ± observed or simulated streamflow → stream temperature | Temporal, 118 basins | Median RMSE about 0.69 °C, NSE 0.985 (search summary). Streamflow input "clearly improved" | A clean added-input test, not spatial |
| Zhi et al., ES&T 2021 [R150] | Hydrometeorology + attributes → DO (LSTM, 236 CAMELS-Chem basins) | Temporal, plus discussion of ungauged basins | Learns the DO-solubility relationship; misses peaks and troughs; more data did not help (abstract) | Honest limits stated |
| Zhi et al., Nature Water 2023 [R152] | Drivers → DO in US rivers | — | Title: "Temperature outweighs light and flow as the predominant driver of dissolved oxygen" | Driver attribution |
| Read et al., WRR 2019 [R155] | Process-model pretraining + LSTM + energy penalty → lake temperature | Temporal; 68 lakes | Median RMSE 1.65 vs 1.78 (DL) vs 2.03 °C (process model) | Gain needs pretraining variability |
| Willard et al., WRR 2021 [R157] | Meta-transfer from 145 monitored lakes → 305 "unmonitored" lakes | **Spatial** (targets held out) | Median RMSE 1.88 °C (ensemble) vs 2.52 °C for an uncalibrated process model | Honest spatial transfer |
| Willard et al., L&O Letters 2022 [R156] | Weather + lake metadata → surface temperature for 185,549 lakes | **Held-out lakes** CV | Median error 1.24 °C, RMSE 1.61 °C vs 2.01 and 1.79 °C for the baselines | Honest spatial evaluation at scale |
| Topp et al., WRR 2023 [R86]; Jia et al., SDM 2021 [R85] | River-network graph models → stream temperature and flow | Unseen-condition partitions | RMSE 1.64 vs 1.65 °C in-distribution; Graph WaveNet better in 4 of 5 shifts. Jia: +33%/+14% vs process model (search summary) | Graph structure aids generalisation |
| Sadler et al., WRR 2022 [R120] | Multi-task flow + temperature | Temporal | Helps some sites only; naïve weighting harmful (47/53) | Cautionary for multi-task learning |
| Feng et al., WRR 2020 [R162] | LSTM + **recent observations** ("data integration") → streamflow | Temporal | Record median NSE 0.86 | Lagged observations (persistence) are a strong "modality" |
| Gauch et al., EMS 2021 [R163] | Tree models vs LSTM vs training-data size | — | "Similarly accurate predictions on small datasets"; LSTM better with more data | Expect GBM ≈ deep at SENTINEL's n |
| Schaeffer et al., J. Env. Manage. 2024 [R164] | Sentinel-3 CI_cyano + precipitation + modelled water temperature + depth/area → weekly bloom exceedance (INLA vs RF/SVM/DNN/LSTM) | 70/30 *random* train/validation split; 2021 out-of-time test; **no spatial holdout, no input ablation** | 2021: AUC 0.95, accuracy 0.90 vs 0.84–0.85 for the best ML/NN | Strong operational result; fusion gain not isolated |
| Tick Tick Bloom (DrivenData 2023) [R165] | S2/Landsat colour statistics + HRRR weather + elevation + location → cyanobacteria severity | Region-averaged RMSE on a private test set | Winning RMSE 0.76. **Gradient boosting dominated; CNNs had "very high RMSEs"**. Temperature and humidity helped; precipitation and wind did not | Real-world evidence that GBMs win at this n |
| Presto on Tick Tick Bloom (Midwest) [R61] | Pretrained pixel-time-series encoder | **Geographically separated test**, 777 training samples | RMSE 0.815 vs 0.850 (linear) vs 1.249 (random forest) | Pretraining matters (random init 0.955) |
| Fournier et al., Water Research 2024 [R166] | Multiparameter sondes (phycocyanin + others), with gaps → phycocyanin, 4–28-day forecasts | Temporal; **skill vs persistence** | Multivariate LSTM best (vs linear and random forest); up to 90% accuracy at the 10 µg/L phycocyanin alarm level; positive skill 16–28 days ahead | Good practice: skill relative to persistence |
| Rodríguez-López et al., Remote Sensing 2023 [R167] | Field vs meteorological vs satellite input cases → chl-a (SARIMAX/LSTM/RNN), Lake Llanquihue | Single lake | R² 0.64–0.93 across cases; LSTM best (search summary) | Per-case gains not verified here |
| Tang et al., IEEE JSTARS 2022 [R168] | Satellite + physics-based simulation → algal bloom maps (ConvLSTM), Lake Okeechobee | — | Hybrid data "improve performance" (search summary of abstract) | No single-source table seen |
| Adil et al., arXiv 2025 [R169] | CyAN CI + MODIS temperature (30% and 90% missing) → CyanoHAB intensity (Transformer + BiLSTM) | Not stated in abstract | F1 89.5% at 1 day, 78.9% at 14 days (AUC 82.6%) | No baselines in abstract |
| Smith et al., mBio 2015 [R170] | **16S communities** → 26 geochemical variables (93 wells), contamination detection | Within site | "Accurately identify" uranium and nitrate; authors call for tests of geographic generalisability | The microbial modality carries a signal |
| Cordier et al., ES&T 2017 [R171] | eDNA metabarcoding + supervised ML → marine ecological quality status | — | (title-level claim) | The biology modality carries a signal |
| Beversdorf et al., PLoS ONE 2015 [R172] vs Vaitomaa et al., AEM 2003 [R173] | mcy gene abundance → microcystin | 4 Wisconsin lakes / 2 Finnish lakes | **No significant correlation after correction** / positive correlation of summed mcyE with microcystin | **Mixed:** the "metagenomics will fix transfer" hypothesis is unproven |
| Searcy & Boehm, ES&T 2023 [R174] | High-frequency FIB sampling + tides, temperature, waves → beach FIB forecasts | Season-ahead (search summary) | Models from 1–2 days of sampling stayed accurate for a season (search summary) | — |
| **SENTINEL (internal)** | Satellite + pigment + drivers → microcystin | Site-grouped / 5-km / 20-km / region | GBM 0.823 / 0.755 / 0.602 / about 0.5; Perceiver 0.754 | Fusion gain is real but local |

**Reviews.**
- The 2024 Nature Water review argues that deep learning is under-used for WQ and can fill spatio-temporal gaps [R151]. Varadharajan et al. survey ML for river WQ, including data integration and uncertainty [R153].
- The cyanobacteria-forecasting systematic review reportedly covers over 120 studies, most site- and species-specific, with low comparability (as described by [R166]; the original [R175] was not fetched).

**Synthesis.**
- Verified gains from adding sources in hydrology and WQ are typically **small to moderate**: about +0.05 NSE in the multi-forcing LSTM; positive but site-dependent for multi-task learning.
- Spatial or unmonitored-site validation is less common than temporal validation. Many HAB studies use random or within-lake splits. Persistence and climatology are strong.
- We found **no** verified study of a deep, arbitrary-missing-modality fusion model for WQ that was evaluated against GBM under spatial blocking. That gap is SENTINEL's opportunity, provided the evaluation is airtight.

---

## (e) Recommended SENTINEL fusion design and evaluation protocol

### e.1 Design principles derived from the evidence

1. Missing modalities are **absent tokens**, never imputed raw values [R1, R15, R22, R29].
2. Every modality must be able to **stand alone** (unimodal heads or teachers), so fusion cannot fall below unimodal performance [R10, R40, R44].
3. Pretrained frozen encoders plus a small fusion core (5–20M parameters) suit the 10³–10⁴ labelled rows and 11 GB GPUs. Data volume, not architecture, is the binding constraint.
4. The **context hub** (forcings, catchment, satellite) is present for every row and binds non-co-located modalities [R6, R83].
5. Evaluation is designed **before** training, and GBM, stacking and TabPFN are first-class competitors [R90, R93, R106].

### e.2 Data plan: getting five or more genuinely contributing modalities

**Tier D ("deep", NEON, 34 aquatic sites).** Facts come from the NEON product API, retrieved 2026-10-22.

| Modality | NEON product (code) | Cadence and design (official description) |
|---|---|---|
| In-situ sonde: SpC, DO, pH, turbidity, fDOM, chl-a | Water quality (DP1.20288.001) | 1-min in wadeable streams, 5-min in lakes and rivers; 34 sites; 2016-03 → 2026-09 |
| Nitrate sensor | DP1.20033.001 | 15-min averages; 34 sites |
| Hydrology | Continuous discharge (DP4.00130.001); surface water elevation (DP1.20016.001) | 28 sites / 34 sites |
| Lab chemistry: nutrients, anions, cations, DOC | Chemical properties of surface water (DP1.20093.001) | Streams and rivers **15–26×/yr** (12 regular + up to 14 targeted at flow events); lakes **12×/yr**; 34 sites; 2012 → 2026 |
| Dissolved gases (CO₂, CH₄, N₂O) | DP1.20097.001 | Grab samples; 34 sites |
| Microbial 16S/ITS | Surface water microbe marker gene sequences (DP1.20282.001) | **Same time and location as chemistry**; monthly in wadeable streams (12×/yr), every other month in lakes and rivers (6×/yr) |
| Metagenomes (functional genes, e.g., toxin genes) | DP1.20281.001 | **Once per year** near peak productivity, with chemistry |
| Benthic macroinvertebrates | DP1.20120.001 | 3×/yr (spring, summer, fall); 34 sites |
| Periphyton/phytoplankton: chl-a, C, N, P | DP1.20163.001 (chemistry); DP1.20166.001 (collection/taxonomy) | 3×/yr. **DP1.20166.001 taxonomy was suspended in Jan 2026; morphological taxonomy runs only through 2020** |
| Fish | DP1.20107.001 | 2×/yr; 28 sites |
| Zooplankton | DP1.20219.001 | 7 lake sites |
| Airborne hyperspectral (426 bands) and RGB | DP3.30006.001; DP3.30010.001 | Planned yearly at peak greenness, covering at least 80% of a 10 km × 10 km box; 1 km tiles (59 site codes, including aquatic sites such as ARIK, BLUE, HOPB, LIRO, PRIN, SYCA) |
| Satellite + forcings + catchment (hub) | S2/S1/S3 (Galileo/Presto/AlphaEarth); Daymet/gridMET/ERA5-Land; StreamCat/LakeCat | Everywhere |

**Co-location audit** from the NEON API's `availableMonths`, by site-month:
- 2,590 site-months have at least 5 of 8 non-discharge products, across all 34 sites.
- **616 site-months** have sonde, chemistry, 16S, macroinvertebrates and periphyton chemistry all present.
- The sonde co-occurs with chemistry in 3,214 site-months and with 16S in 2,829.

These are **upper bounds**. Month-level co-occurrence is not same-day, and product availability can include metadata-only records; for example, the metagenome tables are shared with other genetic products. Confirm actual sequenced-sample counts after download with `neonutilities` (PyPI 2.0.2, AGPL-3.0) [R180].

**Tier D site types.** Inferred from the product site lists, the 34 sites are 24 wadeable streams (sites with the stream PRT-temperature product), 7 lakes (BARC, CRAM, LIRO, PRLA, PRPO, SUGG, TOOK, the zooplankton sites) and 3 large rivers (BLWA, FLNT, TOMB).

**Tier D targets.** DO, temperature, pH, SpC, turbidity, nitrate and chl-a at h = 0, 1, 3, 7 and 14 days; lab TN, TP and DOC; and events such as hypoxia, turbidity spikes, nitrate and salinisation. No standard NEON cyanotoxin product was found in this session (**UNVERIFIED** absence).

**Tier C ("cyanotoxin", EPA NLA, plus WQP and SENTINEL's existing microcystin set).** NLA 2022 sampled **981 lakes**, representing about 268,020 lakes, and reports:
- chl-a, benthic macroinvertebrates and zooplankton;
- atrazine, DO, total N and P;
- microcystins and cylindrospermopsin;
- enterococci and fish-tissue contaminants.

Microcystin was detected in **50%** of lakes in 2022 (21% in 2017) and exceeded the EPA recreational criterion in **2%** [R181]. Combined with S2/S3 and CyAN [R185], LakeCat [R186] and forcings, NLA yields at least 5 co-located modalities per visit: chemistry, pigment, biology (macroinvertebrates, zooplankton), satellite, and weather/catchment. Survey cycles: 2007, 2012, 2017, 2022. The NARS indicator list also includes "eDNA" and "Cyanobacteria", but which survey and year use them is **UNVERIFIED** [R183].

**Tier W ("wide", USGS NWIS + WQP + CAMELS-Chem + AquaSat/GLORIA).**
- Sensors + discrete chemistry + FIB + satellite + forcings + catchment give 5 modalities at many sites, without microbes or biology. Use this tier for pretraining and for the FIB and nutrient tasks.
- **Data warning:** the WQP's legacy WQX 2.2 profiles "do NOT contain USGS data added after March 11, 2024". Use the WQX 3.0 (beta) profiles [R184].
- CAMELS-Chem: 516 catchments, 18 constituents, 1980–2018 [R177]. GRQA: 42 parameters, over 17M measurements, 1898–2020 [R178].

### e.3 Architecture: "SENTINEL-Fuse v2"

**Tokens.** Every input is a set of tokens. Each token has a content embedding, plus a modality embedding, a variable or channel embedding, a **continuous Δt embedding** (sinusoidal over log Δt, as in SeFT/mTAND [R52, R53]), a cadence and QC flag, and optional depth. Absent modalities produce no tokens.

| Modality | Tokenizer (frozen or fine-tuned) | Tokens per sample (illustrative) | Pretraining data |
|---|---|---|---|
| Sonde and nitrate sensors | Per-variable daily patches (mean, min, max, last value, fraction missing) over a 60–90-day window → MLP; or SeFT-style observation tokens [R52] | ≤ 8 variables × 9–13 weekly patches | USGS NWIS + NEON (masked reconstruction or forecasting) |
| Lab chemistry | One token per analyte observation (log value, censor flag, detection limit, Δt) | 5–60 | WQP, CAMELS-Chem, NEON |
| Satellite | **Frozen Galileo-tiny or Presto** on monthly pixel or patch time series around the site (water-masked), plus an AlphaEarth annual 64-d token, plus an MDN chl-a token | 6–12 + 1 + 1 | Pretrained (no SENTINEL training) |
| Forcings | Weekly patches of Daymet/gridMET (past window only, unless running a declared "perfect-forcing" experiment) | about 9–13 | — |
| Catchment and static | StreamCat/LakeCat (and CAMELS) → PCA → 1–2 tokens (entity-aware conditioning as in EA-LSTM [R160]) | 1–2 | — |
| Microbial 16S | Genus-level CLR → PCA (fit on training data only) → 2–4 tokens, plus explicit indicator-taxa tokens (e.g., *Microcystis*). Validate against logistic/GBM baselines first | 3–6 | NEON + EMP (masked taxa) [R176] |
| Metagenome | Functional-gene abundance tokens (e.g., toxin-synthesis, N-cycling genes) | 2–4 | NEON |
| Biology (macroinvertebrates, periphyton, fish, zooplankton) | Community vector → indices + embedding | 2–4 each | NEON, NLA |
| Airborne/drone imagery | Frozen image encoder (AnySat aerial, TerraMind, or DOFA) on chips → 1–4 tokens per year | 1–4 | Pretrained |
| Neighbourhood (optional) | SetConv/ConvGNP-style field of recent observations at other sites, with flow-connectivity and distance [R80, R81, R85] | 4–16 | — |

**Fusion core.** A HEALNet/Perceiver hybrid:
- 64 latents × d = 256;
- per-modality cross-attention into the latents, with **modality-specific K/V projections**;
- 4–6 latent self-attention blocks (weights optionally shared across depth);
- a **query decoder** with one query per (target × horizon) [R2], plus event queries.

Estimated size: about 5–10M parameters plus tokenizers. With fp16 AMP and loss scaling [R57], and PyTorch's built-in attention (not FlashAttention-2), batches of 64–128 should fit in 11 GB. Measure this.

**Output heads.** Per target family:
- Student-t or Gaussian on the transformed scale, trained with β-NLL, with a censored likelihood below the detection limit;
- optionally 9 quantiles or CMAL;
- event probabilities from the CDF.

**Missingness and imbalance controls.**
- *Modality dropout schedule.* Warm up with all available modalities. Then, per sample:
  - 20%: keep all;
  - 20%: keep exactly one modality, plus the hub;
  - otherwise: drop each modality independently with p = 0.3–0.5, keeping at least one.

  This follows HeMIS's pseudo-curriculum [R15] and ModDrop [R14].
- *Unimodal auxiliary heads.* Each modality's tokens alone go through the same decoder, sampled one modality per batch.
- *Distillation* from the best unimodal teacher (UMT [R44]); the teacher can be a GBM.
- *OGM-GE* [R42] only if a dominant modality is observed to starve the others.
- *Batch sampling* stratified by availability pattern, so hub-only rows do not dominate.

**Why not the alternatives as the core.**
- Flex-MoE needs one expert per combination (255 for 8 modalities) and enough rows for each [R24].
- FuseMoE is large [R24] Tab. 4.
- PoE posteriors are prone to over-confident experts, and multimodal VAEs face the sub-sampling bound [R33].

Keep them as **baselines**: precision-weighted (PoE) late fusion of unimodal Gaussians; a gated mixture of unimodal heads [R34, R35]; and MultiMoDN-style sequential fusion for MNAR robustness [R22].

### e.4 Training curriculum

- **Stage 0 — Audit and baselines (no deep learning).**
  - Build the sample × modality matrix and the co-location statistics.
  - Freeze the splits (§e.6) and fit persistence, climatology, per-modality GBMs, GBM on concatenated features (native NaN), out-of-fold stacking (Super Learner), UME, and TabPFN (n ≤ 10k).
  - Register the primary endpoints.
- **Stage 1 — Unimodal encoders.** Use frozen pretrained models for satellite and imagery. Pretrain the sensor, chemistry and microbial tokenizers by masked reconstruction or forecasting on unimodal data (USGS, EMP, NEON). **Gate:** each encoder must match or beat its own GBM/logistic baseline on its own task under the same blocked split, or it is replaced by engineered features.
- **Stage 2 — Hub alignment and masked multimodal pretraining**, on all site-time samples with or without labels:
  - MultiMAE-style masked prediction of modality summaries from the remaining tokens, with Dirichlet sampling over modalities [R3];
  - plus contrastive alignment of each modality summary to the hub summary at the same site-time (ImageBind-style) [R6].
- **Stage 3 — Supervised fine-tuning.**
  - Masked multi-target NLL, equal weights (plus a Kendall variant), with the dropout schedule and unimodal heads.
  - Encoders stay frozen, or get LoRA/adapters [R55, R56].
  - Train 5 seeds, one per GPU.
- **Stage 4 — Calibration and conformal.** Isotonic CDF recalibration on a blocked calibration fold, then CQR with Mondrian groups (pattern × region), then a coverage audit.
- **Stage 5 — Pre-registered evaluation** (§e.6), run once on the final test sets.

### e.5 Plug-and-play protocol for a new modality

1. Define a featurizer, a frozen or pretrained encoder, and a projection into d = 256, plus a new modality embedding and **new modality-specific cross-attention K/V weights**.
2. Freeze the fusion core. Train only the new components:
   - supervised, on rows where the new modality co-occurs with labels;
   - self-supervised, by hub alignment (Stage-2 losses) where it does not.

   Optionally add missing-aware prompts or LoRA on the latent blocks (fewer than 1% of parameters [R12]).
3. **Regression guarantee.** When the new modality is absent, it contributes no tokens and the core is unchanged, so predictions on old rows are bit-identical. Verify this numerically.
4. **Acceptance test.** On paired, blocked validation rows where the new modality is present, ΔCRPS (or ΔAUPRC) must exceed 0 with a site-bootstrap CI excluding 0 for at least one target, with no target worse than a pre-set non-inferiority margin. Otherwise the modality is logged as "non-contributing".

### e.6 Evaluation protocol for a skeptical reviewer (pre-registered)

1. **Questions and endpoints**, written down before final training:
   - **P1 (NEON).** 7-day-ahead CRPS skill for DO, turbidity, nitrate and chl-a vs (a) the best single modality and (b) GBM on concatenated features, under leave-site-out.
   - **P2 (cyanotoxins).** Microcystin exceedance AUPRC and AUROC under 5-km and 20-km blocks and leave-region-out. HUC codes must be parsed as **strings**, since the review found float parsing split and merged regions.
   - **P3 (≥5 modalities).** At least 5 modalities each have a positive LOMO ΔCRPS, with CI excluding 0, on paired rows.
   - **P4 (graceful degradation).** For every availability pattern with enough rows, the fusion model is at least as good as the baseline trained on that pattern's modalities.
   - **P5 (calibration).** Conformal coverage within ±3 points of nominal per pattern group and per region.
2. **Splits.**
   - Spatial: leave-site-out (NEON; also by NEON domain), 5-km and 20-km agglomerative clusters, and HUC-2/HUC-4 leave-region-out.
   - Temporal: forward-chaining by year, plus a final untouched period (e.g., 2024–2026) evaluated once.
   - Combined space-time holdouts.
   - Use kNNDM to match the deployment distance distribution [R99], and report the area of applicability [R97].
3. **Leakage rules.**
   - Every input timestamp is before the issue time.
   - No same-day neighbour observations from the test block in "unmonitored" mode.
   - Normalisation, PCA and CLR are fit on training folds only.
   - No labels derived from inputs (the review found a circular |z| > 3 label).
   - No model or hyperparameter selection on test folds (the review found best-of-N leave-one-study-out selection).
4. **Paired comparisons.** All models are scored on identical rows. Results are reported (a) overall, (b) by availability pattern, (c) on rows with ≥2 and ≥5 modalities. The "best single modality" is chosen on validation, and the test-set oracle-best single is also reported as a conservative comparator.
5. **Baselines with equal tuning budgets** [R100]: persistence and climatology (the review found a day-of-year + latitude baseline at R² 0.87 for temperature); per-modality GBM and deep models; GBM on concatenated features; Super Learner stacking [R107]; UME; PoE late fusion; TabPFN [R93]; and the published model types (LSTM, EA-LSTM [R160]).
6. **Metrics.**
   - Regression: CRPS, RMSE/MAE on the transformed scale, and NSE/KGE for series.
   - Events: AUPRC, AUROC, Brier score with Murphy decomposition, reliability, and lead time at a fixed false-alarm rate measured against **random clean dates** (the placebo test from the review).
   - Intervals: coverage and width per group.
7. **Statistics.**
   - Site-cluster bootstrap (B = 2,000) of *paired* metric differences, plus block bootstrap in time [R104, R105].
   - Diebold–Mariano tests per site, aggregated [R103].
   - Holm correction across primary endpoints; at least 5 seeds, with seed variance included in the CIs [R100].
   - A Friedman/Wilcoxon summary across tasks [R101].
   - Effect sizes always reported with n rows, n sites and prevalence.
8. **Modality contribution.**
   - Inference-time LOMO and **exact Shapley over all 2^M subsets** (feasible because the model is trained with dropout).
   - Retrain-based LOMO for headline claims.
   - **Shuffled-modality negative controls**: permute a modality across rows within a block, then retrain.
   - Perceptual-score-style permutation reliance [R49]. Report redundancy vs synergy where possible [R51].
9. **Robustness.**
   - Performance vs number of available modalities (should not decrease).
   - MNAR stress: drop a modality preferentially at extremes [R22].
   - Region shift, with conformal coverage under shift.
10. **Reporting.** A "claims ledger" maps every number to a script, config hash and JSON file. Negative results are kept, and the code and splits are released.

**What counts as success.** P1 or P2 shows a significant paired gain over **both** the best single modality and GBM on concatenated features under spatial blocking, and P3 holds for ≥5 modalities on NEON. If deep fusion only ties GBM, the honest headline is "a missing-modality-robust deep fusion matches strong GBM fusion and adds calibrated, any-subset inference", with the comparison table shown.

### e.7 Order of work, with go/no-go gates

1. Stage 0 audit and baselines on NEON, NLA and the existing microcystin set. **Gate:** the co-location counts and power are adequate. Use a site bootstrap of baseline residuals to estimate the detectable ΔCRPS.
2. Satellite tokenizer bake-off: Galileo-tiny vs Presto vs AlphaEarth vs band-statistics GBM, on chl-a and microcystin under blocking.
3. Sensor and chemistry tokenizers; check them against persistence.
4. A fusion core with the hub + 2 modalities. **Gate:** at least as good as GBM on concatenated features on paired rows.
5. Add modalities one at a time via the §e.5 protocol, which also demonstrates plug-and-play.
6. Calibration and conformal; then the final pre-registered test.

---

## (f) Risks and fallbacks

| Risk | Evidence | Mitigation / fallback |
|---|---|---|
| Deep fusion loses to GBM at n ≈ 10³–10⁴ | [R90, R93, R163, R165]; SENTINEL's Perceiver 0.754 vs GBM 0.823 | Hybrid: use deep encoders as feature extractors and fuse with GBM or stacking; report TabPFN; a tie with better any-subset robustness and calibration is still a contribution |
| Too few co-located rows for ≥5-modality claims | 616 site-months upper bound at NEON; biology 2–3×/yr | Restrict ≥5-modality claims to NEON and NLA; pool targets via multi-task learning; use hub pretraining; pre-compute power |
| Spatial leakage / autocorrelation | [R94, R95]; review: 40% of rows have same-day neighbours within 5 km | Blocking at multiple radii; neighbour exclusion; AOA; kNNDM |
| Greedy modality / imbalance | [R40–R45] | Unimodal heads, UMT, single-modality episodes in the dropout schedule, OGM-GE |
| Hub (emergent) alignment is weak | ImageBind AudioSet 17.6 vs 28.4 [R6]; modality gap [R9] | Use it only as pretraining; all claims come from supervised co-located evaluation |
| Microbial modality adds no transferable toxin signal | Mixed mcy-gene evidence [R172, R173]; metagenomes only 1×/yr | Treat as a pre-registered hypothesis; use it as privileged information [R21]; report nulls |
| EO foundation models don't beat simple features | PANGAEA [R72]; Tick Tick Bloom CNNs [R165] | Bake-off against band statistics and AlphaEarth; choose on validation |
| Multi-task negative transfer | [R120] | Single-task comparators for headline targets; target grouping |
| Calibration breaks under shift | Pattern-dependent coverage [R146] | Mondrian by pattern × region; ACI online [R140]; report per-group coverage |
| Compute and precision (Turing) | No bf16, no FlashAttention-2 [R58] | fp16 AMP with GradScaler [R57]; small core; frozen encoders; gradient checkpointing |
| Data drift and product changes | NEON taxonomy suspended (DP1.20166); WQX 2.2 lacks post-2024 USGS data [R184] | Version-locked snapshots, data cards, WQX 3.0 profiles |
| Licences | neonutilities AGPL-3.0; PANGAEA GPL-3.0; AlphaEarth CC-BY 4.0; ImageBind and MultiMAE custom | Research use is fine; check before redistributing models or derived data |
| Over-claiming (again) | REVIEW_REPORT.md | Pre-registration, claims ledger, independent re-run before any paper text |

---

## (g) References

Verification tags: **[arXiv]** arXiv API or abstract page fetched; **[DOI]** Crossref record fetched; **[proc]** official proceedings index; **[page]** official page or README fetched; **[search]** search summary only.

**Fusion and missing modality**
- [R1] Jaegle A, Gimeno F, Brock A, Zisserman A, Vinyals O, Carreira J. *Perceiver: General Perception with Iterative Attention.* ICML 2021. arXiv:2103.03206. https://arxiv.org/abs/2103.03206 [arXiv]
- [R2] Jaegle A, Borgeaud S, Alayrac J-B, et al. (15 authors; last: Carreira J). *Perceiver IO: A General Architecture for Structured Inputs & Outputs.* ICLR 2022. arXiv:2107.14795. Code: https://github.com/google-deepmind/deepmind-research (perceiver/) [arXiv]
- [R3] Bachmann R, Mizrahi D, Atanov A, Zamir A. *MultiMAE: Multi-modal Multi-task Masked Autoencoders.* ECCV 2022. arXiv:2204.01678. https://github.com/EPFL-VILAB/MultiMAE [arXiv][search: ECVA page]
- [R4] Mizrahi D, Bachmann R, Kar OF, et al. (7; last: Zamir A). *4M: Massively Multimodal Masked Modeling.* NeurIPS 2023. arXiv:2312.06647. https://github.com/apple-aiml-research/ml-4m [arXiv][proc]
- [R5] Bachmann R, Kar OF, Mizrahi D, et al. (9; last: Zamir A). *4M-21: An Any-to-Any Vision Model for Tens of Tasks and Modalities.* NeurIPS 2024. arXiv:2406.09406. [arXiv][proc]
- [R6] Girdhar R, El-Nouby A, Liu Z, Singh M, Alwala KV, Joulin A, Misra I. *ImageBind: One Embedding Space To Bind Them All.* CVPR 2023. arXiv:2305.05665. https://github.com/facebookresearch/ImageBind [arXiv]
- [R7] Zhu B, Lin B, Ning M, et al. (14; last: Yuan L). *LanguageBind: Extending Video-Language Pretraining to N-modality by Language-based Semantic Alignment.* ICLR 2024. arXiv:2310.01852. [arXiv][proc]
- [R8] Zhang Y, Gong K, Zhang K, Li H, Qiao Y, Ouyang W, Yue X. *Meta-Transformer: A Unified Framework for Multimodal Learning.* arXiv:2307.10802 (2023; no venue listed). https://github.com/invictus717/MetaTransformer [arXiv]
- [R9] Liang W, Zhang Y, Kwon Y, Yeung S, Zou J. *Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning.* NeurIPS 2022. arXiv:2203.02053. [arXiv]
- [R10] Ma M, Ren J, Zhao L, Testuggine D, Peng X. *Are Multimodal Transformers Robust to Missing Modality?* CVPR 2022. arXiv:2204.05454. [arXiv]
- [R11] Ma M, Ren J, Zhao L, Tulyakov S, Wu C, Peng X. *SMIL: Multimodal Learning with Severely Missing Modality.* AAAI 2021. arXiv:2103.05677. https://github.com/deep-real/SMIL [arXiv]
- [R12] Lee Y-L, Tsai Y-H, Chiu W-C, Lee C-Y. *Multimodal Prompting with Missing Modalities for Visual Recognition.* CVPR 2023. arXiv:2303.03369. https://github.com/YiLunLee/missing_aware_prompts [arXiv][page: CVF]
- [R13] Hu L, Shi T, Feng W, Shang F, Wan L. *Deep Correlated Prompting for Visual Recognition with Missing Modalities.* NeurIPS 2024. [proc]
- [R14] Neverova N, Wolf C, Taylor G, Nebout F. *ModDrop: Adaptive Multi-Modal Gesture Recognition.* IEEE TPAMI 38(8):1692–1706, 2016. doi:10.1109/TPAMI.2015.2461544 [DOI]
- [R15] Havaei M, Guizard N, Chapados N, Bengio Y. *HeMIS: Hetero-Modal Image Segmentation.* MICCAI 2016. arXiv:1607.05194. [arXiv]
- [R16] Ding Y, Yu X, Yang Y. *RFNet: Region-Aware Fusion Network for Incomplete Multi-Modal Brain Tumor Segmentation.* ICCV 2021. https://openaccess.thecvf.com/content/ICCV2021/html/Ding_RFNet_Region-Aware_Fusion_Network_for_Incomplete_Multi-Modal_Brain_Tumor_Segmentation_ICCV_2021_paper.html [page]
- [R17] Zhang Y, He N, Yang J, et al. (9; last: Zheng Y). *mmFormer: Multimodal Medical Transformer for Incomplete Multimodal Learning of Brain Tumor Segmentation.* MICCAI 2022. arXiv:2206.02425. [arXiv]
- [R18] Wang H, Chen Y, Ma C, Avery J, Hull L, Carneiro G. *Multi-Modal Learning With Missing Modality via Shared-Specific Feature Modelling.* CVPR 2023. arXiv:2307.14126. https://github.com/billhhh/ShaSpec [arXiv][page: CVF]
- [R19] Gupta S, Hoffman J, Malik J. *Cross Modal Distillation for Supervision Transfer.* CVPR 2016. arXiv:1507.00448. [arXiv][page: CVF]
- [R20] Wei S, Luo Y, Luo C. *MMANet: Margin-aware Distillation and Modality-aware Regularization for Incomplete Multimodal Learning.* CVPR 2023. arXiv:2304.08028. [arXiv]
- [R21] Vapnik V, Vashist A. *A new learning paradigm: Learning using privileged information.* Neural Networks 22(5–6):544–557, 2009. doi:10.1016/j.neunet.2009.06.042 [DOI]
- [R22] Swamy V, Satayeva M, Frej J, Bossy T, Vogels T, Jaggi M, Käser T, Hartley M-A. *MultiMoDN—Multimodal, Multi-Task, Interpretable Modular Networks.* NeurIPS 2023. arXiv:2309.14118. https://github.com/EPFLiGHT/MultiModN (MIT) [arXiv][proc]
- [R23] Han X, Nguyen H, Harris C, Ho N, Saria S. *FuseMoE: Mixture-of-Experts Transformers for Fleximodal Fusion.* NeurIPS 2024. arXiv:2402.03226. https://github.com/aaronhan223/FuseMoE [arXiv][proc]
- [R24] Yun S, Choi I, Peng J, Wu Y, Bao J, Zhang Q, Xin J, Long Q, Chen T. *Flex-MoE: Modeling Arbitrary Modality Combination via the Flexible Mixture-of-Experts.* NeurIPS 2024 (spotlight). arXiv:2410.08245. https://github.com/UNITES-Lab/Flex-MoE [arXiv][proc]
- [R25] Zhang C, Chu X, Ma L, Zhu Y, Wang Y, Wang J, Zhao J. *M3Care: Learning with Missing Modalities in Multimodal Healthcare Data.* KDD 2022. arXiv:2210.17292. https://github.com/choczhang/M3Care [arXiv][page]
- [R26] Wu Z, Dadu A, Tustison N, Avants B, Nalls M, Sun J, Faghri F. *Multimodal Patient Representation Learning with Missing Modalities and Labels.* ICLR 2024. https://iclr.cc/virtual/2024/poster/18934 [page][proc]
- [R27] Yao W, Yin K, Cheung WK, Liu J, Qin J. *DrFuse: Learning Disentangled Representation for Clinical Multi-Modal Fusion with Missing Modality and Modal Inconsistency.* AAAI 2024, 38:16416–16424. doi:10.1609/aaai.v38i15.29578. arXiv:2403.06197 [DOI][arXiv]
- [R28] Hayat N, Geras KJ, Shamout FE. *MedFuse: Multi-modal fusion with clinical time-series data and chest X-ray images.* MLHC 2022 (PMLR 182). arXiv:2207.07027 [arXiv][proc]
- [R29] Hemker K, Simidjievski N, Jamnik M. *HEALNet: Multimodal Fusion for Heterogeneous Biomedical Data.* NeurIPS 2024. arXiv:2311.09115. https://github.com/konst-int-i/healnet (Apache-2.0) [arXiv][proc]
- [R30] Wu M, Goodman N. *Multimodal Generative Models for Scalable Weakly-Supervised Learning.* NeurIPS 2018. arXiv:1802.05335 [arXiv][proc]
- [R31] Shi Y, Siddharth N, Paige B, Torr PHS. *Variational Mixture-of-Experts Autoencoders for Multi-Modal Deep Generative Models.* NeurIPS 2019. arXiv:1911.03393 [arXiv][proc]
- [R32] Sutter TM, Daunhawer I, Vogt JE. *Generalized Multimodal ELBO.* ICLR 2021. arXiv:2105.02470 [arXiv]
- [R33] Daunhawer I, Sutter TM, Chin-Cheong K, Palumbo E, Vogt JE. *On the Limitations of Multimodal VAEs.* ICLR 2022. arXiv:2110.04121 [arXiv]
- [R34] Arevalo J, Solorio T, Montes-y-Gómez M, González FA. *Gated Multimodal Units for Information Fusion.* arXiv:1702.01992 (2017). Venue **UNVERIFIED**. [arXiv]
- [R35] Han Z, Zhang C, Fu H, Zhou JT. *Trusted Multi-View Classification.* ICLR 2021. arXiv:2102.02051 [arXiv]
- [R36] Zhang Q, Wu H, Zhang C, Hu Q, Fu H, et al. (7; last: Peng X). *Provable Dynamic Fusion for Low-Quality Multimodal Data.* ICML 2023. arXiv:2306.02050 [arXiv][proc]
- [R37] Zhang C, Han Z, Cui Y, Fu H, Zhou JT, Hu Q. *CPM-Nets: Cross Partial Multi-View Networks.* NeurIPS 2019. [proc]
- [R38] Lin Y, Gou Y, Liu Z, Li B, Lv J, Peng X. *COMPLETER: Incomplete Multi-View Clustering via Contrastive Prediction.* CVPR 2021. [page: CVF]
- [R39] Huang Z, Hu P, Zhou JT, Lv J, Peng X. *Partially View-aligned Clustering.* NeurIPS 2020. [proc]
- [R40] Wang W, Tran D, Feiszli M. *What Makes Training Multi-Modal Classification Networks Hard?* CVPR 2020. arXiv:1905.12681 [arXiv]
- [R41] Wu N, Jastrzębski S, Cho K, Geras KJ. *Characterizing and Overcoming the Greedy Nature of Learning in Multi-modal Deep Neural Networks.* ICML 2022. arXiv:2202.05306 [arXiv][proc]
- [R42] Peng X, Wei Y, Deng A, Wang D, Hu D. *Balanced Multimodal Learning via On-the-fly Gradient Modulation.* CVPR 2022. arXiv:2203.15332. https://github.com/GeWu-Lab/OGM-GE_CVPR2022 (MIT) [arXiv]
- [R43] Huang Y, Lin J, Zhou C, Yang H, Huang L. *Modality Competition: What Makes Joint Training of Multi-modal Network Fail in Deep Learning? (Provably).* ICML 2022. arXiv:2203.12221 [arXiv][proc]
- [R44] Du C, Teng J, Li T, Liu Y, Yuan T, Wang Y, Yuan Y, Zhao H. *On Uni-Modal Feature Learning in Supervised Multi-Modal Learning.* ICML 2023. arXiv:2305.01233 [arXiv][proc]
- [R45] Wei Y, Hu D. *MMPareto: Boosting Multimodal Learning with Innocent Unimodal Assistance.* ICML 2024. arXiv:2405.17730 [arXiv][proc]
- [R46] Wu R, Wang H, Chen H-T, Carneiro G. *Deep Multimodal Learning with Missing Modality: A Survey.* TMLR (accepted; v4 Feb 2026). arXiv:2409.07825 [arXiv]. A further 2025 review, *A systematic literature review on incomplete multimodal learning* (Systems Science & Control Engineering), was seen in search results only; details **UNVERIFIED**.
- [R47] Liang PP, Lyu Y, Fan X, et al. (12; last: Morency L-P). *MultiBench: Multiscale Benchmarks for Multimodal Representation Learning.* NeurIPS 2021 Datasets & Benchmarks. arXiv:2107.07502. https://github.com/pliang279/MultiBench (MIT) [arXiv][proc]
- [R48] Gupta S, Sundaram S, Wang C, Jegelka S, Isola P. *Better Together: Leveraging Unpaired Multimodal Data for Stronger Unimodal Models.* arXiv:2510.08492 (2025). NeurIPS 2025 and ICLR 2026 pages appear in search results; venue **UNVERIFIED**. [arXiv]
- [R49] Gat I, Schwartz I, Schwing A. *Perceptual Score: What Data Modalities Does Your Model Perceive?* NeurIPS 2021. arXiv:2110.14375 [arXiv][proc]
- [R50] Parcalabescu L, Frank A. *MM-SHAP: A Performance-agnostic Metric for Measuring Multimodal Contributions in Vision and Language Models & Tasks.* ACL 2023. doi:10.18653/v1/2023.acl-long.223 [arXiv]
- [R51] Liang PP, Cheng Y, Fan X, et al. (12; last: Morency L-P). *Quantifying & Modeling Multimodal Interactions: An Information Decomposition Framework.* NeurIPS 2023. arXiv:2302.12247. https://github.com/pliang279/PID [arXiv][proc]
- [R52] Horn M, Moor M, Bock C, Rieck B, Borgwardt K. *Set Functions for Time Series.* ICML 2020. arXiv:1909.12064 [arXiv][proc]
- [R53] Shukla SN, Marlin BM. *Multi-Time Attention Networks for Irregularly Sampled Time Series.* ICLR 2021. arXiv:2101.10318 [arXiv]
- [R54] Zhang X, Zeman M, Tsiligkaridis T, Zitnik M. *Graph-Guided Network for Irregularly Sampled Multivariate Time Series.* ICLR 2022. arXiv:2110.05357 [arXiv]
- [R55] Houlsby N, Giurgiu A, Jastrzebski S, et al. *Parameter-Efficient Transfer Learning for NLP.* ICML 2019. arXiv:1902.00751 [arXiv][proc]
- [R56] Hu EJ, Shen Y, Wallis P, et al. *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685 (2021). Venue not re-verified here. [arXiv]
- [R57] Micikevicius P, Narang S, Alben J, et al. *Mixed Precision Training.* ICLR 2018. arXiv:1710.03740 [arXiv]
- [R58] Dao-AILab. *FlashAttention README* (GPU and dtype support). https://github.com/Dao-AILab/flash-attention [page]

**Earth observation**
- [R60] Tseng G, Fuller A, Reil M, Herzog H, Beukema P, Bastani F, Green JR, Shelhamer E, Kerner H, Rolnick D. *Galileo: Learning Global & Local Features of Many Remote Sensing Modalities.* ICML 2025 (PMLR 267). arXiv:2502.09356. https://github.com/nasaharvest/galileo (MIT) [arXiv][proc][page]
- [R61] Tseng G, Cartuyvels R, Zvonkov I, Purohit M, Rolnick D, Kerner H. *Lightweight, Pre-trained Transformers for Remote Sensing Timeseries* (Presto). arXiv:2304.14065 (v4 2024). https://github.com/nasaharvest/presto (MIT) [arXiv][page]
- [R62] Astruc G, Gonthier N, Mallet C, Landrieu L. *AnySat: One Earth Observation Model for Many Resolutions, Scales, and Modalities.* CVPR 2025. arXiv:2412.14123. https://github.com/gastruc/AnySat (MIT) [arXiv][page]
- [R63] Astruc G, Gonthier N, Mallet C, Landrieu L. *OmniSat: Self-Supervised Modality Fusion for Earth Observation.* ECCV 2024. arXiv:2404.08351. https://github.com/gastruc/OmniSat (MIT) [arXiv]
- [R64] Jakubik J, Yang F, Blumenstiel B, et al. (16; last: Longépé N). *TerraMind: Large-Scale Generative Multimodality for Earth Observation.* ICCV 2025. arXiv:2504.11171. https://github.com/IBM/terramind (Apache-2.0); https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base [arXiv][page]
- [R65] Brown CF, Kazmierski MR, Pasquarella VJ, et al. (19; last: Kohli P). *AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data.* arXiv:2507.22291 (2025). Data: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL (CC-BY 4.0) [arXiv][page]
- [R66] Fuller A, Millard K, Green JR. *CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders.* NeurIPS 2023. arXiv:2311.00566 [arXiv][proc]
- [R67] Xiong Z, Wang Y, Zhang F, et al. (10; last: Zhu XX). *Neural Plasticity-Inspired Multimodal Foundation Model for Earth Observation* (DOFA). arXiv:2403.15356 (2024) [arXiv]
- [R68] Cong Y, Khanna S, Meng C, et al. (9; last: Ermon S). *SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery.* NeurIPS 2022. arXiv:2207.08051 [arXiv][proc]
- [R69] Szwarcman D, Roy S, Fraccaro P, et al. (36; last: Bernabe Moreno J). *Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications.* arXiv:2412.02732 (2024) [arXiv]
- [R70] Wang Y, Xiong Z, Liu C, et al. (11; last: Zhu XX). *Towards a Unified Copernicus Foundation Model for Earth Vision.* ICCV 2025. arXiv:2503.11849 [arXiv]
- [R71] Guo X, Lao J, Dang B, et al. (16; last: Li Y). *SkySense: A Multi-Modal Remote Sensing Foundation Model Towards Universal Interpretation for Earth Observation Imagery.* CVPR 2024. arXiv:2312.10115 [arXiv]
- [R72] Marsocci V, Jia Y, Le Bellier G, et al. (15; last: Nascetti A). *PANGAEA: A Global and Inclusive Benchmark for Geospatial Foundation Models.* arXiv:2412.04204. https://github.com/VMarsocci/pangaea-bench (GPL-3.0) [arXiv]
- [R73] Lacoste A, Lehmann N, Rodriguez P, et al. (17; last: Zhu XX). *GEO-Bench: Toward Foundation Models for Earth Monitoring.* NeurIPS 2023 (Datasets & Benchmarks). arXiv:2306.03831 [arXiv][proc]
- [R74] Pahlevan N, Smith B, Schalles J, et al. (17; last: Stumpf R). *Seamless retrievals of chlorophyll-a from Sentinel-2 (MSI) and Sentinel-3 (OLCI) in inland and coastal waters: A machine-learning approach.* Remote Sens. Environ. 240:111604, 2020. doi:10.1016/j.rse.2019.111604 [DOI][search: abstract numbers]
- [R75] Pahlevan N, Smith B, Alikas K, et al. (23; last: Ruiz-Verdù A). *Simultaneous retrieval of selected optical water quality indicators from Landsat-8, Sentinel-2, and Sentinel-3.* Remote Sens. Environ. 270:112860, 2022. doi:10.1016/j.rse.2021.112860 [DOI]
- [R76] Lehmann MK, Gurlin D, Pahlevan N, et al. (77). *GLORIA – A globally representative hyperspectral in situ dataset for optical sensing of water quality.* Sci. Data 10:100, 2023. doi:10.1038/s41597-023-01973-y [DOI]
- [R77] Ross MRV, Topp SN, Appling AP, et al. (8; last: Pavelsky TM). *AquaSat: A Data Set to Enable Remote Sensing of Water Quality for Inland Waters.* Water Resour. Res. 55(11):10012–10025, 2019. doi:10.1029/2019WR024883 [DOI]
- [R78] Topp SN, Pavelsky TM, Stanley EH, Yang X, Griffin CG, Ross MRV. *Multi-decadal improvement in US Lake water clarity.* Environ. Res. Lett. 16:055025, 2021. doi:10.1088/1748-9326/abf002 [DOI]

**Partial co-location and data integration**
- [R80] Andersson TR, Bruinsma WP, Markou S, et al. (11; last: Turner RE). *Environmental sensor placement with convolutional Gaussian neural processes.* Environ. Data Sci. 2:e32, 2023. doi:10.1017/eds.2023.22. DeepSensor: https://github.com/alan-turing-institute/deepsensor (MIT) [DOI]
- [R81] Allen A, Markou S, Tebbutt W, et al. (11; last: Turner RE). *End-to-end data-driven weather prediction.* Nature 641:1172–1179, 2025. doi:10.1038/s41586-025-08897-0 [DOI][page: PMC]
- [R82] Garnelo M, Rosenbaum D, Maddison CJ, et al. *Conditional Neural Processes.* ICML 2018. arXiv:1807.01613 [arXiv][proc]
- [R83] Isaac NJB, Jarzyna MA, Keil P, et al. (17; last: O'Hara RB). *Data Integration for Large-Scale Models of Species Distributions.* Trends Ecol. Evol. 35(1):56–67, 2020. doi:10.1016/j.tree.2019.08.006 [DOI]
- [R84] Fletcher RJ, Hefley TJ, Robertson EP, Zuckerberg B, McCleery RA, Dorazio RM. *A practical guide for combining data to model species distributions.* Ecology 100(6):e02710, 2019. doi:10.1002/ecy.2710 [DOI]
- [R85] Jia X, Zwart J, Sadler J, et al. (11; last: Kumar V). *Physics-Guided Recurrent Graph Model for Predicting Flow and Temperature in River Networks.* SDM 2021, 612–620. doi:10.1137/1.9781611976700.69 [DOI]
- [R86] Topp SN, Barclay J, Diaz J, et al. (8; last: Appling AP). *Stream Temperature Prediction in a Shifting Environment: Explaining the Influence of Deep Learning Architecture.* Water Resour. Res. 59(4):e2022WR033880, 2023. doi:10.1029/2022WR033880 [DOI]
- [R87] Fang K, Kifer D, Lawson K, Feng D, Shen C. *The Data Synergy Effects of Time-Series Deep Learning Models in Hydrology.* Water Resour. Res. 58:e2021WR029583, 2022. doi:10.1029/2021WR029583 [DOI]

**Honest evaluation**
- [R90] Grinsztajn L, Oyallon E, Varoquaux G. *Why do tree-based models still outperform deep learning on typical tabular data?* NeurIPS 2022 (Datasets & Benchmarks). arXiv:2207.08815 [arXiv][proc]
- [R91] Shwartz-Ziv R, Armon A. *Tabular data: Deep learning is not all you need.* Inf. Fusion 81:84–90, 2022. doi:10.1016/j.inffus.2021.11.011 [DOI]
- [R92] McElfresh D, Khandagale S, Valverde J, et al. (9; last: White C). *When Do Neural Nets Outperform Boosted Trees on Tabular Data?* NeurIPS 2023 (Datasets & Benchmarks). arXiv:2305.02997 [arXiv][proc]
- [R93] Hollmann N, Müller S, Purucker L, et al. (8; last: Hutter F). *Accurate predictions on small data with a tabular foundation model.* Nature 637:319–326, 2025. doi:10.1038/s41586-024-08328-6. https://github.com/PriorLabs/TabPFN [DOI][page: PMC]
- [R94] Roberts DR, Bahn V, Ciuti S, et al. (14; last: Dormann CF). *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure.* Ecography 40(8):913–929, 2017. doi:10.1111/ecog.02881 [DOI]
- [R95] Ploton P, Mortier F, Réjou-Méchain M, et al. (13; last: Pélissier R). *Spatial validation reveals poor predictive performance of large-scale ecological mapping models.* Nat. Commun. 11:4540, 2020. doi:10.1038/s41467-020-18321-y [DOI]
- [R96] Valavi R, Elith J, Lahoz-Monfort JJ, Guillera-Arroita G. *blockCV: An r package for generating spatially or environmentally separated folds for k-fold cross-validation of species distribution models.* Methods Ecol. Evol. 10(2):225–232, 2019. doi:10.1111/2041-210X.13107 [DOI]
- [R97] Meyer H, Pebesma E. *Predicting into unknown space? Estimating the area of applicability of spatial prediction models.* Methods Ecol. Evol. 12(9):1620–1633, 2021. doi:10.1111/2041-210X.13650 [DOI]
- [R98] Wadoux AMJ-C, Heuvelink GBM, de Bruin S, Brus DJ. *Spatial cross-validation is not the right way to evaluate map accuracy.* Ecol. Model. 457:109692, 2021. doi:10.1016/j.ecolmodel.2021.109692 [DOI]
- [R99] Linnenbrink J, Milà C, Ludwig M, Meyer H. *kNNDM CV: k-fold nearest-neighbour distance matching cross-validation for map accuracy estimation.* Geosci. Model Dev. 17:5897–5912, 2024. doi:10.5194/gmd-17-5897-2024 [DOI]
- [R100] Bouthillier X, Delaunay P, Bronzi M, et al. (17; last: Vincent P). *Accounting for Variance in Machine Learning Benchmarks.* MLSys 2021. arXiv:2103.03098 [arXiv][proc]
- [R101] Demšar J. *Statistical Comparisons of Classifiers over Multiple Data Sets.* JMLR 7:1–30, 2006. https://www.jmlr.org/papers/v7/demsar06a.html [page]
- [R102] Nadeau C, Bengio Y. *Inference for the Generalization Error.* Mach. Learn. 52(3):239–281, 2003. doi:10.1023/A:1024068626366 [DOI]
- [R103] Diebold FX, Mariano RS. *Comparing Predictive Accuracy.* J. Bus. Econ. Stat. 13(3):253–263, 1995. doi:10.1080/07350015.1995.10524599 [DOI]
- [R104] Künsch HR. *The Jackknife and the Bootstrap for General Stationary Observations.* Ann. Stat. 17(3):1217–1241, 1989. doi:10.1214/aos/1176347265 [DOI]
- [R105] Politis DN, Romano JP. *The Stationary Bootstrap.* JASA 89(428):1303–1313, 1994. doi:10.1080/01621459.1994.10476870 [DOI]
- [R106] Wolpert DH. *Stacked generalization.* Neural Netw. 5(2):241–259, 1992. doi:10.1016/S0893-6080(05)80023-1 [DOI]
- [R107] van der Laan MJ, Polley EC, Hubbard AE. *Super Learner.* Stat. Appl. Genet. Mol. Biol. 6(1), 2007. doi:10.2202/1544-6115.1309 [DOI]

**Multi-task learning, uncertainty and conformal prediction**
- [R110] Kendall A, Gal Y, Cipolla R. *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics.* CVPR 2018. arXiv:1705.07115 [arXiv]
- [R111] Chen Z, Badrinarayanan V, Lee C-Y, Rabinovich A. *GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks.* ICML 2018. arXiv:1711.02257 [arXiv][proc]
- [R112] Yu T, Kumar S, Gupta A, Levine S, Hausman K, Finn C. *Gradient Surgery for Multi-Task Learning.* NeurIPS 2020. arXiv:2001.06782 [arXiv][proc]
- [R113] Liu B, Liu X, Jin X, Stone P, Liu Q. *Conflict-Averse Gradient Descent for Multi-task Learning.* NeurIPS 2021. arXiv:2110.14048 [arXiv][proc]
- [R114] Navon A, Shamsian A, Achituve I, et al. *Multi-Task Learning as a Bargaining Game* (Nash-MTL). ICML 2022. arXiv:2202.01017 [arXiv][proc]
- [R115] Liu B, Feng Y, Stone P, Liu Q. *FAMO: Fast Adaptive Multitask Optimization.* NeurIPS 2023. arXiv:2306.03792 [arXiv][proc]
- [R116] Kurin V, De Palma A, Kostrikov I, Whiteson S, Kumar MP. *In Defense of the Unitary Scalarization for Deep Multi-Task Learning.* NeurIPS 2022. arXiv:2201.04122 [arXiv][proc]
- [R117] Xin D, Ghorbani B, Gilmer J, Garg A, Firat O. *Do Current Multi-Task Optimization Methods in Deep Learning Even Help?* NeurIPS 2022. arXiv:2209.11379 [proc][search]
- [R118] Royer A, Blankevoort T, Ehteshami Bejnordi B. *Scalarization for Multi-Task and Multi-Domain Learning at Scale.* NeurIPS 2023. arXiv:2310.08910 [arXiv][proc]
- [R119] Li W-H, Liu X, Bilen H. *Learning Multiple Dense Prediction Tasks from Partially Annotated Data.* CVPR 2022. arXiv:2111.14893 [arXiv]
- [R120] Sadler JM, Appling AP, Read JS, et al. (7; last: Kumar V). *Multi-Task Deep Learning of Daily Streamflow and Water Temperature.* Water Resour. Res. 58(4):e2021WR030138, 2022. doi:10.1029/2021WR030138 [DOI]
- [R121] Nix DA, Weigend AS. *Estimating the mean and variance of the target probability distribution.* ICNN'94, vol. 1, 55–60. doi:10.1109/ICNN.1994.374138 [DOI]
- [R122] Kendall A, Gal Y. *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* NeurIPS 2017. arXiv:1703.04977 [arXiv][proc]
- [R123] Seitzer M, Tavakoli A, Antic D, Martius G. *On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks.* ICLR 2022. arXiv:2203.09168. https://github.com/martius-lab/beta-nll (MIT) [arXiv]
- [R124] Stirn A, Wessels H-H, Schertzer M, Pereira L, Sanjana NE, Knowles DA. *Faithful Heteroscedastic Regression with Neural Networks.* AISTATS 2023 (PMLR 206). arXiv:2212.09184 [arXiv][proc]
- [R125] Gneiting T, Raftery AE. *Strictly Proper Scoring Rules, Prediction, and Estimation.* JASA 102(477):359–378, 2007. doi:10.1198/016214506000001437 [DOI]
- [R126] Tagasovska N, Lopez-Paz D. *Single-Model Uncertainties for Deep Learning.* NeurIPS 2019. arXiv:1811.00908 [arXiv][proc]
- [R127] Klotz D, Kratzert F, Gauch M, Keefe Sampson A, Brandstetter J, Klambauer G, Hochreiter S, Nearing G. *Uncertainty estimation with deep learning for rainfall–runoff modeling.* HESS 26:1673–1693, 2022. doi:10.5194/hess-26-1673-2022 [page]
- [R128] Lakshminarayanan B, Pritzel A, Blundell C. *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS 2017. arXiv:1612.01474 [arXiv][proc]
- [R129] Gal Y, Ghahramani Z. *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.* ICML 2016. arXiv:1506.02142 [arXiv][proc]
- [R130] Tobin J. *Estimation of Relationships for Limited Dependent Variables.* Econometrica 26(1):24–36, 1958. doi:10.2307/1907382 [DOI]
- [R131] Pearce T, Jeong J-H, Jia Y, Zhu J. *Censored Quantile Regression Neural Networks for Distribution-Free Survival Analysis.* NeurIPS 2022. arXiv:2205.13496 [arXiv]
- [R132] Kuleshov V, Fenner N, Ermon S. *Accurate Uncertainties for Deep Learning Using Calibrated Regression.* ICML 2018. arXiv:1807.00263 [arXiv][proc]
- [R133] Guo C, Pleiss G, Sun Y, Weinberger KQ. *On Calibration of Modern Neural Networks.* ICML 2017. arXiv:1706.04599 [arXiv][proc]
- [R134] Brier GW. *Verification of Forecasts Expressed in Terms of Probability.* Mon. Weather Rev. 78(1):1–3, 1950. doi:10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2 [DOI]
- [R135] Murphy AH. *A New Vector Partition of the Probability Score.* J. Appl. Meteorol. 12(4):595–600, 1973. doi:10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2 [DOI]
- [R136] Lei J, G'Sell M, Rinaldo A, Tibshirani RJ, Wasserman L. *Distribution-Free Predictive Inference for Regression.* JASA 113(523):1094–1111, 2018. doi:10.1080/01621459.2017.1307116 [DOI]
- [R137] Romano Y, Patterson E, Candès EJ. *Conformalized Quantile Regression.* NeurIPS 2019. arXiv:1905.03222 [arXiv][proc]
- [R138] Angelopoulos AN, Bates S. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* arXiv:2107.07511 [arXiv]
- [R139] Tibshirani RJ, Foygel Barber R, Candès EJ, Ramdas A. *Conformal Prediction Under Covariate Shift.* NeurIPS 2019. arXiv:1904.06019 [arXiv][proc]
- [R140] Gibbs I, Candès E. *Adaptive Conformal Inference Under Distribution Shift.* NeurIPS 2021. arXiv:2106.00170 [arXiv][proc]
- [R141] Zaffran M, Dieuleveut A, Féron O, Goude Y, Josse J. *Adaptive Conformal Predictions for Time Series.* ICML 2022. arXiv:2202.07282 [arXiv][proc]
- [R142] Barber RF, Candès EJ, Ramdas A, Tibshirani RJ. *Conformal prediction beyond exchangeability.* Ann. Stat. 51(2):816–845, 2023. doi:10.1214/23-AOS2276 [DOI]
- [R143] Xu C, Xie Y. *Conformal prediction interval for dynamic time-series.* ICML 2021 (PMLR 139). arXiv:2010.09107 (journal version titled *Conformal prediction for time series*) [arXiv][proc]
- [R144] Mao H, Martin R, Reich BJ. *Valid Model-Free Spatial Prediction.* JASA 119(546):904–914, 2024. doi:10.1080/01621459.2022.2147531 [DOI][arXiv]
- [R145] Guan L. *Localized conformal prediction: a generalized inference framework for conformal prediction.* Biometrika 110(1):33–50, 2023. doi:10.1093/biomet/asac040 [DOI]
- [R146] Zaffran M, Dieuleveut A, Josse J, Romano Y. *Conformal Prediction with Missing Values.* ICML 2023. arXiv:2306.02732. https://github.com/mzaffran/ConformalPredictionMissingValues (MIT) [arXiv][proc]
- [R147] Angelopoulos AN, Bates S, Fisch A, Lei L, Schuster T. *Conformal Risk Control.* ICLR 2024. arXiv:2208.02814 [arXiv][proc]
- [R148] Messoudi S, Destercke S, Rousseau S. *Copula-based conformal prediction for Multi-Target Regression.* arXiv:2101.12002 (2021). Journal version **UNVERIFIED**. [arXiv]
- [R149] Vovk V, Gammerman A, Shafer G. *Algorithmic Learning in a Random World.* Springer, 2005. doi:10.1007/b106715 [DOI]

**Environmental and water-quality examples**
- [R150] Zhi W, Feng D, Tsai W-P, Sterle G, Harpold A, Shen C, Li L. *From Hydrometeorology to River Water Quality: Can a Deep Learning Model Predict Dissolved Oxygen at the Continental Scale?* Environ. Sci. Technol. 55(4):2357–2368, 2021. doi:10.1021/acs.est.0c06783 [DOI]
- [R151] Zhi W, Appling AP, Golden HE, Podgorski J, Li L. *Deep learning for water quality.* Nature Water 2(3):228–241, 2024. doi:10.1038/s44221-024-00202-z [DOI]
- [R152] Zhi W, Ouyang W, Shen C, Li L. *Temperature outweighs light and flow as the predominant driver of dissolved oxygen in US rivers.* Nature Water 1(3):249–260, 2023. doi:10.1038/s44221-023-00038-z [DOI]
- [R153] Varadharajan C, Appling AP, Arora B, et al. (16; last: Zwart J). *Can machine learning accelerate process understanding and decision-relevant predictions of river water quality?* Hydrol. Process. 36(4):e14565, 2022. doi:10.1002/hyp.14565 [DOI]
- [R154] Rahmani F, Lawson K, Ouyang W, Appling A, Oliver S, Shen C. *Exploring the exceptional performance of a deep learning stream temperature model and the value of streamflow data.* Environ. Res. Lett. 16:024025, 2021. doi:10.1088/1748-9326/abd501 [DOI][search: numbers]
- [R155] Read JS, Jia X, Willard J, et al. (12; last: Kumar V). *Process-Guided Deep Learning Predictions of Lake Water Temperature.* Water Resour. Res. 55(11):9173–9190, 2019. doi:10.1029/2019WR024922 [DOI]
- [R156] Willard JD, Read JS, Topp S, Hansen GJA, Kumar V. *Daily surface temperatures for 185,549 lakes in the conterminous United States estimated using deep learning (1980–2020).* Limnol. Oceanogr. Lett. 7(4):287–301, 2022. doi:10.1002/lol2.10249 [DOI]
- [R157] Willard JD, Read JS, Appling AP, Oliver SK, Jia X, Kumar V. *Predicting Water Temperature Dynamics of Unmonitored Lakes With Meta-Transfer Learning.* Water Resour. Res. 57:e2021WR029579, 2021. doi:10.1029/2021WR029579 [DOI]
- [R158] Willard JD, Ciulla F, Weierbach H, Kumar V, Varadharajan C. *Evaluating Deep Learning Approaches for Predictions in Unmonitored Basins with Continental-scale Stream Temperature Models.* arXiv:2410.19865 (2024) [arXiv]
- [R159] Kratzert F, Klotz D, Hochreiter S, Nearing GS. *A note on leveraging synergy in multiple meteorological data sets with deep learning for rainfall–runoff modeling.* HESS 25:2685–2703, 2021. doi:10.5194/hess-25-2685-2021 [DOI][page]
- [R160] Kratzert F, Klotz D, Shalev G, Klambauer G, Hochreiter S, Nearing G. *Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets.* HESS 23:5089–5110, 2019. doi:10.5194/hess-23-5089-2019 [DOI]
- [R161] Kratzert F, Gauch M, Nearing G, Klotz D. *NeuralHydrology — A Python library for Deep Learning research in hydrology.* JOSS 7:4050, 2022. doi:10.21105/joss.04050. https://github.com/neuralhydrology/neuralhydrology (BSD-3) [DOI]
- [R162] Feng D, Fang K, Shen C. *Enhancing Streamflow Forecast and Extracting Insights Using Long-Short Term Memory Networks With Data Integration at Continental Scales.* Water Resour. Res. 56:e2019WR026793, 2020. doi:10.1029/2019WR026793 [DOI]
- [R163] Gauch M, Mai J, Lin J. *The proper care and feeding of CAMELS: How limited training data affects streamflow prediction.* Environ. Model. Softw. 135:104926, 2021. doi:10.1016/j.envsoft.2020.104926. arXiv:1911.07249 [DOI][arXiv]
- [R164] Schaeffer BA, Reynolds N, Ferriby H, Salls W, Smith D, Johnston JM, Myer M. *Forecasting freshwater cyanobacterial harmful algal blooms for Sentinel-3 satellite resolved U.S. lakes and reservoirs.* J. Environ. Manage. 349:119518, 2024. doi:10.1016/j.jenvman.2023.119518 [DOI][page: PMC]
- [R165] DrivenData. *Tick Tick Bloom: Harmful Algal Bloom Detection Challenge — Meet the winners* (2023). https://blog.drivendata.org/blog/tick-tick-bloom-challenge-winners [page]
- [R166] Fournier C, Fernandez-Fernandez R, Cirés S, López-Orozco JA, Besada-Portas E, Quesada A. *LSTM networks provide efficient cyanobacterial blooms forecasting even with incomplete spatio-temporal data.* Water Res. 267:122553, 2024. doi:10.1016/j.watres.2024.122553. arXiv:2410.08237 [DOI][arXiv]
- [R167] Rodríguez-López L, Bustos Usta D, Duran-Llacer I, Bravo Alvarez L, Yépez S, et al. *Estimation of Water Quality Parameters through a Combination of Deep Learning and Remote Sensing Techniques in a Lake in Southern Chile.* Remote Sens. 15(17):4157, 2023. doi:10.3390/rs15174157 [DOI][search]
- [R168] Tang Y, Feng Y, Fung S, Ruiz Xomchuk V, Jiang M, Moore T, et al. *Spatiotemporal Deep-Learning-Based Algal Bloom Prediction for Lake Okeechobee Using Multisource Data Fusion.* IEEE JSTARS 15:8318–8331, 2022. doi:10.1109/JSTARS.2022.3208620 [DOI][search]
- [R169] Adil M, Clemins PJ, Schroth AW, et al. (12; last: Wshah S). *From Remote Sensing to Multiple Time Horizons Forecasts: Transformers Model for CyanoHAB Intensity in Lake Champlain.* arXiv:2512.06598 (2025) [arXiv]
- [R170] Smith MB, Rocha AM, Smillie CS, et al. (32; last: Hazen TC). *Natural Bacterial Communities Serve as Quantitative Geochemical Biosensors.* mBio 6(3):e00326-15, 2015. doi:10.1128/mBio.00326-15 [DOI]
- [R171] Cordier T, Esling P, Lejzerowicz F, et al. (8; last: Pawlowski J). *Predicting the Ecological Quality Status of Marine Environments from eDNA Metabarcoding Data Using Supervised Machine Learning.* Environ. Sci. Technol. 51(16):9118–9126, 2017. doi:10.1021/acs.est.7b01518 [DOI]
- [R172] Beversdorf LJ, Chaston SD, Miller TR, McMahon KD. *Microcystin mcyA and mcyE Gene Abundances Are Not Appropriate Indicators of Microcystin Concentrations in Lakes.* PLoS ONE 10(5):e0125353, 2015. doi:10.1371/journal.pone.0125353 [DOI][page: PMC]
- [R173] Vaitomaa J, Rantala A, Halinen K, Rouhiainen L, Tallberg P, Mokelke L, Sivonen K. *Quantitative Real-Time PCR for Determination of Microcystin Synthetase E Copy Numbers for Microcystis and Anabaena in Lakes.* Appl. Environ. Microbiol. 69(12):7289–7297, 2003. doi:10.1128/AEM.69.12.7289-7297.2003 [DOI][page: PMC]
- [R174] Searcy RT, Boehm AB. *Know Before You Go: Data-Driven Beach Water Quality Forecasting.* Environ. Sci. Technol. 57(46):17930–17939, 2023 (online 2022). doi:10.1021/acs.est.2c05972 [DOI][search]
- [R175] Rousso BZ, Bertone E, Stewart R, Hamilton DP. *A systematic literature review of forecasting and predictive models for cyanobacteria blooms in freshwater lakes.* Water Res. 182:115959, 2020. doi:10.1016/j.watres.2020.115959 [DOI] (content via secondary description only)
- [R176] Thompson LR, Sanders JG, McDonald D, et al. (Earth Microbiome Project Consortium). *A communal catalogue reveals Earth's multiscale microbial diversity.* Nature 551:457–463, 2017. doi:10.1038/nature24621 [DOI]
- [R177] Sterle G, Perdrial J, Kincaid DW, et al. (12; last: Harpold AA). *CAMELS-Chem: augmenting CAMELS (Catchment Attributes and Meteorology for Large-sample Studies) with atmospheric and stream water chemistry data.* HESS 28:611–630, 2024. doi:10.5194/hess-28-611-2024 [page]
- [R178] Virro H, Amatulli G, Kmoch A, Shen L, Uuemaa E. *GRQA: Global River Water Quality Archive.* Earth Syst. Sci. Data 13:5483–5507, 2021. doi:10.5194/essd-13-5483-2021 [DOI]

**Data portals** (fetched 2026-10-22)
- [R180] NEON data product API: `https://data.neonscience.org/api/v0/products/{DP1.20288.001, DP1.20033.001, DP1.20093.001, DP1.20097.001, DP1.20282.001, DP1.20281.001, DP1.20138.001, DP1.20141.001, DP1.20120.001, DP1.20107.001, DP1.20163.001, DP1.20166.001, DP1.20219.001, DP4.00130.001, DP1.20016.001, DP1.20053.001, DP3.30006.001, DP3.30010.001}`. `neonutilities` on PyPI (v2.0.2, AGPL-3.0). [page]
- [R181] EPA National Lakes Assessment: https://www.epa.gov/national-aquatic-resource-surveys/nla ; NLA 2022 web report: https://nationallakesassessment.epa.gov/webreport [page]
- [R182] EPA National Rivers and Streams Assessment (2008–09, 2013–14, 2018–19): https://www.epa.gov/national-aquatic-resource-surveys/nrsa. Indicator details **UNVERIFIED**. [page]
- [R183] EPA NARS indicators: https://www.epa.gov/national-aquatic-resource-surveys/indicators-used-national-aquatic-resource-surveys [page]
- [R184] Water Quality Portal (WQX 3.0 notice): https://www.waterqualitydata.us/ [page]
- [R185] EPA CyAN: https://www.epa.gov/water-research/cyanobacteria-assessment-network-cyan [page]
- [R186] EPA StreamCat/LakeCat: https://www.epa.gov/national-aquatic-resource-surveys/streamcat-dataset [page]
- [R187] Daymet: https://daymet.ornl.gov/ ; gridMET: https://www.climatologylab.org/gridmet.html ; ERA5-Land: https://www.ecmwf.int/en/era5-land [page]

**Internal SENTINEL sources:** `/home/bcheng/STOCKHOLM/REVIEW_REPORT.md`; `results/trackB_fusion/FUSION_FINDINGS.md`; `results/trackB_fusion/fusion_cycle2/modality_contribution.json`; `results/trackB_fusion/complementary_fusion.json`.
