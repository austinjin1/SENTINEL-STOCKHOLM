# M7 SENTINEL-Lite: architecture research and SOTA plan

**Date:** 2026-11-10. **Type:** a read-only research synthesis for the lead's decision. AI-assisted.

**What this document is not:**
- a pre-registration;
- a result;
- an edit to any protocol.

Every protocol change below is a *proposal*. The lead writes it as a dated amendment, in a new file or in
`lite_national_protocol.md` (which is not hashed), before any result on it exists (`sota_campaign.md` rule 2).

**Scope.** M7 estimates Secchi, turbidity, TSS and chl-a (numeric) from 4-band NAIP chips matched to WQP samples. The
budget is ≤ 3 M params, ≤ 2 GMACs per 512² tile, INT8 ≤ 4 MB and ≤ 100 ms.

### How the evidence was checked (read first)

**Repo numbers.** Every M7 number was re-read this session from these files:
- `sentinel_runs/lite/selection_national2.json`;
- the crop256 national-2 `summary.json` and `compare_natbaselines_20261109-211150_all.json`;
- `natbaselines_20261109-211150/summary.json`;
- the SSL crop128 `summary.json` and the crop256 `train_log.txt`;
- `docs/preregistration/lite_split_national_report.json`, which gives **role counts only**;
- `processed/naip_matchups/national_pm7/matchups.parquet` (counts by target, unit and censoring);
- `processed/wqp_discrete/_coverage_by_variable.csv`;
- the `processed/nars_lakes` table columns.

No test-role label value was read.

**Web search.** The session's shared search budget ran out after **one** query. Every other reference was checked
by fetching the arXiv abstract page, the Crossref DOI record or the landing page directly. That is a fetch, not a
search. Tags:

| Tag | Meaning |
|---|---|
| **[S]** | Found by web search this session |
| **[F]** | Fetched directly this session (arXiv abs page, Crossref record or landing page) |
| **[L]** | From this workflow's literature step (web-searched this session by that agent; not re-fetched by me) |
| **[R08]** | Carried from `docs/research/08_drone_rgb_nir.md` (verified 2026-10-22). **UNVERIFIED this session.** |
| **[U]** | Unverified |

**Design inputs were truncated.** The workflow handed me one full design (CIR-Lite, "Calibrate-Invert-Render") and
none of the critics' text. Section 4 therefore applies my own critique, grounded in the repo evidence and the
fetched literature, and lists every change it makes to CIR-Lite.

**Prior plan.** This builds on the M7 SOTA plan in `docs/handoff/M7-lite.md` ("SOTA plan", 2026-11-09, approved). It
does not redo it.

---

## 0. Decision summary

1. **Simple ML wins on Secchi because, with today's inputs, per-chip colour statistics plus location are close to a
   sufficient statistic. The CNN is also handicapped by an input asymmetry.**

   National-2 val, Secchi on lakes (n = 1,310 / 470 water bodies):

   | Model | MdSA | log-R² |
   |---|---|---|
   | crop256 KD student | 48.9 | 0.479 |
   | GBM-RGBN | 47.3 | 0.501 |
   | **GBM + lat/lon/DOY** | **36.5** | **0.627** |
   | Location kNN | 51.5 | 0.29 |

   - Colour and location are complementary, and only the GBM gets both.
   - **The selected student has never seen the DOQQ scene statistics that GBM-RGBN uses** (the `scene_p50_*`
     features and the ratios to them).
   - The queued context config passes lat/lon/DOY *instead of* the scene statistics (`film_source: scene` carries 8
     terms).
   - So no student has yet had information parity with GBM + context. The diagnosis is mostly a problem of
     information and comparison design, not of backbone capacity.
2. **Where the CNN already wins, it wins through spatial information: turbidity.**
   - National-2 val: student 91.0 vs GBM-RGBN 116.8, Δ −25.7 [−45.4, −1.9]; vs GBM + context 111.9.
   - log-R²: 0.428 vs 0.285 / 0.349.
   - Widening the crop from 128 to 256 px helped turbidity (94.8 → 91.0) and Secchi (50.7 → 48.9).
   - Turbidity has the fewest training labels (1,555), and several hundred more are already matched on disk
     (§5).
3. **The current architecture is adequate for the budget, but not optimal for the task.**
   - The backbone is fine: MobileNetV4-Conv-S, 1.32 M params, 1.13 GMACs, 30 ms proxy. A bigger backbone is not the
     lever (rule 5).
   - What is missing:
     - the scene and context inputs;
     - a way to stop re-learning what the GBM already captures;
     - the censored rows and the extra turbidity units / SSC labels;
     - a self-supervised objective that keeps colour (SimSiam was trained to *ignore* colour).
4. **What to build, ranked** (§4):

   | Rank | Design | Score /10 | GPU-h |
   |---|---|---|---|
   | 1 | **ANCHOR-Lite**: a GBM-anchored, context-complete, multi-target residual CNN | 6.5 | ≈ 6, + ≈ 2.5 for the published-DL comparator |
   | 2 | **WaterSet-Lite**: a set/distribution encoder over water pixels, as ANCHOR's alternative residual branch | 5.0 | ≈ 1.5, CPU-capable |
   | 3 | **CIR-Lite-R**: reduced physics (calibrate, then 4-latent IOP heads), gated on a calibration diagnostic | 4.0 | ≈ 7 |
   | 4 | **SparK-style masked-image SSL**: replaces SimSiam, keeps colour | 3.0 | ≈ 4 |
   | 5 | Queued context-FiLM config: already counted; no new configs of this kind | 2.5 | ≈ 0.6 |

5. **Targets and data first. Both are nearly free.**
   - Already matched in `national_pm7`, with chips on disk, but dropped by the Lite labels:
     - turbidity **FNU 471** and **NTRU 323** rows;
     - **SSC 456**;
     - censored Secchi 316, TSS 282, chl-a 162.
   - That is about +50 % turbidity labels at 0 GB.
   - Next:
     - USGS NWIS daily FNU at sonde sites (on disk);
     - WQP transparency tube and DOC at new water bodies (≈ 0.1–0.3 GB of chips).
6. **Harder tracks, to pre-register now:**
   - **P2**, a new-source block of rivers and sondes: FNU turbidity at unseen water bodies;
   - **P3**, France: zero-shot transfer to a new sensor and country (conditional on the lead and M2 coordination);
   - **P4**, NLA whole-lake (conditional, many cross-workstream exclusions).
   - The existing Test-N strata X1–X4 stay as they are.
7. **Two rule facts that limit any claim** (§9):
   - The frozen `lite_protocol.md` §8a label rule says "State of the art does not apply: no published method … has
     been re-run". "First-in-class" needs Secchi **and** turbidity-or-TSS to beat **every** baseline, including
     GBM + context.
   - A turbidity-only national win would therefore be labelled "competitive" unless the lead adds a target-scoped
     label. Adding one *now*, after val has shown turbidity is the student's best target, is a forking-paths risk
     that must be disclosed.
   - A clean scoped SOTA route needs two things:
     - a **re-run published method**: a HydroVision-style DenseNet121 regressor and a CyFi-style LightGBM;
     - a **new track pre-registered before any result** (P2).
8. **Honest outlook.**
   - Any §15.1-grade scoped label for M7 within about 3 weeks: **≈ 30 % (20–40 %)**. Most likely: P2 FNU turbidity,
     or national turbidity under a lead-approved scoped rule.
   - Secchi SOTA vs the strongest simple baseline: **≈ 10 %**.

---

## 1. Diagnosis: why simple ML wins or ties

### 1.1 The numbers (re-read this session)

National-2 val (val ∪ val_n), MdSA % with water-body cluster CIs. Secchi is lakes only; turbidity and TSS are all
water bodies. log-R² and log-slope come from each run's `summary.json`.

| Model | Secchi MdSA (log-R², slope) | Turbidity MdSA (log-R², slope) | TSS MdSA (log-R², slope) |
|---|---|---|---|
| crop256 KD student (selected, S 0.916) | 48.9 [45.3, 53.7] (0.479, 0.458) | 91.0 (0.428, 0.411) | 91.6 (0.349, 0.391) |
| crop128 KD + SimSiam init | 53.6 (0.448, 0.419) | 103.0 (0.401, 0.392) | 103.3 |
| GBM-RGBN | 47.3 (0.501, 0.488) | 116.8 (0.285, 0.345) | 97.9 (0.247, 0.344) |
| **GBM-RGBN + lat/lon/DOY** | **36.5 (0.627, 0.589)** | 111.9 (0.349, 0.419) | 102.7 (0.262, 0.332) |
| Location kNN | 51.5 (0.29) | 129.1 | 116.5 |
| Location/season GBM | 58.1 (0.397) | 142.1 | 126.4 |
| Band-ratio (val-best form) | 78.6 (0.193) | 129.6 | 100.4 |
| Train median | 101.0 | 151.9 | 107.7 |

Paired Δ for the crop256 student (student − comparator):

| Comparator | Secchi | Turbidity | TSS |
|---|---|---|---|
| GBM-RGBN | +1.5 [−0.9, +5.5] | **−25.7 [−45.4, −1.9]** | −6.4 [−24.7, +2.3] |
| GBM + context | **+12.4 [+8.9, +16.2]** (student worse) | −20.8 [−55.1, +9.5] | — |

**4-state Test-A** (spent; single look 2026-11-09): the student ensemble scored Secchi 54.4 vs GBM-RGBN 44.2,
Δ +10.2 [+4.4, +14.2]. Label "negative". The student lost 7.8 points from val (46.6) to test.

### 1.2 Causes, in order of weight

1. **Location and season carry much of the Secchi signal, and only the GBM gets both colour and location.**
   - On Secchi log-R², colour alone gives 0.50, location alone 0.29–0.40, and both together 0.63.
   - That is the Tick Tick Bloom pattern [F]. The winners used water-pixel colour statistics, location, date and
     (2nd place) elevation in boosted trees. The 1st place reported that a CNN on the imagery gave "very high
     RMSEs".
   - A drone *does* know its GPS position and date. The lead's 2026-11-10 ruling to test a context-parity student
     is therefore legitimate.
2. **Input asymmetry against the student: scene radiometry.**
   - GBM-RGBN's 45 features include the DOQQ band medians and the water quantiles divided by them. These are a
     per-scene reference against NAIP's per-state, per-year vendor stretch.
   - The selected raw-mode student sees only DN/255.
   - The 52-statistic scene table (`naip_scene_stats_merged.parquet`: p01…p99, mean, std and dark-NIR median per
     band, verified) exists, but no selected student uses it. The context config *replaces* the 8 scene terms with
     lat/lon/DOY.
   - This is the cheapest unexplored fix.
3. **On ~10 k rows, tree models beat neural networks on tabular-like signal.**
   - Grinsztajn, Oyallon & Varoquaux (2022) [F] find that tree models still lead on medium-sized (~10 k) tabular
     data. They attribute this to three inductive biases that networks lack: robustness to uninformative features,
     preserved feature orientation, and irregular target functions.
   - The Secchi signal in NAIP is mostly "chip colour quantiles × location". That is a tabular problem dressed as
     an image. A CNN has to re-learn it from 9,371 labels, through radiometric augmentation, without the scene
     reference.
4. **The label-noise ceiling compresses every model.** Sources of noise:
   - ±7 d Secchi pairing; ±1 d for the other targets;
   - unknown NAIP time of day;
   - a point sample vs a ~30 m pooling window;
   - 8-bit vendor-stretched DN, with NIR near the floor.

   Consequences:
   - Every model has log-slope 0.33–0.59, i.e. strong shrinkage toward the mean.
   - Under MdSA, shrinkage is close to optimal when inputs are noisy, so "de-shrinking" would hurt the headline
     metric. Extremes are the job of strata X2–X4.
5. **SimSiam was taught to ignore the signal.**
   - `litessl/simsiam.py` uses ±30 % per-band gain and gamma 0.7–1.4 (checked, lines 7 and 65). That is invariance
     to exactly the colour that encodes clarity.
   - This is the failure mode named by Xiao et al. (ICLR 2021) [F]: augmentation-invariant contrastive learning
     "can perform poorly when a downstream task violates this assumption" (their example is colour).
   - Observed: SSL init was worse on every target (53.6 / 103.0 / 103.3 vs 50.7 / 94.8 / 98.8).
   - The handoff (2026-11-10 midday) also notes that the SimSiam loss plateaued at −0.95, suggesting partial
     collapse.
6. **KD adds little.**
   - KD targets cover 2,581 of 10,535 train chips.
   - The val Secchi gain (−2.6) did not replicate on Test-A (+1.6 [−1.7, +3.5]).
   - The red-edge information that KD was meant to carry cannot be recovered from 4 bands (report 08 §a.4 [R08]).
7. **Turbidity is the exception, because its information is spatial.**
   - Plumes, gradients, and bank or bottom contamination of fixed centre windows are visible to a CNN with a 154 m
     field of view, but are absent from centre medians.
   - The data agree: the student's turbidity log-R² is 0.428 vs GBM 0.285, and it grows with crop size.

### 1.3 What follows

Following plan rule 5, the network must get **(a)** every input the GBM has (scene statistics, location, season)
and then **(b)** something the GBM structurally cannot use:
- spatial structure;
- unlabelled pixels, through a colour-preserving objective;
- privileged S2 reflectance;
- coupling across targets and units.

The task should also move toward **where per-pixel statistics fail by construction**: narrow rivers, small water
bodies, plume conditions, new sensors and countries.

---

## 2. Literature: SOTA summary (verified sources only)

| Work | What it does | Result as reported | What it implies for M7 | Tag |
|---|---|---|---|---|
| Deshmukh, Wilchek & Batarseh 2025, *HydroVision*, arXiv:2509.01882 | ImageNet CNNs/ViT (VGG16, ResNet50, MobileNetV2, DenseNet121, ViT) on >500k fixed-camera USGS HIVIS RGB stills (2022–24), sonde-paired: chl-a, chlorophylls, CDOM, phycocyanin, suspended sediment, turbidity | Abstract: DenseNet121 best; R² 0.89 for CDOM on validation. Lit step: turbidity R² 0.498, SSC negative. Split method not stated in the abstract. | The only published deep RGB water-quality regressor. **Re-implement it as a published-method comparator** (DenseNet121, 4-ch stem, our rows). Its numbers are not comparable to held-out water bodies. | [F] (abstract) + [L] |
| DrivenData 2023, Tick Tick Bloom winners blog; CyFi | LightGBM on water-pixel colour statistics + location + date (+ elevation, climate) | CNN on the imagery gave "very high RMSEs" (1st place) | The published form of our "GBM + context". **CyFi-style LightGBM is the strongest published method** for NAIP chips. | [F] (blog) + [L] (CyFi) |
| Pahlevan et al. 2020, RSE 240:111604, doi:10.1016/j.rse.2019.111604; Pahlevan et al. 2022, RSE 270:112860, doi:10.1016/j.rse.2021.112860 | Mixture-density networks from calibrated Rrs to chl-a (and multi-parameter) | Lit step: metadata verified | SOTA for *calibrated* reflectance. Not applicable to 8-bit DN without calibration. Usable only on CIR-Lite-R's calibrated ρ̂ (§4.3). | [L] |
| Kloiber, Brezonik, Olmanson & Bauer 2002, RSE 82:38–47, doi:10.1016/S0034-4257(02)00022-6 | Regional Landsat blue/red Secchi regression | — | Published band-ratio comparator (already in the baselines) | [F] (Crossref) |
| Nechad, Ruddick & Park 2010, RSE 114:854–866, doi:10.1016/j.rse.2009.11.022; Dogliotti et al. 2015, RSE 156:157–168, doi:10.1016/j.rse.2014.09.020 | Single-band semi-analytical TSM / turbidity from red or NIR reflectance | — | Published turbidity/TSS comparator forms. Physics prior for CIR-Lite-R's b_bp heads. | [F] (Crossref) |
| Lee et al. 2015, RSE 169:139–149, doi:10.1016/j.rse.2015.08.002; Lee, Carder & Arnone 2002 (QAA), Appl. Opt. 41:5755, doi:10.1364/AO.41.005755 | Mechanistic Secchi theory; quasi-analytical IOP inversion | — | Physics heads and priors for CIR-Lite-R. Both need *calibrated* Rrs. | [F] (Crossref) |
| Rubin et al. 2021, Remote Sens. 13:1434, doi:10.3390/rs13081434 | RF vs 13 historical Secchi algorithms on Landsat | RF best; per-scene models beat pooled ones | Supports scene-reference conditioning (the GBM's scene medians) | [L] |
| He et al. 2022, Water Research 215:118241 (DGRN) | GRU-based Landsat Secchi model over 16,475 lakes | Depth the dominant driver of clarity | Supports morphometry context (HydroLAKES) for both the network and the GBM | [L] |
| Neil, Spyrakos, Hunter & Tyler 2019, RSE 229:159–178, doi:10.1016/j.rse.2019.04.027 | Chl-a retrieval switched by optical water type | — | Precedent for water-type-conditioned heads. Only weakly applicable to 4 broad bands. | [F] (Crossref) |
| Grinsztajn, Oyallon & Varoquaux 2022, arXiv:2207.08815 | 45-dataset benchmark: tree models vs deep learning | Trees still win at ~10 k samples, for reasons of inductive bias | Explains §1.2.3. Motivates *anchoring* the network on the GBM rather than competing with it. | [F] |
| Hollmann et al. 2025, *Nature* 637:319–326, doi:10.1038/s41586-024-08328-6 (TabPFN) | Tabular foundation model, in-context learning on ≤ 10 k samples | Abstract: "outperforms all previous methods on datasets with up to 10,000 samples by a wide margin" | **A possibly stronger simple baseline** than LightGBM on our 45–110 features (Secchi train 9,371 rows fits). Must be decided *before* results whether it joins the "strongest simple" set. Not drone-deployable as the student (it needs the training set at inference). | [S] |
| Sigrist, *Gaussian Process Boosting*, arXiv:2004.02653 | Boosting combined with Gaussian-process / mixed-effects random effects | Better accuracy than existing approaches (abstract) | A stronger *location-aware* simple baseline (spatial random effect). Optional comparator. | [F] |
| Barnwal, Cho & Hocking, arXiv:2006.04920 | XGBoost accelerated-failure-time survival loss | Native since XGBoost 1.2.0. The censoring types are not listed in the abstract [U]. | Comparator parity for censored rows (Tobit in the network ↔ AFT in the GBM). Check the right/left/interval support in the docs before use. | [F] |
| Zaheer et al. 2017 (Deep Sets), arXiv:1703.06114; Lee et al. 2019 (Set Transformer, ICML), arXiv:1810.00825 | Permutation-invariant networks on sets; attention with inducing points, linear in set size | — | Basis of WaterSet-Lite: a *learned* generalisation of the GBM's water-pixel quantiles | [F] |
| He et al. 2021 (MAE), arXiv:2111.06377; Tian et al. 2023 (SparK), arXiv:2301.03580 | Masked-image reconstruction pretraining; SparK adapts it to CNNs (sparse + hierarchical) | SparK beats contrastive learning and transformer MIM on downstream tasks (abstract) | A **colour-preserving** SSL replacement for SimSiam on MobileNetV4 | [F] |
| Xiao, Wang, Efros & Darrell 2021, ICLR, arXiv:2008.05659 | Augmentation invariances hurt tasks that need the invariant factor (e.g. colour) | — | Explains the SimSiam failure | [F] |
| Lopez-Paz, Bottou, Schölkopf & Vapnik 2016, ICLR, arXiv:1511.03643 | Generalized distillation = distillation + privileged information | — | Framing for S2-only-at-training supervision (KD, calibration, red-edge auxiliary) | [F] |
| Zérah, Valero & Inglada 2024, RSE 312:114309, doi:10.1016/j.rse.2024.114309 | Physics-constrained deep learning: inversion of PROSAIL from S2 (vegetation) | — | Closest precedent for CIR-Lite's physics decoder (land, not water) | [F] (Crossref) |
| Seitzer et al. 2022, ICLR, arXiv:2203.09168 (β-NLL); Romano, Patterson & Candès 2019, arXiv:1905.03222 (CQR) | Stable heteroscedastic NLL; conformalized quantile intervals | — | Loss already used (β = 0.5). CQR-style intervals as a secondary for the 80 % PI (N7). | [F] |
| Qin et al. 2024, MobileNetV4, arXiv:2404.10518 | Mobile-efficient UIB backbones, "mostly Pareto optimal" across mobile hardware | — | Supports keeping the backbone | [F] |
| Messager et al. 2016, Nat. Commun. 7:13603, doi:10.1038/ncomms13603 (HydroLAKES) | Global lake volume and depth (Depth_avg, area, etc.) | — | Morphometry context. HydroLAKES is on disk (`processed/hydrolakes`). | [F] (Crossref) |

**Gaps (the search budget prevented closing them; UNVERIFIED):**
- whether any peer-reviewed NAIP (or other 4-band aerial) + in-situ deep-learning water-quality study exists;
- whether "GBM-anchored residual CNN" or "physics-decoder autoencoder for inland water" has prior work.

Report 08 found no NAIP + in-situ study in two searches [R08]. A dated Scopus/Scholar check
(`docs/novelty_checks/lite.md`) is required before any "first" wording.

**Bottom line from the literature.**
- On sparse in-situ labels, the published state of practice for *this kind of imagery* is a GBM on colour
  statistics plus location. That is CyFi, and it is our "GBM + context".
- Deep RGB regressors (HydroVision) need hundreds of thousands of images and still give weak turbidity.
- No published method makes a 4-band 8-bit aerial CNN beat the GBM. Our best chance is to *combine* the two and to
  pick tasks where spatial structure matters.

---

## 3. Is the current architecture optimal? (honest)

**For the budget, the backbone is a sound choice.**
- MobileNetV4-Conv-S + a 48-channel FPN uses 44 % of the parameter budget, 56 % of the GMAC budget and 41 % of the
  INT8 size budget.
- It runs at 30 ms on the desktop proxy.
- Swapping to EfficientNet-Lite0, EfficientViT, a ViT, or a foundation-model encoder (DOFA, DINOv3, Prithvi) would
  add cost without adding the information that is missing. Report 08 §c.4 [R08] found no foundation model with a
  validated water-constituent regression benchmark.

**It is not optimal for the task, in seven ways.** In expected order of impact:

| # | Deficit | Evidence | Fix (where) |
|---|---|---|---|
| D1 | No scene-radiometry input (the GBM has one) | `film_source: scene` (8 terms) was never selected; the ctx config drops it | Scene descriptor (52 statistics + gsd) via FiLM/tokens (ANCHOR-Lite) |
| D2 | No location/season input in selected models | GBM + context 36.5 vs student 48.9 | Context tokens (ANCHOR-Lite; the queued ctx config tests a partial version) |
| D3 | The network re-learns the colour→clarity map the GBM already captures | Student log-R² below the GBM's on Secchi | Anchor on a cross-fitted GBM and learn only the residual (ANCHOR-Lite) |
| D4 | Censored rows and three label types are dropped | Censored Secchi 316, TSS 282, chl-a 162; FNU 471, NTRU 323, SSC 456 already matched | Tobit/censored NLL; unit-aware turbidity heads; SSC head |
| D5 | The SSL objective destroys colour | §1.2.5 | Masked-image reconstruction (SparK-style) or physics reconstruction (CIR-Lite-R) |
| D6 | Fixed ~30 m centre-window pooling; narrow channels and small ponds get mixed pixels | river_stream Secchi 126.6 (SSL run); X1 stratum exists | Learned water-attention pooling over the whole chip (ANCHOR-Lite) or a set encoder (WaterSet-Lite) |
| D7 | KD covers 25 % of chips and did not replicate | H2 on Test-A null | Keep as in the leader for comparability; do not invest further |

**What would not help, or is ruled out:**
- a larger backbone (rule 5; budget);
- more SimSiam;
- per-region models (they fragment ≤ 10 k labels);
- spectral "hallucination" presented as a measurement (report 08 §c.3);
- de-shrinking calibration of μ (it hurts MdSA).

**Verdict.**
- The *backbone* is near-optimal for the budget.
- The *model* (inputs, objective, output structure) is not. Fixing D1–D4 is necessary before any architecture claim
  can be fair.
- Even then, the honest expected ceiling on Secchi is a tie to small gains over GBM-parity. Real gains are plausible
  only where spatial structure carries information (turbidity, rivers, small water bodies, plume strata).

---

## 4. Ranked architecture proposals

### Common fair-test frame (applies to every design)

**Comparators,** all refitted on identical train rows, including new labels and censored rows:

| ID | Comparator | Inputs / notes |
|---|---|---|
| C1 | GBM-RGBN | The frozen 45 features |
| **C2** | **GBM-parity** | C1 + all 52 DOQQ scene statistics + lat/lon/DOY + site class + HydroLAKES morphometry (Depth_avg, log area, shoreline development, elevation, missing flag) + turbidity unit flag. Fixed LightGBM hyper-parameters (400 trees, lr 0.03, 15 leaves), mean of 3 seeds. |
| C2-AFT | C2 with XGBoost AFT | Used where censored rows enter. Confirm the supported censoring types first. |
| C2-TPF (optional) | TabPFN v2 on the C2 features | Needs a package install: lead or infra decision. If admitted, it is a "simple" comparator. Decide **before** any result. |
| C3 | GBM-stacked | C2 + a 16-d PCA of the frozen leader student's pooled embedding. The "GBM gets the CNN features" parity check: if C3 ties ANCHOR-Lite, the gain is credited to CNN features, not to the end-to-end residual design. |
| C4a | Published band-ratio forms | Kloiber Secchi; Nechad/Dogliotti-form turbidity/TSS; val-best form |
| C4b | **Published deep model: HydroVision-style DenseNet121** | ImageNet init, 4-ch stem, global-pool regressor, β-NLL heads, same crops, augmentation and rows. About 8 M params; not drone-bound because it is a comparator. |
| C4c | CyFi-style LightGBM | ≈ GBM + lat/lon/DOY on colour statistics; the existing `gbm_rgbn_ctx` |
| C5 | Current national leader | crop256 KD |
| C6 | Location kNN, location/season GBM, train median | — |

- **Strongest simple baseline:** the val-best of {C1, C2, C2-AFT, C2-TPF if admitted, GBM + context}.
- **Strongest published method:** the val-best of {C4a, C4b, C4c}.
- Both are fixed on val and written into the freeze candidate before the look.

**Statistics,** unchanged from the protocol:
- MdSA with a water-body cluster bootstrap (B = 2000) and paired Δ on identical chips;
- one-sided p from the correct tail;
- a comparison counts only if Holm rejects **and** the CI excludes 0 (`evaluate.holm_with_ci`);
- seed rule: Δ < 0 in each of seeds 0, 1 and 2;
- the 3-seed ensemble is the headline; INT8 seed 0 is a separate deployable family;
- N2 minimum n (≥ 10 chips per target, ≥ 5 water bodies for a paired comparison);
- for a confirmatory family on a *new* block: ≥ 30 chips and ≥ 10 water bodies per target, otherwise descriptive.

**Configuration caps.**
- National-2 has used **6 of 8**: crop128, crop192, crop256, SSL-crop128, SSL-crop256 (running) and ctx-crop256
  (running, 2026-11-10 midday). **Two slots remain.**
- Each new track (P2, P3) gets its own cap of 8 configs × 3 seeds.
- Comparators (C2–C4b) are counted separately as baselines, with fixed hyper-parameters and no tuning.

**Timing basis.** The measured crop256 national-2 epoch takes 17.5 s on a 2080 Ti (`train_log.txt`). So 40 epochs ×
3 seeds ≈ 0.6–0.7 GPU-h per config.

---

### 4.1 Rank 1: ANCHOR-Lite (GBM-anchored, context-complete, multi-target residual CNN). Score 6.5 / 10

**One line.** The network predicts a bounded *residual* on top of a cross-fitted GBM-parity prediction. It sees
everything the GBM sees (scene statistics, location, season, morphometry) plus the full-resolution image. Its
capacity therefore goes only to what per-chip statistics cannot encode: plume texture, narrow-channel and shoreline
geometry, bottom visibility and glint patterns.

**Architecture** (≈ 1.40 M params; ≈ 1.13 GMACs per 512² tile; INT8 CNN ≈ 1.7 MB + one LightGBM seed ≈ 0.5–1 MB;
≈ 35 ms proxy):

1. **Inputs** [inference]:
   - chip uint8 4×256×256 (the full chip, the lever that helped);
   - scene descriptor s: 52 DOQQ statistics + gsd, standardised. On a drone these are per-flight statistics: a
     deployment caveat, since a flight footprint is ≪ a DOQQ;
   - context c: lat/lon as unit-sphere xyz, DOY sin/cos, site class, and HydroLAKES morphometry with a missing flag
     (an offline lookup table on the device; size to be measured);
   - anchor a_t: the GBM-parity prediction (log10) for each target t.
2. **Image branch:** MobileNetV4-Conv-S (4-ch stem) + the existing 48-ch FPN at stride 4. Initialised from the
   national-2 crop256 leader's seed-matched weights (trained on the same train rows, so no leakage).
3. **Conditioning:** FiLM on the decoder from MLP([s, c]) (66 → 64 → 96), zero-initialised as in `model.py`.
4. **Pooling:** two pooled vectors are concatenated (96-d):
   - (a) the existing water-weighted centre window (~30 m);
   - (b) learned water-attention pooling over the whole chip. Its logits come from decoder features plus a fixed
     radial distance encoding, masked by dark-NIR water.
5. **Residual heads** for each target t ∈ {Secchi, turbidity-NTU, turbidity-FNU, TSS, SSC, chl-a}, plus
   transparency tube and DOC on P2 only:
   - MLP (96 + 64 → 64 → 2) gives (Δ_t, log σ²_t);
   - **μ_t = a_t + κ_t·tanh(Δ_t/κ_t)**, with κ fixed a priori: 0.3 log10 for Secchi, 0.5 for the rest;
   - NTRU rows go to the NTU head with one learned unit offset. JTU is dropped.
6. **Censoring:** right-censored Secchi ("on bottom" / > reading) and left-censored TSS/chl-a use a Tobit NLL in log
   space.

**Training:**
- Anchors come from 5-fold GroupKFold by `national2_wb` on train: out-of-fold anchors for train rows, and the mean
  of the 5 fold models for val/test rows. This is the standard stacking guard against leakage.
- Loss: Σ_t w_t·β-NLL (β = 0.5; Tobit for censored rows) + 0.3·KD (as in the leader) + 0.01·‖Δ‖².
  w = 1 for Secchi, NTU, FNU and TSS; 0.5 for SSC and chl-a. All fixed now, never tuned.
- Radiometric augmentation as now. **Required:** apply the *same* per-band gain/gamma/offset to the scene quantiles.
  Quantiles commute exactly with monotone per-band maps, so scene and chip stay consistent. The anchor is a fixed
  input and is not recomputed.
- 40 epochs fixed; AdamW 1e-3 for the heads, 5e-4 for the backbone; batch 32; AMP fp16.

**Inductive bias the GBM lacks.**
- Spatial structure at 0.6 m within a 154 m field: plumes, gradients, bank and bottom contamination, glint.
- Coupling across targets and units: FNU, NTU and SSC share one trunk.
- Censored likelihoods.

The anchor removes the network's handicap on the tabular part of the signal (Grinsztajn [F]).

**Changes applied (my critique of the obvious version):**
- (i) Out-of-fold anchors, never in-sample.
- (ii) A bounded residual, so the hybrid cannot be worse than the GBM by more than ×2 to ×3.2 per chip, and its
  failure mode is "= GBM".
- (iii) Synchronous augmentation of the scene statistics.
- (iv) C3 stacking parity, so that "GBM + CNN features" cannot silently explain the gain.
- (v) A published deep comparator (C4b), so a §15.1 "published method re-run" exists.

**Fair test.**
- National Test-N (if admitted before the national freeze) and Block P2.
- Primary comparisons: vs the strongest simple and strongest published. Secondary: vs C3, C5, and every C6 baseline.
- Scoped wording: "hybrid (GBM-anchored CNN) vs GBM-parity". It is never described as "CNN beats GBM".

**Ablations** (national: slot 8 only; the rest in the P2 cap):

| ID | Change | What it tests | Where |
|---|---|---|---|
| A1 | No anchor (context-complete direct student) | Whether anchoring is the lever | National slot 8 |
| A3 | Anchor, no s/c tokens | Whether the image residual needs the context | P2 cap |
| A4 | Original 4 targets only, no new units or censored rows | Multi-target data gain (GBMs refitted on the same reduced rows) | P2 cap |
| A5 | Centre-window pooling only | Learned whole-chip pooling (D6) | P2 cap |

**Budget.**
- National: 2 configs ≈ 1.4 GPU-h.
- P2: up to 4 configs ≈ 3 GPU-h.
- C4b DenseNet121 comparator: ≈ 1.5 GPU-h national + ≈ 1 GPU-h P2, chunked one seed per job.
- Anchors, C2 and C3 on CPU (minutes).
- **Total ≈ 6 GPU-h + ≈ 2.5 for C4b.**
- Peak VRAM ≈ 4.5 GB (C4b ≈ 7 GB, estimated, batch 32 fp16), so it can share an 11 GB card.

**Expected (val, selection-optimistic):**

| Target | Expected vs GBM-parity | Notes |
|---|---|---|
| Secchi | −1 to −4 MdSA | GBM-parity expected ≈ 34–36 |
| NTU turbidity | −15 to −30 | |
| TSS | −3 to −10 | |
| FNU (P2) | unknown | |

Probabilities are in §9.

**Export.**
- The frozen `ExportWrapper` has no conditioning input. The handoff (2026-11-10) already notes that config 6 needs a
  two-input national export.
- ANCHOR-Lite needs a new multi-input wrapper (chip, s, c, anchor) in a new, unhashed file, plus an on-device
  LightGBM (one seed) and the feature extractor.
- Budget compliance is checked on the whole pipeline, not on the CNN alone.

**Risks.**
- With these inputs the residual may carry almost no Secchi information. Then ANCHOR = GBM-parity, a tie.
- The OOF-vs-full-model anchor shift is small but real.
- A deployment pipeline with GBM features is more complex than a pure CNN (still in budget).

---

### 4.2 Rank 2: WaterSet-Lite (learned distribution summary over water pixels). Score 5.0 / 10

**One line.** Replace hand-made water-pixel quantiles with a learned, permutation-invariant summary of *all* water
pixels in the chip. This is a set encoder over pixel tokens plus scene, context and anchor tokens. Used as
ANCHOR-Lite's alternative residual branch ("ANCHOR-Set").

**Architecture** (≈ 0.15 M params; ≈ 20–40 MMACs; INT8 < 0.3 MB):
- **Tokens:**
  - Up to N = 1,024 dark-NIR water pixels sampled from the full 256 px chip, each with ~16 features: log DN
    R/G/B/NIR, log band ratios, DN / scene-p50, distance to chip centre (m), distance to shore (m, from the
    water-mask distance transform), and local 5×5 std per band. Linear 16 → 64.
  - 3 global tokens: s, c and the anchor vector.
- **Encoder:** 2 ISAB blocks (16 inducing points, d = 64, 4 heads; Set Transformer [F]), then PMA with 4 seeds,
  then the residual heads as in §4.1.

**Inductive bias.**
- Learned quantiles and joint-band distributions instead of fixed q25/q50/q75.
- Explicit distance-to-shore, so narrow rivers and small ponds (X1) are summarised from *their own* water pixels
  rather than from a fixed window that contains land.
- Position-free, so the centre misregistration of WQP coordinates matters less.

**Changes applied:**
- It is a residual on the anchor, not a stand-alone model (Grinsztajn: a stand-alone set network on tabular-like
  features would probably tie or lose).
- Pixel sampling is seeded and frozen at evaluation (the same 1,024 pixels per chip).

**Fair test:** as §4.1, inside the P2 cap (the national slots go to ANCHOR and A1).
**Budget:** ≈ 0.3 GPU-h per config, or CPU at 6 threads (≈ 1–2 h). Total ≈ 1.5 GPU-h. Peak VRAM ≈ 1.5 GB.
**Expected:** a tie with ANCHOR-CNN on lakes; possibly better on rivers and X1. Low cost makes it worth one or two
P2 slots.

---

### 4.3 Rank 3: CIR-Lite-R (reduced Calibrate-Invert-Render). Score 4.0 / 10

The full CIR-Lite design (calibration hypernetwork distilled from S2, 7 IOP/bottom/glint latents, a differentiable
shallow-water renderer, physics heads for 8 targets, about 13 GPU-h) is scientifically attractive but over-scoped.

**Required changes (my critique):**

1. **Identifiability.** 7 per-pixel latents cannot be identified from 4 broad 8-bit bands. Reconstruction would be
   satisfied by many latent combinations, and the KL priors would decide them.
   - Reduce to **4 latents**: log b_bp(red), log a_nw(443) (combined non-water absorption with a QAA-prior spectral
     shape), glint g_s, and one shape scalar.
   - Fix η by the QAA empirical relation (Lee 2002 [F]).
   - Drop the bottom and H latents from the main model. The shallow-water renderer becomes ablation A7 on X1 only.
2. **Gate before GPU spend.** The calibration stage C (a monotone per-scene curve from the 52 scene statistics,
   supervised by S2 L2A block reflectance) is fitted first on CPU. Go/no-go, pre-declared:
   - val/val_n S2 water-cell RMSE must beat a per-scene linear gain + offset fit by ≥ 10 % relative;
   - C3-physics (GBM-parity + ρ̂ statistics + QAA IOPs + Lee 2015 Secchi as features) is fitted **before** any
     CIR network val result exists.
   - If the gate fails, CIR-Lite-R is not trained, saving ≈ 6 GPU-h.
3. **Physical constants.** Each one written from memory is checked against Lee 2002 / Lee 2015 / Nechad 2010 /
   Dogliotti 2015 (all [F]) before coding. Constants absent from them (WASI bottom spectra, Lee 2013 Kd) are verified
   or removed.
4. **Calibration-target caveat.** S2 L2A water reflectance is biased (adjacency; Sen2Cor is not water-specific).
   About 2 M cell targets come from about 2.6 k scenes, so the effective sample size is the number of scenes. Report
   the calibration diagnostic per state.
5. **Scope.**
   - P2 and France (P3) only. It cannot be ready for the national freeze, and no national slots remain.
   - Context enters only through the bounded residual.
   - Latent IOPs are never claimed.
6. **Parity.** The GBM gets ρ̂ statistics (C3-physics). A gain is credited to the network only if C3-physics does not
   tie it.

**Inductive bias that survives the cuts:**
- monotone physical extrapolation (turbidity and TSS linear in b_bp; Secchi ∝ 1/Kd);
- colour-preserving self-supervision on ~47 k unlabelled chips (25 k SSL + 22.6 k national_pm7);
- privileged radiometric supervision (generalized distillation, Lopez-Paz [F]);
- a radiometric prior aimed at new sensors (P3 France).

**Budget:**
- P1 calibration on CPU;
- P2 physics pretraining ≈ 1 GPU-h in 2 chunks;
- 4 configs (main, A1 direct heads, A2 no L_rec, A7 shallow) ≈ 4 GPU-h;
- evaluation 0.5 GPU-h; contingency 1.5 GPU-h.
- **≈ 7 GPU-h. Peak VRAM ≈ 5 GB.**

**Expected:**
- no Secchi gain over C2;
- possible gains on X2 (high turbidity) and on France transfer;
- P(scoped win) ≈ 5–10 %.

It is ranked third because the cost per expected gain is high and the gate may fail.

---

### 4.4 Rank 4: SparK-style masked-image SSL init (replacing SimSiam). Score 3.0 / 10

- **Method.** Masked reconstruction of DN on MobileNetV4-S with a light hierarchical decoder (SparK [F]). A dense
  mask-token variant is acceptable if sparse convolution is unavailable under torch 2.7 / CUDA 11.8.
- **Corpus:** the 25 k SSL chips + 22.6 k national_pm7 train chips. No new disk. Exclusion manifest as before.
- **Augmentation:** geometric only, plus *mild* (±5 %) gain, so that colour stays informative.
- **Budget:** pretraining ≈ 3 GPU-h (2 chunks of ≤ 1.5 h) + 1 config ≈ 0.7 GPU-h. Peak VRAM ≈ 6 GB.
- **Expected:** 0 to −2 MdSA vs the same model without SSL. 10.5 k labels is not label-starved, so SSL gains should
  be small.
- **Placement:** one P2 slot at most, and only if a slot remains after ANCHOR and WaterSet.

### 4.5 Rank 5: the queued context-FiLM config (already counted). Score 2.5 / 10

- `mnv4s_raw_crop256_kd03_ctx_national2` gets lat/lon/DOY but *loses* the scene statistics. It is therefore not at
  information parity with GBM + context.
- Let it run; it is pre-declared and counted.
- Do not add more configs of this kind. ANCHOR's A1 (slot 8) is its context-complete successor.

### Rejected (with reason)

| Rejected design | Reason |
|---|---|
| Larger backbones, ViTs, foundation encoders (DOFA, DINOv3, Prithvi) as the student | Budget, and no information gain (rule 5) |
| More SimSiam or other contrastive SSL with colour jitter | Spent and harmful (§1.2.5) |
| Spectral super-resolution presented as a measurement | Ill-posed; report 08 §c.3 |
| TabPFN as the drone student | Needs the training set at inference; not drone-deployable. Comparator only. |
| Per-region (CyFi-style) networks | Fragment ≤ 10 k labels |
| Full 7-latent CIR-Lite | Identifiability, cost, constants unchecked. Replaced by CIR-Lite-R. |

---

## 5. New numeric predictive targets (multi-task)

Counts marked **verified** were re-read this session. **ESTIMATE** means scaled from measured yields and requires a
counts-only check first.

| Target | Unit | Label source | Matched rows (status) | Already chipped? | Optical basis | Proposed role |
|---|---|---|---|---|---|---|
| Turbidity, FNU | FNU | WQP (in `national_pm7`) | **471 (verified)** | **yes, 0 GB** | Strong (backscatter, red/NIR) | New primary on P2; train-role rows join national-2b training; secondary on Test-N |
| Turbidity, NTRU | NTRU | WQP (in `national_pm7`) | **323 (verified)** | **yes** | Strong | Pooled with NTU through a learned unit offset |
| SSC | mg/L | WQP (in `national_pm7`) | **456 (verified; 231 sites; mostly rivers)** | **yes** | Strong (with TSS) | Secondary |
| Censored Secchi / TSS / chl-a / turbidity | — | `national_pm7` | **316 / 282 / 162 / 26 (verified)** | yes | — | Tobit in the network, AFT in the GBM (parity) |
| Turbidity, FNU (sondes) | FNU | USGS NWIS daily mean on the NAIP date (`processed/sonde_daily`) | ESTIMATE 600–1,000 site-days after exclusions (data step) | no (≈ 0.1–0.2 GB) | Strong | P2 primary |
| Transparency tube | cm | WQP: 632,927 rows / 27,597 sites / 34 states, 27.75 % censored (**verified**, all years) | ESTIMATE 400–600 at ±1 d | no (≈ 0.1 GB) | Strong (clarity) | P2 secondary, right-censored at tube length |
| DOC | mg/L | WQP: 673,329 rows / 60,886 sites (**verified**, all years) | ESTIMATE 700–1,400 | no (≈ 0.1–0.2 GB) | Weak (CDOM proxy; report 08: CDOM MdSA 85–110 % [R08]) | Auxiliary only, never claimed |
| Chl-a | µg/L | already in the labels (N7) | 5,552 matched (**verified**) | yes | Weak without red-edge | Secondary (already registered) |
| NLA lake-level Secchi, turbidity, chl-a, colour (PCU), DOC, TSS | — | `processed/nars_lakes/chemistry_wide` (columns verified: `secchi_m`, `turbidity_ntu`, `chla_ug_l`, `color_pcu`, `doc_mg_l`, `tss_mg_l`) | ESTIMATE ~90–120 lakes per cycle at ±7 d (data step) | no | as above | P4 (conditional) |
| NLA physical habitat: AMITOTAL (macrophyte cover), LITCVRQC3OE (littoral cover), RVEGQC3OE (riparian vegetation), RDIS_IX (riparian disturbance) | fraction / index | `nars_lakes/phab_key_wide` (3,466 rows, columns verified) | ESTIMATE 700–1,100 lake-visits at ±45 d (all cycles, before exclusions) | no | **Spatial texture: a task where colour statistics should fail by construction** | P4 (conditional; scope extension, lead to rule) |
| S2 L2A block reflectance B2/B3/B4/B8 (+ B5) | reflectance | Co-incident S2 chips: 12,132 Secchi rows have an S2 chip (**verified**) | — | yes | — | Training-only privileged auxiliary (CIR-Lite-R) |

**Not recommended:**

| Target | Reason |
|---|---|
| TP / TN | Non-optical. Any skill is a trophic proxy through location. ESTIMATE 5–11 k / 2–4.5 k matchups would cost ≈ 0.7–1.5 GB of chips, which the disk cap cannot absorb. Only as auxiliary training targets if the cap is raised, and never claimed. |
| Phycocyanin | 19,814 rows / 1,237 sites; too few at NAIP dates; not retrievable without 620 nm [R08] |
| Microcystins | 54 % censored; non-optical |
| a_CDOM(440) | 5,996 rows; not retrievable [R08] |

**Multi-task design rule (pre-register):**
- one shared trunk, with per-target residual heads (§4.1);
- fixed task weights;
- the unit-aware turbidity head;
- selection score S unchanged (Secchi, turbidity-NTU, TSS), so the new targets cannot steer selection.

---

## 6. More data: sources, licence and size

**Disk.** `df -h` this session: **86 GB free** (floor 60 GB). The handoff projects M7 new data at ≈ 9.4 GB of its
10 GB cap. The national-1 train cache is already gone; the M7 run root is 4.8 GB. Anything beyond ≈ 0.6 GB needs
either deletion of regenerable caches after the national freeze or a lead-approved cap increase. I propose ≤ 1.5 GB
in total.

| # | Source | URL / location | Licence | New disk | Counts | Status |
|---|---|---|---|---|---|---|
| D1 | Extra labels already matched: FNU, NTRU, SSC, censored rows | `processed/naip_matchups/national_pm7/matchups.parquet` | WQP US public data (cite doi:10.5066/P9QRKUVJ per the datasheet) | **0** | 471 + 323 + 456 + 786 censored (verified) | **Do first** (labels-only rebuild: "national-2b") |
| D2 | USGS NWIS continuous turbidity, daily (FNU) | `processed/sonde_daily` (on disk); waterservices.usgs.gov | US public domain [L] | ≈ 0.1–0.2 GB of NAIP chips | ~958 sites (data step); ESTIMATE 600–1,000 site-days | P2 core. **Mandatory exclusions:** 729 `sonde_newsites` (sha 6200a77c), M1 `sonde_split` test sites, M6 test-role sites, dates ≥ 2023-10-01 |
| D3 | WQP transparency tube, DOC, FNU/NTRU at *new* water bodies | `processed/wqp_discrete` (on disk) | as D1 | ≈ 0.1–0.3 GB | ESTIMATES in §5 | P2. Needs a non-JRC water test (NHD flowline or a NAIP-NDWI test) for narrow streams. |
| D4 | NLA 2012/2017/2022 lake rings | `processed/nars_lakes` (on disk); EPA NARS | US Government work; no licence statement on the hub page (datasheet, UNVERIFIED) | ≈ 0.25 GB if rings are stored at 2 m GSD; ≈ 1 GB at native resolution | 4,918 visits (1,231 / 1,210 / 1,225 in 2012 / 17 / 22, verified) | Conditional (P4). **Heavy exclusions:** Fuse tier-C test sites (2,211 NLA visits), D2n test (149), M2-orbit NLA 2017 test events (218 visits); M5-X uses NLA 2022 as *test* rows. |
| D5 | France: Naïades via Hub'Eau + IGN BD ORTHO | https://hubeau.eaufrance.fr/page/api-qualite-cours-deau (fetched [F]: river and lake physico-chemistry from Naïades); BD ORTHO via cartes.gouv.fr (JS-only page) | Hub'Eau: Etalab Licence Ouverte per its CGU [L], not shown on the API page. BD ORTHO licence and **IRC/RVB band layout: UNVERIFIED this session** | ≈ 0.2–0.4 GB | Data step, from live API counts since 2018: Secchi 61,053, turbidity 461,778, MES 487,416, chl-a 206,000 analyses. ESTIMATE ±1 d site-days: turbidity 200–450, MES 200–450, Secchi 150–300 (±7 d). | Conditional (P3). Coordinate with M2 Track G (non-US test lakes × 2024–25). |
| D6 | SSL corpus reuse (25 k SSL chips + 22.6 k national_pm7 train chips) | on disk | as NAIP | 0 | ~47 k chips | For SparK or CIR-Lite-R pretraining |

**Not recommended now:**

| Source | Reason |
|---|---|
| NAIP 2024+ | Not on Planetary Computer (STAC returns 0 items, handoff 2026-11-09); needs an account or requester-pays, i.e. a user action |
| USGS HIVIS (HydroVision's data) | > 500 k oblique images ≫ cap; different geometry; site-held-out skill unknown |
| LAGOS-US LIMNO | EDI package id still unresolved: my direct fetch of an EDI page did not resolve it [U]. Heavy overlap with WQP agencies. |
| PlanetScope | University-only access [R08] |

---

## 7. Harder task tracks

**Rule check (`sota_campaign.md` rule 2).** Each track below is justified by deployment practice, and no M7 val
result exists on any of them. Two cases are *not* eligible:
- the river/stream class on national val, where by-class val results already exist;
- glint robustness, where val results exist.

| Track | Practical justification | Unseen test block | Power (counts-only check first) | Status |
|---|---|---|---|---|
| **T-N (existing) + national-2b labels** | — | Test-N (773 chips, 42 states; unchanged) | Turbidity NTU 486 / 135 wb, Secchi lakes 301 / 152, TSS 270 / 105 (report, verified). Protocol power: CI half-width ≈ 13 (turbidity), 9 (Secchi), 17 (TSS). FNU/NTRU/SSC on Test-N: count first. | Strata X1–X4 already pre-registered (N1). Add FNU/SSC as **secondary** Test-N targets **before the national freeze**. |
| **P2: rivers and sondes, FNU turbidity at new water bodies** | Drones beat 10–20 m satellites on narrow rivers and small water bodies. USGS sondes report FNU, and that is where operational turbidity monitoring happens. | A salted hash assigns 40 % of new water bodies (NWIS sites plus WQP FNU/NTRU/tube sites outside every existing Lite water body and its 2 km buffer) to test-P2 **before any chip is extracted**. Exclusions as D2. | ESTIMATE 150–400 test chips / 50–150 sites for FNU. Confirmatory only if ≥ 30 chips and ≥ 10 water bodies; otherwise descriptive. Strata: "> train P90 FNU" and channel width < 31 px (from NHD or the water mask, train-only thresholds). | **Pre-register now** (new file `lite_p2_protocol.md`) |
| **P3: France zero-shot (new sensor and country)** | A drone deployed abroad meets a new camera colour balance and new limnology. This is the stated radiometric failure mode. | All French site-days. No French data in training; models trained on US data are frozen first. | ESTIMATE turbidity 200–450, MES 200–450, Secchi 150–300 site-days. | Conditional: lead approval, M2 Track G coordination, BD ORTHO band and licence check |
| **P4: NLA whole-lake (clarity, colour, littoral/riparian habitat)** | A national probability survey. A drone shoreline-and-lake survey could stand in for part of a field visit. Habitat cover is spatial, so per-pixel statistics should fail by construction. | NLA 2022 (never trained on by M7; it is M5-X *test*, which M7 test use does not contaminate). Train on 2012/2017 minus the Fuse/D2n/M2 exclusions. | ESTIMATE ~100 lakes for Secchi at ±7 d in 2022. More for habitat at ±45 d. Training rows may be too few after exclusions: count first. | Conditional; also a scope extension (habitat ≠ water quality). Lead to rule. |
| Not eligible | — | — | — | Test-A (spent); glint as a new track; "river class" on national val; test_time_n beyond descriptive |

**Why these make the task harder in the right way.** In each, the information a GBM on centre medians lacks should
matter more:
- narrow channels and mixed pixels (P2);
- a radiometric domain shift (P3);
- spatial habitat structure (P4).

Each also adds labels that every comparator gets too (parity).

---

## 8. Sequencing within the shared 2-GPU budget and config caps

**Hard constraints:**
- GPUs 0/1 only, through the lock helper, ≤ 2 h per job. Update: the handoff (2026-11-10 midday) records a GPU rule
  change to GPUs 0–9 via the lock helper with `--min-free-mb`, still ≤ 2 h per job and never on another user's GPU.
  This plan is sized for 0/1 and only gets faster under the new rule; the lead should confirm which rule applies;
- about 7 workstreams, so roughly 6–7 GPU-h a day of fair share at best;
- CPU ≤ 6 threads;
- national freeze deadline ≈ 2026-11-20 (lead: ≤ 9 working days from 2026-11-09);
- national-2: 2 config slots left;
- disk: ≤ 1.5 GB new, subject to the cap ruling.

### Step 0: pre-register first (paper only; before any new val result)

**Amendment N8** to `lite_national_protocol.md`, before the national freeze:

1. **National-2b label version.**
   - FNU, NTRU, SSC and censored rows on existing chips enter training only when the chip is train-role.
   - Evaluation labels for Secchi/NTU/TSS are unchanged.
   - FNU and SSC on Test-N are *secondary* targets (N2 n-rules).
2. **Comparator set.**
   - C2 GBM-parity, C2-AFT, C3 stacking and C4b HydroVision-style DenseNet121.
   - Whether C2-TabPFN is admitted (lead decision; package install).
   - The fixed definitions of "strongest simple" and "strongest published" as in §4.
3. **The two remaining config slots, named now:**
   - slot 7: ANCHOR-Lite;
   - slot 8: A1, the context-complete student without the anchor.
   - With every constant fixed: κ, task weights, anchor folds, augmentation sync.
   - SSL-crop256 (config 5) and ctx (config 6) were relaunched on 2026-11-10 midday on GPUs 2 and 5 (handoff), so
     withdrawing config 5 to free a slot is no longer an option. The no-KD crop256 H2 comparator (GPU 7) is not a
     candidate and does not use a slot.
4. **Claim-label wording.**
   - State that the frozen §8a label rule ("first-in-class" needs Secchi AND turbidity-or-TSS) remains the national
     headline.
   - Any **target-scoped** national label (for example "SOTA on national turbidity") needs a lead ruling, with the
     disclosure that it was proposed *after* val showed turbidity as the student's best target.
   - My recommendation: report national turbidity as the per-comparison Holm results already pre-registered in H1,
     and reserve scoped SOTA wording for P2.

**New file `lite_p2_protocol.md`, before any P2 chip is extracted:**
- sources and exclusions;
- the salted 40 % test assignment and its sha;
- targets (FNU primary; NTU, TSS and tube secondary);
- the Holm family {FNU} × {strongest simple, strongest published}, with secondaries;
- cap 8;
- power rule;
- a scoped label rule: "SOTA on P2 FNU turbidity" iff Holm ∧ CI excludes 0 vs both comparators, plus the seed rule.

### Step 1 (days 1–2; CPU, network)

Counts-only checks:
- FNU/NTRU/SSC by role (no values);
- NWIS × NAIP same-day after exclusions;
- WQP tube/DOC/FNU at new water bodies;
- France via API counts (if P3 is approved);
- NLA × NAIP dates.

Then:
- report sizes; run `df` first;
- build the national-2b labels (0 GB);
- fit C2, C2-AFT and the anchors on CPU.

### Step 2 (days 2–6; ≈ 4–5 GPU-h)

1. The running SSL-crop256 and ctx configs finish (already counted).
2. ANCHOR-Lite slot 7 and A1 slot 8 (≈ 0.7 GPU-h each).
3. The C4b DenseNet121 comparator (3 jobs ≤ 0.6 h each).
4. Fit C3 (CPU).
5. National selection (S rule), export of INT8 seed 0, candidate, dry run on val ∪ val_n, pre-test review, freeze.
   Target: by about 2026-11-18, leaving slack before the deadline.
6. **If ANCHOR is not ready by about 2026-11-16, freeze the national track without it.** ANCHOR then goes to P2 only.
   Do not slip the deadline.

### Step 3 (days 5–12; ≈ 6–9 GPU-h)

1. P2 builds (≤ 0.5 GB).
2. P2 configs within cap 8, in this order:
   1. ANCHOR-Lite;
   2. ANCHOR-Set (WaterSet);
   3. A3, A4, A5;
   4. CIR-Lite-R main + A1-direct, **only if the P1 calibration gate passes**;
   5. SparK init if a slot remains.
3. Selection on P2-val. Freeze. Single look under the v1.7 hardening.

### Step 4 (days 10–15; mostly CPU)

- If P3 is approved: build France (≤ 0.4 GB) and score the **frozen** P2/national models zero-shot. There is no
  training, so it costs ≈ 0.3 GPU-h of inference.
- If P4 is approved: as for P3, after the exclusion audit.

**Total ≈ 15–20 GPU-h over about 3 weeks,** none above 2 h per job, peak ≤ 7 GB, so it can co-reside on an 11 GB
card.

---

## 9. Honest outlook

| Track / target | Comparator bar | P(Holm ∧ CI-excludes-0 win, seed rule) | Main reason |
|---|---|---|---|
| National Test-N, Secchi | Strongest simple (GBM + context / GBM-parity: val 36.5 or better) | **≈ 10 %** | The student is 12.4 behind GBM + context on val. ANCHOR can at best add a small residual. |
| National Test-N, NTU turbidity | GBM-RGBN and band-ratio (frozen H1) | ≈ 35–40 % | Val Δ −25.7 is ~2× the expected Test-N half-width (≈ 13), but val is selection-optimistic, and Test-A turbidity reversed (n = 44 wb) |
| National Test-N, NTU turbidity | Strongest simple incl. GBM + context / parity, and published incl. C4b | ≈ 25–30 % | GBM + context Δ on val was −20.8 [−55.1, +9.5] |
| National Test-N, TSS | Strongest simple / published | ≈ 10 % | Val Δ −6.4 [−24.7, +2.3]; half-width ≈ 17 |
| National headline label under the frozen §8a rule ("first-in-class") | Every baseline, Secchi AND turbidity-or-TSS | ≈ 5 % | It needs Secchi |
| P2 FNU turbidity (new block) | Strongest simple / published | ≈ 30 % (wide: 15–45 %) | Rivers and narrow channels favour spatial models. Counts and power are not yet known. |
| P3 France zero-shot | — | ≈ 10–15 % | Every model degrades under a new sensor; whoever handles radiometry best wins. Power is uncertain. |
| P4 NLA habitat (if admitted) | — | ≈ 20 % | A spatial target favours CNNs, but training rows after exclusions may be too few |

**Overall.**
- At least one §15.1-grade, track-scoped label for M7 within about 3 weeks: **≈ 30 % (20–40 %)**. The likely
  carriers are P2 FNU or national turbidity under a lead-approved scoped rule.
- A Secchi SOTA label: ≈ 10 %.
- The most likely honest outcome is "competitive, with a scoped turbidity advantage". The defensible scientific
  story would then be: "a 1.4 M-parameter drone model with GPS-level context matches the strongest GBM on clarity
  and beats it on turbidity, where spatial structure matters".

**What would change these numbers:**
- If GBM-parity or TabPFN turns out *much* stronger than GBM + context, every probability drops by about a third.
- If the P2 counts come in at the high end of the estimates (≥ 300 test chips / ≥ 100 sites), P2 rises to about
  40 %.
- If ANCHOR's residual is ≈ 0 on val (the hybrid equals GBM-parity), stop investing in architecture for Secchi and
  report M7 as competitive.

**Integrity notes.**
- Test-A is spent and is never reused.
- Every configuration is counted, including withdrawn ones.
- Every comparator gets the new labels and inputs.
- Every scoped label names its track, and losses are reported alongside wins.
