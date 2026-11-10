# M3 SENTINEL-Microbe: architecture research and SOTA plan (2026-11-10)

**Status:** decision document for the lead. This was read-only research: nothing was trained or downloaded, and no
file under `docs/preregistration/` was edited. Every item marked "pre-register" needs a dated, lead-written
amendment **before** any result on it exists (sota_campaign.md rule 2).

**Scope:** NEON 16S (DP1.20141.002, harmonised V4 units) → co-located water chemistry, generalising to unseen
domains, plus external tracks.

**Inputs used:**
- docs/handoff/M3-microbe.md;
- microbe_protocol.md §1–§10.8;
- microbe_ea_protocol.md;
- the registry (`results/registry.jsonl`, 2,628 M3 rows, including the W6 `XGB[genus-CLR]-tuned` rows that
  landed 2026-11-10 12:16 UTC);
- the power report (`microbe_split_lodo_x001_report.json`);
- docs/research/03;
- the workflow's literature, diagnosis, data and design inputs.

**Provenance caveats (read first):**
1. **The workflow input was truncated.** It ended after the first design (NicheGeo) and before the critics' text,
   and no saved copy exists on disk. I could not read the other designs or critiques verbatim. The scratch
   scripts show what the critics checked:
   - `m3crit/analyte_counts.py`, `nn*.py`: GEO label counts;
   - `emp_near.py`, `emp_meta.py`: EMP proximity and metadata;
   - `vocab.py`, `tokstats.py`: vocabulary and token sizes;
   - `m3adv/emp_*.py`: EMP metadata as weak labels.

   I re-ran the decisive check myself (§5) and applied my own critique of every design (§4).
2. **The web-search budget for the session was exhausted** when I started. Citations are either (a) verified
   earlier in this session by the literature agent ("session-verified"), or (b) checked by me through a direct
   DOI/Crossref/DataCite/arXiv fetch ("DOI-checked"). Anything else is marked UNVERIFIED.

---

## 1. Diagnosis: why simple ML wins or ties

### 1.1 Where M3 stands (development LODO only, selection-biased: about 21 SENTINEL and about 30 baseline configurations)

**Mean pooled R² over 8 targets (registry, `neon_dev/lodo_cv`):**
- Ens[V4-Ridge+V4-XGB] 0.325
- V4-XGB+context 0.317
- V4-XGB 0.314
- Ens[Ridge+XGB genus] 0.307
- V4-Ridge 0.291; TabPFN[genus] 0.291; XGB[genus] 0.289; Ridge[genus] 0.284
- XGB[genus]-tuned 0.282 (W6 tuning *lowered* the score)
- SeqClust 0.256; RF 0.252; SeqSet-kmer 0.239; SeqSet-DNABERT 0.174
- Ridge[DOY+location] 0.024

**V4-XGB against the per-target strongest simple baseline** (§15.1 bar; point estimates from the registry):

| Target | V4-XGB | Strongest simple baseline (model) | Δ |
|---|---|---|---|
| TN | 0.092 | 0.106 (Ens[Ridge+XGB]+Ridge[DOY+loc]) | −0.01 |
| TP | 0.284 | 0.305 (Ens[Ridge+XGB]) | −0.02 |
| DOC | 0.354 | 0.328 (XGB[genus+ctx]) | +0.03 |
| NO3 | 0.210 | 0.292 (Ens+ctx) | −0.08 |
| SC | 0.756 | 0.778 (Ridge[genus]) | −0.02 |
| Turbidity | 0.149 | 0.206 (Ridge[genus]) | −0.06 |
| DO | 0.434 | 0.471 (XGB[genus+ctx]) | −0.04 |
| chl-a | 0.236 | 0.219 (TabPFN) | +0.02 |

- H-M3 dev analogue (beat tier0_best, RF and XGB; Holm; both site and domain units): **1 target (SC)**. The bar
  needs 2.
- §15.1 SOTA analogue (beat the strongest simple baseline): **0 targets**.
- Correction to the lead's summary: the season+location tier-0 does **not** tie overall (best tier-0 mean is
  about 0.02). It is strong only for DOC, NO3 and DO in unseen domains. The model ties with *Ridge, TabPFN and
  the ensembles*, not with tier-0.

### 1.2 Root causes, ranked by how much they bind

1. **The effective n for the thing being learned is about 32 sites, not 1,215 events.**
   - The site-identity probe recovers the site from the community 96.2% of the time, and the domain 98.3%.
   - Most predictable variance is between sites. V4-XGB's within-domain anomaly R² is −0.11 (TN), 0.28 (TP),
     0.18 (DOC), 0.14 (NO3), 0.56 (SC), 0.03 (turbidity), 0.48 (DO) and −0.06 (chl-a).
   - Under LODO each fold extrapolates a mapping learned from about 32 site centroids, and domains hold only 1–3
     sites each. With 32 effective points in a space of 500 or more dimensions, every regularised learner
     (ridge, boosted trees, a TabPFN prior) converges to similar answers.
   - Evidence: Ridge ≈ XGB ≈ TabPFN within 0.01–0.03; tuning *hurts* (0.289 → 0.282); late fusion of two simple
     models is the best single number (0.325).
2. **Power.** The minimum detectable ΔR² on dev LODO (80% power, Holm α = 0.05/8, paired site-cluster bootstrap)
   is 0.105 (TP), 0.117 (DOC), 0.141 (DO), 0.155 (SC), 0.177 (NO3), 0.190 (turbidity) and 0.203 (TN)
   (`microbe_split_lodo_x001_report.json`).
   - The 2024 holdout has 214 events against 1,215, over the same 33 site clusters. Its MDD is roughly 1.5–2.4×
     larger (≈ 0.16–0.45). This is my approximation and **must be replaced by a counts-only power check**.
   - The architecture differences observed so far (≤ 0.04) are invisible at that resolution. *No* architecture
     change can earn the bar on the 2024 block unless it gains more than about 0.15 R² on two targets.
3. **Neural set models are data-starved and carry no prior.**
   - SeqSet has 0.48–8.2M parameters on about 1,005 fit events, with a best epoch of about 11–17.
   - The frozen-embedding variants lost information: mean-pooled DNABERT-S scored −0.006, and whole-amplicon
     encoders separate V3V4 from V4V5 instead of aligning organisms (cross-release unit overlap 0.04 and 0.00).
   - The one lever that helped was the **unit** (V4-harmonised exact sequences), and XGB already exploits it.
4. **Level shifts dominate the transfer error.** Cross-release 002→001 TP R² runs from −0.88 to −1.49 for every
   model, while SC transfers (V4-XGB 0.479 vs genus XGB 0.142). The failure is a lab/pipeline/era offset that no
   input can see. This matches Bell et al. 2026's finding that bias dominates the cross-period failures.
5. **Information is thrown away, but it is not obvious this binds.**
   - The top-500 V4 table covers a median 52% of the held-out domain's read mass (event-weighted 50%; 20% for D09).
     Units with ≥ 5% prevalence carry 93%.
   - About 25 co-measured analytes from the same visit are unused (§5).
   - Each target is fitted independently.

   These are real inefficiencies. They are also exactly what a GBM can be given by parity, so they are levers
   for *both* sides.
6. **Label and matrix mismatch.** A single grab (TN/TP/DOC/NO3) or a sonde daily mean is compared with a
   community that integrates days to weeks, so part of the residual is irreducible.

**Literature agrees with this picture.**
- Bell et al. 2026 (Baltic 16S, 397 samples): XGB ≥ RF ≈ TabPFN > all learned embeddings.
- Large microbiome benchmarks find that algorithm choice barely matters (docs/research/03 §a.2, previous-session
  citations: Li 2025, Karwowska 2025, Kubinski 2022).
- In domain generalisation, carefully tuned ERM matches specialised DG methods (Gulrajani & Lopez-Paz 2020).

The binding constraint is information (sites, labels, priors), not model capacity. That is campaign rule 5.

---

## 2. Literature: SOTA summary with citations

| # | Work | What it does / result | Relevance to M3 | Verification |
|---|---|---|---|---|
| L1 | Warren J, de Vries C, Hunt LH, Thorpe AC, Busi SB, Kelly MG, Simons D-L, Taylor JD, Read DS, Walsh K. *Water Research* 300:125959 (2026). doi:10.1016/j.watres.2026.125959 | 1,574 EA biofilm V4 (515F/806R) samples at 694 sites. RF/XGB on 699 SILVA genera (relative abundance) → log10 nitrate-N and TON. Labels are 3-month WIMS means. **Random 70/30 split**: XGB R² 0.654 / 0.650, RF 0.596 / 0.597. | Strongest published method for the EA track. Re-implement its Table 1 recipe and train it on NEON. Its R² **is not a bar**: the random split is leaky. | DOI-checked (Crossref) + session-verified |
| L2 | Thorpe AC, Busi SB, Warren J, Newbold LK, Taylor JD, Walsh K, Read DS. *Global Change Biology* 32(3):e70809 (2026). doi:10.1111/gcb.70809. Data: Zenodo concept 10.5281/zenodo.17515762 (version 10.5281/zenodo.18668846), CC BY 4.0 | More than 1,600 biofilms at 700 sites; descriptive (modules, TITAN). The Zenodo files are `water_chem.csv` (309 kB; 3-month means of about 14 determinands), `ASV_counts_16S_processed.csv` (75.7 MB), `ASV_tax` (2.5 MB) and a FAPROTAX table. The record does **not** document sample-ID columns. | Turns the EA track from 1 target into about 11. It contradicts EA protocol §2.1 ("SC, TP, DO, DOC not available"). | DOI-checked; Zenodo API fetched this session |
| L3 | Bell E, Garefelt K, Jurdzinski KT, Delgado LF, Johansson F, Karlson B, Andersson AF. bioRxiv doi:10.64898/2026.01.25.701552 (2026, preprint) | 16S/18S → physicochemistry with RF, nested-CV XGB, TabPFN and deep embeddings. Mean R² XGB 0.786 vs RF 0.778; TabPFN comparable; raw ASVs beat embeddings. Cross-period transfer failures came from bias, not noise. | Closest analogue. It has the same ranking as SENTINEL, and it argues for rank metrics and offset handling. | Session-verified |
| L4 | Hermans S et al. *Environmental Microbiome* 19:45 (2024). PMC11232138 | NZ stream biofilm 16S at 204 sites, RF, random stratified site split. Log stream N R² 0.02; air temperature 0.69. | Honest prior: freshwater nutrient skill from 16S is low even without blocking. | Session-verified |
| L5 | Connors E et al. *ISME J* 18(1):wrae158 (2024). doi:10.1093/ismejo/wrae158 | Community → bacterial production, RF: R² 0.84 against 0.32 for environment-only. | Microbially mediated **process** targets are where communities beat context. This supports the GAS-3 targets. | Session-verified |
| L6 | Smith MB et al. *mBio* 6 (2015). doi:10.1128/mBio.00326-15 | Groundwater communities as "quantitative geochemical biosensors": 26 of 38 parameters significant, OOB only. | Prior art for many-analyte inference. Its protocol is not blocked. | DOI-checked |
| L7 | ter Braak CJF, van Dam H. *Hydrobiologia* 178:209–223 (1989). doi:10.1007/BF00006028 | Weighted averaging and Gaussian-logit maximum-likelihood calibration (diatoms → pH). | Published family for niche inversion (WA, ML calibration). Mandatory comparator for D1/D2. | DOI-checked |
| L8 | Telford RJ, Birks HJB. *Quat Sci Rev* 28:1309–1316 (2009). doi:10.1016/j.quascirev.2008.12.020. Trachsel M, Telford RJ. *Clim Past* 12:1215–1223 (2016). doi:10.5194/cp-12-1215-2016 | Under spatial autocorrelation, analogue and ML transfer functions are optimistically biased. WA's h-block and LOO RMSEP stay close. | Mechanistic reason to expect inverse calibration to lose *less* than trees under LODO. | DOI-checked |
| L9 | Juggins S. *Quat Sci Rev* 64:20–32 (2013). doi:10.1016/j.quascirev.2012.12.014 | Secondary gradients confound quantitative reconstructions. | Main risk for any niche model: inferring TN through a co-varying ionic or temperature gradient. | DOI-checked |
| L10 | Lin Y, Jeon Y. *JASA* 101:578–590 (2006). doi:10.1198/016214505000001230 | Random forests are adaptive nearest-neighbour methods. | Formalises "GBM/RF = analogue matching", which fails when no analogue site exists (LODO). | DOI-checked |
| L11 | Clark JS et al. *Ecol Monogr* 87:34–56 (2017). doi:10.1002/ecm.1241 (GJAM); Ovaskainen O et al. *Ecol Lett* 20:561–576 (2017). doi:10.1111/ele.12757 (HMSC) | Joint species distribution models; GJAM supports inverse prediction of the environment. | Prior art for the generative side of D1/D2. Both use MCMC and do not scale to about 70k ASVs or sequence priors. | DOI-checked |
| L12 | McCleskey RB et al. *GCA* 77:369–382 (2012). doi:10.1016/j.gca.2011.10.031 | Ionic conductance model for natural waters. | Physics decoder in D2 (SC from ions). | DOI-checked |
| L13 | Hollmann N et al. *Nature* 637:319–326 (2025). doi:10.1038/s41586-024-08328-6 | TabPFN v2. | Already a baseline (genus). Add TabPFN[V4-CLR]. | DOI-checked |
| L14 | Gulrajani I, Lopez-Paz D. "In Search of Lost Domain Generalization", arXiv:2007.01434 (2020) | Tuned ERM matches DG algorithms across 7 datasets. | Reason to **reject** DANN/IRM-style designs with 19 domains. | arXiv-checked |
| L15 | Gorishniy Y, Kotelnikov A, Babenko A. "TabM", arXiv:2410.24210 (ICLR 2025) | Best tabular DL model in their benchmark (parameter-efficient ensembling). | Optional strong-simple-baseline addition, not a candidate. | arXiv-checked |
| L16 | Vandeputte D et al. *Nature* 551:507–511 (2017). doi:10.1038/nature24460 | Quantitative microbiome profiling: relative abundance × total load (flow cytometry or qPCR). | Basis for D4, the QMP input from NEON qPCR. | DOI-checked |
| L17 | Louca S, Parfrey LW, Doebeli M. *Science* 353:1272–1277 (2016). doi:10.1126/science.aaf4507 | FAPROTAX taxon → function mapping. | Optional guild features or priors (nitrifiers, methanotrophs) for the NH4/NO3/CH4 targets. | DOI-checked |
| L18 | Silverman JD et al. *eLife* 6:e21887 (2017). doi:10.7554/eLife.21887 | PhILR phylogenetic isometric log-ratio transform. | Alternative multi-resolution representation for D3 (needs a tree; deferred). | DOI-checked |
| L19 | Jurburg S, Arboleda-Baena C, Kazem A, Frøslev T, Jeppesen T. MiCoDa. iDiv/GBIF 2025. doi:10.15468/ver9ne, CC BY 4.0 | More than 30,000 16S samples, all reprocessed over 515–806 and including EMP. | Larger, region-matched corpus for pretraining or niche priors. Download footprint UNVERIFIED. | DataCite-checked |
| L20 | Yoo & Rosen, arXiv 2508.11075 (2025); SetBERT, *Bioinformatics* 2025 btaf370 | Set transformers over DNA-LM ASV embeddings. | Architecture prior art for SeqSet, which is not claimed as novel. | Cited in novelty file; arXiv ID from the design agent this session |
| L21 | Plummer & Busenberg 1982 (doi:10.1016/0016-7037(82)90056-4); Benson & Krause 1984 (doi:10.4319/lo.1984.29.3.0620); Martiny et al. 2015 (doi:10.1126/science.aac9323); ter Braak & Juggins 1993 (WA-PLS, *Hydrobiologia* 269/270:485–502) | Carbonate constants; O2 solubility; microbial niche conservatism; WA-PLS. | D1/D2 components and comparators. | DOI-checked by the design agent this session, not re-fetched by me |
| — | Li 2025 (*Gut Microbes*), Karwowska 2025 (*Microbiome*), Pope 2025 (*PLOS CB*), Mu 2026 (preprint), Pilgrim 2022 (*Water* 14:2361), Ng & Jordan 2001 (NIPS) | Benchmarks and priors quoted in docs/research/03 or the design. | Context only. | **UNVERIFIED in this session** (previous-session or design citations). Do not quote numbers from them without re-checking. |

**Reading.** No published method beats GBMs on 16S → environment tables under blocked evaluation. The relevant
published families to re-run are:
- RF/XGB on genus or ASV relative abundance (L1, L3, L4);
- WA, WA-PLS and ML calibration (L7, L8), never yet run on M3;
- TabPFN (L13).

---

## 3. Is the current architecture optimal? (honest)

**For the primary NEON 8-target LODO task, at this n: close to the frontier of what the data support. Not
optimal in four specific, testable ways.**

- **Near-optimal.** Ridge, XGB, TabPFN and their ensembles agree within 0.01–0.04 on every target, and tuning
  does not help. When unrelated inductive biases converge on the same score, the remaining error is information-
  and noise-limited, not capacity-limited. The neural set models are below the tables for reasons that more
  epochs or parameters cannot fix (§1.2 item 3). Building a bigger neural architecture on the same 1,215 events
  is the **wrong** lever. Cycles 5–6 are already declared and should stay at their cap; nothing should be added
  to them.
- **Not optimal (each is a falsifiable hypothesis, ranked):**
  1. **Analogue matching versus inversion.** Trees predict from look-alike training sites (L10). Under LODO
     there is none, and the literature on spatially structured calibration (L8) predicts that inverse
     calibration from all taxa degrades less. *This has never been tested in M3.* It is the one genuinely new
     inductive bias available (D1).
  2. **Discarded evidence.** About 48% of held-out read mass goes into one "other" column, and about 25
     same-visit analytes are unused. Multi-resolution units and multi-task stacking recover this (D3), but
     parity gives the same to the GBM, so the expected *relative* gain is small.
  3. **No prior knowledge.** Every model learns niches from 32 sites. Sequence- or trait-informed niche priors
     (D1) and coupling through physics (D2) are the only ways to put information into the model that the 32
     sites do not contain.
  4. **No level-shift handling or uncertainty.** These matter for EA and cross-release. Rank metrics (Spearman)
     and calibrated predictive SDs are the honest response; a model architecture cannot see a lab offset.
- **What will not change the verdict on the primary block:** any architecture whose gain is below about 0.15 R²
  on two targets (§1.2 item 2). The realistic SOTA routes are **tracks with more sites and more labels** (§7).
  Architecture work should therefore be **CPU-first and gated**.

---

## 4. Ranked architecture proposals (merged, critiqued, revised)

**Score** = 0–10, combining expected contribution to a scoped §15.1 label, novelty, and cost to the shared
2-GPU pool and the config caps. All candidates use identical folds, rows, labels and the seed rule (mean of seeds
0–2 where stochastic). Parity baselines get the same inputs and auxiliary labels.

### Critique applied to the incoming NicheGeo design (required changes)

| # | Problem found | Required change |
|---|---|---|
| C1 | **Label counts were wrong.** I re-counted with the builder's own `microbe_wq.VISIT_SUFFIX` (non-null values only, dev events < 2024-01-01, counts only, no values seen). Ca 1,105; ANC 342 (`anc_meql`) / 487 (`lab_anc_meql`), **not 1,207**; TPN 290, **not 1,206**; TSS 769, **not 1,214**. All 7 ions + DIC are jointly present on 1,026 events. The design's reimplemented key over-matched. | ANC leaves the Holm family (descriptive only). PN/TSS latents become partially observed and descriptive. Every count in any protocol must come from the builder join. |
| C2 | Too many coupled components (21 latents, mass-effect gate, Laplace, physics decoder, 30 heads) for a 33-site problem. A failure would not be diagnosable. | **Stage it.** D1 tests the core inductive bias (inverse calibration) on CPU with no GPU. D2 (end-to-end + physics) runs only if D1 passes a pre-declared gate. |
| C3 | 40 GPU-h is a large share of 2 shared GPUs used by about 7 workstreams. | D2 is capped at ≤ 12 GPU-h with a stop after 4 GPU-h. D1 and D3 use 0 GPU-h. |
| C4 | Config cap: the existing track has only reserve slots #6–#8 (stacking, DNA-LM, spare). NicheGeo's ablations are forks too. | The lead reassigns #6–#8 by dated amendment (§8). Ablations are counted and pre-listed; new families get their own cap (≤ 10). |
| C5 | The taxonomy prior uses RDP labels. §6's candidate class is "DNA-encoded" (no taxonomy), and EA windows have no RDP labels without a separate classifier run. | The claim config uses a **sequence-only** (k-mer PCA) prior. Taxonomy shrinkage is an ablation. |
| C6 | Juggins (L9): niches inferred along confounded gradients. | Use a multivariate latent and report partial-gradient diagnostics. Wording stays predictive, never causal. |
| C7 | The mass-effect gate g_k is weakly identified. | Remove it from the claim config (ablation only). |
| C8 | Stage 2 (benthic matrix-invariant niches) needs DP1.20086.002, which is not on disk and has UNVERIFIED pairing. | Defer it. It is not part of the first plan. |
| C9 | The unseen-unit lever is about 0.4% of held-out read mass inside NEON. | State plainly that the sequence prior matters for EA and cross-release, not for primary LODO. |
| C10 | EA has no elevation, so DOsat must assume sea-level pressure. | On EA, score DO% saturation directly (it is in `water_chem.csv`). |

The incoming parity list is kept unchanged and extended (§4 D1 fair test): XGB-chained-physics, XGB on the
inferred z0, XGB with a 3,000-unit cap, WA and WA-PLS, TabPFN[V4-CLR], RF[V4-CLR] and multi-output XGB.

### D1. NicheInvert: multivariate inverse niche calibration with sequence-shrunk niches and a season/location prior. Score 7.0

- **Idea.** Treat the community as a measuring instrument and invert it rather than pattern-matching it.
  - For each V4 unit k (the fold vocabulary: units in ≥ 2 training events; at most 73,360 units overall), fit a
    Gaussian-logit response over a D = 14 standardised master vector:
    - the 8 primary targets;
    - Cl, SO4, Ca, DIC, lab pH and field water temperature (each ≥ 1,105 dev labels).
  - The response is log λ_k(z) = a_k − ½ Σ_d w_kd (z_d − μ_kd)². μ_kd and w_kd start from abundance-weighted WA
    optima and tolerances (L7) and are **shrunk toward a sequence prior**: a ridge map from the unit's k-mer
    PCA-64 to its optimum, with weight n_eff / (n_eff + κ) and κ = 5 fixed a priori.
  - For a new event, maximise Σ_k n_jk log π_k(z)/T + log N(z; m(ctx), Σ(ctx)) with 4–6 damped Newton steps
    (softmax over all units, so **absence evidence counts**).
  - m and Σ come from a ridge/low-rank context prior on DOY, latitude, longitude, elevation and aquatic type.
    This is the tier-0 model *inside* a Bayesian fusion.
  - Output: posterior mean and Laplace SD per dimension.
- **Inductive bias the GBM lacks:**
  - inversion instead of analogue matching (L8, L10);
  - evidence from every unit, not the top 500;
  - niches for rare or unseen units from DNA;
  - joint multivariate inference, which partly controls secondary gradients (L9);
  - calibrated uncertainty.
- **Novelty (provisional; a systematic database search is still owed in docs/novelty_checks/microbe.md):**
  multivariate ML calibration (L7 is univariate) across about 10⁴–10⁵ exact sequence units, with
  empirical-Bayes sequence priors and a context-prior posterior, evaluated leave-one-domain-out against GBM
  parity. JSDMs (L11) do not scale to this and have no sequence prior.
- **Cost:** 0 GPU-h, VRAM 0.
  - CPU, 6 threads. About 30k–70k closed-form niche fits per fold, plus per-event Newton steps on a 14-d
    problem: under 2 CPU-h per track for all 19 folds. The fit is deterministic, so seeds are not needed. If a
    stochastic step is added, use the mean of seeds 0–2.
  - Disk under 0.5 GB.
- **Configs (3, counted):**
  - #6 claim config, NicheInvert with the sequence prior;
  - descriptive ablations: WA-tol with inverse deshrinking (published family), and NicheInvert with a taxonomy
    prior.
- **Fair test:**
  - Dev: `lodo` and `lodo_x001`.
  - Comparators: tier0_best; RF/XGB[genus-CLR], stronger of tuned/untuned; Ridge[genus], TabPFN[genus], Ens[Ridge+XGB]; V4-XGB(+ctx),
    V4-Ridge, **RF[V4-CLR] and TabPFN[V4-CLR] (never run; required)**, XGB[V4-CLR, 3,000 units], Ens[V4-Ridge+V4-XGB];
    WA-PLS (1–5 components chosen by inner LODO) on the same units; **XGB on NicheInvert's z-posterior + context**
    (feature parity for the niche summary).
  - Unseen blocks: the 2024 holdout (H-M3 family + GEO family, one sealed look); cross-release pre-2017 reserve;
    Test-B; EA (NEON-trained, Spearman primary).
  - Holm within each family. Both site- and domain-cluster bootstrap units must pass.
- **Pre-declared gate to D2:** dev-LODO mean R² over the 8 targets ≥ V4-XGB − 0.02, **and** at least 2 targets
  where the point ΔR² vs the strongest simple baseline is ≥ +0.03. Otherwise D2 is not run and the stop is
  reported.
- **Honest expectation:**
  - P(D1 mean ≥ V4-XGB − 0.02) ≈ 0.45.
  - P(beats the strongest simple baseline by ≥ 0.05 on ≥ 2 targets on dev) ≈ 0.15.
  - It is most likely to help DOC, TP and NO3 (gradient extrapolation) and EA rank skill, and least likely for
    SC, where Ridge is already 0.78.

### D2. NicheGeo-Phys: end-to-end NicheInvert with a differentiable aqueous-geochemistry decoder (revised NicheGeo). Score 4.5 (gated on D1)

- **Idea.** D1 trained end-to-end through an unrolled inverse layer, followed by a fixed physics decoder. All
  required changes C1–C10 are applied.
  - Latent: 16-d. The 14 dimensions of D1 plus NH4 and SRP; no PN, DOP or PP latents.
  - Physics, exact identities only:
    - TN ≥ DIN and TP ≥ SRP via pool sums;
    - SC from ions via McCleskey (L12);
    - DO = DOsat(T, P(elevation))·exp(s) (Benson & Krause);
    - pH from DIC and alkalinity via the carbonate system in fp64 (Plummer & Busenberg).
  - At most 3 learned scalars with tight priors.
  - Heads: the 8 primary targets + GEO-8 + the field-meter SC and DO as separate source heads.
- **Extra bias over D1:** labels of one analyte become evidence about the others; extrapolation stays physically
  correct beyond the training range.
- **Cost:** ≤ 12 GPU-h (about 2–3 min per fold-seed; 19 folds × 3 seeds ≈ 3 GPU-h per config-track), peak VRAM
  ≤ 4 GB, so it co-tenants.
  - Jobs: ≤ 1 h 55 min each through the `sentinel.core.gpu` lock on GPU 0 or 1. Checkpoint every fold-seed and
    every 25 epochs; resume skips finished fold-seeds.
  - Stop after 4 GPU-h if dev mean R² < V4-XGB − 0.02 and no GEO target beats its best parity comparator by its
    point estimate.
- **Configs:** 1 claim config (#8, or on the GEO track) plus 2 counted ablations (no physics; no context prior).
- **Fair test:** as D1, plus **XGB-chained-physics**. XGB predicts the latents, and the *identical* decoder
  derives SC, pH, DO, TN and TP. This is mandatory: a GBM can use the same physics after the fact.
- **Honest expectation:** P(passes the D1 gate) ≈ 0.35. Given that it passes, P(adds ≥ 0.03 over D1 on ≥ 2
  targets) ≈ 0.3. The physics mostly helps derived targets (SC, DO, pH) that are already high-R² or
  season-driven.

### D3. V4-MR: multi-resolution DNA units + cross-fitted multi-task stacking (CPU). Score 6.0

- **Two one-change configs:**
  - **D3a.** V4-XGB with the unit cap raised to 3,000, plus aggregated V4-unit CLR at genus, family and order
    level (RDP labels carried by the units). The "other" bucket shrinks from about 48% to under 10% of
    held-out read mass.
  - **D3b.** D3a plus **cross-fitted GEO auxiliaries**: inner-LODO out-of-fold predictions of the 14 auxiliary
    analytes are fed as features to each target model (community → chemistry manifold → target). This is the
    declared "cross-fitted stacking" reserve item, now multi-task.
- **Bias:** it recovers discarded evidence and shares strength across co-measured analytes. It is not a new
  model class.
- **Parity:** the baseline twins are XGB[genus+family+order] with the same cap, and stacked XGB[genus-CLR] with
  the same auxiliaries. If D3a wins only because of the cap, that is reported as a representation result.
- **Cost:** 0 GPU-h; CPU about 2–4 h per config-track at 6 threads (inner LODO for stacking is the expensive
  part).
- **Honest expectation:** P(D3a ≥ V4-XGB + 0.02 mean) ≈ 0.35. P(D3b beats its parity twin with CI ≠ 0 on ≥ 1
  target) ≈ 0.10. Its main value is as a stronger, cheaper candidate on GEO and EA, where V4 units transfer
  better than genus labels (SC cross-release 0.479 vs 0.142).

### D4. QMP input: absolute abundance from NEON qPCR. Score 4.5 (an input lever, applied to all models)

- **Idea.** Multiply relative abundances by total 16S copies per mL (Vandeputte et al. 2017, L16). NEON
  DP1.20278.001 `mga_swGroupAbundances` (16S "bacteria and archaea" copies) is **already on disk**
  (`processed/neon_microbe_qpcr`). I counted label-free that **1,158 of 1,215 dev events** have a same-site,
  same-date qPCR record. The proper join is by dnaSampleID, plus the filtered volume from `amc_fieldGenetic`.
- **Inputs:** log total copies per mL, and log absolute abundance of the top units, alongside the CLR.
- **Why:** biomass and load carry trophic-state information (TP, chl-a, turbidity) that compositional data
  cannot express.
- **Parity:** every baseline gets the same feature.
- **Risks:** extraction and qPCR batch effects; 16S copy-number variation. Report a qPCR-batch confound check.
- **Cost:** 0 GPU-h; about 1 CPU-day of data-building.
- **Pre-registration:** a §4 amendment ("microbial qPCR copies are a community-side input, never chemistry").
  Put it on the GEO track to avoid using the existing cap.
- **Expectation:** small gain, P(≥ +0.03 on TP or chl-a for all models) ≈ 0.25. It cannot by itself separate
  SENTINEL from baselines.

### D5. Cycle 6: EMP-pretrained SeqSet (already declared, #3–#5). Score 2.0

- Keep it as declared, at ≤ 6 GPU-h of the 8 GPU-h budget for cycles 5–6.
- The EMP freshwater corpus is about 4,700 samples, and 89% of it comes from three lake studies (docs/research/03
  §b.2). Pretraining gains in the literature are "modest at best" (previous-session citations), and SeqSet sits
  0.07 below the tables.
- MiCoDa (L19: more than 30k samples, 515–806, CC BY 4.0) is the only corpus large enough to change this. Fetch
  it **only if** #4 beats #3, after checking its footprint against the 5 GB cap.
- P(the pretrained SeqSet reaches V4-XGB on dev) ≈ 0.05.

### Rejected (not to be built)

| Design | Reason |
|---|---|
| Larger set transformers or fine-tuned DNA-LMs (DNABERT-2, end-to-end) | SeqSet is 0.07–0.14 below the tables. Bell 2026 found raw ASVs beat embeddings. It is data-starved (§1.2 item 3). Drop the reserve slot "V4-window DNA-LM configuration". |
| Domain-adversarial, IRM or other DG losses | 19 domains with 1–3 sites each, and ERM ≈ DG (L14). Unidentifiable. |
| GNNs over co-occurrence networks | No evidence of gains on tables; networks estimated from 32 sites are noise. |
| Training NEON models on EA chemistry | Conflicts with the M5-X shared-target precedent and with M3E's evaluation-only design. It is allowed only through Track H4, after a lead ruling. |
| "Few-shot site calibration" track | The k-shot site mean would dominate for slowly varying targets, so the track would be a context-baseline benchmark, not a microbe one. |

---

## 5. New numeric predictive targets (multi-task)

**Counts:** NEON development events (< 2024-01-01; 1,215 events; 33 sites; 19 domains), same field visit
(`parentSampleID` through the builder's `VISIT_SUFFIX`), non-null values only. Censored values that the builder
substitutes add more rows (e.g. NO3: 1,106 non-null vs the builder's 1,180). These are counts only; no value was
inspected. Every listed analyte covers all 33 sites and 19 domains.

| Analyte (chem_event_wide) | Dev events | Practical role | Proposed status |
|---|---|---|---|
| lab pH | 1,165 | acidification | **GEO-8** |
| Cl | 1,171 | road-salt salinisation | **GEO-8** |
| SO4 | 1,176 | mining, acid deposition | **GEO-8** |
| Ca | 1,105 | hardness, buffering | **GEO-8** |
| DIC | 1,171 | carbonate system, CO2 | **GEO-8** |
| field DO % saturation | 1,174 | hypoxia, metabolism | **GEO-8** |
| NH4-N | ≥ 1,141 (+ censored) | toxicity, nitrification (microbially mediated) | **GEO-8**, if the counts-only censored fraction ≤ 30% (rule fixed before results); otherwise descriptive |
| UV254 | 1,175 | DOM aromaticity (SUVA with DOC) | **GEO-8** |
| Mg 1,110; Na 1,115; K 1,174; Si 1,137; F 1,117; TDS 1,080; TDN 1,154; TDP 1,181 | — | ionic/nutrient detail | descriptive heads (D2), reported |
| field SC 1,191; field DO 1,172; field temp 1,194 | — | same-visit meter readings | **alternate label source** for SC/DO, with about 60% more rows than the sonde (730 / 683). Descriptive in H-M3; GEO-track heads. |
| SRP ≥ 1,064 (+ censored); TSS 769; Br 907; lab ANC 487; ANC 342; Fe 471; Mn 479; TPC/TPN about 290 | — | — | descriptive only (low n or heavy censoring) |
| **GAS-3:** dissolved CO2, CH4, N2O (NEON DP1.20097.001) | UNVERIFIED until joined. The user guide states the samples are taken with surface-water chemistry and microbes and link via parentSampleID (session-verified). | greenhouse gases; microbially mediated (L5 rationale) | **separate family**, pre-registered before download |

- **Why these targets.** These are the quantities agencies need for salinisation, acidification, hypoxia and
  GHG flux, so the choice is made on practical grounds before any result (rule 2).
- **Honest note:** ionic targets (Cl, SO4, Ca, DIC) are between-site geology. Community models will beat tier-0
  easily, as with SC (tier-0 −0.10 → 0.76), but **Ridge/XGB on CLR will be just as good**. They add power and
  practical value, not an easy SOTA.
- **The most promising targets for a model-vs-GBM gap** are the microbially mediated ones: NH4, DO% saturation,
  CH4 and N2O.
- **2024 holdout counts.** Expect about 190–210 per GEO target (214 events × about 90–97% coverage). This is an
  estimate; I did not count holdout rows, and a counts-only check must be run before freeze.

---

## 6. More data: sources with licence and size

| Source | Licence | Size | On disk? | Use | Status |
|---|---|---|---|---|---|
| NEON `chem_event_wide` co-measured analytes (DP1.20093.001) | CC BY 4.0 | 0 new | **yes** | GEO-8 targets, D2 heads, D3b auxiliaries | Counted this session (§5) |
| NEON qPCR group abundances (DP1.20278.001) | CC BY 4.0 | 0 new | **yes** (`neon_microbe_qpcr`, 2,653 16S rows) | D4 QMP input; 1,158 of 1,215 dev events matched by site-date | Counted this session |
| Thorpe et al. 2026 Zenodo (`water_chem.csv` 309 kB; ASV counts 75.7 MB; tax 2.5 MB; FAPROTAX 4.8 MB) | **CC BY 4.0** (better than the Warren GitHub file, which has no licence) | 84.6 MB | no | EA multi-determinand labels (about 11 overlap NEON). The DADA2 ASV table may replace our own dereplication **only if** it carries sequences (UNVERIFIED) | Zenodo API fetched this session. Sample-ID columns are undocumented: run a header-only ID-linkage check (range request, discard everything after the first newline) **before** download. The EA flagged a taxa-vs-nitrate row-order bug (corrected 2025-10-24), so join by ID only. |
| ENA PRJEB90117 (EA reads, 1,649 runs) | INSDC open | streamed, under 0.3 GB processed | no | EA V4 windows (approved M3E §2.3) | Session-verified |
| NEON DP1.20097.001 dissolved gases | CC BY 4.0 | tens to hundreds of MB (UNVERIFIED) | no | GAS-3 targets. Requires a reimplementation of NEON's neonDissGas headspace calculation, unit-tested | Session-verified product and pairing text |
| NEON DP1.20086.002 benthic 16S | CC BY 4.0 | a few hundred MB (UNVERIFIED) | no | deferred cross-matrix diagnostic (H5) | Session-verified |
| MiCoDa (515–806, more than 30k samples) | CC BY 4.0 | UNVERIFIED (GBIF occurrence form) | no | only if cycle 6 #4 > #3 | DataCite-checked |

- **Disk:** `df -h /home/bcheng` shows 86 GB free (above the 60 GB floor). The planned new data is about 0.5 GB,
  well inside the M3 5 GB cap. Run `df` again before each download (disk rule).
- **Not recommended:** EMP metadata as weak niche labels. It is study-confounded: three lake studies dominate,
  pH is on about half the samples, and phosphate on about 7% (docs/research/03 §b.2).

---

## 7. Harder tracks: practical justification, unseen test block, power

Every track below must be pre-registered with this justification before any result on it (rule 2). Spent blocks
are never reused.

| Track | Practical justification | Unseen test block | Power (pre-scoring, counts-only) | Priority |
|---|---|---|---|---|
| **H1. NEON GEO-8** (+ the existing H-M3) | Agencies need ionic, carbonate and oxygen state in unmonitored domains | **2024 holdout, scored in the SAME sealed look as H-M3** (separate Holm family). Secondary: Test-B 2025 and pooled 2024+2025 | Dev MDD comparable to H-M3 (≈ 0.10–0.20). The holdout is worse. The pooled analysis (≈ 2× events) is the realistic power source | **1. Must be pre-registered before the H-M3 freeze**, or GEO can only use Test-B |
| **H2. NEON GAS-3** | GHG flux and microbial process targets | 2024 holdout, if downloaded and pre-registered before the look; otherwise Test-B only | UNVERIFIED n (join first, counts only) | 3 |
| **H3. EA multi-determinand, NEON → England** (amend M3E) | Transfer to another national network (the approved rationale) | All EA samples, or the EA-test partition if H4 is approved. Single look `M3E/test` | n ≈ 1,558. SE(ρ) ≈ 0.024 under independence; with 22 group clusters and a design effect of 3–5, SE ≈ 0.04–0.055. Paired Δρ MDE plausibly 0.04–0.08 (estimate; compute it in `microbe_ea_ids.json`) | **2** |
| **H4. EA-internal, area-blocked** (needs a lead ruling on the M5-X shared target) | A national network extrapolating to unsampled regions. The published Warren recipe is re-run on the *same* data under blocking, so this is the cleanest head-to-head with the strongest published method | **Partition before any EA label is downloaded:** a seeded draw of about 7 of the 22 letter groups as EA-test (also the M3E test rows); the other groups are EA-dev for blocked CV. Optional: EA 2023 samples (395) as a time-out stratum | About 700 sites. Site-cluster units give far more power than NEON (about 33). The group-cluster unit (≈ 7 test groups) is the weak point, so report both | 2 (conditional) |
| H5. NEON water → NEON benthic biofilm (cross-matrix) | Separates matrix shift from geography for the EA result | Benthic 2024 events | 24 sites; pairing to chemistry UNVERIFIED | 5 (deferred) |
| Existing: cross-release pre-2017 reserve; Test-B; event track | — | as pre-registered | — | unchanged |

**Rules carried into every new track:**
- campaign hardening of the guard;
- `*_test_results.md` write-ups;
- every result reported, wins and losses;
- claims scoped to their track ("SOTA on NEON→England rank skill for nitrate").

---

## 8. Sequencing within the shared 2-GPU budget and config caps

### 8.1 Must be pre-registered first (lead-written; nothing is run on these before the amendment)

1. **`docs/preregistration/microbe_geo_protocol.md` + microbe_protocol §10.9**:
   - the GEO-8 list, transforms, floors and censoring (NH4 rule);
   - the builder visit-key join;
   - counts and power (counts-only);
   - the frozen split (unchanged, sha `aaeb0530…`);
   - families and Holm;
   - **the same sealed look as H-M3**;
   - a cap of ≤ 10 configs;
   - the full parity list (§4 D1).
2. **Reassign the existing-track reserve #6–#8:**
   - #6 = NicheInvert (D1);
   - #7 = V4-MR (D3a);
   - #8 = NicheGeo-Phys (D2), only if the D1 gate passes; otherwise the declared stacking (D3b).

   Drop the DNA-LM slot.
3. **M3E amendment, before any EA label download:**
   - multi-determinand labels from the Thorpe Zenodo `water_chem.csv` (CC BY 4.0), with sha256 and
     scorer-only handling;
   - header-only ID-linkage check;
   - the Warren Table 1 recipe as the published comparator, trained on NEON;
   - Spearman primary;
   - if H4 is approved, the EA-dev/EA-test partition (seeded, by letter group) written to
     `microbe_ea_ids.json` **before** labels.
4. **Lead rulings:** (a) H4 versus M5-X X-U (EA WIMS chemistry as M3 training labels); (b) tell M6 about EA use
   (no direct overlap, since M6 reserves Hydrology-API sonde data from 2023-10).
5. **GAS-3 amendment before the DP1.20097.001 download**, or declare it Test-B-only.
6. **§4 input amendment for D4 (qPCR)**, on the GEO track.

### 8.2 Execution order (≈ 3 weeks; GPU total ≤ 21 GPU-h, likely ≤ 12)

| Week | CPU (≤ 6 threads, one heavy job) | GPU 0/1 (≤ 2 h jobs, lock helper) |
|---|---|---|
| 1 (Nov 10–16) | Finish W6 (RF-tuned); run the logged comparisons; build GEO targets and the qPCR join; **D1** and **D3a** on `lodo` + `lodo_x001` (dev); parity baselines RF[V4], TabPFN[V4], XGB[V4-3000], WA-PLS | Cycle 5 #1–2 (≈ 40 min) |
| 2 (Nov 17–23) | D3b; GEO-track baselines and candidates; EA read processing (stream, `df` every 50 runs); `microbe_ea_ids.json` frozen | Cycle 6 #3/#4 (≤ 6 GPU-h); **D2 only if the D1 gate passes** (≤ 4 GPU-h to the stop rule, ≤ 12 total) |
| 3 (Nov 24–30) | §6 selection; freeze candidate; event dev run | Final per-domain fits for the frozen model (≤ 2 GPU-h if neural; 0 if D1/D3) |
| Look | Lead writes `FREEZE_microbe.json`. **One sealed look: H-M3 + GEO-8 (+ GAS-3 if ready)** on the 2024 holdout. Then the `M3E/test` look | — |
| Jan 2027 | Test-B (RELEASE-2027) and the pooled analysis | — |

**Budget:**

| Item | GPU-h |
|---|---|
| Cycles 5–6 (already approved) | ≤ 8 |
| D2 (gated) | ≤ 12 |
| Final fits | ≤ 1 |

Everything else is CPU. If the D1 gate fails, the total is ≤ 9 GPU-h.

---

## 9. Honest outlook

| Track / label | P(earns the §15.1 label, scoped) | Why |
|---|---|---|
| Primary H-M3 (≥ 2 targets vs tier-0 + RF + XGB) on the 2024 holdout | ≈ 0.15 | 1 target on dev (SC). The holdout MDD is likely ≥ 0.16. |
| §15.1 SOTA on the primary NEON 8 (vs the strongest simple baseline, incl. Ridge/TabPFN/Ens) | ≈ 0.05 | 0 targets on dev; architecture gains so far are ≤ 0.04. |
| GEO-8 SOTA (NEON 2024 + pooled) | ≈ 0.10 | Ionic targets are easy vs tier-0 but tie with the GBM. NH4 and DO% saturation are the best chances. |
| GAS-3 SOTA | ≈ 0.10–0.15 (very uncertain) | Process targets are where community data beat context (L5). n UNVERIFIED. |
| EA NEON→England (Spearman, vs the Warren recipe trained on NEON + tier-0), ≥ 1 determinand | ≈ 0.25–0.30 | V4 units transfer where genus labels do not (SC cross-release 0.479 vs 0.142). Level shift is neutralised by rank metrics. |
| EA-internal area-blocked (if approved) | ≈ 0.15–0.20 | Power is good, but with about 1,000 training samples the GBM is strong too. |
| **At least one scoped SOTA label for M3 by Feb 2027** | **≈ 0.35** | Mostly through H3; the multiple tracks are separate, pre-registered families. |

**Bottom line:**
- The current GBM is not beaten because, at 32 effective sites, nothing can be beaten detectably. More capacity
  will not change that.
- The one untested inductive bias worth building is **inverse niche calibration (D1)**. It is CPU-only, cheap and
  falsifiable. It upgrades to the physics-coupled NicheGeo (D2) only if it earns the gate.
- The honest routes to a label are **more sites and more labels**: the EA multi-determinand and EA-internal
  tracks, GEO/GAS targets in the same sealed look, and Test-B pooling.
- All comparators get the same data and auxiliary labels, and every outcome, wins and losses, is reported.
