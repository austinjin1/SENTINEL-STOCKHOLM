# Novelty check — M3 SENTINEL-Microbe (16S community → co-located water quality)

**Status:** provisional, web-search pass only (2026-10-24). A Scopus / Google Scholar / PubMed pass is
still required before any "first-in-class" wording leaves the repo (RESEARCH_PLAN §15.1, §15.5
amendment item 3). A novelty finding does **not** earn the label by itself: the model must also beat
every applicable baseline with CI excluding 0 (H-M3, `docs/preregistration/microbe_protocol.md` §1).

## Claim under test (RESEARCH_PLAN §5.3)
"The first model predicting co-located water chemistry and sonde water quality from freshwater 16S
community profiles that is evaluated by leave-one-ecoclimatic-domain-out on a sensor-co-located
continental network (NEON), and that transfers across 16S primer sets (V3V4 ↔ V4V5) and
bioinformatics releases by representing each sequence by a frozen DNA language-model embedding."

## Explicitly NOT claimed (prior art exists)
- Predicting water/sediment chemistry from 16S with ML in general: Smith et al. 2015 (groundwater
  wells, RF, 26 geochemical features); Hermans et al. 2024 (NZ stream biofilms, RF; log N R² = 0.02,
  air temperature R² = 0.69); headwater-stream condition from 16S (PLOS ONE 2020); river WQ
  indicators + micropollutants from 16S with explainable ML (Process Saf. Environ. Prot. 2025);
  marine microbiome → chemistry with RF (bioRxiv 2026.01.25.701552). None found uses NEON or a
  held-out-domain design (see searches 1, 4).
- Set transformers over DNA-LM-embedded amplicon sequences: **Yoo & Rosen 2025**, "Abundance-Aware
  Set Transformer for Microbiome Sample Embedding" (arXiv 2508.11075) — DNABERT-2 [CLS] embeddings of
  sequences, abundance-replicated tokens, set transformer; classification only (bladder tumour,
  Leptospira co-occurrence, soil vs non-soil cross-study), random 80/20 splits. The SeqSet
  architecture is therefore **not** claimed as novel; only the task/evaluation is.
- Microbiome foundation models on taxon tokens: MGM (Adv. Sci. 2026; genus tokens, 260k samples),
  BiomeGPT (bioRxiv 2026, gut-focused); reported gains over tabular baselines are modest at best.
- Read-level attention models on 16S (Read2Pheno, PLOS Comput Biol 2021) — host phenotypes.

## Searches run (2026-10-24, general web search engine; US results)
| # | Query | Relevant hits | Conflicts with claim? |
|---|---|---|---|
| 1 | NEON aquatic microbial 16S predict water chemistry machine learning | marine microbiome→chemistry RF (bioRxiv 2026); river WQ from 16S (2025); Smith 2015 wells; cold-seep hydrocarbons | No NEON-based or domain-held-out study |
| 2 | 16S amplicon community predict environmental variables nitrogen phosphorus deep learning DNA language model 2025 | Read2Pheno; ANNA16 (copy number); unrelated soil N/P spectroscopy | No DNA-LM → nutrient regression |
| 3 | microbiome foundation model environmental prediction transfer across primers ASV sequence embedding set transformer 2025 2026 | Yoo & Rosen 2025 (DNABERT-2 + set transformer, classification); MGM; BiomeGPT; 2026 review of microbiome FMs | Architecture prior art (not claimed); no primer transfer, no WQ regression |
| 4 | "leave-one-domain-out" OR "spatially blocked" 16S microbial community predict stream water quality nutrient random forest NEON | Hermans 2024 (NZ); headwater streams 2020 | No blocked/held-out-domain WQ evaluation found |
| 5 | NEON surface water microbial community composition water quality predict study 2024 2025 | NEON product pages; 16S vs WGS resolution on NEON (2020); non-NEON seasonal/urban studies | None |
| 6 | cross-primer 16S rRNA region transfer machine learning V3V4 V4V5 classifier generalization sequence embedding | QIIME2 forum (region-specific taxonomic classifiers); mock-community region comparisons | Taxonomic-classifier region transfer only; no sample-level model transfer |

## Provisional conclusion
No conflicting prior work found by web search for (a) NEON 16S → co-located WQ prediction under
leave-one-domain-out, or (b) sample-level model transfer across 16S primer sets / releases for an
environmental regression task. The architecture (DNA-LM-embedded set transformer) has prior art and
is not claimed. Web search is not a systematic database search; recall is limited. The label further
depends on the H-M3 result (protocol §1) — `[PENDING]`.

Sources: arxiv.org/abs/2508.11075; biorxiv.org/content/10.64898/2026.01.25.701552v1;
pmc.ncbi.nlm.nih.gov/articles/PMC11232138 (Hermans 2024); pmc.ncbi.nlm.nih.gov/articles/PMC7398514;
advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202513333 (MGM); biorxiv.org/content/10.64898/2026.01.05.697599v1
(BiomeGPT); journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009345 (Read2Pheno).

## Update 2026-11-10 (second web-search pass; resumed M3 session)
Restored the 2026-10-24 record above verbatim (an accidental overwrite was reverted from git).

**New or re-checked items:**
- **England river-biofilm 16S** (NERC NORA 541561): 1,574 samples, 694 sites, tree-based ML, taxa importance. This
  is a national dataset. No held-out-region water-chemistry prediction was identified; data access is to be
  checked as a possible external track.
- **Pilgrim et al. 2022** (periphyton 16S, 25 streams; PMC9534034): TP/TN response thresholds (gradient forest).
  This is indicator development, not prediction under domain shift.
- **SetBERT** (Bioinformatics 2025, btaf370): a set model over DNA sequences. Architecture prior art only, so it
  is not claimed.
- Searches: "NEON aquatic microbial 16S predict water chemistry nutrients machine learning cross-site 2024
  2025"; "eDNA 16S bacterial community predict stream nitrogen phosphorus conductivity random forest spatial
  validation unseen catchments 2023 2024 2025". Neither found a held-out-region freshwater WQ evaluation.

**Claim wording must change.** The protocol's cycle-2 diagnosis (microbe_protocol.md §10.3) contradicts the claim
text above ("transfers … by representing each sequence by a frozen DNA language-model embedding"). Whole-amplicon
DNABERT-S and k-mer embeddings do not align V3V4 with V4V5: the cross-release read-mass overlap of their units is
0.04 and 0.00. Transfer instead uses **in-silico V4 harmonisation**, which raises the overlap to 0.337. Any
first-in-class wording must name that representation. It must also state the results: cross-release transfer
fails for TN, TP and DOC (development CV) and works for SC.

**Provisional conclusion (unchanged in kind):** no conflicting freshwater prior work was found for 16S → water
chemistry under leave-one-region-out evaluation. The label still requires the H-M3 bar on the frozen test (`[PENDING]`).
Additional sources: https://link.springer.com/article/10.1186/s40793-024-00588-z ·
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9534034/ · https://nora.nerc.ac.uk/id/eprint/541561/1/N541561JA.pdf ·
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12245400/
