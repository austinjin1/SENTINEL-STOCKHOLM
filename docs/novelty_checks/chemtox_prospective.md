# Novelty check — ChemTox: prospective cross-chemical evaluation of acute aquatic toxicity models

**Search date: 2026-10-24.** Required by RESEARCH_PLAN §15.1 (a "first-in-class" label needs a dated
literature search) and §15.5 guard 3. Searched with web search and publisher/arXiv/Crossref lookups;
every citation below carries its verification status, because a claim resting on an unverified
reference is worthless. This document records what was looked for, what was found, and what the
search does **not** establish.

## 1. Questions asked

1. Has anyone reported a number on the **ADORE** benchmark (Schür et al. 2023) since the published
   baseline (Gasser et al. 2024, micro-RMSE 0.927 on `t-F2F`/`split_occurrence`)?
2. Has anyone evaluated an aquatic-toxicity model **prospectively** — on records or chemicals added
   to a database *after* the model's training data, or on chemicals absent from the training
   benchmark, with the test set pre-registered and scored once?
3. What is the current best method for this task (to make sure the comparison includes it)?

## 2. Findings

**Q1 — no published number beats 0.927, and no one has reported an ADORE `t-F2F` RMSE at all.**
Of the works citing Schür et al. 2023 and Gasser et al. 2024 (OpenAlex, checked 2026-10-24), the only
re-use of the `t-F2F` split found is `G.AI.A` (*J. Chem. Inf. Model.* 2026, DOI 10.1021/acs.jcim.5c02286),
which uses it as an external **classification** set (ExtraTrees accuracy 90.44%), not for regression.
Schür, Schirmer & Baity-Jesi, "On the comparability between studies in predictive ecotoxicology"
(*Computational Toxicology* 35:100367, 2025, DOI 10.1016/j.comtox.2025.100367 — **verified**) report
that in recent fish-acute-toxicity literature no two studies fully meet their five comparability
criteria, which is consistent with the benchmark being effectively uncontested.

**Q2 — no prospective evaluation of an aquatic-toxicity model was found.** Searches for temporal /
time-split / prospective validation in ecotoxicology returned methodological work from
cheminformatics, not ecotoxicology: Sheridan's time-split cross-validation argument and Kearnes,
"Pursuing a Prospective Perspective" (arXiv:2009.00707 — **verified to exist**), which cautions that
a temporal split is *not* equivalent to prospective validation. Within ecotoxicology the relevant
2024–2026 works evaluate on held-out chemicals of the same curated dataset (Gasser et al. 2024;
Zubrod et al., *Environment International* 186:108607, 2024; Yuan et al., *Environ. Toxicol. Chem.*
45(9):2466–2482, 2026; Posthuma, Price & Viljanen, *ES&T* 59(31):16250–16260, 2025 — **metadata
verified**, full texts not read for the last three), not on records a later database release added or
on chemicals a benchmark excluded. Posthuma et al. state explicitly that their pairwise-learning model
interpolates and that "extrapolation beyond the set of tested species and chemicals remains
challenging", i.e. the prospective question is named as future work rather than answered.

**What this supports, and what it does not.** It supports a narrow claim: *as of 2026-10-24 we found
no prior study that pre-registers a previously unseen set of chemicals absent from the training
benchmark, freezes it with a hash, and scores aquatic-toxicity models on it once.* It does **not**
support "first model to predict acute aquatic toxicity", nor does the absence of a published ADORE
RMSE make our ADORE result state of the art — that pass came out "competitive, not SOTA"
(`docs/preregistration/chemtox_test_plan.md`) and stays reported as such. Per §15.5 guard 3, a
protocol-novelty claim does not substitute for beating the baselines: the prospective pass must beat
the reproduced published method, the read-across comparator and the trivial baselines with
chemical-cluster CIs excluding 0 after Holm, in all three seeds, or it is reported as competitive or
negative.

**Q3 — the methods the comparison must include** (from the same search; used to choose what to
implement in improvement cycle 2, see `sentinel/models/chemtox/cycle.py`):

| Method | Status here | Reference (verification) |
|---|---|---|
| XGBoost / RF on MACCS + the published 37-column block — the published method for ADORE | reproduced and scored under the identical protocol (and re-tuned with the same 32-configuration budget in cycle 2) | Gasser et al., *Environ. Sci.: Adv.* 3:1124–1138 (2024), DOI 10.1039/D4VA00072B (**verified**, numbers from the bioRxiv full text DOI 10.1101/2024.03.14.584983) |
| Tabular foundation model on **frozen** CheMeleon fingerprints (beats fine-tuned CheMeleon and Chemprop over 58 tasks) | implemented (`chemeleon_fp.py`, families `chemeleon_*`) | Ben Hicham, Rittig, Grohe & Mitsos, arXiv:2604.16123 (**verified**); CheMeleon: Burns et al., arXiv:2506.15792 (**verified**) |
| GPBoost — boosted trees + grouped random effects for cross-species toxicity | implemented (`gpboost_model.py`) | Zubrod et al., *Environ. Int.* 186:108607 (2024) (**metadata verified**); Yuan et al., *ETC* 45(9) (2026) (**metadata verified**) |
| k-NN read-across, distance-gated — the regulatory comparator | implemented (`readacross.py`), pre-registered as a required comparator | Cassotti/Todeschini-style kNN with an applicability-domain threshold (**method verified in secondary sources only**) |
| TRIDENT (transformer, multi-endpoint) — the strongest published deep model for aquatic toxicity | **not comparable here and deliberately not run**: it was trained on ECOTOX-derived data covering the period our prospective records come from, so its training set very likely contains them; running it would measure memorisation. Recorded as a limitation in the model card | Gustavsson et al., *Science Advances* 10(10):eadk6669 (2024), DOI 10.1126/sciadv.adk6669 (**verified, full text**) |
| Multi-task across species conditioned on per-species data volume | informed the interpretation of our failed flat multitask run (gain concentrates at ≤ ~20 assays per species, loss at ≥128) | Schlender et al., *ES&T* 57(46):17818–17830 (2023), DOI 10.1021/acs.est.3c00334 (**verified**) |
| Species embeddings from taxonomy + genetic sequence | **not implemented** (needs per-species cytochrome-*b* sequences; recorded as the most promising untried avenue) | Yang et al., *ES&T* 60(3):2556–2565 (2026), DOI 10.1021/acs.est.5c12494 (**metadata verified**); Forastiere et al., *Ecotox. Environ. Saf.* 314:120057 (2026) (**metadata verified**) |

## 3. Noise floor and what "good" can mean

Two independent estimates bound what any model can reach on this kind of data: the replicate floor
measured inside our own prospective primary track is **0.452** log units (pooled within
(chemical, species, duration) SD, 361 groups / 1,180 rows), and Posthuma et al. (2025) estimate an
"ideal" RMSE of 0.51 on ECOTOX-wide pairwise prediction with ~92% of variance explainable
(**metadata verified**). Round-robin studies put intra-/inter-laboratory CVs at 15–31% (≈0.11 log
units) for standard fish assays, so the ~0.45–0.5 floor is dominated by cross-study and
cross-protocol heterogeneity rather than measurement error — which is what an occurrence-style split
exposes.

## 4. Reproducing this check

The searches were run by an assistant with web access on 2026-10-24; the queries covered ADORE +
RMSE, aquatic toxicity + prospective/temporal validation, CheMeleon/TabPFN/GPBoost/TRIDENT, and
cross-species toxicity prediction 2024–2026. Citations marked **verified** were confirmed from the
publisher page, the arXiv listing or the full text; those marked **metadata verified** were confirmed
through Crossref/OpenAlex records only; nothing in this document rests on a reference that was not at
least metadata-verified. A fuller annotated list, including the items that could **not** be verified
(e.g. the `tox-learn` library announced by Yuan et al. 2026, TabPFN-3.5 release details), is in the
workstream notes and must be re-verified by `bibliography-verifier` before any of it enters a paper.
