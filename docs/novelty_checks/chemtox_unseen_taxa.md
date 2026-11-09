# Novelty check — ChemTox unseen-taxa track (dated 2026-11-09)

Required by RESEARCH_PLAN §15.1 / §15.5(3) before any "first-in-class" or "SOTA" wording on this
track (review item C-M4U-3). **Verification status is marked per item and is deliberately
conservative: nothing here is claimed as verified that was not checked against a source.** No
network search was run in this session, so items below carry either (a) *repo-verified* — the
citation was verified earlier in this repository's own novelty checks, or (b) **to verify** — the
work is known to the author but its metadata must be confirmed before any claim text cites it.
A claim may not be published from this track until every item it relies on is verified.

## The claim this track could support

"A chemistry-only model predicts acute lethality for a taxonomic **class** absent from the
training data (insects / molluscs / amphibians) better than the strongest published interspecies
extrapolation methods." Scope is per block, never pooled.

## Prior art that must be beaten, and how it enters the protocol

| Prior work | What it does | Status here |
|---|---|---|
| **EPA Web-ICE** (interspecies correlation estimation; Raimondo, Barron et al., EPA; Web-ICE 3.x) | Least-squares log–log regression predicting an untested species' acute LC50/EC50 from a surrogate species' measured value; the regulatory standard for exactly this extrapolation, with published models for fish, invertebrates and amphibians | **Mandatory comparator** (`ice_regression`), re-fitted on the pool in the method's own form; applicability (82–91 % of primary rows) pre-registered. **Metadata to verify** before any claim cites it |
| **Chemical-matched surrogate / read-across** (regulatory default, e.g. ECHA/OECD guidance) | Carry the same chemical's value from a tested surrogate taxon, as geometric mean or most-sensitive | **Mandatory comparator** (`fish_surrogate_geomean`, `fish_surrogate_most_sensitive`) |
| **Zubrod et al., *Environment International* 186:108607 (2024)** | GPBoost (boosted trees + grouped random effects) for cross-chemical **and cross-species** sensitivity prediction | **Mandatory published comparator** (`gpboost_published`), now reproducible via `SeededGPBoost`. *Repo-verified* (`docs/novelty_checks/chemtox_prospective.md`) |
| **Yuan et al., *ET&C* 45(9):2466–2482 (2026)** | Benchmarks GPBoost as the best 2D-structure-based model for cross-species prediction | Sets the bar this track must clear to claim SOTA. *Repo-verified* |
| **Gasser et al. (2024) / ADORE** (Schür et al. 2023, DOI 10.25678/0008C9) | The within-taxa acute benchmark and its RF/XGBoost reference | Comparator, but explicitly **not** "the strongest published method for this task": it is a within-taxa benchmark. *Repo-verified* |
| **Yang et al., *ES&T* 60(3):2556–2565 (2026)**; Forastiere et al., *Ecotox. Environ. Saf.* 314:120057 (2026) | Species embeddings from taxonomy + genetic sequence | Not implemented; the only known route to a genuine cross-taxon channel (see the design caveat in the protocol §9). *Repo-verified metadata* |
| **TRIDENT** | Transformer ecotoxicity model | Deliberately not run: its training data covers the period and records these blocks come from, so it would measure memorisation. Recorded as a limitation |

## Honest position on novelty

The *data arrangement* (whole-class hold-out from ECOTOX under ADORE's curation rules, with
chemical novelty as a reported stratum) is, as far as the author knows, not published in this form —
but **"not found" is not "does not exist"**, and no literature search was run in this session. The
*methods* are all published. So the defensible contribution of this track is **the evaluation**:
a pre-registered measurement of how far chemistry-only models carry to an unseen class, against the
regulatory extrapolation methods that exist for precisely that purpose. Any stronger wording
requires a dated search that is actually performed and recorded here first.

**Amendment (2026-11-09, before any fit): the two cross-taxon channels of protocol §1.4.** Models
may now use (A) chemical-matched surrogate information computed from training-pool rows only —
fish/pool geomean, most-sensitive value, ICE prediction, availability flag and record counts — and
(B) taxonomy **strictly above** the held-out class (phylum / subphylum / clade). Both are given to
the mandatory comparators as well (`*_chan` variants, including `gpboost_published_chan`), so the
comparison stays a model comparison.

Novelty consequence, stated honestly: **neither channel is itself novel.** Channel A is the input
of the ICE/read-across methods in the table above, and Channel B is ordinary taxonomic encoding;
giving them to a learner is an engineering choice, not a new method. What remains potentially
novel is the *evaluation*: a pre-registered, whole-class hold-out comparison of learners and
regulatory extrapolation methods **given identical information**. The residual-learning variant
(`ICE + learned correction`) is the closest thing here to a methodological contribution, and even
that is a standard residual-model construction; any claim about it must say so and must cite the
ICE literature as the base method.

The design caveat of the first revision (no cross-taxon channel at all) is therefore superseded,
but the expectation remains measured rather than optimistic: the surrogate channel should dominate
wherever a surrogate exists, so the informative results are how much a learner adds on top of ICE
and what happens on the surrogate-unavailable rows.
