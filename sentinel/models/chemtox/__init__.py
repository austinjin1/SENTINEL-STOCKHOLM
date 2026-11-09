# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-ChemTox (M4): chemical structure -> aquatic toxicity on the ADORE benchmark.

Modules
  * :mod:`.features`  featurisation — the published ADORE feature set (Gasser et al. 2024) and
    RDKit MACCS/Morgan/descriptor featurisers for molecules outside ADORE;
  * :mod:`.data`      assembling (X, y, groups) for one ADORE challenge/split without re-splitting;
  * :mod:`.cv`        the official-fold CV loop, scoring, registry logging, configuration counting;
  * :mod:`.evaluate`  micro/macro metrics of Gasser et al. plus the regulatory and fold-error ones;
  * :mod:`.xgb`       XGBoost on a fingerprint + the published features (published-baseline model);
  * :mod:`.tabpfn_model` TabPFN v2 on fingerprints + descriptors;
  * :mod:`.chemprop_model` Chemprop v2 D-MPNN with species/taxonomy embedding and a
    duration/endpoint-conditioned head (optional CheMeleon foundation initialisation);
  * :mod:`.embeddings` frozen ChemBERTa-2 / MoLFormer embeddings + gradient boosting;
  * :mod:`.severity`  species-sensitivity distribution -> HC5 -> risk quotient (interface only).

Protocol (RESEARCH_PLAN.md §5.4, docs/research/04 §D.3): only ADORE's own split columns are used
(primary ``split_occurrence``, secondary ``split_scaffold-murcko``); model selection uses the
official CV folds 0-4 of the training portion; the ``test`` rows are scored once, after freezing.
"""

from . import evaluate, features

__all__ = ["evaluate", "features"]
