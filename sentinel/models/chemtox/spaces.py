# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Search spaces for the ChemTox improvement cycle — one fixed budget per model family.

The published baseline got 32 grid points (`scripts/train/chemtox_xgb.py`), so every SENTINEL
model gets the same budget here: :func:`sample` draws ``budget`` configurations from a documented
space with a fixed seed, so the search is reproducible and the count that lands in
``extra.n_configurations`` is the real number of configurations scored on the official folds.

Each sampled configuration is a dict with two keys:

* ``features`` — keyword arguments for :class:`sentinel.models.chemtox.data.FeatureSpec`;
* ``params``   — model hyper-parameters.

Nothing in these spaces touches the test rows: selection is by mean RMSE over folds 0-4.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_BUDGET = 32

#: Chemprop v2 D-MPNN. Includes the architectural variants the improvement cycle asks for:
#: CheMeleon initialisation, ECOTOX-pretrained encoder, the taxonomy embedding and the
#: duration/endpoint FiLM head as switchable components, multitask across taxa, seed ensembles.
CHEMPROP_SPACE: dict[str, Any] = {
    "features": {
        "fingerprint": [None, "MACCS"],
        "published": [True],
    },
    "params": {
        "message_hidden": [200, 300, 400],
        "depth": [3, 4, 5],
        "ffn_hidden": [200, 300, 600],
        "ffn_layers": [1, 2, 3],
        "dropout": [0.0, 0.05, 0.15],
        "max_lr": [5e-4, 1e-3, 2e-3],
        "batch_size": [64, 128],
        "max_epochs": [20, 30, 40],
        "species_dim": [0, 8, 16, 32],
        "use_conditioning": [True, True, False],       # the ablation is sampled, not forced
        "init": ["none", "none", "chemeleon", "ecotox"],
        "multitask": [False, False, False, True],
        "n_seeds": [1, 1, 1, 3],
    },
}

#: TabPFN v2: representation choice matters more than its own (nearly absent) hyper-parameters.
TABPFN_SPACE: dict[str, Any] = {
    "features": {
        "fingerprint": ["MACCS", "Morgan", "PubChem", "ToxPrint"],
        "published": [True],
        "rdkit_descriptors": [True, False],
    },
    "params": {
        "n_estimators": [1, 2, 4, 8],
        "max_train_rows": [3000, 6000, 9000],
        "max_features": [100, 200, 300, 500],
        "subsample_by": ["chemical", "row"],
        "predict_chunk": [2000],
    },
}

#: Frozen chemical-language-model embeddings + gradient boosting.
EMBEDDINGS_SPACE: dict[str, Any] = {
    "features": {
        "fingerprint": [None, "MACCS"],
        "published": [True],
    },
    "params": {
        "encoder": ["ChemBERTa-2", "ChemBERTa-2-MTR", "MoLFormer"],
        "pooling": ["cls", "mean"],
        "n_estimators": [300, 800, 1500],
        "eta": [0.02, 0.05, 0.1],
        "max_depth": [4, 6, 8],
        "min_child_weight": [1, 5],
        "subsample": [0.8, 1.0],
    },
}

SPACES = {"chemprop": CHEMPROP_SPACE, "tabpfn": TABPFN_SPACE, "embeddings": EMBEDDINGS_SPACE}


def sample(model: str, budget: int = DEFAULT_BUDGET, *, seed: int = 20261022) -> list[dict]:
    """``budget`` configurations drawn from ``model``'s space with a fixed seed (deduplicated)."""
    if model not in SPACES:
        raise KeyError(f"no search space for {model!r}; known: {sorted(SPACES)}")
    space = SPACES[model]
    rng = np.random.default_rng(seed)
    seen: set[str] = set()
    out: list[dict] = []
    attempts = 0
    while len(out) < budget and attempts < budget * 200:
        attempts += 1
        cfg = {
            "features": {k: _pick(rng, v) for k, v in space["features"].items()},
            "params": {k: _pick(rng, v) for k, v in space["params"].items()},
        }
        key = repr(sorted(cfg["features"].items())) + repr(sorted(cfg["params"].items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(cfg)
    if len(out) < budget:
        raise RuntimeError(f"space for {model!r} exhausted at {len(out)} < {budget} configurations")
    return out


def _pick(rng: np.random.Generator, values: list):
    v = values[int(rng.integers(0, len(values)))]
    return bool(v) if isinstance(v, (bool, np.bool_)) else v


def shard(configs: list[dict], index: int, count: int) -> list[tuple[int, dict]]:
    """Deterministic round-robin shard of a sampled configuration list (keeps global indices)."""
    if not (0 <= index < count):
        raise ValueError(f"shard index {index} out of range for {count} shards")
    return [(i, c) for i, c in enumerate(configs) if i % count == index]
