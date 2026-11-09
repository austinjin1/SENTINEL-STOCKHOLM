# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Frozen pretrained chemical-language-model embeddings + gradient boosting.

The molecule is encoded once by a frozen transformer (no fine-tuning in phase 1), the embedding is
cached per unique SMILES, and a gradient-boosted tree model is fit on
``[embedding | published experimental/chemical/taxonomic features]``.

Supported encoders (both load through ``transformers``, as recommended in docs/research/04 §B.2):

* ``ChemBERTa-2`` — ``DeepChem/ChemBERTa-77M-MLM`` (the 77M-molecule MLM checkpoint);
* ``MoLFormer`` — ``ibm/MoLFormer-XL-both-10pct`` (needs ``trust_remote_code=True``; the licence is
  Apache-2.0 for the code and the released weights, research use).

Pooling is the ``[CLS]``/first-token state by default (``mean`` pooling is also available), which
is what TRIDENT uses on its own ChemBERTa encoder. The embedding is a *frozen* representation, so
the fitted part of the pipeline is only the GBM — no transformer weights are updated, nothing is
learned from validation or test rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ENCODERS = {
    "ChemBERTa-2": {"model": "DeepChem/ChemBERTa-77M-MLM", "trust_remote_code": False},
    "ChemBERTa-2-MTR": {"model": "DeepChem/ChemBERTa-77M-MTR", "trust_remote_code": False},
    "MoLFormer": {"model": "ibm/MoLFormer-XL-both-10pct", "trust_remote_code": True},
}
SMILES_COL = "chem_rdkit_can_smiles"

#: process-wide embedding cache keyed by (encoder, pooling): a frozen encoder gives the same vector
#: for the same SMILES in every fold and configuration, so each molecule is embedded once.
_SHARED_CACHE: dict[tuple[str, str], dict[str, np.ndarray]] = {}


@dataclass
class FrozenEncoder:
    """Cached frozen transformer embeddings of SMILES strings."""

    name: str = "ChemBERTa-2"
    pooling: str = "cls"
    device: str = "cpu"
    batch_size: int = 64
    max_length: int = 256
    cache_: dict[str, np.ndarray] = field(default_factory=dict)
    dim_: int | None = None

    def _load(self):
        from transformers import AutoModel, AutoTokenizer

        if self.name not in ENCODERS:
            raise KeyError(f"unknown encoder {self.name!r}; known: {sorted(ENCODERS)}")
        spec = ENCODERS[self.name]
        tok = AutoTokenizer.from_pretrained(spec["model"], trust_remote_code=spec["trust_remote_code"])
        mdl = AutoModel.from_pretrained(spec["model"], trust_remote_code=spec["trust_remote_code"])
        mdl.eval()
        return tok, mdl.to(self.device)

    def embed(self, smiles: list[str]) -> np.ndarray:
        """Embeddings for a list of SMILES (cached per unique string)."""
        import torch

        todo = sorted({s for s in smiles if s not in self.cache_})
        if todo:
            tok, mdl = self._load()
            with torch.no_grad():
                for i in range(0, len(todo), self.batch_size):
                    chunk = todo[i:i + self.batch_size]
                    enc = tok(chunk, padding=True, truncation=True, max_length=self.max_length,
                              return_tensors="pt").to(self.device)
                    out = mdl(**enc).last_hidden_state
                    if self.pooling == "mean":
                        m = enc["attention_mask"].unsqueeze(-1).to(out.dtype)
                        vec = (out * m).sum(1) / m.sum(1).clamp(min=1)
                    else:
                        vec = out[:, 0]
                    for s, v in zip(chunk, vec.cpu().numpy()):
                        self.cache_[s] = v.astype(np.float32)
            del mdl
        self.dim_ = len(next(iter(self.cache_.values())))
        zero = np.zeros(self.dim_, dtype=np.float32)
        return np.vstack([self.cache_.get(s, zero) for s in smiles])


@dataclass
class EmbeddingGBM:
    """Frozen-embedding + GBM model with the ChemTox ``fit``/``predict`` interface.

    ``params``: ``encoder`` (key of :data:`ENCODERS`), ``pooling``, ``device``, plus any XGBoost
    hyper-parameter (``n_estimators``, ``eta``, ``max_depth``, ...).
    """

    params: dict = field(default_factory=dict)
    seed: int = 0
    encoder_: FrozenEncoder | None = None

    def __post_init__(self):
        self.model_ = None

    def _embed_rows(self, rows: pd.DataFrame) -> np.ndarray:
        smiles = rows[SMILES_COL].astype("string").fillna("").to_list()
        return self.encoder_.embed(smiles)

    def fit(self, X, y, *, train_rows: pd.DataFrame, **_ctx) -> EmbeddingGBM:
        from .xgb import XGBoostTox

        p = dict(self.params)
        name = str(p.pop("encoder", "ChemBERTa-2"))
        pooling = str(p.pop("pooling", "cls"))
        self.encoder_ = FrozenEncoder(name=name, pooling=pooling,
                                      device=str(p.pop("device", "cpu")),
                                      batch_size=int(p.pop("encoder_batch_size", 64)),
                                      cache_=_SHARED_CACHE.setdefault((name, pooling), {}))
        Z = np.column_stack([np.asarray(X, dtype=np.float32), self._embed_rows(train_rows)])
        self.model_ = XGBoostTox(params=p, seed=self.seed).fit(Z, y)
        return self

    def predict(self, X, *, rows: pd.DataFrame | None = None, **_ctx) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("EmbeddingGBM.predict before fit")
        if rows is None:
            raise ValueError("EmbeddingGBM.predict needs the row frame (SMILES column)")
        Z = np.column_stack([np.asarray(X, dtype=np.float32), self._embed_rows(rows)])
        return self.model_.predict(Z)


def make_embedding_gbm(seed: int = 0):
    return lambda params: EmbeddingGBM(params=params, seed=seed)
