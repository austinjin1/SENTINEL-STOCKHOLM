"""ChemTox-RX: the concatenated-embedding arms, and the single-task chemprop wrapper.

New module, no project row read. RX has **no novelty whatsoever** and is in the plan because it is
the highest expected-gain-per-GPU-hour item on the board: embed every unique structure once, append
the vector to the published block (never substitute it, see :mod:`arch_v2_embed`), and give the same
block to every parity comparator (R4, amendment §4.3).

``rx_chemprop`` is a thin wrapper over the installed ``chemprop`` package. Two pre-registered
properties are enforced here rather than hoped for:

* **single-task** — amendment §4.1 / the T2 lesson C-M4U-4: a multitask chemprop routes an unknown
  ``tax_group`` to its fish head. :func:`assert_single_task` makes ``multitask=True`` inadmissible.
* **no species key** — ``ConditionEncoder`` keys a species embedding on ``tax_gs``; a constant
  placeholder is used so a ``species_dim > 0`` draw is an **information-free learned bias**, which
  is disclosed rather than silently helpful (the T2 A7.2 ruling, carried forward).

``chemprop_model.py`` is existing code shared with the closed tracks and is **not** edited; this
module wraps it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["RxError", "SPECIES_PLACEHOLDER", "assert_single_task", "assert_no_species_key",
           "RxGbmConfig", "build_rx_gbm", "RxChempropConfig", "rx_chemprop_params",
           "placeholder_species_frame"]


class RxError(ValueError):
    """A pre-registered rule about an RX arm was violated."""


#: The constant key put on the frames handed to chemprop, never on the pool.
SPECIES_PLACEHOLDER = "__dropped__"


def assert_single_task(params: Mapping) -> None:
    """``multitask=True`` is inadmissible: it routes an unseen taxon to the fish head."""
    if params.get("multitask"):
        raise RxError(
            "chemprop must be pre-registered SINGLE-TASK on these tracks: a multitask head routes "
            "an unknown tax_group to the fish head (the C-M4U-4 defect). multitask=True is "
            "inadmissible.")


def assert_no_species_key(frame: pd.DataFrame, *, column: str = "tax_gs") -> None:
    """The frame handed to chemprop must carry a CONSTANT species key, or none at all."""
    if column not in frame.columns:
        return
    vals = frame[column].astype(str).unique()
    if len(vals) > 1:
        raise RxError(
            f"{column!r} takes {len(vals)} distinct values on the frame handed to chemprop: a real "
            f"species key would give the species embedding information the protocol does not "
            f"admit. Use placeholder_species_frame().")
    if vals[0] != SPECIES_PLACEHOLDER:
        raise RxError(f"{column!r} must be the constant placeholder {SPECIES_PLACEHOLDER!r}")


def placeholder_species_frame(frame: pd.DataFrame, *, column: str = "tax_gs") -> pd.DataFrame:
    """A COPY of the frame with a constant species key. The pool itself is never modified."""
    out = frame.copy()
    out[column] = SPECIES_PLACEHOLDER
    return out


@dataclass
class RxGbmConfig:
    """``rx_lightgbm`` / ``rx_catboost`` / ``rx_lightgbm_dur``: a GBM on block + embedding."""

    family: str = "lightgbm"
    n_estimators: int = 600
    learning_rate: float = 0.05
    num_leaves: int = 63
    max_depth: int = -1
    subsample: float = 1.0
    colsample: float = 1.0
    with_embedding: bool = True
    with_duration: bool = False

    def __post_init__(self) -> None:
        if self.family not in ("lightgbm", "catboost"):
            raise RxError(f"unsupported family {self.family!r}")


def build_rx_gbm(cfg: RxGbmConfig, feature_columns: Sequence[str]):
    """Construct the estimator and record its feature contract for the parity log."""
    from sentinel.models.chemtox import arch_v2_embed as E

    cols = list(feature_columns)
    has_emb = any(c.startswith(E.EMB_PREFIX) for c in cols)
    if cfg.with_embedding and not has_emb:
        raise RxError(
            f"with_embedding is set but no {E.EMB_PREFIX}* column is present: the embedding is "
            f"CONCATENATED to the published block, so it must be in the feature list")
    if not cfg.with_embedding and has_emb:
        raise RxError("with_embedding is False but embedding columns were supplied "
                      "(this is the lightgbm_noemb information control)")
    if cfg.with_duration and "duration_h" not in cols:
        raise RxError("with_duration is set but duration_h is absent (F10)")
    if cfg.family == "lightgbm":
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(n_estimators=cfg.n_estimators, learning_rate=cfg.learning_rate,
                              num_leaves=cfg.num_leaves, max_depth=cfg.max_depth,
                              subsample=cfg.subsample, colsample_bytree=cfg.colsample,
                              verbose=-1)
    else:
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(iterations=cfg.n_estimators, learning_rate=cfg.learning_rate,
                                  depth=6 if cfg.max_depth < 0 else cfg.max_depth,
                                  verbose=0, allow_writing_files=False)
    model._sentinel_feature_columns = cols
    model._sentinel_rx = {"family": cfg.family, "with_embedding": cfg.with_embedding,
                          "with_duration": cfg.with_duration,
                          "n_embedding_columns": sum(c.startswith(E.EMB_PREFIX) for c in cols)}
    return model


@dataclass
class RxChempropConfig:
    """``rx_chemprop``: the D-MPNN, single-task, with a constant species key."""

    message_hidden: int = 300
    depth: int = 3
    ffn_hidden: int = 300
    ffn_layers: int = 2
    dropout: float = 0.0
    max_lr: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 30
    n_seeds: int = 1
    species_dim: int = 0
    multitask: bool = False
    accelerator: str = "gpu"

    def __post_init__(self) -> None:
        assert_single_task({"multitask": self.multitask})
        if self.n_seeds < 1:
            raise RxError("n_seeds must be at least 1")


def rx_chemprop_params(cfg: RxChempropConfig) -> dict:
    """The parameter dict handed to the existing chemprop wrapper, with the disclosures attached."""
    assert_single_task({"multitask": cfg.multitask})
    out = {"message_hidden": cfg.message_hidden, "depth": cfg.depth,
           "ffn_hidden": cfg.ffn_hidden, "ffn_layers": cfg.ffn_layers, "dropout": cfg.dropout,
           "max_lr": cfg.max_lr, "batch_size": cfg.batch_size, "max_epochs": cfg.max_epochs,
           "n_seeds": cfg.n_seeds, "species_dim": cfg.species_dim, "multitask": False,
           "accelerator": cfg.accelerator}
    if cfg.species_dim > 0:
        out["_disclosure_species_dim"] = (
            "species_dim > 0 with a constant placeholder species key is an INFORMATION-FREE "
            "learned bias, not a species channel; disclosed with every result.")
    return out
