# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Improvement cycle 2 for ChemTox: CV with optional extra training rows, and its search spaces.

Everything here is **selection-only**: scores come from ADORE's official CV folds 0–4 inside the
``t-F2F``/``split_occurrence`` training partition, and predictions are always scored on the
*ADORE* validation rows (extra rows are training data, never evaluation data). The prospective
test set is never read.

Two things this adds over :mod:`sentinel.models.chemtox.cv`:

* **Augmented training rows.** ECOTOX records curated by ADORE's rules that ADORE itself does not
  contain (:func:`sentinel.models.chemtox.prospective.augmentation_rows`), and/or the other ADORE
  challenges' training rows (``t-C2C``, ``t-A2A``) for multi-taxa training. Per fold, every
  chemical in the validation fold is removed from the extra rows (``aux.drop_chemicals``), and
  every prospective-test chemical is removed globally and asserted gone, so neither the fold's
  validation rows nor the future test rows can be learned through a side door.
* **Equal-budget random search** over the new model families (LightGBM, CatBoost, k-NN
  read-across) *and* over the baselines (XGBoost, random forest) under the new feature options, so
  the comparison is between equally tuned models (RESEARCH_PLAN §15.5 guard 4).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import aux as AUX
from . import evaluate as E
from . import prospective as PRO
from .cv import CVResult
from .data import ChemToxDataset, FeatureSpec

DEFAULT_BUDGET = 32

#: Columns that exist for ADORE's chemicals but **cannot be obtained for a new chemical**: ADORE
#: compiled experimental water solubility and melting point per chemical from PubChem, and there is
#: no way to look them up for a chemical outside the benchmark. At prospective-test time they are
#: therefore missing for every row and take the training mean
#: (:class:`sentinel.models.chemtox.features.PublishedFeatures`).
#:
#: If CV let the validation rows keep their real values, selection would measure a feature that the
#: deployed model will not have, and would favour the ``chemical='four'`` spec precisely because it
#: degrades at test time — a bias against the specs that only use computable properties, which the
#: pre-registration's §4.3 comparison exists to decide. :func:`run_cv_extra` therefore blanks these
#: columns on the **validation** rows of every fold (training rows keep them, as a deployed model's
#: training data would). ``deployment_masking=False`` reproduces the unmasked behaviour.
DEPLOYMENT_MISSING_COLUMNS = ("chem_ws", "chem_mp")

# --------------------------------------------------------------------------- search spaces

#: feature options shared by the tabular families. ``chemical`` is the pre-registered question of
#: §4.3 of the prospective plan: ``four`` uses ADORE's experimental water solubility and melting
#: point (imputed for new chemicals), ``mw_logp`` uses only what RDKit can compute for any molecule.
TABULAR_FEATURES = {
    "fingerprint": ["MACCS", "Morgan"],
    "published": [True],
    "chemical": ["four", "mw_logp"],
    "rdkit_descriptors": [True, False],
    "rdkit_descriptors_full": [True, False],
}

XGB_SPACE = {
    "features": TABULAR_FEATURES,
    "params": {
        "n_estimators": [300, 800, 1500],
        "eta": [0.02, 0.03, 0.1],
        "max_depth": [6, 8, 10],
        "min_child_weight": [1, 5],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.5, 0.8, 1.0],
        "reg_lambda": [1.0, 5.0],
    },
}
RF_SPACE = {
    "features": TABULAR_FEATURES,
    "params": {
        "n_estimators": [300, 600],
        "max_depth": [None, 20, 40],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": [0.3, "sqrt", 0.6],
    },
}
LGBM_SPACE = {
    "features": TABULAR_FEATURES,
    "params": {
        "n_estimators": [500, 1500, 3000],
        "eta": [0.01, 0.03, 0.05],
        "num_leaves": [31, 63, 127, 255],
        "min_child_samples": [5, 20, 50],
        "feature_fraction": [0.3, 0.6, 1.0],
        "bagging_fraction": [0.7, 1.0],
        "bagging_freq": [1],
        "lambda_l2": [0.0, 1.0, 5.0],
    },
}
CATBOOST_SPACE = {
    "features": TABULAR_FEATURES,
    "params": {
        "iterations": [500, 1500, 3000],
        "eta": [0.03, 0.05, 0.1],
        "depth": [6, 8, 10],
        "l2_leaf_reg": [1.0, 3.0, 10.0],
        "rsm": [0.3, 0.7, 1.0],
        "bootstrap_type": ["Bernoulli"],
        "subsample": [0.8, 1.0],
    },
}
READACROSS_SPACE = {
    "features": {"fingerprint": ["MACCS"], "published": [True], "chemical": ["mw_logp"]},
    "params": {
        "fingerprint": ["Morgan", "MACCS"],
        "k": [1, 3, 5, 10, 20],
        "power": [1.0, 3.0, 6.0],
        "min_similarity": [0.0, 0.3, 0.5],
        "min_neighbours": [3, 10],
        "duration_penalty": [1.0, 0.7, 0.4],
        "match_species": [True, False],
    },
}
#: GPBoost: boosted trees + grouped random effects on the taxonomy (see
#: :mod:`sentinel.models.chemtox.gpboost_model` for the literature behind it).
GPBOOST_SPACE = {
    "features": TABULAR_FEATURES,
    "params": {
        "num_boost_round": [300, 800, 1500],
        "learning_rate": [0.02, 0.05, 0.1],
        "num_leaves": [31, 63, 127],
        "min_data_in_leaf": [5, 20, 50],
        "lambda_l2": [0.0, 1.0, 5.0],
        "feature_fraction": [0.3, 0.6, 1.0],
        "group_columns": [("tax_gs",), ("tax_gs", "tax_genus"),
                          ("tax_gs", "tax_genus", "tax_group"),
                          ("tax_gs", "tax_family", "tax_group")],
    },
}
#: Frozen CheMeleon fingerprints + gradient boosting (the representation change that the 2026
#: literature says should replace fine-tuning; :mod:`sentinel.models.chemtox.chemeleon_fp`).
CHEMELEON_FEATURES = {
    "fingerprint": [None, "MACCS"],
    "published": [True],
    "chemical": ["four", "mw_logp"],
    "chemeleon": [True],
    "chemeleon_components": [0, 128, 256],
    "rdkit_descriptors": [True, False],
    "rdkit_descriptors_full": [True, False],
}
CHEMELEON_GBDT_SPACE = {"features": CHEMELEON_FEATURES, "params": LGBM_SPACE["params"]}
CHEMELEON_XGB_SPACE = {"features": CHEMELEON_FEATURES, "params": XGB_SPACE["params"]}
CHEMELEON_GPBOOST_SPACE = {"features": CHEMELEON_FEATURES, "params": GPBOOST_SPACE["params"]}
#: TabPFN v2 on frozen CheMeleon fingerprints — the configuration that won 86% of 58 tasks in
#: arXiv:2604.16123. TabPFN wants few hundred features, so the fingerprint is PCA-reduced here
#: (fit on training rows only) and the row budget stays as in phase 1.
CHEMELEON_TABPFN_SPACE = {
    "features": {
        "fingerprint": [None],
        "published": [True],
        "chemical": ["four", "mw_logp"],
        "chemeleon": [True],
        "chemeleon_components": [64, 128, 256],
        "rdkit_descriptors": [True, False],
        "rdkit_descriptors_full": [False],
    },
    "params": {
        "n_estimators": [1, 2, 4, 8],
        "max_train_rows": [3000, 6000, 9000],
        "max_features": [200, 300, 500],
        "subsample_by": ["chemical", "row"],
        "predict_chunk": [2000],
    },
}

SPACES = {"xgboost": XGB_SPACE, "rf": RF_SPACE, "lightgbm": LGBM_SPACE,
          "catboost": CATBOOST_SPACE, "readacross": READACROSS_SPACE,
          "gpboost": GPBOOST_SPACE, "chemeleon_lightgbm": CHEMELEON_GBDT_SPACE,
          "chemeleon_xgboost": CHEMELEON_XGB_SPACE,
          "chemeleon_gpboost": CHEMELEON_GPBOOST_SPACE,
          "chemeleon_tabpfn": CHEMELEON_TABPFN_SPACE}
#: families whose model class differs from the search-space name
MODEL_ALIAS = {"chemeleon_lightgbm": "lightgbm", "chemeleon_xgboost": "xgboost",
               "chemeleon_gpboost": "gpboost", "chemeleon_tabpfn": "tabpfn"}


def sample(model: str, budget: int = DEFAULT_BUDGET, *, seed: int = 20261024) -> list[dict]:
    """``budget`` deduplicated configurations drawn from ``model``'s space with a fixed seed."""
    space = SPACES[model]
    rng = np.random.default_rng(seed)
    seen, out, attempts = set(), [], 0
    while len(out) < budget and attempts < budget * 500:
        attempts += 1
        cfg = {"features": {k: _pick(rng, v) for k, v in space["features"].items()},
               "params": {k: _pick(rng, v) for k, v in space["params"].items()}}
        if not cfg["features"].get("rdkit_descriptors", False):
            cfg["features"]["rdkit_descriptors_full"] = False   # the flag is meaningless then
        key = repr(sorted(cfg["features"].items())) + repr(sorted(
            (k, list(v) if isinstance(v, tuple) else v) for k, v in cfg["params"].items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(cfg)
    if len(out) < budget:
        raise RuntimeError(f"space for {model!r} exhausted at {len(out)}")
    return out


def _pick(rng: np.random.Generator, values: list):
    v = values[int(rng.integers(0, len(values)))]
    return bool(v) if isinstance(v, (bool, np.bool_)) else v


def shard(configs: list[dict], index: int, count: int) -> list[tuple[int, dict]]:
    return [(i, c) for i, c in enumerate(configs) if i % count == index]


#: Documented screening caps, applied only with ``--screen``. Screening scores all 32 sampled
#: configurations cheaply on a fold subset; the finalists are then re-run **as sampled** on all five
#: folds and only those full 5-fold scores are used for selection — the protocol phase 1 used for
#: chemprop and TabPFN (docs/preregistration/chemtox_test_plan.md §2).
SCREEN_CAPS: dict[str, dict] = {
    "xgboost": {"n_estimators": 500},
    "rf": {"n_estimators": 300},
    "lightgbm": {"n_estimators": 800},
    "catboost": {"iterations": 800},
    "gpboost": {"num_boost_round": 300},
    "chemeleon_lightgbm": {"n_estimators": 800},
    "chemeleon_xgboost": {"n_estimators": 500},
    "chemeleon_gpboost": {"num_boost_round": 300},
    "chemeleon_tabpfn": {"max_train_rows": 3000, "n_estimators": 2},
    "readacross": {},
}


def attach_device(family: str, params: dict, accelerator: str) -> dict:
    """Give the families that need one the device/accelerator they run on.

    TabPFN refuses to fit more than 1,000 rows on CPU, and chemprop needs its Lightning
    accelerator, so a configuration without this is not runnable — the first cycle-2 TabPFN screen
    failed on all 32 draws for exactly this reason.
    """
    out = dict(params)
    fam = MODEL_ALIAS.get(family, family)
    if fam in ("tabpfn", "embeddings") and "device" not in out:
        out["device"] = "cuda" if accelerator == "gpu" else "cpu"
    if fam.startswith("chemprop") and "accelerator" not in out:
        out["accelerator"] = accelerator
    return out


def apply_screen_caps(family: str, params: dict) -> dict:
    """Cap the expensive parameters of one sampled configuration (screening only)."""
    out = dict(params)
    for k, cap in SCREEN_CAPS.get(family, {}).items():
        if k in out and isinstance(out[k], (int, float)) and out[k] > cap:
            out[k] = cap
    return out


def make_model(model: str, seed: int) -> Callable[[dict], object]:
    model = MODEL_ALIAS.get(model, model)
    if model == "gpboost":
        from .gpboost_model import make_gpboost

        return make_gpboost(seed)
    if model == "xgboost":
        from .xgb import make_xgb

        return make_xgb(seed)
    if model == "rf":
        from .xgb import make_rf

        return make_rf(seed)
    if model == "lightgbm":
        from .gbdt import make_lgbm

        return make_lgbm(seed)
    if model == "catboost":
        from .gbdt import make_catboost

        return make_catboost(seed)
    if model == "readacross":
        from .readacross import make_readacross

        return make_readacross(seed)
    if model == "chemprop":
        from .chemprop_model import make_chemprop

        return make_chemprop(seed)
    if model == "tabpfn":
        from .tabpfn_model import make_tabpfn

        return make_tabpfn(seed)
    if model == "embeddings":
        from .embeddings import make_embedding_gbm

        return make_embedding_gbm(seed)
    raise KeyError(model)


# --------------------------------------------------------------------------- extra training rows


@dataclass
class ExtraRows:
    """Extra training rows (ADORE schema) plus the provenance of where they came from."""

    rows: pd.DataFrame
    info: dict
    name: str

    def __len__(self) -> int:
        return len(self.rows)


#: the other ADORE challenges, used as multi-taxa training data
OTHER_CHALLENGES = ("t-C2C", "t-A2A")


def adore_other_taxa_rows(challenge: str = "t-F2F",
                          others: tuple[str, ...] = OTHER_CHALLENGES) -> tuple[pd.DataFrame, dict]:
    """**All** rows of the other ADORE challenges, filtered against the prospective test set only.

    Phase 1 could use almost none of these rows (842 of 7,334), because the ADORE test pass required
    every chemical of *any* ``t-F2F`` test partition to be removed, and the crustacean/algae
    chemical space lies almost entirely inside it. The prospective protocol has a different and
    weaker constraint: the ADORE test rows are **training data** here (no claim is made on them
    again), so the only chemicals that must go are the prospective ones — which multiplies the
    available cross-taxa data and is the explicit lever on the cross-taxa weakness
    (``a-CA2F-diff`` test RMSE 1.257 in the ADORE pass).

    Inside CV the validation fold's chemicals are still removed per fold by
    :func:`run_cv_extra`, so no fold ever trains on its own validation chemicals.
    """
    from sentinel.data.sources import adore as A

    frames, info = [], {}
    for ch in others:
        if ch == challenge:
            continue
        df = A.load_challenge(ch, columns="all")
        n_in = len(df)
        kept = df
        try:
            kept = kept[~PRO._up(kept["test_cas"]).isin(PRO.prospective_keys()["cas"]).fillna(False)]
            kept = kept[~PRO._up(kept["chem_inchikey"]).isin(
                PRO.prospective_keys()["inchikey"]).fillna(False)]
        except FileNotFoundError:
            pass
        kept = kept.assign(aux_source=ch)
        PRO.assert_prospective_chemicals_absent(kept, what=f"ADORE {ch} multi-taxa rows")
        frames.append(kept)
        info[ch] = {"n_in": int(n_in), "n_kept": int(len(kept))}
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    info["n_rows"] = int(len(out))
    return out, info


def load_extra_rows(kind: str, *, challenge: str = "t-F2F", split: str = "split_occurrence",
                    require_adore_species: bool = True) -> ExtraRows:
    """Extra training rows, guaranteed free of the prospective test set.

    ``none``; ``ecotox`` (ADORE-curated ECOTOX records ADORE does not contain);
    ``adore_taxa`` (training rows of the other challenges under phase 1's stricter rule);
    ``adore_taxa_all`` (all rows of the other challenges, prospective chemicals removed — see
    :func:`adore_other_taxa_rows`); ``both`` = ``ecotox`` + ``adore_taxa_all``.
    """
    if kind == "none":
        return ExtraRows(rows=pd.DataFrame(), info={"kind": "none", "n_rows": 0}, name="none")
    frames, info = [], {"kind": kind}
    if kind in ("ecotox", "both"):
        eco, einfo = PRO.augmentation_rows(require_adore_species=require_adore_species)
        frames.append(eco)
        info["ecotox"] = {k: v for k, v in einfo.items() if k != "curation"}
    if kind == "adore_taxa":
        extra, ainfo = AUX.adore_auxiliary(challenge, split=split, columns="all")
        if len(extra):
            PRO.assert_prospective_chemicals_absent(extra, what="ADORE auxiliary rows")
        frames.append(extra)
        info["adore_taxa"] = {k: v for k, v in ainfo.items() if k != "keys"}
    if kind in ("adore_taxa_all", "both"):
        extra, ainfo = adore_other_taxa_rows(challenge)
        frames.append(extra)
        info["adore_taxa_all"] = ainfo
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(rows):
        PRO.assert_prospective_chemicals_absent(rows, what=f"extra training rows [{kind}]")
    info["n_rows"] = int(len(rows))
    return ExtraRows(rows=rows, info=info, name=kind)


def _align_columns(adore_rows: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """Give the extra rows the ADORE frame's columns (missing ones become NA, as for new chemicals)."""
    out = extra.reindex(columns=adore_rows.columns)
    return out


def mask_deployment_missing(rows: pd.DataFrame,
                            columns: Sequence[str] = DEPLOYMENT_MISSING_COLUMNS) -> pd.DataFrame:
    """Copy of ``rows`` with the columns a new chemical cannot have set to NaN.

    See :data:`DEPLOYMENT_MISSING_COLUMNS`. Applied to validation rows during CV so that model
    selection measures what the model will actually have at prospective-test time.
    """
    out = rows.copy()
    for c in columns:
        if c in out.columns:
            out[c] = np.nan
    return out


def run_cv_extra(
    ds: ChemToxDataset,
    spec: FeatureSpec,
    model_factory: Callable[[dict], object],
    params: dict | None = None,
    *,
    model_name: str = "model",
    folds: Sequence[int] | None = None,
    extra: ExtraRows | None = None,
    extra_weight: float = 1.0,
    deployment_masking: bool = True,
) -> CVResult:
    """CV on ADORE's official folds, optionally adding ``extra`` rows to each fold's training set.

    Scored **only** on the ADORE validation rows of each fold. Per fold the validation fold's
    chemicals are removed from the extra rows by CAS and InChIKey, and (by default) the columns a
    new chemical cannot have are blanked on the validation rows — see
    :data:`DEPLOYMENT_MISSING_COLUMNS`.
    """
    params = dict(params or {})
    y, chem, taxon = ds.y, ds.chem, ds.taxon
    oof = np.full(len(ds.df), np.nan)
    mask = np.zeros(len(ds.df), dtype=bool)
    fold_metrics: list[dict] = []
    n_extra_used: list[int] = []
    t0 = time.time()
    for k, trn, val in ds.cv_splits():
        if folds is not None and k not in folds:
            continue
        train_rows = ds.rows(trn)
        y_tr = y[trn]
        weights = None
        if extra is not None and len(extra):
            val_inchikeys = (ds.df["chem_inchikey"].iloc[val]
                             if "chem_inchikey" in ds.df.columns else ())
            keep = AUX.drop_chemicals(extra.rows, chem[val], val_inchikeys,
                                      cas_col="test_cas", inchikey_col="chem_inchikey")
            keep = _align_columns(train_rows, keep)
            y_extra = pd.to_numeric(keep[ds.TARGETS[ds.target]], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(y_extra)
            keep, y_extra = keep.loc[ok].reset_index(drop=True), y_extra[ok]
            n_extra_used.append(int(len(keep)))
            train_rows = pd.concat([train_rows, keep], ignore_index=True)
            y_tr = np.concatenate([y_tr, y_extra])
            if extra_weight != 1.0:
                weights = np.concatenate([np.ones(len(trn)), np.full(len(keep), extra_weight)])
        val_rows = ds.rows(val)
        if deployment_masking:
            val_rows = mask_deployment_missing(val_rows)
        builder = ds.builder(spec)
        Xtr = builder.fit_transform(train_rows)
        Xva = builder.transform(val_rows)
        model = model_factory(params)
        model.fit(Xtr, y_tr, train_rows=train_rows, valid_rows=val_rows,
                  feature_names=builder.names_, builder=builder, sample_weight=weights)
        p = np.asarray(model.predict(Xva, rows=val_rows), dtype=float)
        oof[val] = p
        mask[val] = True
        m = E.report(y[val], p, chem=chem[val], taxon=taxon[val])
        m.update({"fold": k, "n_features": builder.n_features, "n_train_rows": int(len(train_rows))})
        fold_metrics.append(m)
    keys = [k for k in E.METRIC_NAMES if all(k in m for m in fold_metrics)]
    mean = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
    sd = {k: float(np.std([m[k] for m in fold_metrics], ddof=1)) if len(fold_metrics) > 1
          else float("nan") for k in keys}
    pooled = E.report(y[mask], oof[mask], chem=chem[mask], taxon=taxon[mask])
    res = CVResult(model=model_name, spec=spec.to_dict(), params=params, fold_metrics=fold_metrics,
                   mean=mean, sd=sd, pooled=pooled, oof=oof, oof_mask=mask,
                   seconds=time.time() - t0)
    res.params = {**res.params, "_deployment_masking": bool(deployment_masking)}
    if n_extra_used:
        res.params = {**res.params, "_extra_rows_per_fold": n_extra_used,
                      "_extra_kind": extra.name if extra else "none",
                      "_extra_weight": extra_weight}
    return res
