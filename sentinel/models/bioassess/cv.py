# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Out-of-fold prediction runner for baselines and BioSet candidates (protocol §6).

Each fold sees only its training rows: features, vocabularies, standardisers and early-stopping
validation (a 15% block-held-out subset of the *training* locations) are all derived from them.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.models.bioassess import baselines as B
from sentinel.models.bioassess import bioset as BS
from sentinel.models.bioassess import splits as SP
from sentinel.models.bioassess.data import TARGET_NAMES
from sentinel.models.bioassess.features import DESCRIPTORS, GEOGRAPHY, SEASON

log = logging.getLogger("bioassess.cv")


def load_bioset_config(path: str | Path, candidate: str) -> dict:
    import yaml

    doc = yaml.safe_load(Path(path).read_text())
    cfg = dict(doc["defaults"])
    cfg.update(doc["candidates"][candidate])
    cfg["candidate"] = candidate
    return cfg


def oof_baseline(name: str, design, folds, *, seed: int = 0, params: dict | None = None, n_jobs: int = 8,
                 targets=TARGET_NAMES) -> pd.DataFrame:
    parts = []
    for i, f in enumerate(folds):
        tr, te = SP.masks(design.samples, f)
        if not te.any():
            continue
        p = B.fit_predict(name, design, design.samples.index[tr], design.samples.index[te], seed=seed,
                          params=params, n_jobs=n_jobs, targets=targets)
        p["fold"] = i
        parts.append(p)
    return pd.concat(parts).reindex(design.samples.index)


def _glob(design, ids, geography, fill: dict | None = None) -> np.ndarray:
    """Sample descriptors (+ geography [+ sampling DOY when ``geography == "context"``]);
    missing geography filled with *training-fold* medians. A column with no observed value in the
    training fold (non-finite fill) is unavailable in that fold: it is set to 0 for *every* row, so
    observed test values never reach an input weight that training could not fit."""
    cols = list(DESCRIPTORS) + (list(GEOGRAPHY) if geography else []) + \
        (list(SEASON) if geography == "context" else [])
    s = design.samples.loc[pd.Index(ids), cols].astype(float)
    if geography:
        fill = fill or {}
        dead = [c for c, v in fill.items() if not np.isfinite(v)]
        s = s.fillna({c: v for c, v in fill.items() if c not in dead})
        if dead:
            s[dead] = 0.0
    return s.to_numpy(np.float32)


class PretrainedEncoder:
    """A UK-pretrained encoder checkpoint (vocab + encoder weights + names seen in the pool)."""

    def __init__(self, path: str | Path):
        import torch

        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.path = str(path)
        self.vocab = BS.TaxonVocab.from_dict(ck["vocab"])
        self.encoder_state = ck["encoder_state"]
        self.pool_allowed = [np.asarray(a, bool) for a in ck["pool_allowed"]]
        self.cfg = ck["cfg"]


def _check_pretrained(cfg, pretrained):
    if pretrained is not None:
        for k in ("arch", "d_model", "n_layers", "n_heads", "embedding"):
            if pretrained.cfg.get(k) != cfg.get(k):
                raise ValueError(f"pretrained encoder {k}={pretrained.cfg.get(k)} != candidate {cfg.get(k)}")


def _target_matrix(cfg, design, targets):
    Y = design.Y[list(targets)].to_numpy(np.float32)
    n_primary = Y.shape[1]
    if cfg.get("aux_targets"):
        if design.A is None:
            raise ValueError("candidate uses auxiliary targets but the design has none")
        Y = np.hstack([Y, design.A.to_numpy(np.float32)])   # training-only outputs
    return Y, n_primary


def bioset_rows(cfg: dict, design, train_ids, test_ids, *, vocab, seeds=(0, 1, 2), device: str = "cpu",
                geography=False, pretrained: PretrainedEncoder | None = None, targets=TARGET_NAMES,
                fold: int = 0, fold_log=None):
    """Fit the candidate on ``train_ids`` (inner block-held-out validation from *their* locations)
    and predict ``test_ids``. Returns (seed-mean predictions, {seed: predictions}, diagnostics)."""
    Y, n_primary = _target_matrix(cfg, design, targets)
    row = pd.Series(np.arange(len(design.samples)), index=design.samples.index)
    train_ids, te_ids = pd.Index(train_ids), pd.Index(test_ids)
    if set(train_ids) & set(te_ids):
        raise AssertionError("train and test rows overlap")
    s_tr = design.samples.loc[train_ids]
    fit_locs, val_locs = SP.inner_validation(s_tr, sorted(set(s_tr["location_id"])))
    fit_ids = train_ids[s_tr["location_id"].isin(fit_locs).to_numpy()]
    val_ids = train_ids[s_tr["location_id"].isin(val_locs).to_numpy()]
    fit_tok = design.tokens[design.tokens["visit_id"].isin(fit_ids)]
    allowed = vocab.allowed_from(fit_tok, int(cfg.get("min_name_count", 3)))
    if pretrained is not None:
        allowed = [a | p for a, p in zip(allowed, pretrained.pool_allowed)]
    geo_fill = ({c: float(design.samples.loc[fit_ids, c].median()) for c in GEOGRAPHY} if geography else None)
    if geo_fill and not all(np.isfinite(v) for v in geo_fill.values()):
        log.warning("fold %d: geography column(s) %s unobserved in the fit rows; unavailable in this fold",
                    fold, [c for c, v in geo_fill.items() if not np.isfinite(v)])

    def enc(ids):
        e = BS.encode(design.tokens, np.asarray(ids), vocab, _glob(design, ids, geography, geo_fill))
        return BS.restrict(e, allowed)

    e_fit, e_val, e_te = enc(fit_ids), enc(val_ids), enc(te_ids)
    preds, per_seed, diag = [], {}, []
    for s in seeds:
        fitted = BS.fit_bioset(cfg, vocab, e_fit, Y[row[fit_ids].to_numpy()], e_val, Y[row[val_ids].to_numpy()],
                               seed=s, device=device,
                               init_encoder=None if pretrained is None else pretrained.encoder_state,
                               log=(lambda m, i=fold, s=s: log.info("fold %d seed %d: %s", i, s, m)),
                               n_primary=n_primary,
                               region_tr=pd.factorize(design.samples.loc[fit_ids, "ag_eco9"])[0])
        p = fitted.predict(e_te, device=device)[:, :n_primary]
        per_seed[s] = p
        preds.append(p)
        diag.append({"fold": fold, "seed": s, "best_epoch": fitted.best_epoch, "n_epochs": len(fitted.history),
                     "best_val_loss": float(min(h[2] for h in fitted.history)), "n_params": fitted.n_params,
                     "n_fit": len(fit_ids), "n_val": len(val_ids), "n_test": len(te_ids),
                     "n_names_allowed": int(allowed[-1].sum())})
        if fold_log:
            fold_log(diag[-1])
    return np.mean(preds, 0), per_seed, diag


def oof_bioset(cfg: dict, design, folds, *, seeds=(0, 1, 2), device: str = "cpu", geography=False,
               pretrained: PretrainedEncoder | None = None, targets=TARGET_NAMES,
               fold_log=None) -> tuple[pd.DataFrame, dict[int, pd.DataFrame], list[dict]]:
    """Return (ensemble OOF, per-seed OOF, fold diagnostics)."""
    vocab = pretrained.vocab if pretrained is not None else BS.TaxonVocab.build(design.tokens)
    _check_pretrained(cfg, pretrained)
    row = pd.Series(np.arange(len(design.samples)), index=design.samples.index)
    ens = np.full((len(row), len(targets)), np.nan)
    per_seed = {s: np.full((len(row), len(targets)), np.nan) for s in seeds}
    diag = []
    for i, f in enumerate(folds):
        tr_mask, te_mask = SP.masks(design.samples, f)
        if not te_mask.any():
            continue
        tr_ids, te_ids = design.samples.index[tr_mask], design.samples.index[te_mask]
        e, ps, d = bioset_rows(cfg, design, tr_ids, te_ids, vocab=vocab, seeds=seeds, device=device,
                               geography=geography, pretrained=pretrained, targets=targets, fold=i,
                               fold_log=fold_log)
        r = row[te_ids].to_numpy()
        ens[r] = e
        for s in seeds:
            per_seed[s][r] = ps[s]
        diag += d
    cols = list(targets)
    to_df = lambda a: pd.DataFrame(a, index=design.samples.index, columns=cols)  # noqa: E731
    return to_df(ens), {s: to_df(a) for s, a in per_seed.items()}, diag
