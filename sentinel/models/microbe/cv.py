# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Out-of-fold runners for SENTINEL-Microbe baselines and SeqSet candidates (protocol §6).

Each fold sees only its training rows: genus filters, CLR, context fills, sequence-table
standardisation, the id vocabulary (``seq_mode='id'``) and early stopping (whole training-fold
sites held out by :func:`splits.inner_validation`) are derived from them.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.core import paths
from sentinel.models.microbe import baselines as B
from sentinel.models.microbe import features as F
from sentinel.models.microbe import seqset as SS
from sentinel.models.microbe import splits as SP

log = logging.getLogger("microbe.cv")
EMB_DIR_NAME = "microbe_seqemb"


def load_config(path: str | Path, candidate: str) -> dict:
    import yaml

    doc = yaml.safe_load(Path(path).read_text())
    cfg = dict(doc["defaults"])
    cfg.update(doc["candidates"][candidate])
    cfg["candidate"] = candidate
    return cfg


def load_seq_table(encoder: str, seq_ids) -> tuple[np.ndarray, dict]:
    """Frozen sequence table for ``seq_ids`` (row 0 = zeros/UNK) and the seq_id -> row map."""
    d = paths.interim(EMB_DIR_NAME)
    ids = pd.read_parquet(d / f"{encoder}_ids.parquet")["seq_id"].to_numpy()
    emb = np.load(d / f"{encoder}.npy", mmap_mode="r")
    pos = pd.Series(np.arange(len(ids)), index=ids)
    want = pd.Index(pd.unique(np.asarray(seq_ids)))
    missing = want.difference(pos.index)
    if len(missing):
        raise KeyError(f"{len(missing)} sequences have no {encoder} embedding (re-run microbe_embed.py)")
    rows = pos.reindex(want).to_numpy()
    table = np.zeros((len(want) + 1, emb.shape[1]), np.float32)
    table[1:] = np.asarray(emb[rows], np.float32)
    return table, {s: i + 1 for i, s in enumerate(want)}


def oof_baseline(name: str, design, folds, *, targets, seed: int = 0, n_jobs: int = 6, device: str = "cpu",
                 params: dict | None = None, permute_seed: int | None = None) -> pd.DataFrame:
    """Out-of-fold predictions. ``permute_seed`` (permutation null): the target block of each fold's
    TRAINING rows is row-permuted among those rows only (held-out targets never enter)."""
    parts = []
    for i, f in enumerate(folds):
        d = design
        if permute_seed is not None:
            d = _with_targets(design, permuted_training_targets(design.samples, f.train, targets,
                                                                seed=permute_seed * 1000 + i))
        p = B.fit_predict(name, d, f.train, f.test, targets=targets, seed=seed, n_jobs=n_jobs,
                          params=params, device=device)
        p["fold"] = i
        parts.append(p)
    return pd.concat(parts)


def _with_targets(design, Y: pd.DataFrame):
    import copy

    d = copy.copy(design)
    s = design.samples.copy()
    s[list(Y.columns)] = Y.reindex(s.index).to_numpy()
    d.samples = s
    return d


def oof_seqset(cfg: dict, design, folds, *, targets, seeds=(0, 1, 2), device: str = "cpu",
               table: np.ndarray | None = None, seq_row: dict | None = None, fold_log=None,
               permute_seed: int | None = None, return_embeddings: bool = False):
    """Return (ensemble OOF, per-seed OOF, diagnostics[, OOF sample embeddings of seed 0])."""
    S = design.samples
    Ytrue = S[list(targets)].to_numpy(np.float32)
    row = pd.Series(np.arange(len(S)), index=S.index)
    ens = np.full((len(S), len(targets)), np.nan)
    per_seed = {s: np.full((len(S), len(targets)), np.nan) for s in seeds}
    embs = {}
    diag = []
    max_tokens = int(cfg.get("max_tokens", 256))
    mode = cfg.get("seq_mode", "frozen")
    for i, f in enumerate(folds):
        Yall = Ytrue
        if permute_seed is not None:
            Yall = permuted_training_targets(S, f.train, targets, seed=permute_seed * 1000 + i).reindex(
                S.index).to_numpy(np.float32)
        fit_ids, val_ids = SP.inner_validation(S, f.train, frac=float(cfg.get("inner_val_frac", 0.15)), seed=0,
                                               unit=cfg.get("inner_val_unit", "site"))
        te_ids = np.asarray(f.test)
        assert not (set(fit_ids) | set(val_ids)) & set(te_ids)
        if mode == "frozen":
            srow, tab, n_vocab = seq_row, table, 1
        else:
            tok = design.tokens[design.tokens["sample_id"].isin(fit_ids)]
            cnt = tok["seq_id"].value_counts()
            vocab = sorted(cnt.index[cnt >= int(cfg.get("min_seq_count", 3))])
            srow, tab, n_vocab = {s: k + 1 for k, s in enumerate(vocab)}, None, len(vocab) + 1
        ctx_fill = None
        if cfg.get("context", False):
            _, ctx_fill = F.context_features(S, fit_ids)

        def enc(ids):
            g = F.sample_descriptors(S, ids, primer_onehot=bool(cfg.get("primer_onehot", True)),
                                     enabled=bool(cfg.get("descriptors", True)))
            if cfg.get("context", False):
                g = np.column_stack([g, F.context_features(S, ids, ctx_fill)[0]])
            return SS.encode_sets(design.tokens, ids, srow, g, max_tokens)

        b_fit, b_val, b_te = enc(fit_ids), enc(val_ids), enc(te_ids)
        preds = []
        for s in seeds:
            fitted = SS.fit_seqset(cfg, tab, b_fit, Yall[row[fit_ids].to_numpy()], b_val,
                                   Yall[row[val_ids].to_numpy()], seed=s, device=device, n_vocab=n_vocab,
                                   log=(lambda m, i=i, s=s: log.info("fold %d seed %d: %s", i, s, m)))
            p = fitted.predict(b_te, device=device)
            per_seed[s][row[te_ids].to_numpy()] = p
            preds.append(p)
            if return_embeddings and s == seeds[0]:
                embs.update(dict(zip(te_ids, fitted.embed(b_te, device=device))))
            diag.append({"fold": f.name, "seed": s, "best_epoch": fitted.best_epoch, "n_epochs": len(fitted.history),
                         "best_val_loss": float(min(h[2] for h in fitted.history)), "n_params": fitted.n_params,
                         "n_fit": len(fit_ids), "n_val": len(val_ids), "n_test": len(te_ids)})
            if fold_log:
                fold_log(diag[-1])
        ens[row[te_ids].to_numpy()] = np.mean(preds, 0)
    cols = list(targets)
    to_df = lambda a: pd.DataFrame(a, index=S.index, columns=cols)
    out = (to_df(ens), {s: to_df(a) for s, a in per_seed.items()}, diag)
    if return_embeddings:
        out = out + (embs,)
    return out


def permuted_training_targets(samples: pd.DataFrame, train_ids, targets, seed: int) -> pd.DataFrame:
    """Copy of the target block where the rows of ``train_ids`` are permuted among themselves (all
    targets together: joint distribution kept, community -> target link broken); other rows keep
    their true values (they are only ever used for scoring)."""
    rs = np.random.default_rng(seed)
    Y = samples[list(targets)].copy()
    tr = pd.Index(train_ids)
    Y.loc[tr] = Y.loc[tr].to_numpy()[rs.permutation(len(tr))]
    return Y
