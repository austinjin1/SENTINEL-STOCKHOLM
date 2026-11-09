# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M5-X BioSet configurations (protocol §7, amendment v4.9): one fold = one fit on the fold's training rows.

Shared by the development runner (``scripts/train/bioassess_x_bioset.py``) and the one-time scorer
(``scripts/eval/bioassess_x_final.py``) so both use the identical code path. Encoders are loaded only through
``xtrack.load_pretrained_encoder`` (then ``require_guarded``); per-fold UK pretraining writes its training-id record
from the ids it used. The frozen M5 modules (``bioset``, ``cv``, ``pretrain``, ``splits``) are imported, never edited.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.core import paths
from sentinel.core.registry import file_sha256
from sentinel.models.bioassess import xdev as XD
from sentinel.models.bioassess import xtrack as X

CONFIGS = {
    "bsx_flat_geo": {"embedding": "flat"},
    "bsx_flat_geo_rloss": {"embedding": "flat", "region_loss_weight": 1.0},
    "bsx_flat_geo_ukpt": {"embedding": "flat", "ukpt": True},
    "bsx_flat_geo_same": {"embedding": "flat", "same_continent": True},
}
SRC_PREFIX = "§"
TOKEN_LEVELS = ("phylum", "class", "order", "family", "genus", "name", "mixed_otu")


def special_tokens(D) -> pd.DataFrame:
    """One source token per sample and one analysis-method token per UK sample (set elements, ~0 abundance)."""
    s = D.samples
    src = SRC_PREFIX + "src:" + s["source"].astype(str)
    am = s["analysis_method"].astype(object).where(s["analysis_method"].isin(XD.METHODS), "other") \
        if "analysis_method" in s else pd.Series("other", index=s.index)
    uk = s["source"] == "uk"
    t = pd.concat([pd.DataFrame({"visit_id": s.index, "tok": src.to_numpy()}),
                   pd.DataFrame({"visit_id": s.index[uk],
                                 "tok": (SRC_PREFIX + "am:" + am[uk].astype(str)).to_numpy()})], ignore_index=True)
    out = pd.DataFrame({"visit_id": t["visit_id"]})
    for lv in TOKEN_LEVELS:
        out[lv] = t["tok"].to_numpy()
    out["count"] = 1e-6
    return out


def union_labels(D, same_block: str | None = None) -> pd.DataFrame:
    """The 9 union columns (shared targets merged across continents under the US name). ``same_block``: keep only
    rows of that block's continent."""
    Y = pd.DataFrame(index=D.samples.index)
    us = D.samples["source"].isin(XD.CONTINENT["X-L"]).to_numpy()
    for c in XD.UNION_COLUMNS:
        if c in XD.SHARED:
            Y[c] = np.where(us, D.Y[c].to_numpy(float), D.Y[XD.SHARED[c]].to_numpy(float))
        elif c in XD.TARGETS["X-L"]:
            Y[c] = np.where(us, D.Y[c].to_numpy(float), np.nan)
        else:
            Y[c] = np.where(~us, D.Y[c].to_numpy(float), np.nan)
    if same_block is not None:
        own = D.samples["source"].isin(XD.CONTINENT[same_block]).to_numpy()
        Y.loc[~own] = np.nan
    return Y


def samples_view(D) -> pd.DataFrame:
    """One region per 1-km location cluster (the cluster's first region). UK clusters can span EA areas and the frozen
    M5 inner-validation splitter requires one region per cluster. Used only for the inner early-stopping split and
    the within-batch region loss on training rows; folds and scoring keep the real region."""
    s = D.samples.copy()
    s["ag_eco9"] = s.groupby("location_id")["ag_eco9"].transform("first")
    return s


def to_block(P: pd.DataFrame, block: str) -> pd.DataFrame:
    out = pd.DataFrame(index=P.index)
    for t in XD.TARGETS[block]:
        out[t] = P[XD.SHARED_INV.get(t, t)]
    return out


def base_cfg(name: str) -> dict:
    from sentinel.models.bioassess.cv import load_bioset_config

    cfg = load_bioset_config(paths.REPO_ROOT / "configs/bioassess/bioset.yaml", "bioset_st_flat")
    cfg.update({k: v for k, v in CONFIGS[name].items() if k not in ("ukpt", "same_continent")})
    cfg["candidate"] = name
    return cfg


MAX_NOGRAD_ROWS = 2048


def _chunk_nograd(model, max_rows: int = MAX_NOGRAD_ROWS):
    """Evaluate no-grad forwards (the frozen ``fit_bioset`` validates on its whole validation set in one batch) in
    chunks of ``max_rows`` samples. The model is per-sample (set attention, token-wise layers), so outputs are the
    same; training forwards (grad enabled) are untouched."""
    import torch

    orig = model.forward

    def forward(lev, ab, mask, glob):
        if torch.is_grad_enabled() or lev.shape[0] <= max_rows:
            return orig(lev, ab, mask, glob)
        return torch.cat([orig(lev[i:i + max_rows], ab[i:i + max_rows], mask[i:i + max_rows], glob[i:i + max_rows])
                          for i in range(0, lev.shape[0], max_rows)])
    model.forward = forward
    return model


class _ChunkedBuild:
    """Context: models built by the frozen ``bioset.build_model`` get chunked no-grad forwards."""

    def __enter__(self):
        from sentinel.models.bioassess import bioset as BS

        self.BS, self.orig = BS, BS.build_model
        BS.build_model = lambda *a, **k: _chunk_nograd(self.orig(*a, **k))
        return self

    def __exit__(self, *exc):
        self.BS.build_model = self.orig
        return False


class Prepared:
    """Per-design state shared across folds/configs: token table with special tokens and the samples view."""

    def __init__(self, D):
        self.D = D
        self.tokens = pd.concat([D.tokens, special_tokens(D)], ignore_index=True)
        self.samples = samples_view(D)
        self._Y = {}

    def labels(self, same_block):
        if same_block not in self._Y:
            self._Y[same_block] = union_labels(self.D, same_block)
        return self._Y[same_block]


def pretrain_fold(prep: Prepared, block: str, fo, S, frozen, cfg, device, out_root: Path, seed: int = 0,
                  log=print) -> Path:
    """Per-fold UK pretraining (C-MX-1): the fold's training UK rows only; the sidecar records the ids used."""
    import torch
    import yaml

    from sentinel.models.bioassess import splits as SP
    from sentinel.models.bioassess.pretrain import pretrain_uk
    from sentinel.models.bioassess.uk import UK_TARGETS

    D = prep.D
    ids = fo.train[D.samples.loc[fo.train, "source"].to_numpy() == "uk"]
    key = hashlib.sha256(("\n".join(sorted(ids)) + f"|seed{seed}").encode()).hexdigest()[:16]
    ck_path = out_root / f"{block}_{key}" / "encoder.pt"
    if ck_path.exists() and ck_path.with_name("encoder.pt.train_ids.json").exists():
        return ck_path
    smp = prep.samples.loc[ids].copy()
    _, val_l = SP.inner_validation(smp, sorted(set(smp["location_id"])), frac=0.05)
    smp["role"] = np.where(smp["location_id"].isin(set(val_l)), "pool_val", "pool_fit")
    X.assert_fit_ids(XD.keys_of(smp.index), S, frozen)
    uk_tok = prep.tokens[prep.tokens["visit_id"].isin(set(ids))].rename(columns={"visit_id": "analysis_id"})
    other = prep.tokens[prep.tokens["visit_id"].isin(set(fo.train.difference(ids)))]
    pool = type("UKPool", (), {"samples": smp, "tokens": uk_tok, "Y": D.Y.loc[ids, list(UK_TARGETS)]})
    pcfg = dict(yaml.safe_load((paths.REPO_ROOT / "configs/bioassess/bioset.yaml").read_text())["pretrain_uk"])
    t0 = time.time()
    ck, hist = pretrain_uk(cfg, pcfg, pool, other, device=device, seed=seed, log=lambda m: log("pretrain", m))
    ck_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, ck_path)
    X.write_train_ids_sidecar(ck_path, XD.keys_of(smp.index))
    (ck_path.parent / "history.json").write_text(json.dumps(hist, default=float))
    log("pretrained", block, fo.g, len(ids), f"{(time.time() - t0) / 60:.1f} min", file_sha256(ck_path)[:12])
    return ck_path


def fit_fold(prep: Prepared, config: str, block: str, fo, S, frozen, *, seeds=(0,), device="cuda",
             pretrain_root: Path, fold_index: int = 0, log=print):
    """Fit configuration ``config`` on fold ``fo`` (training ids only) and predict its test ids.
    Returns ({seed: DataFrame[test ids x block targets]}, diagnostics)."""
    from sentinel.models.bioassess import bioset as BS
    from sentinel.models.bioassess.cv import bioset_rows
    from sentinel.models.bioassess.data import BioDesign

    spec = CONFIGS[config]
    cfg = base_cfg(config)
    Yu = prep.labels(block if spec.get("same_continent") else None)
    view = BioDesign(prep.samples, prep.tokens, Yu, prep.D.info)
    lab = fo.train[Yu.loc[fo.train].notna().any(axis=1).to_numpy()]
    X.assert_fit_ids(XD.keys_of(lab), S, frozen)          # fit + inner-validation ids are a subset of ``lab``
    out, diags = {}, []
    vocab0 = None if spec.get("ukpt") else \
        BS.TaxonVocab.build(prep.tokens[prep.tokens["visit_id"].isin(set(fo.train))])     # per fold (W-MX-9)
    for s in seeds:
        enc, vocab = None, vocab0
        if spec.get("ukpt"):
            ck = pretrain_fold(prep, block, fo, S, frozen, cfg, device, pretrain_root, seed=s, log=log)
            enc = X.load_pretrained_encoder(ck, S, block, fo.g, frozen)
            X.require_guarded(enc, block, fo.g)
            vocab = enc.vocab
        with _ChunkedBuild():
            _, per_seed, diag = bioset_rows(cfg, view, lab, fo.test, vocab=vocab, seeds=(s,), device=device,
                                            geography=True, pretrained=enc, targets=XD.UNION_COLUMNS, fold=fold_index)
        out[s] = to_block(pd.DataFrame(per_seed[s], index=fo.test, columns=list(XD.UNION_COLUMNS)), block)
        diags += diag
    return out, diags
