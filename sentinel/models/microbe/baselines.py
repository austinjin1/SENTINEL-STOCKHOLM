# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Baselines for SENTINEL-Microbe (protocol §6, tiers 0–2). Fixed hyperparameters, one model per
target, every transform fit on the fold's training rows only.

Tier 0 (season + location, no community): ``global_mean``; ``context_ridge`` / ``context_rf`` on
day-of-year harmonics + latitude + longitude + elevation + site type; ``site_mean_doy`` (training
site mean + DOY harmonics fitted to within-site residuals; only defined when the held-out site is in
training, i.e. the random scheme — under domain/site hold-out the site mean does not exist and the
tier-0 comparator is ``context_rf``/``context_ridge``).
Tier 1 (community tables): ``ridge_clr`` (alpha by site-grouped inner CV), ``rf_clr``, ``xgb_clr``,
``rf_pa`` (presence/absence), ``rf_clr_context`` (community + context).
Tier 2: ``tabpfn_clr`` (TabPFN v2.2.1 regressor on the 300 most prevalent training-fold genera).
DNA-encoded tables (SENTINEL candidates, not baselines): ``ridge_seqemb`` / ``rf_seqemb`` on the
relative-abundance-weighted mean of frozen DNABERT-S sequence vectors (+ descriptors) — primer- and
release-agnostic because no OTU id or taxonomy label is used. Pass ``params={"seq_table": (table,
seq_row)}`` (see ``cv.load_seq_table``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.models.microbe import features as F

BASELINES = ("global_mean", "context_ridge", "context_rf", "site_mean_doy", "ridge_clr", "rf_clr", "xgb_clr",
             "rf_pa", "rf_clr_context", "tabpfn_clr", "ridge_seqemb", "rf_seqemb", "rf_clr_tuned", "xgb_clr_tuned",
             "ridge_seqclust", "ridge_seqclust_kmer", "ridge_seqclust_multi", "ridge_clr_seqclust",
             "ridge_v4clr", "xgb_v4clr", "xgb_clr_context", "xgb_v4clr_context")
LABELS = {"global_mean": "GlobalMean", "context_ridge": "Ridge[DOY+location]", "context_rf": "RF[DOY+location]",
          "site_mean_doy": "SiteMean+DOY", "ridge_clr": "Ridge[genus-CLR]", "rf_clr": "RF[genus-CLR]",
          "xgb_clr": "XGB[genus-CLR]", "rf_pa": "RF[genus-PA]", "rf_clr_context": "RF[genus-CLR+DOY+location]",
          "tabpfn_clr": "TabPFNv2[genus-CLR]",
          "ridge_seqemb": "SeqEmb-Ridge[DNABERT-S]", "rf_seqemb": "SeqEmb-RF[DNABERT-S]",
          "rf_clr_tuned": "RF[genus-CLR]-tuned", "xgb_clr_tuned": "XGB[genus-CLR]-tuned",
          "ridge_seqclust": "SeqClust-Ridge[DNABERT-S]", "ridge_seqclust_kmer": "SeqClust-Ridge[kmer5]",
          "ridge_seqclust_multi": "SeqClustMR-Ridge[DNABERT-S]", "ridge_clr_seqclust": "Ridge[genus-CLR+SeqClust]",
          "ridge_v4clr": "V4-Ridge[harmonised-ESV-CLR]", "xgb_v4clr": "V4-XGB[harmonised-ESV-CLR]",
          "xgb_clr_context": "XGB[genus-CLR+DOY+location]", "xgb_v4clr_context": "V4-XGB[harmonised-ESV-CLR+DOY+location]"}
TIER0 = {"global_mean", "context_ridge", "context_rf", "site_mean_doy"}
STOCHASTIC = {"context_rf", "rf_clr", "xgb_clr", "rf_pa", "rf_clr_context", "tabpfn_clr", "rf_seqemb",
              "rf_clr_tuned", "xgb_clr_tuned", "xgb_v4clr",
              "xgb_clr_context", "xgb_v4clr_context"}
# Pre-registered tuning grids (protocol §10, review W6); selected by site-grouped inner CV on training rows.
TUNE_GRIDS = {
    "rf_clr_tuned": [{"min_samples_leaf": a, "max_features": b} for a in (1, 3, 10) for b in (0.1, 0.3, 0.6)],
    "xgb_clr_tuned": [{"max_depth": a, "learning_rate": b} for a in (2, 4, 6) for b in (0.03, 0.1)],
}
NEEDS_GPU = {"tabpfn_clr"}

DEFAULTS = {
    "rf": {"n_estimators": 500, "min_samples_leaf": 3, "max_features": 0.3},
    "xgb": {"n_estimators": 600, "learning_rate": 0.03, "max_depth": 4, "subsample": 0.8,
            "colsample_bytree": 0.3, "min_child_weight": 3, "reg_lambda": 1.0},
    "ridge_alphas": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
    "clr": {"min_prev": 0.05, "max_features": 500, "pseudo": 0.5},
    "tabpfn": {"n_estimators": 8, "max_features": 300, "max_train_rows": 6000},
}


SEQEMB = {"ridge_seqemb": "dnabert_s", "rf_seqemb": "dnabert_s", "ridge_seqclust": "dnabert_s",
          "ridge_seqclust_kmer": "kmer5", "ridge_seqclust_multi": "dnabert_s", "ridge_clr_seqclust": "dnabert_s"}
SEQCLUST = {"ridge_seqclust", "ridge_seqclust_kmer", "ridge_seqclust_multi", "ridge_clr_seqclust"}
SEQCLUST_MULTI_K = (64, 256, 1024)
# Primer-harmonised V4 exact-sequence units (cycle 3): same CLR pipeline as the genus tables, units = V4 window
V4MODELS = {"ridge_v4clr": "ridge_clr", "xgb_v4clr": "xgb_clr", "xgb_v4clr_context": "xgb_clr_context"}
CONTEXT_TABLES = {"rf_clr_context", "xgb_clr_context", "xgb_v4clr_context"}   # community + tier-0 context
CANDIDATE_TABLES = set(SEQEMB) | set(V4MODELS)                      # SENTINEL candidates, not baselines
_V4_CACHE: dict = {}


def v4_tokens(design):
    from sentinel.models.microbe.dna import add_v4_units

    key = (id(design.tokens), len(design.tokens))
    if key not in _V4_CACHE:
        _V4_CACHE.clear()
        _V4_CACHE[key] = add_v4_units(design.tokens, design.sequences, design.samples)
    return _V4_CACHE[key]   # multi-resolution DNA-defined taxa (cycle 3; analogue of genus/family/order)


def seq_cluster_clr(tokens: pd.DataFrame, train_ids, ids_list, table: np.ndarray, seq_row: dict, *, k: int = 512,
                    n_pca: int = 64, min_prev: float = 0.05, seed: int = 0):
    """DNA-defined taxa (improvement cycle 1): PCA + k-means on the frozen vectors of the sequences seen
    in TRAINING rows (label-free), every sequence -> nearest centroid, per-sample cluster counts ->
    training-fold prevalence filter -> CLR (pseudocount 0.5, other pooled). Primer/release-agnostic:
    a sequence from another amplicon region is binned by its embedding, not by an id or a taxonomy label."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA

    tr_tok = tokens[tokens["sample_id"].isin(pd.Index(train_ids))]
    rows_tr = np.unique(tr_tok["seq_id"].map(seq_row).dropna().to_numpy(int))
    n_pca = min(n_pca, len(rows_tr) - 1, table.shape[1])
    k = min(k, len(rows_tr))
    pca = PCA(n_pca, random_state=seed).fit(table[rows_tr])
    km = MiniBatchKMeans(k, random_state=seed, n_init=3, batch_size=4096).fit(pca.transform(table[rows_tr]))
    all_rows = np.unique(tokens["seq_id"].map(seq_row).dropna().to_numpy(int))
    lab = np.full(table.shape[0], -1)
    lab[all_rows] = km.predict(pca.transform(table[all_rows]))
    t = tokens.assign(_c=lab[tokens["seq_id"].map(seq_row).fillna(0).to_numpy(int)])
    C = t.groupby(["sample_id", "_c"])["count"].sum().unstack("_c", fill_value=0).astype(float)
    Ctr = C.reindex(pd.Index(train_ids)).fillna(0.0)
    prev = (Ctr > 0).mean(0)
    keep = prev[prev >= min_prev].index
    out = []
    for ids in ids_list:
        X = C.reindex(pd.Index(ids)).fillna(0.0)
        kept = X.reindex(columns=keep, fill_value=0.0)
        other = X.sum(1) - kept.sum(1)
        out.append(F.clr(np.column_stack([kept.to_numpy(), other.to_numpy()])).astype(np.float32))
    return out


_SE_CACHE: dict = {}


def seq_embedding_mean(tokens: pd.DataFrame, ids, table: np.ndarray, seq_row: dict) -> np.ndarray:
    """Relative-abundance-weighted mean of frozen sequence vectors per sample (label-free, row-wise;
    computed once for all samples of ``tokens`` with a sparse product and cached)."""
    from scipy import sparse

    key = (id(tokens), len(tokens), id(table))
    if key not in _SE_CACHE:
        samples = pd.Index(pd.unique(tokens["sample_id"]))
        r = samples.get_indexer(tokens["sample_id"])
        w = tokens["count"].to_numpy(float)
        w = w / np.bincount(r, w, minlength=len(samples))[r]
        rows = tokens["seq_id"].map(seq_row).fillna(0).to_numpy(int)
        W = sparse.csr_matrix((w, (r, rows)), shape=(len(samples), table.shape[0]))
        _SE_CACHE.clear()
        _SE_CACHE[key] = (samples, np.asarray(W @ table, np.float32))
    samples, E = _SE_CACHE[key]
    return E[samples.get_indexer(pd.Index(ids))]


def _rf(seed, n_jobs, p):
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(n_estimators=p["n_estimators"], min_samples_leaf=p["min_samples_leaf"],
                                 max_features=p["max_features"], random_state=seed, n_jobs=n_jobs)


def _ridge_grouped(X, y, groups, alphas):
    """Ridge with alpha chosen by leave-sites-out inner CV (GroupKFold, <= 5 folds) on training rows."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    k = min(5, len(np.unique(groups)))
    best_a, best = alphas[0], np.inf
    if k >= 2:
        for a in alphas:
            err = 0.0
            for tr, va in GroupKFold(k).split(Xs, y, groups):
                m = Ridge(alpha=a).fit(Xs[tr], y[tr])
                err += float(np.sum((m.predict(Xs[va]) - y[va]) ** 2))
            if err < best:
                best, best_a = err, a
    m = Ridge(alpha=best_a).fit(Xs, y)
    return lambda Z: m.predict(sc.transform(Z)), best_a


def _make(name, cfg, seed, n_jobs, p):
    if name == "rf_clr_tuned":
        return _rf(seed, n_jobs, {**p["rf"], **cfg})
    import xgboost as xgb

    return xgb.XGBRegressor(**{**p["xgb"], **cfg}, random_state=seed, n_jobs=n_jobs, tree_method="hist")


def _tuned_fit_predict(name, X, y, groups, Xte, seed, n_jobs, p):
    """Grid (TUNE_GRIDS) chosen by 3-fold site-grouped inner CV MSE on the training rows, then refit."""
    from sklearn.model_selection import GroupKFold

    k = min(3, len(np.unique(groups)))
    best, best_cfg = np.inf, TUNE_GRIDS[name][0]
    if k >= 2:
        for cfg in TUNE_GRIDS[name]:
            err = 0.0
            for tr, va in GroupKFold(k).split(X, y, groups):
                # inner CV with a reduced ensemble (200 trees / 300 rounds) for cost; refit at full size
                small = {"n_estimators": 200 if name == "rf_clr_tuned" else 300}
                m = _make(name, {**cfg, **small}, seed, n_jobs, p).fit(X[tr], y[tr])
                err += float(np.sum((m.predict(X[va]) - y[va]) ** 2))
            if err < best:
                best, best_cfg = err, cfg
    return _make(name, best_cfg, seed, n_jobs, p).fit(X, y).predict(Xte)


def _doy_harm(dates):
    doy = pd.to_datetime(pd.Series(dates)).dt.dayofyear.to_numpy(float)
    a = 2 * np.pi * (doy - 1) / 365.25
    return np.column_stack([np.sin(a), np.cos(a), np.sin(2 * a), np.cos(2 * a)])


def fit_predict(name: str, design, train_ids, test_ids, *, targets, seed: int = 0, n_jobs: int = 8,
                params: dict | None = None, device: str = "cpu") -> pd.DataFrame:
    p = {**DEFAULTS, **(params or {})}
    S = design.samples
    train_ids = np.asarray(train_ids)
    test_ids = np.asarray(test_ids)
    out = pd.DataFrame(np.nan, index=pd.Index(test_ids, name="sample_id"), columns=list(targets))
    Ytr = S.loc[train_ids, list(targets)].astype(float)

    if name == "global_mean":
        for t in targets:
            out[t] = float(Ytr[t].mean())
        return out

    if name == "site_mean_doy":
        Htr, Hte = _doy_harm(S.loc[train_ids, "collect_date"]), _doy_harm(S.loc[test_ids, "collect_date"])
        site_tr = S.loc[train_ids, "site_id"].astype(str).to_numpy()
        site_te = S.loc[test_ids, "site_id"].astype(str).to_numpy()
        for t in targets:
            y = Ytr[t].to_numpy()
            ok = np.isfinite(y)
            if ok.sum() < 3:
                continue
            sm = pd.Series(y[ok]).groupby(site_tr[ok]).mean()
            resid = y[ok] - sm.reindex(site_tr[ok]).to_numpy()
            beta, *_ = np.linalg.lstsq(np.column_stack([Htr[ok], np.ones(ok.sum())]), resid, rcond=None)
            base = sm.reindex(site_te).to_numpy()          # NaN for sites absent from training
            out[t] = base + np.column_stack([Hte, np.ones(len(test_ids))]) @ beta
        return out

    # ---- feature matrices
    if name in ("context_ridge", "context_rf"):
        Xtr, fill = F.context_features(S, train_ids)
        Xte, _ = F.context_features(S, test_ids, fill)
    elif name in SEQEMB:
        table, seq_row = p["seq_table"]
        if name == "ridge_seqclust_multi":
            parts = [seq_cluster_clr(design.tokens, train_ids, [train_ids, test_ids], table, seq_row, k=k, seed=seed)
                     for k in SEQCLUST_MULTI_K]
            Xtr, Xte = np.column_stack([q[0] for q in parts]), np.column_stack([q[1] for q in parts])
        elif name in SEQCLUST:
            Xtr, Xte = seq_cluster_clr(design.tokens, train_ids, [train_ids, test_ids], table, seq_row,
                                       k=int(p.get("seqclust_k", 512)), seed=seed)
            if name == "ridge_clr_seqclust":    # hybrid: RDP genus CLR (+ "other") next to the DNA-defined units
                clr = F.GenusCLR(**p["clr"]).fit(design.tokens, train_ids)
                Xtr = np.column_stack([clr.transform(design.tokens, train_ids, "clr"), Xtr])
                Xte = np.column_stack([clr.transform(design.tokens, test_ids, "clr"), Xte])
        else:
            Xtr = seq_embedding_mean(design.tokens, train_ids, table, seq_row)
            Xte = seq_embedding_mean(design.tokens, test_ids, table, seq_row)
        Dtr = F.sample_descriptors(S, train_ids, primer_onehot=True, enabled=p.get("descriptors", True))
        Dte = F.sample_descriptors(S, test_ids, primer_onehot=True, enabled=p.get("descriptors", True))
        Xtr, Xte = np.column_stack([Xtr, Dtr]), np.column_stack([Xte, Dte])
    else:
        clr_p = dict(p["clr"])
        toks = design.tokens
        if name in V4MODELS:
            toks, clr_p["rank"] = v4_tokens(design), "v4"
        if name == "tabpfn_clr":     # 300 most prevalent genera, the rest pooled into "other" (review O3)
            clr_p["max_features"] = int(p["tabpfn"]["max_features"])
        clr = F.GenusCLR(**clr_p).fit(toks, train_ids)
        kind = "pa" if name == "rf_pa" else "clr"
        Xtr = clr.transform(toks, train_ids, kind)
        Xte = clr.transform(toks, test_ids, kind)
        if name in CONTEXT_TABLES:
            Ctr, fill = F.context_features(S, train_ids)
            Cte, _ = F.context_features(S, test_ids, fill)
            Xtr, Xte = np.column_stack([Xtr, Ctr]), np.column_stack([Xte, Cte])
        if name != "rf_pa":
            Dtr = F.sample_descriptors(S, train_ids, primer_onehot=True, enabled=p.get("descriptors", True))
            Dte = F.sample_descriptors(S, test_ids, primer_onehot=True, enabled=p.get("descriptors", True))
            Xtr, Xte = np.column_stack([Xtr, Dtr]), np.column_stack([Xte, Dte])
    groups = S.loc[train_ids, "site_id"].astype(str).to_numpy()

    for t in targets:
        y = Ytr[t].to_numpy()
        ok = np.isfinite(y)
        if ok.sum() < 10:
            continue
        if name in ("context_ridge", "ridge_clr", "ridge_seqemb", "ridge_v4clr") or name in SEQCLUST:
            f, _ = _ridge_grouped(Xtr[ok], y[ok], groups[ok], p["ridge_alphas"])
            out[t] = f(Xte)
        elif name in ("context_rf", "rf_clr", "rf_pa", "rf_clr_context", "rf_seqemb"):
            out[t] = _rf(seed, n_jobs, p["rf"]).fit(Xtr[ok], y[ok]).predict(Xte)
        elif name in TUNE_GRIDS:
            out[t] = _tuned_fit_predict(name, Xtr[ok], y[ok], groups[ok], Xte, seed, n_jobs, p)
        elif name in ("xgb_clr", "xgb_v4clr", "xgb_clr_context", "xgb_v4clr_context"):
            import xgboost as xgb

            m = xgb.XGBRegressor(**p["xgb"], random_state=seed, n_jobs=n_jobs, tree_method="hist")
            out[t] = m.fit(Xtr[ok], y[ok]).predict(Xte)
        elif name == "tabpfn_clr":
            from sentinel.models.chemtox.tabpfn_model import load_tabpfn

            TabPFNRegressor, _ = load_tabpfn()
            idx = np.flatnonzero(ok)
            mx = int(p["tabpfn"]["max_train_rows"])
            if idx.size > mx:
                idx = np.random.default_rng(seed).choice(idx, mx, replace=False)
            m = TabPFNRegressor(n_estimators=int(p["tabpfn"]["n_estimators"]), random_state=seed, device=device,
                                ignore_pretraining_limits=True)
            m.fit(Xtr[idx], y[idx])
            out[t] = np.concatenate([m.predict(Xte[s:s + 2000]) for s in range(0, len(Xte), 2000)])
        else:
            raise ValueError(f"unknown baseline {name}")
    return out
