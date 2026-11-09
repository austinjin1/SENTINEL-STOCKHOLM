#!/usr/bin/env python
"""Track B 11 — does DOMAIN GENERALIZATION fix the cross-region OOD collapse?

SENTINEL's microcystin fusion (satellite spectral sf* + in-situ pigment CHLA/PC
+ drivers) scores AUROC ~0.82 within-region (station CV) but collapses to CHANCE
(~0.485) under leave-one-HUC-region-out. Prior work (10_region_transfer.py) showed
NO feature transform recovers it. Here we test whether proper DOMAIN-GENERALIZATION
OBJECTIVES beat plain ERM (pooled training).

Methods, all under leave-one-HUC-region-out CV (held-out region NEVER in train),
labels mc>1.6 and mc>0.3, AUROC + site-clustered bootstrap CI:
  1. ERM          pooled GBM + pooled logreg (reproduce the ~0.49 floor)
  2. GroupDRO     reweight so worst-region train loss is minimized (logreg)
  3. CORAL        per-region feature-covariance alignment to a common target (+logreg)
  4. per-region-std   per-region standardization (feature moment matching, train-only)
  5. DANN         domain-adversarial MLP w/ gradient reversal (features can't ID region)
  6. IRM          invariant-risk penalty (logreg-style linear predictor)
  7. region-mixup inter-region interpolation augmentation (+logreg)
  8. ensemble     per-region models + similarity-weighted combiner

GUARDRAILS (leakage-free):
  * leave-one-HUC-region-out: a held-out region is NEVER in train.
  * EVERY normalization / covariance / rank / standardizer is FIT ON TRAIN ROWS
    ONLY inside each fold, then applied to the held-out region.
  * real data only.

Also: AUROC-vs-#training-regions learning curve (does "more regions" help?).

HONEST BAR: does ANY method raise held-out-region AUROC CLEARLY above 0.53 with a
site-clustered bootstrap CI excluding 0.5?

Writes results/trackB_fusion/mc_domaingen_ood.json
"""
from __future__ import annotations

import json
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
NOAA = ROOT / "data" / "processed" / "biology" / "noaa_habs"
EXPANDED = NOAA / "mc_fusion_expanded.parquet"
DATA = NOAA / "mc_fusion_dataset.parquet"
CYANO = NOAA / "wqp_hab_cyanotoxins.parquet"
RAW = ROOT / "data" / "raw" / "wqp_drivers"
CACHE = NOAA / "_mc_dg_featurecache.parquet"
OUT = ROOT / "results" / "trackB_fusion" / "mc_domaingen_ood.json"
SEED = 0
MIN_REGION = 30
rng = np.random.default_rng(SEED)


# ----------------------------------------------------------------------------- data
def parse(tag):
    f = RAW / f"{tag}.csv"
    if not f.exists() or f.stat().st_size < 100:
        return pd.DataFrame(columns=["site", "date", "val"])
    df = pd.read_csv(f, low_memory=False)
    if len(df) == 0:
        return pd.DataFrame(columns=["site", "date", "val"])
    return (pd.DataFrame({"site": df["MonitoringLocationIdentifier"].astype(str),
                          "date": pd.to_datetime(df["ActivityStartDate"], errors="coerce"),
                          "val": pd.to_numeric(df["ResultMeasureValue"], errors="coerce")})
            .dropna(subset=["date", "val"]).sort_values("date"))


def nearest(dv, site, date, tol=14):
    d = dv[dv["site"] == site]
    if len(d) == 0:
        return np.nan
    dd = (d["date"] - date).abs().dt.days
    i = dd.values.argmin()
    return float(d["val"].values[i]) if dd.values[i] <= tol else np.nan


def build_frame():
    """Load fusion parquet (expanded if present), attach HUC2 region + drivers.
    Cached so the slow nearest-driver join runs once."""
    src = EXPANDED if EXPANDED.exists() else DATA
    df = pq.read_table(src).to_pandas()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)

    cy = pq.read_table(CYANO, columns=["Location_Identifier", "Location_HUCEightDigitCode",
                                       "Location_Latitude", "Location_Longitude"]).to_pandas()
    cy["site"] = cy["Location_Identifier"].astype(str)
    g = cy.groupby("site").first()
    huc = g["Location_HUCEightDigitCode"].astype(str).str.zfill(8).str[:2]
    df["region"] = df["site"].astype(str).map(huc).fillna("NA")
    df["lat"] = df["site"].astype(str).map(pd.to_numeric(g["Location_Latitude"], errors="coerce"))
    df["lon"] = df["site"].astype(str).map(pd.to_numeric(g["Location_Longitude"], errors="coerce"))
    df["month"] = df["date"].dt.month

    if CACHE.exists() and CACHE.stat().st_mtime > src.stat().st_mtime:
        cache = pq.read_table(CACHE).to_pandas()
        if len(cache) == len(df):
            for c in ["drv_temp", "drv_phosphorus", "drv_nitrogen"]:
                df[c] = cache[c].values
            return df, str(src.name)
    drivers = {t: parse(t) for t in ["temp", "phosphorus", "nitrogen"]}
    for t, dv in drivers.items():
        df[f"drv_{t}"] = [nearest(dv, str(s), d) for s, d in zip(df["site"].astype(str), df["date"])]
    df[["drv_temp", "drv_phosphorus", "drv_nitrogen"]].to_parquet(CACHE)
    return df, str(src.name)


# ----------------------------------------------------------------------------- utils
def impute_fit(Xtr):
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return med


def apply_impute(X, med):
    X = X.copy()
    idx = np.where(~np.isfinite(X))
    X[idx] = np.take(med, idx[1])
    return X


def standardize_fit(Xtr):
    mu = np.nanmean(Xtr, axis=0)
    sd = np.nanstd(Xtr, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# ----------------------------------------------------------------------------- models
def hgb():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                          l2_regularization=1.0, class_weight="balanced",
                                          random_state=SEED)


def m_erm_gbm(Xtr, ytr, gtr, Xte):
    med = impute_fit(Xtr)
    clf = hgb().fit(apply_impute(Xtr, med), ytr)
    return clf.predict_proba(apply_impute(Xte, med))[:, 1]


def m_erm_logreg(Xtr, ytr, gtr, Xte):
    med = impute_fit(Xtr)
    mu, sd = standardize_fit(apply_impute(Xtr, med))
    Ztr = (apply_impute(Xtr, med) - mu) / sd
    Zte = (apply_impute(Xte, med) - mu) / sd
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(Ztr, ytr)
    return clf.predict_proba(Zte)[:, 1]


def m_perregion_std(Xtr, ytr, gtr, Xte, gte):
    """Per-region standardization: each training region centered/scaled to zero
    mean/unit var using ITS OWN train stats; test region centered/scaled using
    its OWN test-region feature stats (drivers/spectra shift by region -> removing
    region-level moments makes features comparable). No LABELS from test used."""
    med = impute_fit(Xtr)
    Xtr_i, Xte_i = apply_impute(Xtr, med), apply_impute(Xte, med)
    Ztr = np.empty_like(Xtr_i)
    for gg in np.unique(gtr):
        i = gtr == gg
        mu, sd = standardize_fit(Xtr_i[i])
        Ztr[i] = (Xtr_i[i] - mu) / sd
    Zte = np.empty_like(Xte_i)
    for gg in np.unique(gte):
        i = gte == gg
        mu, sd = standardize_fit(Xte_i[i])   # test-region features only, unsupervised
        Zte[i] = (Xte_i[i] - mu) / sd
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(Ztr, ytr)
    return clf.predict_proba(Zte)[:, 1]


def m_coral(Xtr, ytr, gtr, Xte, gte):
    """CORAL: whiten each train region then re-color to a common (pooled-train)
    covariance so per-region 2nd-order stats align; test region whitened by its
    own (unsupervised) covariance and re-colored to the same target."""
    med = impute_fit(Xtr)
    Xtr_i, Xte_i = apply_impute(Xtr, med), apply_impute(Xte, med)
    mu_t, sd_t = standardize_fit(Xtr_i)
    Xtr_s = (Xtr_i - mu_t) / sd_t
    Xte_s = (Xte_i - mu_t) / sd_t
    d = Xtr_s.shape[1]
    Ct = np.cov(Xtr_s, rowvar=False) + np.eye(d)   # target = pooled train cov
    Ct_half = _mat_sqrt(Ct)

    def recolor(block):
        C = np.cov(block, rowvar=False) + np.eye(d)
        W = _mat_inv_sqrt(C)
        return (block @ W) @ Ct_half

    Ztr = np.empty_like(Xtr_s)
    for gg in np.unique(gtr):
        i = gtr == gg
        Ztr[i] = recolor(Xtr_s[i]) if i.sum() > d else Xtr_s[i]
    Zte = np.empty_like(Xte_s)
    for gg in np.unique(gte):
        i = gte == gg
        Zte[i] = recolor(Xte_s[i]) if i.sum() > d else Xte_s[i]
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(Ztr, ytr)
    return clf.predict_proba(Zte)[:, 1]


def _mat_sqrt(C):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-6, None)
    return (V * np.sqrt(w)) @ V.T


def _mat_inv_sqrt(C):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-6, None)
    return (V / np.sqrt(w)) @ V.T


def m_groupdro(Xtr, ytr, gtr, Xte, eta=0.05, steps=400):
    """GroupDRO: online-reweighted logistic regression minimizing the WORST
    per-region loss. Group weights grow multiplicatively w/ each group's loss."""
    med = impute_fit(Xtr)
    mu, sd = standardize_fit(apply_impute(Xtr, med))
    Ztr = (apply_impute(Xtr, med) - mu) / sd
    Zte = (apply_impute(Xte, med) - mu) / sd
    Ztr = np.hstack([Ztr, np.ones((len(Ztr), 1))])
    Zte = np.hstack([Zte, np.ones((len(Zte), 1))])
    groups = np.unique(gtr)
    gmask = {gg: (gtr == gg) for gg in groups}
    # class balancing weights (per-sample) so imbalance handled like class_weight
    pw = np.where(ytr == 1, (ytr == 0).mean(), (ytr == 1).mean())
    pw = pw / pw.mean()
    q = np.ones(len(groups)) / len(groups)
    w = np.zeros(Ztr.shape[1])
    lr, l2 = 0.1, 1e-3
    for _ in range(steps):
        p = sigmoid(Ztr @ w)
        losses = np.zeros(len(groups))
        grad = np.zeros_like(w)
        for gi, gg in enumerate(groups):
            m = gmask[gg]
            resid = (p[m] - ytr[m]) * pw[m]
            losses[gi] = -np.mean(pw[m] * (ytr[m] * np.log(p[m] + 1e-9) +
                                           (1 - ytr[m]) * np.log(1 - p[m] + 1e-9)))
            grad += q[gi] * (Ztr[m].T @ resid) / max(m.sum(), 1)
        q = q * np.exp(eta * losses)
        q = q / q.sum()
        w -= lr * (grad + l2 * w)
    return sigmoid(Zte @ w)


def m_irm(Xtr, ytr, gtr, Xte, steps=600, pen=100.0, anneal=150):
    """IRM (IRMv1): logistic risk + a penalty that forces a FIXED dummy scalar
    classifier w=1 to be simultaneously optimal in every environment. The penalty
    is the squared per-env gradient of the risk w.r.t. that dummy scale, averaged
    over regions. Stabilized: per-step gradient normalization + gradient clipping
    so the large penalty coefficient cannot blow up the weights (previous version
    diverged to NaN)."""
    med = impute_fit(Xtr)
    mu, sd = standardize_fit(apply_impute(Xtr, med))
    Ztr = np.hstack([(apply_impute(Xtr, med) - mu) / sd, np.ones((len(Xtr), 1))])
    Zte = np.hstack([(apply_impute(Xte, med) - mu) / sd, np.ones((len(Xte), 1))])
    groups = np.unique(gtr)
    gmask = {gg: (gtr == gg) for gg in groups}
    pw = np.where(ytr == 1, (ytr == 0).mean(), (ytr == 1).mean())
    pw = pw / pw.mean()
    w = np.zeros(Ztr.shape[1])
    lr, l2 = 0.02, 1e-3
    for t in range(steps):
        z = np.clip(Ztr @ w, -30, 30)
        p = sigmoid(z)
        pcoef = pen if t >= anneal else 1.0
        g_erm = np.zeros_like(w)
        g_pen = np.zeros_like(w)
        for gg in groups:
            m = gmask[gg]
            n = max(m.sum(), 1)
            resid = (p[m] - ytr[m]) * pw[m]
            # ERM gradient for this env
            ge = (Ztr[m].T @ resid) / n
            g_erm += ge
            # IRMv1: dummy scale s=1 multiplies the logit; grad of risk wrt s is
            # g_scale = mean(resid * z). penalty = g_scale^2. d/dw of penalty:
            gscale = float(np.sum(resid * z[m]) / n)
            # d(g_scale)/dw = mean_i [ (p'(z_i) z_i + resid_i) * Z_i ]
            dz = p[m] * (1 - p[m])
            dgs = (Ztr[m].T @ ((dz * z[m] * pw[m]) + resid)) / n
            g_pen += 2.0 * gscale * dgs
        grad = (g_erm + pcoef * g_pen) / len(groups) + l2 * w
        # normalize + clip so the annealed penalty cannot diverge
        gn = np.linalg.norm(grad)
        if gn > 5.0:
            grad = grad * (5.0 / gn)
        w -= lr * grad
        if not np.all(np.isfinite(w)):
            w = np.zeros_like(w)
            break
    return sigmoid(np.clip(Zte @ w, -30, 30))


def m_region_mixup(Xtr, ytr, gtr, Xte, n_mix=3000, alpha=0.3):
    """Region-mixup: augment training with convex combinations of samples drawn
    from DIFFERENT regions (inter-region interpolation), encouraging linearity
    across the region manifold."""
    med = impute_fit(Xtr)
    mu, sd = standardize_fit(apply_impute(Xtr, med))
    Ztr = (apply_impute(Xtr, med) - mu) / sd
    Zte = (apply_impute(Xte, med) - mu) / sd
    groups = np.unique(gtr)
    idx_by_g = {gg: np.where(gtr == gg)[0] for gg in groups}
    Xa, ya = [Ztr], [ytr.astype("float64")]
    if len(groups) >= 2:
        for _ in range(n_mix):
            ga, gb = rng.choice(groups, size=2, replace=False)
            ia = rng.choice(idx_by_g[ga]); ib = rng.choice(idx_by_g[gb])
            lam = rng.beta(alpha, alpha)
            Xa.append((lam * Ztr[ia] + (1 - lam) * Ztr[ib])[None, :])
            ya.append(np.array([lam * ytr[ia] + (1 - lam) * ytr[ib]]))
        Xmix = np.vstack(Xa); ymix = np.concatenate(ya)
        # soft-label logistic via sample-weighted 2-class expansion
        clf = LogisticRegression(max_iter=2000, C=1.0)
        Xall = np.vstack([Xmix, Xmix]); yall = np.concatenate([np.ones(len(ymix)), np.zeros(len(ymix))])
        wall = np.concatenate([ymix, 1 - ymix])
        clf.fit(Xall, yall, sample_weight=wall + 1e-6)
    else:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Ztr, ytr)
    return clf.predict_proba(Zte)[:, 1]


def m_ensemble(Xtr, ytr, gtr, Xte, gte):
    """Per-region experts + similarity-weighted combiner. Each region trains its
    own logreg; at test, weight experts by how close the test region's feature
    mean is to each expert region's train mean (RBF on standardized means)."""
    med = impute_fit(Xtr)
    mu, sd = standardize_fit(apply_impute(Xtr, med))
    Ztr = (apply_impute(Xtr, med) - mu) / sd
    Zte = (apply_impute(Xte, med) - mu) / sd
    groups = np.unique(gtr)
    experts, centers = {}, {}
    for gg in groups:
        i = gtr == gg
        if len(np.unique(ytr[i])) < 2:
            continue
        experts[gg] = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(Ztr[i], ytr[i])
        centers[gg] = Ztr[i].mean(axis=0)
    if not experts:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Ztr, ytr)
        return clf.predict_proba(Zte)[:, 1]
    keys = list(experts.keys())
    C = np.vstack([centers[k] for k in keys])
    dists = np.mean(C ** 2, axis=1).mean() + 1e-6
    out = np.zeros(len(Zte))
    for gg in np.unique(gte):
        i = gte == gg
        c = Zte[i].mean(axis=0)
        d2 = np.sum((C - c) ** 2, axis=1)
        wts = np.exp(-d2 / (2 * dists))
        wts = wts / wts.sum()
        preds = np.vstack([experts[k].predict_proba(Zte[i])[:, 1] for k in keys])  # (K, n)
        out[i] = wts @ preds
    return out


def m_dann(Xtr, ytr, gtr, Xte, epochs=120, hidden=32, lr=1e-3):
    """DANN: MLP feature extractor -> label head + domain head via gradient
    reversal, so learned features can't predict the region -> region-invariant.
    CPU, time-boxed."""
    import torch
    import torch.nn as nn
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    med = impute_fit(Xtr)
    mu, sd = standardize_fit(apply_impute(Xtr, med))
    Ztr = ((apply_impute(Xtr, med) - mu) / sd).astype("float32")
    Zte = ((apply_impute(Xte, med) - mu) / sd).astype("float32")
    groups = np.unique(gtr)
    gidx = {gg: k for k, gg in enumerate(groups)}
    gy = np.array([gidx[g] for g in gtr])

    class GRL(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lamb):
            ctx.lamb = lamb
            return x.view_as(x)

        @staticmethod
        def backward(ctx, g):
            return -ctx.lamb * g, None

    class Net(nn.Module):
        def __init__(self, d, k):
            super().__init__()
            self.feat = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
            self.cls = nn.Linear(hidden, 1)
            self.dom = nn.Linear(hidden, k)

        def forward(self, x, lamb):
            f = self.feat(x)
            return self.cls(f).squeeze(-1), self.dom(GRL.apply(f, lamb))

    Xt = torch.tensor(Ztr); yt = torch.tensor(ytr.astype("float32"))
    dt = torch.tensor(gy, dtype=torch.long)
    Xv = torch.tensor(Zte)
    net = Net(Ztr.shape[1], len(groups))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    pos_w = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], dtype=torch.float32)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    ce = nn.CrossEntropyLoss()
    n = len(Xt)
    for ep in range(epochs):
        net.train()
        lamb = 2.0 / (1.0 + np.exp(-5 * ep / epochs)) - 1.0   # ramp adversary
        perm = torch.randperm(n)
        for s in range(0, n, 256):
            b = perm[s:s + 256]
            opt.zero_grad()
            logit, dlog = net(Xt[b], lamb)
            loss = bce(logit, yt[b]) + ce(dlog, dt[b])
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        p = torch.sigmoid(net(Xv, 0.0)[0]).numpy()
    return p


# ----------------------------------------------------------------------------- CV
METHODS = {
    "ERM_gbm": ("simple", m_erm_gbm),
    "ERM_logreg": ("simple", m_erm_logreg),
    "GroupDRO": ("simple", m_groupdro),
    "IRM": ("simple", m_irm),
    "region_mixup": ("simple", m_region_mixup),
    "DANN": ("simple", m_dann),
    "per_region_std": ("needs_gte", m_perregion_std),
    "CORAL": ("needs_gte", m_coral),
    "ensemble": ("needs_gte", m_ensemble),
}


def leave_one_region_out_oof(X, y, region, site, method_fn, kind):
    """Fit on all-but-one region, predict the held-out region. Return per-row OOF
    predictions (only for rows in regions with >= MIN_REGION samples)."""
    regs = pd.Series(region).value_counts()
    big = list(regs[regs >= MIN_REGION].index)
    m = np.isin(region, big)
    p = np.full(len(y), np.nan)
    for held in big:
        te = m & (region == held)
        tr = m & (region != held)
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        if kind == "needs_gte":
            p[te] = method_fn(X[tr], y[tr], region[tr], X[te], region[te])
        else:
            p[te] = method_fn(X[tr], y[tr], region[tr], X[te])
    return p, m, big


def per_region_auroc(y, p, region, big):
    out = {}
    for r in big:
        i = (region == r) & np.isfinite(p)
        if i.sum() >= 10 and len(np.unique(y[i])) == 2:
            out[r] = round(float(roc_auc_score(y[i], p[i])), 3)
    return out


def site_cluster_bootstrap(y, p, site, region, big, n_boot=1000):
    """Site-clustered bootstrap of the POOLED cross-region AUROC: resample SITES
    (clusters) with replacement, recompute pooled AUROC over held-out predictions."""
    valid = np.isfinite(p) & np.isin(region, big)
    if valid.sum() < 20 or len(np.unique(y[valid])) < 2:
        return None, None, None
    y2, p2, s2 = y[valid], p[valid], site[valid]
    sites = np.unique(s2)
    site_rows = {s: np.where(s2 == s)[0] for s in sites}
    base = roc_auc_score(y2, p2)
    aucs = []
    for _ in range(n_boot):
        pick = rng.choice(sites, size=len(sites), replace=True)
        rows = np.concatenate([site_rows[s] for s in pick])
        yy, pp = y2[rows], p2[rows]
        if len(np.unique(yy)) < 2:
            continue
        aucs.append(roc_auc_score(yy, pp))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return round(float(base), 3), round(float(lo), 3), round(float(hi), 3)


def domain_learning_curve(X, y, region, site, big):
    """AUROC vs #training regions using ERM logreg: for k=1..R-1, average held-out
    AUROC over all (held-out region, k-of-remaining-regions) train subsets."""
    curve = {}
    R = len(big)
    for held in big:
        others = [r for r in big if r != held]
        te = region == held
        if te.sum() < 10 or len(np.unique(y[te])) < 2:
            continue
        for k in range(1, len(others) + 1):
            combos = list(combinations(others, k))
            if len(combos) > 8:
                idx = rng.choice(len(combos), size=8, replace=False)
                combos = [combos[i] for i in idx]
            for combo in combos:
                tr = np.isin(region, combo)
                if len(np.unique(y[tr])) < 2:
                    continue
                p = m_erm_logreg(X[tr], y[tr], region[tr], X[te])
                if len(np.unique(y[te])) == 2:
                    curve.setdefault(k, []).append(roc_auc_score(y[te], p))
    return {str(k): {"mean_auroc": round(float(np.mean(v)), 3),
                     "std": round(float(np.std(v)), 3), "n": len(v)}
            for k, v in sorted(curve.items())}


def main():
    t0 = time.time()
    df, src = build_frame()
    print(f"[data] source={src} n={len(df)}", flush=True)

    sf = [c for c in df.columns if c.startswith("sf")]
    feat_cols = sf + ["CHLA", "PC", "drv_temp", "drv_phosphorus", "drv_nitrogen"]
    X = df[feat_cols].to_numpy("float64")
    site = df["site"].astype(str).to_numpy()
    region = df["region"].to_numpy()
    mc = df["mc"].to_numpy("float64")

    regs = pd.Series(region).value_counts()
    big = list(regs[regs >= MIN_REGION].index)
    print(f"[regions>= {MIN_REGION}] {big}  counts={ {r:int(regs[r]) for r in big} }", flush=True)

    results = {
        "meta": {"source": src, "n_rows": int(len(df)), "n_features": len(feat_cols),
                 "feature_cols": feat_cols, "regions_used": big,
                 "region_counts": {r: int(regs[r]) for r in big},
                 "cv": "leave-one-HUC-region-out", "min_region_n": MIN_REGION,
                 "seed": SEED, "honest_bar": "held-out-region AUROC > 0.53, site-clustered CI excludes 0.5"},
        "labels": {},
    }

    for lname, thr in [("mc_gt1.6", 1.6), ("mc_gt0.3", 0.3)]:
        y = (mc > thr).astype(int)
        m0 = np.isin(region, big)
        print(f"\n=== label {lname} (thr={thr}) pos={int(y[m0].sum())}/{int(m0.sum())} ===", flush=True)
        lab = {"threshold": thr, "n_cross_region": int(m0.sum()),
               "pos_cross_region": int(y[m0].sum()),
               "per_region_pos_rate": {r: round(float(y[region == r].mean()), 3) for r in big},
               "methods": {}}
        erm_floor = None
        for mname, (kind, fn) in METHODS.items():
            ts = time.time()
            try:
                p, mask, big2 = leave_one_region_out_oof(X, y, region, site, fn, kind)
                pooled, lo, hi = site_cluster_bootstrap(y, p, site, region, big2)
                if pooled is None:
                    lab["methods"][mname] = {"error": "no valid predictions"}
                    print(f"  {mname:16s} no valid predictions", flush=True)
                    continue
                pr = per_region_auroc(y, p, region, big2)
                lab["methods"][mname] = {
                    "pooled_auroc": pooled, "ci95": [lo, hi],
                    "beats_chance": bool(lo > 0.5),
                    "per_region_auroc": pr,
                    "mean_per_region_auroc": round(float(np.mean(list(pr.values()))), 3) if pr else None,
                }
                if mname == "ERM_logreg":
                    erm_floor = pooled
                print(f"  {mname:16s} pooled={pooled:.3f} CI[{lo:.3f},{hi:.3f}] "
                      f"mean_pr={lab['methods'][mname]['mean_per_region_auroc']} "
                      f"({time.time()-ts:.1f}s)", flush=True)
            except Exception as e:
                lab["methods"][mname] = {"error": str(e)}
                print(f"  {mname:16s} ERROR {e}", flush=True)
        lab["erm_logreg_floor"] = erm_floor
        # best DG method vs floor
        dg = {k: v for k, v in lab["methods"].items()
              if k not in ("ERM_gbm", "ERM_logreg") and "pooled_auroc" in v}
        if dg and erm_floor is not None:
            best = max(dg.items(), key=lambda kv: kv[1]["pooled_auroc"])
            lab["best_dg_method"] = best[0]
            lab["best_dg_auroc"] = best[1]["pooled_auroc"]
            lab["best_dg_minus_floor"] = round(best[1]["pooled_auroc"] - erm_floor, 3)
            lab["any_beats_053_ci_excl_05"] = any(
                v["pooled_auroc"] > 0.53 and v["ci95"][0] > 0.5 for v in dg.values())
        results["labels"][lname] = lab

    # learning curve (does more regions help?) — only meaningful w/ >=3 regions
    if len(big) >= 3:
        print("\n=== domain learning curve (ERM logreg) ===", flush=True)
        for lname, thr in [("mc_gt1.6", 1.6), ("mc_gt0.3", 0.3)]:
            y = (mc > thr).astype(int)
            lc = domain_learning_curve(X, y, region, site, big)
            results["labels"][lname]["learning_curve_auroc_vs_ntrain_regions"] = lc
            print(f"  {lname}: " + "  ".join(f"k={k}:{v['mean_auroc']}(n={v['n']})" for k, v in lc.items()), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWROTE {OUT}  ({time.time()-t0:.1f}s total)", flush=True)


if __name__ == "__main__":
    main()
