#!/usr/bin/env python
"""Track B 13 — LAST OOD attempt for the OPTICAL/bloom task: WATER-TYPE-INVARIANT correction.

CONTEXT (established, not re-derived here):
  - The chl-a bloom cross-region collapse is driven by region-specific water OPTICAL TYPE +
    atmosphere, NOT a missing biological variable (unlike microcystin/genotype).
  - Satellite OPTICS ALONE cross-region (leave-one-HUC-region-out, LORO) = ~0.41-0.45 AUROC,
    i.e. AT or BELOW chance. The only transferable channel was antecedent chl PERSISTENCE.
  - HYPOTHESIS of THIS script: a correction that removes water-type variance from the optics
    MIGHT re-express the optics->chl mapping invariantly and recover cross-region OPTICAL skill.

DATA: mc_fusion_expanded.parquet (4196 rows, 18 HUC2 regions). Real, measured chl-a label
  (CHLA column, lab/probe) — NOT derived from the input spectral features (non-circular).
  Spectral features sf0..sf25 map to Sentinel-2 bands [B02,B03,B04,B05,B08,B11] as
  3 stats/band (full-tile mean, center mean, std) then 4 indices x2 (NDVI, NDCI, NDWI, G/R):
    sf0-2 B02(blue) | sf3-5 B03(green) | sf6-8 B04(red) | sf9-11 B05(red-edge) |
    sf12-14 B08(NIR) | sf15-17 B11(SWIR) | sf18-19 NDVI | sf20-21 NDCI | sf22-23 NDWI |
    sf24-25 G/R.  (Definition per expand_03_build_dataset.py; verified against value ranges.)

  With a real measured chl label, only 2 HUC regions (03 = Florida/SE, 18 = California) carry
  >=40 labelled rows with BOTH classes present. LORO therefore realises as a Florida<->California
  optical transfer — a genuinely hard, water-type-divergent OOD test that matches the ~0.41-0.45
  optics-only baseline. This is the same collapse the correction must fix.

TASK: predict chl-a EXCEEDANCE (thr = 8 ug/L eutrophic screen; 24 ug/L WHO Alert-2 high) from
  SATELLITE OPTICS ONLY, under LORO, and test whether a water-type-invariant correction lifts
  cross-region AUROC meaningfully above chance with the region-clustered bootstrap CI EXCLUDING 0.5.
  We deliberately isolate OPTICS (no antecedent-chl persistence) because persistence already
  transfers and would mask whether the OPTICAL channel itself was recovered — the actual question.

METHODS (ALL transforms fit on TRAIN folds only; NO test leakage — the held-out region's rows
  never touch any fit of clustering, scaler, or model):
  M0  ERM baseline           : raw sf* -> HGB (the ~0.45 chance baseline).
  M1  water-type clustering  : k-means on a water-type SIGNATURE (band-shape: each band mean
                               normalised by the tile's total brightness = spectral shape,
                               invariant to overall magnitude/illumination), fit on TRAIN;
                               assign every sample (train+test) to nearest TRAIN centroid;
                               standardize predictive spectral features WITHIN each water-type
                               cluster using TRAIN-fold per-cluster mean/std -> optics expressed
                               relative-to-water-type -> HGB.
  M2  robust bloom indices   : baseline-subtracted line-height / red-edge indices that are more
                               atmosphere/water-type robust, computed from band means:
                               MCI-like red-edge line height, NDCI, red-edge slope, FLH-like,
                               plus brightness-normalised band-shape -> HGB.
  M3  M1 + M2 combined       : robust indices, water-type-cluster-standardised.
  M4  water-type domain-adapt: CORAL-style feature de-correlation/whitening conditioned on
                               water-type cluster (align each cluster's covariance to a global
                               TRAIN target), fit train-only -> HGB.

EVALUATION / GUARDRAILS:
  - CROSS-region: LORO (each region predicted by a model trained on the others). Region-clustered
    bootstrap 95% CI on the AUROC (resample whole regions... with 2 regions we ALSO report a
    site-clustered bootstrap within the pooled LORO OOF as the primary CI, since 2 regions cannot
    themselves be bootstrapped; both are reported honestly).
  - WITHIN-region: GroupKFold by SITE, to confirm the correction does NOT destroy within-region skill.
  - A WIN (ood_recovered) requires LORO AUROC meaningfully > 0.5 AND the CI excluding 0.5.
  Honest null is a valid, reportable outcome — if all methods null, fusion is scoped to within-region.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "data" / "processed" / "biology" / "noaa_habs" / "mc_fusion_expanded.parquet"
OUT = ROOT / "results" / "trackB_fusion" / "ood_watertype" / "watertype_invariant_ood.json"
SEED = 0
NBOOT = 2000
THRESHOLDS = {"chla_gt8_eutrophic": 8.0, "chla_gt24_WHO_high": 24.0}
MIN_REGION_N = 40
N_WATERTYPES = 4

SF = [f"sf{i}" for i in range(26)]
# band-mean feature indices (full-tile means) per expand_03_build_dataset.py layout
BM = {"blue": 0, "green": 3, "red": 6, "rededge": 9, "nir": 12, "swir": 15}


def hgb():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                          l2_regularization=1.0, class_weight="balanced",
                                          random_state=SEED)


# ---------------------------------------------------------------- feature engineering
def band_means(X):
    """Return dict of the 6 full-tile band means as columns of X (float64)."""
    return {k: X[:, i].astype("float64") for k, i in BM.items()}


def watertype_signature(X):
    """Illumination/magnitude-invariant water-type signature = band SHAPE.

    Each band mean divided by the tile's summed brightness -> spectral shape only.
    This captures water OPTICAL TYPE (clear vs turbid vs CDOM vs sediment) independent
    of overall reflectance magnitude/atmosphere/illumination, which is exactly the
    nuisance we want the clustering to key on (NOT the bloom signal directly)."""
    b = band_means(X)
    stack = np.vstack([b["blue"], b["green"], b["red"], b["rededge"], b["nir"], b["swir"]]).T
    tot = stack.sum(axis=1, keepdims=True)
    tot[tot <= 0] = 1e-6
    return stack / tot


def robust_indices(X):
    """Baseline-subtracted line-height / red-edge indices robust to atmosphere & water type.

    Computed from band MEANS (units cancel in ratios / line heights):
      NDCI      = (RE - R)/(RE + R)              red-edge chlorophyll index
      red_slope = RE - R                          red-edge slope (line height numerator)
      MCI_like  = RE - R - (B08 - R)*(w)          maximum chlorophyll index: RE height above
                                                  a red->NIR baseline (atmosphere-robust)
      FLH_like  = green - (blue + red)/2          fluorescence-line-height analogue in VIS
      NDVI, NDWI                                   vegetation / water discriminators
      GtoR      = green/red
    All are ratios or differences of co-located bands, so multiplicative atmospheric gain and
    additive path-radiance offsets largely cancel -> more transferable than raw reflectance."""
    b = band_means(X)
    eps = 1e-6
    R, RE, NIR, G, B, SW = b["red"], b["rededge"], b["nir"], b["green"], b["blue"], b["swir"]
    lam_r, lam_re, lam_nir = 665.0, 705.0, 842.0  # S2 central wavelengths
    w = (lam_re - lam_r) / (lam_nir - lam_r)
    ndci = (RE - R) / (RE + R + eps)
    red_slope = RE - R
    mci = RE - R - w * (NIR - R)                     # baseline-subtracted red-edge line height
    flh = G - 0.5 * (B + R)
    ndvi = (NIR - R) / (NIR + R + eps)
    ndwi = (G - NIR) / (G + NIR + eps)
    gtor = G / (R + eps)
    # brightness-normalised band shape (water-type-robust magnitude removal)
    sig = watertype_signature(X)
    feats = np.column_stack([ndci, red_slope, mci, flh, ndvi, ndwi, gtor])
    return np.hstack([feats, sig])


# ---------------------------------------------------------------- train-only transforms
class WaterTypeStandardizer:
    """Cluster by water-type signature (TRAIN centroids), standardize features WITHIN cluster.

    Fit on TRAIN only: k-means centroids on the signature + per-cluster mean/std of the
    predictive features. Transform assigns any sample to its nearest TRAIN centroid and applies
    that cluster's TRAIN mean/std -> features expressed relative to their water type."""

    def __init__(self, k=N_WATERTYPES, seed=SEED):
        self.k = k
        self.seed = seed

    def fit(self, X_pred, X_full):
        sig = watertype_signature(X_full)
        self.km = KMeans(n_clusters=self.k, random_state=self.seed, n_init=10).fit(sig)
        lab = self.km.labels_
        self.mu, self.sd = {}, {}
        for c in range(self.k):
            m = lab == c
            if m.sum() >= 2:
                self.mu[c] = X_pred[m].mean(axis=0)
                self.sd[c] = X_pred[m].std(axis=0) + 1e-6
            else:
                self.mu[c] = X_pred.mean(axis=0)
                self.sd[c] = X_pred.std(axis=0) + 1e-6
        return self

    def transform(self, X_pred, X_full):
        lab = self.km.predict(watertype_signature(X_full))
        out = np.empty_like(X_pred, dtype="float64")
        for c in range(self.k):
            m = lab == c
            if m.any():
                out[m] = (X_pred[m] - self.mu[c]) / self.sd[c]
        return out


class WaterTypeCORAL:
    """Per-water-type CORAL whitening: align each cluster's covariance to a global TRAIN target.

    Fit train-only: cluster (signature), compute each cluster's whitening + the global TRAIN
    coloring, so every cluster's features are mapped to a common covariance -> water-type domain
    shift removed at 2nd order. Test rows are assigned to nearest TRAIN centroid and mapped identically."""

    def __init__(self, k=N_WATERTYPES, seed=SEED, eps=1e-3):
        self.k = k
        self.seed = seed
        self.eps = eps

    @staticmethod
    def _mat_pow(C, p, eps):
        C = C + eps * np.eye(C.shape[0])
        w, V = np.linalg.eigh(C)
        w = np.clip(w, eps, None)
        return (V * (w ** p)) @ V.T

    def fit(self, X_pred, X_full):
        Xs = (X_pred - X_pred.mean(0)) / (X_pred.std(0) + 1e-6)
        self.gmu = X_pred.mean(0)
        self.gsd = X_pred.std(0) + 1e-6
        sig = watertype_signature(X_full)
        self.km = KMeans(n_clusters=self.k, random_state=self.seed, n_init=10).fit(sig)
        lab = self.km.labels_
        Ct = np.cov(Xs, rowvar=False) + self.eps * np.eye(Xs.shape[1])
        self.color = self._mat_pow(Ct, 0.5, self.eps)  # target coloring (global TRAIN)
        self.whiten = {}
        self.cmu = {}
        for c in range(self.k):
            m = lab == c
            if m.sum() > Xs.shape[1] + 2:
                Cc = np.cov(Xs[m], rowvar=False)
                self.whiten[c] = self._mat_pow(Cc, -0.5, self.eps)
                self.cmu[c] = Xs[m].mean(0)
            else:
                self.whiten[c] = np.eye(Xs.shape[1])
                self.cmu[c] = Xs[m].mean(0) if m.any() else np.zeros(Xs.shape[1])
        return self

    def transform(self, X_pred, X_full):
        Xs = (X_pred - self.gmu) / self.gsd
        lab = self.km.predict(watertype_signature(X_full))
        out = np.empty_like(Xs)
        for c in range(self.k):
            m = lab == c
            if m.any():
                out[m] = (Xs[m] - self.cmu[c]) @ self.whiten[c] @ self.color
        return out


# ---------------------------------------------------------------- CV drivers
def loro_oof(build_pred, X_full, y, region, regions):
    """Leave-one-region-out OOF probs. build_pred(tr_idx, te_idx)->(Xtr,ytr,Xte) so any
    train-only transform is fit inside the fold. Here we pass closures per method."""
    p = np.full(len(y), np.nan)
    for r in regions:
        te = region == r
        tr = ~te
        Xtr, Xte = build_pred(tr, te, X_full)
        p[te] = hgb().fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
    return p


def within_oof(build_pred, X_full, y, groups, ns):
    p = np.full(len(y), np.nan)
    for tr_i, te_i in GroupKFold(n_splits=ns).split(X_full, y, groups):
        tr = np.zeros(len(y), bool); tr[tr_i] = True
        te = np.zeros(len(y), bool); te[te_i] = True
        Xtr, Xte = build_pred(tr, te, X_full)
        p[te] = hgb().fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
    return p


# ---------------------------------------------------------------- method feature builders
def make_builders():
    """Each builder: (train_mask, test_mask, X_full) -> (X_train_pred, X_test_pred).
    ALL fitting uses only train_mask rows."""

    def b_erm(tr, te, Xf):
        return Xf[tr][:, :26], Xf[te][:, :26]

    def b_wtstd(tr, te, Xf):
        std = WaterTypeStandardizer().fit(Xf[tr][:, :26], Xf[tr])
        return std.transform(Xf[tr][:, :26], Xf[tr]), std.transform(Xf[te][:, :26], Xf[te])

    def b_robust(tr, te, Xf):
        return robust_indices(Xf[tr]), robust_indices(Xf[te])

    def b_robust_wtstd(tr, te, Xf):
        Rtr, Rte = robust_indices(Xf[tr]), robust_indices(Xf[te])
        std = WaterTypeStandardizer().fit(Rtr, Xf[tr])
        return std.transform(Rtr, Xf[tr]), std.transform(Rte, Xf[te])

    def b_coral(tr, te, Xf):
        cor = WaterTypeCORAL().fit(Xf[tr][:, :26], Xf[tr])
        return cor.transform(Xf[tr][:, :26], Xf[tr]), cor.transform(Xf[te][:, :26], Xf[te])

    return {"M0_ERM_raw_optics": b_erm,
            "M1_watertype_standardized": b_wtstd,
            "M2_robust_bloom_indices": b_robust,
            "M3_robust_plus_watertype": b_robust_wtstd,
            "M4_watertype_CORAL": b_coral}


# ---------------------------------------------------------------- bootstrap CIs
def cluster_bootstrap_auc(y, p, clusters, seed=SEED, B=NBOOT):
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    by = {c: np.where(clusters == c)[0] for c in uniq}
    aucs = []
    for _ in range(B):
        chosen = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([by[c] for c in chosen])
        yi = y[idx]
        if yi.min() == yi.max():
            continue
        aucs.append(roc_auc_score(yi, p[idx]))
    aucs = np.array(aucs)
    return {"auroc_boot_median": round(float(np.median(aucs)), 3),
            "ci95": [round(float(np.percentile(aucs, 2.5)), 3),
                     round(float(np.percentile(aucs, 97.5)), 3)],
            "P_gt_0.5": round(float((aucs > 0.5).mean()), 3),
            "n_boot": int(len(aucs))}


def main():
    df = pd.read_parquet(DATA)
    df["CHLA"] = pd.to_numeric(df["CHLA"], errors="coerce")
    df.loc[(df["CHLA"] <= 0) | (df["CHLA"] > 5000), "CHLA"] = np.nan
    lab = df[df["CHLA"].notna()].copy().reset_index(drop=True)
    lab["region"] = lab["huc2"].astype(str)

    # eligible regions for LORO: >=MIN_REGION_N rows (both classes checked per threshold)
    Xfull = lab[SF].to_numpy("float64")
    site = lab["site"].astype(str).to_numpy()
    region_all = lab["region"].to_numpy()

    builders = make_builders()
    results = {
        "dataset": DATA.name,
        "task": "chl-a exceedance from SATELLITE OPTICS ONLY; water-type-invariant OOD correction",
        "n_labelled": int(len(lab)),
        "n_sites": int(lab["site"].nunique()),
        "label_source": "measured CHLA (lab/probe); NOT derived from sf* input features (non-circular)",
        "spectral_map": "sf0-2 blue|3-5 green|6-8 red|9-11 rededge|12-14 nir|15-17 swir|"
                        "18-19 NDVI|20-21 NDCI|22-23 NDWI|24-25 G/R (expand_03_build_dataset.py)",
        "n_watertypes_kmeans": N_WATERTYPES,
        "protocol": "LORO cross-region + GroupKFold-by-site within-region; ALL transforms fit "
                    "train-only (held-out region never touches any clustering/scaler/model fit).",
        "methods": list(builders.keys()),
        "labels": {},
    }

    for name, thr in THRESHOLDS.items():
        y = (lab["CHLA"].to_numpy() > thr).astype(int)
        # region eligibility (both classes, >= MIN_REGION_N)
        keep = []
        for r in np.unique(region_all):
            m = region_all == r
            if m.sum() >= MIN_REGION_N and 0 < y[m].sum() < m.sum():
                keep.append(r)
        m = np.isin(region_all, keep)
        Xf_m, y_m, reg_m, site_m = Xfull[m], y[m], region_all[m], site[m]
        ns = min(5, len(np.unique(site_m)))

        lab_res = {"threshold": thr, "pos": int(y_m.sum()), "neg": int(len(y_m) - y_m.sum()),
                   "regions_used": list(keep), "n_regions": len(keep),
                   "n_cross_region_rows": int(m.sum()),
                   "loro_realization": ("Florida(03) <-> California(18) optical transfer"
                                        if set(keep) == {"03", "18"} else "multi-region LORO"),
                   "methods": {}}

        for mname, builder in builders.items():
            # cross-region LORO
            p_x = loro_oof(builder, Xf_m, y_m, reg_m, keep)
            auc_x = float(roc_auc_score(y_m, p_x))
            # region-clustered bootstrap (whole regions) — primary if >=3 regions
            boot_reg = cluster_bootstrap_auc(y_m, p_x, reg_m)
            # site-clustered bootstrap within pooled LORO OOF (usable with only 2 regions)
            boot_site = cluster_bootstrap_auc(y_m, p_x, site_m)
            # per-region held-out
            per_region = {}
            for r in keep:
                ri = reg_m == r
                yr = y_m[ri]
                if 0 < yr.sum() < len(yr):
                    per_region[r] = {"auroc": round(float(roc_auc_score(yr, p_x[ri])), 3),
                                     "n": int(ri.sum()), "pos": int(yr.sum())}
            # within-region (site GroupKFold)
            p_w = within_oof(builder, Xf_m, y_m, site_m, ns)
            auc_w = float(roc_auc_score(y_m, p_w))

            n_reg = len(np.unique(reg_m))
            primary = boot_reg if n_reg >= 3 else boot_site
            ci = primary["ci95"]
            lab_res["methods"][mname] = {
                "loro_auroc": round(auc_x, 3),
                "within_region_auroc": round(auc_w, 3),
                "region_cluster_bootstrap": boot_reg,
                "site_cluster_bootstrap": boot_site,
                "primary_ci_basis": ("region-clustered" if n_reg >= 3 else
                                     "site-clustered (only 2 regions; region-bootstrap degenerate)"),
                "ci95_primary": ci,
                "ci_excludes_chance": bool(ci[0] > 0.5),
                "above_chance": bool(auc_x > 0.5 and ci[0] > 0.5),
                "per_region_heldout": per_region,
            }
            print(f"[{name}] {mname}: LORO={auc_x:.3f} within={auc_w:.3f} "
                  f"CI{ci}({'region' if n_reg>=3 else 'site'}) "
                  f"excl0.5={ci[0]>0.5}", flush=True)

        results["labels"][name] = lab_res

    # verdict
    best = None
    for name, lr in results["labels"].items():
        erm = lr["methods"]["M0_ERM_raw_optics"]["loro_auroc"]
        for mname, mr in lr["methods"].items():
            if mname == "M0_ERM_raw_optics":
                continue
            if mr["above_chance"]:
                cand = (mr["loro_auroc"], name, mname, erm)
                if best is None or cand[0] > best[0]:
                    best = cand
    erm_baselines = {name: lr["methods"]["M0_ERM_raw_optics"]["loro_auroc"]
                     for name, lr in results["labels"].items()}
    ood_recovered = best is not None
    results["verdict"] = {
        "erm_optics_only_loro_auroc": erm_baselines,
        "ood_recovered": ood_recovered,
        "best_corrected": ({"loro_auroc": best[0], "label": best[1], "method": best[2],
                            "vs_erm": round(best[0] - best[3], 3)} if best else None),
        "conclusion": (
            "RECOVERED: a water-type-invariant correction lifted optics-only cross-region AUROC "
            "above chance with the primary bootstrap CI excluding 0.5." if ood_recovered else
            "NULL: no water-type-invariant correction (clustering-standardization, robust "
            "line-height indices, per-water-type CORAL, or their combination) lifted OPTICS-ONLY "
            "cross-region AUROC meaningfully above chance with CI excluding 0.5. The optical/bloom "
            "cross-region collapse is NOT recoverable by removing water-type variance -> the "
            "region-specific optics->chl mapping is not merely a water-type nuisance transform. "
            "Fusion for the bloom task should be scoped to WITHIN-region (where it wins), with "
            "cross-region carried only by antecedent chl-a persistence (established separately).")}

    OUT.write_text(json.dumps(results, indent=2))
    print("\nWROTE", OUT)
    print("ood_recovered =", ood_recovered)


if __name__ == "__main__":
    main()
