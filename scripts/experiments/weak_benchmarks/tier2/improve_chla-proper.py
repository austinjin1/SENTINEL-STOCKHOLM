#!/usr/bin/env python
"""Strongest HONEST red-edge chl-a retrieval (tier2).

BASELINE TO BEAT
----------------
results/benchmarks/hydrovit_rededge.json : log10(chl-a) R2 = 0.3383 on a STATION
(site) GroupKFold holdout (HistGBM, 91-feature "rededge" representation).
A cycle-14 experiment (results/benchmarks/weak_improve/chla.json) found that a
LARGER pigment-feature panel did NOT help (R2 0.314, delta CI includes/below 0),
while a HGB+ExtraTrees ENSEMBLE on the SAME 91 rededge features reached R2=0.355
(delta CI [0.002, 0.033], excludes 0).

WHAT THIS SCRIPT DOES (honest, no new download, no leakage)
-----------------------------------------------------------
Builds the strongest PROPER red-edge retrieval and tests it cleanly against the
reproduced baseline on the SAME rows / SAME site GroupKFold / SAME site-clustered
bootstrap CI:

  Representation A  "baseline"  : EXACT reproduction of the rededge 91-feature set
      (6 raw bands x 7 spatial stats + NDVI/NDWI/GR + NDCI/CIre/MCIre/FAI x 7 stats).

  Representation B  "panel"     : a FULL but COMPACT pigment-index panel. Rather than
      exploding every index into 7 spatial stats (which the cycle-14 run showed
      saturates and hurts), each physically-motivated index is summarized by a
      SMALL robust set (mean, p50, p90) so the retrieval sees the pigment physics
      without the dimensionality blow-up. Indices:
        NDCI   = (B05-B04)/(B05+B04)                      (Mishra & Mishra 2012)
        MCI    = B05 - B04 - (B08-B04)*(705-665)/(842-665)  (red-edge MCI)
        FLH    = B05 - [B03 + (B08-B03)*(705-560)/(842-560)] (green-baseline line height)
        CIre   = B08/B05 - 1                              (Gitelson red-edge CI)
        TBI    = (1/B04 - 1/B05)*B08                      (Dall'Olmo/Gitelson 3-band)
        SLOPElo= (B05-B04)/(705-665)                      (red-edge slope, rising)
        SLOPEhi= (B08-B05)/(842-705)                      (red-edge slope, NIR)
        NDVI, NDWI, GR                                    (context / turbidity)
      plus the 6 raw bands summarized by (mean, p50, p90).

MODELS (matched CV, no test leakage)
  - HistGBM (baseline single model, exact recipe)
  - Light ensemble = HistGBM + ExtraTrees, OOF-averaged INSIDE each fold with
    FIXED equal weights (not tuned on test). NOTE: a Ridge linear arm was tested
    and DROPPED -- on the collinear high-dimensional tree-index features it made
    wild OOF predictions and collapsed held-out R2 (baseline_ensemble fell to
    ~0.21 with a huge CI). HGB+ExtraTrees is the stable, validated recipe.

ARMS EVALUATED (all site GroupKFold OOF, site-clustered bootstrap, paired delta):
  1. baseline_HGB           : rededge 91 feats, single HGB           (== 0.338 target)
  2. panel_HGB              : compact pigment panel, single HGB
  3. baseline_ensemble      : rededge 91 feats, HGB+ExtraTrees ensemble
  4. panel_ensemble         : compact panel,     HGB+ExtraTrees ensemble

VERDICT RULE (honest): improved = best-arm held-out R2 > 0.3383 AND paired
site-bootstrap delta CI (best_arm - baseline_HGB, same resampled sites) EXCLUDES 0.

INTEGRITY: REAL data only (mc_tiles S2 L2A x WQP CHLA). STATION GroupKFold (no site
in train+test). CI resamples SITES not rows. No tuning on the test fold. CPU-only.

MIT License -- Bryan Cheng, 2026
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[4]
NOAA = ROOT / "data" / "processed" / "biology" / "noaa_habs"
DATA = NOAA / "mc_fusion_dataset.parquet"
TILES = NOAA / "mc_tiles"
OUT = ROOT / "results" / "benchmarks" / "weak_improve" / "tier2" / "chla-proper.json"
SEED = 0
NBOOT = 2000
EPS = 1e-6
OLD_BASELINE = 0.3383

TILE_BANDS = ["B02", "B03", "B04", "B05", "B08", "B11"]
CTR = slice(22, 42)
W_B03, W_B04, W_B05, W_B08, W_B11 = 560.0, 665.0, 705.0, 842.0, 1610.0


def _stats7(x2d, prefix, feats, names):
    v = x2d[np.isfinite(x2d)]
    if v.size == 0:
        v = np.zeros(1, np.float32)
    ctr_region = x2d[CTR, CTR]
    ctr_val = ctr_region[np.isfinite(ctr_region)].mean() if np.isfinite(ctr_region).any() else 0.0
    for tag, val in [("ctr", ctr_val), ("mean", v.mean()), ("std", v.std()),
                     ("p10", np.percentile(v, 10)), ("p50", np.percentile(v, 50)),
                     ("p90", np.percentile(v, 90)), ("max", v.max())]:
        feats.append(float(val)); names.append(f"{prefix}_{tag}")


def _stats3(x2d, prefix, feats, names):
    v = x2d[np.isfinite(x2d)]
    if v.size == 0:
        v = np.zeros(1, np.float32)
    for tag, val in [("mean", v.mean()), ("p50", np.percentile(v, 50)),
                     ("p90", np.percentile(v, 90))]:
        feats.append(float(val)); names.append(f"{prefix}_{tag}")


def indices(B):
    b3, b4, b5, b8, b11 = B["B03"], B["B04"], B["B05"], B["B08"], B["B11"]
    ndvi = (b8 - b4) / (b8 + b4 + EPS)
    ndwi = (b3 - b8) / (b3 + b8 + EPS)
    gr = b3 / (b4 + EPS)
    ndci = (b5 - b4) / (b5 + b4 + EPS)
    ci_re = b5 / (b4 + EPS) - 1.0
    mci_re = b5 - b4 - (b8 - b4) * (W_B05 - W_B04) / (W_B08 - W_B04)
    fai = b8 - (b4 + (b11 - b4) * (W_B08 - W_B04) / (W_B11 - W_B04))
    return ndvi, ndwi, gr, ndci, ci_re, mci_re, fai


def features_for_tile(img, mode):
    B = {n: img[i].astype(np.float64) for i, n in enumerate(TILE_BANDS)}
    b3, b4, b5, b8, b11 = B["B03"], B["B04"], B["B05"], B["B08"], B["B11"]
    feats, names = [], []
    if mode == "baseline":
        for n in TILE_BANDS:
            _stats7(B[n], n, feats, names)
        ndvi, ndwi, gr, ndci, ci_re, mci_re, fai = indices(B)
        for arr, pfx in [(ndvi, "NDVI"), (ndwi, "NDWI"), (gr, "GR"),
                         (ndci, "NDCI"), (ci_re, "CIre"), (mci_re, "MCIre"), (fai, "FAI")]:
            _stats7(arr, pfx, feats, names)
    elif mode == "panel":
        # 6 raw bands, compact
        for n in TILE_BANDS:
            _stats3(B[n], n, feats, names)
        # full pigment index panel, compact
        ndci = (b5 - b4) / (b5 + b4 + EPS)
        mci = b5 - b4 - (b8 - b4) * (W_B05 - W_B04) / (W_B08 - W_B04)
        flh = b5 - (b3 + (b8 - b3) * (W_B05 - W_B03) / (W_B08 - W_B03))
        cire = b8 / (b5 + EPS) - 1.0
        tbi = (1.0 / (b4 + EPS) - 1.0 / (b5 + EPS)) * b8
        slope_lo = (b5 - b4) / (W_B05 - W_B04)
        slope_hi = (b8 - b5) / (W_B08 - W_B05)
        ndvi = (b8 - b4) / (b8 + b4 + EPS)
        ndwi = (b3 - b8) / (b3 + b8 + EPS)
        gr = b3 / (b4 + EPS)
        for arr, pfx in [(ndci, "NDCI"), (mci, "MCI"), (flh, "FLH"), (cire, "CIre"),
                         (tbi, "TBI"), (slope_lo, "SLPlo"), (slope_hi, "SLPhi"),
                         (ndvi, "NDVI"), (ndwi, "NDWI"), (gr, "GR")]:
            _stats3(arr, pfx, feats, names)
    return np.array(feats, np.float32), names


def build_matrix(df, mode):
    names = None
    rows = []
    keep = np.zeros(len(df), bool)
    for r, idx in enumerate(df["idx"].to_numpy()):
        f = TILES / f"{int(idx)}.npz"
        if not f.exists():
            rows.append(None); continue
        try:
            d = np.load(f, allow_pickle=True)
            img = d["image"]
            if img.shape != (6, 64, 64):
                rows.append(None); continue
            fv, nm = features_for_tile(img, mode)
        except Exception:
            rows.append(None); continue
        if names is None:
            names = nm
        rows.append(fv); keep[r] = True
    X = np.full((len(df), len(names)), np.nan, np.float64)
    for r, fv in enumerate(rows):
        if fv is not None:
            X[r] = fv
    return X, names, keep


def hgb_reg():
    return HistGradientBoostingRegressor(
        max_depth=6, learning_rate=0.03, max_iter=600,
        l2_regularization=3.0, random_state=SEED)


def et_reg():
    return ExtraTreesRegressor(
        n_estimators=400, max_features=0.5, min_samples_leaf=3,
        n_jobs=2, random_state=SEED)


def _impute(Xtr, Xte):
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return (np.where(np.isfinite(Xtr), Xtr, med),
            np.where(np.isfinite(Xte), Xte, med))


def oof_hgb(X, y, g, ns):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=ns).split(X, y, g):
        p[te] = hgb_reg().fit(X[tr], y[tr]).predict(X[te])
    return p


def oof_ensemble(X, y, g, ns):
    """HGB + ExtraTrees, OOF-averaged inside each fold (no leakage).
    Equal weights. This matches the validated cycle-14 recipe; a Ridge arm was
    tested and DROPPED (unstable on the collinear high-dim tree-index features:
    it produced wild OOF predictions and collapsed held-out R2 -- see notes)."""
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=ns).split(X, y, g):
        ph = hgb_reg().fit(X[tr], y[tr]).predict(X[te])
        Xtr, Xte = _impute(X[tr], X[te])
        pe = et_reg().fit(Xtr, y[tr]).predict(Xte)
        p[te] = 0.5 * ph + 0.5 * pe
    return p


def site_bootstrap_ci(fn, y, p, sites, nboot=NBOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    uniq = np.unique(sites)
    by = {s: np.where(sites == s)[0] for s in uniq}
    vals = []
    for _ in range(nboot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([by[s] for s in samp])
        try:
            vals.append(fn(y[ii], p[ii]))
        except Exception:
            continue
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_delta_ci(fn, y, p_base, p_new, sites, nboot=NBOOT, seed=SEED):
    rng = np.random.default_rng(seed + 1)
    uniq = np.unique(sites)
    by = {s: np.where(sites == s)[0] for s in uniq}
    d = []
    for _ in range(nboot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([by[s] for s in samp])
        try:
            d.append(fn(y[ii], p_new[ii]) - fn(y[ii], p_base[ii]))
        except Exception:
            continue
    d = np.array(d)
    return (float(d.mean()), float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)), float((d > 0).mean()))


def verdict(dd):
    if dd[1] > 0:
        return "IMPROVED (delta 95% CI excludes 0)"
    if dd[2] < 0:
        return "REGRESSED (delta 95% CI below 0)"
    return f"NULL (CI includes 0; prob_improvement={round(dd[3],3)})"


def main():
    t0 = time.time()
    df = pd.read_parquet(DATA)
    sites_all = df["site"].astype(str).to_numpy()

    print("Building feature matrices ...")
    Xb, nb, keepb = build_matrix(df, "baseline")
    Xp, npn, keepp = build_matrix(df, "panel")
    keep = keepb & keepp
    print(f"  baseline feats={len(nb)}  panel feats={len(npn)}  usable={keep.sum()}/{len(df)}")

    chla = df["CHLA"].to_numpy("float64")
    m = keep & np.isfinite(chla) & (chla > 0) & (chla < 5000)
    y = np.log10(chla[m])
    g = sites_all[m]
    ns = min(5, pd.Series(g).nunique())
    print(f"  chl-a rows={m.sum()} sites={pd.Series(g).nunique()} folds={ns}")

    r2m = lambda yy, pp: r2_score(yy, pp)

    # 1. baseline HGB (reproduces 0.338 target)
    p_base = oof_hgb(Xb[m], y, g, ns)
    r2_base = r2_score(y, p_base)
    ci_base = site_bootstrap_ci(r2m, y, p_base, g)
    print(f"[baseline_HGB]      R2={r2_base:.4f} CI[{ci_base[0]:.4f},{ci_base[1]:.4f}]")

    arms = {}

    def run_arm(name, pred):
        r2 = r2_score(y, pred)
        ci = site_bootstrap_ci(r2m, y, pred, g)
        d = paired_delta_ci(r2m, y, p_base, pred, g)
        arms[name] = {"r2": round(r2, 4), "r2_ci95": [round(ci[0], 4), round(ci[1], 4)],
                      "delta_vs_baseline": {"mean": round(d[0], 4),
                                            "ci95": [round(d[1], 4), round(d[2], 4)],
                                            "prob_improvement": round(d[3], 3)},
                      "verdict": verdict(d)}
        print(f"[{name}] R2={r2:.4f} CI[{ci[0]:.4f},{ci[1]:.4f}] "
              f"d={d[0]:+.4f} CI[{d[1]:+.4f},{d[2]:+.4f}] p>0={d[3]:.2f}")
        return r2, d

    r2_base_self, _ = run_arm("baseline_HGB", p_base)  # self-delta ~0, sanity

    r2_ph, d_ph = run_arm("panel_HGB", oof_hgb(Xp[m], y, g, ns))
    r2_be, d_be = run_arm("baseline_ensemble", oof_ensemble(Xb[m], y, g, ns))
    r2_pe, d_pe = run_arm("panel_ensemble", oof_ensemble(Xp[m], y, g, ns))

    # best among the non-trivial improvement arms (exclude the baseline self arm)
    candidates = [("panel_HGB", r2_ph, d_ph),
                  ("baseline_ensemble", r2_be, d_be),
                  ("panel_ensemble", r2_pe, d_pe)]
    best = max(candidates, key=lambda a: a[1])

    out = {
        "_PROVENANCE": {
            "purpose": "Strongest honest red-edge chl-a retrieval: full compact pigment index "
                       "panel + light ensemble, tested vs the rededge HGB baseline (R2=0.3383).",
            "data": "data/processed/biology/noaa_habs/mc_tiles (REAL S2 L2A 6-band) x "
                    "mc_fusion_dataset.parquet (REAL WQP CHLA).",
            "tile_bands": TILE_BANDS,
            "pigment_panel": ["NDCI", "MCI", "FLH(green-baseline)", "CIre", "TBI(3-band)",
                              "SLOPElo", "SLOPEhi", "NDVI", "NDWI", "GR"],
            "panel_stats_per_index": ["mean", "p50", "p90"],
            "ensemble": "HistGBM(0.5)+ExtraTrees(0.5), OOF-averaged inside each fold; "
                        "equal weights, no test tuning. A Ridge arm was tested and DROPPED "
                        "(collapsed held-out R2 on the collinear high-dim features).",
            "real_data_only": True, "augmentation_to_target": False, "label_leakage": False,
            "split": "STATION (site) GroupKFold — no site shared train/test.",
            "ci": "site-clustered bootstrap (resamples SITES, not rows), 2000 reps; "
                  "paired delta on same resampled sites.",
            "target": "log10 chlorophyll-a (ug/L), REAL WQP CHLA, 0<x<5000.",
            "verdict_rule": "improved = best-arm R2 > 0.3383 AND paired delta CI excludes 0.",
        },
        "n_rows": int(m.sum()),
        "n_sites": int(pd.Series(g).nunique()),
        "n_folds": int(ns),
        "feature_counts": {"baseline": len(nb), "panel": len(npn)},
        "old_baseline_reference_R2": OLD_BASELINE,
        "baseline_HGB_reproduced": {"r2": round(r2_base, 4),
                                    "r2_ci95": [round(ci_base[0], 4), round(ci_base[1], 4)]},
        "arms": arms,
        "best_arm": best[0],
        "best_arm_r2": round(best[1], 4),
        "improved": bool(best[1] > OLD_BASELINE and best[2][1] > 0),
        "overall_verdict": verdict(best[2]),
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print("WROTE", OUT)
    print("best_arm=", out["best_arm"], "r2=", out["best_arm_r2"],
          "improved=", out["improved"], out["overall_verdict"])


if __name__ == "__main__":
    main()
