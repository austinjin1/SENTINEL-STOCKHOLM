#!/usr/bin/env python
"""Track B 8 — STRENGTHEN THE SATELLITE ARM for microcystin.

The baseline satellite arm (sf0-sf25) scores only ~0.65-0.67 AUROC on
microcystin. It used only per-band mean/center-mean/std + mean/center-mean of
NDVI/NDCI/NDWI/green-red. This script recomputes LITERATURE-STANDARD cyanobacteria
indices from the RAW Sentinel-2 tiles and richer spatial statistics, then re-runs
the exact 07_microcystin_fusion.py protocol (station-grouped CV, site-clustered
bootstrap CI) to test whether the stronger satellite arm raises satellite-alone
AND fusion AUROC above the current 0.67 / 0.78.

Available bands (from download_microcystin_s2.py):
  B02 Blue ~490, B03 Green ~560, B04 Red ~665, B05 RedEdge1 ~705,
  B08 NIR ~842, B11 SWIR ~1610.

Computable literature indices (given these bands):
  NDCI  = (B05-B04)/(B05+B04)              chl-a / phycocyanin proxy (Mishra 2012)
  RE_ratio = B05/B04                       Soria-Perpinya 2021 PC index base
  PC_potes = (B03*B04)/B02                 Potes 2018 phycocyanin index base
  MCI_re = B05 - B04 - (B08-B04)*(705-665)/(842-665)   red-edge peak height (MCI variant;
                                          true MCI needs B06/740 which we lack)
  FAI    = B08 - (B04 + (B11-B04)*(842-665)/(1610-665))  floating-algae / scum index
  NDVI   = (B08-B04)/(B08+B04)
  NDWI   = (B03-B08)/(B03+B08)             water mask / turbidity
  GR     = B03/B04  ,  GB = B03/B02        green/red, green/blue ratios
  B05B03 = B05/B03                         red-edge / green

NOT computable (honest limits): true CI/MCI at 681/709/753 nm (no 740 nm B06),
and the direct 620 nm phycocyanin absorption feature (Sentinel-2 has NO 620 nm
band — a fundamental sensor limitation; needs Sentinel-3/MERIS/OLCI).

We add SPATIAL statistics (percentiles p10/p50/p90, max, fraction-above-threshold)
of the bloom indices, since cyanobacteria scum is spatially patchy and the tile
center may miss it.

MIT License -- Bryan Cheng, 2026
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[3]
NOAA = ROOT / "data" / "processed" / "biology" / "noaa_habs"
DATA = NOAA / "mc_fusion_dataset.parquet"
TILES = NOAA / "mc_tiles"
OUT = ROOT / "results" / "trackB_fusion" / "mc_satellite_arm.json"
SEED = 0
NBOOT = 2000
EPS = 1e-6
BANDS = ["B02", "B03", "B04", "B05", "B08", "B11"]
WHO_THRESHOLDS = {"mc_gt1.6_WHO_drinking": 1.6, "mc_gt0.3_detection": 0.3, "mc_gt8_WHO_recreational": 8.0}


# ---------- improved satellite feature extraction ----------
def spatial_stats(x, prefix, feats, names):
    """Robust spatial summary of a 2D index field."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        x = np.zeros(1, np.float32)
    for tag, v in [("mean", x.mean()), ("std", x.std()),
                   ("p10", np.percentile(x, 10)), ("p50", np.percentile(x, 50)),
                   ("p90", np.percentile(x, 90)), ("max", x.max())]:
        feats.append(float(v))
        names.append(f"{prefix}_{tag}")


def improved_features(img):
    """img [6,64,64] surface reflectance -> literature cyanobacteria feature vec."""
    B = {n: img[i].astype(np.float64) for i, n in enumerate(BANDS)}
    b2, b3, b4, b5, b8, b11 = B["B02"], B["B03"], B["B04"], B["B05"], B["B08"], B["B11"]
    feats, names = [], []
    # 1) per-band central-region reflectance (water pixels near site)
    c = slice(22, 42)
    for n in BANDS:
        feats.append(float(B[n][c, c].mean())); names.append(f"{n}_ctr")
    # 2) literature cyanobacteria indices (spatial stats each)
    ndci = (b5 - b4) / (b5 + b4 + EPS)
    re_ratio = b5 / (b4 + EPS)
    pc_potes = (b3 * b4) / (b2 + EPS)
    mci_re = b5 - b4 - (b8 - b4) * (705.0 - 665.0) / (842.0 - 665.0)
    fai = b8 - (b4 + (b11 - b4) * (842.0 - 665.0) / (1610.0 - 665.0))
    ndvi = (b8 - b4) / (b8 + b4 + EPS)
    ndwi = (b3 - b8) / (b3 + b8 + EPS)
    gr = b3 / (b4 + EPS)
    gb = b3 / (b2 + EPS)
    b5b3 = b5 / (b3 + EPS)
    for arr, pfx in [(ndci, "NDCI"), (re_ratio, "REratio"), (pc_potes, "PCpotes"),
                     (mci_re, "MCIre"), (fai, "FAI"), (ndvi, "NDVI"),
                     (ndwi, "NDWI"), (gr, "GR"), (gb, "GB"), (b5b3, "B5B3")]:
        spatial_stats(arr, pfx, feats, names)
    # 3) bloom-coverage fractions (scum patchiness)
    feats.append(float((ndci > 0.0).mean())); names.append("frac_NDCI_pos")
    feats.append(float((fai > 0.0).mean())); names.append("frac_FAI_pos")
    return np.array(feats, np.float32), names


def load_improved_sat(df):
    """Recompute improved satellite features from raw tiles, aligned to df rows."""
    feat_names = None
    X = np.full((len(df), 0), np.nan)
    cols = []
    cache = {}
    idxs = df["idx"].to_numpy()
    for row, idx in enumerate(idxs):
        f = TILES / f"{int(idx)}.npz"
        if not f.exists():
            cols.append(None); continue
        try:
            d = np.load(f, allow_pickle=True)
            img = d["image"]
            if img.shape != (6, 64, 64):
                cols.append(None); continue
            fv, nm = improved_features(img)
        except Exception:
            cols.append(None); continue
        if feat_names is None:
            feat_names = nm
            X = np.full((len(df), len(nm)), np.nan, np.float64)
        X[row] = fv
    return X, feat_names


# ---------- 07-style evaluation protocol (identical rigor) ----------
def hgb():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                          l2_regularization=1.0, class_weight="balanced",
                                          random_state=SEED)


def oof(X, y, g, ns):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=ns).split(X, y, g):
        p[te] = hgb().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def hm_ci(a, y):
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a**2) + (n0 - 1) * (q2 - a**2)) / (n1 * n0))
    return [round(float(max(0, a - 1.96 * se)), 3), round(float(min(1, a + 1.96 * se)), 3)]


def site_bootstrap_delta(y, p_a, p_b, sites, seed=SEED, B=NBOOT):
    """Paired site-clustered bootstrap on AUROC(a) - AUROC(b)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(sites)
    by_site = {s: np.where(sites == s)[0] for s in uniq}
    deltas = []
    for _ in range(B):
        chosen = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([by_site[s] for s in chosen])
        yi = y[idx]
        if yi.min() == yi.max():
            continue
        deltas.append(roc_auc_score(yi, p_a[idx]) - roc_auc_score(yi, p_b[idx]))
    deltas = np.array(deltas)
    return {"delta_median": round(float(np.median(deltas)), 3),
            "ci95": [round(float(np.percentile(deltas, 2.5)), 3),
                     round(float(np.percentile(deltas, 97.5)), 3)],
            "P_delta_gt0": round(float((deltas > 0).mean()), 3)}


def main():
    df = pq.read_table(DATA).to_pandas()
    sf_cols = [c for c in df.columns if c.startswith("sf")]
    sat_base = df[sf_cols].to_numpy("float64")            # BASELINE arm (26 feats)
    insitu = df[["CHLA", "PC"]].to_numpy("float64")
    sites = df["site"].astype(str).to_numpy()
    mc = df["mc"].to_numpy("float64")
    ns = min(5, df["site"].nunique())

    print("Recomputing improved satellite features from raw tiles...", flush=True)
    sat_impr, feat_names = load_improved_sat(df)
    n_have = int(np.isfinite(sat_impr).all(axis=1).sum())
    print(f"Improved sat feats: {sat_impr.shape[1]} dims, {n_have}/{len(df)} tiles readable", flush=True)

    fus_base = np.hstack([sat_base, insitu])
    fus_impr = np.hstack([sat_impr, insitu])

    results = {"dataset": DATA.name, "n": int(len(df)), "n_sites": int(df["site"].nunique()),
               "n_improved_sat_feats": int(sat_impr.shape[1]),
               "improved_feat_names": feat_names,
               "note": "Baseline sat arm = sf0-25 (band mean/ctr/std + NDVI/NDCI/NDWI/GR mean). "
                       "Improved sat arm = literature cyano indices (NDCI, RE-ratio, Potes-PC, "
                       "MCI-red-edge, FAI, NDVI/NDWI/GR/GB/B5B3) with p10/p50/p90/max spatial stats. "
                       "In-situ = CHLA+PC pigment. Same 07-style GroupKFold + site-bootstrap CI. "
                       "NOT computable from S2: 620nm phycocyanin abs, true MCI at 740nm (no B06).",
               "labels": {}}

    for name, thr in WHO_THRESHOLDS.items():
        y = (mc > thr).astype(int)
        if y.sum() < 15 or (len(y) - y.sum()) < 15:
            continue
        p_sat_b = oof(sat_base, y, sites, ns)
        p_sat_i = oof(sat_impr, y, sites, ns)
        p_ins = oof(insitu, y, sites, ns)
        p_fus_b = oof(fus_base, y, sites, ns)
        p_fus_i = oof(fus_impr, y, sites, ns)
        a_sat_b, a_sat_i, a_ins, a_fus_b, a_fus_i = (
            float(roc_auc_score(y, p)) for p in (p_sat_b, p_sat_i, p_ins, p_fus_b, p_fus_i))
        # best single for the IMPROVED fusion (sat_impr vs insitu)
        best_p, best_name, best_a = ((p_ins, "insitu", a_ins) if a_ins >= a_sat_i
                                     else (p_sat_i, "satellite_improved", a_sat_i))
        boot_fus = site_bootstrap_delta(y, p_fus_i, best_p, sites)           # fusion vs best single
        boot_sat = site_bootstrap_delta(y, p_sat_i, p_sat_b, sites)          # improved vs baseline sat
        boot_fusgain = site_bootstrap_delta(y, p_fus_i, p_fus_b, sites)      # improved fusion vs baseline fusion
        res = {"pos": int(y.sum()),
               "satellite_baseline_auroc": round(a_sat_b, 3),
               "satellite_improved_auroc": round(a_sat_i, 3),
               "insitu_auroc": round(a_ins, 3),
               "fusion_baseline_auroc": round(a_fus_b, 3),
               "fusion_improved_auroc": round(a_fus_i, 3),
               "fusion_improved_ci95": hm_ci(a_fus_i, y),
               "best_single_improved": best_name,
               "fusion_improved_minus_best": round(a_fus_i - best_a, 3),
               "sat_improved_minus_baseline": round(a_sat_i - a_sat_b, 3),
               "fusion_improved_minus_baseline": round(a_fus_i - a_fus_b, 3),
               "boot_fusion_vs_best_single": boot_fus,
               "boot_satarm_improved_vs_baseline": boot_sat,
               "boot_fusion_improved_vs_baseline": boot_fusgain,
               "fusion_helps_over_best_single": bool(boot_fus["ci95"][0] > 0),
               "sat_arm_improved_signif": bool(boot_sat["ci95"][0] > 0)}
        results["labels"][name] = res
        print(f"[{name}] pos={res['pos']}", flush=True)
        print(f"  SAT baseline={a_sat_b:.3f} -> improved={a_sat_i:.3f} "
              f"(Δ{res['sat_improved_minus_baseline']:+.3f}, CI {boot_sat['ci95']} signif={res['sat_arm_improved_signif']})", flush=True)
        print(f"  insitu={a_ins:.3f} | FUSION baseline={a_fus_b:.3f} -> improved={a_fus_i:.3f} "
              f"(Δ{res['fusion_improved_minus_baseline']:+.3f})", flush=True)
        print(f"  fusion_improved vs best single: Δ{res['fusion_improved_minus_best']:+.3f} "
              f"CI {boot_fus['ci95']} helps={res['fusion_helps_over_best_single']}", flush=True)

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWROTE {OUT}", flush=True)


if __name__ == "__main__":
    main()
