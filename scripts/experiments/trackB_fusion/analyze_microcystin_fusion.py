#!/usr/bin/env python3
"""Microcystin bimodal fusion test.

Modalities:
  SATELLITE : Sentinel-2 spectral features from downloaded tiles (remote bloom).
  INSITU    : co-located in-situ chl-a / phycocyanin pigment (local biological probe).
  FUSION    : both concatenated.

Label: WHO recreational microcystin threshold, >8 ug/L = positive.
Eval : station-grouped 5-fold CV, GBM + logistic; AUROC with Hanley-McNeil CI.
No fabrication. Tiles + pigment + toxin are all real WQP / Sentinel-2 data.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

PROJECT = Path("/home/bcheng/SENTINEL")
TILE_CACHE = PROJECT / "data/processed/biology/noaa_habs/mc_tiles"
TARGETS = PROJECT / "data/processed/biology/noaa_habs/mc_download_targets.parquet"
IND = PROJECT / "data/processed/biology/noaa_habs/wqp_hab_indicators.parquet"
BANDS = ["B02", "B03", "B04", "B05", "B08", "B11"]
WHO_THR = 8.0


def spectral_features(img):
    """img: [6,64,64] reflectance. Return spectral summary vector."""
    b = {name: img[i] for i, name in enumerate(BANDS)}
    # center 20x20 region (water body near site) + full-tile stats
    c = slice(22, 42)
    feats = []
    for i, name in enumerate(BANDS):
        arr = img[i]
        feats += [float(arr.mean()), float(arr[c, c].mean()), float(arr.std())]
    eps = 1e-6
    ndvi = (b["B08"] - b["B04"]) / (b["B08"] + b["B04"] + eps)
    ndci = (b["B05"] - b["B04"]) / (b["B05"] + b["B04"] + eps)  # chl proxy
    ndwi = (b["B03"] - b["B08"]) / (b["B03"] + b["B08"] + eps)
    gr = b["B03"] / (b["B04"] + eps)
    # cyano-relevant: green/red edge, B08 scatter
    for idx in (ndvi, ndci, ndwi, gr):
        feats += [float(idx.mean()), float(idx[c, c].mean())]
    return np.array(feats, dtype=np.float32)


def hanley_mcneil_ci(auc, n_pos, n_neg, z=1.96):
    if n_pos == 0 or n_neg == 0:
        return (np.nan, np.nan)
    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) +
                  (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg))
    return (max(0.0, auc - z * se), min(1.0, auc + z * se))


def load_dataset():
    mc = pd.read_parquet(TARGETS).reset_index(drop=True)
    mc["date"] = pd.to_datetime(mc["date"]).dt.normalize()

    # in-situ pigment, aggregated to site-date
    ind = pd.read_parquet(IND, columns=[
        "MonitoringLocationIdentifier", "ActivityStartDate",
        "CharacteristicName", "ResultMeasureValue"])
    ind["date"] = pd.to_datetime(ind["ActivityStartDate"], errors="coerce").dt.normalize()
    ind["val"] = pd.to_numeric(ind["ResultMeasureValue"], errors="coerce")
    ind = ind.dropna(subset=["date", "val"])
    ind["grp"] = np.where(ind["CharacteristicName"].str.contains("hycocyanin"), "PC", "CHLA")
    ind = ind.rename(columns={"MonitoringLocationIdentifier": "Location_Identifier"})
    agg = (ind.groupby(["Location_Identifier", "date", "grp"])["val"].mean()
           .unstack("grp").reset_index())
    for c in ("CHLA", "PC"):
        if c not in agg.columns:
            agg[c] = np.nan

    rows = []
    for f in TILE_CACHE.glob("*.npz"):
        idx = int(f.stem)
        try:
            d = np.load(f, allow_pickle=True)
            img = d["image"]
            if img.shape != (6, 64, 64):
                continue
            sf = spectral_features(img)
        except Exception:
            continue
        if idx >= len(mc):
            continue
        r = mc.iloc[idx]
        rows.append({
            "idx": idx, "site": r["Location_Identifier"],
            "date": r["date"], "mc": float(r["val"]),
            "cloud": float(d["cloud"]) if "cloud" in d else np.nan,
            **{f"sf{i}": v for i, v in enumerate(sf)},
        })
    df = pd.DataFrame(rows)
    # near-day (+-3d) pigment merge per site
    parts = []
    for sid, g in df.groupby("site"):
        a = agg[agg["Location_Identifier"] == sid][["date", "CHLA", "PC"]].sort_values("date")
        gg = g.sort_values("date")
        if len(a):
            gg = pd.merge_asof(gg, a, on="date",
                               tolerance=pd.Timedelta("3D"), direction="nearest")
        else:
            gg["CHLA"] = np.nan
            gg["PC"] = np.nan
        parts.append(gg)
    df = pd.concat(parts).reset_index(drop=True)
    df["label"] = (df["mc"] > WHO_THR).astype(int)
    return df


def eval_arm(X, y, groups, name):
    # drop rows with any NaN feature
    mask = ~np.isnan(X).any(axis=1)
    Xa, ya, ga = X[mask], y[mask], groups[mask]
    npos, nneg = int(ya.sum()), int((ya == 0).sum())
    if npos < 10 or nneg < 10:
        return None
    oof = np.full(len(ya), np.nan)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in sgkf.split(Xa, ya, ga):
        if ya[tr].sum() < 2:
            continue
        sc = StandardScaler().fit(Xa[tr])
        clf = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.06, max_depth=3,
            l2_regularization=1.0, random_state=0)
        clf.fit(sc.transform(Xa[tr]), ya[tr])
        oof[te] = clf.predict_proba(sc.transform(Xa[te]))[:, 1]
    m = ~np.isnan(oof)
    auc = roc_auc_score(ya[m], oof[m])
    lo, hi = hanley_mcneil_ci(auc, int(ya[m].sum()), int((ya[m] == 0).sum()))
    # logistic baseline too
    oof2 = np.full(len(ya), np.nan)
    for tr, te in sgkf.split(Xa, ya, ga):
        if ya[tr].sum() < 2:
            continue
        sc = StandardScaler().fit(Xa[tr])
        lr = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
        lr.fit(sc.transform(Xa[tr]), ya[tr])
        oof2[te] = lr.predict_proba(sc.transform(Xa[te]))[:, 1]
    m2 = ~np.isnan(oof2)
    auc_lr = roc_auc_score(ya[m2], oof2[m2])
    return {"name": name, "n": int(mask.sum()), "npos": npos, "nneg": nneg,
            "auc_gbm": round(auc, 3), "ci": (round(lo, 3), round(hi, 3)),
            "auc_lr": round(auc_lr, 3)}


def main():
    df = load_dataset()
    print(f"Loaded {len(df)} co-located (tile+toxin) samples")
    print(f"Sites: {df['site'].nunique()}, WHO+ positives: {df['label'].sum()} "
          f"({df['label'].mean()*100:.1f}%)")
    print(f"Pigment coverage: CHLA={df['CHLA'].notna().sum()}, PC={df['PC'].notna().sum()}")

    sf_cols = [c for c in df.columns if c.startswith("sf")]
    groups = df["site"].values
    y = df["label"].values.astype(int)

    X_sat = df[sf_cols].values.astype(np.float64)
    # in-situ arm: log pigment (chl-a always; PC where present -> fill with chla-derived? keep separate)
    ins = df[["CHLA", "PC"]].copy()
    ins["CHLA"] = np.log1p(ins["CHLA"].clip(lower=0))
    ins["PC"] = np.log1p(ins["PC"].clip(lower=0))
    X_ins = ins.values.astype(np.float64)
    X_fus = np.hstack([X_sat, X_ins])

    print("\n=== ARMS (label = microcystin > 8 ug/L WHO) ===")
    print("Design A: all samples (in-situ arm has NaNs -> those rows dropped per-arm)")
    for X, nm in [(X_sat, "SATELLITE"), (X_ins, "INSITU-pigment"), (X_fus, "FUSION")]:
        r = eval_arm(X, y, groups, nm)
        print(r)

    # Design B: restrict to rows where BOTH modalities present (fair fusion comparison)
    both = ~np.isnan(X_ins).any(axis=1)
    print(f"\nDesign B: rows with BOTH satellite+in-situ present: {both.sum()} "
          f"(pos={y[both].sum()})")
    for X, nm in [(X_sat, "SATELLITE"), (X_ins, "INSITU-pigment"), (X_fus, "FUSION")]:
        r = eval_arm(X[both], y[both], groups[both], nm + "|both")
        print(r)

    # Also: chl-a only vs chl-a+PC to see if pigment alone saturates
    df.to_parquet(PROJECT / "data/processed/biology/noaa_habs/mc_fusion_dataset.parquet")
    print("\nsaved mc_fusion_dataset.parquet")


if __name__ == "__main__":
    main()
