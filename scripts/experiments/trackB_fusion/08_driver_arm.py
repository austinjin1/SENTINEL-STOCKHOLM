#!/usr/bin/env python
"""Track B 8 — add the physicochemical DRIVER modality to the microcystin fusion.

The cyanotoxin literature: microcystin production is driven by water temperature
and nutrients (N, P), and the pigment<->toxin relationship is time/space-variable.
So a 3rd modality — the physicochemical drivers — is plausibly complementary to
BOTH satellite optics (bloom) and in-situ pigment (biomass proxy). This downloads
real WQP water-temperature / phosphorus / nitrogen at the microcystin sites,
co-locates them, and tests whether a 3-way fusion beats the 2-way satellite+pigment
win (~0.78). Real data only; labels non-circular (microcystin threshold).
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "processed" / "biology" / "noaa_habs" / "mc_fusion_dataset.parquet"
RAW = ROOT / "data" / "raw" / "wqp_drivers"
RAW.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "results" / "trackB_fusion" / "mc_driver_fusion.json"
CHARS = {"temp": "Temperature, water", "phosphorus": "Phosphorus", "nitrogen": "Nitrogen"}
SEED = 0


def wqp_download(sites, char, tag):
    cache = RAW / f"{tag}.csv"
    if cache.exists():
        return pd.read_csv(cache, low_memory=False)
    import requests
    frames = []
    for i in range(0, len(sites), 40):
        batch = ";".join(sites[i:i + 40])
        url = (f"https://www.waterqualitydata.us/data/Result/search?siteid={quote(batch)}"
               f"&characteristicName={quote(char)}&mimeType=csv&zip=no&startDateLo=01-01-2016")
        try:
            r = requests.get(url, timeout=120)
            if r.ok and len(r.text) > 200:
                frames.append(pd.read_csv(io.StringIO(r.text), low_memory=False))
        except Exception as e:
            print(f"  {tag} batch {i}: {type(e).__name__}", flush=True)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df.to_csv(cache, index=False)
    print(f"  {tag}: {len(df)} records downloaded ({df['MonitoringLocationIdentifier'].nunique() if len(df) else 0} sites)", flush=True)
    return df


def parse_driver(df):
    if len(df) == 0:
        return pd.DataFrame(columns=["site", "date", "val"])
    out = pd.DataFrame({
        "site": df["MonitoringLocationIdentifier"].astype(str),
        "date": pd.to_datetime(df["ActivityStartDate"], errors="coerce"),
        "val": pd.to_numeric(df["ResultMeasureValue"], errors="coerce"),
    }).dropna(subset=["date", "val"])
    return out.sort_values("date")


def nearest(driver, site, date, tol_days=14):
    d = driver[driver["site"] == site]
    if len(d) == 0:
        return np.nan
    dd = (d["date"] - date).abs().dt.days
    i = dd.values.argmin()
    return float(d["val"].values[i]) if dd.values[i] <= tol_days else np.nan


def hgb():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                          l2_regularization=1.0, class_weight="balanced", random_state=SEED)


def oof(X, y, g, ns):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=ns).split(X, y, g):
        p[te] = hgb().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def boot_delta(y, p_a, p_b, sites, B=2000):
    rng = np.random.default_rng(SEED)
    uniq = np.unique(sites); by = {s: np.where(sites == s)[0] for s in uniq}
    ds = []
    for _ in range(B):
        idx = np.concatenate([by[s] for s in rng.choice(uniq, len(uniq), replace=True)])
        yi = y[idx]
        if yi.min() != yi.max():
            ds.append(roc_auc_score(yi, p_a[idx]) - roc_auc_score(yi, p_b[idx]))
    ds = np.array(ds)
    return {"median": round(float(np.median(ds)), 3),
            "ci95": [round(float(np.percentile(ds, 2.5)), 3), round(float(np.percentile(ds, 97.5)), 3)],
            "P_gt0": round(float((ds > 0).mean()), 3)}


def boot_absolute(y, p, sites, B=2000):
    """Site-clustered bootstrap 95% CI on an ABSOLUTE AUROC (resample sites w/ replacement)."""
    rng = np.random.default_rng(SEED)
    uniq = np.unique(sites); by = {s: np.where(sites == s)[0] for s in uniq}
    aucs = []
    for _ in range(B):
        idx = np.concatenate([by[s] for s in rng.choice(uniq, len(uniq), replace=True)])
        yi = y[idx]
        if yi.min() != yi.max():
            aucs.append(roc_auc_score(yi, p[idx]))
    aucs = np.array(aucs)
    return {"median": round(float(np.median(aucs)), 3),
            "ci95": [round(float(np.percentile(aucs, 2.5)), 3), round(float(np.percentile(aucs, 97.5)), 3)]}


def main():
    df = pq.read_table(DATA).to_pandas()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    sites = sorted(df["site"].astype(str).unique())
    print(f"{len(df)} microcystin samples, {len(sites)} sites. Downloading WQP drivers...", flush=True)

    drivers = {}
    for tag, char in CHARS.items():
        drivers[tag] = parse_driver(wqp_download(sites, char, tag))

    # co-locate driver values (nearest within +/-14 d)
    for tag, dv in drivers.items():
        col = np.array([nearest(dv, str(s), d) for s, d in zip(df["site"].astype(str), df["date"])])
        df[f"drv_{tag}"] = col
        print(f"  {tag}: coverage {np.isfinite(col).mean():.1%}", flush=True)

    sf = [c for c in df.columns if c.startswith("sf")]
    sat = df[sf].to_numpy("float64")
    pig = df[["CHLA", "PC"]].to_numpy("float64")
    drv = df[[f"drv_{t}" for t in CHARS]].to_numpy("float64")
    mc = df["mc"].to_numpy("float64")
    site_arr = df["site"].astype(str).to_numpy()
    ns = min(5, len(sites))

    results = {"n": int(len(df)), "n_sites": len(sites),
               "driver_coverage": {t: round(float(np.isfinite(df[f"drv_{t}"]).mean()), 3) for t in CHARS},
               "note": "3rd modality = WQP physicochemical drivers (water temp, P, N), literature drivers of "
                       "cyanotoxin production. Tests whether drivers add over the satellite+pigment fusion.",
               "labels": {}}
    for name, thr in [("mc_gt1.6", 1.6), ("mc_gt0.3", 0.3)]:
        y = (mc > thr).astype(int)
        arms = {"satellite": sat, "pigment": pig, "drivers": drv,
                "sat+pigment": np.hstack([sat, pig]),
                "sat+pigment+drivers": np.hstack([sat, pig, drv])}
        oofs = {k: oof(X, y, site_arr, ns) for k, X in arms.items()}
        res = {"pos": int(y.sum())}
        for k, p in oofs.items():
            res[k] = round(float(roc_auc_score(y, p)), 3)
        # does adding drivers beat sat+pigment?
        res["driver_gain_vs_satpig"] = round(res["sat+pigment+drivers"] - res["sat+pigment"], 3)
        res["boot_3way_vs_2way"] = boot_delta(y, oofs["sat+pigment+drivers"], oofs["sat+pigment"], site_arr)
        best2 = max(res["satellite"], res["pigment"])
        res["boot_3way_vs_bestsingle"] = boot_delta(y, oofs["sat+pigment+drivers"],
                                                     oofs["satellite" if res["satellite"] >= res["pigment"] else "pigment"], site_arr)
        res["boot_3way_absolute"] = boot_absolute(y, oofs["sat+pigment+drivers"], site_arr)
        results["labels"][name] = res
        print(f"\n[{name}] pos={res['pos']} sat={res['satellite']} pig={res['pigment']} drv={res['drivers']} "
              f"| 2way(sat+pig)={res['sat+pigment']} 3way={res['sat+pigment+drivers']} "
              f"(driver gain {res['driver_gain_vs_satpig']:+.3f}, boot vs 2way {res['boot_3way_vs_2way']['ci95']} P={res['boot_3way_vs_2way']['P_gt0']})", flush=True)

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWROTE {OUT}")


if __name__ == "__main__":
    main()
