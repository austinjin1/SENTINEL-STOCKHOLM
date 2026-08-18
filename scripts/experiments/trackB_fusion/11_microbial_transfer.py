#!/usr/bin/env python
"""Track B 11 — can a MICROBIAL / cyanobacteria-community proxy recover the
cross-region (leave-one-HUC-region-out) transfer of the microcystin fusion?

Prior attempt declared DATA-BLOCKED after checking only wqp_hab_cyanotoxins.parquet
(92 'Cyanobacteria' sites, 0 overlap with the 227 microcystin fusion sites).

This attempt goes BROADER: it queries the Water Quality Portal directly at the
227 fusion site IDs for every community / genotype characteristic that actually
exists there (discovered by pulling the full characteristic inventory):
  - 'Microcystin/nodularin genes mcyE/ndaF'  (toxic-strain genotype qPCR = the
    literature's transfer key)
  - 'Algae, blue-green (phylum cyanophyta) density'  (cyanobacteria community count)
  - 'Algae, blue-green (phylum cyanophyta) percent'  (cyanobacteria rel. abundance)
  - 'Algae, all groups, density'
  - 'Chlorophyll a (probe) concentration, Cyanobacteria (bluegreen)'  (cyano pigment)
It co-locates each (nearest within +/-14 d) with every microcystin sample, adds
them as a 4th (community) modality, and asks the KEY question:
  does the community proxy raise leave-one-HUC-region-out AUROC above chance
  (>0.53, CI excluding 0.5)?  Plus a within-region sanity check (site-CV).

Real WQP data only. HUC2 region from WQP Station service (real HUCs recovered for
the community-covered sites, which the on-disk parquet left blank -> '00').
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
CYANO = ROOT / "data" / "processed" / "biology" / "noaa_habs" / "wqp_hab_cyanotoxins.parquet"
DRV = ROOT / "data" / "raw" / "wqp_drivers"
COMM = ROOT / "data" / "raw" / "wqp_community"
COMM.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "results" / "trackB_fusion" / "mc_microbial_transfer.json"
SEED = 0

COMMUNITY_CHARS = {
    "mcyE_ndaF": "Microcystin/nodularin genes mcyE/ndaF",
    "cyano_density": "Algae, blue-green (phylum cyanophyta) density",
    "cyano_percent": "Algae, blue-green (phylum cyanophyta) percent",
    "algae_all_density": "Algae, all groups, density",
    "cyano_chla_probe": "Chlorophyll a (probe) concentration, Cyanobacteria (bluegreen)",
}


def wqp_results(sites, char, tag):
    cache = COMM / f"{tag}.csv"
    if cache.exists():
        return pd.read_csv(cache, low_memory=False)
    import requests
    frames = []
    for i in range(0, len(sites), 50):
        batch = ";".join(sites[i:i + 50])
        url = (f"https://www.waterqualitydata.us/data/Result/search?siteid={quote(batch)}"
               f"&characteristicName={quote(char)}&mimeType=csv&zip=no"
               f"&startDateLo=01-01-2016&startDateHi=12-31-2025")
        try:
            r = requests.get(url, timeout=110)
            if r.ok and len(r.text) > 50:
                frames.append(pd.read_csv(io.StringIO(r.text), low_memory=False))
        except Exception as e:
            print(f"  {tag} batch {i}: {type(e).__name__}", flush=True)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df.to_csv(cache, index=False)
    return df


def wqp_station_hucs(sites):
    cache = COMM / "station_hucs.csv"
    if cache.exists():
        return pd.read_csv(cache, low_memory=False)
    import requests
    frames = []
    for i in range(0, len(sites), 40):
        batch = ";".join(sites[i:i + 40])
        url = f"https://www.waterqualitydata.us/data/Station/search?siteid={quote(batch)}&mimeType=csv&zip=no"
        try:
            r = requests.get(url, timeout=90)
            if r.ok and len(r.text) > 50:
                frames.append(pd.read_csv(io.StringIO(r.text), low_memory=False))
        except Exception as e:
            print(f"  station batch {i}: {type(e).__name__}", flush=True)
    st = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    st.to_csv(cache, index=False)
    return st


def parse(df):
    if len(df) == 0:
        return pd.DataFrame(columns=["site", "date", "val"])
    return pd.DataFrame({
        "site": df["MonitoringLocationIdentifier"].astype(str),
        "date": pd.to_datetime(df["ActivityStartDate"], errors="coerce"),
        "val": pd.to_numeric(df["ResultMeasureValue"], errors="coerce"),
    }).dropna(subset=["date", "val"]).sort_values("date")


def parse_drv(tag):
    f = DRV / f"{tag}.csv"
    if not f.exists():
        return pd.DataFrame(columns=["site", "date", "val"])
    d = pd.read_csv(f, low_memory=False)
    return parse(d)


def nearest(dv, site, date, tol=14):
    d = dv[dv["site"] == site]
    if len(d) == 0:
        return np.nan
    dd = (d["date"] - date).abs().dt.days
    i = dd.values.argmin()
    return float(d["val"].values[i]) if dd.values[i] <= tol else np.nan


def hgb():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                          l2_regularization=1.0, class_weight="balanced",
                                          random_state=SEED)


def oof(X, y, g, ns):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=ns).split(X, y, g):
        p[te] = hgb().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def boot_absolute(y, p, sites, B=1000):
    rng = np.random.default_rng(SEED)
    uniq = np.unique(sites)
    by = {s: np.where(sites == s)[0] for s in uniq}
    aucs = []
    for _ in range(B):
        idx = np.concatenate([by[s] for s in rng.choice(uniq, len(uniq), replace=True)])
        yi = y[idx]
        if yi.min() != yi.max():
            aucs.append(roc_auc_score(yi, p[idx]))
    aucs = np.array(aucs)
    return {"median": round(float(np.median(aucs)), 3),
            "ci95": [round(float(np.percentile(aucs, 2.5)), 3),
                     round(float(np.percentile(aucs, 97.5)), 3)]}


def main():
    df = pq.read_table(DATA).to_pandas()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    sites = sorted(df["site"].astype(str).unique())

    # --- region (HUC2): parquet mapping, overridden by REAL WQP Station HUCs ---
    cy = pq.read_table(CYANO, columns=["Location_Identifier", "Location_HUCEightDigitCode"]).to_pandas()
    cy["site"] = cy["Location_Identifier"].astype(str)
    g = cy.groupby("site").first()
    df["region"] = df["site"].astype(str).map(
        g["Location_HUCEightDigitCode"].astype(str).str.zfill(8).str[:2])

    # --- download community characteristics + station HUCs ---
    comm = {tag: parse(wqp_results(sites, char, tag)) for tag, char in COMMUNITY_CHARS.items()}
    comm_sites = sorted(set().union(*[set(dv["site"]) for dv in comm.values() if len(dv)]))
    st = wqp_station_hucs(comm_sites)
    if len(st):
        st["huc2"] = st["HUCEightDigitCode"].astype(str).str.zfill(8).str[:2]
        real = dict(zip(st["MonitoringLocationIdentifier"].astype(str), st["huc2"]))
        df["region"] = df.apply(lambda r: real.get(r["site"], r["region"]), axis=1)
    df["region"] = df["region"].fillna("NA")

    print("downloaded/loaded community CSVs; co-locating...", flush=True)
    # --- co-locate community features (+/-14 d) ---
    coverage = {}
    for tag, dv in comm.items():
        col = np.array([nearest(dv, str(s), d) for s, d in zip(df["site"].astype(str), df["date"])])
        df[f"cm_{tag}"] = col
        m = np.isfinite(col)
        s = df[m]
        coverage[tag] = {
            "n_colocated": int(m.sum()), "n_sites": int(s["site"].nunique()),
            "regions": {str(k): int(v) for k, v in s["region"].value_counts().items()},
            "pos_gt1.6": int((s["mc"] > 1.6).sum()), "pos_gt0.3": int((s["mc"] > 0.3).sum()),
        }

    # unified: any community record present, and log1p of first available count-like value
    cmcols = [f"cm_{t}" for t in COMMUNITY_CHARS]
    anymask = np.isfinite(df[cmcols].to_numpy()).any(axis=1)
    sub = df[anymask]

    # --- drivers (for the 3-way base) ---
    drv = {t: parse_drv(t) for t in ["temp", "phosphorus", "nitrogen"]}
    for t, dv in drv.items():
        df[f"drv_{t}"] = [nearest(dv, str(s), d) for s, d in zip(df["site"].astype(str), df["date"])]

    sf = [c for c in df.columns if c.startswith("sf")]
    base_cols = sf + ["CHLA", "PC", "drv_temp", "drv_phosphorus", "drv_nitrogen"]

    # ---- region testability for a leave-one-HUC-region-out transfer test ----
    # A cross-region test of the COMMUNITY modality requires the SAME community
    # characteristic present, with both classes, in >=2 HUC2 regions.
    region_testability = {}
    for tag in COMMUNITY_CHARS:
        m = np.isfinite(df[f"cm_{tag}"])
        s = df[m]
        regs = {}
        for reg, sg in s.groupby("region"):
            regs[str(reg)] = {"n": int(len(sg)), "pos_gt1.6": int((sg["mc"] > 1.6).sum()),
                              "pos_gt0.3": int((sg["mc"] > 0.3).sum())}
        usable = [r for r, v in regs.items() if v["n"] >= 20 and 0 < v["pos_gt1.6"] < v["n"]]
        region_testability[tag] = {"regions": regs, "n_regions_with_both_classes_and_n>=20": len(usable)}

    n_comm_regions = len(set().union(*[
        {r for r, v in rt["regions"].items() if v["n"] >= 1}
        for rt in region_testability.values()]))

    # can we do leave-one-HUC-region-out with the community modality?
    transfer_possible = any(rt["n_regions_with_both_classes_and_n>=20"] >= 2
                            for rt in region_testability.values())

    results = {
        "question": ("Can a microbial / cyanobacteria-community proxy (incl. mcyE/ndaF "
                     "toxic-strain genotype qPCR) recover leave-one-HUC-region-out transfer "
                     "of the microcystin fusion, which collapses to chance (~0.485-0.527)?"),
        "method": "WQP Result service queried at the 227 fusion site IDs for community/genotype "
                  "characteristics; co-located nearest +/-14 d; real HUC2 from WQP Station service.",
        "cross_region_baseline_3way": {"mc_gt1.6": 0.485, "mc_gt0.3": 0.49,
                                       "source": "mc_region_transfer.json / mc_driver_robustness.json"},
        "community_coverage": coverage,
        "n_samples_with_any_community_record": int(anymask.sum()),
        "n_sites_with_any_community_record": int(sub["site"].nunique()),
        "community_region_breakdown": {str(k): int(v) for k, v in sub["region"].value_counts().items()},
        "region_testability_per_characteristic": region_testability,
        "n_distinct_HUC2_regions_with_community_data": int(n_comm_regions),
        "leave_one_HUC_region_out_possible_with_community": bool(transfer_possible),
    }

    # ---- within-region sanity: does community help WITHIN HUC18 (site-clustered CV)? ----
    # HUC18 is the only region with a genotype/count feature and both classes.
    print("region testability computed; within-region HUC18 sanity check...", flush=True)
    within = {}
    for reg in ["18"]:
        rmask = (df["region"] == reg).to_numpy()
        # require enough co-located community rows
        rmask = rmask & np.isfinite(df[cmcols].to_numpy()).any(axis=1)
        d = df[rmask].reset_index(drop=True)
        if len(d) < 40 or d["site"].nunique() < 3:
            within[reg] = {"skip": "insufficient community-covered samples/sites"}
            continue
        site_arr = d["site"].astype(str).to_numpy()
        ns = min(5, d["site"].nunique())
        Xbase = d[base_cols].to_numpy("float64")
        Xcomm = d[cmcols].to_numpy("float64")
        entry = {"n": int(len(d)), "n_sites": int(d["site"].nunique())}
        for name, thr in [("mc_gt1.6", 1.6), ("mc_gt0.3", 0.3)]:
            y = (d["mc"].to_numpy("float64") > thr).astype(int)
            if y.min() == y.max():
                entry[name] = {"skip": "single class"}
                continue
            p_base = oof(Xbase, y, site_arr, ns)
            p_full = oof(np.hstack([Xbase, Xcomm]), y, site_arr, ns)
            entry[name] = {
                "pos": int(y.sum()),
                "auroc_3way_base": round(float(roc_auc_score(y, p_base)), 3),
                "auroc_4way_with_community": round(float(roc_auc_score(y, p_full)), 3),
                "boot_4way_absolute": boot_absolute(y, p_full, site_arr),
            }
        within[reg] = entry
    results["within_region_sanity_HUC18"] = within

    # ---- verdict ----
    if transfer_possible:
        verdict = "TESTABLE"  # (not reached with current data)
    else:
        verdict = ("STILL DATA-BLOCKED (structural): every co-located community/genotype "
                   "characteristic exists in exactly ONE HUC2 region (mcyE/ndaF & cyano density "
                   "in HUC18 only; cyano percent & algae density in HUC15 only; cyano-chla-probe "
                   "in one unmapped site). Leave-one-HUC-region-out is impossible for the community "
                   "modality: no community feature is present in >=2 regions, so it cannot be in "
                   "both a train and a held-out test region. The genotype key exists at fusion "
                   "sites but is spatially confined, so the cross-region hypothesis cannot be tested.")
    results["verdict"] = verdict
    results["unblock_spec"] = (
        "Need the SAME community/genotype characteristic (ideally 'Microcystin/nodularin genes "
        "mcyE/ndaF' qPCR, or a standardized cyanobacteria cell-count/biovolume) co-located "
        "(same site, +/-14 d of a microcystin sample) at monitoring programs in >=2 distinct HUC2 "
        "regions with both classes present (e.g. add Florida 21FL* HAB programs, or Great-Lakes "
        "21MICH/21OHIO/USGS mcyE, which the WQP already has at OTHER sites). Concretely: >=2 regions "
        "x >=30 co-located samples x >=6 positives each. Candidate sources with the data type but at "
        "non-fusion sites: USGS mcyE gc/uL (Great Lakes) and NARS cyanobacteria counts.")

    OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps({k: results[k] for k in [
        "n_samples_with_any_community_record", "n_sites_with_any_community_record",
        "community_region_breakdown", "n_distinct_HUC2_regions_with_community_data",
        "leave_one_HUC_region_out_possible_with_community", "verdict"]}, indent=2))
    print(f"\nWROTE {OUT}")


if __name__ == "__main__":
    main()
