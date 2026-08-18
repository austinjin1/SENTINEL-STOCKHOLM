#!/usr/bin/env python3
"""Step 1 for the Species-Mechanism test: build the exact leakage-free site
universe (same as train_species_health_leakagefree.py) and resolve each site's
coordinates to an NHDPlus v2 COMID via the EPA WATERS PointIndexing service.

Output:
  data/processed/biology/species_mechanism/site_table.parquet   (labels + meta)
  data/processed/biology/species_mechanism/site_comid.parquet   (site_id -> comid)

The COMID resolution is CHECKPOINTED: it re-reads site_comid.parquet on start and
only queries sites not yet resolved, so it is safe to re-run / resume.

MIT License -- Bryan Cheng, 2026
"""
from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import urllib.request, urllib.parse

PROJECT = Path("/home/bcheng/SENTINEL")
BIODATA_DIR = PROJECT / "data" / "processed" / "biology" / "usgs_biodata"
OUT_DIR = PROJECT / "data" / "processed" / "biology" / "species_mechanism"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SITE_TABLE = OUT_DIR / "site_table.parquet"
SITE_COMID = OUT_DIR / "site_comid.parquet"

SEED = 42
SPECIES_NAMES = [
    "Freshwater mussels (Unionidae)",
    "Mayflies (Ephemeroptera)",
    "Brook trout (Salvelinus fontinalis)",
    "Hellbender (Cryptobranchus alleganiensis)",
    "Freshwater pearl mussel (Margaritifera margaritifera)",
    "American eel (Anguilla rostrata)",
]
NUM_SPECIES = 6
TAXON_PATTERNS = {
    0: ["unionidae", "unionid", "lampsilis", "amblema", "elliptio", "quadrula",
        "fusconaia", "pleurobema", "villosa", "corbicula", "anodonta", "pyganodon",
        "utterbackia", "lasmigona", "strophitus"],
    1: ["ephemeroptera", "ephemerell", "hexagenia", "baetis", "heptageni",
        "leptophlebi", "caenis", "tricorythodes", "isonychia", "stenonema",
        "epeorus", "drunella", "paraleptophlebia", "ameletus", "cinygmula"],
    2: ["salvelinus fontinalis", "brook trout", "salvelinus"],
    3: ["cryptobranchus", "hellbender"],
    4: ["margaritifera", "pearl mussel"],
    5: ["anguilla rostrata", "american eel", "anguilla"],
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def split_of(site_id, seed=SEED):
    h = hashlib.sha256(f"{seed}:{site_id}".encode()).hexdigest()
    fold = int(h[:8], 16) % 10
    return "train" if fold < 7 else ("val" if fold < 9 else "test")


def build_site_table():
    inv = pd.read_parquet(BIODATA_DIR / "biodata_invertebrates.parquet")
    fish = pd.read_parquet(BIODATA_DIR / "biodata_fish.parquet")
    sites = pd.read_parquet(BIODATA_DIR / "biodata_sites.parquet")
    bio = pd.concat([inv, fish], ignore_index=True)
    bio["count"] = pd.to_numeric(bio["ResultMeasureValue"], errors="coerce").fillna(0)
    bio = bio[bio["count"] > 0].copy()
    tl = bio["SubjectTaxonomicName"].astype(str).str.lower()
    bio["species_group"] = -1
    for g, pats in TAXON_PATTERNS.items():
        bio.loc[tl.str.contains("|".join(pats), na=False), "species_group"] = g
    site_col = "MonitoringLocationIdentifier"
    matched = bio[bio["species_group"] >= 0]
    ss = matched.groupby([site_col, "species_group"])["count"].sum().reset_index()
    pivot = ss.pivot(index=site_col, columns="species_group", values="count").fillna(0.0)
    for g in range(NUM_SPECIES):
        if g not in pivot.columns:
            pivot[g] = 0.0
    pivot = pivot[[g for g in range(NUM_SPECIES)]]

    bio_nt = bio[bio["species_group"] < 0]
    comm_rich = bio_nt.groupby(site_col)["SubjectTaxonomicName"].nunique().rename("comm_richness_nontarget")
    comm_count = bio_nt.groupby(site_col)["count"].sum().rename("comm_count_nontarget")
    if "ActivityIdentifier" in bio.columns:
        n_events = bio.groupby(site_col)["ActivityIdentifier"].nunique().rename("n_events")
    else:
        n_events = bio.groupby(site_col)["ActivityStartDate"].nunique().rename("n_events")

    s = sites.drop_duplicates(subset=[site_col]).set_index(site_col)
    meta = pd.DataFrame(index=s.index)
    meta["lat"] = pd.to_numeric(s["LatitudeMeasure"], errors="coerce")
    meta["lon"] = pd.to_numeric(s["LongitudeMeasure"], errors="coerce")
    meta["elev"] = pd.to_numeric(s["VerticalMeasure/MeasureValue"], errors="coerce")
    meta["drainage"] = pd.to_numeric(s["DrainageAreaMeasure/MeasureValue"], errors="coerce")
    meta["huc2"] = s["_huc2"].astype(str) if "_huc2" in s.columns else "NA"

    tab = pivot.join(meta, how="inner")
    tab = tab.join(comm_rich, how="left").join(comm_count, how="left").join(n_events, how="left")
    tab = tab.dropna(subset=["lat", "lon"])
    tab[["comm_richness_nontarget", "comm_count_nontarget", "n_events"]] = \
        tab[["comm_richness_nontarget", "comm_count_nontarget", "n_events"]].fillna(0.0)
    tab["split"] = [split_of(sid) for sid in tab.index]
    tab.index.name = "site_id"
    return tab


def resolve_comid(lon, lat, timeout=25):
    """EPA WATERS PointIndexing -> NHDPlusV2 COMID (int) or None."""
    base = "https://ofmpub.epa.gov/waters10/PointIndexing.Service"
    q = {
        "pGeometry": f"POINT({lon} {lat})",
        "pGeometryMod": "WKT,SRSNAME=urn:ogc:def:crs:OGC::CRS84",
        "pResolution": "3",
        "optOutPathFlag": "FALSE",
        "optOutCU": "TRUE",
        "f": "json",
    }
    url = base + "?" + urllib.parse.urlencode(q)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            d = json.loads(r.read().decode())
        fl = d.get("output", {}).get("ary_flowlines")
        if fl:
            c = fl[0].get("comid")
            return int(c) if c is not None else None
    except Exception:
        return None
    return None


def main():
    if SITE_TABLE.exists():
        tab = pd.read_parquet(SITE_TABLE)
        log(f"Loaded cached site_table: {len(tab)} sites")
    else:
        tab = build_site_table()
        tab.to_parquet(SITE_TABLE)
        log(f"Built site_table: {len(tab)} sites -> {SITE_TABLE}")
    log(f"Split: train={int((tab.split=='train').sum())} "
        f"val={int((tab.split=='val').sum())} test={int((tab.split=='test').sum())}")

    # Load / init comid cache.
    if SITE_COMID.exists():
        cache = pd.read_parquet(SITE_COMID)
        done = set(cache["site_id"].tolist())
        rows = cache.to_dict("records")
        log(f"Resuming COMID cache: {len(done)} already resolved.")
    else:
        done, rows = set(), []

    todo = [sid for sid in tab.index if sid not in done]
    log(f"To resolve: {len(todo)} sites")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    t0 = time.time()

    def _one(sid):
        lat = float(tab.loc[sid, "lat"]); lon = float(tab.loc[sid, "lon"])
        return {"site_id": sid, "lat": lat, "lon": lon, "comid": resolve_comid(lon, lat)}

    n_done = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(_one, sid): sid for sid in todo}
        for fut in as_completed(futs):
            rows.append(fut.result())
            n_done += 1
            if n_done % 200 == 0 or n_done == len(todo):
                pd.DataFrame(rows).to_parquet(SITE_COMID)
                rate = n_done / (time.time() - t0)
                n_ok = sum(1 for r in rows if r["comid"] is not None)
                log(f"  {n_done}/{len(todo)} resolved | {n_ok} have COMID | "
                    f"{rate:.1f}/s | eta {(len(todo)-n_done)/max(rate,1e-6)/60:.1f}min")
    pd.DataFrame(rows).to_parquet(SITE_COMID)
    dfc = pd.DataFrame(rows)
    log(f"DONE. {int(dfc['comid'].notna().sum())}/{len(dfc)} sites resolved to COMID.")


if __name__ == "__main__":
    main()
