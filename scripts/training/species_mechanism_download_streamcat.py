#!/usr/bin/env python3
"""Step 2: download REAL StreamCat watershed/catchment environmental metrics for
every resolved COMID (from species_mechanism_build_sites.py) and cache them.

StreamCat (EPA) provides landscape metrics summarised over each NHDPlusV2
catchment (local, 'cat') and the full upstream watershed ('ws'). These are
genuine environmental predictors, independent of the biological survey counts:
  * land cover (NLCD-derived): forest / urban / crop / wetland / open-water /
    grass / shrub / imperviousness
  * climate (PRISM): mean/max/min temperature, precipitation
  * hydrology: runoff, base-flow index
  * soils (STATSGO): clay/sand %, organic matter, permeability, K-factor,
    hydraulic conductivity, water-table depth, bedrock depth
  * terrain: elevation
  * anthropogenic stressors: road density, dam density, population density,
    fertilizer N, NPDES/TRI/Superfund discharge densities

We keep the '*cat' (local catchment) and '*ws' (upstream watershed) variants,
which together describe the local + integrated environment of each site.

API: https://api.epa.gov/StreamCat/streams/metrics?name=<base1,base2,...>&comid=<c1,c2,...>
The API returns every column matching each base name (e.g. base 'elev' returns
elevcat + elevws). We request in batches of COMIDs.

Output: data/processed/biology/species_mechanism/streamcat.parquet
Checkpointed: resumes from any partial cache.

MIT License -- Bryan Cheng, 2026
"""
from __future__ import annotations
import json, time
from pathlib import Path
import pandas as pd
import urllib.request, urllib.parse

PROJECT = Path("/home/bcheng/SENTINEL")
OUT_DIR = PROJECT / "data" / "processed" / "biology" / "species_mechanism"
SITE_COMID = OUT_DIR / "site_comid.parquet"
STREAMCAT = OUT_DIR / "streamcat.parquet"

# Base metric names verified live against the StreamCat API (2026-07).
# Each expands to <name>cat and <name>ws (+ rp100 riparian variants we drop).
METRIC_BASES = [
    # land cover (2019 NLCD)
    "pctforest2019", "pctconif2019", "pctdecid2019", "pctmxfst2019",
    "pcturbop2019", "pcturblo2019", "pcturbmd2019", "pcturbhi2019",
    "pctcrop2019", "pcthay2019", "pctgrs2019", "pctshrb2019",
    "pctow2019", "pctwdwet2019", "pcthbwet2019", "pctbl2019", "pctimp2019",
    # climate (PRISM 1981-2010 normals)
    "precip8110", "tmean8110", "tmax8110", "tmin8110",
    # hydrology
    "runoff", "bfi",
    # soils / geology
    "clay", "sand", "om", "perm", "kffact", "hydrlcond", "wtdep", "rckdep", "al2o3",
    # terrain
    "elev",
    # anthropogenic stressors
    "rddens", "damdens", "popden2010", "fert", "cbnf",
    "npdesdens", "tridens", "superfunddens",
]

BATCH = 40           # comids per request
BASE_URL = "https://api.epa.gov/StreamCat/streams/metrics"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fetch(comids, timeout=60):
    q = {"name": ",".join(METRIC_BASES), "comid": ",".join(str(c) for c in comids)}
    url = BASE_URL + "?" + urllib.parse.urlencode(q)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                d = json.loads(r.read().decode())
            return d.get("items", [])
        except Exception as e:
            if attempt == 3:
                log(f"  batch failed after retries: {e}")
                return []
            time.sleep(2 * (attempt + 1))
    return []


def main():
    sc = pd.read_parquet(SITE_COMID)
    sc = sc[sc["comid"].notna()].copy()
    sc["comid"] = sc["comid"].astype(int)
    comids = sorted(sc["comid"].unique().tolist())
    log(f"{len(sc)} sites, {len(comids)} unique COMIDs to query StreamCat.")

    if STREAMCAT.exists():
        cache = pd.read_parquet(STREAMCAT)
        have = set(cache["comid"].astype(int).tolist())
        rows = cache.to_dict("records")
        log(f"Resuming: {len(have)} comids cached.")
    else:
        have, rows = set(), []

    todo = [c for c in comids if c not in have]
    log(f"To fetch: {len(todo)} comids in batches of {BATCH}.")
    t0 = time.time()
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        items = fetch(batch)
        for it in items:
            # Drop riparian buffer (rp100) columns; keep cat + ws.
            row = {k: v for k, v in it.items() if not k.endswith("rp100")}
            rows.append(row)
        n = i + len(batch)
        if (i // BATCH) % 10 == 0 or n >= len(todo):
            pd.DataFrame(rows).to_parquet(STREAMCAT)
            rate = n / (time.time() - t0 + 1e-9)
            log(f"  {n}/{len(todo)} comids | {len(rows)} rows | "
                f"{rate:.1f}/s | eta {(len(todo)-n)/max(rate,1e-6)/60:.1f}min")
    df = pd.DataFrame(rows)
    df.to_parquet(STREAMCAT)
    # Coverage report.
    metric_cols = [c for c in df.columns if c != "comid"]
    nonnull = {c: float(df[c].notna().mean()) for c in metric_cols}
    log(f"DONE. {len(df)} comid rows, {len(metric_cols)} metric columns.")
    worst = sorted(nonnull.items(), key=lambda kv: kv[1])[:8]
    log(f"Lowest-coverage columns: {worst}")


if __name__ == "__main__":
    main()
