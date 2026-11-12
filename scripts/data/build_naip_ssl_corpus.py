#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SOTA lever 2: an unlabelled NAIP water-chip corpus for self-supervised pretraining of the Lite encoder
(lite_national_protocol.md N4; lead approval 2026-11-09; within the 10 GB M7 cap). No label is read.

Candidates are (site, DOQQ) pairs from the cached NAIP STAC items of the JRC-water WQP sites
(``interim/naip_catalog/wqp_sites``). Only coordinates are used, never sample values.

**Exclusion (coordinate manifest).** A candidate site is dropped if it lies within 2 km of, or shares a HydroLAKES
lake id (Lite 100-m rule) with, any site whose role is evaluation or buffer in either split:
- the 4-state split: val, test, test_time, excluded;
- the national split (national-1, and national-2, which has identical evaluation blocks): val, val_n, test,
  test_n, test_time, test_time_n, excluded.

The excluded sites, the rule and the counts are written to ``exclusion_manifest.json`` (+ a parquet of excluded
coordinates).

**Sampling.** Up to ``--n-chips`` pairs (default 25,000), at most ``--per-site`` DOQQs per site (different NAIP
years), seed 20261110, stratified by state.

**Store.** A separate store ``processed/naip_chips/ssl`` (manifest ``naip_ssl_corpus``). The labelled stores are
never touched.

    python -B scripts/data/build_naip_ssl_corpus.py [--n-chips 25000] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data import chips as C  # noqa: E402
from sentinel.data import naip as N  # noqa: E402
from sentinel.data.sources import stac as S  # noqa: E402
from sentinel.eval.splits import nearest_distance_km  # noqa: E402

log = logging.getLogger("naip_ssl")
EVAL_4STATE = ("val", "test", "test_time", "excluded")
EVAL_NATIONAL = ("val", "val_n", "test", "test_n", "test_time", "test_time_n", "excluded")


def eval_sites() -> pd.DataFrame:
    """Coordinates + HydroLAKES ids of every evaluation/buffer site of both splits (no label column read)."""
    from sentinel.models.lite import data as D
    from sentinel.models.lite import guard as G
    from sentinel.models.lite import national as NL

    cols = ["naip_chip_id", "site_id", "lat", "lon", "hylak_id"]
    out = []
    s4 = G.load_split()
    a = pd.read_parquet(D.labels_path(), columns=cols)
    a = a[a["naip_chip_id"].astype(str).map(s4["chip_role"]).isin(EVAL_4STATE)]
    out.append(a.assign(split="4state"))
    sn = NL.load_split()
    b = pd.read_parquet(NL.labels_path(), columns=cols)
    b = b[b["naip_chip_id"].astype(str).map(sn["chip_role"]).isin(EVAL_NATIONAL)]
    out.append(b.assign(split="national"))
    return pd.concat(out, ignore_index=True).drop_duplicates(["site_id", "split"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-chips", type=int, default=25000)
    ap.add_argument("--per-site", type=int, default=2)
    ap.add_argument("--radius-km", type=float, default=2.0)
    ap.add_argument("--dry-run", action="store_true", help="counts + exclusion manifest only, no chip reads")
    ap.add_argument("--read-workers", type=int, default=4)
    a = ap.parse_args()
    out = paths.ensure(paths.processed("naip_chips") / "ssl")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out / "build.log"), logging.StreamHandler()])
    jrc = pd.read_parquet(paths.interim("naip_matchups") / "site_jrc.parquet")
    jrc = jrc[jrc.jrc_water_ok.astype("boolean").fillna(False).astype(bool)]
    # site coordinates from the point key (lon_lat in the key; no WQP values read)
    xy = jrc["point_key"].str.split("_", expand=True).astype(float)
    sites = pd.DataFrame({"point_key": jrc["point_key"].to_numpy(), "lat": xy[0].to_numpy(), "lon": xy[1].to_numpy()})
    ev = eval_sites()
    from sentinel.models.lite import data as D

    hl = D.hydrolakes_ids(sites.rename(columns={"point_key": "site_id"}).assign(name="", site_class="", huc8=""))
    sites["hylak_id"] = sites["point_key"].map(hl)
    d = nearest_distance_km(sites["lat"], sites["lon"], ev["lat"], ev["lon"])
    ev_lakes = set(ev["hylak_id"].dropna().astype(int))
    excl = (d <= a.radius_km) | sites["hylak_id"].isin(ev_lakes)
    keep = sites[~excl.to_numpy()]
    man = {"rule": f"drop candidate sites within {a.radius_km} km of, or in the same HydroLAKES lake (100-m rule) as, "
                   f"any evaluation/buffer site: 4-state roles {EVAL_4STATE}; national roles {EVAL_NATIONAL}",
           "n_eval_sites": int(len(ev)), "n_eval_lakes": len(ev_lakes), "n_candidate_sites": int(len(sites)),
           "n_excluded_sites": int(excl.sum()), "n_kept_sites": int(len(keep))}
    ev[["split", "site_id", "lat", "lon", "hylak_id"]].to_parquet(out / "exclusion_eval_sites.parquet", index=False)
    items = {}
    for line in (paths.interim("naip_catalog") / "wqp_sites" / "items.jsonl").read_text().splitlines():
        it = json.loads(line)
        items[it["id"]] = it
    cat = pd.DataFrame([{"item_id": i["id"], "date": S.naip_acquisition_date(i),
                         "year": int(i["properties"].get("naip:year", 0)), "gsd": i["properties"].get("gsd"),
                         "state": i["properties"].get("naip:state"), "geometry": json.dumps(i["geometry"])}
                        for i in items.values()])
    si = N.sites_in_items(keep, cat).merge(cat[["item_id", "state"]], on="item_id")
    rs = np.random.default_rng(20261110)
    si = si.sample(frac=1.0, random_state=rs.integers(1 << 31)).groupby("point_key").head(a.per_site)
    n_states = si["state"].nunique()
    per_state = int(np.ceil(a.n_chips / max(1, n_states)))
    pick = si.groupby("state").head(per_state)
    if len(pick) < a.n_chips:                       # fill from the remaining pool
        rest = si.drop(pick.index)
        pick = pd.concat([pick, rest.head(a.n_chips - len(pick))])
    pick = pick.head(a.n_chips).merge(keep[["point_key", "lon", "lat"]], on="point_key")
    pick["request_key"] = pick.item_id + "|" + pick.point_key
    man.update({"n_candidate_pairs": int(len(si)), "n_selected_pairs": int(len(pick)), "n_states": int(n_states),
                "years": pick["naip_year"].value_counts().sort_index().astype(int).to_dict(), "seed": 20261110})
    (out / "exclusion_manifest.json").write_text(json.dumps(man, indent=1, default=str))
    log.info("SSL corpus plan: %s", {k: v for k, v in man.items() if k != "years"})
    if a.dry_run:
        return 0
    w = C.ChipWriter(out, layers=N.NAIP_LAYERS, dtype="uint8", semantics=N.NAIP_SEMANTICS)
    nx = N.NaipExtractor(w, out / "naip_scene_stats.parquet", workers=min(4, a.read_workers), logger=log.info)
    counts = nx.run(pick[["request_key", "item_id", "lon", "lat", "point_key"]], items)
    idx = C.consolidate(w.root)
    write_manifest("naip_ssl_corpus", files=[out / "index.parquet", out / "exclusion_manifest.json",
                                             out / "exclusion_eval_sites.parquet", out / "naip_scene_stats.parquet"],
                   sources=["interim/naip_catalog/wqp_sites (NAIP STAC, Planetary Computer)", "manifest:lite_labels",
                            "manifest:lite_labels_national", "manifest:hydrolakes"],
                   description="Unlabelled NAIP 4-band 256x256 chips at JRC-water WQP site locations (coordinates "
                               "only), excluding every Lite evaluation/buffer water body of both splits; for "
                               "self-supervised pretraining (M7 SOTA lever 2)",
                   license="NAIP public domain (USDA FSA)", synthetic=False,
                   stats={**man, "chips_ok": int((idx.status == "ok").sum()), "counts": counts})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
