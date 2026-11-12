# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Counts-only check for the SOTA lever 1(a), ±7-day Secchi window (lite_national_protocol.md N4/N5). Nothing is
built and nothing is downloaded.

It uses the cached inputs of ``build_naip_matchups.py --national``: the processed WQP surface samples, the cached
JRC site flags and the cached NAIP STAC items. For Secchi it counts samples, (site, DOQQ) pairs and sites matched at
|Δt| ≤ 1 d (current rule) and ≤ 7 d (proposed), and the pairs that would need a NEW NAIP chip. Turbidity and TSS
are listed at ±1 d for reference only. The disk estimate uses the mean bytes per chip of the current NAIP store.
Labels are counted, never inspected. Roles are assigned later by the national-2 rule, which keeps every evaluation
water body out of training.

    python -B scripts/eval/national_lite_counts.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sentinel.core import paths  # noqa: E402
from sentinel.data import matchups as M  # noqa: E402
from sentinel.data import naip as N  # noqa: E402
from sentinel.data.sources import stac as S  # noqa: E402


def main() -> int:
    res, stations, _ = M.load_wqp("processed")
    smp = M.wqp_surface_samples(res, stations)
    del res
    smp = smp[smp.target.isin(("secchi", "turbidity", "tss")) & smp.date.between("2010-01-01", "2024-01-31")]
    smp = smp[smp.lat.between(17, 72) & smp.lon.between(-170, -64)].copy()
    smp["point_key"] = [M.point_key(lo, la) for lo, la in zip(smp.lon, smp.lat)]
    jrc = pd.read_parquet(paths.interim("naip_matchups") / "site_jrc.parquet")
    ok_pts = set(jrc.loc[jrc.jrc_water_ok.astype("boolean").fillna(False).astype(bool), "point_key"])
    smp = smp[smp.point_key.isin(ok_pts)]
    items = {}
    f = paths.interim("naip_catalog") / "wqp_sites" / "items.jsonl"
    for line in f.read_text().splitlines():
        it = json.loads(line)
        items[it["id"]] = it
    cat = pd.DataFrame([{"item_id": i["id"], "date": S.naip_acquisition_date(i),
                         "year": int(i["properties"].get("naip:year", 0)), "gsd": i["properties"].get("gsd"),
                         "geometry": json.dumps(i["geometry"])} for i in items.values()])
    sites = smp.drop_duplicates("point_key")[["point_key", "site_id", "lat", "lon", "site_class", "state"]]
    si = N.sites_in_items(sites, cat)
    mt = smp.merge(si, on="point_key")
    mt["dt"] = (mt.date - mt.naip_date).dt.days.abs()
    store = pd.read_parquet(paths.processed("naip_chips") / "wqp" / "index.parquet")
    have = set(store["request_key"].astype(str))
    shard_bytes = sum(p.stat().st_size for p in (paths.processed("naip_chips") / "wqp" / "shards").glob("*.npz"))
    per_chip = shard_bytes / max(1, int((store["status"] == "ok").sum()))
    out = {"per_chip_bytes": per_chip}
    for t in ("secchi", "turbidity", "tss"):
        for w in ((1, 7) if t == "secchi" else (1,)):
            m = mt[(mt.target == t) & (mt.dt <= w)]
            pairs = m.drop_duplicates(["point_key", "item_id"])
            keys = pairs.item_id + "|" + pairs.point_key
            new = int((~keys.isin(have)).sum())
            out[f"{t}_pm{w}d"] = {"samples": int(len(m)), "site_doqq_pairs": int(len(pairs)),
                                  "sites": int(m.point_key.nunique()), "lake_samples": int((m.site_class == "lake_reservoir").sum()),
                                  "states": int(m.state.nunique()), "pairs_needing_new_chip": new,
                                  "new_chip_gb_est": round(new * per_chip / 1e9, 2)}
    d = paths.run_dir("lite", "counts_national2")
    (d / "counts.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
