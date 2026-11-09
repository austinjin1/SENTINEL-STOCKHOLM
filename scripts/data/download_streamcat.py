#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""StreamCat catchment/watershed attributes for every COMID-linked SENTINEL site.

Reads COMIDs from raw('nldi')/comid_links.parquet (or processed('site_network')/nodes.parquet),
POSTs batches of COMIDs x curated metric groups to the keyless StreamCat API (cached, sequential,
<= 1,000 requests/h), and writes processed('catchment')/streamcat_sites.parquet (one row per site),
streamcat_dictionary.csv (column -> metric, AOI, year, units, description, dataset, source) and
streamcat_pctfull.parquet (data-coverage QC columns).

    python -B scripts/data/download_streamcat.py --batch 500
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data.sources import streamcat as SC  # noqa: E402

log = logging.getLogger("streamcat")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batch", type=int, default=500)
    a = ap.parse_args()
    raw = paths.raw("streamcat")
    out_dir = paths.processed("catchment")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(raw / "download_streamcat.log"), logging.StreamHandler()],
    )
    nodes_p = paths.PROCESSED / "site_network" / "nodes.parquet"
    links = pd.read_parquet(nodes_p if nodes_p.exists() else paths.RAW / "nldi" / "comid_links.parquet")
    links = links[["site_key", "comid", "comid_method"]]
    comids = sorted(links["comid"].dropna().astype(str).unique(), key=int)
    log.info(
        "%d sites, %d with COMID, %d unique COMIDs", len(links), links["comid"].notna().sum(), len(comids)
    )
    client = SC.StreamCatClient(raw / "cache")

    # metadata: variable info (units/descriptions) + DB version
    vi_p, cl_p = raw / "variable_info.csv", raw / "changelog.json"
    if not vi_p.exists():
        vi_p.write_text(client.get_text(SC.VARINFO_URL))
    varinfo = pd.read_csv(vi_p)
    if not cl_p.exists():
        cl_p.write_text(json.dumps(json.loads(client.get_text(SC.CHANGELOG_URL)), indent=1))
    changelog = json.loads(cl_p.read_text())
    db_version = (changelog.get("items") or [{}])[0]

    wide = pd.DataFrame({"comid": comids}).set_index("comid")
    missing_cols, failed_metrics = [], []
    curated = []
    for group, aoi, names in SC.CURATED:
        # probe the group on the first batch; a deterministic API/SQL error -> find the bad names
        url, data = SC.metrics_request(names, comids[: a.batch], aoi)
        try:
            client.post(url, data)
        except RuntimeError as e:
            if "ORA-" not in str(e) and "SQL" not in str(e):
                raise  # transient outage (5xx/timeouts): stop; a re-run resumes from the cache
            log.warning("group %s failed (%s); testing metrics one by one", group, str(e)[:120])
            ok_names = []
            for n in names:
                try:
                    client.post(*SC.metrics_request([n], comids[:5], aoi))
                    ok_names.append(n)
                except RuntimeError as e2:
                    if "ORA-" not in str(e2) and "SQL" not in str(e2):
                        raise
                    failed_metrics.append({"group": group, "metric": n, "aoi": aoi, "error": str(e2)[:300]})
                    log.warning("  metric %s (%s) rejected by the API", n, aoi)
            names = ok_names
        curated.append((group, aoi, names))
    for group, aoi, names in curated:
        if not names:
            continue
        frames = []
        for b in range(0, len(comids), a.batch):
            chunk = comids[b : b + a.batch]
            url, data = SC.metrics_request(names, chunk, aoi)
            rec = client.post(url, data)
            frames.append(SC.parse_metrics(rec["body"]))
        df = pd.concat(frames, ignore_index=True).drop_duplicates("comid").set_index("comid")
        exp = SC.expected_columns(names, aoi)
        miss = [c for c in exp if c not in df.columns]
        missing_cols += miss
        keep = [c for c in df.columns if c not in wide.columns]
        wide = wide.join(df[keep], how="left")
        log.info(
            "group %-18s aoi=%-16s %d metrics -> %d rows, %d cols, missing cols=%s; remaining quota=%s",
            group,
            aoi,
            len(names),
            len(df),
            len(keep),
            miss,
            client.stats["ratelimit_remaining"],
        )
    wide = wide.reset_index()
    pct_cols = [c for c in wide.columns if "pctfull" in c]
    area_cols = [c for c in ["catareasqkm", "wsareasqkm"] if c in wide.columns]
    metric_cols = [
        c for g, aoi, names in curated for c in SC.expected_columns(names, aoi) if c in wide.columns
    ]
    metric_cols = list(dict.fromkeys(metric_cols))
    found = set(wide.loc[wide[metric_cols].notna().any(axis=1), "comid"])

    # optional NHDPlus V2 VAA reach attributes (slope, order, drainage area)
    vaa_p = paths.RAW / "nhdplus_vaa" / "nhdplus_vaa_subset.parquet"
    extra = []
    if vaa_p.exists():
        vaa = pd.read_parquet(vaa_p, columns=["comid", "slope", "streamorde", "totdasqkm", "lengthkm"])
        vaa["comid"] = vaa["comid"].astype("int64").astype(str)
        vaa = vaa.rename(
            columns={
                "slope": "nhd_slope",
                "streamorde": "nhd_streamorde",
                "totdasqkm": "nhd_totdasqkm",
                "lengthkm": "nhd_lengthkm",
            }
        )
        vaa.loc[vaa["nhd_slope"] < 0, "nhd_slope"] = float("nan")  # -9998 = not computed
        wide = wide.merge(vaa, on="comid", how="left")
        extra = ["nhd_slope", "nhd_streamorde", "nhd_totdasqkm", "nhd_lengthkm"]

    sites = links.merge(wide[["comid"] + area_cols + metric_cols + extra], on="comid", how="left")
    sites["streamcat_found"] = sites["comid"].isin(found)
    out = out_dir / "streamcat_sites.parquet"
    sites.to_parquet(out, index=False, compression="zstd")
    links.merge(wide[["comid"] + pct_cols], on="comid", how="left").to_parquet(
        out_dir / "streamcat_pctfull.parquet", index=False, compression="zstd"
    )
    dic = SC.column_dictionary(area_cols + metric_cols, varinfo)
    for c, desc, unit in [
        ("nhd_slope", "NHDPlus V2 VAA flowline slope (m/m; -9998 -> NaN)", "m/m"),
        ("nhd_streamorde", "NHDPlus V2 VAA modified Strahler stream order", "order"),
        ("nhd_totdasqkm", "NHDPlus V2 VAA total upstream drainage area", "km2"),
        ("nhd_lengthkm", "NHDPlus V2 VAA flowline length", "km"),
    ]:
        if c in sites.columns:
            dic.loc[len(dic)] = {
                "column": c,
                "metric": c,
                "aoi": "reach",
                "year": None,
                "description": desc,
                "units": unit,
                "category": "Network",
                "dataset": "NHDPlusV2 VAA",
                "source_name": "nhdplusTools VAA (HydroShare 6092c8a6...)",
                "source_url": None,
            }
    nni = dic["column"].str.match(r"^[np]_(tin|ff|lw|hw|usgsww)_")
    dic.loc[nni, "description"] = (
        dic.loc[nni, "description"].fillna("")
        + " [total mass in AOI per year; divide by catareasqkm/wsareasqkm for intensity]"
    )
    dic.to_csv(out_dir / "streamcat_dictionary.csv", index=False)
    stats = {
        "n_sites": int(len(sites)),
        "n_sites_with_comid": int(sites["comid"].notna().sum()),
        "n_sites_streamcat_found": int(sites["streamcat_found"].sum()),
        "n_unique_comids": len(comids),
        "n_comids_found": len(found),
        "n_attribute_columns": len(area_cols) + len(metric_cols) + len(extra),
        "n_streamcat_metric_columns": len(metric_cols),
        "missing_columns": missing_cols,
        "failed_metrics": failed_metrics,
        "groups": {g: {"aoi": aoi, "metrics": n} for g, aoi, n in curated},
        "excluded": sorted(SC.EXCLUDED),
        "streamcat_db_version": db_version,
        "batch": a.batch,
        "http": client.stats,
        "dictionary_without_units": dic.loc[dic["units"].isna(), "column"].tolist(),
    }
    write_manifest(
        "streamcat",
        files=[
            out,
            out_dir / "streamcat_dictionary.csv",
            out_dir / "streamcat_pctfull.parquet",
            raw / "variable_info.csv",
            raw / "changelog.json",
        ],
        sources=[SC.METRICS_URL + " (POST)", SC.VARINFO_URL, SC.CHANGELOG_URL],
        description="Curated EPA StreamCat catchment (cat) / watershed (ws) / riparian (rp100) attributes "
        "joined to SENTINEL sites by NHDPlus V2 COMID, plus NHDPlus VAA reach attributes.",
        license="US public domain (EPA StreamCat; NHDPlus V2)",
        stats=stats,
    )
    log.info("done: %s", {k: v for k, v in stats.items() if k != "groups"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
