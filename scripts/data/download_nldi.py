#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Link every SENTINEL site to an NHDPlus V2 COMID (NLDI) and build the flow-connection edge list.

Steps (all HTTP cached under raw('nldi')/cache -> resume-safe; <= 4 concurrent requests):
  1. COMID lookups: USGS -> ``nwissite/USGS-<no>`` (reach-indexed) AND ``comid/position`` (point in
     catchment) for comparison; chosen COMID = nwissite if available else position.
     NEON -> position lookup at the sensor-station / buoy coordinates.
  2. Main-stem navigation UM and DM (``distance`` km, default 500) with data source ``nwissite``
     for every linked surface-water site (USGS by nwissite id; others by COMID).
  3. NHDPlus V2 VAA table (HydroShare nhdplusVAA.parquet, md5-checked) for along-channel distance:
     position = pathlength + lengthkm * measure/100 (km to terminal outlet).
  4. Local VAA downstream-main-path traversal as a cross-check and to connect sites that NLDI
     nwissite navigation cannot return (position-linked USGS sites, NEON<->NEON).
Outputs: processed('site_network')/{nodes,edges}.parquet, interim('nldi')/navigation_raw.parquet.

    python -B scripts/data/download_nldi.py --workers 4 --distance 500
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data import sites as S  # noqa: E402
from sentinel.data.sources import nldi as N  # noqa: E402

log = logging.getLogger("nldi")

VAA_URL = (
    "https://www.hydroshare.org/resource/6092c8a62fac45be97a09bfd0b0bf726/data/contents/nhdplusVAA.parquet"
)
VAA_MD5 = "5691af50ffc1c1d66960a9dae690dc6e"  # HydroShare file checksum (verified 2026-10-22)
VAA_COLS = [
    "comid",
    "hydroseq",
    "dnhydroseq",
    "uphydroseq",
    "levelpathi",
    "terminalpa",
    "pathlength",
    "lengthkm",
    "totdasqkm",
    "streamorde",
    "slope",
    "pathtimema",
    "reachcode",
    "ftype",
    "fcode",
    "divergence",
    "startflag",
    "terminalfl",
    "vpuid",
    "gnis_name",
]
NO_NAV_TYPES = ("GW", "SB", "AT")  # groundwater / subsurface / atmosphere sites are not navigated


def get_vaa(dest_dir: Path) -> pd.DataFrame:
    out = dest_dir / "nhdplus_vaa_subset.parquet"
    if out.exists():
        return pd.read_parquet(out)
    tmp = dest_dir / "nhdplusVAA.parquet.download"
    log.info("downloading NHDPlus VAA (258 MB) ...")
    with requests.get(VAA_URL, headers={"User-Agent": N.USER_AGENT}, stream=True, timeout=600) as r:
        r.raise_for_status()
        h = hashlib.md5()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
                h.update(chunk)
    if h.hexdigest() != VAA_MD5:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"VAA md5 mismatch {h.hexdigest()} != {VAA_MD5}")
    v = pd.read_parquet(tmp, columns=VAA_COLS)
    for c in ["comid", "hydroseq", "dnhydroseq", "uphydroseq", "levelpathi", "terminalpa"]:
        v[c] = v[c].astype("int64")
    v["reachcode"] = v["reachcode"].astype(str)
    v.to_parquet(out, index=False, compression="zstd")
    tmp.unlink()
    return v


def _sample(key: str, every: int) -> bool:
    """Deterministic ~1/every sample of site keys (sha1-based; no RNG state)."""
    return int(hashlib.sha1(key.encode()).hexdigest(), 16) % every == 0


def _run_lookups(jobs, client, workers, res, label):
    def run(job):
        key, kind, arg = job
        feat, st = client.feature(*arg) if kind == "nwissite" else client.position(*arg)
        return key, kind, feat, st

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run, j) for j in jobs]
        for k, f in enumerate(as_completed(futs)):
            key, kind, feat, st = f.result()
            res[(key, kind)] = (feat, st)
            if (k + 1) % 500 == 0:
                log.info("%s lookups %d/%d  stats=%s", label, k + 1, len(jobs), client.stats)


def lookup_all(reg: pd.DataFrame, client: N.NLDIClient, workers: int, sample_every: int = 10) -> pd.DataFrame:
    """nwissite for every USGS site; position for NEON, for USGS sites without an nwissite COMID,
    and for a deterministic ~1/``sample_every`` sample of the rest (method-agreement statistic)."""
    res: dict = {}
    usgs = reg[reg["network"] == "USGS"]
    _run_lookups(
        [(k, "nwissite", ("nwissite", k)) for k in usgs["site_key"]], client, workers, res, "nwissite"
    )
    pos_jobs = []
    for r in reg.itertuples():
        fn, _ = res.get((r.site_key, "nwissite"), ({}, None))
        need = r.network != "USGS" or not (fn or {}).get("comid") or _sample(r.site_key, sample_every)
        if need:
            pos_jobs.append((r.site_key, "position", (float(r.lon), float(r.lat))))
    _run_lookups(pos_jobs, client, workers, res, "position")
    rows = []
    for r in reg.itertuples():
        fn, sn = res.get((r.site_key, "nwissite"), ({}, None))
        fp, sp = res.get((r.site_key, "position"), ({}, None))
        row = {
            "site_key": r.site_key,
            "network": r.network,
            "comid_nwissite": fn.get("comid") if fn else None,
            "status_nwissite": sn,
            "reachcode_nwissite": fn.get("reachcode") if fn else None,
            "measure_nwissite": fn.get("measure") if fn else None,
            "mainstem": fn.get("mainstem") if fn else None,
            "comid_position": fp.get("comid") if fp else None,
            "status_position": sp,
        }
        if row["comid_nwissite"]:
            row.update(
                comid=row["comid_nwissite"],
                comid_method="nldi_nwissite",
                reachcode=row["reachcode_nwissite"],
                measure=row["measure_nwissite"],
            )
        elif row["comid_position"]:
            row.update(
                comid=row["comid_position"], comid_method="nldi_position", reachcode=None, measure=np.nan
            )
        else:
            row.update(comid=None, comid_method=None, reachcode=None, measure=np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def navigate_all(
    nodes: pd.DataFrame,
    reg: pd.DataFrame,
    client: N.NLDIClient,
    workers: int,
    distance: float,
    modes=("DM", "UM"),
) -> pd.DataFrame:
    stype = reg.set_index("site_key")["site_type"]
    todo = nodes[nodes["comid"].notna()].copy()
    todo = todo[~todo["site_key"].map(stype).str.startswith(NO_NAV_TYPES)]
    jobs = []
    for mode in modes:  # all DM first (sufficient for downstream relations), then UM (cross-check)
        for r in todo.itertuples():
            if r.comid_method == "nldi_nwissite":
                jobs.append((r.site_key, mode, "nwissite", r.site_key))
            else:
                jobs.append((r.site_key, mode, "comid", str(r.comid)))

    def run(job):
        key, mode, src, ident = job
        df, st = client.navigate(src, ident, mode, "nwissite", distance)
        df = df.assign(query_site_key=key, mode=mode, query_source=src, status=st)
        return df, st, key, mode

    frames, failed = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run, j) for j in jobs]
        for k, f in enumerate(as_completed(futs)):
            df, st, key, mode = f.result()
            if st != 200:
                failed.append((key, mode, st))
            if len(df):
                frames.append(df)
            if (k + 1) % 500 == 0:
                log.info("navigation %d/%d  failed=%d stats=%s", k + 1, len(jobs), len(failed), client.stats)
    nav = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    log.info("navigation done: %d rows, %d failed requests", len(nav), len(failed))
    nav.attrs["failed"] = failed
    return nav, failed, len(jobs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--distance", type=float, default=500.0)
    ap.add_argument("--lookups-only", action="store_true")
    ap.add_argument("--modes", default="DM,UM")
    ap.add_argument(
        "--min-interval",
        type=float,
        default=1.25,
        help="global seconds between NLDI requests (gateway limit 3,600/h per IP)",
    )
    ap.add_argument("--position-sample-every", type=int, default=10)
    ap.add_argument("--tag", default="", help="suffix for output files/manifest (e.g. nldi_full)")
    ap.add_argument(
        "--offline",
        action="store_true",
        help="no HTTP: build nodes/edges from cached NLDI responses + local VAA traversal (provisional)",
    )
    a = ap.parse_args()
    if a.workers > 4:
        raise SystemExit("politeness rule: at most 4 concurrent requests")
    raw = paths.raw("nldi")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(raw / "download_nldi.log"), logging.StreamHandler()],
    )
    reg = pd.read_parquet(paths.PROCESSED / "sites" / "sites.parquet")
    client = N.NLDIClient(raw / "cache", min_interval=a.min_interval, offline=a.offline)
    links = lookup_all(reg, client, a.workers, a.position_sample_every)
    if not a.offline:
        links.to_parquet(raw / "comid_links.parquet", index=False)
    log.info("COMID methods: %s", links["comid_method"].value_counts(dropna=False).to_dict())
    if a.lookups_only:
        return 0
    vaa = get_vaa(paths.raw("nhdplus_vaa"))
    nodes = S.attach_vaa(links, vaa)
    nav, failed, n_jobs = navigate_all(
        nodes, reg, client, max(1, min(a.workers, 4)), a.distance, tuple(a.modes.split(","))
    )
    inter = paths.interim("nldi")
    if len(nav):
        nav.to_parquet(inter / "navigation_raw.parquet", index=False, compression="zstd")
    stype = reg.set_index("site_key")["site_type"]
    eligible = set(
        nodes.loc[
            nodes["comid"].notna() & ~nodes["site_key"].map(stype).str.startswith(NO_NAV_TYPES), "site_key"
        ]
    )
    vedges = S.vaa_downstream_edges(nodes, vaa, a.distance, eligible)
    edges = S.combine_edges(nodes, nav, vedges, set(reg["site_key"]))
    edges = edges[edges["src"].isin(eligible) & edges["dst"].isin(eligible)].reset_index(drop=True)
    nodes["navigated"] = nodes["site_key"].isin(eligible)
    nodes["flow_group"] = nodes["site_key"].map(S.flow_components(list(nodes["site_key"]), edges))
    nodes["n_upstream_sites"] = nodes["site_key"].map(edges.groupby("dst").size()).fillna(0).astype(int)
    nodes["n_downstream_sites"] = nodes["site_key"].map(edges.groupby("src").size()).fillna(0).astype(int)
    # linking QC: VAA total drainage area at the COMID vs NWIS drainage area (1 = consistent)
    da = nodes["site_key"].map(reg.set_index("site_key")["drain_area_km2"])
    nodes["da_ratio_vaa_nwis"] = (nodes["vaa_totdasqkm"] / da.where(da > 0)).round(3)
    out = paths.processed("site_network")
    sfx = f"_{a.tag}" if a.tag else ""
    nodes.to_parquet(out / f"nodes{sfx}.parquet", index=False, compression="zstd")
    edges.to_parquet(out / f"edges{sfx}.parquet", index=False, compression="zstd")
    agree = links.dropna(subset=["comid_nwissite", "comid_position"])
    stats = {
        "distance_km": a.distance,
        "n_sites": int(len(nodes)),
        "comid_method_counts": nodes["comid_method"].value_counts(dropna=False).astype(int).to_dict(),
        "n_with_comid": int(nodes["comid"].notna().sum()),
        "nwissite_vs_position_same_comid": float((agree["comid_nwissite"] == agree["comid_position"]).mean())
        if len(agree)
        else None,
        "n_nwissite_and_position": int(len(agree)),
        "n_with_vaa": int(nodes["vaa_hydroseq"].notna().sum()),
        "n_navigation_requests": n_jobs,
        "n_navigation_failed": len([f for f in failed if f[2] != -2]),
        "n_navigation_not_cached_offline": len([f for f in failed if f[2] == -2]),
        "provisional_offline_build": bool(a.offline),
        "failed_examples": failed[:20],
        "n_edges": int(len(edges)),
        "edges_found_by": edges["found_by"].value_counts().astype(int).to_dict(),
        "direction_check": edges["direction_check"].value_counts().astype(int).to_dict(),
        "n_sites_in_edges": int(len(set(edges["src"]) | set(edges["dst"]))),
        "n_flow_groups": int(nodes["flow_group"].nunique()),
        "largest_flow_group": int(nodes["flow_group"].value_counts().iloc[0]),
        "http": client.stats,
    }
    files = [
        out / f"nodes{sfx}.parquet",
        out / f"edges{sfx}.parquet",
        raw / "comid_links.parquet",
        paths.RAW / "nhdplus_vaa" / "nhdplus_vaa_subset.parquet",
    ]
    if len(nav):
        files.append(inter / "navigation_raw.parquet")
    ok = nodes["da_ratio_vaa_nwis"].notna()
    for meth in ("nldi_nwissite", "nldi_position"):
        rr = nodes.loc[ok & (nodes["comid_method"] == meth), "da_ratio_vaa_nwis"]
        stats[f"da_ratio_within_0.5_2_{meth}"] = float(rr.between(0.5, 2).mean()) if len(rr) else None
        stats[f"n_da_checked_{meth}"] = int(len(rr))
    write_manifest(
        f"site_network{sfx}",
        files=files,
        sources=[
            N.BASE + "/nwissite/USGS-<no>",
            N.BASE + "/comid/position",
            N.BASE + "/<src>/<id>/navigation/{UM,DM}/nwissite?distance=" + f"{a.distance:g}",
            VAA_URL,
        ],
        description="Site -> NHDPlus V2 COMID links (method recorded) and flow-connected site edges "
        "(src upstream -> dst downstream, main stem within distance) with along-channel "
        "distance from NHDPlus V2 VAA pathlength.",
        license="US public domain (USGS NLDI, NHDPlus V2)",
        stats=stats,
    )
    log.info("done: %s", stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
