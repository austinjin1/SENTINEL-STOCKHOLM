#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Fetch recent gridMET daily weather for the live-trial weather sites (H2, H4) for issue day D.

Same product, variables, cells and unit conversion as ``scripts/data/download_forcings.py`` (bulk yearly netCDF,
frozen nearest valid cells in ``raw/gridmet/site_cells.parquet``). Recent slices are gridMET "provisional"/"early";
the status is kept. Output: ``$SENTINEL_RUNS/live/ew_v2[/dryrun]/inputs/gridmet/<D>/weather.parquet`` + manifest;
the netCDFs are deleted after extraction.

    python -B scripts/live/fetch_gridmet.py --date 2026-12-01 [--mode dryrun]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json

import numpy as np
import pandas as pd

from live_common import DRYRUN_ROOT, LIVE_BLOCK, LIVE_ROOT, excl_write, load_config, sha256_file
from sentinel.core import paths
from sentinel.data.sources import gridmet as G

VARS = {"pr": "precip_mm", "tmmx": "tmax_c", "tmmn": "tmin_c", "srad": "srad_wm2", "vs": "wind_ms", "vpd": "vpd_kpa"}
WEATHER_HAZARDS = ("H2_chloride", "H4_nitrate")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=_dt.datetime.now(_dt.timezone.utc).date().isoformat())
    ap.add_argument("--mode", choices=("live", "dryrun"), default="live")
    a = ap.parse_args()
    cfg, cfg_sha = load_config()
    D = pd.Timestamp(a.date)
    root = (LIVE_ROOT if a.mode == "live" else DRYRUN_ROOT) / "inputs" / "gridmet" / D.date().isoformat()
    if (root / "manifest.json").exists():
        print(f"{root} already fetched")
        return 0
    roles = None if a.mode == "live" else {"train", "val"}
    sites = sorted({r["site_no"] for hz in WEATHER_HAZARDS for r in cfg["sites"][hz]
                    if roles is None or r["role"] in roles})
    cpath = paths.RAW / "gridmet" / "site_cells.parquet"
    if sha256_file(cpath) != cfg["data_pins"]["gridmet_site_cells_sha256"]:
        raise PermissionError("gridMET site_cells.parquet differs from the frozen pin")
    cells = pd.read_parquet(cpath).set_index("site_key").reindex([f"USGS-{s}" for s in sites])
    cells["i"] = cells["i"].fillna(-1).astype(int)
    cells["j"] = cells["j"].fillna(-1).astype(int)
    lb = int(cfg["fetch"]["lookback_days"])
    d0 = max(D - pd.Timedelta(days=lb), pd.Timestamp(LIVE_BLOCK[0]))
    d1 = D - pd.Timedelta(days=1)
    years = sorted({d0.year, d1.year})
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / "tmp"
    tmp.mkdir(exist_ok=True)
    frames, status, info = {}, None, {}
    for v, col in VARS.items():
        parts = []
        for y in years:
            dest = tmp / f"{v}_{y}.nc"
            G.download(G.file_url(v, y), dest)
            g = G.read_grid(dest)
            if g["units"] != G.VARIABLES[v][1]:
                raise ValueError(f"{v}_{y}: units {g['units']!r}")
            vals = G.to_harmonised(v, G.extract_cells(g, cells.reset_index()))
            df = pd.DataFrame(vals, columns=cells.index.tolist())
            df.insert(0, "date", g["dates"])
            df.insert(1, "gridmet_status", g["status"])
            info[f"{v}_{y}"] = {"last_date": str(g["dates"][-1].date()), "n_days": int(len(g["dates"]))}
            dest.unlink()
            parts.append(df)
        w = pd.concat(parts, ignore_index=True)
        w = w[(w["date"] >= d0) & (w["date"] <= d1)]
        status = w[["date", "gridmet_status"]] if status is None else status
        frames[col] = w.drop(columns="gridmet_status").melt(id_vars="date", var_name="site_key", value_name=col)
    tmp.rmdir()
    out = None
    for col, f in frames.items():
        out = f if out is None else out.merge(f, on=["date", "site_key"], how="outer")
    out = out.merge(status.drop_duplicates("date"), on="date", how="left")
    out = out[["site_key", "date", *VARS.values(), "gridmet_status"]].sort_values(["site_key", "date"])
    out = out[np.isfinite(out[list(VARS.values())]).any(axis=1)]
    f = root / "weather.parquet"
    out.to_parquet(f, index=False)
    f.chmod(0o444)
    man = {"issue_date": D.date().isoformat(), "mode": a.mode, "window": [str(d0.date()), str(d1.date())],
           "n_sites": len(sites), "n_sites_with_cell": int((cells["i"] >= 0).sum()), "files": info,
           "last_date_available": str(out["date"].max().date()) if len(out) else None,
           "status_counts": out.drop_duplicates("date")["gridmet_status"].value_counts().to_dict() if len(out) else {},
           "weather_sha256": sha256_file(f), "config_sha256": cfg_sha,
           "finished_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    excl_write(root / "manifest.json", man)
    print(json.dumps(man))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
