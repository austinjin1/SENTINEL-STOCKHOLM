#!/usr/bin/env python
"""Track B, step 1 — enumerate NON-CIRCULAR co-located labels (real N).

Two labels, neither computable from the encoder inputs:
  * chl-a (held-out parameter): chlorophyll-a is NOT one of the 5 sensor inputs
    [DO,pH,Turb,Temp,SpCond], so predicting chl-a > 8 ug/L from a Sentinel-2
    tile + a sensor window is a genuine cross-modal task where the satellite
    carries information the point sensor lacks. Labels from WQP chl-a (ug/L),
    joined to each tile by USGS site_id within +/- TOL_CHL days.
  * forecast DO < 5 mg/L at t+H (future value): the label lies H days AFTER the
    inputs, so it is not contained in them. Read from the raw USGS sensor
    parquet at date+H.

Emits the per-task sample tables (tile index + label + site + date) and reports
the real N and positive/negative counts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import _fusion as F

ROOT = Path(__file__).resolve().parents[3]
PAIRS = ROOT / "data" / "processed" / "satellite" / "drone_wq_pairs.npz"
WQP = ROOT / "data" / "processed" / "biology" / "noaa_habs" / "wqp_hab_indicators.parquet"
OUT = ROOT / "data" / "processed" / "trackB"
OUT.mkdir(parents=True, exist_ok=True)

CHL_THR = 8.0          # ug/L EPA recreational HAB indicator
TOL_CHL = 30           # days
H_LIST = [7, 14]       # forecast horizons (days)
TOL_FC = 2             # days tolerance on the forecast lookup
DO_FC_THR = 5.0


def load_chl_by_site():
    """site_id -> sorted (date, value_ugL) for chlorophyll-a in ug/L."""
    cols = ["MonitoringLocationIdentifier", "ActivityStartDate", "CharacteristicName",
            "ResultMeasureValue", "ResultMeasure/MeasureUnitCode"]
    df = pq.read_table(WQP, columns=cols).to_pandas()
    df = df[df["CharacteristicName"].astype(str).str.contains("Chlorophyll", case=False, na=False)]
    unit = df["ResultMeasure/MeasureUnitCode"].astype(str).str.lower()
    df = df[unit.isin(["ug/l", "µg/l", "ug/l ", "ppb"])]
    df["val"] = pd.to_numeric(df["ResultMeasureValue"], errors="coerce")
    df["d"] = pd.to_datetime(df["ActivityStartDate"], errors="coerce")
    df = df.dropna(subset=["val", "d"])
    df["site"] = df["MonitoringLocationIdentifier"].astype(str).str.split("-", n=1).str[-1]
    out = {}
    for site, g in df.groupby("site"):
        g = g.sort_values("d")
        out[site] = (g["d"].to_numpy(), g["val"].to_numpy())
    return out


def nearest_chl(chl_site, date, tol_days):
    dates, vals = chl_site
    dd = np.abs((dates - np.datetime64(date)).astype("timedelta64[D]").astype(int))
    i = dd.argmin()
    return (float(vals[i]), int(dd[i])) if dd[i] <= tol_days else (None, None)


def forecast_do(site, date, h, tol):
    df = F._site_df(site)
    if df is None or "DO" not in df.columns:
        return None
    t = pd.Timestamp(date, tz="UTC") + pd.Timedelta(days=h)
    lo, hi = t - pd.Timedelta(days=tol), t + pd.Timedelta(days=tol)
    w = df.loc[(df.index >= lo) & (df.index <= hi), "DO"]
    w = pd.to_numeric(w, errors="coerce").dropna()
    return float(w.mean()) if len(w) else None


def main():
    z = np.load(PAIRS, allow_pickle=True)
    md = json.loads(str(z["metadata"]))
    F.set_sensor_stats_from_targets(z["targets"])
    print(f"{len(md)} tiles; sensor stats set.", flush=True)

    chl_by_site = load_chl_by_site()
    print(f"WQP chl-a sites (ug/L): {len(chl_by_site)}", flush=True)

    chl_rows, fc_rows = [], []
    for i, m in enumerate(md):
        site, date = m["site_id"], m["date"]
        # chl-a held-out label
        if site in chl_by_site:
            val, dd = nearest_chl(chl_by_site[site], date, TOL_CHL)
            if val is not None:
                chl_rows.append((i, site, date, val, int(val > CHL_THR), dd))
        # forecast DO label
        labs = {h: forecast_do(site, date, h, TOL_FC) for h in H_LIST}
        if all(v is not None for v in labs.values()):
            fc_rows.append((i, site, date, labs[7], labs[14],
                            int(labs[7] < DO_FC_THR), int(labs[14] < DO_FC_THR)))
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(md)}  chl={len(chl_rows)} fc={len(fc_rows)}", flush=True)

    # --- chl-a table ---
    if chl_rows:
        idx = np.array([r[0] for r in chl_rows])
        np.savez(OUT / "pairs_chl.npz",
                 tile_idx=idx,
                 site=np.array([r[1] for r in chl_rows]),
                 date=np.array([r[2] for r in chl_rows]),
                 value=np.array([r[3] for r in chl_rows], "float32"),
                 label=np.array([r[4] for r in chl_rows], "int64"),
                 day_offset=np.array([r[5] for r in chl_rows], "int64"))
    # --- forecast table ---
    if fc_rows:
        np.savez(OUT / "pairs_forecast.npz",
                 tile_idx=np.array([r[0] for r in fc_rows]),
                 site=np.array([r[1] for r in fc_rows]),
                 date=np.array([r[2] for r in fc_rows]),
                 do_h7=np.array([r[3] for r in fc_rows], "float32"),
                 do_h14=np.array([r[4] for r in fc_rows], "float32"),
                 label_h7=np.array([r[5] for r in fc_rows], "int64"),
                 label_h14=np.array([r[6] for r in fc_rows], "int64"))

    summary = {
        "chl": {"n": len(chl_rows),
                "pos": int(sum(r[4] for r in chl_rows)),
                "n_sites": len(set(r[1] for r in chl_rows)),
                "thr_ugL": CHL_THR, "tol_days": TOL_CHL},
        "forecast": {"n": len(fc_rows),
                     "pos_h7": int(sum(r[5] for r in fc_rows)),
                     "pos_h14": int(sum(r[6] for r in fc_rows)),
                     "n_sites": len(set(r[1] for r in fc_rows)),
                     "thr_mgL": DO_FC_THR, "horizons": H_LIST},
    }
    (OUT / "pairs_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Track B labels ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
