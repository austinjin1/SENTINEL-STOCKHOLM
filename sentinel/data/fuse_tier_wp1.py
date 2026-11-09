# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Fuse harder track W-P1 (docs/preregistration/fuse_protocol.md §12): the P1 forecasting task at
USGS continuous-sonde sites of the frozen M1 split, unseen sites × unseen years.

Row unit: (USGS site, local issue day d) on a fixed 3-day calendar grid (days since 2008-10-01 divisible
by 3 — label-free thinning of strongly autocorrelated daily rows), WY2009–2025, kept when a W-P1 target
(DO, SC, turbidity FNU, NO3+NO2 daily mean) is observed at d + 7 or d + 14. Inputs are dated ≤ d:

* ``sonde`` (DO, SC, pH, turbidity FNU, fDOM), ``nitrate`` (NO3+NO2 sensor), ``temp``, ``hydro``
  (log discharge; gage height minus its trailing 365-d median) — USGS daily values, same engineered
  features as Tier D (``fuse_tiers.sensor_features``);
* ``labhist`` — previous WQP lab sample of each Tier W analyte strictly before d (≤ 90 d);
* hub ``wx`` (gridMET/ERA5 windows ending at d), ``static`` (StreamCat + site registry), ``time``.

Targets ``y__<var>__h{7,14}`` (model space as Tier D: log(x + offset) for SC, turbidity, nitrate).
Roles/folds come from ``fuse_tier_wp1_split.json``; test-role rows are written (for the single final
scoring) but every development reader drops them on load.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from ..core import paths
from . import fuse_tier_w as ftw
from . import fuse_tiers as ftd

CAL = ("2008-10-01", "2025-09-30")
GRID_DAYS = 3
HORIZONS = (7, 14)
SENSORS = {  # modality: [(var, log offset or None, anomaly window or None)]
    "sonde": [("do_mgl", None, None), ("spcond_uscm", 1.0, None), ("ph", None, None), ("turb_fnu", 0.1, None),
              ("fdom_qse", 0.1, None)],
    "nitrate": [("no3no2_mgl", 0.01, None)],
    "temp": [("temp_c", None, None)],
    "hydro": [("discharge_cfs", 0.01, None), ("gage_ft", None, 365)],
}
TARGETS = {"do_mgl": "sonde", "spcond_uscm": "sonde", "turb_fnu": "sonde", "no3no2_mgl": "nitrate"}
NON_HUB = ["sonde", "nitrate", "temp", "hydro", "labhist"]
HUB = ["wx", "static", "time"]
SPLIT_SHA = "b9e0d5c44d8a83c3bc4abc8517d66fb520186fcf6e64db59aa3e4d40e050288a"


def _series(raw: pd.DataFrame, v: str, off, anom, cal) -> pd.DataFrame | None:
    mean_c = f"{v}_mean" if f"{v}_mean" in raw else (f"{v}_median" if f"{v}_median" in raw else None)
    if mean_c is None:
        return None

    def tz(c):
        x = raw[c].astype(float) if c in raw else pd.Series(np.nan, index=raw.index)
        return np.log(np.maximum(x, 0) + off) if off else x

    z = pd.DataFrame({f"{v}__mean": tz(mean_c), f"{v}__min": tz(f"{v}_min"), f"{v}__max": tz(f"{v}_max")})
    z = z[~z.index.duplicated()].reindex(cal)
    if anom:
        z = ftd.trailing_anomaly(z, v, cal, anom).reindex(cal)
    return z


def build(log=lambda *a: print(*a, flush=True), max_sites: int | None = None) -> pd.DataFrame:
    from ..models.fuse import splits as sp
    split = sp.load_split(sp.TIER_WP1_SPLIT, SPLIT_SHA)
    roles = split["site_role"]
    cal = pd.date_range(*CAL, freq="D")
    grid = cal[(np.arange(len(cal)) % GRID_DAYS) == 0]
    sdir = paths.processed("sonde_daily") / "sites"
    sites = sorted(s for s in roles if (sdir / f"{s}.parquet").exists())
    if max_sites:                     # smoke only
        sites = sites[:max_sites]
    labs = {v: ftw._lab(v, {f"USGS-{s}" for s in sites}) for v in ftw.W_TARGETS}
    sc = pd.read_parquet(paths.processed("catchment") / "streamcat_sites.parquet").set_index("site_key")
    reg = pd.read_parquet(paths.processed("sites") / "sites.parquet").set_index("site_key")
    wxds = ds.dataset(str(paths.processed("forcings_daily") / "forcings_daily.parquet"))
    out = []
    for i, no in enumerate(sites):
        raw = pd.read_parquet(sdir / f"{no}.parquet")
        raw.index = pd.to_datetime(raw.index)
        if not any(f"{v}_mean" in raw or f"{v}_median" in raw for v in TARGETS):
            continue
        series = {}
        blocks = []
        for mod, vs in SENSORS.items():
            feats = []
            for v, off, anom in vs:
                z = _series(raw, v, off, anom, cal)
                if z is None or z[f"{v}__mean"].notna().sum() == 0:
                    continue
                series[v] = z
                feats.append(ftd.sensor_features(z, cal, mod, v))
            if feats:
                fb = pd.concat(feats, axis=1)
                ages = fb[[c for c in fb.columns if c.endswith("__age")]].min(axis=1)
                fb[f"{mod}__present"] = ages < ftd.SENSOR_PRESENT_DAYS
            else:
                fb = pd.DataFrame({f"{mod}__present": False}, index=cal)
            blocks.append(fb.reindex(grid))
        yt = pd.DataFrame(index=grid)
        for v in TARGETS:
            m = series[v][f"{v}__mean"] if v in series else pd.Series(np.nan, index=cal)
            for h in HORIZONS:
                yt[f"y__{v}__h{h}"] = m.shift(-h).reindex(grid)
        keep = yt.notna().any(axis=1).to_numpy()
        if not keep.any():
            continue
        dates = grid[keep]
        blk = pd.concat([b.loc[dates] for b in blocks] + [yt.loc[dates]], axis=1)
        site = f"USGS-{no}"
        anyh = np.zeros(len(dates), bool)
        cd = dates.to_numpy().astype("datetime64[D]")
        for v, L in labs.items():
            l = L[L.site_id == site]
            ld = l["date"].to_numpy().astype("datetime64[D]")
            pos = np.searchsorted(ld, cd, side="left") - 1          # strictly before d
            age = np.where(pos >= 0, (cd - ld[np.clip(pos, 0, None)]).astype(int) if len(ld) else 10**6, 10**6)
            ok = (pos >= 0) & (age <= ftw.LABHIST_MAX_AGE)
            zz = l["z"].to_numpy(float)
            blk[f"labhist__{v}__value"] = np.where(ok, zz[np.clip(pos, 0, None)] if len(zz) else np.nan, np.nan)
            blk[f"labhist__{v}__age"] = np.where(ok, age, np.nan)
            anyh |= ok
        blk["labhist__present"] = anyh
        wx = wxds.to_table(filter=ds.field("site_key") == site, columns=["date"] + ftd.WX_VARS).to_pandas()
        if len(wx):
            wx["date"] = pd.to_datetime(wx["date"])
            wb = ftd.weather_features(wx.drop_duplicates("date"), cal).loc[dates]
            wb["wx__present"] = wb["wx__precip_sum7"].notna()
        else:
            wb = pd.DataFrame({"wx__present": False}, index=dates)
        blk = pd.concat([blk, wb], axis=1)
        blk["static__lat"] = float(reg.loc[site, "lat"]) if site in reg.index else np.nan
        blk["static__lon"] = float(reg.loc[site, "lon"]) if site in reg.index else np.nan
        blk["static__log_drain_km2"] = np.log10(float(reg.loc[site, "drain_area_km2"]) + 1) \
            if site in reg.index and pd.notna(reg.loc[site, "drain_area_km2"]) else np.nan
        found = site in sc.index and bool(sc.loc[site, "streamcat_found"])
        for col in ftd.STATIC_COLS:
            v = float(sc.loc[site, col]) if found and col in sc.columns and pd.notna(sc.loc[site, col]) else np.nan
            blk[f"static__{col}"] = np.log1p(v) if col in ("wsareasqkm", "popden2010ws", "fertws", "manurews") \
                and np.isfinite(v) else v
        blk["static__present"] = True
        doy = dates.dayofyear.to_numpy(float)
        blk["time__doy_sin"] = np.sin(2 * np.pi * (doy - 1) / 365.25)
        blk["time__doy_cos"] = np.cos(2 * np.pi * (doy - 1) / 365.25)
        blk["time__present"] = True
        blk.index.name = "date"
        blk = blk.reset_index()
        blk.insert(0, "site", no)
        blk.insert(1, "role", roles[no])
        blk.insert(2, "fold", split["site_fold"].get(no))
        blk = blk.astype({c: "float32" for c in blk.columns if blk[c].dtype == np.float64})
        out.append(blk)
        if i % 200 == 0:
            log(f"  {i}/{len(sites)} sites; rows so far {sum(len(o) for o in out)}")
    rows = pd.concat(out, ignore_index=True)
    # test-window rule (split windows): dev rows need every target date <= 2023-09-30
    w = split["windows"]
    d = rows["date"]
    dev_ok = (rows["role"] == "dev") & (d >= w["dev_issue"][0]) & (d + pd.Timedelta(days=max(HORIZONS)) <= w["dev_target_max"])
    test_a = (rows["role"] == "test") & (d >= w["test_a_issue"][0]) & (d + pd.Timedelta(days=max(HORIZONS)) <= w["test_a_target_max"])
    test_s = (rows["role"] == "test") & (d + pd.Timedelta(days=max(HORIZONS)) <= w["dev_target_max"])
    rows["subset"] = np.select([dev_ok, test_a, test_s], ["dev", "test_w_a", "test_w_s"], "unused")
    return rows[rows["subset"] != "unused"].reset_index(drop=True)


def main_build() -> None:
    import datetime as dt
    from ..core.manifest import write_manifest
    rows = build()
    out = paths.processed("fuse_tier_wp1")
    f = out / "rows.parquet"
    rows.to_parquet(f, compression="zstd", index=False)
    cols = {m: [c for c in rows.columns if c.startswith(f"{m}__") and not c.endswith("__present")] for m in NON_HUB + HUB}
    (out / "columns.json").write_text(json.dumps(cols, indent=1))
    stats = {"rows": int(len(rows)), "sites": int(rows.site.nunique()),
             "by_subset": {k: {"rows": int(len(g)), "sites": int(g.site.nunique())} for k, g in rows.groupby("subset")},
             "present_rows": {m: int(rows[f"{m}__present"].sum()) for m in NON_HUB + HUB},
             "target_rows": {c: {k: int(g[c].notna().sum()) for k, g in rows.groupby("subset")}
                             for c in rows.columns if c.startswith("y__")},
             "grid_days": GRID_DAYS, "split_sha256": SPLIT_SHA, "built": dt.date.today().isoformat()}
    write_manifest("fuse_tier_wp1", files=[f, out / "columns.json"],
                   sources=["manifest:sonde_daily", "manifest:wqp_discrete", "manifest:forcings_daily",
                            "manifest:streamcat", "manifest:sites"],
                   description="SENTINEL-Fuse W-P1 harder track: USGS sonde site × issue day (3-day grid) multimodal table",
                   license="US public domain (USGS NWIS, WQP, gridMET, StreamCat)", synthetic=False, stats=stats)
    print(json.dumps({k: stats[k] for k in ("rows", "sites", "by_subset", "present_rows")}, indent=1))
