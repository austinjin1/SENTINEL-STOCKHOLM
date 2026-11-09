# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Observed antecedent weather (gridMET) for arbitrary downstream label points.

`processed/forcings_daily` holds gridMET only at the ~5,100 4-km cells that contain a SENTINEL site.
Downstream units (beaches, fish-kill points, NRSA reaches, lakes) are matched to the **nearest such
cell** (great-circle distance between the point and the cell centre, stored as ``wx_cell_km``); the
caller decides a maximum distance. Weather is OBSERVED analysis: it may only be used for days strictly
before the decision date (the windows below end at ``d - 1`` by construction, ``lag0`` is opt-in for
nowcasts where same-day observed weather is legitimately known, e.g. an afternoon beach nowcast).
Anomalies use a day-of-year climatology fitted on caller-supplied training years only (R4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from ...core import paths
from ...eval.splits import haversine_km

VARS = ("precip_mm", "tmax_c", "tmin_c", "srad_wm2", "wind_ms", "vpd_kpa", "rhmin_pct")


def cell_table() -> pd.DataFrame:
    c = pd.read_parquet(paths.RAW / "gridmet" / "site_cells.parquet")
    have = set(pd.read_parquet(paths.PROCESSED / "forcings_daily" / "coverage.parquet")["site_key"].astype(str)) \
        if (paths.PROCESSED / "forcings_daily" / "coverage.parquet").exists() else set(c["site_key"])
    c = c[c["site_key"].isin(have) & c["cell_lat"].notna() & c["cell_lon"].notna() & c["cell_key"].notna()]
    return c.sort_values("site_key").drop_duplicates("cell_key")[["cell_key", "site_key", "cell_lat", "cell_lon"]]


def nearest_cells(lat, lon, cells: pd.DataFrame | None = None, chunk: int = 512) -> pd.DataFrame:
    """For each point: nearest available gridMET cell (``wx_cell``, ``wx_site_key``, ``wx_cell_km``)."""
    cells = cell_table() if cells is None else cells
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    clat, clon = cells["cell_lat"].to_numpy(), cells["cell_lon"].to_numpy()
    idx = np.empty(lat.size, int)
    dist = np.empty(lat.size)
    for s in range(0, lat.size, chunk):
        d = haversine_km(lat[s:s + chunk, None], lon[s:s + chunk, None], clat[None], clon[None])
        idx[s:s + chunk] = d.argmin(1)
        dist[s:s + chunk] = d.min(1)
    return pd.DataFrame({"wx_cell": cells["cell_key"].to_numpy()[idx], "wx_site_key": cells["site_key"].to_numpy()[idx],
                         "wx_cell_km": dist})


def load_series(site_keys, start: str = "2000-01-01", end: str | None = None, variables=VARS) -> pd.DataFrame:
    """Daily forcings for the given proxy site keys (one row per site_key × date)."""
    keys = sorted(set(map(str, site_keys)))
    dset = ds.dataset(paths.PROCESSED / "forcings_daily" / "forcings_daily.parquet")
    flt = ds.field("site_key").isin(keys) & (ds.field("date") >= pd.Timestamp(start).date())
    if end is not None:
        flt = flt & (ds.field("date") <= pd.Timestamp(end).date())
    t = dset.to_table(columns=["site_key", "date", *variables], filter=flt).to_pandas()
    t["date"] = pd.to_datetime(t["date"])
    return t.sort_values(["site_key", "date"]).reset_index(drop=True)


def doy_climatology(series: pd.DataFrame, years, variables=("tmax_c", "tmin_c", "precip_mm"),
                    window: int = 15) -> pd.DataFrame:
    """Per site_key × day-of-year mean over ``years`` only (circular ±window smoothing)."""
    s = series[series["date"].dt.year.isin(list(years))].copy()
    s["doy"] = np.minimum(s["date"].dt.dayofyear, 365)
    g = s.groupby(["site_key", "doy"])[list(variables)].mean()
    out = []
    for key, sub in g.groupby(level=0):
        sub = sub.droplevel(0).reindex(range(1, 366))
        ext = pd.concat([sub.iloc[-window:], sub, sub.iloc[:window]])
        sm = ext.rolling(2 * window + 1, center=True, min_periods=window).mean().iloc[window:-window]
        sm.index.name = "doy"
        sm = sm.reset_index()
        sm.insert(0, "site_key", key)
        out.append(sm)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["site_key", "doy", *variables])


def antecedent_features(queries: pd.DataFrame, series: pd.DataFrame, *, clim: pd.DataFrame | None = None,
                        windows=(1, 3, 7, 14, 30), lag0: bool = False, prefix: str = "wx_") -> pd.DataFrame:
    """Window features ending the day before ``date`` (or on ``date`` if ``lag0``).

    ``queries`` needs ``wx_site_key`` and ``date``. Returns one row per query (same index):
    precipitation sums, Tmax/Tmin means, srad/vpd/wind means per window, max 1-day precipitation over
    7 d, and Tmax/Tmin/precip anomalies vs ``clim`` (if given).
    """
    q = queries[["wx_site_key", "date"]].copy()
    q["date"] = pd.to_datetime(q["date"])
    q["_row"] = np.arange(len(q))
    res = pd.DataFrame(index=q["_row"])
    shift = 0 if lag0 else 1
    feats = []
    by_site = {k: g for k, g in series.groupby("site_key", sort=False)}
    by_clim = {k: g for k, g in clim.groupby("site_key", sort=False)} if clim is not None else {}
    for key, qs in q.groupby("wx_site_key"):
        s = by_site.get(key)
        if s is None or s.empty:
            continue
        s = s.set_index("date").sort_index()
        # only the span needed by this site's queries (longest window + margin)
        lo = qs["date"].min() - pd.Timedelta(days=max(windows) + 40)
        s = s.loc[s.index >= lo]
        if s.empty:
            continue
        full = s.reindex(pd.date_range(s.index.min(), max(s.index.max(), qs["date"].max()), freq="D"))
        if clim is not None and key in by_clim:
            c = by_clim[key].set_index("doy")
            doy = np.minimum(full.index.dayofyear, 365)
            for v in ("tmax_c", "tmin_c", "precip_mm"):
                if v in c:
                    full[f"{v}_anom"] = full[v].to_numpy() - c[v].reindex(doy).to_numpy()
        cols = {}
        for w in windows:
            r = full.rolling(w, min_periods=max(1, int(0.8 * w)))
            sh = lambda x: x.shift(shift)  # noqa: E731 — window ends d-1 unless lag0
            cols[f"precip_sum_{w}d"] = sh(r["precip_mm"].sum())
            cols[f"tmax_mean_{w}d"] = sh(r["tmax_c"].mean())
            cols[f"tmin_mean_{w}d"] = sh(r["tmin_c"].mean())
            if w in (1, 3, 7):
                for v in ("srad_wm2", "vpd_kpa", "wind_ms"):
                    if v in full:
                        cols[f"{v.split('_')[0]}_mean_{w}d"] = sh(r[v].mean())
            if "tmax_c_anom" in full:
                cols[f"tmax_anom_{w}d"] = sh(full["tmax_c_anom"].rolling(w, min_periods=max(1, int(0.8 * w))).mean())
                cols[f"tmin_anom_{w}d"] = sh(full["tmin_c_anom"].rolling(w, min_periods=max(1, int(0.8 * w))).mean())
                cols[f"precip_anom_{w}d"] = sh(full["precip_mm_anom"].rolling(w, min_periods=max(1, int(0.8 * w))).sum())
        cols["precip_max1d_7d"] = full["precip_mm"].rolling(7, min_periods=5).max().shift(shift)
        dry = (full["precip_mm"] < 1.0).astype(float).where(full["precip_mm"].notna())  # missing stays missing
        cols["dry_days_14d"] = dry.rolling(14, min_periods=10).sum().shift(shift)
        fr = pd.DataFrame(cols, index=full.index)
        got = fr.reindex(qs["date"].to_numpy())
        got.index = qs["_row"].to_numpy()
        feats.append(got)
    if feats:
        res = pd.concat(feats).reindex(res.index)
    res.columns = [prefix + c for c in res.columns]
    res.index = queries.index
    return res
