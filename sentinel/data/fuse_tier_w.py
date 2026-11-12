# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Fuse Tier W ("wide") — WQP lab samples at USGS continuous-sonde sites.

Row unit: (USGS site, local sample date) with at least one target lab analyte in
``processed('wqp_discrete')`` at a site of the frozen M1 split (docs/preregistration/sonde_split.json),
water years 2009–2025. Used for pretraining and for the nutrient / FIB virtual-sensing tasks.

Modalities
* ``sonde``    — USGS daily sonde features on or before d (same engineered features as Tier D).
* ``labhist``  — the previous lab sample of each analyte at the site, strictly before d (age <= 90 d).
* ``satellite``— Sentinel-2 matchup features (``processed('s2_wqp_matchups')`` qc_pass rows, nearest |Δt|).
* hub: ``wx`` (gridMET/ERA5 windows ending at d), ``static`` (StreamCat + site registry), ``time``.

Targets: ``ylab__<analyte>`` (model space log10(x + offset)), ``__cens`` flag (value = bound).
Site roles are copied from the M1 split (train / val / test basin groups); test-role rows are never
used by Fuse development code.
"""

from __future__ import annotations

from .reserved import is_reserved

import json

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from ..core import paths
from ..core.manifest import read_manifest, write_manifest
from . import fuse_tiers as ftd

W_TARGETS = {"tp": 0.001, "tn": 0.01, "no3no2": 0.001, "nh3": 0.001, "orthop": 0.0005, "tss": 0.1, "ssc": 0.1,
             "chla": 0.1, "ecoli": 1.0, "doc": 0.1, "chloride": 0.01}
SONDE_VARS = {"temp_c": "temp_c", "do_mgl": "do_mgl", "spcond_uscm": "spcond_uscm", "ph": "ph",
              "turb_fnu": "turb_fnu", "no3no2_mgl": "no3no2_mgl", "discharge_cfs": "discharge_cfs"}
SONDE_LOG = {"spcond_uscm": 1.0, "turb_fnu": 0.1, "no3no2_mgl": 0.01, "discharge_cfs": 0.01}
CAL = ("2008-10-01", "2025-09-30")
LABHIST_MAX_AGE = 90
M1_SPLIT_SHA = "fd4bb14fe7304d05310e286e9aced22675fd6d07aa4dcac6415ed039a1e06a80"


def _lab(var: str, sites: set) -> pd.DataFrame:
    d = pd.read_parquet(paths.processed("wqp_discrete"), filters=[("variable", "=", var)],
                        columns=["site_id", "activity_start_date", "value", "censored", "detection_limit"])
    d = d[d["site_id"].isin(sites)].copy()
    d["date"] = pd.to_datetime(d["activity_start_date"]).dt.normalize()
    d = d[(d["date"] >= CAL[0]) & (d["date"] <= CAL[1])]
    d["censored"] = d["censored"].fillna(False).astype(bool)
    d["v"] = np.where(d["censored"] & d["value"].isna(), d["detection_limit"], d["value"])
    g = d.groupby(["site_id", "date"])
    out = pd.DataFrame({"value": g["v"].median(), "cens": g["censored"].all()}).reset_index()
    out = out[np.isfinite(out["value"])]
    out["z"] = np.log10(np.maximum(out["value"], 0) + W_TARGETS[var])
    return out


def s2_matchup_features() -> pd.DataFrame:
    """(site_id, date) → nearest-|Δt| qc_pass Sentinel-2 features from ``processed('s2_wqp_matchups')``."""
    root = paths.processed("s2_wqp_matchups")
    if not (root / "matchups.parquet").exists():
        return pd.DataFrame()
    m = pd.read_parquet(root / "matchups.parquet")
    m = m[m["qc_pass"].fillna(False).astype(bool)]
    s = pd.read_parquet(root / "samples.parquet", columns=["sample_key", "site_id", "date"]).drop_duplicates("sample_key")
    m = m.merge(s, on="sample_key", how="inner", suffixes=("", "_s"))
    m["date"] = pd.to_datetime(m["date_s"] if "date_s" in m else m["date"]).dt.normalize()
    m = m[m["dt_days"] <= 0]   # imagery on/before the sample date (decision-time rule)
    m = m.sort_values("abs_dt_days").drop_duplicates(["site_id", "date"])
    feats = {"site_id": m["site_id"].to_numpy(), "date": m["date"].to_numpy()}
    for b in ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"):
        feats[f"satellite__rw3_{b}"] = m[f"rw3_{b}"].to_numpy(float)
        feats[f"satellite__rw7_{b}"] = m[f"rw7_{b}"].to_numpy(float)
    r = m["rw3_B04"].to_numpy(float)
    feats["satellite__ndci"] = (m["rw3_B05"].to_numpy(float) - r) / (m["rw3_B05"].to_numpy(float) + r)
    feats["satellite__g_r"] = m["rw3_B03"].to_numpy(float) / np.where(np.abs(r) > 1e-6, r, np.nan)
    feats["satellite__b5_b4"] = m["rw3_B05"].to_numpy(float) / np.where(np.abs(r) > 1e-6, r, np.nan)
    feats["satellite__abs_dt_days"] = m["abs_dt_days"].to_numpy(float)
    feats["satellite__dt_days"] = m["dt_days"].to_numpy(float)
    feats["satellite__water_frac_c7"] = (m["scl_c7_n_water"] / m["scl_c7_n"].where(m["scl_c7_n"] > 0)).to_numpy(float)
    return pd.DataFrame(feats)


def build_tier_w(log=lambda *a: print(*a, flush=True)) -> pd.DataFrame:
    sf = paths.REPO_ROOT / "docs" / "preregistration" / "sonde_split.json"
    import hashlib
    got = hashlib.sha256(sf.read_bytes()).hexdigest()
    if got != M1_SPLIT_SHA:
        raise ValueError(f"sonde_split.json sha256 {got} != {M1_SPLIT_SHA}")
    split = json.loads(sf.read_text())
    roles = split["site_role"]
    sites = {f"USGS-{s}" for s in roles}
    labs = {}
    for v in W_TARGETS:
        labs[v] = _lab(v, sites)
        log(f"lab {v}: {len(labs[v])} site-days")
    keys = pd.concat([l[["site_id", "date"]] for l in labs.values()]).drop_duplicates()
    keys = keys.sort_values(["site_id", "date"]).reset_index(drop=True)
    log(f"rows (site-days with any target): {len(keys)} at {keys.site_id.nunique()} sites")
    sc = pd.read_parquet(paths.processed("catchment") / "streamcat_sites.parquet").set_index("site_key")
    reg = pd.read_parquet(paths.processed("sites") / "sites.parquet").set_index("site_key")
    wxds = ds.dataset(str(paths.processed("forcings_daily") / "forcings_daily.parquet"))
    cal = pd.date_range(*CAL, freq="D")
    sdir = paths.processed("sonde_daily") / "sites"
    out = []
    for i, (site, g) in enumerate(keys.groupby("site_id", sort=True)):
        no = site[len("USGS-"):]
        dates = pd.DatetimeIndex(g["date"])
        blk = pd.DataFrame(index=dates)
        f = sdir / f"{no}.parquet"
        feats = []
        if f.exists() and not is_reserved(no):     # reserved new-site block: its sonde values are never read
            raw = pd.read_parquet(f)
            raw.index = pd.to_datetime(raw["date"]) if "date" in raw.columns else pd.to_datetime(raw.index)
            for v in SONDE_VARS:
                mean_c = f"{v}_mean" if f"{v}_mean" in raw else (f"{v}_median" if f"{v}_median" in raw else None)
                if mean_c is None:
                    continue
                def tz(c):
                    x = raw[c].astype(float) if c in raw else pd.Series(np.nan, index=raw.index)
                    off = SONDE_LOG.get(v)
                    return np.log(np.maximum(x, 0) + off) if off else x
                z = pd.DataFrame({f"{v}__mean": tz(mean_c), f"{v}__min": tz(f"{v}_min"), f"{v}__max": tz(f"{v}_max")})
                z = z[~z.index.duplicated()]
                feats.append(ftd.sensor_features(z, cal, "sonde", v).reindex(dates))
        if feats:
            sb = pd.concat(feats, axis=1)
            ages = sb[[c for c in sb.columns if c.endswith("__age")]].min(axis=1)
            sb["sonde__present"] = ages < ftd.SENSOR_PRESENT_DAYS
        else:
            sb = pd.DataFrame({"sonde__present": False}, index=dates)
        blk = pd.concat([blk, sb], axis=1)
        # lab history (strictly before d) + targets (same day)
        anyh = np.zeros(len(dates), bool)
        for v, L in labs.items():
            l = L[L.site_id == site]
            ld = l["date"].to_numpy().astype("datetime64[D]")
            cd = dates.to_numpy().astype("datetime64[D]")
            pos = np.searchsorted(ld, cd, side="left") - 1
            age = np.where(pos >= 0, (cd - ld[np.clip(pos, 0, None)]).astype(int) if len(ld) else 10**6, 10**6)
            ok = (pos >= 0) & (age <= LABHIST_MAX_AGE)
            zz = l["z"].to_numpy(float)
            blk[f"labhist__{v}__value"] = np.where(ok, zz[np.clip(pos, 0, None)] if len(zz) else np.nan, np.nan)
            blk[f"labhist__{v}__age"] = np.where(ok, age, np.nan)
            anyh |= ok
            li = l.set_index("date")
            blk[f"ylab__{v}"] = li["z"].reindex(dates).to_numpy(float)
            blk[f"ylab__{v}__cens"] = li["cens"].reindex(dates).fillna(False).to_numpy(bool)
        blk["labhist__present"] = anyh
        # weather
        wx = wxds.to_table(filter=ds.field("site_key") == site, columns=["date"] + ftd.WX_VARS).to_pandas()
        if len(wx):
            wx["date"] = pd.to_datetime(wx["date"])
            wb = ftd.weather_features(wx.drop_duplicates("date"), cal).reindex(dates)
            wb["wx__present"] = wb["wx__precip_sum7"].notna()
        else:
            wb = pd.DataFrame({"wx__present": False}, index=dates)
        blk = pd.concat([blk, wb], axis=1)
        # static
        blk["static__lat"] = float(reg.loc[site, "lat"]) if site in reg.index else np.nan
        blk["static__lon"] = float(reg.loc[site, "lon"]) if site in reg.index else np.nan
        blk["static__log_drain_km2"] = np.log10(float(reg.loc[site, "drain_area_km2"]) + 1) \
            if site in reg.index and pd.notna(reg.loc[site, "drain_area_km2"]) else np.nan
        found = site in sc.index and bool(sc.loc[site, "streamcat_found"])
        for col in ftd.STATIC_COLS:
            v = float(sc.loc[site, col]) if found and col in sc.columns and pd.notna(sc.loc[site, col]) else np.nan
            blk[f"static__{col}"] = np.log1p(v) if col in ("wsareasqkm", "popden2010ws", "fertws", "manurews") and np.isfinite(v) else v
        blk["static__present"] = True
        doy = dates.dayofyear.to_numpy(float)
        blk["time__doy_sin"] = np.sin(2 * np.pi * (doy - 1) / 365.25)
        blk["time__doy_cos"] = np.cos(2 * np.pi * (doy - 1) / 365.25)
        blk["time__present"] = True
        blk.index.name = "date"
        blk = blk.reset_index()
        blk.insert(0, "site_id", site)
        blk.insert(1, "role", roles.get(no))
        blk.insert(2, "basin_group", split["basin_group"].get(no))
        out.append(blk)
        if i % 200 == 0:
            log(f"  {i} sites done")
    rows = pd.concat(out, ignore_index=True)
    sat = s2_matchup_features()
    if len(sat):
        sat = sat.set_index(["site_id", "date"])
        key = pd.MultiIndex.from_frame(rows[["site_id", "date"]])
        S = sat.reindex(key)
        for c in S.columns:
            rows[c] = S[c].to_numpy(float)
        rows["satellite__present"] = S.notna().any(axis=1).to_numpy()
    else:
        rows["satellite__present"] = False
    return rows


NON_HUB_W = ["sonde", "labhist", "satellite"]
HUB_W = ["wx", "static", "time"]


def main_build() -> None:
    import datetime as dt
    from .fuse_tiers import availability_patterns
    rows = build_tier_w()
    out = paths.processed("fuse_tier_w")
    f1 = out / "rows.parquet"
    rows.to_parquet(f1, compression="zstd", index=False)
    cols = {m: [c for c in rows.columns if c.startswith(f"{m}__") and not c.endswith("__present")]
            for m in NON_HUB_W + HUB_W}
    (out / "columns.json").write_text(json.dumps(cols, indent=1))
    av = availability_patterns(rows.rename(columns={"site_id": "site"}), NON_HUB_W)
    av.to_csv(out / "availability_all.csv", index=False)
    stats = {"rows": int(len(rows)), "sites": int(rows.site_id.nunique()),
             "rows_by_role": rows["role"].value_counts().to_dict(),
             "first": str(rows.date.min().date()), "last": str(rows.date.max().date()),
             "present_rows": {m: int(rows[f"{m}__present"].sum()) for m in NON_HUB_W + HUB_W},
             "target_rows": {v: int(rows[f"ylab__{v}"].notna().sum()) for v in W_TARGETS},
             "patterns": av.to_dict("records"),
             "rules": {"calendar": CAL, "labhist_max_age_days": LABHIST_MAX_AGE, "labhist_strictly_before": True,
                       "sonde_on_or_before_d": True, "log10_offsets": W_TARGETS,
                       "roles": "copied from docs/preregistration/sonde_split.json (sha256 fd4bb14f…)"},
             "retrieved_inputs": {n: (read_manifest(n)["created"] if (paths.MANIFESTS / f"{n}.json").exists() else None)
                                  for n in ("wqp_discrete", "sonde_daily", "s2_wqp_matchups", "forcings_daily", "streamcat")},
             "built": dt.date.today().isoformat()}
    files = [f1, out / "columns.json", out / "availability_all.csv"]
    write_manifest("fuse_tier_w", files=files,
                   sources=["manifest:wqp_discrete", "manifest:sonde_daily", "manifest:s2_wqp_matchups",
                            "manifest:forcings_daily", "manifest:streamcat", "manifest:sites"],
                   description="SENTINEL-Fuse Tier W: WQP lab site-days at USGS sonde sites with sonde, lab-history, "
                               "satellite and hub modalities.",
                   license="US public domain (USGS, WQP, gridMET, StreamCat); Copernicus open (S2)", synthetic=False,
                   stats=stats)
    print(json.dumps({k: stats[k] for k in ("rows", "sites", "rows_by_role", "present_rows", "target_rows")}, indent=1))
    print(av.to_string())
