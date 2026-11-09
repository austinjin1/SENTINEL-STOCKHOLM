# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Fuse Tier C — cyanotoxin (microcystin) rows with co-located modalities.

Row unit: (site, local sample date) with a total-microcystin result.

Sources
* WQP WQX 3.0 ``processed('wqp_discrete')`` variable ``microcystins`` (total microcystins/nodularins);
  EPA NARS submissions (organization ``NARS_WQX``) are dropped because NLA is taken from
  ``processed('nars_lakes')`` directly; ``facility`` and ``estuary_ocean`` site classes are dropped.
* EPA NLA 2007/2012/2017/2022 (``processed('nars_lakes')``): index-site microcystin per visit.
* Satellite: re-extracted Sentinel-2 features of the v1 benchmark (``processed('mc_benchmark_v2')``,
  ``qc_lenient`` rows, ±7 d window) joined on (WQP site_id, date).

Modalities (pre-registered in docs/preregistration/fuse_protocol.md §3.2)
* ``pigment``  — same-day lab chlorophyll-a / phycocyanin (WQP) or NLA chl-a.
* ``nutrient`` — same-day TP, TN (or TKN + NO3NO2), NO3NO2, NH3/NH4, ortho-P, DOC.
* ``physchem`` — same-day water temperature, DO, pH, specific conductance, turbidity, Secchi.
* ``satellite``— S2 L2A water-masked band medians and indices (mc_benchmark_v2).
* ``history``  — prior microcystin results at the same site dated <= d − 7 (lab latency), 365-d window.
* ``profile``  — NLA DO / temperature depth profile summaries.
* ``zoop``, ``benthic``, ``phyto`` — NLA community summaries (phytoplankton includes cyanobacteria
  biovolume, which is causally proximal to toxin production: reported with and without, see protocol).
* hub: ``wx`` (antecedent gridMET at the nearest SENTINEL forcing cell, windows ending d − 1),
  ``static`` (coordinates, site class, NLA lake area), ``time`` (day-of-year harmonics).

Other cyanotoxins measured in the same sample are never inputs (toxin-from-toxin circularity).
Censored microcystin results keep their detection limit (``mc_cens``, ``mc_dl``); exceedance labels
are NaN when a censored bound cannot decide them.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from ..core import paths
from ..core.manifest import read_manifest, write_manifest

THRESHOLDS = {"gt8": 8.0, "gt1p6": 1.6, "gt0p3": 0.3}   # EPA 2019 recreational; v1; EPA DW HA (children)
KEEP_CLASSES = {"lake", "stream", "unknown", "wetland", "other"}
WQP_MODS = {
    "pigment": {"chla": ["chla", "chla_corrected", "chla_uncorrected"], "phycocyanin": ["phycocyanin"]},
    "nutrient": {"tp": ["tp"], "tn": ["tn"], "tkn": ["tkn"], "no3no2": ["no3no2", "no3"], "nh3": ["nh3"],
                 "orthop": ["orthop"], "doc": ["doc"]},
    "physchem": {"temp": ["temp"], "do": ["do"], "ph": ["ph"], "spcond": ["spcond"],
                 "turbidity": ["turbidity_ntu", "turbidity_fnu", "turbidity_ntru"], "secchi": ["secchi"]},
}
LOG_VARS = {"chla": 0.1, "phycocyanin": 0.1, "tp": 0.001, "tn": 0.01, "tkn": 0.01, "no3no2": 0.001,
            "nh3": 0.001, "orthop": 0.0005, "doc": 0.1, "spcond": 1.0, "turbidity": 0.1, "secchi": 0.01}
SAT_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]


def _log(v, x):
    return np.log10(np.maximum(np.asarray(x, float), 0) + LOG_VARS[v]) if v in LOG_VARS else np.asarray(x, float)


def _labels(vmax_unc: pd.Series, dl_cens_max: pd.Series) -> dict:
    out = {}
    for k, t in THRESHOLDS.items():
        pos = vmax_unc > t
        neg = (vmax_unc.fillna(-np.inf) <= t) & (dl_cens_max.fillna(-np.inf) <= t)
        out[f"y_{k}"] = np.where(pos, 1.0, np.where(neg & (vmax_unc.notna() | dl_cens_max.notna()), 0.0, np.nan))
    return out


CONGENERS = ["microcystin_la", "microcystin_lr", "microcystin_rr", "microcystin_yr", "microcystin_other_congener"]
_WQP_COLS = ["site_id", "state", "huc8", "lat", "lon", "site_class", "activity_start_date", "value", "censored",
             "detection_limit", "organization", "characteristic"]


def _read_mc(variable: str) -> pd.DataFrame:
    m = pd.read_parquet(paths.processed("wqp_discrete"), filters=[("variable", "=", variable)], columns=_WQP_COLS)
    m = m[(m["organization"] != "NARS_WQX") & m["site_class"].isin(KEEP_CLASSES)]
    m = m[m["lat"].notna() & m["lon"].notna()].copy()
    m["date"] = pd.to_datetime(m["activity_start_date"]).dt.normalize()
    m["censored"] = m["censored"].fillna(False).astype(bool)
    return m


def wqp_microcystin() -> pd.DataFrame:
    """Total microcystins per site-day; site-days that report only congeners get the congener sum.

    Total: max uncensored result (``mc_max_unc``) and max censored DL (``mc_dl_cens``).
    Congener sum (``mc_basis='congener_sum'``, only where no total exists): sum over congeners of the
    per-congener max uncensored value (non-detects contribute 0); if no congener is detected the row
    is censored with bound = max congener DL.
    """
    m = _read_mc("microcystins")
    n0 = len(m)
    unc = m[~m["censored"]].groupby(["site_id", "date"])["value"].max()
    cdl = m[m["censored"]].groupby(["site_id", "date"])["detection_limit"].max()
    meta = m.groupby(["site_id", "date"]).agg(lat=("lat", "first"), lon=("lon", "first"), state=("state", "first"),
                                               huc8=("huc8", "first"), site_class=("site_class", "first"),
                                               organization=("organization", "first"), n_results=("value", "size"))
    meta["mc_max_unc"] = unc.reindex(meta.index)
    meta["mc_dl_cens"] = cdl.reindex(meta.index)
    meta["mc_basis"] = "total"
    cg = pd.concat([_read_mc(v) for v in CONGENERS], ignore_index=True)
    cg = cg[~pd.MultiIndex.from_frame(cg[["site_id", "date"]]).isin(meta.index)]
    if len(cg):
        per = cg[~cg["censored"]].groupby(["site_id", "date", "characteristic"])["value"].max()
        s_unc = per.groupby(level=[0, 1]).sum()
        s_dl = cg[cg["censored"]].groupby(["site_id", "date"])["detection_limit"].max()
        cm = cg.groupby(["site_id", "date"]).agg(lat=("lat", "first"), lon=("lon", "first"), state=("state", "first"),
                                                  huc8=("huc8", "first"), site_class=("site_class", "first"),
                                                  organization=("organization", "first"), n_results=("value", "size"))
        cm["mc_max_unc"] = s_unc.reindex(cm.index)
        cm["mc_dl_cens"] = np.where(cm["mc_max_unc"].isna(), s_dl.reindex(cm.index), np.nan)
        cm["mc_basis"] = "congener_sum"
        meta = pd.concat([meta, cm])
    meta = meta.reset_index()
    meta["source"] = "wqp"
    meta.attrs["n_raw"] = n0
    return meta


def nla_microcystin() -> pd.DataFrame:
    v = pd.read_parquet(paths.processed("nars_lakes") / "visits.parquet")
    lg = pd.read_parquet(paths.processed("nars_lakes") / "chemistry_long.parquet",
                         columns=["visit_id", "analyte", "value_canonical", "censored", "detection_limit_canonical"])
    mc = lg[lg["analyte"] == "microcystin"].copy()
    mc["censored"] = mc["censored"].fillna(False).astype(bool)
    unc = mc[~mc["censored"]].groupby("visit_id")["value_canonical"].max()
    cdl = mc[mc["censored"]].groupby("visit_id")["detection_limit_canonical"].max()
    vv = v.set_index("visit_id")
    ids = sorted(set(unc.index) | set(cdl.index))
    out = pd.DataFrame({"visit_id": ids})
    out["site_id"] = "NLA-" + vv.loc[ids, "site_id"].astype(str).to_numpy()
    out["date"] = pd.to_datetime(vv.loc[ids, "date"].to_numpy()).normalize()
    for c in ("lat", "lon", "state", "huc8", "area_ha", "survey"):
        out[c] = vv.loc[ids, c].to_numpy()
    out["huc8"] = out["huc8"].astype(str).str.zfill(8)
    out["site_class"] = "lake"
    out["organization"] = "EPA_NLA"
    out["n_results"] = 1
    out["mc_max_unc"] = unc.reindex(ids).to_numpy()
    out["mc_dl_cens"] = cdl.reindex(ids).to_numpy()
    out["source"] = "nla"
    out["mc_basis"] = "total"
    return out


def _wqp_var(var_names: list[str], sites: set) -> pd.DataFrame:
    parts = []
    for vn in var_names:
        try:
            d = pd.read_parquet(paths.processed("wqp_discrete"), filters=[("variable", "=", vn)],
                                columns=["site_id", "activity_start_date", "value", "censored", "detection_limit"])
        except Exception:   # noqa: BLE001 — variable partition may not exist
            continue
        parts.append(d[d["site_id"].isin(sites)])
    if not parts:
        return pd.DataFrame(columns=["site_id", "date", "value", "cens"])
    d = pd.concat(parts, ignore_index=True)
    d["date"] = pd.to_datetime(d["activity_start_date"]).dt.normalize()
    d["censored"] = d["censored"].fillna(False).astype(bool)
    d["v"] = np.where(d["censored"] & d["value"].isna(), d["detection_limit"], d["value"])
    g = d.groupby(["site_id", "date"])
    return pd.DataFrame({"value": g["v"].median(), "cens": g["censored"].all()}).reset_index()


def add_wqp_modalities(rows: pd.DataFrame, log=print) -> pd.DataFrame:
    w = rows["source"] == "wqp"
    sites = set(rows.loc[w, "site_id"])
    key = pd.MultiIndex.from_frame(rows[["site_id", "date"]])
    for mod, vars_ in WQP_MODS.items():
        anyp = np.zeros(len(rows), bool)
        for v, names in vars_.items():
            t = _wqp_var(names, sites).set_index(["site_id", "date"])
            val = t["value"].reindex(key).to_numpy(float)
            val = np.where(w.to_numpy(), val, np.nan)
            rows[f"{mod}__{v}"] = _log(v, val)
            anyp |= np.isfinite(val)
            log(f"  {mod}/{v}: {int(np.isfinite(val).sum())} rows")
        rows[f"{mod}__present"] = anyp
    return rows


def add_nla_modalities(rows: pd.DataFrame) -> pd.DataFrame:
    n = rows["source"] == "nla"
    cw = pd.read_parquet(paths.processed("nars_lakes") / "chemistry_wide.parquet").set_index("visit_id")
    vid = rows.loc[n, "visit_id"]
    def put(col, src, var):
        vals = cw[src].reindex(vid).to_numpy(float) if src in cw else np.full(len(vid), np.nan)
        rows.loc[n, col] = _log(var, vals)
    # NLA: TP in µg/L -> mg/L (exact 1e-3) to share the WQP column
    tp = cw["tp_ug_l"].reindex(vid).to_numpy(float) * 1e-3
    rows.loc[n, "nutrient__tp"] = _log("tp", tp)
    for col, src, var in [("nutrient__tn", "tn_mg_l", "tn"), ("nutrient__no3no2", "no3no2_n_mg_l", "no3no2"),
                          ("nutrient__nh3", "nh4_n_mg_l", "nh3"), ("nutrient__doc", "doc_mg_l", "doc"),
                          ("pigment__chla", "chla_ug_l", "chla"), ("physchem__ph", "ph_lab_su", "ph"),
                          ("physchem__spcond", "cond_us_cm", "spcond"), ("physchem__turbidity", "turbidity_ntu", "turbidity"),
                          ("physchem__secchi", "secchi_m", "secchi")]:
        put(col, src, var)
    # NO3 fallback where no3no2 missing
    no3 = _log("no3no2", cw["no3_n_mg_l"].reindex(vid).to_numpy(float))
    cur = rows.loc[n, "nutrient__no3no2"].to_numpy(float)
    rows.loc[n, "nutrient__no3no2"] = np.where(np.isfinite(cur), cur, no3)
    for mod in WQP_MODS:
        cols = [c for c in rows.columns if c.startswith(f"{mod}__") and c != f"{mod}__present"]
        rows.loc[n, f"{mod}__present"] = rows.loc[n, cols].notna().any(axis=1).to_numpy()
    # profiles: surface temp/DO, bottom DO, min DO, stratification (surface - bottom temp), depth
    pr = pd.read_parquet(paths.processed("nars_lakes") / "profiles.parquet")
    pr = pr[pr["visit_id"].isin(set(vid))].sort_values(["visit_id", "depth_m"])
    g = pr.groupby("visit_id")
    prof = pd.DataFrame({
        "profile__t_surf": g["temperature_c"].first(), "profile__t_bot": g["temperature_c"].last(),
        "profile__do_surf": g["do_mgl"].first(), "profile__do_bot": g["do_mgl"].last(),
        "profile__do_min": g["do_mgl"].min(), "profile__depth_max": g["depth_m"].max(),
        "profile__ph_surf": g["ph"].first(),
    })
    prof["profile__strat_dt"] = prof["profile__t_surf"] - prof["profile__t_bot"]
    for c in prof.columns:
        rows.loc[n, c] = prof[c].reindex(vid).to_numpy(float)
    rows["profile__present"] = rows[[c for c in prof.columns]].notna().any(axis=1) if len(prof) else False
    # communities: richness + log total abundance; phytoplankton cyanobacteria share
    for mod, fn, val in [("zoop", "zooplankton_counts.parquet", "density"), ("benthic", "benthic_counts.parquet", "count"),
                         ("phyto", "phytoplankton_counts.parquet", "biovolume")]:
        cc = pd.read_parquet(paths.processed("nars_lakes") / fn)
        cc = cc[cc["visit_id"].isin(set(vid))]
        vcol = val if val in cc.columns else "count"
        cc[vcol] = pd.to_numeric(cc[vcol], errors="coerce")
        tcol = next((c for c in ("taxa_id", "taxon", "target_taxon", "taxon_name", "name") if c in cc.columns), None)
        gg = cc.groupby("visit_id")
        feat = pd.DataFrame({f"{mod}__log_total": np.log10(gg[vcol].sum(min_count=1) + 1.0),
                             f"{mod}__richness": gg[tcol].nunique() if tcol else gg[vcol].size()})
        if mod == "phyto":
            grp = next((c for c in ("src_algal_group", "algal_group", "division", "phylum") if c in cc.columns), None)
            if grp is not None:
                cy = cc[grp].astype(str).str.lower().str.contains("cyano|blue")
                feat["phyto__cyano_frac"] = (cc[vcol].where(cy, 0).groupby(cc["visit_id"]).sum()
                                             / gg[vcol].sum(min_count=1)).astype(float)
                feat["phyto__log_cyano"] = np.log10(cc[vcol].where(cy, 0).groupby(cc["visit_id"]).sum() + 1.0)
        for c in feat.columns:
            rows.loc[n, c] = feat[c].reindex(vid).to_numpy(float)
        rows[f"{mod}__present"] = rows[list(feat.columns)].notna().any(axis=1)
    return rows


def add_satellite(rows: pd.DataFrame) -> pd.DataFrame:
    b = pd.read_parquet(paths.processed("mc_benchmark_v2") / "mc_benchmark_v2.parquet")
    b = b[b["qc_lenient"].fillna(False).astype(bool)].copy()
    b = b[b["dt_days"] <= 0]          # decision-time rule: imagery acquired on/before the sample date
    b["date"] = pd.to_datetime(b["date"]).dt.normalize()
    b = b.sort_values("abs_dt_days").drop_duplicates(["site_id", "date"])
    feats = {}
    for band in SAT_BANDS:
        feats[f"satellite__rw3_{band}"] = b[f"rw3_{band}"]
        feats[f"satellite__rw7_{band}"] = b[f"rw7_{band}"]
    r = b["rw3_B04"]
    feats["satellite__ndci"] = (b["rw3_B05"] - r) / (b["rw3_B05"] + r)
    feats["satellite__g_r"] = b["rw3_B03"] / r.where(r.abs() > 1e-6)
    feats["satellite__b5_b4"] = b["rw3_B05"] / r.where(r.abs() > 1e-6)
    feats["satellite__abs_dt_days"] = b["abs_dt_days"].astype(float)
    feats["satellite__dt_days"] = b["dt_days"].astype(float)
    feats["satellite__water_frac_c7"] = b["scl_c7_n_water"] / b["scl_c7_n"].where(b["scl_c7_n"] > 0)
    f = pd.DataFrame(feats)
    f.index = pd.MultiIndex.from_frame(b[["site_id", "date"]])
    # second source: national S2 x WQP matchups (qc_pass), only where mc_benchmark_v2 has nothing
    from .fuse_tier_w import s2_matchup_features
    s2 = s2_matchup_features()
    if len(s2):
        s2 = s2.set_index(["site_id", "date"])
        s2 = s2[~s2.index.isin(f.index)]
        s2 = s2[s2["satellite__dt_days"] <= 0] if "satellite__dt_days" in s2 else s2
        f = pd.concat([f, s2[[c for c in f.columns if c in s2.columns]]])
    key = pd.MultiIndex.from_frame(rows[["site_id", "date"]])
    sat = f.reindex(key)
    for c in sat.columns:
        rows[c] = sat[c].to_numpy(float)
    rows["satellite__present"] = sat.notna().any(axis=1).to_numpy() & (rows["source"] == "wqp").to_numpy()
    return rows


def add_history(rows: pd.DataFrame, lag_days: int = 7, window: int = 365) -> pd.DataFrame:
    """Prior microcystin results at the same site dated <= d - lag (lab latency)."""
    r = rows[["site_id", "date", "mc_max_unc", "mc_dl_cens"]].copy()
    r["v"] = np.log10(np.where(r["mc_max_unc"].notna(), r["mc_max_unc"], r["mc_dl_cens"]).astype(float) + 0.01)
    r["ex8"] = (r["mc_max_unc"] > 8).astype(float)
    out = {k: np.full(len(rows), np.nan) for k in ("n", "last", "max", "days", "rate8")}
    for site, g in r.groupby("site_id", sort=False):
        if len(g) < 2:
            continue
        g = g.sort_values("date")
        t = g["date"].to_numpy().astype("datetime64[D]").astype(int)
        v, e = g["v"].to_numpy(float), g["ex8"].to_numpy(float)
        for i, idx in enumerate(g.index):
            hi = np.searchsorted(t, t[i] - lag_days, side="right")
            lo = np.searchsorted(t, t[i] - window, side="left")
            if hi <= lo:
                continue
            out["n"][rows.index.get_loc(idx)] = hi - lo
            out["last"][rows.index.get_loc(idx)] = v[hi - 1]
            out["max"][rows.index.get_loc(idx)] = np.nanmax(v[lo:hi])
            out["days"][rows.index.get_loc(idx)] = t[i] - t[hi - 1]
            out["rate8"][rows.index.get_loc(idx)] = np.nanmean(e[lo:hi])
    for k, a in out.items():
        rows[f"history__{k}"] = a
    rows["history__present"] = np.isfinite(rows["history__n"].to_numpy())
    return rows


def add_hub(rows: pd.DataFrame, log=print) -> pd.DataFrame:
    from ..models.downstream import weather as wx  # shared gridMET-nearest-cell helper (read-only use)
    sites = rows.drop_duplicates("site_id")[["site_id", "lat", "lon"]].reset_index(drop=True)
    nc = wx.nearest_cells(sites["lat"], sites["lon"])
    sites = pd.concat([sites, nc], axis=1)
    m = sites.set_index("site_id")
    rows["wx_site_key"] = rows["site_id"].map(m["wx_site_key"])
    rows["wx_cell_km"] = rows["site_id"].map(m["wx_cell_km"]).astype(float)
    ser = wx.load_series(rows["wx_site_key"].dropna().unique(), start="2002-01-01",
                         variables=("precip_mm", "tmax_c", "tmin_c", "srad_wm2", "wind_ms", "vpd_kpa"))
    feats = []
    ser_by = {k: g for k, g in ser.groupby("site_key", sort=False)}
    for sk, g in rows.groupby("wx_site_key", sort=False):
        if sk not in ser_by:
            continue
        s = ser_by[sk].set_index("date").sort_index()
        s = s[~s.index.duplicated()]
        cal = pd.date_range(s.index.min(), s.index.max(), freq="D")
        s = s.reindex(cal)
        f = pd.DataFrame(index=cal)
        p = s["precip_mm"]
        for k in (3, 7, 30):
            f[f"wx__precip_sum{k}"] = p.rolling(k, min_periods=k // 2 + 1).sum()
        for v in ("tmax_c", "tmin_c"):
            for k in (7, 30):
                f[f"wx__{v}_m{k}"] = s[v].rolling(k, min_periods=k // 2 + 1).mean()
        for v in ("srad_wm2", "wind_ms", "vpd_kpa"):
            f[f"wx__{v}_m7"] = s[v].rolling(7, min_periods=4).mean()
        f = f.shift(1)  # windows end at d - 1
        q = f.reindex(pd.DatetimeIndex(g["date"]))
        q.index = g.index
        feats.append(q)
    F = pd.concat(feats).reindex(rows.index)
    for c in F.columns:
        rows[c] = F[c].to_numpy(float)
    rows["wx__present"] = rows["wx__precip_sum7"].notna() & (rows["wx_cell_km"] <= 50)
    for c in F.columns:
        rows.loc[~rows["wx__present"], c] = np.nan
    rows["static__lat"] = rows["lat"].astype(float)
    rows["static__lon"] = rows["lon"].astype(float)
    for cl in ("lake", "stream"):
        rows[f"static__class_{cl}"] = (rows["site_class"] == cl).astype(float)
    rows["static__log_area_ha"] = np.log10(pd.to_numeric(rows.get("area_ha"), errors="coerce").astype(float) + 1)
    rows["static__present"] = True
    doy = rows["date"].dt.dayofyear.to_numpy(float)
    rows["time__doy_sin"] = np.sin(2 * np.pi * (doy - 1) / 365.25)
    rows["time__doy_cos"] = np.cos(2 * np.pi * (doy - 1) / 365.25)
    rows["time__year"] = rows["date"].dt.year.astype(float)
    rows["time__present"] = True
    return rows


def drop_nla_duplicates(w: pd.DataFrame, n: pd.DataFrame, km: float = 1.0, log=print) -> pd.DataFrame:
    """Drop WQP site-days that duplicate an NLA visit (same date, <= ``km`` apart): state agencies
    (e.g. MNPCA, OREGONDEQ) also submit NLA samples to WQP under their own site ids. NLA is kept."""
    from ..eval.splits import haversine_km
    m = w.reset_index().merge(n[["date", "lat", "lon"]].rename(columns={"lat": "nlat", "lon": "nlon"}), on="date")
    d = haversine_km(m["lat"], m["lon"], m["nlat"], m["nlon"])
    drop = set(m.loc[d <= km, "index"])
    log(f"dropped {len(drop)} WQP site-days duplicating an NLA visit (<= {km} km, same date)")
    return w.drop(index=list(drop)).reset_index(drop=True)


NON_HUB_C = ["pigment", "nutrient", "physchem", "satellite", "history", "profile", "zoop", "benthic", "phyto"]
HUB_C = ["wx", "static", "time"]


def build_tier_c(log=lambda *a: print(*a, flush=True)) -> pd.DataFrame:
    w = wqp_microcystin()
    log(f"WQP microcystin site-days: {len(w)} (raw results {w.attrs.get('n_raw')})")
    n = nla_microcystin()
    log(f"NLA microcystin visits: {len(n)}")
    n = n.drop_duplicates(["site_id", "date"]).reset_index(drop=True)
    w = drop_nla_duplicates(w, n, km=1.0, log=log)
    rows = pd.concat([w, n], ignore_index=True)
    log(f"rows by basis: {rows['mc_basis'].value_counts().to_dict()}")
    rows["huc8"] = rows["huc8"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(8)
    rows["huc2"] = rows["huc8"].str[:2]
    rows["huc4"] = rows["huc8"].str[:4]
    lab = _labels(rows["mc_max_unc"], rows["mc_dl_cens"])
    for k, v in lab.items():
        rows[k] = v
    val = np.where(rows["mc_max_unc"].notna(), rows["mc_max_unc"], rows["mc_dl_cens"]).astype(float)
    rows["mc_cens"] = rows["mc_max_unc"].isna() & rows["mc_dl_cens"].notna()
    rows["mc_log10"] = np.log10(np.maximum(val, 1e-3))   # value, or the DL when censored (see mc_cens)
    # Congener sums: several programmes (e.g. Florida) report non-detect congeners at their reporting
    # limit without a censoring flag, so a sum of ~10 congeners carries ~2-3 ug/L of reporting limits.
    # Only the 8 ug/L label is kept for them (an RL sum alone cannot exceed it); lower thresholds and
    # the regression target are voided (NaN) for congener-sum rows.
    cg = (rows["mc_basis"] == "congener_sum").to_numpy()
    rows["y_gt8_total"] = np.where(cg, np.nan, rows["y_gt8"])   # P2 primary label (total-microcystin basis)
    for k in ("y_gt1p6", "y_gt0p3", "mc_log10"):
        rows.loc[cg, k] = np.nan
    rows = add_wqp_modalities(rows, log=log)
    log("nla modalities"); rows = add_nla_modalities(rows)
    log("satellite"); rows = add_satellite(rows)
    log("history"); rows = add_history(rows)
    log("hub"); rows = add_hub(rows, log=log)
    for m in NON_HUB_C + HUB_C:
        if f"{m}__present" in rows:
            rows[f"{m}__present"] = rows[f"{m}__present"].fillna(False).astype(bool)
        else:
            rows[f"{m}__present"] = False
    rows["row_id"] = (rows["source"] + "|" + rows["site_id"] + "|" + rows["date"].dt.strftime("%Y-%m-%d"))
    return rows


def modality_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    return {m: [c for c in df.columns if c.startswith(f"{m}__") and not c.endswith("__present")]
            for m in NON_HUB_C + HUB_C}


def main_build() -> None:
    import datetime as dt
    from .fuse_tiers import availability_patterns
    rows = build_tier_c()
    out = paths.processed("fuse_tier_c")
    f1 = out / "rows.parquet"
    rows.to_parquet(f1, compression="zstd", index=False)
    (out / "columns.json").write_text(json.dumps(modality_columns(rows), indent=1))
    av = availability_patterns(rows.rename(columns={"site_id": "site"}), NON_HUB_C)
    av.to_csv(out / "availability_all.csv", index=False)
    lab = rows["y_gt8"].notna().to_numpy()
    av8 = availability_patterns(rows.rename(columns={"site_id": "site"}), NON_HUB_C, extra_mask=lab)
    av8.to_csv(out / "availability_labelled_gt8.csv", index=False)
    pos = {k: {"labelled": int(rows[f"y_{k}"].notna().sum()), "pos": int((rows[f"y_{k}"] == 1).sum()),
               "pos_sites": int(rows.loc[rows[f"y_{k}"] == 1, "site_id"].nunique())} for k in THRESHOLDS}
    by_src = rows.groupby("source").agg(rows=("site_id", "size"), sites=("site_id", "nunique"),
                                        pos8=("y_gt8", lambda s: int((s == 1).sum()))).to_dict("index")
    stats = {"rows": int(len(rows)), "sites": int(rows.site_id.nunique()), "by_source": by_src, "labels": pos,
             "first": str(rows.date.min().date()), "last": str(rows.date.max().date()),
             "present_rows": {m: int(rows[f"{m}__present"].sum()) for m in NON_HUB_C + HUB_C},
             "huc2_regions": int(rows["huc2"].nunique()),
             "top_patterns_labelled_gt8": av8.head(15).to_dict("records"),
             "rules": {"thresholds_ugL": THRESHOLDS, "history_lag_days": 7, "history_window_days": 365,
                       "wx_window_end": "d-1", "wx_max_cell_km": 50, "keep_site_classes": sorted(KEEP_CLASSES),
                       "dropped_orgs": ["NARS_WQX"], "satellite": "mc_benchmark_v2 qc_lenient, ±7 d"},
             "retrieved_inputs": {n: (read_manifest(n)["created"] if (paths.MANIFESTS / f"{n}.json").exists() else None)
                                  for n in ("wqp_discrete", "nars_lakes", "mc_benchmark_v2", "forcings_daily")},
             "built": dt.date.today().isoformat()}
    files = [f1, out / "columns.json", out / "availability_all.csv", out / "availability_labelled_gt8.csv"]
    write_manifest("fuse_tier_c", files=files,
                   sources=["manifest:wqp_discrete", "manifest:nars_lakes", "manifest:mc_benchmark_v2",
                            "manifest:forcings_daily"],
                   description="SENTINEL-Fuse Tier C: microcystin site-days (WQP + NLA) with co-located pigment, "
                               "nutrient, physchem, satellite, history, NLA profile/community modalities and hub.",
                   license="US public domain (WQP, EPA NARS); Copernicus open (S2)", synthetic=False, stats=stats)
    print(json.dumps({k: stats[k] for k in ("rows", "sites", "by_source", "labels", "present_rows")}, indent=1,
                     default=str))
    print(av8.head(25).to_string())
