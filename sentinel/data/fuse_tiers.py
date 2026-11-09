# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Aligned multimodal tables for SENTINEL-Fuse (RESEARCH_PLAN.md §6, docs/research/05 §e.2).

Tier D (NEON, 34 aquatic sites): one row per (site, local issue date ``date``). Every input block
is computed only from observations dated <= the issue date (end-of-day issue time), with these
pre-registered freshness rules (docs/preregistration/fuse_protocol.md §3):

* sensor modalities (``sonde``, ``nitrate``, ``temp``, ``hydro``): daily series of the downstream
  sensor station (HOR 102 → 112 for wadeable streams, 103 for lakes/rivers); a block is *present*
  when at least one of its variables was observed on d-6 … d;
* ``chem`` (lab chemistry): the latest chemistry event strictly BEFORE d (so a same-day lab sample
  can be a target without being an input), age <= 35 days;
* ``micro`` (16S community): the latest 16S event on or before d, age <= 65 days (pointer
  ``micro__sample_id`` into ``micro_genus.parquet``);
* ``macroinv``/``periphyton``: latest event on or before d, age <= 120 days (NEON P2 tables,
  engineered community / lab metrics — :func:`macroinv_events`, :func:`periphyton_events`);
* ``hydro``: surface-water elevation anomaly (one HOR per site, minus trailing 365-d median) and
  log discharge (RELEASE-2026 discharge ends WY2021);
* hub (always present when the source covers the site): ``wx`` antecedent observed weather windows
  ending at d (gridMET/ERA5 *analysis*, never future days), ``static`` catchment/site attributes,
  ``time`` (day-of-year harmonics).

Targets (``y__<var>__h<k>``) are the station's daily values at d + k; lab targets ``ylab__<analyte>``
exist on days with a chemistry event (value, ``__cens`` flag, ``__dl`` detection limit). Hazard
labels use ``sentinel.eval.hazards`` definitions. Provisional NEON values are dropped (RELEASE-2026
only). Missing modalities stay missing — nothing is imputed here.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from ..core import paths
from ..eval import hazards as hz

# --------------------------------------------------------------------------- constants
SENSOR_PRODUCTS = {
    # modality: (processed dir, daily subdir or None, variables)
    "sonde": ("neon_sonde_daily", "sites", ["do_mgl", "spcond_uscm", "ph", "turb_fnu", "fdom_qsu", "chl_ugl"]),
    "nitrate": ("neon_nitrate", "daily", ["no3n_mgl"]),
    "temp_tsw": ("neon_tsw", "daily", ["temp_c"]),
    "temp_tsd": ("neon_tsd", "daily", ["temp_c"]),
    "hydro_eos": ("neon_eos", "daily", ["wse_m"]),     # P2 surface-water elevation (one HOR per site)
    "hydro_q": ("neon_discharge", "daily", ["q_ls"]),  # P2 continuous discharge (RELEASE ends WY2021)
}
SINGLE_HOR_PRODUCTS = {"hydro_eos", "hydro_q"}   # HORs are not interchangeable (absolute elevations)
ANOMALY_VARS = {"wse_m": 365}                     # value − trailing 365-d median (past-only) per site
# log(x + offset) like M1 (sentinel/models/sonde/data.py LOG_VARS); others linear.
LOG_OFFSET = {"spcond_uscm": 1.0, "turb_fnu": 0.1, "fdom_qsu": 0.1, "chl_ugl": 0.1, "no3n_mgl": 0.01, "q_ls": 1.0}
HOR_PREF = [102, 112, 103]          # downstream station set (streams), buoy (lakes / rivers)
TARGET_VARS = ["do_mgl", "spcond_uscm", "ph", "turb_fnu", "fdom_qsu", "chl_ugl", "no3n_mgl", "temp_c"]
HORIZONS = [1, 3, 7, 14]
CHEM_MAX_AGE = 35
MICRO_MAX_AGE = 65
BIO_MAX_AGE = 120
SENSOR_PRESENT_DAYS = 7
CHEM_ANALYTES = ["tn_mgl", "tp_mgl", "tdn_mgl", "tdp_mgl", "no3_no2_n_mgl", "nh4_n_mgl", "ortho_p_mgl",
                 "doc_mgl", "toc_mgl", "dic_mgl", "tss_mgl", "cl_mgl", "so4_mgl", "ca_mgl", "mg_mgl",
                 "na_mgl", "k_mgl", "si_mgl", "f_mgl", "br_mgl", "fe_mgl", "mn_mgl", "tds_mgl",
                 "uv_absorbance_254_nm_au", "tpc_ugl", "tpn_ugl"]
# µg/L duplicates of mg/L analytes (a handful of rows): converted with the exact factor 1e-3.
UGL_TO_MGL = {"no3_no2_n_ugl": "no3_no2_n_mgl", "nh4_n_ugl": "nh4_n_mgl", "ortho_p_ugl": "ortho_p_mgl",
              "tp_ugl": "tp_mgl", "tdp_ugl": "tdp_mgl", "no2_n_ugl": "no2_n_mgl"}
LAB_TARGETS = ["tn_mgl", "tp_mgl", "doc_mgl", "no3_no2_n_mgl", "nh4_n_mgl", "ortho_p_mgl", "tss_mgl",
               "cl_mgl", "so4_mgl", "si_mgl", "ca_mgl"]
LAB_LOG_OFFSET = {"tn_mgl": 0.01, "tp_mgl": 0.001, "doc_mgl": 0.1, "no3_no2_n_mgl": 0.001,
                  "nh4_n_mgl": 0.001, "ortho_p_mgl": 0.0005, "tss_mgl": 0.1, "cl_mgl": 0.01,
                  "so4_mgl": 0.01, "si_mgl": 0.01, "ca_mgl": 0.01}
WX_VARS = ["precip_mm", "tmax_c", "tmin_c", "srad_wm2", "wind_ms", "rhmin_pct", "vpd_kpa", "pet_mm"]
STATIC_COLS = ["wsareasqkm", "pctow2019ws", "pcturbop2019ws", "pcturblo2019ws", "pcturbmd2019ws",
               "pcturbhi2019ws", "pctdecid2019ws", "pctconif2019ws", "pctmxfst2019ws", "pctshrb2019ws",
               "pctgrs2019ws", "pcthay2019ws", "pctcrop2019ws", "pctwdwet2019ws", "pcthbwet2019ws",
               "pctimp2019ws", "pctagdrainagews", "fertws", "manurews", "popden2010ws", "rddensws",
               "damdensws", "wwtpalldensws", "precip9120ws", "tmean9120ws", "bfiws", "runoffws",
               "wetindexws", "elevws", "claycat", "sandcat", "omcat", "permcat", "kffactcat", "iwi",
               "ici", "nhd_slope", "nhd_streamorde"]
# Hazards evaluated on Tier D (daily-tier definitions; NEON column names mapped).
TIER_D_HAZARDS = {"H1_hypoxia": "do_mgl__min", "H1_hypoxia_t4": "do_mgl__min",
                  "H5b_chl_bloom": "chl_ugl__mean", "H4_nitrate": "no3n_mgl__mean"}
CALENDAR = ("2016-01-01", "2025-06-30")   # RELEASE-2026 window (provisional data dropped)


def to_model(var: str, x):
    x = np.asarray(x, dtype="float64")
    off = LOG_OFFSET.get(var)
    return np.log(np.maximum(x, 0.0) + off) if off is not None else x


def lab_to_model(var: str, x):
    x = np.asarray(x, dtype="float64")
    return np.log(np.maximum(x, 0.0) + LAB_LOG_OFFSET[var])


# --------------------------------------------------------------------------- sensor series
def station_daily(df: pd.DataFrame, variables: list[str], stats=("mean", "min", "max")) -> pd.DataFrame:
    """Coalesce a NEON daily sensor table to one downstream-station series per date.

    For each (date, variable) the first HOR in ``HOR_PREF`` (then lowest VER = shallowest) with a
    valid value is used; provisional rows are dropped. Returns columns ``<var>__<stat>`` and
    ``hor_used`` (HOR of the first variable's chosen value).
    """
    if df.empty:
        return pd.DataFrame()
    d = df[~df.get("provisional", pd.Series(False, index=df.index)).fillna(False).astype(bool)].copy()
    d["hor"] = pd.to_numeric(d["hor"], errors="coerce")
    d["ver"] = pd.to_numeric(d["ver"], errors="coerce")
    d = d[d["hor"].isin(HOR_PREF)]
    if d.empty:
        return pd.DataFrame()
    d["date"] = pd.to_datetime(d["date"])
    d["_pref"] = d["hor"].map({h: i for i, h in enumerate(HOR_PREF)})
    d = d.sort_values(["date", "_pref", "ver"])
    out = {}
    for v in variables:
        mcol = f"{v}_mean"
        if mcol not in d.columns:
            continue
        sub = d[d[mcol].notna()].drop_duplicates("date")
        for s in stats:
            c = f"{v}_{s}"
            if c in sub.columns:
                out[f"{v}__{s}"] = sub.set_index("date")[c]
        if "hor_used" not in out:
            out["hor_used"] = sub.set_index("date")["hor"]
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).sort_index()


def single_hor_daily(df: pd.DataFrame, var: str, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily series from ONE horizontal position per site (hydrology products).

    NEON EOS/discharge HORs measure different absolute levels (e.g. S1 vs S2 transducers, lake inlet
    vs outlet), so mixing them would create artificial jumps. The HOR (then lowest VER) with the most
    valid non-provisional daily means inside the calendar is used — a label-free, site-level choice.
    """
    if df.empty or f"{var}_mean" not in df.columns:
        return pd.DataFrame()
    d = df[~df.get("provisional", pd.Series(False, index=df.index)).fillna(False).astype(bool)].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d[d[f"{var}_mean"].notna() & d["date"].between(cal[0], cal[-1])]
    if d.empty:
        return pd.DataFrame()
    cnt = d.groupby(["hor", "ver"]).size().reset_index(name="n")
    cnt["hor_n"] = pd.to_numeric(cnt["hor"], errors="coerce")
    best = cnt.sort_values(["n", "hor_n", "ver"], ascending=[False, True, True]).iloc[0]
    sub = d[(d["hor"] == best["hor"]) & (d["ver"] == best["ver"])].drop_duplicates("date").set_index("date").sort_index()
    out = {f"{var}__{st}": sub[f"{var}_{st}"] for st in ("mean", "min", "max") if f"{var}_{st}" in sub}
    out = pd.DataFrame(out)
    out["hor_used"] = pd.to_numeric(best["hor"], errors="coerce")
    return out


def trailing_anomaly(z: pd.DataFrame, var: str, cal: pd.DatetimeIndex, window: int) -> pd.DataFrame:
    """Subtract the trailing (past-and-today only) ``window``-day median of the daily mean from
    mean/min/max — removes the site-specific absolute datum (e.g. water-surface elevation a.s.l.)."""
    m = z[f"{var}__mean"].reindex(cal)
    base = m.rolling(window, min_periods=30).median()
    out = pd.DataFrame(index=cal)
    for st in ("mean", "min", "max"):
        if f"{var}__{st}" in z:
            out[f"{var}__{st}"] = z[f"{var}__{st}"].reindex(cal) - base
    return out.dropna(how="all")


def _roll(s: pd.Series, w: int, fn: str, minp: int) -> pd.Series:
    r = s.rolling(w, min_periods=minp)
    return getattr(r, fn)()


def sensor_features(z: pd.DataFrame, cal: pd.DatetimeIndex, prefix: str, var: str) -> pd.DataFrame:
    """Engineered features at issue day d from a model-space daily series (mean/min/max columns).

    All windows end at d (inclusive). ``age`` = days since the last valid daily mean (capped at 30).
    """
    m = z[f"{var}__mean"].reindex(cal) if f"{var}__mean" in z else pd.Series(np.nan, index=cal)
    mn = z[f"{var}__min"].reindex(cal) if f"{var}__min" in z else pd.Series(np.nan, index=cal)
    mx = z[f"{var}__max"].reindex(cal) if f"{var}__max" in z else pd.Series(np.nan, index=cal)
    obs = m.notna().astype(float)
    idx = pd.Series(np.where(m.notna(), np.arange(len(cal)), np.nan), index=cal).ffill()
    age = (np.arange(len(cal)) - idx).clip(upper=30)
    lastvalid = m.ffill(limit=30)
    p = f"{prefix}__{var}"
    f = pd.DataFrame({
        f"{p}__last": m, f"{p}__lastmin": mn, f"{p}__lastmax": mx,
        f"{p}__lag1": m.shift(1), f"{p}__m3": _roll(m, 3, "mean", 2), f"{p}__m7": _roll(m, 7, "mean", 4),
        f"{p}__m30": _roll(m, 30, "mean", 10), f"{p}__sd7": _roll(m, 7, "std", 4),
        f"{p}__min7": _roll(mn, 7, "min", 4), f"{p}__max7": _roll(mx, 7, "max", 4),
        f"{p}__lastvalid": lastvalid, f"{p}__age": age.where(idx.notna(), 30.0),
        f"{p}__frac30": _roll(obs, 30, "mean", 1),
    }, index=cal)
    f[f"{p}__d7"] = f[f"{p}__last"] - f[f"{p}__m7"]
    return f


def _read_site_tables(name: str, sub: str | None, site: str) -> pd.DataFrame:
    root = paths.PROCESSED / name
    if not root.exists():
        return pd.DataFrame()
    f = root / sub / f"{site}.parquet" if sub else root / f"{site}.parquet"
    return pd.read_parquet(f) if f.exists() else pd.DataFrame()


def discover_vars(name: str, sub: str | None) -> list[str]:
    root = paths.PROCESSED / name / (sub or "")
    files = sorted(glob.glob(str(root / "*.parquet")))
    if not files:
        return []
    cols = pd.read_parquet(files[0]).columns
    return sorted({c[:-5] for c in cols if c.endswith("_mean")} - {"thermistor_depth_m", "sensor_depth_m"})


# --------------------------------------------------------------------------- chemistry
def chem_events(site_offsets: dict[str, float]) -> pd.DataFrame:
    """One row per (site, local date) chemistry event: per-analyte value (mean over replicate
    analyses of non-POM fractions), ``__cens`` (all analyses censored) and ``__dl`` (max DL).

    Censored analyses keep the published value (NaN for ND) — no DL/2 substitution.
    """
    c = pd.read_parquet(paths.processed("neon_chem") / "chem_long.parquet",
                        columns=["site", "time_utc", "var", "value", "censored", "detection_limit",
                                 "sample_fraction", "provisional"])
    c = c[~c["provisional"].fillna(False).astype(bool)]
    c = c[~c["sample_fraction"].astype(str).str.startswith("POM")]
    for ugl, mgl in UGL_TO_MGL.items():  # documented exact factor
        m = c["var"] == ugl
        c.loc[m, "value"] = c.loc[m, "value"] * 1e-3
        c.loc[m, "detection_limit"] = c.loc[m, "detection_limit"] * 1e-3
        c.loc[m, "var"] = mgl
    c = c[c["var"].isin(CHEM_ANALYTES)]
    off = c["site"].map(site_offsets).astype(float)
    t = pd.to_datetime(c["time_utc"], utc=True).dt.tz_localize(None)
    c["date"] = (t + pd.to_timedelta(off, unit="h")).dt.normalize()
    c["censored"] = c["censored"].fillna(False).astype(bool)
    g = c.groupby(["site", "date", "var"])
    agg = pd.DataFrame({"value": g["value"].mean(), "cens": g["censored"].all(),
                        "dl": g["detection_limit"].max()}).reset_index()
    idx = agg.set_index(["site", "date", "var"])
    wide = idx["value"].unstack("var")
    cens = idx["cens"].astype(float).unstack("var")
    dl = idx["dl"].unstack("var")
    out = wide.copy()
    for v in wide.columns:
        out[f"{v}__cens"] = cens[v] > 0.5
        out[f"{v}__dl"] = dl[v]
    return out.reset_index()


# --------------------------------------------------------------------------- 16S
def micro_events() -> tuple[pd.DataFrame, pd.DataFrame]:
    """16S events (``processed('microbe_wq')``) and their genus-level read-count matrix.

    The count matrix is label-free and row-wise, so building it for all events is identical to
    building it per fold; prevalence filters / CLR / PCA are fit on training folds in the model.
    """
    s = pd.read_parquet(paths.processed("microbe_wq") / "samples.parquet")
    t = pd.read_parquet(paths.processed("microbe_wq") / "tokens.parquet",
                        columns=["sample_id", "count", "genus"])
    g = t["genus"].fillna("__unassigned__").astype(str).replace({"": "__unassigned__"})
    gm = t.assign(genus=g).groupby(["sample_id", "genus"])["count"].sum().unstack(fill_value=0)
    ev = s[["sample_id", "event_key", "site_id", "collect_date", "n_reads", "n_seqs", "primer_set"]].copy()
    ev["date"] = pd.to_datetime(ev["collect_date"]).dt.normalize()
    return ev, gm.astype("float32")


def latest_event(cal: pd.DatetimeIndex, ev_dates: pd.Series, max_age: int, strictly_before: bool
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Index into the sorted event list of the latest event <= d (or < d) and its age in days."""
    ed = np.sort(pd.to_datetime(ev_dates).to_numpy().astype("datetime64[D]"))
    cd = cal.to_numpy().astype("datetime64[D]")
    if ed.size == 0:
        return np.full(cd.size, -1), np.full(cd.size, np.nan)
    side = "left" if strictly_before else "right"
    pos = np.searchsorted(ed, cd, side=side) - 1
    age = np.where(pos >= 0, (cd - ed[np.clip(pos, 0, None)]).astype(int), 10**6)
    ok = (pos >= 0) & (age <= max_age)
    return np.where(ok, pos, -1), np.where(ok, age, np.nan)


# --------------------------------------------------------------------------- biology (NEON P2)
EPT_ORDERS = {"Ephemeroptera", "Plecoptera", "Trichoptera"}
BENTHIC_HABITATS = {"EPILITHON", "EPIPSAMMON", "EPIXYLON", "EPIPHYTON", "EPIPELON"}
WATER_COLUMN_HABITATS = {"PHYTOPLANKTON", "SESTON"}


def _local_date(t, site, offs) -> pd.Series:
    tt = pd.to_datetime(t, utc=True).dt.tz_localize(None)
    return (tt + pd.to_timedelta(site.map(offs).astype(float), unit="h")).dt.normalize()


def macroinv_events(offs: dict) -> pd.DataFrame:
    """Per (site, local date) benthic macroinvertebrate metrics (DP1.20120.001, non-provisional).

    Densities (individuals m⁻²) are averaged over the event's samples; composition metrics use the
    event's summed densities: log10 total density, family richness, Shannon (family), and density
    fractions of EPT, Chironomidae, Oligochaeta, non-insects, Gastropoda, Amphipoda+Isopoda.
    """
    root = paths.PROCESSED / "neon_macroinv" / "density.parquet"
    if not root.exists():
        return pd.DataFrame(columns=["site", "date"])
    d = pd.read_parquet(root, columns=["siteID", "collectDate", "sampleID", "order", "family", "class",
                                       "density_per_m2", "provisional"])
    d = d[~d["provisional"].fillna(False).astype(bool) & d["density_per_m2"].notna()].copy()
    d["site"] = d["siteID"].astype(str)
    d["date"] = _local_date(d["collectDate"], d["site"], offs)
    d["family"] = d["family"].astype(object).where(d["family"].notna(), "__unk__").astype(str)
    n_samp = d.groupby(["site", "date"])["sampleID"].nunique()
    g = d.groupby(["site", "date"])
    tot = g["density_per_m2"].sum()
    fam = d.groupby(["site", "date", "family"])["density_per_m2"].sum()
    p = fam / fam.groupby(level=[0, 1]).transform("sum")
    shannon = -(p * np.log(p)).groupby(level=[0, 1]).sum()
    rich = fam[fam.index.get_level_values(2) != "__unk__"].groupby(level=[0, 1]).size()

    def frac(mask):
        return d[mask].groupby(["site", "date"])["density_per_m2"].sum().reindex(tot.index, fill_value=0.0) / tot

    out = pd.DataFrame({
        "log_density": np.log10(tot / n_samp.reindex(tot.index)),
        "fam_richness": rich.reindex(tot.index, fill_value=0).astype(float),
        "shannon_fam": shannon.reindex(tot.index),
        "frac_ept": frac(d["order"].isin(EPT_ORDERS)),
        "frac_chironomidae": frac(d["family"] == "Chironomidae"),
        "frac_oligochaeta": frac(d["class"] == "Oligochaeta"),
        "frac_noninsect": frac(d["class"] != "Insecta"),
        "frac_gastropoda": frac(d["class"] == "Gastropoda"),
        "frac_amph_isop": frac(d["order"].isin({"Amphipoda", "Isopoda"})),
        "n_samples": n_samp.reindex(tot.index).astype(float),
    })
    return out.reset_index()


def periphyton_events(offs: dict) -> pd.DataFrame:
    """Per (site, local date) periphyton / phytoplankton lab metrics (DP1.20163.001 external lab).

    Extract concentrations depend on field volumes, so the features are medians of log10 extract
    concentrations plus scale-free ratios (chl-a / (chl-a + pheophytin), C:N, N:P, C:chl-a) and stable
    isotopes (δ13C, δ15N, δ34S), separately for benthic habitats and the water column (sample-ID token).
    """
    root = paths.PROCESSED / "neon_periphyton" / "alg_algaeExternalLabDataPerSample.parquet"
    if not root.exists():
        return pd.DataFrame(columns=["site", "date"])
    c = pd.read_parquet(root, columns=["siteID", "collectDate", "sampleID", "analyte", "analyteConcentration",
                                       "provisional"])
    c = c[~c["provisional"].fillna(False).astype(bool) & c["analyteConcentration"].notna()].copy()
    c["site"] = c["siteID"].astype(str)
    c["date"] = _local_date(c["collectDate"], c["site"], offs)
    hab = c["sampleID"].astype(str).str.split(".").str[2].str.upper()
    c["zone"] = np.where(hab.isin(BENTHIC_HABITATS), "benthic", np.where(hab.isin(WATER_COLUMN_HABITATS), "water", "other"))
    c = c[c["zone"] != "other"]
    c["parent"] = c["sampleID"].astype(str).str.split(".").str[:4].str.join(".")
    w = c.pivot_table(index=["site", "date", "zone", "parent"], columns="analyte", values="analyteConcentration",
                      aggfunc="mean")
    pos = lambda k: w[k].where(w[k] > 0) if k in w else pd.Series(np.nan, index=w.index)
    f = pd.DataFrame(index=w.index)
    f["log_chl"] = np.log10(pos("chlorophyll a"))
    f["chl_frac"] = (pos("chlorophyll a") / (pos("chlorophyll a") + w["pheophytin"].clip(lower=0))).clip(0, 1) \
        if "pheophytin" in w else np.nan
    f["log_c"] = np.log10(pos("carbon"))
    f["log_cn"] = np.log10(pos("carbon") / pos("nitrogen"))
    f["log_np"] = np.log10(pos("nitrogen") / pos("phosphorus"))
    f["log_c_chl"] = np.log10(pos("carbon") / pos("chlorophyll a"))
    for k, n in (("d13C", "d13c"), ("d15N", "d15n"), ("d34S", "d34s")):
        f[n] = w[k] if k in w else np.nan
    f = f.replace([np.inf, -np.inf], np.nan)
    med = f.groupby(level=["site", "date", "zone"]).median()
    wide = med.unstack("zone")
    wide.columns = [f"{z}_{v}" for v, z in wide.columns]
    for z in ("benthic", "water"):
        for v in f.columns:
            if f"{z}_{v}" not in wide:
                wide[f"{z}_{v}"] = np.nan
    return wide.reset_index()


def bio_block(mod: str, ev: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Latest biology event on or before d (age <= BIO_MAX_AGE): ``<mod>__<metric>`` + present/age."""
    b = pd.DataFrame(index=cal)
    feats = [c for c in ev.columns if c not in ("site", "date")]
    if ev is None or len(ev) == 0:
        b[f"{mod}__present"] = False
        b[f"{mod}__age"] = np.nan
        return b
    ev = ev.sort_values("date").reset_index(drop=True)
    pos, age = latest_event(cal, ev["date"], BIO_MAX_AGE, strictly_before=False)
    b[f"{mod}__present"] = pos >= 0
    b[f"{mod}__age"] = age
    for k in feats:
        v = ev[k].to_numpy(float)
        b[f"{mod}__{k}"] = np.where(pos >= 0, v[np.clip(pos, 0, None)], np.nan)
    return b


# --------------------------------------------------------------------------- hub
def weather_features(wx: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Antecedent observed-weather windows ending at d (inclusive)."""
    w = wx.set_index("date").reindex(cal)
    f = {}
    p = w["precip_mm"]
    for k in (1, 3, 7, 30):
        f[f"wx__precip_sum{k}"] = p.rolling(k, min_periods=max(1, k // 2)).sum()
    f["wx__dry_days30"] = (p < 1.0).astype(float).where(p.notna()).rolling(30, min_periods=15).sum()
    for v in ("tmax_c", "tmin_c"):
        for k in (3, 7, 30):
            f[f"wx__{v}_m{k}"] = w[v].rolling(k, min_periods=max(1, k // 2)).mean()
    for v in ("srad_wm2", "wind_ms", "rhmin_pct", "vpd_kpa"):
        f[f"wx__{v}_m7"] = w[v].rolling(7, min_periods=4).mean()
    f["wx__pet_sum7"] = w["pet_mm"].rolling(7, min_periods=4).sum()
    out = pd.DataFrame(f, index=cal)
    return out


def _onset_labels(series: np.ndarray, hazard: str, h: int) -> tuple[np.ndarray, np.ndarray]:
    H = hz.HAZARDS[hazard]
    ev, on = hz.daily_onsets(series, H)
    y, valid = hz.issue_labels(series, ev, on, h, refractory_days=H.refractory_days)
    return y, valid


def _any_exceed(series: np.ndarray, hazard: str, h: int) -> tuple[np.ndarray, np.ndarray]:
    """y=1 if any day in (d, d+h] is beyond the threshold; valid if y=1 or >= 80% of days observed."""
    H = hz.HAZARDS[hazard]
    x = np.asarray(series, float)
    ev = (x < H.threshold) if H.direction == "below" else (x > H.threshold)
    ev &= np.isfinite(x)
    n = x.size
    c_ev = np.concatenate([[0], np.cumsum(ev)])
    c_ob = np.concatenate([[0], np.cumsum(np.isfinite(x))])
    y = np.zeros(n, int)
    valid = np.zeros(n, bool)
    for d in range(n - h):
        ne = c_ev[d + 1 + h] - c_ev[d + 1]
        cov = (c_ob[d + 1 + h] - c_ob[d + 1]) / h
        y[d] = int(ne > 0)
        valid[d] = y[d] == 1 or cov >= 0.8
    return y, valid


# --------------------------------------------------------------------------- Tier D
def build_tier_d(log=print) -> dict:
    """Build Tier D tables; returns a dict of DataFrames (``daily``, ``chem_events``,
    ``micro_events``, ``micro_genus``, ``modalities``)."""
    sites = pd.read_parquet(paths.processed("neon_sites") / "sites.parquet")
    offs = dict(zip(sites["site"], sites["utc_offset_h"].astype(float)))
    cal = pd.date_range(*CALENDAR, freq="D")
    chem = chem_events(offs)
    bio = {"macroinv": macroinv_events(offs), "periphyton": periphyton_events(offs)}
    log(f"bio events: macroinv={len(bio['macroinv'])} periphyton={len(bio['periphyton'])}")
    mev, mgen = micro_events()
    mev = mev.sort_values(["site_id", "date"]).reset_index(drop=True)
    sc = pd.read_parquet(paths.processed("catchment") / "streamcat_sites.parquet")
    sc = sc[sc["site_key"].str.startswith("NEON-")].set_index("site_key")
    reg = pd.read_parquet(paths.processed("sites") / "sites.parquet").set_index("site_key")
    hydro_vars = {k: discover_vars(SENSOR_PRODUCTS[k][0], SENSOR_PRODUCTS[k][1]) for k in ("hydro_eos", "hydro_q")}
    avail_products = {k: (paths.PROCESSED / v[0]).exists() for k, v in SENSOR_PRODUCTS.items()}
    log(f"sensor products present: {avail_products}; hydro vars: {hydro_vars}")
    import pyarrow.dataset as ds
    wxds = ds.dataset(str(paths.processed("forcings_daily") / "forcings_daily.parquet"))
    frames = []
    for site in sorted(sites["site"]):
        sk = f"NEON-{site}"
        stype = str(sites.loc[sites.site == site, "aquatic_type"].iloc[0])
        blocks = []
        series = {}   # model-space daily (mean/min/max) for targets
        # ---- sensors
        for mod, prods in (("sonde", ["sonde"]), ("nitrate", ["nitrate"]), ("temp", ["temp_tsw", "temp_tsd"]),
                           ("hydro", ["hydro_eos", "hydro_q"])):
            feats = []
            for pk in prods:
                name, sub, vars_ = SENSOR_PRODUCTS[pk]
                vars_ = vars_ or hydro_vars.get(pk, [])
                raw = _read_site_tables(name, sub, site)
                if raw.empty or not vars_:
                    continue
                if pk in SINGLE_HOR_PRODUCTS:
                    parts = [single_hor_daily(raw, v, cal) for v in vars_]
                    parts = [q.drop(columns="hor_used") for q in parts if not q.empty]
                    sd = pd.concat(parts, axis=1) if parts else pd.DataFrame()
                else:
                    sd = station_daily(raw, vars_)
                if sd.empty:
                    continue
                for v in vars_:
                    if f"{v}__mean" not in sd:
                        continue
                    zt = pd.DataFrame({f"{v}__{s}": to_model(v, sd[f"{v}__{s}"]) if f"{v}__{s}" in sd else np.nan
                                       for s in ("mean", "min", "max")}, index=sd.index)
                    if v in ANOMALY_VARS:
                        zt = trailing_anomaly(zt, v, cal, ANOMALY_VARS[v])
                    key = v if mod != "hydro" else f"{pk}_{v}"
                    if key in series:   # tsw vs tsd: a site has one of them
                        continue
                    series[key] = zt
                    feats.append(sensor_features(zt.rename(columns=lambda c: c.replace(v, key)), cal, mod, key))
            if feats:
                fb = pd.concat(feats, axis=1)
                ages = fb[[c for c in fb.columns if c.endswith("__age")]].min(axis=1)
                fb[f"{mod}__present"] = ages < SENSOR_PRESENT_DAYS
            else:
                fb = pd.DataFrame({f"{mod}__present": False}, index=cal)
            blocks.append(fb)
        # ---- chemistry (strictly before d)
        ce = chem[chem.site == site].sort_values("date").reset_index(drop=True)
        pos, age = latest_event(cal, ce["date"], CHEM_MAX_AGE, strictly_before=True)
        cb = pd.DataFrame(index=cal)
        cb["chem__present"] = pos >= 0
        cb["chem__age"] = age
        for a in CHEM_ANALYTES:
            if a in ce.columns:
                vals = ce[a].to_numpy(float)
                dl = ce[f"{a}__dl"].to_numpy(float) if f"{a}__dl" in ce else np.full(len(ce), np.nan)
                cens = ce[f"{a}__cens"].fillna(False).to_numpy(bool) if f"{a}__cens" in ce else np.zeros(len(ce), bool)
                # model-space value: published value, or the detection limit when censored & missing
                v_in = np.where(np.isnan(vals) & cens, dl, vals)
                off = LAB_LOG_OFFSET.get(a, 0.01)
                zv = np.log(np.maximum(v_in, 0) + off) if a != "uv_absorbance_254_nm_au" else v_in
                cb[f"chem__{a}"] = np.where(pos >= 0, zv[np.clip(pos, 0, None)], np.nan) if len(ce) else np.nan
                cb[f"chem__{a}__cens"] = np.where(pos >= 0, cens[np.clip(pos, 0, None)], False) if len(ce) else False
            else:
                cb[f"chem__{a}"] = np.nan
                cb[f"chem__{a}__cens"] = False
        blocks.append(cb)
        # ---- lab targets on event days (same-day event)
        lt = pd.DataFrame(index=cal)
        ced = ce.set_index("date") if len(ce) else pd.DataFrame()
        lt["is_chem_event"] = cal.isin(ced.index) if len(ce) else False
        for a in LAB_TARGETS:
            if len(ce) and a in ced:
                v = ced[a].reindex(cal)
                cens = (ced[f"{a}__cens"].astype(float).reindex(cal) > 0.5) if f"{a}__cens" in ced else False
                dl = ced[f"{a}__dl"].reindex(cal) if f"{a}__dl" in ced else np.nan
                lt[f"ylab__{a}"] = v
                lt[f"ylab__{a}__cens"] = cens
                lt[f"ylab__{a}__dl"] = dl
            else:
                lt[f"ylab__{a}"] = np.nan
                lt[f"ylab__{a}__cens"] = False
                lt[f"ylab__{a}__dl"] = np.nan
        # ---- 16S (on or before d)
        me = mev[mev.site_id == site].sort_values("date").reset_index(drop=True)
        pos, age = latest_event(cal, me["date"], MICRO_MAX_AGE, strictly_before=False)
        mb = pd.DataFrame(index=cal)
        mb["micro__present"] = pos >= 0
        mb["micro__age"] = age
        mb["micro__sample_id"] = np.where(pos >= 0, me["sample_id"].to_numpy(object)[np.clip(pos, 0, None)]
                                          if len(me) else None, None)
        mb["micro__log_reads"] = np.where(pos >= 0, np.log10(me["n_reads"].to_numpy(float)[np.clip(pos, 0, None)])
                                          if len(me) else np.nan, np.nan)
        mb["micro__log_nseq"] = np.where(pos >= 0, np.log10(me["n_seqs"].to_numpy(float)[np.clip(pos, 0, None)])
                                         if len(me) else np.nan, np.nan)
        lt["is_micro_event"] = cal.isin(pd.DatetimeIndex(me["date"])) if len(me) else False
        blocks.append(mb)
        # ---- biology (P2): latest event on or before d, age <= BIO_MAX_AGE
        for mod, evt in (("macroinv", bio["macroinv"]), ("periphyton", bio["periphyton"])):
            blocks.append(bio_block(mod, evt[evt.site == site] if len(evt) else evt, cal))
        # ---- hub: weather
        wx = wxds.to_table(filter=ds.field("site_key") == sk, columns=["date"] + WX_VARS).to_pandas()
        if len(wx):
            wx["date"] = pd.to_datetime(wx["date"])
            wb = weather_features(wx, cal)
            wb["wx__present"] = wb["wx__precip_sum7"].notna()
        else:
            wb = pd.DataFrame({"wx__present": False}, index=cal)
        blocks.append(wb)
        # ---- hub: static
        st = pd.DataFrame(index=cal)
        st["static__lat"] = float(reg.loc[sk, "lat"])
        st["static__lon"] = float(reg.loc[sk, "lon"])
        st["static__elev_m"] = float(reg.loc[sk, "elev_m"])
        for t in ("stream", "lake", "river"):
            st[f"static__type_{t}"] = float({"Aquatic Wadeable Stream": "stream", "Aquatic Lake": "lake",
                                             "Aquatic Non-wadeable Stream": "river"}.get(stype) == t)
        found = sk in sc.index and bool(sc.loc[sk, "streamcat_found"])
        for col in STATIC_COLS:
            v = float(sc.loc[sk, col]) if found and col in sc.columns and pd.notna(sc.loc[sk, col]) else np.nan
            st[f"static__{col}"] = np.log1p(v) if col in ("wsareasqkm", "popden2010ws", "fertws", "manurews") and np.isfinite(v) else v
        st["static__present"] = True
        st["static__streamcat_found"] = found
        blocks.append(st)
        # ---- time
        doy = cal.dayofyear.to_numpy(float)
        tb = pd.DataFrame({"time__doy_sin": np.sin(2 * np.pi * (doy - 1) / 365.25),
                           "time__doy_cos": np.cos(2 * np.pi * (doy - 1) / 365.25)}, index=cal)
        blocks.append(tb)
        # ---- sensor targets
        yt = pd.DataFrame(index=cal)
        for v in TARGET_VARS:
            if v not in series:
                for h in HORIZONS:
                    yt[f"y__{v}__h{h}"] = np.nan
                continue
            m = series[v][f"{v}__mean"].reindex(cal)
            for h in HORIZONS:
                yt[f"y__{v}__h{h}"] = m.shift(-h)
        if "do_mgl" in series:
            mn = series["do_mgl"]["do_mgl__min"].reindex(cal)
            for h in HORIZONS:
                yt[f"y__do_mgl_min__h{h}"] = mn.shift(-h)
        # ---- hazard labels (original units)
        for hzk, col in TIER_D_HAZARDS.items():
            var, stat = col.split("__")
            if var in series and f"{var}__{stat}" in series[var]:
                zz = series[var][f"{var}__{stat}"].reindex(cal).to_numpy(float)
                x = zz if var not in LOG_OFFSET else np.exp(zz) - LOG_OFFSET[var]
            else:
                x = np.full(len(cal), np.nan)
            for h in (7,):
                y, ok = _onset_labels(x, hzk, h)
                yt[f"ev__{hzk}__onset_h{h}"] = np.where(ok, y, np.nan)
                y2, ok2 = _any_exceed(x, hzk, h)
                yt[f"ev__{hzk}__any_h{h}"] = np.where(ok2, y2, np.nan)
        df = pd.concat(blocks + [lt, yt], axis=1)
        df.index.name = "date"
        df = df.reset_index()
        df.insert(0, "site", site)
        df.insert(1, "site_key", sk)
        df.insert(2, "domain", str(sites.loc[sites.site == site, "domain"].iloc[0]))
        df.insert(3, "aquatic_type", stype)
        frames.append(df)
        log(f"{site}: rows={len(df)} sonde={int(df['sonde__present'].sum())} chem={int(df['chem__present'].sum())} "
            f"micro={int(df['micro__present'].sum())} wx={int(df['wx__present'].sum())}")
    daily = pd.concat(frames, ignore_index=True)
    # keep rows where at least one non-hub modality is present or a target exists
    mods = modality_columns(daily)
    any_obs = np.zeros(len(daily), bool)
    for m in NON_HUB_D:
        any_obs |= daily[f"{m}__present"].fillna(False).to_numpy(bool)
    ycols = [c for c in daily.columns if c.startswith(("y__", "ylab__")) and not c.endswith(("__cens", "__dl"))]
    any_y = daily[ycols].notna().any(axis=1).to_numpy()
    daily = daily[any_obs | any_y].reset_index(drop=True)
    return {"daily": daily, "chem_events": chem, "micro_events": mev, "micro_genus": mgen, "modalities": mods}


NON_HUB_D = ["sonde", "nitrate", "temp", "hydro", "chem", "micro", "macroinv", "periphyton"]
HUB_D = ["wx", "static", "time"]


def modality_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Feature columns per modality (``<mod>__*`` minus presence/pointer columns)."""
    out = {}
    for m in NON_HUB_D + HUB_D:
        cols = [c for c in df.columns if c.startswith(f"{m}__") and not c.endswith(("__present", "__sample_id"))
                and c != "static__streamcat_found"]
        out[m] = cols
    return out


def availability_patterns(df: pd.DataFrame, mods: list[str], extra_mask=None) -> pd.DataFrame:
    """Counts of rows / sites per modality-availability pattern (only non-hub modalities)."""
    pres = pd.DataFrame({m: df[f"{m}__present"].fillna(False).astype(bool) for m in mods})
    if extra_mask is not None:
        pres = pres[extra_mask]
        site = df.loc[extra_mask, "site"]
    else:
        site = df["site"]
    key = pres.apply(lambda r: "+".join([m for m in mods if r[m]]) or "(hub only)", axis=1)
    out = pd.DataFrame({"pattern": key, "site": site}).groupby("pattern").agg(
        rows=("site", "size"), sites=("site", "nunique")).sort_values("rows", ascending=False)
    out["n_modalities"] = [0 if p == "(hub only)" else p.count("+") + 1 for p in out.index]
    return out.reset_index()
