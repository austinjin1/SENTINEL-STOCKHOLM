# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M5-F designs: multi-assemblage rows (invertebrates + benthic diatoms) with two label windows.

Protocol: ``docs/preregistration/bioassess_f_protocol.md`` §3 (rows, pairing, pools) and §6 (targets).

Only the UK allowed rows are built here (lead decision 2026-11-10). The French builder is added after the
protocol stamp and the lock list. Label values are read for allowed rows only: the chemistry table is filtered by
the allowed ``sample_id`` list inside the parquet reader, so no label of a disallowed row is ever loaded.

Target columns are ``<target>__pm30d`` (median of the analyses in [d-30, d+30]) and ``<target>__p365`` (arithmetic
mean of the analyses in [d-365, d], at least 6 analyses), then the transform. Both windows are the ones of
``sentinel/data/bio_uk_ea.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sentinel.core import paths
from sentinel.models.bioassess import ftrack as F
from sentinel.models.bioassess.bioset import LEVELS

MODALITIES = ("invertebrate", "diatom")
ATTACH_DAYS = 90
MIN_TAXA = 3
MIN_OBS_365 = 6
WINDOWS = {"pm30d": ("pm30d", "median", 1), "p365": ("prior365d", "mean", MIN_OBS_365)}  # suffix -> (window, stat, min n)

# name -> (UK determinand short name or None, transform, floor). Concentrations in mg/L (N or P where stated).
TARGETS = {
    "ph": ("ph", None, None),
    "log10_cond25_us_cm": ("cond25", "log10", 1.0),
    "log10_chloride_mg_l": ("chloride", "log10", 0.01),
    "log10_no3n_mg_l": ("nitrate_n", "log10", 0.01),
    "log10_nh4n_mg_l": ("ammoniacal_n", "log10", 0.001),
    "log10_po4p_mg_l": ("orthophosphate_p", "log10", 0.001),
    "log10_tp_mg_l": ("total_p", "log10", 0.001),
    "log10_ss_mg_l": ("ss_105c", "log10", 0.1),
    "do_sat_pct": ("do_pctsat", None, None),
    "log10_do_mg_l": ("do_mgl", "log10", 0.1),
    "log10_bod5_mg_l": ("bod5_atu", "log10", 0.1),
    "log10_no2n_mg_l": ("nitrite_n", "log10", 0.001),
    "log10_ton_mg_l": ("ton_n", "log10", 0.01),          # UK only
    "log10_nkj_mg_l": (None, "log10", 0.01),             # France only (Kjeldahl N)
    "temp_c": ("temp_water", None, None),
    "log10_alk_mg_l": ("alkalinity_ph45", "log10", 0.1),  # UK only
}
PRIMARY = {
    "F-A": tuple(f"{t}__pm30d" for t in ("ph", "log10_cond25_us_cm", "log10_chloride_mg_l", "log10_no3n_mg_l",
                                         "log10_nh4n_mg_l", "log10_po4p_mg_l")),
    "F-T": tuple(f"{t}__p365" for t in ("ph", "log10_po4p_mg_l", "log10_nh4n_mg_l", "log10_no3n_mg_l",
                                        "log10_bod5_mg_l", "do_sat_pct")),
}
PRIMARY["F-Z"] = PRIMARY["F-A"]
Y_COLUMNS = tuple(f"{t}__{w}" for w in WINDOWS for t in TARGETS)
# French units -> N or P by molar mass (§6.3)
FR_FACTORS = {"no3_to_n": 0.2259, "nh4_to_n": 0.7765, "po4_to_p": 0.3261, "no2_to_n": 0.3045}
PLAUSIBLE = {"ph": (3.0, 11.0), "temp_c": (-1.0, 40.0), "do_sat_pct": (0.0, 300.0)}


@dataclass
class MADesign:
    """samples: index = row key; columns source, station, date, lat, lon, altitude_m, doy_sin, doy_cos, region,
    location_id, has_inv, inv_key, diatom_key, diatom_offset_days, protocol_invertebrate, protocol_diatom.
    tokens[modality]: columns ``key`` (modality sample key), LEVELS, ``count`` [, ``log_class``].
    Y: index = row key, columns Y_COLUMNS."""

    samples: pd.DataFrame
    tokens: dict
    Y: pd.DataFrame
    info: dict = field(default_factory=dict)


def transform(x, how, floor) -> np.ndarray:
    x = np.asarray(x, float)
    if how == "log10":
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.log10(np.clip(x, floor, None))
    return x


def doy_features(dates) -> tuple[np.ndarray, np.ndarray]:
    doy = pd.to_datetime(pd.Series(dates)).dt.dayofyear.to_numpy(float)
    a = 2 * np.pi * doy / 365.25
    return np.sin(a), np.cos(a)


def attach_nearest(inv: pd.DataFrame, dia: pd.DataFrame, days: int = ATTACH_DAYS) -> pd.DataFrame:
    """For each row of ``inv`` (columns station, date) the ``dia`` row (columns station, date, key) nearest in time
    at the same station within ``days``. Returns columns diatom_key, diatom_offset_days (diatom date - row date)."""
    out = pd.DataFrame({"diatom_key": None, "diatom_offset_days": np.nan}, index=inv.index)
    if len(inv) == 0 or len(dia) == 0:
        return out
    a = inv[["station", "date"]].copy()
    a["date"] = pd.to_datetime(a["date"])
    a["_i"] = np.arange(len(a))
    b = dia[["station", "date", "key"]].copy()
    b["date"] = pd.to_datetime(b["date"])
    b = b.rename(columns={"date": "ddate"}).sort_values(["ddate", "key"], kind="stable")
    b["_d"] = b["ddate"]
    m = pd.merge_asof(a.sort_values("date", kind="stable"), b, left_on="date", right_on="_d", by="station",
                      direction="nearest", tolerance=pd.Timedelta(days=days))
    m = m.sort_values("_i")
    out["diatom_key"] = m["key"].astype(object).where(m["key"].notna(), None).to_numpy()
    out["diatom_offset_days"] = (m["ddate"] - m["date"]).dt.days.to_numpy(float)
    return out


def _uk_tokens(counts: pd.DataFrame, dataset: str, group: str, prefix: str, log_classes: bool) -> pd.DataFrame:
    T = paths.processed("taxonomy_harmonised")
    tm = pd.read_parquet(T / "taxon_map.parquet",
                         columns=["dataset", "group", "source_taxon_id", "name_clean", "h_phylum", "h_class", "h_order",
                                  "h_family", "h_genus", "mixed_otu"])
    tm = tm[(tm["dataset"] == dataset) & (tm["group"] == group)].drop(columns=["dataset", "group"]).rename(
        columns={"source_taxon_id": "taxon_list_item_key", "name_clean": "name", "h_phylum": "phylum",
                 "h_class": "class", "h_order": "order", "h_family": "family", "h_genus": "genus"})
    tm = tm.drop_duplicates("taxon_list_item_key")
    tok = counts.merge(tm, on="taxon_list_item_key", how="inner")
    ab = pd.to_numeric(tok["abundance"], errors="coerce")
    if log_classes:      # EA invertebrate abundances: cross-method log class, as in ``uk.load_uk``
        lc = pd.to_numeric(tok["log_abundance_class"], errors="coerce")
        from_ab = np.clip(np.floor(np.log10(ab.where(ab > 0))) + 1, 1, 5)
        lc = lc.fillna(from_ab).fillna(1.0).clip(1, 5)
        tok["log_class"] = lc.to_numpy(float)
        tok["count"] = 3.0 * 10.0 ** (tok["log_class"] - 1.0)
    else:                # diatom valve counts
        tok["count"] = ab.where(ab > 0).fillna(1.0).to_numpy(float)
        tok["log_class"] = np.clip(np.floor(np.log10(np.maximum(tok["count"], 1.0))) + 1.0, 1.0, 5.0)
    keys = ["analysis_id", "name", "phylum", "class", "order", "family", "genus", "mixed_otu"]
    tok[keys[1:]] = tok[keys[1:]].astype(object).where(tok[keys[1:]].notna(), None)
    tok = tok.groupby(keys, dropna=False, as_index=False).agg(count=("count", "sum"), log_class=("log_class", "max"))
    tok["key"] = prefix + tok["analysis_id"].astype(str)
    return tok.drop(columns=["analysis_id"])


def _uk_labels(sample_ids, assemblage: str) -> pd.DataFrame:
    """Wide label frame indexed by sample_id for the given (allowed) sample ids only."""
    import pyarrow.dataset as ds

    P = paths.processed("uk_ea_ecology")
    ids = sorted(set(map(str, sample_ids)))
    dset = ds.dataset(P / "chemistry_matched_long.parquet")
    flt = ds.field("sample_id").isin(ids) & (ds.field("assemblage") == assemblage)
    long = dset.to_table(columns=["sample_id", "short_name", "window", "n_obs", "median", "mean"],
                         filter=flt).to_pandas()
    out = pd.DataFrame(index=pd.Index(ids, name="sample_id"))
    for suf, (win, stat, min_n) in WINDOWS.items():
        w = long[(long["window"] == win) & (long["n_obs"] >= min_n)]
        wide = w.pivot_table(index="sample_id", columns="short_name", values=stat, aggfunc="first")
        for t, (short, how, floor) in TARGETS.items():
            x = wide[short].reindex(out.index).to_numpy(float) if (short is not None and short in wide) \
                else np.full(len(out), np.nan)
            if t in PLAUSIBLE:
                lo, hi = PLAUSIBLE[t]
                x = np.where((x < lo) | (x > hi), np.nan, x)
            out[f"{t}__{suf}"] = transform(x, how, floor)
    return out[list(Y_COLUMNS)]


def load_uk_allowed(allowed: dict | None = None, min_taxa: int = MIN_TAXA) -> MADesign:
    """UK allowed rows as a multi-assemblage design: invertebrate rows (``uk:<analysis_id>``, with the nearest
    diatom analysis at the same site within 90 days attached) and diatom-only rows (``ukd:<analysis_id>``)."""
    from sentinel.models.bioassess.data import location_clusters

    allowed = allowed or F.uk_allowed()
    P = paths.processed("uk_ea_ecology")
    sites = pd.read_parquet(P / "sites.parquet",
                            columns=["site_id", "assemblage", "agency_area", "site_altitude_m", "lat", "lon"])
    sites = sites.assign(site_id=sites["site_id"].astype(str)).dropna(subset=["lat", "lon"])
    site = {a: s.drop_duplicates("site_id").set_index("site_id") for a, s in sites.groupby("assemblage")}

    def samples(assemblage, ids, cols):
        s = pd.read_parquet(P / f"samples_{assemblage}.parquet", columns=["site_id", "sample_id", "analysis_id",
                                                                         "sample_date", *cols])
        s = s.assign(site_id=s["site_id"].astype(str), sample_id=s["sample_id"].astype(str),
                     analysis_id=s["analysis_id"].astype(str))
        s = s[s["analysis_id"].isin(ids) & s["site_id"].isin(site[assemblage].index)]
        if (pd.to_datetime(s["sample_date"]).dt.date > F.UK_ALLOWED_TO).any():
            raise F.FitGuardError("UK row dated after the allowed period")
        return s

    def counts(assemblage, ids, cols):
        c = pd.read_parquet(P / f"counts_{assemblage}.parquet", columns=["analysis_id", "taxon_list_item_key", *cols])
        c = c.assign(analysis_id=c["analysis_id"].astype(str))
        return c[c["analysis_id"].isin(ids)]

    si = samples("invertebrate", allowed["invertebrate"], ["analysis_method"])
    sd = samples("diatom", allowed["diatom"], ["analysis_type"])
    tok_i = _uk_tokens(counts("invertebrate", set(si["analysis_id"]), ["abundance", "log_abundance_class"]),
                       "ea_invertebrate", "macroinvertebrate", "uk:", True)
    tok_d = _uk_tokens(counts("diatom", set(sd["analysis_id"]), ["abundance"]), "ea_diatom", "algae", "ukd:", False)
    keep_i = tok_i.groupby("key").size()
    keep_d = tok_d.groupby("key").size()
    keep_i, keep_d = set(keep_i.index[keep_i >= min_taxa]), set(keep_d.index[keep_d >= min_taxa])
    tok_i, tok_d = tok_i[tok_i["key"].isin(keep_i)], tok_d[tok_d["key"].isin(keep_d)]

    def frame(s, assemblage, prefix, keep):
        f = pd.DataFrame({"key": prefix + s["analysis_id"], "source": "uk", "station": s["site_id"],
                          "sample_id": s["sample_id"], "date": pd.to_datetime(s["sample_date"])})
        f = f[f["key"].isin(keep)].drop_duplicates("key")
        st = site[assemblage].reindex(f["station"])
        f["lat"], f["lon"] = st["lat"].to_numpy(float), st["lon"].to_numpy(float)
        f["altitude_m"] = st["site_altitude_m"].to_numpy(float)
        f["region"] = st["agency_area"].astype(object).to_numpy()
        return f

    fi = frame(si, "invertebrate", "uk:", keep_i)
    fi["protocol_invertebrate"] = "uk:" + si.set_index("analysis_id")["analysis_method"].astype(object).fillna(
        "other").reindex(fi["key"].str[3:]).to_numpy()
    fd = frame(sd, "diatom", "ukd:", keep_d)
    att = attach_nearest(fi.reset_index(drop=True), fd.reset_index(drop=True))
    fi = fi.reset_index(drop=True).assign(has_inv=True, inv_key=lambda d: d["key"], diatom_key=att["diatom_key"],
                                          diatom_offset_days=att["diatom_offset_days"])
    fd = fd.reset_index(drop=True).assign(has_inv=False, inv_key=None, diatom_key=lambda d: d["key"],
                                          diatom_offset_days=0.0, protocol_invertebrate=None)
    S = pd.concat([fi, fd], ignore_index=True)
    S["protocol_diatom"] = np.where(S["diatom_key"].notna(), "uk:LABP", None)
    S["doy_sin"], S["doy_cos"] = doy_features(S["date"])
    S["location_id"] = ["ukL%d" % k for k in location_clusters(S["lat"].to_numpy(), S["lon"].to_numpy())]
    S["fold"], S["lock_role"], S["uk_allowed"] = "UK", None, True
    S["source"] = np.where(S["has_inv"], "uk", "ukd")
    Yi = _uk_labels(fi["sample_id"], "invertebrate").reindex(fi["sample_id"]).set_axis(fi["key"].to_numpy())
    Yd = _uk_labels(fd["sample_id"], "diatom").reindex(fd["sample_id"]).set_axis(fd["key"].to_numpy())
    S = S.set_index("key")
    Y = pd.concat([Yi, Yd]).reindex(S.index)
    info = {"n_rows": int(len(S)), "n_invertebrate_rows": int(S["has_inv"].sum()),
            "n_diatom_rows": int((~S["has_inv"]).sum()),
            "n_invertebrate_rows_with_diatom": int((S["has_inv"] & S["diatom_key"].notna()).sum()),
            "n_labelled": {c: int(Y[c].notna().sum()) for c in Y.columns}, "allowed_counts": allowed["counts"],
            "uk_allowed_to": F.UK_ALLOWED_TO.isoformat()}
    return MADesign(S, {"invertebrate": tok_i.reset_index(drop=True), "diatom": tok_d.reset_index(drop=True)}, Y, info)


def restrict(D: MADesign, ids) -> MADesign:
    """Sub-design of the given rows (tokens of their modality samples only)."""
    ids = pd.Index(ids)
    S = D.samples.loc[ids]
    tok = {"invertebrate": D.tokens["invertebrate"][D.tokens["invertebrate"]["key"].isin(set(S["inv_key"].dropna()))],
           "diatom": D.tokens["diatom"][D.tokens["diatom"]["key"].isin(set(S["diatom_key"].dropna()))]}
    return MADesign(S, tok, D.Y.loc[ids], dict(D.info))


__all__ = ["MADesign", "MODALITIES", "TARGETS", "PRIMARY", "Y_COLUMNS", "LEVELS", "load_uk_allowed", "attach_nearest",
           "restrict", "transform", "doy_features"]
