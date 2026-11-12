# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D7 TroutRefuge — coldwater fish occupancy (protocol docs/preregistration/downstream_troutrefuge.md).

Labels, split and power check. Validation only: test-cycle rows are never built here unless the D7 guard
(``troutrefuge_guard.require_open_claim``) passes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core import paths
from . import common as C

APP = "troutrefuge"
SURVEY_ROLE = {"nrsa0809": "train", "nrsa1314": "train", "nrsa1819": "val", "nrsa2324": "test"}
DATA_MANIFESTS = ("nars_streams", "downstream_catchment", "forcings_daily")
MODELLED_TEMP_PREFIXES = ("mast", "msst", "mwst")   # d7_mast: comparator score ONLY (lead decision 2026-11-09)


def assert_no_modelled_temperature(cols) -> None:
    """MAST/MSST/MWST (and the incumbent score derived from them) may never enter a D7 model's feature matrix."""
    import re

    pat = re.compile(r"(^|_)(" + "|".join(MODELLED_TEMP_PREFIXES) + r")", re.IGNORECASE)
    bad = [c for c in cols if pat.search(str(c))]
    if bad:
        raise AssertionError(f"modelled stream temperature in a D7 feature matrix: {bad}")


SPLIT_SHA = "587caeca6c634e944d8268b001becbcca8ba650d3244f68f4c2bdad3e6f84a3c"   # frozen 2026-11-09 (protocol §8)
P = paths.PROCESSED / "nars_streams"
D5_SPLIT = C.PREREG / "downstream_biointegrity_split.json"


def temp_classes() -> pd.Series:
    t = pd.read_parquet(P / "taxa_source.parquet", columns=["survey", "assemblage", "taxon_name", "src_temp_nrsa"])
    t = t[(t["assemblage"] == "fish") & t["src_temp_nrsa"].notna()]
    t["name"] = t["taxon_name"].str.upper().str.strip()
    order = {"nrsa1314": 0, "nrsa2324": 1}
    t = t[t["survey"].isin(order)].sort_values(by="survey", key=lambda s: s.map(order))
    return t.drop_duplicates("name").set_index("name")["src_temp_nrsa"]


def load_units() -> pd.DataFrame:
    """First visits with a fish sample; label coldwater (>= 1 CD taxon) and salmonid presence."""
    v = pd.read_parquet(P / "visits.parquet", columns=["visit_id", "survey", "site_id", "visit_no", "date", "lat", "lon",
                                                         "ag_eco9", "huc8", "comid", "unique_id"])
    v = v[v["visit_no"].astype(str) == "1"]
    f = pd.read_parquet(P / "fish_counts.parquet", columns=["visit_id", "taxon_name", "count"])
    f = f[f["visit_id"].isin(set(v["visit_id"]))]
    cls = temp_classes()
    f["name"] = f["taxon_name"].str.upper().str.strip()
    f["cls"] = f["name"].map(cls)
    f["cold"] = f["cls"].eq("CD")
    tx = pd.read_parquet(P / "taxa_source.parquet", columns=["assemblage", "taxon_name", "src_family"])
    fam = tx[tx["assemblage"] == "fish"].assign(name=lambda d: d["taxon_name"].str.upper().str.strip()) \
        .dropna(subset=["src_family"]).drop_duplicates("name").set_index("name")["src_family"]
    f["salmonid"] = f["name"].map(fam).eq("SALMONIDAE")
    g = f.groupby("visit_id")
    lab = pd.DataFrame({"y_cold": g["cold"].any().astype(float), "y_salmonid": g["salmonid"].any().astype(float),
                        "n_fish": g["count"].sum(), "share_classified": g["cls"].apply(lambda s: s.notna().mean())})
    u = v.merge(lab, left_on="visit_id", right_index=True, how="inner")
    u["role"] = u["survey"].map(SURVEY_ROLE)
    u["date"] = pd.to_datetime(u["date"])
    u["huc8"] = u["huc8"].astype("string")
    u["huc4"] = u["huc8"].str[:4]
    import json

    d5 = json.loads(D5_SPLIT.read_text())["assignments"]["visit_location"]
    u["location_id"] = u["visit_id"].map(d5)
    if u["location_id"].isna().any():   # visits outside D5's benthic set: own location id (no 1-km merge available)
        u.loc[u["location_id"].isna(), "location_id"] = "d7:" + u.loc[u["location_id"].isna(), "site_id"].astype(str)
    return u.reset_index(drop=True)


def freeze() -> str:
    from ...core.registry import file_sha256

    u = load_units()
    meta = {"unit": "NRSA first visit with a fish sample", "survey_role": SURVEY_ROLE,
            "label": "y_cold = >= 1 NRSA CD (coldwater) taxon; secondary y_salmonid",
            "trait_rule": "taxon name -> src_temp_nrsa from 2013-14 then 2023-24 taxa tables",
            "location_clusters": "D5 frozen visit_location (1-km), else own site id",
            "new_location_rule": "location with no first visit in any fitting cycle",
            "fish_counts_sha256": file_sha256(P / "fish_counts.parquet"), "visits_sha256": file_sha256(P / "visits.parquet"),
            "taxa_source_sha256": file_sha256(P / "taxa_source.parquet"), "freeze_date": "2026-11-09"}
    return C.freeze_split(APP, {"visit_role": dict(zip(u["visit_id"], u["role"])),
                                "visit_location": dict(zip(u["visit_id"], u["location_id"].astype(str)))}, meta)


def power_check(u: pd.DataFrame, label: str = "y_cold", auc: float = 0.8, rho: float = 0.6) -> dict:
    fit_locs = set(u.loc[u["role"] == "train", "location_id"])
    out = {}
    for name, m in (("train", u["role"] == "train"), ("val_all", u["role"] == "val"),
                    ("val_newloc", (u["role"] == "val") & ~u["location_id"].isin(fit_locs))):
        g = u[m]
        npos, nneg = int(g[label].sum()), int((1 - g[label]).sum())
        q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
        se = float(np.sqrt((auc * (1 - auc) + (npos - 1) * (q1 - auc ** 2) + (nneg - 1) * (q2 - auc ** 2)) / max(npos * nneg, 1)))
        out[name] = {"n": int(len(g)), "n_pos": npos, "prevalence": float(g[label].mean()) if len(g) else None,
                     "huc4_with_pos": int(g.loc[g[label] == 1, "huc4"].nunique()),
                     "auroc_se": se, "paired_mde": float(2.8 * se * np.sqrt(2 * (1 - rho))),
                     "share_classified_median": float(g["share_classified"].median()) if len(g) else None}
    return out


# --------------------------------------------------------------------------- features (observed only; MAST never a feature)
MAST_YEAR = {"nrsa0809": 2008, "nrsa1314": 2013, "nrsa1819": 2014, "nrsa2324": 2014}


def add_unit_fields(u: pd.DataFrame) -> pd.DataFrame:
    v = pd.read_parquet(P / "visits.parquet", columns=["visit_id", "elevation_m", "ag_eco9"])
    u = u.drop(columns=[c for c in ("elevation_m",) if c in u]).merge(v[["visit_id", "elevation_m"]], on="visit_id", how="left")
    u["unit_id"] = u["survey"].astype(str) + ":" + u["site_id"].astype(str)
    chem = pd.read_parquet(P / "chemistry_wide.parquet", columns=["visit_id", "temp_field_degc"])
    return u.merge(chem, on="visit_id", how="left")


def summer_climate(u: pd.DataFrame) -> pd.DataFrame:
    """gridMET mean Jun–Aug Tmax and Jun–Aug precipitation total, averaged over the 3 calendar years before the visit
    (ends before the visit); nearest SENTINEL cell <= 30 km; plus the 7-day Tmax ending the day before the visit."""
    from . import weather as W

    nc = W.nearest_cells(u["lat"], u["lon"])
    nc.index = u.index
    keys = nc.loc[nc["wx_cell_km"] <= 30, "wx_site_key"].dropna().unique()
    s = W.load_series(keys, start="1999-01-01")
    s["year"], s["month"] = s["date"].dt.year, s["date"].dt.month
    jja = s[s["month"].isin([6, 7, 8])].groupby(["site_key", "year"]).agg(tmax=("tmax_c", "mean"), pr=("precip_mm", "sum"))
    out = pd.DataFrame(index=u.index, columns=["wx_jja_tmax_3y", "wx_jja_precip_3y"], dtype=float)
    yr = u["date"].dt.year
    for i in u.index:
        k = nc.at[i, "wx_site_key"]
        if nc.at[i, "wx_cell_km"] > 30 or k is None:
            continue
        rows = [jja.loc[(k, y)] for y in range(int(yr[i]) - 3, int(yr[i])) if (k, y) in jja.index]
        if rows:
            out.at[i, "wx_jja_tmax_3y"] = float(np.mean([r["tmax"] for r in rows]))
            out.at[i, "wx_jja_precip_3y"] = float(np.mean([r["pr"] for r in rows]))
    q = pd.DataFrame({"wx_site_key": nc["wx_site_key"].where(nc["wx_cell_km"] <= 30), "date": u["date"]}, index=u.index)
    a = W.antecedent_features(q.dropna(subset=["wx_site_key"]), s.drop(columns=["year", "month"]), windows=(7,))
    out["wx_tmax_mean_7d"] = a.reindex(u.index)["wx_tmax_mean_7d"] if "wx_tmax_mean_7d" in a else np.nan
    return out


def modelled_temp(u: pd.DataFrame, metric: str = "mast") -> pd.Series:
    """Published thermal incumbent (comparator ONLY): Hill et al. 2013 modelled stream temperature of the visit's
    COMID, year-matched (d7_mast). ``mast`` = mean ANNUAL, ``msst`` = mean SUMMER (Jul-Aug) stream temperature
    (protocol §8 correction 2026-11-10). Never a feature (``assert_no_modelled_temperature``)."""
    if metric not in ("mast", "msst"):
        raise ValueError(metric)
    links = pd.read_parquet(paths.PROCESSED / "downstream_catchment" / "unit_links.parquet", columns=["unit_id", "comid"])
    m = pd.read_parquet(paths.PROCESSED / "d7_mast" / "mast_comid.parquet")
    uc = u[["unit_id", "survey"]].merge(links, on="unit_id", how="left")
    uc["comid"] = uc["comid"].astype("string")
    m["comid"] = m["comid"].astype("string")
    uc = uc.merge(m, on="comid", how="left")
    val = [uc.at[i, f"{metric}{MAST_YEAR[s]}"] if f"{metric}{MAST_YEAR[s]}" in uc else np.nan
           for i, s in zip(uc.index, uc["survey"])]
    return pd.Series(np.asarray(val, float), index=u.index, name=f"{metric}_c")


def mast_score(u: pd.DataFrame) -> pd.Series:
    return modelled_temp(u, "mast")


def build_frame(include_test: bool = False, split_sha: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Units + feature blocks. MAST is returned separately (comparator score), never inside a feature block."""
    from . import biointegrity as BI

    if include_test:
        from .troutrefuge_guard import require_open

        require_open(split_sha)
    u = load_units()
    u = u if include_test else C.drop_protected(u)
    if not include_test:
        C.assert_no_test(u)
    u = add_unit_fields(u.reset_index(drop=True))
    geo = BI.geo_features(u)
    sc, _ = BI.streamcat_features(u)
    wx = summer_climate(u)
    blocks = {"geo": geo, "streamcat": sc, "wx": wx[["wx_jja_tmax_3y", "wx_jja_precip_3y"]], "wx7": wx[["wx_tmax_mean_7d"]]}
    for b in blocks.values():
        assert_no_modelled_temperature(b.columns)
    from .crossfit import phase1_no_upstream

    phase1_no_upstream([c for b in blocks.values() for c in b.columns])
    u["mast_c"] = mast_score(u)
    u["msst_c"] = modelled_temp(u, "msst")
    return u, blocks
