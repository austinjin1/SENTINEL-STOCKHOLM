# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M5-X development design: one community → chemistry design over NRSA (development cycles only), NLA and
UK EA rows that are allowed in an M5-X training pool (protocol ``bioassess_x_protocol.md`` §2–§3, §7).

Protected rows — test / excluded rows of either M5-X block and frozen-test rows of any workstream
(``xtrack.frozen_test_keys``) — are dropped **before any chemistry is read**; chemistry files are read with
parquet row filters on the allowed ids, so protected labels are never materialised. NRSA 2018-19 / 2023-24 are
frozen-test cycles and are not loaded at all.

Unified sample columns (BioDesign-compatible for ``baselines.fit_predict`` and ``cv.bioset_rows``):
``source``, ``survey``, ``site_id``, ``location_id`` (= split cluster), ``ag_eco9`` (= region: NRSA/NLA
ecoregion, UK EA agency area), ``lat``, ``lon``, ``elevation_m`` (UK: site altitude), descriptors, sampling
day-of-year, ``year``, ``reference_class`` (NRSA/NLA only), UK context features (protocol §6, no chemistry),
``analysis_method`` (UK; ``NA`` otherwise). Ids are ``"<source>:<id>"``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.core import paths
from sentinel.models.bioassess import xtrack as X
from sentinel.models.bioassess.data import (DEV_SURVEYS, LEVELS, TARGET_NAMES, BioDesign, load_nrsa,
                                            target_frame)
from sentinel.models.bioassess.uk import UK_TARGETS

UK_CONTEXT = ("site_altitude_m", "site_slope_m_per_km", "site_dist_from_source_km", "site_width_m", "site_depth_cm",
              "site_discharge_category", "site_boulders_cobbles_pct", "site_pebbles_gravel_pct", "site_sand_pct",
              "site_silt_clay_pct")
UK_CHEMISTRY_COLUMNS = ("site_alkalinity_mg_per_l_caco3", "site_conductivity_us_per_cm",
                        "site_total_hardness_mg_per_l_caco3", "site_calcium_mg_per_l")
UK_INDICES = ("idx_whpt_aspt", "idx_whpt_n_taxa", "idx_psi_family_score", "idx_epsi_family_score",
              "idx_life_family_index", "idx_bmwp_aspt")
X_TARGETS = {"X-L": TARGET_NAMES, "X-U": tuple(X.UK_PRIMARY)}
TOKEN_COLS = ["visit_id", *LEVELS, "mixed_otu", "count"]


def allowed_keys(S: pd.DataFrame, frozen: dict) -> set[tuple[str, str]]:
    """(source, id) of rows that may appear in some M5-X training pool."""
    pool = S[S["role"] == "train_pool"]
    prot = S[S["role"].isin(["test", "excluded"])]
    bad = set(zip(prot["source"], prot["id"].astype(str))) | set(frozen)
    return {k for k in zip(pool["source"], pool["id"].astype(str)) if k not in bad}


def _doy(dates) -> tuple[np.ndarray, np.ndarray]:
    d = pd.to_datetime(dates).dt.dayofyear.to_numpy(float)
    return np.sin(2 * np.pi * d / 365.25), np.cos(2 * np.pi * d / 365.25)


def _nrsa(allowed) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = load_nrsa(DEV_SURVEYS)                                     # development cycles only
    keep = [i for i in d.samples.index if ("nrsa", str(i)) in allowed]
    s = d.samples.loc[keep].copy()
    s["source"] = "nrsa"
    s["year"] = s["survey"].map({"nrsa0809": 2008, "nrsa1314": 2013})
    tok = d.tokens[d.tokens["visit_id"].isin(set(keep))].copy()
    return s, tok, d.Y.loc[keep].copy()


def _nla(allowed) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    P, T = paths.processed("nars_lakes"), paths.processed("taxonomy_harmonised")
    v = pd.read_parquet(P / "visits.parquet")
    v = v[[("nla", str(i)) in allowed for i in v["visit_id"]]].set_index("visit_id")
    ids = list(v.index)
    cnt = pd.read_parquet(P / "benthic_counts.parquet", filters=[("visit_id", "in", ids)],
                          columns=["visit_id", "survey", "taxa_id", "taxon_name", "count_300", "sample_type"])
    cnt = cnt[cnt["count_300"] > 0]
    tm = pd.read_parquet(T / "taxon_map.parquet")
    tm = tm[(tm["source"] == "nars_lakes") & (tm["assemblage"] == "benthic")]
    tm = tm[["dataset", "source_taxon_id", "source_taxon_name", "name_clean", "h_phylum", "h_class", "h_order",
             "h_family", "h_genus", "mixed_otu"]].rename(columns={
                 "dataset": "survey", "source_taxon_id": "taxa_id", "source_taxon_name": "taxon_name",
                 "name_clean": "name", "h_phylum": "phylum", "h_class": "class", "h_order": "order",
                 "h_family": "family", "h_genus": "genus"}).drop_duplicates(["survey", "taxa_id", "taxon_name"])
    tok = cnt.merge(tm, on=["survey", "taxa_id", "taxon_name"], how="inner")
    keys = ["visit_id", *LEVELS, "mixed_otu"]
    tok[keys[1:]] = tok[keys[1:]].astype(object).where(tok[keys[1:]].notna(), None)
    tok = tok.groupby(keys, dropna=False, as_index=False).agg(count=("count_300", "sum"))
    agg = tok.groupby("visit_id").agg(total=("count", "sum"), n_taxa=("name", "size"))
    v = v[v.index.isin(agg.index)]
    s = pd.DataFrame({"source": "nla", "survey": v["survey"], "site_id": v["site_id"].astype(str),
                      "ag_eco9": v["ag_eco9"], "lat": v["lat"].astype(float), "lon": v["lon"].astype(float),
                      "elevation_m": pd.to_numeric(v["elevation_m"], errors="coerce"),
                      "reference_class": v["reference_class"], "year": pd.to_numeric(v["year"], errors="coerce"),
                      "log10_total_count": np.log10(agg["total"].reindex(v.index).to_numpy(float)),
                      "log10_n_taxa": np.log10(agg["n_taxa"].reindex(v.index).to_numpy(float)),
                      "boatable": 0.0}, index=v.index)
    s["doy_sin"], s["doy_cos"] = _doy(v["date"])
    ids = list(s.index)
    cw = pd.read_parquet(P / "chemistry_wide.parquet", filters=[("visit_id", "in", ids)])
    cl = pd.read_parquet(P / "chemistry_long.parquet", filters=[("visit_id", "in", ids)],
                         columns=["visit_id", "analyte", "censored", "detection_limit_canonical"])
    Y, _ = target_frame(cw, cl, s.index)
    return s, tok[tok["visit_id"].isin(set(ids))], Y


def _uk(allowed) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    P, T = paths.processed("uk_ea_ecology"), paths.processed("taxonomy_harmonised")
    want = sorted(i for s, i in allowed if s == "uk")
    smp = pd.read_parquet(P / "samples_invertebrate.parquet",
                          columns=["site_id", "sample_id", "analysis_id", "sample_date", "analysis_method",
                                   *UK_INDICES])
    smp = smp.assign(analysis_id=smp["analysis_id"].astype(str), site_id=smp["site_id"].astype(str),
                     sample_id=smp["sample_id"].astype(str))
    smp = smp[smp["analysis_id"].isin(set(want))].drop_duplicates("analysis_id").set_index("analysis_id")
    sites = pd.read_parquet(P / "sites.parquet")
    sites = sites[sites["assemblage"] == "invertebrate"].assign(site_id=lambda d: d["site_id"].astype(str))
    sites = sites.drop_duplicates("site_id").set_index("site_id")
    if set(UK_CHEMISTRY_COLUMNS) & set(UK_CONTEXT):
        raise AssertionError("chemistry column listed as context")
    cnt = pd.read_parquet(P / "counts_invertebrate.parquet",
                          columns=["analysis_id", "taxon_list_item_key", "abundance", "log_abundance_class"])
    cnt = cnt.assign(analysis_id=cnt["analysis_id"].astype(str))
    cnt = cnt[cnt["analysis_id"].isin(set(smp.index))]
    tm = pd.read_parquet(T / "taxon_map.parquet")
    tm = tm[(tm["dataset"] == "ea_invertebrate") & (tm["group"] == "macroinvertebrate")]
    tm = tm[["source_taxon_id", "name_clean", "h_phylum", "h_class", "h_order", "h_family", "h_genus"]].rename(
        columns={"source_taxon_id": "taxon_list_item_key", "name_clean": "name", "h_phylum": "phylum",
                 "h_class": "class", "h_order": "order", "h_family": "family", "h_genus": "genus"})
    tok = cnt.merge(tm.drop_duplicates("taxon_list_item_key"), on="taxon_list_item_key", how="inner")
    lc = pd.to_numeric(tok["log_abundance_class"], errors="coerce")
    ab = pd.to_numeric(tok["abundance"], errors="coerce")
    lc = lc.fillna(np.clip(np.floor(np.log10(ab.where(ab > 0))) + 1, 1, 5)).fillna(1.0).clip(1, 5)
    tok["count"] = 3.0 * 10.0 ** (lc - 1.0)
    tok = tok.rename(columns={"analysis_id": "visit_id"})
    tok["mixed_otu"] = tok["name"]
    keys = ["visit_id", *LEVELS, "mixed_otu"]
    tok[keys[1:]] = tok[keys[1:]].astype(object).where(tok[keys[1:]].notna(), None)
    tok = tok.groupby(keys, dropna=False, as_index=False).agg(count=("count", "sum"))
    agg = tok.groupby("visit_id").agg(total=("count", "sum"), n_taxa=("name", "size"))
    smp = smp[smp.index.isin(agg.index[agg["n_taxa"] >= 3])]
    st = sites.reindex(smp["site_id"])
    s = pd.DataFrame({"source": "uk", "survey": "uk_ea", "site_id": smp["site_id"].to_numpy(),
                      "lat": st["lat"].to_numpy(float), "lon": st["lon"].to_numpy(float),
                      "elevation_m": pd.to_numeric(st["site_altitude_m"], errors="coerce").to_numpy(),
                      "ag_eco9": st["agency_area"].to_numpy(), "reference_class": None,
                      "year": pd.to_datetime(smp["sample_date"]).dt.year.to_numpy(),
                      "log10_total_count": np.log10(agg["total"].reindex(smp.index).to_numpy(float)),
                      "log10_n_taxa": np.log10(agg["n_taxa"].reindex(smp.index).to_numpy(float)),
                      "boatable": 0.0, "analysis_method": smp["analysis_method"].to_numpy()}, index=smp.index)
    s["doy_sin"], s["doy_cos"] = _doy(smp["sample_date"])
    for c in UK_CONTEXT:
        s[c] = pd.to_numeric(st[c], errors="coerce").to_numpy() if c in st else np.nan
    for c in UK_INDICES:
        s[c] = pd.to_numeric(smp[c], errors="coerce").to_numpy()
    sid = list(smp.loc[s.index, "sample_id"].unique())
    chem = pd.read_parquet(P / "chemistry_matched_wide_pm30d_invertebrate.parquet",
                           filters=[("sample_id", "in", sid)])
    chem = chem.assign(sample_id=chem["sample_id"].astype(str)).drop_duplicates("sample_id").set_index("sample_id")
    Y = pd.DataFrame(index=s.index)
    for name, (col, fac, floor) in UK_TARGETS.items():
        x = chem[col].reindex(smp.loc[s.index, "sample_id"]).to_numpy(float) if col in chem else np.full(len(s), np.nan)
        if fac is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                x = np.log10(np.clip(x * fac, floor, None))
        Y[name] = x
    Y.loc[(Y["uk_ph"] < 3) | (Y["uk_ph"] > 11), "uk_ph"] = np.nan
    return s, tok[tok["visit_id"].isin(set(s.index))], Y


def load_x_design(S: pd.DataFrame, frozen: dict) -> BioDesign:
    """All allowed rows of all sources in one BioDesign (ids "<source>:<id>"); location_id = split cluster."""
    allowed = allowed_keys(S, frozen)
    parts = [_nrsa(allowed), _nla(allowed), _uk(allowed)]
    cl = S.drop_duplicates(["source", "id"]).set_index(["source", "id"])["cluster"]
    smp, toks, ys = [], [], []
    for (s, tok, y), src in zip(parts, ("nrsa", "nla", "uk")):
        key = src + ":" + s.index.astype(str)
        s = s.copy()
        s["location_id"] = cl.reindex(list(zip([src] * len(s), s.index.astype(str)))).to_numpy()
        tok = tok.assign(visit_id=src + ":" + tok["visit_id"].astype(str))
        s.index, y.index = key, key
        smp.append(s)
        toks.append(tok[[c for c in TOKEN_COLS if c in tok.columns]])
        ys.append(y)
    samples = pd.concat(smp, sort=False)
    samples.index.name = "visit_id"
    Y = pd.concat(ys, sort=False).reindex(samples.index)
    for c in TARGET_NAMES + tuple(UK_TARGETS):
        if c not in Y:
            Y[c] = np.nan
    keys = [tuple(k.split(":", 1)) for k in samples.index]
    X.assert_fit_ids(keys, S, frozen)                      # loader guard (protocol §7)
    if samples["location_id"].isna().any():
        raise AssertionError("rows without a split cluster")
    tokens = pd.concat(toks, ignore_index=True)
    info = {"n_samples": int(len(samples)), "by_source": samples["source"].value_counts().to_dict()}
    return BioDesign(samples, tokens, Y, info)
