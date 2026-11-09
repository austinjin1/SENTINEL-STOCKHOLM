# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""UK Environment Agency invertebrate design for BioSet pretraining (protocol §2, §7.4).

* Tokens: EA invertebrate taxon records mapped through ``processed/taxonomy_harmonised``
  (dataset ``ea_invertebrate``, ``group == "macroinvertebrate"`` only — EA nets also list
  macrophytes, algae and vertebrates). Abundance = the cross-method ``log_abundance_class`` (1-5;
  from ``abundance`` when the class is missing; presence-only -> 1), with representative count
  3·10^(class−1); EA abundances are not comparable across analysis methods otherwise.
* Chemistry (pretraining targets only): ±30-day medians at the paired WQ point
  (``chemistry_matched_wide_pm30d_invertebrate``), log10 with documented floors. Units are part of
  the column name and never mixed.
* **Reserved UK test**: sites in 20% of 50-km blocks (``spatial_block_split``, seed 0) never enter
  pretraining, vocabulary building or selection; 5% of blocks are the pretraining validation set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sentinel.core import paths
from sentinel.eval import splits as ES

UK_TARGETS = {  # name -> (wide column, factor, floor)  (log10 of factor*value, floored)
    "uk_ph": ("ph__phunits__pm30d__median", None, None),
    "uk_log10_cond25_us_cm": ("cond25__microsiemenspercentimetre__pm30d__median", 1.0, 1.0),
    "uk_log10_chloride_mg_l": ("chloride__milligramperlitre__pm30d__median", 1.0, 0.01),
    "uk_log10_tp_ug_l": ("total_p__milligramperlitre__pm30d__median", 1000.0, 1.0),
    "uk_log10_orthop_mg_l": ("orthophosphate_p__milligramperlitre__pm30d__median", 1.0, 0.001),
    "uk_log10_nh4n_mg_l": ("ammoniacal_n__milligramperlitre__pm30d__median", 1.0, 0.001),
    "uk_log10_ton_mg_l": ("ton_n__milligramperlitre__pm30d__median", 1.0, 0.01),
    "uk_log10_nitrate_mg_l": ("nitrate_n__milligramperlitre__pm30d__median", 1.0, 0.01),
    "uk_log10_ss_mg_l": ("ss_105c__milligramperlitre__pm30d__median", 1.0, 0.1),
    "uk_log10_turb_ntu": ("turbidity_6396__nephelometricturbidityunits__pm30d__median", 1.0, 0.01),
    "uk_log10_bod_mg_l": ("bod5_atu__milligramperlitre__pm30d__median", 1.0, 0.1),
    "uk_log10_alk_mg_l": ("alkalinity_ph45__milligramperlitre__pm30d__median", 1.0, 0.1),
}
RESERVED = {"block_km": 50.0, "test_frac": 0.20, "val_frac": 0.05, "seed": 0}


@dataclass
class UKDesign:
    samples: pd.DataFrame   # index analysis_id (str): site_id, role (pool_fit | pool_val | reserved_test), lat, lon
    tokens: pd.DataFrame    # analysis_id, LEVELS, count, log_class
    Y: pd.DataFrame         # index analysis_id, UK_TARGETS
    info: dict


def uk_site_roles(sites: pd.DataFrame) -> pd.Series:
    s = sites.dropna(subset=["lat", "lon"]).sort_values("site_id")
    sp = ES.spatial_block_split(list(s["site_id"]), s["lat"].to_numpy(), s["lon"].to_numpy(),
                                block_km=RESERVED["block_km"], test_frac=RESERVED["test_frac"],
                                val_frac=RESERVED["val_frac"], seed=RESERVED["seed"])
    role = {x: "pool_fit" for x in sp.train}
    role.update({x: "pool_val" for x in sp.val})
    role.update({x: "reserved_test" for x in sp.test})
    return pd.Series(role, name="role")


def load_uk(min_taxa: int = 3, include_reserved: bool = False) -> UKDesign:
    P = paths.processed("uk_ea_ecology")
    T = paths.processed("taxonomy_harmonised")
    sites = pd.read_parquet(P / "sites.parquet")
    sites = sites[(sites["assemblage"] == "invertebrate") & (sites["waterbody_type"] != "WBES")]
    sites = sites.assign(site_id=sites["site_id"].astype(str))
    roles = uk_site_roles(sites)
    smp = pd.read_parquet(P / "samples_invertebrate.parquet",
                          columns=["site_id", "sample_id", "analysis_id", "sample_date", "analysis_method"])
    smp = smp.assign(site_id=smp["site_id"].astype(str), analysis_id=smp["analysis_id"].astype(str),
                     sample_id=smp["sample_id"].astype(str))
    smp["role"] = smp["site_id"].map(roles)
    smp = smp[smp["role"].notna()]
    if not include_reserved:
        smp = smp[smp["role"] != "reserved_test"]
    cnt = pd.read_parquet(P / "counts_invertebrate.parquet",
                          columns=["analysis_id", "taxon_list_item_key", "abundance", "log_abundance_class"])
    cnt = cnt.assign(analysis_id=cnt["analysis_id"].astype(str))
    cnt = cnt[cnt["analysis_id"].isin(set(smp["analysis_id"]))]
    tm = pd.read_parquet(T / "taxon_map.parquet")
    tm = tm[(tm["dataset"] == "ea_invertebrate") & (tm["group"] == "macroinvertebrate")]
    tm = tm[["source_taxon_id", "name_clean", "h_phylum", "h_class", "h_order", "h_family", "h_genus"]].rename(
        columns={"source_taxon_id": "taxon_list_item_key", "name_clean": "name", "h_phylum": "phylum",
                 "h_class": "class", "h_order": "order", "h_family": "family", "h_genus": "genus"})
    tm = tm.drop_duplicates("taxon_list_item_key")
    tok = cnt.merge(tm, on="taxon_list_item_key", how="inner")
    lc = pd.to_numeric(tok["log_abundance_class"], errors="coerce")
    ab = pd.to_numeric(tok["abundance"], errors="coerce")
    from_ab = np.clip(np.floor(np.log10(ab.where(ab > 0))) + 1, 1, 5)
    lc = lc.fillna(from_ab).fillna(1.0).clip(1, 5)
    tok["log_class"] = lc.to_numpy(float)
    tok["count"] = 3.0 * 10.0 ** (tok["log_class"] - 1.0)
    keys = ["analysis_id", "name", "phylum", "class", "order", "family", "genus"]
    tok[keys[1:]] = tok[keys[1:]].astype(object).where(tok[keys[1:]].notna(), None)
    tok = tok.groupby(keys, dropna=False, as_index=False).agg(count=("count", "sum"), log_class=("log_class", "max"))
    ntax = tok.groupby("analysis_id").size()
    keep_ids = set(ntax.index[ntax >= min_taxa])
    smp = smp[smp["analysis_id"].isin(keep_ids)]
    tok = tok[tok["analysis_id"].isin(set(smp["analysis_id"]))].reset_index(drop=True)
    coords = sites.set_index("site_id")[["lat", "lon"]]
    smp = smp.join(coords, on="site_id")
    smp["log10_total_count"] = np.log10(tok.groupby("analysis_id")["count"].sum().reindex(smp["analysis_id"]).to_numpy())
    smp["log10_n_taxa"] = np.log10(tok.groupby("analysis_id").size().reindex(smp["analysis_id"]).to_numpy())
    smp = smp.set_index("analysis_id").sort_index()
    chem = pd.read_parquet(P / "chemistry_matched_wide_pm30d_invertebrate.parquet")
    chem = chem.assign(sample_id=chem["sample_id"].astype(str)).set_index("sample_id")
    Y = pd.DataFrame(index=smp.index)
    for name, (col, fac, floor) in UK_TARGETS.items():
        x = chem[col].reindex(smp["sample_id"]).to_numpy(float) if col in chem else np.full(len(smp), np.nan)
        if fac is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                x = np.log10(np.clip(x * fac, floor, None))
        Y[name] = x
    # pH outside 3-11 is physically implausible for these rivers (recording errors) -> missing
    Y.loc[(Y["uk_ph"] < 3) | (Y["uk_ph"] > 11), "uk_ph"] = np.nan
    info = {"n_samples": int(len(smp)), "n_sites": int(smp["site_id"].nunique()),
            "roles": smp["role"].value_counts().to_dict(), "n_token_rows": int(len(tok)),
            "n_with_any_chem": int(Y.notna().any(axis=1).sum()),
            "n_targets_present": {k: int(Y[k].notna().sum()) for k in UK_TARGETS}, "reserved": RESERVED}
    return UKDesign(smp, tok, Y, info)
