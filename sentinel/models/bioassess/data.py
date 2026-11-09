# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""NRSA community -> chemistry design for SENTINEL-BioAssess (M5).

Builds, from ``processed/nars_streams`` and ``processed/taxonomy_harmonised``:

* ``samples`` — one row per first visit (``visit_no == "1"``) with a benthic sample: ids, survey,
  ``location_id`` (1-km location cluster over *all* NRSA cycles), ``ag_eco9``, coordinates and the
  three sample descriptors that may accompany the community (log10 total count_300, log10 number
  of taxa, boatable flag). Chemistry is never a column of ``samples``.
* ``tokens`` — long table (visit_id × harmonised source taxon) of 300-count abundances with the
  harmonised lineage (phylum … genus, finest name) and the ``mixed_otu`` key.
* ``Y`` — the six targets (log10 TP/TN/cond/Cl/turbidity, lab pH), censored values set to the
  detection limit before the documented floor (docs/preregistration/bioassess_protocol.md §3).

**Holdout guard.** NRSA 2018-19 and 2023-24 are external tests. :func:`load_nrsa` refuses them
unless ``allow_holdout=True``; phase-1 code never passes that flag (unit-tested).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.core import paths

DEV_SURVEYS = ("nrsa0809", "nrsa1314")
HOLDOUT_SURVEYS = ("nrsa1819", "nrsa2324")
ALL_SURVEYS = DEV_SURVEYS + HOLDOUT_SURVEYS
LOCATION_RADIUS_KM = 1.0
LEVELS = ("phylum", "class", "order", "family", "genus", "name")
BOATABLE_TYPES = {"BELGB", "BETB"}

# name -> (chemistry analyte, chemistry_wide column, transform, floor)
TARGETS = {
    "log10_tp_ug_l": ("tp", "tp_ug_l", "log10", 1.0),
    "log10_tn_mg_l": ("tn", "tn_mg_l", "log10", 0.01),
    "log10_cond_us_cm": ("cond", "cond_us_cm", "log10", 1.0),
    "log10_chloride_mg_l": ("chloride", "chloride_mg_l", "log10", 0.01),
    "log10_turbidity_ntu": ("turbidity", "turbidity_ntu", "log10", 0.01),
    "ph_lab": ("ph_lab", "ph_lab_su", "identity", None),
}
TARGET_NAMES = tuple(TARGETS)
# Auxiliary co-measured chemistry used ONLY as extra training outputs (multi-task supervision) by
# BioSet candidates with ``aux_targets: true``; never inputs, never evaluated as targets.
AUX_TARGETS = {  # name -> (chemistry_wide column, floor) ; log10 transform
    "aux_log10_sulfate": ("sulfate_mg_l", 0.01), "aux_log10_doc": ("doc_mg_l", 0.1),
    "aux_log10_nh4n": ("nh4_n_mg_l", 0.001), "aux_log10_no3n": ("no3_n_mg_l", 0.001),
    "aux_log10_tss": ("tss_mg_l", 0.1), "aux_log10_ca": ("calcium_mg_l", 0.01),
    "aux_log10_mg": ("magnesium_mg_l", 0.01), "aux_log10_na": ("sodium_mg_l", 0.01),
    "aux_log10_k": ("potassium_mg_l", 0.01), "aux_log10_silica": ("silica_mg_l", 0.01),
}
# Every chemistry_wide column is forbidden as an input (protocol §4).
CHEMISTRY_INPUT_BLOCKLIST_PREFIXES = ("tp_", "tn_", "cond", "chloride", "turbidity", "ph_", "nh4", "no2", "no3",
                                      "doc", "tss", "anc", "calcium", "magnesium", "sodium", "potassium",
                                      "sulfate", "silica", "chla", "microcystin", "cylindro", "color", "tdn",
                                      "tdp", "tkn", "aluminum", "pheophytin", "do_field", "temp_field")


class HoldoutAccessError(RuntimeError):
    """Raised when phase-1 code asks for an external-test survey."""


# ------------------------------------------------------------------------ helpers

def location_clusters(lat, lon, radius_km: float = LOCATION_RADIUS_KM) -> np.ndarray:
    """Connected components of the graph "coordinates <= radius_km apart" (labels 0..n-1)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from sklearn.neighbors import BallTree

    from sentinel.eval.splits import EARTH_RADIUS_KM

    X = np.radians(np.column_stack([np.asarray(lat, float), np.asarray(lon, float)]))
    if len(X) == 0:
        return np.zeros(0, int)
    ind = BallTree(X, metric="haversine").query_radius(X, r=radius_km / EARTH_RADIUS_KM)
    r = np.concatenate([np.full(len(i), k) for k, i in enumerate(ind)])
    c = np.concatenate(ind)
    _, lab = connected_components(coo_matrix((np.ones(len(r)), (r, c)), shape=(len(X), len(X))), directed=False)
    return lab


def transform_target(x, transform: str, floor: float | None) -> np.ndarray:
    x = np.asarray(x, float)
    if transform == "log10":
        with np.errstate(invalid="ignore"):
            return np.log10(np.clip(x, floor, None))
    return x


def target_frame(chem_wide: pd.DataFrame, chem_long: pd.DataFrame, visit_ids) -> tuple[pd.DataFrame, dict]:
    """The six targets for ``visit_ids`` (index = visit_id). Censored -> detection limit, then floor."""
    w = chem_wide.set_index("visit_id").reindex(pd.Index(visit_ids, name="visit_id"))
    out = pd.DataFrame(index=w.index)
    n_censored = {}
    for name, (analyte, col, tf, floor) in TARGETS.items():
        x = w[col].astype(float)
        cl = chem_long[(chem_long["analyte"] == analyte) & chem_long["censored"].fillna(False).astype(bool)]
        cl = cl[cl["visit_id"].isin(w.index)]
        dl = cl.groupby("visit_id")["detection_limit_canonical"].min()
        fill = dl.reindex(w.index)
        sub = x.isna() & fill.notna()
        x = x.where(~sub, fill)
        n_censored[name] = int(sub.sum())
        out[name] = transform_target(x.to_numpy(), tf, floor)
    return out, n_censored


# ------------------------------------------------------------------------ dataset

@dataclass
class BioDesign:
    samples: pd.DataFrame      # index visit_id
    tokens: pd.DataFrame       # long, visit_id × source taxon
    Y: pd.DataFrame            # index visit_id, TARGET_NAMES
    info: dict
    A: pd.DataFrame | None = None   # auxiliary training-only outputs (AUX_TARGETS), same index

    @property
    def visit_ids(self) -> np.ndarray:
        return self.samples.index.to_numpy()

    def subset(self, visit_ids) -> "BioDesign":
        ids = pd.Index(visit_ids)
        return BioDesign(self.samples.loc[ids], self.tokens[self.tokens["visit_id"].isin(ids)],
                         self.Y.loc[ids], dict(self.info), None if self.A is None else self.A.loc[ids])


def _nrsa_paths() -> tuple[Path, Path]:
    return paths.processed("nars_streams"), paths.processed("taxonomy_harmonised")


def all_site_locations(visits: pd.DataFrame, radius_km: float = LOCATION_RADIUS_KM) -> pd.DataFrame:
    """(survey, site_id) -> location_id over every NRSA cycle (coordinates only, no labels)."""
    s = visits.groupby(["survey", "site_id"], as_index=False)[["lat", "lon"]].first()
    s["location_id"] = [f"L{k:05d}" for k in location_clusters(s["lat"], s["lon"], radius_km)]
    return s


def harmonised_benthic_taxa(taxon_map: pd.DataFrame) -> pd.DataFrame:
    t = taxon_map[(taxon_map["source"] == "nars_streams") & (taxon_map["assemblage"] == "benthic")]
    t = t[["dataset", "source_taxon_id", "source_taxon_name", "name_clean", "identification_rank", "h_phylum",
           "h_class", "h_order", "h_family", "h_genus", "mixed_otu", "analysis_taxon"]].copy()
    return t.rename(columns={"dataset": "survey", "source_taxon_id": "taxa_id", "source_taxon_name": "taxon_name",
                             "h_phylum": "phylum", "h_class": "class", "h_order": "order", "h_family": "family",
                             "h_genus": "genus", "name_clean": "name"})


def load_nrsa(surveys=DEV_SURVEYS, *, allow_holdout: bool = False,
              radius_km: float = LOCATION_RADIUS_KM) -> BioDesign:
    """Load the community -> chemistry design for ``surveys`` (first visits with a benthic sample)."""
    surveys = tuple(surveys)
    bad = [s for s in surveys if s not in ALL_SURVEYS]
    if bad:
        raise ValueError(f"unknown surveys {bad}")
    held = [s for s in surveys if s in HOLDOUT_SURVEYS]
    if held and not allow_holdout:
        raise HoldoutAccessError(f"{held} are external-test surveys (docs/preregistration/bioassess_protocol.md "
                                 "§2); phase-1 code must not load them")
    proc, tax = _nrsa_paths()
    visits = pd.read_parquet(proc / "visits.parquet")
    loc = all_site_locations(visits, radius_km)
    v = visits[visits["survey"].isin(surveys) & (visits["visit_no"] == "1") & visits["has_benthic"].fillna(False)]
    v = v.merge(loc[["survey", "site_id", "location_id"]], on=["survey", "site_id"], how="left")
    counts = pd.read_parquet(proc / "benthic_counts.parquet",
                             columns=["visit_id", "survey", "sample_type", "taxa_id", "taxon_name", "count_300"])
    counts = counts[counts["visit_id"].isin(v["visit_id"]) & (counts["count_300"] > 0)]
    tm = harmonised_benthic_taxa(pd.read_parquet(tax / "taxon_map.parquet"))
    tok = counts.merge(tm, on=["survey", "taxa_id", "taxon_name"], how="left", validate="many_to_one")
    if tok["name"].isna().any():
        raise AssertionError(f"{int(tok['name'].isna().sum())} benthic rows without a harmonised taxon")
    keys = ["visit_id", "name", "phylum", "class", "order", "family", "genus", "mixed_otu", "identification_rank"]
    tok[keys[1:]] = tok[keys[1:]].astype(object).where(tok[keys[1:]].notna(), None)
    tok = (tok.groupby(keys, dropna=False, as_index=False)
              .agg(count=("count_300", "sum"), sample_type=("sample_type", "first")))
    stype = tok.groupby("visit_id")["sample_type"].first()
    agg = tok.groupby("visit_id").agg(total=("count", "sum"), n_taxa=("name", "size"))
    v = v[v["visit_id"].isin(agg.index)].set_index("visit_id")
    samples = pd.DataFrame({
        "survey": v["survey"], "site_id": v["site_id"].astype(str), "location_id": v["location_id"],
        "ag_eco9": v["ag_eco9"], "huc8": v["huc8"], "lat": v["lat"].astype(float), "lon": v["lon"].astype(float),
        "elevation_m": v["elevation_m"].astype(float), "reference_class": v["reference_class"],
        "log10_total_count": np.log10(agg["total"].reindex(v.index).to_numpy(float)),
        "log10_n_taxa": np.log10(agg["n_taxa"].reindex(v.index).to_numpy(float)),
        "boatable": stype.reindex(v.index).isin(BOATABLE_TYPES).astype(float).to_numpy(),
    }, index=v.index)
    # sampling day-of-year (collection metadata, not chemistry) for the context variants
    doy = pd.to_datetime(v["date"]).dt.dayofyear.to_numpy(float)
    samples["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    samples["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    chem_wide = pd.read_parquet(proc / "chemistry_wide.parquet")
    chem_long = pd.read_parquet(proc / "chemistry_long.parquet",
                                columns=["visit_id", "analyte", "censored", "detection_limit_canonical"])
    Y, n_cens = target_frame(chem_wide, chem_long, samples.index)
    keep = Y.notna().any(axis=1)
    samples, Y = samples[keep], Y[keep]
    tok = tok[tok["visit_id"].isin(samples.index)].drop(columns="sample_type").reset_index(drop=True)
    info = {"surveys": list(surveys), "n_samples": int(len(samples)), "n_sites": int(samples["site_id"].nunique()),
            "n_locations": int(samples["location_id"].nunique()), "n_token_rows": int(len(tok)),
            "n_censored_set_to_dl": n_cens, "location_radius_km": radius_km,
            "n_targets_present": {k: int(Y[k].notna().sum()) for k in TARGET_NAMES}}
    w = chem_wide.set_index("visit_id").reindex(samples.index)
    A = pd.DataFrame({k: transform_target(w[c].to_numpy(float), "log10", f) for k, (c, f) in AUX_TARGETS.items()},
                     index=samples.index)
    info["n_aux_present"] = {k: int(A[k].notna().sum()) for k in AUX_TARGETS}
    return BioDesign(samples.sort_index(), tok, Y.sort_index(), info, A.sort_index())


def assert_dev_only(samples: pd.DataFrame) -> None:
    """Phase-1 guard used by every CV script."""
    held = sorted(set(samples["survey"]) & set(HOLDOUT_SURVEYS))
    if held:
        raise HoldoutAccessError(f"external-test surveys present in a phase-1 design: {held}")


def input_hashes() -> dict:
    from sentinel.core.registry import file_sha256

    proc, tax = _nrsa_paths()
    out = {f"nars_streams/{f}": file_sha256(proc / f) for f in
           ("visits.parquet", "benthic_counts.parquet", "chemistry_wide.parquet", "chemistry_long.parquet")}
    out["taxonomy_harmonised/taxon_map.parquet"] = file_sha256(tax / "taxon_map.parquet")
    return out
