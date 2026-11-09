# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Builder for processed('uk_ea_ecology'): EA invertebrate + diatom communities with paired
Water Quality Archive chemistry.

Pairing rule (fixed before looking at any model result; documented in
docs/datasheets/uk_ea_ecology.md):

1. **Space.** Each ecology site (invertebrate and diatom sites; all waterbody types except
   estuarine ``WBES``) is paired with
   the *nearest* WQA sampling point of a freshwater river-like type (F6 rivers, FJ canals,
   FK land drains, FZ unspecified freshwater; any open/closed status) within
   ``PAIR_MAX_DIST_M`` = 1,000 m straight-line distance in British National Grid. The
   distance is stored (``pair_distance_m``) so users can tighten it (e.g. <= 250 m).
   The WQA does not publish WFD water-body IDs for its points, so "same water body" cannot
   be enforced; a tributary/confluence mismatch is possible for larger distances.
2. **Time.** For each invertebrate or diatom sample (date d, 2000 onwards) the paired point's
   observations of each core determinand are summarised in two windows:
   ``pm30d`` = [d-30, d+30] days and ``prior365d`` = [d-365, d]. Summaries: n, median, mean,
   min, max, fraction censored, and the value nearest in time with its lag (days, obs - d).
   Censored results enter the summaries at their reported limit and ``frac_censored`` says how
   many did (raw censoring flags are kept in ``wq_observations``). Units are never mixed:
   every summary is per (determinand, unit). Only routine samples of the water itself are
   summarised: sample material river/running surface water, canal water or "any water", and
   not reactive pollution-incident sampling (effluent, precipitation, estuarine and incident
   samples stay in ``wq_observations`` but are excluded from the windows).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .bio_common import write_parquet
from .sources import uk_ea as U

log = logging.getLogger(__name__)

PAIR_MAX_DIST_M = 1000.0
EXCLUDED_SITE_TYPES = ("WBES",)
WINDOWS = {"pm30d": (-30, 30), "prior365d": (-365, 0)}
# Only routine samples of the water itself enter the window summaries (wq_observations keeps all):
SUMMARY_MATERIALS = ("RIVER / RUNNING SURFACE WATER", "CANAL WATER", "ANY WATER")
EXCLUDED_PURPOSE_PREFIX = "UNPLANNED REACTIVE MONITORING"   # pollution-incident samples


# ------------------------------------------------------------------------ loading

def load_sites(zip_path: Path, assemblage: str) -> pd.DataFrame:
    raw = U.read_zip_table(zip_path, "_SITE.csv")
    sites = U.tidy_sites(raw, assemblage)
    lon, lat = U.bng_to_wgs84(sites["easting_bng"], sites["northing_bng"])
    sites["lon"] = lon
    sites["lat"] = lat
    return sites


def load_metrics(zip_path: Path, assemblage: str) -> pd.DataFrame:
    raw = U.read_zip_table(zip_path, "_METRICS.csv")
    return U.tidy_metrics(raw, assemblage)


# ------------------------------------------------------------------------ pairing

def pair_sites_to_points(sites: pd.DataFrame, points: pd.DataFrame, *, max_dist_m: float = PAIR_MAX_DIST_M,
                         point_types=U.WQ_PAIR_POINT_TYPES) -> pd.DataFrame:
    """Nearest eligible WQ point within ``max_dist_m`` for every eligible site (BNG metres)."""
    from scipy.spatial import cKDTree

    s = sites[~sites["waterbody_type"].isin(EXCLUDED_SITE_TYPES)]
    s = s[np.isfinite(s["easting_bng"]) & np.isfinite(s["northing_bng"])]
    p = points[points["wq_type"].isin(point_types)]
    p = p[np.isfinite(p["wq_easting_bng"]) & np.isfinite(p["wq_northing_bng"])].reset_index(drop=True)
    if s.empty or p.empty:
        return pd.DataFrame(columns=["site_id", "wq_point", "pair_distance_m"])
    tree = cKDTree(p[["wq_easting_bng", "wq_northing_bng"]].to_numpy(float))
    d, i = tree.query(s[["easting_bng", "northing_bng"]].to_numpy(float), k=2,
                      distance_upper_bound=max_dist_m)
    ok = np.isfinite(d[:, 0])
    out = pd.DataFrame({
        "site_id": s["site_id"].to_numpy()[ok],
        "wq_point": p["wq_point"].to_numpy()[i[ok, 0]],
        "pair_distance_m": d[ok, 0],
        "second_wq_point": np.where(np.isfinite(d[ok, 1]),
                                    p["wq_point"].to_numpy()[np.minimum(i[ok, 1], len(p) - 1)], None),
        "second_distance_m": np.where(np.isfinite(d[ok, 1]), d[ok, 1], np.nan),
    })
    meta = p.set_index("wq_point")[["wq_type", "wq_status", "wq_point_label"]]
    out = out.join(meta, on="wq_point")
    out["pair_rule"] = f"nearest {'/'.join(point_types)} point within {max_dist_m:.0f} m (BNG)"
    return out


def years_needed(pairs: pd.DataFrame, samples: pd.DataFrame, *, first_year: int = U.WQ_FIRST_YEAR,
                 last_year: int | None = None) -> pd.DataFrame:
    """(wq_point, year) pairs covering every window of every paired invertebrate sample."""
    last_year = last_year or pd.Timestamp.today().year
    lo_off = min(w[0] for w in WINDOWS.values())
    hi_off = max(w[1] for w in WINDOWS.values())
    sm = samples[["site_id", "sample_date"]].dropna().drop_duplicates()
    sm = sm.merge(pairs[["site_id", "wq_point"]], on="site_id")
    d = pd.to_datetime(sm["sample_date"])
    sm = sm.assign(y0=(d + pd.Timedelta(days=lo_off)).dt.year, y1=(d + pd.Timedelta(days=hi_off)).dt.year)
    sm = sm[(sm["y1"] >= first_year)]
    rows = set()
    for pt, y0, y1 in sm[["wq_point", "y0", "y1"]].drop_duplicates().itertuples(index=False):
        for y in range(max(int(y0), first_year), min(int(y1), last_year) + 1):
            rows.add((pt, y))
    return pd.DataFrame(sorted(rows), columns=["wq_point", "year"])


# --------------------------------------------------------------- chemistry windows

def summarise_windows(samples: pd.DataFrame, pairs: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per (sample, determinand, unit, window) with summary statistics."""
    sm = samples[["sample_id", "site_id", "sample_date"]].dropna().drop_duplicates("sample_id")
    sm = sm.merge(pairs[["site_id", "wq_point", "pair_distance_m"]], on="site_id")
    sm["sample_date"] = pd.to_datetime(sm["sample_date"])
    ob = obs[np.isfinite(obs["value"])]
    if "sample_material" in ob:
        ob = ob[ob["sample_material"].isin(SUMMARY_MATERIALS)]
    if "sampling_purpose" in ob:
        ob = ob[~ob["sampling_purpose"].fillna("").str.startswith(EXCLUDED_PURPOSE_PREFIX)]
    ob = ob[["wq_point", "date", "determinand", "unit", "value", "censored"]].copy()
    ob["date"] = pd.to_datetime(ob["date"])
    by_point = {pt: g.reset_index(drop=True) for pt, g in sm.groupby("wq_point", sort=False)}
    out = []
    for (pt, det, unit), g in ob.groupby(["wq_point", "determinand", "unit"], sort=False):
        ss = by_point.get(pt)
        if ss is None or ss.empty:
            continue
        od = g["date"].to_numpy("datetime64[D]").astype(np.int64)
        order = np.argsort(od, kind="stable")
        od = od[order]
        ov = g["value"].to_numpy(float)[order]
        oc = g["censored"].to_numpy(bool)[order]
        sd = ss["sample_date"].to_numpy("datetime64[D]").astype(np.int64)
        for wname, (a, b) in WINDOWS.items():
            lo = np.searchsorted(od, sd + a, side="left")
            hi = np.searchsorted(od, sd + b, side="right")
            for k in np.flatnonzero(hi > lo):
                v = ov[lo[k]:hi[k]]
                c = oc[lo[k]:hi[k]]
                lag = od[lo[k]:hi[k]] - sd[k]
                j = int(np.argmin(np.abs(lag)))
                out.append((ss["sample_id"].iat[k], ss["site_id"].iat[k], pt, ss["pair_distance_m"].iat[k],
                            det, unit, wname, len(v), float(np.median(v)), float(v.mean()), float(v.min()),
                            float(v.max()), float(c.mean()), float(v[j]), int(lag[j])))
    cols = ["sample_id", "site_id", "wq_point", "pair_distance_m", "determinand", "unit", "window", "n_obs",
            "median", "mean", "min", "max", "frac_censored", "nearest_value", "nearest_lag_days"]
    df = pd.DataFrame(out, columns=cols)
    df["short_name"] = df["determinand"].map(U.DETERMINANDS)
    return df


def _unit_slug(unit: str) -> str:
    u = str(unit).lower().replace("/", "_per_").replace(" ", "").replace("%", "pct")
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in u).strip("_")


def chemistry_wide(long: pd.DataFrame, window: str = "pm30d") -> pd.DataFrame:
    """Wide convenience table (one row per invertebrate sample): ``<short>_<unit>_<window>_median``
    and ``..._n`` columns; the unit is part of the column name so units are never mixed."""
    d = long[long["window"] == window].copy()
    if d.empty:
        return pd.DataFrame(columns=["sample_id"])
    d["var"] = d["short_name"] + "__" + d["unit"].map(_unit_slug) + "__" + window
    med = d.pivot_table(index="sample_id", columns="var", values="median", aggfunc="first")
    n = d.pivot_table(index="sample_id", columns="var", values="n_obs", aggfunc="first")
    med.columns = [c + "__median" for c in med.columns]
    n.columns = [c + "__n" for c in n.columns]
    meta = d.drop_duplicates("sample_id").set_index("sample_id")[["site_id", "wq_point", "pair_distance_m"]]
    return meta.join(med).join(n).reset_index()


# ------------------------------------------------------------------------ builder

def build(raw_dir: Path, out_dir: Path) -> dict:
    """Build every table of processed('uk_ea_ecology'); returns stats for the manifest."""
    eco = raw_dir / "ecology"
    wq = raw_dir / "wq"
    stats: dict = {}
    inv_zip, diat_zip = eco / "INV_OPEN_DATA.zip", eco / "DIAT_OPEN_DATA.zip"
    # NB: pairing and chemistry windows are computed for invertebrate AND diatom samples.

    taxa_info = U.tidy_taxon_info(U.read_zip_table(eco / "OPEN_DATA_TAXON_INFO.zip", "TAXON_INFO.csv"))
    write_parquet(taxa_info, out_dir / "taxon_dictionary.parquet")
    stats["taxon_dictionary_rows"] = len(taxa_info)

    sites = pd.concat([load_sites(inv_zip, "invertebrate"), load_sites(diat_zip, "diatom")], ignore_index=True)
    write_parquet(sites, out_dir / "sites.parquet")
    stats["sites"] = sites.groupby("assemblage")["site_id"].nunique().to_dict()

    for assemblage, zp in (("invertebrate", inv_zip), ("diatom", diat_zip)):
        met = load_metrics(zp, assemblage)
        write_parquet(met, out_dir / f"samples_{assemblage}.parquet")
        parts = []
        n = 0
        for ch in U.iter_zip_table(zp, "_TAXA.csv", chunksize=1_000_000):
            t = U.tidy_taxa_chunk(ch, assemblage)
            parts.append(t)
            n += len(t)
        counts = pd.concat(parts, ignore_index=True)
        for c in ("analysis_method", "abundance_category", "analysis_type"):
            if c in counts:
                counts[c] = counts[c].astype("category")
        write_parquet(counts, out_dir / f"counts_{assemblage}.parquet")
        sd = pd.to_datetime(met["sample_date"])
        stats[assemblage] = {
            "analyses": int(met["analysis_id"].nunique()), "samples": int(met["sample_id"].nunique()),
            "sites_with_samples": int(met["site_id"].nunique()), "taxon_records": int(n),
            "taxa": int(counts["taxon_list_item_key"].nunique()),
            "date_min": str(sd.min().date()), "date_max": str(sd.max().date()),
            "analysis_type_counts": met["analysis_type"].value_counts().to_dict(),
            "analysis_method_counts": met["analysis_method"].value_counts().to_dict(),
            "taxa_keys_missing_from_dictionary": int((~counts["taxon_list_item_key"].drop_duplicates()
                                                      .isin(taxa_info["taxon_list_item_key"])).sum()),
        }
        log.info("%s: %s", assemblage, stats[assemblage])

    pts = pd.read_parquet(wq / "sampling_points_freshwater.parquet")
    write_parquet(pts, out_dir / "wq_points.parquet")
    pair_frames = []
    stats["pairing"] = {}
    for assemblage in ("invertebrate", "diatom"):
        asites = sites[sites["assemblage"] == assemblage]
        pr = pair_sites_to_points(asites, pts)
        pr["assemblage"] = assemblage
        pair_frames.append(pr)
        stats["pairing"][assemblage] = {
            "rule": pr["pair_rule"].iat[0] if len(pr) else None,
            "eligible_sites": int((~asites["waterbody_type"].isin(EXCLUDED_SITE_TYPES)).sum()),
            "paired_sites": len(pr),
            "distance_m_quantiles": {str(q): float(pr["pair_distance_m"].quantile(q))
                                     for q in (0.1, 0.25, 0.5, 0.75, 0.9)} if len(pr) else {},
            "paired_within_250m": int((pr["pair_distance_m"] <= 250).sum()),
            "paired_at_0m": int((pr["pair_distance_m"] == 0).sum()),
            "distinct_wq_points": int(pr["wq_point"].nunique()),
        }
    pairs_all = pd.concat(pair_frames, ignore_index=True)
    write_parquet(pairs_all, out_dir / "site_wq_pairs.parquet")

    obs_files = sorted((wq / "observations").rglob("*.parquet"))
    obs = pd.concat([d for d in (pd.read_parquet(f) for f in obs_files) if len(d)], ignore_index=True) \
        if obs_files else pd.DataFrame()
    if obs.empty:
        stats["wq_observations"] = "not downloaded"
        return stats
    obs = obs.drop_duplicates(["wq_point", "sample_notation", "observation_notation"])
    obs["censored"] = obs["censored"].fillna(False).astype(bool)
    obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
    obs["short_name"] = obs["determinand"].map(U.DETERMINANDS)
    write_parquet(obs, out_dir / "wq_observations.parquet")
    stats["wq_observations"] = {
        "rows": len(obs), "points": int(obs["wq_point"].nunique()),
        "date_min": str(pd.to_datetime(obs["date"]).min().date()),
        "date_max": str(pd.to_datetime(obs["date"]).max().date()),
        "units_by_determinand": obs.groupby("short_name")["unit"].unique().map(list).to_dict(),
        "frac_censored_by_determinand": obs.groupby("short_name")["censored"].mean().round(4).to_dict(),
        "rows_by_determinand": obs.groupby("short_name").size().to_dict(),
        "non_numeric_results": int(obs["value"].isna().sum()),
        "sample_materials": obs["sample_material"].value_counts().head(12).to_dict(),
        "sampling_purposes": obs["sampling_purpose"].value_counts().head(12).to_dict(),
        "rows_eligible_for_summaries": int((obs["sample_material"].isin(SUMMARY_MATERIALS)
                                            & ~obs["sampling_purpose"].fillna("").str.startswith(
                                                EXCLUDED_PURPOSE_PREFIX) & obs["value"].notna()).sum()),
    }
    longs = []
    stats["chemistry_matched"] = {}
    for assemblage in ("invertebrate", "diatom"):
        smp = pd.read_parquet(out_dir / f"samples_{assemblage}.parquet")
        smp = smp[pd.to_datetime(smp["sample_date"]) >= pd.Timestamp(U.WQ_FIRST_YEAR, 1, 1)]
        pr = pairs_all[pairs_all["assemblage"] == assemblage]
        lg = summarise_windows(smp, pr, obs)
        lg["assemblage"] = assemblage
        longs.append(lg)
        wide = chemistry_wide(lg, "pm30d")
        write_parquet(wide, out_dir / f"chemistry_matched_wide_pm30d_{assemblage}.parquet")
        stats["chemistry_matched"][assemblage] = {
            "samples_2000_on": int(smp["sample_id"].nunique()),
            "samples_at_paired_sites_2000_on": int(smp["site_id"].isin(pr["site_id"]).groupby(smp["sample_id"]).any().sum()),
            "samples_with_any_pm30d": int(lg.loc[lg["window"] == "pm30d", "sample_id"].nunique()),
            "samples_with_any_prior365d": int(lg.loc[lg["window"] == "prior365d", "sample_id"].nunique()),
            "samples_with_pm30d_by_determinand": lg[lg["window"] == "pm30d"].groupby("short_name")[
                "sample_id"].nunique().to_dict(),
        }
    write_parquet(pd.concat(longs, ignore_index=True), out_dir / "chemistry_matched_long.parquet")
    return stats
