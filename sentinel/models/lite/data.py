# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Labels, water-body groups and chip access for SENTINEL-Lite (protocol §2–§3).

* :func:`load_matchups` reads ``processed/naip_matchups/matchups.parquet`` (manifest ``naip_matchups``).
* :func:`harmonise` converts units (documented factors only; turbidity keeps NTU only — FNU/NTRU are
  different variables per ``sentinel/data/wqp_harmonize.py``), applies plausibility bounds and drops
  censored results (they are counted, not substituted).
* :func:`chip_labels` aggregates samples to one label per (NAIP chip, target): median over the
  sample dates within ±1 day of the NAIP acquisition (one chip = one site × one DOQQ acquisition).
* :func:`water_body_groups` is the water-body proxy: connected components of sites ≤ ``radius_km``
  apart (no lake polygons are available locally; NHD waterbody joins are future work).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ...core import paths
from ...data.wqp_harmonize import LENGTH_TO_M

MANIFESTS = ("naip_matchups",)
TARGETS = ("secchi", "turbidity", "tss", "chla")
REG_TARGETS = ("secchi", "turbidity", "tss")          # reported as concentrations / depths
CLASS_TARGETS = ("chla",)                             # reported as trophic classes only (scope rule)
TARGET_UNIT = {"secchi": "m", "turbidity": "NTU", "tss": "mg/L", "chla": "ug/L"}
# unit -> factor to TARGET_UNIT (exact, documented). Anything else is dropped and counted.
UNIT_FACTORS = {
    "secchi": {k: v for k, v in LENGTH_TO_M.items() if k in ("m", "cm", "ft")},
    "turbidity": {"NTU": 1.0},                       # FNU / NTRU / FNRU / FNMU / JTU are other variables
    "tss": {"mg/L": 1.0, "ppm": 1.0},                # ppm (w/v, water) == mg/L
    "chla": {"ug/L": 1.0, "mg/m3": 1.0, "ppb": 1.0}, # mg/L (1 row) and blank units are dropped
}
BOUNDS = {"secchi": (0.05, 30.0), "turbidity": (0.05, 1000.0), "tss": (0.1, 1000.0), "chla": (0.05, 1000.0)}
# Carlson (1977) TSI(chl) = 9.81 ln(chl) + 30.6; class edges at TSI 40 / 50 / 70.
TROPHIC_EDGES_UGL = (2.6, 7.3, 56.0)
TROPHIC_NAMES = ("oligotrophic", "mesotrophic", "eutrophic", "hypereutrophic")
NAIP_BANDS = ("R", "G", "B", "NIR")
CHIP_SIZE = 256


def matchups_path() -> Path:
    return paths.PROCESSED / "naip_matchups" / "matchups.parquet"


def load_matchups(path: str | Path | None = None) -> pd.DataFrame:
    m = pd.read_parquet(path or matchups_path())
    return m[m["naip_status"] == "ok"].reset_index(drop=True)


def trophic_class(chl_ugl) -> np.ndarray:
    """0..3 = oligo / meso / eutro / hypereutrophic (NaN -> -1)."""
    c = np.asarray(chl_ugl, float)
    out = np.digitize(c, TROPHIC_EDGES_UGL).astype(int)
    out[~np.isfinite(c)] = -1
    return out


def trophic_class_log10(log10_chl) -> np.ndarray:
    return trophic_class(np.power(10.0, np.asarray(log10_chl, float)))


def harmonise(m: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return samples with ``value_h`` in TARGET_UNIT, plus a drop report."""
    rep: dict = {"n_in": int(len(m))}
    d = m[m["target"].isin(TARGETS)].copy()
    rep["dropped_other_targets"] = int(len(m) - len(d))
    unit = d["unit"].fillna("").astype(str).str.strip()
    fac = np.array([UNIT_FACTORS[t].get(u, np.nan) for t, u in zip(d["target"], unit)], float)
    rep["dropped_unit"] = {t: int(np.sum(np.isnan(fac) & (d["target"] == t).to_numpy())) for t in TARGETS}
    val = d["value"].astype(float).to_numpy()
    d["value_h"] = val * fac
    d["unit_h"] = d["target"].map(TARGET_UNIT)
    cens = d["censored"].fillna(False).astype(bool)
    # censored results (all have NaN value in this source) are counted here, before the finite filter (W11).
    # Right-censored Secchi (disk visible on the bottom) are the clearest waters: dropping them truncates the
    # clear end of the distribution (stated in the protocol).
    rep["dropped_censored"] = {t: int(((d["target"] == t) & cens & ~np.isnan(fac)).sum()) for t in TARGETS}
    rep["dropped_missing_value"] = {t: int(((d["target"] == t) & ~cens & ~np.isnan(fac)
                                            & ~np.isfinite(val)).sum()) for t in TARGETS}
    d = d[~cens]
    d = d[np.isfinite(d["value_h"])]
    lo = d["target"].map(lambda t: BOUNDS[t][0])
    hi = d["target"].map(lambda t: BOUNDS[t][1])
    ok = (d["value_h"] >= lo) & (d["value_h"] <= hi)
    rep["dropped_bounds"] = {t: int(((d["target"] == t) & ~ok).sum()) for t in TARGETS}
    d = d[ok].copy()
    rep["n_out"] = int(len(d))
    rep["n_out_by_target"] = d["target"].value_counts().to_dict()
    return d.reset_index(drop=True), rep


def chip_labels(samples: pd.DataFrame) -> pd.DataFrame:
    """One row per NAIP chip: log10 label per target (NaN if absent) + metadata."""
    g = samples.groupby(["naip_chip_id", "target"])
    agg = g["value_h"].median().unstack("target")
    nsam = g["value_h"].size().unstack("target")
    out = pd.DataFrame(index=agg.index)
    for t in TARGETS:
        out[f"y_{t}"] = np.log10(agg[t]) if t in agg else np.nan
        out[f"n_{t}"] = nsam[t].fillna(0).astype(int) if t in nsam else 0
    meta_cols = ["site_id", "lat", "lon", "state", "site_class", "naip_item_id", "naip_date", "naip_year",
                 "gsd_m", "chip_crs", "chip_left", "chip_top", "naip_shard", "organization", "huc8", "name",
                 "s2_chip_id", "dt_days_s2_naip"]
    meta = samples.sort_values("naip_chip_id").groupby("naip_chip_id")[meta_cols].first()
    # sample date nearest the NAIP date (for DOY / season features)
    sd = samples.assign(_ad=samples["dt_days_naip"].abs()).sort_values(["naip_chip_id", "_ad"])
    meta["sample_date"] = sd.groupby("naip_chip_id")["date"].first()
    out = meta.join(out).reset_index()
    out["doy"] = pd.to_datetime(out["naip_date"]).dt.dayofyear.astype(int)
    out["y_chla_class"] = trophic_class_log10(out["y_chla"])
    return out


_NAME_CUT = re.compile(r"\s+-\s+|\(|,|#|\s@\s|\s(?:at|near|nr|abv|blw|above|below|upstream|downstream)\s", re.I)
_NAME_DROP = {"lake", "lk", "lac", "deep", "hole", "deepest", "spot", "basin", "main", "site", "station", "sta",
              "mid", "middle", "north", "south", "east", "west", "n", "s", "e", "w", "end", "center", "centre",
              "the", "of"}
FLOWING_CLASSES = ("river_stream", "estuary_coastal", "canal", "other", "wetland")


_WATER_WORDS = {"river", "creek", "bay", "lagoon", "sound", "harbor", "harbour", "inlet", "reservoir", "canal",
                "slough", "bayou", "estuary", "flowage", "pond", "channel"}
_ABBREV = {"r": "river", "cr": "creek", "ck": "creek", "res": "reservoir", "st": "st", "lk": "lake"}


def water_body_name(name: str) -> str:
    """Normalised water-body name for merging ('' = unusable).

    'Green Lake - West Basin - Deep Hole' -> 'green'; 'MINNETONKA (HALSTEDS BAY)' -> 'minnetonka';
    'Mississippi River 701.1B' / 'MISSISSIPPI R OFF OF BOOM ISLAND' -> 'mississippi river';
    'St. Croix River Just Upstream from ...' -> 'st croix river'; 'JUP122' -> 'jup' (station-code prefix).
    Merging is only ever *more* conservative (bigger groups), so false merges cost data, not validity."""
    if not isinstance(name, str):
        return ""
    head = _NAME_CUT.split(name.strip(), maxsplit=1)[0].lower()
    toks = [re.sub(r"\d+", "", t) for t in re.split(r"[^a-z0-9']+", head) if t]
    toks = [_ABBREV.get(t, t) if i > 0 else t for i, t in enumerate(toks) if t]
    for i, t in enumerate(toks):          # keep the name up to the first water-type word
        if t in _WATER_WORDS and i > 0:
            toks = toks[:i + 1]
            break
    kept = [t for t in toks if t not in _NAME_DROP]
    out = " ".join(kept)
    return out if len(out) >= 3 else ""


LAKE_MAX_DIST_M = 100.0   # same rule as SENTINEL-Orbit (sentinel/models/orbit/split.py)


def hydrolakes_ids(sites: pd.DataFrame, use_cache: bool = True) -> pd.Series:
    """HydroLAKES v1.0 lake id per site (polygon containing the site, else nearest within 100 m; <NA> = none).
    Read-only use of ``processed/hydrolakes`` (manifest ``hydrolakes``); cached under $SENTINEL_RUNS/lite/cache."""
    s = sites.drop_duplicates("site_id")
    cp = paths.run_dir("lite", "cache") / "hydrolakes_assign_v1.parquet"
    key = s["site_id"].astype(str) + "|" + s["lat"].round(6).astype(str) + "|" + s["lon"].round(6).astype(str)
    cache = pd.read_parquet(cp) if (use_cache and cp.exists()) else pd.DataFrame(columns=["key", "hylak_id"])
    todo = s[~key.isin(set(cache["key"]))]
    if len(todo):
        import geopandas as gpd

        from ...data.sources import hydrolakes as H

        lakes = gpd.read_parquet(paths.PROCESSED / "hydrolakes" / "hydrolakes_na_eu.parquet",
                                 columns=["Hylak_id", "Lake_area", "Pour_long", "Pour_lat", "Lake_name", "region",
                                          "geometry"])
        a = H.assign_points(todo["lon"].to_numpy(), todo["lat"].to_numpy(), lakes, max_dist_m=LAKE_MAX_DIST_M)
        new = pd.DataFrame({"key": key[~key.isin(set(cache["key"]))].to_numpy(),
                            "hylak_id": a["hylak_id"].astype("Int64").to_numpy()})
        cache = pd.concat([cache, new], ignore_index=True) if len(cache) else new
        if use_cache:
            cache.to_parquet(cp, index=False)
    m = dict(zip(cache["key"], cache["hylak_id"]))
    return pd.Series([m[k] for k in key], index=s["site_id"].to_numpy(), name="hylak_id", dtype="Int64")


def water_body_groups(sites: pd.DataFrame, radius_km: float = 2.0, name_km: float = 40.0,
                      hylak: pd.Series | None = None) -> pd.Series:
    """Water-body proxy id per site (union-find over four edge types, review W1):
      (a) sites <= ``radius_km`` apart; (b) same normalised name (:func:`water_body_name`) and <= ``name_km``
      apart (large lakes / bays sampled at distant stations); (c) flowing/estuarine sites (``FLOWING_CLASSES``)
      in the same HUC8 (flow-connected stations together, RESEARCH_PLAN R4); (d) same HydroLAKES lake
      (``hylak``: site_id -> Hylak_id, lakes >= 10 ha; same assignment rule as M2 Orbit).
    ``sites`` needs columns site_id, lat, lon, name, site_class, huc8."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from sklearn.neighbors import BallTree

    from ...eval.splits import EARTH_RADIUS_KM

    s = sites.drop_duplicates("site_id").reset_index(drop=True)
    X = np.radians(s[["lat", "lon"]].to_numpy(float))
    tree = BallTree(X, metric="haversine")
    rows, cols = [], []
    for k, idx in enumerate(tree.query_radius(X, r=radius_km / EARTH_RADIUS_KM)):
        rows += [k] * len(idx)
        cols += list(idx)
    nm = s["name"].map(water_body_name).to_numpy()
    near = tree.query_radius(X, r=name_km / EARTH_RADIUS_KM)
    for k, idx in enumerate(near):
        if nm[k]:
            same = [j for j in idx if nm[j] == nm[k]]
            rows += [k] * len(same)
            cols += same
    flow = s["site_class"].isin(FLOWING_CLASSES).to_numpy()
    huc = s["huc8"].fillna("").astype(str).to_numpy()
    for h in np.unique(huc[flow & (huc != "")]):
        members = np.flatnonzero(flow & (huc == h))
        rows += [int(members[0])] * len(members)
        cols += [int(m) for m in members]
    if hylak is not None:
        hl = s["site_id"].map(hylak)
        for _, members in pd.Series(np.arange(len(s))).groupby(hl.to_numpy()).groups.items():
            members = list(members)
            rows += [int(members[0])] * len(members)
            cols += [int(m) for m in members]
    n = len(s)
    _, lab = connected_components(coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)), directed=False)
    return pd.Series([f"wb{int(x):05d}" for x in lab], index=s["site_id"].to_numpy(), name="water_body")


def build_label_table(matchups: pd.DataFrame | None = None, *, radius_km: float = 2.0,
                      name_km: float = 40.0, use_hydrolakes: bool = True) -> tuple[pd.DataFrame, dict]:
    m = load_matchups() if matchups is None else matchups
    s, rep = harmonise(m)
    lab = chip_labels(s)
    sites = lab[["site_id", "lat", "lon", "name", "site_class", "huc8"]]
    hylak = hydrolakes_ids(sites) if use_hydrolakes else None
    wb = water_body_groups(sites, radius_km, name_km, hylak)
    lab["water_body"] = lab["site_id"].map(wb)
    if hylak is not None:
        lab["hylak_id"] = lab["site_id"].map(hylak).astype("Int64")
        rep["n_sites_in_hydrolakes"] = int(hylak.notna().sum())
    rep["n_chips"] = int(len(lab))
    rep["n_sites"] = int(lab["site_id"].nunique())
    rep["n_water_bodies"] = int(lab["water_body"].nunique())
    rep["water_body_rule"] = {"radius_km": radius_km, "name_km": name_km, "flowing_huc8": list(FLOWING_CLASSES),
                              "hydrolakes": bool(use_hydrolakes), "lake_max_dist_m": LAKE_MAX_DIST_M}
    return lab, rep


# ------------------------------------------------------------------------------ chips

def naip_store():
    from ...data.chips import ChipStore

    return ChipStore(paths.PROCESSED / "naip_chips" / "wqp")


def teacher_store():
    from ...data.chips import ChipStore

    return ChipStore(paths.PROCESSED / "s2_chips" / "naip_teacher")


def load_chips(chip_ids, store=None) -> np.ndarray:
    """uint8 array (N, 4, 256, 256) in NAIP band order R, G, B, NIR."""
    st = store or naip_store()
    if st.layers != list(NAIP_BANDS):
        raise ValueError(f"unexpected NAIP layers {st.layers}")
    out = np.zeros((len(chip_ids), 4, CHIP_SIZE, CHIP_SIZE), np.uint8)
    for i, cid in enumerate(chip_ids):
        a = st.load(cid)
        if a.shape != (4, CHIP_SIZE, CHIP_SIZE):
            raise ValueError(f"chip {cid} has shape {a.shape}")
        out[i] = a
    return out


def scene_stats(path: str | Path | None = None) -> pd.DataFrame:
    p = Path(path) if path else paths.PROCESSED / "naip_matchups" / "naip_scene_stats.parquet"
    return pd.read_parquet(p).drop_duplicates("naip_item_id", keep="last").set_index("naip_item_id")


SCENE_NORM_COLS = {b: (f"{n}_p02", f"{n}_p50", f"{n}_p98") for b, n in
                   zip(NAIP_BANDS, ("red", "green", "blue", "nir"))}


def scene_norm_params(lab: pd.DataFrame, stats: pd.DataFrame) -> np.ndarray:
    """(N, 4, 2) per-chip (centre, scale) from the chip's DOQQ statistics: (p50, p98 - p02)."""
    st = stats.reindex(lab["naip_item_id"])
    out = np.zeros((len(lab), 4, 2), np.float32)
    for k, b in enumerate(NAIP_BANDS):
        p02, p50, p98 = (st[c].to_numpy(float) for c in SCENE_NORM_COLS[b])
        out[:, k, 0] = p50
        out[:, k, 1] = np.maximum(p98 - p02, 1.0)
    bad = ~np.isfinite(out).all(axis=(1, 2))
    if bad.any():
        raise ValueError(f"{bad.sum()} chips lack DOQQ statistics")
    return out


def write_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=str))


# ------------------------------------------------------------------------- role arrays (guarded)

def labels_path() -> Path:
    return paths.PROCESSED / "lite_labels" / "labels.parquet"


def load_labels() -> pd.DataFrame:
    return pd.read_parquet(labels_path())


def role_frame(split: dict, roles, *, allow_test: bool = False, training: bool = True, freeze_file=None):
    """Label rows for ``roles`` (guarded: protected roles need the freeze and never reach training)."""
    from . import guard

    guard.check_access(roles, allow_test=allow_test, training=training, split=split, freeze_file=freeze_file)
    lab = load_labels()
    from .splits import labels_sha256

    if split.get("labels_sha256") and labels_sha256(lab) != split["labels_sha256"]:
        raise guard.ProtectedDataError("labels.parquet content differs from the frozen split's labels_sha256")
    lab["role"] = lab["naip_chip_id"].astype(str).map(split["chip_role"])
    return lab[lab["role"].isin(list(roles))].reset_index(drop=True)


def cached_chips(lab: pd.DataFrame, cache_name: str) -> np.ndarray:
    """Chips for ``lab`` rows, cached as one uint8 .npy under $SENTINEL_RUNS/lite/cache (order = lab order)."""
    cdir = paths.run_dir("lite", "cache")
    ids = lab["naip_chip_id"].astype(str).tolist()
    import hashlib

    key = hashlib.sha256("|".join(ids).encode()).hexdigest()[:12]
    f = cdir / f"{cache_name}_{key}.npy"
    if f.exists():
        return np.load(f, mmap_mode="r")
    arr = load_chips(ids)
    tmp = f.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    tmp.rename(f)
    return np.load(f, mmap_mode="r")
