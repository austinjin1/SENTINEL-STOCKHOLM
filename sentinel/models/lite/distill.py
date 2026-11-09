# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Cross-scale distillation data path (research 08 §c.6 R2): NAIP student pixels ↔ S2 10-m teacher cells.

Teacher-target file (written by ``scripts/train/lite_teacher_targets.py`` from the M2 Orbit teacher):
parquet with one row per (s2_chip_id, row, col, target): ``mu`` and ``sigma`` in log10 units of the Lite
target units (Secchi m, turbidity NTU, TSS mg/L, chl-a µg/L) and ``trust`` ∈ [0, 1] (0 = do not use; e.g.
teacher's own held-out error band, glint/adjacency flags). Only cells whose S2 SCL class is water (6) are used.
The headline is never student-vs-teacher; teacher targets only shape training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TARGETS, teacher_store
from .features import water_mask

S2_RES = 10.0
S2_SIZE = 64


def cell_map_for_chip(naip_crs: str, naip_left: float, naip_top: float, gsd: float, s2_crs: str, s2_left: float,
                      s2_top: float, size: int = 256, stride: int = 4) -> np.ndarray:
    """(size/stride, size/stride) int array of S2 chip flat indices (row*64+col), -1 outside the S2 chip.
    NAD83 vs WGS84 UTM datum differences (< ~1-2 m) are ignored (documented)."""
    from pyproj import Transformer

    h = size // stride
    jj, ii = np.meshgrid(np.arange(h), np.arange(h))
    px = (jj * stride + stride / 2.0)
    py = (ii * stride + stride / 2.0)
    x = naip_left + px * gsd
    y = naip_top - py * gsd
    if naip_crs != s2_crs:
        x, y = Transformer.from_crs(naip_crs, s2_crs, always_xy=True).transform(x, y)
    col = np.floor((np.asarray(x) - s2_left) / S2_RES).astype(int)
    row = np.floor((s2_top - np.asarray(y)) / S2_RES).astype(int)
    ok = (col >= 0) & (col < S2_SIZE) & (row >= 0) & (row < S2_SIZE)
    return np.where(ok, row * S2_SIZE + col, -1)


def load_kd_tensors(lab: pd.DataFrame, targets_path: str, *, chips: np.ndarray | None = None, stride: int = 4,
                    k_max: int | None = None) -> dict:
    """Build per-chip KD tensors aligned with ``lab`` order (chips without a teacher get t_w = 0)."""
    tt = pd.read_parquet(targets_path)
    need = {"s2_chip_id", "row", "col", "target", "mu", "sigma", "trust"}
    if not need <= set(tt.columns):
        raise ValueError(f"teacher targets need columns {sorted(need)}")
    st = teacher_store()
    sidx = st.index.set_index("chip_id")
    tt["flat"] = tt["row"].astype(int) * S2_SIZE + tt["col"].astype(int)
    by_chip = {k: g for k, g in tt.groupby("s2_chip_id")}
    from .data import cached_chips

    dn = np.asarray(chips) if chips is not None else np.asarray(cached_chips(lab, "train"))   # same cache file as the train inputs
    wm = water_mask(dn)
    n = len(lab)
    h = dn.shape[-1] // stride
    water_w = wm.reshape(n, h, stride, h, stride).mean(axis=(2, 4)).astype(np.float32)
    maps, cells = [], []
    for i, r in enumerate(lab.itertuples(index=False)):
        sid = getattr(r, "s2_chip_id")
        if not isinstance(sid, str) or sid not in by_chip or sid not in sidx.index:
            maps.append(np.full((h, h), -1))
            cells.append(np.zeros(0, int))
            continue
        s = sidx.loc[sid]
        cm = cell_map_for_chip(r.chip_crs, r.chip_left, r.chip_top, r.gsd_m, s["chip_crs"], s["chip_left"],
                               s["chip_top"], dn.shape[-1], stride)
        scl = st.layer(st.load(sid), "SCL").reshape(-1)
        cm = np.where((cm >= 0) & (scl[np.clip(cm, 0, None)] == 6), cm, -1)
        u = np.unique(cm[cm >= 0])
        loc = np.full(S2_SIZE * S2_SIZE, -1)
        loc[u] = np.arange(len(u))
        maps.append(np.where(cm >= 0, loc[np.clip(cm, 0, None)], -1))
        cells.append(u)
    K = k_max or max(1, max(len(c) for c in cells))
    maps = [np.where(m < K, m, -1) for m in maps]          # cells beyond k_max are dropped (review W7)
    T = len(TARGETS)
    t_mu = np.full((n, K, T), np.nan, np.float32)
    t_var = np.full((n, K, T), np.nan, np.float32)
    t_w = np.zeros((n, K, T), np.float32)                 # trust per target (review W7)
    for i, (r, u) in enumerate(zip(lab.itertuples(index=False), cells)):
        if not len(u):
            continue
        g = by_chip[r.s2_chip_id]
        pos = {c: k for k, c in enumerate(u[:K])}
        for row in g.itertuples(index=False):
            k = pos.get(int(row.flat))
            if k is None or row.target not in TARGETS:
                continue
            j = TARGETS.index(row.target)
            t_mu[i, k, j] = row.mu
            t_var[i, k, j] = row.sigma ** 2
            t_w[i, k, j] = float(row.trust)
    return {"cell_map": np.stack(maps).astype(np.int64), "water_w": water_w, "t_mu": t_mu, "t_var": t_var, "t_w": t_w}


# ------------------------------------------------------------------ teacher overlap check (protocol §5, review W8)
# The teacher must not have trained on Lite ``test``/``test_time`` water bodies. "Trained on" = any row the
# Orbit teacher saw in fitting: matchup rows of its fit cell incl. the inner early-stopping rows, and the GLORIA
# pretraining rows. A Lite site overlaps a teacher row if (a) same WQP site id, (b) <= OVERLAP_KM apart, or
# (c) same HydroLAKES lake (Orbit 500-m capture rule, or the Lite 100-m ``hylak_id``). A Lite water body
# overlaps if any of its sites does. Limitation: river reaches > 2 km from every Lite site are not matched
# (teacher rows carry no HUC8), so river overlap can be under-counted.
OVERLAP_KM = 2.0
SITE_COLS = ("naip_chip_id", "site_id", "lat", "lon", "water_body", "hylak_id", "site_class")   # no label columns


def lite_sites(split: dict, labels_file=None) -> pd.DataFrame:
    """One row per (Lite site, role) with coordinates only. Reads NO label column (``y_*``), so protected-role
    labels are never loaded; the role comes from the frozen split."""
    from .data import labels_path

    lab = pd.read_parquet(labels_file or labels_path(), columns=list(SITE_COLS))
    lab["role"] = lab["naip_chip_id"].astype(str).map(split["chip_role"])
    n_chips = lab.groupby(["site_id", "role"]).size().rename("n_chips")
    s = lab.drop_duplicates(["site_id", "role"]).drop(columns="naip_chip_id").set_index(["site_id", "role"])
    return s.join(n_chips).reset_index()


def _orbit_lake(lon, lat) -> np.ndarray:
    """Orbit water-body id (``L<hylak>`` for lakes) under the frozen Orbit rule; never writes M2's lake cache."""
    from ..orbit import split as SP

    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    lk = SP.assign_lakes(lon, lat, use_cache=False, log=lambda *a: None)
    return SP.assign(lon, lat, np.full(len(lon), np.datetime64("2015-01-01")), lakes=lk,
                     log=lambda *a: None)["waterbody"].to_numpy().astype(str)


def orbit_seen_points(fit_scheme: str = "base", version: str | None = None, *, gloria: bool = True) -> pd.DataFrame:
    """Every row the stock Orbit teacher fitted on (fit + inner rows of ``fit_scheme``; GLORIA pretraining rows):
    columns source, key, site_id, lat, lon, orbit_wb. Reads only coordinates/ids of non-protected Orbit rows."""
    from ...core import paths
    from ..orbit import data as OD
    from ..orbit import features as OF

    version = version or OD.CACHE_VERSION
    t = pd.read_parquet(paths.RUNS_ROOT / "orbit" / "cache" / version / "table.parquet",
                        columns=["row_id", "site_id", "lat", "lon", "waterbody", "role", "block", "cell",
                                 "buffer_excluded", "protected"])
    fit, inner = OF.fit_and_inner(t, fit_scheme)
    t = t[fit | inner]
    out = [pd.DataFrame({"source": "matchup", "key": t["row_id"].astype(str), "site_id": t["site_id"].astype(str),
                         "lat": t["lat"].astype(float), "lon": t["lon"].astype(float),
                         "orbit_wb": t["waterbody"].astype(str)})]
    if gloria:
        g = OD.gloria_rows(log=lambda *a: None)
        out.append(pd.DataFrame({"source": "gloria", "key": g["gloria_id"].astype(str), "site_id": "",
                                 "lat": g["lat"].astype(float), "lon": g["lon"].astype(float),
                                 "orbit_wb": g["waterbody"].astype(str)}))
    return pd.concat(out, ignore_index=True)


def flag_overlap(sites: pd.DataFrame, seen: pd.DataFrame, radius_km: float = OVERLAP_KM) -> pd.DataFrame:
    """Per Lite site: nearest teacher-row distance (km) and the three overlap criteria."""
    from ...eval.splits import nearest_distance_km

    s = sites.copy()
    pts = seen.drop_duplicates(["lat", "lon"])
    s["teacher_km"] = nearest_distance_km(s["lat"].to_numpy(float), s["lon"].to_numpy(float),
                                          pts["lat"].to_numpy(float), pts["lon"].to_numpy(float)) if len(pts) else np.inf
    seen_lakes = {w for w in seen["orbit_wb"] if w.startswith("L")}
    s["orbit_wb"] = _orbit_lake(s["lon"], s["lat"])
    lite_lake = "L" + s["hylak_id"].astype("Int64").astype(str)
    s["ov_site"] = s["site_id"].astype(str).isin(set(seen["site_id"]) - {""})
    s["ov_dist"] = s["teacher_km"] <= radius_km
    s["ov_lake"] = (s["orbit_wb"].str.startswith("L") & s["orbit_wb"].isin(seen_lakes)) \
        | (s["hylak_id"].notna() & lite_lake.isin(seen_lakes))
    s["overlap"] = s["ov_site"] | s["ov_dist"] | s["ov_lake"]
    return s


def overlap_report(flagged: pd.DataFrame) -> dict:
    """Counts per Lite role (sites, water bodies, chips) overlapping the teacher's training rows."""
    rep = {}
    for role, s in flagged.groupby("role"):
        wb = s.groupby("water_body")["overlap"].any()
        ov_wb = set(wb.index[wb])
        rep[str(role)] = {"sites": int(len(s)), "sites_overlap": int(s["overlap"].sum()),
                          "sites_same_site_id": int(s["ov_site"].sum()), "sites_within_km": int(s["ov_dist"].sum()),
                          "sites_same_lake": int(s["ov_lake"].sum()), "water_bodies": int(len(wb)),
                          "water_bodies_overlap": int(len(ov_wb)), "chips": int(s["n_chips"].sum()),
                          "chips_in_overlap_wb": int(s.loc[s["water_body"].isin(ov_wb), "n_chips"].sum())}
    return rep


def overlap_water_bodies(flagged: pd.DataFrame, roles) -> list[str]:
    s = flagged[flagged["role"].isin(list(roles))]
    wb = s.groupby("water_body")["overlap"].any()
    return sorted(map(str, wb.index[wb]))


def assert_teacher_disjoint(flagged: pd.DataFrame, roles=("test", "test_time")) -> None:
    """Protocol §5 / W8: raise if the teacher trained on any Lite water body of ``roles``."""
    from .guard import ProtectedDataError

    bad = overlap_water_bodies(flagged, roles)
    if bad:
        raise ProtectedDataError(f"W8: the teacher trained on {len(bad)} Lite {'/'.join(roles)} water bodies, "
                                 f"e.g. {bad[:5]}")


def litesafe_drop_mask(seen: pd.DataFrame, flagged_eval: pd.DataFrame, radius_km: float = OVERLAP_KM) -> np.ndarray:
    """Teacher rows to remove so that a retrained teacher is disjoint from the given Lite (eval) sites: same site
    id, <= radius_km from any eval site, or in a lake (Orbit 500-m rule or Lite hylak_id) holding an eval site."""
    from ...eval.splits import nearest_distance_km

    ev = flagged_eval
    d = nearest_distance_km(seen["lat"].to_numpy(float), seen["lon"].to_numpy(float),
                            ev["lat"].to_numpy(float), ev["lon"].to_numpy(float))
    lakes = {w for w in ev["orbit_wb"] if w.startswith("L")} \
        | {f"L{int(h)}" for h in ev["hylak_id"].dropna()}
    same_site = seen["site_id"].astype(str).isin(set(ev["site_id"].astype(str)))
    return (d <= radius_km) | seen["orbit_wb"].isin(lakes).to_numpy() | same_site.to_numpy()
