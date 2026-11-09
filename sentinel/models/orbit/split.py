# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Frozen split RULE for SENTINEL-Orbit, version 3 (docs/preregistration/orbit_protocol.md §2.2).

The split is a deterministic function of (longitude, latitude, sample date) and HydroLAKES: rows that
do not exist yet (e.g. national 2024-2025 WQP matchups built later) receive their role from the same
frozen rule — nothing depends on labels or on which rows happen to be present.

v2 superseded v1 (``orbit_split.json``) and v3 supersedes v2 (``orbit_split_v2.json``), both BEFORE any model
selection, after the pre-training review (C1: non-lake estuary/river points were split across roles at 50-km
blocks; N1: the 500-m capture radius pulled estuary stations into tiny lake polygons of another role).

1. **Lake water bodies.** A point inside or within 500 m of a HydroLAKES polygon belongs to that lake
   (``"L<Hylak_id>"``), anchored at the lake's pour point; role block = ``blocks(anchor, 50 km)``,
   hashed with ``SALT``. All stations of a lake share one role (Lake Erie is one unit).
2. **Non-lake points** (rivers, estuaries, ponds < 10 ha, coast): role block = ``blocks(point, 200 km)``
   hashed with ``SALT`` and prefix ``N`` (coarse, so estuaries/river reaches mostly stay together);
   water-body unit ``"P<200-km block>|<10-km cell>"`` (nested in its role block).
3. **Buffer.** ``buffer_excluded`` (never fitted, never scored): (a) a non-lake point within 1 km of a
   HydroLAKES lake of a different role; (b) v3: a lake point OUTSIDE its polygon (captured within 500 m)
   whose 200-km non-lake block has a different role than the lake.
4. **Role** from the salted SHA-256: u < 0.20 → ``test``, u < 0.40 → ``val``, else ``train``.
5. **Time block** from the in-situ date: ≤ 2021 ``train``; 2022–2023 ``val``; 2024–2025 ``test``;
   ≥ 2026 ``test_b``.
6. **Cells.** fit = train role × train years (in-lake and non-lake, minus buffer and inner blocks);
   **headline = in-lake rows of test role × test years**; non-lake rows are a secondary track.
   Protected (guard.py): any test-role row and any test/test_b-year row.

``spatial_blocks`` is vendored here (identical to ``sentinel.eval.splits.spatial_blocks`` at freeze time)
so later edits of shared code cannot move roles; :func:`verify_split_file` refuses to run if this
module's rule code hash differs from the frozen file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...core import paths
from ...eval.splits import nearest_distance_km

EARTH_RADIUS_KM = 6371.0088
SPLIT_VERSION = "orbit-v3"
SALT = "SENTINEL-Orbit-v1|2026-10-24"       # unchanged from v1: lake-block roles identical to v1
BLOCK_KM = 50.0
NONLAKE_BLOCK_KM = 200.0
CLUSTER_KM = 10.0
LAKE_MAX_DIST_M = 500.0
BUFFER_M = 1000.0
ROLE_CUTS = (("test", 0.20), ("val", 0.40), ("train", 1.0))
YEAR_BLOCKS = (("train", None, 2021), ("val", 2022, 2023), ("test", 2024, 2025), ("test_b", 2026, None))
SPLIT_FILE = paths.REPO_ROOT / "docs" / "preregistration" / "orbit_split_v3.json"
SPLIT_FILE_V1 = paths.REPO_ROOT / "docs" / "preregistration" / "orbit_split.json"      # superseded
SPLIT_FILE_V2 = paths.REPO_ROOT / "docs" / "preregistration" / "orbit_split_v2.json"   # superseded
LAKE_CACHE = "lake_assign_v3_{hl}.parquet"      # {hl} = HydroLAKES file sha256[:12] (review C3 note)
RULE_CODE = ("sentinel/models/orbit/split.py", "sentinel/data/sources/hydrolakes.py")


def rule_spec() -> dict:
    return {"version": SPLIT_VERSION, "salt": SALT, "block_km": BLOCK_KM, "nonlake_block_km": NONLAKE_BLOCK_KM,
            "cluster_km": CLUSTER_KM, "lake_max_dist_m": LAKE_MAX_DIST_M, "buffer_m": BUFFER_M,
            "role_cuts": [list(x) for x in ROLE_CUTS], "year_blocks": [list(x) for x in YEAR_BLOCKS],
            "waterbody_source": "HydroLAKES v1.0 polygons (manifest 'hydrolakes'), North America + Europe",
            "anchor": "lake pour point (Pour_long/Pour_lat) for lake points, else the point itself",
            "block_function": "split.spatial_blocks (vendored copy of sentinel.eval.splits.spatial_blocks)",
            "role_hash": "u = int(sha256(salt + '|' + block).hexdigest()[:12], 16) / 16**12; non-lake blocks "
                         "are prefixed 'N'",
            "headline": "in-lake rows only (test role x test years)",
            "buffer_v3": "also buffer_excluded: lake rows outside their polygon (lake_dist_m > 0) whose 200-km "
                         "non-lake block role differs from the lake role",
            "supersedes": "orbit_split_v2.json (sha256 06e5067d...b70e) and orbit_split.json (v1, 27854499...6437), "
                          "before any model selection"}


def spatial_blocks(lat, lon, block_km: float) -> np.ndarray:
    """Frozen copy of sentinel.eval.splits.spatial_blocks (2026-10-24)."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    row = np.floor((lat + 90.0) * (np.pi * EARTH_RADIUS_KM / 180.0) / block_km).astype(int)
    row_lat = np.radians((row + 0.5) * block_km / (np.pi * EARTH_RADIUS_KM / 180.0) - 90.0)
    km_per_deg_lon = np.pi * EARTH_RADIUS_KM / 180.0 * np.maximum(np.cos(row_lat), 1e-3)
    col = np.floor((lon + 180.0) * km_per_deg_lon / block_km).astype(int)
    return np.array([f"{r}_{c}" for r, c in zip(row, col)])


def block_u(block: str) -> float:
    return int(hashlib.sha256(f"{SALT}|{block}".encode()).hexdigest()[:12], 16) / 16 ** 12


def role_of_block(block: str) -> str:
    u = block_u(block)
    for role, cut in ROLE_CUTS:
        if u < cut:
            return role
    return "train"


def time_block(dates) -> np.ndarray:
    yr = pd.to_datetime(pd.Series(dates)).dt.year.to_numpy()
    out = np.full(yr.shape, "", dtype=object)
    for name, lo, hi in YEAR_BLOCKS:
        m = np.ones(yr.shape, bool)
        if lo is not None:
            m &= yr >= lo
        if hi is not None:
            m &= yr <= hi
        out[m & (out == "")] = name
    return out


def point_key(lon, lat) -> np.ndarray:
    return np.array([f"{y:.5f}_{x:.5f}" for x, y in zip(np.asarray(lon, float), np.asarray(lat, float))])


def _lake_cache_path() -> Path:
    import json as _json

    man = _json.loads((paths.MANIFESTS / "hydrolakes.json").read_text())
    sha = [f["sha256"] for f in man["files"] if f["path"].endswith("hydrolakes_na_eu.parquet")][0]
    return paths.ensure(paths.RUNS_ROOT / "orbit" / "cache") / LAKE_CACHE.format(hl=sha[:12])


_LAKES = None


def _load_lakes():
    global _LAKES
    if _LAKES is None:
        import geopandas as gpd

        _LAKES = gpd.read_parquet(paths.processed("hydrolakes") / "hydrolakes_na_eu.parquet",
                                  columns=["Hylak_id", "Lake_name", "Lake_area", "Pour_long", "Pour_lat",
                                           "region", "geometry"])
    return _LAKES


def assign_lakes(lon, lat, *, use_cache: bool = True, log=print) -> pd.DataFrame:
    """Per point: lake within LAKE_MAX_DIST_M (``hylak_id`` ...) and nearest lake within BUFFER_M
    (``near_hylak_id``, ``near_pour_lon/lat``). Cached by 1e-5° point key (pure geometry)."""
    from ...data.sources import hydrolakes as H

    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    keys = point_key(lon, lat)
    cp = _lake_cache_path()
    cache = pd.read_parquet(cp) if (use_cache and cp.exists()) else pd.DataFrame(columns=["point_key"])
    uk, first = np.unique(keys, return_index=True)
    todo = np.setdiff1d(uk, cache["point_key"].to_numpy())
    if todo.size:
        sel = first[np.searchsorted(uk, todo)]
        lakes = _load_lakes()
        log(f"[split] HydroLAKES assignment for {todo.size} new points")
        a = H.assign_points(lon[sel], lat[sel], lakes, max_dist_m=LAKE_MAX_DIST_M)
        b = H.assign_points(lon[sel], lat[sel], lakes, max_dist_m=BUFFER_M)
        a["near_hylak_id"] = b["hylak_id"].to_numpy()
        a["near_dist_m"] = b["lake_dist_m"].to_numpy()
        a["near_pour_lon"] = b["lake_pour_lon"].to_numpy()
        a["near_pour_lat"] = b["lake_pour_lat"].to_numpy()
        a.insert(0, "point_key", todo)
        cache = pd.concat([cache, a], ignore_index=True) if len(cache) else a
        if use_cache:
            cache.to_parquet(cp, index=False)
    return cache.set_index("point_key").loc[keys].reset_index()


def assign(lon, lat, dates, *, lakes: pd.DataFrame | None = None, log=print) -> pd.DataFrame:
    """Split columns per row: waterbody, in_lake, block (role unit), role, tblock, cell, buffer_excluded,
    protected. ``lakes`` = precomputed :func:`assign_lakes` output (else computed)."""
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    lk = assign_lakes(lon, lat, log=log) if lakes is None else lakes
    finite = np.isfinite(lon) & np.isfinite(lat)
    in_lake = lk["hylak_id"].notna().to_numpy() & finite
    a_lon = np.where(in_lake, lk["lake_pour_lon"].to_numpy(float), lon)
    a_lat = np.where(in_lake, lk["lake_pour_lat"].to_numpy(float), lat)
    lake_block = spatial_blocks(a_lat, a_lon, BLOCK_KM)
    nl_block = np.char.add("N", spatial_blocks(lat, lon, NONLAKE_BLOCK_KM).astype(str))
    block = np.where(in_lake, lake_block, nl_block)
    cell = spatial_blocks(lat, lon, CLUSTER_KM)
    hid = lk["hylak_id"].astype("Int64").astype(str).to_numpy().astype(str)
    p_name = np.char.add(np.char.add(np.char.add("P", block.astype(str)), "|"), cell.astype(str))
    waterbody = np.where(in_lake, np.char.add("L", hid), p_name)
    roles = {b: role_of_block(b) for b in np.unique(block)}
    role = np.array([roles[b] for b in block], dtype=object)
    # buffer: non-lake point within BUFFER_M of a lake whose role differs from the point's role
    near = lk["near_hylak_id"].notna().to_numpy() & ~in_lake & finite
    near_block = spatial_blocks(lk["near_pour_lat"].to_numpy(float), lk["near_pour_lon"].to_numpy(float), BLOCK_KM)
    near_role = np.array([role_of_block(b) if n else "" for b, n in zip(near_block, near)], dtype=object)
    nl_role = np.array([role_of_block(b) for b in nl_block], dtype=object)
    outside = in_lake & (np.nan_to_num(lk["lake_dist_m"].to_numpy(float), nan=0.0) > 0)
    buffer_excluded = (near & (near_role != role)) | (outside & (nl_role != role))
    tb = time_block(dates)
    out = pd.DataFrame({"waterbody": waterbody, "in_lake": in_lake, "hylak_id": lk["hylak_id"].to_numpy(),
                        "lake_area_km2": lk["lake_area_km2"].to_numpy(), "lake_dist_m": lk["lake_dist_m"].to_numpy(),
                        "anchor_lon": a_lon, "anchor_lat": a_lat, "block": block, "role": role, "tblock": tb,
                        "buffer_excluded": buffer_excluded, "near_lake_role": near_role})
    out["cell"] = out["role"] + "x" + out["tblock"]
    out["protected"] = (out["role"] == "test") | out["tblock"].isin(["test", "test_b"]) | ~finite
    return out


def rule_code_sha256() -> str:
    h = hashlib.sha256()
    for rel in RULE_CODE:
        h.update(rel.encode())
        h.update((paths.REPO_ROOT / rel).read_bytes())
    return h.hexdigest()


def write_split_file(path: Path = SPLIT_FILE, extra: dict | None = None) -> str:
    """Write the frozen rule file (canonical JSON) and return its sha256. Refuses to overwrite."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists: the orbit split is frozen; a new version needs a new file")
    spec = {"rule": rule_spec(), "rule_code_sha256": rule_code_sha256(), **(extra or {})}
    path.write_text(json.dumps(spec, indent=1, sort_keys=True, default=str) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_sha256(path: Path = SPLIT_FILE) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_split_file(path: Path = SPLIT_FILE) -> dict:
    """The frozen file must exist, match the rule constants AND the rule code hash of the running code."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: build it with scripts/train/orbit_build_split.py --freeze-rule")
    spec = json.loads(path.read_text())
    if spec["rule"] != json.loads(json.dumps(rule_spec())):
        raise RuntimeError("orbit split rule constants in code differ from the frozen split file")
    now = rule_code_sha256()
    if spec["rule_code_sha256"] != now:
        raise RuntimeError(f"orbit split rule code changed since the freeze ({now[:12]} != "
                           f"{spec['rule_code_sha256'][:12]}); a new split version is required")
    return {"path": str(path), "sha256": split_sha256(path), "rule_code_sha256": now}


def distance_report(df: pd.DataFrame, lat="lat", lon="lon") -> dict:
    """Nearest train-role point distances for val/test-role points (km), all rows and in-lake rows."""
    rep = {}
    for tag, sub in (("all", df), ("in_lake", df[df.in_lake])):
        tr = df[(df.role == "train") & ~df.buffer_excluded].drop_duplicates([lat, lon])
        for r in ("val", "test"):
            q = sub[(sub.role == r) & ~sub.buffer_excluded].drop_duplicates([lat, lon])
            if len(q) and len(tr):
                d = nearest_distance_km(q[lat], q[lon], tr[lat], tr[lon])
                rep[f"{tag}_{r}_to_train_km"] = {"min": float(d.min()), "p05": float(np.quantile(d, .05)),
                                                 "median": float(np.median(d)), "n_points": int(len(q)),
                                                 "frac_within_10km": float(np.mean(d < 10))}
    return rep
