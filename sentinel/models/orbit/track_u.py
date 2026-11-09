# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Orbit harder track U — unseen hydrologic regions × unseen years (orbit_protocol.md §11.2;
RESEARCH_PLAN §15.5 amendment 2026-11-08, items 5–7).

Rule (pure function of the row's split anchor and the WBD HUC12 table; never of labels):
  * region = HUC2 of the WBD HUC12 unit whose centroid is nearest to the row's split anchor (lake pour point
    for in-lake rows, so a lake has one region; the point itself otherwise); ``"nonUS"`` if that centroid is
    > ``MAX_KM`` away (Europe/Canada rows). Nearest-centroid is an approximation of polygon membership near
    region boundaries (documented).
  * held-out regions = the ``N_HELD`` CONUS regions (HUC2 01–18) with the lowest
    u = int(sha256(SALT|huc2)[:12], 16) / 16**12.
  * U-test = in-lake, non-buffer rows in held-out regions × test years (2024–2025), any main-split role.
  * U-fit = the main fit cell (train role × ≤ 2021, minus inner blocks and buffer rows) minus every row whose
    anchor lies in a held-out region or within ``BUFFER_KM`` of a held-out region's HUC12 centroid; U-inner =
    main inner rows under the same exclusion.
  * No model selection happens on U: the configuration and comparator members selected on the main val cell
    are refitted on U-fit and scored once on U-test after the freeze.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from ...core import paths

SALT = "SENTINEL-Orbit-trackU|2026-11-08"
N_HELD = 4
MAX_KM = 30.0
BUFFER_KM = 25.0
CONUS = tuple(f"{i:02d}" for i in range(1, 19))
TRACK_FILE = paths.REPO_ROOT / "docs" / "preregistration" / "orbit_track_u.json"
R_EARTH = 6371.0088


def region_u(huc2: str) -> float:
    return int(hashlib.sha256(f"{SALT}|{huc2}".encode()).hexdigest()[:12], 16) / 16 ** 12


def held_out_regions() -> tuple[str, ...]:
    return tuple(sorted(sorted(CONUS, key=region_u)[:N_HELD]))


def _huc12():
    h = pd.read_parquet(paths.PROCESSED / "wbd_huc12" / "huc12.parquet", columns=["huc12", "huc2", "lon", "lat"])
    return h[np.isfinite(h.lon) & np.isfinite(h.lat)].reset_index(drop=True)


def assign(anchor_lon, anchor_lat) -> pd.DataFrame:
    """Per row: ``u_region`` (HUC2 or nonUS), ``u_heldout`` (in a held-out region), ``u_dist_heldout_km``
    (distance to the nearest held-out HUC12 centroid), ``u_excluded_from_fit``."""
    from sklearn.neighbors import BallTree

    h = _huc12()
    lon, lat = np.asarray(anchor_lon, float), np.asarray(anchor_lat, float)
    fin = np.isfinite(lon) & np.isfinite(lat)
    q = np.radians(np.c_[np.where(fin, lat, 0.0), np.where(fin, lon, 0.0)])
    tree = BallTree(np.radians(h[["lat", "lon"]].to_numpy()), metric="haversine")
    d, i = tree.query(q, k=1)
    d_km = d[:, 0] * R_EARTH
    region = np.where(fin & (d_km <= MAX_KM), h.huc2.to_numpy()[i[:, 0]], "nonUS").astype(object)
    held = held_out_regions()
    hh = h[h.huc2.isin(held)]
    t2 = BallTree(np.radians(hh[["lat", "lon"]].to_numpy()), metric="haversine")
    d2 = t2.query(q, k=1)[0][:, 0] * R_EARTH
    d2 = np.where(fin, d2, np.nan)
    heldout = np.isin(region, held)
    excl = heldout | (np.nan_to_num(d2, nan=np.inf) < BUFFER_KM) | ~fin
    return pd.DataFrame({"u_region": region, "u_heldout": heldout, "u_dist_heldout_km": d2,
                         "u_excluded_from_fit": excl})


def test_mask(df: pd.DataFrame, u: pd.DataFrame) -> np.ndarray:
    return (u.u_heldout.to_numpy(bool) & df.in_lake.to_numpy(bool) & ~df.buffer_excluded.to_numpy(bool)
            & (df.tblock.to_numpy() == "test"))


def rule() -> dict:
    return {"track": "U (unseen HUC2 regions x 2024-2025)", "salt": SALT, "n_held": N_HELD, "conus_regions": list(CONUS),
            "held_out_regions": list(held_out_regions()), "region_u": {r: region_u(r) for r in CONUS},
            "region_rule": f"HUC2 of nearest WBD HUC12 centroid to the split anchor (lake pour point for in-lake rows); "
                           f"'nonUS' if > {MAX_KM} km",
            "fit_exclusion": f"held-out region or within {BUFFER_KM} km of a held-out HUC12 centroid",
            "test_cell": "in-lake, non-buffer rows in held-out regions x test years (2024-2025), any main-split role",
            "selection": "none on U: main-track val-selected configuration and comparator members, refitted on U-fit",
            "huc12_table": "processed/wbd_huc12/huc12.parquet"}


def write_track_file(extra: dict | None = None) -> str:
    """Freeze the track rule (canonical JSON; refuses to overwrite) and return its sha256."""
    if TRACK_FILE.exists():
        raise FileExistsError(f"{TRACK_FILE} exists (frozen)")
    h12 = paths.PROCESSED / "wbd_huc12" / "huc12.parquet"
    code = (paths.REPO_ROOT / "sentinel" / "models" / "orbit" / "track_u.py").read_bytes()
    body = {**rule(), "huc12_sha256": hashlib.sha256(h12.read_bytes()).hexdigest(),
            "rule_code_sha256": hashlib.sha256(code).hexdigest(), **(extra or {})}
    txt = json.dumps(body, indent=1, sort_keys=True) + "\n"
    TRACK_FILE.write_text(txt)
    return hashlib.sha256(txt.encode()).hexdigest()


def verify_track_file() -> dict:
    body = json.loads(TRACK_FILE.read_text())
    code = (paths.REPO_ROOT / "sentinel" / "models" / "orbit" / "track_u.py").read_bytes()
    if hashlib.sha256(code).hexdigest() != body["rule_code_sha256"]:
        raise RuntimeError("track_u.py differs from the frozen track U rule")
    if list(held_out_regions()) != body["held_out_regions"]:
        raise RuntimeError("held-out regions differ from the frozen track file")
    return {"path": str(TRACK_FILE), "sha256": hashlib.sha256(TRACK_FILE.read_bytes()).hexdigest()}
