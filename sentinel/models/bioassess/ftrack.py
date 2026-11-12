# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M5-F (France block) rules: folds, dates, lock list, allowed rows, fit-id guard, guarded checkpoint loads.

Protocol: ``docs/preregistration/bioassess_f_protocol.md`` (§2.3 lock-on-download, §3 pools, §4 folds).
This module holds constants and guards, plus the only code that touches the locked store: the fetcher
(``LockedFetcher``), the extractor (``extract_locked``) and the counting reader (``locked_availability``).
It never parses a label value. The frozen M5 / M5-X modules are imported, never edited.

Keys. Every row has a key ``"<source>:<id>"``:

* ``uk:<analysis_id>``   UK EA invertebrate analysis (the M5-X key),
* ``ukd:<analysis_id>``  UK EA diatom analysis used as a training row of its own,
* ``fr:<station>:<YYYY-MM-DD>:<support>`` French biology operation (upper-case ``FR:`` in the lock file).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.core import paths

PROTOCOL_FILE = "docs/preregistration/bioassess_f_protocol.md"
LOCK_FILE = "docs/preregistration/bioassess_f_lock.json"
SPLIT_FILE = "docs/preregistration/bioassess_f_split.json"

# ------------------------------------------------------------------------------------------------ folds (§4)
FOLDS = {
    "G1": ("02", "08", "54", "55", "57", "59", "62", "67", "68", "80", "88"),
    "G2": ("10", "51", "52", "60", "75", "77", "78", "89", "91", "92", "93", "94", "95"),
    "G3": ("14", "22", "27", "29", "35", "50", "56", "61", "76"),
    "G4": ("18", "28", "36", "37", "41", "44", "45", "49", "53", "58", "72", "79", "85", "86"),
    "G5": ("03", "15", "19", "23", "42", "43", "48", "63", "87"),
    "G6": ("09", "12", "16", "17", "24", "31", "32", "33", "40", "46", "47", "64", "65", "81", "82"),
    "G7": ("01", "21", "25", "38", "39", "69", "70", "71", "73", "74", "90"),
    "G8": ("04", "05", "06", "07", "11", "13", "26", "2A", "2B", "30", "34", "66", "83", "84"),
}
FOLD_NAMES = {
    "G1": "Nord and Nord-Est", "G2": "Seine amont and Ile-de-France", "G3": "Normandie and Bretagne",
    "G4": "Loire moyenne et aval, Centre-Ouest", "G5": "Massif central", "G6": "Adour-Garonne",
    "G7": "Rhone, Saone, Alpes du Nord, Jura", "G8": "Mediterranee and Corse",
}
MERGE_PARTNER = {"G1": "G2", "G2": "G1", "G3": "G4", "G4": "G3", "G5": "G6", "G6": "G5", "G7": "G8", "G8": "G7"}
DEPT_TO_FOLD = {d: g for g, ds in FOLDS.items() for d in ds}
METROPOLITAN = frozenset(DEPT_TO_FOLD)

# ------------------------------------------------------------------------------------------------ dates (§2.3, §3)
LOCK_FROM = dt.date(2022, 1, 1)            # every French chemistry row from this date on is locked
TEST_FROM, TEST_TO = dt.date(2023, 1, 1), dt.date(2025, 12, 31)
TRAIN_TO = dt.date(2021, 12, 31)           # test design: French training operations and their label windows
DEV_TEST_FROM, DEV_TEST_TO = dt.date(2019, 1, 1), dt.date(2021, 12, 31)
DEV_TRAIN_TO = dt.date(2017, 12, 31)       # 2018 is the development buffer year
UK_ALLOWED_TO = dt.date(2020, 12, 1)      # UK biological sample dates (lead ruling (b)); wq_observations unchanged

BUFFER_KM = 1.0
MIN_TEST_CLUSTERS = 650                    # power gate per target (§10)
MERGE_MIN_CLUSTERS = 100                   # fold merge rule (§4)
NEW_DATA_CAP_GB = 5.0
DISK_FLOOR_GB = 60.0

TRACKS = ("F-A", "F-Z", "F-T")
SECONDARY_TRACKS = ("F-E", "F-MA")
EXPLORATORY_TRACKS = ("F-TOX",)
DESIGNS = ("test", "dev")
CONFIG_CAP = 8
CONFIGS = ("ma_inv", "ma_full", "ma_full_rc", "ma_full_coh", "pfn_hybrid", "ens_f", "ma_zs", "ens_zs")
SEEDS = (0, 1, 2)
HOLM_FAMILY = 36                           # 3 tracks x 6 targets x 2 co-primary metrics

# ------------------------------------------------------------------------------------------------ context (§9.1)
CONTEXT_COLS = ("doy_sin", "doy_cos", "lat", "lon", "altitude_m")
CONTEXT_COLS_NO_COORDS = ("doy_sin", "doy_cos", "altitude_m")
FZ_INPUT_SETS = {"coords": CONTEXT_COLS, "nocoords": CONTEXT_COLS_NO_COORDS}


def context_cols(track: str, role: str = "candidate", input_set: str | None = None) -> tuple:
    """Numeric context columns of a model (closed list of protocol §9.1; lead ruling (a)).

    F-A / F-T and the secondary tracks: day of year, latitude, longitude, altitude, for every model.
    F-Z candidates: day of year and altitude only (no latitude or longitude).
    F-Z comparators: ``input_set`` must be given, "coords" or "nocoords". Both are named comparators (§8);
    neither is a development-chosen variant, so there is no default."""
    if role not in ("candidate", "comparator"):
        raise ValueError(role)
    if track != "F-Z":
        if input_set not in (None, "coords"):
            raise ValueError(f"{track} has one context input set; got {input_set!r}")
        return CONTEXT_COLS
    if role == "candidate":
        if input_set not in (None, "nocoords"):
            raise FitGuardError("F-Z candidates take no latitude or longitude (lead ruling (a))")
        return CONTEXT_COLS_NO_COORDS
    if input_set not in FZ_INPUT_SETS:
        raise ValueError("an F-Z comparator names its input set: 'coords' or 'nocoords'")
    return FZ_INPUT_SETS[input_set]


def fz_comparator_names(base: str) -> tuple:
    """The two named F-Z comparators of an ML baseline or baseline ensemble that takes context (§8)."""
    return tuple(f"{base}__{k}" for k in FZ_INPUT_SETS)


# ------------------------------------------------------------------------------------------------ mirror rule (§9.1)
MIRROR_SEED = 20261110
REGION_CONTEXT_MAX = 256


def mirror_pick(keys, unit_index: int, seed: int, cap: int = REGION_CONTEXT_MAX) -> np.ndarray:
    """Positions (into ``keys``) of the region-context rows of one region unit.

    At most ``cap`` candidates: all of them. More: the keys are sorted in byte order and ``cap`` of them are drawn
    without replacement with ``numpy.random.default_rng([20261110, unit_index, seed])``. Every row of the unit
    uses this one sample. The result is sorted by key and does not depend on the order of ``keys``."""
    ks = [str(k).encode("utf-8") for k in keys]
    if len(set(ks)) != len(ks):
        raise ValueError("region-context candidate keys must be unique")
    order = np.asarray(sorted(range(len(ks)), key=ks.__getitem__), np.int64)
    if len(ks) <= cap:
        return order
    rng = np.random.default_rng([MIRROR_SEED, int(unit_index), int(seed)])
    return order[np.sort(rng.choice(len(ks), int(cap), replace=False))]


class FitGuardError(RuntimeError):
    """A fit, a scaler or a checkpoint load asked for a row that the M5-F protocol protects."""


def locked_dir() -> Path:
    """The locked store (§2.3 item 1): every raw French API response lands here first, and chemistry rows dated
    >= LOCK_FROM never leave it. Only this module names or opens it before the single look (static test): the
    fetcher writes it, ``extract_locked`` copies whitelisted columns out of it by date, and
    ``locked_availability`` returns counts from it."""
    return paths.RAW / "naiades" / "locked"


# ------------------------------------------------------------------------------------------------ departments
def department_code(dept=None, commune=None) -> str | None:
    """Normalised metropolitan department code, or None (overseas / unknown).

    ``dept``: the station's department code; ``commune``: its commune INSEE code (fallback, first two characters).
    The legacy Corsican code "20" maps to "2A" (both Corsican departments are in fold G8).
    """
    for raw in (dept, None if commune is None else str(commune).strip()[:2]):
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            continue
        c = str(raw).strip().upper()
        if c.endswith(".0"):
            c = c[:-2]
        if c.isdigit():
            c = c.zfill(2)
        if c == "20":
            c = "2A"
        if c in METROPOLITAN:
            return c
        if c.startswith("97") or c.startswith("98"):
            return None
    return None


def fold_of(dept=None, commune=None) -> str | None:
    c = department_code(dept, commune)
    return None if c is None else DEPT_TO_FOLD[c]


def validate_folds() -> None:
    codes = [d for ds in FOLDS.values() for d in ds]
    expected = {f"{i:02d}" for i in range(1, 96) if i != 20} | {"2A", "2B"}
    if len(codes) != 96 or set(codes) != expected:
        raise AssertionError("fold table must hold each of the 96 metropolitan departments exactly once")


# ------------------------------------------------------------------------------------------------ lock list (§2.3)
def _as_date(x) -> dt.date:
    if isinstance(x, dt.datetime):
        return x.date()
    if isinstance(x, dt.date):
        return x
    return pd.Timestamp(x).date()


def lock_role(date) -> str | None:
    """Role of a French biology operation by its date: test (2023-25), buffer (2022), future, or None (unlocked)."""
    d = _as_date(date)
    if d < LOCK_FROM:
        return None
    if d < TEST_FROM:
        return "buffer"
    return "test" if d <= TEST_TO else "future"


def lock_key(station, date, support) -> str:
    return f"FR:{str(station).strip()}:{_as_date(date).isoformat()}:{str(support).strip()}"


def row_key(station, date, support) -> str:
    """Design key of a French operation (lower-case source prefix, same fields as the lock key)."""
    return "fr" + lock_key(station, date, support)[2:]


def build_lock(ops: pd.DataFrame) -> dict:
    """Lock document from operation METADATA only (columns ``station``, ``date``, ``support`` [, ``op_id``]).

    One key per operation dated >= LOCK_FROM. The caller must not have requested any chemistry file yet.
    """
    need = {"station", "date", "support"}
    if not need <= set(ops.columns):
        raise ValueError(f"operations table needs columns {sorted(need)}")
    d = pd.to_datetime(ops["date"]).dt.date
    keep = (d >= LOCK_FROM).to_numpy()
    o = ops.loc[keep]
    d = d[keep]
    keys = [lock_key(s, x, p) for s, x, p in zip(o["station"], d, o["support"])]
    roles = [lock_role(x) for x in d]
    rows = {}
    op = o["op_id"].astype(str).to_numpy() if "op_id" in o else [None] * len(o)
    for k, r, i in zip(keys, roles, op):
        e = rows.setdefault(k, {"role": r, "op_ids": []})
        if i is not None and i not in e["op_ids"]:
            e["op_ids"].append(i)
    skeys = sorted(rows)
    yr = pd.Series([int(k.split(":")[2][:4]) for k in skeys], dtype="int64")
    sup = pd.Series([k.split(":")[3] for k in skeys], dtype="object")
    role = pd.Series([rows[k]["role"] for k in skeys], dtype="object")
    return {
        "protocol": PROTOCOL_FILE, "lock_from": LOCK_FROM.isoformat(),
        "test": [TEST_FROM.isoformat(), TEST_TO.isoformat()],
        "n_keys": len(skeys),
        "counts": {"role": role.value_counts().sort_index().to_dict(),
                   "year": {str(k): int(v) for k, v in yr.value_counts().sort_index().items()},
                   "support": sup.value_counts().sort_index().to_dict()},
        "keys_sha256": hashlib.sha256("\n".join(skeys).encode()).hexdigest(),
        "keys": {k: rows[k] for k in skeys},
    }


def write_lock(ops: pd.DataFrame, out: Path | None = None) -> tuple[Path, str]:
    """Write the lock file once (refuses to overwrite), make it read-only, return (path, sha256)."""
    from sentinel.core.registry import file_sha256

    out = Path(out) if out is not None else paths.REPO_ROOT / LOCK_FILE
    if out.exists():
        raise FitGuardError(f"{out} already exists; the lock list is written once")
    doc = build_lock(ops)
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    os.chmod(out, 0o444)
    return out, file_sha256(out)


def read_lock(path: Path | None = None) -> dict[str, str]:
    """{lower-case design key -> role} from the lock file (verifies its key hash)."""
    path = Path(path) if path is not None else paths.REPO_ROOT / LOCK_FILE
    doc = json.loads(path.read_text())
    skeys = sorted(doc["keys"])
    if hashlib.sha256("\n".join(skeys).encode()).hexdigest() != doc["keys_sha256"]:
        raise FitGuardError("lock file keys do not match their recorded hash")
    return {"fr" + k[2:]: v["role"] for k, v in doc["keys"].items()}


def chemistry_request_allowed(lock_path: Path | None = None) -> bool:
    """Gate G2: no French physico-chemistry request before the lock file exists and is read-only."""
    p = Path(lock_path) if lock_path is not None else paths.REPO_ROOT / LOCK_FILE
    return p.exists() and not os.access(p, os.W_OK)


def window_counts(ops: pd.DataFrame, obs: pd.DataFrame, windows: dict | None = None) -> pd.DataFrame:
    """Availability counts only (§2.3 item 4): number of analyses per (operation key, parameter, window).

    ``ops``: columns ``key``, ``station``, ``date``. ``obs``: columns ``station``, ``date``, ``parameter`` ONLY.
    Passing any other column (in particular a result value) is refused, so this function cannot leak a label.
    """
    if set(obs.columns) != {"station", "date", "parameter"}:
        raise FitGuardError("window_counts takes station, date and parameter columns only (no result values)")
    windows = windows or {"pm30d": (-30, 30), "prior365d": (-365, 0)}
    o = ops[["key", "station", "date"]].copy()
    o["date"] = pd.to_datetime(o["date"])
    b = obs.copy()
    b["date"] = pd.to_datetime(b["date"])
    m = o.merge(b, on="station", suffixes=("", "_obs"))
    lag = (m["date_obs"] - m["date"]).dt.days
    out = []
    for w, (lo, hi) in windows.items():
        k = m[(lag >= lo) & (lag <= hi)].groupby(["key", "parameter"]).size().rename("n_obs").reset_index()
        k["window"] = w
        out.append(k)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["key", "parameter", "n_obs", "window"])


# ------------------------------------------------------------------------------------------------ UK allowed rows (§3)
def uk_allowed() -> dict:
    """Allowed UK rows (lead decision 2026-11-10). Reads sample and site METADATA only.

    Invertebrates: role pool_fit / pool_val, sample date <= 2020-12-01, not a frozen test key of any workstream.
    Diatoms: light-microscopy analyses dated <= 2020-12-01 whose site is not a reserved_test site, has no
    reserved_test invertebrate site within 1 km, whose paired WQ point is not paired to a reserved_test site, and
    (stricter than the protocol) whose 50-km block holds no reserved_test site.
    """
    from sklearn.neighbors import BallTree

    from sentinel.eval import splits as ES
    from sentinel.models.bioassess import uk as UK
    from sentinel.models.bioassess import xdev as XD

    P = paths.processed("uk_ea_ecology")
    sites = pd.read_parquet(P / "sites.parquet",
                            columns=["site_id", "assemblage", "waterbody_type", "lat", "lon"])
    sites = sites.assign(site_id=sites["site_id"].astype(str))
    inv_sites = sites[(sites["assemblage"] == "invertebrate") & (sites["waterbody_type"] != "WBES")]
    roles = UK.uk_site_roles(inv_sites)
    reserved = set(roles.index[roles == "reserved_test"])
    frozen = XD.frozen_keys_cached()     # cached key list of xtrack.frozen_test_keys(); keys only, no labels
    frozen_uk = {i for (s, i) in frozen if s == "uk"}

    smp = pd.read_parquet(P / "samples_invertebrate.parquet", columns=["site_id", "analysis_id", "sample_date"])
    smp = smp.assign(site_id=smp["site_id"].astype(str), analysis_id=smp["analysis_id"].astype(str))
    role = smp["site_id"].map(roles)
    d = pd.to_datetime(smp["sample_date"]).dt.date
    ok = role.isin(["pool_fit", "pool_val"]) & (d <= UK_ALLOWED_TO) & ~smp["analysis_id"].isin(frozen_uk)
    inv_ids = set(smp.loc[ok, "analysis_id"])

    res_xy = inv_sites[inv_sites["site_id"].isin(reserved)].dropna(subset=["lat", "lon"])
    res_blocks = set(ES.spatial_blocks(res_xy["lat"].to_numpy(), res_xy["lon"].to_numpy(), UK.RESERVED["block_km"]))
    pairs = pd.read_parquet(P / "site_wq_pairs.parquet", columns=["site_id", "wq_point", "assemblage"])
    pairs = pairs.assign(site_id=pairs["site_id"].astype(str))
    res_wq = set(pairs.loc[pairs["site_id"].isin(reserved), "wq_point"].dropna())
    dsites = sites[(sites["assemblage"] == "diatom") & (sites["waterbody_type"] != "WBES")].dropna(
        subset=["lat", "lon"]).copy()
    tree = BallTree(np.radians(res_xy[["lat", "lon"]].to_numpy(float)), metric="haversine")
    dist, _ = tree.query(np.radians(dsites[["lat", "lon"]].to_numpy(float)), k=1)
    dsites["near_reserved"] = dist[:, 0] * ES.EARTH_RADIUS_KM <= BUFFER_KM
    dsites["block"] = ES.spatial_blocks(dsites["lat"].to_numpy(), dsites["lon"].to_numpy(), UK.RESERVED["block_km"])
    dwq = pairs[pairs["assemblage"] == "diatom"].groupby("site_id")["wq_point"].agg(set)
    bad_wq = dsites["site_id"].map(lambda s: bool(dwq.get(s, set()) & res_wq))
    site_ok = ~(dsites["site_id"].isin(reserved) | dsites["near_reserved"] | dsites["block"].isin(res_blocks)
                | bad_wq.to_numpy())
    ok_sites = set(dsites.loc[site_ok, "site_id"])
    ds = pd.read_parquet(P / "samples_diatom.parquet",
                         columns=["site_id", "analysis_id", "sample_date", "analysis_type"])
    ds = ds.assign(site_id=ds["site_id"].astype(str), analysis_id=ds["analysis_id"].astype(str))
    dd = pd.to_datetime(ds["sample_date"]).dt.date
    dok = ds["site_id"].isin(ok_sites) & (dd <= UK_ALLOWED_TO) & (ds["analysis_type"] == "LABP")
    return {"invertebrate": inv_ids, "diatom": set(ds.loc[dok, "analysis_id"]),
            "counts": {"invertebrate": len(inv_ids), "diatom": int(dok.sum()), "diatom_sites": len(ok_sites),
                       "reserved_sites": len(reserved), "reserved_blocks": len(res_blocks)}}


# ------------------------------------------------------------------------------------------------ pools (§3)
SPLIT_COLUMNS = ("source", "date", "lat", "lon", "fold", "lock_role", "uk_allowed", "has_inv")


def _check_split(S: pd.DataFrame) -> None:
    miss = set(SPLIT_COLUMNS) - set(S.columns)
    if miss:
        raise ValueError(f"split table lacks columns {sorted(miss)}")


def _dates(S: pd.DataFrame) -> np.ndarray:
    return pd.to_datetime(S["date"]).dt.date.to_numpy()


def test_ids(S: pd.DataFrame, g: str, design: str = "dev") -> pd.Index:
    """Scored rows of fold ``g``: lock-role ``test`` operations (test design) or 2019-21 operations (dev design).
    Only rows with an invertebrate list are scored units."""
    _check_split(S)
    if design not in DESIGNS:
        raise ValueError(design)
    fr = (S["source"] == "fr").to_numpy() & (S["fold"] == g).to_numpy() & S["has_inv"].to_numpy(bool)
    if design == "test":
        return S.index[fr & (S["lock_role"] == "test").to_numpy()]
    d = _dates(S)
    return S.index[fr & (d >= DEV_TEST_FROM) & (d <= DEV_TEST_TO) & S["lock_role"].isna().to_numpy()]


def training_ids(S: pd.DataFrame, track: str, g: str | None, design: str = "dev",
                 buffer_km: float = BUFFER_KM) -> pd.Index:
    """Training pool of (track, fold, design). F-Z: allowed UK rows only (no French row). F-A / F-T and the
    secondary tracks: French rows up to the design's end date outside fold ``g`` and farther than ``buffer_km``
    from every scored station of ``g``, plus allowed UK rows. Locked rows are never returned."""
    _check_split(S)
    if design not in DESIGNS:
        raise ValueError(design)
    uk_ok = S["source"].isin(["uk", "ukd"]).to_numpy() & S["uk_allowed"].fillna(False).to_numpy(bool)
    if track == "F-Z":
        return S.index[uk_ok]
    if g is None:
        raise ValueError("fold needed for tracks that train on French rows")
    d = _dates(S)
    end = TRAIN_TO if design == "test" else DEV_TRAIN_TO
    fr = (S["source"] == "fr").to_numpy() & S["fold"].notna().to_numpy() & (S["fold"] != g).to_numpy() \
        & (d <= end) & S["lock_role"].isna().to_numpy()
    te = test_ids(S, g, design)
    if len(te) and fr.any():
        from sklearn.neighbors import BallTree

        from sentinel.eval.splits import EARTH_RADIUS_KM

        q = S.loc[te, ["lat", "lon"]].drop_duplicates().to_numpy(float)
        tree = BallTree(np.radians(q), metric="haversine")
        idx = np.flatnonzero(fr)
        dist, _ = tree.query(np.radians(S.iloc[idx][["lat", "lon"]].to_numpy(float)), k=1)
        fr[idx[dist[:, 0] * EARTH_RADIUS_KM <= buffer_km]] = False
    return S.index[fr | uk_ok]


def assert_fit_ids(ids, S: pd.DataFrame, track: str, g: str | None, design: str = "dev") -> None:
    """THE M5-F fit guard (§3): every id used to fit, scale, early-stop or calibrate must lie in the training pool of
    (track, fold, design). Locked, buffer, pseudo-test, disallowed UK and frozen-test rows are all outside it."""
    bad = pd.Index(ids).difference(training_ids(S, track, g, design))
    if len(bad):
        raise FitGuardError(f"refusing to fit on {len(bad)} rows outside the {track}/{g}/{design} training pool "
                            f"(e.g. {list(bad[:3])})")


# ------------------------------------------------------------------------------------------------ checkpoints
SIDECAR_SUFFIX = ".f_train_ids.json"


def write_train_ids_sidecar(checkpoint, train_ids, *, track: str, g, design: str) -> Path:
    """Written by the code that trained ``checkpoint`` from the ids it actually used."""
    side = Path(str(checkpoint) + SIDECAR_SUFFIX)
    side.write_text(json.dumps({"protocol": "M5-F", "track": track, "fold": g, "design": design,
                                "train_ids": sorted(map(str, train_ids))}))
    return side


def guard_checkpoint(path, S: pd.DataFrame, track: str, g, design: str) -> None:
    """Refuse a checkpoint unless its M5-F training-id record exists, was written for the same (track, fold, design)
    and lies inside that training pool. M5 / M5-X checkpoints have no such record and are therefore never loadable."""
    side = Path(str(path) + SIDECAR_SUFFIX)
    if not side.exists():
        raise FitGuardError(f"{path} has no M5-F training-id record; M5 / M5-X checkpoints are not reused")
    rec = json.loads(side.read_text())
    if rec.get("protocol") != "M5-F" or (rec.get("track"), rec.get("fold"), rec.get("design")) != (track, g, design):
        raise FitGuardError(f"{path} was trained for {rec.get('track')}/{rec.get('fold')}/{rec.get('design')}, "
                            f"not {track}/{g}/{design}")
    assert_fit_ids(rec["train_ids"], S, track, g, design)


def load_checkpoint(path, S: pd.DataFrame, track: str, g, design: str, map_location="cpu"):
    """THE M5-F checkpoint loader (single choke point; a static test forbids ``torch.load`` elsewhere in M5-F)."""
    import torch

    guard_checkpoint(path, S, track, g, design)
    obj = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(obj, dict):
        obj["f_guard"] = (track, g, design)
    return obj


def require_guarded(obj, track: str, g, design: str) -> None:
    if obj is not None and (not isinstance(obj, dict) or obj.get("f_guard") != (track, g, design)):
        raise FitGuardError("checkpoint was not loaded through ftrack.load_checkpoint for this fold")


# ------------------------------------------------------------------------------------------------ disk (§2.2)
def disk_ok(path="/home/bcheng", need_gb: float = 0.0) -> tuple[bool, float]:
    """(ok, free GB): True if ``need_gb`` more data keeps free space at or above the 60 GB floor."""
    import shutil

    free = shutil.disk_usage(path).free / 1e9
    return (free - need_gb) >= DISK_FLOOR_GB, free


# ------------------------------------------------------------------------------------------------ GPUs (§2.2)
def unoccupied_gpus() -> list[int]:
    """GPU indices on which no process of ANOTHER user runs (an unreadable owner counts as another user).
    M5-F jobs pass this list as ``SENTINEL_GPUS`` to the lock helper (``python -m sentinel.core.gpu --min-free-mb
    ...``) and do not start when it is empty (the helper would fall back to its whole cap)."""
    import subprocess

    q = subprocess.run(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"], capture_output=True,
                       text=True, timeout=60).stdout
    bus = {b.strip().lower(): int(i) for i, b in (ln.split(",") for ln in q.strip().splitlines())}
    a = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id", "--format=csv,noheader"],
                       capture_output=True, text=True, timeout=60).stdout
    busy = set()
    for ln in a.strip().splitlines():
        if "," not in ln:
            continue
        pid, b = (x.strip() for x in ln.split(","))
        try:
            mine = os.stat(f"/proc/{int(pid)}").st_uid == os.getuid()
        except (OSError, ValueError):
            mine = False
        if not mine:
            busy.add(bus.get(b.lower()))
    return sorted(set(bus.values()) - busy)


# ------------------------------------------------------------------------------------- locked store (§2.1, §2.3)
# Fetcher, extractor and counting reader of the French data. Nothing here runs at import. The fetcher refuses every
# request until the lead's stamp of the protocol is in the registry (gate G0). It parses no response except the
# ``count`` field of a count response. The extractor copies whitelisted columns and routes rows by date alone.
STAMP_WORKSTREAM = "bioassess"
STAMP_EXPERIMENT = "bioassess_f/prereg_snapshot"     # the lead's stamp row (lead rulings L1)
G2_EXPERIMENT = "bioassess_f/gate_g2"                # the lead's row for gate G2 (L2): lock list + codes file
G4_EXPERIMENT = "bioassess_f/gate_g4"                # the lead's row for gate G4 (L2): split file + codes file
GATE_WRITER = "lead"                                 # extra.written_by of the stamp row and of both gate rows
RULINGS_FILE = "docs/preregistration/bioassess_f_lead_rulings.md"
CODES_FILE = "docs/preregistration/bioassess_f_codes.json"
STAMPED_FILES = (PROTOCOL_FILE, RULINGS_FILE)
HUBEAU = "https://hubeau.eaufrance.fr/api"
ENDPOINTS = {                                     # the only endpoints ever requested (§2.1 item 5)
    "stations_hydrobio": "/v1/hydrobio/stations_hydrobio",
    "indices": "/v1/hydrobio/indices",
    "taxons": "/v1/hydrobio/taxons",
    "station_pc": "/v2/qualite_rivieres/station_pc",
    "analyse_pc": "/v2/qualite_rivieres/analyse_pc",
}
CHEMISTRY_ENDPOINTS = ("station_pc", "analyse_pc")   # physico-chemistry API: gate G2 applies to both
STATION_PARAM = {"stations_hydrobio": "code_station_hydrobio", "indices": "code_station_hydrobio",
                 "taxons": "code_station_hydrobio", "station_pc": "code_station", "analyse_pc": "code_station"}
BIO_SUPPORTS = ("13", "10")                      # macroinvertebrates, benthic diatoms
BIO_FROM = dt.date(2007, 1, 1)
CHEM_FROM = dt.date(2006, 1, 1)
ROUTE_MIN = dt.date(1990, 1, 1)                  # rows dated earlier stay in the locked store (§2.3 item 1)
PC_PARAMETERS = ("1301", "1302", "1303", "1305", "1311", "1312", "1313", "1319", "1335", "1337", "1339", "1340",
                 "1350", "1433")                 # SANDRE parameter codes of §6.3
API_MAX_RECORDS = 20000                          # documented access depth (page x size) of both APIs
PAGE_SIZE = 5000
STATION_BATCH = 50
MIN_INTERVAL_S = 0.5
RETRY_STATUS = (0, 429, 500, 502, 503, 504)      # 0 = network error or timeout; only these are sent again
RETRY_WAIT_S = 10.0                              # doubled at every further attempt of the same request
SIZE_MARGIN = 1.5
USER_AGENT = "SENTINEL-v2 M5-F research pull (filtered requests, one at a time, >= 0.5 s apart)"
LICENCE = "Licence Ouverte / Open Licence Etalab 2.0"
ATTRIBUTION = "Source: Naiades / Hub'eau (eaufrance), OFB and the French water agencies"

HYDROBIO_STATION_COLS = ("code_station_hydrobio", "longitude", "latitude", "code_departement", "code_commune",
                         "code_cours_eau", "code_masse_eau")
BIO_WHITELIST = (
    "code_station_hydrobio", "date_prelevement", "code_support", "code_operation_prelevement", "code_prelevement",
    "code_banque_reference", "code_appel_taxon", "libelle_appel_taxon", "codes_taxons_parents",
    "libelles_taxons_parents", "code_type_resultat", "resultat_taxon", "code_qualification", "code_methode",
    "libelle_methode", "code_lot", "codes_indices_operation", "code_indice", "libelle_indice", "resultat_indice",
    "unite_indice", "longitude", "latitude", "code_departement", "code_commune", "code_cours_eau", "code_masse_eau")
STATION_PC_WHITELIST = ("code_station", "longitude", "latitude", "code_departement", "code_commune",
                        "code_cours_eau", "code_masse_deau", "altitude_point_caracteristique")
CHEM_WHITELIST = (
    "code_station", "date_prelevement", "heure_prelevement", "code_parametre", "resultat", "code_unite",
    "code_fraction", "code_support", "code_remarque", "code_insitu", "limite_detection", "limite_quantification",
    "limite_saturation", "code_statut", "code_qualification", "code_operation", "code_prelevement", "code_analyse",
    "code_banque_reference")
CHEM_VALUE_COLS = ("resultat", "limite_detection", "limite_quantification", "limite_saturation")
CHEM_COUNT_COLS = ("code_station", "date_prelevement", "heure_prelevement", "code_parametre", "resultat",
                   "code_unite", "code_fraction", "code_support", "code_remarque", "code_insitu",
                   "code_qualification")         # all that locked_availability loads (rules 1-8 of §6.3)
# §6.3 label table: parameter -> (fractions kept, units kept). Fraction 23 raw water, 3 filtered phase.
PC_LABEL_CODES = {
    "1302": (("23",), ("264",)), "1303": (("23",), ("147",)), "1337": (("3", "23"), ("164", "162")),
    "1340": (("3", "23"), ("173", "168")), "1335": (("3", "23"), ("169", "168")),
    "1433": (("3", "23"), ("176", "177")), "1350": (("23",), ("177",)), "1305": (("23",), ("162",)),
    "1311": (("23",), ("175",)), "1312": (("23",), ("243",)), "1313": (("23",), ("175",)),
    "1339": (("3", "23"), ("171", "168")), "1319": (("23",), ("168",)), "1301": (("23",), ("27",)),
}
PC_SUPPORT = "3"                                 # rule 1: water
QUALIFICATION_KEEP = ("1", "4", "0")             # rule 3
REMARK_KEEP = ("1", "2", "7", "10", "3")         # rule 5: reported, censored below (2, 7, 10), censored above (3)
WHITELIST = {"stations_hydrobio": HYDROBIO_STATION_COLS, "indices": BIO_WHITELIST, "taxons": BIO_WHITELIST,
             "station_pc": STATION_PC_WHITELIST, "analyse_pc": CHEM_WHITELIST}
AVAILABILITY_WINDOWS = {"pm30d": (-30, 30), "prior365d": (-365, 0)}


class FetchRefused(FitGuardError):
    """A French request or extraction that a gate of the M5-F protocol does not allow."""


class BlockTooLarge(RuntimeError):
    """A block above the 20,000-record access depth; it is split by month, then by smaller station batches."""


def unlocked_dir() -> Path:
    """Tables written by ``extract_locked``. The fetcher never writes here."""
    return paths.PROCESSED / "naiades_f"


class StampError(FitGuardError):
    """The lead's stamp (or a gate row of the lead) is missing, or a stamped file changed or is writable."""


def gate_rows(experiment: str, registry_path=None) -> list[dict]:
    from sentinel.core.registry import read_registry

    return [r for r in read_registry(registry_path)
            if r.get("workstream") == STAMP_WORKSTREAM and r.get("experiment") == experiment]


def _assert_gate(experiment: str, files, registry_path=None, repo=None, rows=None) -> dict:
    """The registry row of ``experiment`` (workstream ``bioassess``, ``extra.written_by == "lead"``) whose
    ``extra.sha256`` maps every file of ``files`` to its current sha256. Every file must exist and be read-only (no
    write bit). StampError otherwise. Only the lead writes these rows; no code of this workstream does."""
    from sentinel.core.registry import file_sha256

    repo = Path(repo) if repo is not None else paths.REPO_ROOT
    cur = {}
    for f in files:
        q = repo / f
        if not q.is_file():
            raise StampError(f"{experiment}: {f} does not exist")
        if q.stat().st_mode & 0o222:
            raise StampError(f"{experiment}: {f} is writable; a stamped file is read-only")
        cur[f] = file_sha256(q)
    rows = gate_rows(experiment, registry_path) if rows is None else rows
    if not rows:
        raise StampError(f"no registry row {STAMP_WORKSTREAM} / {experiment}")
    for r in rows:
        extra = r.get("extra") or {}
        m = extra.get("sha256")
        if extra.get("written_by") == GATE_WRITER and isinstance(m, dict) and all(m.get(f) == cur[f] for f in files):
            return r
    raise StampError(f"{experiment}: no registry row written by the lead records the current sha256 of "
                     f"{', '.join(files)} (a stamped file changed, or the row does not list it)")


def assert_stamped(registry_path=None, repo=None, rows=None) -> dict:
    """Gate G0 as the lead's clarification L1 defines it. Passes only if (i) the registry holds a row of workstream
    ``bioassess``, experiment ``bioassess_f/prereg_snapshot``, whose ``extra.sha256`` maps the protocol AND the lead
    rulings file to their current sha256, and (ii) both files are read-only. Returns the stamp row.

    Called first by the fetcher (before every request), the extractor, the counting reader, every French
    development entry point and ``ftestlook``."""
    return _assert_gate(STAMP_EXPERIMENT, STAMPED_FILES, registry_path, repo, rows)


def stamp_ok(registry_path=None, repo=None) -> bool:
    try:
        assert_stamped(registry_path, repo)
    except StampError:
        return False
    return True


def assert_g2(registry_path=None, repo=None, rows=None) -> dict:
    """Gate G2 (L2): the lock list and the codes file exist read-only and the LEAD has logged their sha256 in a
    registry row ``bioassess`` / ``bioassess_f/gate_g2`` (``extra.sha256``). No chemistry request before it."""
    return _assert_gate(G2_EXPERIMENT, (LOCK_FILE, CODES_FILE), registry_path, repo, rows)


def assert_g4(registry_path=None, repo=None, rows=None) -> dict:
    """Gate G4 (L2): split file and codes file read-only, sha256 logged by the LEAD in ``bioassess_f/gate_g4``.
    No French chemistry value leaves the locked store, and no fit on French rows, before it."""
    return _assert_gate(G4_EXPERIMENT, (SPLIT_FILE, CODES_FILE), registry_path, repo, rows)


def _http_get(url: str, headers: dict) -> tuple[int, bytes]:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=180) as r:
            return int(r.status), r.read()
    except urllib.error.HTTPError as e:
        return int(e.code), b""


def _year_windows(first: dt.date, last: dt.date) -> list[tuple[str, str]]:
    return [(max(first, dt.date(y, 1, 1)).isoformat(), min(last, dt.date(y, 12, 31)).isoformat())
            for y in range(first.year, last.year + 1)]


def biology_blocks(today, departments=None) -> dict[str, list[dict]]:
    """Step B request blocks: stations by department; indices and taxa by department x support x calendar year."""
    deps = sorted(METROPOLITAN if departments is None else departments)
    if not set(deps) <= METROPOLITAN:
        raise FetchRefused("only the metropolitan departments of §4 are requested")
    out = {"stations_hydrobio": [{"code_departement": d} for d in deps]}
    for ep in ("indices", "taxons"):
        out[ep] = [{"code_departement": d, "code_support": s, "date_debut_prelevement": a, "date_fin_prelevement": b}
                   for d in deps for s in BIO_SUPPORTS for a, b in _year_windows(BIO_FROM, _as_date(today))]
    return out


def chemistry_blocks(stations, today) -> list[dict]:
    """Step C request blocks, one form only (lead ruling 2026-11-10): a station batch (<= 50 codes) x ALL the
    parameter codes of §6.3 in one request x one calendar year. A calendar year never spans the 2021-12-31 /
    2022-01-01 boundary, so locked and unlocked dates are separate requests."""
    st = sorted({str(s).strip() for s in stations})
    return [{"code_station": ",".join(st[i:i + STATION_BATCH]), "code_parametre": ",".join(PC_PARAMETERS),
             "date_debut_prelevement": a, "date_fin_prelevement": b}
            for i in range(0, len(st), STATION_BATCH) for a, b in _year_windows(CHEM_FROM, _as_date(today))]


def split_block(endpoint: str, params: dict) -> list[dict]:
    """Smaller blocks of one block above the access depth: calendar months first, then halves of the station list."""
    a, b = params.get("date_debut_prelevement"), params.get("date_fin_prelevement")
    if a and b:
        a, b = dt.date.fromisoformat(a), dt.date.fromisoformat(b)
        if (a.year, a.month) != (b.year, b.month):
            out, cur = [], a
            while cur <= b:
                nxt = dt.date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
                out.append({**params, "date_debut_prelevement": cur.isoformat(),
                            "date_fin_prelevement": min(b, nxt - dt.timedelta(days=1)).isoformat()})
                cur = nxt
            return out
    key = STATION_PARAM[endpoint]
    codes = [c for c in str(params.get(key, "")).split(",") if c]
    if len(codes) > 1:
        h = len(codes) // 2
        return [{**params, key: ",".join(codes[:h])}, {**params, key: ",".join(codes[h:])}]
    raise FetchRefused(f"a {endpoint} block above {API_MAX_RECORDS} records cannot be split further; stop and report")


class LockedFetcher:
    """The only writer of French data (§2.1 route, §2.3 item 1). Every response body goes gzip-compressed into the
    locked store (directory mode 0700, file mode 0400) and nowhere else; every request goes to the manifest.

    Gates checked before EVERY request: G0 (``assert_stamped``), G1 (60 GB disk floor and the 5 GB cap, with the
    1.5 x size check once the bytes per record of an endpoint are measured), G2 (``assert_g2``: lock list, codes
    file and the lead's row, before any request to the physico-chemistry API). One request at a time, at least
    0.5 s apart. A request already in the manifest with status 200/206 is not sent again (resume).
    A network error, a timeout or HTTP 429/5xx is logged and the same request is sent again at most ``retries``
    times (default 0), waiting 10 s, 20 s, 40 s, ...; any other status stops the pull at once.
    ``transport(url, headers) -> (status, body)`` is injectable so that the tests run without a network.
    """

    def __init__(self, *, transport=None, root=None, manifest=None, today=None, registry_path=None, repo=None,
                 extra_parameters=(), cap_gb: float = NEW_DATA_CAP_GB, other_dirs=None, disk=None, sleep=None,
                 clock=None, retries: int = 0):
        import time

        self.registry_path, self.repo, self.retries = registry_path, repo, int(retries)
        self.transport = transport or _http_get
        self.root = Path(root) if root is not None else locked_dir()
        self.manifest = Path(manifest) if manifest is not None else paths.MANIFESTS / "naiades_f_fetch.jsonl"
        self.today = _as_date(today) if today is not None else dt.datetime.now(dt.timezone.utc).date()
        self.parameters = frozenset(PC_PARAMETERS) | {str(p) for p in extra_parameters}   # extra: F-TOX (§2.1 item 7)
        self.cap_bytes = float(cap_gb) * 1e9
        self.other_dirs = [Path(p) for p in (other_dirs if other_dirs is not None else [unlocked_dir()])]
        self.disk = disk or disk_ok
        self.sleep, self.clock = sleep or time.sleep, clock or time.monotonic
        self._last, self._stamp_row, self._g2_row = None, None, None
        self._bpr: dict[str, float] = {}           # endpoint -> gzip bytes per record of its first completed block
        self._done = {r["url"]: r for r in self.manifest_rows() if r.get("status") in (200, 206) and r.get("file")}
        self._used = float(sum(_tree_bytes(p) for p in [self.root, *self.other_dirs]))
        self.peak_bytes = self._used

    # -------------------------------------------------------------------------------------------- bookkeeping
    def manifest_rows(self) -> list[dict]:
        if not self.manifest.exists():
            return []
        return [json.loads(ln) for ln in self.manifest.read_text().splitlines() if ln.strip()]

    def used_gb(self) -> float:
        return self._used / 1e9

    # -------------------------------------------------------------------------------------------- gates
    def _gate(self, endpoint: str) -> None:
        if endpoint not in ENDPOINTS:
            raise FetchRefused(f"endpoint {endpoint!r} is never requested (§2.1 item 5)")
        # The registry is read until a matching row is found; that row is then kept, and the files' sha256 and
        # modes are checked again against it before every request.
        try:
            self._stamp_row = assert_stamped(self.registry_path, self.repo,
                                             rows=None if self._stamp_row is None else [self._stamp_row])
        except StampError as e:
            raise FetchRefused(f"gate G0: no French request without the lead's stamp ({e})") from e
        if endpoint in CHEMISTRY_ENDPOINTS:
            try:
                self._g2_row = assert_g2(self.registry_path, self.repo,
                                         rows=None if self._g2_row is None else [self._g2_row])
            except StampError as e:
                raise FetchRefused(f"gate G2: no physico-chemistry request before the lead logs G2 ({e})") from e
        ok, free = self.disk()
        if not ok:
            raise FetchRefused(f"gate G1: {free:.1f} GB free, below the {DISK_FLOOR_GB:.0f} GB floor")
        if self._used >= self.cap_bytes:
            raise FetchRefused(f"gate G1: {self.used_gb():.2f} GB of M5-F data, at the cap of "
                               f"{self.cap_bytes / 1e9:.0f} GB")

    def _check_params(self, endpoint: str, params: dict) -> None:
        if endpoint not in ENDPOINTS:
            raise FetchRefused(f"endpoint {endpoint!r} is never requested (§2.1 item 5)")
        if "durete" in str(params.get("fields", "")).split(","):
            raise FetchRefused("the durete field is never requested or extracted")
        a, b = params.get("date_debut_prelevement"), params.get("date_fin_prelevement")
        if endpoint == "analyse_pc":
            codes = [c for c in str(params.get("code_parametre", "")).split(",") if c]
            if not codes or not set(codes) <= self.parameters:
                raise FetchRefused("analyse_pc requests name parameter codes of §6.3 (or the F-TOX list) only")
            if set(codes) & set(PC_PARAMETERS) and codes != list(PC_PARAMETERS):
                raise FetchRefused("a §6.3 request carries all 14 parameter codes, in the fixed order (one form only)")
            st = [c for c in str(params.get("code_station", "")).split(",") if c]
            if not 1 <= len(st) <= STATION_BATCH:
                raise FetchRefused(f"analyse_pc requests name 1..{STATION_BATCH} station codes")
            if not (a and b):
                raise FetchRefused("analyse_pc requests carry both date filters")
            if dt.date.fromisoformat(a) <= TRAIN_TO < dt.date.fromisoformat(b):
                raise FetchRefused("requests for dates <= 2021-12-31 and >= 2022-01-01 are separate (§2.3 item 3)")
        if endpoint in ("indices", "taxons"):
            if not set(str(params.get("code_support", "")).split(",")) <= set(BIO_SUPPORTS):
                raise FetchRefused("biology requests name supports 13 and 10 only")
            if not (a and b):
                raise FetchRefused("biology requests carry both date filters")

    def url(self, endpoint: str, params: dict, csv: bool = True) -> str:
        from urllib.parse import urlencode

        self._check_params(endpoint, params)
        q = urlencode(sorted((k, str(v)) for k, v in params.items()), safe=",")
        return f"{HUBEAU}{ENDPOINTS[endpoint]}{'.csv' if csv else ''}?{q}"

    # -------------------------------------------------------------------------------------------- one request
    def _fetch(self, endpoint: str, url: str, kind: str) -> dict:
        import gzip

        if url in self._done:
            return self._done[url]
        attempt = 0
        while True:
            self._gate(endpoint)
            if self._last is not None:
                wait = MIN_INTERVAL_S - (self.clock() - self._last)
                if wait > 0:
                    self.sleep(wait)
            err = None
            try:
                status, body = self.transport(url, {"User-Agent": USER_AGENT})
            except OSError as e:                    # network error or timeout: logged like a response, status 0
                status, body, err = 0, b"", type(e).__name__
            self._last = self.clock()
            rec = {"endpoint": endpoint, "kind": kind, "url": url,
                   "utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "status": int(status),
                   "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(), "gz_bytes": 0, "file": None,
                   "licence": LICENCE, "attribution": ATTRIBUTION}
            if attempt:
                rec["attempt"] = attempt
            if err:
                rec["error"] = err
            if status in (200, 206):
                d = self.root / endpoint
                d.mkdir(parents=True, exist_ok=True)
                os.chmod(self.root, 0o700)
                os.chmod(d, 0o700)
                name = (f"{kind}_{hashlib.sha256(url.encode()).hexdigest()[:24]}."
                        f"{'json' if kind == 'count' else 'csv'}.gz")
                if (d / name).exists():             # left by an interrupted run: this request is not in the manifest
                    self._used -= (d / name).stat().st_size
                    (d / name).unlink()
                fd = os.open(d / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
                with os.fdopen(fd, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
                    gz.write(body)
                os.chmod(d / name, 0o400)
                rec["gz_bytes"], rec["file"] = (d / name).stat().st_size, f"{endpoint}/{name}"
                self._used += rec["gz_bytes"]
                self.peak_bytes = max(self.peak_bytes, self._used)
            self.manifest.parent.mkdir(parents=True, exist_ok=True)
            with open(self.manifest, "a") as fh:
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
            if status in (200, 206):
                self._done[url] = rec
                return rec
            if status not in RETRY_STATUS or attempt >= self.retries:
                raise RuntimeError(f"HTTP {status} for {url}{' (' + err + ')' if err else ''}; stop and report")
            attempt += 1
            self.sleep(RETRY_WAIT_S * 2 ** (attempt - 1))

    def count(self, endpoint: str, params: dict) -> int:
        """Number of records of a block, from a ``size=1`` request that asks for the station code field only.
        The ``count`` field is the only thing this class ever parses out of a response."""
        import gzip

        self._check_params(endpoint, params)
        q = {**params, "size": 1, "fields": STATION_PARAM[endpoint]}
        rec = self._fetch(endpoint, self.url(endpoint, q, csv=False), "count")
        with gzip.open(self.root / rec["file"], "rb") as fh:
            return int(json.loads(fh.read())["count"])

    def _size_check(self, endpoint: str, n_records: int) -> None:
        if endpoint in self._bpr:
            need = SIZE_MARGIN * n_records * self._bpr[endpoint]
            if need > self.cap_bytes - self._used:
                raise FetchRefused(f"gate G1: {endpoint} projects {need / 1e9:.2f} GB (1.5 x), "
                                   f"{(self.cap_bytes - self._used) / 1e9:.2f} GB of the cap remain; stop and report")

    def pull(self, endpoint: str, params: dict, n: int | None = None) -> dict:
        """All pages of one block (<= 20,000 records). The CSV bodies are stored, never parsed."""
        self._check_params(endpoint, params)
        n = self.count(endpoint, params) if n is None else int(n)
        if n > API_MAX_RECORDS:
            raise BlockTooLarge(f"{endpoint}: {n} records")
        gz = 0
        for page in range(1, -(-n // PAGE_SIZE) + 1):
            q = {**params, "page": page, "size": PAGE_SIZE, "sort": "asc", "fields": ",".join(WHITELIST[endpoint])}
            gz += self._fetch(endpoint, self.url(endpoint, q), "page")["gz_bytes"]
        if n > 0 and endpoint not in self._bpr:
            self._bpr[endpoint] = gz / n
        return {"records": n, "gz_bytes": gz}

    def resolve(self, endpoint: str, blocks, unsplit: list | None = None) -> list[tuple[dict, int]]:
        """Count every block; split the ones above the access depth until every block fits. A block that is still
        too large and cannot be split stops the step, unless the caller passes the list ``unsplit``: the block
        and its record count are then appended to it, the block is NOT pulled, and the caller reports it."""
        out, todo = [], list(blocks)
        while todo:
            p = todo.pop(0)
            n = self.count(endpoint, p)
            if n > API_MAX_RECORDS:
                try:
                    todo = split_block(endpoint, p) + todo
                except FetchRefused:
                    if unsplit is None:
                        raise
                    unsplit.append((p, n))
            else:
                out.append((p, n))
        return out

    def run_step(self, endpoint: str, blocks, unsplit: list | None = None) -> dict:
        """One step of the route: counts first, then the pulls, with the size check before every block."""
        res = self.resolve(endpoint, blocks, unsplit)
        total, done = sum(n for _, n in res), 0
        for p, n in res:
            if n == 0:
                continue
            self._size_check(endpoint, total - done)
            self.pull(endpoint, p, n)
            done += n
        return {"endpoint": endpoint, "blocks": len(res), "records": total,
                "gz_bytes_per_record": self._bpr.get(endpoint), "used_gb": self.used_gb(),
                "peak_gb": self.peak_bytes / 1e9}


def _tree_bytes(p: Path) -> int:
    p = Path(p)
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def _stored_csvs(root: Path, endpoint: str) -> list[Path]:
    return sorted((Path(root) / endpoint).glob("page_*.csv.gz"))


def _read_stored(path: Path, columns) -> pd.DataFrame:
    """Whitelisted columns of one stored CSV page, every field as text (no value is parsed as a number)."""
    import gzip

    with gzip.open(path, "rt", encoding="utf-8") as fh:
        head = fh.readline()
    if not head.strip():
        return pd.DataFrame(columns=[c for c in columns])
    sep = ";" if head.count(";") >= head.count(",") else ","
    keep = set(columns)
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, compression="gzip",
                       usecols=lambda c: c in keep)


def _route(dates: pd.Series, download_date: dt.date) -> tuple[np.ndarray, np.ndarray]:
    """(unlocked, locked) masks from ``date_prelevement`` read only as a date. Missing, unparseable, pre-1990 and
    post-download dates are in neither mask: those rows stay in the locked store."""
    d = pd.to_datetime(dates.astype(str).str.slice(0, 10), format="%Y-%m-%d", errors="coerce")
    ok = d.notna() & (d >= pd.Timestamp(ROUTE_MIN)) & (d <= pd.Timestamp(download_date))
    return (ok & (d <= pd.Timestamp(TRAIN_TO))).to_numpy(), (ok & (d >= pd.Timestamp(LOCK_FROM))).to_numpy()


def extract_locked(download_date, *, chemistry: str = "none", root=None, out=None, registry_path=None,
                   repo=None) -> dict:
    """The one audited extractor (§2.3 item 1). Copies out of the locked store, by whitelist and date only:

    * biology input columns: rows dated <= 2021-12-31 to ``<endpoint>_unlocked``, rows dated >= 2022-01-01 to
      ``<endpoint>_locked_inputs`` (permitted uses: §2.3 item 6);
    * station columns of ``stations_hydrobio`` and ``station_pc`` (never ``durete``);
    * chemistry rows dated <= 2021-12-31 only. ``chemistry="codes"``: the code columns without any result or limit
      value (the G4 tabulation); ``"full"``: the whole chemistry whitelist, allowed only after the lead's G4 row
      (``assert_g4``); ``"none"``: no chemistry.

    Chemistry rows dated >= 2022-01-01 never leave the store. No result value is parsed, logged or summarised:
    every field is copied as text. Returns row counts only.
    """
    if chemistry not in ("none", "codes", "full"):
        raise ValueError("chemistry is 'none', 'codes' or 'full'")
    assert_stamped(registry_path, repo)
    if chemistry == "full":
        try:
            assert_g4(registry_path, repo)
        except StampError as e:
            raise FetchRefused(f"gate G4: no French chemistry value leaves the locked store before the lead's "
                               f"G4 stamp ({e})") from e
    root = Path(root) if root is not None else locked_dir()
    out = Path(out) if out is not None else unlocked_dir()
    day = _as_date(download_date)
    counts: dict = {"download_date": day.isoformat(), "chemistry": chemistry, "tables": {}, "stay_locked_bad_date": {},
                    "chemistry_rows_left_locked": 0}

    def write(df: pd.DataFrame, table: str, src: Path) -> None:
        counts["tables"][table] = counts["tables"].get(table, 0) + len(df)
        if len(df):
            (out / table).mkdir(parents=True, exist_ok=True)
            df.reset_index(drop=True).to_parquet(out / table / (src.name[:-len(".csv.gz")] + ".parquet"), index=False)

    for ep, cols in (("stations_hydrobio", HYDROBIO_STATION_COLS), ("station_pc", STATION_PC_WHITELIST)):
        for f in _stored_csvs(root, ep):
            write(_read_stored(f, cols), ep, f)
    for ep in ("indices", "taxons"):
        for f in _stored_csvs(root, ep):
            df = _read_stored(f, BIO_WHITELIST)
            if "date_prelevement" not in df:
                counts["stay_locked_bad_date"][ep] = counts["stay_locked_bad_date"].get(ep, 0) + len(df)
                continue
            un, lk = _route(df["date_prelevement"], day)
            counts["stay_locked_bad_date"][ep] = counts["stay_locked_bad_date"].get(ep, 0) + int((~un & ~lk).sum())
            write(df[un], f"{ep}_unlocked", f)
            write(df[lk], f"{ep}_locked_inputs", f)
    if chemistry != "none":
        cols = CHEM_WHITELIST if chemistry == "full" else tuple(c for c in CHEM_WHITELIST if c not in CHEM_VALUE_COLS)
        table = "analyse_pc_unlocked" if chemistry == "full" else "analyse_pc_codes_unlocked"
        for f in _stored_csvs(root, "analyse_pc"):
            df = _read_stored(f, cols)
            if "date_prelevement" not in df:
                counts["stay_locked_bad_date"]["analyse_pc"] = (counts["stay_locked_bad_date"].get("analyse_pc", 0)
                                                                + len(df))
                continue
            un, lk = _route(df["date_prelevement"], day)
            counts["stay_locked_bad_date"]["analyse_pc"] = (counts["stay_locked_bad_date"].get("analyse_pc", 0)
                                                            + int((~un & ~lk).sum()))
            counts["chemistry_rows_left_locked"] += int(lk.sum())
            write(df[un], table, f)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"EXTRACT_{chemistry}.json").write_text(json.dumps(counts, indent=1, sort_keys=True))
    return counts


def _rules_1_8(b: pd.DataFrame, stations: set, codes: dict) -> tuple[pd.DataFrame, dict]:
    """Rules 1-6 and 8 of §6.3 on one stored page. Returns the kept analyses with the identity columns of rule 7
    (the parsed result is used for that identity only) and the number of analyses dropped by each rule."""
    drop = {}

    def cut(mask, rule):
        nonlocal b
        mask = np.asarray(mask, bool)
        drop[rule] = int((~mask).sum())
        b = b[mask]

    for c in CHEM_COUNT_COLS:
        b[c] = b[c].str.strip() if c in b else ""
    b = b[b["code_station"].isin(stations)]                                   # the paired stations of ``ops``
    cut(b["code_support"] == PC_SUPPORT, "rule1_support")
    table = codes["label_codes"]
    ok = np.fromiter((p in table and f in table[p][0] and u in table[p][1]
                      for p, f, u in zip(b["code_parametre"], b["code_fraction"], b["code_unite"])), bool, len(b))
    cut(ok, "rule2_parameter_fraction_unit")
    cut(b["code_qualification"].isin(codes["qualification"]), "rule3_qualification")
    cut(b["code_remarque"].isin(codes["remark"]) & (b["resultat"] != ""), "rule5_remark_or_empty")
    v = pd.to_numeric(b["resultat"], errors="coerce").to_numpy(float)
    b = b.assign(_v=v)
    cut(np.isfinite(v), "rule8_not_numeric")
    d = pd.to_datetime(b["date_prelevement"].str.slice(0, 10), format="%Y-%m-%d", errors="coerce")
    b = b.assign(date_obs=d.to_numpy())
    cut(d.notna(), "no_date")
    ident = ["code_station", "date_obs", "heure_prelevement", "code_parametre", "code_fraction", "code_unite", "_v",
             "code_remarque", "code_insitu"]
    return b[ident], drop


def locked_availability(ops: pd.DataFrame, *, root=None, codes: dict | None = None, windows: dict | None = None,
                        station_chunk: int = 500, registry_path=None, repo=None) -> pd.DataFrame:
    """The one counting reader of chemistry in the locked store before the single look (§2.3 item 4; lead rulings
    §2.4). It applies rules 1-8 of §6.3 inside and returns, per (operation key, parameter, window), the number of
    analyses ``n_obs`` (after the duplicate rule 7) and the number of distinct sampling dates ``n_dates``, and
    nothing else. ``.attrs["dropped"]`` holds the number of analyses dropped by each rule (counts only).

    The result column is loaded only to apply rules 5 (empty), 7 (identity of exact duplicates) and 8 (numeric and
    finite). No result value is returned, logged, summarised or written; the limit columns are never loaded.

    ``ops``: columns ``key``, ``station`` (the paired chemistry station code), ``date``. ``codes`` (optional, from
    the stamped ``bioassess_f_codes.json``): ``label_codes`` {parameter: (fractions, units)}, ``qualification``,
    ``remark``; default: the table and rules of §6.3 as stamped.
    """
    assert_stamped(registry_path, repo)
    if not {"key", "station", "date"} <= set(ops.columns):
        raise ValueError("ops needs columns key, station, date")
    root = Path(root) if root is not None else locked_dir()
    windows = windows or AVAILABILITY_WINDOWS
    codes = dict(codes or {})
    rules = {"label_codes": {str(p): (tuple(map(str, f)), tuple(map(str, u)))
                             for p, (f, u) in codes.get("label_codes", PC_LABEL_CODES).items()},
             "qualification": {str(c) for c in codes.get("qualification", QUALIFICATION_KEEP)},
             "remark": {str(c) for c in codes.get("remark", REMARK_KEEP)}}
    cols = ["key", "parameter", "window", "n_obs", "n_dates"]
    o = ops[["key", "station", "date"]].copy()
    o["station"] = o["station"].astype(str).str.strip()
    o["date"] = pd.to_datetime(o["date"]).dt.normalize()
    stations = set(o["station"])
    parts, dropped = [], {}
    for f in _stored_csvs(root, "analyse_pc"):
        b = _read_stored(f, CHEM_COUNT_COLS)
        if not len(b):
            continue
        kept, drop = _rules_1_8(b, stations, rules)
        for k, n in drop.items():
            dropped[k] = dropped.get(k, 0) + n
        parts.append(kept)
    empty = pd.DataFrame(columns=cols)
    empty.attrs["dropped"] = dropped
    if not parts or not len(o):
        return empty
    obs = pd.concat(parts, ignore_index=True)
    n0 = len(obs)
    obs = obs.drop_duplicates()                                               # rule 7: exact duplicates count once
    dropped["rule7_exact_duplicates"] = n0 - len(obs)
    obs = obs[["code_station", "date_obs", "code_parametre"]].rename(
        columns={"code_station": "station", "code_parametre": "parameter"})   # the result is gone from here on
    out, st = [], sorted(stations)
    for i in range(0, len(st), int(station_chunk)):
        chunk = set(st[i:i + int(station_chunk)])
        m = o[o["station"].isin(chunk)].merge(obs[obs["station"].isin(chunk)], on="station")
        lag = (m["date_obs"] - m["date"]).dt.days
        for w, (lo, hi) in windows.items():
            k = m[(lag >= lo) & (lag <= hi)].groupby(["key", "parameter"])["date_obs"].agg(
                n_obs="size", n_dates="nunique").reset_index()
            k["window"] = w
            out.append(k)
    res = pd.concat(out, ignore_index=True) if out else empty
    res = res[cols].sort_values(["key", "parameter", "window"]).reset_index(drop=True)
    res.attrs["dropped"] = dropped
    return res
