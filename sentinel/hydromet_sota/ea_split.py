# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""The M6 T-E England split: one canonical core, the role draw, the status of every catalogue station and the
minimum-group check (``docs/preregistration/hydromet_ea_protocol.md`` §4–§6; lead rulings
``hydromet_arch_v2_lead_rulings.md`` §2.4).

The writer (``scripts/data/sota_hydromet_ea_build.py split``) and **every** reader (``ea_data.load_split``, M1, the look)
hash the split through :func:`core_sha256` and check it through :func:`verify`; there is no second definition of the
core anywhere. The check is never loosened to make a file load: a file that fails is refused.

Pure functions on plain dicts (no I/O except :func:`load` / :func:`load_pair`, no EA value of any kind): only
catalogue metadata and geometry enter a split.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from ..core import paths

SPLIT_FILE = paths.REPO_ROOT / "docs" / "preregistration" / "hydromet_ea_split.json"
M1_SPLIT_FILE = paths.REPO_ROOT / "docs" / "preregistration" / "sonde_ea_split.json"
VERSION = "1.0"
SALT = "M6-EA-2026-11-10:"
SHARE_TEST = 0.50          # S, lead ruling 2026-11-10 (hydromet_arch_v2_lead_rulings.md §2.1 Q4)
SHARE_VAL = 0.15
BIG_GROUP_FRAC = 0.15      # a group holding more than this share of eligible stations is train-role
G_MIN = 15                 # minimum basin groups of a claim endpoint (never lowered)
G_EFF_MIN = 8.0            # minimum effective number of groups (never lowered)
ROLES = ("train", "val", "test")
STATUSES = ROLES + ("reserved",)
CLOSED = ("test", "reserved")          # closed to M1 and to every other workstream for all years
_PRECEDENCE = {"test": 0, "val": 1, "train": 2}   # several flow-connected roles: the most closed one wins
# endpoint -> the catalogue measure a station needs to be expected in the endpoint's cell
ENDPOINTS = {"temp_c_mean": "temp_c", "temp_c_7dadm": "temp_c", "do_mgl_mean": "do_mgl", "do_mgl_min": "do_mgl",
             "heatwave_rmse_temp_c_mean": "temp_c", "heatwave_rmse_temp_c_7dadm": "temp_c"}

RULE_TEXT = (
    "Eligible (catalogue metadata and geometry only): nearest HydroRIVERS v1.0 reach within 1 km (EPSG:27700), the "
    "station point inside a HydroBASINS level-7 unit (hybas_l7 not null), and catalogue open span "
    "(dateOpened..dateClosed, clipped to 2012-01-01..2026-09-30) >= 60 days. Groups (eligible stations only): the "
    "level-7 unit of the station point, merged (union-find) over direct station-station flow connections (same reach, "
    "or one station's reach on the other's HydroRIVERS NEXT_DOWN chain, any distance). Group id: the smallest level-7 "
    "HYBAS_ID (decimal string) among the merged units. Roles: groups ordered by sha256(salt + group id) hex digest; a "
    "group with more than big_group_frac of the eligible stations is train; otherwise, by the cumulative eligible-"
    "station share BEFORE the group: test while < share_test, val while < share_test + share_val, train after. "
    "Every catalogue station has a status: an eligible station has its group's role; an ineligible station takes the "
    "role of the eligible station(s) it is flow-connected to (through its nearest reach within 5 km; several roles: "
    "test before val before train), otherwise the role of the group that holds its level-7 unit, otherwise "
    "'reserved'. Ineligible stations never supply M6 rows. Stations with status test or reserved are closed to M1 and "
    "to every other workstream for all years.")

# The canonical core: every key that defines the split. Writer and readers hash exactly these keys.
CORE_KEYS = ("version", "rule", "salt", "share_test", "share_val", "big_group_frac", "eligibility", "role_of_group",
             "group_units", "station_group", "station_role", "stations", "closed_stations", "inputs_sha256")


class SplitError(ValueError):
    """The split file is incomplete, inconsistent or does not match its hash."""


def canonical_core(body: dict) -> dict:
    missing = [k for k in CORE_KEYS if k not in body]
    if missing:
        raise SplitError(f"split core keys missing: {missing}")
    return {k: body[k] for k in CORE_KEYS}


def core_sha256(body: dict) -> str:
    """sha256 of the canonical JSON of the core (sorted keys, compact separators, ASCII)."""
    blob = json.dumps(canonical_core(body), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("ascii")).hexdigest()


def group_order(groups, salt: str = SALT) -> list:
    return sorted((str(g) for g in groups), key=lambda g: hashlib.sha256((salt + g).encode()).hexdigest())


def draw_roles(group_sizes: dict, share_test: float = SHARE_TEST, share_val: float = SHARE_VAL, *, salt: str = SALT,
               big_group_frac: float = BIG_GROUP_FRAC) -> dict:
    """group id -> role from the number of eligible stations per group (nothing else)."""
    sizes = {str(g): int(n) for g, n in group_sizes.items()}
    tot, cum, role = sum(sizes.values()), 0, {}
    for g in group_order(sizes, salt):
        fr = cum / tot
        if sizes[g] > big_group_frac * tot:
            role[g] = "train"
        else:
            role[g] = "test" if fr < share_test else ("val" if fr < share_test + share_val else "train")
        cum += sizes[g]
    return role


def station_table(all_stations, station_group: dict, role_of_group: dict, unit_of: dict, flow_partners: dict) -> dict:
    """Status of every catalogue station.

    ``station_group``: eligible station -> group; ``unit_of``: station -> level-7 unit (str) or None;
    ``flow_partners``: station -> eligible stations it is flow-connected to (either direction).
    """
    unit_group = {}
    for s, g in station_group.items():
        u = unit_of.get(s)
        if u is None:
            raise SplitError(f"eligible station {s} has no level-7 unit")
        if unit_group.setdefault(str(u), str(g)) != str(g):
            raise SplitError(f"level-7 unit {u} lies in two groups")
    out = {}
    for s in sorted(str(x) for x in all_stations):
        if s in station_group:
            g = str(station_group[s])
            out[s] = {"eligible": True, "status": role_of_group[g], "basis": "eligible", "groups": [g]}
            continue
        gs = sorted({str(station_group[p]) for p in flow_partners.get(s, ()) if p in station_group})
        if gs:
            role = min((role_of_group[g] for g in gs), key=_PRECEDENCE.__getitem__)
            out[s] = {"eligible": False, "status": role, "basis": "flow_connected", "groups": gs}
            continue
        u = unit_of.get(s)
        if u is not None and str(u) in unit_group:
            g = unit_group[str(u)]
            out[s] = {"eligible": False, "status": role_of_group[g], "basis": "l7_unit", "groups": [g]}
        else:
            out[s] = {"eligible": False, "status": "reserved", "basis": "none", "groups": []}
    return out


def group_units(station_group: dict, unit_of: dict) -> dict:
    """group id -> sorted level-7 units merged into it (units of its eligible stations)."""
    gu = defaultdict(set)
    for s, g in station_group.items():
        gu[str(g)].add(str(unit_of[s]))
    return {g: sorted(u) for g, u in sorted(gu.items())}


def build_core(*, all_stations, station_group: dict, unit_of: dict, flow_partners: dict, eligibility: dict,
               inputs_sha256: dict, share_test: float = SHARE_TEST, share_val: float = SHARE_VAL, salt: str = SALT,
               big_group_frac: float = BIG_GROUP_FRAC) -> dict:
    """The core of a split from metadata-derived inputs; ``sha256`` is added by :func:`seal`."""
    station_group = {str(s): str(g) for s, g in station_group.items()}
    sizes = defaultdict(int)
    for g in station_group.values():
        sizes[g] += 1
    role = draw_roles(sizes, share_test, share_val, salt=salt, big_group_frac=big_group_frac)
    stations = station_table(all_stations, station_group, role, unit_of, flow_partners)
    return {"version": VERSION, "rule": RULE_TEXT, "salt": salt, "share_test": float(share_test),
            "share_val": float(share_val), "big_group_frac": float(big_group_frac), "eligibility": dict(eligibility),
            "role_of_group": {g: role[g] for g in sorted(role)}, "group_units": group_units(station_group, unit_of),
            "station_group": dict(sorted(station_group.items())),
            "station_role": {s: role[g] for s, g in sorted(station_group.items())}, "stations": stations,
            "closed_stations": sorted(s for s, v in stations.items() if v["status"] in CLOSED),
            "inputs_sha256": dict(sorted(inputs_sha256.items()))}


def seal(core: dict, **extra) -> dict:
    """Core + non-core information + ``core_keys`` + ``sha256`` (the body that is written to disk)."""
    clash = [k for k in extra if k in CORE_KEYS or k in ("sha256", "core_keys")]
    if clash:
        raise SplitError(f"non-core fields reuse core names: {clash}")
    body = {**canonical_core(core), **extra, "core_keys": list(CORE_KEYS)}
    body["sha256"] = core_sha256(body)
    return body


def verify(body: dict, *, ruled_parameters: bool = True) -> dict:
    """Refuse a split whose hash, core-key list or internal consistency fails. Returns ``body``.

    ``ruled_parameters``: also require the ruled salt, S, val share and guard (False only for synthetic test splits)."""
    if list(body.get("core_keys", ())) != list(CORE_KEYS):
        raise SplitError("split core-key list differs from ea_split.CORE_KEYS")
    if core_sha256(body) != body.get("sha256"):
        raise SplitError("split sha256 does not match its canonical core")
    if body["version"] != VERSION or body["rule"] != RULE_TEXT:
        raise SplitError("split version or rule text differs from this code")
    if ruled_parameters and (body["salt"], body["share_test"], body["share_val"], body["big_group_frac"]) != (
            SALT, SHARE_TEST, SHARE_VAL, BIG_GROUP_FRAC):
        raise SplitError("split parameters differ from the ruled salt / S / val share / guard")
    sg, sr, rg, st = body["station_group"], body["station_role"], body["role_of_group"], body["stations"]
    sizes = defaultdict(int)
    for g in sg.values():
        sizes[g] += 1
    redraw = draw_roles(sizes, body["share_test"], body["share_val"], salt=body["salt"],
                        big_group_frac=body["big_group_frac"])
    if redraw != rg:
        raise SplitError("role_of_group is not the draw of the stated rule")
    if set(sr) != set(sg) or any(sr[s] != rg[sg[s]] for s in sg):
        raise SplitError("station_role is not the role of the station's group")
    if {s for s, v in st.items() if v["eligible"]} != set(sg):
        raise SplitError("eligible stations and station_group differ")
    for s, v in st.items():
        if v["status"] not in STATUSES or (v["eligible"] and (v["status"] != sr[s] or v["groups"] != [sg[s]])):
            raise SplitError(f"station {s}: inconsistent status")
        if not v["eligible"] and v["groups"] and v["status"] != min((rg[g] for g in v["groups"]),
                                                                    key=_PRECEDENCE.__getitem__):
            raise SplitError(f"station {s}: status is not the role of its group(s)")
        if not v["groups"] and v["status"] != "reserved":
            raise SplitError(f"station {s}: no group and not reserved")
    if set(body["group_units"]) != set(rg):
        raise SplitError("group_units and role_of_group differ")
    units = [u for us in body["group_units"].values() for u in us]
    if len(units) != len(set(units)):
        raise SplitError("a level-7 unit lies in two groups")
    if body["closed_stations"] != sorted(s for s, v in st.items() if v["status"] in CLOSED):
        raise SplitError("closed_stations is not the set of test/reserved stations")
    return body


def load(path) -> dict:
    """Read and verify a split file (the only way any reader opens it)."""
    return verify(json.loads(Path(path).read_text()))


def m6_stations(body: dict, roles) -> list:
    """Eligible stations of the given roles — the only stations that ever supply M6 rows."""
    return sorted(s for s, r in body["station_role"].items() if r in tuple(roles))


def min_groups(n_by_group: dict) -> dict:
    """G, G_eff = (sum n_g)^2 / sum n_g^2 and the claim / descriptive status of one endpoint cell.

    ``n_by_group``: group -> number of stations (expected from metadata before the freeze; scored and paired inside
    the look). Applied mechanically; the thresholds are module constants and are never lowered."""
    n = [int(v) for v in n_by_group.values() if int(v) > 0]
    tot, sq = sum(n), sum(v * v for v in n)
    g_eff = (tot * tot / sq) if sq else 0.0
    ok = len(n) >= G_MIN and g_eff >= G_EFF_MIN
    return {"n_stations": tot, "G": len(n), "G_eff": round(g_eff, 2), "status": "claim" if ok else "descriptive"}


# ---- M1's draw (docs/preregistration/sonde_ea_split.json; ea_england_allocation.md, decision of 2026-11-09) ----
# Contract the M6 loader checks (M1 owns the file and its draw rule): the file lists its own ``core_keys`` (they include
# the two keys below), and ``sha256`` is the sha256 of the canonical JSON of exactly those keys (as :func:`core_sha256`).
M1_REQUIRED_KEYS = ("hydromet_ea_split_sha256", "m1_test_groups")
M1_MAX_SHARE = 0.20        # of the groups that are not M6 test-role (lead decision 2026-11-09)


def _sha_of(body: dict, keys) -> str:
    blob = json.dumps({k: body[k] for k in keys}, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("ascii")).hexdigest()


def seal_m1(m6: dict, m1_test_groups, **core_extra) -> dict:
    """An M1 file in the contract's form (used by M1's writer and by the tests); the list may be empty."""
    body = {"hydromet_ea_split_sha256": m6["sha256"], "m1_test_groups": sorted(str(g) for g in m1_test_groups),
            **core_extra}
    body["core_keys"] = sorted(body)
    body["sha256"] = _sha_of(body, body["core_keys"])
    return body


def verify_m1(m1: dict, m6: dict) -> list:
    """M1-test groups of a verified M1 file drawn on this M6 split (the list may be empty; the file must exist)."""
    keys = list(m1.get("core_keys", ()))
    if any(k not in keys for k in M1_REQUIRED_KEYS) or any(k not in m1 for k in keys):
        raise SplitError(f"sonde_ea_split.json: core_keys must exist and include {M1_REQUIRED_KEYS}")
    if _sha_of(m1, keys) != m1.get("sha256"):
        raise SplitError("sonde_ea_split.json sha256 does not match its core")
    if m1["hydromet_ea_split_sha256"] != m6["sha256"]:
        raise SplitError("sonde_ea_split.json was drawn on another M6 England split")
    groups = [str(g) for g in m1["m1_test_groups"]]
    bad = [g for g in groups if m6["role_of_group"].get(g) not in ("train", "val")]
    if bad or len(groups) != len(set(groups)):
        raise SplitError(f"M1-test groups must be distinct M6 train/val-role groups (bad: {bad})")
    n_open = sum(r != "test" for r in m6["role_of_group"].values())
    if len(groups) > M1_MAX_SHARE * n_open:
        raise SplitError(f"{len(groups)} M1-test groups exceed {M1_MAX_SHARE:.0%} of the {n_open} non-test groups")
    return sorted(groups)


def load_pair(m6_path=None, m1_path=None) -> tuple:
    """(verified M6 split, M1-test groups). Both files must exist and verify (lead ruling Q8): no England fit or
    validation number is computed before M6's split is frozen **and** M1's draw is recorded (possibly empty)."""
    m6_path, m1_path = Path(m6_path or SPLIT_FILE), Path(m1_path or M1_SPLIT_FILE)
    for f in (m6_path, m1_path):
        if not f.exists():
            raise SplitError(f"{f} does not exist: no England row is loaded before both split files are frozen")
    m6 = load(m6_path)
    return m6, verify_m1(json.loads(m1_path.read_text()), m6)
