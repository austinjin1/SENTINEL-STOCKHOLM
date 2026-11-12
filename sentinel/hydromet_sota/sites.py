# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Lever site pools for the M6 SOTA campaign (``docs/preregistration/hydromet_sota_protocol.md`` §3).

Every added training location must sit in a FINAL **train-role, non-inner** basin group and must not be
flow-connected to any FINAL val/test-role site or any track-N val/test/excluded station. Rules, in order
(the first that applies wins; ``rule`` column):

1. ``reserved``           — listed in ``sonde_newsites_list.json`` (M1 EW-v3 / M1-N block; :mod:`.reserved`), or the
                            location key contains a reserved USGS number anywhere (regex, as the M1 pre-claim audit);
1b. ``reserved_comid``    — its NHDPlus V2 COMID holds a reserved site, or (WQP stations) a FINAL-split site
                            (architecture-plan data-hygiene rule 3, mandatory per the lead's M6 rulings of 2026-11-10);
2. ``in_final_split``     — already a FINAL-split site (USGS pools only; those are used under their FINAL role);
3. ``new_region``         — its HUC4 is not owned by any FINAL basin group (track N gives such regions val/test roles
                            by hash, so they are never training regions);
4. ``val_group`` / ``test_group`` — the owning FINAL group is val/test-role;
5. ``inner_group``        — the owning group is one of the 13 inner (early-stopping) groups: excluded so that the
                            held-out-group criterion stays identical to the reference runs;
6. ``flow_connected``     — on the NHDPlus V2 downstream main path (≤ 500 km) of a FINAL val/test site or a track-N
                            val/test/excluded station, or one of those on its own downstream path;
7. ``fit``                — usable as a training location (fit groups only), ``basin_group`` = the FINAL group.

Nothing here reads a target value outside the train block; site/day counts of the train block are the only
target-derived quantities (eligibility, as in the FINAL split's own day rule).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..core import paths
from ..data.reserved import assert_no_reserved as shared_assert_no_reserved
from . import reserved as R

TRAIN_START, TRAIN_END = "2007-10-01", "2021-09-30"     # FINAL split train block (WY2008–2021)
WQ_VARS = ("temp_c_mean", "temp_c_max", "do_mgl_mean", "do_mgl_min")
Q_VAR = "discharge_cfs_mean"
MIN_DV_DAYS = 365              # lever 1a: ≥ 365 train-block days of temperature or DO (FINAL split used 730)
MIN_Q_DAYS = 365               # lever 2: ≥ 365 train-block days of discharge
MIN_WEAK_DAYS = 4              # lever 1b: ≥ 4 WQP sample days of the target in the train block
MAX_KM = 500.0
TRACKN_FILE = paths.REPO_ROOT / "docs" / "preregistration" / "hydromet_track_n_stations.json"
TRACKN_SHA256 = "ffb41471c9a75dcbebedf083f273cfa41aae3ffeb7dabe7fd420c5f23120e2d0"
REF_DATA_INFO = paths.RUNS_ROOT / "hydromet" / "ealstm_ws_v1_s0" / "data_info.json"
_ANY_ID = re.compile(r"(?<!\d)(\d{8,15})(?!\d)")


def contains_reserved_id(key: str, res: set[str]) -> bool:
    """True if any 8–15-digit run inside ``key`` is a reserved USGS number (the M1 pre-claim audit regex)."""
    return bool(set(_ANY_ID.findall(str(key))) & res)


class _Stub:
    """Minimal object for :func:`sentinel.models.hydromet.data.inner_split` (sites, roles, basin_group)."""

    def __init__(self, sites, roles, groups):
        self.sites, self.roles, self.basin_group = list(sites), np.asarray(roles), np.asarray(groups)

    def site_idx(self, roles):
        return np.flatnonzero(np.isin(self.roles, list(roles)))


def final_frame():
    """FINAL split as a frame: site_no, role, basin_group, huc4, comid (StreamCat COMID, str), site_key; plus the
    M6 eligible train/val sites and the inner (early-stopping) groups recomputed exactly as ``runner.prepare``."""
    from ..models.hydromet import data as D

    sp, info = D.load_split()
    reg = pd.read_parquet(D.SITES_FILE)
    reg = reg[reg["network"] == "USGS"].drop_duplicates("site_id").set_index("site_id")
    s_all = sorted(sp.site_role)
    ssc = pd.read_parquet(D.STREAMCAT_FILE, columns=["site_key", "comid"]).drop_duplicates("site_key").set_index("site_key")
    keys = reg["site_key"].reindex(s_all)
    df = pd.DataFrame({"site_no": s_all, "role": [sp.site_role[s] for s in s_all],
                       "basin_group": [str(sp.basin_group[s]) for s in s_all],
                       "huc4": reg["huc8"].reindex(s_all).astype(str).str[:4].to_numpy(),
                       "comid": ssc["comid"].reindex(keys).astype("string").to_numpy(),
                       "site_key": keys.to_numpy()})
    elig, _ = D.eligible_sites(sp, ("train", "val"))
    stub = _Stub(elig, [sp.site_role[s] for s in elig], [str(sp.basin_group[s]) for s in elig])
    _, inner_idx = D.inner_split(stub, frac=0.15, seed=0)
    inner_sites = sorted(elig[i] for i in inner_idx)
    inner_groups = sorted({str(sp.basin_group[s]) for s in inner_sites})
    return sp, info, df, elig, inner_sites, inner_groups


def check_inner_matches_reference(inner_sites, ref=REF_DATA_INFO) -> bool:
    """The recomputed inner sites must equal those of the reference run ``ealstm_ws_v1_s0`` (protocol §2)."""
    if not Path(ref).is_file():
        return False
    return sorted(json.loads(Path(ref).read_text())["inner_sites"]) == sorted(inner_sites)


def huc4_owner(final: pd.DataFrame) -> dict:
    """HUC4 -> (basin_group, role) of the FINAL group that owns it (groups are unions of HUC4s)."""
    own = final.groupby("huc4")[["basin_group", "role"]].agg(lambda s: sorted(set(s)))
    out = {}
    for h, r in own.iterrows():
        if len(r["basin_group"]) != 1 or len(r["role"]) != 1:
            raise ValueError(f"HUC4 {h} owned by several FINAL groups {r['basin_group']}")
        out[h] = (r["basin_group"][0], r["role"][0])
    return out


def load_vaa() -> pd.DataFrame:
    vaa = pd.read_parquet(paths.RAW / "nhdplus_vaa" / "nhdplus_vaa_subset.parquet",
                          columns=["comid", "hydroseq", "dnhydroseq", "pathlength"])
    vaa["comid"] = vaa["comid"].astype("int64").astype(str)
    return vaa


def load_trackn_roles() -> dict:
    import hashlib

    body = json.loads(TRACKN_FILE.read_text())
    core = {k: body[k] for k in ("split_sha256", "role_of", "basin_group_of")}
    if hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest() != body["sha256"] or \
            body["sha256"] != TRACKN_SHA256:
        raise ValueError("track-N role file changed (sha256 mismatch)")
    return body


def protected_comids(final: pd.DataFrame, trackn: dict) -> set[str]:
    """COMIDs of FINAL val/test-role sites and of track-N val/test/excluded stations."""
    st = pd.read_parquet(paths.PROCESSED / "hydromet_wqp_national" / "stations.parquet", columns=["station_key", "comid"])
    bad_st = {k for k, r in trackn["role_of"].items() if r in ("val", "test", "excluded")}
    c1 = set(final.loc[final["role"].isin(["val", "test"]), "comid"].dropna().astype(str))
    c2 = set(st.loc[st["station_key"].isin(bad_st), "comid"].dropna().astype(str))
    return c1 | c2


def flow_connected(comids, protected: set[str], vaa: pd.DataFrame, max_km: float = MAX_KM) -> dict[str, bool]:
    """COMID -> True if it lies on the ≤ max_km downstream main path of a protected COMID, a protected COMID lies on
    its own downstream path, or it is itself protected."""
    from ..models.hydromet.track_n import downstream_paths

    comids = [str(c) for c in comids if isinstance(c, str) or pd.notna(c)]
    own = downstream_paths(comids, vaa, max_km)
    prot_paths = downstream_paths(sorted(protected), vaa, max_km)
    below_protected = set()
    for p in prot_paths.values():
        below_protected.update(p)
    return {c: (c in protected) or (c in below_protected) or bool(set(own.get(c, [])) & protected) for c in set(comids)}


def train_block_days(site_nos, variables, *, log=print) -> pd.DataFrame:
    """Per site: number of finite train-block days of each variable (counts only; values are not kept). Refuses any
    M1-reserved site before a file is opened (shared helper + M6 guard)."""
    from ..models.sonde.data import SITE_DIR

    shared_assert_no_reserved(site_nos, where="hydromet_sota.sites.train_block_days")
    R.assert_no_reserved(site_nos)

    rows = []
    for k, s in enumerate(site_nos):
        p = SITE_DIR / f"{s}.parquet"
        rec = {"site_no": s}
        if p.is_file():
            import pyarrow.parquet as pq

            names = set(pq.read_schema(p).names)
            cols = [v for v in variables if v in names]
            if cols:
                df = pd.read_parquet(p, columns=cols)
                df.index = pd.to_datetime(df.index)
                df = df.loc[TRAIN_START:TRAIN_END]
                for v in cols:
                    rec[v] = int(np.isfinite(df[v].to_numpy(dtype="float64")).sum())
        rows.append(rec)
        if k % 500 == 0:
            log(f"[sites] train-block day counts {k}/{len(site_nos)}")
    out = pd.DataFrame(rows).set_index("site_no").reindex(columns=list(variables)).fillna(0).astype(int)
    return out


def reserved_comids(final: pd.DataFrame | None = None) -> set[str]:
    """COMIDs (str) of the M1-reserved sites (site registry + StreamCat COMID); with ``final`` also the COMIDs of every
    FINAL-split site (used for WQP stations, hygiene rule 3)."""
    from ..models.hydromet import data as D

    res = R.reserved_usgs_sites()
    s = pd.read_parquet(D.SITES_FILE)
    s = s[s["network"] == "USGS"].drop_duplicates("site_id")
    r = s[s["site_id"].astype(str).isin(res)]
    sc = pd.read_parquet(D.STREAMCAT_FILE, columns=["site_key", "comid"])
    out = set(r["comid"].dropna().astype("int64").astype(str)) | set(sc.loc[sc["site_key"].isin(r["site_key"]), "comid"]
                                                                       .dropna().astype(str))
    if final is not None:
        out |= set(final["comid"].dropna().astype(str))
    return out


def apply_rules(pool: pd.DataFrame, *, owner: dict, inner_groups, connected: dict, res: set[str],
                final_sites: set[str] | None = None, bad_comids: set[str] | None = None) -> pd.DataFrame:
    """``pool``: key, site_no (USGS number or WQP id), huc4, comid. Adds ``rule``, ``basin_group``, ``group_role``.
    ``bad_comids``: COMIDs that exclude a location (rule 1b)."""
    out = pool.copy()
    bad_comids = bad_comids or set()
    rules, groups, groles = [], [], []
    inner = set(inner_groups)
    for r in out.itertuples(index=False):
        g, gr = owner.get(str(r.huc4), (None, None))
        if R._norm(r.site_no) in res or contains_reserved_id(r.key, res):
            rule = "reserved"
        elif str(r.comid) in bad_comids:
            rule = "reserved_comid"
        elif final_sites is not None and str(r.site_no) in final_sites:
            rule = "in_final_split"
        elif g is None:
            rule = "new_region"
        elif gr in ("val", "test"):
            rule = f"{gr}_group"
        elif g in inner:
            rule = "inner_group"
        elif connected.get(str(r.comid), False):
            rule = "flow_connected"
        else:
            rule = "fit"
        rules.append(rule)
        groups.append(g)
        groles.append(gr)
    out["rule"], out["basin_group"], out["group_role"] = rules, groups, groles
    return out
