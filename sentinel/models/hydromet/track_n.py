# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M6 track N — national unsensed reaches (protocol §12): WQP discrete stream temperature / DO samples at
stations without any continuous sensor, roles assigned consistently with the frozen FINAL sonde split.

Role rule (per station; ``assign_roles``):
* candidate roles = roles of the FINAL-split basin groups that own the station's HUC4, plus roles of FINAL-split
  sites flow-connected to the station (station on a site's NHDPlus V2 downstream main path or vice versa, ≤ 500 km);
* candidates = {test} -> ``test``; test among several candidates -> ``excluded`` (never used); {val} -> ``val``;
  {train} -> ``train`` (seen region; secondary in-region analysis only); {train, val} -> ``excluded``;
* no candidate (a region without any sonde site) -> a *new-region* group: HUC4 units merged over station-station
  flow connections; the whole group is ``val`` or ``test`` by sha1 of its smallest HUC4 code (even -> val).
Basin group (resampling unit) = the FINAL basin group, or ``N:<HUC4…>`` for new-region groups. No model is ever
fitted on track-N stations; models come from the original track (FINAL train-role sites × train block).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

MAX_KM = 500.0
TRACK_N_FILE = "docs/preregistration/hydromet_track_n_stations.json"


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def downstream_paths(start_comids, vaa: pd.DataFrame, max_km: float = MAX_KM) -> dict[str, list[str]]:
    """COMID -> COMIDs on its downstream main path (``dnhydroseq`` chain) within ``max_km`` of path length."""
    hs = dict(zip(vaa["comid"], vaa["hydroseq"]))
    by_hs = dict(zip(vaa["hydroseq"], vaa["comid"]))
    dn = dict(zip(vaa["comid"], vaa["dnhydroseq"]))
    pl = dict(zip(vaa["comid"], vaa["pathlength"]))
    out = {}
    for c in set(start_comids):
        if c not in hs:
            out[c] = []
            continue
        path, cur, p0 = [], c, pl.get(c, np.nan)
        for _ in range(100000):
            nxt = by_hs.get(dn.get(cur))
            if nxt is None or nxt == cur:
                break
            if np.isfinite(p0) and np.isfinite(pl.get(nxt, np.nan)) and p0 - pl[nxt] > max_km:
                break
            path.append(nxt)
            cur = nxt
        out[c] = path
    return out


def _hash_role(huc4: str) -> str:
    return "val" if int(hashlib.sha1(huc4.encode()).hexdigest(), 16) % 2 == 0 else "test"


def assign_roles(stations: pd.DataFrame, split_sites: pd.DataFrame, vaa: pd.DataFrame) -> pd.DataFrame:
    """``stations``: station_key, comid, huc8. ``split_sites``: site_no, comid (may be NaN), huc8, role,
    basin_group (all FINAL-split sites). ``vaa``: comid, hydroseq, dnhydroseq, pathlength (comid as str)."""
    st = stations.copy()
    st["huc4"] = st["huc8"].astype(str).str[:4]
    ss = split_sites.copy()
    ss["huc4"] = ss["huc8"].astype(str).str[:4]
    huc_groups = ss.groupby("huc4").apply(lambda d: sorted(set(zip(d["basin_group"].astype(str), d["role"]))),
                                          include_groups=False).to_dict()
    site_comid = ss.dropna(subset=["comid"]).groupby("comid").apply(
        lambda d: sorted(set(zip(d["basin_group"].astype(str), d["role"]))), include_groups=False).to_dict()
    st_paths = downstream_paths(st["comid"].dropna().unique(), vaa)
    ss_paths = downstream_paths(list(site_comid), vaa)
    on_site_path = {}   # station comid -> (group, role) of split sites whose downstream path contains it
    for sc, path in ss_paths.items():
        for c in path:
            on_site_path.setdefault(c, set()).update(site_comid[sc])
    st_comids = set(st["comid"].dropna())
    # station-station connections (for merging new-region groups)
    uf = _UF()
    for c, path in st_paths.items():
        for d in path:
            if d in st_comids:
                uf.union(c, d)
    rows = []
    for r in st.itertuples(index=False):
        cand = set(huc_groups.get(r.huc4, []))
        for c in [r.comid] + st_paths.get(r.comid, []):
            cand |= set(site_comid.get(c, []))
        cand |= on_site_path.get(r.comid, set())
        roles = {x[1] for x in cand}
        groups = sorted({x[0] for x in cand})
        if not roles:
            role, group, why = None, None, "new_region"
        elif roles == {"test"}:
            role, group, why = "test", groups[0], "split_test"
        elif "test" in roles:
            role, group, why = "excluded", None, "touches_test"
        elif roles == {"val"}:
            role, group, why = "val", groups[0], "split_val"
        elif roles == {"train"}:
            role, group, why = "train", groups[0], "split_train"
        else:
            role, group, why = "excluded", None, "train_and_val"
        rows.append({"station_key": r.station_key, "role": role, "basin_group": group, "rule": why,
                     "n_candidate_groups": len(groups), "huc4": r.huc4, "comid": r.comid})
    out = pd.DataFrame(rows)
    # new-region groups: HUC4 merged over station flow connections
    nr = out["rule"] == "new_region"
    huf = _UF()
    comp_huc = {}
    for r in out[nr].itertuples(index=False):
        huf.find(r.huc4)
        comp_huc.setdefault(uf.find(r.comid), set()).add(r.huc4)
    for hs_ in comp_huc.values():
        hs_ = sorted(hs_)
        for h in hs_[1:]:
            huf.union(hs_[0], h)
    members = {}
    for h in out.loc[nr, "huc4"].unique():
        members.setdefault(huf.find(h), set()).add(h)
    gname = {h: "N:" + "+".join(sorted(members[huf.find(h)])) for h in out.loc[nr, "huc4"].unique()}
    out.loc[nr, "basin_group"] = out.loc[nr, "huc4"].map(gname)
    out.loc[nr, "role"] = out.loc[nr, "huc4"].map(lambda h: _hash_role(min(members[huf.find(h)])))
    return out


def payload(roles: pd.DataFrame, split_sha256: str, data_manifest_sha: dict) -> dict:
    r = roles.sort_values("station_key")
    body = {"track": "M6 track N (national unsensed reaches, WQP discrete)", "split_sha256": split_sha256,
            "data": data_manifest_sha,
            "role_of": dict(zip(r["station_key"], r["role"])),
            "basin_group_of": dict(zip(r["station_key"], r["basin_group"].astype(object).where(r["basin_group"].notna(), None))),
            "rule_of": dict(zip(r["station_key"], r["rule"]))}
    core = {k: body[k] for k in ("split_sha256", "role_of", "basin_group_of")}
    body["sha256"] = hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()
    return body
