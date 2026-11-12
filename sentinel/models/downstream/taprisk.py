# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D4 TapRisk — CWS nitrate (D4a, GW) / DBP (D4b, SW+GU) violation risk (protocol downstream_taprisk.md).

Labels, decision-time features, state-role split and the frame builder. Validation only: test-role rows
(test states, test origin) are never built unless ``testlook.require_authorised`` passes.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ...core import paths
from . import common as C

APP = "taprisk"
SEED = 20261108
ORIGINS = {2010: "train", 2013: "train", 2016: "val", 2019: "test"}
RESERVE_ORIGINS = (2022,)
HORIZON = 3
ROLE_TARGETS = {"test": 0.20, "val": 0.15}
TASKS = {"d4a_gw_nitrate": {"label": "nitrate_mcl", "sources": ("GW",), "group": "nitrate", "ambient": ("no3", "no3no2")},
         "d4b_sw_dbp": {"label": "dbp_mcl", "sources": ("SW", "GU"), "group": "dbp", "ambient": ("doc",)},
         "sw_nitrate_descriptive": {"label": "nitrate_mcl", "sources": ("SW", "GU"), "group": "nitrate", "ambient": ("no3", "no3no2")}}
DATA_MANIFESTS = ("labels_sdwis", "labels_cws_boundaries", "taprisk_catchment", "wqp_discrete", "wbd_huc12")
SPLIT_SHA = "a4a1fc7bb5f997d0a656c527c884ed4691c59da341eff81d0d424bbd3e5b3298"   # frozen 2026-11-08 (protocol §9)

SDWIS = paths.PROCESSED / "labels_sdwis"
TC = paths.PROCESSED / "taprisk_catchment"


# --------------------------------------------------------------------------- systems and split
def load_systems() -> pd.DataFrame:
    pts = pd.read_parquet(TC / "system_points.parquet")
    s = pd.read_parquet(SDWIS / "sdwis_systems.parquet",
                        columns=["pwsid", "owner_type_code", "service_connections", "is_wholesaler_ind",
                                 "dbpr_schedule_cat_code", "pop_cat_5_code"])
    out = pts.merge(s, on="pwsid", how="left")
    out["deact"] = pd.to_datetime(out["pws_deactivation_date"], errors="coerce")
    out["first"] = pd.to_datetime(out["first_reported_date"], errors="coerce")
    return out


def _sha(s):
    return hashlib.sha1(f"{SEED}:{s}".encode()).hexdigest()


def state_roles(systems: pd.DataFrame) -> dict:
    """State -> role from SYSTEM COUNTS only (no labels), largest first to the role furthest below target."""
    counts = systems.groupby("state_code").size()
    total = float(counts.sum())
    target = {**ROLE_TARGETS, "train": 1 - sum(ROLE_TARGETS.values())}
    got = {k: 0.0 for k in target}
    roles = {}
    for st in sorted(counts.index, key=lambda g: (-counts[g], _sha(g))):
        r = max(target, key=lambda k: (target[k] - got[k] / total, k == "train"))
        roles[str(st)] = r
        got[r] += counts[st]
    return roles


def freeze() -> str:
    from ...core.registry import file_sha256

    sy = load_systems()
    roles = state_roles(sy)
    meta = {"unit": "CWS with own source (GW / SW / GU), SAB v3 polygon representative point",
            "origins": {str(k): v for k, v in ORIGINS.items()}, "reserve_origins": list(RESERVE_ORIGINS),
            "horizon_years": HORIZON, "role_targets": ROLE_TARGETS, "seed": SEED,
            "row_rule": "train = train states x T in {2010, 2013}; val_known = train states x 2016; val_new = val states x "
                        "2016; test_known = train+val states x 2019; test_new = test states x 2019",
            "system_points_sha256": file_sha256(TC / "system_points.parquet"),
            "sdwis_violations_sha256": file_sha256(SDWIS / "sdwis_violations.parquet"),
            "sdwis_systems_sha256": file_sha256(SDWIS / "sdwis_systems.parquet"), "freeze_date": "2026-11-08"}
    return C.freeze_split(APP, {"state_role": roles}, meta)


# --------------------------------------------------------------------------- violations
def load_violations() -> pd.DataFrame:
    v = pd.read_parquet(SDWIS / "sdwis_violations.parquet",
                        columns=["pwsid", "label", "label_group", "is_health_based", "is_monitoring_reporting",
                                 "compl_per_begin_date", "compl_per_end_date", "viol_first_reported_date"])
    for c in ("compl_per_begin_date", "compl_per_end_date", "viol_first_reported_date"):
        v[c] = pd.to_datetime(v[c], errors="coerce")
    v["end"] = v["compl_per_end_date"].fillna(v["compl_per_begin_date"] + pd.Timedelta(days=90))
    return v


def labels_at(v: pd.DataFrame, T: int, label: str, report_cutoff: int | None = None) -> set:
    """Systems with a task-label health-based violation overlapping the window T+1..T+3. ``report_cutoff`` (protocol
    §11 C6): only violations first reported by 31 Dec of that year count (fit rows of a scored origin S)."""
    lo, hi = pd.Timestamp(f"{T + 1}-01-01"), pd.Timestamp(f"{T + HORIZON}-12-31")
    m = (v["label"] == label) & v["is_health_based"].astype(bool) & (v["compl_per_begin_date"] <= hi) & (v["end"] >= lo)
    if report_cutoff is not None:
        m &= v["viol_first_reported_date"].notna() & (v["viol_first_reported_date"] <= pd.Timestamp(f"{report_cutoff}-12-31"))
    return set(v.loc[m, "pwsid"])


def history_at(v: pd.DataFrame, T: int, label: str, group: str) -> pd.DataFrame:
    """Only violations FIRST REPORTED by 31 Dec T (known on the decision date)."""
    cut = pd.Timestamp(f"{T}-12-31")
    k = v[v["viol_first_reported_date"].notna() & (v["viol_first_reported_date"] <= cut)]
    hb = k[k["is_health_based"].astype(bool)]
    task = hb[hb["label"] == label]
    yr = task["compl_per_begin_date"].dt.year
    out = pd.DataFrame(index=pd.Index(sorted(set(k["pwsid"])), name="pwsid"))
    out["hist_task_3y"] = task[yr >= T - 2].groupby("pwsid").size()
    out["hist_task_10y"] = task[yr >= T - 9].groupby("pwsid").size()
    last = task.groupby("pwsid")["compl_per_begin_date"].max()
    out["hist_years_since_task"] = (cut - last).dt.days / 365.25
    other = hb[(hb["label"] != label) & (hb["compl_per_begin_date"].dt.year >= T - 2)]
    out["hist_other_hb_3y"] = other.groupby("pwsid").size()
    mr = k[k["is_monitoring_reporting"].astype(bool) & (k["label_group"] == group) & (k["compl_per_begin_date"].dt.year >= T - 2)]
    out["hist_mr_group_3y"] = mr.groupby("pwsid").size()
    for c in ("hist_task_3y", "hist_task_10y", "hist_other_hb_3y", "hist_mr_group_3y"):
        out[c] = out[c].fillna(0.0)
    return out


# --------------------------------------------------------------------------- ambient water quality (observed, <= T)
def ambient_huc8(variables, T: int, years_back: int = 9) -> pd.DataFrame:
    out = []
    for var in variables:
        d = pd.read_parquet(paths.PROCESSED / "wqp_discrete", filters=[("variable", "=", var)],
                            columns=["huc8", "activity_start_date", "value", "site_class"])
        yr = pd.to_datetime(d["activity_start_date"], errors="coerce").dt.year
        d = d[(yr >= T - years_back) & (yr <= T) & d["huc8"].notna() & (d["huc8"] != "") & d["value"].notna()]
        out.append(d)
    d = pd.concat(out, ignore_index=True)
    sw = d[d["site_class"].isin(["stream", "lake"])].groupby(d["huc8"].astype(str))["value"].median()
    gw = d[d["site_class"] == "groundwater"].groupby(d["huc8"].astype(str))["value"].median()
    return pd.DataFrame({"amb_sw_median": sw, "amb_gw_median": gw})


# --------------------------------------------------------------------------- frame
SYS_FEATURES = ["log_pop", "log_conn", "owner_federal", "owner_state", "owner_local", "owner_private", "owner_public_private",
                "is_wholesaler", "dbpr_cat", "is_gu", "is_modeled_boundary"]
HIST_FEATURES = ["hist_task_3y", "hist_task_10y", "hist_years_since_task", "hist_other_hb_3y", "hist_mr_group_3y"]
AMB_FEATURES = ["amb_sw_median", "amb_gw_median"]


def system_features(sy: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=sy.index)
    f["log_pop"] = np.log10(1 + pd.to_numeric(sy["population_served"], errors="coerce"))
    f["log_conn"] = np.log10(1 + pd.to_numeric(sy["service_connections"], errors="coerce"))
    for code, name in (("F", "federal"), ("S", "state"), ("L", "local"), ("P", "private"), ("M", "public_private")):
        f[f"owner_{name}"] = (sy["owner_type_code"] == code).astype(float)
    f["is_wholesaler"] = (sy["is_wholesaler_ind"].astype(str).str.upper() == "Y").astype(float)
    f["dbpr_cat"] = pd.to_numeric(sy["dbpr_schedule_cat_code"].astype(str).str.extract(r"(\d)")[0], errors="coerce")
    f["is_gu"] = (sy["primary_source_code"] == "GU").astype(float)
    f["is_modeled_boundary"] = sy["is_modeled"].astype(float)
    return f


def build(task: str, split: dict, *, include_test: bool = False, split_sha: str | None = None,
          with_streamcat: bool = True, scored_origin: int | None = None) -> pd.DataFrame:
    """One row per candidate system × origin (train/val origins; + test origin only in an authorised look).
    ``scored_origin`` S (protocol §11 C6): labels of rows with origin < S use only violations first reported by
    31 Dec S; rows of origin S keep uncensored labels (outcomes)."""
    if include_test:
        from .taprisk_look import require_open

        require_open(split_sha)
    cfg = TASKS[task]
    roles = split["assignments"]["state_role"]
    sy = load_systems()
    sy = sy[sy["primary_source_code"].isin(cfg["sources"])].reset_index(drop=True)
    from .wqp_track import huc_of_points

    h = huc_of_points(sy["lat"].to_numpy(float), sy["lon"].to_numpy(float))
    sy["huc8"], sy["huc4"] = h["huc8"].to_numpy(), h["huc4"].to_numpy()
    sf = system_features(sy)
    v = load_violations()
    v = v[v["pwsid"].isin(set(sy["pwsid"]))]
    frames = []
    for T, orole in ORIGINS.items():
        if orole == "test" and not include_test:
            continue
        day = pd.Timestamp(f"{T + 1}-01-01")
        alive = (sy["deact"].isna() | (sy["deact"] >= day)) & (sy["first"].isna() | (sy["first"] < day))
        srole = sy["state_code"].astype(str).map(roles)
        if orole == "train":
            keep, evalset = srole == "train", None
        elif orole == "val":
            keep = srole.isin(["train", "val"])
            evalset = np.where(srole == "train", "val_known", "val_new")
        else:
            keep = srole.notna()
            evalset = np.where(srole == "test", "test_new", "test_known")
        m = (alive & keep).to_numpy()
        f = sy.loc[m, ["pwsid", "state_code", "primary_source_code", "huc8", "huc4", "lat", "lon"]].copy()
        f = pd.concat([f, sf.loc[m]], axis=1)
        f["origin"], f["role"] = T, orole
        f["evalset"] = None if evalset is None else np.asarray(evalset)[m]
        cut = scored_origin if (scored_origin is not None and T < scored_origin) else None
        f["y"] = f["pwsid"].isin(labels_at(v, T, cfg["label"], report_cutoff=cut)).astype(float)
        f = f.merge(history_at(v, T, cfg["label"], cfg["group"]), left_on="pwsid", right_index=True, how="left")
        f[["hist_task_3y", "hist_task_10y", "hist_other_hb_3y", "hist_mr_group_3y"]] = \
            f[["hist_task_3y", "hist_task_10y", "hist_other_hb_3y", "hist_mr_group_3y"]].fillna(0.0)
        amb = ambient_huc8(cfg["ambient"], T)
        f = f.merge(amb, left_on="huc8", right_index=True, how="left")
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    if not include_test:
        C.assert_no_test(df)
    if with_streamcat:
        df = join_streamcat(df)
    from .crossfit import phase1_no_upstream

    phase1_no_upstream(df.columns)
    return df


def streamcat_columns() -> list[str]:
    sc = pd.read_parquet(TC / "streamcat_comid.parquet")
    return [c for c in sc.columns if c not in ("comid", "streamcat_found")]


def join_streamcat(df: pd.DataFrame) -> pd.DataFrame:
    links = pd.read_parquet(TC / "system_links.parquet", columns=["pwsid", "comid", "snap_status"])
    sc = pd.read_parquet(TC / "streamcat_comid.parquet")
    sc["comid"] = sc["comid"].astype(str)
    links["comid"] = links["comid"].astype("string")
    out = df.merge(links, on="pwsid", how="left").merge(sc.drop(columns=["streamcat_found"]), on="comid", how="left")
    return out


def power_check(df: pd.DataFrame, auc: float = 0.75, rho: float = 0.6) -> dict:
    out = {}
    for key, g in list(df.groupby("evalset")) + [("train", df[df["role"] == "train"])]:
        npos, nneg = int(g["y"].sum()), int((1 - g["y"]).sum())
        q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
        se = np.sqrt((auc * (1 - auc) + (npos - 1) * (q1 - auc ** 2) + (nneg - 1) * (q2 - auc ** 2)) / max(npos * nneg, 1))
        out[str(key)] = {"n": int(len(g)), "n_pos": npos, "prevalence": float(g["y"].mean()) if len(g) else None,
                         "states": int(g["state_code"].nunique()), "huc8": int(g["huc8"].nunique()),
                         "auroc_se_at_0.75": float(se), "paired_mde_auroc": float(2.8 * se * np.sqrt(2 * (1 - rho)))}
    return out
