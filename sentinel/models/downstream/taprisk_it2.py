# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D4 TapRisk it2 (protocol §10 it2 spec): richer pre-decision violation history, SDWIS source/treatment
proxies, the parity comparator ``pennino_rf_hist_rich`` and an inner-fold RF+GBM stack. Validation only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import common as C
from . import taprisk as TR
from . import taprisk_models as TM
from . import wqp_track as WT

MCL = {"nitrate_mcl": {None: 10.0}, "dbp_mcl": {"2950": 0.080, "2456": 0.060}}


def _violations_full() -> pd.DataFrame:
    v = pd.read_parquet(TR.SDWIS / "sdwis_violations.parquet",
                        columns=["pwsid", "violation_id", "label", "label_group", "is_health_based", "is_monitoring_reporting",
                                 "compl_per_begin_date", "viol_first_reported_date", "viol_measure_mgl", "contaminant_code",
                                 "public_notification_tier", "n_enf_formal"])
    for c in ("compl_per_begin_date", "viol_first_reported_date"):
        v[c] = pd.to_datetime(v[c], errors="coerce")
    return v


ENF_SCAN = TR.paths.INTERIM / "sdwis" / "violations_enforcement_scan.parquet"
_ENF = None


def formal_enforcement() -> pd.DataFrame:
    """Dated formal enforcement actions per violation (protocol §11 C5): pwsid, violation_id, enforcement date and
    first-reported date of the action."""
    global _ENF
    if _ENF is None:
        e = pd.read_parquet(ENF_SCAN, columns=["PWSID", "VIOLATION_ID", "ENFORCEMENT_ID", "ENFORCEMENT_DATE",
                                               "ENF_ACTION_CATEGORY", "ENF_FIRST_REPORTED_DATE"])
        e = e[e["ENF_ACTION_CATEGORY"].astype(str).str.lower() == "formal"].drop_duplicates(["PWSID", "VIOLATION_ID", "ENFORCEMENT_ID"])
        e = e.rename(columns={"PWSID": "pwsid", "VIOLATION_ID": "violation_id"})
        e["enf_date"] = pd.to_datetime(e["ENFORCEMENT_DATE"], errors="coerce")
        e["enf_rep"] = pd.to_datetime(e["ENF_FIRST_REPORTED_DATE"], errors="coerce")
        e["violation_id"] = e["violation_id"].astype(str)
        _ENF = e[["pwsid", "violation_id", "enf_date", "enf_rep"]]
    return _ENF


def rich_history(v: pd.DataFrame, T: int, label: str, group: str) -> pd.DataFrame:
    cut = pd.Timestamp(f"{T}-12-31")
    k = v[v["viol_first_reported_date"].notna() & (v["viol_first_reported_date"] <= cut)].copy()
    k["yr"] = k["compl_per_begin_date"].dt.year
    hb = k[k["is_health_based"].astype(bool)]
    task = hb[hb["label"] == label]
    out = pd.DataFrame(index=pd.Index(sorted(set(k["pwsid"])), name="pwsid"))
    for w in (1, 3, 5, 10):
        out[f"rh_task_{w}y"] = task[task["yr"] >= T - w + 1].groupby("pwsid").size()
    mcl = task["contaminant_code"].astype(str).map(MCL[label]) if label == "dbp_mcl" else pd.Series(10.0, index=task.index)
    sev = (pd.to_numeric(task["viol_measure_mgl"], errors="coerce") / mcl).where(task["yr"] >= T - 4)
    out["rh_task_severity_5y"] = sev.groupby(task["pwsid"]).max()
    t5 = task[task["yr"] >= T - 4]
    out["rh_task_pn_tier_5y"] = pd.to_numeric(t5["public_notification_tier"], errors="coerce").groupby(t5["pwsid"]).max()
    e = formal_enforcement()
    e = e[e["enf_date"].notna() & (e["enf_date"] <= cut) & (e["enf_rep"].isna() | (e["enf_rep"] <= cut))]
    t5k = t5.assign(violation_id=t5["violation_id"].astype(str))[["pwsid", "violation_id"]]
    n_enf = t5k.merge(e, on=["pwsid", "violation_id"], how="inner").groupby("pwsid").size()
    out["rh_task_enf_formal_5y"] = n_enf      # only actions dated (and first reported) <= 31 Dec T
    mr = k[k["is_monitoring_reporting"].astype(bool) & (k["label_group"] == group)]
    for w in (1, 3, 5):
        out[f"rh_mr_{w}y"] = mr[mr["yr"] >= T - w + 1].groupby("pwsid").size()
    oth = hb[hb["label"] != label]
    for w in (3, 10):
        out[f"rh_other_hb_{w}y"] = oth[oth["yr"] >= T - w + 1].groupby("pwsid").size()
    fill0 = [c for c in out.columns if c not in ("rh_task_severity_5y", "rh_task_pn_tier_5y")]
    out[fill0] = out[fill0].fillna(0.0)
    return out


def source_proxies(T: int) -> pd.DataFrame:
    f = pd.read_parquet(TR.SDWIS / "sdwis_source_facilities.parquet",
                        columns=["pwsid", "facility_type_code", "facility_activity_code", "facility_deactivation_date",
                                 "availability_code", "seller_treatment_code", "seller_pwsid", "filtration_status_code",
                                 "is_source_treated_ind", "first_reported_date"])
    day = pd.Timestamp(f"{T + 1}-01-01")
    de = pd.to_datetime(f["facility_deactivation_date"], errors="coerce")
    fr = pd.to_datetime(f["first_reported_date"], errors="coerce")
    f = f[(de.isna() | (de >= day)) & (fr.isna() | (fr < day))]
    g = f.groupby("pwsid")
    out = pd.DataFrame(index=g.size().index)
    out["sp_n_sources"] = g.size()
    for code in ("WL", "IN", "RS", "SP", "CC"):
        out[f"sp_n_{code.lower()}"] = (f["facility_type_code"] == code).groupby(f["pwsid"]).sum()
    out["sp_share_treated"] = f["is_source_treated_ind"].astype(float).groupby(f["pwsid"]).mean()
    out["sp_any_unfiltered"] = f["filtration_status_code"].astype(str).str.upper().str.startswith("U").groupby(f["pwsid"]).max().astype(float)
    out["sp_share_purchased"] = out["sp_n_cc"] / out["sp_n_sources"]
    out["sp_any_treated_purchase"] = f["seller_treatment_code"].notna().groupby(f["pwsid"]).max().astype(float)
    for a in ("P", "E", "I", "O"):
        out[f"sp_avail_{a.lower()}"] = (f["availability_code"] == a).groupby(f["pwsid"]).mean()
    return out


def build_rich(task: str, split: dict, *, include_test: bool = False, split_sha: str | None = None,
               scored_origin: int | None = None) -> pd.DataFrame:
    cfg = TR.TASKS[task]
    df = TR.build(task, split, include_test=include_test, split_sha=split_sha, scored_origin=scored_origin)
    if not include_test:
        C.assert_no_test(df)
    v = _violations_full()
    v = v[v["pwsid"].isin(set(df["pwsid"]))]
    parts = []
    for T in sorted(df["origin"].unique()):
        sub = df[df["origin"] == T]
        rh = rich_history(v, int(T), cfg["label"], cfg["group"])
        sp = source_proxies(int(T))
        parts.append(sub.merge(rh, left_on="pwsid", right_index=True, how="left")
                        .merge(sp, left_on="pwsid", right_index=True, how="left"))
    out = pd.concat(parts, ignore_index=True)
    for c in [c for c in out.columns if c.startswith(("rh_task_", "rh_mr_", "rh_other_")) and c not in
              ("rh_task_severity_5y", "rh_task_pn_tier_5y")]:
        out[c] = out[c].fillna(0.0)
    return out


def rich_cols(df) -> list[str]:
    return [c for c in df.columns if c.startswith(("rh_", "sp_"))]


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def fit_all(task: str, df: pd.DataFrame, fit_roles, p1: dict, *, n_est: int | None = None):
    """Fit every it2 model and comparator on the fit rows (frozen hyperparameters; protocol §11). Returns a scorer
    ``f(frame) -> {name: scores}`` and the per-seed scorer for the GBM-based candidates."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.neighbors import BallTree

    sel = p1["selection"]
    gpar = sel["gbm_challenger"]["params"]
    gn = int(n_est if n_est is not None else sel["gbm_challenger"]["n_estimators"])
    fr = df[df["role"].isin(fit_roles)].reset_index(drop=True)
    y = fr["y"].to_numpy(float)
    sc_cols = TR.streamcat_columns()
    histf = sc_cols + TR.HIST_FEATURES + TR.SYS_FEATURES + TR.AMB_FEATURES
    allf = histf + ["lat", "lon"]
    rich = histf + rich_cols(df) + ["lat", "lon"]          # W3: parity RF and stack RF get lat/lon too

    def topk_rf(cols, k):
        imp = TM.BalancedRF(seed=0, cols=cols).fit(fr, y).importances().sort_values(ascending=False)
        return TM.BalancedRF(seed=0, cols=list(imp.index) if k is None else list(imp.index[:k])).fit(fr, y)

    rf = topk_rf(sc_cols, sel["pennino_rf"]["topk"])
    rfh = topk_rf(histf, sel["pennino_rf_hist"]["topk"])
    rfr = TM.BalancedRF(seed=0, cols=rich).fit(fr, y)
    g1 = [WT.fit_gbm(fr[allf], y, gpar, s, n_estimators=gn) for s in TM.SEEDS]      # phase-1 GBM, censored labels
    gr = [WT.fit_gbm(fr[rich], y, gpar, s, n_estimators=gn) for s in TM.SEEDS]
    oof = np.zeros((len(fr), 2))
    for a, b in GroupKFold(5).split(fr, y, groups=fr["state_code"].astype(str)):
        oof[b, 0] = WT.fit_gbm(fr.loc[a, rich], y[a], gpar, 0, n_estimators=gn).predict_proba(fr.loc[b, rich])[:, 1]
        oof[b, 1] = TM.BalancedRF(seed=0, cols=rich).fit(fr.loc[a], y[a]).predict(fr.loc[b])
    meta = LogisticRegression(C=1.0, max_iter=1000).fit(_logit(oof), y)
    p0 = y.mean()
    st = fr.groupby("state_code")["y"].agg(["sum", "size"])
    st_rate = (st["sum"] + 50 * p0) / (st["size"] + 50)
    size_cols = ["log_pop", "log_conn", "owner_federal", "owner_state", "owner_local", "owner_private", "owner_public_private", "is_gu"]
    fsz = WT.fit_logistic(fr[size_cols], y, 1.0)
    tree = BallTree(np.radians(fr[["lat", "lon"]].to_numpy(float)), metric="haversine")
    amb = "amb_gw_median" if task.endswith("nitrate") else "amb_sw_median"
    amb_med = np.nanmedian(fr[amb].to_numpy(float))

    def score(s: pd.DataFrame) -> tuple[dict, dict]:
        pg1 = [g.predict_proba(s[allf])[:, 1] for g in g1]
        pgr = [g.predict_proba(s[rich])[:, 1] for g in gr]
        r_rich = rfr.predict(s)
        stack = [meta.predict_proba(_logit(np.column_stack([q, r_rich])))[:, 1] for q in pgr]
        _, idx = tree.query(np.radians(s[["lat", "lon"]].to_numpy(float)), k=25)
        out = {"gbm_phase1": np.mean(pg1, axis=0), "gbm_rich": np.mean(pgr, axis=0),
               "stack_rf_gbm": meta.predict_proba(_logit(np.column_stack([np.mean(pgr, axis=0), r_rich])))[:, 1],
               "pennino_rf": rf.predict(s), "pennino_rf_hist": rfh.predict(s), "pennino_rf_hist_rich": r_rich,
               "location_knn25": y[idx].mean(axis=1),
               "persistence": (s["hist_task_3y"] + 0.01 * s["hist_task_10y"]).to_numpy(float),
               "state_rate": s["state_code"].map(st_rate).fillna(p0).to_numpy(float),
               "ambient_rule": s[amb].fillna(amb_med).to_numpy(float), "size_source_logit": fsz(s[size_cols])}
        return out, {"gbm_phase1": pg1, "gbm_rich": pgr, "stack_rf_gbm": stack}

    return score, {"gbm_params": gpar, "n_estimators": gn, "meta_coef": meta.coef_.tolist(), "n_fit": int(len(fr)),
                   "n_fit_pos": int(y.sum())}


def run(task: str, split: dict, sha: str) -> dict:
    """it2 (corrected, protocol §11): C5 enforcement dates, C6 censored fit labels (scored origin 2016), parity RF
    with lat/lon; every comparator refit under the same labels with its frozen phase-1 hyperparameters."""
    import json

    from sentinel.core.runlog import TrainingRun

    p1 = json.loads((C.run_dir(TR.APP, f"phase1_{task}") / "phase1_results.json").read_text())
    df = build_rich(task, split, scored_origin=2016)
    C.assert_no_test(df)
    out_dir = C.run_dir(TR.APP, f"it2c_{task}")
    for name in ("gbm_rich", "pennino_rf_hist_rich", "stack_rf_gbm"):
        C.log_config_tried(TR.APP, f"{task}/{name}@it2c", {"correction": "protocol §11 C5/C6/W3 re-run of it2"},
                           status="correction (not a new configuration)")
    with TrainingRun(C.WORKSTREAM, f"{TR.APP}/it2c/{task}", config={"split_sha256": sha, "task": task},
                     data=list(TR.DATA_MANIFESTS), plan_ref=C.PLAN_REF, artifacts_dir=str(out_dir)):
        scorer, info = fit_all(task, df, ("train",), p1)
        res = {"task": task, "split_sha256": sha, "fit": info, "sets": {}}
        for es in ("val_known", "val_new"):
            s = df[df["evalset"] == es].reset_index(drop=True)
            st, per = scorer(s)
            yy, g8 = s["y"].to_numpy(float), s["huc8"].astype(str).to_numpy()
            simple = p1["sets"][es]["strongest_simple_baseline"]
            r = {"n": int(len(s)), "n_pos": int(yy.sum()), "strongest_simple": simple,
                 "auroc": {k: C.fast_auroc(yy, v) for k, v in st.items()}, "paired": {}}
            for ch in ("gbm_phase1", "gbm_rich", "stack_rf_gbm"):
                for b in ("pennino_rf", "pennino_rf_hist", "pennino_rf_hist_rich", simple):
                    d = C.paired(C.fast_auroc, yy, st[ch], st[b], g8)
                    d["per_seed_diff"] = [C.fast_auroc(yy, q) - C.fast_auroc(yy, st[b]) for q in per[ch]]
                    r["paired"][f"{ch}-{b}"] = d
                    C.log_metric(app=TR.APP, experiment=f"{task}/{es}@it2c", model=ch, metric=f"auroc_diff_vs_{b}",
                                 value=d["diff"], split=f"{es}_origin2016", n=int(len(s)), ci=d,
                                 ci_method="paired cluster bootstrap over HUC8, B=2000", data_manifests=TR.DATA_MANIFESTS,
                                 extra={"p_value": d["p_value"]})
            for k in st:
                C.log_metric(app=TR.APP, experiment=f"{task}/{es}@it2c", model=k, metric="auroc", value=r["auroc"][k],
                             split=f"{es}_origin2016", n=int(len(s)), is_baseline=k not in ("gbm_phase1", "gbm_rich", "stack_rf_gbm"),
                             data_manifests=TR.DATA_MANIFESTS)
            res["sets"][es] = r
            pd.DataFrame({"pwsid": s["pwsid"], "state_code": s["state_code"], "y": yy, "huc8": g8,
                          **{f"s_{k}": v for k, v in st.items()}}).to_parquet(out_dir / f"val_predictions_{es}.parquet", index=False)
        res["selection"] = select(res)
        C.write_json(out_dir / "it2_results.json", res)
    return res


def select(res: dict) -> dict:
    """Pre-registered it2 selection rule (D4a): qualify = beats all four comparators on both sets with Holm and
    per-seed Δ > 0; among qualifiers the largest minimum margin; else the highest mean validation AUROC."""
    from sentinel.eval.bootstrap import holm_bonferroni

    out = {}
    for ch in ("gbm_phase1", "gbm_rich", "stack_rf_gbm"):
        pv, diffs, seeds_ok = {}, [], True
        for es, r in res["sets"].items():
            for k, d in r["paired"].items():
                if k.startswith(ch + "-"):
                    pv[f"{es}|{k}"] = d["p_value"]
                    diffs.append(d["diff"])
                    seeds_ok &= d.get("per_seed_diff") is not None and all(x > 0 for x in d["per_seed_diff"])
        h = holm_bonferroni(pv)
        q = all(h.values()) and min(diffs) > 0 and seeds_ok
        out[ch] = {"qualifies": bool(q), "min_margin": float(min(diffs)),
                   "mean_auroc": float(np.mean([r["auroc"][ch] for r in res["sets"].values()]))}
    quals = [c for c, v in out.items() if v["qualifies"]]
    chosen = max(quals, key=lambda c: out[c]["min_margin"]) if quals else max(out, key=lambda c: out[c]["mean_auroc"])
    return {"candidates": out, "chosen": chosen, "rule": "qualifier with largest min margin" if quals else
            "no qualifier -> highest mean val AUROC (expected test label: competitive)"}
