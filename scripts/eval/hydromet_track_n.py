#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M6 track N — national unsensed reaches (protocol §12). No model is fitted here.

    # 1. freeze the station roles + power check (before any prediction at a track-N station)
    python -B scripts/eval/hydromet_track_n.py roles
    # 2. predict with saved original-track artifacts (val block: val- and train-role stations)
    python -B scripts/eval/hydromet_track_n.py predict --run <run dirs of one input set> --block val --out <dir> [--gpu]
    # 3. score (descriptive on val; registered as M6N/val_*)
    python -B scripts/eval/hydromet_track_n.py score --predictions <dirs> --block val --roles val --comparators <json> \\
        --out <dir> [--register --experiment M6N/val_v1]

Test-role stations × test block need --allow-test and the frozen FREEZE_hydromet.json (same guard as
``hydromet_predict.py``); every artifact must then match the frozen weights_sha256.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.core import paths
from sentinel.models.hydromet import runner

ND = paths.PROCESSED / "hydromet_wqp_national"
ROLES_FILE = paths.REPO_ROOT / "docs" / "preregistration" / "hydromet_track_n_stations.json"
POWER_FILE = paths.REPO_ROOT / "docs" / "preregistration" / "hydromet_track_n_power.json"
TARGETS_N = ("temp_c_mean", "do_mgl_mean")
MIN_DAYS = 8


def cmd_roles(a) -> int:
    from sentinel.models.hydromet import data as D
    from sentinel.models.hydromet import track_n as TN

    sp, _ = D.load_split()
    st = pd.read_parquet(ND / "stations.parquet")
    sc = pd.read_parquet(ND / "streamcat_comid.parquet", columns=["comid", "streamcat_found"])
    cells = pd.read_parquet(ND / "station_cells.parquet")
    st = st.merge(sc, on="comid", how="left").merge(cells[["station_key", "i"]], on="station_key", how="left")
    excl = {"no_streamcat": int((~st["streamcat_found"].fillna(False)).sum()), "no_gridmet_cell": int((st["i"] < 0).sum())}
    st = st[st["streamcat_found"].fillna(False) & (st["i"] >= 0)]
    reg = pd.read_parquet(D.SITES_FILE)
    reg = reg[reg["network"] == "USGS"].drop_duplicates("site_id").set_index("site_id")
    s_all = sorted(sp.site_role)
    ssc = pd.read_parquet(D.STREAMCAT_FILE, columns=["site_key", "comid"]).drop_duplicates("site_key").set_index("site_key")
    keys = reg["site_key"].reindex(s_all)
    split_sites = pd.DataFrame({"site_no": s_all, "role": [sp.site_role[s] for s in s_all],
                                "basin_group": [sp.basin_group[s] for s in s_all],
                                "huc8": reg["huc8"].reindex(s_all).astype(str).to_numpy(),
                                "comid": ssc["comid"].reindex(keys).to_numpy()})
    vaa = pd.read_parquet(paths.RAW / "nhdplus_vaa" / "nhdplus_vaa_subset.parquet",
                          columns=["comid", "hydroseq", "dnhydroseq", "pathlength"])
    vaa["comid"] = vaa["comid"].astype("int64").astype(str)
    roles = TN.assign_roles(st[["station_key", "comid", "huc8"]], split_sites, vaa)
    man = json.loads((paths.MANIFESTS / "hydromet_wqp_national.json").read_text())
    data_sha = {Path(f["path"]).name: f.get("sha256") for f in man.get("files", [])}
    body = TN.payload(roles, sp.sha256(), data_sha)
    body["excluded_before_roles"] = excl
    if ROLES_FILE.exists() and json.loads(ROLES_FILE.read_text())["sha256"] != body["sha256"]:
        raise SystemExit(f"{ROLES_FILE} exists with a different sha256 — track-N roles are frozen")
    ROLES_FILE.write_text(json.dumps(body, indent=0))
    # power check: eligible station x target x block counts by role, groups
    el = pd.read_parquet(ND / "eligibility.parquet")
    el["station_key"] = "WQP:" + el["site_id"]
    el = el.merge(roles[["station_key", "role", "basin_group", "rule"]], on="station_key")
    cells_ = el[((el["role"] == "val") & (el["block"] == "val")) | ((el["role"] == "test") & (el["block"] == "test"))
                | ((el["role"] == "train") & (el["block"] == "val"))]
    pw = {}
    for (v, r), g in cells_.groupby(["target", "role"]):
        pw[f"{v}|{r}"] = {"stations": int(g["station_key"].nunique()), "groups": int(g["basin_group"].nunique()),
                          "new_region_groups": int(g.loc[g["rule"] == "new_region", "basin_group"].nunique()),
                          "median_days": float(g["n_days"].median())}
    rep = {"roles_sha256": body["sha256"], "role_counts": roles["role"].value_counts().to_dict(),
           "rule_counts": roles["rule"].value_counts().to_dict(), "cells": pw, "excluded_before_roles": excl,
           "note": ("Primary unit = basin group (FINAL group or new-region HUC4 union). Original-track val had 12 / 9 groups "
                    "for temp_c_mean / do_mgl_mean; MDE scales ~ sqrt(G_orig / G_N) (see hydromet_track_r_power.json for "
                    "the SE reference).")}
    POWER_FILE.write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))
    return 0


def load_frozen_roles() -> dict:
    from sentinel.models.hydromet import track_n as TN  # noqa: F401

    body = json.loads(ROLES_FILE.read_text())
    import hashlib

    core = {k: body[k] for k in ("split_sha256", "role_of", "basin_group_of")}
    if hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest() != body["sha256"]:
        raise ValueError("track-N role file hash mismatch")
    return body


def build_data(block: str, roles_wanted, *, allow_test=False, freeze_file=None, static_ablation=None, upstream=None,
               log=print):
    """HydroMetData over track-N stations (sites = station keys). Targets: temp_c_mean / do_mgl_mean daily means of
    the discrete samples; inputs exactly as for sonde sites (site-cell gridMET, StreamCat, elevcat for elev_m)."""
    import pyarrow.dataset as ds

    from sentinel.models.hydromet import data as D
    from sentinel.models.hydromet import guard

    body = load_frozen_roles()
    sp, sinfo = D.load_split()
    if body["split_sha256"] != sp.sha256():
        raise ValueError("track-N roles built from another split")
    keys = sorted(k for k, r in body["role_of"].items() if r in roles_wanted)
    blocks = tuple(dict.fromkeys(["train", block]))
    access = guard.check_access(blocks, set(roles_wanted), allow_test=allow_test, freeze_file=freeze_file, split=sp,
                                training=False)
    end = guard.data_end_date(blocks)
    dates = pd.date_range(D.DATA_START, end, freq="D")
    S, T = len(keys), len(dates)
    pos = {k: i for i, k in enumerate(keys)}
    forc = np.full((S, T, len(D.FORCING_VARS)), np.nan, dtype="float32")
    dset = ds.dataset(ND / "forcings.parquet", format="parquet")
    for c in range(0, S, 1000):
        kk = keys[c:c + 1000]
        tab = dset.to_table(filter=ds.field("station_key").isin(kk) & (ds.field("date") <= dates[-1].date())).to_pandas()
        si = tab["station_key"].map(pos).to_numpy()
        ti = (pd.to_datetime(tab["date"]) - dates[0]).dt.days.to_numpy()
        for j, v in enumerate(D.FORCING_RAW):
            forc[si, ti, j] = tab[v].to_numpy(dtype="float32")
    forc[:, :, D.FORCING_VARS.index("tmean_c")] = 0.5 * (forc[:, :, D.FORCING_VARS.index("tmax_c")]
                                                       + forc[:, :, D.FORCING_VARS.index("tmin_c")])
    if upstream:   # protocol v0.10 inputs at track-N stations (processed/hydromet_upstream_forcing/*_trackn)
        if upstream not in D.UPSTREAM_MODES:
            raise ValueError(f"unknown upstream_forcing {upstream!r}")
        up = D.load_upstream_forcings(keys, dates, path=paths.PROCESSED / "hydromet_upstream_forcing" /
                                      "upstream_forcing_trackn.parquet")
        miss = ~np.isfinite(up) & np.isfinite(forc)
        up = np.where(miss, forc, up).astype("float32")
        forc = np.concatenate([forc, up], axis=-1) if upstream == "append" else up
        log(f"[track N] upstream forcing ({upstream}); {miss.mean():.4%} cells filled from the station cell")
    st = pd.read_parquet(ND / "stations.parquet").set_index("station_key").loc[keys]
    sc = pd.read_parquet(ND / "streamcat_comid.parquet").drop_duplicates("comid").set_index("comid")
    feat = D.streamcat_feature_columns()   # the sonde-site StreamCat feature columns, same order
    static = sc.reindex(st["comid"].astype(str))[[c for c in feat if c in sc.columns]].reindex(columns=feat).astype("float64")
    static.index = keys
    static.insert(0, "elev_m", np.nan)
    static.insert(0, "lon", st["lon"].to_numpy(dtype="float64"))
    static.insert(0, "lat", st["lat"].to_numpy(dtype="float64"))
    static["elev_m"] = static["elev_m"].fillna(static.get("elevcat"))   # no gauge datum at WQP stations
    if static_ablation:
        static = D.apply_static_ablation(static, static_ablation)
    y = np.full((S, T, len(D.TARGETS)), np.nan, dtype="float32")
    obs = pd.read_parquet(ND / "obs_daily.parquet")
    obs["station_key"] = "WQP:" + obs["site_id"]
    obs = obs[obs["station_key"].isin(pos) & (obs["date"] <= dates[-1]) & (obs["date"] >= dates[0])]
    for v in TARGETS_N:
        o = obs[obs["target"] == v]
        y[o["station_key"].map(pos).to_numpy(), (o["date"] - dates[0]).dt.days.to_numpy(), D.TARGETS.index(v)] = o["value"].to_numpy()
    prov = np.zeros_like(y, dtype=bool)
    roles_arr = np.array([body["role_of"][k] for k in keys])
    bg = np.array([str(body["basin_group_of"][k]) for k in keys])
    data = D.HydroMetData(keys, roles_arr, bg, dates, forc, y, prov, static, sp, sinfo, access, blocks,
                          {"n_sites": S, "track": "N", "roles_sha256": body["sha256"]})
    log(f"[track N] {S} stations {pd.Series(roles_arr).value_counts().to_dict()}; dates {dates[0].date()}..{dates[-1].date()}")
    return data


def cmd_predict(a) -> int:
    from sentinel.models.hydromet import pipeline

    cfgs = [json.loads((Path(r) / "config.json").read_text()) for r in a.run]
    lags = {runner.check_mode(c) for c in cfgs}
    abl = {c.get("static_ablation") for c in cfgs}
    ups = {c.get("upstream_forcing") for c in cfgs}
    if len(lags) != 1 or len(abl) != 1 or len(ups) != 1 or any(c.get("_smoke") or c.get("track_r") for c in cfgs):
        raise ValueError("predict runs of one input set at a time (no smoke / track-R artifacts)")
    roles = tuple(r.strip() for r in a.roles.split(",") if r.strip())
    if a.gpu:
        runner.setup_gpu("hydromet_trackn_predict")
    data = build_data(a.block, roles, allow_test=a.allow_test, freeze_file=a.freeze_file, static_ablation=abl.pop(),
                      upstream=ups.pop())
    si, ti = np.arange(len(data.sites)), np.flatnonzero(data.block_mask(a.block))
    arts = pipeline.discover_artifacts(a.run)
    if a.only:
        arts = {k: v for k, v in arts.items() if k in set(a.only.split(","))}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    pipeline.predict_artifacts(data, arts, si, ti, a.block, a.out, lag=lags.pop(), access=data.access,
                               device="cuda" if a.gpu else "cpu")
    return 0


def cmd_score(a) -> int:
    from sentinel.models.hydromet import pipeline

    roles = tuple(r.strip() for r in a.roles.split(",") if r.strip())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = runner.make_logger(out / "log.txt")
    data = build_data(a.block, roles, allow_test=a.allow_test, freeze_file=a.freeze_file, log=log)
    si, ti = np.flatnonzero(np.isin(data.roles, roles)), np.flatnonzero(data.block_mask(a.block))
    preds, sources = pipeline.load_prediction_set(a.predictions, data, si, ti, a.block, smoke=False, access=data.access)
    comps = json.loads(Path(a.comparators).read_text())
    comps = comps.get("comparators", comps)
    models = [m for m in a.models.split(",") if m] or None
    score_track_n(data, preds, sources, roles, a.block, comps, models, out, log=log,
                  experiment=(a.experiment or f"M6N/{a.block}") if a.register else None)
    return 0


def score_track_n(data, preds: dict, sources: dict, roles, block: str, comps: dict, models, out: Path, *, log=print,
                  experiment: str | None = None, n_hypotheses: int | None = None) -> pd.DataFrame:
    """Track-N scoring (protocol §12): per-station metrics on stations with >= MIN_DAYS sample days; primary Δ median
    per-station RMSE vs the fixed comparators, IUT, every-seed rule, Holm. Hypotheses that cannot be tested (model or
    comparator missing, < 3 paired stations) enter Holm with p = 1 (never dropped). ``models``/``comps`` come from the
    freeze block on test (hydromet_test_look.py)."""
    from sentinel.eval.bootstrap import bootstrap_ci, holm_bonferroni, paired_bootstrap
    from sentinel.models.hydromet import evaluate as E
    from sentinel.models.hydromet import pipeline

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    si, ti = np.flatnonzero(np.isin(data.roles, roles)), np.flatnonzero(data.block_mask(block))
    preds = {**preds, **pipeline.seed_blends(preds)}
    ens, members = pipeline.seed_ensembles(preds)
    preds = {**preds, **ens}
    site_df, info = E.site_metrics(data, preds, si, ti, block, min_days=MIN_DAYS, approved_only=False, targets=list(TARGETS_N))
    site_df.to_parquet(out / f"site_metrics_{block}.parquet", index=False)
    runner.write_json(out / "rows.json", info)
    rows = []
    for (m, v), g in site_df.groupby(["model", "var"]):
        for met in ("rmse", "kge", "bias"):
            x = g[met].to_numpy(dtype=float)
            ok = np.isfinite(x)
            if ok.sum() < 3:
                continue
            ci = bootstrap_ci(lambda z: np.nanmedian(z), x[ok], groups=g["basin_group"].to_numpy()[ok], n_boot=2000)
            rows.append({"model": m, "var": v, "metric": f"median_site_{met}", "value": float(np.nanmedian(x[ok])),
                         "ci_low": ci["ci_low"], "ci_high": ci["ci_high"], "n_sites": int(ok.sum()),
                         "n_groups": int(g["basin_group"][ok].nunique())})
    summ = pd.DataFrame(rows)
    summ.to_csv(out / f"summary_{block}.csv", index=False)
    family = list(models) if models else [m for m in E.SENTINEL_MODELS if m in preds]
    prim = []
    for m in family:
        for v in TARGETS_N:
            rec = {"model": m, "var": v, "status": "tested"}
            for fam in ("gbm", "physics"):
                c = comps.get(v, {}).get(fam)
                A = site_df[(site_df["model"] == m) & (site_df["var"] == v)].set_index("site_no")
                B = site_df[(site_df["model"] == c) & (site_df["var"] == v)].set_index("site_no")["rmse"]
                j = A[["rmse", "basin_group"]].join(B.rename("b"), how="inner").dropna() if len(A) else pd.DataFrame()
                rec[f"{fam}_comparator"] = c
                if c is None or len(j) < 3:
                    rec["status"] = "not available"
                    continue
                r = paired_bootstrap(lambda _y, p: np.nanmedian(p), np.zeros(len(j)), j["rmse"].to_numpy(), j["b"].to_numpy(),
                                     groups=j["basin_group"].to_numpy(), n_boot=2000, higher_is_better=False)
                rec.update({f"{fam}_diff": r["diff"], f"{fam}_ci_low": r["ci_low"], f"{fam}_ci_high": r["ci_high"],
                            f"{fam}_p": r["p_value"], f"{fam}_n_sites": len(j), f"{fam}_n_groups": int(j["basin_group"].nunique())})
                sd = []
                for sname in members.get(m, []):
                    As = site_df[(site_df["model"] == sname) & (site_df["var"] == v)].set_index("site_no")["rmse"]
                    jj = pd.concat([As.rename("a"), B.rename("b")], axis=1, join="inner").dropna()
                    sd.append(float(np.nanmedian(jj["a"]) - np.nanmedian(jj["b"])))
                rec[f"{fam}_every_seed_better"] = bool(sd) and all(d < 0 for d in sd)
            rec["p_iut"] = max(rec["gbm_p"], rec["physics_p"]) if rec["status"] == "tested" else 1.0
            prim.append(rec)
    prim = pd.DataFrame(prim)
    if len(prim):
        rej = holm_bonferroni({str(i): float(p) for i, p in prim["p_iut"].items()}, alpha=0.05)
        prim["holm_reject"] = [rej.get(str(i), False) for i in prim.index]
        prim["n_hypotheses"] = len(prim)
        if n_hypotheses is not None and len(prim) != n_hypotheses:
            raise RuntimeError(f"track-N family has {len(prim)} hypotheses, freeze says {n_hypotheses}")
        for c in ("gbm_ci_high", "physics_ci_high"):
            if c not in prim:
                prim[c] = np.nan
        for c in ("gbm_every_seed_better", "physics_every_seed_better"):
            prim[c] = prim[c].fillna(False).astype(bool) if c in prim else False
        prim["won"] = (prim["holm_reject"] & (prim["gbm_ci_high"] < 0) & (prim["physics_ci_high"] < 0)
                       & prim["gbm_every_seed_better"] & prim["physics_every_seed_better"] & (prim["status"] == "tested"))
    prim.to_csv(out / f"primary_{block}.csv", index=False)
    runner.write_json(out / "sources.json", sources)
    log(prim.round(3).to_string(index=False) if len(prim) else "no primary rows")
    if experiment:
        from sentinel.core.registry import log_result

        exp = experiment
        split = f"track N (national unsensed reaches, WQP discrete) {'/'.join(roles)}-role stations x {block} block"
        cfg = {"track": "N", "roles_sha256": data.info["roles_sha256"], "min_days": MIN_DAYS, "block": block,
               "roles": list(roles), "comparators": comps, "family": family}
        arts = sorted({x for s in sources.values() for x in s.split(";")})
        n = 0
        for r in summ.itertuples(index=False):
            log_result(workstream=runner.WORKSTREAM, experiment=exp, model=runner.MODEL_PREFIX + r.model, metric=r.metric,
                       value=float(r.value), ci_low=float(r.ci_low), ci_high=float(r.ci_high), n=int(r.n_sites), target=r.var,
                       split=split, is_baseline=runner._is_baseline(r.model), config=cfg,
                       ci_method="percentile bootstrap of the median, resampling basin groups (B=2000)",
                       data_manifests=["hydromet_wqp_national", "wqp_discrete", "streamcat"], artifacts=arts,
                       extra={"n_groups": int(r.n_groups), "split_sha256": data.split.sha256()})
            n += 1
        for r in prim.to_dict("records"):
            for fam in ("gbm", "physics"):
                if np.isfinite(r.get(f"{fam}_diff", np.nan)):
                    log_result(workstream=runner.WORKSTREAM, experiment=exp, model=runner.MODEL_PREFIX + r["model"],
                               metric=f"diff_median_site_rmse_vs_best_{fam}", value=float(r[f"{fam}_diff"]),
                               ci_low=float(r[f"{fam}_ci_low"]), ci_high=float(r[f"{fam}_ci_high"]),
                               n=int(r[f"{fam}_n_sites"]), target=r["var"], split=split, is_baseline=False, config=cfg,
                               ci_method="paired percentile bootstrap resampling basin groups (B=2000)",
                               data_manifests=["hydromet_wqp_national", "wqp_discrete", "streamcat"], artifacts=arts,
                               extra={"comparator": r[f"{fam}_comparator"], "p": r[f"{fam}_p"], "p_iut": r.get("p_iut"),
                                      "holm_reject": bool(r.get("holm_reject", False)), "won": bool(r.get("won", False)),
                                      "n_hypotheses": int(r.get("n_hypotheses", 0)), "n_groups": int(r[f"{fam}_n_groups"])})
                    n += 1
        log(f"[registry] {n} entries ({exp})")
    return prim


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("roles")
    for name in ("predict", "score"):
        p = sub.add_parser(name)
        p.add_argument("--block", default="val")
        p.add_argument("--roles", default="val,train" if name == "predict" else "val")
        p.add_argument("--allow-test", action="store_true")
        p.add_argument("--freeze-file", default=None)
        p.add_argument("--out", required=True)
        if name == "predict":
            p.add_argument("--run", nargs="+", required=True)
            p.add_argument("--only", default="")
            p.add_argument("--gpu", action="store_true")
        else:
            p.add_argument("--predictions", nargs="+", required=True)
            p.add_argument("--comparators", required=True)
            p.add_argument("--models", default="", help="primary family (default: all SENTINEL models present)")
            p.add_argument("--register", action="store_true")
            p.add_argument("--experiment", default=None)
    a = ap.parse_args()
    if getattr(a, "allow_test", False) or getattr(a, "block", "val") in ("test", "test_b") or "test" in getattr(a, "roles", ""):
        raise SystemExit("protected data (test blocks / test-role sites) are scored ONLY through "
                         "scripts/eval/hydromet_test_look.py (single-pass guard; models, comparators and the Holm family "
                         "come from the freeze block)")
    runner.env_threads(6)
    return {"roles": cmd_roles, "predict": cmd_predict, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
