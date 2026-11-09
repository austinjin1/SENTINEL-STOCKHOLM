# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Fuse Stage 0: cross-validated comparators on development folds (validation only).

    python -B scripts/train/fuse_stage0.py --tier d --targets p1 --threads 6
    python -B scripts/train/fuse_stage0.py --tier c --targets gt8 --scheme block20 --threads 6

Tier D: leave-domain-group-out over the 5 development folds of the frozen split
(docs/preregistration/fuse_tier_d_split_v2.json); Tier C: development folds of
fuse_tier_c_split.json. Test-role rows are dropped right after loading. Predictions for every
development row (out-of-fold) are written to $SENTINEL_RUNS/fuse/<run>/oof_<target>.parquet;
summary metrics go to the registry (split = 'val_cv').
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

import numpy as np
import pandas as pd

from sentinel.core import paths
from sentinel.core.registry import log_result
from sentinel.core.runlog import TrainingRun
from sentinel.data import fuse_tier_c as ftc
from sentinel.data import fuse_tiers as ftd
from sentinel.eval.bootstrap import bootstrap_ci, paired_bootstrap
from sentinel.eval.metrics import auprc, auroc, brier
from sentinel.models.fuse import baselines as bl
from sentinel.models.fuse import data as fd
from sentinel.models.fuse import splits as sp
from sentinel.models.fuse import stage0 as s0
from sentinel.models.fuse import stats as fst
from sentinel.eval.splits import spatial_blocks

SPLIT_SHA_D = "20adbe595ed42a46c490da31e16743353ce819e9d84cb8eedffa15303a26610a"
SPLIT_SHA_C = "e2fd5de9da4079091445c0635f8ccf1febe7c878a7dc7a3beb679ad0b68e3f37"


def log(*a):
    print(*a, flush=True)


def tier_d_setup():
    split = sp.load_split(sp.TIER_D_SPLIT, SPLIT_SHA_D)
    df, genus = fd.load_tier_d()
    df = s0.drop_protected(df, split["site_role"], "site")
    fold = df["site"].map(split["site_fold"]).astype(int).to_numpy()
    specs = fd.tier_d_modalities(df)
    mods = [s.name for s in specs if not s.hub]
    hub = [s.name for s in specs if s.hub]
    return df, genus, fold, specs, mods, hub


def evaluate_gauss(pred: pd.DataFrame, y: np.ndarray, groups: np.ndarray, configs: list[str]) -> dict:
    res = {}
    for c in configs:
        p = pred[pred.config == c].set_index("row").sort_index()
        rows = p.index.to_numpy()
        ok = np.isfinite(y[rows]) & np.isfinite(p["mu"].to_numpy()) & np.isfinite(p["sigma"].to_numpy())
        cr = bl.crps_gauss_rows(y[rows][ok], p["mu"].to_numpy()[ok], p["sigma"].to_numpy()[ok])
        res[c] = {"crps": float(cr.mean()), "n": int(ok.sum()), "rows": rows[ok], "crps_rows": cr}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["d", "c"], default="d")
    ap.add_argument("--targets", default="p1")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--scheme", default="block20", help="tier C dev-fold scheme: block20 | block5 | huc2")
    ap.add_argument("--no-stack", action="store_true")
    ap.add_argument("--pattern", action="store_true")
    ap.add_argument("--exclude-mods", default="", help="comma list of modalities to leave out (e.g. phyto)")
    ap.add_argument("--anchored", action="store_true", help="also fit persistence-anchored <config>@a GBMs")
    ap.add_argument("--max-folds", type=int, default=99)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args(argv)
    run = args.run_name or f"stage0_{args.tier}_{args.targets}_{dt.datetime.now():%Y%m%dT%H%M%S}"
    out = paths.run_dir("fuse", run)
    cfg = vars(args) | {"run": run}
    if args.tier == "d":
        df, genus, fold, specs, mods, hub = tier_d_setup()
        targets = fd.tier_d_targets(df)
        if args.targets == "p1":
            targets = [t for t in targets if t.name in fd.P1_TARGETS]
        elif args.targets == "lab":
            targets = [t for t in targets if t.name in ("ylab__tn_mgl", "ylab__tp_mgl", "ylab__doc_mgl")]
        elif args.targets == "events":
            targets = [t for t in targets if t.kind == "binary"]
        else:
            targets = [t for t in targets if t.name in args.targets.split(",")]
        groups = df["site"].to_numpy()
        boot_groups = groups          # NEON site (domain jackknife reported for P1, §10)
        manifests = ["fuse_tier_d"]
        split_name = "val_cv_lodgo"
        site_col = "site"
    else:
        split = sp.load_split(sp.TIER_C_SPLIT, SPLIT_SHA_C)
        df = pd.read_parquet(paths.processed("fuse_tier_c") / "rows.parquet")
        df = s0.drop_protected(df, split["site_role"], "site_id")
        fmap = split["dev_folds"][args.scheme]
        if args.scheme == "huc2":
            hs = sorted(set(fmap.values()))
            fold = df["site_id"].map(fmap).map({h: i for i, h in enumerate(hs)}).to_numpy()
        else:
            fold = df["site_id"].map(fmap).astype(int).to_numpy()
        mods = [m for m in ftc.NON_HUB_C if df[f"{m}__present"].any()]
        hub = ftc.HUB_C
        tname = {"gt8total": "y_gt8_total", "gt8": "y_gt8", "gt1p6": "y_gt1p6", "gt0p3": "y_gt0p3", "mc": "mc_log10"}
        targets = [fd.Target(tname[t], "gauss" if t == "mc" else "binary", t, 0) for t in args.targets.split(",")]
        # bootstrap unit for Tier C = 20-km spatial block (review W8); model inner splits stay by site
        groups = df["site_id"].to_numpy()
        boot_groups = spatial_blocks(df["lat"], df["lon"], 20.0)
        manifests = ["fuse_tier_c"]
        split_name = f"val_cv_{args.scheme}"
        site_col = "site_id"
        specs = None
    all_folds = sorted(np.unique(fold))
    folds = all_folds[: args.max_folds]
    if len(folds) < len(all_folds):
        split_name = "smoke"          # partial CV never shares a split name with full runs
    else:
        split_name = split_name + f"{len(all_folds)}"
    if args.exclude_mods:
        ex = set(args.exclude_mods.split(","))
        mods = [m for m in mods if m not in ex]
        if specs is not None:
            specs = [s_ for s_ in specs if s_.name not in ex]
    pres = pd.DataFrame({m: df[f"{m}__present"].fillna(False).astype(bool).to_numpy() for m in mods})
    log(f"rows={len(df)} folds={folds} mods={mods} hub={hub} targets={[t.name for t in targets]}")
    with TrainingRun("fuse", f"stage0_{args.tier}_{args.targets}", config=cfg, data=manifests,
                     plan_ref="RESEARCH_PLAN §6 Stage 0", iteration="stage0", artifacts_dir=str(out)) as tr:
        summary = {}
        for t in targets:
            if args.tier == "d":
                Y, M, C = fd.tier_d_target_matrix(df, [t])
                y = Y[:, 0]
                y_eval = np.where(C[:, 0], np.nan, y)   # censored lab values are bounds: trained on, never scored
            else:
                y = df[t.name].to_numpy(float)
                y_eval = np.where(df["mc_cens"].to_numpy(bool), np.nan, y) if t.name == "mc_log10" else y
            parts = []
            for k in folds:
                trm = (fold != k) & np.isfinite(y)
                evm = (fold == k) & np.isfinite(y)
                if trm.sum() < 100 or evm.sum() < 20:
                    continue
                if args.tier == "d":
                    pre = fd.FoldPreprocessor(specs)     # GBM path: only the 16S CLR/PCA is fit (train rows)
                    if "micro" in mods:
                        pre.fit_micro(df, trm, genus)
                    X = fd.tier_d_feature_frame(df, specs, pre, genus)
                else:
                    cols = [c for c in df.columns if c.split("__")[0] in set(mods) | set(hub)
                            and not c.endswith("__present")]
                    X = df[cols].astype(float).copy()
                    for m in mods:
                        X.loc[~pres[m].to_numpy(), [c for c in cols if c.startswith(f"{m}__")]] = np.nan
                tri, evi = np.flatnonzero(trm), np.flatnonzero(evm)
                log(f"  {t.name} fold {k}: train {len(tri)} eval {len(evi)}")
                p = s0.run_fold_baselines(X, pres, y, t.kind, groups, tri, evi, fold[tri], mods, hub,
                                          persistence_col=s0.PERSISTENCE_D.get(t.name) if args.tier == "d" else None,
                                          threads=args.threads, do_stack=not args.no_stack, do_pattern=args.pattern,
                                          anchored=args.anchored, log=log)
                p["fold"] = k
                parts.append(p)
            pred = pd.concat(parts, ignore_index=True)
            pred["site"] = df[site_col].to_numpy()[pred["row"]]
            pred["date"] = df["date"].to_numpy()[pred["row"]]
            pred["y"] = y[pred["row"]]
            pred.to_parquet(out / f"oof_{t.name}.parquet", index=False)
            configs = sorted(pred.config.unique())
            res = {}
            if t.kind == "gauss":
                ev = evaluate_gauss(pred, y_eval, groups, configs)
                # common rows: all configs except persistence (which needs the sensor)
                base = [c for c in configs if c != "persistence"]
                common = set(ev[base[0]]["rows"])
                for c in base[1:]:
                    common &= set(ev[c]["rows"])
                common = np.array(sorted(common))
                for c in configs:
                    e = ev[c]
                    m = np.isin(e["rows"], common) if c != "persistence" else np.ones(len(e["rows"]), bool)
                    cr = e["crps_rows"][m]
                    gr = boot_groups[e["rows"][m]]
                    ci = bootstrap_ci(np.mean, cr, groups=gr, n_boot=1000)
                    res[c] = {"crps": float(cr.mean()), "ci": [ci["ci_low"], ci["ci_high"]], "n": int(m.sum()),
                              "n_sites": int(len(np.unique(gr)))}
                    log_result(workstream="fuse", experiment=f"fuse_stage0_{args.tier}", model=c, metric="crps",
                               value=float(cr.mean()), split=split_name, n=int(m.sum()), ci_low=ci["ci_low"],
                               ci_high=ci["ci_high"], ci_method="site-cluster bootstrap (B=1000) of mean CRPS",
                               target=t.name, is_baseline=True, config=cfg, data_manifests=manifests,
                               artifacts=[str(out / f"oof_{t.name}.parquet")],
                               notes="model-space CRPS; common rows" if c != "persistence" else
                               "model-space CRPS; rows with persistence available")
                # paired: concat vs best single (chosen by CV CRPS on common rows) and vs hub
                singles = {c: res[c]["crps"] for c in configs if c.startswith("single:")}
                best = min(singles, key=singles.get) if singles else None
                for ref in [r for r in (best, "hub", "persistence") if r]:
                    for c in ("concat", "concat@a", "stack", "ume"):
                        if c not in ev or ref not in ev:
                            continue
                        ra = pd.Series(ev[c]["crps_rows"], index=ev[c]["rows"])
                        rb = pd.Series(ev[ref]["crps_rows"], index=ev[ref]["rows"])
                        idx = ra.index.intersection(rb.index)
                        a, b = ra.loc[idx].to_numpy(), rb.loc[idx].to_numpy()
                        skill = 1 - a.mean() / b.mean()
                        bs = fst.cluster_ci(fst.skill, a, b, groups=boot_groups[idx], n_boot=1000)
                        res.setdefault("paired", {})[f"{c}_vs_{ref}"] = {"skill": float(skill), "ci": [bs["ci_low"], bs["ci_high"]],
                                                                         "n": int(len(idx))}
                        log_result(workstream="fuse", experiment=f"fuse_stage0_{args.tier}", model=c,
                                   metric=f"crps_skill_vs_{ref}", value=float(skill), split=split_name, n=int(len(idx)),
                                   ci_low=bs["ci_low"], ci_high=bs["ci_high"],
                                   ci_method="cluster bootstrap (B=1000; Tier D site, Tier C 20-km block) of "
                                             "1-CRPS_a/CRPS_b, small-cluster factor sqrt(G/(G-1)), paired rows",
                                   target=t.name, is_baseline=True, config=cfg, data_manifests=manifests)
                res["best_single"] = best
            else:
                for c in configs:
                    p = pred[pred.config == c]
                    ok = np.isfinite(p["p"].to_numpy()) & np.isfinite(p["y"].to_numpy())
                    yy, pp = p["y"].to_numpy()[ok], p["p"].to_numpy()[ok]
                    gg = boot_groups[p["row"].to_numpy()[ok]]
                    if len(np.unique(yy)) < 2:
                        continue
                    ci = bootstrap_ci(auprc, yy, pp, groups=gg, n_boot=1000)
                    ci2 = bootstrap_ci(auroc, yy, pp, groups=gg, n_boot=1000)
                    res[c] = {"auprc": ci["point"], "auprc_ci": [ci["ci_low"], ci["ci_high"]], "auroc": ci2["point"],
                              "auroc_ci": [ci2["ci_low"], ci2["ci_high"]], "brier": float(brier(yy, pp)),
                              "n": int(ok.sum()), "prev": float(yy.mean())}
                    for met, cc in (("auprc", ci), ("auroc", ci2)):
                        log_result(workstream="fuse", experiment=f"fuse_stage0_{args.tier}", model=c, metric=met,
                                   value=float(cc["point"]), split=split_name, n=int(ok.sum()), ci_low=cc["ci_low"],
                                   ci_high=cc["ci_high"], ci_method="cluster bootstrap (B=1000; Tier D site, Tier C 20-km block)",
                                   target=t.name,
                                   is_baseline=True, config=cfg, data_manifests=manifests,
                                   extra={"prevalence": float(yy.mean())})
            summary[t.name] = res
            tr.metric(f"{t.name}_n_configs", len(configs))
            log(json.dumps({t.name: {k: v for k, v in res.items()}}, default=str, indent=0)[:3000])
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
