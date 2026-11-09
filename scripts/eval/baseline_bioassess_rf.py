#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Blocked random-forest baseline: benthic community -> co-measured chemistry (NRSA 2018-19).

Reproduces the research-phase baseline (docs/research/04_ecotox_bioassessment.md §A.4) from the
processed tables of processed('nars_streams'):

* samples  : NRSA 2018-19 first visits (VISIT_NO == 1) with a benthic sample and chemistry
* features : community = sqrt(relative abundance) (Hellinger) of TOTAL300 counts per TARGET_TAXON,
             taxa present in >= 10 samples; geography = lat, lon, elevation (median-filled);
             community+geography = both
* targets  : log10 TP (ug/L, floor 1), log10 TN (mg/L, floor 0.01), log10 conductivity (uS/cm,
             floor 1), log10 chloride (mg/L, floor 0.01), log10 turbidity (NTU, floor 0.01), lab pH
* model    : RandomForestRegressor(300 trees, min_samples_leaf=3, max_features=0.3)
* splits   : leave-one-ecoregion-out over the 9 NARS aggregated ecoregions (primary), random
             5-fold over sites (leakage-gap context only), HUC2 groups in 9 folds (context);
             built as sentinel.eval.splits.SiteSplit objects, saved with split_report()
* metric   : R^2 (sentinel.eval.metrics.r2) of pooled out-of-fold predictions
* CI       : percentile bootstrap over sites (sentinel.eval.bootstrap.bootstrap_ci, B=2000)
* seeds    : 0 (the research configuration; registry value) plus 1 and 2 for LOEO (seed spread
             stored in the entry's ``extra``)

Every R^2 is logged with ``sentinel.core.registry.log_result`` (workstream 'bioassess',
experiment 'rf_blocked_baseline', is_baseline=True). These are cross-validation estimates of a
fixed, pre-specified baseline configuration (no tuning), not a held-out test score.

    python -B scripts/eval/baseline_bioassess_rf.py [--no-log] [--runlog] [--n-boot 2000]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.registry import file_sha256, log_result, registry_frame  # noqa: E402
from sentinel.eval import bootstrap as EB  # noqa: E402
from sentinel.eval import metrics as EM  # noqa: E402
from sentinel.eval import splits as ES  # noqa: E402

log = logging.getLogger("baseline_bioassess_rf")

WORKSTREAM = "bioassess"
EXPERIMENT = "rf_blocked_baseline"
SURVEY = "nrsa1819"
MIN_SAMPLES_PER_TAXON = 10
RF_PARAMS = {"n_estimators": 300, "min_samples_leaf": 3, "max_features": 0.3}
TARGETS = {  # name -> (chemistry_wide column, transform, floor)
    "log10_tp_ug_l": ("tp_ug_l", "log10", 1.0),
    "log10_tn_mg_l": ("tn_mg_l", "log10", 0.01),
    "log10_cond_us_cm": ("cond_us_cm", "log10", 1.0),
    "log10_chloride_mg_l": ("chloride_mg_l", "log10", 0.01),
    "log10_turbidity_ntu": ("turbidity_ntu", "log10", 0.01),
    "ph_lab": ("ph_lab_su", "identity", None),
}
# Research-phase numbers (docs/research/04 §A.4) for the comparison column.
RESEARCH = {
    ("loeo_ag_eco9", "community"): {"log10_tp_ug_l": 0.34, "log10_tn_mg_l": 0.39, "log10_cond_us_cm": 0.33,
                                     "log10_chloride_mg_l": 0.33, "log10_turbidity_ntu": 0.39, "ph_lab": 0.29},
    ("loeo_ag_eco9", "geography"): {"log10_tp_ug_l": 0.06, "log10_tn_mg_l": 0.31, "log10_cond_us_cm": 0.15,
                                     "log10_chloride_mg_l": 0.27, "log10_turbidity_ntu": 0.13, "ph_lab": 0.13},
    ("loeo_ag_eco9", "community+geography"): {"log10_tp_ug_l": 0.34, "log10_tn_mg_l": 0.48,
                                               "log10_cond_us_cm": 0.48, "log10_chloride_mg_l": 0.43,
                                               "log10_turbidity_ntu": 0.37, "ph_lab": 0.30},
    ("random5", "community"): {"log10_tp_ug_l": 0.43, "log10_tn_mg_l": 0.51, "log10_cond_us_cm": 0.51,
                                "log10_chloride_mg_l": 0.46, "log10_turbidity_ntu": 0.49, "ph_lab": 0.39},
    ("random5", "geography"): {"log10_tp_ug_l": 0.50, "log10_tn_mg_l": 0.60, "log10_cond_us_cm": 0.65,
                                "log10_chloride_mg_l": 0.62, "log10_turbidity_ntu": 0.47, "ph_lab": 0.46},
}
SPLIT_LABEL = {"loeo_ag_eco9": "LOEO_AG_ECO9", "random5": "random5_sites", "huc2_group9": "HUC2_group9"}


# ------------------------------------------------------------------------- data

def load_design(proc: Path) -> tuple[pd.DataFrame, list[str]]:
    visits = pd.read_parquet(proc / "visits.parquet")
    counts = pd.read_parquet(proc / "benthic_counts.parquet",
                             columns=["visit_id", "survey", "taxon_name", "count_300"])
    chem = pd.read_parquet(proc / "chemistry_wide.parquet")
    v = visits[(visits["survey"] == SURVEY) & (visits["visit_no"] == "1")]
    c = counts[(counts["survey"] == SURVEY) & counts["visit_id"].isin(v["visit_id"])]
    cm = c.pivot_table(index="visit_id", columns="taxon_name", values="count_300", aggfunc="sum", fill_value=0)
    cm = cm.loc[:, (cm > 0).sum(0) >= MIN_SAMPLES_PER_TAXON]
    ra = cm.div(cm.sum(1).replace(0, np.nan), axis=0).fillna(0.0)
    feats = np.sqrt(ra)
    feats.columns = [f"taxon::{t}" for t in feats.columns]
    meta = v.set_index("visit_id")[["site_id", "ag_eco9", "huc2_nars", "lat", "lon", "elevation_m"]]
    tcols = sorted({c for c, _, _ in TARGETS.values()})
    df = feats.join(meta, how="inner").join(chem.set_index("visit_id")[tcols + ["cond_unit_source"]], how="inner")
    return df, list(feats.columns)


def target_vector(df: pd.DataFrame, name: str) -> np.ndarray:
    col, tf, floor = TARGETS[name]
    x = df[col].to_numpy(float)
    if tf == "log10":
        x = np.log10(np.clip(x, floor, None))
    return x


# ----------------------------------------------------------------------- splits

def loeo_splits(sites: pd.Series, groups: pd.Series) -> list[ES.SiteSplit]:
    out = []
    site_group = pd.Series(groups.to_numpy(), index=sites.to_numpy())
    site_group = site_group[~site_group.index.duplicated()]
    for g in sorted(site_group.unique()):
        test = sorted(site_group.index[site_group == g])
        train = sorted(site_group.index[site_group != g])
        out.append(ES.SiteSplit(train=train, val=[], test=test, method="leave_one_group_out",
                                params={"group_column": "ag_eco9", "held_out": g}))
    return out


def random_kfold_splits(sites: pd.Series, k: int, seed: int) -> list[ES.SiteSplit]:
    uniq = np.array(sorted(pd.unique(sites)))
    perm = np.random.default_rng(seed).permutation(len(uniq))
    folds = np.array_split(uniq[perm], k)
    out = []
    for i, f in enumerate(folds):
        test = sorted(f.tolist())
        train = sorted(set(uniq) - set(test))
        out.append(ES.SiteSplit(train=train, val=[], test=test, method="random_site_kfold",
                                params={"k": k, "fold": i, "seed": seed}))
    return out


def group_kfold_splits(sites: pd.Series, groups: pd.Series, k: int) -> list[ES.SiteSplit]:
    """Whole groups (HUC2 regions) assigned to k folds, largest first to the smallest fold
    (the sklearn GroupKFold rule, used by the research-phase script)."""
    sg = pd.Series(groups.to_numpy(), index=sites.to_numpy())
    sg = sg[~sg.index.duplicated()]
    sizes = sg.value_counts()
    load = np.zeros(k)
    fold_of = {}
    for g, n in sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0])):
        f = int(np.argmin(load))
        fold_of[g] = f
        load[f] += n
    out = []
    for f in range(k):
        test = sorted(sg.index[sg.map(fold_of) == f])
        train = sorted(sg.index[sg.map(fold_of) != f])
        out.append(ES.SiteSplit(train=train, val=[], test=test, method="group_kfold",
                                params={"group_column": "huc2_nars", "k": k, "fold": f,
                                        "groups": sorted(g for g, ff in fold_of.items() if ff == f)}))
    return out


def check_partition(folds: list[ES.SiteSplit], all_sites) -> None:
    tests = {f"fold{i}": f.test for i, f in enumerate(folds)}
    ES.assert_disjoint(**tests)
    covered = set().union(*map(set, tests.values()))
    if covered != set(all_sites):
        raise AssertionError("test folds do not cover every site exactly once")
    for f in folds:
        ES.assert_disjoint(train=f.train, test=f.test)


# ------------------------------------------------------------------------- model

def oof_predict(X: np.ndarray, y: np.ndarray, site: np.ndarray, folds: list[ES.SiteSplit], seed: int,
                n_jobs: int) -> np.ndarray:
    from sklearn.ensemble import RandomForestRegressor

    pred = np.full(len(y), np.nan)
    for f in folds:
        tr = np.isin(site, f.train) & np.isfinite(y)
        te = np.isin(site, f.test) & np.isfinite(y)
        if not te.any():
            continue
        m = RandomForestRegressor(**RF_PARAMS, n_jobs=n_jobs, random_state=seed)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-jobs", type=int, default=40)
    ap.add_argument("--extra-seeds", type=int, nargs="*", default=[1, 2])
    ap.add_argument("--schemes", nargs="*", default=["loeo_ag_eco9", "random5", "huc2_group9"])
    ap.add_argument("--no-log", action="store_true", help="compute only; do not write the registry")
    ap.add_argument("--runlog", action="store_true", help="also wrap in sentinel.core.runlog.TrainingRun")
    ap.add_argument("--supersede-previous", action="store_true",
                    help="mark the current registry entry with the same experiment/model/split/target as superseded")
    ap.add_argument("--i-know-2018-19-is-a-test-cycle", dest="ack_test_cycle", action="store_true",
                    help="required: NRSA 2018-19 is the M5 external test cycle (docs/preregistration/"
                         "bioassess_protocol.md §2); re-running this research baseline re-exposes it")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not a.ack_test_cycle:
        from sentinel.models.bioassess.data import HoldoutAccessError

        raise HoldoutAccessError("NRSA 2018-19 is the M5 external test cycle; this research baseline fits and scores "
                                 "within it. Re-run only with --i-know-2018-19-is-a-test-cycle and a lead decision.")

    proc = paths.processed("nars_streams")
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = paths.run_dir(WORKSTREAM, f"{EXPERIMENT}_{stamp}")
    df, taxa_cols = load_design(proc)
    site = df["site_id"].to_numpy()
    n_dup = len(site) - len(set(site))
    if n_dup:
        # A few sites carry VISIT_NO == 1 in both 2018 and 2019; both rows are kept (as in the
        # research script) and always fall in the same fold / bootstrap cluster (grouped by site).
        log.info("%d sites have two first-visit rows (kept; grouped by site in splits and CIs)", n_dup)
    geo = df[["lat", "lon", "elevation_m"]].astype(float)
    geo = geo.fillna({"elevation_m": geo["elevation_m"].median()})
    feature_sets = {"community": df[taxa_cols].to_numpy(float), "geography": geo.to_numpy(float)}
    feature_sets["community+geography"] = np.hstack([feature_sets["community"], feature_sets["geography"]])
    log.info("design: %d first-visit samples (%d sites), %d taxa features, %d ecoregions", len(df),
             len(set(site)), len(taxa_cols), df["ag_eco9"].nunique())

    schemes = {}
    if "loeo_ag_eco9" in a.schemes:
        schemes["loeo_ag_eco9"] = loeo_splits(df["site_id"], df["ag_eco9"])
    if "random5" in a.schemes:
        schemes["random5"] = random_kfold_splits(df["site_id"], 5, seed=0)
    if "huc2_group9" in a.schemes:
        schemes["huc2_group9"] = group_kfold_splits(df["site_id"], df["huc2_nars"], 9)
    lat_of = dict(zip(df["site_id"], df["lat"]))
    lon_of = dict(zip(df["site_id"], df["lon"]))
    split_doc = {}
    for name, folds in schemes.items():
        check_partition(folds, site)
        split_doc[name] = {"folds": [{"report": ES.split_report(f, lat_of, lon_of), "split": f.to_dict()}
                                     for f in folds]}
    split_path = run_dir / "splits.json"
    split_path.write_text(json.dumps(split_doc, indent=1, default=str))

    input_sha256 = {f: file_sha256(proc / f) for f in ("visits.parquet", "benthic_counts.parquet",
                                                         "chemistry_wide.parquet")}
    config = {"survey": SURVEY, "visit_filter": "VISIT_NO == 1", "n_rows": int(len(df)),
              "n_sites": int(len(set(site))), "min_samples_per_taxon": MIN_SAMPLES_PER_TAXON,
              "features": "sqrt(relative abundance of TOTAL300 per TARGET_TAXON)", "n_taxa": len(taxa_cols),
              "geography": ["lat", "lon", "elevation_m (median-filled)"], "rf": RF_PARAMS, "seed": 0,
              "targets": {k: {"column": c, "transform": t, "floor": f} for k, (c, t, f) in TARGETS.items()},
              "metric": "r2 of pooled out-of-fold predictions", "n_boot": a.n_boot}

    def run_all(run=None):
        rows, preds = [], {}
        for sname, folds in schemes.items():
            for fname, X in feature_sets.items():
                for tname in TARGETS:
                    y = target_vector(df, tname)
                    ok = np.isfinite(y)
                    p0 = oof_predict(X, y, site, folds, 0, a.n_jobs)
                    r2 = EM.r2(y[ok], p0[ok])
                    ci = EB.bootstrap_ci(EM.r2, y[ok], p0[ok], groups=site[ok], n_boot=a.n_boot, seed=0)
                    seeds = {0: r2}
                    diag = {}
                    if sname == "loeo_ag_eco9":
                        for s in a.extra_seeds:
                            ps = oof_predict(X, y, site, folds, s, a.n_jobs)
                            seeds[s] = EM.r2(y[ok], ps[ok])
                        eco = df["ag_eco9"].to_numpy()[ok]
                        cc = EB.bootstrap_ci(EM.r2, y[ok], p0[ok], groups=eco, n_boot=a.n_boot, seed=0)
                        diag["ci_ecoregion_cluster"] = [cc["ci_low"], cc["ci_high"]]
                        diag["r2_within_held_out_ecoregion"] = {
                            g: EM.r2(y[ok][eco == g], p0[ok][eco == g]) for g in sorted(set(eco))}
                    ref = RESEARCH.get((sname, fname), {}).get(tname)
                    row = {"scheme": sname, "features": fname, "target": tname, "n": int(ok.sum()), "r2": r2,
                           "ci_low": ci["ci_low"], "ci_high": ci["ci_high"], "rmse": EM.rmse(y[ok], p0[ok]),
                           "spearman": EM.spearman(y[ok], p0[ok]), "seed_r2": seeds,
                           "seed_sd": float(np.std(list(seeds.values()), ddof=1)) if len(seeds) > 1 else None,
                           "research_r2": ref, "delta_vs_research": None if ref is None else r2 - ref,
                           "diagnostics": diag}
                    rows.append(row)
                    preds[(sname, fname, tname)] = p0
                    log.info("%s %-20s %-20s R2=%.3f [%.3f, %.3f] research=%s", sname, fname, tname, r2,
                             ci["ci_low"], ci["ci_high"], ref)
                    if run is not None:
                        run.metric(f"{sname}/{fname}/{tname}/r2", r2)
        return rows, preds

    if a.runlog:
        from sentinel.core.runlog import TrainingRun

        with TrainingRun(WORKSTREAM, EXPERIMENT, config=config, data=["nars_streams"],
                         plan_ref="RESEARCH_PLAN.md §5.5", artifacts_dir=str(run_dir)) as run:
            rows, preds = run_all(run)
    else:
        rows, preds = run_all()

    res = pd.DataFrame(rows)
    res.to_csv(run_dir / "results.csv", index=False)
    pr = pd.DataFrame({"visit_id": df.index, "site_id": site, "ag_eco9": df["ag_eco9"].to_numpy()})
    for (s, f, t), p in preds.items():
        pr[f"{s}|{f}|{t}|pred"] = p
    for t in TARGETS:
        pr[f"obs|{t}"] = target_vector(df, t)
    pr.to_parquet(run_dir / "oof_predictions.parquet", index=False)
    (run_dir / "config.json").write_text(json.dumps(config, indent=1))

    ids = []
    previous = {}
    if a.supersede_previous and not a.no_log:
        cur = registry_frame()
        if not cur.empty:
            cur = cur[(cur["workstream"] == WORKSTREAM) & (cur["experiment"] == EXPERIMENT)]
            previous = {(m, sp, t): i for m, sp, t, i in cur[["model", "split", "target", "id"]].itertuples(index=False)}
    if not a.no_log:
        for r in rows:
            model = f"RandomForest[{r['features']}]"
            e = log_result(
                supersedes=previous.get((model, SPLIT_LABEL[r["scheme"]], r["target"])),
                workstream=WORKSTREAM, experiment=EXPERIMENT, model=model,
                metric="r2", value=r["r2"], split=SPLIT_LABEL[r["scheme"]], n=r["n"],
                ci_low=r["ci_low"], ci_high=r["ci_high"],
                ci_method=f"percentile bootstrap over sites (B={a.n_boot}), pooled out-of-fold predictions",
                target=r["target"], is_baseline=True, config=config, data_manifests=["nars_streams"],
                artifacts=[str(split_path), str(run_dir / "results.csv"), str(run_dir / "oof_predictions.parquet")],
                notes=("NRSA 2018-19 first visits, community->chemistry RF baseline (research protocol, "
                       "docs/research/04 §A.4). CV estimate of a fixed configuration, not a test-set score."
                       + ("" if r["research_r2"] is None else
                          f" Research-phase value {r['research_r2']:.2f}; delta {r['delta_vs_research']:+.3f}.")),
                extra={"seed_r2": r["seed_r2"], "seed_sd": r["seed_sd"], "rmse": r["rmse"],
                       "spearman": r["spearman"], "research_r2": r["research_r2"], "scheme": r["scheme"],
                       "resampling_unit": "site", "input_sha256": input_sha256, **r["diagnostics"]},
            )
            ids.append(e["id"])
        res["registry_id"] = ids
        res.to_csv(run_dir / "results.csv", index=False)
    print(res[["scheme", "features", "target", "n", "r2", "ci_low", "ci_high", "research_r2",
               "delta_vs_research"] + (["registry_id"] if ids else [])].to_string(index=False))
    print(f"artifacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
