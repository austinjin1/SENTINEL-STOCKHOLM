#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D1 SafeSwim driver (docs/preregistration/downstream_safeswim.md): Chicago beach FIB exceedance.

Phase 1 = validation years only. Test-role years are dropped inside ``safeswim.build`` right after
role assignment (``common.drop_protected``); this script has no option to score test data.

    python -B scripts/train/downstream_safeswim.py --build-only            # labels/feature inspection
    python -B scripts/train/downstream_safeswim.py --tasks d1b_nowcast      # fit on train, score val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.models.downstream import common as C  # noqa: E402

C.limit_threads(C.N_THREADS)

from sentinel.core.runlog import TrainingRun  # noqa: E402
from sentinel.eval import metrics as M  # noqa: E402
from sentinel.models.downstream import safeswim as S  # noqa: E402

TASKS = {"d1a_forecast": ("d1a", "forecast"), "d1b_nowcast": ("d1b", "nowcast"), "d1b_forecast": ("d1b", "forecast")}
VAL_SPLIT = {"d1a": "val_2014_2015", "d1b": "val_2022_2023"}
# Pre-specified primary comparator per task (protocol §6); others are reported as secondary.
COMPARATOR = {"d1a_forecast": "persistence", "d1b_nowcast": "official_city_model", "d1b_forecast": "persistence"}
PRIMARY = {"d1a_forecast": ["persistence"], "d1b_nowcast": ["official_city_model", "feature_beach_gm"],
           "d1b_forecast": ["persistence"]}
MODELS = ("logistic_l2", "gbm_hgb")


def describe(X: pd.DataFrame, cols: list[str]) -> dict:
    by = X.groupby("role").agg(n_beach_days=("y", "size"), n_pos=("y", "sum"), n_beaches=("beach_key", "nunique"),
                               first=("date", "min"), last=("date", "max"))
    by["prevalence"] = by["n_pos"] / by["n_beach_days"]
    miss = X[cols].isna().mean().sort_values(ascending=False)
    extra = {}
    if "official_pred" in X:
        extra["official_pred_available"] = X.groupby("role")["official_pred"].apply(lambda s: int(s.notna().sum())).to_dict()
    extra["persist_eligible"] = X.groupby("role")["persist_eligible"].sum().astype(int).to_dict()
    return {"by_role": json.loads(by.to_json(date_format="iso")), "missing_frac": miss.round(4).to_dict(),
            "n_features": len(cols), **extra}


def _rowsets(task_name: str, scores: dict) -> dict:
    """Row subsets on which every listed scorer is defined (identical rows, review fix 1)."""
    n = len(next(iter(scores.values())))
    rs = {"all": np.ones(n, bool), "persist_rows": np.isfinite(scores["persistence"])}
    if "official_city_model" in scores:
        rs["official_rows"] = np.isfinite(scores["official_city_model"])
    return rs


def evaluate(task_name: str, X: pd.DataFrame, cols: list[str], run_dir: Path, register: bool, cfg: dict,
             models=MODELS, tag: str = "", rowsets=None, keep_baselines=None) -> dict:
    task, mode = TASKS[task_name]
    C.assert_no_test(X)
    tr, ev = X[X["role"] == "train"], X[X["role"] == "val"].copy()
    week = S.cluster_ids(ev)
    beach = ev["beach_key"].to_numpy()
    y = ev["y"].to_numpy(float)
    scores = S.baseline_scores(tr, ev, task, mode)
    if keep_baselines:
        scores = {k: v for k, v in scores.items() if k in keep_baselines}
    baselines = list(scores)
    for m in models:
        C.log_config_tried(S.APP, f"{task_name}/{m}{tag}", {**S.MODEL_CONFIGS[m], "features": cols})
        scores[m] = S.fit_predict(m, tr[cols], tr["y"].to_numpy(), ev[cols])
    pred = ev[["beach_key", "date", "y", "role"]].copy()
    for k, v in scores.items():
        pred[f"score_{k}"] = v
    pred_p = run_dir / f"{task_name}{tag.replace('/', '_')}_val_predictions.parquet"
    pred.to_parquet(pred_p, index=False)
    out = {"task": task_name, "comparator": COMPARATOR[task_name], "cluster_unit": S.CLUSTER_UNIT,
           "n_train": int(len(tr)), "n_val": int(len(ev)), "rowsets": {}}
    split = VAL_SPLIT[task]
    for rname, rm in _rowsets(task_name, scores).items():
        if rowsets and rname not in rowsets:
            continue
        defined = [k for k, v in scores.items() if np.isfinite(v[rm]).all()]
        yy, wk, bc = y[rm], week[rm], beach[rm]
        res = {}
        for name in defined:
            s_ = scores[name][rm]
            prob = name in models or name in ("prevalence", "location_only", "climatology_beach_month")
            r = C.binary_scores(yy, s_, wk, prob=prob)
            r["sens_at_spec85"] = S.sens_at_spec(yy, s_)
            res[name] = r
        paired = {}
        for m in [k for k in models if k in defined]:
            for base in [b for b in baselines if b in defined and b != "prevalence"]:
                paired[f"{m}_vs_{base}"] = {
                    "week": C.paired(S.fast_auroc, yy, scores[m][rm], scores[base][rm], wk),
                    "beach": C.paired(S.fast_auroc, yy, scores[m][rm], scores[base][rm], bc)}
        out["rowsets"][rname] = {"n": int(rm.sum()), "n_pos": int(yy.sum()), "scores": res, "paired_auroc": paired}
        if not register:
            continue
        exp = f"{task_name}{tag}/rows={rname}"
        for name, r in res.items():
            for metric in ("auroc", "auprc", "brier", "sens_at_spec85"):
                v = r.get(metric)
                if v is None or not np.isfinite(v):
                    continue
                ci = {"ci_low": r[f"{metric}_ci"][0], "ci_high": r[f"{metric}_ci"][1]} if f"{metric}_ci" in r else None
                C.log_metric(app=S.APP, experiment=exp, model=name, metric=metric, value=v, split=split, n=r["n"], ci=ci,
                             ci_method=f"cluster bootstrap B={C.N_BOOT}, unit={S.CLUSTER_UNIT}" if ci else None,
                             target=f"{task}_exceedance", is_baseline=name not in models, config=cfg,
                             data_manifests=S.DATA_MANIFESTS, artifacts=[str(pred_p)],
                             extra={"n_pos": r["n_pos"], "prevalence": r["prevalence"], "rows": rname})
        for k, pp in paired.items():
            for unit, p in pp.items():
                if np.isfinite(p.get("diff", np.nan)):
                    C.log_metric(app=S.APP, experiment=exp, model=k, metric=f"auroc_diff_paired_{unit}", value=p["diff"],
                                 split=split, n=int(rm.sum()), ci=p,
                                 ci_method=f"paired cluster bootstrap B={C.N_BOOT}, unit="
                                           + (S.CLUSTER_UNIT if unit == "week" else "beach"),
                                 target=f"{task}_exceedance", config=cfg, data_manifests=S.DATA_MANIFESTS,
                                 extra={"p_value": p.get("p_value"), "rows": rname})
    return out


def incumbent_reproduction(X: pd.DataFrame) -> dict:
    """Official City model AUROC on development years (2017-2019, 2021-2023), persistence-eligible rows,
    vs the research value 0.702 (2017-2025, n=9,436); tolerance +-0.03 (protocol §4a)."""
    d = X[X["official_pred"].notna() & (X["persist_eligible"] == 1) & (X["date"].dt.year >= 2017)]
    r = C.binary_scores(d["y"], d["official_pred"], S.cluster_ids(d))
    r["research_value"] = 0.702
    r["within_tolerance"] = bool(abs(r["auroc"] - 0.702) <= 0.03) if np.isfinite(r.get("auroc", np.nan)) else False
    return r


def leave_band_out(task_name: str, X: pd.DataFrame, cols: list[str]) -> dict:
    """Secondary spatial check: fit on train years of the other latitude bands, score val years of the band."""
    out = {}
    for band in sorted(X["lat_band"].dropna().unique()):
        tr = X[(X["role"] == "train") & (X["lat_band"] != band)]
        ev = X[(X["role"] == "val") & (X["lat_band"] == band)]
        if ev["y"].sum() == 0 or len(ev) < 50:
            out[int(band)] = {"n": int(len(ev)), "note": "too few rows/positives"}
            continue
        s = S.fit_predict("gbm_hgb", tr[cols], tr["y"].to_numpy(), ev[cols])
        out[int(band)] = C.binary_scores(ev["y"], s, S.cluster_ids(ev), prob=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--build-only", action="store_true", help="labels + features + descriptive counts only")
    ap.add_argument("--with-sensors", action="store_true", help="add lake-wide d-1 sensor features (ablation)")
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--features", default="base", help="feature-set name in safeswim.FEATURE_SETS")
    ap.add_argument("--baselines", default="", help="iterations: only these baselines (identical across runs)")
    ap.add_argument("--rowsets", default="", help="restrict scored row sets (iterations), e.g. official_rows")
    ap.add_argument("--tag", default="", help="experiment-name suffix for iteration variants, e.g. /it1_rf")
    a = ap.parse_args()
    split, sha = S.load_frozen_split()
    models = tuple(m.strip() for m in a.models.split(",") if m.strip())
    run_tag = a.run_name or ("build_only" if a.build_only else pd.Timestamp.now().strftime("%Y%m%d-%H%M%S"))
    rd = C.run_dir(S.APP, run_tag)
    summary = {"split_sha256": sha}
    feature_sets = [f.strip() for f in a.features.split(",") if f.strip()]
    import contextlib

    # one slot for the whole invocation (all tasks × feature sets): the shared slot is a FIFO queue
    slot = contextlib.nullcontext() if a.build_only else C.cpu_slot(f"safeswim/{a.tasks}/{a.tag or 'base'}")
    with slot:
        for tn in [t.strip() for t in a.tasks.split(",") if t.strip()]:
            task, mode = TASKS[tn]
            X = S.build(task, mode, split, with_sensors=a.with_sensors)
            for fs in feature_sets:
                tag = a.tag if len(feature_sets) == 1 else f"{a.tag}_{fs}"
                key = tn if len(feature_sets) == 1 else f"{tn}|{fs}"
                cols = S.feature_columns(X, task, mode, with_sensors=a.with_sensors, feature_set=fs)
                summary[key] = {"describe": describe(X, cols)}
                if a.build_only:
                    continue
                cfg = {"task": tn, "split_sha256": sha, "models": {m: S.MODEL_CONFIGS[m] for m in models},
                       "features": cols, "feature_set": fs, "with_sensors": a.with_sensors, "tag": tag}
                with TrainingRun(C.WORKSTREAM, f"{S.APP}/{tn}{tag}", config=cfg, data=list(S.DATA_MANIFESTS),
                                 plan_ref=C.PLAN_REF, artifacts_dir=str(rd)) as run:
                    r = evaluate(tn, X, cols, rd, register=not a.no_register, cfg=cfg, models=models, tag=tag,
                                 rowsets=[x for x in a.rowsets.split(",") if x] or None,
                                 keep_baselines=[x for x in a.baselines.split(",") if x] or None)
                    r["leave_band_out_gbm"] = leave_band_out(tn, X, cols) if "gbm_hgb" in models and not tag else None
                    if tn == "d1b_nowcast" and not tag:
                        r["incumbent_reproduction"] = rep = incumbent_reproduction(X)
                        if not a.no_register:
                            C.log_metric(app=S.APP, experiment="incumbent_reproduction", model="official_city_model",
                                         metric="auroc", value=rep["auroc"], split="dev_2017_2023", n=rep["n"],
                                         ci={"ci_low": rep["auroc_ci"][0], "ci_high": rep["auroc_ci"][1]},
                                         ci_method=f"cluster bootstrap B={C.N_BOOT}, unit={S.CLUSTER_UNIT}",
                                         is_baseline=True, target="d1b_exceedance", config=cfg,
                                         data_manifests=S.DATA_MANIFESTS,
                                         extra={"research_value": 0.702, "within_tolerance": rep["within_tolerance"]})
                    rs0 = next(iter(r["rowsets"].values()))
                    for m in models:
                        run.metric(f"val_auroc_{m}", rs0["scores"][m]["auroc"])
                    summary[key]["results"] = r
                C.write_json(rd / "summary.json", summary)
    C.write_json(rd / "summary.json", summary)
    print(json.dumps({k: (v.get("describe") if isinstance(v, dict) else v) for k, v in summary.items()}, indent=1,
                     default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
