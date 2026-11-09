#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M5-X development runner (protocol ``bioassess_x_protocol.md`` §6–§7, amendment v4). Training-pool rows only.

Sub-commands
------------
* ``prepare``   — build/cached design + frozen keys, the development folds (``x_track/dev/folds_dev_v1.json``),
  and log the ordering-evidence row ``bioassess_x_dev/folds`` (fold sha256 + protocol sha256).
* ``baselines`` — out-of-fold predictions of the §6 tabular comparators (``--block``, ``--models``), every
  (model, variant, fold, target) fitted on the fold's training ids only; cached per (block, model, variant, seed)
  in ``x_track/dev/preds/<block>/``.
* ``export-r``  — per-fold inputs for the R comparators (bio.infer, GJAM): training rows with labels, pseudo-test
  rows without labels (``x_track/dev/r_in/``).
* ``ingest-r``  — R outputs -> prediction files.
* ``score``     — development metrics of every prediction file (r2_anomaly + location-cluster CI, B = 2,000),
  variant choice per (block, model, target), registry rows (experiment ``bioassess_x_dev``) logged once per file
  hash.

Every fit goes through ``xtrack.assert_fit_ids`` (folds are re-checked when read).

    python -B scripts/eval/bioassess_x_dev.py prepare
    python -B scripts/eval/bioassess_x_dev.py baselines --block X-L --models rf_community,wapls2
    python -B scripts/eval/bioassess_x_dev.py score
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "6")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sentinel.core import paths  # noqa: E402
from sentinel.core.registry import file_sha256, log_result  # noqa: E402
from sentinel.core.runlog import TrainingRun  # noqa: E402
from sentinel.models.bioassess import xdev as XD  # noqa: E402
from sentinel.models.bioassess import xtrack as X  # noqa: E402

EXPERIMENT = "bioassess_x_dev"
FOLDS = XD.DEV_DIR / "folds_dev_v1.json"
PREDS = XD.DEV_DIR / "preds"
PROTOCOL = paths.REPO_ROOT / "docs/preregistration/bioassess_x_protocol.md"
MANIFESTS = ["nars_streams", "nars_lakes", "uk_ea_ecology", "taxonomy_harmonised"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def context():
    S, frozen, D = XD.load_context()
    folds = XD.read_folds(FOLDS, S, frozen)
    return S, frozen, D, folds


def pred_path(block, model, variant, seed=0) -> Path:
    return PREDS / block / f"{model}__{variant}__s{seed}.parquet"


def write_preds(df: pd.DataFrame, f: Path) -> None:
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp.parquet")
    df.to_parquet(tmp)
    tmp.replace(f)


# --------------------------------------------------------------------------------------------- prepare
def cmd_prepare(a) -> int:
    t0 = time.time()
    S, frozen, D = XD.load_context(rebuild=a.rebuild)
    log("design", D.info, f"{time.time() - t0:.0f}s")
    folds = {b: XD.dev_folds(S, frozen, D, b) for b in XD.BLOCKS}
    sha = XD.save_folds(folds, FOLDS)
    rep = {b: [{"g": f.g, **f.report} for f in fl] for b, fl in folds.items()}
    for b in XD.BLOCKS:
        r = pd.DataFrame(rep[b])
        log(b, "folds", len(r), "test", int(r.n_test.sum()), "train", int(r.n_train.min()), "-", int(r.n_train.max()),
            "buffer", int(r.removed_buffer.sum()))
    (XD.DEV_DIR / "folds_dev_v1_report.json").write_text(json.dumps({"sha256": sha, "folds": rep}, indent=1))
    if a.log:
        log_result(workstream="bioassess", experiment=f"{EXPERIMENT}/folds", model="dev_folds", metric="n_folds",
                   value=float(sum(len(v) for v in folds.values())), split="dev", n=int(len(D.samples)),
                   notes="M5-X development folds (amendment v4.1); written before any M5-X fit",
                   data_manifests=MANIFESTS, artifacts=[str(FOLDS)],
                   extra={"folds_sha256": sha, "protocol_sha256": file_sha256(PROTOCOL),
                          "split_sha256": XD.SPLIT_SHA256, "n_frozen_keys": len(frozen), "report": rep})
    return 0


# --------------------------------------------------------------------------------------------- baselines
def run_model(model, block, D, TM, folds, S, frozen, seed, device, n_jobs, variants=("same", "multi")) -> list[Path]:
    out = []
    for variant in variants:
        tg = [t for t in XD.TARGETS[block] if variant in XD.variants_for(t)]
        if model in ("region_mean", "rf_geography", "nearest_reference") + XD.INDEX_MODELS and variant == "multi":
            continue
        f = pred_path(block, model, variant, seed)
        if f.exists():
            log("exists", f.name)
            continue
        cfg = {"model": model, "block": block, "variant": variant, "seed": seed, "folds_sha256": file_sha256(FOLDS)}
        with TrainingRun("bioassess", f"{EXPERIMENT}/{block}/{model}/{variant}", config=cfg, data=MANIFESTS,
                         plan_ref="bioassess_x_protocol.md §6, amendment v4", iteration="dev",
                         artifacts_dir=str(f.parent)) as run:
            parts = []
            t0 = time.time()
            for fo in folds[block]:
                X.assert_fit_ids(XD.keys_of(fo.train), S, frozen)       # fit guard (protocol §7)
                P = pd.DataFrame(index=fo.test, columns=list(XD.TARGETS[block]), dtype=float)
                for t in tg:
                    P[t] = XD.fit_predict(model, block, t, D, TM, fo.train, fo.test, variant=variant, seed=seed,
                                          n_jobs=n_jobs, device=device)
                P["fold"] = fo.g
                parts.append(P)
                log(block, model, variant, fo.g, f"{time.time() - t0:.0f}s")
            df = pd.concat(parts)
            write_preds(df, f)
            run.metric("n_rows", len(df))
            run.metric("minutes", (time.time() - t0) / 60)
        out.append(f)
    return out


def cmd_baselines(a) -> int:
    S, frozen, D, folds = context()
    if a.device == "cuda":
        import torch
        torch.set_num_threads(2)
    TM = XD.TaxonMatrix(D)
    blocks = XD.BLOCKS if a.block == "all" else (a.block,)
    for b in blocks:
        models = XD.models_for(b) if a.models == "all" else [m for m in a.models.split(",") if m in XD.models_for(b)]
        for m in models:
            run_model(m, b, D, TM, folds, S, frozen, a.seed, a.device, a.n_jobs, tuple(a.variants.split(",")))
    return 0


# --------------------------------------------------------------------------------------------- R exchange
def cmd_export_r(a) -> int:
    """Per fold: same-continent labelled training rows (taxon names + counts + labels) and pseudo-test taxa."""
    S, frozen, D, folds = context()
    out = XD.DEV_DIR / "r_in"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"name": sorted(D.tokens["name"].dropna().astype(str).unique())}).to_csv(out / "names.csv",
                                                                                         index=False)
    TM = XD.TaxonMatrix(D)
    for b in XD.BLOCKS:
        for k, fo in enumerate(folds[b]):
            X.assert_fit_ids(XD.keys_of(fo.train), S, frozen)
            write_r_fold(out / b / f"fold{k:02d}", b, fo, D, TM, seed=0)
    log("exported", out)
    return 0


def write_r_fold(d: Path, block, fo, D, TM, seed=0, test_labels=False):
    d.mkdir(parents=True, exist_ok=True)
    tr = fo.train[D.samples.loc[fo.train, "source"].isin(XD.CONTINENT[block]).to_numpy()]
    Y = pd.DataFrame({t: XD.labels(D, block, t, tr, "same") for t in XD.TARGETS[block]}, index=tr)
    Y = Y[Y.notna().any(axis=1)]
    tr = Y.index
    tok = D.tokens
    cols = ["visit_id", "name", "mixed_otu", "count"]
    tok[tok["visit_id"].isin(set(tr))][[c for c in cols if c in tok]].to_csv(d / "train_tokens.csv", index=False)
    tok[tok["visit_id"].isin(set(fo.test))][[c for c in cols if c in tok]].to_csv(d / "test_tokens.csv", index=False)
    Y.rename_axis("visit_id").reset_index().to_csv(d / "train_y.csv", index=False)
    s = D.samples.loc[tr]
    s[["ag_eco9"]].rename_axis("visit_id").reset_index().to_csv(d / "train_region.csv", index=False)
    pd.DataFrame({"visit_id": fo.test}).to_csv(d / "test_ids.csv", index=False)
    (d / "meta.json").write_text(json.dumps({"block": block, "g": fo.g, "n_train": len(tr), "n_test": len(fo.test),
                                             "targets": list(XD.TARGETS[block]), "seed": seed}))


def cmd_ingest_r(a) -> int:
    S, frozen, D, folds = context()
    for b in XD.BLOCKS:
        if pred_path(b, a.method, "same", 0).exists():
            log("already ingested", a.method, b)
            continue
        parts = []
        for k, fo in enumerate(folds[b]):
            f = XD.DEV_DIR / "r_out" / a.method / b / f"fold{k:02d}.csv"
            if not f.exists():
                log("missing", f)
                parts = None
                break
            p = pd.read_csv(f).set_index("visit_id").reindex(fo.test)
            p = p[[t for t in XD.TARGETS[b] if t in p]].reindex(columns=list(XD.TARGETS[b]))
            p["fold"] = fo.g
            parts.append(p)
        if parts:
            write_preds(pd.concat(parts), pred_path(b, a.method, "same", 0))
            log("ingested", a.method, b)
    return 0


# --------------------------------------------------------------------------------------------- score
def truth(D, block, ids):
    return pd.DataFrame({t: XD.labels(D, block, t, ids, "same") for t in XD.TARGETS[block]}, index=ids)


def cmd_score(a) -> int:
    S, frozen, D, folds = context()
    logged_f = XD.DEV_DIR / "logged.json"
    logged = json.loads(logged_f.read_text()) if logged_f.exists() else {}
    rows = []
    for b in XD.BLOCKS:
        ids = pd.Index(np.concatenate([fo.test for fo in folds[b]]))
        Y = truth(D, b, ids)
        region = D.samples.loc[ids, "ag_eco9"].astype(str).to_numpy()
        cluster = D.samples.loc[ids, "location_id"].astype(str).to_numpy()
        for f in sorted((PREDS / b).glob("*.parquet")):
            model, variant, seed = f.stem.split("__")
            P = pd.read_parquet(f).reindex(ids)
            h = file_sha256(f)
            for t in XD.TARGETS[b]:
                if t not in P or not np.isfinite(P[t].to_numpy(float)).any():
                    continue
                key = f"{b}|{f.stem}|{t}"
                if key in logged and logged[key]["sha"] == h:
                    rows.append(logged[key]["row"])
                    continue
                sc = XD.score(Y[t].to_numpy(float), P[t].to_numpy(float), region, cluster, n_boot=2000)
                row = {"block": b, "model": model, "variant": variant, "seed": seed, "target": t,
                       "n": sc["n"], "r2_anomaly": sc["r2_anomaly"], "r2": sc["r2"], "rmse": sc["rmse"],
                       "spearman": sc["spearman"], "r2_region_median": sc["r2_region_median"],
                       "ci_low": sc["ci"]["r2_anomaly"]["ci_low"], "ci_high": sc["ci"]["r2_anomaly"]["ci_high"],
                       "r2_ci": [sc["ci"]["r2"]["ci_low"], sc["ci"]["r2"]["ci_high"]],
                       "n_missing_pred": int((~np.isfinite(P[t].to_numpy(float)) & Y[t].notna().to_numpy()).sum())}
                if a.log and np.isfinite(row["r2_anomaly"]):
                    sentinel_side = model.startswith(("bsx_", "ens_"))
                    log_result(workstream="bioassess", experiment=EXPERIMENT, model=model_name(model, variant),
                               metric="r2_anomaly", value=row["r2_anomaly"], split=f"{b}_dev", n=row["n"],
                               ci_low=row["ci_low"], ci_high=row["ci_high"],
                               ci_method="percentile location-cluster bootstrap B=2000", target=t,
                               is_baseline=not sentinel_side,
                               config={"model": model, "variant": variant, "seed": seed, "block": b},
                               data_manifests=MANIFESTS, artifacts=[str(f)],
                               notes=f"M5-X development ({XD.DEV_PSEUDO[b]}); seed 0 (amendment v4.3)",
                               extra={k: row[k] for k in ("r2", "rmse", "spearman", "r2_region_median", "r2_ci",
                                                          "n_missing_pred")} | {"pred_sha256": h})
                if a.log:                       # only rows actually written to the registry are remembered
                    logged[key] = {"sha": h, "row": row}
                rows.append(row)
            logged_f.write_text(json.dumps(logged))
    R = pd.DataFrame(rows)
    R.to_csv(XD.DEV_DIR / "dev_scores.csv", index=False)
    log("scored rows", len(R))
    return 0


def cmd_snapshot(a) -> int:
    """Ordering-evidence registry row: sha256 + mtime of the amended protocol and the M5-X development code."""
    S, frozen, D, folds = context()
    for b in XD.BLOCKS:
        for fo in folds[b]:
            X.assert_fit_ids(XD.keys_of(fo.train), S, frozen)
    repo = paths.REPO_ROOT
    files = [PROTOCOL] + [repo / f for f in (
        "sentinel/models/bioassess/xdev.py", "sentinel/models/bioassess/xbioset.py", "scripts/eval/bioassess_x_dev.py",
        "scripts/train/bioassess_x_bioset.py", "scripts/eval/bioassess_x_select.py", "scripts/eval/bioassess_x_final.py",
        "scripts/eval/bioassess_x_freeze_candidate.py", "scripts/eval/bioassess_x_bioinfer.R",
        "scripts/eval/bioassess_x_gjam.R", "tests/test_bioassess_x_dev.py", "tests/test_bioassess_models.py")] + [FOLDS]
    snap = {str(f): {"sha256": file_sha256(f), "mtime": time.strftime("%Y-%m-%dT%H:%M:%S",
                                                                      time.localtime(f.stat().st_mtime))} for f in files}
    log_result(workstream="bioassess", experiment="bioassess_x/prereg_snapshot", model="amendment_snapshot",
               metric="n_files", value=float(len(snap)), split="prereg", n=len(snap), notes=a.note,
               extra={"files": snap, "snapshot_time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    log("snapshot logged", len(snap))
    return 0


def model_name(model, variant):
    base = XD.REGISTRY_NAME.get(model, model)
    return base + ("" if variant == "same" else " (multi-source)")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--log", action="store_true")
    p = sub.add_parser("baselines")
    p.add_argument("--block", default="all")
    p.add_argument("--models", default="all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-jobs", type=int, default=6)
    p.add_argument("--variants", default="same,multi")
    sub.add_parser("export-r")
    p = sub.add_parser("ingest-r")
    p.add_argument("--method", required=True)
    p = sub.add_parser("snapshot")
    p.add_argument("--note", required=True)
    p = sub.add_parser("score")
    p.add_argument("--log", action="store_true")
    a = ap.parse_args()
    return {"prepare": cmd_prepare, "baselines": cmd_baselines, "export-r": cmd_export_r, "ingest-r": cmd_ingest_r,
            "score": cmd_score, "snapshot": cmd_snapshot}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
