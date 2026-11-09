#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Phase-1 cross-validation of SENTINEL-BioAssess models and baselines on NRSA 2008-09 + 2013-14.

Protocol: docs/preregistration/bioassess_protocol.md (written before any model was trained).
Development data only — the loader refuses NRSA 2018-19 / 2023-24 (external tests).

For every (model, scheme): out-of-fold predictions (mean over seeds for stochastic models), pooled
R², per-held-out-ecoregion R², anomaly R², between-region R², RMSE, Spearman; percentile cluster
bootstrap CIs (B = --n-boot) with resampling unit = location cluster, plus an ecoregion-unit CI in
``extra``. Each (model, scheme) is one ``TrainingRun``; each metric one registry entry
(workstream 'bioassess', experiment 'bioassess_phase1_cv', split 'nrsa0814/<scheme>_cv').

    python -B scripts/train/bioassess_cv.py --models rf_community gbm_community ... --schemes loeo block50
    python -B scripts/train/bioassess_cv.py --models bioset_st bioset_ds --schemes loeo --gpu
"""

from __future__ import annotations

import os

# CPU budget (docs/CONTRIBUTING_V2.md): <= 6 threads per job, one CPU-heavy job per workstream. BLAS pools are pinned to 1 thread
# before numpy is imported; model-level parallelism (RF/LightGBM n_jobs, joblib, torch) is capped
# at MAX_THREADS below.
MAX_THREADS = 6   # CONTRIBUTING_V2 "Shared-host allocation" (2026-10-24 22:30)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"   # joblib/loky workers do not inherit ``python -B``
os.environ["SENTINEL_GPUS"] = "4"   # M5 uses only the shared queue GPU 4 (CONTRIBUTING_V2)

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
from sentinel.core.registry import file_sha256, log_result  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
WORKSTREAM = "bioassess"
EXPERIMENT = "bioassess_phase1_cv"
STOCHASTIC = {"rf_community", "rf_taxa_only", "rf_geography", "rf_community_geography", "gbm_community",
              "rf_context", "rf_community_context",
              "tabpfn_community"}
NEEDS_GPU = {"tabpfn_community"}
BASELINE_PARAM_KEY = {"rf_community": "rf", "rf_taxa_only": "rf", "rf_geography": "rf",
                      "rf_community_geography": "rf", "gbm_community": "gbm", "knn_community": "knn_community",
                      "rf_context": "rf", "rf_community_context": "rf",
                      "wapls2": "wapls", "tabpfn_community": "tabpfn", "nearest_reference": "nearest_reference",
                      "ecoregion_mean": None}
REG_METRICS = ("r2", "r2_anomaly", "r2_region_median")
log = logging.getLogger("bioassess_cv")


def resolve_pretrained(cfg: dict, cli_path: str | None):
    if not cfg.get("pretrained"):
        return None, None
    p = cli_path
    if os.environ.get("SENTINEL_TRACK") == "M5-X":
        raise SystemExit("refusing: M5-X never uses resolve_pretrained/'auto' (C-MXR-2); use xtrack.load_pretrained_encoder")
    if p is None and cfg["pretrained"] == "auto":
        ptr = paths.RUNS_ROOT / "bioassess" / "uk_pretrain_latest.json"
        if not ptr.exists():
            raise FileNotFoundError("candidate needs a UK-pretrained encoder; run scripts/train/bioassess_pretrain_uk.py")
        p = json.loads(ptr.read_text())["path"]
    elif p is None:
        p = cfg["pretrained"]
    from sentinel.models.bioassess.cv import PretrainedEncoder

    return PretrainedEncoder(p), {"path": str(p), "sha256": file_sha256(p)}


def score(design, oof: pd.DataFrame, targets, n_boot: int, per_seed: dict | None, n_jobs: int) -> dict:
    from joblib import Parallel, delayed

    from sentinel.models.bioassess import evaluate as E

    region = design.samples["ag_eco9"].to_numpy()
    cluster = design.samples["location_id"].to_numpy()

    def one(t):
        y = design.Y[t].to_numpy(float)
        p = oof[t].to_numpy(float)
        s = E.summary(y, p, region)
        s["ci"] = E.cis(y, p, region, cluster, n_boot=n_boot)
        if per_seed:
            s["seed_r2"] = {int(k): E.summary(y, v[t].to_numpy(float), region)["r2"] for k, v in per_seed.items()}
            s["seed_r2_anomaly"] = {int(k): E.anomaly_r2(y, v[t].to_numpy(float), region) for k, v in per_seed.items()}
        return t, s

    return dict(Parallel(n_jobs=max(1, min(n_jobs - 1, len(targets))))(delayed(one)(t) for t in targets))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--schemes", nargs="+", default=["loeo"], choices=["loeo", "block50", "random5"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-jobs", type=int, default=MAX_THREADS, help=f"capped at {MAX_THREADS}")
    ap.add_argument("--gpu", action="store_true", help="acquire one GPU via sentinel.core.gpu (prefer --gpu-prefer)")
    ap.add_argument("--gpu-prefer", type=int, default=4)
    ap.add_argument("--geography", action="store_true", help="BioSet community+geography variant (control)")
    ap.add_argument("--context", action="store_true",
                    help="BioSet community+context variant: geography + sampling day-of-year")
    ap.add_argument("--pretrained", default=None, help="explicit UK-pretrained encoder checkpoint")
    ap.add_argument("--bioset-config", default=str(REPO / "configs/bioassess/bioset.yaml"))
    ap.add_argument("--baseline-config", default=str(REPO / "configs/bioassess/baselines.yaml"))
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--max-folds", type=int, default=None, help="smoke tests only (never with registry logging)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a.n_jobs = max(1, min(a.n_jobs, MAX_THREADS))
    import torch

    torch.set_num_threads(2 if a.gpu else MAX_THREADS)
    geo_mode = "context" if a.context else bool(a.geography)
    suffix = "+context" if a.context else ("+geography" if a.geography else "")
    if a.max_folds is not None and not a.no_log:
        ap.error("--max-folds requires --no-log")
    import yaml

    from sentinel.core.runlog import TrainingRun
    from sentinel.models.bioassess import baselines as B
    from sentinel.models.bioassess import cv as CV
    from sentinel.models.bioassess import splits as SP
    from sentinel.models.bioassess.data import TARGET_NAMES, assert_dev_only, input_hashes, load_nrsa

    device = "cpu"
    bioset_cands = yaml.safe_load(Path(a.bioset_config).read_text())["candidates"]
    if a.gpu:
        from sentinel.core.gpu import acquire_gpu

        g = acquire_gpu("bioassess_cv", prefer=a.gpu_prefer)
        log.info("GPU %d", g)
        device = "cuda"
    elif any(m in NEEDS_GPU for m in a.models):
        ap.error("TabPFN needs --gpu")
    base_cfg = yaml.safe_load(Path(a.baseline_config).read_text())
    design = load_nrsa()
    assert_dev_only(design.samples)
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = paths.run_dir(WORKSTREAM, f"phase1_cv_{stamp}")
    hashes = input_hashes()
    (run_dir / "design_info.json").write_text(json.dumps(design.info, indent=1))
    folds_of = {}
    for sch in a.schemes:
        folds = SP.make_folds(design.samples, sch)
        SP.check_folds(folds, design.samples)
        (run_dir / f"splits_{sch}.json").write_text(json.dumps(SP.fold_reports(folds, design.samples), default=str))
        folds_of[sch] = folds if a.max_folds is None else folds[: a.max_folds]
    rows = []
    for model in a.models:
        is_bioset = model in bioset_cands
        for sch, folds in folds_of.items():
            split_name = SP.SCHEME_SPLIT_NAME[sch]
            if is_bioset:
                cfg = CV.load_bioset_config(a.bioset_config, model)
                pre, pre_info = resolve_pretrained(cfg, a.pretrained)
                reg_model = f"BioSet[{model}]" + suffix
                config = {"model": model, "bioset": cfg, "seeds": a.seeds, "geography": geo_mode,
                          "pretrained": pre_info, "ensemble": "mean of seeds", "scheme": sch}
                seeds = a.seeds
            else:
                key = BASELINE_PARAM_KEY[model]
                params = dict(base_cfg.get(key) or {}) if key else {}
                reg_model = B.REGISTRY_MODEL_NAME[model]
                seeds = a.seeds if model in STOCHASTIC else [0]
                config = {"model": model, "params": params, "seeds": seeds, "scheme": sch,
                          "ensemble": "mean of seeds" if len(seeds) > 1 else "single fit"}
            config.update({"design": design.info, "input_sha256": hashes, "protocol": "docs/preregistration/bioassess_protocol.md",
                           "protocol_sha256": file_sha256(REPO / "docs/preregistration/bioassess_protocol.md")})
            diag = []
            with TrainingRun(WORKSTREAM, f"{EXPERIMENT}/{model}/{sch}", config=config,
                             data=["nars_streams", "taxonomy_harmonised"] + (["uk_ea_ecology"] if is_bioset and cfg.get("pretrained") else []),
                             plan_ref="RESEARCH_PLAN.md §5.5; docs/preregistration/bioassess_protocol.md §6",
                             artifacts_dir=str(run_dir)) as run:
                if is_bioset:
                    oof, per_seed, diag = CV.oof_bioset(cfg, design, folds, seeds=seeds, device=device,
                                                        geography=geo_mode, pretrained=pre)
                else:
                    per_seed = {}
                    for s in seeds:
                        per_seed[s] = CV.oof_baseline(model, design, folds, seed=s, params=params, n_jobs=a.n_jobs)
                    oof = sum(per_seed[s][list(TARGET_NAMES)] for s in seeds) / len(seeds)
                    if len(seeds) == 1:
                        per_seed = {}
                res = score(design, oof, TARGET_NAMES, a.n_boot, per_seed or None, a.n_jobs)
                for t, s in res.items():
                    for m in REG_METRICS:
                        run.metric(f"{t}/{m}", s[m])
                run.metric("mean_r2", float(np.mean([res[t]["r2"] for t in TARGET_NAMES])))
                run.metric("mean_r2_anomaly", float(np.mean([res[t]["r2_anomaly"] for t in TARGET_NAMES])))
                if diag:
                    run.metric("median_best_epoch", float(np.median([d["best_epoch"] for d in diag])))
                    run.note(f"n_params {diag[0]['n_params']}")
            tag = f"{model}{('_ctx' if a.context else '_geo') if (is_bioset and suffix) else ''}_{sch}"
            out = pd.DataFrame({"visit_id": design.samples.index, "location_id": design.samples["location_id"].to_numpy(),
                                "ag_eco9": design.samples["ag_eco9"].to_numpy()})
            for t in TARGET_NAMES:
                out[f"pred|{t}"] = oof[t].to_numpy()
                out[f"obs|{t}"] = design.Y[t].to_numpy()
                for s, v in per_seed.items():
                    out[f"pred_seed{s}|{t}"] = v[t].to_numpy()
            oof_path = run_dir / f"oof_{tag}.parquet"
            out.to_parquet(oof_path, index=False)
            if diag:
                pd.DataFrame(diag).to_csv(run_dir / f"diag_{tag}.csv", index=False)
            for t, s in res.items():
                row = {"model": reg_model, "scheme": sch, "target": t, "n": s["n"],
                       **{m: s[m] for m in ("r2", "r2_anomaly", "r2_region_median", "r2_region_min", "r2_region_max",
                                            "r2_between_region", "rmse", "spearman")},
                       "r2_ci": (s["ci"]["r2"]["ci_low"], s["ci"]["r2"]["ci_high"]),
                       "r2_anomaly_ci": (s["ci"]["r2_anomaly"]["ci_low"], s["ci"]["r2_anomaly"]["ci_high"]),
                       "seed_r2": s.get("seed_r2"), "oof": str(oof_path)}
                if not a.no_log:
                    for m in REG_METRICS:
                        e = log_result(
                            workstream=WORKSTREAM, experiment=EXPERIMENT, model=reg_model, metric=m, value=s[m],
                            split=split_name, n=s["n"], ci_low=s["ci"][m]["ci_low"], ci_high=s["ci"][m]["ci_high"],
                            ci_method=(f"percentile cluster bootstrap, resampling unit = location cluster (1 km), "
                                       f"B={a.n_boot}; out-of-fold predictions"),
                            target=t, is_baseline=not is_bioset, config=config,
                            data_manifests=["nars_streams", "taxonomy_harmonised"],
                            artifacts=[str(oof_path), str(run_dir / f"splits_{sch}.json")],
                            notes=("Phase-1 CV on NRSA 2008-09+2013-14 development data (not a test score). "
                                   "r2 = pooled; r2_anomaly = within-region skill (anomalies from ecoregion means); "
                                   "r2_region_median = median of per-ecoregion R2."),
                            extra={"scheme": sch, "rmse": s["rmse"], "spearman": s["spearman"],
                                   "r2_by_region": s["r2_by_region"], "r2_region_min": s["r2_region_min"],
                                   "r2_region_max": s["r2_region_max"], "r2_region_mean": s["r2_region_mean"],
                                   "r2_between_region": s["r2_between_region"],
                                   "ci_ecoregion_unit": s["ci"][m].get("ci_ecoregion"),
                                   "seed_r2": s.get("seed_r2"), "seed_r2_anomaly": s.get("seed_r2_anomaly"),
                                   "resampling_unit": "location_cluster", "region_definition": "ag_eco9"})
                        row[f"id_{m}"] = e["id"]
                rows.append(row)
                log.info("%-34s %-8s %-20s R2=%.3f [%.3f,%.3f] anom=%.3f regmed=%.3f", reg_model, sch, t, s["r2"],
                         s["ci"]["r2"]["ci_low"], s["ci"]["r2"]["ci_high"], s["r2_anomaly"], s["r2_region_median"])
            pd.DataFrame(rows).to_csv(run_dir / "results.csv", index=False)
    print(pd.DataFrame(rows).drop(columns=["oof", "seed_r2"]).to_string(index=False))
    print(f"artifacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
