#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""POST-HOC, NON-CONFIRMATORY comparison of M5 with a bio.infer-style ML inference (2026-11-09).

Requested by the lead after the post-results review (REVIEW_REPORT_POST_RESULTS.md, W-M5P-4): the
US EPA taxon–environment maximum-likelihood inference (Yuan 2007; R ``bio.infer``) was not in the
frozen comparator set. This script
1. runs a development-only diagnostic (leave-one-ecoregion-out on NRSA 2008-14) of the
   re-implementation (``sentinel/models/bioassess/bioinfer.py``);
2. fits it under the Test A protocol (per ecoregion g: development rows outside g) and predicts the
   **saved Test A rows** of the frozen single pass (``pred_testA.parquet``; same rows, same
   observations), then compares candidate C (and A) with it by the frozen scorer's paired cluster
   bootstrap (B = 10,000, plus 50-km block and ecoregion sensitivity).
Nothing here changes the frozen pass or its registry rows. Rows go to registry experiment
``bioassess_test_posthoc``; comparisons to ``$SENTINEL_RUNS/bioassess/posthoc_bioinfer_<stamp>/``.
Label rule (stated before running): a SOTA label of C stands only if C's Δ vs bio.infer has a
location-cluster CI above 0 on that target; otherwise it becomes "competitive".

    python -B scripts/eval/bioassess_posthoc_bioinfer.py --run-dir <final_run_dir>
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from sentinel.core import paths  # noqa: E402
from sentinel.core.registry import file_sha256, log_result  # noqa: E402

TARGETS = ("log10_tp_ug_l", "log10_tn_mg_l", "log10_cond_us_cm", "log10_chloride_mg_l", "log10_turbidity_ntu",
           "ph_lab")
MODEL = "bio.infer-ML[community]"
C_NAME = "Ensemble[BioSet[bioset_st_flat]+geography+RandomForest[community+geography]+TabPFNv2[community]]"
A_NAME = "BioSet[bioset_st_flat]"
DEV = ("nrsa0809", "nrsa1314")
TEST_A = "nrsa1819"
N_JOBS = 6
log = logging.getLogger("bioassess_posthoc_bioinfer")
NOTE = ("POST HOC, NON-CONFIRMATORY (2026-11-09; lead request after the post-results review W-M5P-4): bio.infer-style "
        "taxon-environment ML inference (Yuan 2007), re-implemented; not in the frozen record; not Holm-adjusted. "
        "Fitted on development rows only; scored on the saved Test A rows of the frozen pass.")


def presence(design, ids, taxa) -> np.ndarray:
    from sentinel.models.bioassess.features import abundance_matrix

    return (abundance_matrix(design.tokens, ids).reindex(columns=taxa, fill_value=0.0) > 0).to_numpy()


def fit_predict_fold(design, tr, te, seed_unused=0) -> pd.DataFrame:
    from sentinel.models.bioassess.bioinfer import BioInferML
    from sentinel.models.bioassess.features import abundance_matrix

    A = abundance_matrix(design.tokens, tr)
    taxa = list(A.columns)
    Ptr = (A > 0).to_numpy()
    Pte = presence(design, te, taxa)
    out = {}
    for t in TARGETS:
        m = BioInferML().fit(Ptr, design.Y.loc[tr, t].to_numpy(float))
        out[t] = m.predict(Pte)
    return pd.DataFrame(out, index=pd.Index(te))


def loeo(design, rows_pool, test_rows) -> pd.DataFrame:
    """Per ecoregion g of the test rows: fit on pool rows outside g, predict test rows in g."""
    S = design.samples
    folds = []
    for g in sorted(S.loc[test_rows, "ag_eco9"].unique()):
        tr = rows_pool[S.loc[rows_pool, "ag_eco9"] != g]
        te = test_rows[S.loc[test_rows, "ag_eco9"] == g]
        if set(S.loc[tr, "location_id"]) & set(S.loc[te, "location_id"]):
            raise AssertionError("location overlap")
        folds.append((tr, te))
    parts = Parallel(n_jobs=N_JOBS)(delayed(fit_predict_fold)(design, tr, te) for tr, te in folds)
    return pd.concat(parts).loc[test_rows]


def main() -> int:
    from sentinel.core.runlog import TrainingRun
    from sentinel.eval.splits import spatial_blocks
    from sentinel.models.bioassess import evaluate as E
    from sentinel.models.bioassess.data import load_nrsa

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-boot-paired", type=int, default=10000)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    src = Path(a.run_dir)
    out = paths.run_dir("bioassess", f"posthoc_bioinfer_{dt.datetime.now():%Y%m%dT%H%M%S}")
    FA = pd.read_parquet(src / "pred_testA.parquet")
    with TrainingRun("bioassess", "posthoc_bioinfer", config={"model": MODEL, "min_occ": 30, "n_grid": 200, "C": 100.0,
                                                               "source_run": str(src)},
                     data=["nars_streams", "taxonomy_harmonised"], plan_ref="REVIEW_REPORT_POST_RESULTS.md W-M5P-4",
                     artifacts_dir=str(out)) as run:
        design = load_nrsa(DEV + (TEST_A,), allow_holdout=True)
        S = design.samples
        dev = S.index[S["survey"].isin(DEV)]
        # 1. development-only diagnostic (LOEO over NRSA 2008-14)
        Pdev = loeo(design, dev, dev)
        diag = {}
        for t in TARGETS:
            y = design.Y.loc[dev, t].to_numpy(float)
            ok = np.isfinite(y)
            s = E.summary(y[ok], Pdev[t].to_numpy(float)[ok], S.loc[dev, "ag_eco9"].to_numpy()[ok])
            diag[t] = {"r2": s["r2"], "r2_anomaly": s["r2_anomaly"], "n": s["n"]}
        run.note(f"dev LOEO diagnostic: {json.dumps({t: round(v['r2'], 3) for t, v in diag.items()})}")
        # 2. Test A protocol on the saved Test A rows
        tst = pd.Index(FA["visit_id"])
        if set(S.loc[tst, "survey"]) != {TEST_A}:
            raise AssertionError("saved Test A rows are not all NRSA 2018-19")
        P = loeo(design, dev, tst)
        for t in TARGETS:
            if not np.allclose(design.Y.loc[tst, t].to_numpy(float), FA[f"obs|{t}"].to_numpy(float), equal_nan=True):
                raise AssertionError(f"observations differ from the frozen pass for {t}")
        P.to_parquet(out / "pred_testA_bioinfer.parquet")
        region, cluster = FA["ag_eco9"].to_numpy(), FA["location_id"].to_numpy()
        block50 = spatial_blocks(S.loc[tst, "lat"].to_numpy(), S.loc[tst, "lon"].to_numpy(), 50.0)
        res = {"note": NOTE, "source_run": str(src), "pred_testA_sha256": file_sha256(src / "pred_testA.parquet"),
               "dev_loeo_diagnostic": diag, "marginal": {}, "comparisons": []}
        rows = []
        for t in TARGETS:
            y = FA[f"obs|{t}"].to_numpy(float)
            p = P[t].to_numpy(float)
            s = E.summary(y, p, region)
            ci = E.cis(y, p, region, cluster, n_boot=a.n_boot)
            res["marginal"][t] = {"r2": s["r2"], "ci": [ci["r2"]["ci_low"], ci["r2"]["ci_high"]],
                                  "r2_anomaly": s["r2_anomaly"], "rmse": s["rmse"]}
            rows.append(dict(model=MODEL, metric="r2", value=s["r2"], target=t, n=s["n"], ci_low=ci["r2"]["ci_low"],
                             ci_high=ci["r2"]["ci_high"], is_baseline=True,
                             ci_method=f"percentile cluster bootstrap, unit = location cluster, B={a.n_boot}",
                             extra={"r2_anomaly": s["r2_anomaly"], "rmse": s["rmse"], "dev_loeo_r2": diag[t]["r2"]}))
            for cand in (C_NAME, A_NAME):
                pc = FA[f"pred|{cand}|{t}"].to_numpy(float)
                pv = E.paired_vs(y, pc, p, region, cluster, metric="r2", n_boot=a.n_boot_paired)
                sens = {u: E.paired_vs(y, pc, p, region, g, metric="r2", n_boot=a.n_boot)
                        for u, g in (("block50", block50), ("ecoregion", region))}
                seeds = {sd: float(E.summary(y, FA[f"pred_seed{sd}|{cand}|{t}"].to_numpy(float), region)["r2"] - s["r2"])
                         for sd in (0, 1, 2)}
                comp = {"candidate": cand, "reference": MODEL, "target": t, "diff": pv["diff"], "ci_low": pv["ci_low"],
                        "ci_high": pv["ci_high"], "p_value_two_sided": pv["p_value"], "per_seed_delta": seeds,
                        "sensitivity_ci": {u: [v["ci_low"], v["ci_high"]] for u, v in sens.items()},
                        "c_beats_reference": bool(pv["ci_low"] > 0)}
                res["comparisons"].append(comp)
                rows.append(dict(model=cand, metric="delta_r2_vs_bioinfer", value=pv["diff"], target=t,
                                 ci_low=pv["ci_low"], ci_high=pv["ci_high"], is_baseline=False,
                                 ci_method=f"paired percentile cluster bootstrap, unit = location cluster, "
                                           f"B={a.n_boot_paired}",
                                 extra={k: comp[k] for k in ("p_value_two_sided", "per_seed_delta", "sensitivity_ci",
                                                             "c_beats_reference")}))
        (out / "posthoc_bioinfer_comparisons.json").write_text(json.dumps(res, indent=1, default=float))
        for r in rows:
            log_result(workstream="bioassess", experiment="bioassess_test_posthoc", split=f"{TEST_A}/loeo_x_cycle_test",
                       config={"model": MODEL, "min_occ": 30, "n_grid": 200, "C": 100.0, "source_run": str(src),
                               "posthoc": True},
                       data_manifests=["nars_streams", "taxonomy_harmonised"], artifacts=[str(out)], notes=NOTE, **r)
        run.note(f"{len(rows)} posthoc rows logged; C beats bio.infer on "
                 f"{sum(c['c_beats_reference'] for c in res['comparisons'] if c['candidate'] == C_NAME)}/6 targets")
    print(json.dumps({t: res["marginal"][t]["r2"] for t in TARGETS}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
