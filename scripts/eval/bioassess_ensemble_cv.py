#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Equal-weight ensemble of existing out-of-fold predictions (no fitting; development CV only).

The members' OOF predictions come from the current registry entries of experiment
'bioassess_phase1_cv' for the same split; rows must match exactly. Equal weights are fixed in
advance (no weights are learned, so the ensemble adds no new leakage path), and the result is
logged as model ``Ensemble[<a>+<b>]`` with the same metrics/CIs as scripts/train/bioassess_cv.py.

    python -B scripts/eval/bioassess_ensemble_cv.py --members "BioSet[bioset_st_flat]" "TabPFNv2[community]" \
        --split nrsa0814/loeo_cv [--no-log]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.registry import log_result, registry_frame  # noqa: E402

TARGETS = ("log10_tp_ug_l", "log10_tn_mg_l", "log10_cond_us_cm", "log10_chloride_mg_l", "log10_turbidity_ntu",
           "ph_lab")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--split", default="nrsa0814/loeo_cv")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()
    from joblib import Parallel, delayed

    from sentinel.models.bioassess import evaluate as E

    df = registry_frame()
    df = df[(df["experiment"] == "bioassess_phase1_cv") & (df["split"] == a.split) & (df["metric"] == "r2")]
    df = df.sort_values("timestamp").groupby(["model", "target"], as_index=False).last()
    oofs, entries = [], []
    for m in a.members:
        e = df[df["model"] == m]
        if e.empty:
            raise KeyError(f"no current entry for {m} on {a.split}")
        e = e.iloc[0]
        entries.append(e["id"])
        oofs.append(pd.read_parquet(next(x for x in e["artifacts"] if "/oof_" in x)))
    base = oofs[0]
    for o in oofs[1:]:
        if not (o["visit_id"].to_numpy() == base["visit_id"].to_numpy()).all():
            raise AssertionError("member OOF rows differ")
    name = "Ensemble[" + "+".join(a.members) + "]"
    out = base[["visit_id", "location_id", "ag_eco9"]].copy()
    for t in TARGETS:
        out[f"pred|{t}"] = np.mean([o[f"pred|{t}"].to_numpy() for o in oofs], 0)
        out[f"obs|{t}"] = base[f"obs|{t}"].to_numpy()
    run_dir = paths.run_dir("bioassess", "ensembles")
    slug = "_".join("".join(ch if ch.isalnum() else "_" for ch in m) for m in a.members)
    oof_path = run_dir / f"oof_ens_{slug}_{a.split.split('/')[-1]}.parquet"
    out.to_parquet(oof_path, index=False)
    region, cluster = out["ag_eco9"].to_numpy(), out["location_id"].to_numpy()

    def one(t):
        y, p = out[f"obs|{t}"].to_numpy(float), out[f"pred|{t}"].to_numpy(float)
        s = E.summary(y, p, region)
        s["ci"] = E.cis(y, p, region, cluster, n_boot=a.n_boot)
        return t, s

    res = dict(Parallel(n_jobs=min(a.n_jobs, 6))(delayed(one)(t) for t in TARGETS))
    config = {"members": a.members, "member_entries": entries, "weights": "equal (fixed a priori)",
              "split": a.split}
    for t, s in res.items():
        print(f"{name} {t}: R2={s['r2']:.3f} [{s['ci']['r2']['ci_low']:.3f},{s['ci']['r2']['ci_high']:.3f}] "
              f"anom={s['r2_anomaly']:.3f} regmed={s['r2_region_median']:.3f}")
        if a.no_log:
            continue
        for m in ("r2", "r2_anomaly", "r2_region_median"):
            log_result(workstream="bioassess", experiment="bioassess_phase1_cv", model=name, metric=m, value=s[m],
                       split=a.split, n=s["n"], ci_low=s["ci"][m]["ci_low"], ci_high=s["ci"][m]["ci_high"],
                       ci_method=f"percentile cluster bootstrap, resampling unit = location cluster (1 km), B={a.n_boot}; "
                                 "out-of-fold predictions",
                       target=t, is_baseline=False, config=config, data_manifests=["nars_streams"],
                       artifacts=[str(oof_path)],
                       notes="Phase-1 CV (development data); equal-weight ensemble of existing OOF predictions; "
                             "improvement cycle 1 (RESEARCH_PLAN §15.5 amendment).",
                       extra={"rmse": s["rmse"], "spearman": s["spearman"], "r2_by_region": s["r2_by_region"],
                              "r2_region_min": s["r2_region_min"], "r2_region_max": s["r2_region_max"],
                              "r2_between_region": s["r2_between_region"],
                              "ci_ecoregion_unit": s["ci"][m].get("ci_ecoregion"), "resampling_unit": "location_cluster",
                              "region_definition": "ag_eco9", "scheme": a.split.split("/")[-1].replace("_cv", "")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
