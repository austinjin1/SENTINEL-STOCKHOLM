#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M5-X power check (pre-registration). Empirical SE of the paired Δ (ensemble C − per-target comparator)
on the primary metric r2_anomaly from the frozen Test A predictions (location-cluster bootstrap), scaled by
sqrt(n_clusters_A / n_clusters_X) to each M5-X test block; MDE at 80 % power for two-sided alpha (best case)
and alpha / m (Holm worst case, m = family size). Uses only already-scored Test A predictions and the
cluster counts of the frozen split table (no M5-X label is read).

    python -B scripts/eval/bioassess_x_power.py --run-dir <final_run_dir> --splits <splits.parquet> --out <json>
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse, json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import norm  # noqa: E402

TARGETS = ("log10_tp_ug_l", "log10_tn_mg_l", "log10_cond_us_cm", "log10_chloride_mg_l", "log10_turbidity_ntu", "ph_lab")
C = "Ensemble[BioSet[bioset_st_flat]+geography+RandomForest[community+geography]+TabPFNv2[community]]"


def one(F, rec, t, metric):
    from sentinel.models.bioassess import evaluate as E
    y = F[f"obs|{t}"].to_numpy(float)
    pv = E.paired_vs(y, F[f"pred|{C}|{t}"].to_numpy(float), F[f"pred|{rec['sota_comparators'][t]}|{t}"].to_numpy(float),
                     F["ag_eco9"].to_numpy(), F["location_id"].to_numpy(), metric=metric, n_boot=2000)
    return t, metric, pv["diff"], (pv["ci_high"] - pv["ci_low"]) / (2 * 1.96)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--family-size", type=int, default=36)
    a = ap.parse_args()
    d = Path(a.run_dir)
    F = pd.read_parquet(d / "pred_testA.parquet")
    rec = json.loads((d / "freeze_record.json").read_text())
    nA = F["location_id"].nunique()
    res = Parallel(n_jobs=6)(delayed(one)(F, rec, t, m) for t in TARGETS for m in ("r2_anomaly", "r2"))
    S = pd.read_parquet(a.splits)
    blocks = {b: int(S[(S["block"] == b) & (S["role"] == "test")]["cluster"].nunique()) for b in ("X-L", "X-U")}
    z80 = norm.ppf(0.8)
    out = {"n_clusters_testA": int(nA), "n_clusters": blocks, "family_size": a.family_size, "rows": []}
    for t, m, diff, se in res:
        row = {"target": t, "metric": m, "testA_delta": diff, "testA_se": se}
        for b, n in blocks.items():
            seX = se * np.sqrt(nA / n)
            row[f"{b}_se"] = seX
            row[f"{b}_mde_alpha05"] = (norm.ppf(0.975) + z80) * seX
            row[f"{b}_mde_holm_worst"] = (norm.ppf(1 - 0.05 / (2 * a.family_size)) + z80) * seX
        out["rows"].append(row)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    df = pd.DataFrame(out["rows"])
    print(f"clusters: Test A {nA}, {blocks}")
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
