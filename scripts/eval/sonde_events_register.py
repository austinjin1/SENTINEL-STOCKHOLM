#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Register the out-of-fold (by water year) VAL descriptives of an EW-v2 calibrate run
(``<run>/event_scores.csv`` written by scripts/train/sonde_events.py --phase calibrate) in the
results registry. Val only: refuses any row whose block is not ``val_oof_by_wy``. These are
selection-biased descriptives (the configuration count accompanies them), never headline results.

    python -B scripts/eval/sonde_events_register.py --run <sentinel_runs/sonde/ew_...> [--hazards H1_hypoxia,...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.core.registry import log_result

METRICS = {  # column -> (metric name, ci columns)
    "bss": ("bss_vs_recal_climatology", ("bss_ci_low", "bss_ci_high")),
    "vs_persistence_dbs": ("dbs_vs_persistence", ("vs_persistence_dbs_ci_low", "vs_persistence_dbs_ci_high")),
    "vs_logistic_dbs": ("dbs_vs_logistic", ("vs_logistic_dbs_ci_low", "vs_logistic_dbs_ci_high")),
    "auroc": ("auroc", None),
    "auprc": ("auprc", None),
    "b1_pod_real": ("pod_real_budget1", None),
    "b1_pod_placebo": ("pod_placebo_budget1", None),
    "b1_pod_real_ge24h": ("pod_real_ge24h_budget1", None),
    "b1_pooled_alarm_days_per_30d": ("alarm_days_per_30d_budget1", None),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--hazards", default="")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    run = Path(a.run)
    res = pd.read_csv(run / "event_scores.csv")
    if set(res["block"]) != {"val_oof_by_wy"}:
        raise SystemExit(f"only val_oof_by_wy rows are registrable here, found {sorted(set(res['block']))}")
    if a.hazards:
        res = res[res["hazard"].isin(a.hazards.split(","))]
    cal = json.loads((run / "event_calibration.json").read_text())
    split = f"val/sonde_split_{cal['split_sha256'][:12]}"
    n = 0
    for r in res.to_dict("records"):
        for col, (name, ci) in METRICS.items():
            v = r.get(col)
            if v is None or not np.isfinite(v):
                continue
            lo, hi = (r.get(ci[0]), r.get(ci[1])) if ci else (None, None)
            log_result(workstream="M1-sonde", experiment="ew_v2_val_oof", model=r["method"],
                       metric=f"{name}_L{int(r['lead'])}", value=float(v), split=split, n=int(r["n"]) if np.isfinite(r.get("n", np.nan)) else None,
                       ci_low=None if lo is None or not np.isfinite(lo) else float(lo),
                       ci_high=None if hi is None or not np.isfinite(hi) else float(hi),
                       ci_method="basin-group two-stage block bootstrap (site-months within groups)" if ci else None,
                       target=r["hazard"], is_baseline=r["method"] in ("climatology", "persistence", "threshold", "logistic"),
                       artifacts=[str(run)], data_manifests=["sonde_daily"],
                       notes=f"VAL out-of-fold by water year (Platt/thresholds/stack fitted on the other WY); "
                             f"calibration sha {cal.get('config_hash')}; {a.notes}",
                       extra={"n_pos": r.get("n_pos"), "n_basin_groups_with_onsets": r.get("n_basin_groups_with_onsets"),
                              "run": run.name})
            n += 1
    print(f"registered {n} rows from {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
