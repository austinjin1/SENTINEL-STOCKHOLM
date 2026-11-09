#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Quantile-averaging (Vincentization) ensemble of saved SENTINEL-Sonde forecasts.

The ensemble quantile at level τ is the (weighted) mean of the members' τ-quantiles in model
space, on the common rows (members must have identical row keys; mismatches raise). Default
weights are EQUAL (no tuning on validation, so no selection effect); any other weights must be
pre-specified on the command line and are recorded. The result is a regular forecast file that
sonde_score.py / sonde_events.py consume like any model.

    python -B scripts/eval/sonde_ensemble.py --name sentinel_ens_s0 --block val \
        --member chronos2_ft_multi@<ft_run>/forecasts --member lstm@<lstm_run>/forecasts --out <dir>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import numpy as np

from sentinel.core import paths
from sentinel.models.sonde import common, guard

VARS = ["temp_c_mean", "do_mgl_mean", "do_mgl_min", "spcond_uscm_mean", "ph_median", "turb_fnu_mean",
        "turb_fnu_median", "no3no2_mgl_mean"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--block", default="val")
    ap.add_argument("--member", action="append", required=True, help="prefix@forecast_dir (repeatable)")
    ap.add_argument("--weights", default=None, help="comma list, pre-specified (default: equal)")
    ap.add_argument("--variables", default=",".join(VARS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.block in guard.PROTECTED_BLOCKS:
        raise guard.ProtectedDataError("ensembles of protected-block forecasts are built only after the freeze "
                                       "(members must come from sonde_predict.py)")
    members = [m.split("@", 1) for m in a.member]
    w = np.array([float(x) for x in a.weights.split(",")]) if a.weights else np.ones(len(members))
    if w.size != len(members) or (w < 0).any():
        raise ValueError("one non-negative weight per member")
    w = w / w.sum()
    out = Path(a.out) / "forecasts"
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for v in a.variables.split(","):
        fcs = []
        for pre, d in members:
            f = Path(d) / f"{pre}__{v}__{a.block}.npz"
            if not f.is_file():
                break
            fcs.append(common.load_forecast(f))
        if len(fcs) != len(members):
            print(f"[ensemble] {v}: not every member has a forecast; skipped")
            continue
        base = fcs[0]
        for fc in fcs[1:]:
            if not ((fc["site_no"] == base["site_no"]).all() and (fc["issue_date"] == base["issue_date"]).all()
                    and (fc["horizons"] == base["horizons"]).all()):
                raise ValueError(f"{v}: member rows differ (common-row violation)")
        q = np.sort(np.tensordot(w, np.stack([fc["q"].astype(np.float64) for fc in fcs]), axes=1), axis=-1)
        p = out / f"{a.name}__{v}__{a.block}.npz"
        np.savez_compressed(p, q=q.astype(np.float32), site_no=base["site_no"], issue_date=base["issue_date"],
                            horizons=base["horizons"], quantiles=base["quantiles"],
                            header=json.dumps({"model": a.name, "var": v, "block": a.block, "ensemble": members,
                                               "weights": w.tolist()}))
        files[str(p)] = guard.artifact_sha256(p)
        print(f"[ensemble] {v}: {len(base['site_no'])} rows")
    (Path(a.out) / "ensemble_manifest.json").write_text(json.dumps(
        {"name": a.name, "block": a.block, "members": members, "weights": w.tolist(), "files": files,
         "created": _dt.datetime.now().isoformat(timespec="seconds")}, indent=1))
    rec = {"time": _dt.datetime.now().isoformat(timespec="seconds"), "model": a.name, "smoke": False,
           "config_hash": None, "run_dir": str(a.out), "status": "ensemble",
           "members": members, "weights": w.tolist()}
    with open(paths.RUNS_ROOT / "sonde" / "configs_tried.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
