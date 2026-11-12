#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Track-N val cell (unsensed WQP val-role stations × val block) for the M6 SOTA levers — descriptive
(``hydromet_sota_protocol.md`` §7).

    python -B scripts/eval/sota_hydromet_trackn.py predict --configs C2a [--gpu]
    python -B scripts/eval/sota_hydromet_trackn.py score --configs C2a [--register]

Inputs at the stations exactly as track N (station-cell gridMET + `upstream_forcing_trackn`, StreamCat of the COMID,
`elev_m` = `elevcat`); lever EA-LSTM seeds, lever GBM bag (with the stacked discharge inputs for lever 2) and lever
physics members. Δ median per-station RMSE vs the lever's own GBM bag and physics, basin-group bootstrap; never a claim.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np


def _tn():
    from sentinel.core import paths

    spec = importlib.util.spec_from_file_location("_tn", paths.REPO_ROOT / "scripts" / "eval" / "hydromet_track_n.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _refuse_if_val_hold(what: str) -> None:
    """Val-cell work (§7, descriptive) waits while ``<runs>/hydromet/sota/HOLD_VAL_SCORING`` exists."""
    from sentinel.hydromet_sota import runner as SR

    f = SR.ROOT / "HOLD_VAL_SCORING"
    if f.exists():
        raise SystemExit(f"{what}: on hold ({f}): {f.read_text().strip()}")


def cmd_predict(a) -> int:
    from sentinel.hydromet_sota import gbm as G
    from sentinel.hydromet_sota import runner as SR
    from sentinel.hydromet_sota.lstm import SotaEALSTM
    from sentinel.models.hydromet.baselines import BaselineBase
    from sentinel.models.hydromet.data import FORCING_VARS
    from sentinel.models.hydromet.evaluate import save_predictions
    from sentinel.models.hydromet.features import FeatureSet

    if a.gpu:
        SR.acquire_gpu("hydromet_sota_trackn_predict")
    TN = _tn()
    out = SR.run_dir(a.out) / "predictions"
    data = TN.build_data("val", ("val",), upstream="append")
    si, ti = np.arange(len(data.sites)), np.flatnonzero(data.block_mask("val"))
    meta = {"smoke": False, "split_sha256": data.split.sha256(), "track_r_fold": None, "freeze_sha256": None,
            "track": "N", "roles_sha256": data.info["roles_sha256"]}
    dev = "cuda" if a.gpu else "cpu"

    def save(name, pred):
        save_predictions(out / f"{name}__val.npz", model=name, preds=pred, sites=[data.sites[i] for i in si],
                         dates=data.dates[ti], block="val", meta=meta)

    fs = FeatureSet.build(data, lag=0)
    phys = copy.copy(data)
    phys.forc = data.forc[:, :, len(FORCING_VARS):]
    fsp = FeatureSet.build(phys, lag=0)
    for c in a.configs.split(","):
        for k in range(3):
            ck = SR.ROOT / f"sota_{c}_s{k}" / f"sota_{c}_s{k}_best.pt"
            if (SR.ROOT / f"sota_{c}_s{k}" / "predictions" / ".done").exists():
                m = SotaEALSTM.load(ck)
                save(f"sota_{c}_s{k}", m.predict(data, si, ti, device=dev))
            gdir = SR.ROOT / f"sota_gbm_{c}_s{k}"
            pk = sorted((gdir / "models").glob("*.pkl")) if gdir.exists() else []
            if pk:
                fsk = fs
                if (gdir / "qstack.pkl").exists():
                    with open(gdir / "qstack.pkl", "rb") as fh:
                        q = pickle.load(fh)
                    fsk = G.append_qhat(fs, G._predict_full(q["model"], fs, q["columns"], si))
                pred = {}
                for f in pk:
                    pred.update(BaselineBase.load(f).predict(data, fsk, si, ti))
                save(f"sota_gbm_{c}_bag_s{k}", pred)
        pdir = SR.ROOT / f"sota_phys_{c}" / "models"
        if pdir.exists():
            q0 = SR.ROOT / f"sota_gbm_{c}_s0" / "qstack.pkl"
            for f in sorted(pdir.glob("*.pkl")):
                m = BaselineBase.load(f)
                tm = getattr(m, "temp_model", m)
                if hasattr(tm, "set_qhat"):
                    with open(q0, "rb") as fh:
                        q = pickle.load(fh)
                    tm.set_qhat(G._predict_full(q["model"], fs, q["columns"], si))
                save(m.name, m.predict(phys, fsp, si, ti))
    print(f"predictions in {out}")
    return 0


def cmd_score(a) -> int:
    from sentinel.core import paths
    from sentinel.core.registry import log_result
    from sentinel.hydromet_sota import runner as SR
    from sentinel.models.hydromet import pipeline

    TN = _tn()
    data = TN.build_data("val", ("val",))
    si, ti = np.arange(len(data.sites)), np.flatnonzero(data.block_mask("val"))
    dirs = [SR.ROOT / a.out / "predictions", paths.RUNS_ROOT / "hydromet" / "trackn_val_v1" / "ws_inputs"]
    dirs = [d for d in dirs if Path(d).is_dir()]
    preds, sources = pipeline.load_prediction_set(dirs, data, si, ti, "val", smoke=False, access=data.access)
    res = []
    for c in a.configs.split(","):
        comps = {"temp_c_mean": {"gbm": f"sota_gbm_{c}_bag", "physics": f"sota_phys_{c}_air2stream_regional_ws"},
                 "do_mgl_mean": {"gbm": f"sota_gbm_{c}_bag",
                                 "physics": f"sota_phys_{c}_dosat_ratio[air2stream_regional]_ws"}}
        prim = TN.score_track_n(data, preds, sources, ("val",), "val", comps, [f"sota_{c}"], SR.ROOT / a.out / f"score_{c}")
        prim["config"] = c
        res.append(prim)
    import pandas as pd

    df = pd.concat(res, ignore_index=True)
    df.to_csv(SR.ROOT / a.out / "parity_trackn_val.csv", index=False)
    if a.register:
        for r in df.to_dict("records"):
            for fam in ("gbm", "physics"):
                if np.isfinite(r.get(f"{fam}_diff", np.nan)):
                    log_result(workstream=SR.WORKSTREAM, experiment="M6SOTA/trackn_val_v1", model=f"hydromet_{r['model']}",
                               metric=f"diff_median_site_rmse_vs_own_{fam}", value=float(r[f"{fam}_diff"]),
                               ci_low=float(r[f"{fam}_ci_low"]), ci_high=float(r[f"{fam}_ci_high"]),
                               n=int(r[f"{fam}_n_sites"]), target=r["var"], is_baseline=False,
                               split="track N val-role stations x val block (descriptive, hydromet_sota_protocol.md §7)",
                               ci_method="paired percentile bootstrap resampling basin groups (B=2000); descriptive",
                               extra={"comparator": r[f"{fam}_comparator"], "n_groups": int(r[f"{fam}_n_groups"]),
                                      "every_seed_better": bool(r.get(f"{fam}_every_seed_better", False))},
                               notes="M6 SOTA lever, track-N val cell, descriptive (no Holm claim)")
    print(json.dumps({"rows": len(df)}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("predict", "score"):
        p = sub.add_parser(n)
        p.add_argument("--configs", required=True)
        p.add_argument("--out", default="trackn_val_v1")
        if n == "predict":
            p.add_argument("--gpu", action="store_true")
        else:
            p.add_argument("--register", action="store_true")
    a = ap.parse_args()
    _refuse_if_val_hold(f"track-N val {a.cmd}")
    return cmd_predict(a) if a.cmd == "predict" else cmd_score(a)


if __name__ == "__main__":
    sys.exit(main())
