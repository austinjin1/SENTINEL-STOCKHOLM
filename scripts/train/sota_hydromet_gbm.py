#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""GBM parity comparator for one M6 SOTA lever configuration × seed (``hydromet_sota_protocol.md`` §5).

    python -B scripts/train/sota_hydromet_gbm.py --config configs/hydromet_sota/C2a.yaml --seed 0 --stage qstack
    python -B scripts/train/sota_hydromet_gbm.py --config configs/hydromet_sota/C2a.yaml --seed 0 --target temp_c_mean

* Same lever data as the EA-LSTM configuration (``load_lever_train``), the frozen ``forcing_gbm`` feature set on the
  site-cell + basin-mean channels (``forcing_gbm_ws``), the frozen 3-point grid and LightGBM settings
  (``configs/hydromet/baselines_ws_gbmbag.yaml``), one job per primary target (jobs stay < 2 h, ≤ 6 threads).
* Weak rows (lever 1) weighted to the configuration's weak share; lever 2: stacked discharge-GBM inputs from the
  ``qstack`` stage of the same seed (``qhat.npy`` + ``qstack.pkl``).
* Writes ``predictions/sota_gbm_<id>_bag_s<seed>__val__<target>.npz`` (FINAL val-role sites × val block).
"""

from __future__ import annotations

import argparse
import dataclasses
import pickle
import sys
import time

import numpy as np

PRIMARY = ("temp_c_mean", "temp_c_7dadm", "do_mgl_mean", "do_mgl_min")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stage", default="target", choices=["qstack", "target"])
    ap.add_argument("--target", default=None, choices=PRIMARY)
    a = ap.parse_args()
    from sentinel.core import paths
    from sentinel.hydromet_sota import gbm as G
    from sentinel.hydromet_sota import runner as SR
    from sentinel.hydromet_sota.data import load_lever_train, load_val_cell
    from sentinel.models.hydromet import runner as HR
    from sentinel.models.hydromet.baselines import LGBMConfig
    from sentinel.models.hydromet.evaluate import save_predictions
    from sentinel.models.hydromet.features import FeatureSet

    HR.env_threads(6)
    cfg = SR.load_lever_config(a.config)
    base = HR.load_config(paths.REPO_ROOT / "configs" / "hydromet" / "baselines_ws_gbmbag.yaml")
    lcfg = LGBMConfig(**{**base["lgbm"], "n_jobs": 6})
    grid = [dataclasses.replace(lcfg, **g) for g in base["forcing_gbm_grid"]]
    lid = cfg["lever_id"]
    name = f"sota_gbm_{lid}_bag_s{a.seed}"
    rdir = SR.run_dir(f"sota_gbm_{lid}_s{a.seed}")
    log = HR.make_logger(rdir / f"log_{a.stage}{'_' + a.target if a.target else ''}.txt")
    if a.stage == "target" and not a.target:
        ap.error("--target required")
    out = rdir / "predictions" / f"{name}__val__{a.target}.npz"
    if a.stage == "target" and out.exists():
        log(f"{out.name} exists")
        return 0
    if a.stage == "qstack" and (rdir / "qhat.npy").exists():
        log("qstack exists")
        return 0
    if a.stage == "qstack" and not cfg["data"].get("aux_q"):
        log("no discharge lever in this configuration")
        return 0
    scfg = dict(cfg, seed=a.seed, gbm={"lgbm": dataclasses.asdict(lcfg), "grid": base["forcing_gbm_grid"],
                                       "target": a.target, "stage": a.stage})
    SR.write_json(rdir / "config.json", HR.public_config(scfg))
    t0 = time.time()
    data, fit_idx, inner_idx, weak_idx, aux_y, dinfo = load_lever_train(**cfg["data"], log=log)
    SR.assert_training_sites_clean(data.sites, where=f"{cfg['model']} training set")
    fs = FeatureSet.build(data, lag=0)
    log(f"[{name}] data + features in {(time.time() - t0) / 60:.1f} min; {len(fs.dyn_names)} forcing features")
    qcols = fs.columns()
    if a.stage == "qstack":
        with SR.training_run(f"sota_gbm_{lid}_qstack", scfg, rdir) as run:
            qs = G.QStack(qcols, dataclasses.replace(lcfg, max_rows_per_site=1000), seed=a.seed)
            qhat = qs.fit_oof(data, fs, aux_y[:, :, 0], fit_idx, inner_idx, log=log)
            np.save(rdir / "qhat.npy", qhat)
            with open(rdir / "qstack.pkl", "wb") as fh:
                pickle.dump({"model": qs.full, "columns": qcols, "info": qs.info, "sites": data.sites}, fh)
            SR.write_json(rdir / "qstack_info.json", qs.info)
            run.metric("inner_rmse_logq", qs.info["inner_rmse_logq"])
        return 0
    if cfg["data"].get("aux_q"):
        qp = rdir / "qhat.npy"
        if not qp.exists():
            raise SystemExit("run the qstack stage first")
        with open(rdir / "qstack.pkl", "rb") as fh:
            qmeta = pickle.load(fh)
        if qmeta["sites"] != data.sites:
            raise SystemExit("qstack built on a different location set")
        fs = G.append_qhat(fs, np.load(qp))
    v = a.target
    with SR.training_run(name + f"_{v}", scfg, rdir) as run:
        m = G.WeightedForcingGBM(name, fs.columns(), targets=(v,), cfg=lcfg, seed=a.seed, grid=grid)
        m.fit(data, fs, fit_idx, inner_idx, weak_idx=weak_idx, weak_share=float(cfg.get("weak_share", 0.0)), log=log)
        m.save(rdir / "models" / f"{name}__{v}.pkl")
        SR.write_json(rdir / f"fit_info_{v}.json", m.info)
        sel = m.info.get(v, {})
        if not sel.get("skipped"):
            run.metric(f"inner_rmse_{v}", sel["grid"][sel["selected_grid_index"]]["inner_rmse_model_space"])
    del data, fs
    vdata = load_val_cell(log=log)
    fsv = FeatureSet.build(vdata, lag=0)
    if cfg["data"].get("aux_q"):
        allv = np.arange(len(vdata.sites))
        q = G._predict_full(qmeta["model"], fsv, qmeta["columns"], allv)
        fsv = G.append_qhat(fsv, q)
    si = vdata.site_idx(["val"])
    ti = np.flatnonzero(vdata.block_mask("val"))
    pred = m.predict(vdata, fsv, si, ti)
    save_predictions(out, model=name, preds=pred, sites=[vdata.sites[i] for i in si], dates=vdata.dates[ti], block="val",
                     meta={"smoke": False, "seed": a.seed, "lever_id": lid, "target": v,
                           "split_sha256": vdata.split.sha256(), "track_r_fold": None, "freeze_sha256": None})
    log(f"[{name}] {v}: val predictions written ({(time.time() - t0) / 60:.1f} min total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
