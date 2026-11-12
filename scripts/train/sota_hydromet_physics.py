#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Physics-family parity comparators for one M6 SOTA lever configuration (``hydromet_sota_protocol.md`` §5).

    python -B scripts/train/sota_hydromet_physics.py --config configs/hydromet_sota/C1a.yaml
    python -B scripts/train/sota_hydromet_physics.py --config configs/hydromet_sota/C2a.yaml --gpu   # lever 2 (air2stream_q)

Basin-mean ("replace") forcings as for ``air2stream_regional_ws``. Lever 1: reference donors + added sites,
weighted DO-saturation ratio with the weak rows. Lever 2: ``air2stream_q_regional_ws`` with θ from the seed-0 discharge
stack of the same configuration (``sota_gbm_<id>_s0/qhat.npy``); calibrated on GPU with ``--gpu``.
Writes ``predictions/<model>__val.npz`` (FINAL val-role sites × val block).
"""

from __future__ import annotations

import argparse
import copy
import pickle
import sys

import numpy as np


def _replace(data):
    from sentinel.models.hydromet.data import FORCING_VARS

    d = copy.copy(data)
    d.forc = data.forc[:, :, len(FORCING_VARS):]
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    from sentinel.core import paths
    from sentinel.hydromet_sota import gbm as G
    from sentinel.hydromet_sota import physics as PH
    from sentinel.hydromet_sota import runner as SR
    from sentinel.hydromet_sota.data import load_lever_train, load_val_cell
    from sentinel.models.hydromet import runner as HR
    from sentinel.models.hydromet.baselines import BaselineBase
    from sentinel.models.hydromet.evaluate import save_predictions
    from sentinel.models.hydromet.features import FeatureSet

    HR.env_threads(6)
    cfg = SR.load_lever_config(a.config)
    lid = cfg["lever_id"]
    rdir = SR.run_dir(f"sota_phys_{lid}")
    if (rdir / "predictions" / ".done").exists():
        print("done")
        return 0
    log = HR.make_logger(rdir / "log.txt")
    if a.gpu:
        g = SR.acquire_gpu(f"hydromet_sota_phys_{lid}")
        log(f"GPU {g}")
    base = HR.load_config(paths.REPO_ROOT / "configs" / "hydromet" / "baselines_ws_physics.yaml")
    a2s_kw = dict(base["air2stream"], device="cuda" if a.gpu else "cpu")
    ref_dir = paths.RUNS_ROOT / "hydromet" / "basews_v1" / "models"
    ref = BaselineBase.load(ref_dir / "air2stream_regional_ws.pkl")
    SR.write_json(rdir / "config.json", HR.public_config(dict(cfg, physics={"air2stream": a2s_kw, "reference": str(ref_dir)})))
    data, fit_idx, inner_idx, weak_idx, aux_y, dinfo = load_lever_train(**cfg["data"], log=log)
    SR.assert_training_sites_clean(data.sites, where=f"{cfg['model']} training set")
    dphys = _replace(data)
    fs = FeatureSet.build(dphys, lag=0)
    nfinal = dinfo["n_final_train"]
    new_idx = np.array([i for i in fit_idx if i >= nfinal], int)   # added USGS sites (appended after the FINAL ones)
    models = {}
    with SR.training_run(f"sota_phys_{lid}", cfg, rdir) as run:
        a2s = PH.extend_air2stream(ref, dphys, fs, new_idx, name=f"sota_phys_{lid}_air2stream_regional_ws", log=log)
        models[a2s.name] = a2s
        dos = PH.WeightedDOSatRatio(a2s, "air2stream_regional")
        dos.name = f"sota_phys_{lid}_dosat_ratio[air2stream_regional]_ws"
        dos.fit(dphys, fs, fit_idx, inner_idx, weak_idx=weak_idx, weak_share=float(cfg.get("weak_share", 0.0)), log=log)
        models[dos.name] = dos
        if cfg["data"].get("aux_q"):
            qdir = SR.ROOT / f"sota_gbm_{lid}_s0"
            qhat = np.load(qdir / "qhat.npy")
            with open(qdir / "qstack.pkl", "rb") as fh:
                qmeta = pickle.load(fh)
            if qmeta["sites"] != data.sites:
                raise SystemExit("qstack built on a different location set")
            aq = PH.Air2StreamQ(qhat, **a2s_kw)
            aq.name = f"sota_phys_{lid}_air2stream_q_regional_ws"
            aq.fit(dphys, fs, fit_idx, inner_idx, log=log)
            models[aq.name] = aq
            dq = PH.WeightedDOSatRatio(aq, "air2stream_q_regional")
            dq.name = f"sota_phys_{lid}_dosat_ratio[air2stream_q_regional]_ws"
            dq.fit(dphys, fs, fit_idx, inner_idx, weak_idx=weak_idx, weak_share=float(cfg.get("weak_share", 0.0)), log=log)
            models[dq.name] = dq
            run.metric("air2stream_q_donors", float(aq.info.get("n_donors", 0)))
        run.metric("air2stream_donors", float(len(a2s.donor_sites)))
        SR.write_json(rdir / "fit_info.json", {k: m.info for k, m in models.items()})
    del data, dphys, fs
    vdata = load_val_cell(log=log)
    vphys = _replace(vdata)
    fsv = FeatureSet.build(vphys, lag=0)
    if cfg["data"].get("aux_q"):
        fsq = FeatureSet.build(vdata, lag=0)
        qv = G._predict_full(qmeta["model"], fsq, qmeta["columns"], np.arange(len(vdata.sites)))
        aq.set_qhat(qv)
    si = vdata.site_idx(["val"])
    ti = np.flatnonzero(vdata.block_mask("val"))
    for nm, m in models.items():
        p = m.predict(vphys, fsv, si, ti)
        save_predictions(rdir / "predictions" / f"{nm}__val.npz", model=nm, preds=p, sites=[vdata.sites[i] for i in si],
                         dates=vdata.dates[ti], block="val",
                         meta={"smoke": False, "lever_id": lid, "split_sha256": vdata.split.sha256(), "track_r_fold": None,
                               "freeze_sha256": None})
    for nm, m in models.items():
        if hasattr(m, "_theta"):
            m._theta = None   # site-specific θ of the last data set; not part of the fitted model
        m.save(rdir / "models" / f"{nm}.pkl")
    (rdir / "predictions" / ".done").touch()
    log(f"[sota_phys_{lid}] done: {list(models)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
