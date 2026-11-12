#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Train one M6 SOTA lever configuration × seed (``docs/preregistration/hydromet_sota_protocol.md`` §4).

    python -B scripts/train/sota_hydromet_lstm.py --config configs/hydromet_sota/C1a.yaml --seed 0

* Data: FINAL train-role sites (+ lever locations) × dates up to the end of the train block only
  (:func:`sentinel.hydromet_sota.data.load_lever_train`); early stopping on the 256 reference inner sites.
* One GPU through ``sentinel.core.gpu`` with ≥ 10,000 MiB physically free. Jobs stop after ``job_minutes`` at an epoch
  boundary with a resume checkpoint and exit code 75; rerunning the same command continues exactly.
* After training: predictions for FINAL val-role sites × val block (descriptive; scored by
  ``scripts/eval/sota_hydromet_score.py``). ``predictions/.done`` marks a finished run.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--cpu", action="store_true", help="debug only: no GPU lock (never for a lever result)")
    a = ap.parse_args()
    from sentinel.hydromet_sota import runner as SR
    from sentinel.models.hydromet import runner as HR

    HR.env_threads(6)
    cfg = SR.load_lever_config(a.config)
    name = a.run_name or f"{cfg['model']}_s{a.seed}"
    rdir = SR.run_dir(name)
    if (rdir / "predictions" / ".done").exists():
        print(f"{name}: already done")
        return 0
    log = HR.make_logger(rdir / "log.txt")
    gpu = None if a.cpu else SR.acquire_gpu(f"hydromet_sota_{name}")
    from sentinel.hydromet_sota.data import load_lever_train, load_val_cell
    from sentinel.hydromet_sota.lstm import ResumeNeeded, SotaEALSTM
    from sentinel.models.hydromet.evaluate import save_predictions
    from sentinel.models.hydromet.guard import artifact_sha256
    from sentinel.models.hydromet.lstm import EALSTMConfig

    resume = rdir / "resume.pt"
    fresh = not resume.exists()
    scfg = dict(cfg, seed=a.seed)
    SR.write_json(rdir / "config.json", HR.public_config(scfg))
    if fresh:
        SR.log_config_tried(cfg["model"], scfg, rdir)
    log(f"[{name}] GPU {gpu}; lever {cfg['lever_id']} data {cfg['data']} weak_share {cfg['weak_share']} aux {cfg['aux']}; "
        f"{'fresh start' if fresh else 'resume'}")
    t0 = time.time()
    data, fit_idx, inner_idx, weak_idx, aux_y, dinfo = load_lever_train(**cfg["data"], log=log)
    SR.assert_training_sites_clean(data.sites, where=f"{cfg['model']} training set")
    SR.write_json(rdir / "data_info.json", {**dinfo, "blocks": ["train"],
                                            "fit_sites": [data.sites[i] for i in fit_idx],
                                            "inner_sites": [data.sites[i] for i in inner_idx],
                                            "weak_stations": [data.sites[i] for i in weak_idx],
                                            "basin_group": dict(zip(data.sites, data.basin_group.tolist()))})
    log(f"[{name}] data loaded in {(time.time() - t0) / 60:.1f} min")
    ecfg = dict(cfg["ealstm"], forcing_lag_days=HR.check_mode(cfg), seed=a.seed)
    ecfg["ratio_bounds"] = tuple(ecfg.get("ratio_bounds", (0.2, 1.6)))
    model = SotaEALSTM(EALSTMConfig(**ecfg), weak_share=float(cfg.get("weak_share", 0.0)), aux=cfg.get("aux") or {},
                       store=cfg.get("store", "cpu"))
    model.info["model_name"] = cfg["model"]
    ckpt = rdir / f"{cfg['model']}_s{a.seed}_best.pt"
    need_resume = False
    with SR.training_run(cfg["model"], scfg, rdir) as run:
        try:
            info = model.fit(data, fit_idx, inner_idx, weak_idx=weak_idx if len(weak_idx) else None,
                             aux_y=aux_y if cfg.get("aux") else None, device="cpu" if a.cpu else "cuda", log=log,
                             checkpoint_path=ckpt, resume_path=resume, job_minutes=float(cfg.get("job_minutes", 100)))
        except ResumeNeeded:
            need_resume = True
            run.note(f"job chunk ended after {len(model.history)} epochs (budget {cfg.get('job_minutes')} min); "
                     "resumed by the next job")
        if not need_resume:
            model.save(ckpt)
            SR.write_json(rdir / "train_info.json", {**info, "history": model.history, "seed": a.seed})
            run.metric("best_inner_val_loss", info["best_inner_val_loss"])
            if info.get("best_inner_primary_mse") is not None:
                run.metric("best_inner_primary_mse", info["best_inner_primary_mse"])
            run.note(f"epochs {info['epochs']}, best epoch {info['best_epoch']}, params {info['n_params']}, fit "
                     f"{info['n_fit_sites']} / inner {info['n_inner_sites']} / weak {info['n_weak_stations']}")
    if need_resume:
        log(f"[{name}] exiting with code {SR.EXIT_RESUME} (resume)")
        return SR.EXIT_RESUME
    del data
    vdata = load_val_cell(log=log)
    si = vdata.site_idx(["val"])
    ti = np.flatnonzero(vdata.block_mask("val"))
    point, dist = model.predict(vdata, si, ti, device="cpu" if a.cpu else "cuda", return_dist=True)
    sha = artifact_sha256(ckpt)
    art = f"{cfg['model']}_s{a.seed}"
    save_predictions(rdir / "predictions" / f"{art}__val.npz", model=art, preds=point,
                     sites=[vdata.sites[i] for i in si], dates=vdata.dates[ti], block="val", dist=dist,
                     meta={"smoke": False, "seed": a.seed, "base_model": cfg["model"], "lever_id": cfg["lever_id"],
                           "forcing_lag_days": 0, "split_sha256": vdata.split.sha256(), "track_r_fold": None,
                           "artifact": str(ckpt), "artifact_sha256": sha, "freeze_sha256": None,
                           "grid_point": cfg["hparams_selected"]})
    SR.write_json(rdir / "checkpoints.json", {art: sha})
    (rdir / "predictions" / ".done").touch()
    log(f"[{name}] done: val predictions for {len(si)} sites; artifact sha {sha[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
