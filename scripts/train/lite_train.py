# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Train SENTINEL-Lite students (≥ 3 seeds) on the train role and score the val role (protocol §5).

    python -m sentinel.core.gpu --name lite_train --prefer 2 -- \
        python -B scripts/train/lite_train.py --config configs/lite/mnv4s_raw.yaml [--seeds 0 1 2] [--smoke]

Per seed: checkpoint + val predictions; the ensemble (mean μ over seeds, σ² = mean σ² + var μ) is scored and
registered (experiment ``lite_val``); per-seed primary metrics go into the registry entry's ``extra``.
"""

from __future__ import annotations

import argparse
import os

from sentinel.models.lite import runner as R

import sys as _sys

R.cpu_threads(6 if "--device" in _sys.argv and "cpu" in _sys.argv else 2)   # shared-host rule: CPU job 6, GPU job 2
os.environ.setdefault("HF_HOME", str(__import__("sentinel.core.paths", fromlist=["x"]).RUNS_ROOT / "hf_cache"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sentinel.models.lite import data as D  # noqa: E402
from sentinel.models.lite import evaluate as E  # noqa: E402
from sentinel.models.lite import features as F  # noqa: E402
from sentinel.models.lite import guard as G  # noqa: E402

EXPERIMENT = "lite_val"


def prepare(lab: pd.DataFrame, name: str, device: str, smoke: bool, stats_path=None):
    from sentinel.models.lite.train import Inputs

    chips = np.asarray(D.cached_chips(lab, name))
    sp = D.scene_norm_params(lab, D.scene_stats(stats_path))
    w = F.pooling_weights(chips, lab["gsd_m"].to_numpy(), stride=4)
    y = lab[[f"y_{t}" for t in D.TARGETS]].to_numpy(float)
    return Inputs(chips, sp, w, y, device), y


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--name", default=None)
    ap.add_argument("--track", default="original", choices=["original", "national"],
                    help="national = protocol §12 (train on the national train role, val = val + val_n)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="cpu: no GPU lock, 6 threads, fp32 (GPU budget is GPUs 0/1 shared; prefer CPU)")
    a = ap.parse_args(argv)
    cfg = R.load_config(a.config)
    name = a.name or cfg.get("name") or os.path.splitext(os.path.basename(a.config))[0]
    if a.smoke:
        cfg["epochs"] = 2
    if a.device == "cuda":
        R.setup_gpu(f"lite_train_{name}")
    import torch

    from sentinel.models.lite import distill as KD
    from sentinel.models.lite.train import DEFAULT_CFG, predict, train

    full = {**DEFAULT_CFG, **{k: v for k, v in cfg.items() if not k.startswith("_")}}
    from sentinel.models.lite import national as N

    split, tr, va, stats_path, experiment = N.train_val_frames(a.track)
    if a.smoke:
        tr, va = tr.head(256), va.head(128)
    prefix = "student_" if a.track == "original" else "natstudent_"
    run_dir = R.run_directory(f"{prefix}{name}", a.smoke)
    full["track"] = a.track
    full["split_sha256"] = split["_sha256"]
    full["smoke"] = bool(a.smoke)          # stored in every checkpoint; registered tools refuse smoke (W9)
    full["seeds"] = a.seeds
    with R.training_run(f"{prefix.rstrip('_')}/{name}", full, run_dir, a.smoke) as run:
        dev = a.device
        tri, ytr = prepare(tr, "train", dev, a.smoke, stats_path)
        vai, yva = prepare(va, "val", dev, a.smoke, stats_path)
        kd = None
        if full.get("kd_weight", 0) > 0:
            kd = KD.load_kd_tensors(tr, full["kd_targets"])
            run.note(f"KD: {int((kd['t_w'] > 0).sum())} teacher cells over {int((kd['t_w'].sum((1, 2)) > 0).sum())} chips")
        mus, sds, per_seed = [], [], {}
        logf = open(run_dir / "train_log.txt", "a")

        def log(s):
            logf.write(s + "\n")
            logf.flush()

        for seed in a.seeds:
            model, hist = train(full, tri, ytr, val=vai, val_y=yva, kd=kd, seed=seed, log=log, device=dev)
            ck = run_dir / f"model_seed{seed}.pt"
            torch.save({"state_dict": model.state_dict(), "cfg": full, "seed": seed, "hist": hist}, ck)
            mu, sd = predict(model, vai, full)
            mus.append(mu)
            sds.append(sd)
            R.save_predictions(run_dir / f"val_pred_seed{seed}.parquet", va,
                               {t: mu[:, j] for j, t in enumerate(D.TARGETS)},
                               {t: sd[:, j] for j, t in enumerate(D.TARGETS)})
            per_seed[seed] = {t: E.score_primary(t, yva[:, j], mu[:, j], va["water_body"], va["site_class"],
                                                 n_boot=200)["value"]
                              for j, t in enumerate(D.TARGETS)}
            for t, v in per_seed[seed].items():
                run.metric(f"seed{seed}/{t}", v)
            del model
            if dev == "cuda":
                torch.cuda.empty_cache()
        mu = np.mean(mus, 0)
        sd = np.sqrt(np.mean(np.square(sds), 0) + np.var(mus, 0))
        R.save_predictions(run_dir / "val_pred_ensemble.parquet", va,
                           {t: mu[:, j] for j, t in enumerate(D.TARGETS)}, {t: sd[:, j] for j, t in enumerate(D.TARGETS)})
        scores = {t: E.score_primary(t, yva[:, j], mu[:, j], va["water_body"], va["site_class"], sigma_log=sd[:, j])
                  for j, t in enumerate(D.TARGETS)}
        for t, s in scores.items():
            run.metric(f"ensemble/{t}/{s['primary']}", s["value"])
        seed_stats = {t: {"mean": float(np.mean([per_seed[s][t] for s in a.seeds])),
                          "sd": float(np.std([per_seed[s][t] for s in a.seeds], ddof=1)) if len(a.seeds) > 1 else 0.0,
                          "per_seed": {str(s): per_seed[s][t] for s in a.seeds}} for t in D.TARGETS}
        R.write_json(run_dir / "summary.json", {"scores": scores, "seed_stats": seed_stats, "cfg": full})
        R.log_scores(scores, model=f"lite_student_{name}", experiment=experiment, split="val", cfg=full,
                     artifacts=[run_dir], smoke=a.smoke, is_baseline=False,
                     notes=f"{len(a.seeds)}-seed ensemble (mean mu); fixed epochs, no val-based early stopping",
                     extra={"seed_stats": seed_stats})
        print({t: round(s["value"], 3) for t, s in scores.items()}, {t: round(v["sd"], 2) for t, v in seed_stats.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
