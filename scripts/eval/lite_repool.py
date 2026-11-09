# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Inference-time pooling variant of a trained SENTINEL-Lite run (handoff idea "median pooling"; val only).

Re-scores the run's seed checkpoints on the val role with ``pool: median`` (weighted median of the per-pixel μ over
the centre water window instead of the weighted mean; same weights, no retraining) and writes a new run directory
``student_<name>_medpool_<stamp>`` (summary.json, val predictions, ``source_run.json``; checkpoints are symlinked
with the pooling choice recorded in ``pool.json``). Registered in ``lite_val`` as a separate configuration.

    python -B scripts/eval/lite_repool.py --run $SENTINEL_RUNS/lite/<run> [--device cpu]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentinel.models.lite import runner as R

R.cpu_threads(2)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sentinel.models.lite import data as D  # noqa: E402
from sentinel.models.lite import evaluate as E  # noqa: E402
from sentinel.models.lite import features as F  # noqa: E402
from sentinel.models.lite import guard as G  # noqa: E402
from sentinel.models.lite.train import Inputs, build_model, predict  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--pool", default="median", choices=["median"])
    ap.add_argument("--device", default="cpu", choices=["cpu"])
    a = ap.parse_args(argv)
    src = Path(a.run)
    cks = sorted(src.glob("model_seed*.pt"))
    split = G.load_split()
    va = D.role_frame(split, ("val",), training=True)
    G.assert_no_protected(va["role"])
    chips = np.asarray(D.cached_chips(va, "val"))
    inp = Inputs(chips, D.scene_norm_params(va, D.scene_stats()), F.pooling_weights(chips, va["gsd_m"].to_numpy(), 4),
                 None, a.device)
    y = va[[f"y_{t}" for t in D.TARGETS]].to_numpy(float)
    mus, sds, per_seed, cfg = [], [], {}, None
    for ck in cks:
        c = torch.load(ck, map_location="cpu", weights_only=False)
        if c["cfg"].get("smoke"):
            raise SystemExit("smoke checkpoint")
        cfg = {**c["cfg"], "pretrained": False, "gloria_init": None, "amp": False, "pool": a.pool}
        m = build_model(cfg)
        m.load_state_dict(c["state_dict"])
        mu, sd = predict(m.eval(), inp, cfg)
        mus.append(mu)
        sds.append(sd)
        per_seed[c["seed"]] = {t: E.score_primary(t, y[:, j], mu[:, j], va["water_body"], va["site_class"],
                                                  n_boot=200)["value"] for j, t in enumerate(D.TARGETS)}
    name = f"{cfg.get('name', src.name)}_medpool"
    run_dir = R.run_directory(f"student_{name}")
    for ck in cks:
        (run_dir / ck.name).symlink_to(ck)
    R.write_json(run_dir / "pool.json", {"pool": a.pool, "note": "apply cfg['pool'] at inference"})
    R.write_json(run_dir / "source_run.json", {"source_run": str(src), "pool": a.pool})
    mu = np.mean(mus, 0)
    sd = np.sqrt(np.mean(np.square(sds), 0) + np.var(mus, 0))
    R.save_predictions(run_dir / "val_pred_ensemble.parquet", va, {t: mu[:, j] for j, t in enumerate(D.TARGETS)},
                       {t: sd[:, j] for j, t in enumerate(D.TARGETS)})
    scores = {t: E.score_primary(t, y[:, j], mu[:, j], va["water_body"], va["site_class"], sigma_log=sd[:, j])
              for j, t in enumerate(D.TARGETS)}
    seeds = sorted(per_seed)
    seed_stats = {t: {"mean": float(np.mean([per_seed[s][t] for s in seeds])),
                      "sd": float(np.std([per_seed[s][t] for s in seeds], ddof=1)) if len(seeds) > 1 else 0.0,
                      "per_seed": {str(s): per_seed[s][t] for s in seeds}} for t in D.TARGETS}
    full = {**cfg, "name": name, "source_run": str(src)}
    R.write_json(run_dir / "summary.json", {"scores": scores, "seed_stats": seed_stats, "cfg": full})
    with R.training_run(f"student/{name}", full, run_dir) as run:
        run.note(f"inference-time pooling variant of {src.name} (no retraining)")
        R.log_scores(scores, model=f"lite_student_{name}", experiment="lite_val", split="val", cfg=full,
                     artifacts=[run_dir], smoke=False, is_baseline=False,
                     notes=f"{len(seeds)}-seed ensemble; weighted-median pooling of {src.name} weights",
                     extra={"seed_stats": seed_stats})
    print(json.dumps({t: round(s["value"], 2) for t, s in scores.items()}), run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
