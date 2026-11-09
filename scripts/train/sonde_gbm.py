#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Train the global LightGBM SENTINEL-Sonde baseline (one model per variable × horizon) and
score it on the common rows of the requested blocks (default: val).

    python -B scripts/train/sonde_gbm.py --config configs/sonde/gbm.yaml            # full (DEV split)
    python -B scripts/train/sonde_gbm.py --config configs/sonde/gbm.yaml --smoke --temp-split
"""

from __future__ import annotations

import numpy as np

from sentinel.core.seed import seed_everything
from sentinel.models.sonde import runner
from sentinel.models.sonde.gbm import GBMConfig, SondeGBM


def main() -> int:
    a = runner.base_parser(__doc__).parse_args()
    cfg = runner.apply_overrides(runner.load_config(a.config, smoke=a.smoke), a.set)
    if a.seed is not None:
        cfg["seed"] = a.seed
    seed_everything(int(cfg["seed"]))
    run_dir = runner.run_directory("gbm", cfg, a.run_name)
    log = runner.make_logger(run_dir / "log.txt")
    runner.log_config_tried("gbm", cfg, run_dir)
    with runner.training_run("gbm", cfg, run_dir) as run:
        data = runner.prepare(a, cfg, run_dir, training=True, log=log)
        variables = runner.model_variables(cfg, data)
        model = SondeGBM(GBMConfig(**cfg["gbm"], seed=int(cfg["seed"])), data.horizons)
        train_sites = data.site_idx(("train",))
        min_sites = int(cfg.get("min_train_sites_per_var", 3))
        for v in variables:
            j = data.cube.var_index(v)
            n_sites = int(np.isfinite(data.clim_mean[train_sites, 0, j]).sum())
            if n_sites < min_sites:
                log(f"[gbm] {v}: only {n_sites} train-role sites (< {min_sites}); not fitted")
                continue
            model.fit_var(data, v, train_sites=train_sites, es_rows=data.rows(v, "val"), log=log)
        model.save(run_dir / "model")
        runner.record_checkpoint(run_dir, "gbm", run_dir / "model", log=log)
        fitted = [v for v in variables if all((v, h) in model.models for h in data.horizons)]
        skipped = sorted(set(variables) - set(fitted))
        if skipped:
            log(f"[gbm] variables without a model at every horizon (not scored): {skipped}")
            run.note(f"not scored (too few training rows): {skipped}")
        for block in runner.eval_blocks(a):
            runner.write_json(run_dir / f"rows_{block}.json", runner.describe_rows(data, block, fitted))
            fc = {v: model.predict_rows(data, v, data.rows(v, block)) for v in fitted}
            runner.save_forecasts(data, "gbm", fc, block, run_dir, meta={"smoke": cfg["_smoke"]})
            summ = runner.score_and_summarize(data, "gbm", fc, block, run_dir, log=log)
            runner.print_summary(summ, f"[gbm] {'SMOKE ' if cfg['_smoke'] else ''}skill on {block} "
                                       "(median over sites; descriptive, not registered)", log)
        n_fit = int(np.sum([d.get("n_fit", 0) for dv in model.diag.values() for d in dv.values()]))
        run.note(f"variables fitted: {fitted}; total fit rows {n_fit}")
        if not cfg["_smoke"]:
            run.metric("n_fit_rows_total", n_fit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
