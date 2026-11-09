#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Fit the SENTINEL-HydroMet baseline ladder (no sensor inputs) on train-role sites × train block,
predict the evaluation cells and score them (val-role sites × val block; train-role sites × val block
as a descriptive temporal-only cell). CPU, plus one GPU (prefer 3) when
``air2stream.device: cuda``.

    python -B scripts/train/hydromet_baselines.py --config configs/hydromet/baselines.yaml --smoke
    python -B scripts/train/hydromet_baselines.py --config configs/hydromet/baselines.yaml

Results are registered only by scripts/eval/hydromet_score.py --register (one consolidated experiment).
"""

from __future__ import annotations

import numpy as np

from sentinel.models.hydromet import runner
from sentinel.models.hydromet.data import TARGETS, assert_allowed_inputs
from sentinel.models.hydromet.evaluate import save_predictions


def main() -> int:
    ap = runner.base_parser(__doc__)
    a = ap.parse_args()
    runner.env_threads(8)
    cfg = runner.load_config(a.config, smoke=a.smoke)
    if a.track_r:   # protocol §10 (leave-region-out track)
        cfg["track_r"] = a.track_r
        cfg["secondary_eval_roles"] = []
    if a.seed is not None:
        cfg["seed"] = a.seed
    lag = runner.check_mode(cfg)
    run_dir = runner.run_directory("baselines", cfg, a.run_name)
    log = runner.make_logger(run_dir / "log.txt")
    if (cfg.get("air2stream") or {}).get("device", "cpu") == "cuda":   # air2stream calibration only
        gpu = runner.setup_gpu("hydromet_baselines" + ("_smoke" if a.smoke else ""))
        log(f"[baselines] GPU {gpu} (CUDA_VISIBLE_DEVICES) for air2stream calibration")
    runner.log_config_tried("baselines", cfg, run_dir)
    runner.write_json(run_dir / "config.json", runner.public_config(cfg))
    from sentinel.models.hydromet.baselines import fit_suite
    from sentinel.models.hydromet.features import FeatureSet
    from sentinel.models.hydromet.guard import artifact_sha256

    with runner.training_run("baselines", cfg, run_dir) as run:
        data, fit_idx, inner_idx = runner.prepare(a, cfg, run_dir, training=True, log=log,
                                                  roles=("train",) if cfg["_smoke"] else ("train", "val"))
        fs = FeatureSet.build(data, lag=lag)
        groups = assert_allowed_inputs(fs.columns(), static_cols=list(fs.static.columns), forcing_feature_cols=fs.dyn_names)
        runner.write_json(run_dir / "inputs.json", {"forcing_lag_days": lag, "mode": cfg.get("mode"),
                                                   **{k: v for k, v in groups.items()}})
        log(f"[inputs] forcing {len(groups['forcing'])}, static {len(groups['static'])}, calendar {len(groups['calendar'])}")
        suite = fit_suite(cfg, data, fs, fit_idx, inner_idx, log=log)
        shas = {}
        for name, m in suite.models.items():
            p = m.save(run_dir / "models" / f"{name}.pkl")
            shas[name] = artifact_sha256(p)
        runner.write_json(run_dir / "checkpoints.json", shas)
        runner.write_json(run_dir / "fit_info.json", {n: m.info for n, m in suite.models.items()})
        roles_all = list(dict.fromkeys(list(cfg.get("eval_roles", ["val"])) + list(cfg.get("secondary_eval_roles", []))))
        for block in runner.eval_blocks(a):
            si, ti = runner.eval_cells(data, block, roles_all)
            preds = {}
            for name, m in suite.models.items():
                preds[name] = m.predict(data, fs, si, ti)
                save_predictions(run_dir / "predictions" / f"{name}__{block}.npz", model=name, preds=preds[name],
                                 sites=[data.sites[i] for i in si], dates=data.dates[ti], block=block,
                                 meta={"smoke": cfg["_smoke"], "forcing_lag_days": lag, "split_sha256": data.split.sha256(), "track_r_fold": cfg.get("track_r")})
            full = {n: {v: _embed(p[v], si, ti, data) for v in p} for n, p in preds.items()}
            for roles in ([cfg.get("eval_roles", ["val"])] + ([cfg["secondary_eval_roles"]]
                                                               if cfg.get("secondary_eval_roles") else [])):
                _, summ = runner.score_and_write(data, _restrict(full, data, roles, block), block, run_dir,
                                                 roles=tuple(roles), refs=tuple(cfg.get("refs", ())),
                                                 n_boot=int(cfg.get("n_boot", 2000)),
                                                 min_days=int(cfg.get("min_days_per_site", 60)), log=log)
                runner.print_summary(summ, f"[baselines] {'SMOKE ' if cfg['_smoke'] else ''}{block} block, "
                                           f"{'/'.join(roles)}-role sites: median over sites [site-bootstrap 95% CI]"
                                           " (descriptive)", log)
        run.note(f"baselines {list(suite.models)}; sites {data.info['roles']}")
    return 0


def _embed(P, si, ti, data):
    """Predictions on (si × ti) -> dense (S, len(ti)) aligned with data.sites (NaN elsewhere)."""
    out = np.full((len(data.sites), len(ti)), np.nan, dtype="float32")
    out[si] = P
    return out


def _restrict(full, data, roles, block):
    """Score helper input: predictions as (n_role_sites, n_block_days) for the requested roles."""
    idx = data.site_idx(roles)
    return {n: {v: P[idx] for v, P in p.items()} for n, p in full.items()}


if __name__ == "__main__":
    raise SystemExit(main())
