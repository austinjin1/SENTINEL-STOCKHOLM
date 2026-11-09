#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Train the SENTINEL-HydroMet EA-LSTM (no sensor inputs) on train-role sites × train block.

1. **Selection** (protocol §8): every point of the config ``grid`` is trained with the first seed and
   early-stopped on the inner train-role basin groups; the point with the lowest inner loss is selected
   (val-role sites are never used; non-selected points are never applied to val). ``--hparams-from``
   reuses an earlier selection instead (physics-DO variant, ablations).
2. **Seeds**: the selected point is trained for every seed; checkpoints / predictions are keyed
   ``<model>_s<seed>``; predictions of the selected runs go to ``<root>/predictions``.
3. Descriptive in-run scoring per seed (with --baseline-run, the baselines on the same cells).
   Results are registered only by scripts/eval/hydromet_score.py --register.

One GPU via sentinel.core.gpu (prefer GPU 3).

    python -B scripts/train/hydromet_lstm.py --config configs/hydromet/lstm.yaml --run-name ealstm_v1 --baseline-run <dir>
    python -B scripts/train/hydromet_lstm.py --config configs/hydromet/lstm_physics_do.yaml --run-name physdo_v1 \\
        --hparams-from $SENTINEL_RUNS/hydromet/ealstm_v1/selection.json --baseline-run <dir>
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from sentinel.models.hydromet import runner


def grid_points(cfg: dict) -> list[dict]:
    """Config ``grid``: an explicit list of override dicts, or a dict of lists (cartesian product)."""
    g = cfg.get("grid") or {}
    if isinstance(g, list):
        return [dict(x) for x in g] or [{}]
    keys = sorted(g)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(g[k] for k in keys))] or [{}]


def main() -> int:
    ap = runner.base_parser(__doc__)
    ap.add_argument("--baseline-run", default=None, help="hydromet_baselines run dir to compare against (descriptive)")
    ap.add_argument("--hparams-from", default=None, help="selection.json of an earlier EA-LSTM run (skip selection)")
    ap.add_argument("--gpu-prefer", type=int, default=runner.GPU_PREFER)
    ap.add_argument("--refit-full-from", default=None,
                    help="protocol v0.8: early-stopped run root (may contain '{seed}'); refit on fit + inner train-role "
                         "sites for that seed's inner-selected epoch count (no early stopping; mirrors the GBM refit)")
    a = ap.parse_args()
    runner.env_threads(8)
    cfg = runner.load_config(a.config, smoke=a.smoke)
    if a.track_r:   # protocol §10 (leave-region-out track)
        cfg["track_r"] = a.track_r
        cfg["secondary_eval_roles"] = []
    lag = runner.check_mode(cfg)
    seeds = [a.seed] if a.seed is not None else list(cfg.get("seeds", [0]))
    model_name = cfg.get("model", "ealstm")
    if a.hparams_from:
        sel = json.loads(Path(a.hparams_from).read_text())
        points, inherited = [sel["selected"]], a.hparams_from
    else:
        points, inherited = grid_points(cfg), None
    gpu = runner.setup_gpu(f"hydromet_{model_name}" + ("_smoke" if a.smoke else ""), prefer=a.gpu_prefer)
    from sentinel.models.hydromet.evaluate import align_predictions, load_predictions, save_predictions
    from sentinel.models.hydromet.guard import artifact_sha256
    from sentinel.models.hydromet.lstm import EALSTMConfig, HydroMetEALSTM

    root = runner.run_directory(model_name, cfg, a.run_name)
    log = runner.make_logger(root / "log.txt")
    log(f"[{model_name}] GPU {gpu}; seeds {seeds}; grid {points}; forcing lag {lag} d; hparams from {inherited}")
    runner.write_json(root / "config.json", runner.public_config(cfg))
    data, fit_idx, inner_idx = runner.prepare(a, cfg, root, training=True, log=log,
                                              roles=("train",) if cfg["_smoke"] else ("train", "val"))

    def train(k: int, over: dict, seed: int):
        run_dir = root / f"g{k}_s{seed}"
        scfg = dict(cfg, seed=seed, ealstm={**cfg["ealstm"], **over}, grid_point=over)
        runner.write_json(run_dir / "config.json", runner.public_config(scfg))
        runner.log_config_tried(model_name, scfg, run_dir)
        ecfg = dict(scfg["ealstm"], forcing_lag_days=lag, seed=seed)
        ecfg["ratio_bounds"] = tuple(ecfg.get("ratio_bounds", (0.2, 1.6)))
        refit = None
        if a.refit_full_from:
            es_dir = Path(a.refit_full_from.format(seed=seed))
            es_sel = json.loads((es_dir / "selection.json").read_text())
            es = json.loads((es_dir / f"g{es_sel['selected_index']}_s{seed}" / "train_info.json").read_text())
            if es["grid_point"] != over:
                raise ValueError(f"{es_dir}: grid point {es['grid_point']} != {over}")
            es_cfg = json.loads((es_dir / "config.json").read_text())
            if es_cfg.get("track_r") != cfg.get("track_r") or es_cfg.get("upstream_forcing") != cfg.get("upstream_forcing"):
                raise ValueError(f"{es_dir}: early-stopping run is from another track-R fold / input set")
            hist = [h["inner_val_loss"] for h in es["history"]]
            n_ep = int(np.nanargmin(hist)) + 1
            refit = {"from": str(es_dir), "inner_selected_epochs": n_ep, "inner_val_loss": float(np.nanmin(hist))}
            ecfg.update(max_epochs=n_ep, patience=10 ** 6, max_minutes=None)
            scfg["refit_full"] = refit
            runner.write_json(run_dir / "config.json", runner.public_config(scfg))
        with runner.training_run(model_name, scfg, run_dir) as run:
            model = HydroMetEALSTM(EALSTMConfig(**ecfg))
            model.info["model_name"] = model_name
            ckpt = run_dir / f"{model_name}_best.pt"
            if refit:   # all train-role sites, fixed epoch count, no inner validation (last epoch kept)
                model.info["refit_full"] = refit
                log(f"[{model_name}] refit on fit + inner sites for {refit['inner_selected_epochs']} epochs ({refit['from']})")
                info = model.fit(data, np.r_[fit_idx, inner_idx], np.zeros(0, int), device="cuda", log=log,
                                 checkpoint_path=ckpt)
            else:
                info = model.fit(data, fit_idx, inner_idx, device="cuda", log=log, checkpoint_path=ckpt)
            model.save(ckpt)
            runner.write_json(run_dir / "train_info.json", {**info, "history": model.history, "grid_point": over})
            run.note(f"grid {over}, epochs {info['epochs']}, params {info['n_params']}")
            if not scfg["_smoke"] and not refit:
                run.metric("best_inner_val_loss", info["best_inner_val_loss"])
        return model, ckpt, info

    # 1. selection on inner groups (first seed)
    s0 = seeds[0]
    trained = {}
    for k, over in enumerate(points):
        trained[(k, s0)] = train(k, over, s0)
    losses = [trained[(k, s0)][2]["best_inner_val_loss"] for k in range(len(points))]
    k_best = 0 if len(points) == 1 else int(np.nanargmin(losses))
    runner.write_json(root / "selection.json", {"grid": points, "inner_val_loss": losses, "selected_index": k_best,
                                                "selected": points[k_best], "selection_seed": s0,
                                                "n_configs_tried": len(points), "inherited_from": inherited,
                                                "criterion": "lowest inner (train-role basin groups) loss"})
    log(f"[{model_name}] selected grid point {k_best} {points[k_best]} (inner losses {np.round(losses, 4).tolist()})")
    # 2. remaining seeds for the selected point
    for seed in seeds[1:]:
        trained[(k_best, seed)] = train(k_best, points[k_best], seed)
    # 3. predictions (selected runs only) + descriptive scoring
    shas = {}
    roles_all = list(dict.fromkeys(list(cfg.get("eval_roles", ["val"])) + list(cfg.get("secondary_eval_roles", []))))
    for seed in seeds:
        model, ckpt, _ = trained[(k_best, seed)]
        name = f"{model_name}_s{seed}"
        shas[name] = artifact_sha256(ckpt)
        for block in runner.eval_blocks(a):
            si, ti = runner.eval_cells(data, block, roles_all)
            point, dist = model.predict(data, si, ti, device="cuda", return_dist=True)
            save_predictions(root / "predictions" / f"{name}__{block}.npz", model=name, preds=point,
                             sites=[data.sites[i] for i in si], dates=data.dates[ti], block=block, dist=dist,
                             meta={"smoke": cfg["_smoke"], "seed": seed, "base_model": model_name,
                                   "forcing_lag_days": lag, "split_sha256": data.split.sha256(), "track_r_fold": cfg.get("track_r"),
                                   "artifact": str(ckpt), "artifact_sha256": shas[name], "freeze_sha256": None,
                                   "grid_point": points[k_best]})
            for roles in [cfg.get("eval_roles", ["val"])]:
                ridx = data.site_idx(roles)
                pos = {s: k for k, s in enumerate(si)}
                preds = {name: {v: P[[pos[i] for i in ridx]] for v, P in point.items()}}
                if a.baseline_run:
                    for f in sorted((Path(a.baseline_run) / "predictions").glob(f"*__{block}.npz")):
                        pf = load_predictions(f)
                        if pf["header"].get("split_sha256") != data.split.sha256() or \
                                bool(pf["header"].get("smoke")) != bool(cfg["_smoke"]):
                            raise ValueError(f"{f}: split or smoke flag mismatch")
                        preds[pf["header"]["model"]] = align_predictions(pf, data, ridx, ti)
                _, summ = runner.score_and_write(data, preds, block, root / f"g{k_best}_s{seed}", roles=tuple(roles),
                                                 refs=tuple(cfg.get("refs", ())) if a.baseline_run else (),
                                                 n_boot=int(cfg.get("n_boot", 2000)),
                                                 min_days=int(cfg.get("min_days_per_site", 60)), log=log)
                runner.print_summary(summ, f"[{model_name}] {'SMOKE ' if cfg['_smoke'] else ''}{block} block, "
                                           f"{'/'.join(roles)} sites, seed {seed}: median over sites "
                                           "[basin-group bootstrap 95% CI] (descriptive)", log)
    runner.write_json(root / "checkpoints.json", shas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
