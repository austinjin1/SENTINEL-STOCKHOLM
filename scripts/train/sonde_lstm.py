#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Train the multi-site quantile LSTM for SENTINEL-Sonde (one run per seed) and score it on the
common rows of the requested blocks (default: val). GPU from the M1 budget (prefer 0, else 4).

    python -B scripts/train/sonde_lstm.py --config configs/sonde/lstm.yaml
    python -B scripts/train/sonde_lstm.py --config configs/sonde/lstm.yaml --smoke --temp-split
"""

from __future__ import annotations

from sentinel.models.sonde import runner


def main() -> int:
    ap = runner.base_parser(__doc__)
    ap.add_argument("--gpu-prefer", type=int, default=0)
    a = ap.parse_args()
    cfg = runner.apply_overrides(runner.load_config(a.config, smoke=a.smoke), a.set)
    seeds = [a.seed] if a.seed is not None else list(cfg.get("seeds", [0]))
    gpu = runner.setup_gpu("sonde_lstm" + ("_smoke" if a.smoke else ""), prefer=a.gpu_prefer)
    from sentinel.models.sonde.lstm import LSTMConfig, SondeLSTM  # after CUDA_VISIBLE_DEVICES is set

    for seed in seeds:
        scfg = dict(cfg, seed=seed)
        run_dir = runner.run_directory(f"lstm_s{seed}", scfg, f"{a.run_name}_s{seed}" if a.run_name else None)
        log = runner.make_logger(run_dir / "log.txt")
        log(f"[lstm] GPU {gpu} (CUDA_VISIBLE_DEVICES), seed {seed}")
        runner.log_config_tried("lstm", scfg, run_dir)
        with runner.training_run("lstm", scfg, run_dir) as run:
            data = runner.prepare(a, scfg, run_dir, training=True, log=log)
            variables = runner.model_variables(scfg, data)
            model_vars = variables + [v for v in scfg.get("lstm_aux_outputs", []) if v not in variables
                                      and v in data.cube.variables]
            model = SondeLSTM(LSTMConfig(**scfg["lstm"], seed=seed), data.horizons, model_vars)
            info = model.fit(data, log=log, checkpoint_path=run_dir / "lstm_best.pt")
            model.save(run_dir / "lstm_best.pt")
            runner.record_checkpoint(run_dir, "lstm", run_dir / "lstm_best.pt", log=log)
            runner.write_json(run_dir / "train_info.json", {**info, "history": model.history,
                                                            "variables": model.variables})
            # score only variables the model was trained on (enough train-role sites with the variable)
            trained = [v for v in model.trained_variables(min_sites=int(scfg.get("min_train_sites_per_var", 3)))
                       if v in variables]
            skipped = sorted(set(model.variables) - set(trained))
            if skipped:
                log(f"[lstm] not scored (fewer than min_train_sites_per_var train-role sites): {skipped}")
                run.note(f"not scored: {skipped}")
            for block in runner.eval_blocks(a):
                rows = {v: data.rows(v, block) for v in trained}
                runner.write_json(run_dir / f"rows_{block}.json", runner.describe_rows(data, block, trained))
                fc = model.predict_rows(rows)
                runner.save_forecasts(data, "lstm", fc, block, run_dir, meta={"seed": seed, "smoke": scfg["_smoke"]})
                summ = runner.score_and_summarize(data, "lstm", fc, block, run_dir, log=log)
                runner.print_summary(summ, f"[lstm] {'SMOKE ' if scfg['_smoke'] else ''}skill on {block} seed {seed} "
                                           "(median over sites; descriptive, not registered)", log)
            run.note(f"epochs {info['epochs']}, params {info['n_params']}, train samples {info['n_train_samples']}")
            if not scfg["_smoke"]:
                run.metric("best_val_pinball", info["best_val_loss"])
                run.metric("n_params", info["n_params"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
