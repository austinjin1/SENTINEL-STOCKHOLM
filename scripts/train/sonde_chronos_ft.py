#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Fine-tune Chronos-2 on train-role sites × TRAIN block with the official Chronos2Pipeline.fit()
(checkpoint selection on validation-block windows), then forecast and score the common rows of the
requested blocks exactly like the zero-shot run. One GPU from the M1 budget (prefer 0, else 4).

    python -B scripts/train/sonde_chronos_ft.py --config configs/sonde/chronos_ft.yaml
    python -B scripts/train/sonde_chronos_ft.py --config configs/sonde/chronos_ft.yaml --smoke \
        --split docs/preregistration/sonde_split_dev.json
"""

from __future__ import annotations

import time

from sentinel.core.seed import seed_everything
from sentinel.models.sonde import runner


def main() -> int:
    ap = runner.base_parser(__doc__)
    ap.add_argument("--gpu-prefer", type=int, default=0)
    a = ap.parse_args()
    cfg = runner.apply_overrides(runner.load_config(a.config, smoke=a.smoke), a.set)
    if a.seed is not None:
        cfg["seed"] = a.seed
    seed = int(cfg["seed"])
    from sentinel.models.sonde import gpu_budget

    gpu_budget.widen()   # lead ruling 2026-11-10: GPUs 0-9 via the cap file, idle GPUs only
    gpu = runner.setup_gpu("sonde_chronos_ft" + ("_smoke" if a.smoke else ""), prefer=a.gpu_prefer)
    from sentinel.models.sonde.chronos import ChronosForecaster, load_pipeline
    from sentinel.models.sonde.chronos_ft import ChronosFTConfig, finetune

    seed_everything(seed)
    run_dir = runner.run_directory(f"chronos2_ft_s{seed}", cfg, a.run_name)
    log = runner.make_logger(run_dir / "log.txt")
    log(f"[chronos-ft] GPU {gpu}; seed {seed}")
    runner.log_config_tried("chronos2_ft", cfg, run_dir)
    with runner.training_run("chronos2_ft", cfg, run_dir) as run:
        data = runner.prepare(a, cfg, run_dir, training=True, log=log)
        variables = runner.model_variables(cfg, data)
        fcfg = ChronosFTConfig(**cfg["chronos_ft"], seed=seed)
        base = load_pipeline(cfg.get("base_model", "amazon/chronos-2"), device="cuda")
        t0 = time.time()
        ft, info = finetune(base, data, fcfg, run_dir / "trainer", log=log)
        info["minutes"] = (time.time() - t0) / 60
        runner.write_json(run_dir / "finetune_info.json", info)
        runner.record_checkpoint(run_dir, f"chronos2_ft_{fcfg.mode}", run_dir / "trainer" / "finetuned-ckpt", log=log)
        log(f"[chronos-ft] fine-tuning done in {info['minutes']:.1f} min; checkpoint {run_dir / 'trainer' / 'finetuned-ckpt'}")
        name = f"chronos2_ft_{fcfg.mode}"
        fcr = ChronosForecaster(ft, fcfg.inference_config(int(cfg.get("inference_batch_size", 1024))), data.horizons)
        for block in runner.eval_blocks(a):
            rows = {v: data.rows(v, block) for v in variables if len(data.rows(v, block))}
            t1 = time.time()
            fc = fcr.predict(data, rows)
            log(f"[chronos-ft] {name} {block}: {sum(len(r) for r in rows.values())} rows in {(time.time() - t1) / 60:.1f} min")
            runner.save_forecasts(data, name, fc, block, run_dir, meta={"seed": seed, "smoke": cfg["_smoke"]})
            summ = runner.score_and_summarize(data, name, fc, block, run_dir, log=log)
            runner.print_summary(summ, f"[chronos-ft] {'SMOKE ' if cfg['_smoke'] else ''}{name} skill on {block} "
                                       "(median over sites; descriptive, not registered)", log)
        run.note(f"fine-tune {info['minutes']:.1f} min; train inputs {info['train']}; val inputs {info['val']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
