#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Chronos-2 zero-shot forecasts on the common SENTINEL-Sonde rows ('uni' and/or 'multi' mode),
fp32, batched on one GPU from the M1 budget. No training (the run is still logged).

Contamination note: Chronos-2's disclosed pretraining corpora contain no USGS series
(docs/research/01 §a.4), but its release (2025-10) post-dates the val block; the clean
contamination test is Test-B (WY2026), scored only after the freeze.

    python -B scripts/eval/sonde_chronos_zeroshot.py --config configs/sonde/chronos_zeroshot.yaml
    python -B scripts/eval/sonde_chronos_zeroshot.py --config configs/sonde/chronos_zeroshot.yaml --smoke --temp-split
"""

from __future__ import annotations

import time

from sentinel.models.sonde import runner


def main() -> int:
    ap = runner.base_parser(__doc__)
    ap.add_argument("--gpu-prefer", type=int, default=0)
    ap.add_argument("--modes", default=None, help="comma list overriding config 'modes'")
    a = ap.parse_args()
    cfg = runner.apply_overrides(runner.load_config(a.config, smoke=a.smoke), a.set)
    modes = a.modes.split(",") if a.modes else list(cfg.get("modes", ["uni"]))
    gpu = runner.setup_gpu("sonde_chronos_zs" + ("_smoke" if a.smoke else ""), prefer=a.gpu_prefer)
    from sentinel.models.sonde.chronos import ChronosConfig, ChronosForecaster, load_pipeline

    run_dir = runner.run_directory("chronos2_zeroshot", cfg, a.run_name)
    log = runner.make_logger(run_dir / "log.txt")
    log(f"[chronos-zs] GPU {gpu}; modes {modes}")
    runner.log_config_tried("chronos2_zeroshot", cfg, run_dir)
    with runner.training_run("chronos2_zeroshot", cfg, run_dir) as run:
        run.note("zero-shot inference only (no training)")
        data = runner.prepare(a, cfg, run_dir, log=log)
        variables = runner.model_variables(cfg, data)
        base = cfg["chronos"]
        pipe = load_pipeline(base["model_id"], device="cuda", local_files_only=base.get("local_files_only", True))
        for mode in modes:
            ccfg = ChronosConfig(**{**base, "mode": mode})
            fcr = ChronosForecaster(pipe, ccfg, data.horizons)
            name = f"chronos2_zs_{mode}"
            for block in runner.eval_blocks(a):
                rows = {v: data.rows(v, block) for v in variables if len(data.rows(v, block))}
                t0 = time.time()
                fc = fcr.predict(data, rows)
                dt = time.time() - t0
                n = sum(len(r) for r in rows.values())
                log(f"[chronos-zs] {name} {block}: {n} rows in {dt / 60:.1f} min")
                runner.save_forecasts(data, name, fc, block, run_dir, meta={"mode": mode, "smoke": cfg["_smoke"]})
                summ = runner.score_and_summarize(data, name, fc, block, run_dir, log=log)
                runner.print_summary(summ, f"[chronos-zs] {'SMOKE ' if cfg['_smoke'] else ''}{name} skill on {block} "
                                           "(median over sites; descriptive, not registered)", log)
                run.note(f"{name}/{block}: {n} rows, {dt / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
