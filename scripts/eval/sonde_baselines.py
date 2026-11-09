#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Naive SENTINEL-Sonde baselines (climatology, persistence, damped anomaly) on the common rows.

Writes forecast files (for event heads / ensembles) and per-sample scores for the requested
blocks (default: val). Test blocks need --allow-test AND docs/preregistration/FREEZE_sonde.json.

    python -B scripts/eval/sonde_baselines.py --config configs/sonde/common.yaml [--smoke --temp-split]
"""

from __future__ import annotations

from sentinel.models.sonde import common, runner

MODELS = ("climatology", "persistence", "damped_anomaly")


def main() -> int:
    a = runner.base_parser(__doc__).parse_args()
    cfg = runner.load_config(a.config, smoke=a.smoke)
    run_dir = runner.run_directory("baselines", cfg, a.run_name)
    log = runner.make_logger(run_dir / "log.txt")
    with runner.training_run("naive_baselines", cfg, run_dir) as run:
        data = runner.prepare(a, cfg, run_dir, log=log)
        variables = runner.model_variables(cfg, data)
        for block in runner.eval_blocks(a):
            runner.write_json(run_dir / f"rows_{block}.json", runner.describe_rows(data, block, variables))
            for m in MODELS:
                fc = {v: common.baseline_forecast(data, v, data.rows(v, block), m) for v in variables}
                runner.save_forecasts(data, m, fc, block, run_dir, meta={"smoke": cfg["_smoke"]})
            others = [m for m in MODELS if m != "damped_anomaly"]
            fc = {v: common.baseline_forecast(data, v, data.rows(v, block), "damped_anomaly") for v in variables}
            summ = runner.score_and_summarize(data, "damped_anomaly", fc, block, run_dir, refs=others, log=log)
            runner.print_summary(summ, f"[baselines] damped_anomaly on {block} "
                                 f"({'SMOKE, ' if cfg['_smoke'] else ''}descriptive, not registered)", log)
        run.note(f"forecasts for {MODELS} on {runner.eval_blocks(a)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
