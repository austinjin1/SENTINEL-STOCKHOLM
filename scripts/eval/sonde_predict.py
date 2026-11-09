#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Predict-only path for SENTINEL-Sonde: load a saved checkpoint, verify its SHA-256, write forecast
files on the common rows of the requested blocks. Never fits anything (naive baselines are
deterministic functions of the training block and have no checkpoint).

For protected blocks (test, test_b) or test-role sites this needs --allow-test AND a valid
docs/preregistration/FREEZE_sonde.json whose ``weights_sha256[<name>]`` equals the checkpoint's
SHA-256 (sentinel.models.sonde.guard.artifact_sha256). Score the output with scripts/eval/sonde_score.py.

    python -B scripts/eval/sonde_predict.py --config configs/sonde/common.yaml --model gbm --checkpoint <run>/model
    python -B scripts/eval/sonde_predict.py --config configs/sonde/lstm.yaml --model lstm --checkpoint <run>/lstm_best.pt
    python -B scripts/eval/sonde_predict.py --config configs/sonde/chronos_zeroshot.yaml --model chronos2_zs_uni
    python -B scripts/eval/sonde_predict.py --config configs/sonde/chronos_ft.yaml --model chronos2_ft_multi \
        --checkpoint <run>/trainer/finetuned-ckpt
    ... --eval-blocks test --allow-test --include-test-sites     # only after the freeze
"""

from __future__ import annotations

from pathlib import Path

from sentinel.models.sonde import common, guard, runner

MODELS = ("climatology", "persistence", "damped_anomaly", "gbm", "lstm", "chronos2_zs_uni", "chronos2_zs_multi",
          "chronos2_ft_uni", "chronos2_ft_multi", "timesfm25_zs")


def chronos_snapshot(model_id: str) -> Path:
    from sentinel.models.sonde.chronos import HF_CACHE

    snaps = sorted((HF_CACHE / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots").glob("*"))
    if len(snaps) != 1:
        raise FileNotFoundError(f"expected exactly one local snapshot of {model_id}, found {len(snaps)}")
    return snaps[0]


def main() -> int:
    ap = runner.base_parser(__doc__)
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--checkpoint", default=None, help="GBM model dir / LSTM .pt / Chronos checkpoint dir")
    ap.add_argument("--gpu-prefer", type=int, default=0)
    a = ap.parse_args()
    cfg = runner.apply_overrides(runner.load_config(a.config, smoke=a.smoke), a.set)
    m = a.model
    ck = a.checkpoint
    if m.startswith("chronos2_zs"):
        ck = ck or str(chronos_snapshot(cfg.get("chronos", {}).get("model_id", "amazon/chronos-2")))
    if m == "timesfm25_zs":
        ck = ck or str(chronos_snapshot("google/timesfm-2.5-200m-pytorch"))
    if m in ("gbm", "lstm", "timesfm25_zs") or m.startswith("chronos2"):
        if not ck or not Path(ck).exists():
            ap.error(f"--checkpoint required (and must exist) for {m}")
    if m in ("lstm", "timesfm25_zs") or m.startswith("chronos2"):
        runner.setup_gpu(f"sonde_predict_{m}", prefer=a.gpu_prefer)
    run_dir = runner.run_directory(f"predict_{m}", cfg, a.run_name)
    log = runner.make_logger(run_dir / "log.txt")
    data = runner.prepare(a, cfg, run_dir, need_blocks=("train",), log=log)
    ck_sha = guard.artifact_sha256(ck) if ck else None
    if data.access["protected"]:  # weights must be exactly the frozen ones
        guard.verify_weights(data.access["freeze"], m, ck)
        log(f"[predict] checkpoint sha256 verified against the freeze record: {ck_sha}")
    variables = runner.model_variables(cfg, data)
    if m == "gbm":
        from sentinel.models.sonde.gbm import SondeGBM

        model = SondeGBM.load(ck)
        variables = [v for v in variables if all((v, h) in model.models for h in data.horizons)]
    elif m == "lstm":
        from sentinel.models.sonde.lstm import SondeLSTM

        model = SondeLSTM()
        model.load_state(ck, data)
        variables = [v for v in model.trained_variables(int(cfg.get("min_train_sites_per_var", 3))) if v in variables]
    elif m == "timesfm25_zs":
        from sentinel.models.sonde.timesfm_tsfm import TimesFMForecaster

        model = TimesFMForecaster(context=int(cfg.get("chronos", {}).get("context", 512)),
                                  batch_size=int(cfg.get("timesfm_batch_size", 256)), horizons=data.horizons)
    elif m.startswith("chronos2"):
        from sentinel.models.sonde.chronos import ChronosConfig, ChronosForecaster, load_pipeline

        mode = m.rsplit("_", 1)[1]
        base = {**cfg.get("chronos", {}), **{k: v for k, v in cfg.get("chronos_ft", {}).items()
                                             if k in ("context", "future_clim", "weather_covariates")}}
        ccfg = ChronosConfig(**{k: v for k, v in base.items() if k in ChronosConfig.__dataclass_fields__
                                and k != "mode"},
                             mode=mode)
        if "inference_batch_size" in cfg:
            ccfg.batch_size = int(cfg["inference_batch_size"])
        model = ChronosForecaster(load_pipeline(ck, device="cuda"), ccfg, data.horizons)
    files = []
    for block in runner.eval_blocks(a):
        rows = {v: data.rows(v, block) for v in variables if len(data.rows(v, block))}
        if m in ("climatology", "persistence", "damped_anomaly"):
            fc = {v: common.baseline_forecast(data, v, r, m) for v, r in rows.items()}
        elif m == "gbm":
            fc = {v: model.predict_rows(data, v, r) for v, r in rows.items()}
        elif m == "lstm":
            fc = model.predict_rows(rows)
        else:
            fc = model.predict(data, rows)
        files += runner.save_forecasts(data, m, fc, block, run_dir,
                                       meta={"checkpoint": ck, "checkpoint_sha256": ck_sha, "smoke": cfg["_smoke"]})
        log(f"[predict] {m} {block}: {sum(len(r) for r in rows.values())} rows")
    runner.write_json(run_dir / "predict_manifest.json",
                      {"model": m, "checkpoint": ck, "checkpoint_sha256": ck_sha, "blocks": runner.eval_blocks(a),
                       "files": {f: guard.artifact_sha256(f) for f in files}, "access": data.access})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
