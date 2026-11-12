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

``--weather-exclude-sources era5_seamless`` sets the lagged observed weather to NaN at every site whose forcing source
(``forcings_daily/coverage.parquet``) is in the list. It reproduces the inputs of runs trained before the 2026-10-27
forcings rebuild, which added ERA5 forcings for sites that had none (M1 Test-A rehearsal finding, 2026-11-10). It does
not edit the frozen loader; it wraps ``common.load_weather`` in this process only and is recorded in the manifest.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.models.sonde import common, gpu_budget, guard, runner

MODELS = ("climatology", "persistence", "damped_anomaly", "gbm", "lstm", "chronos2_zs_uni", "chronos2_zs_multi",
          "chronos2_ft_uni", "chronos2_ft_multi", "timesfm25_zs")


def chronos_snapshot(model_id: str) -> Path:
    from sentinel.models.sonde.chronos import HF_CACHE

    snaps = sorted((HF_CACHE / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots").glob("*"))
    if len(snaps) != 1:
        raise FileNotFoundError(f"expected exactly one local snapshot of {model_id}, found {len(snaps)}")
    return snaps[0]


WEATHER_MASK: dict = {}


def mask_weather_sources(sources: list[str]) -> None:
    """Wrap ``common.load_weather`` (process-local) so sites forced by any of ``sources`` get NaN weather."""
    cov = pd.read_parquet(common.FORCINGS.parent / "coverage.parquet", columns=["site_key", "source"])
    masked = set(cov.loc[cov["source"].isin(sources), "site_key"].astype(str))
    orig = common.load_weather

    def _load(cube, **kw):
        w = orig(cube, **kw)
        idx = [i for i, s in enumerate(cube.sites) if f"USGS-{s}" in masked]
        w[idx] = np.nan
        WEATHER_MASK.update({"exclude_sources": sorted(sources), "n_sites_masked": len(idx),
                             "sites_masked": sorted(cube.sites[i] for i in idx)})
        return w

    common.load_weather = _load


def main() -> int:
    ap = runner.base_parser(__doc__)
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--checkpoint", default=None, help="GBM model dir / LSTM .pt / Chronos checkpoint dir")
    ap.add_argument("--gpu-prefer", type=int, default=0)
    ap.add_argument("--min-free-mb", type=int, default=None,
                    help="GPU physically-free memory needed (default 10000 = idle 2080 Ti; never share another user's GPU)")
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="multivariate Chronos only: stop after the first batch flush past this wall time (from process "
                         "start), save a resume state and exit 75; re-running the same command resumes. Forecasts are "
                         "identical to an uninterrupted run (stops only at flush boundaries). Keeps GPU jobs <= 2 h.")
    ap.add_argument("--weather-exclude-sources", default="",
                    help="comma list of forcing sources whose sites get NaN weather (e.g. era5_seamless)")
    a = ap.parse_args()
    t_start = time.time()
    if a.weather_exclude_sources:
        mask_weather_sources(a.weather_exclude_sources.split(","))
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
        gpu_budget.widen(a.min_free_mb)   # lead ruling 2026-11-10: GPUs 0-9 via the cap file
        runner.setup_gpu(f"sonde_predict_{m}", prefer=a.gpu_prefer)
    run_dir = runner.run_directory(f"predict_{m}", cfg, a.run_name)
    log = runner.make_logger(run_dir / "log.txt")
    data = runner.prepare(a, cfg, run_dir, need_blocks=("train",), log=log)
    if WEATHER_MASK:
        log(f"[predict] weather masked at {WEATHER_MASK['n_sites_masked']} sites (sources "
            f"{WEATHER_MASK['exclude_sources']})")
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
        elif a.max_minutes and m.startswith("chronos2") and model.cfg.mode == "multi":
            if len(runner.eval_blocks(a)) != 1:
                ap.error("--max-minutes needs exactly one eval block")
            state_f = run_dir / f"resume_{block}.npz"
            key = {"model": m, "checkpoint_sha256": ck_sha, "block": block,
                   "n_rows": {v: int(len(r)) for v, r in rows.items()}}
            resume = None
            if state_f.exists():
                with np.load(state_f, allow_pickle=False) as z:
                    if json.loads(str(z["key"])) != key:
                        raise SystemExit(f"resume state {state_f} does not match this job")
                    resume = {"out": {v: z[f"out__{v}"] for v in rows}, "next_site_pos": int(z["next_site_pos"])}
                log(f"[predict] resuming {m} {block} at site position {resume['next_site_pos']}")
            fc, info = model.predict_multi_resumable(data, rows, resume=resume,
                                                     deadline=t_start + 60 * float(a.max_minutes))
            if not info["complete"]:
                tmp = state_f.with_name(".tmp_" + state_f.name)
                with open(tmp, "wb") as fh:
                    np.savez(fh, key=json.dumps(key), next_site_pos=info["next_site_pos"],
                             **{f"out__{v}": q for v, q in fc.items()})
                tmp.replace(state_f)
                log(f"[predict] PARTIAL {m} {block}: {info['next_site_pos']}/{info['n_sites']} sites; "
                    f"state saved, exit 75 (re-run the same command to resume)")
                return 75
            log(f"[predict] {m} {block}: complete ({info['n_sites']} sites)")
        else:
            fc = model.predict(data, rows)
        files += runner.save_forecasts(data, m, fc, block, run_dir,
                                       meta={"checkpoint": ck, "checkpoint_sha256": ck_sha, "smoke": cfg["_smoke"]})
        log(f"[predict] {m} {block}: {sum(len(r) for r in rows.values())} rows")
        (run_dir / f"resume_{block}.npz").unlink(missing_ok=True)
    runner.write_json(run_dir / "predict_manifest.json",
                      {"model": m, "checkpoint": ck, "checkpoint_sha256": ck_sha, "blocks": runner.eval_blocks(a),
                       "files": {f: guard.artifact_sha256(f) for f in files}, "access": data.access,
                       "weather_mask": WEATHER_MASK or None})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
