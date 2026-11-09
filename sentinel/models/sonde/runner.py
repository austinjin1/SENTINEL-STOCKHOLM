# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Shared plumbing for the SENTINEL-Sonde driver scripts (scripts/train/sonde_*.py,
scripts/eval/sonde_*.py): YAML configs with ``smoke:`` overrides, the M1 GPU budget, run
directories, TrainingRun wrapping, guarded data preparation and scoring summaries.

Smoke runs (``--smoke``) use ≤ 30 train-role + ≤ 10 val-role sites, may use a TEMPORARY split
(``--temp-split``), write under ``$SENTINEL_RUNS/sonde/smoke/`` and can never write to the results
registry (:func:`assert_registrable`).
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ...core import paths
from ...core.registry import config_hash
from . import common, guard

M1_GPUS = (0,)  # M1 home GPU (CONTRIBUTING_V2 shared-host allocation); GPU 4 = shared queue, not used
WORKSTREAM = "M1-sonde"
PLAN_REF = "RESEARCH_PLAN.md §5.1 (M1 SENTINEL-Sonde), §12 (EW-v2)"


# ----------------------------------------------------------------------------- config

def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(path: str | Path, *, smoke: bool = False) -> dict:
    import yaml

    p = Path(path)
    cfg = yaml.safe_load(p.read_text()) or {}
    common_p = p.parent / "common.yaml"
    if common_p.is_file() and p.name != "common.yaml":
        cfg = _merge(yaml.safe_load(common_p.read_text()) or {}, cfg)
    smoke_over = cfg.pop("smoke", {}) or {}
    if smoke:
        cfg = _merge(cfg, smoke_over)
    cfg["_config_file"] = str(p)
    cfg["_smoke"] = bool(smoke)
    return cfg


def apply_overrides(cfg: dict, sets: list[str] | None) -> dict:
    """--set a.b=value overrides (value parsed as YAML); recorded in the config (and its hash)."""
    import yaml

    for item in sets or []:
        key, val = item.split("=", 1)
        d = cfg
        parts = key.split(".")
        for k in parts[:-1]:
            d = d.setdefault(k, {})
        d[parts[-1]] = yaml.safe_load(val)
    if sets:
        cfg["overrides"] = list(sets)
    return cfg


def base_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny smoke test (not a result; never registered)")
    ap.add_argument("--temp-split", action="store_true",
                    help="(smoke only) build a TEMPORARY split if the DEV split file does not exist")
    ap.add_argument("--split", default=None,
                    help="split JSON (default: config 'split' = docs/preregistration/sonde_split.json; "
                         "docs/preregistration/sonde_split_dev.json is for --smoke only)")
    ap.add_argument("--eval-blocks", default="val", help="comma list of blocks to score (default: val)")
    ap.add_argument("--allow-test", action="store_true",
                    help="permit test/test_b blocks or test-role sites (also needs docs/preregistration/FREEZE_sonde.json)")
    ap.add_argument("--freeze-file", default=None)
    ap.add_argument("--include-test-sites", action="store_true",
                    help="(predict/score only) also load test-role sites; needs --allow-test + freeze file")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", action="append", default=[], help="config override key=value (YAML value)")
    return ap


# ----------------------------------------------------------------------------- GPU

def setup_gpu(name: str, prefer: int = 0) -> int:
    """Lock one GPU from the M1 budget (0, 4) via sentinel.core.gpu; call before CUDA init."""
    env = os.environ.get("SENTINEL_GPUS")
    allowed = [g for g in (int(x) for x in env.split(",") if x.strip())] if env else list(M1_GPUS)
    allowed = [g for g in allowed if g in M1_GPUS] or list(M1_GPUS)
    os.environ["SENTINEL_GPUS"] = ",".join(map(str, allowed))
    import sys

    already = "sentinel.core.gpu" in sys.modules
    from ...core import gpu as gpu_mod

    if already:  # module read SENTINEL_GPUS at import time; narrow it to the M1 budget
        if gpu_mod._held:
            raise RuntimeError("a GPU is already held by this process")
        gpu_mod.ALLOWED = list(allowed)
    g = gpu_mod.acquire_gpu(name, prefer=prefer if prefer in allowed else allowed[0])
    limit_cpu_threads(2)   # shared-host rule: GPU jobs use 2 CPU threads
    return g


def limit_cpu_threads(n: int) -> None:
    """Cap torch intra-op and BLAS/OpenMP thread pools (CONTRIBUTING_V2 shared-host allocation)."""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(n)
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(n)
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        torch.set_num_threads(n)
    except Exception:  # noqa: BLE001
        pass


# ----------------------------------------------------------------------------- runs

def run_directory(model: str, cfg: dict, run_name: str | None = None) -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = run_name or f"{model}_{stamp}"
    sub = f"smoke/{name}" if cfg.get("_smoke") else name
    return paths.run_dir("sonde", sub)


def experiment_name(model: str, cfg: dict) -> str:
    return f"SMOKE/{model}" if cfg.get("_smoke") else model


def training_run(model: str, cfg: dict, run_dir: Path, data_manifests=("sonde_daily",)):
    from ...core.runlog import TrainingRun

    run = TrainingRun(WORKSTREAM, experiment_name(model, cfg), config=public_config(cfg),
                      data=list(data_manifests), plan_ref=PLAN_REF,
                      iteration="smoke" if cfg.get("_smoke") else str(cfg.get("iteration", "base")),
                      artifacts_dir=str(run_dir))
    if cfg.get("_smoke"):
        run.note("SMOKE TEST (<=30 train sites, val block, tiny budget); NOT a result; nothing registered")
    return run


def public_config(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if not k.startswith("_") or k in ("_smoke", "_config_file")}


def log_config_tried(model: str, cfg: dict, run_dir: Path, status: str = "started") -> None:
    """Append-only count of configurations tried per model (RESEARCH_PLAN R6)."""
    p = paths.RUNS_ROOT / "sonde" / "configs_tried.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {"time": _dt.datetime.now().isoformat(timespec="seconds"), "model": model, "smoke": bool(cfg.get("_smoke")),
           "config_hash": config_hash(public_config(cfg)), "run_dir": str(run_dir), "status": status}
    with open(p, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def make_logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"{_dt.datetime.now().strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(path, "a") as fh:
            fh.write(line + "\n")

    return log


def record_checkpoint(run_dir: Path, model: str, path: Path, log=print) -> str:
    """Write <run_dir>/checkpoint.json with the checkpoint's SHA-256 (the value that goes into
    FREEZE_sonde.json weights_sha256[model] if this run is frozen)."""
    sha = guard.artifact_sha256(path)
    write_json(run_dir / "checkpoint.json", {"model": model, "path": str(path), "sha256": sha})
    log(f"[checkpoint] {model}: {path} sha256 {sha}")
    return sha


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def assert_registrable(cfg: dict, split_info: dict) -> None:
    """Smoke runs and non-final (DEV / temporary) splits can never be written to the registry."""
    if cfg.get("_smoke") or split_info.get("temporary") or not split_info.get("final"):
        raise RuntimeError("refusing to register results from a smoke run or a non-final split")


# ----------------------------------------------------------------------------- data

def eval_blocks(args) -> tuple:
    return tuple(b.strip() for b in args.eval_blocks.split(",") if b.strip())


def prepare(args, cfg: dict, run_dir: Path, *, need_blocks=("train", "val"), training: bool = False,
            log=print) -> common.SondeData:
    """Guarded data preparation. ``training=True`` (every script that fits parameters) refuses
    test/test_b blocks and test-role sites outright — test forecasts come only from
    scripts/eval/sonde_predict.py with frozen, hash-verified checkpoints."""
    if getattr(args, "include_test_sites", False):
        cfg["include_test_sites"] = True
    blocks = tuple(dict.fromkeys(list(need_blocks) + list(eval_blocks(args))))
    if training and (set(blocks) & set(guard.PROTECTED_BLOCKS) or cfg.get("include_test_sites")):
        raise guard.ProtectedDataError("training scripts never load test/test_b blocks or test-role sites; "
                                       "use scripts/eval/sonde_predict.py with frozen checkpoints")
    split_path = args.split or cfg.get("split")
    if cfg.get("_smoke") and args.temp_split and not Path(paths.REPO_ROOT / split_path).is_file() \
            and not Path(split_path).is_file():
        split, info = common.make_temp_split(paths.RUNS_ROOT / "sonde" / "smoke" / "temp_split.json",
                                             site_types=cfg.get("temp_split_site_types"))
        log(f"[prepare] TEMPORARY split for smoke test: {info}")
    else:
        if args.temp_split and not cfg.get("_smoke"):
            raise RuntimeError("--temp-split is only allowed together with --smoke")
        sp = Path(split_path)
        split, info = common.load_split(sp if sp.is_absolute() else paths.REPO_ROOT / sp)
    if not info.get("final") and not cfg.get("_smoke"):
        raise RuntimeError(f"split {info['path']} is not the FINAL split; non-final splits are for --smoke only")
    roles = ["train", "val"] + (["test"] if cfg.get("include_test_sites") else [])
    if cfg.get("_smoke"):
        pick = common.smoke_sites(split, n_train=int(cfg.get("smoke_n_train", 30)),
                                  n_val=int(cfg.get("smoke_n_val", 10)), seed=int(cfg.get("seed", 0)))
        sites = sorted(pick["train"] + pick["val"])
    else:
        sites = sorted(s for r in roles for s in split.sites(r))
    missing = common.wait_for_sites(sites)
    if missing:
        log(f"[prepare] {len(missing)} split sites have no daily table (excluded, recorded): {missing[:10]}")
    sites = [s for s in sites if s not in set(missing)]
    if cfg.get("site_types"):
        reg = common.site_registry()
        sites = [s for s in sites if s in reg.index and reg.loc[s, "site_type"] in set(cfg["site_types"])]
    data = common.prepare_data(split, info, sites, blocks=blocks, allow_test=args.allow_test,
                               freeze_file=args.freeze_file, horizons=tuple(cfg.get("horizons", common.HORIZONS)))
    if cfg.get("use_catchment"):
        sc = common.load_catchment(data.sites)
        data.static = pd.concat([data.static, sc.set_index(data.static.index)], axis=1)
        log(f"[prepare] catchment attributes: {sc.shape[1]} StreamCat columns, "
            f"{int(sc.notna().any(axis=1).sum())} sites matched")
    if cfg.get("use_weather"):
        data.weather = common.load_weather(data.cube)
        log(f"[prepare] weather covariates (observed, lag {common.WEATHER_LAG_DAYS} d): "
            f"{int(np.isfinite(data.weather[..., 0]).any(axis=1).sum())} sites with forcings")
    n_role = pd.Series(data.roles).value_counts().to_dict()
    log(f"[prepare] {len(data.sites)} sites {n_role}; dates {data.cube.dates[0].date()}..{data.cube.dates[-1].date()}; "
        f"blocks {blocks}")
    write_json(run_dir / "data_info.json", {"split": info, "access": data.access, "blocks": blocks,
                                            "sites_without_table": missing,
                                            "n_unscorable_no_climatology": len(data.access.get(
                                                "unscorable_sites_no_climatology", [])),
                                            "n_sites": len(data.sites), "roles": n_role,
                                            "sites": {s: r for s, r in zip(data.sites, data.roles)},
                                            "cube_end": str(data.cube.dates[-1].date())})
    return data


def model_variables(cfg: dict, data: common.SondeData) -> list[str]:
    vs = cfg.get("variables") or list(data.cube.variables)
    return [v for v in vs if v in data.cube.variables]


# ----------------------------------------------------------------------------- scoring

def score_and_summarize(data: common.SondeData, model: str, forecasts: dict, block: str, out_dir: Path, *,
                        refs=("damped_anomaly", "climatology", "persistence"), log=print) -> pd.DataFrame:
    """Score ``model`` and the naive baselines on the common rows; write samples / site metrics /
    skill summary; return the skill summary (descriptive, no CIs)."""
    from .evaluate import score_forecast, site_metrics, skill_summary

    frames = []
    for v, q in forecasts.items():
        rows = data.rows(v, block)
        bad = common.scorable_finite_check(q, rows, data.horizons)
        if any(bad.values()):
            raise ValueError(f"{model}/{v}: non-finite forecasts on common rows {bad}")
        frames.append(score_forecast(v, q, rows.score_mask, rows.obs, rows.site_no, rows.issue_date, rows.role,
                                     data.horizons, model))
        for r in refs:
            qb = common.baseline_forecast(data, v, rows, r)
            frames.append(score_forecast(v, qb, rows.score_mask, rows.obs, rows.site_no, rows.issue_date,
                                         rows.role, data.horizons, r))
    samples = pd.concat([f for f in frames if len(f)], ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(out_dir / f"samples_{model}_{block}.parquet", index=False, compression="zstd")
    site = site_metrics(samples, min_n=30)
    site["role"] = site["site_no"].map(dict(zip(data.sites, data.roles)))
    site.to_parquet(out_dir / f"site_metrics_{model}_{block}.parquet", index=False)
    summ = []
    for role in (None, "train", "val", "test"):
        sd = site if role is None else site[site["role"] == role]
        if not len(sd):
            continue
        s = skill_summary(sd, model, refs)
        s.insert(1, "site_role", "all" if role is None else role)
        summ.append(s)
    summary = pd.concat(summ, ignore_index=True) if summ else pd.DataFrame()
    summary.to_csv(out_dir / f"skill_summary_{model}_{block}.csv", index=False)
    return summary


def save_forecasts(data: common.SondeData, model: str, forecasts: dict, block: str, out_dir: Path,
                   meta: dict | None = None) -> list[str]:
    files = []
    for v, q in forecasts.items():
        p = common.save_forecast(out_dir / "forecasts" / f"{model}__{v}__{block}.npz", model=model,
                                 rows=data.rows(v, block), q=q, horizons=data.horizons, meta=meta)
        files.append(str(p))
    return files


def print_summary(summary: pd.DataFrame, title: str, log=print, horizons=(1, 3, 7, 14)) -> None:
    if summary.empty:
        log(f"{title}: (no scored rows)")
        return
    s = summary[(summary["site_role"] == "all") & (summary["h"].isin(horizons))]
    cols = [c for c in ["var", "h", "n_sites", "median_skill_vs_damped_anomaly", "median_skill_vs_climatology",
                        "median_skill_vs_persistence", "median_cov90"] if c in s.columns]
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        log(f"{title}\n{s[cols].to_string(index=False)}")


def describe_rows(data: common.SondeData, block: str, variables) -> dict:
    out = {}
    for v in variables:
        r = data.rows(v, block)
        out[v] = {"n_rows": len(r), "n_sites": int(np.unique(r.si).size) if len(r) else 0,
                  "n_scorable": {int(h): int(m.sum()) for h, m in r.score_mask.items()},
                  "dropped_baseline_undefined": {int(h): n for h, n in r.n_dropped_baseline_undefined.items()},
                  "dropped_provisional_target": {int(h): n for h, n in r.n_dropped_provisional.items()}}
    return out

