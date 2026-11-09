# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Shared plumbing for the SENTINEL-HydroMet drivers (scripts/train/hydromet_*.py,
scripts/eval/hydromet_*.py): YAML configs with ``smoke:`` overrides, one GPU via sentinel.core.gpu
(prefer GPU 3), run directories, TrainingRun wrapping, guarded data preparation, split reports,
scoring summaries and the (non-smoke, FINAL-split-only) registry path.

Smoke runs (``--smoke``) use ≤ 50 train-role sites, write under ``$SENTINEL_RUNS/hydromet/smoke/``
and can never write to the results registry (:func:`assert_registrable`).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ...core import paths
from ...core.registry import config_hash
from ..sonde.runner import load_config, make_logger, write_json  # noqa: F401  (generic helpers, M1-owned)
from . import data as D
from . import guard

WORKSTREAM = "M6-hydromet"
PLAN_REF = "RESEARCH_PLAN.md §5.6 (M6 SENTINEL-HydroMet)"
GPU_PREFER = 3
MODEL_PREFIX = "hydromet_"


def base_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny smoke test (not a result; never registered)")
    ap.add_argument("--split", default=None, help="split JSON (default: config 'split' = FINAL sonde split)")
    ap.add_argument("--eval-blocks", default="val", help="comma list of blocks to score (default: val)")
    ap.add_argument("--allow-test", action="store_true",
                    help="permit test/test_b blocks or test-role sites (eval scripts only; also needs "
                         "docs/preregistration/FREEZE_hydromet.json)")
    ap.add_argument("--freeze-file", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--track-r", default=None,
                    help="protocol §10 track R: held-out region (training: R1_northeast ... R6_pacific; scoring: cv)")
    ap.add_argument("--seed", type=int, default=None)
    return ap


def eval_blocks(args) -> tuple:
    return tuple(b.strip() for b in args.eval_blocks.split(",") if b.strip())


def setup_gpu(name: str, prefer: int = GPU_PREFER) -> int:
    """Lock exactly one GPU via sentinel.core.gpu (preferring GPU 3); call before CUDA init."""
    from ...core import gpu as gpu_mod

    if gpu_mod._held:
        raise RuntimeError("a GPU is already held by this process (max 1 per job)")
    return gpu_mod.acquire_gpu(name, prefer=prefer if prefer in gpu_mod.ALLOWED else None)


def run_directory(model: str, cfg: dict, run_name: str | None = None) -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = run_name or f"{model}_{stamp}"
    return paths.run_dir("hydromet", f"smoke/{name}" if cfg.get("_smoke") else name)


def public_config(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if not k.startswith("_") or k in ("_smoke", "_config_file")}


def training_run(model: str, cfg: dict, run_dir: Path, data_manifests=("sonde_daily", "forcings_daily", "streamcat")):
    from ...core.runlog import TrainingRun

    run = TrainingRun(WORKSTREAM, f"SMOKE/{model}" if cfg.get("_smoke") else model, config=public_config(cfg),
                      data=list(data_manifests), plan_ref=PLAN_REF,
                      iteration="smoke" if cfg.get("_smoke") else str(cfg.get("iteration", "base")),
                      artifacts_dir=str(run_dir))
    if cfg.get("_smoke"):
        run.note("SMOKE TEST (<=50 train-role sites, val block); NOT a result; nothing registered")
    return run


def log_config_tried(model: str, cfg: dict, run_dir: Path, status: str = "started") -> None:
    """Append-only count of configurations tried (RESEARCH_PLAN R6)."""
    p = paths.RUNS_ROOT / "hydromet" / "configs_tried.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {"time": _dt.datetime.now().isoformat(timespec="seconds"), "model": model, "smoke": bool(cfg.get("_smoke")),
           "config_hash": config_hash(public_config(cfg)), "run_dir": str(run_dir), "status": status}
    with open(p, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def assert_registrable(cfg: dict, split_info: dict) -> None:
    if cfg.get("_smoke") or not split_info.get("final"):
        raise RuntimeError("refusing to register results from a smoke run or a non-final split")


def split_report_for(data: D.HydroMetData) -> dict:
    """``sentinel.eval.splits.split_report`` over the loaded sites (+ nearest train-site distances)."""
    from ...eval.splits import SiteSplit, split_report

    tr = [data.sites[i] for i in data.site_idx(["train"])]
    va = [data.sites[i] for i in data.site_idx(["val"])]
    te = [data.sites[i] for i in data.site_idx(["test"])]
    ss = SiteSplit(tr, va, te, method="sonde_split basin groups (HUC4 + NLDI flow connectivity)",
                   params={"split_sha256": data.split.sha256(), "path": data.split_info.get("path")})
    lat = dict(zip(data.sites, data.static["lat"].to_numpy()))
    lon = dict(zip(data.sites, data.static["lon"].to_numpy()))
    rep = split_report(ss, lat, lon)
    if tr and va:
        from ...eval.splits import nearest_distance_km

        d = nearest_distance_km([lat[s] for s in va], [lon[s] for s in va], [lat[s] for s in tr], [lon[s] for s in tr])
        rep.update({"val_to_train_km_min": float(np.nanmin(d)), "val_to_train_km_median": float(np.nanmedian(d))})
    return rep


def _track_label(cfg: dict) -> str:
    """Registry split-label prefix for the harder tracks (protocol §10)."""
    tr = cfg.get("track_r")
    if not tr:
        return ""
    if tr == "test":
        return "track R (leave-region-out) test look: "
    return "track R (leave-region-out) CV, held-out-region " if tr == "cv" else f"track R fold {tr}: "


def apply_track_r(data: D.HydroMetData, fold: str, *, log=print) -> None:
    """Protocol §10. Re-label the loaded FINAL train/val-role sites for track R: ``fold = <region>`` (training)
    -> 'train' outside the region, 'val' inside it; ``fold = 'cv'`` (scoring the merged CV cell) -> every
    train/val-role site 'val'. FINAL test-role sites keep 'test'. The fold file's sha256 is checked."""
    from . import tracks as TR

    body = TR.load_folds()
    if body["split_sha256"] != data.split.sha256():
        raise ValueError("track-R fold file was built from a different split")
    if fold == "test":   # M6R/test look: FINAL test-role sites keep their role; folds are attached by the look script
        data.info["track_r"] = {"fold": "test", "fold_sha256": body["sha256"]}
        return
    if fold != "cv" and fold not in TR.REGION_FOLDS:
        raise ValueError(f"unknown track-R fold {fold!r}")
    new = []
    for s, r in zip(data.sites, data.roles):
        if r == "test":
            new.append("test")
        elif s not in body["fold_of"]:
            raise ValueError(f"site {s} missing from the track-R fold file")
        else:
            new.append("val" if fold == "cv" or body["fold_of"][s] == fold else "train")
    data.roles = np.array(new)
    data.info["track_r"] = {"fold": fold, "fold_sha256": body["sha256"],
                            "roles": pd.Series(new).value_counts().to_dict()}
    log(f"[prepare] track R fold {fold}: roles {data.info['track_r']['roles']}")


def prepare(args, cfg: dict, run_dir: Path, *, training: bool = True, log=print,
            roles=("train", "val")) -> tuple[D.HydroMetData, np.ndarray, np.ndarray]:
    """Guarded data preparation + inner (early-stopping) split of the train-role sites."""
    blocks = tuple(dict.fromkeys(["train"] + list(eval_blocks(args))))
    if training and (set(blocks) & set(guard.PROTECTED_BLOCKS) or "test" in roles):
        raise guard.ProtectedDataError("training scripts never load test/test_b blocks or test-role sites; use "
                                       "scripts/eval/hydromet_predict.py with frozen artifacts")
    split, info = D.load_split(args.split or cfg.get("split"))   # refuses any non-FINAL split (review C3a)
    roles = tuple(dict.fromkeys("train" if r in ("inner", "fit") else r for r in roles))   # pseudo-roles
    if cfg.get("_smoke"):
        if set(roles) - {"train"}:
            raise guard.ProtectedDataError("smoke runs load FINAL-split train-role sites only")
        sites = D.smoke_sites(split, n_train=int(cfg.get("smoke_n_train", 50)), n_val=0,
                              seed=int(cfg.get("smoke_site_seed", 0)))   # train-role only; never the model seed
        if len(sites) > 50 or any(split.site_role[s] != "train" for s in sites):
            raise RuntimeError("smoke tests use at most 50 FINAL-split train-role sites and nothing else")
        elig_info = {"smoke": True, "note": "train-role sites only; scored on inner train-role groups"}
    else:
        # track-R CV scoring: every FINAL train/val-role site is a held-out-region site of some fold (protocol §10)
        sites, elig_info = D.eligible_sites(split, ("train", "val") if cfg.get("track_r") == "cv" else roles)
    data = D.prepare_data(split, info, sites, blocks=blocks, allow_test=getattr(args, "allow_test", False),
                          freeze_file=getattr(args, "freeze_file", None), training=training, log=log)
    if cfg.get("track_r"):
        apply_track_r(data, cfg["track_r"], log=log)
    if cfg.get("upstream_forcing"):
        D.attach_upstream_forcing(data, cfg["upstream_forcing"], log=log)
    if cfg.get("static_ablation"):
        data.static = D.apply_static_ablation(data.static, cfg["static_ablation"])
        log(f"[prepare] static ablation {cfg['static_ablation']!r}: {data.static.shape[1]} static columns")
    if len(data.site_idx(["train"])):
        # fixed across model seeds (protocol §2): every seed and every baseline sees the same inner groups
        fit_idx, inner_idx = D.inner_split(data, frac=float(cfg.get("inner_frac", 0.15)),
                                           seed=int(cfg.get("inner_seed", 0)))
    else:  # prediction-only load (e.g. frozen artifacts on test-role sites)
        fit_idx = inner_idx = np.zeros(0, int)
    data.info["inner_sites"] = [data.sites[i] for i in inner_idx]
    log(f"[prepare] {len(data.sites)} sites {data.info['roles']}; fit {len(fit_idx)} / inner {len(inner_idx)} "
        f"train-role sites; dates {data.dates[0].date()}..{data.dates[-1].date()}; blocks {blocks}")
    write_json(run_dir / "data_info.json", {"split": info, "access": {k: v for k, v in data.access.items()
                                                                     if k != "freeze"},
                                            "eligibility": elig_info, "blocks": blocks, **data.info,
                                            "fit_sites": [data.sites[i] for i in fit_idx],
                                            "inner_sites": [data.sites[i] for i in inner_idx],
                                            "sites": dict(zip(data.sites, data.roles.tolist()))})
    write_json(run_dir / "split_report.json", split_report_for(data))
    return data, fit_idx, inner_idx


def check_mode(cfg: dict) -> int:
    """Forcing mode: ``nowcast`` (same-day observed weather, lag 0) or ``lagged`` (forcings of days
    <= t - lag only, lag >= 1). An NWP-driven forecast mode needs ``nwp_archive`` and is not
    implemented: observed future weather is never used under the name "forecast"."""
    mode = cfg.get("mode", "nowcast")
    lag = int(cfg.get("forcing_lag_days", 0))
    if mode == "nowcast" and lag != 0:
        raise ValueError("mode 'nowcast' requires forcing_lag_days = 0")
    if mode == "lagged" and lag < 1:
        raise ValueError("mode 'lagged' requires forcing_lag_days >= 1")
    if mode not in ("nowcast", "lagged"):
        raise NotImplementedError(f"mode {mode!r}: NWP-forecast mode (nwp_archive inputs) is not implemented yet")
    return lag


def eval_cells(data: D.HydroMetData, block: str, roles=("val",)) -> tuple[np.ndarray, np.ndarray]:
    """(site_idx, t_idx) of an evaluation cell: sites of ``roles`` × dates of ``block`` (embargoed)."""
    return data.site_idx(roles), np.flatnonzero(data.block_mask(block))


def score_and_write(data: D.HydroMetData, preds: dict, block: str, out_dir: Path, *, roles=("val",), refs=(),
                    n_boot: int = 2000, min_days: int = 60, tag: str = "", log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    from .evaluate import site_metrics, summarize

    si, ti = eval_cells(data, block, roles)
    site_df, row_info = site_metrics(data, preds, si, ti, block, min_days=min_days)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{block}_{'-'.join(roles)}{tag}"
    site_df.to_parquet(out_dir / f"site_metrics_{suffix}.parquet", index=False)
    write_json(out_dir / f"rows_{suffix}.json", row_info)
    summ = summarize(site_df, refs=refs, n_boot=n_boot)
    summ.to_csv(out_dir / f"summary_{suffix}.csv", index=False)
    return site_df, summ


def print_summary(summ: pd.DataFrame, title: str, log=print, metrics=("nse", "kge", "rmse", "nse_log")) -> None:
    if summ.empty:
        log(f"{title}: (nothing scored)")
        return
    s = summ[summ["metric"].isin(metrics)].copy()
    s["value"] = s.apply(lambda r: f"{r['median']:.3f} [{r['ci_low']:.3f},{r['ci_high']:.3f}]", axis=1)
    tab = s.pivot_table(index=["var", "model"], columns="metric", values="value", aggfunc="first")
    ns = s.groupby(["var", "model"])["n_sites"].max()
    tab["n_sites"] = ns
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_rows", 200):
        log(f"{title}\n{tab.to_string()}")


SENTINEL_MODEL_RE = r"ealstm.*"


def _is_baseline(model: str) -> bool:
    import re

    return re.fullmatch(SENTINEL_MODEL_RE, model) is None


def register_summary(summ: pd.DataFrame, cfg: dict, split_info: dict, *, experiment: str, block: str,
                     artifacts=(), roles=("val",)) -> int:
    """Median-over-sites metrics with basin-group-bootstrap CIs (site-bootstrap CI in ``extra``) to the
    registry (FINAL split, non-smoke only)."""
    from ...core.registry import log_result

    assert_registrable(cfg, split_info)
    n = 0
    for r in summ.itertuples(index=False):
        if not np.isfinite(r.median):
            continue
        log_result(workstream=WORKSTREAM, experiment=experiment, model=MODEL_PREFIX + r.model,
                   metric=f"median_site_{r.metric}", value=float(r.median), split=_track_label(cfg) + f"{r.role}-role sites x {block} block",
                   n=int(r.n_sites), ci_low=float(r.ci_low), ci_high=float(r.ci_high),
                   ci_method=f"percentile bootstrap of the median, resampling basin groups (B={cfg.get('n_boot')})",
                   target=r.var, is_baseline=_is_baseline(r.model), config=public_config(cfg),
                   data_manifests=["sonde_daily", "forcings_daily", "streamcat"], artifacts=list(artifacts),
                   extra={"split_sha256": split_info.get("sha256"), "n_cells": int(r.n_cells),
                          "n_groups": int(r.n_groups), "ci_site_resampling": [float(r.ci_low_site), float(r.ci_high_site)]})
        n += 1
    return n


def register_pooled_and_primary(pooled: pd.DataFrame, primary: pd.DataFrame | None, cfg: dict, split_info: dict, *,
                                experiment: str, block: str, roles=("val",), artifacts=()) -> int:
    """Registry entries for pooled cross-site skill and the pre-registered primary comparison."""
    from ...core.registry import log_result

    assert_registrable(cfg, split_info)
    split = _track_label(cfg) + f"{'/'.join(roles)}-role sites x {block} block"
    common = dict(workstream=WORKSTREAM, experiment=experiment, split=split, config=public_config(cfg),
                  data_manifests=["sonde_daily", "forcings_daily", "streamcat"], artifacts=list(artifacts))
    n = 0
    for r in pooled.itertuples(index=False):
        if np.isfinite(r.value):
            log_result(model=MODEL_PREFIX + r.model, metric=r.metric, value=float(r.value), n=int(r.n_sites),
                       ci_low=float(r.ci_low), ci_high=float(r.ci_high), is_baseline=_is_baseline(r.model),
                       ci_method=f"percentile bootstrap resampling basin groups (B={cfg.get('n_boot')})",
                       target=r.var, extra={"n_cells": int(r.n_cells), "n_groups": int(r.n_groups),
                                            "ci_site_resampling": [float(r.ci_low_site), float(r.ci_high_site)],
                                            "split_sha256": split_info.get("sha256")}, **common)
            n += 1
    for r in (primary.to_dict("records") if primary is not None else []):
        if r.get("status") != "tested":
            continue
        for fam in ("gbm", "physics"):
            log_result(model=MODEL_PREFIX + r["model"], metric=f"diff_median_site_kge_vs_best_{fam}",
                       value=float(r[f"{fam}_diff"]), n=int(r[f"{fam}_n_sites"]), ci_low=float(r[f"{fam}_ci_low"]),
                       ci_high=float(r[f"{fam}_ci_high"]),
                       ci_method="paired percentile bootstrap resampling basin groups; IUT over both families; "
                                 "Holm across primary hypotheses", target=r["var"],
                       extra={"comparator": r[f"{fam}_comparator"], "p_value": float(r[f"{fam}_p"]),
                              "p_iut": float(r["p_iut"]), "holm_reject": bool(r["holm_reject"]), "won": bool(r["won"]),
                              "every_seed_positive": bool(r[f"{fam}_every_seed_positive"]),
                              "seed_diffs": r[f"{fam}_seed_diffs"], "n_groups": int(r[f"{fam}_n_groups"]),
                              "ci_site_resampling": [float(r[f"{fam}_ci_low_site"]), float(r[f"{fam}_ci_high_site"])],
                              "n_hypotheses": int(r["n_hypotheses"]), "split_sha256": split_info.get("sha256")},
                       **common)
            n += 1
    return n


def register_secondary(sec: pd.DataFrame | None, cfg: dict, split_info: dict, *, experiment: str, block: str,
                       roles=("val",), artifacts=()) -> int:
    """Registry entries for the protocol §7 secondary paired comparisons (descriptive, no multiplicity
    control): static-input ablation and physics-DO head vs plain EA-LSTM."""
    from ...core.registry import log_result

    if sec is None or not len(sec):
        return 0
    assert_registrable(cfg, split_info)
    n = 0
    for r in sec.to_dict("records"):
        if not np.isfinite(r["diff"]):
            continue
        log_result(workstream=WORKSTREAM, experiment=experiment, model=MODEL_PREFIX + r["model"],
                   metric=f"diff_median_site_kge_vs_{r['reference']}", value=float(r["diff"]), n=int(r["n_sites"]),
                   ci_low=float(r["ci_low"]), ci_high=float(r["ci_high"]), target=r["var"],
                   is_baseline=_is_baseline(r["model"]),
                   ci_method="paired percentile bootstrap resampling basin groups (descriptive secondary analysis)",
                   split=_track_label(cfg) + f"{'/'.join(roles)}-role sites x {block} block", config=public_config(cfg),
                   data_manifests=["sonde_daily", "forcings_daily", "streamcat"], artifacts=list(artifacts),
                   extra={"analysis": r["analysis"], "reference": r["reference"], "p_value": float(r["p_value"]),
                          "median_model": float(r["median_model"]), "median_reference": float(r["median_reference"]),
                          "n_groups": int(r["n_groups"]),
                          "ci_site_resampling": [float(r["ci_low_site"]), float(r["ci_high_site"])],
                          "split_sha256": split_info.get("sha256")})
        n += 1
    return n


def env_threads(n: int = 8) -> None:
    """Cap BLAS/OpenMP/torch threads at ``n`` (CONTRIBUTING_V2: <= 8 CPU threads per job); call
    before numpy/torch do heavy work."""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        cur = os.environ.get(k)
        os.environ[k] = str(min(n, int(cur)) if cur and cur.isdigit() else n)
    try:
        import torch

        torch.set_num_threads(min(n, torch.get_num_threads()))
    except ImportError:  # pragma: no cover
        pass
