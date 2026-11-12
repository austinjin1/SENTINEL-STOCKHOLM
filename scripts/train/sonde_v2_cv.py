#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""5-fold basin-group CV for SENTINEL-Sonde v2 arms (``sonde_arch_v2_amendment.md`` §2h; lead ruling L3; the
consolidated deviations file ``sonde_arch_v2_deviations_v2.md``).

* Guard: every phase that writes a prediction or a registry row first calls ``parity.require_stamp`` (lead-written
  stamp row with the current sha256 of the deviations file and of the v2 code) and, for runs,
  ``parity.require_declared`` (the declared configuration lists; ``--phase declare``).
* Folds: train+val-role split sites, whole basin groups, seed 20261110, written ONCE by ``--phase folds``.
* Fold k: fitting on the in-fold fit sites × TRAIN block only; early stopping on the inner 15 % of the in-fold basin
  groups × TRAIN block (same site sets for every fitted arm). Out-of-fold (OOF) forecasts on fold-k sites × VAL
  block common rows with the archived, bias-corrected GEFS c00 forecasts (the one declared product). Val-block rows
  are used only as OOF scores and, later, for calibration / ensemble weights fitted on OOF predictions.
* Networks: stage 1 on perfect-prog inputs (observed gridMET + stage-1 error model), stage 2 fine-tune on the
  archived GEFS of the train block (NaN = past-weather mode where no archive exists: 2020-01-01..2020-09-22; nothing
  is substituted), same optimiser settings as stage 1 except lr/3, 10 epochs, patience 3.
* GBM-type references: fitted on the archived GEFS of the train block (same no-substitution rule), settings from
  ``declared.GBM_SETTINGS`` (refused until ruled), on FULL fit rows, one GPU per process, resumable at booster
  level (lead GBM ruling G-2, G-3, G-6; :mod:`sentinel.models.sonde_v2.gbm_resume`). One variable at a time; a
  finished variable is not refitted and a finished booster is not refitted. Before every CV GBM process the
  GPU-hour gate of G-7(d) is checked against the CV projection written by ``--phase gbm_budget``.
* Targets written: references the 8 scored variables; SONDE-HF the scored and the §2d auxiliary variables (C3: its
  5 primary targets only); DO % saturation = deterministic transform of SONDE-HF's DO-mean quantiles at its T-mean
  median.
* Output: ``sentinel_runs/sonde/v2_cv/<arm>_<config>[_<target>]_s<seed>/fold<k>/forecasts/*.npz`` (frozen forecast
  format); ``--phase combine`` concatenates the 5 folds after the cross-arm parity check. No score is computed here.

    python -B scripts/train/sonde_v2_cv.py --phase declare
    python -B scripts/train/sonde_v2_cv.py --phase folds
    python -B scripts/train/sonde_v2_cv.py --phase run --arm sonde_hf --config C1 --fold 0      # GPU, exit 75 = resume
    python -B scripts/train/sonde_v2_cv.py --phase run --arm rnh --config R1 --target temp_c_mean --fold 0
    python -B scripts/train/sonde_v2_cv.py --phase gbm_budget                                   # CV GPU-h projection
    python -B scripts/train/sonde_v2_cv.py --phase run --arm gbm_nwp --config Q19 --fold 0      # GPU, exit 75 = resume
    python -B scripts/train/sonde_v2_cv.py --phase parity
    python -B scripts/train/sonde_v2_cv.py --phase combine --arm sonde_hf --config C1
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sentinel.core import paths  # noqa: E402
from sentinel.models.sonde_v2 import declared as D  # noqa: E402

CV = paths.RUNS_ROOT / "sonde" / "v2_cv"
FOLDS = CV / "FOLDS.json"
SEED = D.SEED
DEVICE = "cuda"                                    # the networks and Chronos-2 (tests: "cpu")
HORIZONS = D.HORIZONS
LEADS = [h - 1 for h in HORIZONS]
GPU_PREFER = 0                                     # first choice of the lock helper (it takes any idle allowed GPU)
STEP1_BOOSTERS = paths.RUNS_ROOT / "sonde" / "v2_step1" / "gbm_boosters"


def log(m):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def load_data():
    from sentinel.models.sonde_v2.prep import prepare_v2
    return prepare_v2(blocks=("train", "val"), roles=("train", "val"), log=log)


def net_variables(data) -> list:
    from sentinel.models.sonde_v2.prep import NOT_MODELLED
    return [v for v in data.cube.variables if v not in NOT_MODELLED]


def tag_of(a) -> str:
    return f"{a.arm}_{a.config}" + (f"_{a.target}" if (a.target and a.arm == "rnh") else "") + f"_s{a.seed}"


def phase_declare() -> int:
    from sentinel.models.sonde_v2 import parity
    parity.declare(log=log)
    return 0


def phase_folds() -> int:
    from sentinel.core.registry import log_result
    from sentinel.models.sonde import common
    from sentinel.models.sonde_v2 import parity
    from sentinel.models.sonde_v2.data_v2 import basin_folds

    parity.require_stamp("CV folds")
    if FOLDS.exists():
        raise SystemExit(f"{FOLDS} exists (folds are fixed once)")
    split, _ = common.load_split(paths.REPO_ROOT / "docs" / "preregistration" / "sonde_split.json")
    sites = np.array(sorted(s for r in ("train", "val") for s in split.sites(r)))
    basin = np.array([str(split.basin_group.get(s, s)) for s in sites])
    f = basin_folds(np.arange(sites.size), basin, D.N_FOLDS, SEED)
    rec = {"seed": SEED, "rule": "5-fold basin-group split of train+val-role split sites (data_v2.basin_folds)",
           "folds": {str(k): sites[f == k].tolist() for k in range(D.N_FOLDS)}}
    CV.mkdir(parents=True, exist_ok=True)
    FOLDS.write_text(json.dumps(rec, indent=1))
    sha = hashlib.sha256(FOLDS.read_bytes()).hexdigest()
    log_result(workstream="M1-sonde", experiment="sonde_arch_v2/cv_folds", model="folds", metric="n_sites",
               value=float(sites.size), split="cv", notes=f"CV folds fixed before any CV run; sha {sha}",
               extra={"sha256": sha, "per_fold": {k: len(v) for k, v in rec["folds"].items()}})
    log(f"folds written: {FOLDS} sha {sha[:12]}")
    return 0


def fold_sites(data, k: int):
    """(fit, early-stopping, held-out) site indices of fold k; the same for every arm."""
    rec = json.loads(FOLDS.read_text())
    pos = {s: i for i, s in enumerate(data.sites)}
    held = np.array(sorted(pos[s] for s in rec["folds"][str(k)] if s in pos), dtype=np.int64)
    inner = np.array(sorted(pos[s] for kk, v in rec["folds"].items() if int(kk) != k for s in v if s in pos),
                     dtype=np.int64)
    basin = np.array([str(data.split.basin_group.get(s, s)) for s in data.sites])
    g = np.unique(basin[inner])
    es_g = set(np.random.default_rng(SEED + k).choice(g, max(1, int(round(D.ES_FRACTION * g.size))), replace=False))
    es = inner[np.isin(basin[inner], list(es_g))]
    fit = inner[~np.isin(basin[inner], list(es_g))]
    return fit, es, held


def held_rows(data, v: str, held: np.ndarray):
    from sentinel.models.sonde_v2.prep import subset_rows
    rows = data.rows(v, "val")
    keep = np.isin(rows.si, held)
    return subset_rows(rows, keep) if keep.any() else None


def eval_slot_share(x: np.ndarray, si: np.ndarray, ti: np.ndarray, what: str) -> float:
    """Share of rows whose lead-1 temperature of the archived product is finite; hard error below the declared
    coverage threshold (an empty or thin product slot must never pass silently)."""
    from sentinel.models.sonde_v2 import nwp as NWP
    from sentinel.models.sonde_v2 import products as P
    share = float(np.isfinite(x[si, ti, 0, NWP.F_IDX["tmean_c"]]).mean()) if si.size else 1.0
    if share < P.COVERAGE_MIN:
        raise RuntimeError(f"{what}: only {share:.3f} of the held-out val rows have an archived {P.PRODUCT} "
                           f"forecast (< {P.COVERAGE_MIN}); refusing to write forecasts")
    return share


def fit_window(dates, ti: np.ndarray, h: int) -> dict:
    if ti.size == 0:
        return {}
    return {"block": "train", "issue_min": str(dates[int(ti.min())].date()),
            "issue_max": str(dates[int(ti.max())].date()),
            "target_max": str((dates[int(ti.max())] + pd.Timedelta(days=h)).date()), "rule": D.ROW_RULE}


# ----------------------------------------------------------------------------------------------- networks

def run_net(a, data, prod, out_dir, t0) -> int:
    import torch

    from sentinel.models.sonde import common
    from sentinel.models.sonde import data as sdata
    from sentinel.models.sonde_v2 import parity
    from sentinel.models.sonde_v2 import train_v2 as TV
    from sentinel.models.sonde_v2.data_v2 import V2Tensors
    from sentinel.models.sonde_v2.prep import do_pctsat_from_quantiles

    arm, cname, k, seed = a.arm, a.config, a.fold, a.seed
    target = a.target if arm == "rnh" else None
    if arm == "rnh" and target not in D.SCORED:
        raise SystemExit("R-NH: --target must be one of the scored variables")
    if arm == "sonde_hf" and a.target:
        raise SystemExit("SONDE-HF is one multi-task model: no --target")
    sreg = pd.read_parquet(paths.PROCESSED / "sites" / "sites.parquet").set_index("site_key")["utc_offset_h"]
    utc = {s: float(sreg.get(f"USGS-{s}", -6.0)) for s in data.sites}
    cfg = TV.TrainConfig(kind="rnh" if arm == "rnh" else "sonde_hf", seed=seed, rnh_target=target,
                         **D.train_overrides(arm, cname))
    fit, es, held = fold_sites(data, k)
    ten = V2Tensors(data, variables=net_variables(data), L=D.L, use_weather=True,
                    hourly=(arm == "sonde_hf" and cfg.hourly), utc_offset=utc, stat_sites=fit)
    ten.set_products("train", prod.pp, stat_sites=fit)       # stage 1: perfect-prog
    ten.set_products("real", prod.gefs)                      # stage 2: archived forecasts of the train block
    ten.set_products("eval", prod.gefs)                      # OOF: archived issued forecasts
    lt = TV.loss_targets(cfg, ten)
    tsi, tti = ten.train_rows(fit, "train", lt)
    esi, eti = ten.train_rows(es, "train", lt)
    if tsi.size == 0 or esi.size == 0:
        raise RuntimeError(f"no labelled rows ({tsi.size} fit, {esi.size} early stopping)")
    cfg2 = dataclasses.replace(cfg, lr=cfg.lr / D.STAGE2["lr_divisor"], max_epochs=D.STAGE2["max_epochs"],
                               patience=D.STAGE2["patience"])
    run = f"cv_{tag_of(a)}_f{k}"
    parity.record(
        run, data=data, use="cv", arm=arm, config=cname, fold=k, fit_sites=fit, es_sites=es, held_sites=held,
        n_rows=int(tsi.size),
        feature_columns=[f"dyn_channels:{ten.st.n_dyn}"] + list(ten.st.static_stats) + ["huc2_embedding"]
        + [f"nwp_{D.PRODUCT}:{ten.products['train'].shape[-1]}x2", "fdet:clearsky_sw,day_length,doy_sin,doy_cos",
           "lead_onehot", "static_embedding"]
        + ([f"hourly:{ten.hourly.shape[-1]}"] if (ten.hourly is not None and cfg.hourly) else []),
        input_sha256={"pp": prod.info["pp_sha256"], "gefs_bc": prod.info["gefs_bc_sha256"]},
        fit_window=fit_window(data.cube.dates, tti, D.L),
        extra={"seed": seed, "target": target, "loss_targets": lt, "n_es_rows": int(esi.size),
               "es_max_samples": cfg.es_max_samples,
               "labelled_fit_rows_per_target": {v: int(ten.train_rows(fit, "train", [v])[0].size) for v in lt},
               "train_config": dataclasses.asdict(cfg),
               "optimizer": {"stage1": {"name": "AdamW", "lr": cfg.lr, "weight_decay": cfg.weight_decay},
                             "stage2": {"name": "AdamW", "lr": cfg2.lr, "weight_decay": cfg2.weight_decay,
                                        "max_epochs": cfg2.max_epochs, "patience": cfg2.patience}},
               "nwp_stat_n": ten.nwp_stat_n, "tensor_variables": list(ten.variables),
               "cross_site_constants": "fit sites"})
    ck1, ck2 = out_dir / "stage1", out_dir / "stage2"
    budget = lambda: 95 - (time.time() - t0) / 60  # noqa: E731
    if not (ck1 / "DONE").exists():
        r = TV.train(cfg, ten, tsi, tti, esi, eti, ck1, device=DEVICE, max_minutes=budget(), log=log, use="train")
        if not r["complete"]:
            return 75
        (ck1 / "DONE").write_text(json.dumps({"epochs": r["epoch"], "best": r["best"],
                                              "optimizer": r["optimizer"]}))
    if not (ck2 / "DONE").exists():
        if budget() < 10:
            return 75
        r = TV.train(cfg2, ten, tsi, tti, esi, eti, ck2, device=DEVICE, max_minutes=budget(), log=log,
                     init=ck1 / "best.pt", use="real")
        if not r["complete"]:
            return 75
        (ck2 / "DONE").write_text(json.dumps({"epochs": r["epoch"], "best": r["best"],
                                              "optimizer": r["optimizer"]}))
    if budget() < 25:
        return 75                                                 # predict in a fresh process (<= 2 h rule)
    model = TV.build_model(cfg, ten).to(DEVICE)
    model.load_state_dict(torch.load(ck2 / "best.pt", map_location=DEVICE))
    out = out_dir / "forecasts"
    name = f"{arm}_{cname}"
    if target:
        variables = [target]
    elif not cfg.multitask:
        variables = [v for v in D.PRIMARY if v in ten.variables]            # C3: trained targets only
    else:
        variables = [v for v in D.SCORED + D.AUX_REPORTED if v in ten.variables]
    shares = {}
    for v in variables:
        r = held_rows(data, v, held)
        if r is None:
            continue
        si, ti = r.si.astype(np.int64), r.ti.astype(np.int64)
        shares[v] = eval_slot_share(prod.gefs, si, ti, f"{name} {v}")
        q = TV.predict(model, cfg, ten, si, ti, "eval", DEVICE, leads=LEADS,
                       var_idx=None if target else [ten.variables.index(v)])[:, :, 0]
        common.save_forecast(out / f"{name}__{v}__val.npz", model=name, rows=r, q=q, horizons=HORIZONS,
                             meta={"fold": k, "fold_sites": int(held.size), "oof": True,
                                   "nwp_lead1_share": shares[v]})
    no_held = [v for v in variables if v not in shares]
    if arm == "sonde_hf" and cfg.multitask and "do_pctsat" in data.cube.variables:
        # §2d DO % saturation: deterministic transform of the DO-mean quantiles at the T-mean median (ruling H)
        assert not ({"do_mgl_mean", "temp_c_mean", "do_pctsat"} & set(sdata.LOG_VARS))
        r = held_rows(data, "do_pctsat", held)
        if r is None:
            no_held.append("do_pctsat")
        else:
            si, ti = r.si.astype(np.int64), r.ti.astype(np.int64)
            q2 = TV.predict(model, cfg, ten, si, ti, "eval", DEVICE, leads=LEADS,
                            var_idx=[ten.variables.index("do_mgl_mean"), ten.variables.index("temp_c_mean")])
            elev = data.static["elev_m"].to_numpy(float)[si]
            q = np.stack([do_pctsat_from_quantiles(q2[:, j, 0], q2[:, j, 1, 9], elev) for j in range(len(LEADS))], 1)
            common.save_forecast(out / f"{name}__do_pctsat__val.npz", model=name, rows=r, q=q, horizons=HORIZONS,
                                 meta={"fold": k, "oof": True, "transformed": D.TRANSFORMED["do_pctsat"]})
    (out_dir / "OOF_DONE").write_text(json.dumps({"variables": variables, "nwp_lead1_share": shares,
                                                  "no_held_rows": no_held}))
    return 0


# ----------------------------------------------------------------------------------------------- GBM references

def run_gbm(a, data, prod, out_dir, t0) -> int:
    """One GBM process of a fold (GBM ruling G-3): finished boosters and finished variables are never refitted;
    the process returns 75 when it may not start the next booster and the same command continues it."""
    from sentinel.models.sonde import common
    from sentinel.models.sonde_v2 import gbm_resume as GR
    from sentinel.models.sonde_v2 import parity
    from sentinel.models.sonde_v2.gbm_nwp import SondeGBMNWP

    arm, cname, k = a.arm, a.config, a.fold
    cfg = D.gbm_config(arm, cname, seed=a.seed)
    proc = GR.GBMProcess.start("cv", f"cv_{tag_of(a)}_f{k}" + (f"_{a.target}" if a.target else ""), cfg, t0=t0,
                               prefer=GPU_PREFER, log=log)
    store_root = CV / "gbm_boosters" / f"{arm}_s{a.seed}" / f"fold{k}"
    try:
        rc = _gbm_variables(a, data, prod, out_dir, cfg, proc, store_root, common, parity, GR, SondeGBMNWP)
        proc.end = "complete" if rc == 0 else f"rc{rc}"
    except GR.ProcessBudgetReached as e:
        log(f"{e}; finished boosters are saved, run the same command again (exit {GR.EXIT_RESUME})")
        proc.end = "budget"
        return GR.EXIT_RESUME
    finally:
        proc.finish()
    return rc


def _gbm_variables(a, data, prod, out_dir, cfg, proc, store_root, common, parity, GR, SondeGBMNWP) -> int:
    arm, cname, k = a.arm, a.config, a.fold
    m = SondeGBMNWP(cfg, horizons=HORIZONS)
    fit, es, held = fold_sites(data, k)
    x = prod.gefs                                  # archived forecasts for fitting, early stopping and OOF
    name = f"{arm}_{cname}"
    scored = [v for v in D.SCORED if v in data.cube.variables]
    for v in scored:
        if a.target and v != a.target:
            continue
        f, dg = out_dir / "forecasts" / f"{name}__{v}__val.npz", out_dir / f"gbm_diag__{v}.json"
        if dg.exists():                            # written after the forecast file of the variable
            continue
        t1 = time.time()
        diag = m.fit_var_nwp(data, v, train_sites=fit, es_sites=es, nwp_fit=x, store_root=store_root, process=proc,
                             config=cname, log=log)
        r = held_rows(data, v, held)
        if r is not None:
            proc.require_room(f"OOF forecast {name} {v}")
            share = eval_slot_share(x, r.si.astype(np.int64), r.ti.astype(np.int64), f"{name} {v}")
            common.save_forecast(f, model=name, rows=r, q=m.predict_rows_nwp(data, v, r, x), horizons=HORIZONS,
                                 meta={"fold": k, "oof": True, "nwp_lead1_share": share})
        dg.write_text(json.dumps({"diag": diag, "feature_columns": m.feature_columns, "held_rows": r is not None,
                                  "minutes": round((time.time() - t1) / 60, 1)}, indent=1, default=str))
        for h in HORIZONS:                         # boosters are not needed after the OOF forecast
            for store in (m.qmodels, m.smodels, m.models):
                store.pop((v, h), None)
    diags = {v: json.loads((out_dir / f"gbm_diag__{v}.json").read_text()) for v in scored
             if (out_dir / f"gbm_diag__{v}.json").exists()}
    if len(diags) < len(scored):
        log(f"{len(diags)} of {len(scored)} variables done")
        return 0
    rows = {f"{v}|h{h}": d for v, dv in diags.items() for h, d in dv["diag"].items()}
    fw = {key: {kk: d.get(kk) for kk in ("fit_issue_min", "fit_issue_max", "fit_target_max")}
          for key, d in rows.items()}
    parity.record(f"cv_{tag_of(a)}_f{k}", data=data, use="cv", arm=arm, config=cname, fold=k, fit_sites=fit,
                  es_sites=es, held_sites=held, n_rows=sum(int(d.get("n_fit", 0)) for d in rows.values()),
                  feature_columns=next(iter(diags.values()))["feature_columns"],
                  input_sha256={"gefs_bc": prod.info["gefs_bc_sha256"]},
                  fit_window={"block": "train", "rule": D.ROW_RULE, "per_variable_horizon": fw},
                  extra={"seed": a.seed, "gbm_config": dataclasses.asdict(cfg),
                         "gbm_ruling": D.GBM_SETTINGS["ruling"], "device": cfg.device,
                         "device_settings": D.GBM_SETTINGS["device_settings"],
                         "gbm_processes": GR.read_processes(store_root), "store_root": str(store_root),
                         "rows_per_variable_horizon": {key: {kk: d.get(kk) for kk in (
                             "n_fit_total", "n_fit", "max_fit_rows", "n_es_total", "n_es", "es_max_rows",
                             "fit_rows_sha256", "es_rows_sha256", "n_residual_rows", "skipped")}
                             for key, d in rows.items()},
                         "boosters_per_variable_horizon": {key: d.get("boosters") for key, d in rows.items()},
                         "ls_residual_rows": "early-stopping rows" if cfg.quantile_mode == "location_scale" else None})
    (out_dir / "OOF_DONE").write_text(json.dumps({"variables": scored, "no_held_rows": [
        v for v in scored if not diags[v]["held_rows"]]}))
    return 0


# ----------------------------------------------------------------------------------------------- Chronos-2-cov

def run_chronos(a, data, prod, out_dir, t0) -> int:
    """Zero-shot Chronos-2 multi with the archived GEFS forecasts as known-future covariates on fold-k held-out
    sites × val rows. Resumable at batch-flush boundaries (stops past ~95 min, exit 75)."""
    from scripts.eval.sonde_predict import chronos_snapshot
    from sentinel.models.sonde import common
    from sentinel.models.sonde.chronos import ChronosConfig, load_pipeline
    from sentinel.models.sonde_v2 import parity
    from sentinel.models.sonde_v2.chronos_cov import COV_FIELDS, ChronosCovForecaster

    k = a.fold
    fit, es, held = fold_sites(data, k)
    pipe = load_pipeline(str(chronos_snapshot("amazon/chronos-2")), device=DEVICE)
    f = ChronosCovForecaster(pipe, ChronosConfig(**D.train_overrides("chronos_cov", a.config)), HORIZONS,
                             nwp_x=prod.gefs)
    rows, shares = {}, {}
    for v in [v for v in D.SCORED if v in data.cube.variables]:
        r = held_rows(data, v, held)
        if r is not None:
            rows[v] = r
            shares[v] = eval_slot_share(prod.gefs, r.si.astype(np.int64), r.ti.astype(np.int64), f"chronos {v}")
    parity.record(f"cv_{tag_of(a)}_f{k}", data=data, use="cv", arm=a.arm, config=a.config, fold=k,
                  fit_sites=np.zeros(0, np.int64), es_sites=np.zeros(0, np.int64), held_sites=held, n_rows=0,
                  feature_columns=[f"known_future:{c}" for c in COV_FIELDS],
                  input_sha256={"gefs_bc": prod.info["gefs_bc_sha256"]},
                  extra={"zero_shot": True, "n_oof_rows": {v: len(r) for v, r in rows.items()},
                         "group_variables": net_variables(data)})
    state_f = out_dir / "resume.npz"
    resume = None
    if state_f.exists():
        z = np.load(state_f)
        resume = {"out": {v: z[f"out__{v}"] for v in rows}, "next_site_pos": int(z["next_site_pos"])}
    fc, info = f.predict_multi_resumable(data, rows, resume=resume, deadline=t0 + 95 * 60)
    if not info["complete"]:
        np.savez(state_f, next_site_pos=info["next_site_pos"], **{f"out__{v}": q for v, q in fc.items()})
        return 75
    for v, q in fc.items():
        common.save_forecast(out_dir / "forecasts" / f"chronos_cov_{a.config}__{v}__val.npz",
                             model=f"chronos_cov_{a.config}", rows=rows[v], q=q, horizons=HORIZONS,
                             meta={"fold": k, "oof": True, "zero_shot": True, "nwp_lead1_share": shares[v]})
    state_f.unlink(missing_ok=True)
    scored = [v for v in D.SCORED if v in data.cube.variables]
    (out_dir / "OOF_DONE").write_text(json.dumps({"variables": scored, "nwp_lead1_share": shares,
                                                  "no_held_rows": [v for v in scored if v not in rows]}))
    return 0


# ----------------------------------------------------------------------------------------------- CV GPU-h budget

def phase_gbm_budget(a) -> int:
    """GBM ruling G-7(c): the CV GPU-hour projection, re-computed from the stopping rounds of the step-1 boosters
    and the fit rows of every CV unit, written to ``gbm_budget.json`` and to the registry. No fit; the engineer
    sends it to the lead before the first CV GBM process, and :func:`gbm_resume.cv_gate` reads it."""
    from sentinel.core.registry import log_result
    from sentinel.models.sonde import runner
    from sentinel.models.sonde_v2 import gbm_resume as GR
    from sentinel.models.sonde_v2 import parity
    from sentinel.models.sonde_v2.gbm_nwp import SondeGBMNWP

    stamp = parity.require_stamp("CV GBM budget")
    parity.require_declared("gbm_nwp", D.MANDATORY["gbm_nwp"])
    if not FOLDS.exists():
        raise SystemExit("run --phase folds first")
    metas = GR.read_metas(STEP1_BOOSTERS)
    if not metas:
        raise SystemExit(f"no step-1 booster record under {STEP1_BOOSTERS}: run step 1 --phase gbm first")
    out = CV / "gbm_budget.json"
    if out.exists():
        log(f"{out} exists (the projection is written once)")
        return 0
    runner.limit_cpu_threads(6)
    seed = a.seed
    cfg = D.gbm_config("gbm_nwp", D.MANDATORY["gbm_nwp"], seed=seed)
    data = load_data()
    m = SondeGBMNWP(cfg, horizons=HORIZONS)
    scored = [v for v in D.SCORED if v in data.cube.variables]
    rows = {}
    for k in range(D.N_FOLDS):
        fit, es, _ = fold_sites(data, k)
        for v in scored:
            for h in HORIZONS:
                si, _, _, _, _ = m.labelled_rows(data, v, h, fit, cap=None, salt=0)
                esi, _, _, _, _ = m.labelled_rows(data, v, h, es, cap=cfg.es_max_rows, salt=1)
                rows[f"{k}|{v}|{h}"] = {"n_fit": int(si.size), "n_es": int(esi.size),
                                        "skipped": bool(si.size < 1000 or esi.size < 200)}
        log(f"fold {k}: rows counted")
    proj = GR.cv_projection(metas, rows, {a: D.CONFIGS[a] for a in ("gbm_nwp", "pf_gbm")}, seed=seed)
    rec = {**proj, "rows": rows, "step1_boosters": STEP1_BOOSTERS.as_posix(), "n_step1_records": len(metas),
           "ruling": D.GBM_SETTINGS["ruling"], "stamp_row": stamp["row_id"],
           "declaration_sha256": parity.require_declared("gbm_nwp", D.MANDATORY["gbm_nwp"])}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, default=str))
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    log_result(workstream="M1-sonde", experiment="sonde_arch_v2/gbm_cv_budget", model="gbm_nwp+pf_gbm",
               metric="projected_cv_gpu_hours", value=float(proj["total_gpu_h"]), split="cv",
               notes="GBM ruling G-7(c): CV GPU-hours projected from the stopping rounds of the step-1 boosters "
                     "and the fit rows of every CV unit; compute projection only, no score",
               extra={"file": out.as_posix(), "sha256": sha, "limit_gpu_h": proj["limit_gpu_h"],
                      "per_arm_and_config_alone": proj["gpu_hours_per_arm_and_config_alone"],
                      "step1": proj["step1"], "stamp_row": stamp["row_id"]})
    log(f"{out} sha {sha[:12]}: {proj['total_gpu_h']:.1f} GPU-h projected, limit {proj['limit_gpu_h']:.0f}")
    return 0


# ----------------------------------------------------------------------------------------------- phases

def phase_run(a) -> int:
    from sentinel.core.registry import log_result
    from sentinel.models.sonde_v2 import parity
    from sentinel.models.sonde_v2 import products as P

    t0 = time.time()
    tag = tag_of(a)
    stamp = parity.require_stamp(f"CV run {tag} fold {a.fold}")
    dsha = parity.require_declared(a.arm, a.config)
    if a.seed not in D.SEEDS["cv"]:
        raise SystemExit(f"seed {a.seed} is not declared for CV ({D.SEEDS['cv']})")
    if a.arm in ("gbm_nwp", "pf_gbm"):
        from sentinel.models.sonde_v2 import gbm_resume as GR
        D.gbm_config(a.arm, a.config)                           # refuses while the GBM settings are not ruled
        log(f"GBM GPU-hour gate (G-7d): {GR.cv_gate(CV / 'gbm_budget.json', CV / 'gbm_boosters')}")
    if not FOLDS.exists():
        raise SystemExit("run --phase folds first")
    out_dir = CV / tag / f"fold{a.fold}"
    if (out_dir / "OOF_DONE").exists():
        log("already done")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.arm in ("sonde_hf", "rnh", "chronos_cov"):
        from sentinel.models.sonde import gpu_budget, runner
        gpu_budget.widen(10000)
        gpu = runner.setup_gpu(f"sonde_v2_cv_{tag}_f{a.fold}", prefer=0)
        log(f"GPU {gpu}")
    data = load_data()
    prod = P.build(data, use="cv", log=log)
    try:
        if a.arm in ("sonde_hf", "rnh"):
            rc = run_net(a, data, prod, out_dir, t0)
        elif a.arm == "chronos_cov":
            rc = run_chronos(a, data, prod, out_dir, t0)
        else:
            rc = run_gbm(a, data, prod, out_dir, t0)
    except BaseException as e:  # noqa: BLE001
        log_result(workstream="M1-sonde", experiment="sonde_arch_v2/cv", model=tag, metric="run_failed", value=1.0,
                   split=f"cv/fold{a.fold}", notes=repr(e)[:300])
        raise
    if rc == 0 and (out_dir / "OOF_DONE").exists():
        log_result(workstream="M1-sonde", experiment="sonde_arch_v2/cv", model=tag, metric="oof_written", value=1.0,
                   split=f"cv/fold{a.fold}", notes="OOF forecasts written (no score computed here)",
                   extra={"minutes": round((time.time() - t0) / 60, 1), "products": prod.info,
                          "declaration_sha256": dsha, "stamp_row": stamp["row_id"]})
    return rc


def done_arms() -> list:
    """(arm, config) pairs with at least one fold directory under the CV root."""
    out = set()
    for arm, cfgs in D.CONFIGS.items():
        for c in cfgs:
            if any(CV.glob(f"{arm}_{c}_*/fold*")) or any(CV.glob(f"{arm}_{c}_s*/fold*")):
                out.add((arm, c))
    return sorted(out)


def phase_parity(a) -> int:
    """Cross-arm parity of every fold over every arm that has run (before any CV score)."""
    from sentinel.core.registry import log_result
    from sentinel.models.sonde_v2 import parity

    parity.require_stamp("CV parity check")
    arms = done_arms()
    res = [parity.check("cv", k, require_arms=arms) for k in range(D.N_FOLDS)]
    (CV / "PARITY_CHECK.json").write_text(json.dumps({"arms": arms, "folds": res}, indent=1))
    log_result(workstream="M1-sonde", experiment="sonde_arch_v2/parity_check", model="cv", metric="folds_passed",
               value=float(len(res)), split="cv", notes="cross-arm parity (sites, statics, input hashes) passed",
               extra={"arms": [f"{x}_{y}" for x, y in arms], "folds": res})
    log(f"parity passed: {len(arms)} arms x {len(res)} folds")
    return 0


def phase_combine(a) -> int:
    from sentinel.models.sonde import common
    from sentinel.models.sonde_v2 import parity

    parity.require_stamp("CV combine")
    parity.require_declared(a.arm, a.config)
    for k in range(D.N_FOLDS):
        parity.check("cv", k, require_arms=[(a.arm, a.config)])
    tag = tag_of(a)
    out = CV / tag / "oof" / "forecasts"
    out.mkdir(parents=True, exist_ok=True)
    files, empty = {}, {}
    for k in range(D.N_FOLDS):
        if not (CV / tag / f"fold{k}" / "OOF_DONE").exists():
            raise SystemExit(f"fold {k} is not complete")
        empty[k] = set(json.loads((CV / tag / f"fold{k}" / "OOF_DONE").read_text()).get("no_held_rows", []))
        for f in (CV / tag / f"fold{k}" / "forecasts").glob("*.npz"):
            files.setdefault(f.name, {})[k] = f
    for name, by_fold in files.items():
        var = name.split("__")[1]
        # a fold may lack a variable only when the fold has no held-out val row of it (recorded by the run)
        bad = [k for k in range(D.N_FOLDS) if k not in by_fold and var not in empty[k]]
        if bad:
            raise SystemExit(f"{name}: no forecast file in fold(s) {bad}")
        fs = [by_fold[k] for k in sorted(by_fold)]
        parts = [common.load_forecast(f) for f in fs]
        z = {k: np.concatenate([p[k] for p in parts]) for k in ("q", "site_no", "issue_date")}
        np.savez_compressed(out / name, q=z["q"], site_no=z["site_no"], issue_date=z["issue_date"],
                            horizons=parts[0]["horizons"], quantiles=parts[0]["quantiles"],
                            header=json.dumps({**parts[0]["header"], "oof_folds": D.N_FOLDS,
                                               "folds_with_rows": sorted(by_fold)}))
    log(f"combined {len(files)} variables into {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("declare", "folds", "gbm_budget", "run", "parity", "combine"),
                    required=True)
    ap.add_argument("--arm", choices=tuple(D.CONFIGS))
    ap.add_argument("--config")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", default=None, help="R-NH: one model per target; GBM arms: fit this variable only")
    a = ap.parse_args()
    if a.phase == "declare":
        return phase_declare()
    if a.phase == "folds":
        return phase_folds()
    if a.phase == "parity":
        return phase_parity(a)
    if a.phase == "gbm_budget":
        return phase_gbm_budget(a)
    if a.phase == "run":
        return phase_run(a)
    return phase_combine(a)


if __name__ == "__main__":
    raise SystemExit(main())
