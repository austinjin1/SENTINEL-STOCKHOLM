#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Step-1 information-use test (``sonde_arch_v2_amendment.md`` §2f; consolidated deviations file
``sonde_arch_v2_deviations_v2.md``).

TRAIN-ROLE sites × TRAIN block only: ``prepare_v2(blocks=("train",), roles=("train",))`` — the val block is not
authorised and no val-role site is loaded.
* Guard: every phase first calls ``parity.require_stamp`` and ``parity.require_declared`` (declaration record:
  ``--phase declare`` or ``sonde_v2_cv.py --phase declare``).
* Inner split: 5-fold basin-group split of the train-role sites (seed 20261110); fold 0 is held out. Early stopping:
  15 % of the basin groups of folds 1–4 (seed 20261110 + 0), × train block — the same fit and early-stopping site
  sets for both models.
* Inputs: perfect-prog = observed gridMET of the target days + the stage-1 error model fitted on the bias-corrected
  GEFS reforecast c00 (WY2008–2019, issue days whose 7 valid days end by 2019-09-30). Every reforecast month
  2007-10 … 2019-12 must be on disk (hard error otherwise). The SAME perfect-prog array (fixed seed; sha256
  written by both phases, equality asserted by ``eval``) is used for training and held-out evaluation by both models.
* Contrast: SONDE-HF C1 (1 seed) vs GBM-NWP Q19 (the mandatory declared configuration; lead ruling B) with the
  ruled GBM settings.
* Statistic: for T mean and DO mean, the median over held-out sites (≥ 30 scored rows) of the per-site CRPS skill
  1 − CRPS_HF / CRPS_GBM, averaged over h ∈ {3, 5, 7}, on fold-0 sites × train-block issue dates from 2015-10-01
  that pass the reference-window issue rule. CRPS = 2 × mean pinball over the 19 levels, model space.
* Rule: if the statistic is < 0.01 for BOTH T and DO, PhiDA-Lite and Chronos-2-NWP are not built.

Phases (each its own process; ``hf`` is a ≤ 2 h GPU process that exits 75 to be resumed):
    python -B scripts/train/sonde_v2_step1.py --phase declare
    python -B scripts/train/sonde_v2_step1.py --phase hf      # SONDE-HF C1 train + predict held-out rows
    python -B scripts/train/sonde_v2_step1.py --phase gbm     # GBM-NWP Q19 + predict (GPU; exits 75 to be resumed)
    python -B scripts/train/sonde_v2_step1.py --phase eval    # statistic, decision, registry rows
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sentinel.core import paths  # noqa: E402
from sentinel.models.sonde_v2 import declared as D  # noqa: E402

RUN = paths.RUNS_ROOT / "sonde" / "v2_step1"
S1 = D.STEP1
SEED = D.SEED
EVAL_FROM = pd.Timestamp(S1["eval_from"])
VARS_DECISION = tuple(S1["variables"])
H_DECISION = tuple(S1["horizons"])
FOLD = int(S1["held_fold"])
CAND, COMP = tuple(S1["candidate"]), tuple(S1["comparator"])


DEVICE = "cuda"                                    # SONDE-HF (tests: "cpu")
GPU_PREFER = 0                                     # first choice of the lock helper; it takes any idle allowed GPU


def log(m):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def guard(what: str) -> dict:
    from sentinel.models.sonde_v2 import parity
    stamp = parity.require_stamp(f"step 1 {what}")
    for arm, cname in (CAND, COMP):
        parity.require_declared(arm, cname)
    D.gbm_config(*COMP)                       # refuses while the GBM settings are not ruled (comparator undefined)
    return stamp


def site_sets(data):
    """(fit, early-stopping, held-out) site indices: the same for both models."""
    from sentinel.models.sonde_v2.data_v2 import basin_folds
    tr = data.site_idx(["train"])
    basin = np.array([str(data.split.basin_group.get(s, s)) for s in data.sites])
    folds = basin_folds(tr, basin, D.N_FOLDS, SEED)
    held, inner = tr[folds == FOLD], tr[folds != FOLD]
    g = np.unique(basin[inner])
    es_g = set(np.random.default_rng(SEED + FOLD).choice(g, max(1, int(round(D.ES_FRACTION * g.size))),
                                                         replace=False))
    es = inner[np.isin(basin[inner], list(es_g))]
    fit = inner[~np.isin(basin[inner], list(es_g))]
    return fit, es, held


def setup():
    from sentinel.models.sonde_v2 import products as P
    from sentinel.models.sonde_v2.prep import prepare_v2

    data = prepare_v2(blocks=("train",), roles=("train",), log=log)
    fit, es, held = site_sets(data)
    prod = P.build(data, use="step1", log=log)
    info = {"n_train_role": int(data.site_idx(["train"]).size), "n_fit_sites": int(fit.size),
            "n_es_sites": int(es.size), "n_heldout_sites": int(held.size), **prod.info}
    return data, prod, fit, es, held, info


def eval_rows(data, held: np.ndarray):
    """Held-out rows (site, issue day ≥ 2015-10-01, every lead inside the train block) with ≥ 1 finite target of a
    decision variable at a decision horizon on an issue day allowed by the reference-window rule. Built from the
    cube only, so both phases get the same rows."""
    from sentinel.models.sonde.data import to_model_space
    dates = data.cube.dates
    t_idx = data.issue_idx("train", D.L)
    ev_t = t_idx[dates[t_idx] >= EVAL_FROM]
    si, ti = np.repeat(held, ev_t.size), np.tile(ev_t, held.size)
    has = np.zeros(si.size, bool)
    for v in VARS_DECISION:
        z = to_model_space(v, data.cube.y[:, :, data.cube.var_index(v)])
        ok = data.issue_ok(v, si, ti)
        for h in H_DECISION:
            has |= ok & np.isfinite(z[si, ti + h])
    return si[has].astype(np.int64), ti[has].astype(np.int64)


class _Rows:
    def __init__(self, si, ti):
        self.si, self.ti = si, ti

    def __len__(self):
        return int(self.si.size)


def phase_declare() -> int:
    from sentinel.models.sonde_v2 import parity
    parity.declare(log=log)
    return 0


def phase_hf() -> int:
    import torch

    from sentinel.models.sonde import gpu_budget, runner
    from sentinel.models.sonde_v2 import parity
    from sentinel.models.sonde_v2 import train_v2 as TV
    from sentinel.models.sonde_v2.data_v2 import V2Tensors
    from sentinel.models.sonde_v2.prep import NOT_MODELLED

    guard("hf")
    gpu_budget.widen(10000)                          # GPUs 0-9 (cap file), idle GPUs only
    gpu = runner.setup_gpu("sonde_v2_step1_hf", prefer=0)
    log(f"[hf] GPU {gpu}")
    t0 = time.time()
    data, prod, fit, es, held, info = setup()
    sreg = pd.read_parquet(paths.PROCESSED / "sites" / "sites.parquet").set_index("site_key")["utc_offset_h"]
    utc = {s_: float(sreg.get(f"USGS-{s_}", -6.0)) for s_ in data.sites}
    cfg = TV.TrainConfig(kind="sonde_hf", seed=D.SEEDS["step1"][0], **D.train_overrides(*CAND))
    ten = V2Tensors(data, variables=[v for v in data.cube.variables if v not in NOT_MODELLED], L=D.L,
                    use_weather=True, hourly=cfg.hourly, utc_offset=utc, stat_sites=fit)
    ten.set_products("train", prod.pp, stat_sites=fit)
    lt = TV.loss_targets(cfg, ten)
    tsi, tti = ten.train_rows(fit, "train", lt)
    esi, eti = ten.train_rows(es, "train", lt)
    dates = data.cube.dates
    ck = RUN / "hf_c1"
    ck.mkdir(parents=True, exist_ok=True)
    parity.record(
        "step1_sonde_hf_C1", data=data, use="step1", arm=CAND[0], config=CAND[1], fold=FOLD, fit_sites=fit,
        es_sites=es, held_sites=held, n_rows=int(tsi.size),
        feature_columns=[f"dyn_channels:{ten.st.n_dyn}"] + list(ten.st.static_stats) + ["huc2_embedding"]
        + [f"nwp_{D.PRODUCT}:{ten.products['train'].shape[-1]}x2", "fdet:clearsky_sw,day_length,doy_sin,doy_cos",
           "lead_onehot", "static_embedding"]
        + ([f"hourly:{ten.hourly.shape[-1]}"] if (ten.hourly is not None and cfg.hourly) else []),
        input_sha256={"pp": prod.info["pp_sha256"]},
        fit_window={"block": "train", "issue_min": str(dates[int(tti.min())].date()),
                    "issue_max": str(dates[int(tti.max())].date()),
                    "target_max": str((dates[int(tti.max())] + pd.Timedelta(days=D.L)).date()), "rule": D.ROW_RULE},
        extra={"seed": cfg.seed, "loss_targets": lt, "n_es_rows": int(esi.size),
               "es_max_samples": cfg.es_max_samples, "train_config": dataclasses.asdict(cfg),
               "optimizer": {"stage1": {"name": "AdamW", "lr": cfg.lr, "weight_decay": cfg.weight_decay}},
               "nwp_stat_n": ten.nwp_stat_n, "tensor_variables": list(ten.variables),
               "cross_site_constants": "fit sites", "stage2": None})
    if not (ck / "DONE").exists():
        res = TV.train(cfg, ten, tsi, tti, esi, eti, ck, device=DEVICE, max_minutes=95 - (time.time() - t0) / 60,
                       log=log, use="train")
        (ck / "info.json").write_text(json.dumps({**info, "train": {k: v for k, v in res.items()
                                                                     if k != "history"}}, indent=1, default=str))
        if not res["complete"]:
            log("[hf] partial: exit 75 (re-run to resume)")
            return 75
        (ck / "DONE").write_text(json.dumps({"epochs": res["epoch"], "best": res["best"],
                                             "optimizer": res["optimizer"]}))
        if (time.time() - t0) / 60 > 75:
            log("[hf] trained; prediction in the next process (<= 2 h rule): exit 75")
            return 75
    model = TV.build_model(cfg, ten).to(DEVICE)
    model.load_state_dict(torch.load(ck / "best.pt", map_location=DEVICE))
    si, ti = eval_rows(data, held)
    q = TV.predict(model, cfg, ten, si, ti, "train", DEVICE, leads=[h - 1 for h in H_DECISION],
                   var_idx=[ten.variables.index(v) for v in VARS_DECISION])       # (n, 3, 2, 19) model space
    np.savez_compressed(RUN / "pred_hf.npz", si=si, ti=ti, q=q, pp_sha256=prod.info["pp_sha256"],
                        sites=np.asarray(data.sites)[si].astype(str))
    res = json.loads((ck / "DONE").read_text())
    log(f"[hf] done: {res['epochs']} epochs, best {res['best']:.5f}; {si.size} held-out rows predicted")
    return 0


def phase_gbm() -> int:
    """GBM-NWP Q19 on the step-1 rows. Booster-level resume (GBM ruling G-3): one GPU process of at most 2 h; the
    process ends with exit code 75 when it may not start the next booster, and the same command continues it."""
    from sentinel.models.sonde_v2 import gbm_resume as GR
    from sentinel.models.sonde_v2 import parity
    from sentinel.models.sonde_v2.gbm_nwp import SondeGBMNWP

    t0 = time.time()
    guard("gbm")
    cfg = D.gbm_config(*COMP)
    if (RUN / "pred_gbm.npz").exists():
        log("[gbm] pred_gbm.npz exists; nothing to do")
        return 0
    data, prod, fit, es, held, info = setup()
    proc = GR.GBMProcess.start("step1", f"step1_{COMP[0]}_{COMP[1]}", cfg, t0=t0, prefer=GPU_PREFER, log=log)
    store = RUN / "gbm_boosters" / f"{COMP[0]}_s{cfg.seed}"
    try:
        m = SondeGBMNWP(cfg, horizons=H_DECISION)
        si, ti = eval_rows(data, held)
        qs, diag = [], {}
        for v in VARS_DECISION:
            diag[v] = m.fit_var_nwp(data, v, train_sites=fit, es_sites=es, nwp_fit=prod.pp, store_root=store,
                                    process=proc, config=COMP[1], log=log)
        proc.require_room("step-1 GBM forecast")
        for v in VARS_DECISION:
            qs.append(m.predict_rows_nwp(data, v, _Rows(si, ti), prod.pp))        # (n, H, 19) model space
        rows = {f"{v}|h{h}": d for v, dv in diag.items() for h, d in dv.items()}
        parity.record("step1_gbm_nwp_Q19", data=data, use="step1", arm=COMP[0], config=COMP[1], fold=FOLD,
                      fit_sites=fit, es_sites=es, held_sites=held,
                      n_rows=sum(int(d.get("n_fit", 0)) for d in rows.values()), feature_columns=m.feature_columns,
                      input_sha256={"pp": prod.info["pp_sha256"]},
                      fit_window={"block": "train", "rule": D.ROW_RULE, "per_variable_horizon": {
                          key: {kk: d.get(kk) for kk in ("fit_issue_min", "fit_issue_max", "fit_target_max")}
                          for key, d in rows.items()}},
                      extra={"seed": cfg.seed, "gbm_config": dataclasses.asdict(cfg),
                             "gbm_ruling": D.GBM_SETTINGS["ruling"], "device": cfg.device,
                             "device_settings": D.GBM_SETTINGS["device_settings"],
                             "gbm_processes": GR.read_processes(store), "store_root": str(store),
                             "rows_per_variable_horizon": {key: {kk: d.get(kk) for kk in (
                                 "n_fit_total", "n_fit", "max_fit_rows", "n_es_total", "n_es", "es_max_rows",
                                 "fit_rows_sha256", "es_rows_sha256", "skipped")} for key, d in rows.items()},
                             "boosters_per_variable_horizon": {key: d.get("boosters") for key, d in rows.items()}})
        np.savez_compressed(RUN / "pred_gbm.npz", si=si, ti=ti, q=np.stack(qs, 2),
                            pp_sha256=prod.info["pp_sha256"], sites=np.asarray(data.sites)[si].astype(str))
        (RUN / "gbm_diag.json").write_text(json.dumps({**info, "diag": diag}, indent=1, default=str))
        proc.end = "complete"
    except GR.ProcessBudgetReached as e:
        log(f"[gbm] {e}; finished boosters are saved, run the same command again (exit {GR.EXIT_RESUME})")
        proc.end = "budget"
        return GR.EXIT_RESUME
    finally:
        proc.finish()
    log("[gbm] done")
    return 0


def crps_pinball(y: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per-row CRPS approximation: 2 × mean pinball loss over the 19 levels 0.05 … 0.95."""
    taus = np.round(np.arange(0.05, 0.951, 0.05), 2)
    d = y[:, None] - np.sort(q, axis=1)
    return 2 * np.mean(np.maximum(taus * d, (taus - 1) * d), axis=1)


def step1_statistic(si, y_by_h: dict, ok_by_h: dict, q_hf: np.ndarray, q_gbm: np.ndarray, min_rows: int) -> dict:
    """One variable. ``q_*``: (n, H, 19); ``y_by_h`` / ``ok_by_h``: per horizon position. Median over sites (≥
    ``min_rows`` scored rows) of 1 − CRPS_HF / CRPS_GBM per horizon, then the mean over the horizons."""
    out, sk = {}, []
    for k in sorted(y_by_h):
        ok = ok_by_h[k] & np.isfinite(q_hf[:, k]).all(1) & np.isfinite(q_gbm[:, k]).all(1)
        n_nan = int((ok_by_h[k] & ~ok).sum())
        df = pd.DataFrame({"s": si[ok], "hf": crps_pinball(y_by_h[k][ok], q_hf[ok, k]),
                           "gbm": crps_pinball(y_by_h[k][ok], q_gbm[ok, k])})
        g = df.groupby("s").agg(n=("hf", "size"), hf=("hf", "mean"), gbm=("gbm", "mean"))
        g = g[g["n"] >= min_rows]
        skill = 1 - g["hf"] / g["gbm"]
        out[k] = {"median_site_skill": float(skill.median()), "n_sites": int(len(g)), "n_rows": int(ok.sum()),
                  "n_rows_without_forecast": n_nan}
        sk.append(float(skill.median()))
    out["mean"] = float(np.mean(sk))
    return out


def phase_eval() -> int:
    from sentinel.core.registry import log_result
    from sentinel.models.sonde.data import to_model_space
    from sentinel.models.sonde_v2 import parity
    from sentinel.models.sonde_v2.prep import prepare_v2

    stamp = guard("eval")
    a, b = np.load(RUN / "pred_hf.npz"), np.load(RUN / "pred_gbm.npz")
    if str(a["pp_sha256"]) != str(b["pp_sha256"]):
        raise RuntimeError("the perfect-prog input of the HF phase is not that of the GBM phase (sha256 differ)")
    if not (np.array_equal(a["si"], b["si"]) and np.array_equal(a["ti"], b["ti"])
            and np.array_equal(a["sites"], b["sites"])):
        raise RuntimeError("the two phases predicted different rows")
    par = parity.check("step1", FOLD, require_arms=[CAND, COMP])
    if par["input_sha256"].get("pp") != str(a["pp_sha256"]):
        raise RuntimeError("the prediction files do not carry the perfect-prog hash of the parity records")
    data = prepare_v2(blocks=("train",), roles=("train",), use_weather=False, use_catchment=False, log=log)
    si, ti = a["si"], a["ti"]
    if not np.array_equal(np.asarray(data.sites)[si].astype(str), a["sites"]):
        raise RuntimeError("site axis differs from the prediction phases")
    res = {}
    for j, v in enumerate(VARS_DECISION):
        z = to_model_space(v, data.cube.y[:, :, data.cube.var_index(v)])
        iok = data.issue_ok(v, si, ti)
        y = {k: z[si, ti + h] for k, h in enumerate(H_DECISION)}
        ok = {k: iok & np.isfinite(y[k]) for k in y}
        st = step1_statistic(si, y, ok, a["q"][:, :, j], b["q"][:, :, j], int(S1["min_rows_per_site"]))
        for k, h in enumerate(H_DECISION):
            res[f"{v}_h{h}"] = st[k]
        res[v] = st["mean"]
    go = any(res[v] >= float(S1["threshold"]) for v in VARS_DECISION)
    res["decision"] = "BUILD PhiDA-Lite (C3 gate) and Chronos-2-NWP" if go else \
        "DO NOT BUILD PhiDA-Lite or Chronos-2-NWP; GPU to M1-H"
    res["parity"] = par
    res["stamp_row"] = stamp["row_id"]
    (RUN / "STEP1_RESULT.json").write_text(json.dumps(res, indent=1))
    for v in VARS_DECISION:
        log_result(workstream="M1-sonde", experiment="sonde_arch_v2/step1", model="sonde_hf_C1_vs_gbm_nwp_Q19",
                   metric="median_site_crps_skill_mean_h357", value=res[v], split="train/inner_cv_fold0",
                   target=v, notes="step-1 information-use test (amendment §2f), train-role x train block only; "
                                   "perfect-prog + GEFS-reforecast WY2008-2019 error model; rule: < 0.01 for both "
                                   "T and DO -> skip",
                   extra={"per_h": {k: r for k, r in res.items() if k.startswith(v + "_h")},
                          "decision": res["decision"], "pp_sha256": str(a["pp_sha256"]),
                          "stamp_row": stamp["row_id"]})
    log(json.dumps(res, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("declare", "hf", "gbm", "eval"), required=True)
    a = ap.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    return {"declare": phase_declare, "hf": phase_hf, "gbm": phase_gbm, "eval": phase_eval}[a.phase]()


if __name__ == "__main__":
    raise SystemExit(main())
