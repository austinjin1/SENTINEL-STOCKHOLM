# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Train one SENTINEL-Sim ensemble member (validation-only development; protocol §3).

    python -m sentinel.core.gpu --name sim_s0 --prefer 4 -- \
        python -B scripts/train/sim_train.py --run-name base_s0 --seed 0

Writes $SENTINEL_RUNS/sim/<run>/{model.pt, history.json, sim_eval.npz} (simulations for cal + val
sites from WY2008 on) and a TrainingRun entry. ``--dsst`` trains on the coolest 70% of train-block
water years only (train-cool / test-hot, protocol §5.4) and also saves cal-site simulations.
``--smoke N`` uses the first N candidate sites, is never registered and writes under sim/smoke/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch

from sentinel.core import paths
from sentinel.core.runlog import TrainingRun
from sentinel.models.sim import data as D
from sentinel.models.sim import runner
from sentinel.models.sim import train as TR


def cool_years(d: D.SimData, frac: float = 0.7) -> tuple[list[int], list[int], dict]:
    """Rank train-block water years by the mean JJA gridMET air temperature over train-role sites."""
    tr = d.idx("train")
    from sentinel.models.sim.dsst import water_year

    wy = water_year(d.dates)
    tm = D.block_mask(d.dates, "train")
    jja = np.isin(d.dates.month, (6, 7, 8))
    ta = 0.5 * (d.forcing[tr][..., 0] + d.forcing[tr][..., 1])
    years = sorted(set(wy[tm]))
    score = {int(y): float(np.nanmean(ta[:, (wy == y) & jja & tm])) for y in years}
    order = sorted(years, key=lambda y: score[y])
    n_cool = int(round(frac * len(order)))
    return sorted(int(y) for y in order[:n_cool]), sorted(int(y) for y in order[n_cool:]), score


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value (json values)")
    ap.add_argument("--dsst", action="store_true", help="train on the coolest 70%% of train-block water years")
    ap.add_argument("--dsst-control", action="store_true",
                    help="train on 10 randomly chosen train-block water years (seed 0) = DSST control")
    ap.add_argument("--exclude-huc2", default="",
                    help="comma list: train/cal-role sites in these HUC2 regions are removed (region-held-out "
                         "track, protocol §4.6); normalisation refitted without them")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    cfg = {"seed": a.seed}
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = json.loads(v)
    if a.smoke:
        d = D.build("validation", max_sites=a.smoke)
        out = paths.ensure(paths.RUNS_ROOT / "sim" / "smoke" / a.run_name)
        cfg.update(batches_per_epoch=5, max_epochs=2)
        model, info = TR.train(d, cfg, device)
        print("smoke ok", info["best_cal_loss"])
        return 0
    d = runner.load_data("validation")
    out = paths.ensure(paths.RUNS_ROOT / "sim" / a.run_name)
    train_days = None
    extra = {}
    if a.exclude_huc2:
        from sentinel.models.sim.strata import site_huc2

        regions = sorted(x.strip().zfill(2) for x in a.exclude_huc2.split(",") if x.strip())
        h2 = site_huc2(d)
        drop = np.isin(d.role, ["train", "cal"]) & np.isin(h2, regions)
        d.role = np.where(drop, "excluded_region", d.role)
        D.fit_norm(d)
        extra = {"excluded_huc2": regions, "n_excluded_train_cal_sites": int(drop.sum()),
                 "n_train_left": int((d.role == "train").sum()), "n_cal_left": int((d.role == "cal").sum())}
        (out / "region_holdout.json").write_text(json.dumps(extra, indent=1))
    if a.dsst or a.dsst_control:
        from sentinel.models.sim.dsst import water_year

        cool, hot, score = cool_years(d)
        wy = water_year(d.dates)
        if a.dsst_control:
            years = sorted(cool + hot)
            rs = np.random.default_rng(0)
            fit = sorted(int(y) for y in rs.choice(years, size=len(cool), replace=False))
            held = [y for y in years if y not in fit]
        else:
            fit, held = cool, hot
        train_days = D.block_mask(d.dates, "train") & np.isin(wy, fit)
        D.fit_norm(d, days=train_days)      # no statistic may see the held-out (hot) years
        extra = {"fit_wy": fit, "eval_wy": held, "dsst_cool_wy": cool, "dsst_hot_wy": hot, "jja_tair_by_wy": score,
                 "kind": "control_random_years" if a.dsst_control else "cool_years"}
        (out / "dsst_years.json").write_text(json.dumps(extra, indent=1))
    exp = "sim_dsst_control" if a.dsst_control else ("sim_dsst" if a.dsst else
                                                       ("sim_region_holdout" if a.exclude_huc2 else "sim_train"))
    full_cfg = {**TR.DEFAULTS, **cfg, "device": device, "dsst": a.dsst, "dsst_control": a.dsst_control,
                **({"fit_wy": extra["fit_wy"]} if "fit_wy" in extra else {}),
                **({"excluded_huc2": extra["excluded_huc2"]} if "excluded_huc2" in extra else {})}
    with TrainingRun("sim", exp, config=full_cfg, data=list(D.DATA_MANIFESTS), plan_ref="RESEARCH_PLAN §7.1",
                     iteration=a.run_name, artifacts_dir=str(out)) as run:
        run.note(f"data {d.provenance()}")
        t0 = time.time()
        model, info = TR.train(d, cfg, device, train_days=train_days, log=lambda s: print(s, flush=True))
        runner.save_model(model, info["config"], d, out / "model.pt")
        (out / "history.json").write_text(json.dumps(info, indent=1, default=str))
        n_par = sum(p.numel() for p in model.parameters())
        run.metric("best_cal_loss", info["best_cal_loss"])
        run.metric("epochs_run", info["epochs_run"])
        run.metric("n_params", n_par)
        run.metric("train_minutes", (time.time() - t0) / 60)
        idx = np.concatenate([d.idx("cal"), d.idx("val")])
        sim = runner.simulate_sites(model, d, idx, device=device)
        runner.save_sim(out / "sim_eval.npz", sim, d, idx)
        run.note(f"saved simulations for {len(idx)} cal+val sites; params {n_par}")
    print(f"done {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
