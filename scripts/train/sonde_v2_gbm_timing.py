#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Timing-only benchmark of the GBM references (lead ruling E; rule text written before the run:
``docs/preregistration/sonde_arch_v2_gbm_timing_rule.md``). TRAIN-role sites × TRAIN block, PF-GBM features,
50 boosting rounds per booster, no validation set. Kept: wall times, the training loss of the CPU and GPU boosters
on a fixed sample of fit rows (device equality only), labelled-row counts, load averages. NO skill, NO held-out or
early-stopping loss, no forecast file, no registry row.

    python -B scripts/train/sonde_v2_gbm_timing.py [--parts cpu,gpu] [--out gbm_timing.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sentinel.core import paths  # noqa: E402
from sentinel.models.sonde_v2 import declared as D  # noqa: E402

RULE = paths.REPO_ROOT / "docs" / "preregistration" / "sonde_arch_v2_gbm_timing_rule.md"
OUT = paths.RUNS_ROOT / "sonde" / "v2_smoke"
VAR, H, ROUNDS, MAX_LOAD = "temp_c_mean", 1, 50, 36.0
ROWS = (None, 500_000)
BOOSTERS = (("q05", {"objective": "quantile", "alpha": 0.05}), ("q50", {"objective": "quantile", "alpha": 0.50}),
            ("q95", {"objective": "quantile", "alpha": 0.95}), ("l1", {"objective": "l1"}),
            ("l2", {"objective": "l2"}))
N_BOOSTERS = {"Q19": 19, "Q": 7, "LS": 2}


def log(m):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def pinball(y, p, a):
    d = y - p
    return float(np.mean(np.maximum(a * d, (a - 1) * d)))


def train_loss(name, extra, y, p):
    if extra["objective"] == "quantile":
        return pinball(y, p, extra["alpha"])
    return float(np.mean(np.abs(y - p))) if name == "l1" else float(np.mean((y - p) ** 2))


def time_fits(X, y, params, sample) -> dict:
    """Binning time and seconds per round of each booster (50 rounds, no validation set)."""
    import lightgbm as lgb

    t0 = time.time()
    ds = lgb.Dataset(X, y, params={k: params[k] for k in ("max_bin", "num_threads", "verbose")},
                     free_raw_data=False).construct()
    out = {"n_rows": int(len(y)), "n_features": int(X.shape[1]), "t_build_s": round(time.time() - t0, 2),
           "boosters": {}}
    Xs, ys = X.iloc[sample], y[sample]
    for name, extra in BOOSTERS:
        t1 = time.time()
        b = lgb.train({**params, **extra}, ds, num_boost_round=ROUNDS)
        el = time.time() - t1
        out["boosters"][name] = {"s_per_round": round(el / ROUNDS, 4), "seconds": round(el, 2),
                                 "train_loss_sample": train_loss(name, extra, ys, b.predict(Xs))}
        log(f"  {name}: {el:.1f} s ({el / ROUNDS:.3f} s/round)")
    q = [out["boosters"][k]["s_per_round"] for k in ("q05", "q50", "q95")]
    out["s_per_round"] = {"quantile": round(float(np.mean(q)), 4), "l1": out["boosters"]["l1"]["s_per_round"],
                          "l2": out["boosters"]["l2"]["s_per_round"]}
    return out


def synthetic_check(params_cpu, params_gpu) -> dict:
    """CPU vs GPU pinball loss on a seeded synthetic task (rule §6)."""
    import lightgbm as lgb

    rs = np.random.default_rng(D.SEED)
    X = rs.normal(size=(250_000, 40)).astype(np.float32)
    y = (X[:, 0] * 2 + np.sin(3 * X[:, 1]) + X[:, 2] * X[:, 3]
         + (0.5 + 0.5 * np.abs(X[:, 4])) * rs.standard_t(4, size=X.shape[0])).astype(np.float32)
    Xa, ya, Xb, yb = X[:200_000], y[:200_000], X[200_000:], y[200_000:]
    out = {}
    for a in (0.1, 0.5, 0.9):
        loss = {}
        for dev, p in (("cpu", params_cpu), ("gpu", params_gpu)):
            b = lgb.train({**p, "objective": "quantile", "alpha": a}, lgb.Dataset(Xa, ya), num_boost_round=200)
            loss[dev] = pinball(yb, b.predict(Xb), a)
        loss["rel_diff"] = abs(loss["gpu"] - loss["cpu"]) / loss["cpu"]
        out[f"alpha_{a}"] = loss
    out["within_1pct"] = bool(all(v["rel_diff"] <= 0.01 for v in out.values()))
    return out


def s_at(n, n_full, t) -> float:
    """Seconds per round at ``n`` rows from the two timed row counts (rule §5): linear between (and beyond)
    500,000 and full rows, proportional below 500,000."""
    lo = 500_000
    if n < lo:
        return t["500k"] * n / lo
    return t["500k"] + (t["full"] - t["500k"]) * (n - lo) / max(n_full - lo, 1)


def projections(res: dict, counts: dict, site_ratio: float) -> dict:
    """Upper-bound hours at 2000 rounds (no early-stopping credit) per device, configuration and cap."""
    out = {}
    n_full = counts[VAR]
    for dev in ("cpu", "gpu"):
        if dev not in res or not {"full", "500k"} <= set(res[dev]):     # both timed tables are needed (rule §5)
            continue
        r = res[dev]
        pd_ = {}
        for cname, nb in N_BOOSTERS.items():
            for cap in D.GBM_SETTINGS["cap_ladder"]:
                def unit(n_rows, cap=cap, cname=cname, nb=nb):
                    n = n_rows if cap is None else min(cap, n_rows)
                    if cname == "LS":
                        s = sum(s_at(n, n_full, {k: r[k]["s_per_round"][o] for k in ("full", "500k")})
                                for o in ("l1", "l2"))
                    else:
                        s = nb * s_at(n, n_full, {k: r[k]["s_per_round"]["quantile"] for k in ("full", "500k")})
                    tb = r["500k"]["t_build_s"] + (r["full"]["t_build_s"] - r["500k"]["t_build_s"]) * \
                        max(n - 500_000, 0) / max(n_full - 500_000, 1)
                    return (tb + D.GBM_SETTINGS["max_rounds"] * s) / 3600
                step1 = sum(unit(counts[v]) for v in D.STEP1["variables"]) * len(D.STEP1["horizons"])
                cv = sum(unit(int(counts[v] * site_ratio)) for v in D.SCORED) * len(D.HORIZONS)
                pd_[f"{cname}|cap={cap or 'full'}"] = {
                    "unit_hours_largest_variable": round(unit(n_full), 2),
                    "step1_hours": round(step1, 1) if cname == "Q19" else None,
                    "cv_hours_per_fold_arm": round(cv, 1)}
        out[dev] = pd_
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="cpu,gpu")
    ap.add_argument("--out", default="gbm_timing.json")
    ap.add_argument("--loaded", action="store_true",
                    help="start at any load (docs/preregistration/sonde_arch_v2_dev_lead_rulings.md, ruling Q): the "
                         "record may only support the conclusion 'full rows'")
    a = ap.parse_args()
    parts = set(a.parts.split(","))
    if not RULE.is_file():
        raise SystemExit(f"{RULE} must be written before the timing run")
    load0 = os.getloadavg()
    if load0[0] > MAX_LOAD and not a.loaded:
        raise SystemExit(f"1-minute load average {load0[0]:.1f} > {MAX_LOAD}: not starting")
    if a.loaded and a.out == "gbm_timing.json":
        raise SystemExit("--loaded needs its own --out file (gbm_timing.json is the record at load <= 36)")
    t_start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    from sentinel.models.sonde import runner
    from sentinel.models.sonde_v2 import nwp as NWP
    from sentinel.models.sonde_v2.gbm_nwp import GBMNWPConfig, SondeGBMNWP
    from sentinel.models.sonde_v2.prep import observed_weather_fields, prepare_v2

    g = D.GBM_SETTINGS
    runner.limit_cpu_threads(g["num_threads"])
    data = prepare_v2(blocks=("train",), roles=("train",), log=log)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sonde_v2_step1 import site_sets
    fit, es, held = site_sets(data)
    nwp = NWP.observed_future(observed_weather_fields(data), 7)
    cfg = GBMNWPConfig(quantile_mode="quantile", quantile_levels=19, physics=True, learning_rate=g["learning_rate"],
                       num_threads=g["num_threads"], max_bin=g["max_bin"], seed=g["seed"])
    m = SondeGBMNWP(cfg, horizons=(H,))
    fb = m._builder(data)
    counts = {v: int(m.labelled_rows(data, v, H, fit, cap=None, salt=0)[4]) for v in D.SCORED
              if v in data.cube.variables}
    log(f"labelled fit rows at h={H}: {counts}")
    n_tr = len(data.split.sites("train"))
    site_ratio = (n_tr + len(data.split.sites("val"))) / n_tr
    res = {"rule_sha256": hashlib.sha256(RULE.read_bytes()).hexdigest(), "utc_start": t_start,
           "load_start": [round(x, 2) for x in load0], "max_load_rule": MAX_LOAD,
           "load_route": ("dev rulings Q: started above the load limit; usable only for the conclusion 'full rows', "
                          "otherwise not_used_for_cap") if load0[0] > MAX_LOAD else "timing rule: load <= 36",
           "variable": VAR, "horizon": H, "rounds": ROUNDS,
           "arm_features": "pf_gbm", "threads": g["num_threads"], "max_bin": g["max_bin"],
           "labelled_fit_rows_h1": counts, "n_fit_sites": int(fit.size), "cv_site_ratio": round(site_ratio, 4),
           "lightgbm": __import__("lightgbm").__version__}
    tabs = {}
    for cap in ROWS:
        si, ti, y, _, n_total = m.labelled_rows(data, VAR, H, fit, cap=cap, salt=0)
        X = fb.table(VAR, si, ti, H, blocks=m.blocks_h(data, VAR, nwp, H))
        sample = np.sort(np.random.default_rng(D.SEED).choice(len(y), min(200_000, len(y)), replace=False))
        tabs["full" if cap is None else "500k"] = (X, np.asarray(y, np.float32), sample)
    p_cpu = m._params()
    out_f = OUT / a.out
    OUT.mkdir(parents=True, exist_ok=True)

    def save():
        res["load_end"] = [round(x, 2) for x in os.getloadavg()]
        res["utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        res["projection_upper_bound_hours"] = projections(res, counts, site_ratio)
        out_f.write_text(json.dumps(res, indent=1))

    if "cpu" in parts:
        res["cpu"] = {}
        for k, (X, y, sample) in tabs.items():
            log(f"CPU {k}: {len(y)} rows x {X.shape[1]} features")
            res["cpu"][k] = time_fits(X, y, p_cpu, sample)
            save()
    if "gpu" in parts:
        try:
            from sentinel.models.sonde import gpu_budget
            gpu_budget.widen(10000)
            gpu = runner.setup_gpu("sonde_v2_gbm_timing", prefer=7)
            runner.limit_cpu_threads(g["num_threads"])
            m_gpu = SondeGBMNWP(GBMNWPConfig(**{**cfg.__dict__, "device": "gpu"}), horizons=(H,))
            p_gpu = m_gpu._params()
            res["gpu"] = {"gpu_index": int(gpu), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                          "params": {k: v for k, v in p_gpu.items() if k not in p_cpu or p_cpu[k] != v}}
            res["gpu"]["synthetic"] = synthetic_check(p_cpu, p_gpu)
            log(f"GPU {gpu}: synthetic check {res['gpu']['synthetic']}")
            save()
            for k, (X, y, sample) in tabs.items():
                log(f"GPU {k}: {len(y)} rows x {X.shape[1]} features")
                res["gpu"][k] = time_fits(X, y, p_gpu, sample)
                if "cpu" in res and k in res["cpu"]:
                    res["gpu"][k]["train_loss_rel_diff_vs_cpu"] = {
                        b: abs(v["train_loss_sample"] - res["cpu"][k]["boosters"][b]["train_loss_sample"])
                        / max(res["cpu"][k]["boosters"][b]["train_loss_sample"], 1e-12)
                        for b, v in res["gpu"][k]["boosters"].items()}
                save()
        except Exception as e:  # noqa: BLE001  (e.g. a LightGBM build without GPU support: recorded, not hidden)
            res.setdefault("gpu", {})["error"] = repr(e)[:500]
            log(f"GPU part failed: {e!r}")
    save()
    log(json.dumps({k: v for k, v in res.items() if k != "labelled_fit_rows_h1"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
