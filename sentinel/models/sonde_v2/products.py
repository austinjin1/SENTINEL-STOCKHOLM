# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""The ONE declared NWP product of SENTINEL-Sonde v2 runs: GEFSv12 control member c00 (amendment §2c; lead rulings
R3 and G(b)). Built once per process; no other forecast product is read by any arm in this round (the Open-Meteo GFS
table is not read at all).

* ``obs_f``: observed gridMET of the target days (perfect-prog reference; TRAINING and bias/error fitting only).
* ``gefs``: GEFS c00 point extraction, reforecast (issues 2007-10-01..2019-12-31) ∪ operational (issues from
  2020-09-23; loaded for use="cv" only), bias-corrected. NaN where no archived forecast exists (2020-01-01..
  2020-09-22: such issues run in past-weather mode; nothing is substituted).
* Bias map: fitted on REFORECAST pairs only, every pair whose valid day is ≤ 2021-09-30 (:func:`nwp.fit_bias_map`),
  and applied unchanged to the operational product. The same months are required for step 1 and for CV, so the
  per-site maps are identical in both.
* ``pp``: perfect-prog training input = ``obs_f`` + one seeded sample of the stage-1 error model. The error model is
  fitted on bias-corrected reforecast minus gridMET at TRAIN-role sites, issue days whose 7 valid days all lie in
  WY2008–2019 (≤ 2019-09-30).
All fits use weather only (no target). A missing month file or a coverage below ``COVERAGE_MIN`` is a hard error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import nwp as NWP
from .prep import GEFS_REQUIRED, gefs_coverage, load_gefs, observed_weather_fields

TRAIN_END = pd.Timestamp("2021-09-30")
ERROR_FIT_END = pd.Timestamp("2019-09-30")
OPERATIONAL_START = pd.Timestamp("2020-09-23")
COVERAGE_MIN = 0.90        # share of site × issue-day cells with a finite lead-1 temperature, per required window
PRODUCT = "gefs_c00"
SEED = 20261110


@dataclass
class Products:
    obs_f: np.ndarray
    gefs: np.ndarray
    pp: np.ndarray
    info: dict


def sha256_array(a: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast("B")).hexdigest()


def coverage_windows(use: str, dates: pd.DatetimeIndex) -> dict:
    """Required windows (issue days) per use; the last 7 days before the end of the loaded dates are left out of
    the last window (their leads run past the cube)."""
    w = {"reforecast_2007-10-01_2019-09-30": ("2007-10-01", "2019-09-30")}
    if use == "cv":
        w["operational_train_2020-09-23_2021-09-23"] = (OPERATIONAL_START, TRAIN_END - pd.Timedelta(days=7))
        w["operational_val_2021-10-01_2023-09-23"] = ("2021-10-01", dates[-1] - pd.Timedelta(days=7))
    return w


def build(data, *, use: str, seed: int = SEED, root=None, log=print) -> Products:
    """``use`` = "step1" (reforecast only) or "cv" (reforecast ∪ operational)."""
    req = GEFS_REQUIRED[use]
    dates = data.cube.dates
    obs = observed_weather_fields(data)
    obs_f = NWP.observed_future(obs, 7)
    ref = load_gefs(data, "reforecast", months=req["reforecast"], root=root)
    g = ref
    if "operational" in req:
        op = load_gefs(data, "operational", months=req["operational"], root=root)
        g = np.where(np.isfinite(ref), ref, op)
    cov = {k: gefs_coverage(g, dates, a, b) for k, (a, b) in coverage_windows(use, dates).items()}
    low = {k: v for k, v in cov.items() if v < COVERAGE_MIN}
    if low:
        raise RuntimeError(f"GEFS coverage below {COVERAGE_MIN}: {low}; refusing to run")
    huc2 = data.static["huc2"].astype(str).to_numpy()
    bm = NWP.fit_bias_map(ref, obs_f, dates, np.ones(len(dates), bool), huc2, valid_end=TRAIN_END)
    g_bc = bm.apply(g, dates).astype(np.float32)
    tr = data.site_idx(["train"])
    efit = np.asarray(dates + pd.Timedelta(days=7) <= ERROR_FIT_END)
    em = NWP.fit_error_model(g_bc[tr], obs_f[tr], dates, efit, L=7)
    pp = em.sample(obs_f, dates, np.random.default_rng(seed))
    ti = NWP.F_IDX["tmean_c"]
    info = {"product": PRODUCT, "use": use, "required_months": {k: list(v) for k, v in req.items()},
            "coverage": cov, "coverage_min": COVERAGE_MIN,
            "n_sites": int(g.shape[0]), "dates": [str(dates[0].date()), str(dates[-1].date()), int(len(dates))],
            "gefs_issue_days": int(np.isfinite(g[..., 0, ti]).any(0).sum()),
            "bias_fit": {"pairs": "reforecast only, valid day <= 2021-09-30", "min_days": 60,
                         "pool_min_pairs": NWP.POOL_MIN_PAIRS, "slope_clip": list(NWP.SLOPE_CLIP),
                         "source_cells": NWP.bias_source_counts(bm),
                         "sites_all_identity_tmean": int((bm.source[:, :, ti] == 0).all(axis=(1, 2)).sum())},
            "error_model": {"fit_sites": int(tr.size), "fit_end_valid_day": str(ERROR_FIT_END.date()),
                            "fit_issue_days": int(efit.sum()), "sd_l1_tmean": em.sd[0, ti].tolist(),
                            "sd_l7_tmean": em.sd[6, ti].tolist(), "rho": em.rho.tolist(),
                            "sha256": sha256_array(np.concatenate([em.mean.ravel(), em.sd.ravel(), em.rho]))},
            "pp_seed": int(seed), "pp_sha256": sha256_array(pp), "gefs_bc_sha256": sha256_array(g_bc)}
    log(f"[products] {PRODUCT} use={use} coverage {cov} pp {info['pp_sha256'][:12]}")
    return Products(obs_f, g_bc, pp, info)
