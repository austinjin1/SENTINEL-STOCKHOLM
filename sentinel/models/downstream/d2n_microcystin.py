# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D2n — national lake microcystin track (protocol docs/preregistration/downstream_d2n_microcystin.md).

Enlarged track of D2 ToxinWatch (RESEARCH_PLAN §15.5 amendment 2026-11-08). The Washington track is
unchanged; Washington rows are excluded here so that no WA test-block row (test lakes, 2022+) can enter
this track's training data.
"""

from __future__ import annotations

from .wqp_track import TrackConfig

APP = "d2n_microcystin"
DATA_MANIFESTS = ("wqp_discrete", "forcings_daily", "wbd_huc12")
CFG = TrackConfig(app=APP, variable="microcystins", threshold=8.0, agg="max", lag_days=7, net_window=30, net_km=50.0,
                  cluster_km=2.0, train_years=(2006, 2017), val_years=(2018, 2020), test_years=(2021, 2025),
                  excluded_years=(), site_filter="lake", exclude_states=("WA",), persist_gap=30)
INCUMBENT = "st_glmm"


def strata():
    """A-priori difficulty strata (protocol §5): cold-start lakes, hot periods."""
    import numpy as np

    def cold(ev, Xe, Xtr):
        return Xe["tw_n_prior"].to_numpy(float) == 0

    def hot(ev, Xe, Xtr):
        p90 = np.nanpercentile(Xtr["wx_tmax_anom_7d"].to_numpy(float), 90)
        return Xe["wx_tmax_anom_7d"].to_numpy(float) > p90

    return {"cold_start_no_prior": cold, "tmax_anom7d_gt_train_p90": hot}
SPLIT_SHA = "0d9eb684285155c7d73a8454b3b211aea626e5bb6b1bc809970d55ded4176295"  # frozen 2026-11-08 (protocol §8)
