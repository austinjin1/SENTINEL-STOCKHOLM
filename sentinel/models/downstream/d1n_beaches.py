# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D1n — national freshwater-beach E. coli track (protocol docs/preregistration/downstream_d1n_beaches.md).

Enlarged track of D1 SafeSwim (RESEARCH_PLAN §15.5 amendment 2026-11-08, items 5–7). The Chicago track is
unchanged; Chicago lakefront beaches are excluded here (box below) so that no Chicago D1a/D1b test-block
row can enter this track's training data.
"""

from __future__ import annotations

from .wqp_track import TrackConfig

APP = "d1n_beaches"
DATA_MANIFESTS = ("wqp_discrete", "forcings_daily", "wbd_huc12")
CHICAGO_BOX = (41.60, 42.10, -87.80, -87.45)
CFG = TrackConfig(app=APP, variable="ecoli", threshold=235.0, agg="gmean", lag_days=1, net_window=3, net_km=25.0,
                  cluster_km=1.0, train_years=(2004, 2015), val_years=(2016, 2019), test_years=(2021, 2025),
                  excluded_years=(2020,), site_filter="beach", exclude_boxes=(CHICAGO_BOX,), persist_gap=3)
INCUMBENT = "vb_mlr"


def strata():
    """A-priori difficulty strata (protocol §5): rain-event days and cold-start beaches."""
    import numpy as np

    def rain(ev, Xe, Xtr):
        p90 = np.nanpercentile(Xtr["wx_precip_sum_2d"].to_numpy(float), 90)
        return (Xe["wx_precip_sum_2d"].to_numpy(float) > p90)

    def cold(ev, Xe, Xtr):
        return Xe["tw_n_prior"].to_numpy(float) < 10

    def no_recent(ev, Xe, Xtr):
        return ~(Xe["tw_days_since_last"].to_numpy(float) <= 3)

    return {"rain_gt_train_p90_2d": rain, "cold_start_lt10_prior": cold, "no_result_within_3d": no_recent}
SPLIT_SHA = "aa601b505530f2d93bf90881e106d882ffc46efdf38ac9cda2db1816e6f96861"  # frozen 2026-11-08 (protocol §8)
