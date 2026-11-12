# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
import pytest

from sentinel.models.downstream import troutrefuge as TT


@pytest.mark.parametrize("cols", [["lat", "mast2014"], ["msst2008"], ["sc_mwst2013", "lat"], ["thermal_mast_score"],
                                  ["MAST2009"]])
def test_modelled_temperature_never_a_feature(cols):
    with pytest.raises(AssertionError):
        TT.assert_no_modelled_temperature(cols)


def test_clean_features_pass():
    TT.assert_no_modelled_temperature(["lat", "lon", "pctforest2019ws", "tmax9120ws", "elev_m", "wx_tmax_mean_30d"])


def test_cqr_margin_is_the_finite_sample_quantile():
    import numpy as np

    from sentinel.models.downstream import troutrefuge_models as TM

    y = np.arange(10.0)
    lo, hi = y - 1.0, y + 1.0          # every row inside by 1 -> E = -1
    assert TM.cqr_margin(y, lo, hi, 0.9) == -1.0
    y2 = y.copy()
    y2[-1] += 5.0                      # one row 4 above its upper bound; ceil(11*0.9)=10 -> the largest score
    assert TM.cqr_margin(y2, lo, hi, 0.9) == 4.0
    assert TM.cqr_margin(y2, lo, hi, 0.5) == -1.0


def test_modelled_temp_refuses_unknown_metric():
    import pandas as pd

    with pytest.raises(ValueError):
        TT.modelled_temp(pd.DataFrame({"unit_id": [], "survey": []}), "mwst")
