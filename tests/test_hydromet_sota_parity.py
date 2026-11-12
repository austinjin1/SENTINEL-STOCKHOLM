# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M6 SOTA parity comparators and lever-site rules (synthetic arrays only; code paths, no results)."""

import numpy as np
import pandas as pd
import pytest

from sentinel.hydromet_sota import gbm as G
from sentinel.hydromet_sota import physics as PH
from sentinel.hydromet_sota import sites as LS
from sentinel.models.hydromet.features import FeatureSet
from sentinel.models.hydromet.physics import air2stream_simulate

from test_hydromet import synth


def test_weak_weight_gives_the_share():
    w = G.weak_weight(9000, 300, 0.25)
    assert abs(300 * w / (9000 + 300 * w) - 0.25) < 1e-12
    assert G.weak_weight(10, 0, 0.5) == 0.0 and G.weak_weight(10, 5, 0.0) == 0.0


def test_air2stream_q_with_a4_zero_is_air2stream():
    rs = np.random.default_rng(0)
    ta = rs.normal(10, 5, (3, 200))
    doy = (np.arange(200) % 365) + 1
    p = {"a1": np.full(3, 0.3), "a2": np.full(3, 0.1), "a3": np.full(3, 0.12), "a5": np.full(3, 0.2), "a6": np.full(3, 0.6)}
    a = air2stream_simulate(ta, doy, p)
    b = PH.air2stream_q_simulate(ta, np.exp(rs.normal(0, 1, (3, 200))), doy, {**p, "a4": np.zeros(3)})
    np.testing.assert_allclose(a, b)


def test_weighted_median():
    assert PH._wmedian(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 10.0])) == 3.0
    assert PH._wmedian(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0])) == 2.0


def test_weighted_gbm_uses_weak_rows_and_never_other_roles():
    d = synth(S=8, roles=["train"] * 8, groups=["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"], end="2021-09-30",
              blocks=("train",))
    fs = FeatureSet.build(d, lag=0)
    from sentinel.models.hydromet.baselines import LGBMConfig

    cfg = LGBMConfig(n_estimators=30, early_stopping_rounds=5, n_jobs=1, min_child_samples=20)
    m = G.WeightedForcingGBM("g", fs.columns(), targets=("temp_c_mean",), cfg=cfg, seed=0,
                             grid=[cfg])
    m.fit(d, fs, np.array([0, 1, 2, 3]), np.array([4, 5]), weak_idx=np.array([6, 7]), weak_share=0.25,
          log=lambda *_: None)
    rec = m.info["temp_c_mean"]["grid"][0]
    assert rec["n_weak"] > 0 and m.info["temp_c_mean"]["weak_share"] == 0.25
    d.roles[6] = "val"
    with pytest.raises(ValueError):
        m.fit(d, fs, np.array([0, 1, 2, 3]), np.array([4, 5]), weak_idx=np.array([6, 7]), weak_share=0.25,
              log=lambda *_: None)


def test_append_qhat_is_causal_and_named():
    d = synth(S=4, roles=["train"] * 4, groups=["g1", "g1", "g2", "g2"], end="2021-09-30", blocks=("train",))
    fs = FeatureSet.build(d, lag=0)
    q = np.random.default_rng(0).normal(0, 1, d.y.shape[:2]).astype("float32")
    f2 = G.append_qhat(fs, q)
    assert f2.dyn_names[-3:] == list(G.QHAT_NAMES)
    q2 = q.copy()
    q2[:, 500:] += 100.0                       # future change must not alter earlier features
    f3 = G.append_qhat(fs, q2)
    np.testing.assert_array_equal(f2.dyn[:, :500, -3:], f3.dyn[:, :500, -3:])


def test_site_rules_order_and_reserved():
    res = {"01234567"}
    owner = {"0101": ("0101", "train"), "0202": ("0202", "val"), "0303": ("0303", "train")}
    pool = pd.DataFrame({"key": ["USGS-01234567", "USGS-1", "USGS-2", "USGS-3", "USGS-4", "WQP:X-01234567"],
                         "site_no": ["01234567", "1", "2", "3", "4", "X-01234567"],
                         "huc4": ["0101", "0202", "0909", "0303", "0101", "0101"],
                         "comid": ["a", "b", "c", "d", "e", "f"]})
    out = LS.apply_rules(pool, owner=owner, inner_groups=["0303"], connected={"e": True}, res=res)
    assert out["rule"].tolist() == ["reserved", "val_group", "new_region", "inner_group", "flow_connected", "reserved"]
