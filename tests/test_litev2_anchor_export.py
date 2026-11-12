# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Anchor-tree exporter of M7 architecture round v2 (amendment §4.6: "parity with LightGBM within 1e-4 log10 is a
unit test") — synthetic data only (no real row is loaded)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.models.litev2 import anchor_export as AX
from sentinel.models.litev2 import tabular as T

TOL = 1e-4                                                       # log10, amendment §4.6


def _synthetic(n=500, seed=0, nan_cols=("red_p50", "nir_mean"), zero_cols=("valid_frac",)):
    rng = np.random.default_rng(seed)
    cols = T.parity_columns()
    X = pd.DataFrame(rng.uniform(0.1, 1.0, size=(n, len(cols))), columns=cols)
    for c in nan_cols:                                           # features with missing values in training
        X.loc[rng.random(n) < 0.15, c] = np.nan
    for c in zero_cols:
        X.loc[rng.random(n) < 0.2, c] = 0.0
    sig = 2.0 * X["c9s_R"].to_numpy() - X["c9s_B"].to_numpy() + np.nan_to_num(X["red_p50"].to_numpy(), nan=0.7)
    Y = pd.DataFrame({f"y_{q}": np.nan for q in T.QUANTITIES}, index=X.index)
    Y["y_secchi"] = -sig + 0.05 * rng.normal(size=n)
    unit = rng.integers(0, 3, n)
    for k, u in enumerate(("turbidity", "turb_fnu", "turb_ntru")):
        Y.loc[unit == k, f"y_{u}"] = (sig + (0.0, 0.5, -0.5)[k] + 0.05 * rng.normal(size=n))[unit == k]
    Y.loc[::2, "y_tss"] = (sig + 0.05 * rng.normal(size=n))[::2]
    Y.loc[::5, "y_ssc"] = (sig + 0.05 * rng.normal(size=n))[::5]
    Y.loc[::3, "y_chla"] = (X["c9s_G"].to_numpy() + 0.05 * rng.normal(size=n))[::3]
    return X, Y


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    X, Y = _synthetic()
    model = T.fit_parity(X, Y, T.parity_columns())               # the frozen GBM_PARAMS, three seeds, pooled
    path = tmp_path_factory.mktemp("anchor") / "anchor.slanch"
    info = AX.export_anchor(model, path)
    return model, path, info


def _f32(X: pd.DataFrame) -> pd.DataFrame:
    return X.astype(np.float32).astype(np.float64)


def test_floor32_is_the_largest_float32_not_above():
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(size=2000) * 10.0 ** rng.integers(-6, 6, 2000), [0.0, 1e300, -1e300, 0.1, 1.0]])
    f = AX.floor32(x)
    assert f.dtype == np.float32
    fin = np.isfinite(f)
    assert (f[fin].astype(np.float64) <= x[fin]).all()
    with np.errstate(over="ignore"):
        assert (np.nextafter(f[fin], np.float32(np.inf)).astype(np.float64) > x[fin]).all()
    v = rng.normal(size=2000).astype(np.float32)                 # float32 values decide exactly as with the double
    t = rng.normal(size=2000)
    assert ((v <= AX.floor32(t)) == (v.astype(np.float64) <= t)).all()
    # beyond the float32 range: the largest float32 below +1e300 is the largest finite one; below -1e300 only -inf
    assert AX.floor32([1e300])[0] == np.finfo(np.float32).max and AX.floor32([-1e300])[0] == -np.inf


def test_parity_with_lightgbm_on_float32_features(fitted):
    model, path, _ = fitted
    af = AX.load_anchor(path)
    Xn, _ = _synthetic(n=700, seed=5)                            # rows the models have not seen
    for X in (_synthetic()[0], Xn):
        ref = T.predict_parity(model, _f32(X))
        got = AX.predict_anchor(af, X)
        assert list(got.columns) == list(T.QUANTITIES) and np.isfinite(got.to_numpy()).all()
        assert np.abs(got.to_numpy() - ref.to_numpy()).max() < TOL


def test_parity_with_missing_and_zero_values_not_seen_in_training(fitted):
    """NaN in a column that had none in training (LightGBM then reads it as 0), NaN in one that had, exact zeros."""
    model, path, _ = fitted
    af = AX.load_anchor(path)
    X, _ = _synthetic(n=400, seed=9)
    rng = np.random.default_rng(3)
    for c in ("c9s_R", "c9s_B", "lat", "green_p95", "red_p50", "doy"):
        X.loc[rng.random(len(X)) < 0.3, c] = np.nan
    for c in ("c9s_G", "blue_mean", "valid_frac"):
        X.loc[rng.random(len(X)) < 0.3, c] = 0.0
    ref = T.predict_parity(model, _f32(X))
    got = AX.predict_anchor(af, X)
    assert np.abs(got.to_numpy() - ref.to_numpy()).max() < TOL


def test_counts_and_file_size(fitted):
    model, path, info = fitted
    n_nodes = n_leaves = n_trees = 0
    for g in AX.GROUPS:
        assert len(model["models"][g]) == len(T.SEEDS)
        for m in model["models"][g]:
            for tr in m.booster_.dump_model()["tree_info"]:
                n_trees += 1
                n_leaves += int(tr["num_leaves"])
                n_nodes += int(tr["num_leaves"]) - 1
    assert (info["n_trees"], info["n_nodes"], info["n_leaves"]) == (n_trees, n_nodes, n_leaves)
    assert info["n_numbers"] == n_nodes + n_leaves == AX.load_anchor(path).n_numbers
    raw = path.read_bytes()
    head = 12 + int.from_bytes(raw[8:12], "little")
    assert len(raw) == info["bytes"] == head + n_trees + 8 * n_nodes + 4 * n_leaves
    assert [g["name"] for g in info["groups"]] == list(AX.GROUPS)
    assert info["turb_cols"][-2:] == list(T.UNIT_COLS)


def test_units_of_the_pooled_turbidity_model_differ(fitted):
    _, path, _ = fitted
    got = AX.predict_anchor(AX.load_anchor(path), _synthetic(n=300, seed=7)[0])
    assert np.median(got["turb_fnu"] - got["turbidity"]) > 0.2 and np.median(got["turb_ntru"] - got["turbidity"]) < -0.2


def test_refusals(fitted, tmp_path):
    model, path, _ = fitted
    with pytest.raises(AX.AnchorExportError):
        AX.export_anchor({**model, "pooled": False}, tmp_path / "a")
    with pytest.raises(AX.AnchorExportError):
        AX.export_anchor({**model, "models": {**model["models"], "ssc": []}}, tmp_path / "b")
    bad = tmp_path / "c"
    bad.write_bytes(path.read_bytes()[:-5])
    with pytest.raises(AX.AnchorExportError):
        AX.load_anchor(bad)
    bad.write_bytes(b"XXXXXXXX" + path.read_bytes()[8:])
    with pytest.raises(AX.AnchorExportError):
        AX.load_anchor(bad)
