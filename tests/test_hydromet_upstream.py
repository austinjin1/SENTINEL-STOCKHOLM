# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Upstream-basin forcing aggregation (sentinel.data.hydromet_upstream) and its M6 integration (protocol v0.10).
Synthetic arrays only (unit tests)."""

import numpy as np
import pytest
from shapely.geometry import box

from sentinel.data import hydromet_upstream as U
from sentinel.models.hydromet import data as D
from sentinel.models.hydromet.features import forcing_features


def _grid(ny=20, nx=30, lat_top=45.0, lon_left=-100.0, d=1 / 24):
    lat = lat_top - d / 2 - d * np.arange(ny)
    lon = lon_left + d / 2 + d * np.arange(nx)
    return U.grid_geometry(lat, lon)


def test_weights_cover_exact_cells_and_sum_to_one():
    g = _grid()
    valid = np.ones((g["ny"], g["nx"]), bool)
    d = 1 / 24
    # polygon covering exactly cells i = 2..3, j = 5..7 (6 cells)
    poly = box(-100 + 5 * d, 45 - 4 * d, -100 + 8 * d, 45 - 2 * d)
    idx, w, info = U.basin_cell_weights(poly, g, valid)
    assert info["n_cells"] == 6 and not info["fallback"]
    ij = sorted((int(k) // g["nx"], int(k) % g["nx"]) for k in idx)
    assert ij == [(i, j) for i in (2, 3) for j in (5, 6, 7)]
    assert np.isclose(w.sum(), 1.0)


def test_masked_cells_excluded_and_cover_frac():
    g = _grid()
    valid = np.ones((g["ny"], g["nx"]), bool)
    valid[2, 5] = False
    d = 1 / 24
    poly = box(-100 + 5 * d, 45 - 4 * d, -100 + 8 * d, 45 - 2 * d)
    idx, w, info = U.basin_cell_weights(poly, g, valid)
    assert info["n_cells"] == 5 and (2 * g["nx"] + 5) not in set(idx.tolist())
    assert 0.8 < info["cover_frac"] < 0.85


def test_tiny_basin_falls_back_to_cell():
    g = _grid()
    valid = np.ones((g["ny"], g["nx"]), bool)
    d = 1 / 24
    c = (-100 + 10.6 * d, 45 - 3.4 * d)   # off the sub-cell centres
    poly = box(c[0] - 1e-4, c[1] - 1e-4, c[0] + 1e-4, c[1] + 1e-4)
    idx, w, info = U.basin_cell_weights(poly, g, valid)
    assert info["fallback"] and idx.tolist() == [3 * g["nx"] + 10] and w.tolist() == [1.0]


def test_aggregate_weighted_mean_and_nan_renormalisation():
    g = _grid(ny=2, nx=2)
    W = U.weight_matrix([(np.array([0, 1]), np.array([0.25, 0.75])), (np.array([3]), np.array([1.0]))], 4)
    data = np.array([[[1.0, 3.0], [0.0, 5.0]], [[np.nan, 3.0], [0.0, np.nan]]], dtype="float32")
    out = U.aggregate_grid(W, data)
    assert out.shape == (2, 2)                       # (sites, days)
    assert np.allclose(out[:, 0], [0.25 * 1 + 0.75 * 3, 5.0])
    assert np.isclose(out[0, 1], 3.0) and np.isnan(out[1, 1])
    assert g["ny"] == 2


def test_forcing_names_and_feature_prefixes():
    nf = len(D.FORCING_VARS)
    assert D.forcing_names(nf) == D.FORCING_VARS
    names = D.forcing_names(2 * nf)
    assert names[nf:] == ["ws_" + v for v in D.FORCING_VARS]
    with pytest.raises(ValueError):
        D.forcing_names(nf + 1)
    rs = np.random.default_rng(0)
    f = rs.normal(10, 3, (2, 400, 2 * nf)).astype("float32")
    a, na = forcing_features(f[:, :, :nf], np.array([100.0, 200.0]))
    b, nb = forcing_features(f, np.array([100.0, 200.0]))
    assert nb[:len(na)] == na and all(n.startswith("f_ws_") for n in nb[len(na):]) and len(nb) == 2 * len(na)
    assert np.allclose(a, b[:, :, :len(na)], equal_nan=True)
    # upstream block = same features computed on the upstream channels
    c, _ = forcing_features(f[:, :, nf:], np.array([100.0, 200.0]))
    assert np.allclose(c, b[:, :, len(na):], equal_nan=True)
    # names pass the input whitelist and do not look sensor-derived
    D.assert_allowed_inputs(nb, static_cols=[], forcing_feature_cols=nb)


def test_reverse_graph_upstream_set():
    import pandas as pd

    # 1 -> 2 -> 4, 3 -> 4 (hydroseq decreases downstream); 5 isolated
    vaa = pd.DataFrame({"comid": ["a", "b", "c", "d", "e"], "hydroseq": [50, 40, 45, 10, 99],
                        "dnhydroseq": [40, 10, 10, 0, 0]})
    g, pos = U.reverse_graph(vaa)
    assert sorted(U.upstream_nodes(g, pos["d"]).tolist()) == [0, 1, 2, 3]
    assert sorted(U.upstream_nodes(g, pos["b"]).tolist()) == [0, 1]
    assert U.upstream_nodes(g, pos["e"]).tolist() == [4]


def test_catchment_subcells_and_local_weights():
    import geopandas as gpd

    g = _grid()
    d = 1 / 24
    polys = gpd.GeoDataFrame({"idx": [0, 1]}, geometry=[box(-100 + 5 * d, 45 - 4 * d, -100 + 7 * d, 45 - 2 * d),
                                                       box(-100 + 7 * d, 45 - 4 * d, -100 + 8 * d, 45 - 2 * d)])
    c, cell, n = U.catchment_subcells(polys, g)
    assert set(c.tolist()) == {0, 1} and np.isclose(n[c == 0].sum(), 4 * 25) and np.isclose(n[c == 1].sum(), 2 * 25)
    from scipy import sparse

    C = sparse.csr_matrix((n, (c, cell)), shape=(2, g["ny"] * g["nx"]))
    coslat = np.repeat(np.cos(np.radians(g["lat"])), g["nx"])
    idx, w, info = U.basin_weights_local(np.array([0, 1]), C, coslat, np.ones(g["ny"] * g["nx"], bool))
    assert info["n_cells"] == 6 and np.isclose(w.sum(), 1.0)
