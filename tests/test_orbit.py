# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Unit tests for SENTINEL-Orbit (M2): split rule, guard, pixel extraction, metrics, mixtures, model, baselines.
Synthetic arrays are used only here (RESEARCH_PLAN R9); no network access."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sentinel.data import chips as C
from sentinel.data.sources import stac as S
from sentinel.eval import metrics as M
from sentinel.models.orbit import data as D
from sentinel.models.orbit import evaluate as E
from sentinel.models.orbit import features as F
from sentinel.models.orbit import guard as G
from sentinel.models.orbit import split as SP


# ----------------------------------------------------------------------------------------- split

def _fake_lakes(lat, lon, lake=None, near=None):
    n = len(lat)
    out = pd.DataFrame({"hylak_id": pd.array([pd.NA] * n, dtype="Int64"), "lake_dist_m": np.nan,
                        "lake_area_km2": np.nan, "lake_name": None, "lake_pour_lon": np.nan, "lake_pour_lat": np.nan,
                        "near_hylak_id": pd.array([pd.NA] * n, dtype="Int64"), "near_dist_m": np.nan,
                        "near_pour_lon": np.nan, "near_pour_lat": np.nan})
    for i, (hid, plon, plat) in (lake or {}).items():
        out.loc[i, ["hylak_id", "near_hylak_id"]] = hid
        out.loc[i, ["lake_pour_lon", "lake_pour_lat", "lake_dist_m", "near_pour_lon", "near_pour_lat"]] = \
            [plon, plat, 0.0, plon, plat]
    for i, (hid, plon, plat) in (near or {}).items():
        out.loc[i, ["near_hylak_id"]] = hid
        out.loc[i, ["near_pour_lon", "near_pour_lat", "near_dist_m"]] = [plon, plat, 800.0]
    return out


def test_time_blocks():
    tb = SP.time_block(pd.to_datetime(["2015-07-01", "2021-12-31", "2022-01-01", "2023-12-31", "2024-01-01",
                                       "2025-12-31", "2026-03-01"]))
    assert list(tb) == ["train", "train", "val", "val", "test", "test", "test_b"]


def test_role_is_deterministic_and_fractions_reasonable():
    blocks = [f"{i}_{j}" for i in range(60) for j in range(60)]
    roles = [SP.role_of_block(b) for b in blocks]
    assert roles == [SP.role_of_block(b) for b in blocks]
    fr = pd.Series(roles).value_counts(normalize=True)
    assert abs(fr["test"] - 0.2) < 0.03 and abs(fr["val"] - 0.2) < 0.03


def test_lake_stations_share_role_even_far_apart():
    # two stations 350 km apart on one (fake) lake -> same anchor, block, role
    lat, lon = np.array([41.7, 42.6, 41.7]), np.array([-83.3, -79.5, -83.31])
    lk = _fake_lakes(lat, lon, {0: (9, -78.9, 42.9), 1: (9, -78.9, 42.9)})
    a = SP.assign(lon, lat, pd.to_datetime(["2019-01-01"] * 3).to_numpy(), lakes=lk)
    assert a.block[0] == a.block[1] and a.role[0] == a.role[1] and a.waterbody[0] == a.waterbody[1] == "L9"
    assert a.waterbody[2].startswith("P")


def test_protected_flags():
    lat, lon = np.full(3, 45.0), np.full(3, -93.0)
    a = SP.assign(lon, lat, pd.to_datetime(["2019-06-01", "2023-06-01", "2024-06-01"]).to_numpy(),
                  lakes=_fake_lakes(lat, lon))
    assert a.protected[2]
    assert a.protected[0] == (a.role[0] == "test") and a.protected[1] == (a.role[1] == "test")


def test_frozen_split_file_matches_code():
    v = SP.verify_split_file()
    assert v["sha256"] == "ffe288f2fafecada964fc8125dcd35ddd9b332d3a04f782d0298e01f097a5e32"
    assert SP.SPLIT_FILE_V1.exists() and SP.SPLIT_FILE_V2.exists()   # superseded versions kept for provenance
    with pytest.raises(FileExistsError):
        SP.write_split_file()


# ----------------------------------------------------------------------------------------- guard

def test_guard_refuses_protected_rows(tmp_path):
    prot = np.array([False, True])
    with pytest.raises(G.ProtectedDataError):
        G.check_rows(prot, training=True)
    with pytest.raises(G.ProtectedDataError):
        G.check_rows(prot, allow_test=False)
    with pytest.raises(G.ProtectedDataError):      # opt-in without a freeze file
        G.check_rows(prot, allow_test=True, freeze_file=tmp_path / "missing.json")
    bad = tmp_path / "freeze.json"
    bad.write_text(json.dumps({"created": "x", "split_sha256": "0" * 64, "scoring_code_sha256": "y",
                               "weights_sha256": {"m": "z"}, "comparators": {"chla": "gbm"}}))
    with pytest.raises(G.ProtectedDataError):      # wrong split hash
        G.check_rows(prot, allow_test=True, freeze_file=bad)
    assert G.check_rows(np.array([False, False]), training=True)["n_protected"] == 0


def test_drop_protected_and_cache_refusal(tmp_path, monkeypatch):
    df = pd.DataFrame({"protected": [False, True, False], "x": [1, 2, 3]})
    out, rec = G.drop_protected(df, log=lambda *a: None)
    assert list(out.x) == [1, 3] and rec["n_protected_dropped"] == 1
    with pytest.raises(G.ProtectedDataError):
        D.write_cache(df, {}, {}, version="__never__")


def test_freeze_file_absent_in_repo():
    # the test configuration is not frozen during development
    assert not G.FREEZE_FILE.exists()


# ------------------------------------------------------------------------------------ data / QC

def _chip(water_rc=((32, 32), (32, 33), (31, 32), (40, 40)), nodata_rc=((31, 32),)):
    L = C.S2_STORE_LAYERS
    chip = np.full((len(L), 64, 64), 500, np.int16)
    chip[L.index("SCL")] = 4
    for k, (r, c) in enumerate(water_rc):
        chip[L.index("SCL"), r, c] = 6
        for j, b in enumerate(D.BANDS):
            chip[L.index(b), r, c] = 100 + 10 * k + j
    for r, c in nodata_rc:
        chip[L.index("B05"), r, c] = S.NODATA_I16
    return chip


def test_chip_pixels_nearest_valid_water_only():
    refl, off, n, nw = D.chip_pixels(_chip(), p_max=8, win=15)
    assert n == 2 and nw == 2            # (31,32) has a nodata band; (40,40) outside the 15x15 window
    assert np.allclose(refl[0, :2], [0.0100, 0.0101])        # centre pixel first
    assert tuple(off[0]) == (0, 0) and np.isnan(refl[3:]).all()


def test_image_qc_ignores_labels():
    base = {"scl_c3_n_water": [9, 9], "jrc_water_ok": [True, True], "offset_suspect": ["", ""],
            "scl_c3_cloud_medium": [0, 0], "scl_c3_cloud_high": [0, 0.2], "scl_c3_thin_cirrus": [0, 0],
            "scl_c3_cloud_shadow": [0, 0], "rw3_B02": [0.03, 0.03], "rw3_B11": [0.01, 0.01],
            "same_acquisition": [True, True]}
    a = pd.DataFrame({**base, "chla": [1.0, 2.0]})
    b = pd.DataFrame({**base, "chla": [500.0, np.nan]})
    assert list(D.image_qc(a)) == list(D.image_qc(b)) == [True, False]


def test_valid_ranges_set_missing_not_clipped():
    df = pd.DataFrame({"chla": [0.01, 5.0, 4000.0], "tss": [1, 1, 1.0], "turbidity": [1, 1, 1.0], "secchi": [100, 1, 1.0]})
    c = D.apply_valid_ranges(df)
    assert c["chla"] == 2 and c["secchi"] == 1 and np.isnan(df.chla[0]) and df.chla[1] == 5.0


def test_inner_mask_whole_blocks_train_role_only():
    df = pd.DataFrame({"block": [f"b{i % 40}" for i in range(400)], "role": ["train"] * 300 + ["val"] * 100})
    m = F.inner_mask(df)
    assert not m[300:].any()
    per_block = pd.Series(m[:300]).groupby(df.block[:300].to_numpy()).nunique()
    assert (per_block == 1).all()


# ------------------------------------------------------------------------------------ metrics

def test_mdsa_matches_core_metric_and_paired_sign():
    rs = np.random.default_rng(0)
    y = rs.normal(1, 0.5, 300)
    p = y + rs.normal(0, 0.2, 300)
    assert np.isclose(E.mdsa_log10(p - y), M.mdsa(10 ** y, 10 ** p))
    q = y + rs.normal(0, 0.6, 300)
    r = E.paired_mdsa(y, p, q, groups=np.repeat(np.arange(30), 10), n_boot=200)
    assert r["diff"] < 0 and r["ci_high"] < 0


def test_mixture_quantiles_single_gaussian():
    from sentinel.models.orbit.satwq import mixture_quantiles

    q = mixture_quantiles(np.ones((2, 1)), np.array([[0.0], [1.0]]), np.array([[1.0], [0.5]]), (0.5, 0.8413))
    assert np.allclose(q[:, 0], [0, 1], atol=1e-2) and np.allclose(q[:, 1], [1, 1.5], atol=2e-2)


# ------------------------------------------------------------------------------------ model

def test_satwq_forward_band_subsets_and_single_pixel():
    import torch

    from sentinel.models.orbit.satwq import Batchable, SatWQNet

    rs = np.random.default_rng(0)
    n, P, B = 6, 8, len(D.BANDS)
    refl = rs.uniform(0, 0.1, (n, P, B)).astype(np.float32)
    refl[0, :, 9:] = np.nan
    ds = Batchable(refl, np.array([8, 1, 3, 8, 8, 2]), np.zeros((n, P, 2), np.float32),
                   np.tile(np.linspace(443, 2200, B, dtype=np.float32), (n, 1)), np.zeros((n, 7), np.float32),
                   np.zeros((n, 5), np.float32), np.zeros((n, 4), np.float32))
    m = SatWQNet(7, 5, d=16, n_layers=1, n_heads=2)
    xb, _, _ = ds.batch(np.arange(n), "cpu")
    logw, mu, sg = m(**xb)
    assert logw.shape == (n, 4, 5) and torch.isfinite(mu).all() and (sg > 0).all()
    rgbn = torch.tensor([b in D.RGBN for b in D.BANDS])
    xb["band_mask"] &= rgbn[None, None]
    assert torch.isfinite(m(**xb)[1]).all()
    xr, _, _ = ds.batch(np.arange(n), "cpu", train=True, rng=np.random.default_rng(1),
                        aug={"p_pixel_subset": 1.0, "p_rgbn": 0.5, "p_band_drop": 0.5})
    assert xr["pix_mask"].any(1).all()


def test_knn_baseline_ignores_non_fit_labels():
    from sentinel.models.orbit import baselines as BL

    rs = np.random.default_rng(0)
    df = pd.DataFrame({"lat": rs.uniform(40, 45, 200), "lon": rs.uniform(-95, -90, 200)})
    y = {t: rs.normal(0, 1, 200) for t in D.TARGETS}
    fit = np.arange(200) < 150
    y2 = {t: np.where(fit, v, 0.0) for t, v in y.items()}       # only non-fit labels differ
    p = [BL.fit_location_knn(df[fit], {t: v[fit] for t, v in yy.items()}, df[~fit],
                             {t: v[~fit] for t, v in yy.items()}, ks=(8,))[0](df[~fit])["chla"] for yy in (y, y2)]
    assert np.isfinite(p[0]).all() and np.array_equal(p[0], p[1])


def test_mdn_reimplementation_learns():
    from sentinel.models.orbit.mdn import MDNRegressor

    rs = np.random.default_rng(0)
    X = rs.uniform(0.005, 0.05, (400, 3))
    y = 1.5 * np.log10(X[:, 1] / X[:, 0]) + 0.5
    m = MDNRegressor(n_ens=2, n_iter=400, seed=0).fit(X, y)
    assert np.median(np.abs(m.predict(X) - y)) < np.median(np.abs(y - y.mean()))


def test_non_lake_units_nested_in_role_blocks():
    rs = np.random.default_rng(0)
    lat, lon = rs.uniform(30, 48, 3000), rs.uniform(-120, -70, 3000)
    a = SP.assign(lon, lat, pd.to_datetime(["2019-01-01"] * 3000).to_numpy(), lakes=_fake_lakes(lat, lon))
    assert (a.groupby("waterbody").block.nunique() == 1).all()
    assert (a.groupby("waterbody").role.nunique() == 1).all()


def test_official_mdn_parser_if_weights_present():
    from sentinel.core import paths
    from sentinel.models.orbit.mdn import OfficialMDN

    root = paths.RAW / "mdn_stream_weights" / "MSI" / "x2025"
    if not root.exists():
        pytest.skip("official MDN weights not downloaded")
    m = OfficialMDN(root)
    assert len(m.rounds) == 10 and m.rounds[0]["out"][0].shape == (100, 50)
    rrs = np.tile(np.array([[0.004, 0.006, 0.010, 0.005, 0.005, 0.002, 0.002]]), (3, 1))
    rrs[2, 3] = np.nan
    p = m.predict_log10(rrs)
    assert np.isfinite(p["chl"][:2]).all() and np.isnan(p["chl"][2]) and -2 < p["chl"][0] < 4


def test_buffer_excludes_points_next_to_other_role_lakes():
    # find a lake anchor whose role differs from the 200-km non-lake block of a nearby point
    lat0, lon0 = 45.0, -93.0
    for k in range(400):
        plat, plon = 40 + (k % 20) * 0.5, -100 + (k // 20) * 0.7
        lake_role = SP.role_of_block(SP.spatial_blocks([plat], [plon], SP.BLOCK_KM)[0])
        pt_role = SP.role_of_block("N" + SP.spatial_blocks([lat0], [lon0], SP.NONLAKE_BLOCK_KM)[0])
        if lake_role != pt_role:
            break
    lat, lon = np.array([lat0]), np.array([lon0])
    a = SP.assign(lon, lat, pd.to_datetime(["2019-01-01"]).to_numpy(), lakes=_fake_lakes(lat, lon, near={0: (7, plon, plat)}))
    assert a.buffer_excluded[0] and not a.in_lake[0]


def test_vendored_spatial_blocks_matches_shared():
    from sentinel.eval.splits import spatial_blocks as shared

    rs = np.random.default_rng(1)
    lat, lon = rs.uniform(25, 60, 500), rs.uniform(-130, 30, 500)
    for km in (10, 50, 200):
        assert (SP.spatial_blocks(lat, lon, km) == shared(lat, lon, km)).all()


def test_v3_out_of_polygon_lake_rows_buffered_when_estuary_role_differs():
    # a point captured by a lake within 500 m but outside its polygon, where the 200-km non-lake block role
    # differs from the lake role, must be buffer_excluded (review N1); inside the polygon it is a normal lake row
    lat0, lon0 = 27.79, -82.42
    nl_role = SP.role_of_block("N" + SP.spatial_blocks([lat0], [lon0], SP.NONLAKE_BLOCK_KM)[0])
    for k in range(400):
        plat, plon = 30 + (k % 20) * 0.5, -95 + (k // 20) * 0.7
        if SP.role_of_block(SP.spatial_blocks([plat], [plon], SP.BLOCK_KM)[0]) != nl_role:
            break
    lat, lon = np.array([lat0, lat0]), np.array([lon0, lon0])
    lk = _fake_lakes(lat, lon, {0: (11, plon, plat), 1: (11, plon, plat)})
    lk.loc[0, "lake_dist_m"] = 268.0
    a = SP.assign(lon, lat, pd.to_datetime(["2019-01-01"] * 2).to_numpy(), lakes=lk)
    assert a.in_lake.all() and a.buffer_excluded[0] and not a.buffer_excluded[1]


def test_turbidity_unit_policy_never_mixes_fnu_and_ntu():
    import numpy as np
    import pandas as pd

    from sentinel.models.orbit import data as D

    df = pd.DataFrame({"turbidity": [1.0, 2.0, 3.0, 4.0, 5.0],
                       "turbidity_unit": ["NTU", "FNU", "FNU|NTU", "NTRU", "NTU_or_FNU_unspecified"]})
    cnt = D.apply_turbidity_unit_policy(df)
    assert list(df.turbidity_unit_class) == ["NTU", "FNU", "mixed", "NTU", "unspecified"]
    assert np.isnan(df.turbidity[1]) and np.isnan(df.turbidity[2])          # FNU / mixed never fitted or scored
    assert df.turbidity_fnu[1] == 2.0 and np.isnan(df.turbidity_fnu[0])
    assert list(df.turbidity_all_units) == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert cnt == {"NTU": 2, "FNU": 1, "mixed": 1, "unspecified": 1}
    assert D.apply_turbidity_unit_policy(df) == {}                         # idempotent
