# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Unit tests for SENTINEL-Lite (M7). Synthetic arrays only (allowed in unit tests, R9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from sentinel.models.lite import augment as A
from sentinel.models.lite import baselines as B
from sentinel.models.lite import data as D
from sentinel.models.lite import distill as KD
from sentinel.models.lite import evaluate as E
from sentinel.models.lite import features as F
from sentinel.models.lite import guard as G
from sentinel.models.lite import splits as S
from sentinel.models.lite.losses import distill_nll, gaussian_nll, pool_maps


def _samples():
    return pd.DataFrame({
        "target": ["secchi", "secchi", "turbidity", "turbidity", "chla", "tss", "secchi", "ssc"],
        "unit": ["ft", "cm", "NTU", "FNU", "mg/m3", "ppm", "m", "mg/L"],
        "value": [10.0, 150.0, 5.0, 5.0, 20.0, 7.0, 99.0, 3.0],
        "censored": [False] * 8,
    })


def test_harmonise_units_and_bounds():
    d, rep = D.harmonise(_samples())
    sec = d[d.target == "secchi"]["value_h"].tolist()
    assert sec == pytest.approx([3.048, 1.5])          # 99 m is out of bounds
    assert rep["dropped_bounds"]["secchi"] == 1
    assert d[d.target == "turbidity"]["value_h"].tolist() == [5.0]   # FNU dropped, never converted
    assert rep["dropped_unit"]["turbidity"] == 1
    assert "ssc" not in set(d.target)
    assert d[d.target == "tss"]["value_h"].tolist() == [7.0]


def test_trophic_classes_carlson_edges():
    assert D.trophic_class([1.0, 2.6, 7.29, 7.3, 55.9, 56.0, np.nan]).tolist() == [0, 1, 1, 2, 2, 3, -1]


def _lab(n_wb=60, seed=0):
    rs = np.random.default_rng(seed)
    rows = []
    for w in range(n_wb):
        lat, lon = 44 + rs.uniform(0, 4), -95 + rs.uniform(0, 6)
        for k in range(3):
            rows.append({"naip_chip_id": f"c{w}_{k}", "site_id": f"s{w}_{k % 2}", "lat": lat + 0.001 * k,
                         "lon": lon, "water_body": f"wb{w}", "naip_year": int(rs.choice([2015, 2019, 2023])),
                         "site_class": "lake_reservoir", "y_secchi": 0.1, "y_turbidity": np.nan, "y_tss": np.nan,
                         "y_chla": 1.0})
    return pd.DataFrame(rows)


def test_split_disjoint_buffer_and_year_rules():
    lab = _lab()
    sp = S.build_split(lab, block_km=50, buffer_km=5, seed=1)
    role = lab["naip_chip_id"].map(sp["chip_role"])
    assert set(role) <= set(S.ROLES)
    # year rule
    assert (lab.loc[role.isin(["train", "val"]), "naip_year"] <= 2021).all()
    assert (role[(lab.naip_year > 2021) & ~role.isin(["test", "excluded"])] == "test_time").all()
    # water bodies never split across train/val/test
    for wb, g in lab.assign(r=role).groupby("water_body"):
        assert len(set(g.r) & {"train", "val", "test"}) <= 1
    assert S.split_sha256(sp) == S.split_sha256(S.build_split(lab, block_km=50, buffer_km=5, seed=1))


def test_check_split_detects_leak():
    lab = _lab()
    sp = S.build_split(lab, block_km=50, buffer_km=5, seed=1)
    tr = [c for c, r in sp["chip_role"].items() if r == "train"][0]
    te = [c for c, r in sp["chip_role"].items() if r == "test"][0]
    lab2 = lab.copy()
    lab2.loc[lab2.naip_chip_id == te, "water_body"] = lab2.loc[lab2.naip_chip_id == tr, "water_body"].iloc[0]
    with pytest.raises(AssertionError):
        S.check_split(sp, lab2)


def test_guard_blocks_protected_roles(tmp_path):
    with pytest.raises(G.ProtectedDataError):
        G.assert_no_protected(["train", "test"])
    with pytest.raises(G.ProtectedDataError):
        G.check_access(["test_time"], training=True)
    with pytest.raises(G.ProtectedDataError):
        G.check_access(["test"], allow_test=True, freeze_file=tmp_path / "missing.json")
    assert G.check_access(["train", "val"])["protected"] == []


def test_frozen_split_hash_matches_file():
    if not G.SPLIT_FILE.exists():
        pytest.skip("split not built")
    sp = G.load_split()
    assert sp["_sha256"] == G.FROZEN_SPLIT_SHA256


def test_pooling_weights_centred_and_normalised():
    chips = np.zeros((2, 4, 64, 64), np.uint8)
    chips[:, 1] = 50        # G
    chips[:, 3] = 10        # NIR < G -> water everywhere
    chips[1, 3, :, :32] = 100   # left half land on chip 1
    w = F.pooling_weights(chips, np.array([1.0, 1.0]), stride=4)
    assert np.allclose(w.sum((1, 2)), 1.0)
    assert w[0, 8, 8] > 0 and w[0, 0, 0] == 0
    assert w[1, :, :8].sum() == 0      # land cells get no weight


def test_gaussian_nll_ignores_missing():
    mu = torch.zeros(4, 2)
    var = torch.ones(4, 2)
    y = torch.tensor([[0.0, 100.0]] * 4)
    m = torch.tensor([[True, False]] * 4)
    assert float(gaussian_nll(mu, var, y, m, beta=0)) == pytest.approx(0.0)


def test_pool_maps():
    out = torch.zeros(1, 4, 4, 4)
    out[0, 0, 1, 1] = 2.0
    w = torch.zeros(1, 4, 4)
    w[0, 1, 1] = 0.5
    w[0, 2, 2] = 0.5
    mu, var = pool_maps(out, w, 2)
    assert float(mu[0, 0]) == pytest.approx(1.0)
    assert float(var[0, 0]) == pytest.approx(1.0)


def test_distill_block_means():
    T = 2
    out = torch.zeros(1, 2 * T, 2, 2)
    out[0, 0] = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
    cm = torch.tensor([[[0, 0], [1, 1]]])
    ww = torch.ones(1, 2, 2)
    t_mu = torch.tensor([[[2.0, np.nan], [6.0, np.nan]]])
    t_var = torch.ones(1, 2, T)
    t_w = torch.ones(1, 2, T)
    l_match = float(distill_nll(out, cm, ww, t_mu, t_var, t_w, T, 0.0))
    t_mu2 = t_mu.clone()
    t_mu2[0, 0, 0] = 4.0
    assert float(distill_nll(out, cm, ww, t_mu2, t_var, t_w, T, 0.0)) > l_match
    assert l_match == pytest.approx(0.0)


def test_cell_map_same_crs():
    # NAIP chip at 1 m with left/top on the S2 grid: each 10x10 m block of pixels maps to one S2 cell
    cm = KD.cell_map_for_chip("EPSG:32615", 1000.0, 2000.0, 1.0, "EPSG:32615", 1000.0, 2000.0, size=40, stride=4)
    assert cm.shape == (10, 10)
    assert cm[0, 0] == 0 and cm[0, 3] == 1 and cm[3, 0] == 64
    cm2 = KD.cell_map_for_chip("EPSG:32615", 0.0, 2000.0, 1.0, "EPSG:32615", 1000.0, 2000.0, size=40, stride=4)
    assert (cm2 == -1).all()


def test_perturbations():
    x = torch.full((2, 4, 64, 64), 100.0)
    assert torch.equal(A.perturb(x, "clean", 0), x)
    g = A.perturb(x, "gain", 0.05, seed=3)
    assert torch.allclose(g[0, :, 0, 0].sub(100).abs(), torch.full((4,), 5.0))
    h = A.perturb(x, "haze", 10)
    assert float(h[0, 2, 0, 0]) == pytest.approx(110.0)
    gl = A.perturb(x, "glint", 0.25, seed=1)
    frac = float((gl[:, 0] > 100).float().mean())
    assert 0.15 < frac < 0.35
    assert A.perturb(x, "resolution", 4).shape == x.shape
    xr = x.clone()
    xr[:, 3, :, 0] = 0
    assert float(A.perturb(xr, "misreg", 1)[0, 3, 0, 1]) == 0.0


def test_model_shapes_and_budget():
    pytest.importorskip("timm")
    from sentinel.models.lite.model import ExportWrapper, LiteStudent, count_macs, count_params

    m = LiteStudent("mnv4s", pretrained=False)
    out = m(torch.rand(1, 4, 128, 128))
    assert out.shape == (1, 8, 32, 32)
    assert count_params(m) <= 3_000_000
    assert count_macs(ExportWrapper(m)) <= 2e9


def test_metrics():
    y = np.log10([1.0, 2.0, 4.0])
    assert E.mdsa_log(y, y) == pytest.approx(0.0)
    assert E.mdsa_log(y, y + np.log10(2)) == pytest.approx(100.0)
    assert E.balanced_accuracy([0, 0, 1, 1], [0, 1, 1, 1]) == pytest.approx(0.75)
    assert E.qw_kappa([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(1.0)


def test_band_ratio_recovers_linear_form():
    rs = np.random.default_rng(0)
    Fr = pd.DataFrame({"c9s_B": rs.uniform(0.5, 2, 200), "c9s_R": rs.uniform(0.5, 2, 200),
                       "c9s_G": rs.uniform(0.5, 2, 200), "c9s_NIR": rs.uniform(0.1, 1, 200)})
    y = 0.3 + 0.5 * Fr.c9s_B / Fr.c9s_R - 0.2 * Fr.c9s_B
    p, info = B.band_ratio(Fr, y.to_numpy(), Fr, "secchi")
    assert np.allclose(p, y, atol=1e-8)


def test_distill_kmax_and_per_target_trust():
    T = 2
    out = torch.zeros(1, 2 * T, 2, 2)
    cm = torch.tensor([[[0, 1], [2, -1]]])
    ww = torch.ones(1, 2, 2)
    t_mu = torch.zeros(1, 3, T)
    t_var = torch.ones(1, 3, T)
    t_w = torch.zeros(1, 3, T)
    t_w[0, :, 0] = 1.0                       # target 1 untrusted everywhere
    t_mu[0, :, 1] = 100.0                    # would dominate if (wrongly) used
    assert float(distill_nll(out, cm, ww, t_mu, t_var, t_w, T, 0.0)) == pytest.approx(0.0)


def test_water_body_name_normalisation():
    f = D.water_body_name
    assert f("Green Lake - West Basin - Deep Hole (USGS 434756089020500)") == "green"
    assert f("Mississippi River 701.1B") == f("MISSISSIPPI R OFF OF BOOM ISLAND") == "mississippi river"
    assert f("St Croix River upstream of Hwy 243") == f("St. Croix River Just Upstream from X") == "st croix river"
    assert f("EAST") == "" and f("L006") == ""


def test_water_body_groups_merge_rules():
    sites = pd.DataFrame({"site_id": ["a", "b", "c", "d", "e"], "lat": [45.0, 45.2, 46.0, 46.3, 47.0],
                          "lon": [-93.0, -93.0, -94.0, -94.0, -95.0],
                          "name": ["Lake Foo - N", "FOO (S BASIN)", "X River 1", "X River 9", "Bar"],
                          "site_class": ["lake_reservoir", "lake_reservoir", "river_stream", "river_stream",
                                         "lake_reservoir"], "huc8": ["1", "1", "2", "3", "4"]})
    g = D.water_body_groups(sites)
    assert g["a"] == g["b"]                  # same name, 22 km apart
    assert g["c"] == g["d"]                  # same river name, 33 km
    assert g["e"] not in {g["a"], g["c"]}


def test_export_wrapper_rgb_only_matches_training_path():
    pytest.importorskip("timm")
    from sentinel.models.lite.model import ExportWrapper, LiteStudent
    from sentinel.models.lite.train import normalise_input

    m = LiteStudent("mnv4s", pretrained=False, spectral=True).eval()
    dn = torch.rand(1, 4, 64, 64) * 255
    x, x01 = normalise_input(dn, torch.ones(1, 4, 2), m, "raw", rgb_only=True)
    with torch.no_grad():
        a = m(x, x01)[:, :4] * m.y_std[None, :, None, None] + m.y_mean[None, :, None, None]
        b = ExportWrapper(m, rgb_only=True)(dn / 255.0)[:, :4]
    assert torch.allclose(a, b, atol=1e-5)


def test_film_and_crop():
    pytest.importorskip("timm")
    from sentinel.models.lite.model import ExportWrapper, LiteStudent
    from sentinel.models.lite.train import centre_crop

    m = LiteStudent("mnv4s", pretrained=False, film=True).eval()
    x01 = torch.rand(2, 4, 64, 64)
    assert ExportWrapper(m)(x01).shape == (2, 8, 16, 16)
    dn, w = centre_crop(torch.zeros(2, 4, 256, 256), torch.ones(2, 64, 64) / 4096, 128)
    assert dn.shape[-1] == 128 and w.shape[-1] == 32 and torch.allclose(w.sum((1, 2)), torch.ones(2))


def test_teacher_overlap_check_and_litesafe_drop(monkeypatch):
    """W8: synthetic toy sites (test only). Site/distance/lake criteria flag overlap; the drop mask removes it."""
    monkeypatch.setattr(KD, "_orbit_lake", lambda lon, lat: np.array(["L7" if x > -90 else "P1|2" for x in lon]))
    sites = pd.DataFrame({"site_id": ["a", "b", "c", "d"], "role": ["test", "test", "val", "train"],
                          "lat": [45.0, 46.0, 47.0, 48.0], "lon": [-95.0, -95.0, -89.0, -95.0],
                          "water_body": ["w1", "w2", "w3", "w4"], "hylak_id": [np.nan, np.nan, np.nan, np.nan],
                          "site_class": "lake_reservoir", "n_chips": [2, 1, 1, 3]})
    seen = pd.DataFrame({"source": "matchup", "key": ["r1", "r2", "r3", "r4"], "site_id": ["a", "x", "y", "z"],
                         "lat": [40.0, 46.01, 30.0, 48.0], "lon": [-80.0, -95.0, -80.0, -95.0],
                         "orbit_wb": ["P0|0", "P0|1", "L7", "P0|3"]})
    f = KD.flag_overlap(sites, seen)
    assert f.set_index("site_id")[["ov_site", "ov_dist", "ov_lake"]].to_dict("index") == {
        "a": {"ov_site": True, "ov_dist": False, "ov_lake": False},
        "b": {"ov_site": False, "ov_dist": True, "ov_lake": False},
        "c": {"ov_site": False, "ov_dist": False, "ov_lake": True},
        "d": {"ov_site": False, "ov_dist": True, "ov_lake": False}}
    rep = KD.overlap_report(f)
    assert rep["test"]["water_bodies_overlap"] == 2 and rep["test"]["chips_in_overlap_wb"] == 3
    with pytest.raises(G.ProtectedDataError):
        KD.assert_teacher_disjoint(f, ("test", "test_time"))
    drop = KD.litesafe_drop_mask(seen, f[f.role.isin(["val", "test"])])
    assert drop.tolist() == [True, True, True, False]          # r4 is only near a train site: kept
    f2 = KD.flag_overlap(sites, seen[~drop])
    KD.assert_teacher_disjoint(f2, ("test", "test_time", "val"))
    assert f2.set_index("site_id").loc["d", "overlap"]


def test_kd_crop_window_matches_centre_crop():
    """The KD-map crop in train() uses the same window as centre_crop at stride 4."""
    from sentinel.models.lite.train import centre_crop

    dn = torch.zeros(1, 4, 256, 256)
    w = torch.arange(64 * 64, dtype=torch.float32).reshape(1, 64, 64) + 1
    _, wc = centre_crop(dn, w, 128)
    c0 = (256 - 128) // 2 // 4
    cm = w[:, c0:c0 + 32, c0:c0 + 32]
    assert torch.allclose(wc * cm.sum(), cm)


def test_pool_maps_median_robust_to_outliers():
    from sentinel.models.lite.losses import pool_maps_median

    out = torch.zeros(1, 2, 4, 4)
    out[0, 0] = 1.0
    out[0, 0, 0, :3] = 50.0                   # 3 of 16 equally weighted pixels corrupted
    w = torch.full((1, 4, 4), 1 / 16)
    mu_mean, _ = pool_maps(out, w, 1)
    mu_med, var = pool_maps_median(out, w, 1)
    assert abs(float(mu_med) - 1.0) < 1e-6 and float(mu_mean) > 5
    assert torch.allclose(var, torch.ones(1, 1))


def test_glint_aug_adds_bright_blobs_only_to_selected():
    x = torch.full((8, 4, 64, 64), 20.0)
    y = A.glint_aug(x, torch.Generator().manual_seed(0), p=0.5)
    changed = (y != x).flatten(1).any(1)
    assert 0 < int(changed.sum()) < 8
    assert float(y.min()) >= 20.0 and float(y.max()) <= 255.0
