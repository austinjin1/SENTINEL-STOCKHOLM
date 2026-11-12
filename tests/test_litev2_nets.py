# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Networks and training code of M7 architecture round v2 (``litev2.nets``, ``litev2.fit``) on random tensors only:
no chip, label, teacher output or registry row is read."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from sentinel.models.litev2 import fit as FT
from sentinel.models.litev2 import labels2b as L
from sentinel.models.litev2 import nets as NN

torch.set_num_threads(4)
SIZE = 64


def _rows_chips(n=12, seed=0, size=SIZE):
    rng = np.random.default_rng(seed)
    chips = rng.integers(1, 255, (n, 4, size, size), dtype=np.uint8)
    chips[:, 3, : size // 2] = 0                                   # upper half: NIR < G → water
    chips[:, 3, size // 2:] = 255                                  # lower half: land
    rows = pd.DataFrame({"uid": [f"c{i}" for i in range(n)], "gsd_m": np.where(np.arange(n) % 2, 0.6, 1.0)})
    for j, t in enumerate(L.TARGETS):
        y = rng.normal(0.2 * j, 0.3, n)
        y[rng.random(n) < 0.4] = np.nan
        rows[f"y_{t}"] = y
        rows[f"cens_{t}"] = 0
        rows[f"lim_{t}"] = np.nan
    rows.loc[0, ["y_secchi", "cens_secchi", "lim_secchi"]] = [np.nan, 1, 0.5]      # right-censored
    rows.loc[1, ["y_tss", "cens_tss", "lim_tss"]] = [np.nan, -1, -0.3]             # left-censored
    rows["y_turb_ntru"] = np.nan
    rows.loc[2, "y_turb_ntru"] = 0.7
    sc = pd.DataFrame(rng.uniform(5, 200, (n, NN.N_SC)).astype(np.float32))
    sc.iloc[3, 5] = np.nan
    anchors = pd.DataFrame(rng.normal(0, 0.3, (n, len(L.TARGETS))), columns=list(L.TARGETS))
    return rows, chips, sc, anchors


def _kd(n, size=SIZE, k=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    h = size // 4
    return {"cell_map": torch.randint(-1, k, (n, h, h), generator=g).numpy(),
            "water_w": torch.rand(n, h, h, generator=g).numpy(),
            "t_mu": torch.randn(n, k, 4, generator=g).numpy(), "t_var": torch.rand(n, k, 4, generator=g).numpy() + 0.01,
            "t_w": torch.ones(n, k, 4).numpy()}


# ------------------------------------------------------------------------------------------------ nets
def test_anchor_lite_starts_at_the_anchor_and_has_the_planned_size():
    torch.manual_seed(0)
    m = NN.AnchorLite().eval()
    assert 1.3e6 < NN.count_parameters(m) < 1.5e6                  # amendment: about 1.4 M
    n, h = 3, SIZE // 4
    x, sc, a = torch.randn(n, 4, SIZE, SIZE), torch.randn(n, NN.N_SC) * 50 + 100, torch.randn(n, 6)
    pw = torch.full((n, h, h), 1.0 / (h * h))
    water = torch.zeros(n, h, h)
    water[0, :4], water[2] = 1.0, 0.4                              # chip 1 and 2: no water cell
    with torch.no_grad():
        out = m(x, sc, a, pw, water, torch.tensor([1.0, 0.6, 1.0]))
        aw = m.attention_weights(m.decoder(x), water, torch.tensor([1.0, 0.6, 1.0]))
    assert out["mu"].shape == (n, 6) and out["aux"].shape == (n, 8, h, h) and out["pooled"].shape == (n, 96)
    assert torch.equal(out["mu"], a) and float(out["delta"].abs().max()) == 0.0
    assert torch.allclose(aw.flatten(1).sum(1), torch.ones(n), atol=1e-5)
    assert float(aw[0, 4:].sum()) == 0.0                           # no weight outside the water cells
    assert bool((aw[1] > 0).all()) and bool((aw[2] > 0).all())     # no water cell: every cell is used


@pytest.mark.parametrize("kw,n_t,n_pool", [(dict(use_anchor=False), 6, 96), (dict(use_context=False), 6, 96),
                                           (dict(targets=tuple(L.ORIG)), 4, 96), (dict(attention=False), 6, 48)])
def test_ablation_switches(kw, n_t, n_pool):
    torch.manual_seed(0)
    m = NN.AnchorLite(**kw).eval()
    n, h = 2, SIZE // 4
    a = torch.randn(n, n_t)
    with torch.no_grad():
        m.res.base.fill_(0.25)
        out = m(torch.randn(n, 4, SIZE, SIZE), torch.randn(n, NN.N_SC), a, torch.full((n, h, h), 1.0 / (h * h)),
                torch.ones(n, h, h), torch.ones(n))
    assert out["mu"].shape == (n, n_t) and out["pooled"].shape == (n, n_pool)
    if kw.get("use_anchor", True):
        assert torch.equal(out["mu"], a)
    else:                                                          # A1: train median + Δ, the anchor is not read
        assert torch.allclose(out["mu"], torch.full((n, n_t), 0.25))
    if kw.get("use_context") is False:
        assert m.film is None and m.res.embed[0].in_features == n_t


def test_residual_is_bounded_by_kappa():
    h = NN.ResidualHeads(8, 4, L.TARGETS)
    with torch.no_grad():
        for hd in h.heads:
            hd[-1].bias[0] = 50.0
    out = h(torch.randn(5, 8), torch.randn(5, 4), torch.zeros(5, 6))
    k = NN.kappa_vector(L.TARGETS)
    assert float(k[0]) == pytest.approx(0.3) and bool((k[1:] == 0.5).all())
    assert torch.allclose(out["mu"], k.expand(5, -1), atol=1e-4)


def test_waterset_ignores_padding():
    torch.manual_seed(0)
    m = NN.WaterSet().eval()
    assert NN.count_parameters(m) < 0.5e6
    px, valid = torch.randn(2, 80, NN.N_PIXEL_FEATURES), torch.ones(2, 80, dtype=torch.bool)
    valid[:, 64:] = False
    sc, a = torch.randn(2, NN.N_SC), torch.randn(2, 6)
    px2 = px.clone()
    px2[:, 64:] = 99.0
    with torch.no_grad():
        o1, o2 = m(px, valid, sc, a), m(px2, valid, sc, a)
    assert torch.allclose(o1["pooled"], o2["pooled"], atol=1e-5) and torch.equal(o1["mu"], a)


@pytest.mark.parametrize("parity", [False, True])
def test_densenet_shapes_and_start(parity):
    torch.manual_seed(0)
    m = NN.DenseNetRegressor(parity=parity).eval()
    m.set_output_bias(torch.arange(6.0) / 10, torch.full((6,), 0.04))
    with torch.no_grad():
        out = m(torch.randn(2, 4, SIZE, SIZE), torch.randn(2, NN.N_SC) if parity else None)
    assert out["mu"].shape == (2, 6) and out["logvar"].shape == (2, 6)
    assert ("kd" in out) == parity and out["pooled"].shape[1] == (256 if parity else 1024)
    w = m.features.conv0.weight
    assert w.shape[1] == 4 and torch.allclose(w[:, 3], w[:, :3].mean(1))


# ------------------------------------------------------------------------------------------------ loss
def test_uncensored_terms_are_the_beta_nll():
    torch.manual_seed(0)
    mu, lv, y = torch.randn(7, 3), torch.randn(7, 3) * 0.3, torch.randn(7, 3)
    y[0, 0] = y[3, 1] = float("nan")
    y[:, 2] = float("nan")
    z = torch.zeros(7, 3, dtype=torch.int8)
    per_t, den = FT.target_terms(mu, lv, y, z, torch.full((7, 3), float("nan")), beta=0.5)
    assert den.tolist() == [6.0, 6.0, 0.0] and float(per_t[2]) == 0.0
    var = torch.exp(lv)
    ref = 0.5 * (lv + (y - mu) ** 2 / var) * var ** 0.5
    assert float(per_t[0]) == pytest.approx(float(ref[1:, 0].mean()), rel=1e-5)


def test_tobit_terms_point_the_right_way():
    lv = torch.zeros(1, 1)
    y = torch.full((1, 1), float("nan"))
    lim = torch.zeros(1, 1)

    def loss(mu, c):
        return float(FT.target_terms(torch.tensor([[mu]]), lv, y, torch.tensor([[c]], dtype=torch.int8), lim)[0][0])

    assert loss(-1.0, -1) < loss(0.0, -1) < loss(1.0, -1)          # left-censored (value < limit): low μ is right
    assert loss(1.0, 1) < loss(0.0, 1) < loss(-1.0, 1)             # right-censored (value > limit): high μ is right
    assert loss(0.0, -1) == pytest.approx(-math.log(0.5), rel=1e-5)
    assert loss(-8.0, -1) == pytest.approx(0.0, abs=1e-6)          # far inside the censored range: no loss


def test_ntru_rows_enter_the_ntu_head_through_the_offset():
    mu = torch.zeros(4, 2)
    lv = torch.zeros(4, 2)
    y = torch.full((4, 2), float("nan"))
    y[0, 1] = 0.0
    yn = torch.tensor([float("nan"), 0.4, 0.4, float("nan")])
    off = torch.zeros((), requires_grad=True)
    z = torch.zeros(4, 2, dtype=torch.int8)
    per_t, den = FT.target_terms(mu, lv, y, z, y, y_ntru=yn, ntru_offset=off, turb_index=1)
    assert den.tolist() == [0.0, 3.0]
    assert float(per_t[1]) == pytest.approx((0.0 + 2 * 0.5 * 0.16) / 3, rel=1e-5)
    per_t[1].backward()
    assert float(off.grad) < 0                                     # the offset moves towards the NTRU values
    best, _ = FT.target_terms(mu, lv, y, z, y, y_ntru=yn, ntru_offset=torch.tensor(0.4), turb_index=1)
    assert float(best[1]) == pytest.approx(0.0, abs=1e-7)
    none, d0 = FT.target_terms(mu, lv, y, z, y, y_ntru=torch.full((4,), float("nan")), ntru_offset=off, turb_index=1)
    assert d0.tolist() == [0.0, 1.0]


def test_total_loss_weights_and_terms():
    cfg = FT.arm_config("anchor_lite")
    assert FT.weights_vector(cfg["targets"], "cpu").tolist() == [1.0, 1.0, 1.0, 1.0, 0.5, 0.5]
    assert (cfg["epochs"], cfg["batch"], cfg["lr"], cfg["lr_backbone"], cfg["kd_weight"]) == (40, 32, 1e-3, 5e-4, 0.3)
    assert FT.arm_config("P3a_densenet")["epochs"] == 80 and FT.arm_config("anchor_set")["batch"] == 64
    assert FT.arm_config("anchor_set")["kd_weight"] == 0.0 and FT.arm_config("A4")["targets"] == tuple(L.ORIG)
    assert FT.arm_config("P3a_densenet", lr_backbone=3e-4)["lr_backbone"] == 3e-4
    with pytest.raises(KeyError):
        FT.arm_config("A2")

    class _M:
        ntru_offset = torch.zeros(())

    n = 5
    out = {"mu": torch.zeros(n, 6), "logvar": torch.zeros(n, 6), "delta": torch.full((n, 6), 2.0)}
    y = torch.full((n, 6), float("nan"))
    y[:, 0], y[:, 5] = 1.0, 1.0
    b = {"y": y, "cens": torch.zeros(n, 6, dtype=torch.int8), "lim": y * float("nan"),
         "y_ntru": torch.full((n,), float("nan"))}
    loss, parts = FT.total_loss(out, b, _M(), cfg)
    assert parts["label"] == pytest.approx(0.5 * 1.0 + 0.5 * 0.5) and parts["delta2"] == pytest.approx(4.0)
    assert float(loss) == pytest.approx(0.75 + 0.01 * 4.0)


# ------------------------------------------------------------------------------------------------ augmentation
def test_scene_statistics_follow_the_pixel_map():
    torch.manual_seed(0)
    cfg = {**FT.BASE_CFG, "aug_noise_dn": 0.0, "aug_p_resample": 0.0}
    x = torch.rand(6, 4, 32, 32) * 200 + 20
    sc = torch.zeros(6, NN.N_SC)
    q = torch.tensor([0.01, 0.02, 0.05, 0.25, 0.5, 0.75, 0.95, 0.98, 0.99])
    for b in range(4):
        flat = x[:, b].flatten(1)
        sc[:, 12 * b] = flat.mean(1)
        sc[:, 12 * b + 1] = flat.std(1)
        sc[:, 12 * b + 2:12 * b + 11] = torch.quantile(flat, q, dim=1).T
        sc[:, 12 * b + 11] = flat.median(1).values
    sc[:, 48:] = torch.rand(6, 8)
    sc[0, 3] = float("nan")
    y, p = FT.radiometric(x, cfg)
    out = FT.scene_under_map(sc, p)
    assert p["gamma"].shape == (6, 1) and p["gain"].shape == (6, 4) and p["offset"].shape == (6, 4)
    assert torch.equal(out[:, 48:], sc[:, 48:]) and math.isnan(float(out[0, 3]))
    for b in range(4):
        flat = y[:, b].flatten(1)
        got = out[:, 12 * b + 2:12 * b + 11]
        ref = torch.quantile(flat, q, dim=1).T
        ok = torch.isfinite(got)
        assert torch.allclose(got[ok], ref[ok], atol=0.05)         # a monotone map commutes with quantiles
        assert torch.allclose(out[:, 12 * b], flat.mean(1), rtol=0.02)
        assert torch.allclose(out[:, 12 * b + 1], flat.std(1), rtol=0.05)
    ident = {"gamma": torch.ones(6, 1), "gain": torch.ones(6, 4), "offset": torch.zeros(6, 4)}
    same = FT.scene_under_map(sc, ident)
    ok = torch.isfinite(sc)
    assert torch.allclose(same[ok], sc[ok], atol=1e-3)


def test_geometric_moves_every_map_together():
    a = torch.arange(16.0).view(1, 1, 4, 4)
    b = torch.arange(16.0).view(1, 4, 4)
    x, y, none = FT.geometric(1, True, a, b, None)
    assert none is None and torch.equal(x[0, 0], y[0]) and not torch.equal(x, a)
    assert sorted(x.flatten().tolist()) == a.flatten().tolist()


# ------------------------------------------------------------------------------------------------ tensors
def test_build_tensors_and_reduced_labels():
    rows, chips, sc, anchors = _rows_chips()
    t = FT.build_tensors(rows, chips, sc, anchors, L.TARGETS, with_set=True)
    h = SIZE // 4
    assert t.n == 12 and t.pool_w.shape == (12, h, h) and t.water.shape == (12, h, h)
    assert float(t.water[0, 0, 0]) == 1.0 and float(t.water[0, -1, -1]) == 0.0
    assert int((t.cens != 0).sum()) == 2 and bool(torch.isfinite(t.y_ntru[2]))
    assert t.water_px.shape == (12, SIZE, SIZE) and float(t.shore_px[0, 0, 0]) > float(t.shore_px[0, SIZE // 2 - 1, 0])
    r = FT.build_tensors(rows, chips, sc, anchors, L.ORIG, reduced=True)           # A4
    assert r.y.shape == (12, 4) and int((r.cens != 0).sum()) == 0 and not bool(torch.isfinite(r.y_ntru).any())
    assert not bool(torch.isfinite(r.lim).any())
    bad = rows.copy()
    bad.loc[0, "y_secchi"] = 0.1                                   # a censored entry with a value
    with pytest.raises(ValueError):
        FT.build_tensors(bad, chips, sc, anchors, L.TARGETS)
    with pytest.raises(ValueError):
        FT.build_tensors(rows, chips, sc, anchors.assign(secchi=np.nan), L.TARGETS)


def test_attach_kd_removes_masked_pairings():
    rows, chips, sc, anchors = _rows_chips()
    t = FT.build_tensors(rows, chips, sc, anchors, L.TARGETS)
    m = np.zeros(12, bool)
    m[[1, 5]] = True
    kd = _kd(12)
    info = FT.attach_kd(t, kd, m)
    assert info == {"n_chips_with_teacher_before_mask": 12, "n_kd_masked": 2, "n_chips_with_teacher": 10}
    assert float(t.kd["t_w"][[1, 5]].abs().sum()) == 0.0 and not bool(torch.isfinite(t.kd["t_mu"][[1, 5]]).any())
    assert bool(torch.isfinite(t.kd["t_mu"][0]).all()) and np.isfinite(kd["t_mu"][1]).all()   # the input is not edited
    assert t.kd["cell_map"].dtype == torch.int16


# ------------------------------------------------------------------------------------------------ ANCHOR-Set
def test_pixel_sampling_rules():
    wp = torch.zeros(3, 40, 40, dtype=torch.uint8)
    wp[0] = 1                                                      # 1,600 water pixels
    wp[1, 0, :10] = 1                                              # 10 water pixels in a corner
    g = torch.Generator().manual_seed(0)
    idx, valid = FT.sample_pixels(wp, torch.rand(3, 1600, generator=g))
    assert idx.shape == (3, NN.SET_MAX) and valid.sum(1).tolist() == [NN.SET_MAX, NN.SET_MIN, NN.SET_MIN]
    assert len(set(idx[0].tolist())) == NN.SET_MAX
    assert set(idx[1, :10].tolist()) == set(range(10))             # the water pixels come first
    rest = idx[1, 10:64]
    r = torch.sqrt((rest // 40 - 19.5) ** 2 + (rest % 40 - 19.5) ** 2)
    assert float(r.max()) < 6.0                                    # then the pixels nearest the chip centre
    r2 = idx[2, :64]
    assert float(torch.sqrt((r2 // 40 - 19.5) ** 2 + (r2 % 40 - 19.5) ** 2).max()) < 6.0
    idx_b, _ = FT.sample_pixels(wp, torch.rand(3, 1600, generator=g))
    assert not torch.equal(idx[0], idx_b[0])                       # resampled with new random numbers
    assert FT.set_seed("c1") == FT.set_seed("c1") != FT.set_seed("p2:c1")


def test_pixel_features_and_fixed_evaluation_sample():
    rows, chips, sc, anchors = _rows_chips(n=4)
    t = FT.build_tensors(rows, chips, sc, anchors, L.TARGETS, with_set=True)
    i = torch.arange(4)
    px, valid = FT._set_inputs(t, i, "cpu", None)
    px2, _ = FT._set_inputs(t, i, "cpu", None)
    assert px.shape == (4, NN.SET_MAX, NN.N_PIXEL_FEATURES) and bool(torch.isfinite(px).all())
    assert torch.equal(px, px2) and bool(valid.all())
    assert bool((px[..., 18] == 1).all())                          # 2,048 water pixels per chip: all sampled are water
    assert float(px[..., 12].max()) <= math.hypot(31.5, 31.5) * 1.0 + 1e-3        # distance to the centre in metres
    one = torch.log10(torch.tensor(float(chips[0, 0, 0, 0]) + 1.0))
    flat = FT.pixel_features(t.dn[:1], t.water_px[:1], t.shore_px[:1], t.gsd[:1], t.sc[:1, [6, 18, 30, 42]],
                             torch.zeros(1, 1, dtype=torch.long))
    assert float(flat[0, 0, 0]) == pytest.approx(float(one), rel=1e-5)
    assert float(flat[0, 0, 4]) == pytest.approx(float(torch.log10(torch.tensor(chips[0, 2, 0, 0] + 1.0))
                                                       - torch.log10(torch.tensor(chips[0, 0, 0, 0] + 1.0))), abs=1e-5)


# ------------------------------------------------------------------------------------------------ training
@pytest.mark.parametrize("arm", ["anchor_lite", "A1", "A3", "A4", "A5", "anchor_set", "P3a_densenet",
                                 "P3b_densenet_parity"])
def test_two_epochs_on_random_tensors(arm):
    cfg = FT.arm_config(arm, epochs=2, batch=4, warmup_epochs=1)
    rows, chips, sc, anchors = _rows_chips()
    a = anchors[list(cfg["targets"])]
    t = FT.build_tensors(rows, chips, sc, a, cfg["targets"], reduced=cfg["reduced"], with_set=cfg["net"] == "set")
    if cfg["net"] == "anchor":
        FT.attach_kd(t, _kd(12), np.arange(12) == 3)
    if arm == "P3b_densenet_parity":
        tm = torch.randn(12, 4)
        tm[3] = float("nan")
        t.teacher = tm
    lines = []
    model, hist, info = FT.train(cfg, t, seed=0, device="cpu", log=lines.append)
    assert len(hist) == 2 and all(math.isfinite(h["loss"]) for h in hist) and len(lines) == 2
    assert ("kd" in hist[0]) == (cfg["kd_weight"] > 0)
    assert ("delta2" in hist[0]) == (cfg["net"] != "densenet")
    p = FT.predict(model, cfg, t, device="cpu", batch=5)
    nt = len(cfg["targets"])
    assert p["mu"].shape == (12, nt) and p["sd"].shape == (12, nt) and np.isfinite(p["mu"]).all() and (p["sd"] > 0).all()
    p2 = FT.predict(model, cfg, t, device="cpu", batch=12)
    assert np.allclose(p["mu"], p2["mu"], atol=1e-4)               # evaluation does not depend on the batching
    d = FT.prediction_dict(cfg, p["mu"])
    assert set(d) == (set(L.ORIG) | {"turb_fnu"} if arm == "A4" else set(L.TARGETS))
    if arm == "A4":
        assert np.array_equal(d["turb_fnu"], d["turbidity"])
    if cfg["net"] != "densenet" and cfg["use_anchor"]:
        k = NN.kappa_vector(cfg["targets"]).numpy()
        assert (np.abs(p["mu"] - a.to_numpy(np.float32)) <= k + 1e-5).all()


def test_ensemble_rule():
    mus = [np.array([[1.0]]), np.array([[3.0]])]
    sds = [np.array([[1.0]]), np.array([[1.0]])]
    mu, sd = FT.ensemble(mus, sds)
    assert mu[0, 0] == 2.0 and sd[0, 0] == pytest.approx(math.sqrt(1.0 + 1.0))
