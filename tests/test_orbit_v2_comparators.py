# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Parity comparator building blocks (amendment §2.5, §3.5; lead rulings R-10 to R-14b) on synthetic arrays."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from sentinel.models.orbit import baselines as BL
from sentinel.models.orbit.data import BANDS, TARGETS
from sentinel.models.orbit_v2 import chipfeat as CF
from sentinel.models.orbit_v2 import comparators as C
from sentinel.models.orbit_v2 import context as CX


def _arrays(n=5, m=4, seed=0):
    rs = np.random.default_rng(seed)
    cidx = np.full((n, CX.K), -1)
    cdt = np.full((n, CX.K), np.nan, np.float32)
    cidx[0, :3], cdt[0, :3] = [2, 0, 3], [-5, 9, 30]
    cidx[1, :1], cdt[1, :1] = [1], [12]
    feats = {"cidx": cidx, "cdt": cdt, "ring": rs.normal(size=(n, CF.RING_DIM)), "glint": rs.normal(size=(n, 3)),
             "dp_m": np.abs(rs.normal(50, 20, size=(n, 6)))}
    carrays = {"rw3": rs.uniform(0.01, 0.1, size=(m, len(BANDS))), "q": rs.normal(size=(m, C.N_Q))}
    return feats, carrays


def test_folds_are_grouped_by_role_block_and_fixed_by_the_block_ids():
    blocks = np.array([f"b{i % 23}" for i in range(400)])
    f = C.role_block_folds(blocks)
    assert set(f) == set(range(5)) and C.N_FOLDS == 5
    assert pd.Series(f).groupby(blocks).nunique().max() == 1                # a block is in one fold
    p = np.random.default_rng(0).permutation(400)
    assert np.array_equal(C.role_block_folds(blocks[p]), f[p])              # row order and counts play no part
    assert np.array_equal(C.role_block_folds(np.unique(blocks)), pd.Series(f).groupby(blocks).first().to_numpy())
    with pytest.raises(C.ComparatorError):
        C.role_block_folds(["a", "b", "c", "d"])


def test_out_of_fold_never_fits_on_the_predicted_blocks():
    blocks = np.array([f"b{i % 10}" for i in range(100)])
    folds = C.role_block_folds(blocks)
    use = np.arange(100) % 3 != 0
    seen = []

    def fp(tr, te):
        assert not set(blocks[tr]) & set(blocks[te]) and use[tr].all()
        seen.append(len(te))
        return np.full(len(te), float(folds[te[0]]))
    out = C.out_of_fold(fp, folds, use)
    assert np.array_equal(out, folds.astype(float)) and sum(seen) == 100 and len(seen) == 5


def test_context_summaries_and_date_slots():
    feats, carr = _arrays()
    rw3, q, dt = C.slot_arrays(feats["cidx"], feats["cdt"], carr)
    assert rw3.shape == (5, 6, 12) and q.shape == (5, 6, 8) and dt.shape == (5, 6)
    assert np.array_equal(rw3[0, 1], carr["rw3"][0]) and np.isnan(rw3[0, 3:]).all() and np.isnan(rw3[2]).all()
    med, iqr = C.context_band_summary(rw3)
    assert np.allclose(med[0], np.median(carr["rw3"][[2, 0, 3]], 0))
    q75, q25 = np.percentile(carr["rw3"][[2, 0, 3]], [75, 25], axis=0)
    assert np.allclose(iqr[0], q75 - q25)
    assert np.allclose(med[1], carr["rw3"][1]) and np.all(iqr[1] == 0)      # one context chip: IQR 0
    assert np.isnan(med[2]).all() and np.isnan(iqr[2]).all()               # no context: NaN, not zero
    X = C.context_date_features(feats, carr)                                # R-13
    assert X.shape == (5, 6 * (12 + 8 + 1))
    assert list(X.columns[:21]) == [f"s0_rw3_{b}" for b in BANDS] + [f"s0_q{j}" for j in range(8)] + ["s0_dt"]
    assert X.loc[0, "s0_dt"] == -5 and X.loc[0, "s1_dt"] == 9 and X.loc[0, "s2_dt"] == 30    # the stored order
    assert np.allclose(X.loc[0, [f"s1_rw3_{b}" for b in BANDS]].to_numpy(float), carr["rw3"][0])
    assert np.allclose(X.loc[0, [f"s2_q{j}" for j in range(8)]].to_numpy(float), carr["q"][3])
    assert X.loc[0, [c for c in X.columns if c.startswith("s3_")]].isna().all() and X.loc[3].isna().all()
    with pytest.raises(C.ComparatorError):
        C.slot_arrays(feats["cidx"][:, :4], feats["cdt"][:, :4], carr)
    empty = {"rw3": np.zeros((0, 12)), "q": np.zeros((0, 8))}
    with pytest.raises(C.ComparatorError):
        C.slot_arrays(feats["cidx"], feats["cdt"], empty)
    none = np.full((3, 6), -1)
    assert np.isnan(C.slot_arrays(none, np.full((3, 6), np.nan), empty)[0]).all()


def test_ring_context_features_have_no_count_feature():
    feats, carr = _arrays()
    n_pix = np.array([6, 3, 1, 0, 4])
    X = C.ring_context_features(feats, n_pix, carr)
    assert X.shape == (5, CF.RING_DIM + 3 + 2 + 24)
    assert not [c for c in X.columns if "count" in c or c.startswith("n_")]
    assert X.loc[1, "dp_median"] == pytest.approx(np.median(feats["dp_m"][1, :3]))      # valid pixels only
    assert X.loc[1, "dp_min"] == pytest.approx(feats["dp_m"][1, :3].min())
    assert np.isnan(X.loc[3, "dp_median"]) and np.isnan(X.loc[3, "dp_min"])
    v1 = pd.DataFrame({"a": np.arange(5.0)}, index=[7, 8, 9, 10, 11])
    assert list(C.gbm_feature_set("v1", v1, feats, n_pix, carr).columns) == ["a"]
    full = C.gbm_feature_set("v1+ring+context", v1, feats, n_pix, carr)
    assert full.shape == (5, 1 + X.shape[1]) and full["a"].notna().all()
    dates = C.gbm_feature_set("v1+ring+context_dates", v1, feats, n_pix, carr)
    assert not [c for c in dates.columns if c.startswith("ctx_")] and "s5_dt" in dates and "ring_0" in dates
    assert dates.shape == (5, 1 + CF.RING_DIM + 3 + 2 + 126)
    with pytest.raises(C.ComparatorError):
        C.gbm_feature_set("v1+teacher", v1, feats, n_pix, carr)


def test_stacking_adds_the_other_targets_only():
    X = pd.DataFrame({"a": np.arange(4.0)})
    others = {"chla": np.ones(4), "secchi": np.zeros(4), "tp": np.full(4, 2.0)}
    S = C.stack_features(X, others, "tp")
    assert list(S.columns) == ["a", "stack_chla", "stack_secchi"]
    with pytest.raises(C.ComparatorError):
        C.stack_features(X, {"tp": np.ones(4)}, "tp")
    with pytest.raises(C.ComparatorError):
        C.stack_features(X, {"chla": np.ones(3)}, "tp")


def test_mdn_inputs_and_the_fit_row_fill():
    feats, carr = _arrays()
    rw3 = np.random.default_rng(1).uniform(0.01, 0.1, size=(5, 12))
    x7, _ = C.mdn_inputs("7band", rw3, feats, carr)
    assert np.array_equal(x7, rw3[:, :7]) and C.MDN_BANDS7 == tuple(BANDS[:7])
    assert np.array_equal(C.mdn_inputs("12band", rw3, feats, carr)[0], rw3)
    x, fill = C.mdn_inputs("12band+context+ring", rw3, feats, carr)
    assert x.shape == (5, 12 + 12 + CF.RING_DIM) and np.isfinite(x).all()
    med = np.stack([np.median(carr["rw3"][[2, 0, 3]], 0), carr["rw3"][1]])
    assert np.allclose(x[2, 12:24], np.median(med, 0))                      # no context: the fit-row median
    f2 = {k: v[2:] if k not in ("ring",) else v[2:] for k, v in feats.items()}
    x2, fill2 = C.mdn_inputs("12band+context+ring", rw3[2:], f2, carr, fill=fill)
    assert fill2 is fill and np.allclose(x2[:, 12:24], np.median(med, 0)[None])
    hole = rw3.copy()
    hole[4, 1] = np.nan                                                     # a band without a water pixel
    for inp in C.MDN_INPUTS:
        xh, fh = C.mdn_inputs(inp, hole, feats, carr)
        assert np.isfinite(xh).all() and np.isclose(xh[4, 1], np.median(rw3[:4, 1]))
        assert np.isclose(C.mdn_inputs(inp, hole[4:], {k: v[4:] for k, v in feats.items()},
                                       carr, fill=fh)[0][0, 1], np.median(rw3[:4, 1]))
    with pytest.raises(C.ComparatorError, match="another input"):
        C.mdn_inputs("12band", rw3, feats, carr, fill=C.mdn_inputs("7band", rw3, feats, carr)[1])
    with pytest.raises(C.ComparatorError):
        C.mdn_inputs("9band", rw3, feats, carr)


def test_censoring_variants_of_r14b():
    y = np.array([1.0, np.nan, 0.5, np.nan])
    cens = np.array([False, True, False, False])
    dl = np.array([np.nan, 0.3, np.nan, np.nan])
    a = C.censoring_variant("dl_sqrt2", y, cens, dl)
    assert a[0] == 1.0 and a[1] == pytest.approx(math.log10(10 ** 0.3 / math.sqrt(2))) and np.isnan(a[3])
    b = C.censoring_variant("quantified", y, cens, dl)
    assert np.array_equal(np.isnan(b), [False, True, False, True]) and C.CENSORING == ("dl_sqrt2", "quantified")
    with pytest.raises(C.ComparatorError):
        C.censoring_variant("half_dl", y, cens, dl)


def test_knn_is_the_v1_formula():
    rs = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({"lat": rs.uniform(40, 45, n).round(1), "lon": rs.uniform(-90, -80, n).round(1)})
    y = {t: rs.normal(size=n) for t in TARGETS}
    y["chla"][::7] = np.nan
    dq = pd.DataFrame({"lat": rs.uniform(40, 45, 50), "lon": rs.uniform(-90, -80, 50)})
    v1, _ = BL.fit_location_knn(df, y, df, y)
    mine = C.fit_knn(df.lat, df.lon, y["chla"])
    for k in C.KNN_KS:
        assert np.allclose(mine(dq.lat, dq.lon, k), v1(dq, k=k)["chla"])
    assert C.KNN_KS == (4, 8, 16, 32)
    with pytest.raises(C.ComparatorError):
        C.fit_knn(df.lat, df.lon, np.full(n, np.nan))


def test_gbm_family_members_and_bags():
    mem = C.gbm_members()
    assert len(mem) == 16 * 3 * 2 and len({C.member_name(*m) for m in mem}) == 96
    assert C.GBM_MEMBER_SEEDS == (0, 1, 2) and len(C.GBM_SELECTED_SEEDS) == 9 and C.GBM_GRID is BL.GBM_GRID
    assert C.member_name("v1", "single", 3, "quantified") == "gbm[v1|single|g03|quantified]"
    rs = np.random.default_rng(0)
    X = rs.normal(size=(400, 5))
    y = X[:, 0] + 0.1 * rs.normal(size=400)
    y[::9] = np.nan
    p = C.fit_gbm_bag(X[:300], y[:300], X[300:], y[300:], C.GBM_GRID[0], n_threads=2, max_rounds=60, patience=10)
    assert len(p.n_rounds) == 3 and p.seeds == [0, 1, 2]
    pred = p(X[300:])
    assert pred.shape == (100,) and np.corrcoef(pred, X[300:, 0])[0, 1] > 0.8
    with pytest.raises(C.ComparatorError):
        C.fit_gbm_bag(X[:30], y[:30], X[300:], y[300:], C.GBM_GRID[0])


def test_average_chain_and_selection():
    assert np.allclose(C.average_log10([1.0, 2.0], [3.0, 2.0]), [2.0, 2.0])
    with pytest.raises(C.ComparatorError):
        C.average_log10([1.0], [1.0, 2.0])
    rs = np.random.default_rng(0)
    zc, zs = rs.normal(size=200), rs.normal(size=200)
    y = 0.3 + 0.8 * zc - 0.5 * zs
    y[:5] = np.nan
    zc[5:8] = np.nan
    coef = C.ols_fit(zc, zs, y)
    assert np.allclose(coef, [0.3, 0.8, -0.5])
    assert np.allclose(C.ols_predict(coef, [1.0], [2.0]), [0.3 + 0.8 - 1.0])
    with pytest.raises(C.ComparatorError):
        C.ols_fit(zc[:12], zs[:12], y[:12])
    assert C.select_member({"a": 30.0, "b": 25.0, "c": float("nan"), "d": None}) == "b"
    assert C.select_member({"a": 25.0, "b": 25.0}) == "a"                   # exact tie: the earlier member
    assert C.select_member({"a": 25.0, "b": 25.0}, order=["b", "a"]) == "b"
    with pytest.raises(C.ComparatorError):
        C.select_member({"a": float("nan")})
    with pytest.raises(C.ComparatorError):
        C.select_member({"a": 1.0}, order=["a", "b"])
    yv = np.array([0.0, 0.0, np.nan, 0.0])
    cell = np.array([True, True, True, False])
    assert C.val_mdsa(np.array([0.1, -0.1, 5.0, 9.0]), yv, cell) == pytest.approx(100 * (10 ** 0.1 - 1))
    assert np.isnan(C.val_mdsa(np.array([0.1, np.nan, 5.0, 9.0]), yv, cell))   # a member must predict the cell
    assert np.isnan(C.val_mdsa(np.zeros(4), yv, np.zeros(4, bool)))


def test_mdn_family_and_joint_likelihood():
    torch = pytest.importorskip("torch")
    from sentinel.models.orbit import mdn as M1

    mem = C.mdn_members()
    names = {C.mdn_name(*m, r) for m in mem for r in C.MDN_READOUTS}
    assert len(mem) == 3 * 3 * 2 * 2 and len(names) == 72 and C.MDN_N_INIT == 10
    g = torch.Generator().manual_seed(0)
    logw = torch.log_softmax(torch.randn(2, 7, 5, generator=g), -1)
    mu, sg = torch.randn(2, 7, 5, 3, generator=g), torch.rand(2, 7, 5, 3, generator=g) + 0.2
    y = torch.randn(2, 7, 3, generator=g)
    one = torch.ones(2, 7, 1, dtype=torch.long)
    a = C.joint_nll(logw, mu[..., :1], sg[..., :1], y[..., :1], one)          # T = 1, quantified: v1's likelihood
    assert torch.allclose(a, M1.mixture_nll(logw, mu[..., 0], sg[..., 0], y[..., 0]), atol=1e-5)
    st = torch.tensor([1, 0, 0]).expand(2, 7, 3)                               # missing targets leave the product
    assert torch.allclose(C.joint_nll(logw, mu, sg, y, st), a, atol=1e-5)
    st = torch.tensor([2, 0, 0]).expand(2, 7, 3)                               # left-censored: P(Y <= limit)
    cdf = torch.distributions.Normal(mu[..., 0], sg[..., 0]).cdf(y[..., :1])
    assert torch.allclose(C.joint_nll(logw, mu, sg, y, st), -torch.log((logw.exp() * cdf).sum(-1)), atol=1e-4)
    st = torch.tensor([1, 2, 1]).expand(2, 7, 3)
    d = torch.distributions.Normal(mu, sg)
    comp = d.log_prob(y[..., None, :])[..., 0] + torch.log(d.cdf(y[..., None, :])[..., 1]) + d.log_prob(y[..., None, :])[..., 2]
    assert torch.allclose(C.joint_nll(logw, mu, sg, y, st), -torch.logsumexp(logw + comp, -1), atol=1e-4)


def test_joint_mdn_fits_and_reads_out_both_ways():
    pytest.importorskip("torch")
    rs = np.random.default_rng(0)
    X = rs.uniform(0.01, 0.1, size=(300, 4))
    y = np.stack([20 * X[:, 0] + 0.02 * rs.normal(size=300), -10 * X[:, 1] + 0.02 * rs.normal(size=300)], 1)
    state = np.ones((300, 2), int)
    state[::5, 1] = 2                                                          # limits on target 2
    state[1::5, 1] = 0
    y[1::5, 1] = np.nan
    m = C.JointMDN(2, n_ens=3, n_iter=400, seed=0).fit(X, y, state)
    Xq = X[:40].copy()
    Xq[3, 0] = np.nan
    for r in C.MDN_READOUTS:
        p = m.predict(Xq, r)
        assert p.shape == (40, 2) and np.isnan(p[3]).all() and np.isfinite(np.delete(p, 3, 0)).all()
        assert np.corrcoef(np.delete(p[:, 0], 3), np.delete(y[:40, 0], 3))[0, 1] > 0.8
    logw, mu, sg = m.mixture(Xq)
    assert logw.shape == (3, 40, 5) and mu.shape == (3, 40, 5, 2) and sg.shape == (3, 40, 5, 2)
    single = C.JointMDN(1, n_ens=2, n_iter=50, x_log=True).fit(X, y[:, 0], np.ones(300, int))
    assert single.predict(X[:5]).shape == (5, 1)
    with pytest.raises(C.ComparatorError):
        m.predict(Xq, "mean")
    bad = state.copy()
    bad[:, 1] = 2
    with pytest.raises(C.ComparatorError):
        C.JointMDN(2, n_ens=2, n_iter=5).fit(X, np.nan_to_num(y), bad)
