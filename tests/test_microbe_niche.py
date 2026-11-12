# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Unit tests of NicheInvert-S and the weighted-averaging comparators (amendment microbe_arch_v2 §13).

SYNTHETIC data only (declared; RESEARCH_PLAN R9): hand-worked arrays and the simulator of
``tests/microbe_synth.py``. No real label is read.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).parent))

import microbe_synth as SY  # noqa: E402

from sentinel.models.microbe import baselines as B  # noqa: E402
from sentinel.models.microbe import niche as N  # noqa: E402
from sentinel.models.microbe import wa as W  # noqa: E402

FAST = {"n_eff_grid": (30.0, 300.0), "newton_steps": 4}


@pytest.fixture(scope="module")
def synth():
    return SY.make_design(seed=1)


def _flat_problem(seed=0, n=90, K=60, D=3, depth=3000, missing=True):
    """SYNTHETIC matrices for the fold-level functions (no design)."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, 1, (n, 2))
    mu = rng.uniform(-2, 2, (K, 2))
    C = SY.simulate(Z, mu, rng.normal(0, 1, K), 0.7, depth, rng)
    Y = np.column_stack([3.0 + 2.0 * Z[:, 0], -1.0 + 0.5 * Z[:, 1], rng.normal(0, 1, n)])[:, :D]
    Y = Y + rng.normal(0, 0.02, Y.shape)
    if missing:
        Y[::7, 1] = np.nan
    ctx = rng.normal(0, 1, (n, 4))
    site = np.array([f"s{i % 9}" for i in range(n)])
    dom = np.array([f"d{i % 9 // 3}" for i in range(n)])
    prof = rng.dirichlet(np.ones(64), K).astype(np.float32)          # stands in for k-mer profiles
    return C, Y, ctx, site, dom, prof, Z


# ------------------------------------------------------------------------------------- niche statistics

def test_niche_stats_hand_worked_example():
    # unit 0: p = (0.2, 0.3, 0.5) at z = (1, 2, 4): optimum 2.8, variance 1.56, N2 = 1 / 0.38
    P = sparse.csr_matrix(np.array([[0.2, 0.0, 0.0], [0.3, 1.0, 0.0], [0.5, 0.0, 0.0]]))
    Z = np.array([[1.0, 1.0], [2.0, np.nan], [4.0, 4.0]])
    st = N.niche_stats(P, Z)
    assert st["mu"][0, 0] == pytest.approx(2.8)
    assert st["n2"][0, 0] == pytest.approx(1 / 0.38)
    assert st["var"][0, 0] == pytest.approx(1.56 / 0.62)             # divided by (1 - 1/N2)
    # dimension 1 lacks the middle event: weights (0.2, 0.5), optimum 2.2 / 0.7, N2 = 0.49 / 0.29, tolerance² 4.5
    assert st["mu"][0, 1] == pytest.approx(2.2 / 0.7)
    assert st["n2"][0, 1] == pytest.approx(0.49 / 0.29)
    assert st["var"][0, 1] == pytest.approx(4.5)
    # unit 1 lives in one event: N2 = 1, no correction, zero variance; on dimension 1 it has no labelled event
    assert st["mu"][1, 0] == pytest.approx(2.0) and st["n2"][1, 0] == pytest.approx(1.0) and st["var"][1, 0] == 0.0
    assert np.isnan(st["mu"][1, 1]) and st["n2"][1, 1] == 0.0 and st["s0"][1, 1] == 0.0
    # unit 2 is absent everywhere
    assert np.isnan(st["mu"][2]).all() and (st["n2"][2] == 0).all()


def test_weighted_median():
    assert N.weighted_median(np.array([3.0, 1.0, 2.0]), np.array([1.0, 1.0, 1.0])) == 2.0
    assert N.weighted_median(np.array([3.0, 1.0, 2.0]), np.array([10.0, 1.0, 1.0])) == 3.0
    assert N.weighted_median(np.array([5.0]), np.array([2.0])) == 5.0


def test_ridge_gcv_matches_sklearn_and_brute_force_gcv():
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(3)
    X, w = rng.normal(0, 1, (40, 5)), rng.uniform(0.5, 4.0, 40)
    y = X @ rng.normal(0, 1, 5) + rng.normal(0, 0.5, 40) + 2.0
    alphas = (0.1, 1.0, 10.0, 100.0)
    r = N.ridge_gcv(X, y, w, alphas)
    gcv = []
    for a in alphas:
        xm = (w[:, None] * X).sum(0) / w.sum()
        Xt = np.sqrt(w)[:, None] * (X - xm)
        one = np.sqrt(w)[:, None]
        H = Xt @ np.linalg.solve(Xt.T @ Xt + a * np.eye(5), Xt.T) + one @ one.T / w.sum()
        yt = np.sqrt(w) * y
        gcv.append(40 * float(((yt - H @ yt) ** 2).sum()) / (40 - np.trace(H)) ** 2)
    assert np.allclose(r["gcv"], gcv, rtol=1e-8)
    assert r["alpha"] == alphas[int(np.argmin(gcv))]
    m = Ridge(alpha=r["alpha"]).fit(X, y, sample_weight=w)
    assert np.allclose(r["coef"], m.coef_, atol=1e-8) and r["intercept"] == pytest.approx(m.intercept_, abs=1e-8)


def test_seq_basis_is_orthonormal_deterministic_and_sign_fixed():
    rng = np.random.default_rng(4)
    prof = rng.dirichlet(np.ones(64), 50).astype(np.float32)
    b1, b2 = N.fit_seq_basis(prof, 8), N.fit_seq_basis(prof.copy(), 8)
    assert b1.comp.shape == (64, 8)
    assert np.allclose(b1.comp.T @ b1.comp, np.eye(8), atol=1e-10)
    assert np.array_equal(b1.comp, b2.comp)
    big = np.abs(b1.comp).argmax(0)
    assert (b1.comp[big, np.arange(8)] > 0).all()
    sc = b1.transform(prof)
    assert np.allclose(sc.var(0, ddof=1), b1.explained, rtol=1e-6)      # scores are not whitened
    assert (np.diff(b1.explained) <= 1e-12).all()
    assert N.fit_seq_basis(prof[:5], 8).comp.shape[1] <= 4              # at most n - 1 components


def test_taxonomy_prior_hand_worked():
    s0 = np.array([[1.0], [3.0], [2.0], [4.0]])
    s1 = np.array([[2.0], [0.0], [8.0], [4.0]])
    genus = np.array(["a", "a", "", ""], object)
    family = np.array(["f", "f", "f", ""], object)
    out = N.taxonomy_prior(s0, s1, genus, family)
    assert out[0, 0] == pytest.approx(0.5) and out[1, 0] == pytest.approx(0.5)     # genus a: 2 / 4
    assert out[2, 0] == pytest.approx(10.0 / 6.0)                                 # family f: (2 + 0 + 8) / 6
    assert out[3, 0] == pytest.approx(14.0 / 10.0)                                # all units


def test_deshrink_fit_clips_the_slope_and_recomputes_the_intercept():
    x = np.arange(10.0)
    assert N.deshrink_fit(x, 2.0 * x + 1.0) == pytest.approx((2.0, 1.0))
    b, a = N.deshrink_fit(x, 10.0 * x + 1.0)
    assert b == 3.0 and a == pytest.approx((10.0 * x + 1.0).mean() - 3.0 * x.mean())
    assert N.deshrink_fit(x, 0.1 * x)[0] == 0.5
    y = 2.0 * x + 1.0
    y[3] = np.nan
    assert N.deshrink_fit(x, y) == pytest.approx((2.0, 1.0))
    assert N.deshrink_fit(x[:2], y[:2]) == (1.0, 0.0)


# ------------------------------------------------------------------------------------------ unit table

def test_unit_table_matches_v4_token_aggregation(synth):
    d, truth = synth
    ut = N.build_unit_table(d)
    v4 = B.v4_tokens(d)
    ref = (v4.dropna(subset=["v4"]).assign(c=lambda t: t["count"].astype(float))
           .groupby(["sample_id", "v4"])["c"].sum().unstack(fill_value=0.0))
    ref = ref.reindex(index=d.samples.index, columns=ut.units, fill_value=0.0)
    assert list(ut.events) == list(d.samples.index)
    assert sorted(ut.units) == sorted(ref.columns) == list(ut.units)
    assert np.array_equal(ut.counts.toarray(), ref.to_numpy())
    # both primers and the twin sequence fall on the same units: one unit per planted window
    keys = {"v4:" + hashlib.sha1(w.encode()).hexdigest()[:16] for w in truth["windows"]}
    assert set(ut.units) == keys
    assert {w for w in ut.windows} == set(truth["windows"])
    # reads without a window are counted in the event total only
    tot = d.tokens.assign(c=d.tokens["count"].astype(float)).groupby("sample_id")["c"].sum().reindex(ut.events)
    assert np.array_equal(ut.n_reads_all, tot.to_numpy())
    assert np.allclose(ut.n_reads_all - np.asarray(ut.counts.sum(1)).ravel(), 37.0)
    assert ut.profile(5).shape == (len(ut.units), 1024)
    assert N.unit_table(d) is N.unit_table(d)
    with pytest.raises(KeyError):
        ut.rows(["not_an_event"])


def test_unit_table_majority_taxonomy(synth):
    d, truth = synth
    ut = N.build_unit_table(d)
    key = {"v4:" + hashlib.sha1(w.encode()).hexdigest()[:16]: k for k, w in enumerate(truth["windows"])}
    k_of = np.array([key[u] for u in ut.units])
    assert all(f == f"f{k // 15:02d}" for f, k in zip(ut.family, k_of))
    for g, k in zip(ut.genus, k_of):
        assert g == ("" if (k % 11 == 0 and k != 0) else f"g{k // 5:02d}") or k == 0


# ---------------------------------------------------------------------------------------- model fitting

def test_intercept_matches_mean_abundance():
    C, Y, ctx, site, dom, prof, _ = _flat_problem(missing=False)
    m = N.fit_niche(C, Y, ctx, site, prof)
    P, n = N._row_normalise(sparse.csr_matrix(C)[:, m.vocab])
    Z = (Y - m.y_mean) / m.y_sd
    q0 = -0.5 * (m.w[None] * (Z[:, None, :] - m.mu[None]) ** 2).sum(2)
    modelled = np.exp(m.a[None, :] + q0).sum(0)
    assert np.allclose(modelled, np.asarray(P.sum(0)).ravel(), rtol=1e-9)
    assert m.alive.all() and m.diag["n_units"] == len(m.vocab)
    assert (m.w <= 1.0 / 0.25 ** 2 + 1e-9).all() and (m.w > 0).all()
    ev = np.linalg.eigvalsh(m.sigma)
    assert ev.min() >= 1e-3 - 1e-12 and np.allclose(m.sigma @ m.sigma_inv, np.eye(3), atol=1e-8)


def test_vocabulary_needs_two_training_events():
    C, Y, ctx, site, dom, prof, _ = _flat_problem()
    C = C.copy()
    C[:, 0] = 0.0
    C[5, 0] = 50.0                      # a unit in a single event
    m = N.fit_niche(C, Y, ctx, site, prof)
    assert 0 not in m.vocab and len(m.vocab) == C.shape[1] - 1


def _log_posterior(model, c, ctx_row, z, n_eff):
    """Independent, dense evaluation of the objective for one event."""
    p = c[model.vocab] / c[model.vocab].sum()
    q = model.a - 0.5 * (model.w * (z[None, :] - model.mu) ** 2).sum(1)
    lse = q.max() + np.log(np.exp(q - q.max()).sum())
    dz = z - model.prior_mean(ctx_row[None, :])[0]
    return n_eff * float(p @ q - lse) - 0.5 * float(dz @ model.sigma_inv @ dz)


def test_likelihood_never_decreases_and_matches_independent_evaluation():
    C, Y, ctx, site, dom, prof, _ = _flat_problem(seed=5)
    tr, te = np.arange(60), np.arange(60, 90)
    m = N.fit_niche(C[tr], Y[tr], ctx[tr], site[tr], prof)
    for n_eff in (10.0, 1000.0):
        r = N.infer(m, C[te], ctx[te], n_eff)
        assert np.isfinite(r["trace"]).all() and np.isfinite(r["mode"]).all() and np.isfinite(r["sd"]).all()
        assert (np.diff(r["trace"], axis=1) >= 0).all()
        assert r["first_ok"].mean() > 0.95
        prior = m.prior_mean(ctx[te])
        for j in (0, 7, 19):
            assert r["trace"][j, 0] == pytest.approx(_log_posterior(m, C[te][j], ctx[te][j], prior[j], n_eff),
                                                     rel=1e-9, abs=1e-9)
            best = _log_posterior(m, C[te][j], ctx[te][j], r["mode"][j], n_eff)
            assert r["trace"][j, -1] == pytest.approx(best, rel=1e-9, abs=1e-9)
            rng = np.random.default_rng(j)
            for _ in range(20):                      # the mode is a local maximum
                zp = r["mode"][j] + rng.normal(0, 0.05, 3)
                assert _log_posterior(m, C[te][j], ctx[te][j], zp, n_eff) <= best + 1e-6 * (1 + abs(best))
        assert (r["sd"] > 0).all()


def test_batched_hessian_covariance_equals_the_rowwise_definition():
    """The moment form of the covariance term (one matrix product per chunk) against the per-event definition, on
    SYNTHETIC unit parameters with sharp and flat unit probabilities and units below the threshold."""
    rng = np.random.default_rng(11)
    K, D, B = 400, 17, 9
    W = 1.0 / rng.uniform(0.25, 3.0, (K, D)) ** 2
    WM = W * rng.normal(0, 2.0, (K, D))
    z = rng.normal(0, 1.5, (B, D))
    Pi = np.vstack([rng.dirichlet(np.full(K, a)) for a in (0.01, 0.05, 0.2, 1.0, 1.0, 5.0, 50.0, 0.02, 0.5)])
    Pi[3, :50] = 1e-18                                   # below the threshold: left out of the covariance term
    Pi[3] /= Pi[3].sum()
    thr = 1e-9 / K
    assert (Pi <= thr).any() and (Pi > thr).any(1).all()
    gbar = Pi @ WM - (Pi @ W) * z                        # as in infer(): the mean uses every unit
    ref = N.cov_rowwise(Pi, WM, W, z, gbar, thr)
    FM = N.moment_features(WM, W)
    assert FM.shape == (K, D * (D + 1) + D * D + 2 * D + 1)
    got = N.cov_from_moments(np.where(Pi > thr, Pi, 0.0) @ FM, z, gbar)
    assert got.shape == ref.shape == (B, D, D)
    assert np.array_equal(got, got.transpose(0, 2, 1))
    scale = np.abs(ref).max((1, 2), keepdims=True)
    assert np.abs(got - ref).max() <= 1e-9 * scale.max() and (np.abs(got - ref) <= 1e-9 * (1.0 + scale)).all()
    one = np.zeros((1, K))                               # all mass on one unit: the covariance term is zero
    one[0, 7] = 1.0
    g1 = one @ WM - (one @ W) * z[:1]
    assert np.abs(N.cov_from_moments(one @ FM, z[:1], g1)).max() < 1e-9
    assert np.abs(N.cov_rowwise(one, WM, W, z[:1], g1, thr)).max() < 1e-9


def test_inversion_recovers_planted_gradients():
    C, Y, ctx, site, dom, prof, Z = _flat_problem(seed=6, n=120, missing=True)
    tr, te = np.arange(80), np.arange(80, 120)
    m = N.fit_niche(C[tr], Y[tr], ctx[tr], site[tr], prof)
    r = N.infer(m, C[te], ctx[te], 100.0)
    y = m.to_target(r["mode"])
    assert np.corrcoef(y[:, 0], Z[te, 0])[0, 1] > 0.9
    assert np.corrcoef(y[:, 1], Z[te, 1])[0, 1] > 0.9
    res = N.fit_fold(C[tr], C[te], Y[tr], ctx[tr], ctx[te], site[tr], dom[tr], prof, params=FAST)
    assert res.n_eff in FAST["n_eff_grid"] and res.pred.shape == (40, 3)
    assert N.r2(3.0 + 2.0 * Z[te, 0], res.pred[:, 0]) > 0.8
    assert N.r2(-1.0 + 0.5 * Z[te, 1], res.pred[:, 1]) > 0.8
    assert ((res.slope >= 0.5) & (res.slope <= 3.0)).all()
    assert res.train_oof.shape == (80, 3) and np.isfinite(res.train_oof).all()
    assert set(res.diag["inner"]) == {"30.0", "300.0"}


def test_determinism_and_no_seed():
    C, Y, ctx, site, dom, prof, _ = _flat_problem(seed=7)
    tr, te = np.arange(60), np.arange(60, 90)
    a = N.fit_fold(C[tr], C[te], Y[tr], ctx[tr], ctx[te], site[tr], dom[tr], prof, params=FAST)
    b = N.fit_fold(C[tr], C[te], Y[tr], ctx[tr], ctx[te], site[tr], dom[tr], prof, params=FAST)
    assert a.n_eff == b.n_eff
    assert np.allclose(a.pred, b.pred, rtol=0, atol=1e-10) and np.allclose(a.sd, b.sd, rtol=0, atol=1e-10)


def test_dead_and_missing_dimensions_and_empty_events():
    C, Y, ctx, site, dom, prof, _ = _flat_problem(seed=8)
    Y = np.column_stack([Y, np.full(len(Y), np.nan), np.r_[np.ones(5), np.full(len(Y) - 5, np.nan)]])
    tr, te = np.arange(60), np.arange(60, 90)
    Cte = C[te].copy()
    Cte[3] = 0.0                                       # an event without reads
    res = N.fit_fold(C[tr], Cte, Y[tr], ctx[tr], ctx[te], site[tr], dom[tr], prof, params=FAST)
    assert np.isnan(res.pred[:, 3:]).all() and np.isnan(res.sd[:, 3:]).all()      # no label / fewer than 10 labels
    assert np.isfinite(res.pred[:, :3]).all() and np.isfinite(res.sd[:, :3]).all()
    m = N.fit_niche(C[tr], Y[tr], ctx[tr], site[tr], prof)
    assert list(m.alive) == [True, True, True, False, False]
    r = N.infer(m, Cte, ctx[te], 100.0)
    assert not r["has_reads"][3] and np.allclose(r["mode"][3], m.prior_mean(ctx[te])[3])
    assert np.allclose(r["sd"][3], np.sqrt(np.diag(m.sigma)))
    assert res.diag["n_test_without_reads"] == 1


def test_taxonomy_prior_variant_runs():
    C, Y, ctx, site, dom, prof, _ = _flat_problem(seed=9)
    K = C.shape[1]
    taxa = (np.array([f"g{k // 4}" if k % 5 else "" for k in range(K)], object),
            np.array([f"f{k // 12}" for k in range(K)], object))
    m = N.fit_niche(C[:60], Y[:60], ctx[:60], site[:60], None, {"prior": "taxonomy"}, taxa)
    assert m.basis is None and np.isfinite(m.mu).all()
    r = N.infer(m, C[60:], ctx[60:], 100.0)
    assert np.isfinite(r["mode"]).all()
    with pytest.raises(ValueError):
        m.open_units(prof[:2], N.DEFAULTS)


def test_open_vocabulary_only_adds_the_events_own_unseen_units():
    C, Y, ctx, site, dom, prof, _ = _flat_problem(seed=10)
    tr, te = np.arange(60), np.arange(60, 90)
    m = N.fit_niche(C[tr], Y[tr], ctx[tr], site[tr], prof)
    rng = np.random.default_rng(0)
    extra_prof = rng.dirichlet(np.ones(64), 2).astype(np.float32)
    Ce = np.zeros((30, 2))
    Ce[4, 0], Ce[9, 1] = 400.0, 150.0
    closed = N.infer(m, C[te], ctx[te], 100.0)
    opened = N.infer(m, C[te], ctx[te], 100.0, extra=(sparse.csr_matrix(Ce), extra_prof))
    same = np.setdiff1d(np.arange(30), [4, 9])
    assert np.allclose(opened["mode"][same], closed["mode"][same], atol=1e-7)
    assert not np.allclose(opened["mode"][4], closed["mode"][4], atol=1e-7)
    assert opened["in_vocab_share"][4] == pytest.approx(1.0) and np.isfinite(opened["mode"]).all()
    assert (np.diff(opened["trace"], axis=1) >= 0).all()


# ----------------------------------------------------------------------------------------- design level

def test_design_level_recovers_gradients_and_ignores_held_out_labels(synth):
    d, truth = synth
    folds = SY.lodo_folds(d.samples)
    preds = []
    for tr, te in folds:
        preds.append(N.fit_predict("niche_s", d, tr, te, targets=N.MASTER, params=FAST))
    P = pd.concat(preds).reindex(d.samples.index)
    assert list(P.columns) == list(N.MASTER) and np.isfinite(P.to_numpy()).all()
    for t in SY.PLANTED:
        assert N.r2(d.samples[t].to_numpy(float), P[t].to_numpy()) > 0.6, t
    res = N.LAST["result"]
    assert res.sd.shape == (len(folds[-1][1]), len(N.MASTER)) and (res.sd > 0).all()
    assert 0.0 <= res.diag["test_first_step_fail_share"] <= 0.05

    # changing every label of the held-out rows changes nothing
    tr, te = folds[0]
    base = N.fit_predict("niche_s", d, tr, te, targets=N.MASTER, params=FAST)
    import copy

    d2 = copy.copy(d)
    S2 = d.samples.copy()
    S2.loc[te, list(N.MASTER)] = 12345.0
    d2.samples = S2
    N._FOLD_CACHE.clear()
    moved = N.fit_predict("niche_s", d2, tr, te, targets=N.MASTER, params=FAST)
    assert np.array_equal(base.to_numpy(), moved.to_numpy())
    # ... while a training label does matter
    S3 = d.samples.copy()
    S3.loc[tr[:20], "log_sc"] += 3.0
    d3 = copy.copy(d)
    d3.samples = S3
    other = N.fit_predict("niche_s", d3, tr, te, targets=N.MASTER, params=FAST)
    assert not np.allclose(base["log_sc"], other["log_sc"])


def test_design_level_variants_control_and_guards(synth):
    d, _ = synth
    tr, te = SY.lodo_folds(d.samples)[1]
    sub = ["log_sc", "ph"]
    a = N.fit_predict("niche_s", d, tr, te, targets=sub, params=FAST)
    for name in ("niche_s_tax", "niche_s_doy", N.CONTROL):
        p = N.fit_predict(name, d, tr, te, targets=sub, params=FAST)
        assert p.shape == a.shape and np.isfinite(p.to_numpy()).all(), name
        assert not np.allclose(p.to_numpy(), a.to_numpy()), name
    assert N.fit_predict("niche_s", d, tr, te, targets=sub, params=FAST, seed=5).equals(a)    # no stochastic step
    with pytest.raises(ValueError):
        N.fit_predict("niche_s", d, tr, te, targets=["not_a_target"])
    with pytest.raises(ValueError):
        N.fit_predict("niche_x", d, tr, te, targets=sub)
    import copy

    d2 = copy.copy(d)
    d2.samples = d.samples.drop(columns=["ph"])
    with pytest.raises(KeyError):
        N.fit_predict("niche_s", d2, tr, te, targets=["log_sc"])


# ------------------------------------------------------------------------------------ weighted averaging

def test_wa_hand_worked_example():
    Y = sparse.csr_matrix(np.array([[0.2, 0.8], [0.3, 0.7], [0.5, 0.5]]))
    x = np.array([1.0, 2.0, 4.0])
    f = W.wa_fit(Y, x, tol=False)
    assert np.allclose(f["u"], [2.8, 2.1])                              # optima
    assert np.allclose(W._wa_initial(f, Y), [2.24, 2.31, 2.45])         # initial estimates
    assert f["b1"] == pytest.approx(10.0 / 0.7) and f["b0"] == pytest.approx(7 / 3 - (10.0 / 0.7) * 7 / 3)
    assert np.allclose(W.wa_predict(f, Y), np.polyval(np.polyfit([2.24, 2.31, 2.45], x, 1), [2.24, 2.31, 2.45]))
    # an event without any fitted unit takes the training mean
    assert W.wa_predict(f, sparse.csr_matrix(np.zeros((1, 2))))[0] == pytest.approx(x.mean())


def test_wa_tolerance_hand_worked_example():
    Y = sparse.csr_matrix(np.array([[0.2, 0.8, 0.0], [0.3, 0.7, 0.9], [0.5, 0.5, 0.0]]))
    x = np.array([1.0, 2.0, 4.0])
    f = W.wa_fit(Y, x, tol=True)
    t1 = 1.56 / 0.62                                                    # as in the niche example
    v2 = (0.8 * 1 + 0.7 * 4 + 0.5 * 16) / 2.0 - 2.1 ** 2
    n2 = 4.0 / (0.64 + 0.49 + 0.25)
    t2 = v2 / (1 - 1 / n2)
    pooled = (t1 * (1 / 0.38) + t2 * n2) / (1 / 0.38 + n2)              # unit 2 has N2 = 1: pooled tolerance²
    assert np.allclose(1.0 / f["w"], [t1, t2, pooled])
    x0 = W._wa_initial(f, Y)
    w = np.array([1 / t1, 1 / t2, 1 / pooled])
    u = np.array([2.8, 2.1, 2.0])
    Yd = Y.toarray()
    assert np.allclose(x0, (Yd * w) @ u / (Yd * w).sum(1))
    # the floor: 0.1 x SD of the target
    g = W.wa_fit(sparse.csr_matrix(np.array([[1.0, 0.5], [1.0, 0.5], [1.0, 0.5], [0.0, 0.5]])),
                 np.array([2.0, 2.0, 2.0, 5.0]), tol=True)
    assert g["w"][0] == pytest.approx(1.0 / (0.1 * np.std([2.0, 2.0, 2.0, 5.0])) ** 2)


def test_wapls_first_component_is_wa_with_inverse_deshrinking():
    rng = np.random.default_rng(11)
    Y = rng.dirichlet(np.ones(12) * 0.5, 40)                            # constant row sums
    x = rng.normal(0, 1, 40) + 3.0 * Y[:, 0]
    Ys = sparse.csr_matrix(Y)
    wa_ = W.wa_predict(W.wa_fit(Ys, x, tol=False), Ys)
    pls = W.wapls_fit(Ys, x, 5)
    assert np.allclose(W.wapls_predict(pls, Ys, 1), wa_, atol=1e-9)
    # training error cannot increase with more components; scores are orthogonal by construction
    sse = [float(((W.wapls_predict(pls, Ys, m) - x) ** 2).sum()) for m in range(1, len(pls["beta"]) + 1)]
    assert len(sse) >= 2 and all(b <= a + 1e-9 for a, b in zip(sse, sse[1:]))
    assert W.wapls_predict(pls, Ys, 99).shape == (40,)                  # capped at the fitted components


def test_wapls_select_uses_only_training_rows_and_prefers_fewer_on_ties():
    rng = np.random.default_rng(12)
    Y = sparse.csr_matrix(rng.dirichlet(np.ones(10) * 0.5, 60))
    x = rng.normal(0, 1, 60)
    g = np.array([f"d{i % 4}" for i in range(60)])
    m, sse = W.wapls_select(Y, x, g, {"max_components": 3})
    assert 1 <= m <= 3 and len(sse) == 3 and m == int(np.argmin(sse)) + 1
    assert W.wapls_select(Y, x, np.array(["one"] * 60))[0] == 1         # no inner CV possible


def test_wa_design_level_runs_recovers_gradient_and_is_label_safe(synth):
    d, _ = synth
    folds = SY.lodo_folds(d.samples)
    tgt = ["log_sc", "ph", "log_chla"]
    for name in W.NAMES:
        P = pd.concat([W.fit_predict(name, d, tr, te, targets=tgt) for tr, te in folds]).reindex(d.samples.index)
        assert np.isfinite(P.to_numpy()).all(), name
        assert N.r2(d.samples["log_sc"].to_numpy(float), P["log_sc"].to_numpy()) > 0.5, name
        assert N.r2(d.samples["ph"].to_numpy(float), P["ph"].to_numpy()) > 0.5, name
        tr, te = folds[2]
        import copy

        d2 = copy.copy(d)
        S2 = d.samples.copy()
        S2.loc[te, tgt] = -999.0
        d2.samples = S2
        a, b = (W.fit_predict(name, x, tr, te, targets=tgt) for x in (d, d2))
        assert np.array_equal(a.to_numpy(), b.to_numpy()), name
    assert W.LAST["name"] == "wa_pls" and W.LAST["n_units"] > 0
    with pytest.raises(ValueError):
        W.fit_predict("wa_x", d, *folds[0], targets=tgt)
