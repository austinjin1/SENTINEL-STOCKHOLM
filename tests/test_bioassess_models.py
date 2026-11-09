# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Leakage and correctness tests for SENTINEL-BioAssess (M5).

Covers protocol §8 (docs/preregistration/bioassess_protocol.md): no location cluster / site / visit
in both training and held-out folds; within-cycle revisits and cross-cycle re-sampled locations
grouped; chemistry never an input when it is the target; normalisers fit on training rows only;
the development loader refuses the external-test surveys. Tests on the real NRSA design are skipped
when the processed data are absent; everything else uses small synthetic arrays (allowed in tests).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.core import paths
from sentinel.models.bioassess import baselines as B
from sentinel.models.bioassess import bioset as BS
from sentinel.models.bioassess import evaluate as E
from sentinel.models.bioassess import splits as SP
from sentinel.models.bioassess.data import (CHEMISTRY_INPUT_BLOCKLIST_PREFIXES, TARGET_NAMES, BioDesign,
                                            HoldoutAccessError, assert_dev_only, load_nrsa, location_clusters)
from sentinel.models.bioassess.features import CommunityFeaturizer, Standardiser


def _have_nrsa() -> bool:
    p = paths.PROCESSED / "nars_streams" / "visits.parquet"
    t = paths.PROCESSED / "taxonomy_harmonised" / "taxon_map.parquet"
    return p.exists() and t.exists()


needs_nrsa = pytest.mark.skipif(not _have_nrsa(), reason="processed NRSA / taxonomy not on this machine")


# ----------------------------------------------------------------------------- synthetic design

def synthetic_design(n_loc: int = 60, seed: int = 0) -> BioDesign:
    """Synthetic community/chemistry design (unit tests only): 3 ecoregions, revisits, 2 cycles."""
    rs = np.random.default_rng(seed)
    rows, toks = [], []
    fams = [f"FAM{k}" for k in range(12)]
    for L in range(n_loc):
        eco = ["AAA", "BBB", "CCC"][L % 3]
        lat, lon = 30 + 2 * (L % 3) + rs.uniform(0, 1.5), -100 + rs.uniform(0, 8)
        for cyc, survey in enumerate(("nrsa0809", "nrsa1314")):
            if cyc == 1 and L % 2:
                continue
            vid = f"{survey}-{L}"
            rows.append({"visit_id": vid, "survey": survey, "site_id": f"S{cyc}_{L}", "location_id": f"L{L:03d}",
                         "ag_eco9": eco, "huc8": "01010101", "lat": lat + cyc * 1e-4, "lon": lon,
                         "elevation_m": rs.uniform(0, 500), "reference_class": "R" if L % 5 == 0 else None,
                         "boatable": float(L % 4 == 0)})
            present = rs.choice(len(fams), size=rs.integers(3, 9), replace=False)
            for k in present:
                toks.append({"visit_id": vid, "name": f"G{k}", "phylum": "ARTHROPODA", "class": "INSECTA",
                             "order": f"ORD{k % 3}", "family": fams[k], "genus": f"G{k}", "mixed_otu": f"G{k}",
                             "identification_rank": "genus", "count": float(rs.integers(1, 60))})
    s = pd.DataFrame(rows).set_index("visit_id")
    t = pd.DataFrame(toks)
    agg = t.groupby("visit_id").agg(total=("count", "sum"), n=("name", "size"))
    s["log10_total_count"] = np.log10(agg["total"].reindex(s.index))
    s["log10_n_taxa"] = np.log10(agg["n"].reindex(s.index))
    A = t.pivot_table(index="visit_id", columns="name", values="count", aggfunc="sum", fill_value=0).reindex(s.index)
    sig = np.sqrt(A.div(A.sum(1), axis=0)).to_numpy() @ rs.normal(size=A.shape[1])
    Y = pd.DataFrame(index=s.index)
    for j, tname in enumerate(TARGET_NAMES):
        Y[tname] = sig * (1 + 0.1 * j) + rs.normal(scale=0.3, size=len(s))
    return BioDesign(s.sort_index(), t, Y.sort_index(), {})


# ------------------------------------------------------------------------------ holdout guard

def test_loader_refuses_external_test_surveys():
    for s in (["nrsa1819"], ["nrsa2324"], ["nrsa0809", "nrsa1819"]):
        with pytest.raises(HoldoutAccessError):
            load_nrsa(s)
    df = pd.DataFrame({"survey": ["nrsa0809", "nrsa1819"]})
    with pytest.raises(HoldoutAccessError):
        assert_dev_only(df)
    assert_dev_only(pd.DataFrame({"survey": ["nrsa0809", "nrsa1314"]}))


# ------------------------------------------------------------------------- location clusters

def test_location_clusters_chain_and_separate():
    # 0.5 km steps chain three points; a fourth point 50 km away is separate
    lat = np.array([40.0, 40.0045, 40.009, 40.45])
    lon = np.array([-100.0, -100.0, -100.0, -100.0])
    lab = location_clusters(lat, lon, 1.0)
    assert lab[0] == lab[1] == lab[2] and lab[3] != lab[0]


# ----------------------------------------------------------------------------- fold integrity

@pytest.mark.parametrize("scheme", ["loeo", "block50", "random5"])
def test_synthetic_folds_partition_and_group(scheme):
    d = synthetic_design()
    folds = SP.make_folds(d.samples, scheme)
    SP.check_folds(folds, d.samples)
    fold_of = {}
    for i, f in enumerate(folds):
        _, te = SP.masks(d.samples, f)
        for loc in d.samples.loc[te, "location_id"]:
            fold_of.setdefault(loc, set()).add(i)
    assert all(len(v) == 1 for v in fold_of.values())   # both cycles of a location in the same fold


def test_check_folds_detects_leak():
    from sentinel.eval.splits import SiteSplit

    d = synthetic_design()
    locs = sorted(d.samples["location_id"].unique())
    # same site_id under two different locations, split across train/test -> must raise
    s = d.samples.copy()
    s.loc[s["location_id"] == locs[0], "site_id"] = "DUP"
    s.loc[s["location_id"] == locs[1], "site_id"] = "DUP"
    f = [SiteSplit(train=locs[1:], val=[], test=[locs[0]], method="t"),
         SiteSplit(train=[locs[0]], val=[], test=locs[1:], method="t")]
    with pytest.raises(AssertionError):
        SP.check_folds(f, s)


def test_inner_validation_is_inside_training_fold():
    d = synthetic_design(90)
    for f in SP.make_folds(d.samples, "loeo"):
        fit, val = SP.inner_validation(d.samples, f.train)
        assert set(fit) | set(val) == set(f.train)
        assert not set(fit) & set(val)
        assert not (set(fit) | set(val)) & set(f.test)


@needs_nrsa
@pytest.mark.parametrize("scheme", ["loeo", "block50", "random5"])
def test_real_folds_no_site_or_visit_leak(scheme):
    d = load_nrsa()
    assert set(d.samples["survey"]) == {"nrsa0809", "nrsa1314"}
    folds = SP.make_folds(d.samples, scheme)
    SP.check_folds(folds, d.samples)
    test_fold = pd.Series(-1, index=d.samples.index)
    for i, f in enumerate(folds):
        _, te = SP.masks(d.samples, f)
        assert (test_fold[te] == -1).all()
        test_fold[te] = i
    assert (test_fold >= 0).all()
    # within-cycle / duplicate-first-visit revisits (same site_id) share one fold
    assert d.samples.assign(f=test_fold).groupby("site_id")["f"].nunique().max() == 1
    # cross-cycle re-sampled locations (< 1 km apart, different site ids) share one fold
    from sklearn.neighbors import BallTree

    s = d.samples
    X = np.radians(s[["lat", "lon"]].to_numpy(float))
    ind = BallTree(X, metric="haversine").query_radius(X, r=0.99 / 6371.0088)
    for i, nb in enumerate(ind):
        assert (test_fold.iloc[nb] == test_fold.iloc[i]).all()


@needs_nrsa
def test_real_design_has_no_chemistry_inputs():
    d = load_nrsa()
    bad = [c for c in d.samples.columns if c.startswith(CHEMISTRY_INPUT_BLOCKLIST_PREFIXES)]
    assert not bad, bad
    fz = CommunityFeaturizer().fit(d, d.samples.index[:500])
    names = [n for n in fz.feature_names() if not n.startswith("taxon::")]
    assert set(names) == {"log10_total_count", "log10_n_taxa", "boatable"}


# ----------------------------------------------------------- chemistry is never an input

def test_features_do_not_depend_on_targets():
    d = synthetic_design()
    ids = d.samples.index
    X1 = CommunityFeaturizer(min_samples=2).fit(d, ids[:60]).transform(d, ids)
    d2 = BioDesign(d.samples, d.tokens, d.Y * -3.0 + 7.0, {})
    X2 = CommunityFeaturizer(min_samples=2).fit(d2, ids[:60]).transform(d2, ids)
    np.testing.assert_array_equal(X1, X2)
    vocab = BS.TaxonVocab.build(d.tokens)
    e1 = BS.encode(d.tokens, ids.to_numpy(), vocab)
    e2 = BS.encode(d2.tokens, ids.to_numpy(), vocab)
    np.testing.assert_array_equal(e1.lev, e2.lev)
    np.testing.assert_array_equal(e1.ab, e2.ab)


def test_single_target_baseline_ignores_other_targets():
    d = synthetic_design()
    ids = d.samples.index
    tr, te = ids[:70], ids[70:]
    p1 = B.fit_predict("rf_community", d, tr, te, seed=0, params={"n_estimators": 30}, n_jobs=1,
                       min_samples=2, targets=["log10_tp_ug_l"])
    Y2 = d.Y.copy()
    Y2.loc[:, [c for c in TARGET_NAMES if c != "log10_tp_ug_l"]] = np.random.default_rng(1).normal(size=(len(ids), 5))
    p2 = B.fit_predict("rf_community", BioDesign(d.samples, d.tokens, Y2, {}), tr, te, seed=0,
                       params={"n_estimators": 30}, n_jobs=1, min_samples=2, targets=["log10_tp_ug_l"])
    np.testing.assert_allclose(p1.to_numpy(), p2.to_numpy())


def test_fit_predict_rejects_overlap():
    d = synthetic_design()
    ids = d.samples.index
    with pytest.raises(AssertionError):
        B.fit_predict("ecoregion_mean", d, ids[:50], ids[40:])


# --------------------------------------------------------- normalisation on training rows only

def test_featurizer_vocabulary_from_training_rows_only():
    d = synthetic_design()
    ids = d.samples.index
    tr = ids[:60]
    extra = pd.DataFrame([{"visit_id": i, "name": "ONLYTEST", "phylum": "X", "class": "X", "order": "X",
                           "family": "XF", "genus": "ONLYTEST", "mixed_otu": "ONLYTEST",
                           "identification_rank": "genus", "count": 50.0} for i in ids[60:]])
    d2 = BioDesign(d.samples, pd.concat([d.tokens, extra], ignore_index=True), d.Y, {})
    fz = CommunityFeaturizer(min_samples=1).fit(d2, tr)
    assert "ONLYTEST" not in fz.taxa_
    vocab = BS.TaxonVocab.build(d2.tokens)
    allowed = vocab.allowed_from(d2.tokens[d2.tokens["visit_id"].isin(tr)], 1)
    e = BS.restrict(BS.encode(d2.tokens, ids[60:].to_numpy(), vocab), allowed)
    assert vocab.maps["name"]["ONLYTEST"] not in set(e.lev[..., BS.LEVELS.index("name")].ravel())


def test_standardiser_uses_training_statistics():
    rs = np.random.default_rng(0)
    Xtr, Xte = rs.normal(2, 3, (100, 4)), rs.normal(50, 1, (30, 4))
    st = Standardiser().fit(Xtr)
    np.testing.assert_allclose(st.mean_, Xtr.mean(0))
    st2 = Standardiser().fit(Xtr)
    _ = st2.transform(Xte * 100)          # transforming other rows never refits
    np.testing.assert_allclose(st2.mean_, Xtr.mean(0))


def test_bioset_normalisers_fit_on_training_rows_only():
    d = synthetic_design()
    ids = d.samples.index.to_numpy()
    vocab = BS.TaxonVocab.build(d.tokens)
    g = d.samples[["log10_total_count", "log10_n_taxa", "boatable"]].to_numpy(np.float32)
    tr, va = np.arange(0, 60), np.arange(60, 75)
    e = BS.encode(d.tokens, ids, vocab, g)
    Y = d.Y.to_numpy(np.float32)
    Yva = Y[va] + 100.0                       # validation targets far off: must not affect normalisers
    cfg = {"arch": "settransformer", "d_model": 16, "n_layers": 1, "n_heads": 2, "dropout": 0.0,
           "token_dropout": 0.1, "resolution_dropout": 0.2, "embedding": "hierarchical", "lr": 1e-3,
           "weight_decay": 0.0, "batch_size": 16, "max_epochs": 2, "patience": 5}
    f = BS.fit_bioset(cfg, vocab, e.take(tr), Y[tr], e.take(va), Yva, seed=0, device="cpu")
    np.testing.assert_allclose(f.y_mean, np.nanmean(Y[tr], 0), rtol=1e-5)
    np.testing.assert_allclose(f.glob_mean, g[tr].mean(0), rtol=1e-5)
    p = f.predict(e.take(np.arange(75, len(ids))))
    assert p.shape == (len(ids) - 75, len(TARGET_NAMES)) and np.isfinite(p).all()


def test_bioset_is_permutation_invariant():
    d = synthetic_design()
    ids = d.samples.index.to_numpy()
    vocab = BS.TaxonVocab.build(d.tokens)
    e = BS.encode(d.tokens, ids, vocab, d.samples[["log10_total_count"]].to_numpy(np.float32))
    cfg = {"arch": "settransformer", "d_model": 16, "n_layers": 2, "n_heads": 2, "dropout": 0.0,
           "embedding": "hierarchical", "lr": 1e-3, "weight_decay": 0.0, "batch_size": 32, "max_epochs": 1,
           "patience": 1}
    f = BS.fit_bioset(cfg, vocab, e, d.Y.to_numpy(np.float32), e.take(slice(0, 10)),
                      d.Y.to_numpy(np.float32)[:10], seed=0)
    perm = np.random.default_rng(3).permutation(e.lev.shape[1])
    e2 = BS.Encoded(e.ids, e.lev[:, perm], e.ab[:, perm], e.mask[:, perm], e.glob)
    np.testing.assert_allclose(f.predict(e), f.predict(e2), atol=1e-4)


# ---------------------------------------------------------------------------- metrics / WA-PLS

def test_anomaly_and_region_metrics():
    rs = np.random.default_rng(0)
    region = np.repeat(["a", "b", "c"], 50)
    y = rs.normal(size=150) + np.repeat([0.0, 3.0, -2.0], 50)
    const = pd.Series(y).groupby(region).transform("mean").to_numpy()
    assert abs(E.anomaly_r2(y, const, region)) < 1e-12               # region-constant predictor -> 0
    assert E.anomaly_r2(y, y, region) == pytest.approx(1.0)
    shifted = y + np.repeat([5.0, -1.0, 2.0], 50)                    # region offsets do not matter
    assert E.anomaly_r2(y, shifted, region) == pytest.approx(1.0)
    pr = E.per_region_r2(y, const, region)
    assert all(abs(v) < 1e-12 for v in pr.values())
    s = E.summary(y, const, region)
    assert s["r2_between_region"] == pytest.approx(1.0) and s["r2"] > 0.5


def test_wapls_recovers_gradient():
    rs = np.random.default_rng(0)
    n, m = 300, 40
    x = rs.uniform(0, 10, n)
    optima = np.linspace(0, 10, m)
    A = np.exp(-((x[:, None] - optima[None]) ** 2) / 2.0) * rs.gamma(2.0, 1.0, (n, m))
    A = A / A.sum(1, keepdims=True)
    p1 = B.wapls_fit_predict(A[:200], x[:200], A[200:], 1)
    p2 = B.wapls_fit_predict(A[:200], x[:200], A[200:], 2)
    from sentinel.eval import metrics as EM

    assert EM.r2(x[200:], p1) > 0.8 and EM.r2(x[200:], p2) > 0.8
    # in-sample: component 2 never fits worse than component 1
    q1 = B.wapls_fit_predict(A[:200], x[:200], A[:200], 1)
    q2 = B.wapls_fit_predict(A[:200], x[:200], A[:200], 2)
    assert EM.rmse(x[:200], q2) <= EM.rmse(x[:200], q1) + 1e-9


# ------------------------------------------------------------------------------ UK pretraining

def test_uk_site_roles_deterministic_and_disjoint():
    from sentinel.models.bioassess.uk import uk_site_roles

    rs = np.random.default_rng(0)
    sites = pd.DataFrame({"site_id": [f"s{i}" for i in range(400)], "lat": rs.uniform(50, 55, 400),
                          "lon": rs.uniform(-4, 1, 400)})
    r1, r2 = uk_site_roles(sites), uk_site_roles(sites.sample(frac=1.0, random_state=1))
    assert r1.sort_index().equals(r2.sort_index())
    assert set(r1.unique()) == {"pool_fit", "pool_val", "reserved_test"}


def test_pretraining_refuses_reserved_uk_samples():
    from sentinel.models.bioassess.pretrain import pretrain_uk
    from sentinel.models.bioassess.uk import UKDesign

    s = pd.DataFrame({"role": ["pool_fit", "reserved_test"]}, index=["a1", "a2"])
    uk = UKDesign(s, pd.DataFrame(), pd.DataFrame(index=s.index), {})
    with pytest.raises(AssertionError):
        pretrain_uk({}, {}, uk, pd.DataFrame(), device="cpu")


def test_bioset_geography_fill_uses_training_fold_medians():
    from sentinel.models.bioassess import cv

    d = synthetic_design()
    s = d.samples.copy()
    s.iloc[:5, s.columns.get_loc("elevation_m")] = np.nan
    d2 = BioDesign(s, d.tokens, d.Y, {})
    fill = {"lat": 0.0, "lon": 0.0, "elevation_m": float(s["elevation_m"].iloc[10:].median())}
    g = cv._glob(d2, s.index[:5], True, fill)
    assert np.isfinite(g).all() and np.allclose(g[:, -1], fill["elevation_m"])


def test_bioset_geography_column_unobserved_in_fold_is_dropped_for_all_rows():
    """Dry-run case: elevation is missing for every NRSA 2008-09 row, so a 2008-09-only fit fold has
    no elevation; the column must be constant for train *and* test rows (not NaN, not raw values)."""
    from sentinel.models.bioassess import cv

    d = synthetic_design()
    s = d.samples.copy()
    s.iloc[:5, s.columns.get_loc("elevation_m")] = np.nan
    d2 = BioDesign(s, d.tokens, d.Y, {})
    fill = {"lat": 0.0, "lon": 0.0, "elevation_m": float("nan")}
    g_tr = cv._glob(d2, s.index[:5], True, fill)        # fit rows: all missing
    g_te = cv._glob(d2, s.index[10:15], True, fill)     # test rows: observed
    assert np.isfinite(g_tr).all() and np.isfinite(g_te).all()
    assert np.allclose(g_tr[:, -1], 0.0) and np.allclose(g_te[:, -1], 0.0)
    assert np.allclose(g_te[:, -3], s["lat"].iloc[10:15].to_numpy())   # observed columns untouched


def test_aux_outputs_are_training_only():
    """Auxiliary chemistry enters only the loss: encodings ignore it and predictions are primary-only."""
    d = synthetic_design()
    rs = np.random.default_rng(5)
    A = pd.DataFrame(rs.normal(size=(len(d.samples), 3)), index=d.samples.index, columns=["a1", "a2", "a3"])
    ids = d.samples.index.to_numpy()
    vocab = BS.TaxonVocab.build(d.tokens)
    e = BS.encode(d.tokens, ids, vocab, d.samples[["log10_total_count"]].to_numpy(np.float32))
    Y = np.hstack([d.Y.to_numpy(np.float32), A.to_numpy(np.float32)])
    cfg = {"arch": "settransformer", "d_model": 16, "n_layers": 1, "n_heads": 2, "dropout": 0.0,
           "embedding": "flat", "lr": 1e-3, "weight_decay": 0.0, "batch_size": 32, "max_epochs": 2,
           "patience": 5, "aux_weight": 0.5}
    f = BS.fit_bioset(cfg, vocab, e.take(np.arange(60)), Y[:60], e.take(np.arange(60, 75)), Y[60:75],
                      seed=0, n_primary=len(TARGET_NAMES))
    assert f.predict(e.take(np.arange(75, len(ids)))).shape[1] == Y.shape[1]
    e2 = BS.encode(d.tokens, ids, vocab, d.samples[["log10_total_count"]].to_numpy(np.float32))
    np.testing.assert_array_equal(e.lev, e2.lev)
    np.testing.assert_array_equal(e.glob, e2.glob)


def test_region_anomaly_loss_ignores_region_offsets():
    import torch

    rs = np.random.default_rng(0)
    y = torch.tensor(rs.normal(size=(40, 3)), dtype=torch.float32)
    region = torch.tensor(np.repeat([0, 1, 2, 3], 10))
    offs = torch.tensor([[5.0], [-2.0], [1.0], [0.5]])[region]
    assert float(BS.region_anomaly_mse(y + offs, y, region)) < 1e-10      # pure region offsets: no loss
    assert float(BS.region_anomaly_mse(y.flip(0), y, region)) > 0.1
    y2 = y.clone()
    y2[:5, 0] = float("nan")
    assert np.isfinite(float(BS.region_anomaly_mse(y + offs, y2, region)))


def test_context_features_are_metadata_only():
    d = synthetic_design()
    s = d.samples.copy()
    s["doy_sin"], s["doy_cos"] = np.sin(np.arange(len(s))), np.cos(np.arange(len(s)))
    d2 = BioDesign(s, d.tokens, d.Y, {})
    fz = CommunityFeaturizer(min_samples=2, geography=True, season=True).fit(d2, s.index[:60])
    extra = [n for n in fz.feature_names() if not n.startswith("taxon::")]
    assert extra == ["log10_total_count", "log10_n_taxa", "boatable", "lat", "lon", "elevation_m", "doy_sin", "doy_cos"]
    p = B.fit_predict("rf_context", d2, s.index[:60], s.index[60:], params={"n_estimators": 20}, n_jobs=1,
                      targets=["ph_lab"])
    assert np.isfinite(p.to_numpy()).all()


def test_oof_bioset_equals_row_level_refactor():
    """oof_bioset (fold runner) and bioset_rows (used by the final scorer) give identical predictions."""
    from sentinel.models.bioassess import cv

    d = synthetic_design(90)
    folds = SP.make_folds(d.samples, "loeo")[:1]
    cfg = {"arch": "settransformer", "d_model": 16, "n_layers": 1, "n_heads": 2, "dropout": 0.0, "token_dropout": 0.1,
           "embedding": "flat", "lr": 1e-3, "weight_decay": 0.0, "batch_size": 32, "max_epochs": 3, "patience": 5,
           "min_name_count": 1}
    ens, _, _ = cv.oof_bioset(cfg, d, folds, seeds=(0,))
    tr, te = SP.masks(d.samples, folds[0])
    vocab = BS.TaxonVocab.build(d.tokens)
    p, _, _ = cv.bioset_rows(cfg, d, d.samples.index[tr], d.samples.index[te], vocab=vocab, seeds=(0,))
    np.testing.assert_allclose(ens.loc[d.samples.index[te]].to_numpy(), p, rtol=1e-5, atol=1e-6)


def test_final_scorer_refuses_without_freeze_and_parses_ensembles():
    import importlib.util
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts/eval/bioassess_final.py"
    freeze = repo / "docs/preregistration/bioassess_freeze.json"
    if not freeze.exists():
        r = subprocess.run([sys.executable, "-B", str(script), "--authorised-by", "pytest"], capture_output=True,
                           text=True, timeout=120)
        assert r.returncode != 0 and "refusing" in (r.stderr + r.stdout)
    spec = importlib.util.spec_from_file_location("bioassess_final", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.ensemble_members("Ensemble[BioSet[bioset_st_flat]+TabPFNv2[community]]") == \
        ["BioSet[bioset_st_flat]", "TabPFNv2[community]"]
    assert mod.ensemble_members("Ensemble[BioSet[bioset_st_flat]+geography+TabPFNv2[community]]") == \
        ["BioSet[bioset_st_flat]+geography", "TabPFNv2[community]"]
    assert mod.ensemble_members("Ensemble[RandomForest[community+geography]+TabPFNv2[community]]") == \
        ["RandomForest[community+geography]", "TabPFNv2[community]"]


def test_transfer_family_classes_shared_rule():
    from sentinel.models.bioassess import transfer as TR

    us_tok = pd.DataFrame({"visit_id": ["a", "a", "b"], "name": ["BAETIS", "ACENTRELLA", "OLIGOCHAETA"],
                           "family": ["BAETIDAE", "BAETIDAE", None], "count": [6.0, 5.0, 150.0]})
    X = TR.family_classes(us_tok, "visit_id", ["a", "b"], ["BAETIDAE", "OLIGOCHAETA", "HEPTAGENIIDAE"])
    np.testing.assert_array_equal(X, [[2, 0, 0], [0, 3, 0]])      # 11 -> class 2; 150 -> class 3; absent 0
    uk_tok = pd.DataFrame({"analysis_id": ["u", "u"], "name": ["BAETIDAE", "BAETIS"], "family": ["BAETIDAE"] * 2,
                           "count": [30.0, 3.0], "log_class": [2.0, 1.0]})
    np.testing.assert_array_equal(TR.family_classes(uk_tok, "analysis_id", ["u"], ["BAETIDAE"]), [[2]])


def _final_module():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts/eval/bioassess_final.py"
    spec = importlib.util.spec_from_file_location("bioassess_final_mod", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_final_verify_freeze_refuses_second_pass_and_changes(tmp_path):
    import json

    from sentinel.core.registry import file_sha256, log_result

    mod = _final_module()
    f = "scripts/eval/bioassess_final.py"
    rec = {"sha256": {f: file_sha256(mod.REPO / f)}, "package_versions": mod.package_versions(["numpy"])}
    rp, marker, reg = tmp_path / "freeze.json", tmp_path / "marker.json", tmp_path / "registry.jsonl"
    rp.write_text(json.dumps(rec))
    assert mod.verify_freeze(rp, marker, reg)["sha256"] == rec["sha256"]          # clean state passes
    marker.write_text("{}")
    with pytest.raises(SystemExit, match="already scored"):                       # second pass refused
        mod.verify_freeze(rp, marker, reg)
    marker.unlink()
    log_result(workstream="bioassess", experiment="bioassess_test", model="m", metric="r2", value=0.1, split="s",
               registry_path=reg)
    with pytest.raises(SystemExit, match="registry already holds"):               # test rows present
        mod.verify_freeze(rp, marker, reg)
    reg.unlink()
    rp.write_text(json.dumps({**rec, "sha256": {f: "0" * 64}}))
    with pytest.raises(SystemExit, match="changed since freeze"):                 # code changed
        mod.verify_freeze(rp, marker, reg)
    rp.write_text(json.dumps({**rec, "package_versions": {"numpy": "0.0.0"}}))
    with pytest.raises(SystemExit, match="changed since freeze"):                 # package changed
        mod.verify_freeze(rp, marker, reg)


def test_final_holm_and_nan_safe_logging(tmp_path):
    from sentinel.core.registry import read_registry
    from sentinel.eval.bootstrap import holm_bonferroni

    mod = _final_module()
    p = {"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.3}
    adj = mod.holm_adjust(p)
    assert {k: v <= 0.05 for k, v in adj.items()} == holm_bonferroni(p)
    assert adj["a"] == pytest.approx(0.004) and adj["b"] == pytest.approx(0.06)
    slog = mod.SafeLog(tmp_path / "r.jsonl", {"x": 1})
    slog(model="m", metric="r2", value=float("nan"), split="s")
    slog(model="m", metric="r2", value=0.5, split="s", ci_low=float("nan"), ci_high=0.6)
    assert not (tmp_path / "r.jsonl").exists()                    # buffered: nothing written before flush
    assert slog.flush() == 1
    rows = read_registry(tmp_path / "r.jsonl")
    assert len(rows) == 1 and rows[0]["ci_low"] is None and len(slog.not_estimable) == 1


def test_final_score_frame_end_to_end_synthetic(tmp_path):
    """The scorer's statistics path on a synthetic prediction frame (scratch registry)."""
    from sentinel.core.registry import read_registry

    mod = _final_module()
    rs = np.random.default_rng(0)
    n = 300
    F = pd.DataFrame({"visit_id": [f"v{i}" for i in range(n)], "survey": "nrsa1314",
                      "location_id": [f"L{i // 2}" for i in range(n)], "ag_eco9": np.repeat(list("ABC"), n // 3),
                      "block50": [f"b{i // 10}" for i in range(n)]})
    models = ["Cand", "Ref", "RF"]
    for t in mod.TARGETS:
        y = rs.normal(size=n)
        F[f"obs|{t}"] = y
        for m, noise in (("Cand", 0.5), ("Ref", 0.9), ("RF", 1.2)):
            for s in mod.SEEDS:
                F[f"pred_seed{s}|{m}|{t}"] = y + rs.normal(scale=noise, size=n)
            F[f"pred|{m}|{t}"] = F[[f"pred_seed{s}|{m}|{t}" for s in mod.SEEDS]].mean(1)
    rec = {"n_boot": 50, "n_boot_confirmatory": 100, "candidates": ["Cand"], "h_m5_reference": "RF",
           "h_m5_family_candidate": "Cand", "sota_comparators": {t: "Ref" for t in mod.TARGETS},
           "controls_for": {"Cand": {"leave_bioset_out": "Ref"}}, "alpha": 0.05}
    slog = mod.SafeLog(tmp_path / "r.jsonl", {})
    dec = mod.score_frame(F, rec, "A", "nrsa1314/loeo_x_cycle_test", slog, confirmatory=True, run_dir=tmp_path,
                          models=models)
    assert len(dec) == 12 and all(v["rejected"] for v in dec.values())
    assert all(v["sota_label"] for k, v in dec.items() if k.startswith("sota|"))
    slog.flush()
    rows = read_registry(tmp_path / "r.jsonl")
    hm = [r for r in rows if r["metric"] == "h_m5_n_targets_rejected"]
    assert len(hm) == 1 and hm[0]["value"] == 6 and hm[0]["extra"]["endpoint_met"]
    metrics = {r["metric"] for r in rows}
    assert {"holm_rejected_sota", "holm_rejected_h_m5", "delta_r2_vs_leave_bioset_out", "r2"} <= metrics
    d = next(r for r in rows if r["metric"] == "delta_r2_vs_sota")
    assert set(d["extra"]["per_seed_delta"]) == {"0", "1", "2"} or set(d["extra"]["per_seed_delta"]) == {0, 1, 2}
    assert "block50" in d["extra"]["sensitivity_ci"] and "ecoregion" in d["extra"]["sensitivity_ci"]
    assert all(r["is_baseline"] for r in rows if r["metric"] == "r2" and r["model"] in ("Ref", "RF"))
    sec = mod.score_frame(F, rec, "B_all", "x/cycle_out_test_all", slog, confirmatory=False, run_dir=tmp_path,
                          models=models)
    assert sec == {}                                                    # no Holm outside Test A


def test_final_prediction_cache_resume_reuses_units_and_refuses_tampering(tmp_path):
    """W-a: units are cached with sha256; a new process (resume) reuses them without refitting;
    ensembles are rebuilt from cached members; a tampered unit file is refused."""
    mod = _final_module()
    te = pd.Index([f"v{i}" for i in range(5)])
    calls = []

    def fake_fit(model, design, tr, te_ids):
        calls.append(model)
        return {s: pd.DataFrame(np.full((len(te_ids), len(mod.TARGETS)), float(len(model) + s)),
                                index=pd.Index(te_ids), columns=list(mod.TARGETS)) for s in mod.SEEDS}

    class D:
        samples = pd.DataFrame({"survey": ["x"] * 5}, index=te)

    def predictor():
        p = mod.Predictor.__new__(mod.Predictor)
        p.memo, p.cache, p._fit = {}, mod.PredCache(tmp_path), fake_fit
        return p

    ens = "Ensemble[RandomForest[community]+TabPFNv2[community]]"
    first = predictor()(ens, D, te[:0], te, unit="A|CPL")
    assert sorted(calls) == ["RandomForest[community]", "TabPFNv2[community]"]
    calls.clear()
    p2 = predictor()                                                   # a resumed process
    again = p2(ens, D, te[:0], te, unit="A|CPL")
    assert calls == [] and p2.cache.n_reused == 2
    for s in mod.SEEDS:
        pd.testing.assert_frame_equal(first[s], again[s])
    p2("LightGBM[community]", D, te[:0], te, unit="A|CPL")             # a unit missing from the manifest is fitted
    assert calls == ["LightGBM[community]"]
    e = p2.cache.entries["A|CPL|RandomForest[community]"]
    (tmp_path / "cache" / e["file"]).write_bytes(b"tampered")
    with pytest.raises(SystemExit, match="sha256"):
        predictor()(ens, D, te[:0], te, unit="A|CPL")


def test_final_resume_rule_guards(tmp_path):
    """The pre-registered resume rule: once only, same record and authorisation, no registry row."""
    import json

    from sentinel.core.registry import file_sha256, log_result

    mod = _final_module()
    run = tmp_path / "run"
    (run / "cache").mkdir(parents=True)
    (run / "cache" / "manifest.jsonl").write_text("")
    f = "scripts/eval/bioassess_final.py"
    rec = {"sha256": {f: file_sha256(mod.REPO / f)}, "package_versions": mod.package_versions(["numpy"])}
    (run / "freeze_record.json").write_text(json.dumps(rec))
    freeze, marker, reg = tmp_path / "freeze.json", tmp_path / "marker.json", tmp_path / "registry.jsonl"
    freeze.write_text(json.dumps(rec))
    kw = dict(dry=False, freeze_path=freeze, marker=marker, registry_path=reg)
    with pytest.raises(SystemExit, match="no one-time marker"):
        mod.verify_resume(run, "lead", **kw)
    marker.write_text(json.dumps({"run_dir": str(run), "authorised_by": "lead"}))
    assert mod.verify_resume(run, "lead", **kw) == rec                  # crashed pass, clean state: allowed
    with pytest.raises(SystemExit, match="same authorisation"):
        mod.verify_resume(run, "someone else", **kw)
    log_result(workstream="bioassess", experiment="bioassess_test", model="m", metric="r2", value=0.1, split="s",
               registry_path=reg)
    with pytest.raises(SystemExit, match="registry already holds"):
        mod.verify_resume(run, "lead", **kw)
    reg.unlink()
    marker.write_text(json.dumps({"run_dir": str(run), "authorised_by": "lead", "resumed": "t"}))
    with pytest.raises(SystemExit, match="already resumed"):
        mod.verify_resume(run, "lead", **kw)
    marker.write_text(json.dumps({"run_dir": str(run), "authorised_by": "lead"}))
    (run / "resume.json").write_text("{}")
    with pytest.raises(SystemExit, match="already resumed once"):
        mod.verify_resume(run, "lead", **kw)
    (run / "resume.json").unlink()
    (run / "scratch_registry.jsonl").write_text('{"x": 1}\n')
    with pytest.raises(SystemExit, match="scratch registry"):
        mod.verify_resume(run, None, dry=True)


def test_bioinfer_ml_recovers_a_synthetic_gradient():
    """bio.infer-style ML inference: unimodal synthetic taxa along a gradient -> the inferred value
    tracks the true one; rare taxa (< min_occ) are not modelled."""
    from sentinel.models.bioassess.bioinfer import BioInferML

    rs = np.random.default_rng(0)
    n, T = 1500, 40
    y = rs.uniform(-2, 2, n)
    optima = np.linspace(-2.5, 2.5, T)
    p = 1 / (1 + np.exp(-(1.5 - 2.0 * (y[:, None] - optima[None, :]) ** 2)))
    P = rs.random((n, T)) < p
    P[:, 0] = False
    P[:5, 0] = True                                                   # a rare taxon
    m = BioInferML(min_occ=30).fit(P[:1000], y[:1000])
    assert 0 not in set(m.taxa_idx_)
    pred = m.predict(P[1000:])
    assert np.corrcoef(pred, y[1000:])[0, 1] > 0.9
    assert pred.min() >= y[:1000].min() - 1e-9 and pred.max() <= y[:1000].max() + 1e-9


def test_xtrack_training_rows_and_split_invariants():
    """M5-X: fold pools exclude region g (same continent), frozen-test rows and anything within 1 km of a
    fold-g test location (any source/region); the split validator catches leaks."""
    from sentinel.models.bioassess import xtrack as X

    cols = ["block", "role", "fold", "source", "id", "survey", "year", "region", "uk_role", "lat", "lon", "cluster"]
    rows = [
        ("X-L", "test", "WMT", "nla", "t1", "nla22", 2022, "WMT", None, 45.0, -110.0, "nla_1"),
        ("X-L", "train_pool", None, "nla", "a", "nla17", 2017, "WMT", None, 46.0, -111.0, "nla_2"),
        ("X-L", "train_pool", None, "nrsa", "b", "nrsa1314", 2013, "CPL", None, 35.0, -80.0, "nrsa_1"),
        ("X-L", "train_pool", None, "nrsa", "near", "nrsa1314", 2013, "NPL", None, 45.0005, -110.0005, "nrsa_9"),
        ("X-L", "train_pool", None, "nrsa", "frz", "nrsa1819", 2018, "CPL", None, 36.0, -81.0, "nrsa_2"),
        ("X-L", "train_pool", None, "uk", "u1", "uk_ea", 2019, "THAMES - WEST", "pool_fit", 51.5, -1.0, "uk_1"),
        ("X-U", "test", "THAMES - WEST", "uk", "u2", "uk_ea", 2022, "THAMES - WEST", "pool_fit", 51.6, -1.2, "uk_2"),
        ("X-U", "train_pool", None, "uk", "u1", "uk_ea", 2019, "THAMES - WEST", "pool_fit", 51.5, -1.0, "uk_1"),
        ("X-U", "train_pool", None, "uk", "u3", "uk_ea", 2018, "ANGLIAN - CENTRAL", "pool_fit", 52.5, 0.5, "uk_3"),
        ("X-U", "train_pool", None, "uk", "same_site", "uk_ea", 2015, "ANGLIAN - CENTRAL", "pool_fit", 51.6001,
         -1.2001, "uk_2"),
        ("X-U", "train_pool", None, "nla", "a", "nla17", 2017, "WMT", None, 46.0, -111.0, "nla_2"),
    ]
    S = pd.DataFrame(rows, columns=cols)
    frozen = {("nrsa", "frz"): "M5 test"}
    tr = X.training_rows(S, "X-L", "WMT", frozen)
    assert set(tr["id"]) == {"b", "u1"}           # WMT NLA, frozen NRSA and the 70-m neighbour removed; UK kept
    rep = {}
    tr = X.training_rows(S, "X-U", "THAMES - WEST", frozen, report=rep)
    assert set(tr["id"]) == {"u3", "a"}            # area g dropped; same site in another area removed by buffer
    assert rep["removed_buffer"] == 1
    assert X.validate_splits(S) == []
    bad, _ = X.validate_training_pools(S, frozen)
    assert bad == []
    leak = pd.concat([S, S.iloc[[0]].assign(role="train_pool", fold=None)], ignore_index=True)
    assert any("training pool" in m for m in X.validate_splits(leak))
    late = pd.concat([S, S.iloc[[7]].assign(id="u9", year=2023)], ignore_index=True)
    assert any(">= 2021" in m for m in X.validate_splits(late))


def _xtrack_toy_split():
    cols = ["block", "role", "fold", "source", "id", "survey", "year", "region", "uk_role", "lat", "lon", "cluster"]
    rows = [
        ("X-L", "test", "WMT", "nla", "t1", "nla22", 2022, "WMT", None, 45.0, -110.0, "nla_1"),
        ("X-L", "train_pool", None, "nrsa", "b", "nrsa1314", 2013, "CPL", None, 35.0, -80.0, "nrsa_1"),
        ("X-L", "train_pool", None, "uk", "u1", "uk_ea", 2019, "THAMES - WEST", "pool_fit", 51.5, -1.0, "uk_1"),
        ("X-L", "train_pool", None, "uk", "u3", "uk_ea", 2018, "ANGLIAN - CENTRAL", "pool_fit", 52.5, 0.5, "uk_3"),
        ("X-L", "excluded", None, "uk", "u2", "uk_ea", 2022, "THAMES - WEST", "pool_fit", 51.6, -1.2, "uk_2"),
        ("X-U", "test", "THAMES - WEST", "uk", "u2", "uk_ea", 2022, "THAMES - WEST", "pool_fit", 51.6, -1.2, "uk_2"),
        ("X-U", "train_pool", None, "uk", "u1", "uk_ea", 2019, "THAMES - WEST", "pool_fit", 51.5, -1.0, "uk_1"),
        ("X-U", "train_pool", None, "uk", "u3", "uk_ea", 2018, "ANGLIAN - CENTRAL", "pool_fit", 52.5, 0.5, "uk_3"),
        ("X-U", "train_pool", None, "nla", "t1", "nla22", 2022, "WMT", None, 45.0, -110.0, "nla_1"),   # X-L test
        ("X-U", "train_pool", None, "nla", "a7", "nla07", 2007, "CPL", None, 33.0, -84.0, "nla_7"),
        ("X-U", "train_pool", None, "nrsa", "b", "nrsa1314", 2013, "CPL", None, 35.0, -80.0, "nrsa_1"),
    ]
    return pd.DataFrame(rows, columns=cols)


def test_xtrack_cross_block_test_rows_never_in_any_pool():
    """C-MXR-1: X-L test rows are never in an X-U pool (and X-U test/excluded never in X-L)."""
    from sentinel.models.bioassess import xtrack as X

    S = _xtrack_toy_split()
    tr = X.training_rows(S, "X-U", "THAMES - WEST", {})
    assert "t1" not in set(tr["id"]) and set(tr["id"]) == {"u3", "a7", "b"}
    tr = X.training_rows(S, "X-L", "WMT", {})
    assert "u2" not in set(tr["id"]) and set(tr["id"]) == {"b", "u1", "u3"}
    bad, _ = X.validate_training_pools(S, {})
    assert bad == []


def _tiny_checkpoint(path):
    import torch

    from sentinel.models.bioassess import bioset as BS

    tok = pd.DataFrame({"visit_id": ["v"], "phylum": ["Arthropoda"], "class": ["Insecta"], "order": ["Ephemeroptera"],
                        "family": ["Baetidae"], "genus": ["Baetis"], "name": ["Baetis rhodani"], "count": [1.0]})
    v = BS.TaxonVocab.build(tok)
    torch.save({"vocab": v.to_dict(), "encoder_state": {}, "pool_allowed": [], "cfg": {}}, path)


def test_xtrack_encoder_loader_refuses_auto_m5_and_out_of_pool_encoders(tmp_path, monkeypatch):
    """C-MXR-2 via the real loaders: 'auto' (xtrack loader and bioassess_cv.resolve_pretrained under M5-X), the M5
    UK encoder, an encoder without a training-id record and a full-UK encoder (ids outside the fold pool) are
    refused; a per-fold encoder whose ids lie in the pool loads through cv.PretrainedEncoder."""
    import importlib.util

    from sentinel.models.bioassess import xtrack as X

    S = _xtrack_toy_split()
    g = "THAMES - WEST"
    with pytest.raises(X.EncoderLeakError, match="auto"):
        X.load_pretrained_encoder("auto", S, "X-U", g, {})
    m5 = X.paths.RUNS_ROOT / "bioassess" / "uk_pretrain_20261022T091937" / "encoder.pt"
    if m5.exists():
        with pytest.raises(X.EncoderLeakError, match="forbidden M5 checkpoint"):
            X.load_pretrained_encoder(m5, S, "X-U", g, {})
    ck = tmp_path / "enc.pt"
    _tiny_checkpoint(ck)
    with pytest.raises(X.EncoderLeakError, match="no training-id record"):
        X.load_pretrained_encoder(ck, S, "X-U", g, {})
    X.write_train_ids_sidecar(ck, [("uk", "u1"), ("uk", "u3")])            # all-UK <= 2020: u1 is in area g
    with pytest.raises(X.EncoderLeakError, match="outside this fold's training pool"):
        X.load_pretrained_encoder(ck, S, "X-U", g, {})
    X.write_train_ids_sidecar(ck, [("uk", "u3")])
    with pytest.raises(X.EncoderLeakError, match="outside"):                 # frozen-test row in the record
        X.load_pretrained_encoder(ck, S, "X-U", g, {("uk", "u3"): "frozen"})
    enc = X.load_pretrained_encoder(ck, S, "X-U", g, {})                     # clean per-fold encoder
    assert enc.path == str(ck)
    spec = importlib.util.spec_from_file_location(
        "bioassess_cv_mod", X.paths.REPO_ROOT / "scripts/train/bioassess_cv.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setenv("SENTINEL_TRACK", "M5-X")
    with pytest.raises(SystemExit, match="M5-X never uses"):
        mod.resolve_pretrained({"pretrained": "auto"}, None)


def test_xtrack_fit_id_guard_refuses_protected_rows():
    from sentinel.models.bioassess import xtrack as X

    S = _xtrack_toy_split()
    X.assert_fit_ids([("uk", "u3"), ("nrsa", "b")], S, {})
    with pytest.raises(X.EncoderLeakError, match="protected"):
        X.assert_fit_ids([("nla", "t1")], S, {})                               # X-L test row
    with pytest.raises(X.EncoderLeakError, match="protected"):
        X.assert_fit_ids([("nrsa", "b")], S, {("nrsa", "b"): "frozen"})


M5X_NON_FITTING_SCRIPTS = {"bioassess_x_prereg.py", "bioassess_x_power.py"}   # pre-registration utilities only


def test_m5x_static_entry_points_use_fit_guard_and_guarded_encoder_loader():
    """Lead condition (2026-11-09): every M5-X script/entry point calls assert_fit_ids and loads encoders only via
    xtrack.load_pretrained_encoder. Fails on any new M5-X script (scripts/**/bioassess_x_*.py) or module
    (sentinel/models/bioassess/x*.py) that does not, or that touches a raw encoder path."""
    import re
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    scripts = sorted(repo.glob("scripts/**/bioassess_x_*.py"))
    modules = sorted(p for p in (repo / "sentinel/models/bioassess").glob("x*.py"))
    assert scripts and modules
    forbidden = [r"PretrainedEncoder\(", r"torch\.load\(", r"resolve_pretrained", r"uk_pretrain_latest\b(?!\.json\"\))",
                 r"init_encoder\s*="]
    problems = []
    for f in scripts + modules:
        src = f.read_text()
        if f.name == "xtrack.py":       # the loader itself: strip the one sanctioned function before scanning
            src = re.sub(r"def load_pretrained_encoder\(.*?\n(?=def )", "", src, flags=re.S)
        for pat in forbidden:
            if re.search(pat, src):
                problems.append(f"{f.relative_to(repo)}: forbidden raw-encoder pattern {pat}")
        if f in scripts and f.name not in M5X_NON_FITTING_SCRIPTS and "assert_fit_ids(" not in src:
            problems.append(f"{f.relative_to(repo)}: does not call assert_fit_ids")
        if re.search(r"\bpretrained\s*=\s*(?!None\b)", src) and "require_guarded(" not in src and f.name != "xtrack.py":
            problems.append(f"{f.relative_to(repo)}: passes a pretrained encoder without require_guarded")
    assert problems == [], problems


def test_m5x_require_guarded_marks(tmp_path):
    from sentinel.models.bioassess import xtrack as X

    S = _xtrack_toy_split()
    ck = tmp_path / "enc.pt"
    _tiny_checkpoint(ck)
    X.write_train_ids_sidecar(ck, [("uk", "u3")])
    enc = X.load_pretrained_encoder(ck, S, "X-U", "THAMES - WEST", {})
    X.require_guarded(enc, "X-U", "THAMES - WEST")
    X.require_guarded(None, "X-U", "THAMES - WEST")
    with pytest.raises(X.EncoderLeakError):
        X.require_guarded(enc, "X-U", "ANGLIAN - CENTRAL")
    with pytest.raises(X.EncoderLeakError):
        X.require_guarded(object(), "X-U", "THAMES - WEST")
