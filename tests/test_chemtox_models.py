# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Tests for SENTINEL-ChemTox (M4): split integrity, featuriser determinism, tiny smoke fits.

Split-integrity tests read the converted ADORE release through the data workstream's loader
(``sentinel.data.sources.adore``) and are skipped when the release is not on this machine. No test
touches the network: the TabPFN and transformer smoke tests run only when the weights are already
in the local cache.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.data.sources import adore as A
from sentinel.models.chemtox import evaluate as E
from sentinel.models.chemtox import features as F
from sentinel.models.chemtox import severity as S
from sentinel.models.chemtox import xgb as XG
from sentinel.models.chemtox.cv import expand_grid, run_cv
from sentinel.models.chemtox.data import ChemToxDataset, FeatureSpec

CHALLENGE = "t-F2F"
SPLITS = (A.PRIMARY_SPLIT, A.SECONDARY_SPLIT)


def _have_release(challenge: str = CHALLENGE) -> bool:
    try:
        return A.challenge_path(challenge).exists()
    except Exception:  # noqa: BLE001
        return False


needs_adore = pytest.mark.skipif(not _have_release(), reason="ADORE release not converted here")


@pytest.fixture(scope="module")
def ds() -> ChemToxDataset:
    return ChemToxDataset.load(CHALLENGE, A.PRIMARY_SPLIT, specs=FeatureSpec(fingerprint="MACCS"))


@pytest.fixture(scope="module")
def train_rows(ds) -> pd.DataFrame:
    return ds.rows(np.flatnonzero(ds.train_mask()))


# ----------------------------------------------------------------------------- split integrity


@needs_adore
@pytest.mark.parametrize("split", SPLITS)
def test_no_chemical_in_both_train_and_test(split):
    """Occurrence and Murcko-scaffold splits must share no chemical between train and test."""
    df = A.load_challenge(CHALLENGE, columns=[A.CHEM_KEY])
    overlap = A.chemical_overlap(df, split)
    assert overlap["chem_overlap"] == 0, overlap
    assert overlap["n_train"] > 0 and overlap["n_test"] > 0


@needs_adore
@pytest.mark.parametrize("split", SPLITS)
def test_cv_folds_are_chemical_disjoint(split):
    """Inside the training portion, no chemical is shared between a fold and the other folds."""
    d = ChemToxDataset.load(CHALLENGE, split, specs=FeatureSpec(fingerprint=None))
    chem = d.chem
    folds = d.cv_splits()
    assert len(folds) == 5
    for _k, trn, val in folds:
        assert not (set(chem[trn]) & set(chem[val]))


@needs_adore
def test_train_test_and_folds_partition_the_rows(ds):
    tr, te = ds.train_mask(), ds.test_mask()
    assert not (tr & te).any()
    fold_union = np.zeros(len(ds.df), dtype=bool)
    for k in ds.folds():
        m = ds.fold_mask(k)
        assert not (fold_union & m).any()
        fold_union |= m
    assert fold_union.sum() <= tr.sum()  # some splits leave 'trainvalid' rows without a fold
    ds.assert_no_chemical_leakage()


@needs_adore
def test_leakage_guard_raises_on_a_planted_overlap_but_allows_the_cross_taxa_challenge():
    """A chemical in both partitions stops the run, except where ADORE shares them by design."""
    from sentinel.models.chemtox.data import SHARED_CHEMICALS_BY_DESIGN

    d = ChemToxDataset.load("a-CA2F-diff", A.PRIMARY_SPLIT, specs=FeatureSpec(fingerprint=None))
    d.assert_no_chemical_leakage()
    d.df.loc[np.flatnonzero(d.test_mask())[:3], A.CHEM_KEY] = d.chem[d.train_mask()][0]
    with pytest.raises(AssertionError):
        d.assert_no_chemical_leakage()
    assert "a-CA2F-same" in SHARED_CHEMICALS_BY_DESIGN
    same = ChemToxDataset.load("a-CA2F-same", A.PRIMARY_SPLIT, specs=FeatureSpec(fingerprint=None))
    assert same.split_report()["chem_overlap"] > 0
    same.assert_no_chemical_leakage()  # reported, not raised: taxonomic extrapolation by design


@needs_adore
def test_totally_random_split_is_rejected_as_leaky():
    """The leakage demonstration of ADORE: the totally-random column does share chemicals."""
    df = A.load_challenge(CHALLENGE, columns=[A.CHEM_KEY])
    assert A.chemical_overlap(df, "split_totallyrandom")["chem_overlap"] > 0


# ----------------------------------------------------------------------------- featurisation


@needs_adore
def test_published_feature_block_matches_published_width(train_rows):
    """The published non-structural block has 37 columns (Gasser et al. Table 1)."""
    assert F.published_feature_count(train_rows) == F.PUBLISHED_OTHER_FEATURE_COUNT


@needs_adore
def test_informative_bit_counts_match_published(train_rows):
    """Published informative-bit counts for the occurrence split (Gasser et al. Table 1)."""
    expected = {"chem_MACCS_fp": 141, "chem_pcp_fp": 471, "chem_Morgan_fp": 380, "chem_ToxPrint_fp": 172}
    df = A.load_challenge(CHALLENGE, columns=list(expected))
    a = df[A.PRIMARY_SPLIT].astype("string")
    tr = df[(a.notna() & (a != "test")).to_numpy()]
    for col, n in expected.items():
        assert F.InformativeBits(column=col).fit(tr).n_features == n, col


@needs_adore
def test_featuriser_is_deterministic_and_fit_on_train_only(ds, train_rows):
    spec = FeatureSpec(fingerprint="MACCS")
    b1 = ds.builder(spec).fit(train_rows)
    b2 = ds.builder(spec).fit(train_rows)
    X1, X2 = b1.transform(ds.df), b2.transform(ds.df)
    assert np.array_equal(X1, X2)
    assert b1.names_ == b2.names_ and b1.n_features == 178  # 141 MACCS bits + 37 published
    # levels come from the training rows only: a level seen only in test adds no column
    te = ds.rows(np.flatnonzero(ds.test_mask()))
    assert b1.transform(te).shape[1] == b1.n_features


def test_published_features_transform_is_pure():
    rows = pd.DataFrame({
        F.DURATION_COL: [24.0, 96.0, 48.0],
        "test_media_type": ["FW", "SW", "FW"],
        "test_exposure_type": ["S", "R", "S"],
        "result_conc1_type": ["A", "T", "A"],
        "tax_eco_climate": ["C_D", "M_D", "A"],
        "tax_eco_ecozone": ["TH", "MP_TH", "TN"],
        "tax_eco_food": ["D_H", "C", "O"],
        "tax_eco_migrate2": ["migration", "no_migration", "migration"],
        "chem_mw": [100.0, 200.0, 300.0],
        "chem_ws": [1.0, 2.0, 3.0],
        "chem_mp": [10.0, 20.0, 30.0],
        "chem_rdkit_clogp": [1.0, 2.0, 3.0],
        "tax_lh_amd": [1.0, 2.0, 3.0],
        "tax_lh_licm": [1.0, 2.0, 3.0],
        "tax_ps_ampv": [1.0, 2.0, 3.0],
        "tax_ps_ampkap": [1.0, 2.0, 3.0],
        "tax_ps_amppm": [1.0, 2.0, 3.0],
    })
    pf = F.PublishedFeatures().fit(rows)
    X1, X2 = pf.transform(rows), pf.transform(rows)
    assert np.array_equal(X1, X2) and X1.shape[1] == pf.n_features == len(pf.names_)
    # duration is scaled to [0, 1] with 24 h -> 0 and 96 h -> 1
    assert X1[0, 0] == pytest.approx(0.0) and X1[1, 0] == pytest.approx(1.0)
    # many-hot: 'C_D' sets both C and D
    ci = [pf.names_.index("tax_eco_climate~C"), pf.names_.index("tax_eco_climate~D")]
    assert X1[0, ci].tolist() == [1.0, 1.0]
    with_mig = F.PublishedFeatures(include_migration=True).fit(rows)
    assert with_mig.n_features == pf.n_features + 1


def test_rdkit_featuriser_is_deterministic():
    smiles = ["c1ccccc1", "CCO", "CC(=O)Oc1ccccc1C(=O)O"]
    X1, names1 = F.rdkit_features(smiles, maccs=True, morgan=True, descriptors=True, n_bits=256)
    X2, names2 = F.rdkit_features(smiles, maccs=True, morgan=True, descriptors=True, n_bits=256)
    assert names1 == names2 and np.array_equal(np.nan_to_num(X1), np.nan_to_num(X2))
    assert X1.shape == (3, 167 + 256 + len(F.RDKIT_DESCRIPTORS))
    # benzene: one aromatic ring, no rotatable bonds
    d = dict(zip(F.RDKIT_DESCRIPTORS, F.rdkit_descriptors("c1ccccc1")))
    assert d["NumAromaticRings"] == 1 and d["NumRotatableBonds"] == 0
    assert np.isnan(F.maccs_bits("not-a-smiles")).all()


# ----------------------------------------------------------------------------- metrics


def test_macro_metrics_follow_the_published_definitions():
    y = np.array([0.0, 2.0, 1.0, 1.0])
    p = np.array([1.0, 2.0, 1.0, 2.0])          # errors 1, 0, 0, 1
    chem = np.array(["a", "a", "b", "b"])
    taxon = np.array(["x", "y", "x", "y"])
    rep = E.report(y, p, chem=chem, taxon=taxon)
    assert rep["rmse"] == pytest.approx(np.sqrt(0.5))
    # per chemical: a -> mean(1, 0) = 0.5, b -> mean(0, 1) = 0.5 -> sqrt(0.5)
    assert rep["rmse_chemical"] == pytest.approx(np.sqrt(0.5))
    assert rep["rmse_taxon"] == pytest.approx(np.sqrt(0.5))
    assert rep["rmse_macro_cells"] == 4
    assert rep["within_1_log"] == pytest.approx(1.0)
    assert rep["median_abs_fold_error"] == pytest.approx(np.median([10.0, 1.0, 1.0, 10.0]))


def test_grouped_rmse_weights_groups_not_rows():
    y = np.zeros(11)
    p = np.concatenate([np.ones(10), np.zeros(1)])   # group 'a' all wrong, 'b' right
    chem = np.array(["a"] * 10 + ["b"])
    assert E.grouped_rmse(y, p, chem) == pytest.approx(np.sqrt(0.5))
    from sentinel.eval.metrics import rmse

    assert rmse(y, p) == pytest.approx(np.sqrt(10 / 11))


def test_bootstrap_ci_resamples_chemicals():
    rng = np.random.default_rng(0)
    chem = np.repeat([f"c{i}" for i in range(40)], 5)
    y = rng.normal(size=200)
    p = y + rng.normal(scale=0.5, size=200)
    cis = E.bootstrap_report(y, p, chem=chem, taxon=chem, metrics=("rmse",), n_boot=100, seed=1)
    ci = cis["rmse"]
    assert ci["ci_low"] < ci["point"] < ci["ci_high"]
    assert ci["unit"] == "group" and ci["n_groups"] == 40


# ----------------------------------------------------------------------------- smoke fits


def test_expand_grid_is_deterministic():
    g = {"a": (1, 2), "b": ("x",)}
    assert expand_grid(g) == [{"a": 1, "b": "x"}, {"a": 2, "b": "x"}]


@needs_adore
def test_xgboost_smoke_fit_on_one_fold(ds):
    spec = FeatureSpec(fingerprint="MACCS")
    r = run_cv(ds, spec, XG.make_xgb(0), {"n_estimators": 10, "max_depth": 3}, model_name="xgb",
               folds=[0])
    assert len(r.fold_metrics) == 1
    m = r.fold_metrics[0]
    assert np.isfinite(m["rmse"]) and 0.3 < m["rmse"] < 3.0
    # the bit filter is re-fit inside the fold, so the width is near but need not equal the
    # full-training-partition width of 178 (141 MACCS bits + 37 published columns)
    assert 170 <= m["n_features"] <= 185
    assert np.isfinite(r.oof[ds.fold_mask(0)]).all()
    assert np.isnan(r.oof[ds.test_mask()]).all()  # test rows are never predicted in CV


@needs_adore
def test_random_forest_smoke_fit(ds):
    r = run_cv(ds, FeatureSpec(fingerprint="MACCS"), XG.make_rf(0),
               {"n_estimators": 10, "max_depth": 6, "n_jobs": 2}, model_name="rf", folds=[0])
    assert np.isfinite(r.fold_metrics[0]["rmse"])


def _hf_cached(repo: str) -> bool:
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    return any(root.glob(f"models--{repo.replace('/', '--')}*")) if root.exists() else False


def _tabpfn_cached() -> bool:
    """TabPFN v2 caches its checkpoint outside the HF hub layout (``~/.cache/tabpfn``)."""
    cache = Path(os.environ.get("TABPFN_MODEL_CACHE_DIR", Path.home() / ".cache" / "tabpfn"))
    return cache.exists() and any(cache.glob("*regressor*.ckpt"))


@needs_adore
@pytest.mark.skipif(not _tabpfn_cached(),
                    reason="TabPFN v2 weights not in the local cache (tests must not use the network)")
def test_tabpfn_smoke_fit(ds):
    # import the wrapper first: it installs the scikit-learn compatibility shim TabPFN 2.x needs
    tm = pytest.importorskip("sentinel.models.chemtox.tabpfn_model")
    pytest.importorskip("tabpfn")
    make_tabpfn = tm.make_tabpfn

    r = run_cv(ds, FeatureSpec(fingerprint="MACCS"), make_tabpfn(0),
               {"device": "cpu", "n_estimators": 1, "max_train_rows": 600, "max_features": 40},
               model_name="tabpfn", folds=[0])
    assert np.isfinite(r.fold_metrics[0]["rmse"])


@needs_adore
def test_chemprop_smoke_fit():
    pytest.importorskip("chemprop")
    pytest.importorskip("lightning")
    from sentinel.models.chemtox.chemprop_model import ChempropToxModel

    d = ChemToxDataset.load(CHALLENGE, A.PRIMARY_SPLIT,
                            specs=FeatureSpec(fingerprint=None), extra_columns=("chem_rdkit_can_smiles",))
    trn = np.flatnonzero(d.fold_mask(1))[:128]
    val = np.flatnonzero(d.fold_mask(0))[:64]
    Xtr, Xva, _b = d.matrices(FeatureSpec(fingerprint=None), trn, val)
    m = ChempropToxModel(params={"max_epochs": 1, "batch_size": 32, "message_hidden": 32,
                                 "ffn_hidden": 32, "depth": 2, "species_dim": 4,
                                 "accelerator": "cpu"}, seed=0)
    m.fit(Xtr, d.y[trn], train_rows=d.rows(trn))
    p = m.predict(Xva, rows=d.rows(val))
    assert p.shape == (len(val),) and np.isfinite(p).all()
    enc = m.enc_
    assert enc.n_species == len(enc.species_) + 1  # out-of-vocabulary slot exists
    assert enc.species_index(d.rows(val)).max() <= len(enc.species_)


# ----------------------------------------------------------------------------- severity (synthetic)


def test_ssd_recovers_a_lognormal_and_hc5_is_its_5th_percentile():
    rng = np.random.default_rng(0)
    taxa = [f"sp{i}" for i in range(200)]
    values = rng.normal(loc=-5.0, scale=0.8, size=200)   # log10 mol/L
    ssd = S.species_sensitivity_distribution(dict(zip(taxa, values)))
    assert ssd.n_taxa == 200
    assert ssd.mu == pytest.approx(-5.0, abs=0.2) and ssd.sigma == pytest.approx(0.8, abs=0.15)
    h = S.hc5(ssd, molar_mass_g_mol=100.0)
    assert h["hc5_log10_mol_per_L"] == pytest.approx(ssd.mu - 1.6449 * ssd.sigma, abs=1e-3)
    assert h["ci_low"] < h["hc5_log10_mol_per_L_with_model_error"] < h["ci_high"]
    assert h["hc5_mg_per_L"] == pytest.approx(10 ** h["hc5_log10_mol_per_L_with_model_error"] * 1e5)


def test_ssd_averages_repeated_taxa_and_needs_enough_taxa():
    s = pd.Series([-5.0, -3.0, -4.0], index=["a", "a", "b"])
    ssd = S.species_sensitivity_distribution(s, min_taxa=2)
    assert ssd.n_taxa == 2 and ssd.mu == pytest.approx(-4.0)
    with pytest.raises(ValueError):
        S.species_sensitivity_distribution(s, min_taxa=5)


def test_model_error_widens_the_hc5_band_but_not_the_point():
    ssd_a = S.SSD(mu=-5.0, sigma=0.5, n_taxa=20)
    ssd_b = S.SSD(mu=-5.0, sigma=0.5, n_taxa=20, pred_sd=0.9)
    assert ssd_b.total_sigma > ssd_a.total_sigma
    a, b = S.hc5(ssd_a), S.hc5(ssd_b)
    assert a["hc5_log10_mol_per_L"] == pytest.approx(b["hc5_log10_mol_per_L"])
    assert (b["ci_high"] - b["ci_low"]) > (a["ci_high"] - a["ci_low"])


def test_pec_dilution_arithmetic():
    out = S.predicted_environmental_concentration(1.0, discharge_m3_s=10.0, mixing_window_h=1.0,
                                                  molar_mass_g_mol=100.0)
    # 1 kg into 10 m3/s * 3600 s = 36,000 m3 = 3.6e7 L -> 1e6 mg / 3.6e7 L
    assert out["dilution_volume_m3"] == pytest.approx(36000.0)
    assert out["pec_mg_per_L"] == pytest.approx(1e6 / 3.6e7)
    assert out["pec_mol_per_L"] == pytest.approx(out["pec_mg_per_L"] * 1e-3 / 100.0)
    # halving the dilution factor doubles the concentration
    half = S.predicted_environmental_concentration(1.0, discharge_m3_s=10.0, mixing_window_h=1.0,
                                                   dilution_factor=0.5)
    assert half["pec_mg_per_L"] == pytest.approx(2 * out["pec_mg_per_L"])
    for bad in ({"amount_in_water_kg": -1.0, "discharge_m3_s": 1.0},
                {"amount_in_water_kg": 1.0, "discharge_m3_s": 0.0}):
        with pytest.raises(ValueError):
            S.predicted_environmental_concentration(**bad)


def test_risk_quotient_and_pnec():
    assert S.risk_quotient(2.0, 1.0) == pytest.approx(2.0)
    assert S.pnec_from_hc5(10.0, assessment_factor=5.0) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        S.risk_quotient(1.0, 0.0)
    with pytest.raises(ValueError):
        S.pnec_from_hc5(1.0, assessment_factor=0.0)


def test_severity_score_chain_and_monotonicity():
    rng = np.random.default_rng(1)
    preds = {f"sp{i}": v for i, v in enumerate(rng.normal(-5.0, 0.5, size=30))}
    base = S.severity_score(preds, amount_in_water_kg=100.0, discharge_m3_s=5.0,
                            molar_mass_g_mol=150.0, pred_sd=0.9)
    assert base.risk_quotient > 0
    assert base.risk_quotient_low <= base.risk_quotient <= base.risk_quotient_high
    assert base.pec_mg_per_L == pytest.approx(base.inputs["amount_in_water_kg"] * 1e6
                                              / (base.hc5["n_taxa"] * 0 + base.pec["dilution_volume_m3"] * 1e3))
    bigger = S.severity_score(preds, amount_in_water_kg=1000.0, discharge_m3_s=5.0,
                              molar_mass_g_mol=150.0, pred_sd=0.9)
    assert bigger.risk_quotient == pytest.approx(10 * base.risk_quotient)
    diluted = S.severity_score(preds, amount_in_water_kg=100.0, discharge_m3_s=50.0,
                               molar_mass_g_mol=150.0, pred_sd=0.9)
    assert diluted.risk_quotient == pytest.approx(base.risk_quotient / 10)
    more_toxic = S.severity_score({k: v - 1.0 for k, v in preds.items()}, amount_in_water_kg=100.0,
                                  discharge_m3_s=5.0, molar_mass_g_mol=150.0, pred_sd=0.9)
    assert more_toxic.risk_quotient > base.risk_quotient
    assert "unvalidated" in " ".join(base.caveats)


# ----------------------------------------------------------------------------- auxiliary data rules


@needs_adore
def test_excluded_keys_covers_every_split_and_both_identifiers():
    """The exclusion key set is the union over all split columns, on CAS *and* InChIKey."""
    from sentinel.models.chemtox import aux as AUX

    strict = AUX.excluded_keys(CHALLENGE)
    one = AUX.excluded_keys(CHALLENGE, splits=(A.PRIMARY_SPLIT,))
    assert len(strict.cas) > len(one.cas) > 0          # more splits -> more held-out chemicals
    assert len(strict.inchikey) > 0
    assert strict.n_test_rows > one.n_test_rows


@needs_adore
def test_adore_auxiliary_has_no_test_chemical_of_the_evaluated_challenge():
    from sentinel.models.chemtox import aux as AUX

    rows, info = AUX.adore_auxiliary(CHALLENGE)
    keys = AUX.excluded_keys(CHALLENGE)
    AUX.assert_no_test_chemicals(rows, keys)           # the assertion the rule rests on
    assert info["n_rows"] == len(rows)
    assert set(rows["aux_source"]) <= {"t-C2C", "t-A2A"}
    # planting one back must be caught, by CAS and by InChIKey independently
    bad_cas = rows.copy()
    bad_cas.loc[bad_cas.index[:2], A.CHEM_KEY] = next(iter(keys.cas))
    with pytest.raises(AssertionError):
        AUX.assert_no_test_chemicals(bad_cas, keys)
    bad_ik = rows.copy()
    bad_ik.loc[bad_ik.index[:2], A.CHEM_KEY] = "0000-00-0"
    bad_ik.loc[bad_ik.index[:2], "chem_inchikey"] = next(iter(keys.inchikey))
    with pytest.raises(AssertionError):
        AUX.assert_no_test_chemicals(bad_ik, keys)


@needs_adore
@pytest.mark.skipif(not (Path.home() / "STOCKHOLM/sentinel_data/processed/ecotox"
                         / "ecotox_aquatic.parquet").exists(),
                    reason="processed ECOTOX table not built here")
def test_ecotox_auxiliary_is_filtered_and_clean():
    from sentinel.models.chemtox import aux as AUX

    rows, info = AUX.ecotox_auxiliary(CHALLENGE)
    keys = AUX.excluded_keys(CHALLENGE)
    AUX.assert_no_test_chemicals(rows, keys, cas_col="cas", inchikey_col="chem_inchikey")
    assert info["n_dropped"] > 0 and info["n_final"] == len(rows) > 0
    assert rows["y"].notna().all() and rows["smiles"].notna().all()
    assert set(rows["endpoint"].unique()) <= set(AUX.ECOTOX_ENDPOINTS)
    assert set(rows["duration_h"].unique()) <= set(AUX.ECOTOX_DURATIONS_H)


@needs_adore
def test_drop_chemicals_removes_a_folds_validation_chemicals():
    from sentinel.models.chemtox import aux as AUX

    d = ChemToxDataset.load(CHALLENGE, A.PRIMARY_SPLIT, specs=FeatureSpec(fingerprint=None),
                            extra_columns=("chem_inchikey",))
    val = d.rows(np.flatnonzero(d.fold_mask(0)))
    kept = AUX.drop_chemicals(d.df, val[A.CHEM_KEY].unique(), val["chem_inchikey"].unique())
    assert not (set(kept[A.CHEM_KEY].astype(str)) & set(val[A.CHEM_KEY].astype(str)))
    assert 0 < len(kept) < len(d.df)


# ----------------------------------------------------------------------------- search + ensembles


def test_search_space_sampling_is_deterministic_and_within_budget():
    from sentinel.models.chemtox import spaces

    a = spaces.sample("chemprop", 32)
    b = spaces.sample("chemprop", 32)
    assert a == b and len(a) == 32
    keys = {repr(sorted(c["params"].items())) + repr(sorted(c["features"].items())) for c in a}
    assert len(keys) == 32                      # deduplicated
    for c in a:
        assert c["params"]["init"] in ("none", "chemeleon", "ecotox")
    shards = [spaces.shard(a, k, 3) for k in range(3)]
    assert sorted(i for s in shards for i, _ in s) == list(range(32))
    for model in ("tabpfn", "embeddings"):
        assert len(spaces.sample(model, 32)) == 32


def test_ensemble_mean_and_stack_are_leak_free_and_report_components():
    from sentinel.models.chemtox import ensemble as EN

    rng = np.random.default_rng(0)
    n = 500
    chem = np.array([f"c{i // 5}" for i in range(n)])
    folds = np.array([i % 5 for i in range(n)])
    y = rng.normal(size=n)
    good = y + rng.normal(scale=0.4, size=n)
    poor = y + rng.normal(scale=1.2, size=n)
    oof = {"good": good, "poor": poor}
    avg = EN.average_ensemble(oof, y, chem=chem, taxon=chem)
    stk = EN.stacked_ensemble(oof, y, chem=chem, taxon=chem, folds=folds)
    assert set(avg.component_metrics) == {"good", "poor"} == set(stk.component_metrics)
    # the stack must not be worse than the mean here, and must beat the weak component
    assert stk.metrics["rmse"] <= avg.metrics["rmse"] + 1e-9
    assert stk.metrics["rmse"] < stk.component_metrics["poor"]["rmse"]
    # leave-one-fold-out: one weight vector per held-out fold, all non-negative and summing to 1
    assert set(stk.weights) == {"0", "1", "2", "3", "4"}
    for w in stk.weights.values():
        assert all(v >= 0 for v in w.values())
        assert abs(sum(w.values()) - 1.0) < 1e-6
    assert stk.weights["0"]["good"] > stk.weights["0"]["poor"]
    assert "ensemble" in stk.name and "ensemble" in avg.name     # named as ensembles


def test_ensemble_ignores_rows_a_component_did_not_predict():
    from sentinel.models.chemtox import ensemble as EN

    y = np.array([0.0, 1.0, 2.0, 3.0])
    a = np.array([0.1, 1.1, np.nan, 3.1])
    b = np.array([0.2, 1.2, 2.2, np.nan])
    chem = np.array(["x", "y", "z", "w"])
    res = EN.average_ensemble({"a": a, "b": b}, y, chem=chem, taxon=chem)
    assert res.mask.tolist() == [True, True, False, False]
    assert res.metrics["n"] == 2
