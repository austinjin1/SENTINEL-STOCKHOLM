# MIT License — Bryan Cheng, 2026
"""Unseen-taxa track: freeze pins, the trait drop, the hardened guards and the one-time look.

Written for the stage-1 delta review (review items C-M4U-1, C-M4U-4, C-M4U-5, C-M4U-6). These
tests **fail** rather than skip when the data root exists, so a silent rebuild of a frozen file
cannot leave the suite green.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.core import paths
from sentinel.models.chemtox import unseen_taxa as U
from sentinel.models.chemtox import unseen_taxa_guard as G

DATA = paths.DATA_ROOT / "processed" / "chemtox_unseen_taxa"
PINS = paths.REPO_ROOT / "docs" / "preregistration" / "chemtox_unseen_taxa_pins.json"
PLAN = paths.REPO_ROOT / "docs" / "preregistration" / "chemtox_unseen_taxa_plan.md"


def _require_data() -> None:
    if not paths.DATA_ROOT.exists():
        pytest.skip("SENTINEL_DATA not present")
    assert DATA.is_dir(), f"{DATA} is missing — the unseen-taxa freeze must exist"


def test_frozen_files_match_the_pinned_hashes():
    """C-M4U-6: the five pinned sha256 values must still hold (fails, never skips)."""
    _require_data()
    assert PINS.is_file(), "the pins file must exist before any scoring"
    pins = json.loads(PINS.read_text())
    for name, sha in pins["files"].items():
        p = DATA / name
        assert p.is_file(), f"{name} missing"
        assert G.file_sha(p) == sha, f"{name} changed since the freeze"
    assert G.file_sha(PLAN) == pins["plan_sha256"], "the protocol changed after the freeze"
    assert G.file_sha(paths.REPO_ROOT / "sentinel/models/chemtox/unseen_taxa.py") \
        == pins["module_sha256"]
    # the frozen data must be read-only
    for name in pins["files"]:
        assert (DATA / name).stat().st_mode & 0o222 == 0, f"{name} is writable"


def test_class_level_holdout_and_strata():
    _require_data()
    test = pd.read_parquet(DATA / "test_blocks.parquet")
    pool = pd.read_parquet(DATA / "training_pool.parquet")
    U.assert_no_test_classes(pool, what="pool")
    U.assert_no_test_rows(pool, test, what="pool")
    assert not (set(pool["class"].astype(str)) & set(test["class"].astype(str)))
    # no shared lineage at any rank (the <NA> sentinel is not a lineage, so it is excluded)
    for col in ("species_number", "latin_name", "genus", "family", "tax_order"):
        shared = (set(pool[col].dropna().astype(str)) & set(test[col].dropna().astype(str))) - {"<NA>"}
        assert not shared, (col, sorted(shared)[:5])
    assert (test["parent_novel"] & ~test["chem_novel"]).sum() == 0
    rep = json.loads((DATA / "report.json").read_text())
    for b, s in rep["blocks"].items():
        sub = test[test["block"] == b]
        assert len(sub) == s["rows_all"]
        assert int(sub["in_adore_domain"].sum()) == s["rows_primary"]


def test_guards_raise_instead_of_passing_vacuously():
    """C-M4U-5: a frame without the keying column, or with an NA class, must raise."""
    with pytest.raises(U.TaxonHoldoutError):
        U.assert_no_test_classes(pd.DataFrame({"x": [1, 2]}), what="no keying column")
    with pytest.raises(U.TaxonHoldoutError):   # NA class that is not an ADORE group
        U.assert_no_test_classes(pd.DataFrame({"class": [None], "taxon_group": ["insects"]}))
    with pytest.raises(U.TaxonHoldoutError):   # ADORE-schema frame of a held-out group
        U.assert_no_test_classes(pd.DataFrame({"tax_group": ["insects"]}), group_col="tax_group")
    with pytest.raises(U.TaxonHoldoutError):   # held-out class present
        U.assert_no_test_classes(pd.DataFrame({"class": ["Insecta"], "taxon_group": ["insects"]}))
    with pytest.raises(U.TaxonHoldoutError):   # no result_id to check identity with
        U.assert_no_test_rows(pd.DataFrame({"a": [1]}), pd.DataFrame({"result_id": ["1"]}))
    # an NA class whose taxon_group is an ADORE group is allowed (65 such pool rows exist)
    assert U.assert_no_test_classes(pd.DataFrame({"class": [None], "taxon_group": ["fish"]})) == 1
    assert U.assert_no_test_classes(pd.DataFrame({"taxon_group": ["fish", "algae"]})) == 2


def test_no_trait_column_reaches_any_model():
    """C-M4U-1: TaxonBlindBuilder must remove every taxonomy-derived column."""
    _require_data()
    from sentinel.models.chemtox.data import FeatureBuilder, FeatureSpec

    pool = U.to_model_schema(pd.read_parquet(DATA / "training_pool.parquet").head(2000))
    test = U.to_model_schema(pd.read_parquet(DATA / "test_blocks.parquet").head(200))
    spec = FeatureSpec(fingerprint="MACCS", published=True, chemical="mw_logp")
    b = U.TaxonBlindBuilder(FeatureBuilder(spec))
    X, Xt = b.fit_transform(pool), b.transform(test)
    assert X.shape[1] == Xt.shape[1] == len(b.names_)
    assert b.dropped_, "the wrapper must actually drop something"
    U.assert_no_trait_columns(b.names_)
    with pytest.raises(U.TaxonHoldoutError):
        U.assert_no_trait_columns(["tax_eco_food=algae"])
    assert bool(np.isfinite(Xt).all())
    # the placeholders are constant, so they can carry no information even before being dropped
    for c in U.TRAIT_PLACEHOLDER_MULTIHOT:
        assert pool[c].nunique() == 1 and test[c].nunique() == 1


def test_chemprop_multitask_is_inadmissible():
    """C-M4U-4: an unseen tax_group must not be routed to the fish head."""
    U.assert_taxon_agnostic("chemprop", {"multitask": False})
    U.assert_taxon_agnostic("lightgbm", {"multitask": True})      # not a chemprop head
    with pytest.raises(U.TaxonHoldoutError):
        U.assert_taxon_agnostic("chemprop", {"multitask": True})


def test_surrogate_baselines_and_coverage():
    """C-M4U-2: the chemical-matched comparators work and report their applicability."""
    _require_data()
    test = pd.read_parquet(DATA / "test_blocks.parquet")
    pool = pd.read_parquet(DATA / "training_pool.parquet")
    cov = U.surrogate_coverage(test, pool)
    for b in U.BLOCKS:
        assert 0.5 < cov[b]["fraction"] <= 1.0, cov[b]
    m = U.make_fish_surrogate(0)({"reducer": "most_sensitive"}).fit(train_rows=pool)
    sub = test.head(300)
    p = m.predict(rows=sub)
    assert len(p) == len(sub) and bool(np.isfinite(p).all())
    assert sum(m.coverage_.values()) == len(sub)
    ice = U.make_ice(0)({}).fit(train_rows=pool)
    q = ice.predict(rows=sub)
    assert len(q) == len(sub) and bool(np.isfinite(q).all()) and ice.n_pairs_ > 100


def test_test_rows_cannot_be_read_without_an_authorised_look():
    """C-M4U-6: the loader is the choke point and refuses without a claim."""
    _require_data()
    assert not G._marker("insects", "AUTHORISED.json").exists(), \
        "no block may be claimed before the lead authorises the pass"
    with pytest.raises(G.ProtectedBlockError):
        G.load_test_block("insects")
    with pytest.raises(G.ProtectedBlockError):
        G.require_authorised("molluscs")
    # the pool is readable (it holds no test row) and hash-verified
    assert len(G.load_training_pool(columns=["result_id"])) > 0


# ---------------------------------------------------------------- stage-1 amendment: the channels


def test_surrogate_features_come_only_from_training_pool_rows():
    """Channel A: no test-block value may influence a surrogate feature."""
    _require_data()
    pool = U.to_model_schema(pd.read_parquet(DATA / "training_pool.parquet"))
    test = U.to_model_schema(pd.read_parquet(DATA / "test_blocks.parquet").head(400))

    s = U.SurrogateFeatures().fit(pool)
    X = s.transform(test)

    # 1. corrupting every test target (and the test rows themselves) cannot change the features
    poisoned = test.copy()
    poisoned[U.TARGET] = 999.0
    assert np.array_equal(X, s.transform(poisoned)), "a test target changed a surrogate feature"

    # 2. fitting on pool + test must be *refused*: the block asserts the hold-out on its fit frame
    with pytest.raises(U.TaxonHoldoutError):
        U.SurrogateFeatures().fit(pd.concat([pool.head(100), test.head(10)], ignore_index=True))

    # 3. the lookup tables contain no test chemical value: a chemical absent from the pool must be
    #    flagged unavailable and take the pool mean, not any test-derived number
    avail = X[:, U.SURROGATE_FEATURE_NAMES.index("surr_available")]
    cas = test["test_cas"].astype(str).str.upper().str.strip()
    pool_cas = set(pool["test_cas"].astype(str).str.upper().str.strip())
    expect = np.array([1.0 if c in pool_cas else 0.0 for c in cas])
    # parent-key matching may rescue a few rows, so availability can only ever be >= the CAS match
    assert np.all(avail >= expect)
    unmatched = avail == 0.0
    if unmatched.any():
        geo = X[unmatched, U.SURROGATE_FEATURE_NAMES.index("surr_pool_geomean")]
        assert np.allclose(geo, s.global_)
        assert np.all(X[unmatched, U.SURROGATE_FEATURE_NAMES.index("surr_n_records")] == 0)

    # 4. a hand-computed surrogate for one matched chemical must equal the feature
    m = np.flatnonzero(avail == 1.0)
    if len(m):
        i = int(m[0])
        c = cas.iloc[i]
        fish = pool[pool["taxon_group"].astype(str).str.lower() == "fish"]
        fc = fish["test_cas"].astype(str).str.upper().str.strip()
        vals = fish.loc[fc == c, U.TARGET].astype(float)
        if len(vals):
            assert np.isclose(X[i, U.SURROGATE_FEATURE_NAMES.index("surr_fish_geomean")],
                              vals.mean())
            assert np.isclose(X[i, U.SURROGATE_FEATURE_NAMES.index("surr_fish_most_sensitive")],
                              vals.min())


def test_higher_rank_channel_never_reaches_class_rank_or_below():
    """Channel B: only ranks above the held-out class, and the mapping is a function of the class."""
    _require_data()
    test = pd.read_parquet(DATA / "test_blocks.parquet")
    pool = pd.read_parquet(DATA / "training_pool.parquet")

    hr_t, hr_p = U.higher_ranks(test), U.higher_ranks(pool)
    # the feature only ever takes the pre-declared levels, so no class name can leak through it
    for c in U.HIGHER_RANK_NAMES:
        assert set(hr_t[c].astype(str)) <= set(U.HIGHER_RANK_LEVELS[c]), c
        assert set(hr_p[c].astype(str)) <= set(U.HIGHER_RANK_LEVELS[c]), c
    # no held-out class (or any rank below it) appears as a feature value
    held = {s.lower() for s in U.HELD_OUT_CLASSES}
    for c in U.HIGHER_RANK_NAMES:
        assert not ({v.lower() for v in set(hr_t[c].astype(str))} & held)
    lower = set(test["genus"].dropna().astype(str)) | set(test["family"].dropna().astype(str)) \
        | set(test["tax_order"].dropna().astype(str)) | set(test["class"].dropna().astype(str))
    block = U.HigherRankFeatures()
    assert not (set(block.names_) & lower)
    U.assert_no_class_or_lower(block.names_)
    with pytest.raises(U.TaxonHoldoutError):
        U.assert_no_class_or_lower(["class=Insecta"])
    with pytest.raises(U.TaxonHoldoutError):
        U.assert_no_class_or_lower(["tax_order=Anura"])
    # fitting the block on a frame containing held-out taxa is refused
    with pytest.raises(U.TaxonHoldoutError):
        U.HigherRankFeatures().fit(test)


def test_channel_builder_matrices_align_and_pass_both_guards():
    _require_data()
    from sentinel.models.chemtox.data import FeatureBuilder, FeatureSpec

    pool = U.to_model_schema(pd.read_parquet(DATA / "training_pool.parquet").head(3000))
    test = U.to_model_schema(pd.read_parquet(DATA / "test_blocks.parquet").head(200))
    spec = FeatureSpec(fingerprint="MACCS", published=True, chemical="mw_logp")
    plain = U.TaxonBlindBuilder(FeatureBuilder(spec))
    plain.fit(pool)
    b = U.ChannelBuilder(FeatureBuilder(spec), channels=("surrogate", "higher_rank"))
    X, Xt = b.fit_transform(pool), b.transform(test)
    assert X.shape[1] == Xt.shape[1] == len(b.names_)
    assert len(b.names_) == plain.n_features + len(U.SURROGATE_FEATURE_NAMES) \
        + len(b.higher_.names_)
    assert bool(np.isfinite(Xt).all())
    U.assert_no_trait_columns(b.names_)
    U.assert_no_class_or_lower(b.names_)


def test_residual_surrogate_model_adds_the_base_back():
    _require_data()
    from sentinel.models.chemtox.data import FeatureBuilder, FeatureSpec

    pool = U.to_model_schema(pd.read_parquet(DATA / "training_pool.parquet").head(2000))
    test = U.to_model_schema(pd.read_parquet(DATA / "test_blocks.parquet").head(100))
    spec = FeatureSpec(fingerprint="MACCS", published=True, chemical="mw_logp")
    b = U.TaxonBlindBuilder(FeatureBuilder(spec))
    X, Xt = b.fit_transform(pool), b.transform(test)

    class _Zero:                      # a model that always predicts zero residual
        def fit(self, X, y, **ctx):
            self.n_ = len(y)
            return self

        def predict(self, X, **ctx):
            return np.zeros(len(X))

    m = U.make_residual(lambda p: _Zero())({})
    m.fit(X, pool[U.TARGET].to_numpy(float), train_rows=pool)
    p = m.predict(Xt, rows=test)
    assert np.allclose(p, m.surrogate.surrogate_prediction(test, "ice"))


# ---------------------------------------------------- delta review D-1 / D-2 / D-3 regression tests


def test_fold_parent_key_blocking_actually_removes_rows():
    """D-1: parent-structure blocking must be in force, not a silent no-op."""
    _require_data()
    pool_raw = pd.read_parquet(DATA / "training_pool.parquet")
    assert "parent_key" not in pool_raw.columns, "the frozen pool has no parent_key by design"
    with pytest.raises(U.TaxonHoldoutError):
        list(U.validation_splits(pool_raw))          # must raise, never skip the blocking
    pool = U.add_parent_key(pool_raw)
    assert pool["parent_key"].notna().sum() > 0.9 * len(pool)

    folds = {c: (tr, va) for c, tr, va in U.validation_splits(pool)}
    assert set(folds) == set(U.VALIDATION_CLASSES)
    cls = pool["class"].astype("string")
    up = lambda s: s.astype(str).str.upper().str.strip()          # noqa: E731
    for c, (tr, va) in folds.items():
        v = pool.iloc[va]
        bad_cas = set(up(v["test_cas"]))
        bad_key = set(up(v["chem_inchikey"]))
        bad_pk = set(v["parent_key"].astype("string").dropna())
        # what CAS/InChIKey blocking alone would have kept
        cas_only = ((cls != c).fillna(True).to_numpy()
                    & ~up(pool["test_cas"]).isin(bad_cas).to_numpy()
                    & ~up(pool["chem_inchikey"]).isin(bad_key).to_numpy())
        extra = int(cas_only.sum() - len(tr))
        assert extra > 0, f"{c}: parent-key blocking removed nothing"
        # and nothing with a held-out parent key survived
        assert not set(pool.iloc[tr]["parent_key"].astype("string").dropna()) & bad_pk


def test_higher_rank_drops_levels_constant_in_the_fitting_frame():
    """D-2: a level that is constant in training must not become a test-only indicator."""
    _require_data()
    pool = pd.read_parquet(DATA / "training_pool.parquet")
    test = pd.read_parquet(DATA / "test_blocks.parquet")
    h = U.HigherRankFeatures().fit(pool)
    X = h.transform(pool)
    assert X.shape[1] == len(h.names_)
    # every retained column varies in the fitting frame
    for j, name in enumerate(h.names_):
        col = X[:, j]
        assert 0 < col.mean() < 1, f"{name} is constant in the training frame"
    # the four one-to-one class proxies the review found must be gone
    for proxy in ("rank_phylum=Mollusca", "rank_subphylum=Hexapoda",
                  "rank_subphylum=Conchifera", "rank_subphylum=Aculifera"):
        assert proxy in h.dropped_ and proxy not in h.names_, proxy
    # no retained column is an "is a test row" indicator: each must appear in the pool too
    Xt = h.transform(test)
    for j, name in enumerate(h.names_):
        if Xt[:, j].any():
            assert X[:, j].any(), f"{name} fires on test rows but never in training"
    # per-fold behaviour is the same rule
    pool_pk = U.add_parent_key(pool)
    for c, tr, _ in U.validation_splits(pool_pk):
        hf = U.HigherRankFeatures().fit(pool_pk.iloc[tr])
        Xf = hf.transform(pool_pk.iloc[tr])
        for j, name in enumerate(hf.names_):
            assert 0 < Xf[:, j].mean() < 1, f"fold {c}: {name} constant"


def test_one_surrogate_available_definition_everywhere():
    """D-3: the claim-gating subset has exactly one rule, shared by feature and comparator."""
    _require_data()
    pool = U.add_parent_key(pd.read_parquet(DATA / "training_pool.parquet"))
    test = U.to_model_schema(pd.read_parquet(DATA / "test_blocks.parquet"))
    cov = U.surrogate_coverage(test, pool)
    for b in U.BLOCKS:
        assert cov[b]["rule"] == "any pool record, by CAS or parent_key"
        # the fish-only number is reported but must not be the gating one
        assert cov[b]["fraction"] >= cov[b]["fish_only_fraction"]
    # the feature's flag equals the function, row for row
    sub = test[test["in_adore_domain"]].head(800)
    s = U.SurrogateFeatures().fit(pool)
    flag = s.transform(sub)[:, U.SURROGATE_FEATURE_NAMES.index("surr_available")]
    assert np.array_equal(flag.astype(bool), U.surrogate_available(sub, pool).to_numpy())


def test_ice_fit_is_fish_to_all_non_fish_and_degenerates_below_min_pairs():
    """W-D-4: the ICE fit matches its description, and the small-sample fallback is explicit."""
    _require_data()
    pool = U.add_parent_key(pd.read_parquet(DATA / "training_pool.parquet"))
    ice = U.make_ice(0)({}).fit(train_rows=pool)
    feats = U.SurrogateFeatures().fit(pool)
    assert ice.n_pairs_ == feats.ice_pairs_ == 17081
    # a fitting frame without fish rows leaves the surrogate unavailable (no silent substitution),
    # so there are no pairs and the regression must stay the identity
    nofish = pool[pool["taxon_group"].astype(str).str.lower() != "fish"]
    ice2 = U.make_ice(0)({}).fit(train_rows=nofish)
    assert ice2.surrogate.groups_present_ is False
    assert ice2.n_pairs_ < U.ICERegressionBaseline.MIN_PAIRS
    assert (ice2.a_, ice2.b_) == (0.0, 1.0)
    f2 = U.SurrogateFeatures().fit(nofish)
    assert f2.fish_groups_present_ is False and f2.ice_pairs_ == 0
    # the Actinopterygii fold keeps 6 fish rows (other fish classes survive the hold-out), so the
    # surrogate exists but has almost no coverage and the ICE fit falls below MIN_PAIRS -> identity
    folds = {c: tr for c, tr, _ in U.validation_splits(pool)}
    f3 = U.SurrogateFeatures().fit(pool.iloc[folds["Actinopterygii"]])
    assert f3.fish_groups_present_ is True and len(f3.fish_geo_) < 20
    assert f3.ice_pairs_ < U.ICERegressionBaseline.MIN_PAIRS
    assert (f3.ice_a_, f3.ice_b_) == (0.0, 1.0)


def test_guard_accepts_adore_schema_frames():
    """W-D-2: ADORE's tax_group vocabulary (incl. 'crusta') must pass, not raise."""
    assert U.assert_no_test_classes(pd.DataFrame({"tax_group": ["fish", "crusta", "algae"]})) == 3
    with pytest.raises(U.TaxonHoldoutError):
        U.assert_no_test_classes(pd.DataFrame({"tax_group": ["insects"]}))


def test_verify_freeze_checks_the_code_pins():
    """W-D-1: the module and guard sha256 are part of the freeze."""
    _require_data()
    pins = json.loads(PINS.read_text())
    assert pins["module_sha256"] == G.file_sha(
        paths.REPO_ROOT / "sentinel/models/chemtox/unseen_taxa.py")
    assert pins["guard_sha256"] == G.file_sha(
        paths.REPO_ROOT / "sentinel/models/chemtox/unseen_taxa_guard.py")
    G.verify_freeze()
