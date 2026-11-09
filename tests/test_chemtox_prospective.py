# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Tests for the ChemTox **prospective** test set and the cycle-2 models.

The integrity properties that matter are asserted here as tests, not only inside the build:

* the prospective tracks share no chemical with ADORE — by CAS *and* by InChIKey — and no ECOTOX
  ``result_id`` with ADORE (the property that makes the set a valid test set);
* ``assert_prospective_chemicals_absent`` raises when a training frame contains a prospective
  chemical, and passes on the ADORE training rows and on the ECOTOX augmentation rows;
* ADORE's own representations are reproducible from SMILES (MACCS bit-identical, Morgan radius 2 /
  2048 bits bit-identical, Crippen logP equal), which is what lets new chemicals be featurised in
  the same definitions;
* the frozen data file still has the sha256 recorded in the pre-registration;
* the new models (read-across, GPBoost with unseen taxa, the frozen CheMeleon block) behave as
  documented on tiny inputs.

Tests that need the converted ADORE release, the processed ECOTOX table or the built prospective
file are skipped when those are not on the machine; nothing here touches the network.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.data.sources import adore as A
from sentinel.models.chemtox import prospective as PRO
from sentinel.models.chemtox import readacross as RA

PREREG = Path(__file__).resolve().parents[1] / "docs/preregistration/chemtox_prospective_plan.md"


def _have(path) -> bool:
    try:
        return Path(path).exists()
    except Exception:  # noqa: BLE001
        return False


needs_adore = pytest.mark.skipif(not _have(A.challenge_path("t-F2F")),
                                 reason="ADORE release not converted here")
needs_prospective = pytest.mark.skipif(not _have(PRO.prospective_path()),
                                       reason="prospective test set not built here")
needs_ecotox = pytest.mark.skipif(not _have(PRO.ecotox_path()),
                                  reason="processed ECOTOX table not available here")


# --------------------------------------------------------------------------- the test set itself


@needs_prospective
def test_file_hash_matches_the_preregistration():
    """The frozen file must still be the one the pre-registration sealed."""
    from sentinel.core.registry import file_sha256

    text = PREREG.read_text()
    hashes = set(re.findall(r"\b[0-9a-f]{64}\b", text))
    assert file_sha256(PRO.prospective_path()) in hashes


@needs_prospective
@needs_adore
def test_novel_tracks_share_no_chemical_or_row_with_adore():
    inv = PRO.adore_inventory()
    df = pd.read_parquet(PRO.prospective_path())
    for track in ("fish_novel", "crust_novel", "algae_novel", "noncarbon"):
        sub = df[df["prospective_track"] == track]
        if sub.empty:
            continue
        rep = PRO.assert_disjoint_from_adore(sub, inv, what=f"test[{track}]")
        assert rep["cas_overlap_rows"] == 0
        assert rep["inchikey_overlap_rows"] == 0
        assert rep["result_id_overlap_rows"] == 0


@needs_prospective
@needs_adore
def test_temporal_track_is_after_the_adore_cutoff_and_not_an_adore_row():
    inv = PRO.adore_inventory()
    df = pd.read_parquet(PRO.prospective_path())
    sub = df[df["prospective_track"] == "temporal"]
    if sub.empty:
        pytest.skip("no temporal rows")
    year = pd.to_numeric(sub["publication_year"], errors="coerce")
    assert (year > PRO.ADORE_LAST_REFERENCE_YEAR).all()
    assert not sub["result_id"].astype("string").str.strip().isin(inv.result_id).any()


@needs_prospective
def test_curation_rules_hold_on_every_row():
    df = pd.read_parquet(PRO.prospective_path())
    assert set(df["result_effect"].astype("string").unique()) <= set(PRO.EFFECTS)
    assert set(df["result_endpoint"].astype("string").unique()) <= set(PRO.ENDPOINTS)
    assert set(pd.to_numeric(df["result_obs_duration_mean"]).unique()) <= set(PRO.DURATIONS_H)
    assert set(df["test_media_type"].astype("string").unique()) <= set(PRO.MEDIA_TYPES)
    assert set(df["result_conc1_type"].astype("string").unique()) <= set(PRO.CONC_TYPES)
    assert df["result_conc1_mean_mol_log"].notna().all()
    assert df["chem_rdkit_can_smiles"].notna().all()
    # the primary endpoint's domain flag must match ADORE's chemical classes
    dom = df[df["in_adore_domain"].fillna(False)]
    assert set(dom["structure_class"].astype("string").unique()) <= set(PRO.ADORE_DOMAIN_CLASSES)


@needs_prospective
def test_primary_track_is_powered_as_preregistered():
    df = pd.read_parquet(PRO.prospective_path())
    primary = df[(df["prospective_track"] == "fish_novel") & df["in_adore_domain"].fillna(False)]
    assert len(primary) > 1500
    assert primary["test_cas"].astype("string").str.upper().nunique() >= 300
    assert (primary["tax_group"].astype("string") == "fish").all()


# --------------------------------------------------------------------------- the exclusion guard


@needs_prospective
def test_guard_raises_on_a_prospective_chemical():
    df = pd.read_parquet(PRO.prospective_path(), columns=["test_cas", "chem_inchikey"]).head(5)
    with pytest.raises(AssertionError):
        PRO.assert_prospective_chemicals_absent(df, what="unit test")


@needs_prospective
@needs_adore
def test_guard_passes_on_adore_training_rows():
    rows = pd.read_parquet(A.challenge_path("t-F2F"),
                           columns=["test_cas", "chem_inchikey"]).head(2000)
    PRO.assert_prospective_chemicals_absent(rows, what="ADORE rows")


@needs_prospective
@needs_ecotox
@needs_adore
def test_augmentation_rows_are_clean_and_not_adore_rows():
    rows, info = PRO.augmentation_rows()
    assert info["n_rows"] == len(rows)
    PRO.assert_prospective_chemicals_absent(rows, what="augmentation")
    inv = PRO.adore_inventory()
    assert not rows["result_id"].astype("string").str.strip().isin(inv.result_id).any()


# --------------------------------------------------------------------------- representations


@needs_adore
def test_adore_representations_are_reproducible_from_smiles():
    """MACCS and Morgan bits and Crippen logP must be recomputable for new chemicals."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, MACCSkeys, rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    d = pd.read_parquet(A.challenge_path("t-F2F"),
                        columns=["test_cas", "chem_rdkit_can_smiles", "chem_MACCS_fp",
                                 "chem_Morgan_fp", "chem_rdkit_clogp"]).drop_duplicates("test_cas")
    sample = d.head(50)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for _, r in sample.iterrows():
        mol = Chem.MolFromSmiles(r["chem_rdkit_can_smiles"])
        assert mol is not None
        assert MACCSkeys.GenMACCSKeys(mol).ToBitString() == str(r["chem_MACCS_fp"])
        assert gen.GetFingerprint(mol).ToBitString() == str(r["chem_Morgan_fp"])
        assert abs(Crippen.MolLogP(mol) - float(r["chem_rdkit_clogp"])) < 1e-6


def test_bits_matrix_and_tanimoto():
    s = pd.Series(["1100", "1010", "0000", None])
    B = RA.bits_matrix(s)
    assert B.shape == (4, 4)
    assert B[2].sum() == 0 and B[3].sum() == 0
    sim = RA.tanimoto(B[0], B[:2])
    assert sim[0] == pytest.approx(1.0)
    assert sim[1] == pytest.approx(1 / 3)


def test_readacross_predicts_the_nearest_neighbour_target():
    rows = pd.DataFrame({
        "chem_Morgan_fp": ["1100", "1010", "1100"],
        "chem_MACCS_fp": ["1100", "1010", "1100"],
        "tax_gs": ["a", "a", "a"], "tax_group": ["fish"] * 3,
        "result_obs_duration_mean": [96.0, 96.0, 96.0],
    })
    y = np.array([-5.0, -1.0, -5.0])
    model = RA.ReadAcrossKNN(params={"k": 1, "min_similarity": 0.0, "min_neighbours": 2,
                                     "power": 6.0})
    model.fit(None, y, train_rows=rows)
    q = rows.iloc[[0]].copy()
    assert model.predict(None, rows=q)[0] == pytest.approx(-5.0)


def test_readacross_falls_back_to_the_pool_mean_without_similar_neighbours():
    rows = pd.DataFrame({
        "chem_Morgan_fp": ["1100", "1100"], "chem_MACCS_fp": ["1100", "1100"],
        "tax_gs": ["a", "a"], "tax_group": ["fish", "fish"],
        "result_obs_duration_mean": [96.0, 96.0],
    })
    y = np.array([-4.0, -6.0])
    model = RA.ReadAcrossKNN(params={"k": 1, "min_similarity": 0.99, "min_neighbours": 2})
    model.fit(None, y, train_rows=rows)
    q = pd.DataFrame({"chem_Morgan_fp": ["0011"], "chem_MACCS_fp": ["0011"], "tax_gs": ["a"],
                      "tax_group": ["fish"], "result_obs_duration_mean": [96.0]})
    assert model.predict(None, rows=q)[0] == pytest.approx(-5.0)
    assert model.widening_["fallback_mean"] == 1


# --------------------------------------------------------------------------- cycle-2 models


def test_gpboost_handles_a_taxon_it_never_saw():
    gpb = pytest.importorskip("gpboost")
    from sentinel.models.chemtox.gpboost_model import GPBoostTox

    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 4))
    rows = pd.DataFrame({"tax_gs": rng.choice(["sp1", "sp2", "sp3"], n),
                         "tax_genus": "g", "tax_group": "fish"})
    y = X[:, 0] + np.where(rows["tax_gs"] == "sp1", 1.0, -1.0) + 0.1 * rng.normal(size=n)
    model = GPBoostTox(params={"num_boost_round": 20, "learning_rate": 0.1,
                               "group_columns": ("tax_gs", "tax_genus", "tax_group")})
    model.fit(X, y, train_rows=rows)
    unseen = pd.DataFrame({"tax_gs": ["sp_new"] * 5, "tax_genus": ["g"] * 5,
                           "tax_group": ["fish"] * 5})
    p = model.predict(X[:5], rows=unseen)
    assert p.shape == (5,)
    assert np.isfinite(p).all()


def test_chemeleon_block_is_deterministic_and_standardised():
    pytest.importorskip("chemprop")
    from sentinel.models.chemtox.chemeleon_fp import CheMeleonBlock, compute

    smiles = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "CCCCCCCC"]
    tab = compute(smiles)
    assert all(tab[s].shape == (2048,) for s in smiles)
    again = compute(smiles)
    for s in smiles:
        assert np.array_equal(tab[s], again[s])
    df = pd.DataFrame({"chem_rdkit_can_smiles": smiles})
    block = CheMeleonBlock(n_components=3).fit(df)
    Z = block.transform(df)
    assert Z.shape == (4, 3)
    assert np.isfinite(Z).all()


def test_cycle_search_spaces_are_deduplicated_and_reproducible():
    from sentinel.models.chemtox import cycle as CYC

    for family in CYC.SPACES:
        a = CYC.sample(family, 16, seed=7)
        b = CYC.sample(family, 16, seed=7)
        assert a == b
        keys = {repr(sorted(c["features"].items())) +
                repr(sorted((k, list(v) if isinstance(v, tuple) else v)
                            for k, v in c["params"].items())) for c in a}
        assert len(keys) == 16


@needs_prospective
@needs_adore
def test_prospective_rows_featurise_with_the_adore_pipeline():
    """A builder fitted on ADORE training rows must transform the prospective rows finitely."""
    from sentinel.models.chemtox.data import ChemToxDataset, FeatureSpec

    specs = [FeatureSpec(fingerprint="MACCS"),
             FeatureSpec(fingerprint="Morgan", chemical="mw_logp", rdkit_descriptors=True,
                         rdkit_descriptors_full=True)]
    ds = ChemToxDataset.load("t-F2F", A.PRIMARY_SPLIT, specs=specs,
                             extra_columns=("chem_rdkit_can_smiles",))
    pro = PRO.load(("fish_novel", "crust_novel"))
    tr = np.flatnonzero(ds.train_mask())
    for spec in specs:
        b = ds.builder(spec)
        X = b.fit_transform(ds.rows(tr))
        Xp = b.transform(pro)
        assert Xp.shape[1] == X.shape[1]
        assert np.isfinite(Xp).all()


# --------------------------------------------------------------------------- structural novelty


def test_parent_key_ignores_salt_charge_and_stereochemistry():
    assert PRO.parent_key("Nc1ccccc1") == PRO.parent_key("Cl.Nc1ccccc1")
    assert PRO.parent_key("C[C@H](N)C(=O)O") == PRO.parent_key("C[C@@H](N)C(=O)O")
    assert PRO.parent_key("CC(N)C(=O)O") == PRO.parent_key("C[C@H](N)C(=O)O")
    assert PRO.parent_key("[Na+].CC([S-])=S") == PRO.parent_key("[Zn+2].CC([S-])=S.CC([S-])=S")
    assert PRO.parent_key("Nc1ccccc1") != PRO.parent_key("Oc1ccccc1")
    assert PRO.parent_key("not a molecule") is None


@needs_prospective
def test_parent_novel_flag_and_similarity_columns_exist_and_are_consistent():
    df = pd.read_parquet(PRO.prospective_path())
    assert {"parent_novel", "parent_key", "nn_tanimoto_adore_train"} <= set(df.columns)
    # a flagged-novel row must not carry a parent key that ADORE contains
    inv = PRO.adore_inventory()
    novel = df[df["parent_novel"].fillna(False)]
    assert not novel["parent_key"].astype("string").isin(inv.parent).any()
    sim = pd.to_numeric(df["nn_tanimoto_adore_train"], errors="coerce")
    assert sim.between(0, 1).fillna(True).all()
    # the strict primary subset is the one the pre-registration quotes
    primary = df[(df["prospective_track"] == "fish_novel") & df["in_adore_domain"].fillna(False)]
    strict = primary[primary["parent_novel"].fillna(False)]
    assert len(strict) < len(primary)
    assert strict["test_cas"].astype("string").str.upper().nunique() >= 200


@needs_prospective
@needs_ecotox
@needs_adore
def test_augmentation_excludes_prospective_parents():
    rows, info = PRO.augmentation_rows()
    assert info["n_dropped_as_prospective_parents"] >= 0
    PRO.assert_prospective_chemicals_absent(rows, parent_col="parent_key", what="augmentation")


# --------------------------------------------------------------------------- the sealed scorer


def _scorer():
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts/eval/chemtox_score_prospective.py"
    spec = importlib.util.spec_from_file_location("chemtox_score_prospective", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@needs_prospective
def test_scorer_refuses_without_approval_and_on_a_wrong_hash():
    m = _scorer()
    base = ["--expect-data-hash", "0" * 64, "--expect-config-hash", "deadbeefdeadbeef"]
    assert m.main(base) == 2                                   # no --i-have-approval
    assert m.main(["--i-have-approval", *base]) == 3           # wrong data hash


@needs_prospective
def test_scorer_refuses_a_wrong_configuration_hash(tmp_path):
    import json

    from sentinel.core.registry import file_sha256

    m = _scorer()
    sel = tmp_path / "selection.json"
    frozen = {"xgboost": {"features": {}, "params": {}, "seeds": [0]}}
    sel.write_text(json.dumps({"frozen": frozen, "families": {}}))
    rc = m.main(["--i-have-approval", "--expect-data-hash", file_sha256(PRO.prospective_path()),
                 "--expect-config-hash", "0123456789abcdef", "--selection", str(sel)])
    assert rc == 5


@needs_prospective
def test_scorer_track_masks_match_the_preregistration():
    m = _scorer()
    df = pd.read_parquet(PRO.prospective_path())
    primary = m.track_mask(df, "fish_novel")
    assert primary.sum() == 1912
    assert m.track_mask(df, "fish_novel_all").sum() == 2313
    assert m.track_mask(df, "crust_novel").sum() == 230
    assert m.track_mask(df, "crust_novel_all").sum() == 295
    assert m.PRIMARY_TRACK == "fish_novel"
    assert m.STRICT_TRACK == "fish_novel_strict"
    assert m.track_mask(df, "fish_novel_strict").sum() == 1362
    assert m.track_mask(df, "crust_novel_strict").sum() == 154
    assert m.track_role("fish_novel") == "PRIMARY"
    assert "CO-PRIMARY" in m.track_role("fish_novel_strict")
    # comparator/candidate comes from the frozen role, with the family list only as a fallback
    assert m.BASELINE_FAMILIES == ("xgboost", "rf", "readacross")
    assert m.HOLM_REFERENCES == ("published_reference", "best_baseline")
    assert m.model_family("xgboost_published_mwlogp") == "xgboost"


def test_scorer_statistics_helpers():
    m = _scorer()
    rng = np.random.default_rng(0)
    y = rng.normal(size=400)
    good = y + 0.1 * rng.normal(size=400)
    bad = y + 1.0 * rng.normal(size=400)
    chem = np.repeat(np.arange(40), 10).astype(str)
    t = m.paired_delta_ci(y, good, bad, chem, n_boot=200)
    assert t["diff"] < 0 and t["ci_high"] < 0            # the better model has the lower RMSE
    assert m.rmse(y, good) < m.rmse(y, bad)


# --------------------------------------------------------------------------- deployment masking


@needs_adore
def test_deployment_masking_blanks_only_the_unavailable_columns():
    from sentinel.models.chemtox import cycle as CYC

    rows = pd.DataFrame({"chem_ws": [1.0, 2.0], "chem_mp": [3.0, 4.0], "chem_mw": [5.0, 6.0]})
    out = CYC.mask_deployment_missing(rows)
    assert out["chem_ws"].isna().all() and out["chem_mp"].isna().all()
    assert (out["chem_mw"] == rows["chem_mw"]).all()
    assert not rows["chem_ws"].isna().any()               # the input is not mutated


def test_gpboost_rejects_or_honours_sample_weight():
    """W15: weights must either be honoured or raise — never be silently ignored."""
    pytest.importorskip("gpboost")
    from sentinel.models.chemtox.gpboost_model import GPBoostTox

    rng = np.random.default_rng(1)
    n = 300
    X = rng.normal(size=(n, 3))
    rows = pd.DataFrame({"tax_gs": rng.choice(["a", "b"], n), "tax_genus": "g",
                         "tax_group": "fish"})
    y = X[:, 0] + 0.1 * rng.normal(size=n)
    w = np.where(np.arange(n) < n // 2, 1.0, 0.01)
    m1 = GPBoostTox(params={"num_boost_round": 30, "learning_rate": 0.1,
                            "group_columns": ("tax_gs",)}).fit(X, y, train_rows=rows)
    try:
        m2 = GPBoostTox(params={"num_boost_round": 30, "learning_rate": 0.1,
                                "group_columns": ("tax_gs",)}).fit(X, y, train_rows=rows,
                                                                   sample_weight=w)
    except Exception:            # raising is an acceptable outcome; silence is not
        return
    p1 = m1.predict(X[:50], rows=rows.iloc[:50])
    p2 = m2.predict(X[:50], rows=rows.iloc[:50])
    assert np.isfinite(p2).all()
    assert not np.allclose(p1, p2), "sample_weight appears to be ignored by GPBoost"


def test_readacross_refuses_sample_weight():
    rows = pd.DataFrame({"chem_Morgan_fp": ["1100", "1010"], "chem_MACCS_fp": ["1100", "1010"],
                         "tax_gs": ["a", "a"], "tax_group": ["fish", "fish"],
                         "result_obs_duration_mean": [96.0, 96.0]})
    model = RA.ReadAcrossKNN(params={"k": 1})
    with pytest.raises(ValueError):
        model.fit(None, np.array([-5.0, -1.0]), train_rows=rows,
                  sample_weight=np.array([1.0, 0.5]))


# --------------------------------------------------------------------------- end to end (mocked)


@needs_prospective
def test_scorer_runs_end_to_end_and_enforces_the_single_pass(tmp_path, monkeypatch):
    """A full mocked pass: no model is fitted and no real target is read.

    `fit_predict` is replaced by deterministic random predictions, the prospective file is copied
    without its target columns and given random ones, and the registry, run directory and training
    log are redirected into ``tmp_path``. This is the test whose absence let two ADORE test passes
    crash after printing scores (chemtox_test_plan.md §6b) and the round-1 review finding C1.
    """
    import hashlib
    import importlib.util
    import json
    import sys

    import pyarrow.parquet as pq

    from sentinel.core import paths, registry, runlog

    real = PRO.prospective_path()
    keep = [c for c in pq.ParquetFile(real).schema_arrow.names
            if c not in ("result_conc1_mean_mol_log", "result_conc1_mean_log",
                         "result_conc1_mean")]
    df = pd.read_parquet(real, columns=keep)
    df["result_conc1_mean_mol_log"] = np.random.default_rng(1).normal(-4.5, 1.7, len(df))
    fake = tmp_path / "fake_prospective.parquet"
    df.to_parquet(fake)

    monkeypatch.setattr(PRO, "prospective_path", lambda root=None: fake)
    monkeypatch.setattr(paths, "REGISTRY_PATH", tmp_path / "registry.jsonl")
    monkeypatch.setattr(runlog, "TRAINING_LOG_JSONL", tmp_path / "runs.jsonl", raising=False)
    monkeypatch.setattr(runlog, "TRAINING_LOG_MD", tmp_path / "TRAINING_LOG.md", raising=False)
    monkeypatch.setattr(paths, "run_dir",
                        lambda ws, name: paths.ensure(tmp_path / "runs" / ws / name))

    path = Path(__file__).resolve().parents[1] / "scripts/eval/chemtox_score_prospective.py"
    spec = importlib.util.spec_from_file_location("sc_e2e", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["sc_e2e"] = m
    spec.loader.exec_module(m)

    def fake_fit_predict(family, fspec, params, train_rows, y_train, pro, *, seed, device,
                         chemeleon_device, frozen=None, sample_weight=None):
        m.CYC.make_model(m.model_family(family, frozen), seed)   # name resolution must work
        h = int(hashlib.md5(f"{family}{seed}".encode()).hexdigest()[:8], 16)
        return np.random.default_rng(h).normal(-4.5, 1.5, len(pro))

    monkeypatch.setattr(m, "fit_predict", fake_fit_predict)
    maccs = {"fingerprint": "MACCS"}
    def entry(family, role, **kw):
        return {"features": maccs, "params": {}, "extra": "none", "family": family,
                "role": role, "seeds": [0, 1, 2], "cv_rmse": 1.0, **kw}
    # a stand-in out-of-fold matrix for the stack, as a real freeze would record: the arrays the
    # CV runner writes span the whole challenge frame (26,114 t-F2F rows), with NaN outside the
    # training folds, so a wrong length here is exactly what this test should catch
    ds_rows = 26114
    oof_dir = tmp_path / "oof"
    oof_dir.mkdir()
    rng = np.random.default_rng(7)
    sources = {}
    for c in ("lightgbm_aug", "gpboost"):
        f = oof_dir / f"oof_{c}.npy"
        arr = rng.normal(-4.5, 1.5, ds_rows)
        arr[-4133:] = np.nan              # the held-out partition carries no out-of-fold value
        np.save(f, arr)
        sources[c] = str(f)
    frozen = {
        "xgboost_published": entry("xgboost", "comparator", cv_rmse=1.46),
        "xgboost_published_mwlogp": entry("xgboost", "comparator", cv_rmse=0.9945),
        "rf_published": entry("rf", "comparator", cv_rmse=1.2561),
        "rf_published_mwlogp": entry("rf", "comparator", cv_rmse=0.9909),
        "xgboost": entry("xgboost", "comparator"),
        "rf": entry("rf", "comparator"),
        "readacross": entry("readacross", "comparator"),
        "lightgbm_aug": entry("lightgbm", "candidate", cv_rmse=0.95),
        "gpboost": entry("gpboost", "candidate", cv_rmse=0.96),
        "ensemble_mean": {"ensemble": True, "role": "candidate",
                          "components": ["lightgbm_aug", "gpboost"], "seeds": [0]},
        "ensemble_stack_nnls": {"ensemble": True, "role": "candidate",
                                "components": ["lightgbm_aug", "gpboost"],
                                "oof_sources": sources, "seeds": [0]},
        # the meta keys a real freeze writes into the hashed block
        "__published_reference__": "rf_published_mwlogp",
        "__published_reference_cv__": {"xgboost_published": 1.46,
                                       "xgboost_published_mwlogp": 0.9945,
                                       "rf_published": 1.2561, "rf_published_mwlogp": 0.9909},
        "__allow_unmasked__": False,
    }
    sel = tmp_path / "selection.json"
    sel.write_text(json.dumps({"frozen": frozen, "families": {}}))
    args = ["--i-have-approval", "--expect-data-hash", registry.file_sha256(fake),
            "--expect-config-hash", m.frozen_hash(frozen), "--selection", str(sel),
            "--n-boot", "20", "--threads", "2", "--tracks", "fish_novel", "fish_novel_strict"]

    assert m.main(args) == 0
    rows = registry.read_registry(paths.REGISTRY_PATH)
    assert rows, "the pass logged nothing"
    models = {r["model"] for r in rows}
    # comparators, candidates, the trivial baselines and the ensemble all land, on both tracks
    assert {"xgboost_published", "rf_published_mwlogp", "lightgbm_aug", "gpboost",
            "ensemble_mean", "ensemble_stack_nnls", "species_duration_mean"} <= models
    assert "best_baseline" not in models and "published_reference" not in models
    splits = {r["split"] for r in rows}
    assert splits == {"prospective/fish_novel/test", "prospective/fish_novel_strict/test"}
    # the Holm family is the candidates only, and the published reference was chosen by masked CV
    report = json.loads((tmp_path / "runs/chemtox/prospective_test/prospective_results.json")
                        .read_text())
    primary = report["tracks"]["fish_novel"]
    assert set(primary["holm_family_vs_published_reference"]) <= {
        "lightgbm_aug", "gpboost", "ensemble_mean", "ensemble_stack_nnls"}
    assert primary["published_reference"]["model"] == "rf_published_mwlogp"
    # the stack's weights came from the frozen out-of-fold sources, and the ensemble has per-seed
    # predictions so the all-seeds rule is computable for it
    stack = primary["models"]["ensemble_stack_nnls"]
    assert len(stack["seed_rmse"]) == 3
    assert stack["vs_published_reference"]["all_seeds_better"] in (True, False)
    assert primary["best_baseline"]["model"] in {"xgboost_published", "xgboost_published_mwlogp",
                                                 "rf_published", "xgboost", "rf", "readacross",
                                                 "global_mean", "species_mean",
                                                 "species_duration_mean"}
    for name, e in primary["models"].items():
        assert e["rmse_ci"]["ci_low"] <= e["metrics"]["rmse"] <= e["rmse_ci"]["ci_high"], name

    # a second identical pass must be refused (the test set is scored once)
    assert m.main(args) == 6
    # ... and an ensemble whose component was not scored is refused **before anything is fitted**
    # (exit code 9), never scored as a smaller ensemble
    frozen["ensemble_mean"]["components"] = ["lightgbm_aug", "does_not_exist"]
    sel.write_text(json.dumps({"frozen": frozen, "families": {}}))
    args2 = list(args)
    args2[args2.index("--expect-config-hash") + 1] = m.frozen_hash(frozen)
    assert m.main(args2 + ["--allow-rescore"]) == 9


def test_selection_keeps_each_training_data_variant_separate():
    """Tuning parity: each (family, training-data variant) is its own 32-configuration search,
    and explicit masked re-runs (``_dm_cfg-<tag>``) count as masked."""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "eval" / "chemtox_cycle_select.py"
    spec = importlib.util.spec_from_file_location("chemtox_cycle_select_t", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    base = "t-F2F_split_occurrence"
    assert m.variant_extra(f"lightgbm_{base}_extra-ecotox_dm") == "ecotox"
    assert m.variant_extra(f"gpboost_{base}_extra-adore_taxa_all_dm") == "adore_taxa_all"
    assert m.variant_extra(f"chemprop_{base}_extra-none_dm_cfg-v1_chemprop_mwlogp") == "none"
    assert m.is_masked_dir(f"chemprop_{base}_extra-none_dm_cfg-v1frozen")
    assert m.is_masked_dir(f"xgboost_{base}_extra-none_dm")
    assert not m.is_masked_dir(f"xgboost_{base}_extra-none")
    assert not m.is_masked_dir(f"xgboost_{base}_extra-none_nodm")
    assert m.EXTRA_SUFFIX["ecotox"] == "_aug" and m.EXTRA_SUFFIX["adore_taxa_all"] == "_multitaxa"


def test_unseen_taxa_blocks_hold_out_whole_classes_and_match_the_protocol():
    """The frozen unseen-taxa blocks: class-level hold-out, no row overlap, sizes as pre-registered."""
    import json

    import pandas as pd

    from sentinel.core import paths
    from sentinel.models.chemtox import unseen_taxa as U

    d = paths.DATA_ROOT / "processed" / "chemtox_unseen_taxa"
    if not (d / "test_blocks.parquet").exists():
        pytest.skip("unseen-taxa blocks not built")
    test = pd.read_parquet(d / "test_blocks.parquet")
    pool = pd.read_parquet(d / "training_pool.parquet")
    rep = json.loads((d / "report.json").read_text())

    # the training pool contains no row of any held-out class, and no test row
    U.assert_no_test_classes(pool, what="pool")
    U.assert_no_test_rows(pool, test, what="pool")
    assert set(pool["taxon_group"].unique()) <= set(U.TRAINING_GROUPS)
    assert set(test["class"].astype(str)) <= set(U.HELD_OUT_CLASSES)
    # every block is held out at class level: no class is split between pool and test
    assert not (set(pool["class"].astype(str)) & set(test["class"].astype(str)))
    # the strata the protocol reports exist and are consistent
    for col in ("in_adore_domain", "chem_novel", "parent_novel"):
        assert col in test.columns
    assert (test["parent_novel"] & ~test["chem_novel"]).sum() == 0
    # sizes match report.json (which the protocol table was written from)
    for b, s in rep["blocks"].items():
        sub = test[test["block"] == b]
        assert len(sub) == s["rows_all"]
        assert int(sub["in_adore_domain"].sum()) == s["rows_primary"]
