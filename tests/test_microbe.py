# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Unit tests for SENTINEL-Microbe (M3). Tiny SYNTHETIC arrays only (declared; RESEARCH_PLAN R9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.models.microbe import baselines as B
from sentinel.models.microbe import dna
from sentinel.models.microbe import evaluate as E
from sentinel.models.microbe import features as F
from sentinel.models.microbe import seqset as SS
from sentinel.models.microbe import splits as SP
from sentinel.models.microbe.design import TARGET_NAMES, Design, add_targets, holdout_mask

RNG = np.random.default_rng(0)


def _synthetic_design(n_sites=8, per_site=12, n_seq=40):
    """SYNTHETIC: random communities and targets for code-path tests only."""
    rows, toks = [], []
    seqs = ["".join(RNG.choice(list("ACGT"), 60)) for _ in range(n_seq)]
    sid = [dna.seq_key(s) for s in seqs]
    genera = [f"g{i % 10}" for i in range(n_seq)]
    for si in range(n_sites):
        for j in range(per_site):
            smp = f"S{si}_{j}"
            rows.append({"sample_id": smp, "site_id": f"SITE{si}", "domain_id": f"D{si // 2:02d}",
                         "aquatic_type": ["stream", "lake", "river"][si % 3], "latitude": 30 + si,
                         "longitude": -100 + si, "elevation": np.nan if si == 0 else 100.0 * si,
                         "collect_date": pd.Timestamp("2019-01-15") + pd.Timedelta(days=30 * j),
                         "primer_set": "V4V5" if j % 2 else "V3V4", "n_reads": 5000, "n_seqs": 20,
                         "raw_tn_mgl": RNG.lognormal(-1, 0.5), "raw_tp_mgl": RNG.lognormal(-3, 0.5),
                         "raw_doc_mgl": RNG.lognormal(1, 0.3), "raw_no3_mgl": RNG.lognormal(-2, 1),
                         "raw_sc_uscm": RNG.lognormal(5, 0.5), "raw_turb_fnu": RNG.lognormal(1, 1),
                         "raw_do_mgl": RNG.normal(9, 1), "raw_chla_ugl": np.nan if j == 0 else RNG.lognormal(0, 1)})
            pick = RNG.choice(n_seq, 20, replace=False)
            for k in pick:
                toks.append({"sample_id": smp, "seq_id": sid[k], "count": int(RNG.integers(1, 500)),
                             "genus": genera[k]})
    s = add_targets(pd.DataFrame(rows)).set_index("sample_id", drop=False)
    t = pd.DataFrame(toks)
    q = pd.DataFrame({"seq_id": sid, "sequence": seqs, "length": [len(x) for x in seqs]})
    return Design(s, t, q, {"kind": "date_from", "date": "2020-01-01"}, False), seqs, sid


def test_kmer_profile_sums_to_one_and_skips_n():
    p = dna.kmer_profile(["ACGTACGT", "ACGNNNNT", "AC"], k=3)
    assert p.shape == (3, 64)
    assert np.isclose(p[0].sum(), 1.0)
    assert p[1].sum() == 1.0 or p[1].sum() == 0.0
    assert p[2].sum() == 0.0
    names = dna.kmer_names(3)
    assert p[0][names.index("ACG")] == pytest.approx(2 / 6)


def test_alibi_slopes_power_of_two():
    s = dna._alibi_slopes(8)
    assert s[0] == pytest.approx(0.5) and len(s) == 8
    assert len(dna._alibi_slopes(12)) == 12


def test_dnaberts_reimplementation_runs_if_checkpoint_present():
    try:
        enc = dna.DNABERTS(device="cpu", fp16=False)
    except Exception as exc:  # checkpoint absent offline
        pytest.skip(f"DNABERT-S checkpoint unavailable: {exc}")
    e = enc.embed(["ACGTACGTTTGACCA" * 4, "TTGACCAACGT" * 5])
    assert e.shape == (2, 768) and np.isfinite(e).all()


def test_folds_hold_out_whole_groups():
    d, _, _ = _synthetic_design()
    for scheme in ("lodo", "loso", "primer_v4v5_to_v3v4"):
        folds = SP.make_folds(d.samples, scheme)
        assert folds
        for f in folds:
            SP.check_fold(d.samples, f)
    lodo = SP.make_folds(d.samples, "lodo")
    assert sorted(np.concatenate([f.test for f in lodo])) == sorted(d.samples.index)
    for f in SP.make_folds(d.samples, "primer_v4v5_to_v3v4"):
        assert (d.samples.loc[f.train, "primer_set"] == "V4V5").all()
        assert (d.samples.loc[f.test, "primer_set"] == "V3V4").all()
        assert d.samples.loc[f.test, "domain_id"].nunique() == 1


def test_inner_validation_is_site_disjoint_and_inside_training():
    d, _, _ = _synthetic_design()
    f = SP.make_folds(d.samples, "lodo")[0]
    fit, val = SP.inner_validation(d.samples, f.train)
    assert set(fit) | set(val) == set(f.train)
    assert not set(d.samples.loc[fit, "site_id"]) & set(d.samples.loc[val, "site_id"])


def test_frozen_split_hash_is_deterministic(tmp_path):
    d, _, _ = _synthetic_design()
    m = holdout_mask(d.samples, {"kind": "date_from", "date": "2019-09-01"})
    rec = SP.split_record(d.samples[~m], d.samples[m], {"kind": "date_from", "date": "2019-09-01"},
                          schemes=("lodo",))
    h1 = SP.write_frozen_split(rec, tmp_path / "a.json")
    h2 = SP.write_frozen_split(rec, tmp_path / "b.json")
    assert h1 == h2 and rec["n_holdout"] > 0


def test_genus_clr_fit_on_training_rows_only():
    d, _, _ = _synthetic_design()
    tr = d.samples.index[:48]
    te = d.samples.index[48:]
    g1 = F.GenusCLR(min_prev=0.05).fit(d.tokens, tr)
    tok2 = d.tokens.copy()
    tok2.loc[tok2["sample_id"].isin(te), "genus"] = "LEAK"   # held-out rows cannot change the filter
    g2 = F.GenusCLR(min_prev=0.05).fit(tok2, tr)
    assert g1.keep_ == g2.keep_
    X = g1.transform(d.tokens, te)
    assert np.allclose(X.sum(1), 0, atol=1e-4)  # CLR rows centre to 0


def test_context_fill_comes_from_training_rows():
    d, _, _ = _synthetic_design()
    tr = d.samples.index[d.samples["site_id"] != "SITE1"]
    _, fill = F.context_features(d.samples, tr)
    assert fill["elevation"] == pytest.approx(d.samples.loc[tr, "elevation"].median())


def test_targets_never_enter_features():
    """Perturbing every target (and the raw target columns) leaves every feature matrix unchanged."""
    d, seqs, sid = _synthetic_design()
    tr, te = d.samples.index[:60], d.samples.index[60:]
    d2 = Design(d.samples.copy(), d.tokens, d.sequences, d.holdout_rule, False)
    for t in list(TARGET_NAMES) + [c for c in d.samples.columns if c.startswith("raw_")]:
        d2.samples[t] = RNG.normal(size=len(d2.samples))
    for dd in (d, d2):
        dd.X = (F.GenusCLR().fit(dd.tokens, tr).transform(dd.tokens, te),
                F.sample_descriptors(dd.samples, te), F.context_features(dd.samples, te)[0])
    for a, b in zip(d.X, d2.X):
        assert np.array_equal(a, b)
    assert d.X[1].shape[1] == 5
    table = np.zeros((len(sid) + 1, 64), np.float32)
    table[1:] = dna.kmer_profile(seqs, k=3)
    sr = {s: i + 1 for i, s in enumerate(sid)}
    b1 = SS.encode_sets(d.tokens, te, sr, F.sample_descriptors(d.samples, te), 8)
    b2 = SS.encode_sets(d2.tokens, te, sr, F.sample_descriptors(d2.samples, te), 8)
    assert np.array_equal(b1.ab, b2.ab) and np.array_equal(b1.seq, b2.seq) and np.array_equal(b1.glob, b2.glob)


def test_within_fold_permutation_touches_training_rows_only():
    from sentinel.models.microbe import cv as CV

    d, _, _ = _synthetic_design()
    f = SP.make_folds(d.samples, "lodo")[0]
    Y = CV.permuted_training_targets(d.samples, f.train, TARGET_NAMES, seed=1)
    orig = d.samples[list(TARGET_NAMES)]
    assert Y.loc[f.test].equals(orig.loc[f.test])                    # held-out targets untouched
    assert not Y.loc[f.train].equals(orig.loc[f.train])
    a = np.sort(Y.loc[f.train, "do"].to_numpy())
    b = np.sort(orig.loc[f.train, "do"].to_numpy())
    assert np.allclose(a, b, equal_nan=True)                          # a permutation among training rows
    p = CV.oof_baseline("global_mean", d, [f], targets=TARGET_NAMES, permute_seed=1)
    assert np.allclose(p["do"].to_numpy(), np.nanmean(orig.loc[f.train, "do"]))


def test_id_vocabulary_is_built_from_training_fit_rows(monkeypatch):
    from sentinel.models.microbe import cv as CV

    d, _, _ = _synthetic_design()
    folds = SP.make_folds(d.samples, "lodo")[:1]
    seen = {}

    def fake_fit(cfg, tab, b_fit, Ytr, b_val, Yva, **kw):
        seen["n_vocab"] = kw["n_vocab"]
        seen["fit_ids"] = set(b_fit.ids)
        raise RuntimeError("stop")

    monkeypatch.setattr(SS, "fit_seqset", fake_fit)
    with pytest.raises(RuntimeError, match="stop"):
        CV.oof_seqset({"seq_mode": "id", "min_seq_count": 1, "max_tokens": 8}, d, folds, targets=TARGET_NAMES, seeds=(0,))
    fit_tok = d.tokens[d.tokens["sample_id"].isin(seen["fit_ids"])]
    assert seen["n_vocab"] == fit_tok["seq_id"].nunique() + 1
    assert not seen["fit_ids"] & set(folds[0].test)


def test_release_folds_are_domain_disjoint():
    d, _, _ = _synthetic_design()
    s = d.samples.copy()
    s["source_product"] = np.where(s["primer_set"] == "V4V5", "DP1.20141.002", "DP1.20141.001")
    folds = SP.release_out(s, min_test=1)
    assert folds
    for f in folds:
        assert (s.loc[f.train, "source_product"] == "DP1.20141.002").all()
        assert (s.loc[f.test, "source_product"] == "DP1.20141.001").all()
        assert not set(s.loc[f.train, "domain_id"]) & set(s.loc[f.test, "domain_id"])


def test_enlarged_lodo_tests_only_002_rows_and_trains_on_both_releases():
    d, _, _ = _synthetic_design()
    s = d.samples.copy()
    s["source_product"] = np.where(s["primer_set"] == "V4V5", "DP1.20141.002", "DP1.20141.001")
    folds = SP.make_folds(s, "lodo_x001")
    assert folds
    primary = {f.name.split("|")[0]: set(f.test) for f in SP.lodo(s[s["source_product"] == "DP1.20141.002"])}
    for f in folds:
        assert (s.loc[f.test, "source_product"] == "DP1.20141.002").all()
        assert set(f.test) == primary[f.name.split("|")[0]]          # same test rows as primary LODO
        assert not set(s.loc[f.train, "domain_id"]) & set(s.loc[f.test, "domain_id"])
    assert any((s.loc[f.train, "source_product"] == "DP1.20141.001").any() for f in folds)


def test_frozen_split_guard_rejects_changed_rows(tmp_path, monkeypatch):
    import json

    from sentinel.core import paths
    from sentinel.models.microbe import design as DS

    d, _, _ = _synthetic_design()
    root = tmp_path / "processed" / "microbe_wq"
    root.mkdir(parents=True)
    d.samples.drop(columns=list(TARGET_NAMES)).reset_index(drop=True).to_parquet(root / "samples.parquet")
    d.tokens.to_parquet(root / "tokens.parquet")
    d.sequences.to_parquet(root / "sequences.parquet")
    repo = tmp_path / "repo"
    (repo / "docs" / "preregistration").mkdir(parents=True)
    (repo / "docs" / "preregistration" / "microbe_split.json").write_text(json.dumps({"development_ids": ["x"]}))
    monkeypatch.setattr(paths, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(paths, "REPO_ROOT", repo)
    rule = {"kind": "date_from", "date": "2030-01-01"}
    with pytest.raises(RuntimeError, match="frozen split"):
        DS.load_design(holdout_rule=rule)
    assert len(DS.load_design(holdout_rule=rule, verify_frozen=False).samples) == len(d.samples)


@pytest.mark.parametrize("name", ["global_mean", "context_ridge", "context_rf", "site_mean_doy", "ridge_clr",
                                  "rf_clr", "xgb_clr", "rf_pa", "rf_clr_context"])
def test_baselines_run(name):
    d, _, _ = _synthetic_design()
    f = SP.make_folds(d.samples, "lodo")[0]
    p = B.fit_predict(name, d, f.train, f.test, targets=TARGET_NAMES, seed=0, n_jobs=1,
                      params={"rf": {"n_estimators": 20, "min_samples_leaf": 3, "max_features": 0.3},
                              "xgb": {"n_estimators": 20, "learning_rate": 0.1, "max_depth": 2}})
    assert list(p.index) == list(f.test)
    if name == "site_mean_doy":
        assert p.isna().all().all()   # held-out sites have no training mean
    else:
        assert np.isfinite(p.to_numpy()).all()


def test_seqset_trains_and_predicts_both_modes():
    torch = pytest.importorskip("torch")
    d, seqs, sid = _synthetic_design()
    table = np.zeros((len(sid) + 1, 64), np.float32)
    table[1:] = dna.kmer_profile(seqs, k=3)
    seq_row = {s: i + 1 for i, s in enumerate(sid)}
    f = SP.make_folds(d.samples, "lodo")[0]
    fit, val = SP.inner_validation(d.samples, f.train)
    Y = d.samples[list(TARGET_NAMES)].to_numpy(np.float32)
    row = pd.Series(np.arange(len(d.samples)), index=d.samples.index)
    cfg = {"d_model": 16, "n_layers": 1, "n_heads": 2, "dropout": 0.0, "token_dropout": 0.1, "lr": 1e-3,
           "weight_decay": 0.0, "batch_size": 16, "max_epochs": 3, "patience": 5}
    enc = lambda ids, sr: SS.encode_sets(d.tokens, ids, sr, F.sample_descriptors(d.samples, ids), 8)
    m = SS.fit_seqset(cfg, table, enc(fit, seq_row), Y[row[fit]], enc(val, seq_row), Y[row[val]], seed=0)
    p = m.predict(enc(f.test, seq_row))
    assert p.shape == (len(f.test), len(TARGET_NAMES)) and np.isfinite(p).all()
    e = m.embed(enc(f.test, seq_row))
    assert e.shape == (len(f.test), 32)
    cfg_id = {**cfg, "seq_mode": "id"}
    sr = {s: i + 1 for i, s in enumerate(sid[:10])}       # others -> UNK
    m2 = SS.fit_seqset(cfg_id, None, enc(fit, sr), Y[row[fit]], enc(val, sr), Y[row[val]], seed=0, n_vocab=11)
    assert np.isfinite(m2.predict(enc(f.test, sr))).all()
    _ = torch


def test_encode_sets_truncates_by_abundance():
    toks = pd.DataFrame({"sample_id": ["a"] * 4, "seq_id": ["w", "x", "y", "z"], "count": [1, 50, 10, 5]})
    b = SS.encode_sets(toks, ["a"], {"w": 1, "x": 2, "y": 3, "z": 4}, None, max_tokens=2)
    assert b.seq.tolist() == [[2, 3]]
    assert b.ab[0, 0, 1] == pytest.approx(50 / 66)   # relative abundance uses the full sample total


def test_metrics_anomaly_and_skill():
    y = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    reg = np.array(["a"] * 3 + ["b"] * 3)
    p_const = np.array([2.0] * 3 + [11.0] * 3)          # perfect domain means, no within-domain skill
    assert E.anomaly_r2(y, p_const, reg) == pytest.approx(0.0)
    assert E.mse_skill(y, y, p_const) == pytest.approx(1.0)
    pr = E.per_region_r2(y, y, reg, min_n=3)
    assert pr["a"] == pytest.approx(1.0)


def test_paired_bootstrap_uses_common_rows():
    y = RNG.normal(size=60)
    site = np.repeat(np.arange(12), 5)
    reg = np.repeat(np.arange(4), 15)
    a = y + RNG.normal(0, 0.1, 60)
    b = y + RNG.normal(0, 1.0, 60)
    b[:5] = np.nan
    res = E.paired_vs(y, a, b, reg, site, n_boot=200)
    assert res["n"] == 55 and res["diff"] > 0 and res["ci_low"] > 0


def test_site_identity_probe_detects_site_signal():
    site = np.repeat(np.arange(6), 10)
    X = np.eye(6)[site] * 3 + RNG.normal(0, 0.3, (60, 6))
    r = E.site_identity_probe(X, site, n_folds=5)
    assert r["accuracy"] > 0.9 and r["chance_uniform"] == pytest.approx(1 / 6)
    Xn = RNG.normal(size=(60, 6))
    assert E.site_identity_probe(Xn, site, n_folds=5)["accuracy"] < 0.6


def test_holdout_guard(tmp_path):
    d, _, _ = _synthetic_design()
    s = d.samples.drop(columns=list(TARGET_NAMES)).reset_index(drop=True)
    s.to_parquet(tmp_path / "samples.parquet")
    d.tokens.to_parquet(tmp_path / "tokens.parquet")
    d.sequences.to_parquet(tmp_path / "sequences.parquet")
    from sentinel.models.microbe.design import load_design

    rule = {"kind": "date_from", "date": "2019-06-01"}
    dev = load_design(root=tmp_path, holdout_rule=rule)
    assert (pd.to_datetime(dev.samples["collect_date"]) < "2019-06-01").all()
    assert set(dev.tokens["sample_id"]) <= set(dev.samples.index)
    full = load_design(root=tmp_path, holdout_rule=rule, allow_holdout=True)
    assert len(full.samples) > len(dev.samples)


def test_seq_embedding_mean_and_seqemb_models():
    d, seqs, sid = _synthetic_design()
    table = np.zeros((len(sid) + 1, 64), np.float32)
    table[1:] = dna.kmer_profile(seqs, k=3)
    seq_row = {s: i + 1 for i, s in enumerate(sid)}
    ids = d.samples.index[:3]
    X = B.seq_embedding_mean(d.tokens, ids, table, seq_row)
    t = d.tokens[d.tokens["sample_id"] == ids[0]]
    w = t["count"].to_numpy(float) / t["count"].sum()
    ref = (table[t["seq_id"].map(seq_row).to_numpy()] * w[:, None]).sum(0)
    assert np.allclose(X[0], ref, atol=1e-6)
    f = SP.make_folds(d.samples, "lodo")[0]
    for name in ("ridge_seqemb", "rf_seqemb"):
        p = B.fit_predict(name, d, f.train, f.test, targets=TARGET_NAMES, n_jobs=1,
                          params={"seq_table": (table, seq_row),
                                  "rf": {"n_estimators": 10, "min_samples_leaf": 3, "max_features": 0.3}})
        assert np.isfinite(p.to_numpy()).all()


def test_seq_cluster_clr_uses_training_sequences_only():
    d, seqs, sid = _synthetic_design()
    table = np.zeros((len(sid) + 1, 64), np.float32)
    table[1:] = dna.kmer_profile(seqs, k=3)
    sr = {s: i + 1 for i, s in enumerate(sid)}
    f = SP.make_folds(d.samples, "lodo")[0]
    Xtr, Xte = B.seq_cluster_clr(d.tokens, f.train, [f.train, f.test], table, sr, k=5, n_pca=4)
    assert Xtr.shape[1] == Xte.shape[1] and np.allclose(Xtr.sum(1), 0, atol=1e-4)
    p = B.fit_predict("ridge_seqclust", d, f.train, f.test, targets=TARGET_NAMES, params={"seq_table": (table, sr),
                      "seqclust_k": 5})
    assert np.isfinite(p.to_numpy()).all()


def test_multires_and_hybrid_seqclust_models_run(monkeypatch):
    d, seqs, sid = _synthetic_design()
    table = np.zeros((len(sid) + 1, 64), np.float32)
    table[1:] = dna.kmer_profile(seqs, k=3)
    sr = {s: i + 1 for i, s in enumerate(sid)}
    monkeypatch.setattr(B, "SEQCLUST_MULTI_K", (3, 5))
    f = SP.make_folds(d.samples, "lodo")[0]
    for name in ("ridge_seqclust_multi", "ridge_clr_seqclust"):
        p = B.fit_predict(name, d, f.train, f.test, targets=TARGET_NAMES, params={"seq_table": (table, sr),
                          "seqclust_k": 5})
        assert np.isfinite(p.to_numpy()).all()


def test_v4_window_maps_both_primer_sets_of_one_organism_to_one_unit():
    rs = np.random.default_rng(0)
    rnd = lambda n: "".join(rs.choice(list("ACGT"), n))
    v3, v4, v5 = rnd(150), rnd(260), rnd(130)
    primer = "GTGCCAGCAGCCGCGGTAA"                       # 515F-Y instance
    v3v4 = v3 + primer + v4[:240]                        # Pro341F/805R amplicon contains the 515F site
    v4v5 = v4 + v5                                       # 515F-Y/926R amplicon starts right after 515F
    assert dna.v4_window(v3v4, "V3V4") == dna.v4_window(v4v5, "V4V5") == v4[:dna.V4_LEN]
    assert dna.v4_window(v3, "V3V4") is None and dna.v4_window(v4v5, "V4") is None
    tok = pd.DataFrame({"sample_id": ["a", "b"], "seq_id": ["s1", "s2"], "count": [5, 7]})
    seqs = pd.DataFrame({"seq_id": ["s1", "s2"], "sequence": [v3v4, v4v5]})
    smp = pd.DataFrame({"primer_set": ["V3V4", "V4V5"]}, index=pd.Index(["a", "b"], name="sample_id"))
    out = dna.add_v4_units(tok, seqs, smp)
    assert out["v4"].nunique() == 1 and out["v4"].notna().all()


def test_v4_models_run(monkeypatch):
    import functools

    d, seqs, sid = _synthetic_design()          # 60-bp synthetic reads: use a 40-bp window for the code path
    monkeypatch.setattr(dna, "add_v4_units", functools.partial(dna.add_v4_units, L=40))
    monkeypatch.setattr(dna, "V4_CORE_515F", __import__("re").compile("A"))
    B._V4_CACHE.clear()
    f = SP.make_folds(d.samples, "lodo")[0]
    for name in ("ridge_v4clr", "xgb_v4clr", "xgb_v4clr_context", "xgb_clr_context"):
        p = B.fit_predict(name, d, f.train, f.test, targets=TARGET_NAMES, n_jobs=1)
        assert p.shape[0] == len(f.test)


def test_event_derived_probability_is_cross_fitted_by_domain():
    """SYNTHETIC: a domain's own residuals never set its sigma; probabilities in [0, 1]."""
    from sentinel.events import microbe as ME

    d, _, _ = _synthetic_design()
    S = d.samples.copy()
    oof = S[["log_tp"]].copy() + 0.1
    big = S.index[S["domain_id"] == "D00"]
    oof.loc[big, "log_tp"] += 5.0                        # huge errors in D00 only
    r = ME.derived_probability(S, oof, "E5b")
    res = (S["log_tp"] - oof["log_tp"]).to_numpy()
    dom = S["domain_id"].to_numpy()
    assert np.isclose(r["sigma_by_domain"]["D00"], np.std(res[dom != "D00"], ddof=1))
    p = r["p"][np.isfinite(r["p"])]
    assert ((p >= 0) & (p <= 1)).all()
    assert set(np.unique(r["y"][np.isfinite(r["y"])])) <= {0.0, 1.0}
