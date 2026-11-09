# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Unit tests for the M5-X development machinery and the one-time scorer (synthetic toy data only)."""
from __future__ import annotations

import importlib.util
import json
import stat
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


def _toy_design(n=40, seed=0):
    from sentinel.models.bioassess.data import BioDesign

    rng = np.random.default_rng(seed)
    ids = [f"uk:{i}" for i in range(n // 2)] + [f"nrsa:{i}" for i in range(n // 2)]
    taxa = [f"t{k}" for k in range(12)]
    rows = []
    for i in ids:
        for t in rng.choice(taxa, size=rng.integers(3, 8), replace=False):
            rows.append({"visit_id": i, "mixed_otu": t, "name": t, "phylum": "p", "class": "c", "order": "o",
                         "family": "f" + t, "genus": "g" + t, "count": float(rng.integers(1, 50))})
    tok = pd.DataFrame(rows)
    agg = tok.groupby("visit_id")["count"].agg(["sum", "size"])
    s = pd.DataFrame({"source": [i.split(":")[0] for i in ids], "survey": "x",
                      "ag_eco9": rng.choice(["A", "B", "C"], n), "lat": rng.uniform(40, 50, n),
                      "lon": rng.uniform(-5, 5, n), "elevation_m": rng.uniform(0, 500, n),
                      "log10_total_count": np.log10(agg["sum"].reindex(ids).to_numpy()),
                      "log10_n_taxa": np.log10(agg["size"].reindex(ids).to_numpy()), "boatable": 0.0,
                      "analysis_method": ["ANLA" if i.startswith("uk") else None for i in ids],
                      "location_id": [f"L{k}" for k in range(n)], "reference_class": "R"}, index=pd.Index(ids))
    Y = pd.DataFrame(index=s.index)
    Y["ph_lab"] = np.where(s["source"] == "nrsa", rng.normal(7, 1, n), np.nan)
    Y["uk_ph"] = np.where(s["source"] == "uk", rng.normal(7.5, 1, n), np.nan)
    Y["log10_tp_ug_l"] = np.where(s["source"] == "nrsa", rng.normal(1, 0.3, n), np.nan)
    return BioDesign(s, tok, Y, {})


def test_xfeaturizer_matches_m5_community_featurizer():
    from sentinel.models.bioassess import xdev as XD
    from sentinel.models.bioassess.features import CommunityFeaturizer

    D = _toy_design()
    tr, te = D.samples.index[:30], D.samples.index[30:]
    a = CommunityFeaturizer(min_samples=3, descriptors=False).fit(D, tr)
    TM = XD.TaxonMatrix(D)
    b = XD.XFeaturizer("X-L", min_samples=3, descriptors=False, onehots=False).fit(TM, D, tr)
    assert a.taxa_ == b.taxa_
    np.testing.assert_allclose(a.transform(D, te), b.transform(TM, D, te), atol=1e-12)


def test_stratified_subsample_is_proportional_and_seeded():
    from sentinel.models.bioassess import xdev as XD

    g = np.array(["a"] * 600 + ["b"] * 300 + ["c"] * 100)
    p1, p2 = XD.stratified_subsample(g, 100, 0), XD.stratified_subsample(g, 100, 0)
    assert len(p1) == 100 and np.array_equal(p1, p2)
    assert {k: int((g[p1] == k).sum()) for k in "abc"} == {"a": 60, "b": 30, "c": 10}
    assert len(XD.stratified_subsample(g, 5000, 0)) == len(g)


def test_multi_source_labels_only_for_shared_targets():
    from sentinel.models.bioassess import xdev as XD

    D = _toy_design()
    ids = D.samples.index
    same = XD.labels(D, "X-L", "ph_lab", ids, "same")
    multi = XD.labels(D, "X-L", "ph_lab", ids, "multi")
    uk = (D.samples["source"] == "uk").to_numpy()
    assert np.isnan(same[uk]).all() and np.allclose(multi[uk], D.Y["uk_ph"].to_numpy()[uk])
    assert np.allclose(multi[~uk], same[~uk])
    tp = XD.labels(D, "X-L", "log10_tp_ug_l", ids, "multi")
    assert np.isnan(tp[uk]).all()                      # not a shared target: same-continent only
    assert XD.variants_for("log10_tp_ug_l") == ("same",) and XD.variants_for("uk_ph") == ("same", "multi")


def test_union_labels_and_block_mapping():
    from sentinel.models.bioassess import xbioset as XB

    D = _toy_design()
    D.Y["uk_log10_cond25_us_cm"] = np.nan
    D.Y["log10_cond_us_cm"] = np.nan
    for c in ("log10_tn_mg_l", "log10_chloride_mg_l", "log10_turbidity_ntu", "uk_log10_chloride_mg_l",
              "uk_log10_orthop_mg_l", "uk_log10_nh4n_mg_l", "uk_log10_nitrate_mg_l"):
        D.Y[c] = np.nan
    U = XB.union_labels(D)
    uk = (D.samples["source"] == "uk").to_numpy()
    assert np.allclose(U["ph_lab"].to_numpy()[uk], D.Y["uk_ph"].to_numpy()[uk])
    assert U["log10_tp_ug_l"].to_numpy()[uk].tolist() == [] or np.isnan(U["log10_tp_ug_l"].to_numpy()[uk]).all()
    Us = XB.union_labels(D, "X-U")
    assert np.isnan(Us["ph_lab"].to_numpy()[~uk]).all()
    P = pd.DataFrame(np.arange(18.0).reshape(2, 9), columns=list(__import__(
        "sentinel.models.bioassess.xdev", fromlist=["x"]).UNION_COLUMNS))
    B = XB.to_block(P, "X-U")
    assert list(B.columns)[0] == "uk_ph" and (B["uk_ph"] == P["ph_lab"]).all()


def _final():
    spec = importlib.util.spec_from_file_location("x_final", REPO / "scripts/eval/bioassess_x_final.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_x_final_decision_table_rows():
    F = _final()
    base = {"block": "X-U", "holm_rejected": True, "all_seeds_positive": True, "region_ci_low": 0.01,
            "ci_low": 0.01, "ci_high": 0.05, "realised_mde": 0.03}
    assert F.label(base)[0].startswith("SOTA")
    assert F.label({**base, "region_ci_low": -0.01}) == ("Above, not robust", "region-unit paired CI > 0")
    nr = {**base, "holm_rejected": False}
    assert F.label({**nr, "ci_low": -0.05, "ci_high": -0.01})[0] == "Below"
    assert F.label({**nr, "ci_low": -0.01, "ci_high": 0.03, "realised_mde": 0.04})[0].startswith("Inconclusive")
    assert F.label({**nr, "ci_low": -0.005, "ci_high": 0.005, "realised_mde": 0.009})[0] == "Competitive"
    assert F.label({**nr, "delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")})[0] == "Not estimable"


def test_x_final_holm_matches_definition():
    F = _final()
    adj = F.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == {"a": 0.03, "c": 0.06, "b": 0.06}


def _guard(tmp_path, monkeypatch):
    from sentinel.core import paths
    from sentinel.models.bioassess import xtestlook as XL

    monkeypatch.setattr(XL, "FREEZE_FILE", tmp_path / "bioassess_x_freeze.json")
    monkeypatch.setattr(XL, "CANDIDATE_FILE", tmp_path / "bioassess_x_freeze_candidate.json")
    monkeypatch.setattr(paths, "REGISTRY_PATH", tmp_path / "registry.jsonl")
    monkeypatch.setattr(XL, "scoring_code_sha256", lambda: "c0de")
    return XL


def _freeze_rec(XL, **kw):
    rec = {"status": "frozen", "candidate_sha256": XL.sha_file(XL.CANDIDATE_FILE), "scoring_code_sha256": "c0de",
           "data_snapshots": {"x": 1}, "candidates": {"a": 1}, "comparators": {"a": 1}, "seeds": [0, 1, 2],
           "n_boot": 2000, "n_boot_confirmatory": 10000, "alpha": 0.05}
    rec.update(kw)
    return rec


def test_x_testlook_accepts_only_the_lead_freeze_file(tmp_path, monkeypatch):
    XL = _guard(tmp_path, monkeypatch)
    XL.CANDIDATE_FILE.write_text(json.dumps({"status": "candidate"}))
    with pytest.raises(XL.SingleLookError, match="does not exist"):
        XL.load_freeze()
    with pytest.raises(XL.SingleLookError, match="candidate record is never accepted"):
        XL.load_freeze(XL.CANDIDATE_FILE)
    XL.FREEZE_FILE.write_text(json.dumps(_freeze_rec(XL, status="candidate")))
    with pytest.raises(XL.SingleLookError, match="not 'frozen'"):
        XL.load_freeze()
    XL.FREEZE_FILE.write_text(json.dumps(_freeze_rec(XL, candidate_sha256="0" * 64)))
    with pytest.raises(XL.SingleLookError, match="does not match the candidate"):
        XL.load_freeze()
    XL.FREEZE_FILE.write_text(json.dumps(_freeze_rec(XL, scoring_code_sha256="other")))
    with pytest.raises(XL.SingleLookError, match="scoring code changed"):
        XL.load_freeze()
    XL.FREEZE_FILE.write_text(json.dumps(_freeze_rec(XL)))
    assert XL.load_freeze()["_sha256"] == XL.sha_file(XL.FREEZE_FILE)


def test_x_testlook_claim_seal_score_single_look(tmp_path, monkeypatch):
    XL = _guard(tmp_path, monkeypatch)
    XL.CANDIDATE_FILE.write_text("{}")
    XL.FREEZE_FILE.write_text(json.dumps(_freeze_rec(XL)))
    fr = XL.load_freeze()
    root = tmp_path / "look"
    with pytest.raises(XL.SingleLookError, match="open, unscored"):
        XL.require_open_claim(root)
    XL.claim(fr, "lead", tmp_path / "run", root=root)
    assert XL.registry_rows()[0]["metric"] == "look_claimed"
    assert not (root / "TEST_LOOK.json").stat().st_mode & stat.S_IWUSR
    XL.require_open_claim(root)
    with pytest.raises(XL.SingleLookError, match="registry already holds"):
        XL.claim(fr, "lead", tmp_path / "run", root=root)
    (tmp_path / "reg2.jsonl").write_text("")
    with pytest.raises(XL.SingleLookError, match="already claimed"):     # O_EXCL marker even with a clean registry
        XL.claim(fr, "lead", tmp_path / "run", root=root, registry_path=tmp_path / "reg2.jsonl")
    XL.claim_resume("lead", tmp_path / "run", root=root)                 # the single resume
    with pytest.raises(XL.SingleLookError, match="already used"):
        XL.claim_resume("lead", tmp_path / "run", root=root)
    pdir = tmp_path / "run" / "predictions"
    pdir.mkdir(parents=True)
    pd.DataFrame({"a": [1.0]}).to_parquet(pdir / "m__s0.parquet")
    XL.seal_predictions(pdir, root=root)
    with pytest.raises(XL.SingleLookError, match="already sealed"):
        XL.seal_predictions(pdir, root=root)
    assert XL.check_sealed(root=root) == pdir
    XL.mark_scored({"x": 1}, root=root)
    with pytest.raises(XL.SingleLookError, match="already scored"):
        XL.check_sealed(root=root)
    with pytest.raises(XL.SingleLookError, match="open, unscored"):
        XL.require_open_claim(root)
    with pytest.raises(FileExistsError):
        XL.mark_scored({"x": 2}, root=root)


def test_x_testlook_sealed_prediction_tamper_detected(tmp_path, monkeypatch):
    XL = _guard(tmp_path, monkeypatch)
    root, pdir = tmp_path / "look", tmp_path / "p"
    pdir.mkdir()
    pd.DataFrame({"a": [1.0]}).to_parquet(pdir / "m.parquet")
    XL.seal_predictions(pdir, root=root)
    f = pdir / "m.parquet"
    f.chmod(0o644)
    pd.DataFrame({"a": [2.0]}).to_parquet(f)
    with pytest.raises(XL.SingleLookError, match="modified"):
        XL.check_sealed(root=root)


def test_x_testlook_data_pins_detect_content_change(tmp_path, monkeypatch):
    from sentinel.core import paths

    XL = _guard(tmp_path, monkeypatch)
    proc, man = tmp_path / "processed", tmp_path / "manifests"
    (proc / "d").mkdir(parents=True)
    man.mkdir()
    (proc / "d" / "a.parquet").write_bytes(b"1")
    (man / "m.json").write_text("{}")
    monkeypatch.setattr(paths, "PROCESSED", proc)
    monkeypatch.setattr(paths, "MANIFESTS", man)
    monkeypatch.setattr(XL, "DATA_PINS", {"r": ("m", "d", ("a.parquet",))})
    rec = {"data_snapshots": XL.make_data_pins()}
    XL.verify_data_pins(rec)
    (proc / "d" / "a.parquet").write_bytes(b"2")
    with pytest.raises(XL.SingleLookError, match="content changed"):
        XL.verify_data_pins(rec)


def test_x_test_rows_need_an_open_claim(monkeypatch, tmp_path):
    from sentinel.models.bioassess import xdev as XD
    from sentinel.models.bioassess import xtestlook as XL

    monkeypatch.setattr(XL, "look_dir", lambda root=None: tmp_path / "nolook")
    with pytest.raises(XL.SingleLookError):
        XD.load_test_rows(pd.DataFrame(), "X-L")


def test_x_final_record_verification_detects_changed_hash(tmp_path, monkeypatch):
    F = _final()
    f = tmp_path / "code.py"
    f.write_text("a = 1\n")
    from sentinel.core.registry import file_sha256

    rec = {"sha256": {str(f): file_sha256(f)}, "data_sha256": {}, "package_versions": F.package_versions(["numpy"]),
           "r_package_versions": {"x": "y"}}
    monkeypatch.setattr(F, "r_package_versions", lambda: {"x": "y"})
    assert F.verify_record(rec) == []
    f.write_text("a = 2\n")
    assert any("hash changed" in b for b in F.verify_record(rec))


def test_x_final_seal_and_cache_tamper(tmp_path):
    F = _final()
    c = F.Cache(tmp_path / "cache")
    c.put("k", pd.DataFrame({"a": [1.0]}))
    assert c.get("k")["a"].iloc[0] == 1.0
    pd.DataFrame({"a": [2.0]}).to_parquet(tmp_path / "cache" / "k.parquet")
    with pytest.raises(SystemExit, match="tampered"):
        F.Cache(tmp_path / "cache").get("k")
    (tmp_path / "out.json").write_text("{}")
    F.seal(tmp_path)
    assert (tmp_path / "SHA256SUMS").exists()
    assert not (tmp_path / "out.json").stat().st_mode & stat.S_IWUSR


def test_chunked_nograd_forward_equals_full_batch():
    import torch

    from sentinel.models.bioassess import bioset as BS
    from sentinel.models.bioassess import xbioset as XB
    from sentinel.models.bioassess.cv import load_bioset_config

    cfg = load_bioset_config(REPO / "configs/bioassess/bioset.yaml", "bioset_st_flat")
    torch.manual_seed(0)
    sizes = {lv: 20 for lv in BS.LEVELS}
    m = BS.build_model(cfg, sizes, 3, 4).eval()
    n, T = 50, 7
    lev = torch.randint(2, 20, (n, T, len(BS.LEVELS)))
    ab = torch.rand(n, T, BS.N_ABUND)
    mask = torch.rand(n, T) > 0.3
    mask[:, 0] = True
    g = torch.randn(n, 3)
    with torch.no_grad():
        full = m(lev, ab, mask, g)
        XB._chunk_nograd(m, max_rows=8)
        chunked = m(lev, ab, mask, g)
    torch.testing.assert_close(full, chunked, rtol=1e-5, atol=1e-5)
    with XB._ChunkedBuild():
        assert BS.build_model is not XB._ChunkedBuild
    from sentinel.models.bioassess import bioset as BS2
    assert BS2.build_model.__name__ == "build_model"


def test_x_event_track_units_climatology_and_power_sets():
    from sentinel.events import bioassess_x as XE

    assert np.isclose(XE.to_event_units([np.log10(230.0)], "X-L", "E2a")[0], 230.0)
    assert np.isclose(XE.to_event_units([2.0], "X-L", "E5b")[0], 0.1)          # 100 µg/L = 0.1 mg/L
    assert XE.to_event_units([7.1], "X-U", "E6")[0] == 7.1
    assert XE.POWERED.isdisjoint(XE.DESCRIPTIVE_ONLY)
    assert ("X-U", "E6") in XE.POWERED and ("X-U", "E2a") in XE.DESCRIPTIVE_ONLY   # v8a: the power rule governs
    assert set(XE.TRACK) == XE.POWERED | XE.DESCRIPTIVE_ONLY
    doy = np.array([15.0, 196.0])
    m = XE.month_of(np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25))
    assert m.tolist() == [1, 7]
    prev, clim = XE.prevalence_and_climatology(np.array([1, 0, 0, 1, np.nan]), np.array([1, 1, 7, 7, 7]),
                                               np.array([1, 7, 12]))
    assert np.allclose(prev, 0.5) and np.isclose(clim[0], (1 + 0.5) / 3) and np.isclose(clim[2], 0.5)
    pw = XE.power_flag(np.array([1] * 30 + [0] * 5), np.array([f"c{i % 12}" for i in range(35)]))
    assert pw["n_pos"] == 30 and pw["n_clusters_pos"] == 12 and pw["flag"] == "F"


def test_x_scoring_code_hash_tolerates_missing_shared_events_module():
    from sentinel.models.bioassess import xtestlook as XL

    assert "sentinel/eval/events.py" in XL.SCORING_CODE and len(XL.scoring_code_sha256()) == 64


def test_x_event_derived_probability_monotone_with_shared_module():
    pytest.importorskip("sentinel.eval.events")
    from sentinel.events import bioassess_x as XE

    mu = np.log10(np.array([50.0, 230.0, 1000.0]))
    p = XE.derived_probability(mu, 0.2, "X-L", "E2a")
    assert np.all(np.diff(p) > 0) and abs(p[1] - 0.5) < 0.05
    assert XE.labels(np.log10([100.0, 300.0]), "X-L", "E2a").tolist() == [0.0, 1.0]
    assert XE.labels(np.log10([230.0, 20.0]), "X-L", "E2a").tolist() == [0.0, 0.0]     # boundary: not > 230
    assert XE.labels([2.0], "X-L", "E5b").tolist() == [0.0]                              # 100 µg/L = 0.1, not > 0.1
