# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Unit tests for the M5-X development machinery and the one-time scorer (synthetic toy data only)."""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import pickle
import re
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


def test_x_final_label_listed_comparators_v9_v10():
    """Amendments v9 (row 1 needs every listed comparator beaten; row 2 names those not beaten) and v10 (row 3 fires
    when any listed comparator is significantly above C-X; a non-estimable comparator cannot fire it)."""
    F = _final()
    win = {"block": "X-U", "holm_rejected": True, "all_seeds_positive": True, "region_ci_low": 0.01, "delta": 0.03,
           "ci_low": 0.01, "ci_high": 0.05, "realised_mde": 0.03, "listed_not_beaten": [],
           "listed_above_candidate": [], "listed_not_estimable": []}
    tie = {**win, "holm_rejected": False, "delta": 0.0, "ci_low": -0.005, "ci_high": 0.005, "realised_mde": 0.009}
    # (i) tie with the per-target comparator + another listed comparator significantly above -> Below, named
    lab, why = F.label({**tie, "listed_not_beaten": ["tabpfn_community", "gbm_community"],
                        "listed_above_candidate": ["gbm_community"]})
    assert lab == "Below" and "gbm_community" in why and "significantly above" in why and "tabpfn_community" in why
    assert F.label({**tie, "listed_not_beaten": ["tabpfn_community"]})[0] == "Competitive"       # nobody above
    # ... also when the family test is Holm-rejected, or would be only inconclusive
    assert F.label({**win, "listed_not_beaten": ["wapls2"], "listed_above_candidate": ["wapls2"]})[0] == "Below"
    assert F.label({**tie, "ci_low": -0.03, "ci_high": 0.03, "realised_mde": 0.06, "listed_not_beaten": ["a"],
                    "listed_above_candidate": ["a"]})[0] == "Below"
    # (ii) every listed comparator beaten (all lower bounds > 0) -> row 1
    assert F.label(win) == ("SOTA (M5-X, X-U, within-region)", "")
    # row 2: one listed comparator not beaten -> "Above, not robust", each one named
    lab, why = F.label({**win, "listed_not_beaten": ["rf_community", "bioinfer"]})
    assert lab == "Above, not robust" and "rf_community" in why and "bioinfer" in why
    assert "paired CI lower bound > 0 against every listed comparator" in why
    # (iii) a comparator with no finite estimate: not beaten for row 1, cannot fire row 3, named "not estimable"
    lab, why = F.label({**win, "listed_not_beaten": ["knn_community"], "listed_not_estimable": ["knn_community"]})
    assert lab == "Above, not robust" and "not estimable" in why and "knn_community" in why
    lab, why = F.label({**tie, "listed_not_beaten": ["knn_community"], "listed_not_estimable": ["knn_community"]})
    assert lab == "Competitive" and "not estimable" in why
    # the change can only make a label harsher: with the same family row, adding listed information never improves it
    order = ["Below", "Competitive", "Inconclusive (underpowered)", "Above, not robust", "SOTA (M5-X, X-U, within-region)"]
    for fam in (win, tie, {**tie, "ci_low": -0.03, "ci_high": 0.03, "realised_mde": 0.06},
                {**tie, "ci_low": -0.05, "ci_high": -0.01}):
        plain = F.label({k: v for k, v in fam.items() if not k.startswith("listed_")})[0]
        for extra in ({"listed_not_beaten": ["a"]}, {"listed_not_beaten": ["a"], "listed_above_candidate": ["a"]},
                      {"listed_not_beaten": ["a"], "listed_not_estimable": ["a"]}):
            assert order.index(F.label({**fam, **extra})[0]) <= order.index(plain)


def _B(n):
    return {"kind": "baseline", "name": n, "variant": {}}


_BS = {"kind": "bioset", "config": "bsx_flat_geo"}
_NOT_COMPARED = {"gjam": {"status": "not compared", "fidelity": {b: {"fidelity_pass": False} for b in ("X-L", "X-U")}},
                 "trpi": {"status": "not compared"}}


def _listed_rec(F):
    """A record with the protocol's full comparator list (v11): C-X = mean(BioSet, RF[community+geography], TabPFN);
    ``gbm_community`` is the best listed single baseline outside C-X, so rule (ii) picks it."""
    import copy

    from sentinel.models.bioassess import xdev as XD

    rec = {"seeds": [0, 1, 2], "candidates": {}, "comparators": {}, "comparator_set": {},
           "reproducibility_tolerance": dict(F.TOLERANCE), "not_compared": copy.deepcopy(_NOT_COMPARED)}
    for b in XD.BLOCKS:
        cs = {n: {"spec": _B(n), "class": "c", "dev_mean_r2_anomaly": 0.3 if n == "gbm_community" else 0.1}
              for n in F.LISTED[b] if not n.startswith(("bens_", "cxr_")) and n != "bioinfer"}
        cs["bioinfer"] = {"spec": {"kind": "r", "method": "bioinfer"}, "class": "c"}
        for n in F.LISTED[b]:
            if n.startswith("bens_"):
                cs[n] = {"spec": {"kind": "mean", "members": [_B("rf_community_geography"), _B("tabpfn_community")]},
                         "class": "c"}
        rec["comparator_set"][b] = cs
        rec["candidates"][b] = {"confirmatory": {"name": "ens", "spec": {"kind": "mean", "members": [
            _BS, _B("rf_community_geography"), _B("tabpfn_community")]}},
                                "secondary": [{"name": "bsx_flat_geo", "spec": _BS}]}
        rec["comparators"][b] = {t: {"name": "tabpfn_community", "spec": _B("tabpfn_community")} for t in XD.TARGETS[b]}
        for n, sp in F.rule_comparators(rec, b)[0].items():
            cs[n] = {"spec": sp, "class": "rule comparator"}
    return rec


def test_x_final_refuses_record_whose_comparator_list_differs(tmp_path, monkeypatch):
    """v9 item 1 + v11 item 1: the listed comparators are the protocol's (16 X-L, 19 X-U, the two rule comparators
    included and derived from the record's own C-X); any other list is refused before anything is set up."""
    import copy

    F = _final()
    from sentinel.models.bioassess import xdev as XD
    from sentinel.models.bioassess import xtestlook as XL

    extra = ["bioinfer", "bens_rfg_tabpfn", "bens_rfg_gbm_tabpfn", "bens_rfctx_tabpfn", "cxr_nobioset", "cxr_swap"]
    for b, n in (("X-L", 16), ("X-U", 19)):
        assert len(F.LISTED[b]) == n == len(set(F.LISTED[b]))
        assert sorted(F.LISTED[b]) == sorted([*XD.models_for(b), *extra])
    assert "nearest_reference" not in F.LISTED["X-U"] and "idx_rf_geo" not in F.LISTED["X-L"]
    rec = _listed_rec(F)
    assert F.check_listed(rec) == []
    specs = F.model_specs(rec, "X-U")                      # every listed comparator is fitted, each once
    assert set(F.LISTED["X-U"]) <= set(specs) and specs["rf_geography"] == rec["comparator_set"]["X-U"]["rf_geography"]["spec"]
    # the rule comparators (v11 item 1): no BioSet member; (ii) adds the best listed single baseline outside C-X
    for b in XD.BLOCKS:
        rule, swap = F.rule_comparators(rec, b)
        assert swap == "gbm_community" and sorted(rule) == sorted(F.RULE_COMPARATORS)
        assert [m["name"] for m in rule["cxr_nobioset"]["members"]] == ["rf_community_geography", "tabpfn_community"]
        assert [m["name"] for m in rule["cxr_swap"]["members"]] == ["rf_community_geography", "tabpfn_community",
                                                                    "gbm_community"]
        assert not any(lf["kind"] == "bioset" for sp in rule.values() for lf in F.leaf_specs(sp))

    def faults(change):
        r = copy.deepcopy(rec)
        change(r)
        return F.check_listed(r)

    assert any("missing ['bioinfer']" in x for x in faults(lambda r: r["comparator_set"]["X-L"].pop("bioinfer")))
    bad = faults(lambda r: r["comparator_set"]["X-U"].update(gjam={"spec": {"kind": "r", "method": "gjam"}, "class": "c"}))
    assert any("extra ['gjam']" in x for x in bad) and any("not a comparator" in x for x in bad)
    bad = faults(lambda r: r["comparators"]["X-L"].update(log10_tp_ug_l={"name": "rf_community", "spec": {
        "kind": "baseline", "name": "rf_community", "variant": {"log10_tp_ug_l": "multi"}}}))
    assert any("not listed with the same spec" in x for x in bad)
    bad = faults(lambda r: r["not_compared"]["gjam"]["fidelity"]["X-U"].update(fidelity_pass=True))
    assert any("GJAM fidelity record for X-U" in x for x in bad)
    assert any("not_compared" in x for x in faults(lambda r: r["not_compared"].pop("trpi")))
    # rule comparators that do not follow from the record's own C-X
    bad = faults(lambda r: r["comparator_set"]["X-U"]["cxr_swap"]["spec"]["members"].pop())
    assert any("X-U: rule comparator cxr_swap does not follow" in x for x in bad)
    bad = faults(lambda r: r["comparator_set"]["X-L"]["knn_community"].update(dev_mean_r2_anomaly=0.9))
    assert any("X-L: rule comparator cxr_swap does not follow" in x for x in bad)       # another single is the best
    bad = faults(lambda r: r["comparator_set"]["X-L"]["cxr_nobioset"]["spec"]["members"].append(dict(_BS)))
    assert any("not a comparator" in x for x in bad) and any("cxr_nobioset does not follow" in x for x in bad)
    bad = faults(lambda r: r["candidates"]["X-L"]["confirmatory"].update(spec=dict(_BS)))
    assert any("rule comparators cannot be derived" in x for x in bad)
    bad = faults(lambda r: r["comparator_set"]["X-U"]["wapls2"].pop("dev_mean_r2_anomaly"))
    assert any("rule comparators cannot be derived" in x for x in bad)
    bad = faults(lambda r: r["comparator_set"]["X-U"]["tabpfn_community"]["spec"].update(variant={"uk_ph": "multi"}))
    assert any("C-X baseline member tabpfn_community is not listed with the same spec" in x for x in bad)
    assert any("reproducibility_tolerance" in x for x in faults(lambda r: r["reproducibility_tolerance"].update(delta=1e-3)))
    assert any("reproducibility_tolerance" in x for x in faults(lambda r: r.pop("reproducibility_tolerance")))
    # a pre-v9 record is refused by the first thing every command does
    old = {k: v for k, v in rec.items() if k not in ("comparator_set", "not_compared")}
    cand = tmp_path / "candidate.json"
    cand.write_text(json.dumps(old))
    monkeypatch.setattr(XL, "CANDIDATE_FILE", cand)
    monkeypatch.setattr(XL, "verify_data_pins", lambda rec: {})
    monkeypatch.setattr(XL, "verify_artifacts", lambda rec: None)
    monkeypatch.setattr(F, "verify_record", lambda rec, check_data=True: [])
    with pytest.raises(SystemExit, match="record verification failed"):
        F.load_record(True)
    cand.write_text(json.dumps(rec))
    assert F.load_record(True)[1] == XL.sha_file(cand)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["candidate.json"]   # refused before anything is written


def test_x_final_scores_every_listed_comparator():
    """score_blocks: one paired estimate per (target, listed comparator); the family row is unchanged; labels follow
    v9/v10; the parallel bootstrap equals the serial one."""
    F = _final()
    from sentinel.models.bioassess import xdev as XD

    b, rs, n = "X-L", np.random.default_rng(0), 360
    T = list(XD.TARGETS[b])
    ids = pd.Index([f"v{i}" for i in range(n)])
    region = np.repeat(["r1", "r2", "r3"], n // 3)
    meta = pd.DataFrame({"region": region, "cluster": [f"c{i // 2}" for i in range(n)], "lat": 40 + rs.random(n),
                         "lon": -100 + rs.random(n)}, index=ids)
    Y = pd.DataFrame({t: rs.normal(size=n) + (region == "r2") for t in T}, index=ids)
    noisy = lambda sd: Y + rs.normal(scale=sd, size=Y.shape)                    # noqa: E731
    cx = noisy(0.4)
    frames = {"cx": cx, "good": noisy(0.9), "tie": cx + rs.normal(scale=1e-3, size=Y.shape), "better": noisy(0.05),
              "nanc": Y * np.nan}
    pred = {b: {m: {s: f for s in (0, 1, 2)} for m, f in frames.items()}}

    def rec_for(names):
        return {"n_boot": 40, "n_boot_confirmatory": 200, "alpha": 0.05,
                "candidates": {b: {"confirmatory": {"name": "cx"}, "secondary": []}},
                "comparators": {b: {t: {"name": "good"} for t in T}},
                "comparator_set": {b: {m: {"spec": {}, "class": "k"} for m in names}}}

    C, M, L = F.score_blocks(pred, {b: Y}, {b: meta}, rec_for(["good", "tie", "better", "nanc"]), n_jobs=1)
    assert len(L) == len(T) * 4 and len(C) == len(T) and set(M["model"]) == {"cx", "good", "tie", "better"}
    fam = L[L["is_per_target_comparator"]].set_index("target").loc[C["target"]]
    assert (fam["comparator"] == "good").all()
    for k in ("delta", "ci_low", "ci_high"):             # the listed row of the per-target comparator = the family row
        assert np.allclose(fam[k].to_numpy(), C[k].to_numpy())
    for r in C.itertuples():
        assert r.n_listed == 4 and r.listed_above_candidate == ["better"] and r.listed_not_estimable == ["nanc"]
        assert r.listed_not_beaten == ["tie", "better", "nanc"] and not r.all_listed_beaten
        assert r.label == "Below" and "better" in r.label_reason and "nanc" in r.label_reason
    Lp = pd.DataFrame(F.score_listed(pred[b], Y, meta["region"].to_numpy(), meta["cluster"].to_numpy(),
                                     rec_for(["good", "tie", "better", "nanc"]), b, n_jobs=2))
    pd.testing.assert_frame_equal(L.reset_index(drop=True), Lp)                # parallel == serial, bit for bit
    C1, _, _ = F.score_blocks(pred, {b: Y}, {b: meta}, rec_for(["good"]), n_jobs=1)
    assert C1["all_listed_beaten"].all() and C1["label"].str.startswith("SOTA").all()
    C2, _, _ = F.score_blocks(pred, {b: Y}, {b: meta}, rec_for(["good", "nanc"]), n_jobs=1)
    assert (C2["label"] == "Above, not robust").all() and C2["label_reason"].str.contains("not estimable").all()
    C3, _, _ = F.score_blocks(pred, {b: Y}, {b: meta}, rec_for(["good", "tie"]), n_jobs=1)
    assert (C3["label"] == "Above, not robust").all() and C3["label_reason"].str.contains("tie").all()


def test_x_freeze_candidate_lists_every_comparator_and_parses_the_gjam_log():
    F = _final()
    spec = importlib.util.spec_from_file_location("x_cand", REPO / "scripts/eval/bioassess_x_freeze_candidate.py")
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
    for b in ("X-L", "X-U"):
        assert sorted(G.listed_names(b)) == sorted(F.LISTED[b])
        assert all(G.comparator_class(m) for m in G.listed_names(b))
    assert G.comparator_spec("bioinfer", {}) == {"kind": "r", "method": "bioinfer"}
    assert [m["name"] for m in G.comparator_spec("bens_rfg_gbm_tabpfn", {})["members"]] == [
        "rf_community_geography", "gbm_community", "tabpfn_community"]
    txt = ("=====Warning messages:\n1: In min(x) : no non-missing\n$fidelity_r_insample_mode\n"
           "      log10_tp_ug_l       log10_tn_mg_l    log10_cond_us_cm log10_chloride_mg_l \n"
           "          0.7311942           0.7756483           0.8839625           0.8395583 \n"
           "log10_turbidity_ntu              ph_lab \n          0.7783815           0.8155498 \n\n"
           "$xpredMu_n_finite\n  x1   x2   x3   x4   x5   x6 \n4474 4474 4473 4474 4439 4443 \n\n"
           "$fidelity_pass\n[1] FALSE\n\n$n_fit\n[1] 4474\n\n$fit_minutes\n[1] 99.32365\n\n")
    info = G.parse_r_print(txt)
    assert list(info["fidelity_r_insample_mode"]) == ["log10_tp_ug_l", "log10_tn_mg_l", "log10_cond_us_cm",
                                                      "log10_chloride_mg_l", "log10_turbidity_ntu", "ph_lab"]
    assert info["fidelity_r_insample_mode"]["ph_lab"] == 0.8155498 and info["fidelity_pass"] is False
    assert list(info["xpredMu_n_finite"].values()) == [4474, 4474, 4473, 4474, 4439, 4443]
    assert info["n_fit"] == 4474 and info["fit_minutes"] == 99.32365
    with pytest.raises(ValueError):
        G.parse_r_print("$a\n x1 x2 \n 1 \n")


def test_x_final_holm_matches_definition():
    F = _final()
    adj = F.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == {"a": 0.03, "c": 0.06, "b": 0.06}


def _guard(tmp_path, monkeypatch):
    from sentinel.core import paths, registry
    from sentinel.models.bioassess import xtestlook as XL

    monkeypatch.setattr(XL, "FREEZE_FILE", tmp_path / "bioassess_x_freeze.json")
    monkeypatch.setattr(XL, "CANDIDATE_FILE", tmp_path / "bioassess_x_freeze_candidate.json")
    monkeypatch.setattr(XL, "REGISTRY", tmp_path / "registry.jsonl")
    monkeypatch.setattr(paths, "REGISTRY_PATH", tmp_path / "registry.jsonl")
    monkeypatch.setattr(XL, "scoring_code_sha256", lambda: "c0de")
    monkeypatch.setattr(registry, "code_state", lambda: {"toy": True})
    return XL


def _write_freeze(XL, tmp_path, **kw):
    """A toy candidate record with a pinned toy code snapshot, and the lead's freeze file made from it (the tmp
    paths of ``_guard``). ``kw`` changes the freeze file only."""
    src = tmp_path / "toy_src"
    src.mkdir(exist_ok=True)
    (src / "code.py").write_text("a = 1\n")
    snap = XL.make_snapshot(["code.py"], dest_root=tmp_path / "snaps", src_root=src)
    cand = {"status": "candidate", "scoring_code_sha256": "c0de", "data_snapshots": {"x": 1}, "candidates": {"a": 1},
            "comparators": {"a": 1}, "seeds": [0, 1, 2], "n_boot": 2000, "n_boot_confirmatory": 10000, "alpha": 0.05,
            "code_snapshot": snap}
    XL.CANDIDATE_FILE.write_text(json.dumps(cand))
    fr = {**cand, "status": "frozen", "candidate_sha256": XL.sha_file(XL.CANDIDATE_FILE), "frozen_by": "lead", **kw}
    XL.FREEZE_FILE.write_text(json.dumps({k: v for k, v in fr.items() if v is not None}))
    return fr


def test_x_testlook_accepts_only_the_lead_freeze_file(tmp_path, monkeypatch):
    XL = _guard(tmp_path, monkeypatch)
    XL.CANDIDATE_FILE.write_text(json.dumps({"status": "candidate"}))
    with pytest.raises(XL.SingleLookError, match="does not exist"):
        XL.load_freeze()
    with pytest.raises(XL.SingleLookError, match="candidate record is never accepted"):
        XL.load_freeze(XL.CANDIDATE_FILE)
    _write_freeze(XL, tmp_path, status="candidate")
    with pytest.raises(XL.SingleLookError, match="not 'frozen'"):
        XL.load_freeze(require_running=False)
    _write_freeze(XL, tmp_path, candidate_sha256="0" * 64)
    with pytest.raises(XL.SingleLookError, match="does not match the candidate"):
        XL.load_freeze(require_running=False)
    _write_freeze(XL, tmp_path, seeds=[0, 1])                     # any key other than the freeze-only ones
    with pytest.raises(XL.SingleLookError, match=r"differs from the candidate record in \['seeds'\]"):
        XL.load_freeze(require_running=False)
    _write_freeze(XL, tmp_path, scoring_code_sha256="other")
    with pytest.raises(XL.SingleLookError, match=r"differs from the candidate record in \['scoring_code_sha256'\]"):
        XL.load_freeze(require_running=False)
    _write_freeze(XL, tmp_path, code_snapshot=None)               # v11 item 3: no snapshot, no freeze
    with pytest.raises(XL.SingleLookError, match=r"lacks \['code_snapshot'\]"):
        XL.load_freeze(require_running=False)
    fr = _write_freeze(XL, tmp_path, frozen_at="2026-11-10", review="r", note="n")   # freeze-only keys may be added
    assert XL.load_freeze(require_running=False)["_sha256"] == XL.sha_file(XL.FREEZE_FILE)
    with pytest.raises(XL.SingleLookError, match="not running from the pinned code snapshot"):
        XL.load_freeze()                                          # the look runs from the snapshot only
    with monkeypatch.context() as m:
        m.setattr(XL, "scoring_code_sha256", lambda: "other")
        with pytest.raises(XL.SingleLookError, match="scoring code changed since the freeze"):
            XL.load_freeze(require_running=False)
    f = Path(fr["code_snapshot"]["dir"]) / "code.py"
    f.chmod(0o644)
    f.write_text("a = 2\n")
    with pytest.raises(XL.SingleLookError, match="code snapshot files changed"):
        XL.load_freeze(require_running=False)


def test_x_testlook_claim_units_seal_score_single_look(tmp_path, monkeypatch):
    import socket

    from sentinel.core.registry import log_result

    XL = _guard(tmp_path, monkeypatch)
    _write_freeze(XL, tmp_path)
    fr = XL.load_freeze(require_running=False)
    root, run = tmp_path / "look", tmp_path / "run"
    with pytest.raises(XL.SingleLookError, match="open, unscored"):
        XL.require_open_claim(root)
    XL.claim(fr, "lead", run, root=root)
    assert XL.registry_rows()[0]["metric"] == "look_claimed"
    assert not (root / "TEST_LOOK.json").stat().st_mode & stat.S_IWUSR
    XL.require_open_claim(root)                                          # default: the sha256 of the freeze file
    XL.require_open_claim(root, run_dir=run, freeze_sha256=fr["_sha256"])
    with pytest.raises(XL.SingleLookError, match="different freeze record"):
        XL.require_open_claim(root, freeze_sha256="0" * 64)
    with pytest.raises(XL.SingleLookError, match="belongs to run directory"):
        XL.require_open_claim(root, run_dir=tmp_path / "other")
    with pytest.raises(XL.SingleLookError, match="registry already holds"):
        XL.claim(fr, "lead", run, root=root)
    (tmp_path / "reg2.jsonl").write_text("")
    with pytest.raises(XL.SingleLookError, match="already claimed"):     # O_EXCL marker even with a clean registry
        XL.claim(fr, "lead", run, root=root, registry_path=tmp_path / "reg2.jsonl")
    assert not hasattr(XL, "claim_resume")                               # v11: chunks restart; no claim resume
    # write-once units
    df = pd.DataFrame({"a": [1.0, 2.0]}, index=["r1", "r2"])
    assert XL.unit_write(run, "u1", df, {"unit": {"key": "u1"}}) is True
    assert XL.unit_write(run, "u1", df * 2, {"unit": {"key": "u1"}}) is False     # never overwritten
    got, meta = XL.unit_read(run, "u1")
    ud = XL.unit_dir(run, "u1")
    assert got["a"].tolist() == [1.0, 2.0] and meta["sha256"] == XL.sha_file(ud / "pred.parquet")
    assert stat.S_IMODE(ud.stat().st_mode) == 0o555
    assert {stat.S_IMODE(p.stat().st_mode) for p in ud.iterdir()} == {0o444}
    assert XL.unit_write(run, "u_failed", None, {"error": "x"}) and XL.unit_read(run, "u_failed")[0] is None
    assert XL.unit_exists(run, "u1") and not XL.unit_exists(run, "u2")
    # advisory unit locks
    assert XL.unit_lock(run, "u2") and not XL.unit_lock(run, "u2")
    XL.unit_unlock(run, "u2")
    lock = run / "locks" / "u2.lock"
    lock.write_text(json.dumps({"pid": 2147483647, "host": socket.gethostname(), "t": 0}))
    assert XL.unit_lock(run, "u2")                                       # owner process gone: taken over
    XL.unit_unlock(run, "u2")
    lock.write_text(json.dumps({"pid": 2147483647, "host": "another-host", "t": 0}))
    assert not XL.unit_lock(run, "u2")                                   # another host: never taken over
    # labels: not before the seal
    with pytest.raises(XL.SingleLookError, match="load only after sealing"):
        XL.require_scoring(root)
    with pytest.raises(XL.SingleLookError, match="exactly one sealing row"):
        XL.begin_scoring("lead", run, root=root)
    pdir = run / "predictions"
    pdir.mkdir(parents=True)
    df.to_parquet(pdir / "m__s0.parquet")
    XL.seal_predictions(pdir, root=root)
    with pytest.raises(XL.SingleLookError, match="already sealed"):
        XL.seal_predictions(pdir, root=root)
    with pytest.raises(XL.SingleLookError, match="closed at sealing"):   # inputs and units close at sealing
        XL.require_open_claim(root)
    assert XL.check_sealed(root=root) == pdir
    with pytest.raises(XL.SingleLookError, match="load only after sealing"):   # sealed, scoring not opened
        XL.require_scoring(root)
    with pytest.raises(XL.SingleLookError, match="needs a scoring run that was started"):
        XL.begin_scoring("lead", run, rescore=True, root=root)
    with pytest.raises(XL.SingleLookError, match="claimed, unscored look of this run directory"):
        XL.begin_scoring("lead", tmp_path / "other", root=root)
    # single-shot scoring
    assert XL.begin_scoring("lead", run, root=root) == pdir
    assert XL.require_scoring(root) == pdir
    with pytest.raises(XL.SingleLookError, match="never resumed"):
        XL.begin_scoring("lead", run, root=root)
    with monkeypatch.context() as m:
        m.setattr(XL.os, "getpid", lambda: -1)
        with pytest.raises(XL.SingleLookError, match="held by another process"):
            XL.require_scoring(root)
    XL.begin_scoring("lead", run, rescore=True, root=root)               # starts again from the sealed predictions
    assert (root / "RESCORE_1.json").is_file() and XL.require_scoring(root) == pdir
    pending = tmp_path / "pending.jsonl"
    row = {"workstream": "bioassess", "experiment": XL.EXPERIMENT, "model": "x", "split": "t", "registry_path": pending}
    log_result(metric="look_claimed", value=1.0, **row)
    with pytest.raises(XL.SingleLookError, match="empty or not scoring rows"):
        XL.flush_rows(pending)
    pending.unlink()
    for v in (0.1, 0.2):
        log_result(metric="r2_anomaly", value=v, **row)
    n0 = len(XL.registry_rows())
    assert XL.flush_rows(pending) == 2 and len(XL.scoring_rows()) == 2 and len(XL.registry_rows()) == n0 + 2
    with pytest.raises(XL.SingleLookError, match="nothing is written twice"):
        XL.flush_rows(pending)
    for call in (lambda: XL.check_sealed(root=root), lambda: XL.require_scoring(root),
                 lambda: XL.begin_scoring("lead", run, rescore=True, root=root)):
        with pytest.raises(XL.SingleLookError, match="re-scoring is refused"):
            call()
    assert XL.check_sealed(root=root, allow_scored=True) == pdir         # verification only; opens no loader
    XL.mark_scored({"x": 1}, root=root)
    with pytest.raises(XL.SingleLookError, match="already scored"):
        XL.check_sealed(root=root)
    with pytest.raises(XL.SingleLookError, match="open, unscored"):
        XL.require_open_claim(root)
    with pytest.raises(FileExistsError):
        XL.mark_scored({"x": 2}, root=root)


def test_x_testlook_hardening_registry_seal_rescore_access_unlisted(tmp_path, monkeypatch):
    from sentinel.core.registry import log_result

    XL = _guard(tmp_path, monkeypatch)
    _write_freeze(XL, tmp_path)
    fr, root, pdir = XL.load_freeze(require_running=False), tmp_path / "look", tmp_path / "run" / "predictions"
    XL.claim(fr, "lead", tmp_path / "run", root=root)
    XL.require_open_claim(root)
    pdir.mkdir(parents=True)
    pd.DataFrame({"a": [1.0]}).to_parquet(pdir / "m__s0.parquet")
    XL.seal_predictions(pdir, root=root)
    assert len(XL.sealed_rows()) == 1 and "m__s0.parquet" in XL.sealed_rows()[0]["extra"]["files"]
    with pytest.raises(XL.SingleLookError, match="closed at sealing"):        # access closes at sealing
        XL.require_open_claim(root)
    (root / "PREDICTIONS_SAVED.json").chmod(0o644)
    (root / "PREDICTIONS_SAVED.json").unlink()                                 # deleting the marker does not reopen
    with pytest.raises(XL.SingleLookError, match="closed at sealing"):
        XL.require_open_claim(root)
    with pytest.raises(XL.SingleLookError, match="PREDICTIONS_SAVED.json missing"):
        XL.check_sealed(root=root)
    assert XL.seal_predictions(pdir, root=root)["completed_after_crash"] is True   # same sealing, completed
    assert len(XL.sealed_rows()) == 1 and XL.check_sealed(root=root) == pdir
    (pdir / "extra.parquet").write_bytes(b"x")                                 # unlisted file
    with pytest.raises(XL.SingleLookError, match="unlisted"):
        XL.check_sealed(root=root)
    (pdir / "extra.parquet").unlink()
    log_result(workstream="bioassess", experiment=XL.EXPERIMENT, model="x", metric="r2_anomaly", value=0.1, split="t",
               registry_path=XL.REGISTRY)
    with pytest.raises(XL.SingleLookError, match="re-scoring is refused"):    # any scoring row blocks rescore
        XL.check_sealed(root=root)
    with pytest.raises(XL.SingleLookError, match="closed at sealing"):        # even with SCORED.json absent
        XL.require_open_claim(root)


def test_x_testlook_claim_is_not_tied_to_a_process_and_an_abort_is_recorded(tmp_path, monkeypatch):
    XL = _guard(tmp_path, monkeypatch)
    _write_freeze(XL, tmp_path)
    root, run = tmp_path / "look", tmp_path / "run"
    XL.claim(XL.load_freeze(require_running=False), "lead", run, root=root)
    with monkeypatch.context() as m:
        m.setattr(XL.os, "getpid", lambda: 4242)                   # any chunk process works under the open claim
        XL.require_open_claim(root, run_dir=run)
    XL.record_abort(KeyboardInterrupt("ctrl-c"), root=root)
    assert list(root.glob("ABORTED_*.json")) and XL.registry_rows()[-1]["metric"] == "look_aborted"
    assert XL.scoring_rows() == []                                 # an abort is not a scoring row
    XL.require_open_claim(root, run_dir=run)                       # ... and neither seals nor closes the claim


def test_x_testlook_artifacts_verified(tmp_path, monkeypatch):
    XL = _guard(tmp_path, monkeypatch)
    f = tmp_path / "a.parquet"
    f.write_bytes(b"1")
    rec = {"artifacts": {str(f): XL.sha_file(f)}}
    XL.verify_artifacts(rec)
    f.write_bytes(b"2")
    with pytest.raises(XL.SingleLookError, match="artifacts changed"):
        XL.verify_artifacts(rec)
    with pytest.raises(XL.SingleLookError, match="no artifacts"):
        XL.verify_artifacts({})


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


def test_x_test_inputs_need_an_open_claim_and_labels_need_the_scoring_phase(tmp_path):
    from sentinel.models.bioassess import xdev as XD
    from sentinel.models.bioassess import xtestlook as XL

    kw = {"root": tmp_path / "nolook", "registry_path": tmp_path / "registry.jsonl"}
    with pytest.raises(XL.SingleLookError, match="open, unscored"):
        XD.load_test_inputs(pd.DataFrame(), "X-L", **kw)
    with pytest.raises(XL.SingleLookError, match="only after sealing"):
        XD.load_test_labels(pd.DataFrame(), "X-L", **kw)
    assert not hasattr(XD, "load_test_rows")          # v11: no loader returns test inputs and labels together


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

    files = XL.scoring_code_files()
    for f in ("sentinel/eval/events.py", "sentinel/eval/hazards.py", "sentinel/eval/ew_scoring.py",
              "sentinel/eval/early_warning.py", "sentinel/events/bioassess_x.py", "sentinel/models/bioassess/cv.py",
              "scripts/eval/bioassess_x_gjam.R"):
        assert f in files, f
    assert len(XL.scoring_code_sha256()) == 64


def test_x_scoring_closure_covers_every_module_the_event_path_loads():
    """Run the event path (and import the scorer) in a fresh interpreter; every sentinel module it loads must be in
    the hashed scoring-code closure (M7-review lesson 1)."""
    import subprocess
    import sys

    from sentinel.models.bioassess import xtestlook as XL

    code = r"""
import importlib.util, json, sys
import numpy as np
from sentinel.events import bioassess_x as XE
rng = np.random.default_rng(0)
mu = np.log10(rng.uniform(5, 600, 200)); y = mu + rng.normal(0, .2, 200)
ye = XE.labels(y, "X-L", "E2a"); p = XE.derived_probability(mu, 0.2, "X-L", "E2a")
pc = XE.frozen_calibrated(p, ye, p); XE.crossfit_calibrated(p, ye, np.arange(200) % 4)
cl = np.array([f"c{i % 40}" for i in range(200)])
XE.metrics(ye, pc, cl, p_ref=np.full(200, np.nanmean(ye)), n_boot=20); XE.compare(ye, pc, np.full(200, .1), cl)
XE.power_flag(ye, cl)
spec = importlib.util.spec_from_file_location("xf", "scripts/eval/bioassess_x_final.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import sentinel.models.bioassess.xtestlook, sentinel.models.bioassess.xbioset
print(json.dumps(sorted({v.__file__ for k, v in sys.modules.items() if k.startswith("sentinel") and getattr(v, "__file__", None)})))
"""
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=600,
                         env={**__import__("os").environ, "OMP_NUM_THREADS": "1"})
    assert out.returncode == 0, out.stderr[-2000:]
    loaded = {str(Path(f).resolve().relative_to(REPO)) for f in json.loads(out.stdout.strip().splitlines()[-1])}
    missing = loaded - set(XL.scoring_code_files())
    assert not missing, f"modules loaded by the event/scoring path but not hashed: {sorted(missing)}"



def test_x_event_derived_probability_monotone_with_shared_module():
    pytest.importorskip("sentinel.eval.events")
    from sentinel.events import bioassess_x as XE

    mu = np.log10(np.array([50.0, 230.0, 1000.0]))
    p = XE.derived_probability(mu, 0.2, "X-L", "E2a")
    assert np.all(np.diff(p) > 0) and abs(p[1] - 0.5) < 0.05
    assert XE.labels(np.log10([100.0, 300.0]), "X-L", "E2a").tolist() == [0.0, 1.0]
    assert XE.labels(np.log10([230.0, 20.0]), "X-L", "E2a").tolist() == [0.0, 0.0]     # boundary: not > 230
    assert XE.labels([2.0], "X-L", "E5b").tolist() == [0.0]                              # 100 µg/L = 0.1, not > 0.1


def test_x_final_seal_write_once_and_unit_tamper(tmp_path):
    F = _final()
    from sentinel.models.bioassess import xtestlook as XL

    assert XL.unit_write(tmp_path, "k", pd.DataFrame({"a": [1.0]}), {})
    assert XL.unit_read(tmp_path, "k")[0]["a"].iloc[0] == 1.0
    f = XL.unit_dir(tmp_path, "k") / "pred.parquet"
    f.chmod(0o644)
    pd.DataFrame({"a": [2.0]}).to_parquet(f)
    with pytest.raises(XL.SingleLookError, match="modified or lost"):
        XL.unit_read(tmp_path, "k")
    XL.unit_dir(tmp_path, "k").chmod(0o755)
    f.unlink()
    with pytest.raises(XL.SingleLookError, match="modified or lost"):
        XL.unit_read(tmp_path, "k")
    assert XL.unit_write(tmp_path, "k2", pd.DataFrame({"a": [1.0]}), {})
    os.rename(XL.unit_dir(tmp_path, "k2"), XL.unit_dir(tmp_path, "k3"))
    with pytest.raises(XL.SingleLookError, match="sidecar belongs to another unit"):
        XL.unit_read(tmp_path, "k3")
    out = tmp_path / "o"
    sha = F.write_once(out / "a.json", "{}")
    assert sha == hashlib.sha256(b"{}").hexdigest() and stat.S_IMODE((out / "a.json").stat().st_mode) == 0o444
    assert F.write_once(out / "b.bin", b"\x00") == hashlib.sha256(b"\x00").hexdigest()
    assert sorted(p.name for p in out.iterdir()) == ["a.json", "b.bin"]          # no temporary file is left
    F.seal(out)
    assert (out / "SHA256SUMS").read_text().splitlines() == [f"{sha}  a.json", f"{hashlib.sha256(bytes(1)).hexdigest()}  b.bin"]
    assert not (out / "SHA256SUMS").stat().st_mode & stat.S_IWUSR
    F.verify_sealed_results(out)
    (out / "a.json").chmod(0o644)
    (out / "a.json").write_text("{ }")
    with pytest.raises(XL.SingleLookError, match="sealed result file a.json changed"):
        F.verify_sealed_results(out)


def _toy_rec(F):
    """A small record: C-X = mean(BioSet, RF[community+geography], TabPFN); five single baselines listed; the two
    rule comparators derived from it (rule (ii) adds ``gbm_community``). 16 fits per fold, 6 of them GPU-kind."""
    import copy

    from sentinel.models.bioassess import xdev as XD

    dev = {"rf_geography": 0.1, "rf_community_geography": 0.3, "tabpfn_community": 0.32, "gbm_community": 0.31,
           "knn_community": 0.2}
    rec = {"seeds": [0, 1, 2], "n_boot": 20, "n_boot_confirmatory": 60, "alpha": 0.05, "n_configurations_tried": 1,
           "selection_note": "toy", "reproducibility_tolerance": dict(F.TOLERANCE), "event_track": {"status": "off"},
           "not_compared": copy.deepcopy(_NOT_COMPARED), "candidates": {}, "comparators": {}, "comparator_set": {}}
    for b in XD.BLOCKS:
        cs = {n: {"spec": _B(n), "class": "c", "dev_mean_r2_anomaly": v} for n, v in dev.items()}
        rec["comparator_set"][b] = cs
        rec["candidates"][b] = {"confirmatory": {"name": "ens_toy", "spec": {"kind": "mean", "members": [
            dict(_BS), _B("rf_community_geography"), _B("tabpfn_community")]}},
                                "secondary": [{"name": "bsx_flat_geo", "spec": dict(_BS)}]}
        rec["comparators"][b] = {t: {"name": "tabpfn_community", "spec": _B("tabpfn_community")} for t in XD.TARGETS[b]}
        for n, sp in F.rule_comparators(rec, b)[0].items():
            cs[n] = {"spec": sp, "class": "rule"}
    return rec


def test_x_final_plan_has_one_unit_per_block_leaf_fold_and_seed():
    import copy

    F = _final()
    from sentinel.models.bioassess.xdev import Fold

    rec = _toy_rec(F)
    folds = {b: [Fold(b, f"g{k}", pd.Index([]), pd.Index([])) for k in range(2)] for b in ("X-L", "X-U")}
    units = F.plan_units(rec, folds)
    keys = [u["key"] for u in units]
    assert len(keys) == len(set(keys)) == 2 * 2 * 16
    assert all(set(u) == {"key", "kind", "block", "spec", "k", "g", "seed", "gpu"} for u in units)
    assert all(u["key"] == f"leaf_{u['block']}_{F.spec_id(u['spec'])}_f{u['k']:02d}_s{u['seed']}" for u in units)
    gpu = [u for u in units if u["gpu"]]
    assert len(gpu) == 2 * 2 * 3 * 2                  # blocks x folds x seeds x {bioset, tabpfn}; bioset deduplicated
    assert all(F.is_gpu_spec(u["spec"]) for u in gpu)
    assert not F.is_gpu_spec(_B("rf_community")) and not F.is_gpu_spec({"kind": "r", "method": "bioinfer"})
    knn = [u for u in units if u["spec"].get("name") == "knn_community"]
    assert len(knn) == 4 and {u["seed"] for u in knn} == {0}                    # deterministic model: one fit
    r2 = copy.deepcopy(rec)                                                       # the rule comparators add no fit
    for b in ("X-L", "X-U"):
        for n in F.RULE_COMPARATORS:
            r2["comparator_set"][b].pop(n)
    assert [u["key"] for u in F.plan_units(r2, folds)] == keys
    parts = [F.select_units(units, None, f"{i}/3") for i in range(3)]             # shards partition the plan
    assert sorted(u["key"] for p in parts for u in p) == sorted(keys)
    assert [u["key"] for u in F.select_units(units, "gpu", None)] == [u["key"] for u in gpu]
    assert len(F.select_units(units, "cpu", "1/2")) == (64 - 24) // 2
    with pytest.raises(SystemExit):
        F.select_units(units, None, "3/3")
    plan = F.make_plan(rec, folds, dry=True, max_folds=None, cap=None)
    assert plan["dry"] is True and plan["units"] == units and [len(plan["folds"][b]) for b in ("X-L", "X-U")] == [2, 2]


class _ToyLook:
    """A whole look on toy tables (two blocks, three regions, ``n_test`` predicted rows per block) with a fake leaf
    fit. ``dry=False`` is the REAL code path (label-free test bundles, label loader) on toy data; the look directory
    and the registry are always inside ``tmp`` (never the repository's). ``patch(obj, name, value)`` is
    ``monkeypatch.setattr`` in a test and ``setattr`` in the snapshot subprocess."""

    def __init__(self, tmp, patch, *, dry=False, name="run", rec=None, record_sha="toy-record", n_train=24, n_test=30,
                 keep_freeze=False, cap=None):
        from sentinel.core import registry
        from sentinel.models.bioassess import xdev as XD
        from sentinel.models.bioassess import xtestlook as XL
        from sentinel.models.bioassess.data import BioDesign

        self.F = F = _final()
        self.XD, self.XL, self.dry, self.cap = XD, XL, dry, cap
        self.run_dir = Path(tmp) / name
        self.root, self.reg = self.run_dir / "look", self.run_dir / "registry_dryrun.jsonl"
        self.record_sha = record_sha
        self.rec = {"_path": "toy", **(rec or _toy_rec(F)), "_sha256": record_sha}
        self.fits, self.fail = [], None
        rng = np.random.default_rng(7)
        cols = sorted(set(XD.TARGETS["X-L"]) | set(XD.TARGETS["X-U"]) | set(XD.SHARED.values()))

        def table(src, tag, n):
            ids = pd.Index([f"{src}:{tag}{i}" for i in range(n)], name="visit_id")
            lat = rng.uniform(45, 55, n)
            s = pd.DataFrame({"source": src, "ag_eco9": np.repeat(("A", "B", "C"), n // 3), "lat": lat,
                              "lon": rng.uniform(-5, 5, n), "location_id": [f"{src}{tag}L{i // 2}" for i in range(n)],
                              "analysis_method": "ANLA" if src == "uk" else None}, index=ids)
            y = pd.DataFrame({c: (lat - 50) + rng.normal(0, 0.3, n) for c in cols}, index=ids)
            return s, pd.DataFrame({"visit_id": list(ids), "name": "t"}), y

        self.train = {b: table(src, "tr", n_train) for b, src in (("X-L", "nla"), ("X-U", "uk"))}
        self.test = {b: table(src, "te", n_test) for b, src in (("X-L", "nla"), ("X-U", "uk"))}
        parts = [self.train[b] for b in XD.BLOCKS] + ([self.test[b] for b in XD.BLOCKS] if dry else [])
        self.D = BioDesign(pd.concat([p[0] for p in parts]), pd.concat([p[1] for p in parts], ignore_index=True),
                           pd.concat([p[2] for p in parts]), {})
        rows = []
        for b in XD.BLOCKS:
            for tag, (s, _, _) in (("tr", self.train[b]), ("te", self.test[b])):
                role = "test" if tag == "te" and not dry else "train_pool"
                rows += [{"block": b, "role": role, "fold": g, "source": i.split(":")[0], "id": i.split(":")[1],
                          "cluster": c} for i, g, c in zip(s.index, s["ag_eco9"], s["location_id"])]
        self.S, self.frozen = pd.DataFrame(rows), {}
        self.folds = {b: [XD.Fold(b, g, self.train[b][0].index,
                                  self.test[b][0].index[(self.test[b][0]["ag_eco9"] == g).to_numpy()])
                          for g in "ABC"] for b in XD.BLOCKS}
        look = self

        def leaf(fitter, spec, block, k, fo, s_eff):
            D, T = fitter.designs[block], list(XD.TARGETS[block])
            assert not np.isfinite(D.Y.reindex(fo.test).to_numpy(float)).any()   # no label of a predicted row
            if look.fail is not None and look.fail(spec, block, k, s_eff):
                raise RuntimeError("toy fit failed")
            look.fits.append((F.spec_id(spec), block, k, s_eff))
            h = int(hashlib.sha256(f"{F.spec_id(spec)}|{block}|{k}|{s_eff}".encode()).hexdigest()[:8], 16)
            q = 0.9 if spec["kind"] == "bioset" else 0.4 + 0.1 * (h % 5)
            base = float(np.nanmean(D.Y.loc[fo.train, T].to_numpy(float)))
            lat, r = D.samples.loc[fo.test, "lat"].to_numpy(float), np.random.default_rng(h)
            return pd.DataFrame({t: base + q * (lat - 50) + r.normal(0, 0.05, len(lat)) for t in T}, index=fo.test)

        patch(F.Fitter, "leaf", leaf)
        patch(F, "paired_many", lambda tasks, n_jobs=1: [F._paired(*t) for t in tasks])
        patch(XD, "read_folds", lambda f, S, frozen: {b: list(v) for b, v in self.folds.items()})
        patch(XD, "test_folds", lambda S, frozen, D, Dt, b: [dataclasses.replace(fo, design="test")
                                                             for fo in self.folds[b]])
        patch(XD, "_load_test", lambda S, b: (self.test[b][0].copy(), self.test[b][1].copy(), self.test[b][2].copy(),
                                              n_test))
        patch(XD, "load_context", lambda rebuild=False: (self.S, self.frozen, self.D))
        patch(XD, "load_split", lambda: self.S)
        patch(registry, "code_state", lambda: {"toy": True})
        if not keep_freeze:
            patch(XL, "FREEZE_FILE", Path(tmp) / "no_freeze_file.json")

    def kw(self):
        return {"dry": self.dry, "root": self.root, "reg": self.reg, "record_sha": self.record_sha}

    def claim(self, max_folds=None):
        self.run_dir.mkdir(parents=True)
        if self.dry:
            (self.run_dir / "DRY_RUN").write_text("toy\n")
        self.XL.claim(self.rec, "toy", self.run_dir, root=self.root, registry_path=self.reg)
        return self.F.write_inputs(self.rec, self.run_dir, self.S, self.frozen, self.D, max_folds=max_folds,
                                   cap=self.cap, code="toy", **self.kw())

    def chunk(self, only=None, shard=None, **kw):
        return self.F.run_chunk(self.rec, self.run_dir, self.S, self.frozen, self.D, only=only, shard=shard,
                                **self.kw(), **kw)

    def seal(self):
        return self.F.do_seal(self.rec, self.run_dir, **self.kw())

    def score(self, rescore=False):
        pred = self.XL.begin_scoring("toy", self.run_dir, rescore=rescore, root=self.root, registry_path=self.reg)
        return self.F.do_score(self.rec, self.run_dir, pred, code="toy", authorised_by="toy", rescore=rescore,
                               **self.kw())

    def rows(self):
        return self.XL.registry_rows(registry_path=self.reg)

    def unit_shas(self):
        d = self.run_dir / "units"
        return {p.name: json.loads((p / "unit.json").read_text())["sha256"]
                for p in sorted(d.iterdir()) if not p.name.startswith(".")} if d.is_dir() else {}


# ---------------------------------------------------------------- the five points of the look (protocol v11 item 2)
def test_x_look_point1_prediction_phase_is_label_free(tmp_path, monkeypatch):
    """(1) After the claim only test INPUTS are loaded; no fitting/predicting process holds a finite label of a
    predicted row; the label loader and the truth files are refused until the scoring command opens after sealing."""
    L = _ToyLook(tmp_path, monkeypatch.setattr)
    F, XD, XL = L.F, L.XD, L.XL
    lk = {"root": L.root, "registry_path": L.reg}
    with pytest.raises(XL.SingleLookError, match="open, unscored"):              # no claim, no test input
        XD.load_test_inputs(L.S, "X-L", **lk)
    info = L.claim()
    assert info["n_units"] == 96
    assert info["test_rows_loaded"] == {b: {"n_test_rows_loaded": 30, "n_test_rows_split": 30} for b in XD.BLOCKS}
    for b in XD.BLOCKS:
        f = L.run_dir / "inputs" / f"{b}.pkl"
        assert stat.S_IMODE(f.stat().st_mode) == 0o444 and F.file_sha256(f) == info["inputs"][f.name]
        with open(f, "rb") as fh:
            Dt = pickle.load(fh)
        assert len(Dt.samples) == 30 and Dt.Y.shape[0] == 30 and Dt.Y.isna().all().all()   # stored without labels
    designs, folds, _ = F.look_design(L.run_dir, info["inputs"], L.S, L.frozen, L.D, dry=False, max_folds=None, cap=None)
    for b in XD.BLOCKS:
        te = pd.Index([i for fo in folds[b] for i in fo.test])
        assert len(te) == 30 and designs[b].Y.loc[te].isna().all().all()
        assert np.isfinite(designs[b].Y.loc[folds[b][0].train, list(XD.TARGETS[b])].to_numpy(float)).all()
    F.assert_label_free(designs, folds)
    leak = designs["X-U"].Y.copy()
    leak.loc[folds["X-U"][1].test[0], "uk_ph"] = 7.0
    with pytest.raises(XL.SingleLookError, match="X-U/B: a predicted row carries a finite label"):
        F.assert_label_free({**designs, "X-U": dataclasses.replace(designs["X-U"], Y=leak)}, folds)
    fo = folds["X-L"][0]
    with pytest.raises(XL.SingleLookError, match="X-L/A: training and predicted rows overlap"):
        F.assert_label_free(designs, {**folds, "X-L": [dataclasses.replace(fo, train=fo.train.append(fo.test[:1]))]})

    def labels_refused(msg):
        with pytest.raises(XL.SingleLookError, match=msg):
            XD.load_test_labels(L.S, "X-L", **lk)
        with pytest.raises(XL.SingleLookError, match=msg):
            F.write_truth(L.rec, L.run_dir, L.run_dir / "predictions", L.run_dir / "results" / "truth", dry=False,
                          root=L.root, reg=L.reg)
        assert not (L.run_dir / "results").exists()

    labels_refused("only after sealing")
    assert L.chunk() == 0 and len(L.fits) == 96            # the fake fit asserts label-freeness in every unit
    labels_refused("only after sealing")
    L.seal()
    labels_refused("only after sealing")                  # sealed, scoring phase not opened
    with pytest.raises(XL.SingleLookError, match="closed at sealing"):
        XD.load_test_inputs(L.S, "X-L", **lk)
    XL.begin_scoring("toy", L.run_dir, **lk)
    Y = XD.load_test_labels(L.S, "X-U", **lk)
    assert list(Y.columns) == list(XD.TARGETS["X-U"]) and len(Y) == 30 and np.isfinite(Y.to_numpy(float)).all()


def test_x_look_point1_inputs_that_carry_a_label_are_refused(tmp_path, monkeypatch):
    L = _ToyLook(tmp_path, monkeypatch.setattr)
    real = L.XD.load_test_inputs

    def leaky(S, b, **kw):
        Dt = real(S, b, **kw)
        Dt.Y.iloc[0, 0] = 1.0
        return Dt

    monkeypatch.setattr(L.XD, "load_test_inputs", leaky)
    with pytest.raises(L.XL.SingleLookError, match="test inputs carry labels"):
        L.claim()
    assert not (L.root / "INPUTS.json").exists() and not (L.run_dir / "plan.json").exists()
    assert not (L.run_dir / "inputs").exists()
    with pytest.raises(L.XL.SingleLookError, match="INPUTS.json missing"):
        L.chunk()
    assert L.fits == []


def test_x_look_point1_dry_look_masks_the_pseudo_test_labels(tmp_path, monkeypatch):
    L = _ToyLook(tmp_path, monkeypatch.setattr, dry=True)
    F, XD = L.F, L.XD
    info = L.claim()
    assert info["test_rows_loaded"] == {} and info["n_units"] == 96
    assert F.look_of(L.run_dir) == (True, L.root, L.reg) and F.look_of(tmp_path) == (False, None, None)
    for b in XD.BLOCKS:
        j = json.loads((L.run_dir / "inputs" / f"{b}.json").read_text())
        assert set(j) == {"block", "test_ids"} and j["test_ids"] == sorted(L.test[b][0].index)   # ids only
    designs, folds, _ = F.look_design(L.run_dir, info["inputs"], L.S, L.frozen, L.D, dry=True, max_folds=None, cap=None)
    for b in XD.BLOCKS:
        te = L.test[b][0].index
        assert designs[b].Y.loc[te].isna().all().all()                 # masked in the copy the fits see ...
        assert np.isfinite(L.D.Y.loc[te].to_numpy(float)).all()        # ... not in the development design
        assert designs[b].samples is L.D.samples and len(folds[b]) == 3
    assert L.chunk() == 0 and len(L.fits) == 96


def test_x_look_point2_units_are_write_once_any_order_and_resumable(tmp_path, monkeypatch):
    """(2) One write-once unit per (block, leaf model, fold, seed), each with its own file and hash; chunks run in
    any order and are resumed any number of times; every chunk re-verifies record, plan and inputs; a failed chunk
    does not burn the look."""
    import copy

    ref = _ToyLook(tmp_path, monkeypatch.setattr, name="ref")
    ref.claim()
    assert ref.chunk() == 0
    want = ref.unit_shas()
    assert len(want) == 96 and len(set(ref.fits)) == 96
    L = _ToyLook(tmp_path, monkeypatch.setattr)
    F, XL = L.F, L.XL
    L.claim()
    pf = L.run_dir / "plan.json"
    plan_sha = F.file_sha256(pf)
    keys = [u["key"] for u in json.loads(pf.read_text())["units"]]
    assert sorted(keys) == sorted(want)
    assert L.chunk(max_minutes=0) == 3 and L.unit_shas() == {}                 # time limit: stops, nothing lost
    assert XL.unit_lock(L.run_dir, keys[0])                                    # a unit another chunk is fitting
    assert L.chunk(shard="0/96") == 5 and L.unit_shas() == {}
    XL.unit_unlock(L.run_dir, keys[0])
    assert L.chunk(shard="2/3") == 0 and len(L.fits) == 32                     # any order
    with pytest.raises(XL.SingleLookError, match="64 of 96 planned units do not exist yet"):
        L.seal()
    assert L.chunk(only="cpu", shard="1/2") == 0
    done = L.unit_shas()
    mtime = {k: (XL.unit_dir(L.run_dir, k) / "pred.parquet").stat().st_mtime_ns for k in done}
    assert all(want[k] == v for k, v in done.items())
    # a failing unit: nothing is written for it, the chunk goes on and ends 6, its finished units stay, no lock is
    # left, the claim stays open, nothing is recorded as a look
    L.fail = lambda spec, block, k, s: spec["kind"] == "bioset" and block == "X-U" and k == 1
    assert L.chunk() == 6
    after = L.unit_shas()
    bad = [u["key"] for u in json.loads(pf.read_text())["units"]
           if u["block"] == "X-U" and u["k"] == 1 and u["spec"].get("kind") == "bioset"]
    bad = [k for k in bad if k not in done]                         # the ones this chunk had to fit
    assert bad and not any(k in after for k in bad)                 # the failing units carry no unit record
    chunks = [json.loads(p.read_text()) for p in (L.run_dir / "chunks").glob("*.json")]
    last = max(chunks, key=lambda c: c["started"])
    assert sorted(x["key"] for x in last["failed"]) == sorted(bad)  # named in the chunk record
    assert all("toy fit failed" in x["error"] for x in last["failed"])
    assert len(done) < len(after) == 96 - len(bad) and all(after[k] == v for k, v in done.items())
    assert not list((L.run_dir / "locks").glob("*.lock")) and not list((L.run_dir / "units").glob(".tmp_*"))
    XL.require_open_claim(L.root, L.reg, run_dir=L.run_dir, freeze_sha256=L.record_sha)
    assert not list(L.root.glob("ABORTED_*")) and [r["metric"] for r in L.rows()] == ["look_claimed"]
    with pytest.raises(XL.SingleLookError, match="planned units do not exist yet"):
        L.seal()
    # resumed: only the missing units are fitted; every unit equals the uninterrupted look's; none is rewritten
    L.fail, n = None, len(L.fits)
    assert L.chunk() == 0 and len(L.fits) - n == 96 - len(after)
    assert L.unit_shas() == want and len(L.fits) == len(set(L.fits)) == 96      # no unit was fitted twice
    assert all((XL.unit_dir(L.run_dir, k) / "pred.parquet").stat().st_mtime_ns == v for k, v in mtime.items())
    assert L.chunk() == 0 and L.chunk(only="gpu", shard="0/2") == 0 and len(L.fits) == 96
    assert F.file_sha256(pf) == plan_sha
    # every chunk re-verifies the freeze record, the plan and the inputs
    with pytest.raises(XL.SingleLookError, match="different freeze record"):
        F.run_chunk(L.rec, L.run_dir, L.S, L.frozen, L.D, only=None, **{**L.kw(), "record_sha": "other"})
    other = copy.deepcopy(L.rec)
    other["seeds"] = [0, 1]
    with pytest.raises(XL.SingleLookError, match="differs from the plan of the claim"):
        F.run_chunk(other, L.run_dir, L.S, L.frozen, L.D, only=None, **L.kw())
    inp = L.run_dir / "inputs" / "X-U.pkl"
    raw = inp.read_bytes()
    inp.chmod(0o644)
    inp.write_bytes(raw + b"x")
    with pytest.raises(XL.SingleLookError, match="inputs of X-U changed or missing"):
        L.chunk()
    inp.write_bytes(raw)
    rawp = pf.read_bytes()
    pf.chmod(0o644)
    pf.write_bytes(rawp.replace(b'"seed": 2', b'"seed": 5', 1))
    with pytest.raises(XL.SingleLookError, match="unit plan of this look changed"):
        L.chunk()
    pf.write_bytes(rawp)
    assert L.chunk() == 0 and L.unit_shas() == want and len(L.fits) == 96
    assert L.seal()["n_files"] == 57


def test_x_look_point2_chunks_run_in_parallel_without_fitting_a_unit_twice(tmp_path, monkeypatch):
    import threading

    ref = _ToyLook(tmp_path, monkeypatch.setattr, name="ref")
    ref.claim()
    ref.chunk()
    L = _ToyLook(tmp_path, monkeypatch.setattr)
    L.claim()
    out, lock = [], threading.Lock()

    def work():
        try:
            r = L.chunk()                                   # the same selection in every chunk: they compete
        except BaseException as e:                          # noqa: BLE001
            r = e
        with lock:
            out.append(r)

    th = [threading.Thread(target=work) for _ in range(4)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    assert all(r in (0, 5) for r in out), out               # 5: the rest was being fitted by another chunk
    assert L.chunk() == 0
    assert len(L.fits) == len(set(L.fits)) == 96 and L.unit_shas() == ref.unit_shas()
    assert not list((L.run_dir / "locks").glob("*.lock")) and len(list((L.run_dir / "chunks").glob("*.json"))) == 5


def test_x_look_point3_chunks_print_and_store_identities_counts_and_times_only(tmp_path, monkeypatch, capsys):
    """(3) Between chunks nothing about the predictions is printed or stored outside the write-once units: unit
    identities, row counts and wall time only."""
    L = _ToyLook(tmp_path, monkeypatch.setattr, dry=True)
    L.claim()
    capsys.readouterr()
    assert L.chunk(only="gpu") == 0 and L.chunk(only="cpu") == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    unit = re.compile(r"^\d\d:\d\d:\d\d unit leaf_X-[LU]_[0-9a-f]{16}_f\d\d_s\d \S+ [ABC] rows \d+ \d+\.\ds\s*$")
    done = re.compile(r"^\d\d:\d\d:\d\d chunk done: fitted \d+, failed \d+, left \d+ of \d+ selected, \d+\.\d min$")
    assert sum(bool(unit.match(ln)) for ln in lines) == 96 and sum(bool(done.match(ln)) for ln in lines) == 2
    assert all(unit.match(ln) or done.match(ln) for ln in lines), [ln for ln in lines if not unit.match(ln)][:3]
    audits = sorted((L.run_dir / "chunks").glob("*.json"))
    assert len(audits) == 2
    for f in audits:
        a = json.loads(f.read_text())
        assert set(a) == {"host", "pid", "only", "shard", "device", "gpu", "started", "minutes", "selected", "fitted",
                          "failed", "left", "record_sha256", "code_tree"}
        assert a["failed"] == []
        assert a["fitted"] and all(set(x) == {"key", "wall_s", "written"} for x in a["fitted"])
    for d in (L.run_dir / "units").iterdir():
        assert sorted(p.name for p in d.iterdir()) == ["pred.parquet", "unit.json"]
        assert set(json.loads((d / "unit.json").read_text())) == {
            "unit", "rows", "wall_s", "device", "error", "record_sha256", "key", "sha256", "pid", "host", "written"}
    assert sorted(p.name for p in L.root.iterdir()) == ["INPUTS.json", "TEST_LOOK.json"]
    assert [r["metric"] for r in L.rows()] == ["look_claimed"]
    assert sorted(p.name for p in L.run_dir.iterdir()) == ["DRY_RUN", "chunks", "inputs", "locks", "look", "plan.json",
                                                           "registry_dryrun.jsonl", "units"]
    assert L.F.cmd_status(L.run_dir) == 0
    out = capsys.readouterr().out
    assert "gpu: 36 of 36 units written" in out and "cpu: 60 of 60 units written" in out
    assert "sealed: False" in out and not re.search(r"\d\.\d{3}", out)       # counts and states, no value


def test_x_look_point4_seal_once_when_every_unit_exists(tmp_path, monkeypatch):
    """(4) Sealed once, only when every planned unit exists and verifies; nothing can be fitted or loaded after."""
    L = _ToyLook(tmp_path, monkeypatch.setattr)
    F, XD, XL = L.F, L.XD, L.XL
    L.claim()
    assert L.chunk(only="cpu") == 0
    pred = L.run_dir / "predictions"

    def unsealed():
        return not pred.exists() and XL.sealed_rows(L.reg) == [] and not list(L.run_dir.glob(".predictions_tmp_*")) \
            and not (L.root / "PREDICTIONS_SAVED.json").exists()

    with pytest.raises(XL.SingleLookError, match="36 of 96 planned units do not exist yet"):
        L.seal()
    assert unsealed()
    assert L.chunk(only="gpu") == 0
    f = XL.unit_dir(L.run_dir, min(L.unit_shas())) / "pred.parquet"
    raw = f.read_bytes()
    f.chmod(0o644)
    (pd.read_parquet(f) + 1).to_parquet(f)
    with pytest.raises(XL.SingleLookError, match="modified or lost"):          # a changed unit is never sealed
        L.seal()
    assert unsealed()
    f.write_bytes(raw)
    pred.mkdir()
    (pred / "leftover.parquet").write_bytes(b"x")                              # of a stopped seal command: rebuilt
    s = L.seal()
    assert s["n_files"] == 57 and len(XL.sealed_rows(L.reg)) == 1 and (L.root / "PREDICTIONS_SAVED.json").is_file()
    assert XL.check_sealed(root=L.root, registry_path=L.reg) == pred and not (pred / "leftover.parquet").exists()
    with pytest.raises(XL.SingleLookError, match="already sealed"):
        L.seal()
    for call in (L.chunk, lambda: XD.load_test_inputs(L.S, "X-L", root=L.root, registry_path=L.reg),
                 lambda: F.write_inputs(L.rec, L.run_dir, L.S, L.frozen, L.D, max_folds=None, cap=None, code="toy",
                                        **L.kw())):
        with pytest.raises(XL.SingleLookError, match="closed at sealing"):
            call()
    assert len(L.fits) == 96
    for b in XD.BLOCKS:
        def g(n, s=0, b=b):
            return pd.read_parquet(pred / b / f"{n}__s{s}.parquet")

        rf, tp, gb, bs = g("rf_community_geography"), g("tabpfn_community"), g("gbm_community"), g("bsx_flat_geo")
        assert len(rf) == 30 and set(rf.index) == set(L.test[b][0].index) and list(rf.columns) == list(XD.TARGETS[b])
        assert np.isfinite(rf.to_numpy(float)).all()
        assert np.allclose(g("cxr_nobioset"), (rf + tp) / 2, rtol=0, atol=1e-12)        # protocol v11 item 1 (i)
        assert np.allclose(g("cxr_swap"), (rf + tp + gb) / 3, rtol=0, atol=1e-12)       # ... and (ii)
        assert np.allclose(g("ens_toy"), (bs + rf + tp) / 3, rtol=0, atol=1e-12)
        assert g("knn_community").equals(g("knn_community", 2)) and not g("rf_geography").equals(g("rf_geography", 2))
    idx = json.loads((pred / "units_index.json").read_text())
    assert {k: v["sha256"] for k, v in idx.items()} == L.unit_shas()
    m = L.root / "PREDICTIONS_SAVED.json"                                      # crash between row and marker
    m.chmod(0o644)
    m.unlink()
    assert L.seal()["completed_after_crash"] is True and len(XL.sealed_rows(L.reg)) == 1


def test_x_look_point5_scoring_is_single_shot_with_one_registry_append(tmp_path, monkeypatch):
    """(5) The label loader refuses unless the seal verifies and this process opened the scoring phase; scoring is
    never resumed; every registry row goes in one append."""
    L = _ToyLook(tmp_path, monkeypatch.setattr)
    F, XD, XL = L.F, L.XD, L.XL
    lk = {"root": L.root, "registry_path": L.reg}
    L.claim()
    L.chunk()
    with pytest.raises(XL.SingleLookError, match="exactly one sealing row"):       # not before the seal
        L.score()
    assert not (L.root / "SCORING_STARTED.json").exists() and not (L.run_dir / "results").exists()
    L.seal()
    pred, res = L.run_dir / "predictions", L.run_dir / "results"
    flushes, real_flush, real_seal = [], XL.flush_rows, F.seal

    def flush(pending, *, registry_path=None):
        flushes.append(Path(pending))
        return real_flush(pending, registry_path=registry_path)

    def stop(d):
        raise RuntimeError("scoring stopped")

    monkeypatch.setattr(XL, "flush_rows", flush)
    monkeypatch.setattr(F, "seal", stop)
    with pytest.raises(RuntimeError, match="scoring stopped"):                    # a scoring run that stops ...
        L.score()
    monkeypatch.setattr(F, "seal", real_seal)
    assert (res / "confirmatory.json").is_file() and XL.scoring_rows(L.reg) == [] and flushes == []
    with pytest.raises(XL.SingleLookError, match="never resumed"):                # ... is never resumed
        L.score()
    with pytest.raises(XL.SingleLookError, match="single-shot"):
        F.do_score(L.rec, L.run_dir, pred, code="toy", authorised_by="toy", rescore=False, **L.kw())
    with monkeypatch.context() as m:
        m.setattr(XL.os, "getpid", lambda: -1)
        with pytest.raises(XL.SingleLookError, match="held by another process"):  # labels: the scoring process only
            XD.load_test_labels(L.S, "X-L", **lk)
    C = L.score(rescore=True)                                                     # everything recomputed
    assert (L.run_dir / "results_aborted_1" / "confirmatory.json").is_file() and (L.root / "RESCORE_1.json").is_file()
    cmp = json.loads((res / "rescore_comparison.json").read_text())
    assert len(cmp) == 1 and cmp[0]["agree"] and cmp[0]["tolerance_abs"] == 1e-9 and cmp[0]["not_comparable"] == []
    assert {"confirmatory.json", "marginal.json", "listed_comparators.json", "truth/X-L.parquet"} <= set(cmp[0]["compared"])
    assert flushes == [res / "registry_rows.jsonl"]                               # ONE append of every scoring row
    pend = [json.loads(ln) for ln in (res / "registry_rows.jsonl").read_text().splitlines()]
    rows = L.rows()
    assert [r["metric"] for r in rows[:2]] == ["look_claimed", "predictions_sealed"] and rows[2:] == pend
    assert XL.scoring_rows(L.reg) == pend
    conf = [r for r in pend if r["metric"] == "delta_r2_anomaly_confirmatory"]
    assert len(C) == len(conf) == 12 and (C["n_listed"] == 7).all() and {r["split"] for r in conf} == {"X-L_test", "X-U_test"}
    assert sum(r["metric"] == "delta_r2_anomaly_listed_comparator" for r in pend) == 12 * 7
    scored = json.loads((L.root / "SCORED.json").read_text())
    assert scored["n_registry_rows"] == len(pend) and scored["n_confirmatory"] == 12
    info = json.loads((res / "pass_info.json").read_text())
    assert info["rescore_from_saved"] is True and info["rehearsal"] is None and info["dry_run"] is False
    assert info["rows_scored"] == {"X-L": 30, "X-U": 30} and info["reproducibility_tolerance"] == F.TOLERANCE
    assert sum(v["n"] for v in info["unit_wall_seconds"].values()) == 96
    F.verify_sealed_results(res)
    T = pd.read_parquet(res / "truth" / "X-L.parquet")
    assert list(T.columns) == [*XD.TARGETS["X-L"], "region", "cluster", "lat", "lon", "analysis_method"]
    assert np.allclose(T[list(XD.TARGETS["X-L"])], L.test["X-L"][2].loc[T.index, list(XD.TARGETS["X-L"])])
    # after the append nothing can be scored, loaded or written again
    for rescore in (False, True):
        with pytest.raises(XL.SingleLookError, match="claimed, unscored look"):
            L.score(rescore=rescore)
    with pytest.raises(XL.SingleLookError, match="already scored"):
        XD.load_test_labels(L.S, "X-L", **lk)
    with pytest.raises(XL.SingleLookError, match="nothing is written twice"):
        real_flush(res / "registry_rows.jsonl", registry_path=L.reg)
    assert XL.scoring_rows(L.reg) == pend and len(flushes) == 1


def test_x_look_point5_rescore_must_reproduce_and_sealed_predictions_must_verify(tmp_path, monkeypatch):
    L = _ToyLook(tmp_path, monkeypatch.setattr)
    F, XD, XL = L.F, L.XD, L.XL
    L.claim()
    L.chunk()
    L.seal()
    real_seal = F.seal
    monkeypatch.setattr(F, "seal", lambda d: (_ for _ in ()).throw(RuntimeError("scoring stopped")))
    with pytest.raises(RuntimeError, match="scoring stopped"):
        L.score()
    monkeypatch.setattr(F, "seal", real_seal)
    res, pred = L.run_dir / "results", L.run_dir / "predictions"
    f = pred / "X-L" / "ens_toy__s0.parquet"                      # a sealed prediction changed after the seal
    raw = f.read_bytes()
    f.chmod(0o644)
    (pd.read_parquet(f) + 1e-6).to_parquet(f)
    with pytest.raises(XL.SingleLookError, match="modified"):
        XD.load_test_labels(L.S, "X-L", root=L.root, registry_path=L.reg)
    with pytest.raises(XL.SingleLookError, match="modified"):
        L.score(rescore=True)
    assert not (L.root / "RESCORE_1.json").exists() and res.is_dir()
    f.write_bytes(raw)
    cj = res / "confirmatory.json"                                 # the stopped run's stored numbers (v11 item 6b)
    rows = json.loads(cj.read_text())
    rows[0]["delta"] += 1e-6
    cj.write_text(json.dumps(rows))
    with pytest.raises(XL.SingleLookError, match="does not reproduce the stopped run's stored numbers"):
        L.score(rescore=True)
    cmp = json.loads((res / "rescore_comparison.json").read_text())
    assert cmp[0]["agree"] is False and any(d.startswith("confirmatory.json[0]/delta") for d in cmp[0]["differences"])
    assert (L.run_dir / "results_aborted_1").is_dir() and res.is_dir()            # both result sets are kept
    assert XL.scoring_rows(L.reg) == [] and not (L.root / "SCORED.json").exists() and not (res / "SHA256SUMS").exists()
    out = []
    F._same({"a": [1.0, float("nan"), "x", True]}, {"a": [1.0 + 1e-12, float("nan"), "x", True]}, 1e-9, "f", out)
    assert out == []
    F._same({"a": [1.0, "x", True, 1]}, {"a": [1.0 + 1e-6, "y", False, 1]}, 1e-9, "f", out)
    assert len(out) == 3 and "numbers differ by more than 1e-09" in out[0]
    F._same({"a": 1}, {"b": 1}, 1e-9, "f", out)
    F._same([1], [1, 2], 1e-9, "f", out)
    assert out[3:] == ["f: keys differ", "f: lengths differ"]


def test_x_look_commands_run_a_whole_dry_look_and_a_stopped_scoring_is_recorded(tmp_path, monkeypatch, capsys):
    """The command layer on a toy development look: claim -> chunk -> seal -> score (stopped, recorded) -> refused
    resume -> rescore-from-saved -> verify-scored. Dry mode only: its look directory and registry are in the run
    directory."""
    import argparse
    import signal

    L = _ToyLook(tmp_path, monkeypatch.setattr, dry=True)
    F, XL = L.F, L.XL
    monkeypatch.setattr(F, "load_record", lambda dry: (L.rec, L.record_sha, "toy") if dry else 1 / 0)
    monkeypatch.setattr(F, "setup", lambda: (L.S, L.frozen, L.D))
    monkeypatch.setattr(F.paths, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    handlers = {s: signal.getsignal(s) for s in (signal.SIGHUP, signal.SIGTERM)}
    ns = argparse.Namespace
    try:
        assert F.cmd_claim(ns(dry_run_dev=True, max_folds=None, rehearsal_cap=None, authorised_by=None)) == 0
        run_dir = Path(capsys.readouterr().out.strip().splitlines()[-1])
        assert run_dir.parent == tmp_path / "runs" / "bioassess" and run_dir.name.startswith("x_final_dryrun_dev_")
        assert F.look_of(run_dir)[0] is True and json.loads((run_dir / "claim_args.json").read_text()) == {
            "max_folds": None, "rehearsal_cap": None}
        chunk = {"chunk": str(run_dir), "only": None, "shard": None, "max_minutes": 100.0, "cpu_smoke": True}
        with pytest.raises(SystemExit, match="needs --only gpu or --only cpu"):
            F.cmd_chunk(ns(**{**chunk, "cpu_smoke": False}))
        with pytest.raises(SystemExit, match="--max-minutes must be in"):
            F.cmd_chunk(ns(**{**chunk, "max_minutes": 101.0}))
        with pytest.raises(SystemExit, match="planned units do not exist yet"):
            F.cmd_seal(run_dir)
        assert F.cmd_chunk(ns(**{**chunk, "shard": "0/2"})) == 0 and F.cmd_chunk(ns(**chunk)) == 0
        with pytest.raises(SystemExit, match="exactly one sealing row"):
            F.cmd_score(run_dir, None, False)
        assert F.cmd_seal(run_dir) == 0
        root, reg = F.look_of(run_dir)[1:]
        real_seal = F.seal
        monkeypatch.setattr(F, "seal", lambda d: (_ for _ in ()).throw(KeyboardInterrupt()))
        with pytest.raises(KeyboardInterrupt):                                   # scoring killed
            F.cmd_score(run_dir, None, False)
        monkeypatch.setattr(F, "seal", real_seal)
        assert list(root.glob("ABORTED_*.json")) and XL.registry_rows(registry_path=reg)[-1]["metric"] == "look_aborted"
        assert XL.scoring_rows(reg) == []
        with pytest.raises(SystemExit, match="never resumed"):
            F.cmd_score(run_dir, None, False)
        with pytest.raises(SystemExit, match="no scoring row exists"):
            F.cmd_verify_scored(run_dir)
        assert F.cmd_score(run_dir, None, True) == 0
        rows = XL.scoring_rows(reg)
        assert rows and {r["split"] for r in rows if r["metric"] == "r2_anomaly"} == {"X-L_dry_run_dev", "X-U_dry_run_dev"}
        assert json.loads((run_dir / "results" / "pass_info.json").read_text())["dry_run"] is True
        assert F.cmd_verify_scored(run_dir) == 0
        ver = sorted(run_dir.glob("results_verify_*"))
        assert len(ver) == 1 and json.loads((ver[0] / "comparison.json").read_text())["agree"] is True
        assert XL.scoring_rows(reg) == rows                                      # verification writes no row
        f = run_dir / "results" / "marginal.json"
        f.chmod(0o644)
        f.write_text(f.read_text() + " ")
        with pytest.raises(SystemExit, match="sealed result file marginal.json changed"):
            F.cmd_verify_scored(run_dir)
        capsys.readouterr()
        assert F.cmd_status(run_dir) == 0
        assert "scoring started: True | scored: True" in capsys.readouterr().out
    finally:
        for s, h in handlers.items():
            signal.signal(s, h)
    # the real-mode record loader was never reached, and the repository's look directory was never touched
    assert not (tmp_path / "runs" / "bioassess" / "x_track").exists()


# ------------------------------------------------------------------- pinned code snapshot (protocol v11 item 3)
def test_x_code_snapshot_is_read_only_content_addressed_and_verified(tmp_path, monkeypatch):
    from sentinel.core import paths
    from sentinel.models.bioassess import xtestlook as XL

    src, snaps = tmp_path / "src", tmp_path / "snaps"
    (src / "pkg").mkdir(parents=True)
    (src / "pkg" / "a.py").write_text("A = 1\n")
    (src / "c.py").write_text("C = 1\n")
    blk = XL.make_snapshot(["pkg/a.py", "c.py"], dest_root=snaps, src_root=src)
    d = Path(blk["dir"])
    assert d == snaps / f"code_{blk['tree_sha256'][:16]}" and blk["live_root"] == str(src)
    assert blk["files"] == {"c.py": XL.sha_file(src / "c.py"), "pkg/a.py": XL.sha_file(src / "pkg" / "a.py")}
    assert stat.S_IMODE(d.stat().st_mode) == 0o555 and stat.S_IMODE((d / "pkg").stat().st_mode) == 0o555
    assert all(stat.S_IMODE(f.stat().st_mode) == 0o444 for f in d.rglob("*") if f.is_file())
    assert json.loads((d / XL.SNAPSHOT_MARKER).read_text())["live_root"] == str(src)
    rec = {"code_snapshot": blk}
    assert XL.verify_snapshot(rec, require_running=False) == d
    with pytest.raises(XL.SingleLookError, match="not running from the pinned code snapshot"):
        XL.verify_snapshot(rec)                                 # the running tree must BE the snapshot
    with monkeypatch.context() as m:
        m.setattr(paths, "REPO_ROOT", d)
        assert XL.verify_snapshot(rec) == d
    (src / "pkg" / "a.py").write_text("A = 2\n")                # the live tree is never looked at
    assert XL.verify_snapshot(rec, require_running=False) == d
    blk2 = XL.make_snapshot(["pkg/a.py", "c.py"], dest_root=snaps, src_root=src)
    assert blk2["dir"] != blk["dir"] and (d / "pkg" / "a.py").read_text() == "A = 1\n"   # new content, new folder
    (src / "pkg" / "a.py").write_text("A = 1\n")
    assert XL.make_snapshot(["c.py", "pkg/a.py", "c.py"], dest_root=snaps, src_root=src) == blk   # reused

    def bad(msg, block=blk):
        with pytest.raises(XL.SingleLookError, match=msg):
            XL.verify_snapshot({"code_snapshot": block}, require_running=False)

    bad("no code_snapshot block", {})
    bad("no code_snapshot block", {**blk, "files": {}})
    bad("does not exist", {**blk, "dir": str(snaps / "code_none")})
    bad("tree hash does not match", {**blk, "tree_sha256": "0" * 64})
    bad("marker changed", {**blk, "marker_sha256": "0" * 64})
    d.chmod(0o755)
    (d / "b.R").write_text("b\n")
    bad(r"file set changed: unlisted \['b.R'\]")
    (d / "b.R").unlink()
    (d / "__pycache__").mkdir()
    (d / "__pycache__" / "c.cpython-311.pyc").write_bytes(b"x")
    bad("file set changed: unlisted")                           # the look runs with python -B
    (d / "__pycache__" / "c.cpython-311.pyc").unlink()
    (d / "__pycache__").rmdir()
    (d / "c.py").unlink()
    bad(r"missing \['c.py'\]")
    (d / "c.py").write_text("C = 1\n")
    XL.verify_snapshot(rec, require_running=False)
    (d / "pkg" / "a.py").chmod(0o644)
    (d / "pkg" / "a.py").write_text("A = 3\n")
    bad(r"code snapshot files changed: \['pkg/a.py'\]")


def _snapshot_driver(run_dir, shard):
    """Runs INSIDE a code snapshot (a subprocess of the test below): one chunk of a toy look under the real freeze
    loader, which verifies the snapshot and never the live tree."""
    from sentinel.core import paths
    from sentinel.models.bioassess import xtestlook as XL

    rec = XL.load_freeze()
    run_dir = Path(run_dir)
    L = _ToyLook(run_dir.parent, setattr, name=run_dir.name, rec=rec, record_sha=rec["_sha256"], keep_freeze=True)
    snap = Path(rec["code_snapshot"]["dir"]).resolve()
    assert L.F.REPO.resolve() == REPO.resolve() == paths.REPO_ROOT.resolve() == snap, (L.F.REPO, REPO, snap)
    assert Path(XL.__file__).resolve().is_relative_to(snap) and XL.LIVE_ROOT != paths.REPO_ROOT
    if XL.open_claim(L.root) is None:
        L.claim()
    rc = L.chunk(shard=shard)
    print("UNITS " + json.dumps({"rc": rc, "fits": len(L.fits), "units": L.unit_shas()}))


def test_x_look_runs_from_the_pinned_snapshot_and_ignores_the_live_tree(tmp_path, monkeypatch):
    """X-B2: the look runs from the read-only snapshot the freeze record names. A live-tree edit neither aborts a
    chunk nor changes its result; any change of the snapshot aborts the chunk before a unit is fitted."""
    import shutil
    import subprocess
    import sys

    from sentinel.models.bioassess import xtestlook as XL

    live, snaps, run_dir = tmp_path / "live", tmp_path / "snaps", tmp_path / "look" / "run"
    files = sorted({*XL.scoring_code_files(), "tests/test_bioassess_x_dev.py"})
    for rel in files:
        (live / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, live / rel)
    blk = XL.make_snapshot(files, dest_root=snaps, src_root=live)
    snap = Path(blk["dir"])
    cand = {**_toy_rec(_final()), "status": "candidate", "scoring_code_sha256": XL.scoring_code_sha256(),
            "data_snapshots": {"toy": 1}, "code_snapshot": blk}
    pre = live / "docs" / "preregistration"
    pre.mkdir(parents=True)
    (pre / "bioassess_x_freeze_candidate.json").write_text(json.dumps(cand))
    (pre / "bioassess_x_freeze.json").write_text(json.dumps({
        **cand, "status": "frozen", "frozen_by": "toy",
        "candidate_sha256": XL.sha_file(pre / "bioassess_x_freeze_candidate.json")}))
    driver = ("import importlib.util, sys\n"
              "spec = importlib.util.spec_from_file_location('m5x_tests', 'tests/test_bioassess_x_dev.py')\n"
              "m = importlib.util.module_from_spec(spec)\n"
              "spec.loader.exec_module(m)\n"
              "m._snapshot_driver(sys.argv[1], sys.argv[2])\n")
    env = {**os.environ, "PYTHONPATH": str(snap), "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "SENTINEL_RUNS": str(tmp_path / "runs")}

    def chunk(shard):
        p = subprocess.run([sys.executable, "-B", "-c", driver, str(run_dir), shard], cwd=snap, env=env,
                           capture_output=True, text=True, timeout=900, check=False)
        out = [ln for ln in p.stdout.splitlines() if ln.startswith("UNITS ")]
        return p.returncode, (json.loads(out[-1][6:]) if out else None), p.stderr[-3000:]

    def n_units():
        return len([p for p in (run_dir / "units").iterdir() if not p.name.startswith(".")])

    rc, a, err = chunk("0/3")
    assert rc == 0 and a["rc"] == 0 and a["fits"] == 32 and len(a["units"]) == 32, err
    # the live tree changes (three files of the scoring closure): the next chunk neither aborts nor changes
    for rel in ("sentinel/models/bioassess/xdev.py", "scripts/eval/bioassess_x_final.py",
                "sentinel/models/bioassess/xtestlook.py"):
        with open(live / rel, "a") as fh:
            fh.write("\nraise SystemExit('the live tree was used')\n")
    rc, b, err = chunk("1/3")
    assert rc == 0 and b["rc"] == 0 and b["fits"] == 32 and len(b["units"]) == 64, err
    assert all(b["units"][k] == v for k, v in a["units"].items())

    def refused(msg):
        rc, out, err = chunk("2/3")
        assert rc != 0 and out is None and msg in err, err
        assert n_units() == 64                                    # aborted before any unit

    snap.chmod(0o755)                                             # a file the record does not list
    (snap / "extra.py").write_text("x = 1\n")
    refused("code snapshot file set changed")
    (snap / "extra.py").unlink()
    snap.chmod(0o555)
    for rel, msg in (("sentinel/models/bioassess/xdev.py", "scoring code changed since the freeze"),
                     ("tests/test_bioassess_x_dev.py", "code snapshot files changed")):
        f = snap / rel
        raw = f.read_bytes()
        f.chmod(0o644)
        f.write_bytes(raw + b"\n# edited after the freeze\n")
        refused(msg)
        f.write_bytes(raw)
        f.chmod(0o444)
    rc, c, err = chunk("2/3")                                     # restored byte for byte: the look goes on
    assert rc == 0 and c["rc"] == 0 and c["fits"] == 32 and len(c["units"]) == 96, err
    ref = _ToyLook(tmp_path, monkeypatch.setattr, name="ref")     # the same look fitted in one process, live code
    ref.claim()
    assert ref.chunk() == 0 and c["units"] == ref.unit_shas()
    assert not list(snap.rglob("__pycache__"))


# ----------------------------------------------------------------- development rehearsal cap (protocol v11 item 7)
def test_x_rehearsal_cap_and_max_folds_are_refused_in_real_mode(tmp_path, monkeypatch):
    import argparse

    L = _ToyLook(tmp_path, monkeypatch.setattr, cap=5)
    F, XL = L.F, L.XL
    with pytest.raises(XL.SingleLookError, match="refused in real mode"):
        L.claim()
    assert not (L.root / "INPUTS.json").exists() and not (L.run_dir / "plan.json").exists()
    assert not (L.run_dir / "inputs").exists()
    L.cap = None
    with pytest.raises(XL.SingleLookError, match="refused in real mode"):
        L.F.write_inputs(L.rec, L.run_dir, L.S, L.frozen, L.D, max_folds=1, cap=None, code="toy", **L.kw())
    with pytest.raises(XL.SingleLookError, match="rehearsal cap is refused in real mode"):
        F.apply_rehearsal(False)
    assert F._REHEARSAL_ON is False

    def boom(dry):
        raise AssertionError("the record was loaded")

    monkeypatch.setattr(F, "load_record", boom)                    # refused before anything else happens
    for kw in ({"rehearsal_cap": 5, "max_folds": None}, {"rehearsal_cap": None, "max_folds": 1}):
        with pytest.raises(SystemExit, match="refused in real mode"):
            F.cmd_claim(argparse.Namespace(dry_run_dev=False, authorised_by="lead", **kw))
    info = F.write_inputs(L.rec, L.run_dir, L.S, L.frozen, L.D, max_folds=None, cap=None, code="toy", **L.kw())
    for kw in ({"cap": 5, "max_folds": None}, {"cap": None, "max_folds": 1}):
        with pytest.raises(XL.SingleLookError, match="refused in real mode"):
            F.look_design(L.run_dir, info["inputs"], L.S, L.frozen, L.D, dry=False, **kw)
    with pytest.raises(XL.SingleLookError, match="--cpu-smoke is refused in real mode"):
        L.chunk(cpu_smoke=True)
    with pytest.raises(XL.SingleLookError, match="belongs to the other mode"):
        F.chunk_state(L.run_dir, **{**L.kw(), "dry": True})
    # a real plan edited to carry a cap (with its hash in INPUTS.json updated to match) is still refused
    pf, inf = L.run_dir / "plan.json", L.root / "INPUTS.json"
    raw = pf.read_bytes()
    for key, val in (("rehearsal_cap", 5), ("max_folds", 1)):
        plan = json.loads(raw)
        plan[key] = val
        pf.chmod(0o644)
        pf.write_text(json.dumps(plan, sort_keys=True))
        j = json.loads(inf.read_text())
        j["plan_sha256"] = F.file_sha256(pf)
        inf.chmod(0o644)
        inf.write_text(json.dumps(j))
        with pytest.raises(XL.SingleLookError, match="refused in real mode"):
            L.chunk()
    assert L.fits == [] and L.unit_shas() == {}


def test_x_rehearsal_cap_runs_in_the_dry_look_and_marks_its_numbers_unusable(tmp_path, monkeypatch):
    from sentinel.models.bioassess import xbioset as XB
    from sentinel.models.bioassess import xdev as XD

    full = {n: P["n_estimators"] for n, P in (("rf", XD.RF_PARAMS), ("gbm", XD.GBM_PARAMS), ("tp", XD.TABPFN_PARAMS))}
    for P in (XD.RF_PARAMS, XD.GBM_PARAMS, XD.TABPFN_PARAMS):       # restored after the test
        monkeypatch.setitem(P, "n_estimators", P["n_estimators"])
    monkeypatch.setattr(XB, "base_cfg", XB.base_cfg)
    L = _ToyLook(tmp_path, monkeypatch.setattr, dry=True, cap=5)
    F = L.F
    fo = XD.Fold("X-L", "A", L.D.samples.index, pd.Index(["nla:te0"]))          # two source tables
    c1, c2 = F.cap_fold(fo, L.D, 5), F.cap_fold(fo, L.D, 5)
    assert list(c1.train) == list(c2.train) and len(c1.train) == 10 and list(c1.test) == ["nla:te0"]
    assert L.D.samples.loc[c1.train, "source"].value_counts().to_dict() == {"nla": 5, "uk": 5}
    assert set(c1.train) <= set(fo.train) and list(F.cap_fold(fo, L.D, 1000).train) == list(fo.train)
    info = L.claim()
    plan = json.loads((L.run_dir / "plan.json").read_text())
    assert info["n_units"] == 96 and plan["rehearsal_cap"] == 5 and plan["dry"] is True
    assert all(f["n_train"] == 5 for b in XD.BLOCKS for f in plan["folds"][b])
    assert XD.RF_PARAMS["n_estimators"] == full["rf"]                # nothing is reduced by the claim
    assert L.chunk() == 0 and len(L.fits) == 96
    assert (XD.RF_PARAMS["n_estimators"], XD.GBM_PARAMS["n_estimators"], XD.TABPFN_PARAMS["n_estimators"]) == (
        F.REHEARSAL["rf_n_estimators"], F.REHEARSAL["gbm_n_estimators"], F.REHEARSAL["tabpfn_n_estimators"])
    assert XB.base_cfg.__name__ == "capped" and F._REHEARSAL_ON is True
    L.seal()
    L.score()
    info = json.loads((L.run_dir / "results" / "pass_info.json").read_text())
    assert "numbers not usable" in info["rehearsal"]["NOTE"] and info["dry_run"] is True
    assert info["rehearsal"]["max_training_rows_per_source"] == 5
    assert {k: info["rehearsal"][k] for k in F.REHEARSAL} == F.REHEARSAL
    rows = L.XL.scoring_rows(L.reg)
    assert rows and all(r["split"].endswith("_dry_run_dev") for r in rows)      # never a test split


def test_x_dry_look_max_folds_limits_the_plan(tmp_path, monkeypatch):
    L = _ToyLook(tmp_path, monkeypatch.setattr, dry=True)
    info = L.claim(max_folds=1)
    plan = json.loads((L.run_dir / "plan.json").read_text())
    assert info["n_units"] == 32 and plan["max_folds"] == 1 and [len(plan["folds"][b]) for b in L.XD.BLOCKS] == [1, 1]
    assert L.chunk() == 0 and len(L.fits) == 32 and L.seal()["n_files"] > 0


def test_x_rehearsal_cap_leaves_the_r_methods_at_full_size(tmp_path, monkeypatch):
    """v11 item 7: the dry-run cap thins the training rows of the ML fits only. The R methods (bio.infer) are
    fitted on the whole training set of the fold, so the rehearsal is faithful for them and their R code is not
    run on a degenerate training set; the unit plan is the same with and without the cap."""
    L = _ToyLook(tmp_path, monkeypatch.setattr, dry=True, n_train=60)
    F = L.F
    L.claim()
    info = L.XL.inputs_info(L.root)
    kw = {"dry": True, "max_folds": None}
    _d0, f0, r0 = F.look_design(L.run_dir, info["inputs"], L.S, L.frozen, L.D, cap=None, **kw)
    d1, f1, r1 = F.look_design(L.run_dir, info["inputs"], L.S, L.frozen, L.D, cap=4, **kw)
    for b in F.XD.BLOCKS:
        # R: thinned to F.R_CAP_FACTOR x the cap, so its R code gets a training set it can build coefficients from
        assert all(len(g.train) == min(len(f.train), F.R_CAP_FACTOR * len(fo.train))
                   for fo, g, f in zip(f1[b], r1[b], r0[b]))
        assert all(len(fo.train) < len(g.train) <= len(f.train) for fo, g, f in zip(f1[b], r1[b], r0[b]))
        assert [str(fo.g) for fo in f1[b]] == [str(fo.g) for fo in r1[b]]             # same folds, same rows
        assert all(list(fo.test) == list(g.test) for fo, g in zip(f1[b], r1[b]))
    rec = L.rec
    assert json.dumps(F.make_plan(rec, f1, dry=True, max_folds=None, cap=4), sort_keys=True) != \
        json.dumps(F.make_plan(rec, f0, dry=True, max_folds=None, cap=None), sort_keys=True)   # the cap is in it
    assert [u["key"] for u in F.make_plan(rec, f1, dry=True, max_folds=None, cap=4)["units"]] == \
        [u["key"] for u in F.make_plan(rec, f0, dry=True, max_folds=None, cap=None)["units"]]  # same units
    # the fitter hands an R unit the full fold and every other unit the thinned one
    fitter = F.Fitter({"S": L.S, "frozen": L.frozen, "designs": d1, "folds": f1, "folds_r": r1}, L.run_dir, "cpu")
    seen = {}
    monkeypatch.setattr(F.Fitter, "leaf", lambda self, spec, b, k, fo, s: seen.setdefault(spec["kind"], len(fo.train)))
    b0 = F.XD.BLOCKS[0]
    base = {"kind": "leaf", "block": b0, "k": 0, "g": str(f1[b0][0].g), "seed": 0}
    for spec in ({"kind": "r", "method": "bioinfer"}, {"kind": "baseline", "name": "rf_community", "variant": {}}):
        try:
            fitter.fit({**base, "key": f"u_{spec['kind']}", "spec": spec})
        except (TypeError, AttributeError):              # the patched leaf returns an int, not a frame
            pass
    assert seen["r"] == len(r1[b0][0].train) > seen["baseline"] == len(f1[b0][0].train)
