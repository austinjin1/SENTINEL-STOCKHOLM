"""Scoring rules of the M2S/test look (sentinel/models/orbit_v2/claims.py) on synthetic sealed tables. No real
row, no file under the runs root."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.models.orbit import evaluate as E
from sentinel.models.orbit_v2 import claims as CL
from sentinel.models.orbit_v2 import looks as LK
from sentinel.models.orbit_v2 import units as U

REPO = Path(__file__).resolve().parents[1]
N_V1 = 5
TARGETS = ("chla", "secchi", "tss", "turbidity", "tp", "tn", "doc")
STATUS = {"H1-S": {"chla": {"status": "primary"}},
          "H2-S": {"secchi": {"status": "confirmatory"}, "tss": {"status": "descriptive"},
                   "turbidity": {"status": "confirmatory"}},
          "TrackN": {"tp": {"status": "confirmatory"}, "tn": {"status": "descriptive"},
                     "doc": {"status": "no SENTINEL claim"}},
          "small_lakes": {t: {"status": "descriptive"} for t in ("chla", "secchi", "tss", "turbidity")}}


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _table(n=600, n_blocks=40, seed=0, sd=None):
    """A sealed cell table: per target ``n`` rows; error sd (log10) per model column, default: arms 0.10, units
    0.11, networks 0.12, comparators 0.20."""
    rng = np.random.default_rng(seed)
    cols = LK.sealed_columns({"v1_networks": N_V1, "v2": True, "track_n": True, "comparators": list(CL.TRACK_N_COMPARATORS)})
    sd = sd or {}
    tabs = []
    for t in TARGETS:
        y = rng.normal(1.0, 0.5, n)
        tab = pd.DataFrame({"target": t, "row_id": [f"{t}{i}" for i in range(n)], "y": y,
                            "block": [f"b{i % n_blocks}" for i in range(n)], "small_lake": np.arange(n) % 3 == 0,
                            "ntu_class": np.arange(n) % 4 != 0, "scene_disjoint": np.arange(n) % 5 != 0,
                            "year": 2015 + np.arange(n) % 9, "source": np.where(np.arange(n) % 2, "wqp", "aquacorr"),
                            "n_context": np.arange(n) % 3, "dist_train_km": rng.uniform(0, 40, n)})
        for c in cols:
            base = 0.20 if c.startswith("cmp:") else 0.12 if "i" in c.split("@")[-1] or "@s" in c else \
                0.11 if "@unit" in c else 0.10
            tab[c] = y + rng.normal(0, sd.get((t, c), sd.get(c, base)), n)
        tabs.append(tab)
    return pd.concat(tabs, ignore_index=True)


def test_families_agree_with_the_power_script_and_the_comparator_script():
    P = _load("orbitv2_power", "scripts/train/orbitv2_power.py")
    S = _load("orbitv2_comparators", "scripts/train/orbitv2_comparators.py")
    assert set(CL.FAMILIES) == set(P.FAMILIES)
    for k, f in CL.FAMILIES.items():
        for key in ("divisor", "targets", "stratum"):
            assert f[key] == P.FAMILIES[k][key], (k, key)
        assert set(f["comparators"]) == set(P.FAMILIES[k]["comparators"]) and tuple(f["arms"]) == tuple(P.FAMILIES[k]["arms"])
    assert [CL.FAMILIES[k]["divisor"] for k in ("H1-S", "H2-S", "TrackN", "small_lakes")] == [2, 6, 3, 32]
    assert set(CL.TRACK_N_COMPARATORS) == set(S.CLAIM) and "mdn_b+gbm" in CL.BASE_COMPARATORS       # R-10
    assert (CL.N_BOOT, CL.SEED) == (2000, 0) == (E.N_BOOT, 0)
    assert CL.label_scope("scene_disjoint") == "unseen lakes, archive years 2015–2023, scene-disjoint"       # R-18b
    assert CL.label_scope("all_rows") == "unseen lakes, archive years 2015–2023, scenes shared with training"
    with pytest.raises(CL.ClaimError):
        CL.label_scope("val")


def test_paired_p_has_the_plus_one_correction_on_the_draws_of_evaluate():
    rng = np.random.default_rng(1)
    n = 400
    y, g = rng.normal(0, 1, n), np.arange(n) % 25
    a, b = y + rng.normal(0, 0.10, n), y + rng.normal(0, 0.11, n)
    ref, got = E.paired_mdsa(y, a, b, g, n_boot=300, seed=0), CL.paired(y, a, b, g, n_boot=300, seed=0)
    for k in ("diff", "ci_low", "ci_high", "n_clusters", "n"):
        assert got[k] == pytest.approx(ref[k])
    d = np.array([E.mdsa_log10(a[ix] - y[ix]) - E.mdsa_log10(b[ix] - y[ix]) for ix in E._boot_indices(g, 300, 0)])
    assert got["p_value"] == pytest.approx(min(1, 2 * min(((d <= 0).sum() + 1) / 301, ((d >= 0).sum() + 1) / 301)))
    assert got["p_value"] > ref["p_value"] or ref["p_value"] == 1.0
    strong = CL.paired(y, y + rng.normal(0, 0.01, n), y + rng.normal(0, 0.5, n), g, n_boot=200)
    assert strong["p_value"] == pytest.approx(2 / 201) and strong["diff"] < 0          # never zero
    with pytest.raises(CL.ClaimError):
        CL.paired(y, np.where(np.arange(n) == 3, np.nan, a), b, g, n_boot=10)


def test_holm_is_step_down_monotone_and_keeps_the_family_divisor():
    h = CL.holm({"v1": 0.04, "v2": 0.02}, 2)                    # §6.2: smaller at 0.025, the other at 0.05
    assert h["v2"]["rejected"] and h["v1"]["rejected"] and h["v2"]["alpha_at_rank"] == 0.025
    h = CL.holm({"v1": 0.04, "v2": 0.03}, 2)                    # the first fails: step-down stops, monotone p
    assert not h["v2"]["rejected"] and not h["v1"]["rejected"] and h["v1"]["p_holm"] == pytest.approx(0.06)
    h = CL.holm({"a": 0.009, "b": 0.02}, 6)                     # R-18: 2 hypotheses left, the divisor stays 6
    assert h["a"]["p_holm"] == pytest.approx(0.054) and not h["a"]["rejected"]
    assert CL.holm({"a": 0.008, "b": 0.009}, 6)["b"]["p_holm"] == pytest.approx(0.048)
    assert CL.holm({"a": 0.025}, 2)["a"]["rejected"] is False   # adjusted p must be below 0.05
    with pytest.raises(CL.ClaimError):
        CL.holm({i: 0.01 for i in range(3)}, 2)
    with pytest.raises(CL.ClaimError):
        CL.holm({"a": float("nan")}, 2)


def test_endpoint_intersection_union_and_per_seed_rule():
    tab = _table()
    g = CL.endpoint_rows(tab, "chla")
    e = CL.endpoint(g, "v2", CL.BASE_COMPARATORS, N_V1, n_boot=200)
    assert e["n"] == 600 and e["n_dropped_nonfinite"] == 0 and e["all_favour"]
    assert e["p_iu"] == max(p["p_value"] for p in e["pairs"].values()) == e["p_counted"]
    assert e["per_seed"]["passed"] and e["per_seed"]["sentence"] == U.PER_SEED_SENTENCE.format(k=9)
    assert set(e["mdsa"]) == {*LK.member_columns("v2"), *[f"cmp:{c}" for c in CL.BASE_COMPARATORS]}
    e1 = CL.endpoint(g, "v1", CL.BASE_COMPARATORS, N_V1, n_boot=200)
    assert e1["per_seed"] == {**e1["per_seed"], "passed": True, "k_networks": N_V1, "n_networks": N_V1}
    assert "5 of 5" in e1["per_seed"]["sentence"]
    # one comparator better than the arm: the hypothesis does not count whatever the other p-values
    tab2 = _table(sd={("chla", "cmp:knn"): 0.05})
    e = CL.endpoint(CL.endpoint_rows(tab2, "chla"), "v2", CL.BASE_COMPARATORS, N_V1, n_boot=200)
    assert not e["all_favour"] and e["p_counted"] == 1.0 and e["pairs"]["knn"]["diff"] > 0
    assert not e["per_seed"]["passed"] and e["per_seed"]["k_networks"] == 0
    # one seed unit worse than a comparator: the arm may win, the per-seed rule fails
    tab3 = _table(sd={("chla", "v2@unit1"): 0.30, ("chla", "v1@s3"): 0.30})
    g3 = CL.endpoint_rows(tab3, "chla")
    e = CL.endpoint(g3, "v2", CL.BASE_COMPARATORS, N_V1, n_boot=200)
    assert e["all_favour"] and not e["per_seed"]["passed"]
    e = CL.endpoint(g3, "v1", CL.BASE_COMPARATORS, N_V1, n_boot=200)                  # v1: every single network
    assert e["all_favour"] and not e["per_seed"]["passed"] and e["per_seed"]["k_networks"] == N_V1 - 1
    with pytest.raises(CL.ClaimError, match="lacks"):
        CL.endpoint(g.drop(columns=["v2@u0i0"]), "v2", CL.BASE_COMPARATORS, N_V1, n_boot=10)
    # turbidity: NTU-class rows only (R-18); small-lake stratum
    assert len(CL.endpoint_rows(tab, "turbidity")) == 450 and len(CL.endpoint_rows(tab, "chla", "small_lake")) == 200
    assert len(CL.endpoint_rows(tab, "turbidity", "small_lake")) == int(((np.arange(600) % 3 == 0) & (np.arange(600) % 4 != 0)).sum())
    # a left-censored row is sealed with y = NaN and y_dl set (§5): it never enters an endpoint
    tabc = _table()
    cens = (tabc["target"] == "chla").to_numpy() & (np.arange(len(tabc)) % 7 == 0)
    tabc["censored"] = cens
    tabc["y_dl"] = np.where(cens, tabc["y"], np.nan)
    tabc.loc[cens, "y"] = np.nan
    gc = CL.endpoint_rows(tabc, "chla")
    assert len(gc) == 600 - int(cens.sum()) < 600 and np.isfinite(gc["y"].to_numpy(float)).all()
    ec = CL.endpoint(gc, "v2", CL.BASE_COMPARATORS, N_V1, n_boot=200)
    assert ec["n"] == len(gc) and ec["n_dropped_nonfinite"] == 0
    # §1.6 cells
    assert len(CL.primary_rows(tab, "all_rows")) == len(tab)
    assert CL.primary_rows(tab, "scene_disjoint").scene_disjoint.all()
    assert CL.primary_rows(tab, "all_rows", sensitivity=True).scene_disjoint.all()
    assert len(CL.primary_rows(tab, "scene_disjoint", sensitivity=True)) == len(tab)


def test_families_labels_statuses_and_divisors():
    tab = _table(sd={("secchi", "cmp:gbm"): 0.05, ("turbidity", "v1@s0"): 0.4})
    res = CL.all_families(tab, STATUS, N_V1, has_track_n=True, n_boot=1000)     # 6 x 2 / 1001 < 0.05
    h1 = res["H1-S"]
    assert h1["n_hypotheses"] == 2 and set(h1["labels"]) == {("chla", "v1"), ("chla", "v2")}
    assert h1["endpoints"]["chla", "v2"]["holm"]["iu"]["p_holm"] <= 2 * (2 / 1001) + 1e-12
    h2 = res["H2-S"]
    # secchi: a comparator beats both arms (no label); tss: descriptive (scored, outside Holm); turbidity: v2 only
    assert h2["divisor"] == 6 and h2["n_hypotheses"] == 4 and h2["labels"] == [("turbidity", "v2")]
    assert h2["endpoints"]["tss", "v2"]["label"] is False and h2["endpoints"]["tss", "v2"]["holm"] is None
    assert h2["endpoints"]["tss", "v2"]["mdsa"] and h2["endpoints"]["secchi", "v2"]["holm"]["iu"]["p"] == 1.0
    assert h2["endpoints"]["turbidity", "v1"]["holm"]["iu"]["rejected"]                # only the per-seed rule fails
    assert not h2["endpoints"]["turbidity", "v1"]["per_seed"]["passed"]
    assert h2["endpoints"]["turbidity", "v2"]["n"] == 450
    tn = res["TrackN"]
    assert tn["n_hypotheses"] == 1 and tn["labels"] == [("tp", "track_n")]
    assert set(tn["endpoints"]["tp", "track_n"]["pairs"]) == set(CL.TRACK_N_COMPARATORS)
    assert res["small_lakes"]["n_hypotheses"] == 0 and res["small_lakes"]["labels"] == []
    assert res["small_lakes"]["endpoints"]["chla", "v1"]["n"] == 200
    assert CL.counts_sentence(res) == ("confirmatory families in this look: 3 (H1-S, H2-S, TrackN); "
                                       "confirmatory hypotheses: 7")
    # small lakes confirmatory: 32 elements (K-1); a label needs all four elements of the target and arm
    st = {**STATUS, "small_lakes": {t: {"status": "confirmatory"} for t in ("chla", "secchi", "tss", "turbidity")}}
    sm = CL.family("small_lakes", tab, st, N_V1, n_boot=200)
    assert sm["n_hypotheses"] == 32 and sm["divisor"] == 32
    assert set(sm["endpoints"]["chla", "v2"]["holm"]) == set(CL.BASE_COMPARATORS)
    assert ("secchi", "v1") not in sm["labels"] and ("secchi", "v2") not in sm["labels"]
    e = sm["endpoints"]["secchi", "v2"]
    assert e["holm"]["gbm"]["p"] == 1.0 and not e["holm"]["gbm"]["rejected"]
    # smallest possible p (2/201) x 32 > 0.05: with B = 200 nothing can be rejected in a family of 32
    assert sm["labels"] == [] and e["holm"]["knn"]["p_holm"] >= 32 * 2 / 201 - 1e-12
    # a dropped row blocks the label and is counted
    tab4 = tab.copy()
    tab4.loc[(tab4.target == "chla") & (tab4.row_id == "chla7"), "cmp:knn"] = np.nan
    h = CL.family("H1-S", tab4, STATUS, N_V1, n_boot=200)
    assert h["labels"] == [] and h["endpoints"]["chla", "v2"]["n_dropped_nonfinite"] == 1
    assert h["endpoints"]["chla", "v2"]["holm"]["iu"]["rejected"]
    # freeze statuses are required; Track N without an arm
    with pytest.raises(CL.ClaimError, match="no status"):
        CL.family("H2-S", tab, {**STATUS, "H2-S": {}}, N_V1, n_boot=10)
    with pytest.raises(CL.ClaimError, match="primary endpoint"):
        CL.family("H1-S", tab, {**STATUS, "H1-S": {"chla": {"status": "descriptive"}}}, N_V1, n_boot=10)
    with pytest.raises(CL.ClaimError, match="no Track N arm"):
        CL.all_families(tab, STATUS, N_V1, has_track_n=False, n_boot=10)
    none = {**STATUS, "TrackN": {t: {"status": "no SENTINEL claim"} for t in ("tp", "tn", "doc")}}
    res = CL.all_families(tab.drop(columns=LK.member_columns("track_n")), none, N_V1, has_track_n=False, n_boot=50)
    assert res["TrackN"]["endpoints"] == {} and res["TrackN"]["labels"] == []


def test_descriptive_strata_and_intervals():
    tab = _table()
    g = CL.endpoint_rows(tab, "chla")
    s = CL.strata(g)
    assert set(s) == {"year<=2021", "year>2021", "source=aquacorr", "source=wqp", "context>=1", "context=0",
                      ">10km_from_train"}
    assert (s["year<=2021"] ^ s["year>2021"]).all() and (s["source=aquacorr"] ^ s["source=wqp"]).all()
    assert s["year>2021"].sum() == (g.year >= 2022).sum() and s["context=0"].sum() == (g.n_context == 0).sum()
    d = CL.descriptive(g, ["v1", "v2", "cmp:gbm", "cmp:trophic_ols"], n_boot=50)
    assert set(d.stratum) == {"all", *s} and set(d.model.dropna()) == {"v1", "v2", "cmp:gbm", "cmp:trophic_ols"}
    assert (d[d.stratum == "all"].n == 600).all() and d.mdsa_ci_low.notna().all()
    few = CL.descriptive(g.iloc[:10], ["v1"], n_boot=10)
    assert few.model.isna().all() and few.note.str.contains("counts only").all()
    g = g.assign(lo=g.y - 0.1, hi=g.y + 0.2)
    g.loc[:299, "lo"] = g.y[:300] + 0.05
    it = CL.interval_table(g, "lo", "hi")
    assert it == {"n": 600, "coverage": 0.5, "median_width_log10": pytest.approx(0.3 - 0.075, abs=0.08)} or it["coverage"] == 0.5
    assert CL.interval_table(g, "lo", "hi", offset=0.06)["coverage"] == 1.0
