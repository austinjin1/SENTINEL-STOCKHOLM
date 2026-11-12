# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""scripts/train/orbitv2_power.py on synthetic rows (lead rulings R-15, R-16, R-18, R-18b; amendment §1.7, §2.6).

No real validation prediction, no protected count and no file under the real runs root is touched."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[1]
BASE = ("chla", "secchi", "tss", "turbidity")
TRACK_N = ("tp", "tn", "doc")


@pytest.fixture(scope="module")
def P():
    spec = importlib.util.spec_from_file_location("orbitv2_power", REPO / "scripts" / "train" / "orbitv2_power.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def table(n_blocks=12, per_block=8, sd_arm=0.10, sd_cmp=0.20, seed=0, sd=None) -> pd.DataFrame:
    """Synthetic validation rows: every target in both lake cells, arms better than comparators by default.
    ``sd`` overrides the error SD of single columns, e.g. {"arm:v2": 0.3}."""
    rs = np.random.default_rng(seed)
    sd = sd or {}
    out = []
    for t in BASE + TRACK_N:
        n = n_blocks * per_block
        d = pd.DataFrame({"target": t, "row_id": [f"{t}{i}" for i in range(n)],
                          "block": np.repeat([f"B{i}" for i in range(n_blocks)], per_block),
                          "y": rs.normal(0.0, 0.4, n), "small_lake": np.arange(n) % 2 == 0,
                          "ntu_class": np.arange(n) % 4 != 3})
        for c in ("arm:v1", "arm:v2", "arm:track_n"):
            d[c] = d.y + rs.normal(0.0, sd.get(c, sd_arm), n)
        for c in ("mdn_b", "gbm", "knn", "mdn_b+gbm", "season_loc_gbm", "trophic_ols"):
            d[f"cmp:{c}"] = d.y + rs.normal(0.0, sd.get(f"cmp:{c}", sd_cmp), n)
        for cell in ("valxval_lake", "valxany_lake"):
            out.append(d.assign(cell=cell))
    return pd.concat(out, ignore_index=True)


def counts(n_blocks=40) -> dict:
    one = {t: {"n": 10 * n_blocks, "n_waterbodies": 3 * n_blocks, "n_blocks": n_blocks} for t in BASE + TRACK_N}
    return {"all": one, "small_lake": {t: dict(v) for t, v in one.items()}}


def test_the_ruled_constants(P):
    assert (P.N_SIM, P.SEED_VAL, P.SEED_TEST, P.Z_POWER, P.ALPHA) == (2000, 0, 1, 0.84, 0.05)
    assert {f: v["divisor"] for f, v in P.FAMILIES.items()} == {"H1-S": 2, "H2-S": 6, "TrackN": 3, "small_lakes": 32}
    assert P.holm_alpha("H1-S") == 0.025 and P.holm_alpha("H2-S") == 0.05 / 6 and P.holm_alpha("TrackN") == 0.05 / 3
    assert P.mde_factor(0.05) == pytest.approx(2.80, abs=1e-3)              # the v1 factor at alpha 0.05
    assert P.mde_factor(0.025) == pytest.approx(norm.ppf(1 - 0.0125) + 0.84)
    assert P.FAMILIES["H2-S"]["max_mde"] == 15 and P.FAMILIES["H2-S"]["min_blocks"] == 30
    assert P.FAMILIES["TrackN"]["max_mde"] == 15 and P.FAMILIES["small_lakes"]["max_mde"] == 10
    assert P.FAMILIES["small_lakes"]["gate_targets"] == ("chla",)
    for f in ("H1-S", "H2-S", "small_lakes"):                               # kNN and the average are claim comparators
        assert P.FAMILIES[f]["comparators"] == ("mdn_b", "gbm", "knn", "mdn_b+gbm")
        assert P.FAMILIES[f]["arms"] == ("v1", "v2")
    assert P.FAMILIES["TrackN"]["comparators"] == ("mdn_b", "gbm", "knn", "season_loc_gbm", "trophic_ols", "mdn_b+gbm")
    assert P.EXPERIMENT == "orbit_power_S_dev_v5m"


def test_base_selection_cells_are_those_of_v1(P):
    src = (REPO / "scripts" / "train" / "orbit_compare.py").read_text()
    assert ('PRIMARY_TARGET_CELL = {"chla": PL.PRIMARY_CELL, "secchi": PL.PRIMARY_CELL, "tss": "valxany_lake",\n'
            '                       "turbidity": "valxany_lake"}') in src
    assert P.BASE_CELL == {"chla": "valxval_lake", "secchi": "valxval_lake", "tss": "valxany_lake",
                           "turbidity": "valxany_lake"}


def test_block_draws_is_the_v1_function_with_the_ruled_arguments(P):
    real = P.load_block_draws()
    assert real.__code__.co_filename.endswith("scripts/train/orbit_power.py") and real.__name__ == "block_draws"
    calls, draws = [], {}

    def rec(df, comps, n_blocks, n_sim, seed):
        calls.append((n_blocks, n_sim, seed))
        out = real(df, comps, n_blocks, n_sim, seed)
        draws[(df.target.iloc[0], len(df), float(df["SENTINEL"].iloc[0]), seed)] = (out, list(comps))
        return out

    vp = table()
    res, _ = P.run(vp, counts(40), block_draws=rec)
    assert {c[1] for c in calls} == {2000}
    assert {c for c in calls} == {(12, 2000, 0), (40, 2000, 1)}
    assert len(calls) == 2 * (2 + 3 * 2 + 3 + 4 * 2)                       # two calls per (family, target, arm)
    r = res[(res.family == "H1-S") & (res.arm == "v2") & (res.comparator == "gbm")].iloc[0]
    d = P.cell_rows(vp, "chla", "all")
    out, comps = draws[("chla", len(d), float(d["arm:v2"].iloc[0]), 1)]
    se = out[:, comps.index("cmp:gbm")].std(ddof=1)
    assert r.se_test == pytest.approx(se) and r.mde80_test == pytest.approx((norm.ppf(1 - 0.0125) + 0.84) * se)
    assert r.alpha_min == 0.025 and r.n_blocks_test == 40 and r.n_blocks_val == 12 and r.val_cell == "valxval_lake"


def test_binding_mde_is_the_largest_over_arms_and_comparators(P):
    vp = table(sd={"cmp:knn": 0.6})
    res, st = P.run(vp, counts(40), n_sim=200)
    h1 = res[res.family == "H1-S"]
    assert len(h1) == 2 * 4
    s = st["H1-S"]["chla"]
    assert s["status"] == "primary" and s["binding_mde80"] == pytest.approx(h1.mde80_test.max())
    assert s["binding_comparator"] == "knn" and s["arms"] == ["v1", "v2"] and s["holm_divisor"] == 2


def test_h2s_gate_needs_30_blocks_and_15_points(P):
    vp = table(n_blocks=30, per_block=20, sd_arm=0.02, sd_cmp=0.03)
    _, st = P.run(vp, counts(60), n_sim=200)
    assert all(st["H2-S"][t]["binding_mde80"] <= 15 for t in ("secchi", "tss", "turbidity"))
    assert {st["H2-S"][t]["status"] for t in ("secchi", "tss", "turbidity")} == {"confirmatory"}
    c = counts(60)
    c["all"]["tss"]["n_blocks"] = 29
    _, st = P.run(vp, c, n_sim=200)
    assert st["H2-S"]["tss"]["status"] == "descriptive" and "fewer than 30 role blocks" in st["H2-S"]["tss"]["reason"]
    assert st["H2-S"]["secchi"]["status"] == "confirmatory" and st["H2-S"]["tss"]["holm_divisor"] == 6
    _, st = P.run(table(sd_arm=0.3, sd_cmp=0.5), counts(40), n_sim=200)       # wide errors: MDE80 above 15 points
    assert st["H2-S"]["secchi"]["binding_mde80"] > 15 and st["H2-S"]["secchi"]["status"] == "descriptive"
    assert st["H2-S"]["secchi"]["alpha_min"] == 0.05 / 6


def test_track_n_gate_and_the_cases_without_an_arm(P):
    vp = table(n_blocks=30, per_block=20, sd_arm=0.02, sd_cmp=0.03, sd={"cmp:trophic_ols": 0.9})
    res, st = P.run(vp, counts(60), n_sim=200)
    tn = res[res.family == "TrackN"]
    assert set(tn.arm) == {"track_n"} and len(tn) == 3 * 6 and set(tn.alpha_min) == {0.05 / 3}
    assert st["TrackN"]["tp"]["binding_comparator"] == "trophic_ols"          # every §2.5 comparator counts
    wide = table(sd_arm=0.3, sd_cmp=0.5)
    _, st = P.run(wide, counts(40), n_sim=200)
    assert st["TrackN"]["tp"]["binding_mde80"] > 15 and st["TrackN"]["tp"]["status"] == "descriptive"
    res, st = P.run(vp, counts(60), n_sim=200, track_n_arm=False)
    assert "TrackN" not in set(res.family)
    assert {s["status"] for s in st["TrackN"].values()} == {"no SENTINEL claim"}
    assert all("kill criterion" in s["reason"] and s["holm_divisor"] == 3 for s in st["TrackN"].values())
    res, st = P.run(vp, counts(60), n_sim=200, dropped_heads=("doc",))
    assert set(res[res.family == "TrackN"].target) == {"tp", "tn"}
    assert st["TrackN"]["doc"]["status"] == "no SENTINEL claim" and "C-4" in st["TrackN"]["doc"]["reason"]
    assert st["TrackN"]["tp"]["holm_divisor"] == 3 and st["TrackN"]["tp"]["alpha_min"] == 0.05 / 3


def test_small_lake_status_is_set_by_chla_for_the_whole_family(P):
    vp = table(n_blocks=30, per_block=20, sd_arm=0.01, sd_cmp=0.015)
    res, st = P.run(vp, counts(80), n_sim=200)
    sl = res[res.family == "small_lakes"]
    assert set(sl.alpha_min) == {0.05 / 32} and set(sl.target) == set(BASE)
    assert int(sl[sl.target == "chla"].n_val_rows.iloc[0]) == 300              # small-lake rows only
    assert st["small_lakes"]["chla"]["binding_mde80"] <= 10
    assert {st["small_lakes"][t]["status"] for t in BASE} == {"confirmatory"}
    _, st = P.run(table(sd_arm=0.3, sd_cmp=0.5), counts(40), n_sim=200)
    assert st["small_lakes"]["chla"]["binding_mde80"] > 10
    assert {st["small_lakes"][t]["status"] for t in BASE} == {"descriptive"}
    assert all("set by chla" in st["small_lakes"][t]["reason"] for t in BASE)


def test_strata_and_selection_cells(P):
    vp = table(n_blocks=30, per_block=20)
    assert len(P.cell_rows(vp, "turbidity", "all")) == 450                     # NTU-class rows only (R-18)
    assert set(P.cell_rows(vp, "turbidity", "all").cell) == {"valxany_lake"}
    assert len(P.cell_rows(vp, "chla", "small_lake")) == 300 and len(P.cell_rows(vp, "chla", "all")) == 600
    assert P.selection_cell(vp, "tp") == "valxval_lake"                        # 600 rows in 30 blocks
    few_rows = vp[~((vp.target == "tn") & (vp.cell == "valxval_lake") & (vp.groupby(["target", "cell"]).cumcount() >= 199))]
    assert P.selection_cell(few_rows, "tn") == "valxany_lake"                  # 199 rows
    few_blocks = vp[~((vp.target == "doc") & (vp.cell == "valxval_lake") & ~vp.block.isin([f"B{i}" for i in range(19)]))]
    assert P.selection_cell(few_blocks, "doc") == "valxany_lake"               # 19 blocks
    with pytest.raises(P.PowerError):
        P.cell_rows(vp, "chla", "scene_disjoint")


def test_under_powered_flag_and_small_cells(P):
    res, st = P.run(table(sd_arm=0.3, sd_cmp=0.15), counts(40), n_sim=200)      # arms worse: every endpoint under-powered
    assert res.underpowered_at_val_gain.all() and st["H1-S"]["chla"]["underpowered_arms"] == ["v1", "v2"]
    res, st = P.run(table(n_blocks=40, per_block=30, sd_arm=0.01, sd_cmp=0.5), counts(200), n_sim=200)
    assert not res[res.family == "H1-S"].underpowered_at_val_gain.any()
    assert st["H1-S"]["chla"]["underpowered_arms"] == []
    c = counts(40)
    c["all"]["secchi"]["n_blocks"] = 1
    res, st = P.run(table(), c, n_sim=200)
    assert set(res[(res.family == "H2-S") & (res.target == "secchi")].note) == {"test cell too small"}
    assert st["H2-S"]["secchi"]["status"] == "descriptive" and st["H2-S"]["secchi"]["binding_mde80"] is None
    res, st = P.run(table(n_blocks=3, per_block=4), counts(40), n_sim=50)       # 12 rows: below the 20-row floor
    assert set(res.note) == {"validation cell too small"} and st["H1-S"]["chla"]["status"] == "descriptive"


def test_bad_inputs_are_refused(P):
    vp = table()
    with pytest.raises(P.PowerError, match="lacks"):
        P.run(vp.drop(columns=["cmp:mdn_b+gbm"]), counts(), n_sim=50)
    with pytest.raises(P.PowerError, match="lacks"):
        P.run(vp.drop(columns=["arm:v1"]), counts(), n_sim=50)
    with pytest.raises(P.PowerError, match="lacks columns"):
        P.run(vp.drop(columns=["ntu_class"]), counts(), n_sim=50)
    bad = vp.copy()
    bad.loc[(bad.target == "chla") & (bad.cell == "valxval_lake"), "cmp:knn"] = np.nan
    with pytest.raises(P.PowerError, match="non-finite predictions"):
        P.run(bad, counts(), n_sim=50)
    bad = vp.copy()
    bad.loc[0, "y"] = np.nan
    with pytest.raises(P.PowerError, match="non-finite labels"):
        P.run(bad, counts(), n_sim=50)
    with pytest.raises(P.PowerError, match="repeats"):
        P.run(pd.concat([vp, vp.iloc[:1]]), counts(), n_sim=50)
    c = counts()
    del c["small_lake"]["tss"]
    with pytest.raises(P.PowerError, match="no protected count"):
        P.run(vp, c, n_sim=50)


def _files(tmp_path, P, primary="all_rows"):
    vp = table(n_blocks=6, per_block=5)
    vp.to_parquet(tmp_path / "val_rows.parquet")
    rep = {"test_s_primary_cell": primary, "test_s_primary_counts": counts(40)}
    (tmp_path / "counts.json").write_text(json.dumps(rep))
    return ["--val-rows", str(tmp_path / "val_rows.parquet"), "--counts-json", str(tmp_path / "counts.json")]


def test_main_is_refused_before_the_stamp(P, tmp_path, monkeypatch):
    args = _files(tmp_path, P)
    monkeypatch.setattr(P.ST, "stamp_status", lambda *a, **k: {"stamped": False, "reason": "test: no stamp"})
    with pytest.raises(P.ST.StampError):
        P.main([*args, "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_main_writes_once_and_only_into_its_folder(P, tmp_path, monkeypatch):
    args = _files(tmp_path, P)
    monkeypatch.setattr(P.ST, "stamp_status", lambda *a, **k: {"stamped": True, "reason": "stamped", "row_id": "t0"})
    monkeypatch.setattr(P, "N_SIM", 2000)
    out = tmp_path / "out"
    assert P.main([*args, "--out", str(out)]) == 0
    assert sorted(f.name for f in out.iterdir()) == ["power_S.csv", "status.json"]
    rec = json.loads((out / "status.json").read_text())
    assert rec["n_sim"] == 2000 and rec["seeds"] == [0, 1] and rec["stamp_row"] == "t0"
    assert rec["holm_divisors"] == {"H1-S": 2, "H2-S": 6, "TrackN": 3, "small_lakes": 32}
    assert rec["test_s_primary_cell"] == "all_rows" and set(rec["status"]) == set(P.FAMILIES)
    assert rec["sha256"]["scripts/train/orbitv2_power.py"] == P.sha256(REPO / "scripts" / "train" / "orbitv2_power.py")
    assert rec["sha256"]["scripts/train/orbit_power.py"] == P.sha256(REPO / "scripts" / "train" / "orbit_power.py")
    with pytest.raises(P.PowerError, match="computed once"):                   # R-16: no second calculation
        P.main([*args, "--out", str(out)])


def test_main_needs_the_primary_cell_of_section_1_6(P, tmp_path, monkeypatch):
    args = _files(tmp_path, P, primary="undecided")
    monkeypatch.setattr(P.ST, "stamp_status", lambda *a, **k: {"stamped": True, "reason": "stamped", "row_id": "t0"})
    with pytest.raises(P.PowerError, match="primary Test-S cell"):
        P.main([*args, "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()
