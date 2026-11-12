# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""scripts/train/orbitv2_comparators.py on a synthetic dev_v5m-shaped table (amendment §2.5, §3.5; lead rulings
R-10 to R-14b; lead clarification 01, K-2 and K-3). Reduced families, CPU only; no cache, no chip store, no real
registry and no file under the real runs root is touched."""

from __future__ import annotations

import importlib.util
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sentinel.core import paths
from sentinel.models.orbit import data as D
from sentinel.models.orbit_v2 import comparators as C
from sentinel.models.orbit_v2 import stamp as ST

REPO = Path(__file__).resolve().parents[1]
LIMITS = dict(grid=(0, 1), feature_sets=("v1", "v1+ring+context_dates"), ks=(4, 8), max_rounds=30, threads=2,
              member_seeds=(0,), selected_seeds=(0, 1),
              mdn=(("7band", 150, False, "single"), ("12band+context+ring", 150, True, "joint")))
CLAIM7 = ["chla", "tss", "turbidity", "secchi", "tp", "tn", "doc"]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synth(n=1500, n_added=200):
    """The arm tests' synthetic table with the v1 band-median columns and context band medians added."""
    df, pix, feats, (cmeta, carr) = _load("arm_tests", "tests/test_orbit_v2_arm.py")._synth(n=n, n_added=n_added)
    sig = pix["refl"][:, 0, :].astype(float)
    for j, b in enumerate(D.BANDS):
        df[f"rw3_{b}"], df[f"rw7_{b}"] = sig[:, j], sig[:, j]
    df.loc[3, "rw3_B03"] = np.nan                                         # a band without a water pixel
    df["turbidity_unit_class"] = np.where(np.arange(n) % 3 == 0, "FNU", "NTU")
    carr["rw3"] = np.random.default_rng(5).uniform(0.2, 3.0, (len(cmeta), 12))
    return df, pix, feats, carr


@pytest.fixture(scope="module")
def S():
    mod = _load("orbitv2_comparators", "scripts/train/orbitv2_comparators.py")
    mod.LIM = dict(mod.full_limits(), **LIMITS)
    mod.lake_area_km2 = lambda wb: np.array([0.5 if int(str(w)[2:]) % 2 else 5.0 for w in wb])   # "Lb<i>"
    return mod


@pytest.fixture(scope="module")
def data(S):
    df, pix, feats, carr = _synth()
    return S.prepare(df, pix, feats, carr, "base")


@pytest.fixture(scope="module")
def fitted(S, data, tmp_path_factory):
    """Every stage once, in the script's order."""
    root = tmp_path_factory.mktemp("cmp")
    b = S.Budget(60, "test_part0")
    for stage in ("knn", "season", "gbm_single", "stack", "gbm_stacked", "mdn"):
        assert S.run_stage(stage, data, root, "cpu", b) == (True, None)
    sel = S.run_stage("select", data, root, "cpu", b)[1]
    done, res = S.run_stage("finalize", data, root, "cpu", b, ntu=S.ntu_class(data.df), n_boot=50)
    assert done
    return root, sel, res


# ------------------------------------------------------------------------------------------------- constants, rows

def test_constants_are_the_ruled_ones_and_agree_with_the_power_script(S):
    P = _load("orbitv2_power", "scripts/train/orbitv2_power.py")
    assert S.BASE_CELL == P.BASE_CELL and S.LAKE_CELLS == P.TRACK_N_CELLS
    assert (S.TRACK_N_MIN_ROWS, S.TRACK_N_MIN_BLOCKS) == (P.TRACK_N_MIN_ROWS, P.TRACK_N_MIN_BLOCKS) == (200, 20)
    assert S.TRACK_N == P.FAMILIES["TrackN"]["targets"]
    assert set(P.BASE_COMPARATORS) | set(P.TRACK_N_COMPARATORS) == set(S.CLAIM)
    assert S.track_n_cell(200, 20) == "valxval_lake"
    assert S.track_n_cell(199, 20) == S.track_n_cell(200, 19) == "valxany_lake"
    full = S.full_limits()
    assert len(full["grid"]) == 16 and full["feature_sets"] == C.GBM_FEATURE_SETS and full["ks"] == (4, 8, 16, 32)
    assert len(full["mdn"]) == 36 and full["member_seeds"] == (0, 1, 2) and full["selected_seeds"] == tuple(range(9))
    assert full["max_rounds"] == 3000 and full["threads"] <= 6                              # R-14, K-3
    assert S.STAGES == ("knn", "season", "gbm_single", "stack", "gbm_stacked", "mdn", "select", "finalize")
    assert (S.MIN_FREE_GB, S.SMALL_LAKE_KM2, S.ARM_SCHEMES) == (60.0, 1.0, ("base", "recent"))
    lead = REPO / "docs" / "preregistration" / "orbit_arch_v2_lead_clarification_01.md"
    assert S.CLARIFICATION_01 == __import__("hashlib").sha256(lead.read_bytes()).hexdigest()


def test_rows_fit_inner_stored_and_folds(S, data):
    d = data
    assert d.targets == CLAIM7 and d.claim == CLAIM7                      # the three rare heads are dropped (C-4)
    assert all(d.head_counts[t] < 200 for t in ("turbidity_fnu", "ssc", "acdom440"))
    assert not (d.fit & d.inner).any() and not (d.fit & d.keep).any() and (d.keep == ~d.fit).all()
    assert not d.added[d.inner].any()                                     # both arms stop on dev_v5 rows (C-3)
    assert d.added[d.fit].any()                                           # the fit cell holds the added rows
    for m in d.cells.values():
        assert m.any() and not (m & (d.fit | d.inner)).any() and (d.df["role"].to_numpy()[m] == "val").all()
    assert (d.folds[~d.fit] == -1).all() and set(d.folds[d.fit]) == set(range(5))
    assert pd.Series(d.folds[d.fit]).groupby(d.block[d.fit]).nunique().max() == 1
    assert (d.state == 2).sum() == d.cens.sum() > 0 and np.isnan(d.yq[d.cens]).all() and np.isfinite(d.y[d.cens]).all()
    jc, jp = d.targets.index("chla"), d.targets.index("tp")
    assert not d.added[S.score_mask(d, "chla", "valxval_lake")].any()     # a base target is scored on dev_v5 rows
    assert np.isfinite(d.yq[S.score_mask(d, "tp", "valxany_lake"), jp]).all()
    assert S.selection_cell(d, "chla") == "valxval_lake" and S.selection_cell(d, "tss") == "valxany_lake"
    assert S.selection_cell(d, "tp") == "valxany_lake"                    # fewer than 200 rows in valxval_lake
    assert S.variants("tp") == C.CENSORING and S.variants("chla") == (None,)
    lab = {v: S.labels(d, "tp", v) for v in C.CENSORING}
    c = d.cens[:, jp]
    assert np.isnan(lab["quantified"][c]).all() and np.allclose(lab["dl_sqrt2"][c], d.y[c, jp] - 0.5 * np.log10(2))
    assert np.array_equal(S.labels(d, "chla", None), d.yq[:, jc], equal_nan=True)
    df2 = d.df.copy()
    df2.loc[np.flatnonzero(d.cells["valxval_lake"])[0], ["role", "cell"]] = ["val", "trainxtrain"]
    with pytest.raises(C.ComparatorError, match="validation-cell row"):
        S.prepare(df2, {"refl": np.zeros((d.n, 1, 12)), "n_pix": np.ones(d.n), "n_water_win": np.ones(d.n)},
                  d.feats, d.carrays, "base")


# ------------------------------------------------------------------------------------------------- families

def _meta(f):
    return json.loads(str(np.load(f, allow_pickle=False)["meta"]))


def test_every_family_member_has_a_file_and_predicts_every_stored_row(S, data, fitted):
    root, d = fitted[0], data
    n = {k: len(list((root / k).rglob("*.npz"))) for k in ("knn", "season", "gbm", "stack", "mdn")}
    tv = 4 + 2 * 3                                                         # target x censoring variant (R-14b)
    assert n == {"knn": 7 * 2, "season": tv * 2, "gbm": tv * 2 * 2 * 2, "stack": 2 * 7, "mdn": 7 + 1}
    for f in sorted(root.rglob("*.npz")):
        if "final" in f.parts:
            continue
        z, meta = np.load(f, allow_pickle=False), _meta(f)
        assert str(z["rows_sha"]) == d.rows_sha and "skipped" not in meta and meta["parts"] == ["test_part0"]
        assert z["pred"].shape[-2 if z["pred"].ndim == 3 else 0] == d.keep.sum() and np.isfinite(z["pred"]).all(), f
    m = _meta(S.mfile(root, "gbm", "tp", "gbm[v1|single|g01|dl_sqrt2]"))
    assert (m["feature_set"], m["mode"], m["grid"], m["censoring"], m["seeds"]) == ("v1", "single", 1, "dl_sqrt2", [0])
    assert len(m["rounds"]) == 1 and np.isfinite(m["inner_mdsa"])
    joint = np.load(S.mfile(root, "mdn", "joint", S.mdn_trained("12band+context+ring", 150, True, "joint")))["pred"]
    assert joint.shape == (2, d.keep.sum(), 7)                             # both read-outs, every head


def test_stacking_features_are_out_of_fold_on_the_fit_cell(S, data, fitted):
    root, d = fitted[0], data
    st = S.load_stack(d, root, "v1")
    assert sorted(st) == sorted(d.targets) and all(np.isfinite(p).all() and len(p) == d.n for p in st.values())
    f = S.stack_file(root, "v1", "chla")
    meta, z = _meta(f), np.load(f, allow_pickle=False)
    scores = {S.gbm_name(stem, None): _meta(S.mfile(root, "gbm", "chla", S.gbm_name(stem, None)))["inner_mdsa"]
              for stem, fs, _, _ in S.gbm_family("gbm_single") if fs == "v1"}
    assert meta["member"] == min(scores, key=scores.get)                  # the inner-best single-target member
    X = d.X["v1"].to_numpy(float)
    with open(f.with_suffix(".pkl"), "rb") as fh:
        model = pickle.load(fh)["predict"]
    assert np.allclose(model(X[d.keep]), z["pred"])                       # outside the fit cell: the kept model
    full_fit = model(X[d.fit])
    assert not np.allclose(full_fit, z["oof"])                            # on the fit cell: out-of-fold, not in-sample
    fi = np.flatnonzero(d.fit)
    te = d.folds[d.fit] == 0
    rows = np.zeros(d.n, bool)
    rows[fi[~te]] = True
    again = S.fit_bag(d, X, "chla", None, meta["grid"], (0,), rows=rows)(X[fi[te]])
    assert np.allclose(again, z["oof"][te])                                # fold 0 comes from a fit without fold 0
    Xs = S.gbm_matrix(d, root, "v1", "stacked", "chla")
    assert Xs.shape[1] == X.shape[1] + len(d.targets) - 1                  # every other target, not the target
    assert np.array_equal(S.gbm_matrix(d, root, "season", "single", "chla"), d.Xseason)


def test_selection_uses_the_selection_cell_and_the_lowest_score(S, data, fitted):
    root, sel, _ = fitted
    d = data
    assert list(sel) == CLAIM7 and json.loads((root / "selection.json").read_text())["selection"].keys() == sel.keys()
    ms = pd.read_parquet(root / "member_scores.parquet")
    for t, rec in sel.items():
        assert rec["cell"] == S.selection_cell(d, t) and rec["n"] == S.score_mask(d, t, rec["cell"]).sum() > 0
        assert set(rec["families"]) == {"gbm", "season_loc_gbm", "knn", "mdn_b"}
        nv = len(S.variants(t))
        assert {k: v["n_members"] for k, v in rec["families"].items()} == {
            "gbm": 2 * 2 * 2 * nv, "season_loc_gbm": 2 * nv, "knn": 2, "mdn_b": 2 * 2}
        for fam, r in rec["families"].items():
            g = ms[(ms.target == t) & (ms.family == fam)]
            assert r["n_scored"] == len(g) and g.mdsa.notna().all()        # every member predicts every row
            assert g.loc[g.selected, "member"].tolist() == [r["selected"]] and np.isclose(g.mdsa.min(), r["mdsa"])
            assert (g.cell == rec["cell"]).all()
    jt = d.targets.index("tp")
    p, _ = S.load_member(S.mfile(root, "knn", "tp", "knn[k04]"), d)
    one = ms[(ms.target == "tp") & (ms.member == "knn[k04]")].mdsa.iloc[0]
    assert np.isclose(one, C.val_mdsa(S.expand(d, p), d.yq[:, jt], S.score_mask(d, "tp", "valxany_lake")))


def test_final_comparators_average_chain_and_validation_rows(S, data, fitted):
    root, sel, res = fitted
    d, val, summ = data, res["val"], res["summary"]
    P = _load("orbitv2_power", "scripts/train/orbitv2_power.py")
    assert set(P.REQUIRED) <= set(val.columns) and {f"cmp:{c}" for c in S.CLAIM} <= set(val.columns)
    assert val.equals(pd.read_parquet(root / "val_rows.parquet")) and summ["n_val_rows"] == len(val)
    for (t, cell), g in val.groupby(["target", "cell"]):
        m = S.score_mask(d, t, cell)
        assert g.row_id.tolist() == d.row_id[m].tolist() and np.allclose(g.y, d.yq[m, d.targets.index(t)])
        assert np.allclose(g["cmp:mdn_b+gbm"], 0.5 * (g["cmp:mdn_b"] + g["cmp:gbm"]))          # R-10
        for c in ("mdn_b", "gbm", "knn", "season_loc_gbm"):
            assert g[f"cmp:{c}"].notna().all()
        assert g["cmp:trophic_ols"].notna().all() == (t in S.TRACK_N)
        assert not g.added.any() if t in D.TARGETS else g.added.any() or cell == "valxval_lake"
    small = S.small_lake(d.df)
    assert small.any() and not small.all()
    assert val.small_lake.tolist() == [bool(small[i]) for i in pd.Series(np.arange(d.n), index=d.row_id)[val.row_id]]
    tu = val[val.target == "turbidity"]
    assert tu.ntu_class.any() and not tu.ntu_class.all()
    # the 9-seed bag (2 seeds here) of the selected configuration, kept as a model
    s = sel["tp"]["families"]["gbm"]
    with open(root / "final" / "gbm" / "tp" / "bag9.pkl", "rb") as fh:
        bag = pickle.load(fh)
    assert bag["spec"]["name"] == s["selected"] and bag["predict"].seeds == [0, 1]
    X = S.gbm_matrix(d, root, s["spec"]["feature_set"], s["spec"]["mode"], "tp")
    stored = S.load_member(root / "final" / "gbm" / "tp" / "bag9.npz", d)[0]
    assert np.allclose(bag["predict"](X[d.keep]), stored)
    # R-11 / R-14b: two chains x two censoring variants, fitted on out-of-fold chl-a and Secchi predictions
    for t in S.TRACK_N:
        ch = summ["chains"][t]
        assert set(ch["scores"]) == {f"trophic_ols[{c}|{v}]" for c in S.CHAINS for v in C.CENSORING}
        assert ch["selected"] == min(ch["scores"], key=ch["scores"].get)
    with open(root / "bundle.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    b = bundle["comparators"]["tp"]["trophic_ols"]
    chain, v = b["chain"], b["censoring"]
    z = {}
    for zt in S.CHAIN_INPUTS:
        f = S.mfile(root / "final", "oof", chain, zt)
        zz = np.load(f, allow_pickle=False)
        full = S.expand(d, zz["pred"])
        full[d.fit] = zz["oof"]
        z[zt] = full
        assert np.isfinite(zz["oof"]).all() and len(zz["oof"]) == d.fit.sum()
    coef = C.ols_fit(z["chla"][d.fit], z["secchi"][d.fit], S.labels(d, "tp", v)[d.fit])
    assert np.allclose(coef, b["coef"])
    g = val[(val.target == "tp") & (val.cell == "valxany_lake")]
    m = S.score_mask(d, "tp", "valxany_lake")
    assert np.allclose(g["cmp:trophic_ols"], C.ols_predict(coef, z["chla"][m], z["secchi"][m]))
    assert set(bundle["comparators"]["chla"]) == {"gbm", "season_loc_gbm", "knn", "mdn_b"}
    assert all((root / c["file"]).exists() for t in CLAIM7 for k, c in bundle["comparators"][t].items() if "file" in c)
    # hashes of every comparator file
    sums = dict(reversed(line.split("  ")) for line in (root / "SHA256SUMS").read_text().splitlines())
    assert "bundle.pkl" in sums and "val_rows.parquet" in sums and "selection.json" in sums
    assert json.loads((root / "comparators.json").read_text())["sha256sums_sha256"] == summ["sha256sums_sha256"]
    # the power script reads the table once the arm columns are joined
    vp = val.assign(**{f"arm:{a}": val.y for a in ("v1", "v2", "track_n")})
    P.check_table(vp)


def test_a_stage_resumes_per_member_and_refuses_other_rows(S, data, fitted, tmp_path):
    d = data
    b0 = S.Budget(0, "p0")
    assert S.run_stage("knn", d, tmp_path, "cpu", b0) == (False, None) and not list(tmp_path.rglob("*.npz"))
    old = dict(S.LIM)
    try:
        S.LIM = dict(old, ks=(4,))
        assert S.run_knn(d, tmp_path, S.Budget(5, "p1"))
        S.LIM = old
        f4, f8 = (S.mfile(tmp_path, "knn", "tp", f"knn[k{k:02d}]") for k in (4, 8))
        t4 = f4.stat().st_mtime_ns
        b = S.Budget(5, "p2")
        assert S.run_knn(d, tmp_path, b) and b.n == 7                     # only the missing members are fitted
    finally:
        S.LIM = old
    assert f4.stat().st_mtime_ns == t4 and _meta(f4)["parts"] == ["p1"] and _meta(f8)["parts"] == ["p2"]
    (f8.with_suffix(".started")).write_text("p1\np2\n")                  # a member a stopped job had started
    S.save_member(f8, d, np.zeros(d.keep.sum()), name="knn[k08]")
    assert _meta(f8)["parts"] == ["p1", "p2"]                             # K-3: its record names both parts
    other = SimpleNamespace(**dict(vars(d), rows_sha="0" * 64))
    with pytest.raises(SystemExit, match="other rows"):
        S.load_member(f4, other)
    with pytest.raises(SystemExit, match="earlier stage is not complete"):
        S.select(d, tmp_path)
    assert S.finalize.__defaults__ is None                                 # the time budget is not optional
    assert S.run_stage("finalize", d, fitted[0], "cpu", S.Budget(0), ntu=S.ntu_class(d.df), n_boot=10)[0] is True
    for f in (fitted[0] / "final" / "oof").rglob("mdn_b/secchi.*"):        # a missing piece and no time left
        f.rename(f.with_name(f.name + ".away"))
    assert S.run_stage("finalize", d, fitted[0], "cpu", S.Budget(0), ntu=S.ntu_class(d.df)) == (False, None)
    for f in (fitted[0] / "final" / "oof").rglob("*.away"):
        f.rename(f.with_name(f.name[:-5]))


# ------------------------------------------------------------------------------------------------- registry, gates

def test_registry_rows_carry_the_fit_scheme_and_the_v1_selection_row(S, fitted, tmp_path):
    from sentinel.core.registry import log_result, read_registry

    root, _, res = fitted
    reg = tmp_path / "registry.jsonl"
    v1 = log_result(workstream="M2-orbit", experiment="orbit_select_dev_v5", model="SENTINEL", metric="mdsa",
                    value=30.0, split="valxval_lake@dev_v5", n=10, registry_path=reg)
    other = log_result(workstream="M1-sonde", experiment="x", model="m", metric="mdsa", value=1.0, split="s", n=1,
                       registry_path=reg)
    v1id = v1["id"] if isinstance(v1, dict) else v1
    oid = other["id"] if isinstance(other, dict) else other
    row = S.v1_selection(v1id, reg)
    assert row["id"] == v1id and row["experiment"] == "orbit_select_dev_v5"
    for bad in (oid, "nope"):
        with pytest.raises(SystemExit, match="not a registry row"):
            S.v1_selection(bad, reg)
    plan = {"fit_scheme": "base", "v1_selection_row": row}
    n = S.register(res, root, "dev_v5m", 50, "stamprow", plan, registry_path=reg)
    b = S.Budget(1, "knn_part0")
    b.n = 3
    assert S.register_job("knn", "knn_part0", b, True, 0.01, "cpu", "dev_v5m", "stamprow", plan, reg) >= 0
    rows = [r for r in read_registry(reg) if r["experiment"] == "orbit_v2_cmp_dev_v5m"]
    val = [r for r in rows if r["metric"] == "mdsa"]
    assert n == len(val) == len(res["scores"]) and all(r["is_baseline"] and r["model"].startswith("cmp:") for r in val)
    assert {r["target"] for r in val} == set(CLAIM7) and {r["split"] for r in val} == {
        "valxval_lake@dev_v5m", "valxany_lake@dev_v5m"}
    for r in rows:
        assert r["config"]["fit_scheme"] == "base" and r["config"]["v1_selection_row"]["id"] == v1id
        assert r["extra"]["stamp_row"] == "stamprow" and r["extra"]["clarification_01_sha256"] == S.CLARIFICATION_01
    job = [r for r in rows if r["metric"] == "cpu_hours"]
    assert len(job) == 1 and job[0]["model"] == "job:knn_part0" and job[0]["n"] == 3 and job[0]["extra"]["complete"]
    assert sum(r["model"] == "cmp:trophic_ols" for r in val) == 2 * 3     # TP, TN, DOC in both lake cells
    assert all(r["extra"]["chain"] for r in val if r["model"] == "cmp:trophic_ols")


def test_fit_scheme_rule_of_clarification_01(S, tmp_path):
    assert S.resolve_fit_scheme("recent", None, "gbm_single") == "recent"        # before R-19: the value passed
    assert S.resolve_fit_scheme("base", "base", "finalize") == "base"
    for stage in ("select", "finalize"):
        with pytest.raises(SystemExit, match="no fit scheme yet"):
            S.resolve_fit_scheme("recent", None, stage)
    for stage in ("knn", "mdn", "select", "finalize"):                             # after R-19: the arms' value only
        with pytest.raises(SystemExit, match="refitted in full"):
            S.resolve_fit_scheme("recent", "base", stage)
    with pytest.raises(SystemExit, match="one of"):
        S.resolve_fit_scheme("track_u", None, "knn")
    assert S.arm_fit_scheme() is None                                      # as shipped: R-19 is not met
    for i, v in enumerate(("recent", "recent")):
        (tmp_path / f"a{i}.yaml").write_text(f"base:\n  fit_scheme: {v}\n")
    assert S.arm_fit_scheme(tmp_path) == "recent"
    (tmp_path / "a1.yaml").write_text("base:\n  fit_scheme: base\n")
    with pytest.raises(SystemExit, match="different fit schemes"):
        S.arm_fit_scheme(tmp_path)
    # a first fit under another scheme is kept, marked, and lives in its own folder
    for sch in ("base", "recent"):
        (tmp_path / "c" / sch).mkdir(parents=True)
    (tmp_path / "c" / "base" / "plan.json").write_text("{}")
    assert S.mark_superseded(tmp_path / "c", "recent") == ["base"]
    assert json.loads((tmp_path / "c" / "base" / "SUPERSEDED.json").read_text())["superseded_by_fit_scheme"] == "recent"
    assert S.mark_superseded(tmp_path / "c", "base") == [] and (tmp_path / "c" / "base" / "plan.json").exists()


class _Run:
    def __init__(self, *a, **k):
        self.notes, self.metrics, self.name = [], {}, a[1]
        _Run.last = self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def note(self, s):
        self.notes.append(s)

    def metric(self, k, v):
        self.metrics[k] = v


def test_script_gates_and_smoke_run(S, tmp_path, monkeypatch):
    import sentinel.core.runlog as RL

    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(S.PL, "setup_gpu", lambda *a, **k: pytest.fail("a GPU was requested"))
    monkeypatch.setattr(RL, "TrainingRun", _Run)
    unst = {"stamped": False, "amendment_sha256": "0" * 64, "row_id": None, "timestamp": None, "n_rows": 0,
            "rulings_sha256": "0" * 64, "reason": "test"}
    monkeypatch.setattr(ST, "stamp_status", lambda *a, **k: unst)
    with pytest.raises(ST.StampError, match="refused before the lead's stamp"):
        S.main(["--stage", "knn", "--fit-scheme", "base", "--v1-selection-row", "x"])
    monkeypatch.setattr(ST, "stamp_status", lambda *a, **k: dict(unst, stamped=True, row_id="r1"))
    with pytest.raises(SystemExit, match="names the v1 selection row"):
        S.main(["--stage", "knn", "--fit-scheme", "base"])
    for stage in ("select", "finalize"):                                   # as shipped the arm files name no scheme
        with pytest.raises(SystemExit, match="no fit scheme yet"):
            S.main(["--stage", stage, "--fit-scheme", "base", "--v1-selection-row", "x"])
        with pytest.raises(SystemExit, match="no validation number"):
            S.main(["--stage", stage, "--fit-scheme", "base", "--smoke", "--data-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="not a registry row"):
        S.main(["--stage", "knn", "--fit-scheme", "base", "--v1-selection-row", "no-such-row"])
    for extra in (["--stage", "knn", "--data-dir", str(tmp_path)], ["--stage", "mdn", "--device", "cpu"]):
        with pytest.raises(SystemExit, match="smoke tests"):
            S.main(["--fit-scheme", "base", "--v1-selection-row", "x", *extra])
    with pytest.raises(SystemExit, match="needs --data-dir"):
        S.main(["--stage", "knn", "--fit-scheme", "base", "--smoke"])
    with pytest.raises(SystemExit):                                        # argparse: the scheme is not optional
        S.main(["--stage", "knn", "--smoke", "--data-dir", str(tmp_path)])
    assert not (tmp_path / "runs").exists()                                # every refusal comes before any write
    monkeypatch.setattr(S.shutil, "disk_usage", lambda p: type("U", (), {"free": 62.9e9})())
    with pytest.raises(SystemExit, match="R-25"):
        S.check_disk()
    monkeypatch.setattr(S.shutil, "disk_usage", lambda p: type("U", (), {"free": 63.1e9})())
    assert S.check_disk() > 63
    # smoke run: one configuration per family, inner rows only, nothing registered, no validation row stored
    df, pix, feats, carr = _synth(n=900, n_added=120)
    info = {"version": "dev_v5m", "sha256": {S.DM.TABLE: "ab" * 32}}
    monkeypatch.setattr(S.DM, "load", lambda *a, **k: (df, pix, feats, (None, carr), info))
    monkeypatch.setattr(S, "register", lambda *a, **k: pytest.fail("a smoke test registered a row"))
    monkeypatch.setattr(S, "register_job", lambda *a, **k: pytest.fail("a smoke test registered a row"))
    lim = dict(S.LIM)
    try:
        for stage in ("knn", "gbm_single"):
            assert S.main(["--stage", stage, "--fit-scheme", "base", "--smoke", "--data-dir", str(tmp_path)]) == 0
            assert _Run.last.name.startswith("SMOKE/") and any("NOT a result" in s for s in _Run.last.notes)
        assert S.LIM["grid"] == (0,) and S.LIM["max_rounds"] == 20
        d = S.prepare(df, pix, feats, carr, "base", keep="inner")
    finally:
        S.LIM = lim
    root = tmp_path / "runs" / "orbit" / "smoke" / "v2" / "comparators"
    assert json.loads((root / "plan.json").read_text())["fit_scheme"] == "base"
    files = sorted(root.rglob("*.npz"))
    assert len(files) == 7 + 4 + 2 * 3
    for f in files:
        assert len(np.load(f)["pred"]) == d.inner.sum() and str(np.load(f)["rows_sha"]) == d.rows_sha
    assert not any((d.keep & m).any() for m in d.cells.values())           # no validation-cell row is predicted
    assert json.loads((root / "part_knn_part0.json").read_text())["complete"] is True
    assert not (root / "selection.json").exists() and not (root / "val_rows.parquet").exists()


# ------------------------------------------------------------------------- prediction of rows no stage has seen

def test_predict_rows_reproduces_the_validation_rows_from_the_bundle_alone(S, data, fitted):
    """The Test-S path: the rows are handed over as a separate table (another order, no fit row among them) and
    predicted from ``bundle.pkl`` and the files it names; every claim comparator equals the stored validation
    value."""
    root, _, res = fitted
    df, pix, feats, carr = _synth()
    assert df.row_id.astype(str).tolist() == data.row_id.tolist()
    idx = np.flatnonzero(data.cells["valxany_lake"])[::-1]                     # reversed order
    n = len(df)

    def sub(dct):
        return {k: (np.asarray(v)[idx] if hasattr(v, "__len__") and not isinstance(v, str) and len(v) == n else v)
                for k, v in dct.items()}
    with open(root / "bundle.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    assert len(bundle["knn_fit"]["lat"]) == data.fit.sum() == len(bundle["comparators"]["tp"]["knn"]["y_fit"])
    new = df.iloc[idx].reset_index(drop=True)
    x = S.features(new, sub(pix), sub(feats), carr, S.bundle_feature_sets(bundle))
    assert x.n == len(idx) and set(x.X) == set(S.bundle_feature_sets(bundle))
    got = S.predict_rows(root, x)
    assert set(got) == set(bundle["claim_targets"]) == set(CLAIM7)
    pos = pd.Series(np.arange(len(new)), index=new.row_id.astype(str))
    val, n_checked = res["val"], 0
    for (t, cell), g in val.groupby(["target", "cell"]):
        assert set(got[t]) == ({*S.CLAIM} if t in S.TRACK_N else {*S.CLAIM} - {"trophic_ols"})
        for c, p in got[t].items():
            assert np.allclose(p[pos[g.row_id].to_numpy()], g[f"cmp:{c}"].to_numpy(float), rtol=1e-5, atol=1e-6), (t, c)
            n_checked += len(g)
    assert n_checked > 0
    # a stacked selection reads the stacking members named by the bundle (and no stored prediction)
    modes = {bundle["comparators"][t]["gbm"]["spec"]["mode"] for t in CLAIM7}
    for t in CLAIM7:
        e = bundle["comparators"][t]["gbm"]
        assert ("stack_files" in e) == (e["spec"]["mode"] == "stacked")
    assert modes <= {"single", "stacked"}


def test_verify_sums_names_a_changed_or_missing_file(S, fitted, tmp_path):
    root, _, res = fitted
    # (an earlier test ran finalize again with another bootstrap size: comparators.json holds the current value)
    assert S.verify_sums(root) == json.loads((root / "comparators.json").read_text())["sha256sums_sha256"]
    import hashlib
    (tmp_path / "final").mkdir()
    (tmp_path / "bundle.pkl").write_bytes(b"a")
    (tmp_path / "final" / "m.pkl").write_bytes(b"b")
    with pytest.raises(SystemExit, match="no SHA256SUMS"):
        S.verify_sums(tmp_path)
    lines = {p: hashlib.sha256((tmp_path / p).read_bytes()).hexdigest() for p in ("bundle.pkl", "final/m.pkl")}
    (tmp_path / "SHA256SUMS").write_text("".join(f"{h}  {p}\n" for p, h in lines.items()))
    assert S.verify_sums(tmp_path) == hashlib.sha256((tmp_path / "SHA256SUMS").read_bytes()).hexdigest()
    (tmp_path / "final" / "m.pkl").write_bytes(b"c")
    with pytest.raises(SystemExit, match="missing or changed"):
        S.verify_sums(tmp_path)
    (tmp_path / "final" / "m.pkl").unlink()
    with pytest.raises(SystemExit, match="missing or changed"):
        S.verify_sums(tmp_path)
    (tmp_path / "SHA256SUMS").write_text(f"{lines['final/m.pkl']}  final/m.pkl\n")
    with pytest.raises(SystemExit, match="bundle.pkl is not listed"):
        S.verify_sums(tmp_path)
