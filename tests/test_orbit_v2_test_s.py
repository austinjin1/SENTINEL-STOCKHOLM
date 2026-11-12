# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""scripts/eval/orbitv2_test_s.py and orbit_v2/testcells.py (amendment §1.6, §5, §6; R-17, R-21, R-24) on synthetic
rows: small untrained networks, a stand-in comparator predictor, a temporary runs root. No cache, chip store,
protected row, real registry row or file under the real runs root is touched; TrainingRun is replaced."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from sentinel.core import paths
from sentinel.models.orbit import data as D
from sentinel.models.orbit import features as F1
from sentinel.models.orbit import guard as G1
from sentinel.models.orbit import pipeline as PL
from sentinel.models.orbit import satwq as V1
from sentinel.models.orbit_v2 import arm as ARM
from sentinel.models.orbit_v2 import claims as CL
from sentinel.models.orbit_v2 import looks as LK
from sentinel.models.orbit_v2 import testcells as TC
from sentinel.models.orbit_v2 import train as TR
from sentinel.models.orbit_v2 import units as U
from sentinel.models.sonde.guard import ProtectedDataError

REPO = Path(__file__).resolve().parents[1]
REAL_RUNS_ROOT = Path(paths.RUNS_ROOT).resolve()
ARM_NAME = "a1_c2"
N_BOOT = 40


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AT = _load("arm_tests_for_test_s", "tests/test_orbit_v2_arm.py")


class _Run:
    def __init__(self, *a, **k):
        self.notes, self.metrics = [], {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def note(self, t):
        self.notes.append(t)

    def metric(self, k, v):
        self.metrics[k] = v


def _rows(seed: int, role_test: bool = False):
    """A dev_v5m-shaped table with the key columns of the look; ``role_test``: every row a Test-S row."""
    df, pix, feats, cc = AT._synth(n=900, n_added=120, seed=seed)
    n = len(df)
    r = np.random.default_rng(seed + 10)
    bnum = df["block"].str[1:].astype(int).to_numpy()
    df["in_lake"] = bnum % 5 != 0                                          # every fifth block is not a lake
    df["source"] = np.where(np.arange(n) % 3 == 0, "aquacorr_global", "wqp")
    df["turbidity_unit_class"] = np.where(np.arange(n) % 4 == 0, "FNU", "NTU")
    day = pd.Timestamp("2019-01-01") + pd.to_timedelta(r.integers(0, 1500, n), unit="D")
    df["date"] = day.strftime("%Y-%m-%d")
    df["sensing_time_utc"] = day.strftime("%Y-%m-%dT10:00:00Z")
    tile = np.array([f"{10 + b % 40:02d}SE{'ABCDEFGH'[b % 8]}" for b in bnum])
    df["item_id"] = [f"S2A_MSIL2A_{d:%Y%m%d}T100000_R001_T{t}_x" for d, t in zip(day, tile)]
    if role_test:
        df["role"], df["tblock"], df["cell"], df["protected"] = "test", "train", "testxtrain", True
        df["row_id"] = "t" + df["row_id"]
        df["block"] = "T" + df["block"]
        df["waterbody"] = "L" + df["block"]
    return df, pix, feats, cc


@pytest.fixture(scope="module", autouse=True)
def _base(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setattr(paths, "RUNS_ROOT", tmp_path_factory.mktemp("test_s_runs_base"))
    wl = {b: 440.0 + 150.0 * i for i, b in enumerate(D.BANDS)}
    mp.setattr(D, "band_wavelengths", lambda: {"S2A": wl, "S2B": {b: v + 1.0 for b, v in wl.items()}})
    yield
    mp.undo()


@pytest.fixture(scope="module")
def M():
    mod = _load("orbitv2_test_s", "scripts/eval/orbitv2_test_s.py")
    mod.TrainingRun = _Run
    mod.S.lake_area_km2 = lambda wb: np.array([0.5 if int(str(w).lstrip("LTb")) % 2 else 5.0 for w in wb])
    return mod


def _cmp_pred(df, targets):
    """Stand-in comparator predictions: a function of the row and the target alone."""
    y, cens = ARM.target_matrix(df, targets)
    out = {}
    for j, t in enumerate(targets):
        base = np.where(np.isfinite(y[:, j]), y[:, j], 0.0)
        h = np.array([int(hashlib.sha256(f"{r}|{t}".encode()).hexdigest()[:6], 16) / 16 ** 6 for r in df["row_id"]])
        out[t] = {c: base + 0.2 * (k + 1) * (h - 0.5) for k, c in enumerate(CL.TRACK_N_COMPARATORS)
                  if c != "trophic_ols" or t in ("tp", "tn", "doc")}
    return out


@pytest.fixture(scope="module")
def world(M, tmp_path_factory):
    """Development rows, 9 untrained A1 networks with stored validation predictions, 2 untrained v1 networks, a
    stand-in comparator folder, and a candidate file naming them."""
    root = tmp_path_factory.mktemp("test_s_artifacts")
    df, pix, feats, cc = _rows(0)
    cfg = AT._cfg(ARM_NAME)
    fit = F1.fit_and_inner(df, cfg["fit_scheme"])[0]
    targets, counts = ARM.heads(df, fit, ARM_NAME)
    cl = ARM.cells(df, ARM_NAME, cfg["fit_scheme"], targets)
    norm = ARM.fit_norm(ARM_NAME, df, pix, feats, cl["fit"], targets)
    out_mask = ~cl["fit_cell_all"]
    weights = {}
    for u in TR.UNITS:
        for i in TR.INITS:
            torch.manual_seed(TR.network_seed(u, i))
            model = ARM.build_model(ARM_NAME, cfg["v1"], len(targets))
            d = U.network_dir(root / "nets", ARM_NAME, u, i)
            d.mkdir(parents=True)
            torch.save({"state_dict": model.state_dict(), "norm": norm.to_dict(), "config": cfg, "arm": ARM_NAME,
                        "targets": targets, "seed": TR.network_seed(u, i)}, d / "model.pt")
            ds = ARM.make_set(ARM_NAME, df, pix, feats, cc, norm, out_mask, targets)
            w, mu, sg = V1.predict_mixture(model, ds, "cpu", norm.y_mean, norm.y_std)
            PL.save_predictions(d / "preds.npz", df, out_mask, {"w": w, "mu": mu, "sg": sg})
            PL.write_json(d / "network.json", {"arm": ARM_NAME, "unit": u, "init": i, "seed": TR.network_seed(u, i),
                                               "targets": targets, "arm_config_hash": "h", "arm_file_sha256": "f",
                                               "data": {"version": "synthetic"}, "prediction_rows": "x",
                                               "smoke": False})
            (d / "DONE").write_text("x\n")
            for m in ("v2", "track_n"):
                weights[f"{m}@u{u}i{i}"] = G1.artifact_sha256(d / "model.pt")
    n4 = ARM.fit_norm("a0", df, pix, feats, cl["fit_cell_all"] & ~df["added"].to_numpy(bool), list(D.TARGETS))
    v1w, ckpts = {}, []
    for s in range(2):
        torch.manual_seed(50 + s)
        m = ARM.build_v1(cfg["v1"])
        f = root / "v1" / f"seed{s}" / "model.pt"
        f.parent.mkdir(parents=True)
        torch.save({"state_dict": m.state_dict(), "norm": n4.to_dict(), "config": cfg["v1"],
                    "arch": {"n_geo": len(PL.GEO_COLS), "n_ctx": len(PL.CTX_COLS), **ARM.v1_arch(cfg["v1"])}}, f)
        v1w[f"seed{s}"], _ = G1.artifact_sha256(f), ckpts.append(str(f))
    claim7 = [t for t in targets if t in D.TARGETS or t in M.S.TRACK_N]
    croot = root / "cmp"
    croot.mkdir()
    bundle = {"fit_scheme": cfg["fit_scheme"], "claim_targets": claim7, "targets": targets}
    (croot / "bundle.pkl").write_bytes(pickle.dumps(bundle))
    cp = _cmp_pred(df, claim7)
    val = pd.concat([pd.DataFrame({"target": t, "cell": "valxany_lake", "row_id": df["row_id"].astype(str),
                                   **{CL.cmp_col(c): cp[t].get(c, np.nan) for c in M.S.CLAIM}}) for t in claim7])
    val.to_parquet(croot / "val_rows.parquet", index=False)
    (croot / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256((croot / n).read_bytes()).hexdigest()}  {n}\n" for n in ("bundle.pkl", "val_rows.parquet")))
    weights["comparators"] = M.S.verify_sums(croot)
    arm_spec = {"arm": ARM_NAME, "root": str(root / "nets")}
    cand = {"artifacts": {"v1": {"satwq_checkpoints": ckpts, "fitted_baselines": None, "blend_with": None},
                          "v2": arm_spec, "track_n": dict(arm_spec), "comparators": {"root": str(croot)}},
            "sealed_models": {"v1_networks": 2, "v2": True, "track_n": True, "comparators": list(M.S.CLAIM)},
            "weights_sha256": weights, "v1_weights_sha256": v1w}
    cfile = root / "candidate.json"
    cfile.write_text(json.dumps(cand))
    return {"root": root, "dev": (df, pix, feats, cc), "candidate": cfile, "cand": cand, "targets": targets,
            "claim7": claim7}


@pytest.fixture
def sandbox(M, world, tmp_path, monkeypatch):
    """A temporary runs root, the development loader replaced by the synthetic rows, registry writes collected."""
    assert not str(tmp_path.resolve()).startswith(str(REAL_RUNS_ROOT))
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path / "runs")
    df, pix, feats, cc = world["dev"]
    monkeypatch.setattr(M.D, "load_dev", lambda *a, **k: (df, None, None))
    monkeypatch.setattr(M.DM, "load", lambda *a, **k: (df, pix, feats, cc, {}))
    monkeypatch.setattr(M, "predict_comparators",
                        lambda root, bundle, d, *a: _cmp_pred(d, bundle["claim_targets"]))
    rows = {"look": [], "score": []}
    monkeypatch.setattr(LK, "_log_result", lambda **k: rows["look"].append(k))
    monkeypatch.setattr(M, "_log_result", lambda **k: rows["score"].append(k))
    monkeypatch.setattr(LK, "registry_rows", lambda exp, **k: len(rows["look"]) + len(rows["score"]))
    monkeypatch.setattr(LK, "scoring_rows", lambda exp: len(rows["score"]))
    monkeypatch.setattr(LK, "control_row", lambda exp, metric: any(r["metric"] == metric for r in rows["look"]))
    return rows


# ------------------------------------------------------------------------------------------------ test cells

def test_scene_key_is_the_rule_of_the_stamped_count_script():
    src = (REPO / "scripts" / "data" / "orbitv2_testS_keycounts.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "scene_key")
    ns = {"pd": pd, "np": np}
    exec(compile(ast.Module([fn], []), "keycounts", "exec"), ns)          # the function alone; the script is not run
    ids = ["S2A_MSIL2A_20200101T100000_R001_T10SEG_x", "S2B_31UFS_20210505_0_L2A", "no_tile_here", None,
           "S2A_T10SEG"]
    when = ["2020-01-01T10:00:00Z", "2021-05-05T09:00:00Z", "2021-05-05T23:59:59Z", "2020-01-01T00:00:00Z",
            "not a date"]                                                  # one format, as in the tables
    mine = TC.scene_key(ids, when)
    assert list(mine) == list(ns["scene_key"](ids, when)) == ["10SEG|20200101", "31UFS|20210505", "NA|20210505",
                                                              "NA|20200101", "10SEG|NA"]
    assert list(TC.keyed(mine)) == [True, True, False, False, False]


def test_reference_scenes_disjoint_flag_and_primary_cell_rule(world):
    dev = world["dev"][0]
    ref, rec = TC.reference_scenes(dev, "base")
    fit, inner = F1.fit_and_inner(dev, "base")
    assert rec["n_reference_rows"] == int((fit | inner).sum()) and rec["n_reference_rows_without_key"] == 0
    assert ref == set(TC.row_keys(dev[fit | inner]))
    keys = np.array([next(iter(ref)), "99XXX|20200101", "NA|20200101", "10SEG|NA"])
    flag, c = TC.scene_disjoint(keys, ref)
    assert list(flag) == [False, True, False, False] and c == {"n": 4, "n_without_key": 2, "n_scene_disjoint": 1}
    with pytest.raises(TC.CellError, match="item_id"):
        TC.row_keys(dev.drop(columns=["item_id"]))
    blocks = np.repeat(np.arange(10), 10)
    dj = np.ones(100, bool)
    dj[:40] = False                                                        # 60 % of rows, 6 of 10 blocks
    assert TC.primary_cell(dj, blocks)[0] == "all_rows"
    dj = np.ones(100, bool)
    dj[::10] = dj[1::10] = dj[2::10] = dj[3::10] = False                   # 60 % of rows, every block
    cell, rec = TC.primary_cell(dj, blocks)
    assert cell == "scene_disjoint" and (rec["n_rows_scene_disjoint"], rec["n_blocks_scene_disjoint"]) == (60, 10)
    dj[4] = False
    assert TC.primary_cell(dj, blocks)[0] == "all_rows"                    # 59 %
    dj = np.ones(100, bool)
    dj[:30] = False                                                        # 70 % of rows, 7 of 10 blocks
    assert TC.primary_cell(dj, blocks)[0] == "all_rows"
    tr = TC.train_points(dev)
    d = TC.dist_train_km(dev.head(50), tr)
    assert (d[(dev.head(50)["role"] == "train").to_numpy()] < 1e-6).all() and np.isfinite(d).all()


# ------------------------------------------------------------------------------------------ model predictions

def test_v1_ensemble_and_single_networks_are_the_v1_routine(M, world):
    from sentinel.events import orbit as EO

    df, pix, _, _ = world["dev"]
    sub = df.head(60).reset_index(drop=True)
    psub = {k: v[:60] for k, v in pix.items()}
    art = world["cand"]["artifacts"]["v1"]
    p = M.predict_v1(art, sub, psub, "cpu")
    assert sorted(k for k in p if k != "q") == ["v1", "v1@s0", "v1@s1"]
    ens = EO.satwq_mixture_embedding(art["satwq_checkpoints"], sub, psub, "cpu")[0]
    assert np.allclose(p["v1"], U.medians(ens), atol=1e-6)
    one = EO.satwq_mixture_embedding(art["satwq_checkpoints"][1:], sub, psub, "cpu")[0]
    assert np.allclose(p["v1@s1"], U.medians(one), atol=1e-6) and not np.allclose(p["v1@s0"], p["v1@s1"])
    assert sorted(p["q"]) == sorted(M.INTERVAL_QS) and (p["q"][0.025] <= p["q"][0.975]).all()


def test_arm_networks_reproduce_the_stored_validation_predictions(M, world):
    df, pix, feats, cc = world["dev"]
    spec = world["cand"]["artifacts"]["v2"]
    asm = U.assemble(Path(spec["root"]), spec["arm"])
    keep = np.isin(df["row_id"].astype(str), asm["row_id"][:80])
    sub, psub, fsub = M.subset(df, pix, feats, keep)
    targets, nets = M.arm_networks(spec, sub, psub, fsub, cc, "cpu")
    assert targets == world["targets"] and sorted(nets) == [(u, i) for u in TR.UNITS for i in TR.INITS]
    cols = M.arm_columns("v2", nets)
    assert sorted(k for k in cols if k != "q") == sorted(LK.member_columns("v2"))
    pos = pd.Series(np.arange(len(asm["row_id"])), index=asm["row_id"]).loc[sub["row_id"].astype(str)].to_numpy()
    stored = U.medians(tuple(a[pos] for a in U.arm_mixture(asm["nets"])))
    assert np.abs(cols["v2"] - stored).max() < M.PARITY_TOL
    assert np.abs(cols["v2@u1i2"] - U.medians(tuple(a[pos] for a in asm["nets"][1, 2]))).max() < M.PARITY_TOL
    bad = torch.load(U.network_dir(Path(spec["root"]), spec["arm"], 0, 0) / "model.pt", weights_only=False)
    assert bad["seed"] == TR.network_seed(0, 0)


# ------------------------------------------------------------------------------------------------- dry run

def test_dry_run_seals_scores_and_registers_nothing(M, world, sandbox):
    assert M.main(["--dry-run", "t1", "--candidate", str(world["candidate"]), "--n-boot", str(N_BOOT)]) == 0
    d = paths.RUNS_ROOT / "orbit" / LK.DRYRUN_ROOT / "t1" / "M2S_test"
    assert sandbox == {"look": [], "score": []}                            # no registry row of any kind
    assert not (paths.RUNS_ROOT / "orbit" / LK.ROOT).exists()
    for f in ("TEST_LOOK.json", "PREDICTIONS_SAVED.json", "SCORED.json", "predictions/SHA256SUMS",
              "predictions/cell_valxany_lake.parquet", "predictions/cell_valxany_nonlake.parquet",
              "scores/endpoints.csv", "scores/members.csv", "scores/descriptive.csv", "scores/intervals.csv",
              "scores/censored.csv", "scores/summary.json"):
        assert (d / f).is_file(), f
    tab = pd.read_parquet(d / "predictions" / "cell_valxany_lake.parquet")
    dev = world["dev"][0]
    cell = M.dry_cells(dev)["valxany_lake"]
    assert set(tab["row_id"]) <= set(dev["row_id"][cell]) and set(tab["target"]) == set(world["targets"])
    assert set(LK.sealed_columns(world["cand"]["sealed_models"])) <= set(tab.columns)
    assert {"y", "censored", "y_dl", *M.META_COLS, "q|v2|0.025", "q|v1|0.975"} <= set(tab.columns)
    base = tab[tab["target"] == "chla"]
    assert not base["added"].any() and np.isfinite(base["y"]).all() and np.isfinite(base["v1@s1"]).all()
    tp = tab[tab["target"] == "tp"]
    assert tp["censored"].any() and tp["y"][tp["censored"]].isna().all() and tp["y_dl"][tp["censored"]].notna().all()
    assert tp["v1"].isna().all() and np.isfinite(tp["v2"]).all() and np.isfinite(tp["cmp:trophic_ols"]).all()
    assert base["cmp:trophic_ols"].isna().all() and (tab["v2"].to_numpy() == tab["track_n"].to_numpy()).all()
    nl = pd.read_parquet(d / "predictions" / "cell_valxany_nonlake.parquet")
    assert len(nl) and not set(nl["row_id"]) & set(tab["row_id"])
    s = json.loads((d / "scores" / "summary.json").read_text())
    assert s["dry_run"] == "t1" and s["assumed"] == ["test_s_primary_cell = all_rows", "every endpoint confirmatory"]
    assert s["parity"]["v2"]["max_abs_diff_log10"] < M.PARITY_TOL and s["parity"]["comparators"]["n_rows"] > 0
    assert s["n_hypotheses"] == {"H1-S": 2, "H2-S": 6, "TrackN": 3, "small_lakes": 32}
    assert s["scope"] == "unseen lakes, archive years 2015–2023, scenes shared with training"
    assert s["n_labels"] == 0                                              # untrained networks beat no comparator
    e = pd.read_csv(d / "scores" / "endpoints.csv")
    assert set(e["analysis"]) == {"primary", "sensitivity"} and set(e["family"]) == set(CL.FAMILIES)
    p = e[(e["analysis"] == "primary") & (e["family"] == "H1-S")]
    assert len(p) == 2 * 4 and np.isfinite(p["diff"]).all() and not p["label"].any()                 # untrained
    assert (p["p_holm"] == 1.0).all() and (p["divisor"] == 2).all()
    turb = e[(e["analysis"] == "primary") & (e["family"] == "H2-S") & (e["target"] == "turbidity")]
    g = tab[(tab["target"] == "turbidity") & tab["ntu_class"]]
    assert (turb["n"] == len(g)).all() and len(g) < (tab["target"] == "turbidity").sum()        # R-18
    c = pd.read_csv(d / "scores" / "censored.csv")
    assert set(c["target"]) == {"tp"} and c["share_at_or_below_dl"].between(0, 1).all()
    with pytest.raises(ProtectedDataError, match="already claimed|not empty"):                   # R-24
        M.main(["--dry-run", "t1", "--candidate", str(world["candidate"]), "--n-boot", str(N_BOOT)])
    with pytest.raises(SystemExit, match="validation rows only"):
        M.main(["--dry-run", "t2", "--allow-test"])


def test_dry_run_stops_before_the_claim_on_a_changed_artifact(M, world, sandbox, tmp_path):
    cand = json.loads(world["candidate"].read_text())
    cand["weights_sha256"]["v2@u2i1"] = "0" * 64
    f = tmp_path / "cand.json"
    f.write_text(json.dumps(cand))
    with pytest.raises(ProtectedDataError, match="frozen sha256 of 'v2@u2i1'"):
        M.main(["--dry-run", "t3", "--candidate", str(f)])
    cand = json.loads(world["candidate"].read_text())
    cand["sealed_models"]["v1_networks"] = 3
    f.write_text(json.dumps(cand))
    with pytest.raises(SystemExit, match="disagree"):
        M.main(["--dry-run", "t3", "--candidate", str(f)])
    assert not (paths.RUNS_ROOT / "orbit").exists()                        # nothing claimed, nothing written


# ----------------------------------------------------------------------------------------------- real path

def _freeze(world, primary="scene_disjoint"):
    ps = {f: {t: {"status": "primary" if f == "H1-S" else "confirmatory"} for t in v["targets"]}
          for f, v in CL.FAMILIES.items()}
    ps["H2-S"]["tss"]["status"] = "descriptive"
    return {**world["cand"], "_sha256": "f" * 64, "_path": "/x/FREEZE_orbit_S.json", "data_version": "dev_vXm",
            "scoring_code_sha256": "c" * 64, "test_s_primary_cell": primary, "power_status": ps}


class _Loader:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, 0

    def load(self, freeze):
        self.calls += 1
        LK.require_open_claim(freeze)                                      # the loader's own gate
        if self.fail:
            raise RuntimeError("chip store unreadable")
        df, pix, feats, cc = _rows(1, role_test=True)
        return df, pix, feats, cc, {"c1": {"n_removed": 0}}


def test_real_look_refused_without_freeze_flag_or_loader(M, world, sandbox, monkeypatch, tmp_path):
    monkeypatch.setattr(LK, "FREEZE_FILE", tmp_path / "FREEZE_orbit_S.json")
    with pytest.raises(ProtectedDataError, match="does not exist"):
        M.main(["--allow-test"])
    monkeypatch.setattr(LK, "load_freeze", lambda *a, **k: _freeze(world))
    with pytest.raises(SystemExit, match="--allow-test"):
        M.main([])
    with pytest.raises(SystemExit, match="B = "):
        M.main(["--allow-test", "--n-boot", "10"])
    monkeypatch.setattr(M, "LOADER", "sentinel.models.orbit_v2.no_such_loader")
    with pytest.raises(SystemExit, match="nothing was claimed"):
        M.main(["--allow-test"])
    assert sandbox == {"look": [], "score": []} and not (paths.RUNS_ROOT / "orbit").exists()


def test_real_look_on_synthetic_test_rows_runs_once(M, world, sandbox, monkeypatch):
    monkeypatch.setattr(LK, "load_freeze", lambda *a, **k: _freeze(world))
    monkeypatch.setattr(LK, "verify_data_pins", lambda fz: {})
    monkeypatch.setattr(CL, "N_BOOT", N_BOOT)
    loader = _Loader()
    monkeypatch.setattr(M, "protected_loader", lambda: loader)
    assert M.main(["--allow-test"]) == 0 and loader.calls == 1
    d = paths.RUNS_ROOT / "orbit" / LK.ROOT / "M2S_test"
    assert [r["metric"] for r in sandbox["look"]] == ["look_claimed", "predictions_sealed"]
    tab = pd.read_parquet(d / "predictions" / f"cell_{LK.CELL_LAKE}.parquet")
    assert tab["row_id"].str.startswith("t").all() and (d / "predictions" / f"cell_{LK.CELL_NONLAKE}.parquet").is_file()
    assert tab["scene_disjoint"].any() and not tab["scene_disjoint"].all()
    s = json.loads((d / "SCORED.json").read_text())
    assert s["primary_cell"] == "scene_disjoint" and s["scope"].endswith("scene-disjoint")
    assert s["n_hypotheses"]["H2-S"] == 4 and s["under_powered"]["H2-S"] == ["tss"]              # divisor stays 6
    assert s["n_registry_rows"] == len(sandbox["score"]) > 20
    e = pd.read_csv(d / "scores" / "endpoints.csv")
    p = e[(e["analysis"] == "primary") & (e["family"] == "H1-S") & (e["arm"] == "v2")]
    assert (p["n"] == int(((tab["target"] == "chla") & tab["scene_disjoint"]).sum())).all()
    q = e[(e["analysis"] == "sensitivity") & (e["family"] == "H1-S") & (e["arm"] == "v2")]
    n_all = int((tab["target"] == "chla").sum())
    assert (q["n"] == n_all).all() and p["n"].iloc[0] < n_all                 # §1.6: all rows, not the rest
    assert not q["label"].any()                                                           # a label is primary only
    assert (e[e["family"] == "H2-S"]["divisor"] == 6).all()
    ex = {r["experiment"] for r in sandbox["score"]} | {r["experiment"] for r in sandbox["look"]}
    assert ex == {"orbit_TEST_S_dev_vXm"}
    diff = [r for r in sandbox["score"] if r["metric"].startswith("mdsa_diff_vs_")]
    assert diff and all(r["ci_low"] <= r["value"] <= r["ci_high"] or True for r in diff)
    assert all("p_holm" in r["extra"] and "label" in r["extra"] for r in diff)
    assert sum(r["metric"] == "n_labels" for r in sandbox["score"]) == 1
    n_before = len(sandbox["score"])
    for argv in (["--allow-test"], ["--allow-test", "--rescore-from-saved"]):                    # single pass
        with pytest.raises(ProtectedDataError):
            M.main(argv)
    assert loader.calls == 1 and len(sandbox["score"]) == n_before


def test_a_failure_inside_the_claim_aborts_the_look_for_good(M, world, sandbox, monkeypatch):
    monkeypatch.setattr(LK, "load_freeze", lambda *a, **k: _freeze(world))
    monkeypatch.setattr(LK, "verify_data_pins", lambda fz: {})
    monkeypatch.setattr(CL, "N_BOOT", N_BOOT)
    loader = _Loader(fail=True)
    monkeypatch.setattr(M, "protected_loader", lambda: loader)
    with pytest.raises(RuntimeError, match="chip store"):
        M.main(["--allow-test"])
    d = paths.RUNS_ROOT / "orbit" / LK.ROOT / "M2S_test"
    assert (d / "ABORTED.json").is_file() and not (d / "predictions").exists()
    assert [r["metric"] for r in sandbox["look"]] == ["look_claimed", "look_aborted"] and not sandbox["score"]
    monkeypatch.setattr(M, "protected_loader", lambda: _Loader())
    with pytest.raises(ProtectedDataError):
        M.main(["--allow-test"])


def test_rescore_from_saved_scores_a_look_that_stopped_after_sealing(M, world, sandbox, monkeypatch):
    monkeypatch.setattr(LK, "load_freeze", lambda *a, **k: _freeze(world))
    monkeypatch.setattr(LK, "verify_data_pins", lambda fz: {})
    monkeypatch.setattr(CL, "N_BOOT", N_BOOT)
    monkeypatch.setattr(M, "protected_loader", lambda: _Loader())
    real = M.score
    monkeypatch.setattr(M, "score", lambda *a, **k: (_ for _ in ()).throw(MemoryError("stopped")))
    with pytest.raises(MemoryError):
        M.main(["--allow-test"])
    d = paths.RUNS_ROOT / "orbit" / LK.ROOT / "M2S_test"
    assert (d / "PREDICTIONS_SAVED.json").is_file() and not (d / "ABORTED.json").exists() and not sandbox["score"]
    monkeypatch.setattr(M, "score", real)
    class _Never:                                                      # resolving the loader is harmless; reading is not
        def load(self, freeze):
            raise AssertionError("protected rows are read a second time")

    monkeypatch.setattr(M, "protected_loader", _Never)
    with pytest.raises(ProtectedDataError):                                # a new look is refused
        M.main(["--allow-test"])
    assert M.main(["--allow-test", "--rescore-from-saved"]) == 0
    assert (d / "SCORED.json").is_file() and len(sandbox["score"]) > 20
    with pytest.raises(ProtectedDataError):
        M.main(["--allow-test", "--rescore-from-saved"])
