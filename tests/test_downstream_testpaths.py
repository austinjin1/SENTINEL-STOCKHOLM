# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Refusal paths of the five downstream test looks (synthetic fixtures only; no real test row is read)."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from sentinel.models.downstream import common as C
from sentinel.models.downstream import testlook as TL

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {"safeswim": "downstream_safeswim_test.py", "d1n_beaches": "downstream_d1n_beaches_test.py",
           "biointegrity": "downstream_biointegrity_test.py", "musselguard": "downstream_musselguard_test.py",
           "toxinwatch": "downstream_toxinwatch_test.py"}


def _load(app):
    spec = importlib.util.spec_from_file_location(f"tp_{app}", ROOT / "scripts" / "train" / SCRIPTS[app])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """FREEZE files and test-look markers live in a temp dir; the registry is empty."""
    monkeypatch.setattr(C, "freeze_file", lambda app: tmp_path / f"FREEZE_downstream_{app}.json")
    monkeypatch.setattr(TL, "look_dir", lambda app: tmp_path / app / "test_look")
    monkeypatch.setattr(TL, "registry_test_rows", lambda app: 0)
    return tmp_path


@pytest.mark.parametrize("app", sorted(SCRIPTS))
def test_script_refuses_without_freeze(app, isolated):
    mod = _load(app)
    with pytest.raises(C.ProtectedDataError):
        mod.main(["--allow-test"])
    assert not (isolated / app / "test_look" / "TEST_LOOK.json").exists()     # nothing claimed


@pytest.mark.parametrize("app", sorted(SCRIPTS))
@pytest.mark.parametrize("argv", [[], ["--allow-test", "--dry-run-on-val"], ["--allow-test", "--rescore-from-saved"]])
def test_script_needs_exactly_one_mode(app, argv, isolated):
    with pytest.raises(SystemExit):
        _load(app).main(argv)


def _freeze(path, split_sha, code_sha):
    path.write_text(json.dumps({"split_sha256": split_sha, "scoring_code_sha256": code_sha}))


def test_freeze_with_wrong_code_hash_refused(isolated):
    _freeze(C.freeze_file("demo"), "s", "not-the-code-hash")
    with pytest.raises(C.ProtectedDataError, match="scoring code"):
        TL.authorise("demo", "s")


def test_freeze_with_wrong_split_refused(isolated):
    _freeze(C.freeze_file("demo"), "other", C.scoring_code_sha256("demo"))
    with pytest.raises(C.ProtectedDataError, match="split"):
        TL.authorise("demo", "s")


def test_authorise_claims_once_and_builders_need_the_claim(isolated):
    _freeze(C.freeze_file("demo"), "s", C.scoring_code_sha256("demo"))
    with pytest.raises(C.ProtectedDataError, match="without a claimed"):
        TL.require_authorised("demo", "s")
    TL.authorise("demo", "s")
    TL.require_authorised("demo", "s")                       # now allowed
    with pytest.raises(C.ProtectedDataError):
        TL.authorise("demo", "s")                            # single pass
    # a new freeze invalidates the old claim
    C.freeze_file("demo").write_text(json.dumps({"split_sha256": "s", "scoring_code_sha256": C.scoring_code_sha256("demo"),
                                                 "note": "re-frozen"}))
    with pytest.raises(C.ProtectedDataError, match="different freeze"):
        TL.require_authorised("demo", "s")


def test_log_test_metric_guards(isolated):
    with pytest.raises(ValueError):
        TL.log_test_metric(app="demo", split_sha="s", experiment="e", model="m", metric="auroc", value=0.5, split="val_x")
    with pytest.raises(C.ProtectedDataError):
        TL.log_test_metric(app="demo", split_sha="s", experiment="e", model="m", metric="auroc", value=0.5, split="test_x")


def test_common_log_metric_still_refuses_test():
    with pytest.raises(C.ProtectedDataError):
        C.log_metric(app="demo", experiment="e", model="m", metric="auroc", value=0.5, split="test_x")


# --- library builders refuse test-role rows without an authorised look (synthetic frames, no data read)
def test_toxinwatch_builder_refuses(isolated):
    from sentinel.models.downstream import toxinwatch as T

    rows = pd.DataFrame({"role": ["test"], "date": pd.to_datetime(["2023-07-01"])})
    with pytest.raises(C.ProtectedDataError):
        T.build_features(rows, pd.DataFrame(), p0=0.1)


def test_biointegrity_builder_refuses(isolated):
    from sentinel.models.downstream import biointegrity as BI

    with pytest.raises(C.ProtectedDataError):
        BI.build_features(pd.DataFrame({"role": ["test"]}), range(2000, 2015))


def test_musselguard_test_origin_refused(isolated):
    from sentinel.models.downstream import musselguard as MG

    with pytest.raises(C.ProtectedDataError):
        MG.build_origin(2018, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())


def test_safeswim_builder_refuses(isolated):
    from sentinel.models.downstream import safeswim as S

    with pytest.raises(C.ProtectedDataError):
        S.build("d1b", "nowcast", {"assignments": {}, "meta": {}}, include_test=True, split_sha=S.SPLIT_SHA)


def test_wqp_track_builder_refuses(isolated):
    from sentinel.models.downstream import d1n_beaches as A
    from sentinel.models.downstream import wqp_track as WT

    with pytest.raises(C.ProtectedDataError):
        WT.build_frame(A.CFG, {"assignments": {}}, include_test=True, split_sha=A.SPLIT_SHA)


# --- synthetic end-to-end test of the test-mode write path (pre-test gate S-W3)
import numpy as np  # noqa: E402

SPLIT_SHA = {"safeswim": "faabfff9bfbe6d5d72cca8c30f26c0ff79faeb8d1771626bbfd8d3dc854b25b9",
             "d1n_beaches": "aa601b505530f2d93bf90881e106d882ffc46efdf38ac9cda2db1816e6f96861",
             "biointegrity": "0d9de79a17b33e06e57703e15b2f4309bcabd0dbc55e8e21f3eb2a9558f1601c",
             "musselguard": "04daee8a36e240443a8f711d8b8286c4db2f0e9cd2b9377f0b5d963eeb66dd46",
             "toxinwatch": "13852e4a97490da97ddafeefeb757074c2b832c7cb361ab9bf85c077e11bef85"}


def _synthetic_pred(app, mod, rng, n=240):
    y = (rng.random(n) < 0.3).astype(float)
    good = y + rng.normal(0, 0.6, n)                  # challenger: informative
    noise = lambda: rng.normal(0, 1, n)               # noqa: E731
    g8 = rng.integers(0, 24, n).astype(str)
    if app == "d1n_beaches":
        p = pd.DataFrame({"cluster": rng.integers(0, 40, n).astype(str), "date": pd.Timestamp("2022-07-01"), "y": y, "huc8": g8,
                          "set": np.where(np.arange(n) % 2, "test_known_2021_2025", "test_new_2021_2025")})
        for k in mod.MODELS:
            p[f"s_{k}"] = good if k in ("logistic", "gbm_challenger") else noise()
        for i in range(3):
            p[f"s_gbm_challenger_seed{i}"] = good
        p["stratum_rain"] = np.arange(n) % 3 == 0
    elif app == "safeswim":
        p = pd.DataFrame({"beach_key": rng.integers(0, 12, n).astype(str), "date": pd.Timestamp("2024-07-01"), "y": y,
                          "week": rng.integers(0, 30, n).astype(str)})
        for k in mod.MODELS:
            p[f"s_{k}"] = good if k in ("logistic_l2", "gbm_hgb") else noise()
    elif app == "biointegrity":
        p = pd.DataFrame({"visit_id": [f"v{i}" for i in range(n)], "y_poor": y, "y_ord": 2 * y, "huc4": rng.integers(0, 20, n).astype(str),
                          "huc8": g8, "location_id": np.arange(n), "is_new": np.arange(n) % 4 != 0, "is_frozen_new": np.arange(n) % 3 != 0})
        for k in mod.ALL + ["persistence"]:
            p[f"s_{k}"] = good if k == "ens_tuned_gbm_rf" else noise()
        for k in ["ens_tuned_gbm_rf"] + [k for k, v in mod.MODELS.items() if v[0] in ("rf", "gbm")]:
            for i in range(3):
                p[f"s_{k}_seed{i}"] = good if k == "ens_tuned_gbm_rf" else noise()
    elif app == "musselguard":
        p = pd.DataFrame({"huc12": np.arange(n).astype(str), "origin": np.where(np.arange(n) % 2, 2018, 2021), "y": y,
                          "front": np.arange(n) % 5 != 0, "mn": np.arange(n) % 7 == 0, "huc8": g8})
        for k in mod.MODELS:
            p[f"s_{k}"] = good if k == "pressure_gbm" else noise()
        for k in ("pressure_gbm", "location_only_gbm"):
            for i in range(3):
                p[f"s_{k}_seed{i}"] = good if k == "pressure_gbm" else noise()
    else:  # toxinwatch
        p = pd.DataFrame({"lake_cluster": rng.integers(0, 30, n).astype(str), "site_id": "s", "date": pd.Timestamp("2023-07-01"),
                          "y": y, "scum": np.arange(n) % 4 == 0,
                          "set": np.where(np.arange(n) % 2, "test_known_2022_2026", "test_new_lakes_2022_2026")})
        for k in mod.MODELS:
            p[f"s_{k}"] = good if k in ("gbm_challenger", "logistic") else noise()
        for i in range(3):
            p[f"s_gbm_challenger_seed{i}"] = good
    return p


@pytest.mark.parametrize("app", sorted(SCRIPTS))
def test_test_mode_write_path_end_to_end(app, isolated, monkeypatch):
    import sentinel.core.registry as R

    rows = []
    monkeypatch.setattr(R, "log_result", lambda **kw: rows.append(kw) or kw)
    mod = _load(app)
    sha = SPLIT_SHA[app]
    _freeze(C.freeze_file(app), sha, C.scoring_code_sha256(app))
    TL.authorise(app, sha)
    pred = _synthetic_pred(app, mod, np.random.default_rng(0))
    TL.save_predictions(app, sha, pred)
    with pytest.raises(C.ProtectedDataError):              # re-fit / second save refused
        TL.save_predictions(app, sha, pred)
    res = mod.score(TL.authorise_rescore(app, sha), "test", sha)   # rescore path reads the saved predictions
    assert rows and all(r["split"].startswith("test") for r in rows)
    assert all(r["extra"]["test_look"] for r in rows)
    diffs = {r["metric"] for r in rows if "diff" in r["metric"]}
    assert diffs and all(d.startswith("auprc" if app == "musselguard" else "auroc") for d in diffs)
    assert res["claim_met"] is True                         # informative synthetic challenger
    TL.mark_scored(app, {"claim_met": res["claim_met"]})
    with pytest.raises(C.ProtectedDataError):
        TL.authorise_rescore(app, sha)                     # nothing re-scored after completion


def test_rescore_detects_tampering(isolated):
    _freeze(C.freeze_file("demo"), "s", C.scoring_code_sha256("demo"))
    TL.authorise("demo", "s")
    rec = TL.save_predictions("demo", "s", pd.DataFrame({"a": [1, 2]}))
    pd.DataFrame({"a": [9, 9]}).to_parquet(rec["file"], index=False)
    with pytest.raises(C.ProtectedDataError, match="modified"):
        TL.authorise_rescore("demo", "s")


def test_seed_rule():
    h = {"x": {"holm_reject": True, "diff": 0.02, "per_seed_diff": [0.01, 0.02, -0.001]}}
    assert TL.claim_met(h)["claim_met"] is False
    h["x"]["per_seed_diff"] = [0.01, 0.02, 0.001]
    assert TL.claim_met(h)["claim_met"] is True
    assert TL.claim_met({"x": {"holm_reject": True, "diff": 0.01, "per_seed_diff": None}}, deterministic=True)["claim_met"]


def test_holm_counts_unscored():
    res = {"s": {"paired": {"a-b": {"kind": "primary", "u": {"p_value": 0.02, "diff": 0.1, "ci_low": 0.01, "ci_high": 0.2}}}}}
    assert TL.holm_primary(res, "u")["s|a-b"]["holm_reject"]
    h = TL.holm_primary(res, "u", unscored=("c", "d", "e"))     # m = 4: 0.02 * 4 = 0.08 -> not rejected
    assert not h["s|a-b"]["holm_reject"] and h["s|a-b"]["m"] == 4


def test_claim_met_fails_closed_on_empty_family():
    assert TL.claim_met({})["claim_met"] is False
    assert TL.claim_met({"unscored|x": {"unscored": True, "diff": None, "holm_reject": False}})["claim_met"] is False


def test_degenerate_stratum_not_registered(isolated, monkeypatch):
    """W-N1: a single-class stratum gives NaN paired diffs -> recorded as not estimable, no registry crash."""
    import sentinel.core.registry as R

    rows = []
    monkeypatch.setattr(R, "log_result", lambda **kw: rows.append(kw) or kw)
    _freeze(C.freeze_file("demo"), "s", C.scoring_code_sha256("demo"))
    TL.authorise("demo", "s")
    y = np.zeros(30)
    r = TL.score_set(app="demo", mode="test", split_sha="s", set_name="test_x", experiment="e", y=y,
                     scores={"a": np.random.default_rng(0).random(30), "b": np.random.default_rng(1).random(30)},
                     groups={"g": np.arange(30) % 5}, primary={"a": ["b"]})
    assert r["paired"]["a-b"]["g"]["not_estimable"] is True
    assert not [x for x in rows if "diff" in x["metric"]]
    assert TL.claim_met(TL.holm_primary({"test_x": r}, "g"))["claim_met"] is False
