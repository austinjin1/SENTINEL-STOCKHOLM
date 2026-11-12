# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""D4 TapRisk test path refusals (synthetic fixtures only; no test row is read)."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from sentinel.models.downstream import common as C
from sentinel.models.downstream import testlook as TL

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("tp_taprisk", ROOT / "scripts" / "train" / "downstream_taprisk_test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "freeze_file", lambda app: tmp_path / f"FREEZE_downstream_{app}.json")
    monkeypatch.setattr(TL, "look_dir", lambda app: tmp_path / app / "test_look")
    monkeypatch.setattr(TL, "registry_test_rows", lambda app: 0)
    monkeypatch.setattr(TL, "verify_data", lambda app: {})
    return tmp_path


def test_refuses_without_freeze(isolated):
    with pytest.raises((C.ProtectedDataError, FileNotFoundError)):
        _load().main(["--allow-test"])
    assert not (isolated / "taprisk" / "test_look" / "TEST_LOOK.json").exists()


@pytest.mark.parametrize("argv", [[], ["--allow-test", "--dry-run-on-val"]])
def test_needs_one_mode(argv, isolated):
    with pytest.raises(SystemExit):
        _load().main(argv)


def test_builder_refuses_test_rows(isolated):
    from sentinel.models.downstream import taprisk as TR

    with pytest.raises(C.ProtectedDataError):
        TR.build("d4a_gw_nitrate", {"assignments": {"state_role": {}}}, include_test=True, split_sha=TR.SPLIT_SHA)


def _repeated_rows(n=300, k=8, seed=0):
    """Synthetic frame with each feature row repeated three times under different labels (impure leaves, so the
    trees vote with fractions — the case in which a threaded forest sum depends on thread-arrival order)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    x0 = pd.DataFrame(rng.normal(size=(n, k)), columns=[f"f{i}" for i in range(k)])
    x = pd.concat([x0, x0, x0], ignore_index=True)
    noise = rng.normal(size=3 * n)
    return x0, x, np.tile(x0["f0"].to_numpy(), 3), noise


def test_forest_prediction_is_single_threaded_and_bit_reproducible():
    """Lead ruling 2026-11-10: forest PREDICTION runs with n_jobs = 1, so repeated predictions and a second threaded
    fit give the same bytes; fitting keeps its threads."""
    from sentinel.models.downstream import taprisk_models as TM

    x0, x, sig, noise = _repeated_rows()
    y = (sig + noise > 0.8).astype(int)
    m = TM.BalancedRF(n=2, seed=0).fit(x, y)
    assert all(f.n_jobs == C.N_THREADS for f in m.models_)      # fitting is untouched
    p = [m.predict(x0) for _ in range(4)]
    assert all(f.n_jobs == 1 for f in m.models_)
    assert all(p[0].tobytes() == q.tobytes() for q in p[1:])
    assert TM.BalancedRF(n=2, seed=0).fit(x, y).predict(x0).tobytes() == p[0].tobytes()


def test_numeric_forest_prediction_is_single_threaded(monkeypatch):
    """Same ruling for the numeric track's Pennino-form RF regression: n_jobs is 1 when ``predict`` runs, the fit is
    threaded, and two calls give byte-identical predictions."""
    from sklearn.ensemble import RandomForestRegressor

    from sentinel.models.downstream import taprisk_numeric as TN

    x0, x, sig, noise = _repeated_rows(n=200, k=6)
    df = x.copy()
    df["target"] = sig * 0.5 + noise
    df["role"] = ["train"] * 400 + ["val"] * 200
    df["pwsid"] = [f"S{i:05d}" for i in range(len(df))]
    df["state_code"] = ["AA", "BB", "CC"] * 200
    df["huc8"] = "01010101"
    df["hist_task_10y"] = (noise > 0).astype(float)
    feats = [f"f{i}" for i in range(6)]
    monkeypatch.setattr(TN, "feature_cols", lambda d: feats)
    monkeypatch.setattr(TN.TR, "streamcat_columns", lambda: feats[:4])
    seen = {"fit": [], "predict": []}
    fit0, pred0 = RandomForestRegressor.fit, RandomForestRegressor.predict
    monkeypatch.setattr(RandomForestRegressor, "fit", lambda self, *a, **k: (seen["fit"].append(self.n_jobs), fit0(self, *a, **k))[1])
    monkeypatch.setattr(RandomForestRegressor, "predict", lambda self, *a, **k: (seen["predict"].append(self.n_jobs), pred0(self, *a, **k))[1])
    a = TN.fit_predict(df, ("train",), "val", "target")
    b = TN.fit_predict(df, ("train",), "val", "target")
    assert seen == {"fit": [C.N_THREADS, C.N_THREADS], "predict": [1, 1]}
    assert a["pennino_rf_reg"].to_numpy().tobytes() == b["pennino_rf_reg"].to_numpy().tobytes()
