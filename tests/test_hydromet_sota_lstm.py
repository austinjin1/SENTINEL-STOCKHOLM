# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M6 SOTA lever trainer: with every option off it reproduces the frozen EA-LSTM exactly; resume is bit-exact;
weak-label and auxiliary-target options run and never touch non-train roles. Synthetic arrays only (code paths)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sentinel.hydromet_sota.lstm import ResumeNeeded, SotaEALSTM  # noqa: E402
from sentinel.models.hydromet.lstm import EALSTMConfig, HydroMetEALSTM  # noqa: E402

from test_hydromet import synth  # noqa: E402

CFG = dict(seq_len=60, predict_last=20, hidden=8, static_emb=4, dropout=0.2, batch_size=16, steps_per_epoch=4,
           max_epochs=4, patience=10, val_max_windows=50, lr=1e-3, static_noise=0.3, seed=3)


def _data():
    d = synth(S=8, roles=["train"] * 6 + ["val"] * 2, groups=["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"],
              end="2021-09-30", blocks=("train",))
    return d


def _hist(m):
    return [{k: v for k, v in h.items() if k in ("epoch", "train_loss", "inner_val_nll", "inner_val_mse")}
            for h in m.history]


@pytest.mark.parametrize("store", ["gpu", "cpu"])
def test_options_off_reproduce_frozen_trainer(store):
    d = _data()
    fit, inner = np.array([0, 1, 2, 3]), np.array([4, 5])
    a = HydroMetEALSTM(EALSTMConfig(**CFG))
    a.fit(d, fit, inner, device="cpu", log=lambda *_: None)
    b = SotaEALSTM(EALSTMConfig(**CFG), store=store)
    b.fit(d, fit, inner, device="cpu", log=lambda *_: None)
    assert _hist(a) == _hist(b)
    for k in a.state:
        assert torch.equal(a.state[k], b.state[k]), k
    ti = np.flatnonzero(d.block_mask("train"))[-100:]
    pa, pb = a.predict(d, [6, 7], ti, device="cpu"), b.predict(d, [6, 7], ti, device="cpu")
    for v in pa:
        np.testing.assert_array_equal(pa[v], pb[v])


def test_resume_is_exact(tmp_path):
    d = _data()
    fit, inner = np.array([0, 1, 2, 3]), np.array([4, 5])
    ref = SotaEALSTM(EALSTMConfig(**CFG), store="cpu")
    ref.fit(d, fit, inner, device="cpu", log=lambda *_: None)
    rp = tmp_path / "resume.pt"
    m = SotaEALSTM(EALSTMConfig(**CFG), store="cpu")
    with pytest.raises(ResumeNeeded):
        m.fit(d, fit, inner, device="cpu", log=lambda *_: None, resume_path=rp, job_minutes=1e-9)
    n_jobs = 1
    while True:
        m = SotaEALSTM(EALSTMConfig(**CFG), store="cpu")
        try:
            m.fit(d, fit, inner, device="cpu", log=lambda *_: None, resume_path=rp, job_minutes=1e-9)
            break
        except ResumeNeeded:
            n_jobs += 1
    assert n_jobs >= 2
    assert _hist(ref) == _hist(m)
    for k in ref.state:
        assert torch.equal(ref.state[k], m.state[k]), k


def test_weak_and_aux_options_run_and_save(tmp_path):
    d = _data()
    # stations 4..5 inner, 6..7 relabelled as weak-label train stations with sparse temp / DO only
    d.roles = np.array(["train"] * 8)
    d.y[6:, :, 1] = np.nan
    d.y[6:, :, 3:] = np.nan
    keep = np.zeros(d.y.shape[1], bool)
    keep[::25] = True
    d.y[6:, ~keep, :] = np.nan
    aux = np.log(np.abs(d.forc[:, :, 0:1]) + 1.0).astype("float32")
    fit, inner, weak = np.array([0, 1, 2, 3]), np.array([4, 5]), np.array([6, 7])
    m = SotaEALSTM(EALSTMConfig(**CFG), weak_share=0.25, aux={"logq": 0.5}, store="cpu")
    info = m.fit(d, fit, inner, weak_idx=weak, aux_y=aux, device="cpu", log=lambda *_: None)
    assert info["n_weak_stations"] == 2 and "logq" in m.aux_norm
    assert all(h["inner_primary_mse"] is not None for h in m.history)
    p = tmp_path / "m.pt"
    m.save(p)
    m2 = SotaEALSTM.load(p)
    ti = np.flatnonzero(d.block_mask("train"))[-50:]
    a, b = m.predict(d, [0, 6], ti, device="cpu"), m2.predict(d, [0, 6], ti, device="cpu")
    assert set(a) == set(b) and "temp_c_mean" in a and "logq" not in a
    for v in a:
        np.testing.assert_array_equal(a[v], b[v])


def test_weak_stations_must_be_train_role():
    d = _data()
    m = SotaEALSTM(EALSTMConfig(**CFG), weak_share=0.25, store="cpu")
    with pytest.raises(ValueError, match="train-role"):
        m.fit(d, np.array([0, 1, 2, 3]), np.array([4, 5]), weak_idx=np.array([6]), device="cpu", log=lambda *_: None)
