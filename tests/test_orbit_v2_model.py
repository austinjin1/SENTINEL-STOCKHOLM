# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Unit tests for M2 architecture round v2 (orbit_arch_v2_amendment.md): stamp gate, imagery-only chip features,
SatWQ-LakeSet-MT, the EMA seed-unit trainer. Synthetic arrays only (RESEARCH_PLAN R9); no data files are read."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sentinel.data import chips as C
from sentinel.models.orbit import satwq as V1
from sentinel.models.orbit.data import BANDS, CENTER, TARGETS
from sentinel.models.orbit_v2 import chipfeat as CF
from sentinel.models.orbit_v2 import lakeset as LS
from sentinel.models.orbit_v2 import stamp as ST
from sentinel.models.orbit_v2 import train as TR

DEV = "cpu"


# ----------------------------------------------------------------------------------------- stamp gate (R-23)
def _repo(tmp_path, amendment="amendment v1\n", rulings="rulings v1\n"):
    """A scratch repository with the two stamped files, read-only as after the stamp."""
    out = []
    for rel, text in ((ST.AMENDMENT, amendment), (ST.RULINGS, rulings)):
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        if f.exists():
            f.chmod(0o644)
        f.write_text(text)
        f.chmod(0o444)
        out.append(f)
    return out


def _sha(f):
    return hashlib.sha256(f.read_bytes()).hexdigest()


def _row(shas, experiment=ST.STAMP_EXPERIMENT, written_by="lead", rid="r1"):
    extra = {"sha256": shas}
    if written_by is not None:
        extra["written_by"] = written_by
    return json.dumps({"id": rid, "timestamp": "2026-11-10T00:00:00Z", "experiment": experiment,
                       "model": "prereg_snapshot", "metric": "n_files", "split": "prereg", "value": 1,
                       "extra": extra}) + "\n"


def test_stamp_needs_one_lead_row_with_both_current_hashes(tmp_path):
    fa, fr = _repo(tmp_path)
    reg = tmp_path / "registry.jsonl"                     # scratch registry
    assert not ST.stamp_status(reg, tmp_path)["stamped"]  # no registry at all
    reg.write_text("")
    with pytest.raises(ST.StampError, match="refused before the lead's stamp"):
        ST.require_stamp("training", reg, tmp_path)
    both = {ST.AMENDMENT: _sha(fa), ST.RULINGS: _sha(fr)}
    closed = [
        _row(both, experiment="sonde/prereg_snapshot"),                      # another experiment
        _row({ST.AMENDMENT: _sha(fa)}),                                      # amendment only (the pre-R-23 rule)
        _row({ST.RULINGS: _sha(fr)}),                                        # rulings only
        _row({**both, ST.RULINGS: "0" * 64}),                                # wrong rulings hash
        _row({**both, ST.AMENDMENT: "0" * 64}),                              # wrong amendment hash
        _row({ST.AMENDMENT: _sha(fr), ST.RULINGS: _sha(fa)}),                # hashes swapped
        _row(both, written_by="M2"), _row(both, written_by=None),            # not written by the lead
        _row(both, written_by="Lead"), _row(both, written_by=["lead"]),
        "not json " + ST.STAMP_EXPERIMENT + "\n",
        json.dumps({"experiment": ST.STAMP_EXPERIMENT, "extra": "sha256"}) + "\n",
        json.dumps({"experiment": ST.STAMP_EXPERIMENT, "extra": {"sha256": [both], "written_by": "lead"}}) + "\n",
        json.dumps({"experiment": ST.STAMP_EXPERIMENT, "notes": json.dumps(both), "written_by": "lead"}) + "\n",
    ]
    for i in range(len(closed)):
        reg.write_text("".join(closed[: i + 1]))
        st = ST.stamp_status(reg, tmp_path)
        assert not st["stamped"] and st["row_id"] is None, closed[i]
    st = ST.stamp_status(reg, tmp_path)
    assert st["n_rows"] == len(closed) - 2 and "written_by" in st["reason"]
    # two half rows together do not make a stamp; one row with both hashes does
    reg.write_text("".join(closed) + _row(both, rid="good"))
    st = ST.require_stamp("training", reg, tmp_path)
    assert st["stamped"] and st["row_id"] == "good" and st["reason"] == "stamped"
    assert st["amendment_sha256"] == _sha(fa) and st["rulings_sha256"] == _sha(fr)
    # further files in the map do not matter
    reg.write_text(_row({**both, "docs/other.md": "1" * 64}, rid="more"))
    assert ST.stamp_status(reg, tmp_path)["row_id"] == "more"


@pytest.mark.parametrize("which", [0, 1])
def test_stamp_closes_when_either_file_is_edited_writable_or_missing(tmp_path, which):
    files = _repo(tmp_path)
    reg = tmp_path / "registry.jsonl"
    reg.write_text(_row({ST.AMENDMENT: _sha(files[0]), ST.RULINGS: _sha(files[1])}))
    assert ST.require_stamp("x", reg, tmp_path)["stamped"]
    f = files[which]
    for mode in (0o644, 0o464, 0o446):                    # any write bit: not read-only
        f.chmod(mode)
        st = ST.stamp_status(reg, tmp_path)
        assert not st["stamped"] and "not read-only" in st["reason"] and f.name in st["reason"]
    f.chmod(0o444)
    assert ST.stamp_status(reg, tmp_path)["stamped"]
    f.chmod(0o644)
    f.write_text("edited after the stamp\n")
    f.chmod(0o444)
    st = ST.stamp_status(reg, tmp_path)
    assert not st["stamped"] and "maps both" in st["reason"]
    with pytest.raises(ST.StampError):
        ST.require_stamp("x", reg, tmp_path)
    f.unlink()
    st = ST.stamp_status(reg, tmp_path)
    assert not st["stamped"] and "missing" in st["reason"]


def test_stamped_files_exist_in_repo_and_no_orbit_v2_code_writes_a_stamp_row():
    assert len(ST.amendment_sha256()) == 64
    assert set(ST.file_state()) == {ST.AMENDMENT, ST.RULINGS} == set(ST.STAMPED_FILES)
    repo = Path(__file__).resolve().parents[1]
    files = sorted((repo / "sentinel" / "models" / "orbit_v2").glob("*.py")) + \
        sorted((repo / "scripts").glob("*/orbitv2_*.py"))
    assert len(files) >= 12
    for f in files:
        text = f.read_text()
        if f.name == "stamp.py":
            assert "log_result" not in text and "registry.jsonl" in text and ".write" not in text and "open(" in text
            continue
        assert "prereg_snapshot" not in text, f            # only stamp.py names the stamp experiment


# ----------------------------------------------------------------------------------------- chip features
def _chip(water):
    """Synthetic int16 chip [L, 64, 64]: water (SCL 6, rho 0.02) / vegetation (SCL 4, red 0.05, NIR 0.40)."""
    layers = list(C.S2_STORE_LAYERS)
    chip = np.zeros((len(layers), 64, 64), np.int16)
    for i, name in enumerate(layers):
        if name == "SCL":
            chip[i] = np.where(water, CF.SCL_WATER, 4)
        elif name == "B08":
            chip[i] = np.where(water, 200, 4000)
        elif name in BANDS:
            chip[i] = np.where(water, 200, 500)
    return chip, layers


def test_dp_is_distance_to_nearest_non_water_pixel():
    water = np.zeros((64, 64), bool)
    water[:, 20:] = True                                  # shore between columns 19 and 20
    chip, layers = _chip(water)
    dp = CF.dp_map(chip, layers)
    assert dp.shape == (64, 64) and dp[5, 10] == 0.0
    assert dp[30, 20] == pytest.approx(10.0) and dp[30, 25] == pytest.approx(60.0)
    assert dp[0, 40] == pytest.approx(10.0)               # the chip edge counts as non-water (conservative)
    off = np.array([[0, 0], [0, -5], [3, 4]], np.float32)  # (dy, dx) from the chip centre
    got = CF.dp_at(dp, off, 3)
    assert got[0] == pytest.approx(dp[CENTER, CENTER]) and got[1] == pytest.approx(dp[CENTER, CENTER - 5])
    assert got[2] == pytest.approx(dp[CENTER + 3, CENTER + 4])


def test_ring_features_land_ring_and_missing_flag():
    water = np.zeros((64, 64), bool)
    water[:, 28:] = True                                  # land 50 .. 320 m west of the centre
    chip, layers = _chip(water)
    r = CF.ring_features(chip, layers)
    assert r.shape == (CF.RING_DIM,) and r[-1] == 0.0
    assert r[BANDS.index("B08")] == pytest.approx(np.arcsinh(0.40 / CF.REFL_SCALE), rel=1e-4)
    assert 0.0 < r[len(BANDS)] < 1.0                      # land fraction of the ring
    assert r[len(BANDS) + 1] == pytest.approx((0.40 - 0.05) / (0.40 + 0.05), rel=1e-4)   # land NDVI
    chip_w, _ = _chip(np.ones((64, 64), bool))            # open water: no land in the ring
    r0 = CF.ring_features(chip_w, layers)
    assert r0[-1] == 1.0 and np.all(r0[:len(BANDS) + 2] == 0.0)


def test_glint_geometry_and_fresnel():
    # sun at 30 deg from the south, sensor at 30 deg looking from the north: specular direction, glint angle 0
    g = CF.glint_features([30.0], [30.0], [180.0], [0.0])
    assert g.shape == (1, 3) and g[0, 1] == pytest.approx(0.0, abs=1e-3)
    assert g[0, 0] == pytest.approx(np.cos(np.radians(60.0)), abs=1e-5)
    # nadir view: glint angle = sun zenith
    g = CF.glint_features([40.0], [0.0], [100.0], [250.0])
    assert g[0, 1] == pytest.approx(np.radians(40.0), abs=1e-4)
    n = CF.N_WATER
    assert CF.fresnel(np.array([0.0]))[0] == pytest.approx(((n - 1) / (n + 1)) ** 2, rel=1e-4)
    assert CF.fresnel(np.array([np.radians(89.9)]))[0] > 0.9
    assert list(CF.baseline_group(["02.12", "03.00", "04.00", "05.09", "05.11", None])) == [0, 0, 1, 2, 2, 2]
    q = CF.date_quality([30.0, 50.0], [5.0, 8.0], [150.0, 160.0], [100.0, 280.0], [0.1, np.nan], ["04.00", "05.10"])
    assert q.shape == (2, LS.N_Q) and q[0, 5:].tolist() == [0.0, 1.0, 0.0] and q[1, 5:].tolist() == [0.0, 0.0, 1.0]


# ----------------------------------------------------------------------------------------- model
def _inputs(c=5, p=64, seed=0, n_geo=10, n_ctx=5):
    g = torch.Generator().manual_seed(seed)
    pm = torch.zeros(c, p, dtype=torch.bool)
    pm[:, :20] = True
    return dict(refl=torch.rand(c, p, 12, generator=g), lam=torch.rand(c, 12, generator=g),
                band_mask=torch.ones(c, p, 12, dtype=torch.bool), pix_mask=pm,
                offsets=torch.randn(c, p, 2, generator=g), geo=torch.randn(c, n_geo, generator=g),
                ctx=torch.randn(c, n_ctx, generator=g), ctx_on=torch.tensor([1, 1, 0, 1, 0][:c], dtype=torch.bool),
                dp=torch.rand(c, p, generator=g), ring=torch.randn(c, CF.RING_DIM, generator=g))


def _members(owner, slot, seed=1):
    g = torch.Generator().manual_seed(seed)
    m = len(owner)
    pm = torch.ones(m, LS.P_CONTEXT, dtype=torch.bool)
    pm[:, 5:] = False
    return dict(c_refl=torch.rand(m, LS.P_CONTEXT, 12, generator=g), c_lam=torch.rand(m, 12, generator=g),
                c_band_mask=torch.ones(m, LS.P_CONTEXT, 12, dtype=torch.bool), c_pix_mask=pm,
                c_offsets=torch.randn(m, LS.P_CONTEXT, 2, generator=g), c_dp=torch.rand(m, LS.P_CONTEXT, generator=g),
                c_q=torch.randn(m, LS.N_Q, generator=g),
                c_dt=torch.as_tensor(LS.dt_features(np.linspace(-183, 183, m))),
                c_owner=torch.as_tensor(owner), c_slot=torch.as_tensor(slot))


def test_lakeset_shapes_empty_set_and_permutation_invariance():
    torch.manual_seed(0)
    m = LS.SatWQLakeSetMT(n_geo=10, n_ctx=5, n_targets=10).eval()
    x = _inputs()
    mem = _members([0, 0, 0, 2, 2, 4, 4], [0, 1, 2, 0, 1, 0, 3])
    logw, mu, sg = m(**x, **mem)
    assert logw.shape == mu.shape == sg.shape == (5, 10, 5)
    assert torch.isfinite(logw).all() and torch.isfinite(mu).all() and (sg > 0).all()
    assert torch.allclose(torch.exp(logw).sum(-1), torch.ones(5, 10), atol=1e-5)
    e_logw, e_mu, e_sg = m(**x)                                    # no context member at all
    assert torch.isfinite(e_mu).all()
    assert torch.allclose(mu[[1, 3]], e_mu[[1, 3]], atol=1e-6)     # rows without members use the empty embedding
    assert not torch.allclose(mu[[0, 2, 4]], e_mu[[0, 2, 4]], atol=1e-4)
    mem2 = dict(mem, c_slot=torch.as_tensor([2, 0, 1, 1, 0, 3, 0]))
    assert torch.allclose(mu, m(**x, **mem2)[1], atol=1e-5)        # a set: slot order carries no information
    # a member of one row does not change another row's prediction
    mem3 = _members([0, 0, 0, 2, 2], [0, 1, 2, 0, 1])
    for k in ("c_refl", "c_lam", "c_offsets", "c_dp", "c_q", "c_dt"):
        mem3[k] = mem[k][:5]
    assert torch.allclose(mu[[0, 2]], m(**x, **mem3)[1][[0, 2]], atol=1e-5)


def test_c3_has_no_set_and_fewer_parameters():
    a1 = LS.SatWQLakeSetMT(n_geo=10, n_ctx=2, n_targets=10)
    c3 = LS.SatWQLakeSetMT(n_geo=10, n_ctx=2, n_targets=10, use_set=False).eval()
    assert V1.n_params(c3) < V1.n_params(a1) < 400_000
    x = _inputs(n_ctx=2)
    assert c3(**x)[1].shape == (5, 10, 5)
    assert not any(k.startswith(("sab", "pma", "q_mlp", "dt_mlp", "lake_proj", "set_empty")) for k in c3.state_dict())


def test_v1_encoder_transfer_reproduces_v1_chip_embedding():
    torch.manual_seed(3)
    v1 = V1.SatWQNet(n_geo=7, n_ctx=5).eval()
    m = LS.SatWQLakeSetMT(n_geo=10, n_ctx=5, n_targets=10).eval()
    done = m.load_v1_encoder(v1.state_dict())
    assert {n.split(".")[0] for n in done} == set(LS.TRANSFER)
    x = _inputs()
    feats = {}
    h = v1.pool_proj.register_forward_hook(lambda mod, i, o: feats.setdefault("chip", o))
    with torch.no_grad():
        v1(x["refl"], x["lam"], x["band_mask"], x["pix_mask"], x["offsets"], torch.randn(5, 7), x["ctx"], x["ctx_on"])
        mine = m.embed_chips(x["refl"], x["lam"], x["band_mask"], x["pix_mask"], x["offsets"], x["dp"])
    h.remove()
    assert torch.allclose(mine, feats["chip"], atol=1e-6)          # d_p column starts at zero
    assert float(m.pix_pos[0].weight[:, 3].abs().max()) == 0.0


# ----------------------------------------------------------------------------------------- loss
def test_mixture_loss_equals_v1_and_interval_term():
    torch.manual_seed(0)
    logw = torch.log_softmax(torch.randn(6, 4, 5), -1)
    mu, sg = torch.randn(6, 4, 5), torch.rand(6, 4, 5) + 0.2
    y = torch.randn(6, 4)
    y[0, 1] = float("nan")
    ym = torch.isfinite(y).float()
    assert torch.allclose(TR.mixture_loss(logw, mu, sg, y, ym), V1.masked_nll(logw, mu, sg, y, ym))
    assert torch.allclose(TR.mixture_loss(logw, mu, sg, y, ym, torch.zeros(6, 4, dtype=torch.bool), torch.ones(4)),
                          V1.masked_nll(logw, mu, sg, y, ym))
    # one Gaussian, one left-censored value: loss = -log Phi((DL - mu) / sigma)
    one = torch.zeros(1, 1, 1)
    got = TR.mixture_loss(one, one + 0.3, one + 2.0, torch.tensor([[1.1]]), torch.ones(1, 1),
                          torch.ones(1, 1, dtype=torch.bool))
    want = -torch.log(torch.distributions.Normal(0.3, 2.0).cdf(torch.tensor(1.1)))
    assert float(got) == pytest.approx(float(want), rel=1e-5)
    # weights: a zero-weight target does not contribute
    w = torch.tensor([1.0, 0.0, 0.5, 0.25])
    a = TR.mixture_loss(logw, mu, sg, y, ym, None, w)
    mu2 = mu.clone()
    mu2[:, 1] += 5.0
    assert torch.allclose(a, TR.mixture_loss(logw, mu2, sg, y, ym, None, w))
    # far below every component the censored term stays finite (log_ndtr, not log(cdf))
    lo = TR.mixture_loss(logw, mu, sg, y - 60.0, ym, torch.ones(6, 4, dtype=torch.bool))
    assert torch.isfinite(lo)


# ----------------------------------------------------------------------------------------- trainer
def _toy_set(n=96, seed=0, nt=len(TARGETS)):
    r = np.random.default_rng(seed)
    refl = r.uniform(0.5, 3.0, (n, 8, 12)).astype(np.float32)
    n_pix = r.integers(2, 9, n)
    for i in range(n):
        refl[i, n_pix[i]:] = np.nan
    y = (refl[:, 0, :nt] - 1.5 + 0.1 * r.standard_normal((n, nt))).astype(np.float32)
    y[r.random((n, nt)) < 0.2] = np.nan
    return V1.Batchable(refl, n_pix, r.integers(-3, 4, (n, 8, 2)).astype(np.float32),
                        np.tile(np.linspace(0.44, 2.2, 12, dtype=np.float32), (n, 1)),
                        r.standard_normal((n, 7)).astype(np.float32), r.standard_normal((n, 5)).astype(np.float32), y)


def _net(seed=0):
    torch.manual_seed(seed)
    return V1.SatWQNet(n_geo=7, n_ctx=5, d=16, n_layers=1, n_heads=2)


AUG = {"mult_noise": 0.02, "add_noise": 0.001, "band_add_noise": 0.0005, "p_rgbn": 0.1, "p_band_drop": 0.1,
       "p_ctx_drop": 0.5, "p_pixel_subset": 0.3}
KW = dict(epochs=4, lr=1e-3, batch=16, device=DEV, seed=5, aug=AUG, y_mean=np.zeros(4), y_std=np.ones(4),
          log=lambda *_: None)


def test_ema_trainer_with_decay_zero_is_the_v1_loop():
    """A0 = v1 recipe + EMA: with decay 0 the EMA weights are the raw weights, which must equal v1's run_epochs
    (same optimiser, schedule, batch order and augmentation draws)."""
    ds = _toy_set()
    a, b = _net(), _net()
    V1.run_epochs(a, ds, inner_set=None, **KW)
    TR.run_epochs_ema(b, ds, inner_set=None, select_idx=[0, 3], ema_decay=0.0, **KW)
    for k, v in a.state_dict().items():
        assert torch.allclose(v, b.state_dict()[k], atol=1e-6), k


def test_ema_weights_differ_from_raw_and_are_returned():
    ds = _toy_set()
    a, b = _net(), _net()
    TR.run_epochs_ema(a, ds, inner_set=None, select_idx=[0, 3], ema_decay=0.0, **KW)
    TR.run_epochs_ema(b, ds, inner_set=None, select_idx=[0, 3], ema_decay=0.9, **KW)
    init = _net().state_dict()
    k = "head.weight"
    d_raw, d_ema = (a.state_dict()[k] - init[k]).norm(), (b.state_dict()[k] - init[k]).norm()
    assert 0 < float(d_ema) < float(d_raw)                # the average lags behind the raw weights


def test_ema_update_rule():
    m = torch.nn.Linear(2, 1)
    e = TR.EMA(m, 0.9)
    w0 = m.weight.detach().clone()
    with torch.no_grad():
        m.weight.add_(1.0)
    e.update(m)
    assert torch.allclose(e.model.weight, 0.9 * w0 + 0.1 * (w0 + 1.0))


def test_checkpoint_resume_is_bit_identical(tmp_path):
    ds, inner = _toy_set(), _toy_set(64, seed=9)
    kw = dict(KW, epochs=7)
    a = _net()
    ra = TR.run_epochs_ema(a, ds, inner_set=inner, select_idx=[0, 3], ckpt=tmp_path / "a.pt", ckpt_every=5, **kw)
    assert ra["complete"]
    b = _net()
    n_calls = 0
    while True:                                           # a time budget of 0 s: one epoch per call
        n_calls += 1
        rb = TR.run_epochs_ema(b if n_calls == 1 else _net(seed=99), ds, inner_set=inner, select_idx=[0, 3],
                               ckpt=tmp_path / "b.pt", ckpt_every=5, max_seconds=0.0, **kw)
        if rb["complete"]:
            fin = b if n_calls == 1 else None
            break
        assert n_calls < 20
    assert n_calls == 7 and [h["epoch"] for h in rb["history"]] == list(range(7))
    assert ra["best_epoch"] == rb["best_epoch"] and ra["best_inner"] == rb["best_inner"]
    assert [h["train_nll"] for h in ra["history"]] == [h["train_nll"] for h in rb["history"]]
    c = _net(seed=123)                                    # a finished checkpoint: the call only restores the result
    rc = TR.run_epochs_ema(c, ds, inner_set=inner, select_idx=[0, 3], ckpt=tmp_path / "b.pt", **kw)
    assert rc["complete"] and rc["best_epoch"] == ra["best_epoch"]
    for k, v in a.state_dict().items():
        assert torch.equal(v, c.state_dict()[k]), k
    with pytest.raises(RuntimeError):                     # a checkpoint of another run is refused
        TR.run_epochs_ema(_net(), ds, inner_set=inner, select_idx=[0, 3], ckpt=tmp_path / "b.pt",
                          **dict(kw, seed=6))


def test_inner_criterion_needs_30_rows():
    q = np.zeros((40, 4, 1))
    y = np.zeros((40, 4), np.float32)
    y[:, 3] = np.nan
    v, n = TR.inner_criterion(q, y, np.zeros(4), np.ones(4), [0, 3])
    assert n == 1 and v == pytest.approx(0.0)
    with pytest.raises(RuntimeError):
        TR.inner_criterion(q, y, np.zeros(4), np.ones(4), [3])


# ----------------------------------------------------------------------------------------- seed units
def test_seed_unit_average_and_median():
    assert [TR.network_seed(u, i) for u in TR.UNITS for i in TR.INITS] == [0, 1, 2, 100, 101, 102, 200, 201, 202]
    w = np.ones((3, 2, 1))
    a = (w, np.full((3, 2, 1), -1.0), np.full((3, 2, 1), 0.5))
    b = (w, np.full((3, 2, 1), 1.0), np.full((3, 2, 1), 0.5))
    mix = TR.average_mixtures([a, b])
    assert mix[0].shape == (3, 2, 2) and np.allclose(mix[0].sum(-1), 1.0)
    assert np.allclose(TR.mixture_median(mix)[..., 0], 0.0, atol=1e-4)          # symmetric: median in the middle
    assert np.allclose(TR.mixture_cdf(mix, np.zeros((3, 2))), 0.5)
    same = TR.average_mixtures([a, a, a])
    assert np.allclose(TR.mixture_median(same)[..., 0], -1.0, atol=1e-4)
    # the median of the averaged mixture is not the mean of the medians when the members differ in spread
    c = (w, np.full((3, 2, 1), 1.0), np.full((3, 2, 1), 0.05))
    med = TR.mixture_median(TR.average_mixtures([a, c, c]))[..., 0]
    assert np.all(med > 0.5)


# ----------------------------------------------------------------------------------------- batches
def _set_batchable(n=40, m=30, seed=0):
    base = _toy_set(n, seed)
    r = np.random.default_rng(seed + 1)
    cs = _toy_set(m, seed + 2)
    cidx = np.full((n, LS.K_SET), -1)
    for i in range(n):
        k = r.integers(0, LS.K_SET + 1)
        cidx[i, :k] = r.choice(m, k, replace=False)
    cdt = np.where(cidx >= 0, r.integers(3, 184, cidx.shape) * r.choice([-1, 1], cidx.shape), 0)
    return LS.SetBatchable(base, r.uniform(0, 300, (n, 8)), r.standard_normal((n, CF.RING_DIM)), cstore=cs,
                           c_dp_m=r.uniform(0, 300, (m, 8)), c_q=r.standard_normal((m, LS.N_Q)), cidx=cidx, cdt=cdt), base


def test_set_batchable_target_tensors_are_v1_and_set_rules_hold():
    sb, base = _set_batchable()
    idx = np.arange(40)
    xb, y, ym = sb.batch(idx, DEV, train=True, rng=np.random.default_rng(1), aug=AUG)
    x1, y1, ym1 = base.batch(idx, DEV, train=True, rng=np.random.default_rng(1), aug=AUG)
    for k in x1:                                          # the target-date part is v1's batch, draw for draw
        assert torch.equal(xb[k], x1[k]), k
    assert torch.equal(y, y1) and torch.equal(ym, ym1)
    xe, _, _ = sb.batch(idx, DEV, train=False)
    n_have = (sb.cidx >= 0).sum(1)
    assert len(xe["c_owner"]) == n_have.sum()             # inference: the full set
    assert xe["c_refl"].shape[1:] == (8, 12) and xe["c_dt"].shape[1] == LS.N_DT
    assert "c_owner" not in sb.batch(idx, DEV, train=False, aug={"set_off": True})[0]
    # training: whole-set dropout near p, otherwise 1..n members, always a subset of the row's members
    rng = np.random.default_rng(2)
    dropped, sizes = 0, []
    for _ in range(60):
        xt, _, _ = sb.batch(idx, DEV, train=True, rng=rng, aug=dict(AUG, p_set_drop=0.3))
        cnt = np.bincount(xt["c_owner"].numpy(), minlength=40) if "c_owner" in xt else np.zeros(40, int)
        assert np.all(cnt <= n_have)
        dropped += int(((cnt == 0) & (n_have > 0)).sum())
        sizes.append(cnt[(cnt > 0)])
    frac = dropped / (60 * int((n_have > 0).sum()))
    assert 0.2 < frac < 0.4
    assert np.concatenate(sizes).min() == 1
    m = LS.SatWQLakeSetMT(n_geo=7, n_ctx=5, n_targets=4, d=16, n_layers=1, n_heads=2).eval()
    assert m(**xe)[1].shape == (40, 4, 5)
    q = V1.predict_quantiles(m, sb, DEV, np.zeros(4), np.ones(4))      # v1 prediction code runs on the v2 set
    assert q.shape == (40, 4, 1)
