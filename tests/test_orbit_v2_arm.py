# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Arms of architecture round v2 (orbit_arch_v2_amendment.md §3): configuration files, target matrices, cells, data
sets and the training of one network. Synthetic tables and arrays only, CPU only; the tests read no chip store, no
label table and no cache, and write only under pytest's temporary directory."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from sentinel.core import paths
from sentinel.models.orbit import data as D
from sentinel.models.orbit import features as F1
from sentinel.models.orbit import pipeline as PL
from sentinel.models.orbit import satwq as V1
from sentinel.models.orbit_v2 import arm as ARM
from sentinel.models.orbit_v2 import chipfeat as CF
from sentinel.models.orbit_v2 import lakeset as LS
from sentinel.models.orbit_v2 import stamp as ST
from sentinel.models.orbit_v2 import targets as T
from sentinel.models.orbit_v2 import train as TR

REPO = Path(__file__).resolve().parents[1]
ARM_FILES = {a: REPO / "configs" / "orbit_v2" / f"{a}.yaml" for a in ARM.ARMS}
V1_CFG = "configs/orbit/satwq.yaml"


@pytest.fixture(autouse=True)
def _synthetic_wavelengths(monkeypatch):
    wl = {b: 440.0 + 150.0 * i for i, b in enumerate(D.BANDS)}
    monkeypatch.setattr(D, "band_wavelengths", lambda: {"S2A": wl, "S2B": {b: v + 1.0 for b, v in wl.items()}})


# ----------------------------------------------------------------------------------------- configuration files
def test_arm_files_cover_the_four_arms_and_repeat_the_amendment_constants():
    for arm, f in ARM_FILES.items():
        spec = yaml.safe_load(f.read_text())
        assert spec["arm"] == arm
        for k, v in ARM.FIXED.items():
            assert spec[k] == v, (arm, k)
    assert ARM.ARMS["a1_c1"]["latlon"] and not ARM.ARMS["a1_c2"]["latlon"] and not ARM.ARMS["a1_c3"]["latlon"]
    assert ARM.ARMS["a1_c1"]["use_set"] and ARM.ARMS["a1_c2"]["use_set"] and not ARM.ARMS["a1_c3"]["use_set"]
    assert ARM.LOSS_WEIGHTS == {"chla": 1.0, "secchi": 1.0, "tss": 1.0, "turbidity": 1.0, "tp": 0.5, "tn": 0.5,
                                "doc": 0.5, "turbidity_fnu": 0.25, "ssc": 0.25, "acdom440": 0.25}
    assert set(ARM.LOSS_WEIGHTS) == set(D.TARGETS) | set(T.SPECS)
    assert [TR.network_seed(u, i) for u in TR.UNITS for i in TR.INITS] == [0, 1, 2, 100, 101, 102, 200, 201, 202]


def _arm_file(tmp_path, src="a1_c2", **change):
    spec = yaml.safe_load(ARM_FILES[src].read_text())
    spec.update(change)
    f = tmp_path / f"{src}.yaml"
    f.write_text(yaml.safe_dump(spec))
    return f


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _base(**kw):
    return {"config_file": V1_CFG, "config_sha256": _sha(REPO / V1_CFG), "set": [], "fit_scheme": "base",
            "selection_registry_row": "abc123", **kw}


def test_real_run_needs_the_selected_v1_configuration(tmp_path):
    for f in ARM_FILES.values():                          # as shipped: the base is not filled in yet
        with pytest.raises(ARM.ArmError, match="not set"):
            ARM.resolve_config(f)
        with pytest.raises(ARM.ArmError, match="not set"):
            ARM.resolve_config(f, smoke=True)
    with pytest.raises(ARM.ArmError, match="smoke tests only"):
        ARM.resolve_config(ARM_FILES["a0"], smoke_base=V1_CFG)
    cfg = ARM.resolve_config(_arm_file(tmp_path, base=_base(set=["finetune.lr=0.001"])))
    assert cfg["arm"] == "a1_c2" and cfg["v1"]["finetune"]["lr"] == 0.001 and cfg["v1"]["finetune"]["epochs"] == 80
    assert cfg["fit_scheme"] == "base" and not cfg["smoke"] and "smoke" not in cfg["v1"]
    for k in ("config_sha256", "selection_registry_row", "fit_scheme"):
        with pytest.raises(ARM.ArmError, match="not set"):
            ARM.resolve_config(_arm_file(tmp_path, base=_base(**{k: None})))
    with pytest.raises(ARM.ArmError, match="differs from base.config_sha256"):
        ARM.resolve_config(_arm_file(tmp_path, base=_base(config_sha256="0" * 64)))


def test_smoke_configuration_and_refused_arm_files(tmp_path):
    cfg = ARM.resolve_config(ARM_FILES["a0"], smoke=True, smoke_base=V1_CFG)
    assert cfg["smoke"] and cfg["v1"]["finetune"]["epochs"] == 2 and cfg["v1"]["pretrain"]["epochs"] == 2
    assert cfg["ema_decay"] == 0.999 and cfg["amp"] is True and cfg["budget_minutes"] <= 115
    for change, msg in [({"ema_decay": 0.99}, "ema_decay"), ({"p_set_drop": 0.5}, "p_set_drop"),
                        ({"loss_weights": {**ARM.LOSS_WEIGHTS, "tp": 1.0}}, "loss_weights"),
                        ({"min_head_fit": 50}, "min_head_fit"), ({"units": [0, 1, 2, 3]}, "units"),
                        ({"arm": "a2"}, "unknown arm"), ({"lr": 0.1}, "unknown keys"),
                        ({"budget_minutes": 130}, "below 2 h")]:
        with pytest.raises(ARM.ArmError, match=msg):
            ARM.resolve_config(_arm_file(tmp_path, **change), smoke=True, smoke_base=V1_CFG)
    with pytest.raises(ARM.ArmError, match="unknown base keys"):
        ARM.resolve_config(_arm_file(tmp_path, base=_base(lr=1)))
    blend = tmp_path / "blend.yaml"
    blend.write_text((REPO / V1_CFG).read_text() + "\nblend: {enabled: true}\n")
    with pytest.raises(ARM.ArmError, match="no blend"):
        ARM.resolve_config(ARM_FILES["a0"], smoke=True, smoke_base=blend, repo_root=tmp_path)


# ----------------------------------------------------------------------------------------- synthetic dev_v5m
N_BLOCKS = 60


def _synth(n=900, n_added=120, seed=0, tp_fit_values=None):
    """A dev_v5m-shaped table with arrays. Train-role rows of train years plus a few val-role rows."""
    r = np.random.default_rng(seed)
    blocks = np.array([f"b{i}" for i in range(N_BLOCKS)])
    blk = blocks[r.integers(0, N_BLOCKS, n)]
    role = np.where(np.isin(blk, blocks[:50]), "train", "val")
    tb = np.where(r.random(n) < 0.85, "train", "val")
    refl = r.uniform(0.2, 3.0, (n, 64, 12)).astype(np.float32)
    n_pix = r.integers(3, 30, n).astype(np.int32)
    for i in range(n):
        refl[i, n_pix[i]:] = np.nan
    sig = refl[:, 0, :]
    df = pd.DataFrame({
        "row_id": [f"r{i}" for i in range(n)], "chip_id": [f"k{i}" for i in range(n)], "block": blk, "role": role,
        "tblock": tb, "cell": pd.Series(role) + "x" + pd.Series(tb), "buffer_excluded": False, "protected": False,
        "in_lake": True, "waterbody": pd.Series(blk).map(lambda b: "L" + b), "platform": r.choice(["S2A", "S2B"], n),
        "lat": r.uniform(30, 48, n), "lon": r.uniform(-120, -70, n), "doy": r.integers(90, 300, n),
        "sun_zenith": r.uniform(20, 60, n), "view_zenith": r.uniform(0, 10, n), "sun_azimuth": r.uniform(100, 200, n),
        "view_azimuth": r.uniform(0, 360, n), "aot_c3": r.uniform(0.02, 0.4, n),
        "jrc_dist_land_m": r.uniform(30, 3000, n), "scl_c3_n_water": 9.0})
    df["added"] = np.arange(n) >= n - n_added
    for j, t in enumerate(D.TARGETS):
        v = 10 ** (0.4 * (sig[:, j] - 1.5) + 0.05 * r.standard_normal(n))
        v[r.random(n) < 0.3] = np.nan
        df[t] = np.where(df["added"], np.nan, v)
    for j, t in enumerate(T.SPECS):
        sp = T.SPECS[t]
        v = np.clip(10 ** (0.3 * (sig[:, 4 + j] - 1.5) + np.log10(sp.lo * 30)), sp.lo, sp.hi)
        have = r.random(n) < (0.9 if t in ("tp", "tn") else 0.6 if t == "doc" else 0.05)
        df[t] = np.where(have, v, np.nan)
        df[f"{t}_cens"], df[f"{t}_dl"] = False, np.nan
    c = np.flatnonzero(np.isfinite(df["tp"].to_numpy()))[::7]             # some left-censored TP rows
    df.loc[c, "tp"], df.loc[c, "tp_cens"], df.loc[c, "tp_dl"] = np.nan, True, T.SPECS["tp"].lo * 5
    none = df["added"] & ~(np.isfinite(df[list(T.SPECS)]).any(axis=1) | df["tp_cens"])
    df.loc[none, "tn"] = 1.0                                               # an added row carries a new-target value
    pix = {"refl": refl.astype(np.float16), "n_pix": n_pix, "offsets": r.integers(-7, 8, (n, 64, 2)).astype(np.int8),
           "n_water_win": r.integers(9, 226, n).astype(np.int32)}
    m = 40
    cidx = np.full((n, LS.K_SET), -1, np.int64)
    cdt = np.full((n, LS.K_SET), np.nan, np.float32)
    for i in range(0, n, 2):                                               # every second row has 1..6 context dates
        k = int(r.integers(1, LS.K_SET + 1))
        cidx[i, :k] = r.choice(m, k, replace=False)
        cdt[i, :k] = r.choice(np.r_[-183:-2, 3:184], k, replace=False)
    feats = {"dp_m": r.uniform(10, 300, (n, 64)).astype(np.float32),
             "ring": r.standard_normal((n, CF.RING_DIM)).astype(np.float32),
             "glint": r.uniform(0, 1, (n, 3)).astype(np.float32), "cidx": cidx, "cdt": cdt}
    cmeta = pd.DataFrame({"chip_id": [f"wqp:c{i}" for i in range(m)], "platform": r.choice(["S2A", "S2B"], m)})
    carr = {"refl": r.uniform(0.2, 3.0, (m, LS.P_CONTEXT, 12)).astype(np.float16),
            "n_pix": r.integers(1, LS.P_CONTEXT + 1, m).astype(np.int32),
            "offsets": r.integers(-7, 8, (m, LS.P_CONTEXT, 2)).astype(np.int8),
            "dp_m": r.uniform(10, 300, (m, LS.P_CONTEXT)).astype(np.float32),
            "q": r.standard_normal((m, LS.N_Q)).astype(np.float32)}
    return df, pix, feats, (cmeta, carr)


def _gloria(n=200, seed=1):
    r = np.random.default_rng(seed)
    g = pd.DataFrame({b: r.uniform(0.2, 3.0, n) for b in D.GLORIA_BANDS})
    for t in D.TARGETS:
        g[t] = 10 ** r.normal(0.5, 0.3, n)
    return g


def _cfg(arm, tmp_path=None):
    cfg = ARM.resolve_config(ARM_FILES[arm], smoke=True, smoke_base=V1_CFG)
    cfg["v1"].update(d=16, n_layers=1, n_heads=2)
    cfg["v1"]["pretrain"]["batch"] = 64
    cfg["v1"]["finetune"]["batch"] = 32
    return cfg


# ----------------------------------------------------------------------------------------- targets, heads, cells
def test_target_matrix_and_censored_values():
    df, *_ = _synth()
    tg = list(D.TARGETS) + ["tp", "doc"]
    y, cens = ARM.target_matrix(df, tg)
    assert y.shape == (len(df), 6) and not cens[:, :4].any() and not cens[:, 5].any() and cens[:, 4].sum() > 10
    c = cens[:, 4]
    assert np.allclose(y[c, 4], np.log10(T.SPECS["tp"].lo * 5)) and np.isnan(df["tp"].to_numpy()[c]).all()
    q = np.isfinite(df["chla"].to_numpy())
    assert np.allclose(y[q, 0], np.log10(df["chla"].to_numpy()[q])) and np.isnan(y[~q, 0]).all()
    bad = df.copy()
    bad.loc[np.flatnonzero(c)[0], "tp"] = 0.1
    with pytest.raises(ARM.ArmError, match="both quantified and censored"):
        ARM.target_matrix(bad, tg)
    bad = df.copy()
    bad.loc[np.flatnonzero(c)[0], "tp_dl"] = np.nan
    with pytest.raises(ARM.ArmError, match="no positive detection limit"):
        ARM.target_matrix(bad, tg)
    with pytest.raises(ARM.ArmError, match="unknown target"):
        ARM.target_matrix(df.assign(foo=1.0), ["foo"])


def test_heads_drop_rule_counts_fit_cell_values():
    df, *_ = _synth()
    fit, _ = F1.fit_and_inner(df, "base")
    tg, counts = ARM.heads(df, fit, "a1_c2")
    assert tg[:4] == list(D.TARGETS) and set(counts) == set(T.SPECS)
    for t in T.SPECS:
        n = int((np.isfinite(df[t]) | df[f"{t}_cens"]).to_numpy()[fit].sum())
        assert counts[t] == n and ((t in tg) == (n >= ARM.MIN_HEAD_FIT))
    assert "tp" in tg and "tn" in tg and "ssc" not in tg and "acdom440" not in tg     # as built by _synth
    assert ARM.heads(df, fit, "a0") == (list(D.TARGETS), {})
    with pytest.raises(ARM.ArmError, match="unknown arm"):
        ARM.heads(df, fit, "a9")


def test_cells_a0_uses_dev_v5_rows_and_both_arms_stop_on_the_same_rows():
    df, *_ = _synth()
    fit_all, inner_all = F1.fit_and_inner(df, "base")
    added = df["added"].to_numpy()
    a0 = ARM.cells(df, "a0", "base", list(D.TARGETS))
    tg, _ = ARM.heads(df, fit_all, "a1_c1")
    a1 = ARM.cells(df, "a1_c1", "base", tg)
    assert np.array_equal(a0["fit"], fit_all & ~added) and not a0["fit"][added].any()
    assert a1["fit"][added].any() and not (a1["fit"] & ~fit_all).any()
    assert np.array_equal(a0["inner"], a1["inner"]) and np.array_equal(a0["inner"], inner_all & ~added)
    assert a0["inner"].sum() >= 30 and not (a0["inner"] & a1["fit"]).any()
    for cl in (a0, a1):                                   # fit and inner rows: train role, train years only
        used = cl["fit"] | cl["inner"]
        assert set(df["role"][used]) == {"train"} and set(df["tblock"][used]) == {"train"}
    # an added row whose only values belong to dropped heads is not a fit row
    y, _ = ARM.target_matrix(df, tg)
    assert np.array_equal(a1["fit"], fit_all & np.isfinite(y).any(1))


# ----------------------------------------------------------------------------------------- data sets
def test_a0_set_is_the_v1_set_and_a1_sets_follow_the_configuration():
    df, pix, feats, cc = _synth()
    fit_all, _ = F1.fit_and_inner(df, "base")
    tg, _ = ARM.heads(df, fit_all, "a1_c1")
    m = ARM.cells(df, "a1_c1", "base", tg)["fit"]
    n4 = ARM.fit_norm("a0", df, pix, feats, m, list(D.TARGETS))
    a0, v1 = ARM.make_set("a0", df, pix, feats, cc, n4, m, list(D.TARGETS)), PL.make_set(df, pix, n4, m)
    for k in ("refl", "n_pix", "offsets", "lam", "geo", "ctx", "y"):
        assert np.array_equal(getattr(a0, k), getattr(v1, k), equal_nan=True), k
    with pytest.raises(ARM.ArmError, match="four base targets"):
        ARM.make_set("a0", df, pix, feats, cc, n4, m, tg)
    nm = ARM.fit_norm("a1_c1", df, pix, feats, m, tg)
    assert np.allclose(nm.y_mean[:4], n4.y_mean) and len(nm.g_mean) == len(PL.GEO_COLS) + 3
    c1 = ARM.make_set("a1_c1", df, pix, feats, cc, nm, m, tg)
    c2 = ARM.make_set("a1_c2", df, pix, feats, cc, nm, m, tg)
    c3 = ARM.make_set("a1_c3", df, pix, feats, cc, nm, m, tg)
    assert c1.base.ctx.shape[1] == 5 and c2.base.ctx.shape[1] == 2 and c3.base.ctx.shape[1] == 2
    assert np.array_equal(c2.base.ctx, c1.base.ctx[:, :2])               # DOY only: no latitude/longitude
    assert c1.base.geo.shape[1] == 10 and np.array_equal(c1.base.geo[:, :7], a0.geo)
    assert c1.cstore is not None and c2.cstore is not None and c3.cstore is None
    assert c1.y.shape[1] == len(tg) and c1.cens.shape == c1.y.shape and c1.cens.any()
    idx = np.arange(8)
    xb, y, ym = c2.batch(idx, "cpu")
    assert xb["ctx"].shape == (8, 2) and xb["geo"].shape == (8, 10) and xb["dp"].shape == (8, 64)
    assert "c_refl" in xb and xb["c_refl"].shape[1:] == (LS.P_CONTEXT, 12)
    xo, _, _ = ARM.SetOff(c2).batch(idx, "cpu")
    assert "c_refl" not in xo and torch.equal(xo["refl"], xb["refl"])
    x3, _, _ = c3.batch(idx, "cpu")
    assert "c_refl" not in x3
    # the mean and SD of a target ignore censored entries
    yq, cens = ARM.target_matrix(df, tg)
    j = tg.index("tp")
    assert nm.y_mean[j] == pytest.approx(np.nanmean(np.where(cens, np.nan, yq)[m, j]))


def test_init_a0_is_the_pretrained_v1_network_and_a1_takes_its_encoder():
    cfg = _cfg("a1_c2")
    torch.manual_seed(3)
    pre = ARM.build_v1(cfg["v1"])
    a0, _ = ARM.init_network("a0", cfg["v1"], pre, 4, 7, "cpu")
    assert a0 is pre
    m1, done = ARM.init_network("a1_c2", cfg["v1"], pre, 7, 7, "cpu")
    m2, _ = ARM.init_network("a1_c2", cfg["v1"], pre, 7, 7, "cpu")
    m3, _ = ARM.init_network("a1_c2", cfg["v1"], pre, 7, 8, "cpu")
    assert isinstance(m1, LS.SatWQLakeSetMT) and m1.use_set and m1.nt == 7 and m1.ctx[0].in_features == 2
    assert ARM.build_model("a1_c1", cfg["v1"], 7).ctx[0].in_features == 5
    assert not ARM.build_model("a1_c3", cfg["v1"], 7).use_set
    sd, sp = m1.state_dict(), pre.state_dict()
    assert done and all(n.split(".")[0] in LS.TRANSFER for n in done)
    for n in done:
        if n != "pix_pos.0.weight":
            assert torch.equal(sd[n], sp[n]), n
    assert torch.equal(sd["pix_pos.0.weight"][:, :3], sp["pix_pos.0.weight"]) and not sd["pix_pos.0.weight"][:, 3].any()
    assert torch.equal(sd["head.weight"], m2.state_dict()["head.weight"])            # same seed, same new layers
    assert not torch.equal(sd["head.weight"], m3.state_dict()["head.weight"])


# ----------------------------------------------------------------------------------------- one network, end to end
@pytest.mark.parametrize("arm", list(ARM.ARMS))
def test_train_network_smoke_on_synthetic_rows(tmp_path, arm):
    df, pix, feats, cc = _synth()
    cfg = _cfg(arm)
    res = ARM.train_network(cfg, df, pix, feats, cc, unit=1, init=2, device="cpu", gloria=_gloria(),
                            run_dir=tmp_path / "run", pretrain_cache=tmp_path / "pre", smoke_n_fit=200,
                            log=lambda *_: None)
    assert res["complete"] and res["seed"] == 102 and res["cells"]["fit_used"].sum() == 200
    assert not (res["cells"]["fit_used"] & ~res["cells"]["fit"]).any()
    assert (tmp_path / "run" / "finetune.ckpt").is_file() and len(list((tmp_path / "pre").glob("*.pt"))) == 1
    tg = res["targets"]
    assert tg[:4] == list(D.TARGETS) and (len(tg) == 4) == (arm == "a0")
    ds = ARM.make_set(arm, df, pix, feats, cc, res["norm"], res["cells"]["inner"], tg)
    w, mu, sg = V1.predict_mixture(res["model"], ds, "cpu", res["norm"].y_mean, res["norm"].y_std)
    assert w.shape == (int(res["cells"]["inner"].sum()), len(tg), 5) and np.isfinite(mu).all() and (sg > 0).all()
    assert ("p_set_drop" in res["aug"]) == (arm != "a0")
    ARM.save_network(tmp_path / "run" / "model.pt", cfg, res)
    st = torch.load(tmp_path / "run" / "model.pt", weights_only=False)
    assert st["arm"] == arm and st["targets"] == tg and st["seed"] == 102
    again = ARM.build_model(arm, cfg["v1"], len(tg))
    again.load_state_dict(st["state_dict"])
    # a second network of the same seed (any arm) starts from the stored pretrained v1 network
    other = "a1_c3" if arm == "a0" else "a0"
    res2 = ARM.train_network(_cfg(other), df, pix, feats, cc, unit=1, init=2, device="cpu", gloria=None,
                             run_dir=tmp_path / "run2", pretrain_cache=tmp_path / "pre", smoke_n_fit=200,
                             log=lambda *_: None)
    assert res2["pretrain"]["reused"] and res2["pretrain"]["key"] == res["pretrain"]["key"]
    assert len(list((tmp_path / "pre").glob("*.pt"))) == 1


def test_set_off_predictions_ignore_the_context_and_seed_design_is_enforced(tmp_path):
    df, pix, feats, cc = _synth()
    cfg = _cfg("a1_c2")
    kw = dict(device="cpu", gloria=_gloria(), pretrain_cache=None, smoke_n_fit=150, log=lambda *_: None)
    for unit, init in [(3, 0), (0, 3), (-1, 0)]:
        with pytest.raises(ARM.ArmError, match="outside the seed-unit design"):
            ARM.train_network(cfg, df, pix, feats, cc, unit=unit, init=init, run_dir=tmp_path / "x", **kw)
    res = ARM.train_network(cfg, df, pix, feats, cc, unit=0, init=0, run_dir=tmp_path / "r", **kw)
    inner = res["cells"]["inner"]
    ds = ARM.make_set("a1_c2", df, pix, feats, cc, res["norm"], inner, res["targets"])
    cm, ca = cc
    other = (cm, {**ca, "refl": (ca["refl"].astype(np.float32) * 2.0).astype(np.float16)})
    ds2 = ARM.make_set("a1_c2", df, pix, feats, other, res["norm"], inner, res["targets"])
    p = lambda d: V1.predict_mixture(res["model"], d, "cpu", res["norm"].y_mean, res["norm"].y_std)[1]  # noqa: E731
    assert np.allclose(p(ARM.SetOff(ds)), p(ARM.SetOff(ds2)))
    has = (feats["cidx"][inner] >= 0).any(1)
    d = np.abs(p(ds) - p(ds2)).max(axis=(1, 2))
    assert (d[has] > 0).all() and np.allclose(d[~has], 0.0)               # only rows with a context set change


def test_time_budget_stops_with_a_checkpoint_and_the_same_call_continues(tmp_path):
    df, pix, feats, cc = _synth()
    cfg = _cfg("a1_c3")
    cfg["v1"]["finetune"]["epochs"] = 6
    cfg["checkpoint_every"] = 1
    kw = dict(unit=0, init=1, device="cpu", gloria=_gloria(), pretrain_cache=tmp_path / "pre", smoke_n_fit=150,
              log=lambda *_: None)
    full = ARM.train_network(cfg, df, pix, feats, cc, run_dir=tmp_path / "a", **kw)
    orig = TR.run_epochs_ema

    def short(*a, **k):
        k["max_seconds"] = 0.0                            # the budget runs out after the first epoch
        return orig(*a, **k)

    TR.run_epochs_ema = short
    try:
        part = ARM.train_network(cfg, df, pix, feats, cc, run_dir=tmp_path / "b", **kw)
    finally:
        TR.run_epochs_ema = orig
    assert not part["complete"] and len(part["history"]["history"]) == 1
    rest = ARM.train_network(cfg, df, pix, feats, cc, run_dir=tmp_path / "b", **kw)
    assert rest["complete"] and len(rest["history"]["history"]) == len(full["history"]["history"])
    for k, v in full["model"].state_dict().items():
        assert torch.allclose(v, rest["model"].state_dict()[k], atol=1e-6), k


# ----------------------------------------------------------------------------------------- script gate
def _script():
    import importlib.util

    spec = importlib.util.spec_from_file_location("orbitv2_arm_script", REPO / "scripts" / "train" / "orbitv2_arm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_refuses_a_real_run_before_the_stamp_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path / "runs")
    mod = _script()
    monkeypatch.setattr(mod.PL, "setup_gpu", lambda *a, **k: pytest.fail("a GPU was requested"))
    with pytest.raises(ARM.ArmError, match="not set"):                    # as shipped: no selected v1 configuration
        mod.main(["--config", str(ARM_FILES["a0"]), "--unit", "0", "--init", "0"])
    f = _arm_file(tmp_path, base=_base())
    monkeypatch.setattr(ST, "stamp_status", lambda *a, **k: {"stamped": False, "amendment_sha256": "0" * 64,
                                                             "row_id": None, "timestamp": None, "n_rows": 0,
                                                             "rulings_sha256": "0" * 64, "reason": "test"})
    with pytest.raises(ST.StampError, match="refused before the lead's stamp"):
        mod.main(["--config", str(f), "--unit", "0", "--init", "0"])
    for extra in (["--data-dir", str(tmp_path)], ["--device", "cpu"]):
        with pytest.raises(SystemExit, match="smoke tests"):
            mod.main(["--config", str(f), "--unit", "0", "--init", "0", *extra])
    with pytest.raises(SystemExit, match="needs --data-dir"):
        mod.main(["--config", str(f), "--unit", "0", "--init", "0", "--smoke", "--smoke-base", V1_CFG])
    with pytest.raises(SystemExit, match="not the dev_v5m cache"):
        mod.main(["--config", str(f), "--unit", "0", "--init", "0", "--smoke", "--smoke-base", V1_CFG,
                  "--data-dir", str(tmp_path / "runs" / "orbit" / "cache" / "dev_v5m")])
    assert not (tmp_path / "runs").exists()
