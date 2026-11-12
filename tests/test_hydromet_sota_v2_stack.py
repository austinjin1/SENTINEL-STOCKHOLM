# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M6 architecture-v2 (stack solver and builder, decision rules, inner-row decision script, P driver helpers):
code-path and property tests on synthetic arrays. No real observation is read and no number printed here is a
research result."""

import importlib.util
import itertools
import json
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from sentinel.hydromet_sota.gso_mt import criterion as C  # noqa: E402
from sentinel.hydromet_sota.gso_mt import gates as GT  # noqa: E402
from sentinel.hydromet_sota.gso_mt import nets as N  # noqa: E402
from sentinel.hydromet_sota.gso_mt import runner as VR  # noqa: E402
from sentinel.hydromet_sota.gso_mt.data import DAILY, PRIMARY  # noqa: E402
from sentinel.hydromet_sota.stack import build as B  # noqa: E402
from sentinel.hydromet_sota.stack import convex as X  # noqa: E402

ROOT = Path(VR.__file__).resolve().parents[3]
CF = {v: 1.0 for v in PRIMARY}


def _script(rel):
    spec = importlib.util.spec_from_file_location(Path(rel).stem, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------------ simplex solver
def _grid_min(P, y, r, step=0.01):
    """Brute force over the simplex grid (M = 3)."""
    best, bw = np.inf, None
    k = int(round(1 / step))
    for i, j in itertools.product(range(k + 1), repeat=2):
        if i + j > k:
            continue
        w = np.array([i, j, k - i - j]) / k
        s = float((r * (y - P @ w) ** 2).sum())
        if s < best:
            best, bw = s, w
    return bw, best


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_simplex_lstsq_is_the_constrained_minimum(seed):
    rs = np.random.default_rng(seed)
    y = rs.normal(size=300)
    P = y[:, None] + rs.normal(0, [0.3, 0.6, 1.0], (300, 3)) + [0.0, 0.2, -0.1]
    r = rs.uniform(0.1, 1.0, 300)
    w = X.simplex_lstsq(P, y, r)
    assert w.min() >= 0 and w.sum() == pytest.approx(1.0, abs=1e-12)
    bw, best = _grid_min(P, y, r)
    sse = float((r * (y - P @ w) ** 2).sum())
    assert sse <= best + 1e-9                                   # at least as good as every grid point
    assert np.abs(w - bw).max() < 0.02


def test_simplex_lstsq_known_cases_and_ties():
    rs = np.random.default_rng(0)
    y = rs.normal(size=50)
    P = np.stack([y, y + 1.0, rs.normal(size=50)], axis=1)
    assert X.simplex_lstsq(P, y) == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)        # a perfect member takes it all
    P2 = np.stack([y - 1.0, y + 1.0], axis=1)                                       # the mean of two is exact
    assert X.simplex_lstsq(P2, y) == pytest.approx([0.5, 0.5], abs=1e-9)
    P3 = np.stack([y + 0.5, y + 0.5], axis=1)                                       # identical members: listed first
    assert X.simplex_lstsq(P3, y) == pytest.approx([1.0, 0.0])
    w = X.simplex_lstsq(np.stack([y + 3.0, -y], axis=1), y)                          # negative unconstrained weight
    assert w.min() >= 0 and w.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        X.simplex_lstsq(np.full((3, 2), np.nan), np.zeros(3))
    with pytest.raises(ValueError):
        X.simplex_lstsq(np.zeros((3, 2)), np.zeros(3), r=np.zeros(3))


def test_row_weights_change_the_solution_and_zero_weight_rows_do_not_count():
    rs = np.random.default_rng(1)
    y = rs.normal(size=200)
    P = np.stack([y + rs.normal(0, 0.2, 200), y + rs.normal(0, 0.2, 200)], axis=1)
    P[:100, 0] += 2.0                                           # member 0 is bad on the first half
    r = np.r_[np.zeros(100), np.ones(100)]
    assert X.simplex_lstsq(P, y, r) == pytest.approx(X.simplex_lstsq(P[100:], y[100:]), abs=1e-9)
    assert X.simplex_lstsq(P, y)[0] < X.simplex_lstsq(P, y, r)[0]


def test_cross_fit_never_uses_the_held_out_group():
    rs = np.random.default_rng(2)
    y = rs.normal(size=120)
    P = np.stack([y + rs.normal(0, 0.3, 120), y + rs.normal(0, 0.5, 120)], axis=1)
    g = np.repeat(np.arange(4), 30)
    pred, folds = X.cross_fit(P, y, g)
    assert set(folds) == {"0", "1", "2", "3"} and np.isfinite(pred).all()
    y2, P2 = y.copy(), P.copy()
    y2[g == 2] += 100.0                                         # changing group 2 leaves its own fold's weights
    P2[g == 2] *= -3.0
    _, folds2 = X.cross_fit(P2, y2, g)
    assert folds2["2"] == pytest.approx(folds["2"], abs=1e-12)
    assert pred[g == 2] == pytest.approx(P[g == 2] @ folds["2"])
    one, f1 = X.cross_fit(P, y, np.zeros(120, int))             # one group: the full-data weights
    assert one == pytest.approx(P @ X.simplex_lstsq(P, y)) and list(f1) == ["0"]


def test_grab_row_weights_are_those_of_j_grab():
    ok = np.zeros((3, 50), bool)
    ok[0, :40], ok[1, :5] = True, True
    w = X.grab_row_weights(ok, 30)
    assert w[0, 0] == pytest.approx(0.75) and w[1, 0] == 1.0 and w[2].sum() == 0 and w[0, 45] == 0
    rs = np.random.default_rng(0)                               # Σ w·e² / Σ w equals the criterion's term
    y = np.where(ok, rs.normal(size=ok.shape), np.nan)
    p = rs.normal(size=ok.shape)
    yi, pi = np.stack([y, y], -1), np.stack([p, p], -1)
    want = C.j_grab(pi, yi)["by_variable"]["T"]
    assert float((w[ok] * (p[ok] - y[ok]) ** 2).sum() / w[ok].sum()) == pytest.approx(want)


# ------------------------------------------------------------------------------------------------ stack builder
def _members(n=8, T=60, m=6, seed=0, names=("rtd", "g1", "g2")):
    rs = np.random.default_rng(seed)
    y = rs.normal(size=(n, T, 7))
    y[:, ::7, :] = np.nan
    yi = np.where(rs.uniform(size=(m, T, 2)) < 0.4, rs.normal(size=(m, T, 2)), np.nan)
    mem = {}
    for k, nm in enumerate(names):
        mem[nm] = {"daily": np.where(np.isfinite(y), y, 0.0) + rs.normal(0, 0.2 + 0.2 * k, y.shape),
                   "inst": np.where(np.isfinite(yi), yi, 0.0) + rs.normal(0, 0.2 + 0.2 * k, yi.shape)}
    return mem, y, yi, np.repeat(np.arange(4), n // 4), np.repeat(np.arange(3), m // 3)


def test_fit_stack_weights_cross_fitted_predictions_and_criterion():
    mem, y, yi, g, gg = _members()
    res = B.fit_stack(mem, y, yi, g, gg, CF, form="grab", min_rows=20)
    for ep in list(PRIMARY) + ["T", "DO"]:
        w = res["weights"][ep]
        assert list(w) == ["rtd", "g1", "g2"] and sum(w.values()) == pytest.approx(1.0) and min(w.values()) >= 0
        assert res["folds"][ep]["n_folds"] == (4 if ep in PRIMARY else 3)
    assert res["n_rows_dropped"] == 0 and res["form"] == "grab"
    js = C.j_sonde(res["daily"], y, CF)
    assert res["j_sonde"] == pytest.approx(js["j_sonde"]) and res["J"] == pytest.approx(
        0.8 * res["j_sonde"] + 0.2 * res["j_grab"])
    j = DAILY.index("T_mean")                                   # a stack is no worse in-sample than its best member
    ok = np.isfinite(y[:, :, j])
    full = B.apply_weights(res["weights"]["temp_c_mean"], {k: v["daily"][:, :, j] for k, v in mem.items()})
    assert np.mean((full[ok] - y[:, :, j][ok]) ** 2) <= min(
        np.mean((v["daily"][:, :, j][ok] - y[:, :, j][ok]) ** 2) for v in mem.values()) + 1e-12
    assert np.isnan(res["daily"][:, :, [DAILY.index(k) for k in DAILY[:7] if k not in PRIMARY.values()]]).all()


def test_fit_stack_no_grab_form_uses_no_grab_row():
    mem, y, yi, g, gg = _members()
    res = B.fit_stack(mem, y, yi, g, gg, CF, form="nograb", min_rows=20)
    assert res["inst"] is None and res["j_grab"] is None and res["J"] == pytest.approx(res["j_sonde"])
    assert res["weights"]["T"] == res["weights"]["temp_c_mean"] and res["weights"]["DO"] == res["weights"]["do_mgl_mean"]
    assert res["rows"]["T"]["used"] == 0 and res["rows"]["T"]["from"] == "temp_c_mean"
    poisoned = {k: {"daily": v["daily"], "inst": v["inst"] + 1e6} for k, v in mem.items()}
    res2 = B.fit_stack(poisoned, y, yi + 1e6, g, gg, CF, form="nograb", min_rows=20)
    assert res2["weights"] == res["weights"] and res2["j_sonde"] == res["j_sonde"]


def test_fit_stack_counts_rows_a_member_cannot_score_and_skips_an_absent_member():
    mem, y, yi, g, gg = _members()
    mem["g1"]["daily"][0, 1:5, DAILY.index("T_mean")] = np.nan
    res = B.fit_stack(mem, y, yi, g, gg, CF, min_rows=20)
    assert res["n_rows_dropped"] == 4 and res["rows"]["temp_c_mean"]["used"] == res["rows"]["temp_c_mean"]["labelled"] - 4
    mem["g2"]["inst"] = None                                    # a member without hour-resolved output is left out
    res = B.fit_stack(mem, y, yi, g, gg, CF, min_rows=20)
    assert list(res["weights"]["T"]) == ["rtd", "g1"]
    few = B.fit_stack(mem, y, yi, g, gg, CF, min_rows=10 ** 6)
    assert all(v is None for v in few["weights"].values()) and few["j_sonde"] is None


def test_member_run_dirs_follow_the_label_form(tmp_path):
    sel = {"d1": "g2", "d2": None}
    f = lambda m, form: [p.name for p in B.member_run_dirs(tmp_path, m, form, sel)]  # noqa: E731
    assert f("d1", "grab") == ["d1_g2_s0", "d1_g2_s1", "d1_g2_s2"]
    assert f("d1", "nograb")[0] == "d1_abl_nograb_s0" and f("d2", "grab") == [] and f("d2", "nograb")[0] == "d2_nograb_s0"
    assert f("rtd", "grab")[0] == "rtd_fixed_s0" and f("rtd", "nograb")[0] == "rtd_published_s0"
    assert f("rtd_w", "grab")[2] == "rtd_w_fixed_s2" and f("rtd_w", "nograb")[1] == "rtd_w_published_s1"
    assert f("g1", "grab")[0] == "g1_sel_s0" and f("g2", "nograb")[1] == "g2_nograb_s1" and f("p", "nograb") == ["p_fixed"]
    with pytest.raises(ValueError):
        B.member_run_dirs(tmp_path, "c2a", "grab", sel)


REL = {"amendment_sha256": "a" * 64, "amendment_registry_id": "row-a", "deviations_sha256": "d" * 64,
       "deviations_registry_id": "row-d", "pins_sha256": "b" * 64, "changed_under_stamp": {},
       "code_pins_sha256": "c" * 64, "code_changed_under_stamp": {}, "reader": VR.POOL_READER}


def _run(d, daily, inst=None, n_inner=2, n_grab_inner=3, date_start="2005-10-01", done=True, release=REL):
    d.mkdir(parents=True)
    if release is not None:
        VR.write_json(d / "release.json", release)
    kw = {"daily": daily, "date_start": date_start, "n_inner": n_inner, "n_grab_inner": n_grab_inner}
    if inst is not None:
        kw["inst"] = inst
    np.savez_compressed(d / "inner_pred.npz", **kw)
    if done:
        (d / ".done").touch()


def test_load_member_is_the_seed_mean_and_refuses_an_incomplete_seed_set(tmp_path):
    dirs = [tmp_path / f"rtd_fixed_s{s}" for s in range(3)]
    kw = dict(n_inner=2, n_grab_inner=3, date_start="2005-10-01", release=REL)
    assert B.load_member(dirs, **kw) is None
    with pytest.raises(TypeError):
        B.load_member(dirs, n_inner=2, n_grab_inner=3, date_start="2005-10-01")   # the state is not optional
    for s, d in enumerate(dirs[:2]):
        _run(d, np.full((2, 4, 7), float(s)), np.full((3, 4, 2), 10.0 * s))
    with pytest.raises(RuntimeError, match="2 of 3"):
        B.load_member(dirs, **kw)
    _run(dirs[2], np.full((2, 4, 7), 2.0), np.full((3, 4, 2), 20.0), done=False)
    with pytest.raises(RuntimeError, match="2 of 3"):                       # not finished = not complete
        B.load_member(dirs, **kw)
    (dirs[2] / ".done").touch()
    z = B.load_member(dirs, **kw)
    assert z["n_seeds"] == 3 and (z["daily"] == 1.0).all() and (z["inst"] == 10.0).all()
    assert z["combined"]["n_runs"] == 3 and z["combined"]["code_pins_sha256"] == REL["code_pins_sha256"]
    with pytest.raises(RuntimeError, match="pinned pool"):
        B.load_member(dirs, n_inner=5, n_grab_inner=3, date_start="2005-10-01", release=REL)
    with pytest.raises(RuntimeError, match="inner grab set"):
        B.load_member(dirs, n_inner=2, n_grab_inner=9, date_start="2005-10-01", release=REL)


def test_load_member_refuses_runs_of_another_code_pin_or_pin_state(tmp_path):
    """Entry-1 lead rulings, decision 4 and decision 2 condition 4."""
    kw = dict(n_inner=2, n_grab_inner=3, date_start="2005-10-01")
    dirs = [tmp_path / f"g1_sel_s{s}" for s in range(3)]
    for s, d in enumerate(dirs):
        _run(d, np.full((2, 4, 7), float(s)), release={**REL, "code_pins_sha256": ("c" if s < 2 else "e") * 64})
    with pytest.raises(VR.NotStampedError, match="another state"):            # one seed under another code pin
        B.load_member(dirs, release=REL, **kw)
    with pytest.raises(VR.NotStampedError, match="another state"):            # the process itself is in another
        B.load_member(dirs[:2], release={**REL, "code_pins_sha256": "e" * 64}, **kw)
    assert B.load_member(dirs[:2], release=REL, **kw)["n_seeds"] == 2
    with pytest.raises(VR.NotStampedError, match="another state"):            # a superseded pin state is void
        B.load_member(dirs[:2], release={**REL, "changed_under_stamp": {"configs/hydromet_sota/v2/d1.yaml": "0" * 64}},
                      **kw)
    (dirs[0] / "release.json").unlink()
    with pytest.raises(VR.NotStampedError, match="no release record"):
        B.load_member(dirs[:2], release=REL, **kw)


# ------------------------------------------------------------------------------------------------ decision rules
def test_select_point_takes_the_lowest_j_and_breaks_ties_to_the_earlier_point():
    assert GT.select_point({"g1": 0.50, "g2": 0.40, "g3": 0.45}) == "g2"
    assert GT.select_point({"g1": 0.40005, "g2": 0.40, "g3": 0.39996}) == "g1"      # all within 1e-4 of the lowest
    assert GT.select_point({"g1": 0.4002, "g2": 0.40, "g3": 0.40}) == "g2"
    assert GT.select_point({"g1": None, "g2": float("nan"), "g3": 0.7}) == "g3"
    with pytest.raises(ValueError):
        GT.select_point({"g1": None})


def test_gate_d1a_is_one_outcome_on_the_three_seed_mean_of_j_sonde():
    r = GT.g_d1a([0.95, 0.96, 0.97], [1.0, 1.0, 1.0])
    assert r["passed"] and r["label_form"] == "grab" and r["relative_reduction"] == pytest.approx(0.04)
    assert GT.g_d1a([0.5, 1.0, 1.25], [1.0, 1.0, 1.0])["passed"]                     # the mean decides, not a seed
    r = GT.g_d1a([0.971, 0.971, 0.971], [1.0, 1.0, 1.0])
    assert not r["passed"] and r["label_form"] == "nograb"
    assert not GT.g_d1a([1.1, 1.1, 1.1], [1.0, 1.0, 1.0])["passed"]
    for bad in (([0.9, 0.9], [1.0, 1.0, 1.0]), ([0.9, 0.9, float("nan")], [1.0, 1.0, 1.0])):
        with pytest.raises(ValueError):
            GT.g_d1a(*bad)


def test_comparator_set_holds_both_rtd_recipes_in_both_forms_and_the_reference_stacks_for_d3():
    base = GT.comparator_set("d1")
    assert {("rtd", "grab"), ("rtd", "published"), ("rtd_w", "grab"), ("rtd_w", "published")} <= set(base)
    assert {m for m, _ in base} == {"rtd", "rtd_w", "g1", "g2", "p"} and len(base) == 7
    d3 = GT.comparator_set("d3")
    assert len(d3) == 9 and {m for m, _ in d3[7:]} == {"ref_stack", "ref_stack_w"}
    assert GT.comparator_set("d2") == base


def test_gate_d1b_needs_every_comparator():
    keys = [f"{m}/{f}" for m, f in GT.comparator_set("d1")]
    good = {k: dict(zip(GT.PRIMARY_TB, (0.01, 0.0, 0.02, -0.01))) for k in keys}
    assert GT.g_d1b(good, "T-B", "d1")["go"]                                        # 3 of 4, Δ = 0 counts
    bad = json.loads(json.dumps(good))
    bad["rtd_w/published"]["temp_c_mean"] = -0.001                                  # 2 of 4 against one comparator
    r = GT.g_d1b(bad, "T-B", "d1")
    assert not r["go"] and not r["by_comparator"]["rtd_w/published"]["go"] and r["by_comparator"]["rtd/grab"]["go"]
    miss = {k: v for k, v in good.items() if k != "p/one"}
    assert not GT.g_d1b(miss, "T-B", "d1")["go"]                                    # a missing comparator is a no-go
    assert not GT.g_d1b(good, "T-B", "d3")["go"]                                    # D3 also needs both ref stacks
    n2 = {k: {"T": 0.01, "DO": 0.0} for k in keys}
    assert GT.g_d1b(n2, "N2", "d1")["go"]
    n2["g2/gate"]["DO"] = -1e-9
    assert not GT.g_d1b(n2, "N2", "d1")["go"]                                       # N2: both endpoints
    n2["g2/gate"]["DO"] = float("nan")
    assert not GT.g_d1b(n2, "N2", "d1")["go"]


def test_min_groups_floors():
    r = GT.min_groups([1] * 15)
    assert r == {"G": 15, "G_eff": 15.0, "n_sites": 15, "claim_endpoint": True}
    assert not GT.min_groups([1] * 14)["claim_endpoint"]
    r = GT.min_groups([100] + [1] * 20)                                             # one dominant group
    assert r["G"] == 21 and r["G_eff"] == pytest.approx(120 ** 2 / (100 ** 2 + 20), abs=0.01) and not r["claim_endpoint"]
    assert GT.min_groups([0, 0]) == {"G": 0, "G_eff": 0.0, "n_sites": 0, "claim_endpoint": False}
    assert GT.min_groups([3, 0, 2])["G"] == 2


# ------------------------------------------------------------------------------------------------ inner-row script
def _nn_run(root, name, J, js, jg, done=True, release=REL):
    d = root / "d1" / name
    d.mkdir(parents=True)
    if release is not None:
        VR.write_json(d / "release.json", release)
    (d / "train_info.json").write_text(json.dumps({"best_J": J, "best_j_sonde": js, "best_j_grab": jg,
                                                   "best_epoch": 3}))
    if done:
        (d / ".done").touch()


def test_inner_script_selects_gates_and_writes_once(tmp_path):
    S = _script("scripts/eval/sota_hydromet_v2_inner.py")
    grid = ["g1", "g2", "g3"]
    _nn_run(tmp_path, "d1_g1_s0", 0.50, 0.45, 0.70)
    _nn_run(tmp_path, "d1_g2_s0", 0.40, 0.36, 0.56)
    with pytest.raises(SystemExit, match="g3 seed 0 is not finished"):
        S.select(tmp_path, "d1", grid, REL)
    _nn_run(tmp_path, "d1_g3_s0", 0.40004, 0.35, 0.60, done=False, release={**REL, "code_pins_sha256": "e" * 64})
    with pytest.raises(SystemExit):                                                 # no .done = not finished
        S.select(tmp_path, "d1", grid, REL)
    (tmp_path / "d1" / "d1_g3_s0" / ".done").touch()
    with pytest.raises(VR.NotStampedError, match="another state"):                  # a grid point of another code pin
        S.select(tmp_path, "d1", grid, REL)
    VR.write_json(tmp_path / "d1" / "d1_g3_s0" / "release.json", REL)
    rec = S.select(tmp_path, "d1", grid, REL)
    assert rec["selected"] == "g2" and list(rec["seed0_J"]) == grid                # tie within 1e-4: earlier point
    assert rec["release"] == REL and rec["combined_runs"]["n_runs"] == 3
    with pytest.raises(SystemExit, match="seed 1 is not finished"):
        S.gate_d1a(tmp_path, "g2", REL)
    for s in (1, 2):
        _nn_run(tmp_path, f"d1_g2_s{s}", 0.41, 0.36, 0.58)
    for s in (0, 1, 2):
        _nn_run(tmp_path, f"d1_abl_nograb_s{s}", 0.40, 0.40, None, release=None if s == 2 else REL)
    with pytest.raises(VR.NotStampedError, match="no release record"):              # an ablation seed without a record
        S.gate_d1a(tmp_path, "g2", REL)
    VR.write_json(tmp_path / "d1" / "d1_abl_nograb_s2" / "release.json", REL)
    with pytest.raises(VR.NotStampedError, match="another state"):
        S.gate_d1a(tmp_path, "g2", {**REL, "pins_sha256": "0" * 64})
    g = S.gate_d1a(tmp_path, "g2", REL)
    assert g["release"] == REL and g["combined_runs"]["n_runs"] == 6
    assert g["passed"] and g["label_form"] == "grab" and g["relative_reduction"] == pytest.approx(0.1)
    assert g["j_grab_selected_reported_only"] == [0.56, 0.58, 0.58]
    f = tmp_path / "gates" / "g_d1a.json"
    assert S.write_once(f, g, ("passed", "label_form")) is True
    assert S.write_once(f, g, ("passed", "label_form")) is False
    with pytest.raises(SystemExit, match="written once"):
        S.write_once(f, {**g, "passed": False, "label_form": "nograb"}, ("passed", "label_form"))
    labels = S.model_dirs(tmp_path)
    assert set(labels) == {"d1_g1", "d1_g2", "d1_g3", "d1_abl_nograb"} and len(labels["d1_g2"]) == 3


def test_inner_script_never_names_a_validation_block_reader():
    src = (ROOT / "scripts/eval/sota_hydromet_v2_inner.py").read_text()
    code = src.split('"""', 2)[2]
    for word in ("val_idx", "test_idx", "load_v2_val", "ea_data", "WY2026", "val_group", "test_group"):
        assert word not in code


# ------------------------------------------------------------------------------------------------ P driver helpers
def test_physics_driver_member_choice_and_views():
    S = _script("scripts/train/sota_hydromet_v2_physics.py")
    assert S.choose({"a": 0.5, "b": 0.5}, ("a", "b")) == "a" and S.choose({"a": 0.5, "b": 0.4}, ("a", "b")) == "b"
    assert S.choose({"a": None, "b": 0.9}, ("a", "b")) == "b"
    with pytest.raises(RuntimeError):
        S.choose({"a": None}, ("a",))
    from sentinel.models.hydromet.data import FORCING_VARS

    n = len(FORCING_VARS)
    d = types.SimpleNamespace(forc=np.arange(2 * 3 * 2 * n, dtype="float32").reshape(2, 3, 2 * n))
    a, b = S.member_view(d, "site_cell"), S.member_view(d, "basin_mean")
    assert a.forc.shape == b.forc.shape == (2, 3, n) and d.forc.shape == (2, 3, 2 * n)
    assert (a.forc == d.forc[:, :, :n]).all() and (b.forc == d.forc[:, :, n:]).all()
    with pytest.raises(RuntimeError):
        S.member_view(types.SimpleNamespace(forc=np.zeros((2, 3, n))), "site_cell")


# ------------------------------------------------------------------------------------------------ arm release, decisions
def test_d2_needs_its_own_stamped_entry(tmp_path, monkeypatch):
    dv = tmp_path / "dev.md"
    monkeypatch.setattr(VR, "DEVIATIONS", dv)
    for arm in ("d1", "rtd", "rtd_w", "g1", "g2", "p", "ref_stack", "ref_stack_w", "d3"):
        VR.assert_arm_released({"arm": arm})                                        # no token needed
    VR.assert_arm_released({"arm": "d2", "_smoke": True})
    with pytest.raises(VR.NotStampedError):
        VR.assert_arm_released({"arm": "d2"})
    dv.write_text("entry 1 ...\n")
    with pytest.raises(VR.NotStampedError):
        VR.assert_arm_released({"arm": "d2"})
    tok = VR.arm_release_token("d2")
    assert tok.startswith("D2 RELEASED: sentinel/hydromet_sota/thermox/nets.py sha256 ")
    assert "file missing" not in tok
    for text in (f"the line `{tok}` is added by the D2 entry\n", f"  {tok}\n", f"{tok} (draft)\n", f"> {tok}\n",
                 tok[:-1] + "\n", tok.replace("sha256 ", "sha256 <sha256> ") + "\n"):
        dv.write_text("entry 1 ...\n" + text)                                # review F6: a whole line, nothing else
        with pytest.raises(VR.NotStampedError, match="whole line"):
            VR.assert_arm_released({"arm": "d2"})
    dv.write_text("entry 1 ...\n" + tok + "\n")
    VR.assert_arm_released({"arm": "d2"})
    dv.write_text("entry 1 ...\r\n" + tok + "\r\nmore\r\n")
    VR.assert_arm_released({"arm": "d2"})


def test_decision_rows_are_inner_rows_and_never_stamps(monkeypatch):
    import sentinel.core.registry as REG

    seen = {}
    monkeypatch.setattr(REG, "log_result", lambda **kw: seen.update(kw) or {"id": "x"})
    with pytest.raises(TypeError):                                            # the release record is not optional
        VR.log_decision("gate_G-D1a", "d1", {"passed": True}, rdir="/tmp/x", value=0.05, notes="n")
    with pytest.raises(VR.NotStampedError, match="release fields"):
        VR.log_decision("gate_G-D1a", "d1", {"passed": True}, rdir="/tmp/x", release={"reader": "x"}, value=0.05)
    assert not seen
    VR.log_decision("gate_G-D1a", "d1", {"passed": True}, rdir="/tmp/x", release=REL, value=0.05, notes="n")
    assert seen["experiment"] == "M6V2/decisions_v1" and not seen["experiment"].endswith(VR.STAMP_SUFFIX)
    assert seen["experiment"] != VR.STAMP_EXPERIMENT and "written_by" not in seen["extra"]
    assert seen["workstream"] == VR.WORKSTREAM and seen["split"].startswith("inner") and seen["value"] == 0.05
    assert seen["extra"] == {"passed": True, "release": REL} and seen["metric"] == "gate_G-D1a"


# ------------------------------------------------------------------------------------------------ multi-layer R-TD-W
NET = {"hidden": 12, "heads": "indep_op", "static": "concat", "dropout": 0.2, "static_noise": 0.3, "attr_dropout": 0.0}


def test_multi_layer_network_trains_and_one_layer_keeps_its_parameter_names():
    torch = pytest.importorskip("torch")
    one = N.make_net(7, 5, dict(NET))
    same = N.make_net(7, 5, {**NET, "layers": 1})
    assert list(one.state_dict()) == list(same.state_dict())
    deep = N.make_net(7, 5, {**NET, "layers": 4, "dropout": 2.216e-05})
    keys = list(deep.state_dict())
    assert any(k.endswith("weight_ih_l3") for k in keys) and not any(k.endswith("weight_ih_l1") for k in one.state_dict())
    out = deep(torch.randn(3, 30, 7), torch.randn(3, 5), 5)
    assert out["mu"].shape == (3, 5 + N.PAD, 2) and torch.isfinite(out["mu"]).all()
    out["mu"].sum().backward()
    g = [p.grad for p in deep.parameters() if p.grad is not None]
    assert g and all(torch.isfinite(x).all() for x in g)
