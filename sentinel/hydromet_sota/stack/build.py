# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Fit one stack (D3, or a references-only stack) on inner rows: weights per endpoint and the cross-fitted inner
predictions that the shared criterion scores (amendment §6.3; ``configs/hydromet_sota/v2/stack.yaml``).

Everything is in model space (:class:`sentinel.hydromet_sota.gso_mt.data.Scales`); the weights are invariant to the
affine label map because they sum to one. Train-role inner groups × train block only."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..gso_mt import criterion as C
from ..gso_mt import runner as VR
from ..gso_mt.data import DAILY, INST, PRIMARY
from .convex import cross_fit, grab_row_weights, simplex_lstsq

INST_FROM_DAILY = {"T": "temp_c_mean", "DO": "do_mgl_mean"}
NN_ARMS = ("d1", "d2", "rtd", "rtd_w")
NOGRAB_POINT = {"d1": "abl_nograb", "d2": "nograb", "rtd": "published", "rtd_w": "published"}


def member_run_dirs(root, member: str, form: str, selected: dict, seeds=(0, 1, 2)) -> list:
    """Run directories that hold the early-stopped inner predictions of one member in one label form.
    ``form``: "grab" or "nograb" (gate G-D1a). ``selected``: arm → selected grid point (d1, d2)."""
    root = Path(root)
    if member in NN_ARMS:
        if form == "nograb":
            point = NOGRAB_POINT[member]
        else:
            point = selected.get(member) if member in ("d1", "d2") else "fixed"
        if point is None:
            return []
        return [root / member / f"{member}_{point}_s{s}" for s in seeds]
    if member in ("g1", "g2"):
        return [root / member / f"{member}_{'nograb' if form == 'nograb' else 'sel'}_s{s}" for s in seeds]
    if member == "p":
        return [root / "p" / "p_fixed"]                       # deterministic, one label form
    raise ValueError(f"unknown stack member {member!r}")


def load_member(dirs, *, n_inner: int, n_grab_inner: int, date_start: str, release: dict,
                what: str = "stack member") -> dict | None:
    """Seed mean of the members' ``inner_pred.npz`` (raw units). ``None`` if no run directory is complete; an
    incomplete seed set raises (a member is never averaged over fewer seeds than it has). ``release``: the release
    record of the combining process — a run made under another pin state or another code-pin sha256 is refused
    (:func:`sentinel.hydromet_sota.gso_mt.runner.assert_combinable`; entry-1 lead rulings, decision 4)."""
    done = [d for d in dirs if (Path(d) / ".done").exists() and (Path(d) / "inner_pred.npz").is_file()]
    if not done:
        return None
    if len(done) != len(dirs):
        raise RuntimeError(f"{len(done)} of {len(dirs)} seeds are complete under {Path(dirs[0]).parent}")
    combined = VR.assert_combinable(done, release, what)
    daily, inst = [], []
    for d in done:
        z = np.load(Path(d) / "inner_pred.npz")
        if int(z["n_inner"]) != n_inner or str(z["date_start"]) != date_start:
            raise RuntimeError(f"{d}: inner predictions are not on the pinned pool's inner rows")
        daily.append(z["daily"].astype("float64"))
        if "inst" in z.files:
            if int(z["n_grab_inner"]) != n_grab_inner:
                raise RuntimeError(f"{d}: hour-resolved inner predictions are not on the inner grab set")
            inst.append(z["inst"].astype("float64"))
    if inst and len(inst) != len(daily):
        raise RuntimeError(f"{Path(dirs[0]).parent}: some seeds lack hour-resolved inner predictions")
    return {"daily": np.mean(daily, axis=0), "inst": np.mean(inst, axis=0) if inst else None, "n_seeds": len(done),
            "combined": combined}


def fit_stack(members: dict, y_daily: np.ndarray, y_inst, g_inner, g_grab, crit_factor: dict, *, form: str = "grab",
              cap: int = C.GRAB_CAP, ridge: float = 0.0, min_rows: int = 200) -> dict:
    """``members``: name → {"daily": (n, T, ≥7), "inst": (m, T, 2) or None}, model space, in member order.
    Returns weights per endpoint, the cross-fitted predictions (``daily`` (n, T, 7), ``inst`` (m, T, 2) or None),
    J of the cross-fitted stack and row counts."""
    names = list(members)
    n, T = y_daily.shape[:2]
    out_d = np.full((n, T, 7), np.nan)
    weights, folds, rows = {}, {}, {}
    for ep, key in PRIMARY.items():
        j = DAILY.index(key)
        use = [m for m in names if np.isfinite(members[m]["daily"][:, :, j]).any()]
        lab = np.isfinite(y_daily[:, :, j])
        ok = lab & np.all([np.isfinite(members[m]["daily"][:, :, j]) for m in use], axis=0) if use else lab & False
        rows[ep] = {"labelled": int(lab.sum()), "used": int(ok.sum()), "members": use}
        if ok.sum() < min_rows or not use:
            weights[ep] = None
            continue
        P = np.stack([members[m]["daily"][:, :, j][ok] for m in use], axis=-1)
        g = np.broadcast_to(np.asarray(g_inner)[:, None], (n, T))[ok]
        w = simplex_lstsq(P, y_daily[:, :, j][ok], None, ridge)
        cv, fw = cross_fit(P, y_daily[:, :, j][ok], g, None, ridge)
        out_d[:, :, j][ok] = cv
        weights[ep] = {m: float(x) for m, x in zip(use, w)}
        folds[ep] = {"n_folds": len(fw), "max_abs_weight_change": float(max(np.abs(v - w).max() for v in fw.values()))}
    out_i = None
    if form == "grab" and y_inst is not None and len(y_inst):
        out_i = np.full(y_inst.shape, np.nan)
        for jv, v in enumerate(INST):
            use = [m for m in names if members[m].get("inst") is not None
                   and np.isfinite(members[m]["inst"][:, :, jv]).any()]
            lab = np.isfinite(y_inst[:, :, jv])
            ok = lab & np.all([np.isfinite(members[m]["inst"][:, :, jv]) for m in use], axis=0) if use else lab & False
            rows[v] = {"labelled": int(lab.sum()), "used": int(ok.sum()), "members": use}
            if ok.sum() < min_rows or not use:
                weights[v] = None
                continue
            P = np.stack([members[m]["inst"][:, :, jv][ok] for m in use], axis=-1)
            r = grab_row_weights(ok, cap)[ok]
            g = np.broadcast_to(np.asarray(g_grab)[:, None], ok.shape)[ok]
            w = simplex_lstsq(P, y_inst[:, :, jv][ok], r, ridge)
            cv, fw = cross_fit(P, y_inst[:, :, jv][ok], g, r, ridge)
            out_i[:, :, jv][ok] = cv
            weights[v] = {m: float(x) for m, x in zip(use, w)}
            folds[v] = {"n_folds": len(fw), "max_abs_weight_change": float(max(np.abs(x - w).max() for x in fw.values()))}
    else:                                # no-grab form: no grab row enters; the daily-mean weights are reused
        for v, ep in INST_FROM_DAILY.items():
            weights[v] = None if weights.get(ep) is None else dict(weights[ep])
            rows[v] = {"labelled": 0, "used": 0, "members": list(weights[v] or {}), "from": ep}
    js = C.j_sonde(out_d, y_daily, crit_factor)
    jg = C.j_grab(out_i, y_inst) if out_i is not None else None
    return {"weights": weights, "folds": folds, "rows": rows, "daily": out_d, "inst": out_i, "j_sonde": js["j_sonde"],
            "by_target": js["by_target"], "j_grab": None if jg is None else jg["j_grab"], "J": C.combine(js, jg),
            "n_rows_dropped": int(sum(r_["labelled"] - r_["used"] for k, r_ in rows.items() if "from" not in r_)),
            "form": form}


def apply_weights(weights: dict, preds: dict) -> np.ndarray:
    """Σ_m w_m · pred_m for one endpoint (``weights``: member → w; ``preds``: member → array)."""
    return sum(float(w) * np.asarray(preds[m], "float64") for m, w in weights.items())
