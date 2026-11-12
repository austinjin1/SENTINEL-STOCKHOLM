# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Decision rules of the M6 architecture-v2 round as pure functions (amendment §7, §8, §9; lead rulings §2.1, §2.2,
L11). Nothing here reads a file or a label; the drivers pass numbers in and write the outcome to the registry.

* :func:`select_point` — lowest J among the grid points at seed 0; ties within 1e-4 go to the earlier grid point.
* :func:`g_d1a` — ONE gate for all tracks, on the 3-seed mean of J_sonde: grab rows stay only if the selected D1
  lowers J_sonde by ≥ 3 % against its no-grab ablation (a). J_grab is reported and decides nothing.
* :func:`comparator_set` / :func:`g_d1b` — the claim model must show a point Δ ≥ 0 on ≥ 3 of the 4 primary T-B
  endpoints (N2: on both endpoints) against **every** comparator: R-TD and R-TD-W in both label forms, G1, G2, P,
  and the references-only stacks when D3 is the claim model. A no-go is final.
* :func:`min_groups` — an endpoint is a claim endpoint only with G ≥ 15 basin groups and G_eff ≥ 8.
"""

from __future__ import annotations

import numpy as np

TIE_TOL = 1e-4
G_D1A_MIN_REL = 0.03
PRIMARY_TB = ("temp_c_mean", "temp_c_7dadm", "do_mgl_mean", "do_mgl_min")
N2_ENDPOINTS = ("T", "DO")
MIN_G, MIN_G_EFF = 15, 8.0
# comparator → label forms it is run in (lead rulings §2.2 as amended by L11.4); P and the GBMs enter in the gate form
COMPARATORS = (("rtd", "grab"), ("rtd", "published"), ("rtd_w", "grab"), ("rtd_w", "published"), ("g1", "gate"),
               ("g2", "gate"), ("p", "one"))
REF_STACKS = (("ref_stack", "gate"), ("ref_stack_w", "gate"))


def select_point(j_by_point: dict, tol: float = TIE_TOL) -> str:
    """``j_by_point``: grid point → seed-0 J, in grid order (insertion order = "earlier")."""
    pts = [(k, float(v)) for k, v in j_by_point.items() if v is not None and np.isfinite(v)]
    if not pts:
        raise ValueError("no grid point has a finite J")
    lo = min(v for _, v in pts)
    return next(k for k, v in pts if v <= lo + tol)


def g_d1a(js_selected, js_nograb, min_rel: float = G_D1A_MIN_REL) -> dict:
    """``js_selected`` / ``js_nograb``: J_sonde of seeds 0–2 of the selected D1 point and of ablation (a)."""
    a, b = np.asarray(js_selected, "float64"), np.asarray(js_nograb, "float64")
    if len(a) != 3 or len(b) != 3 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("gate G-D1a needs J_sonde of 3 seeds of the selected point and of ablation (a)")
    ma, mb = float(a.mean()), float(b.mean())
    rel = (mb - ma) / mb
    ok = bool(rel >= min_rel)
    return {"gate": "G-D1a", "j_sonde_selected_mean": ma, "j_sonde_nograb_mean": mb, "relative_reduction": float(rel),
            "threshold": min_rel, "passed": ok, "label_form": "grab" if ok else "nograb",
            "rule": "3-seed mean of J_sonde; one outcome for all tracks; J_grab reported only (rulings Q2)"}


def comparator_set(claim_arm: str) -> list:
    """Every comparator the claim model must clear, as (model, label form)."""
    out = list(COMPARATORS)
    if claim_arm == "d3":
        out += list(REF_STACKS)
    return out


def g_d1b(delta: dict, family: str, claim_arm: str) -> dict:
    """``delta``: "model/form" → endpoint → point Δ (claim − comparator; KGE scale, higher is better). ``family``:
    "T-B" (≥ 3 of the 4 primary endpoints) or "N2" (both endpoints). Missing comparator or endpoint → no-go."""
    eps = PRIMARY_TB if family == "T-B" else N2_ENDPOINTS
    need = 3 if family == "T-B" else len(eps)
    rows, go = {}, True
    for model, form in comparator_set(claim_arm):
        key = f"{model}/{form}"
        d = delta.get(key)
        vals = [None if d is None or d.get(e) is None else float(d[e]) for e in eps]
        n_ok = sum(v is not None and np.isfinite(v) and v >= 0.0 for v in vals)
        complete = all(v is not None and np.isfinite(v) for v in vals)
        rows[key] = {"n_nonnegative": int(n_ok), "complete": complete, "go": bool(complete and n_ok >= need)}
        go &= rows[key]["go"]
    return {"gate": "G-D1b", "family": family, "claim_arm": claim_arm, "needed": need, "by_comparator": rows,
            "go": bool(go), "rule": "point Δ ≥ 0 against every comparator, R-TD and R-TD-W in both label forms; a "
                                    "no-go is final (rulings §2.2, L11.4)"}


def min_groups(n_by_group) -> dict:
    """G and G_eff = (Σ n_g)² / Σ n_g² of a paired cell from its sites-per-basin-group counts."""
    n = np.asarray([x for x in n_by_group if x > 0], "float64")
    G = int(len(n))
    ge = float(n.sum() ** 2 / (n ** 2).sum()) if G else 0.0
    return {"G": G, "G_eff": round(ge, 2), "n_sites": int(n.sum()), "claim_endpoint": bool(G >= MIN_G and ge >= MIN_G_EFF)}
