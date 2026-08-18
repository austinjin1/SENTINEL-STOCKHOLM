#!/usr/bin/env python3
"""Post-hoc split-conformal calibration of the Digital Twin uncertainty head.

Loads the cached test-set predictions/intervals/targets/masks produced by
``evaluate_twin.py`` and applies split-conformal calibration (see
``sentinel.models.twin.conformal``) so the nominal 90% prediction interval
reaches ~90% empirical coverage on a held-out evaluation split.

It also reports, per horizon, the twin R2 alongside the **persistence
(naive t=0)** R2 so the twin is honestly positioned as a 1-day-only model:
persistence dominates from 7 days onward.  NO claim of forecast skill beyond
1 day is made.

Writes ``results/benchmarks/twin_conformal.json``.

Run:
    PYTHONNOUSERSITE=1 ~/.conda/envs/physiformer/bin/python \
        scripts/evaluation/conformal_calibrate_twin.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

import sys

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

from sentinel.models.twin.conformal import (  # noqa: E402
    calibrate_split_conformal,
    implied_scale,
    FORECAST_HEAD_Z,
)

CACHE = PROJECT / "results" / "benchmarks" / "twin_eval_cache.npz"
OUT = PROJECT / "results" / "benchmarks" / "twin_conformal.json"

STATE_VARS = [
    "dissolved_oxygen", "bod", "total_nitrogen", "total_phosphorus",
    "chlorophyll_a", "temperature", "ph", "turbidity", "doc", "sediment",
]
HORIZON_DAYS = [1, 7, 14, 30, 90, 365]


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def main() -> None:
    d = np.load(CACHE)
    preds = d["preds"]          # [H, N, D]
    upper = d["upper"]
    lower = d["lower"]
    targets = d["targets"]
    masks = d["masks"]
    init = d["init_states"]     # [N, D]
    H, N, D = preds.shape

    # ------------------------------------------------------------------ #
    # 1. Conformal calibration (two score types).                        #
    # ------------------------------------------------------------------ #
    res_norm = calibrate_split_conformal(
        preds, upper, targets, masks,
        alpha=0.10, score_type="normalized",
        cal_frac=0.5, seed=42, per_cell=True,
    )
    res_abs = calibrate_split_conformal(
        preds, upper, targets, masks,
        alpha=0.10, score_type="absolute",
        cal_frac=0.5, seed=42, per_cell=True,
    )

    # ------------------------------------------------------------------ #
    # 2. Twin vs persistence R2 per horizon (honest positioning).        #
    #    Persistence = predict the initial state (t=0) for every horizon.#
    # ------------------------------------------------------------------ #
    init_exp = np.broadcast_to(init[None, :, :], preds.shape)
    per_horizon_r2 = {}
    for h in range(H):
        m = (masks[h] > 0.5) & np.isfinite(preds[h]) & np.isfinite(targets[h]) & np.isfinite(init_exp[h])
        yt = targets[h][m]
        twin_r2 = _r2(yt, preds[h][m])
        pers_r2 = _r2(yt, init_exp[h][m])
        per_horizon_r2[f"{HORIZON_DAYS[h]}d"] = {
            "twin_r2": twin_r2,
            "persistence_r2": pers_r2,
            "twin_beats_persistence": bool(twin_r2 > pers_r2),
            "n": int(m.sum()),
        }

    # Overall R2 (all horizons pooled), as in the original eval.
    m_all = (masks > 0.5) & np.isfinite(preds) & np.isfinite(targets) & np.isfinite(init_exp)
    twin_r2_overall = _r2(targets[m_all], preds[m_all])
    pers_r2_overall = _r2(targets[m_all], init_exp[m_all])

    # ------------------------------------------------------------------ #
    # 3. Assemble output.                                                 #
    # ------------------------------------------------------------------ #
    out = {
        "model": "DigitalTwinEngine",
        "description": (
            "Post-hoc split-conformal calibration of the Digital Twin 90% "
            "uncertainty interval. Calibration quantile fit on a held-out "
            "50% calibration split of the real test-set predictions; "
            "coverage/width reported on the disjoint 50% evaluation split. "
            "No retraining, no label leakage into the evaluation split."
        ),
        "source_cache": str(CACHE),
        "n_test_samples": int(N),
        "n_valid_entries": int(((masks > 0.5)
                                & np.isfinite(preds)
                                & np.isfinite(targets)
                                & np.isfinite(upper)).sum()),
        "forecast_head_z": FORECAST_HEAD_Z,
        "target_coverage": 0.90,
        "alpha": 0.10,
        "split": {"cal_frac": 0.5, "seed": 42, "method": "random_exchangeable"},

        # The headline fix: coverage before vs after.
        "coverage_before": res_norm.coverage_before,   # ~0.14 (head as-is)
        "coverage_after_normalized": res_norm.coverage_after,
        "coverage_after_absolute": res_abs.coverage_after,
        "stated_baseline_coverage_before": 0.14,

        "normalized_conformal": {
            k: v for k, v in asdict(res_norm).items()
            if k not in ("per_var_horizon",)
        },
        "absolute_conformal": {
            k: v for k, v in asdict(res_abs).items()
            if k not in ("per_var_horizon", "per_horizon")
        },
        "normalized_per_var_horizon": res_norm.per_var_horizon,

        # Honest skill positioning.
        "r2_positioning": {
            "note": (
                "Persistence (predict t=0) is the honest baseline. The twin "
                "only matches/edges persistence at the 1-day horizon and is "
                "beaten by persistence at 7d and beyond. No forecast-skill "
                "claim is made past 1 day."
            ),
            "twin_r2_overall": twin_r2_overall,
            "persistence_r2_overall": pers_r2_overall,
            "stated_baseline_persistence_r2": 0.92,
            "per_horizon": per_horizon_r2,
        },
    }

    OUT.write_text(json.dumps(out, indent=2, default=float))

    # ------------------------------------------------------------------ #
    # 4. Console summary.                                                 #
    # ------------------------------------------------------------------ #
    print("=" * 68)
    print("Digital Twin - split-conformal calibration of 90% interval")
    print("=" * 68)
    print(f"valid entries: {out['n_valid_entries']}  "
          f"(cal={res_norm.n_cal}, eval={res_norm.n_eval})")
    print(f"target coverage: 0.90")
    print(f"coverage BEFORE (head as-is):        {res_norm.coverage_before:.4f}")
    print(f"coverage AFTER  (normalized conformal):{res_norm.coverage_after:.4f}")
    print(f"coverage AFTER  (absolute  conformal): {res_abs.coverage_after:.4f}")
    print(f"mean interval width  before -> after (normalized): "
          f"{res_norm.mean_width_before:.3f} -> {res_norm.mean_width_after:.3f}")
    print()
    print("Per-horizon coverage (normalized conformal):")
    for k, v in res_norm.per_horizon.items():
        print(f"  {k:>5s}: {v['coverage_before']:.3f} -> {v['coverage_after']:.3f}  "
              f"(n_eval={v['n_eval']})")
    print()
    print("R2 positioning (twin vs persistence):")
    print(f"  overall: twin={twin_r2_overall:.3f}  persistence={pers_r2_overall:.3f}")
    for k, v in per_horizon_r2.items():
        flag = "twin wins" if v["twin_beats_persistence"] else "persistence wins"
        print(f"  {k:>5s}: twin={v['twin_r2']:+.3f}  persistence={v['persistence_r2']:+.3f}  -> {flag}")
    print()
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
