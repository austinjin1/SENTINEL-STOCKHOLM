#!/usr/bin/env python
"""Independent reproduction + robustness probe of the csd-ews-channel win.

Confirms the claimed Pareto lift (11/15, median lead 41.3 d at matched <=5% held-out
control FPR, up from 36.1 d) reproduces, and — critically — that it is NOT knife-edge
on the a-priori EWS hyperparameters. A sibling candidate (do-soft-upweight) was
refuted because its 'win' hinged on a single boundary-value change of min_base; here
we vary EWS_MIN_BASE, EWS_WIN and BETA one-at-a-time and require the win to survive a
NEIGHBOURHOOD, not a point. Interleave split (the primary), which is what the headline
uses. Reuses the candidate module verbatim (monkeypatching only the swept global).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

# load the hyphenated candidate module by path
_spec = importlib.util.spec_from_file_location(
    "cand_csd_ews_channel", HERE / "cand_csd-ews-channel.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


def run(min_base=20, ews_win=180, beta=1.0):
    C.EWS_MIN_BASE = min_base
    C.EWS_WIN = ews_win
    C.BETA = beta
    r = C.run_split("interleave", C.M.cal_hold_split)
    return r


def show(tag, r):
    print(f"{tag:28s} det={r['episodes_detected']:>2}/{r['n_episodes']}  "
          f"lead={r['median_lead_days']:>7}d  fpr={r['control_heldout_fpr']:.4f}  "
          f"pareto={r['is_pareto_win']}")
    return r


def main():
    print("=== reproduce (min_base=20, win=180, beta=1.0) ===")
    base = show("base", run())

    print("\n=== EWS_MIN_BASE sweep (win=180, beta=1.0) ===")
    for mb in (15, 20, 30, 45, 60):
        show(f"min_base={mb}", run(min_base=mb))

    print("\n=== EWS_WIN sweep (min_base=20, beta=1.0) ===")
    for w in (120, 150, 180, 240):
        show(f"ews_win={w}", run(ews_win=w))

    print("\n=== BETA sweep (min_base=20, win=180) ===")
    for b in (0.5, 1.0, 1.5, 2.0):
        show(f"beta={b}", run(beta=b))

    print("\nHEADLINE-TO-BEAT: 11/15, 36.1 d, fpr<=0.05")
    print("WIN IS ROBUST iff detected>=11 AND lead>=36.1 across the neighbourhood, "
          "not just at the point.")


if __name__ == "__main__":
    main()
