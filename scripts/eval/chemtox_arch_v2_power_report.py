#!/usr/bin/env python
"""Emit the M4 arch-v2 counts and MDE reports for T1/T6. Counts and keys only (lead ruling L6).

Reads **only** ``chemtox_t1t6_split_v2.parquet``, which carries no target column, and the power
helper, which takes unit-size vectors. No label of any sealed block can reach this script, so the
pre-registered power table exists before any ``FREEZE`` file by construction and not by promise.

    python -B scripts/eval/chemtox_arch_v2_power_report.py --out $SENTINEL_RUNS/chemtox/arch_v2

Per block it tabulates, for icc in {0, 0.5, 1} and rho in {0.9, 0}:

* ``all``                   the full block (the primary reading of a claim)
* ``in_adore_domain``       the organic subset T2's figures were computed on (MDE-DOMAIN-BYPASS);
                            the STRICTER of ``all`` and this one governs condition (f)
* ``no_shared_reference``   rows whose publication is absent from the pool (SHARED-REFERENCE-GATE)
* one row per duration band, with ``claimable`` = units >= MIN_UNITS_FOR_CLAIM
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sentinel.models.chemtox import arch_v2 as A
from sentinel.models.chemtox import arch_v2_power as P

TEST_BLOCKS = ("T1_test", "T6a_test", "T6b_test")


def _sizes(frame: pd.DataFrame) -> list[int]:
    return frame.groupby("holdout_unit").size().tolist()


def _entry(frame: pd.DataFrame, block: str, stratum: str, *, n_boot: int, n_sim: int) -> dict:
    sizes = _sizes(frame)
    out = {"block": block, "stratum": stratum, "rows": int(len(frame)), "units": len(sizes),
           "chemicals": int(frame["test_cas"].nunique()),
           "clusters": int(frame["t1_cluster"].nunique()),
           "max_unit_rows": max(sizes) if sizes else 0,
           "max_unit_row_share": round(max(sizes) / len(frame), 4) if sizes else None,
           "claimable_units": bool(len(sizes) >= A.MIN_UNITS_FOR_CLAIM)}
    if len(sizes) < 5:
        out["skipped"] = "fewer than 5 resampling units"
        return out
    for icc in P.ICC_GRID:
        for rho in (P.DEFAULT_RHO, 0.0):
            tag = f"icc{icc}_rho{rho}"
            out[f"mde_{tag}"] = round(P.mde_closed_form(sizes, icc=icc, rho=rho), 4)
        out[f"effective_n_icc{icc}"] = round(P.effective_sample_size(sizes, icc), 1)
    chk = P.mde_bootstrap(sizes, icc=0.5, n_boot=n_boot, seed=0, n_sim=n_sim)
    out["bootstrap_check_icc0.5_rho0.9"] = round(chk["mde_bootstrap"], 4)
    out["bootstrap_check_closed_form"] = round(chk["mde_closed_form"], 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=str(A.SPLIT_PATH))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-sim", type=int, default=8)
    a = ap.parse_args()

    split = pd.read_parquet(a.split)
    banned = [c for c in split.columns if "conc" in c or "log10" in c or c == "target"]
    if banned:
        raise SystemExit(f"the split file carries label-like columns {banned}; refusing (L6)")
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for b in TEST_BLOCKS:
        g = split[split["block"] == b]
        rows.append(_entry(g, b, "all", n_boot=a.n_boot, n_sim=a.n_sim))
        rows.append(_entry(g[g["in_adore_domain"]], b, "in_adore_domain",
                           n_boot=a.n_boot, n_sim=a.n_sim))
        rows.append(_entry(g[~g["shared_reference"]], b, "no_shared_reference",
                           n_boot=a.n_boot, n_sim=a.n_sim))
        for band in A.BAND_NAMES:
            sub = g[g["duration_band"] == band]
            if len(sub):
                rows.append(_entry(sub, b, f"band:{band}", n_boot=a.n_boot, n_sim=a.n_sim))

    # the governing gate per block: the stricter of 'all' and 'in_adore_domain' (MDE-DOMAIN-BYPASS)
    gate = {}
    for b in TEST_BLOCKS:
        pair = [r for r in rows if r["block"] == b and r["stratum"] in ("all", "in_adore_domain")]
        for icc in P.ICC_GRID:
            k = f"mde_icc{icc}_rho{P.DEFAULT_RHO}"
            vals = {r["stratum"]: r.get(k) for r in pair if r.get(k) is not None}
            if vals:
                worst = max(vals, key=lambda s: vals[s])
                gate.setdefault(b, {})[f"icc{icc}"] = {
                    "governing_mde": vals[worst], "from_stratum": worst, "all": vals.get("all"),
                    "in_adore_domain": vals.get("in_adore_domain")}

    report = {
        "generated_by": "scripts/eval/chemtox_arch_v2_power_report.py",
        "split_file": str(a.split), "split_sha256": A.sha256_file(Path(a.split)),
        "helper": "sentinel.models.chemtox.arch_v2_power (NOT unseen_taxa.mde; see L5)",
        "rmse_level_assumed": P.DEFAULT_RMSE_LEVEL,
        "primary_icc": 0.5, "icc_grid": list(P.ICC_GRID),
        "min_units_for_claim": A.MIN_UNITS_FOR_CLAIM,
        "label_derived_statistics": "none: the split file carries no target column and the power "
                                    "helper takes unit-size vectors only",
        "governing_gate_per_block": gate,
        "strata": rows,
    }
    p = out_dir / "t1t6_mde_v2.json"
    p.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"wrote {p}  sha256 {A.sha256_file(p)}")
    for b in TEST_BLOCKS:
        for r in [x for x in rows if x["block"] == b]:
            print(f"  {b:10s} {r['stratum']:22s} rows={r['rows']:6d} units={r['units']:5d} "
                  f"maxU={str(r['max_unit_row_share']):>6s} "
                  f"icc0={r.get('mde_icc0.0_rho0.9','-')} icc.5={r.get('mde_icc0.5_rho0.9','-')} "
                  f"icc1={r.get('mde_icc1.0_rho0.9','-')} claimable={r['claimable_units']}")


if __name__ == "__main__":
    main()
