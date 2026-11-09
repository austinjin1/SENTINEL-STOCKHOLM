#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Orbit power analysis for the headline test cell (pre-training review W4; orbit_protocol.md §1).

Uses ONLY (i) validation per-row predictions written by ``orbit_compare.py`` (``val_row_preds.parquet``) and
(ii) the COUNTS of the protected test cell from the split report (``protected_cell_counts``: rows, water bodies
and role blocks per target). No test label, statistic or prediction is read.

Method (plug-in, "the test cell behaves like the val cell"): for each H1 comparator c the paired MdSA
difference d_c = MdSA(SENTINEL) - MdSA(c) is re-estimated on S pseudo-test samples, each drawing B_test role
blocks with replacement from the val blocks (B_test = number of test-cell role blocks for the target). The SD
of these draws is the expected test standard error SE_c. Reported per comparator: val difference, SE at the
val and test block counts, the minimum detectable effect at 80 % power (2.80 SE, two-sided alpha = 0.05) and
the power to exclude 0 if the true difference equals the val point estimate, Phi(|d|/SE - 1.96) with the
sign required by H1. The joint (intersection-union) power is the fraction of pseudo-test draws in which
every comparator satisfies d_c + 1.96 SE_c < 0.

    python -B scripts/train/orbit_power.py --compare compare_satwq_noaug+gbm --version dev_v4 --register
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

from sentinel.core import paths  # noqa: E402
from sentinel.models.orbit import evaluate as E  # noqa: E402
from sentinel.models.orbit import pipeline as PL  # noqa: E402

TEST_CELL = "testxtest_lake"


def block_draws(df: pd.DataFrame, comps: list[str], n_blocks: int, n_sim: int, seed: int) -> np.ndarray:
    """[n_sim, n_comp] paired MdSA differences on pseudo samples of ``n_blocks`` val blocks."""
    g = {b: ix for b, ix in df.groupby("block").indices.items()}
    keys = np.array(list(g))
    y = df.y.to_numpy()
    es = df["SENTINEL"].to_numpy() - y
    ec = {c: df[c].to_numpy() - y for c in comps}
    rs = np.random.default_rng(seed)
    out = np.empty((n_sim, len(comps)))
    for s in range(n_sim):
        ix = np.concatenate([g[k] for k in keys[rs.integers(0, len(keys), n_blocks)]])
        ms = E.mdsa_log10(es[ix])
        out[s] = [ms - E.mdsa_log10(ec[c][ix]) for c in comps]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", required=True, help="compare_* run dir holding val_row_preds.parquet")
    ap.add_argument("--version", required=True, help="data version whose split report holds the test-cell counts")
    ap.add_argument("--targets", nargs="*", default=["chla", "secchi", "tss", "turbidity"])
    ap.add_argument("--n-sim", type=int, default=2000)
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--track", default="headline", choices=["headline", "U"],
                    help="U: test counts from orbit_tracks_counts_<version>.json (u_test; orbit_protocol.md §11.2)")
    a = ap.parse_args()
    cdir = paths.RUNS_ROOT / "orbit" / a.compare
    rp = json.loads((paths.REPO_ROOT / "docs" / "preregistration" / f"orbit_split_report_{a.version}.json").read_text())
    cell_name = TEST_CELL
    counts = rp["protected_cell_counts"][TEST_CELL]
    if a.track == "U":
        cell_name = "trackU_test"
        counts = json.loads((paths.REPO_ROOT / "docs" / "preregistration" /
                             f"orbit_tracks_counts_{a.version}.json").read_text())["u_test"]
    vp = pd.read_parquet(cdir / "val_row_preds.parquet")
    import importlib.util as _iu

    spec = _iu.spec_from_file_location("oc", paths.REPO_ROOT / "scripts" / "train" / "orbit_compare.py")
    oc = _iu.module_from_spec(spec)
    spec.loader.exec_module(oc)
    out = []
    for t in a.targets:
        cell = oc.PRIMARY_TARGET_CELL[t]
        d = vp[(vp.cell == cell) & (vp.target == t)].dropna(axis=1, how="all")
        comps = sorted(c for c in d.columns if c.endswith("[val-sel]"))
        if d.empty or not comps:
            continue
        bt = int(counts[t]["n_blocks"])
        bv = int(d.block.nunique())
        if bt < 2:
            out.append({"target": t, "val_cell": cell, "n_blocks_val": bv, "n_blocks_test": bt, "note": "test cell too small"})
            continue
        dv = block_draws(d, comps, bv, a.n_sim, 0)
        dt = block_draws(d, comps, bt, a.n_sim, 1)
        se_t = dt.std(0, ddof=1)
        joint = float(np.mean(np.all(dt + 1.96 * se_t < 0, axis=1)))
        for j, c in enumerate(comps):
            ys = d.y.to_numpy()
            diff = E.mdsa_log10(d["SENTINEL"].to_numpy() - ys) - E.mdsa_log10(d[c].to_numpy() - ys)
            se_v = float(dv[:, j].std(ddof=1))
            pw = float(norm.cdf(-diff / se_t[j] - 1.96))       # P(upper CI bound < 0) if true diff = val diff
            out.append({"target": t, "val_cell": cell, "comparator": c, "val_diff": diff, "n_val_rows": int(len(d)),
                        "n_blocks_val": bv, "se_val": se_v, "n_test_rows": int(counts[t]["n"]),
                        "n_waterbodies_test": int(counts[t]["n_waterbodies"]), "n_blocks_test": bt,
                        "se_test": float(se_t[j]), "mde80_test": float(2.80 * se_t[j]), "power_at_val_diff": pw,
                        "joint_power_all_comparators": joint})
    res = pd.DataFrame(out)
    res.to_csv(cdir / f"power_{cell_name}_{a.version}.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))
    if a.register:
        from sentinel.core.registry import log_result

        for r in res.dropna(subset=["comparator"]).itertuples():
            for met in ("mde80_test", "power_at_val_diff", "se_test"):
                log_result(workstream=PL.WORKSTREAM, experiment=f"orbit_power_{a.version}" + ("" if a.track == "headline" else "_trackU"), model=a.compare,
                           metric=f"{met}_vs_{r.comparator}", value=float(getattr(r, met)), target=r.target,
                           split=f"{cell_name}@{a.version} (counts only) / {r.val_cell} (val predictions)",
                           n=int(r.n_test_rows), is_baseline=False, data_manifests=[f"orbit_{a.version}"],
                           artifacts=[str(cdir)],
                           notes="design calculation: val per-row predictions + test-cell COUNTS only (no test labels); "
                                 "plug-in block resampling, two-sided alpha 0.05",
                           extra={"val_diff": r.val_diff, "n_blocks_val": r.n_blocks_val, "n_blocks_test": r.n_blocks_test,
                                  "joint_power_all_comparators": r.joint_power_all_comparators})
        print(f"registered under orbit_power_{a.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
