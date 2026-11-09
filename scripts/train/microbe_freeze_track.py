#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Freeze the fold record of an additional SENTINEL-Microbe benchmark track and write its power check
BEFORE any model is run on it (RESEARCH_PLAN §15.5 amendment 2026-11-08, items 5–7; protocol §10.2).

Track ``lodo_x001`` (enlarged training): test rows = the primary LODO test rows (.002 development
events of domain g), training = all development events outside g from both releases (.001 2017-01 →
2018-10 added; the pre-2017 cross-release reserve and the ≥ 2024 holdout never appear).

Power check: because the test rows are the primary-LODO rows, the paired site-cluster bootstrap
standard error of an R² difference is estimated from existing primary-LODO comparisons (``--power-
model`` vs ``--power-ref``, OOF files in ``cv/lodo``) and turned into a minimum detectable difference
(80% power, two-sided, Holm worst case alpha = 0.05/8 and the 16-hypothesis test alpha = 0.05/16).

    python -B scripts/train/microbe_freeze_track.py --scheme lodo_x001
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths

REPO = Path(__file__).resolve().parents[2]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="lodo_x001")
    ap.add_argument("--power-model", default="ens_seqset_kmer+ridge_clr")
    ap.add_argument("--power-ref", default="rf_clr")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)
    from scipy.stats import norm

    from sentinel.models.microbe import evaluate as E
    from sentinel.models.microbe import splits as SP
    from sentinel.models.microbe.design import TARGET_NAMES, load_design

    design = load_design(with_release_v1=True)
    S = design.samples
    folds = SP.make_folds(S, a.scheme)
    prim = {f.name: set(f.test) for f in SP.make_folds(load_design().samples, "lodo")}
    for f in folds:     # guard: identical test rows to the primary LODO folds
        assert set(f.test) == prim[f.name.split("|")[0]], f.name
    src = S["source_product"].astype(str)
    rec = {"scheme": a.scheme, "holdout_rule": design.holdout_rule,
           "release_reserve": "DP1.20141.001 events before 2017-01-01 excluded (configs/microbe/common.yaml)",
           "n_train_rows_by_release": {str(k): int(v) for k, v in src.value_counts().items()},
           "folds": [f.to_dict() for f in folds]}
    blob = json.dumps(rec, sort_keys=True, indent=1).encode()
    out = REPO / "docs" / "preregistration" / f"microbe_split_{a.scheme}.json"
    out.write_bytes(blob)
    sha = hashlib.sha256(blob).hexdigest()
    (out.with_suffix(".sha256")).write_text(sha + "\n")

    # power check from primary-LODO paired comparisons (same test rows)
    d = paths.RUNS_ROOT / "microbe" / "cv" / "lodo"
    Sd = load_design().samples
    m, r = pd.read_parquet(d / f"{a.power_model}.parquet"), pd.read_parquet(d / f"{a.power_ref}.parquet")
    idx = m.index.intersection(r.index)
    z80 = norm.ppf(0.8)
    power = {}
    for t in TARGET_NAMES:
        y = Sd.loc[idx, t].to_numpy(float)
        o = E.paired_vs(y, m.loc[idx, t], r.loc[idx, t], Sd.loc[idx, "domain_id"].astype(str),
                        Sd.loc[idx, "site_id"].astype(str), n_boot=a.n_boot)
        se = (o["ci_high"] - o["ci_low"]) / (2 * norm.ppf(0.975))
        power[t] = {"n_test": int(np.isfinite(y).sum()), "paired_se_site": float(se),
                    "mdd_alpha_0.05/8": float((norm.ppf(1 - 0.05 / 16) + z80) * se),
                    "mdd_alpha_0.05/16": float((norm.ppf(1 - 0.05 / 32) + z80) * se)}
    rep = {"scheme": a.scheme, "split_sha256": sha, "n_folds": len(folds),
           "test_rows_identical_to_primary_lodo": True,
           "train_rows_added_per_target_001": {t: int(S.loc[src == "DP1.20141.001", t].notna().sum())
                                                for t in TARGET_NAMES},
           "power_basis": f"paired site-cluster bootstrap SE of {a.power_model} vs {a.power_ref} on primary LODO "
                          f"(B={a.n_boot}); MDD = (z_(1-alpha/2) + z_0.8) * SE",
           "power": power}
    (REPO / "docs" / "preregistration" / f"microbe_split_{a.scheme}_report.json").write_text(json.dumps(rep, indent=1))
    print(sha)
    print(json.dumps({t: round(v["mdd_alpha_0.05/8"], 3) for t, v in power.items()}))


if __name__ == "__main__":
    main()
