#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Apply the holdout rule (protocol §2.3), audit the design and freeze the split BEFORE model selection.

1. Realise the holdout cut: collect_date >= 2023-01-01 unless that holds out < 15% or > 30% of
   eligible events (≥ 1 non-missing target), in which case the 1 January closest to 20%.
2. Write configs/microbe/common.yaml's realised cut (only if it differs — printed, not silently),
   ``docs/preregistration/microbe_split.json`` (development ids, holdout ids, every development
   fold of every scheme) and its sha256 -> ``docs/preregistration/microbe_split.sha256``.
3. Sanity audit (no model): counts per site/domain/primer/year, target ranges and units, sentinel
   values (< 0, −9999-like), duplicate events, primer × year confound, token coverage at 256.
   -> ``docs/preregistration/microbe_split_report.json``.

    python -B scripts/train/microbe_freeze_split.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths
from sentinel.models.microbe import seqset as SS
from sentinel.models.microbe import splits as SP
from sentinel.models.microbe.design import TARGET_NAMES, TARGETS, holdout_mask, load_design

REPO = Path(__file__).resolve().parents[2]
PRE = REPO / "docs" / "preregistration"


def realise_cut(s: pd.DataFrame) -> tuple[str, float, dict]:
    """Fractions over ALL community events (the split is target-independent; amendment 2026-10-24)."""
    d = pd.to_datetime(s["collect_date"])
    frac = {y: float((d >= pd.Timestamp(f"{y}-01-01")).mean()) for y in range(2016, 2026)}
    cut = 2023
    if not 0.15 <= frac[cut] <= 0.30:
        cut = min(frac, key=lambda y: abs(frac[y] - 0.20))
    return f"{cut}-01-01", frac[cut], frac


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--audit-out", default=None, help="with --dry-run: write the audit JSON (target audit after joins)")
    a = ap.parse_args(argv)
    full = load_design(allow_holdout=True)  # the ONLY non-final use: realise the cut + audit (no model)
    S = full.samples
    cut, frac, fracs = realise_cut(S)
    rule = {"kind": "date_from", "date": cut}
    if cut != full.holdout_rule["date"]:
        print(f"NOTE: rule moves the cut from {full.holdout_rule['date']} to {cut} (holdout fraction {frac:.3f}); "
              f"update configs/microbe/common.yaml before freezing")
        if not a.dry_run:
            raise SystemExit(2)
    elig = S
    hm = holdout_mask(elig, rule)
    dev, hold = elig[~hm], elig[hm]
    rep = {"holdout_rule": rule, "holdout_fraction": frac, "holdout_fraction_by_year": fracs,
           "n_events_total": int(len(S)), "n_events_eligible": int(len(elig)), "n_dev": int(len(dev)),
           "n_holdout": int(len(hold)),
           "dev_sites": int(dev["site_id"].nunique()), "dev_domains": int(dev["domain_id"].nunique()),
           "holdout_sites": int(hold["site_id"].nunique()), "holdout_domains": int(hold["domain_id"].nunique()),
           "dev_per_domain": dev["domain_id"].value_counts().sort_index().to_dict(),
           "dev_per_site": dev["site_id"].value_counts().sort_index().to_dict(),
           "primer_by_year": pd.crosstab(pd.to_datetime(S["collect_date"]).dt.year, S["primer_set"]).to_dict(),
           "aquatic_type": dev["aquatic_type"].value_counts().to_dict(),
           "chem_match": dev["chem_match"].value_counts().to_dict() if "chem_match" in dev else {},
           "duplicate_event_keys": int(S["event_key"].duplicated().sum()) if "event_key" in S else None,
           "targets": {}}
    for t, (raw, how, floor) in TARGETS.items():
        v = elig[raw].astype(float)
        rep["targets"][t] = {"raw": raw, "n_dev": int(dev[t].notna().sum()), "n_holdout": int(hold[t].notna().sum()),
                             "raw_min": float(np.nanmin(v)) if v.notna().any() else None,
                             "raw_median": float(np.nanmedian(v)) if v.notna().any() else None,
                             "raw_max": float(np.nanmax(v)) if v.notna().any() else None,
                             "n_negative": int((v < 0).sum()), "n_sentinel_like": int((v.abs() >= 9999).sum()),
                             "n_at_or_below_floor": int((v <= floor).sum()) if floor else 0,
                             "n_censored": int(elig.get(raw.replace("raw_", "") + "_censored",
                                                        pd.Series(False, index=elig.index)).fillna(False).sum())}
    rep["token_coverage_256"] = SS.token_coverage(full.tokens[full.tokens["sample_id"].isin(dev.index)], 256)
    rep["sequence_length"] = full.sequences["length"].describe().to_dict()
    rec = SP.split_record(dev, hold, rule)
    v1p = paths.PROCESSED / "microbe_wq_v1" / "samples.parquet"
    if v1p.exists():   # cross-release events (.001, 2014-2018): development-only transfer set
        v1 = load_design(allow_holdout=True, root=v1p.parent, holdout_rule=rule, verify_frozen=False).samples
        if holdout_mask(v1, rule).any():
            raise SystemExit("v1 events fall in the holdout period — rule conflict")
        rec["release_v1_ids"] = sorted(map(str, v1.index))
        rep["n_release_v1_eligible"] = int(len(v1))
        rep["release_v1_per_domain"] = v1["domain_id"].value_counts().sort_index().to_dict()
    rep["folds"] = {k: len(v) for k, v in rec["schemes"].items()}
    rep["primer_fold_test_sizes"] = {k: [len(f["test"]) for f in v] for k, v in rec["schemes"].items()
                                     if k.startswith("primer")}
    print(json.dumps({k: v for k, v in rep.items() if k not in ("dev_per_site",)}, indent=1, default=str)[:6000])
    if a.dry_run:
        if a.audit_out:
            Path(a.audit_out).write_text(json.dumps(rep, indent=1, default=str))
        return
    h = SP.write_frozen_split(rec, PRE / "microbe_split.json")
    (PRE / "microbe_split.sha256").write_text(f"{h}  microbe_split.json\n")
    rep["split_sha256"] = h
    (PRE / "microbe_split_report.json").write_text(json.dumps(rep, indent=1, default=str))
    out = paths.ensure(paths.RUNS_ROOT / "microbe")
    (out / "split_sha256.txt").write_text(h + "\n")
    print("FROZEN split sha256:", h)


if __name__ == "__main__":
    main()
