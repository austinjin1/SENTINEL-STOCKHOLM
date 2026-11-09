#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Diebold–Mariano tests (RESEARCH_PLAN §14.4) on saved per-sample scores from sonde_score.py.

Panel version: d_t = mean over sites of CRPS_model − CRPS_ref on issue date t (common rows only),
HAC lag = max(h − 1, 7) (serial correlation of daily overlapping forecasts), HLN correction.
Also per-site DM tests (lag max(h − 1, 7)) summarised as counts of significantly better / worse
sites at 5%. Descriptive for val (selection-biased); the test-block run is the confirmatory one.

    python -B scripts/eval/sonde_dm.py --samples <score_run>/samples_val.parquet --ref damped_anomaly
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.models.sonde.evaluate import diebold_mariano


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--ref", default="damped_anomaly")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    s = pd.read_parquet(a.samples, columns=["model", "var", "h", "site_no", "issue_date", "crps"])
    ref = s[s["model"] == a.ref].set_index(["var", "h", "site_no", "issue_date"])["crps"]
    rows = []
    for m in sorted(set(s["model"]) - {a.ref, "climatology", "persistence"}):
        mm = s[s["model"] == m].set_index(["var", "h", "site_no", "issue_date"])["crps"]
        d = (mm - ref.reindex(mm.index)).rename("d").reset_index()
        if d["d"].isna().any():
            raise ValueError(f"{m}: rows missing in reference (common-row violation)")
        for (v, h), g in d.groupby(["var", "h"]):
            lag = max(int(h) - 1, 7)
            panel = g.groupby("issue_date")["d"].mean().sort_index()
            r = diebold_mariano(panel.to_numpy(), lag)
            better = worse = n_sites = 0
            for _, gs in g.groupby("site_no"):
                if len(gs) < 60:
                    continue
                rs = diebold_mariano(gs.sort_values("issue_date")["d"].to_numpy(), lag)
                n_sites += 1
                if np.isfinite(rs["p_value"]) and rs["p_value"] < 0.05:
                    better += rs["dm"] < 0
                    worse += rs["dm"] > 0
            rows.append({"model": m, "ref": a.ref, "var": v, "h": int(h), "hac_lag": lag, "n_days": r["n"],
                         "mean_crps_diff": r.get("mean_d"), "dm_stat": r["dm"], "p_value": r["p_value"],
                         "n_sites_tested": n_sites, "n_sites_sig_better": int(better), "n_sites_sig_worse": int(worse)})
    out = pd.DataFrame(rows)
    path = Path(a.out) if a.out else Path(a.samples).with_name(f"dm_vs_{a.ref}.csv")
    out.to_csv(path, index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 500, "display.float_format", "{:.4g}".format):
        print(out[out["h"].isin([1, 7])].to_string(index=False))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
