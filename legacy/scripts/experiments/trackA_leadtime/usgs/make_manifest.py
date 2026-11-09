#!/usr/bin/env python
"""Build data/processed/trackA_usgs/usgs_manifest.json from the built .npz set.

Recomputes per-site facts directly from the written .npz (source of truth),
verifies each has adequate covered pre-onset windows, and reports counts.
enough = (>=6 episode sites AND >=6 control sites) each with adequate
pre-onset (episode) / broad clean (control) coverage.

Adequate pre-onset window (episode): >=1 detector window (128 bins = 32 h) of
real DO coverage exists strictly before onset within a >=7 d clean runway. We
require the onset to have at least WIN*STRIDE-worth of covered pre-onset bins so
the detector has scored windows before onset.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/bcheng/SENTINEL")
OUTDIR = ROOT / "data" / "processed" / "trackA_usgs"
BIN = 900
DAY = 86400
WIN = 128            # detector window bins (matches _detector.WIN)
DO_THR = 4.0
BASELINE_DAYS = 7
REAL_PARAMS = ["DO", "pH", "turbidity", "conductivity"]

# Adequate pre-onset coverage: need at least this many real DO bins before onset
MIN_PREONSET_DO_BINS = WIN * 4     # >= 512 real DO bins (>=4 scored windows worth)
MIN_CONTROL_CLEAN_BINS = 2000      # broad clean record for controls


def site_facts(npz: Path) -> dict:
    d = np.load(npz, allow_pickle=True)
    ts = d["ts"]; do = d["DO"]; mdo = d["mask_DO"]
    role = str(d["role"]); onset_i = int(d["onset_index"])
    present = {p: bool(d[f"mask_{p}"].sum() > 0) for p in REAL_PARAMS}
    n_days = round((ts[-1] - ts[0]) / DAY, 1)
    clean = mdo & np.isfinite(do)
    rec = {"site": npz.stem, "role": role, "onset_index": onset_i,
           "n_bins": int(ts.size), "n_days": n_days,
           "params_present": present,
           "do_clean_min": round(float(np.nanmin(do[clean])), 3) if clean.any() else None,
           "do_cov": round(float(mdo.mean()), 3)}
    if onset_i >= 0:
        t0 = int(ts[onset_i])
        rec["onset_ts"] = t0
        rec["onset_utc"] = pd.to_datetime(t0, unit="s", utc=True).isoformat()
        pre = (ts < t0)
        preonset_do = int((mdo & pre).sum())
        rec["n_preonset_do_bins"] = preonset_do
        # runway: clean days before onset with no earlier <4
        base = clean & pre
        if base.any():
            first_t = ts[np.where(base)[0][0]]
            rec["runway_days"] = round((t0 - first_t) / DAY, 1)
        else:
            rec["runway_days"] = 0.0
        rec["adequate"] = bool(preonset_do >= MIN_PREONSET_DO_BINS
                               and rec["runway_days"] >= BASELINE_DAYS)
    else:
        rec["onset_ts"] = None
        rec["onset_utc"] = None
        rec["runway_days"] = None
        rec["n_clean_bins"] = int(clean.sum())
        rec["adequate"] = bool(clean.sum() >= MIN_CONTROL_CLEAN_BINS
                               and rec["do_clean_min"] is not None
                               and rec["do_clean_min"] > 4.5)
    return rec


def main():
    npzs = sorted(p for p in OUTDIR.glob("*.npz"))
    sites = {}
    ep = []; ct = []
    for p in npzs:
        f = site_facts(p)
        sites[f["site"]] = f
        if f["role"] == "episode" and f["adequate"]:
            ep.append(f["site"])
        elif f["role"] == "control" and f["adequate"]:
            ct.append(f["site"])

    manifest = {
        "network": "USGS_NWIS_iv",
        "threshold_do_mgL": DO_THR,
        "baseline_days": BASELINE_DAYS,
        "bin_seconds": BIN,
        "real_params": REAL_PARAMS,
        "min_preonset_do_bins": MIN_PREONSET_DO_BINS,
        "n_episode_sites": len(ep),
        "n_control_sites": len(ct),
        "episode_sites": ep,
        "control_sites": ct,
        "enough": bool(len(ep) >= 6 and len(ct) >= 6),
        "sites": sites,
    }
    (OUTDIR / "usgs_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"episode(adequate)={len(ep)} control(adequate)={len(ct)} enough={manifest['enough']}")
    print("episodes:", ep)
    print("controls:", ct)
    # list adequate-failing episodes/controls for transparency
    for s, f in sites.items():
        if not f["adequate"] and f["role"] in ("episode", "control"):
            print(f"  NOT-adequate {s} role={f['role']} "
                  f"preonset_do={f.get('n_preonset_do_bins')} runway={f.get('runway_days')} "
                  f"clean={f.get('n_clean_bins')} doMin={f.get('do_clean_min')}")
    return manifest


if __name__ == "__main__":
    main()
