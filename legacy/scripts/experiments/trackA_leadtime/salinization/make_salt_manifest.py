#!/usr/bin/env python
"""Build the salinization manifest from the built .npz set (source of truth).

Recomputes per-site facts directly from the written .npz, recomputing the SAME
site-relative conductance threshold used in build (median + 4*IQR/1.349 or p99.5),
verifies adequate CONDUCTANCE coverage before onset (episode) / broad clean
(control), and reports counts.

Writes TWO files:
  results/trackA_leadtime/salinization/conductance_events.json   (task deliverable)
  data/processed/trackA_salt/salt_manifest.json                  (Phase-2 downstream,
      same shape as usgs_manifest.json: episode_sites/control_sites/sites[...])

enough = (>=6 episode sites AND >=6 control sites) each with adequate covered
pre-onset (episode) / broad clean (control) conductance windows.

Adequate pre-onset (episode): >= WIN*4 real CONDUCTANCE bins strictly before onset
AND runway_days >= BASELINE_DAYS. (The detector residual uses err[:,3]=conductance,
so conductance coverage is what matters for scoring pre-onset windows.)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/bcheng/SENTINEL")
NPZDIR = ROOT / "data" / "processed" / "trackA_salt"
RESDIR = ROOT / "results" / "trackA_leadtime" / "salinization"
RESDIR.mkdir(parents=True, exist_ok=True)

BIN = 900
DAY = 86400
WIN = 128
BASELINE_DAYS = 7
REAL_PARAMS = ["DO", "pH", "turbidity", "conductivity"]
MIN_PREONSET_COND_BINS = WIN * 4      # >=512 real conductance bins pre-onset
MIN_CONTROL_CLEAN_BINS = 4000         # broad clean conductance record for controls


def cond_threshold(cvals: np.ndarray):
    med = float(np.nanmedian(cvals))
    q25, q75 = np.nanpercentile(cvals, [25, 75])
    iqr_sd = (q75 - q25) / 1.349
    p995 = float(np.nanpercentile(cvals, 99.5))
    thr = float(max(med + 4.0 * iqr_sd, p995))
    return med, thr, p995


def site_facts(npz: Path) -> dict:
    d = np.load(npz, allow_pickle=True)
    ts = d["ts"]
    c = d["conductivity"]; mc = d["mask_conductivity"]
    role = str(d["role"]); onset_i = int(d["onset_index"])
    present = {p: bool(d[f"mask_{p}"].sum() > 0) for p in REAL_PARAMS}
    n_days = round((ts[-1] - ts[0]) / DAY, 1)
    clean = mc & np.isfinite(c)
    cvals = c[clean]
    med, thr, p995 = cond_threshold(cvals) if clean.any() else (None, None, None)
    rec = {"site": npz.stem, "role": role, "onset_index": onset_i,
           "n_bins": int(ts.size), "n_days": n_days,
           "params_present": present,
           "cond_median": round(med, 1) if med is not None else None,
           "cond_thr": round(thr, 1) if thr is not None else None,
           "cond_p995": round(p995, 1) if p995 is not None else None,
           "cond_max_clean": round(float(np.nanmax(cvals)), 1) if clean.any() else None,
           "cond_cov": round(float(mc.mean()), 3),
           "n_clean_cond_bins": int(clean.sum())}
    if onset_i >= 0:
        t0 = int(ts[onset_i])
        rec["onset_ts"] = t0
        rec["onset_utc"] = pd.to_datetime(t0, unit="s", utc=True).isoformat()
        rec["cond_peak"] = round(float(c[onset_i]), 1)
        pre = (ts < t0)
        preonset = int((mc & pre).sum())
        rec["n_preonset_cond_bins"] = preonset
        base = clean & pre & (c < (thr if thr else np.inf))
        if base.any():
            first_t = ts[np.where(base)[0][0]]
            rec["runway_days"] = round((t0 - first_t) / DAY, 1)
        else:
            rec["runway_days"] = 0.0
        # baseline = median clean conductance in the 7 d before onset
        blo = t0 - BASELINE_DAYS * DAY
        bmask = clean & (ts >= blo) & (ts < t0)
        rec["baseline"] = round(float(np.nanmedian(c[bmask])), 1) if bmask.any() else None
        rec["adequate"] = bool(preonset >= MIN_PREONSET_COND_BINS
                               and rec["runway_days"] >= BASELINE_DAYS)
    else:
        rec["onset_ts"] = None
        rec["onset_utc"] = None
        rec["cond_peak"] = None
        rec["baseline"] = round(med, 1) if med is not None else None
        rec["runway_days"] = None
        rec["adequate"] = bool(clean.sum() >= MIN_CONTROL_CLEAN_BINS)
    return rec


def main():
    npzs = sorted(p for p in NPZDIR.glob("*.npz"))
    sites = {}
    ep = []; ct = []
    for p in npzs:
        f = site_facts(p)
        sites[f["site"]] = f
        if f["role"] == "episode" and f["adequate"]:
            ep.append(f["site"])
        elif f["role"] == "control" and f["adequate"]:
            ct.append(f["site"])

    enough = bool(len(ep) >= 6 and len(ct) >= 6)

    # Phase-2 downstream manifest (usgs_manifest.json shape)
    manifest = {
        "network": "USGS_NWIS_iv_salinization",
        "event_channel": "specific_conductance_00095",
        "event_definition": ("first QC-clean specific-conductance > site-relative "
                             "HIGH threshold [max(median+4*IQR/1.349, p99.5)] sustained "
                             ">=6 h, preceded by >=7 d clean baseline; artifact + "
                             "monotone-ramp screened; peak/thr>=1.05 separation"),
        "baseline_days": BASELINE_DAYS,
        "bin_seconds": BIN,
        "real_params": REAL_PARAMS,
        "min_preonset_cond_bins": MIN_PREONSET_COND_BINS,
        "n_episode_sites": len(ep),
        "n_control_sites": len(ct),
        "episode_sites": ep,
        "control_sites": ct,
        "enough": enough,
        "sites": sites,
    }
    (NPZDIR / "salt_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    # Task deliverable: conductance_events.json (per site role/onset/cond_peak/baseline/runway)
    events = {}
    for s, f in sites.items():
        events[s] = {
            "role": f["role"],
            "onset_ts": f["onset_ts"],
            "onset_utc": f["onset_utc"],
            "cond_peak": f["cond_peak"],
            "cond_thr": f["cond_thr"],
            "baseline": f["baseline"],
            "runway_days": f["runway_days"],
            "adequate": f["adequate"],
            "params_present": f["params_present"],
            "cond_cov": f["cond_cov"],
            "n_clean_cond_bins": f["n_clean_cond_bins"],
            "n_days": f["n_days"],
        }
    deliverable = {
        "event_definition": manifest["event_definition"],
        "n_event_sites_adequate": len(ep),
        "n_control_sites_adequate": len(ct),
        "enough": enough,
        "event_sites": ep,
        "control_sites": ct,
        "sites": events,
    }
    (RESDIR / "conductance_events.json").write_text(
        json.dumps(deliverable, indent=2, default=str))

    print(f"episode(adequate)={len(ep)} control(adequate)={len(ct)} enough={enough}")
    print("episodes:", ep)
    print("controls:", ct)
    for s, f in sites.items():
        if not f["adequate"] and f["role"] in ("episode", "control"):
            print(f"  NOT-adequate {s} role={f['role']} "
                  f"preonset_cond={f.get('n_preonset_cond_bins')} "
                  f"runway={f.get('runway_days')} clean={f.get('n_clean_cond_bins')}")
    return manifest


if __name__ == "__main__":
    main()
