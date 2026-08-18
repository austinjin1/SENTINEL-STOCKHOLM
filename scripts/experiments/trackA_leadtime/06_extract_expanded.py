#!/usr/bin/env python
"""Track A expansion — extract ADDITIONAL real NEON sites (episodes + controls).

Reuses 01_extract_episodes.py's EXACT logic (same QC gating on *FinalQF==0, same
15-min resample grid, same DO<4 hypoxia threshold, same >=7 d clean-baseline
onset rule) but applies it to NEON aquatic sites NOT in the original 22-site set.
This lets us grow the real-event set (sub-goal 2) without touching the frozen
originals.

Candidate sites are auto-classified from the data itself:
  * EPISODE  if find_onset() returns a real QC-clean DO<4 crossing preceded by
             >=7 d clean baseline (a well-defined lead time exists).
  * CONTROL  if the site has DO coverage but essentially no hypoxia (min clean
             DO comfortably >= threshold, and no qualifying onset).
  * DROP     if too little clean DO coverage to be usable either way.

No model here — pure real-data extraction. Writes one npz per NEW site into the
same data/processed/trackA/ dir (distinct site names, originals untouched) and a
JSON manifest of the classification with the raw evidence for each decision.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import importlib.util

# import the frozen extractor to reuse resample_site / find_onset verbatim
spec = importlib.util.spec_from_file_location("extract01", HERE / "01_extract_episodes.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)

ROOT = Path(__file__).resolve().parents[3]
PARQUET = ROOT / "data" / "raw" / "neon_aquatic" / "neon_DP1.20288.001_all.parquet"
OUTDIR = ROOT / "data" / "processed" / "trackA"
OUTDIR.mkdir(parents=True, exist_ok=True)

SITE_RE = re.compile(r"_([A-Z]{4})_\d{4}-\d{2}_")

# Sites present in raw dir but NOT in the original 22-site set.
ORIGINAL = set(E.ALL_SITES)
CANDIDATES = ["BLWA", "POSE", "PRLA", "PRPO", "REDB",
              "SUGG", "SYCA", "TECR", "TOMB", "TOOK", "WALK", "WLOU"]

DO_THR = E.DO_THRESHOLD          # 4.0
MIN_CLEAN_DO_BINS = 2000         # need enough clean DO to be a usable control/episode
CONTROL_DO_MARGIN = 0.5          # control: clean-DO min stays >= thr + margin


def stream_site_frames(sites):
    pf = pq.ParquetFile(PARQUET)
    want = set(sites)
    buckets = {s: [] for s in sites}
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=E.READ_COLS).to_pandas()
        s = df["source_file"].str.extract(SITE_RE, expand=False)
        df = df.assign(_site=s)
        hit = df["_site"].isin(want)
        if not hit.any():
            continue
        df = df[hit]
        for site, sub in df.groupby("_site"):
            buckets[site].append(sub.drop(columns=["source_file", "_site"]))
        print(f"  rg {rg+1}/{pf.num_row_groups}: +{int(hit.sum())} rows", flush=True)
    return buckets


def main():
    print(f"Streaming {PARQUET.name} for {len(CANDIDATES)} candidate sites ...", flush=True)
    buckets = stream_site_frames(CANDIDATES)
    manifest = {"threshold_do_mgL": DO_THR, "baseline_days": E.BASELINE_DAYS,
                "bin_seconds": E.BIN_SECONDS, "sites": {}}
    for site in CANDIDATES:
        if site in ORIGINAL:
            continue
        frames = buckets[site]
        if not frames:
            manifest["sites"][site] = {"decision": "DROP", "reason": "no_rows"}
            print(f"[{site}] DROP no rows", flush=True)
            continue
        df = pd.concat(frames, ignore_index=True)
        series = E.resample_site(df)
        ts, do, dom = series["ts"], series["DO"], series["mask_DO"]
        n_clean = int(dom.sum())
        if n_clean < MIN_CLEAN_DO_BINS:
            manifest["sites"][site] = {"decision": "DROP",
                                       "reason": "too_few_clean_do_bins",
                                       "n_clean_do_bins": n_clean}
            print(f"[{site}] DROP clean_do={n_clean}", flush=True)
            continue

        onset_i = E.find_onset(ts, do, dom)
        # is the onset a STRONG one (>=7d clean baseline verified, not fallback)?
        strong_onset = False
        if onset_i is not None:
            t0 = ts[onset_i]
            window = (ts >= t0 - E.BASELINE_DAYS * 86400) & (ts < t0)
            prior = dom & window
            strong_onset = bool(prior.sum() > 0 and np.nanmin(do[prior]) >= DO_THR)

        do_min = float(np.nanmin(do[dom]))
        n_below = int((dom & (do < DO_THR)).sum())

        if strong_onset:
            role, decision = "episode", "EPISODE"
            oi = onset_i
        elif do_min >= DO_THR + CONTROL_DO_MARGIN and n_below == 0:
            role, decision = "control", "CONTROL"
            oi = None
        else:
            # has some hypoxia but no clean-baselined onset -> not usable cleanly
            manifest["sites"][site] = {
                "decision": "DROP",
                "reason": "hypoxia_without_clean_baseline_onset" if n_below else "ambiguous",
                "n_clean_do_bins": n_clean, "do_clean_min": do_min,
                "n_below_thr": n_below, "onset_index": onset_i,
                "strong_onset": strong_onset,
            }
            print(f"[{site}] DROP do_min={do_min:.2f} n_below={n_below} "
                  f"onset_i={onset_i} strong={strong_onset}", flush=True)
            continue

        onset_ts = int(ts[oi]) if oi is not None else None
        np.savez_compressed(
            OUTDIR / f"{site}.npz",
            ts=ts.astype("int64"),
            pH=series["pH"], DO=series["DO"],
            turbidity=series["turbidity"], conductivity=series["conductivity"],
            mask_pH=series["mask_pH"], mask_DO=series["mask_DO"],
            mask_turbidity=series["mask_turbidity"],
            mask_conductivity=series["mask_conductivity"],
            onset_index=np.array(-1 if oi is None else oi, dtype="int64"),
            role=np.array(role),
        )
        info = {
            "decision": decision, "role": role,
            "n_rows_raw": int(len(df)), "n_bins": int(ts.size),
            "n_clean_do_bins": n_clean, "do_clean_min": do_min,
            "n_below_thr": n_below,
            "onset_index": oi,
            "onset_utc": (pd.Timestamp(onset_ts, unit="s", tz="UTC").isoformat()
                          if onset_ts else None),
            "span_utc": [pd.Timestamp(int(ts[0]), unit="s", tz="UTC").isoformat(),
                         pd.Timestamp(int(ts[-1]), unit="s", tz="UTC").isoformat()],
        }
        manifest["sites"][site] = info
        print(f"[{site}] {decision} bins={ts.size} cleanDO={n_clean} "
              f"onset={info['onset_utc']} doMin={do_min:.2f}", flush=True)

    out = OUTDIR / "expanded_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    eps = [s for s, v in manifest["sites"].items() if v.get("role") == "episode"]
    ctl = [s for s, v in manifest["sites"].items() if v.get("role") == "control"]
    print(f"\nEPISODES(+{len(eps)}): {eps}")
    print(f"CONTROLS(+{len(ctl)}): {ctl}")
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
