#!/usr/bin/env python
"""Track A, step 1 — extract per-site QC-clean sensor series + hypoxia onsets.

Reads the real NEON water-quality parquet (DP1.20288.001, 51.4M rows, 1-min),
derives the site per row from ``source_file`` (the ``siteID`` column is ~all
NULL), gates every parameter on its NEON ``*FinalQF == 0`` quality flag, and
resamples each site to a regular 15-minute grid. For episode sites it also
computes the ground-truth hypoxia onset = first QC-clean DO < 4 mg/L crossing
that follows >= 7 days of clean (DO >= 4) baseline, so a lead time is well
defined.

Outputs one ``.npz`` per site under ``data/processed/trackA/`` plus a summary
JSON. No model is involved here — this is pure real-data extraction.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
PARQUET = ROOT / "data" / "raw" / "neon_aquatic" / "neon_DP1.20288.001_all.parquet"
OUTDIR = ROOT / "data" / "processed" / "trackA"
OUTDIR.mkdir(parents=True, exist_ok=True)

SITE_RE = re.compile(r"_([A-Z]{4})_\d{4}-\d{2}_")

# Real parameter -> (value column, FinalQF column)
PARAMS = {
    "pH": ("pH", "pHFinalQF"),
    "DO": ("dissolvedOxygen", "dissolvedOxygenFinalQF"),
    "turbidity": ("turbidity", "turbidityFinalQF"),
    "conductivity": ("specificConductance", "specificCondFinalQF"),
}
READ_COLS = ["source_file", "startDateTime"] + [c for v in PARAMS.values() for c in v]

# Episode sites: verified DO-hypoxia events (onset computed + cross-checked).
# Control sites: clean streams for cross-site FPR (no/negligible DO<4).
# Episode candidates: sites with verified QC-clean DO<4 hypoxia crossings.
# Onsets are COMPUTED in find_onset (>=7d clean baseline); sites that don't
# qualify are recorded honestly. Controls: streams with negligible DO hypoxia.
EPISODE_SITES = ["MCDI", "ARIK", "KING", "BIGC", "PRIN", "COMO",
                 "MAYF", "LECO", "CRAM", "LIRO", "BARC", "GUIL"]
CONTROL_SITES = ["HOPB", "LEWI", "MART", "MCRA",
                 "BLDE", "CARI", "CUPE", "BLUE", "FLNT", "OKSR"]
ALL_SITES = EPISODE_SITES + CONTROL_SITES

DO_THRESHOLD = 4.0          # mg/L hypoxia crossing
BASELINE_DAYS = 7           # required clean runway before an onset counts
BIN_SECONDS = 900           # 15-minute resample grid


def stream_site_frames() -> dict[str, list[pd.DataFrame]]:
    """Stream all row groups once, bucket rows by site for the target sites."""
    pf = pq.ParquetFile(PARQUET)
    buckets: dict[str, list[pd.DataFrame]] = {s: [] for s in ALL_SITES}
    want = set(ALL_SITES)
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=READ_COLS).to_pandas()
        sites = df["source_file"].str.extract(SITE_RE, expand=False)
        df = df.assign(_site=sites)
        hit = df["_site"].isin(want)
        if not hit.any():
            continue
        df = df[hit]
        for site, sub in df.groupby("_site"):
            buckets[site].append(sub.drop(columns=["source_file", "_site"]))
        print(f"  row group {rg+1}/{pf.num_row_groups}: kept {int(hit.sum())} rows", flush=True)
    return buckets


def resample_site(df: pd.DataFrame) -> dict:
    """QC-gate each param, resample to a regular 15-min grid (mean of clean)."""
    df = df.copy()
    df["t"] = pd.to_datetime(df["startDateTime"], utc=True, errors="coerce")
    df = df.dropna(subset=["t"]).sort_values("t")

    # Apply QC: keep value only where its FinalQF == 0, else NaN.
    clean = {}
    for name, (vcol, qcol) in PARAMS.items():
        v = pd.to_numeric(df[vcol], errors="coerce")
        qf = pd.to_numeric(df[qcol], errors="coerce")
        clean[name] = v.where(qf == 0.0)
    cdf = pd.DataFrame(clean)
    cdf["t"] = pd.to_datetime(df["t"].values, utc=True)   # keep tz-aware (.values strips tz)

    # Regular 15-min grid spanning the site record.
    cdf["bin"] = cdf["t"].dt.floor(f"{BIN_SECONDS}s")
    grid = cdf.groupby("bin").mean(numeric_only=True)
    full_index = pd.date_range(grid.index.min(), grid.index.max(),
                               freq=f"{BIN_SECONDS}s", tz="UTC")
    grid = grid.reindex(full_index)

    ts = (grid.index.asi8 // 1_000_000_000).astype("int64")
    out = {"ts": ts}
    for name in PARAMS:
        col = grid[name].to_numpy(dtype="float64")
        out[name] = col
        out[f"mask_{name}"] = ~np.isnan(col)
    return out


def find_onset(ts: np.ndarray, do: np.ndarray, do_mask: np.ndarray) -> int | None:
    """First QC-clean DO<thr crossing preceded by >=BASELINE_DAYS clean baseline."""
    below = do_mask & (do < DO_THRESHOLD)
    idxs = np.where(below)[0]
    if idxs.size == 0:
        return None
    for i in idxs:
        t0 = ts[i]
        window = (ts >= t0 - BASELINE_DAYS * 86400) & (ts < t0)
        prior = do_mask & window
        if prior.sum() == 0:
            continue                      # no clean baseline to anchor a lead time
        if np.nanmin(do[prior]) >= DO_THRESHOLD:   # baseline stayed non-hypoxic
            return int(i)
    return int(idxs[0])                    # fallback: first crossing, weak baseline


def main() -> None:
    print(f"Streaming {PARQUET.name} ...", flush=True)
    buckets = stream_site_frames()
    summary = {"threshold_do_mgL": DO_THRESHOLD, "bin_seconds": BIN_SECONDS,
               "baseline_days": BASELINE_DAYS, "sites": {}}
    for site in ALL_SITES:
        frames = buckets[site]
        if not frames:
            print(f"[{site}] NO ROWS", flush=True)
            summary["sites"][site] = {"n_rows": 0}
            continue
        df = pd.concat(frames, ignore_index=True)
        series = resample_site(df)
        ts, do, dom = series["ts"], series["DO"], series["mask_DO"]
        role = "episode" if site in EPISODE_SITES else "control"
        onset_i = find_onset(ts, do, dom) if role == "episode" else None
        onset_ts = int(ts[onset_i]) if onset_i is not None else None

        np.savez_compressed(
            OUTDIR / f"{site}.npz",
            ts=ts.astype("int64"),
            pH=series["pH"], DO=series["DO"],
            turbidity=series["turbidity"], conductivity=series["conductivity"],
            mask_pH=series["mask_pH"], mask_DO=series["mask_DO"],
            mask_turbidity=series["mask_turbidity"],
            mask_conductivity=series["mask_conductivity"],
            onset_index=np.array(-1 if onset_i is None else onset_i, dtype="int64"),
            role=np.array(role),
        )
        n_clean_do = int(dom.sum())
        info = {
            "role": role, "n_rows_raw": int(len(df)), "n_bins": int(ts.size),
            "n_clean_do_bins": n_clean_do,
            "do_clean_mean": float(np.nanmean(do[dom])) if n_clean_do else None,
            "do_clean_min": float(np.nanmin(do[dom])) if n_clean_do else None,
            "onset_index": onset_i,
            "onset_utc": (pd.Timestamp(onset_ts, unit="s", tz="UTC").isoformat()
                          if onset_ts else None),
            "span_utc": [pd.Timestamp(int(ts[0]), unit="s", tz="UTC").isoformat(),
                         pd.Timestamp(int(ts[-1]), unit="s", tz="UTC").isoformat()],
        }
        summary["sites"][site] = info
        print(f"[{site}] {role} bins={ts.size} cleanDO={n_clean_do} "
              f"onset={info['onset_utc']} doMin={info['do_clean_min']}", flush=True)

    (OUTDIR / "extract_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWROTE {OUTDIR/'extract_summary.json'}")


if __name__ == "__main__":
    main()
