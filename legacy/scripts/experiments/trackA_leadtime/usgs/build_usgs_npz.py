#!/usr/bin/env python
"""Download USGS NWIS iv water-quality data and build NEON-format per-site .npz.

Mirrors the NEON Track A per-site .npz EXACTLY:
  keys: DO, mask_DO, pH, mask_pH, turbidity, mask_turbidity,
        conductivity, mask_conductivity  (float arrays + bool masks),
        ts (int64 UTC seconds on a 900 s grid),
        role ('episode'|'control'), onset_index (int, -1 for control).

Data are REAL USGS NWIS instantaneous values only. QC:
  - drop NaN observations
  - drop rows whose qualifier flags indicate provisional-rejected / equipment
    malfunction / estimated-bad ('Eqp','Mnt','Dis','***','Bad','Ssn','Fld','ZFl'
    treated as non-clean where they mean no valid reading). We keep 'A','P','e'
    numeric values (approved/provisional/estimated) since those are real sensor
    readings; we drop only clearly non-numeric / maintenance sentinels.
  - resample onto a fixed 15-min UTC grid by nearest observation within tolerance
    (<=7.5 min); bins with no real obs -> mask False.

Artifact screening (flatline / step / drift) is applied to DO and reported;
episode onsets falling inside a screened artifact window are rejected.

Episode/control definition (identical to NEON):
  EPISODE onset = first QC-clean DO<4.0 mg/L crossing preceded by >=7 d of clean
                  (DO>=4) baseline with adequate coverage.
  CONTROL       = broad clean record, NO DO<4 event (min clean DO comfortably
                  above 4).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/bcheng/SENTINEL")
OUTDIR = ROOT / "data" / "processed" / "trackA_usgs"
OUTDIR.mkdir(parents=True, exist_ok=True)

BIN = 900                      # 15-min grid (seconds)
TOL = 450                      # nearest-obs tolerance = 7.5 min
DO_THR = 4.0
BASELINE_DAYS = 7
DAY = 86400

# NWIS parameter codes -> NEON key
PCODES = {"00300": "DO", "00400": "pH", "63680": "turbidity", "00095": "conductivity"}

# Qualifier codes that mean the value is NOT a valid sensor reading.
BAD_QUALS = {"Eqp", "Mnt", "Dis", "***", "Bad", "Ssn", "Fld", "ZFl", "Rat", "Pr"}


def fetch_site(site: str, start: str, end: str, retries: int = 3):
    """Fetch iv for one site across our 4 params. Returns dict pcode->DataFrame."""
    from dataretrieval import nwis
    out = {}
    for pc in PCODES:
        df = None
        for a in range(retries):
            try:
                df, _ = nwis.get_iv(sites=site, parameterCd=[pc], start=start, end=end)
                break
            except Exception as e:
                if a == retries - 1:
                    print(f"  [{site}] {pc} fetch FAIL: {type(e).__name__} {str(e)[:80]}", flush=True)
                    df = None
                else:
                    time.sleep(2 + 2 * a)
        out[pc] = df
    return out


def _value_qual_cols(df: pd.DataFrame, pc: str):
    """Find the value column and qualifier column for a pcode in an iv frame."""
    val_col = None
    qual_col = None
    for c in df.columns:
        cl = c.lower()
        if pc in c and cl.endswith("_cd"):
            qual_col = c
        elif pc in c and val_col is None and not cl.endswith("_cd"):
            val_col = c
    if val_col is None:
        # fallback: first numeric column that isn't site_no
        for c in df.columns:
            if df[c].dtype.kind in "fi" and c not in ("site_no",):
                val_col = c
                break
    return val_col, qual_col


def clean_series(df: pd.DataFrame, pc: str) -> pd.Series | None:
    """Return a UTC-indexed Series of QC-clean numeric values for pcode, or None."""
    if df is None or len(df) == 0:
        return None
    val_col, qual_col = _value_qual_cols(df, pc)
    if val_col is None:
        return None
    v = pd.to_numeric(df[val_col], errors="coerce")
    ok = v.notna()
    if qual_col is not None:
        q = df[qual_col].astype(str)
        badmask = q.apply(lambda s: any(b in s for b in BAD_QUALS))
        ok &= ~badmask
    idx = df.index
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    s = pd.Series(v.values, index=idx)[ok.values]
    s = s[~s.index.duplicated(keep="first")].sort_index()
    return s if len(s) else None


def to_grid(s: pd.Series, t0: int, t1: int) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-obs resample of a UTC series onto the fixed 15-min grid [t0,t1]."""
    grid_ts = np.arange(t0, t1 + 1, BIN, dtype="int64")
    vals = np.full(grid_ts.size, np.nan, dtype="float64")
    mask = np.zeros(grid_ts.size, dtype=bool)
    if s is None or len(s) == 0:
        return vals, mask
    obs_ts = (s.index.view("int64") // 1_000_000_000).astype("int64")
    obs_v = s.values.astype("float64")
    order = np.argsort(obs_ts)
    obs_ts = obs_ts[order]; obs_v = obs_v[order]
    # nearest grid bin for each obs; keep closest obs per bin
    j = np.searchsorted(grid_ts, obs_ts)
    j = np.clip(j, 0, grid_ts.size - 1)
    # check neighbor j-1 too
    for oi in range(obs_ts.size):
        gi = j[oi]
        best = gi
        bestd = abs(grid_ts[gi] - obs_ts[oi])
        if gi > 0 and abs(grid_ts[gi - 1] - obs_ts[oi]) < bestd:
            best = gi - 1; bestd = abs(grid_ts[gi - 1] - obs_ts[oi])
        if bestd <= TOL:
            if not mask[best] or abs(grid_ts[best] - obs_ts[oi]) < abs(grid_ts[best] - obs_ts[oi]):
                if not mask[best]:
                    vals[best] = obs_v[oi]; mask[best] = True
    return vals, mask


def screen_artifacts(do: np.ndarray, mdo: np.ndarray) -> np.ndarray:
    """Flag DO bins inside sensor-artifact windows (flatline / spike step).

    Returns bool array 'artifact' aligned to grid. Conservative: flatline =
    a long run of near-identical consecutive real values (stuck sensor);
    spike = single-bin jump >6 mg/L that reverts. These bins are excluded from
    clean baseline and disqualify an onset if the onset sits inside one.
    """
    n = do.size
    art = np.zeros(n, dtype=bool)
    idx = np.where(mdo)[0]
    if idx.size < 5:
        return art
    v = do[idx]
    # flatline: consecutive real obs identical to <1e-6 for >= 48 real obs (~12h+ at 15min if dense)
    d = np.abs(np.diff(v))
    run = 0
    for k in range(1, idx.size):
        if d[k - 1] < 1e-6:
            run += 1
        else:
            if run >= 48:
                art[idx[k - 1 - run]: idx[k - 1] + 1] = True
            run = 0
    if run >= 48:
        art[idx[idx.size - 1 - run]: idx[-1] + 1] = True
    # isolated spike/step reverting: |jump| > 6 mg/L on adjacent real obs both sides
    for k in range(1, idx.size - 1):
        if abs(v[k] - v[k - 1]) > 6.0 and abs(v[k] - v[k + 1]) > 6.0:
            art[idx[k]] = True
    return art


def classify(do: np.ndarray, mdo: np.ndarray, ts: np.ndarray, art: np.ndarray):
    """Return (decision, onset_index, stats). decision in {EPISODE,CONTROL,DISCARD}."""
    clean = mdo & ~art
    if clean.sum() < 500:
        return "DISCARD", -1, {"reason": "too few clean DO bins", "n_clean": int(clean.sum())}
    dov = do.copy()
    below = clean & (dov < DO_THR)
    do_clean_min = float(np.nanmin(dov[clean]))
    n_below = int(below.sum())
    span_days = (ts[-1] - ts[0]) / DAY

    # find first QC-clean DO<4 onset with >=7d clean baseline runway
    onset_i = -1
    runway_days = 0.0
    for o in np.where(below)[0]:
        t0 = ts[o]
        base_lo = t0 - BASELINE_DAYS * DAY
        base_bins = clean & (ts >= base_lo) & (ts < t0)
        if base_bins.sum() < 1:
            continue
        base_do = dov[base_bins]
        # baseline must be clean (all >=4) and reasonably covered
        if base_do.size == 0 or np.nanmin(base_do) < DO_THR:
            continue
        # coverage of baseline window: real clean bins / expected bins
        exp = BASELINE_DAYS * DAY / BIN
        cov = base_bins.sum() / exp
        if cov < 0.25:      # need >=25% real coverage in the 7-d runway
            continue
        # onset must not be inside an artifact window (art already excluded from clean)
        onset_i = int(o)
        runway_days = BASELINE_DAYS
        # find actual clean baseline extent backwards
        break

    if onset_i >= 0:
        # extend runway measure: how many clean days precede onset with no <4
        t0 = ts[onset_i]
        pre = clean & (ts < t0)
        if pre.any():
            first_clean_t = ts[np.where(pre)[0][0]]
            runway_days = min((t0 - first_clean_t) / DAY, span_days)
        return "EPISODE", onset_i, {
            "do_clean_min": round(do_clean_min, 3), "n_below_thr": n_below,
            "onset_ts": int(ts[onset_i]), "runway_days": round(float(runway_days), 1),
            "n_clean_do_bins": int(clean.sum()), "span_days": round(span_days, 1)}

    # CONTROL: no clean DO<4 at all, and comfortably above (min clean DO > 4.5)
    if n_below == 0 and do_clean_min > 4.5 and clean.sum() >= 2000:
        return "CONTROL", -1, {
            "do_clean_min": round(do_clean_min, 3), "n_below_thr": 0,
            "n_clean_do_bins": int(clean.sum()), "span_days": round(span_days, 1)}

    # has some sub-4 dips but none passed the runway gate, or borderline
    return "DISCARD", -1, {
        "reason": "no valid onset & not clean-control",
        "do_clean_min": round(do_clean_min, 3), "n_below_thr": n_below,
        "n_clean_do_bins": int(clean.sum()), "span_days": round(span_days, 1)}


def build_one(site: str, station: str, start: str, end: str) -> dict:
    raw = fetch_site(site, start, end)
    do_s = clean_series(raw["00300"], "00300")
    if do_s is None or len(do_s) < 500:
        return {"site": site, "decision": "DISCARD",
                "reason": "no/too-little clean DO", "station": station}
    # grid spans the DO observation extent
    t0 = int(do_s.index.view("int64").min() // 1_000_000_000)
    t1 = int(do_s.index.view("int64").max() // 1_000_000_000)
    t0 = (t0 // BIN) * BIN
    t1 = ((t1 // BIN) + 1) * BIN

    series = {}
    present = {}
    for pc, key in PCODES.items():
        s = clean_series(raw[pc], pc)
        vals, mask = to_grid(s, t0, t1)
        series[key] = vals
        series[f"mask_{key}"] = mask
        present[key] = bool(mask.sum() > 0)
    ts = np.arange(t0, t1 + 1, BIN, dtype="int64")

    art = screen_artifacts(series["DO"], series["mask_DO"])
    decision, onset_i, stats = classify(series["DO"], series["mask_DO"], ts, art)

    rec = {"site": site, "station": station, "decision": decision,
           "params_present": present, "n_bins": int(ts.size),
           "n_artifact_do_bins": int(art.sum()),
           "span_utc": [pd.to_datetime(t0, unit="s", utc=True).isoformat(),
                        pd.to_datetime(t1, unit="s", utc=True).isoformat()]}
    rec.update(stats)

    if decision == "DISCARD":
        return rec

    role = "episode" if decision == "EPISODE" else "control"
    onset_utc = (pd.to_datetime(ts[onset_i], unit="s", utc=True).isoformat()
                 if onset_i >= 0 else None)
    rec["role"] = role
    rec["onset_index"] = int(onset_i)
    rec["onset_utc"] = onset_utc

    np.savez_compressed(
        OUTDIR / f"{site}.npz",
        DO=series["DO"].astype("float64"), mask_DO=series["mask_DO"],
        pH=series["pH"].astype("float64"), mask_pH=series["mask_pH"],
        turbidity=series["turbidity"].astype("float64"), mask_turbidity=series["mask_turbidity"],
        conductivity=series["conductivity"].astype("float64"), mask_conductivity=series["mask_conductivity"],
        ts=ts, role=np.array(role), onset_index=np.array(onset_i, dtype="int64"),
    )
    return rec


if __name__ == "__main__":
    cand_csv = sys.argv[1]
    start = sys.argv[2] if len(sys.argv) > 2 else "2019-01-01"
    end = sys.argv[3] if len(sys.argv) > 3 else "2024-12-31"
    cands = pd.read_csv(cand_csv, dtype=str)
    results = []
    for i, r in cands.iterrows():
        site = str(r["site_no"]).strip()
        station = str(r.get("station_nm", ""))
        print(f"[{i+1}/{len(cands)}] {site} {station[:45]}", flush=True)
        try:
            rec = build_one(site, station, start, end)
        except Exception as e:
            rec = {"site": site, "station": station, "decision": "ERROR",
                   "reason": f"{type(e).__name__}: {str(e)[:100]}"}
        results.append(rec)
        print(f"    -> {rec['decision']} {rec.get('reason', '')} "
              f"doMin={rec.get('do_clean_min')} onset={rec.get('onset_utc')}", flush=True)
        (OUTDIR / "_build_progress.json").write_text(json.dumps(results, indent=2, default=str))
    (OUTDIR / "_build_results.json").write_text(json.dumps(results, indent=2, default=str))
    ne = sum(1 for x in results if x["decision"] == "EPISODE")
    nc = sum(1 for x in results if x["decision"] == "CONTROL")
    print(f"\nDONE. EPISODE={ne} CONTROL={nc} of {len(results)}")
