#!/usr/bin/env python
"""Download USGS NWIS iv water-quality data and build NEON-format per-site .npz,
defining EVENTS on RAW specific-conductance (freshwater SALINIZATION) instead of DO.

ADAPTED from scripts/experiments/trackA_leadtime/usgs/build_usgs_npz.py. Same QC,
same 15-min UTC grid, same nearest-obs resample, same NEON key schema. The ONLY
change is that role/onset are defined on the conductance (00095 -> 'conductivity')
channel with a defensible site-relative HIGH threshold, and artifact screening is
applied to the conductance channel.

NEON key schema (unchanged):
  DO, mask_DO, pH, mask_pH, turbidity, mask_turbidity,
  conductivity, mask_conductivity  (float arrays + bool masks),
  ts (int64 UTC seconds, 900 s grid),
  role ('episode'|'control'), onset_index (int, -1 for control).

EVENT (salinization) definition (NON-CIRCULAR vs detector residual; on RAW conductance):
  ONSET = first QC-clean specific-conductance value exceeding a HIGH site-relative
          threshold THR = max( median + 4*(IQR/1.349),  99.5th clean percentile )
          that is SUSTAINED >= 6 h (i.e. clean conductance stays > THR for the next
          6 h of real obs), PRECEDED by >= 7 d of clean below-threshold baseline
          with >= 25% real coverage.
  Artifact screen: flatline (stuck sensor), isolated revert spike/step, and monotone
          non-returning ramp (drift) reject an onset sitting inside the window.
  Guard on stationary tails: require peak/p99.5 ratio separation, i.e. the sustained
          excursion peak must exceed THR by a real margin (max_clean/thr >= 1.05) AND
          the excursion must RETURN toward baseline eventually is NOT required (road
          salt can plateau) but a monotone non-returning full-record ramp is rejected.
  CONTROL = broad clean conductance record with NO qualifying excursion (max clean
          conductance stays below THR by a comfortable margin) and adequate coverage.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/bcheng/SENTINEL")
OUTDIR = ROOT / "data" / "processed" / "trackA_salt"
OUTDIR.mkdir(parents=True, exist_ok=True)

BIN = 900                      # 15-min grid (seconds)
TOL = 450                      # nearest-obs tolerance = 7.5 min
BASELINE_DAYS = 7
SUSTAIN_H = 6                  # excursion must persist >=6 h
DAY = 86400

PCODES = {"00300": "DO", "00400": "pH", "63680": "turbidity", "00095": "conductivity"}
BAD_QUALS = {"Eqp", "Mnt", "Dis", "***", "Bad", "Ssn", "Fld", "ZFl", "Rat", "Pr"}


def fetch_site(site: str, start: str, end: str, retries: int = 3):
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
    val_col = None
    qual_col = None
    for c in df.columns:
        cl = c.lower()
        if pc in c and cl.endswith("_cd"):
            qual_col = c
        elif pc in c and val_col is None and not cl.endswith("_cd"):
            val_col = c
    if val_col is None:
        for c in df.columns:
            if df[c].dtype.kind in "fi" and c not in ("site_no",):
                val_col = c
                break
    return val_col, qual_col


def clean_series(df: pd.DataFrame, pc: str):
    if df is None or len(df) == 0:
        return None
    val_col, qual_col = _value_qual_cols(df, pc)
    if val_col is None:
        return None
    v = pd.to_numeric(df[val_col], errors="coerce")
    ok = v.notna()
    # physically implausible specific-conductance values -> drop
    if pc == "00095":
        ok &= (v > 0) & (v < 200000)
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


def to_grid(s, t0: int, t1: int):
    grid_ts = np.arange(t0, t1 + 1, BIN, dtype="int64")
    vals = np.full(grid_ts.size, np.nan, dtype="float64")
    mask = np.zeros(grid_ts.size, dtype=bool)
    if s is None or len(s) == 0:
        return vals, mask
    obs_ts = (s.index.view("int64") // 1_000_000_000).astype("int64")
    obs_v = s.values.astype("float64")
    order = np.argsort(obs_ts)
    obs_ts = obs_ts[order]; obs_v = obs_v[order]
    j = np.searchsorted(grid_ts, obs_ts)
    j = np.clip(j, 0, grid_ts.size - 1)
    for oi in range(obs_ts.size):
        gi = j[oi]
        best = gi
        bestd = abs(grid_ts[gi] - obs_ts[oi])
        if gi > 0 and abs(grid_ts[gi - 1] - obs_ts[oi]) < bestd:
            best = gi - 1; bestd = abs(grid_ts[gi - 1] - obs_ts[oi])
        if bestd <= TOL and not mask[best]:
            vals[best] = obs_v[oi]; mask[best] = True
    return vals, mask


def screen_artifacts_cond(c: np.ndarray, mc: np.ndarray) -> np.ndarray:
    """Flag conductance bins inside sensor-artifact windows.

    Conservative, conductance-tuned:
      - flatline: >=48 consecutive real obs identical to <1e-6 (stuck sensor).
      - isolated revert spike: single real obs jumps by a large *relative* factor
        (>3x median local level) vs BOTH neighbors and reverts (single-sample glitch).
    Monotone non-returning drift is handled at the onset gate (classify), not here.
    """
    n = c.size
    art = np.zeros(n, dtype=bool)
    idx = np.where(mc)[0]
    if idx.size < 5:
        return art
    v = c[idx]
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
    med = np.nanmedian(v)
    med = med if med > 0 else 1.0
    for k in range(1, idx.size - 1):
        # single-sample spike far above both neighbors (relative), reverting glitch
        if v[k] > 3.0 * v[k - 1] and v[k] > 3.0 * v[k + 1] and v[k] > 2.0 * med:
            art[idx[k]] = True
    return art


def classify_cond(c: np.ndarray, mc: np.ndarray, ts: np.ndarray, art: np.ndarray):
    """Return (decision, onset_index, stats). decision in {EPISODE,CONTROL,DISCARD}.

    EPISODE (salinization) / CONTROL defined on RAW conductance, site-relative.
    """
    clean = mc & ~art
    n_clean = int(clean.sum())
    if n_clean < 2000:
        return "DISCARD", -1, {"reason": "too few clean cond bins", "n_clean": n_clean}
    cv = c.copy()
    cvals = cv[clean]
    span_days = (ts[-1] - ts[0]) / DAY

    med = float(np.nanmedian(cvals))
    q25, q75 = np.nanpercentile(cvals, [25, 75])
    iqr_sd = (q75 - q25) / 1.349
    p995 = float(np.nanpercentile(cvals, 99.5))
    thr = float(max(med + 4.0 * iqr_sd, p995))
    max_clean = float(np.nanmax(cvals))

    # separation guard: genuine spike vs stationary tail
    ratio = max_clean / thr if thr > 0 else 0.0

    # bins that are clean AND above threshold
    above = clean & (cv > thr)
    n_above = int(above.sum())

    # sustained requirement: from an above-thr onset, next SUSTAIN_H hours of REAL
    # clean obs must remain > thr (allow small gaps: >=80% of real obs in window above).
    sustain_bins = int(SUSTAIN_H * 3600 / BIN)
    exp_base = BASELINE_DAYS * DAY / BIN

    def is_monotone_ramp():
        # reject sites whose clean conductance is a monotone non-returning ramp:
        # last-decile median far above first-decile AND never returns below thr after
        # first crossing to end. Cheap check: fraction of record above thr after onset.
        return False  # handled per-onset below

    onset_i = -1
    onset_peak = 0.0
    for o in np.where(above)[0]:
        t0 = ts[o]
        # sustained: window [t0, t0 + SUSTAIN_H]
        w_hi = t0 + SUSTAIN_H * 3600
        wbins = clean & (ts >= t0) & (ts <= w_hi)
        if wbins.sum() < max(4, 0.15 * sustain_bins):
            continue
        wvals_above = (clean & (cv > thr) & (ts >= t0) & (ts <= w_hi)).sum()
        if wvals_above < 0.8 * wbins.sum():
            continue
        # baseline: >=7 d clean below-thr with >=25% coverage
        base_lo = t0 - BASELINE_DAYS * DAY
        base_bins = clean & (ts >= base_lo) & (ts < t0)
        if base_bins.sum() < 1:
            continue
        base_vals = cv[base_bins]
        if np.nanmax(base_vals) >= thr:   # baseline must be sub-threshold (clean)
            continue
        cov = base_bins.sum() / exp_base
        if cov < 0.25:
            continue
        # separation: the sustained excursion peak must clear thr by real margin
        peak = float(np.nanmax(cv[wbins]))
        if peak / thr < 1.05:
            continue
        onset_i = int(o)
        onset_peak = peak
        break

    if onset_i >= 0:
        # monotone-non-returning ramp screen: after onset, does conductance ever
        # come back below thr? If it stays above for essentially the whole remainder
        # AND the record is short (<45 d) it may be an install-time step artifact.
        t0 = ts[onset_i]
        post = clean & (ts >= t0)
        post_frac_above = (clean & (cv > thr) & (ts >= t0)).sum() / max(1, post.sum())
        returns = bool((clean & (cv <= thr) & (ts > t0)).sum() > 20)
        post_days = (ts[-1] - t0) / DAY
        if (not returns) and post_frac_above > 0.98 and post_days < 45:
            return "DISCARD", -1, {
                "reason": "monotone non-returning ramp (possible install/drift step)",
                "thr": round(thr, 1), "max_clean": round(max_clean, 1),
                "n_clean_cond_bins": n_clean, "span_days": round(span_days, 1)}
        # runway = clean days preceding onset
        pre = clean & (ts < t0)
        runway_days = 0.0
        if pre.any():
            first_clean_t = ts[np.where(pre)[0][0]]
            runway_days = min((t0 - first_clean_t) / DAY, span_days)
        return "EPISODE", onset_i, {
            "cond_median": round(med, 1), "cond_thr": round(thr, 1),
            "cond_p995": round(p995, 1), "cond_max_clean": round(max_clean, 1),
            "cond_peak": round(onset_peak, 1), "peak_thr_ratio": round(peak / thr, 3)
            if (peak := onset_peak) else 0.0,
            "onset_ts": int(ts[onset_i]), "runway_days": round(float(runway_days), 1),
            "n_clean_cond_bins": n_clean, "span_days": round(span_days, 1),
            "n_above_thr": n_above, "iqr_sd": round(iqr_sd, 2)}

    # CONTROL: broad clean record, no qualifying excursion, and max stays comfortably
    # below thr (i.e. record is a stationary tail, not a spike) -> ratio near 1.
    if n_clean >= 4000 and n_above < 5 and ratio < 1.05:
        return "CONTROL", -1, {
            "cond_median": round(med, 1), "cond_thr": round(thr, 1),
            "cond_max_clean": round(max_clean, 1), "peak_thr_ratio": round(ratio, 3),
            "n_clean_cond_bins": n_clean, "span_days": round(span_days, 1),
            "n_above_thr": n_above}

    return "DISCARD", -1, {
        "reason": "no valid sustained onset & not clean-control",
        "cond_thr": round(thr, 1), "cond_max_clean": round(max_clean, 1),
        "peak_thr_ratio": round(ratio, 3), "n_above_thr": n_above,
        "n_clean_cond_bins": n_clean, "span_days": round(span_days, 1)}


def build_one(site: str, station: str, start: str, end: str) -> dict:
    raw = fetch_site(site, start, end)
    cond_s = clean_series(raw["00095"], "00095")
    if cond_s is None or len(cond_s) < 2000:
        return {"site": site, "decision": "DISCARD",
                "reason": "no/too-little clean conductance", "station": station,
                "n_cond_obs": (0 if cond_s is None else len(cond_s))}
    t0 = int(cond_s.index.view("int64").min() // 1_000_000_000)
    t1 = int(cond_s.index.view("int64").max() // 1_000_000_000)
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

    art = screen_artifacts_cond(series["conductivity"], series["mask_conductivity"])
    decision, onset_i, stats = classify_cond(
        series["conductivity"], series["mask_conductivity"], ts, art)

    rec = {"site": site, "station": station, "decision": decision,
           "params_present": present, "n_bins": int(ts.size),
           "n_artifact_cond_bins": int(art.sum()),
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
        conductivity=series["conductivity"].astype("float64"),
        mask_conductivity=series["mask_conductivity"],
        ts=ts, role=np.array(role), onset_index=np.array(onset_i, dtype="int64"),
    )
    return rec


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor, as_completed
    cand_csv = sys.argv[1]
    start = sys.argv[2] if len(sys.argv) > 2 else "2021-06-01"
    end = sys.argv[3] if len(sys.argv) > 3 else "2025-06-30"
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    cands = pd.read_csv(cand_csv, dtype=str)
    rows = [r for _, r in cands.iterrows()]

    def work(row):
        site = str(row["site_no"]).strip()
        station = str(row.get("station_nm", ""))
        try:
            rec = build_one(site, station, start, end)
            rec["tidal_or_mainstem"] = str(row.get("tidal_or_mainstem", ""))
            rec["state"] = str(row.get("state", ""))
            return rec
        except Exception as e:
            return {"site": site, "station": station, "decision": "ERROR",
                    "reason": f"{type(e).__name__}: {str(e)[:120]}"}

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, r): str(r["site_no"]).strip() for r in rows}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            results.append(rec)
            print(f"[{i}/{len(rows)}] {rec['site']} -> {rec['decision']} "
                  f"{rec.get('reason','')} thr={rec.get('cond_thr')} "
                  f"peak={rec.get('cond_peak')} onset={rec.get('onset_utc')} "
                  f"runway={rec.get('runway_days')}", flush=True)
            (OUTDIR / "_build_results.json").write_text(
                json.dumps(results, indent=2, default=str))

    ne = sum(1 for x in results if x["decision"] == "EPISODE")
    nc = sum(1 for x in results if x["decision"] == "CONTROL")
    nd = sum(1 for x in results if x["decision"] == "DISCARD")
    nerr = sum(1 for x in results if x["decision"] == "ERROR")
    print(f"\nDONE. EPISODE={ne} CONTROL={nc} DISCARD={nd} ERROR={nerr} of {len(results)}")
