#!/usr/bin/env python
"""TASK B helper — rebuild sensor embeddings with a STRICTLY-BEFORE-date window.

Turbidity and DO are among the 5 sensor channels. The shipped
build_sensor_window() includes observations up to `date + 23:59`, i.e. the
same-day reading — so predicting a same-day Turb/DO exceedance from that window
is circular. Here we rebuild the AquaSSM embedding using only observations
strictly BEFORE the tile date (cutoff = date - 1 day, 23:59), which removes the
same-day label leakage while keeping a real, causal sensor history. The result
is a non-circular sensor embedding for the turbidity / multi-parameter tasks.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import _fusion as F

ROOT = Path(__file__).resolve().parents[3]
PAIRS_NPZ = ROOT / "data" / "processed" / "satellite" / "drone_wq_pairs.npz"
TB = ROOT / "data" / "processed" / "trackB"


def build_sensor_window_strict_before(site: str, date: str):
    """Most-recent <=WIN obs STRICTLY BEFORE `date` (no same-day) -> (x, dt, n)."""
    df = F._site_df(site)
    if df is None or len(df) == 0:
        return None
    # cutoff = end of the PREVIOUS day -> excludes the same-day label reading
    cutoff = pd.Timestamp(date, tz="UTC") - pd.Timedelta(minutes=1)
    sub = df[df.index <= cutoff]
    if len(sub) < 32:
        return None
    sub = sub.iloc[-F.WIN:]
    n = len(sub)
    x = np.zeros((F.WIN, 6), "float32")
    dt = np.full(F.WIN, 900.0, "float32")
    off = F.WIN - n
    idx_s = (sub.index.asi8 // 1_000_000_000).astype("float64")
    d = np.diff(idx_s, prepend=idx_s[0] - 900.0)
    d = np.clip(d, 60.0, 30 * 86400.0)
    dt[off:] = d.astype("float32")
    for col, ch in F.COL2CH.items():
        if col in sub.columns:
            v = pd.to_numeric(sub[col], errors="coerce").to_numpy("float64")
            c, s = F.SENSOR_STATS[ch]
            z = (v - c) / s
            z = np.where(np.isfinite(z), z, 0.0)
            x[off:, ch] = z.astype("float32")
    return x, dt, n


def main():
    z = np.load(PAIRS_NPZ, allow_pickle=True)
    md = json.loads(str(z["metadata"]))
    F.set_sensor_stats_from_targets(z["targets"])

    emb = np.load(TB / "embeddings_v2.npz", allow_pickle=True)
    tiles = emb["tile_idx"].tolist()

    X, DT, kept = [], [], []
    for i in tiles:
        m = md[int(i)]
        w = build_sensor_window_strict_before(m["site_id"], m["date"])
        if w is None:
            continue
        X.append(w[0]); DT.append(w[1]); kept.append(int(i))
    kept = np.array(kept)
    print(f"strict-before sensor windows: {len(kept)} / {len(tiles)}", flush=True)
    X = np.stack(X); DT = np.stack(DT)

    sen_enc = F.load_sensor_encoder()
    sen_emb = F.embed_sensor(sen_enc, X, DT)
    print("sensor_noleak", sen_emb.shape, flush=True)
    np.savez(TB / "sensor_noleak.npz", tile_idx=kept,
             sensor=sen_emb.astype("float32"))
    print("WROTE", TB / "sensor_noleak.npz")


if __name__ == "__main__":
    main()
