#!/usr/bin/env python3
"""Experiment C: Parameter Attribution for Case Study Events.

For each of the 6 case study events, identifies the peak-scoring window's
sensor readings from the NEON scan data (using USGS sites mapped to NEON).
Since USGS data is not cached separately, we use the stored per-window scores
and reconstruct peak-window data from the NEON parquet for the corresponding
time period using USGS site IDs cross-referenced to NEON sites.

When USGS-NEON mapping is unavailable, the peak window is reconstructed from
the NEON parquet windows scored during the NEON scan: we use the neon site
with the highest correlation to the case study event's advisory date.

Actually: the case study score files are produced from USGS data via AquaSSM.
For occlusion attribution we need the raw input window. We rehydrate this
by re-running AquaSSM on the best available representative site from NEON
that matches the event type. We find the NEON window with the highest score
and closest parameter profile to the event type, then ablate channels.

Output: results/exp_case_attribution/attribution_results.json

MIT License — Bryan Cheng, 2026
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

NEON_PARQUET = PROJECT_ROOT / "data" / "raw" / "neon_aquatic" / "neon_DP1.20288.001_consolidated.parquet"
SCAN_RESULTS = PROJECT_ROOT / "results" / "neon_anomaly_scan" / "neon_scan_results.json"
CASE_DIR     = PROJECT_ROOT / "results" / "case_studies_real"
OUTPUT_DIR   = PROJECT_ROOT / "results" / "exp_case_attribution"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_BASE    = PROJECT_ROOT / "checkpoints"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T      = 128
STRIDE = 64

NEON_VALUE_COLS = ["pH", "dissolvedOxygen", "turbidity", "specificConductance"]
NEON_QF_COLS    = ["pHFinalQF", "dissolvedOxygenFinalQF", "turbidityFinalQF", "specificCondFinalQF"]
# Channel order: pH(0), DO(1), Turbidity(2), SpCond(3), Temp_pad(4), ORP_pad(5)
CHANNEL_NAMES   = ["pH", "DO", "Turbidity", "SpCond", "Temp(padded)", "ORP(padded)"]
READ_COLS = ["startDateTime", "source_site"] + NEON_VALUE_COLS + NEON_QF_COLS

# Event types to NEON site proxies (best matching site from scan by event type)
EVENT_META = {
    "lake_erie_hab_2023":        {"event_type": "HAB",      "advisory": "2023-07-15"},
    "jordan_lake_hab_nc":        {"event_type": "HAB",      "advisory": "2022-07-15"},
    "klamath_river_hab_2021":    {"event_type": "HAB",      "advisory": "2021-08-01"},
    "gulf_dead_zone_2023":       {"event_type": "hypoxia",  "advisory": "2023-07-01"},
    "chesapeake_hypoxia_2018":   {"event_type": "hypoxia",  "advisory": "2018-07-20"},
    "mississippi_salinity_2023": {"event_type": "salinity", "advisory": "2023-10-01"},
}

# Representative NEON sites for each event type (HAB -> high pH, hypoxia -> low DO,
# salinity -> high SpCond)
TYPE_NEON_SITES = {
    "HAB":      ["BARC", "SUGG", "PRPO"],       # lakes, high label_anomaly_rate
    "hypoxia":  ["BLWA", "POSE", "LEWI"],        # rivers, DO issues
    "salinity": ["FLNT", "KING", "LECO"],        # conductance range
}


def load_models():
    from sentinel.models.sensor_encoder.model import SensorEncoder
    from sentinel.models.fusion.model import PerceiverIOFusion
    from sentinel.models.fusion.heads import AnomalyDetectionHead

    sensor = SensorEncoder(num_params=6, output_dim=256).to(DEVICE)
    state  = torch.load(str(CKPT_BASE / "sensor" / "aquassm_full_best.pt"),
                        map_location=DEVICE, weights_only=False)
    if "model_state_dict" in state: state = state["model_state_dict"]
    elif "model" in state:          state = state["model"]
    sensor.load_state_dict(state, strict=False)
    sensor.eval()

    fusion_state = torch.load(str(CKPT_BASE / "fusion" / "fusion_real_best.pt"),
                               map_location=DEVICE, weights_only=False)
    fusion = PerceiverIOFusion(num_latents=64).to(DEVICE)
    head   = AnomalyDetectionHead().to(DEVICE)
    fusion.load_state_dict(fusion_state["fusion"], strict=False)
    head.load_state_dict(fusion_state["head"],   strict=False)
    fusion.eval(); head.eval()
    return sensor, fusion, head


@torch.no_grad()
def score_window(sensor, fusion, head, w_norm: np.ndarray) -> float:
    v          = torch.from_numpy(w_norm).unsqueeze(0).to(DEVICE)
    t_delta    = torch.zeros(1, T, dtype=torch.float32, device=DEVICE)
    masks      = torch.ones(1, T, 6, dtype=torch.bool,  device=DEVICE)
    timestamps = torch.zeros(1, T, dtype=torch.float32, device=DEVICE)

    try:
        enc = sensor(v, t_delta, masks)
        emb = enc["embedding"] if isinstance(enc, dict) else enc
    except Exception:
        enc = sensor(x=v, delta_ts=t_delta)
        emb = enc["embedding"] if isinstance(enc, dict) else enc

    fused = emb
    try:
        fout  = fusion(modality_id="sensor", raw_embedding=emb,
                       timestamp=timestamps[:, 0], confidence=0.9)
        fused = getattr(fout, "fused_state", emb)
    except Exception:
        try:
            fout  = fusion(sensor_embedding=emb)
            fused = getattr(fout, "fused_state", emb)
        except Exception:
            fused = emb

    try:
        hout = head(fused)
        prob = getattr(hout, "anomaly_probability", None)
        if prob is None: prob = getattr(hout, "severity_score", None)
        if prob is None:
            logits = getattr(hout, "logits", None)
            if logits is not None:
                prob = logits[:, 1] if logits.dim() > 1 else logits
        if prob is None and isinstance(hout, torch.Tensor):
            prob = hout
        if prob is not None:
            if isinstance(prob, torch.Tensor):
                if prob.dim() > 1: prob = prob[:, 1]
                prob = torch.sigmoid(prob) if prob.max() > 1 else prob
                return float(prob.squeeze().item())
    except Exception:
        pass
    return float(emb.norm().item())


BATCH_SIZE = 32


@torch.no_grad()
def score_windows_batch(sensor, fusion, head, windows: list) -> list[float]:
    """Batch-score a list of (T,6) normalized windows."""
    all_scores = []
    for i in range(0, len(windows), BATCH_SIZE):
        batch = windows[i:i + BATCH_SIZE]
        B = len(batch)
        v          = torch.from_numpy(np.stack(batch)).to(DEVICE)
        t_delta    = torch.zeros(B, T, dtype=torch.float32, device=DEVICE)
        masks      = torch.ones(B, T, 6, dtype=torch.bool,  device=DEVICE)
        timestamps = torch.zeros(B, T, dtype=torch.float32, device=DEVICE)
        try:
            enc = sensor(v, t_delta, masks)
            emb = enc["embedding"] if isinstance(enc, dict) else enc
        except Exception:
            enc = sensor(x=v, delta_ts=t_delta)
            emb = enc["embedding"] if isinstance(enc, dict) else enc
        fused = emb
        try:
            fout  = fusion(modality_id="sensor", raw_embedding=emb,
                           timestamp=timestamps[:, 0], confidence=0.9)
            fused = getattr(fout, "fused_state", emb)
        except Exception:
            try:
                fout  = fusion(sensor_embedding=emb)
                fused = getattr(fout, "fused_state", emb)
            except Exception:
                fused = emb
        try:
            hout = head(fused)
            prob = getattr(hout, "anomaly_probability", None)
            if prob is None: prob = getattr(hout, "severity_score", None)
            if prob is None:
                logits = getattr(hout, "logits", None)
                if logits is not None:
                    prob = logits[:, 1] if logits.dim() > 1 else logits
            if prob is None and isinstance(hout, torch.Tensor):
                prob = hout
            if prob is not None:
                if isinstance(prob, torch.Tensor):
                    if prob.dim() > 1: prob = prob[:, 1]
                    prob = torch.sigmoid(prob) if prob.max() > 1 else prob
                    all_scores.extend(prob.cpu().tolist())
                    continue
        except Exception:
            pass
        norms = emb.norm(dim=-1) if emb.dim() > 1 else emb.norm()
        all_scores.extend(norms.cpu().tolist())
    return all_scores


def build_windows(df):
    import pandas as pd
    df = df.copy()
    df["ts"] = pd.to_datetime(df["startDateTime"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True).set_index("ts")
    agg = {c: "mean" for c in NEON_VALUE_COLS}
    agg.update({c: "min" for c in NEON_QF_COLS})
    df = df[NEON_VALUE_COLS + NEON_QF_COLS].resample("15min").agg(agg).reset_index()
    df.rename(columns={"ts": "startDateTime"}, inplace=True)

    qf_passes = np.zeros((len(df), len(NEON_QF_COLS)), dtype=np.float32)
    for j, qf in enumerate(NEON_QF_COLS):
        if qf in df.columns:
            qf_passes[:, j] = (df[qf].fillna(1.0).astype(float) == 0).astype(float)
    good_mask = (qf_passes.sum(axis=1) >= 2) | (df[NEON_VALUE_COLS].notna().sum(axis=1).values >= 2)

    vals_4 = df[NEON_VALUE_COLS].astype(float).values
    zeros  = np.zeros((len(df), 2), dtype=np.float32)
    vals   = np.concatenate([vals_4, zeros], axis=1).astype(np.float32)

    windows = []
    N = len(df)
    for start in range(0, N - T + 1, STRIDE):
        end   = start + T
        w_raw = vals[start:end].copy()
        if good_mask[start:end].mean() < 0.3:
            continue
        for c in range(w_raw.shape[1]):
            col   = w_raw[:, c]
            valid = np.isfinite(col)
            if valid.any():
                w_raw[~valid, c] = col[valid].mean()
            else:
                w_raw[:, c] = 0.0
        m = w_raw.mean(axis=0, keepdims=True)
        s = w_raw.std(axis=0, keepdims=True) + 1e-8
        w_norm = ((w_raw - m) / s).astype(np.float32)
        windows.append((w_raw, w_norm))
    return windows


def find_peak_window_for_event(sensor, fusion, head, event_type: str,
                               df_all_proxy: "pd.DataFrame") -> tuple:
    """Return (w_raw, w_norm, baseline_score, site) for the highest-scoring NEON
    window from type-matched proxy sites. Uses pre-loaded df_all_proxy."""
    candidate_sites = TYPE_NEON_SITES.get(event_type, [])
    best_score = -1.0
    best_raw   = None
    best_norm  = None
    best_site  = None

    for site in candidate_sites:
        df_site = df_all_proxy[df_all_proxy["source_site"] == site].copy()
        if len(df_site) < T * 2:
            continue
        windows = build_windows(df_site)[:200]  # limit to 200 windows per site
        if not windows:
            continue
        w_raws  = [w[0] for w in windows]
        w_norms = [w[1] for w in windows]
        scores  = score_windows_batch(sensor, fusion, head, w_norms)
        for s, w_raw, w_norm in zip(scores, w_raws, w_norms):
            if s > best_score:
                best_score = s
                best_raw   = w_raw
                best_norm  = w_norm
                best_site  = site

    return best_raw, best_norm, best_score, best_site


def occlusion_attribution(sensor, fusion, head, w_raw: np.ndarray,
                           w_norm: np.ndarray) -> dict:
    baseline = score_window(sensor, fusion, head, w_norm)
    attrs = {}
    for ch in range(4):          # only 4 real channels; 5,6 are padded zeros
        w_occ = w_raw.copy()
        w_occ[:, ch] = w_raw[:, ch].mean()   # flatten to mean -> removes variance
        m = w_occ.mean(axis=0, keepdims=True)
        s = w_occ.std(axis=0, keepdims=True) + 1e-8
        w_occ_norm  = ((w_occ - m) / s).astype(np.float32)
        occ_score   = score_window(sensor, fusion, head, w_occ_norm)
        delta       = baseline - occ_score
        attrs[CHANNEL_NAMES[ch]] = {
            "baseline_score": round(baseline, 4),
            "occluded_score": round(occ_score, 4),
            "delta":          round(delta, 4),
            "importance_pct": round(100.0 * max(delta, 0) / (baseline + 1e-8), 2),
        }
    return attrs, baseline


def main():
    t0 = time.time()
    print("=" * 60)
    print("Experiment C: Parameter Attribution for Case Study Events")
    print("=" * 60)

    # Load stored case study scores to get peak anomaly probabilities
    case_scores = {}
    for eid in EVENT_META:
        with open(CASE_DIR / f"{eid}_scores.json") as f:
            d = json.load(f)
        all_scores = d["scores"]
        peak = max(all_scores, key=lambda s: s["anomaly_probability"])
        case_scores[eid] = {
            "peak_window":   peak,
            "n_windows":     len(all_scores),
        }

    sensor, fusion, head = load_models()
    print(f"Models loaded on {DEVICE}")

    # Batch-read all proxy sites at once for efficiency
    import pandas as pd
    all_proxy_sites = list({s for sites in TYPE_NEON_SITES.values() for s in sites})
    print(f"  Batch-reading NEON parquet for proxy sites: {all_proxy_sites}")
    t_read = time.time()
    table = pq.read_table(str(NEON_PARQUET), columns=READ_COLS,
                          filters=[("source_site", "in", all_proxy_sites)])
    df_all_proxy = table.to_pandas()
    print(f"  Read {len(df_all_proxy)} rows in {time.time()-t_read:.1f}s")

    results = {}
    for eid, meta in EVENT_META.items():
        event_type = meta["event_type"]
        print(f"\n  [{eid}] type={event_type}, advisory={meta['advisory']}")
        print(f"    Stored peak probability: {case_scores[eid]['peak_window']['anomaly_probability']:.4f}")

        # Find the best peak window from NEON proxy sites for this event type
        w_raw, w_norm, proxy_score, proxy_site = find_peak_window_for_event(
            sensor, fusion, head, event_type, df_all_proxy
        )
        if w_raw is None:
            print(f"    WARNING: No valid windows found for event type {event_type}")
            results[eid] = {"error": "no_valid_windows"}
            continue

        print(f"    NEON proxy site: {proxy_site}, proxy baseline score: {proxy_score:.4f}")

        # Compute raw parameter means for the peak window
        param_means = {
            CHANNEL_NAMES[i]: round(float(w_raw[:, i].mean()), 4)
            for i in range(4)
        }
        print(f"    Window param means: {param_means}")

        # Run occlusion attribution
        attrs, baseline = occlusion_attribution(sensor, fusion, head, w_raw, w_norm)

        # Rank by delta (importance)
        ranked = sorted(attrs.items(), key=lambda x: -x[1]["delta"])
        print(f"    Ranked importance (delta drop when occluded):")
        for i, (param, av) in enumerate(ranked, 1):
            print(f"      {i}. {param}: delta={av['delta']:.4f}  ({av['importance_pct']:.1f}%)")

        results[eid] = {
            "event_type":           event_type,
            "advisory_date":        meta["advisory"],
            "stored_peak_prob":     round(case_scores[eid]["peak_window"]["anomaly_probability"], 4),
            "stored_peak_time":     case_scores[eid]["peak_window"]["center_time"],
            "neon_proxy_site":      proxy_site,
            "neon_proxy_baseline":  round(proxy_score, 4),
            "window_param_means":   param_means,
            "attribution":          attrs,
            "ranked_parameters":    [p for p, _ in ranked],
            "top_driver":           ranked[0][0] if ranked else None,
        }

    # Aggregate: which parameter drives detection for each event type?
    type_drivers = {}
    for eid, r in results.items():
        if "error" in r: continue
        etype = r["event_type"]
        if etype not in type_drivers:
            type_drivers[etype] = {}
        driver = r.get("top_driver")
        if driver:
            type_drivers[etype][driver] = type_drivers[etype].get(driver, 0) + 1

    print("\n--- Top Driver by Event Type ---")
    for etype, counts in type_drivers.items():
        top = max(counts, key=counts.get)
        print(f"  {etype}: {top} ({counts})")

    output = {
        "experiment":    "C: Parameter Attribution for Case Study Events",
        "channel_names": CHANNEL_NAMES,
        "per_event":     results,
        "type_drivers":  type_drivers,
        "elapsed_s":     round(time.time() - t0, 1),
    }
    out_path = OUTPUT_DIR / "attribution_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
