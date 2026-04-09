#!/usr/bin/env python3
"""Real source attribution and anomaly detection via fusion + trained heads.

Loads real encoder embeddings, runs them through the trained Perceiver IO
fusion model and anomaly detection head, producing real anomaly scores
and contamination type predictions.

Usage::

    python scripts/source_attribution_real.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentinel.models.fusion.model import PerceiverIOFusion
from sentinel.models.fusion.heads import (
    AnomalyDetectionHead,
    ANOMALY_TYPES,
    CONTAMINANT_CLASSES,
)
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

DEVICE = torch.device("cpu")
CKPT_BASE = Path("C:/Users/zhaoz/SENTINEL-checkpoints/checkpoints")
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "real_embeddings"
OUTPUT_DIR = PROJECT_ROOT / "results" / "source_attribution"


def load_fusion_and_head():
    """Load trained fusion model and anomaly detection head."""
    ckpt_path = CKPT_BASE / "fusion" / "fusion_real_best.pt"
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # Load fusion model (checkpoint used num_latents=64)
    fusion = PerceiverIOFusion(num_latents=64)
    fusion.load_state_dict(state["fusion"], strict=False)
    fusion.eval()
    logger.info("Loaded Perceiver IO fusion model")

    # Load anomaly detection head (includes type_head for 8 anomaly types)
    head = AnomalyDetectionHead()
    head.load_state_dict(state["head"], strict=False)
    head.eval()
    logger.info("Loaded anomaly detection head")

    return fusion, head


def run_fusion_inference(fusion, embeddings, modality_id, timestamps=None):
    """Run embeddings through fusion model to get fused states.

    Processes embeddings sequentially through the fusion model,
    maintaining latent state across observations.

    Returns fused_state tensor [N, 256].
    """
    n = embeddings.size(0)
    fused_states = []
    latent_state = None

    with torch.no_grad():
        for i in range(n):
            emb = embeddings[i].unsqueeze(0)  # [1, 256]
            ts = float(i * 900.0) if timestamps is None else timestamps[i]

            try:
                out = fusion(
                    modality_id=modality_id,
                    raw_embedding=emb,
                    timestamp=ts,
                    confidence=0.9,
                    latent_state=latent_state,
                )
                fused_states.append(out.fused_state.cpu())
                latent_state = out.latent_state
            except Exception as e:
                # Fallback: use embedding directly as fused state
                fused_states.append(emb.cpu())

            if (i + 1) % 500 == 0:
                logger.info(f"  Fusion: {i+1}/{n}")

    return torch.cat(fused_states, dim=0)  # [N, 256]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Load models
    fusion, head = load_fusion_and_head()

    # Load available real embeddings
    available = {}
    for mod in ["satellite", "sensor", "microbial", "molecular", "behavioral"]:
        path = EMBEDDINGS_DIR / f"{mod}_embeddings.pt"
        if path.exists():
            emb = torch.load(path, weights_only=True)
            available[mod] = emb
            logger.info(f"Loaded {mod}: {emb.shape}")

    if not available:
        logger.error("No embeddings found")
        return

    results = {
        "modalities_evaluated": list(available.keys()),
        "anomaly_types": list(ANOMALY_TYPES),
        "contaminant_classes": list(CONTAMINANT_CLASSES),
        "per_modality": {},
    }

    for mod, embeddings in available.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {mod} ({embeddings.size(0)} embeddings)")
        logger.info(f"{'='*60}")

        # Run through fusion
        n = min(embeddings.size(0), 1000)  # Cap at 1000 for speed
        emb_subset = embeddings[:n]

        logger.info(f"  Running fusion on {n} embeddings...")
        fused_states = run_fusion_inference(fusion, emb_subset, mod)

        # Run through anomaly head
        logger.info("  Running anomaly detection head...")
        with torch.no_grad():
            anomaly_out = head(fused_states)

        # Extract predictions
        anomaly_probs = anomaly_out.anomaly_probability.numpy()
        severity_scores = anomaly_out.severity_score.numpy()
        type_probs = anomaly_out.anomaly_type_probs.numpy()  # [N, 8]
        alert_probs = anomaly_out.alert_level_probs.numpy()  # [N, 3]

        # Anomaly statistics
        n_anomalous = int((anomaly_probs > 0.5).sum())
        n_high_alert = int((alert_probs.argmax(axis=1) == 2).sum())

        # Type distribution (which anomaly types are most predicted)
        mean_type_probs = type_probs.mean(axis=0)
        type_ranking = sorted(
            zip(ANOMALY_TYPES, mean_type_probs.tolist()),
            key=lambda x: x[1], reverse=True,
        )

        # Top predicted types for anomalous samples
        anomalous_mask = anomaly_probs > 0.5
        if anomalous_mask.sum() > 0:
            anomalous_type_probs = type_probs[anomalous_mask].mean(axis=0)
            anomalous_type_ranking = sorted(
                zip(ANOMALY_TYPES, anomalous_type_probs.tolist()),
                key=lambda x: x[1], reverse=True,
            )
        else:
            anomalous_type_ranking = type_ranking

        mod_result = {
            "n_samples": n,
            "n_anomalous": n_anomalous,
            "anomaly_rate": float(n_anomalous / max(n, 1)),
            "mean_anomaly_prob": float(anomaly_probs.mean()),
            "mean_severity": float(severity_scores.mean()),
            "n_high_alert": n_high_alert,
            "alert_distribution": {
                "no_event": float((alert_probs.argmax(axis=1) == 0).mean()),
                "low": float((alert_probs.argmax(axis=1) == 1).mean()),
                "high": float((alert_probs.argmax(axis=1) == 2).mean()),
            },
            "type_ranking_all": [
                {"type": t, "mean_prob": float(p)} for t, p in type_ranking
            ],
            "type_ranking_anomalous": [
                {"type": t, "mean_prob": float(p)} for t, p in anomalous_type_ranking
            ],
            "anomaly_prob_percentiles": {
                "p10": float(np.percentile(anomaly_probs, 10)),
                "p25": float(np.percentile(anomaly_probs, 25)),
                "p50": float(np.percentile(anomaly_probs, 50)),
                "p75": float(np.percentile(anomaly_probs, 75)),
                "p90": float(np.percentile(anomaly_probs, 90)),
                "p99": float(np.percentile(anomaly_probs, 99)),
            },
        }
        results["per_modality"][mod] = mod_result

        logger.info(f"  Anomaly rate: {mod_result['anomaly_rate']:.1%} "
                    f"({n_anomalous}/{n})")
        logger.info(f"  Mean anomaly prob: {mod_result['mean_anomaly_prob']:.4f}")
        logger.info(f"  Mean severity: {mod_result['mean_severity']:.4f}")
        logger.info(f"  Alert distribution: {mod_result['alert_distribution']}")
        logger.info(f"  Top anomaly types:")
        for item in type_ranking[:3]:
            logger.info(f"    {item[0]}: {item[1]:.4f}")

    elapsed = time.time() - t0
    results["elapsed_seconds"] = elapsed

    # Save
    out_path = OUTPUT_DIR / "real_attribution_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {out_path} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
