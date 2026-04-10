#!/usr/bin/env python3
"""Experiment 5: Explainability — attention weights and perturbation importance.

Extracts attention weights from the Perceiver IO fusion model and
measures per-modality feature importance via perturbation analysis.

Produces:
  - Temporal attention heatmap (fig_exp5_attention.jpg)
  - Perturbation importance bar chart (fig_exp5_importance.jpg)

Usage::

    python scripts/exp5_explainability.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sentinel.models.fusion.model import PerceiverIOFusion, FusionOutput
from sentinel.models.fusion.heads import AnomalyDetectionHead
from sentinel.models.fusion.embedding_registry import MODALITY_IDS
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEVICE = torch.device("cpu")
CKPT_BASE = Path("C:/Users/zhaoz/SENTINEL-checkpoints/checkpoints")
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "real_embeddings"
RESULTS_DIR = PROJECT_ROOT / "results" / "exp5_explainability"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"

# Number of timesteps to process for temporal attention
N_TEMPORAL_STEPS = 200
# Number of samples for perturbation analysis
N_PERTURBATION = 100
# Simulated time step between observations (seconds)
TIME_STEP = 900.0  # 15 minutes


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

def load_fusion_and_head():
    """Load trained Perceiver IO fusion model and anomaly detection head."""
    ckpt_path = CKPT_BASE / "fusion" / "fusion_real_best.pt"
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    fusion = PerceiverIOFusion(num_latents=64)
    fusion.load_state_dict(state["fusion"], strict=False)
    fusion.eval()
    logger.info("Loaded Perceiver IO fusion model")

    head = AnomalyDetectionHead()
    head.load_state_dict(state["head"], strict=False)
    head.eval()
    logger.info("Loaded anomaly detection head")

    return fusion, head


# ---------------------------------------------------------------------------
# Load embeddings
# ---------------------------------------------------------------------------

def load_embeddings():
    """Load real embeddings for sensor, satellite, and behavioral modalities."""
    sensor_emb = torch.load(
        str(EMBEDDINGS_DIR / "sensor_embeddings.pt"),
        map_location="cpu", weights_only=True,
    )
    satellite_emb = torch.load(
        str(EMBEDDINGS_DIR / "satellite_embeddings.pt"),
        map_location="cpu", weights_only=True,
    )
    behavioral_emb = torch.load(
        str(EMBEDDINGS_DIR / "behavioral_embeddings.pt"),
        map_location="cpu", weights_only=True,
    )

    logger.info(f"Loaded sensor embeddings:     {sensor_emb.shape}")
    logger.info(f"Loaded satellite embeddings:  {satellite_emb.shape}")
    logger.info(f"Loaded behavioral embeddings: {behavioral_emb.shape}")

    return sensor_emb, satellite_emb, behavioral_emb


# ---------------------------------------------------------------------------
# 1. Temporal attention extraction
# ---------------------------------------------------------------------------

def extract_temporal_attention(fusion, sensor_emb, satellite_emb, behavioral_emb):
    """Process embeddings sequentially and collect per-step attention weights.

    Creates a mixed-modality observation stream:
      - Sensor every step (primary)
      - Satellite every 10 steps (lower cadence)
      - Behavioral every 5 steps

    Returns:
        attn_matrix: np.ndarray [n_steps, n_modalities]
            Per-modality attention aggregated over heads and latents.
        modality_labels: list of modality names matching columns.
        step_modalities: list of which modality was the trigger at each step.
    """
    modality_labels = list(MODALITY_IDS)
    n_modalities = len(modality_labels)
    n_steps = min(N_TEMPORAL_STEPS, sensor_emb.shape[0])

    # Attention matrix: rows=timesteps, cols=modalities
    attn_matrix = np.zeros((n_steps, n_modalities))
    step_modalities = []

    # Fallback: norm-based tracking if attn_weights are None
    use_norm_fallback = False

    latent_state = None
    fusion.reset_registry()

    sat_idx = 0
    beh_idx = 0

    with torch.no_grad():
        for t in range(n_steps):
            ts = float(t) * TIME_STEP

            # Determine which modality triggers this step
            if t % 10 == 5 and sat_idx < satellite_emb.shape[0]:
                modality_id = "satellite"
                emb = satellite_emb[sat_idx].unsqueeze(0)
                sat_idx += 1
            elif t % 5 == 3 and beh_idx < behavioral_emb.shape[0]:
                modality_id = "behavioral"
                emb = behavioral_emb[beh_idx].unsqueeze(0)
                beh_idx += 1
            else:
                modality_id = "sensor"
                emb = sensor_emb[min(t, sensor_emb.shape[0] - 1)].unsqueeze(0)

            step_modalities.append(modality_id)

            try:
                out = fusion(
                    modality_id=modality_id,
                    raw_embedding=emb,
                    timestamp=ts,
                    confidence=0.9,
                    latent_state=latent_state,
                )
                latent_state = out.latent_state

                # Extract attention weights
                if out.attn_weights is not None and not use_norm_fallback:
                    # attn_weights shape: [B, H, N, K] where K = num modalities
                    aw = out.attn_weights  # [1, 8, N, K]
                    # Mean over heads and latent positions -> [K]
                    aw_agg = aw.mean(dim=(0, 1, 2)).cpu().numpy()

                    # aw_agg may have K != n_modalities if not all registered
                    if aw_agg.shape[0] == n_modalities:
                        attn_matrix[t, :] = aw_agg
                    elif aw_agg.shape[0] < n_modalities:
                        # Only some modalities have been seen so far
                        attn_matrix[t, :aw_agg.shape[0]] = aw_agg
                    else:
                        # More values than expected — take first n_modalities
                        attn_matrix[t, :] = aw_agg[:n_modalities]
                else:
                    # Fallback: use decay weights as attention proxy
                    use_norm_fallback = True
                    for j, mid in enumerate(modality_labels):
                        dw = out.decay_weights.get(mid, torch.tensor(0.0))
                        attn_matrix[t, j] = float(dw)

            except Exception as e:
                logger.warning(f"Step {t}: fusion error: {e}")
                # Leave zeros for this timestep

            if (t + 1) % 50 == 0:
                logger.info(f"  Attention extraction: {t+1}/{n_steps}")

    # Normalize each row so it sums to 1 (for visualization)
    row_sums = attn_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    attn_matrix_norm = attn_matrix / row_sums

    fallback_note = " (decay-weight fallback)" if use_norm_fallback else ""
    logger.info(f"Extracted attention for {n_steps} steps{fallback_note}")

    return attn_matrix_norm, modality_labels, step_modalities


# ---------------------------------------------------------------------------
# 2. Perturbation importance
# ---------------------------------------------------------------------------

def compute_perturbation_importance(fusion, head, sensor_emb, satellite_emb,
                                     behavioral_emb):
    """Measure each modality's importance via zero-ablation.

    For each modality, replace its embeddings with zeros and measure the
    change in anomaly probability relative to the full-modality baseline.

    Returns:
        importance: dict mapping modality name -> mean |delta| in anomaly_prob
        baseline_probs: np.ndarray of baseline anomaly probabilities
    """
    n_samples = min(N_PERTURBATION, sensor_emb.shape[0])

    # Map modality -> embeddings
    modality_embs = {
        "sensor": sensor_emb[:n_samples],
        "satellite": satellite_emb[:min(n_samples, satellite_emb.shape[0])],
        "behavioral": behavioral_emb[:min(n_samples, behavioral_emb.shape[0])],
    }

    # Modalities to test (only those with real embeddings)
    test_modalities = ["sensor", "satellite", "behavioral"]

    def run_pipeline(embs_override=None, zero_modality=None):
        """Run fusion + head and return anomaly probabilities.

        Args:
            embs_override: dict of modality -> embeddings to use
            zero_modality: if set, replace this modality's embeddings with zeros
        """
        embs = dict(modality_embs)
        if embs_override:
            embs.update(embs_override)

        if zero_modality and zero_modality in embs:
            orig = embs[zero_modality]
            embs[zero_modality] = torch.zeros_like(orig)

        # Process a sequence of interleaved observations
        fusion.reset_registry()
        latent_state = None
        probs = []

        with torch.no_grad():
            for i in range(n_samples):
                ts = float(i) * TIME_STEP

                # Feed sensor
                sensor_e = embs["sensor"][min(i, embs["sensor"].shape[0] - 1)]
                out = fusion(
                    modality_id="sensor",
                    raw_embedding=sensor_e.unsqueeze(0),
                    timestamp=ts,
                    confidence=0.9,
                    latent_state=latent_state,
                )
                latent_state = out.latent_state

                # Feed satellite every 10 steps
                if i % 10 == 5 and i // 10 < embs["satellite"].shape[0]:
                    sat_e = embs["satellite"][i // 10]
                    out = fusion(
                        modality_id="satellite",
                        raw_embedding=sat_e.unsqueeze(0),
                        timestamp=ts + 1.0,
                        confidence=0.85,
                        latent_state=latent_state,
                    )
                    latent_state = out.latent_state

                # Feed behavioral every 5 steps
                if i % 5 == 3 and i // 5 < embs["behavioral"].shape[0]:
                    beh_e = embs["behavioral"][i // 5]
                    out = fusion(
                        modality_id="behavioral",
                        raw_embedding=beh_e.unsqueeze(0),
                        timestamp=ts + 2.0,
                        confidence=0.8,
                        latent_state=latent_state,
                    )
                    latent_state = out.latent_state

                # Get anomaly probability from fused state
                anomaly_out = head(out.fused_state)
                probs.append(float(anomaly_out.anomaly_probability.cpu()))

        return np.array(probs)

    # Baseline: all modalities active
    logger.info("Computing baseline anomaly probabilities...")
    baseline_probs = run_pipeline()
    logger.info(f"  Baseline mean anomaly prob: {baseline_probs.mean():.4f}")

    # Perturbation: zero each modality
    importance = {}
    for mod in test_modalities:
        logger.info(f"Computing perturbation importance for '{mod}'...")
        perturbed_probs = run_pipeline(zero_modality=mod)
        delta = np.abs(perturbed_probs - baseline_probs)
        importance[mod] = float(delta.mean())
        logger.info(f"  {mod}: mean |delta| = {importance[mod]:.6f}")

    # For modalities without real embeddings, report zero importance
    for mid in MODALITY_IDS:
        if mid not in importance:
            importance[mid] = 0.0

    return importance, baseline_probs


# ---------------------------------------------------------------------------
# 3. Figure generation
# ---------------------------------------------------------------------------

def plot_attention_heatmap(attn_matrix, modality_labels, output_path):
    """Plot temporal attention heatmap.

    Args:
        attn_matrix: [n_steps, n_modalities] normalized attention.
        modality_labels: column labels.
        output_path: path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(14, 4))

    # Downsample if too many steps for clear visualization
    n_steps = attn_matrix.shape[0]
    if n_steps > 100:
        # Use a moving average for smoother visualization
        window = max(1, n_steps // 100)
        smoothed = np.zeros((n_steps // window, attn_matrix.shape[1]))
        for i in range(smoothed.shape[0]):
            smoothed[i] = attn_matrix[i * window:(i + 1) * window].mean(axis=0)
        plot_data = smoothed
        x_label = f"Time step (x{window})"
    else:
        plot_data = attn_matrix
        x_label = "Time step"

    # Capitalize labels for display
    display_labels = [m.capitalize() for m in modality_labels]

    sns.heatmap(
        plot_data.T,
        ax=ax,
        cmap="YlOrRd",
        yticklabels=display_labels,
        xticklabels=False,
        cbar_kws={"label": "Normalized attention weight"},
        vmin=0,
        vmax=1,
    )

    # Add x-axis tick marks
    n_ticks = min(10, plot_data.shape[0])
    tick_positions = np.linspace(0, plot_data.shape[0] - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_positions + 0.5)
    ax.set_xticklabels(tick_positions, rotation=0)

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Modality", fontsize=11)
    ax.set_title("Temporal Attention Distribution Across Modalities", fontsize=13)
    plt.tight_layout()

    fig.savefig(str(output_path), dpi=150, bbox_inches="tight",
                format="jpeg", pil_kwargs={"quality": 85})
    plt.close(fig)
    logger.info(f"Saved attention heatmap: {output_path}")


def plot_perturbation_importance(importance, output_path):
    """Plot perturbation importance as a horizontal bar chart.

    Args:
        importance: dict mapping modality name -> importance score.
        output_path: path to save the figure.
    """
    # Sort by importance (descending)
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    names = [item[0].capitalize() for item in sorted_items]
    values = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    bars = ax.barh(names, values, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels
    max_val = max(values) if values else 1.0
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + max_val * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=10,
        )

    ax.set_xlabel("Mean |$\\Delta$ Anomaly Probability|", fontsize=11)
    ax.set_title("Modality Perturbation Importance\n(Zero-Ablation)", fontsize=13)
    ax.invert_yaxis()
    ax.set_xlim(0, max_val * 1.25 if max_val > 0 else 0.1)
    plt.tight_layout()

    fig.savefig(str(output_path), dpi=150, bbox_inches="tight",
                format="jpeg", pil_kwargs={"quality": 85})
    plt.close(fig)
    logger.info(f"Saved importance chart: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    logger.info("=" * 65)
    logger.info("EXPERIMENT 5: Explainability — Attention & Perturbation")
    logger.info("=" * 65)

    # Load models and data
    fusion, head = load_fusion_and_head()
    sensor_emb, satellite_emb, behavioral_emb = load_embeddings()

    # --- Part 1: Temporal attention ---
    logger.info("-" * 50)
    logger.info("Part 1: Temporal Attention Extraction")
    logger.info("-" * 50)

    attn_matrix, modality_labels, step_modalities = extract_temporal_attention(
        fusion, sensor_emb, satellite_emb, behavioral_emb,
    )

    plot_attention_heatmap(
        attn_matrix, modality_labels,
        FIGURES_DIR / "fig_exp5_attention.jpg",
    )

    # Save raw attention data
    np.savez(
        str(RESULTS_DIR / "temporal_attention.npz"),
        attention=attn_matrix,
        modality_labels=modality_labels,
        step_modalities=step_modalities,
    )

    # Summary statistics
    attn_summary = {}
    for j, mod in enumerate(modality_labels):
        col = attn_matrix[:, j]
        attn_summary[mod] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "max": float(np.max(col)),
            "min": float(np.min(col)),
        }
    logger.info("Attention summary (mean per modality):")
    for mod, stats in attn_summary.items():
        logger.info(f"  {mod:12s}: mean={stats['mean']:.4f}  std={stats['std']:.4f}")

    # --- Part 2: Perturbation importance ---
    logger.info("-" * 50)
    logger.info("Part 2: Perturbation Importance (Zero-Ablation)")
    logger.info("-" * 50)

    # Reset fusion registry before perturbation analysis
    fusion.reset_registry()
    importance, baseline_probs = compute_perturbation_importance(
        fusion, head, sensor_emb, satellite_emb, behavioral_emb,
    )

    plot_perturbation_importance(
        importance,
        FIGURES_DIR / "fig_exp5_importance.jpg",
    )

    # Save perturbation results
    perturbation_results = {
        "importance": importance,
        "baseline_anomaly_prob_mean": float(baseline_probs.mean()),
        "baseline_anomaly_prob_std": float(baseline_probs.std()),
        "n_samples": int(len(baseline_probs)),
    }

    with open(RESULTS_DIR / "perturbation_importance.json", "w") as f:
        json.dump(perturbation_results, f, indent=2)

    # --- Combined summary ---
    summary = {
        "attention_summary": attn_summary,
        "perturbation_importance": importance,
        "baseline_anomaly_prob_mean": float(baseline_probs.mean()),
        "n_temporal_steps": attn_matrix.shape[0],
        "n_perturbation_samples": int(len(baseline_probs)),
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    with open(RESULTS_DIR / "exp5_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 65)
    logger.info(f"Experiment 5 complete in {time.time() - t0:.1f}s")
    logger.info(f"Results: {RESULTS_DIR}")
    logger.info(f"Figures: {FIGURES_DIR / 'fig_exp5_*.jpg'}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
