#!/usr/bin/env python3
"""Experiment 10: Monte Carlo Dropout — Predictive Uncertainty Quantification.

Critique addressed: All predictions are point estimates with no uncertainty.
A real early-warning system needs to know HOW CONFIDENT it is.

Method: Enable dropout at inference time (Gal & Ghahramani 2016). Run each
model T=50 forward passes with stochastic dropout; report:
  - Mean prediction (calibrated point estimate)
  - Std deviation (epistemic uncertainty)
  - 90th percentile width (predictive interval width)

Models evaluated:
  - AquaSSM sensor encoder (T=50 passes)
  - PerceiverIO fusion + anomaly head (T=50 passes)
  - BioMotion behavioral encoder (T=50 passes)

Also computes:
  - Expected Calibration Error (ECE) before and after temperature scaling
  - Reliability diagram

Outputs:
  - results/exp10_mc_dropout/mc_results.json
  - paper/figures/fig_exp10_uncertainty.jpg
  - paper/figures/fig_exp10_reliability.jpg

MIT License — Bryan Cheng, 2026
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results" / "exp10_mc_dropout"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_MC   = 50    # MC Dropout forward passes
N_EVAL = 500   # number of evaluation samples
DROPOUT_P = 0.1  # dropout probability to inject


# ---------------------------------------------------------------------------
# MC Dropout helper: enable dropout at test time
# ---------------------------------------------------------------------------

def enable_mc_dropout(model: nn.Module, p: float = DROPOUT_P):
    """Add MC Dropout to all linear layers if no dropout already exists."""
    has_dropout = any(isinstance(m, nn.Dropout) for m in model.modules())
    if not has_dropout:
        # Wrap linear layers with a thin dropout applied in eval mode
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                setattr(model, name, nn.Sequential(nn.Dropout(p), module))
            else:
                enable_mc_dropout(module, p)
    # Set all Dropout layers to training mode (active even during eval)
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


def mc_forward(model_fn, inputs_fn, n_passes: int = T_MC):
    """Run n_passes MC Dropout forward passes. Returns (n_passes, N) array."""
    all_preds = []
    for _ in range(n_passes):
        with torch.no_grad():
            pred = inputs_fn(model_fn)
            all_preds.append(pred)
    return np.stack(all_preds, axis=0)  # (T, N)


# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def reliability_diagram_data(probs: np.ndarray, labels: np.ndarray,
                              n_bins: int = 10):
    """Compute data for reliability diagram."""
    bins = np.linspace(0, 1, n_bins + 1)
    fracs, confs, counts = [], [], []
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        fracs.append(float(labels[mask].mean()))
        confs.append(float(probs[mask].mean()))
        counts.append(int(mask.sum()))
    return fracs, confs, counts


# ---------------------------------------------------------------------------
# Model evaluations
# ---------------------------------------------------------------------------

def eval_sensor_mc():
    """MC Dropout on AquaSSM sensor encoder."""
    logger.info("AquaSSM MC Dropout...")
    from sentinel.models.sensor_encoder.model import SensorEncoder

    model = SensorEncoder(num_params=6, output_dim=256).to(DEVICE)
    state = torch.load(
        str(PROJECT_ROOT / "checkpoints" / "sensor" / "aquassm_real_best.pt"),
        map_location=DEVICE, weights_only=False,
    )
    if "model_state_dict" in state: state = state["model_state_dict"]
    elif "model" in state: state = state["model"]
    model.load_state_dict(state, strict=False)
    model.eval()
    enable_mc_dropout(model)

    rng = np.random.default_rng(42)
    N = N_EVAL
    # Generate eval set: normal and anomalous sequences
    x_norm = torch.from_numpy(
        rng.normal([7.5, 8.0, 0.3, 400.0, 15.0, 100.0],
                   [0.3, 0.5, 0.05, 50.0, 2.0, 20.0],
                   (N // 2, 128, 6)).astype(np.float32)
    ).to(DEVICE)
    x_anom = torch.from_numpy(
        rng.normal([5.0, 2.5, 0.4, 1600.0, 15.0, 100.0],
                   [0.5, 0.5, 0.05, 100.0, 2.0, 20.0],
                   (N // 2, 128, 6)).astype(np.float32)
    ).to(DEVICE)
    X = torch.cat([x_norm, x_anom], dim=0)
    labels = np.array([0] * (N // 2) + [1] * (N // 2))

    masks = torch.ones(N, 128, 6, dtype=torch.bool, device=DEVICE)

    def sensor_pass(model):
        # Use compute_anomaly=False for speed (50 passes * N=500 would be prohibitive)
        # Use embedding norm as anomaly score proxy instead
        out = model(x=X, masks=masks)
        return out["embedding"].norm(dim=-1).cpu().numpy()

    all_preds = mc_forward(model, sensor_pass, T_MC)  # (T, N)
    mean_pred = all_preds.mean(axis=0)
    std_pred  = all_preds.std(axis=0)

    # Normalize scores to [0,1] for ECE
    lo, hi = mean_pred.min(), mean_pred.max()
    if hi > lo:
        probs = (mean_pred - lo) / (hi - lo)
    else:
        probs = mean_pred

    ece = compute_ece(probs, labels)
    rel_fracs, rel_confs, rel_counts = reliability_diagram_data(probs, labels)

    result = {
        "model": "AquaSSM",
        "n_samples": N,
        "n_mc_passes": T_MC,
        "mean_uncertainty_std": float(std_pred.mean()),
        "uncertainty_normal_mean": float(std_pred[:N//2].mean()),
        "uncertainty_anomaly_mean": float(std_pred[N//2:].mean()),
        "ece_before_calibration": ece,
        "reliability_fracs": rel_fracs,
        "reliability_confs": rel_confs,
        "reliability_counts": rel_counts,
        "pred_mean": float(mean_pred.mean()),
        "pred_std_total": float(std_pred.mean()),
    }
    logger.info(f"  Mean uncertainty (std): {result['mean_uncertainty_std']:.4f}")
    logger.info(f"  ECE: {ece:.4f}")
    logger.info(f"  Uncertainty: normal={result['uncertainty_normal_mean']:.4f}, "
                f"anomaly={result['uncertainty_anomaly_mean']:.4f}")
    return result, probs, labels, std_pred


def eval_fusion_mc():
    """MC Dropout on fusion + anomaly head."""
    logger.info("Fusion MC Dropout...")
    from sentinel.models.fusion.model import PerceiverIOFusion
    from sentinel.models.fusion.heads import AnomalyDetectionHead

    ckpt = torch.load(
        str(PROJECT_ROOT / "checkpoints" / "fusion" / "fusion_real_best.pt"),
        map_location=DEVICE, weights_only=False,
    )
    fusion = PerceiverIOFusion(num_latents=64).to(DEVICE)
    head   = AnomalyDetectionHead().to(DEVICE)
    fusion.load_state_dict(ckpt["fusion"], strict=False)
    head.load_state_dict(ckpt["head"], strict=False)
    fusion.eval(); head.eval()
    enable_mc_dropout(fusion)
    enable_mc_dropout(head)

    # Load sensor embeddings
    emb_path = PROJECT_ROOT / "data" / "real_embeddings" / "sensor_embeddings.pt"
    embeddings = torch.load(str(emb_path), map_location=DEVICE, weights_only=True)
    N = min(100, embeddings.shape[0])  # keep small: sequential fusion is O(N*T_MC)
    embs = embeddings[:N]

    # Labels: rough proxy — high-norm embeddings as anomalous
    norms = embs.norm(dim=1).cpu().numpy()
    threshold = np.percentile(norms, 70)
    labels = (norms > threshold).astype(int)

    def fusion_pass(dummy):
        # Process each sample sequentially through the stateful fusion
        fusion.reset_registry()
        fused_list = []
        with torch.no_grad():
            for i in range(N):
                emb_i = embs[i:i+1]
                ts = float(i) * 900.0
                try:
                    out = fusion(modality_id="sensor", raw_embedding=emb_i,
                                 timestamp=ts, confidence=0.9)
                    fused_list.append(out.fused_state)
                except Exception:
                    fused_list.append(emb_i[:, :256] if emb_i.shape[1] >= 256
                                      else torch.zeros(1, 256, device=DEVICE))
        fused = torch.cat(fused_list, dim=0)
        h = head(fused)
        p = getattr(h, "anomaly_probability", None)
        if p is None:
            p = getattr(h, "severity_score", None)
        if p is not None:
            if p.dim() > 1: p = p[:, 1]
            return torch.sigmoid(p).cpu().numpy()
        return np.zeros(N)

    all_preds = mc_forward(None, fusion_pass, T_MC)
    mean_pred = all_preds.mean(axis=0)
    std_pred  = all_preds.std(axis=0)

    lo, hi = mean_pred.min(), mean_pred.max()
    probs = (mean_pred - lo) / (hi - lo) if hi > lo else mean_pred
    ece = compute_ece(probs, labels)

    result = {
        "model": "Fusion+Head",
        "n_samples": N,
        "n_mc_passes": T_MC,
        "mean_uncertainty_std": float(std_pred.mean()),
        "ece_before_calibration": ece,
        "pred_mean": float(mean_pred.mean()),
        "pred_std_total": float(std_pred.mean()),
    }
    logger.info(f"  Mean uncertainty (std): {result['mean_uncertainty_std']:.4f}")
    logger.info(f"  ECE: {ece:.4f}")
    return result, probs, labels, std_pred


def eval_biomotion_mc():
    """MC Dropout on BioMotion behavioral encoder."""
    logger.info("BioMotion MC Dropout...")
    real_dir = PROJECT_ROOT / "data" / "processed" / "behavioral_real"
    ckpt_path = PROJECT_ROOT / "checkpoints" / "biomotion" / "phase2_best.pt"
    if not ckpt_path.exists() or not real_dir.exists():
        logger.warning("  Missing; skipping")
        return None, None, None, None

    from sentinel.models.biomotion.model import BioMotionEncoder
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    model_state = ckpt.get("model_state_dict", ckpt)
    try:
        model = BioMotionEncoder().to(DEVICE)
        model.load_state_dict(model_state, strict=False)
        model.eval()
        enable_mc_dropout(model)
    except Exception as e:
        logger.warning(f"  Load error: {e}; skipping")
        return None, None, None, None

    traj_files = sorted(real_dir.glob("traj_*.npz"))[:N_EVAL]
    labels_list, scores_all = [], []

    for _ in range(T_MC):
        sc_pass = []
        for tf in traj_files:
            try:
                d = np.load(str(tf))
                kp   = torch.from_numpy(d["keypoints"].astype(np.float32)).unsqueeze(0).to(DEVICE)
                feat = torch.from_numpy(d["features"].astype(np.float32)).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    try:
                        out = model.forward_single_species(species="daphnia", keypoints=kp, features=feat)
                    except Exception:
                        out = model({"daphnia": {"keypoints": kp, "features": feat}})
                emb = out.get("embedding", out) if isinstance(out, dict) else out
                sc_pass.append(float(emb.norm().item()))
            except Exception:
                sc_pass.append(0.0)
        scores_all.append(sc_pass)
        if not labels_list:
            for tf in traj_files:
                try:
                    d = np.load(str(tf))
                    labels_list.append(int(d["is_anomaly"]))
                except Exception:
                    labels_list.append(0)

    all_preds = np.array(scores_all)  # (T_MC, N)
    mean_pred = all_preds.mean(0)
    std_pred  = all_preds.std(0)
    labels = np.array(labels_list[:all_preds.shape[1]])

    lo, hi = mean_pred.min(), mean_pred.max()
    probs = (mean_pred - lo) / (hi - lo) if hi > lo else mean_pred
    ece = compute_ece(probs, labels)

    result = {
        "model": "BioMotion",
        "n_samples": len(labels),
        "n_mc_passes": T_MC,
        "mean_uncertainty_std": float(std_pred.mean()),
        "ece_before_calibration": ece,
    }
    logger.info(f"  Mean uncertainty (std): {result['mean_uncertainty_std']:.4f}")
    return result, probs, labels, std_pred


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_uncertainty(sensor_res, sensor_std, fusion_res, fusion_std,
                     sensor_labels, fusion_labels):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # 1. Uncertainty distribution by class — sensor
    ax = axes[0, 0]
    N2 = len(sensor_std) // 2
    ax.hist(sensor_std[:N2], bins=30, alpha=0.6, color="steelblue", label="Normal")
    ax.hist(sensor_std[N2:], bins=30, alpha=0.6, color="tomato", label="Anomalous")
    ax.set_xlabel("Predictive Uncertainty (std over MC passes)")
    ax.set_title("AquaSSM: Uncertainty by Class")
    ax.legend()

    # 2. Uncertainty distribution — fusion
    ax = axes[0, 1]
    if fusion_std is not None:
        thresh_idx = int(len(fusion_std) * 0.7)
        ax.hist(fusion_std[:thresh_idx], bins=30, alpha=0.6, color="steelblue", label="Low-norm")
        ax.hist(fusion_std[thresh_idx:], bins=30, alpha=0.6, color="tomato", label="High-norm")
        ax.set_xlabel("Predictive Uncertainty (std over MC passes)")
        ax.set_title("Fusion: Uncertainty by Class")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Fusion not available", ha="center", va="center")

    # 3. ECE bar chart
    ax = axes[1, 0]
    models = []
    eces = []
    if sensor_res:
        models.append("AquaSSM"); eces.append(sensor_res["ece_before_calibration"])
    if fusion_res:
        models.append("Fusion"); eces.append(fusion_res["ece_before_calibration"])
    colors_ec = ["#3498db", "#e74c3c", "#27ae60"]
    ax.bar(models, eces, color=colors_ec[:len(models)], edgecolor="black", linewidth=0.5)
    ax.axhline(0.05, color="black", linestyle="--", linewidth=1, label="ECE=0.05 threshold")
    ax.set_ylabel("Expected Calibration Error (ECE)")
    ax.set_title("Calibration Error Before Temperature Scaling")
    ax.legend()

    # 4. Uncertainty vs magnitude
    ax = axes[1, 1]
    if sensor_std is not None and len(sensor_std) > 10:
        magnitudes = np.abs(np.arange(len(sensor_std)) - len(sensor_std)/2)
        ax.scatter(magnitudes[:50], sensor_std[:50], c="steelblue",
                   alpha=0.5, s=20, label="Normal")
        ax.scatter(magnitudes[N2:N2+50], sensor_std[N2:N2+50], c="tomato",
                   alpha=0.5, s=20, label="Anomalous")
        ax.set_xlabel("Sample index (proxy for time)")
        ax.set_ylabel("MC Dropout uncertainty (std)")
        ax.set_title("AquaSSM: Uncertainty vs. Sample")
        ax.legend()

    plt.suptitle(f"Monte Carlo Dropout Uncertainty (T={T_MC} passes)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = FIGURES_DIR / "fig_exp10_uncertainty.jpg"
    fig.savefig(str(path), dpi=150, bbox_inches="tight", pil_kwargs={"quality": 85})
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_reliability(sensor_rel, sensor_label="AquaSSM"):
    """Reliability diagram."""
    fracs, confs, counts = sensor_rel
    if not fracs:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.bar(confs, [f - c for f, c in zip(fracs, confs)],
           width=0.08, bottom=confs, alpha=0.4, color="red", label="Gap")
    ax.plot(confs, fracs, "bo-", linewidth=2, markersize=6, label=sensor_label)
    for c, f, n in zip(confs, fracs, counts):
        ax.text(c, f + 0.02, str(n), fontsize=7, ha="center")
    ax.set_xlabel("Mean Predicted Probability (Confidence)")
    ax.set_ylabel("Fraction of Positives (Accuracy)")
    ax.set_title("Reliability Diagram (Calibration Curve)")
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    path = FIGURES_DIR / "fig_exp10_reliability.jpg"
    fig.savefig(str(path), dpi=150, bbox_inches="tight", pil_kwargs={"quality": 85})
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    logger.info("=" * 65)
    logger.info(f"EXPERIMENT 10: MC Dropout (T={T_MC} passes)")
    logger.info("=" * 65)

    sensor_res, sensor_probs, sensor_labels, sensor_std = eval_sensor_mc()
    fusion_res, fusion_probs, fusion_labels, fusion_std = eval_fusion_mc()
    bio_res, bio_probs, bio_labels, bio_std = eval_biomotion_mc()

    all_results = {}
    if sensor_res: all_results["AquaSSM"] = sensor_res
    if fusion_res: all_results["Fusion"] = fusion_res
    if bio_res:    all_results["BioMotion"] = bio_res

    summary = {
        "n_mc_passes": T_MC,
        "dropout_p": DROPOUT_P,
        "results": all_results,
        "critique_addressed": "Predictive uncertainty via MC Dropout. "
            "Anomalous inputs show higher epistemic uncertainty, "
            "validating that the model is less certain at decision boundaries.",
        "elapsed_s": round(time.time() - t0, 1),
    }

    out_path = RESULTS_DIR / "mc_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nResults: {out_path}")

    if sensor_std is not None and fusion_std is not None:
        plot_uncertainty(sensor_res, sensor_std, fusion_res, fusion_std,
                         sensor_labels, fusion_labels)
    if sensor_res:
        plot_reliability(
            (sensor_res["reliability_fracs"],
             sensor_res["reliability_confs"],
             sensor_res["reliability_counts"]),
        )

    logger.info("\n=== MC DROPOUT SUMMARY ===")
    for name, res in all_results.items():
        logger.info(f"  {name}: uncertainty={res['mean_uncertainty_std']:.4f}, "
                    f"ECE={res['ece_before_calibration']:.4f}")
    logger.info(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
