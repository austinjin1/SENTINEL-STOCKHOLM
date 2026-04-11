#!/usr/bin/env python3
"""Exp19: BioMotion Behavioral Profile Analysis.

For 1000 ECOTOX Daphnia trajectories, computes kinematic statistics
(speed, immobility rate, angular momentum, entropy) and correlates
them with BioMotion anomaly scores.

Key questions:
  1. Does the model just detect immobility (trivial) or subtle kinematic changes?
  2. Which kinematic features are most discriminative?
  3. At what concentration/toxicant level do behavioral effects first appear?
  4. Is model confidence (MC Dropout uncertainty) linked to borderline kinematics?

Output:
  results/exp19_behavioral_profile/behavioral_results.json
  paper/figures/fig_exp19_behavioral_profiles.jpg

MIT License — Bryan Cheng, 2026
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

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH  = PROJECT_ROOT / "checkpoints" / "biomotion" / "phase2_best.pt"
DATA_DIR   = PROJECT_ROOT / "data" / "processed" / "behavioral_real"
OUTPUT_DIR = PROJECT_ROOT / "results" / "exp19_behavioral_profile"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR    = PROJECT_ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MAX_TRAJ   = 1000
MC_PASSES  = 20   # fast uncertainty estimate


# ---------------------------------------------------------------------------
# Kinematic feature extraction
# ---------------------------------------------------------------------------

def compute_kinematics(keypoints: np.ndarray) -> dict:
    """Extract kinematic statistics from a trajectory.

    Args:
        keypoints: (T, N_org, 2) array of (x, y) positions per frame.

    Returns:
        dict of scalar kinematic features.
    """
    T, N_org, _ = keypoints.shape

    # Per-organism displacement between frames
    disps = np.linalg.norm(np.diff(keypoints, axis=0), axis=-1)  # (T-1, N_org)

    # Mean speed across organisms and time
    mean_speed = float(disps.mean())
    max_speed  = float(disps.max())

    # Immobility: fraction of (frame, organism) pairs with near-zero displacement
    immo_thresh = 0.005  # 0.5% of image width
    immobility_rate = float((disps < immo_thresh).mean())

    # Speed variability (coefficient of variation)
    speed_cv = float(disps.std() / (disps.mean() + 1e-8))

    # Angular direction change (turns per frame)
    if T >= 3:
        vel = np.diff(keypoints, axis=0)  # (T-1, N_org, 2)
        # Dot product of consecutive velocity vectors → cos(angle)
        v1 = vel[:-1]; v2 = vel[1:]
        dot = (v1 * v2).sum(axis=-1)  # (T-2, N_org)
        n1  = np.linalg.norm(v1, axis=-1) + 1e-8
        n2  = np.linalg.norm(v2, axis=-1) + 1e-8
        cos_theta = np.clip(dot / (n1 * n2), -1, 1)
        mean_turn = float(np.arccos(cos_theta).mean())  # mean turning angle (rad)
    else:
        mean_turn = 0.0

    # Spatial spread: std of positions over time (normalized to [0,1] frame)
    centroid = keypoints.mean(axis=1, keepdims=True)  # (T, 1, 2)
    spatial_spread = float(np.linalg.norm(keypoints - centroid, axis=-1).std())

    # Speed entropy (histogram-based, 10 bins)
    speeds_flat = disps.flatten()
    hist, _ = np.histogram(speeds_flat, bins=10, range=(0, speeds_flat.max() + 1e-8), density=True)
    hist = hist + 1e-10
    hist /= hist.sum()
    speed_entropy = float(-np.sum(hist * np.log(hist)))

    # Organism clustering: mean pairwise distance between organisms per frame
    if N_org > 1:
        # Pick a subsample of frames for efficiency
        sample_frames = keypoints[::max(1, T // 20)]  # ~20 frames
        pairwise_dists = []
        for frame in sample_frames:
            for i in range(N_org):
                for j in range(i+1, N_org):
                    pairwise_dists.append(np.linalg.norm(frame[i] - frame[j]))
        mean_pairwise = float(np.mean(pairwise_dists)) if pairwise_dists else 0.0
    else:
        mean_pairwise = 0.0

    # "Active fraction": fraction of organisms that moved at least once
    ever_moved = (disps > immo_thresh).any(axis=0)  # (N_org,)
    active_fraction = float(ever_moved.mean())

    return {
        "mean_speed":       round(mean_speed,       5),
        "max_speed":        round(max_speed,         5),
        "immobility_rate":  round(immobility_rate,   4),
        "speed_cv":         round(speed_cv,           4),
        "mean_turn_rad":    round(mean_turn,          4),
        "spatial_spread":   round(spatial_spread,     5),
        "speed_entropy":    round(speed_entropy,      4),
        "mean_pairwise_dist": round(mean_pairwise,    5),
        "active_fraction":  round(active_fraction,    4),
    }


# ---------------------------------------------------------------------------
# BioMotion inference
# ---------------------------------------------------------------------------

def load_model():
    from sentinel.models.biomotion.model import BioMotionEncoder
    model = BioMotionEncoder().to(DEVICE)
    ckpt  = torch.load(str(CKPT_PATH), map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


def _call_model(model, kp, feat):
    """Unified model call: try forward_single_species first, then dict API."""
    try:
        out = model.forward_single_species(species="daphnia", keypoints=kp, features=feat)
    except Exception:
        out = model({"daphnia": {"keypoints": kp, "features": feat}})
    sc = out.get("anomaly_score", None) if isinstance(out, dict) else None
    if sc is None:
        emb = out.get("embedding", None) if isinstance(out, dict) else out
        sc  = emb.norm() if emb is not None else torch.tensor(0.0)
    return float(sc.item()) if sc.numel() == 1 else float(sc.squeeze().item())


@torch.no_grad()
def infer(model, keypoints: np.ndarray, features: np.ndarray) -> float:
    kp   = torch.from_numpy(keypoints.astype(np.float32)).unsqueeze(0).to(DEVICE)
    feat = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(DEVICE)
    return _call_model(model, kp, feat)


def infer_mc(model, keypoints: np.ndarray, features: np.ndarray, n_passes: int) -> tuple[float, float]:
    """MC Dropout inference: returns (mean_score, std_score)."""
    model.train()  # enable dropout
    scores = []
    kp   = torch.from_numpy(keypoints.astype(np.float32)).unsqueeze(0).to(DEVICE)
    feat = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        for _ in range(n_passes):
            scores.append(_call_model(model, kp, feat))
    model.eval()
    return float(np.mean(scores)), float(np.std(scores))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("EXP19: BioMotion Behavioral Profile Analysis")
    logger.info("=" * 60)

    model = load_model()
    logger.info(f"BioMotion loaded on {DEVICE}")

    trajs = sorted(DATA_DIR.glob("*.npz"))[:MAX_TRAJ]
    logger.info(f"Processing {len(trajs)} trajectories")

    records = []
    for i, traj_path in enumerate(trajs):
        if i % 100 == 0:
            logger.info(f"  {i}/{len(trajs)}...")
        try:
            d = np.load(traj_path, allow_pickle=True)
            kp   = d["keypoints"]   # (T, N_org, 2)
            feat = d["features"]    # (T, F)
            label = int(bool(d["is_anomaly"]))

            kinematics = compute_kinematics(kp)

            # Point estimate
            score = infer(model, kp, feat)

            # MC uncertainty estimate (every 5th trajectory for speed)
            if i % 5 == 0:
                mc_mean, mc_std = infer_mc(model, kp, feat, MC_PASSES)
            else:
                mc_mean, mc_std = score, 0.0

            records.append({
                "traj_id": traj_path.stem,
                "label": label,
                "anomaly_score": round(score, 5),
                "mc_mean": round(mc_mean, 5),
                "mc_std":  round(mc_std,  6),
                **kinematics,
            })
        except Exception as e:
            logger.warning(f"  Error on {traj_path.name}: {e}")

    if not records:
        logger.error("No records processed!")
        return

    import pandas as pd
    df = pd.DataFrame(records)

    # On GPU, anomaly_score is negative (more negative = more normal)
    # Negate for analysis: higher = more anomalous
    if df["anomaly_score"].mean() < 0:
        df["score_pos"] = -df["anomaly_score"]
        logger.info("GPU mode: negated anomaly_score for analysis")
    else:
        df["score_pos"] = df["anomaly_score"]

    n_normal   = int((df["label"] == 0).sum())
    n_anomaly  = int((df["label"] == 1).sum())
    logger.info(f"Labels: {n_normal} normal, {n_anomaly} anomaly")

    # -----------------------------------------------------------------------
    # Correlation analysis: kinematic features vs. anomaly score
    # -----------------------------------------------------------------------
    from scipy import stats as sp_stats

    kinematic_cols = ["mean_speed", "max_speed", "immobility_rate", "speed_cv",
                      "mean_turn_rad", "spatial_spread", "speed_entropy",
                      "mean_pairwise_dist", "active_fraction"]

    correlations = {}
    for col in kinematic_cols:
        try:
            rho, p = sp_stats.spearmanr(df["score_pos"], df[col])
            correlations[col] = {"spearman_rho": round(float(rho), 4),
                                 "p_value":       round(float(p),  6)}
        except Exception:
            correlations[col] = {"spearman_rho": None, "p_value": None}

    # Sort by |rho|
    sorted_corr = sorted(correlations.items(),
                         key=lambda x: abs(x[1]["spearman_rho"] or 0), reverse=True)

    # -----------------------------------------------------------------------
    # Normal vs. anomaly kinematic comparison
    # -----------------------------------------------------------------------
    normal_df  = df[df["label"] == 0]
    anomaly_df = df[df["label"] == 1]

    group_stats = {}
    for col in kinematic_cols:
        n_mean = float(normal_df[col].mean())
        a_mean = float(anomaly_df[col].mean()) if len(anomaly_df) > 0 else 0.0
        try:
            stat, p = sp_stats.mannwhitneyu(normal_df[col].dropna(),
                                             anomaly_df[col].dropna(),
                                             alternative="two-sided")
        except Exception:
            stat, p = 0.0, 1.0
        # Cohen's d
        n_std = float(normal_df[col].std()) + 1e-8
        a_std = float(anomaly_df[col].std()) if len(anomaly_df) > 0 else 1e-8
        pooled_std = np.sqrt((n_std**2 + a_std**2) / 2)
        cohens_d   = (a_mean - n_mean) / pooled_std
        group_stats[col] = {
            "normal_mean":  round(n_mean, 5),
            "anomaly_mean": round(a_mean, 5),
            "mann_whitney_p": round(float(p), 6),
            "cohens_d":     round(float(cohens_d), 4),
        }

    # -----------------------------------------------------------------------
    # Immobility threshold analysis: at what immobility rate does score jump?
    # -----------------------------------------------------------------------
    df_sorted = df.sort_values("immobility_rate")
    immo_bins = np.linspace(0, 1, 11)
    immo_analysis = []
    for lo, hi in zip(immo_bins[:-1], immo_bins[1:]):
        mask = (df_sorted["immobility_rate"] >= lo) & (df_sorted["immobility_rate"] < hi)
        sub  = df_sorted[mask]
        if len(sub) < 3:
            continue
        immo_analysis.append({
            "bin_lo": round(float(lo), 2),
            "bin_hi": round(float(hi), 2),
            "n":      int(len(sub)),
            "mean_score": round(float(sub["score_pos"].mean()), 4),
            "frac_anomaly_label": round(float(sub["label"].mean()), 3),
        })

    # -----------------------------------------------------------------------
    # Key finding: trivial vs. subtle detection
    # -----------------------------------------------------------------------
    # "Trivial" cases: immobility_rate > 0.8 (obviously stopped)
    # "Subtle" cases: immobility_rate < 0.2 (still moving but anomalous)
    trivial_mask = df["immobility_rate"] > 0.8
    subtle_mask  = (df["immobility_rate"] < 0.2) & (df["label"] == 1)

    trivial_auroc = None
    subtle_auroc  = None
    try:
        from sklearn.metrics import roc_auc_score
        if trivial_mask.sum() >= 10:
            trivial_auroc = round(float(roc_auc_score(
                df[trivial_mask]["label"], df[trivial_mask]["score_pos"])), 4)
        if subtle_mask.sum() >= 5:
            # Among truly-anomaly-labeled low-immobility trajs vs normal low-immobility
            normal_low = (df["immobility_rate"] < 0.2) & (df["label"] == 0)
            sub = df[subtle_mask | normal_low]
            if len(sub) >= 10:
                subtle_auroc = round(float(roc_auc_score(
                    sub["label"], sub["score_pos"])), 4)
    except Exception as e:
        logger.warning(f"AUROC sub-analysis failed: {e}")

    detection_modes = {
        "trivial_immobility_auroc": trivial_auroc,
        "subtle_kinematic_auroc":   subtle_auroc,
        "n_trivial":  int(trivial_mask.sum()),
        "n_subtle_anomaly": int(subtle_mask.sum()),
        "interpretation": (
            "If trivial≈subtle, model detects subtle kinematics. "
            "If trivial>>subtle, model primarily detects immobility."
        ),
    }

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    output = {
        "n_trajectories":   len(df),
        "n_normal":         n_normal,
        "n_anomaly":        n_anomaly,
        "overall_auroc_point": None,  # computed below
        "kinematic_correlations": correlations,
        "top3_kinematic_predictors": [k for k, _ in sorted_corr[:3]],
        "group_stats":      group_stats,
        "immobility_analysis": immo_analysis,
        "detection_mode_analysis": detection_modes,
        "elapsed_s":        round(time.time() - t0, 1),
    }

    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(df["label"], df["score_pos"])
        output["overall_auroc_point"] = round(float(auc), 4)
    except Exception:
        pass

    out_path = OUTPUT_DIR / "behavioral_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved: {out_path}")

    # --- Figure ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        # 1. Kinematic correlations bar chart
        ax = axes[0, 0]
        corr_vals = [correlations[c]["spearman_rho"] or 0 for c in kinematic_cols]
        colors    = ["#CC3333" if v < 0 else "#2266CC" for v in corr_vals]
        ax.barh(kinematic_cols, corr_vals, color=colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Spearman ρ with Anomaly Score")
        ax.set_title("Kinematic Correlates of BioMotion Score")
        ax.set_xlim(-0.6, 0.6)

        # 2. Score distribution by label
        ax = axes[0, 1]
        ax.hist(df[df["label"] == 0]["score_pos"], bins=40, alpha=0.6,
                color="green", label=f"Normal (n={n_normal})", density=True)
        ax.hist(df[df["label"] == 1]["score_pos"], bins=40, alpha=0.6,
                color="red", label=f"Anomaly (n={n_anomaly})", density=True)
        ax.set_xlabel("BioMotion Anomaly Score"); ax.set_ylabel("Density")
        ax.set_title(f"Score Distribution by Label\n(AUROC={output.get('overall_auroc_point','?')})")
        ax.legend()

        # 3. Immobility rate vs anomaly score (scatter)
        ax = axes[1, 0]
        colors_scatter = ["green" if l == 0 else "red" for l in df["label"]]
        ax.scatter(df["immobility_rate"], df["score_pos"],
                   c=colors_scatter, alpha=0.3, s=10)
        ax.set_xlabel("Immobility Rate"); ax.set_ylabel("Anomaly Score")
        ax.set_title("Immobility vs. Anomaly Score\n(green=normal, red=anomaly)")

        # 4. Immobility bin analysis
        ax = axes[1, 1]
        if immo_analysis:
            bin_centers = [(x["bin_lo"] + x["bin_hi"]) / 2 for x in immo_analysis]
            mean_scores = [x["mean_score"] for x in immo_analysis]
            frac_anomaly = [x["frac_anomaly_label"] for x in immo_analysis]
            ax2 = ax.twinx()
            ax.bar(bin_centers, mean_scores, width=0.09, alpha=0.6,
                   color="blue", label="Mean Anomaly Score")
            ax2.plot(bin_centers, frac_anomaly, "r-o", ms=5, lw=1.5,
                     label="Frac. Anomaly Labels")
            ax.set_xlabel("Immobility Rate Bin"); ax.set_ylabel("Mean Score", color="blue")
            ax2.set_ylabel("Fraction Anomaly", color="red")
            ax.set_title("Score vs. Immobility Rate Bins")
            ax.legend(loc="upper left", fontsize=7)
            ax2.legend(loc="upper right", fontsize=7)

        plt.suptitle("Exp19: BioMotion Behavioral Profile Analysis", fontsize=12, y=1.01)
        plt.tight_layout()
        fig_path = FIG_DIR / "fig_exp19_behavioral_profiles.jpg"
        plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {fig_path}")
    except Exception as e:
        logger.warning(f"Figure failed: {e}")

    # Console summary
    logger.info("\n=== BEHAVIORAL ANALYSIS SUMMARY ===")
    logger.info(f"Overall AUROC:  {output.get('overall_auroc_point', '?')}")
    logger.info(f"Top 3 kinematic predictors: {output['top3_kinematic_predictors']}")
    logger.info("\nKinematic correlations with anomaly score:")
    for k, v in sorted_corr:
        sig = "*" if (v["p_value"] or 1) < 0.05 else ""
        logger.info(f"  {k:<25} ρ={v['spearman_rho']:+.4f}  p={v['p_value']:.4f} {sig}")
    logger.info(f"\nDetection mode: trivial AUROC={detection_modes['trivial_immobility_auroc']}, "
                f"subtle AUROC={detection_modes['subtle_kinematic_auroc']}")
    logger.info(f"Elapsed: {output['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
