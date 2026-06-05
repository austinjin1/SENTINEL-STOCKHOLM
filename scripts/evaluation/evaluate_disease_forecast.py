#!/usr/bin/env python3
"""Evaluate Disease Outbreak Forecasting model with classification metrics.

Loads the best checkpoint of IntegratedDiseaseRisk and evaluates on the
held-out test set with:
  - Overall and per-alert-level accuracy (0/1/2/3)
  - Precision, recall, F1 for each alert level
  - AUROC for binary "any alert" (level>0 vs level=0)
  - AUROC for "high alert" (level>=2 vs level<2)
  - Per-pathogen exceedance prediction accuracy and AUROC
  - Confusion matrix for alert levels
  - Calibration analysis (reliability / ECE)

Data source: Real USGS NWIS sensor data (same spatial holdout as training).

Usage:
    /home/bcheng/.conda/envs/physiformer/bin/python3 scripts/evaluate_disease_forecast.py --gpu 1

MIT License -- Bryan Cheng, 2026
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

CKPT_DIR = PROJECT / "checkpoints" / "biology"
RESULTS_DIR = PROJECT / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SENSOR_DIR = PROJECT / "data" / "raw" / "sensor" / "full"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Epidemiological risk score computation (identical to training script)
# ---------------------------------------------------------------------------
def compute_risk_scores(temp, do_val, ph, turb, spcond, day_of_year):
    """Compute disease risk scores from real water quality measurements."""
    temp_risk = max(0, (temp - 15) / 20)
    eutrophication_proxy = 0.0
    if turb is not None and turb > 0:
        eutrophication_proxy += min(turb / 50, 1.0)
    if do_val is not None and do_val < 6:
        eutrophication_proxy += (6 - do_val) / 6
    eutrophication_proxy = min(eutrophication_proxy, 1.0)

    seasonal = max(0, np.sin((day_of_year - 80) * 2 * np.pi / 365))

    mc_risk = temp_risk * (0.3 + 0.7 * eutrophication_proxy) * (0.5 + 0.5 * seasonal)
    ana_risk = mc_risk * 0.7
    cyl_risk = max(0, (temp - 20) / 15) * eutrophication_proxy * seasonal * 0.5

    mc_conc = mc_risk * 5.0
    ana_conc = ana_risk * 2.0
    cyl_conc = cyl_risk * 3.0

    cyano = np.array([
        [mc_conc, mc_conc * 1.1],
        [ana_conc, ana_conc * 1.1],
        [cyl_conc, cyl_conc * 1.1],
    ], dtype=np.float32)

    cyano_exceed = np.array([
        [mc_conc > 1.0, mc_conc * 1.1 > 1.0],
        [ana_conc > 3.0, ana_conc * 1.1 > 3.0],
        [cyl_conc > 1.0, cyl_conc * 1.1 > 1.0],
    ], dtype=np.float32)

    salinity_proxy = spcond / 2000 if spcond is not None else 0.0
    v_temp = max(0, (temp - 20) / 15)
    v_sal = max(0, 1 - abs(salinity_proxy - 15) / 20)
    v_risk = v_temp * v_sal

    vibrio = np.array([
        np.clip(v_risk, 0, 1),
        np.clip(v_risk * 1.05, 0, 1),
        np.clip(v_risk * 0.7, 0, 1),
        np.clip(v_risk * 0.75, 0, 1),
    ], dtype=np.float32)

    naeg_risk = 1 / (1 + np.exp(-(temp - 30) / 3))
    naegleria = np.array([naeg_risk], dtype=np.float32)

    schisto_risk = 1 / (1 + np.exp(-(temp - 25) / 4)) * 0.1
    schistosomiasis = np.array([float(schisto_risk)], dtype=np.float32)

    return {
        "cyano_conc": cyano,
        "cyano_exceed": cyano_exceed,
        "vibrio": vibrio,
        "naegleria": naegleria,
        "schistosomiasis": schistosomiasis,
    }


# ---------------------------------------------------------------------------
# Dataset (identical to training script)
# ---------------------------------------------------------------------------
class DiseaseDataset(Dataset):
    def __init__(self, split: str = "train", seed: int = 42):
        super().__init__()
        self.split = split
        self._build_from_sensor_data(split, seed)

    def _build_from_sensor_data(self, split: str, seed: int):
        import pandas as pd

        sensor_files = sorted(SENSOR_DIR.glob("*.parquet")) if SENSOR_DIR.exists() else []
        if not sensor_files:
            raise FileNotFoundError(f"No USGS sensor data found in {SENSOR_DIR}.")

        split_files = {"train": [], "val": [], "test": []}
        for f in sensor_files:
            h = hashlib.sha256(f"{seed}:{f.stem}".encode()).hexdigest()
            fold = int(h[:8], 16) % 10
            if fold < 7:
                split_files["train"].append(f)
            elif fold < 9:
                split_files["val"].append(f)
            else:
                split_files["test"].append(f)

        selected = split_files[split]
        log(f"  Loading {len(selected)} sensor files for {split}...")

        embeddings = []
        days = []
        vibrio_covs_list = []
        naeg_covs_list = []
        schisto_covs_list = []
        cyano_conc_list = []
        cyano_exceed_list = []
        vibrio_risk_list = []
        naeg_prob_list = []
        schisto_prob_list = []

        for fpath in selected:
            try:
                df = pd.read_parquet(fpath)
                if "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    df = df.sort_values("datetime")
                elif df.index.name == "datetime":
                    df = df.sort_index().reset_index()
                else:
                    continue

                if "Temp" not in df.columns:
                    continue

                df = df.set_index("datetime")
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                daily = df.resample("D").mean().dropna(subset=["Temp"])

                if len(daily) < 14:
                    continue

                daily = daily[(daily["Temp"] > -50) & (daily["Temp"] < 50)]

                for start_idx in range(0, len(daily) - 7, 7):
                    window = daily.iloc[start_idx:start_idx + 7]
                    if len(window) < 3:
                        continue

                    temp = float(window["Temp"].mean())
                    do_val = float(window["DO"].mean()) if "DO" in window.columns and not window["DO"].isna().all() else None
                    ph = float(window["pH"].mean()) if "pH" in window.columns and not window["pH"].isna().all() else None
                    turb = float(window["Turb"].mean()) if "Turb" in window.columns and not window["Turb"].isna().all() else None
                    spcond = float(window["SpCond"].mean()) if "SpCond" in window.columns and not window["SpCond"].isna().all() else None

                    if temp < -10 or temp > 45:
                        continue
                    if do_val is not None and (do_val < 0 or do_val > 25):
                        do_val = None
                    if turb is not None and (turb < 0 or turb > 10000):
                        turb = None

                    doy = float(window.index[len(window) // 2].dayofyear)
                    risks = compute_risk_scores(temp, do_val, ph, turb, spcond, doy)

                    feats = []
                    for col in window.columns:
                        vals = window[col].dropna()
                        if len(vals) > 0:
                            feats.extend([vals.mean(), vals.std(), vals.min(), vals.max()])
                    if len(feats) < 4:
                        continue
                    feats = np.array(feats, dtype=np.float32)
                    if len(feats) < 256:
                        feats = np.pad(feats, (0, 256 - len(feats)))
                    emb = feats[:256]
                    emb = emb / (np.linalg.norm(emb) + 1e-8)

                    salinity_proxy = spcond / 2000 if spcond is not None else 0.0
                    chlorine_proxy = 0.0

                    embeddings.append(emb)
                    days.append(doy)
                    vibrio_covs_list.append(np.array([temp, salinity_proxy], dtype=np.float32))
                    naeg_covs_list.append(np.array([temp, chlorine_proxy], dtype=np.float32))
                    schisto_covs_list.append(np.array([temp, 35.0, 0.5], dtype=np.float32))
                    cyano_conc_list.append(risks["cyano_conc"])
                    cyano_exceed_list.append(risks["cyano_exceed"])
                    vibrio_risk_list.append(risks["vibrio"])
                    naeg_prob_list.append(risks["naegleria"])
                    schisto_prob_list.append(risks["schistosomiasis"])
            except Exception:
                continue

        if not embeddings:
            raise RuntimeError(f"No samples from {len(selected)} sensor files.")

        self.embeddings = np.stack(embeddings)
        self.day_of_year = np.array(days, dtype=np.float32)
        self.vibrio_covs = np.stack(vibrio_covs_list)
        self.naegleria_covs = np.stack(naeg_covs_list)
        self.schisto_covs = np.stack(schisto_covs_list)
        self.cyano_conc = np.stack(cyano_conc_list)
        self.cyano_exceed = np.stack(cyano_exceed_list)
        self.vibrio_risk = np.stack(vibrio_risk_list)
        self.naegleria_prob = np.stack(naeg_prob_list)
        self.schisto_prob = np.stack(schisto_prob_list)

        log(f"  Built {split}: {len(self.embeddings)} samples from real USGS data "
            f"(cyano exceedance rate: {self.cyano_exceed.mean():.3f})")

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            "embedding": torch.from_numpy(self.embeddings[idx]),
            "day_of_year": torch.tensor(self.day_of_year[idx]),
            "vibrio_covariates": torch.from_numpy(self.vibrio_covs[idx]),
            "naegleria_covariates": torch.from_numpy(self.naegleria_covs[idx]),
            "schisto_covariates": torch.from_numpy(self.schisto_covs[idx]),
            "cyano_concentrations": torch.from_numpy(self.cyano_conc[idx]),
            "cyano_exceedance": torch.from_numpy(self.cyano_exceed[idx]),
            "vibrio_risk": torch.from_numpy(self.vibrio_risk[idx]),
            "naegleria_prob": torch.from_numpy(self.naegleria_prob[idx]),
            "schisto_prob": torch.from_numpy(self.schisto_prob[idx]),
        }


# ---------------------------------------------------------------------------
# Metric utilities
# ---------------------------------------------------------------------------
def compute_auroc(labels, scores):
    """Compute AUROC manually (no sklearn dependency required).

    Uses the trapezoidal rule on the ROC curve.
    Returns NaN if only one class is present.
    """
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)

    if len(np.unique(labels)) < 2:
        return float("nan")

    # Sort by decreasing score
    desc = np.argsort(-scores)
    labels_sorted = labels[desc]
    scores_sorted = scores[desc]

    n_pos = labels_sorted.sum()
    n_neg = len(labels_sorted) - n_pos

    tp = 0
    fp = 0
    tpr_prev = 0.0
    fpr_prev = 0.0
    auc = 0.0

    for i in range(len(labels_sorted)):
        if labels_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        # Trapezoidal rule
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0
        tpr_prev = tpr
        fpr_prev = fpr

    return float(auc)


def compute_ece(probs, labels, n_bins=15):
    """Expected Calibration Error.

    probs: predicted probabilities (N,)
    labels: binary ground truth (N,)
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(probs)

    bin_details = []
    for b in range(n_bins):
        lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
        mask = (probs > lo) & (probs <= hi) if b > 0 else (probs >= lo) & (probs <= hi)
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue
        avg_conf = probs[mask].mean()
        avg_acc = labels[mask].mean()
        ece += (n_in_bin / total) * abs(avg_acc - avg_conf)
        bin_details.append({
            "bin": f"({lo:.2f}, {hi:.2f}]",
            "count": int(n_in_bin),
            "avg_confidence": round(float(avg_conf), 4),
            "avg_accuracy": round(float(avg_acc), 4),
            "gap": round(float(abs(avg_acc - avg_conf)), 4),
        })

    return float(ece), bin_details


def compute_target_alert_level(cyano_exceed, vibrio_risk, naeg_prob, schisto_prob):
    """Compute ground-truth alert level using same thresholds as model.

    Mirrors IntegratedDiseaseRisk._compute_alerts but on numpy targets.
    """
    # Cyanotoxin: max exceedance across all toxins/horizons
    max_cyano_exc = cyano_exceed.max()

    # Vibrio: max risk
    max_vibrio = vibrio_risk.max()

    # Naegleria: max prob
    max_naeg = naeg_prob.max()

    # Schisto: max prob
    max_schisto = schisto_prob.max()

    level = 0  # LOW

    # Cyanotoxin thresholds (based on exceedance prob)
    if max_cyano_exc >= 0.85:
        level = max(level, 3)
    elif max_cyano_exc >= 0.6:
        level = max(level, 2)
    elif max_cyano_exc >= 0.3:
        level = max(level, 1)

    # Vibrio thresholds
    if max_vibrio >= 0.85:
        level = max(level, 3)
    elif max_vibrio >= 0.6:
        level = max(level, 2)
    elif max_vibrio >= 0.3:
        level = max(level, 1)

    # Naegleria thresholds
    if max_naeg >= 0.75:
        level = max(level, 3)
    elif max_naeg >= 0.5:
        level = max(level, 2)
    elif max_naeg >= 0.25:
        level = max(level, 1)

    # Schistosomiasis thresholds
    if max_schisto >= 0.85:
        level = max(level, 3)
    elif max_schisto >= 0.6:
        level = max(level, 2)
    elif max_schisto >= 0.3:
        level = max(level, 1)

    return level


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_evaluation(model, loader, dataset, device):
    """Run full evaluation and return all metrics."""
    model.eval()

    # Accumulators
    all_pred_alert = []
    all_true_alert = []

    # Per-pathogen accumulators
    all_cyano_exc_pred = []   # predicted exceedance probs (B, 6)
    all_cyano_exc_true = []   # true exceedance binary (B, 6)

    all_vibrio_pred = []      # predicted risk index (B, 4)
    all_vibrio_true = []      # true risk index (B, 4)

    all_naeg_pred = []        # predicted habitat prob (B, 2)
    all_naeg_true = []        # true habitat prob (B, 2)

    all_schisto_pred = []     # predicted habitat prob (B, 2)
    all_schisto_true = []     # true habitat prob (B, 2)

    total_loss = 0.0
    n_samples = 0
    per_disease_loss_sum = defaultdict(float)

    for batch in loader:
        emb = torch.nan_to_num(batch["embedding"].to(device), nan=0.0)
        doy = batch["day_of_year"].to(device).long()
        v_covs = torch.nan_to_num(batch["vibrio_covariates"].to(device), nan=0.0)
        n_covs = torch.nan_to_num(batch["naegleria_covariates"].to(device), nan=0.0)
        s_covs = torch.nan_to_num(batch["schisto_covariates"].to(device), nan=0.0)

        summary = model(
            embedding=emb, day_of_year=doy,
            vibrio_covariates=v_covs,
            naegleria_covariates=n_covs,
            schisto_covariates=s_covs,
        )

        B_size = emb.size(0)
        cyano_conc = torch.nan_to_num(batch["cyano_concentrations"].to(device), nan=1e-3)
        cyano_exc = torch.nan_to_num(batch["cyano_exceedance"].to(device), nan=0.0)
        naeg = torch.nan_to_num(batch["naegleria_prob"].to(device), nan=0.0)
        schisto = torch.nan_to_num(batch["schisto_prob"].to(device), nan=0.0)
        targets = {
            "cyanotoxin": {
                "log_concentration": torch.log10(cyano_conc.reshape(B_size, 6).clamp(min=1e-3)),
                "drinking_exceedance": cyano_exc.reshape(B_size, 6).clamp(0, 1),
            },
            "vibrio": {"risk_index": torch.nan_to_num(batch["vibrio_risk"].to(device), nan=0.0).clamp(0, 1)},
            "naegleria": {"habitat": torch.cat([naeg, naeg * 0.9], dim=-1).clamp(0, 1)},
            "schistosomiasis": {"habitat": torch.cat([schisto, schisto * 0.95], dim=-1).clamp(0, 1)},
        }

        loss, per_disease = model.compute_loss(summary, targets)
        total_loss += loss.item() * B_size
        n_samples += B_size
        if per_disease:
            for k, v in per_disease.items():
                per_disease_loss_sum[k] += v.item() * B_size

        # Collect alert levels
        pred_levels = summary.alert_level.cpu().numpy()
        all_pred_alert.extend(pred_levels.tolist())

        # Compute true alert levels from targets
        cyano_exc_np = cyano_exc.reshape(B_size, 6).cpu().numpy()
        vibrio_np = batch["vibrio_risk"].numpy()
        naeg_np = batch["naegleria_prob"].numpy()
        schisto_np = batch["schisto_prob"].numpy()

        for i in range(B_size):
            true_level = compute_target_alert_level(
                cyano_exc_np[i], vibrio_np[i], naeg_np[i], schisto_np[i]
            )
            all_true_alert.append(true_level)

        # Per-pathogen predictions
        all_cyano_exc_pred.append(summary.cyanotoxin.drinking_exceedance_prob.cpu().numpy())
        all_cyano_exc_true.append(cyano_exc.reshape(B_size, 6).cpu().numpy())

        all_vibrio_pred.append(summary.vibrio.risk_index.cpu().numpy())
        all_vibrio_true.append(batch["vibrio_risk"].numpy())

        all_naeg_pred.append(summary.naegleria.habitat_probability.cpu().numpy())
        all_naeg_true.append(torch.cat([naeg, naeg * 0.9], dim=-1).clamp(0, 1).cpu().numpy())

        all_schisto_pred.append(summary.schistosomiasis.habitat_probability.cpu().numpy())
        all_schisto_true.append(torch.cat([schisto, schisto * 0.95], dim=-1).clamp(0, 1).cpu().numpy())

    # -----------------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------------
    pred_alert = np.array(all_pred_alert)
    true_alert = np.array(all_true_alert)

    cyano_exc_pred = np.concatenate(all_cyano_exc_pred, axis=0)
    cyano_exc_true = np.concatenate(all_cyano_exc_true, axis=0)
    vibrio_pred = np.concatenate(all_vibrio_pred, axis=0)
    vibrio_true = np.concatenate(all_vibrio_true, axis=0)
    naeg_pred = np.concatenate(all_naeg_pred, axis=0)
    naeg_true = np.concatenate(all_naeg_true, axis=0)
    schisto_pred = np.concatenate(all_schisto_pred, axis=0)
    schisto_true = np.concatenate(all_schisto_true, axis=0)

    avg_loss = total_loss / max(n_samples, 1)
    avg_per_disease = {k: v / max(n_samples, 1) for k, v in per_disease_loss_sum.items()}

    results = {}
    results["test_loss"] = round(avg_loss, 6)
    results["per_disease_loss"] = {k: round(v, 6) for k, v in avg_per_disease.items()}
    results["n_test_samples"] = int(n_samples)

    # -----------------------------------------------------------------------
    # 1. Alert level classification metrics
    # -----------------------------------------------------------------------
    # Overall accuracy
    overall_acc = float((pred_alert == true_alert).mean())
    results["alert_overall_accuracy"] = round(overall_acc, 4)

    # Per-level accuracy, precision, recall, F1
    alert_metrics = {}
    for level in range(4):
        level_name = ["LOW", "MODERATE", "HIGH", "CRITICAL"][level]
        true_mask = (true_alert == level)
        pred_mask = (pred_alert == level)

        tp = int((true_mask & pred_mask).sum())
        fp = int((~true_mask & pred_mask).sum())
        fn = int((true_mask & ~pred_mask).sum())
        tn = int((~true_mask & ~pred_mask).sum())

        support = int(true_mask.sum())
        pred_count = int(pred_mask.sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(true_alert) if len(true_alert) > 0 else 0.0

        alert_metrics[level_name] = {
            "level": level,
            "support": support,
            "predicted_count": pred_count,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
        }

    results["alert_per_level"] = alert_metrics

    # Alert distribution
    results["alert_distribution"] = {
        "true": {["LOW", "MODERATE", "HIGH", "CRITICAL"][i]: int((true_alert == i).sum()) for i in range(4)},
        "predicted": {["LOW", "MODERATE", "HIGH", "CRITICAL"][i]: int((pred_alert == i).sum()) for i in range(4)},
    }

    # Confusion matrix (rows = true, cols = predicted)
    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(true_alert, pred_alert):
        cm[t, p] += 1
    results["confusion_matrix"] = {
        "rows_are_true_cols_are_predicted": cm.tolist(),
        "labels": ["LOW", "MODERATE", "HIGH", "CRITICAL"],
    }

    # Macro and weighted F1
    f1_scores = [alert_metrics[n]["f1"] for n in ["LOW", "MODERATE", "HIGH", "CRITICAL"]]
    supports = [alert_metrics[n]["support"] for n in ["LOW", "MODERATE", "HIGH", "CRITICAL"]]
    results["alert_macro_f1"] = round(np.mean(f1_scores), 4)
    total_support = sum(supports)
    if total_support > 0:
        results["alert_weighted_f1"] = round(
            sum(f * s for f, s in zip(f1_scores, supports)) / total_support, 4
        )
    else:
        results["alert_weighted_f1"] = 0.0

    # -----------------------------------------------------------------------
    # 2. Binary AUROC: "any alert" (level>0 vs level=0)
    # -----------------------------------------------------------------------
    any_alert_true = (true_alert > 0).astype(int)
    any_alert_pred_score = (pred_alert > 0).astype(float)

    # For a better AUROC, use max predicted probability across pathogens as a continuous score
    # Combine: max cyanotoxin exceedance prob, max vibrio risk, max naegleria prob, max schisto prob
    any_alert_continuous = np.maximum.reduce([
        cyano_exc_pred.max(axis=1),
        vibrio_pred.max(axis=1),
        naeg_pred.max(axis=1),
        schisto_pred.max(axis=1),
    ])
    auroc_any_alert = compute_auroc(any_alert_true, any_alert_continuous)
    results["auroc_any_alert"] = round(auroc_any_alert, 4) if not np.isnan(auroc_any_alert) else "N/A (single class)"

    # "any alert" accuracy
    any_alert_pred_binary = (pred_alert > 0).astype(int)
    results["any_alert_accuracy"] = round(float((any_alert_pred_binary == any_alert_true).mean()), 4)

    # -----------------------------------------------------------------------
    # 3. Binary AUROC: "high alert" (level>=2 vs level<2)
    # -----------------------------------------------------------------------
    high_alert_true = (true_alert >= 2).astype(int)
    # Use same continuous scores but threshold differently
    auroc_high_alert = compute_auroc(high_alert_true, any_alert_continuous)
    results["auroc_high_alert"] = round(auroc_high_alert, 4) if not np.isnan(auroc_high_alert) else "N/A (single class)"

    high_alert_pred_binary = (pred_alert >= 2).astype(int)
    results["high_alert_accuracy"] = round(float((high_alert_pred_binary == high_alert_true).mean()), 4)

    # -----------------------------------------------------------------------
    # 4. Per-pathogen metrics
    # -----------------------------------------------------------------------
    pathogen_metrics = {}

    # --- Cyanotoxin exceedance ---
    # Binary: threshold exceedance at 0.5 decision boundary
    cyano_true_binary = (cyano_exc_true > 0.5).astype(int).flatten()
    cyano_pred_probs = cyano_exc_pred.flatten()
    cyano_pred_binary = (cyano_pred_probs > 0.5).astype(int)

    cyano_acc = float((cyano_pred_binary == cyano_true_binary).mean())
    cyano_auroc = compute_auroc(cyano_true_binary, cyano_pred_probs)

    tp_c = int((cyano_true_binary & cyano_pred_binary).sum())
    fp_c = int(((1 - cyano_true_binary) & cyano_pred_binary).sum())
    fn_c = int((cyano_true_binary & (1 - cyano_pred_binary)).sum())
    prec_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
    rec_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
    f1_c = 2 * prec_c * rec_c / (prec_c + rec_c) if (prec_c + rec_c) > 0 else 0.0

    pathogen_metrics["cyanotoxin_exceedance"] = {
        "accuracy": round(cyano_acc, 4),
        "auroc": round(cyano_auroc, 4) if not np.isnan(cyano_auroc) else "N/A",
        "precision": round(prec_c, 4),
        "recall": round(rec_c, 4),
        "f1": round(f1_c, 4),
        "positive_rate_true": round(float(cyano_true_binary.mean()), 4),
        "positive_rate_pred": round(float(cyano_pred_binary.mean()), 4),
    }

    # --- Vibrio risk ---
    # Binarize at 0.3 (moderate threshold)
    vibrio_true_binary = (vibrio_true > 0.3).astype(int).flatten()
    vibrio_pred_flat = vibrio_pred.flatten()
    vibrio_pred_binary = (vibrio_pred_flat > 0.3).astype(int)

    vibrio_acc = float((vibrio_pred_binary == vibrio_true_binary).mean())
    vibrio_auroc = compute_auroc(vibrio_true_binary, vibrio_pred_flat)
    vibrio_mae = float(np.abs(vibrio_pred.flatten() - vibrio_true.flatten()).mean())

    tp_v = int((vibrio_true_binary & vibrio_pred_binary).sum())
    fp_v = int(((1 - vibrio_true_binary) & vibrio_pred_binary).sum())
    fn_v = int((vibrio_true_binary & (1 - vibrio_pred_binary)).sum())
    prec_v = tp_v / (tp_v + fp_v) if (tp_v + fp_v) > 0 else 0.0
    rec_v = tp_v / (tp_v + fn_v) if (tp_v + fn_v) > 0 else 0.0
    f1_v = 2 * prec_v * rec_v / (prec_v + rec_v) if (prec_v + rec_v) > 0 else 0.0

    pathogen_metrics["vibrio_risk"] = {
        "accuracy": round(vibrio_acc, 4),
        "auroc": round(vibrio_auroc, 4) if not np.isnan(vibrio_auroc) else "N/A",
        "mae": round(vibrio_mae, 4),
        "precision": round(prec_v, 4),
        "recall": round(rec_v, 4),
        "f1": round(f1_v, 4),
        "positive_rate_true": round(float(vibrio_true_binary.mean()), 4),
        "positive_rate_pred": round(float(vibrio_pred_binary.mean()), 4),
    }

    # --- Naegleria habitat ---
    # Binarize at 0.25 (moderate threshold)
    naeg_true_binary = (naeg_true > 0.25).astype(int).flatten()
    naeg_pred_flat = naeg_pred.flatten()
    naeg_pred_binary = (naeg_pred_flat > 0.25).astype(int)

    naeg_acc = float((naeg_pred_binary == naeg_true_binary).mean())
    naeg_auroc = compute_auroc(naeg_true_binary, naeg_pred_flat)
    naeg_mae = float(np.abs(naeg_pred.flatten() - naeg_true.flatten()).mean())

    tp_n = int((naeg_true_binary & naeg_pred_binary).sum())
    fp_n = int(((1 - naeg_true_binary) & naeg_pred_binary).sum())
    fn_n = int((naeg_true_binary & (1 - naeg_pred_binary)).sum())
    prec_n = tp_n / (tp_n + fp_n) if (tp_n + fp_n) > 0 else 0.0
    rec_n = tp_n / (tp_n + fn_n) if (tp_n + fn_n) > 0 else 0.0
    f1_n = 2 * prec_n * rec_n / (prec_n + rec_n) if (prec_n + rec_n) > 0 else 0.0

    pathogen_metrics["naegleria_habitat"] = {
        "accuracy": round(naeg_acc, 4),
        "auroc": round(naeg_auroc, 4) if not np.isnan(naeg_auroc) else "N/A",
        "mae": round(naeg_mae, 4),
        "precision": round(prec_n, 4),
        "recall": round(rec_n, 4),
        "f1": round(f1_n, 4),
        "positive_rate_true": round(float(naeg_true_binary.mean()), 4),
        "positive_rate_pred": round(float(naeg_pred_binary.mean()), 4),
    }

    # --- Schistosomiasis habitat ---
    # Binarize at 0.3 (moderate threshold)
    schisto_true_binary = (schisto_true > 0.3).astype(int).flatten()
    schisto_pred_flat = schisto_pred.flatten()
    schisto_pred_binary = (schisto_pred_flat > 0.3).astype(int)

    schisto_acc = float((schisto_pred_binary == schisto_true_binary).mean())
    schisto_auroc = compute_auroc(schisto_true_binary, schisto_pred_flat)
    schisto_mae = float(np.abs(schisto_pred.flatten() - schisto_true.flatten()).mean())

    tp_s = int((schisto_true_binary & schisto_pred_binary).sum())
    fp_s = int(((1 - schisto_true_binary) & schisto_pred_binary).sum())
    fn_s = int((schisto_true_binary & (1 - schisto_pred_binary)).sum())
    prec_s = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0.0
    rec_s = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0.0
    f1_s = 2 * prec_s * rec_s / (prec_s + rec_s) if (prec_s + rec_s) > 0 else 0.0

    pathogen_metrics["schistosomiasis_habitat"] = {
        "accuracy": round(schisto_acc, 4),
        "auroc": round(schisto_auroc, 4) if not np.isnan(schisto_auroc) else "N/A",
        "mae": round(schisto_mae, 4),
        "precision": round(prec_s, 4),
        "recall": round(rec_s, 4),
        "f1": round(f1_s, 4),
        "positive_rate_true": round(float(schisto_true_binary.mean()), 4),
        "positive_rate_pred": round(float(schisto_pred_binary.mean()), 4),
    }

    results["pathogen_metrics"] = pathogen_metrics

    # -----------------------------------------------------------------------
    # 5. Calibration analysis
    # -----------------------------------------------------------------------
    calibration = {}

    # Cyanotoxin exceedance calibration
    ece_cyano, bins_cyano = compute_ece(cyano_pred_probs, cyano_true_binary)
    calibration["cyanotoxin_exceedance"] = {
        "ece": round(ece_cyano, 4),
        "n_bins_populated": len(bins_cyano),
        "bin_details": bins_cyano,
    }

    # Vibrio risk calibration (treat risk as probability, binarize true at 0.5)
    vibrio_true_cal = (vibrio_true > 0.5).astype(int).flatten()
    ece_vibrio, bins_vibrio = compute_ece(vibrio_pred_flat, vibrio_true_cal)
    calibration["vibrio_risk"] = {
        "ece": round(ece_vibrio, 4),
        "n_bins_populated": len(bins_vibrio),
        "bin_details": bins_vibrio,
    }

    # Naegleria calibration (binarize true at 0.5)
    naeg_true_cal = (naeg_true > 0.5).astype(int).flatten()
    ece_naeg, bins_naeg = compute_ece(naeg_pred_flat, naeg_true_cal)
    calibration["naegleria_habitat"] = {
        "ece": round(ece_naeg, 4),
        "n_bins_populated": len(bins_naeg),
        "bin_details": bins_naeg,
    }

    results["calibration"] = calibration

    return results


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
def print_summary(results):
    """Print a clear human-readable summary."""
    print("\n" + "=" * 70)
    print("DISEASE FORECAST MODEL -- TEST SET EVALUATION")
    print("=" * 70)

    print(f"\nTest samples: {results['n_test_samples']}")
    print(f"Test loss: {results['test_loss']:.4f}")
    if results.get("per_disease_loss"):
        for k, v in results["per_disease_loss"].items():
            print(f"  {k}: {v:.4f}")

    print("\n" + "-" * 70)
    print("ALERT LEVEL CLASSIFICATION")
    print("-" * 70)

    print(f"\nOverall accuracy: {results['alert_overall_accuracy']:.4f}")
    print(f"Macro F1:         {results['alert_macro_f1']:.4f}")
    print(f"Weighted F1:      {results['alert_weighted_f1']:.4f}")

    print(f"\n{'Level':<12} {'Support':>8} {'Pred':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("-" * 60)
    for name in ["LOW", "MODERATE", "HIGH", "CRITICAL"]:
        m = results["alert_per_level"][name]
        print(f"{name:<12} {m['support']:>8d} {m['predicted_count']:>8d} "
              f"{m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}")

    print(f"\nConfusion Matrix (rows=true, cols=predicted):")
    cm = results["confusion_matrix"]["rows_are_true_cols_are_predicted"]
    labels = results["confusion_matrix"]["labels"]
    header = f"{'':>12}" + "".join(f"{l:>10}" for l in labels)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{labels[i]:>12}" + "".join(f"{v:>10d}" for v in row)
        print(row_str)

    print(f"\nAlert distribution:")
    print(f"  True:      {results['alert_distribution']['true']}")
    print(f"  Predicted: {results['alert_distribution']['predicted']}")

    print("\n" + "-" * 70)
    print("BINARY ALERT AUROC")
    print("-" * 70)
    print(f"Any alert (level>0 vs 0)  -- AUROC: {results['auroc_any_alert']}, "
          f"Accuracy: {results['any_alert_accuracy']:.4f}")
    print(f"High alert (level>=2 vs <2) -- AUROC: {results['auroc_high_alert']}, "
          f"Accuracy: {results['high_alert_accuracy']:.4f}")

    print("\n" + "-" * 70)
    print("PER-PATHOGEN METRICS")
    print("-" * 70)
    for pathogen, m in results["pathogen_metrics"].items():
        print(f"\n  {pathogen}:")
        print(f"    Accuracy:  {m['accuracy']:.4f}    AUROC: {m['auroc']}")
        print(f"    Precision: {m['precision']:.4f}    Recall: {m['recall']:.4f}    F1: {m['f1']:.4f}")
        if 'mae' in m:
            print(f"    MAE:       {m['mae']:.4f}")
        print(f"    Positive rate -- true: {m['positive_rate_true']:.4f}, pred: {m['positive_rate_pred']:.4f}")

    print("\n" + "-" * 70)
    print("CALIBRATION (Expected Calibration Error)")
    print("-" * 70)
    for pathogen, c in results["calibration"].items():
        print(f"  {pathogen}: ECE = {c['ece']:.4f} ({c['n_bins_populated']} bins populated)")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate Disease Forecasting Model")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str,
                        default=str(CKPT_DIR / "disease_forecast_best.pt"))
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log("=" * 60)
    log("Disease Risk Forecasting -- Evaluation (REAL USGS data)")
    log("=" * 60)
    log(f"Device: {device}")
    log(f"Checkpoint: {args.checkpoint}")

    # Load model
    from sentinel.models.biology.disease_forecast import IntegratedDiseaseRisk
    model = IntegratedDiseaseRisk().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model parameters: {n_params:,}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    log(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    if "val_metrics" in ckpt:
        log(f"  Checkpoint val loss: {ckpt['val_metrics'].get('loss', '?')}")

    # Load test set
    log("Loading test dataset...")
    test_ds = DiseaseDataset(split="test", seed=args.seed)
    log(f"Test set: {len(test_ds)} samples")

    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # Run evaluation
    log("Running evaluation...")
    t0 = time.time()
    results = run_evaluation(model, test_loader, test_ds, device)
    dt = time.time() - t0
    log(f"Evaluation completed in {dt:.1f}s")

    # Add metadata
    results["model"] = "IntegratedDiseaseRisk"
    results["checkpoint"] = args.checkpoint
    results["checkpoint_epoch"] = ckpt.get("epoch", None)
    results["n_params"] = n_params
    results["data_source"] = "USGS NWIS real sensor data"
    results["evaluation_device"] = str(device)

    # Save results
    out_path = RESULTS_DIR / "disease_forecast_evaluation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Results saved to {out_path}")

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
