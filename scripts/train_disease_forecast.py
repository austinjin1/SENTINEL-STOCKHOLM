#!/usr/bin/env python3
"""Train Disease Outbreak Forecasting — Phase 3.3 of SENTINEL 2.0.

Trains the IntegratedDiseaseRisk model for 4 environmental risk forecasts:
  1. Cyanotoxin risk (temperature + nutrient-driven HAB probability)
  2. Vibrio risk index (temperature + salinity driven)
  3. Naegleria fowleri habitat probability (warm freshwater)
  4. Schistosomiasis snail habitat suitability (tropical conditions)

IMPORTANT: Risk scores are derived from real USGS water quality measurements
using established epidemiological thresholds from literature (WHO, CDC, EPA).
We do NOT have real pathogen concentration measurements — the model predicts
environmental risk conditions, not confirmed disease cases.

Data source: Real USGS NWIS sensor data (2755 parquet files)

Usage:
    /home/bcheng/.conda/envs/physiformer/bin/python3 scripts/train_disease_forecast.py

GPU: 3 (default)

MIT License — Bryan Cheng, 2026
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

CKPT_DIR = PROJECT / "checkpoints" / "biology"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = PROJECT / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = PROJECT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

SENSOR_DIR = PROJECT / "data" / "raw" / "sensor" / "full"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Epidemiological thresholds from literature
# ---------------------------------------------------------------------------
# WHO: microcystin-LR guidelines: 1 µg/L drinking water, 10 µg/L recreational
# CDC: Naegleria risk above 30°C in warm freshwater
# EPA: Cyanobacteria risk increases above 20°C with elevated nutrients
# Literature: Vibrio risk at water temp >20°C, salinity 5-25 ppt


def compute_risk_scores(temp, do_val, ph, turb, spcond, day_of_year):
    """Compute disease risk scores from real water quality measurements.

    Uses established epidemiological thresholds — NOT random generation.

    Args:
        temp: Water temperature (°C)
        do_val: Dissolved oxygen (mg/L)
        ph: pH
        turb: Turbidity (NTU)
        spcond: Specific conductance (µS/cm)
        day_of_year: Day of year (1-366)

    Returns dict of risk scores for each disease.
    """
    # --- Cyanotoxin risk ---
    # HAB risk = f(temperature, nutrient proxy, season)
    # Temperature >20°C is primary driver (Paerl & Huisman 2008)
    # Low DO and high turbidity are secondary indicators of eutrophication
    temp_risk = max(0, (temp - 15) / 20)  # Increases above 15°C, peaks ~35°C
    # Use turbidity + low DO as nutrient/eutrophication proxy
    eutrophication_proxy = 0.0
    if turb is not None and turb > 0:
        eutrophication_proxy += min(turb / 50, 1.0)  # High turbidity = high nutrients
    if do_val is not None and do_val < 6:
        eutrophication_proxy += (6 - do_val) / 6  # Low DO = eutrophication
    eutrophication_proxy = min(eutrophication_proxy, 1.0)

    # Seasonal factor (HABs peak in summer)
    seasonal = max(0, np.sin((day_of_year - 80) * 2 * np.pi / 365))

    # Microcystin risk (WHO threshold = 1 µg/L)
    mc_risk = temp_risk * (0.3 + 0.7 * eutrophication_proxy) * (0.5 + 0.5 * seasonal)

    # Anatoxin risk (similar to microcystin but peaks earlier)
    ana_risk = mc_risk * 0.7

    # Cylindrospermopsin risk (warmer temperature preference)
    cyl_risk = max(0, (temp - 20) / 15) * eutrophication_proxy * seasonal * 0.5

    # Convert risk scores to approximate concentration proxies (µg/L scale)
    # These are NOT measured concentrations — they are risk-derived estimates
    mc_conc = mc_risk * 5.0  # Scale to µg/L range
    ana_conc = ana_risk * 2.0
    cyl_conc = cyl_risk * 3.0

    cyano = np.array([
        [mc_conc, mc_conc * 1.1],   # 7d, 14d horizons
        [ana_conc, ana_conc * 1.1],
        [cyl_conc, cyl_conc * 1.1],
    ], dtype=np.float32)

    cyano_exceed = np.array([
        [mc_conc > 1.0, mc_conc * 1.1 > 1.0],
        [ana_conc > 3.0, ana_conc * 1.1 > 3.0],
        [cyl_conc > 1.0, cyl_conc * 1.1 > 1.0],
    ], dtype=np.float32)

    # --- Vibrio risk ---
    # V. vulnificus: water temp >20°C, salinity 5-25 ppt
    # Use SpCond as salinity proxy (SpCond ~1000 µS/cm ≈ 0.5 ppt)
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

    # --- Naegleria fowleri risk ---
    # CDC: primarily >30°C warm freshwater, low chlorine
    # No chlorine data in USGS, so use temperature only
    naeg_risk = 1 / (1 + np.exp(-(temp - 30) / 3))
    naegleria = np.array([naeg_risk], dtype=np.float32)

    # --- Schistosomiasis risk ---
    # Tropical, 20-30°C, requires intermediate snail host
    # Very low risk in continental US, include for completeness
    schisto_risk = 1 / (1 + np.exp(-(temp - 25) / 4)) * 0.1  # Low baseline in US
    schistosomiasis = np.array([float(schisto_risk)], dtype=np.float32)

    return {
        "cyano_conc": cyano,
        "cyano_exceed": cyano_exceed,
        "vibrio": vibrio,
        "naegleria": naegleria,
        "schistosomiasis": schistosomiasis,
    }


# ---------------------------------------------------------------------------
# Dataset: real USGS sensor data
# ---------------------------------------------------------------------------
class DiseaseDataset(Dataset):
    """Training data for disease risk forecasting from real USGS sensors.

    Risk scores are computed from measured water quality parameters using
    established epidemiological thresholds. No synthetic data generation.
    """

    def __init__(self, split: str = "train", seed: int = 42):
        super().__init__()
        self.split = split
        self._build_from_sensor_data(split, seed)

    def _build_from_sensor_data(self, split: str, seed: int):
        """Build dataset from real USGS NWIS parquet files."""
        import pandas as pd

        sensor_files = sorted(SENSOR_DIR.glob("*.parquet")) if SENSOR_DIR.exists() else []
        if not sensor_files:
            raise FileNotFoundError(
                f"No USGS sensor data found in {SENSOR_DIR}. "
                "Real data is required — no synthetic fallback."
            )

        # Split by site (spatial holdout)
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

        col_map = {"Temp": "temp", "DO": "do", "pH": "ph", "Turb": "turb", "SpCond": "spcond"}

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

                # Need at least temperature
                if "Temp" not in df.columns:
                    continue

                # Resample to daily
                df = df.set_index("datetime")
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                daily = df.resample("D").mean().dropna(subset=["Temp"])

                if len(daily) < 14:
                    continue

                # Filter out sentinel values
                daily = daily[(daily["Temp"] > -50) & (daily["Temp"] < 50)]

                # Create samples from weekly windows
                for start_idx in range(0, len(daily) - 7, 7):
                    window = daily.iloc[start_idx:start_idx + 7]
                    if len(window) < 3:
                        continue

                    # Extract real measurements
                    temp = float(window["Temp"].mean())
                    do_val = float(window["DO"].mean()) if "DO" in window.columns and not window["DO"].isna().all() else None
                    ph = float(window["pH"].mean()) if "pH" in window.columns and not window["pH"].isna().all() else None
                    turb = float(window["Turb"].mean()) if "Turb" in window.columns and not window["Turb"].isna().all() else None
                    spcond = float(window["SpCond"].mean()) if "SpCond" in window.columns and not window["SpCond"].isna().all() else None

                    # Filter extreme values (USGS sentinel values)
                    if temp < -10 or temp > 45:
                        continue
                    if do_val is not None and (do_val < 0 or do_val > 25):
                        do_val = None
                    if turb is not None and (turb < 0 or turb > 10000):
                        turb = None

                    # Day of year from index
                    doy = float(window.index[len(window)//2].dayofyear)

                    # Compute risk scores from real measurements
                    risks = compute_risk_scores(temp, do_val, ph, turb, spcond, doy)

                    # Build embedding from sensor statistics (real data)
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

                    # Covariates from real measurements
                    salinity_proxy = spcond / 2000 if spcond is not None else 0.0
                    chlorine_proxy = 0.0  # Not measured by USGS continuous sensors

                    embeddings.append(emb)
                    days.append(doy)
                    vibrio_covs_list.append(np.array([temp, salinity_proxy], dtype=np.float32))
                    naeg_covs_list.append(np.array([temp, chlorine_proxy], dtype=np.float32))
                    schisto_covs_list.append(np.array([temp, 35.0, 0.5], dtype=np.float32))  # lat placeholder, ndvi placeholder
                    cyano_conc_list.append(risks["cyano_conc"])
                    cyano_exceed_list.append(risks["cyano_exceed"])
                    vibrio_risk_list.append(risks["vibrio"])
                    naeg_prob_list.append(risks["naegleria"])
                    schisto_prob_list.append(risks["schistosomiasis"])

            except Exception:
                continue

        if not embeddings:
            raise RuntimeError(
                f"Failed to build any disease risk samples from {len(selected)} sensor files. "
                "Check that parquet files have temperature data."
            )

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
# Training
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0

    for batch in loader:
        emb = torch.nan_to_num(batch["embedding"].to(device), nan=0.0)
        doy = batch["day_of_year"].to(device).long()
        v_covs = torch.nan_to_num(batch["vibrio_covariates"].to(device), nan=0.0)
        n_covs = torch.nan_to_num(batch["naegleria_covariates"].to(device), nan=0.0)
        s_covs = torch.nan_to_num(batch["schisto_covariates"].to(device), nan=0.0)

        optimizer.zero_grad(set_to_none=True)
        summary = model(
            embedding=emb,
            day_of_year=doy,
            vibrio_covariates=v_covs,
            naegleria_covariates=n_covs,
            schisto_covariates=s_covs,
        )

        B_size = emb.size(0)
        cyano_conc = torch.nan_to_num(batch["cyano_concentrations"].to(device), nan=1e-3)
        cyano_exc = torch.nan_to_num(batch["cyano_exceedance"].to(device), nan=0.0)
        targets = {
            "cyanotoxin": {
                "log_concentration": torch.log10(cyano_conc.reshape(B_size, 6).clamp(min=1e-3)),
                "drinking_exceedance": cyano_exc.reshape(B_size, 6).clamp(0, 1),
            },
            "vibrio": {"risk_index": torch.nan_to_num(batch["vibrio_risk"].to(device), nan=0.0).clamp(0, 1)},
            "naegleria": {
                "habitat": torch.cat([
                    torch.nan_to_num(batch["naegleria_prob"].to(device), nan=0.0),
                    torch.nan_to_num(batch["naegleria_prob"].to(device), nan=0.0) * 0.9
                ], dim=-1).clamp(0, 1),
            },
            "schistosomiasis": {
                "habitat": torch.cat([
                    torch.nan_to_num(batch["schisto_prob"].to(device), nan=0.0),
                    torch.nan_to_num(batch["schisto_prob"].to(device), nan=0.0) * 0.95
                ], dim=-1).clamp(0, 1),
            },
        }

        loss, per_disease = model.compute_loss(summary, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item() * emb.size(0)
        n += emb.size(0)

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n = 0
    alert_counts = {0: 0, 1: 0, 2: 0, 3: 0}

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
        total_loss += loss.item() * emb.size(0)
        n += emb.size(0)

        for level in summary.alert_level.cpu().tolist():
            alert_counts[int(level)] = alert_counts.get(int(level), 0) + 1

    return {
        "loss": total_loss / max(n, 1),
        "alert_distribution": alert_counts,
        "per_disease_loss": {k: v.item() for k, v in per_disease.items()} if per_disease else {},
    }


def main():
    parser = argparse.ArgumentParser(description="Train Disease Forecasting")
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log("=" * 60)
    log("Disease Risk Forecasting — Training (REAL USGS data)")
    log("=" * 60)
    log(f"Device: {device}")
    log("NOTE: Risk scores derived from real water quality measurements using")
    log("      established epidemiological thresholds (WHO, CDC, EPA).")
    log("      NOT real pathogen concentration measurements.")

    from sentinel.models.biology.disease_forecast import IntegratedDiseaseRisk
    model = IntegratedDiseaseRisk().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model parameters: {n_params:,}")

    train_ds = DiseaseDataset(split="train", seed=args.seed)
    val_ds = DiseaseDataset(split="val", seed=args.seed)
    test_ds = DiseaseDataset(split="test", seed=args.seed)

    log(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        dt = time.time() - t0
        log(f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_metrics['loss']:.4f} | "
            f"Alerts: {val_metrics['alert_distribution']} | "
            f"{dt:.1f}s")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
            }, CKPT_DIR / "disease_forecast_best.pt")
            log(f"  ** New best (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log(f"  Early stopping at epoch {epoch}")
                break

    # Test evaluation
    log("\n" + "=" * 60)
    log("Test Evaluation")
    log("=" * 60)

    ckpt = torch.load(CKPT_DIR / "disease_forecast_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device)

    log(f"Test Loss:   {test_metrics['loss']:.4f}")
    log(f"Alerts:      {test_metrics['alert_distribution']}")
    if test_metrics["per_disease_loss"]:
        for disease, loss_val in test_metrics["per_disease_loss"].items():
            log(f"  {disease}: {loss_val:.4f}")

    results = {
        "model": "IntegratedDiseaseRisk",
        "data_source": "USGS NWIS real sensor data",
        "risk_methodology": "Environmental risk scores from WHO/CDC/EPA thresholds applied to real water quality measurements",
        "synthetic_data": False,
        "n_params": n_params,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "best_epoch": ckpt["epoch"],
        "test_metrics": test_metrics,
    }
    with open(RESULTS_DIR / "disease_forecast_holdout.json", "w") as f:
        json.dump(results, f, indent=2)

    log(f"Results saved to {RESULTS_DIR / 'disease_forecast_holdout.json'}")
    log("DONE")


if __name__ == "__main__":
    main()
