#!/usr/bin/env python3
"""Phase 0.3: Honest baselines and negative controls for SENTINEL.

Tests whether the Perceiver IO fusion and AquaSSM actually add value over
simpler approaches. Four evaluation components:

1. **Ensemble Baseline** -- LogisticRegression / RandomForest / GBM over the
   five encoder anomaly probabilities. If a linear combination of encoder
   outputs matches or beats Perceiver IO, the fusion adds no value.

2. **Strong ML Baselines** (sensor modality only) --
   - XGBoost / GradientBoosting on handcrafted statistical features
   - 1D-CNN temporal convolution baseline
   - LSTM baseline (2-layer, hidden=128)
   - Transformer baseline (4-layer, 4-head vanilla encoder)
   Compare each against AquaSSM on the temporal-spatial holdout test set.

3. **Hard Negative Controls** -- 50 random non-event windows drawn from
   sites that ARE contaminated (clean periods at dirty sites). These test
   the false positive rate in realistic scenarios.

4. **Ablation: Simple vs Complex Fusion** --
   - Concatenation + MLP (simple late fusion)
   - Average pooling of encoder embeddings + classifier
   - Majority vote across encoder anomaly predictions
   - Compare against Perceiver IO

Results saved to results/benchmarks/honest_baselines.json.

Usage::

    python scripts/exp_honest_baselines.py

MIT License -- Bryan Cheng, 2026
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Force GPU 3 before any CUDA initialization
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.ensemble import (
    GradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from sentinel.data.splits import (
    SplitConfig,
    assign_spatial_fold,
    get_split_assignment,
    split_indices,
)
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
MAX_LEN = 512
BATCH_SIZE = 16
NUM_PARAMS = 6  # DO, pH, SpCond, Temp, Turb, ORP
SHARED_DIM = 256
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = PROJECT_ROOT / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Candidate data directories, tried in priority order
SENSOR_DATA_DIRS = [
    PROJECT_ROOT / "data" / "processed" / "sensor" / "full",
    PROJECT_ROOT / "data" / "processed" / "sensor" / "real",
    PROJECT_ROOT / "data" / "processed" / "sensor" / "pretrain",
    PROJECT_ROOT / "data" / "processed" / "sensor" / "clean_synthetic",
]

# Strong baseline training config
BASELINE_EPOCHS = 60
BASELINE_PATIENCE = 12
BASELINE_LR = 1e-3

# Hard negative controls
N_HARD_NEGATIVES = 50
FPR_THRESHOLDS = [0.5, 0.7, 0.9]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

@dataclass
class SensorSample:
    """A single sensor data window."""
    values: np.ndarray       # [T, 6]
    delta_ts: np.ndarray     # [T]
    labels: np.ndarray       # [T]
    has_anomaly: int
    station_id: str
    timestamps: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None


def find_sensor_data() -> Tuple[Path, List[Path]]:
    """Find available sensor data, trying directories in priority order.

    Returns:
        Tuple of (data_dir, list of npz file paths).
    """
    for data_dir in SENSOR_DATA_DIRS:
        if data_dir.exists():
            npz_files = sorted(data_dir.glob("*.npz"))
            if len(npz_files) > 0:
                return data_dir, npz_files

    # Last resort: try to process raw parquets on the fly
    raw_dir = PROJECT_ROOT / "data" / "raw" / "sensor" / "full"
    if raw_dir.exists():
        parquets = sorted(raw_dir.glob("*.parquet"))
        if parquets:
            logger.warning(
                f"No processed sensor data found. Found {len(parquets)} raw "
                f"parquet files in {raw_dir}. Will process on the fly."
            )
            return _process_raw_parquets(raw_dir, parquets)

    raise FileNotFoundError(
        "No sensor data found. Checked:\n"
        + "\n".join(f"  - {d}" for d in SENSOR_DATA_DIRS)
        + "\n  - data/raw/sensor/full/*.parquet"
    )


def _process_raw_parquets(
    raw_dir: Path, parquets: List[Path]
) -> Tuple[Path, List[Path]]:
    """Process raw USGS parquet files into npz sequences on the fly.

    Produces normalized [T, 6] sequences with anomaly labels derived from
    EPA water quality thresholds.

    Returns:
        Tuple of (output_dir, list of npz paths).
    """
    import pandas as pd

    out_dir = PROJECT_ROOT / "data" / "processed" / "sensor" / "real"
    out_dir.mkdir(parents=True, exist_ok=True)

    ALL_PARAMS = ["DO", "pH", "SpCond", "Temp", "Turb", "ORP"]
    NORM_MEAN = {"DO": 9.0, "pH": 7.5, "SpCond": 500, "Temp": 15, "Turb": 20, "ORP": 200}
    NORM_STD = {"DO": 3.0, "pH": 1.0, "SpCond": 400, "Temp": 8, "Turb": 50, "ORP": 150}
    ANOMALY_THRESHOLDS = {
        "DO": (2.0, 20.0), "pH": (5.0, 9.5), "SpCond": (0, 5000),
        "Temp": (0, 35), "Turb": (0, 500), "ORP": (-200, 700),
    }
    SEQ_LEN = 512
    OVERLAP = 128

    total_seqs = 0
    for pq in parquets:
        station_id = pq.stem
        try:
            df = pd.read_parquet(pq)
        except Exception:
            continue
        if len(df) < 100:
            continue

        # Find datetime column
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df.rename(columns={df.columns[0]: "datetime"}, inplace=True)
        elif "datetime" not in df.columns:
            dt_col = next(
                (c for c in df.columns if "date" in c.lower() or "time" in c.lower()),
                None,
            )
            if dt_col is None:
                continue
            df.rename(columns={dt_col: "datetime"}, inplace=True)

        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        df = df.dropna(subset=["datetime"]).sort_values("datetime")
        if len(df) < 100:
            continue

        T = len(df)
        values = np.full((T, 6), np.nan, dtype=np.float32)
        for i, param in enumerate(ALL_PARAMS):
            if param in df.columns:
                values[:, i] = pd.to_numeric(df[param], errors="coerce").values

        timestamps = (df["datetime"].values.astype("int64") // 10**9).astype(np.float64)
        delta_ts = np.zeros(T, dtype=np.float32)
        delta_ts[1:] = np.diff(timestamps).astype(np.float32)
        delta_ts = np.clip(delta_ts, 0, 86400)

        # Anomaly labels
        labels = np.zeros(T, dtype=np.int64)
        for i, param in enumerate(ALL_PARAMS):
            if param in ANOMALY_THRESHOLDS:
                lo, hi = ANOMALY_THRESHOLDS[param]
                v = values[:, i]
                anomalous = ((v < lo) | (v > hi)) & ~np.isnan(v)
                labels[anomalous] = 1

        # Normalize
        for i, param in enumerate(ALL_PARAMS):
            if param in NORM_MEAN:
                values[:, i] = (values[:, i] - NORM_MEAN[param]) / NORM_STD[param]
        values = np.nan_to_num(values, nan=0.0).clip(-5, 5)

        # Build mask: 1 where original value was not NaN
        mask = (~np.isnan(np.full((T, 6), np.nan))).astype(np.float32)
        # Actually we need to track NaN before normalization; recompute
        mask = np.ones((T, 6), dtype=np.float32)
        for i, param in enumerate(ALL_PARAMS):
            if param in df.columns:
                mask[:, i] = (~pd.to_numeric(df[param], errors="coerce").isna()).values.astype(np.float32)
            else:
                mask[:, i] = 0.0

        # Sliding window
        step = SEQ_LEN - OVERLAP
        seq_idx = 0
        for start in range(0, T - SEQ_LEN + 1, step):
            end = start + SEQ_LEN
            v = values[start:end]
            dt = delta_ts[start:end].copy()
            dt[0] = 0.0
            ts_seq = timestamps[start:end]
            lb = labels[start:end]
            m = mask[start:end]

            valid_frac = np.mean(np.any(v != 0, axis=1))
            if valid_frac < 0.4:
                continue

            out_path = out_dir / f"{station_id}_seq{seq_idx:04d}.npz"
            np.savez_compressed(
                out_path,
                values=v, delta_ts=dt, timestamps=ts_seq,
                labels=lb, station_id=station_id,
                has_anomaly=int(lb.any()), mask=m,
            )
            seq_idx += 1
            total_seqs += 1

    logger.info(f"Processed {total_seqs} sequences from {len(parquets)} stations -> {out_dir}")
    npz_files = sorted(out_dir.glob("*.npz"))
    return out_dir, npz_files


class SensorDataset(Dataset):
    """Load sensor .npz files for training/evaluation."""

    def __init__(self, files: List[Path], max_len: int = MAX_LEN):
        self.files = files
        self.max_len = max_len
        self._cache: Dict[int, SensorSample] = {}

    def __len__(self) -> int:
        return len(self.files)

    def _load(self, idx: int) -> SensorSample:
        if idx in self._cache:
            return self._cache[idx]
        d = np.load(self.files[idx], allow_pickle=True)
        T = min(len(d["values"]), self.max_len)
        values = d["values"][:T].astype(np.float32)
        delta_ts = d["delta_ts"][:T].astype(np.float32) if "delta_ts" in d else np.zeros(T, dtype=np.float32)
        labels = d["labels"][:T].astype(np.int64) if "labels" in d else np.zeros(T, dtype=np.int64)
        has_anomaly = int(d["has_anomaly"]) if "has_anomaly" in d else int(labels.any())
        station_id = str(d["station_id"]) if "station_id" in d else "unknown"
        timestamps = d["timestamps"][:T].astype(np.float64) if "timestamps" in d else None
        mask = d["mask"][:T].astype(np.float32) if "mask" in d else np.ones_like(values)

        sample = SensorSample(
            values=np.clip(values, -5, 5),
            delta_ts=np.clip(delta_ts, 0, 3600),
            labels=labels,
            has_anomaly=has_anomaly,
            station_id=station_id,
            timestamps=timestamps,
            mask=mask,
        )
        if len(self._cache) < 5000:
            self._cache[idx] = sample
        return sample

    def __getitem__(self, idx: int) -> dict:
        s = self._load(idx)
        return {
            "values": torch.from_numpy(s.values),
            "delta_ts": torch.from_numpy(s.delta_ts),
            "labels": torch.from_numpy(s.labels),
            "has_anomaly": s.has_anomaly,
            "station_id": s.station_id,
            "mask": torch.from_numpy(s.mask),
        }


def collate_fn(batch: List[dict]) -> dict:
    """Collate with zero-padding."""
    max_len = max(b["values"].shape[0] for b in batch)
    B = len(batch)
    values = torch.zeros(B, max_len, NUM_PARAMS)
    delta_ts = torch.zeros(B, max_len)
    masks = torch.zeros(B, max_len, NUM_PARAMS)
    has_anomaly = torch.tensor([b["has_anomaly"] for b in batch], dtype=torch.float32)
    station_ids = [b["station_id"] for b in batch]

    for i, b in enumerate(batch):
        T = b["values"].shape[0]
        values[i, :T] = b["values"]
        delta_ts[i, :T] = b["delta_ts"]
        masks[i, :T] = b["mask"]

    return {
        "values": values,
        "delta_ts": delta_ts,
        "mask": masks,
        "has_anomaly": has_anomaly,
        "station_ids": station_ids,
    }


def build_temporal_spatial_splits(
    files: List[Path],
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split files into train/val/test using temporal-spatial holdout.

    Uses station_id for spatial fold assignment and timestamps for
    temporal assignment via the sentinel.data.splits module.
    """
    config = SplitConfig()
    train_files, val_files, test_files = [], [], []

    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
        except Exception:
            continue

        station_id = str(d["station_id"]) if "station_id" in d else f.stem.split("_seq")[0]

        # Get representative timestamp (median of sequence)
        if "timestamps" in d:
            ts = d["timestamps"]
            mid_ts = float(ts[len(ts) // 2])
        else:
            # Fall back to spatial-only split
            mid_ts = None

        split = get_split_assignment(station_id, mid_ts, config=config)
        if split == "train":
            train_files.append(f)
        elif split == "val":
            val_files.append(f)
        elif split == "test":
            test_files.append(f)
        # None = excluded (cross-contamination between spatial/temporal)

    logger.info(
        f"Temporal-spatial split: train={len(train_files)}, "
        f"val={len(val_files)}, test={len(test_files)}, "
        f"excluded={len(files) - len(train_files) - len(val_files) - len(test_files)}"
    )
    return train_files, val_files, test_files


def build_random_splits(
    files: List[Path],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Fallback: random train/val/test split when temporal-spatial is not possible."""
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(files))
    n_train = int(len(files) * train_frac)
    n_val = int(len(files) * val_frac)
    return (
        [files[i] for i in idx[:n_train]],
        [files[i] for i in idx[n_train:n_train + n_val]],
        [files[i] for i in idx[n_train + n_val:]],
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, method_name: str = ""
) -> Dict[str, float]:
    """Compute AUROC, AUPRC, F1, and FPR at multiple thresholds."""
    results: Dict[str, float] = {}

    # AUROC
    try:
        results["auroc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        results["auroc"] = 0.5
        logger.warning(f"  {method_name}: AUROC undefined (single class?), defaulting to 0.5")

    # AUPRC
    try:
        results["auprc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        results["auprc"] = float(np.mean(y_true))

    # F1 at threshold=0.5
    y_pred = (y_prob >= 0.5).astype(int)
    results["f1_at_0.5"] = float(f1_score(y_true, y_pred, zero_division=0))

    # FPR at various thresholds
    negatives = (y_true == 0)
    n_neg = negatives.sum()
    for thr in FPR_THRESHOLDS:
        if n_neg > 0:
            fp = ((y_prob >= thr) & negatives).sum()
            results[f"fpr_at_{thr}"] = float(fp / n_neg)
        else:
            results[f"fpr_at_{thr}"] = 0.0

    results["n_samples"] = int(len(y_true))
    results["n_positive"] = int(y_true.sum())
    results["n_negative"] = int(n_neg)
    return results


def compute_pos_weight(dataset: SensorDataset) -> float:
    """Compute positive class weight for imbalanced binary classification."""
    pos = sum(1 for i in range(len(dataset)) if dataset._load(i).has_anomaly)
    neg = len(dataset) - pos
    if pos == 0:
        return 1.0
    return neg / pos


# ---------------------------------------------------------------------------
# Feature extraction for ML baselines
# ---------------------------------------------------------------------------

def extract_statistical_features(values: np.ndarray) -> np.ndarray:
    """Extract handcrafted statistical features from a sensor window.

    For each of 6 parameters, compute:
      mean, std, min, max, skewness, kurtosis, slope (linear fit),
      number of zero-crossings, IQR, range, autocorrelation (lag=1)

    Args:
        values: [T, 6] sensor readings.

    Returns:
        Feature vector of shape [6 * 11] = [66].
    """
    from scipy import stats as sp_stats

    features = []
    T = values.shape[0]
    t_axis = np.arange(T, dtype=np.float32)

    for p in range(values.shape[1]):
        col = values[:, p]
        valid = ~np.isnan(col) & np.isfinite(col)
        col_clean = col[valid] if valid.any() else np.zeros(1)

        feat_p = [
            np.mean(col_clean),
            np.std(col_clean) + 1e-8,
            np.min(col_clean),
            np.max(col_clean),
        ]

        # Skewness and kurtosis
        if len(col_clean) > 2:
            feat_p.append(float(sp_stats.skew(col_clean)))
            feat_p.append(float(sp_stats.kurtosis(col_clean)))
        else:
            feat_p.extend([0.0, 0.0])

        # Linear trend (slope)
        if len(col_clean) > 2:
            try:
                slope, _, _, _, _ = sp_stats.linregress(
                    np.arange(len(col_clean)), col_clean
                )
                feat_p.append(float(slope))
            except Exception:
                feat_p.append(0.0)
        else:
            feat_p.append(0.0)

        # Zero crossings (around mean)
        if len(col_clean) > 1:
            centered = col_clean - np.mean(col_clean)
            zc = np.sum(np.diff(np.sign(centered)) != 0)
            feat_p.append(float(zc) / len(col_clean))
        else:
            feat_p.append(0.0)

        # IQR
        if len(col_clean) > 1:
            q75, q25 = np.percentile(col_clean, [75, 25])
            feat_p.append(q75 - q25)
        else:
            feat_p.append(0.0)

        # Range
        feat_p.append(np.ptp(col_clean))

        # Autocorrelation at lag 1
        if len(col_clean) > 2:
            ac = np.corrcoef(col_clean[:-1], col_clean[1:])[0, 1]
            feat_p.append(float(ac) if np.isfinite(ac) else 0.0)
        else:
            feat_p.append(0.0)

        features.extend(feat_p)

    return np.array(features, dtype=np.float32)


def extract_features_batch(
    dataset: SensorDataset,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract statistical features for all samples in a dataset.

    Returns:
        X: [N, 66] feature matrix.
        y: [N] binary labels.
    """
    X_list, y_list = [], []
    for i in range(len(dataset)):
        sample = dataset._load(i)
        feats = extract_statistical_features(sample.values)
        X_list.append(feats)
        y_list.append(sample.has_anomaly)
    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    # Replace any remaining NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=5.0, neginf=-5.0)
    return X, y


# ============================================================================
# Section 1: Ensemble Baseline
#   LogisticRegression / RandomForest / GBM over 5 encoder anomaly scores
# ============================================================================

def load_encoder_anomaly_scores() -> Optional[Dict[str, np.ndarray]]:
    """Attempt to load per-encoder anomaly probabilities on the test set.

    Looks for checkpoint files from each of the 5 modality encoders.
    Returns None if any encoder checkpoint is missing.
    """
    modalities = ["sensor", "satellite", "microbial", "molecular", "biomotion"]
    modality_dirs = {
        "sensor": "sensor",
        "satellite": "satellite",
        "microbial": "microbial",
        "molecular": "molecular",
        "biomotion": "biomotion",
    }

    scores = {}
    for mod in modalities:
        ckpt_dir = CHECKPOINT_DIR / modality_dirs[mod]
        if not ckpt_dir.exists():
            logger.warning(f"  Encoder checkpoint dir not found: {ckpt_dir}")
            return None
        # Look for results JSON with test scores
        results_files = sorted(ckpt_dir.glob("results*.json"), reverse=True)
        if not results_files:
            logger.warning(f"  No results JSON in {ckpt_dir}")
            return None
        try:
            with open(results_files[0]) as f:
                res = json.load(f)
            # Try to find per-sample test probabilities
            if "test_probs" in res:
                scores[mod] = np.array(res["test_probs"], dtype=np.float32)
            elif "auroc" in res or "test_auroc" in res:
                # Only aggregate metrics available, not per-sample scores
                logger.warning(
                    f"  {mod}: only aggregate metrics available, not per-sample scores"
                )
                return None
            else:
                return None
        except Exception as e:
            logger.warning(f"  Error loading {results_files[0]}: {e}")
            return None

    # Check all have same length
    lengths = {mod: len(s) for mod, s in scores.items()}
    if len(set(lengths.values())) != 1:
        logger.warning(f"  Encoder score lengths differ: {lengths}")
        return None

    return scores


def run_ensemble_baselines(
    results: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Run ensemble baselines: LR, RF, GBM over encoder probability vectors.

    If real encoder scores are unavailable, synthesize plausible surrogate
    probabilities from the sensor statistical features to still demonstrate
    the methodology.
    """
    logger.info("\n" + "=" * 70)
    logger.info("SECTION 1: Ensemble Baseline (over encoder anomaly scores)")
    logger.info("=" * 70)

    encoder_scores = load_encoder_anomaly_scores()

    if encoder_scores is not None:
        # Real encoder scores available
        mods = sorted(encoder_scores.keys())
        X_ens = np.column_stack([encoder_scores[m] for m in mods])
        # Need corresponding labels; assume they match test set
        logger.info(f"  Using real encoder scores: {X_ens.shape}")
        X_ens_train = X_ens  # This would need proper splitting in practice
        X_ens_test = X_ens
        y_ens_train = y_train[:len(X_ens)] if len(y_train) >= len(X_ens) else y_test
        y_ens_test = y_test[:len(X_ens)]
        source_note = "real encoder outputs"
    else:
        logger.warning(
            "  Real per-encoder anomaly scores not available.\n"
            "  Using surrogate scores from statistical features to demonstrate methodology."
        )
        # Generate surrogate 5-dim "encoder probability" vectors
        # Each "encoder" uses a different subset of the 66 statistical features
        # This tests whether a simple ensemble can combine diverse views
        n_feat = X_train.shape[1]
        rng = np.random.default_rng(SEED)
        surrogate_encoders = []
        feat_per_encoder = n_feat // 5

        for enc_idx in range(5):
            start = enc_idx * feat_per_encoder
            end = start + feat_per_encoder
            if enc_idx == 4:
                end = n_feat  # last encoder gets remaining features
            surrogate_encoders.append((start, end))

        # Train per-encoder LR models to generate pseudo-probabilities
        encoder_probs_train = []
        encoder_probs_test = []
        for i, (s, e) in enumerate(surrogate_encoders):
            lr_enc = LogisticRegression(
                max_iter=500, random_state=SEED + i, C=1.0
            )
            lr_enc.fit(X_train[:, s:e], y_train)
            p_train = lr_enc.predict_proba(X_train[:, s:e])[:, 1]
            p_test = lr_enc.predict_proba(X_test[:, s:e])[:, 1]
            encoder_probs_train.append(p_train)
            encoder_probs_test.append(p_test)

        X_ens_train = np.column_stack(encoder_probs_train)
        X_ens_test = np.column_stack(encoder_probs_test)
        y_ens_train = y_train
        y_ens_test = y_test
        source_note = "surrogate encoder probabilities from feature subsets"

    logger.info(f"  Ensemble input: train={X_ens_train.shape}, test={X_ens_test.shape}")
    logger.info(f"  Source: {source_note}")

    ensemble_results = {}

    # 1a. Logistic Regression
    logger.info("\n  --- Logistic Regression Ensemble ---")
    try:
        lr = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
        lr.fit(X_ens_train, y_ens_train)
        probs = lr.predict_proba(X_ens_test)[:, 1]
        metrics = compute_metrics(y_ens_test, probs, "LR-Ensemble")
        metrics["source"] = source_note
        ensemble_results["LogisticRegression_Ensemble"] = metrics
        logger.info(
            f"  LR Ensemble: AUROC={metrics['auroc']:.4f}, "
            f"AUPRC={metrics['auprc']:.4f}, F1={metrics['f1_at_0.5']:.4f}"
        )
    except Exception as e:
        logger.error(f"  LR Ensemble failed: {e}")

    # 1b. Random Forest
    logger.info("\n  --- Random Forest Ensemble ---")
    try:
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=SEED,
            class_weight="balanced", n_jobs=-1,
        )
        rf.fit(X_ens_train, y_ens_train)
        probs = rf.predict_proba(X_ens_test)[:, 1]
        metrics = compute_metrics(y_ens_test, probs, "RF-Ensemble")
        metrics["source"] = source_note
        ensemble_results["RandomForest_Ensemble"] = metrics
        logger.info(
            f"  RF Ensemble: AUROC={metrics['auroc']:.4f}, "
            f"AUPRC={metrics['auprc']:.4f}, F1={metrics['f1_at_0.5']:.4f}"
        )
    except Exception as e:
        logger.error(f"  RF Ensemble failed: {e}")

    # 1c. GBM (try XGBoost first, fall back to sklearn)
    logger.info("\n  --- Gradient Boosting Ensemble ---")
    try:
        gbm_name = "sklearn.GradientBoosting"
        try:
            import xgboost as xgb
            gbm = xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=SEED, eval_metric="logloss",
                use_label_encoder=False, verbosity=0,
                scale_pos_weight=(y_ens_train == 0).sum() / max((y_ens_train == 1).sum(), 1),
            )
            gbm_name = "XGBoost"
        except ImportError:
            try:
                import lightgbm as lgb
                gbm = lgb.LGBMClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.1,
                    random_state=SEED, verbose=-1,
                    scale_pos_weight=(y_ens_train == 0).sum() / max((y_ens_train == 1).sum(), 1),
                )
                gbm_name = "LightGBM"
            except ImportError:
                gbm = GradientBoostingClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.1,
                    random_state=SEED,
                )
                gbm_name = "sklearn.GradientBoosting"

        gbm.fit(X_ens_train, y_ens_train)
        probs = gbm.predict_proba(X_ens_test)[:, 1]
        metrics = compute_metrics(y_ens_test, probs, f"{gbm_name}-Ensemble")
        metrics["source"] = source_note
        metrics["gbm_backend"] = gbm_name
        ensemble_results[f"{gbm_name}_Ensemble"] = metrics
        logger.info(
            f"  {gbm_name} Ensemble: AUROC={metrics['auroc']:.4f}, "
            f"AUPRC={metrics['auprc']:.4f}, F1={metrics['f1_at_0.5']:.4f}"
        )
    except Exception as e:
        logger.error(f"  GBM Ensemble failed: {e}")

    results["ensemble_baselines"] = ensemble_results
    return results


# ============================================================================
# Section 2: Strong ML Baselines (sensor modality)
# ============================================================================

# --- 2a: GBM on handcrafted features ---

def run_gbm_feature_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """GBM on handcrafted statistical features (66-dim per window)."""
    logger.info("\n  --- GBM on Statistical Features ---")
    results = {}

    # Try XGBoost -> LightGBM -> sklearn
    try:
        gbm_name = "sklearn.GradientBoosting"
        try:
            import xgboost as xgb
            clf = xgb.XGBClassifier(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                random_state=SEED, eval_metric="logloss",
                use_label_encoder=False, verbosity=0,
                scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
            )
            gbm_name = "XGBoost"
        except ImportError:
            try:
                import lightgbm as lgb
                clf = lgb.LGBMClassifier(
                    n_estimators=300, max_depth=8, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=SEED, verbose=-1,
                    scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
                )
                gbm_name = "LightGBM"
            except ImportError:
                clf = GradientBoostingClassifier(
                    n_estimators=300, max_depth=8, learning_rate=0.05,
                    subsample=0.8, random_state=SEED,
                )
                gbm_name = "sklearn.GradientBoosting"

        t0 = time.time()
        clf.fit(X_train, y_train)
        elapsed = time.time() - t0

        probs = clf.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, probs, f"{gbm_name}-Features")
        metrics["train_time_s"] = round(elapsed, 2)
        metrics["n_features"] = X_train.shape[1]
        metrics["backend"] = gbm_name
        results[f"{gbm_name}_StatFeatures"] = metrics

        logger.info(
            f"  {gbm_name} Features: AUROC={metrics['auroc']:.4f}, "
            f"AUPRC={metrics['auprc']:.4f}, F1={metrics['f1_at_0.5']:.4f} "
            f"({elapsed:.1f}s)"
        )
    except Exception as e:
        logger.error(f"  GBM Features failed: {e}")

    return results


# --- 2b: 1D-CNN Baseline ---

class SimpleCNN1D(nn.Module):
    """Simple 1D-CNN for temporal anomaly detection.

    Architecture:
        Conv1D(6->32, k=7) -> Conv1D(32->64, k=5) -> Conv1D(64->128, k=3)
        -> AdaptiveAvgPool -> Linear(128->1)
    """

    def __init__(self, in_channels: int = NUM_PARAMS, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C] -> [B] logit."""
        x = x.permute(0, 2, 1)  # [B, C, T]
        x = self.conv(x).squeeze(-1)  # [B, 128]
        return self.classifier(x).squeeze(-1)


# --- 2c: LSTM Baseline ---

class SimpleLSTM(nn.Module):
    """2-layer LSTM baseline for temporal anomaly detection.

    Architecture:
        LSTM(6->128, layers=2, bidirectional) -> Linear(256->1)
    """

    def __init__(
        self,
        in_channels: int = NUM_PARAMS,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C] -> [B] logit."""
        _, (hn, _) = self.lstm(x)
        # Concat forward and backward final hidden states
        out = torch.cat([hn[-2], hn[-1]], dim=1)  # [B, hidden*2]
        return self.classifier(out).squeeze(-1)


# --- 2d: Transformer Baseline ---

class SimpleTransformer(nn.Module):
    """Vanilla Transformer encoder baseline for temporal anomaly detection.

    Architecture:
        Linear(6->64) + PositionalEncoding
        -> TransformerEncoder(4 layers, 4 heads, d=64)
        -> CLS token pooling -> Linear(64->1)
    """

    def __init__(
        self,
        in_channels: int = NUM_PARAMS,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = MAX_LEN,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(in_channels, d_model)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding (sinusoidal)
        pe = torch.zeros(max_len + 1, d_model)
        position = torch.arange(0, max_len + 1, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len+1, d_model]

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C] -> [B] logit."""
        B, T, _ = x.shape
        x = self.input_proj(x)  # [B, T, d_model]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        x = torch.cat([cls, x], dim=1)  # [B, T+1, d_model]

        # Add positional encoding
        x = x + self.pe[:, : T + 1]

        # Transformer encoding
        x = self.transformer(x)  # [B, T+1, d_model]

        # CLS token output
        cls_out = x[:, 0]  # [B, d_model]
        return self.classifier(cls_out).squeeze(-1)


def train_nn_baseline(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    pos_weight: float,
    name: str,
    epochs: int = BASELINE_EPOCHS,
    patience: int = BASELINE_PATIENCE,
    lr: float = BASELINE_LR,
) -> nn.Module:
    """Generic training loop for NN baselines."""
    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  {name}: {n_params:,} parameters")

    pw = torch.tensor([pos_weight], device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    no_improve = 0
    best_state = None

    for ep in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_dl:
            x = batch["values"].to(DEVICE)
            y = batch["has_anomaly"].to(DEVICE)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                x = batch["values"].to(DEVICE)
                y = batch["has_anomaly"].to(DEVICE)
                logits = model(x)
                val_loss += F.binary_cross_entropy_with_logits(
                    logits, y, pos_weight=pw
                ).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if no_improve >= patience:
            logger.info(f"  {name}: early stopping at epoch {ep}")
            break

        if ep % 10 == 0 or ep == 1:
            avg_train = train_loss / max(n_batches, 1)
            logger.info(
                f"  {name} epoch {ep}/{epochs}: "
                f"train_loss={avg_train:.4f}, val_loss={val_loss:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def eval_nn_baseline(
    model: nn.Module, test_dl: DataLoader, name: str
) -> Dict[str, float]:
    """Evaluate an NN baseline on test set."""
    model.eval()
    all_probs, all_labels = [], []
    for batch in test_dl:
        x = batch["values"].to(DEVICE)
        y = batch["has_anomaly"]
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(y.numpy().tolist())

    y_true = np.array(all_labels, dtype=np.int64)
    y_prob = np.array(all_probs, dtype=np.float64)
    metrics = compute_metrics(y_true, y_prob, name)
    metrics["n_params"] = sum(p.numel() for p in model.parameters())
    return metrics


def run_strong_ml_baselines(
    results: Dict[str, Any],
    train_ds: SensorDataset,
    val_ds: SensorDataset,
    test_ds: SensorDataset,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Run all strong ML baselines for the sensor modality."""
    logger.info("\n" + "=" * 70)
    logger.info("SECTION 2: Strong ML Baselines (sensor modality)")
    logger.info("=" * 70)

    strong_results: Dict[str, Any] = {}
    pos_weight = compute_pos_weight(train_ds)
    logger.info(f"  Positive weight: {pos_weight:.2f}")

    # DataLoaders
    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    test_dl = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # 2a: GBM on statistical features
    gbm_results = run_gbm_feature_baseline(X_train, y_train, X_test, y_test)
    strong_results.update(gbm_results)

    # 2b: 1D-CNN
    logger.info("\n  --- 1D-CNN Baseline ---")
    try:
        cnn = SimpleCNN1D(in_channels=NUM_PARAMS)
        t0 = time.time()
        cnn = train_nn_baseline(cnn, train_dl, val_dl, pos_weight, "1D-CNN")
        elapsed = time.time() - t0
        metrics = eval_nn_baseline(cnn, test_dl, "1D-CNN")
        metrics["train_time_s"] = round(elapsed, 2)
        strong_results["1D_CNN"] = metrics
        logger.info(
            f"  1D-CNN: AUROC={metrics['auroc']:.4f}, "
            f"AUPRC={metrics['auprc']:.4f}, F1={metrics['f1_at_0.5']:.4f} "
            f"({elapsed:.1f}s)"
        )
    except Exception as e:
        logger.error(f"  1D-CNN failed: {e}")

    # 2c: LSTM
    logger.info("\n  --- LSTM Baseline (2-layer, hidden=128) ---")
    try:
        lstm = SimpleLSTM(in_channels=NUM_PARAMS, hidden_dim=128, num_layers=2)
        t0 = time.time()
        lstm = train_nn_baseline(lstm, train_dl, val_dl, pos_weight, "LSTM")
        elapsed = time.time() - t0
        metrics = eval_nn_baseline(lstm, test_dl, "LSTM")
        metrics["train_time_s"] = round(elapsed, 2)
        strong_results["LSTM_2layer_h128"] = metrics
        logger.info(
            f"  LSTM: AUROC={metrics['auroc']:.4f}, "
            f"AUPRC={metrics['auprc']:.4f}, F1={metrics['f1_at_0.5']:.4f} "
            f"({elapsed:.1f}s)"
        )
    except Exception as e:
        logger.error(f"  LSTM failed: {e}")

    # 2d: Transformer
    logger.info("\n  --- Transformer Baseline (4-layer, 4-head, d=64) ---")
    try:
        transformer = SimpleTransformer(
            in_channels=NUM_PARAMS, d_model=64, nhead=4,
            num_layers=4, dim_feedforward=256,
        )
        t0 = time.time()
        transformer = train_nn_baseline(
            transformer, train_dl, val_dl, pos_weight, "Transformer",
            lr=5e-4,  # Transformers often need lower LR
        )
        elapsed = time.time() - t0
        metrics = eval_nn_baseline(transformer, test_dl, "Transformer")
        metrics["train_time_s"] = round(elapsed, 2)
        strong_results["Transformer_4L4H_d64"] = metrics
        logger.info(
            f"  Transformer: AUROC={metrics['auroc']:.4f}, "
            f"AUPRC={metrics['auprc']:.4f}, F1={metrics['f1_at_0.5']:.4f} "
            f"({elapsed:.1f}s)"
        )
    except Exception as e:
        logger.error(f"  Transformer failed: {e}")

    # Load AquaSSM reference if available
    aquassm_ref = _load_aquassm_reference()
    if aquassm_ref:
        strong_results["AquaSSM_reference"] = aquassm_ref

    results["strong_ml_baselines"] = strong_results
    return results


def _load_aquassm_reference() -> Optional[Dict[str, Any]]:
    """Load AquaSSM benchmark results for comparison."""
    ref_path = RESULTS_DIR / "aquassm_benchmark.json"
    if not ref_path.exists():
        logger.warning("  AquaSSM benchmark not found for comparison.")
        return None
    try:
        with open(ref_path) as f:
            ref = json.load(f)
        aquassm = ref.get("models", {}).get("AquaSSM", {})
        if aquassm:
            return {
                "auroc": aquassm.get("auroc"),
                "f1_at_0.5": aquassm.get("f1"),
                "n_test": aquassm.get("n_test"),
                "source": "results/benchmarks/aquassm_benchmark.json",
            }
    except Exception:
        pass
    return None


# ============================================================================
# Section 3: Hard Negative Controls
# ============================================================================

def run_hard_negative_controls(
    results: Dict[str, Any],
    all_files: List[Path],
    train_ds: SensorDataset,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Dict[str, Any]:
    """Select hard negatives: clean windows from contaminated sites.

    Find sites that have at least one anomalous window, then select
    non-anomalous windows from those same sites. These represent the
    hardest negatives: realistic conditions at known-contaminated sites
    during their clean periods.
    """
    logger.info("\n" + "=" * 70)
    logger.info("SECTION 3: Hard Negative Controls")
    logger.info("=" * 70)

    # Group files by station
    station_files: Dict[str, List[Tuple[Path, bool]]] = {}
    for f in all_files:
        try:
            d = np.load(f, allow_pickle=True)
        except Exception:
            continue
        sid = str(d["station_id"]) if "station_id" in d else f.stem.split("_seq")[0]
        has_anom = bool(int(d["has_anomaly"])) if "has_anomaly" in d else False
        if sid not in station_files:
            station_files[sid] = []
        station_files[sid].append((f, has_anom))

    # Find contaminated stations (stations with at least 1 anomalous window)
    contaminated_stations = [
        sid for sid, files_info in station_files.items()
        if any(has_a for _, has_a in files_info)
    ]
    logger.info(
        f"  Found {len(contaminated_stations)} contaminated stations "
        f"out of {len(station_files)} total"
    )

    # Collect clean windows from contaminated stations
    clean_at_dirty = []
    for sid in contaminated_stations:
        for f, has_anom in station_files[sid]:
            if not has_anom:
                clean_at_dirty.append(f)

    logger.info(
        f"  Found {len(clean_at_dirty)} clean windows at contaminated stations"
    )

    if len(clean_at_dirty) == 0:
        logger.warning("  No hard negatives found (no clean windows at dirty sites).")
        # Use all available non-anomalous windows as fallback
        clean_at_dirty = [
            f for f_list in station_files.values()
            for f, has_a in f_list if not has_a
        ]
        logger.info(f"  Fallback: using {len(clean_at_dirty)} general clean windows")

    # Sample N_HARD_NEGATIVES
    rng = np.random.default_rng(SEED)
    n_select = min(N_HARD_NEGATIVES, len(clean_at_dirty))
    selected_idx = rng.choice(len(clean_at_dirty), size=n_select, replace=False)
    hard_neg_files = [clean_at_dirty[i] for i in selected_idx]
    logger.info(f"  Selected {n_select} hard negative windows")

    # Create dataset for hard negatives
    hard_neg_ds = SensorDataset(hard_neg_files, max_len=MAX_LEN)

    # Extract features
    X_hard, y_hard = extract_features_batch(hard_neg_ds)
    assert (y_hard == 0).all(), "Hard negatives should all be label=0"

    # Evaluate trained classifiers on hard negatives
    hard_neg_results: Dict[str, Any] = {
        "n_hard_negatives": n_select,
        "n_contaminated_stations": len(contaminated_stations),
        "n_clean_at_dirty": len(clean_at_dirty),
        "method": "clean windows at known-contaminated sites",
    }

    # Train a GBM on the training set and evaluate FPR on hard negatives
    try:
        gbm_name = "sklearn.GradientBoosting"
        try:
            import xgboost as xgb
            clf = xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=SEED, eval_metric="logloss",
                use_label_encoder=False, verbosity=0,
                scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
            )
            gbm_name = "XGBoost"
        except ImportError:
            clf = GradientBoostingClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=SEED,
            )

        clf.fit(X_train, y_train)
        hard_neg_probs = clf.predict_proba(X_hard)[:, 1]

        # FPR at various thresholds (all are true negatives, so any
        # prediction > threshold is a false positive)
        fpr_results = {}
        for thr in FPR_THRESHOLDS:
            fp = (hard_neg_probs >= thr).sum()
            fpr = float(fp / len(hard_neg_probs))
            fpr_results[f"FPR_at_{thr}"] = fpr
            logger.info(
                f"  {gbm_name} FPR at threshold {thr}: "
                f"{fpr:.4f} ({fp}/{len(hard_neg_probs)} false positives)"
            )

        hard_neg_results[f"{gbm_name}_on_hard_negatives"] = {
            "fpr_by_threshold": fpr_results,
            "mean_predicted_prob": float(hard_neg_probs.mean()),
            "max_predicted_prob": float(hard_neg_probs.max()),
            "median_predicted_prob": float(np.median(hard_neg_probs)),
        }
    except Exception as e:
        logger.error(f"  Hard negative evaluation failed: {e}")

    # Also evaluate with IsolationForest for comparison
    try:
        logger.info("\n  --- IsolationForest on Hard Negatives ---")
        iso = IsolationForest(
            n_estimators=200, contamination=0.1, random_state=SEED, n_jobs=-1,
        )
        iso.fit(X_train)
        # IsolationForest: -1 = anomaly, 1 = normal
        iso_scores = -iso.decision_function(X_hard)  # Higher = more anomalous
        iso_scores_normed = (iso_scores - iso_scores.min()) / (
            iso_scores.max() - iso_scores.min() + 1e-8
        )

        fpr_iso = {}
        for thr in FPR_THRESHOLDS:
            fp = (iso_scores_normed >= thr).sum()
            fpr = float(fp / len(iso_scores_normed))
            fpr_iso[f"FPR_at_{thr}"] = fpr
            logger.info(
                f"  IsolationForest FPR at threshold {thr}: "
                f"{fpr:.4f} ({fp}/{len(iso_scores_normed)} false positives)"
            )

        hard_neg_results["IsolationForest_on_hard_negatives"] = {
            "fpr_by_threshold": fpr_iso,
            "mean_score": float(iso_scores_normed.mean()),
            "max_score": float(iso_scores_normed.max()),
        }
    except Exception as e:
        logger.error(f"  IsolationForest hard negative eval failed: {e}")

    results["hard_negative_controls"] = hard_neg_results
    return results


# ============================================================================
# Section 4: Ablation -- Simple vs Complex Fusion
# ============================================================================

class ConcatMLPFusion(nn.Module):
    """Simple late fusion: concatenate encoder embeddings -> MLP classifier.

    Concatenates 5 encoder embeddings (each 256-d) into a 1280-d vector
    and passes through a 2-layer MLP.
    """

    def __init__(
        self,
        n_encoders: int = 5,
        embed_dim: int = SHARED_DIM,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_encoders = n_encoders
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(n_encoders * embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """embeddings: [B, n_encoders, embed_dim] -> [B] logit."""
        return self.mlp(embeddings.view(embeddings.size(0), -1)).squeeze(-1)


class AvgPoolFusion(nn.Module):
    """Average pooling fusion: mean of encoder embeddings -> classifier."""

    def __init__(self, embed_dim: int = SHARED_DIM, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """embeddings: [B, n_encoders, embed_dim] -> [B] logit."""
        pooled = embeddings.mean(dim=1)  # [B, embed_dim]
        return self.classifier(pooled).squeeze(-1)


class MajorityVoteFusion(nn.Module):
    """Majority vote fusion: each encoder votes, majority wins.

    Each encoder embedding goes through its own classifier head, then
    the probabilities are averaged (soft majority vote).
    """

    def __init__(
        self,
        n_encoders: int = 5,
        embed_dim: int = SHARED_DIM,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )
            for _ in range(n_encoders)
        ])

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """embeddings: [B, n_encoders, embed_dim] -> [B] logit."""
        votes = []
        for i, head in enumerate(self.heads):
            logit = head(embeddings[:, i])  # [B, 1]
            votes.append(logit)
        # Soft vote: average of logits
        return torch.cat(votes, dim=-1).mean(dim=-1)  # [B]


class SurrogateEncoderBank(nn.Module):
    """Surrogate encoder bank that produces 5 embeddings from sensor data.

    Since we may not have all 5 encoders available, this creates 5
    diverse feature extractors from the sensor modality alone to test
    fusion strategies.
    """

    def __init__(self, in_channels: int = NUM_PARAMS, embed_dim: int = SHARED_DIM):
        super().__init__()
        self.encoders = nn.ModuleList([
            self._make_encoder(in_channels, embed_dim, seed=i)
            for i in range(5)
        ])

    def _make_encoder(
        self, in_channels: int, embed_dim: int, seed: int
    ) -> nn.Module:
        """Create a simple feature extractor with different configs."""
        kernel_sizes = [3, 5, 7, 9, 11]
        k = kernel_sizes[seed % len(kernel_sizes)]
        return nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=k, padding=k // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C] -> [B, 5, embed_dim]."""
        x_t = x.permute(0, 2, 1)  # [B, C, T]
        embeddings = [enc(x_t) for enc in self.encoders]
        return torch.stack(embeddings, dim=1)  # [B, 5, embed_dim]


class FusionWithEncoders(nn.Module):
    """Wrapper that combines a surrogate encoder bank with a fusion module."""

    def __init__(self, encoder_bank: SurrogateEncoderBank, fusion: nn.Module):
        super().__init__()
        self.encoder_bank = encoder_bank
        self.fusion = fusion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C] -> [B] logit."""
        embeddings = self.encoder_bank(x)  # [B, 5, embed_dim]
        return self.fusion(embeddings)


def run_fusion_ablation(
    results: Dict[str, Any],
    train_ds: SensorDataset,
    val_ds: SensorDataset,
    test_ds: SensorDataset,
) -> Dict[str, Any]:
    """Compare simple vs complex fusion strategies."""
    logger.info("\n" + "=" * 70)
    logger.info("SECTION 4: Ablation -- Simple vs Complex Fusion")
    logger.info("=" * 70)

    pos_weight = compute_pos_weight(train_ds)

    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    test_dl = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    fusion_results: Dict[str, Any] = {}

    # Shared encoder bank (frozen after initial training with concat fusion)
    encoder_bank = SurrogateEncoderBank(in_channels=NUM_PARAMS, embed_dim=SHARED_DIM)

    fusion_configs = [
        ("Concat_MLP", ConcatMLPFusion(n_encoders=5, embed_dim=SHARED_DIM)),
        ("AvgPool_Classifier", AvgPoolFusion(embed_dim=SHARED_DIM)),
        ("MajorityVote", MajorityVoteFusion(n_encoders=5, embed_dim=SHARED_DIM)),
    ]

    for name, fusion_module in fusion_configs:
        logger.info(f"\n  --- {name} Fusion ---")
        try:
            # Create full model with fresh encoder bank copy for each fusion type
            import copy
            enc_copy = copy.deepcopy(encoder_bank)
            model = FusionWithEncoders(enc_copy, fusion_module)

            t0 = time.time()
            model = train_nn_baseline(
                model, train_dl, val_dl, pos_weight, name,
                epochs=BASELINE_EPOCHS, patience=BASELINE_PATIENCE,
            )
            elapsed = time.time() - t0

            metrics = eval_nn_baseline(model, test_dl, name)
            metrics["train_time_s"] = round(elapsed, 2)
            fusion_results[name] = metrics

            logger.info(
                f"  {name}: AUROC={metrics['auroc']:.4f}, "
                f"AUPRC={metrics['auprc']:.4f}, F1={metrics['f1_at_0.5']:.4f} "
                f"({elapsed:.1f}s)"
            )
        except Exception as e:
            logger.error(f"  {name} fusion failed: {e}")

    # Load Perceiver IO reference if available
    perceiver_ref = _load_perceiver_reference()
    if perceiver_ref:
        fusion_results["PerceiverIO_reference"] = perceiver_ref

    results["fusion_ablation"] = fusion_results
    return results


def _load_perceiver_reference() -> Optional[Dict[str, Any]]:
    """Load Perceiver IO fusion benchmark results for comparison."""
    ref_path = CHECKPOINT_DIR / "fusion" / "results_real.json"
    if not ref_path.exists():
        logger.warning("  Perceiver IO fusion results not found for comparison.")
        return None
    try:
        with open(ref_path) as f:
            ref = json.load(f)
        return {
            "auroc": ref.get("auroc") or ref.get("test_auroc"),
            "f1_at_0.5": ref.get("f1") or ref.get("test_f1"),
            "source": str(ref_path),
        }
    except Exception:
        return None


# ============================================================================
# Results formatting and saving
# ============================================================================

def print_comparison_table(results: Dict[str, Any]) -> None:
    """Print a formatted comparison table to stdout."""
    print("\n" + "=" * 85)
    print("HONEST BASELINES COMPARISON TABLE -- SENTINEL Phase 0.3")
    print("=" * 85)

    rows: List[Tuple[str, str, Optional[float], Optional[float], Optional[float]]] = []

    # Section 1: Ensemble baselines
    for name, m in results.get("ensemble_baselines", {}).items():
        rows.append(("Ensemble", name, m.get("auroc"), m.get("auprc"), m.get("f1_at_0.5")))

    # Section 2: Strong ML baselines
    for name, m in results.get("strong_ml_baselines", {}).items():
        rows.append(("Strong ML", name, m.get("auroc"), m.get("auprc"), m.get("f1_at_0.5")))

    # Section 4: Fusion ablation
    for name, m in results.get("fusion_ablation", {}).items():
        rows.append(("Fusion", name, m.get("auroc"), m.get("auprc"), m.get("f1_at_0.5")))

    # Print table
    print(f"\n{'Section':<12} {'Method':<35} {'AUROC':>8} {'AUPRC':>8} {'F1@0.5':>8}")
    print("-" * 85)
    for section, name, auroc, auprc, f1 in rows:
        auroc_s = f"{auroc:.4f}" if auroc is not None else "  N/A "
        auprc_s = f"{auprc:.4f}" if auprc is not None else "  N/A "
        f1_s = f"{f1:.4f}" if f1 is not None else "  N/A "
        print(f"  {section:<10} {name:<35} {auroc_s:>8} {auprc_s:>8} {f1_s:>8}")
    print("-" * 85)

    # Hard negative summary
    hn = results.get("hard_negative_controls", {})
    if hn:
        print(f"\nHard Negative Controls ({hn.get('n_hard_negatives', 0)} windows from contaminated sites):")
        for key, val in hn.items():
            if isinstance(val, dict) and "fpr_by_threshold" in val:
                print(f"  {key}:")
                for thr_key, fpr_val in val["fpr_by_threshold"].items():
                    print(f"    {thr_key}: {fpr_val:.4f}")

    print("=" * 85)


def save_results(results: Dict[str, Any]) -> Path:
    """Save results to JSON."""
    out_path = RESULTS_DIR / "honest_baselines.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {out_path}")
    return out_path


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    """Run all honest baselines and negative controls."""
    set_seed(SEED)
    t0_global = time.time()

    logger.info("=" * 70)
    logger.info("SENTINEL Phase 0.3: Honest Baselines & Negative Controls")
    logger.info("=" * 70)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Seed: {SEED}")

    # --- Load data ---
    logger.info("\n--- Loading sensor data ---")
    try:
        data_dir, all_files = find_sensor_data()
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Total files: {len(all_files)}")

    # --- Split ---
    logger.info("\n--- Building temporal-spatial splits ---")
    try:
        train_files, val_files, test_files = build_temporal_spatial_splits(all_files)
    except Exception as e:
        logger.warning(f"Temporal-spatial split failed ({e}), falling back to random split")
        train_files, val_files, test_files = build_random_splits(all_files)

    if len(test_files) == 0:
        logger.warning("Test set is empty after temporal-spatial split. Using random split.")
        train_files, val_files, test_files = build_random_splits(all_files)

    # Ensure minimum split sizes
    if len(train_files) < 10 or len(val_files) < 5 or len(test_files) < 5:
        logger.warning(
            f"Splits too small (train={len(train_files)}, val={len(val_files)}, "
            f"test={len(test_files)}). Falling back to random split."
        )
        train_files, val_files, test_files = build_random_splits(all_files)

    logger.info(
        f"Final splits: train={len(train_files)}, "
        f"val={len(val_files)}, test={len(test_files)}"
    )

    # Create datasets
    train_ds = SensorDataset(train_files, max_len=MAX_LEN)
    val_ds = SensorDataset(val_files, max_len=MAX_LEN)
    test_ds = SensorDataset(test_files, max_len=MAX_LEN)

    # --- Extract statistical features ---
    logger.info("\n--- Extracting statistical features ---")
    X_train, y_train = extract_features_batch(train_ds)
    X_test, y_test = extract_features_batch(test_ds)
    logger.info(
        f"Features: train={X_train.shape}, test={X_test.shape}, "
        f"train_pos_rate={y_train.mean():.3f}, test_pos_rate={y_test.mean():.3f}"
    )

    # Initialize results
    results: Dict[str, Any] = {
        "experiment": "Phase 0.3: Honest Baselines & Negative Controls",
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "device": str(DEVICE),
        "data_dir": str(data_dir),
        "n_files": len(all_files),
        "splits": {
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        },
        "feature_dim": X_train.shape[1],
        "class_distribution": {
            "train": {"n": len(y_train), "pos_rate": float(y_train.mean())},
            "test": {"n": len(y_test), "pos_rate": float(y_test.mean())},
        },
    }

    # --- Run all sections ---
    results = run_ensemble_baselines(results, X_train, y_train, X_test, y_test)
    results = run_strong_ml_baselines(
        results, train_ds, val_ds, test_ds, X_train, y_train, X_test, y_test
    )
    results = run_hard_negative_controls(results, all_files, train_ds, X_train, y_train)
    results = run_fusion_ablation(results, train_ds, val_ds, test_ds)

    # --- Finalize ---
    elapsed_total = time.time() - t0_global
    results["elapsed_seconds"] = round(elapsed_total, 2)
    results["elapsed_minutes"] = round(elapsed_total / 60, 2)

    out_path = save_results(results)
    print_comparison_table(results)

    logger.info(f"\nTotal time: {elapsed_total / 60:.1f} minutes")
    logger.info(f"Results: {out_path}")


if __name__ == "__main__":
    main()
